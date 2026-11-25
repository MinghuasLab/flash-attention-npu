// Copyright (c) 2026 Huawei Technologies Co., Ltd.

#pragma once

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_map>

#include <torch/extension.h>

#include "acl/acl.h"
#include "fag_tiling_data.h"
#include "fag_zero_gradient.hpp"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

extern __global__ __aicpu__ uint32_t PrepareFlashAttnBwd950(void* args);

namespace fag_bwd_host {

inline bool HasShortTndDkvOutput(const FAGTiling950::FAGInfo& info)
{
    return info.deterministic == 0 && info.qSeqlen <= 16 && info.qkHeadDim <= 128 && info.vHeadDim <= 128;
}

inline uint64_t TndWorkspaceCapacity(const FAGTiling950::FAGInfo& info)
{
    using namespace FAGTiling950;
    using Wide = unsigned __int128;
    const auto align = [](Wide bytes) { return (bytes + GM_ALIGNMENT - 1) / GM_ALIGNMENT * GM_ALIGNMENT; };
    const Wide delta = static_cast<Wide>(info.totalQ) * info.qHeadNum * sizeof(float) * 2;
    Wide capacity = 0;
    if (info.deterministic != 0) {
        const uint64_t d = (info.qkHeadDim + 7) / 8 * 8;
        const uint64_t dv = (info.vHeadDim + 7) / 8 * 8;
        const Wide base = MULTI_CORE_SYNC_BYTES +
                          align(static_cast<Wide>(info.totalQ) * info.qHeadNum * d * sizeof(float)) + align(delta);
        const Wide sharedRows = static_cast<Wide>(info.totalKv) * info.kvHeadNum;
        // TND private BN2S2 uses one 128-token buffer per AIC.
        const Wide privateRows = static_cast<Wide>(info.aicNum) * 128;
        const auto kvBytes = [&](Wide rows) {
            return align(rows * d * sizeof(float)) + align(rows * dv * sizeof(float));
        };
        capacity = base + std::max(kvBytes(sharedRows), kvBytes(privateRows));
    } else {
        capacity = static_cast<Wide>(info.totalQ) * info.qHeadNum * info.qkHeadDim * sizeof(float) + delta;
        // If the declared Q bound is short, both the actual-length and
        // empty-batch fallback plans cast dK/dV directly into the output.
        if (!HasShortTndDkvOutput(info)) {
            capacity +=
                static_cast<Wide>(info.totalKv) * info.kvHeadNum * (info.qkHeadDim + info.vHeadDim) * sizeof(float);
        }
    }
    TORCH_CHECK(capacity > 0 && capacity <= static_cast<Wide>(std::numeric_limits<int64_t>::max()),
                "Ascend950 bwd: invalid TND workspace capacity");
    return static_cast<uint64_t>(capacity);
}

inline at::Tensor PrepareTnd(const FAGTiling950::FAGInfo& info, const FagInputStrides& strides, const at::Tensor& cuQ,
                             const at::Tensor& cuKv, const std::optional<at::Tensor>& usedQ,
                             const std::optional<at::Tensor>& usedKv, uint64_t totalK, uint64_t totalV,
                             uint64_t workspaceCapacity, bool exactPackedLength)
{
    using namespace FAGTiling950;
    at::Tensor tiling = at::empty({static_cast<int64_t>(sizeof(FAGTilingData))}, cuQ.options().dtype(at::kByte));
    FagPrepareArgs args{};
    args.info = info;
    args.inputStrides = strides;
    args.tilingAddr = reinterpret_cast<uint64_t>(tiling.data_ptr());
    args.cuQAddr = reinterpret_cast<uint64_t>(cuQ.data_ptr());
    args.cuKvAddr = reinterpret_cast<uint64_t>(cuKv.data_ptr());
    args.usedQAddr = usedQ.has_value() ? reinterpret_cast<uint64_t>(usedQ->data_ptr()) : 0;
    args.usedKvAddr = usedKv.has_value() ? reinterpret_cast<uint64_t>(usedKv->data_ptr()) : 0;
    args.totalK = totalK;
    args.totalV = totalV;
    args.workspaceCapacity = workspaceCapacity;
    args.exactPackedLength = exactPackedLength;

    const auto currentStream = c10_npu::getCurrentNPUStream();
    auto aicpuStream = c10_npu::getNPUStreamFromPool();
    const auto currentHandle = currentStream.stream(false);
    if (aicpuStream.stream(false) == currentHandle) {
        aicpuStream = c10_npu::getNPUStreamFromPool();
    }
    const auto aicpuHandle = aicpuStream.stream(false);
    TORCH_CHECK(aicpuHandle != currentHandle, "Ascend950 bwd: AICPU planner requires a separate stream");

    struct PlanEvents {
        aclrtEvent inputReady = nullptr;
        aclrtEvent planReady = nullptr;
    };
    static thread_local std::unordered_map<aclrtStream, PlanEvents> streamEvents;
    auto& events = streamEvents[currentHandle];
    if (events.inputReady == nullptr) {
        constexpr auto flags = ACL_EVENT_SYNC | ACL_EVENT_CAPTURE_STREAM_PROGRESS;
        TORCH_CHECK(aclrtCreateEventExWithFlag(&events.inputReady, flags) == ACL_SUCCESS,
                    "Ascend950 bwd: failed to create AICPU input event");
        if (aclrtCreateEventExWithFlag(&events.planReady, flags) != ACL_SUCCESS) {
            aclrtDestroyEvent(events.inputReady);
            events.inputReady = nullptr;
            TORCH_CHECK(false, "Ascend950 bwd: failed to create AICPU completion event");
        }
    }

    auto submit = [currentHandle, aicpuHandle, events, args]() mutable -> int {
        TORCH_CHECK(aclrtRecordEvent(events.inputReady, currentHandle) == ACL_SUCCESS &&
                        aclrtStreamWaitEvent(aicpuHandle, events.inputReady) == ACL_SUCCESS,
                    "Ascend950 bwd: failed to order AICPU inputs");
        PrepareFlashAttnBwd950<<<1, nullptr, aicpuHandle>>>(&args, sizeof(args));
        TORCH_CHECK(aclrtRecordEvent(events.planReady, aicpuHandle) == ACL_SUCCESS &&
                        aclrtStreamWaitEvent(currentHandle, events.planReady) == ACL_SUCCESS,
                    "Ascend950 bwd: failed to order AICPU plan consumer");
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("PrepareFlashAttnBwd950", submit);
    const auto record = [&](const at::Tensor& tensor) {
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), aicpuStream);
    };
    record(tiling);
    record(cuQ);
    record(cuKv);
    if (usedQ.has_value()) {
        record(usedQ.value());
    }
    if (usedKv.has_value()) {
        record(usedKv.value());
    }
    return tiling;
}

__global__ __vector__ void ClearFlashAttnBwdKv950(GM_ADDR tiling, GM_ADDR dk, GM_ADDR dv)
{
#ifdef __DAV_VEC__
    using namespace FAGTiling950;
    const auto* plan = reinterpret_cast<const __gm__ FAGTilingData*>(tiling);
    if (plan->planStatus != FAGPlanStatus::READY) {
        return;
    }
    const FAGGradClearRange dkRange{plan->clearDk.begin, plan->clearDk.count};
    const FAGGradClearRange dvRange{plan->clearDv.begin, plan->clearDv.count};
    if (dkRange.count == 0 && dvRange.count == 0) {
        return;
    }
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroBuffer;
    pipe.InitBuffer(zeroBuffer, GRAD_ZERO_TILE_ELEMENTS * sizeof(uint16_t));
    const auto zeros = zeroBuffer.Get<uint16_t>();
    AscendC::Duplicate(zeros, uint16_t{0}, GRAD_ZERO_TILE_ELEMENTS);
    const auto zeroReady = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(zeroReady);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(zeroReady);
    const auto coreId = AscendC::GetBlockIdx();
    const auto coreCount = AscendC::GetBlockNum();
    ZeroGradient(dk, dkRange, zeros, coreId, coreCount);
    ZeroGradient(dv, dvRange, zeros, coreId, coreCount);
    AscendC::PipeBarrier<PIPE_MTE3>();
#endif
}

inline aclrtStream ClearTndKvOutputs(const FAGTiling950::FAGInfo& info, bool exactPackedLength,
                                     const at::Tensor& tiling, at::Tensor& dk, at::Tensor& dv)
{
    bool useDeviceRanges = false;
    if (HasShortTndDkvOutput(info)) {
        dk.zero_();
        dv.zero_();
    } else if (info.deterministic != 0) {
        // FA3 can have packed tails. A left window can also leave columns without a task.
        useDeviceRanges =
            !exactPackedLength || (info.maskType == FAGTiling950::MaskType::BAND && info.window_size_left >= 0);
    } else if (!exactPackedLength) {
        // FA3 can select earlyDkvCast using actual lengths below the declared Q bound.
        useDeviceRanges = info.qkHeadDim <= 128 && info.vHeadDim <= 128;
        if (!useDeviceRanges) {
            const auto firstTail = static_cast<int64_t>(info.totalKv);
            if (dk.size(0) > firstTail) {
                dk.slice(0, firstTail).zero_();
            }
            if (dv.size(0) > firstTail) {
                dv.slice(0, firstTail).zero_();
            }
        }
    }

    const auto stream = c10_npu::getCurrentNPUStream().stream(true);
    if (useDeviceRanges) {
        // The current stream already waits for the device plan. The backward
        // kernel is queued after this clear on the same stream.
        ClearFlashAttnBwdKv950<<<info.aivNum, nullptr, stream>>>(static_cast<uint8_t*>(tiling.data_ptr()),
                                                                 static_cast<uint8_t*>(dk.data_ptr()),
                                                                 static_cast<uint8_t*>(dv.data_ptr()));
    }
    return stream;
}

} // namespace fag_bwd_host
