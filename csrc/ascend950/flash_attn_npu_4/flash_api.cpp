/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 *
 * Pybind entrypoint for `flash_attn_npu_4_950` — the Ascend 950 backend
 * for FlashAttention v4.
 *
 */

#include <torch/extension.h>
#include <cmath>
#include <cstring>
#include <unordered_map>

#include "device_triu_mask.h"
#include "fa_metadata_args.h"
#include "mha_fwd.cpp"
#include "mha_bwd.cpp"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "acl/acl.h"

extern __global__ __aicpu__ uint32_t ComputeFAMetadata(void* args);
#define ACL_CHECK(expr) TORCH_CHECK((expr) == ACL_SUCCESS, #expr " failed")

static at::Tensor GetSchedulerMetadataImpl(FAMetadataArgs args, const at::Tensor& cache_seqlens,
                                           const std::optional<at::Tensor>& cu_seqlens_q,
                                           const std::optional<at::Tensor>& seqused_q)
{
    auto metadata = at::empty({static_cast<int64_t>(fa_metadata::MetadataBytes(false))},
                              at::device(at::kPrivateUse1).dtype(at::kByte));
    args.metaOutAddr = reinterpret_cast<uint64_t>(metadata.data_ptr());
    auto current = c10_npu::getCurrentNPUStream();
    auto aicpu = c10_npu::getNPUStreamFromPool();
    aclrtStream current_handle = current.stream(false);
    aclrtStream aicpu_handle = aicpu.stream(false);
    struct MetadataEvents {
        aclrtEvent input_ready = nullptr;
        aclrtEvent done = nullptr;
    };
    static thread_local std::unordered_map<c10::DeviceIndex, MetadataEvents> events_by_device;
    auto& events = events_by_device[current.device_index()];
    if (events.input_ready == nullptr) {
        ACL_CHECK(aclrtCreateEvent(&events.input_ready));
        ACL_CHECK(aclrtCreateEvent(&events.done));
    }
    auto task = [current_handle, aicpu_handle, input_ready = events.input_ready, done = events.done,
                 args]() mutable -> int {
        ACL_CHECK(aclrtRecordEvent(input_ready, current_handle));
        ACL_CHECK(aclrtStreamWaitEvent(aicpu_handle, input_ready));
        ComputeFAMetadata<<<1, nullptr, aicpu_handle>>>(&args, sizeof(args));
        ACL_CHECK(aclrtRecordEvent(done, aicpu_handle));
        ACL_CHECK(aclrtStreamWaitEvent(current_handle, done));
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fa_metadata", task);
    c10_npu::NPUCachingAllocator::recordStream(metadata.storage().data_ptr(), aicpu);
    c10_npu::NPUCachingAllocator::recordStream(cache_seqlens.storage().data_ptr(), aicpu);
    if (cu_seqlens_q.has_value()) {
        c10_npu::NPUCachingAllocator::recordStream(cu_seqlens_q->storage().data_ptr(), aicpu);
    }
    if (seqused_q.has_value()) {
        c10_npu::NPUCachingAllocator::recordStream(seqused_q->storage().data_ptr(), aicpu);
    }
    return metadata;
}

at::Tensor get_scheduler_metadata(int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_k, int64_t num_heads_q,
                                  int64_t num_heads_kv, int64_t headdim, int64_t headdim_v,
                                  at::Tensor cache_seqlens, std::optional<at::Tensor> cu_seqlens_q = std::nullopt,
                                  bool causal = false,
                                  int64_t window_left = -1, int64_t window_right = -1,
                                  int64_t num_splits = 0, std::optional<bool> pack_gqa = std::nullopt, int64_t = 0,
                                  std::optional<at::Tensor> seqused_q = std::nullopt)
{
    const c10::OptionalDeviceGuard device_guard(device_of(cache_seqlens));
    TORCH_CHECK(cache_seqlens.device().type() == at::kPrivateUse1, "cache_seqlens must be on NPU");
    TORCH_CHECK(cache_seqlens.dtype() == at::kInt && cache_seqlens.dim() == 1 && cache_seqlens.numel() == batch_size,
                "cache_seqlens must be int32 [batch]");
    TORCH_CHECK(batch_size > 0 && num_heads_q > 0 && num_heads_kv > 0 && num_heads_q % num_heads_kv == 0,
                "invalid batch or head dimensions");
    TORCH_CHECK(!pack_gqa.has_value() || !pack_gqa.value(), "pack_gqa is not supported on Ascend 950 FA4");
    const uint32_t block_dim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    TORCH_CHECK(num_splits >= 0 && num_splits <= static_cast<int64_t>(block_dim), "num_splits must be in [0, ",
                block_dim, "]");
    const bool is_varlen_q = cu_seqlens_q.has_value();
    if (is_varlen_q) {
        auto cq = cu_seqlens_q.value();
        TORCH_CHECK(cq.device() == cache_seqlens.device() && cq.dtype() == at::kInt && cq.is_contiguous() &&
                        cq.numel() == batch_size + 1,
                    "invalid cu_seqlens_q");
    }
    if (seqused_q.has_value()) {
        const auto& sq = seqused_q.value();
        TORCH_CHECK(sq.device() == cache_seqlens.device() && sq.dtype() == at::kInt && sq.is_contiguous() &&
                        sq.dim() == 1 && sq.numel() == batch_size,
                    "seqused_q must be a contiguous int32 tensor of shape [batch] on NPU");
    }
    FAMetadataArgs args{};
    args.cuSeqlensQAddr = is_varlen_q ? reinterpret_cast<uint64_t>(cu_seqlens_q->data_ptr()) : 0;
    args.seqlensQAddr = seqused_q.has_value() ? reinterpret_cast<uint64_t>(seqused_q->data_ptr()) : 0;
    args.hasSequsedQ = seqused_q.has_value() ? 1u : 0u;
    args.seqlensKAddr = reinterpret_cast<uint64_t>(cache_seqlens.data_ptr());
    args.batch = batch_size;
    args.numHeads = num_heads_q;
    args.numHeadsK = num_heads_kv;
    args.embeddingSize = headdim;
    args.embeddingSizeV = headdim_v;
    args.maxQSeqlen = max_seqlen_q;
    args.blockDim = block_dim;
    args.isVarlen = is_varlen_q;
    args.numSplits = num_splits;
    if (max_seqlen_k > 0 && window_left >= max_seqlen_k)
        window_left = -1;
    if (max_seqlen_k > 0 && window_right >= max_seqlen_k)
        window_right = -1;
    if (causal)
        window_right = 0;
    const bool normalized_causal = window_left < 0 && window_right == 0;
    const bool local = (window_left >= 0 || window_right >= 0) && !normalized_causal;
    args.maskType = normalized_causal ? 1 : (local ? 2 : 0);
    args.windowSizeLeft = local && window_left < 0 ? max_seqlen_k : window_left;
    args.windowSizeRight = local && window_right < 0 ? max_seqlen_k : window_right;
    return GetSchedulerMetadataImpl(args, cache_seqlens, cu_seqlens_q, seqused_q);
}

PYBIND11_MODULE(flash_attn_npu_4_950, m)
{
    m.doc() = "FlashAttention v4 — Ascend 950 backend";
    m.def("fwd", &mha_fwd, "Forward pass, with KV-cache (Ascend 950)");
    m.def("bwd", &mha_bwd, "Backward pass (Ascend 950)");
    m.def("get_scheduler_metadata", &get_scheduler_metadata, "Precompute scheduler metadata");
}
