/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 *
 * Pybind entrypoint for `flash_attn_npu_3_950` — the Ascend 950 backend
 * for FlashAttention v3.
 *
 */

#include <torch/extension.h>
#include <unordered_map>

#include "mha_fwd.cpp"
#include "mha_bwd.cpp"
#include "fa_metadata_args.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

extern __global__ __aicpu__ uint32_t ComputeFAMetadata(void *args);

#define ACL_CHECK(expr) TORCH_CHECK((expr) == ACL_SUCCESS, #expr " failed")

static at::Tensor GetSchedulerMetadataImpl(FAMetadataArgs args,
                                           const at::Tensor &seqlensK,
                                           const std::optional<at::Tensor> &seqlensQ)
{
    const int64_t bytes = static_cast<int64_t>(
        fa_metadata::MetadataBytesWithKv(args.maskType != 0, args.batch));
    at::Tensor meta = at::empty({bytes},
                                at::device(at::kPrivateUse1).dtype(at::kByte));
    args.metaOutAddr = reinterpret_cast<uint64_t>(meta.data_ptr());

    c10_npu::NPUStream currentStream = c10_npu::getCurrentNPUStream();
    c10_npu::NPUStream aicpuStream = c10_npu::getNPUStreamFromPool();
    aclrtStream curHandle = currentStream.stream(false);
    aclrtStream aicpuHandle = aicpuStream.stream(false);

    struct MetadataEvents {
        aclrtEvent inputReady = nullptr;
        aclrtEvent metadataDone = nullptr;
    };
    static thread_local std::unordered_map<c10::DeviceIndex, MetadataEvents> eventsByDevice;
    MetadataEvents &events = eventsByDevice[currentStream.device_index()];
    if (events.inputReady == nullptr) {
        ACL_CHECK(aclrtCreateEvent(&events.inputReady));
        ACL_CHECK(aclrtCreateEvent(&events.metadataDone));
    }

    FAMetadataArgs metaArgs = args;
    auto metadata_task = [curHandle, aicpuHandle,
                          inputReady = events.inputReady,
                          metadataDone = events.metadataDone, metaArgs, meta, seqlensQ, seqlensK]() mutable -> int {
        ACL_CHECK(aclrtRecordEvent(inputReady, curHandle));
        ACL_CHECK(aclrtStreamWaitEvent(aicpuHandle, inputReady));
        ComputeFAMetadata<<<1, nullptr, aicpuHandle>>>(&metaArgs, sizeof(metaArgs));
        ACL_CHECK(aclrtRecordEvent(metadataDone, aicpuHandle));
        ACL_CHECK(aclrtStreamWaitEvent(curHandle, metadataDone));
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fa_metadata", metadata_task);

    c10_npu::NPUCachingAllocator::recordStream(meta.storage().data_ptr(), aicpuStream);
    c10_npu::NPUCachingAllocator::recordStream(seqlensK.storage().data_ptr(), aicpuStream);
    if (seqlensQ.has_value()) {
        c10_npu::NPUCachingAllocator::recordStream(seqlensQ->storage().data_ptr(), aicpuStream);
    }
    return meta;
}

at::Tensor get_scheduler_metadata(
        int64_t batch_size,
        int64_t max_seqlen_q,
        int64_t num_heads_q,
        int64_t num_heads_kv,
        int64_t headdim,
        int64_t headdim_v,
        std::optional<at::Tensor> seqlens_q,
        at::Tensor seqlens_k,
        bool is_seqlens_q_cumulative,
        bool is_seqlens_k_cumulative,
        std::optional<int64_t> page_size,
        std::optional<int64_t> num_blocks,
        std::optional<int64_t> max_num_blocks_per_seq,
        bool causal,
        double softmax_scale,
        int64_t num_splits,
        int64_t max_seqlen_k,
        int64_t window_size_left,
        int64_t window_size_right)
{
    const c10::OptionalDeviceGuard device_guard(device_of(seqlens_k));
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32,
                "seqlens_k must have dtype int32");
    TORCH_CHECK(seqlens_k.is_contiguous(), "seqlens_k must be contiguous");
    TORCH_CHECK(seqlens_k.device().type() == at::kPrivateUse1,
                "seqlens_k must be an NPU tensor");
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(num_heads_q % num_heads_kv == 0,
                "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(num_splits == 0 || num_splits == 1,
                "950 backend (v3) only supports num_splits=0 or 1");

    const bool has_q = seqlens_q.has_value();
    if (has_q) {
        TORCH_CHECK(seqlens_q->dtype() == torch::kInt32,
                    "seqlens_q must have dtype int32");
        TORCH_CHECK(seqlens_q->is_contiguous(), "seqlens_q must be contiguous");
        TORCH_CHECK(seqlens_q->numel() ==
                    (is_seqlens_q_cumulative ? batch_size + 1 : batch_size),
                    "seqlens_q has invalid number of elements");
    }
    TORCH_CHECK(seqlens_k.numel() ==
                (is_seqlens_k_cumulative ? batch_size + 1 : batch_size),
                "seqlens_k has invalid number of elements");

    const uint32_t blockDim =
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    const uint32_t ps = page_size.has_value()
        ? static_cast<uint32_t>(page_size.value()) : 128;

    FAMetadataArgs args{};
    args.seqlensQAddr = has_q
        ? reinterpret_cast<uint64_t>(seqlens_q->data_ptr()) : 0;
    args.seqlensKAddr = reinterpret_cast<uint64_t>(seqlens_k.data_ptr());
    args.isSeqlensQCumulative = is_seqlens_q_cumulative ? 1U : 0U;
    args.isSeqlensKCumulative = is_seqlens_k_cumulative ? 1U : 0U;
    args.batch = static_cast<uint32_t>(batch_size);
    args.numHeads = static_cast<uint32_t>(num_heads_q);
    args.numHeadsK = static_cast<uint32_t>(num_heads_kv);
    args.embeddingSize = static_cast<uint32_t>(headdim);
    args.embeddingSizeV = static_cast<uint32_t>(headdim_v);
    args.numBlocks = page_size.has_value()
        ? static_cast<uint32_t>(num_blocks.value_or(0)) : 0;
    args.blockSize = ps;
    args.maxNumBlocksPerBatch = page_size.has_value()
        ? static_cast<uint32_t>(max_num_blocks_per_seq.value_or(0)) : 0;
    args.maxQSeqlen = static_cast<uint32_t>(max_seqlen_q);
    FwdMaskDerivation maskDer = DeriveFwdMask(
        causal, window_size_left, window_size_right, max_seqlen_q, max_seqlen_k);
    args.maskType = maskDer.maskType;
    args.windowSizeLeft = static_cast<int32_t>(maskDer.window_left);
    args.windowSizeRight = static_cast<int32_t>(maskDer.window_right);
    args.blockDim = blockDim;
    args.pagedKV = page_size.has_value() ? 1u : 0u;
    args.softmaxScale = static_cast<float>(softmax_scale);
    return GetSchedulerMetadataImpl(args, seqlens_k, seqlens_q);
}

PYBIND11_MODULE(flash_attn_npu_3_950, m)
{
    m.doc() = "FlashAttention v3 — Ascend 950 backend";
    m.def("fwd", &mha_fwd, "Forward pass, with KV-cache (Ascend 950)");
    m.def("get_scheduler_metadata", &get_scheduler_metadata,
          "Precompute scheduler metadata (tiling + mask) on AICPU");
    m.def("bwd", &mha_bwd, "Backward pass (Ascend 950)");
}
