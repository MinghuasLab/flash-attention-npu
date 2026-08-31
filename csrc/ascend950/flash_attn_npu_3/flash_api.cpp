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
#include "fa_metadata_args.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

extern __global__ __aicpu__ uint32_t ComputeFAMetadata(void *args);

#define ACL_CHECK(expr) TORCH_CHECK((expr) == ACL_SUCCESS, #expr " failed")

static at::Tensor GetSchedulerMetadataImpl(FAMetadataArgs args,
                                           const at::Tensor &seqlensK,
                                           const std::optional<at::Tensor> &cuSeqlensQ,
                                           const std::optional<at::Tensor> &cuSeqlensK)
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

    // AICPU 启动参数必须跨函数保持有效（启动后仍可能异步读取 args），
    // 且 RunOpApiV2 可能在其它线程执行回调，所以赋值放在回调内进行。
    static thread_local FAMetadataArgs metaArgs;
    auto metadata_task = [curHandle, aicpuHandle,
                          inputReady = events.inputReady,
                          metadataDone = events.metadataDone,
                          args]() -> int {
        metaArgs = args;
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
    if (cuSeqlensQ.has_value()) {
        c10_npu::NPUCachingAllocator::recordStream(cuSeqlensQ->storage().data_ptr(), aicpuStream);
    }
    if (cuSeqlensK.has_value()) {
        c10_npu::NPUCachingAllocator::recordStream(cuSeqlensK->storage().data_ptr(), aicpuStream);
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
        at::Tensor cache_seqlens,
        std::optional<at::Tensor> cu_seqlens_q,
        std::optional<at::Tensor> cu_seqlens_k,
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
    const c10::OptionalDeviceGuard device_guard(device_of(cache_seqlens));
    TORCH_CHECK(cache_seqlens.dtype() == torch::kInt32,
                "cache_seqlens must have dtype int32");
    TORCH_CHECK(cache_seqlens.is_contiguous(),
                "cache_seqlens must be contiguous");
    TORCH_CHECK(cache_seqlens.device().type() == at::kPrivateUse1,
                "cache_seqlens must be an NPU tensor");
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(num_heads_q % num_heads_kv == 0,
                "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(num_splits == 0 || num_splits == 1,
                "950 backend (v3) only supports num_splits=0 or 1");

    const bool is_varlen_q = cu_seqlens_q.has_value();
    const bool is_varlen_kv = cu_seqlens_k.has_value();
    if (is_varlen_q) {
        auto cu_q = cu_seqlens_q.value();
        TORCH_CHECK(cu_q.dtype() == torch::kInt32,
                    "cu_seqlens_q must have dtype int32");
        TORCH_CHECK(cu_q.is_contiguous(), "cu_seqlens_q must be contiguous");
        TORCH_CHECK(cu_q.numel() == batch_size + 1,
                    "cu_seqlens_q must have batch_size+1 elements");
    }
    if (is_varlen_kv) {
        auto cu_k = cu_seqlens_k.value();
        TORCH_CHECK(cu_k.dtype() == torch::kInt32,
                    "cu_seqlens_k must have dtype int32");
        TORCH_CHECK(cu_k.is_contiguous(), "cu_seqlens_k must be contiguous");
        TORCH_CHECK(cu_k.numel() == batch_size + 1,
                    "cu_seqlens_k must have batch_size+1 elements");
    } else {
        TORCH_CHECK(cache_seqlens.numel() == batch_size,
                    "cache_seqlens must have batch_size elements");
    }

    const uint32_t blockDim =
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    const uint32_t ps = page_size.has_value()
        ? static_cast<uint32_t>(page_size.value()) : 128;

    FAMetadataArgs args{};
    args.cuSeqlensQAddr = is_varlen_q
        ? reinterpret_cast<uint64_t>(cu_seqlens_q.value().data_ptr()) : 0;
    args.seqlensKAddr = is_varlen_kv
        ? reinterpret_cast<uint64_t>(cu_seqlens_k.value().data_ptr())
        : reinterpret_cast<uint64_t>(cache_seqlens.data_ptr());
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
    args.isVarlen = is_varlen_q ? 1u : 0u;
    args.isVarlenKv = is_varlen_kv ? 1u : 0u;
    args.pagedKV = page_size.has_value() ? 1u : 0u;
    args.softmaxScale = static_cast<float>(softmax_scale);
    return GetSchedulerMetadataImpl(args, cache_seqlens, cu_seqlens_q, cu_seqlens_k);
}

PYBIND11_MODULE(flash_attn_npu_3_950, m)
{
    m.doc() = "FlashAttention v3 — Ascend 950 backend";
    m.def("fwd", &mha_fwd, "Forward pass, with KV-cache (Ascend 950)");
    m.def("get_scheduler_metadata", &get_scheduler_metadata,
          "Precompute scheduler metadata (tiling + mask) on AICPU");
}
