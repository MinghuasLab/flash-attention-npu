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
#include "combine_dispatch.hpp"
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
                                  int64_t num_heads_kv, int64_t headdim, int64_t headdim_v, at::Tensor cache_seqlens,
                                  std::optional<at::Tensor> cu_seqlens_q = std::nullopt, bool causal = false,
                                  int64_t window_left = -1, int64_t window_right = -1, int64_t num_splits = 0,
                                  std::optional<bool> pack_gqa = std::nullopt, int64_t = 0,
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

// ---------------------------------------------------------------------------
// Split-attention combine (flash_attn_combine / fwd_combine).
//
// The FA2/FA3/FA4 forward kernels already merge FlashDecode partials inside
// their own workspace; that path is gated to paged-KV/TND decode shapes and its
// buffers are not caller-visible. This operator is the standalone, public
// counterpart: it consumes caller-owned dense fp16/bf16/fp32 partials
// (num_splits, B, S, H, D) or (num_splits, T, H, D) and mirrors the NV
// flash_attn_3::fwd_combine contract.
// ---------------------------------------------------------------------------
std::vector<at::Tensor> mha_combine(at::Tensor out_partial, at::Tensor lse_partial,
                                    std::optional<at::Tensor> out_ = std::nullopt,
                                    std::optional<at::ScalarType> out_dtype_ = std::nullopt,
                                    std::optional<at::Tensor> cu_seqlens = std::nullopt,
                                    std::optional<at::Tensor> seqused = std::nullopt,
                                    std::optional<at::Tensor> virtual_batch_idx = std::nullopt, bool return_lse = true)
{
    TORCH_CHECK(out_partial.device().type() == at::kPrivateUse1, "out_partial must be on NPU");
    TORCH_CHECK(lse_partial.device().type() == at::kPrivateUse1, "lse_partial must be on NPU");
    const auto partial_dtype = out_partial.scalar_type();
    TORCH_CHECK(partial_dtype == at::kFloat || partial_dtype == at::kHalf || partial_dtype == at::kBFloat16,
                "out_partial must be fp16, bf16, or fp32");
    if (partial_dtype != at::kFloat) {
        out_partial = out_partial.to(at::kFloat);
    }
    TORCH_CHECK(lse_partial.scalar_type() == at::kFloat, "Attention combine function only support fp32 data type");
    TORCH_CHECK(out_partial.dim() == 4 || out_partial.dim() == 5, "out_partial must have 4 or 5 dimensions");
    TORCH_CHECK(out_partial.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const bool is_dense = out_partial.dim() == 5;
    TORCH_CHECK(lse_partial.dim() == out_partial.dim() - 1,
                "lse_partial must have one dimension less than out_partial");

    const int64_t num_splits = out_partial.size(0);
    TORCH_CHECK(num_splits >= 1, "num_splits must be positive");
    TORCH_CHECK(num_splits <= 256, "FlashAttention combine only supports num_splits at most 256");

    int64_t batch_size = 1;
    int64_t seqlen = 0;
    int64_t num_heads = 0;
    int64_t head_dim = 0;
    if (is_dense) {
        batch_size = out_partial.size(1);
        seqlen = out_partial.size(2);
        num_heads = out_partial.size(3);
        head_dim = out_partial.size(4);
        TORCH_CHECK(lse_partial.size(0) == num_splits && lse_partial.size(1) == batch_size &&
                        lse_partial.size(2) == seqlen && lse_partial.size(3) == num_heads,
                    "lse_partial shape must match out_partial, got lse_partial=", lse_partial.sizes(),
                    " out_partial=", out_partial.sizes());
    } else {
        seqlen = out_partial.size(1);
        num_heads = out_partial.size(2);
        head_dim = out_partial.size(3);
        TORCH_CHECK(lse_partial.size(0) == num_splits && lse_partial.size(1) == seqlen &&
                        lse_partial.size(2) == num_heads,
                    "lse_partial shape must match out_partial, got lse_partial=", lse_partial.sizes(),
                    " out_partial=", out_partial.sizes());
    }

    const auto out_type = out_.has_value() ? out_->scalar_type() : out_dtype_.value_or(partial_dtype);
    TORCH_CHECK(out_type == at::kFloat || out_type == at::kHalf || out_type == at::kBFloat16,
                "Output type must be FP32, FP16 or BF16");

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.device().type() == at::kPrivateUse1, "out must be on NPU");
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        if (is_dense) {
            TORCH_CHECK(out.dim() == 4 && out.size(0) == batch_size && out.size(1) == seqlen &&
                            out.size(2) == num_heads && out.size(3) == head_dim,
                        "out shape must be (batch, seqlen, num_heads, head_dim)");
        } else {
            TORCH_CHECK(out.dim() == 3 && out.size(0) == seqlen && out.size(1) == num_heads && out.size(2) == head_dim,
                        "out shape must be (total_q, num_heads, head_dim)");
        }
    } else if (is_dense) {
        out = at::empty({batch_size, seqlen, num_heads, head_dim}, out_partial.options().dtype(out_type));
    } else {
        out = at::empty({seqlen, num_heads, head_dim}, out_partial.options().dtype(out_type));
    }

    // NV returns the combined LSE as (B, S, H) / (T, H) with the seqlen dim
    // contiguous; allocate (B, H, S) / (H, T) and return the transposed view.
    at::Tensor lse_out;
    if (return_lse) {
        if (is_dense) {
            lse_out =
                at::empty({batch_size, num_heads, seqlen}, out_partial.options().dtype(at::kFloat)).transpose(1, 2);
        } else {
            lse_out = at::empty({num_heads, seqlen}, out_partial.options().dtype(at::kFloat)).transpose(0, 1);
        }
    }

    // Accepted for API parity; combine still processes every row independently.
    if (cu_seqlens.has_value()) {
        TORCH_CHECK(cu_seqlens->scalar_type() == at::kInt && cu_seqlens->dim() == 1 && cu_seqlens->numel() >= 2,
                    "cu_seqlens must be a 1-D int32 tensor with at least 2 elements");
    }
    if (seqused.has_value()) {
        TORCH_CHECK(seqused->scalar_type() == at::kInt && seqused->dim() == 1 && seqused->numel() >= 1,
                    "seqused must be a 1-D int32 tensor");
    }
    if (virtual_batch_idx.has_value()) {
        TORCH_CHECK(virtual_batch_idx->device().type() == at::kPrivateUse1 &&
                        virtual_batch_idx->scalar_type() == at::kInt && virtual_batch_idx->dim() == 1 &&
                        virtual_batch_idx->is_contiguous(),
                    "virtual_batch_idx must be a contiguous 1-D int32 NPU tensor");
        if (is_dense) {
            TORCH_CHECK(virtual_batch_idx->numel() == batch_size, "virtual_batch_idx must have batch_size elements");
        }
    }

    if (seqlen == 0 || num_heads == 0 || head_dim == 0) {
        return {out, lse_out};
    }

    const uint32_t ns = static_cast<uint32_t>(num_splits);
    const uint32_t ns_align = (ns + 7U) / 8U * 8U;
    const uint32_t out_bytes = (out_type == at::kFloat) ? 4U : 2U;
    const uint32_t elem_align = (out_bytes == 4U) ? 8U : 16U;
    const uint32_t d_align = (static_cast<uint32_t>(head_dim) + elem_align - 1U) / elem_align * elem_align;
    // Choose tileRows up to 32 while per-tile UB usage stays below 150 KiB.
    const uint64_t per_row =
        static_cast<uint64_t>(ns_align) * 16U + static_cast<uint64_t>(16U + out_bytes + 4U) * d_align;
    uint32_t tile_rows = 8U;
    for (uint32_t candidate : {32U, 24U, 16U, 8U}) {
        if (static_cast<uint64_t>(candidate) * per_row <= 150U * 1024U) {
            tile_rows = candidate;
            break;
        }
    }
    TORCH_CHECK(static_cast<uint64_t>(tile_rows) * per_row <= 150U * 1024U,
                "combine tile exceeds 150 KiB UB budget for head_dim=", head_dim);

    const uint64_t head_batch =
        is_dense ? static_cast<uint64_t>(batch_size) * num_heads : static_cast<uint64_t>(num_heads);
    const uint64_t row_blocks = (static_cast<uint64_t>(seqlen) + tile_rows - 1U) / tile_rows;
    const uint64_t total_tasks = head_batch * row_blocks;
    const uint32_t aiv = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    const uint32_t launch_block_dim =
        static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(aiv, 1U), std::max<uint64_t>(total_tasks, 1U)));

    CombineLaunchArgs args{};
    args.aclStream = c10_npu::getCurrentNPUStream().stream(false);
    args.launchBlockDim = launch_block_dim;
    args.isDense = is_dense ? 1U : 0U;
    args.writeLse = return_lse ? 1U : 0U;
    args.numSplits = ns;
    args.batch = static_cast<uint32_t>(batch_size);
    args.seqlen = static_cast<uint32_t>(seqlen);
    args.numHeads = static_cast<uint32_t>(num_heads);
    args.headDim = static_cast<uint32_t>(head_dim);
    args.tileRows = tile_rows;
    args.outPartial = static_cast<uint8_t*>(out_partial.data_ptr());
    args.lsePartial = static_cast<uint8_t*>(lse_partial.data_ptr());
    args.out = static_cast<uint8_t*>(out.data_ptr());
    args.lseOut = return_lse ? static_cast<uint8_t*>(lse_out.data_ptr()) : nullptr;
    args.virtualBatchIdx =
        virtual_batch_idx.has_value() ? static_cast<uint8_t*>(virtual_batch_idx->data_ptr()) : nullptr;
    if (is_dense) {
        args.oSplit = out_partial.stride(0);
        args.oBatch = out_partial.stride(1);
        args.oSeq = out_partial.stride(2);
        args.oHead = out_partial.stride(3);
        args.lSplit = lse_partial.stride(0);
        args.lBatch = lse_partial.stride(1);
        args.lSeq = lse_partial.stride(2);
        args.lHead = lse_partial.stride(3);
        args.outBatch = out.stride(0);
        args.outSeq = out.stride(1);
        args.outHead = out.stride(2);
        args.lseOutBatch = num_heads * seqlen;
        args.lseOutHead = seqlen;
    } else {
        args.oSplit = out_partial.stride(0);
        args.oBatch = 0;
        args.oSeq = out_partial.stride(1);
        args.oHead = out_partial.stride(2);
        args.lSplit = lse_partial.stride(0);
        args.lBatch = 0;
        args.lSeq = lse_partial.stride(1);
        args.lHead = lse_partial.stride(2);
        args.outBatch = 0;
        args.outSeq = out.stride(0);
        args.outHead = out.stride(1);
        args.lseOutBatch = 0;
        args.lseOutHead = seqlen;
    }

    const bool is_fp16 = (out_type == at::kHalf);
    const bool is_bf16 = (out_type == at::kBFloat16);
    auto launch_combine = [args, is_fp16, is_bf16]() -> int {
        if (is_fp16) {
            launch_combine_fp16(args);
        } else if (is_bf16) {
            launch_combine_bf16(args);
        } else {
            launch_combine_fp32(args);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fa_combine", launch_combine);
    return {out, lse_out};
}

PYBIND11_MODULE(flash_attn_npu_4_950, m)
{
    m.doc() = "FlashAttention v4 — Ascend 950 backend";
    m.def("fwd", &mha_fwd, "Forward pass, with KV-cache (Ascend 950)");
    m.def("bwd", &mha_bwd, "Backward pass (Ascend 950)");
    m.def("get_scheduler_metadata", &get_scheduler_metadata, "Precompute scheduler metadata");
    m.def("fwd_combine", &mha_combine, "Combine split attention partial outputs", py::arg("out_partial"),
          py::arg("lse_partial"), py::arg("out") = py::none(), py::arg("out_dtype") = py::none(),
          py::arg("cu_seqlens") = py::none(), py::arg("seqused") = py::none(),
          py::arg("virtual_batch_idx") = py::none(), py::arg("return_lse") = true);
}
