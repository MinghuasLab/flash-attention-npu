// Copyright (c) 2026, Minghua Shen.

// Pybind entrypoint for `flash_attn_npu_3_950` — the Ascend 950 backend
// for FlashAttention v3.

#include <torch/extension.h>

#include "../../profiling/fa_aicpu_profile.hpp"
#include "mha_fwd.cpp"
#include "mha_bwd.cpp"
#include "fa_metadata_args.h"
#include "combine_dispatch.hpp"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

extern __global__ __aicpu__ uint32_t ComputeFAMetadata(void* args);

template <typename Writer>
static at::Tensor GetSchedulerMetadataImpl(FAMetadataArgs args, const at::Tensor& kvLengths,
                                           const std::optional<at::Tensor>& qLengths, const char* kv_name,
                                           const char* q_name, Writer&& write)
{
    const int64_t bytes = static_cast<int64_t>(fa_metadata::MetadataBytesWithKv(args.maskType != 0, args.batch));
    at::Tensor meta = at::empty({bytes}, kvLengths.options().dtype(at::kByte));
    args.metaOutAddr = reinterpret_cast<uint64_t>(meta.data_ptr());

    const auto stream = c10_npu::getCurrentNPUStream();
    const auto handle = stream.stream(false);
    // The caller's stream orders both the length producers and the attention launch.
    fa_profile::TensorInfo profile("950/FA3/ComputeFAMetadata",
                                   {{q_name, fa_profile::Tensor(qLengths)}, {kv_name, &kvLengths}}, {&meta}, write);
    auto metadata_task = [handle, args, profile = std::move(profile)]() mutable -> int {
        fa_profile::LaunchAicpuWithProfile(profile,
                                           [&] { ComputeFAMetadata<<<1, nullptr, handle>>>(&args, sizeof(args)); });
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fa_metadata", metadata_task);

    c10_npu::NPUCachingAllocator::recordStream(meta.storage().data_ptr(), stream);
    c10_npu::NPUCachingAllocator::recordStream(kvLengths.storage().data_ptr(), stream);
    if (qLengths.has_value()) {
        c10_npu::NPUCachingAllocator::recordStream(qLengths->storage().data_ptr(), stream);
    }
    return meta;
}

at::Tensor get_scheduler_metadata(int64_t batch_size, int64_t max_seqlen_q, int64_t num_heads_q, int64_t num_heads_kv,
                                  int64_t headdim, int64_t headdim_v, std::optional<at::Tensor> cache_seqlens,
                                  std::optional<at::Tensor> seqlens_q, std::optional<at::Tensor> cu_seqlens_k,
                                  std::optional<int64_t> page_size, bool causal, bool hasSoftcap, int64_t num_splits,
                                  int64_t max_seqlen_k, int64_t window_size_left, int64_t window_size_right,
                                  int64_t quant_mode, std::optional<at::Tensor> cu_seqlens_q)
{
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(num_heads_q % num_heads_kv == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(num_splits == 0 || num_splits == 1, "950 backend (v3) only supports num_splits=0 or 1");
    TORCH_CHECK(quant_mode == 0 || quant_mode == 1, "unsupported quantization mode");
    TORCH_CHECK(quant_mode != 1 || (headdim == 128 && headdim_v == 128),
                "950 V3 FP8 requires Q/K/V head dimensions of 128");

    TORCH_CHECK(cache_seqlens.has_value() || cu_seqlens_k.has_value(), "cache_seqlens or cu_seqlens_k is required");
    const bool kv_cumulative = !cache_seqlens.has_value();
    const auto& kv_lengths = kv_cumulative ? *cu_seqlens_k : *cache_seqlens;
    const c10::OptionalDeviceGuard device_guard(device_of(kv_lengths));
    TORCH_CHECK(kv_lengths.dtype() == torch::kInt32, "KV lengths must have dtype int32");
    TORCH_CHECK(kv_lengths.is_contiguous(), "KV lengths must be contiguous");
    TORCH_CHECK(kv_lengths.device().type() == at::kPrivateUse1, "KV lengths must be an NPU tensor");
    if (cache_seqlens.has_value()) {
        TORCH_CHECK(cache_seqlens->numel() == batch_size, "cache_seqlens must have batch_size elements");
    }
    if (cu_seqlens_k.has_value()) {
        const auto& cu_k = *cu_seqlens_k;
        TORCH_CHECK(cu_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");
        TORCH_CHECK(cu_k.is_contiguous(), "cu_seqlens_k must be contiguous");
        TORCH_CHECK(cu_k.numel() == batch_size + 1, "cu_seqlens_k must have batch_size+1 elements");
    }
    const bool q_cumulative = !seqlens_q.has_value() && cu_seqlens_q.has_value();
    const auto& q_lengths = q_cumulative ? cu_seqlens_q : seqlens_q;
    if (q_lengths.has_value()) {
        TORCH_CHECK(q_lengths->dtype() == torch::kInt32 && q_lengths->is_contiguous() &&
                        q_lengths->device() == kv_lengths.device() &&
                        q_lengths->numel() == batch_size + static_cast<int64_t>(q_cumulative),
                    "query lengths must be contiguous int32 on the same NPU as the KV lengths");
    }

    const uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    const uint32_t ps = page_size.has_value() ? static_cast<uint32_t>(page_size.value()) : 128;

    FAMetadataArgs args{};
    args.seqlensQAddr = q_lengths.has_value() ? reinterpret_cast<uint64_t>(q_lengths->data_ptr()) : 0;
    args.cacheSeqlensAddr = reinterpret_cast<uint64_t>(kv_lengths.data_ptr());
    args.batch = static_cast<uint32_t>(batch_size);
    args.numHeads = static_cast<uint32_t>(num_heads_q);
    args.numHeadsK = static_cast<uint32_t>(num_heads_kv);
    args.embeddingSize = static_cast<uint32_t>(headdim);
    args.embeddingSizeV = static_cast<uint32_t>(headdim_v);
    // Paged geometry comes per launch; derive a bound for the mask only.
    args.blockSize = ps;
    const uint32_t maxNumBlocksPerBatch =
        page_size.has_value() ? ((static_cast<uint32_t>(max_seqlen_k) + ps - 1) / ps) : 0;
    args.maxQSeqlen = static_cast<uint32_t>(max_seqlen_q);
    // Match paged forward's mask selection using cache capacity.
    const int64_t maskKvBound = page_size.has_value() ? static_cast<int64_t>(maxNumBlocksPerBatch) * ps : max_seqlen_k;
    FwdMaskDerivation maskDer = DeriveFwdMask(causal, window_size_left, window_size_right, max_seqlen_q, maskKvBound);
    args.maskType = maskDer.maskType;
    args.windowSizeLeft = maskDer.window_left;
    args.windowSizeRight = maskDer.window_right;
    args.blockDim = blockDim;
    args.qCumulative = q_cumulative;
    args.kvCumulative = kv_cumulative;
    args.pagedKV = page_size.has_value() ? 1u : 0u;
    args.quantMode = quant_mode;
    // Schedule is scale/softcap independent; hasSoftcap is API parity only.
    (void)hasSoftcap;
    const auto write = [&](std::ostream& fields) {
        fields << " batch_size=" << batch_size << " max_seqlen_q=" << max_seqlen_q << " max_seqlen_k=" << max_seqlen_k
               << " num_heads_q=" << num_heads_q << " num_heads_kv=" << num_heads_kv << " headdim=" << headdim
               << " headdim_v=" << headdim_v;
        fa_profile::Window(fields, causal, window_size_left, window_size_right);
        if (page_size) {
            fields << " page_size=" << *page_size;
        }
        fields << " quant_mode=" << args.quantMode;
    };
    return GetSchedulerMetadataImpl(args, kv_lengths, q_lengths, kv_cumulative ? "cu_seqlens_k" : "cache_seqlens",
                                    q_cumulative ? "cu_seqlens_q" : "seqlens_q", write);
}

// ---------------------------------------------------------------------------
// Split-attention combine (flash_attn_combine / fwd_combine).
//
// The FA2/FA3/FA4 forward kernels already merge FlashDecode partials inside
// their own workspace; that path is gated to paged-KV/TND decode shapes and its
// buffers are not caller-visible. This operator is the standalone, public
// counterpart: it consumes caller-owned dense fp32 partials
// (num_splits, B, S, H, D) or (num_splits, T, H, D) and mirrors the NV
// flash_attn_3::fwd_combine contract.
// ---------------------------------------------------------------------------
std::vector<at::Tensor> mha_combine(at::Tensor out_partial, at::Tensor lse_partial,
                                    std::optional<at::Tensor> out_ = std::nullopt,
                                    std::optional<at::ScalarType> out_dtype_ = std::nullopt,
                                    std::optional<at::Tensor> cu_seqlens = std::nullopt,
                                    std::optional<at::Tensor> seqused = std::nullopt, bool return_lse = true)
{
    TORCH_CHECK(out_partial.device().type() == at::kPrivateUse1, "out_partial must be on NPU");
    TORCH_CHECK(lse_partial.device().type() == at::kPrivateUse1, "lse_partial must be on NPU");
    TORCH_CHECK(out_partial.scalar_type() == at::kFloat, "Attention combine function only support fp32 data type");
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

    const auto out_type = out_.has_value() ? out_->scalar_type() : out_dtype_.value_or(out_partial.scalar_type());
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

    // cu_seqlens/seqused are validated only; combine processes every row independently.
    if (cu_seqlens.has_value()) {
        TORCH_CHECK(cu_seqlens->scalar_type() == at::kInt && cu_seqlens->dim() == 1 && cu_seqlens->numel() >= 2,
                    "cu_seqlens must be a 1-D int32 tensor with at least 2 elements");
    }
    if (seqused.has_value()) {
        TORCH_CHECK(seqused->scalar_type() == at::kInt && seqused->dim() == 1 && seqused->numel() >= 1,
                    "seqused must be a 1-D int32 tensor");
    }

    if (seqlen == 0 || num_heads == 0) {
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
    fa_profile::TensorInfo profile({&out_partial, &lse_partial}, {&out, return_lse ? &lse_out : nullptr});
    auto launch_combine = [args, is_fp16, is_bf16, profile = std::move(profile)]() -> int {
        profile(args.aclStream, args.launchBlockDim, [&] {
            if (is_fp16) {
                launch_combine_fp16(args);
            } else if (is_bf16) {
                launch_combine_bf16(args);
            } else {
                launch_combine_fp32(args);
            }
        });
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fa_combine", launch_combine);
    return {out, lse_out};
}

PYBIND11_MODULE(flash_attn_npu_3_950, m)
{
    fa_profile::RegisterProfilingCallback();
    m.doc() = "FlashAttention v3 — Ascend 950 backend";
    m.def("fwd", &mha_fwd, "Forward pass, with KV-cache (Ascend 950)");
    m.def("get_scheduler_metadata", &get_scheduler_metadata, "Precompute scheduler metadata (tiling) on AICPU",
          py::arg("batch_size"), py::arg("max_seqlen_q"), py::arg("num_heads_q"), py::arg("num_heads_kv"),
          py::arg("headdim"), py::arg("headdim_v"), py::arg("cache_seqlens"), py::arg("seqlens_q"),
          py::arg("cu_seqlens_k"), py::arg("page_size"), py::arg("causal"), py::arg("has_softcap"),
          py::arg("num_splits"), py::arg("max_seqlen_k"), py::arg("window_size_left"), py::arg("window_size_right"),
          py::arg("quant_mode"), py::arg("cu_seqlens_q") = py::none());
    m.def("bwd", &mha_bwd, "Backward pass (Ascend 950)");
    m.def("fwd_combine", &mha_combine, "Combine split attention partial outputs", py::arg("out_partial"),
          py::arg("lse_partial"), py::arg("out") = py::none(), py::arg("out_dtype") = py::none(),
          py::arg("cu_seqlens") = py::none(), py::arg("seqused") = py::none(), py::arg("return_lse") = true);
}
