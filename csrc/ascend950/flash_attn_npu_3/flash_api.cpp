// Copyright (c) 2026, Minghua Shen.

// Pybind entrypoint for `flash_attn_npu_3_950` — the Ascend 950 backend
// for FlashAttention v3.

#include <torch/extension.h>
#include <unordered_map>

#include "mha_fwd.cpp"
#include "mha_bwd.cpp"
#include "fa_metadata_args.h"
#include "combine_dispatch.hpp"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

extern __global__ __aicpu__ uint32_t ComputeFAMetadata(void* args);

#define ACL_CHECK(expr) TORCH_CHECK((expr) == ACL_SUCCESS, #expr " failed")

static at::Tensor GetSchedulerMetadataImpl(FAMetadataArgs args, const at::Tensor& cacheSeqlens,
                                           const std::optional<at::Tensor>& seqlensQ)
{
    const int64_t bytes = static_cast<int64_t>(fa_metadata::MetadataBytesWithKv(args.maskType != 0, args.batch));
    at::Tensor meta = at::empty({bytes}, at::device(at::kPrivateUse1).dtype(at::kByte));
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
    MetadataEvents& events = eventsByDevice[currentStream.device_index()];
    if (events.inputReady == nullptr) {
        ACL_CHECK(aclrtCreateEvent(&events.inputReady));
        ACL_CHECK(aclrtCreateEvent(&events.metadataDone));
    }

    FAMetadataArgs metaArgs = args;
    auto metadata_task = [curHandle, aicpuHandle, inputReady = events.inputReady, metadataDone = events.metadataDone,
                          metaArgs]() mutable -> int {
        ACL_CHECK(aclrtRecordEvent(inputReady, curHandle));
        ACL_CHECK(aclrtStreamWaitEvent(aicpuHandle, inputReady));
        ComputeFAMetadata<<<1, nullptr, aicpuHandle>>>(&metaArgs, sizeof(metaArgs));
        ACL_CHECK(aclrtRecordEvent(metadataDone, aicpuHandle));
        ACL_CHECK(aclrtStreamWaitEvent(curHandle, metadataDone));
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fa_metadata", metadata_task);

    c10_npu::NPUCachingAllocator::recordStream(meta.storage().data_ptr(), aicpuStream);
    c10_npu::NPUCachingAllocator::recordStream(cacheSeqlens.storage().data_ptr(), aicpuStream);
    if (seqlensQ.has_value()) {
        c10_npu::NPUCachingAllocator::recordStream(seqlensQ->storage().data_ptr(), aicpuStream);
    }
    return meta;
}

at::Tensor get_scheduler_metadata(int64_t batch_size, int64_t max_seqlen_q, int64_t num_heads_q, int64_t num_heads_kv,
                                  int64_t headdim, int64_t headdim_v, at::Tensor cache_seqlens,
                                  std::optional<at::Tensor> seqlens_q, std::optional<at::Tensor> cu_seqlens_k,
                                  std::optional<int64_t> page_size, bool causal, bool hasSoftcap, int64_t num_splits,
                                  int64_t max_seqlen_k, int64_t window_size_left, int64_t window_size_right,
                                  int64_t quant_mode)
{
    const c10::OptionalDeviceGuard device_guard(device_of(cache_seqlens));
    TORCH_CHECK(cache_seqlens.dtype() == torch::kInt32, "cache_seqlens must have dtype int32");
    TORCH_CHECK(cache_seqlens.is_contiguous(), "cache_seqlens must be contiguous");
    TORCH_CHECK(cache_seqlens.device().type() == at::kPrivateUse1, "cache_seqlens must be an NPU tensor");
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(num_heads_q % num_heads_kv == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(num_splits == 0 || num_splits == 1, "950 backend (v3) only supports num_splits=0 or 1");
    TORCH_CHECK(quant_mode >= 0 && quant_mode <= 2, "unsupported quantization mode");
    TORCH_CHECK(quant_mode != 1 || (headdim == 128 && headdim_v == 128),
                "950 V3 FP8 requires Q/K/V head dimensions of 128");
    TORCH_CHECK(quant_mode != 2 || (headdim == 192 && headdim_v == 192),
                "950 V3 mixed BF16/FP8 KV requires Dqk=Dv=192");

    const bool is_varlen_q = seqlens_q.has_value();
    const bool is_varlen_kv = cu_seqlens_k.has_value();
    if (is_varlen_q) {
        auto seqlens_q_t = seqlens_q.value();
        TORCH_CHECK(seqlens_q_t.dtype() == torch::kInt32, "seqlens_q must have dtype int32");
        TORCH_CHECK(seqlens_q_t.is_contiguous(), "seqlens_q must be contiguous");
        TORCH_CHECK(seqlens_q_t.numel() == batch_size, "seqlens_q must have batch_size elements");
    }
    if (is_varlen_kv) {
        auto cu_k = cu_seqlens_k.value();
        TORCH_CHECK(cu_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");
        TORCH_CHECK(cu_k.is_contiguous(), "cu_seqlens_k must be contiguous");
        TORCH_CHECK(cu_k.numel() == batch_size + 1, "cu_seqlens_k must have batch_size+1 elements");
    } else {
        TORCH_CHECK(cache_seqlens.numel() == batch_size, "cache_seqlens must have batch_size elements");
    }

    const uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    const uint32_t ps = page_size.has_value() ? static_cast<uint32_t>(page_size.value()) : 128;

    FAMetadataArgs args{};
    args.seqlensQAddr = is_varlen_q ? reinterpret_cast<uint64_t>(seqlens_q.value().data_ptr()) : 0;
    args.cacheSeqlensAddr = reinterpret_cast<uint64_t>(cache_seqlens.data_ptr());
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
    args.isVarlen = is_varlen_q ? 1u : 0u;
    args.isVarlenKv = is_varlen_kv ? 1u : 0u;
    args.pagedKV = page_size.has_value() ? 1u : 0u;
    args.quantMode = quant_mode;
    // The schedule only depends on softcap through the paired-prefill gate, so keep
    // the flag: a buffer built for softcap==0 must not select the paired wave for a
    // softcap>0 call.
    args.softcapActive = hasSoftcap ? 1U : 0U;
    return GetSchedulerMetadataImpl(args, cache_seqlens, seqlens_q);
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

// Small host-side readback of the AICPU-written tiling (call outside graph
// capture): reports the launch-shape flags that the host launch must mirror.
int64_t flash_attn_metadata_flags(const at::Tensor& scheduler_metadata)
{
    TORCH_CHECK(scheduler_metadata.dtype() == at::kByte, "scheduler_metadata must be a byte tensor");
    TORCH_CHECK(scheduler_metadata.is_contiguous(), "scheduler_metadata must be contiguous");
    TORCH_CHECK(scheduler_metadata.device().type() == at::kPrivateUse1, "scheduler_metadata must be an NPU tensor");
    TORCH_CHECK(scheduler_metadata.nbytes() >= static_cast<int64_t>(sizeof(FAInferTilingData)),
                "scheduler_metadata is too small to contain tiling");
    const c10::OptionalDeviceGuard device_guard(device_of(scheduler_metadata));
    FAInferTilingData tiling{};
    auto readTiling = [&]() {
        return aclrtMemcpy(&tiling, sizeof(tiling), scheduler_metadata.data_ptr(), sizeof(tiling),
                           ACL_MEMCPY_DEVICE_TO_HOST);
    };
    // The metadata launch already bridges the AICPU stream back to the current
    // stream with events, so waiting on the current stream is enough. A device
    // wide sync here would also block on unrelated work in other streams.
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    aclrtSynchronizeStream(aclStream);
    auto ret = readTiling();
    TORCH_CHECK(ret == ACL_SUCCESS, "flash_attn_metadata_flags: device-to-host copy failed");
    // The tiling always sets qL1BufNum to 1 or 2; anything else means the read
    // raced the AICPU write, in which case fall back to a device sync once.
    if (tiling.qL1BufNum != 1U && tiling.qL1BufNum != 2U) {
        aclrtSynchronizeDevice();
        ret = readTiling();
        TORCH_CHECK(ret == ACL_SUCCESS, "flash_attn_metadata_flags: device-to-host copy failed");
    }
    int64_t flags = 0;
    if (tiling.qL1BufNum == 2U) {
        flags |= 1; // paired Q-head pipeline (kernel selected from this flag)
    }
    if (tiling.flashDecodeFlag != 0U) {
        flags |= 2;
    }
    return flags;
}

// One-time FlashDecode tiling preparation for scheduler_metadata.
//
// FlashDecode's launch shape (active core count) and workspace size depend on
// the per-batch sequence lengths, so they cannot be derived from a static
// capture-safe rule. This helper runs the existing host tiling once (outside
// NPUGraph capture), uploads the resulting tiling into the metadata buffer and
// returns {activeCores, workspaceBytes, flashDecodeFlag} to the caller, which
// caches it on the metadata tensor for the per-call launch.
at::Tensor flash_attn_prepare_fd_metadata(const at::Tensor& scheduler_metadata, const at::Tensor& q,
                                          const at::Tensor& k, const at::Tensor& v,
                                          const std::optional<at::Tensor>& page_table,
                                          const std::optional<at::Tensor>& cu_seqlens_q,
                                          const std::optional<at::Tensor>& seqused_q, const at::Tensor& cache_seqlens,
                                          int64_t max_seqlen_q, std::optional<double> softmax_scale, bool causal,
                                          int64_t window_size_left, int64_t window_size_right, double softcap,
                                          bool return_softmax_lse, int64_t layout, int64_t num_splits)
{
    TORCH_CHECK(scheduler_metadata.dtype() == at::kByte && scheduler_metadata.is_contiguous(),
                "scheduler_metadata must be a contiguous byte tensor");
    TORCH_CHECK(scheduler_metadata.device().type() == at::kPrivateUse1, "scheduler_metadata must be on NPU");
    TORCH_CHECK(scheduler_metadata.nbytes() >= static_cast<int64_t>(sizeof(FAInferTilingData)),
                "scheduler_metadata is too small to contain tiling");
    const c10::OptionalDeviceGuard device_guard(device_of(q));

    const bool paged_KV = page_table.has_value();
    const bool is_varlen_q = cu_seqlens_q.has_value();
    const bool is_bf16 = q.scalar_type() == torch::kBFloat16;
    const bool mixedKv = is_bf16 && k.scalar_type() == at::ScalarType::Float8_e4m3fn;
    const uint32_t quantMode = mixedKv ? 2U : (q.scalar_type() == at::ScalarType::Float8_e4m3fn ? 1U : 0U);
    const FwdLayout layout_enum = static_cast<FwdLayout>(layout);
    TORCH_CHECK((layout_enum == FwdLayout::TND) == is_varlen_q,
                "layout code conflicts with cu_seqlens_q: TND requires it, BSND and BNSD forbid it");

    const auto sizes = q.sizes();
    int batch_size = 0;
    int seqlen_q = 0;
    int num_heads = 0;
    int head_size_q = 0;
    if (layout_enum == FwdLayout::TND) {
        batch_size = static_cast<int>(cu_seqlens_q->size(0)) - 1;
        seqlen_q = static_cast<int>(max_seqlen_q);
        num_heads = static_cast<int>(sizes[1]);
        head_size_q = static_cast<int>(sizes[2]);
    } else {
        batch_size = static_cast<int>(sizes[0]);
        head_size_q = static_cast<int>(sizes[3]);
        if (layout_enum == FwdLayout::BNSD) {
            num_heads = static_cast<int>(sizes[1]);
            seqlen_q = static_cast<int>(sizes[2]);
        } else {
            num_heads = static_cast<int>(sizes[2]);
            seqlen_q = static_cast<int>(sizes[1]);
        }
    }
    const int max_num_blocks_per_seq = !paged_KV ? 0 : static_cast<int>(page_table->size(1));
    const int num_blocks = !paged_KV ? 0 : static_cast<int>(k.size(0));
    const int page_block_size = !paged_KV ? 128 : static_cast<int>(k.size(1));
    const int num_heads_k = layout_enum == FwdLayout::BNSD ? static_cast<int>(k.size(1))
                                                           : static_cast<int>(k.dim() == 3 ? k.size(1) : k.size(2));
    const int head_size_v = static_cast<int>(v.size(-1));

    at::Tensor seqlens_k_cpu = cache_seqlens.to(at::Device(at::kCPU)).to(at::kInt).contiguous();
    TORCH_CHECK(seqlens_k_cpu.numel() == batch_size, "cache_seqlens must contain one length per batch");
    at::Tensor cu_seqlen_q_cpu;
    if (is_varlen_q) {
        cu_seqlen_q_cpu = cu_seqlens_q->to(at::Device(at::kCPU)).to(at::kInt).contiguous();
    }
    at::Tensor seqused_q_cpu;
    if (seqused_q.has_value()) {
        seqused_q_cpu = seqused_q->to(at::Device(at::kCPU)).to(at::kInt).contiguous();
    }

    const int32_t* seqlens_k_ptr = seqlens_k_cpu.data_ptr<int32_t>();
    int32_t max_kv_seqlen = 0;
    int64_t min_q_seqlen = std::numeric_limits<int64_t>::max();
    int64_t max_q_seqlen = 0;
    const int32_t* q_cu_ptr = is_varlen_q ? cu_seqlen_q_cpu.data_ptr<int32_t>() : nullptr;
    const int32_t* q_used_ptr = seqused_q_cpu.defined() ? seqused_q_cpu.data_ptr<int32_t>() : nullptr;
    for (int b = 0; b < batch_size; ++b) {
        max_kv_seqlen = std::max(max_kv_seqlen, seqlens_k_ptr[b]);
        const int64_t q_len = q_used_ptr != nullptr
                                  ? static_cast<int64_t>(q_used_ptr[b])
                                  : (is_varlen_q ? static_cast<int64_t>(q_cu_ptr[b + 1]) - q_cu_ptr[b] : seqlen_q);
        min_q_seqlen = std::min(min_q_seqlen, q_len);
        max_q_seqlen = std::max(max_q_seqlen, q_len);
    }

    const int64_t declared_k =
        paged_KV ? static_cast<int64_t>(max_num_blocks_per_seq) * page_block_size
                 : (layout_enum == FwdLayout::BNSD ? k.size(2) : (k.dim() == 4 ? k.size(1) : k.size(0)));
    const auto maskDer = DeriveFwdMask(causal, window_size_left, window_size_right, seqlen_q, declared_k);
    const float scale = softmax_scale.has_value() ? static_cast<float>(softmax_scale.value())
                                                  : 1.0f / std::sqrt(static_cast<float>(head_size_q));
    const bool fd_shape_supported = !maskDer.is_local && paged_KV &&
                                    (is_varlen_q || (mixedKv && layout_enum == FwdLayout::BSND)) && min_q_seqlen > 0 &&
                                    max_q_seqlen <= 16 && max_kv_seqlen >= 1024;
    const bool flash_decode = num_splits != 1 && fd_shape_supported;

    SeqlenScratch scratch;
    optiling::FAInferContext ctx;
    fill_inference_context(ctx, scratch, q, k, v, is_varlen_q ? &cu_seqlen_q_cpu : nullptr,
                           seqused_q_cpu.defined() ? &seqused_q_cpu : nullptr, &seqlens_k_cpu, paged_KV,
                           page_block_size, num_blocks, max_num_blocks_per_seq, maskDer.is_causal, maskDer.is_local,
                           maskDer.is_local ? maskDer.window_left : 0, maskDer.is_local ? maskDer.window_right : 0,
                           is_varlen_q, is_bf16, quantMode, batch_size, seqlen_q, num_heads, num_heads_k, head_size_q,
                           head_size_v, scale, static_cast<float>(softcap), return_softmax_lse, layout_enum);
    ctx.mixedKvStorage = mixedKv;
    ctx.flashDecodeFlag = flash_decode;
    ctx.numSplits = static_cast<uint32_t>(num_splits);

    FAInferTilingData hostTiling{};
    {
        optiling::FAInferTiling tiler(ctx);
        tiler.SetCoreNum(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic());
        tiler.DoTiling(hostTiling);
    }

    at::Tensor hints = at::zeros({3}, at::device(c10::kCPU).dtype(at::kLong));
    auto* hints_ptr = hints.data_ptr<int64_t>();
    if (hostTiling.flashDecodeFlag != 0U) {
        // The metadata launch already ordered the AICPU tiling write before this
        // stream, so the upload below simply follows it: no device-wide wait.
        auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
        auto ret = aclrtMemcpyAsync(scheduler_metadata.data_ptr(), sizeof(FAInferTilingData), &hostTiling,
                                    sizeof(FAInferTilingData), ACL_MEMCPY_HOST_TO_DEVICE, aclStream);
        TORCH_CHECK(ret == ACL_SUCCESS, "flash_attn_prepare_fd_metadata: host-to-device copy failed");
        aclrtSynchronizeStream(aclStream);
        hints_ptr[0] = static_cast<int64_t>(hostTiling.fdActiveCoreNum);
        hints_ptr[1] = static_cast<int64_t>(hostTiling.workSpaceSize);
        hints_ptr[2] = 1;
    }
    return hints;
}

PYBIND11_MODULE(flash_attn_npu_3_950, m)
{
    m.doc() = "FlashAttention v3 — Ascend 950 backend";
    m.def("fwd", &mha_fwd, "Forward pass, with KV-cache (Ascend 950)");
    m.def("get_scheduler_metadata", &get_scheduler_metadata, "Precompute scheduler metadata (tiling) on AICPU");
    m.def("flash_attn_metadata_flags", &flash_attn_metadata_flags,
          "Host readback of launch-shape flags from scheduler metadata (not capture-safe)");
    m.def("flash_attn_prepare_fd_metadata", &flash_attn_prepare_fd_metadata,
          "One-time FlashDecode host tiling uploaded into scheduler metadata (not capture-safe)",
          py::arg("scheduler_metadata"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("page_table"),
          py::arg("cu_seqlens_q"), py::arg("seqused_q"), py::arg("cache_seqlens"), py::arg("max_seqlen_q"),
          py::arg("softmax_scale") = py::none(), py::arg("causal") = true, py::arg("window_size_left") = -1,
          py::arg("window_size_right") = -1, py::arg("softcap") = 0.0, py::arg("return_softmax_lse") = true,
          py::arg("layout") = 1, py::arg("num_splits") = 0);
    m.def("bwd", &mha_bwd, "Backward pass (Ascend 950)");
    m.def("fwd_combine", &mha_combine, "Combine split attention partial outputs", py::arg("out_partial"),
          py::arg("lse_partial"), py::arg("out") = py::none(), py::arg("out_dtype") = py::none(),
          py::arg("cu_seqlens") = py::none(), py::arg("seqused") = py::none(), py::arg("return_lse") = true);
}
