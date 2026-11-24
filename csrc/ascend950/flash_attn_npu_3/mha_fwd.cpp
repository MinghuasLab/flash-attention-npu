// Copyright (c) 2026, Minghua Shen.

// ✅ FP16 / BF16
// ✅ Causal mask
// ✅ SWA / window_size (host normalize + MASK_SWA dispatch; kernel Phase 2/3)
// ✅ Paged KV (page_table)
// ✅ MQA / GQA
// ✅ Varlen Q (cu_seqlens_q + max_seqlen_q)
// ✅ return_softmax_lse
// ✅ num_splits (FlashDecode for paged KV + TND)
// ✅ scheduler_metadata for non-FD execution
// ❌ pack_gqa, leftpad_k

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>
#include <c10/core/Device.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "fwd_dispatch.hpp"
#include "fa_fwd_prepare.h"
#include "tilingdata.h"
#include "fa_metadata_args.h"
#include "tiling.cpp"
#include "device_triu_mask.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "tiling/platform/platform_ascendc.h"

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

extern __global__ __aicpu__ uint32_t PrepareFlashAttnFwd950V3(void* args);

namespace {
void LaunchFwdPlan950V3(fa3_fwd::PrepareArgs args, const at::Tensor& plan, const at::Tensor& cuQ,
                       const at::Tensor& cuKv, const std::optional<at::Tensor>& usedQ,
                       const at::Tensor& usedKv, const at::Tensor& pageTable,
                       const std::optional<at::Tensor>& metadata)
{
    const auto current = c10_npu::getCurrentNPUStream();
    const auto currentHandle = current.stream(false);
    auto aicpu = c10_npu::getNPUStreamFromPool();
    if (aicpu.stream(false) == currentHandle) {
        aicpu = c10_npu::getNPUStreamFromPool();
    }
    const auto aicpuHandle = aicpu.stream(false);
    TORCH_CHECK(aicpuHandle != currentHandle, "950 FA3 forward planning requires a separate AICPU stream");

    struct PlanEvents {
        aclrtEvent ready = nullptr;
        aclrtEvent done = nullptr;
    };
    static thread_local std::unordered_map<aclrtStream, PlanEvents> eventsByStream;
    auto& events = eventsByStream[currentHandle];
    if (events.ready == nullptr) {
        constexpr uint32_t flags = ACL_EVENT_SYNC | ACL_EVENT_CAPTURE_STREAM_PROGRESS;
        TORCH_CHECK(aclrtCreateEventExWithFlag(&events.ready, flags) == ACL_SUCCESS,
                    "failed to create FA3 forward plan input event");
        TORCH_CHECK(aclrtCreateEventExWithFlag(&events.done, flags) == ACL_SUCCESS,
                    "failed to create FA3 forward plan completion event");
    }
    auto task = [currentHandle, aicpuHandle, events, args]() mutable -> int {
        TORCH_CHECK(aclrtRecordEvent(events.ready, currentHandle) == ACL_SUCCESS,
                    "failed to record FA3 forward plan input event");
        TORCH_CHECK(aclrtStreamWaitEvent(aicpuHandle, events.ready) == ACL_SUCCESS,
                    "failed to order FA3 forward plan inputs");
        PrepareFlashAttnFwd950V3<<<1, nullptr, aicpuHandle>>>(&args, sizeof(args));
        TORCH_CHECK(aclrtRecordEvent(events.done, aicpuHandle) == ACL_SUCCESS,
                    "failed to record FA3 forward plan completion event");
        TORCH_CHECK(aclrtStreamWaitEvent(currentHandle, events.done) == ACL_SUCCESS,
                    "failed to order FA3 forward plan consumer");
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fa3_fwd_prepare", task);

    const auto record = [&](const at::Tensor& t) {
        if (t.defined()) {
            c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(), aicpu);
        }
    };
    record(plan);
    record(cuQ);
    record(cuKv);
    record(usedKv);
    record(pageTable);
    if (usedQ) {
        record(*usedQ);
    }
    if (metadata) {
        record(*metadata);
    }
}

at::Tensor MakeStaticFwdPlan950V3(const fa3_fwd::PrepareArgs& args, const FwdMaskDerivation& mask,
                                 const at::Device& device)
{
    auto host = at::empty({static_cast<int64_t>(fa3_fwd::PlanBytes(args.batch))},
                          at::TensorOptions().device(at::kCPU).dtype(at::kByte));
    auto* const data = host.data_ptr<uint8_t>();
    auto* const qLengths = reinterpret_cast<int32_t*>(data + fa3_fwd::QScratchOffset());
    auto* const kvLengths = reinterpret_cast<int32_t*>(data + fa3_fwd::KvScratchOffset(args.batch));
    std::fill_n(qLengths, args.batch + 1U, static_cast<int32_t>(args.maxQ));
    std::fill_n(kvLengths, args.batch + 1U, static_cast<int32_t>(args.kvCapacity));

    optiling::FAInferContext context;
    context.batch = args.batch;
    context.numHeads = args.numHeads;
    context.kvHeads = args.numKvHeads;
    context.embeddingSize = args.headDim;
    context.embeddingSizeV = args.headDimV;
    context.maxQSeqlen = args.maxQ;
    context.maxKvSeqlen = args.kvCapacity;
    context.qSeqlenList = qLengths;
    context.kvSeqlenList = kvLengths;
    context.layout = args.layout;
    context.pagedCacheFlag = false;
    context.dataType = args.quantized ? optiling::DataType::FP8 : optiling::DataType::FP16;
    context.maskType = mask.is_local ? optiling::MaskType::MASK_BAND
                                   : (mask.is_causal ? optiling::MaskType::MASK_SPEC : optiling::MaskType::NO_MASK);
    context.windowSizeLeft = mask.is_local ? mask.window_left : 0;
    context.windowSizeRight = mask.is_local ? mask.window_right : 0;
    context.numSplits = args.numSplits;
    auto& plan = *reinterpret_cast<fa3_fwd::Plan*>(data);
    plan = {};
    optiling::FAInferTiling tiler(context);
    tiler.SetCoreNum(args.blockDim);
    tiler.DoTiling(plan.tiling);
    plan.tiling.maskType = mask.maskType;
    plan.status = fa3_fwd::PLAN_VALID;
    return host.to(device);
}

uint64_t DeviceAddress950V3(const at::Tensor& t)
{
    return t.defined() ? reinterpret_cast<uint64_t>(t.data_ptr()) : 0;
}
} // namespace

std::vector<at::Tensor>
mha_fwd(at::Tensor q, at::Tensor k, at::Tensor v, std::optional<at::Tensor> k_new_, std::optional<at::Tensor> v_new_,
        std::optional<at::Tensor> q_v_, std::optional<at::Tensor> out_, std::optional<at::Tensor> cu_seqlens_q_,
        std::optional<at::Tensor> cu_seqlens_k_, std::optional<at::Tensor> cu_seqlens_k_new_,
        std::optional<at::Tensor> seqused_q_, std::optional<at::Tensor> seqused_k_,
        std::optional<int64_t> max_seqlen_q_, std::optional<int64_t> max_seqlen_k_,
        std::optional<at::Tensor> page_table_, std::optional<at::Tensor> kv_batch_idx_,
        std::optional<at::Tensor> leftpad_k_, std::optional<at::Tensor> rotary_cos_,
        std::optional<at::Tensor> rotary_sin_, std::optional<at::Tensor> seqlens_rotary_,
        std::optional<at::Tensor> q_descale_, std::optional<at::Tensor> k_descale_,
        std::optional<at::Tensor> v_descale_, std::optional<float> softmax_scale_, bool is_causal,
        int64_t window_size_left, int64_t window_size_right, int64_t attention_chunk, float softcap,
        bool is_rotary_interleaved, std::optional<at::Tensor> scheduler_metadata_, int64_t num_splits,
        std::optional<bool> pack_gqa_, int64_t sm_margin, bool return_softmax_lse, int64_t layout)
{
    // ============================================================
    // 0. Device guard + stream + AIC core count
    // ============================================================
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // ============================================================
    // 1. dtype + stride sanity
    // ============================================================
    auto q_dtype = q.dtype();
    const bool is_bf16 = (q_dtype == torch::kBFloat16);
    const bool is_fp16 = (q_dtype == torch::kFloat16);
    const uint32_t quantMode = q_dtype == at::ScalarType::Float8_e4m3fn ? 1U : 0U;
    TORCH_CHECK(is_bf16 || is_fp16 || quantMode == 1, "FlashAttention only supports FP16, BF16 and FP8_E4M3FN inputs");
    TORCH_CHECK(q.device().type() == at::kPrivateUse1, "query must be on NPU");
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device(),
                "query, key and value must be on the same NPU device");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    CHECK_CONTIGUOUS(q);
    if (!page_table_.has_value() || quantMode == 1) {
        CHECK_CONTIGUOUS(k);
        CHECK_CONTIGUOUS(v);
    }

    // ============================================================
    // 2. reject list
    // ============================================================
    TORCH_CHECK(!leftpad_k_.has_value(), "950 backend (v3) does not support leftpad_k");
    TORCH_CHECK(!rotary_cos_.has_value() && !rotary_sin_.has_value() && !seqlens_rotary_.has_value(),
                "950 backend (v3) does not support rotary embedding");
    TORCH_CHECK(quantMode == 1 || (!q_descale_ && !k_descale_ && !v_descale_),
                "descales are only supported for FP8_E4M3FN inputs");
    TORCH_CHECK(attention_chunk == 0, "950 backend (v3) does not support attention_chunk");
    TORCH_CHECK(num_splits >= 0 && num_splits <= static_cast<int64_t>(blockDim),
                "950 backend (v3) requires num_splits in [0, ", blockDim, "]");
    if (scheduler_metadata_.has_value()) {
        TORCH_CHECK(num_splits <= 1, "950 backend (v3) scheduler_metadata does not support "
                                     "explicit FlashDecode splits");
    }
    TORCH_CHECK(!pack_gqa_.has_value() || !pack_gqa_.value(), "950 backend (v3) does not support pack_gqa");

    // ============================================================
    // 3. paged / varlen mode + per-tensor + softcap checks
    // ============================================================
    const bool paged_KV = page_table_.has_value();
    const bool is_varlen_q = cu_seqlens_q_.has_value();
    const bool is_varlen_kv = cu_seqlens_k_.has_value();
    const bool is_softcap = softcap > 0.0f;

    TORCH_CHECK(!k_new_.has_value() && !v_new_.has_value() && !q_v_.has_value() && !cu_seqlens_k_new_.has_value() &&
                    !kv_batch_idx_.has_value(),
                "950 backend (v3) does not support KV-cache update inputs");
    if (num_splits > 1) {
        TORCH_CHECK(paged_KV && is_varlen_q, "950 backend (v3) num_splits>1 requires paged KV cache and "
                                             "TND varlen query");
    }

    at::Tensor cu_seqlens_q, cu_seqlens_k, page_table, seqlens_k;

    if (paged_KV) {
        page_table = page_table_.value();
        TORCH_CHECK(page_table.dim() == 2, "page_table must be rank 2");
        TORCH_CHECK(page_table.device() == q.device(), "page_table must be on the same NPU device as query");
        TORCH_CHECK(page_table.dtype() == torch::kInt32, "page_table must have dtype int32");
        CHECK_CONTIGUOUS(page_table);
    }
    if (is_varlen_q) {
        cu_seqlens_q = cu_seqlens_q_.value();
        CHECK_CONTIGUOUS(cu_seqlens_q);
        TORCH_CHECK(cu_seqlens_q.device() == q.device(), "cu_seqlens_q must be on NPU");
        TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
        TORCH_CHECK(max_seqlen_q_.has_value(), "max_seqlen_q must be provided if cu_seqlens_q is provided");
    }
    if (is_varlen_kv) {
        TORCH_CHECK(is_varlen_q, "cu_seqlens_k requires a packed query");
        cu_seqlens_k = cu_seqlens_k_.value();
        CHECK_CONTIGUOUS(cu_seqlens_k);
        TORCH_CHECK(cu_seqlens_k.device() == q.device(), "cu_seqlens_k must be on NPU");
        TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");
        TORCH_CHECK(!paged_KV, "If cu_seqlens_k is passed in, paged table is not supported");
    }
    if (seqused_k_) {
        seqlens_k = *seqused_k_;
        CHECK_CONTIGUOUS(seqlens_k);
        TORCH_CHECK(seqlens_k.device() == q.device(), "seqused_k must be on the query device");
        TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqlens_k.dim() == 1, "seqused_k must be rank 1");
    }
    TORCH_CHECK(!paged_KV || seqlens_k.defined(), "paged KV requires seqused_k");
    TORCH_CHECK(!is_varlen_q || is_varlen_kv || paged_KV || seqlens_k.defined(),
                "packed KV requires cu_seqlens_k or seqused_k");
    TORCH_CHECK(softcap >= 0.0f, "softcap must be non-negative (0.0 disables softcap)");
    // seqused_q caps the *used* Q lengths; the packed region layout stays on cu.
    if (seqused_q_.has_value()) {
        TORCH_CHECK(is_varlen_q, "seqused_q requires cu_seqlens_q");
        const at::Tensor& used_q = seqused_q_.value();
        TORCH_CHECK(used_q.dtype() == torch::kInt32, "seqused_q must have dtype int32");
        TORCH_CHECK(used_q.dim() == 1, "seqused_q must be rank 1");
        CHECK_CONTIGUOUS(used_q);
        TORCH_CHECK(used_q.device() == q.device(), "seqused_q must be on NPU");
    }

    // ============================================================
    // 4. Shape extraction and output tensor
    // ============================================================
    const int64_t expected_q_dim = is_varlen_q ? 3 : 4;
    const int64_t expected_kv_dim = (is_varlen_q && !paged_KV) ? 3 : 4;
    TORCH_CHECK(q.dim() == expected_q_dim, "query must be rank ", expected_q_dim,
                is_varlen_q ? " in TND layout" : " in BSND layout");
    TORCH_CHECK(k.dim() == expected_kv_dim && v.dim() == expected_kv_dim, "key and value must both be rank ",
                expected_kv_dim);
    TORCH_CHECK(k.dim() == v.dim(), "key and value must have the same rank");
    if (is_varlen_q) {
        TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.numel() >= 2,
                    "cu_seqlens_q must be a rank-1 tensor with at least two elements");
    }
    if (is_varlen_kv) {
        TORCH_CHECK(cu_seqlens_k.dim() == 1 && cu_seqlens_k.numel() == cu_seqlens_q.numel(),
                    "cu_seqlens_k must be a rank-1 tensor of length batch_size + 1");
    }
    const auto sizes = q.sizes();
    int batch_size, seqlen_q, num_heads, head_size_q;
    // 4b. Layout (FwdLayout, resolved by the Python caller; Q/K/V share one layout)
    TORCH_CHECK(layout >= static_cast<int64_t>(FwdLayout::TND) && layout <= static_cast<int64_t>(FwdLayout::BNSD),
                "950 backend (v3) received an unsupported layout code (", layout, ")");
    const FwdLayout layout_enum = static_cast<FwdLayout>(layout);
    TORCH_CHECK((layout_enum == FwdLayout::TND) == is_varlen_q,
                "950 backend (v3) layout code conflicts with cu_seqlens_q: TND requires it, "
                "BSND and BNSD forbid it");

    if (layout_enum == FwdLayout::TND) {
        batch_size = static_cast<int>(cu_seqlens_q.size(0)) - 1;
        TORCH_CHECK(*max_seqlen_q_ >= 0 && *max_seqlen_q_ <= std::numeric_limits<int32_t>::max(),
                    "max_seqlen_q must fit in int32");
        seqlen_q = static_cast<int>(max_seqlen_q_.value());
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

    int64_t kv_seqlen_allocated = 0;
    if (layout_enum == FwdLayout::BNSD) {
        // BNSD requires Q, K and V to share the same rank-4 (B, H, S, D) layout.
        TORCH_CHECK(!paged_KV && !is_varlen_kv && k.dim() == 4 && v.dim() == 4,
                    "950 backend (v3) BNSD layout requires rank-4, non-paged key/value tensors with "
                    "the same (B, H, S, D) layout as the query");
        kv_seqlen_allocated = static_cast<int64_t>(k.size(2));
        TORCH_CHECK(k.size(0) == v.size(0) && k.size(1) == v.size(1) && k.size(2) == v.size(2),
                    "950 backend (v3) BNSD key and value must share batch, head and sequence dimensions "
                    "(the head dimension may differ)");
    } else if (layout_enum == FwdLayout::BSND && !paged_KV && k.dim() == 4) {
        // Dense BSND cache is [B, S_alloc, H, D]. S_alloc is the physical batch
        // stride; seqused_k / cache_seqlens is only the used prefix.
        TORCH_CHECK(v.dim() == 4 && k.size(0) == v.size(0) && k.size(1) == v.size(1) && k.size(2) == v.size(2),
                    "950 backend (v3) dense BSND key and value must share batch, sequence and head dimensions");
        kv_seqlen_allocated = static_cast<int64_t>(k.size(1));
    }

    const int max_num_blocks_per_seq = !paged_KV ? 0 : static_cast<int>(page_table.size(1));
    const int num_blocks = !paged_KV ? 0 : static_cast<int>(k.size(0));
    const int page_block_size = !paged_KV ? 128 : static_cast<int>(k.size(1));
    // k is 3D TND for flash_attn_varlen_func, but 4D
    // (batch/num_blocks, seqlen/block, kv_heads, head) for flash_attn_with_kvcache.
    const int num_heads_k = layout_enum == FwdLayout::BNSD ? static_cast<int>(k.size(1))
                                                           : static_cast<int>(k.dim() == 3 ? k.size(1) : k.size(2));
    const int head_size_v = static_cast<int>(v.size(-1));

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    const bool empty_query = is_varlen_q && q.size(0) == 0 && seqlen_q == 0;
    TORCH_CHECK((seqlen_q > 0 || empty_query) && num_heads > 0,
                "query sequence length and head count must be positive unless TND total_q=max_seqlen_q=0");
    TORCH_CHECK(!is_varlen_q || q.size(0) > 0 || empty_query, "total_q must be positive when max_seqlen_q is positive");
    TORCH_CHECK(!seqlens_k.defined() || seqlens_k.numel() == batch_size,
                "seqused_k must contain one KV length per batch");
    TORCH_CHECK(!seqused_q_ || seqused_q_->numel() == batch_size,
                "seqused_q must contain one Q length per batch");
    TORCH_CHECK(k.size(-1) == head_size_q, "query and key must have the same head dimension");
    // D512 opens only the exact Dq=Dk=Dv=512 profile; 257..511 and asymmetric
    // combinations remain rejected by the [1, 256] check.
    const bool d512_heads = (head_size_q == 512 && head_size_v == 512);
    TORCH_CHECK((head_size_q >= 1 && head_size_q <= 256) || d512_heads,
                "FlashAttention supports q/k head dimensions in [1, 256] plus the exact D512 profile (Dq=Dk=Dv=512)");
    TORCH_CHECK((head_size_v >= 1 && head_size_v <= 256) || d512_heads,
                "FlashAttention supports value head dimensions in [1, 256] plus the exact D512 profile (Dq=Dk=Dv=512)");
    TORCH_CHECK(
        !(page_block_size != 128 && page_block_size != 256 && page_block_size != 512 && page_block_size != 1024),
        "FlashAttention only supports page_block_size dimension 128 or 256 or 512 or 1024");
    TORCH_CHECK(v.size(layout_enum == FwdLayout::BNSD || v.dim() == 3 ? 1 : 2) == num_heads_k,
                "key and value must have the same number of heads");
    TORCH_CHECK(num_heads_k > 0, "key/value head count must be positive");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    if (!is_varlen_q && !paged_KV) {
        TORCH_CHECK(q.size(0) == k.size(0), "query and key/value batch sizes must match in BSND layout");
    }
    if (paged_KV) {
        TORCH_CHECK(k.size(0) == v.size(0) && k.size(1) == v.size(1) && k.size(2) == v.size(2),
                    "paged key/value caches must have matching page, page-size, and head dimensions");
        const auto isBBND = [](const at::Tensor& t) {
            return t.stride(1) == t.size(2) * t.size(3) && t.stride(2) == t.size(3) && t.stride(3) == 1;
        };
        const auto isBNBD = [](const at::Tensor& t) {
            return t.stride(1) == t.size(3) && t.stride(2) == t.size(1) * t.size(3) && t.stride(3) == 1;
        };
        TORCH_CHECK((isBBND(k) && isBBND(v)) || (isBNBD(k) && isBNBD(v)),
                    "paged key/value caches must use the same BBND or BNBD inner layout; "
                    "only dimension 0 may otherwise be non-contiguous");
        TORCH_CHECK(k.size(0) <= 1 || k.stride(0) >= k.size(1) * k.size(2) * k.size(3),
                    "paged key cache first-axis stride must not overlap physical pages");
        TORCH_CHECK(v.size(0) <= 1 || v.stride(0) >= v.size(1) * v.size(2) * v.size(3),
                    "paged value cache first-axis stride must not overlap physical pages");
        TORCH_CHECK(num_blocks > 0 && max_num_blocks_per_seq > 0,
                    "paged KV cache and page_table must contain at least one block");
        TORCH_CHECK(page_table.size(0) == batch_size, "page_table batch dimension must match query batch size");
    }

    TORCH_CHECK(quantMode == 0 || (head_size_q == 128 && head_size_v == 128),
                "950 V3 FP8 requires Q/K/V head dimensions of 128");
    GM_ADDR qDescale = nullptr;
    GM_ADDR kDescale = nullptr;
    GM_ADDR vDescale = nullptr;
    auto set_descale = [&](const std::optional<at::Tensor>& scale, GM_ADDR& ptr) {
        if (!scale)
            return;
        TORCH_CHECK(scale->device() == q.device() && scale->scalar_type() == at::kFloat,
                    "descale must be FP32 on the query device");
        TORCH_CHECK(scale->dim() == 2 && scale->size(0) == batch_size && scale->size(1) == num_heads_k,
                    "descale must have shape [batch, num_kv_heads]");
        TORCH_CHECK(scale->is_contiguous(), "descale must be contiguous");
        ptr = static_cast<uint8_t*>(scale->data_ptr());
    };
    set_descale(q_descale_, qDescale);
    set_descale(k_descale_, kDescale);
    set_descale(v_descale_, vDescale);
    const auto out_dtype = quantMode == 1 ? at::kBFloat16 : q.scalar_type();

    std::vector<int64_t> output_sizes(q.sizes().begin(), q.sizes().end());
    output_sizes.back() = head_size_v;
    if (layout_enum == FwdLayout::BNSD) {
        // The BNSD zero-copy path reads Q/K/V as (B, H, S, D) but the kernel
        // writes O with BSND addressing, so the returned tensor is a contiguous
        // (B, S, H, D) block equal to the caller's BSND query shape.
        output_sizes = {batch_size, seqlen_q, num_heads, head_size_v};
    }
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == out_dtype, "output dtype must be BF16 for FP8, otherwise match inputs");
        TORCH_CHECK(out.device() == q.device(), "output must be on the same device as query");
        TORCH_CHECK(out.sizes().vec() == output_sizes,
                    "output shape must match query except for the last dimension, which must match value");
        CHECK_CONTIGUOUS(out);
    } else {
        out = at::empty(output_sizes, q.options().dtype(out_dtype));
    }

    const int64_t kvCapacity = paged_KV ? static_cast<int64_t>(max_num_blocks_per_seq) * page_block_size
                                       : (is_varlen_q ? std::min(k.size(0), v.size(0)) : kv_seqlen_allocated);
    TORCH_CHECK(kvCapacity <= std::numeric_limits<int32_t>::max() &&
                    (!is_varlen_q || q.size(0) <= std::numeric_limits<int32_t>::max()),
                "sequence capacity must fit in int32");
    if (max_seqlen_k_) {
        TORCH_CHECK(*max_seqlen_k_ >= 0 && *max_seqlen_k_ <= std::numeric_limits<int32_t>::max(),
                    "max_seqlen_k must fit in int32");
    }
    if (scheduler_metadata_) {
        const auto& meta = *scheduler_metadata_;
        TORCH_CHECK(meta.dtype() == at::kByte, "scheduler_metadata must be a byte tensor");
        TORCH_CHECK(meta.is_contiguous(), "scheduler_metadata must be contiguous");
        TORCH_CHECK(meta.device() == q.device(), "scheduler_metadata must be on the query device");
        TORCH_CHECK(static_cast<uint64_t>(meta.nbytes()) == fa_metadata::MetadataBytesWithKv(false, batch_size),
                    "scheduler_metadata buffer size must exactly match the batch and backend");
    }

    const bool allowFd = !empty_query && !scheduler_metadata_ && paged_KV && is_varlen_q && num_splits != 1 && kvCapacity >= 1024;
    const int64_t maxKv = max_seqlen_k_.value_or(!is_varlen_q && !paged_KV && !seqlens_k.defined() ? kvCapacity : -1);
    const bool runtimeMask = maxKv < 0 && (window_size_left >= 0 || (!is_causal && window_size_right >= 0));
    const auto staticMask = DeriveFwdMask(is_causal, window_size_left, window_size_right, seqlen_q, maxKv);
    const bool enableDN = head_size_q <= 256 && head_size_v <= 256;
    const bool dynamicDispatch = runtimeMask || (allowFd && enableDN && staticMask.maskType == 0);
    const uint64_t workspaceBytes = allowFd ? fa3_fwd::FdWorkspaceBound(blockDim, head_size_v) : 0;
    const auto byteOptions = q.options().dtype(at::kByte);
    const auto workspace = workspaceBytes != 0
                               ? at::empty({static_cast<int64_t>(workspaceBytes)}, byteOptions)
                               : at::Tensor();
    const auto mask = !empty_query && (runtimeMask || staticMask.maskType != 0) &&
                              quantMode == 0 && !d512_heads
                          ? MakeDeviceTriuMask()
                          : at::Tensor();
    fa3_fwd::PrepareArgs planArgs{
        0,
        DeviceAddress950V3(cu_seqlens_q),
        DeviceAddress950V3(cu_seqlens_k),
        seqused_q_ ? DeviceAddress950V3(*seqused_q_) : 0,
        DeviceAddress950V3(seqlens_k),
        DeviceAddress950V3(page_table),
        scheduler_metadata_ ? DeviceAddress950V3(*scheduler_metadata_) : 0,
        is_varlen_q ? q.size(0) : seqlen_q,
        kvCapacity,
        maxKv,
        window_size_left,
        window_size_right,
        workspaceBytes,
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(seqlen_q),
        static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(num_heads_k),
        static_cast<uint32_t>(head_size_q),
        static_cast<uint32_t>(head_size_v),
        blockDim,
        static_cast<uint32_t>(page_block_size),
        static_cast<uint32_t>(num_blocks),
        static_cast<uint32_t>(max_num_blocks_per_seq),
        static_cast<uint32_t>(num_splits),
        layout_enum,
        is_causal,
        allowFd,
        quantMode != 0
    };
    at::Tensor plan;
    const bool staticDense = !is_varlen_q && !paged_KV && !seqlens_k.defined() && !scheduler_metadata_;
    if (staticDense) {
        TORCH_CHECK(maxKv >= kvCapacity, "max_seqlen_k must cover the dense KV sequence length");
    }
    if (staticDense && c10_npu::currentStreamCaptureStatusMayInitCtx() == c10_npu::CaptureStatus::None) {
        plan = MakeStaticFwdPlan950V3(planArgs, staticMask, q.device());
    } else {
        // Replay regenerates the plan without capturing a temporary Host buffer.
        plan = at::empty({static_cast<int64_t>(fa3_fwd::PlanBytes(batch_size))}, byteOptions);
        planArgs.plan = DeviceAddress950V3(plan);
        LaunchFwdPlan950V3(planArgs, plan, cu_seqlens_q, cu_seqlens_k, seqused_q_, seqlens_k, page_table,
                           scheduler_metadata_);
    }

    if (empty_query) {
        auto lse = at::empty(return_softmax_lse ? std::vector<int64_t>{num_heads, 0} : std::vector<int64_t>{0},
                             q.options().dtype(at::kFloat));
        auto empty_accum = at::empty({0}, q.options().dtype(at::kFloat));
        auto empty_lse_accum = at::empty({0}, q.options().dtype(at::kFloat));
        return {out, lse, empty_accum, empty_lse_accum};
    }

    // ============================================================
    // 8. Allocate output-side buffers on NPU
    // ============================================================
    at::Tensor softmaxlse = at::empty({0}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    if (return_softmax_lse && is_varlen_q) {
        softmaxlse = at::empty({num_heads, sizes[0]}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    } else if (return_softmax_lse) {
        softmaxlse = at::empty({batch_size, num_heads, seqlen_q}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    }
    if (return_softmax_lse) {
        softmaxlse.fill_(std::numeric_limits<float>::infinity());
    }
    const auto fmt = static_cast<Format>(layout_enum);
    const auto maskCategory = static_cast<MaskCategory>(staticMask.maskType);
    auto* const planDevice = static_cast<uint8_t*>(plan.data_ptr());
    const auto address = [](const at::Tensor& t) {
        return t.defined() ? static_cast<uint8_t*>(t.data_ptr()) : nullptr;
    };
    auto* const kvSequenceDevice = is_varlen_kv ? address(cu_seqlens_k)
                                               : (planDevice + fa3_fwd::KvScratchOffset(batch_size));

    const float softmax_scale = softmax_scale_.value_or(1.0f / std::sqrt(static_cast<float>(head_size_q)));
    const float scale_value = is_softcap ? softmax_scale / softcap : softmax_scale;
    const FwdLaunchArgs fwdArgs{
        is_bf16,
        fmt,
        maskCategory,
        paged_KV,
        enableDN,
        return_softmax_lse,
        is_softcap,
        blockDim,
        aclStream,
        address(q),
        address(k),
        address(v),
        address(mask),
        address(page_table),
        address(out),
        return_softmax_lse ? address(softmaxlse) : address(out),
        address(cu_seqlens_q),
        kvSequenceDevice,
        seqused_q_ ? address(*seqused_q_) : nullptr,
        address(seqlens_k),
        address(workspace),
        planDevice,
        scale_value,
        softcap,
        static_cast<uint32_t>(num_blocks),
        static_cast<uint32_t>(max_num_blocks_per_seq),
        kv_seqlen_allocated,
        quantMode,
        qDescale,
        kDescale,
        vDescale,
        {paged_KV ? static_cast<uint64_t>(k.stride(0)) : 0U, paged_KV ? static_cast<uint64_t>(v.stride(0)) : 0U,
         paged_KV ? static_cast<uint64_t>(k.stride(1)) : 0U, paged_KV ? static_cast<uint64_t>(v.stride(1)) : 0U,
         paged_KV ? static_cast<uint64_t>(k.stride(2)) : 0U, paged_KV ? static_cast<uint64_t>(v.stride(2)) : 0U,
         is_varlen_q ? static_cast<uint64_t>(q.size(0)) : 0U},
        dynamicDispatch};
    auto launch_fa_infer = [fwdArgs]() -> int {
        launch_fwd(fwdArgs);
        return 0;
    };
    // RunOpApiV2 keeps the kernel launch inside a torch_npu op context so it can
    // be captured by NPUGraph; an explicit aclrtSynchronizeStream here would
    // fail graph capture with a device error.
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fa_infer", launch_fa_infer);

    at::Tensor empty_accum = at::empty({0}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    at::Tensor empty_softmax_lse_accum = at::empty({0}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    return {out, softmaxlse, empty_accum, empty_softmax_lse_accum};
}
