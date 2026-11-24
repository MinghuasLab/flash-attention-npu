// Copyright (c) 2026, Minghua Shen.

// ✅ FP16 / BF16
// ✅ Causal mask
// ✅ Paged KV (page_table)
// ✅ MQA / GQA
// ✅ Varlen Q (cu_seqlens_q + max_seqlen_q)
// ✅ return_softmax_lse
// ✅ SWA / window_size (host normalize + MASK_SWA dispatch)
// ✅ num_splits (FlashDecode for paged KV + TND)
// ❌ pack_gqa, min_seqlen_k, gather_kv_indices, learnable_sink
// ✅ softcap (tanh, folded into the online-softmax epilogue)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <c10/core/Device.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "fwd_dispatch.hpp"
#include "tilingdata.h"
#include "../attention_window.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "tiling/platform/platform_ascendc.h"
#include "fa_fwd_prepare_plan.h"
#include "fa_metadata_args.h"

extern __global__ __aicpu__ uint32_t PrepareFlashAttnFwd950V4(void* args);

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace {
uint64_t DeviceAddress(const at::Tensor& tensor)
{
    return tensor.defined() ? reinterpret_cast<uint64_t>(tensor.data_ptr()) : 0;
}

void LaunchForwardPreparation(FA4FwdPrepareArgs args, const at::Tensor& plan,
                              const std::vector<at::Tensor>& inputs)
{
    const auto current = c10_npu::getCurrentNPUStream();
    auto aicpu = c10_npu::getNPUStreamFromPool();
    if (aicpu == current) {
        aicpu = c10_npu::getNPUStreamFromPool();
    }
    const auto currentStream = current.stream(false);
    const auto aicpuStream = aicpu.stream(false);
    TORCH_CHECK(aicpuStream != currentStream, "FA4 preparation requires a separate stream");
    struct PrepareEvents {
        aclrtEvent ready = nullptr;
        aclrtEvent done = nullptr;
    };
    static thread_local std::unordered_map<aclrtStream, PrepareEvents> eventsByStream;
    auto& events = eventsByStream[currentStream];
    if (events.ready == nullptr) {
        constexpr uint32_t flags = ACL_EVENT_SYNC | ACL_EVENT_CAPTURE_STREAM_PROGRESS;
        TORCH_CHECK(aclrtCreateEventExWithFlag(&events.ready, flags) == ACL_SUCCESS,
                    "create FA4 preparation input event failed");
        TORCH_CHECK(aclrtCreateEventExWithFlag(&events.done, flags) == ACL_SUCCESS,
                    "create FA4 preparation completion event failed");
    }
    auto prepare = [args, currentStream, aicpuStream, events]() mutable -> int {
        TORCH_CHECK(aclrtRecordEvent(events.ready, currentStream) == ACL_SUCCESS,
                    "record FA4 preparation input event failed");
        TORCH_CHECK(aclrtStreamWaitEvent(aicpuStream, events.ready) == ACL_SUCCESS,
                    "wait for FA4 preparation input failed");
        PrepareFlashAttnFwd950V4<<<1, nullptr, aicpuStream>>>(&args, sizeof(args));
        TORCH_CHECK(aclrtRecordEvent(events.done, aicpuStream) == ACL_SUCCESS,
                    "record FA4 preparation completion event failed");
        TORCH_CHECK(aclrtStreamWaitEvent(currentStream, events.done) == ACL_SUCCESS,
                    "wait for FA4 preparation completion failed");
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fa4_prepare_forward", prepare);
    c10_npu::NPUCachingAllocator::recordStream(plan.storage().data_ptr(), aicpu);
    for (const auto& tensor : inputs) {
        if (tensor.defined()) {
            c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), aicpu);
            c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), current);
        }
    }
}
} // namespace

std::vector<at::Tensor> mha_fwd(at::Tensor q, at::Tensor k, at::Tensor v, std::optional<at::Tensor> q_v_,
                                std::optional<at::Tensor> out_, std::optional<at::Tensor> cu_seqlens_q_,
                                std::optional<at::Tensor> cu_seqlens_k_, std::optional<at::Tensor> seqused_q_,
                                std::optional<at::Tensor> seqused_k_, std::optional<int64_t> max_seqlen_q_,
                                std::optional<int64_t> max_seqlen_k_, std::optional<int64_t> min_seqlen_k_,
                                std::optional<at::Tensor> page_table_, std::optional<at::Tensor> gather_kv_indices_,
                                std::optional<float> softmax_scale_, bool is_causal, int64_t window_size_left,
                                int64_t window_size_right, float softcap, int64_t num_splits,
                                std::optional<bool> pack_gqa_, std::optional<at::Tensor> learnable_sink_,
                                bool return_lse, std::optional<at::Tensor> scheduler_metadata_, int64_t layout,
                                std::optional<at::Tensor> q_descale_, std::optional<at::Tensor> k_descale_,
                                std::optional<at::Tensor> v_descale_, bool disable_scheduler_metadata,
                                bool infer_max_seqlen_k)
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
    const uint32_t quantMode = q.scalar_type() == at::ScalarType::Float8_e4m3fn ? 1U : 0U;
    TORCH_CHECK(is_bf16 || is_fp16 || quantMode == 1, "FlashAttention only supports FP16, BF16 and FP8_E4M3FN inputs");
    TORCH_CHECK(quantMode == 1 || (!q_descale_ && !k_descale_ && !v_descale_),
                "descales are only supported for FP8_E4M3FN inputs");
    TORCH_CHECK(q.device().type() == at::kPrivateUse1, "query must be on NPU");
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device(),
                "query, key and value must be on the same NPU device");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    CHECK_CONTIGUOUS(q);
    if (!page_table_.has_value() || quantMode != 0) {
        CHECK_CONTIGUOUS(k);
        CHECK_CONTIGUOUS(v);
    }

    // ============================================================
    // 2. reject list
    // ============================================================
    TORCH_CHECK(!min_seqlen_k_.has_value(), "950 backend (v4) does not support min_seqlen_k");
    TORCH_CHECK(!gather_kv_indices_.has_value(), "950 backend (v4) does not support gather_kv_indices");
    TORCH_CHECK(!learnable_sink_.has_value(), "950 backend (v4) does not support learnable_sink");
    TORCH_CHECK(num_splits >= 0 && num_splits <= static_cast<int64_t>(blockDim),
                "950 backend (v4) requires num_splits in [0, ", blockDim, "]");
    TORCH_CHECK(!pack_gqa_.has_value() || !pack_gqa_.value(), "950 backend (v4) does not support pack_gqa");

    // ============================================================
    // 3. paged / varlen mode + per-tensor checks
    // ============================================================
    const bool paged_KV = page_table_.has_value();
    const bool is_varlen_q = cu_seqlens_q_.has_value();
    const bool is_varlen_kv = cu_seqlens_k_.has_value();
    const bool is_softcap = softcap > 0.0f;

    if (num_splits > 1) {
        TORCH_CHECK(paged_KV && is_varlen_q, "950 backend (v4) num_splits>1 requires paged KV cache and "
                                             "TND varlen query");
    }

    at::Tensor cu_seqlens_q, cu_seqlens_k, page_table, seqlens_k;

    if (paged_KV) {
        page_table = page_table_.value();
        TORCH_CHECK(page_table.device() == q.device(), "page_table must be on the same NPU device as query");
        TORCH_CHECK(page_table.dtype() == torch::kInt32, "page_table must have dtype int32");
        CHECK_CONTIGUOUS(page_table);
        TORCH_CHECK(page_table.dim() == 2, "page_table must be rank 2");
    }
    if (is_varlen_q) {
        cu_seqlens_q = cu_seqlens_q_.value();
        CHECK_CONTIGUOUS(cu_seqlens_q);
        TORCH_CHECK(cu_seqlens_q.device() == q.device(), "cu_seqlens_q must be on the query device");
        TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
        TORCH_CHECK(max_seqlen_q_.has_value(), "max_seqlen_q must be provided if cu_seqlens_q is provided");
    }
    if (is_varlen_kv) {
        cu_seqlens_k = cu_seqlens_k_.value();
        CHECK_CONTIGUOUS(cu_seqlens_k);
        TORCH_CHECK(cu_seqlens_k.device() == q.device(), "cu_seqlens_k must be on the query device");
        TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");
        TORCH_CHECK(!paged_KV, "If cu_seqlens_k is passed in, paged table is not supported");
    }
    if (seqused_k_.has_value()) {
        seqlens_k = *seqused_k_;
        CHECK_CONTIGUOUS(seqlens_k);
        TORCH_CHECK(seqlens_k.device() == q.device() && seqlens_k.dtype() == at::kInt && seqlens_k.dim() == 1,
                    "seqused_k must be rank-1 int32 on the query device");
    }
    // seqused_q caps the *used* Q lengths; the packed region layout stays on cu.
    if (seqused_q_.has_value()) {
        TORCH_CHECK(is_varlen_q, "seqused_q requires cu_seqlens_q");
        const at::Tensor& used_q = seqused_q_.value();
        TORCH_CHECK(used_q.dtype() == torch::kInt32, "seqused_q must have dtype int32");
        TORCH_CHECK(used_q.dim() == 1, "seqused_q must be rank 1");
        CHECK_CONTIGUOUS(used_q);
        TORCH_CHECK(used_q.device() == q.device(), "seqused_q must be on the query device");
    }
    TORCH_CHECK(softcap >= 0.0f, "softcap must be non-negative (0.0 disables softcap)");

    // ============================================================
    // 4. Shape extraction
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

    // ============================================================
    // 5. Extract logical dimensions
    // ============================================================
    const auto sizes = q.sizes();
    int batch_size, seqlen_q, num_heads, head_size_q;
    if (is_varlen_q) {
        batch_size = static_cast<int>(cu_seqlens_q.size(0)) - 1;
        TORCH_CHECK(*max_seqlen_q_ >= 0 && *max_seqlen_q_ <= std::numeric_limits<int32_t>::max(),
                    "max_seqlen_q must fit in nonnegative int32");
        seqlen_q = static_cast<int>(*max_seqlen_q_);
        num_heads = static_cast<int>(sizes[1]);
        head_size_q = static_cast<int>(sizes[2]);
    } else {
        batch_size = static_cast<int>(sizes[0]);
        seqlen_q = static_cast<int>(sizes[1]);
        num_heads = static_cast<int>(sizes[2]);
        head_size_q = static_cast<int>(sizes[3]);
    }

    // 5b. Layout (FwdLayout, resolved by the Python caller; Q/K/V share one layout)
    TORCH_CHECK(layout >= static_cast<int64_t>(FwdLayout::TND) && layout <= static_cast<int64_t>(FwdLayout::BNSD),
                "950 backend (v4) received an unsupported layout code (", layout, ")");
    const FwdLayout layout_enum = static_cast<FwdLayout>(layout);
    TORCH_CHECK((layout_enum == FwdLayout::TND) == is_varlen_q,
                "950 backend (v4) layout code conflicts with cu_seqlens_q: TND requires it, "
                "BSND and BNSD forbid it");
    if (layout_enum == FwdLayout::BNSD) {
        // BNSD stores heads on dim1 and the sequence on dim2, i.e. the transpose
        // of the BSND reading assigned above.
        num_heads = static_cast<int>(sizes[1]);
        seqlen_q = static_cast<int>(sizes[2]);
    }

    int64_t kv_seqlen_allocated = 0;
    if (layout_enum == FwdLayout::BNSD) {
        // BNSD requires Q, K and V to share the same rank-4 (B, H, S, D) layout.
        TORCH_CHECK(!paged_KV && !is_varlen_kv && k.dim() == 4 && v.dim() == 4,
                    "950 backend (v4) BNSD layout requires rank-4, non-paged key/value tensors with "
                    "the same (B, H, S, D) layout as the query");
        kv_seqlen_allocated = static_cast<int64_t>(k.size(2));
        TORCH_CHECK(k.size(0) == v.size(0) && k.size(1) == v.size(1) && k.size(2) == v.size(2),
                    "950 backend (v4) BNSD key and value must share batch, head and sequence dimensions "
                    "(the head dimension may differ)");
    }

    if (layout_enum == FwdLayout::BSND && !paged_KV && k.dim() == 4) {
        kv_seqlen_allocated = static_cast<int64_t>(k.size(1));
    }

    const int max_num_blocks_per_seq = !paged_KV ? 0 : static_cast<int>(page_table.size(1));
    const int num_blocks = !paged_KV ? 0 : static_cast<int>(k.size(0));
    const int page_block_size = !paged_KV ? 128 : static_cast<int>(k.size(1));
    const int num_heads_k = layout_enum == FwdLayout::BNSD
                                ? static_cast<int>(k.size(1))
                                : (k.dim() == 3 ? static_cast<int>(k.size(1)) : static_cast<int>(k.size(2)));
    const int head_size_v = static_cast<int>(v.size(-1));

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    const bool empty_query = is_varlen_q && q.size(0) == 0 && seqlen_q == 0;
    TORCH_CHECK((seqlen_q > 0 || empty_query) && num_heads > 0,
                "query sequence length and head count must be positive");
    if (is_varlen_kv) {
        TORCH_CHECK(is_varlen_q && cu_seqlens_k.dim() == 1 && cu_seqlens_k.numel() == batch_size + 1,
                    "cu_seqlens_k requires a rank-1 batch+1 tensor with packed queries");
    }
    if (seqlens_k.defined() && seqlens_k.numel() == batch_size + 1 && is_varlen_q && !paged_KV && !is_varlen_kv) {
        cu_seqlens_k = seqlens_k;
        seqlens_k = at::Tensor();
    }
    TORCH_CHECK(!seqlens_k.defined() || seqlens_k.numel() == batch_size,
                "seqused_k must contain one used KV length per batch");
    TORCH_CHECK(!seqused_q_ || seqused_q_->numel() == batch_size,
                "seqused_q must contain one used Q length per batch");
    TORCH_CHECK(!paged_KV || seqlens_k.defined(), "paged KV requires used KV lengths");
    TORCH_CHECK(!is_varlen_q || paged_KV || cu_seqlens_k.defined() || seqlens_k.defined(),
                "packed KV requires cumulative offsets or used KV lengths");
    TORCH_CHECK(k.size(-1) == head_size_q, "query and key must have the same head dimension");
    const bool d512_heads = head_size_q == 512 && head_size_v == 512;
    TORCH_CHECK((head_size_q >= 1 && head_size_q <= 256) || d512_heads,
                "FlashAttention supports q/k head dimensions in [1, 256] plus exact D512");
    TORCH_CHECK((head_size_v >= 1 && head_size_v <= 256) || d512_heads,
                "FlashAttention supports value head dimensions in [1, 256] plus exact D512");
    TORCH_CHECK(
        !(page_block_size != 128 && page_block_size != 256 && page_block_size != 512 && page_block_size != 1024),
        "FlashAttention only supports page_block_size dimension 128 or 256 or 512 or 1024");
    TORCH_CHECK(num_heads_k > 0, "key/value head count must be positive");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    if (!is_varlen_q && !paged_KV) {
        TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "query and key/value batch sizes must match");
    }
    if (paged_KV) {
        TORCH_CHECK(num_blocks > 0 && max_num_blocks_per_seq > 0,
                    "paged KV cache and page_table must contain at least one block");
        TORCH_CHECK(page_table.size(0) == batch_size, "page_table batch dimension must match query batch size");
        TORCH_CHECK(k.size(0) == v.size(0) && k.size(1) == v.size(1), "K/V page counts and sizes must match");
        const auto isBBND = [](const at::Tensor& tensor) {
            return tensor.stride(1) == tensor.size(2) * tensor.size(3) &&
                   tensor.stride(2) == tensor.size(3) && tensor.stride(3) == 1;
        };
        const auto isBNBD = [](const at::Tensor& tensor) {
            return tensor.stride(1) == tensor.size(3) &&
                   tensor.stride(2) == tensor.size(1) * tensor.size(3) && tensor.stride(3) == 1;
        };
        TORCH_CHECK((isBBND(k) && isBBND(v)) || (isBNBD(k) && isBNBD(v)),
                    "paged K/V must share packed BBND or BNBD inner layout");
    }

    TORCH_CHECK(quantMode == 0 || (head_size_q == 128 && head_size_v == 128),
                "950 V4 FP8 requires Q/K/V head dimensions of 128");
    uint8_t* qDescale = nullptr;
    uint8_t* kDescale = nullptr;
    uint8_t* vDescale = nullptr;
    auto set_descale = [&](const std::optional<at::Tensor>& scale, uint8_t*& ptr) {
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
    const auto output_dtype = quantMode == 1 ? at::kBFloat16 : q.scalar_type();
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
        TORCH_CHECK(out.scalar_type() == output_dtype, "output dtype must be BF16 for FP8, otherwise match inputs");
        TORCH_CHECK(out.device() == q.device(), "output must be on the same device as query");
        TORCH_CHECK(out.sizes().vec() == output_sizes,
                    "output shape must match query except for the last dimension, which must match value");
        CHECK_CONTIGUOUS(out);
    } else {
        out = at::empty(output_sizes, q.options().dtype(output_dtype));
    }

    const int64_t kvCapacity =
        paged_KV ? int64_t(max_num_blocks_per_seq) * page_block_size
                 : (is_varlen_q ? std::min(k.size(0), v.size(0))
                                : (layout_enum == FwdLayout::BNSD ? std::min(k.size(2), v.size(2))
                                                                 : std::min(k.size(1), v.size(1))));
    TORCH_CHECK(kvCapacity <= std::numeric_limits<int32_t>::max(), "KV length capacity must fit in int32");
    if (is_varlen_q) {
        TORCH_CHECK(q.size(0) <= std::numeric_limits<int32_t>::max(), "packed Q length must fit in int32");
    }
    TORCH_CHECK(v.size(layout_enum == FwdLayout::BNSD || k.dim() == 3 ? 1 : 2) == num_heads_k,
                "key and value must have the same number of heads");
    if (!is_varlen_q && !paged_KV) {
        const auto axis = layout_enum == FwdLayout::BNSD ? 2 : 1;
        TORCH_CHECK(k.size(axis) == v.size(axis), "dense K/V sequence lengths must match");
    }
    if (max_seqlen_k_) {
        TORCH_CHECK(*max_seqlen_k_ >= 0 && *max_seqlen_k_ <= std::numeric_limits<int32_t>::max(),
                    "max_seqlen_k must fit in nonnegative int32");
    }
    const auto byteOptions = q.options().dtype(at::kByte);
    at::Tensor metadata = scheduler_metadata_.value_or(at::Tensor());
    if (metadata.defined()) {
        TORCH_CHECK(metadata.device() == q.device() && metadata.dtype() == at::kByte && metadata.is_contiguous(),
                    "scheduler_metadata must be a contiguous NPU byte tensor");
        TORCH_CHECK(metadata.nbytes() == fa_metadata::MetadataBytes(false),
                    "scheduler_metadata has incompatible mask/layout size");
    }
    const bool makeKvOffsets = is_varlen_q && !paged_KV && !cu_seqlens_k.defined();
    const bool writeKvLengths = !seqlens_k.defined();
    const int64_t kvOffsetBytes = makeKvOffsets ? (int64_t(batch_size) + 1) * sizeof(int32_t) : 0;
    const int64_t kvLengthBytes = writeKvLengths ? int64_t(batch_size) * sizeof(int32_t) : 0;
    const auto planSize = static_cast<int64_t>(sizeof(FA4FwdExecution)) + kvOffsetBytes + kvLengthBytes;
    const uint64_t workspaceSize = std::max<uint64_t>(128U * 1024U * 1024U, fa_metadata::WorkSpaceSize(blockDim));
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, byteOptions);

    FA4FwdPrepareArgs prepare{};
    prepare.qOffsets = DeviceAddress(cu_seqlens_q);
    prepare.kOffsets = DeviceAddress(cu_seqlens_k);
    prepare.qUsed = DeviceAddress(seqused_q_.value_or(at::Tensor()));
    prepare.kUsed = DeviceAddress(seqlens_k);
    prepare.pageTable = DeviceAddress(page_table);
    prepare.metadata = DeviceAddress(metadata);
    prepare.workspaceSize = workspaceSize;
    prepare.qTokens = is_varlen_q ? q.size(0) : int64_t(batch_size) * seqlen_q;
    prepare.kvCapacity = kvCapacity;
    prepare.maxKv = max_seqlen_k_.value_or(-1);
    prepare.left = window_size_left;
    prepare.right = window_size_right;
    prepare.batch = batch_size;
    prepare.maxQ = seqlen_q;
    prepare.heads = num_heads;
    prepare.kvHeads = num_heads_k;
    prepare.qDim = head_size_q;
    prepare.vDim = head_size_v;
    prepare.pageSize = page_block_size;
    prepare.pageCount = num_blocks;
    prepare.maxPages = max_num_blocks_per_seq;
    prepare.coreCount = blockDim;
    prepare.splits = num_splits;
    prepare.quantMode = quantMode;
    prepare.layout = layout_enum;
    prepare.softcap = softcap;
    const float scale = softmax_scale_.value_or(1.0f / std::sqrt(static_cast<float>(head_size_q)));
    prepare.scale = is_softcap ? scale / softcap : scale;
    prepare.causal = is_causal;
    prepare.useHostSchedule = disable_scheduler_metadata && !metadata.defined();
    prepare.inferMaxKv = infer_max_seqlen_k && !prepare.useHostSchedule && !metadata.defined();
    prepare.bf16 = is_bf16;
    prepare.lse = return_lse;
    // With no length tensors the entire plan can be derived from tensor shape
    // on Host. Capture uses the device producer to avoid staging host memory.
    const bool staticDense = !is_varlen_q && !paged_KV && !seqlens_k.defined() && !metadata.defined() &&
                             c10_npu::currentStreamCaptureStatusMayInitCtx() == c10_npu::CaptureStatus::None;
    at::Tensor plan;
    if (staticDense) {
        auto cpuPlan = at::empty({planSize}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
        auto* cpuAddr = cpuPlan.data_ptr<uint8_t>();
        auto* cpuExecution = reinterpret_cast<FA4FwdExecution*>(cpuAddr);
        auto staticArgs = prepare;
        staticArgs.kvLengths = reinterpret_cast<uint64_t>(cpuAddr + sizeof(FA4FwdExecution));
        const auto status = fa4_fwd_plan::Prepare(staticArgs, *cpuExecution);
        TORCH_CHECK(status == FA4FwdStatus::Valid, "FA4 dense forward plan failed");
        cpuExecution->status = static_cast<uint32_t>(status);
        plan = cpuPlan.to(q.device());
    } else {
        plan = at::empty({planSize}, byteOptions);
    }
    auto* planAddr = plan.data_ptr<uint8_t>();
    auto* kvOffsetsDev = makeKvOffsets ? planAddr + sizeof(FA4FwdExecution) : nullptr;
    auto* kvLengthsDev = writeKvLengths ? planAddr + sizeof(FA4FwdExecution) + kvOffsetBytes
                                        : reinterpret_cast<uint8_t*>(seqlens_k.data_ptr());
    if (!staticDense) {
        prepare.execution = reinterpret_cast<uint64_t>(planAddr);
        prepare.kvLengths = reinterpret_cast<uint64_t>(kvLengthsDev);
        prepare.kvOffsets = reinterpret_cast<uint64_t>(kvOffsetsDev);
        LaunchForwardPreparation(prepare, plan,
                                 {cu_seqlens_q, cu_seqlens_k, seqused_q_.value_or(at::Tensor()), seqlens_k,
                                  page_table, metadata});
    }

    auto softmaxlse = at::empty(
        !return_lse ? std::vector<int64_t>{0}
                    : (is_varlen_q ? std::vector<int64_t>{num_heads, q.size(0)}
                                   : std::vector<int64_t>{batch_size, num_heads, seqlen_q}),
        q.options().dtype(at::kFloat));
    if (return_lse) {
        softmaxlse.fill_(std::numeric_limits<float>::infinity());
    }

    if (!empty_query) {
        bool is_local = false;
        bool normalizedCausal = false;
        fa_window::NormalizeAttentionWindow(is_causal, seqlen_q, max_seqlen_k_.value_or(kvCapacity),
                                            window_size_left, window_size_right, normalizedCausal, is_local);
        const bool adaptiveMask = prepare.inferMaxKv && !max_seqlen_k_.has_value() && prepare.left >= 0 && is_local;
        const bool adaptiveDn = paged_KV && is_varlen_q && num_splits != 1 && !is_local && !normalizedCausal &&
                                max_seqlen_k_.value_or(kvCapacity) >= 1024 &&
                                (prepare.useHostSchedule || seqlen_q <= 16);
        const MaskCategory maskCategory = is_local ? MaskCategory::MASK_SWA
                                                  : (normalizedCausal ? MaskCategory::MASK_CAUSAL : MaskCategory::NO_MASK);
        at::Tensor mask;
        if ((is_local || normalizedCausal || adaptiveMask) && quantMode == 0 && !d512_heads) {
            mask = MakeDeviceTriuMask();
        }
        auto* kvSeqDev = cu_seqlens_k.defined() ? reinterpret_cast<uint8_t*>(cu_seqlens_k.data_ptr())
                                               : (makeKvOffsets ? kvOffsetsDev : kvLengthsDev);
        FwdLaunchArgs fwdArgs{};
        fwdArgs.is_bf16 = is_bf16;
        fwdArgs.layout = static_cast<Format>(layout_enum);
        fwdArgs.mask_category = maskCategory;
        fwdArgs.paged_kv = paged_KV;
        fwdArgs.enable_dn = !is_local && !normalizedCausal && head_size_q <= 256 && head_size_v <= 256;
        fwdArgs.adaptive_mask = adaptiveMask;
        fwdArgs.adaptive_dn = adaptiveDn;
        fwdArgs.lse_mode = return_lse;
        fwdArgs.has_softcap = is_softcap;
        fwdArgs.block_dim = blockDim;
        fwdArgs.stream = aclStream;
        fwdArgs.q_device = static_cast<uint8_t*>(q.data_ptr());
        fwdArgs.k_device = static_cast<uint8_t*>(k.data_ptr());
        fwdArgs.v_device = static_cast<uint8_t*>(v.data_ptr());
        fwdArgs.mask_device = reinterpret_cast<uint8_t*>(DeviceAddress(mask));
        fwdArgs.block_table_device = reinterpret_cast<uint8_t*>(prepare.pageTable);
        fwdArgs.o_device = static_cast<uint8_t*>(out.data_ptr());
        fwdArgs.lse_device = return_lse ? static_cast<uint8_t*>(softmaxlse.data_ptr()) : fwdArgs.o_device;
        fwdArgs.q_seq_device = reinterpret_cast<uint8_t*>(prepare.qOffsets);
        fwdArgs.kv_seq_device = kvSeqDev;
        fwdArgs.seq_used_q_device = reinterpret_cast<uint8_t*>(prepare.qUsed);
        fwdArgs.seq_used_kv_device = kvLengthsDev;
        fwdArgs.workspace_device = static_cast<uint8_t*>(workspace.data_ptr());
        fwdArgs.tiling_device = planAddr + offsetof(FA4FwdExecution, tiling);
        fwdArgs.runtime_tiling_device = planAddr;
        fwdArgs.kv_seqlen_allocated = kv_seqlen_allocated;
        fwdArgs.quant_mode = quantMode;
        fwdArgs.q_descale_device = qDescale;
        fwdArgs.k_descale_device = kDescale;
        fwdArgs.v_descale_device = vDescale;
        fwdArgs.storage =
            FwdStorageParams{static_cast<uint64_t>(paged_KV ? k.stride(0) : 0),
                             static_cast<uint64_t>(paged_KV ? v.stride(0) : 0),
                             static_cast<uint64_t>(paged_KV ? k.stride(1) : 0),
                             static_cast<uint64_t>(paged_KV ? v.stride(1) : 0),
                             static_cast<uint64_t>(paged_KV ? k.stride(2) : 0),
                             static_cast<uint64_t>(paged_KV ? v.stride(2) : 0),
                             static_cast<uint64_t>(prepare.qTokens)};
        auto launch_fa_infer = [fwdArgs]() -> int {
            launch_fwd(fwdArgs);
            return 0;
        };
        at_npu::native::OpCommand::RunOpApiV2("ascendc_fa_infer", launch_fa_infer);

    }

    at::Tensor empty_accum = at::empty({0}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    at::Tensor empty_softmax_lse_accum = at::empty({0}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    return {out, softmaxlse, empty_accum, empty_softmax_lse_accum};
}
