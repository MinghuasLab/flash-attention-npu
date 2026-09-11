/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend950 FlashAttention v3 backward scaffold.
 *
 * The host ABI matches the Python v3 backward entry. The device kernel is a
 * no-op placeholder; replace the implementation in fag_kernel.cpp when the
 * real backward algorithm lands.
 */

#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include <torch/extension.h>

#include "acl/acl.h"
#include "fag_tiling.cpp"
#include "fag_kernel.cpp"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

std::vector<at::Tensor>
mha_bwd(
    at::Tensor dout,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor softmax_lse,
    std::optional<at::Tensor> dq_,
    std::optional<at::Tensor> dk_,
    std::optional<at::Tensor> dv_,
    std::optional<at::Tensor> cu_seqlens_q_,
    std::optional<at::Tensor> cu_seqlens_k_,
    std::optional<at::Tensor> seqused_q_,
    std::optional<at::Tensor> seqused_k_,
    std::optional<int64_t> max_seqlen_q_,
    std::optional<int64_t> max_seqlen_k_,
    std::optional<double> softmax_scale_,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    double softcap,
    bool deterministic,
    int64_t sm_margin)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const bool is_varlen = cu_seqlens_q_.has_value();
    const uint32_t aic_num =
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    const uint32_t aiv_num =
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();

    TORCH_CHECK(is_varlen == cu_seqlens_k_.has_value(),
                "Ascend950 v3 bwd: cu_seqlens_q and cu_seqlens_k must be provided together");
    TORCH_CHECK(!seqused_q_.has_value() && !seqused_k_.has_value(),
                "Ascend950 v3 bwd does not support seqused_q/seqused_k now");
    TORCH_CHECK(window_size_left == -1 && window_size_right == -1,
                "Ascend950 v3 bwd does not support sliding-window attention now");
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0.0 &&
                    softcap <= std::numeric_limits<float>::max(),
                "Ascend950 v3 bwd: softcap must be finite and non-negative");
    TORCH_CHECK(sm_margin == 0,
                "Ascend950 v3 bwd does not support sm_margin");
    TORCH_CHECK(q.dtype() == at::kHalf || q.dtype() == at::kBFloat16,
                "Ascend950 v3 bwd only supports FP16 and BF16");
    TORCH_CHECK(k.dtype() == q.dtype() && v.dtype() == q.dtype() &&
                dout.dtype() == q.dtype() && out.dtype() == q.dtype(),
                "Ascend950 v3 bwd: q/k/v/out/dout must have the same dtype");
    TORCH_CHECK(q.dim() == (is_varlen ? 3 : 4),
                "Ascend950 v3 bwd: q must use TND or BSND layout");
    TORCH_CHECK(k.dim() == q.dim() && v.dim() == q.dim() &&
                dout.dim() == q.dim() && out.dim() == q.dim(),
                "Ascend950 v3 bwd: q/k/v/out/dout must use the same layout rank");
    TORCH_CHECK(!is_varlen || (max_seqlen_q_.has_value() && max_seqlen_k_.has_value()),
                "Ascend950 v3 bwd: max sequence lengths are required for TND");

    const auto q_sizes = q.sizes();
    const auto k_sizes = k.sizes();
    const auto v_sizes = v.sizes();
    const int64_t batch_size =
        is_varlen ? cu_seqlens_q_.value().size(0) - 1 : q_sizes[0];
    const int64_t q_seqlen =
        is_varlen ? max_seqlen_q_.value() : q_sizes[1];
    const int64_t kv_seqlen =
        is_varlen ? max_seqlen_k_.value() : k_sizes[1];
    const int64_t num_heads = is_varlen ? q_sizes[1] : q_sizes[2];
    const int64_t num_heads_kv = is_varlen ? k_sizes[1] : k_sizes[2];
    const int64_t qk_head_dim = q_sizes.back();
    const int64_t k_head_dim = k_sizes.back();
    const int64_t v_head_dim = v_sizes.back();

    TORCH_CHECK(batch_size > 0, "Ascend950 v3 bwd: batch size must be positive");
    TORCH_CHECK(q_seqlen > 0 && kv_seqlen > 0,
                "Ascend950 v3 bwd: sequence lengths must be positive");
    TORCH_CHECK(num_heads > 0 && num_heads_kv > 0 &&
                num_heads % num_heads_kv == 0,
                "Ascend950 v3 bwd: KV heads must divide query heads");
    TORCH_CHECK(qk_head_dim == k_head_dim,
                "Ascend950 v3 bwd: q and k must share the same head dimension");
    TORCH_CHECK(qk_head_dim > 0 && qk_head_dim <= 256,
                "Ascend950 v3 bwd: q/k head dimension must be in (0, 256]");
    TORCH_CHECK(v_head_dim > 0 && v_head_dim <= 256,
                "Ascend950 v3 bwd: v head dimension must be in (0, 256]");
    TORCH_CHECK(dout.size(-1) == v_head_dim && out.size(-1) == v_head_dim,
                "Ascend950 v3 bwd: dout/out head dimension must match v");

    at::Tensor dq = dq_.has_value() ? dq_.value() : at::empty_like(q);
    at::Tensor dk = dk_.has_value() ? dk_.value() : at::empty_like(k);
    at::Tensor dv = dv_.has_value() ? dv_.value() : at::empty_like(v);

    // ============================================================
    // FAG tiling (host)
    // ============================================================
    FAGTiling950::FAGInfo fag_info{};
    fag_info.scaleValue = static_cast<float>(
        softmax_scale_.value_or(1.0 / std::sqrt(static_cast<double>(qk_head_dim))));
    const bool has_softcap = softcap > 0.0;
    fag_info.softcapValue = static_cast<float>(softcap);
    fag_info.layout = is_varlen ? FAGTiling950::Layout::TND
                                : FAGTiling950::Layout::BSND;
    fag_info.maskType = is_causal ? FAGTiling950::MaskType::CAUSAL
                                  : FAGTiling950::MaskType::NO_MASK;
    fag_info.deterministic = deterministic ? 1U : 0U;
    fag_info.batch = batch_size;
    fag_info.qSeqlen = q_seqlen;
    fag_info.qHeadNum = num_heads;
    fag_info.qkHeadDim = qk_head_dim;
    fag_info.kvSeqlen = kv_seqlen;
    fag_info.kvHeadNum = num_heads_kv;
    fag_info.vHeadDim = v_head_dim;
    fag_info.totalQ = is_varlen
        ? static_cast<uint64_t>(q_sizes[0])
        : static_cast<uint64_t>(batch_size * q_seqlen);
    fag_info.totalKv = is_varlen
        ? static_cast<uint64_t>(k_sizes[0])
        : static_cast<uint64_t>(batch_size * kv_seqlen);
    fag_info.aicNum = aic_num;
    fag_info.aivNum = aiv_num;

    // Pull TND cumulative lengths to host for ABI/shape validation.  The
    // arch35 tiler uses totalQ/totalKv from the actual packed tensors rather
    // than carrying host-only sequence vectors in its device tiling ABI.
    at::Tensor cu_q_cpu;
    at::Tensor cu_k_cpu;
    if (is_varlen) {
        const at::Tensor &cu_q_tensor = cu_seqlens_q_.value();
        const at::Tensor &cu_k_tensor = cu_seqlens_k_.value();
        TORCH_CHECK(cu_q_tensor.dim() == 1 && cu_k_tensor.dim() == 1 &&
                    cu_q_tensor.size(0) == batch_size + 1 &&
                    cu_k_tensor.size(0) == batch_size + 1,
                    "Ascend950 v3 bwd: cu_seqlens must be 1D tensors of length B + 1");
        TORCH_CHECK(cu_q_tensor.dtype() == at::kInt &&
                    cu_k_tensor.dtype() == at::kInt,
                    "Ascend950 v3 bwd: cu_seqlens must have dtype int32");
        TORCH_CHECK(cu_q_tensor.device().type() == at::kPrivateUse1 &&
                    cu_k_tensor.device().type() == at::kPrivateUse1,
                    "Ascend950 v3 bwd: cu_seqlens must be on NPU");
        cu_q_cpu = cu_q_tensor.to(at::Device(at::kCPU)).to(at::kInt).contiguous();
        cu_k_cpu = cu_k_tensor.to(at::Device(at::kCPU)).to(at::kInt).contiguous();
        const int32_t *q_lengths = cu_q_cpu.data_ptr<int32_t>();
        const int32_t *kv_lengths = cu_k_cpu.data_ptr<int32_t>();
        TORCH_CHECK(q_lengths[0] == 0 && kv_lengths[0] == 0,
                    "Ascend950 v3 bwd: cu_seqlens must start at zero");
        for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            TORCH_CHECK(
                q_lengths[batch_idx + 1] >= q_lengths[batch_idx] &&
                    kv_lengths[batch_idx + 1] >= kv_lengths[batch_idx],
                "Ascend950 v3 bwd: cu_seqlens must be nondecreasing");
            TORCH_CHECK(
                q_lengths[batch_idx + 1] - q_lengths[batch_idx] <= q_seqlen &&
                    kv_lengths[batch_idx + 1] - kv_lengths[batch_idx] <=
                        kv_seqlen,
                "Ascend950 v3 bwd: an actual sequence length exceeds max_seqlen");
        }
        TORCH_CHECK(q_lengths[batch_size] ==
                        static_cast<int64_t>(fag_info.totalQ) &&
                    kv_lengths[batch_size] ==
                        static_cast<int64_t>(fag_info.totalKv),
                    "Ascend950 v3 bwd: final cu_seqlens values must match packed tensor lengths");
    }

    uint64_t ub_size = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
        platform_ascendc::CoreMemType::UB, ub_size);
    fag_info.ubSize = ub_size;

    FAGTiling950::FAGTilingData fag_tiling_data{};
    const int64_t tiling_status =
        FAGTiling950::GetFAGTilingParam(fag_info, fag_tiling_data);
    TORCH_CHECK(tiling_status == 0,
                "Ascend950 v3 bwd: arch35 GetFAGTilingParam failed");
    at::Tensor tiling_cpu = at::empty(
        {static_cast<int64_t>(sizeof(FAGTiling950::FAGTilingData))},
        at::device(c10::kCPU).dtype(at::kByte));
    std::memcpy(tiling_cpu.data_ptr<uint8_t>(), &fag_tiling_data,
                sizeof(FAGTiling950::FAGTilingData));
    at::Tensor tiling_device =
        tiling_cpu.to(at::Device(at::kPrivateUse1));

    // ============================================================
    // FAG workspace (device)
    // ============================================================
    const uint64_t workspace_size =
        static_cast<uint64_t>(fag_tiling_data.workspaceSize);
    TORCH_CHECK(workspace_size > 0 &&
                workspace_size <= static_cast<uint64_t>(
                    std::numeric_limits<int64_t>::max()),
                "Ascend950 v3 bwd: invalid workspace size from tiling");
    TORCH_CHECK(workspace_size % sizeof(float) == 0 &&
                    fag_tiling_data.deltaOffset % sizeof(float) == 0,
                "Ascend950 v3 bwd: FP32 workspace offsets must be aligned");
    at::Tensor workspace = at::empty(
        {static_cast<int64_t>(workspace_size / sizeof(float))},
        at::device(at::kPrivateUse1).dtype(at::kFloat));

    auto ptr = [](const at::Tensor &tensor) {
        return static_cast<uint8_t *>(tensor.data_ptr());
    };
    // Arch35 FAG device ABI: cumulative lengths have B entries and do not
    // include the leading zero.
    at::Tensor cu_q_device;
    at::Tensor cu_k_device;
    uint8_t *cu_q = nullptr;
    uint8_t *cu_k = nullptr;
    if (is_varlen) {
        cu_q_device = cu_seqlens_q_.value().slice(
            0, 1, cu_seqlens_q_.value().size(0)).contiguous();
        cu_k_device = cu_seqlens_k_.value().slice(
            0, 1, cu_seqlens_k_.value().size(0)).contiguous();
        cu_q = ptr(cu_q_device);
        cu_k = ptr(cu_k_device);
    }

    at::Tensor mask_cpu_tensor;
    at::Tensor mask_npu_tensor;
    uint8_t* mask = nullptr;
    if (is_causal) {
        mask_cpu_tensor = at::triu(
            at::ones({256, 256}, at::device(c10::kCPU).dtype(at::kByte)), 1)
            .to(at::Device(at::kPrivateUse1));
        mask_npu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
        mask = ptr(mask_npu_tensor);
    }

    // Flush queued PyTorch NPU work (for example tiling, mask and cumulative
    // lengths) before launching the raw mixed AIC/AIV kernel on the ACL stream.
    stream = c10_npu::getCurrentNPUStream().stream(true);

#define LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, IS_CAUSAL, IS_DETERMINISTIC, IS_SOFTCAP) \
    FlashAttentionV3Bwd950<                                                     \
        DTYPE, FAGTiling950::Layout::INPUT_LAYOUT,                              \
        IS_CAUSAL, IS_DETERMINISTIC, IS_SOFTCAP><<<aic_num, nullptr, stream>>>( \
            ptr(dout), ptr(q), ptr(k), ptr(v), ptr(out), mask,                  \
            ptr(softmax_lse), cu_q, cu_k, ptr(dq), ptr(dk), ptr(dv),            \
            ptr(workspace), ptr(tiling_device))

#define DISPATCH_BWD950_FLAGS(DTYPE, INPUT_LAYOUT)                               \
    do {                                                                         \
        if (is_causal) {                                                         \
            if (deterministic) {                                                 \
                if (has_softcap) {                                               \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, true, true, true);         \
                } else {                                                         \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, true, true, false);        \
                }                                                                \
            } else {                                                             \
                if (has_softcap) {                                               \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, true, false, true);        \
                } else {                                                         \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, true, false, false);       \
                }                                                                \
            }                                                                    \
        } else {                                                                 \
            if (deterministic) {                                                 \
                if (has_softcap) {                                               \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, false, true, true);        \
                } else {                                                         \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, false, true, false);       \
                }                                                                \
            } else {                                                             \
                if (has_softcap) {                                               \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, false, false, true);       \
                } else {                                                         \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, false, false, false);      \
                }                                                                \
            }                                                                    \
        }                                                                        \
    } while (0)

    if (q.dtype() == at::kBFloat16) {
        if (is_varlen) {
            DISPATCH_BWD950_FLAGS(bfloat16_t, TND);
        } else {
            DISPATCH_BWD950_FLAGS(bfloat16_t, BSND);
        }
    } else {
        if (is_varlen) {
            DISPATCH_BWD950_FLAGS(half, TND);
        } else {
            DISPATCH_BWD950_FLAGS(half, BSND);
        }
    }

#undef DISPATCH_BWD950_FLAGS
#undef LAUNCH_BWD950

    at::Tensor softmax_d = is_varlen
        ? at::empty({num_heads, q.size(0)}, q.options().dtype(at::kFloat))
        : at::empty({batch_size, num_heads, q_seqlen},
                    q.options().dtype(at::kFloat));

    return {dq, dk, dv, softmax_d};
}
