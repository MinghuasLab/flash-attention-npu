/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include <torch/extension.h>

#include "acl/acl.h"
// Reuse Ascend950 v3 FAG kernels in place (same pattern as 910 v2 → v3).
#include "../flash_attn_npu_3/fag_tiling.cpp"
#include "../flash_attn_npu_3/fag_kernel.cpp"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

std::vector<at::Tensor> mha_bwd(at::Tensor dout, at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out,
                                at::Tensor softmax_lse, std::optional<at::Tensor> dq_, std::optional<at::Tensor> dk_,
                                std::optional<at::Tensor> dv_, std::optional<at::Tensor> cu_seqlens_q_,
                                std::optional<at::Tensor> cu_seqlens_k_, std::optional<at::Tensor> seqused_q_,
                                std::optional<at::Tensor> seqused_k_, std::optional<int64_t> max_seqlen_q_,
                                std::optional<int64_t> max_seqlen_k_, std::optional<double> softmax_scale_,
                                bool is_causal, int64_t window_size_left, int64_t window_size_right, double softcap,
                                bool deterministic, int64_t sm_margin)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const bool is_varlen = cu_seqlens_q_.has_value();
    const uint32_t aic_num = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    const uint32_t aiv_num = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();

    TORCH_CHECK(is_varlen == cu_seqlens_k_.has_value(),
                "Ascend950 v4 bwd: cu_seqlens_q and cu_seqlens_k must be provided together");
    // seqused caps used lengths only; offsets and sizing still follow cu.
    TORCH_CHECK(!seqused_q_.has_value() || is_varlen, "Ascend950 v4 bwd: seqused_q requires cu_seqlens_q");
    TORCH_CHECK(!seqused_k_.has_value() || is_varlen, "Ascend950 v4 bwd: seqused_k requires cu_seqlens_q");
    TORCH_CHECK(window_size_left == -1 && window_size_right == -1,
                "Ascend950 v4 bwd does not support sliding-window attention now");
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0.0 && softcap <= std::numeric_limits<float>::max(),
                "Ascend950 v4 bwd: softcap must be finite and non-negative");
    TORCH_CHECK(sm_margin == 0, "Ascend950 v4 bwd does not support sm_margin");
    TORCH_CHECK(q.dtype() == at::kHalf || q.dtype() == at::kBFloat16, "Ascend950 v4 bwd only supports FP16 and BF16");
    TORCH_CHECK(k.dtype() == q.dtype() && v.dtype() == q.dtype() && dout.dtype() == q.dtype() &&
                    out.dtype() == q.dtype(),
                "Ascend950 v4 bwd: q/k/v/out/dout must have the same dtype");
    TORCH_CHECK(q.dim() == (is_varlen ? 3 : 4), "Ascend950 v4 bwd: q must use TND or BSND layout");
    TORCH_CHECK(k.dim() == q.dim() && v.dim() == q.dim() && dout.dim() == q.dim() && out.dim() == q.dim(),
                "Ascend950 v4 bwd: q/k/v/out/dout must use the same layout rank");
    TORCH_CHECK(!is_varlen || (max_seqlen_q_.has_value() && max_seqlen_k_.has_value()),
                "Ascend950 v4 bwd: max sequence lengths are required for TND");

    const auto q_sizes = q.sizes();
    const auto k_sizes = k.sizes();
    const auto v_sizes = v.sizes();
    const int64_t batch_size = is_varlen ? cu_seqlens_q_.value().size(0) - 1 : q_sizes[0];
    const int64_t q_seqlen = is_varlen ? max_seqlen_q_.value() : q_sizes[1];
    const int64_t kv_seqlen = is_varlen ? max_seqlen_k_.value() : k_sizes[1];
    const int64_t num_heads = is_varlen ? q_sizes[1] : q_sizes[2];
    const int64_t num_heads_kv = is_varlen ? k_sizes[1] : k_sizes[2];
    const int64_t qk_head_dim = q_sizes.back();
    const int64_t k_head_dim = k_sizes.back();
    const int64_t v_head_dim = v_sizes.back();

    TORCH_CHECK(batch_size > 0, "Ascend950 v4 bwd: batch size must be positive");
    if (seqused_q_.has_value()) {
        const at::Tensor& used_q = seqused_q_.value();
        TORCH_CHECK(used_q.dtype() == at::kInt, "Ascend950 v4 bwd: seqused_q must have dtype int32");
        TORCH_CHECK(used_q.dim() == 1 && used_q.size(0) == batch_size,
                    "Ascend950 v4 bwd: seqused_q must be 1D of length B");
        TORCH_CHECK(used_q.is_contiguous(), "Ascend950 v4 bwd: seqused_q must be contiguous");
        TORCH_CHECK(used_q.device().type() == at::kPrivateUse1, "Ascend950 v4 bwd: seqused_q must be on NPU");
    }
    if (seqused_k_.has_value()) {
        const at::Tensor& used_k = seqused_k_.value();
        TORCH_CHECK(used_k.dtype() == at::kInt, "Ascend950 v4 bwd: seqused_k must have dtype int32");
        TORCH_CHECK(used_k.dim() == 1 && used_k.size(0) == batch_size,
                    "Ascend950 v4 bwd: seqused_k must be 1D of length B");
        TORCH_CHECK(used_k.is_contiguous(), "Ascend950 v4 bwd: seqused_k must be contiguous");
        TORCH_CHECK(used_k.device().type() == at::kPrivateUse1, "Ascend950 v4 bwd: seqused_k must be on NPU");
    }
    TORCH_CHECK(!deterministic || (!seqused_q_.has_value() && !seqused_k_.has_value()),
                "Ascend950 v4 bwd: seqused_q/seqused_k with deterministic is not supported yet");
    TORCH_CHECK(q_seqlen > 0 && kv_seqlen > 0, "Ascend950 v4 bwd: sequence lengths must be positive");
    TORCH_CHECK(num_heads > 0 && num_heads_kv > 0 && num_heads % num_heads_kv == 0,
                "Ascend950 v4 bwd: KV heads must divide query heads");
    TORCH_CHECK(qk_head_dim == k_head_dim, "Ascend950 v4 bwd: q and k must share the same head dimension");
    TORCH_CHECK(qk_head_dim > 0 && qk_head_dim <= 256, "Ascend950 v4 bwd: q/k head dimension must be in (0, 256]");
    TORCH_CHECK(v_head_dim > 0 && v_head_dim <= 256, "Ascend950 v4 bwd: v head dimension must be in (0, 256]");
    TORCH_CHECK(dout.size(-1) == v_head_dim && out.size(-1) == v_head_dim,
                "Ascend950 v4 bwd: dout/out head dimension must match v");

    at::Tensor dq = dq_.has_value() ? dq_.value() : at::empty_like(q);
    at::Tensor dk = dk_.has_value() ? dk_.value() : at::empty_like(k);
    at::Tensor dv = dv_.has_value() ? dv_.value() : at::empty_like(v);

    // ============================================================
    // FAG tiling (host)
    // ============================================================
    FAGTiling950::FAGInfo fag_info{};
    fag_info.scaleValue =
        static_cast<float>(softmax_scale_.value_or(1.0 / std::sqrt(static_cast<double>(qk_head_dim))));
    const bool has_softcap = softcap > 0.0;
    fag_info.softcapValue = static_cast<float>(softcap);
    fag_info.layout = is_varlen ? FAGTiling950::Layout::TND : FAGTiling950::Layout::BSND;
    fag_info.maskType = is_causal ? FAGTiling950::MaskType::CAUSAL : FAGTiling950::MaskType::NO_MASK;
    fag_info.deterministic = deterministic ? 1U : 0U;
    // Deterministic accumulation strategy.  BN2S2 applies the arch35
    // column-private schedule when the shape is covered; the tiler rejects
    // any deterministic shape it cannot map, so there is no silent
    // non-deterministic fallback.
    fag_info.detSchedule = 0;
    if (deterministic && !is_varlen) {
        const int64_t bh = batch_size * num_heads_kv;
        const int64_t m = (q_seqlen + 127) / 128;
        const int64_t n = (kv_seqlen + 127) / 128;
        const int64_t group_num = num_heads / num_heads_kv;
        const fag_det_host::Selection sel =
            fag_det_host::SelectSchedule(is_causal, bh, m, n, group_num, static_cast<int64_t>(aic_num));
        if (sel.supported) {
            fag_info.detSchedule = static_cast<uint32_t>(FAGTiling950::DetSchedule::BN2S2);
        }
    }
    fag_info.batch = batch_size;
    fag_info.qSeqlen = q_seqlen;
    fag_info.qHeadNum = num_heads;
    fag_info.qkHeadDim = qk_head_dim;
    fag_info.kvSeqlen = kv_seqlen;
    fag_info.kvHeadNum = num_heads_kv;
    fag_info.vHeadDim = v_head_dim;
    fag_info.totalQ = is_varlen ? static_cast<uint64_t>(q_sizes[0]) : static_cast<uint64_t>(batch_size * q_seqlen);
    fag_info.totalKv = is_varlen ? static_cast<uint64_t>(k_sizes[0]) : static_cast<uint64_t>(batch_size * kv_seqlen);
    fag_info.aicNum = aic_num;
    fag_info.aivNum = aiv_num;

    // Pull TND cumulative lengths to host for ABI/shape validation.  The
    // arch35 tiler uses totalQ/totalKv from the actual packed tensors rather
    // than carrying host-only sequence vectors in its device tiling ABI.
    at::Tensor cu_q_cpu;
    at::Tensor cu_k_cpu;
    std::vector<int64_t> actual_seq_q;
    std::vector<int64_t> actual_seq_kv;
    if (is_varlen) {
        const at::Tensor& cu_q_tensor = cu_seqlens_q_.value();
        const at::Tensor& cu_k_tensor = cu_seqlens_k_.value();
        TORCH_CHECK(cu_q_tensor.dim() == 1 && cu_k_tensor.dim() == 1 && cu_q_tensor.size(0) == batch_size + 1 &&
                        cu_k_tensor.size(0) == batch_size + 1,
                    "Ascend950 v4 bwd: cu_seqlens must be 1D tensors of length B + 1");
        TORCH_CHECK(cu_q_tensor.dtype() == at::kInt && cu_k_tensor.dtype() == at::kInt,
                    "Ascend950 v4 bwd: cu_seqlens must have dtype int32");
        TORCH_CHECK(cu_q_tensor.device().type() == at::kPrivateUse1 && cu_k_tensor.device().type() == at::kPrivateUse1,
                    "Ascend950 v4 bwd: cu_seqlens must be on NPU");
        cu_q_cpu = cu_q_tensor.to(at::Device(at::kCPU)).to(at::kInt).contiguous();
        cu_k_cpu = cu_k_tensor.to(at::Device(at::kCPU)).to(at::kInt).contiguous();
        const int32_t* q_lengths = cu_q_cpu.data_ptr<int32_t>();
        const int32_t* kv_lengths = cu_k_cpu.data_ptr<int32_t>();
        TORCH_CHECK(q_lengths[0] == 0 && kv_lengths[0] == 0, "Ascend950 v4 bwd: cu_seqlens must start at zero");
        actual_seq_q.resize(batch_size);
        actual_seq_kv.resize(batch_size);
        bool tndAllNonEmpty = batch_size > 0;
        for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            TORCH_CHECK(q_lengths[batch_idx + 1] >= q_lengths[batch_idx] &&
                            kv_lengths[batch_idx + 1] >= kv_lengths[batch_idx],
                        "Ascend950 v4 bwd: cu_seqlens must be nondecreasing");
            TORCH_CHECK(q_lengths[batch_idx + 1] - q_lengths[batch_idx] <= q_seqlen &&
                            kv_lengths[batch_idx + 1] - kv_lengths[batch_idx] <= kv_seqlen,
                        "Ascend950 v4 bwd: an actual sequence length exceeds max_seqlen");
            actual_seq_q[batch_idx] = q_lengths[batch_idx + 1] - q_lengths[batch_idx];
            actual_seq_kv[batch_idx] = kv_lengths[batch_idx + 1] - kv_lengths[batch_idx];
            if (actual_seq_q[batch_idx] <= 0 || actual_seq_kv[batch_idx] <= 0) {
                tndAllNonEmpty = false;
            }
        }
        TORCH_CHECK(q_lengths[batch_size] == static_cast<int64_t>(fag_info.totalQ) &&
                        kv_lengths[batch_size] == static_cast<int64_t>(fag_info.totalKv),
                    "Ascend950 v4 bwd: final cu_seqlens values must match packed tensor lengths");
        at::Tensor used_q_cpu = seqused_q_.has_value()
            ? seqused_q_->to(at::Device(at::kCPU)).to(at::kInt).contiguous() : at::Tensor();
        at::Tensor used_k_cpu = seqused_k_.has_value()
            ? seqused_k_->to(at::Device(at::kCPU)).to(at::kInt).contiguous() : at::Tensor();
        const int32_t* used_q = used_q_cpu.defined() ? used_q_cpu.data_ptr<int32_t>() : nullptr;
        const int32_t* used_kv = used_k_cpu.defined() ? used_k_cpu.data_ptr<int32_t>() : nullptr;
        for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            if (used_q != nullptr) {
                TORCH_CHECK(used_q[batch_idx] >= 0 && used_q[batch_idx] <= actual_seq_q[batch_idx],
                            "Ascend950 v4 bwd: seqused_q must lie within the allocated sequence length");
            }
            if (used_kv != nullptr) {
                TORCH_CHECK(used_kv[batch_idx] >= 0 && used_kv[batch_idx] <= actual_seq_kv[batch_idx],
                            "Ascend950 v4 bwd: seqused_k must lie within the allocated sequence length");
            }
        }
        if (used_q != nullptr || used_kv != nullptr) {
            // Used-based trailing trim; totalQ/totalKv stay region totals.
            uint64_t tail_zero = 0;
            for (int64_t i = batch_size - 1; i >= 1; --i) {
                const bool q_empty = used_q != nullptr ? used_q[i] <= 0 : actual_seq_q[i] <= 0;
                const bool kv_empty = used_kv != nullptr ? used_kv[i] <= 0 : actual_seq_kv[i] <= 0;
                if (q_empty && kv_empty) {
                    ++tail_zero;
                } else {
                    break;
                }
            }
            fag_info.batch -= tail_zero;
        }
        // TND BN2S2: hand the per-batch lengths to the tiler, which
        // serializes the round/area prefix table.  MHA needs the swizzle's
        // intra-round uniqueness condition; GQA uses opst's flat partition.
        // Causal TND reuses the dense schedule with the causal mask applied
        // in the epilogue (masked blocks contribute exact zeros), same as
        // the rectangular BSND causal path.
        if (deterministic && tndAllNonEmpty &&
            batch_size + 1 <= static_cast<int64_t>(FAGTiling950::TND_SWIZZLE_PREFIX_NUM)) {
            fag_info.actualSeqQ = actual_seq_q.data();
            fag_info.actualSeqKv = actual_seq_kv.data();
            fag_info.detSchedule = static_cast<uint32_t>(FAGTiling950::DetSchedule::BN2S2);
        }
    }

    uint64_t ub_size = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    fag_info.ubSize = ub_size;

    // Deterministic shapes must map to the BN2S2 schedule; reject here so the
    // error is precise and the legacy VecDTM layout is never reached.
    TORCH_CHECK(!deterministic || fag_info.detSchedule == static_cast<uint32_t>(FAGTiling950::DetSchedule::BN2S2),
                "Ascend950 v4 bwd: deterministic backward requires the BN2S2 "
                "schedule for this shape");

    FAGTiling950::FAGTilingData fag_tiling_data{};
    const int64_t tiling_status = FAGTiling950::GetFAGTilingParam(fag_info, fag_tiling_data);
    TORCH_CHECK(tiling_status == 0, "Ascend950 v4 bwd: arch35 GetFAGTilingParam failed");
    // The old in-loop VecDTM deterministic fallback was replaced by the cube
    // BN2S2 schedule.  Any deterministic shape the tiler cannot map to BN2S2
    // must fail loudly instead of silently running a non-deterministic loop.
    if (deterministic) {
        TORCH_CHECK(fag_tiling_data.detSchedule == static_cast<uint32_t>(FAGTiling950::DetSchedule::BN2S2),
                    "Ascend950 v4 bwd: deterministic backward requires the BN2S2 "
                    "schedule for this shape");
    }
    at::Tensor tiling_cpu =
        at::empty({static_cast<int64_t>(sizeof(FAGTiling950::FAGTilingData))}, at::device(c10::kCPU).dtype(at::kByte));
    std::memcpy(tiling_cpu.data_ptr<uint8_t>(), &fag_tiling_data, sizeof(FAGTiling950::FAGTilingData));
    at::Tensor tiling_device = tiling_cpu.to(at::Device(at::kPrivateUse1));

    // ============================================================
    // FAG workspace (device)
    // ============================================================
    const uint64_t workspace_size = static_cast<uint64_t>(fag_tiling_data.workspaceSize);
    TORCH_CHECK(workspace_size > 0 && workspace_size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "Ascend950 v4 bwd: invalid workspace size from tiling");
    TORCH_CHECK(workspace_size % sizeof(float) == 0 && fag_tiling_data.deltaOffset % sizeof(float) == 0,
                "Ascend950 v4 bwd: FP32 workspace offsets must be aligned");
    if (deterministic) {
        if (fag_tiling_data.detPrivDkv != 0) {
            TORCH_CHECK(fag_tiling_data.dqOffset % FAGTiling950::GM_ALIGNMENT == 0 &&
                            fag_tiling_data.dkPrivOffset % FAGTiling950::GM_ALIGNMENT == 0 &&
                            fag_tiling_data.dvPrivOffset % FAGTiling950::GM_ALIGNMENT == 0 &&
                            fag_tiling_data.dvPrivOffset < workspace_size,
                        "Ascend950 v4 bwd: deterministic BN2S2 private workspace "
                        "offsets must be 512B-aligned and inside the workspace");
        } else {
            TORCH_CHECK(fag_tiling_data.dqOffset % FAGTiling950::GM_ALIGNMENT == 0 &&
                            fag_tiling_data.dkOffset % FAGTiling950::GM_ALIGNMENT == 0 &&
                            fag_tiling_data.dvOffset % FAGTiling950::GM_ALIGNMENT == 0 &&
                            fag_tiling_data.deltaOffset < workspace_size,
                        "Ascend950 v4 bwd: deterministic BN2S2 shared workspace "
                        "offsets must be 512B-aligned and inside the workspace");
        }
    }
    at::Tensor workspace = at::empty({static_cast<int64_t>(workspace_size / sizeof(float))},
                                     at::device(at::kPrivateUse1).dtype(at::kFloat));

    auto ptr = [](const at::Tensor& tensor) { return static_cast<uint8_t*>(tensor.data_ptr()); };
    // Arch35 FAG device ABI: cumulative lengths have B entries and do not
    // include the leading zero.
    at::Tensor cu_q_device;
    at::Tensor cu_k_device;
    uint8_t* cu_q = nullptr;
    uint8_t* cu_k = nullptr;
    if (is_varlen) {
        cu_q_device = cu_seqlens_q_.value().slice(0, 1, cu_seqlens_q_.value().size(0)).contiguous();
        cu_k_device = cu_seqlens_k_.value().slice(0, 1, cu_seqlens_k_.value().size(0)).contiguous();
        cu_q = ptr(cu_q_device);
        cu_k = ptr(cu_k_device);
    }
    uint8_t* seq_used_q = nullptr;
    uint8_t* seq_used_k = nullptr;
    if (seqused_q_.has_value()) {
        seq_used_q = ptr(seqused_q_.value());
    }
    if (seqused_k_.has_value()) {
        seq_used_k = ptr(seqused_k_.value());
    }

    at::Tensor mask_cpu_tensor;
    at::Tensor mask_npu_tensor;
    uint8_t* mask = nullptr;
    if (is_causal) {
        mask_cpu_tensor =
            at::triu(at::ones({256, 256}, at::device(c10::kCPU).dtype(at::kByte)), 1).to(at::Device(at::kPrivateUse1));
        mask_npu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
        mask = ptr(mask_npu_tensor);
    }

    // Flush queued PyTorch NPU work (for example tiling, mask and cumulative
    // lengths) before launching the raw mixed AIC/AIV kernel on the ACL stream.
    stream = c10_npu::getCurrentNPUStream().stream(true);

    // The deterministic BN2S2 tiling trims idle cores for small shapes
    // (fag_tiling.cpp): every launched core pays the deterministic prologue
    // and the per-round grid barrier, so launch exactly what the tiling asks
    // for.  Non-deterministic paths keep usedCoreNum == aic_num.
    const uint32_t launch_cores = (fag_tiling_data.usedCoreNum > 0 && fag_tiling_data.usedCoreNum <= aic_num)
                                      ? fag_tiling_data.usedCoreNum
                                      : aic_num;

#define LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, IS_CAUSAL, IS_DETERMINISTIC, IS_SOFTCAP)                                    \
    FlashAttentionV3Bwd950<DTYPE, FAGTiling950::Layout::INPUT_LAYOUT, IS_CAUSAL, IS_DETERMINISTIC, IS_SOFTCAP>         \
        <<<launch_cores, nullptr, stream>>>(ptr(dout), ptr(q), ptr(k), ptr(v), ptr(out), mask, ptr(softmax_lse), cu_q, \
                                            cu_k, seq_used_q, seq_used_k, ptr(dq), ptr(dk), ptr(dv), ptr(workspace), \
                                            ptr(tiling_device))

#define DISPATCH_BWD950_FLAGS(DTYPE, INPUT_LAYOUT)                                                                     \
    do {                                                                                                               \
        if (is_causal) {                                                                                               \
            if (deterministic) {                                                                                       \
                if (has_softcap) {                                                                                     \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, true, true, true);                                              \
                } else {                                                                                               \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, true, true, false);                                             \
                }                                                                                                      \
            } else {                                                                                                   \
                if (has_softcap) {                                                                                     \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, true, false, true);                                             \
                } else {                                                                                               \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, true, false, false);                                            \
                }                                                                                                      \
            }                                                                                                          \
        } else {                                                                                                       \
            if (deterministic) {                                                                                       \
                if (has_softcap) {                                                                                     \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, false, true, true);                                             \
                } else {                                                                                               \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, false, true, false);                                            \
                }                                                                                                      \
            } else {                                                                                                   \
                if (has_softcap) {                                                                                     \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, false, false, true);                                            \
                } else {                                                                                               \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, false, false, false);                                           \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
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

    at::Tensor softmax_d = is_varlen ? at::empty({num_heads, q.size(0)}, q.options().dtype(at::kFloat))
                                     : at::empty({batch_size, num_heads, q_seqlen}, q.options().dtype(at::kFloat));

    return {dq, dk, dv, softmax_d};
}
