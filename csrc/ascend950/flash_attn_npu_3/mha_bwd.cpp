// Copyright (c) 2026, Minghua Shen.

#include "../attention_window.h"
// Ascend950 FlashAttention v3 backward scaffold.
// The host ABI matches the Python v3 backward entry. The device kernel is a
// no-op placeholder; replace the implementation in fag_kernel.cpp when the
// real backward algorithm lands.

#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include <torch/extension.h>

#include "acl/acl.h"
#include "fa_bwd_prepare_host.hpp"
#include "device_triu_mask.h"
#include "fag_tiling.cpp"
#include "fag_kernel.cpp"
#include "fag_bwd_d512_partial.hpp"
#include "fag_bwd_d512_partial_host.hpp"
#include "fa_bwd_mla_kernel.cpp"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

std::vector<at::Tensor> mha_bwd(at::Tensor dout, at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out,
                                at::Tensor softmax_lse, std::optional<at::Tensor> dq_, std::optional<at::Tensor> dk_,
                                std::optional<at::Tensor> dv_, std::optional<at::Tensor> cu_seqlens_q_,
                                std::optional<at::Tensor> cu_seqlens_k_, std::optional<at::Tensor> seqused_q_,
                                std::optional<at::Tensor> seqused_k_, std::optional<int64_t> max_seqlen_q_,
                                std::optional<int64_t> max_seqlen_k_, std::optional<double> softmax_scale_,
                                bool is_causal, int64_t window_size_left, int64_t window_size_right, double softcap,
                                bool deterministic, int64_t sm_margin, int64_t layout)
{
    TORCH_CHECK(q.device().type() == at::kPrivateUse1, "Ascend950 v3 bwd: q must be on NPU");
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const bool is_varlen = cu_seqlens_q_.has_value();
    TORCH_CHECK(layout >= 0 && layout <= 2, "mha_bwd: invalid layout");
    const bool is_bnsd = layout == 2;
    TORCH_CHECK(layout != 0 || is_varlen, "mha_bwd: TND requires cu_seqlens_q/k");
    TORCH_CHECK(!is_bnsd ||
                    (!is_varlen && !cu_seqlens_k_.has_value() && !seqused_q_.has_value() && !seqused_k_.has_value()),
                "mha_bwd: BNSD does not support cu_seqlens or seqused");

    const uint32_t aic_num = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    const uint32_t aiv_num = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();

    TORCH_CHECK(is_varlen == cu_seqlens_k_.has_value(),
                "Ascend950 v3 bwd: cu_seqlens_q and cu_seqlens_k must be provided together");
    // seqused caps used lengths only; offsets and sizing still follow cu.
    TORCH_CHECK(!seqused_q_.has_value() || is_varlen, "Ascend950 v3 bwd: seqused_q requires cu_seqlens_q");
    TORCH_CHECK(!seqused_k_.has_value() || is_varlen, "Ascend950 v3 bwd: seqused_k requires cu_seqlens_q");
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0.0 && softcap <= std::numeric_limits<float>::max(),
                "Ascend950 v3 bwd: softcap must be finite and non-negative");
    TORCH_CHECK(sm_margin == 0, "Ascend950 v3 bwd does not support sm_margin");
    TORCH_CHECK(q.dtype() == at::kHalf || q.dtype() == at::kBFloat16, "Ascend950 v3 bwd only supports FP16 and BF16");
    TORCH_CHECK(k.dtype() == q.dtype() && v.dtype() == q.dtype() && dout.dtype() == q.dtype() &&
                    out.dtype() == q.dtype(),
                "Ascend950 v3 bwd: q/k/v/out/dout must have the same dtype");
    TORCH_CHECK(q.dim() == (is_varlen ? 3 : 4), "Ascend950 v3 bwd: q must use TND or BSND layout");
    TORCH_CHECK(k.dim() == q.dim() && v.dim() == q.dim() && dout.dim() == q.dim() && out.dim() == q.dim(),
                "Ascend950 v3 bwd: q/k/v/out/dout must use the same layout rank");
    TORCH_CHECK(!is_varlen || (max_seqlen_q_.has_value() && max_seqlen_k_.has_value()),
                "Ascend950 v3 bwd: max sequence lengths are required for TND");

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

    const bool is_mla = (qk_head_dim == 192 && v_head_dim == 128);
    TORCH_CHECK(!is_mla || !deterministic,
                "Ascend950 v3 bwd: deterministic backward is not supported for Q/K=192 and V=128");

    TORCH_CHECK(batch_size > 0, "Ascend950 v3 bwd: batch size must be positive");
    if (seqused_q_.has_value()) {
        const at::Tensor& used_q = seqused_q_.value();
        TORCH_CHECK(used_q.dtype() == at::kInt, "Ascend950 v3 bwd: seqused_q must have dtype int32");
        TORCH_CHECK(used_q.dim() == 1 && used_q.size(0) == batch_size,
                    "Ascend950 v3 bwd: seqused_q must be 1D of length B");
        TORCH_CHECK(used_q.is_contiguous(), "Ascend950 v3 bwd: seqused_q must be contiguous");
        TORCH_CHECK(used_q.device() == q.device(), "Ascend950 v3 bwd: seqused_q must be on NPU");
    }
    if (seqused_k_.has_value()) {
        const at::Tensor& used_k = seqused_k_.value();
        TORCH_CHECK(used_k.dtype() == at::kInt, "Ascend950 v3 bwd: seqused_k must have dtype int32");
        TORCH_CHECK(used_k.dim() == 1 && used_k.size(0) == batch_size,
                    "Ascend950 v3 bwd: seqused_k must be 1D of length B");
        TORCH_CHECK(used_k.is_contiguous(), "Ascend950 v3 bwd: seqused_k must be contiguous");
        TORCH_CHECK(used_k.device() == q.device(), "Ascend950 v3 bwd: seqused_k must be on NPU");
    }
    TORCH_CHECK(!deterministic || (!seqused_q_.has_value() && !seqused_k_.has_value()),
                "Ascend950 v3 bwd: seqused_q/seqused_k with deterministic is not supported yet");
    TORCH_CHECK(q_seqlen > 0 && kv_seqlen > 0, "Ascend950 v3 bwd: sequence lengths must be positive");
    TORCH_CHECK(num_heads > 0 && num_heads_kv > 0 && num_heads % num_heads_kv == 0,
                "Ascend950 v3 bwd: KV heads must divide query heads");
    TORCH_CHECK(qk_head_dim == k_head_dim, "mha_bwd: query and key must have the same head dimension");
    TORCH_CHECK(qk_head_dim > 0 && qk_head_dim <= 512, "Ascend950 v3 bwd: q/k head dimension must be in (0, 512]");
    TORCH_CHECK(v_head_dim > 0 && v_head_dim <= 512, "Ascend950 v3 bwd: v head dimension must be in (0, 512]");
    TORCH_CHECK(!deterministic || (qk_head_dim <= 256 && v_head_dim <= 256),
                "Ascend950 v3 bwd: deterministic with head dimension > 256 is not supported yet");
    TORCH_CHECK(dout.size(-1) == v_head_dim && out.size(-1) == v_head_dim,
                "Ascend950 v3 bwd: dout/out head dimension must match v");

    at::Tensor dq = dq_.has_value() ? dq_.value() : at::empty(q.sizes(), q.options());
    at::Tensor dk = dk_.has_value() ? dk_.value() : at::empty(k.sizes(), k.options());
    at::Tensor dv = dv_.has_value() ? dv_.value() : at::empty(v.sizes(), v.options());
    if (is_bnsd) {
        TORCH_CHECK(k.sizes().slice(0, 3) == v.sizes().slice(0, 3), "mha_bwd: BNSD k/v must share logical B/S/N shape");
        TORCH_CHECK(q.size(0) == k.size(0) && q.sizes().slice(0, 3) == out.sizes().slice(0, 3) &&
                        out.sizes() == dout.sizes(),
                    "mha_bwd: invalid BNSD batch or out/dout shape");
        TORCH_CHECK(dq.sizes() == q.sizes() && dk.sizes() == k.sizes() && dv.sizes() == v.sizes(),
                    "mha_bwd: dq/dk/dv must match q/k/v shapes");
        TORCH_CHECK(dq.dtype() == q.dtype() && dk.dtype() == q.dtype() && dv.dtype() == q.dtype(),
                    "mha_bwd: dq/dk/dv must match q dtype");
        TORCH_CHECK(softmax_lse.dim() == 3 && softmax_lse.size(0) == batch_size && softmax_lse.size(1) == num_heads &&
                        softmax_lse.size(2) == q_seqlen && softmax_lse.scalar_type() == at::kFloat &&
                        softmax_lse.device() == q.device(),
                    "mha_bwd: BNSD LSE must be float32 [B,Nq,Sq] on q device");
        softmax_lse = softmax_lse.contiguous();
    }

    if (is_varlen) {
        TORCH_CHECK(v.size(1) == num_heads_kv && out.size(0) == q.size(0) && out.size(1) == num_heads &&
                        dout.sizes() == out.sizes(),
                    "mha_bwd: invalid packed k/v or out/dout shape");
        TORCH_CHECK(dq.sizes() == q.sizes() && dk.sizes() == k.sizes() && dv.sizes() == v.sizes() &&
                        dq.dtype() == q.dtype() && dk.dtype() == q.dtype() && dv.dtype() == q.dtype(),
                    "mha_bwd: gradients must match q/k/v shapes and dtypes");
        TORCH_CHECK(softmax_lse.dim() == 2 && softmax_lse.size(0) == num_heads && softmax_lse.size(1) == q.size(0) &&
                        softmax_lse.scalar_type() == at::kFloat && softmax_lse.device() == q.device(),
                    "mha_bwd: TND LSE must be float32 [Nq,Tq] on q device");
        softmax_lse = softmax_lse.contiguous();
    }

    // Static shape, stride and layout validation.
    for (const auto& tensor : {q, k, v, out, dout}) {
        TORCH_CHECK(tensor.stride(-1) == 1 && tensor.device() == q.device(),
                    "mha_bwd: inputs require contiguous D and the same device");
    }
    for (const auto& tensor : {dq, dk, dv}) {
        TORCH_CHECK(tensor.is_contiguous() && tensor.device() == q.device(),
                    "mha_bwd: gradients must be contiguous BSND/TND on q device");
    }

    FAGTiling950::FAGInfo fag_info{};
    fag_info.scaleValue =
        static_cast<float>(softmax_scale_.value_or(1.0 / std::sqrt(static_cast<double>(qk_head_dim))));
    const bool has_softcap = softcap > 0.0;
    fag_info.softcapValue = static_cast<float>(softcap);
    fag_info.layout =
        is_varlen ? FAGTiling950::Layout::TND : (is_bnsd ? FAGTiling950::Layout::BNSD : FAGTiling950::Layout::BSND);
    bool is_local = false;
    fa_window::NormalizeAttentionWindow(is_causal, q_seqlen, kv_seqlen, window_size_left, window_size_right, is_causal,
                                        is_local);
    const bool has_attn_mask = is_causal || is_local;
    if (is_causal) {
        fag_info.maskType = FAGTiling950::MaskType::CAUSAL;
        fag_info.window_size_left = window_size_left;
        fag_info.window_size_right = 0;
    } else if (is_local) {
        fag_info.maskType = FAGTiling950::MaskType::BAND;
        fag_info.window_size_left = window_size_left;
        fag_info.window_size_right = window_size_right;
    } else {
        fag_info.maskType = FAGTiling950::MaskType::NO_MASK;
        fag_info.window_size_left = window_size_left;
        fag_info.window_size_right = window_size_right;
    }
    fag_info.deterministic = deterministic ? 1U : 0U;
    // Deterministic accumulation strategy.  BN2S2 applies the opst arch35
    // column-private schedule when the shape is covered; otherwise fall back
    // to the per-round det-slot + streaming VecDTM reduction.
    fag_info.detSchedule = 0;
    if (deterministic && !is_varlen) {
        const int64_t bh = batch_size * num_heads_kv;
        const uint32_t seqTile = 128;
        const int64_t m = (q_seqlen + seqTile - 1) / seqTile;
        const int64_t n = (kv_seqlen + seqTile - 1) / seqTile;
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
    // K and V may have different unused packed tails. The common workspace
    // extent must fit both gradient outputs; longer tails are zeroed below.
    fag_info.totalKv = is_varlen ? static_cast<uint64_t>(std::min(k_sizes[0], v_sizes[0]))
                                 : static_cast<uint64_t>(batch_size * kv_seqlen);
    fag_info.aicNum = aic_num;
    fag_info.aivNum = aiv_num;

    at::Tensor cu_q_device;
    at::Tensor cu_k_device;
    if (is_varlen) {
        const at::Tensor& cu_q_tensor = cu_seqlens_q_.value();
        const at::Tensor& cu_k_tensor = cu_seqlens_k_.value();
        TORCH_CHECK(cu_q_tensor.dim() == 1 && cu_k_tensor.dim() == 1 && cu_k_tensor.size(0) == batch_size + 1,
                    "Ascend950 v3 bwd: cu_seqlens must be 1D tensors of length B + 1");
        TORCH_CHECK(cu_q_tensor.scalar_type() == at::kInt && cu_k_tensor.scalar_type() == at::kInt,
                    "Ascend950 v3 bwd: cu_seqlens must have dtype int32");
        TORCH_CHECK(cu_q_tensor.device() == q.device() && cu_k_tensor.device() == q.device(),
                    "Ascend950 v3 bwd: cu_seqlens must be on the query device");
        cu_q_device = cu_q_tensor.contiguous();
        cu_k_device = cu_k_tensor.contiguous();
    }

    uint64_t ub_size = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    fag_info.ubSize = ub_size;

    for (const auto& tensor : {q, k, v, out, dout}) {
        TORCH_CHECK(tensor.stride(is_varlen ? 0 : 1) >= tensor.size(-1), "mha_bwd: token rows must not overlap");
    }
    const auto tensor_strides = [is_varlen](const at::Tensor& tensor) {
        return FagTensorStrides{is_varlen ? 0 : tensor.stride(0), is_varlen ? tensor.stride(0) : tensor.stride(1),
                                is_varlen ? tensor.stride(1) : tensor.stride(2)};
    };
    const FagInputStrides input_strides{tensor_strides(q),    tensor_strides(k),   tensor_strides(v),
                                        tensor_strides(dout), tensor_strides(out), 1};
    FAGTiling950::FAGTilingData fag_tiling_data{};
    at::Tensor tiling_device;
    uint64_t workspace_size = 0;
    if (is_varlen) {
        workspace_size = fag_bwd_host::TndWorkspaceCapacity(fag_info);
    } else {
        // Dense tiling consumes only static tensor geometry.
        TORCH_CHECK(!deterministic || fag_info.detSchedule == static_cast<uint32_t>(FAGTiling950::DetSchedule::BN2S2),
                    "Ascend950 v3 bwd: deterministic backward requires the BN2S2 schedule for this shape");
        TORCH_CHECK(FAGTiling950::GetFAGTilingParam(fag_info, fag_tiling_data) == 0,
                    "Ascend950 v3 bwd: arch35 GetFAGTilingParam failed");
        if (is_local && fag_tiling_data.detPrivDkv != 0 && kv_seqlen - q_seqlen > fag_tiling_data.s1Token) {
            dk.zero_();
            dv.zero_();
        }
        if (is_local && fag_tiling_data.dqPostAbsorb != 0) {
            dq.zero_();
        }
        fag_tiling_data.inputStrides = input_strides;
        auto tiling_cpu = at::empty({static_cast<int64_t>(sizeof(FAGTiling950::FAGTilingData))},
                                    at::device(c10::kCPU).dtype(at::kByte));
        std::memcpy(tiling_cpu.data_ptr<uint8_t>(), &fag_tiling_data, sizeof(fag_tiling_data));
        tiling_device = tiling_cpu.to(q.device());
        workspace_size = fag_tiling_data.workspaceSize;
        TORCH_CHECK(workspace_size > 0 && workspace_size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                    "Ascend950 v3 bwd: invalid workspace size from tiling");
    }
    TORCH_CHECK(workspace_size % sizeof(float) == 0, "Ascend950 bwd: FP32 workspace size must be aligned");
    at::Tensor workspace = at::empty({static_cast<int64_t>(workspace_size / sizeof(float))},
                                     at::device(at::kPrivateUse1).dtype(at::kFloat));

    auto ptr = [](const at::Tensor& tensor) { return static_cast<uint8_t*>(tensor.data_ptr()); };
    // The kernels consume the B entries following the leading zero.
    uint8_t* cu_q = is_varlen ? ptr(cu_q_device) + sizeof(int32_t) : nullptr;
    uint8_t* cu_k = is_varlen ? ptr(cu_k_device) + sizeof(int32_t) : nullptr;
    uint8_t* seq_used_q = nullptr;
    uint8_t* seq_used_k = nullptr;
    if (seqused_q_.has_value()) {
        seq_used_q = ptr(seqused_q_.value());
    }
    if (seqused_k_.has_value()) {
        seq_used_k = ptr(seqused_k_.value());
    }

    // Causal/band: cached on-device 256x256 compressed triu (epilogue RowMajor).
    at::Tensor mask_npu_tensor;
    uint8_t* mask = nullptr;
    if (has_attn_mask) {
        mask_npu_tensor = MakeDeviceTriuMask(/*dim=*/256);
        mask = ptr(mask_npu_tensor);
    }

    if (is_varlen) {
        tiling_device = fag_bwd_host::PrepareTnd(fag_info, input_strides, cu_q_device, cu_k_device, seqused_q_,
                                                 seqused_k_, k_sizes[0], v_sizes[0], workspace_size,
                                                 /*exactPackedLength=*/false);
        stream = fag_bwd_host::ClearTndKvOutputs(fag_info, /*exactPackedLength=*/false, tiling_device, dk, dv);
    } else {
        // Flush queued PyTorch NPU work before the raw kernel launch.
        stream = c10_npu::getCurrentNPUStream().stream(true);
    }

    // The deterministic BN2S2 tiling trims idle cores for small shapes
    // (fag_tiling.cpp): every launched core pays the deterministic prologue
    // and the per-round grid barrier, so launch exactly what the tiling asks
    // for.  Non-deterministic paths keep usedCoreNum == aic_num.
    const uint32_t launch_cores = (fag_tiling_data.usedCoreNum > 0 && fag_tiling_data.usedCoreNum <= aic_num)
                                      ? fag_tiling_data.usedCoreNum
                                      : aic_num;

    // shortQ/shortBiDir: SINGLE_VEC=true → NO_SPLIT Fixpipe + AIV0-only handshake.
    // All other shapes (neither flag) stay on SPLIT_M dual-AIV.
    const bool launch_single_vec = is_varlen ? (!deterministic && q_seqlen <= 16)
                                             : (fag_tiling_data.shortQ != 0 || fag_tiling_data.shortBiDir != 0);

#define LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, IS_ATTEN_MASK, IS_DETERMINISTIC, IS_SOFTCAP, SINGLE_VEC)                    \
    FlashAttentionV3Bwd950<DTYPE, FAGTiling950::Layout::INPUT_LAYOUT, IS_ATTEN_MASK, IS_DETERMINISTIC, IS_SOFTCAP,     \
                           SINGLE_VEC><<<launch_cores, nullptr, stream>>>(                                             \
        ptr(dout), ptr(q), ptr(k), ptr(v), ptr(out), mask, ptr(softmax_lse), cu_q, cu_k, seq_used_q, seq_used_k,       \
        ptr(dq), ptr(dk), ptr(dv), ptr(workspace), ptr(tiling_device))

#define DISPATCH_BWD950_FLAGS_SV(DTYPE, INPUT_LAYOUT, SINGLE_VEC)                                                      \
    do {                                                                                                               \
        if (has_attn_mask) {                                                                                           \
            if (deterministic) {                                                                                       \
                if (has_softcap) {                                                                                     \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, true, true, true, SINGLE_VEC);                                  \
                } else {                                                                                               \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, true, true, false, SINGLE_VEC);                                 \
                }                                                                                                      \
            } else {                                                                                                   \
                if (has_softcap) {                                                                                     \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, true, false, true, SINGLE_VEC);                                 \
                } else {                                                                                               \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, true, false, false, SINGLE_VEC);                                \
                }                                                                                                      \
            }                                                                                                          \
        } else {                                                                                                       \
            if (deterministic) {                                                                                       \
                if (has_softcap) {                                                                                     \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, false, true, true, SINGLE_VEC);                                 \
                } else {                                                                                               \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, false, true, false, SINGLE_VEC);                                \
                }                                                                                                      \
            } else {                                                                                                   \
                if (has_softcap) {                                                                                     \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, false, false, true, SINGLE_VEC);                                \
                } else {                                                                                               \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, false, false, false, SINGLE_VEC);                               \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

#define DISPATCH_BWD950_FLAGS(DTYPE, INPUT_LAYOUT)                                                                     \
    do {                                                                                                               \
        if (launch_single_vec) {                                                                                       \
            DISPATCH_BWD950_FLAGS_SV(DTYPE, INPUT_LAYOUT, true);                                                       \
        } else {                                                                                                       \
            DISPATCH_BWD950_FLAGS_SV(DTYPE, INPUT_LAYOUT, false);                                                      \
        }                                                                                                              \
    } while (0)

#define LAUNCH_BWD950_MLA(DTYPE, INPUT_LAYOUT, IS_ATTEN_MASK, IS_SOFTCAP)                                              \
    FlashAttentionV3BwdMla950<DTYPE, FAGTiling950::Layout::INPUT_LAYOUT, IS_ATTEN_MASK, IS_SOFTCAP>                    \
        <<<launch_cores, nullptr, stream>>>(ptr(dout), ptr(q), ptr(k), ptr(v), ptr(out), mask, ptr(softmax_lse), cu_q, \
                                            cu_k, seq_used_q, seq_used_k, ptr(dq), ptr(dk), ptr(dv), ptr(workspace),   \
                                            ptr(tiling_device))

#define DISPATCH_BWD950_MLA_FLAGS(DTYPE, INPUT_LAYOUT)                                                                 \
    do {                                                                                                               \
        if (has_attn_mask) {                                                                                           \
            if (has_softcap) {                                                                                         \
                LAUNCH_BWD950_MLA(DTYPE, INPUT_LAYOUT, true, true);                                                    \
            } else {                                                                                                   \
                LAUNCH_BWD950_MLA(DTYPE, INPUT_LAYOUT, true, false);                                                   \
            }                                                                                                          \
        } else {                                                                                                       \
            if (has_softcap) {                                                                                         \
                LAUNCH_BWD950_MLA(DTYPE, INPUT_LAYOUT, false, true);                                                   \
            } else {                                                                                                   \
                LAUNCH_BWD950_MLA(DTYPE, INPUT_LAYOUT, false, false);                                                  \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

    if (qk_head_dim > 256 || v_head_dim > 256) {
        TORCH_CHECK(!deterministic, "Ascend950 v3 bwd: d512Partial requires non-deterministic path");
        launch_fag_d512_partial(fag_d512::MakeD512PartialArgs(
            stream, launch_cores, dout, q, k, v, out, mask, softmax_lse, cu_q, cu_k, seq_used_q, seq_used_k, dq, dk, dv,
            workspace, tiling_device, q.dtype() == at::kBFloat16,
            is_varlen ? static_cast<uint32_t>(FAGTiling950::Layout::TND)
                      : (is_bnsd ? static_cast<uint32_t>(FAGTiling950::Layout::BNSD)
                                 : static_cast<uint32_t>(FAGTiling950::Layout::BSND)),
            has_attn_mask, has_softcap));
    } else if (is_mla) {
        if (q.dtype() == at::kBFloat16) {
            if (is_varlen) {
                DISPATCH_BWD950_MLA_FLAGS(bfloat16_t, TND);
            } else if (is_bnsd) {
                DISPATCH_BWD950_MLA_FLAGS(bfloat16_t, BNSD);
            } else {
                DISPATCH_BWD950_MLA_FLAGS(bfloat16_t, BSND);
            }
        } else {
            if (is_varlen) {
                DISPATCH_BWD950_MLA_FLAGS(half, TND);
            } else if (is_bnsd) {
                DISPATCH_BWD950_MLA_FLAGS(half, BNSD);
            } else {
                DISPATCH_BWD950_MLA_FLAGS(half, BSND);
            }
        }
    } else if (q.dtype() == at::kBFloat16) {
        if (is_varlen) {
            DISPATCH_BWD950_FLAGS(bfloat16_t, TND);
        } else if (is_bnsd) {
            DISPATCH_BWD950_FLAGS(bfloat16_t, BNSD);
        } else {
            DISPATCH_BWD950_FLAGS(bfloat16_t, BSND);
        }
    } else {
        if (is_varlen) {
            DISPATCH_BWD950_FLAGS(half, TND);
        } else if (is_bnsd) {
            DISPATCH_BWD950_FLAGS(half, BNSD);
        } else {
            DISPATCH_BWD950_FLAGS(half, BSND);
        }
    }

#undef DISPATCH_BWD950_FLAGS
#undef DISPATCH_BWD950_FLAGS_SV
#undef LAUNCH_BWD950
#undef DISPATCH_BWD950_MLA_FLAGS
#undef LAUNCH_BWD950_MLA

    at::Tensor softmax_d = is_varlen ? at::empty({num_heads, q.size(0)}, q.options().dtype(at::kFloat))
                                     : at::empty({batch_size, num_heads, q_seqlen}, q.options().dtype(at::kFloat));

    return {dq, dk, dv, softmax_d};
}
