/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend 950 FlashAttention v4 MLA backward — independent host wrapper.
 *
 * This operator specialises the dense FlashAttention backward path for
 * MLA-style shapes where Q/K head dim = 192 and V head dim = 128. It
 * supports BSND/TND/BNSD, causal, sliding-window, softcap and GQA/MQA,
 * with non-deterministic gradient reduction, and lives in a separate
 * source file so the MLA memory layout and pipeline can evolve independently.
 *
 * The device kernel is reused from flash_attn_npu_3 FAG 950, matching the
 * reuse pattern already used by the generic mha_bwd.cpp.
 */

#include "../attention_window.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "../flash_attn_npu_3/fag_tiling.cpp"
#include "../flash_attn_npu_3/fag_kernel.cpp"
#include "fa_bwd_mla_kernel.cpp"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {

constexpr int64_t kMlaQkHeadDim = 192;
constexpr int64_t kMlaVHeadDim = 128;

} // namespace

inline at::Tensor MakeMlaBwdTriuMask()
{
    const auto idx = c10_npu::getCurrentNPUStream().device_index();
    static thread_local std::unordered_map<c10::DeviceIndex, at::Tensor> masks;
    auto found = masks.find(idx);
    if (found != masks.end()) {
        return found->second;
    }
    c10_npu::NPUGuard guard(idx);
    at::Tensor mask =
        at::ones({256, 256}, at::TensorOptions().dtype(at::kByte).device(at::Device(at::kPrivateUse1, idx)));
    mask.triu_(/*diagonal=*/1);
    c10_npu::getCurrentNPUStream().synchronize();
    return masks.emplace(idx, std::move(mask)).first->second;
}

std::vector<at::Tensor> fa_bwd_mla(at::Tensor dout, at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out,
                                   at::Tensor softmax_lse, std::optional<at::Tensor> dq_, std::optional<at::Tensor> dk_,
                                   std::optional<at::Tensor> dv_, std::optional<at::Tensor> cu_seqlens_q_,
                                   std::optional<at::Tensor> cu_seqlens_k_, std::optional<at::Tensor> seqused_q_,
                                   std::optional<at::Tensor> seqused_k_, std::optional<int64_t> max_seqlen_q_,
                                   std::optional<int64_t> max_seqlen_k_, std::optional<double> softmax_scale_,
                                   bool is_causal, int64_t window_size_left, int64_t window_size_right, double softcap,
                                   bool deterministic, int64_t sm_margin, int64_t layout)
{

    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const bool is_varlen = cu_seqlens_q_.has_value();
    TORCH_CHECK(layout >= 0 && layout <= 2, "mha_bwd_mla: invalid layout");
    const bool is_bnsd = layout == 2;
    TORCH_CHECK(layout != 0 || is_varlen, "mha_bwd_mla: TND requires cu_seqlens_q/k");
    TORCH_CHECK(!is_bnsd ||
                    (!is_varlen && !cu_seqlens_k_.has_value() && !seqused_q_.has_value() && !seqused_k_.has_value()),
                "mha_bwd_mla: BNSD does not support cu_seqlens or seqused");

    const uint32_t aic_num = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    const uint32_t aiv_num = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();

    TORCH_CHECK(is_varlen == cu_seqlens_k_.has_value(),
                "Ascend950 v4 mla bwd: cu_seqlens_q and cu_seqlens_k must be provided together");
    TORCH_CHECK(!seqused_q_.has_value() || is_varlen, "Ascend950 v4 mla bwd: seqused_q requires cu_seqlens_q");
    TORCH_CHECK(!seqused_k_.has_value() || is_varlen, "Ascend950 v4 mla bwd: seqused_k requires cu_seqlens_q");
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0.0 && softcap <= std::numeric_limits<float>::max(),
                "Ascend950 v4 mla bwd: softcap must be finite and non-negative");
    TORCH_CHECK(sm_margin == 0, "Ascend950 v4 mla bwd does not support sm_margin");
    TORCH_CHECK(q.dtype() == at::kHalf || q.dtype() == at::kBFloat16,
                "Ascend950 v4 mla bwd only supports FP16 and BF16");
    TORCH_CHECK(k.dtype() == q.dtype() && v.dtype() == q.dtype() && dout.dtype() == q.dtype() &&
                    out.dtype() == q.dtype(),
                "Ascend950 v4 mla bwd: q/k/v/out/dout must have the same dtype");
    TORCH_CHECK(q.dim() == (is_varlen ? 3 : 4), "Ascend950 v4 mla bwd: q must use TND or BSND layout");
    TORCH_CHECK(k.dim() == q.dim() && v.dim() == q.dim() && dout.dim() == q.dim() && out.dim() == q.dim(),
                "Ascend950 v4 mla bwd: q/k/v/out/dout must use the same layout rank");
    TORCH_CHECK(!is_varlen || (max_seqlen_q_.has_value() && max_seqlen_k_.has_value()),
                "Ascend950 v4 mla bwd: max sequence lengths are required for TND");

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

    TORCH_CHECK(batch_size > 0, "Ascend950 v4 mla bwd: batch size must be positive");
    if (seqused_q_.has_value()) {
        const at::Tensor& used_q = seqused_q_.value();
        TORCH_CHECK(used_q.dtype() == at::kInt, "Ascend950 v4 mla bwd: seqused_q must have dtype int32");
        TORCH_CHECK(used_q.dim() == 1 && used_q.size(0) == batch_size,
                    "Ascend950 v4 mla bwd: seqused_q must be 1D of length B");
        TORCH_CHECK(used_q.is_contiguous(), "Ascend950 v4 mla bwd: seqused_q must be contiguous");
        TORCH_CHECK(used_q.device().type() == at::kPrivateUse1, "Ascend950 v4 mla bwd: seqused_q must be on NPU");
    }
    if (seqused_k_.has_value()) {
        const at::Tensor& used_k = seqused_k_.value();
        TORCH_CHECK(used_k.dtype() == at::kInt, "Ascend950 v4 mla bwd: seqused_k must have dtype int32");
        TORCH_CHECK(used_k.dim() == 1 && used_k.size(0) == batch_size,
                    "Ascend950 v4 mla bwd: seqused_k must be 1D of length B");
        TORCH_CHECK(used_k.is_contiguous(), "Ascend950 v4 mla bwd: seqused_k must be contiguous");
        TORCH_CHECK(used_k.device().type() == at::kPrivateUse1, "Ascend950 v4 mla bwd: seqused_k must be on NPU");
    }
    TORCH_CHECK(!deterministic || (!seqused_q_.has_value() && !seqused_k_.has_value()),
                "Ascend950 v4 mla bwd: seqused_q/seqused_k with deterministic is not supported yet");
    TORCH_CHECK(q_seqlen > 0 && kv_seqlen > 0, "Ascend950 v4 mla bwd: sequence lengths must be positive");
    TORCH_CHECK(num_heads > 0 && num_heads_kv > 0 && num_heads % num_heads_kv == 0,
                "Ascend950 v4 mla bwd: KV heads must divide query heads");
    TORCH_CHECK(qk_head_dim == k_head_dim, "Ascend950 v4 mla bwd: q and k must share the same head dimension");

    // MLA shape gate.  The wrapper is specialised for 192/128; other shapes
    // should fall back to the generic v4 backward.
    TORCH_CHECK(qk_head_dim == kMlaQkHeadDim && v_head_dim == kMlaVHeadDim,
                "Ascend950 v4 mla bwd: this entry only supports Q/K head dim = ", kMlaQkHeadDim,
                " and V head dim = ", kMlaVHeadDim, "; got qk=", qk_head_dim, " v=", v_head_dim);

    TORCH_CHECK(dout.size(-1) == v_head_dim && out.size(-1) == v_head_dim,
                "Ascend950 v4 mla bwd: dout/out head dimension must match v");

    at::Tensor dq = dq_.has_value() ? dq_.value() : at::empty(q.sizes(), q.options());
    at::Tensor dk = dk_.has_value() ? dk_.value() : at::empty(k.sizes(), k.options());
    at::Tensor dv = dv_.has_value() ? dv_.value() : at::empty(v.sizes(), v.options());
    if (is_bnsd) {
        TORCH_CHECK(k.sizes().slice(0, 3) == v.sizes().slice(0, 3),
                    "mha_bwd_mla: BNSD k/v must share logical B/S/N shape");
        TORCH_CHECK(q.size(0) == k.size(0) && q.sizes().slice(0, 3) == out.sizes().slice(0, 3) &&
                        out.sizes() == dout.sizes(),
                    "mha_bwd_mla: invalid BNSD batch or out/dout shape");
        TORCH_CHECK(dq.sizes() == q.sizes() && dk.sizes() == k.sizes() && dv.sizes() == v.sizes(),
                    "mha_bwd_mla: dq/dk/dv must match q/k/v shapes");
        TORCH_CHECK(dq.dtype() == q.dtype() && dk.dtype() == q.dtype() && dv.dtype() == q.dtype(),
                    "mha_bwd_mla: dq/dk/dv must match q dtype");
        TORCH_CHECK(softmax_lse.dim() == 3 && softmax_lse.size(0) == batch_size && softmax_lse.size(1) == num_heads &&
                        softmax_lse.size(2) == q_seqlen && softmax_lse.scalar_type() == at::kFloat &&
                        softmax_lse.device() == q.device(),
                    "mha_bwd_mla: BNSD LSE must be float32 [B,Nq,Sq] on q device");
        softmax_lse = softmax_lse.contiguous();
    }

    // ============================================================
    // FAG tiling (host)
    // ============================================================
    for (const auto& tensor : {q, k, v, out, dout}) {
        TORCH_CHECK(tensor.stride(-1) == 1 && tensor.device() == q.device(),
                    "mha_bwd_mla: inputs require contiguous D and the same device");
    }
    for (const auto& tensor : {dq, dk, dv}) {
        TORCH_CHECK(tensor.is_contiguous() && tensor.device() == q.device(),
                    "mha_bwd_mla: gradients must be contiguous BSND/TND on q device");
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
    // MLA backward does not support the deterministic schedule; ignore the
    // user-provided flag and always use the non-deterministic path.
    fag_info.deterministic = 0;
    fag_info.detSchedule = 0;
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

    at::Tensor cu_q_cpu;
    at::Tensor cu_k_cpu;
    std::vector<int64_t> actual_seq_q;
    std::vector<int64_t> actual_seq_kv;
    if (is_varlen) {
        const at::Tensor& cu_q_tensor = cu_seqlens_q_.value();
        const at::Tensor& cu_k_tensor = cu_seqlens_k_.value();
        TORCH_CHECK(cu_q_tensor.dim() == 1 && cu_k_tensor.dim() == 1 && cu_q_tensor.size(0) == batch_size + 1 &&
                        cu_k_tensor.size(0) == batch_size + 1,
                    "Ascend950 v4 mla bwd: cu_seqlens must be 1D tensors of length B + 1");
        TORCH_CHECK(cu_q_tensor.dtype() == at::kInt && cu_k_tensor.dtype() == at::kInt,
                    "Ascend950 v4 mla bwd: cu_seqlens must have dtype int32");
        TORCH_CHECK(cu_q_tensor.device().type() == at::kPrivateUse1 && cu_k_tensor.device().type() == at::kPrivateUse1,
                    "Ascend950 v4 mla bwd: cu_seqlens must be on NPU");
        cu_q_cpu = cu_q_tensor.to(at::Device(at::kCPU)).to(at::kInt).contiguous();
        cu_k_cpu = cu_k_tensor.to(at::Device(at::kCPU)).to(at::kInt).contiguous();
        const int32_t* q_lengths = cu_q_cpu.data_ptr<int32_t>();
        const int32_t* kv_lengths = cu_k_cpu.data_ptr<int32_t>();
        TORCH_CHECK(q_lengths[0] == 0 && kv_lengths[0] == 0, "Ascend950 v4 mla bwd: cu_seqlens must start at zero");
        actual_seq_q.resize(batch_size);
        actual_seq_kv.resize(batch_size);
        bool tndAllNonEmpty = batch_size > 0;
        for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            TORCH_CHECK(q_lengths[batch_idx + 1] >= q_lengths[batch_idx] &&
                            kv_lengths[batch_idx + 1] >= kv_lengths[batch_idx],
                        "Ascend950 v4 mla bwd: cu_seqlens must be nondecreasing");
            TORCH_CHECK(q_lengths[batch_idx + 1] - q_lengths[batch_idx] <= q_seqlen &&
                            kv_lengths[batch_idx + 1] - kv_lengths[batch_idx] <= kv_seqlen,
                        "Ascend950 v4 mla bwd: an actual sequence length exceeds max_seqlen");
            actual_seq_q[batch_idx] = q_lengths[batch_idx + 1] - q_lengths[batch_idx];
            actual_seq_kv[batch_idx] = kv_lengths[batch_idx + 1] - kv_lengths[batch_idx];
            if (actual_seq_q[batch_idx] <= 0 || actual_seq_kv[batch_idx] <= 0) {
                tndAllNonEmpty = false;
            }
        }
        TORCH_CHECK(q_lengths[batch_size] == static_cast<int64_t>(fag_info.totalQ) &&
                        kv_lengths[batch_size] == static_cast<int64_t>(fag_info.totalKv),
                    "Ascend950 v4 mla bwd: final cu_seqlens values must match packed tensor lengths");
        at::Tensor used_q_cpu =
            seqused_q_.has_value() ? seqused_q_->to(at::Device(at::kCPU)).to(at::kInt).contiguous() : at::Tensor();
        at::Tensor used_k_cpu =
            seqused_k_.has_value() ? seqused_k_->to(at::Device(at::kCPU)).to(at::kInt).contiguous() : at::Tensor();
        const int32_t* used_q = used_q_cpu.defined() ? used_q_cpu.data_ptr<int32_t>() : nullptr;
        const int32_t* used_kv = used_k_cpu.defined() ? used_k_cpu.data_ptr<int32_t>() : nullptr;
        for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            if (used_q != nullptr) {
                TORCH_CHECK(used_q[batch_idx] >= 0 && used_q[batch_idx] <= actual_seq_q[batch_idx],
                            "Ascend950 v4 mla bwd: seqused_q must lie within the allocated sequence length");
            }
            if (used_kv != nullptr) {
                TORCH_CHECK(used_kv[batch_idx] >= 0 && used_kv[batch_idx] <= actual_seq_kv[batch_idx],
                            "Ascend950 v4 mla bwd: seqused_k must lie within the allocated sequence length");
            }
        }
        if (used_q != nullptr || used_kv != nullptr) {
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

    FAGTiling950::FAGTilingData fag_tiling_data{};
    const int64_t tiling_status = FAGTiling950::GetFAGTilingParam(fag_info, fag_tiling_data);
    TORCH_CHECK(tiling_status == 0, "Ascend950 v4 mla bwd: arch35 GetFAGTilingParam failed");

    for (const auto& tensor : {q, k, v, out, dout}) {
        TORCH_CHECK(tensor.stride(is_varlen ? 0 : 1) >= tensor.size(-1), "mha_bwd_mla: token rows must not overlap");
    }
    auto tensor_strides = [is_varlen](const at::Tensor& tensor) {
        return FagTensorStrides{is_varlen ? 0 : tensor.stride(0), tensor.stride(is_varlen ? 0 : 1),
                                tensor.stride(is_varlen ? 1 : 2)};
    };
    fag_tiling_data.inputStrides = {tensor_strides(q),    tensor_strides(k),   tensor_strides(v),
                                    tensor_strides(dout), tensor_strides(out), 1};
    at::Tensor tiling_cpu =
        at::empty({static_cast<int64_t>(sizeof(FAGTiling950::FAGTilingData))}, at::device(c10::kCPU).dtype(at::kByte));
    std::memcpy(tiling_cpu.data_ptr<uint8_t>(), &fag_tiling_data, sizeof(FAGTiling950::FAGTilingData));
    at::Tensor tiling_device = tiling_cpu.to(at::Device(at::kPrivateUse1));

    // ============================================================
    // FAG workspace (device)
    // ============================================================
    const uint64_t workspace_size = static_cast<uint64_t>(fag_tiling_data.workspaceSize);
    TORCH_CHECK(workspace_size > 0 && workspace_size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "Ascend950 v4 mla bwd: invalid workspace size from tiling");
    TORCH_CHECK(workspace_size % sizeof(float) == 0 && fag_tiling_data.deltaOffset % sizeof(float) == 0,
                "Ascend950 v4 mla bwd: FP32 workspace offsets must be aligned");
    at::Tensor workspace = at::empty({static_cast<int64_t>(workspace_size / sizeof(float))},
                                     at::device(at::kPrivateUse1).dtype(at::kFloat));

    auto ptr = [](const at::Tensor& tensor) { return static_cast<uint8_t*>(tensor.data_ptr()); };
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

    at::Tensor mask_npu_tensor;
    uint8_t* mask = nullptr;
    if (has_attn_mask) {
        mask_npu_tensor = MakeMlaBwdTriuMask();
        mask = ptr(mask_npu_tensor);
    }

    stream = c10_npu::getCurrentNPUStream().stream(true);

    const uint32_t launch_cores = (fag_tiling_data.usedCoreNum > 0 && fag_tiling_data.usedCoreNum <= aic_num)
                                      ? fag_tiling_data.usedCoreNum
                                      : aic_num;

#define LAUNCH_BWD950_MLA(DTYPE, INPUT_LAYOUT, IS_ATTEN_MASK, IS_SOFTCAP)                                              \
    FlashAttentionV4BwdMla950<DTYPE, FAGTiling950::Layout::INPUT_LAYOUT, IS_ATTEN_MASK, IS_SOFTCAP>                    \
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

#undef DISPATCH_BWD950_MLA_FLAGS
#undef LAUNCH_BWD950_MLA

    at::Tensor softmax_d = is_varlen ? at::empty({num_heads, q.size(0)}, q.options().dtype(at::kFloat))
                                     : at::empty({batch_size, num_heads, q_seqlen}, q.options().dtype(at::kFloat));

    return {dq, dk, dv, softmax_d};
}
