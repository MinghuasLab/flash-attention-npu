#include "fag_general_host.hpp"

#include <cstring>

#include "acl/acl.h"
#include "catlass/catlass.hpp"
#include "kernel_operator.h"
#include "tiling/platform/platform_ascendc.h"
#include "../flash_attn_npu_3/fag_tiling.h"
#include "../flash_attn_npu_3/fag_tiling.cpp"
#include "runtime/rt_ffts.h"
#include "fag_general_dispatch.hpp"

std::vector<at::Tensor> launch_fag_general(
    const at::Tensor &dout,
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &out,
    const at::Tensor &softmax_lse,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    const std::optional<at::Tensor> &cu_seqlens_q,
    const std::optional<at::Tensor> &cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    float softmax_scale,
    float softcap,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    bool deterministic,
    float p_dropout,
    const std::optional<at::Tensor> &rng_state)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    bool is_bf16 = q.dtype() == torch::kBFloat16;

    const bool is_varlen_q = cu_seqlens_q.has_value();
    TORCH_CHECK(!is_varlen_q || cu_seqlens_k.has_value(),
                "launch_fag_general: cu_seqlens_k must be provided when cu_seqlens_q is set.");

    at::Tensor cu_seqlens_q_tensor;
    at::Tensor cu_seqlens_k_tensor;
    if (is_varlen_q) {
        cu_seqlens_q_tensor = cu_seqlens_q.value();
        cu_seqlens_k_tensor = cu_seqlens_k.value();
    }

    auto qsizes = q.sizes();
    auto ksizes = k.sizes();
    auto vsizes = v.sizes();
    auto dout_sizes = dout.sizes();
    auto out_sizes = out.sizes();
    const int64_t expected_dim = is_varlen_q ? 3 : 4;
    TORCH_CHECK(q.dim() == expected_dim && k.dim() == expected_dim && v.dim() == expected_dim &&
                    dout.dim() == expected_dim && out.dim() == expected_dim,
                "launch_fag_general: q/k/v/dout/out must be ", expected_dim, "-D (",
                (is_varlen_q ? "TND" : "BSND"), ")");
    TORCH_CHECK(dout_sizes == out_sizes, "launch_fag_general: out and dout must have the same shape");
    TORCH_CHECK(dq.sizes() == qsizes && dk.sizes() == ksizes && dv.sizes() == vsizes,
                "launch_fag_general: dq/dk/dv must match q/k/v shapes");

    uint32_t nheads = is_varlen_q ? qsizes[1] : qsizes[2];
    uint32_t nheads_k = is_varlen_q ? ksizes[1] : ksizes[2];
    uint32_t q_headdim = is_varlen_q ? qsizes[2] : qsizes[3];
    uint32_t v_headdim = is_varlen_q ? vsizes[2] : vsizes[3];
    uint32_t k_headdim = is_varlen_q ? ksizes[2] : ksizes[3];
    uint32_t dout_headdim = static_cast<uint32_t>(dout_sizes[expected_dim - 1]);
    TORCH_CHECK(nheads > 0 && nheads_k > 0, "launch_fag_general: number of Q/KV heads must be positive");
    TORCH_CHECK(nheads % nheads_k == 0,
                "launch_fag_general: number of heads in key/value must divide number of heads in query");
    // NPU FAG bwd currently requires q/k/v/dout headdim to be equal.
    TORCH_CHECK(q_headdim == k_headdim && q_headdim == v_headdim && q_headdim == dout_headdim,
                "launch_fag_general: q/k/v/dout must share the same headdim (unequal headdim is not supported)");
    TORCH_CHECK(q_headdim > 0 && q_headdim <= 256, "launch_fag_general: headdim must be in (0, 256].");
    TORCH_CHECK(qsizes == dout_sizes, "launch_fag_general: q and dout must have the same shape");
    TORCH_CHECK(ksizes == vsizes, "launch_fag_general: k and v must have the same shape");
    if (is_varlen_q) {
        TORCH_CHECK(static_cast<uint32_t>(vsizes[1]) == nheads_k,
                    "launch_fag_general: v nheads_k must match k");
    } else {
        TORCH_CHECK(qsizes[0] == ksizes[0], "launch_fag_general: q and k must share the same batch size");
        TORCH_CHECK(static_cast<uint32_t>(vsizes[2]) == nheads_k,
                    "launch_fag_general: v nheads_k must match k");
    }
    uint32_t qk_headdim_kernel = q_headdim <= 64 ? 64 : (q_headdim <= 128 ? 128 : (q_headdim <= 192 ? 192 : 256));
    int64_t batch_size = is_varlen_q ? (cu_seqlens_q_tensor.size(0) - 1) : qsizes[0];

    uint32_t tilingSize = sizeof(FAGTilingData);
    at::Tensor tiling_cpu_tensor = at::empty({static_cast<long>(tilingSize)}, at::device(c10::kCPU).dtype(at::kByte));
    FAGTiling::FAGInfo fagInfo;
    
    bool has_softcap = (softcap > 0.0f);
    if (has_softcap) {
        fagInfo.scaleValue = softmax_scale / softcap;
    } else {
        fagInfo.scaleValue = softmax_scale;
    }
    fagInfo.softcapValue = softcap;

    bool has_dropout = (p_dropout > 0.0f);
    fagInfo.keepProb = 1.0f - p_dropout;
    if (window_size_left >= max_seqlen_k - 1) {
        window_size_left = -1;
    }
    if (window_size_right >= max_seqlen_q - 1) {
        window_size_right = -1;
    }
    if (is_causal) {
        window_size_right = 0;
    }
    is_causal = window_size_left < 0 && window_size_right == 0;
    const bool is_local = (window_size_left >= 0 || window_size_right >= 0) && !is_causal;
    const bool has_attn_mask = is_causal || is_local;
    if (is_causal) {
        fagInfo.maskType = static_cast<int32_t>(FAGTiling::MaskType::MASK_CAUSUAL);
        fagInfo.window_size_left = window_size_left;
        fagInfo.window_size_right = 0;
    } else if (is_local) {
        fagInfo.maskType = static_cast<int32_t>(FAGTiling::MaskType::MASK_BAND);
        fagInfo.window_size_left = window_size_left;
        fagInfo.window_size_right = window_size_right;
    } else {
        fagInfo.maskType = static_cast<int32_t>(FAGTiling::MaskType::NO_MASK);
        fagInfo.window_size_left = window_size_left;
        fagInfo.window_size_right = window_size_right;
    }
    fagInfo.batch = batch_size;
    fagInfo.qSeqlen = max_seqlen_q;
    fagInfo.qHeadNum = nheads;
    fagInfo.qkHeadDim = q_headdim;
    fagInfo.kvSeqlen = max_seqlen_k;
    fagInfo.kvHeadNum = nheads_k;
    fagInfo.vHeadDim = v_headdim;
    fagInfo.isDeterministic = deterministic;
    fagInfo.layout = static_cast<int32_t>(is_varlen_q ? TND : BSND);

    at::Tensor cu_seqlens_q_cpu_for_tiling;
    at::Tensor cu_seqlens_k_cpu_for_tiling;
    if (is_varlen_q) {
        cu_seqlens_q_cpu_for_tiling = cu_seqlens_q_tensor.to(at::Device(at::kCPU)).to(at::kInt).contiguous();
        cu_seqlens_k_cpu_for_tiling = cu_seqlens_k_tensor.to(at::Device(at::kCPU)).to(at::kInt).contiguous();
        fagInfo.qSeqlenList = static_cast<int32_t *>(cu_seqlens_q_cpu_for_tiling.data_ptr()) + 1;
        fagInfo.kvSeqlenList = static_cast<int32_t *>(cu_seqlens_k_cpu_for_tiling.data_ptr()) + 1;
    }

    uint32_t aivNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    FAGTilingData fagTilingData;
    int64_t tilingStatus = FAGTiling::GetFAGTilingParam(fagInfo, blockDim, aivNum, ubSize, fagTilingData);
    TORCH_CHECK(tilingStatus == 0, "launch_fag_general: GetFAGTilingParam failed.");
    fagTilingData.actualSeqQlen.clear();
    fagTilingData.actualSeqKvlen.clear();
    std::memcpy(tiling_cpu_tensor.data_ptr<uint8_t>(), &fagTilingData, sizeof(FAGTilingData));
    at::Tensor tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));

    uint64_t workspaceSize = static_cast<uint64_t>(fagTilingData.workspaceSize);
    TORCH_CHECK(workspaceSize > 0, "launch_fag_general: invalid workspace size from tiling.");
    at::Tensor workspace_tensor =
        at::empty({static_cast<long>(workspaceSize)}, at::device(at::kPrivateUse1).dtype(at::kByte));

    at::Tensor mask_gpu_tensor;
    if (has_attn_mask) {
        const int64_t mask_dim = FAGTiling::ATTEN_MASK_COMPRESS_DIM;
        mask_gpu_tensor = at::triu(
            at::ones({mask_dim, mask_dim}, at::device(at::kPrivateUse1).dtype(at::kByte)), 1);
    }

    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    auto qDevice = static_cast<uint8_t *>(const_cast<void *>(q.storage().data()));
    auto kDevice = static_cast<uint8_t *>(const_cast<void *>(k.storage().data()));
    auto vDevice = static_cast<uint8_t *>(const_cast<void *>(v.storage().data()));
    auto outDevice = static_cast<uint8_t *>(const_cast<void *>(out.storage().data()));
    auto dOutDevice = static_cast<uint8_t *>(const_cast<void *>(dout.storage().data()));
    uint8_t *attenMaskDevice = nullptr;
    if (mask_gpu_tensor.defined()) {
        attenMaskDevice = static_cast<uint8_t *>(const_cast<void *>(mask_gpu_tensor.storage().data()));
    }

    at::Tensor softmax_lse_kernel = softmax_lse;
        if (!is_varlen_q) {
        TORCH_CHECK(softmax_lse.dim() == 3, "launch_fag_general: softmax_lse for BSND must be a 3D tensor.");
        TORCH_CHECK(softmax_lse.size(1) == nheads && softmax_lse.size(2) == max_seqlen_q,
                    "launch_fag_general: softmax_lse must be BNS in BSND mode.");
        if (!softmax_lse.is_contiguous()) {
            softmax_lse_kernel = softmax_lse.contiguous();
        }
    } else {
        TORCH_CHECK(softmax_lse.dim() == 2, "launch_fag_general: softmax_lse for TND must be a 2D tensor.");
        const int64_t total_q = qsizes[0];
        TORCH_CHECK(softmax_lse.size(0) == nheads && softmax_lse.size(1) == total_q,
                    "launch_fag_general: softmax_lse must be NT (nheads, total_q) in TND mode.");
        if (!softmax_lse.is_contiguous()) {
            softmax_lse_kernel = softmax_lse.contiguous();
        }
    }
    auto softMaxLseDevice = static_cast<uint8_t *>(const_cast<void *>(softmax_lse_kernel.storage().data()));

    auto workspaceDevice = static_cast<uint8_t *>(const_cast<void *>(workspace_tensor.storage().data()));
    auto tilingDevice = static_cast<uint8_t *>(const_cast<void *>(tiling_gpu_tensor.storage().data()));
    auto dqDevice = static_cast<uint8_t *>(const_cast<void *>(dq.storage().data()));
    auto dkDevice = static_cast<uint8_t *>(const_cast<void *>(dk.storage().data()));
    auto dvDevice = static_cast<uint8_t *>(const_cast<void *>(dv.storage().data()));

    uint8_t *cuSeqQlenDevice = nullptr;
    uint8_t *cuSeqKvlenDevice = nullptr;
    at::Tensor seqlenq_gpu_tensor;
    at::Tensor seqlenk_gpu_tensor;
    if (is_varlen_q) {
        seqlenq_gpu_tensor = cu_seqlens_q_tensor.slice(0, 1, cu_seqlens_q_tensor.size(0)).contiguous();
        seqlenk_gpu_tensor = cu_seqlens_k_tensor.slice(0, 1, cu_seqlens_k_tensor.size(0)).contiguous();
        cuSeqQlenDevice = static_cast<uint8_t *>(const_cast<void *>(seqlenq_gpu_tensor.data_ptr()));
        cuSeqKvlenDevice = static_cast<uint8_t *>(const_cast<void *>(seqlenk_gpu_tensor.data_ptr()));
    }

    // Regenerate the dropout bit-mask from the forward's rng_state (seed,
    // offset). Same aclnnDropoutGenMask call and the same bit layout as the
    // forward (batch x heads x max_seqlen_q rows x ceil(max_seqlen_k/8) bytes
    // per row, bit 1 = keep), so the backward applies the exact same mask
    // element-wise as the forward.
    at::Tensor drop_mask_npu_tensor;
    if (has_dropout) {
        TORCH_CHECK(rng_state.has_value(),
                    "launch_fag_general: rng_state must be provided when p_dropout > 0.");
        const at::Tensor &rng_state_tensor = rng_state.value();
        TORCH_CHECK(rng_state_tensor.is_cpu() && rng_state_tensor.numel() == 2,
                    "launch_fag_general: rng_state must be a CPU tensor of 2 int64 (seed, offset).");
        const uint64_t *rng_state_ptr = reinterpret_cast<const uint64_t *>(rng_state_tensor.data_ptr());
        int64_t drop_mask_bit_num =
            static_cast<int64_t>(batch_size) * nheads * max_seqlen_q * ((max_seqlen_k + 7) / 8 * 8);
        drop_mask_npu_tensor = at_npu::native::npu_dropout_gen_mask(
            torch::empty({0}, at::device(at::kPrivateUse1).dtype(at::kFloat)), {drop_mask_bit_num}, p_dropout,
            rng_state_ptr[0], rng_state_ptr[1], false, false);
    }

    // FAGGeneral backward kernel launches live in
    // fag_general_dispatch_<dtype>_<layout>.cpp. gen_args.is_causal carries
    // main's has_attn_mask flag (causal OR local), which selects the kernel's
    // IS_ATTEN_MASK template param — same role as LaunchFAGGeneralKernel's 2nd
    // arg. The OpCommand ("ascendc_fag") wrapper is applied inside the dispatch.
    FagGeneralLaunchArgs gen_args;
    gen_args.blockDim = blockDim;
    gen_args.aclStream = aclStream;
    gen_args.fftsAddr = fftsAddr;
    gen_args.is_causal = has_attn_mask;
    gen_args.is_softcap = has_softcap;
    gen_args.has_dropout = has_dropout;
    gen_args.dropMaskDevice =
        has_dropout ? static_cast<uint8_t *>(const_cast<void *>(drop_mask_npu_tensor.data_ptr())) : nullptr;
    gen_args.deterministic = deterministic;
    gen_args.qk_headdim_kernel = qk_headdim_kernel;
    gen_args.dOutDevice = dOutDevice;
    gen_args.qDevice = qDevice;
    gen_args.kDevice = kDevice;
    gen_args.vDevice = vDevice;
    gen_args.outDevice = outDevice;
    gen_args.attenMaskDevice = attenMaskDevice;
    gen_args.softMaxLseDevice = softMaxLseDevice;
    gen_args.cuSeqQlenDevice = cuSeqQlenDevice;
    gen_args.cuSeqKvlenDevice = cuSeqKvlenDevice;
    gen_args.dqDevice = dqDevice;
    gen_args.dkDevice = dkDevice;
    gen_args.dvDevice = dvDevice;
    gen_args.workspaceDevice = workspaceDevice;
    gen_args.tilingDevice = tilingDevice;

    if (is_varlen_q) {
        launch_fag_general_dispatch<TND>(is_bf16, gen_args);
    } else {
        launch_fag_general_dispatch<BSND>(is_bf16, gen_args);
    }

    auto opts = q.options();
    auto softmax_d = torch::empty({batch_size, nheads, max_seqlen_q}, opts.dtype(at::kFloat));
    return {dq, dk, dv, softmax_d};
}
