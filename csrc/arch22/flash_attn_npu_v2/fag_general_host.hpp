#ifndef FAG_GENERAL_HOST_HPP
#define FAG_GENERAL_HOST_HPP

#include <torch/extension.h>
#include <optional>
#include <vector>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

// Launch general-purpose FAG (FlashAttentionGrad) kernel (BSND or TND).
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
    bool deterministic);

#endif
