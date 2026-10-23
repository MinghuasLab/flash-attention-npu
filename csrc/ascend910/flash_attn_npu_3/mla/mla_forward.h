/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once
#include <torch/extension.h>
namespace flash_attn::mla {
std::optional<std::vector<at::Tensor>>
TryForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const std::optional<at::Tensor>& k_new_,
           const std::optional<at::Tensor>& v_new_, const std::optional<at::Tensor>& q_v_,
           const std::optional<at::Tensor>& out_, const std::optional<at::Tensor>& cu_seqlens_q_,
           const std::optional<at::Tensor>& cu_seqlens_k_, const std::optional<at::Tensor>& cu_seqlens_k_new_,
           const std::optional<at::Tensor>& seqused_q_, const std::optional<at::Tensor>& seqused_k_,
           std::optional<int64_t> max_seqlen_q_, std::optional<int64_t> max_seqlen_k_,
           const std::optional<at::Tensor>& page_table_, const std::optional<at::Tensor>& kv_batch_idx_,
           const std::optional<at::Tensor>& leftpad_k_, const std::optional<at::Tensor>& rotary_cos_,
           const std::optional<at::Tensor>& rotary_sin_, const std::optional<at::Tensor>& seqlens_rotary_,
           const std::optional<at::Tensor>& q_descale_, const std::optional<at::Tensor>& k_descale_,
           const std::optional<at::Tensor>& v_descale_, std::optional<double> softmax_scale_, bool is_causal,
           int64_t window_size_left, int64_t window_size_right, int64_t attention_chunk, double softcap_option,
           bool is_rotary_interleaved, const std::optional<at::Tensor>& scheduler_metadata_, int64_t num_splits,
           std::optional<bool> pack_gqa_, int64_t sm_margin, int64_t layout);

at::Tensor SchedulerMetadata(int64_t batch, int64_t sq, int64_t sk, int64_t heads, int64_t kvHeads, int64_t dim,
                             int64_t valueDim, at::ScalarType dtype, const at::Tensor& lengths,
                             const std::optional<at::Tensor>& cuQ, std::optional<int64_t> page, int64_t left,
                             int64_t right, bool causal, bool hasSoftcap, int64_t splits, std::optional<bool> pack,
                             int64_t margin);

} // namespace flash_attn::mla
