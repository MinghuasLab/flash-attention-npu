/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef QUANT_ATTENTION_HPP
#define QUANT_ATTENTION_HPP

#include <optional>
#include <string>
#include <vector>
#include <torch/extension.h>

enum class QuantInputDType : int64_t {
    Float4E2M1 = 0,
};

at::Tensor npu_quant_attention(
    at::Tensor query, at::Tensor key, at::Tensor value, at::Tensor q_dequant_scale, at::Tensor k_dequant_scale,
    at::Tensor v_dequant_scale, std::string input_layout, int64_t num_key_value_heads, double softmax_scale,
    int64_t quant_mode, std::optional<std::vector<int64_t>> actual_seq_lengths,
    std::optional<std::vector<int64_t>> actual_seq_lengths_kv, std::optional<at::Tensor> block_sparse_mask,
    std::optional<std::vector<int64_t>> block_shape, std::optional<at::ScalarType> output_dtype, double dst_type_max,
    int64_t quant_input_dtype);

#endif // QUANT_ATTENTION_HPP
