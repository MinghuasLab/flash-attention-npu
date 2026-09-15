/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 *
 * Pybind entrypoint for `flash_attn_npu_4_950` — the Ascend 950 backend
 * for FlashAttention v4.
 *

 */

#include <torch/extension.h>

#include "mha_fwd.cpp"
#include "mha_bwd.cpp"

PYBIND11_MODULE(flash_attn_npu_4_950, m)
{
    m.doc() = "FlashAttention v4 — Ascend 950 backend";
    m.def("fwd", &mha_fwd, "Forward pass, with KV-cache (Ascend 950)");
    m.def("bwd", &mha_bwd, "Backward pass (Ascend 950)");
}
