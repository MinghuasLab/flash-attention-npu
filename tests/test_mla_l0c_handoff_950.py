# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Full-tile L0C handoffs with asymmetric Q/K lengths and resident dK/dV."""

import importlib

import pytest
import torch
import torch_npu  # noqa: F401

from tests.test_mla_interface_padding_950 import _check, _segment


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "sq,sk,heads,kv_heads",
    [
        (128, 128, 1, 1),
        (128, 128, 2, 2),
        (128, 128, 3, 3),
        (128, 256, 16, 2),
        (256, 128, 16, 2),
        (384, 512, 8, 2),
        (256, 256, 32, 32),
    ],
)
def test_mla_l0c_handoff(version, dtype, sq, sk, heads, kv_heads):
    if "950" not in torch.npu.get_device_name():
        pytest.skip("Ascend 950 required")
    scale = 192**-0.5
    values, refs = _segment(192, 128, dtype, sq, sk, False, -1, -1, 0.0, scale, heads, kv_heads)
    q, k, v, out, dout, lse = [x.npu() for x in values]
    grads = [torch.empty_like(x) for x in (q, k, v)]
    module = importlib.import_module(f"flash_attn_npu_{version}_950")
    backward = module.bwd if version == 3 else module.bwd_mla
    backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        *grads,
        None,
        None,
        None,
        None,
        None,
        None,
        scale,
        False,
        -1,
        -1,
        0.0,
        False,
        0,
        1,
    )
    torch.npu.synchronize()
    for actual, expected in zip(grads, refs):
        _check(actual, expected, dtype)
