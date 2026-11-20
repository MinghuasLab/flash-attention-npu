# Copyright (c) 2026, Minghua Shen.
"""Generic FAG keeps its original metadata layout beside MLA epilogues."""

import importlib

import pytest
import torch
import torch_npu  # noqa: F401

from tests.test_mla_interface_padding_950 import _check, _segment


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [128, 192, 256])
@pytest.mark.parametrize("causal", [False, True])
def test_generic_epilogue(version, dtype, head_dim, causal):
    if "950" not in torch.npu.get_device_name():
        pytest.skip("Ascend 950 required")
    scale = head_dim**-0.5
    values, refs = _segment(head_dim, head_dim, dtype, 128, 256, causal, -1, -1, 0.0, scale)
    q, k, v, out, dout, lse = [x.npu() for x in values]
    grads = [torch.empty_like(x) for x in (q, k, v)]
    # Equal QK/V dimensions always select the generic C++ backward entry.
    backend = importlib.import_module(f"flash_attn_npu_{version}_950")
    backend.bwd(
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
        causal,
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
