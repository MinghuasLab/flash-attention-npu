# Copyright (c) 2026, Minghua Shen.
"""Two-block MHA causal partitions, split KV accumulation and guard boundaries."""

import importlib
import pytest
import torch
import torch_npu  # noqa: F401
from tests.test_mla_interface_padding_950 import _segment, _check


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "sq,sk,heads,layout",
    [
        (129, 129, 16, "BSND"),
        (198, 198, 32, "TND"),
        (256, 256, 64, "BSND"),
        (198, 198, 1, "TND"),
        (128, 128, 32, "BSND"),
        (257, 257, 32, "TND"),
        (129, 198, 32, "TND"),
    ],
)
def test_small_mha_causal(version, dtype, sq, sk, heads, layout):
    if "950" not in torch.npu.get_device_name():
        pytest.skip("Ascend 950 required")
    scale = 0.14467962580268923
    values, refs = _segment(192, 128, dtype, sq, sk, True, -1, -1, 0.0, scale, heads, heads)
    if layout == "TND":
        values = [x.squeeze(0) for x in values]
        refs = [x.squeeze(0) for x in refs]
    q, k, v, o, do, lse = [x.npu() for x in values]
    grads = [torch.empty_like(x) for x in (q, k, v)]
    cuq = torch.tensor([0, sq], device="npu", dtype=torch.int32) if layout == "TND" else None
    cuk = cuq if sq == sk else torch.tensor([0, sk], device="npu", dtype=torch.int32)
    if layout != "TND":
        cuk = None
    module = importlib.import_module(f"flash_attn_npu_{version}_950")
    fn = module.bwd
    print("Launch", version, dtype, sq, sk, heads, layout, flush=True)
    fn(
        do,
        q,
        k,
        v,
        o,
        lse,
        *grads,
        cuq,
        cuk,
        None,
        None,
        sq + 32 if layout == "TND" else None,
        sk + 32 if layout == "TND" else None,
        scale,
        True,
        -1,
        -1,
        0.0,
        False,
        0,
        0 if layout == "TND" else 1,
    )
    torch.npu.synchronize()
    for got, expected in zip(grads, refs):
        _check(got, expected, dtype)
