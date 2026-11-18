# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""MLA resident-K/V lifetime across KV groups assigned to the same AIC."""

import importlib
import time

import pytest
import torch
import torch_npu  # noqa: F401

from tests.common.attention_ref import ref_flash_attention


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch,seqlen,heads,kv_heads,causal",
    [
        (1, 1792, 16, 2, True),
        (1, 2048, 16, 2, True),
        (1, 4096, 16, 2, False),
        (1, 2049, 16, 2, True),
        (8, 512, 8, 2, False),
        (1, 1024, 32, 4, True),
    ],
)
def test_mla_k_reuse(version, dtype, batch, seqlen, heads, kv_heads, causal):
    if "950" not in torch.npu.get_device_name():
        pytest.skip("Ascend 950 MLA backend required")
    backend = importlib.import_module(f"flash_attn_npu_{version}_950")
    # Build the existing CPU reference a head at a time to bound quadratic memory.
    q = torch.randn(batch, seqlen, heads, 192, dtype=dtype)
    k = torch.randn(batch, seqlen, kv_heads, 192, dtype=dtype)
    v = torch.randn(batch, seqlen, kv_heads, 128, dtype=dtype)
    dout = torch.randn(batch, seqlen, heads, 128, dtype=dtype)
    out = torch.empty_like(dout)
    lse = torch.empty(batch, heads, seqlen)
    expected = [torch.zeros_like(x, dtype=torch.float32) for x in (q, k, v)]
    mask = torch.ones(seqlen, seqlen, dtype=torch.bool).triu(1) if causal else None
    scale = 192**-0.5
    for h in range(heads):
        kh = h // (heads // kv_heads)
        inputs = [
            q[:, :, h : h + 1].float().requires_grad_(),
            k[:, :, kh : kh + 1].float().requires_grad_(),
            v[:, :, kh : kh + 1].float().requires_grad_(),
        ]
        ref_out, ref_lse = ref_flash_attention(*inputs, scale, mask, torch.float32)
        grads = torch.autograd.grad(ref_out, inputs, dout[:, :, h : h + 1].float())
        out[:, :, h : h + 1] = ref_out.detach().to(dtype)
        lse[:, h : h + 1] = ref_lse.detach()
        expected[0][:, :, h : h + 1] = grads[0]
        expected[1][:, :, kh : kh + 1] += grads[1]
        expected[2][:, :, kh : kh + 1] += grads[2]

    q, k, v, out, dout, lse = [x.npu() for x in (q, k, v, out, dout, lse)]
    actual = [torch.empty_like(x) for x in (q, k, v)]
    bwd = backend.bwd if version == 3 else backend.bwd_mla
    torch.npu.synchronize()
    print(f"Launching v{version} B={batch} S={seqlen} H={heads}/{kv_heads} {dtype}", flush=True)
    start = time.monotonic()
    bwd(
        dout,
        q,
        k,
        v,
        out,
        lse,
        *actual,
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
    print(f"v{version} S={seqlen} {dtype} backward returned in {time.monotonic() - start:.6f}s")
    # Match the frozen MLA precision-policy.json (no relaxed large-shape tolerance).
    tolerance = 0.00195 if dtype == torch.float16 else 0.0156
    max_abs_limit = 0.1 if dtype == torch.float16 else 1.0
    for name, got, ref in zip(("dq", "dk", "dv"), actual, expected):
        got = got.cpu().float()
        error = (got - ref).abs()
        matched = error <= tolerance + tolerance * ref.abs()
        print(f"{name}: max_abs={error.max().item():.8f}, matched={matched.float().mean():.8f}")
        assert torch.isfinite(got).all(), name
        assert matched.float().mean().item() >= 0.999, name
        assert error.max().item() < max_abs_limit, name
