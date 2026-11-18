# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Public backward head-dim buckets, numerical semantics and output-buffer mutation."""

import importlib

import pytest
import torch
import torch_npu  # noqa: F401

from tests.common.attention_ref import ref_flash_attention
from tests.common.test_utils import make_golden_attention_mask

# Frozen MLA backward tolerances, also used by test_mla_k_reuse_950.py.
# Keep this test independent of ignored local operators/ workflow artifacts.
TOLERANCES = {torch.float16: (0.00195, 0.1), torch.bfloat16: (0.0156, 1.0)}


@pytest.fixture(autouse=True)
def _ascend950_only():
    if "950" not in torch.npu.get_device_name():
        pytest.skip("Ascend 950 backward interface required")


CASES = [
    (64, 32, (64, 64), "BSND", False, -1, -1, 0.0, None),
    (96, 64, (128, 128), "TND", True, -1, -1, 0.0, None),
    (128, 64, (128, 128), "BSND", False, 31, 17, 0.0, 0.11),
    (145, 81, (192, 128), "TND", True, -1, -1, 2.0, None),
    (129, 127, (192, 128), "BSND", True, -1, -1, 0.0, None),
    (145, 129, (192, 192), "TND", False, -1, -1, 0.0, None),
    (193, 128, (256, 256), "BSND", True, -1, -1, 0.0, None),
    (160, 128, (192, 128), "BSND", True, -1, -1, 0.0, None),
    (192, 64, (192, 128), "BSND", False, -1, -1, 0.0, None),
    (192, 128, (192, 128), "TND", True, -1, -1, 0.0, 0.14467962580268923),
    (192, 160, (192, 192), "BSND", False, -1, -1, 0.0, None),
    (224, 80, (256, 256), "TND", False, -1, -1, 0.0, None),
    (256, 128, (256, 256), "BSND", True, -1, -1, 0.0, None),
    (256, 192, (256, 256), "BSND", False, -1, -1, 0.0, 0.08),
    (160, 81, (192, 128), "BNSD", True, -1, -1, 0.0, None),
]


def _segment(qdim, vdim, dtype, sq, sk, causal, wl, wr, softcap, scale, heads=4, kv_heads=2):
    inputs = [
        torch.randn(1, sq, heads, qdim, dtype=dtype),
        torch.randn(1, sk, kv_heads, qdim, dtype=dtype),
        torch.randn(1, sk, kv_heads, vdim, dtype=dtype),
    ]
    do = torch.randn(1, sq, heads, vdim, dtype=dtype)
    xs = [x.float().requires_grad_() for x in inputs]
    mask = make_golden_attention_mask(sq, sk, causal, wl, wr)[0]
    out, lse = ref_flash_attention(*xs, scale, mask, torch.float32, softcap=softcap)
    refs = torch.autograd.grad(out, xs, do.float())
    return (*inputs, out.detach().to(dtype), do, lse.detach()), refs


def _check(actual, expected, dtype):
    tolerance, max_abs_limit = TOLERANCES[dtype]
    diff = (actual.cpu().float() - expected.detach()).abs()
    matched = diff <= tolerance + tolerance * expected.detach().abs()
    assert torch.isfinite(actual).all()
    assert matched.float().mean().item() >= 0.999
    assert diff.max().item() < max_abs_limit


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", CASES)
def test_backward_head_dim_buckets(version, dtype, case, monkeypatch):
    qdim, vdim, bucket, layout, causal, wl, wr, softcap, scale_arg = case
    scale = qdim ** (-0.5) if scale_arg is None else scale_arg
    api = importlib.import_module(f"flash_attn_npu_{version}.flash_attn_npu_interface_950")
    backend = importlib.import_module(f"flash_attn_npu_{version}_950")
    seen = []
    for name in ["bwd"] if version == 3 else ["bwd", "bwd_mla"]:
        original = getattr(backend, name)

        def record(*args, _original=original, _name=name):
            seen.append((_name, args[1].shape[-1], args[3].shape[-1], args[15]))
            return _original(*args)

        monkeypatch.setattr(backend, name, record)
    lengths = [(17, 33), (65, 97)] if layout == "TND" else [(65, 97)]
    pieces = [
        _segment(qdim, vdim, dtype, sq, sk, causal, wl, wr, softcap, scale) for sq, sk in lengths
    ]
    if layout == "TND":
        values = [
            torch.cat([p[0][i].squeeze(0) for p in pieces], dim=1 if i == 5 else 0).npu()
            for i in range(6)
        ]
        refs = [torch.cat([p[1][i].squeeze(0) for p in pieces], dim=0) for i in range(3)]
        cuq = torch.tensor([0, 17, 82], dtype=torch.int32, device="npu")
        cuk = torch.tensor([0, 33, 130], dtype=torch.int32, device="npu")
    else:
        values = [x.npu() for x in pieces[0][0]]
        refs = list(pieces[0][1])
        cuq = cuk = None
        if layout == "BNSD":
            if version == 3:
                values[:5] = [x.transpose(1, 2).contiguous().transpose(1, 2) for x in values[:5]]
            else:
                values[:5] = [x.transpose(1, 2) for x in values[:5]]
                refs = [x.transpose(1, 2) for x in refs]
    q, k, v, out, do, lse = values
    grads = [torch.full_like(x, float("nan")) for x in (q, k, v)]
    print(f"Launch v{version} {dtype} {qdim}/{vdim}->{bucket} {layout}", flush=True)
    common = dict(
        cu_seqlens_q=cuq,
        cu_seqlens_k=cuk,
        max_seqlen_q=65 if cuq is not None else None,
        max_seqlen_k=97 if cuk is not None else None,
        softmax_scale=scale_arg,
        softcap=softcap,
        window_size_left=wl,
        window_size_right=wr,
        dq=grads[0],
        dk=grads[1],
        dv=grads[2],
    )
    if version == 3:
        api._flash_attn_backward(do, q, k, v, out, lse, is_causal=causal, **common)
    else:
        returned = api._flash_attn_backward(
            q,
            k,
            v,
            out,
            do,
            lse,
            causal=causal,
            input_layout="bnsd" if layout == "BNSD" else None,
            **common,
        )
        assert all(got is buf for got, buf in zip(returned, grads))
    torch.npu.synchronize()
    assert len(seen) == 1
    assert seen[0][1:3] == bucket
    if version == 4:
        assert seen[0][0] == ("bwd_mla" if bucket == (192, 128) else "bwd")
    assert seen[0][3] == pytest.approx(scale)
    for got, ref in zip(grads, refs):
        _check(got, ref, dtype)


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("qdim,vdim", [(96, 64), (160, 96), (192, 128), (256, 128)])
@pytest.mark.parametrize("layout", ["BSND", "TND"])
def test_public_autograd_head_dim_buckets(version, dtype, qdim, vdim, layout):
    api = importlib.import_module(f"flash_attn_npu_{version}.flash_attn_npu_interface_950")
    scale = qdim ** (-0.5)
    values, refs = _segment(qdim, vdim, dtype, 65, 97, True, -1, -1, 0.0, scale)
    q, k, v, _, do, _ = values
    if layout == "TND":
        q, k, v, do = [x.squeeze(0) for x in (q, k, v, do)]
        refs = [x.squeeze(0) for x in refs]
    inputs = [x.npu().detach().requires_grad_() for x in (q, k, v)]
    if layout == "TND":
        cuq = torch.tensor([0, 65], dtype=torch.int32, device="npu")
        cuk = torch.tensor([0, 97], dtype=torch.int32, device="npu")
        result = api.flash_attn_varlen_func(
            *inputs,
            cu_seqlens_q=cuq,
            cu_seqlens_k=cuk,
            max_seqlen_q=65,
            max_seqlen_k=97,
            causal=True,
        )
    else:
        result = api.flash_attn_func(*inputs, causal=True)
    if isinstance(result, tuple):
        result = result[0]
    result.backward(do.npu())
    torch.npu.synchronize()
    assert result.shape == do.shape
    for inp, ref in zip(inputs, refs):
        assert inp.grad.shape == inp.shape
        _check(inp.grad, ref, dtype)
