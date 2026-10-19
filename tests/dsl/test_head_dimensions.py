"""Independent QK/value dimensions share the attention and modifier pipeline."""

from importlib.util import find_spec
import os

import pytest

if find_spec("catlass") is None or find_spec("torch") is None:
    pytest.skip("optional CATLASS DSL / PyTorch runtime is absent", allow_module_level=True)

import torch
import catlass.tla as tla

from flash_attn_npu_dsl import (
    compute_block_sparsity,
    flash_attn_func,
    flash_attn_varlen_func,
    simd,
)


@simd
def _causal_simd(b, h, q, kv, info, aux):
    return tla.cmp(kv, q + info.seqlen_k - info.seqlen_q, "le")


def _causal_simt(b, h, q, kv, info, aux):
    return kv <= q + info.seqlen_k - info.seqlen_q


def _reference(q, k, v):
    group = q.shape[-2] // k.shape[-2]
    k = k.float().repeat_interleave(group, dim=-2)
    v = v.float().repeat_interleave(group, dim=-2)
    scores = torch.einsum("bmhd,bnhd->bhmn", q.float(), k) * q.shape[-1] ** -0.5
    rows, columns = q.shape[1], k.shape[1]
    visible = torch.arange(columns)[None, :] <= torch.arange(rows)[:, None] + columns - rows
    scores.masked_fill_(~visible, -torch.inf)
    lse = scores.logsumexp(-1)
    probs = torch.nan_to_num(scores.softmax(-1), nan=0.0)
    out = torch.einsum("bhmn,bnhd->bmhd", probs, v)
    return out, lse


def _check(out, lse, expected, expected_lse):
    out = out.cpu().float()
    assert out.shape == expected.shape
    assert torch.isfinite(out).all()
    assert (out - expected).abs().max().item() <= 0.05
    empty = torch.isneginf(expected_lse)
    assert torch.count_nonzero(out.permute(0, 2, 1, 3)[empty]) == 0
    if lse is not None:
        lse = lse.cpu()
        assert torch.equal(torch.isneginf(lse), empty)
        assert (lse[~empty] - expected_lse[~empty]).abs().max().item() <= 0.05


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"), reason="explicit NPU opt-in required"
)
@pytest.mark.parametrize(
    "dtype,dq,dv,batch,q_len,k_len,heads,mode",
    [
        (torch.float16, 128, 64, 2, 65, 257, (8, 2), "builtin"),
        (torch.bfloat16, 64, 128, 2, 128, 256, (4, 2), "builtin"),
        (torch.float16, 32, 96, 2, 129, 257, (4, 2), "simt"),
        (torch.bfloat16, 96, 16, 2, 257, 129, (4, 2), "sparse"),
        (torch.float16, 80, 8, 32, 3, 257, (8, 1), "builtin"),
        (torch.bfloat16, 8, 80, 32, 3, 257, (8, 2), "simd"),
        (torch.float16, 32, 96, 2, 192, 257, (4, 2), "sparse"),
        (torch.float16, 192, 128, 2, 65, 257, (4, 2), "builtin"),
        (torch.bfloat16, 128, 256, 2, 129, 257, (4, 2), "simt"),
        (torch.float16, 256, 64, 2, 129, 257, (4, 2), "sparse"),
        (torch.bfloat16, 129, 255, 32, 3, 257, (8, 2), "simd"),
        (torch.float16, 59, 65, 2, 257, 129, (4, 2), "sparse"),
    ],
)
def test_independent_value_dimension(dtype, dq, dv, batch, q_len, k_len, heads, mode):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(batch, q_len, heads[0], dq, generator=generator).to(dtype)
    k = torch.randn(batch, k_len, heads[1], dq, generator=generator).to(dtype)
    v = torch.randn(batch, k_len, heads[1], dv, generator=generator).to(dtype)
    expected, expected_lse = _reference(q, k, v)
    inputs = tuple(x.to(device) for x in (q, k, v))
    kwargs = {"causal": True}
    if mode != "builtin":
        kwargs = {"mask_mod": _causal_simt if mode == "simt" else _causal_simd}
    if mode == "sparse":
        kwargs["block_sparse_tensors"] = compute_block_sparsity(
            128, 128, 1, 1, q_len, k_len, _causal_simd, None, device
        )
    previous = None
    for return_lse in (True, False):
        out, lse = flash_attn_func(*inputs, **kwargs, return_lse=return_lse)
        _check(out, lse, expected, expected_lse)
        if previous is not None:
            assert torch.equal(previous, out)
        previous = out


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"), reason="explicit NPU opt-in required"
)
@pytest.mark.parametrize("empty_k", [False, True])
@pytest.mark.parametrize("dq,dv", [(64, 96), (256, 192), (65, 59)])
def test_packed_independent_value_dimension(empty_k, dq, dv):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(29)
    qs, ks = [3, 0, 65], [0, 0, 0] if empty_k else [5, 1, 129]
    q = torch.randn(sum(qs), 4, dq, generator=generator).half()
    k = torch.randn(sum(ks), 2, dq, generator=generator).half()
    v = torch.randn(sum(ks), 2, dv, generator=generator).half()
    cuq, cuk = (
        torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
        for lens in (qs, ks)
    )
    inputs = tuple(x.to(device) for x in (q, k, v))
    out, lse = flash_attn_varlen_func(
        *inputs,
        cu_seqlens_q=cuq.to(device),
        cu_seqlens_k=cuk.to(device),
        max_seqlen_q=max(qs),
        max_seqlen_k=max(ks),
        causal=True,
        return_lse=True,
    )
    if empty_k:
        assert out.shape == (sum(qs), 4, dv)
        assert torch.count_nonzero(out) == 0
        assert torch.isneginf(lse).all()
    else:
        for b, length in enumerate(qs):
            if not length:
                continue
            qi, ki = slice(cuq[b], cuq[b + 1]), slice(cuk[b], cuk[b + 1])
            expected, expected_lse = _reference(q[qi][None], k[ki][None], v[ki][None])
            _check(out[qi][None], lse[:, qi][None], expected, expected_lse)
