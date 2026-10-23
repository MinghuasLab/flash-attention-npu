# Copyright (c) 2026, Huawei Technologies Co., Ltd.

"""FA3 split-attention combine (flash_attn_combine) tests.

Reference mirrors Dao-AILab/flash-attention tests/cute/test_flash_attn_combine.py.
"""

import pytest
import torch
import torch_npu

from flash_attn_npu_3 import flash_attn_combine


def _combine_ref(out_partial, lse_partial):
    lse = torch.logsumexp(lse_partial, dim=0)
    scale = torch.exp(lse_partial - lse)
    scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
    out = (scale.unsqueeze(-1) * out_partial).sum(0)
    return out, lse


def _make_lse_partial(num_splits, batch, seqlen, heads):
    # NV requires the seqlen dimension to be contiguous: build (NS, B, H, S)
    # and expose it as (NS, B, S, H).
    return torch.randn(num_splits, batch, heads, seqlen, device="npu", dtype=torch.float32).transpose(-1, -2)


def _assert_combine_close(out, lse, out_ref, lse_ref, dtype):
    torch.testing.assert_close(lse.float(), lse_ref, atol=1e-4, rtol=1e-4)
    cast_ref = out_ref.to(dtype)
    cast_err = (cast_ref.float() - out_ref).abs().max().item()
    diff = (out.float() - out_ref).abs().max().item()
    assert diff <= 2.0 * cast_err + 1e-3, f"out max diff {diff:.6f} exceeds 2x cast error {cast_err:.6f}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_splits", [1, 2, 5, 17])
@pytest.mark.parametrize("head_dim", [64, 96, 128])
@pytest.mark.parametrize("seqlen", [1, 33, 113])
def test_combine_dense(dtype, num_splits, head_dim, seqlen):
    batch, heads = 2, 4
    out_partial = torch.randn(num_splits, batch, seqlen, heads, head_dim, device="npu", dtype=torch.float32)
    lse_partial = _make_lse_partial(num_splits, batch, seqlen, heads)
    lse_partial[num_splits // 2 :, : batch // 2] = float("-inf")

    out, lse = flash_attn_combine(out_partial, lse_partial, out_dtype=dtype)
    assert out.shape == (batch, seqlen, heads, head_dim)
    assert out.dtype == dtype
    assert lse.shape == (batch, seqlen, heads)
    assert lse.dtype == torch.float32

    out_ref, lse_ref = _combine_ref(out_partial, lse_partial)
    _assert_combine_close(out, lse, out_ref, lse_ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_splits", [2, 3, 32])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_combine_varlen(dtype, num_splits, head_dim):
    total_q, heads = 257, 6
    out_partial = torch.randn(num_splits, total_q, heads, head_dim, device="npu", dtype=torch.float32)
    lse_partial = torch.randn(num_splits, heads, total_q, device="npu", dtype=torch.float32).transpose(-1, -2)
    lse_partial[num_splits // 2 :, : total_q // 3] = float("-inf")

    out, lse = flash_attn_combine(out_partial, lse_partial, out_dtype=dtype)
    assert out.shape == (total_q, heads, head_dim)
    assert lse.shape == (total_q, heads)

    out_ref, lse_ref = _combine_ref(out_partial.unsqueeze(1), lse_partial.unsqueeze(1))
    _assert_combine_close(out, lse, out_ref.squeeze(0), lse_ref.squeeze(0), dtype)


def test_combine_large_seqlen_and_dtype_default():
    num_splits, batch, seqlen, heads, head_dim = 32, 1, 1024, 8, 128
    out_partial = torch.randn(num_splits, batch, seqlen, heads, head_dim, device="npu", dtype=torch.float32)
    lse_partial = _make_lse_partial(num_splits, batch, seqlen, heads)

    # out_dtype=None must default to fp32, matching the NV interface.
    out, lse = flash_attn_combine(out_partial, lse_partial)
    assert out.dtype == torch.float32
    out_ref, lse_ref = _combine_ref(out_partial, lse_partial)
    torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(lse, lse_ref, atol=1e-4, rtol=1e-4)


def test_combine_preallocated_out_is_written_in_place():
    num_splits, batch, seqlen, heads, head_dim = 4, 1, 64, 4, 128
    out_partial = torch.randn(num_splits, batch, seqlen, heads, head_dim, device="npu", dtype=torch.float32)
    lse_partial = _make_lse_partial(num_splits, batch, seqlen, heads)
    out = torch.empty(batch, seqlen, heads, head_dim, device="npu", dtype=torch.bfloat16)
    ptr = out.data_ptr()

    result, _ = flash_attn_combine(out_partial, lse_partial, out=out, out_dtype=torch.bfloat16)
    assert result.data_ptr() == ptr
    out_ref, _ = _combine_ref(out_partial, lse_partial)
    cast_ref = out_ref.to(torch.bfloat16)
    cast_err = (cast_ref.float() - out_ref).abs().max().item()
    assert (result.float() - out_ref).abs().max().item() <= 2.0 * cast_err + 1e-3


def test_combine_all_masked_rows_produce_zero():
    num_splits, batch, seqlen, heads, head_dim = 3, 1, 8, 2, 64
    out_partial = torch.randn(num_splits, batch, seqlen, heads, head_dim, device="npu", dtype=torch.float32)
    lse_partial = _make_lse_partial(num_splits, batch, seqlen, heads)
    lse_partial[:, :, :2] = float("-inf")

    out, lse = flash_attn_combine(out_partial, lse_partial, out_dtype=torch.bfloat16)
    assert torch.isfinite(out).all()
    assert (out[:, :2].abs().max().item() == 0.0)
    assert torch.isinf(lse[:, :2]).all() and (lse[:, :2] < 0).all()


def test_combine_error_contract():
    num_splits, batch, seqlen, heads, head_dim = 2, 1, 8, 2, 64
    out_partial = torch.randn(num_splits, batch, seqlen, heads, head_dim, device="npu", dtype=torch.float32)
    lse_partial = _make_lse_partial(num_splits, batch, seqlen, heads)

    with pytest.raises(RuntimeError, match="fp32"):
        flash_attn_combine(out_partial.to(torch.float16), lse_partial, out_dtype=torch.float16)

    # Non-contiguous seqlen dimension must be rejected.
    bad_lse = torch.randn(num_splits, batch, seqlen, heads, device="npu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="seqlen"):
        flash_attn_combine(out_partial, bad_lse, out_dtype=torch.float16)

    # num_splits > 256 must be rejected.
    big_partial = torch.randn(257, 1, 1, 1, 64, device="npu", dtype=torch.float32)
    big_lse = torch.randn(257, 1, 1, 1, device="npu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="256"):
        flash_attn_combine(big_partial, big_lse, out_dtype=torch.float16)

    # out dtype must match out_dtype.
    out = torch.empty(batch, seqlen, heads, head_dim, device="npu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="dtype"):
        flash_attn_combine(out_partial, lse_partial, out=out, out_dtype=torch.float16)
