# Copyright (c) 2026, Huawei Technologies Co., Ltd.

"""FA4 split-attention combine (flash_attn_combine) tests.

Covers the FA4-only surface: cu_seqlens / seqused / return_lse, plus the
virtual_batch_idx not-supported contract.
"""

import pytest
import torch
import torch_npu

from flash_attn_npu_4 import flash_attn_combine


def _combine_ref(out_partial, lse_partial):
    lse = torch.logsumexp(lse_partial, dim=0)
    scale = torch.exp(lse_partial - lse)
    scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
    return (scale.unsqueeze(-1) * out_partial).sum(0), lse


def _assert_close(out, lse, out_ref, lse_ref, dtype):
    torch.testing.assert_close(lse.float(), lse_ref, atol=1e-4, rtol=1e-4)
    cast_ref = out_ref.to(dtype)
    cast_err = (cast_ref.float() - out_ref).abs().max().item()
    diff = (out.float() - out_ref).abs().max().item()
    assert diff <= 2.0 * cast_err + 1e-3, f"out max diff {diff:.6f} vs cast err {cast_err:.6f}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_splits", [2, 5, 16])
@pytest.mark.parametrize("head_dim", [64, 96, 128, 256])
def test_combine_dense_with_seqused(dtype, num_splits, head_dim):
    batch, seqlen, heads = 3, 64, 4
    out_partial = torch.randn(num_splits, batch, seqlen, heads, head_dim, device="npu", dtype=torch.float32)
    lse_partial = torch.randn(num_splits, batch, heads, seqlen, device="npu", dtype=torch.float32).transpose(-1, -2)
    seqused = torch.tensor([64, 33, 1], device="npu", dtype=torch.int32)
    lse_partial[num_splits // 2 :, 1] = float("-inf")

    out, lse = flash_attn_combine(out_partial, lse_partial, out_dtype=dtype, seqused=seqused)
    assert out.shape == (batch, seqlen, heads, head_dim)
    assert lse.shape == (batch, seqlen, heads)
    out_ref, lse_ref = _combine_ref(out_partial, lse_partial)
    _assert_close(out, lse, out_ref, lse_ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_splits", [2, 3, 32])
def test_combine_varlen_with_cu_seqlens(dtype, num_splits):
    seqlens = [17, 1, 64, 33]
    heads, head_dim = 2, 128
    cu = torch.tensor([0, 17, 18, 82, 115], device="npu", dtype=torch.int32)
    total_q = int(cu[-1].item())
    out_partial = torch.randn(num_splits, total_q, heads, head_dim, device="npu", dtype=torch.float32)
    lse_partial = torch.randn(num_splits, heads, total_q, device="npu", dtype=torch.float32).transpose(-1, -2)
    lse_partial[num_splits // 2 :, : total_q // 4] = float("-inf")
    seqused = torch.tensor(seqlens, device="npu", dtype=torch.int32)

    out, lse = flash_attn_combine(
        out_partial, lse_partial, out_dtype=dtype, cu_seqlens=cu, seqused=seqused
    )
    assert out.shape == (total_q, heads, head_dim)
    assert lse.shape == (total_q, heads)
    out_ref, lse_ref = _combine_ref(out_partial.unsqueeze(1), lse_partial.unsqueeze(1))
    _assert_close(out, lse, out_ref.squeeze(0), lse_ref.squeeze(0), dtype)


def test_combine_return_lse_false_and_inplace():
    num_splits, batch, seqlen, heads, head_dim = 4, 1, 32, 2, 64
    out_partial = torch.randn(num_splits, batch, seqlen, heads, head_dim, device="npu", dtype=torch.float32)
    lse_partial = torch.randn(num_splits, batch, heads, seqlen, device="npu", dtype=torch.float32).transpose(-1, -2)
    out = torch.empty(batch, seqlen, heads, head_dim, device="npu", dtype=torch.bfloat16)
    ptr = out.data_ptr()

    result, lse = flash_attn_combine(
        out_partial, lse_partial, out=out, out_dtype=torch.bfloat16, return_lse=False
    )
    assert result.data_ptr() == ptr
    assert lse is None
    out_ref, _ = _combine_ref(out_partial, lse_partial)
    cast_ref = out_ref.to(torch.bfloat16)
    cast_err = (cast_ref.float() - out_ref).abs().max().item()
    assert (result.float() - out_ref).abs().max().item() <= 2.0 * cast_err + 1e-3


def test_combine_virtual_batch_idx_not_supported():
    num_splits, batch, seqlen, heads, head_dim = 2, 1, 8, 2, 64
    out_partial = torch.randn(num_splits, batch, seqlen, heads, head_dim, device="npu", dtype=torch.float32)
    lse_partial = torch.randn(num_splits, batch, heads, seqlen, device="npu", dtype=torch.float32).transpose(-1, -2)
    with pytest.raises(NotImplementedError, match="virtual_batch_idx"):
        flash_attn_combine(
            out_partial, lse_partial, virtual_batch_idx=torch.arange(batch, device="npu", dtype=torch.int32)
        )


def test_combine_error_contract():
    num_splits, batch, seqlen, heads, head_dim = 2, 1, 8, 2, 64
    out_partial = torch.randn(num_splits, batch, seqlen, heads, head_dim, device="npu", dtype=torch.float32)
    lse_partial = torch.randn(num_splits, batch, heads, seqlen, device="npu", dtype=torch.float32).transpose(-1, -2)
    with pytest.raises(RuntimeError, match="fp32"):
        flash_attn_combine(out_partial.bfloat16(), lse_partial, out_dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="seqlen"):
        flash_attn_combine(
            out_partial,
            torch.randn(num_splits, batch, seqlen, heads, device="npu", dtype=torch.float32),
            out_dtype=torch.bfloat16,
        )
