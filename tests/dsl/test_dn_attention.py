"""Coordinate and numerical contracts for the query-lane score layout."""

from importlib.util import find_spec
import os

import pytest

if find_spec("catlass") is None:
    pytest.skip("CATLASS DSL runtime is not installed", allow_module_level=True)

import catlass.tla as tla
from flash_attn_npu_dsl import BlockSparseTensorsTorch, flash_attn_func, simd
from flash_attn_npu_dsl import interface

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"),
    reason="set FLASH_ATTN_DSL_TEST_DEVICE to opt into NPU execution",
)


@simd
@tla.jit
def causal(b, h, q, kv, info, tensors):
    return tla.cmp(kv, q + info.seqlen_k - info.seqlen_q, "le")


@simd
def coordinate_score(score, b, h, *, q_idx, kv_idx, seqlen_info, aux_tensors):
    cast = tla.params.CastParams()
    return (
        score * 0.75
        + b.to(tla.Float32, cast) * 0.005
        + h.to(tla.Float32, cast) * 0.01
        + q_idx.to(tla.Float32, cast) * 0.02
        - kv_idx.to(tla.Float32, cast) * 0.03
    )


@pytest.mark.parametrize("dtype,dim,kv_heads", [(torch.float16, 64, 1), (torch.bfloat16, 96, 3)])
def test_dn_packed_coordinates_sparse_rebinding(dtype, dim, kv_heads):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    batch, sq, sk, heads = 64, 4, 388, 32 * kv_heads
    generator = torch.Generator().manual_seed(17)
    host = tuple(
        torch.randn(shape, generator=generator).to(dtype)
        for shape in (
            (batch, sq, heads, dim),
            (batch, sk, kv_heads, dim),
            (batch, sk, kv_heads, dim),
        )
    )
    inputs = tuple(value.to(device) for value in host)
    q, k, v = (value.float().permute(0, 2, 1, 3) for value in host)
    k, v = k.repeat_interleave(32, dim=1), v.repeat_interleave(32, dim=1)
    scores = q @ k.transpose(-1, -2) * dim**-0.5 * 0.75
    scores += (
        torch.arange(batch)[:, None, None, None] * 0.005
        + torch.arange(heads)[None, :, None, None] * 0.01
        + torch.arange(sq)[None, None, :, None] * 0.02
        - torch.arange(sk)[None, None, None, :] * 0.03
    )
    pc = torch.ones(batch, 1, 1, dtype=torch.int32)
    fc = torch.full_like(pc, 2)
    pc[0] = fc[0] = 0
    blocks = BlockSparseTensorsTorch(
        pc.to(device),
        torch.tensor([[[[3]]]], dtype=torch.int32, device=device),
        fc.to(device),
        torch.tensor([[[[1, 0]]]], dtype=torch.int32, device=device),
        block_size=(128, 128),
    )
    interface._compiled_kernels.clear()
    try:
        for iteration in range(3):
            if iteration == 1:
                fc[1] = 1
                blocks.full_block_cnt.copy_(fc)
            elif iteration == 2:
                blocks = blocks._replace(
                    **{name: value.clone() for name, value in zip(blocks._fields[:4], blocks[:4])}
                )
                inputs = tuple(value.clone() for value in inputs)
            visible = torch.zeros(batch, 1, sq, sk, dtype=torch.bool)
            for b in range(batch):
                if pc[b, 0, 0]:
                    visible[b, 0, :, 384:] = torch.arange(4)[None, :] <= torch.arange(sq)[:, None]
                if fc[b, 0, 0] >= 1:
                    visible[b, 0, :, 128:256] = True
                if fc[b, 0, 0] == 2:
                    visible[b, 0, :, :128] = True
            masked = scores.masked_fill(~visible, -torch.inf)
            expected_lse = masked.logsumexp(-1)
            expected = (masked.softmax(-1).nan_to_num() @ v).permute(0, 2, 1, 3)
            for lse_requested in (False, True):
                out, lse = flash_attn_func(
                    *inputs,
                    mask_mod=causal,
                    score_mod=coordinate_score,
                    block_sparse_tensors=blocks,
                    return_lse=lse_requested,
                )
                torch.npu.synchronize()
                torch.testing.assert_close(out.cpu().float(), expected, rtol=0, atol=0.05)
                assert torch.equal(out[0].cpu(), torch.zeros_like(out[0].cpu()))
                if lse_requested:
                    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0, atol=0.05)
                else:
                    assert lse is None
                constants = next(reversed(interface._compiled_kernels))[1][-1]
                assert constants[-2][1] is True and constants[-1][1] is False
            assert len(interface._compiled_kernels) == 2
    finally:
        interface._compiled_kernels.clear()


@pytest.mark.parametrize("scale", [0.125, 0.0, -0.125])
@pytest.mark.parametrize("q_len,kv_len", [(128, 256), (513, 512)])
def test_dn_large_negative_and_nonpositive_scale(scale, q_len, kv_len):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    q = torch.full((1, q_len, 1, 64), -100.0, dtype=torch.float16, device=device)
    k = torch.full((1, kv_len, 1, 64), 100.0, dtype=torch.float16, device=device)
    v = torch.ones_like(k)
    out, lse = flash_attn_func(q, k, v, softmax_scale=scale, return_lse=True)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.ones_like(out.cpu()), rtol=0, atol=0.05)
    expected = torch.tensor(-640000.0 * scale, dtype=torch.float32) + torch.log(
        torch.tensor(float(kv_len))
    )
    torch.testing.assert_close(lse.cpu(), expected.expand_as(lse.cpu()), rtol=0, atol=0.05)


@pytest.mark.parametrize("dtype,dim", [(torch.float16, 16), (torch.bfloat16, 128)])
def test_dn_query_tails_preserve_empty_rows_and_dynamic_reuse(dtype, dim):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(29)
    interface._compiled_kernels.clear()
    try:
        # Exercise one inactive AIV, both AIVs with partial rows, and an aligned
        # tile, then return to the first lengths using new buffers and data.
        for sq, sk in ((513, 513), (543, 513), (577, 1025), (640, 768), (513, 513)):
            host = tuple(
                torch.randn(shape, generator=generator).to(dtype)
                for shape in ((2, sq, 4, dim), (2, sk, 2, dim), (2, sk, 2, dim))
            )
            q, k, v = (tensor.float().permute(0, 2, 1, 3) for tensor in host)
            k, v = k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1)
            scores = q @ k.transpose(-1, -2) * dim**-0.5
            keep = torch.arange(sk)[None, :] == torch.arange(sq)[:, None] + sk - sq
            scores.masked_fill_(~keep, -torch.inf)
            expected_lse = scores.logsumexp(-1)
            expected = (scores.softmax(-1).nan_to_num() @ v).permute(0, 2, 1, 3)
            inputs = tuple(tensor.to(device) for tensor in host)
            for return_lse in (False, True):
                out, lse = flash_attn_func(*inputs, window_size=(0, 0), return_lse=return_lse)
                torch.npu.synchronize()
                torch.testing.assert_close(out.cpu().float(), expected, rtol=0, atol=0.05)
                assert torch.count_nonzero(out.cpu()[:, ~keep.any(-1)]) == 0
                if return_lse:
                    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0, atol=0.05)
                else:
                    assert lse is None
                constants = next(reversed(interface._compiled_kernels))[1][-1]
                assert constants[-2][1] is True and constants[-1][1] is False
            assert len(interface._compiled_kernels) == 2
    finally:
        interface._compiled_kernels.clear()
