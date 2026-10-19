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


@simd
def _strict_causal(b, h, q, kv, info, tensors):
    return tla.cmp(kv, q + info.seqlen_k - info.seqlen_q, "lt")


def _scalar_strict_causal(b, h, q, kv, info, tensors):
    return kv < q + info.seqlen_k - info.seqlen_q + tensors[0][b]


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
                assert constants[16][1] is True and constants[17][1] is False
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
                assert constants[16][1] is True and constants[17][1] is False
            assert len(interface._compiled_kernels) == 2
    finally:
        interface._compiled_kernels.clear()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("mask", [_strict_causal, _scalar_strict_causal])
def test_dn_partial_then_full_preserves_empty_rows(dtype, return_lse, mask):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(29)
    host = tuple(
        torch.randn(shape, generator=generator).to(dtype)
        for shape in ((2, 128, 2, 64), (2, 256, 1, 64), (2, 256, 1, 64))
    )
    q, k, v = (value.float().permute(0, 2, 1, 3) for value in host)
    k, v = (value.repeat_interleave(2, dim=1) for value in (k, v))
    scores = (q @ k.transpose(-1, -2)) / 8
    inputs = tuple(value.to(device) for value in host)
    offsets = torch.tensor([0, 1] if mask is _scalar_strict_causal else [0, 0], dtype=torch.int32)
    aux = (offsets.to(device),) if mask is _scalar_strict_causal else None
    blocks = BlockSparseTensorsTorch(
        torch.ones((1, 1, 1), dtype=torch.int32, device=device),
        torch.ones((1, 1, 1, 1), dtype=torch.int32, device=device),
        torch.ones((1, 1, 1), dtype=torch.int32, device=device),
        torch.zeros((1, 1, 1, 1), dtype=torch.int32, device=device),
        block_size=(128, 128),
    )
    interface._compiled_kernels.clear()
    executable = None
    try:
        # Partial block 1 is visited before full block 0. The first query has
        # no visible value in the partial block, so the full update starts
        # from max=-inf and sum=0, not from an ordinary finite softmax state.
        for iteration, (partial, full) in enumerate(((1, 1), (1, 0), (0, 1), (0, 0), (1, 1))):
            blocks.mask_block_cnt.fill_(partial)
            blocks.full_block_cnt.fill_(full)
            if iteration == 1 and aux is not None:
                offsets = offsets.flip(0)
                aux[0].copy_(offsets)
            if iteration == 4:
                inputs = tuple(value.clone() for value in inputs)
                if aux is not None:
                    aux = (aux[0].clone(),)
            visible = torch.arange(256)[None, None, :] < (
                torch.arange(128)[None, :, None] + 128 + offsets[:, None, None]
            )
            if not partial:
                visible[:, :, 128:] = False
            if not full:
                visible[:, :, :128] = False
            masked = scores.masked_fill(~visible[:, None], -torch.inf)
            expected = (masked.softmax(-1).nan_to_num(0) @ v).permute(0, 2, 1, 3)
            out, lse = flash_attn_func(
                *inputs,
                mask_mod=mask,
                aux_tensors=aux,
                block_sparse_tensors=blocks,
                return_lse=return_lse,
            )
            torch.npu.synchronize()
            actual = out.cpu().float()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0.05)
            assert torch.count_nonzero(actual[~visible.any(-1)]) == 0
            if return_lse:
                torch.testing.assert_close(lse.cpu(), masked.logsumexp(-1), rtol=0, atol=0.05)
            else:
                assert lse is None
            assert len(interface._compiled_kernels) == 1
            current = next(iter(interface._compiled_kernels.values()))
            if executable is None:
                executable = current
            assert current is executable
    finally:
        interface._compiled_kernels.clear()


@pytest.mark.parametrize(
    "dtype,dim,window",
    [
        (torch.float16, 16, (None, 0)),
        (torch.bfloat16, 128, (256, 0)),
        (torch.float16, 64, (0, 256)),
        (torch.bfloat16, 96, (129, 64)),
    ],
)
def test_dn_intrinsic_full_and_boundary_tiles_reuse_dynamic_lengths(dtype, dim, window):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(29)
    interface._compiled_kernels.clear()
    executables = {}
    try:
        # Cross full/partial tiles, rectangular right alignment and empty rows,
        # then revisit the first shape with new data and storage addresses.
        for sq, sk in ((256, 512), (513, 769), (640, 384), (256, 512)):
            host = tuple(
                torch.randn(shape, generator=generator).to(dtype)
                for shape in ((2, sq, 4, dim), (2, sk, 2, dim), (2, sk, 2, dim))
            )
            q, k, v = (tensor.float().permute(0, 2, 1, 3) for tensor in host)
            k, v = k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1)
            scores = q @ k.transpose(-1, -2) * dim**-0.5
            center = torch.arange(sq)[:, None] + sk - sq
            keys = torch.arange(sk)[None, :]
            keep = torch.ones(sq, sk, dtype=torch.bool)
            if window[0] is not None:
                keep &= keys >= center - window[0]
            if window[1] is not None:
                keep &= keys <= center + window[1]
            scores.masked_fill_(~keep, -torch.inf)
            expected_lse = scores.logsumexp(-1)
            expected = (scores.softmax(-1).nan_to_num() @ v).permute(0, 2, 1, 3)
            inputs = tuple(tensor.to(device) for tensor in host)
            for return_lse in (False, True):
                out, lse = flash_attn_func(*inputs, window_size=window, return_lse=return_lse)
                torch.npu.synchronize()
                actual = out.cpu().float()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0.05)
                assert torch.count_nonzero(actual[:, ~keep.any(-1)]) == 0
                if return_lse:
                    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0, atol=0.05)
                else:
                    assert lse is None
                executable = next(reversed(interface._compiled_kernels.values()))
                if return_lse not in executables:
                    executables[return_lse] = executable
                assert executable is executables[return_lse]
            assert len(interface._compiled_kernels) == 2
    finally:
        interface._compiled_kernels.clear()
