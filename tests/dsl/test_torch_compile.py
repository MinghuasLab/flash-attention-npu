"""Fullgraph treats the DSL kernel as an opaque op, not as Python to inline."""

from importlib.util import find_spec
import os

import pytest

if find_spec("catlass") is None or find_spec("torch") is None:
    pytest.skip("optional CATLASS DSL / PyTorch runtime is absent", allow_module_level=True)

import torch
from catlass import tla
from flash_attn_npu_dsl import flash_attn_func, flash_attn_varlen_func, compute_block_sparsity, simd


@tla.jit
def _scalar_mask(b, h, q, kv, info, aux, aux_scalars):
    return (kv <= q + info.seqlen_k - info.seqlen_q) & (kv >= aux[0][b])


@simd
def _vector_score(s, b, h, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
    return s * aux_scalars[0]


@pytest.mark.skipif(not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"), reason="explicit NPU opt-in")
@pytest.mark.parametrize(
    "layout,sparse",
    [("BSND", False), ("BSND", True), ("BSND-used", True), ("TND", True), ("paged", True)],
)
@pytest.mark.parametrize("return_lse", [True, False])
def test_fullgraph_rebinds_inputs_metadata_and_auxiliary(layout, sparse, return_lse):
    import torch_npu  # noqa: F401

    # Parametrized tests share one Python code object, not one application graph.
    torch._dynamo.reset()
    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(29)
    q = torch.randn(2, 3, 4, 64, generator=gen).half()
    k = torch.randn(2, 129, 2, 64, generator=gen).half()
    v = torch.randn(2, 129, 2, 96, generator=gen).half()
    if layout in ("TND", "paged"):
        q = q.flatten(0, 1)
    if layout == "TND":
        k, v = k.flatten(0, 1), v.flatten(0, 1)
    elif layout == "paged":
        k = torch.randn(4, 128, 2, 64, generator=gen).half()
        v = torch.randn(4, 128, 2, 96, generator=gen).half()
    inputs = [x.to(device) for x in (q, k, v)]
    cuq = (
        torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
        if layout in ("TND", "paged")
        else None
    )
    cuk = torch.tensor([0, 129, 258], dtype=torch.int32, device=device) if layout == "TND" else None
    used = (
        torch.tensor([129, 129], dtype=torch.int32, device=device)
        if layout in ("paged", "BSND-used")
        else None
    )
    pages = (
        torch.tensor([[2, 0], [0, 3]], dtype=torch.int32, device=device)
        if layout == "paged"
        else None
    )
    threshold = torch.zeros(2, dtype=torch.int32, device=device)

    def invoke(q, k, v, threshold, scale, cuq, cuk, used, pages, blocks):
        kwargs = dict(
            mask_mod=_scalar_mask,
            score_mod=_vector_score,
            aux_tensors=[threshold],
            aux_scalars=(scale,),
            block_sparse_tensors=blocks,
            return_lse=return_lse,
            pack_gqa=False,
        )
        if layout == "BSND":
            return flash_attn_func(q, k, v, **kwargs)
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cuq,
            cu_seqlens_k=cuk,
            seqused_k=used,
            page_table=pages,
            max_seqlen_q=3,
            max_seqlen_k=129,
            **kwargs,
        )

    compiled = torch.compile(invoke, backend="inductor", fullgraph=True)
    for changed in (False, True):
        scale = 0.5 if changed else 0.75
        if changed:
            threshold = torch.tensor([128, 129], dtype=torch.int32, device=device)
            inputs = [x.clone() for x in inputs]
            if used is not None:
                used = torch.tensor([129, 0], dtype=torch.int32, device=device)
                if pages is not None:
                    pages = pages.flip(1).contiguous()
            if cuq is not None:
                cuq = cuq.clone()
        blocks = None
        if sparse:
            blocks = compute_block_sparsity(
                128,
                128,
                2,
                1,
                3,
                129,
                _scalar_mask,
                [threshold],
                device,
                aux_scalars=(scale,),
                cu_seqlens_q=cuq,
                cu_seqlens_k=cuk,
                seqused_k=used,
            )
        args = (*inputs, threshold, scale, cuq, cuk, used, pages, blocks)
        expected = invoke(*args)
        actual = compiled(*args)
        torch.npu.synchronize()
        assert actual[0].shape == expected[0].shape
        assert actual[0].data_ptr() != expected[0].data_ptr()
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
        if return_lse:
            torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
        else:
            assert actual[1] is None


@pytest.mark.skipif(not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"), reason="explicit NPU opt-in")
def test_fullgraph_dynamic_lengths_reuse_the_dsl_executable():
    import torch_npu  # noqa: F401
    from flash_attn_npu_dsl import interface

    torch._dynamo.reset()
    interface._compiled_kernels.clear()
    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)

    def forward(q, k, v):
        # Exercise actual Inductor-generated NPU work on both sides of the
        # opaque DSL call, including its producer/consumer stream dependencies.
        out, lse = flash_attn_func(q * 0.75, k, v, causal=True, return_lse=True)
        return out + 0.25, lse

    compiled = torch.compile(forward, backend="inductor", fullgraph=True, dynamic=True)
    gen = torch.Generator().manual_seed(17)
    for sq, sk in ((17, 129), (65, 257)):
        inputs = [
            torch.randn(s, generator=gen).half().to(device)
            for s in ((1, sq, 4, 64), (1, sk, 2, 64), (1, sk, 2, 64))
        ]
        expected = forward(*inputs)
        actual = compiled(*inputs)
        torch.npu.synchronize()
        assert len(interface._compiled_kernels) == 1
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
