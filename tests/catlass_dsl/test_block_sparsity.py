"""Exact/sampled metadata, sparse forward/backward and packed layouts."""

from functools import lru_cache

import catlass.tla as tla
import pytest
import torch
from .test_utils import DTYPES, check_attention, make_inputs
from .reference import reference_block_sparsity

from flash_attn_npu_4.catlass_dsl import (
    compute_block_sparsity,
    flash_attn_func,
    flash_attn_varlen_func,
    simt,
)
from flash_attn_npu_4.catlass_dsl.block_sparsity import (
    BlockSparseTensorsTorch,
    normalize_block_sparse_tensors,
    sparse_tensor_versions,
    transpose_block_sparsity,
    validate_sparse_versions,
)


@lru_cache(maxsize=8)
def mask_pair(kind):
    @simt
    @tla.jit
    def native(b, h, q, k, info, aux, scalars):
        visible = tla.Bool(False)
        if tla.const_expr(kind == "all"):
            visible = tla.Bool(True)
        elif tla.const_expr(kind == "causal"):
            visible = q >= k
        elif tla.const_expr(kind == "window"):
            visible = (q >= k) & (q - k < 33)
        elif tla.const_expr(kind == "mixed"):
            visible = ((q // 128 == k // 128) | (k // 128 == 2)) & (q >= 13)
        return visible

    def reference(b, h, q, k, info, aux, scalars):
        if kind == "all":
            return torch.ones_like(q + k, dtype=torch.bool)
        if kind == "causal":
            return q >= k
        if kind == "window":
            return (q >= k) & (q - k < 33)
        if kind == "mixed":
            return ((q // 128 == k // 128) | (k // 128 == 2)) & (q >= 13)
        return torch.zeros_like(q + k, dtype=torch.bool)

    return native, reference


def active_lists(tensors, full=False):
    counts = (tensors.full_block_cnt if full else tensors.mask_block_cnt).cpu()
    indices = (tensors.full_block_idx if full else tensors.mask_block_idx).cpu()
    rows = indices.reshape(counts.numel(), indices.shape[-1])
    return [sorted(row[: int(count)].tolist()) for count, row in zip(counts.flatten(), rows)]


@pytest.mark.npu
@pytest.mark.parametrize("kind", ["all", "none", "causal", "window", "mixed"])
@pytest.mark.parametrize("compute_full", [False, True])
def test_exact_block_classification(device, kind, compute_full):
    native, reference = mask_pair(kind)
    args = (128, 128, 1, 1, 257, 260, native, None, device)
    if kind == "all":
        blocks = compute_block_sparsity(
            *args, None, None, None, None, None, None, None, compute_full, False
        )
    else:
        blocks = compute_block_sparsity(*args, compute_full_blocks=compute_full)
    expected = reference_block_sparsity(1, 1, 257, 260, reference, compute_full_blocks=compute_full)
    assert blocks.mask_block_cnt.device == device
    assert active_lists(blocks) == active_lists(expected)
    if compute_full:
        assert active_lists(blocks, full=True) == active_lists(expected, full=True)
    else:
        assert blocks.full_block_cnt is blocks.full_block_idx is None
    if kind == "causal":
        sampled = compute_block_sparsity(
            128,
            128,
            1,
            1,
            257,
            260,
            native,
            aux_tensors=None,
            device=device,
            compute_full_blocks=compute_full,
            use_fast_sampling=True,
        )
        assert active_lists(sampled) == active_lists(expected)
        if compute_full:
            assert active_lists(sampled, full=True) == active_lists(expected, full=True)


def test_transpose_roundtrip_and_sampling():
    _, reference = mask_pair("causal")
    args = (2, 2, 257, 260, reference)
    exact = reference_block_sparsity(*args)
    sampled = reference_block_sparsity(*args, use_fast_sampling=True)
    transposed = transpose_block_sparsity(
        exact, batch_size=2, num_heads=2, seqlen_q=257, seqlen_k=260
    )
    restored = transpose_block_sparsity(
        transposed, batch_size=2, num_heads=2, seqlen_q=260, seqlen_k=257
    )
    for full in (False, True):
        assert (
            active_lists(exact, full) == active_lists(sampled, full) == active_lists(restored, full)
        )


@pytest.mark.parametrize(
    "changes", [{"block_size": (64, 128)}, {"dq_write_order": torch.zeros(1, dtype=torch.int32)}]
)
def test_unsupported_sparse_metadata(changes):
    _, reference = mask_pair("causal")
    blocks = reference_block_sparsity(1, 1, 17, 23, reference)
    with pytest.raises(NotImplementedError):
        transpose_block_sparsity(
            blocks._replace(**changes), batch_size=1, num_heads=1, seqlen_q=17, seqlen_k=23
        )


@pytest.mark.parametrize(
    "qp,kp,compute_full",
    [(False, False, True), (True, False, False), (False, True, True), (True, True, True)],
)
def test_normalized_fixed_and_packed_lists(qp, kp, compute_full):
    _, reference = mask_pair("causal")
    cq = torch.tensor([0, 0, 129, 266], dtype=torch.int32) if qp else None
    ck = torch.tensor([0, 260, 260, 515], dtype=torch.int32) if kp else None
    dimensions = dict(
        batch_size=3, num_heads=1, seqlen_q=137, seqlen_k=260, cu_seqlens_q=cq, cu_seqlens_k=ck
    )
    blocks = reference_block_sparsity(
        3,
        1,
        137,
        260,
        reference,
        cu_seqlens_q=cq,
        cu_seqlens_k=ck,
        compute_full_blocks=compute_full,
    )
    normalized = normalize_block_sparse_tensors(blocks, device=torch.device("cpu"), **dimensions)
    assert isinstance(normalized, BlockSparseTensorsTorch)
    assert normalized.mask_block_cnt.shape == normalized.full_block_cnt.shape == (3, 1, 2)
    assert normalized.mask_block_idx.shape == normalized.full_block_idx.shape
    assert normalized.cu_total_m_blocks is normalized.cu_block_idx_offsets is None
    if qp:
        assert not normalized.mask_block_cnt[0].any() and not normalized.full_block_cnt[0].any()
    if kp:
        assert not normalized.mask_block_cnt[1].any() and not normalized.full_block_cnt[1].any()
    backward = transpose_block_sparsity(blocks, **dimensions)
    restored = normalize_block_sparse_tensors(
        backward, device=torch.device("cpu"), backward=True, **dimensions
    )
    for full in (False, True):
        assert active_lists(restored, full) == active_lists(normalized, full)


def test_sparse_versions_and_cache_invalidation():
    blocks = BlockSparseTensorsTorch(
        torch.ones(1, 1, 2, dtype=torch.int32),
        torch.tensor([[[[0], [2]]]], dtype=torch.int32),
    )
    dimensions = dict(
        batch_size=1, num_heads=1, seqlen_q=129, seqlen_k=257, device=torch.device("cpu")
    )
    normalized = normalize_block_sparse_tensors(blocks, **dimensions)
    assert normalize_block_sparse_tensors(blocks, **dimensions) is normalized
    versions = sparse_tensor_versions(blocks, normalized)
    # Missing full lists are synthesized; their mutation must invalidate a cache hit.
    normalized.full_block_cnt.add_(1)
    with pytest.raises(ValueError, match="changed in-place"):
        validate_sparse_versions(versions)
    rebuilt = normalize_block_sparse_tensors(blocks, **dimensions)
    assert rebuilt is not normalized and not rebuilt.full_block_cnt.any()
    versions = sparse_tensor_versions(blocks, rebuilt)
    blocks.mask_block_idx[0, 0, 1, 0] = 1
    with pytest.raises(ValueError, match="changed in-place"):
        validate_sparse_versions(versions)
    updated = normalize_block_sparse_tensors(blocks, **dimensions)
    assert active_lists(updated) == [[0], [1]]


@pytest.mark.npu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "qp,kp,compute_full",
    [
        (False, False, True),
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_sparse_forward_backward(device, dtype, qp, kp, compute_full):
    cpu, inputs, cq, ck, kwargs = make_inputs(
        device,
        dtype,
        qp=qp,
        kp=kp,
        sq=137,
        sk=385,
        q_lengths=(0, 137, 134),
        k_lengths=(385, 0, 380),
    )
    native, reference = mask_pair("mixed")
    blocks = compute_block_sparsity(
        128,
        128,
        3,
        4,
        137,
        385,
        native,
        aux_tensors=None,
        device=device,
        **kwargs,
        compute_full_blocks=compute_full,
    )
    options = dict(mask_mod=native, block_sparse_tensors=blocks, return_lse=True)
    if qp or kp:
        out, lse = flash_attn_varlen_func(
            *inputs, **kwargs, max_seqlen_q=137, max_seqlen_k=385, **options
        )
    else:
        backward = transpose_block_sparsity(
            blocks, batch_size=3, num_heads=4, seqlen_q=137, seqlen_k=385
        )
        out, lse = flash_attn_func(*inputs, block_sparse_tensors_bwd=backward, **options)
    check_attention(cpu, inputs, out, lse, cq=cq, ck=ck, mask_mod=reference, loss="both")


@pytest.mark.npu
@pytest.mark.parametrize("mutate_backward", [False, True])
def test_sparse_mutation_between_forward_and_backward(device, mutate_backward):
    _, inputs, _, _, _ = make_inputs(device, torch.float16)
    blocks = compute_block_sparsity(128, 128, 1, 1, 17, 23, None, aux_tensors=None, device=device)
    backward = transpose_block_sparsity(blocks, batch_size=2, num_heads=4, seqlen_q=17, seqlen_k=23)
    out, _ = flash_attn_func(
        *inputs, block_sparse_tensors=blocks, block_sparse_tensors_bwd=backward
    )
    (backward if mutate_backward else blocks).mask_block_cnt.add_(0)
    with pytest.raises(ValueError, match="changed in-place"):
        out.sum().backward()


@pytest.mark.npu
def test_sparse_subset_with_builtin_mask(device):
    cpu, inputs, _, _, _ = make_inputs(device, torch.float16, sq=256, sk=256)
    blocks = BlockSparseTensorsTorch(
        torch.ones(1, 1, 2, dtype=torch.int32, device=device),
        torch.zeros(1, 1, 2, 1, dtype=torch.int32, device=device),
    )
    out, lse = flash_attn_func(*inputs, causal=True, block_sparse_tensors=blocks, return_lse=True)
    reference = lambda b, h, q, k, info, aux, scalars: (q >= k) & (k < 128)
    check_attention(cpu, inputs, out, lse, mask_mod=reference, loss="both")


@pytest.mark.npu
def test_packed_sparse_cache_across_streams(device):
    cpu, inputs, cq, ck, kwargs = make_inputs(
        device, torch.float16, qp=True, kp=True, sq=137, sk=260
    )
    native, reference = mask_pair("causal")
    blocks = compute_block_sparsity(
        128, 128, 2, 1, 137, 260, native, aux_tensors=None, device=device, **kwargs
    )
    producer, consumer = torch.npu.Stream(device=device), torch.npu.Stream(device=device)
    producer.wait_stream(torch.npu.current_stream(device))
    consumer.wait_stream(torch.npu.current_stream(device))
    with torch.npu.stream(producer):
        normalize_block_sparse_tensors(
            blocks,
            batch_size=2,
            num_heads=4,
            seqlen_q=137,
            seqlen_k=260,
            device=device,
            **kwargs,
        )
    with torch.npu.stream(consumer):
        out, lse = flash_attn_varlen_func(
            *inputs,
            **kwargs,
            max_seqlen_q=137,
            max_seqlen_k=260,
            mask_mod=native,
            block_sparse_tensors=blocks,
            return_lse=True,
        )
        check_attention(cpu, inputs, out, lse, cq=cq, ck=ck, mask_mod=reference, loss="both")
    torch.npu.current_stream(device).wait_stream(consumer)
