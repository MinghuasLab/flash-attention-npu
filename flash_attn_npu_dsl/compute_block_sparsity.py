# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Exact device-side classification for explicit Flex Attention block sparsity."""

from __future__ import annotations

from typing import Any, Callable

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

from .block_sparsity import (
    BlockSparseTensorsTorch,
    _dimensions,
    _require_npu,
)
from .modifiers import (
    _validate_mask_vector,
    call_mask_mod,
    uses_simd,
    validate_callback,
)
from .interface import _compilation_key, _compile_kernel, _validate_aux
from .seqlen_info import create_seqlen_info


@tla.kernel
def flex_attention_classify_blocks_kernel(
    mask_counts: tla.Tensor,
    mask_indices: tla.Tensor,
    full_counts: tla.Tensor,
    full_indices: tla.Tensor,
    mask_mod: tla.Constexpr[Callable[..., Any]],
    num_heads_q: tla.Constexpr[int],
    q_tiles: tla.Constexpr[int],
    kv_tiles: tla.Constexpr[int],
    seqlen_q: tla.Constexpr[int],
    seqlen_k: tla.Constexpr[int],
    aux_tensors: Any,
    aux_scalars: Any,
) -> None:
    """Exact any/all classification with O(128) scratch, not a dense mask.

    One SIMT block owns one [batch, query-head, query-tile] row. All threads
    join all barriers, including physical tails. Invalid coordinates are
    clamped before callback invocation and excluded from the visible count.
    """
    work = tla.arch.block_idx()
    row_count_ptr = tla.allocate(128, tla.Int32, tla.AddressSpace.ub, 256)
    row_counts = tla.make_tensor(
        row_count_ptr, tla.make_layout(tla.make_shape(128), tla.make_stride(1))
    )
    info = create_seqlen_info(tla.Int32(seqlen_q), tla.Int32(seqlen_k))
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=128):
            tid, _, _ = tla.arch.thread_idx()
            q_tile = work % q_tiles
            head = (work // q_tiles) % num_heads_q
            batch = work // (q_tiles * num_heads_q)
            q_idx = q_tile * 128 + tid
            q_in_bounds = q_idx < seqlen_q
            safe_q = tla.where(q_in_bounds, q_idx, tla.Int32(seqlen_q - 1))
            if tid == 0:
                mask_counts[work] = 0
                full_counts[work] = 0
                for slot in tla.range(0, kv_tiles, 1):
                    mask_indices[work, slot] = 0
                    full_indices[work, slot] = 0
            tla.arch.sync_threads()

            for kv_tile in tla.range(0, kv_tiles, 1):
                visible_count = tla.Int32(0)
                for local_k in tla.range(0, 128, 1):
                    kv_idx = kv_tile * 128 + local_k
                    kv_in_bounds = kv_idx < seqlen_k
                    safe_k = tla.where(kv_in_bounds, kv_idx, tla.Int32(seqlen_k - 1))
                    keep = call_mask_mod(
                        mask_mod,
                        batch,
                        head,
                        safe_q,
                        safe_k,
                        info,
                        aux_tensors,
                        aux_scalars,
                    )
                    visible_count = visible_count + tla.where(
                        q_in_bounds & kv_in_bounds & keep, tla.Int32(1), tla.Int32(0)
                    )
                row_counts[tid] = visible_count
                tla.arch.sync_threads()

                for offset in (64, 32, 16, 8, 4, 2, 1):
                    if tid < offset:
                        row_counts[tid] = row_counts[tid] + row_counts[tid + offset]
                    tla.arch.sync_threads()

                if tid == 0:
                    visible_total = row_counts[0]
                    if visible_total > 0:
                        valid_q = seqlen_q - q_tile * 128
                        if valid_q > 128:
                            valid_q = tla.Int32(128)
                        valid_k = seqlen_k - kv_tile * 128
                        if valid_k > 128:
                            valid_k = tla.Int32(128)
                        if visible_total == valid_q * valid_k:
                            full_count = full_counts[work]
                            full_indices[work, full_count] = kv_tile
                            full_counts[work] = full_count + 1
                        else:
                            partial_count = mask_counts[work]
                            mask_indices[work, partial_count] = kv_tile
                            mask_counts[work] = partial_count + 1
                tla.arch.sync_threads()
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def flex_attention_classify_blocks_simd_kernel(
    mask_counts: tla.Tensor,
    mask_indices: tla.Tensor,
    full_counts: tla.Tensor,
    full_indices: tla.Tensor,
    mask_mod: tla.Constexpr[Callable[..., Any]],
    num_heads_q: tla.Constexpr[int],
    q_tiles: tla.Constexpr[int],
    kv_tiles: tla.Constexpr[int],
    seqlen_q: tla.Constexpr[int],
    seqlen_k: tla.Constexpr[int],
    aux_tensors: Any,
    aux_scalars: Any,
) -> None:
    """Explicit x64 mask evaluation, with one scalar metadata-recording VF.

    Scratch holds only one exact visible-element count per KV tile. Q loops
    include valid rows only; padded KV lanes are clamped for callback safety
    then excluded from counting. Unused public index slots are unspecified.
    """
    work = tla.arch.block_idx()
    q_tile = work % q_tiles
    head = (work // q_tiles) % num_heads_q
    batch = work // (q_tiles * num_heads_q)
    valid_q = min(128, seqlen_q - q_tile * 128)
    info = create_seqlen_info(tla.Int32(seqlen_q), tla.Int32(seqlen_k))
    totals_ptr = tla.allocate(kv_tiles, tla.Int32, tla.AddressSpace.ub, 256)
    totals = tla.make_tensor(
        totals_ptr, tla.make_layout(tla.make_shape(kv_tiles), tla.make_stride(1))
    )
    vector_done = tla.flag("classifier_vector_done", tla.arch.VECTOR, tla.arch.SCALAR)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            zero = tla.full(0, tla.Int32)
            one = tla.full(1, tla.Int32)
            last_k = tla.full(seqlen_k - 1, tla.Int32)
            full_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)
            b = zero + batch
            h = zero + head
            for kv_tile in tla.range(kv_tiles):
                visible = zero
                for row in tla.range(valid_q):
                    q = zero + (q_tile * 128 + row)
                    for chunk in tla.range_constexpr(2):
                        kv = tla.arange(kv_tile * 128 + chunk * 64, dtype=tla.Int32)
                        valid_k = tla.cmp(kv, seqlen_k, "lt")
                        safe_k = tla.where(valid_k, kv, last_k)
                        keep = call_mask_mod(
                            mask_mod, b, h, q, safe_k, info, aux_tensors, aux_scalars
                        )
                        _validate_mask_vector(keep)
                        keep = tla.bitwise_and(keep, valid_k)
                        visible = visible + tla.where(keep, one, zero)
                total = visible.reduce(tla.ReductionOp.ADD, mask=full_mask)
                slot = tla.tile_view(totals, tla.make_shape(1), tla.make_coord(kv_tile))
                slot.store(total, params=tla.params.UnalignStoreParams())
        # Complete SIMD UB stores before the scalar pipe launches the recording
        # VF. This is a cross-pipe dependency, not just an intra-SIMT barrier.
        tla.set_flag(vector_done)
        tla.wait_flag(vector_done)
        with tla.vec.func(mode="simt", thread_block_dim=1):
            partial_count = tla.Int32(0)
            full_count = tla.Int32(0)
            for kv_tile in tla.range(kv_tiles):
                visible_total = totals[kv_tile]
                if visible_total > 0:
                    valid_k = min(128, seqlen_k - kv_tile * 128)
                    if visible_total == valid_q * valid_k:
                        full_indices[work, full_count] = kv_tile
                        full_count = full_count + 1
                    else:
                        mask_indices[work, partial_count] = kv_tile
                        partial_count = partial_count + 1
            mask_counts[work] = partial_count
            full_counts[work] = full_count
        tla.pipe_barrier(tla.pipes.ALL)


def compute_block_sparsity(
    tile_m,
    tile_n,
    batch_size,
    num_heads,
    seqlen_q,
    seqlen_k,
    mask_mod: Callable,
    aux_tensors: list | None,
    device,
    aux_scalars: tuple | None = None,
    cu_seqlens_q: Any = None,
    cu_seqlens_k: Any = None,
    seqused_q: Any = None,
    seqused_k: Any = None,
    cu_total_m_blocks: Any = None,
    cu_block_idx_offsets: Any = None,
    compute_full_blocks: bool = True,
    use_fast_sampling: bool = False,
) -> BlockSparseTensorsTorch:
    """Build exact partial/full KV lists for each batch, Q head and Q block.

    Only fixed-length 128 x 128 classification with full-block detection is
    implemented. Unsupported options fail before any device guard or work.
    Pass the result as ``block_sparse_tensors`` alongside the same mask and
    auxiliaries. Rebuild metadata after changing inputs consumed by the mask.
    """
    if type(tile_m) is not int or type(tile_n) is not int or (tile_m, tile_n) != (128, 128):
        raise NotImplementedError("only tile_m=tile_n=128 is supported")
    for name, value in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
        ("seqused_q", seqused_q),
        ("seqused_k", seqused_k),
        ("cu_total_m_blocks", cu_total_m_blocks),
        ("cu_block_idx_offsets", cu_block_idx_offsets),
    ):
        if value is not None:
            raise NotImplementedError(f"{name} is not supported by fixed-length classification")
    if compute_full_blocks is not True:
        raise NotImplementedError("compute_full_blocks=False is not supported")
    if use_fast_sampling is not False or getattr(mask_mod, "use_fast_sampling", False):
        raise NotImplementedError("use_fast_sampling is not supported; classification is exact")

    import torch

    if mask_mod is None:
        raise TypeError("compute_block_sparsity requires a mask_mod callback")
    validate_callback(mask_mod, "mask")
    classifier = (
        flex_attention_classify_blocks_simd_kernel
        if uses_simd(mask_mod)
        else flex_attention_classify_blocks_kernel
    )
    batch_size, num_heads, q_tiles, kv_tiles = _dimensions(
        batch_size, num_heads, seqlen_q, seqlen_k
    )
    device = _require_npu(device)
    aux_tensors, aux_scalars = _validate_aux(aux_tensors, aux_scalars, device)
    with torch.npu.device(device):
        row_shape = (batch_size, num_heads, q_tiles)
        mask_counts = torch.empty(row_shape, dtype=torch.int32, device=device)
        mask_indices = torch.empty((*row_shape, kv_tiles), dtype=torch.int32, device=device)
        full_counts = torch.empty_like(mask_counts)
        full_indices = torch.empty_like(mask_indices)
        num_works = batch_size * num_heads * q_tiles
        tensors = (
            mask_counts.reshape(num_works),
            mask_indices.reshape(num_works, kv_tiles),
            full_counts.reshape(num_works),
            full_indices.reshape(num_works, kv_tiles),
        )
        outputs = tuple(from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in tensors)
        runtime_aux = (
            None
            if aux_tensors is None
            else [from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in aux_tensors]
        )
        constants = (mask_mod, num_heads, q_tiles, kv_tiles, seqlen_q, seqlen_k)
        key = _compilation_key(tensors, aux_tensors, aux_scalars, constants, device)
        compiled = _compile_kernel(
            classifier, key, lambda: (*outputs, *constants, runtime_aux, aux_scalars)
        )
        compiled(*outputs, runtime_aux, aux_scalars, block_num=num_works)
        return BlockSparseTensorsTorch(
            mask_counts, mask_indices, full_counts, full_indices, block_size=(128, 128)
        )
