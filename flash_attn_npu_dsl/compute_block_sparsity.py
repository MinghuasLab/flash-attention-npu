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
from .interface import _check_launch_tensors, _compilation_key, _compile_kernel, _validate_aux
from .seqlen_info import create_packed_seqlen_info, create_seqlen_info, validate_sequence_tensor


@tla.jit
def _classifier_context(
    work,
    num_heads: tla.Constexpr,
    q_tiles: tla.Constexpr,
    q_length: tla.Constexpr,
    k_length: tla.Constexpr,
    counts_per_head: tla.Constexpr,
    indices_per_head: tla.Constexpr,
    metadata,
):
    """Locate one public count row and its matching packed index region."""
    if tla.const_expr(metadata is None):
        q_tile = work % q_tiles
        head = (work // q_tiles) % num_heads
        batch = work // (q_tiles * num_heads)
        info = create_seqlen_info(tla.Int32(q_length), tla.Int32(k_length))
        index_row, index_base = work, 0
    else:
        head = work // counts_per_head
        global_m = work % counts_per_head
        batch = tla.Int32(0)
        while metadata[4][batch + 1] <= global_m:
            batch += 1
        info = create_packed_seqlen_info(batch, metadata)
        q_tile = global_m - info.m_block_offset
        index_row = 0
        index_base = head * indices_per_head + info.block_idx_offset + q_tile * info.num_n_blocks
    return batch, head, q_tile, info, index_row, index_base


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
    sequence_metadata=None,
) -> None:
    """Exact any/all classification with O(128) scratch, not a dense mask.

    Each SIMT block visits [batch, query-head, query-tile] rows by grid stride. All threads
    join all barriers, including physical tails. Invalid coordinates are
    clamped before callback invocation and excluded from the visible count.
    """
    row_count_ptr = tla.allocate(128, tla.Int32, tla.AddressSpace.ub, 256)
    row_counts = tla.make_tensor(
        row_count_ptr, tla.make_layout(tla.make_shape(128), tla.make_stride(1))
    )
    for work in tla.range(tla.arch.block_idx(), mask_counts.shape[0], tla.arch.block_num()):
        batch, head, q_tile, info, index_row, index_base = _classifier_context(
            work,
            num_heads_q,
            q_tiles,
            seqlen_q,
            seqlen_k,
            mask_counts.shape[0] // num_heads_q,
            mask_indices.shape[-1] // num_heads_q,
            sequence_metadata,
        )
        valid_rows = max(0, min(128, info.seqlen_q - q_tile * 128))
        active_kv_tiles = info.num_n_blocks
        if tla.const_expr(sequence_metadata is not None):
            if valid_rows == 0:
                active_kv_tiles = 0
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=128):
                tid, _, _ = tla.arch.thread_idx()
                q_idx = q_tile * 128 + tid
                q_in_bounds = q_idx < info.seqlen_q
                safe_q = tla.where(q_in_bounds, q_idx, info.seqlen_q - 1)
                if tid == 0:
                    mask_counts[work] = 0
                    full_counts[work] = 0
                    for slot in tla.range(0, active_kv_tiles, 1):
                        mask_indices[index_row, index_base + slot] = 0
                        full_indices[index_row, index_base + slot] = 0
                tla.arch.sync_threads()

                for kv_tile in tla.range(0, active_kv_tiles, 1):
                    visible_count = tla.Int32(0)
                    for local_k in tla.range(0, 128, 1):
                        kv_idx = kv_tile * 128 + local_k
                        kv_in_bounds = kv_idx < info.seqlen_k
                        safe_k = tla.where(kv_in_bounds, kv_idx, info.seqlen_k - 1)
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
                            valid_q = info.seqlen_q - q_tile * 128
                            if valid_q > 128:
                                valid_q = tla.Int32(128)
                            valid_k = info.seqlen_k - kv_tile * 128
                            if valid_k > 128:
                                valid_k = tla.Int32(128)
                            if visible_total == valid_q * valid_k:
                                full_count = full_counts[work]
                                full_indices[index_row, index_base + full_count] = kv_tile
                                full_counts[work] = full_count + 1
                            else:
                                partial_count = mask_counts[work]
                                mask_indices[index_row, index_base + partial_count] = kv_tile
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
    sequence_metadata=None,
) -> None:
    """Explicit x64 mask evaluation, with one scalar metadata-recording VF.

    Scratch holds only one exact visible-element count per KV tile. Q loops
    include valid rows only; padded KV lanes are clamped for callback safety
    then excluded from counting. Unused public index slots are unspecified.
    """
    totals_ptr = tla.allocate(kv_tiles, tla.Int32, tla.AddressSpace.ub, 256)
    totals = tla.make_tensor(
        totals_ptr, tla.make_layout(tla.make_shape(kv_tiles), tla.make_stride(1))
    )
    vector_done = tla.flag("classifier_vector_done", tla.arch.VECTOR, tla.arch.SCALAR)
    for work in tla.range(tla.arch.block_idx(), mask_counts.shape[0], tla.arch.block_num()):
        batch, head, q_tile, info, index_row, index_base = _classifier_context(
            work,
            num_heads_q,
            q_tiles,
            seqlen_q,
            seqlen_k,
            mask_counts.shape[0] // num_heads_q,
            mask_indices.shape[-1] // num_heads_q,
            sequence_metadata,
        )
        if tla.const_expr(sequence_metadata is None and q_tiles == 1):
            # Keep the known row count constant across the SIMD VF boundary.
            valid_q = seqlen_q
        else:
            valid_q = max(0, min(128, info.seqlen_q - q_tile * 128))
        active_kv_tiles = info.num_n_blocks
        if tla.const_expr(sequence_metadata is not None):
            if valid_q == 0:
                active_kv_tiles = 0
        with tla.vector():
            with tla.vec.func(mode="simd"):
                zero = tla.full(0, tla.Int32)
                one = tla.full(1, tla.Int32)
                last_k = zero + (info.seqlen_k - 1)
                full_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)
                b = zero + batch
                h = zero + head
                for kv_tile in tla.range(active_kv_tiles):
                    visible = zero
                    for row in tla.range(valid_q):
                        q = zero + (q_tile * 128 + row)
                        for chunk in tla.range_constexpr(2):
                            kv = tla.arange(kv_tile * 128 + chunk * 64, dtype=tla.Int32)
                            valid_k = tla.cmp(kv, info.seqlen_k, "lt")
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
                for kv_tile in tla.range(active_kv_tiles):
                    visible_total = totals[kv_tile]
                    if visible_total > 0:
                        valid_k = min(128, info.seqlen_k - kv_tile * 128)
                        if visible_total == valid_q * valid_k:
                            full_indices[index_row, index_base + full_count] = kv_tile
                            full_count = full_count + 1
                        else:
                            mask_indices[index_row, index_base + partial_count] = kv_tile
                            partial_count = partial_count + 1
                mask_counts[work] = partial_count
                full_counts[work] = full_count
            tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def flex_attention_block_offsets_kernel(
    cu_q,
    cu_k,
    cu_m,
    cu_idx,
    totals,
    batch: tla.Constexpr,
    write_m: tla.Constexpr,
    write_idx: tla.Constexpr,
):
    """Scan per-thread contiguous segments, then add their shared prefix sums."""
    m_sums = [
        tla.make_tensor(
            tla.allocate(128, tla.Int32, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(128), tla.make_stride(1)),
        )
        for _ in tla.range_constexpr(2)
    ]
    idx_sums = [
        tla.make_tensor(
            tla.allocate(128, tla.Int32, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(128), tla.make_stride(1)),
        )
        for _ in tla.range_constexpr(2)
    ]
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=128):
            tid, _, _ = tla.arch.thread_idx()
            chunk = (batch + 127) // 128
            start = tid * chunk
            end = min(batch, start + chunk)
            m_sum = tla.Int32(0)
            idx_sum = tla.Int32(0)
            for b in tla.range(start, end):
                if tla.const_expr(write_m):
                    m_rows = (cu_q[b + 1] - cu_q[b] + 127) // 128
                else:
                    m_rows = cu_m[b + 1] - cu_m[b]
                m_sum += m_rows
                if tla.const_expr(write_idx):
                    n_blocks = (cu_k[b + 1] - cu_k[b] + 127) // 128
                    idx_sum += m_rows * n_blocks
            m_sums[0][tid] = m_sum
            idx_sums[0][tid] = idx_sum
            tla.arch.sync_threads()
            # Ping-pong scratch prevents any thread from overwriting a
            # value still read by another thread in the same scan level.
            for stage in tla.range_constexpr(7):
                if tid >= (1 << stage):
                    m_sum += m_sums[stage % 2][tid - (1 << stage)]
                    idx_sum += idx_sums[stage % 2][tid - (1 << stage)]
                m_sums[(stage + 1) % 2][tid] = m_sum
                idx_sums[(stage + 1) % 2][tid] = idx_sum
                tla.arch.sync_threads()
            m_offset = tla.Int32(0)
            idx_offset = tla.Int32(0)
            if tid > 0:
                m_offset = m_sums[1][tid - 1]
                idx_offset = idx_sums[1][tid - 1]
            for write_b in tla.range(start, end):
                if tla.const_expr(write_m):
                    write_rows = (cu_q[write_b + 1] - cu_q[write_b] + 127) // 128
                    m_offset += write_rows
                    cu_m[write_b + 1] = m_offset
                else:
                    write_rows = cu_m[write_b + 1] - cu_m[write_b]
                if tla.const_expr(write_idx):
                    idx_offset += write_rows * ((cu_k[write_b + 1] - cu_k[write_b] + 127) // 128)
                    cu_idx[write_b + 1] = idx_offset
            if tid == 0:
                if tla.const_expr(write_m):
                    cu_m[0] = 0
                if tla.const_expr(write_idx):
                    cu_idx[0] = 0
                totals[0] = m_sums[1][127] if write_m else cu_m[batch]
                totals[1] = idx_sums[1][127] if write_idx else cu_idx[batch]


def _packed_block_offsets(cu_q, cu_k, cu_m, cu_idx):
    import torch

    if cu_m is not None and cu_idx is not None:
        # No prefix computation is needed; only allocation capacities are read.
        return cu_m, cu_idx, torch.stack((cu_m[-1], cu_idx[-1])).cpu().tolist()
    batch = cu_q.numel() - 1
    write_m, write_idx = cu_m is None, cu_idx is None
    if write_m:
        cu_m = torch.empty_like(cu_q)
    if write_idx:
        cu_idx = torch.empty_like(cu_q)
    totals = torch.empty(2, dtype=torch.int32, device=cu_q.device)
    tensors = (cu_q, cu_k, cu_m, cu_idx, totals)
    _check_launch_tensors(tensors)
    constants = (batch, write_m, write_idx)
    key = _compilation_key(tensors, None, None, constants, cu_q.device)

    def compile_args():
        return (*tuple(from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in tensors), *constants)

    compiled = _compile_kernel(flex_attention_block_offsets_kernel, key, compile_args)
    compiled(*tensors, block_num=1)
    return cu_m, cu_idx, totals.cpu().tolist()


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
    """Build exact partial/full 128 x 128 KV lists for BSND or packed TND.

    Packed inputs reserve rows from storage segments, while classification uses
    effective lengths. Allocation reads the two terminal capacities together;
    attention reuse never needs this readback.
    Pass the result as ``block_sparse_tensors`` alongside the same mask and
    auxiliaries. Rebuild metadata after changing inputs consumed by the mask.
    """
    if type(tile_m) is not int or type(tile_n) is not int or (tile_m, tile_n) != (128, 128):
        raise NotImplementedError("only tile_m=tile_n=128 is supported")
    packed = cu_seqlens_q is not None or cu_seqlens_k is not None
    if not packed and any(
        x is not None for x in (seqused_q, seqused_k, cu_total_m_blocks, cu_block_idx_offsets)
    ):
        raise NotImplementedError("sequence metadata requires both cu_seqlens_q and cu_seqlens_k")
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
    for name, value in (("seqlen_q", seqlen_q), ("seqlen_k", seqlen_k)):
        if type(value) is not int or value < (0 if packed else 1):
            raise ValueError(f"{name} must be {'nonnegative' if packed else 'positive'}")
    batch_size, num_heads, q_tiles, kv_tiles = _dimensions(
        batch_size, num_heads, max(1, seqlen_q), max(1, seqlen_k)
    )
    device = _require_npu(device)
    aux_tensors, aux_scalars = _validate_aux(aux_tensors, aux_scalars, device)
    with torch.npu.device(device):
        sequence_metadata = None
        if packed:
            for name, tensor, size in (
                ("cu_seqlens_q", cu_seqlens_q, batch_size + 1),
                ("cu_seqlens_k", cu_seqlens_k, batch_size + 1),
                ("seqused_q", seqused_q, batch_size),
                ("seqused_k", seqused_k, batch_size),
                ("cu_total_m_blocks", cu_total_m_blocks, batch_size + 1),
                ("cu_block_idx_offsets", cu_block_idx_offsets, batch_size + 1),
            ):
                if tensor is not None or name in ("cu_seqlens_q", "cu_seqlens_k"):
                    validate_sequence_tensor(tensor, name, size, device)
            cu_total_m_blocks, cu_block_idx_offsets, (total_m, total_indices) = (
                _packed_block_offsets(
                    cu_seqlens_q, cu_seqlens_k, cu_total_m_blocks, cu_block_idx_offsets
                )
            )
            sequence_metadata = (
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                cu_total_m_blocks,
                cu_block_idx_offsets,
            )
            row_shape, index_shape = (num_heads, total_m), (num_heads, total_indices)
            num_works = num_heads * total_m
            mask_counts = torch.zeros(row_shape, dtype=torch.int32, device=device)
        else:
            row_shape = (batch_size, num_heads, q_tiles)
            index_shape = (*row_shape, kv_tiles)
            num_works = batch_size * num_heads * q_tiles
            mask_counts = torch.empty(row_shape, dtype=torch.int32, device=device)
        mask_indices = torch.empty(index_shape, dtype=torch.int32, device=device)
        full_counts = torch.empty_like(mask_counts)
        full_indices = torch.empty_like(mask_indices)
        if packed and (num_works == 0 or total_indices == 0):
            full_counts.zero_()
            return BlockSparseTensorsTorch(
                mask_counts,
                mask_indices,
                full_counts,
                full_indices,
                cu_total_m_blocks,
                cu_block_idx_offsets,
                (128, 128),
            )
        tensors = (
            mask_counts.reshape(num_works),
            mask_indices.reshape(1, -1) if packed else mask_indices.reshape(num_works, kv_tiles),
            full_counts.reshape(num_works),
            full_indices.reshape(1, -1) if packed else full_indices.reshape(num_works, kv_tiles),
        )
        _check_launch_tensors((*tensors, *(aux_tensors or ()), *(sequence_metadata or ())))
        outputs = tensors
        runtime_aux = aux_tensors
        constants = (mask_mod, num_heads, q_tiles, kv_tiles, seqlen_q, seqlen_k)
        key = _compilation_key(
            (*tensors, *(sequence_metadata or ())),
            aux_tensors,
            aux_scalars,
            (*constants, packed),
            device,
        )

        def compile_args():
            samples = tuple(from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in tensors)
            sample_aux = (
                None
                if aux_tensors is None
                else [from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in aux_tensors]
            )
            sample_sequences = (
                tuple(
                    from_dlpack(t, layout_tag=tla.arch.RowMajor) if t is not None else None
                    for t in sequence_metadata
                )
                if packed
                else None
            )
            return (*samples, *constants, sample_aux, aux_scalars, sample_sequences)

        compiled = _compile_kernel(classifier, key, compile_args)
        # Logical rows beyond the launch grid are covered by the kernel's grid stride.
        compiled(
            *outputs, runtime_aux, aux_scalars, sequence_metadata, block_num=min(num_works, 65535)
        )
        return BlockSparseTensorsTorch(
            mask_counts,
            mask_indices,
            full_counts,
            full_indices,
            cu_total_m_blocks,
            cu_block_idx_offsets,
            block_size=(128, 128),
        )
