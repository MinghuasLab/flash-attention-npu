# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Flex Attention block metadata and metadata-only validation.

The public Flex Attention representation describes partial and full 128 x 128 blocks.
Compact index capacities and independent batch/head broadcasting are accepted.
Only entries before each count are meaningful. The kernel consumes partial
then full blocks directly, preserving the order within each caller-owned list.
"""

from __future__ import annotations

from typing import Any, NamedTuple


BLOCK_SIZE = 128


class BlockSparseTensorsTorch(NamedTuple):
    """Per-query-block partial/full KV lists for fixed-length attention.

    Counts are NPU int32 ``[B,Hq,Tq]``; indices are ``[B,Hq,Tq,capacity]``
    with capacity at most the number of KV blocks, including zero. Each tensor
    may independently broadcast batch/head dimensions. The optional full pair
    bypasses ``mask_mod``. Unused index entries are ignored. Varlen and backward
    metadata fields are present for interface parity but must remain ``None``.
    """

    mask_block_cnt: Any
    mask_block_idx: Any
    full_block_cnt: Any = None
    full_block_idx: Any = None
    cu_total_m_blocks: Any = None
    cu_block_idx_offsets: Any = None
    block_size: tuple[int, int] | None = None
    dq_write_order: Any = None
    dq_write_order_full: Any = None
    spt: bool | None = None


class _SparseMetadata(NamedTuple):
    """Original tensors and independent B/H/Q strides for kernel binding.

    Empty indices bind as None, so DLPack never sees a zero-capacity tensor.
    Their strides still describe capacity and batch/head broadcasting.
    """

    tensors: tuple
    strides: tuple


def _dimensions(batch_size, num_heads_q, seqlen_q, seqlen_k):
    for name, value in (
        ("batch_size", batch_size),
        ("num_heads_q", num_heads_q),
        ("seqlen_q", seqlen_q),
        ("seqlen_k", seqlen_k),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return (
        batch_size,
        num_heads_q,
        (seqlen_q + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (seqlen_k + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )


def _require_npu(device):
    import torch

    device = torch.device(device)
    if device.type != "npu":
        raise ValueError("Flex Attention block sparsity requires an NPU device")
    if device.index is None:
        device = torch.device("npu", torch.npu.current_device())
    return device


def _same_device(actual, requested):
    return actual.type == requested.type and (
        requested.index is None or actual.index == requested.index
    )


def _check_pair(counts, indices, name, dimensions, device):
    import torch

    batch_size, num_heads_q, q_tiles, kv_tiles = dimensions
    for suffix, tensor in (("cnt", counts), ("idx", indices)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}_block_{suffix} must be a torch.Tensor")
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name}_block_{suffix} must have dtype torch.int32")
        if not _same_device(tensor.device, device):
            raise ValueError(f"{name}_block_{suffix} must be on {device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name}_block_{suffix} must be contiguous")
    if counts.ndim != 3:
        raise ValueError(f"{name}_block_cnt must have shape [B,Hq,Tq]")
    if (
        counts.shape[0] not in (1, batch_size)
        or counts.shape[1] not in (1, num_heads_q)
        or counts.shape[2] != q_tiles
    ):
        raise ValueError(f"{name}_block_cnt has incompatible batch/head/Q extents")
    if indices.ndim != 4:
        raise ValueError(f"{name}_block_idx must have shape [B,Hq,Tq,capacity]")
    if (
        indices.shape[0] not in (1, batch_size)
        or indices.shape[1] not in (1, num_heads_q)
        or indices.shape[2] != q_tiles
    ):
        raise ValueError(f"{name}_block_idx has incompatible batch/head/Q extents")
    capacity = indices.shape[-1]
    if capacity > kv_tiles:
        raise ValueError(f"{name}_block_idx capacity must be <= {kv_tiles}")
    if counts.device != indices.device:
        raise ValueError(f"{name} counts and indices must be on the same NPU device")


def _metadata_strides(tensor):
    return (
        tensor.stride(0) if tensor.shape[0] != 1 else 0,
        tensor.stride(1) if tensor.shape[1] != 1 else 0,
        1 if tensor.ndim == 3 else tensor.shape[-1],
    )


def _validate_sparse_options(block_sparse, seqlen_q):
    for name in (
        "cu_total_m_blocks",
        "cu_block_idx_offsets",
        "dq_write_order",
        "dq_write_order_full",
        "spt",
    ):
        if getattr(block_sparse, name) is not None:
            raise NotImplementedError(f"{name} is not supported by fixed-length forward")
    block_size = block_sparse.block_size
    if block_size is None:
        indices = block_sparse.mask_block_idx
        if indices is None or getattr(indices, "ndim", None) != 4:
            raise ValueError("mask_block_idx must have shape [B,Hq,Tq,capacity]")
        num_m_blocks = indices.shape[2]
        if num_m_blocks <= 0:
            raise ValueError("mask_block_idx must have a nonempty Q-block dimension")
        minimum = (seqlen_q + num_m_blocks - 1) // num_m_blocks
        maximum = seqlen_q if num_m_blocks == 1 else (seqlen_q - 1) // (num_m_blocks - 1)
        if minimum != maximum:
            raise ValueError("ambiguous Q block size; provide block_size=(128, 128)")
        block_size = (minimum, BLOCK_SIZE)
    if (
        not isinstance(block_size, (tuple, list))
        or len(block_size) != 2
        or any(type(size) is not int or size <= 0 for size in block_size)
    ):
        raise ValueError("block_size must contain two positive integers")
    if tuple(block_size) != (BLOCK_SIZE, BLOCK_SIZE):
        raise NotImplementedError("only block_size=(128, 128) is supported")


def validate_block_sparse(
    block_sparse: BlockSparseTensorsTorch,
    *,
    batch_size: int,
    num_heads_q: int,
    seqlen_q: int,
    seqlen_k: int,
    device: Any,
) -> _SparseMetadata:
    """Check metadata specifications without reading or changing tensor contents.

    Count bounds, active index bounds, uniqueness within/across lists and mask
    consistency are caller guarantees. No sorting or schedule is materialized.
    """
    if not isinstance(block_sparse, BlockSparseTensorsTorch):
        raise TypeError("block_sparse must be a BlockSparseTensorsTorch, not an opaque adapter")
    dimensions = _dimensions(batch_size, num_heads_q, seqlen_q, seqlen_k)
    _validate_sparse_options(block_sparse, seqlen_q)
    device = _require_npu(device)
    if (block_sparse.full_block_cnt is None) != (block_sparse.full_block_idx is None):
        raise ValueError("full_block_cnt and full_block_idx must both be present or None")
    _check_pair(
        block_sparse.mask_block_cnt,
        block_sparse.mask_block_idx,
        "mask",
        dimensions,
        device,
    )
    if block_sparse.full_block_cnt is not None:
        _check_pair(
            block_sparse.full_block_cnt,
            block_sparse.full_block_idx,
            "full",
            dimensions,
            device,
        )
    tensors = block_sparse[:4]
    strides = tuple(_metadata_strides(tensor) if tensor is not None else None for tensor in tensors)
    bindings = tuple(
        None if tensor is not None and tensor.ndim == 4 and tensor.shape[-1] == 0 else tensor
        for tensor in tensors
    )
    return _SparseMetadata(bindings, strides)
