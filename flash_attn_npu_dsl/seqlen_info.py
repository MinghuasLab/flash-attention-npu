"""Sequence-local coordinates and packed-storage offsets shared by both kernels."""

from typing import NamedTuple

import catlass.tla as tla


def create_seqlen_info(seqlen_q, seqlen_k, tile_n=128):
    return SeqlenInfoQK(
        0,
        0,
        0,
        0,
        seqlen_q,
        seqlen_k,
        0,
        0,
        (seqlen_k + tile_n - 1) // tile_n,
    )


class SeqlenInfoQK(NamedTuple):
    offset_q: int
    offset_k: int
    padded_offset_q: int
    padded_offset_k: int
    seqlen_q: int
    seqlen_k: int
    m_block_offset: int
    block_idx_offset: int
    num_n_blocks: int
    has_cu_seqlens_q: bool = False
    has_cu_seqlens_k: bool = False
    has_seqused_q: bool = False
    has_seqused_k: bool = False
    has_cu_block_idx_offsets: bool = False

    create = staticmethod(create_seqlen_info)


@tla.jit
def create_packed_seqlen_info(batch, metadata, fixed_q=0, fixed_k=0):
    """Read storage offsets separately from the optional effective lengths.

    The final two entries carry packed block offsets when sparse metadata is
    present. All values remain device-side, including when inputs are rebound.
    """
    cu_q, cu_k, used_q, used_k, cu_m, cu_idx = metadata
    offset_q = tla.Int32(0)
    offset_k = tla.Int32(0)
    length_q = tla.Int32(fixed_q)
    length_k = tla.Int32(fixed_k)
    if tla.const_expr(cu_q is not None):
        offset_q = cu_q[batch]
        length_q = cu_q[batch + 1] - offset_q
    if tla.const_expr(cu_k is not None):
        offset_k = cu_k[batch]
        length_k = cu_k[batch + 1] - offset_k
    if tla.const_expr(used_q is not None):
        length_q = used_q[batch]
    if tla.const_expr(used_k is not None):
        length_k = used_k[batch]
    m_offset = tla.Int32(0)
    idx_offset = tla.Int32(0)
    if tla.const_expr(cu_m is not None):
        m_offset = cu_m[batch]
    num_n = (length_k + 127) // 128
    if tla.const_expr(cu_idx is not None):
        idx_offset = cu_idx[batch]
    else:
        idx_offset = m_offset * num_n
    return SeqlenInfoQK(
        offset_q,
        offset_k,
        (offset_q + batch * 128) // 128 * 128 if cu_q is not None else 0,
        (offset_k + batch * 128) // 128 * 128 if cu_k is not None else 0,
        length_q,
        length_k,
        m_offset,
        idx_offset,
        num_n,
        cu_q is not None,
        cu_k is not None,
        used_q is not None,
        used_k is not None,
        cu_idx is not None,
    )


def validate_sequence_tensor(tensor, name, size, device):
    """Validate storage only; prefix monotonicity and bounds belong to the caller."""
    import torch

    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (size,):
        raise ValueError(f"{name} must have shape [{size}]")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on the same device as Q")
