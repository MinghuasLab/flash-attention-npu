"""Fixed-length sequence information exposed to Flex Attention callbacks."""

from typing import NamedTuple


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
