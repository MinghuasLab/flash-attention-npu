"""Torch-only reference snapshot from the FlashAttention test suite.

BSD-3-Clause; copyright and authors are preserved in
flash_attn/FLASH_ATTENTION_LICENSE and FLASH_ATTENTION_AUTHORS.
Provenance and selected functions: upstream_reference_manifest.json.
Reference expressions are evaluated locally with Torch.
"""


def get_flex_causal_mask(offset: int):
    def _flex_causal_mask(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx + offset

    return _flex_causal_mask


def get_flex_block_causal_mask(offset: int):
    def _flex_block_causal_mask(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx + offset

    return _flex_block_causal_mask


def get_flex_sliding_window_mask(window_left: int, window_right: int, offset: int):
    def _flex_sliding_window_mask(b, h, q_idx, kv_idx):
        center = q_idx + offset
        lower = center - window_left
        upper = center + window_right
        return (kv_idx >= lower) & (kv_idx <= upper)

    return _flex_sliding_window_mask


def flex_block_diagonal_mask(b, h, q_idx, kv_idx):
    block_size = 128
    return (q_idx // block_size) == (kv_idx // block_size)


def flex_mini_causal_mask(b, h, q_idx, kv_idx):
    return (q_idx % 128) >= (kv_idx % 128)


def flex_prefix_lm_mask(b, h, q_idx, kv_idx):
    """Prefix LM mask: first 512 tokens attend bidirectionally, rest use causal masking."""
    prefix_size = 512
    both_in_prefix = (q_idx < prefix_size) & (kv_idx < prefix_size)
    causal_part = q_idx >= kv_idx
    return both_in_prefix | causal_part


def flex_dilated_sliding_window_mask(b, h, q_idx, kv_idx):
    """Dilated sliding window: every other position in a 256-position window."""
    window_size = 256
    dilation = 2
    in_window = (q_idx >= kv_idx) & (q_idx - kv_idx < window_size)
    dilated = ((q_idx - kv_idx) % dilation) == 0
    return in_window & dilated


def flex_document_mask(b, h, q_idx, kv_idx, doc_id):
    return doc_id[b, h, q_idx] == doc_id[b, h, kv_idx]


def flex_ima_mask(b, h, q_idx, kv_idx, bias):
    return kv_idx >= bias[kv_idx]


def global_packed_doc_flex_factory(doc_ids_q, doc_ids_k, cu_seqlens_q, cu_seqlens_k):
    """Build the per-sequence reference for a global packed document mask."""

    def factory(seq_idx, sq, sk):
        q_offset = cu_seqlens_q[seq_idx].item()
        k_offset = cu_seqlens_k[seq_idx].item()

        def mask(b, h, q_idx, kv_idx):
            q_global = q_offset + q_idx
            k_global = k_offset + kv_idx
            return doc_ids_q[q_global] == doc_ids_k[k_global]

        return mask

    return factory


def global_ima_flex_factory(thresholds, cu_seqlens_k):
    """Build the per-sequence reference for a global IMA mask."""

    def factory(seq_idx, sq, sk):
        k_offset = cu_seqlens_k[seq_idx].item()

        def mask(b, h, q_idx, kv_idx):
            k_global = k_offset + kv_idx
            return kv_idx >= thresholds[k_global]

        return mask

    return factory


def global_causal_window_flex_factory(windows, cu_seqlens_q):
    """Build the per-sequence reference for a global causal window mask."""

    def factory(seq_idx, sq, sk):
        q_offset = cu_seqlens_q[seq_idx].item()

        def mask(b, h, q_idx, kv_idx):
            q_global = q_offset + q_idx
            window = windows[q_global]
            return (kv_idx <= q_idx) & ((q_idx - kv_idx) <= window)

        return mask

    return factory
