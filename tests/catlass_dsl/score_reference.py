"""Torch-only reference snapshot from the FlashAttention test suite.

BSD-3-Clause; copyright and authors are preserved in
flash_attn/FLASH_ATTENTION_LICENSE and FLASH_ATTENTION_AUTHORS.
Provenance and selected functions: upstream_reference_manifest.json.
Reference expressions are evaluated locally with Torch.
"""

import torch

REL_BIAS_CLAMP = 256


def identity_eager(score, b, h, q_idx, kv_idx):
    return score


def causal_eager(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, float("-inf"))


def rel_bias_eager(score, b, h, q_idx, kv_idx):
    return score + torch.abs(q_idx - kv_idx)


def rel_bias_clamped_eager(score, b, h, q_idx, kv_idx):
    return score + torch.clamp(torch.abs(q_idx - kv_idx), max=REL_BIAS_CLAMP)


def rel_bias_x2_eager(score, b, h, q_idx, kv_idx):
    return score + 2 * torch.abs(q_idx - kv_idx)


def times_two_eager(score, b, h, q_idx, kv_idx):
    return score * 2


def alibi_eager(score, b, h, q_idx, kv_idx):
    slope = 2 ** (-8 * (h + 1) / 8)
    return score - slope * torch.abs(q_idx - kv_idx)


def sliding_window_eager(score, b, h, q_idx, kv_idx):
    return torch.where(torch.abs(q_idx - kv_idx) <= 256, score, float("-inf"))


def block_diagonal_eager(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx // 64 == kv_idx // 64, score, float("-inf"))


def causal_v2_eager(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx - kv_idx >= 0, score, float("-inf"))


def squared_eager(score, b, h, q_idx, kv_idx):
    return score * score


def batch_bias_factory(bias_tensor):
    def mod(score, b, h, q_idx, kv_idx):
        return score + bias_tensor[b]

    return mod


def dual_buffer_factory(head_bias, pos_bias):
    def mod(score, b, h, q_idx, kv_idx):
        return score + head_bias[h] + pos_bias[q_idx]

    return mod


def packed_kv_bias_factory(bias_tensor, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        # Calculate valid length for this sequence
        start = cu_seqlens_k[b]
        seq_len = cu_seqlens_k[b + 1] - start

        # Clamp kv_idx.
        safe_kv_idx = torch.clamp(kv_idx, max=seq_len - 1)

        return score + bias_tensor[start + safe_kv_idx]

    return mod


def packed_q_bias_factory(bias_tensor, cu_seqlens_q):
    def mod(score, b, h, q_idx, kv_idx):
        start = cu_seqlens_q[b]
        seq_len = cu_seqlens_q[b + 1] - start

        # Clamp q_idx
        safe_q_idx = torch.clamp(q_idx, max=seq_len - 1)

        return score + bias_tensor[start + safe_q_idx]

    return mod


def packed_rel_plus_kv_bias_factory(bias_tensor, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        start = cu_seqlens_k[b]
        seq_len = cu_seqlens_k[b + 1] - start

        # Clamp kv_idx
        safe_kv_idx = torch.clamp(kv_idx, max=seq_len - 1)

        rel_bias = torch.abs(q_idx - kv_idx).float() * 0.1
        return score + rel_bias + bias_tensor[start + safe_kv_idx]

    return mod


def packed_q_and_kv_bias_factory(q_bias, kv_bias, cu_seqlens_q, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        # Handle Q bounds
        q_start = cu_seqlens_q[b]
        q_len = cu_seqlens_q[b + 1] - q_start
        safe_q_idx = torch.clamp(q_idx, max=q_len - 1)

        # Handle KV bounds
        kv_start = cu_seqlens_k[b]
        kv_len = cu_seqlens_k[b + 1] - kv_start
        safe_kv_idx = torch.clamp(kv_idx, max=kv_len - 1)

        return score + q_bias[q_start + safe_q_idx] + kv_bias[kv_start + safe_kv_idx]

    return mod


def packed_logical_rel_plus_kv_bias_factory(bias_tensor, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        rel_bias = torch.abs(q_idx - kv_idx).float() * 0.01
        return score + rel_bias + bias_tensor[cu_seqlens_k[b] + kv_idx]

    return mod


def stress_complex_arithmetic_factory(bias, cu_seqlens_q):
    def mod(score, b, h, q_idx, kv_idx):
        # Use absolute value instead of squaring to avoid overflow with large sequences
        rel_pos_abs = torch.abs(q_idx - kv_idx)
        q_global = cu_seqlens_q[b] + q_idx
        bias_q = bias[q_global]
        scale = (b + 1) * (h + 1) * 0.001
        rel_bias = rel_pos_abs * 0.001
        return score + rel_bias + bias_q * scale

    return mod


def stress_conditional_mask_factory(token_bias, cu_seqlens_q, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        kv_global = cu_seqlens_k[b] + kv_idx
        bias_val = token_bias[kv_global]
        is_causal = q_idx >= kv_idx
        q_global = cu_seqlens_q[b] + q_idx
        global_diff = q_global - kv_global
        is_nearby = torch.abs(global_diff) <= 512
        both_conditions = is_causal & is_nearby
        return torch.where(both_conditions, score + bias_val, float("-inf"))

    return mod


def stress_multi_buffer_factory(
    batch_bias,
    head_scale,
    q_pos_bias,
    kv_pos_bias,
    rel_pos_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    max_rel_pos=512,
):
    def mod(score, b, h, q_idx, kv_idx):
        bb_val = batch_bias[b]
        hs_val = head_scale[h]
        qpb_val = q_pos_bias[cu_seqlens_q[b] + q_idx]
        kvpb_val = kv_pos_bias[cu_seqlens_k[b] + kv_idx]
        rel_idx = (q_idx - kv_idx + max_rel_pos).clamp(0, max_rel_pos * 2)
        rps_val = rel_pos_scale[rel_idx]
        return score * hs_val + bb_val + qpb_val + kvpb_val + rps_val * 0.1

    return mod


def stress_global_offset_factory(token_bias, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        return score + token_bias[cu_seqlens_k[b] + kv_idx]

    return mod


def stress_xor_pattern_factory(token_bias, cu_seqlens_q, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        xor_logical = q_idx ^ kv_idx
        pattern_bias = (xor_logical & 0xFF).float() * 0.001
        kv_global = cu_seqlens_k[b] + kv_idx
        return score + pattern_bias + token_bias[kv_global] * 0.1

    return mod


def debug_global_idx_factory(bias, cu_seqlens_k):
    offsets = cu_seqlens_k.tolist()

    def mod(score, b, h, q_idx, kv_idx):
        global_kv = offsets[b] + kv_idx
        return score + global_kv.float() * 0.001

    return mod


def scaled_squared_factory(scale_tensor, cu_seqlens_k):
    """Eager reference for score_mod_scaled_squared (varlen: cu_seqlens_k provides offsets)."""

    def mod(score, b, h, q_idx, kv_idx):
        kv_global = cu_seqlens_k[b] + kv_idx
        return scale_tensor[kv_global] * score * score

    return mod
