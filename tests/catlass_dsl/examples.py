"""Test-only native mask and score callback examples.

These are illustrative SIMT callbacks, not dispatch hints. Torch references
are deliberately separate expressions. Aux slots are documented per factory.
"""

from functools import lru_cache
import torch
import catlass.tla as tla
from flash_attn_npu_4.catlass_dsl.mods import Mod, MaskMod
from flash_attn_npu_4.catlass_dsl.native import exp2, load

MASK_NAMES = (
    "block_diagonal",
    "mini_causal",
    "prefix_lm",
    "dilated_sliding_window",
    "document",
    "ima",
    "global_packed_doc",
    "global_ima",
    "global_causal_window",
    "packed_aux",
    "causal",
    "block_causal",
    "sliding_window",
)


@lru_cache(maxsize=128)
def mask(name, *, left=128, right=0, offset=0):
    """document: aux[0][B,H,S]; global_doc: aux[0/1][totalQ/K].

    ima/window: aux[0] thresholds/windows; packed_aux: aux[0][B,Q,ceil(K/32)]
    UInt32 words. Global variants add the respective sequence offset.
    """
    if name not in MASK_NAMES:
        raise ValueError(f"unknown mask example: {name}")

    @tla.jit
    def native(b, h, q, k, info, aux, scalars):
        result = tla.Bool(False)
        if tla.const_expr(name == "block_diagonal"):
            result = q // 128 == k // 128
        elif tla.const_expr(name == "mini_causal"):
            result = q % 128 >= k % 128
        elif tla.const_expr(name == "prefix_lm"):
            result = ((q < 512) & (k < 512)) | (q >= k)
        elif tla.const_expr(name == "dilated_sliding_window"):
            result = (q >= k) & (q - k < 256) & ((q - k) % 2 == 0)
        elif tla.const_expr(name == "document"):
            result = load(aux[0], (b, h, q)) == load(aux[0], (b, h, k))
        elif tla.const_expr(name == "ima"):
            result = k >= aux[0][k]
        elif tla.const_expr(name == "global_packed_doc"):
            result = aux[0][q + info[2]] == aux[1][k + info[3]]
        elif tla.const_expr(name == "global_ima"):
            result = k >= aux[0][k + info[3]]
        elif tla.const_expr(name == "global_causal_window"):
            result = (k <= q) & (q - k <= aux[0][q + info[2]])
        elif tla.const_expr(name == "packed_aux"):
            word = load(aux[0], (b, q, k // 32))
            result = ((word >> (k % 32).to(tla.Int64)) & tla.Int64(1)) != tla.Int64(0)
        elif tla.const_expr(name == "causal"):
            result = k <= q + info[1] - info[0]
        elif tla.const_expr(name == "block_causal"):
            result = k <= q + offset
        elif tla.const_expr(name == "sliding_window"):
            center = q + info[1] - info[0]
            result = (k >= center - left) & (k <= center + right)
        return result

    def reference(b, h, q, k, info, aux, scalars):
        if name == "block_diagonal":
            return q // 128 == k // 128
        if name == "mini_causal":
            return q % 128 >= k % 128
        if name == "prefix_lm":
            return ((q < 512) & (k < 512)) | (q >= k)
        if name == "dilated_sliding_window":
            return (q >= k) & (q - k < 256) & ((q - k) % 2 == 0)
        if name == "document":
            return aux[0][b, h, q] == aux[0][b, h, k]
        if name == "ima":
            return k >= aux[0][k]
        if name == "global_packed_doc":
            return aux[0][q + info[2]] == aux[1][k + info[3]]
        if name == "global_ima":
            return k >= aux[0][k + info[3]]
        if name == "global_causal_window":
            return (k <= q) & (q - k <= aux[0][q + info[2]])
        if name == "packed_aux":
            return ((aux[0].to(torch.int64)[b, q, k // 32] >> (k % 32)) & 1) != 0
        if name == "causal":
            return k <= q + info[1] - info[0]
        if name == "block_causal":
            return k <= q + offset
        center = q + info[1] - info[0]
        return (k >= center - left) & (k <= center + right)

    return MaskMod(
        native,
        torch_ref=reference,
        broadcast_batch=name not in ("document", "packed_aux"),
        broadcast_head=name != "document",
    )


SCORE_NAMES = (
    "identity",
    "causal",
    "causal_v2",
    "rel_bias",
    "rel_bias_clamped",
    "rel_bias_x2",
    "times_two",
    "alibi",
    "sliding_window",
    "block_diagonal",
    "batch_bias",
    "dual_buffer",
    "squared",
    "scaled_squared",
    "global_kv_bias",
    "global_q_bias",
    "global_rel_plus_kv_bias",
    "global_q_and_kv_bias",
    "global_logical_rel_plus_kv_bias",
    "stress_complex_arithmetic",
    "stress_conditional_mask",
    "stress_multi_buffer",
    "stress_global_offset",
    "stress_xor_pattern",
    "debug_global_idx",
)


@lru_cache(maxsize=128)
def score(name):
    """Return (forward, explicit backward) Mods.

    Single-buffer variants use aux[0]; dual uses head/position vectors;
    global_q_and_kv uses Q/K vectors. multi_buffer uses batch,head,Q,K,relative
    vectors (relative table length >=1025). Offsets are sequence-local metadata.
    """
    if name not in SCORE_NAMES:
        raise ValueError(f"unknown score example: {name}")

    @tla.jit
    def forward(s, b, h, q, k, info, aux, scalars):
        z = s
        if tla.const_expr(name == "causal" or name == "causal_v2"):
            z = tla.where(q >= k, s, tla.Float32(float("-inf")))
        elif tla.const_expr(
            name == "rel_bias" or name == "rel_bias_x2" or name == "rel_bias_clamped"
        ):
            distance = tla.abs((q - k).to(tla.Float32))
            if tla.const_expr(name == "rel_bias_clamped"):
                distance = tla.min(distance, tla.Float32(256.0))
            if tla.const_expr(name == "rel_bias_x2"):
                distance = distance * 2.0
            z = s + distance
        elif tla.const_expr(name == "times_two"):
            z = s * 2.0
        elif tla.const_expr(name == "alibi"):
            z = s - exp2(-(h + 1).to(tla.Float32)) * tla.abs((q - k).to(tla.Float32))
        elif tla.const_expr(name == "sliding_window"):
            z = tla.where(tla.abs((q - k).to(tla.Float32)) <= 256.0, s, tla.Float32(float("-inf")))
        elif tla.const_expr(name == "block_diagonal"):
            z = tla.where(q // 64 == k // 64, s, tla.Float32(float("-inf")))
        elif tla.const_expr(name == "batch_bias"):
            z = s + aux[0][b].to(tla.Float32)
        elif tla.const_expr(name == "dual_buffer"):
            z = s + aux[0][h].to(tla.Float32) + aux[1][q].to(tla.Float32)
        elif tla.const_expr(name == "squared"):
            z = s * s
        elif tla.const_expr(name == "scaled_squared"):
            z = aux[0][k + info[3]].to(tla.Float32) * s * s
        elif tla.const_expr(name == "global_kv_bias" or name == "stress_global_offset"):
            z = s + aux[0][k + info[3]].to(tla.Float32)
        elif tla.const_expr(name == "global_q_bias"):
            z = s + aux[0][q + info[2]].to(tla.Float32)
        elif tla.const_expr(
            name == "global_rel_plus_kv_bias" or name == "global_logical_rel_plus_kv_bias"
        ):
            coefficient = 0.1 if tla.const_expr(name == "global_rel_plus_kv_bias") else 0.01
            z = (
                s
                + tla.abs((q - k).to(tla.Float32)) * coefficient
                + aux[0][k + info[3]].to(tla.Float32)
            )
        elif tla.const_expr(name == "global_q_and_kv_bias"):
            z = s + aux[0][q + info[2]].to(tla.Float32) + aux[1][k + info[3]].to(tla.Float32)
        elif tla.const_expr(name == "stress_complex_arithmetic"):
            z = (
                s
                + tla.abs((q - k).to(tla.Float32)) * 0.001
                + aux[0][q + info[2]].to(tla.Float32) * ((b + 1) * (h + 1)).to(tla.Float32) * 0.001
            )
        elif tla.const_expr(name == "stress_conditional_mask"):
            keep = (q >= k) & (tla.abs((q + info[2] - k - info[3]).to(tla.Float32)) <= 512.0)
            z = tla.where(
                keep,
                s + aux[0][k + info[3]].to(tla.Float32),
                tla.Float32(float("-inf")),
            )
        elif tla.const_expr(name == "stress_multi_buffer"):
            index = q - k + 512
            index = tla.where(
                index < 0, tla.Int32(0), tla.where(index > 1024, tla.Int32(1024), index)
            )
            z = (
                s * aux[1][h].to(tla.Float32)
                + aux[0][b].to(tla.Float32)
                + aux[2][q + info[2]].to(tla.Float32)
                + aux[3][k + info[3]].to(tla.Float32)
                + aux[4][index].to(tla.Float32) * 0.1
            )
        elif tla.const_expr(name == "stress_xor_pattern"):
            z = (
                s
                + ((q ^ k) & 255).to(tla.Float32) * 0.001
                + aux[0][k + info[3]].to(tla.Float32) * 0.1
            )
        elif tla.const_expr(name == "debug_global_idx"):
            z = s + (k + info[3]).to(tla.Float32) * 0.001
        return z

    @tla.jit
    def backward(g, s, b, h, q, k, info, aux, scalars):
        dx = g
        if tla.const_expr(name == "times_two"):
            dx = g * 2.0
        elif tla.const_expr(name == "squared"):
            dx = g * 2.0 * s
        elif tla.const_expr(name == "scaled_squared"):
            dx = g * 2.0 * s * aux[0][k + info[3]].to(tla.Float32)
        elif tla.const_expr(name == "stress_multi_buffer"):
            dx = g * aux[1][h].to(tla.Float32)
        return dx

    def reference(s, b, h, q, k, i, a, c):
        distance = torch.abs(q - k).float()
        if name in ("causal", "causal_v2"):
            return torch.where(q >= k, s, -torch.inf)
        if name == "rel_bias":
            return s + distance
        if name == "rel_bias_x2":
            return s + distance * 2
        if name == "rel_bias_clamped":
            return s + distance.clamp(max=256)
        if name == "times_two":
            return s * 2
        if name == "alibi":
            return s - torch.exp2(-(h + 1).float()) * distance
        if name == "sliding_window":
            return torch.where(distance <= 256, s, -torch.inf)
        if name == "block_diagonal":
            return torch.where(q // 64 == k // 64, s, -torch.inf)
        if name == "batch_bias":
            return s + a[0][b].float()
        if name == "dual_buffer":
            return s + a[0][h].float() + a[1][q].float()
        if name == "squared":
            return s * s
        if name == "scaled_squared":
            return a[0][k + i[3]].float() * s * s
        if name in ("global_kv_bias", "stress_global_offset"):
            return s + a[0][k + i[3]].float()
        if name == "global_q_bias":
            return s + a[0][q + i[2]].float()
        if name == "global_rel_plus_kv_bias":
            return s + distance * 0.1 + a[0][k + i[3]].float()
        if name == "global_logical_rel_plus_kv_bias":
            return s + distance * 0.01 + a[0][k + i[3]].float()
        if name == "global_q_and_kv_bias":
            return s + a[0][q + i[2]].float() + a[1][k + i[3]].float()
        if name == "stress_complex_arithmetic":
            return s + distance * 0.001 + a[0][q + i[2]].float() * (b + 1) * (h + 1) * 0.001
        if name == "stress_conditional_mask":
            return torch.where(
                (q >= k) & (torch.abs(q + i[2] - k - i[3]) <= 512),
                s + a[0][k + i[3]].float(),
                -torch.inf,
            )
        if name == "stress_multi_buffer":
            return (
                s * a[1][h].float()
                + a[0][b].float()
                + a[2][q + i[2]].float()
                + a[3][k + i[3]].float()
                + a[4][(q - k + 512).clamp(0, 1024)].float() * 0.1
            )
        if name == "stress_xor_pattern":
            return s + ((q ^ k) & 255).float() * 0.001 + a[0][k + i[3]].float() * 0.1
        if name == "debug_global_idx":
            return s + (k + i[3]).float() * 0.001
        return s

    def ref_backward(g, s, b, h, q, k, i, a, c):
        if name == "times_two":
            return g * 2
        if name == "squared":
            return g * 2 * s
        if name == "scaled_squared":
            return g * 2 * s * a[0][k + i[3]].float()
        if name == "stress_multi_buffer":
            return g * a[1][h].float()
        return g

    return Mod(forward, torch_ref=reference), Mod(backward, torch_ref=ref_backward)
