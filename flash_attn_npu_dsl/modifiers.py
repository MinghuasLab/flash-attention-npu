"""Explicit scalar/vector callback contracts and score/mask stages."""

import inspect
from functools import lru_cache

import catlass.tla as tla
from catlass.core_api import (
    MaskSSA,
    VectorSSA,
    _mask_ssa_type_for_mlir_value,
    _vector_ssa_type_for_mlir_value,
)


def simd(fn):
    """Declare a vector callback without wrapping or vectorizing its body.

    Coordinates are Int32 vectors, scores are Float32 vectors, and masks are
    predicates. The current implementation processes 64 positions per call.
    """
    if not inspect.isfunction(fn):
        raise TypeError("simd expects a Python function or @tla.jit helper")
    fn._use_simd = True
    return fn


def uses_simd(fn) -> bool:
    return getattr(fn, "_use_simd", False) is True


@lru_cache(maxsize=128)
def softcap_score_mod(cap):
    """Keep a stable callback identity for each compile-time softcap value."""

    @simd
    def capped_score(
        score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars=None, limit=cap
    ):
        scaled = score / limit
        # exp(-2*abs(x)) stays in [0, 1], including saturated large scores.
        decay = tla.exp(-2.0 * tla.abs(scaled))
        one = tla.full(1.0, tla.Float32)
        magnitude = ((one - decay) / (one + decay)) * limit
        return tla.where(tla.cmp(scaled, 0.0, "lt"), magnitude * -1.0, magnitude)

    return capped_score


def validate_callback(fn, kind: str) -> None:
    """Admit functions; argument binding follows the actual callback call."""
    if kind not in ("mask", "score"):
        raise ValueError(f"unknown modifier kind: {kind}")
    if fn is None:
        return
    if not inspect.isfunction(fn):
        raise TypeError(f"{kind}_mod must be a Python function or @tla.jit helper")


def validate_modifiers(mask_mod, score_mod) -> tuple[bool, bool]:
    """Validate callbacks and independently select their execution modes."""
    validate_callback(mask_mod, "mask")
    validate_callback(score_mod, "score")
    return uses_simd(mask_mod), uses_simd(score_mod)


@tla.jit
def call_mask_mod(mask_mod, b, h, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
    if tla.const_expr(aux_scalars is not None):
        return mask_mod(b, h, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars)
    return mask_mod(b, h, q_idx, kv_idx, seqlen_info, aux_tensors)


@tla.jit
def call_score_mod(score_mod, score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
    tensors = aux_tensors if aux_tensors is not None else ()
    if tla.const_expr(aux_scalars is not None):
        return score_mod(
            score,
            b,
            h,
            q_idx=q_idx,
            kv_idx=kv_idx,
            seqlen_info=seqlen_info,
            aux_tensors=tensors,
            aux_scalars=aux_scalars,
        )
    return score_mod(
        score,
        b,
        h,
        q_idx=q_idx,
        kv_idx=kv_idx,
        seqlen_info=seqlen_info,
        aux_tensors=tensors,
    )


def _validate_score_vector(score):
    """Validate the callback score vector's dtype and active-lane metadata."""
    if not isinstance(score, VectorSSA):
        raise TypeError("vector score_mod must return a Float32 VectorSSA with 64 lanes")
    descriptor = _vector_ssa_type_for_mlir_value(score.value)
    # None means unknown valid lanes, not register width: Float32 has 64 physical lanes.
    if descriptor.element_type != "f32" or descriptor.valid_lanes not in (None, 64):
        raise TypeError("vector score_mod must return a Float32 VectorSSA with 64 lanes")


def _validate_mask_vector(keep):
    if not isinstance(keep, MaskSSA):
        raise TypeError("vector mask_mod must return a MaskSSA with 64 predicate lanes")
    if _mask_ssa_type_for_mlir_value(keep.value).physical_lanes != 64:
        raise TypeError("vector mask_mod must return a MaskSSA with 64 predicate lanes")


@tla.jit
def apply_score_mod_and_mask_mod(
    score,
    b,
    h,
    q_idx,
    kv_idx,
    info,
    aux_tensors,
    aux_scalars,
    mask_mod,
    score_mod,
    apply_mask: tla.Constexpr[bool],
    use_simd: tla.Constexpr[bool],
):
    """Apply score then mask to one scaled scalar or Float32x64 vector.

    The caller owns iteration and coordinate bounds. Full sparse blocks skip
    only the mask callback; execution mode is fixed before entering the VF.
    """
    if tla.const_expr(score_mod is not None):
        score = call_score_mod(
            score_mod, score, b, h, q_idx, kv_idx, info, aux_tensors, aux_scalars
        )
        if tla.const_expr(use_simd):
            _validate_score_vector(score)
    if tla.const_expr(mask_mod is not None):
        if tla.const_expr(apply_mask):
            keep = call_mask_mod(mask_mod, b, h, q_idx, kv_idx, info, aux_tensors, aux_scalars)
            if tla.const_expr(use_simd):
                _validate_mask_vector(keep)
                padding = tla.full(float("-inf"), tla.Float32)
            else:
                padding = tla.Float32(float("-inf"))
            score = tla.where(keep, score, padding)
    return score


@tla.jit
def process_score_tile_simd(
    scores,
    rows,
    q_start,
    kv_start,
    batch_idx,
    head_idx,
    kv_len,
    scale,
    info,
    aux_tensors,
    aux_scalars,
    window_left,
    window_right,
    mask_mod,
    score_mod,
    apply_scale: tla.Constexpr[bool],
    apply_mask: tla.Constexpr[bool],
    packed_heads: tla.Constexpr[int] = 1,
):
    """Apply built-in bounds and explicit x64 callbacks to a rows-by-128 tile.

    Rows are valid Q positions. Callback KV coordinates clamp padded lanes to
    the final valid key, so auxiliary gathers stay in bounds; the original KV
    coordinates still determine built-in and tail masking after the callbacks.
    """
    # Loop-carried names must also exist in the unpacked specialization.
    first_q = tla.Int32(0)
    first_head = tla.Int32(0)
    if tla.const_expr(packed_heads > 1):
        first_q = (q_start // packed_heads).to(tla.Int32)
        first_head = (q_start - first_q * packed_heads).to(tla.Int32)
    with tla.vec.func(mode="simd"):
        logical_q = first_q
        logical_head = first_head
        full_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        padding = tla.full(float("-inf"), tla.Float32)
        if tla.const_expr(apply_scale):
            multiplier = tla.full(scale, tla.Float32)
        if tla.const_expr(mask_mod is not None or score_mod is not None):
            zero_idx = tla.full(0, tla.Int32)
            b = zero_idx + batch_idx
            h = zero_idx + head_idx
            last_kv = tla.full(kv_len - 1, tla.Int32)
        for row in tla.range(rows):
            q = (q_start + row).to(tla.Int32)
            if tla.const_expr(packed_heads > 1):
                q = logical_q
            if tla.const_expr(mask_mod is not None or score_mod is not None):
                q_idx = zero_idx + q
                if tla.const_expr(packed_heads > 1):
                    row_head = zero_idx + h * packed_heads + logical_head
                else:
                    row_head = h
            for chunk in tla.range_constexpr(2):
                kv_idx = tla.arange(kv_start + chunk * 64, dtype=tla.Int32)
                score_chunk = tla.tile_view(
                    scores, tla.make_shape(1, 64), tla.make_coord(row, chunk)
                )
                score = score_chunk.load()
                if tla.const_expr(apply_scale):
                    score = score * multiplier
                tail_keep = tla.cmp(kv_idx, kv_len, "lt")
                keep = tail_keep
                if tla.const_expr(window_left is not None):
                    keep = tla.bitwise_and(
                        keep,
                        tla.cmp(
                            kv_idx,
                            q + info.seqlen_k - info.seqlen_q - window_left,
                            "ge",
                        ),
                    )
                if tla.const_expr(window_right is not None):
                    keep = tla.bitwise_and(
                        keep,
                        tla.cmp(
                            kv_idx,
                            q + info.seqlen_k - info.seqlen_q + window_right,
                            "le",
                        ),
                    )
                if tla.const_expr(mask_mod is not None or score_mod is not None):
                    callback_kv = tla.where(tail_keep, kv_idx, last_kv)
                    score = apply_score_mod_and_mask_mod(
                        score,
                        b,
                        row_head,
                        q_idx,
                        callback_kv,
                        info,
                        aux_tensors,
                        aux_scalars,
                        mask_mod,
                        score_mod,
                        apply_mask,
                        True,
                    )
                score_chunk.store(tla.where(keep, score, padding), mask=full_mask)
            if tla.const_expr(packed_heads > 1):
                # Each physical row advances one head; wrapping advances Q.
                # Keep the VF row loop free of division and dynamic branches.
                carry = (logical_head == packed_heads - 1).to(tla.Int32)
                logical_q = logical_q + carry
                logical_head = logical_head + 1 - carry * packed_heads


@tla.jit
def process_score_tile_simt(
    scores,
    rows,
    q_start,
    kv_start,
    batch_idx,
    head_idx,
    block_kind,
    aux_tensors,
    aux_scalars,
    q_len,
    kv_len,
    scale,
    info,
    mask_mod,
    score_mod,
    apply_scale: tla.Constexpr[bool] = True,
    packed_heads: tla.Constexpr[int] = 1,
):
    """Dispatch full/partial once before entering the SIMT callback VF."""
    apply_mask = False
    if tla.const_expr(mask_mod is not None):
        apply_mask = block_kind == 0
    if apply_mask:
        _process_score_tile_simt(
            scores,
            rows,
            q_start,
            kv_start,
            batch_idx,
            head_idx,
            True,
            aux_tensors,
            aux_scalars,
            q_len,
            kv_len,
            scale,
            info,
            mask_mod,
            score_mod,
            apply_scale,
            packed_heads,
        )
    else:
        _process_score_tile_simt(
            scores,
            rows,
            q_start,
            kv_start,
            batch_idx,
            head_idx,
            False,
            aux_tensors,
            aux_scalars,
            q_len,
            kv_len,
            scale,
            info,
            mask_mod,
            score_mod,
            apply_scale,
            packed_heads,
        )


@tla.jit
def _process_score_tile_simt(
    scores,
    rows,
    q_start,
    kv_start,
    batch_idx,
    head_idx,
    apply_mask: tla.Constexpr[bool],
    aux_tensors,
    aux_scalars,
    q_len,
    kv_len,
    scale,
    info,
    mask_mod,
    score_mod,
    apply_scale: tla.Constexpr[bool],
    packed_heads: tla.Constexpr[int],
):
    """SIMT iteration over the shared QK FP32 UB tile, one scalar per thread."""
    threads = 512
    columns = 128
    row_step = threads // columns
    if tla.const_expr(packed_heads > 1):
        # Decode the tile origin outside SIMT; lanes advance logical coordinates
        # without forming a remainder-dependent callback predicate.
        base_q = (q_start // packed_heads).to(tla.Int32)
        base_h = (q_start - base_q * packed_heads).to(tla.Int32)
    with tla.vec.func(mode="simt", thread_block_dim=512):
        tid, _, _ = tla.arch.thread_idx()
        q = q_start + tid // columns
        kv = kv_start + tid % columns
        valid_kv = kv < kv_len
        logical_h = head_idx
        if tla.const_expr(packed_heads > 1):
            q = base_q
            logical_h = base_h + tid // columns
            for _ in tla.range_constexpr((row_step - 1 + packed_heads - 1) // packed_heads):
                if logical_h >= packed_heads:
                    logical_h = logical_h - packed_heads
                    q = q + 1
        # A 512-element stride preserves each lane's KV column and advances
        # four Q rows. Keep those coordinates explicit across iterations.
        for index in tla.range(tid, rows * 128, 512):
            result = tla.Float32(float("-inf"))
            # Callbacks may gather auxiliary memory: never invoke at padded indices.
            if (q < q_len // packed_heads) & valid_kv:
                result = scores[index]
                if tla.const_expr(apply_scale):
                    result = result * scale
                result = apply_score_mod_and_mask_mod(
                    result,
                    batch_idx,
                    logical_h + head_idx * packed_heads if packed_heads > 1 else logical_h,
                    q,
                    kv,
                    info,
                    aux_tensors,
                    aux_scalars,
                    mask_mod,
                    score_mod,
                    apply_mask,
                    False,
                )
            scores[index] = result
            if tla.const_expr(packed_heads > 1):
                # 512 threads / 128 columns advance each lane by four packed rows.
                q = q + row_step // packed_heads
                logical_h = logical_h + row_step % packed_heads
                if logical_h >= packed_heads:
                    logical_h = logical_h - packed_heads
                    q = q + 1
            else:
                q = q + row_step
