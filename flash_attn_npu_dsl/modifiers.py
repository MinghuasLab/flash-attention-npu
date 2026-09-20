"""Explicit scalar/vector callback contracts and score/mask stages."""

import inspect

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
):
    """Apply built-in bounds and explicit x64 callbacks to a rows-by-128 tile.

    Rows are valid Q positions. Callback KV coordinates clamp padded lanes to
    the final valid key, so auxiliary gathers stay in bounds; the original KV
    coordinates still determine built-in and tail masking after the callbacks.
    """
    with tla.vec.func(mode="simd"):
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
            if tla.const_expr(mask_mod is not None or score_mod is not None):
                q_idx = zero_idx + q
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
                        h,
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
):
    """SIMT iteration over the shared QK FP32 UB tile, one scalar per thread."""
    with tla.vec.func(mode="simt", thread_block_dim=512):
        tid, _, _ = tla.arch.thread_idx()
        # Match the thread count with a constant stride to specialize the loop.
        for index in tla.range(tid, rows * 128, 512):
            q = q_start + index // 128
            kv = kv_start + index % 128
            result = tla.Float32(float("-inf"))
            # Callbacks may gather auxiliary memory: never invoke at padded indices.
            if q < q_len and kv < kv_len:
                result = scores[index]
                if tla.const_expr(apply_scale):
                    result = result * scale
                result = apply_score_mod_and_mask_mod(
                    result,
                    batch_idx,
                    head_idx,
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
