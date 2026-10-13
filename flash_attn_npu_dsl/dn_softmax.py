"""KV-by-Q softmax: each SIMD lane carries one query/head coordinate."""

import catlass.tla as tla
from .dn_full import full_dn_softmax
from .modifiers import (
    call_mask_mod,
    call_score_mod,
    _validate_mask_vector,
    _validate_score_vector,
)


@tla.jit
def _dn_impl(
    scores,
    probs,
    running_max,
    running_sum,
    alpha,
    first,
    scale,
    columns,
    rows,
    probability_dtype,
    q_start,
    kv_start,
    batch,
    head,
    packed_heads,
    info,
    mask_mod,
    score_mod,
    aux_tensors,
    aux_scalars,
    apply_mask,
    window_left,
    window_right,
):
    flat_probs = tla.make_tensor(
        probs.ptr, tla.make_layout(tla.make_shape(8320), tla.make_stride(1))
    )
    first_q = (q_start // packed_heads).to(tla.Int32)
    first_h = (q_start - first_q * packed_heads).to(tla.Int32)
    with tla.vec.func(mode="simd"):
        all_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        row_mask, _ = tla.update_mask(rows, dtype=tla.Float32)
        packed_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
        zero_idx = tla.full(0, tla.Int32)
        lane = tla.arange(0, dtype=tla.Int32)
        q = zero_idx + first_q
        local_h = lane + first_h
        if tla.const_expr(packed_heads == 1):
            q = lane + first_q
            local_h = zero_idx
        else:
            for carry in tla.range_constexpr((63 + packed_heads - 1) // packed_heads):
                overflow = tla.cmp(local_h, packed_heads, "ge")
                local_h = tla.where(overflow, local_h - packed_heads, local_h)
                q = tla.where(overflow, q + 1, q)
        q = tla.min(q, zero_idx + info.seqlen_q - 1)
        h = zero_idx + head * packed_heads + local_h
        b = zero_idx + batch
        neg_inf = tla.full(float("-inf"), tla.Float32)
        zero = tla.full(0.0, tla.Float32)
        tile_max = neg_inf
        for max_kv in tla.range(128):
            src = tla.tile_view(scores, tla.make_shape(1, 64), tla.make_coord(max_kv, 0))
            value = src.load() * tla.full(scale, tla.Float32)
            raw_k = zero_idx + kv_start + max_kv
            k = tla.min(raw_k, zero_idx + info.seqlen_k - 1)
            if tla.const_expr(score_mod is not None):
                value = call_score_mod(score_mod, value, b, h, q, k, info, aux_tensors, aux_scalars)
                _validate_score_vector(value)
            keep = tla.bitwise_and(row_mask, tla.cmp(zero_idx + max_kv, columns, "lt"))
            if tla.const_expr(mask_mod is not None):
                if tla.const_expr(apply_mask):
                    callback_keep = call_mask_mod(
                        mask_mod, b, h, q, k, info, aux_tensors, aux_scalars
                    )
                    _validate_mask_vector(callback_keep)
                    keep = tla.bitwise_and(keep, callback_keep)
            else:
                center = q + info.seqlen_k - info.seqlen_q
                if tla.const_expr(window_left is not None):
                    keep = tla.bitwise_and(keep, tla.cmp(raw_k, center - window_left, "ge"))
                if tla.const_expr(window_right is not None):
                    keep = tla.bitwise_and(keep, tla.cmp(raw_k, center + window_right, "le"))
            value = tla.where(keep, value, neg_inf)
            tile_max = tla.max(tile_max, value)
            src.store(value)
        old_max = neg_inf
        if tla.const_expr(not first):
            old_max = running_max.load()
        maximum = tla.max(old_max, tile_max)
        safe_max = tla.where(tla.cmp(maximum, neg_inf, "gt"), maximum, zero)
        rescale = tla.where(tla.cmp(old_max, neg_inf, "gt"), tla.exp(old_max - safe_max), zero)
        running_max.store(maximum, mask=row_mask)
        alpha.store(rescale, mask=row_mask)
        total_lo = zero
        total_hi = zero
        cast = tla.params.CastParams(
            reg_slot=tla.params.RegSlot.ZERO,
            sat_mode=tla.params.SatMode.SAT,
            round_mode=tla.params.RoundMode.CAST_ROUND,
        )
        for exp_kv in tla.range(64):
            src_lo = tla.tile_view(scores, tla.make_shape(1, 64), tla.make_coord(exp_kv, 0))
            src_hi = tla.tile_view(scores, tla.make_shape(1, 64), tla.make_coord(exp_kv + 64, 0))
            prob_lo = tla.exp(tla.sub(src_lo.load(), safe_max, mask=all_mask), mask=all_mask)
            prob_hi = tla.exp(tla.sub(src_hi.load(), safe_max, mask=all_mask), mask=all_mask)
            total_lo = tla.add(total_lo, prob_lo, mask=all_mask)
            total_hi = tla.add(total_hi, prob_hi, mask=all_mask)
            low = prob_lo.to(probability_dtype, cast, mask=all_mask)
            high = prob_hi.to(probability_dtype, cast, mask=all_mask)
            compact, unused = tla.deinterleave(low, high)
            # Two 65x64 padded zN halves. The first 64 lanes hold KV=k,
            # the second 64 hold KV=k+64; each 16-lane block advances 65 blocks.
            dst = tla.tile_view(flat_probs, tla.make_shape(1), tla.make_coord(exp_kv * 16))
            dst.store(
                compact, params=tla.params.BlockStoreParams(block_stride=65), mask=packed_mask
            )
        total = tla.add(total_lo, total_hi, mask=all_mask)
        if tla.const_expr(not first):
            total = running_sum.load() * rescale + total
        running_sum.store(total, mask=row_mask)


@tla.jit
def native_dn_softmax(
    scores,
    probs,
    running_max,
    running_sum,
    alpha,
    first,
    scale,
    columns,
    rows,
    probability_dtype,
    q_start,
    kv_start,
    batch,
    head,
    packed_heads,
    info,
    mask_mod,
    score_mod,
    aux_tensors,
    aux_scalars,
    apply_mask,
    window_left,
    window_right,
):
    # A full metadata block must also be physically full. Never bypass score
    # modification or an intrinsic window.
    eligible = False
    if tla.const_expr(
        score_mod is None and window_left is None and window_right is None and scale > 0
    ):
        eligible = (rows == 64) & (columns == 128)
        if tla.const_expr(mask_mod is not None):
            eligible = eligible & (not apply_mask)
    if eligible:
        full_dn_softmax(
            scores, probs, running_max, running_sum, alpha, first, scale, probability_dtype
        )
    else:
        _general_native_dn_softmax(
            scores,
            probs,
            running_max,
            running_sum,
            alpha,
            first,
            scale,
            columns,
            rows,
            probability_dtype,
            q_start,
            kv_start,
            batch,
            head,
            packed_heads,
            info,
            mask_mod,
            score_mod,
            aux_tensors,
            aux_scalars,
            apply_mask,
            window_left,
            window_right,
        )


@tla.jit
def _general_native_dn_softmax(
    scores,
    probs,
    running_max,
    running_sum,
    alpha,
    first,
    scale,
    columns,
    rows,
    probability_dtype,
    q_start,
    kv_start,
    batch,
    head,
    packed_heads,
    info,
    mask_mod,
    score_mod,
    aux_tensors,
    aux_scalars,
    apply_mask,
    window_left,
    window_right,
):
    if first:
        if apply_mask:
            _dn_impl(
                scores,
                probs,
                running_max,
                running_sum,
                alpha,
                True,
                scale,
                columns,
                rows,
                probability_dtype,
                q_start,
                kv_start,
                batch,
                head,
                packed_heads,
                info,
                mask_mod,
                score_mod,
                aux_tensors,
                aux_scalars,
                True,
                window_left,
                window_right,
            )
        else:
            _dn_impl(
                scores,
                probs,
                running_max,
                running_sum,
                alpha,
                True,
                scale,
                columns,
                rows,
                probability_dtype,
                q_start,
                kv_start,
                batch,
                head,
                packed_heads,
                info,
                mask_mod,
                score_mod,
                aux_tensors,
                aux_scalars,
                False,
                window_left,
                window_right,
            )
    else:
        if apply_mask:
            _dn_impl(
                scores,
                probs,
                running_max,
                running_sum,
                alpha,
                False,
                scale,
                columns,
                rows,
                probability_dtype,
                q_start,
                kv_start,
                batch,
                head,
                packed_heads,
                info,
                mask_mod,
                score_mod,
                aux_tensors,
                aux_scalars,
                True,
                window_left,
                window_right,
            )
        else:
            _dn_impl(
                scores,
                probs,
                running_max,
                running_sum,
                alpha,
                False,
                scale,
                columns,
                rows,
                probability_dtype,
                q_start,
                kv_start,
                batch,
                head,
                packed_heads,
                info,
                mask_mod,
                score_mod,
                aux_tensors,
                aux_scalars,
                False,
                window_left,
                window_right,
            )
