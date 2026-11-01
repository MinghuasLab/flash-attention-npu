"""KV-by-Q softmax: each SIMD lane carries one query/head coordinate."""

import catlass.tla as tla
from .dn_full import full_dn_softmax
from .modifiers import (
    call_mask_mod,
    call_score_mod,
    _validate_mask_vector,
    process_score_tile_simt,
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
    scalar_callbacks: tla.Constexpr[bool] = False,
):
    common = (scores, probs, running_max, running_sum, alpha, first)
    coordinates = (q_start, kv_start, batch, head, packed_heads, info)
    needs_simt = False
    if tla.const_expr(scalar_callbacks):
        if tla.const_expr(score_mod is not None):
            needs_simt = True
        else:
            needs_simt = apply_mask
    if needs_simt:
        flat = tla.make_tensor(
            scores.ptr, tla.make_layout(tla.make_shape(8192), tla.make_stride(1))
        )
        block_kind = 0 if apply_mask else 1
        # FIX -> SIMT -> SIMD transfers ownership of the same score bank.
        tla.pipe_barrier(tla.pipes.ALL)
        process_score_tile_simt(
            flat,
            rows,
            q_start,
            kv_start,
            batch,
            head,
            block_kind,
            aux_tensors,
            aux_scalars,
            info.seqlen_q,
            info.seqlen_k,
            scale,
            info,
            mask_mod,
            score_mod,
            dn_layout=True,
        )
        tla.pipe_barrier(tla.pipes.ALL)
        if tla.const_expr(window_left is None and window_right is None):
            # SIMT wrote all 128x64 physical positions, including padding.
            # Reuse the same optimized reduction/exp/P packing as raw full
            # blocks, with safe empty-row normalization for modified scores.
            full_dn_softmax(
                *common,
                1.0,
                probability_dtype,
                masked=True,
            )
        else:
            _general_native_dn_softmax(
                *common,
                1.0,
                columns,
                rows,
                probability_dtype,
                *coordinates,
                None,
                None,
                aux_tensors,
                aux_scalars,
                False,
                window_left,
                window_right,
            )
    else:
        # Full blocks without a score callback remain unmodified and retain
        # the existing full-tile path. Never invoke a scalar callback as SIMD.
        _vector_native_dn_softmax(
            *common,
            scale,
            columns,
            rows,
            probability_dtype,
            *coordinates,
            None if scalar_callbacks else mask_mod,
            None if scalar_callbacks else score_mod,
            aux_tensors,
            aux_scalars,
            False if scalar_callbacks else apply_mask,
            window_left,
            window_right,
        )


@tla.jit
def _fill_dn_padding(scores, rows, columns, clear_rows: tla.Constexpr):
    with tla.vec.func(mode="simd"):
        full = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        padding = tla.full(float("-inf"), tla.Float32)
        if tla.const_expr(clear_rows):
            invalid_rows = tla.cmp(tla.arange(0, dtype=tla.Int32), rows, "ge")
            for k in tla.range(columns):
                dst = tla.tile_view(scores, tla.make_shape(1, 64), tla.make_coord(k, 0))
                dst.store(padding, mask=invalid_rows)
        for k in tla.range(columns, 128):
            dst = tla.tile_view(scores, tla.make_shape(1, 64), tla.make_coord(k, 0))
            dst.store(padding, mask=full)


@tla.jit
def _vector_native_dn_softmax(
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
    # Only a proved fully visible tile with unmodified scores can use this
    # path. Padding is initialized explicitly before the common full-tile VF.
    eligible = False
    if tla.const_expr(score_mod is None and scale > 0):
        eligible = True
        if tla.const_expr(mask_mod is not None):
            eligible = eligible & (not apply_mask)
        else:
            # Every key must lie inside every query's window. The largest Q
            # gives the tightest left bound; the smallest Q gives the right.
            if tla.const_expr(window_left is not None):
                last_q = (q_start + rows - 1) // packed_heads
                eligible = eligible & (
                    kv_start >= last_q + info.seqlen_k - info.seqlen_q - window_left
                )
            if tla.const_expr(window_right is not None):
                first_q = q_start // packed_heads
                eligible = eligible & (
                    kv_start + columns - 1 <= first_q + info.seqlen_k - info.seqlen_q + window_right
                )
    if eligible:
        if (rows == 64) & (columns == 128):
            full_dn_softmax(
                scores, probs, running_max, running_sum, alpha, first, scale, probability_dtype
            )
        else:
            if rows == 64:
                _fill_dn_padding(scores, rows, columns, False)
            else:
                _fill_dn_padding(scores, rows, columns, True)
            full_dn_softmax(
                scores,
                probs,
                running_max,
                running_sum,
                alpha,
                first,
                scale,
                probability_dtype,
                masked=True,
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
