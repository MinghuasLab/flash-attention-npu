# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""SIMD online softmax and output correction with fixed 128-column scratch."""

import catlass.tla as tla


def _row(tensor, row):
    return tla.tile_view(tensor, tla.make_shape(1), tla.make_coord(row))


def _chunk(tensor, row, column):
    return tla.tile_view(tensor, tla.make_shape(1, 64), tla.make_coord(row, column))


def _broadcast(tensor, row):
    return _row(tensor, row).load(
        params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
    )


def _store_scalar(tensor, row, value, mask):
    _row(tensor, row).store(value, params=tla.params.UnalignStoreParams(), mask=mask)


@tla.jit
def _load_score_row(scores, row, scale, columns, apply_scale):
    """Read the same scaled, tail-masked values in both softmax row passes."""
    score_row = tla.tile_view(scores, tla.make_shape(1, 128), tla.make_coord(row, 0))
    odd, even = score_row.load(
        params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_DINTLV_B32)
    )
    if tla.const_expr(apply_scale):
        first_mask, _ = tla.update_mask((columns + 1) // 2, dtype=tla.Float32)
        second_mask, _ = tla.update_mask(columns // 2, dtype=tla.Float32)
        multiplier = tla.full(scale, tla.Float32)
        padding = tla.full(float("-inf"), tla.Float32)
        odd = tla.where(first_mask, odd * multiplier, padding)
        even = tla.where(second_mask, even * multiplier, padding)
    return odd, even


@tla.jit
def online_softmax_row(
    scores,
    packed_probs,
    tile_sum,
    row_max,
    scale,
    columns,
    apply_scale,
    row,
    probability_dtype,
    full_mask,
    packed_mask,
    one_mask,
):
    """Compute exp(score - row_max), its row sum and packed low-precision P."""
    odd, even = _load_score_row(scores, row, scale, columns, apply_scale)
    safe_max = _broadcast(row_max, row)
    odd_exp = tla.exp(odd - safe_max, mask=full_mask)
    even_exp = tla.exp(even - safe_max, mask=full_mask)
    total = tla.full((odd_exp + even_exp).reduce(tla.ReductionOp.ADD, mask=full_mask), tla.Float32)
    _store_scalar(tile_sum, row, total, one_mask)
    zero_slot = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    one_slot = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ONE,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    packed = tla.bitwise_or(
        odd_exp.to(probability_dtype, zero_slot, mask=full_mask),
        even_exp.to(probability_dtype, one_slot, mask=full_mask),
        mask=packed_mask,
    )
    destination = tla.tile_view(packed_probs, tla.make_shape(1, 128), tla.make_coord(row, 0))
    destination.store(
        packed,
        params=tla.params.BlockStoreParams(block_stride=65),
        mask=packed_mask,
    )


@tla.jit
def _online_softmax_tile_impl(
    scores,
    packed_probs,
    running_max,
    running_sum,
    alpha,
    first,
    scale,
    columns,
    apply_scale,
    rows,
    probability_dtype,
    row_max,
    tile_sum,
):
    # Four SIMD VFs exchange row statistics through UB. The row loops write
    # separate scratch; running state is updated only by whole-row-vector VFs.
    with tla.vec.func(mode="simd"):
        full_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
        for max_row in tla.range(rows):
            row_odd, row_even = _load_score_row(scores, max_row, scale, columns, apply_scale)
            tile_row_max = tla.full(
                tla.max(row_odd, row_even).reduce(tla.ReductionOp.MAX, mask=full_mask),
                tla.Float32,
            )
            _store_scalar(row_max, max_row, tile_row_max, one_mask)

    with tla.vec.func(mode="simd"):
        full_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        rows_mask, _ = tla.update_mask(rows, dtype=tla.Float32)
        zero = tla.full(0.0, tla.Float32)
        neg_inf = tla.full(float("-inf"), tla.Float32)
        tile_max = tla.where(rows_mask, row_max.load(), neg_inf)
        old_max = neg_inf
        if tla.const_expr(not first):
            old_max = tla.where(rows_mask, running_max.load(), neg_inf)
        maximum = tla.max(old_max, tile_max, mask=full_mask)
        valid = tla.cmp(maximum, neg_inf, "gt", mask=full_mask)
        old_valid = tla.cmp(old_max, neg_inf, "gt", mask=full_mask)
        safe_max = tla.where(valid, maximum, zero)
        rescale = tla.where(old_valid, tla.exp(old_max - safe_max, mask=full_mask), zero)
        running_max.store(maximum, mask=rows_mask)
        alpha.store(rescale, mask=rows_mask)
        row_max.store(safe_max, mask=rows_mask)

    with tla.vec.func(mode="simd"):
        full_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        packed_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
        one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
        for exp_row in tla.range(rows):
            online_softmax_row(
                scores,
                packed_probs,
                tile_sum,
                row_max,
                scale,
                columns,
                apply_scale,
                exp_row,
                probability_dtype,
                full_mask,
                packed_mask,
                one_mask,
            )

    with tla.vec.func(mode="simd"):
        rows_mask, _ = tla.update_mask(rows, dtype=tla.Float32)
        zero = tla.full(0.0, tla.Float32)
        total = tla.where(rows_mask, tile_sum.load(), zero)
        next_sum = total
        if tla.const_expr(not first):
            old_sum = tla.where(rows_mask, running_sum.load(), zero)
            rescale = tla.where(rows_mask, alpha.load(), zero)
            next_sum = old_sum * rescale + total
        running_sum.store(next_sum, mask=rows_mask)


@tla.jit
def online_softmax_tile(
    scores,
    packed_probs,
    running_max,
    running_sum,
    alpha,
    first,
    scale,
    columns,
    apply_scale,
    rows,
    probability_dtype,
    row_max,
    tile_sum,
):
    """Choose initialization or rescaling once per tile, outside the SIMD row loops."""
    if first:
        _online_softmax_tile_impl(
            scores,
            packed_probs,
            running_max,
            running_sum,
            alpha,
            True,
            scale,
            columns,
            apply_scale,
            rows,
            probability_dtype,
            row_max,
            tile_sum,
        )
    else:
        _online_softmax_tile_impl(
            scores,
            packed_probs,
            running_max,
            running_sum,
            alpha,
            False,
            scale,
            columns,
            apply_scale,
            rows,
            probability_dtype,
            row_max,
            tile_sum,
        )


def _load_pv_row(pv, row, head_mask):
    odd, even = _chunk(pv, row, 0).load(
        params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_DINTLV_B32)
    )
    if tla.const_expr(head_mask is not None):
        # FIX writes only D columns into a 128-column row. Each deinterleaved
        # register has D/2 valid lanes; never propagate stale scratch padding.
        zero = tla.full(0.0, tla.Float32)
        odd = tla.where(head_mask, odd, zero)
        even = tla.where(head_mask, even, zero)
    return odd, even


def output_first_row(pv, output, row, full_mask, head_mask):
    odd, even = _load_pv_row(pv, row, head_mask)
    _chunk(output, row, 0).store(odd, mask=full_mask)
    _chunk(output, row, 1).store(even, mask=full_mask)


def output_update_row(pv, output, alpha, row, full_mask, head_mask):
    rescale = _broadcast(alpha, row)
    odd, even = _load_pv_row(pv, row, head_mask)
    first = _chunk(output, row, 0)
    second = _chunk(output, row, 1)
    first.store(first.load() * rescale + odd, mask=full_mask)
    second.store(second.load() * rescale + even, mask=full_mask)


def output_finalize_row(output, running_max, running_sum, lse, row, full_mask, one_mask):
    total = _broadcast(running_sum, row)
    zero = tla.full(0.0, tla.Float32)
    valid = tla.cmp(total, zero, "gt", mask=full_mask)
    safe_total = tla.where(valid, total, tla.full(1.0, tla.Float32))
    if tla.const_expr(lse is not None):
        logsumexp = tla.where(
            valid,
            _broadcast(running_max, row) + tla.log(safe_total, mask=full_mask),
            tla.full(float("-inf"), tla.Float32),
        )
        _store_scalar(lse, row, logsumexp, one_mask)
    for chunk in tla.range_constexpr(2):
        dst = _chunk(output, row, chunk)
        dst.store(tla.where(valid, dst.load() / safe_total, zero), mask=full_mask)


def output_pack_row(output, packed_output, row, output_dtype, full_mask, packed_mask):
    zero_slot = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    one_slot = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ONE,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    packed = tla.bitwise_or(
        _chunk(output, row, 0).load().to(output_dtype, zero_slot, mask=full_mask),
        _chunk(output, row, 1).load().to(output_dtype, one_slot, mask=full_mask),
        mask=packed_mask,
    )
    tla.tile_view(packed_output, tla.make_shape(1, 128), tla.make_coord(row, 0)).store(
        packed, mask=packed_mask
    )


def output_empty_row(output, lse, row, full_mask, one_mask):
    zero = tla.full(0.0, tla.Float32)
    for chunk in tla.range_constexpr(2):
        _chunk(output, row, chunk).store(zero, mask=full_mask)
    if tla.const_expr(lse is not None):
        _store_scalar(lse, row, tla.full(float("-inf"), tla.Float32), one_mask)


@tla.jit
def _output_update_tile_impl(
    pv,
    output,
    alpha,
    running_max,
    running_sum,
    lse,
    rows,
    valid_idx,
    active_tiles,
    first,
    head_dim,
):
    """Update O, order its UB stores, then normalize the final tile."""
    with tla.vec.func(mode="simd"):
        full_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
        head_mask = None
        if tla.const_expr(head_dim < 128):
            head_mask, _ = tla.update_mask(head_dim // 2, dtype=tla.Float32)
        for row in tla.range(rows):
            if tla.const_expr(first):
                output_first_row(pv, output, row, full_mask, head_mask)
            else:
                output_update_row(pv, output, alpha, row, full_mask, head_mask)
        tla.local_mem_bar(
            tla.params.MemType.VEC_STORE,
            tla.params.MemType.VEC_LOAD,
        )
        if valid_idx == active_tiles - 1:
            for normalize_row in tla.range(rows):
                output_finalize_row(
                    output,
                    running_max,
                    running_sum,
                    lse,
                    normalize_row,
                    full_mask,
                    one_mask,
                )


@tla.jit
def output_update_tile(
    pv,
    output,
    alpha,
    running_max,
    running_sum,
    lse,
    rows,
    valid_idx,
    active_tiles,
    head_dim=128,
):
    """Initialize O from the first PV tile; rescale and accumulate later PV tiles.

    Select initialization outside the SIMD row loops. Complete O's UB stores
    before reading them for normalization, which runs only after the last PV tile.
    """
    if valid_idx == 0:
        _output_update_tile_impl(
            pv,
            output,
            alpha,
            running_max,
            running_sum,
            lse,
            rows,
            valid_idx,
            active_tiles,
            True,
            head_dim,
        )
    else:
        _output_update_tile_impl(
            pv,
            output,
            alpha,
            running_max,
            running_sum,
            lse,
            rows,
            valid_idx,
            active_tiles,
            False,
            head_dim,
        )
