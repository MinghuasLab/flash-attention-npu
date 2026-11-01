"""Online softmax for fully initialized 128-by-64 KV-by-Q tiles.

Unmodified tiles require valid coordinates and positive scale. Padded tiles
initialize invalid positions to -inf and preserve the empty-row state. Modified
tiles also contain pre-scaled scores and therefore use unit scale.
"""

import catlass.tla as tla


@tla.jit
def full_dn_softmax(
    scores,
    probs,
    running_max,
    running_sum,
    alpha,
    first,
    scale,
    dtype,
    masked: tla.Constexpr[bool] = False,
):
    source = tla.make_tensor(scores.ptr, tla.make_layout(tla.make_shape(8192), tla.make_stride(1)))
    packed = tla.make_tensor(probs.ptr, tla.make_layout(tla.make_shape(8320), tla.make_stride(1)))
    if first:
        with tla.vec.func(mode="simd"):
            full = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            alpha.store(tla.full(0.0, tla.Float32), mask=full)
        _full_dn_impl(source, running_max, running_sum, packed, scale, dtype, alpha, True, masked)
    else:
        _full_dn_impl(source, running_max, running_sum, packed, scale, dtype, alpha, False, masked)


@tla.jit
def _full_dn_impl(
    src,
    mx,
    sm,
    packed,
    scale,
    ptype,
    alpha,
    first: tla.Constexpr,
    masked: tla.Constexpr[bool],
):
    with tla.vec.func(mode="simd"):
        full = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        full16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
        m0 = tla.full(float("-inf"), tla.Float32)
        m1 = tla.full(float("-inf"), tla.Float32)
        m2 = tla.full(float("-inf"), tla.Float32)
        m3 = tla.full(float("-inf"), tla.Float32)
        for i in tla.range(32):
            max_x0 = tla.tile_view(src, tla.make_shape(64), tla.make_coord(i * 4)).load()
            max_x1 = tla.tile_view(src, tla.make_shape(64), tla.make_coord(i * 4 + 1)).load()
            max_x2 = tla.tile_view(src, tla.make_shape(64), tla.make_coord(i * 4 + 2)).load()
            max_x3 = tla.tile_view(src, tla.make_shape(64), tla.make_coord(i * 4 + 3)).load()
            m0 = tla.max(m0, max_x0, mask=full)
            m1 = tla.max(m1, max_x1, mask=full)
            m2 = tla.max(m2, max_x2, mask=full)
            m3 = tla.max(m3, max_x3, mask=full)
        m0 = tla.max(m0, m2, mask=full)
        m1 = tla.max(m1, m3, mask=full)
        m0 = tla.max(m0, m1, mask=full)
        m0 = tla.mul(m0, scale, mask=full)
        if tla.const_expr(not first):
            previous_max = mx.load()
            previous_sum = sm.load()
            m0 = tla.max(m0, previous_max, mask=full)
        if tla.const_expr(masked):
            mx.store(m0, mask=full)
            # Every padded/masked position has been initialized to -inf.
            # Keep -inf as the true running maximum, but avoid -inf - -inf
            # in empty rows when forming exponentials and rescale factors.
            m0 = tla.where(
                tla.cmp(m0, tla.full(float("-inf"), tla.Float32), "gt"),
                m0,
                tla.full(0.0, tla.Float32),
            )
        if tla.const_expr(not first):
            rescale = tla.exp(tla.sub(previous_max, m0, mask=full), mask=full)
            previous_sum = tla.mul(rescale, previous_sum, mask=full)
            alpha.store(rescale, mask=full)
        if tla.const_expr(not masked):
            mx.store(m0, mask=full)
        s0 = tla.full(0.0, tla.Float32)
        s1 = tla.full(0.0, tla.Float32)
        s2 = tla.full(0.0, tla.Float32)
        s3 = tla.full(0.0, tla.Float32)
        cast = tla.params.CastParams(
            reg_slot=tla.params.RegSlot.ZERO,
            sat_mode=tla.params.SatMode.SAT,
            round_mode=tla.params.RoundMode.CAST_ROUND,
        )
        for k in tla.range(32):
            x0 = tla.tile_view(src, tla.make_shape(64), tla.make_coord(k)).load()
            x1 = tla.tile_view(src, tla.make_shape(64), tla.make_coord(k + 32)).load()
            x2 = tla.tile_view(src, tla.make_shape(64), tla.make_coord(k + 64)).load()
            x3 = tla.tile_view(src, tla.make_shape(64), tla.make_coord(k + 96)).load()
            x0 = tla.mul(x0, scale, mask=full)
            x1 = tla.mul(x1, scale, mask=full)
            x2 = tla.mul(x2, scale, mask=full)
            x3 = tla.mul(x3, scale, mask=full)
            p0 = tla.exp(tla.sub(x0, m0, mask=full), mask=full)
            p1 = tla.exp(tla.sub(x1, m0, mask=full), mask=full)
            p2 = tla.exp(tla.sub(x2, m0, mask=full), mask=full)
            p3 = tla.exp(tla.sub(x3, m0, mask=full), mask=full)
            c0 = p0.to(ptype, cast, mask=full)
            c2 = p2.to(ptype, cast, mask=full)
            pair02, unused02 = tla.deinterleave(c0, c2)
            c1 = p1.to(ptype, cast, mask=full)
            c3 = p3.to(ptype, cast, mask=full)
            pair13, unused13 = tla.deinterleave(c1, c3)
            dst0 = tla.tile_view(packed, tla.make_shape(1), tla.make_coord(k * 16))
            dst0.store(pair02, params=tla.params.BlockStoreParams(block_stride=65), mask=full16)
            s0 = tla.add(s0, p0, mask=full)
            s2 = tla.add(s2, p2, mask=full)
            dst1 = tla.tile_view(packed, tla.make_shape(1), tla.make_coord(k * 16 + 512))
            dst1.store(pair13, params=tla.params.BlockStoreParams(block_stride=65), mask=full16)
            s1 = tla.add(s1, p1, mask=full)
            s3 = tla.add(s3, p3, mask=full)
        s0 = tla.add(s0, s2, mask=full)
        s1 = tla.add(s1, s3, mask=full)
        s0 = tla.add(s0, s1, mask=full)
        if tla.const_expr(not first):
            updated_sum = tla.add(previous_sum, s0, mask=full)
            sm.store(updated_sum, mask=full)
        else:
            sm.store(s0, mask=full)
