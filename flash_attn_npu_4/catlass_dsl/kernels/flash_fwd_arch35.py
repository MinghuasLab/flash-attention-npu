"""M128 DN/NZ forward with a continuous pipeline across query tasks.

Three m/l state slots keep query producers independent of delayed PV consumers.
Every query contributes at least one scheduling event; EMPTY events preserve
queue progress without reading K/V. Only the last physical task drains two
stages. A separate LSE output buffer protects m/l reuse from asynchronous DMA.
"""

import catlass.tla as tla
from .common import (
    pipeline_load_rows,
    l1_matrix as _matrix,
    vector_signal as _vset,
    vector_wait as _vwait,
    cube_signal as _cset,
    cube_wait as _cwait,
)
from .window import window_block_bounds
from .callbacks import prepare_scores_simt


@tla.jit
def _mm(
    a,
    b,
    ap,
    bp,
    cp,
    abslot,
    cslot,
    abload0,
    abload1,
    abfree0,
    abfree1,
    cfree0,
    cfree1,
    done0,
    done1,
    init_c: tla.Constexpr = True,
    publish=True,
):
    la = tla.make_tensor_like(ap, a, tla.arch.zN)
    lb = tla.make_tensor_like(bp, b, tla.arch.nZ)
    cr = tla.make_tensor(cp, tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)))
    lc = tla.make_tensor_like(cp, cr, tla.arch.L0Clayout)
    if abslot == 0:
        tla.wait_flag(abfree0)
    else:
        tla.wait_flag(abfree1)
    tla.copy(la, a)
    tla.copy(lb, b)
    if abslot == 0:
        tla.set_flag(abload0)
        tla.wait_flag(abload0)
    else:
        tla.set_flag(abload1)
        tla.wait_flag(abload1)
    if tla.const_expr(init_c):
        if cslot == 0:
            tla.wait_flag(cfree0)
        else:
            tla.wait_flag(cfree1)
    tla.mmad(lc, la, lb, init_c=init_c)
    if abslot == 0:
        tla.set_flag(abfree0)
    else:
        tla.set_flag(abfree1)
    if publish:
        if cslot == 0:
            tla.set_flag(done0)
        else:
            tla.set_flag(done1)
    return lc


@tla.jit
def _fix(
    lc,
    ubptr,
    slot,
    done0,
    done1,
    free0,
    free1,
    channel0,
    channel1,
    split_n: tla.Constexpr,
):
    if slot == 0:
        tla.wait_flag(done0)
        _cwait(channel0, tla.arch.FIX)
    else:
        tla.wait_flag(done1)
        _cwait(channel1, tla.arch.FIX)
    if tla.const_expr(split_n):
        # The descriptor keeps the full N extent; each AIV destination stride
        # is N/2. The hardware dual-destination split owns those half columns.
        ub = tla.make_tensor(
            ubptr, tla.make_layout(tla.make_shape(128, 128), tla.make_stride(64, 1))
        )
        tla.copy(
            ub,
            lc,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_N),
        )
    else:
        ub = tla.make_tensor(
            ubptr, tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1))
        )
        tla.copy(
            ub,
            lc,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
        )
    if slot == 0:
        tla.set_flag(free0)
        _cset(channel0, tla.arch.FIX)
    else:
        tla.set_flag(free1)
        _cset(channel1, tla.arch.FIX)


@tla.jit
def _cache_mask_pair(
    mask_mod: tla.Constexpr,
    b,
    head,
    qstart,
    cached_vi,
    cached_k0,
    cached_k1,
    cached_mask0,
    cached_mask1,
    aux_tensors,
    aux_scalars,
    info,
    bounds_checked: tla.Constexpr,
    temporal_aux: tla.Constexpr = False,
):
    # Packed unsigned storage can reuse a word across adjacent key predicates.
    # Other auxiliary storage retains the neighboring-key thread layout.
    if tla.const_expr(temporal_aux and bounds_checked):
        cached_bytes0 = tla.make_tensor(
            tla.recast_ptr(cached_mask0.ptr, dtype=tla.Int8),
            tla.make_layout(tla.make_shape(128, 32), tla.make_stride(32, 1)),
        )
        cached_bytes1 = tla.make_tensor(
            tla.recast_ptr(cached_mask1.ptr, dtype=tla.Int8),
            tla.make_layout(tla.make_shape(128, 32), tla.make_stride(32, 1)),
        )
        with tla.vec.func(mode="simt", thread_block_dim=256):
            cached_tid, _, _ = tla.arch.thread_idx()
            cached_chunk = (cached_tid >> 4) & 3
            cached_pair = (cached_tid & 15) | ((cached_tid >> 2) & 16)
            cached_second = (cached_tid & 128) != 0
            cached_start = cached_k0
            if cached_second:
                cached_start = cached_k1
            cached_qbase = (qstart & 0x7FFFFF80) | ((cached_vi & 1) << 6)
            cached_keybase = (cached_start & 0x7FFFFF80) | (cached_chunk << 5)
            # Form dense key-bit planes before expanding to native mask bytes.
            # This keeps callback evaluation separate from predicate layout.
            cached_planes = ()
            for cached_lane in tla.range_constexpr(2):
                cached_q = cached_qbase | (cached_pair << 1) | cached_lane
                cached_plane = tla.Int32(0)
                for cached_key_lane in tla.range_constexpr(32):
                    cached_key = cached_keybase | cached_key_lane
                    if mask_mod(
                        tla.Float32(0.0),
                        b,
                        head,
                        cached_q,
                        cached_key,
                        aux_tensors,
                        aux_scalars,
                        info,
                        bounds_checked,
                    ):
                        cached_plane = cached_plane | (tla.Int32(1) << cached_key_lane)
                cached_planes = cached_planes + (cached_plane,)
            cached_words = ()
            for cached_word_lane in tla.range_constexpr(8):
                cached_bits0 = (cached_planes[0] >> (cached_word_lane * 4)) & 15
                cached_bits1 = (cached_planes[1] >> (cached_word_lane * 4)) & 15
                cached_bits0 = (cached_bits0 | (cached_bits0 << 14)) & 0x00030003
                cached_bits1 = (cached_bits1 | (cached_bits1 << 14)) & 0x00030003
                cached_bits0 = (cached_bits0 | (cached_bits0 << 7)) & 0x01010101
                cached_bits1 = (cached_bits1 | (cached_bits1 << 7)) & 0x01010101
                cached_words = cached_words + (cached_bits0 | (cached_bits1 << 4),)
            if cached_second:
                for cached_word_lane in tla.range_constexpr(8):
                    for cached_key_lane in tla.range_constexpr(4):
                        cached_bytes1[
                            (cached_chunk << 5) | (cached_word_lane * 4 + cached_key_lane),
                            cached_pair,
                        ] = (cached_words[cached_word_lane] >> (cached_key_lane * 8)).to(tla.Int8)
            else:
                for cached_word_lane in tla.range_constexpr(8):
                    for cached_key_lane in tla.range_constexpr(4):
                        cached_bytes0[
                            (cached_chunk << 5) | (cached_word_lane * 4 + cached_key_lane),
                            cached_pair,
                        ] = (cached_words[cached_word_lane] >> (cached_key_lane * 8)).to(tla.Int8)
    else:
        with tla.vec.func(mode="simt", thread_block_dim=1024):
            cached_tid, _, _ = tla.arch.thread_idx()
            # Neighboring lanes share query rows and traverse adjacent keys.
            cached_k = cached_tid & 127
            cached_group = cached_tid >> 7
            # Valid local block starts are nonnegative, aligned and fit Int32.
            # Disjoint fields expose those bounds to callback index arithmetic.
            cached_qbase = (qstart & 0x7FFFFF80) | ((cached_vi & 1) << 6)
            cached_key0 = (cached_k0 & 0x7FFFFF80) | cached_k
            cached_key1 = (cached_k1 & 0x7FFFFF80) | cached_k
            cached_word0 = tla.Int32(0)
            cached_word1 = tla.Int32(0)
            for cached_lane in tla.range_constexpr(8):
                cached_q = cached_qbase | (cached_group * 8 + cached_lane)
                if mask_mod(
                    tla.Float32(0.0),
                    b,
                    head,
                    cached_q,
                    cached_key0,
                    aux_tensors,
                    aux_scalars,
                    info,
                    bounds_checked,
                ):
                    cached_word0 = cached_word0 | (1 << (cached_lane * 4))
                if mask_mod(
                    tla.Float32(0.0),
                    b,
                    head,
                    cached_q,
                    cached_key1,
                    aux_tensors,
                    aux_scalars,
                    info,
                    bounds_checked,
                ):
                    cached_word1 = cached_word1 | (1 << (cached_lane * 4))
            cached_mask0[cached_k, cached_group] = cached_word0
            cached_mask1[cached_k, cached_group] = cached_word1


@tla.jit
def _cache_mask(
    mask_mod: tla.Constexpr,
    b,
    head,
    qstart,
    vi,
    kstart,
    visibility,
    aux_tensors,
    aux_scalars,
    info,
    bounds_checked: tla.Constexpr,
):
    with tla.vec.func(mode="simt", thread_block_dim=1024):
        mask_tid, _, _ = tla.arch.thread_idx()
        mask_k = mask_tid & 127
        mask_group = mask_tid >> 7
        mask_qbase = (qstart & 0x7FFFFF80) | ((vi & 1) << 6)
        mask_key = (kstart & 0x7FFFFF80) | mask_k
        mask_word = tla.Int32(0)
        for mask_lane in tla.range_constexpr(8):
            mask_q = mask_qbase | (mask_group * 8 + mask_lane)
            if mask_mod(
                tla.Float32(0.0),
                b,
                head,
                mask_q,
                mask_key,
                aux_tensors,
                aux_scalars,
                info,
                bounds_checked,
            ):
                mask_word = mask_word | (1 << (mask_lane * 4))
        visibility[mask_k, mask_group] = mask_word


@tla.jit
def _cache_mask_eight(
    mask_mod: tla.Constexpr,
    b,
    head,
    qstart,
    vi,
    kstart,
    visibility,
    aux_tensors,
    aux_scalars,
    info,
    complete: tla.Constexpr,
):
    # Independent words share query-coordinate work; complete chunks have no
    # per-map bounds branches. The caller uses only aligned query/key tiles.
    with tla.vec.func(mode="simt", thread_block_dim=1024):
        tid, _, _ = tla.arch.thread_idx()
        key = tid // 8
        group = tid % 8
        word0 = tla.Int32(0)
        word1 = tla.Int32(0)
        word2 = tla.Int32(0)
        word3 = tla.Int32(0)
        word4 = tla.Int32(0)
        word5 = tla.Int32(0)
        word6 = tla.Int32(0)
        word7 = tla.Int32(0)
        for lane in tla.range_constexpr(8):
            qi = qstart + vi * 64 + group * 8 + lane
            if tla.const_expr(complete) or kstart + 0 < info[1]:
                if mask_mod(
                    tla.Float32(0.0),
                    b,
                    head,
                    qi,
                    kstart + 0 + key,
                    aux_tensors,
                    aux_scalars,
                    info,
                    True,
                ):
                    word0 = word0 | (1 << (lane * 4))
            if tla.const_expr(complete) or kstart + 128 < info[1]:
                if mask_mod(
                    tla.Float32(0.0),
                    b,
                    head,
                    qi,
                    kstart + 128 + key,
                    aux_tensors,
                    aux_scalars,
                    info,
                    True,
                ):
                    word1 = word1 | (1 << (lane * 4))
            if tla.const_expr(complete) or kstart + 256 < info[1]:
                if mask_mod(
                    tla.Float32(0.0),
                    b,
                    head,
                    qi,
                    kstart + 256 + key,
                    aux_tensors,
                    aux_scalars,
                    info,
                    True,
                ):
                    word2 = word2 | (1 << (lane * 4))
            if tla.const_expr(complete) or kstart + 384 < info[1]:
                if mask_mod(
                    tla.Float32(0.0),
                    b,
                    head,
                    qi,
                    kstart + 384 + key,
                    aux_tensors,
                    aux_scalars,
                    info,
                    True,
                ):
                    word3 = word3 | (1 << (lane * 4))
            if tla.const_expr(complete) or kstart + 512 < info[1]:
                if mask_mod(
                    tla.Float32(0.0),
                    b,
                    head,
                    qi,
                    kstart + 512 + key,
                    aux_tensors,
                    aux_scalars,
                    info,
                    True,
                ):
                    word4 = word4 | (1 << (lane * 4))
            if tla.const_expr(complete) or kstart + 640 < info[1]:
                if mask_mod(
                    tla.Float32(0.0),
                    b,
                    head,
                    qi,
                    kstart + 640 + key,
                    aux_tensors,
                    aux_scalars,
                    info,
                    True,
                ):
                    word5 = word5 | (1 << (lane * 4))
            if tla.const_expr(complete) or kstart + 768 < info[1]:
                if mask_mod(
                    tla.Float32(0.0),
                    b,
                    head,
                    qi,
                    kstart + 768 + key,
                    aux_tensors,
                    aux_scalars,
                    info,
                    True,
                ):
                    word6 = word6 | (1 << (lane * 4))
            if tla.const_expr(complete) or kstart + 896 < info[1]:
                if mask_mod(
                    tla.Float32(0.0),
                    b,
                    head,
                    qi,
                    kstart + 896 + key,
                    aux_tensors,
                    aux_scalars,
                    info,
                    True,
                ):
                    word7 = word7 | (1 << (lane * 4))
        visibility[0 + key, group] = word0
        visibility[128 + key, group] = word1
        visibility[256 + key, group] = word2
        visibility[384 + key, group] = word3
        visibility[512 + key, group] = word4
        visibility[640 + key, group] = word5
        visibility[768 + key, group] = word6
        visibility[896 + key, group] = word7


@tla.jit
def _score_max_pass_impl(
    scores,
    scale,
    bv,
    hv,
    qi,
    kstart,
    aux_tensors,
    aux_scalars,
    score_mod: tla.Constexpr,
    mask_mod: tla.Constexpr,
    apply_mask: tla.Constexpr,
    visibility,
    compact_mask: tla.Constexpr,
    info=(),
    bounded: tla.Constexpr = True,
    prepared: tla.Constexpr = False,
):
    zero = tla.full(0.0, dtype=tla.Float32)
    neginf = tla.full(float("-inf"), dtype=tla.Float32)
    full_score_store = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
    blockm0 = neginf
    blockm1 = neginf
    blockm2 = neginf
    blockm3 = neginf
    blockm4 = neginf
    blockm5 = neginf
    blockm6 = neginf
    blockm7 = neginf
    for key_group in tla.range(16):
        for key_lane in tla.range_constexpr(8):
            key_row = key_group * 8 + key_lane
            ki = tla.add(tla.full(0, dtype=tla.Int32), kstart + key_row)
            st = tla.tile_view(scores, tla.make_shape(1, 64), tla.make_coord(key_row, 0))
            z = st.load()
            if tla.const_expr(not prepared):
                x = tla.mul(z, scale)
                callback_q = qi
                callback_k = ki
                if tla.const_expr(len(info) > 0 and bounded):
                    callback_q = tla.min(qi, info[0] - 1)
                    callback_k = tla.min(ki, info[1] - 1)
                z = score_mod(x, bv, hv, callback_q, callback_k, aux_tensors, aux_scalars, info)
            if tla.const_expr(compact_mask):
                visible = tla.tile_view(
                    visibility, tla.make_shape(1, 8), tla.make_coord(key_row, 0)
                ).load(tla.params.MaskLoadParams())
                z = tla.where(visible, z, neginf)
            if tla.const_expr(apply_mask):
                z = tla.where(
                    mask_mod(zero, bv, hv, callback_q, callback_k, aux_tensors, aux_scalars, info),
                    z,
                    neginf,
                )
            if tla.const_expr(len(info) > 0 and bounded):
                z = tla.where(tla.cmp(qi, info[0], mode="lt"), z, neginf)
                z = tla.where(tla.cmp(ki, info[1], mode="lt"), z, neginf)
            st.store(z, mask=full_score_store)
            if tla.const_expr(key_lane == 0):
                blockm0 = tla.max(blockm0, z)
            elif tla.const_expr(key_lane == 1):
                blockm1 = tla.max(blockm1, z)
            elif tla.const_expr(key_lane == 2):
                blockm2 = tla.max(blockm2, z)
            elif tla.const_expr(key_lane == 3):
                blockm3 = tla.max(blockm3, z)
            elif tla.const_expr(key_lane == 4):
                blockm4 = tla.max(blockm4, z)
            elif tla.const_expr(key_lane == 5):
                blockm5 = tla.max(blockm5, z)
            elif tla.const_expr(key_lane == 6):
                blockm6 = tla.max(blockm6, z)
            else:
                blockm7 = tla.max(blockm7, z)
    blockm = tla.max(
        tla.max(tla.max(blockm0, blockm1), tla.max(blockm2, blockm3)),
        tla.max(tla.max(blockm4, blockm5), tla.max(blockm6, blockm7)),
    )
    return blockm


@tla.jit
def _score_max_pass(
    scores,
    scale,
    bv,
    hv,
    qi,
    kstart,
    aux_tensors,
    aux_scalars,
    score_mod: tla.Constexpr,
    mask_mod: tla.Constexpr,
    apply_mask: tla.Constexpr,
    visibility,
    compact_mask: tla.Constexpr,
    info=(),
    needs_bounds=False,
    prepared: tla.Constexpr = False,
):
    if tla.const_expr(len(info) == 0):
        return _score_max_pass_impl(
            scores,
            scale,
            bv,
            hv,
            qi,
            kstart,
            aux_tensors,
            aux_scalars,
            score_mod,
            mask_mod,
            apply_mask,
            visibility,
            compact_mask,
            info,
            False,
            prepared,
        )
    else:
        result = tla.full(float("-inf"), dtype=tla.Float32)
        if needs_bounds:
            result = _score_max_pass_impl(
                scores,
                scale,
                bv,
                hv,
                qi,
                kstart,
                aux_tensors,
                aux_scalars,
                score_mod,
                mask_mod,
                apply_mask,
                visibility,
                compact_mask,
                info,
                True,
                prepared,
            )
        else:
            result = _score_max_pass_impl(
                scores,
                scale,
                bv,
                hv,
                qi,
                kstart,
                aux_tensors,
                aux_scalars,
                score_mod,
                mask_mod,
                apply_mask,
                visibility,
                compact_mask,
                info,
                False,
                prepared,
            )
        return result


@tla.jit
def _accumulate_pv(
    acc_source,
    accub,
    pvub,
    aub,
    out_pv,
    m_pv,
    l_pv,
    lse_pv,
    dtype: tla.Constexpr,
    final: tla.Constexpr,
):
    full_pv_store = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
    half_pv_store = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype)
    if tla.const_expr(final):
        final_m = m_pv.load()
        final_l = l_pv.load()
        safe_l = tla.where(
            tla.cmp(final_l, 0.0, mode="gt"), final_l, tla.full(1.0, dtype=tla.Float32)
        )
        lse_pv.store(tla.add(final_m, tla.log(safe_l)))
        l_pv.store(tla.div(tla.full(1.0, dtype=tla.Float32), safe_l))
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
    for rr in tla.range(64):
        ar = tla.tile_view(aub, tla.make_shape(1), tla.make_coord(rr)).load(
            tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
        )
        if tla.const_expr(final):
            inv_l = tla.tile_view(l_pv, tla.make_shape(1), tla.make_coord(rr)).load(
                tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
            )
        for cc in tla.range(2):
            old = tla.tile_view(acc_source, tla.make_shape(1, 64), tla.make_coord(rr, cc)).load()
            cur = tla.tile_view(pvub, tla.make_shape(1, 64), tla.make_coord(rr, cc)).load()
            value = tla.add(tla.mul(old, ar), cur)
            if tla.const_expr(final):
                oval = tla.mul(value, inv_l).to(
                    dtype,
                    tla.params.CastParams(
                        round_mode=tla.params.RoundMode.CAST_ROUND,
                        reg_slot=tla.params.RegSlot.ZERO,
                    ),
                )
                tla.tile_view(out_pv, tla.make_shape(1, 64), tla.make_coord(rr, cc)).store(
                    oval,
                    tla.params.NormalStoreParams(store_dist=tla.params.StoreDist.DIST_PACK_B32),
                    mask=half_pv_store,
                )
            else:
                tla.tile_view(accub, tla.make_shape(1, 64), tla.make_coord(rr, cc)).store(
                    value, mask=full_pv_store
                )
    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


@tla.jit
def _summarize_mask_pair_vf(visibility0, visibility1, summary, has_second):
    lanes = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)
    for part in tla.range_constexpr(2):
        present = tla.full(0, dtype=tla.Int32)
        if part == 0 or has_second:
            visibility = visibility0 if tla.const_expr(part == 0) else visibility1
            for chunk in tla.range(16):
                words = tla.tile_view(
                    visibility, tla.make_shape(8, 8), tla.make_coord(chunk, 0)
                ).load()
                present = tla.max(present, words)
        maximum = present.reduce(tla.ReductionOp.MAX, mask=lanes)
        tla.tile_view(summary, tla.make_shape(1), tla.make_coord(part)).store(
            maximum,
            tla.params.NormalStoreParams(store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32),
        )
    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


@tla.jit
def _empty_mask_prob(pflat, safem, dtype: tla.Constexpr):
    # Keep the IEEE expression when previous visible scores contain NaN/Inf.
    prob = tla.exp(tla.sub(tla.full(float("-inf"), dtype=tla.Float32), safem))
    p16 = prob.to(
        dtype,
        tla.params.CastParams(
            round_mode=tla.params.RoundMode.CAST_ROUND, reg_slot=tla.params.RegSlot.ZERO
        ),
    )
    packed, unused = tla.deinterleave(p16, p16)
    mask16 = tla.update_mask(64, dtype=dtype)[0]
    for group in tla.range(16):
        for lane in tla.range_constexpr(8):
            tla.tile_view(pflat, tla.make_shape(1, 64), tla.make_coord(group * 8 + lane, 0)).store(
                packed, tla.params.BlockStoreParams(block_stride=129), mask=mask16
            )
    return tla.mul(prob, 128.0)


@tla.jit
def _pair_softmax(
    s0,
    s1,
    p0,
    p1,
    mstate,
    lstate,
    astate,
    has_second,
    iteration,
    scale,
    b,
    head,
    qstart,
    vi,
    kstart,
    aux_tensors,
    aux_scalars,
    score_mod: tla.Constexpr,
    mask_mod: tla.Constexpr,
    dtype: tla.Constexpr,
    kstart_second,
    full0,
    full1,
    masked: tla.Constexpr,
    indexed: tla.Constexpr,
    visibility0,
    visibility1,
    empty0,
    empty1,
    mask_summary,
    classify_mask: tla.Constexpr,
    reuse_empty_probs: tla.Constexpr = False,
    info=(),
    tail0=False,
    tail1=False,
):
    if tla.const_expr(classify_mask):
        _summarize_mask_pair_vf(visibility0, visibility1, mask_summary, has_second)
        vf_empty0 = mask_summary[0] == 0
        vf_empty1 = mask_summary[1] == 0
    else:
        vf_empty0 = empty0
        vf_empty1 = empty1
    zero = tla.full(0.0, dtype=tla.Float32)
    neginf = tla.full(float("-inf"), dtype=tla.Float32)
    bv = tla.add(tla.full(0, dtype=tla.Int32), b)
    hv = tla.add(tla.full(0, dtype=tla.Int32), head)
    qi = tla.arange(qstart + vi * 64, dtype=tla.Int32)
    blockm = neginf
    blockm1 = neginf
    if not vf_empty0:
        if tla.const_expr(masked):
            if full0:
                blockm = _score_max_pass(
                    s0,
                    scale,
                    bv,
                    hv,
                    qi,
                    kstart,
                    aux_tensors,
                    aux_scalars,
                    score_mod,
                    mask_mod,
                    False,
                    s0,
                    False,
                    info,
                    tail0,
                )
            else:
                blockm = _score_max_pass(
                    s0,
                    scale,
                    bv,
                    hv,
                    qi,
                    kstart,
                    aux_tensors,
                    aux_scalars,
                    score_mod,
                    mask_mod,
                    not indexed,
                    visibility0,
                    indexed,
                    info,
                    tail0,
                )
        else:
            blockm = _score_max_pass(
                s0,
                scale,
                bv,
                hv,
                qi,
                kstart,
                aux_tensors,
                aux_scalars,
                score_mod,
                mask_mod,
                False,
                s0,
                False,
                info,
                tail0,
            )
    if has_second:
        if not vf_empty1:
            if tla.const_expr(masked):
                if full1:
                    blockm1 = _score_max_pass(
                        s1,
                        scale,
                        bv,
                        hv,
                        qi,
                        kstart_second,
                        aux_tensors,
                        aux_scalars,
                        score_mod,
                        mask_mod,
                        False,
                        s1,
                        False,
                        info,
                        tail1,
                    )
                else:
                    blockm1 = _score_max_pass(
                        s1,
                        scale,
                        bv,
                        hv,
                        qi,
                        kstart_second,
                        aux_tensors,
                        aux_scalars,
                        score_mod,
                        mask_mod,
                        not indexed,
                        visibility1,
                        indexed,
                        info,
                        tail1,
                    )
            else:
                blockm1 = _score_max_pass(
                    s1,
                    scale,
                    bv,
                    hv,
                    qi,
                    kstart_second,
                    aux_tensors,
                    aux_scalars,
                    score_mod,
                    mask_mod,
                    False,
                    s1,
                    False,
                    info,
                    tail1,
                )
        blockm = tla.max(blockm, blockm1)
    oldm = neginf
    oldl = zero
    newm = blockm
    if iteration != 0:
        oldm = mstate.load()
        oldl = lstate.load()
        newm = tla.max(oldm, blockm)
    safem = tla.where(tla.cmp(newm, neginf, mode="eq"), zero, newm)
    alpha = zero
    if iteration != 0:
        alpha = tla.where(tla.cmp(oldl, zero, mode="gt"), tla.exp(tla.sub(oldm, safem)), zero)
    mstate.store(newm)
    astate.store(alpha)
    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
    blockl0 = zero
    blockl1 = zero
    blockl2 = zero
    blockl3 = zero
    blockl4 = zero
    blockl5 = zero
    blockl6 = zero
    blockl7 = zero
    mask16 = tla.update_mask(64, dtype=dtype)[0]
    for part in tla.range_constexpr(2):
        if (part == 0 and not indexed) or (part == 1 and has_second):
            scores = s0 if tla.const_expr(part == 0) else s1
            pflat = p0 if tla.const_expr(part == 0) else p1
            part_empty = vf_empty0 if tla.const_expr(part == 0) else vf_empty1
            if part_empty:
                blockl0 = tla.add(blockl0, _empty_mask_prob(pflat, safem, dtype))
            else:
                for prob_group in tla.range(16):
                    for prob_lane in tla.range_constexpr(8):
                        row = prob_group * 8 + prob_lane
                        z = tla.tile_view(
                            scores, tla.make_shape(1, 64), tla.make_coord(row, 0)
                        ).load()
                        prob = tla.exp(tla.sub(z, safem))
                        if tla.const_expr(prob_lane == 0):
                            blockl0 = tla.add(blockl0, prob)
                        elif tla.const_expr(prob_lane == 1):
                            blockl1 = tla.add(blockl1, prob)
                        elif tla.const_expr(prob_lane == 2):
                            blockl2 = tla.add(blockl2, prob)
                        elif tla.const_expr(prob_lane == 3):
                            blockl3 = tla.add(blockl3, prob)
                        elif tla.const_expr(prob_lane == 4):
                            blockl4 = tla.add(blockl4, prob)
                        elif tla.const_expr(prob_lane == 5):
                            blockl5 = tla.add(blockl5, prob)
                        elif tla.const_expr(prob_lane == 6):
                            blockl6 = tla.add(blockl6, prob)
                        else:
                            blockl7 = tla.add(blockl7, prob)
                        p16 = prob.to(
                            dtype,
                            tla.params.CastParams(
                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                reg_slot=tla.params.RegSlot.ZERO,
                            ),
                        )
                        packed, unused = tla.deinterleave(p16, p16)
                        tla.tile_view(pflat, tla.make_shape(1, 64), tla.make_coord(row, 0)).store(
                            packed, tla.params.BlockStoreParams(block_stride=129), mask=mask16
                        )
    blockl = tla.add(
        tla.add(tla.add(blockl0, blockl1), tla.add(blockl2, blockl3)),
        tla.add(tla.add(blockl4, blockl5), tla.add(blockl6, blockl7)),
    )
    if tla.const_expr(reuse_empty_probs):
        if has_second & vf_empty0 & vf_empty1:
            # Both score tiles are exactly -inf and use the same joint max.
            # Their probabilities are identical (0 or IEEE NaN per query).
            # Account for both here; P0 can reuse the existing P1 allocation.
            blockl = tla.add(blockl, blockl)
    if iteration == 0:
        lstate.store(blockl)
    else:
        lstate.store(tla.add(tla.mul(oldl, alpha), blockl))
    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


@tla.jit
def _pair_first_prob(scores, pflat, mstate, lstate, dtype: tla.Constexpr, empty):
    # Finish the first tile after the pair's shared max update. The caller
    # retains ownership of this score and the probability destination.
    zero = tla.full(0.0, dtype=tla.Float32)
    neginf = tla.full(float("-inf"), dtype=tla.Float32)
    newm = mstate.load()
    safem = tla.where(tla.cmp(newm, neginf, mode="eq"), zero, newm)
    blockl0 = zero
    blockl1 = zero
    blockl2 = zero
    blockl3 = zero
    blockl4 = zero
    blockl5 = zero
    blockl6 = zero
    blockl7 = zero
    mask16 = tla.update_mask(64, dtype=dtype)[0]
    if empty:
        blockl0 = _empty_mask_prob(pflat, safem, dtype)
    else:
        for prob_group in tla.range(16):
            for prob_lane in tla.range_constexpr(8):
                row = prob_group * 8 + prob_lane
                z = tla.tile_view(scores, tla.make_shape(1, 64), tla.make_coord(row, 0)).load()
                prob = tla.exp(tla.sub(z, safem))
                if tla.const_expr(prob_lane == 0):
                    blockl0 = tla.add(blockl0, prob)
                elif tla.const_expr(prob_lane == 1):
                    blockl1 = tla.add(blockl1, prob)
                elif tla.const_expr(prob_lane == 2):
                    blockl2 = tla.add(blockl2, prob)
                elif tla.const_expr(prob_lane == 3):
                    blockl3 = tla.add(blockl3, prob)
                elif tla.const_expr(prob_lane == 4):
                    blockl4 = tla.add(blockl4, prob)
                elif tla.const_expr(prob_lane == 5):
                    blockl5 = tla.add(blockl5, prob)
                elif tla.const_expr(prob_lane == 6):
                    blockl6 = tla.add(blockl6, prob)
                else:
                    blockl7 = tla.add(blockl7, prob)
                p16 = prob.to(
                    dtype,
                    tla.params.CastParams(
                        round_mode=tla.params.RoundMode.CAST_ROUND, reg_slot=tla.params.RegSlot.ZERO
                    ),
                )
                packed, unused = tla.deinterleave(p16, p16)
                tla.tile_view(pflat, tla.make_shape(1, 64), tla.make_coord(row, 0)).store(
                    packed, tla.params.BlockStoreParams(block_stride=129), mask=mask16
                )
    blockl = tla.add(
        tla.add(tla.add(blockl0, blockl1), tla.add(blockl2, blockl3)),
        tla.add(tla.add(blockl4, blockl5), tla.add(blockl6, blockl7)),
    )
    lstate.store(tla.add(lstate.load(), blockl))
    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


@tla.jit
def _normalize_pair_pv(pv, out, m, l, lse, dtype: tla.Constexpr):
    lv = l.load()
    safe_l = tla.where(tla.cmp(lv, 0.0, mode="gt"), lv, tla.full(1.0, dtype=tla.Float32))
    lse.store(tla.add(m.load(), tla.log(safe_l)))
    l.store(tla.div(tla.full(1.0, dtype=tla.Float32), safe_l))
    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
    mask = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype)
    for row in tla.range(64):
        inv = tla.tile_view(l, tla.make_shape(1), tla.make_coord(row)).load(
            tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
        )
        for col in tla.range(2):
            value = tla.tile_view(pv, tla.make_shape(1, 64), tla.make_coord(row, col)).load()
            value = tla.mul(value, inv).to(
                dtype,
                tla.params.CastParams(
                    round_mode=tla.params.RoundMode.CAST_ROUND, reg_slot=tla.params.RegSlot.ZERO
                ),
            )
            tla.tile_view(out, tla.make_shape(1, 64), tla.make_coord(row, col)).store(
                value,
                tla.params.NormalStoreParams(store_dist=tla.params.StoreDist.DIST_PACK_B32),
                mask=mask,
            )
    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


@tla.jit
def _paired_pipeline(
    q,
    k,
    v,
    out,
    lse,
    aux_tensors,
    aux_scalars,
    partial_count,
    partial_index,
    full_count,
    full_index,
    mask_batch,
    mask_heads,
    batch,
    q_heads,
    kv_heads,
    max_q,
    max_k,
    q_blocks,
    scale,
    total_q,
    varlen: tla.Constexpr,
    score_mod: tla.Constexpr,
    mask_mod: tla.Constexpr,
    dense: tla.Constexpr,
    implicit_window: tla.Constexpr,
    sequence_major: tla.Constexpr,
    indexed: tla.Constexpr,
    dense_mask: tla.Constexpr = False,
    ragged: tla.Constexpr = False,
    temporal_aux: tla.Constexpr = False,
):
    # Two visible KV tiles share one softmax update and one FP32 PV accumulation.
    # Sparse block indices may be nonconsecutive; carry both through the pipeline.
    # PV ping-pong buffers retain the running accumulator, freeing space for
    # four QK buffers (two groups in flight). Aligned SIMT borrows the
    # consumed second score for the first probability tile; ragged sparse
    # SIMT retains its shared score and independent tail staging buffer.
    # Final output compacts the consumed PV buffer's prefix.
    # Tail staging borrows the output slot after its DMA completes.
    # Mask predicates reuse on-chip buffers without a global workspace.
    dtype = q.ptr.dtype
    reuse_score_buffers = (dense_mask and indexed) or (indexed and not ragged)
    head_group = tla.as_numeric(1)
    if tla.const_expr(dense_mask and indexed):
        # Remove common power-of-two factors from the per-core query stride.
        # Grouping heads also keeps K/V reuse within each group of query tasks.
        for group_shift in tla.range_constexpr(1, 8):
            if (q_heads % (1 << group_shift) == 0) & (
                tla.arch.block_num() % (1 << group_shift) == 0
            ):
                head_group = tla.as_numeric(1 << group_shift)
    q_stride = q_heads * 128 if tla.const_expr(sequence_major[0]) else 128
    k_stride = kv_heads * 128 if tla.const_expr(sequence_major[1]) else 128
    q_head_stride = 128 if tla.const_expr(sequence_major[0]) else max_q * 128
    k_head_stride = 128 if tla.const_expr(sequence_major[1]) else max_k * 128
    l_head_stride = total_q if tla.const_expr(varlen[0]) else max_q
    l_batch_stride = max_q if tla.const_expr(varlen[0]) else q_heads * max_q
    lq = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
    lk0 = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
    lk1 = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
    lv0 = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
    lv1 = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
    lp0 = tla.allocate(32768, dtype, tla.AddressSpace.l1, 512)
    lp1 = tla.allocate(32768, dtype, tla.AddressSpace.l1, 512)
    lp2 = tla.allocate(32768, dtype, tla.AddressSpace.l1, 512)
    la0 = tla.allocate(16384, dtype, tla.AddressSpace.l0a, 512)
    la1 = tla.allocate(16384, dtype, tla.AddressSpace.l0a, 512)
    lb0 = tla.allocate(16384, dtype, tla.AddressSpace.l0b, 512)
    lb1 = tla.allocate(16384, dtype, tla.AddressSpace.l0b, 512)
    ls0 = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0c, 512)
    ls1 = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0c, 512)
    lo0 = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0c, 512)
    lo1 = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0c, 512)
    us00 = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
    us01 = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
    us10 = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
    if tla.const_expr(indexed and not reuse_score_buffers):
        us11 = us01
    else:
        us11 = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
    up0 = tla.allocate(8256, dtype, tla.AddressSpace.ub, 256)
    if tla.const_expr(reuse_score_buffers):
        up1 = up0
    else:
        up1 = tla.allocate(8256, dtype, tla.AddressSpace.ub, 256)
    uo0 = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
    uo1 = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
    mask_summary_ptr = uo0
    if tla.const_expr(dense_mask and indexed):
        mask_summary_ptr = tla.allocate(64, tla.Int32, tla.AddressSpace.ub, 256)
    if tla.const_expr(reuse_score_buffers):
        # One probability/output allocation frees space for a fourth score.
        # EMPTY output uses pfree0. Real final output compacts its own PV
        # allocation and returns that PV ownership after MTE3 completes.
        uout = up0
    else:
        uout = tla.allocate(8192, dtype, tla.AddressSpace.ub, 256)
    um0 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    um1 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    um2 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    ul0 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    ul1 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    ul2 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    alpha0 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    alpha1 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    alpha2 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    ulse_out = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    qfree = tla.flag("pair_qfree", tla.arch.MTE1, tla.arch.MTE2)
    qload = tla.flag("pair_qload", tla.arch.MTE2, tla.arch.MTE1)
    kfree0 = tla.flag("pair_kfree0", tla.arch.MTE1, tla.arch.MTE2)
    kfree1 = tla.flag("pair_kfree1", tla.arch.MTE1, tla.arch.MTE2)
    kload0 = tla.flag("pair_kload0", tla.arch.MTE2, tla.arch.MTE1)
    kload1 = tla.flag("pair_kload1", tla.arch.MTE2, tla.arch.MTE1)
    vfree0 = tla.flag("pair_vfree0", tla.arch.MTE1, tla.arch.MTE2)
    vfree1 = tla.flag("pair_vfree1", tla.arch.MTE1, tla.arch.MTE2)
    vload0 = tla.flag("pair_vload0", tla.arch.MTE2, tla.arch.MTE1)
    vload1 = tla.flag("pair_vload1", tla.arch.MTE2, tla.arch.MTE1)
    abfree0 = tla.flag("pair_abfree0", tla.arch.CUBE, tla.arch.MTE1)
    abfree1 = tla.flag("pair_abfree1", tla.arch.CUBE, tla.arch.MTE1)
    abload0 = tla.flag("pair_abload0", tla.arch.MTE1, tla.arch.CUBE)
    abload1 = tla.flag("pair_abload1", tla.arch.MTE1, tla.arch.CUBE)
    sfree0 = tla.flag("pair_sfree0", tla.arch.FIX, tla.arch.CUBE)
    sfree1 = tla.flag("pair_sfree1", tla.arch.FIX, tla.arch.CUBE)
    sdone0 = tla.flag("pair_sdone0", tla.arch.CUBE, tla.arch.FIX)
    sdone1 = tla.flag("pair_sdone1", tla.arch.CUBE, tla.arch.FIX)
    ofree0 = tla.flag("pair_ofree0", tla.arch.FIX, tla.arch.CUBE)
    ofree1 = tla.flag("pair_ofree1", tla.arch.FIX, tla.arch.CUBE)
    odone0 = tla.flag("pair_odone0", tla.arch.CUBE, tla.arch.FIX)
    odone1 = tla.flag("pair_odone1", tla.arch.CUBE, tla.arch.FIX)
    pfree0 = tla.flag("pair_pfree0", tla.arch.MTE3, tla.arch.VECTOR)
    pfree1 = tla.flag("pair_pfree1", tla.arch.MTE3, tla.arch.VECTOR)
    pready0 = tla.flag("pair_pready0", tla.arch.VECTOR, tla.arch.MTE3)
    pready1 = tla.flag("pair_pready1", tla.arch.VECTOR, tla.arch.MTE3)
    outfree = tla.flag("pair_outfree", tla.arch.MTE3, tla.arch.VECTOR)
    outready = tla.flag("pair_outready", tla.arch.VECTOR, tla.arch.MTE3)
    sch00 = tla.cross_flag("pair_sch00", mode=4)
    sch01 = tla.cross_flag("pair_sch01", mode=4)
    sch10 = tla.cross_flag("pair_sch10", mode=4)
    if tla.const_expr(indexed and not reuse_score_buffers):
        sch11 = sch01
    else:
        sch11 = tla.cross_flag("pair_sch11", mode=4)
    pch0 = tla.cross_flag("pair_pch0", mode=4)
    pch1 = tla.cross_flag("pair_pch1", mode=4)
    pch2 = tla.cross_flag("pair_pch2", mode=4)
    och0 = tla.cross_flag("pair_och0", mode=4)
    och1 = tla.cross_flag("pair_och1", mode=4)
    tail_zero = tla.flag("pair_tail_zero", tla.arch.VECTOR, tla.arch.MTE2)
    tail_loaded = tla.flag("pair_tail_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    tail_stored = tla.flag("pair_tail_stored", tla.arch.VECTOR, tla.arch.MTE3)
    tail_free = tla.cross_flag("pair_tail_free", mode=4)
    tail_ready = tla.cross_flag("pair_tail_ready", mode=4)
    with tla.cube():
        tla.set_flag(qfree)
        tla.set_flag(kfree0)
        tla.set_flag(kfree1)
        tla.set_flag(vfree0)
        tla.set_flag(vfree1)
        tla.set_flag(abfree0)
        tla.set_flag(abfree1)
        tla.set_flag(sfree0)
        tla.set_flag(sfree1)
        tla.set_flag(ofree0)
        tla.set_flag(ofree1)
        _cset(pch0, tla.arch.MTE1)
        _cset(pch1, tla.arch.MTE1)
        _cset(pch2, tla.arch.MTE1)
    with tla.vector():
        tla.set_flag(pfree0)
        tla.set_flag(pfree1)
        tla.set_flag(outfree)
        _vset(sch00, tla.arch.VECTOR)
        _vset(sch01, tla.arch.VECTOR)
        _vset(sch10, tla.arch.VECTOR)
        if tla.const_expr(not indexed or reuse_score_buffers):
            _vset(sch11, tla.arch.VECTOR)
        _vset(och0, tla.arch.VECTOR)
        _vset(och1, tla.arch.VECTOR)
    mm_serial = tla.as_numeric(0)
    tick = tla.as_numeric(0)
    q_serial = tla.as_numeric(0)
    pv_q_serial = tla.as_numeric(0)
    pv_iter = tla.as_numeric(0)
    stage0_kind = tla.as_numeric(0)
    stage1_kind = tla.as_numeric(0)
    stage0_kb = tla.as_numeric(0)
    stage1_kb = tla.as_numeric(0)
    stage0_kb_second = tla.as_numeric(0)
    stage1_kb_second = tla.as_numeric(0)
    stage0_second = tla.as_numeric(0)
    stage1_second = tla.as_numeric(0)
    stage0_last = tla.as_numeric(0)
    stage1_last = tla.as_numeric(0)
    kblocks = (max_k + 127) // 128
    info = ()
    if tla.const_expr(ragged):
        info = (max_q, max_k, tla.Int32(0), tla.Int32(0))
    for task in tla.range(tla.arch.block_idx(), batch * q_heads * q_blocks, tla.arch.block_num()):
        if tla.const_expr(dense and not (dense_mask and indexed)):
            qb = task % q_blocks
            head = task // q_blocks % q_heads
        elif tla.const_expr(dense_mask and indexed):
            qb = task // head_group % q_blocks
            head = (
                task // (head_group * q_blocks) % (q_heads // head_group)
            ) * head_group + task % head_group
        else:
            # Adjacent query tiles reuse the same head's K/V working set.
            qb = task % q_blocks
            head = task // q_blocks % q_heads
        b = task // (q_blocks * q_heads)
        kh = head // (q_heads // kv_heads)
        qstart = qb * 128
        valid_q = tla.as_numeric(128)
        if tla.const_expr(ragged):
            valid_q = min(128, max_q - qstart)
        qbase = b * q_heads * max_q * 128 + head * q_head_stride + qstart * q_stride
        kbase = b * kv_heads * max_k * 128 + kh * k_head_stride
        tile_count = kblocks
        pc = tla.as_numeric(0)
        if tla.const_expr(not dense):
            mask_row = ((b % mask_batch) * mask_heads + head % mask_heads) * q_blocks + qb
            pc = partial_count[mask_row]
            tile_count = pc + full_count[mask_row]
        if tla.const_expr(len(implicit_window) > 0):
            window_start, tile_count, window_full_lo, window_full_hi = window_block_bounds(
                qstart, valid_q, max_q, max_k, implicit_window
            )
        count = (tile_count + 1) // 2
        um = um0
        ul = ul0
        if q_serial % 3 == 1:
            um = um1
            ul = ul1
        elif q_serial % 3 == 2:
            um = um2
            ul = ul2
        if count > 0:
            pipeline_load_rows(
                q,
                qbase,
                q_stride,
                valid_q,
                lq,
                True,
                0,
                qfree,
                qfree,
                qload,
                qload,
                ragged,
                uout,
                tail_zero,
                tail_loaded,
                tail_stored,
                tail_free,
                tail_ready,
                True,
                (outfree,),
            )
        steps = count
        if count == 0:
            steps = tla.as_numeric(1)
        drain = tla.as_numeric(0)
        if task + tla.arch.block_num() >= batch * q_heads * q_blocks:
            drain = tla.as_numeric(2)
        for iteration in tla.range(steps + drain):
            stage2_kind = stage1_kind
            stage1_kind = stage0_kind
            stage2_kb = stage1_kb
            stage1_kb = stage0_kb
            stage2_kb_second = stage1_kb_second
            stage1_kb_second = stage0_kb_second
            stage2_second = stage1_second
            stage1_second = stage0_second
            stage2_last = stage1_last
            stage1_last = stage0_last
            stage0_kind = tla.as_numeric(0)
            stage0_kb = tla.as_numeric(0)
            stage0_kb_second = tla.as_numeric(0)
            stage0_second = tla.as_numeric(0)
            stage0_last = tla.as_numeric(0)
            if iteration < steps:
                if iteration == steps - 1:
                    stage0_last = tla.as_numeric(1)
                if count == 0:
                    stage0_kind = tla.as_numeric(2)
            if iteration < count:
                stage0_kind = tla.as_numeric(1)
                first = iteration * 2
                has_second = first + 1 < tile_count
                kb = first
                kb_second = first + 1
                full0 = tla.Bool(True)
                full1 = tla.Bool(True)
                if tla.const_expr(dense_mask):
                    full0 = tla.Bool(False)
                    full1 = tla.Bool(False)
                if tla.const_expr(not dense):
                    full0 = first >= pc
                    full1 = first + 1 >= pc
                    if full0:
                        kb = full_index[mask_row, first - pc]
                    else:
                        kb = partial_index[mask_row, first]
                    if has_second:
                        if full1:
                            kb_second = full_index[mask_row, first + 1 - pc]
                        else:
                            kb_second = partial_index[mask_row, first + 1]
                if tla.const_expr(len(implicit_window) > 0):
                    kb = window_start + first
                    kb_second = kb + 1
                    full0 = (kb * 128 >= window_full_lo) & (
                        min((kb + 1) * 128, max_k) <= window_full_hi
                    )
                    full1 = (kb_second * 128 >= window_full_lo) & (
                        min((kb_second + 1) * 128, max_k) <= window_full_hi
                    )
                stage0_kb = kb
                stage0_kb_second = kb_second
                tail0 = (valid_q < 128) | (kb * 128 + 128 > max_k)
                tail1 = (valid_q < 128) | (kb_second * 128 + 128 > max_k)
                if has_second:
                    stage0_second = tla.as_numeric(1)
                slot = tick % 2
                pslot = tick % 3
                lpptr = lp0
                alphaptr = alpha0
                if pslot == 1:
                    lpptr = lp1
                    alphaptr = alpha1
                elif pslot == 2:
                    lpptr = lp2
                    alphaptr = alpha2
                usptr0 = us00 if slot == 0 else us10
                usptr1 = us01 if slot == 0 else us11
                for lane in tla.range_constexpr(2):
                    if lane == 0 or has_second:
                        lkptr = lk0 if tla.const_expr(lane == 0) else lk1
                        lsptr = ls0 if tla.const_expr(lane == 0) else ls1
                        usptr = usptr0 if tla.const_expr(lane == 0) else usptr1
                        abslot = mm_serial % 2
                        laptr = la0 if abslot == 0 else la1
                        lbptr = lb0 if abslot == 0 else lb1
                        load_kb = kb if tla.const_expr(lane == 0) else kb_second
                        valid_k = tla.as_numeric(128)
                        if tla.const_expr(ragged):
                            valid_k = min(128, max_k - load_kb * 128)
                        pipeline_load_rows(
                            k,
                            kbase + load_kb * 128 * k_stride,
                            k_stride,
                            valid_k,
                            lkptr,
                            False,
                            lane,
                            kfree0,
                            kfree1,
                            kload0,
                            kload1,
                            ragged,
                            uout,
                            tail_zero,
                            tail_loaded,
                            tail_stored,
                            tail_free,
                            tail_ready,
                            True,
                            (outfree,),
                        )
                        with tla.cube():
                            product = _mm(
                                _matrix(lkptr, False),
                                _matrix(lq, True),
                                laptr,
                                lbptr,
                                lsptr,
                                abslot,
                                lane,
                                abload0,
                                abload1,
                                abfree0,
                                abfree1,
                                sfree0,
                                sfree1,
                                sdone0,
                                sdone1,
                            )
                            if tla.const_expr(lane == 0):
                                tla.set_flag(kfree0)
                            else:
                                tla.set_flag(kfree1)
                            if iteration == count - 1 and (lane == 1 or not has_second):
                                tla.set_flag(qfree)
                            if slot == 0:
                                _fix(
                                    product,
                                    usptr,
                                    lane,
                                    sdone0,
                                    sdone1,
                                    sfree0,
                                    sfree1,
                                    sch00,
                                    sch01,
                                    True,
                                )
                            else:
                                _fix(
                                    product,
                                    usptr,
                                    lane,
                                    sdone0,
                                    sdone1,
                                    sfree0,
                                    sfree1,
                                    sch10,
                                    sch11,
                                    True,
                                )
                        mm_serial = mm_serial + 1
                with tla.vector():
                    vi = tla.arch.sub_block_idx()
                    # Acquire free probability slots before waiting for QK, so
                    # independent SIMT mask work overlaps Cube/FIX. Borrow them for
                    # packed SIMT predicates; both score/max passes consume the
                    # masks before the exp/sum pass overwrites either slot.
                    visibility0 = tla.make_tensor(
                        tla.recast_ptr(up0, dtype=tla.Int32),
                        tla.make_layout(tla.make_shape(128, 8), tla.make_stride(8, 1)),
                    )
                    visibility1_ptr = tla.recast_ptr(up1, dtype=tla.Int32)
                    if tla.const_expr(reuse_score_buffers):
                        visibility1_ptr = visibility1_ptr + 1024
                    visibility1 = tla.make_tensor(
                        visibility1_ptr,
                        tla.make_layout(tla.make_shape(128, 8), tla.make_stride(8, 1)),
                    )
                    if tla.const_expr(indexed):
                        tla.wait_flag(pfree0)
                        if tla.const_expr(not reuse_score_buffers):
                            tla.wait_flag(pfree1)
                        if not full0:
                            if has_second and not full1:
                                if tla.const_expr(ragged):
                                    if tail0 | tail1:
                                        _cache_mask_pair(
                                            mask_mod,
                                            b,
                                            head,
                                            qstart,
                                            vi,
                                            kb * 128,
                                            kb_second * 128,
                                            visibility0,
                                            visibility1,
                                            aux_tensors,
                                            aux_scalars,
                                            info,
                                            False,
                                            temporal_aux,
                                        )
                                    else:
                                        _cache_mask_pair(
                                            mask_mod,
                                            b,
                                            head,
                                            qstart,
                                            vi,
                                            kb * 128,
                                            kb_second * 128,
                                            visibility0,
                                            visibility1,
                                            aux_tensors,
                                            aux_scalars,
                                            info,
                                            True,
                                            temporal_aux,
                                        )
                                else:
                                    _cache_mask_pair(
                                        mask_mod,
                                        b,
                                        head,
                                        qstart,
                                        vi,
                                        kb * 128,
                                        kb_second * 128,
                                        visibility0,
                                        visibility1,
                                        aux_tensors,
                                        aux_scalars,
                                        info,
                                        True,
                                        temporal_aux,
                                    )
                            else:
                                if tla.const_expr(ragged):
                                    if tail0:
                                        _cache_mask(
                                            mask_mod,
                                            b,
                                            head,
                                            qstart,
                                            vi,
                                            kb * 128,
                                            visibility0,
                                            aux_tensors,
                                            aux_scalars,
                                            info,
                                            False,
                                        )
                                    else:
                                        _cache_mask(
                                            mask_mod,
                                            b,
                                            head,
                                            qstart,
                                            vi,
                                            kb * 128,
                                            visibility0,
                                            aux_tensors,
                                            aux_scalars,
                                            info,
                                            True,
                                        )
                                else:
                                    _cache_mask(
                                        mask_mod,
                                        b,
                                        head,
                                        qstart,
                                        vi,
                                        kb * 128,
                                        visibility0,
                                        aux_tensors,
                                        aux_scalars,
                                        info,
                                        True,
                                    )
                            tla.pipe_barrier(tla.pipes.ALL)
                        elif has_second and not full1:
                            if tla.const_expr(ragged):
                                if tail1:
                                    _cache_mask(
                                        mask_mod,
                                        b,
                                        head,
                                        qstart,
                                        vi,
                                        kb_second * 128,
                                        visibility1,
                                        aux_tensors,
                                        aux_scalars,
                                        info,
                                        False,
                                    )
                                else:
                                    _cache_mask(
                                        mask_mod,
                                        b,
                                        head,
                                        qstart,
                                        vi,
                                        kb_second * 128,
                                        visibility1,
                                        aux_tensors,
                                        aux_scalars,
                                        info,
                                        True,
                                    )
                            else:
                                _cache_mask(
                                    mask_mod,
                                    b,
                                    head,
                                    qstart,
                                    vi,
                                    kb_second * 128,
                                    visibility1,
                                    aux_tensors,
                                    aux_scalars,
                                    info,
                                    True,
                                )
                            tla.pipe_barrier(tla.pipes.ALL)
                    empty0 = tla.Bool(False)
                    empty1 = tla.Bool(False)
                    mask_summary = visibility0
                    if tla.const_expr(dense_mask and indexed):
                        mask_summary = tla.make_tensor(
                            mask_summary_ptr,
                            tla.make_layout(tla.make_shape(64), tla.make_stride(1)),
                        )
                    if slot == 0:
                        _vwait(sch00, tla.arch.VECTOR)
                        if has_second:
                            _vwait(sch01, tla.arch.VECTOR)
                    else:
                        _vwait(sch10, tla.arch.VECTOR)
                        if has_second:
                            _vwait(sch11, tla.arch.VECTOR)
                    if tla.const_expr(not indexed):
                        tla.wait_flag(pfree0)
                        tla.wait_flag(pfree1)
                    scores0 = tla.make_tensor(
                        usptr0, tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1))
                    )
                    scores1 = tla.make_tensor(
                        usptr1, tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1))
                    )
                    pflat0 = tla.make_tensor(
                        up0, tla.make_layout(tla.make_shape(128, 64), tla.make_stride(16, 1))
                    )
                    pflat1 = tla.make_tensor(
                        up1, tla.make_layout(tla.make_shape(128, 64), tla.make_stride(16, 1))
                    )
                    mstate = tla.make_tensor(
                        um, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                    )
                    lstate = tla.make_tensor(
                        ul, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                    )
                    astate = tla.make_tensor(
                        alphaptr, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                    )
                    with tla.vec.func(mode="simd"):
                        _pair_softmax(
                            scores0,
                            scores1,
                            pflat0,
                            pflat1,
                            mstate,
                            lstate,
                            astate,
                            has_second,
                            iteration,
                            scale,
                            b,
                            head,
                            qstart,
                            vi,
                            kb * 128,
                            aux_tensors,
                            aux_scalars,
                            score_mod,
                            mask_mod,
                            dtype,
                            kb_second * 128,
                            full0,
                            full1,
                            not dense or len(implicit_window) > 0 or dense_mask,
                            indexed,
                            visibility0,
                            visibility1,
                            empty0,
                            empty1,
                            mask_summary,
                            dense_mask and indexed,
                            dense_mask and indexed,
                            info,
                            tail0,
                            tail1,
                        )
                    if tla.const_expr(reuse_score_buffers):
                        if tla.const_expr(dense_mask and indexed):
                            tla.pipe_barrier(tla.pipes.ALL)
                            empty0 = mask_summary[0] == 0
                            if has_second:
                                empty1 = mask_summary[1] == 0
                        # Sparse full tiles do not initialize mask predicates;
                        # keep their empty classification disabled.
                        # P1 has consumed the second score completely. Borrow
                        # that score for P0 while P1 DMA reads the P allocation.
                        # Return the score to FIX only after P0 DMA completes.
                        reuse_probability = has_second & empty0 & empty1
                        first_probability_ptr = up0
                        if has_second:
                            if reuse_probability:
                                if slot == 0:
                                    _vset(sch01, tla.arch.VECTOR)
                                else:
                                    _vset(sch11, tla.arch.VECTOR)
                            else:
                                first_probability_ptr = tla.recast_ptr(usptr1, dtype=dtype)
                        if pslot == 0:
                            _vwait(pch0, tla.arch.MTE3)
                        elif pslot == 1:
                            _vwait(pch1, tla.arch.MTE3)
                        else:
                            _vwait(pch2, tla.arch.MTE3)
                        single_pref = tla.make_tensor(
                            up0, tla.make_layout(tla.make_shape(129, 64), tla.make_stride(64, 1))
                        )
                        single_ptrans = tla.make_tensor_like(up0, single_pref, tla.arch.zNUnAlign)
                        single_psrc = tla.tile_view(
                            single_ptrans, tla.make_shape(128, 64), tla.make_coord(0, 0)
                        )
                        if has_second:
                            tla.set_flag(pready0)
                            tla.wait_flag(pready0)
                            second_dst = tla.tile_view(
                                _matrix(lpptr + 16384, False),
                                tla.make_shape(128, 64),
                                tla.make_coord(0, vi),
                            )
                            tla.copy(second_dst, single_psrc)
                            if not reuse_probability:
                                tla.set_flag(pfree0)
                        if not reuse_probability:
                            first_pflat = tla.make_tensor(
                                first_probability_ptr,
                                tla.make_layout(tla.make_shape(128, 64), tla.make_stride(16, 1)),
                            )
                            with tla.vec.func(mode="simd"):
                                _pair_first_prob(
                                    scores0, first_pflat, mstate, lstate, dtype, empty0
                                )
                        if slot == 0:
                            _vset(sch00, tla.arch.VECTOR)
                        else:
                            _vset(sch10, tla.arch.VECTOR)
                        if not reuse_probability:
                            tla.set_flag(pready0)
                            tla.wait_flag(pready0)
                        first_pref = tla.make_tensor(
                            first_probability_ptr,
                            tla.make_layout(tla.make_shape(129, 64), tla.make_stride(64, 1)),
                        )
                        first_ptrans = tla.make_tensor_like(
                            first_probability_ptr, first_pref, tla.arch.zNUnAlign
                        )
                        first_psrc = tla.tile_view(
                            first_ptrans, tla.make_shape(128, 64), tla.make_coord(0, 0)
                        )
                        first_dst = tla.tile_view(
                            _matrix(lpptr, False), tla.make_shape(128, 64), tla.make_coord(0, vi)
                        )
                        tla.copy(first_dst, first_psrc)
                        if has_second and not reuse_probability:
                            if slot == 0:
                                _vset(sch01, tla.arch.MTE3)
                            else:
                                _vset(sch11, tla.arch.MTE3)
                        else:
                            tla.set_flag(pfree0)
                        if pslot == 0:
                            _vset(pch0, tla.arch.MTE3)
                        elif pslot == 1:
                            _vset(pch1, tla.arch.MTE3)
                        else:
                            _vset(pch2, tla.arch.MTE3)
                    else:
                        if tla.const_expr(indexed):
                            if has_second:
                                _vset(sch01, tla.arch.VECTOR)
                            with tla.vec.func(mode="simd"):
                                _pair_first_prob(scores0, pflat0, mstate, lstate, dtype, empty0)
                        if slot == 0:
                            _vset(sch00, tla.arch.VECTOR)
                            if tla.const_expr(not indexed):
                                if has_second:
                                    _vset(sch01, tla.arch.VECTOR)
                        else:
                            _vset(sch10, tla.arch.VECTOR)
                            if tla.const_expr(not indexed):
                                if has_second:
                                    _vset(sch11, tla.arch.VECTOR)
                        tla.set_flag(pready0)
                        tla.wait_flag(pready0)
                        tla.set_flag(pready1)
                        tla.wait_flag(pready1)
                        if pslot == 0:
                            _vwait(pch0, tla.arch.MTE3)
                        elif pslot == 1:
                            _vwait(pch1, tla.arch.MTE3)
                        else:
                            _vwait(pch2, tla.arch.MTE3)
                        for lane in tla.range_constexpr(2):
                            if lane == 0 or has_second:
                                upptr = up0 if tla.const_expr(lane == 0) else up1
                                pref = tla.make_tensor(
                                    upptr,
                                    tla.make_layout(
                                        tla.make_shape(129, 64), tla.make_stride(64, 1)
                                    ),
                                )
                                ptrans = tla.make_tensor_like(upptr, pref, tla.arch.zNUnAlign)
                                pdst = tla.tile_view(
                                    _matrix(lpptr + lane * 16384, False),
                                    tla.make_shape(128, 64),
                                    tla.make_coord(0, vi),
                                )
                                tla.copy(
                                    pdst,
                                    tla.tile_view(
                                        ptrans, tla.make_shape(128, 64), tla.make_coord(0, 0)
                                    ),
                                )
                        tla.set_flag(pfree0)
                        tla.set_flag(pfree1)
                        if pslot == 0:
                            _vset(pch0, tla.arch.MTE3)
                        elif pslot == 1:
                            _vset(pch1, tla.arch.MTE3)
                        else:
                            _vset(pch2, tla.arch.MTE3)
            if stage2_kind != 0:
                pv_task = tla.arch.block_idx() + pv_q_serial * tla.arch.block_num()
                if tla.const_expr(dense and not (dense_mask and indexed)):
                    pv_qb = pv_task % q_blocks
                    pv_head = pv_task // q_blocks % q_heads
                elif tla.const_expr(dense_mask and indexed):
                    pv_qb = pv_task // head_group % q_blocks
                    pv_head = (
                        pv_task // (head_group * q_blocks) % (q_heads // head_group)
                    ) * head_group + pv_task % head_group
                else:
                    pv_qb = pv_task % q_blocks
                    pv_head = pv_task // q_blocks % q_heads
                pv_b = pv_task // (q_blocks * q_heads)
                pv_kh = pv_head // (q_heads // kv_heads)
                pv_qstart = pv_qb * 128
                pv_qbase = (
                    pv_b * q_heads * max_q * 128 + pv_head * q_head_stride + pv_qstart * q_stride
                )
                pv_kbase = pv_b * kv_heads * max_k * 128 + pv_kh * k_head_stride
                pv_lbase = pv_b * l_batch_stride + pv_head * l_head_stride + pv_qstart
                pv_um = um0
                pv_ul = ul0
                if pv_q_serial % 3 == 1:
                    pv_um = um1
                    pv_ul = ul1
                elif pv_q_serial % 3 == 2:
                    pv_um = um2
                    pv_ul = ul2
                if stage2_kind == 1:
                    pvslot = (tick - 2) % 2
                    pvpslot = (tick - 2) % 3
                    vblock = stage2_kb
                    pv_second = stage2_second != 0
                    loptr = lo0 if pvslot == 0 else lo1
                    uoptr = uo0 if pvslot == 0 else uo1
                    prevptr = uo1 if pvslot == 0 else uo0
                    pvlpptr = lp0
                    pvalphaptr = alpha0
                    if pvpslot == 1:
                        pvlpptr = lp1
                        pvalphaptr = alpha1
                    elif pvpslot == 2:
                        pvlpptr = lp2
                        pvalphaptr = alpha2
                    with tla.cube():
                        if pvpslot == 0:
                            _cwait(pch0, tla.arch.MTE1)
                        elif pvpslot == 1:
                            _cwait(pch1, tla.arch.MTE1)
                        else:
                            _cwait(pch2, tla.arch.MTE1)
                    for pv_lane in tla.range_constexpr(2):
                        if pv_lane == 0 or pv_second:
                            lvptr = lv0 if tla.const_expr(pv_lane == 0) else lv1
                            pvabslot = mm_serial % 2
                            pvlaptr = la0 if pvabslot == 0 else la1
                            pvlbptr = lb0 if pvabslot == 0 else lb1
                            load_vb = vblock if tla.const_expr(pv_lane == 0) else stage2_kb_second
                            valid_v = tla.as_numeric(128)
                            if tla.const_expr(ragged):
                                valid_v = min(128, max_k - load_vb * 128)
                            pipeline_load_rows(
                                v,
                                pv_kbase + load_vb * 128 * k_stride,
                                k_stride,
                                valid_v,
                                lvptr,
                                False,
                                pv_lane,
                                vfree0,
                                vfree1,
                                vload0,
                                vload1,
                                ragged,
                                uout,
                                tail_zero,
                                tail_loaded,
                                tail_stored,
                                tail_free,
                                tail_ready,
                                True,
                                (outfree,),
                            )
                            with tla.cube():
                                _mm(
                                    _matrix(pvlpptr + pv_lane * 16384, True),
                                    _matrix(lvptr, False),
                                    pvlaptr,
                                    pvlbptr,
                                    loptr,
                                    pvabslot,
                                    pvslot,
                                    abload0,
                                    abload1,
                                    abfree0,
                                    abfree1,
                                    ofree0,
                                    ofree1,
                                    odone0,
                                    odone1,
                                    pv_lane == 0,
                                    pv_lane == 1 or not pv_second,
                                )
                                if tla.const_expr(pv_lane == 0):
                                    tla.set_flag(vfree0)
                                else:
                                    tla.set_flag(vfree1)
                            mm_serial = mm_serial + 1
                    with tla.cube():
                        if pvpslot == 0:
                            _cset(pch0, tla.arch.MTE1)
                        elif pvpslot == 1:
                            _cset(pch1, tla.arch.MTE1)
                        else:
                            _cset(pch2, tla.arch.MTE1)
                        pv_ref = tla.make_tensor(
                            loptr,
                            tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
                        )
                        pv_lc = tla.make_tensor_like(loptr, pv_ref, tla.arch.L0Clayout)
                        _fix(
                            pv_lc, uoptr, pvslot, odone0, odone1, ofree0, ofree1, och0, och1, False
                        )
                    with tla.vector():
                        if pvslot == 0:
                            _vwait(och0, tla.arch.VECTOR)
                        else:
                            _vwait(och1, tla.arch.VECTOR)
                        pvub = tla.make_tensor(
                            uoptr, tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1))
                        )
                        previous = tla.make_tensor(
                            prevptr,
                            tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
                        )
                        pv_output_ptr = uout
                        if tla.const_expr(reuse_score_buffers):
                            # Ascending row/column conversion compacts FP32 into
                            # the consumed prefix of this final PV allocation.
                            pv_output_ptr = tla.recast_ptr(uoptr, dtype=dtype)
                        out_pv = tla.make_tensor(
                            pv_output_ptr,
                            tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
                        )
                        a = tla.make_tensor(
                            pvalphaptr, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                        )
                        m = tla.make_tensor(
                            pv_um, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                        )
                        l = tla.make_tensor(
                            pv_ul, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                        )
                        ls = tla.make_tensor(
                            ulse_out, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                        )
                        if stage2_last != 0:
                            tla.wait_flag(outfree)
                        with tla.vec.func(mode="simd"):
                            if pv_iter != 0:
                                if stage2_last != 0:
                                    _accumulate_pv(
                                        previous, pvub, pvub, a, out_pv, m, l, ls, dtype, True
                                    )
                                else:
                                    _accumulate_pv(
                                        previous, pvub, pvub, a, out_pv, m, l, ls, dtype, False
                                    )
                            elif stage2_last != 0:
                                _normalize_pair_pv(pvub, out_pv, m, l, ls, dtype)
                        # The previous buffer is no longer an accumulator source.
                        # Keep the current one owned until the next accumulation.
                        if pv_iter != 0:
                            if pvslot == 0:
                                _vset(och1, tla.arch.VECTOR)
                            else:
                                _vset(och0, tla.arch.VECTOR)
                        if stage2_last != 0:
                            if tla.const_expr(not reuse_score_buffers):
                                if pvslot == 0:
                                    _vset(och0, tla.arch.VECTOR)
                                else:
                                    _vset(och1, tla.arch.VECTOR)
                else:
                    with tla.vector():
                        tla.wait_flag(outfree)
                        if tla.const_expr(reuse_score_buffers):
                            # EMPTY output borrows the probability allocation.
                            tla.wait_flag(pfree0)
                        out_empty = tla.make_tensor(
                            uout, tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1))
                        )
                        lse_empty = tla.make_tensor(
                            ulse_out, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                        )
                        with tla.vec.func(mode="simd"):
                            for row in tla.range(64):
                                tla.tile_view(
                                    out_empty, tla.make_shape(1, 128), tla.make_coord(row, 0)
                                ).store(tla.full(0.0, dtype=dtype))
                            lse_empty.store(tla.full(float("-inf"), dtype=tla.Float32))
                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                            )
                if stage2_last != 0:
                    with tla.vector():
                        vo = tla.arch.sub_block_idx()
                        final_output_ptr = uout
                        if tla.const_expr(reuse_score_buffers):
                            if stage2_kind == 1:
                                final_pv_ptr = uo0 if (tick - 2) % 2 == 0 else uo1
                                final_output_ptr = tla.recast_ptr(final_pv_ptr, dtype=dtype)
                        out_final = tla.make_tensor(
                            final_output_ptr,
                            tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
                        )
                        lse_final = tla.make_tensor(
                            ulse_out, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                        )
                        tla.set_flag(outready)
                        tla.wait_flag(outready)
                        store_rows = tla.as_numeric(64)
                        if tla.const_expr(ragged):
                            store_rows = min(64, max(0, max_q - pv_qstart - vo * 64))
                        if store_rows > 0:
                            ogm = tla.make_tensor(
                                out.ptr + pv_qbase + vo * 64 * q_stride,
                                tla.make_layout(
                                    tla.make_shape(store_rows, 128), tla.make_stride(q_stride, 1)
                                ),
                            )
                            lgm = tla.make_tensor(
                                lse.ptr + pv_lbase + vo * 64,
                                tla.make_layout(tla.make_shape(store_rows), tla.make_stride(1)),
                            )
                            tla.copy(
                                ogm,
                                tla.tile_view(
                                    out_final, tla.make_shape(store_rows, 128), tla.make_coord(0, 0)
                                ),
                            )
                            tla.copy(
                                lgm,
                                tla.tile_view(
                                    lse_final, tla.make_shape(store_rows), tla.make_coord(0)
                                ),
                            )
                        tla.set_flag(outfree)
                        if tla.const_expr(reuse_score_buffers):
                            if stage2_kind == 1:
                                # FIX may reclaim the final PV only after both
                                # output DMAs finish reading its compact prefix.
                                if (tick - 2) % 2 == 0:
                                    _vset(och0, tla.arch.MTE3)
                                else:
                                    _vset(och1, tla.arch.MTE3)
                            else:
                                tla.set_flag(pfree0)
                    pv_q_serial = pv_q_serial + 1
                    pv_iter = tla.as_numeric(0)
                else:
                    pv_iter = pv_iter + 1
            tick = tick + 1
        q_serial = q_serial + 1
    with tla.cube():
        tla.wait_flag(qfree)
        tla.wait_flag(kfree0)
        tla.wait_flag(kfree1)
        tla.wait_flag(vfree0)
        tla.wait_flag(vfree1)
        tla.wait_flag(abfree0)
        tla.wait_flag(abfree1)
        tla.wait_flag(sfree0)
        tla.wait_flag(sfree1)
        tla.wait_flag(ofree0)
        tla.wait_flag(ofree1)
        _cwait(sch00, tla.arch.FIX)
        _cwait(sch01, tla.arch.FIX)
        _cwait(sch10, tla.arch.FIX)
        if tla.const_expr(not indexed or reuse_score_buffers):
            _cwait(sch11, tla.arch.FIX)
        _cwait(och0, tla.arch.FIX)
        _cwait(och1, tla.arch.FIX)
    with tla.vector():
        tla.wait_flag(pfree0)
        tla.wait_flag(pfree1)
        tla.wait_flag(outfree)
        _vwait(pch0, tla.arch.MTE3)
        _vwait(pch1, tla.arch.MTE3)
        _vwait(pch2, tla.arch.MTE3)


@tla.kernel
def attention_kernel(
    q: tla.Tensor,
    k: tla.Tensor,
    v: tla.Tensor,
    out: tla.Tensor,
    lse: tla.Tensor,
    dout: tla.Tensor,
    dq: tla.Tensor,
    dk_acc: tla.Tensor,
    dv_acc: tla.Tensor,
    partial_count: tla.Tensor,
    partial_index: tla.Tensor,
    full_count: tla.Tensor,
    full_index: tla.Tensor,
    cu_q: tla.Tensor,
    cu_k: tla.Tensor,
    aux_tensors: tuple,
    aux_scalars: tuple,
    batch: tla.Int32,
    q_heads: tla.Int32,
    kv_heads: tla.Int32,
    max_q: tla.Int32,
    max_k: tla.Int32,
    q_blocks: tla.Int32,
    mask_batch: tla.Int32,
    mask_heads: tla.Int32,
    scale: tla.Float32,
    total_q: tla.Int32,
    varlen: tla.Constexpr,
    dense: tla.Constexpr,
    backward: tla.Constexpr,
    indexed: tla.Constexpr,
    score_mod: tla.Constexpr,
    mask_mod: tla.Constexpr,
    score_grad: tla.Constexpr,
    sequence_major: tla.Constexpr,
    ragged: tla.Constexpr = False,
    implicit_window: tla.Constexpr = (),
    generic: tla.Constexpr = False,
    dense_mask: tla.Constexpr = False,
    vector_score_mod: tla.Constexpr = None,
    temporal_aux: tla.Constexpr = False,
):
    if tla.const_expr(
        dense
        and generic
        and indexed
        and vector_score_mod is not None
        and not ragged
        and not any(varlen)
    ):
        _paired_pipeline(
            q,
            k,
            v,
            out,
            lse,
            aux_tensors,
            aux_scalars,
            partial_count,
            partial_index,
            full_count,
            full_index,
            mask_batch,
            mask_heads,
            batch,
            q_heads,
            kv_heads,
            max_q,
            max_k,
            q_blocks,
            scale,
            total_q,
            varlen,
            vector_score_mod,
            mask_mod,
            dense,
            implicit_window,
            sequence_major,
            indexed,
            True,
            False,
            temporal_aux,
        )
    elif tla.const_expr(
        not generic and not any(varlen) and (not dense_mask or len(implicit_window) > 0)
    ):
        _paired_pipeline(
            q,
            k,
            v,
            out,
            lse,
            aux_tensors,
            aux_scalars,
            partial_count,
            partial_index,
            full_count,
            full_index,
            mask_batch,
            mask_heads,
            batch,
            q_heads,
            kv_heads,
            max_q,
            max_k,
            q_blocks,
            scale,
            total_q,
            varlen,
            score_mod,
            mask_mod,
            dense,
            implicit_window,
            sequence_major,
            indexed,
            False,
            ragged,
            temporal_aux,
        )
    else:
        # BSND/TND use interleaved heads in GM; on-chip tiles stay unchanged.
        q_stride = q_heads * 128 if tla.const_expr(sequence_major[0]) else 128
        k_stride = kv_heads * 128 if tla.const_expr(sequence_major[1]) else 128
        q_head_stride = 128 if tla.const_expr(sequence_major[0]) else max_q * 128
        k_head_stride = 128 if tla.const_expr(sequence_major[1]) else max_k * 128
        l_stride = 1
        l_head_stride = total_q if tla.const_expr(varlen[0]) else max_q
        l_batch_stride = max_q if tla.const_expr(varlen[0]) else q_heads * max_q
        dtype = q.ptr.dtype
        lq = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
        lk0 = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
        lk1 = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
        lv0 = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
        lv1 = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
        lp0 = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
        lp1 = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
        lp2 = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
        la0 = tla.allocate(16384, dtype, tla.AddressSpace.l0a, 512)
        la1 = tla.allocate(16384, dtype, tla.AddressSpace.l0a, 512)
        lb0 = tla.allocate(16384, dtype, tla.AddressSpace.l0b, 512)
        lb1 = tla.allocate(16384, dtype, tla.AddressSpace.l0b, 512)
        ls0 = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0c, 512)
        ls1 = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0c, 512)
        lo0 = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0c, 512)
        lo1 = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0c, 512)
        us0 = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
        us1 = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
        up0 = tla.allocate(8256, dtype, tla.AddressSpace.ub, 256)
        up1 = tla.allocate(8256, dtype, tla.AddressSpace.ub, 256)
        uo0 = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
        uo1 = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
        ua = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
        # Aligned dense SIMT keeps its accumulator in the PV ping-pong slots,
        # so this allocation instead holds eight 4KiB packed predicate maps.
        uout = tla.allocate(8192, dtype, tla.AddressSpace.ub, 256)
        um0 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        um1 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        um2 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        ul0 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        ul1 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        ul2 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        ulse_out = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        alpha0 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        alpha1 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        alpha2 = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)

        tail_scratch = uout
        tail_zero = tla.flag("tail_zero", tla.arch.VECTOR, tla.arch.MTE2)
        tail_loaded = tla.flag("tail_loaded", tla.arch.MTE2, tla.arch.VECTOR)
        tail_stored = tla.flag("tail_stored", tla.arch.VECTOR, tla.arch.MTE3)
        tail_free = tla.cross_flag("tail_free", mode=4)
        tail_ready = tla.cross_flag("tail_ready", mode=4)

        qfree = tla.flag("qfree", tla.arch.MTE1, tla.arch.MTE2)
        qload = tla.flag("qload", tla.arch.MTE2, tla.arch.MTE1)
        kfree0 = tla.flag("kfree0", tla.arch.MTE1, tla.arch.MTE2)
        kfree1 = tla.flag("kfree1", tla.arch.MTE1, tla.arch.MTE2)
        kload0 = tla.flag("kload0", tla.arch.MTE2, tla.arch.MTE1)
        kload1 = tla.flag("kload1", tla.arch.MTE2, tla.arch.MTE1)
        vfree0 = tla.flag("vfree0", tla.arch.MTE1, tla.arch.MTE2)
        vfree1 = tla.flag("vfree1", tla.arch.MTE1, tla.arch.MTE2)
        vload0 = tla.flag("vload0", tla.arch.MTE2, tla.arch.MTE1)
        vload1 = tla.flag("vload1", tla.arch.MTE2, tla.arch.MTE1)
        abfree0 = tla.flag("abfree0", tla.arch.CUBE, tla.arch.MTE1)
        abfree1 = tla.flag("abfree1", tla.arch.CUBE, tla.arch.MTE1)
        abload0 = tla.flag("abload0", tla.arch.MTE1, tla.arch.CUBE)
        abload1 = tla.flag("abload1", tla.arch.MTE1, tla.arch.CUBE)
        sfree0 = tla.flag("sfree0", tla.arch.FIX, tla.arch.CUBE)
        sfree1 = tla.flag("sfree1", tla.arch.FIX, tla.arch.CUBE)
        sdone0 = tla.flag("sdone0", tla.arch.CUBE, tla.arch.FIX)
        sdone1 = tla.flag("sdone1", tla.arch.CUBE, tla.arch.FIX)
        ofree0 = tla.flag("ofree0", tla.arch.FIX, tla.arch.CUBE)
        ofree1 = tla.flag("ofree1", tla.arch.FIX, tla.arch.CUBE)
        odone0 = tla.flag("odone0", tla.arch.CUBE, tla.arch.FIX)
        odone1 = tla.flag("odone1", tla.arch.CUBE, tla.arch.FIX)
        pfree0 = tla.flag("pfree0", tla.arch.MTE3, tla.arch.VECTOR)
        pfree1 = tla.flag("pfree1", tla.arch.MTE3, tla.arch.VECTOR)
        pready0 = tla.flag("pready0", tla.arch.VECTOR, tla.arch.MTE3)
        pready1 = tla.flag("pready1", tla.arch.VECTOR, tla.arch.MTE3)
        outfree = tla.flag("outfree", tla.arch.MTE3, tla.arch.VECTOR)
        outready = tla.flag("outready", tla.arch.VECTOR, tla.arch.MTE3)
        sch0 = tla.cross_flag("sch0", mode=4)
        sch1 = tla.cross_flag("sch1", mode=4)
        pch0 = tla.cross_flag("pch0", mode=4)
        pch1 = tla.cross_flag("pch1", mode=4)
        pch2 = tla.cross_flag("pch2", mode=4)
        och0 = tla.cross_flag("och0", mode=4)
        och1 = tla.cross_flag("och1", mode=4)

        with tla.cube():
            tla.set_flag(qfree)
            tla.set_flag(kfree0)
            tla.set_flag(kfree1)
            tla.set_flag(vfree0)
            tla.set_flag(vfree1)
            tla.set_flag(abfree0)
            tla.set_flag(abfree1)
            tla.set_flag(sfree0)
            tla.set_flag(sfree1)
            tla.set_flag(ofree0)
            tla.set_flag(ofree1)
            _cset(pch0, tla.arch.MTE1)
            _cset(pch1, tla.arch.MTE1)
            _cset(pch2, tla.arch.MTE1)
        with tla.vector():
            tla.set_flag(pfree0)
            tla.set_flag(pfree1)
            tla.set_flag(outfree)
            _vset(sch0, tla.arch.VECTOR)
            _vset(sch1, tla.arch.VECTOR)
            _vset(och0, tla.arch.VECTOR)
            _vset(och1, tla.arch.VECTOR)

        mm_serial = tla.as_numeric(0)
        tick = tla.as_numeric(0)
        q_serial = tla.as_numeric(0)
        pv_q_serial = tla.as_numeric(0)
        pv_tile_index = tla.as_numeric(0)
        # kind: 0 INVALID, 1 real KV tile, 2 EMPTY query. The queue carries
        # only metadata that cannot be recovered from the ordered consumer.
        stage0_kind = tla.as_numeric(0)
        stage1_kind = tla.as_numeric(0)
        stage2_kind = tla.as_numeric(0)
        stage0_kb = tla.as_numeric(0)
        stage1_kb = tla.as_numeric(0)
        stage2_kb = tla.as_numeric(0)
        stage0_last = tla.as_numeric(0)
        stage1_last = tla.as_numeric(0)
        stage2_last = tla.as_numeric(0)
        for task in tla.range(
            tla.arch.block_idx(), batch * q_heads * q_blocks, tla.arch.block_num()
        ):
            if tla.const_expr(dense):
                qb = task % q_blocks
                head = task // q_blocks % q_heads
            else:
                # Match the producer and consumer order while reusing K/V.
                qb = task % q_blocks
                head = task // q_blocks % q_heads
            b = task // (q_blocks * q_heads)
            kh = head // (q_heads // kv_heads)
            qstart = qb * 128
            qbase = b * q_heads * max_q * 128 + head * q_head_stride + qstart * q_stride
            kbase = b * kv_heads * max_k * 128 + kh * k_head_stride
            qlen = max_q
            klen = max_k
            qs = tla.as_numeric(0)
            ks = tla.as_numeric(0)
            info = ()
            if tla.const_expr(ragged):
                if tla.const_expr(varlen[0]):
                    qs = cu_q[b]
                    qlen = cu_q[b + 1] - qs
                    qbase = qs * q_stride + head * 128 + qstart * q_stride
                if tla.const_expr(varlen[1]):
                    ks = cu_k[b]
                    klen = cu_k[b + 1] - ks
                    kbase = ks * k_stride + kh * 128
                info = (qlen, klen, qs, ks)
            if tla.const_expr(generic):
                if tla.const_expr(not ragged):
                    qs = b * max_q if tla.const_expr(varlen[0]) else tla.as_numeric(0)
                    ks = b * max_k if tla.const_expr(varlen[1]) else tla.as_numeric(0)
                info = (qlen, klen, qs, ks)
            valid_q = min(128, max(0, qlen - qstart))
            mask_row = ((b % mask_batch) * mask_heads + head % mask_heads) * q_blocks + qb
            pc = tla.as_numeric(0)
            fc = (klen + 127) // 128
            if tla.const_expr(not dense):
                pc = partial_count[mask_row]
                fc = full_count[mask_row]
            count = pc + fc
            if tla.const_expr(len(implicit_window) > 0):
                window_start, count, window_full_lo, window_full_hi = window_block_bounds(
                    qstart,
                    valid_q,
                    qlen,
                    klen,
                    implicit_window,
                )
            if valid_q == 0:
                count = tla.as_numeric(0)
                pc = tla.as_numeric(0)
            qslot = q_serial % 3
            um = um0
            ul = ul0
            if qslot == 1:
                um = um1
                ul = ul1
            elif qslot == 2:
                um = um2
                ul = ul2
            if count > 0:
                pipeline_load_rows(
                    q,
                    qbase,
                    q_stride,
                    valid_q,
                    lq,
                    True,
                    0,
                    qfree,
                    qfree,
                    qload,
                    qload,
                    ragged,
                    tail_scratch,
                    tail_zero,
                    tail_loaded,
                    tail_stored,
                    tail_free,
                    tail_ready,
                    True,
                    (outfree,),
                )
            with tla.vector():
                # No output wait or accumulator write here: older queries can
                # still have PV tiles in flight. Three query slots prevent aliasing.
                m_init = tla.make_tensor(
                    um, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                )
                l_init = tla.make_tensor(
                    ul, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                )
                if count == 0:
                    with tla.vec.func(mode="simd"):
                        m_init.store(tla.full(float("-inf"), dtype=tla.Float32))
                        l_init.store(tla.full(0.0, dtype=tla.Float32))
                        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)

            if tla.const_expr(indexed and not generic):
                with tla.vector():
                    if pc == 2:
                        # Two exact partial entries share a SIMT invocation. Wait
                        # for both probability DMAs before borrowing their slots.
                        tla.wait_flag(pfree0)
                        tla.wait_flag(pfree1)
                        cached_vi = tla.arch.sub_block_idx()
                        cached_k0 = partial_index[mask_row, 0] * 128
                        cached_k1 = partial_index[mask_row, 1] * 128
                        cached_ptr0 = up0 if tick % 2 == 0 else up1
                        cached_ptr1 = up1 if tick % 2 == 0 else up0
                        cached_mask0 = tla.make_tensor(
                            tla.recast_ptr(cached_ptr0, dtype=tla.Int32),
                            tla.make_layout(tla.make_shape(128, 8), tla.make_stride(8, 1)),
                        )
                        cached_mask1 = tla.make_tensor(
                            tla.recast_ptr(cached_ptr1, dtype=tla.Int32),
                            tla.make_layout(tla.make_shape(128, 8), tla.make_stride(8, 1)),
                        )
                        if tla.const_expr(ragged):
                            if valid_q < 128 or cached_k0 + 128 > klen or cached_k1 + 128 > klen:
                                _cache_mask_pair(
                                    mask_mod,
                                    b,
                                    head,
                                    qstart,
                                    cached_vi,
                                    cached_k0,
                                    cached_k1,
                                    cached_mask0,
                                    cached_mask1,
                                    aux_tensors,
                                    aux_scalars,
                                    info,
                                    False,
                                    temporal_aux,
                                )
                            else:
                                _cache_mask_pair(
                                    mask_mod,
                                    b,
                                    head,
                                    qstart,
                                    cached_vi,
                                    cached_k0,
                                    cached_k1,
                                    cached_mask0,
                                    cached_mask1,
                                    aux_tensors,
                                    aux_scalars,
                                    info,
                                    True,
                                    temporal_aux,
                                )
                        else:
                            _cache_mask_pair(
                                mask_mod,
                                b,
                                head,
                                qstart,
                                cached_vi,
                                cached_k0,
                                cached_k1,
                                cached_mask0,
                                cached_mask1,
                                aux_tensors,
                                aux_scalars,
                                info,
                                True,
                                temporal_aux,
                            )
                        tla.pipe_barrier(tla.pipes.ALL)
                        tla.set_flag(pfree0)
                        tla.set_flag(pfree1)

            steps = count
            if count == 0:
                steps = tla.as_numeric(1)
            drain = tla.as_numeric(0)
            if task + tla.arch.block_num() >= batch * q_heads * q_blocks:
                drain = tla.as_numeric(2)
            for iteration in tla.range(steps + drain):
                stage2_kind = stage1_kind
                stage1_kind = stage0_kind
                stage2_kb = stage1_kb
                stage1_kb = stage0_kb
                stage2_last = stage1_last
                stage1_last = stage0_last
                stage0_kind = tla.as_numeric(0)
                stage0_kb = tla.as_numeric(0)
                stage0_last = tla.as_numeric(0)
                if iteration < steps:
                    if iteration == steps - 1:
                        stage0_last = tla.as_numeric(1)
                    if count == 0:
                        stage0_kind = tla.as_numeric(2)
                if iteration < count:
                    stage0_kind = tla.as_numeric(1)
                    slot = tick % 2
                    pslot = tick % 3
                    full = iteration >= pc
                    if tla.const_expr(dense_mask and len(implicit_window) == 0):
                        full = tla.Bool(False)
                    kb = iteration
                    if tla.const_expr(len(implicit_window) > 0):
                        kb = window_start + iteration
                        full = (kb * 128 >= window_full_lo) & (
                            min(kb * 128 + 128, klen) <= window_full_hi
                        )
                    if tla.const_expr(not dense):
                        if full:
                            kb = full_index[mask_row, iteration - pc]
                        else:
                            kb = partial_index[mask_row, iteration]
                    stage0_kb = kb
                    kstart = kb * 128
                    lkptr = lk0 if slot == 0 else lk1
                    lsptr = ls0 if slot == 0 else ls1
                    usptr = us0 if slot == 0 else us1
                    upptr = up0 if slot == 0 else up1
                    lpptr = lp0
                    alphaptr = alpha0
                    if pslot == 1:
                        lpptr = lp1
                        alphaptr = alpha1
                    elif pslot == 2:
                        lpptr = lp2
                        alphaptr = alpha2
                    abslot = mm_serial % 2
                    laptr = la0 if abslot == 0 else la1
                    lbptr = lb0 if abslot == 0 else lb1
                    valid_k = min(128, klen - kstart)
                    tail_tile = (valid_q < 128) | (valid_k < 128)
                    pipeline_load_rows(
                        k,
                        kbase + kstart * k_stride,
                        k_stride,
                        valid_k,
                        lkptr,
                        False,
                        slot,
                        kfree0,
                        kfree1,
                        kload0,
                        kload1,
                        ragged,
                        tail_scratch,
                        tail_zero,
                        tail_loaded,
                        tail_stored,
                        tail_free,
                        tail_ready,
                        True,
                        (outfree,),
                    )
                    with tla.cube():
                        product = _mm(
                            _matrix(lkptr, False),
                            _matrix(lq, True),
                            laptr,
                            lbptr,
                            lsptr,
                            abslot,
                            slot,
                            abload0,
                            abload1,
                            abfree0,
                            abfree1,
                            sfree0,
                            sfree1,
                            sdone0,
                            sdone1,
                        )
                        if slot == 0:
                            tla.set_flag(kfree0)
                        else:
                            tla.set_flag(kfree1)
                        if iteration == count - 1:
                            tla.set_flag(qfree)
                        _fix(
                            product,
                            usptr,
                            slot,
                            sdone0,
                            sdone1,
                            sfree0,
                            sfree1,
                            sch0,
                            sch1,
                            True,
                        )
                    mm_serial = mm_serial + 1
                    with tla.vector():
                        if tla.const_expr(
                            dense
                            and generic
                            and indexed
                            and vector_score_mod is not None
                            and not ragged
                        ):
                            if iteration % 8 == 0:
                                mask_visibility = tla.make_tensor(
                                    tla.recast_ptr(ua, dtype=tla.Int32),
                                    tla.make_layout(tla.make_shape(1024, 8), tla.make_stride(8, 1)),
                                )
                                tla.pipe_barrier(tla.pipes.ALL)
                                if iteration + 8 <= count:
                                    _cache_mask_eight(
                                        mask_mod,
                                        b,
                                        head,
                                        qstart,
                                        tla.arch.sub_block_idx(),
                                        kstart,
                                        mask_visibility,
                                        aux_tensors,
                                        aux_scalars,
                                        info,
                                        True,
                                    )
                                else:
                                    _cache_mask_eight(
                                        mask_mod,
                                        b,
                                        head,
                                        qstart,
                                        tla.arch.sub_block_idx(),
                                        kstart,
                                        mask_visibility,
                                        aux_tensors,
                                        aux_scalars,
                                        info,
                                        False,
                                    )
                                tla.pipe_barrier(tla.pipes.ALL)
                        elif tla.const_expr(
                            dense and generic and indexed and vector_score_mod is not None
                        ):
                            if iteration % 2 == 0:
                                # Cache the next two dense masks before QK arrives.
                                # Both P DMAs must finish before either slot is borrowed.
                                tla.wait_flag(pfree0)
                                tla.wait_flag(pfree1)
                                cache_vi = tla.arch.sub_block_idx()
                                cache_other = up1 if slot == 0 else up0
                                cache_first = tla.make_tensor(
                                    tla.recast_ptr(upptr, dtype=tla.Int32),
                                    tla.make_layout(tla.make_shape(128, 8), tla.make_stride(8, 1)),
                                )
                                cache_second = tla.make_tensor(
                                    tla.recast_ptr(cache_other, dtype=tla.Int32),
                                    tla.make_layout(tla.make_shape(128, 8), tla.make_stride(8, 1)),
                                )
                                if iteration + 1 < count:
                                    _cache_mask_pair(
                                        mask_mod,
                                        b,
                                        head,
                                        qstart,
                                        cache_vi,
                                        kstart,
                                        kstart + 128,
                                        cache_first,
                                        cache_second,
                                        aux_tensors,
                                        aux_scalars,
                                        info,
                                        not ragged,
                                        temporal_aux,
                                    )
                                else:
                                    _cache_mask(
                                        mask_mod,
                                        b,
                                        head,
                                        qstart,
                                        cache_vi,
                                        kstart,
                                        cache_first,
                                        aux_tensors,
                                        aux_scalars,
                                        info,
                                        not ragged,
                                    )
                                tla.pipe_barrier(tla.pipes.ALL)
                                tla.set_flag(pfree0)
                                tla.set_flag(pfree1)
                        if slot == 0:
                            _vwait(sch0, tla.arch.VECTOR)
                            tla.wait_flag(pfree0)
                        else:
                            _vwait(sch1, tla.arch.VECTOR)
                            tla.wait_flag(pfree1)
                        vi = tla.arch.sub_block_idx()
                        scores = tla.make_tensor(
                            usptr,
                            tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
                        )
                        mstate = tla.make_tensor(
                            um, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                        )
                        lstate = tla.make_tensor(
                            ul, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                        )
                        astate = tla.make_tensor(
                            alphaptr,
                            tla.make_layout(tla.make_shape(64), tla.make_stride(1)),
                        )
                        pref = tla.make_tensor(
                            upptr,
                            tla.make_layout(tla.make_shape(129, 64), tla.make_stride(64, 1)),
                        )
                        ptrans = tla.make_tensor_like(upptr, pref, tla.arch.zNUnAlign)
                        pflat = tla.make_tensor(
                            upptr,
                            tla.make_layout(tla.make_shape(128, 64), tla.make_stride(16, 1)),
                        )
                        # The probability slot is free until the exp/sum pass below.
                        # FP32 predicates use one low bit per four-bit lane. Eight
                        # callback results share one word; each key row is 32 bytes.
                        visibility_ptr = tla.recast_ptr(upptr, dtype=tla.Int32)
                        if tla.const_expr(
                            dense
                            and generic
                            and indexed
                            and vector_score_mod is not None
                            and not ragged
                        ):
                            visibility_ptr = (
                                tla.recast_ptr(ua, dtype=tla.Int32) + (iteration % 8) * 1024
                            )
                        visibility = tla.make_tensor(
                            visibility_ptr,
                            tla.make_layout(tla.make_shape(128, 8), tla.make_stride(8, 1)),
                        )
                        if tla.const_expr(indexed and not generic):
                            if not full and pc != 2:
                                # Key-major scores: 128 keys x 64 queries per sub-block.
                                # Scale before masking so zero/negative scales preserve -inf.
                                if tla.const_expr(ragged):
                                    if tail_tile:
                                        _cache_mask(
                                            mask_mod,
                                            b,
                                            head,
                                            qstart,
                                            vi,
                                            kstart,
                                            visibility,
                                            aux_tensors,
                                            aux_scalars,
                                            info,
                                            False,
                                        )
                                    else:
                                        _cache_mask(
                                            mask_mod,
                                            b,
                                            head,
                                            qstart,
                                            vi,
                                            kstart,
                                            visibility,
                                            aux_tensors,
                                            aux_scalars,
                                            info,
                                            True,
                                        )
                                else:
                                    _cache_mask(
                                        mask_mod,
                                        b,
                                        head,
                                        qstart,
                                        vi,
                                        kstart,
                                        visibility,
                                        aux_tensors,
                                        aux_scalars,
                                        info,
                                        True,
                                    )
                                tla.pipe_barrier(tla.pipes.ALL)
                        if tla.const_expr(
                            generic and indexed and not (dense and vector_score_mod is not None)
                        ):
                            if tla.const_expr(vector_score_mod is not None):
                                # Dense masks need scalar coordinates, but a vector
                                # score callback can stay fused with the max pass.
                                # Borrow only the free P slot; do not rewrite scores.
                                if not full:
                                    if tail_tile:
                                        _cache_mask(
                                            mask_mod,
                                            b,
                                            head,
                                            qstart,
                                            vi,
                                            kstart,
                                            visibility,
                                            aux_tensors,
                                            aux_scalars,
                                            info,
                                            False,
                                        )
                                    else:
                                        _cache_mask(
                                            mask_mod,
                                            b,
                                            head,
                                            qstart,
                                            vi,
                                            kstart,
                                            visibility,
                                            aux_tensors,
                                            aux_scalars,
                                            info,
                                            True,
                                        )
                            else:
                                prepare_scores_simt(
                                    scores,
                                    scale,
                                    b,
                                    head,
                                    qstart + vi * 64,
                                    kstart,
                                    full,
                                    score_mod,
                                    mask_mod,
                                    aux_tensors,
                                    aux_scalars,
                                    info,
                                )
                            tla.pipe_barrier(tla.pipes.ALL)
                        with tla.vec.func(mode="simd"):
                            # The packed register contains 64 live half values followed by
                            # a duplicate half. Store exactly the first four 32-byte blocks.
                            mask16 = tla.update_mask(64, dtype=dtype)[0]
                            zero = tla.full(0.0, dtype=tla.Float32)
                            neginf = tla.full(float("-inf"), dtype=tla.Float32)
                            bv = tla.add(tla.full(0, dtype=tla.Int32), b)
                            hv = tla.add(tla.full(0, dtype=tla.Int32), head)
                            qi = tla.arange(qstart + vi * 64, dtype=tla.Int32)
                            oldm = neginf
                            oldl = zero
                            if iteration != 0:
                                oldm = mstate.load()
                                oldl = lstate.load()
                            blockm = neginf
                            if full:
                                blockm = _score_max_pass(
                                    scores,
                                    scale,
                                    bv,
                                    hv,
                                    qi,
                                    kstart,
                                    aux_tensors,
                                    aux_scalars,
                                    vector_score_mod
                                    if tla.const_expr(vector_score_mod is not None)
                                    else score_mod,
                                    mask_mod,
                                    False,
                                    visibility,
                                    False,
                                    info,
                                    tail_tile,
                                    generic and indexed and vector_score_mod is None,
                                )
                            else:
                                blockm = _score_max_pass(
                                    scores,
                                    scale,
                                    bv,
                                    hv,
                                    qi,
                                    kstart,
                                    aux_tensors,
                                    aux_scalars,
                                    vector_score_mod
                                    if tla.const_expr(vector_score_mod is not None)
                                    else score_mod,
                                    mask_mod,
                                    not indexed,
                                    visibility,
                                    indexed and (not generic or vector_score_mod is not None),
                                    info,
                                    tail_tile,
                                    generic and indexed and vector_score_mod is None,
                                )
                            newm = blockm
                            if iteration != 0:
                                newm = tla.max(oldm, blockm)
                            safem = tla.where(tla.cmp(newm, neginf, mode="eq"), zero, newm)
                            alpha = zero
                            if iteration != 0:
                                alpha = tla.where(
                                    tla.cmp(oldl, zero, mode="gt"),
                                    tla.exp(tla.sub(oldm, safem)),
                                    zero,
                                )
                            mstate.store(newm)
                            astate.store(alpha)
                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                            )
                            blockl0 = zero
                            blockl1 = zero
                            blockl2 = zero
                            blockl3 = zero
                            blockl4 = zero
                            blockl5 = zero
                            blockl6 = zero
                            blockl7 = zero
                            for prob_group in tla.range(16):
                                for prob_lane in tla.range_constexpr(8):
                                    prob_row = prob_group * 8 + prob_lane
                                    zprob = tla.tile_view(
                                        scores,
                                        tla.make_shape(1, 64),
                                        tla.make_coord(prob_row, 0),
                                    ).load()
                                    prob = tla.exp(tla.sub(zprob, safem))
                                    if tla.const_expr(prob_lane == 0):
                                        blockl0 = tla.add(blockl0, prob)
                                    elif tla.const_expr(prob_lane == 1):
                                        blockl1 = tla.add(blockl1, prob)
                                    elif tla.const_expr(prob_lane == 2):
                                        blockl2 = tla.add(blockl2, prob)
                                    elif tla.const_expr(prob_lane == 3):
                                        blockl3 = tla.add(blockl3, prob)
                                    elif tla.const_expr(prob_lane == 4):
                                        blockl4 = tla.add(blockl4, prob)
                                    elif tla.const_expr(prob_lane == 5):
                                        blockl5 = tla.add(blockl5, prob)
                                    elif tla.const_expr(prob_lane == 6):
                                        blockl6 = tla.add(blockl6, prob)
                                    else:
                                        blockl7 = tla.add(blockl7, prob)
                                    p16 = prob.to(
                                        dtype,
                                        tla.params.CastParams(
                                            round_mode=tla.params.RoundMode.CAST_ROUND,
                                            reg_slot=tla.params.RegSlot.ZERO,
                                        ),
                                    )
                                    packed, unused = tla.deinterleave(p16, p16)
                                    # Physical zNUnAlign address of (K=row,Q=0) is row*16;
                                    # VSSTB places subsequent Q16 blocks 129*16 halves apart.
                                    prow = tla.tile_view(
                                        pflat,
                                        tla.make_shape(1, 64),
                                        tla.make_coord(prob_row, 0),
                                    )
                                    prow.store(
                                        packed,
                                        tla.params.BlockStoreParams(block_stride=129),
                                        mask=mask16,
                                    )
                            blockl = tla.add(
                                tla.add(tla.add(blockl0, blockl1), tla.add(blockl2, blockl3)),
                                tla.add(tla.add(blockl4, blockl5), tla.add(blockl6, blockl7)),
                            )
                            if iteration == 0:
                                lstate.store(blockl)
                            else:
                                lstate.store(tla.add(tla.mul(oldl, alpha), blockl))
                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                            )
                        if slot == 0:
                            tla.set_flag(pready0)
                            tla.wait_flag(pready0)
                            _vset(sch0, tla.arch.VECTOR)
                        else:
                            tla.set_flag(pready1)
                            tla.wait_flag(pready1)
                            _vset(sch1, tla.arch.VECTOR)
                        if pslot == 0:
                            _vwait(pch0, tla.arch.MTE3)
                        elif pslot == 1:
                            _vwait(pch1, tla.arch.MTE3)
                        else:
                            _vwait(pch2, tla.arch.MTE3)
                        pdst = tla.tile_view(
                            _matrix(lpptr, False),
                            tla.make_shape(128, 64),
                            tla.make_coord(0, vi),
                        )
                        psrc = tla.tile_view(ptrans, tla.make_shape(128, 64), tla.make_coord(0, 0))
                        tla.copy(pdst, psrc)
                        if slot == 0:
                            tla.set_flag(pfree0)
                        else:
                            tla.set_flag(pfree1)
                        if pslot == 0:
                            _vset(pch0, tla.arch.MTE3)
                        elif pslot == 1:
                            _vset(pch1, tla.arch.MTE3)
                        else:
                            _vset(pch2, tla.arch.MTE3)

                if stage2_kind != 0:
                    # Ordered consumption recovers the original task without
                    # carrying all batch/head/address fields through the queue.
                    pv_task = tla.arch.block_idx() + pv_q_serial * tla.arch.block_num()
                    if tla.const_expr(dense):
                        pv_qb = pv_task % q_blocks
                        pv_head = pv_task // q_blocks % q_heads
                    else:
                        pv_qb = pv_task % q_blocks
                        pv_head = pv_task // q_blocks % q_heads
                    pv_b = pv_task // (q_blocks * q_heads)
                    pv_kh = pv_head // (q_heads // kv_heads)
                    pv_qstart = pv_qb * 128
                    pv_qbase = (
                        pv_b * q_heads * max_q * 128
                        + pv_head * q_head_stride
                        + pv_qstart * q_stride
                    )
                    pv_kbase = pv_b * kv_heads * max_k * 128 + pv_kh * k_head_stride
                    pv_lbase = (
                        pv_b * l_batch_stride + pv_head * l_head_stride + pv_qstart * l_stride
                    )
                    pv_qlen = max_q
                    pv_klen = max_k
                    if tla.const_expr(ragged):
                        if tla.const_expr(varlen[0]):
                            pv_qs = cu_q[pv_b]
                            pv_qlen = cu_q[pv_b + 1] - pv_qs
                            pv_qbase = pv_qs * q_stride + pv_head * 128 + pv_qstart * q_stride
                            pv_lbase = pv_head * total_q + pv_qs + pv_qstart
                        if tla.const_expr(varlen[1]):
                            pv_ks = cu_k[pv_b]
                            pv_klen = cu_k[pv_b + 1] - pv_ks
                            pv_kbase = pv_ks * k_stride + pv_kh * 128
                    pv_qslot = pv_q_serial % 3
                    pv_um = um0
                    pv_ul = ul0
                    if pv_qslot == 1:
                        pv_um = um1
                        pv_ul = ul1
                    elif pv_qslot == 2:
                        pv_um = um2
                        pv_ul = ul2
                    if stage2_kind == 1:
                        pv_iter = pv_tile_index
                        pvslot = (tick - 2) % 2
                        pvpslot = (tick - 2) % 3
                        vblock = stage2_kb
                        lvptr = lv0 if pvslot == 0 else lv1
                        loptr = lo0 if pvslot == 0 else lo1
                        uoptr = uo0 if pvslot == 0 else uo1
                        pvlpptr = lp0
                        pvalphaptr = alpha0
                        if pvpslot == 1:
                            pvlpptr = lp1
                            pvalphaptr = alpha1
                        elif pvpslot == 2:
                            pvlpptr = lp2
                            pvalphaptr = alpha2
                        pvabslot = mm_serial % 2
                        pvlaptr = la0 if pvabslot == 0 else la1
                        pvlbptr = lb0 if pvabslot == 0 else lb1
                        valid_v = min(128, pv_klen - vblock * 128)
                        pipeline_load_rows(
                            v,
                            pv_kbase + vblock * 128 * k_stride,
                            k_stride,
                            valid_v,
                            lvptr,
                            False,
                            pvslot,
                            vfree0,
                            vfree1,
                            vload0,
                            vload1,
                            ragged,
                            tail_scratch,
                            tail_zero,
                            tail_loaded,
                            tail_stored,
                            tail_free,
                            tail_ready,
                            True,
                            (outfree,),
                        )
                        with tla.cube():
                            if pvpslot == 0:
                                _cwait(pch0, tla.arch.MTE1)
                            elif pvpslot == 1:
                                _cwait(pch1, tla.arch.MTE1)
                            else:
                                _cwait(pch2, tla.arch.MTE1)
                            pv = _mm(
                                _matrix(pvlpptr, True),
                                _matrix(lvptr, False),
                                pvlaptr,
                                pvlbptr,
                                loptr,
                                pvabslot,
                                pvslot,
                                abload0,
                                abload1,
                                abfree0,
                                abfree1,
                                ofree0,
                                ofree1,
                                odone0,
                                odone1,
                            )
                            if pvslot == 0:
                                tla.set_flag(vfree0)
                            else:
                                tla.set_flag(vfree1)
                            if pvpslot == 0:
                                _cset(pch0, tla.arch.MTE1)
                            elif pvpslot == 1:
                                _cset(pch1, tla.arch.MTE1)
                            else:
                                _cset(pch2, tla.arch.MTE1)
                            _fix(
                                pv,
                                uoptr,
                                pvslot,
                                odone0,
                                odone1,
                                ofree0,
                                ofree1,
                                och0,
                                och1,
                                False,
                            )
                        mm_serial = mm_serial + 1
                        with tla.vector():
                            if pvslot == 0:
                                _vwait(och0, tla.arch.VECTOR)
                            else:
                                _vwait(och1, tla.arch.VECTOR)
                            pvub = tla.make_tensor(
                                uoptr,
                                tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
                            )
                            acc_target_ptr = ua
                            if tla.const_expr(
                                dense
                                and generic
                                and indexed
                                and vector_score_mod is not None
                                and not ragged
                            ):
                                acc_target_ptr = uoptr
                            accub = tla.make_tensor(
                                acc_target_ptr,
                                tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
                            )
                            aub = tla.make_tensor(
                                pvalphaptr,
                                tla.make_layout(tla.make_shape(64), tla.make_stride(1)),
                            )
                            # Keep the first PV buffer owned until its only reader
                            # (the second accumulation or singleton output) finishes.
                            acc_source_ptr = ua
                            if tla.const_expr(
                                dense
                                and generic
                                and indexed
                                and vector_score_mod is not None
                                and not ragged
                            ):
                                acc_source_ptr = uo1 if pvslot == 0 else uo0
                            elif pv_iter == 1:
                                acc_source_ptr = uo1 if pvslot == 0 else uo0
                            acc_source = tla.make_tensor(
                                acc_source_ptr,
                                tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
                            )
                            out_pv = tla.make_tensor(
                                uout,
                                tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
                            )
                            m_pv = tla.make_tensor(
                                pv_um, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                            )
                            l_pv = tla.make_tensor(
                                pv_ul, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                            )
                            lse_pv = tla.make_tensor(
                                ulse_out, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                            )
                            if pv_iter != 0:
                                if stage2_last != 0:
                                    tla.wait_flag(outfree)
                                with tla.vec.func(mode="simd"):
                                    if stage2_last != 0:
                                        _accumulate_pv(
                                            acc_source,
                                            accub,
                                            pvub,
                                            aub,
                                            out_pv,
                                            m_pv,
                                            l_pv,
                                            lse_pv,
                                            dtype,
                                            True,
                                        )
                                    else:
                                        _accumulate_pv(
                                            acc_source,
                                            accub,
                                            pvub,
                                            aub,
                                            out_pv,
                                            m_pv,
                                            l_pv,
                                            lse_pv,
                                            dtype,
                                            False,
                                        )
                                if tla.const_expr(
                                    dense
                                    and generic
                                    and indexed
                                    and vector_score_mod is not None
                                    and not ragged
                                ):
                                    # Keep the current PV buffer as the accumulator;
                                    # only its previous source is free for the next FIX.
                                    if pvslot == 0:
                                        _vset(och1, tla.arch.VECTOR)
                                    else:
                                        _vset(och0, tla.arch.VECTOR)
                                    if stage2_last != 0:
                                        if pvslot == 0:
                                            _vset(och0, tla.arch.VECTOR)
                                        else:
                                            _vset(och1, tla.arch.VECTOR)
                                else:
                                    if pvslot == 0:
                                        _vset(och0, tla.arch.VECTOR)
                                        if pv_iter == 1:
                                            _vset(och1, tla.arch.VECTOR)
                                    else:
                                        _vset(och1, tla.arch.VECTOR)
                                        if pv_iter == 1:
                                            _vset(och0, tla.arch.VECTOR)

                    else:
                        # EMPTY is an ordered query event. It uses no Cube/P
                        # tokens and only clears the accumulator when consumed.
                        if pv_qstart < pv_qlen:
                            with tla.vector():
                                acc_empty = tla.make_tensor(
                                    ua,
                                    tla.make_layout(
                                        tla.make_shape(128, 64), tla.make_stride(64, 1)
                                    ),
                                )
                                with tla.vec.func(mode="simd"):
                                    for ri in tla.range(128):
                                        tla.tile_view(
                                            acc_empty,
                                            tla.make_shape(1, 64),
                                            tla.make_coord(ri, 0),
                                        ).store(tla.full(0.0, dtype=tla.Float32))
                                    tla.local_mem_bar(
                                        tla.params.MemType.VEC_STORE,
                                        tla.params.MemType.VEC_LOAD,
                                    )
                    if stage2_last != 0:
                        if pv_qstart < pv_qlen:
                            with tla.vector():
                                # One token protects both asynchronous output DMA sources.
                                if stage2_kind != 1 or pv_tile_index == 0:
                                    tla.wait_flag(outfree)
                                vo = tla.arch.sub_block_idx()
                                final_acc_ptr = ua
                                if stage2_kind == 1 and pv_tile_index == 0:
                                    final_acc_ptr = uo0 if (tick - 2) % 2 == 0 else uo1
                                acc_final = tla.make_tensor(
                                    final_acc_ptr,
                                    tla.make_layout(
                                        tla.make_shape(64, 128), tla.make_stride(128, 1)
                                    ),
                                )
                                out_final = tla.make_tensor(
                                    uout,
                                    tla.make_layout(
                                        tla.make_shape(64, 128), tla.make_stride(128, 1)
                                    ),
                                )
                                m_final = tla.make_tensor(
                                    pv_um,
                                    tla.make_layout(tla.make_shape(64), tla.make_stride(1)),
                                )
                                l_final = tla.make_tensor(
                                    pv_ul,
                                    tla.make_layout(tla.make_shape(64), tla.make_stride(1)),
                                )
                                lse_final = tla.make_tensor(
                                    ulse_out,
                                    tla.make_layout(tla.make_shape(64), tla.make_stride(1)),
                                )
                                if stage2_kind != 1 or pv_tile_index == 0:
                                    with tla.vec.func(mode="simd"):
                                        # All 64 query-row LSEs share one SIMD operation.
                                        final_m = m_final.load()
                                        final_l = l_final.load()
                                        safe_final_l = tla.where(
                                            tla.cmp(final_l, 0.0, mode="gt"),
                                            final_l,
                                            tla.full(1.0, dtype=tla.Float32),
                                        )
                                        lse_final.store(tla.add(final_m, tla.log(safe_final_l)))
                                        # Compute one reciprocal per query, then reuse across D lanes.
                                        l_final.store(
                                            tla.div(tla.full(1.0, dtype=tla.Float32), safe_final_l)
                                        )
                                        tla.local_mem_bar(
                                            tla.params.MemType.VEC_STORE,
                                            tla.params.MemType.VEC_LOAD,
                                        )
                                        half_final = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=dtype
                                        )
                                        for rf in tla.range(64):
                                            lr_final = tla.tile_view(
                                                l_final, tla.make_shape(1), tla.make_coord(rf)
                                            )
                                            lv_final = lr_final.load(
                                                tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                )
                                            )
                                            for cf in tla.range(2):
                                                aval = tla.tile_view(
                                                    acc_final,
                                                    tla.make_shape(1, 64),
                                                    tla.make_coord(rf, cf),
                                                ).load()
                                                oval = tla.mul(aval, lv_final).to(
                                                    dtype,
                                                    tla.params.CastParams(
                                                        round_mode=tla.params.RoundMode.CAST_ROUND,
                                                        reg_slot=tla.params.RegSlot.ZERO,
                                                    ),
                                                )
                                                tla.tile_view(
                                                    out_final,
                                                    tla.make_shape(1, 64),
                                                    tla.make_coord(rf, cf),
                                                ).store(
                                                    oval,
                                                    tla.params.NormalStoreParams(
                                                        store_dist=tla.params.StoreDist.DIST_PACK_B32
                                                    ),
                                                    mask=half_final,
                                                )
                                tla.set_flag(outready)
                                tla.wait_flag(outready)
                                if stage2_kind == 1 and pv_tile_index == 0:
                                    if (tick - 2) % 2 == 0:
                                        _vset(och0, tla.arch.VECTOR)
                                    else:
                                        _vset(och1, tla.arch.VECTOR)
                                store_rows = min(64, max(0, pv_qlen - pv_qstart - vo * 64))
                                if store_rows > 0:
                                    ogm = tla.make_tensor(
                                        out.ptr + pv_qbase + vo * 64 * q_stride,
                                        tla.make_layout(
                                            tla.make_shape(store_rows, 128),
                                            tla.make_stride(q_stride, 1),
                                        ),
                                    )
                                    lgm = tla.make_tensor(
                                        lse.ptr + pv_lbase + vo * 64,
                                        tla.make_layout(
                                            tla.make_shape(store_rows), tla.make_stride(1)
                                        ),
                                    )
                                    tla.copy(
                                        ogm,
                                        tla.tile_view(
                                            out_final,
                                            tla.make_shape(store_rows, 128),
                                            tla.make_coord(0, 0),
                                        ),
                                    )
                                    tla.copy(
                                        lgm,
                                        tla.tile_view(
                                            lse_final, tla.make_shape(store_rows), tla.make_coord(0)
                                        ),
                                    )
                                tla.set_flag(outfree)
                        pv_q_serial = pv_q_serial + 1
                        pv_tile_index = tla.as_numeric(0)
                    else:
                        pv_tile_index = pv_tile_index + 1
                tick = tick + 1
            q_serial = q_serial + 1

        # Drain all initialized ownership tokens before the block terminates.
        with tla.cube():
            tla.wait_flag(qfree)
            tla.wait_flag(kfree0)
            tla.wait_flag(kfree1)
            tla.wait_flag(vfree0)
            tla.wait_flag(vfree1)
            tla.wait_flag(abfree0)
            tla.wait_flag(abfree1)
            tla.wait_flag(sfree0)
            tla.wait_flag(sfree1)
            tla.wait_flag(ofree0)
            tla.wait_flag(ofree1)
            _cwait(sch0, tla.arch.FIX)
            _cwait(sch1, tla.arch.FIX)
            _cwait(och0, tla.arch.FIX)
            _cwait(och1, tla.arch.FIX)
        with tla.vector():
            tla.wait_flag(pfree0)
            tla.wait_flag(pfree1)
            tla.wait_flag(outfree)
            _vwait(pch0, tla.arch.MTE3)
            _vwait(pch1, tla.arch.MTE3)
            _vwait(pch2, tla.arch.MTE3)
