"""Ordinary M128 backward with continuous cross-query producer/consumer overlap.

D128 FP16/BF16 with bounded BSND/BNSD/packed GQA and generic SIMD/SIMT callbacks.
Identity-score specializations retain sparse-mask preprocessing. Each AIV owns 64 rows of a query head.
Query-resident Q/dO and dQ accumulators use two slots; score/PDS slots follow
global tile parity. EMPTY queries emit zero output through normal ownership.
PDS publication and the shared output helper retain their ALL barriers.
"""

import catlass.tla as tla

from .common import (
    l1_matrix,
    store_output,
    pipeline_load_rows,
    cube_wait as _cv_wait,
    cube_signal as _cv_set,
    vector_wait as _v_wait,
    vector_signal as _v_set,
)
from .window import window_block_bounds
from .callbacks import prepare_pds_simt


@tla.jit
def _mm(a, b, ap, bp, cp, ab_free, ab_ready, c_free, c_ready):
    aa = tla.make_tensor_like(ap, a, tla.arch.zN)
    bb = tla.make_tensor_like(bp, b, tla.arch.nZ)
    ref = tla.make_tensor(cp, tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)))
    cc = tla.make_tensor_like(cp, ref, tla.arch.L0Clayout)
    tla.wait_flag(ab_free)
    tla.copy(aa, a)
    tla.copy(bb, b)
    tla.set_flag(ab_ready)
    tla.wait_flag(ab_ready)
    tla.wait_flag(c_free)
    tla.mmad(cc, aa, bb, init_c=True)
    tla.set_flag(ab_free)
    tla.set_flag(c_ready)
    tla.wait_flag(c_ready)
    return cc


@tla.jit
def _publish_dn(ub_ptr, l1_ptr):
    # Two AIVs own complementary 64-query column slices of P'[K,Q].
    vi = tla.arch.sub_block_idx()
    # DMA interpretation: four [129K,16Q] blocks, with the padding row skipped.
    ref = tla.make_tensor(ub_ptr, tla.make_layout(tla.make_shape(129, 64), tla.make_stride(64, 1)))
    nz = tla.make_tensor_like(ub_ptr, ref, tla.arch.zNUnAlign)
    ub = tla.tile_view(nz, tla.make_shape(128, 64), tla.make_coord(0, 0))
    matrix = l1_matrix(l1_ptr, False)
    half = tla.tile_view(matrix, tla.make_shape(128, 64), tla.make_coord(0, vi))
    tla.copy(half, ub)


@tla.jit
def _compact_mask(
    mask_mod: tla.Constexpr,
    batch_idx,
    head,
    qstart,
    vi2,
    kb,
    visibility,
    aux_tensors,
    aux_scalars,
    info,
    bounds_checked: tla.Constexpr,
):
    with tla.vec.func(mode="simt", thread_block_dim=1024):
        mask_tid, _, _ = tla.arch.thread_idx()
        # Adjacent lanes traverse keys; one word packs eight query predicates.
        mask_k = mask_tid % 128
        mask_group = mask_tid // 128
        mask_word = tla.Int32(0)
        for mask_lane in tla.range_constexpr(8):
            mask_q = mask_group * 8 + mask_lane
            if mask_mod(
                tla.Float32(0.0),
                batch_idx,
                head,
                qstart + vi2 * 64 + mask_q,
                kb * 128 + mask_k,
                aux_tensors,
                aux_scalars,
                info,
                bounds_checked,
            ):
                mask_word = mask_word | (1 << (mask_lane * 4))
        visibility[mask_k, mask_group] = mask_word


@tla.jit
def _pds_pass_impl(
    s_ub,
    dp_ub,
    p_ub,
    ds_ub,
    scale,
    neg_inf,
    zero,
    bv,
    hv,
    qidx,
    kb,
    safe_lse,
    empty,
    d2,
    aux_tensors,
    aux_scalars,
    mask_mod: tla.Constexpr,
    apply_mask: tla.Constexpr,
    info=(),
    bounded: tla.Constexpr = True,
    score_mod: tla.Constexpr = (),
    score_grad: tla.Constexpr = (),
    generic: tla.Constexpr = False,
    prepared: tla.Constexpr = False,
    visibility=(),
    compact_mask: tla.Constexpr = False,
):
    dtype = p_ub.ptr.dtype
    packed_mask = tla.update_mask(64, dtype=dtype)[0]
    for group in tla.range(32):
        for row_slot in tla.range_constexpr(4):
            krow = group * 4 + row_slot
            coord = tla.make_coord(krow, 0)
            if tla.const_expr(prepared):
                pp = tla.tile_view(s_ub, tla.make_shape(1, 64), coord).load()
                ds2 = tla.tile_view(dp_ub, tla.make_shape(1, 64), coord).load()
            else:
                raw = tla.mul(tla.tile_view(s_ub, tla.make_shape(1, 64), coord).load(), scale)
                x = raw
                kidx = tla.add(tla.full(0, dtype=tla.Int32), kb * 128 + krow)
                callback_q = qidx
                callback_k = kidx
                if tla.const_expr(len(info) > 0 and bounded):
                    callback_q = tla.min(qidx, info[0] - 1)
                    callback_k = tla.min(kidx, info[1] - 1)
                if tla.const_expr(generic):
                    x = score_mod(
                        raw, bv, hv, callback_q, callback_k, aux_tensors, aux_scalars, info
                    )
                if tla.const_expr(compact_mask):
                    visible = tla.tile_view(visibility, tla.make_shape(1, 8), coord).load(
                        tla.params.MaskLoadParams()
                    )
                    # Mask after scaling to preserve -inf for nonpositive scales.
                    x = tla.where(visible, x, neg_inf)
                if tla.const_expr(apply_mask):
                    visible = mask_mod(
                        zero, bv, hv, callback_q, callback_k, aux_tensors, aux_scalars, info
                    )
                    x = tla.where(visible, x, neg_inf)
                if tla.const_expr(len(info) > 0 and bounded):
                    x = tla.where(tla.cmp(qidx, info[0], mode="lt"), x, neg_inf)
                    x = tla.where(tla.cmp(kidx, info[1], mode="lt"), x, neg_inf)
                pp = tla.exp(tla.sub(x, safe_lse))
                pp = tla.where(empty, zero, pp)
                dp2 = tla.tile_view(dp_ub, tla.make_shape(1, 64), coord).load()
                ds2 = tla.mul(pp, tla.sub(dp2, d2))
                ds2 = tla.where(tla.cmp(pp, zero, mode="gt"), ds2, zero)
                if tla.const_expr(generic):
                    ds2 = score_grad(
                        ds2, raw, bv, hv, callback_q, callback_k, info, aux_tensors, aux_scalars
                    )
                    ds2 = tla.where(tla.cmp(x, neg_inf, mode="ne"), ds2, zero)
            p16 = pp.to(
                dtype,
                tla.params.CastParams(
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                    reg_slot=tla.params.RegSlot.ZERO,
                ),
            )
            ds16 = ds2.to(
                dtype,
                tla.params.CastParams(
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                    reg_slot=tla.params.RegSlot.ZERO,
                ),
            )
            packed_p16, unused_p16 = tla.deinterleave(p16, p16)
            # Flat row*16 addressing avoids zNUnAlign runtime layout division.
            tla.tile_view(p_ub, tla.make_shape(1, 64), coord).store(
                packed_p16,
                tla.params.BlockStoreParams(block_stride=129),
                mask=packed_mask,
            )
            packed_ds16, unused_ds16 = tla.deinterleave(ds16, ds16)
            # Flat row*16 addressing avoids zNUnAlign runtime layout division.
            tla.tile_view(ds_ub, tla.make_shape(1, 64), coord).store(
                packed_ds16,
                tla.params.BlockStoreParams(block_stride=129),
                mask=packed_mask,
            )


@tla.jit
def _pds_pass(
    s_ub,
    dp_ub,
    p_ub,
    ds_ub,
    scale,
    neg_inf,
    zero,
    bv,
    hv,
    qidx,
    kb,
    safe_lse,
    empty,
    d2,
    aux_tensors,
    aux_scalars,
    mask_mod: tla.Constexpr,
    apply_mask: tla.Constexpr,
    info=(),
    needs_bounds=False,
    score_mod: tla.Constexpr = (),
    score_grad: tla.Constexpr = (),
    generic: tla.Constexpr = False,
    prepared: tla.Constexpr = False,
    visibility=(),
    compact_mask: tla.Constexpr = False,
):
    if tla.const_expr(len(info) == 0):
        _pds_pass_impl(
            s_ub,
            dp_ub,
            p_ub,
            ds_ub,
            scale,
            neg_inf,
            zero,
            bv,
            hv,
            qidx,
            kb,
            safe_lse,
            empty,
            d2,
            aux_tensors,
            aux_scalars,
            mask_mod,
            apply_mask,
            info,
            False,
            score_mod,
            score_grad,
            generic,
            prepared,
            visibility,
            compact_mask,
        )
    else:
        if needs_bounds:
            _pds_pass_impl(
                s_ub,
                dp_ub,
                p_ub,
                ds_ub,
                scale,
                neg_inf,
                zero,
                bv,
                hv,
                qidx,
                kb,
                safe_lse,
                empty,
                d2,
                aux_tensors,
                aux_scalars,
                mask_mod,
                apply_mask,
                info,
                True,
                score_mod,
                score_grad,
                generic,
                prepared,
                visibility,
                compact_mask,
            )
        else:
            _pds_pass_impl(
                s_ub,
                dp_ub,
                p_ub,
                ds_ub,
                scale,
                neg_inf,
                zero,
                bv,
                hv,
                qidx,
                kb,
                safe_lse,
                empty,
                d2,
                aux_tensors,
                aux_scalars,
                mask_mod,
                apply_mask,
                info,
                False,
                score_mod,
                score_grad,
                generic,
                prepared,
                visibility,
                compact_mask,
            )


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
    workspace: tuple = (),
    lse_gradient: tuple = (),
):
    # Q/dO/dQ and K/V/dK/dV share the input layout, including FP32 workspaces.
    q_stride = q_heads * 128 if tla.const_expr(sequence_major[0]) else 128
    k_stride = kv_heads * 128 if tla.const_expr(sequence_major[1]) else 128
    q_head_stride = 128 if tla.const_expr(sequence_major[0]) else max_q * 128
    k_head_stride = 128 if tla.const_expr(sequence_major[1]) else max_k * 128
    l_stride = 1
    l_head_stride = total_q if tla.const_expr(varlen[0]) else max_q
    l_batch_stride = max_q if tla.const_expr(varlen[0]) else q_heads * max_q
    dtype = q.ptr.dtype
    lq0 = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
    lq1 = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
    ld0 = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
    ld1 = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
    lk = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
    lv = tla.allocate(16384, dtype, tla.AddressSpace.l1, 512)
    lp = tla.allocate(32768, dtype, tla.AddressSpace.l1, 512)
    lds = tla.allocate(32768, dtype, tla.AddressSpace.l1, 512)
    a0 = tla.allocate(16384, dtype, tla.AddressSpace.l0a, 512)
    a1 = tla.allocate(16384, dtype, tla.AddressSpace.l0a, 512)
    b0 = tla.allocate(16384, dtype, tla.AddressSpace.l0b, 512)
    b1 = tla.allocate(16384, dtype, tla.AddressSpace.l0b, 512)
    c0 = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0c, 512)
    c1 = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0c, 512)
    cq0 = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0c, 512)
    cq1 = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0c, 512)
    # Each AIV receives 64 query columns from SPLIT_N: UB[K,Q/2].
    scores = tla.allocate(16384, tla.Float32, tla.AddressSpace.ub, 256)
    dps = tla.allocate(16384, tla.Float32, tla.AddressSpace.ub, 256)
    probs = tla.allocate(8256, dtype, tla.AddressSpace.ub, 256)
    derivs = tla.allocate(8256, dtype, tla.AddressSpace.ub, 256)
    delta = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    logsum = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    dq_stage = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
    dq_cast = tla.allocate(8192, dtype, tla.AddressSpace.ub, 256)

    if tla.const_expr(len(workspace) > 0):
        workspace_all = tla.cross_flag("workspace_all", mode=0)
        workspace_handoff = tla.cross_flag("workspace_handoff", mode=4)
        # Initialize shared FP32 targets before any core starts atomics; reuse UB.
        ws_init_size = workspace[2]
        with tla.vector():
            ws_init_zeros_ub = tla.make_tensor(
                scores, tla.make_layout(tla.make_shape(4096), tla.make_stride(1))
            )
            with tla.vec.func(mode="simd"):
                ws_init_zero = tla.full(0.0, dtype=tla.Float32)
                for ws_init_row in tla.range(64):
                    tla.tile_view(
                        ws_init_zeros_ub,
                        tla.make_shape(64),
                        tla.make_coord(ws_init_row),
                    ).store(ws_init_zero)
            tla.pipe_barrier(tla.pipes.ALL)
            ws_init_worker = tla.arch.block_idx() * 2 + tla.arch.sub_block_idx()
            for ws_init_base in tla.range(
                ws_init_worker * 4096, ws_init_size, tla.arch.block_num() * 2 * 4096
            ):
                ws_init_count = (
                    4096 if ws_init_size - ws_init_base > 4096 else ws_init_size - ws_init_base
                )
                ws_init_layout = tla.make_layout(tla.make_shape(ws_init_count), tla.make_stride(1))
                ws_init_src = tla.tile_view(
                    ws_init_zeros_ub, tla.make_shape(ws_init_count), tla.make_coord(0)
                )
                tla.copy(
                    tla.make_tensor(dk_acc.ptr + ws_init_base, ws_init_layout),
                    ws_init_src,
                )
                tla.copy(
                    tla.make_tensor(dv_acc.ptr + ws_init_base, ws_init_layout),
                    ws_init_src,
                )
            # All zero stores, on all AIVs, precede the first atomic update.
            tla.cross_core_set_flag(workspace_all, tla.arch.MTE3)
            tla.cross_core_wait_flag(workspace_all, tla.arch.MTE3)
            if tla.arch.sub_block_idx() == 0:
                tla.cross_core_set_flag(workspace_handoff, tla.arch.MTE3, aiv_id=0)
            else:
                tla.cross_core_set_flag(workspace_handoff, tla.arch.MTE3, aiv_id=1)
            tla.pipe_barrier(tla.pipes.ALL)
        with tla.cube():
            tla.cross_core_wait_flag(workspace_handoff, tla.arch.FIX, aiv_id=0)
            tla.cross_core_wait_flag(workspace_handoff, tla.arch.FIX, aiv_id=1)
            tla.pipe_barrier(tla.pipes.ALL)

    ab0_free = tla.flag("ab0_free", tla.arch.CUBE, tla.arch.MTE1)
    ab0_ready = tla.flag("ab0_ready", tla.arch.MTE1, tla.arch.CUBE)
    ab1_free = tla.flag("ab1_free", tla.arch.CUBE, tla.arch.MTE1)
    ab1_ready = tla.flag("ab1_ready", tla.arch.MTE1, tla.arch.CUBE)
    c0_free = tla.flag("c0_free", tla.arch.FIX, tla.arch.CUBE)
    c0_ready = tla.flag("c0_ready", tla.arch.CUBE, tla.arch.FIX)
    c1_free = tla.flag("c1_free", tla.arch.FIX, tla.arch.CUBE)
    c1_ready = tla.flag("c1_ready", tla.arch.CUBE, tla.arch.FIX)
    qdo_ready0 = tla.flag("qdo_ready0", tla.arch.MTE2, tla.arch.MTE1)
    qdo_ready1 = tla.flag("qdo_ready1", tla.arch.MTE2, tla.arch.MTE1)
    tail_scratch = dq_cast
    tail_zero = tla.flag("tail_zero", tla.arch.VECTOR, tla.arch.MTE2)
    tail_loaded = tla.flag("tail_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    tail_stored = tla.flag("tail_stored", tla.arch.VECTOR, tla.arch.MTE3)
    tail_free = tla.cross_flag("tail_free", mode=4)
    tail_ready = tla.cross_flag("tail_ready", mode=4)

    qdo_free0 = tla.flag("qdo_free0", tla.arch.MTE1, tla.arch.MTE2)
    qdo_free1 = tla.flag("qdo_free1", tla.arch.MTE1, tla.arch.MTE2)
    k_ready = tla.flag("k_ready", tla.arch.MTE2, tla.arch.MTE1)
    v_ready = tla.flag("v_ready", tla.arch.MTE2, tla.arch.MTE1)
    k_free = tla.flag("k_free", tla.arch.MTE1, tla.arch.MTE2)
    v_free = tla.flag("v_free", tla.arch.MTE1, tla.arch.MTE2)
    cq_ready0 = tla.flag("cq_ready0", tla.arch.CUBE, tla.arch.FIX)
    cq_ready1 = tla.flag("cq_ready1", tla.arch.CUBE, tla.arch.FIX)
    cq_free0 = tla.flag("cq_free0", tla.arch.FIX, tla.arch.CUBE)
    cq_free1 = tla.flag("cq_free1", tla.arch.FIX, tla.arch.CUBE)
    vec_loaded = tla.flag("vec_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_stored = tla.flag("vec_stored", tla.arch.VECTOR, tla.arch.MTE3)
    cv_ready0 = tla.cross_flag("cv_ready0", mode=4)
    cv_ready1 = tla.cross_flag("cv_ready1", mode=4)
    cv_free0 = tla.cross_flag("cv_free0", mode=4)
    cv_free1 = tla.cross_flag("cv_free1", mode=4)
    pds_ready0 = tla.cross_flag("pds_ready0", mode=4)
    pds_ready1 = tla.cross_flag("pds_ready1", mode=4)
    pds_free0 = tla.cross_flag("pds_free0", mode=4)
    pds_free1 = tla.cross_flag("pds_free1", mode=4)
    output_ready = tla.cross_flag("output_ready", mode=4)
    output_free = tla.cross_flag("output_free", mode=4)

    # One token epoch per physical worker, including workers with no tasks.
    with tla.cube():
        tla.set_flag(ab0_free)
        tla.set_flag(ab1_free)
        tla.set_flag(c0_free)
        tla.set_flag(c1_free)
        tla.set_flag(k_free)
        tla.set_flag(v_free)
        tla.set_flag(qdo_free0)
        tla.set_flag(qdo_free1)
        tla.set_flag(cq_free0)
        tla.set_flag(cq_free1)
        _cv_set(pds_free0, tla.arch.MTE1)
        _cv_set(pds_free1, tla.arch.MTE1)
    with tla.vector():
        _v_set(cv_free0, tla.arch.VECTOR)
        _v_set(cv_free1, tla.arch.VECTOR)
        _v_set(output_free, tla.arch.MTE3)

    tick = tla.as_numeric(0)
    query_serial = tla.as_numeric(0)
    consumer_serial = tla.as_numeric(0)
    consumer_index = tla.as_numeric(0)
    # Event kind: 0 INVALID, 1 real selected KV block, 2 EMPTY query.
    stage0_kind = tla.as_numeric(0)
    stage1_kind = tla.as_numeric(0)
    stage0_kb = tla.as_numeric(0)
    stage1_kb = tla.as_numeric(0)
    stage0_last = tla.as_numeric(0)
    stage1_last = tla.as_numeric(0)
    consumer_qslot = tla.as_numeric(0)
    consumer_qlen = max_q
    consumer_klen = max_k
    consumer_qb = tla.as_numeric(0)
    consumer_qbase = tla.as_numeric(0)
    consumer_kbase = tla.as_numeric(0)
    consumer_lq = lq0
    consumer_ld = ld0
    consumer_cq = cq0

    for task in tla.range(tla.arch.block_idx(), batch * q_heads * q_blocks, tla.arch.block_num()):
        # Nearby tasks share a KV head, limiting the active cache working set.
        qb = task % q_blocks
        head = task // q_blocks % q_heads
        batch_idx = task // (q_blocks * q_heads)
        kh = head // (q_heads // kv_heads)
        qstart = qb * 128
        qbase = batch_idx * q_heads * max_q * 128 + head * q_head_stride + qstart * q_stride
        kbase = batch_idx * kv_heads * max_k * 128 + kh * k_head_stride
        lbase = batch_idx * l_batch_stride + head * l_head_stride + qstart * l_stride
        qlen = max_q
        klen = max_k
        qs = tla.as_numeric(0)
        ks = tla.as_numeric(0)
        info = ()
        if tla.const_expr(ragged):
            if tla.const_expr(varlen[0]):
                qs = cu_q[batch_idx]
                qlen = cu_q[batch_idx + 1] - qs
                qbase = qs * q_stride + head * 128 + qstart * q_stride
                lbase = head * total_q + qs + qstart
            if tla.const_expr(varlen[1]):
                ks = cu_k[batch_idx]
                klen = cu_k[batch_idx + 1] - ks
                kbase = ks * k_stride + kh * 128
            info = (qlen, klen, qs, ks)
        if tla.const_expr(generic):
            if tla.const_expr(not ragged):
                qs = batch_idx * max_q if tla.const_expr(varlen[0]) else tla.as_numeric(0)
                ks = batch_idx * max_k if tla.const_expr(varlen[1]) else tla.as_numeric(0)
            info = (qlen, klen, qs, ks)
        valid_q = min(128, max(0, qlen - qstart))
        mask_row = ((batch_idx % mask_batch) * mask_heads + head % mask_heads) * q_blocks + qb
        pcount = tla.as_numeric(0)
        count = (klen + 127) // 128
        if tla.const_expr(not dense):
            pcount = partial_count[mask_row]
            count = pcount + full_count[mask_row]

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
        qslot = query_serial % 2
        lq = lq0
        ld = ld0
        if qslot == 1:
            lq = lq1
            ld = ld1
        if count > 0:
            pipeline_load_rows(
                q,
                qbase,
                q_stride,
                valid_q,
                lq,
                False,
                qslot,
                qdo_free0,
                qdo_free1,
                qdo_ready0,
                qdo_ready1,
                ragged,
                tail_scratch,
                tail_zero,
                tail_loaded,
                tail_stored,
                tail_free,
                tail_ready,
            )
            with tla.cube():
                if qslot == 0:
                    tla.set_flag(qdo_free0)
                else:
                    tla.set_flag(qdo_free1)
            pipeline_load_rows(
                dout,
                qbase,
                q_stride,
                valid_q,
                ld,
                False,
                qslot,
                qdo_free0,
                qdo_free1,
                qdo_ready0,
                qdo_ready1,
                ragged,
                tail_scratch,
                tail_zero,
                tail_loaded,
                tail_stored,
                tail_free,
                tail_ready,
            )
        if count > 0:
            with tla.vector():
                vi = tla.arch.sub_block_idx()
                cache_layout = tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1))
                du = tla.make_tensor(probs, cache_layout)
                ou = tla.make_tensor(derivs, cache_layout)
                dgm = tla.make_tensor(
                    dout.ptr + qbase + vi * 64 * q_stride,
                    tla.make_layout(tla.make_shape(64, 128), tla.make_stride(q_stride, 1)),
                )
                ogm = tla.make_tensor(
                    out.ptr + qbase + vi * 64 * q_stride,
                    tla.make_layout(tla.make_shape(64, 128), tla.make_stride(q_stride, 1)),
                )
                ls = tla.make_tensor(
                    logsum, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                )
                dd = tla.make_tensor(delta, tla.make_layout(tla.make_shape(64), tla.make_stride(1)))
                lgm = tla.make_tensor(
                    lse.ptr + lbase + vi * 64 * l_stride,
                    tla.make_layout(tla.make_shape(64), tla.make_stride(l_stride)),
                )
                if valid_q < 128:
                    with tla.vec.func(mode="simd"):
                        for row_zero in tla.range(64):
                            tla.tile_view(
                                du, tla.make_shape(1, 128), tla.make_coord(row_zero, 0)
                            ).store(tla.full(0.0, dtype=dtype))
                            tla.tile_view(
                                ou, tla.make_shape(1, 128), tla.make_coord(row_zero, 0)
                            ).store(tla.full(0.0, dtype=dtype))
                        ls.store(tla.full(float("-inf"), dtype=tla.Float32))
                        if tla.const_expr(len(lse_gradient) > 0):
                            dd.store(tla.full(0.0, dtype=tla.Float32))
                    tla.set_flag(tail_zero)
                    tla.wait_flag(tail_zero)
                    rows_ub = min(64, max(0, valid_q - vi * 64))
                    if rows_ub > 0:
                        dgm_valid = tla.make_tensor(
                            dout.ptr + qbase + vi * 64 * q_stride,
                            tla.make_layout(
                                tla.make_shape(rows_ub, 128), tla.make_stride(q_stride, 1)
                            ),
                        )
                        ogm_valid = tla.make_tensor(
                            out.ptr + qbase + vi * 64 * q_stride,
                            tla.make_layout(
                                tla.make_shape(rows_ub, 128), tla.make_stride(q_stride, 1)
                            ),
                        )
                        lgm_valid = tla.make_tensor(
                            lse.ptr + lbase + vi * 64,
                            tla.make_layout(tla.make_shape(rows_ub), tla.make_stride(1)),
                        )
                        tla.copy(
                            tla.tile_view(du, tla.make_shape(rows_ub, 128), tla.make_coord(0, 0)),
                            dgm_valid,
                        )
                        tla.copy(
                            tla.tile_view(ou, tla.make_shape(rows_ub, 128), tla.make_coord(0, 0)),
                            ogm_valid,
                        )
                        tla.copy(
                            tla.tile_view(ls, tla.make_shape(rows_ub), tla.make_coord(0)), lgm_valid
                        )
                else:
                    tla.copy(du, dgm)
                    tla.copy(ou, ogm)
                    tla.copy(ls, lgm)
                if tla.const_expr(len(lse_gradient) > 0):
                    # Reuse delta as the DMA destination; retain dLSE in a
                    # vector register while the ordinary row reductions run.
                    tangent_rows = min(64, max(0, valid_q - vi * 64))
                    if tangent_rows > 0:
                        tangent_gm = tla.make_tensor(
                            lse_gradient[0].ptr + lbase + vi * 64 * l_stride,
                            tla.make_layout(
                                tla.make_shape(tangent_rows), tla.make_stride(l_stride)
                            ),
                        )
                        tla.copy(
                            tla.tile_view(dd, tla.make_shape(tangent_rows), tla.make_coord(0)),
                            tangent_gm,
                        )
                tla.set_flag(vec_loaded)
                tla.wait_flag(vec_loaded)
                with tla.vec.func(mode="simd"):
                    f32mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    b16mask = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype)
                    one = tla.update_mask(1, dtype=tla.Float32)[0]
                    if tla.const_expr(len(lse_gradient) > 0):
                        row_tangents = dd.load()
                    for row in tla.range(64):
                        dr = tla.tile_view(
                            du, tla.make_shape(1, 128), tla.make_coord(row, 0)
                        ).load()
                        ov = tla.tile_view(
                            ou, tla.make_shape(1, 128), tla.make_coord(row, 0)
                        ).load()
                        d0 = dr.to(
                            tla.Float32,
                            tla.params.CastParams(reg_slot=tla.params.RegSlot.ZERO),
                            mask=b16mask,
                        )
                        d1 = dr.to(
                            tla.Float32,
                            tla.params.CastParams(reg_slot=tla.params.RegSlot.ONE),
                            mask=b16mask,
                        )
                        o0 = ov.to(
                            tla.Float32,
                            tla.params.CastParams(reg_slot=tla.params.RegSlot.ZERO),
                            mask=b16mask,
                        )
                        o1 = ov.to(
                            tla.Float32,
                            tla.params.CastParams(reg_slot=tla.params.RegSlot.ONE),
                            mask=b16mask,
                        )
                        r0 = tla.mul(d0, o0, mask=f32mask).reduce(tla.ReductionOp.ADD, mask=f32mask)
                        r1 = tla.mul(d1, o1, mask=f32mask).reduce(tla.ReductionOp.ADD, mask=f32mask)
                        value = tla.add(
                            tla.full(r0, dtype=tla.Float32),
                            tla.full(r1, dtype=tla.Float32),
                        )
                        tla.tile_view(dd, tla.make_shape(1), tla.make_coord(row)).store(
                            value, tla.params.UnalignStoreParams(), mask=one
                        )
                    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                    if tla.const_expr(len(lse_gradient) > 0):
                        dd.store(tla.sub(dd.load(), row_tangents))
                        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)

        steps = count
        if count == 0:
            steps = tla.as_numeric(1)
        drain = tla.as_numeric(0)
        if task + tla.arch.block_num() >= batch * q_heads * q_blocks:
            drain = tla.as_numeric(1)
        for selected in tla.range(steps + drain):
            stage1_kind = stage0_kind
            stage1_kb = stage0_kb
            stage1_last = stage0_last
            stage0_kind = tla.as_numeric(0)
            stage0_kb = tla.as_numeric(0)
            stage0_last = tla.as_numeric(0)
            if selected < steps:
                if selected == steps - 1:
                    stage0_last = tla.as_numeric(1)
                if count == 0:
                    stage0_kind = tla.as_numeric(2)
            slot = tick % 2
            if selected < count:
                stage0_kind = tla.as_numeric(1)
                full = selected >= pcount
                if tla.const_expr(dense_mask and len(implicit_window) == 0):
                    full = tla.Bool(False)
                kb = selected
                if tla.const_expr(len(implicit_window) > 0):
                    kb = window_start + selected
                    full = (kb * 128 >= window_full_lo) & (
                        min(kb * 128 + 128, klen) <= window_full_hi
                    )
                if tla.const_expr(not dense):
                    if full:
                        kb = full_index[mask_row, selected - pcount]
                    else:
                        kb = partial_index[mask_row, selected]
                stage0_kb = kb
                tile_base = kbase + kb * 128 * k_stride
                valid_k = min(128, klen - kb * 128)
                tail_tile = (valid_q < 128) | (valid_k < 128)
                pipeline_load_rows(
                    k,
                    tile_base,
                    k_stride,
                    valid_k,
                    lk,
                    False,
                    0,
                    k_free,
                    k_free,
                    k_ready,
                    k_ready,
                    ragged,
                    tail_scratch,
                    tail_zero,
                    tail_loaded,
                    tail_stored,
                    tail_free,
                    tail_ready,
                )
                pipeline_load_rows(
                    v,
                    tile_base,
                    k_stride,
                    valid_k,
                    lv,
                    False,
                    0,
                    v_free,
                    v_free,
                    v_ready,
                    v_ready,
                    ragged,
                    tail_scratch,
                    tail_zero,
                    tail_loaded,
                    tail_stored,
                    tail_free,
                    tail_ready,
                    False,
                )
                with tla.cube():
                    ss = _mm(
                        l1_matrix(lk, False),
                        l1_matrix(lq, True),
                        a0,
                        b0,
                        c0,
                        ab0_free,
                        ab0_ready,
                        c0_free,
                        c0_ready,
                    )
                    tla.set_flag(k_free)
                    if slot == 0:
                        _cv_wait(cv_free0, tla.arch.FIX)
                    else:
                        _cv_wait(cv_free1, tla.arch.FIX)
                    su = tla.make_tensor(
                        scores + slot * 8192,
                        tla.make_layout(tla.make_shape(128, 128), tla.make_stride(64, 1)),
                    )
                    tla.copy(
                        su,
                        ss,
                        tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_N),
                    )
                    tla.set_flag(c0_free)
                    tla.wait_flag(v_ready)
                    dp = _mm(
                        l1_matrix(lv, False),
                        l1_matrix(ld, True),
                        a1,
                        b1,
                        c1,
                        ab1_free,
                        ab1_ready,
                        c1_free,
                        c1_ready,
                    )
                    tla.set_flag(v_free)
                    dpu = tla.make_tensor(
                        dps + slot * 8192,
                        tla.make_layout(tla.make_shape(128, 128), tla.make_stride(64, 1)),
                    )
                    tla.copy(
                        dpu,
                        dp,
                        tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_N),
                    )
                    tla.set_flag(c1_free)
                    if slot == 0:
                        _cv_set(cv_ready0, tla.arch.FIX)
                    else:
                        _cv_set(cv_ready1, tla.arch.FIX)
                with tla.vector():
                    vi2 = tla.arch.sub_block_idx()
                    # Tail staging has finished. Borrow dQ cast scratch until PDS
                    # consumes these predicates, before consumer staging/output.
                    visibility = tla.make_tensor(
                        tla.recast_ptr(dq_cast, dtype=tla.Int32),
                        tla.make_layout(tla.make_shape(128, 8), tla.make_stride(8, 1)),
                    )
                    if tla.const_expr(indexed and not generic):
                        if not full:
                            if tla.const_expr(ragged):
                                if tail_tile:
                                    _compact_mask(
                                        mask_mod,
                                        batch_idx,
                                        head,
                                        qstart,
                                        vi2,
                                        kb,
                                        visibility,
                                        aux_tensors,
                                        aux_scalars,
                                        info,
                                        False,
                                    )
                                else:
                                    _compact_mask(
                                        mask_mod,
                                        batch_idx,
                                        head,
                                        qstart,
                                        vi2,
                                        kb,
                                        visibility,
                                        aux_tensors,
                                        aux_scalars,
                                        info,
                                        True,
                                    )
                            else:
                                _compact_mask(
                                    mask_mod,
                                    batch_idx,
                                    head,
                                    qstart,
                                    vi2,
                                    kb,
                                    visibility,
                                    aux_tensors,
                                    aux_scalars,
                                    info,
                                    True,
                                )
                            tla.pipe_barrier(tla.pipes.ALL)
                    # Independent mask generation overlaps QK/dP Cube work.
                    if slot == 0:
                        _v_wait(cv_ready0, tla.arch.VECTOR)
                    else:
                        _v_wait(cv_ready1, tla.arch.VECTOR)
                    s_ub = tla.make_tensor(
                        scores + slot * 8192,
                        tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
                    )
                    dp_ub = tla.make_tensor(
                        dps + slot * 8192,
                        tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
                    )
                    p_ub = tla.make_tensor(
                        probs,
                        tla.make_layout(tla.make_shape(128, 64), tla.make_stride(16, 1)),
                    )
                    ds_ub = tla.make_tensor(
                        derivs,
                        tla.make_layout(tla.make_shape(128, 64), tla.make_stride(16, 1)),
                    )
                    d_ub = tla.make_tensor(
                        delta, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                    )
                    l_ub = tla.make_tensor(
                        logsum, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                    )
                    if tla.const_expr(generic and indexed):
                        prepare_pds_simt(
                            s_ub,
                            dp_ub,
                            l_ub,
                            d_ub,
                            scale,
                            batch_idx,
                            head,
                            qstart + vi2 * 64,
                            kb * 128,
                            full,
                            score_mod,
                            mask_mod,
                            score_grad,
                            aux_tensors,
                            aux_scalars,
                            info,
                        )
                        tla.pipe_barrier(tla.pipes.ALL)
                    with tla.vec.func(mode="simd"):
                        neg_inf = tla.full(float("-inf"), dtype=tla.Float32)
                        zero = tla.full(0.0, dtype=tla.Float32)
                        bv = tla.add(tla.full(0, dtype=tla.Int32), batch_idx)
                        hv = tla.add(tla.full(0, dtype=tla.Int32), head)
                        qidx = tla.arange(qstart + vi2 * 64, dtype=tla.Int32)
                        ls2 = l_ub.load()
                        d2 = d_ub.load()
                        empty = tla.cmp(ls2, neg_inf, mode="eq")
                        safe_lse = tla.where(empty, zero, ls2)
                        if full:
                            _pds_pass(
                                s_ub,
                                dp_ub,
                                p_ub,
                                ds_ub,
                                scale,
                                neg_inf,
                                zero,
                                bv,
                                hv,
                                qidx,
                                kb,
                                safe_lse,
                                empty,
                                d2,
                                aux_tensors,
                                aux_scalars,
                                mask_mod,
                                False,
                                info,
                                tail_tile,
                                score_mod,
                                score_grad,
                                generic,
                                generic and indexed,
                            )
                        else:
                            _pds_pass(
                                s_ub,
                                dp_ub,
                                p_ub,
                                ds_ub,
                                1.0 if generic and indexed else scale,
                                neg_inf,
                                zero,
                                bv,
                                hv,
                                qidx,
                                kb,
                                safe_lse,
                                empty,
                                d2,
                                aux_tensors,
                                aux_scalars,
                                mask_mod,
                                not indexed,
                                info,
                                tail_tile,
                                score_mod,
                                score_grad,
                                generic,
                                generic and indexed,
                                visibility,
                                indexed and not generic,
                            )
                    if slot == 0:
                        _v_set(cv_free0, tla.arch.VECTOR)
                        _v_wait(pds_free0, tla.arch.MTE3)
                    else:
                        _v_set(cv_free1, tla.arch.VECTOR)
                        _v_wait(pds_free1, tla.arch.MTE3)
                    tla.set_flag(vec_stored)
                    tla.wait_flag(vec_stored)
                    _publish_dn(probs, lp + slot * 16384)
                    _publish_dn(derivs, lds + slot * 16384)
                    # One UB P/dS pair is reused by the next producer tile.
                    tla.pipe_barrier(tla.pipes.ALL)
                    if slot == 0:
                        _v_set(pds_ready0, tla.arch.MTE3)
                    else:
                        _v_set(pds_ready1, tla.arch.MTE3)

            if stage1_kind != 0:
                if consumer_index == 0:
                    # Reconstruct consumer metadata only at this query's FIRST.
                    consumer_task = tla.arch.block_idx() + consumer_serial * tla.arch.block_num()
                    consumer_qb = consumer_task % q_blocks
                    consumer_head = consumer_task // q_blocks % q_heads
                    consumer_batch = consumer_task // (q_blocks * q_heads)
                    consumer_kh = consumer_head // (q_heads // kv_heads)
                    consumer_qbase = (
                        consumer_batch * q_heads * max_q * 128
                        + consumer_head * q_head_stride
                        + consumer_qb * 128 * q_stride
                    )
                    consumer_kbase = (
                        consumer_batch * kv_heads * max_k * 128 + consumer_kh * k_head_stride
                    )
                    consumer_qlen = max_q
                    consumer_klen = max_k
                    if tla.const_expr(ragged):
                        if tla.const_expr(varlen[0]):
                            consumer_qs = cu_q[consumer_batch]
                            consumer_qlen = cu_q[consumer_batch + 1] - consumer_qs
                            consumer_qbase = (
                                consumer_qs * q_stride
                                + consumer_head * 128
                                + consumer_qb * 128 * q_stride
                            )
                        if tla.const_expr(varlen[1]):
                            consumer_ks = cu_k[consumer_batch]
                            consumer_klen = cu_k[consumer_batch + 1] - consumer_ks
                            consumer_kbase = consumer_ks * k_stride + consumer_kh * 128
                    consumer_qslot = consumer_serial % 2
                    consumer_lq = lq0
                    consumer_ld = ld0
                    consumer_cq = cq0
                    if consumer_qslot == 1:
                        consumer_lq = lq1
                        consumer_ld = ld1
                        consumer_cq = cq1
                if stage1_kind == 1:
                    previous = consumer_index
                    consume_slot = (tick - 1) % 2
                    previous_base = consumer_kbase + stage1_kb * 128 * k_stride
                    with tla.cube():
                        if previous == 0:
                            if consumer_qslot == 0:
                                tla.wait_flag(cq_free0)
                            else:
                                tla.wait_flag(cq_free1)
                        if consume_slot == 0:
                            _cv_wait(pds_ready0, tla.arch.MTE1)
                        else:
                            _cv_wait(pds_ready1, tla.arch.MTE1)
                    consumer_valid_k = min(128, consumer_klen - stage1_kb * 128)
                    pipeline_load_rows(
                        k,
                        previous_base,
                        k_stride,
                        consumer_valid_k,
                        lk,
                        False,
                        0,
                        k_free,
                        k_free,
                        k_ready,
                        k_ready,
                        ragged,
                        tail_scratch,
                        tail_zero,
                        tail_loaded,
                        tail_stored,
                        tail_free,
                        tail_ready,
                    )
                    with tla.cube():
                        dsa = tla.make_tensor_like(
                            a0, l1_matrix(lds + consume_slot * 16384, True), tla.arch.zN
                        )
                        kb0 = tla.make_tensor_like(b0, l1_matrix(lk, False), tla.arch.nZ)
                        dqref = tla.make_tensor(
                            consumer_cq,
                            tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
                        )
                        dqsum = tla.make_tensor_like(consumer_cq, dqref, tla.arch.L0Clayout)
                        tla.wait_flag(ab0_free)
                        tla.copy(dsa, l1_matrix(lds + consume_slot * 16384, True))
                        tla.copy(kb0, l1_matrix(lk, False))
                        tla.set_flag(k_free)
                        tla.set_flag(ab0_ready)
                        tla.wait_flag(ab0_ready)
                        tla.mmad(dqsum, dsa, kb0, init_c=previous == 0)
                        tla.set_flag(ab0_free)
                        dkt = _mm(
                            l1_matrix(lds + consume_slot * 16384, False),
                            l1_matrix(consumer_lq, False),
                            a1,
                            b1,
                            c0,
                            ab1_free,
                            ab1_ready,
                            c0_free,
                            c0_ready,
                        )
                        dkgm = tla.make_tensor(
                            dk_acc.ptr + previous_base,
                            tla.make_layout(
                                tla.make_shape(consumer_valid_k, 128), tla.make_stride(k_stride, 1)
                            ),
                        )
                        tla.copy(
                            dkgm,
                            dkt,
                            tla.params.CopyL0C2DstParams(atomic_mode=tla.params.AtomicMode.ADD),
                        )
                        tla.set_flag(c0_free)
                        dvt = _mm(
                            l1_matrix(lp + consume_slot * 16384, False),
                            l1_matrix(consumer_ld, False),
                            a0,
                            b0,
                            c1,
                            ab0_free,
                            ab0_ready,
                            c1_free,
                            c1_ready,
                        )
                        dvgm = tla.make_tensor(
                            dv_acc.ptr + previous_base,
                            tla.make_layout(
                                tla.make_shape(consumer_valid_k, 128), tla.make_stride(k_stride, 1)
                            ),
                        )
                        tla.copy(
                            dvgm,
                            dvt,
                            tla.params.CopyL0C2DstParams(atomic_mode=tla.params.AtomicMode.ADD),
                        )
                        tla.set_flag(c1_free)
                        if consume_slot == 0:
                            _cv_set(pds_free0, tla.arch.MTE1)
                        else:
                            _cv_set(pds_free1, tla.arch.MTE1)

                        if stage1_last != 0:
                            # Final dV has copied dO; dK copied Q before it.
                            if consumer_qslot == 0:
                                tla.set_flag(qdo_free0)
                                tla.set_flag(cq_ready0)
                            else:
                                tla.set_flag(qdo_free1)
                                tla.set_flag(cq_ready1)
                if stage1_last != 0:
                    if consumer_qb * 128 < consumer_qlen:
                        with tla.cube():
                            # Both AIVs must finish prior output DMA before FIX
                            # can overwrite the dedicated dQ staging buffer.
                            _cv_wait(output_free, tla.arch.FIX)
                            if stage1_kind == 1:
                                if consumer_qslot == 0:
                                    tla.wait_flag(cq_ready0)
                                else:
                                    tla.wait_flag(cq_ready1)
                                qref = tla.make_tensor(
                                    consumer_cq,
                                    tla.make_layout(
                                        tla.make_shape(128, 128), tla.make_stride(128, 1)
                                    ),
                                )
                                qsum = tla.make_tensor_like(consumer_cq, qref, tla.arch.L0Clayout)
                                qub = tla.make_tensor(
                                    dq_stage,
                                    tla.make_layout(
                                        tla.make_shape(128, 128), tla.make_stride(128, 1)
                                    ),
                                )
                                tla.copy(
                                    qub,
                                    qsum,
                                    tla.params.CopyL0C2DstParams(
                                        l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                                    ),
                                )
                                if consumer_qslot == 0:
                                    tla.set_flag(cq_free0)
                                else:
                                    tla.set_flag(cq_free1)
                            _cv_set(output_ready, tla.arch.FIX)
                        with tla.vector():
                            _v_wait(output_ready, tla.arch.VECTOR)
                            if stage1_kind == 2:
                                zub = tla.make_tensor(
                                    dq_stage,
                                    tla.make_layout(tla.make_shape(8192), tla.make_stride(1)),
                                )
                                with tla.vec.func(mode="simd"):
                                    for zv in tla.range(128):
                                        tla.tile_view(
                                            zub, tla.make_shape(64), tla.make_coord(zv)
                                        ).store(tla.full(0.0, dtype=tla.Float32))
                                tla.pipe_barrier(tla.pipes.ALL)
                            # Keep the shared helper and its output DMA barrier.
                            output_base = consumer_qbase
                            store_output(
                                dq,
                                output_base,
                                q_stride,
                                min(128, max(0, consumer_qlen - consumer_qb * 128)),
                                dq_stage,
                                dq_cast,
                                scale,
                                vec_stored,
                            )
                            _v_set(output_free, tla.arch.MTE3)
                    consumer_serial = consumer_serial + 1
                    consumer_index = tla.as_numeric(0)
                else:
                    consumer_index = consumer_index + 1
            tick = tick + 1
        query_serial = query_serial + 1

    # Final physical-worker drain: consume each initialized free once.
    with tla.cube():
        tla.wait_flag(ab0_free)
        tla.wait_flag(ab1_free)
        tla.wait_flag(c0_free)
        tla.wait_flag(c1_free)
        tla.wait_flag(k_free)
        tla.wait_flag(v_free)
        tla.wait_flag(qdo_free0)
        tla.wait_flag(qdo_free1)
        tla.wait_flag(cq_free0)
        tla.wait_flag(cq_free1)
        _cv_wait(cv_free0, tla.arch.FIX)
        _cv_wait(cv_free1, tla.arch.FIX)
        _cv_wait(output_free, tla.arch.FIX)
    with tla.vector():
        _v_wait(pds_free0, tla.arch.MTE3)
        _v_wait(pds_free1, tla.arch.MTE3)

    if tla.const_expr(len(workspace) > 0):
        # All cores join before final dK scaling and dK/dV conversion; reuse UB.
        ws_finish_dk, ws_finish_dv, ws_finish_size = workspace
        # Caller has drained attention's buffer handshakes. FIX includes every
        # atomic add; the all-AIV join extends that completion to all AICs.
        with tla.cube():
            tla.pipe_barrier(tla.pipes.ALL)
            tla.cross_core_set_flag(workspace_handoff, tla.arch.FIX, aiv_id=0)
            tla.cross_core_set_flag(workspace_handoff, tla.arch.FIX, aiv_id=1)
        with tla.vector():
            tla.pipe_barrier(tla.pipes.ALL)
            if tla.arch.sub_block_idx() == 0:
                tla.cross_core_wait_flag(workspace_handoff, tla.arch.MTE2, aiv_id=0)
            else:
                tla.cross_core_wait_flag(workspace_handoff, tla.arch.MTE2, aiv_id=1)
            tla.cross_core_set_flag(workspace_all, tla.arch.MTE2)
            tla.cross_core_wait_flag(workspace_all, tla.arch.MTE2)
            ws_finish_layout_ub = tla.make_layout(tla.make_shape(4096), tla.make_stride(1))
            ws_finish_ku = tla.make_tensor(scores, ws_finish_layout_ub)
            ws_finish_vu = tla.make_tensor(dps, ws_finish_layout_ub)
            ws_finish_ko = tla.make_tensor(probs, ws_finish_layout_ub)
            ws_finish_vo = tla.make_tensor(derivs, ws_finish_layout_ub)
            ws_finish_worker = tla.arch.block_idx() * 2 + tla.arch.sub_block_idx()
            for ws_finish_base in tla.range(
                ws_finish_worker * 4096, ws_finish_size, tla.arch.block_num() * 2 * 4096
            ):
                ws_finish_count = (
                    4096
                    if ws_finish_size - ws_finish_base > 4096
                    else ws_finish_size - ws_finish_base
                )
                ws_finish_layout = tla.make_layout(
                    tla.make_shape(ws_finish_count), tla.make_stride(1)
                )
                ws_finish_kut = tla.tile_view(
                    ws_finish_ku, tla.make_shape(ws_finish_count), tla.make_coord(0)
                )
                ws_finish_vut = tla.tile_view(
                    ws_finish_vu, tla.make_shape(ws_finish_count), tla.make_coord(0)
                )
                ws_finish_kot = tla.tile_view(
                    ws_finish_ko, tla.make_shape(ws_finish_count), tla.make_coord(0)
                )
                ws_finish_vot = tla.tile_view(
                    ws_finish_vo, tla.make_shape(ws_finish_count), tla.make_coord(0)
                )
                tla.copy(
                    ws_finish_kut,
                    tla.make_tensor(dk_acc.ptr + ws_finish_base, ws_finish_layout),
                )
                tla.copy(
                    ws_finish_vut,
                    tla.make_tensor(dv_acc.ptr + ws_finish_base, ws_finish_layout),
                )
                tla.pipe_barrier(tla.pipes.ALL)
                with tla.vec.func(mode="simd"):
                    ws_finish_mask = tla.create_mask(
                        pattern=tla.mask.ALL, dtype=ws_finish_dk.ptr.dtype
                    )
                    for ws_finish_row in tla.range(ws_finish_count // 64):
                        ws_finish_coord = tla.make_coord(ws_finish_row)
                        ws_finish_kval = tla.mul(
                            tla.tile_view(ws_finish_ku, tla.make_shape(64), ws_finish_coord).load(),
                            scale,
                        )
                        ws_finish_vval = tla.tile_view(
                            ws_finish_vu, tla.make_shape(64), ws_finish_coord
                        ).load()
                        ws_finish_kc = ws_finish_kval.to(
                            ws_finish_dk.ptr.dtype,
                            tla.params.CastParams(
                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                reg_slot=tla.params.RegSlot.ZERO,
                            ),
                        )
                        ws_finish_vc = ws_finish_vval.to(
                            ws_finish_dv.ptr.dtype,
                            tla.params.CastParams(
                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                reg_slot=tla.params.RegSlot.ZERO,
                            ),
                        )
                        tla.tile_view(ws_finish_ko, tla.make_shape(64), ws_finish_coord).store(
                            ws_finish_kc,
                            tla.params.NormalStoreParams(
                                store_dist=tla.params.StoreDist.DIST_PACK_B32
                            ),
                            mask=ws_finish_mask,
                        )
                        tla.tile_view(ws_finish_vo, tla.make_shape(64), ws_finish_coord).store(
                            ws_finish_vc,
                            tla.params.NormalStoreParams(
                                store_dist=tla.params.StoreDist.DIST_PACK_B32
                            ),
                            mask=ws_finish_mask,
                        )
                tla.pipe_barrier(tla.pipes.ALL)
                tla.copy(
                    tla.make_tensor(ws_finish_dk.ptr + ws_finish_base, ws_finish_layout),
                    ws_finish_kot,
                )
                tla.copy(
                    tla.make_tensor(ws_finish_dv.ptr + ws_finish_base, ws_finish_layout),
                    ws_finish_vot,
                )
                tla.pipe_barrier(tla.pipes.ALL)
