"""Explicitly synchronized on-chip tile operations for Ascend 950.

All matrix tiles are physically 128x128. AIV0/1 each stage 64 rows,
including zero padding, so odd sequence tails never split a fractal row.
"""

import catlass.tla as tla


@tla.jit
def vector_signal(flag, pipe: tla.Constexpr = tla.arch.MTE3):
    if tla.arch.sub_block_idx() == 0:
        tla.cross_core_set_flag(flag, pipe, aiv_id=0)
    else:
        tla.cross_core_set_flag(flag, pipe, aiv_id=1)


@tla.jit
def vector_wait(flag, pipe: tla.Constexpr = tla.arch.VECTOR):
    if tla.arch.sub_block_idx() == 0:
        tla.cross_core_wait_flag(flag, pipe, aiv_id=0)
    else:
        tla.cross_core_wait_flag(flag, pipe, aiv_id=1)


@tla.jit
def cube_wait(flag, pipe: tla.Constexpr = tla.arch.MTE1):
    tla.cross_core_wait_flag(flag, pipe, aiv_id=0)
    tla.cross_core_wait_flag(flag, pipe, aiv_id=1)


@tla.jit
def cube_signal(flag, pipe: tla.Constexpr = tla.arch.FIX):
    tla.cross_core_set_flag(flag, pipe, aiv_id=0)
    tla.cross_core_set_flag(flag, pipe, aiv_id=1)


@tla.jit
def stage_rows(mem, base, stride, valid_rows, scratch_ptr, l1_ptr, zero_ready, loaded, stored):
    """GM -> padded UB half -> zN L1; called by both AIVs."""
    vi = tla.arch.sub_block_idx()
    dtype = mem.ptr.dtype
    ub = tla.make_tensor(
        scratch_ptr, tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1))
    )
    with tla.vec.func(mode="simd"):
        for r in tla.range(64):
            tla.tile_view(ub, tla.make_shape(1, 128), tla.make_coord(r, 0)).store(
                tla.full(0.0, dtype=dtype)
            )
    tla.set_flag(zero_ready)
    tla.wait_flag(zero_ready)
    count = (
        64 if valid_rows - vi * 64 > 64 else (valid_rows - vi * 64 if valid_rows > vi * 64 else 0)
    )
    if count > 0:
        gm = tla.make_tensor(
            mem.ptr + base + vi * 64 * stride,
            tla.make_layout(tla.make_shape(count, 128), tla.make_stride(stride, 1)),
        )
        ub_valid = tla.tile_view(ub, tla.make_shape(count, 128), tla.make_coord(0, 0))
        tla.copy(ub_valid, gm)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    tla.set_flag(stored)
    tla.wait_flag(stored)
    ref = tla.make_tensor(
        l1_ptr, tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1))
    )
    l1 = tla.make_tensor_like(l1_ptr, ref, tla.arch.zN)
    half = tla.tile_view(l1, tla.make_shape(64, 128), tla.make_coord(vi, 0))
    tla.copy(half, ub)
    # Scratch is reused by the next staging call only after MTE3 completes.
    tla.pipe_barrier(tla.pipes.ALL)


@tla.jit
def l1_matrix(ptr, transpose: tla.Constexpr):
    if tla.const_expr(transpose):
        ref = tla.make_tensor(
            ptr,
            tla.make_layout(
                tla.make_shape(128, 128),
                tla.make_stride(1, 128),
                layoutTag=tla.arch.ColumnMajor,
            ),
        )
        result = tla.make_tensor_like(ptr, ref, tla.arch.nZ)
    else:
        ref = tla.make_tensor(
            ptr, tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1))
        )
        result = tla.make_tensor_like(ptr, ref, tla.arch.zN)
    return result


@tla.jit
def store_output(mem, base, stride, rows, acc_ptr, out_ptr, scale, stored):
    vi = tla.arch.sub_block_idx()
    dtype = mem.ptr.dtype
    acc = tla.make_tensor(acc_ptr, tla.make_layout(tla.make_shape(8192), tla.make_stride(1)))
    out = tla.make_tensor(out_ptr, tla.make_layout(tla.make_shape(8192), tla.make_stride(1)))
    with tla.vec.func(mode="simd"):
        for i in tla.range(128):
            src = tla.tile_view(acc, tla.make_shape(64), tla.make_coord(i))
            dst = tla.tile_view(out, tla.make_shape(64), tla.make_coord(i))
            value = tla.mul(src.load(), scale)
            converted = value.to(
                dtype,
                tla.params.CastParams(
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                    reg_slot=tla.params.RegSlot.ZERO,
                ),
            )
            dst.store(
                converted,
                tla.params.NormalStoreParams(store_dist=tla.params.StoreDist.DIST_PACK_B32),
                mask=tla.create_mask(pattern=tla.mask.ALL, dtype=dtype),
            )
    tla.set_flag(stored)
    tla.wait_flag(stored)
    count = 64 if rows - vi * 64 > 64 else (rows - vi * 64 if rows > vi * 64 else 0)
    if count > 0:
        gm = tla.make_tensor(
            mem.ptr + base + vi * 64 * stride,
            tla.make_layout(tla.make_shape(count, 128), tla.make_stride(stride, 1)),
        )
        ub = tla.make_tensor(
            out_ptr,
            tla.make_layout(tla.make_shape(count, 128), tla.make_stride(128, 1)),
        )
        tla.copy(gm, ub)
    tla.pipe_barrier(tla.pipes.ALL)


@tla.jit
def pipeline_load_rows(
    mem,
    base,
    stride,
    rows,
    l1_ptr,
    transpose: tla.Constexpr,
    slot,
    free0,
    free1,
    ready0,
    ready1,
    ragged: tla.Constexpr,
    scratch,
    zero_ready,
    loaded,
    stored,
    tail_free,
    tail_ready,
    wait_ready: tla.Constexpr = True,
    scratch_free=(),
):
    """Direct full-tile DMA; safe zero-filled boundary transfer in the same kernel."""
    with tla.cube():
        if slot == 0:
            tla.wait_flag(free0)
        else:
            tla.wait_flag(free1)
        if tla.const_expr(ragged):
            if rows < 128:
                tla.cross_core_set_flag(tail_free, tla.arch.MTE2, aiv_id=0)
                tla.cross_core_set_flag(tail_free, tla.arch.MTE2, aiv_id=1)
    if tla.const_expr(ragged):
        with tla.vector():
            if rows < 128:
                vector_wait(tail_free)
                if tla.const_expr(len(scratch_free) > 0):
                    tla.wait_flag(scratch_free[0])
                stage_rows(mem, base, stride, rows, scratch, l1_ptr, zero_ready, loaded, stored)
                if tla.const_expr(len(scratch_free) > 0):
                    tla.set_flag(scratch_free[0])
                vector_signal(tail_ready)
    with tla.cube():
        if tla.const_expr(ragged):
            if rows < 128:
                cube_wait(tail_ready)
        if rows == 128:
            if tla.const_expr(transpose):
                gm = tla.make_tensor(
                    mem.ptr + base,
                    tla.make_layout(
                        tla.make_shape(128, 128),
                        tla.make_stride(1, stride),
                        layoutTag=tla.arch.ColumnMajor,
                    ),
                )
            else:
                gm = tla.make_tensor(
                    mem.ptr + base,
                    tla.make_layout(tla.make_shape(128, 128), tla.make_stride(stride, 1)),
                )
            tla.copy(l1_matrix(l1_ptr, transpose), gm)
        if slot == 0:
            tla.set_flag(ready0)
            if tla.const_expr(wait_ready):
                tla.wait_flag(ready0)
        else:
            tla.set_flag(ready1)
            if tla.const_expr(wait_ready):
                tla.wait_flag(ready1)
