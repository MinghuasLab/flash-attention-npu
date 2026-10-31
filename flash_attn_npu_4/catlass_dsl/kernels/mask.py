"""Exact or explicitly requested five-point block classification."""

import catlass.tla as tla


@tla.kernel
def mask_kernel(
    pc: tla.Tensor,
    pi: tla.Tensor,
    fc: tla.Tensor,
    fi: tla.Tensor,
    cu_q: tla.Tensor,
    cu_k: tla.Tensor,
    aux_tensors: tuple,
    aux_scalars: tuple,
    batches: tla.Int32,
    heads: tla.Int32,
    max_q: tla.Int32,
    max_k: tla.Int32,
    varlen: tla.Constexpr,
    mask_mod: tla.Constexpr,
    use_fast_sampling: tla.Constexpr,
):
    count_ptr = tla.allocate(2048, tla.Int32, tla.AddressSpace.ub, 256)
    kind_ptr = tla.allocate(512, tla.Int32, tla.AddressSpace.ub, 256)
    nq = (max_q + 127) // 128
    nk = (max_k + 127) // 128
    groups = (nk + 127) // 128
    classification_done = tla.cross_flag("classification_done", mode=0)
    with tla.vector():
        counts = tla.make_tensor(
            count_ptr, tla.make_layout(tla.make_shape(2048), tla.make_stride(1))
        )
        kinds = tla.make_tensor(kind_ptr, tla.make_layout(tla.make_shape(512), tla.make_stride(1)))
        for task in tla.range(
            tla.arch.block_idx(), batches * heads * nq * groups, tla.arch.block_num()
        ):
            row = task // groups
            group_base = (task % groups) * 128
            qb = row % nq
            h = row // nq % heads
            b = row // (nq * heads)
            qlen = max_q
            klen = max_k
            if tla.const_expr(varlen[0]):
                qlen = cu_q[b + 1] - cu_q[b]
            if tla.const_expr(varlen[1]):
                klen = cu_k[b + 1] - cu_k[b]
            qs = cu_q[b] if tla.const_expr(varlen[0]) else tla.Int32(0)
            ks = cu_k[b] if tla.const_expr(varlen[1]) else tla.Int32(0)
            info = (qlen, klen, qs, ks)
            m = 128 if qlen - qb * 128 > 128 else (qlen - qb * 128 if qlen > qb * 128 else 0)
            if tla.const_expr(use_fast_sampling):
                # One lane per tile; sample only valid coordinates, including
                # the center of the actual tail extent.
                with tla.vec.func(mode="simt", thread_block_dim=128):
                    sample_lane, _, _ = tla.arch.thread_idx()
                    sample_kb = group_base + sample_lane
                    sample_n = (
                        128
                        if klen - sample_kb * 128 > 128
                        else (klen - sample_kb * 128 if klen > sample_kb * 128 else 0)
                    )
                    sample_kind = tla.Int32(0)
                    if m > 0 and sample_n > 0:
                        sample_visible = tla.Int32(0)
                        for point in tla.range_constexpr(5):
                            sample_q = qb * 128
                            sample_k = sample_kb * 128
                            if tla.const_expr(point == 2 or point == 3):
                                sample_q = sample_q + m - 1
                            elif tla.const_expr(point == 4):
                                sample_q = sample_q + m // 2
                            if tla.const_expr(point == 1 or point == 3):
                                sample_k = sample_k + sample_n - 1
                            elif tla.const_expr(point == 4):
                                sample_k = sample_k + sample_n // 2
                            sample_keep = mask_mod(
                                b, h, sample_q, sample_k, info, aux_tensors, aux_scalars
                            )
                            if sample_keep:
                                sample_visible = sample_visible + 1
                        if sample_visible == 5:
                            sample_kind = tla.Int32(2)
                        elif sample_visible > 0:
                            sample_kind = tla.Int32(1)
                    if sample_kb < nk:
                        pi[row, sample_kb] = sample_kind
            else:
                if m == 128 and group_base + 128 <= klen // 128:
                    with tla.vec.func(mode="simt", thread_block_dim=2048):
                        full_tid, _, _ = tla.arch.thread_idx()
                        full_col = full_tid % 16
                        full_group = full_tid // 16
                        full_kb = group_base + full_group
                        full_visible = tla.Int32(0)
                        for full_qbase in tla.range(0, 128, 2):
                            for full_qoff in tla.range_constexpr(2):
                                for full_koff in tla.range_constexpr(8):
                                    full_keep = mask_mod(
                                        b,
                                        h,
                                        qb * 128 + full_qbase + full_qoff,
                                        full_kb * 128 + full_col + full_koff * 16,
                                        info,
                                        aux_tensors,
                                        aux_scalars,
                                    )
                                    if full_keep:
                                        full_visible = full_visible + 1
                        counts[full_tid] = full_visible
                        tla.arch.sync_threads()
                        for full_level in tla.range_constexpr(4):
                            full_offset = 8 >> full_level
                            if full_col < full_offset:
                                counts[full_tid] = counts[full_tid] + counts[full_tid + full_offset]
                            tla.arch.sync_threads()
                        if full_col == 0:
                            full_nnz = counts[full_tid]
                            full_kind = tla.Int32(0)
                            if full_nnz == 16384:
                                full_kind = tla.Int32(2)
                            elif full_nnz > 0:
                                full_kind = tla.Int32(1)
                            pi[row, full_kb] = full_kind
                else:
                    with tla.vec.func(mode="simt", thread_block_dim=2048):
                        tid, _, _ = tla.arch.thread_idx()
                        col = tid % 16
                        group = tid // 16
                        kb = group_base + group
                        n = (
                            128
                            if klen - kb * 128 > 128
                            else (klen - kb * 128 if klen > kb * 128 else 0)
                        )
                        visible = tla.Int32(0)
                        if m == 128 and n == 128:
                            for qr_base in tla.range(0, 128, 2):
                                for r_offset in tla.range_constexpr(2):
                                    for c_offset in tla.range_constexpr(8):
                                        keep_full = mask_mod(
                                            b,
                                            h,
                                            qb * 128 + qr_base + r_offset,
                                            kb * 128 + col + c_offset * 16,
                                            info,
                                            aux_tensors,
                                            aux_scalars,
                                        )
                                        if keep_full:
                                            visible = visible + 1
                        else:
                            for tail_col in tla.range_constexpr(8):
                                if col + tail_col * 16 < n:
                                    for r in tla.range(m):
                                        keep = mask_mod(
                                            b,
                                            h,
                                            qb * 128 + r,
                                            kb * 128 + col + tail_col * 16,
                                            info,
                                            aux_tensors,
                                            aux_scalars,
                                        )
                                        if keep:
                                            visible = visible + 1
                        counts[tid] = visible
                        tla.arch.sync_threads()
                        for level in tla.range_constexpr(4):
                            offset = 8 >> level
                            if col < offset:
                                counts[tid] = counts[tid] + counts[tid + offset]
                            tla.arch.sync_threads()
                        if col == 0:
                            nnz = counts[tid]
                            kind = tla.Int32(0)
                            if m > 0 and n > 0:
                                if nnz == m * n:
                                    kind = tla.Int32(2)
                                elif nnz > 0:
                                    kind = tla.Int32(1)
                            if kb < nk:
                                pi[row, kb] = kind
            tla.pipe_barrier(tla.pipes.ALL)
        tla.cross_core_set_flag(classification_done, tla.arch.MTE2)
        tla.cross_core_wait_flag(classification_done, tla.arch.MTE2)
        tla.pipe_barrier(tla.pipes.ALL)
        for output_row in tla.range(
            tla.arch.block_idx(), batches * heads * nq, tla.arch.block_num()
        ):
            with tla.vec.func(mode="simt", thread_block_dim=512):
                lane, _, _ = tla.arch.thread_idx()
                partial = tla.Int32(0)
                full = tla.Int32(0)
                for output_base in tla.range(0, nk, 512):
                    selected = tla.Int32(0)
                    if output_base + lane < nk:
                        selected = pi[output_row, output_base + lane]
                    kinds[lane] = selected
                    counts[lane] = tla.Int32(selected == 1) | (tla.Int32(selected == 2) << 16)
                    tla.arch.sync_threads()
                    for scan_level in tla.range_constexpr(9):
                        distance = 1 << scan_level
                        source = (scan_level % 2) * 512
                        target = ((scan_level + 1) % 2) * 512
                        rank = counts[source + lane]
                        if lane >= distance:
                            rank = rank + counts[source + lane - distance]
                        counts[target + lane] = rank
                        tla.arch.sync_threads()
                    inclusive = counts[512 + lane]
                    out_kind = kinds[lane]
                    if out_kind == 2:
                        fi[output_row, full + (inclusive >> 16) - 1] = output_base + lane
                    elif out_kind == 1:
                        pi[output_row, partial + (inclusive & 65535) - 1] = output_base + lane
                    totals = counts[1023]
                    partial = partial + (totals & 65535)
                    full = full + (totals >> 16)
                    tla.arch.sync_threads()
                if lane == 0:
                    pc[output_row] = partial
                    fc[output_row] = full
            tla.pipe_barrier(tla.pipes.ALL)
