"""Bounded SIMT callback stages for the ordinary DN attention pipelines."""

import catlass.tla as tla


@tla.jit
def prepare_scores_simt(
    scores,
    scale,
    b,
    h,
    qstart,
    kstart,
    full,
    score_mod: tla.Constexpr,
    mask_mod: tla.Constexpr,
    aux,
    scalars,
    info,
):
    with tla.vec.func(mode="simt", thread_block_dim=1024):
        tid, _, _ = tla.arch.thread_idx()
        for lane in tla.range_constexpr(8):
            i = tid + lane * 1024
            kr = i // 64
            qr = i % 64
            qi = qstart + qr
            ki = kstart + kr
            visible = qi < info[0] and ki < info[1]
            if visible:
                if not full:
                    visible = mask_mod(tla.Float32(0.0), b, h, qi, ki, aux, scalars, info, True)
            z = tla.Float32(float("-inf"))
            if visible:
                z = score_mod(scores[kr, qr] * scale, b, h, qi, ki, aux, scalars, info)
            scores[kr, qr] = z


@tla.jit
def prepare_pds_simt(
    scores,
    dps,
    lse,
    delta,
    scale,
    b,
    h,
    qstart,
    kstart,
    full,
    score_mod: tla.Constexpr,
    mask_mod: tla.Constexpr,
    score_grad: tla.Constexpr,
    aux,
    scalars,
    info,
):
    # Each element consumes its raw score/dP before replacing them with P/dS.
    # The following SIMD stage packs these buffers to the existing NZ L1 slots.
    with tla.vec.func(mode="simt", thread_block_dim=1024):
        tid, _, _ = tla.arch.thread_idx()
        for lane in tla.range_constexpr(8):
            i = tid + lane * 1024
            kr = i // 64
            qr = i % 64
            qi = qstart + qr
            ki = kstart + kr
            visible = qi < info[0] and ki < info[1]
            if visible:
                if not full:
                    visible = mask_mod(tla.Float32(0.0), b, h, qi, ki, aux, scalars, info, True)
            p = tla.Float32(0.0)
            ds = tla.Float32(0.0)
            if visible:
                x = scores[kr, qr] * scale
                z = score_mod(x, b, h, qi, ki, aux, scalars, info)
                if lse[qr] != float("-inf"):
                    p = tla.exp(z - lse[qr])
                dz = tla.Float32(0.0)
                if p > 0.0:
                    dz = p * (dps[kr, qr] - delta[qr])
                if z != float("-inf"):
                    ds = score_grad(dz, x, b, h, qi, ki, info, aux, scalars)
            scores[kr, qr] = p
            dps[kr, qr] = ds
