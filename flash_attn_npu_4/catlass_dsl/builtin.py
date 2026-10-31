"""Native library callbacks with independent Torch references.

Factories are cached so stable callback specializations reuse executors.
"""

from functools import lru_cache
import torch
from .utils import get_tla
from .mods import Mod, MaskMod, _register
from .native import tanh


tla = get_tla()


@tla.jit
def _identity(s, b, h, q, k, info, aux, scalars):
    return s


@tla.jit
def _identity_bwd(g, s, b, h, q, k, info, aux, scalars):
    return g


@tla.jit
def _true(b, h, q, k, info, aux, scalars):
    return tla.Bool(True)


@tla.jit
def _true_simd(b, h, q, k, info, aux, scalars):
    return tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)


identity = _register(Mod(_identity, _identity, lambda s, b, h, q, k, i, a, c: s), "identity")
identity_bwd = Mod(_identity_bwd, _identity_bwd, lambda g, s, b, h, q, k, i, a, c: g)
all_visible = _register(
    MaskMod(_true, _true_simd, lambda b, h, q, k, i, a, c: True, True, True), "all"
)


@lru_cache(maxsize=128)
def causal(*, alignment="top_left"):
    if alignment not in ("top_left", "bottom_right"):
        raise ValueError("alignment must be top_left or bottom_right")
    shifted = alignment == "bottom_right"

    @tla.jit
    def simt(b, h, q, k, info, aux, scalars):
        shift = info[1] - info[0] if tla.const_expr(shifted) else tla.Int32(0)
        return k <= q + shift

    @tla.jit
    def simd(b, h, q, k, info, aux, scalars):
        shift = info[1] - info[0] if tla.const_expr(shifted) else tla.Int32(0)
        return tla.cmp(k, tla.add(q, shift), mode="le")

    def ref(b, h, q, k, info, aux, scalars):
        return k <= q + (info[1] - info[0] if shifted else 0)

    return _register(
        MaskMod(simt, simd, ref, True, True),
        "causal_bottom_right" if shifted else "causal",
    )


@lru_cache(maxsize=128)
def sliding_window(width):
    if type(width) is not int or not 0 < width < 2**31:
        raise ValueError("width must be a positive int32")

    @tla.jit
    def simt(b, h, q, k, info, aux, scalars):
        return (q >= k) & (q - k < width)

    @tla.jit
    def simd(b, h, q, k, info, aux, scalars):
        return tla.bitwise_and(tla.cmp(q, k, mode="ge"), tla.cmp(tla.sub(q, k), width, mode="lt"))

    return _register(
        MaskMod(
            simt,
            simd,
            lambda b, h, q, k, i, a, c: (q >= k) & (q - k < width),
            True,
            True,
        ),
        "window",
        width,
    )


@lru_cache(maxsize=128)
def alibi(slope=0.01):
    @tla.jit
    def simt(s, b, h, q, k, info, aux, scalars):
        distance = (q - k).to(tla.Float32)
        return s - (h + 1).to(tla.Float32) * slope * tla.abs(distance)

    @tla.jit
    def simd(s, b, h, q, k, info, aux, scalars):
        params = tla.params.CastParams(
            round_mode=tla.params.RoundMode.CAST_ROUND, reg_slot=tla.params.RegSlot.ZERO
        )
        distance = tla.sub(q, k).to(tla.Float32, params)
        heads = tla.add(h, 1).to(tla.Float32, params)
        return tla.sub(s, tla.mul(tla.mul(heads, slope), tla.abs(distance)))

    return Mod(
        simt,
        simd,
        lambda s, b, h, q, k, i, a, c: s - (h + 1) * slope * torch.abs(q - k),
    )


@lru_cache(maxsize=128)
def softcap(slot=0):
    @tla.jit
    def simt(s, b, h, q, k, info, aux, scalars):
        cap = scalars[slot]
        return cap * tanh(s / cap)

    @tla.jit
    def simd(s, b, h, q, k, info, aux, scalars):
        cap = tla.add(tla.full(0.0, tla.Float32), scalars[slot])
        x = tla.div(s, cap)
        e = tla.exp(tla.mul(tla.abs(x), -2.0))
        mag = tla.div(tla.sub(tla.full(1.0, tla.Float32), e), tla.add(e, 1.0))
        y = tla.where(tla.cmp(x, 0.0, mode="lt"), tla.neg(mag), mag)
        return tla.mul(cap, y)

    return Mod(simt, simd, lambda s, b, h, q, k, i, a, c: c[slot] * torch.tanh(s / c[slot]))


@lru_cache(maxsize=128)
def softcap_bwd(slot=0):
    @tla.jit
    def simt(g, s, b, h, q, k, info, aux, scalars):
        t = tanh(s / scalars[slot])
        return g * (1.0 - t * t)

    @tla.jit
    def simd(g, s, b, h, q, k, info, aux, scalars):
        cap = tla.add(tla.full(0.0, tla.Float32), scalars[slot])
        e = tla.exp(tla.mul(tla.abs(tla.div(s, cap)), -2.0))
        mag = tla.div(tla.sub(tla.full(1.0, tla.Float32), e), tla.add(e, 1.0))
        return tla.mul(g, tla.sub(tla.full(1.0, tla.Float32), tla.mul(mag, mag)))

    return Mod(
        simt,
        simd,
        lambda g, s, b, h, q, k, i, a, c: g * (1 - torch.tanh(s / c[slot]) ** 2),
    )


@lru_cache(maxsize=256)
def pipeline_adapter(mod, kind, sq, sk, specs=(), mode="simd", varlen=(False, False)):
    """Bind local lengths and global offsets for fixed or uniform packed inputs."""
    from .auxiliary import bind_aux

    fn = bind_aux(getattr(mod, mode), kind, specs)
    if kind == "mask":

        @tla.jit
        def adapted(
            unused, b, h, q, k, aux, scalars, info=(), bounds_checked: tla.Constexpr = False
        ):
            if tla.const_expr(mode == "simt" and len(info) > 0 and not bounds_checked):
                visible = tla.Bool(False)
                if q < info[0] and k < info[1]:
                    visible = fn(b, h, q, k, info, aux, scalars)
                return visible
            else:
                return fn(
                    b,
                    h,
                    q,
                    k,
                    info
                    if tla.const_expr(len(info) > 0)
                    else (
                        tla.Int32(sq),
                        tla.Int32(sk),
                        b * sq if tla.const_expr(varlen[0]) else tla.Int32(0),
                        b * sk if tla.const_expr(varlen[1]) else tla.Int32(0),
                    ),
                    aux,
                    scalars,
                )
    else:

        @tla.jit
        def adapted(s, b, h, q, k, aux, scalars, info=()):
            return fn(
                s,
                b,
                h,
                q,
                k,
                info
                if tla.const_expr(len(info) > 0)
                else (
                    tla.Int32(sq),
                    tla.Int32(sk),
                    b * sq if tla.const_expr(varlen[0]) else tla.Int32(0),
                    b * sk if tla.const_expr(varlen[1]) else tla.Int32(0),
                ),
                aux,
                scalars,
            )

    return adapted


@lru_cache(maxsize=128)
def local_window(left=None, right=None):
    """Inclusive, bottom-right-aligned window used by the public flags."""

    @tla.jit
    def simt(b, h, q, k, info, aux, scalars):
        center = q + info[1] - info[0]
        keep = tla.Bool(True)
        if tla.const_expr(left is not None):
            keep = keep & (center - k <= left)
        if tla.const_expr(right is not None):
            keep = keep & (k - center <= right)
        return keep

    @tla.jit
    def simd(b, h, q, k, info, aux, scalars):
        center = tla.add(q, info[1] - info[0])
        keep = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)
        if tla.const_expr(left is not None):
            keep = tla.bitwise_and(keep, tla.cmp(tla.sub(center, k), left, mode="le"))
        if tla.const_expr(right is not None):
            keep = tla.bitwise_and(keep, tla.cmp(tla.sub(k, center), right, mode="le"))
        return keep

    def ref(b, h, q, k, info, aux, scalars):
        center = q + info[1] - info[0]
        return ((k >= center - left) if left is not None else True) & (
            (k <= center + right) if right is not None else True
        )

    return _register(MaskMod(simt, simd, ref, True, True), "local_window", (left, right))


@lru_cache(maxsize=128)
def softcap_value(cap):
    """Bind a public softcap value without consuming user scalar slots."""
    forward, backward = softcap(), softcap_bwd()
    f, fv, g, gv = forward.simt, forward.simd, backward.simt, backward.simd

    @tla.jit
    def sf(s, b, h, q, k, info, aux, scalars):
        return f(s, b, h, q, k, info, aux, (cap,))

    @tla.jit
    def vf(s, b, h, q, k, info, aux, scalars):
        return fv(s, b, h, q, k, info, aux, (cap,))

    @tla.jit
    def sg(d, s, b, h, q, k, info, aux, scalars):
        return g(d, s, b, h, q, k, info, aux, (cap,))

    @tla.jit
    def vg(d, s, b, h, q, k, info, aux, scalars):
        return gv(d, s, b, h, q, k, info, aux, (cap,))

    return (
        Mod(sf, vf, lambda s, b, h, q, k, i, a, c: cap * torch.tanh(s / cap)),
        Mod(sg, vg, lambda d, s, b, h, q, k, i, a, c: d * (1 - torch.tanh(s / cap) ** 2)),
    )
