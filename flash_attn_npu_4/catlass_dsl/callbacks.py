"""Compile-time adapters for native callbacks; no tracing or device fallback."""

from functools import lru_cache
import inspect

from .utils import get_tla
from .decorators import _Callback, _native
from .mods import Mod, MaskMod, UnsupportedModError


tla = get_tla()


class SeqlenInfoQK:
    """Expose logical sequence lengths and packed offsets to callbacks.

    Parameters
    ----------
    values : sequence
        Four values in the order ``(seqlen_q, seqlen_k, offset_q, offset_k)``.
        Native callbacks receive TLA scalar expressions; independent reference
        callbacks receive the corresponding host/reference values.

    Attributes
    ----------
    seqlen_q, seqlen_k : scalar
        Actual query and KV lengths of the current batch segment.
    offset_q, offset_k : scalar
        Global token offsets of the current packed segments. A fixed-layout
        side has offset zero.

    Notes
    -----
    Callback adapters create this object automatically. Query/key coordinates
    are local to the current segment; add the corresponding offset to index
    auxiliary data stored by global packed token. Positional indexing is also
    supported: ``info[0]`` through ``info[3]`` follow the constructor order.
    """

    def __init__(self, values):
        self.seqlen_q, self.seqlen_k = values[0], values[1]
        self.offset_q, self.offset_k = values[2], values[3]
        self._values = values

    def __getitem__(self, index):
        return self._values[index]


def _call_style(fn, kind, has_scalars):
    sig = inspect.signature(fn)
    n = {"mask": 6, "score": 7, "score_bwd": 8}[kind]
    # Prefer named arguments for score coordinates (including keyword-only
    # arguments). Positional functions are equally valid native callbacks.
    if kind != "mask":
        names = dict(q_idx=None, kv_idx=None, seqlen_info=None, aux_tensors=None)
        for scalar in (True, False):
            if has_scalars and not scalar:
                continue
            try:
                sig.bind(
                    *([None] * (n - 4)),
                    **names,
                    **({"aux_scalars": None} if scalar else {}),
                )
                return True, scalar
            except TypeError:
                pass
    for scalar in (True, False):
        if has_scalars and not scalar:
            continue
        try:
            sig.bind(*([None] * (n + int(scalar))))
            return False, scalar
        except TypeError:
            pass
    raise UnsupportedModError(
        f"{kind} callback has an incompatible signature or cannot accept aux_scalars"
    )


@lru_cache(maxsize=512)
def adapt_callback(fn, kind, has_scalars=False):
    keyword, scalar = _call_style(fn, kind, has_scalars)
    if kind == "mask":

        @tla.jit
        def adapted(b, h, q, k, info, aux, scalars):
            seq = SeqlenInfoQK(info)
            if tla.const_expr(scalar):
                return fn(b, h, q, k, seq, aux, scalars)
            else:
                return fn(b, h, q, k, seq, aux)

    elif kind == "score":

        @tla.jit
        def adapted(s, b, h, q, k, info, aux, scalars):
            seq = SeqlenInfoQK(info)
            if tla.const_expr(keyword):
                if tla.const_expr(scalar):
                    return fn(
                        s,
                        b,
                        h,
                        q_idx=q,
                        kv_idx=k,
                        seqlen_info=seq,
                        aux_tensors=aux,
                        aux_scalars=scalars,
                    )
                else:
                    return fn(s, b, h, q_idx=q, kv_idx=k, seqlen_info=seq, aux_tensors=aux)
            else:
                if tla.const_expr(scalar):
                    return fn(s, b, h, q, k, seq, aux, scalars)
                else:
                    return fn(s, b, h, q, k, seq, aux)

    else:

        @tla.jit
        def adapted(g, s, b, h, q, k, info, aux, scalars):
            seq = SeqlenInfoQK(info)
            if tla.const_expr(keyword):
                if tla.const_expr(scalar):
                    return fn(
                        g,
                        s,
                        b,
                        h,
                        q_idx=q,
                        kv_idx=k,
                        seqlen_info=seq,
                        aux_tensors=aux,
                        aux_scalars=scalars,
                    )
                else:
                    return fn(g, s, b, h, q_idx=q, kv_idx=k, seqlen_info=seq, aux_tensors=aux)
            else:
                if tla.const_expr(scalar):
                    return fn(g, s, b, h, q, k, seq, aux, scalars)
                else:
                    return fn(g, s, b, h, q, k, seq, aux)

    return adapted


@lru_cache(maxsize=512)
def adapt_reference(fn, kind, has_scalars=False):
    keyword, scalar = _call_style(fn, kind, has_scalars)

    def ref(*args):
        values, info, aux, scalars = args[:-3], args[-3], args[-2], args[-1]
        seq = SeqlenInfoQK(info)
        if keyword:
            return fn(
                *values[:-2],
                q_idx=values[-2],
                kv_idx=values[-1],
                seqlen_info=seq,
                aux_tensors=aux,
                **({"aux_scalars": scalars} if scalar else {}),
            )
        return fn(*values, seq, aux, *((scalars,) if scalar else ()))

    return ref


@lru_cache(maxsize=512)
def prepare_mod(fn, kind, has_scalars=False, simd=None, reference=None):
    if fn is None:
        if simd is not None or reference is not None:
            raise ValueError(f"{kind} callback is required when callback options are supplied")
        return None
    if isinstance(fn, Mod):
        from .mods import normalize_mod

        return normalize_mod(fn, kind, simd=simd, torch_ref=reference)
    scalar = fn
    vector = simd
    if isinstance(fn, _Callback):
        scalar, vector = fn.simt, fn.simd
        if simd is not None:
            if vector is not None:
                raise ValueError(
                    "SIMD implementation supplied both by decorator and callback options"
                )
            vector = _native(simd, "simd")
    elif isinstance(simd, _Callback):
        vector = _native(simd, "simd")
    for value in (scalar, vector):
        if value is not None and not getattr(value, "_tla_jit", False):
            raise UnsupportedModError("callbacks must use @tla.jit; PyTorch/FX capture was removed")
    native = None if scalar is None else adapt_callback(scalar, kind, has_scalars)
    vector = None if vector is None else adapt_callback(vector, kind, has_scalars)
    ref = None if reference is None else adapt_reference(reference, kind, has_scalars)
    return MaskMod(native, vector, ref) if kind == "mask" else Mod(native, vector, ref)
