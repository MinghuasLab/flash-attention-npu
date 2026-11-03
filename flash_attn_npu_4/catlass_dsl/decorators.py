"""Declare the execution mode of native attention callbacks."""

from dataclasses import dataclass
from functools import partial


@dataclass(frozen=True, eq=False)
class _Callback:
    simt: object
    simd: object
    mode: str

    @property
    def __wrapped__(self):
        return getattr(self, self.mode)

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)


def _native(fn, mode):
    if isinstance(fn, _Callback):
        fn = getattr(fn, mode)
        if fn is None:
            raise TypeError(f"the supplied callback has no {mode.upper()} implementation")
    if not callable(fn):
        raise TypeError("callback implementations must be callable")
    if not getattr(fn, "_tla_jit", False):
        raise TypeError(f"@{mode} must be outside @tla.jit; callbacks must already use @tla.jit")
    return fn


def _decorate(fn, mode, alternate):
    if fn is None:
        return partial(_decorate, mode=mode, alternate=alternate)
    if isinstance(fn, _Callback):
        raise TypeError(
            "do not stack @simt and @simd; bind the other implementation with simt= or simd="
        )
    primary = _native(fn, mode)
    other_mode = "simd" if mode == "simt" else "simt"
    other = None if alternate is None else _native(alternate, other_mode)
    return _Callback(primary, other, mode) if mode == "simt" else _Callback(other, primary, mode)


def simt(fn=None, *, simd=None):
    """Mark a @tla.jit callback as scalar, optionally binding an SIMD implementation.

    Place @simt above @tla.jit. This decorator only marks the mode;
    plain @tla.jit callbacks already default to SIMT.
    """
    return _decorate(fn, "simt", simd)


def simd(fn=None, *, simt=None):
    """Mark a @tla.jit callback as 64-lane, optionally binding a SIMT implementation.

    Place @simd above @tla.jit. This decorator only marks the mode.
    Use @simd(simt=scalar_callback) when mixed callbacks or device sparse
    metadata construction also need a scalar implementation.
    """
    return _decorate(fn, "simd", simt)
