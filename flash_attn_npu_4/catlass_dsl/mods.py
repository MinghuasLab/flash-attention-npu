"""Internal callback descriptors and normalization of public TLA callables."""

from dataclasses import dataclass
import inspect

from .decorators import _Callback


class UnsupportedModError(ValueError):
    """Indicate an unsupported attention callback or callback signature.

    Raised when a mask, score, score VJP, or SIMD implementation does not
    satisfy the native TLA callback contract. Public device callbacks must
    use ``@tla.jit`` and accept the documented arguments, including
    ``aux_scalars`` when scalar inputs are supplied. Legacy PyTorch/FX
    callbacks and automatic score differentiation are not supported.

    Notes
    -----
    Subclasses ``ValueError`` so callers may catch either the specific
    callback error or the broader input-validation exception.

    See Also
    --------
    flash_attn_func : Public score, VJP, and mask callback signatures.
    """


# Retain functions in keys, so their identities cannot be recycled. setdefault
# also returns one canonical descriptor if concurrent first calls race.
_NORMALIZED = {}


def normalize_mod(
    callback,
    kind,
    *,
    simd=None,
    torch_ref=None,
    broadcast_batch=None,
    broadcast_head=None,
):
    """Bind a user callable once; existing internal descriptors remain valid.

    No tracing, differentiation, source-equivalence inference, or mutation of
    user functions. Optional metadata is part of the specialization identity.
    """
    if kind not in ("mask", "score", "score_bwd"):
        raise ValueError(f"unknown callback kind: {kind}")
    argument = {"mask": "mask_mod", "score": "score_mod", "score_bwd": "score_mod_bwd"}[kind]
    options = (simd, torch_ref, broadcast_batch, broadcast_head)
    if callback is None:
        if any(value is not None for value in options):
            raise ValueError(f"{kind} callback is required when callback options are supplied")
        return None
    for name, value in (
        ("broadcast_batch", broadcast_batch),
        ("broadcast_head", broadcast_head),
    ):
        if value is not None and type(value) is not bool:
            raise TypeError(f"{name} must be bool")
    if kind != "mask" and (broadcast_batch is not None or broadcast_head is not None):
        raise ValueError("broadcast options apply only to mask callbacks")
    if isinstance(callback, _Callback):
        from .callbacks import prepare_mod

        callback = prepare_mod(callback, kind)
    if isinstance(callback, Mod):
        validate_mod(callback, kind)
        if all(value is None for value in options):
            return callback
        native = callback.simt
        simd = callback.simd if simd is None else simd
        torch_ref = callback.torch_ref if torch_ref is None else torch_ref
        if kind == "mask":
            broadcast_batch = (
                callback.broadcast_batch if broadcast_batch is None else broadcast_batch
            )
            broadcast_head = callback.broadcast_head if broadcast_head is None else broadcast_head
    else:
        if not callable(callback) or not getattr(callback, "_tla_jit", False):
            raise UnsupportedModError(
                f"{argument} must be a @tla.jit callback; legacy PyTorch/FX callbacks were removed"
            )
        native = callback
    if simd is not None and (not callable(simd) or not getattr(simd, "_tla_jit", False)):
        raise UnsupportedModError(f"{kind} SIMD callback must be decorated with @tla.jit")
    batch, head = bool(broadcast_batch), bool(broadcast_head)
    # References may be unhashable callable objects. The descriptor retains
    # them, making id-based metadata keys safe for the lifetime of the cache.
    key = (callback, kind, id(simd), id(torch_ref), batch, head)
    if key not in _NORMALIZED:
        mod = (
            MaskMod(native, simd, torch_ref, batch, head)
            if kind == "mask"
            else Mod(native, simd, torch_ref)
        )
        validate_mod(mod, kind)
        return _NORMALIZED.setdefault(key, mod)
    return _NORMALIZED[key]


@dataclass(frozen=True, eq=False)
class Mod:
    """Immutable, identity-keyed native callback and optional independent oracle.

    ``simt`` is a source-backed TLA callable; ``simd`` operates on 64 lanes.
    At least one implementation must be present.
    Compile-time closures/helpers must remain immutable. Runtime captures belong
    in aux_tensors/aux_scalars. torch_ref is never invoked by the device path.
    """

    simt: object
    simd: object = None
    torch_ref: object = None

    def __post_init__(self):
        if (self.simt is None and self.simd is None) or any(
            x is not None and not callable(x) for x in (self.simt, self.simd, self.torch_ref)
        ):
            raise TypeError(
                "Mod requires a native simt or simd callable and an optional torch_ref callable"
            )

    @property
    def key(self):
        # Keep the object itself in executor keys: id() alone can be recycled.
        return self


@dataclass(frozen=True, eq=False)
class MaskMod(Mod):
    broadcast_batch: bool = False
    broadcast_head: bool = False


def callback_mode(*programs):
    """Select a common implementation without calling SIMD code from SIMT."""
    programs = tuple(p for p in programs if p is not None)
    if all(p.simd is not None for p in programs):
        return "simd"
    if all(p.simt is not None for p in programs):
        return "simt"
    raise UnsupportedModError(
        "SIMD-only and SIMT-only callbacks need a common implementation; "
        "bind an equivalent callback with @simd(simt=...) or @simt(simd=...)"
    )


def validate_mod(mod, kind):
    if mod is None:
        return
    cls = MaskMod if kind == "mask" else Mod
    if not isinstance(mod, cls) or (kind != "mask" and isinstance(mod, MaskMod)):
        raise UnsupportedModError(
            f"invalid internal {kind} descriptor; pass @tla.jit callbacks to the public API. "
            "PyTorch/FX callbacks and automatic differentiation were removed in 0.2"
        )
    arity = {"mask": 7, "score": 8, "score_bwd": 9}[kind]
    for fn in (mod.simt, mod.simd, mod.torch_ref):
        if fn is not None:
            try:
                inspect.signature(fn).bind(*([None] * arity))
            except (ValueError, TypeError) as exc:
                raise UnsupportedModError(
                    f"{kind} callback must accept {arity} positional arguments including seqlen_info and both aux tuples"
                ) from exc


def reference_callable(mod, kind):
    validate_mod(mod, kind)
    if mod is None:
        return None
    if mod.torch_ref is None:
        raise ValueError(f"{kind} callback has no independent torch_ref")
    return mod.torch_ref


# Only library factories populate this registry. User functions are never
# inspected for equivalence and cannot opt into identity/causal dispatch.
_BUILTINS = {}


def _register(mod, kind, value=None, tensor_slots=(), scalar_slots=()):
    _BUILTINS[mod] = (kind, value, tensor_slots, scalar_slots)
    return mod


def builtin_info(mod):
    return _BUILTINS.get(mod)


def is_exact_causal(mod):
    info = builtin_info(mod)
    return info is not None and info[0] == "causal"


def is_causal_equivalent(mod, query_length):
    info = builtin_info(mod)
    return info is not None and (
        info[0] == "causal" or (info[0] == "window" and info[1] >= query_length)
    )
