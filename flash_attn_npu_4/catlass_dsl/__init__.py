"""Native Ascend attention and sparse-mask metadata APIs.

Use ``flash_attn_func`` for fixed BSND inputs and
``flash_attn_varlen_func`` for independently fixed or packed Q/KV inputs.
Both return an ``(out, lse)`` pair and support native TLA callbacks.
``compute_block_sparsity`` constructs explicit query-major sparse block lists.
"""

from importlib import import_module

# FA4 imports utils for routing even when only the C++ backend is used.
# Resolve native APIs lazily so that path never checks or imports CATLASS.
_EXPORTS = {
    "flash_attn_func": "interface",
    "flash_attn_varlen_func": "interface",
    "compute_block_sparsity": "block_sparsity",
    "simd": "decorators",
    "simt": "decorators",
}

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "compute_block_sparsity",
    "simd",
    "simt",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .utils import ensure_backend

    ensure_backend()
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
