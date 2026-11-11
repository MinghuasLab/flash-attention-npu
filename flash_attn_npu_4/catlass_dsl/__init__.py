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
    "precompile_attention": "precompile",
    "precompile_backward_attention": "precompile",
    "compile_cache_info": "precompile",
    "forward_compile_mode": "runtime",
    "get_forward_compile_mode": "runtime",
    "backward_compile_mode": "runtime",
    "get_backward_compile_mode": "runtime",
    "clear_compile_cache": "runtime",
    "CompileCacheInfo": "cache_utils",
    "backward_compile_diagnostics": "diagnostics",
    "BackwardCompileDiagnostics": "diagnostics",
    "BackwardCompileEvent": "diagnostics",
}

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "compute_block_sparsity",
    "simd",
    "simt",
    "precompile_attention",
    "precompile_backward_attention",
    "compile_cache_info",
    "forward_compile_mode",
    "get_forward_compile_mode",
    "backward_compile_mode",
    "get_backward_compile_mode",
    "clear_compile_cache",
    "CompileCacheInfo",
    "backward_compile_diagnostics",
    "BackwardCompileDiagnostics",
    "BackwardCompileEvent",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if _EXPORTS[name] not in ("cache_utils", "diagnostics") and name not in (
        "forward_compile_mode",
        "get_forward_compile_mode",
        "backward_compile_mode",
        "get_backward_compile_mode",
    ):
        from .utils import ensure_backend

        ensure_backend()
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
