"""Process-local caching for compiled functions.

Caching executors here skips CATLASS frontend work on repeated launches.
CATLASS continues to manage its own persistent compilation artifacts.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, MutableMapping
from threading import RLock
from typing import Any, TypeAlias

CompileKeyType: TypeAlias = tuple[Hashable, ...]
CallableFunction: TypeAlias = Callable[..., Any]


class JITCache(MutableMapping[CompileKeyType, CallableFunction]):
    """Cache compiled functions with serialized compilation and clearing.

    Use get_or_compile for lazy, atomic lookup and compilation. The mapping
    interface supports inspecting entries and evicting them for profiling.
    """

    def __init__(self) -> None:
        self._cache: dict[CompileKeyType, CallableFunction] = {}
        self._lock = RLock()

    def __getitem__(self, key: CompileKeyType) -> CallableFunction:
        with self._lock:
            return self._cache[key]

    def __setitem__(self, key: CompileKeyType, fn: CallableFunction) -> None:
        with self._lock:
            self._cache[key] = fn

    def __delitem__(self, key: CompileKeyType) -> None:
        with self._lock:
            del self._cache[key]

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._cache

    def __iter__(self) -> Iterator[CompileKeyType]:
        with self._lock:
            return iter(tuple(self._cache))

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def get_or_compile(
        self, key: CompileKeyType, compile_fn: Callable[[], CallableFunction]
    ) -> CallableFunction:
        """Compile only on a miss; failed compilations leave no cache entry."""
        # Keep compilation under one lock, including across different keys,
        # to preserve the runtime's serialization of CATLASS compiler state.
        with self._lock:
            try:
                return self._cache[key]
            except KeyError:
                pass
            fn = compile_fn()
            self._cache[key] = fn
            return fn

    def clear(self) -> None:
        """Drop executor references without removing CATLASS disk artifacts."""
        with self._lock:
            self._cache.clear()


def get_jit_cache() -> JITCache:
    """Create an independent in-memory cache for compiled functions."""
    return JITCache()
