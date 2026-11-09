"""Executor caches with optional, versioned frontend-free disk loading.

Caching executors here skips CATLASS frontend work on repeated launches.
The attention artifact store is separate from CATLASS's backend cache.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, MutableMapping
from contextlib import ExitStack, contextmanager
import fcntl
from getpass import getuser
import os
from pathlib import Path
import shutil
import tempfile
from threading import RLock
import warnings
from typing import Any, TypeAlias

CompileKeyType: TypeAlias = tuple[Hashable, ...]
CallableFunction: TypeAlias = Callable[..., Any]
_COMPILATION_LOCK = RLock()


def _force_recompile():
    return os.getenv("CATLASS_DSL_FORCE_RECOMPILE", "0").lower() in ("1", "true", "yes", "on")


class JITCache(MutableMapping[CompileKeyType, CallableFunction]):
    """Cache compiled functions with serialized compilation and clearing.

    Use get_or_compile for lazy, atomic lookup and compilation. The mapping
    interface supports inspecting entries and evicting them for profiling.
    """

    def __init__(self) -> None:
        self._cache: dict[CompileKeyType, CallableFunction] = {}
        self._lock = _COMPILATION_LOCK

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
            if not _force_recompile() and key in self._cache:
                return self._cache[key]
            fn = compile_fn()
            self._cache[key] = fn
            return fn

    def clear(self, *, persistent=False) -> None:
        """Drop executor references without removing CATLASS disk artifacts."""
        with self._lock:
            self._cache.clear()


@contextmanager
def _file_lock(path, *, shared=False):
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class JITPersistentCache(JITCache):
    """Memory cache backed by atomic, versioned CATLASS artifact directories."""

    def __init__(self, name):
        super().__init__()
        self.name = name or "default"
        self._path = None
        self._warnings = set()

    def _warn(self, operation, error):
        if operation not in self._warnings:
            self._warnings.add(operation)
            warnings.warn(f"CATLASS attention cache {operation}: {error}; using normal compilation",
                          RuntimeWarning, stacklevel=3)

    def _directory(self):
        if self._path is None:
            from .persistent import source_fingerprint

            root = Path(os.environ.get("FLASH_ATTENTION_CATLASS_DSL_CACHE_DIR",
                                       str(Path(tempfile.gettempdir()) / getuser() / "flash_attention_catlass_dsl_cache")))
            self._path = root.expanduser() / source_fingerprint() / self.name
        self._path.mkdir(parents=True, exist_ok=True)
        return self._path

    def get_or_compile(self, key, compile_fn):
        from .persistent import export_function, key_digest, load_function

        with self._lock:
            forced = _force_recompile()
            if not forced and key in self._cache:
                return self._cache[key]
            digest = key_digest(key)
            if digest is None:
                return super().get_or_compile(key, compile_fn)
            locks = ExitStack()
            try:
                directory = self._directory()
                locks.enter_context(_file_lock(directory / ".namespace.lock", shared=True))
                # The first key field partitions fwd/bwd/mxfp8/mask artifacts.
                family = str(key[0]) if key and str(key[0]) in ("fwd", "bwd", "mask", "mxfp8") else "other"
                directory = directory / family
                directory.mkdir(exist_ok=True)
                locks.enter_context(_file_lock(directory / f"{digest}.lock"))
            except (OSError, RuntimeError, ImportError, AttributeError) as error:
                locks.close()
                self._warn("unavailable", error)
                return super().get_or_compile(key, compile_fn)
            with locks:
                entry = directory / digest
                if not forced and entry.exists():
                    try:
                        fn = load_function(entry)
                    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError):
                        pass  # Truncated, incompatible, or corrupt artifact: rebuild.
                    else:
                        self._cache[key] = fn
                        return fn
                # Compiler failures must propagate and never publish an entry.
                fn = compile_fn()
                self._cache[key] = fn
                temporary = None
                try:
                    temporary = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=directory))
                    export_function(fn, temporary)
                    if entry.exists():
                        shutil.rmtree(entry)
                    os.replace(temporary, entry)
                except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as error:
                    self._warn("export failed", error)
                finally:
                    if temporary is not None and temporary.exists():
                        shutil.rmtree(temporary, ignore_errors=True)
                return fn

    def clear(self, *, persistent=False):
        with self._lock:
            super().clear()
            if persistent:
                try:
                    directory = self._directory()
                    with _file_lock(directory / ".namespace.lock"):
                        for child in directory.iterdir():
                            if child.is_dir():
                                shutil.rmtree(child)
                except (OSError, RuntimeError, ImportError, AttributeError) as error:
                    self._warn("clear failed", error)


def get_jit_cache(name: str | None = None) -> JITCache:
    """Create an independent cache, optionally backed by a named disk partition."""
    if name is not None and (not name or Path(name).name != name or name in (".", "..")):
        raise ValueError("cache name must be a single directory component")
    return JITPersistentCache(name) if os.getenv("FLASH_ATTENTION_CATLASS_DSL_CACHE_ENABLED", "0") == "1" else JITCache()
