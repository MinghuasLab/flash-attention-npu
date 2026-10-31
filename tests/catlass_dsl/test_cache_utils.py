"""Compile-cache lifecycle, concurrent misses and lazy runtime binding."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from flash_attn_npu_4.catlass_dsl import runtime
from flash_attn_npu_4.catlass_dsl.cache_utils import get_jit_cache


def test_cache_reuse_eviction_and_isolation():
    cache = get_jit_cache()
    other = get_jit_cache()
    key = ("attention", "npu:0", "fp16")
    first, second = Mock(), Mock()
    compile_fn = Mock(side_effect=[first, second])

    assert cache.get_or_compile(key, compile_fn) is first
    assert cache.get_or_compile(key, compile_fn) is first
    compile_fn.assert_called_once_with()
    assert key not in other
    assert dict(cache.items()) == {key: first}
    assert list(cache.values()) == [first]

    # Profiling evicts only the outer executor entry, then recompiles it.
    assert cache.pop(key, None) is first
    assert cache.pop(key, None) is None
    assert cache.get_or_compile(key, compile_fn) is second
    assert compile_fn.call_count == 2
    cache.clear()
    assert not cache


@pytest.mark.parametrize(
    "error", [RuntimeError("compile failed"), KeyError("compiler lookup failed")]
)
def test_failed_compilation_can_be_retried(error):
    cache = get_jit_cache()
    key = ("attention",)
    executor = Mock()
    compile_fn = Mock(side_effect=[error, executor])
    with pytest.raises(type(error), match=str(error.args[0])):
        cache.get_or_compile(key, compile_fn)
    assert key not in cache
    assert cache.get_or_compile(key, compile_fn) is executor
    assert cache.get_or_compile(key, compile_fn) is executor
    assert compile_fn.call_count == 2


def test_concurrent_misses_compile_once():
    cache = get_jit_cache()
    executor = Mock()
    ready = Barrier(5)
    compiling, release = Event(), Event()

    def compile_kernel():
        compiling.set()
        assert release.wait(timeout=5)
        return executor

    compile_fn = Mock(side_effect=compile_kernel)

    def lookup():
        ready.wait(timeout=5)
        return cache.get_or_compile(("attention",), compile_fn)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(lookup) for _ in range(4)]
        try:
            ready.wait(timeout=5)
            assert compiling.wait(timeout=5)
        finally:
            release.set()
        assert all(future.result(timeout=5) is executor for future in futures)
    compile_fn.assert_called_once_with()


def test_clear_waits_for_inflight_compilation():
    cache = get_jit_cache()
    compiling, clearing, release = Event(), Event(), Event()

    def compile_kernel():
        compiling.set()
        assert release.wait(timeout=5)
        return Mock()

    def clear():
        clearing.set()
        cache.clear()

    with ThreadPoolExecutor(max_workers=2) as pool:
        compilation = pool.submit(cache.get_or_compile, ("mask",), compile_kernel)
        try:
            assert compiling.wait(timeout=5)
            cleared = pool.submit(clear)
            assert clearing.wait(timeout=5)
            with pytest.raises(TimeoutError):
                cleared.result(timeout=0.05)
        finally:
            release.set()
        compilation.result(timeout=5)
        cleared.result(timeout=5)
    assert not cache


@pytest.mark.parametrize("aux_index", [6, 15], ids=["mask", "attention"])
def test_runtime_binds_samples_only_on_miss(monkeypatch, aux_index):
    cache = get_jit_cache()
    monkeypatch.setattr(runtime, "compile_cache", cache)
    first, second = Mock(), Mock()
    backend = SimpleNamespace(compile=Mock(side_effect=[first, second]))
    get_tla = Mock(return_value=backend)
    monkeypatch.setattr(runtime, "_tla", get_tla)

    def convert(value, shape, *, logical):
        return (value, shape, logical)

    sample = Mock(side_effect=convert)
    monkeypatch.setattr(runtime, "_sample", sample)
    key, kernel = ("specialization",), Mock()
    args = (*range(aux_index), ("aux",))
    shapes = {1: (2, 3)}
    assert runtime._compile(key, kernel, args, shapes) is first
    sample.assert_has_calls(
        [call(arg, shapes.get(i), logical=i == aux_index) for i, arg in enumerate(args)]
    )
    backend.compile.assert_called_once_with(
        kernel,
        *((arg, shapes.get(i), i == aux_index) for i, arg in enumerate(args)),
        options="--npu-arch 3510",
    )

    sample.reset_mock(side_effect=True)
    sample.side_effect = AssertionError("warm cache must skip sample conversion")
    fresh_args = (*range(aux_index), ("fresh aux",))
    assert runtime._compile(key, kernel, fresh_args, shapes) is first
    sample.assert_not_called()
    get_tla.assert_called_once_with()
    assert backend.compile.call_count == 1

    runtime.clear_compile_cache()
    assert runtime.compile_cache is cache and not cache
    sample.side_effect = convert
    assert runtime._compile(key, kernel, fresh_args, shapes) is second
    assert backend.compile.call_count == 2
    assert backend.compile.call_args.args[-1] == (("fresh aux",), None, True)
