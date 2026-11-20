"""Cache keys for the independent CATLASS DSL CPU references."""

import hashlib
import inspect
import marshal
import os
from types import CodeType, ModuleType

import torch

from flash_attn_npu_4.catlass_dsl.mods import Mod, reference_callable
from tests.common import golden_cache


def _global_names(code):
    names = set(code.co_names)
    for value in code.co_consts:
        if isinstance(value, CodeType):
            names.update(_global_names(value))
    return names


def _capture(value, sources, active):
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if isinstance(value, (torch.Tensor, torch.dtype)):
        return value
    if type(value) in (tuple, list):
        return type(value).__name__, [_capture(item, sources, active) for item in value]
    if isinstance(value, ModuleType):
        name = value.__name__
        if name.startswith(f"{__package__}.") and getattr(value, "__file__", None):
            sources.add(value.__file__)
        elif name not in ("torch", "math"):
            raise TypeError("unsupported reference module")
        return "module", name
    if not inspect.isfunction(value) or value in active:
        raise TypeError("unsupported reference callback capture")
    code = value.__code__
    active = active | {value}
    captures = [_capture(cell.cell_contents, sources, active) for cell in value.__closure__ or ()]
    dependencies = {
        name: _capture(value.__globals__[name], sources, active)
        for name in sorted(_global_names(code))
        if name in value.__globals__
    }
    defaults = {
        name: _capture(default, sources, active)
        for name, default in (value.__kwdefaults__ or {}).items()
    }
    return (
        hashlib.sha256(marshal.dumps(code)).hexdigest(),
        captures,
        dependencies,
        _capture(value.__defaults__, sources, active),
        defaults,
    )


def cached_golden(api_name, names, compute, *, inputs, callbacks, source_files):
    nodeid = os.environ.get("GOLDEN_CACHE_NODEID")
    if not nodeid or not golden_cache._cache_enabled():
        return compute()

    sources = {__file__, *source_files}
    test_file = os.environ.get("GOLDEN_CACHE_TEST_FILE")
    if test_file:
        sources.add(test_file)
    try:
        callback_keys = {
            kind: _capture(
                reference_callable(fn, kind) if isinstance(fn, Mod) else fn, sources, set()
            )
            for kind, fn in callbacks.items()
        }
    except (TypeError, ValueError, RecursionError):
        # A callable with unknown state still works as a reference.
        return compute()

    values = golden_cache.get_or_compute_golden(
        nodeid=nodeid,
        api_name=api_name,
        metadata={
            "cpu_threads": torch.get_num_threads(),
            "default_dtype": torch.get_default_dtype(),
        },
        inputs=(inputs, callback_keys),
        compute_fn=lambda: dict(zip(names, compute())),
        expected_keys=names,
        source_files=tuple(sources),
    )
    return tuple(values[name] for name in names)
