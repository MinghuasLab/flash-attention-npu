"""Versioned, device-independent CATLASS artifact storage.

This is the only module that depends on CATLASS's prepared-executor internals.
JSON contains compiler ABI/argument metadata, never executable Python objects,
Tensor owners, locks, or loaded device handles.
"""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import types

ABI_VERSION = 1


def file_digest(path):
    with Path(path).open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
        return digest.hexdigest()


def _json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _constant(value, active):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return ["float", value.hex()]
    if type(value) is bytes:
        return ["bytes", value.hex()]
    if isinstance(value, tuple):
        return [type(value).__name__, *(_constant(x, active) for x in value)]
    if isinstance(value, types.ModuleType):
        # These modules are covered by the namespace/version fingerprint.
        # Arbitrary modules can expose mutable state or import dependencies
        # not visible in the callback; a hash of that module alone is unsafe.
        name = value.__name__
        if name.split(".")[0] in ("catlass", "torch", "math", "builtins", "flash_attn_npu_4"):
            return ["module", name]
        raise TypeError("module dependencies cannot be fingerprinted reliably")
    import torch

    if isinstance(value, torch.device):
        return ["device", value.type]  # Loaded owners remain device-local.
    if isinstance(value, torch.dtype):
        return ["dtype", str(value)]
    if type(value).__module__ == "flash_attn_npu_4.catlass_dsl.mods":
        from .mods import Mod

        if isinstance(value, Mod):
            return [
                "mod",
                _constant(value.simt, active),
                _constant(value.simd, active),
                getattr(value, "broadcast_batch", False),
                getattr(value, "broadcast_head", False),
            ]
    if type(value).__module__ == "catlass.tla.ffi" and type(value).__name__ == "ExternFunction":
        # The declaration's Python stub does not contain the generated C++.
        # Extra include roots need dependency discovery; conservatively leave
        # those declarations to the backend cache instead of guessing a key.
        if value.include_dirs:
            raise TypeError("extern has additional include dependencies")
        return [
            "extern",
            value.symbol,
            hashlib.sha256(value.source.encode()).hexdigest(),
            tuple(str(arg) for arg in value.arg_types),
        ]
    if isinstance(value, type) and value.__module__.split(".")[0] in (
        "builtins",
        "catlass",
        "flash_attn_npu_4",
    ):
        return ["type", value.__module__, value.__qualname__]
    if callable(value):
        return _callable(value, active)
    raise TypeError(f"unsupported compile-time capture: {type(value).__name__}")


def _callable(value, active):
    fn = inspect.unwrap(value)
    if not inspect.isfunction(fn):
        raise TypeError("callback is not a source-backed Python function")
    if fn in active:
        return ["recursive", fn.__module__, fn.__qualname__]
    active = active | {fn}
    source = inspect.getsource(fn)
    dependencies = []

    def names(code):
        result = set(code.co_names)
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                result.update(names(constant))
        return result

    for name in sorted(names(fn.__code__)):
        if name in fn.__globals__:
            dependencies.append((name, _constant(fn.__globals__[name], active)))
    return [
        "function",
        source,
        _constant(fn.__defaults__, active),
        [
            (name, _constant(default, active))
            for name, default in sorted((fn.__kwdefaults__ or {}).items())
        ],
        [_constant(c.cell_contents, active) for c in (fn.__closure__ or ())],
        dependencies,
    ]


@lru_cache(maxsize=2048)
def key_digest(key):
    """None means the specialization is intentionally memory-only."""
    try:
        return _json_digest(_constant(key, set()))
    except (TypeError, ValueError, OSError, RecursionError):
        return None


@lru_cache(maxsize=1)
def source_fingerprint():
    import catlass
    from catlass import execution
    from catlass.bc_compile import get_bc_paths
    from catlass.compiler_bridge import resolve_bridge_extension_path

    versions = {}
    for name in ("ascend-catlass-dsl", "torch", "torch_npu"):
        versions[name] = importlib.metadata.version(name)
    if versions["ascend-catlass-dsl"].split(".")[:2] != ["2", "2"] or not hasattr(
        execution, "_new_jit_compiled_function"
    ):
        raise RuntimeError(
            "persistent attention cache requires the CATLASS 2.2 prepared-executor API"
        )
    sources = []
    for root in (Path(__file__).parent, Path(catlass.__file__).parent):
        for path in sorted(root.rglob("*.py")):
            sources.append((root.name, path.relative_to(root).as_posix(), file_digest(path)))
    ccec, include_roots = execution._ascendc_compiler_inputs()
    tools = [resolve_bridge_extension_path(), execution._find_hivmc_a5(), ccec]
    tools.extend(Path(catlass.__file__).parent.rglob("*.so"))
    for variant in ("normal", "print"):
        tools.extend(Path(p) for p in get_bc_paths("mix", variant=variant).split(","))
    return _json_digest(
        (
            ABI_VERSION,
            sys.version_info[:3],
            versions,
            sources,
            [(str(p), file_digest(p)) for p in sorted(set(p for p in tools if p is not None))],
            tuple(str(root) for root in include_roots),
            "--npu-arch 3510",
        )
    )


def _tree_json(tree):
    from catlass.base_dsl.utils.tree_utils import Leaf

    if isinstance(tree, Leaf):
        if tree.kind == "constexpr":
            raise TypeError("persistent runtime trees cannot capture Python constants")
        return {"leaf": tree.kind, "index": tree.binding_index}
    if tree.node_type.name != "tuple" or tree.node_metadata.python_type is not tuple:
        raise TypeError("only plain runtime tuples are stored by the attention cache")
    return {"children": [_tree_json(child) for child in tree.child_treedefs]}


def _read_tree(data):
    from catlass.base_dsl.runtime.argument_tree import _NODE_TYPES
    from catlass.base_dsl.utils.tree_utils import Leaf, PyTreeDef, TreeMetadata

    if "leaf" in data:
        return Leaf(kind=data["leaf"], binding_index=data["index"])
    children = tuple(_read_tree(child) for child in data["children"])
    return PyTreeDef(
        _NODE_TYPES["tuple"], TreeMetadata(tuple, tuple(range(len(children)))), children
    )


def export_function(fn, directory):
    """Export a compiled function into an unpublished temporary directory."""
    import shutil
    from catlass.compiler_bridge import kernel_abi_to_dict

    module, artifacts = fn.jit_module, fn.artifacts
    if module.uses_scalar_print or module.uses_tensor_print:
        raise TypeError("debug-print kernels use the backend's own artifact cache")
    # Include transitive C++/system headers discovered by the real compiler.
    # Hash contents even where CATLASS itself records only size/mtime.
    backend = json.loads((artifacts.cache_dir / "manifest.json").read_text())
    dependencies = sorted(
        {
            entry["path"]
            for group in backend.get("extern_dependency_groups", ())
            for entry in group["dependency_snapshots"]
        }
    )
    trees = []
    for tree in module.execution_args.argument_trees:
        trees.append(
            None
            if tree is None
            else {
                "tree": _tree_json(tree.treedef),
                "bindings": [
                    {"path": list(b.path), "type": str(b.mlir_type), "kind": b.kind}
                    for b in tree.bindings
                ],
            }
        )
    metadata = {
        field.name: getattr(module, field.name)
        for field in fields(module)
        if field.name not in ("kernel_binary_path", "execution_args", "print_metadata")
    }
    metadata.update(
        version=ABI_VERSION,
        backend_key=artifacts.cache_key,
        guarded_fallback=getattr(fn, "_attention_guarded_fallback", False),
        dependencies={path: file_digest(path) for path in dependencies},
        abi=kernel_abi_to_dict(module.execution_args.kernel_abi),
        trees=trees,
    )
    shutil.copyfile(module.kernel_binary_path, directory / "kernel.o")
    (directory / "tla.mlir").write_text(artifacts.MLIR)
    (directory / "lowered.mlir").write_text(artifacts.LLVM)
    metadata["checksums"] = {
        name: file_digest(directory / name) for name in ("kernel.o", "tla.mlir", "lowered.mlir")
    }
    (directory / "manifest.json").write_text(json.dumps(metadata, sort_keys=True))


def load_function(directory):
    from catlass.base_dsl.jit_executor import (
        ExecutionArgs,
        JitCompiledFunction,
        JitFunctionArtifacts,
        JitModule,
    )
    from catlass.base_dsl.runtime.argument_binding import RuntimeArgumentTree, RuntimeLeafBinding
    from catlass.compiler_bridge import kernel_abi_from_dict
    from catlass.execution import _prepare_abi_packer

    metadata = json.loads((directory / "manifest.json").read_text())
    if metadata.pop("version") != ABI_VERSION:
        raise ValueError("persistent cache ABI changed")
    for path, digest in metadata.pop("dependencies").items():
        if file_digest(path) != digest:
            raise ValueError(f"compiler dependency changed: {path}")
    checksums = metadata.pop("checksums")
    for name in ("kernel.o", "tla.mlir", "lowered.mlir"):
        if file_digest(directory / name) != checksums[name]:
            raise ValueError(f"corrupt artifact: {name}")
    trees = tuple(
        None
        if entry is None
        else RuntimeArgumentTree(
            _read_tree(entry["tree"]),
            tuple(
                RuntimeLeafBinding(tuple(b["path"]), b["type"], b["kind"])
                for b in entry["bindings"]
            ),
        )
        for entry in metadata.pop("trees")
    )
    abi = kernel_abi_from_dict(metadata.pop("abi"))
    backend_key = metadata.pop("backend_key")
    guarded_fallback = metadata.pop("guarded_fallback", False)
    execution_args = ExecutionArgs(
        kernel_abi=abi,
        argument_trees=trees,
        abi_packer=_prepare_abi_packer(abi, expected_entrypoint=metadata["entrypoint"]),
    )
    module = JitModule(
        kernel_binary_path=directory / "kernel.o",
        execution_args=execution_args,
        print_metadata=None,
        **metadata,
    )
    artifacts = JitFunctionArtifacts(
        MLIR=(directory / "tla.mlir").read_text(),
        LLVM=(directory / "lowered.mlir").read_text(),
        BINARY=directory / "kernel.o",
        cache_key=backend_key,
        cache_dir=directory,
        hivmc_path=None,
    )
    fn = JitCompiledFunction(jit_module=module, artifacts=artifacts)
    fn._attention_guarded_fallback = guarded_fallback
    # Materialize the device owner while the caller holds the disk read lock.
    # A concurrent persistent clear may then unlink files without invalidating
    # a function already returned to a launch in another thread/process.
    fn._get_executor()
    return fn
