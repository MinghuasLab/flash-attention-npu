"""Conservative host checks for the reusable attention launch ABI.

This module never loads an executor or queries an NPU. Unknown compiler schemas
use CATLASS's native descriptor provider rather than guessing a field layout.
"""

from importlib import import_module


ATTENTION_MEMREF_FIELDS = (
    "allocated",
    "aligned",
    "offset",
    "size0",
    "size1",
    "size2",
    "size3",
    "stride0",
    "stride1",
    "stride2",
    "stride3",
    "originShape0",
    "originShape1",
)
_MEMREF_GROUPS = (ATTENTION_MEMREF_FIELDS,) + tuple(
    ("allocated", "aligned", "offset")
    + tuple(f"size{axis}" for axis in range(rank))
    + tuple(f"stride{axis}" for axis in range(rank))
    for rank in range(1, 5)
)


def prepared_attention_binding_supported(executor, *, execution_module=None, tensor_type=None):
    """Accept only the observed schema-v4 canonical descriptor protocol.

    Inspect the compiler-produced ABI on the actual compiled function, including
    functions restored from persistent cache. Method presence alone does not
    establish field ordering. The caller may cache this result on that function;
    it must use the backend-native binding when this function returns False.
    No version string or unrecognized schema is treated as compatible.
    """
    try:
        module = executor.jit_module
        layout = module.execution_args.kernel_abi
        if type(layout.schema_version) is not int or layout.schema_version != 4:
            return False
        if execution_module is None:
            execution_module = import_module("catlass.execution")
        if tensor_type is None:
            tensor_type = import_module("catlass.tla.runtime")._Tensor
        if tuple(execution_module._DYNAMIC_GM_MEMREF_FIELDS) != ATTENTION_MEMREF_FIELDS:
            return False
        if not callable(getattr(tensor_type, "build_memref_launch_fields", None)):
            return False
        if not callable(getattr(tensor_type, "__c_pointers__", None)):
            return False

        current = None
        fields = []
        completed = set()
        for argument in layout.arguments:
            kind = getattr(argument.kind, "value", argument.kind)
            if kind not in ("pointer", "scalar", "memref_field"):
                return False
            if kind != "memref_field":
                if argument.field is not None:
                    return False
                if current is not None:
                    if tuple(fields) not in _MEMREF_GROUPS:
                        return False
                    completed.add(current)
                    current, fields = None, []
                continue
            logical = argument.logical_index
            if type(logical) is not int or logical < 0 or argument.storage_size != 8:
                return False
            if logical != current:
                if current is not None:
                    if tuple(fields) not in _MEMREF_GROUPS:
                        return False
                    completed.add(current)
                if logical in completed:
                    return False
                current, fields = logical, []
            fields.append(argument.field)
        return current is None or tuple(fields) in _MEMREF_GROUPS
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        return False


def validate_mxfp8_head_features(features, *, pipeline, repair_only=False):
    """Validate private compile-time head features before native MX compilation.

    Positive head features retain the baseline optimized strategies. Two zero
    fields select runtime heads only in the general pipeline, where per-batch
    geometry and single-head tile widths must remain enabled.
    """
    q_heads, kv_heads = features.q_heads, features.kv_heads
    if type(q_heads) is not int or type(kv_heads) is not int:
        raise TypeError("MXFP8 head features must be Python integers")
    if min(q_heads, kv_heads) < 0:
        raise ValueError("MXFP8 head features must be nonnegative")
    if (q_heads == 0) != (kv_heads == 0):
        raise ValueError("MXFP8 runtime-head fields must both be zero")
    runtime_heads = q_heads == 0
    if runtime_heads:
        if pipeline != "general" or repair_only:
            raise ValueError("MXFP8 runtime heads require the general pipeline without repair")
        if features.paired_heads or features.single_sequence:
            raise ValueError("MXFP8 runtime heads require unpaired per-batch geometry")
    return runtime_heads
