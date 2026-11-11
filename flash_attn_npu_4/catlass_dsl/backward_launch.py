"""Scoped cooperative launch for the fused-workspace backward kernel.

Its mode-0 rendezvous needs all mixed core groups to be scheduled together.
CATLASS currently exposes no per-call launch configuration, so this adapter
uses the compiler's native payload packer and loaded function handle, then
adds the public ACL batch attribute. Forward and MX launches do not use it.
No installed module or process-wide launch function is modified.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from importlib import import_module


# Public acl_rt.h: aclrtLaunchKernelAttr / aclrtLaunchKernelSchedMode.
_ACL_LAUNCH_ATTR_SCHEM_MODE = 1
_ACL_SCHEM_MODE_BATCH = 1
_ACL_LAUNCH_ATTR_DYN_UBUF_SIZE = 2


class BackwardBackendCompatibilityError(RuntimeError):
    """The installed backend cannot safely launch fused-workspace backward."""


def _unsupported(detail):
    raise BackwardBackendCompatibilityError(
        "FA4 FP16/BF16 training and backward precompilation require the "
        "CATLASS schema-v4 mixed executor with cooperative ACL batch launching: "
        f"{detail}. Forward-only execution retains its separate DSL>=2.1 "
        "requirement. Install a matching supported CATLASS/CANN runtime; "
        "normal-scheduling fallback is unsafe for the fused workspace."
    )


@lru_cache(maxsize=1)
def check_backward_backend_capabilities():
    """Check host interfaces only; never load ACL, a binary, or an NPU context.

    A package version does not establish launch compatibility. Successful host
    capability checks are cached; the actual compiled ABI is checked separately
    for every newly compiled or persistent-loaded executor.
    """
    try:
        execution = import_module("catlass.execution")
        jit = import_module("catlass.base_dsl.jit_executor")
        runtime = import_module("catlass.base_dsl.runtime.ascend")
        streams = import_module("catlass.base_dsl.runtime.ascend_stream_adapter")
    except (ImportError, OSError, AttributeError) as exc:
        try:
            _unsupported(f"missing host launch interfaces ({exc})")
        except BackwardBackendCompatibilityError as error:
            raise error from exc
    required = (
        (jit, "_resolved_dyn_ubuf_bytes"),
        (jit, "JitCompiledFunction"),
        (getattr(jit, "JitCompiledFunction", None), "_get_executor"),
        (getattr(jit, "ExecutionArgs", None), "generate_launch_payload"),
        (runtime, "load_acl"),
        (runtime, "check_acl_errors"),
        (streams, "as_stream"),
        (streams, "current_device"),
        (execution, "TlaExecutionResult"),
    )
    for owner, name in required:
        if not callable(getattr(owner, name, None)):
            _unsupported(f"missing host launch interface {name}")
    return execution, jit, runtime, streams


def validate_backward_executor(executor):
    """Validate a compiled module without calling ``_get_executor``.

    This gate applies equally to baseline and broad fused-workspace backward.
    It is deliberately separate from descriptor-provider selection: an unknown
    descriptor can safely use its native provider, whereas normal scheduling
    cannot safely replace the required cooperative launch.
    """
    check_backward_backend_capabilities()
    module = getattr(executor, "jit_module", None)
    if module is None:
        _unsupported("missing jit_module")
    if (
        getattr(module, "kernel_mode", None) != "mix"
        or getattr(module, "is_mixed", None) is not True
    ):
        _unsupported("kernel must be mixed AIC/AIV")
    execution_args = getattr(module, "execution_args", None)
    schema = getattr(getattr(execution_args, "kernel_abi", None), "schema_version", None)
    if type(schema) is not int or schema != 4:
        _unsupported(f"unsupported compiled kernel ABI schema {schema!r}")
    if not callable(getattr(execution_args, "generate_launch_payload", None)):
        _unsupported("missing native payload packer")
    if not callable(getattr(executor, "_get_executor", None)):
        _unsupported("missing loaded executor accessor")
    for name in ("uses_scalar_print", "uses_tensor_print"):
        flag = getattr(module, name, None)
        if type(flag) is not bool:
            _unsupported(f"unknown {name} metadata")
        if flag:
            _unsupported("debug-print kernels need an unsupported extra launch workspace")
    if (
        getattr(module, "print_metadata", object()) is not None
        or getattr(module, "print_tensor_position", object()) is not None
        or getattr(module, "uses_debug_tunnel", False)
    ):
        _unsupported("debug/tunnel kernels need an unsupported extra launch workspace")
    for name in ("ub_static_bytes", "ub_programmable_bytes", "ub_dynamic_base"):
        if type(getattr(module, name, None)) is not int:
            _unsupported(f"unknown {name} metadata")
    return module


def launch_backward_batch(executor, bound_args, *, block_num, stream):
    """Launch fused-workspace backward using an explicitly checked batch ABI.

    This is the independent mode-0 correctness fix, not a broad-only feature.
    Unsupported SDKs fail before launch; installed modules are never patched.
    """
    if type(block_num) is not int or block_num <= 0:
        raise ValueError("backward batch block_num must be a positive integer")
    module = validate_backward_executor(executor)
    execution_mod, jit, ascend_rt, streams = check_backward_backend_capabilities()
    get_executor = executor._get_executor
    pack = module.execution_args.generate_launch_payload
    current_device = streams.current_device
    as_stream = streams.as_stream
    _resolved_dyn_ubuf_bytes = jit._resolved_dyn_ubuf_bytes
    unsupported = _unsupported

    loaded = get_executor()
    if getattr(loaded, "jit_module", None) is not module:
        unsupported("loaded executor does not own the compiled module")
    device = getattr(loaded, "device", None)
    if type(device) is not int or int(current_device()) != device:
        unsupported("loaded executor belongs to a different current device")
    function_handle = getattr(loaded, "function_handle", None)
    if type(function_handle) is not int or function_handle <= 0:
        unsupported("missing loaded function handle")
    binary_handle = getattr(loaded, "binary_handle", None)
    if type(binary_handle) is not int:
        unsupported("missing loaded binary handle")

    dyn_ubuf_bytes = _resolved_dyn_ubuf_bytes(module, 0)
    raw_stream = as_stream(stream, device=device)
    payload = pack(bound_args)
    if not isinstance(payload, bytes):
        unsupported("native payload packer did not return bytes")
    # Match the native launcher: even an empty payload has a non-null buffer.
    payload = payload or (b"\x00" * ctypes.sizeof(ctypes.c_uint64))
    host_buf = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
    cfg = [{"id": _ACL_LAUNCH_ATTR_SCHEM_MODE, "value": _ACL_SCHEM_MODE_BATCH}]
    if dyn_ubuf_bytes > 0:
        cfg.append({"id": _ACL_LAUNCH_ATTR_DYN_UBUF_SIZE, "value": dyn_ubuf_bytes})
    acl = ascend_rt.load_acl()
    ret = acl.rt.launch_kernel_with_host_args(
        function_handle,
        block_num,
        int(raw_stream),
        cfg,
        ctypes.addressof(host_buf),
        len(payload),
        [],
        0,
    )
    try:
        ascend_rt.check_acl_errors(ret, "acl.rt.launch_kernel_with_host_args (backward batch)")
    except execution_mod.TlaRuntimeUnavailableError as exc:
        if dyn_ubuf_bytes <= 0 or getattr(exc, "ret", None) != ascend_rt.ACL_ERROR_INVALID_PARAM:
            raise
        raise execution_mod.TlaExecutionError(
            f"the runtime refused a backward batch launch declaring {dyn_ubuf_bytes} "
            "bytes of dynamic UB within the compiler's calculated bound"
        ) from exc
    return execution_mod.TlaExecutionResult(
        binary_handle=binary_handle,
        function_handle=function_handle,
        device=device,
    )
