"""FA4 input checks, backend initialization and attention routing.

Routing is available without CATLASS. Native dependencies and graph operators
are initialized only when a DSL API is requested or a Flex call is dispatched.
"""

import importlib
from importlib import metadata
import re
import sys

import torch

_RUNTIME_CHECKED = False
_BACKEND_READY = False

_INSTALL = (
    " Install matching PyTorch/torch_npu and ascend-catlass-dsl>=2.1 wheels "
    "for Python 3.10–3.13, then source the CANN environment. "
    "See flash_attn_npu_4/catlass_dsl/README.md for installation commands."
)


def require_fp8_inputs(q, k, v):
    if any(not isinstance(t, torch.Tensor) for t in (q, k, v)):
        raise TypeError("quantized q/k/v must be tensors")
    if any(t.dtype != torch.float8_e4m3fn for t in (q, k, v)):
        raise ValueError("quantized q/k/v must all have dtype torch.float8_e4m3fn")


def require_mxfp8_scales(scales):
    from .. import TensorwiseScales

    if scales is None:
        raise ValueError("MXFP8 requires scales=TensorwiseScales(q=..., k=..., v=...)")
    if not isinstance(scales, TensorwiseScales):
        raise TypeError("scales must be a TensorwiseScales instance")
    if any(t is None for t in scales):
        raise ValueError("MXFP8 requires all three tensors in scales.q, scales.k and scales.v")


def _release(distribution):
    try:
        value = metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise ImportError(f"FA4 requires {distribution}." + _INSTALL) from exc
    match = re.match(r"^(\d+)\.(\d+)", value)
    if match is None:
        raise ImportError(
            f"Cannot determine the FA4 runtime version of {distribution}: {value!r}." + _INSTALL
        )
    return tuple(map(int, match.groups())), value


def check_runtime():
    """Validate requirements for the native DSL backend only."""
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        raise ImportError("FA4 with CATLASS DSL requires Python 3.10 through 3.13." + _INSTALL)
    torch_release, torch_version = _release("torch")
    if torch_release < (2, 9):
        raise ImportError(f"FA4 requires PyTorch >=2.9; found {torch_version}." + _INSTALL)
    npu_release, npu_version = _release("torch_npu")
    if npu_release != torch_release:
        raise ImportError(
            f"FA4 requires matching PyTorch/torch_npu major.minor versions; "
            f"found torch {torch_version} and torch_npu {npu_version}." + _INSTALL
        )
    catlass_release, catlass_version = _release("ascend-catlass-dsl")
    if catlass_release < (2, 1):
        raise ImportError(
            f"FA4 requires ascend-catlass-dsl>=2.1; found {catlass_version}." + _INSTALL
        )
    try:
        importlib.import_module("catlass.tla")
    except (ImportError, OSError, RuntimeError) as exc:
        raise ImportError(
            "FA4 cannot import catlass.tla. Install a built ascend-catlass-dsl>=2.1 "
            "package and source the matching CANN environment. "
            "See flash_attn_npu_4/catlass_dsl/README.md."
        ) from exc
    try:
        importlib.import_module("torch_npu")
    except (ImportError, OSError, RuntimeError) as exc:
        raise ImportError(
            "FA4 cannot import torch_npu. Install a torch_npu build matching PyTorch "
            "and source the CANN environment; see flash_attn_npu_4/catlass_dsl/README.md."
        ) from exc


def get_tla():
    """Check native dependencies once, including for direct helper imports."""
    global _RUNTIME_CHECKED
    if not _RUNTIME_CHECKED:
        check_runtime()
        _RUNTIME_CHECKED = True
    return importlib.import_module("catlass.tla")


@torch._dynamo.assume_constant_result
def ensure_backend():
    """Initialize process-wide DSL state before tracing its first operation.

    No Tensor data or shapes are involved. Dynamo evaluates this initialization
    in Python, so a cold Flex call can register operators without tracing the
    dependency checks or registration code into the attention graph.
    """
    global _BACKEND_READY
    if not _BACKEND_READY:
        get_tla()
        importlib.import_module(".graph", __package__)
        importlib.import_module(".interface", __package__)
        _BACKEND_READY = True


def _has_aux(value, name):
    if value is None:
        return False
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a list, tuple or None")
    return len(value) != 0


def has_flex(
    score_mod=None,
    score_mod_bwd=None,
    mask_mod=None,
    block_sparse_tensors=None,
    block_sparse_tensors_bwd=None,
    aux_tensors=None,
    aux_scalars=None,
):
    # Only Python presence/container length is inspected, never Tensor values.
    tensor_aux = _has_aux(aux_tensors, "aux_tensors")
    scalar_aux = _has_aux(aux_scalars, "aux_scalars")
    return (
        tensor_aux
        or scalar_aux
        or any(
            value is not None
            for value in (
                score_mod,
                score_mod_bwd,
                mask_mod,
                block_sparse_tensors,
                block_sparse_tensors_bwd,
            )
        )
    )


class _VarlenOutputs(torch.autograd.Function):
    """Preserve the legacy four-output ABI without a slice-backward branch."""

    @staticmethod
    def forward(ctx, out, lse):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(out)
        return (
            out,
            lse,
            out.new_empty(0, dtype=torch.float32),
            out.new_empty(0, dtype=torch.float32),
        )

    @staticmethod
    def backward(ctx, dout, dlse, dout_accum, dlse_accum):
        if dout is None and (dout_accum is not None or dlse_accum is not None):
            (out,) = ctx.saved_tensors
            # Keep accumulation-only losses connected with a zero gradient.
            # Broadcast a scalar so NPU Inductor never needs to fill a buffer
            # using a dynamic numel; the attention backward materializes it.
            dout = out.new_zeros(()).expand_as(out)
        return dout, dlse


def attention(
    q,
    k,
    v,
    *,
    varlen,
    window_size,
    num_splits,
    return_lse,
    scheduler_metadata=None,
    disable_scheduler_metadata=False,
    scales=None,
    **options,
):
    ensure_backend()
    if scheduler_metadata is not None:
        raise NotImplementedError("scheduler_metadata is not supported by FA4 Flex attention")
    if disable_scheduler_metadata is not False:
        raise NotImplementedError("FA4 Flex attention requires disable_scheduler_metadata=False")
    if type(num_splits) is not int or num_splits not in (0, 1):
        raise NotImplementedError("FA4 Flex attention supports only num_splits=0 or 1")
    if not isinstance(window_size, (tuple, list)) or len(window_size) != 2:
        raise ValueError("window_size must contain (left, right)")
    if any(bound is not None and type(bound) is not int for bound in window_size):
        raise ValueError("window bounds must be integers or None")
    # FA4 treats each negative bound independently as unlimited. Native DSL
    # also supports signed bounds, so normalize at this compatibility boundary.
    window = tuple(None if bound is None or bound < 0 else bound for bound in window_size)
    from . import interface

    if scales is not None:
        entry = (
            interface.flash_attn_varlen_func_with_quant
            if varlen
            else interface.flash_attn_func_with_quant
        )
        options["scales"] = scales
    else:
        entry = interface.flash_attn_varlen_func if varlen else interface.flash_attn_func
    out, lse = entry(
        q,
        k,
        v,
        window_size=window,
        num_splits=1,
        return_lse=return_lse,
        **options,
    )
    if not return_lse:
        return out
    if not varlen:
        return out, lse
    # Differentiable empty views create a full-size slice-backward/add branch.
    # torch_npu 2.9 Inductor freezes its numel at the first dynamic input shape.
    return _VarlenOutputs.apply(out, lse)
