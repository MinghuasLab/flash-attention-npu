# ATK executor for flash_attn_combine.
#
# CPU  : high-precision (float64) golden, then cast to out_dtype.
# NPU  : the shipped torch extension (flash_attn_npu_3 / flash_attn_npu_4).
#
# Both paths return (out, lse); ATK compares them output-by-output.

import importlib
import os
import sys

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


OUT_DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _resolve_dtype(value):
    if value is None:
        return torch.float32
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        return OUT_DTYPES.get(value.lower(), torch.float32)
    # ATK may hand back an enum-ish object; fall back to its string form.
    return OUT_DTYPES.get(str(value).lower(), torch.float32)


def _pick(kwargs, args, name, index, default=None):
    if name in kwargs:
        return kwargs[name]
    if index is not None and len(args) > index:
        return args[index]
    return default


def _to_seqlen_contiguous(tensor):
    """Re-lay lse_partial so its seqlen dim is contiguous.

    A plain transpose+contiguous is a no-op when the seqlen (or head) dim is 1,
    because size-1 strides are ignored by torch's contiguity check. Build an
    explicit (..., H, S) buffer instead so stride(-2) is always 1.
    """
    if tensor.stride(-2) == 1:
        return tensor
    shape = list(tensor.shape)
    buf = torch.empty(shape[:-2] + [shape[-1], shape[-2]], dtype=tensor.dtype, device=tensor.device)
    buf.copy_(tensor.transpose(-1, -2))
    return buf.transpose(-1, -2)


def _reference(out_partial, lse_partial, out_dtype):
    op = out_partial.to(torch.float64)
    lp = lse_partial.to(torch.float64)
    lse = torch.logsumexp(lp, dim=0)
    scale = torch.exp(lp - lse)
    scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
    out = (scale.unsqueeze(-1) * op).sum(0)
    return out.to(out_dtype), lse.to(torch.float32)


def _small_op_combine(out_partial, lse_partial, out_dtype):
    """Second implementation used as the benchmark (small-op composition).

    Mirrors what a GPU/reference implementation would do: the same math
    expressed with logsumexp/exp/mul/sum instead of the fused kernel. Its error
    against the CPU golden is the `Benchmark_*` term of the L2 ratio.
    """
    op = out_partial.to(torch.float32)
    lp = lse_partial.to(torch.float32)
    lse = torch.logsumexp(lp, dim=0)
    weight = torch.exp(lp - lse)
    weight = torch.where(torch.isfinite(weight), weight, torch.zeros_like(weight))
    out = (weight.unsqueeze(-1) * op).sum(0)
    return out.to(out_dtype), lse.to(torch.float32)


def _nv_combine_fn():
    """Locate NVIDIA flash-attention's combine function, if installed.

    FA3 (hopper) exposes it from `flash_attn_interface`; FA4 (cute) from
    `flash_attn.cute.interface`; some builds re-export it from `flash_attn`.
    """
    for module_name in ("flash_attn_interface", "flash_attn.cute.interface", "flash_attn"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        fn = getattr(module, "flash_attn_combine", None)
        if fn is not None:
            return fn, module_name
    return None, None


class _FlashAttnCombineBase(BaseApi):
    module_name = None  # set by subclasses

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        kwargs = dict(getattr(input_data, "kwargs", {}) or {})
        args = list(getattr(input_data, "args", []) or [])

        out_partial = _pick(kwargs, args, "out_partial", 0)
        lse_partial = _pick(kwargs, args, "lse_partial", 1)
        out_dtype = _resolve_dtype(_pick(kwargs, args, "out_dtype", 2, None))

        if out_partial is None or lse_partial is None:
            raise RuntimeError("flash_attn_combine executor: missing tensor inputs")

        if self.device == "cpu":
            return _reference(out_partial.cpu(), lse_partial.cpu(), out_dtype)

        if self.device == "npu":
            # Import lazily so CPU-only nodes do not need torch_npu.
            try:
                module = __import__(self.module_name, fromlist=["flash_attn_combine"])
            except ModuleNotFoundError:
                # The repo ships the extension as an in-place build; allow the
                # ATK worker (launched from another directory) to find it.
                repo_root = os.environ.get("FLASH_ATTN_NPU_ROOT")
                if not repo_root:
                    raise
                sys.path.insert(0, repo_root)
                module = __import__(self.module_name, fromlist=["flash_attn_combine"])
            flash_attn_combine = module.flash_attn_combine

            # ATK materialises lse_partial as a contiguous (NS, B, S, H)
            # tensor, while the interface requires the seqlen dimension to be
            # contiguous (the layout produced by the flash-attention kernels).
            # Re-layout without changing the logical values.
            lse_kernel = _to_seqlen_contiguous(lse_partial.npu())
            out, lse = flash_attn_combine(
                out_partial.npu(), lse_kernel, out_dtype=out_dtype
            )
            return out, lse

        if self.device == "gpu":
            # GPU node = the "second benchmark" of the L2 double-benchmark
            # standard. Prefer NVIDIA's own combine implementation when the
            # package is available; otherwise fall back to the small-op
            # composition (same math expressed with logsumexp/exp/mul/sum).
            lse_kernel = _to_seqlen_contiguous(lse_partial)
            nv_combine, module_name = _nv_combine_fn()
            if nv_combine is not None:
                out, lse = nv_combine(out_partial, lse_kernel, None, out_dtype)
                return out, lse.to(torch.float32)
            return _small_op_combine(out_partial, lse_partial, out_dtype)

        raise RuntimeError(f"Unsupported device: {self.device}")


@register("function_flash_attn_combine")
class FlashAttnCombineApi(_FlashAttnCombineBase):
    """FA3 interface (flash_attn_npu_3)."""

    module_name = "flash_attn_npu_3"


@register("function_flash_attn_combine_v4")
class FlashAttnCombineV4Api(_FlashAttnCombineBase):
    """FA4 interface (flash_attn_npu_4)."""

    module_name = "flash_attn_npu_4"


class _FlashAttnCombineRefBase(BaseApi):
    """Small-op composition benchmark (second benchmark of the L2 standard)."""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        kwargs = dict(getattr(input_data, "kwargs", {}) or {})
        args = list(getattr(input_data, "args", []) or [])
        out_partial = _pick(kwargs, args, "out_partial", 0)
        lse_partial = _pick(kwargs, args, "lse_partial", 1)
        out_dtype = _resolve_dtype(_pick(kwargs, args, "out_dtype", 2, None))
        if out_partial is None or lse_partial is None:
            raise RuntimeError("flash_attn_combine reference: missing tensor inputs")
        return _small_op_combine(out_partial, lse_partial, out_dtype)


@register("function_flash_attn_combine_ref")
class FlashAttnCombineRefApi(_FlashAttnCombineRefBase):
    """FA3 second benchmark (NPU/CPU small-op composition)."""


@register("function_flash_attn_combine_v4_ref")
class FlashAttnCombineV4RefApi(_FlashAttnCombineRefBase):
    """FA4 second benchmark (NPU/CPU small-op composition)."""
