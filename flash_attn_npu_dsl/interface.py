"""Fixed-length BSND attention on Ascend 950."""

import math
import os
from collections import OrderedDict
from threading import Lock

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

from .block_sparsity import validate_block_sparse
from .flash_fwd import flex_attention_kernel
from .modifiers import softcap_score_mod, validate_modifiers

_compiled_kernels = OrderedDict()
_compile_lock = Lock()


class _DynamicTensorArgument:
    """Bind a contiguous Torch tensor as a dynamic (rows, columns) GM root.

    The owner is held through launch. Columns are fixed by the compiled BSND
    specification; rows and the address are read from the current tensor.
    """

    __slots__ = ("tensor", "columns")

    def __init__(self, tensor, columns):
        self.tensor = tensor
        self.columns = columns

    def compile_sample(self):
        return from_dlpack(
            self.tensor.reshape(-1, self.columns), layout_tag=tla.arch.RowMajor
        ).mark_compact_shape_dynamic(0)

    def build_memref_launch_fields(self):
        address = self.tensor.data_ptr()
        if not address:
            raise RuntimeError("Tensor buffer is not bound")
        columns = self.columns
        rows = self.tensor.numel() // columns
        # Canonical GM fields: allocated/aligned/offset, sizes[4], strides[4],
        # origin[2]. The TLA RowMajor view uses (columns, 1), including one-row
        # views, regardless of DLPack's singleton-stride normalization.
        return (address, address, 0, rows, columns, 1, 1, columns, 1, 1, 1, rows, columns)


def _compilation_key(
    tensors, aux_tensors, aux_scalars, constants, device, *, dynamic_lengths=False
):
    """Build a cache key from device, Tensor layouts and compile-time constants.

    Tensor addresses/contents and auxiliary scalar values are excluded; auxiliary
    scalar types are included. Callback code and captured static values must
    remain unchanged during reuse.
    """

    def tensor_spec(tensor):
        return (
            (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
            if tensor is not None
            else None
        )

    return (
        str(device),
        tuple(
            (tensor.ndim, tensor.dtype)
            if dynamic_lengths and tensor is not None
            else tensor_spec(tensor)
            for tensor in tensors
        ),
        None if aux_tensors is None else tuple(map(tensor_spec, aux_tensors)),
        None if aux_scalars is None else tuple(map(type, aux_scalars)),
        tuple((type(value), value.hex() if type(value) is float else value) for value in constants),
    )


def _compile_kernel(kernel, key, compile_args):
    """Reuse only the executable; every call supplies fresh launch arguments."""
    truthy = {"1", "true", "yes", "on", "y"}
    cache_enabled = os.getenv("CATLASS_DSL_CACHE", "1").strip().lower() in truthy
    force_recompile = os.getenv("CATLASS_DSL_FORCE_RECOMPILE", "0").strip().lower() in truthy
    if not cache_enabled or force_recompile:
        return tla.compile(kernel, *compile_args(), options="--npu-arch 3510")
    key = (kernel.fn, key)
    with _compile_lock:
        if key not in _compiled_kernels:
            _compiled_kernels[key] = tla.compile(kernel, *compile_args(), options="--npu-arch 3510")
            if len(_compiled_kernels) > 128:
                _compiled_kernels.popitem(last=False)
        _compiled_kernels.move_to_end(key)
        return _compiled_kernels[key]


def _validate_aux(aux_tensors, aux_scalars, device):
    import torch

    if aux_tensors is not None and not isinstance(aux_tensors, (tuple, list)):
        raise TypeError("aux_tensors must be None, a tuple or a list of tensors")
    if aux_scalars is not None and not isinstance(aux_scalars, (tuple, list)):
        raise TypeError("aux_scalars must be None, a tuple or a list of scalars")
    for index, tensor in enumerate(aux_tensors or ()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"aux_tensors[{index}] must be a Tensor")
        if tensor.device != torch.device(device) or not tensor.is_contiguous():
            raise ValueError(f"aux_tensors[{index}] must be contiguous on {device}")
        if tensor.requires_grad:
            raise ValueError("auxiliary tensors are read-only inference inputs")
    for index, value in enumerate(aux_scalars or ()):
        if not isinstance(value, (bool, int, float, tla.Numeric)):
            raise TypeError(f"aux_scalars[{index}] must be a scalar")
    return (
        None if aux_tensors is None else list(aux_tensors),
        tuple(aux_scalars) if aux_scalars else None,
    )


def _validate_inputs(q, k, v):
    import torch

    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            raise ValueError(f"{name} must be a rank-4 BSND Tensor")
        if tensor.device.type != "npu" or tensor.device != q.device:
            raise ValueError("q, k and v must be on the same NPU")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("q, k and v must use FP16 or BF16")
        if tensor.dtype != q.dtype:
            raise TypeError("q, k and v must have the same dtype")
        if tensor.shape[-1] not in (8, 16, 32, 64, 80, 96, 128) or any(
            size <= 0 for size in tensor.shape
        ):
            raise ValueError(
                "head dimension must be 8, 16, 32, 64, 80, 96 or 128 and dimensions nonempty"
            )
        if tensor.requires_grad:
            raise ValueError("this implementation supports inference forward only")
    if k.shape != v.shape or q.shape[0] != k.shape[0]:
        raise ValueError("k/v shapes and q/k batch sizes must match")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("Q, K and V head dimensions must match")
    if q.shape[2] % k.shape[2]:
        raise ValueError("Q head count must be divisible by KV head count")


def _resolve_window(causal, window_size, mask_mod):
    left, right = window_size
    if mask_mod is not None:
        return None, None
    left = None if left is None or left < 0 else left
    right = None if right is None or right < 0 else right
    if causal:
        right = 0
    return left, right


def flash_attn_func(
    q,
    k,
    v,
    qv=None,
    gather_kv_indices=None,
    softmax_scale=None,
    causal=False,
    window_size=(None, None),
    learnable_sink=None,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    score_mod=None,
    score_mod_bwd=None,
    mask_mod=None,
    aux_tensors=None,
    aux_scalars=None,
    block_sparse_tensors=None,
    block_sparse_tensors_bwd=None,
    return_lse=False,
):
    """Return (O, LSE), with LSE=None when not requested.

    A mask alone applies to all KV blocks; sparse execution requires explicit
    block metadata. Modifiers consume scaled scores and Q-head indices.
    Unsupported options must retain their defaults in this forward subset.
    """
    import torch

    for name, supported in (
        ("qv", qv is None),
        ("gather_kv_indices", gather_kv_indices is None),
        ("learnable_sink", learnable_sink is None),
        ("num_splits", num_splits == 1),
        ("pack_gqa", pack_gqa is None),
        ("deterministic", deterministic is False),
        ("score_mod_bwd", score_mod_bwd is None),
        ("block_sparse_tensors_bwd", block_sparse_tensors_bwd is None),
    ):
        if not supported:
            raise NotImplementedError(f"{name} is not supported by this forward implementation")
    _validate_inputs(q, k, v)
    softcap = float(softcap)
    if not math.isfinite(softcap) or softcap < 0:
        raise ValueError("softcap must be finite and nonnegative")
    if softcap:
        if score_mod is not None:
            raise ValueError("softcap and score_mod cannot be used together")
        score_mod = softcap_score_mod(softcap)
    window_left, window_right = _resolve_window(causal, window_size, mask_mod)
    batch, q_len, q_heads, head_dim = q.shape
    kv_len, kv_heads = k.shape[1:3]
    scale = float(softmax_scale) if softmax_scale is not None else 1 / math.sqrt(head_dim)
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")
    mask_uses_simd, score_uses_simd = validate_modifiers(mask_mod, score_mod)
    aux_tensors, aux_scalars = _validate_aux(aux_tensors, aux_scalars, q.device)
    with torch.npu.device(q.device):
        # Dense/window execution computes its block interval on device.
        sparse = (
            validate_block_sparse(
                block_sparse_tensors,
                batch_size=batch,
                num_heads_q=q_heads,
                seqlen_q=q_len,
                seqlen_k=kv_len,
                device=q.device,
            )
            if block_sparse_tensors is not None
            else None
        )
        out = torch.empty_like(q)
        lse = (
            torch.empty((batch, q_heads, q_len), device=q.device, dtype=torch.float32)
            if return_lse
            else None
        )
        tensors = (q, k, v, out, lse)
        sparse_tensors = sparse.tensors if sparse is not None else ()
        dynamic_lengths = (
            mask_mod is None
            and score_mod is None
            and sparse is None
            and aux_tensors is None
            and aux_scalars is None
        )

        def bind_tensor(tensor):
            if tensor is None:
                return None
            if dynamic_lengths:
                columns = tensor.shape[-2] * head_dim if tensor.ndim == 4 else 1
                return _DynamicTensorArgument(tensor, columns)
            return from_dlpack(tensor.reshape(-1), layout_tag=tla.arch.RowMajor)

        runtime_args = tuple(bind_tensor(tensor) for tensor in tensors) + (
            tuple(
                from_dlpack(tensor.reshape(-1), layout_tag=tla.arch.RowMajor)
                if tensor is not None
                else None
                for tensor in sparse_tensors
            )
            if sparse is not None
            else None,
            None
            if aux_tensors is None
            else [from_dlpack(tensor, layout_tag=tla.arch.RowMajor) for tensor in aux_tensors],
            aux_scalars,
        )
        constants = (
            batch,
            q_heads,
            kv_heads,
            None if dynamic_lengths else q_len,
            None if dynamic_lengths else kv_len,
            scale,
            q.dtype == torch.float16,
            mask_mod,
            score_mod,
            window_left,
            window_right,
            mask_uses_simd,
            score_uses_simd,
            head_dim,
            sparse.strides if sparse is not None else None,
        )
        key = _compilation_key(
            (*tensors, *sparse_tensors),
            aux_tensors,
            aux_scalars,
            constants,
            q.device,
            dynamic_lengths=dynamic_lengths,
        )

        def compile_args():
            bindings = (
                tuple(arg.compile_sample() if arg is not None else None for arg in runtime_args[:5])
                + runtime_args[5:]
                if dynamic_lengths
                else runtime_args
            )
            return (*bindings, *constants)

        compiled = _compile_kernel(flex_attention_kernel, key, compile_args)
        tasks = batch * q_heads * ((q_len + 127) // 128)
        cores = torch.npu.get_device_properties(q.device.index).cube_core_num
        compiled(*runtime_args, block_num=min(tasks, max(1, int(cores))))
        return out, lse
