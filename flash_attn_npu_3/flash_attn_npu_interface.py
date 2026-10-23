# Copyright (c) 2023, Tri Dao.
# Modified by Minghua Shen, 2026

from typing import Optional, Union, Tuple

import torch

# isort: off
# We need to import the kernels after importing torch
from . import flash_attn_npu_3  # Registers operators with PyTorch

# isort: on

if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:

    def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func

        if fn is None:
            return wrap
        return fn

    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func

        if fn is None:
            return wrap
        return fn

    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper


def _is_mla(q, v, qv, cu_seqlens_q=None):
    return qv is not None or (
        (cu_seqlens_q is None or q.ndim == 3)
        and q.ndim > 0
        and v.ndim > 0
        and q.shape[-1] != v.shape[-1]
    )


def _check_mla_devices(q, **tensors):
    # Fake tensors expose their logical device in Python, while the native
    # shape checks see their underlying Meta tensors.
    for name, tensor in tensors.items():
        if tensor is not None and not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"MLA: {name} must be a Tensor")
        if tensor is not None and tensor.device != q.device:
            raise RuntimeError(f"MLA: {name} must be on the same NPU device")


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


_LAYOUT_TND = 0
_LAYOUT_BSND = 1
_LAYOUT_BNSD = 2


def _transposed_bnsd_view(t):
    """``t.transpose(1, 2)`` if ``t`` is a contiguous BNSD tensor seen as BSND, else ``None``."""
    if t is None or t.dim() != 4 or t.is_contiguous():
        return None
    _, n1, _, n3 = t.shape
    if t.stride(3) == 1 and t.stride(1) == n3 and t.stride(2) == n1 * n3:
        return t.transpose(1, 2)
    return None


def _detect_layout(q, k, v, page_table, cu_seqlens_k):
    """Return ``(layout, q, k, v, out_transposed)``: the layout code shared with the C++
    host, the operands to hand the kernel, and whether ``out`` must be transposed back to
    match the shape of the query the caller passed in.

    Only a stride match selects BNSD; otherwise rank-4 operands are read as BSND. Paged and
    cu-packed (TND) Q/K/V are never BNSD.
    """
    if k is not None and v is not None and v.dim() != k.dim():
        raise RuntimeError(
            "flash_attn_npu_3: key and value must have the same rank for layout detection"
        )

    if q.dim() != 4 or k.dim() != 4 or page_table is not None or cu_seqlens_k is not None:
        return _LAYOUT_BSND, q, k, v, False

    q_bnsd = _transposed_bnsd_view(q)
    k_bnsd = _transposed_bnsd_view(k)
    v_bnsd = _transposed_bnsd_view(v)
    if q_bnsd is not None and k_bnsd is not None and v_bnsd is not None:
        return _LAYOUT_BNSD, q_bnsd, k_bnsd, v_bnsd, True

    return _LAYOUT_BSND, q, k, v, False


def round_multiple(x, m):
    return (x + m - 1) // m * m


def _check_fwd_dimension(value, name):
    if not 0 <= value <= 2**31 - 1:
        raise RuntimeError(
            f"{name} must be in [0, {2**31 - 1}] for the 32-bit forward scheduler; got {value}"
        )


_HEADDIM_BWD_ALIGN = 64


def _pad_bwd_headdim(dout, q, k, v, out, head_size_og):
    """
    Pad headdim to a multiple of 64 for the bwd kernel.
    """
    q_dtype = q.dtype
    if q_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"mha_bwd only supports FP16 and BF16, got {q_dtype}")
    for name, t in (("k", k), ("v", v), ("out", out), ("dout", dout)):
        if t.dtype != q_dtype:
            raise ValueError(
                f"mha_bwd: q/k/v/out/dout must have the same dtype, "
                f"got q={q_dtype}, {name}={t.dtype}"
            )
    if dout.size(-1) != head_size_og:
        raise ValueError(
            f"dout headdim ({dout.size(-1)}) must equal original q/k/v headdim ({head_size_og})"
        )
    qkv_out_headdims = [t.size(-1) for t in (q, k, v, out)]
    if len(set(qkv_out_headdims)) != 1:
        raise ValueError(
            f"q/k/v/out must share the same headdim, got {qkv_out_headdims} "
            "(unequal headdim is not supported)"
        )
    ctx_headdim = qkv_out_headdims[0]
    if ctx_headdim <= 0 or ctx_headdim > 256:
        raise ValueError(f"headdim must be in (0, 256], got {ctx_headdim} ")
    target = round_multiple(ctx_headdim, _HEADDIM_BWD_ALIGN)

    def _pad(t):
        cur = t.size(-1)
        if cur == target:
            return t
        if cur > target:
            raise ValueError(f"headdim {cur} > pad target {target}")
        return torch.nn.functional.pad(t, [0, target - cur])

    return _pad(dout), _pad(q), _pad(k), _pad(v), _pad(out), head_size_og


def _validate_varlen_query(q, cu_seqlens_q):
    if q.dim() != 3:
        raise RuntimeError(f"q must be 3-dimensional; got {q.dim()} dimensions")
    if cu_seqlens_q.numel() <= 1:
        raise RuntimeError("batch size must be positive")
    if q.shape[0] == 0:
        raise RuntimeError("total_q must be positive (q must contain at least one query token)")


def round_up_headdim(head_size: int) -> int:
    return round_multiple(head_size, _HEADDIM_BWD_ALIGN)


def _check_mla_forward(q, k, v, **kwargs):
    _check_mla_devices(
        q,
        k=k,
        v=v,
        qv=kwargs.get("qv"),
        cu_seqlens_q=kwargs.get("cu_seqlens_q"),
        cu_seqlens_k=kwargs.get("cu_seqlens_k"),
        seqused_q=kwargs.get("seqused_q"),
        cache_seqlens=kwargs.get("seqused_k"),
        page_table=kwargs.get("page_table"),
        scheduler_metadata=kwargs.get("scheduler_metadata"),
        q_descale=kwargs.get("q_descale"),
        k_descale=kwargs.get("k_descale"),
        v_descale=kwargs.get("v_descale"),
    )
    if q.device.type != "npu":
        raise RuntimeError("MLA: inputs must be NPU tensors")
    if any(
        t is not None and t.requires_grad
        for t in (
            q,
            k,
            v,
            *(kwargs.get(name) for name in ("qv", "q_descale", "k_descale", "v_descale")),
        )
    ):
        raise RuntimeError("MLA forward does not support gradients")


def _mla_native_fake(op, reference, *args):
    from torch._subclasses.fake_tensor import FakeTensor, in_kernel_invocation_manager

    mode, device = reference.fake_mode, reference.device
    # Reuse native validation with the original tensors' Meta storage and strides.
    with in_kernel_invocation_manager(mode):
        outputs = op(*args)
    if isinstance(outputs, torch.Tensor):
        return FakeTensor(mode, outputs, device)
    return tuple(FakeTensor(mode, output, device) for output in outputs)


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_new: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    kv_batch_idx: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    seqlens_rotary: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    layout: int = _LAYOUT_BSND,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _is_mla(q, v, qv, cu_seqlens_q):
        q, k, k_new, v_new = [maybe_contiguous(x) for x in (q, k, k_new, v_new)]
        v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v
        cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new = [
            maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new)
        ]
        seqused_q, seqused_k = [maybe_contiguous(x) for x in (seqused_q, seqused_k)]
        page_table, kv_batch_idx, leftpad_k = [
            maybe_contiguous(x) for x in (page_table, kv_batch_idx, leftpad_k)
        ]
        rotary_cos, rotary_sin = [maybe_contiguous(x) for x in (rotary_cos, rotary_sin)]
        seqlens_rotary = maybe_contiguous(seqlens_rotary)
    out, softmax_lse, out_accum, softmax_lse_accum = flash_attn_npu_3.fwd(
        q,
        k,
        v,
        k_new,
        v_new,
        qv,
        out_,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        attention_chunk,
        softcap,
        rotary_interleaved,
        scheduler_metadata,
        num_splits,
        pack_gqa,
        sm_margin,
        layout,
    )

    if out_accum is None:
        out_accum = torch.tensor([], device=out.device)

    if softmax_lse_accum is None:
        softmax_lse_accum = torch.tensor([], device=out.device)

    # Native returns own storage except for out_. Preserve the registered
    # operator's alias rejection after native validation and execution.
    if out_ is not None and hasattr(torch.library, "infer_schema"):
        raise RuntimeError("_flash_attn_forward: outputs may not alias any inputs or other returns")
    return out, softmax_lse, out_accum, softmax_lse_accum


def _flash_attn_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_new: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    kv_batch_idx: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    seqlens_rotary: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    layout: int = _LAYOUT_BSND,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Symbolic fake implementation of flash attention forward.
    Returns tensors with the correct shapes and dtypes without actual computation.
    """

    if _is_mla(q, v, qv, cu_seqlens_q):
        _check_mla_forward(
            q,
            k,
            v,
            qv=qv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            page_table=page_table,
            scheduler_metadata=scheduler_metadata,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return _mla_native_fake(
            flash_attn_npu_3.fwd,
            q,
            q,
            k,
            v,
            k_new,
            v_new,
            qv,
            out_,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_seqlens_k_new,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            page_table,
            kv_batch_idx,
            leftpad_k,
            rotary_cos,
            rotary_sin,
            seqlens_rotary,
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            attention_chunk,
            softcap,
            rotary_interleaved,
            scheduler_metadata,
            num_splits,
            pack_gqa,
            sm_margin,
            layout,
        )

    # Determine if we're in varlen mode
    is_varlen_q = cu_seqlens_q is not None
    if is_varlen_q:
        _validate_varlen_query(q, cu_seqlens_q)

    # Get dimensions from query tensor
    if is_varlen_q:
        # varlen mode: q is (total_q, num_heads, head_size)
        total_q, num_heads, head_size = q.shape
        batch_size = cu_seqlens_q.shape[0] - 1

        if max_seqlen_q is None:
            raise ValueError("max_seqlen_q must be provided if cu_seqlens_q is provided")
        seqlen_q = max_seqlen_q
    else:
        # batch mode: q is (batch_size, seqlen_q, num_heads, head_size) in BSND, or
        # (batch_size, num_heads, seqlen_q, head_size) in BNSD.
        if layout == _LAYOUT_BNSD:
            batch_size, num_heads, seqlen_q, head_size = q.shape
            total_q = batch_size * q.shape[2]
        else:
            batch_size, seqlen_q, num_heads, head_size = q.shape
            total_q = batch_size * q.shape[1]

    _check_fwd_dimension(batch_size, "batch_size")
    _check_fwd_dimension(seqlen_q, "max_seqlen_q" if is_varlen_q else "seqlen_q")

    # Get value head dimension
    head_size_v = v.shape[-1]

    # Determine output dtype (FP8 inputs produce BF16 outputs)
    q_type = q.dtype
    if q_type == torch.float8_e4m3fn:
        out_dtype = torch.bfloat16
    else:
        out_dtype = q_type

    # Create output tensor
    if out_ is not None:
        # If out_ is provided, _flash_attn_forward becomes non-functional
        raise TypeError(
            "Tracing (torch.compile/torch.export) with pre-allocated output tensor is not supported."
        )

    if is_varlen_q:
        out = torch.empty((total_q, num_heads, head_size_v), dtype=out_dtype, device=q.device)
    elif layout == _LAYOUT_BNSD:
        out = torch.empty(
            (batch_size, num_heads, seqlen_q, head_size_v), dtype=out_dtype, device=q.device
        )
    else:
        out = torch.empty(
            (batch_size, seqlen_q, num_heads, head_size_v), dtype=out_dtype, device=q.device
        )

    # Create softmax_lse tensor
    if is_varlen_q:
        softmax_lse = torch.empty((num_heads, total_q), dtype=torch.float32, device=q.device)
    else:
        softmax_lse = torch.empty(
            (batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device
        )

    # TODO(guilhermeleobas): Implement "get_num_splits"
    # There's an heuristic to compute num_splits when "num_splits <= 0"
    # assert that num_splits is > 0 for now
    if num_splits <= 0:
        raise ValueError(
            f"tracing (torch.compile/torch.export) with num_splits <= 0 not supported. Got {num_splits=}"
        )

    if num_splits > 1:
        if is_varlen_q:
            out_accum = torch.empty(
                (num_splits, num_heads, total_q, head_size_v), dtype=torch.float32, device=q.device
            )
            softmax_lse_accum = torch.empty(
                (num_splits, num_heads, total_q), dtype=torch.float32, device=q.device
            )
        else:
            out_accum = torch.empty(
                (num_splits, batch_size, num_heads, seqlen_q, head_size_v),
                dtype=torch.float32,
                device=q.device,
            )
            softmax_lse_accum = torch.empty(
                (num_splits, batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device
            )
    else:
        # Tensors are not set when num_splits < 1
        out_accum = torch.tensor([], device=out.device)
        softmax_lse_accum = torch.tensor([], device=out.device)

    return out, softmax_lse, out_accum, softmax_lse_accum


@_torch_custom_op_wrapper(
    "flash_attn_npu_3::_flash_attn_backward", mutates_args=("dq", "dk", "dv"), device_types="npu"
)
def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    sequed_q: Optional[torch.Tensor] = None,
    sequed_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    (
        dq,
        dk,
        dv,
        softmax_d,
    ) = flash_attn_npu_3.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        sequed_q,
        sequed_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        sm_margin,
    )
    return softmax_d


@_torch_register_fake_wrapper("flash_attn_npu_3::_flash_attn_backward")
def _flash_attn_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    sequed_q: Optional[torch.Tensor] = None,
    sequed_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
) -> torch.Tensor:

    is_varlen_q = cu_seqlens_q is not None
    is_varlen_k = cu_seqlens_q is not None
    is_varlen = is_varlen_q or is_varlen_k or sequed_q is not None or sequed_k is not None

    if not is_varlen_q:
        batch_size = q.size(0)
        seqlen_q = q.size(1)
        seqlen_k = k.size(1)
        total_q = batch_size * q.size(1)
    else:
        batch_size = cu_seqlens_q.size(0) - 1
        total_q = q.size(0)
        seqlen_q = max_seqlen_q
        seqlen_k = max_seqlen_k

    if window_size_left >= seqlen_k - 1:
        window_size_left = -1

    if window_size_right >= seqlen_q - 1:
        window_size_right = -1

    if is_causal:
        window_size_right = 0

    is_causal = window_size_left < 0 and window_size_right == 0

    kBlockM = 128

    num_heads = q.shape[-2]
    seqlen_q_rounded = round_multiple(seqlen_q, kBlockM)

    total_q_padded_rounded = round_multiple(total_q + batch_size * kBlockM, kBlockM)

    dq = torch.empty_like(q) if dq is None else dq
    dk = torch.empty_like(k) if dk is None else dk
    dv = torch.empty_like(v) if dv is None else dv

    if not is_varlen:
        softmax_d = torch.empty(
            (batch_size, num_heads, seqlen_q_rounded), dtype=torch.float32, device=q.device
        )
    else:
        softmax_d = torch.empty(
            (num_heads, total_q_padded_rounded), dtype=torch.float32, device=q.device
        )

    return softmax_d


# Reuse custom_op defaults from the unchanged backward registration; tags
# differ across torch versions. Keep the existing schema and fake dispatch.
if hasattr(torch.library, "infer_schema"):
    torch.library.define(
        "flash_attn_npu_3::_flash_attn_forward",
        torch.library.infer_schema(_flash_attn_forward, mutates_args=()),
        tags=torch.ops.flash_attn_npu_3._flash_attn_backward.default.tags,
    )
    torch.library.impl("flash_attn_npu_3::_flash_attn_forward", "npu", _flash_attn_forward)

    def _flash_attn_forward_no_backward(ctx, *grads):
        # Training uses the existing outer autograd.Function; the cache and
        # low-level forward APIs have no backward formula.
        raise RuntimeError(
            "Trying to backward through flash_attn_npu_3._flash_attn_forward.default "
            "but no autograd formula was registered. Please use register_autograd to add one."
        )

    torch.library.register_autograd(
        "flash_attn_npu_3::_flash_attn_forward", _flash_attn_forward_no_backward
    )
    _flash_attn_forward = torch.ops.flash_attn_npu_3._flash_attn_forward.default
else:
    _flash_attn_forward = _torch_custom_op_wrapper(
        "flash_attn_npu_3::_flash_attn_forward",
        _flash_attn_forward,
        mutates_args=(),
        device_types="npu",
    )


_torch_register_fake_wrapper("flash_attn_npu_3::_flash_attn_forward")(_flash_attn_forward_fake)


def setup_context(ctx, inputs, output):
    q, k, v = inputs[:3]
    out, softmax_lse, _, _ = output
    ctx.save_for_backward(q, k, v, out, softmax_lse)
    ctx.head_size_og = q.size(-1)
    ctx.softmax_scale = inputs[-11]
    ctx.causal = inputs[-10]
    ctx.window_size = [inputs[-9], inputs[-8]]
    ctx.attention_chunk = inputs[-7]
    ctx.softcap = inputs[-6]
    ctx.sm_margin = inputs[-1]


def _backward(ctx, dout, *grads):
    q, k, v, out, softmax_lse = ctx.saved_tensors
    dout, q, k, v, out, head_size_og = _pad_bwd_headdim(dout, q, k, v, out, ctx.head_size_og)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        None,
        None,  # cu_seqlens_q, cu_seqlens_k,
        None,
        None,  # sequed_q, sequed_k,
        None,
        None,  # max_seqlen_q, max_seqlen_k,
        dq,
        dk,
        dv,
        ctx.softmax_scale,
        ctx.causal,
        ctx.window_size[0],
        ctx.window_size[1],
        ctx.softcap,
        False,  # deterministic
        ctx.sm_margin,
    )
    dq = dq[..., :head_size_og]
    dk = dk[..., :head_size_og]
    dv = dv[..., :head_size_og]
    return dq, dk, dv, *((None,) * 21)


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        softmax_scale,
        causal,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        deterministic=False,
        num_heads_q=None,
        sm_margin=0,
        return_softmax=False,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        if qkv.dim() == 5:
            assert qkv.shape[-3] == 3
            q, k, v = qkv.unbind(dim=-3)
        else:
            assert qkv.dim() == 4
            assert num_heads_q is not None
            num_heads_k = (qkv.shape[2] - num_heads_q) // 2
            assert num_heads_k * 2 + num_heads_q == qkv.shape[2]
            q, k, v = qkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)
        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None,
            None,  # k_new, v_new
            None,  # qv
            None,  # out
            None,
            None,
            None,  # cu_seqlens_q/k/k_new
            None,
            None,  # seqused_q/k
            None,
            None,  # max_seqlen_q/k
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k,
            None,
            None,
            None,  # rotary_cos/sin, seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            attention_chunk=attention_chunk,
            softcap=softcap,
            sm_margin=sm_margin,
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.ndim = qkv.dim()
        ctx.sm_margin = sm_margin
        ctx.head_size_og = q.size(-1)
        return (out, softmax_lse) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
        dout, q, k, v, out, head_size_og = _pad_bwd_headdim(dout, q, k, v, out, ctx.head_size_og)
        if ctx.ndim == 5:
            qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
            dq, dk, dv = dqkv.unbind(dim=-3)
        else:
            num_heads_q = q.shape[2]
            num_heads_k = k.shape[2]
            qkv_shape = q.shape[:-2] + (num_heads_q + num_heads_k * 2, *q.shape[-1:])
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
            dq, dk, dv = dqkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None,
            None,  # cu_seqlens_q, cu_seqlens_k,
            None,
            None,  # sequed_q, sequed_k,
            None,
            None,  # max_seqlen_q, max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
        )
        dqkv = dqkv[..., :head_size_og]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_softmax=False,
    ):
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
        scheduler_metadata = None
        layout, out_transposed = _LAYOUT_BSND, False
        if not _is_mla(q, v, qv):
            # Training interfaces do not expose scheduler_metadata (aligned with
            # official flash-attn); the scheduler metadata is computed internally on
            # the AICPU so no D2H/H2D sync breaks the pipeline.
            batch_size, seqlen_q, num_heads, head_size = q.shape
            seqlen_k, num_heads_k = k.shape[1], k.shape[2]
            # get_scheduler_metadata only takes numbers and reads them as BSND, so the
            # sizes above come from the pre-detection shapes; BNSD only re-views the
            # same storage.
            layout, q, k, v, out_transposed = _detect_layout(q, k, v, None, None)
            cache_seqlens = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=q.device)
            scheduler_metadata = get_scheduler_metadata(
                batch_size,
                seqlen_q,
                seqlen_k,
                num_heads,
                num_heads_k,
                head_size,
                cache_seqlens,
                qkv_dtype=q.dtype,
                causal=causal,
                window_size=window_size,
                has_softcap=softcap > 0.0,
                num_splits=num_splits,
                sm_margin=sm_margin,
            )
        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None,
            None,  # k_new, v_new
            qv,  # qv
            None,  # out
            None,
            None,
            None,  # cu_seqlens_q/k/k_new
            None,
            None,  # seqused_q/k
            None,
            None,  # max_seqlen_q/k
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k,
            None,
            None,
            None,  # rotary_cos/sin, seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            attention_chunk=attention_chunk,
            softcap=softcap,
            scheduler_metadata=scheduler_metadata,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
            layout=layout,
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        ctx.head_size_og = q.size(-1)
        if out_transposed:
            # Report the result in the shape the caller passed in, not the BNSD view.
            out = out.transpose(1, 2)
        return (out, softmax_lse) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
        dout, q, k, v, out, head_size_og = _pad_bwd_headdim(dout, q, k, v, out, ctx.head_size_og)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None,
            None,  # cu_seqlens_q, cu_seqlens_k,
            None,
            None,  # sequed_q, sequed_k,
            None,
            None,  # max_seqlen_q, max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
        )
        dq = dq[..., :head_size_og]  # We could have padded the head dimension
        dk = dk[..., :head_size_og]
        dv = dv[..., :head_size_og]
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_softmax=False,
    ):
        is_mla = _is_mla(q, v, qv, cu_seqlens_q)
        if not is_mla:
            _validate_varlen_query(q, cu_seqlens_q)
        # The packed varlen query (cu_seqlens_q present) is always TND, never BNSD.
        layout = _LAYOUT_TND
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
        scheduler_metadata = None
        if not is_mla:
            # Training interfaces do not expose scheduler_metadata (aligned with
            # official flash-attn); the scheduler metadata is computed internally on
            # the AICPU so no D2H/H2D sync breaks the pipeline. cache_seqlens holds
            # the per-batch KV lengths and stays on device.
            batch_size = cu_seqlens_q.shape[0] - 1
            num_heads, head_size = q.shape[1], q.shape[2]
            num_heads_k = k.shape[1]
            if seqused_k is not None:
                cache_seqlens = seqused_k
            else:
                cache_seqlens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            if seqused_q is not None:
                seqlens_q = seqused_q
            else:
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            # Pass the per-batch used lengths directly to the internal scheduler.
            scheduler_metadata = _get_scheduler_metadata_op(
                batch_size,
                max_seqlen_q,
                max_seqlen_k,
                num_heads,
                num_heads_k,
                head_size,
                headdim_v=head_size,
                qkv_dtype=q.dtype,
                cache_seqlens=maybe_contiguous(cache_seqlens),
                seqlens_q=seqlens_q,
                cu_seqlens_k=None,
                cu_seqlens_k_new=None,
                cache_leftpad=None,
                page_size=None,
                max_seqlen_k_new=0,
                causal=causal,
                window_size_left=window_size[0],
                window_size_right=window_size[1],
                attention_chunk=0,
                has_softcap=softcap > 0.0,
                num_splits=num_splits,
                pack_gqa=None,
                sm_margin=sm_margin,
            )
        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None,
            None,  # k_new, v_new
            qv,  # qv
            None,  # out
            cu_seqlens_q,
            cu_seqlens_k,
            None,  # cu_seqlens_k_new
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k,
            None,
            None,
            None,  # rotary_cos/sin, seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            attention_chunk=attention_chunk,
            softcap=softcap,
            scheduler_metadata=scheduler_metadata,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
            layout=layout,
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.save_for_backward(
            q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        ctx.head_size_og = q.size(-1)
        return (out, softmax_lse) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = (
            ctx.saved_tensors
        )
        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
        dout, q, k, v, out, head_size_og = _pad_bwd_headdim(dout, q, k, v, out, ctx.head_size_og)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
        )
        dq = dq[..., :head_size_og]  # We could have padded the head dimension
        dk = dk[..., :head_size_og]
        dv = dv[..., :head_size_og]
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def flash_attn_qkvpacked_func(
    qkv,
    softmax_scale=None,
    causal=False,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    deterministic=False,
    num_heads_q=None,
    sm_margin=0,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_kvpacked_func and flash_attn_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        softmax_scale,
        causal,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        attention_chunk,
        softcap,
        deterministic,
        num_heads_q,
        sm_margin,
        return_attn_probs,
    )


def flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).

    MLA on Ascend 910B/C is forward-only with FP16/BF16 inputs. Compressed MLA
    uses q/k RoPE dimension 64, qv/v content dimension 512 and one KV head;
    prefill MLA uses q/k dimension 192, independent v dimension 128 and qv=None.
    Both support causal/window masks, num_splits=0..256 and finite nonnegative
    softcap. Output uses v.shape[-1]. With qv, scores include qv @ v.T and
    softmax_scale defaults to 1 / sqrt(q.shape[-1] + qv.shape[-1]).
    return_attn_probs=True returns (out, FP32 softmax_lse) for MLA.
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
        return_attn_probs,
    )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
):
    """Compute attention for TND inputs with int32 allocation prefixes.

    cu_seqlens_q/k have shape (batch_size + 1,) and start at zero;
    max_seqlen_q/k bound each allocated segment. Optional int32 seqused_q/k
    have shape (batch_size,) and limit the used prefix without moving the
    next sample's allocation. Output keeps the allocated query shape with
    last dimension v.shape[-1]; unused query slots have no value guarantee.
    return_attn_probs=True returns (out, lse), with FP32 lse shaped (nheads, total_q).

    Prefill MLA uses q/k dimensions 192 and independent v dimension 128 with qv=None;
    compressed MLA uses q/k dimensions 64 and qv/v dimensions 512 with one KV head.
    Both support allocation/used lengths, causal/window masks, num_splits=0..256
    and nonnegative finite softcap. Metadata is generated internally on every call.
    There is no explicit metadata or page-table parameter; paged MLA uses
    flash_attn_with_kvcache.
    """
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
        return_attn_probs,
    )


def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None):
    return flash_attn_npu_3.fwd_combine(out_partial, lse_partial, out, out_dtype)


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk=0,
    softcap=0.0,  # 0.0 means deactivated
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
    return_softmax_lse=False,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a page_table (i.e. paged KV cache)
            page_block_size can be arbitrary (e.g, 1, 2, 3, 64, etc.).
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim_v) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim_v) if there's a page_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim_v). Similar to k.
        qv [optional]: (batch_size, seqlen, nheads, headdim_v)
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        page_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        scheduler_metadata: Optional scheduler metadata precomputed on the AICPU by
            get_scheduler_metadata. The forward uses the AICPU-computed schedule,
            avoiding host-side computation that would sync on cache_seqlens.
            Each forward supplies its numerical scale and softcap through tiling. Creation arguments are
            validated against this call: causal / window_size / has_softcap /
            num_splits must match, and max_seqlen_k must equal the effective KV cache
            capacity seen by the forward (page_table: page_block_size * page_table.shape[1];
            otherwise k_cache.shape[1]) — do not overprovision it. Pass the tensor
            returned by get_scheduler_metadata unchanged (do not clone/copy it).
            softmax_scale and the numerical softcap value come from this forward
            call and may change when reusing metadata with the same has_softcap.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).

    MLA on Ascend 910B/C supports compressed 64/512 inputs with qv and prefill
    192/128 inputs with qv=None, as described in flash_attn_func. Caches are
    read-only BSND or ND paged; BSND/TND queries support causal/window masks,
    num_splits=0..256 and finite nonnegative softcap. Append and rotary options
    are unsupported for MLA. Examples are in tests/test_flash_attn_npu_v3_mla.py.

    For MLA TND queries, q/qv have shape (total_q, nheads, headdim/headdim_v)
    and require cu_seqlens_q/max_seqlen_q. None cache_seqlens uses full logical
    capacity; integers broadcast over logical batches. Paged capacity is
    page_table.shape[1] * k_cache.shape[1], at most 16384; page_block_size must
    be a multiple of 16 in [16, 1024]. Output follows q with v_cache.shape[-1].
    With qv, scores include qv @ v.T and the default scale uses both dimensions.

    MLA scheduler_metadata uses int32 records bound to representation, layout,
    mask, split and softcap activation. The device validates current lengths,
    query prefixes and used pages. Records may be cloned or contiguous offset
    views; scale and positive softcap values may change during reuse. Changed
    lengths or softcap activation require regeneration. Reuse skips AICPU
    generation; host workspace budgeting still runs.

    MLA return_softmax_lse=True returns (out, softmax_lse, out_accum, lse_accum).
    LSE is FP32, [B,Hq,Sq] for BSND or [Hq,Tq] for TND, including qv scores.
    num_splits=1 returns empty accumulators. Automatic splitting returns 26 FP32
    slots; the native partition selects actual splits. Slots with LSE=-inf do
    not contribute, and their partial O is unspecified, as in official FA3.
    Automatic direct rows write final out/LSE without populating partial slots.

    INT8 MLA is an NPU extension using the same signature: q/k_cache hold BF16
    RoPE (64), qv/v_cache hold INT8 content (512), and output is BF16. It requires
    ND paged caches with page size 128, one KV head, power-of-two query heads
    up to 128, at most 32 queries per sequence, and causal=True for multi-query.
    num_splits is 0 or 1; window, softcap, sm_margin and nondefault GQA packing
    are unsupported. All three same-device FP32 descales are required:
    q_descale is (B,Sq,Hq) or (Tq,Hq), and k_descale/v_descale are (1,).
    These follow CANN INT8 semantics: Q/K scales multiply the combined content
    and RoPE logits; V scale multiplies the value result. Their shapes differ
    from official FP8 descales. Metadata uses qkv_dtype=torch.int8; scales may
    change during record reuse. INT8 logical KV capacity may exceed 16384.
    """
    is_mla = _is_mla(q, v_cache, qv, cu_seqlens_q)
    if not is_mla:
        assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
        assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        batch = (
            (page_table.shape[0] if page_table is not None else k_cache.shape[0])
            if is_mla
            else q.shape[0]
        )
        if is_mla and not 0 <= cache_seqlens <= k_cache.shape[1] * (
            page_table.shape[1] if page_table is not None else 1
        ):
            raise RuntimeError("MLA: cache_seqlens must be between zero and KV capacity")
        cache_seqlens = torch.full(
            (batch,), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    if scheduler_metadata is not None and not is_mla:
        _validate_scheduler_metadata(
            scheduler_metadata,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            seqlen_q=q.shape[1] if cu_seqlens_q is None else max_seqlen_q,
            varlen_q=cu_seqlens_q is not None,
            num_splits=num_splits,
            page_table=page_table,
            k_cache=k_cache,
            cache_seqlens=cache_seqlens,
        )
    # cu_seqlens_q pins the layout to TND; otherwise _detect_layout picks BSND or BNSD.
    if cu_seqlens_q is not None:
        layout, out_transposed = _LAYOUT_TND, False
    elif is_mla:
        layout, out_transposed = _LAYOUT_BSND, False
    else:
        bsnd_seqlen_k = k_cache.shape[1]
        layout, q, k_cache, v_cache, out_transposed = _detect_layout(
            q, k_cache, v_cache, page_table, None
        )
        if out_transposed:
            if k is not None or v is not None or qv is not None:
                raise NotImplementedError(
                    "flash_attn_npu_3: 910 backend BNSD layout does not support "
                    "append-KV (k/v) or qv"
                )
            if cache_seqlens is not None:
                max_seqlen_k_bound = int(cache_seqlens.max().item())
                if max_seqlen_k_bound > bsnd_seqlen_k:
                    raise ValueError(
                        "flash_attn_npu_3: BNSD KV sequence dimension "
                        f"({bsnd_seqlen_k}) is too small to cover the actual KV length "
                        f"({max_seqlen_k_bound})"
                    )
    out, softmax_lse, *rest = _flash_attn_forward(
        q,
        k_cache,
        v_cache,
        k,
        v,
        qv,
        None,  # out
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_seqlens,
        max_seqlen_q,
        None,  # max_seqlen_k
        page_table,
        cache_batch_idx,
        cache_leftpad,
        rotary_cos,
        rotary_sin,
        rotary_seqlens,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        attention_chunk=attention_chunk,
        softcap=softcap,
        rotary_interleaved=rotary_interleaved,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        layout=layout,
    )
    if out_transposed:
        # Report the result in the shape the caller passed in, not the BNSD view.
        out = out.transpose(1, 2)
    # return (out, softmax_lse) if return_softmax_lse else out
    return (out, softmax_lse, *rest) if return_softmax_lse else out


def _validate_scheduler_metadata(
    scheduler_metadata,
    *,
    causal,
    window_size,
    softcap,
    seqlen_q,
    varlen_q,
    num_splits,
    page_table,
    k_cache,
    cache_seqlens,
):
    """Reject scheduler_metadata created with arguments that do not match this
    call's mask layout, paged geometry, softcap feature and split schedule.
    Per-call numerical values are supplied in tiling without changing the shared schedule."""
    params = getattr(scheduler_metadata, "_fa_scheduler_params", None)
    if params is None:
        raise RuntimeError(
            "scheduler_metadata has no creation-argument fingerprint; it must be "
            "the unchanged tensor returned by get_scheduler_metadata"
        )
    expected = {
        "causal": bool(causal),
        "window_size": (int(window_size[0]), int(window_size[1])),
        "has_softcap": softcap > 0.0,
        "varlen_q": bool(varlen_q),
        "num_splits": int(num_splits),
    }
    if seqlen_q is not None:
        expected["max_seqlen_q"] = int(seqlen_q)
    if page_table is not None:
        # The fwd mask derivation and the page-table row stride both use the
        # paged capacity (page_block_size * max_num_blocks_per_seq) as KV bound.
        expected["page_size"] = int(k_cache.shape[1])
        expected["max_seqlen_k"] = int(k_cache.shape[1]) * int(page_table.shape[1])
    else:
        expected["page_size"] = None
        if varlen_q and k_cache.dim() == 3:
            if cache_seqlens is None:
                raise ValueError("cache_seqlens must be provided for a non-paged TND KV cache")
            expected["max_seqlen_k"] = int(cache_seqlens.max().item())
        else:
            expected["max_seqlen_k"] = int(k_cache.shape[1])
    mismatches = []
    for key, call_val in expected.items():
        meta_val = params.get(key, "<missing>")
        if meta_val != call_val:
            mismatches.append(f"{key}: metadata={meta_val!r} vs call={call_val!r}")
    if mismatches:
        raise ValueError(
            "scheduler_metadata arguments do not match this call: " + "; ".join(mismatches)
        )


# Real metadata contract (sizeof(FAInferTilingData)):
#   all mask types:
#       shape = (2376,)
#   dtype  = torch.uint8
#   device = NPU
#   stride = (1,)
# The custom op prevents TorchDynamo from tracing into the raw pybind
# get_scheduler_metadata() implementation. Schema matches the V3 910
# pybind signature in flash_api.cpp (not the V4 parameter list).
_SCHEDULER_METADATA_TILING_BYTES = 2376


def _get_scheduler_metadata_op(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    headdim: int,
    headdim_v: int,
    qkv_dtype: torch.dtype,
    cache_seqlens: torch.Tensor,
    seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    cu_seqlens_k_new: Optional[torch.Tensor],
    cache_leftpad: Optional[torch.Tensor],
    page_size: Optional[int],
    max_seqlen_k_new: int,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    attention_chunk: int,
    has_softcap: bool,
    num_splits: int,
    pack_gqa: Optional[bool],
    sm_margin: int,
) -> torch.Tensor:
    scheduler_metadata = flash_attn_npu_3.get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        headdim_v,
        qkv_dtype,
        cache_seqlens,
        seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        cache_leftpad,
        page_size,
        max_seqlen_k_new,
        causal,
        window_size_left,
        window_size_right,
        attention_chunk,
        has_softcap,
        num_splits,
        pack_gqa,
        sm_margin,
    )
    if headdim != headdim_v:
        return scheduler_metadata
    # Keep this side effect inside the opaque custom op so Dynamo does not
    # trace the setattr while still attaching the runtime fingerprint.
    scheduler_metadata._fa_scheduler_params = {
        "causal": bool(causal),
        "window_size": (int(window_size_left), int(window_size_right)),
        "has_softcap": bool(has_softcap),
        "page_size": None if page_size is None else int(page_size),
        "max_seqlen_q": int(max_seqlen_q),
        "max_seqlen_k": int(max_seqlen_k),
        "varlen_q": seqlens_q is not None,
        "num_splits": int(num_splits),
    }
    return scheduler_metadata


if hasattr(torch.library, "infer_schema"):
    # Metadata owns fresh integer storage, so it needs no autograd or alias
    # checks. Keep the same schema, compile tags and fake dispatch.
    torch.library.define(
        "flash_attn_npu_3::_get_scheduler_metadata",
        torch.library.infer_schema(_get_scheduler_metadata_op, mutates_args=()),
        tags=torch.ops.flash_attn_npu_3._flash_attn_forward.default.tags,
    )
    torch.library.impl(
        "flash_attn_npu_3::_get_scheduler_metadata", "default", _get_scheduler_metadata_op
    )
    _get_scheduler_metadata_op = torch.ops.flash_attn_npu_3._get_scheduler_metadata.default
else:
    _get_scheduler_metadata_op = torch.library.custom_op(
        "flash_attn_npu_3::_get_scheduler_metadata", _get_scheduler_metadata_op, mutates_args=()
    )


@_torch_register_fake_wrapper("flash_attn_npu_3::_get_scheduler_metadata")
def _get_scheduler_metadata_fake(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    headdim: int,
    headdim_v: int,
    qkv_dtype: torch.dtype,
    cache_seqlens: torch.Tensor,
    seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    cu_seqlens_k_new: Optional[torch.Tensor],
    cache_leftpad: Optional[torch.Tensor],
    page_size: Optional[int],
    max_seqlen_k_new: int,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    attention_chunk: int,
    has_softcap: bool,
    num_splits: int,
    pack_gqa: Optional[bool],
    sm_margin: int,
) -> torch.Tensor:
    if headdim != headdim_v:
        if cache_seqlens.device.type != "npu":
            raise RuntimeError("MLA: cache_seqlens must be an NPU tensor")
        _check_mla_devices(cache_seqlens, seqlens_q=seqlens_q)
        return _mla_native_fake(
            flash_attn_npu_3.get_scheduler_metadata,
            cache_seqlens,
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            num_heads_q,
            num_heads_kv,
            headdim,
            headdim_v,
            qkv_dtype,
            cache_seqlens,
            seqlens_q,
            cu_seqlens_k,
            cu_seqlens_k_new,
            cache_leftpad,
            page_size,
            max_seqlen_k_new,
            causal,
            window_size_left,
            window_size_right,
            attention_chunk,
            has_softcap,
            num_splits,
            pack_gqa,
            sm_margin,
        )
    return torch.empty(
        (_SCHEDULER_METADATA_TILING_BYTES,),
        dtype=torch.uint8,
        device=cache_seqlens.device,
    )


def get_scheduler_metadata(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    num_heads_kv,
    headdim,
    cache_seqlens: torch.Tensor,
    qkv_dtype=torch.bfloat16,
    headdim_v=None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    max_seqlen_k_new=0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk=0,
    has_softcap=False,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
):
    """Precompute NPU scheduling metadata with the official FA3 argument order.

    Ordinary FA3 returns byte scheduling metadata. Numerical softcap and scale
    are supplied by each forward call; has_softcap selects the scheduling feature.
    MLA uses device int32 records for compressed headdim=64/headdim_v=512
    with one KV head, or independent prefill headdim=192/headdim_v=128 with
    GQA. Both accept BSND/TND queries, nonpaged or ND paged caches, all masks
    and num_splits=0..256. The default num_splits=0 selects automatic splitting.
    Nonpaged and paged layouts accept has_softcap=True.

    For INT8 MLA, pass qkv_dtype=torch.int8, headdim=64, headdim_v=512,
    num_heads_kv=1 and page_size=128. Query/mask/split limits are described in
    flash_attn_with_kvcache. max_seqlen_k is a multiple of 128 up to INT32_MAX;
    scale values are supplied only to forward and are not stored in metadata.

    cache_seqlens must be a same-device contiguous int32 tensor of shape
    (batch_size,); None/integer lengths are kvcache conveniences only.
    With cu_seqlens_q, queries use TND allocation segments and max_seqlen_q
    bounds each segment; otherwise it is the BSND query capacity. Without
    page_size, max_seqlen_k is BSND cache capacity. With page_size, it is logical
    capacity. For FP16/BF16, it is divisible by page_size and at most 16384;
    page_size must be a multiple of 16 in [16, 1024]. This generator does not
    inspect a page table or physical cache. Forward validates current lengths,
    query prefixes and all used pages on device, including when reusing a record.

    Records bind shape, lengths, dtype, normalized mask, split, platform and
    softcap activation, but not scale, positive cap values, addresses or strides.
    Regenerate after lengths, prefixes or other bound conditions change.
    Supply numerical softcap and softmax_scale values to forward. There is no
    Q-used or KV allocation-prefix parameter here; compact TND Q/KV uses
    flash_attn_varlen_func and generates metadata internally.
    """
    if headdim_v is None:
        headdim_v = headdim
    if headdim != headdim_v:
        seqlens_q = cu_seqlens_q
    else:
        cache_seqlens = maybe_contiguous(cache_seqlens)
        seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1] if cu_seqlens_q is not None else None
    # Route through the custom op so torch.compile / FakeTensor can use the
    # registered fake instead of tracing into the raw pybind extension.
    scheduler_metadata = _get_scheduler_metadata_op(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        headdim_v,
        qkv_dtype,
        cache_seqlens,
        seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        cache_leftpad,
        page_size,
        max_seqlen_k_new,
        causal,
        window_size[0],
        window_size[1],
        attention_chunk,
        has_softcap,
        num_splits,
        pack_gqa,
        sm_margin,
    )
    return scheduler_metadata
