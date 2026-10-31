# Copyright (c) 2023, Tri Dao.
# Modified by Minghua Shen, 2026

from typing import Optional, Tuple, Union

import torch
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

# isort: off
# We need to import the kernels after importing torch
from . import flash_attn_npu
# isort: on


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
            "flash_attn_npu: key and value must have the same rank for layout detection"
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


def _validate_varlen_query(q, cu_seqlens_q):
    if q.dim() != 3:
        raise RuntimeError(f"q must be 3-dimensional; got {q.dim()} dimensions")
    if cu_seqlens_q.numel() <= 1:
        raise RuntimeError("batch size must be positive")
    if q.shape[0] == 0:
        raise RuntimeError("total_q must be positive (q must contain at least one query token)")


_HEADDIM_BWD_ALIGN = 64


def _pad_bwd_headdim(dout, q, k, v, out, head_size_og):
    """
    Pad headdim to a multiple of 64 for the bwd kernel.
    """
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


# torch.compile() support is only enabled for pytorch >= 2.4
# The reason for this is that we are using the new custom_op and register_fake
# APIs, which support inplace modification of inputs in the function itself
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


@_torch_custom_op_wrapper(
    "flash_attn_npu::_flash_attn_forward", mutates_args=(), device_types="npu"
)
def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool,
    scheduler_metadata: Optional[torch.Tensor] = None,
    layout: int = _LAYOUT_BSND,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = flash_attn_npu.fwd(
        q,
        k,
        v,
        None,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        None,
        scheduler_metadata,
        layout,
    )
    return out, softmax_lse, S_dmask, rng_state


@_torch_register_fake_wrapper("flash_attn_npu::_flash_attn_forward")
def _flash_attn_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool,
    scheduler_metadata: Optional[torch.Tensor] = None,
    layout: int = _LAYOUT_BSND,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if layout == _LAYOUT_BNSD:
        batch_size, num_heads, seqlen_q, head_size = q.shape
        out = torch.empty(
            (batch_size, seqlen_q, num_heads, head_size),
            dtype=q.dtype,
            device=q.device,
            layout=q.layout,
        )
    else:
        batch_size, seqlen_q, num_heads, head_size = q.shape
        out = torch.empty_like(q)
    seqlen_k = k.shape[-2] if layout == _LAYOUT_BNSD else k.shape[1]
    softmax_lse = torch.empty(
        (batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device, layout=q.layout
    )
    p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
    if return_softmax:
        if torch.cuda.is_available() and torch.version.hip:
            p = torch.empty(
                (batch_size, num_heads, seqlen_q, seqlen_k),
                dtype=q.dtype,
                device=q.device,
                layout=q.layout,
            )
        else:
            p = torch.empty(
                (
                    batch_size,
                    num_heads,
                    round_multiple(seqlen_q, 128),
                    round_multiple(seqlen_k, 128),
                ),
                dtype=q.dtype,
                device=q.device,
                layout=q.layout,
            )
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)

    return out, softmax_lse, p, rng_state


_wrapped_flash_attn_forward = _flash_attn_forward


def _flash_attn_varlen_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
    block_table: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
    scheduler_metadata: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = flash_attn_npu.varlen_fwd(
        q,
        k,
        v,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        None,
        scheduler_metadata,
    )
    # if out.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    return out, softmax_lse, S_dmask, rng_state


_flash_attn_varlen_forward = _torch_custom_op_wrapper(
    "flash_attn_npu::_flash_attn_varlen_forward", mutates_args=(), device_types="npu"
)(_flash_attn_varlen_forward_impl)


@_torch_register_fake_wrapper("flash_attn_npu::_flash_attn_varlen_forward")
def _flash_attn_varlen_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
    block_table: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
    scheduler_metadata: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_varlen_query(q, cu_seqlens_q)
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    batch_size = cu_seqlens_q.numel() - 1
    total_q, num_heads, _ = q.shape

    out = torch.empty_like(q)
    softmax_lse = torch.empty(
        (num_heads, total_q), dtype=torch.float32, device=q.device, layout=q.layout
    )
    p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
    if return_softmax:
        if torch.cuda.is_available() and torch.version.hip:
            p = torch.empty(
                (batch_size, num_heads, max_seqlen_q, max_seqlen_k),
                dtype=q.dtype,
                device=q.device,
                layout=q.layout,
            )
        else:
            p = torch.empty(
                (
                    batch_size,
                    num_heads,
                    round_multiple(max_seqlen_q, 128),
                    round_multiple(max_seqlen_k, 128),
                ),
                dtype=q.dtype,
                device=q.device,
                layout=q.layout,
            )
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    return out, softmax_lse, p, rng_state


_wrapped_flash_attn_varlen_forward = _flash_attn_varlen_forward


@_torch_custom_op_wrapper(
    "flash_attn_npu::_flash_attn_backward", mutates_args=("dq", "dk", "dv"), device_types="npu"
)
def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    # The backward kernels use packed dO strides, including across heads/tokens.
    # A last-dimension-contiguous transposed view still needs materialization.
    if not dout.is_contiguous():
        dout = dout.contiguous()
    q, k, v, out = [maybe_contiguous(x) for x in (q, k, v, out)]
    (
        dq,
        dk,
        dv,
        softmax_d,
    ) = flash_attn_npu.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        None,
        rng_state,
    )
    return softmax_d


@_torch_register_fake_wrapper("flash_attn_npu::_flash_attn_backward")
def _flash_attn_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # The backward kernels use packed dO strides, including across heads/tokens.
    # A last-dimension-contiguous transposed view still needs materialization.
    if not dout.is_contiguous():
        dout = dout.contiguous()
    q, k, v, out = [maybe_contiguous(x) for x in (q, k, v, out)]
    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    batch_size, seqlen_q, num_heads, _ = q.shape
    if torch.cuda.is_available() and torch.version.hip:
        softmax_d = torch.empty(
            (batch_size, num_heads, seqlen_q), device=q.device, dtype=torch.float32
        )
    else:
        softmax_d = torch.empty(
            (batch_size, num_heads, round_multiple(seqlen_q, 128)),
            device=q.device,
            dtype=torch.float32,
        )

    return softmax_d


_wrapped_flash_attn_backward = _flash_attn_backward


@_torch_custom_op_wrapper(
    "flash_attn_npu::_flash_attn_varlen_backward",
    mutates_args=("dq", "dk", "dv"),
    device_types="npu",
)
def _flash_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    # The backward kernels use packed dO strides, including across heads/tokens.
    # A last-dimension-contiguous transposed view still needs materialization.
    if not dout.is_contiguous():
        dout = dout.contiguous()
    q, k, v, out = [maybe_contiguous(x) for x in (q, k, v, out)]
    (
        dq,
        dk,
        dv,
        softmax_d,
    ) = flash_attn_npu.varlen_bwd(
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
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        None,
        rng_state,
    )
    # if dk.isnan().any() or dk.isnan().any() or dv.isnan().any() or softmax_d.isnan().any():
    #     breakpoint()
    return softmax_d


@_torch_register_fake_wrapper("flash_attn_npu::_flash_attn_varlen_backward")
def _flash_attn_varlen_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> torch.Tensor:
    # The backward kernels use packed dO strides, including across heads/tokens.
    # A last-dimension-contiguous transposed view still needs materialization.
    if not dout.is_contiguous():
        dout = dout.contiguous()
    q, k, v, out = [maybe_contiguous(x) for x in (q, k, v, out)]
    batch_size = cu_seqlens_q.numel() - 1
    total_q, num_heads, _ = q.shape

    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    if torch.cuda.is_available() and torch.version.hip:
        softmax_d = torch.empty((num_heads, total_q), device=q.device, dtype=torch.float32)
    else:
        softmax_d = torch.empty(
            (num_heads, total_q + 128 * batch_size), device=q.device, dtype=torch.float32
        )

    return softmax_d


_wrapped_flash_attn_varlen_backward = _flash_attn_varlen_backward


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and qkv.requires_grad
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        q, k, v = qkv[:, :, 0].detach(), qkv[:, :, 1].detach(), qkv[:, :, 2].detach()
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size_og = head_size_og
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dout, q, k, v, out, head_size_og = _pad_bwd_headdim(dout, q, k, v, out, ctx.head_size_og)
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        _wrapped_flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dqkv[:, :, 0],
            dqkv[:, :, 1],
            dqkv[:, :, 2],
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dqkv = dqkv[..., :head_size_og]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and qkv.requires_grad
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        q, k, v = qkv[:, 0].detach(), qkv[:, 1].detach(), qkv[:, 2].detach()
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=None,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens, rng_state)
            ctx.dropout_p = dropout_p
            ctx.max_seqlen = max_seqlen
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size_og = head_size_og
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        dout, q, k, v, out, head_size_og = _pad_bwd_headdim(dout, q, k, v, out, ctx.head_size_og)
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        _wrapped_flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dqkv[:, 0],
            dqkv[:, 1],
            dqkv[:, 2],
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dqkv = dqkv[..., :head_size_og]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, kv])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        k, v = kv[:, :, 0].detach(), kv[:, :, 1].detach()
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size_og = head_size_og
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dout, q, k, v, out, head_size_og = _pad_bwd_headdim(dout, q, k, v, out, ctx.head_size_og)
        dq = torch.empty_like(q)
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        _wrapped_flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dkv[:, :, 0],
            dkv[:, :, 1],
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., :head_size_og]  # We could have padded the head dimension
        dkv = dkv[..., :head_size_og]
        return dq, dkv, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, kv])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        k, v = kv[:, 0].detach(), kv[:, 1].detach()
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=None,
        )
        if is_grad:
            ctx.save_for_backward(
                q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
            )
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size_og = head_size_og
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dout, q, k, v, out, head_size_og = _pad_bwd_headdim(dout, q, k, v, out, ctx.head_size_og)
        dq = torch.empty_like(q)
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        _wrapped_flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dkv[:, 0],
            dkv[:, 1],
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., :head_size_og]  # We could have padded the head dimension
        dkv = dkv[..., :head_size_og]
        return dq, dkv, None, None, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        # Training interfaces do not expose scheduler_metadata (aligned with
        # official flash-attn); the scheduler metadata is computed internally on
        # the AICPU so no D2H/H2D sync breaks the pipeline.
        # Detect the layout first, then pad the head dim: get_scheduler_metadata must
        # receive the head dim the kernel actually runs on (padded), and the per-layout
        # sizes must be read from the operand the kernel receives.
        layout, q, k, v, out_transposed = _detect_layout(q, k, v, None, None)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            scheduler_metadata=None,
            layout=layout,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size_og = head_size_og
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dout, q, k, v, out, head_size_og = _pad_bwd_headdim(dout, q, k, v, out, ctx.head_size_og)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _wrapped_flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., :head_size_og]  # We could have padded the head dimension
        dk = dk[..., :head_size_og]
        dv = dv[..., :head_size_og]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        block_table,
        is_grad_enabled,
    ):
        _validate_varlen_query(q, cu_seqlens_q)
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        # The native entry schedules nonpaged, dropout-free TND directly from
        # device sequence lengths. Other modes retain their host tiling path.
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=block_table,
            scheduler_metadata=None,
        )
        if is_grad:
            ctx.save_for_backward(
                q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
            )
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size_og = head_size_og

        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dout, q, k, v, out, head_size_og = _pad_bwd_headdim(dout, q, k, v, out, ctx.head_size_og)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _wrapped_flash_attn_varlen_backward(
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
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
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


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # <=0.0 means deactivate
    alibi_slopes=None,
    deterministic=False,
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
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to
            the attention score of query i and key j.
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
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of K, V.
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
        kv: (batch_size, seqlen, 2, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
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
    return FlashAttnKVPackedFunc.apply(
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
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
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
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
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_varlen_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_varlen_kvpacked_func and flash_attn_varlen_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (total, 3, nheads, headdim), where total = total number of tokens in the batch.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into qkv.
        max_seqlen: int. Maximum sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnVarlenQKVPackedFunc.apply(
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of K, V.
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
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        kv: (total_k, 2, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnVarlenKVPackedFunc.apply(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
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
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    # # Inference needs neither an autograd context nor an output slice when the
    # # head dimension is already aligned. Instrumented/traced calls keep the
    # # registered custom op; ordinary eager tensors can call its implementation.
    # if (dropout_p == 0.0 and q.dim() == 3 and q.shape[-1] % 8 == 0 and
    #         (not torch.is_grad_enabled() or not (q.requires_grad or k.requires_grad or v.requires_grad))):
    #     _validate_varlen_query(q, cu_seqlens_q)
    #     forward = _wrapped_flash_attn_varlen_forward
    #     tensors = (q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes, block_table)
    #     if (not torch.compiler.is_compiling() and not torch.jit.is_tracing()
    #             and not is_in_torch_dispatch_mode()
    #             and all(x is None or type(x) is torch.Tensor for x in tensors)
    #             and not torch.overrides.has_torch_function(tensors)):
    #         forward = _flash_attn_varlen_forward_impl
    #     out, lse, mask, _ = forward(
    #         q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    #         0.0, q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale,
    #         causal=causal, window_size_left=window_size[0], window_size_right=window_size[1],
    #         softcap=softcap, alibi_slopes=alibi_slopes, block_table=block_table,
    #     )
    #     return (out, lse, mask) if return_attn_probs else out

    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_table,
        torch.is_grad_enabled(),
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    scheduler_metadata=None,
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
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
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
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Values greater than 1 are supported only by the paged-KV BSND kvcache path.
           Don't change this unless you know what you are doing.
        scheduler_metadata: Optional scheduler metadata precomputed on the AICPU by
            get_scheduler_metadata. When given, the forward consumes the AICPU-computed
            tiling (and optional mask) directly and skips the host-side computation
            (which would otherwise sync on cache_seqlens). This is a plain byte
            tensor: clone() and copy_() are supported without Python attributes.
            The caller must keep the schedule compatible with this call, including
            mask/window, layout, paging geometry, sequence bounds and num_splits.
            softmax_scale, softcap and the ALiBi batch stride are baked into this
            metadata and must also match the forward. max_seqlen_k must equal the
            KV cache capacity (paged: page_size * block_table.shape[1]; dense:
            the KV sequence dimension), not an overprovisioned bound.
            The backend checks dtype, device, contiguity and byte size, but does
            not validate the metadata contents against the call's arguments.
            For graph replay, copy updates into the captured buffer on the replay
            stream (or establish a stream dependency). Keep the buffer address
            and captured kernel configuration fixed; recapture when it changes.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (q.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
    # _detect_layout picks BNSD only from matching strides; paged KV is never BNSD.
    bsnd_seqlen_k = k_cache.shape[1]
    layout, q, k_cache, v_cache, out_transposed = _detect_layout(
        q, k_cache, v_cache, block_table, None
    )
    if out_transposed:
        if k is not None or v is not None:
            raise NotImplementedError("flash_attn_npu: 910 后端 BNSD 布局不支持 append-KV (k/v)")
        if cache_seqlens is not None:
            max_seqlen_k_bound = int(cache_seqlens.max().item())
            if max_seqlen_k_bound > bsnd_seqlen_k:
                raise ValueError(
                    "flash_attn_npu: BNSD KV 序列维 "
                    f"({bsnd_seqlen_k}) 太小，无法覆盖实际 KV 长度 "
                    f"({max_seqlen_k_bound})"
                )
    out, softmax_lse = flash_attn_npu.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        None,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        num_splits,
        scheduler_metadata,
        layout,
    )
    return (out, softmax_lse) if return_softmax_lse else out


# Real metadata contract:
#   all mask types:
#       shape = (2416,)
#   dtype  = torch.uint8
#   device = NPU
#   stride = (1,)
# Schema matches the V2 910 pybind signature in flash_api.cpp
# (not the V3/V4 parameter lists).
_SCHEDULER_METADATA_TILING_BYTES = 2416


@torch.library.custom_op(
    "flash_attn_npu::_get_scheduler_metadata",
    mutates_args=(),
)
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
    cu_seqlens_q: Optional[torch.Tensor],
    page_size: Optional[int],
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    softmax_scale: Optional[float],
    alibi_slopes_batch_stride: int,
    num_splits: int,
) -> torch.Tensor:
    scheduler_metadata = flash_attn_npu.get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        headdim_v,
        qkv_dtype,
        cache_seqlens,
        cu_seqlens_q,
        page_size,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        softmax_scale,
        alibi_slopes_batch_stride,
        num_splits,
    )
    return scheduler_metadata


@_torch_register_fake_wrapper("flash_attn_npu::_get_scheduler_metadata")
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
    cu_seqlens_q: Optional[torch.Tensor],
    page_size: Optional[int],
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    softmax_scale: Optional[float],
    alibi_slopes_batch_stride: int,
    num_splits: int,
) -> torch.Tensor:
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
    page_size: Optional[int] = None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    softmax_scale=None,  # defaults to 1 / sqrt(headdim); must match the fwd call
    alibi_slopes_batch_stride=0,
    num_splits=0,  # 0 = auto, 1 = no KV split, >1 = requested KV split count
):
    """Precompute the forward scheduler metadata (tiling, and for paged KV cache
    the flash-decode split schedule) on the AICPU. Pass the
    result as scheduler_metadata to flash_attn_with_kvcache to keep the forward
    free of D2H/H2D syncs."""
    cache_seqlens = maybe_contiguous(cache_seqlens)
    if headdim_v is None:
        headdim_v = headdim
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
        cu_seqlens_q,
        page_size,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        softmax_scale,
        alibi_slopes_batch_stride,
        num_splits,
    )
    return scheduler_metadata
