"""Public autograd interface. No Torch math is used to compute attention."""

from __future__ import annotations

import math
import operator
import struct
from typing import Callable, NamedTuple, Optional, Tuple

import torch
from torch.autograd.function import once_differentiable

from .block_sparsity import (
    BlockSparseTensorsTorch,
    normalize_block_sparse_tensors,
    sparse_tensor_versions,
    validate_sparse_versions,
    _FLOAT32_MAX,
    _check_aux_dtype,
    _check_aux_scalars,
    _check_cumulative,
    _check_maximum_lengths,
)


class SchedulerMetadataTensorsTorch(NamedTuple):
    """Signature-compatible metadata declaration; NPU scheduling is not implemented."""

    num_m_blocks_ptr: Optional[torch.Tensor]
    num_splits_dynamic_ptr: Optional[torch.Tensor]
    virtual_batch_idx_ptr: Optional[torch.Tensor]
    num_nheads_in_l2_ptr: Optional[torch.Tensor]
    tile_count_semaphore: Optional[torch.Tensor]
    cu_total_m_blocks: Optional[torch.Tensor] = None
    cu_total_splits_m_blocks: Optional[torch.Tensor] = None
    blocks_to_batch_idx: Optional[torch.Tensor] = None


def _validate(q, k, v, aux_tensors, aux_scalars, scale, cu_q, cu_k, max_q, max_k, *, layout="BNSD"):
    # Internal callers (including torch_npu_flex) use BNSD; public inputs are BSND.
    if not isinstance(layout, str) or layout not in ("BSND", "BNSD"):
        raise ValueError('layout must be "BSND" or "BNSD"')
    if any(not isinstance(t, torch.Tensor) for t in (q, k, v)):
        raise TypeError("q/k/v must be tensors")
    if q.device.type != "npu":
        raise ValueError("attention requires an NPU")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q/k/v must be float16 or bfloat16")
    if any(
        t.ndim not in (3, 4) or t.device != q.device or t.dtype != q.dtype or not t.is_contiguous()
        for t in (q, k, v)
    ):
        raise ValueError("q/k/v must be contiguous BSND, BNSD or TND with matching dtype/device")
    if k.shape != v.shape or any(t.shape[-1] != 128 for t in (q, k, v)):
        raise ValueError("k/v shapes must match and all head dimensions must be 128")

    def side(t, cu, maximum):
        if t.ndim == 4:
            heads, length = (
                (t.shape[2], t.shape[1]) if layout == "BSND" else (t.shape[1], t.shape[2])
            )
            if cu is not None:
                raise ValueError("cumulative lengths are only valid for packed TND")
            if maximum is not None and maximum != length:
                raise ValueError("fixed-layout maximum lengths must equal tensor lengths")
            return t.shape[0], heads, length
        if cu is None or maximum is None:
            raise ValueError("each packed TND side needs cumulative lengths and a maximum length")
        if type(maximum) is not int or maximum < 0 or (maximum == 0 and t.shape[0] != 0):
            raise ValueError(
                "packed maximum sequence lengths must be positive integers, or zero for empty inputs"
            )
        if not isinstance(cu, torch.Tensor):
            raise ValueError("cumulative lengths must be tensors")
        batches = cu.numel() - 1
        _check_cumulative(cu, batches, maximum, t.device, total=t.shape[0])
        return batches, t.shape[1], maximum

    B, H, Sq = side(q, cu_q, max_q)
    BK, KH, Sk = side(k, cu_k, max_k)
    if BK != B:
        raise ValueError("q/k batch dimensions must match")
    if B < 0 or min(H, KH) < 1 or H % KH:
        raise ValueError(
            "nonnegative batch size, positive heads and Hq divisible by Hkv are required"
        )
    _check_maximum_lengths(Sq, Sk)
    if max(q.numel(), k.numel(), B * H * ((Sq + 127) // 128)) >= 2**31:
        raise ValueError("v1 uses int32 device offsets; inputs exceed the supported range")
    for t in aux_tensors:
        if (
            not isinstance(t, torch.Tensor)
            or t.device != q.device
            or any(x < 0 for x in t.stride())
            or t.requires_grad
        ):
            raise ValueError(
                "aux_tensors must be non-differentiable tensors with nonnegative strides on q.device"
            )
        _check_aux_dtype(t)
    _check_aux_scalars(aux_scalars)
    value = 128**-0.5 if scale is None else float(scale)
    if not math.isfinite(value) or abs(value) > _FLOAT32_MAX:
        raise ValueError("scale must be finite and within the float32 range")
    return dict(
        B=B,
        H=H,
        KH=KH,
        Sq=Sq,
        Sk=Sk,
        layout=layout,
        scale=value,
        cu_q=cu_q,
        cu_k=cu_k,
        block_sparse_tensors=None,
        block_sparse_tensors_bwd=None,
        aux_tensors=aux_tensors,
        aux_scalars=aux_scalars,
    )


def _forward(q, k, v, config):
    """Allocate and launch once, shared by eager autograd and graph operators."""
    from .runtime import launch_attention

    output = torch.empty_like(q)
    if q.ndim == 4:
        lse_shape = (
            (q.shape[0], q.shape[2], q.shape[1]) if config.get("layout") == "BSND" else q.shape[:-1]
        )
    else:
        lse_shape = (q.shape[1], q.shape[0])
    # LSE is head-major in both the kernel and the public API.
    lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
    if q.numel() == 0 or k.numel() == 0:
        output.zero_()
        lse.fill_(-torch.inf)
    else:
        launch_attention(q, k, v, output, lse, config)
    return output, lse


def _attention_forward(ctx, q, k, v, config, *aux):
    output, lse = _forward(q, k, v, config)
    ctx.config = config
    ctx.aux_count = len(aux)
    boundaries = tuple(t for t in (config["cu_q"], config["cu_k"]) if t is not None)
    ctx.save_for_backward(q, k, v, output, lse, *aux, *boundaries)
    ctx.set_materialize_grads(False)
    return output, lse


def _backward(q, k, v, out, lse, config, dout, dlse):
    from .runtime import launch_attention

    if not config.get("return_lse", False):
        dlse = None
    if q.numel() == 0 or k.numel() == 0 or (dout is None and dlse is None):
        return tuple(torch.zeros_like(t) for t in (q, k, v))
    if dout is None:
        dout = torch.zeros_like(out)
    validate_sparse_versions(config.get("sparse_versions", ()))
    # Autograd routinely supplies broadcast/strided gradients. Materializing
    # dO changes layout only, not the attention computation.
    dout = dout.contiguous()
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    dk_acc, dv_acc = (
        torch.empty_like(k, dtype=torch.float32),
        torch.empty_like(v, dtype=torch.float32),
    )
    launch_attention(
        q,
        k,
        v,
        out,
        lse,
        config,
        dout=dout,
        dlse=None if dlse is None else dlse.contiguous(),
        dq=dq,
        dk_acc=dk_acc,
        dv_acc=dv_acc,
        dk=dk,
        dv=dv,
    )
    return dq, dk, dv


def _attention_backward(ctx, dout, dlse):
    q, k, v, out, lse, *_ = ctx.saved_tensors
    return (*_backward(q, k, v, out, lse, ctx.config, dout, dlse), None, *((None,) * ctx.aux_count))


class FlashAttnFunc(torch.autograd.Function):
    """Autograd entry for fixed-length attention."""

    @staticmethod
    def forward(ctx, q, k, v, config, *aux):
        return _attention_forward(ctx, q, k, v, config, *aux)

    @staticmethod
    @once_differentiable
    def backward(ctx, dout, dlse):
        return _attention_backward(ctx, dout, dlse)


class FlashAttnVarlenFunc(torch.autograd.Function):
    """Autograd entry for independently fixed or packed Q/KV."""

    @staticmethod
    def forward(ctx, q, k, v, config, *aux):
        return _attention_forward(ctx, q, k, v, config, *aux)

    @staticmethod
    @once_differentiable
    def backward(ctx, dout, dlse):
        return _attention_backward(ctx, dout, dlse)


def _unsupported(**options):
    for name, (value, default) in options.items():
        if value is not default and (isinstance(value, torch.Tensor) or value != default):
            raise NotImplementedError(f"{name} is not supported; use its default {default!r}")


def _canonical_input(t):
    """Normalize storage before apply(), so copies remain in the autograd graph."""
    if not isinstance(t, torch.Tensor):
        raise TypeError("q/k/v must be tensors")
    if t.ndim not in (3, 4):
        raise ValueError("q/k/v must have BSND or packed TND shape")
    if not t.is_contiguous():
        return t.contiguous()
    if t.numel() and t.data_ptr() % 16:
        return t.clone(memory_format=torch.contiguous_format)
    return t


def _maximum_length(t, hint):
    # Ignore tensor hints without synchronizing the device.
    if isinstance(hint, torch.Tensor):
        hint = None
    length = t.shape[0] if t.ndim == 3 else t.shape[1]
    if hint is not None:
        if type(hint) is not int or hint < 0:
            raise ValueError("max_seqlen must be a nonnegative integer, tensor hint or None")
        _check_maximum_lengths(hint, hint)
        if t.ndim == 4 and hint < length:
            raise ValueError("max_seqlen must cover the fixed sequence length")
    # Fixed strides depend on the actual extent, never the scheduler hint.
    return length if t.ndim == 4 or hint is None else hint


def _validate_gather_bwd_kwargs(gather_kv_indices, gather_bwd_recompute_p, gather_bwd_token_chunk):
    if gather_kv_indices is None:
        if gather_bwd_recompute_p or gather_bwd_token_chunk is not None:
            raise ValueError(
                "gather_bwd_recompute_p / gather_bwd_token_chunk require gather_kv_indices"
            )
        return None
    if gather_bwd_token_chunk is not None:
        if isinstance(gather_bwd_token_chunk, bool):
            raise ValueError("gather_bwd_token_chunk must be a positive int or None")
        try:
            gather_bwd_token_chunk = operator.index(gather_bwd_token_chunk)
        except TypeError:
            raise ValueError("gather_bwd_token_chunk must be a positive int or None") from None
        if gather_bwd_token_chunk <= 0:
            raise ValueError("gather_bwd_token_chunk must be a positive int or None")
    return gather_bwd_token_chunk


def _prepare_config(
    q,
    k,
    v,
    *,
    softmax_scale,
    causal,
    window_size,
    softcap,
    score_mod,
    score_mod_bwd,
    mask_mod,
    aux_tensors,
    aux_scalars,
    block_sparse_tensors,
    block_sparse_tensors_bwd,
    return_lse,
    cu_q=None,
    cu_k=None,
    max_q=None,
    max_k=None,
    score_mod_simd=None,
    score_mod_bwd_simd=None,
    mask_mod_simd=None,
    layout="BSND",
    training=False,
):
    from . import builtin
    from .callbacks import prepare_mod
    from .mods import callback_mode

    aux, scalars = tuple(aux_tensors or ()), tuple(aux_scalars or ())
    if not isinstance(window_size, (tuple, list)) or len(window_size) != 2:
        raise ValueError("window_size must contain (left, right)")
    if any(
        x is not None and (type(x) is not int or not -(2**31) <= x < 2**31) for x in window_size
    ):
        raise ValueError("window bounds must be None or signed int32")
    left, right = window_size
    if left is not None and right is not None and left < 0 and right < 0:
        left = right = None
    cap = float(softcap)
    if not math.isfinite(cap) or abs(cap) > _FLOAT32_MAX:
        raise ValueError("softcap must be a finite float32 value")
    if cap != 0 and struct.unpack("f", struct.pack("f", cap))[0] == 0:
        raise ValueError("softcap must remain nonzero when converted to float32")
    if cap != 0 and any(
        x is not None for x in (score_mod, score_mod_bwd, score_mod_simd, score_mod_bwd_simd)
    ):
        raise ValueError("softcap and score_mod/score_mod_bwd cannot be used together")
    score = prepare_mod(score_mod, "score", bool(scalars), score_mod_simd)
    grad = prepare_mod(score_mod_bwd, "score_bwd", bool(scalars), score_mod_bwd_simd)
    if cap != 0:
        score, grad = builtin.softcap_value(cap)
    if score is None and grad is not None:
        raise ValueError("score_mod_bwd requires score_mod")
    if training and score is not None and grad is None:
        raise ValueError("score_mod_bwd is required for training with a custom score_mod")
    mask = prepare_mod(mask_mod, "mask", bool(scalars), mask_mod_simd)
    if mask is None:
        if causal:
            right = 0
        if left is not None or right is not None:
            mask = builtin.local_window(left, right)
    callback_mode(score, mask, grad if training else None)
    config = _validate(
        q, k, v, aux, scalars, softmax_scale, cu_q, cu_k, max_q, max_k, layout=layout
    )
    if q.numel() == 0 or k.numel() == 0:
        blocks = [None, None]
    else:
        with torch.npu.device(q.device):
            blocks = [
                normalize_block_sparse_tensors(
                    s,
                    batch_size=config["B"],
                    num_heads=config["H"],
                    seqlen_q=config["Sq"],
                    seqlen_k=config["Sk"],
                    device=q.device,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    backward=backward,
                )
                for s, backward in ((block_sparse_tensors, False), (block_sparse_tensors_bwd, True))
            ]
    config.update(
        score_mod=score,
        score_mod_bwd=grad,
        mask_mod=mask,
        block_sparse_tensors=blocks[0],
        block_sparse_tensors_bwd=blocks[1],
        sparse_versions=sparse_tensor_versions(
            block_sparse_tensors, block_sparse_tensors_bwd, *blocks
        ),
        return_lse=return_lse,
    )
    return config


def _flash(q, k, v, *, input_requires_grad=None, autograd_func=None, **options):
    if torch.compiler.is_compiling():
        from .utils import ensure_backend

        ensure_backend()
        from .graph import attention

        return attention(q, k, v, input_requires_grad=input_requires_grad, **options)
    training = torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v))
    config = _prepare_config(q, k, v, training=training, **options)
    aux = config["aux_tensors"]
    cu_q, cu_k = config["cu_q"], config["cu_k"]
    if autograd_func is None:
        autograd_func = (
            FlashAttnVarlenFunc if cu_q is not None or cu_k is not None else FlashAttnFunc
        )
    out, lse = autograd_func.apply(q, k, v, config, *aux)
    if input_requires_grad is None:
        input_requires_grad = any(t.requires_grad for t in (q, k, v))
    return out, lse if (options["return_lse"] or input_requires_grad) else None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    block_sparse_tensors_bwd: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    gather_bwd_recompute_p: bool = False,
    gather_bwd_token_chunk: Optional[int] = None,
):
    """Compute fixed-length BSND attention with optional score and mask callbacks.

    Inputs must be FP16 or BF16 tensors on the same Ascend 950 NPU, with
    head dimension 128. Grouped-query and multi-query attention require
    ``Hq % Hkv == 0``. Unsupported options accept only the defaults
    documented below.

    Parameters
    ----------
    q : torch.Tensor
        Queries of shape ``(B, Sq, Hq, 128)`` in BSND layout.
    k : torch.Tensor
        Keys of shape ``(B, Sk, Hkv, 128)`` in BSND layout, with the same
        dtype and device as ``q``.
    v : torch.Tensor
        Values with the same shape, dtype, and device as ``k``.
    qv : None, optional
        Reserved MLA argument; only ``None`` is supported.
    gather_kv_indices : None, optional
        Reserved gather argument; only ``None`` is supported.
    softmax_scale : float or None, optional
        Finite scale within the FP32 range, applied to QK before score
        modification. Defaults to ``1 / sqrt(128)``. Zero and negative
        scales are allowed.
    causal : bool, optional
        Apply bottom-right causal masking: ``kv_idx <= q_idx + Sk - Sq``.
        Sets the right window bound to zero. Ignored with ``mask_mod``.
    window_size : tuple of (int or None, int or None), optional
        Inclusive left/right bounds around ``center = q_idx + Sk - Sq``:
        ``center - left <= kv_idx <= center + right``. ``None`` leaves
        that side unbounded. Two negative bounds together mean unbounded;
        a single negative bound retains its signed boundary semantics.
        Bounds must fit signed int32. Ignored with ``mask_mod``.
    learnable_sink : None, optional
        Reserved sink argument; only ``None`` is supported.
    softcap : float, optional
        Zero disables softcapping. A nonzero value applies
        ``softcap * tanh(scaled_score / softcap)`` with a built-in VJP.
        Positive and negative finite values within the FP32 range are
        supported, provided they remain nonzero after FP32 conversion.
        Cannot be combined with ``score_mod`` or ``score_mod_bwd``.
    num_splits : int, optional
        Reserved SplitKV scheduling argument; only ``1`` is supported.
    pack_gqa : None, optional
        Reserved scheduling argument; only ``None`` is supported.
        GQA/MQA remain supported through the input head counts.
    deterministic : bool, optional
        Reserved deterministic-backward argument; only ``False`` is supported.
    score_mod : callable or None, optional
        ``@tla.jit`` callback transforming the scaled score, with signature
        ``(s, b, h, q_idx, kv_idx, seqlen_info, aux_tensors[, aux_scalars])``.
        ``None`` leaves scores unchanged. Scores are modified before masking.
    score_mod_bwd : callable or None, optional
        Explicit ``@tla.jit`` score VJP with signature
        ``(g, s, b, h, q_idx, kv_idx, seqlen_info, aux_tensors[, aux_scalars])``.
        Return the gradient with respect to the original scaled score ``s``
        for upstream gradient ``g``. Required when autograd is enabled and
        any Q/K/V requires gradients with a custom ``score_mod``.
    mask_mod : callable or None, optional
        ``@tla.jit`` visibility predicate with signature
        ``(b, h, q_idx, kv_idx, seqlen_info, aux_tensors[, aux_scalars])``.
        ``True`` means visible. Overrides ``causal`` and ``window_size``.
    aux_tensors : list of torch.Tensor or None, optional
        Read-only callback inputs on ``q.device``, with nonnegative strides
        and no gradients. Supported dtypes are bool, int32, int64, uint32,
        float16, bfloat16, and float32.
    aux_scalars : tuple of bool, int, or float or None, optional
        Callback inputs: Python booleans, signed int32-range integers, or
        finite FP32-range floats. Nonempty inputs require callbacks to
        accept the final ``aux_scalars`` argument.
    block_sparse_tensors : BlockSparseTensorsTorch or None, optional
        Forward metadata organized by query block, with logical block size
        128 by 128. Full blocks skip the mask callback, partial blocks
        evaluate it, and omitted blocks are skipped. ``None`` uses dense
        traversal or a supported built-in window schedule.
    block_sparse_tensors_bwd : BlockSparseTensorsTorch or None, optional
        Backward metadata organized by KV block, as returned by
        :func:`transpose_block_sparsity`. When omitted, backward reuses the
        forward graph. Supplied forward/backward lists must be consistent.
    return_lse : bool, optional
        Request LSE even when no input requires gradients, and enable its
        upstream gradient in backward. When ``False``, LSE may still be
        returned for gradient-requiring inputs, but its gradient is ignored.
    gather_bwd_recompute_p : bool, optional
        Reserved gather-backward argument; only ``False`` is supported.
    gather_bwd_token_chunk : int or None, optional
        Reserved gather-backward chunk size; only ``None`` is supported.
        Invalid gather combinations are validated before capability checks.

    Returns
    -------
    out : torch.Tensor
        Attention output with the same shape, dtype, and device as ``q``.
    lse : torch.Tensor or None
        Contiguous FP32 natural-log log-sum-exp of shape ``(B, Hq, Sq)``.
        Returned when ``return_lse`` is true or any original Q/K/V input
        has ``requires_grad=True``, including inside ``torch.no_grad()``;
        otherwise ``None``. Autograd controls its ``requires_grad`` flag.
        Its gradient contributes only when ``return_lse=True``.

    Raises
    ------
    ValueError
        Inputs, dimensions, auxiliary data, sparse metadata, or gather
        combinations are invalid; score options conflict; or a custom
        training score has no VJP.
    TypeError
        Sparse metadata has an unsupported container type, or a removed
        extension parameter is passed.
    UnsupportedModError
        A callback is not a supported native TLA callable, has an
        incompatible signature, or callbacks have no common execution mode.
    NotImplementedError
        A reserved argument differs from its supported default, or sparse
        metadata requests unsupported block sizes or scheduling features.

    See Also
    --------
    flash_attn_varlen_func : Attention with independently fixed or packed Q/KV.
    compute_block_sparsity : Build explicit sparse block lists from a mask.

    Notes
    -----
    Non-contiguous or unaligned inputs are copied before entering the custom
    autograd function, preserving gradients to the original views. Contiguous
    inputs with 16-byte-aligned storage are reused. Empty batches and empty
    Q/KV sequences are supported; global empty inputs bypass the kernel.

    Callbacks receive a query-head index and segment-local sequence indices.
    ``seqlen_info`` is a :class:`SeqlenInfoQK`; fixed-side offsets are zero.
    Use ``@simt`` or ``@simd`` outside ``@tla.jit`` to mark the execution mode;
    an unmarked ``@tla.jit`` callback defaults to SIMT. Callbacks must share
    an available mode. Custom masks need explicit block metadata to skip
    blocks. Keep callback closures unchanged and rebuild sparse lists when
    their mask dependencies change.

    Only first-order Q/K/V gradients are supported; auxiliary inputs are
    non-differentiable. With ``return_lse=True``, out-only, LSE-only, and joint
    losses are supported; an LSE-only loss gives zero dV. Empty KV and fully
    masked rows with finite inputs produce zero output, ``-inf`` LSE, and
    zero row gradients. Nonfinite values follow kernel IEEE semantics.

    Examples
    --------
    Compute causal attention with grouped query heads on an NPU:

    >>> import torch
    >>> import torch_npu
    >>> from flash_attn_npu_4.catlass_dsl import flash_attn_func
    >>> q = torch.randn(2, 256, 4, 128, device="npu", dtype=torch.float16)
    >>> k = torch.randn(2, 256, 2, 128, device="npu", dtype=torch.float16)
    >>> out, lse = flash_attn_func(q, k, torch.randn_like(k), causal=True, return_lse=True)
    >>> tuple(out.shape), tuple(lse.shape)
    ((2, 256, 4, 128), (2, 4, 256))
    """
    gather_bwd_token_chunk = _validate_gather_bwd_kwargs(
        gather_kv_indices, gather_bwd_recompute_p, gather_bwd_token_chunk
    )
    _unsupported(
        qv=(qv, None),
        gather_kv_indices=(gather_kv_indices, None),
        learnable_sink=(learnable_sink, None),
        num_splits=(num_splits, 1),
        pack_gqa=(pack_gqa, None),
        deterministic=(deterministic, False),
        gather_bwd_recompute_p=(gather_bwd_recompute_p, False),
        gather_bwd_token_chunk=(gather_bwd_token_chunk, None),
    )
    if any((not isinstance(t, torch.Tensor) or t.ndim != 4 for t in (q, k, v))):
        raise ValueError("flash_attn_func requires BSND; use flash_attn_varlen_func for TND")
    input_requires_grad = any(t.requires_grad for t in (q, k, v))
    if not torch.compiler.is_compiling():
        q, k, v = (_canonical_input(t) for t in (q, k, v))
    return _flash(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        aux_scalars=aux_scalars,
        block_sparse_tensors=block_sparse_tensors,
        block_sparse_tensors_bwd=block_sparse_tensors_bwd,
        return_lse=return_lse,
        layout="BSND",
        input_requires_grad=input_requires_grad,
        autograd_func=FlashAttnFunc,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    return_lse: bool = False,
    scheduler_metadata: Optional[SchedulerMetadataTensorsTorch] = None,
    seqlen_k_per_split: Optional[int] = None,
    disable_scheduler_metadata: bool = False,
    gather_bwd_recompute_p: bool = False,
    gather_bwd_token_chunk: Optional[int] = None,
):
    """Compute attention with independently fixed BSND or packed TND Q/KV.

    Q and KV may independently use either layout and must describe the same
    batch. Inputs must be FP16 or BF16 tensors on the same Ascend 950 NPU,
    with head dimension 128 and ``Hq % Hkv == 0``. Unsupported options
    accept only the defaults documented below.

    Parameters
    ----------
    q : torch.Tensor
        Queries of shape ``(Tq, Hq, 128)`` for packed TND or
        ``(B, Sq, Hq, 128)`` for fixed BSND.
    k : torch.Tensor
        Keys of shape ``(Tk, Hkv, 128)`` for packed TND or
        ``(B, Sk, Hkv, 128)`` for fixed BSND. Dtype and device must match ``q``.
    v : torch.Tensor
        Values with the same shape, dtype, and device as ``k``.
    qv : None, optional
        Reserved MLA argument; only ``None`` is supported.
    cu_seqlens_q : torch.Tensor or None, optional
        Required for packed Q: contiguous int32 cumulative token offsets of
        shape ``(B + 1,)`` on ``q.device``. Must be ``None`` for fixed Q.
    cu_seqlens_k : torch.Tensor or None, optional
        Required for packed KV, with the same contract as ``cu_seqlens_q``.
        Must be ``None`` for fixed KV.
    max_seqlen_q : int, torch.Tensor, or None, optional
        Upper bound on packed query segment lengths. ``None`` uses total
        query tokens for packed Q or the actual sequence length for fixed Q.
        Tensor hints are ignored without reading their values. An integer
        must cover all actual lengths; fixed-layout
        addressing always uses the tensor shape. Zero is allowed for empty Q.
    max_seqlen_k : int, torch.Tensor, or None, optional
        Upper bound on packed KV segment lengths, with the same hint rules
        as ``max_seqlen_q``. ``None`` uses total KV tokens for packed KV or
        the actual length for fixed KV. Zero is allowed for empty KV.
    min_seqlen_k : None, optional
        Reserved minimum-KV-length argument; only ``None`` is supported.
    seqused_q : None, optional
        Reserved used-query-length argument; only ``None`` is supported.
    seqused_k : None, optional
        Reserved used-KV-length argument; only ``None`` is supported.
    gather_kv_indices : None, optional
        Reserved gather argument; only ``None`` is supported.
    page_table : None, optional
        Reserved paged-KV argument; only ``None`` is supported.
    softmax_scale : float or None, optional
        Finite scale within the FP32 range, applied to QK before score
        modification. Defaults to ``1 / sqrt(128)``; zero and negative
        scales are allowed.
    causal : bool, optional
        Apply bottom-right causal masking using each segment's actual lengths:
        ``kv_idx <= q_idx + seqlen_k - seqlen_q``. Sets the right window
        bound to zero. Ignored with ``mask_mod``.
    window_size : tuple of (int or None, int or None), optional
        Inclusive left/right bounds around the bottom-right aligned query
        position, using actual segment lengths. ``None`` leaves that side
        unbounded. Two negative bounds together mean unbounded; a single
        negative bound retains its signed boundary semantics. Bounds must
        fit signed int32. Ignored with ``mask_mod``.
    learnable_sink : None, optional
        Reserved sink argument; only ``None`` is supported.
    softcap : float, optional
        Zero disables softcapping. A nonzero value applies
        ``softcap * tanh(scaled_score / softcap)`` with a built-in VJP.
        Positive and negative finite values within the FP32 range are
        supported, provided they remain nonzero after FP32 conversion.
        Cannot be combined with ``score_mod`` or ``score_mod_bwd``.
    num_splits : int, optional
        Reserved SplitKV scheduling argument; only ``1`` is supported.
    pack_gqa : None, optional
        Reserved scheduling argument; only ``None`` is supported.
        GQA/MQA remain supported through the input head counts.
    deterministic : bool, optional
        Reserved deterministic-backward argument; only ``False`` is supported.
    score_mod : callable or None, optional
        ``@tla.jit`` scaled-score callback with signature
        ``(s, b, h, q_idx, kv_idx, seqlen_info, aux_tensors[, aux_scalars])``.
        ``None`` leaves scores unchanged. Scores are modified before masking.
    score_mod_bwd : callable or None, optional
        Explicit ``@tla.jit`` VJP with signature
        ``(g, s, b, h, q_idx, kv_idx, seqlen_info, aux_tensors[, aux_scalars])``.
        Returns the gradient with respect to the original scaled score ``s``
        for upstream gradient ``g``. Required when autograd is enabled and
        any Q/K/V requires gradients with a custom ``score_mod``.
    mask_mod : callable or None, optional
        ``@tla.jit`` visibility predicate with signature
        ``(b, h, q_idx, kv_idx, seqlen_info, aux_tensors[, aux_scalars])``.
        ``True`` means visible. Overrides ``causal`` and ``window_size``.
    block_sparse_tensors : BlockSparseTensorsTorch or None, optional
        Query-major metadata with logical block size 128 by 128. Full blocks
        skip the mask callback, partial blocks evaluate it, and omitted
        blocks are skipped. Backward uses the forward sparse graph; this
        entry does not accept a separate backward sparse-list argument.
    aux_tensors : list of torch.Tensor or None, optional
        Read-only callback inputs on ``q.device``, with nonnegative strides
        and no gradients. Supported dtypes are bool, int32, int64, uint32,
        float16, bfloat16, and float32.
    aux_scalars : tuple of bool, int, or float or None, optional
        Python booleans, signed int32-range integers, or finite FP32-range
        floats. Nonempty inputs require callbacks to accept ``aux_scalars``.
    return_lse : bool, optional
        Request LSE even when no input requires gradients, and enable its
        upstream gradient in backward. When ``False``, LSE may still be
        returned for gradient-requiring inputs, but its gradient is ignored.
    scheduler_metadata : None, optional
        Reserved scheduling metadata; only ``None`` is supported.
    seqlen_k_per_split : None, optional
        Reserved per-split sequence length; only ``None`` is supported.
    disable_scheduler_metadata : bool, optional
        Reserved scheduling argument; only ``False`` is supported.
    gather_bwd_recompute_p : bool, optional
        Reserved gather-backward argument; only ``False`` is supported.
    gather_bwd_token_chunk : int or None, optional
        Reserved gather-backward chunk size; only ``None`` is supported.
        Invalid gather combinations are validated before capability checks.

    Returns
    -------
    out : torch.Tensor
        Output with the same shape, dtype, device, and layout as ``q``.
    lse : torch.Tensor or None
        Contiguous FP32 natural-log log-sum-exp: ``(Hq, Tq)`` for packed Q
        or ``(B, Hq, Sq)`` for fixed Q. Returned when ``return_lse`` is true
        or any original Q/K/V input has ``requires_grad=True``, including
        inside ``torch.no_grad()``; otherwise ``None``. Autograd controls its
        ``requires_grad`` flag. Its gradient contributes only when
        ``return_lse=True``.

    Raises
    ------
    ValueError
        Input layouts, dimensions, length metadata, auxiliary inputs, sparse
        metadata, or gather combinations are invalid; score options conflict;
        or a custom training score has no VJP.
    TypeError
        An attention input is not a tensor, sparse metadata has an unsupported
        container type, or a removed extension parameter is passed.
    UnsupportedModError
        A callback is not a supported native TLA callable, has an
        incompatible signature, or callbacks have no common execution mode.
    NotImplementedError
        Reserved options differ from their supported defaults, or sparse
        metadata requests unsupported features.

    See Also
    --------
    flash_attn_func : Fixed-length attention with the same score/mask semantics.
    compute_block_sparsity : Construct fixed or packed sparse metadata.

    Notes
    -----
    Cumulative offsets must start at zero, be nondecreasing, end at the
    corresponding total token count, and describe segments no longer than
    an explicit integer maximum. NPU calls do not read these values back to
    validate them; the caller must maintain these invariants. Zero-length
    segments, empty batches, and empty Q/KV inputs are supported. Global
    empty inputs bypass the kernel.

    Callbacks receive a query-head index and segment-local sequence indices.
    ``seqlen_info`` is a :class:`SeqlenInfoQK` with actual segment lengths and
    packed token offsets; fixed-side offsets are zero. Use ``@simt`` or
    ``@simd`` outside ``@tla.jit`` to mark the execution mode; unmarked JIT
    callbacks default to SIMT. Callbacks must share an available mode. Custom
    masks need explicit block metadata to skip blocks. Keep callback closures
    unchanged and rebuild sparse lists when their mask dependencies change.

    Non-contiguous or unaligned inputs are copied before entering the custom
    autograd function, preserving gradients to the original views. Contiguous
    inputs with 16-byte-aligned storage are reused. Only first-order Q/K/V
    gradients are supported; auxiliary inputs are non-differentiable. With
    ``return_lse=True``, out-only, LSE-only, and joint losses are supported;
    an LSE-only loss gives zero dV. Empty KV and fully masked rows with finite
    inputs produce zero output, ``-inf`` LSE, and zero row gradients.
    Nonfinite values follow kernel IEEE semantics.

    Examples
    --------
    Pack two query and KV sequences with different lengths:

    >>> import torch
    >>> import torch_npu
    >>> from flash_attn_npu_4.catlass_dsl import flash_attn_varlen_func
    >>> cu_q = torch.tensor([0, 16, 40], device="npu", dtype=torch.int32)
    >>> cu_k = torch.tensor([0, 20, 52], device="npu", dtype=torch.int32)
    >>> q = torch.randn(40, 4, 128, device="npu", dtype=torch.float16)
    >>> k = torch.randn(52, 2, 128, device="npu", dtype=torch.float16)
    >>> out, lse = flash_attn_varlen_func(
    ...     q, k, torch.randn_like(k), cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
    ...     max_seqlen_q=24, max_seqlen_k=32, causal=True, return_lse=True)
    >>> tuple(out.shape), tuple(lse.shape)
    ((40, 4, 128), (4, 40))
    """
    gather_bwd_token_chunk = _validate_gather_bwd_kwargs(
        gather_kv_indices, gather_bwd_recompute_p, gather_bwd_token_chunk
    )
    _unsupported(
        qv=(qv, None),
        min_seqlen_k=(min_seqlen_k, None),
        seqused_q=(seqused_q, None),
        seqused_k=(seqused_k, None),
        gather_kv_indices=(gather_kv_indices, None),
        page_table=(page_table, None),
        learnable_sink=(learnable_sink, None),
        num_splits=(num_splits, 1),
        pack_gqa=(pack_gqa, None),
        deterministic=(deterministic, False),
        scheduler_metadata=(scheduler_metadata, None),
        seqlen_k_per_split=(seqlen_k_per_split, None),
        disable_scheduler_metadata=(disable_scheduler_metadata, False),
        gather_bwd_recompute_p=(gather_bwd_recompute_p, False),
        gather_bwd_token_chunk=(gather_bwd_token_chunk, None),
    )
    input_requires_grad = any(isinstance(t, torch.Tensor) and t.requires_grad for t in (q, k, v))
    if not torch.compiler.is_compiling():
        q, k, v = (_canonical_input(t) for t in (q, k, v))
        max_seqlen_q = _maximum_length(q, max_seqlen_q)
        max_seqlen_k = _maximum_length(k, max_seqlen_k)
    return _flash(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        aux_scalars=aux_scalars,
        block_sparse_tensors=block_sparse_tensors,
        block_sparse_tensors_bwd=None,
        return_lse=return_lse,
        layout="BSND",
        input_requires_grad=input_requires_grad,
        autograd_func=FlashAttnVarlenFunc,
        cu_q=cu_seqlens_q,
        cu_k=cu_seqlens_k,
        max_q=max_seqlen_q,
        max_k=max_seqlen_k,
    )
