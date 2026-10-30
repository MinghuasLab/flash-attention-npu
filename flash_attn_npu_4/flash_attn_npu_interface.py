# Copyright (c) 2023, Tri Dao.
# Modified by Minghua Shen, 2026

from typing import Any, Callable, Optional, Tuple

import torch

# isort: off
# We need to import the kernels after importing torch
from . import flash_attn_npu_4  # Registers operators with PyTorch

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
            "flash_attn_npu_4: key and value must have the same rank for layout detection"
        )

    if q.dim() != 4 or k.dim() != 4 or page_table is not None or cu_seqlens_k is not None:
        return _LAYOUT_BSND, q, k, v, False

    q_bnsd = _transposed_bnsd_view(q)
    k_bnsd = _transposed_bnsd_view(k)
    v_bnsd = _transposed_bnsd_view(v)
    if q_bnsd is not None and k_bnsd is not None and v_bnsd is not None:
        return _LAYOUT_BNSD, q_bnsd, k_bnsd, v_bnsd, True

    return _LAYOUT_BSND, q, k, v, False


def _paged_kv_inner_layout(x):
    """Return the packed inner layout of a logical [page, token, head, dim] cache."""
    if x is None or x.dim() != 4:
        return None
    page_size, num_heads, head_dim = x.shape[1:]
    if (
        x.stride(1) == num_heads * head_dim
        and x.stride(2) == head_dim
        and x.stride(3) == 1
    ):
        return "BBND"
    if (
        x.stride(1) == head_dim
        and x.stride(2) == page_size * head_dim
        and x.stride(3) == 1
    ):
        return "BNBD"
    return None


def _prepare_forward_qkv(q, k, v, page_table):
    """Match FIA's stride policy before entering the packed FA4 kernel.

    Q and dense K/V are materialized because the kernel uses fixed packed
    strides. Paged K/V keep their zero-copy path when both caches have the
    same packed BBND or BNBD inner layout; dimension 0 may still be strided.
    Unsupported paged inner layouts fall back to contiguous logical BBND.
    """
    q = q.contiguous()
    if page_table is None:
        return q, k.contiguous(), v.contiguous()

    key_layout = _paged_kv_inner_layout(k)
    value_layout = _paged_kv_inner_layout(v)
    if key_layout is not None and key_layout == value_layout:
        return q, k, v
    return q, k.contiguous(), v.contiguous()


def round_multiple(x, m):
    return (x + m - 1) // m * m


_HEADDIM_BWD_ALIGN = 64


def _pad_bwd_headdim(dout, q, k, v, out, head_size_og):
    """Pad headdim to a multiple of 64 for the FAG bwd kernel."""
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


def _validate_varlen_query(q, cu_seqlens_q, *, max_seqlen_q=None):
    if q.dim() != 3:
        raise RuntimeError(f"q must be 3-dimensional; got {q.dim()} dimensions")
    if cu_seqlens_q.numel() <= 1:
        raise RuntimeError("batch size must be positive")
    # max_seqlen_q == 0 is the paged TND fast path, where every sequence has
    # qlen 0. A nonzero declared max length with an empty packed query is
    # inconsistent.
    if q.shape[0] == 0 and max_seqlen_q != 0:
        raise RuntimeError("total_q must be positive (q must contain at least one query token)")


def _window_to_npu(window_size: Optional[int]) -> int:
    """FA4 uses None; FAG kernel uses -1 for 'disabled'."""
    return -1 if window_size is None else int(window_size)


@_torch_custom_op_wrapper(
    "flash_attn_npu_4::_flash_attn_forward",
    mutates_args=(),
    device_types="npu",
)
def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    scheduler_metadata: Optional[torch.Tensor] = None,
    sm_margin: int = 0,
    layout: int = _LAYOUT_BSND,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens_q is not None:
        _validate_varlen_query(q, cu_seqlens_q, max_seqlen_q=max_seqlen_q)
    q, k, v = _prepare_forward_qkv(q, k, v, page_table)
    cu_seqlens_q, cu_seqlens_k = [maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k)]
    seqused_q, seqused_k = [maybe_contiguous(x) for x in (seqused_q, seqused_k)]
    page_table = maybe_contiguous(page_table)
    out, softmax_lse = flash_attn_npu_4.fwd(
        q,
        k,
        v,
        qv,
        out_,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_k,
        page_table,
        gather_kv_indices,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        num_splits,
        pack_gqa,
        learnable_sink,
        scheduler_metadata,
        sm_margin,
        layout,
    )
    return out, softmax_lse


# ----------------------------------------------------------------------
# Scheduler metadata torch.compile support
#
# Keep these constants in sync with:
#   csrc/ascend910/flash_attn_npu_4/fa_metadata_args.h
#
# Real metadata contract:
#
#   all mask types:
#       shape = (2376,)
#
#   dtype  = torch.uint8
#   device = NPU
#   stride = (1,)
#
# The custom op prevents TorchDynamo from tracing into the raw pybind
# get_scheduler_metadata() implementation.
# ----------------------------------------------------------------------

_SCHEDULER_METADATA_TILING_BYTES = 2336


@torch.library.custom_op(
    "flash_attn_npu_4::_get_scheduler_metadata",
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
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    num_splits: int,
    pack_gqa: Optional[bool],
    sm_margin: int,
    seqused_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Keep the raw pybind call behind a torch.library op. Calling a pybind
    # function directly from a torch.compile graph is treated as an opaque
    # Python/C++ builtin and causes a Dynamo graph break.
    return flash_attn_npu_4.get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        headdim_v,
        cache_seqlens,
        cu_seqlens_q,
        causal,
        window_size_left,
        window_size_right,
        num_splits,
        pack_gqa,
        sm_margin,
        seqused_q,
    )


@_torch_register_fake_wrapper("flash_attn_npu_4::_get_scheduler_metadata")
def _get_scheduler_metadata_fake(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    headdim: int,
    headdim_v: int,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    num_splits: int,
    pack_gqa: Optional[bool],
    sm_margin: int,
    seqused_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # The metadata is an opaque byte buffer during tracing. Ascend 910 FA4
    # serializes one fixed-size FAInferStaticTilingData object, so returning a
    # static shape lets Dynamo keep the custom op in the graph.
    return torch.empty(
        (_SCHEDULER_METADATA_TILING_BYTES,),
        dtype=torch.uint8,
        device=cache_seqlens.device,
    )


def get_scheduler_metadata(
    max_seqlen_q,
    max_seqlen_k,
    nheads,
    nheads_kv,
    headdim,
    num_splits,
    headdim_v=None,
    pack_gqa=None,
    causal=False,
    window_size_left=None,
    window_size_right=None,
    seqlen_k_new=0,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    seqlen_k_per_split: Optional[int] = None,
    _arch: Optional[int] = None,
):
    """Precompute Ascend 910 FA4 metadata using the Tri Dao FA4 signature.

    Ascend FA4 does not support KV updates, padded BSND Q lengths, left padding,
    or fixed split lengths. Those Tri Dao parameters remain as compatibility
    placeholders and must retain their default unsupported values.
    """
    if seqlen_k_new != 0:
        raise ValueError("Ascend 910 FA4 does not support seqlen_k_new")
    if cu_seqlens_k_new is not None:
        raise ValueError("Ascend 910 FA4 does not support cu_seqlens_k_new")
    if leftpad_k is not None:
        raise ValueError("Ascend 910 FA4 does not support leftpad_k")
    if seqlen_k_per_split is not None:
        raise ValueError("Ascend 910 FA4 does not support seqlen_k_per_split")
    if _arch is not None:
        raise ValueError("Ascend 910 FA4 does not accept _arch")
    if cu_seqlens_q is not None:
        cu_seqlens_q = maybe_contiguous(cu_seqlens_q)
    if cu_seqlens_k is not None:
        cu_seqlens_k = maybe_contiguous(cu_seqlens_k)
    if seqused_q is not None:
        seqused_q = maybe_contiguous(seqused_q)
    if seqused_k is None:
        if cu_seqlens_k is None:
            raise ValueError("seqused_k or cu_seqlens_k is required")
        seqused_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    seqused_k = maybe_contiguous(seqused_k)
    batch_size = seqused_k.numel()
    if cu_seqlens_q is not None and cu_seqlens_q.numel() != batch_size + 1:
        raise ValueError("cu_seqlens_q batch size does not match seqused_k")
    if seqused_q is not None and seqused_q.numel() != batch_size:
        raise ValueError("seqused_q batch size does not match seqused_k")
    if headdim_v is None:
        headdim_v = headdim
    if window_size_left is None:
        window_size_left = -1
    if window_size_right is None:
        window_size_right = -1
    if pack_gqa is not None and pack_gqa:
        raise ValueError("Ascend 910 FA4 does not support pack_gqa=True")
    return _get_scheduler_metadata_op(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        nheads,
        nheads_kv,
        headdim,
        headdim_v,
        seqused_k,
        cu_seqlens_q,
        causal,
        window_size_left,
        window_size_right,
        num_splits,
        pack_gqa,
        0,
        seqused_q,
    )


@_torch_register_fake_wrapper("flash_attn_npu_4::_flash_attn_forward")
def _flash_attn_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    scheduler_metadata: Optional[torch.Tensor] = None,
    sm_margin: int = 0,
    layout: int = _LAYOUT_BSND,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Metadata-only fake for V4 A2 forward. Returns (out, lse)."""
    if out_ is not None:
        raise TypeError(
            "Tracing (torch.compile/torch.export) with pre-allocated output tensor is not supported."
        )

    is_varlen_q = cu_seqlens_q is not None
    if is_varlen_q:
        _validate_varlen_query(q, cu_seqlens_q, max_seqlen_q=max_seqlen_q)
    if not is_varlen_q and layout == _LAYOUT_BNSD:
        out = torch.empty(
            (q.shape[0], q.shape[2], q.shape[1], q.shape[3]),
            dtype=q.dtype,
            device=q.device,
        )
    else:
        out = torch.empty_like(q)

    if is_varlen_q:
        # (num_heads, total_q)
        num_heads = q.shape[1]
        total_q = q.shape[0]
        softmax_lse = torch.empty((num_heads, total_q), dtype=torch.float32, device=q.device)
    else:
        # (batch_size, num_heads, seqlen_q); q is (B, H, S, D) in BNSD.
        if layout == _LAYOUT_BNSD:
            batch_size = q.shape[0]
            num_heads = q.shape[1]
            seqlen_q = q.shape[2]
        else:
            batch_size = q.shape[0]
            seqlen_q = q.shape[1]
            num_heads = q.shape[2]
        softmax_lse = torch.empty(
            (batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device
        )

    return out, softmax_lse


@_torch_custom_op_wrapper(
    "flash_attn_npu_4::_flash_attn_backward_op",
    mutates_args=("dq", "dk", "dv"),
    device_types="npu",
)
def _flash_attn_backward_op(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    deterministic: bool,
) -> torch.Tensor:
    # Same storage detection as forward; retain logical BSND shapes for host/output.
    if cu_seqlens_q is not None:
        layout = _LAYOUT_TND
    else:
        layout = _detect_layout(q, k, v, None, cu_seqlens_k)[0]

    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    _dq, _dk, _dv, softmax_d = flash_attn_npu_4.bwd(
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
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        0,  # sm_margin
        layout,
    )
    return softmax_d


@_torch_register_fake_wrapper("flash_attn_npu_4::_flash_attn_backward_op")
def _flash_attn_backward_op_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    deterministic: bool,
) -> torch.Tensor:
    """Metadata-only fake for V4 A2 backward_op. Returns softmax_d; mutates dq/dk/dv."""
    is_varlen_q = cu_seqlens_q is not None
    if is_varlen_q:
        batch_size = cu_seqlens_q.shape[0] - 1
        nheads = q.shape[1]
        # Real mha_bwd always allocates (batch, nheads, max_seqlen_q).
        seqlen_q = max_seqlen_q
    else:
        batch_size = q.shape[0]
        nheads = q.shape[2]
        seqlen_q = q.shape[1]

    softmax_d = torch.empty((batch_size, nheads, seqlen_q), dtype=torch.float32, device=q.device)
    return softmax_d


def _flash_attn_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    m_block_size: int = 64,
    n_block_size: int = 128,
    num_threads: int = 256,
    pack_gqa: bool = False,
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    SdP_swapAB: bool = False,
    dKV_swapAB: bool = False,
    dQ_swapAB: bool = False,
    AtomLayoutMSdP: int = 2,
    AtomLayoutNdKV: int = 2,
    AtomLayoutMdQ: int = 2,
    V_in_regs: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    deterministic: bool = False,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    block_sparse_tensors: Optional[Any] = None,
    dlse: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    learnable_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FA4-aligned backward wrapper around FAG_v4. Returns (dq, dk, dv)."""
    del (
        m_block_size,
        n_block_size,
        num_threads,
        num_stages_Q,
        num_stages_dO,
        SdP_swapAB,
        dKV_swapAB,
        dQ_swapAB,
        AtomLayoutMSdP,
        AtomLayoutNdKV,
        AtomLayoutMdQ,
        V_in_regs,
    )

    # Unsupported FA4 / Phase-0 knobs: assert here.
    assert score_mod is None, "flash_attn_npu_v4 bwd does not support score_mod"
    assert score_mod_bwd is None, "flash_attn_npu_v4 bwd does not support score_mod_bwd"
    assert mask_mod is None, "flash_attn_npu_v4 bwd does not support mask_mod"
    assert aux_tensors is None, "flash_attn_npu_v4 bwd does not support aux_tensors"
    assert aux_scalars is None, "flash_attn_npu_v4 bwd does not support aux_scalars"
    assert block_sparse_tensors is None, (
        "flash_attn_npu_v4 bwd does not support block_sparse_tensors"
    )
    assert dlse is None, "flash_attn_npu_v4 bwd does not support dlse"
    assert not pack_gqa, "flash_attn_npu_v4 bwd does not support pack_gqa=True"
    assert qv is None, "flash_attn_npu_v4 bwd does not support qv"
    assert page_table is None, "flash_attn_npu_v4 bwd does not support page_table"
    assert gather_kv_indices is None, "flash_attn_npu_v4 bwd does not support gather_kv_indices"
    assert learnable_sink is None, "flash_attn_npu_v4 bwd does not support learnable_sink"

    if dq is None:
        dq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if dk is None:
        dk = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if dv is None:
        dv = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    _flash_attn_backward_op(
        dout,
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        dq,
        dk,
        dv,
        softmax_scale,
        causal,
        _window_to_npu(window_size_left),
        _window_to_npu(window_size_right),
        softcap,
        deterministic,
    )
    return dq, dk, dv


_flash_attn_bwd = _flash_attn_backward


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        qv=None,
        gather_kv_indices=None,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        learnable_sink=None,
        softcap=0.0,  # 0.0 means deactivated
        num_splits=1,  # Can be tuned for speed
        pack_gqa=None,  # Can be tuned for speed
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
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)

        # Compute scheduler metadata on the AICPU (no D2H/H2D sync), matching
        # the v3 dense path. cache_seqlens holds the per-batch KV lengths.
        batch_size, seqlen_q, num_heads, head_size = q.shape
        seqlen_k = k.shape[1]
        num_heads_k = k.shape[2]
        head_size_v = v.shape[-1]
        # get_scheduler_metadata only takes numbers and reads them as BSND, so the
        # sizes above come from the pre-detection shapes; BNSD only re-views the
        # same storage.
        layout, q, k, v, out_transposed = _detect_layout(q, k, v, None, None)
        seqused_k = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=q.device)
        scheduler_metadata = get_scheduler_metadata(
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            nheads=num_heads,
            nheads_kv=num_heads_k,
            headdim=head_size,
            num_splits=num_splits,
            headdim_v=head_size_v,
            pack_gqa=pack_gqa,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            seqused_k=seqused_k,
        )

        out, softmax_lse = _flash_attn_forward(
            q,
            k,
            v,
            qv,
            None,  # out_
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # seqused_q
            None,  # seqused_k
            None,  # max_seqlen_q
            None,  # max_seqlen_k
            None,  # min_seqlen_k
            None,  # page_table
            gather_kv_indices,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            learnable_sink=learnable_sink,
            scheduler_metadata=scheduler_metadata,
            sm_margin=0,
            layout=layout,
        )

        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.return_lse = return_lse
        ctx.pack_gqa = pack_gqa
        ctx.qv = qv
        ctx.gather_kv_indices = gather_kv_indices
        ctx.learnable_sink = learnable_sink
        ctx.score_mod = score_mod
        ctx.score_mod_bwd = score_mod_bwd
        ctx.mask_mod = mask_mod
        ctx.block_sparse_tensors = block_sparse_tensors
        ctx.aux_tensors = aux_tensors
        ctx.aux_scalars = aux_scalars
        ctx.head_size_og = q.size(-1)
        return (out, softmax_lse) if return_lse else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        # torch_npu may pass a zero tensor (not None) for unused LSE grads.
        dlse = args[0] if ctx.return_lse and len(args) > 0 else None
        if dlse is not None and torch.is_tensor(dlse) and float(dlse.detach().abs().sum()) == 0.0:
            dlse = None
        win_l, win_r = ctx.window_size
        if win_l is not None and win_l < 0:
            win_l = None
        if win_r is not None and win_r < 0:
            win_r = None

        dout, q, k, v, out, head_size_og = _pad_bwd_headdim(dout, q, k, v, out, ctx.head_size_og)
        dq, dk, dv = _flash_attn_backward(
            q,
            k,
            v,
            out,
            dout,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            softcap=ctx.softcap,
            window_size_left=win_l,
            window_size_right=win_r,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            deterministic=ctx.deterministic,
            pack_gqa=bool(ctx.pack_gqa) if ctx.pack_gqa is not None else False,
            score_mod=ctx.score_mod,
            score_mod_bwd=ctx.score_mod_bwd,
            mask_mod=ctx.mask_mod,
            aux_tensors=ctx.aux_tensors,
            aux_scalars=ctx.aux_scalars,
            block_sparse_tensors=ctx.block_sparse_tensors,
            dlse=dlse,
            qv=ctx.qv,
            page_table=None,
            gather_kv_indices=ctx.gather_kv_indices,
            learnable_sink=ctx.learnable_sink,
        )
        dq = dq[..., :head_size_og]
        dk = dk[..., :head_size_og]
        dv = dv[..., :head_size_og]
        return (
            dq,
            dk,
            dv,
            None,  # qv
            None,  # gather_kv_indices
            None,  # softmax_scale
            None,  # causal
            None,  # window_size
            None,  # learnable_sink
            None,  # softcap
            None,  # num_splits
            None,  # pack_gqa
            None,  # deterministic
            None,  # score_mod
            None,  # score_mod_bwd
            None,  # mask_mod
            None,  # aux_tensors
            None,  # aux_scalars
            None,  # block_sparse_tensors
            None,  # block_sparse_tensors_bwd
            None,  # return_lse
        )


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        qv=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        min_seqlen_k=None,
        seqused_q=None,
        seqused_k=None,
        gather_kv_indices=None,
        page_table=None,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        learnable_sink=None,
        softcap=0.0,  # 0.0 means deactivated
        num_splits=0,  # Can be tuned for speed
        pack_gqa=None,  # Can be tuned for speed
        deterministic=False,
        score_mod=None,
        score_mod_bwd=None,
        mask_mod=None,
        block_sparse_tensors=None,
        aux_tensors=None,
        aux_scalars=None,
        return_lse=False,
        scheduler_metadata=None,
        seqlen_k_per_split=None,
        disable_scheduler_metadata=False,
    ):
        if cu_seqlens_q is not None:
            _validate_varlen_query(q, cu_seqlens_q, max_seqlen_q=max_seqlen_q)
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
        # cu_seqlens_q packs the query (TND); otherwise the rank-4 data is BSND.
        layout = _LAYOUT_TND if cu_seqlens_q is not None else _LAYOUT_BSND
        if seqused_k is not None and isinstance(seqused_k, int):
            seqused_k = torch.full((q.shape[0],), seqused_k, dtype=torch.int32, device=k.device)
            seqused_k = maybe_contiguous(seqused_k)

        fwd_max_seqlen_k = max_seqlen_k
        if scheduler_metadata is None and not disable_scheduler_metadata and q.shape[0] != 0:
            is_varlen_q = cu_seqlens_q is not None
            num_heads = q.shape[1] if is_varlen_q else q.shape[2]
            head_size = q.shape[2] if is_varlen_q else q.shape[3]
            head_size_v = v.shape[-1]
            num_heads_k = k.shape[1] if k.dim() == 3 else k.shape[2]
            if seqused_k is not None:
                cache_seqlens = seqused_k
            elif not is_varlen_q:
                cache_seqlens = torch.full(
                    (q.shape[0],), k.shape[1], dtype=torch.int32, device=q.device
                )
            else:
                cache_seqlens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            if seqused_q is not None:
                seqlens_q_arg = seqused_q
            elif not is_varlen_q:
                seqlens_q_arg = None
            else:
                seqlens_q_arg = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            metadata_max_seqlen_q = max_seqlen_q if is_varlen_q else q.shape[1]
            metadata_max_seqlen_k = max_seqlen_k
            if metadata_max_seqlen_k is None and not is_varlen_q and page_table is None:
                metadata_max_seqlen_k = k.shape[1]
            if metadata_max_seqlen_k is None and page_table is not None and k.dim() == 4:
                # Preserve the paged-KV window rule: infer the bound from used
                # lengths, so unused cache capacity cannot change the mask.
                metadata_max_seqlen_k = int(cache_seqlens.max().item())
                fwd_max_seqlen_k = metadata_max_seqlen_k
            scheduler_metadata = get_scheduler_metadata(
                max_seqlen_q=metadata_max_seqlen_q,
                max_seqlen_k=metadata_max_seqlen_k,
                nheads=num_heads,
                nheads_kv=num_heads_k,
                headdim=head_size,
                num_splits=num_splits,
                headdim_v=head_size_v,
                pack_gqa=pack_gqa,
                causal=causal,
                window_size_left=window_size[0],
                window_size_right=window_size[1],
                seqused_k=cache_seqlens,
                seqused_q=seqlens_q_arg,
            )

        out, softmax_lse = _flash_attn_forward(
            q,
            k,
            v,
            qv,
            None,  # out_
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            fwd_max_seqlen_k,
            min_seqlen_k,
            page_table,
            gather_kv_indices,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            learnable_sink=learnable_sink,
            scheduler_metadata=scheduler_metadata,
            sm_margin=0,
            layout=layout,
        )

        ctx.save_for_backward(
            q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.return_lse = return_lse
        ctx.pack_gqa = pack_gqa
        ctx.qv = qv
        ctx.page_table = page_table
        ctx.gather_kv_indices = gather_kv_indices
        ctx.learnable_sink = learnable_sink
        ctx.score_mod = score_mod
        ctx.score_mod_bwd = score_mod_bwd
        ctx.mask_mod = mask_mod
        ctx.block_sparse_tensors = block_sparse_tensors
        ctx.aux_tensors = aux_tensors
        ctx.aux_scalars = aux_scalars
        ctx.head_size_og = q.size(-1)

        # Do not materialize unused output gradients as zero tensors.
        # V4 backward does not support dlse; unused LSE gradients
        # should therefore arrive in backward as None.
        ctx.set_materialize_grads(False)

        # softmax_lse is an auxiliary output; V4 backward does not
        # support gradients with respect to LSE.
        if return_lse:
            ctx.mark_non_differentiable(softmax_lse)
        return (out, softmax_lse) if return_lse else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = (
            ctx.saved_tensors
        )
        # softmax_lse is non-differentiable in V4.
        # The underlying backward operator does not support dlse.
        dlse = None
        win_l, win_r = ctx.window_size
        if win_l is not None and win_l < 0:
            win_l = None
        if win_r is not None and win_r < 0:
            win_r = None

        dout, q, k, v, out, head_size_og = _pad_bwd_headdim(dout, q, k, v, out, ctx.head_size_og)
        dq, dk, dv = _flash_attn_backward(
            q,
            k,
            v,
            out,
            dout,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            softcap=ctx.softcap,
            window_size_left=win_l,
            window_size_right=win_r,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            deterministic=ctx.deterministic,
            pack_gqa=bool(ctx.pack_gqa) if ctx.pack_gqa is not None else False,
            score_mod=ctx.score_mod,
            score_mod_bwd=ctx.score_mod_bwd,
            mask_mod=ctx.mask_mod,
            aux_tensors=ctx.aux_tensors,
            aux_scalars=ctx.aux_scalars,
            block_sparse_tensors=ctx.block_sparse_tensors,
            dlse=dlse,
            qv=ctx.qv,
            page_table=ctx.page_table,
            gather_kv_indices=ctx.gather_kv_indices,
            learnable_sink=ctx.learnable_sink,
        )
        dq = dq[..., :head_size_og]
        dk = dk[..., :head_size_og]
        dv = dv[..., :head_size_og]
        return (
            dq,
            dk,
            dv,
            None,  # qv
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # max_seqlen_q
            None,  # max_seqlen_k
            None,  # min_seqlen_k
            None,  # seqused_q
            None,  # seqused_k
            None,  # gather_kv_indices
            None,  # page_table
            None,  # softmax_scale
            None,  # causal
            None,  # window_size
            None,  # learnable_sink
            None,  # softcap
            None,  # num_splits
            None,  # pack_gqa
            None,  # deterministic
            None,  # score_mod
            None,  # score_mod_bwd
            None,  # mask_mod
            None,  # block_sparse_tensors
            None,  # aux_tensors
            None,  # aux_scalars
            None,  # return_lse
            None,  # scheduler_metadata
            None,  # seqlen_k_per_split
            None,  # disable_scheduler_metadata
        )


def flash_attn_func(
    q,
    k,
    v,
    qv=None,
    gather_kv_indices=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    learnable_sink=None,
    softcap=0.0,  # 0.0 means deactivated
    num_splits=1,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
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
    """Forward pass of FlashAttention for dense (batch-major) inputs.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV
    with fewer heads than Q. The number of heads in Q must be divisible by the
    number of heads in KV. For example, if Q has 6 heads and K, V have 2 heads,
    heads 0, 1, 2 of Q attend to head 0 of K, V, and heads 3, 4, 5 of Q attend to
    head 1 of K, V.

    The scheduler metadata (tiling + attention mask) is computed on the AICPU so
    no device->host->device sync breaks the training pipeline.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim_v)
        qv [optional]: (batch_size, seqlen, nheads, headdim_v). Used for cross-attention.
        gather_kv_indices [optional]: (Not supported on NPU)
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Defaults to 1 / sqrt(headdim + (headdim_v if qv is not None else 0)).
        causal: bool. Whether to apply causal attention mask.
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        learnable_sink [optional]: (num_heads,), dtype bfloat16. One sink logit per query head.
            Added to the softmax denominator. The sink has no value contribution.
        softcap: float. Anything > 0 activates softcapping attention.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
        pack_gqa: bool. If True, pack GQA for better performance. (Not supported on NPU)
        deterministic: bool. Whether to use the deterministic backward pass.
        score_mod / score_mod_bwd / mask_mod: (Not supported on NPU)
        block_sparse_tensors / block_sparse_tensors_bwd: (Not supported on NPU)
        aux_tensors / aux_scalars: (Not supported on NPU)
        return_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim_v).
        softmax_lse [optional, if return_lse=True]: (batch_size, nheads, seqlen).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        qv,
        gather_kv_indices,
        softmax_scale,
        causal,
        window_size,  # -1 means infinite context window
        learnable_sink,
        softcap,  # 0.0 means deactivated
        num_splits,  # Can be tuned for speed
        pack_gqa,  # Can be tuned for speed
        deterministic,
        score_mod,
        score_mod_bwd,
        mask_mod,
        aux_tensors,
        aux_scalars,
        block_sparse_tensors,
        block_sparse_tensors_bwd,
        return_lse,
    )


def flash_attn_varlen_func(
    q,
    k,
    v,
    qv=None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    seqused_q=None,
    seqused_k=None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal: bool = False,
    window_size=(-1, -1),  # -1 means infinite context window
    learnable_sink: Optional[torch.Tensor] = None,
    softcap=0.0,  # 0.0 means deactivated
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    deterministic: bool = False,
    score_mod=None,
    score_mod_bwd=None,
    mask_mod=None,
    block_sparse_tensors=None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    return_lse: bool = False,
    scheduler_metadata=None,
    seqlen_k_per_split: Optional[int] = None,
    disable_scheduler_metadata: bool = False,
):
    """
    FlashAttention for variable-length sequences with optional paged KV cache.

    If cu_seqlens_q is provided, the input is treated as varlen (packed) format,
    where all sequences are concatenated along the sequence dimension. Otherwise,
    q, k, v are treated as dense tensors of shape (batch_size, seqlen, nheads, headdim).

    For paged KV cache, pass page_table and shape k/v as
    (num_pages, page_size, nheads_k, headdim).
    For physical BNBD storage shaped (num_pages, nheads_k, page_size, headdim), pass
    ``k.transpose(1, 2)`` and ``v.transpose(1, 2)``. The transpose is a metadata-only
    view; FA4 reads the original BNBD storage directly. The physical-page axis may
    be non-contiguous and may use different strides for K and V, while the selected
    layout's inner dimensions must be dense. TND batches may contain zero-length Q
    or KV sequences; an all-zero-Q batch returns empty output/LSE tensors without
    launching the attention kernel.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. The number of heads in Q must be divisible by the number of heads in KV.

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
        q: (batch_size, seqlen, nheads, headdim) or (total_q, nheads, headdim) if cu_seqlens_q
            is provided.
        k: (batch_size, seqlen, nheads_k, headdim) or (total_k, nheads_k, headdim) if cu_seqlens_k
            is provided, or (num_pages, page_size, nheads_k, headdim) if page_table is provided.
        v: (batch_size, seqlen, nheads_k, headdim_v) or (total_k, nheads_k, headdim_v) if
            cu_seqlens_k is provided, or (num_pages, page_size, nheads_k, headdim_v) if page_table
            is provided.
        qv [optional]: (batch_size, seqlen, nheads, headdim_v). Used for cross-attention.
        cu_seqlens_q [optional]: (batch_size + 1,), dtype torch.int32. Cumulative sequence lengths
            of q.
        cu_seqlens_k [optional]: (batch_size + 1,), dtype torch.int32. Cumulative sequence lengths
            of k.
        max_seqlen_q [optional]: Maximum sequence length of q.
        max_seqlen_k [optional]: Maximum sequence length of k.
        min_seqlen_k [optional]: Minimum sequence length of k. (Not supported on NPU)
        seqused_q [optional]: (batch_size,), dtype torch.int32. If given, only this many elements
            of each batch element's queries are used.
        seqused_k [optional]: (batch_size,), dtype torch.int32. If given, only this many elements
            of each batch element's keys are used. Equivalent to cache_seqlens in KV cache scenarios.
        gather_kv_indices [optional]: (Not supported on NPU)
        page_table [optional]: (batch_size, max_num_pages_per_seq), dtype torch.int32. Page table
            for paged KV cache.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim + (headdim_v if qv is not None else 0)).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        learnable_sink [optional]: (num_heads,), dtype bfloat16. One sink logit per query head.
            Added to the softmax denominator. The sink has no value contribution.
        softcap: float. Anything > 0 activates softcapping attention.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
            If num_splits == 0, use a heuristic to automatically determine the number of splits.
        pack_gqa: bool. If True, pack GQA for better performance. (Not supported on NPU)
        deterministic: bool. Whether to use deterministic backward pass.
        score_mod: Optional callable. Custom score modification. (Not supported on NPU)
        score_mod_bwd: Optional callable. Custom score modification for backward. (Not supported on NPU)
        mask_mod: Optional callable. Custom attention mask. (Not supported on NPU)
        block_sparse_tensors: Optional block sparse tensors. (Not supported on NPU)
        aux_tensors: Optional list of tensors. Auxiliary tensors for score_mod. (Not supported on NPU)
        aux_scalars: Optional tuple. Auxiliary scalars for score_mod/mask_mod. (Not supported on NPU)
        return_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim_v) or (total_q, nheads, headdim_v) if varlen.
        softmax_lse [optional, if return_lse=True]: (batch_size, nheads, seqlen) or
            (nheads, total_q) for varlen. The logsumexp of each row of the matrix
            QK^T * scaling (e.g., log of the softmax normalization factor).
    """
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        qv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_k,
        seqused_q,
        seqused_k,
        gather_kv_indices,
        page_table,
        softmax_scale,
        causal,
        window_size,  # -1 means infinite context window
        learnable_sink,
        softcap,  # 0.0 means deactivated
        num_splits,  # Can be tuned for speed
        pack_gqa,  # Can be tuned for speed
        deterministic,
        score_mod,
        score_mod_bwd,
        mask_mod,
        block_sparse_tensors,
        aux_tensors,
        aux_scalars,
        return_lse,
        scheduler_metadata,
        seqlen_k_per_split,
        disable_scheduler_metadata,
    )


@_torch_custom_op_wrapper(
    "flash_attn_npu_4::_flash_attn_combine",
    mutates_args=("out",),
    device_types="npu",
)
def _flash_attn_combine_op(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    out_dtype: Optional[torch.dtype],
    cu_seqlens: Optional[torch.Tensor],
    seqused: Optional[torch.Tensor],
    virtual_batch_idx: Optional[torch.Tensor],
    return_lse: bool,
) -> torch.Tensor:
    result = flash_attn_npu_4.fwd_combine(
        out_partial,
        lse_partial,
        out,
        out_dtype,
        cu_seqlens,
        seqused,
        virtual_batch_idx,
        return_lse,
    )
    return result[1] if return_lse else torch.empty(0, dtype=torch.float32, device=result[0].device)


@_torch_register_fake_wrapper("flash_attn_npu_4::_flash_attn_combine")
def _flash_attn_combine_fake(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    out_dtype: Optional[torch.dtype],
    cu_seqlens: Optional[torch.Tensor],
    seqused: Optional[torch.Tensor],
    virtual_batch_idx: Optional[torch.Tensor],
    return_lse: bool,
) -> torch.Tensor:
    is_dense = out_partial.dim() == 5
    if not return_lse:
        return torch.empty(0, dtype=torch.float32, device=out.device)
    lse_shape = (
        (out_partial.shape[1], out_partial.shape[2], out_partial.shape[3])
        if is_dense
        else (out_partial.shape[1], out_partial.shape[2])
    )
    return torch.empty(lse_shape, dtype=torch.float32, device=out.device)


def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    virtual_batch_idx: Optional[torch.Tensor] = None,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Merge split-KV partial attention outputs (FA4 signature).

    Mirrors ``flash_attn.cute.interface.flash_attn_combine``.

    Args:
        out_partial: fp16/bf16/fp32 partial outputs. Batched (BSND):
            (num_splits, batch, seqlen, nheads, headdim); varlen (TND, detected
            by a 4-D input): (num_splits, total_q, nheads, headdim).
        lse_partial: fp32 partial logsumexp in logical (..., seqlen, heads) order. Batched:
            (num_splits, batch, seqlen, nheads); varlen:
            (num_splits, total_q, nheads).
        out [optional]: preallocated output, written in place. When supplied,
            its dtype takes precedence over out_dtype.
        out_dtype [optional]: fp16 / bf16 / fp32; defaults to
            ``out_partial.dtype`` (fp32).
        cu_seqlens [optional]: accepted and validated for API parity; currently
            does not limit the processed rows.
        seqused [optional]: accepted and validated for API parity; currently
            does not limit the processed rows.
        virtual_batch_idx [optional]: mapping from virtual batch slots to real batches.
        return_lse: whether to return the combined logsumexp.

    Returns:
        (out, lse); ``lse`` is None when ``return_lse=False``.
    """
    if out is None:
        output_dtype = out_dtype or out_partial.dtype
        out_shape = (
            out_partial.shape[1:4] + (out_partial.shape[4],)
            if out_partial.dim() == 5
            else out_partial.shape[1:2] + out_partial.shape[2:4]
        )
        out = torch.empty(out_shape, dtype=output_dtype, device=out_partial.device)
    lse = _flash_attn_combine_op(
        out_partial,
        lse_partial,
        out,
        out_dtype,
        cu_seqlens,
        seqused,
        virtual_batch_idx,
        return_lse,
    )
    return out, lse if return_lse else None
