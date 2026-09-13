# Copyright (c) 2023, Tri Dao.
# Modified by Minghua Shen, 2026.

from typing import Any,Callable, Optional, Tuple, Union

import torch

# isort: off
import flash_attn_npu_4_950
# isort: on

if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:

    def _noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func

        if fn is None:
            return wrap
        return fn

    def _noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func

        if fn is None:
            return wrap
        return fn

    _torch_custom_op_wrapper = _noop_custom_op_wrapper
    _torch_register_fake_wrapper = _noop_register_fake_wrapper


def _maybe_contiguous(x):
    """Make sure the inner-most stride is 1; the kernel asserts it."""
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

@_torch_custom_op_wrapper(
    "flash_attn_npu_4_C::_flash_attn_forward", mutates_args=(), device_types="npu"
)

def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
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
    window_size_left: int = -1,
    window_size_right: int = -1,
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k = (_maybe_contiguous(x) for x in (q, k))
    v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v
    cu_seqlens_q, cu_seqlens_k = (
        _maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k)
    )
    seqused_q, seqused_k = (_maybe_contiguous(x) for x in (seqused_q, seqused_k))
    page_table = _maybe_contiguous(page_table)

    out_t, softmax_lse, out_accum, softmax_lse_accum = flash_attn_npu_4_950.fwd(
        q, k, v,
        qv, out_,
        cu_seqlens_q, cu_seqlens_k,
        seqused_q, seqused_k,
        max_seqlen_q, max_seqlen_k,
        min_seqlen_k, page_table,
        gather_kv_indices,
        softmax_scale,
        causal,
        window_size_left, window_size_right,
        softcap,
        num_splits,
        pack_gqa,
        learnable_sink,
        return_lse,
    )

    if out_accum is None:
        out_accum = torch.tensor([], device=out_t.device)
    if softmax_lse_accum is None:
        softmax_lse_accum = torch.tensor([], device=out_t.device)

    return out_t, softmax_lse, out_accum, softmax_lse_accum


# Scheduler metadata torch.compile support
#
# Keep these constants in sync with:
#   csrc/ascend910/flash_attn_npu_4/fa_metadata_args.h
#
# Real metadata contract:
#
#   no mask:
#       shape = (2376,)
#
#   causal / local mask:
#       shape = (2376 + 2048 * 2048,)
#
#   dtype  = torch.uint8
#   device = NPU
#   stride = (1,)
#
# The custom op prevents TorchDynamo from tracing into the raw pybind
# get_scheduler_metadata() implementation.
# ----------------------------------------------------------------------

_SCHEDULER_METADATA_TILING_BYTES = 2376
_SCHEDULER_METADATA_MASK_BYTES = 2048 * 2048


def _scheduler_metadata_has_mask(
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    max_seqlen_k: int,
) -> bool:
    """Mirror the mask/no-mask decision of DeriveFwdMask in flash_api.cpp."""

    # DeriveFwdMask:
    #
    # A window bound covering the whole K sequence is equivalent to
    # an infinite window and therefore collapses to -1.
    if max_seqlen_k > 0 and window_size_left >= max_seqlen_k:
        window_size_left = -1

    if max_seqlen_k > 0 and window_size_right >= max_seqlen_k:
        window_size_right = -1

    # Causal attention forces the right window to zero.
    if causal:
        window_size_right = 0

    is_causal = (
        window_size_left < 0
        and window_size_right == 0
    )

    is_local = (
        (
            window_size_left >= 0
            or window_size_right >= 0
        )
        and not is_causal
    )

    return is_causal or is_local


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
    qkv_dtype: torch.dtype,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    page_size: Optional[int],
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    num_splits: int,
    pack_gqa: Optional[bool],
    sm_margin: int,
    softmax_scale: Optional[float],
) -> torch.Tensor:
    return flash_attn_npu_4_950.get_scheduler_metadata(
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
        num_splits,
        pack_gqa,
        sm_margin,
        softmax_scale,
    )


@_torch_register_fake_wrapper(
    "flash_attn_npu_4::_get_scheduler_metadata"
)
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
    num_splits: int,
    pack_gqa: Optional[bool],
    sm_margin: int,
    softmax_scale: Optional[float],
) -> torch.Tensor:

    has_mask = _scheduler_metadata_has_mask(
        causal,
        window_size_left,
        window_size_right,
        max_seqlen_k,
    )

    metadata_bytes = (
        _SCHEDULER_METADATA_TILING_BYTES
        + (
            _SCHEDULER_METADATA_MASK_BYTES
            if has_mask
            else 0
        )
    )

    return torch.empty(
        (metadata_bytes,),
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
    softcap=0.0,   # 0.0 means deactivated
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,
    softmax_scale=None,  # defaults to 1 / sqrt(headdim); must match the fwd call
):
    """Precompute scheduler metadata (tiling + attention mask) on the AICPU.

    This avoids the device->host->device round trip in the eager tiling path by
    running the tiling/mask derivation on the NPU. The returned byte tensor is
    passed back to ``flash_attn_func`` / ``flash_attn_varlen_func`` through the
    ``scheduler_metadata`` argument.
    """
    cache_seqlens = maybe_contiguous(cache_seqlens)
    if headdim_v is None:
        headdim_v = headdim
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
        num_splits,
        pack_gqa,
        sm_margin,
        softmax_scale,
    )
    return scheduler_metadata
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
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    _dq, _dk, _dv, softmax_d = flash_attn_npu_4_950.bwd(
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
        None,  # seqused_q
        None,  # seqused_k
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        0,  # sm_margin
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

    softmax_d = torch.empty(
        (batch_size, nheads, seqlen_q), dtype=torch.float32, device=q.device
    )
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
    assert block_sparse_tensors is None, "flash_attn_npu_v4 bwd does not support block_sparse_tensors"
    assert dlse is None, "flash_attn_npu_v4 bwd does not support dlse"
    assert seqused_q is None, "flash_attn_npu_v4 bwd does not support seqused_q"
    assert seqused_k is None, "flash_attn_npu_v4 bwd does not support seqused_k"
    assert not pack_gqa, "flash_attn_npu_v4 bwd does not support pack_gqa=True"
    assert qv is None, "flash_attn_npu_v4 bwd does not support qv"
    assert page_table is None, "flash_attn_npu_v4 bwd does not support page_table"
    assert gather_kv_indices is None, "flash_attn_npu_v4 bwd does not support gather_kv_indices"
    assert learnable_sink is None, "flash_attn_npu_v4 bwd does not support learnable_sink"

    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)

    _flash_attn_backward_op(
        dout,
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
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
        softcap=0.0,   # 0.0 means deactivated
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
        assert k.stride(-1) == 1, "k must have contiguous last dimension"
        assert v.stride(-1) == 1, "v must have contiguous last dimension"
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)

        # Compute scheduler metadata on the AICPU (no D2H/H2D sync), matching
        # the v3 dense path. cache_seqlens holds the per-batch KV lengths.
        batch_size, seqlen_q, num_heads, head_size = q.shape
        seqlen_k = k.shape[1]
        num_heads_k = k.shape[2]
        head_size_v = v.shape[-1]
        cache_seqlens = torch.full(
            (batch_size,), seqlen_k, dtype=torch.int32, device=q.device
        )
        #TODO:add scheduler_metadata after support
        # Ascend950 does not support scheduler metadata currently.
        scheduler_metadata = None

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

        dout, q, k, v, out, head_size_og = _pad_bwd_headdim(
            dout, q, k, v, out, ctx.head_size_og
        )
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
        softcap=0.0, # 0.0 means deactivated
        num_splits=0,    # Can be tuned for speed
        pack_gqa=None,   # Can be tuned for speed
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
        assert k.stride(-1) == 1, "k_cache must have contiguous last dimension"
        assert v.stride(-1) == 1, "v_cache must have contiguous last dimension"
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
        if seqused_k is not None and isinstance(seqused_k, int):
            seqused_k = torch.full(
                (q.shape[0],), seqused_k, dtype=torch.int32, device=k.device
            )
            seqused_k = maybe_contiguous(seqused_k)

        fwd_max_seqlen_k = max_seqlen_k
        if scheduler_metadata is None and not disable_scheduler_metadata and cu_seqlens_q is not None:
            batch_size = cu_seqlens_q.shape[0] - 1
            num_heads = q.shape[1]
            head_size = q.shape[2]
            head_size_v = v.shape[-1]
            num_heads_k = k.shape[1] if k.dim() == 3 else k.shape[2]
            if seqused_k is not None:
                cache_seqlens = seqused_k
            else:
                cache_seqlens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            metadata_max_seqlen_k = max_seqlen_k
            if metadata_max_seqlen_k is None and page_table is not None and k.dim() == 4:
                # Paged KV: normalize the SWA window against the actual max KV
                # seqlen (not the page capacity), matching the host tiling /
                # golden rule "both sides vs actual seqlen_k". Otherwise a
                # finite window smaller than the page capacity but larger than
                # the actual seqlen fails to collapse and yields a wrong mask.
                metadata_max_seqlen_k = int(cache_seqlens.max().item())
                fwd_max_seqlen_k = metadata_max_seqlen_k
            #TODO:add scheduler_metadata after support
            # Ascend950 does not support scheduler metadata currently.
        scheduler_metadata = None

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

        dout, q, k, v, out, head_size_og = _pad_bwd_headdim(
            dout, q, k, v, out, ctx.head_size_og
        )
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
    softcap=0.0,   # 0.0 means deactivated
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
        learnable_sink [optional]: (Not supported on NPU)
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
        softcap,   # 0.0 means deactivated
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




@_torch_register_fake_wrapper("flash_attn_npu_4_C::_flash_attn_forward")
def _flash_attn_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
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
    window_size_left: int = -1,
    window_size_right: int = -1,
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Metadata-only fake for V4 A3 forward. Returns 4-tuple."""
    is_varlen_q = cu_seqlens_q is not None
    head_size_v = v.shape[-1]
    out_dtype = q.dtype

    if is_varlen_q:
        total_q = q.shape[0]
        num_heads = q.shape[1]
        out = torch.empty(
            (total_q, num_heads, head_size_v), dtype=out_dtype, device=q.device
        )
        if return_lse:
            softmax_lse = torch.empty(
                (num_heads, total_q), dtype=torch.float32, device=q.device
            )
        else:
            softmax_lse = torch.empty((0,), dtype=torch.float32, device=q.device)
    else:
        batch_size = q.shape[0]
        seqlen_q = q.shape[1]
        num_heads = q.shape[2]
        out = torch.empty(
            (batch_size, seqlen_q, num_heads, head_size_v),
            dtype=out_dtype,
            device=q.device,
        )
        if return_lse:
            softmax_lse = torch.empty(
                (batch_size, num_heads, seqlen_q),
                dtype=torch.float32,
                device=q.device,
            )
        else:
            softmax_lse = torch.empty((0,), dtype=torch.float32, device=q.device)

    # Real mha_fwd currently always returns empty accum tensors.
    out_accum = torch.empty((0,), dtype=torch.float32, device=q.device)
    softmax_lse_accum = torch.empty((0,), dtype=torch.float32, device=q.device)
    return out, softmax_lse, out_accum, softmax_lse_accum


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
    causal:bool = False,
    window_size=(-1, -1),  # -1 means infinite context window
    learnable_sink: Optional[torch.Tensor] = None,
    softcap=0.0, # 0.0 means deactivated
    num_splits=0,    # Can be tuned for speed
    pack_gqa=None,   # Can be tuned for speed
    deterministic:bool = False, 
    score_mod=None,
    score_mod_bwd=None,
    mask_mod=None,
    block_sparse_tensors=None,
    aux_tensors=None,
    aux_scalars=None,
    return_lse:bool = False,
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
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert k.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v.stride(-1) == 1, "v_cache must have contiguous last dimension"

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if seqused_k is not None and isinstance(seqused_k, int):
        seqused_k = torch.full(
            (q.shape[0],), seqused_k, dtype=torch.int32, device=k.device
        )
        seqused_k = _maybe_contiguous(seqused_k)
    if seqused_k is None and cu_seqlens_k is not None:
        seqused_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        seqused_k = _maybe_contiguous(seqused_k)

    out, softmax_lse, *rest = _flash_attn_forward(
        q,
        k,
        v,
        qv,
        None,  # out_
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
        window_size[0],
        window_size[1],
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        return_lse,
    )
    return (out, softmax_lse, *rest) if return_lse else out
