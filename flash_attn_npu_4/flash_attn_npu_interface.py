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
else:
    def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    _torch_custom_op_wrapper = noop_custom_op_wrapper


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def round_multiple(x, m):
    return (x + m - 1) // m * m


_HEADDIM_BWD_ALIGN = 64


def _pad_bwd_headdim(dout, q, k, v, out, head_size_og):
    """Pad headdim to a multiple of 64 for the FAG bwd kernel.
    """
    if dout.size(-1) != head_size_og:
        raise ValueError(
            f"dout headdim ({dout.size(-1)}) must equal original q/k/v "
            f"headdim ({head_size_og})"
        )
    qkv_out_headdims = [t.size(-1) for t in (q, k, v, out)]
    if len(set(qkv_out_headdims)) != 1:
        raise ValueError(
            f"q/k/v/out must share the same headdim, got {qkv_out_headdims} "
            "(unequal headdim is not supported)"
        )
    ctx_headdim = qkv_out_headdims[0]
    if ctx_headdim <= 0 or ctx_headdim > 256:
        raise ValueError(
            f"headdim must be in (0, 256], got {ctx_headdim} "
        )
    target = round_multiple(ctx_headdim, _HEADDIM_BWD_ALIGN)

    def _pad(t):
        cur = t.size(-1)
        if cur == target:
            return t
        if cur > target:
            raise ValueError(f"headdim {cur} > pad target {target}")
        return torch.nn.functional.pad(t, [0, target - cur])

    return _pad(dout), _pad(q), _pad(k), _pad(v), _pad(out), head_size_og


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k = [maybe_contiguous(x) for x in (q, k)]
    v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v
    cu_seqlens_q, cu_seqlens_k = [
        maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k)
    ]
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
    )
    return out, softmax_lse


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
    scheduler_metadata = flash_attn_npu_4.get_scheduler_metadata(
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
        scheduler_metadata = get_scheduler_metadata(
            batch_size,
            seqlen_q,
            seqlen_k,
            num_heads,
            num_heads_k,
            head_size,
            cache_seqlens,
            qkv_dtype=q.dtype,
            headdim_v=head_size_v,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=0,
            softmax_scale=softmax_scale,
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
                # Paged KV: KV length is bounded by the page-table capacity
                # (max_num_blocks_per_seq * page_block_size), matching the
                # kvSeqlenBound used by mha_fwd when consuming metadata.
                metadata_max_seqlen_k = page_table.shape[-1] * k.shape[1]
            scheduler_metadata = get_scheduler_metadata(
                batch_size,
                max_seqlen_q,
                metadata_max_seqlen_k,
                num_heads,
                num_heads_k,
                head_size,
                cache_seqlens,
                qkv_dtype=q.dtype,
                headdim_v=head_size_v,
                cu_seqlens_q=cu_seqlens_q,
                page_size=k.shape[1] if (page_table is not None and k.dim() == 4) else None,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                num_splits=num_splits,
                pack_gqa=pack_gqa,
                sm_margin=0,
                softmax_scale=softmax_scale,
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
            max_seqlen_k,
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
        return (out, softmax_lse) if return_lse else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = (
            ctx.saved_tensors
        )
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
        learnable_sink [optional]: (num_heads,), dtype bfloat16. Learnable sink token.
            (Not supported on NPU)
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
        softcap, # 0.0 means deactivated
        num_splits,    # Can be tuned for speed
        pack_gqa,   # Can be tuned for speed
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