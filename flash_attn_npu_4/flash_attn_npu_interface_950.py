# Copyright (c) 2023, Tri Dao.
# Modified by Minghua Shen, 2026.

from typing import Any, Callable, Optional, Tuple

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
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _window_to_npu(window_size: Optional[int]) -> int:
    return -1 if window_size is None else int(window_size)


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


@_torch_custom_op_wrapper(
    "flash_attn_npu_4_950_C::_flash_attn_backward_op",
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
    dout, q, k, v, out = [_maybe_contiguous(x) for x in (dout, q, k, v, out)]
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
        None,
        None,
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


@_torch_register_fake_wrapper("flash_attn_npu_4_950_C::_flash_attn_backward_op")
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
    dlse: Optional[torch.Tensor] = None,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    block_sparse_tensors: Optional[Any] = None,
    qv: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    pack_gqa: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del dlse

    assert score_mod is None, "flash_attn_npu_v4 950 bwd does not support score_mod"
    assert score_mod_bwd is None, "flash_attn_npu_v4 950 bwd does not support score_mod_bwd"
    assert mask_mod is None, "flash_attn_npu_v4 950 bwd does not support mask_mod"
    assert aux_tensors is None, "flash_attn_npu_v4 950 bwd does not support aux_tensors"
    assert aux_scalars is None, "flash_attn_npu_v4 950 bwd does not support aux_scalars"
    assert block_sparse_tensors is None, (
        "flash_attn_npu_v4 950 bwd does not support block_sparse_tensors"
    )
    assert seqused_q is None, "flash_attn_npu_v4 950 bwd does not support seqused_q"
    assert seqused_k is None, "flash_attn_npu_v4 950 bwd does not support seqused_k"
    assert not pack_gqa, "flash_attn_npu_v4 950 bwd does not support pack_gqa=True"
    assert qv is None, "flash_attn_npu_v4 950 bwd does not support qv"
    assert page_table is None, "flash_attn_npu_v4 950 bwd does not support page_table"
    assert gather_kv_indices is None, (
        "flash_attn_npu_v4 950 bwd does not support gather_kv_indices"
    )
    assert learnable_sink is None, "flash_attn_npu_v4 950 bwd does not support learnable_sink"

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
        window_size=(-1, -1),
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
        assert k.stride(-1) == 1, "k must have contiguous last dimension"
        assert v.stride(-1) == 1, "v must have contiguous last dimension"

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        batch_size = q.shape[0]
        seqused_k = torch.full(
            (batch_size,), k.shape[1], dtype=torch.int32, device=q.device
        )

        out, softmax_lse, out_accum, softmax_lse_accum = _flash_attn_forward(
            q,
            k,
            v,
            qv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            seqused_k,
            gather_kv_indices,
            None,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
            learnable_sink,
            softcap,
            num_splits,
            pack_gqa,
            True,
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

        if return_lse:
            return out, softmax_lse
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dlse = args[0] if ctx.return_lse and len(args) > 0 else None
        if dlse is not None and torch.is_tensor(dlse) and float(dlse.detach().abs().sum()) == 0.0:
            dlse = None
        win_l, win_r = ctx.window_size
        if win_l is not None and win_l < 0:
            win_l = None
        if win_r is not None and win_r < 0:
            win_r = None

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
        window_size=(-1, -1),
        learnable_sink=None,
        softcap=0.0,
        num_splits=0,
        pack_gqa=None,
        deterministic=False,
        score_mod=None,
        score_mod_bwd=None,
        mask_mod=None,
        block_sparse_tensors=None,
        aux_tensors=None,
        aux_scalars=None,
        return_lse=False,
    ):
        assert k.stride(-1) == 1, "k must have contiguous last dimension"
        assert v.stride(-1) == 1, "v must have contiguous last dimension"

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        if seqused_k is None and cu_seqlens_k is not None:
            seqused_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        if seqused_q is None and cu_seqlens_q is not None:
            seqused_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]

        if seqused_k is not None and isinstance(seqused_k, int):
            seqused_k = torch.full(
                (q.shape[0],), seqused_k, dtype=torch.int32, device=k.device
            )
        seqused_q = _maybe_contiguous(seqused_q)
        seqused_k = _maybe_contiguous(seqused_k)

        out, softmax_lse, out_accum, softmax_lse_accum = _flash_attn_forward(
            q,
            k,
            v,
            qv,
            None,
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
            True,
        )

        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k)
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

        if return_lse:
            return out, softmax_lse, out_accum, softmax_lse_accum
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dlse = args[0] if ctx.return_lse and len(args) > 0 else None
        if dlse is not None and torch.is_tensor(dlse) and float(dlse.detach().abs().sum()) == 0.0:
            dlse = None
        win_l, win_r = ctx.window_size
        if win_l is not None and win_l < 0:
            win_l = None
        if win_r is not None and win_r < 0:
            win_r = None

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
            seqused_q=None,
            seqused_k=None,
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
        )


def flash_attn_func(
    q,
    k,
    v,
    qv=None,
    gather_kv_indices=None,
    softmax_scale=None,
    causal: bool = False,
    window_size=(-1, -1),
    learnable_sink=None,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic: bool = False,
    score_mod=None,
    score_mod_bwd=None,
    mask_mod=None,
    aux_tensors=None,
    aux_scalars=None,
    block_sparse_tensors=None,
    block_sparse_tensors_bwd=None,
    return_lse: bool = False,
):
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        qv,
        gather_kv_indices,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
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
    causal: bool = False,
    window_size=(-1, -1),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap=0.0,
    num_splits=0,
    pack_gqa=None,
    deterministic: bool = False,
    score_mod=None,
    score_mod_bwd=None,
    mask_mod=None,
    block_sparse_tensors=None,
    aux_tensors=None,
    aux_scalars=None,
    return_lse: bool = False,
):
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
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        score_mod,
        score_mod_bwd,
        mask_mod,
        block_sparse_tensors,
        aux_tensors,
        aux_scalars,
        return_lse,
    )
