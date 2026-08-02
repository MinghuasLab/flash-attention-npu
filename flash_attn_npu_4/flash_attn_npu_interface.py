# Copyright (c) 2023, Tri Dao.
# Modified by Minghua Shen, 2026

from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn

# isort: off
# We need to import the kernels after importing torch
import flash_attn_npu_4_910  # Registers operators with PyTorch

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


@_torch_custom_op_wrapper("flash_attn_npu_4_910::_flash_attn_forward", mutates_args=(), device_types="npu")
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k = [maybe_contiguous(x) for x in (q, k)]
    v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v
    cu_seqlens_q, cu_seqlens_k = [
        maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k)
    ]
    seqused_q, seqused_k = [maybe_contiguous(x) for x in (seqused_q, seqused_k)]
    page_table = maybe_contiguous(page_table)
    out, softmax_lse = flash_attn_npu_4_910.fwd(
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
    )
    return out, softmax_lse


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
        )
        return (out, softmax_lse) if return_lse else out


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
    return_lse:bool = False,
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

    Note: Does not support backward pass.

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
        deterministic: bool. Whether to use deterministic backward pass. (Not supported on NPU)
        score_mod: Optional callable. Custom score modification. (Not supported on NPU)
        score_mod_bwd: Optional callable. Custom score modification for backward. (Not supported on NPU)
        mask_mod: Optional callable. Custom attention mask. (Not supported on NPU)
        block_sparse_tensors: Optional block sparse tensors. (Not supported on NPU)
        aux_tensors: Optional list of tensors. Auxiliary tensors for score_mod. (Not supported on NPU)
        aux_scalars: Optional tuple. Auxiliary scalars for score_mod/mask_mod. (Not supported on NPU)
        return_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim_v) or (total_q, nheads, headdim_v) if varlen.
        softmax_lse [optional, if return_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
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
    )