# Copyright (c) 2023, Tri Dao.
# Modified by Minghua Shen, 2026.

from typing import Optional, Tuple, Union

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
    "flash_attn_npu_4_950_C::_flash_attn_forward", mutates_args=(), device_types="npu"
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


@_torch_register_fake_wrapper("flash_attn_npu_4_950_C::_flash_attn_forward")
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
