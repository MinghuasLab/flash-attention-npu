# Copyright (c) 2023, Tri Dao.
# Modified by Minghua Shen, 2026.

from typing import Optional, Tuple, Union

import torch

# isort: off
import flash_attn_npu_3_950
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
    "flash_attn_npu_3_950_C::_flash_attn_forward", mutates_args=(), device_types="npu"
)
def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_new: Optional[torch.Tensor],
    v_new: Optional[torch.Tensor],
    qv: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    cu_seqlens_k_new: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    page_table: Optional[torch.Tensor],
    kv_batch_idx: Optional[torch.Tensor],
    leftpad_k: Optional[torch.Tensor],
    rotary_cos: Optional[torch.Tensor],
    rotary_sin: Optional[torch.Tensor],
    seqlens_rotary: Optional[torch.Tensor],
    q_descale: Optional[torch.Tensor],
    k_descale: Optional[torch.Tensor],
    v_descale: Optional[torch.Tensor],
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    attention_chunk: int,
    softcap: float,
    rotary_interleaved: bool,
    scheduler_metadata: Optional[torch.Tensor],
    num_splits: int,
    pack_gqa: Optional[bool],
    sm_margin: int,
    return_softmax_lse: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, k_new, v_new = (_maybe_contiguous(x) for x in (q, k, k_new, v_new))
    v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v
    cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new = (
        _maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new)
    )
    seqused_q, seqused_k = (_maybe_contiguous(x) for x in (seqused_q, seqused_k))
    page_table, kv_batch_idx, leftpad_k = (
        _maybe_contiguous(x) for x in (page_table, kv_batch_idx, leftpad_k)
    )
    rotary_cos, rotary_sin = (_maybe_contiguous(x) for x in (rotary_cos, rotary_sin))
    seqlens_rotary = _maybe_contiguous(seqlens_rotary)

    out_t, softmax_lse, out_accum, softmax_lse_accum = flash_attn_npu_3_950.fwd(
        q, k, v,
        k_new, v_new, qv,
        out,
        cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new,
        seqused_q, seqused_k,
        max_seqlen_q, max_seqlen_k,
        page_table, kv_batch_idx, leftpad_k,
        rotary_cos, rotary_sin, seqlens_rotary,
        q_descale, k_descale, v_descale,
        softmax_scale,
        causal,
        window_size_left, window_size_right,
        attention_chunk,
        softcap,
        rotary_interleaved,
        scheduler_metadata,
        num_splits,
        pack_gqa,
        sm_margin,
        return_softmax_lse,
    )

    if out_accum is None:
        out_accum = torch.tensor([], device=out_t.device)
    if softmax_lse_accum is None:
        softmax_lse_accum = torch.tensor([], device=out_t.device)

    return out_t, softmax_lse, out_accum, softmax_lse_accum

@_torch_register_fake_wrapper("flash_attn_npu_3_950_C::_flash_attn_forward")
def _flash_attn_forward_fake(
    q, k, v, k_new, v_new, qv,
    out,
    cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new,
    seqused_q, seqused_k,
    max_seqlen_q, max_seqlen_k,
    page_table, kv_batch_idx, leftpad_k,
    rotary_cos, rotary_sin, seqlens_rotary,
    q_descale, k_descale, v_descale,
    softmax_scale, causal,
    window_size_left, window_size_right,
    attention_chunk, softcap, rotary_interleaved,
    scheduler_metadata, num_splits, pack_gqa, sm_margin, return_softmax_lse,
):
    is_varlen_q = cu_seqlens_q is not None
    out_dtype = q.dtype
    head_size_v = v.size(-1)

    if is_varlen_q:
        total_q = q.size(0)
        num_heads = q.size(1)
        out = torch.empty((total_q, num_heads, head_size_v), dtype=out_dtype, device=q.device)
        softmax_lse = (torch.empty((num_heads, total_q), dtype=torch.float32, device=q.device)
                       if return_softmax_lse else torch.empty((0,), dtype=torch.float32, device=q.device))
    else:
        batch_size, seqlen_q, num_heads, _ = q.shape
        out = torch.empty((batch_size, seqlen_q, num_heads, head_size_v), dtype=out_dtype, device=q.device)
        softmax_lse = (torch.empty((batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device)
                       if return_softmax_lse else torch.empty((0,), dtype=torch.float32, device=q.device))

    out_accum = torch.tensor([], device=q.device)
    softmax_lse_accum = torch.tensor([], device=q.device)
    return out, softmax_lse, out_accum, softmax_lse_accum


@_torch_custom_op_wrapper(
    "flash_attn_npu_3_950_C::_flash_attn_backward",
    mutates_args=("dq", "dk", "dv"),
    device_types="npu",
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
    # Placeholder registered for interface parity with Ascend 910; the 950
    # backend has no backward kernel, so the actual implementation is absent.
    raise NotImplementedError(
        "Ascend 950 does not support flash attention backward pass."
    )


def get_scheduler_metadata(
    batch_size,
    max_seqlen_q,
    num_heads_q,
    num_heads_kv,
    headdim,
    cache_seqlens: torch.Tensor,
    qkv_dtype=torch.bfloat16,
    headdim_v=None,
    max_seqlen_k=None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    num_blocks: Optional[int] = None,
    max_num_blocks_per_seq: Optional[int] = None,
    causal=False,
    softmax_scale=None,
    num_splits=0,
    window_size=(-1, -1),
    attention_chunk=0,
    has_softcap=False,
    pack_gqa=None,
    sm_margin=0,  # 910-compatible parameter; unused on Ascend 950
):
    """Precompute AICPU scheduler metadata (tiling + causal mask) on Ascend 950.

    The returned NPU byte tensor can be passed to ``flash_attn_func``,
    ``flash_attn_varlen_func``, or ``flash_attn_with_kvcache`` through their
    ``scheduler_metadata`` argument to avoid per-call host tiling and H2D/D2H
    copies. It depends on shapes, dtype-independent tiling constants, causal
    flag, and the actual per-batch sequence lengths; re-create it whenever those
    change.
    """
    cache_seqlens = _maybe_contiguous(cache_seqlens)
    if cu_seqlens_q is not None:
        cu_seqlens_q = _maybe_contiguous(cu_seqlens_q)
    if cu_seqlens_k is not None:
        cu_seqlens_k = _maybe_contiguous(cu_seqlens_k)
    if headdim_v is None:
        headdim_v = headdim
    if softmax_scale is None:
        softmax_scale = headdim ** (-0.5)
    if qkv_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("qkv_dtype must be torch.float16 or torch.bfloat16")
    if page_size is not None and max_num_blocks_per_seq is None and max_seqlen_k is not None:
        max_num_blocks_per_seq = (max_seqlen_k + page_size - 1) // page_size
    if page_size is not None and num_blocks is None and max_num_blocks_per_seq is not None:
        num_blocks = batch_size * max_num_blocks_per_seq
    if attention_chunk != 0:
        raise ValueError("Ascend 950 does not support attention_chunk")
    if has_softcap:
        raise ValueError("Ascend 950 does not support softcap")
    if pack_gqa is not None and pack_gqa:
        raise ValueError("Ascend 950 does not support pack_gqa")
    scheduler_metadata = flash_attn_npu_3_950.get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        num_heads_q,
        num_heads_kv,
        headdim,
        headdim_v,
        cache_seqlens,
        cu_seqlens_q,
        cu_seqlens_k,
        page_size,
        num_blocks,
        max_num_blocks_per_seq,
        causal,
        softmax_scale,
        num_splits,
        max_seqlen_k if max_seqlen_k is not None else 0,
        window_size[0],
        window_size[1],
    )
    return scheduler_metadata


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
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
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
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
            When cu_seqlens_q is provided (TND), non-paged cache must be 3D:
            (total_tokens, nheads_k, headdim).
            page_block_size can be arbitrary (e.g, 1, 2, 3, 64, etc.).
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim_v) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim_v) if there's a page_table (i.e. paged KV cache)
            When cu_seqlens_q is provided (TND), non-paged cache must be 3D:
            (total_tokens, nheads_k, headdim_v).
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
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        num_batch = cu_seqlens_q.numel() - 1 if cu_seqlens_q is not None else q.shape[0]
        cache_seqlens = torch.full(
            (num_batch,), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = _maybe_contiguous(cache_seqlens)

    if scheduler_metadata is None:
        if cu_seqlens_q is not None:
            if max_seqlen_q is None:
                raise ValueError(
                    "max_seqlen_q must be provided when cu_seqlens_q is provided"
                )
            batch_size = cu_seqlens_q.numel() - 1
            num_heads_q = q.shape[1]
            headdim = q.shape[2]
            headdim_v = v_cache.shape[-1]
            kv_heads = k_cache.shape[1] if k_cache.dim() == 3 else k_cache.shape[2]
            max_q = max_seqlen_q
        else:
            batch_size = q.shape[0]
            num_heads_q = q.shape[2]
            headdim = q.shape[3]
            headdim_v = v_cache.shape[-1]
            kv_heads = k_cache.shape[2]
            max_q = q.shape[1]
        if page_table is not None:
            page_size = k_cache.shape[1]
            num_blocks = k_cache.shape[0]
            max_blocks = page_table.shape[1]
        else:
            page_size = None
            num_blocks = None
            max_blocks = None
        if page_table is not None:
            max_seqlen_k_bound = max_blocks * page_size
        elif cu_seqlens_q is not None:
            max_seqlen_k_bound = k_cache.shape[0]  # TND 3D non-paged: total_tokens bound
        else:
            max_seqlen_k_bound = k_cache.shape[1]
        scheduler_metadata = get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_q,
            num_heads_q=num_heads_q,
            num_heads_kv=kv_heads,
            headdim=headdim,
            headdim_v=headdim_v,
            cache_seqlens=cache_seqlens,
            qkv_dtype=q.dtype,
            cu_seqlens_q=cu_seqlens_q,
            page_size=page_size,
            num_blocks=num_blocks,
            max_num_blocks_per_seq=max_blocks,
            causal=causal,
            window_size=window_size,
            max_seqlen_k=max_seqlen_k_bound,
            softmax_scale=softmax_scale,
            num_splits=num_splits,
        )

    out, softmax_lse, *rest = _flash_attn_forward(
        q,
        k_cache,
        v_cache,
        k,
        v,
        qv,
        None,                # out (let the kernel allocate)
        cu_seqlens_q,
        None,                # cu_seqlens_k
        cu_seqlens_k_new,
        None,                # seqused_q
        cache_seqlens,       # seqused_k — required by the 950 wrapper
        max_seqlen_q,
        None,                # max_seqlen_k
        page_table,
        cache_batch_idx,
        cache_leftpad,
        rotary_cos,
        rotary_sin,
        rotary_seqlens,
        q_descale, k_descale, v_descale,
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
        return_softmax_lse=return_softmax_lse,
    )
    return (out, softmax_lse, *rest) if return_softmax_lse else out


class FlashAttnFunc(torch.autograd.Function):
    """Forward-only autograd wrapper for BSND flash attention on Ascend 950."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        sm_margin=0,
        return_attn_probs=False,
    ):
        assert q.stride(-1) == 1, "q must have contiguous last dimension"
        assert k.stride(-1) == 1, "k must have contiguous last dimension"
        assert v.stride(-1) == 1, "v must have contiguous last dimension"

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        batch_size = q.shape[0]
        seqlen_k = k.shape[1]
        seqused_k = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=q.device)

        scheduler_metadata = get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=q.shape[1],
            max_seqlen_k=seqlen_k,
            num_heads_q=q.shape[2],
            num_heads_kv=k.shape[2],
            headdim=q.shape[3],
            headdim_v=v.shape[3],
            cache_seqlens=seqused_k,
            qkv_dtype=q.dtype,
            causal=causal,
            window_size=window_size,
            softmax_scale=softmax_scale,
            num_splits=num_splits,
        )

        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None,                # k_new
            None,                # v_new
            qv,                  # qv
            None,                # out (let the kernel allocate)
            None,                # cu_seqlens_q (BSND, not varlen)
            None,                # cu_seqlens_k
            None,                # cu_seqlens_k_new
            None,                # seqused_q
            seqused_k,           # seqused_k — required by the 950 backend
            None,                # max_seqlen_q
            None,                # max_seqlen_k
            None,                # page_table
            None,                # kv_batch_idx
            None,                # leftpad_k
            None,                # rotary_cos
            None,                # rotary_sin
            None,                # seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            attention_chunk=attention_chunk,
            softcap=softcap,
            rotary_interleaved=True,
            scheduler_metadata=scheduler_metadata,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
            return_softmax_lse=return_attn_probs,
        )
        return (out, softmax_lse.transpose(-1, -2)) if return_attn_probs else out

    @staticmethod
    def backward(ctx, dout, *args):
        raise NotImplementedError(
            "Ascend 950 does not support backward pass; flash_attn_func is forward-only."
        )


def flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    return_attn_probs=False,
):
    """FlashAttention v3 forward pass for BSND layout (Ascend 950).

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

    Note: Ascend 950 does not support backward pass. This is a forward-only implementation.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). Sliding window local attention bounds;
            (-1, -1) means no window restriction.
        return_attn_probs: bool. Whether to return the attention log-sum-exp values.
            If True, returns (out, softmax_lse).

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, seqlen, nheads).
            The logsumexp of each row of the matrix QK^T * scaling.
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv,
        q_descale, k_descale, v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        sm_margin,
        return_attn_probs,
    )


class FlashAttnVarlenFunc(torch.autograd.Function):
    """Forward-only autograd wrapper for TND (varlen) flash attention on Ascend 950."""

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
        seqused_q=None,
        seqused_k=None,
        softmax_scale=None,
        causal=False,
        qv=None,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        sm_margin=0,
        return_attn_probs=False,
    ):
        assert q.stride(-1) == 1, "q must have contiguous last dimension"
        assert k.stride(-1) == 1, "k must have contiguous last dimension"
        assert v.stride(-1) == 1, "v must have contiguous last dimension"

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        # Derive per-batch sequence lengths from cumulative cu_seqlens if not provided.
        # cu_seqlens format: [0, s1, s1+s2, s1+s2+s3, ...]
        # Per-batch: [s1, s2, s3, ...]
        if seqused_q is None:
            seqused_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        if seqused_k is None:
            seqused_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]

        seqused_q = _maybe_contiguous(seqused_q)
        seqused_k = _maybe_contiguous(seqused_k)

        scheduler_metadata = get_scheduler_metadata(
            batch_size=cu_seqlens_q.numel() - 1,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_heads_q=q.shape[1],
            num_heads_kv=k.shape[1],
            headdim=q.shape[2],
            headdim_v=v.shape[2],
            cache_seqlens=seqused_k,
            qkv_dtype=q.dtype,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            causal=causal,
            window_size=window_size,
            softmax_scale=softmax_scale,
            num_splits=num_splits,
        )

        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None,                # k_new
            None,                # v_new
            qv,                  # qv
            None,                # out (let the kernel allocate)
            cu_seqlens_q,
            cu_seqlens_k,
            None,                # cu_seqlens_k_new
            seqused_q,           # seqused_q
            seqused_k,           # seqused_k — required by the 950 backend
            max_seqlen_q,
            max_seqlen_k,
            None,                # page_table
            None,                # kv_batch_idx
            None,                # leftpad_k
            None,                # rotary_cos
            None,                # rotary_sin
            None,                # seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            attention_chunk=attention_chunk,
            softcap=softcap,
            rotary_interleaved=True,
            scheduler_metadata=scheduler_metadata,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
            return_softmax_lse=return_attn_probs,
        )
        return (out, softmax_lse.transpose(-1, -2)) if return_attn_probs else out

    @staticmethod
    def backward(ctx, dout, *args):
        raise NotImplementedError(
            "Ascend 950 does not support backward pass; flash_attn_varlen_func is forward-only."
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
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    return_attn_probs=False,
):
    """FlashAttention v3 forward pass for TND (varlen) layout (Ascend 950).

    Supports variable-length sequences packed into contiguous tensors.
    Q, K, V are in TND layout: (total_tokens, nheads, headdim).

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.

    Note: Ascend 950 does not support backward pass. This is a forward-only implementation.

    Arguments:
        q: (total_q, nheads, headdim) — TND layout.
        k: (total_k, nheads_k, headdim) — TND layout.
        v: (total_k, nheads_k, headdim) — TND layout.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. Cumulative sequence lengths for Q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. Cumulative sequence lengths for K/V.
        max_seqlen_q: int. Maximum query sequence length.
        max_seqlen_k: int. Maximum key sequence length.
        seqused_q: (batch_size,), dtype torch.int32, optional. Per-batch Q sequence lengths.
            If not provided, derived from cu_seqlens_q.
        seqused_k: (batch_size,), dtype torch.int32, optional. Per-batch KV sequence lengths.
            If not provided, derived from cu_seqlens_k.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask.
        return_attn_probs: bool. Whether to return the attention log-sum-exp values.

    Return:
        out: (total_q, nheads, headdim_v).
        softmax_lse [optional, if return_attn_probs=True]: (total_q, nheads).
            The logsumexp of each row of the matrix QK^T * scaling.
    """
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        seqused_q,
        seqused_k,
        softmax_scale,
        causal,
        qv,
        q_descale, k_descale, v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        sm_margin,
        return_attn_probs,
    )
