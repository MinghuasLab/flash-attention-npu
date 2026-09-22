# Copyright (c) 2023, Tri Dao.
# Modified by Minghua Shen, 2026.

import math
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
    """Make tensors fully contiguous for kernels that use linear GM offsets."""
    return x.contiguous() if x is not None and not x.is_contiguous() else x

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
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
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
    dout, q, k, v, out = (
        _maybe_contiguous(x) for x in (dout, q, k, v, out)
    )
    _, _, _, softmax_d = flash_attn_npu_3_950.bwd(
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
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        sm_margin,
    )
    return softmax_d


@_torch_register_fake_wrapper("flash_attn_npu_3_950_C::_flash_attn_backward")
def _flash_attn_backward_fake(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dq=None,
    dk=None,
    dv=None,
    softmax_scale=None,
    is_causal=False,
    window_size_left=-1,
    window_size_right=-1,
    softcap=0.0,
    deterministic=False,
    sm_margin=0,
):
    del (
        dout, k, v, out, softmax_lse, cu_seqlens_k, seqused_q, seqused_k,
        max_seqlen_k, softmax_scale, is_causal, window_size_left,
        window_size_right, softcap, deterministic, sm_margin,
    )
    if cu_seqlens_q is None:
        batch_size, seqlen_q, num_heads = q.shape[:3]
        return torch.empty(
            (batch_size, num_heads, seqlen_q),
            dtype=torch.float32,
            device=q.device,
        )
    return torch.empty(
        (q.shape[1], q.shape[0]),
        dtype=torch.float32,
        device=q.device,
    )



@_torch_custom_op_wrapper(
    "flash_attn_npu_3_950::_get_scheduler_metadata",
    mutates_args=(),
    device_types="npu",
)
def _get_scheduler_metadata_op(
    batch_size: int,
    max_seqlen_q: int,
    num_heads_q: int,
    num_heads_kv: int,
    headdim: int,
    headdim_v: int,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    page_size: Optional[int],
    num_blocks: Optional[int],
    max_num_blocks_per_seq: Optional[int],
    causal: bool,
    softmax_scale: float,
    num_splits: int,
    max_seqlen_k: int,
    window_left: int,
    window_right: int,
) -> torch.Tensor:
    return flash_attn_npu_3_950.get_scheduler_metadata(
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
        max_seqlen_k,
        window_left,
        window_right,
    )


@_torch_register_fake_wrapper(
    "flash_attn_npu_3_950::_get_scheduler_metadata"
)
def _get_scheduler_metadata_fake(
    batch_size: int,
    max_seqlen_q: int,
    num_heads_q: int,
    num_heads_kv: int,
    headdim: int,
    headdim_v: int,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    page_size: Optional[int],
    num_blocks: Optional[int],
    max_num_blocks_per_seq: Optional[int],
    causal: bool,
    softmax_scale: float,
    num_splits: int,
    max_seqlen_k: int,
    window_left: int,
    window_right: int,
) -> torch.Tensor:
    ctx = torch.library.get_ctx()
    metadata_size = ctx.new_dynamic_size()

    return torch.empty(
        (metadata_size,),
        dtype=torch.uint8,
        device=cache_seqlens.device,
    )

def _is_fd_candidate(num_splits, paged, varlen_q, max_seqlen_q):
    # Preserve the pre-metadata-FD host routing. The runtime tiler still decides
    # whether these candidates actually enable FD or fall back to normal FA.
    return num_splits > 1 or (
        num_splits == 0 and paged and varlen_q
        and max_seqlen_q is not None and max_seqlen_q <= 16
    )


def _metadata_window_size(causal, window_size, max_seqlen_k):
    # Mirror DeriveFwdMask in fa_metadata_args.h without reading device lengths.
    left, right = window_size
    if max_seqlen_k > 0:
        if left >= max_seqlen_k:
            left = -1
        if right >= max_seqlen_k:
            right = -1
    if causal:
        right = 0
    is_local = (left >= 0 or right >= 0) and not (left < 0 and right == 0)
    if is_local:
        left = max_seqlen_k if left < 0 else left
        right = max_seqlen_k if right < 0 else right
    return (left, right)


def _validate_fd_scheduler_metadata(
    scheduler_metadata, *, q, k_cache, v_cache, page_table, cu_seqlens_q,
    max_seqlen_q, causal, window_size, softmax_scale, num_splits,
):
    params = getattr(scheduler_metadata, "_fa_scheduler_params", None)
    call_is_fd_candidate = _is_fd_candidate(
        num_splits, page_table is not None, cu_seqlens_q is not None, max_seqlen_q
    )
    # A changed num_splits/layout must not bypass validation of an FD schedule.
    # Normal-FA metadata retains its existing acceptance rules, including raw
    # tensors without a Python fingerprint.
    if not call_is_fd_candidate and not (params and params.get("fd_candidate")):
        return
    if params is None:
        raise RuntimeError(
            "FD scheduler_metadata has no creation-argument fingerprint; pass "
            "the unchanged tensor returned by get_scheduler_metadata"
        )
    if scheduler_metadata.device != q.device:
        raise ValueError("FD scheduler_metadata must be on the same device as q")

    varlen_q = cu_seqlens_q is not None
    paged = page_table is not None
    kv_bound = (
        k_cache.shape[1] * page_table.shape[1] if paged
        else k_cache.shape[0] if varlen_q else k_cache.shape[1]
    )
    expected = {
        "backend": "ascend950",
        "version": 1,
        "device": str(q.device),
        "batch_size": cu_seqlens_q.numel() - 1 if varlen_q else q.shape[0],
        "max_seqlen_q": max_seqlen_q if varlen_q else q.shape[1],
        "num_heads_q": q.shape[-2],
        "num_heads_kv": k_cache.shape[-2],
        "headdim": q.shape[-1],
        "headdim_v": v_cache.shape[-1],
        "qkv_dtype": q.dtype,
        "varlen_q": varlen_q,
        "varlen_kv": False,
        "page_size": k_cache.shape[1] if paged else None,
        "num_blocks": k_cache.shape[0] if paged else None,
        "max_num_blocks_per_seq": page_table.shape[1] if paged else None,
        "causal": bool(causal),
        "window_size": tuple(window_size),
        "normalized_window_size": _metadata_window_size(causal, window_size, kv_bound),
        "softmax_scale": float(softmax_scale),
        "num_splits": int(num_splits),
    }
    mismatches = []
    for key, call_value in expected.items():
        meta_value = params.get(key, "<missing>")
        same = (
            math.isclose(meta_value, call_value, rel_tol=1e-6)
            if key == "softmax_scale" and isinstance(meta_value, float)
            else meta_value == call_value
        )
        if not same:
            mismatches.append(f"{key}: metadata={meta_value!r} vs call={call_value!r}")
    if mismatches:
        raise ValueError(
            "FD scheduler_metadata arguments do not match this call: " + "; ".join(mismatches)
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
    change. For FD candidates, pass this tensor unchanged (do not clone/copy it):
    the Python wrapper checks its creation arguments against the consuming call.
    This check does not detect changes to the contents of sequence-length tensors.
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
    scheduler_metadata = _get_scheduler_metadata_op(
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
    # Attach after the custom-op boundary, which does not preserve Python attrs.
    # Record normal-FA arguments too, so reusing that metadata for an FD call
    # can be checked without imposing new checks on ordinary non-FD calls.
    scheduler_metadata._fa_scheduler_params = {
        "backend": "ascend950",
        "version": 1,
        "device": str(cache_seqlens.device),
        "fd_candidate": _is_fd_candidate(
            num_splits, page_size is not None, cu_seqlens_q is not None, max_seqlen_q
        ),
        "batch_size": int(batch_size),
        "max_seqlen_q": int(max_seqlen_q),
        "num_heads_q": int(num_heads_q),
        "num_heads_kv": int(num_heads_kv),
        "headdim": int(headdim),
        "headdim_v": int(headdim_v),
        "qkv_dtype": qkv_dtype,
        "varlen_q": cu_seqlens_q is not None,
        "varlen_kv": cu_seqlens_k is not None,
        "page_size": page_size,
        "num_blocks": num_blocks if page_size is not None else None,
        "max_num_blocks_per_seq": max_num_blocks_per_seq if page_size is not None else None,
        "causal": bool(causal),
        "window_size": tuple(window_size),
        "normalized_window_size": _metadata_window_size(
            causal, window_size, max_seqlen_k if max_seqlen_k is not None else 0
        ),
        "softmax_scale": float(softmax_scale),
        "num_splits": int(num_splits),
    }
    return scheduler_metadata


def _training_forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    scheduler_metadata,
):
    if cu_seqlens_q is None:
        seqused_k = torch.full(
            (q.shape[0],),
            k.shape[1],
            dtype=torch.int32,
            device=k.device,
        )
    else:
        seqused_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]

    return _flash_attn_forward(
        q,
        k,
        v,
        None, None, None, None,  # k_new, v_new, qv, out
        cu_seqlens_q,
        cu_seqlens_k,
        None,                    # cu_seqlens_k_new
        None,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        None, None, None,        # page_table, kv_batch_idx, leftpad_k
        None, None, None,        # rotary
        None, None, None,        # descales
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        0,
        0.0,
        True,
        scheduler_metadata,
        1,
        None,
        0,
        True,                    # return_softmax_lse (needed by the backward)
    )


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
        return_softmax,
        scheduler_metadata,
    ):
        if any(x is not None for x in (qv, q_descale, k_descale, v_descale)):
            raise NotImplementedError("Ascend950 v3 training scaffold only supports q/k/v inputs")
        if attention_chunk != 0 or softcap != 0.0:
            raise NotImplementedError("Ascend950 v3 training scaffold does not support attention_chunk or softcap")
        if num_splits not in (0, 1) or pack_gqa not in (None, False) or sm_margin != 0:
            raise NotImplementedError("Ascend950 v3 training scaffold does not support split/pack/sm tuning")
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        if scheduler_metadata is None:
            meta_cache_seqlens = torch.full(
                (q.shape[0],), k.shape[1], dtype=torch.int32, device=q.device
            )
            scheduler_metadata = get_scheduler_metadata(
                batch_size=q.shape[0],
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                num_heads_q=q.shape[2],
                num_heads_kv=k.shape[2],
                headdim=q.shape[3],
                headdim_v=v.shape[3],
                cache_seqlens=meta_cache_seqlens,
                qkv_dtype=q.dtype,
                causal=causal,
                window_size=window_size,
                softmax_scale=softmax_scale,
                num_splits=num_splits,
            )

        out, softmax_lse, _, _ = _training_forward(
            q, k, v, None, None, None, None, softmax_scale, causal, window_size, scheduler_metadata
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = tuple(window_size)
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        return (out, softmax_lse.transpose(-1, -2)) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *unused_grads):
        if ctx.window_size != (-1, -1):
            raise NotImplementedError("Ascend950 v3 backward does not support sliding-window attention")
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout, q, k, v, out, softmax_lse,
            None, None, None, None, None, None,
            dq, dk, dv,
            ctx.softmax_scale,
            ctx.causal,
            -1, -1, 0.0,
            ctx.deterministic,
            ctx.sm_margin,
        )
        return dq, dk, dv, *((None,) * 15)


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
        return_softmax,
        scheduler_metadata,
    ):
        if any(x is not None for x in (
            seqused_q, seqused_k, qv, q_descale, k_descale, v_descale,
        )):
            raise NotImplementedError("Ascend950 v3 varlen training scaffold does not support optional tensor inputs")
        if attention_chunk != 0 or softcap != 0.0:
            raise NotImplementedError("Ascend950 v3 training scaffold does not support attention_chunk or softcap")
        if num_splits not in (0, 1) or pack_gqa not in (None, False) or sm_margin != 0:
            raise NotImplementedError("Ascend950 v3 training scaffold does not support split/pack/sm tuning")
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        if scheduler_metadata is None:
            meta_seqused_k = _maybe_contiguous(cu_seqlens_k[1:] - cu_seqlens_k[:-1])
            scheduler_metadata = get_scheduler_metadata(
                batch_size=cu_seqlens_q.numel() - 1,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                num_heads_q=q.shape[1],
                num_heads_kv=k.shape[1],
                headdim=q.shape[2],
                headdim_v=v.shape[2],
                cache_seqlens=meta_seqused_k,
                qkv_dtype=q.dtype,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                causal=causal,
                window_size=window_size,
                softmax_scale=softmax_scale,
                num_splits=num_splits,
            )

        out, softmax_lse, _, _ = _training_forward(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            softmax_scale, causal, window_size,
            scheduler_metadata,
        )
        ctx.save_for_backward(
            q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = tuple(window_size)
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        return (out, softmax_lse.transpose(-1, -2)) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *unused_grads):
        if ctx.window_size != (-1, -1):
            raise NotImplementedError("Ascend950 v3 backward does not support sliding-window attention")
        q, k, v, out, softmax_lse, cu_q, cu_k = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout, q, k, v, out, softmax_lse,
            cu_q, cu_k, None, None,
            ctx.max_seqlen_q, ctx.max_seqlen_k,
            dq, dk, dv,
            ctx.softmax_scale,
            ctx.causal,
            -1, -1, 0.0,
            ctx.deterministic,
            ctx.sm_margin,
        )
        return dq, dk, dv, *((None,) * 21)


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
    scheduler_metadata=None,
):
    return FlashAttnFunc.apply(
        q, k, v, softmax_scale, causal, qv,
        q_descale, k_descale, v_descale,
        window_size, attention_chunk, softcap,
        num_splits, pack_gqa, deterministic, sm_margin,
        return_attn_probs, scheduler_metadata,
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
    scheduler_metadata=None,
):
    return FlashAttnVarlenFunc.apply(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        seqused_q, seqused_k,
        max_seqlen_q, max_seqlen_k,
        softmax_scale, causal, qv,
        q_descale, k_descale, v_descale,
        window_size, attention_chunk, softcap,
        num_splits, pack_gqa, deterministic, sm_margin,
        return_attn_probs, scheduler_metadata,
    )


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
        scheduler_metadata: Optional metadata returned by get_scheduler_metadata.
            For FD candidates (explicit splits, or auto-split paged TND with
            max_seqlen_q <= 16), None selects host tiling; an unchanged metadata
            tensor selects the precomputed AICPU schedule. Other calls retain
            automatic metadata generation. FD creation arguments must match this
            call; regenerate metadata whenever the actual sequence lengths change.
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

    use_host_tiling = _is_fd_candidate(
        num_splits, page_table is not None, cu_seqlens_q is not None, max_seqlen_q
    )
    if scheduler_metadata is not None:
        _validate_fd_scheduler_metadata(
            scheduler_metadata, q=q, k_cache=k_cache, v_cache=v_cache,
            page_table=page_table, cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q, causal=causal, window_size=window_size,
            softmax_scale=softmax_scale, num_splits=num_splits,
        )
    elif not use_host_tiling:
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
        if cache_seqlens is not None:
            max_seqlen_k_bound = int(cache_seqlens.max().item())
        elif page_table is not None:
            max_seqlen_k_bound = max_blocks * page_size
        elif cu_seqlens_q is not None:
            max_seqlen_k_bound = k_cache.shape[0]  # TND 3D non-paged fallback
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
