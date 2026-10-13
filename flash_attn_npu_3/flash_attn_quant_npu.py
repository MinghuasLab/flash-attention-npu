# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Modified by Minghua Shen, 2026.

"""Quantized attention entry. MXFP4 is the current implementation."""

from __future__ import annotations

from enum import IntEnum
from typing import List, Optional, Sequence, Union

import torch

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


class QuantMode(IntEnum):
    MXFP4_OCP = 0
    MXFP4_CX = 1


def _maybe_contiguous(x):
    return x.contiguous() if x is not None and not x.is_contiguous() else x


def _resolve_quant_mode(quant_mode: Union[QuantMode, int]) -> int:
    try:
        return int(QuantMode(quant_mode))
    except (TypeError, ValueError) as exc:
        valid = ", ".join(f"{mode.name} ({int(mode)})" for mode in QuantMode)
        raise ValueError(f"quant_mode must be one of {valid}, got {quant_mode!r}") from exc


def _infer_kv_heads(k: torch.Tensor, layout: str) -> int:
    if layout == "TND":
        return int(k.size(1))
    if layout == "BNSD":
        return int(k.size(1))
    if layout == "BSND":
        return int(k.size(2))
    raise ValueError(f"layout must be TND, BNSD, or BSND, got {layout}")


def _logical_head_dim(q: torch.Tensor) -> int:
    return int(q.size(-1) * 2)


@_torch_custom_op_wrapper(
    "flash_attn_npu_3_950_C::_flash_attn_quant", mutates_args=(), device_types="npu"
)
def _flash_attn_quant(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_dequant_scale: torch.Tensor,
    k_dequant_scale: torch.Tensor,
    v_dequant_scale: torch.Tensor,
    input_layout: str,
    num_key_value_heads: int,
    softmax_scale: float,
    quant_mode: int,
    actual_seq_lengths: Optional[List[int]],
    actual_seq_lengths_kv: Optional[List[int]],
    block_sparse_mask: Optional[torch.Tensor],
    block_shape: Optional[List[int]],
    output_dtype: torch.dtype,
    dst_type_max: float,
    quant_input_dtype: int,
) -> torch.Tensor:
    import flash_attn_npu_3_950

    query, key, value = (_maybe_contiguous(x) for x in (query, key, value))
    q_dequant_scale, k_dequant_scale, v_dequant_scale = (
        _maybe_contiguous(x) for x in (q_dequant_scale, k_dequant_scale, v_dequant_scale)
    )
    block_sparse_mask = _maybe_contiguous(block_sparse_mask)
    return flash_attn_npu_3_950.npu_quant_attention(
        query,
        key,
        value,
        q_dequant_scale,
        k_dequant_scale,
        v_dequant_scale,
        input_layout,
        num_key_value_heads,
        softmax_scale,
        quant_mode,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_sparse_mask,
        block_shape,
        output_dtype,
        dst_type_max,
        quant_input_dtype,
    )


@_torch_register_fake_wrapper("flash_attn_npu_3_950_C::_flash_attn_quant")
def _flash_attn_quant_fake(
    query,
    key,
    value,
    q_dequant_scale,
    k_dequant_scale,
    v_dequant_scale,
    input_layout,
    num_key_value_heads,
    softmax_scale,
    quant_mode,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    block_sparse_mask,
    block_shape,
    output_dtype,
    dst_type_max,
    quant_input_dtype,
):
    out_shape = list(query.shape)
    out_shape[-1] = query.size(-1) * 2
    return query.new_empty(out_shape, dtype=output_dtype)


def flash_attn_quant(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    quant_mode: Union[QuantMode, int],
    block_sparse_mask: Optional[torch.Tensor] = None,
    block_shape: Optional[Sequence[int]] = None,
    *,
    layout: str = "TND",
    num_heads_kv: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    actual_seq_lengths: Optional[Sequence[int]] = None,
    actual_seq_lengths_kv: Optional[Sequence[int]] = None,
    output_dtype: torch.dtype = torch.bfloat16,
    dst_type_max: float = 0.0,
) -> torch.Tensor:
    """Run MXFP4 attention.

    ``quant_mode`` accepts either ``QuantMode`` or its integer value.
    Packed MXFP4 q/k/v use uint8 storage and E8M0 dequant scales.

    The optional ``block_sparse_mask`` and two-element ``block_shape`` enable
    block-sparse attention; omit both for dense attention.
    """
    resolved_quant_mode = _resolve_quant_mode(quant_mode)
    if output_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("output_dtype must be torch.float16 or torch.bfloat16")
    if resolved_quant_mode == int(QuantMode.MXFP4_CX) and not (
        dst_type_max == 0.0 or 6.0 <= dst_type_max <= 12.0
    ):
        raise ValueError(
            "MXFP4_CX quant_mode requires dst_type_max == 0 or dst_type_max in [6, 12], "
            f"got {dst_type_max}"
        )

    layout = layout.upper()
    if layout not in ("TND", "BNSD", "BSND"):
        raise ValueError(f"layout must be TND, BNSD, or BSND, got {layout}")
    kv_heads = num_heads_kv if num_heads_kv is not None else _infer_kv_heads(k, layout)
    scale = softmax_scale if softmax_scale is not None else (_logical_head_dim(q) ** -0.5)

    q_lengths = list(actual_seq_lengths) if actual_seq_lengths is not None else None
    kv_lengths = list(actual_seq_lengths_kv) if actual_seq_lengths_kv is not None else None
    if (block_sparse_mask is None) != (block_shape is None):
        raise ValueError("block_sparse_mask and block_shape must be provided together")
    resolved_block_shape = None if block_shape is None else [int(x) for x in block_shape]
    if resolved_block_shape is not None and len(resolved_block_shape) != 2:
        raise ValueError("block_shape must be [q_block, kv_block]")

    return _flash_attn_quant(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        layout,
        int(kv_heads),
        float(scale),
        resolved_quant_mode,
        q_lengths,
        kv_lengths,
        block_sparse_mask,
        resolved_block_shape,
        output_dtype,
        float(dst_type_max),
        0,
    )


flash_attn_quant.QuantMode = QuantMode
