# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Modified by Minghua Shen, 2026.

"""Quantized attention entry for dtypes PyTorch cannot express natively.

`flash_attn_quant` is the public entry. `QuantizationConfig.input_dtype`
selects the kernel. MXFP4 is the current implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import List, Optional, Sequence, Tuple, Union

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


class NpuQuantDType(Enum):
    """Storage dtypes that PyTorch cannot express natively."""

    FLOAT4_E2M1 = "float4_e2m1"
    FLOAT8_E8M0 = "float8_e8m0"


class QuantizationScheme(IntEnum):
    OCP = 2
    CX = 3


@dataclass(frozen=True)
class QuantizationConfig:
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    input_dtype: NpuQuantDType
    scale_dtype: NpuQuantDType
    scheme: QuantizationScheme
    out_dtype: torch.dtype
    target_max: float = 0.0


@dataclass(frozen=True)
class BlockSparseConfig:
    mask: torch.Tensor
    block_shape: Sequence[int]


Feature = Union[QuantizationConfig, BlockSparseConfig]
FeatureSet = Union[Feature, Sequence[Feature]]


def _maybe_contiguous(x):
    return x.contiguous() if x is not None and not x.is_contiguous() else x


def parse_features(features: FeatureSet) -> Tuple[QuantizationConfig, Optional[BlockSparseConfig]]:
    """Validate and split the feature set. Config order does not matter."""
    if isinstance(features, (QuantizationConfig, BlockSparseConfig)):
        items: List[Feature] = [features]
    else:
        items = list(features)

    quant: Optional[QuantizationConfig] = None
    sparse: Optional[BlockSparseConfig] = None
    seen_types = []
    for item in items:
        item_type = type(item)
        if item_type in seen_types:
            raise ValueError(f"duplicate feature config: {item_type.__name__}")
        seen_types.append(item_type)
        if isinstance(item, QuantizationConfig):
            quant = item
        elif isinstance(item, BlockSparseConfig):
            sparse = item
        else:
            raise TypeError(
                f"unsupported feature {item_type.__name__}; "
                "flash_attn_quant currently accepts QuantizationConfig and "
                "optional BlockSparseConfig"
            )

    if quant is None:
        raise ValueError("flash_attn_quant requires a QuantizationConfig")
    return quant, sparse


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
    *,
    features: FeatureSet,
    layout: str = "TND",
    num_heads_kv: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    actual_seq_lengths: Optional[Sequence[int]] = None,
    actual_seq_lengths_kv: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Quantized attention. `QuantizationConfig.input_dtype` selects the kernel.

    Current kernels:
      * FLOAT4_E2M1 input with FLOAT8_E8M0 scales (MXFP4), dense or block-sparse

    Packed quantized q/k/v use uint8 storage; the last dimension is logical
    head dim / 2. The returned tensor restores the logical head dim.
    """
    quant, sparse = parse_features(features)
    if quant.input_dtype != NpuQuantDType.FLOAT4_E2M1 or quant.scale_dtype != NpuQuantDType.FLOAT8_E8M0:
        raise ValueError(
            "current kernel only supports input_dtype=FLOAT4_E2M1 with scale_dtype=FLOAT8_E8M0"
        )
    if quant.scheme not in (QuantizationScheme.OCP, QuantizationScheme.CX):
        raise ValueError("quantization scheme must be QuantizationScheme.OCP or QuantizationScheme.CX")
    if quant.out_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("QuantizationConfig.out_dtype must be torch.float16 or torch.bfloat16")
    if quant.scheme == QuantizationScheme.CX and not (
        quant.target_max == 0.0 or 6.0 <= quant.target_max <= 12.0
    ):
        raise ValueError(
            "CX scheme requires target_max == 0 or target_max in [6, 12], "
            f"got {quant.target_max}"
        )

    layout = layout.upper()
    if layout not in ("TND", "BNSD", "BSND"):
        raise ValueError(f"layout must be TND, BNSD, or BSND, got {layout}")
    kv_heads = num_heads_kv if num_heads_kv is not None else _infer_kv_heads(k, layout)
    scale = softmax_scale if softmax_scale is not None else (_logical_head_dim(q) ** -0.5)

    q_lengths = list(actual_seq_lengths) if actual_seq_lengths is not None else None
    kv_lengths = list(actual_seq_lengths_kv) if actual_seq_lengths_kv is not None else None
    mask = None
    block_shape = None
    if sparse is not None:
        mask = sparse.mask
        block_shape = [int(x) for x in sparse.block_shape]
        if len(block_shape) != 2:
            raise ValueError("BlockSparseConfig.block_shape must be [q_block, kv_block]")

    return _flash_attn_quant(
        q,
        k,
        v,
        quant.q_scale,
        quant.k_scale,
        quant.v_scale,
        layout,
        int(kv_heads),
        float(scale),
        int(quant.scheme),
        q_lengths,
        kv_lengths,
        mask,
        block_shape,
        quant.out_dtype,
        float(quant.target_max),
        0 if quant.input_dtype == NpuQuantDType.FLOAT4_E2M1 else -1,
    )
