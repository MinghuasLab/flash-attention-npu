# Copyright (c) 2026, Minghua Shen.
import math

import torch

"""
  MXFP4 attention golden.

  ref_mxfp4_attention follows the quant_flash_attn CPU golden: logical FP32
  Q/K/V are quantized to MXFP4 and dequantized, scores stay in fp16 with DN
  layout, and probabilities use the blockwise snap-local FP4 scale. Q tiles
  are 128 rows and KV tiles are 4096 columns.
"""

_FP4_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_FP4_POS_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_FP4_POS_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
_MX_BLOCK = 32
_Q_TILE = 128
_KV_TILE = 4096
_FP4_EMAX = 2


def _quantize_fp4(values):
    """Map every element to the nearest FP4 E2M1 magnitude, kept in fp32."""
    table = _FP4_E2M1_VALUES.to(device=values.device)
    index = (values.unsqueeze(-1) - table).abs().argmin(dim=-1)
    return table[index]


def _mx_quantize(tensor, block_size=_MX_BLOCK):
    """Quantize the last dimension with one shared E8M0 scale per block.

    The scale exponent is floor(log2(max_abs)) - 2, clamped to the E8M0 range.
    An all-zero block uses scale 1.
    """
    original_last = tensor.shape[-1]
    pad = (block_size - original_last % block_size) % block_size
    if pad:
        tensor = torch.nn.functional.pad(tensor, (0, pad))
    flat = tensor.reshape(-1, block_size)
    max_abs = flat.abs().amax(dim=-1)
    zero_block = max_abs < 1e-30
    log_max = torch.log2(max_abs.clamp(min=1e-30))
    exponent = (torch.floor(log_max) - _FP4_EMAX).clamp(-127.0, 127.0)
    scales = torch.where(zero_block, torch.ones_like(max_abs), torch.pow(2.0, exponent))
    quantized = _quantize_fp4(flat / scales.unsqueeze(-1)).reshape(tensor.shape)
    if pad:
        quantized = quantized[..., :original_last]
    return quantized, scales


def _mx_dequantize(quantized, scales, block_size=_MX_BLOCK):
    original_last = quantized.shape[-1]
    pad = (block_size - original_last % block_size) % block_size
    if pad:
        quantized = torch.nn.functional.pad(quantized, (0, pad))
    flat = quantized.reshape(-1, block_size) * scales.unsqueeze(-1)
    restored = flat.reshape(quantized.shape)
    if pad:
        restored = restored[..., :original_last]
    return restored


def _mxfp4_qdq(tensor, axis=-1):
    """Quantize and dequantize ``tensor`` along ``axis``."""
    if axis < 0:
        axis += tensor.dim()
    if axis == tensor.dim() - 1:
        quantized, scales = _mx_quantize(tensor)
        return _mx_dequantize(quantized, scales)
    order = list(range(tensor.dim()))
    order[axis], order[-1] = order[-1], order[axis]
    moved = tensor.permute(order).contiguous()
    quantized, scales = _mx_quantize(moved)
    restored = _mx_dequantize(quantized, scales)
    inverse = [0] * tensor.dim()
    for index, source in enumerate(order):
        inverse[source] = index
    return restored.permute(inverse).contiguous()


def _as_lengths(lengths, batch, limit, name):
    if lengths is None:
        return [limit] * batch
    values = [int(value) for value in lengths]
    if len(values) != batch:
        raise ValueError(f"{name} must contain {batch} entries, got {len(values)}")
    for value in values:
        if value < 0 or value > limit:
            raise ValueError(f"{name} entries must lie in [0, {limit}], got {value}")
    return values


def _snap_local_probabilities(scores, running_max, softmax_scale):
    """Quantize one score tile to FP4 with the blockwise snap-local scale.

    The group max is taken on the unscaled scores, then multiplied by the fp16
    softmax scale and snapped onto the log2 grid. The returned running max is
    that integer grid coordinate, in fp32.
    """
    seq_len = scores.shape[0]
    src_dtype = scores.dtype
    device = scores.device
    ln2 = torch.tensor(math.log(2.0), dtype=torch.float16, device=device)
    inv_ln2 = torch.tensor(1.0 / math.log(2.0), dtype=torch.float16, device=device)
    scale_fp16 = torch.tensor(softmax_scale, dtype=torch.float16, device=device)

    pad = (_MX_BLOCK - seq_len % _MX_BLOCK) % _MX_BLOCK
    if pad:
        pad_shape = (pad,) + scores.shape[1:]
        scores = torch.cat(
            [scores, torch.full(pad_shape, float("-inf"), dtype=src_dtype, device=device)],
            dim=0,
        )
    grouped = scores.movedim(0, -1)
    other_shape = grouped.shape[:-1]
    block_count = grouped.shape[-1] // _MX_BLOCK
    grouped = grouped.reshape(*other_shape, block_count, _MX_BLOCK)

    block_max = grouped.max(dim=-1).values
    block_max = torch.where(
        torch.isinf(block_max) & (block_max < 0), torch.zeros_like(block_max), block_max
    )
    block_max = block_max * scale_fp16
    grid = torch.floor(block_max * inv_ln2)
    snapped = (grid - 2) * ln2
    local = torch.exp(grouped * scale_fp16 - snapped.unsqueeze(-1))
    local = local.to(torch.bfloat16).to(src_dtype)

    levels = _FP4_POS_LEVELS.to(device=device, dtype=src_dtype)
    midpoints = _FP4_POS_MIDPOINTS.to(device=device, dtype=src_dtype)
    codes = torch.bucketize(local.contiguous(), midpoints, right=True)
    quantized = levels[codes]
    quantized = quantized.reshape(*other_shape, block_count * _MX_BLOCK)[..., :seq_len]
    quantized = quantized.movedim(-1, 0)

    tile_max = grid.max(dim=-1).values.float()
    merged_max = tile_max if running_max is None else torch.maximum(running_max, tile_max)
    correction = torch.exp2(grid - merged_max.to(src_dtype).unsqueeze(-1) - 2.0)
    correction = correction.repeat_interleave(_MX_BLOCK, dim=-1)[..., :seq_len].movedim(-1, 0)
    return quantized * correction, merged_max


def _attention_tile(query, key, value, softmax_scale):
    """Run one head over KV tiles. Tensors are [rows, dim] / [keys, dim]."""
    query_len, _ = query.shape
    key_len = key.shape[0]
    output = torch.zeros(query_len, value.shape[-1], dtype=torch.float32, device=query.device)
    query_tiles = (query_len + _Q_TILE - 1) // _Q_TILE
    key_tiles = (key_len + _KV_TILE - 1) // _KV_TILE
    for query_index in range(query_tiles):
        query_start = query_index * _Q_TILE
        query_end = min(query_start + _Q_TILE, query_len)
        query_tile = query[query_start:query_end]
        rows = query_end - query_start
        running_max = torch.full((rows,), float("-inf"), dtype=torch.float32, device=query.device)
        running_sum = torch.zeros(rows, dtype=torch.float32, device=query.device)
        accumulator = torch.zeros(rows, value.shape[-1], dtype=torch.float32, device=query.device)
        for key_index in range(key_tiles):
            key_start = key_index * _KV_TILE
            key_end = min(key_start + _KV_TILE, key_len)
            # DN scores: K @ Q^T, accumulated in fp32 and then rounded to fp16.
            scores = (key[key_start:key_end] @ query_tile.t()).to(torch.float16)
            probability, merged_max = _snap_local_probabilities(scores, running_max, softmax_scale)
            probability = probability.float()
            previous = running_max
            safe_previous = torch.where(
                torch.isinf(previous) & (previous < 0), torch.zeros_like(previous), previous
            )
            safe_merged = torch.where(
                torch.isinf(merged_max) & (merged_max < 0), torch.zeros_like(merged_max), merged_max
            )
            first_tile = torch.isinf(previous) & (previous < 0)
            rescale = torch.where(
                first_tile, torch.zeros_like(safe_previous), torch.exp2(safe_previous - safe_merged)
            )
            running_sum = rescale * running_sum + probability.sum(dim=0)
            accumulator = (
                rescale.unsqueeze(-1) * accumulator + probability.t() @ value[key_start:key_end]
            )
            running_max = merged_max
        safe_sum = torch.where(running_sum > 0, running_sum, torch.ones_like(running_sum))
        output[query_start:query_end] = accumulator / safe_sum.unsqueeze(-1)
    return output


def _packed_mxfp4_attention(query, key, value, cu_q, used_q, cu_kv, used_kv, softmax_scale):
    """Packed [tokens, heads, dim] MXFP4 attention. Unused padding stays zero."""
    num_heads = query.shape[1]
    if num_heads % key.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    query = _mxfp4_qdq(query.float(), axis=-1)
    key = _mxfp4_qdq(key.float(), axis=-1)
    value_quant = value.float().clone()
    for offset, length in zip(cu_kv, used_kv):
        if length == 0:
            continue
        value_quant[offset : offset + length] = _mxfp4_qdq(
            value.float()[offset : offset + length], axis=0
        )
    group = num_heads // key.shape[1]
    if group > 1:
        key = key.repeat_interleave(group, dim=1)
        value_quant = value_quant.repeat_interleave(group, dim=1)

    output = torch.zeros(
        query.shape[0], num_heads, value.shape[-1], dtype=torch.float32, device=query.device
    )
    for batch, (query_offset, query_len) in enumerate(zip(cu_q, used_q)):
        key_offset, key_len = cu_kv[batch], used_kv[batch]
        if query_len == 0 or key_len == 0:
            continue
        query_batch = query[query_offset : query_offset + query_len].transpose(0, 1).contiguous()
        key_batch = key[key_offset : key_offset + key_len].transpose(0, 1).contiguous()
        value_batch = value_quant[key_offset : key_offset + key_len].transpose(0, 1).contiguous()
        for head in range(num_heads):
            attended = _attention_tile(
                query_batch[head], key_batch[head], value_batch[head], softmax_scale
            )
            output[query_offset : query_offset + query_len, head] = attended
    return output


def ref_mxfp4_attention(
    query,
    key,
    value,
    scale=None,
    out_dtype=torch.bfloat16,
    q_seqlens=None,
    kv_seqlens=None,
):
    """BSND MXFP4 attention golden.

    ``query``, ``key``, and ``value`` hold the logical floating-point values.
    Q and K are quantized along the head dimension. Each batch's valid KV
    prefix is quantized along the sequence. ``q_seqlens`` and ``kv_seqlens``
    select that prefix inside a padded sequence; omitted lengths use the full
    tensor extent. Padding positions in the returned tensor stay zero.
    """
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be BSND tensors")
    if query.shape[0] != key.shape[0] or key.shape[:2] != value.shape[:2]:
        raise ValueError("key and value must share batch and sequence length")
    if query.shape[-1] != key.shape[-1] or query.shape[-1] != value.shape[-1]:
        raise ValueError("query, key, and value head dimensions must match")
    batch, query_len, num_heads, head_dim = query.shape
    key_len = key.shape[1]
    query_lengths = _as_lengths(q_seqlens, batch, query_len, "q_seqlens")
    key_lengths = _as_lengths(kv_seqlens, batch, key_len, "kv_seqlens")
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    packed_query = query.reshape(batch * query_len, num_heads, head_dim).float()
    packed_key = key.reshape(batch * key_len, key.shape[2], head_dim).float()
    packed_value = value.reshape(batch * key_len, value.shape[2], head_dim).float()
    output = _packed_mxfp4_attention(
        packed_query,
        packed_key,
        packed_value,
        [index * query_len for index in range(batch + 1)],
        query_lengths,
        [index * key_len for index in range(batch + 1)],
        key_lengths,
        float(scale),
    )
    return output.reshape(batch, query_len, num_heads, head_dim).to(out_dtype)
