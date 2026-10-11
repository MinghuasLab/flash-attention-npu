# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Modified by Minghua Shen, 2026.

import math

import pytest
import torch

from flash_attn_npu_3.flash_attn_quant_npu import (
    QuantMode,
    flash_attn_quant,
)
from tests.common.mxfp4_attention_ref import _mx_quantize, ref_mxfp4_attention

_E8M0_BIAS = 127
_MX_GROUP = 32


def _npu_available():
    try:
        import torch_npu
        from flash_attn_npu_3 import is_ascend950

        return torch.npu.is_available() and is_ascend950()
    except Exception:
        return False


def _pack_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Pack logical FP4 magnitudes into uint8. Low nibble is the even element."""
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    mag = values.abs()
    codes = torch.argmin((mag.unsqueeze(-1) - lut).abs(), dim=-1).to(torch.uint8)
    codes = torch.where(values < 0, codes | 0x8, codes)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def _unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    low = packed & 0x0F
    high = packed >> 4
    nibbles = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    sign = (nibbles & 0x8) != 0
    mag = lut[(nibbles & 0x7).long()]
    return torch.where(sign, -mag, mag)


def _e8m0(scales: torch.Tensor) -> torch.Tensor:
    exponent = torch.log2(scales).round().to(torch.int64)
    return (exponent + _E8M0_BIAS).clamp(0, 254).to(torch.uint8)


def _dequant_last(packed: torch.Tensor, e8m0_scale: torch.Tensor) -> torch.Tensor:
    """Undo Q/K packing. ``e8m0_scale`` is ``[..., D/64, 2]``."""
    values = _unpack_e2m1(packed)
    flat_scale = e8m0_scale.reshape(*e8m0_scale.shape[:-2], -1).to(torch.float32)
    scale = torch.pow(2.0, flat_scale - _E8M0_BIAS).repeat_interleave(_MX_GROUP, dim=-1)
    return values * scale


def _quantize_last(tensor: torch.Tensor):
    quantized, scales = _mx_quantize(tensor)
    groups = (tensor.shape[-1] + _MX_GROUP - 1) // _MX_GROUP
    return quantized, scales.reshape(*tensor.shape[:-1], groups)


def _pack_qk(bnsd: torch.Tensor):
    """Quantize BNSD along the head dim and pack to ``[B, N, S, D/64, 2]`` scales."""
    quantized, scales = _quantize_last(bnsd)
    grouped = scales.reshape(*scales.shape[:-1], scales.shape[-1] // 2, 2)
    return _pack_e2m1(quantized), _e8m0(grouped)


def _pack_v(bnsd: torch.Tensor, kv_lengths):
    """Quantize each batch's valid KV prefix along the sequence.

    Storage stays BNSD. Padding tokens are zero and unused scale groups are
    the E8M0 code for scale 1. The paired layout is ``[B, N, ceil(S/64), D, 2]``,
    where the last dimension is two consecutive 32-token groups.
    """
    batch, heads, seq_len, head_dim = bnsd.shape
    groups = (seq_len + _MX_GROUP - 1) // _MX_GROUP
    packed = torch.zeros(batch, heads, seq_len, head_dim // 2, dtype=torch.uint8)
    scale_groups = torch.full((batch, heads, groups, head_dim), _E8M0_BIAS, dtype=torch.uint8)
    for index, length in enumerate(kv_lengths):
        length = int(length)
        if length == 0:
            continue
        prefix = bnsd[index:index + 1, :, :length, :]
        moved = prefix.movedim(2, -1).contiguous()
        quantized, scales = _quantize_last(moved)
        restored = quantized.movedim(-1, 2).contiguous()
        scales = _e8m0(scales.movedim(-1, 2).contiguous())
        packed[index, :, :length] = _pack_e2m1(restored)[0]
        scale_groups[index, :, :scales.shape[2]] = scales[0]
    if groups % 2:
        pad = torch.full((batch, heads, 1, head_dim), _E8M0_BIAS, dtype=torch.uint8)
        scale_groups = torch.cat((scale_groups, pad), dim=2)
    pair_count = scale_groups.shape[2] // 2
    paired = scale_groups.reshape(batch, heads, pair_count, 2, head_dim).permute(0, 1, 2, 4, 3).contiguous()
    return packed, paired


def _lengths_or_full(lengths, batch, limit):
    if lengths is None:
        return [limit] * batch
    return [int(value) for value in lengths]


def _draw_bnsd(batch, heads, seq, head_dim, generator, low, high, scale_trap):
    if scale_trap:
        if heads < 2:
            raise ValueError("scale-stride trap needs at least two heads")
        values = torch.empty(batch, heads, seq, head_dim)
        for head in range(heads):
            lo, hi = (0.60, 0.90) if head % 2 == 0 else (1.20, 1.80)
            magnitude = lo + (hi - lo) * torch.rand(batch, seq, head_dim, generator=generator)
            sign = torch.where(torch.rand(batch, seq, head_dim, generator=generator) > 0.5, 1.0, -1.0)
            values[:, head] = magnitude * sign
        return values
    if low is None:
        return torch.randn(batch, heads, seq, head_dim, generator=generator)
    return low + (high - low) * torch.rand(batch, heads, seq, head_dim, generator=generator)


def _logical_bnsd(
    batch, heads, kv_heads, seq_q, seq_k, head_dim, q_lengths, kv_lengths, generator, low=None, high=None, scale_trap=False
):
    query = _draw_bnsd(batch, heads, seq_q, head_dim, generator, low, high, scale_trap)
    key = _draw_bnsd(batch, kv_heads, seq_k, head_dim, generator, low, high, scale_trap)
    value = _draw_bnsd(batch, kv_heads, seq_k, head_dim, generator, low, high, scale_trap)
    for index, length in enumerate(q_lengths):
        query[index, :, length:] = 0
    for index, length in enumerate(kv_lengths):
        key[index, :, length:] = 0
    return query, key, value


def _pack_bnsd(query, key, value, kv_lengths):
    query_packed, query_scale = _pack_qk(query)
    key_packed, key_scale = _pack_qk(key)
    value_packed, value_scale = _pack_v(value, kv_lengths)
    return query_packed, key_packed, value_packed, query_scale, key_scale, value_scale


def _to_layout(layout, packed, scales, q_lengths, kv_lengths):
    """Convert BNSD packed Q/K/V and scales into the kernel layout."""
    query, key, value = packed
    query_scale, key_scale, value_scale = scales
    if layout == "BNSD":
        return query, key, value, query_scale, key_scale, value_scale
    if layout == "BSND":
        return (
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            query_scale.transpose(1, 2).contiguous(),
            key_scale.transpose(1, 2).contiguous(),
            value_scale.transpose(1, 2).contiguous(),
        )
    if layout != "TND":
        raise ValueError(f"unsupported layout {layout}")

    def _tokens(tensor, lengths):
        pieces = []
        for index, length in enumerate(lengths):
            piece = tensor[index, :, :length]
            order = (1, 0) + tuple(range(2, piece.dim()))
            pieces.append(piece.permute(order).contiguous())
        return torch.cat(pieces, dim=0)

    def _v_scale_tokens():
        pieces = []
        for index, length in enumerate(kv_lengths):
            groups = (int(length) + 63) // 64
            piece = value_scale[index, :, :groups].permute(1, 0, 2, 3).contiguous()
            pieces.append(piece)
        return torch.cat(pieces, dim=0)

    return (
        _tokens(query, q_lengths),
        _tokens(key, kv_lengths),
        _tokens(value, kv_lengths),
        _tokens(query_scale, q_lengths),
        _tokens(key_scale, kv_lengths),
        _v_scale_tokens(),
    )


def _golden_like(actual, golden_bsnd, layout, q_lengths):
    """Slice the BSND golden into the kernel output layout, valid tokens only."""
    if layout == "TND":
        pieces = [golden_bsnd[index, :length] for index, length in enumerate(q_lengths)]
        return torch.cat(pieces, dim=0)
    if layout == "BNSD":
        golden = golden_bsnd.transpose(1, 2).contiguous()
        seq_axis = 2
    else:
        golden = golden_bsnd
        seq_axis = 1
    if all(int(length) == actual.shape[seq_axis] for length in q_lengths):
        return golden
    actual_pieces = []
    golden_pieces = []
    for index, length in enumerate(q_lengths):
        if layout == "BNSD":
            actual_pieces.append(actual[index, :, :length])
            golden_pieces.append(golden[index, :, :length])
        else:
            actual_pieces.append(actual[index, :length])
            golden_pieces.append(golden[index, :length])
    return torch.cat([piece.reshape(-1) for piece in actual_pieces]), torch.cat(
        [piece.reshape(-1) for piece in golden_pieces]
    )


def _assert_mxfp4_close(actual, expect):
    """Match quant_flash_attn: 99.5% of elements within tolerance, outlier relative error < 10."""
    if actual.dtype == torch.bfloat16 or expect.dtype == torch.bfloat16:
        rtol, atol = 0.0078125, 1e-4
    else:
        rtol, atol = 0.005, 2.5e-5
    actual = actual.detach().to(torch.float32).flatten().cpu()
    expect = expect.detach().to(torch.float32).flatten().cpu()
    assert actual.shape == expect.shape, f"shape mismatch actual={tuple(actual.shape)} expect={tuple(expect.shape)}"
    assert torch.isfinite(actual).all(), "actual contains non-finite values"
    assert torch.isfinite(expect).all(), "golden contains non-finite values"
    if actual.numel() == 0:
        return
    close = torch.isclose(actual, expect, rtol=rtol, atol=atol, equal_nan=True)
    passed = int(close.sum().item())
    total = actual.numel()
    fulfill = 100.0 * passed / total
    if close.all():
        return
    diff = (actual - expect).abs()
    magnitude = torch.maximum(actual.abs(), expect.abs())
    floor = (1.0 / (1 << 14)) / 0.005
    relative = diff / (torch.maximum(magnitude, torch.tensor(floor)) + 1e-12)
    max_error = float(relative[~close].max().item())
    assert fulfill >= 99.5 and max_error < 10.0, (
        f"mxfp4 mismatch: {passed}/{total} ({fulfill:.4f}%) within rtol={rtol} atol={atol}, "
        f"max relative error {max_error}"
    )


def _run_quant(
    query,
    key,
    value,
    query_scale,
    key_scale,
    value_scale,
    *,
    layout,
    kv_heads,
    out_dtype,
    softmax_scale,
    q_lengths,
    kv_lengths,
    sparse=None,
    quant_mode=QuantMode.MXFP4_OCP,
    target_max=0.0,
    pass_lengths=True,
):
    device = "npu"
    block_sparse_mask = None
    block_shape = None
    if sparse is not None:
        mask, block_shape = sparse
        block_sparse_mask = mask.to(device)
    kwargs = {}
    if pass_lengths and (layout == "TND" or q_lengths is not None):
        kwargs["actual_seq_lengths"] = q_lengths
        kwargs["actual_seq_lengths_kv"] = kv_lengths
    return flash_attn_quant(
        query.to(device),
        key.to(device),
        value.to(device),
        query_scale.to(device),
        key_scale.to(device),
        value_scale.to(device),
        quant_mode,
        block_sparse_mask,
        block_shape,
        layout=layout,
        num_heads_kv=kv_heads,
        softmax_scale=softmax_scale,
        output_dtype=out_dtype,
        dst_type_max=target_max,
        **kwargs,
    )


def _dense_case(
    layout,
    batch,
    heads,
    kv_heads,
    seq_q,
    seq_k,
    head_dim,
    out_dtype,
    q_lengths=None,
    kv_lengths=None,
    softmax_scale=None,
    seed=0,
    low=None,
    high=None,
    scale_trap=False,
    pass_lengths=True,
):
    q_lengths = _lengths_or_full(q_lengths, batch, seq_q)
    kv_lengths = _lengths_or_full(kv_lengths, batch, seq_k)
    generator = torch.Generator().manual_seed(seed)
    query, key, value = _logical_bnsd(
        batch,
        heads,
        kv_heads,
        seq_q,
        seq_k,
        head_dim,
        q_lengths,
        kv_lengths,
        generator,
        low=low,
        high=high,
        scale_trap=scale_trap,
    )
    packed = _pack_bnsd(query, key, value, kv_lengths)
    tensors = _to_layout(layout, packed[:3], packed[3:], q_lengths, kv_lengths)
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    actual = _run_quant(
        *tensors,
        layout=layout,
        kv_heads=kv_heads,
        out_dtype=out_dtype,
        softmax_scale=softmax_scale,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        pass_lengths=pass_lengths,
    )
    golden = ref_mxfp4_attention(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        scale=softmax_scale,
        out_dtype=out_dtype,
        q_seqlens=q_lengths,
        kv_seqlens=kv_lengths,
    )
    expected = _golden_like(actual.cpu(), golden, layout, q_lengths)
    if isinstance(expected, tuple):
        _assert_mxfp4_close(expected[0], expected[1])
    else:
        _assert_mxfp4_close(actual, expected)
    return actual


@pytest.mark.parametrize(
    ("layout", "batch", "heads", "kv_heads", "seq_q", "seq_k", "head_dim", "out_dtype", "q_lengths", "kv_lengths"),
    [
        ("BNSD", 1, 4, 2, 128, 128, 64, torch.bfloat16, None, None),
        ("BSND", 1, 4, 2, 128, 128, 64, torch.float16, None, None),
        ("BNSD", 1, 2, 2, 64, 96, 128, torch.bfloat16, None, None),
        ("BSND", 2, 2, 1, 32, 64, 64, torch.float16, None, None),
        ("TND", 1, 4, 2, 128, 128, 64, torch.float16, None, None),
        ("BNSD", 2, 2, 2, 128, 128, 64, torch.bfloat16, [64, 96], [80, 128]),
        ("BSND", 2, 2, 2, 96, 128, 64, torch.float16, [48, 96], [40, 128]),
    ],
)
@pytest.mark.skipif(not _npu_available(), reason="requires Ascend 950")
def test_dense_quantized_attention_matches_golden(
    layout, batch, heads, kv_heads, seq_q, seq_k, head_dim, out_dtype, q_lengths, kv_lengths
):
    _dense_case(layout, batch, heads, kv_heads, seq_q, seq_k, head_dim, out_dtype, q_lengths, kv_lengths)


@pytest.mark.skipif(not _npu_available(), reason="requires Ascend 950")
def test_dense_tnd_varlen_matches_golden():
    _dense_case(
        "TND",
        batch=2,
        heads=2,
        kv_heads=1,
        seq_q=80,
        seq_k=96,
        head_dim=64,
        out_dtype=torch.bfloat16,
        q_lengths=[48, 80],
        kv_lengths=[40, 96],
        seed=1,
    )


@pytest.mark.skipif(not _npu_available(), reason="requires Ascend 950")
def test_dense_softmax_scale_matches_golden():
    _dense_case(
        "BNSD",
        batch=1,
        heads=2,
        kv_heads=2,
        seq_q=64,
        seq_k=64,
        head_dim=64,
        out_dtype=torch.float16,
        softmax_scale=0.5,
        seed=2,
    )


@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.skipif(not _npu_available(), reason="requires Ascend 950")
def test_block_sparse_all_ones_matches_dense_golden(block):
    batch, heads, seq, head_dim = 1, 2, 128, 64
    q_lengths = [seq]
    kv_lengths = [seq]
    generator = torch.Generator().manual_seed(3)
    query, key, value = _logical_bnsd(batch, heads, heads, seq, seq, head_dim, q_lengths, kv_lengths, generator)
    packed = _pack_bnsd(query, key, value, kv_lengths)
    blocks = seq // block
    mask = torch.ones(batch, heads, blocks, blocks, dtype=torch.uint8)
    actual = _run_quant(
        *packed,
        layout="BNSD",
        kv_heads=heads,
        out_dtype=torch.bfloat16,
        softmax_scale=head_dim ** -0.5,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        sparse=(mask, (block, block)),
    )
    golden = ref_mxfp4_attention(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        out_dtype=torch.bfloat16,
    )
    _assert_mxfp4_close(actual, golden.transpose(1, 2).contiguous())


@pytest.mark.skipif(not _npu_available(), reason="requires Ascend 950")
def test_block_sparse_partial_mask_changes_output():
    batch, heads, seq, head_dim, block = 1, 2, 128, 64, 64
    q_lengths = [seq]
    kv_lengths = [seq]
    generator = torch.Generator().manual_seed(4)
    query, key, value = _logical_bnsd(batch, heads, heads, seq, seq, head_dim, q_lengths, kv_lengths, generator)
    packed = _pack_bnsd(query, key, value, kv_lengths)
    common = dict(
        layout="BNSD",
        kv_heads=heads,
        out_dtype=torch.bfloat16,
        softmax_scale=head_dim ** -0.5,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
    )
    dense = _run_quant(*packed, **common)
    # Each query block must keep at least one KV block. A row of all zeros
    # makes the kernel subtract one from an empty selection count.
    blocks = seq // block
    pattern = torch.tril(torch.ones(blocks, blocks, dtype=torch.uint8))
    mask = pattern.view(1, 1, blocks, blocks).expand(batch, heads, blocks, blocks).contiguous()
    sparse = _run_quant(*packed, sparse=(mask, (block, block)), **common)
    assert sparse.shape == (batch, heads, seq, head_dim)
    assert torch.isfinite(sparse.float()).all()
    assert not torch.allclose(sparse.float(), dense.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("target_max", [0.0, 8.0])
@pytest.mark.skipif(not _npu_available(), reason="requires Ascend 950")
def test_cx_dense_attention_is_finite(out_dtype, target_max):
    batch, heads, seq, head_dim = 1, 2, 64, 64
    lengths = [seq]
    generator = torch.Generator().manual_seed(5)
    query, key, value = _logical_bnsd(batch, heads, heads, seq, seq, head_dim, lengths, lengths, generator)
    packed = _pack_bnsd(query, key, value, lengths)
    actual = _run_quant(
        *packed,
        layout="BNSD",
        kv_heads=heads,
        out_dtype=out_dtype,
        softmax_scale=head_dim ** -0.5,
        q_lengths=lengths,
        kv_lengths=lengths,
        quant_mode=QuantMode.MXFP4_CX,
        target_max=target_max,
    )
    assert actual.shape == (batch, heads, seq, head_dim)
    assert actual.dtype == out_dtype
    assert torch.isfinite(actual.float()).all()


@pytest.mark.parametrize(
    (
        "layout",
        "batch",
        "heads",
        "kv_heads",
        "seq_q",
        "seq_k",
        "head_dim",
        "out_dtype",
        "q_lengths",
        "kv_lengths",
        "pass_lengths",
        "scale_trap",
        "low",
        "high",
        "seed",
    ),
    [
        ("BNSD", 1, 2, 2, 128, 256, 128, torch.bfloat16, None, None, True, False, None, None, 11),
        ("BSND", 1, 2, 2, 15, 256, 128, torch.float16, None, None, True, False, None, None, 11),
        ("BNSD", 1, 2, 2, 1, 256, 128, torch.bfloat16, None, None, True, False, None, None, 11),
        ("TND", 1, 2, 2, 1, 512, 64, torch.float16, None, None, True, False, None, None, 11),
        ("BNSD", 1, 2, 2, 127, 159, 128, torch.bfloat16, None, None, True, False, None, None, 11),
        ("BSND", 1, 2, 2, 15, 255, 64, torch.float16, None, None, True, False, None, None, 11),
        ("BNSD", 1, 8, 2, 128, 256, 128, torch.bfloat16, None, None, True, False, None, None, 11),
        ("BSND", 1, 8, 1, 64, 128, 64, torch.float16, None, None, True, False, None, None, 11),
        ("BNSD", 1, 2, 2, 128, 512, 128, torch.bfloat16, None, None, True, False, None, None, 11),
        ("TND", 1, 4, 2, 64, 1024, 64, torch.bfloat16, None, None, True, False, None, None, 11),
        ("BNSD", 2, 2, 2, 128, 256, 128, torch.float16, None, None, True, False, None, None, 11),
        ("TND", 2, 2, 2, 256, 256, 128, torch.bfloat16, [128, 256], [256, 128], True, False, None, None, 12),
        ("TND", 2, 2, 2, 129, 255, 128, torch.bfloat16, [127, 129], [255, 193], True, False, None, None, 12),
        ("TND", 2, 2, 2, 128, 512, 128, torch.bfloat16, [1, 128], [64, 512], True, False, None, None, 12),
        ("BNSD", 1, 2, 2, 128, 256, 128, torch.bfloat16, None, None, False, False, None, None, 13),
        ("BSND", 1, 2, 2, 128, 256, 128, torch.bfloat16, None, None, False, False, None, None, 13),
        ("BSND", 1, 2, 2, 128, 256, 128, torch.bfloat16, None, None, True, True, None, None, 14),
        ("TND", 1, 2, 2, 128, 256, 128, torch.bfloat16, None, None, True, True, None, None, 14),
        ("BNSD", 1, 2, 2, 64, 128, 128, torch.bfloat16, None, None, True, False, -10.0, 10.0, 15),
        ("BSND", 1, 2, 2, 128, 128, 64, torch.float16, None, None, True, False, 0.0, 0.0, 16),
    ],
)
@pytest.mark.skipif(not _npu_available(), reason="requires Ascend 950")
def test_bsa_dense_shapes_match_golden(
    layout,
    batch,
    heads,
    kv_heads,
    seq_q,
    seq_k,
    head_dim,
    out_dtype,
    q_lengths,
    kv_lengths,
    pass_lengths,
    scale_trap,
    low,
    high,
    seed,
):
    _dense_case(
        layout,
        batch,
        heads,
        kv_heads,
        seq_q,
        seq_k,
        head_dim,
        out_dtype,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        pass_lengths=pass_lengths,
        scale_trap=scale_trap,
        low=low,
        high=high,
        seed=seed,
    )


@pytest.mark.parametrize(
    ("layout", "block_q", "block_kv", "seq_q", "seq_k", "mask_kind", "seed"),
    [
        ("BNSD", 64, 128, 128, 256, "all_ones", 17),
        ("BNSD", 128, 256, 128, 512, "all_ones", 17),
        ("BSND", 128, 128, 128, 512, "gap", 18),
    ],
)
@pytest.mark.skipif(not _npu_available(), reason="requires Ascend 950")
def test_bsa_block_sparse_mask(layout, block_q, block_kv, seq_q, seq_k, mask_kind, seed):
    batch, heads, head_dim = 1, 2, 64
    q_lengths = [seq_q]
    kv_lengths = [seq_k]
    generator = torch.Generator().manual_seed(seed)
    query, key, value = _logical_bnsd(batch, heads, heads, seq_q, seq_k, head_dim, q_lengths, kv_lengths, generator)
    packed = _pack_bnsd(query, key, value, kv_lengths)
    tensors = _to_layout(layout, packed[:3], packed[3:], q_lengths, kv_lengths)
    q_blocks, kv_blocks = seq_q // block_q, seq_k // block_kv
    if mask_kind == "all_ones":
        mask = torch.ones(batch, heads, q_blocks, kv_blocks, dtype=torch.uint8)
    else:
        mask = torch.zeros(batch, heads, q_blocks, kv_blocks, dtype=torch.uint8)
        mask[:, :, :, 0] = 1
        mask[:, :, :, -1] = 1
    common = dict(
        layout=layout,
        kv_heads=heads,
        out_dtype=torch.bfloat16,
        softmax_scale=head_dim ** -0.5,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
    )
    actual = _run_quant(*tensors, sparse=(mask, (block_q, block_kv)), **common)
    assert torch.isfinite(actual.float()).all()
    if mask_kind == "all_ones":
        golden = ref_mxfp4_attention(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            out_dtype=torch.bfloat16,
        )
        expected = _golden_like(actual.cpu(), golden, layout, q_lengths)
        _assert_mxfp4_close(actual, expected)
        return
    dense = _run_quant(*tensors, **common)
    assert not torch.allclose(actual.float(), dense.float(), rtol=1e-2, atol=1e-2)
