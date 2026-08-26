# Copyright (c) 2026, Minghua Shen.

import pytest
import torch
import torch_npu
if "Ascend950" in (torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""):
    pytest.skip("flash_attn_func / flash_attn_varlen_func / get_scheduler_metadata not on Ascend950", allow_module_level=True)

from flash_attn_npu_4 import (
    flash_attn_varlen_func,
    get_scheduler_metadata,
)
from tests.test_flash_attn_npu_v4 import ref_flash_attention
from tests.npu_precision_utils import compare_rule


RTOL = 2e-2
ATOL = 2e-2
WINDOW_SIZE = (-1, -1)

WIDE_RANGE = (-5.0, 5.0)


def _rand_npu(shape, data_type, value_range):
    low, high = value_range
    return (low + (high - low) * torch.rand(shape)).to(data_type).npu()


def _prefix_sums(lengths):
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return offsets


def _int32_npu(values):
    return torch.tensor(values, dtype=torch.int32).npu()


def _metadata(
    *,
    batch_size, q_seqlen, kv_seqlen, num_heads, kv_heads, head_size,
    cache_seqlens, data_type, cu_seqlens_q=None, page_size=None,
    is_causal=False, num_splits=0,
):
    return get_scheduler_metadata(
        batch_size=batch_size, max_seqlen_q=q_seqlen, max_seqlen_k=kv_seqlen,
        num_heads_q=num_heads, num_heads_kv=kv_heads, headdim=head_size,
        cache_seqlens=cache_seqlens, qkv_dtype=data_type,
        cu_seqlens_q=cu_seqlens_q, page_size=page_size,
        causal=is_causal, window_size=WINDOW_SIZE, num_splits=num_splits,
    )


def _causal_mask(q_seqlen, kv_seqlen):
    return torch.triu(
        torch.ones(q_seqlen, kv_seqlen),
        diagonal=kv_seqlen - q_seqlen + 1,
    ).bool()


# ---------------------------------------------------------------------------
# Golden builders  (same pattern as test_flash_attn_npu_v4.py)
# ---------------------------------------------------------------------------

def _build_golden_tnd(
    query, kv_for_batch, *, q_offsets, batch_size, num_heads, head_size,
    scale, data_type, is_causal, softcap=0.0,
):
    query_cpu = query.detach().cpu()
    t_q = q_offsets[-1]
    golden_out = torch.empty((t_q, num_heads, head_size), dtype=data_type)
    golden_lse = torch.empty((num_heads, t_q), dtype=torch.float32)
    for bi in range(batch_size):
        s, e = q_offsets[bi], q_offsets[bi + 1]
        k_cpu, v_cpu = kv_for_batch(bi)
        mask = _causal_mask(e - s, k_cpu.shape[0]) if is_causal else None
        out, lse = ref_flash_attention(
            query_cpu[s:e].unsqueeze(0), k_cpu.unsqueeze(0), v_cpu.unsqueeze(0),
            scale, mask, data_type, softcap=softcap,
        )
        golden_out[s:e] = out.reshape(e - s, num_heads, head_size)
        golden_lse[:, s:e] = lse.reshape(num_heads, e - s)
    return golden_out, golden_lse


# ---------------------------------------------------------------------------
# Test cases (same format as test_flash_attn_npu_v4.py)
#   (data_type, B, H, Hkv, Q, KV, D, page_sz, causal, layout, cache_mode)
# ---------------------------------------------------------------------------

VARLEN_CASES = [
    (torch.bfloat16, 1, 1, 1, 512,  1024, 128, 128, True,  "TND", 0),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, 128, False, "TND", 0),
    (torch.float16,   7, 5, 1, 512,  512,  128, 128, True,  "TND", 0),
    (torch.bfloat16, 5, 4, 4, 512,  512,  128, 128, True,  "TND", 0),
    (torch.float16,   7, 5, 1, 777,  888,  192, 128, False, "TND", 0),
    (torch.bfloat16, 1, 1, 1, 7777, 8192, 64,  128, True,  "TND", 0),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, "
    "head_size, block_size, is_causal, layout, cache_mode",
    VARLEN_CASES,
)
def test_flash_attn_varlen_func_metadata_tnd(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen,
    head_size, block_size, is_causal, layout, cache_mode,
):
    """flash_attn_varlen_func + scheduler_metadata — golden vs CANN comparison."""
    q_lengths = [q_seqlen] * batch_size
    kv_lengths = [kv_seqlen] * batch_size
    q_offsets = _prefix_sums(q_lengths)
    kv_offsets = _prefix_sums(kv_lengths)

    query = _rand_npu((q_offsets[-1], num_heads, head_size), data_type, WIDE_RANGE)
    key = _rand_npu((kv_offsets[-1], kv_heads, head_size), data_type, WIDE_RANGE)
    value = _rand_npu((kv_offsets[-1], kv_heads, head_size), data_type, WIDE_RANGE)
    scale = 1.0 / (head_size ** 0.5)

    cu_seqlens_q = _int32_npu(q_offsets)
    cu_seqlens_k = _int32_npu(kv_offsets)
    cache_seqlens = _int32_npu(kv_lengths)

    meta = _metadata(
        batch_size=batch_size, q_seqlen=q_seqlen, kv_seqlen=kv_seqlen,
        num_heads=num_heads, kv_heads=kv_heads, head_size=head_size,
        cache_seqlens=cache_seqlens, data_type=data_type,
        cu_seqlens_q=cu_seqlens_q, is_causal=is_causal,
    )

    out_npu = flash_attn_varlen_func(
        query, key, value, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=q_seqlen, max_seqlen_k=kv_seqlen, softmax_scale=scale,
        causal=is_causal, window_size=WINDOW_SIZE, num_splits=1,
        scheduler_metadata=meta,
    )

    key_cpu = key.detach().cpu()
    value_cpu = value.detach().cpu()
    golden_out, _ = _build_golden_tnd(
        query,
        lambda bi: (key_cpu[kv_offsets[bi]:kv_offsets[bi + 1]],
                    value_cpu[kv_offsets[bi]:kv_offsets[bi + 1]]),
        q_offsets=q_offsets, batch_size=batch_size, num_heads=num_heads,
        head_size=head_size, scale=scale, data_type=data_type, is_causal=is_causal,
    )

    torch.testing.assert_close(out_npu.cpu(), golden_out, rtol=RTOL, atol=ATOL)
    if data_type == torch.bfloat16:
        _, ok = compare_rule(golden_out.cpu().float(), out_npu.cpu().float(),
                             ratios=(0.05, 0.05, 0.05, 0.05))
    else:
        _, ok = compare_rule(golden_out.cpu().float(), out_npu.cpu().float())
    assert ok, "Golden vs CANN (metadata varlen) check FAILED"


def test_flash_attn_varlen_func_metadata_disabled():
    """flash_attn_varlen_func with disable_scheduler_metadata=True should produce same result."""
    data_type = torch.bfloat16
    batch_size = 2
    num_heads = 4
    kv_heads = 4
    q_seqlen = 1024
    kv_seqlen = 1024
    head_size = 128
    is_causal = True

    q_lengths = [q_seqlen] * batch_size
    kv_lengths = [kv_seqlen] * batch_size
    q_offsets = _prefix_sums(q_lengths)
    kv_offsets = _prefix_sums(kv_lengths)

    query = _rand_npu((q_offsets[-1], num_heads, head_size), data_type, WIDE_RANGE)
    key = _rand_npu((kv_offsets[-1], kv_heads, head_size), data_type, WIDE_RANGE)
    value = _rand_npu((kv_offsets[-1], kv_heads, head_size), data_type, WIDE_RANGE)
    scale = 1.0 / (head_size ** 0.5)

    cu_seqlens_q = _int32_npu(q_offsets)
    cu_seqlens_k = _int32_npu(kv_offsets)
    cache_seqlens = _int32_npu(kv_lengths)

    meta = _metadata(
        batch_size=batch_size, q_seqlen=q_seqlen, kv_seqlen=kv_seqlen,
        num_heads=num_heads, kv_heads=kv_heads, head_size=head_size,
        cache_seqlens=cache_seqlens, data_type=data_type,
        cu_seqlens_q=cu_seqlens_q, is_causal=is_causal,
    )

    # With metadata (normal)
    out_with_meta = flash_attn_varlen_func(
        query, key, value, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=q_seqlen, max_seqlen_k=kv_seqlen, softmax_scale=scale,
        causal=is_causal, window_size=WINDOW_SIZE, num_splits=1,
        scheduler_metadata=meta,
    )

    # With disable_scheduler_metadata=True (should ignore meta)
    out_disabled = flash_attn_varlen_func(
        query, key, value, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=q_seqlen, max_seqlen_k=kv_seqlen, softmax_scale=scale,
        causal=is_causal, window_size=WINDOW_SIZE, num_splits=1,
        scheduler_metadata=meta,
        disable_scheduler_metadata=True,
    )

    torch.testing.assert_close(out_with_meta.cpu(), out_disabled.cpu(), rtol=RTOL, atol=ATOL)
