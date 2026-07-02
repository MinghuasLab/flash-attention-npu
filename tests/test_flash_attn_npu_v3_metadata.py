# Copyright (c) 2026, Minghua Shen.

import pytest
import torch
import torch_npu

from flash_attn_npu_v3 import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)
from tests.test_flash_attn_npu_v3 import ref_flash_attention


RTOL = 1e-2
ATOL = 1e-2
WINDOW_SIZE = (-1, -1)

SMALL_RANGE = (-1.0, 1.0)
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


def _causal_mask(q_seqlen, kv_seqlen, is_causal):
    if not is_causal:
        return None
    return torch.triu(
        torch.ones(q_seqlen, kv_seqlen),
        diagonal=kv_seqlen - q_seqlen + 1,
    ).bool()


def _metadata(
    *,
    batch_size,
    q_seqlen,
    kv_seqlen,
    num_heads,
    kv_heads,
    head_size,
    cache_seqlens,
    data_type,
    cu_seqlens_q=None,
    page_size=None,
    is_causal=False,
    num_splits=0,
):
    return get_scheduler_metadata(
        batch_size=batch_size,
        max_seqlen_q=q_seqlen,
        max_seqlen_k=kv_seqlen,
        num_heads_q=num_heads,
        num_heads_kv=kv_heads,
        headdim=head_size,
        cache_seqlens=cache_seqlens,
        qkv_dtype=data_type,
        cu_seqlens_q=cu_seqlens_q,
        page_size=page_size,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        num_splits=num_splits,
        sm_margin=0,
    )


def _make_paged_cache(batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type):
    max_blocks_per_seq = (kv_seqlen + block_size - 1) // block_size
    num_blocks = batch_size * max_blocks_per_seq
    key_cache = _rand_npu((num_blocks, block_size, kv_heads, head_size), data_type, SMALL_RANGE)
    value_cache = _rand_npu((num_blocks, block_size, kv_heads, head_size), data_type, SMALL_RANGE)
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(batch_size, max_blocks_per_seq).npu()
    return key_cache, value_cache, page_table


def _paged_kv_for_batch(key_cache_cpu, value_cache_cpu, page_table_cpu, batch_idx, kv_seqlen, block_size):
    key_blocks = []
    value_blocks = []
    page_row = page_table_cpu[batch_idx]
    for pos in range(kv_seqlen):
        block_number = int(page_row[pos // block_size])
        block_offset = pos % block_size
        key_blocks.append(key_cache_cpu[block_number, block_offset])
        value_blocks.append(value_cache_cpu[block_number, block_offset])
    return torch.stack(key_blocks, dim=0), torch.stack(value_blocks, dim=0)


def _ref_out_lse(query_cpu, key_cpu, value_cpu, scale, data_type, is_causal):
    output, lse = ref_flash_attention(
        query_cpu,
        key_cpu,
        value_cpu,
        scale,
        _causal_mask(query_cpu.shape[0], key_cpu.shape[0], is_causal),
        data_type,
    )
    return output, lse.reshape(query_cpu.shape[1], query_cpu.shape[0])


def _assert_bsnd_matches_ref(
    output_npu,
    softmax_lse_npu,
    query,
    kv_for_batch,
    *,
    batch_size,
    q_seqlen,
    num_heads,
    head_size,
    scale,
    data_type,
    is_causal,
):
    query_cpu = query.detach().cpu()
    golden_out = torch.empty((batch_size, q_seqlen, num_heads, head_size), dtype=data_type)
    golden_lse = torch.empty((batch_size, num_heads, q_seqlen), dtype=torch.float32)

    for batch_idx in range(batch_size):
        key_cpu, value_cpu = kv_for_batch(batch_idx)
        out, lse = _ref_out_lse(query_cpu[batch_idx], key_cpu, value_cpu, scale, data_type, is_causal)
        golden_out[batch_idx] = out.reshape(q_seqlen, num_heads, head_size)
        golden_lse[batch_idx] = lse

    torch.testing.assert_close(output_npu.cpu(), golden_out, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(softmax_lse_npu.cpu(), golden_lse, rtol=RTOL, atol=ATOL)


def _assert_tnd_matches_ref(
    output_npu,
    softmax_lse_npu,
    query,
    kv_for_batch,
    *,
    q_offsets,
    batch_size,
    num_heads,
    head_size,
    scale,
    data_type,
    is_causal,
):
    query_cpu = query.detach().cpu()
    golden_out = torch.empty((q_offsets[-1], num_heads, head_size), dtype=data_type)
    golden_lse = None
    if softmax_lse_npu is not None:
        golden_lse = torch.empty((num_heads, q_offsets[-1]), dtype=torch.float32)

    for batch_idx in range(batch_size):
        q_start, q_end = q_offsets[batch_idx], q_offsets[batch_idx + 1]
        key_cpu, value_cpu = kv_for_batch(batch_idx)
        out, lse = _ref_out_lse(query_cpu[q_start:q_end], key_cpu, value_cpu, scale, data_type, is_causal)
        golden_out[q_start:q_end] = out.reshape(q_end - q_start, num_heads, head_size)
        if golden_lse is not None:
            golden_lse[:, q_start:q_end] = lse

    torch.testing.assert_close(output_npu.cpu(), golden_out, rtol=RTOL, atol=ATOL)
    if softmax_lse_npu is not None:
        torch.testing.assert_close(softmax_lse_npu.cpu(), golden_lse, rtol=RTOL, atol=ATOL)


FLASH_ATTN_FUNC_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, False),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, True),
    (torch.float16, 7, 1, 1, 512, 512, 128, False),
]


FLASH_ATTN_VARLEN_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal
    (torch.bfloat16, 1, 1, 1, 512, 1024, 128, True),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, False),
    (torch.float16, 7, 5, 1, 512, 512, 128, True),
    (torch.bfloat16, 5, 4, 4, 512, 512, 128, True),
    (torch.float16, 7, 5, 1, 777, 888, 192, False),
    (torch.bfloat16, 1, 1, 1, 7777, 8192, 64, True),
]


KV_CACHE_BSND_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, block_size, is_causal
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 128, False),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 128, True),
    (torch.bfloat16, 1, 1, 1, 2048, 2048, 128, 128, False),
]


KV_CACHE_TND_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, block_size, is_causal
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 128, True),
    (torch.bfloat16, 5, 4, 4, 512, 512, 128, 128, True),
]


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal",
    FLASH_ATTN_FUNC_CASES,
)
def test_flash_attn_func_metadata_bsnd(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal
):
    query = _rand_npu((batch_size, q_seqlen, num_heads, head_size), data_type, WIDE_RANGE)
    key = _rand_npu((batch_size, kv_seqlen, kv_heads, head_size), data_type, WIDE_RANGE)
    value = _rand_npu((batch_size, kv_seqlen, kv_heads, head_size), data_type, WIDE_RANGE)
    cache_seqlens = _int32_npu([kv_seqlen] * batch_size)
    scale = 1.0 / (head_size ** 0.5)

    scheduler_metadata = _metadata(
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        kv_seqlen=kv_seqlen,
        num_heads=num_heads,
        kv_heads=kv_heads,
        head_size=head_size,
        cache_seqlens=cache_seqlens,
        data_type=data_type,
        is_causal=is_causal,
    )
    output_npu, softmax_lse_npu = flash_attn_func(
        query,
        key,
        value,
        softmax_scale=scale,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        num_splits=1,
        return_attn_probs=True,
        scheduler_metadata=scheduler_metadata,
    )

    key_cpu = key.detach().cpu()
    value_cpu = value.detach().cpu()
    _assert_bsnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        lambda batch_idx: (key_cpu[batch_idx], value_cpu[batch_idx]),
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        data_type=data_type,
        is_causal=is_causal,
    )


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal",
    FLASH_ATTN_VARLEN_CASES,
)
def test_flash_attn_varlen_func_metadata_tnd(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal
):
    q_lengths = [q_seqlen] * batch_size
    kv_lengths = [kv_seqlen] * batch_size
    q_offsets = _prefix_sums(q_lengths)
    kv_offsets = _prefix_sums(kv_lengths)
    cu_seqlens_q = _int32_npu(q_offsets)
    cu_seqlens_k = _int32_npu(kv_offsets)
    cache_seqlens = _int32_npu(kv_lengths)

    query = _rand_npu((q_offsets[-1], num_heads, head_size), data_type, WIDE_RANGE)
    key = _rand_npu((kv_offsets[-1], kv_heads, head_size), data_type, WIDE_RANGE)
    value = _rand_npu((kv_offsets[-1], kv_heads, head_size), data_type, WIDE_RANGE)
    scale = 1.0 / (head_size ** 0.5)

    scheduler_metadata = _metadata(
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        kv_seqlen=kv_seqlen,
        num_heads=num_heads,
        kv_heads=kv_heads,
        head_size=head_size,
        cache_seqlens=cache_seqlens,
        data_type=data_type,
        cu_seqlens_q=cu_seqlens_q,
        is_causal=is_causal,
    )
    output_npu = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        q_seqlen,
        kv_seqlen,
        softmax_scale=scale,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        num_splits=1,
        scheduler_metadata=scheduler_metadata,
    )

    key_cpu = key.detach().cpu()
    value_cpu = value.detach().cpu()
    _assert_tnd_matches_ref(
        output_npu,
        None,
        query,
        lambda batch_idx: (
            key_cpu[kv_offsets[batch_idx]:kv_offsets[batch_idx + 1]],
            value_cpu[kv_offsets[batch_idx]:kv_offsets[batch_idx + 1]],
        ),
        q_offsets=q_offsets,
        batch_size=batch_size,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        data_type=data_type,
        is_causal=is_causal,
    )


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, block_size, is_causal",
    KV_CACHE_BSND_CASES,
)
def test_flash_attn_kvcache_metadata_bsnd(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, block_size, is_causal
):
    query = _rand_npu((batch_size, q_seqlen, num_heads, head_size), data_type, SMALL_RANGE)
    key_cache, value_cache, page_table = _make_paged_cache(
        batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type
    )
    cache_seqlens = _int32_npu([kv_seqlen] * batch_size)
    scale = 1.0 / (head_size ** 0.5)

    scheduler_metadata = _metadata(
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        kv_seqlen=kv_seqlen,
        num_heads=num_heads,
        kv_heads=kv_heads,
        head_size=head_size,
        cache_seqlens=cache_seqlens,
        data_type=data_type,
        page_size=block_size,
        is_causal=is_causal,
    )
    output_npu, softmax_lse_npu, *_ = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        max_seqlen_q=q_seqlen,
        softmax_scale=None,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        rotary_interleaved=False,
        scheduler_metadata=scheduler_metadata,
        num_splits=0,
        return_softmax_lse=True,
    )

    key_cache_cpu = key_cache.detach().cpu()
    value_cache_cpu = value_cache.detach().cpu()
    page_table_cpu = page_table.cpu()
    _assert_bsnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        lambda batch_idx: _paged_kv_for_batch(
            key_cache_cpu, value_cache_cpu, page_table_cpu, batch_idx, kv_seqlen, block_size
        ),
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        data_type=data_type,
        is_causal=is_causal,
    )


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, block_size, is_causal",
    KV_CACHE_TND_CASES,
)
def test_flash_attn_kvcache_metadata_tnd(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, block_size, is_causal
):
    q_lengths = [q_seqlen] * batch_size
    q_offsets = _prefix_sums(q_lengths)
    cu_seqlens_q = _int32_npu(q_offsets)
    cache_seqlens = _int32_npu([kv_seqlen] * batch_size)

    query = _rand_npu((q_offsets[-1], num_heads, head_size), data_type, SMALL_RANGE)
    key_cache, value_cache, page_table = _make_paged_cache(
        batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type
    )
    scale = 1.0 / (head_size ** 0.5)

    scheduler_metadata = _metadata(
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        kv_seqlen=kv_seqlen,
        num_heads=num_heads,
        kv_heads=kv_heads,
        head_size=head_size,
        cache_seqlens=cache_seqlens,
        data_type=data_type,
        cu_seqlens_q=cu_seqlens_q,
        page_size=block_size,
        is_causal=is_causal,
    )
    output_npu, softmax_lse_npu, *_ = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=q_seqlen,
        softmax_scale=None,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        rotary_interleaved=False,
        scheduler_metadata=scheduler_metadata,
        num_splits=0,
        return_softmax_lse=True,
    )

    key_cache_cpu = key_cache.detach().cpu()
    value_cache_cpu = value_cache.detach().cpu()
    page_table_cpu = page_table.cpu()
    _assert_tnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        lambda batch_idx: _paged_kv_for_batch(
            key_cache_cpu, value_cache_cpu, page_table_cpu, batch_idx, kv_seqlen, block_size
        ),
        q_offsets=q_offsets,
        batch_size=batch_size,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        data_type=data_type,
        is_causal=is_causal,
    )
