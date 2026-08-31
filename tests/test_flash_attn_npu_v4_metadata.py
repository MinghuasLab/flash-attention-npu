# Copyright (c) 2026, Minghua Shen.

import pytest
import torch
import torch_npu

_device_name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
if "Ascend910" not in _device_name:
    pytest.skip("flash_attn_func / flash_attn_varlen_func / get_scheduler_metadata only on Ascend910", allow_module_level=True)

from flash_attn_npu_4 import (
    flash_attn_func,
    flash_attn_varlen_func,
    get_scheduler_metadata,
)
from tests.common.attention_ref import ref_flash_attention_pair
from tests.common.compare import assert_fa_close
from tests.common.test_utils import gather_paged_kv_batch, make_random_tensor


WINDOW_SIZE = (-1, -1)

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


def _band_mask(q_seqlen, kv_seqlen, window_size_left, window_size_right):
    pre_token = kv_seqlen - q_seqlen - window_size_left
    next_token = kv_seqlen - q_seqlen + window_size_right
    rows = torch.arange(q_seqlen).unsqueeze(1)
    cols = torch.arange(kv_seqlen).unsqueeze(0)
    diag = cols - rows
    return (diag < pre_token) | (diag > next_token)


def _attn_mask(q_seqlen, kv_seqlen, is_causal, window_size):
    # Mirrors the window normalization in the C++ host/metadata paths.
    window_left, window_right = window_size
    if kv_seqlen > 0 and window_left >= kv_seqlen - 1:
        window_left = -1
    if q_seqlen > 0 and window_right >= q_seqlen - 1:
        window_right = -1
    if is_causal:
        window_right = 0
    causal_golden = window_left < 0 and window_right == 0
    local_golden = (window_left >= 0 or window_right >= 0) and not causal_golden
    if causal_golden:
        return _causal_mask(q_seqlen, kv_seqlen, True)
    if local_golden:
        return _band_mask(q_seqlen, kv_seqlen, window_left, window_right)
    return None


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
    window_size=WINDOW_SIZE,
    softcap=0.0,
    softmax_scale=None,
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
        window_size=window_size,
        softcap=softcap,
        softmax_scale=softmax_scale,
        num_splits=num_splits,
        sm_margin=0,
    )


def _make_paged_cache(batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type):
    max_blocks_per_seq = (kv_seqlen + block_size - 1) // block_size
    num_blocks = batch_size * max_blocks_per_seq
    key_cache = make_random_tensor((num_blocks, block_size, kv_heads, head_size), data_type, device="npu")
    value_cache = make_random_tensor((num_blocks, block_size, kv_heads, head_size), data_type, device="npu")
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(batch_size, max_blocks_per_seq).npu()
    return key_cache, value_cache, page_table


def _flash_attn_kvcache(
    query,
    key_cache,
    value_cache,
    cache_seqlens,
    page_table,
    scheduler_metadata,
    *,
    cu_seqlens_q=None,
    max_seqlen_q,
    softmax_scale=None,
    causal=False,
    window_size=WINDOW_SIZE,
    softcap=0.0,
):
    return flash_attn_varlen_func(
        query,
        key_cache,
        value_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        page_table=page_table,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        scheduler_metadata=scheduler_metadata,
        num_splits=0,
        return_lse=True,
    )


def _assert_bsnd_matches_ref(
    output_npu,
    softmax_lse_npu,
    query,
    kv_batched,
    *,
    batch_size,
    q_seqlen,
    num_heads,
    head_size,
    scale,
    data_type,
    is_causal,
    window_size=WINDOW_SIZE,
    softcap=0.0,
):
    query_cpu = query.detach().cpu()
    key_cpu, value_cpu = kv_batched
    golden_out_ref, golden_lse_ref, golden_out_pt, golden_lse_pt = ref_flash_attention_pair(
        query_cpu,
        key_cpu,
        value_cpu,
        scale,
        _attn_mask(q_seqlen, key_cpu.shape[1], is_causal, window_size),
        data_type,
        softcap=softcap,
    )

    assert_fa_close(output_npu, golden_out_ref, golden_out_pt, softcap=softcap, name="out")
    assert_fa_close(softmax_lse_npu, golden_lse_ref, golden_lse_pt, softcap=softcap, name="softmax_lse")


def _assert_tnd_matches_ref(
    output_npu,
    softmax_lse_npu,
    query,
    kv_batched,
    *,
    q_offsets,
    batch_size,
    num_heads,
    head_size,
    scale,
    data_type,
    is_causal,
    window_size=WINDOW_SIZE,
    softcap=0.0,
):
    query_cpu = query.detach().cpu().reshape(batch_size, q_offsets[1], num_heads, head_size)
    key_cpu, value_cpu = kv_batched
    key_cpu = key_cpu.reshape(batch_size, key_cpu.shape[1], key_cpu.shape[2], key_cpu.shape[3])
    value_cpu = value_cpu.reshape_as(key_cpu)
    mask = _attn_mask(q_offsets[1], key_cpu.shape[1], is_causal, window_size)
    golden_out_ref, golden_lse_batched_ref, golden_out_pt, golden_lse_batched_pt = ref_flash_attention_pair(
        query_cpu, key_cpu, value_cpu, scale, mask, data_type, softcap=softcap
    )
    golden_out_ref = golden_out_ref.reshape(q_offsets[-1], num_heads, head_size)
    golden_out_pt = golden_out_pt.reshape(q_offsets[-1], num_heads, head_size)
    golden_lse_ref = golden_lse_batched_ref.permute(1, 0, 2).reshape(num_heads, q_offsets[-1])
    golden_lse_pt = golden_lse_batched_pt.permute(1, 0, 2).reshape(num_heads, q_offsets[-1])

    assert_fa_close(output_npu, golden_out_ref, golden_out_pt, softcap=softcap, name="out")
    if softmax_lse_npu is not None:
        assert_fa_close(softmax_lse_npu, golden_lse_ref, golden_lse_pt, softcap=softcap, name="softmax_lse")


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


@pytest.fixture
def metadata_spy(monkeypatch):
    """Spy on get_scheduler_metadata to prove the training interfaces route
    through the AICPU scheduler-metadata path internally (official flash-attn
    only exposes scheduler_metadata on flash_attn_with_kvcache)."""
    from flash_attn_npu_4 import flash_attn_npu_interface as interface
    calls = []
    original = interface.get_scheduler_metadata

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(interface, "get_scheduler_metadata", _spy)
    return calls


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal",
    FLASH_ATTN_FUNC_CASES,
)
def test_flash_attn_func_metadata_bsnd(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal,
    metadata_spy,
):
    query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type, device="npu")
    key = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type, device="npu")
    value = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type, device="npu")
    scale = 1.0 / (head_size ** 0.5)

    output_npu, softmax_lse_npu = flash_attn_func(
        query,
        key,
        value,
        softmax_scale=scale,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        num_splits=1,
        return_lse=True,
    )
    assert len(metadata_spy) == 1

    key_cpu = key.detach().cpu()
    value_cpu = value.detach().cpu()
    _assert_bsnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        (key_cpu, value_cpu),
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
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal,
    metadata_spy,
):
    q_lengths = [q_seqlen] * batch_size
    kv_lengths = [kv_seqlen] * batch_size
    q_offsets = _prefix_sums(q_lengths)
    kv_offsets = _prefix_sums(kv_lengths)
    cu_seqlens_q = _int32_npu(q_offsets)
    cu_seqlens_k = _int32_npu(kv_offsets)

    query = make_random_tensor((q_offsets[-1], num_heads, head_size), data_type, device="npu")
    key = make_random_tensor((kv_offsets[-1], kv_heads, head_size), data_type, device="npu")
    value = make_random_tensor((kv_offsets[-1], kv_heads, head_size), data_type, device="npu")
    scale = 1.0 / (head_size ** 0.5)

    output_npu = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=q_seqlen,
        max_seqlen_k=kv_seqlen,
        softmax_scale=scale,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        num_splits=1,
    )
    assert len(metadata_spy) == 1

    _assert_tnd_matches_ref(
        output_npu,
        None,
        query,
        (key.reshape(batch_size, kv_seqlen, kv_heads, head_size).detach().cpu(),
         value.reshape(batch_size, kv_seqlen, kv_heads, head_size).detach().cpu()),
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
    query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type, device="npu")
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
    output_npu, softmax_lse_npu, *_ = _flash_attn_kvcache(
        query,
        key_cache,
        value_cache,
        cache_seqlens,
        page_table,
        scheduler_metadata,
        max_seqlen_q=q_seqlen,
        causal=is_causal,
    )

    key_cache_cpu = key_cache.detach().cpu()
    value_cache_cpu = value_cache.detach().cpu()
    page_table_cpu = page_table.cpu()
    _assert_bsnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        gather_paged_kv_batch(
            key_cache_cpu, value_cache_cpu, page_table_cpu, kv_seqlen, block_size
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

    query = make_random_tensor((q_offsets[-1], num_heads, head_size), data_type, device="npu")
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
    output_npu, softmax_lse_npu, *_ = _flash_attn_kvcache(
        query,
        key_cache,
        value_cache,
        cache_seqlens,
        page_table,
        scheduler_metadata,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=q_seqlen,
        causal=is_causal,
    )

    key_cache_cpu = key_cache.detach().cpu()
    value_cache_cpu = value_cache.detach().cpu()
    page_table_cpu = page_table.cpu()
    _assert_tnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        gather_paged_kv_batch(
            key_cache_cpu, value_cache_cpu, page_table_cpu, kv_seqlen, block_size
        ),
        q_offsets=q_offsets,
        batch_size=batch_size,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        data_type=data_type,
        is_causal=is_causal,
    )


@pytest.mark.parametrize("is_causal", [False])
def test_flash_attn_kvcache_metadata_flash_decode(is_causal):
    """FD + metadata path with idle cores (needCoreNum < blockDim)."""
    batch_size, num_heads, kv_heads = 1, 1, 1
    q_seqlen, kv_seqlen, head_size, block_size = 1, 1024, 128, 128
    data_type = torch.bfloat16

    query = make_random_tensor((q_seqlen, num_heads, head_size), data_type, low=-1.0, high=1.0, device="npu")
    key_cache, value_cache, page_table = _make_paged_cache(
        batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type
    )
    cu_seqlens_q = _int32_npu([0, q_seqlen])
    cache_seqlens = _int32_npu([kv_seqlen])
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
    output_npu, softmax_lse_npu, *_ = _flash_attn_kvcache(
        query,
        key_cache,
        value_cache,
        cache_seqlens,
        page_table,
        scheduler_metadata,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=q_seqlen,
        softmax_scale=scale,
        causal=is_causal,
    )

    key_cache_cpu = key_cache.detach().cpu()
    value_cache_cpu = value_cache.detach().cpu()
    page_table_cpu = page_table.cpu()
    _assert_bsnd_matches_ref(
        output_npu.reshape(batch_size, q_seqlen, num_heads, head_size),
        softmax_lse_npu.reshape(batch_size, num_heads, q_seqlen),
        query.reshape(batch_size, q_seqlen, num_heads, head_size),
        gather_paged_kv_batch(
            key_cache_cpu, value_cache_cpu, page_table_cpu, kv_seqlen, block_size
        ),
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        data_type=data_type,
        is_causal=is_causal,
    )


FLASH_ATTN_FUNC_SWA_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, False, (256, 256)),
    (torch.bfloat16, 2, 4, 2, 1024, 1024, 128, True, (512, -1)),
    (torch.float16, 1, 2, 1, 777, 1024, 64, False, (300, 0)),
]


FLASH_ATTN_FUNC_SOFTCAP_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, softcap, softmax_scale
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, True, 30.0, None),
    (torch.bfloat16, 2, 4, 4, 512, 512, 128, False, 0.0, 0.05),
    (torch.float16, 1, 2, 2, 512, 512, 128, True, 50.0, 0.06),
]


FLASH_ATTN_VARLEN_SWA_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size
    (torch.bfloat16, 3, 4, 2, 512, 768, 128, False, (200, 200)),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, True, (511, -1)),
    (torch.float16, 2, 4, 4, 512, 512, 128, True, (128, -1)),
]


KV_CACHE_SWA_SOFTCAP_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, block_size, is_causal, window_size, softcap
    (torch.bfloat16, 2, 4, 4, 512, 1024, 128, 128, False, (256, 256), 0.0),
    (torch.bfloat16, 2, 4, 4, 512, 1024, 128, 128, True, (300, -1), 0.0),
    (torch.bfloat16, 2, 4, 2, 128, 1024, 128, 128, True, (-1, -1), 30.0),
    (torch.float16, 1, 2, 1, 256, 512, 128, 128, False, (128, 128), 50.0),
]


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size",
    FLASH_ATTN_FUNC_SWA_CASES,
)
def test_flash_attn_func_metadata_swa(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size,
    metadata_spy,
):
    query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type, device="npu")
    key = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type, device="npu")
    value = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type, device="npu")
    scale = 1.0 / (head_size ** 0.5)

    output_npu, softmax_lse_npu = flash_attn_func(
        query,
        key,
        value,
        softmax_scale=scale,
        causal=is_causal,
        window_size=window_size,
        num_splits=1,
        return_lse=True,
    )
    assert len(metadata_spy) == 1

    key_cpu = key.detach().cpu()
    value_cpu = value.detach().cpu()
    _assert_bsnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        (key_cpu, value_cpu),
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        data_type=data_type,
        is_causal=is_causal,
        window_size=window_size,
    )


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, softcap, softmax_scale",
    FLASH_ATTN_FUNC_SOFTCAP_CASES,
)
def test_flash_attn_func_metadata_softcap_scale(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
    is_causal, softcap, softmax_scale, metadata_spy,
):
    query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type, device="npu")
    key = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type, device="npu")
    value = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type, device="npu")
    scale = softmax_scale if softmax_scale is not None else 1.0 / (head_size ** 0.5)

    output_npu, softmax_lse_npu = flash_attn_func(
        query,
        key,
        value,
        softmax_scale=scale,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        softcap=softcap,
        num_splits=1,
        return_lse=True,
    )
    assert len(metadata_spy) == 1

    key_cpu = key.detach().cpu()
    value_cpu = value.detach().cpu()
    _assert_bsnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        (key_cpu, value_cpu),
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        data_type=data_type,
        is_causal=is_causal,
        softcap=softcap,
    )


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size",
    FLASH_ATTN_VARLEN_SWA_CASES,
)
def test_flash_attn_varlen_func_metadata_swa(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size,
    metadata_spy,
):
    q_lengths = [q_seqlen] * batch_size
    kv_lengths = [kv_seqlen] * batch_size
    q_offsets = _prefix_sums(q_lengths)
    kv_offsets = _prefix_sums(kv_lengths)
    cu_seqlens_q = _int32_npu(q_offsets)
    cu_seqlens_k = _int32_npu(kv_offsets)

    query = make_random_tensor((q_offsets[-1], num_heads, head_size), data_type, device="npu")
    key = make_random_tensor((kv_offsets[-1], kv_heads, head_size), data_type, device="npu")
    value = make_random_tensor((kv_offsets[-1], kv_heads, head_size), data_type, device="npu")
    scale = 1.0 / (head_size ** 0.5)

    output_npu = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=q_seqlen,
        max_seqlen_k=kv_seqlen,
        softmax_scale=scale,
        causal=is_causal,
        window_size=window_size,
        num_splits=1,
    )
    assert len(metadata_spy) == 1

    key_cpu = key.detach().cpu()
    value_cpu = value.detach().cpu()
    _assert_tnd_matches_ref(
        output_npu,
        None,
        query,
        (key_cpu.reshape(batch_size, kv_seqlen, kv_heads, head_size),
         value_cpu.reshape(batch_size, kv_seqlen, kv_heads, head_size)),
        q_offsets=q_offsets,
        batch_size=batch_size,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        data_type=data_type,
        is_causal=is_causal,
        window_size=window_size,
    )


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, block_size, is_causal, window_size, softcap",
    KV_CACHE_SWA_SOFTCAP_CASES,
)
def test_flash_attn_kvcache_metadata_swa_softcap(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
    block_size, is_causal, window_size, softcap
):
    query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type, low=-1.0, high=1.0, device="npu")
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
        window_size=window_size,
        softcap=softcap,
    )
    output_npu, softmax_lse_npu, *_ = _flash_attn_kvcache(
        query,
        key_cache,
        value_cache,
        cache_seqlens,
        page_table,
        scheduler_metadata,
        max_seqlen_q=q_seqlen,
        causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )

    key_cache_cpu = key_cache.detach().cpu()
    value_cache_cpu = value_cache.detach().cpu()
    page_table_cpu = page_table.cpu()
    _assert_bsnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        gather_paged_kv_batch(
            key_cache_cpu, value_cache_cpu, page_table_cpu, kv_seqlen, block_size
        ),
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        data_type=data_type,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )