# Copyright (c) 2026, Minghua Shen.

import ctypes

import numpy as np
import pytest
import torch
import torch_npu

if "Ascend950" in (torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""):
    pytest.skip("flash_attn_npu (v2) not supported on Ascend950", allow_module_level=True)

from flash_attn_npu import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)
from tests.common.test_utils import gather_paged_kv_batch, make_random_tensor
from tests.common.attention_ref import ref_flash_attention_pair
from tests.common.compare import assert_fa_close


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
    )


def _make_paged_cache(batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type):
    max_blocks_per_seq = (kv_seqlen + block_size - 1) // block_size
    num_blocks = batch_size * max_blocks_per_seq
    key_cache = make_random_tensor((num_blocks, block_size, kv_heads, head_size), data_type, low=-1.0, high=1.0, device="npu")
    value_cache = make_random_tensor((num_blocks, block_size, kv_heads, head_size), data_type, low=-1.0, high=1.0, device="npu")
    block_table = torch.arange(num_blocks, dtype=torch.int32).reshape(batch_size, max_blocks_per_seq).npu()
    return key_cache, value_cache, block_table


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


@pytest.fixture
def metadata_spy(monkeypatch):
    """Spy on get_scheduler_metadata to prove the training interfaces route
    through the AICPU scheduler-metadata path internally (official flash-attn
    only exposes scheduler_metadata on flash_attn_with_kvcache)."""
    from flash_attn_npu import flash_attn_npu_interface as interface
    calls = []
    original = interface.get_scheduler_metadata

    def _spy(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result

    monkeypatch.setattr(interface, "get_scheduler_metadata", _spy)
    return calls


FLASH_ATTN_FUNC_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, False),
    (torch.float16, 2, 4, 4, 1024, 1024, 128, True),
    (torch.float16, 7, 1, 1, 512, 512, 128, False),
]


FLASH_ATTN_VARLEN_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal
    (torch.bfloat16, 1, 1, 1, 512, 1024, 128, True),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, False),
    (torch.float16, 7, 5, 1, 512, 512, 128, True),
]


FLASH_ATTN_FUNC_SWA_SOFTCAP_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size, softcap
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, False, (256, 256), 0.0),
    (torch.bfloat16, 2, 4, 2, 1024, 1024, 128, True, (512, -1), 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, (512, 256), 30.0),
    (torch.float16, 2, 2, 2, 512, 512, 128, False, (64, 128), 30.0),
]


FLASH_ATTN_VARLEN_SWA_SOFTCAP_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size, softcap
    (torch.bfloat16, 3, 4, 2, 512, 768, 128, False, (200, 200), 0.0),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, True, (511, -1), 0.0),
    (torch.bfloat16, 1, 1, 1, 512, 1024, 128, True, (512, 0), 30.0),
    (torch.float16, 1, 2, 2, 512, 512, 128, False, (64, 128), 30.0),
]


KV_CACHE_BSND_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, block_size, is_causal, window_size, softcap
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 128, False, (-1, -1), 0.0),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 128, True, (-1, -1), 0.0),
    (torch.bfloat16, 2, 4, 4, 512, 1024, 128, 128, False, (256, 256), 0.0),
    (torch.bfloat16, 2, 4, 4, 512, 1024, 128, 128, True, (300, -1), 0.0),
    (torch.bfloat16, 2, 4, 2, 128, 1024, 128, 128, True, (-1, -1), 30.0),
    (torch.float16, 1, 2, 1, 256, 512, 128, 128, False, (128, 128), 50.0),
]


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal",
    FLASH_ATTN_FUNC_CASES,
    ids=[
        "bfloat16-1-1-1-1024-1024-128-False",
        "float16-2-4-4-1024-1024-128-True",
        "float16-7-1-1-512-512-128-False",
    ],
)
def test_flash_attn_func_metadata_bsnd(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal,
    metadata_spy,
):
    query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type, device="npu")
    key = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type, device="npu")
    value = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type, device="npu")
    scale = 1.0 / (head_size ** 0.5)

    output_npu, softmax_lse_npu, _ = flash_attn_func(
        query,
        key,
        value,
        softmax_scale=scale,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        return_attn_probs=True,
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
    ids=[
        "bfloat16-1-1-1-512-1024-128-True",
        "bfloat16-2-4-4-1024-1024-128-False",
        "float16-7-5-1-512-512-128-True",
    ],
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

    output_npu, softmax_lse_npu, _ = flash_attn_varlen_func(
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
        return_attn_probs=True,
    )
    assert len(metadata_spy) == 1

    key_cpu = key.detach().cpu()
    value_cpu = value.detach().cpu()
    _assert_tnd_matches_ref(
        output_npu,
        softmax_lse_npu,
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
    )


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size, softcap",
    FLASH_ATTN_FUNC_SWA_SOFTCAP_CASES,
    ids=[
        "bfloat16-2-4-4-1024-1024-128-False-(256,256)-0.0",
        "bfloat16-2-4-2-1024-1024-128-True-(512,-1)-0.0",
        "bfloat16-1-1-1-1024-1024-128-True-(512,256)-30.0",
        "float16-2-2-2-512-512-128-False-(64,128)-30.0",
    ],
)
def test_flash_attn_func_metadata_swa_softcap(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
    is_causal, window_size, softcap, metadata_spy,
):
    query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type, device="npu")
    key = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type, device="npu")
    value = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type, device="npu")
    scale = 1.0 / (head_size ** 0.5)

    output_npu, softmax_lse_npu, _ = flash_attn_func(
        query,
        key,
        value,
        softmax_scale=scale,
        causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        return_attn_probs=True,
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
        softcap=softcap,
    )


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size, softcap",
    FLASH_ATTN_VARLEN_SWA_SOFTCAP_CASES,
    ids=[
        "bfloat16-3-4-2-512-768-128-False-(200,200)-0.0",
        "bfloat16-2-4-4-1024-1024-128-True-(511,-1)-0.0",
        "bfloat16-1-1-1-512-1024-128-True-(512,0)-30.0",
        "float16-1-2-2-512-512-128-False-(64,128)-30.0",
    ],
)
def test_flash_attn_varlen_func_metadata_swa_softcap(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
    is_causal, window_size, softcap, metadata_spy,
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

    output_npu, softmax_lse_npu, _ = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        q_seqlen,
        kv_seqlen,
        softmax_scale=scale,
        causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        return_attn_probs=True,
    )
    assert len(metadata_spy) == 1

    key_cpu = key.detach().cpu()
    value_cpu = value.detach().cpu()
    _assert_tnd_matches_ref(
        output_npu,
        softmax_lse_npu,
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
        softcap=softcap,
    )


@pytest.mark.parametrize("dropout_p", [0.1, 0.5])
def test_flash_attn_func_metadata_dropout(dropout_p, metadata_spy):
    batch_size, num_heads, kv_heads = 2, 4, 2
    q_seqlen, kv_seqlen, head_size = 256, 256, 128
    data_type = torch.bfloat16
    query = make_random_tensor(
        (batch_size, q_seqlen, num_heads, head_size), data_type, device="npu"
    )
    key = make_random_tensor(
        (batch_size, kv_seqlen, kv_heads, head_size), data_type, device="npu"
    )
    value = make_random_tensor(
        (batch_size, kv_seqlen, kv_heads, head_size), data_type, device="npu"
    )
    scale = 1.0 / (head_size ** 0.5)

    output_npu, softmax_lse_npu, s_dmask = flash_attn_func(
        query,
        key,
        value,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=False,
        window_size=WINDOW_SIZE,
        return_attn_probs=True,
    )
    torch.npu.synchronize()
    assert len(metadata_spy) == 1
    meta = metadata_spy[0][2]

    # The host patches the AICPU tiling dropout fields on the scheduler-metadata path.
    tiling = _tiling_from_metadata(meta, has_mask=False)
    expected_keep = 1.0 / (1.0 - dropout_p)
    assert abs(tiling.dropoutValue - expected_keep) < 1e-4, \
        f"dropoutValue={tiling.dropoutValue} 期望 {expected_keep}"
    assert tiling.dropMaskDevice != 0, "dropMaskDevice 未被补写（nullptr）"
    assert tiling.pDevice != 0, "pDevice 未被补写（return_attn_probs 时应非空）"

    # Recover the dropout mask from S_dmask and validate the output against both references.
    drop_mask = (s_dmask > 0).to(torch.float32).cpu()
    query_cpu, key_cpu, value_cpu = query.cpu(), key.cpu(), value.cpu()
    golden_ref, golden_lse_ref, golden_pt, golden_lse_pt = ref_flash_attention_pair(
        query_cpu, key_cpu, value_cpu, scale, None, data_type, 0.0,
        drop_mask=drop_mask, dropout_p=dropout_p,
    )
    assert_fa_close(output_npu, golden_ref, golden_pt, name="out")
    assert_fa_close(
        softmax_lse_npu, golden_lse_ref, golden_lse_pt, name="softmax_lse"
    )


def test_flash_attn_varlen_func_metadata_dropout(metadata_spy):
    num_heads, kv_heads, head_size = 4, 1, 128
    max_q, max_k = 512, 1024
    data_type = torch.bfloat16
    q_offsets = [0, 512, 812, 940]
    kv_offsets = [0, 1024, 1536, 1792]
    dropout_p = 0.3
    query = make_random_tensor(
        (q_offsets[-1], num_heads, head_size), data_type, device="npu"
    )
    key = make_random_tensor(
        (kv_offsets[-1], kv_heads, head_size), data_type, device="npu"
    )
    value = make_random_tensor(
        (kv_offsets[-1], kv_heads, head_size), data_type, device="npu"
    )
    scale = 1.0 / (head_size ** 0.5)

    output_npu, lse_npu, s_dmask = flash_attn_varlen_func(
        query,
        key,
        value,
        _int32_npu(q_offsets),
        _int32_npu(kv_offsets),
        max_q,
        max_k,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=False,
        window_size=WINDOW_SIZE,
        return_attn_probs=True,
    )
    torch.npu.synchronize()
    assert len(metadata_spy) == 1
    meta = metadata_spy[0][2]

    tiling = _tiling_from_metadata(meta, has_mask=False)
    expected_keep = 1.0 / (1.0 - dropout_p)
    assert abs(tiling.dropoutValue - expected_keep) < 1e-4, \
        f"dropoutValue={tiling.dropoutValue} 期望 {expected_keep}"
    assert tiling.dropMaskDevice != 0, "dropMaskDevice 未被补写（nullptr）"
    assert tiling.pDevice != 0, "pDevice 未被补写（return_attn_probs 时应非空）"

    drop_mask = (s_dmask > 0).to(torch.float32).cpu()
    golden_ref = torch.empty((q_offsets[-1], num_heads, head_size), dtype=data_type)
    golden_pt = torch.empty_like(golden_ref)
    golden_lse_ref = torch.empty((num_heads, q_offsets[-1]), dtype=torch.float32)
    golden_lse_pt = torch.empty_like(golden_lse_ref)
    query_cpu, key_cpu, value_cpu = query.cpu(), key.cpu(), value.cpu()
    for i in range(len(q_offsets) - 1):
        qs, qe = q_offsets[i], q_offsets[i + 1]
        ks, ke = kv_offsets[i], kv_offsets[i + 1]
        gout_ref, glse_ref, gout_pt, glse_pt = ref_flash_attention_pair(
            query_cpu[qs:qe].unsqueeze(0),
            key_cpu[ks:ke].unsqueeze(0),
            value_cpu[ks:ke].unsqueeze(0),
            scale,
            None,
            data_type,
            0.0,
            drop_mask=drop_mask[i:i + 1, :, :qe - qs, :ke - ks],
            dropout_p=dropout_p,
        )
        golden_ref[qs:qe] = gout_ref[0]
        golden_pt[qs:qe] = gout_pt[0]
        golden_lse_ref[:, qs:qe] = glse_ref[0]
        golden_lse_pt[:, qs:qe] = glse_pt[0]
    assert_fa_close(output_npu, golden_ref, golden_pt, name="out")
    assert_fa_close(lse_npu, golden_lse_ref, golden_lse_pt, name="softmax_lse")


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, block_size, is_causal, window_size, softcap",
    KV_CACHE_BSND_CASES,
    ids=[
        "bfloat16-1-1-1-1024-1024-128-128-False-(-1,-1)-0.0",
        "bfloat16-5-4-4-1024-1024-128-128-True-(-1,-1)-0.0",
        "bfloat16-2-4-4-512-1024-128-128-False-(256,256)-0.0",
        "bfloat16-2-4-4-512-1024-128-128-True-(300,-1)-0.0",
        "bfloat16-2-4-2-128-1024-128-128-True-(-1,-1)-30.0",
        "float16-1-2-1-256-512-128-128-False-(128,128)-50.0",
    ],
)
def test_flash_attn_kvcache_metadata_bsnd(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
    block_size, is_causal, window_size, softcap
):
    query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type, low=-1.0, high=1.0, device="npu")
    key_cache, value_cache, block_table = _make_paged_cache(
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
    output_npu, softmax_lse_npu = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=None,
        causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        num_splits=0,
        scheduler_metadata=scheduler_metadata,
        return_softmax_lse=True,
    )

    key_cache_cpu = key_cache.detach().cpu()
    value_cache_cpu = value_cache.detach().cpu()
    block_table_cpu = block_table.cpu()
    _assert_bsnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        gather_paged_kv_batch(
            key_cache_cpu, value_cache_cpu, block_table_cpu, kv_seqlen, block_size
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


class _CoreNode(ctypes.Structure):
    _fields_ = [
        ("startBIdx", ctypes.c_int), ("startN1Idx", ctypes.c_int),
        ("startS1Idx", ctypes.c_int), ("startS2Idx", ctypes.c_int),
        ("endBIdx", ctypes.c_int), ("endN1Idx", ctypes.c_int),
        ("endS1Idx", ctypes.c_int), ("endS2Idx", ctypes.c_int),
        ("firstSplitKVTaskLseOffset", ctypes.c_int64),
        ("firstSplitKVTaskOOffset", ctypes.c_int64),
    ]


class _SplitNode(ctypes.Structure):
    _fields_ = [
        ("batchIdx", ctypes.c_int), ("headStartIdx", ctypes.c_int),
        ("headEndIdx", ctypes.c_int), ("qStartIdx", ctypes.c_int),
        ("qEndIdx", ctypes.c_int), ("splitNum", ctypes.c_int),
        ("lseTaskOffset", ctypes.c_int64), ("oTaskOffset", ctypes.c_int64),
    ]


class _FAInferTilingData(ctypes.Structure):
    """Mirror of csrc/ascend910/flash_attn_npu/tilingdata.h for flag checks."""
    _fields_ = [
        ("numHeads", ctypes.c_uint32), ("embeddingSize", ctypes.c_uint32),
        ("embeddingSizeV", ctypes.c_uint32), ("numBlocks", ctypes.c_uint32),
        ("blockSize", ctypes.c_uint32), ("maxQSeqlen", ctypes.c_uint32),
        ("maxKvSeqlen", ctypes.c_uint32), ("kvHeads", ctypes.c_uint32),
        ("batch", ctypes.c_uint32), ("maxNumBlocksPerBatch", ctypes.c_uint32),
        ("firstBatchTaskNum", ctypes.c_uint32), ("totalTaskNum", ctypes.c_uint32),
        ("maskType", ctypes.c_uint32),
        ("mm1OutSize", ctypes.c_uint64), ("smOnlineOutSize", ctypes.c_uint64),
        ("mm2OutSize", ctypes.c_uint64), ("UpdateSize", ctypes.c_uint64),
        ("workSpaceSize", ctypes.c_uint64),
        ("scaleValue", ctypes.c_float), ("softcapValue", ctypes.c_float),
        ("dropoutValue", ctypes.c_float),
        ("padding1", ctypes.c_uint64), ("padding2", ctypes.c_uint64),
        ("padding3", ctypes.c_uint32),
        ("windowSizeLeft", ctypes.c_int64), ("windowSizeRight", ctypes.c_int64),
        ("splitLseTotalSize", ctypes.c_uint64), ("splitOTotalSize", ctypes.c_uint64),
        ("totalSplitNodeNum", ctypes.c_uint32), ("needCoreNum", ctypes.c_uint32),
        ("flashDecodeFlag", ctypes.c_uint32),
        ("kvNewSeqlen", ctypes.c_uint32), ("kvCacheSeqlen", ctypes.c_uint32),
        ("coreInfo", _CoreNode * 25),
        ("splitInfo", _SplitNode * 25),
        ("pDevice", ctypes.c_uint64),
        ("dropMaskDevice", ctypes.c_uint64),
    ]


def _tiling_from_metadata(scheduler_metadata, has_mask):
    raw = scheduler_metadata.cpu().numpy()
    mask_bytes = 2048 * 2048 if has_mask else 0
    blob = raw[mask_bytes:mask_bytes + ctypes.sizeof(_FAInferTilingData)]
    tiling = _FAInferTilingData.from_buffer_copy(blob)
    return tiling


@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attn_kvcache_metadata_flash_decode(is_causal):
    """FD (BSND paged, tiny Q, long KV) must be decided and scheduled on the AICPU."""
    batch_size, num_heads, kv_heads = 2, 4, 1
    q_seqlen, kv_seqlen, head_size, block_size = 1, 4096, 128, 128
    data_type = torch.bfloat16

    query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type, low=-1.0, high=1.0, device="npu")
    key_cache, value_cache, block_table = _make_paged_cache(
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
    tiling = _tiling_from_metadata(scheduler_metadata, has_mask=is_causal)
    assert tiling.flashDecodeFlag == 1
    assert tiling.needCoreNum > 0
    assert tiling.splitLseTotalSize > 0
    assert tiling.workSpaceSize > 0

    output_npu, softmax_lse_npu = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=scale,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        num_splits=0,
        scheduler_metadata=scheduler_metadata,
        return_softmax_lse=True,
    )

    key_cache_cpu = key_cache.detach().cpu()
    value_cache_cpu = value_cache.detach().cpu()
    block_table_cpu = block_table.cpu()
    _assert_bsnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        gather_paged_kv_batch(
            key_cache_cpu, value_cache_cpu, block_table_cpu, kv_seqlen, block_size
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
    "meta_causal, meta_window, call_causal, call_window",
    [
        (False, (-1, -1), True, (-1, -1)),     # call needs a causal mask, metadata has none
        (False, (-1, -1), False, (128, 128)),  # call needs a band mask, metadata has none
        (True, (-1, -1), False, (-1, -1)),     # metadata has a causal mask, call needs none
    ],
)
def test_flash_attn_kvcache_metadata_mask_mismatch_rejected(
    meta_causal, meta_window, call_causal, call_window
):
    """Mask-layout mismatches in either direction must be rejected loudly."""
    data_type = torch.bfloat16
    batch_size, num_heads, kv_heads = 1, 2, 2
    q_seqlen, kv_seqlen, head_size, block_size = 512, 512, 128, 128

    query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type, low=-1.0, high=1.0, device="npu")
    key_cache, value_cache, block_table = _make_paged_cache(
        batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type
    )
    cache_seqlens = _int32_npu([kv_seqlen] * batch_size)

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
        is_causal=meta_causal,
        window_size=meta_window,
    )
    with pytest.raises(ValueError, match="do not match this call"):
        flash_attn_with_kvcache(
            query,
            key_cache,
            value_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=call_causal,
            window_size=call_window,
            scheduler_metadata=scheduler_metadata,
        )


def test_flash_attn_kvcache_metadata_paged_mismatch_rejected():
    """Paged geometry baked into the tiling must match the call's cache/page table."""
    data_type = torch.bfloat16
    batch_size, num_heads, kv_heads = 1, 2, 2
    q_seqlen, kv_seqlen, head_size, block_size = 512, 512, 128, 128

    query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type, low=-1.0, high=1.0, device="npu")
    key_cache, value_cache, block_table = _make_paged_cache(
        batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type
    )
    cache_seqlens = _int32_npu([kv_seqlen] * batch_size)
    base = dict(
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        num_heads=num_heads,
        kv_heads=kv_heads,
        head_size=head_size,
        cache_seqlens=cache_seqlens,
        data_type=data_type,
    )

    def call_with(metadata):
        return flash_attn_with_kvcache(
            query,
            key_cache,
            value_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            scheduler_metadata=metadata,
        )

    # Wrong page_size: the block-table page size baked into the tiling differs.
    bad_page = _metadata(**base, kv_seqlen=kv_seqlen, page_size=2 * block_size)
    with pytest.raises(ValueError, match="page_size"):
        call_with(bad_page)

    # Overprovisioned max_seqlen_k: the block-table row stride would not match.
    overprovisioned = _metadata(**base, kv_seqlen=2 * kv_seqlen, page_size=block_size)
    with pytest.raises(ValueError, match="max_seqlen_k"):
        call_with(overprovisioned)

    # Metadata created without paging consumed by a paged call (and vice versa).
    unpaged = _metadata(**base, kv_seqlen=kv_seqlen, page_size=None)
    with pytest.raises(ValueError, match="page_size"):
        call_with(unpaged)


def test_flash_attn_kvcache_metadata_softcap_mismatch_rejected():
    """softcap/softmax_scale are baked into the tiling; mismatches must be rejected."""
    data_type = torch.bfloat16
    batch_size, num_heads, kv_heads = 1, 2, 2
    q_seqlen, kv_seqlen, head_size, block_size = 512, 512, 128, 128

    query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type, low=-1.0, high=1.0, device="npu")
    key_cache, value_cache, block_table = _make_paged_cache(
        batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type
    )
    cache_seqlens = _int32_npu([kv_seqlen] * batch_size)

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
        softcap=0.0,
    )
    with pytest.raises(ValueError, match="softcap"):
        flash_attn_with_kvcache(
            query,
            key_cache,
            value_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            softcap=30.0,
            scheduler_metadata=scheduler_metadata,
        )


def test_flash_attn_kvcache_metadata_unfingerprinted_rejected():
    """A copied metadata tensor loses its creation-argument fingerprint."""
    data_type = torch.bfloat16
    batch_size, num_heads, kv_heads = 1, 2, 2
    q_seqlen, kv_seqlen, head_size, block_size = 512, 512, 128, 128

    query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type, low=-1.0, high=1.0, device="npu")
    key_cache, value_cache, block_table = _make_paged_cache(
        batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type
    )
    cache_seqlens = _int32_npu([kv_seqlen] * batch_size)

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
    ).clone()
    with pytest.raises(RuntimeError, match="fingerprint"):
        flash_attn_with_kvcache(
            query,
            key_cache,
            value_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            scheduler_metadata=scheduler_metadata,
        )


def test_flash_attn_kvcache_metadata_size_mismatch_rejected():
    """A hand-crafted buffer with a forged fingerprint but the wrong size must be
    rejected by the C++ exact-size check (defense in depth behind the Python
    fingerprint validation)."""
    data_type = torch.bfloat16
    batch_size, num_heads, kv_heads = 1, 2, 2
    q_seqlen, kv_seqlen, head_size, block_size = 512, 512, 128, 128

    query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type, low=-1.0, high=1.0, device="npu")
    key_cache, value_cache, block_table = _make_paged_cache(
        batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type
    )
    cache_seqlens = _int32_npu([kv_seqlen] * batch_size)

    good = _metadata(
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        kv_seqlen=kv_seqlen,
        num_heads=num_heads,
        kv_heads=kv_heads,
        head_size=head_size,
        cache_seqlens=cache_seqlens,
        data_type=data_type,
        page_size=block_size,
    )
    bad = torch.empty(good.numel() - 8, dtype=torch.uint8).npu()
    bad._fa_scheduler_params = good._fa_scheduler_params
    with pytest.raises(RuntimeError, match="must exactly match"):
        flash_attn_with_kvcache(
            query,
            key_cache,
            value_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            scheduler_metadata=bad,
        )
