# Copyright (c) 2026, Minghua Shen.

import pytest
import torch

from flash_attn_npu_3 import flash_attn_with_kvcache, get_scheduler_metadata
from tests.common.attention_ref import ref_flash_attention_pair
from tests.common.compare import assert_fa_close
from tests.common.test_utils import make_random_tensor


DATA_TYPE = torch.bfloat16
BATCH_SIZE = 1
NUM_HEADS = 4
NUM_KV_HEADS = 2
Q_SEQLEN = 16
KV_SEQLEN = 128
HEAD_SIZE = 128
BLOCK_SIZE = 128
SCALE = 1.0 / (HEAD_SIZE**0.5)
WINDOW_SIZE = (-1, -1)


def _run_flash_attn(
    query,
    key_cache,
    value_cache,
    cache_seqlens,
    page_table,
    cu_seqlens_q,
    scheduler_metadata,
    is_causal,
):
    return flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=Q_SEQLEN,
        softmax_scale=query.shape[-1] ** -0.5,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        scheduler_metadata=scheduler_metadata,
        return_softmax_lse=True,
    )


@pytest.mark.golden_api("flash_attn_with_kvcache")
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("metadata_copy", ["original", "clone", "copy_"])
def test_flash_attn_kvcache_graph(is_causal, metadata_copy):
    _check_kvcache_graph(is_causal, metadata_copy, HEAD_SIZE)


@pytest.mark.golden_api("flash_attn_with_kvcache")
def test_d512_flash_attn_kvcache_graph():
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("Ascend950 D512 graph replay")
    _check_kvcache_graph(False, "copy_", 512)


def _check_kvcache_graph(is_causal, metadata_copy, head_size):
    scale = head_size**-0.5
    query = make_random_tensor((Q_SEQLEN, NUM_HEADS, head_size), DATA_TYPE, device="npu")
    key_cache = make_random_tensor(
        (BATCH_SIZE, BLOCK_SIZE, NUM_KV_HEADS, head_size), DATA_TYPE, device="npu"
    )
    value_cache = make_random_tensor(
        (BATCH_SIZE, BLOCK_SIZE, NUM_KV_HEADS, head_size), DATA_TYPE, device="npu"
    )
    cache_seqlens = torch.tensor([KV_SEQLEN], dtype=torch.int32).npu()
    cu_seqlens_q = torch.tensor([0, Q_SEQLEN], dtype=torch.int32).npu()
    page_table = torch.tensor([[0]], dtype=torch.int32).npu()
    metadata_q_lengths = {"cu_seqlens_q": cu_seqlens_q}

    metadata_kwargs = dict(
        batch_size=BATCH_SIZE,
        max_seqlen_q=Q_SEQLEN,
        max_seqlen_k=KV_SEQLEN,
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=head_size,
        cache_seqlens=cache_seqlens,
        qkv_dtype=DATA_TYPE,
        page_size=BLOCK_SIZE,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        **metadata_q_lengths,
    )
    scheduler_metadata = get_scheduler_metadata(**metadata_kwargs)
    if metadata_copy == "clone":
        scheduler_metadata = scheduler_metadata.clone()
    elif metadata_copy == "copy_":
        scheduler_metadata = torch.empty_like(scheduler_metadata).copy_(scheduler_metadata)
    metadata_ptr = scheduler_metadata.data_ptr()

    causal_mask = None
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(Q_SEQLEN, KV_SEQLEN),
            diagonal=KV_SEQLEN - Q_SEQLEN + 1,
        ).bool()
    golden_out_ref, golden_lse_ref, golden_out_pt, golden_lse_pt = ref_flash_attention_pair(
        query.cpu().unsqueeze(0),
        key_cache.cpu(),
        value_cache.cpu(),
        scale,
        causal_mask,
        DATA_TYPE,
        0.0,
    )
    golden_out_ref = golden_out_ref.squeeze(0)
    golden_out_pt = golden_out_pt.squeeze(0)

    _run_flash_attn(
        query,
        key_cache,
        value_cache,
        cache_seqlens,
        page_table,
        cu_seqlens_q,
        scheduler_metadata,
        is_causal,
    )
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output_npu, lse_npu, *_ = _run_flash_attn(
            query,
            key_cache,
            value_cache,
            cache_seqlens,
            page_table,
            cu_seqlens_q,
            scheduler_metadata,
            is_causal,
        )

    graph.replay()
    torch.npu.synchronize()

    assert_fa_close(output_npu, golden_out_ref, golden_out_pt, name="graph out")
    if head_size == 512:
        assert_fa_close(lse_npu, golden_lse_ref[0], golden_lse_pt[0], name="graph lse")

    # Keep the captured address and static kernel configuration unchanged while
    # replacing the schedule for shorter/longer actual KV lengths. This capacity
    # is below the FlashDecode threshold, so the kernel branch stays fixed.
    query_cpu = query.cpu().unsqueeze(0)
    key_cpu, value_cpu = key_cache.cpu(), value_cache.cpu()
    previous_out = output_npu.cpu().clone()
    for kv_length in (64, 96, KV_SEQLEN):
        cache_seqlens.fill_(kv_length)
        updated_metadata = get_scheduler_metadata(**metadata_kwargs)
        scheduler_metadata.copy_(updated_metadata)
        assert scheduler_metadata.data_ptr() == metadata_ptr
        graph.replay()
        torch.npu.synchronize()

        mask = None
        if is_causal:
            mask = torch.triu(
                torch.ones(Q_SEQLEN, kv_length),
                diagonal=kv_length - Q_SEQLEN + 1,
            ).bool()
        expected_ref, expected_lse_ref, expected_pt, expected_lse_pt = ref_flash_attention_pair(
            query_cpu,
            key_cpu[:, :kv_length],
            value_cpu[:, :kv_length],
            scale,
            mask,
            DATA_TYPE,
            0.0,
        )
        assert_fa_close(
            output_npu,
            expected_ref.squeeze(0),
            expected_pt.squeeze(0),
            name=f"graph updated KV={kv_length}",
        )
        if head_size == 512:
            assert_fa_close(lse_npu, expected_lse_ref[0], expected_lse_pt[0], name="graph updated lse")
        current_out = output_npu.cpu()
        assert not torch.equal(current_out, previous_out)
        previous_out = current_out.clone()


@pytest.mark.golden_api("flash_attn_with_kvcache")
def test_flash_attn_kvcache_graph_long_kv():
    kv_length = 1024
    q = torch.zeros((1, NUM_HEADS, HEAD_SIZE), dtype=DATA_TYPE, device="npu")
    k = torch.zeros(
        (kv_length // BLOCK_SIZE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE),
        dtype=DATA_TYPE,
        device="npu",
    )
    cache_seqlens = torch.tensor([kv_length], dtype=torch.int32, device="npu")
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    metadata = get_scheduler_metadata(
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=kv_length,
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        cache_seqlens=cache_seqlens,
        qkv_dtype=DATA_TYPE,
        page_size=BLOCK_SIZE,
        cu_seqlens_q=cu_q,
    )
    kwargs = dict(
        q=q,
        k_cache=k,
        v_cache=torch.ones_like(k),
        cache_seqlens=cache_seqlens,
        page_table=torch.arange(kv_length // BLOCK_SIZE, dtype=torch.int32, device="npu").reshape(
            1, -1
        ),
        cu_seqlens_q=cu_q,
        max_seqlen_q=1,
        scheduler_metadata=metadata,
    )
    flash_attn_with_kvcache(**kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = flash_attn_with_kvcache(**kwargs)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)
