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
        max_seqlen_q=query.shape[0],
        softmax_scale=query.shape[-1] ** -0.5,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        scheduler_metadata=scheduler_metadata,
        return_softmax_lse=True,
    )


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("head_size", [HEAD_SIZE, 512])
def test_flash_attn_kvcache_graph(is_causal, head_size):
    if head_size == 512 and "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("D512 graph coverage requires Ascend950")
    q_len = 33 if head_size == 512 else Q_SEQLEN
    kv_len = 257 if head_size == 512 else KV_SEQLEN
    num_pages = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale = head_size ** -0.5
    query = make_random_tensor((q_len, NUM_HEADS, head_size), DATA_TYPE, device="npu")
    key_cache = make_random_tensor(
        (BATCH_SIZE * num_pages, BLOCK_SIZE, NUM_KV_HEADS, head_size), DATA_TYPE, device="npu"
    )
    value_cache = make_random_tensor(
        (BATCH_SIZE * num_pages, BLOCK_SIZE, NUM_KV_HEADS, head_size), DATA_TYPE, device="npu"
    )
    cache_seqlens = torch.tensor([kv_len], dtype=torch.int32).npu()
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32).npu()
    page_table = torch.arange(BATCH_SIZE * num_pages, dtype=torch.int32).reshape(BATCH_SIZE, num_pages).npu()
    metadata_q_lengths = {"cu_seqlens_q": cu_seqlens_q}

    scheduler_metadata = get_scheduler_metadata(
        batch_size=BATCH_SIZE,
        max_seqlen_q=q_len,
        max_seqlen_k=kv_len,
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

    causal_mask = None
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(q_len, kv_len),
            diagonal=kv_len - q_len + 1,
        ).bool()
    golden_out_ref, _, golden_out_pt, _ = ref_flash_attention_pair(
        query.cpu().unsqueeze(0),
        key_cache.cpu().reshape(BATCH_SIZE, -1, NUM_KV_HEADS, head_size)[:, :kv_len],
        value_cache.cpu().reshape(BATCH_SIZE, -1, NUM_KV_HEADS, head_size)[:, :kv_len],
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
        output_npu, *_ = _run_flash_attn(
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
