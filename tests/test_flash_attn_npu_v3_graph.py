# Copyright (c) 2026, Minghua Shen.

import pytest
import torch
import torch_npu

from flash_attn_npu_3 import flash_attn_with_kvcache, get_scheduler_metadata
from tests.common.attention_ref import ref_flash_attention_pair
from tests.common.compare import assert_fa_close
from tests.common.test_utils import make_random_tensor


DATA_TYPE = torch.bfloat16
BATCH_SIZE = 1
NUM_HEADS = 4
NUM_KV_HEADS = 2
Q_SEQLEN = 16
HEAD_SIZE = 128
BLOCK_SIZE = 128
SCALE = 1.0 / (HEAD_SIZE ** 0.5)
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
    num_splits,
    return_softmax_lse,
):
    return flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=Q_SEQLEN,
        softmax_scale=SCALE,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        return_softmax_lse=return_softmax_lse,
    )


@pytest.mark.parametrize(
    "is_causal,kv_seqlen,num_splits,return_softmax_lse",
    [
        (False, 128, 0, True),
        (True, 128, 0, True),
        (False, 4096, 4, True),
        (True, 4096, 0, False),
    ],
)
def test_flash_attn_kvcache_graph(
    is_causal, kv_seqlen, num_splits, return_softmax_lse
):
    num_blocks = (kv_seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE
    query = make_random_tensor((Q_SEQLEN, NUM_HEADS, HEAD_SIZE), DATA_TYPE, device="npu")
    key_cache = make_random_tensor((num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE), DATA_TYPE, device="npu")
    value_cache = make_random_tensor((num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE), DATA_TYPE, device="npu")
    cache_seqlens = torch.tensor([kv_seqlen], dtype=torch.int32).npu()
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, -1).npu()
    cu_seqlens_q = torch.tensor([0, Q_SEQLEN], dtype=torch.int32).npu()

    scheduler_metadata = get_scheduler_metadata(
        batch_size=BATCH_SIZE,
        max_seqlen_q=Q_SEQLEN,
        max_seqlen_k=kv_seqlen,
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        cache_seqlens=cache_seqlens,
        qkv_dtype=DATA_TYPE,
        cu_seqlens_q=cu_seqlens_q,
        page_size=BLOCK_SIZE,
        causal=is_causal,
        num_splits=num_splits,
        window_size=WINDOW_SIZE,
    )

    causal_mask = None
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(Q_SEQLEN, kv_seqlen),
            diagonal=kv_seqlen - Q_SEQLEN + 1,
        ).bool()
    key_linear = key_cache.cpu().reshape(
        BATCH_SIZE, kv_seqlen, NUM_KV_HEADS, HEAD_SIZE
    )
    value_linear = value_cache.cpu().reshape(
        BATCH_SIZE, kv_seqlen, NUM_KV_HEADS, HEAD_SIZE
    )
    golden_out_ref, _, golden_out_pt, _ = ref_flash_attention_pair(
        query.cpu().unsqueeze(0),
        key_linear,
        value_linear,
        SCALE,
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
        num_splits,
        return_softmax_lse,
    )
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = _run_flash_attn(
            query,
            key_cache,
            value_cache,
            cache_seqlens,
            page_table,
            cu_seqlens_q,
            scheduler_metadata,
            is_causal,
            num_splits,
            return_softmax_lse,
        )
        output_npu = result[0] if return_softmax_lse else result

    for _ in range(2):
        graph.replay()
        torch.npu.synchronize()

    assert_fa_close(output_npu, golden_out_ref, golden_out_pt, name="graph out")
