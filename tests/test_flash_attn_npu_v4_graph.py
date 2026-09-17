# Copyright (c) 2026, Minghua Shen.

import pytest
import torch
import torch_npu

_device_name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
if "Ascend910" not in _device_name:
    pytest.skip("flash_attn_varlen_func / get_scheduler_metadata only on Ascend910", allow_module_level=True)

from flash_attn_npu_4 import flash_attn_varlen_func, get_scheduler_metadata
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
):
    return flash_attn_varlen_func(
        query,
        key_cache,
        value_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        page_table=page_table,
        max_seqlen_q=Q_SEQLEN,
        softmax_scale=SCALE,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        scheduler_metadata=scheduler_metadata,
        return_lse=True,
    )


@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attn_varlen_graph(is_causal):
    query = make_random_tensor((Q_SEQLEN, NUM_HEADS, HEAD_SIZE), DATA_TYPE, device="npu")
    key_cache = make_random_tensor((BATCH_SIZE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE), DATA_TYPE, device="npu")
    value_cache = make_random_tensor((BATCH_SIZE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE), DATA_TYPE, device="npu")
    cache_seqlens = torch.tensor([KV_SEQLEN], dtype=torch.int32).npu()
    page_table = torch.tensor([[0]], dtype=torch.int32).npu()
    cu_seqlens_q = torch.tensor([0, Q_SEQLEN], dtype=torch.int32).npu()

    scheduler_metadata = get_scheduler_metadata(
        batch_size=BATCH_SIZE,
        max_seqlen_q=Q_SEQLEN,
        max_seqlen_k=KV_SEQLEN,
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        cache_seqlens=cache_seqlens,
        qkv_dtype=DATA_TYPE,
        cu_seqlens_q=cu_seqlens_q,
        page_size=BLOCK_SIZE,
        causal=is_causal,
        window_size=WINDOW_SIZE,
    )

    causal_mask = None
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(Q_SEQLEN, KV_SEQLEN),
            diagonal=KV_SEQLEN - Q_SEQLEN + 1,
        ).bool()
    golden_out_ref, _, golden_out_pt, _ = ref_flash_attention_pair(
        query.cpu().unsqueeze(0),
        key_cache.cpu(),
        value_cache.cpu(),
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