# Copyright (c) 2026, Minghua Shen.

import pytest
import torch
import torch_npu

if "Ascend950" in (torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""):
    pytest.skip("flash_attn_func graph only on Ascend910", allow_module_level=True)

from flash_attn_npu_4 import flash_attn_func, flash_attn_varlen_func, get_scheduler_metadata
from tests.test_flash_attn_npu_v4 import ref_flash_attention, build_cann_causal_mask

RTOL = 1e-2
ATOL = 1e-2
DATA_TYPE = torch.bfloat16
BATCH_SIZE = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2
Q_SEQLEN = 128
KV_SEQLEN = 128
HEAD_SIZE = 128
SCALE = 1.0 / (HEAD_SIZE ** 0.5)
WINDOW_SIZE = (-1, -1)


@pytest.fixture(autouse=True)
def _cleanup_npu():
    """Clean up NPU state between tests to avoid graph-mode state leaks."""
    yield
    torch.npu.synchronize()
    torch.npu.empty_cache()


def _rand_npu(shape, dtype=DATA_TYPE):
    return (2 * torch.rand(shape) - 1).to(dtype).npu()


# ==============================================================================
# flash_attn_func graph tests
# ==============================================================================


@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attn_func_graph_with_metadata(is_causal):
    """Graph replay with precomputed scheduler_metadata for flash_attn_func."""

    query = _rand_npu((BATCH_SIZE, Q_SEQLEN, NUM_HEADS, HEAD_SIZE))
    key_cache = _rand_npu((BATCH_SIZE, KV_SEQLEN, NUM_KV_HEADS, HEAD_SIZE))
    value_cache = _rand_npu((BATCH_SIZE, KV_SEQLEN, NUM_KV_HEADS, HEAD_SIZE))

    cache_seqlens = torch.full((BATCH_SIZE,), KV_SEQLEN, dtype=torch.int32).npu()
    scheduler_metadata = get_scheduler_metadata(
        batch_size=BATCH_SIZE,
        max_seqlen_q=Q_SEQLEN,
        max_seqlen_k=KV_SEQLEN,
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        qkv_dtype=DATA_TYPE,
        cache_seqlens=cache_seqlens,
        causal=is_causal,
        window_size=WINDOW_SIZE,
    )

    # golden
    atten_mask = None
    if is_causal:
        atten_mask = build_cann_causal_mask()[:Q_SEQLEN, :KV_SEQLEN]
    golden_out = torch.empty((BATCH_SIZE, Q_SEQLEN, NUM_HEADS, HEAD_SIZE), dtype=DATA_TYPE)
    for i in range(BATCH_SIZE):
        query_cpu = query.detach().cpu()[i:i+1]
        key_cpu = key_cache.detach().cpu()[i:i+1]
        value_cpu = value_cache.detach().cpu()[i:i+1]
        output, _ = ref_flash_attention(
            query_cpu, key_cpu, value_cpu,
            SCALE, atten_mask, DATA_TYPE, softcap=0.0,
        )
        golden_out[i:i + 1] = output.reshape(Q_SEQLEN, NUM_HEADS, HEAD_SIZE)

    # warm-up
    flash_attn_func(
        query, key_cache, value_cache,
        softmax_scale=SCALE, causal=is_causal,
        window_size=WINDOW_SIZE, scheduler_metadata=scheduler_metadata,
        return_lse=False,
    )
    torch.npu.synchronize()

    # graph capture
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output_npu = flash_attn_func(
            query, key_cache, value_cache,
            softmax_scale=SCALE, causal=is_causal,
            window_size=WINDOW_SIZE, scheduler_metadata=scheduler_metadata,
            return_lse=False,
        )

    graph.replay()
    torch.npu.synchronize()

    torch.testing.assert_close(output_npu.cpu(), golden_out, rtol=RTOL, atol=ATOL)
    del graph


# ==============================================================================
# flash_attn_varlen_func graph tests (TND layout)
# ==============================================================================

VARLEN_BATCH_SIZE = 2
VARLEN_NUM_HEADS = 4
VARLEN_NUM_KV_HEADS = 2
VARLEN_Q_SEQLEN = 128
VARLEN_KV_SEQLEN = 128
VARLEN_HEAD_SIZE = 128
VARLEN_SCALE = 1.0 / (VARLEN_HEAD_SIZE ** 0.5)


@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attn_varlen_func_graph_with_metadata(is_causal):
    """Graph replay with precomputed scheduler_metadata for flash_attn_varlen_func (TND)."""

    q_sequences = [VARLEN_Q_SEQLEN] * VARLEN_BATCH_SIZE
    kv_sequences = [VARLEN_KV_SEQLEN] * VARLEN_BATCH_SIZE
    t_q = sum(q_sequences)
    t_kv = sum(kv_sequences)

    query = _rand_npu((t_q, VARLEN_NUM_HEADS, VARLEN_HEAD_SIZE))
    key_cache = _rand_npu((t_kv, VARLEN_NUM_KV_HEADS, VARLEN_HEAD_SIZE))
    value_cache = _rand_npu((t_kv, VARLEN_NUM_KV_HEADS, VARLEN_HEAD_SIZE))

    cu_seqlens_q = torch.tensor([0] + [sum(q_sequences[:i + 1]) for i in range(VARLEN_BATCH_SIZE)], dtype=torch.int32).npu()
    cu_seqlens_k = torch.tensor([0] + [sum(kv_sequences[:i + 1]) for i in range(VARLEN_BATCH_SIZE)], dtype=torch.int32).npu()
    cache_seqlens = torch.tensor(kv_sequences, dtype=torch.int32).npu()

    scheduler_metadata = get_scheduler_metadata(
        batch_size=VARLEN_BATCH_SIZE,
        max_seqlen_q=VARLEN_Q_SEQLEN,
        max_seqlen_k=VARLEN_KV_SEQLEN,
        num_heads_q=VARLEN_NUM_HEADS,
        num_heads_kv=VARLEN_NUM_KV_HEADS,
        headdim=VARLEN_HEAD_SIZE,
        qkv_dtype=DATA_TYPE,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        causal=is_causal,
        window_size=WINDOW_SIZE,
    )
    # golden
    atten_mask = None
    if is_causal:
        atten_mask = build_cann_causal_mask()[:VARLEN_Q_SEQLEN, :VARLEN_KV_SEQLEN]
    golden_out = torch.empty((t_q, VARLEN_NUM_HEADS, VARLEN_HEAD_SIZE), dtype=DATA_TYPE)
    golden_lse = torch.empty((VARLEN_BATCH_SIZE, VARLEN_NUM_HEADS, VARLEN_Q_SEQLEN), dtype=torch.float32)
    for i in range(VARLEN_BATCH_SIZE):
        q_start = 0 if i == 0 else sum(q_sequences[:i])
        kv_start = 0 if i == 0 else sum(kv_sequences[:i])
        query_cpu = query.detach().cpu()[q_start:q_start + q_sequences[i]].unsqueeze(0)
        key_cpu = key_cache.detach().cpu()[kv_start:kv_start + kv_sequences[i]].unsqueeze(0)
        value_cpu = value_cache.detach().cpu()[kv_start:kv_start + kv_sequences[i]].unsqueeze(0)
        output, lse = ref_flash_attention(
            query_cpu, key_cpu, value_cpu,
            VARLEN_SCALE, atten_mask, DATA_TYPE, softcap=0.0,
        )
        golden_out[q_start:q_start + q_sequences[i]] = output
        golden_lse[i:i + 1] = lse.reshape(VARLEN_NUM_HEADS, VARLEN_Q_SEQLEN)

    # warm-up
    flash_attn_varlen_func(
        query, key_cache, value_cache,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=VARLEN_Q_SEQLEN,
        max_seqlen_k=VARLEN_KV_SEQLEN,
        softmax_scale=VARLEN_SCALE,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        scheduler_metadata=scheduler_metadata,
        return_lse=False,
    )
    torch.npu.synchronize()

    # graph capture
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output_npu = flash_attn_varlen_func(
            query, key_cache, value_cache,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=VARLEN_Q_SEQLEN,
            max_seqlen_k=VARLEN_KV_SEQLEN,
            softmax_scale=VARLEN_SCALE,
            causal=is_causal,
            window_size=WINDOW_SIZE,
            scheduler_metadata=scheduler_metadata,
            return_lse=False,
        )

    graph.replay()
    torch.npu.synchronize()

    torch.testing.assert_close(output_npu.cpu(), golden_out, rtol=RTOL, atol=ATOL)
    del graph