# Copyright (c) 2026, Minghua Shen.
"""Mixed BF16 Q + FP8(E4M3) KV: scheduler-metadata and NPUGraph replay checks.

Covers the "graph capture" path for the mixed-KV kernel:
  * get_scheduler_metadata(..., kv_dtype=torch.float8_e4m3fn) now produces the
    AICPU tiling for quant_mode=2 (bf16 Q + fp8 KV, Dqk=Dv=192);
  * flash_attn_with_kvcache(..., scheduler_metadata=md) accepts it and must
    return exactly the same tensors as the host-tiling path;
  * the same call can be captured into torch.npu.NPUGraph and replayed.

Run from the repository root:
    python -m pytest tests/test_flash_attn_npu_v3_mixed_kv_graph.py -v
"""

import pytest
import torch
import torch_npu

from flash_attn_npu_3 import flash_attn_with_kvcache, get_scheduler_metadata, prepare_fd_metadata
from test_flash_attn_npu_v3_mixed_kv import _check_sequence, make_case


if not torch_npu.npu.is_available() or "Ascend950" not in torch_npu.npu.get_device_name():
    pytest.skip("mixed KV tests require Ascend950", allow_module_level=True)


def _metadata(batch, sq, sk, heads, causal, window):
    q = torch.empty((batch * sq, heads * 16, 192), dtype=torch.bfloat16)
    cache_seqlens = torch.full((batch,), sk, dtype=torch.int32, device="npu")
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="npu") * sq
    return (
        q,
        cache_seqlens,
        cu_q,
        get_scheduler_metadata(
            batch,
            sq,
            sk * 128,  # page-table capacity bound, same as the forward call
            heads * 16,
            heads,
            192,
            cache_seqlens,
            qkv_dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            cu_seqlens_q=cu_q,
            page_size=128,
            causal=causal,
            window_size=window,
            num_splits=0,
        ),
    )


def _call(query, k, v, ks, vs, table, cache_seqlens, cu_q, sq, causal, window, metadata):
    """All inputs are already on the device: the capture region must contain only
    the operator launch (no host->device copies)."""
    kwargs = dict(
        cache_seqlens=cache_seqlens,
        page_table=table,
        cu_seqlens_q=cu_q,
        max_seqlen_q=sq,
        k_descale=ks,
        v_descale=vs,
        causal=causal,
        window_size=window,
        num_splits=0,
        return_softmax_lse=True,
    )
    if metadata is not None:
        kwargs["scheduler_metadata"] = metadata
    out, lse, *_ = flash_attn_with_kvcache(query, k, v, **kwargs)
    return out, lse


@pytest.mark.parametrize(
    "sq,sk,heads,batch,causal,window",
    [
        (4, 1024, 1, 12, True, (511, 0)),  # step5 decode 形态（GQA 16:1 + SWA）
        (1, 8192, 1, 12, True, (511, 0)),
        (512, 513, 2, 1, True, (511, 0)),  # prefill（走 128 行 Q tile）
        (65, 257, 2, 1, False, (-1, -1)),
    ],
)
def test_mixed_kv_metadata_and_graph(sq, sk, heads, batch, causal, window):
    torch.set_num_threads(4)
    q_cpu, k_cpu, v_cpu, ks_cpu, vs_cpu, table_cpu = make_case(sq, sk, heads, batch)
    _, cache_seqlens, cu_q, scheduler_metadata = _metadata(batch, sq, sk, heads, causal, window)
    query = q_cpu.reshape(batch * sq, heads * 16, 192).npu()
    k, v, ks, vs, table = k_cpu.npu(), v_cpu.npu(), ks_cpu.npu(), vs_cpu.npu(), table_cpu.npu()

    # 1) host tiling vs AICPU metadata must agree bit-exactly
    out_host, lse_host = _call(
        query, k, v, ks, vs, table, cache_seqlens, cu_q, sq, causal, window, None
    )
    out_meta, lse_meta = _call(
        query, k, v, ks, vs, table, cache_seqlens, cu_q, sq, causal, window, scheduler_metadata
    )
    torch.npu.synchronize()
    assert torch.equal(out_meta, out_host), "metadata tiling differs from host tiling"
    assert torch.equal(lse_meta, lse_host), "metadata LSE differs from host tiling"

    # 2) NPUGraph capture + replay must reproduce the same tensors
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out_graph, lse_graph = _call(
            query, k, v, ks, vs, table, cache_seqlens, cu_q, sq, causal, window, scheduler_metadata
        )
    graph.replay()
    torch.npu.synchronize()
    assert torch.equal(out_graph, out_host), "graph replay output differs"
    assert torch.equal(lse_graph, lse_host), "graph replay LSE differs"

    # 3) accuracy against the CPU reference (same oracle as the mixed-KV suite)
    out_cpu = out_host.cpu().reshape(batch, sq, heads * 16, 192).float()
    lse_cpu = lse_host.cpu().reshape(heads * 16, batch, sq).permute(1, 0, 2)
    for b in range(batch):
        _check_sequence(
            q_cpu[b],
            k_cpu,
            v_cpu,
            ks_cpu,
            vs_cpu,
            table_cpu[b],
            sk,
            causal,
            out_cpu[b],
            lse_cpu[b],
            window,
        )


def test_mixed_kv_paired_prefill_flag():
    """Paired-prefill shapes must report the paired launch flag from metadata.

    The AICPU tiling enables the paired Q-head pipeline (two L1 Q banks) for
    full-attention prefill waves. The host launch selects that kernel from a
    flag cached on the metadata tensor; without it the metadata path silently
    falls back to the generic kernel (bit-identical, but ~20% slower).
    """
    torch.set_num_threads(4)
    sq, sk, heads, batch = 4096, 8192, 1, 1
    q_cpu, k_cpu, v_cpu, ks_cpu, vs_cpu, table_cpu = make_case(sq, sk, heads, batch)
    window = (-1, -1)
    _, cache_seqlens, cu_q, scheduler_metadata = _metadata(batch, sq, sk, heads, True, window)
    assert getattr(scheduler_metadata, "paired_prefill", None) is True

    query = q_cpu.reshape(batch * sq, heads * 16, 192).npu()
    k, v, ks, vs, table = k_cpu.npu(), v_cpu.npu(), ks_cpu.npu(), vs_cpu.npu(), table_cpu.npu()
    out_host, lse_host = _call(
        query, k, v, ks, vs, table, cache_seqlens, cu_q, sq, True, window, None
    )
    out_meta, lse_meta = _call(
        query, k, v, ks, vs, table, cache_seqlens, cu_q, sq, True, window, scheduler_metadata
    )
    torch.npu.synchronize()
    assert torch.equal(out_meta, out_host)
    assert torch.equal(lse_meta, lse_host)


def test_mixed_kv_flash_decode_metadata_and_graph():
    """FlashDecode keeps its pipeline through metadata + NPUGraph replay.

    FlashDecode's launch shape (active cores, workspace size) follows the
    per-batch sequence lengths, so ``prepare_fd_metadata`` runs the host tiling
    once, uploads it into the metadata buffer and caches the launch shape.
    """
    torch.set_num_threads(4)
    sq, sk, heads, batch = 4, 8192, 1, 4
    q_cpu, k_cpu, v_cpu, ks_cpu, vs_cpu, table_cpu = make_case(sq, sk, heads, batch)
    window = (-1, -1)
    _, cache_seqlens, cu_q, scheduler_metadata = _metadata(batch, sq, sk, heads, True, window)
    query = q_cpu.reshape(batch * sq, heads * 16, 192).npu()
    k, v, ks, vs, table = k_cpu.npu(), v_cpu.npu(), ks_cpu.npu(), vs_cpu.npu(), table_cpu.npu()

    scheduler_metadata = prepare_fd_metadata(
        scheduler_metadata,
        query,
        k,
        v,
        cache_seqlens,
        page_table=table,
        cu_seqlens_q=cu_q,
        max_seqlen_q=sq,
        causal=True,
        window_size=window,
        return_softmax_lse=True,
        num_splits=0,
    )
    assert scheduler_metadata.flash_decode is True
    assert scheduler_metadata.fd_active_cores > 0

    out_host, lse_host = _call(
        query, k, v, ks, vs, table, cache_seqlens, cu_q, sq, True, window, None
    )
    out_meta, lse_meta = _call(
        query, k, v, ks, vs, table, cache_seqlens, cu_q, sq, True, window, scheduler_metadata
    )
    torch.npu.synchronize()
    assert torch.equal(out_meta, out_host)
    assert torch.equal(lse_meta, lse_host)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out_graph, lse_graph = _call(
            query, k, v, ks, vs, table, cache_seqlens, cu_q, sq, True, window, scheduler_metadata
        )
    graph.replay()
    torch.npu.synchronize()
    assert torch.equal(out_graph, out_host)
    assert torch.equal(lse_graph, lse_host)


def test_mixed_kv_lean_entry_matches_public_api():
    """The lean decode entry must match the public API bit-exactly."""
    from flash_attn_npu_3 import flash_attn_with_kvcache_lean

    torch.set_num_threads(4)
    sq, sk, heads, batch = 4, 8192, 1, 4
    q_cpu, k_cpu, v_cpu, ks_cpu, vs_cpu, table_cpu = make_case(sq, sk, heads, batch)
    window = (511, 0)
    _, cache_seqlens, cu_q, scheduler_metadata = _metadata(batch, sq, sk, heads, True, window)
    query = q_cpu.reshape(batch * sq, heads * 16, 192).npu()
    k, v, ks, vs, table = k_cpu.npu(), v_cpu.npu(), ks_cpu.npu(), vs_cpu.npu(), table_cpu.npu()

    out_ref, lse_ref = _call(
        query, k, v, ks, vs, table, cache_seqlens, cu_q, sq, True, window, scheduler_metadata
    )
    out_lean, lse_lean = flash_attn_with_kvcache_lean(
        query,
        k,
        v,
        cache_seqlens,
        table,
        cu_seqlens_q=cu_q,
        max_seqlen_q=sq,
        scheduler_metadata=scheduler_metadata,
        k_descale=ks,
        v_descale=vs,
        causal=True,
        window_size=window,
        return_softmax_lse=True,
    )
    torch.npu.synchronize()
    assert torch.equal(out_lean, out_ref)
    assert torch.equal(lse_lean, lse_ref)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out_graph, lse_graph = flash_attn_with_kvcache_lean(
            query,
            k,
            v,
            cache_seqlens,
            table,
            cu_seqlens_q=cu_q,
            max_seqlen_q=sq,
            scheduler_metadata=scheduler_metadata,
            k_descale=ks,
            v_descale=vs,
            causal=True,
            window_size=window,
            return_softmax_lse=True,
        )
    graph.replay()
    torch.npu.synchronize()
    assert torch.equal(out_graph, out_ref)
    assert torch.equal(lse_graph, lse_ref)


def test_mixed_kv_metadata_reuse_across_kv_lengths():
    """Non-FlashDecode metadata is independent of the runtime KV lengths.

    Build the metadata from a short cache_seqlens snapshot, then run with
    longer caches: results must stay bit-exact against host tiling, so a decode
    loop does not have to rebuild metadata as the cache grows.
    """
    torch.set_num_threads(4)
    sq, sk, heads, batch = 4, 8192, 1, 4
    q_cpu, k_cpu, v_cpu, ks_cpu, vs_cpu, table_cpu = make_case(sq, sk, heads, batch)
    window = (511, 0)
    _, _, cu_q, scheduler_metadata = _metadata(batch, sq, min(sk, 512), heads, True, window)
    query = q_cpu.reshape(batch * sq, heads * 16, 192).npu()
    k, v, ks, vs, table = k_cpu.npu(), v_cpu.npu(), ks_cpu.npu(), vs_cpu.npu(), table_cpu.npu()
    for kv_len in (2048, sk):
        cache_seqlens = torch.full((batch,), kv_len, dtype=torch.int32, device="npu")
        out_host, lse_host = _call(
            query, k, v, ks, vs, table, cache_seqlens, cu_q, sq, True, window, None
        )
        out_meta, lse_meta = _call(
            query, k, v, ks, vs, table, cache_seqlens, cu_q, sq, True, window, scheduler_metadata
        )
        torch.npu.synchronize()
        assert torch.equal(out_meta, out_host)
        assert torch.equal(lse_meta, lse_host)
