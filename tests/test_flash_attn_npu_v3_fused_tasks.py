"""fuseTask (grouped-head task) coverage for the 910 v3 backend, with emphasis on TND.

v3 fuses several Q heads into one task for the non-paged / non-append forward:

* fixed layout (BSND/BNSD): one geometry for the whole call, published by the Host
  (or the AICPU when scheduler_metadata is precomputed) and merely loaded on device;
* TND (cu_seqlens_q / seqused_q): every batch has its own Q length, so only the
  enable flag is published and the AICore derives a shape per batch.

Both paths must agree with a float64 golden. The metadata (AICPU tiling) path is
also checked to agree with the host-tiling path on the same call.
"""

import math
import struct

import pytest
import torch
import torch_npu  # noqa: F401

from flash_attn_npu_3 import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)
from tests.common.compare import assert_fa_close

DEVICE = "npu"

# FAInferTilingData layout (see csrc/ascend910/flash_attn_npu_3/tilingdata.h).
_FUSED_TASK_FLAG_OFFSET = 112
_FUSED_FIELDS_OFFSET = 2376


def _is_ascend910():
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    return "Ascend910" in name


def _read_tiling(scheduler_metadata):
    raw = scheduler_metadata.detach().cpu().numpy().tobytes()
    (flag,) = struct.unpack_from("<I", raw, _FUSED_TASK_FLAG_OFFSET)
    fields = struct.unpack_from("<8I", raw, _FUSED_FIELDS_OFFSET)
    return flag, fields


def _masked_reference(q_b, k_b, v_b, causal, scale, dtype, offset=0):
    """Golden attention for one (already head-grouped) batch, in float64/float32.

    ``offset`` is the bottom-right alignment used when K is longer than Q:
    row i may attend columns [0, i + offset].
    """
    q_hi = q_b.to(torch.float64)
    k_hi = k_b.to(torch.float64)
    v_hi = v_b.to(torch.float64)
    scores = torch.matmul(q_hi, k_hi.transpose(-1, -2)) * scale
    if causal:
        rows = torch.arange(scores.shape[-2], device=scores.device)
        cols = torch.arange(scores.shape[-1], device=scores.device)
        scores = scores.masked_fill(cols[None, :] > rows[:, None] + offset, -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    out_hi = torch.matmul(probs, v_hi)
    out_lo = torch.matmul(torch.softmax(scores.float(), dim=-1), v_b.to(torch.float32))
    return out_hi.to(dtype), out_lo.to(dtype), lse.float(), torch.logsumexp(scores.float(), -1)


def _golden_varlen(q, k, v, q_lens, kv_lens, causal, scale, dtype, q_offsets=None):
    """Golden for packed TND inputs: out (total_q, H, D), lse (H, total_q).

    ``q_offsets`` gives the packed start row of every batch when ``q_lens`` is
    shorter than the packed rows (``seqused_q``: the tail of each batch is unused).
    """
    heads = q.shape[1]
    q_off = k_off = 0
    out_hi, out_lo, lse_hi, lse_lo = [], [], [], []
    for i, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
        if q_offsets is not None:
            q_off = q_offsets[i]
        qb = q[q_off : q_off + ql].permute(1, 0, 2)
        kb = k[k_off : k_off + kl].permute(1, 0, 2)
        vb = v[k_off : k_off + kl].permute(1, 0, 2)
        group = heads // kb.shape[0]
        kb = kb.repeat_interleave(group, dim=0)
        vb = vb.repeat_interleave(group, dim=0)
        hi, lo, hi_lse, lo_lse = _masked_reference(
            qb, kb, vb, causal, scale, dtype, offset=max(0, kl - ql)
        )
        out_hi.append(hi.transpose(0, 1))
        out_lo.append(lo.transpose(0, 1))
        lse_hi.append(hi_lse)
        lse_lo.append(lo_lse)
        q_off += ql
        k_off += kl
    return (
        torch.cat(out_hi, dim=0),
        torch.cat(out_lo, dim=0),
        torch.cat(lse_hi, dim=-1),
        torch.cat(lse_lo, dim=-1),
    )


def _cu_seqlens(lengths):
    cu = [0]
    for v in lengths:
        cu.append(cu[-1] + v)
    return cu


varlen_cases = [
    # dtype, q_lens, kv_lens, num_heads, kv_heads, head_size, causal
    (torch.bfloat16, [37, 96, 1, 140], [128, 200, 64, 300], 8, 8, 64, True),
    (torch.float16, [37, 96, 1, 140], [128, 200, 64, 300], 8, 8, 64, False),
    (torch.bfloat16, [0, 53, 129, 300], [64, 128, 256, 512], 12, 4, 128, True),
    (torch.bfloat16, [97, 61, 3, 205], [400, 300, 256, 512], 16, 2, 128, True),
    (torch.float16, [64, 128, 192], [512, 512, 512], 5, 1, 256, True),
    (torch.bfloat16, [256, 256, 256, 256], [512, 512, 512, 512], 8, 8, 128, True),
]


@pytest.mark.parametrize(
    "dtype, q_lens, kv_lens, num_heads, kv_heads, head_size, causal", varlen_cases
)
def test_varlen_tnd_fused_task(dtype, q_lens, kv_lens, num_heads, kv_heads, head_size, causal):
    """TND fused tasks: mixed per-batch Q lengths (the case fixed geometry cannot serve)."""
    if not _is_ascend910():
        pytest.skip("v3 fuseTask tests require Ascend910")

    torch.manual_seed(0)
    total_q, total_k = sum(q_lens), sum(kv_lens)
    q = torch.randn(total_q, num_heads, head_size, dtype=dtype).npu()
    k = torch.randn(total_k, kv_heads, head_size, dtype=dtype).npu()
    v = torch.randn(total_k, kv_heads, head_size, dtype=dtype).npu()
    cu_q = torch.tensor(_cu_seqlens(q_lens), dtype=torch.int32).npu()
    cu_k = torch.tensor(_cu_seqlens(kv_lens), dtype=torch.int32).npu()
    scale = 1.0 / math.sqrt(head_size)

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max(q_lens),
        max(kv_lens),
        softmax_scale=scale,
        causal=causal,
        return_attn_probs=True,
    )

    out_hi, out_lo, lse_hi, lse_lo = _golden_varlen(
        q.detach().cpu(), k.detach().cpu(), v.detach().cpu(), q_lens, kv_lens, causal, scale, dtype
    )
    assert_fa_close(out, out_hi, out_lo, name="out")
    assert_fa_close(lse.float(), lse_hi, lse_lo, name="lse")


def test_varlen_tnd_seqused_q_fused_task():
    """TND with an unused packed tail: only the seqused_q rows of each batch are attended."""
    if not _is_ascend910():
        pytest.skip("v3 fuseTask tests require Ascend910")

    torch.manual_seed(1)
    dtype, num_heads, kv_heads, head_size = torch.bfloat16, 12, 4, 128
    q_lens = [61, 7, 129]
    used = [61, 1, 128]
    kv_lens = [128, 96, 256]
    total_q, total_k = sum(q_lens), sum(kv_lens)
    q = torch.randn(total_q, num_heads, head_size, dtype=dtype).npu()
    k = torch.randn(total_k, kv_heads, head_size, dtype=dtype).npu()
    v = torch.randn(total_k, kv_heads, head_size, dtype=dtype).npu()
    cu_q = torch.tensor(_cu_seqlens(q_lens), dtype=torch.int32).npu()
    cu_k = torch.tensor(_cu_seqlens(kv_lens), dtype=torch.int32).npu()
    seqused_q = torch.tensor(used, dtype=torch.int32).npu()
    scale = 1.0 / math.sqrt(head_size)

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max(q_lens),
        max(kv_lens),
        softmax_scale=scale,
        causal=True,
        seqused_q=seqused_q,
        return_attn_probs=True,
    )

    out_hi, out_lo, lse_hi, lse_lo = _golden_varlen(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        used,
        kv_lens,
        True,
        scale,
        dtype,
        q_offsets=_cu_seqlens(q_lens),
    )
    # Only the used rows of each batch (the head of the packed slice) are written.
    keep = torch.cat(
        [torch.arange(sum(q_lens[:i]), sum(q_lens[:i]) + u) for i, u in enumerate(used)]
    )
    assert_fa_close(out[keep], out_hi, out_lo, name="out")
    assert_fa_close(lse.float()[:, keep], lse_hi, lse_lo, name="lse")


def test_bsnd_fixed_layout_fused_task():
    """Fixed layout keeps using the single published geometry."""
    if not _is_ascend910():
        pytest.skip("v3 fuseTask tests require Ascend910")

    torch.manual_seed(2)
    dtype, batch, qlen, kvlen, num_heads, kv_heads, head_size = (
        torch.bfloat16,
        4,
        140,
        256,
        8,
        8,
        128,
    )
    q = torch.randn(batch, qlen, num_heads, head_size, dtype=dtype).npu()
    k = torch.randn(batch, kvlen, kv_heads, head_size, dtype=dtype).npu()
    v = torch.randn(batch, kvlen, kv_heads, head_size, dtype=dtype).npu()
    scale = 1.0 / math.sqrt(head_size)

    out, lse = flash_attn_func(q, k, v, softmax_scale=scale, causal=True, return_attn_probs=True)

    out_hi, out_lo, lse_hi, lse_lo = [], [], [], []
    for b in range(batch):
        hi, lo, hi_lse, lo_lse = _masked_reference(
            q[b].detach().cpu().permute(1, 0, 2),
            k[b].detach().cpu().permute(1, 0, 2),
            v[b].detach().cpu().permute(1, 0, 2),
            True,
            scale,
            dtype,
            offset=max(0, kvlen - qlen),
        )
        out_hi.append(hi.transpose(0, 1))
        out_lo.append(lo.transpose(0, 1))
        lse_hi.append(hi_lse)
        lse_lo.append(lo_lse)
    assert_fa_close(out, torch.stack(out_hi), torch.stack(out_lo), name="out")
    assert_fa_close(lse.float(), torch.stack(lse_hi), torch.stack(lse_lo), name="lse")


def _nonpaged_cache(batch, kv_lens, kv_heads, head_size, dtype, generator):
    capacity = max(kv_lens)
    key = torch.randn(batch, capacity, kv_heads, head_size, dtype=dtype, generator=generator).npu()
    value = torch.randn(
        batch, capacity, kv_heads, head_size, dtype=dtype, generator=generator
    ).npu()
    return key, value


def test_kvcache_tnd_metadata_matches_host_tiling():
    """TND + precomputed metadata (AICPU per-batch geometry) matches the host tiling path."""
    if not _is_ascend910():
        pytest.skip("v3 fuseTask tests require Ascend910")

    dtype, num_heads, kv_heads, head_size = torch.bfloat16, 8, 8, 64
    q_lens = [16, 33, 7]
    kv_lens = [128, 256, 96]
    scale = 1.0 / math.sqrt(head_size)
    generator = torch.Generator().manual_seed(3)
    key, value = _nonpaged_cache(3, kv_lens, kv_heads, head_size, dtype, generator)
    q = torch.randn(sum(q_lens), num_heads, head_size, dtype=dtype, generator=generator).npu()
    cu_q = torch.tensor(_cu_seqlens(q_lens), dtype=torch.int32).npu()
    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32).npu()

    kwargs = dict(
        cache_seqlens=cache_seqlens,
        page_table=None,
        cu_seqlens_q=cu_q,
        max_seqlen_q=max(q_lens),
        softmax_scale=scale,
        causal=True,
        return_softmax_lse=True,
    )
    out_host, lse_host, *_ = flash_attn_with_kvcache(q, key, value, **kwargs)

    metadata = get_scheduler_metadata(
        len(q_lens),
        max(q_lens),
        max(kv_lens),
        num_heads,
        kv_heads,
        head_size,
        cache_seqlens,
        cu_seqlens_q=cu_q,
        page_size=None,
        causal=True,
        window_size=(-1, -1),
    )
    flag, fields = _read_tiling(metadata)
    assert flag == 1, "TND must enable fuseTask in the AICPU tiling"
    assert fields == (0,) * 8, "TND must not publish a fixed geometry"

    out_meta, lse_meta, *_ = flash_attn_with_kvcache(
        q, key, value, scheduler_metadata=metadata, **kwargs
    )
    assert torch.equal(out_host, out_meta), "host tiling and AICPU tiling disagree"
    assert torch.equal(lse_host, lse_meta), "host tiling and AICPU tiling disagree on lse"

    # Reference with per-batch KV coming from the (non-paged) cache rows [0, kv_len).
    out_hi, out_lo, lse_hi, lse_lo = [], [], [], []
    q_off = 0
    for b, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
        hi, lo, hi_lse, lo_lse = _masked_reference(
            q[q_off : q_off + ql].detach().cpu().permute(1, 0, 2),
            key[b, :kl].detach().cpu().permute(1, 0, 2),
            value[b, :kl].detach().cpu().permute(1, 0, 2),
            True,
            scale,
            dtype,
            offset=max(0, kl - ql),
        )
        out_hi.append(hi.transpose(0, 1))
        out_lo.append(lo.transpose(0, 1))
        lse_hi.append(hi_lse)
        lse_lo.append(lo_lse)
        q_off += ql
    assert_fa_close(out_host, torch.cat(out_hi), torch.cat(out_lo), name="out")
    assert_fa_close(
        lse_host.float(), torch.cat(lse_hi, dim=-1), torch.cat(lse_lo, dim=-1), name="lse"
    )


def test_kvcache_fixed_metadata_publishes_geometry():
    """Fixed-layout metadata keeps the single published geometry (regression guard)."""
    if not _is_ascend910():
        pytest.skip("v3 fuseTask tests require Ascend910")

    dtype, batch, qlen, num_heads, kv_heads, head_size = torch.bfloat16, 2, 140, 8, 8, 64
    kv_lens = [256, 256]
    generator = torch.Generator().manual_seed(4)
    key, value = _nonpaged_cache(batch, kv_lens, kv_heads, head_size, dtype, generator)
    q = torch.randn(batch, qlen, num_heads, head_size, dtype=dtype, generator=generator).npu()
    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32).npu()

    metadata = get_scheduler_metadata(
        batch,
        qlen,
        max(kv_lens),
        num_heads,
        kv_heads,
        head_size,
        cache_seqlens,
        page_size=None,
        causal=True,
        window_size=(-1, -1),
    )
    flag, fields = _read_tiling(metadata)
    assert flag == 1, "fixed-layout non-paged forward must enable fuseTask"
    assert fields[7] > 0, "fixed layout must publish the task count"

    out_meta, *_ = flash_attn_with_kvcache(
        q,
        key,
        value,
        cache_seqlens=cache_seqlens,
        page_table=None,
        max_seqlen_q=qlen,
        causal=True,
        scheduler_metadata=metadata,
        return_softmax_lse=True,
    )
    out_host, *_ = flash_attn_with_kvcache(
        q,
        key,
        value,
        cache_seqlens=cache_seqlens,
        page_table=None,
        max_seqlen_q=qlen,
        causal=True,
        return_softmax_lse=True,
    )
    assert torch.equal(out_meta, out_host), "host tiling and AICPU tiling disagree"
