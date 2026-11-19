"""BF16 Q / per-token-per-head E4M3 KV: physical-page and causal checks."""
import math

import pytest
import torch
import torch_npu

if not torch_npu.npu.is_available() or "Ascend950" not in torch_npu.npu.get_device_name():
    pytest.skip("mixed KV tests require Ascend950", allow_module_level=True)

from flash_attn_npu_3.flash_attn_npu_interface_950 import flash_attn_with_kvcache


def make_case(sq, sk, heads, batch=1):
    generator = torch.Generator().manual_seed(7919 + sq + sk + heads)
    pages_per_batch = (sk + 127) // 128
    pages = pages_per_batch * batch
    table = torch.randperm(pages, generator=generator).reshape(batch, pages_per_batch).int()
    q = torch.randn(batch, sq, heads * 16, 192, generator=generator).bfloat16()
    k = torch.randn(pages, 128, heads, 192, generator=generator).to(torch.float8_e4m3fn)
    v = torch.randn(pages, 128, heads, 192, generator=generator).to(torch.float8_e4m3fn)
    # Non-power-of-two, distinct K/V scales vary across pages, tokens and heads.
    ks = 0.25 + torch.rand(pages, 128, heads, generator=generator)
    vs = 0.5 + torch.rand(pages, 128, heads, generator=generator)
    return q, k, v, ks, vs, table


@pytest.mark.parametrize("sq,sk,heads,batch,splits", [
    (1, 1, 1, 1, 1), (2, 63, 2, 1, 1), (8, 127, 4, 1, 1),
    (3, 255, 2, 1, 1), (4, 256, 4, 1, 1),
    (6, 1025, 2, 1, 1), (7, 16385, 2, 1, 4),
    (5, 129, 2, 2, 1), (8, 385, 2, 1, 1), (65, 257, 2, 1, 1),
    (128, 512, 4, 1, 1), (129, 513, 1, 1, 1),
    (1, 4096, 4, 1, 8), (8, 4097, 2, 1, 4),
    (1, 129, 8, 1, 1),  # Exercise the scale-loading fallback beyond TP1's four KV heads.
    # Compact S/P reaches its 16-row/AIV boundary at Sq=2. Exercise input-ring
    # wraparound, a partial final page, and task/split transitions together.
    (2, 1025, 4, 2, 1), (2, 4097, 2, 2, 4), (8, 1025, 1, 3, 0),
    # Enough full query tiles to select the 128-row prefill kernel. Reuse its
    # single P bank across several KV tiles and finish on a one-token page.
    (512, 513, 2, 2, 1),
])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("layout", ["TND", "BSND"])
def test_mixed_kv(sq, sk, heads, batch, splits, causal, layout):
    torch.set_num_threads(4)
    q, k, v, ks, vs, table = make_case(sq, sk, heads, batch)
    cu = torch.arange(batch + 1, dtype=torch.int32) * sq
    query = q.reshape(batch * sq, heads * 16, 192) if layout == "TND" else q
    out, lse, *_ = flash_attn_with_kvcache(
        query.npu(), k.npu(), v.npu(),
        cache_seqlens=torch.full((batch,), sk, dtype=torch.int32, device="npu"),
        page_table=table.npu(), cu_seqlens_q=cu.npu() if layout == "TND" else None,
        max_seqlen_q=sq if layout == "TND" else None,
        k_descale=ks.npu(), v_descale=vs.npu(), causal=causal,
        num_splits=splits, return_softmax_lse=True,
    )
    torch.npu.synchronize()
    out = out.cpu().reshape(batch, sq, heads * 16, 192).float()
    lse = lse.cpu()
    if layout == "TND":
        lse = lse.reshape(heads * 16, batch, sq).permute(1, 0, 2)
    for b in range(batch):
        _check_sequence(q[b], k, v, ks, vs, table[b], sk, causal, out[b], lse[b])


def _check_sequence(q, k, v, ks, vs, pages, sk, causal, out, lse):
    sq, heads = q.shape[0], k.shape[2]
    kd = (k.float() * ks[..., None]).bfloat16()[pages.long()].reshape(-1, heads, 192)[:sk]
    vd = (v.float() * vs[..., None]).bfloat16()[pages.long()].reshape(-1, heads, 192)[:sk]
    kd = kd.repeat_interleave(16, dim=1).float()
    vd = vd.repeat_interleave(16, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), kd) / math.sqrt(192)
    if causal:
        mask = torch.arange(sk)[None, :] > (sk - sq + torch.arange(sq)[:, None])
        scores.masked_fill_(mask[None], -float("inf"))
    ref_lse = scores.logsumexp(-1)
    ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vd)
    # BF16 probability/output rounding; check both aggregate and worst error.
    delta = out - ref
    assert torch.isfinite(out).all()
    assert delta.abs().max() <= 0.025 + 0.008 * ref.abs().max()
    assert delta.square().mean().sqrt() <= 0.006 * ref.square().mean().sqrt() + 0.0002
    torch.testing.assert_close(lse, ref_lse, atol=0.003, rtol=0.0003)


@pytest.mark.parametrize("heads,q_lengths,kv_lengths", [
    (2, (128, 129, 257), (257, 385, 513)),
    (2, (8, 129, 385), (257, 385, 513)),
    (1, (513, 641), (544, 705)),
    (2, (513, 641), (544, 705)),
])
def test_mixed_kv_ragged_prefill(heads, q_lengths, kv_lengths):
    """Check every row across full causal waves, partial waves and ragged batch boundaries."""
    torch.set_num_threads(4)
    q, k, v, ks, vs, table = make_case(max(q_lengths), max(kv_lengths), heads, len(q_lengths))
    packed = torch.cat([q[b, :n] for b, n in enumerate(q_lengths)])
    cu = torch.tensor((0, *q_lengths), dtype=torch.int32).cumsum(0, dtype=torch.int32)
    out, lse, *_ = flash_attn_with_kvcache(
        packed.npu(), k.npu(), v.npu(), cache_seqlens=torch.tensor(kv_lengths, dtype=torch.int32).npu(),
        page_table=table.npu(), cu_seqlens_q=cu.npu(), max_seqlen_q=max(q_lengths),
        k_descale=ks.npu(), v_descale=vs.npu(), causal=True, num_splits=1, return_softmax_lse=True,
    )
    out, lse = out.cpu().float(), lse.cpu()
    for b, sk in enumerate(kv_lengths):
        start, end = cu[b].item(), cu[b + 1].item()
        _check_sequence(packed[start:end], k, v, ks, vs, table[b], sk, True, out[start:end], lse[:, start:end])


@pytest.mark.parametrize("bad_scale", ["missing", "per_head", "dtype", "q_scale"])
def test_mixed_kv_rejects_invalid_scales(bad_scale):
    q, k, v, ks, vs, table = make_case(1, 128, 1)
    kwargs = dict(k_descale=ks.npu(), v_descale=vs.npu())
    if bad_scale == "missing":
        kwargs["v_descale"] = None
    elif bad_scale == "per_head":
        kwargs["k_descale"] = torch.ones(1, 1, device="npu")
    elif bad_scale == "dtype":
        kwargs["k_descale"] = ks.bfloat16().npu()
    else:
        kwargs["q_descale"] = torch.ones(1, 1, device="npu")
    with pytest.raises(RuntimeError, match="descale"):
        flash_attn_with_kvcache(q.npu(), k.npu(), v.npu(), page_table=table.npu(),
                              cache_seqlens=128, num_splits=1, **kwargs)


@pytest.mark.parametrize("layout,sq,splits", [
    ("TND", 2, 1), ("TND", 128, 1), ("BSND", 2, 1), ("BSND", 128, 1),
    ("TND", 2, 0), ("TND", 2, 4), ("BSND", 2, 0), ("BSND", 2, 4),
])
def test_mixed_kv_empty_and_nonempty_tasks(layout, sq, splits):
    """Every empty KV task must initialize its output, including under split-KV."""
    torch.set_num_threads(4)
    batch, heads = 6, 2
    sk = 128 if splits == 1 else 1025
    q, k, v, ks, vs, table = make_case(sq, sk, heads, batch)
    lengths = torch.tensor([0, sk, 63, 0, sk, 0], dtype=torch.int32)
    query = q.reshape(batch * sq, heads * 16, 192) if layout == "TND" else q
    out, lse, *_ = flash_attn_with_kvcache(
        query.npu(), k.npu(), v.npu(), cache_seqlens=lengths.npu(), page_table=table.npu(),
        cu_seqlens_q=(torch.arange(batch + 1, dtype=torch.int32) * sq).npu() if layout == "TND" else None,
        max_seqlen_q=sq if layout == "TND" else None,
        k_descale=ks.npu(), v_descale=vs.npu(), num_splits=splits, return_softmax_lse=True,
    )
    out = out.cpu().reshape(batch, sq, heads * 16, 192).float()
    lse = lse.cpu()
    if layout == "TND":
        lse = lse.reshape(heads * 16, batch, sq).permute(1, 0, 2)
    for b, length in enumerate(lengths.tolist()):
        if length == 0:
            assert torch.count_nonzero(out[b]) == 0
            assert torch.isposinf(lse[b]).all()
            continue
        kd = (k.float() * ks[..., None]).bfloat16()[table[b].long()].reshape(-1, heads, 192)[:length]
        vd = (v.float() * vs[..., None]).bfloat16()[table[b].long()].reshape(-1, heads, 192)[:length]
        kd, vd = (x.repeat_interleave(16, dim=1).float() for x in (kd, vd))
        scores = torch.einsum("qhd,khd->hqk", q[b].float(), kd) / math.sqrt(192)
        ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vd)
        delta = out[b] - ref
        assert delta.abs().max() <= 0.025 + 0.008 * ref.abs().max()
        assert delta.square().mean().sqrt() <= 0.006 * ref.square().mean().sqrt() + 0.0002
        torch.testing.assert_close(lse[b], scores.logsumexp(-1), atol=0.003, rtol=0.0003)


@pytest.mark.parametrize("sq", [1, 8, 512])
def test_mixed_kv_head_major_pages(sq):
    """Changing physical page strides must preserve the per-token scale mapping."""
    sk, heads = 1025, 2
    q, k, v, ks, vs, table = make_case(sq, sk, heads)
    query = q[0].npu()
    kwargs = dict(k_descale=ks.npu(), v_descale=vs.npu(), page_table=table.npu(),
                  cache_seqlens=torch.tensor([sk], dtype=torch.int32, device="npu"),
                  cu_seqlens_q=torch.tensor([0, sq], dtype=torch.int32, device="npu"),
                  max_seqlen_q=sq, causal=True, num_splits=0, return_softmax_lse=True)
    expected, expected_lse, *_ = flash_attn_with_kvcache(query, k.npu(), v.npu(), **kwargs)

    def head_major(x, page_step):
        physical = x.transpose(1, 2).contiguous()
        pages = physical.view(torch.uint8).repeat_interleave(page_step, dim=0).npu()
        return pages[::page_step].view(x.dtype).transpose(1, 2)

    out, lse, *_ = flash_attn_with_kvcache(query, head_major(k, 2), head_major(v, 3), **kwargs)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(lse, expected_lse, rtol=0, atol=0)
