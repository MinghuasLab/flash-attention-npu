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
    (8, 4161, 1, 1, 4),  # A 65-token tail crosses the 64-lane causal mask boundary.
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
    _run_paged_case(sq, sk, heads, batch, splits, causal, layout)


@pytest.mark.parametrize("sq,sk,heads,batch,splits,causal,window", [
    (1, 4097, 2, 2, 0, True, (511, 0)),       # The first visible token is inside a physical page.
    (1, 2048, 4, 1, 0, True, (511, 0)),       # Exact page boundaries.
    (8, 2049, 2, 1, 0, True, (511, 0)),       # Both ends fall inside a page.
    (4, 2048, 1, 2, 0, True, (127, 0)),       # Each packed head repeats the four row masks on both edge pages.
    (8, 1027, 1, 2, 1, False, (17, 3)),       # Both masks on the same page.
    (193, 1025, 1, 1, 0, True, (511, 0)),      # One wave of individual Q128 heads, including a partial Q tile.
    (383, 1027, 1, 1, 0, True, (511, 0)),      # One paired wave, both SWA edges and a short final KV page.
    (257, 515, 2, 1, 0, True, (63, 0)),        # One wave of paired heads, narrow window and unaligned pages.
    (513, 4099, 2, 1, 0, True, (511, 0)),      # A short Q tail can absorb the KV page offset.
    (513, 4223, 2, 1, 0, True, (511, 0)),      # Shifting by 126 leaves two rows in the first Q tile.
    (513, 899, 4, 1, 1, True, (511, 0)),      # Paired heads, partial Q and KV tiles.
    (512, 769, 4, 1, 1, False, (287, 257)),   # Finite noncausal window.
    (513, 193, 4, 1, 1, True, (511, 0)),      # Fully masked prefix followed by valid rows.
])
@pytest.mark.parametrize("layout", ["TND", "BSND"])
def test_mixed_kv_swa(sq, sk, heads, batch, splits, causal, window, layout):
    _run_paged_case(sq, sk, heads, batch, splits, causal, layout, window)


def _run_paged_case(sq, sk, heads, batch, splits, causal, layout, window=(-1, -1)):
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
        num_splits=splits, window_size=window, return_softmax_lse=True,
    )
    torch.npu.synchronize()
    out = out.cpu().reshape(batch, sq, heads * 16, 192).float()
    lse = lse.cpu()
    if layout == "TND":
        lse = lse.reshape(heads * 16, batch, sq).permute(1, 0, 2)
    for b in range(batch):
        _check_sequence(q[b], k, v, ks, vs, table[b], sk, causal, out[b], lse[b], window)


def _check_sequence(q, k, v, ks, vs, pages, sk, causal, out, lse, window=(-1, -1)):
    sq, heads = q.shape[0], k.shape[2]
    kd = (k.float() * ks[..., None]).bfloat16()[pages.long()].reshape(-1, heads, 192)[:sk]
    vd = (v.float() * vs[..., None]).bfloat16()[pages.long()].reshape(-1, heads, 192)[:sk]
    kd = kd.repeat_interleave(16, dim=1).float()
    vd = vd.repeat_interleave(16, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), kd) / math.sqrt(192)
    anchor = (sk - sq + torch.arange(sq))[:, None]
    keys = torch.arange(sk)[None, :]
    if causal or window[1] >= 0:
        right = 0 if causal else window[1]
        scores.masked_fill_((keys > anchor + right)[None], -torch.inf)
    if window[0] >= 0:
        scores.masked_fill_((keys < anchor - window[0])[None], -torch.inf)
    ref_lse = scores.logsumexp(-1)
    empty = ref_lse.isneginf()
    prob = scores.softmax(-1).masked_fill(empty[..., None], 0)
    ref = torch.einsum("hqk,khd->qhd", prob, vd)
    ref_lse.masked_fill_(empty, torch.inf)
    # BF16 probability/output rounding; check both aggregate and worst error.
    delta = out - ref
    assert torch.isfinite(out).all()
    assert delta.abs().max() <= 0.025 + 0.008 * ref.abs().max()
    assert delta.square().mean().sqrt() <= 0.006 * ref.square().mean().sqrt() + 0.0002
    torch.testing.assert_close(lse, ref_lse, atol=0.003, rtol=0.0003)


@pytest.mark.parametrize("heads,q_lengths,kv_lengths,window", [
    (2, (128, 129, 257), (257, 385, 513), (-1, -1)),
    (2, (8, 129, 385), (257, 385, 513), (-1, -1)),
    (1, (513, 641), (544, 705), (-1, -1)),
    (2, (513, 641), (544, 705), (-1, -1)),
    (1, (513, 641), (1041, 1469), (511, 0)),
    (2, (257, 385), (1025, 1283), (767, 0)),
])
def test_mixed_kv_ragged_prefill(heads, q_lengths, kv_lengths, window):
    """Check every row across full causal waves, partial waves and ragged batch boundaries."""
    torch.set_num_threads(4)
    q, k, v, ks, vs, table = make_case(max(q_lengths), max(kv_lengths), heads, len(q_lengths))
    packed = torch.cat([q[b, :n] for b, n in enumerate(q_lengths)])
    cu = torch.tensor((0, *q_lengths), dtype=torch.int32).cumsum(0, dtype=torch.int32)
    out, lse, *_ = flash_attn_with_kvcache(
        packed.npu(), k.npu(), v.npu(), cache_seqlens=torch.tensor(kv_lengths, dtype=torch.int32).npu(),
        page_table=table.npu(), cu_seqlens_q=cu.npu(), max_seqlen_q=max(q_lengths),
        k_descale=ks.npu(), v_descale=vs.npu(), causal=True, window_size=window, num_splits=1, return_softmax_lse=True,
    )
    out, lse = out.cpu().float(), lse.cpu()
    for b, sk in enumerate(kv_lengths):
        start, end = cu[b].item(), cu[b + 1].item()
        _check_sequence(packed[start:end], k, v, ks, vs, table[b], sk, True, out[start:end], lse[:, start:end], window)



@pytest.mark.parametrize("splits", [0, 4])
@pytest.mark.parametrize("q_lens,kv_lens", [
    ((0, 3, 0, 8, 1, 0), (4097, 0, 17, 4097, 1025, 0)),
    ((8, 0, 1), (1, 0, 1024)),  # The widest task writes directly; only the one-token task needs partials.
])
def test_mixed_kv_packed_decode_empty_queries(splits, q_lens, kv_lens):
    q, k, v, ks, vs, table = make_case(max(q_lens), max(kv_lens), 2, len(q_lens))
    packed = torch.cat([q[b, :n] for b, n in enumerate(q_lens)])
    cu = torch.tensor((0, *q_lens), dtype=torch.int32).cumsum(0, dtype=torch.int32)
    out, lse, *_ = flash_attn_with_kvcache(
        packed.npu(), k.npu(), v.npu(), k_descale=ks.npu(), v_descale=vs.npu(),
        cu_seqlens_q=cu.npu(), max_seqlen_q=max(q_lens), page_table=table.npu(),
        cache_seqlens=torch.tensor(kv_lens, dtype=torch.int32).npu(), causal=True,
        num_splits=splits, return_softmax_lse=True,
    )
    out, lse = out.cpu().float(), lse.cpu()
    assert out.shape == packed.shape
    assert lse.shape == (packed.size(1), packed.size(0))
    for b, (nq, nk) in enumerate(zip(q_lens, kv_lens)):
        if nq == 0:
            continue
        rows = slice(cu[b].item(), cu[b + 1].item())
        if nk:
            _check_sequence(packed[rows], k, v, ks, vs, table[b], nk, True, out[rows], lse[:, rows], (-1, -1))
        else:
            assert torch.count_nonzero(out[rows]) == 0
            assert torch.isposinf(lse[:, rows]).all()


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


@pytest.mark.parametrize("batch,sq,kv_heads,window,packed", [
    (1, 1, 1, (511, 0), True),
    (1, 8, 2, (511, 0), False),
    (1, 128, 4, (511, 0), True),
    (1, 505, 2, (511, 0), True),  # A replay moves the interior Q boundary back across a KV page.
    (1, 1, 1, (-1, -1), True),
    (2, 1, 1, (-1, -1), True),
])
def test_mixed_kv_graph_cross_page(batch, sq, kv_heads, window, packed):
    from flash_attn_npu_3 import copy_scheduler_metadata_, get_scheduler_metadata

    torch.set_num_threads(4)
    q, k, v, ks, vs, table = [x.npu() for x in make_case(sq, 2049, kv_heads, batch)]
    cu = None
    if packed:
        q = q.flatten(0, 1)
        cu = torch.arange(batch + 1, device="npu", dtype=torch.int32) * sq
    lengths = torch.full((batch,), 2047, device="npu", dtype=torch.int32)

    def prepare(out=None):
        return get_scheduler_metadata(
            batch, sq, 2176, kv_heads * 16, kv_heads, 192, lengths, cu_seqlens_q=cu,
            page_size=128, causal=True, window_size=window, kv_dtype=torch.float8_e4m3fn, out=out,
        )

    def forward(md=None):
        return flash_attn_with_kvcache(
            q, k, v, cache_seqlens=lengths, k_descale=ks, v_descale=vs, scheduler_metadata=md,
            page_table=table, cu_seqlens_q=cu, max_seqlen_q=sq, causal=True,
            window_size=window, return_softmax_lse=True,
        )[:2]

    def capture(md):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = forward(md)
        return graph, outputs

    def check(graph, outputs):
        graph.replay()
        reference = forward()
        for actual, expected in zip(outputs, reference):
            torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)

    md = prepare()
    graph, outputs = capture(md)
    check(graph, outputs)
    lengths.fill_(2049)
    if window == (-1, -1) and batch == 1:
        with pytest.raises(ValueError, match="launch parameters changed"):
            prepare(out=md)
        new_md = prepare()
        with pytest.raises(ValueError, match="launch parameters changed"):
            copy_scheduler_metadata_(md, new_md)
        new_graph, new_outputs = capture(new_md)
        check(new_graph, new_outputs)
    else:
        if window == (-1, -1):
            prepared = prepare()
            assert copy_scheduler_metadata_(md, prepared) is md
            with pytest.raises(ValueError, match="same device and configuration"):
                copy_scheduler_metadata_(md, prepared.clone())
        check(graph, outputs)
    lengths.fill_(2047)
    if window == (-1, -1) and batch == 2:
        assert prepare(out=md) is md
    check(graph, outputs)


@pytest.mark.parametrize("splits", [0, 4])
def test_mixed_kv_graph_packed_empty_request(splits):
    from flash_attn_npu_3 import get_scheduler_metadata

    q, k, v, ks, vs, table = [x.npu() for x in make_case(1, 2049, 1, 2)]
    q = q[:1, 0].contiguous()
    cu = torch.tensor([0, 0, 1], device="npu", dtype=torch.int32)
    lengths = torch.tensor([0, 2049], device="npu", dtype=torch.int32)
    md = get_scheduler_metadata(2, 1, 2176, 16, 1, 192, lengths, cu_seqlens_q=cu,
                                page_size=128, kv_dtype=torch.float8_e4m3fn, num_splits=splits, causal=True)
    kwargs = dict(cache_seqlens=lengths, cu_seqlens_q=cu, max_seqlen_q=1, page_table=table, causal=True,
                  k_descale=ks, v_descale=vs, num_splits=splits, return_softmax_lse=True)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = flash_attn_with_kvcache(q, k, v, scheduler_metadata=md, **kwargs)[:2]
    graph.replay()
    reference = flash_attn_with_kvcache(q, k, v, **kwargs)[:2]
    for actual, expected in zip(outputs, reference):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("kv_heads,threshold", [(4, 16384), (2, 24576)])
@pytest.mark.parametrize("head_major", [False, True])
def test_mixed_kv_decode_graph_page_transition(kv_heads, threshold, head_major):
    """One prepared plan supports BBND and strided pages, including live length growth."""
    from flash_attn_npu_3 import copy_scheduler_metadata_, get_scheduler_metadata

    q, k, v, ks, vs, table = make_case(1, threshold + 1, kv_heads)
    # Give the new page appreciable weight even in a long context.
    vs[table[0, -1].item(), 0] *= 1024

    def cache(x):
        if head_major:
            return x.transpose(1, 2).contiguous().npu().transpose(1, 2)
        return x.npu()

    q, k, v = q[0].npu(), cache(k), cache(v)
    lengths = torch.tensor([threshold - 1], device="npu", dtype=torch.int32)
    cu = torch.tensor([0, 1], device="npu", dtype=torch.int32)
    options = dict(cache_seqlens=lengths, cu_seqlens_q=cu, max_seqlen_q=1, page_table=table.npu(),
                   k_descale=ks.npu(), v_descale=vs.npu(), causal=True, return_softmax_lse=True)

    def plan():
        return get_scheduler_metadata(1, 1, table.shape[1] * 128, 16 * kv_heads, kv_heads, 192, lengths,
                                      cu_seqlens_q=cu, page_size=128, causal=True, kv_dtype=torch.float8_e4m3fn)

    metadata = plan()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out, lse, *_ = flash_attn_with_kvcache(q, k, v, scheduler_metadata=metadata, **options)
    for sk in (threshold - 1, threshold + 1, threshold - 1):
        lengths.fill_(sk)
        copy_scheduler_metadata_(metadata, plan())
        graph.replay()
        ref, ref_lse, *_ = flash_attn_with_kvcache(q, k, v, **options)
        torch.testing.assert_close(out.cpu(), ref.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(lse.cpu(), ref_lse.cpu(), atol=0, rtol=0)
        if sk == threshold + 1:
            assert not torch.equal(ref.cpu(), first)
        else:
            first = ref.cpu()


def test_mixed_decode_graph_dispatch_change_preserves_plan():
    from flash_attn_npu_3 import copy_scheduler_metadata_, get_scheduler_metadata

    cu = torch.tensor([0, 1], device="npu", dtype=torch.int32)
    lengths = torch.tensor([4095], device="npu", dtype=torch.int32)

    def prepare():
        return get_scheduler_metadata(1, 1, 4224, 64, 4, 192, lengths, cu_seqlens_q=cu,
                                      page_size=128, causal=True, kv_dtype=torch.float8_e4m3fn)

    short = prepare()
    lengths.fill_(4097)
    paired = prepare()
    for dst, src in ((short, paired), (paired, short)):
        before = dst.clone()
        with pytest.raises(ValueError, match="launch parameters changed"):
            copy_scheduler_metadata_(dst, src)
        assert torch.equal(dst, before)


@pytest.mark.parametrize("batch,sq,ragged_kv,ragged_q,window,splits", [
    (150, 8, False, False, (-1, -1), 0), (150, 8, True, False, (-1, -1), 0),
    (150, 8, True, True, (-1, -1), 0), (150, 8, True, True, (511, 0), 0),
    (513, 1, True, False, (-1, -1), 0),  # Multi-task chunks, including a short final chunk.
    (19, 8, False, False, (-1, -1), 0), (19, 8, True, False, (-1, -1), 0),
    (34, 8, False, False, (-1, -1), 0),  # Multiple waves of numerical KV splits.
    (16, 8, True, False, (-1, -1), 0),
    (8, 8, True, False, (-1, -1), 8),  # More requested segments than cores.
])
def test_mixed_mtp_batch_permutation(batch, sq, ragged_kv, ragged_q, window, splits):
    """Permuting complete requests must preserve every request's O and LSE bits."""
    torch.set_num_threads(4)
    sk = 16384
    generator = torch.Generator().manual_seed(12137)
    q = torch.randn(batch, sq, 16, 192, generator=generator).bfloat16()
    # Distinct page tables over a shared physical KV pool also exercise scale lookup.
    _, k, v, ks, vs, base_table = make_case(sq, sk, 1)
    pages = torch.stack([base_table[0, torch.randperm(base_table.size(1), generator=generator)]
                         for _ in range(batch)]).npu()
    k, v, ks, vs = (x.npu() for x in (k, v, ks, vs))
    lengths = torch.full((batch,), sk, dtype=torch.int32)
    if ragged_kv:
        lengths[::2] = 2049
        lengths[1::5] = 8191
    lengths = lengths.npu()
    q_lengths = torch.arange(batch) % (sq + 1) if ragged_q else torch.full((batch,), sq)
    offsets = [0, *q_lengths.cumsum(0).tolist()]
    packed_q = torch.cat([request[:n] for request, n in zip(q, q_lengths.tolist())]).npu()

    def evaluate(order, num_splits=splits):
        indices = order.int().npu()
        rows = torch.cat([torch.arange(offsets[i], offsets[i + 1]) for i in order.tolist()])
        cu = torch.cat((torch.zeros(1, dtype=torch.int32), q_lengths[order].cumsum(0, dtype=torch.int32))).npu()
        o, lse, *_ = flash_attn_with_kvcache(
            packed_q.index_select(0, rows.int().npu()), k, v, k_descale=ks, v_descale=vs,
            page_table=pages.index_select(0, indices), cache_seqlens=lengths.index_select(0, indices),
            cu_seqlens_q=cu, max_seqlen_q=sq, causal=True, num_splits=num_splits, window_size=window, return_softmax_lse=True,
        )
        restore = rows.argsort()
        return (o.cpu()[restore].contiguous().view(torch.int16),
                lse.cpu()[:, restore].contiguous().view(torch.int32))

    identity = torch.arange(batch)
    reference = evaluate(identity)
    if batch in (150, 513) and window == (-1, -1):
        for actual, expected in zip(reference, evaluate(identity, num_splits=1)):
            assert torch.equal(actual, expected)
    for i in (1, 4, batch - 1):
        rows = slice(offsets[i], offsets[i + 1])
        _check_sequence(q[i, :q_lengths[i]], k.cpu(), v.cpu(), ks.cpu(), vs.cpu(), pages[i].cpu(),
                        int(lengths[i]), True, reference[0][rows].view(torch.bfloat16).float(),
                        reference[1][:, rows].view(torch.float32), window)
    for order in (identity, identity.roll(1), torch.randperm(batch, generator=generator)):
        for name, actual, expected in zip(("O", "LSE"), evaluate(order), reference):
            assert torch.equal(actual, expected), f"{name}: {(actual != expected).sum().item()} changed elements"
