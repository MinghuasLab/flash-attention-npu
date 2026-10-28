"""950 regressions ported from fa4-pipeline-paged-strides-rebase (c9a7b26).

Run on Ascend950 after building v3. Unsupported 950 features (D=512 and
append-KV execution) are not enabled by this migration.
"""

import importlib

import pytest
import torch
import torch_npu


@pytest.fixture(autouse=True)
def ascend950_backend():
    if not torch_npu.npu.is_available() or "Ascend950" not in torch_npu.npu.get_device_name():
        pytest.skip("requires an Ascend950 NPU")
    global flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
    from flash_attn_npu_3 import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
    torch.manual_seed(20261028)


def make_cu_seqlens(lengths):
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("physical_layout", ["BBND", "BNBD"])
@pytest.mark.parametrize("num_splits", [0, 1])
def test_paged_tnd_dim0_noncontiguous_with_zero_lengths(physical_layout, causal, dtype, num_splits):
    batch_size, num_heads, kv_heads, head_size = 3, 4, 2, 64
    page_size, pages_per_batch = 128, 2
    num_pages = batch_size * pages_per_batch
    q_lengths = [0, 3, 1]
    kv_lengths = [5, 0, 129]

    query = torch.randn((sum(q_lengths), num_heads, head_size), dtype=dtype, device="npu")
    cu_q = make_cu_seqlens(q_lengths).npu()
    cache_seqlens = torch.tensor(kv_lengths, dtype=torch.int32, device="npu")
    page_table = torch.arange(num_pages, dtype=torch.int32, device="npu").reshape(
        batch_size, pages_per_batch
    )

    # Give K and V different physical-page strides and a non-zero V storage
    # offset. Only dimensions 1..3 remain dense.
    inner_shape = (
        (page_size, kv_heads, head_size)
        if physical_layout == "BBND"
        else (kv_heads, page_size, head_size)
    )
    key_storage = torch.randn((num_pages * 2, *inner_shape), dtype=dtype, device="npu")
    value_storage = torch.randn((num_pages * 3, *inner_shape), dtype=dtype, device="npu")
    key_physical = key_storage[::2]
    value_physical = value_storage[1::3]
    key_view = key_physical if physical_layout == "BBND" else key_physical.transpose(1, 2)
    value_view = value_physical if physical_layout == "BBND" else value_physical.transpose(1, 2)
    assert not key_view.is_contiguous() and not value_view.is_contiguous()
    assert key_view.stride(0) != value_view.stride(0)
    assert key_view.shape == (num_pages, page_size, kv_heads, head_size)
    if physical_layout == "BNBD":
        assert key_view.stride(1) == head_size
        assert key_view.stride(2) == page_size * head_size

    kwargs = dict(
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_q,
        max_seqlen_q=max(q_lengths),
        causal=causal,
        num_splits=num_splits,
        return_softmax_lse=True,
    )
    out_view, lse_view, *_ = flash_attn_with_kvcache(query, key_view, value_view, **kwargs)
    out_dense, lse_dense, *_ = flash_attn_with_kvcache(
        query, key_view.contiguous(), value_view.contiguous(), **kwargs
    )

    torch.testing.assert_close(out_view, out_dense, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(lse_view, lse_dense, rtol=2e-3, atol=2e-3, equal_nan=True)
    # Batch 1 owns packed Q rows [0, 3) and has kvlen=0.
    torch.testing.assert_close(out_view[:3], torch.zeros_like(out_view[:3]), rtol=0, atol=0)
    assert torch.isposinf(lse_view[:, :3]).all()

@pytest.mark.parametrize("head_size,kv_length,num_splits", [(256, 129, 1), (64, 2048, 2)])
def test_paged_tnd_bnbd_dim0_noncontiguous_split_paths(head_size, kv_length, num_splits):
    num_heads, kv_heads, page_size = 8, 2, 128
    num_pages = (kv_length + page_size - 1) // page_size
    query = torch.randn((1, num_heads, head_size), dtype=torch.float16, device="npu")
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    cache_seqlens = torch.tensor([kv_length], dtype=torch.int32, device="npu")
    page_table = torch.arange(num_pages, dtype=torch.int32, device="npu").reshape(1, num_pages)
    key_storage = torch.randn(
        (num_pages * 2, kv_heads, page_size, head_size), dtype=torch.float16, device="npu"
    )
    value_storage = torch.randn(
        (num_pages * 3, kv_heads, page_size, head_size), dtype=torch.float16, device="npu"
    )
    key_bnbd = key_storage[::2].transpose(1, 2)
    value_bnbd = value_storage[1::3].transpose(1, 2)
    assert key_bnbd.shape == (num_pages, page_size, kv_heads, head_size)
    assert key_bnbd.stride(1) == head_size
    assert key_bnbd.stride(2) == page_size * head_size
    kwargs = dict(
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_q,
        max_seqlen_q=1,
        causal=True,
        num_splits=num_splits,
        return_softmax_lse=True,
    )

    out_bnbd, lse_bnbd, *_ = flash_attn_with_kvcache(query, key_bnbd, value_bnbd, **kwargs)
    out_bbnd, lse_bbnd, *_ = flash_attn_with_kvcache(
        query,
        key_bnbd.contiguous(),
        value_bnbd.contiguous(),
        **kwargs,
    )
    torch.testing.assert_close(out_bnbd, out_bbnd, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(lse_bnbd, lse_bbnd, rtol=2e-3, atol=2e-3)


def test_paged_tnd_total_q_zero_fast_path():
    batch_size, num_heads, kv_heads, head_size = 2, 4, 2, 64
    query = torch.empty((0, num_heads, head_size), dtype=torch.float16, device="npu")
    cu_q = torch.zeros((batch_size + 1,), dtype=torch.int32, device="npu")
    cache_seqlens = torch.zeros((batch_size,), dtype=torch.int32, device="npu")
    page_table = torch.arange(batch_size, dtype=torch.int32, device="npu").reshape(batch_size, 1)
    key_storage = torch.randn(
        (batch_size * 2, kv_heads, 128, head_size), dtype=torch.float16, device="npu"
    )
    value_storage = torch.randn_like(key_storage)

    out, lse, *_ = flash_attn_with_kvcache(
        query,
        key_storage[::2].transpose(1, 2),
        value_storage[::2].transpose(1, 2),
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_q,
        max_seqlen_q=0,
        num_splits=1,
        return_softmax_lse=True,
    )
    assert out.shape == query.shape
    assert lse.shape == (num_heads, 0)
    assert out.numel() == 0 and lse.numel() == 0

@pytest.mark.parametrize(
    "q_lengths,kv_lengths,extra_q,extra_kv,causal,paged",
    [
        ([4, 7, 1], [8, 9, 5], 6, 0, False, False),
        ([4, 7, 1], [8, 9, 5], 6, 5, True, False),
        ([0, 3, 6], [4, 0, 20], 9, 0, True, True),
        ([2, 5], [16, 12], 4, 0, False, True),
    ],
)
def test_tnd_total_exceeds_cu_seqlens_q(q_lengths, kv_lengths, extra_q, extra_kv, causal, paged):
    """Packed TND storage may be longer than cu_seqlens[-1]; the tail is unused."""
    num_heads, kv_heads, head_size = 4, 2, 64
    dtype = torch.float16
    used_q = sum(q_lengths)
    used_k = sum(kv_lengths)
    cu_q = make_cu_seqlens(q_lengths).npu()
    assert int(cu_q[-1]) == used_q

    query = torch.randn((used_q, num_heads, head_size), dtype=dtype, device="npu")
    query_padded = torch.cat(
        [query, torch.randn((extra_q, num_heads, head_size), dtype=dtype, device="npu")],
        dim=0,
    )
    assert query_padded.shape[0] > int(cu_q[-1])

    if paged:
        page_size = 128
        num_pages = len(kv_lengths)
        key = torch.randn((num_pages, page_size, kv_heads, head_size), dtype=dtype, device="npu")
        value = torch.randn_like(key)
        kwargs = dict(
            cache_seqlens=torch.tensor(kv_lengths, dtype=torch.int32, device="npu"),
            page_table=torch.arange(num_pages, dtype=torch.int32, device="npu").view(num_pages, 1),
            cu_seqlens_q=cu_q,
            max_seqlen_q=max(q_lengths),
            causal=causal,
            num_splits=1,
            return_softmax_lse=True,
        )
        out_ref, lse_ref, *_ = flash_attn_with_kvcache(query, key, value, **kwargs)
        out_pad, lse_pad, *_ = flash_attn_with_kvcache(query_padded, key, value, **kwargs)
    else:
        cu_k = make_cu_seqlens(kv_lengths).npu()
        key = torch.randn((used_k, kv_heads, head_size), dtype=dtype, device="npu")
        value = torch.randn_like(key)
        if extra_kv:
            key = torch.cat(
                [key, torch.randn((extra_kv, kv_heads, head_size), dtype=dtype, device="npu")],
                dim=0,
            )
            value = torch.cat(
                [value, torch.randn((extra_kv, kv_heads, head_size), dtype=dtype, device="npu")],
                dim=0,
            )
            assert key.shape[0] > int(cu_k[-1])
        out_ref, lse_ref = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_q,
            cu_k,
            max(q_lengths),
            max(kv_lengths),
            causal=causal,
            return_attn_probs=True,
        )
        out_pad, lse_pad = flash_attn_varlen_func(
            query_padded,
            key,
            value,
            cu_q,
            cu_k,
            max(q_lengths),
            max(kv_lengths),
            causal=causal,
            return_attn_probs=True,
        )

    # The 950 training API returns T,H LSE; kvcache returns H,T.
    if not paged:
        lse_ref = lse_ref.transpose(0, 1)
        lse_pad = lse_pad.transpose(0, 1)
    assert out_pad.shape[0] == used_q + extra_q
    assert lse_pad.shape == (num_heads, used_q + extra_q)
    torch.testing.assert_close(out_pad[:used_q], out_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(lse_pad[:, :used_q], lse_ref, rtol=2e-3, atol=2e-3, equal_nan=True)

@pytest.mark.parametrize("paged", [False, True])
def test_tnd_total_shorter_than_cu_seqlens_q_errors(paged):
    """Packed T smaller than cu_seqlens_q[-1] would read past q."""
    num_heads, kv_heads, head_size = 4, 2, 64
    dtype = torch.float16
    q_lengths = [4, 7, 1]
    kv_lengths = [8, 9, 5]
    cu_q = make_cu_seqlens(q_lengths).npu()
    used_q = int(cu_q[-1])
    query = torch.randn((used_q - 1, num_heads, head_size), dtype=dtype, device="npu")
    assert query.shape[0] < used_q
    if paged:
        page_size = 128
        num_pages = len(kv_lengths)
        key = torch.randn((num_pages, page_size, kv_heads, head_size), dtype=dtype, device="npu")
        value = torch.randn_like(key)
        with pytest.raises(RuntimeError, match="cu_seqlens_q"):
            flash_attn_with_kvcache(
                query,
                key,
                value,
                cache_seqlens=torch.tensor(kv_lengths, dtype=torch.int32, device="npu"),
                page_table=torch.arange(num_pages, dtype=torch.int32, device="npu").view(num_pages, 1),
                cu_seqlens_q=cu_q,
                max_seqlen_q=max(q_lengths),
                causal=True,
                num_splits=1,
            )
    else:
        used_k = sum(kv_lengths)
        key = torch.randn((used_k, kv_heads, head_size), dtype=dtype, device="npu")
        value = torch.randn_like(key)
        with pytest.raises(RuntimeError, match="cu_seqlens_q"):
            flash_attn_varlen_func(
                query,
                key,
                value,
                cu_q,
                make_cu_seqlens(kv_lengths).npu(),
                max(q_lengths),
                max(kv_lengths),
                causal=True,
            )


def _fa3_inner_strided(source):
    """Keep logical values and the last-dimension stride while padding every row."""
    padded = torch.zeros(
        *source.shape[:-1], source.shape[-1] + 1, dtype=source.dtype, device=source.device
    )
    view = padded[..., : source.shape[-1]]
    view.copy_(source)
    assert view.stride(-1) == 1 and not view.is_contiguous()
    return view

@pytest.mark.parametrize(
    "targets",
    (("query",), ("key",), ("value",), ("query", "key", "value")),
    ids=("query", "key", "value", "all-qkv"),
)
def test_fa3_dense_outer_noncontiguous_qkv_falls_back_to_contiguous(targets):
    """Match FA4/FIA: dense outer-strided QKV are materialized."""
    batch, q_seqlen, kv_seqlen = 2, 16, 32
    num_heads, kv_heads, head_size = 4, 2, 64
    query = torch.randn(
        (batch, q_seqlen, num_heads, head_size), dtype=torch.float16, device="npu"
    )
    key = torch.randn(
        (batch, kv_seqlen, kv_heads, head_size), dtype=torch.float16, device="npu"
    )
    value = torch.randn_like(key)
    query_view = _fa3_inner_strided(query) if "query" in targets else query
    key_view = _fa3_inner_strided(key) if "key" in targets else key
    value_view = _fa3_inner_strided(value) if "value" in targets else value

    out_view = flash_attn_func(query_view, key_view, value_view, num_splits=1)
    out_ref = flash_attn_func(query, key, value, num_splits=1)
    torch.testing.assert_close(out_view, out_ref, rtol=0.0, atol=0.0)


def test_fa3_append_kv_rejects_noncontiguous_dense_cache():
    """A cache fallback must not hide append-KV's in-place update target."""
    batch, q_seqlen, cache_len, new_len = 1, 1, 16, 1
    num_heads, kv_heads, head_size = 4, 2, 64
    query = torch.randn(
        (batch, q_seqlen, num_heads, head_size), dtype=torch.float16, device="npu"
    )
    key_cache = _fa3_inner_strided(
        torch.randn((batch, cache_len, kv_heads, head_size), dtype=torch.float16, device="npu")
    )
    value_cache = _fa3_inner_strided(torch.randn_like(key_cache.contiguous()))
    key_new = torch.randn(
        (batch, new_len, kv_heads, head_size), dtype=torch.float16, device="npu"
    )
    value_new = torch.randn_like(key_new)
    cache_seqlens = torch.tensor([cache_len - new_len], dtype=torch.int32, device="npu")

    with pytest.raises(RuntimeError, match="append-KV requires contiguous dense K/V caches"):
        flash_attn_with_kvcache(
            query,
            key_cache,
            value_cache,
            k=key_new,
            v=value_new,
            cache_seqlens=cache_seqlens,
            num_splits=1,
        )


@pytest.mark.parametrize(
    "targets",
    (("key",), ("value",), ("key", "value")),
    ids=("key", "value", "all-caches"),
)
def test_fa3_paged_inner_noncontiguous_falls_back_to_contiguous(targets):
    """Match FA4/FIA: unsupported paged inner strides are materialized."""
    page_size, num_heads, kv_heads, head_size = 128, 8, 2, 128
    kv_length = 129
    num_pages = (kv_length + page_size - 1) // page_size
    query = torch.randn((1, 1, num_heads, head_size), dtype=torch.float16, device="npu")
    key = torch.randn(
        (num_pages, page_size, kv_heads, head_size), dtype=torch.float16, device="npu"
    )
    value = torch.randn_like(key)
    key_view = _fa3_inner_strided(key) if "key" in targets else key
    value_view = _fa3_inner_strided(value) if "value" in targets else value
    cache_seqlens = torch.tensor([kv_length], dtype=torch.int32, device="npu")
    page_table = torch.arange(num_pages, dtype=torch.int32, device="npu").reshape(1, num_pages)
    kwargs = dict(
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        num_splits=1,
        return_softmax_lse=True,
    )

    out_view, lse_view, *_ = flash_attn_with_kvcache(query, key_view, value_view, **kwargs)
    out_ref, lse_ref, *_ = flash_attn_with_kvcache(query, key, value, **kwargs)
    torch.testing.assert_close(out_view, out_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(lse_view, lse_ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "nq,nk,window",
    [(8, 4, (4, 4)), (8, 4, (1, 6)), (8, 4, (1, -1)), (257, 65, (65, 65))],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout,unused", [
    ("TND", "none"), ("TND", "kv"), ("TND", "qkv"), ("BSND", "none"),
])
def test_swa_q_longer_than_k(nq, nk, window, dtype, unused, layout):
    module = importlib.import_module("flash_attn_npu_3")
    torch.manual_seed(2026)
    q_lengths = [nq, nq - 1 if layout == "TND" else nq]
    k_lengths = [nk, nk - 1 if layout == "TND" else nk]
    q_alloc = [n + 2 * (unused == "qkv") for n in q_lengths]
    k_alloc = [n + 2 * (unused != "none") for n in k_lengths]
    q = torch.randn(sum(q_alloc), 2, 64, dtype=dtype).npu().requires_grad_()
    k = torch.randn(sum(k_alloc), 2, 64, dtype=dtype).npu().requires_grad_()
    v = torch.randn_like(k).requires_grad_()
    cu_q = torch.tensor([0, q_alloc[0], sum(q_alloc)], dtype=torch.int32, device="npu")
    cu_k = torch.tensor([0, k_alloc[0], sum(k_alloc)], dtype=torch.int32, device="npu")
    kwargs = dict(window_size=window, causal=False)
    if unused != "none":
        kwargs.update(
            seqused_q=(torch.tensor(q_lengths, dtype=torch.int32, device="npu")
                       if unused == "qkv" else None),
            seqused_k=torch.tensor(k_lengths, dtype=torch.int32, device="npu"),
        )
    if layout == "TND":
        out = module.flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_alloc), max_seqlen_k=max(k_alloc), **kwargs,
        )
    else:
        out = module.flash_attn_func(
            q.view(2, nq, 2, 64), k.view(2, nk, 2, 64), v.view(2, nk, 2, 64), **kwargs,
        )
    if isinstance(out, tuple):
        out = out[0]
    out = out.reshape(-1, 2, 64)

    # Independent float64 reference: do not reuse host normalization or cached Golden.
    qr, kr, vr = [x.detach().cpu().double().requires_grad_() for x in (q, k, v)]
    refs, actuals = [], []
    q_start = k_start = 0
    for nq, nk, aq, ak in zip(q_lengths, k_lengths, q_alloc, k_alloc):
        scores = torch.einsum(
            "ihd,jhd->hij", qr[q_start:q_start + nq], kr[k_start:k_start + nk]
        ) / 8
        center = torch.arange(nq, device="cpu")[:, None] + nk - nq
        col = torch.arange(nk, device="cpu")[None, :]
        allowed = torch.ones(nq, nk, dtype=torch.bool, device="cpu")
        if window[0] >= 0:
            allowed &= col >= center - window[0]
        if window[1] >= 0:
            allowed &= col <= center + window[1]
        valid = allowed.any(-1)
        scores = scores.masked_fill(~allowed, -torch.inf)
        scores = torch.where(valid[None, :, None], scores, 0)
        probs = scores.softmax(-1) * valid[None, :, None]
        refs.append(torch.einsum("hij,jhd->ihd", probs, vr[k_start:k_start + nk]))
        actuals.append(out[q_start:q_start + nq])
        q_start += aq
        k_start += ak
    expected, actual = torch.cat(refs), torch.cat(actuals)
    tol = 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual.cpu().double(), expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("physical_layout", ["BBND", "BNBD"])
@pytest.mark.parametrize("page_size", [128, 256, 512, 1024])
@pytest.mark.parametrize("causal,window", [(False, (-1, -1)), (True, (-1, -1)), (False, (4, 2))])
def test_paged_strides_reused_metadata_against_reference(physical_layout, page_size, causal, window):
    from flash_attn_npu_3 import get_scheduler_metadata

    q_lengths, kv_lengths = [3, 5], [page_size + 3, page_size - 1]
    heads, kv_heads, dq, dv = 4, 2, 64, 96
    q = torch.randn(sum(q_lengths), heads, dq, dtype=torch.float16, device="npu")
    cu_q = make_cu_seqlens(q_lengths).npu()
    lengths = torch.tensor(kv_lengths, dtype=torch.int32, device="npu")
    table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device="npu")
    k_inner = (page_size, kv_heads, dq) if physical_layout == "BBND" else (kv_heads, page_size, dq)
    v_inner = (page_size, kv_heads, dv) if physical_layout == "BBND" else (kv_heads, page_size, dv)
    k = torch.randn(8, *k_inner, dtype=q.dtype, device="npu")[::2]
    v = torch.randn(12, *v_inner, dtype=q.dtype, device="npu")[1::3]
    if physical_layout == "BNBD":
        k, v = k.transpose(1, 2), v.transpose(1, 2)
    metadata = get_scheduler_metadata(
        batch_size=2, max_seqlen_q=max(q_lengths), max_seqlen_k=2 * page_size,
        num_heads_q=heads, num_heads_kv=kv_heads, headdim=dq, headdim_v=dv,
        cache_seqlens=lengths, qkv_dtype=q.dtype, cu_seqlens_q=cu_q,
        page_size=page_size, causal=causal, window_size=window, num_splits=1,
    )
    kwargs = dict(cache_seqlens=lengths, cu_seqlens_q=cu_q, max_seqlen_q=max(q_lengths),
                  page_table=table, causal=causal, window_size=window, num_splits=1,
                  scheduler_metadata=metadata, return_softmax_lse=True)
    out, lse, *_ = flash_attn_with_kvcache(q, k, v, **kwargs)
    # Reuse exactly the same schedule with different page pitches and Q allocation.
    padded_q = torch.cat([q, torch.randn(7, heads, dq, dtype=q.dtype, device=q.device)])
    out_pad, lse_pad, *_ = flash_attn_with_kvcache(padded_q, k.contiguous(), v.contiguous(), **kwargs)
    torch.testing.assert_close(out_pad[:q.shape[0]], out, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(lse_pad[:, :q.shape[0]], lse, atol=2e-3, rtol=2e-3)

    q_cpu, k_cpu, v_cpu = [t.cpu().double() for t in (q, k, v)]
    start = 0
    for batch, (nq, nk) in enumerate(zip(q_lengths, kv_lengths)):
        ids = table[batch].cpu().long()
        keys = k_cpu[ids].reshape(-1, kv_heads, dq)[:nk].repeat_interleave(heads // kv_heads, 1)
        values = v_cpu[ids].reshape(-1, kv_heads, dv)[:nk].repeat_interleave(heads // kv_heads, 1)
        scores = torch.einsum("ihd,jhd->hij", q_cpu[start:start + nq], keys) / dq ** 0.5
        center = torch.arange(nq)[:, None] + nk - nq
        cols = torch.arange(nk)[None, :]
        allowed = torch.ones(nq, nk, dtype=torch.bool)
        if causal:
            allowed &= cols <= center
        if window[0] >= 0:
            allowed &= cols >= center - window[0]
        if window[1] >= 0:
            allowed &= cols <= center + window[1]
        scores.masked_fill_(~allowed, -torch.inf)
        expected = torch.einsum("hij,jhd->ihd", scores.softmax(-1), values)
        torch.testing.assert_close(out[start:start + nq].cpu().double(), expected, atol=3e-3, rtol=3e-3)
        torch.testing.assert_close(lse[:, start:start + nq].cpu().double(), scores.logsumexp(-1),
                                   atol=3e-3, rtol=3e-3)
        start += nq


@pytest.mark.parametrize("q_lengths", [[1], [1, 3]])
@pytest.mark.parametrize("return_lse", [False, True])
def test_split_kv_padded_tnd_storage(q_lengths, return_lse):
    batch, heads, kv_heads, dim, page_size = len(q_lengths), 8, 2, 64, 128
    pages_per_batch = 16
    pages = batch * pages_per_batch
    q = torch.randn(sum(q_lengths), heads, dim, dtype=torch.float16, device="npu")
    q_pad = torch.cat([q, torch.randn(5, heads, dim, dtype=q.dtype, device=q.device)])
    k = torch.randn(pages * 2, kv_heads, page_size, dim, dtype=q.dtype, device=q.device)[::2].transpose(1, 2)
    v = torch.randn(pages * 3, kv_heads, page_size, dim, dtype=q.dtype, device=q.device)[1::3].transpose(1, 2)
    kwargs = dict(cu_seqlens_q=make_cu_seqlens(q_lengths).npu(), max_seqlen_q=max(q_lengths),
                  cache_seqlens=torch.full((batch,), 2048, dtype=torch.int32, device=q.device),
                  page_table=torch.arange(pages, dtype=torch.int32, device=q.device).reshape(batch, -1),
                  causal=True, return_softmax_lse=return_lse)
    reference = flash_attn_with_kvcache(q, k.contiguous(), v.contiguous(), num_splits=1, **kwargs)
    actual = flash_attn_with_kvcache(q_pad, k, v, num_splits=2, **kwargs)
    if return_lse:
        out_ref, lse_ref, *_ = reference
        out, lse, *_ = actual
        torch.testing.assert_close(lse[:, :q.shape[0]], lse_ref, atol=3e-3, rtol=3e-3)
    else:
        out_ref, out = reference, actual
    torch.testing.assert_close(out[:q.shape[0]], out_ref, atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("kv_heads", [2, 4])
@pytest.mark.parametrize("extra_kv", [(5, 5), (3, 7), (7, 3)])
def test_packed_tail_backward(causal, deterministic, kv_heads, extra_kv):
    q_lengths, k_lengths = [4, 7], [8, 9]
    q = torch.randn(sum(q_lengths), 4, 64, dtype=torch.float16, device="npu", requires_grad=True)
    k = torch.randn(sum(k_lengths), kv_heads, 64, dtype=q.dtype, device=q.device, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    padded = [torch.cat([t.detach(), torch.randn(extra, *t.shape[1:], dtype=t.dtype, device=t.device)])
              .requires_grad_() for t, extra in zip((q, k, v), (5, *extra_kv))]
    kwargs = dict(cu_seqlens_q=make_cu_seqlens(q_lengths).npu(), cu_seqlens_k=make_cu_seqlens(k_lengths).npu(),
                  max_seqlen_q=max(q_lengths), max_seqlen_k=max(k_lengths), causal=causal,
                  deterministic=deterministic)
    reference = flash_attn_varlen_func(q, k, v, **kwargs)
    actual = flash_attn_varlen_func(*padded, **kwargs)
    grad = torch.randn_like(reference)
    reference.backward(grad)
    actual[:q.shape[0]].backward(grad)
    torch.testing.assert_close(actual[:q.shape[0]], reference, atol=3e-3, rtol=3e-3)
    for original, pad in zip((q, k, v), padded):
        torch.testing.assert_close(pad.grad[:original.shape[0]], original.grad, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(pad.grad[original.shape[0]:], torch.zeros_like(pad.grad[original.shape[0]:]),
                                   atol=0, rtol=0)


@pytest.mark.parametrize("short_tensor", ["k", "v"])
def test_packed_kv_extent_is_validated(short_tensor):
    q = torch.randn(3, 4, 64, dtype=torch.float16, device="npu")
    k = torch.randn(5 - (short_tensor == "k"), 2, 64, dtype=q.dtype, device=q.device)
    v = torch.randn(5 - (short_tensor == "v"), 2, 64, dtype=q.dtype, device=q.device)
    with pytest.raises(RuntimeError, match="cu_seqlens_k"):
        flash_attn_varlen_func(q, k, v, make_cu_seqlens([3]).npu(), make_cu_seqlens([5]).npu(), 3, 5)


def test_paged_cache_rejects_overlapping_pages():
    q = torch.randn(1, 4, 64, dtype=torch.float16, device="npu")
    k = torch.randn(1, 128, 2, 64, dtype=q.dtype, device=q.device).expand(2, -1, -1, -1)
    v = torch.randn(2, 128, 2, 64, dtype=q.dtype, device=q.device)
    with pytest.raises(RuntimeError, match="must not overlap physical pages"):
        flash_attn_with_kvcache(
            q, k, v, cu_seqlens_q=make_cu_seqlens([1]).npu(), max_seqlen_q=1,
            cache_seqlens=torch.tensor([129], dtype=torch.int32, device=q.device),
            page_table=torch.tensor([[0, 1]], dtype=torch.int32, device=q.device), num_splits=1,
        )


def test_paged_bnbd_packed_tail_graph_capture():
    from flash_attn_npu_3 import get_scheduler_metadata

    q = torch.randn(9, 4, 64, dtype=torch.float16, device="npu")
    k = torch.randn(8, 2, 128, 64, dtype=q.dtype, device=q.device)[::2].transpose(1, 2)
    v = torch.randn(12, 2, 128, 64, dtype=q.dtype, device=q.device)[1::3].transpose(1, 2)
    cu_q = make_cu_seqlens([2, 4]).npu()
    lengths = torch.tensor([129, 256], dtype=torch.int32, device=q.device)
    table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device=q.device)
    metadata = get_scheduler_metadata(
        batch_size=2, max_seqlen_q=4, max_seqlen_k=256, num_heads_q=4, num_heads_kv=2,
        headdim=64, headdim_v=64, cache_seqlens=lengths, qkv_dtype=q.dtype,
        cu_seqlens_q=cu_q, page_size=128, causal=True, num_splits=1,
    )
    kwargs = dict(cache_seqlens=lengths, cu_seqlens_q=cu_q, max_seqlen_q=4, page_table=table,
                  scheduler_metadata=metadata, causal=True, num_splits=1, return_softmax_lse=True)
    reference, lse_ref, *_ = flash_attn_with_kvcache(q, k, v, **kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out, lse, *_ = flash_attn_with_kvcache(q, k, v, **kwargs)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(out[:6], reference[:6], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(lse[:, :6], lse_ref[:, :6], atol=2e-3, rtol=2e-3)
