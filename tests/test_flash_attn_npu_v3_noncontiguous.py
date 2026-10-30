# Copyright (c) 2026, Minghua Shen.

"""FA3 ports of the non-contiguous KV cases.

FA3 paged KV goes through ``flash_attn_with_kvcache``. Packed TND tails,
dim0-strided BBND/BNBD, inner-axis fallback, FIA PA grids, and append-KV
cache checks live here. FA4 counterparts stay in
``test_flash_attn_npu_v4_noncontiguous.py``. Learnable-sink cases are FA4-only
and are not ported.
"""

import pytest
import torch
import torch_npu

from flash_attn_npu_3 import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from tests.common.test_utils import make_cu_seqlens

_NPU_910 = torch_npu.npu.is_available() and "Ascend910" in torch_npu.npu.get_device_name()
skip_unless_910 = pytest.mark.skipif(not _NPU_910, reason="Ascend910 only")


def _inner_strided(source):
    """Keep logical values and the last-dimension stride while padding every row."""
    padded = torch.zeros(
        *source.shape[:-1], source.shape[-1] + 1, dtype=source.dtype, device=source.device
    )
    view = padded[..., : source.shape[-1]]
    view.copy_(source)
    assert view.stride(-1) == 1 and not view.is_contiguous()
    return view


def _contiguous_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def make_noncontiguous(tensor, nc_dims):
    """Match flash_attn/tests/pytests/utils/data.py::make_noncontiguous.

    ``0`` pads dim 1 (dim0-only). ``(0, 1)`` pads dim 2 (dim0+dim1).
    """
    if nc_dims is None:
        return tensor
    pad_dim = 1 if nc_dims == 0 else 2
    shape = list(tensor.shape)
    shape[pad_dim] += 1
    padded = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    slices = [slice(None)] * tensor.ndim
    slices[pad_dim] = slice(None, tensor.shape[pad_dim])
    padded[tuple(slices)] = tensor
    return padded[tuple(slices)]


def _packed_physical(logical_bbnd, layout):
    """Allocate a packed BBND or BNBD buffer.

    ``transpose(1, 2).contiguous()`` is a no-op when nheads_k == 1, so BNBD
    must be copied into a freshly allocated ``(pages, heads, page, dim)``
    tensor to keep page/head strides distinguishable.
    """
    if layout == "BBND":
        return logical_bbnd.contiguous()
    pages, page, heads, dim = logical_bbnd.shape
    physical = torch.empty(
        pages, heads, page, dim, dtype=logical_bbnd.dtype, device=logical_bbnd.device
    )
    physical.copy_(logical_bbnd.transpose(1, 2))
    return physical


def _paged_view(logical_bbnd, layout, nc_dims):
    """logical BBND cache -> FA3 BBND/BNBD view, optionally non-contiguous."""
    physical = _packed_physical(logical_bbnd, layout)
    if nc_dims is not None:
        physical = make_noncontiguous(physical, nc_dims)
    return physical if layout == "BBND" else physical.transpose(1, 2)


def _assert_layout(view, layout):
    page, heads, dim = view.shape[1], view.shape[2], view.shape[3]
    if layout == "BBND":
        ok = view.stride(1) == heads * dim and view.stride(2) == dim and view.stride(3) == 1
    else:
        ok = view.stride(1) == dim and view.stride(2) == page * dim and view.stride(3) == 1
    assert ok, f"{layout} strides {view.stride()} shape {tuple(view.shape)}"


def _block_table(kv_lens, block_size, spare=2):
    counts = [(length + block_size - 1) // block_size for length in kv_lens]
    required = sum(counts)
    physical = required + spare
    ids = torch.randperm(physical)
    width = max(counts)
    table = torch.zeros((len(kv_lens), width), dtype=torch.int32)
    cursor = 0
    for row, count in enumerate(counts):
        table[row, :count] = ids[cursor : cursor + count].to(torch.int32)
        cursor += count
    return table, physical


def _dim0_view(source, gap_elements, poison):
    """FIA D512-style dim0 view: poison the unused gap between physical pages."""
    assert source.is_contiguous()
    assert source.shape[0] > 1
    block_num = source.shape[0]
    logical_block = source[0].numel()
    stride0 = logical_block + gap_elements
    storage = torch.full(
        ((block_num - 1) * stride0 + logical_block,),
        poison,
        dtype=source.dtype,
        device=source.device,
    )
    view = torch.as_strided(storage, size=tuple(source.shape), stride=(stride0, *source.stride()[1:]))
    view.copy_(source)
    assert not view.is_contiguous()
    return view, storage, logical_block, stride0, poison


def _assert_poison_untouched(storage, logical_block, stride0, poison, block_num=None):
    if block_num is None:
        block_num = (storage.numel() - logical_block) // stride0 + 1
    for block_index in range(block_num - 1):
        gap_start = block_index * stride0 + logical_block
        gap_end = (block_index + 1) * stride0
        gap = storage[gap_start:gap_end]
        assert torch.equal(gap, torch.full_like(gap, poison)), (
            f"padding after physical block {block_index} was modified"
        )


def _run_paged(query, key, value, *, page_table, kv_lens, q_lens, causal=False, num_splits=1):
    kwargs = dict(
        cache_seqlens=torch.tensor(kv_lens, dtype=torch.int32, device="npu"),
        page_table=page_table,
        causal=causal,
        num_splits=num_splits,
        return_softmax_lse=True,
    )
    if query.dim() == 3:
        kwargs.update(
            cu_seqlens_q=make_cu_seqlens(q_lens).npu(),
            max_seqlen_q=max(q_lens) if q_lens else 0,
        )
    torch.npu.synchronize()
    out, lse, *_ = flash_attn_with_kvcache(query, key, value, **kwargs)
    torch.npu.synchronize()
    return out, lse


# ==============================================================================
# flash_attn functional_nc.py / flash_attn_nc.csv  +  FIA PA noncontiguous
# ==============================================================================
PA_CASES = [
    # flash_attn: PA_BBND dim0
    {
        "id": "NC_PA_BBND_dim0",
        "B": 2,
        "N_q": 8,
        "N_kv": 1,
        "q_seq": 64,
        "kv_seq": 1024,
        "d": 128,
        "dtype": torch.float16,
        "ql": "BSND",
        "layout": "BBND",
        "nc": 0,
        "bs": 128,
    },
    {
        "id": "NC_PA_BBND_D64_dim0",
        "B": 2,
        "N_q": 8,
        "N_kv": 1,
        "q_seq": 64,
        "kv_seq": 1024,
        "d": 64,
        "dtype": torch.float16,
        "ql": "BSND",
        "layout": "BBND",
        "nc": 0,
        "bs": 128,
    },
    # flash_attn: PA_BNBD dim0 / dtype / D / block_size variants
    {
        "id": "NC_PA_BNBD_dim0",
        "B": 2,
        "N_q": 8,
        "N_kv": 1,
        "q_seq": 64,
        "kv_seq": 1024,
        "d": 128,
        "dtype": torch.float16,
        "ql": "BSND",
        "layout": "BNBD",
        "nc": 0,
        "bs": 128,
    },
    {
        "id": "NC_PA_BNBD_D64_dim0",
        "B": 2,
        "N_q": 8,
        "N_kv": 1,
        "q_seq": 64,
        "kv_seq": 1024,
        "d": 64,
        "dtype": torch.float16,
        "ql": "BSND",
        "layout": "BNBD",
        "nc": 0,
        "bs": 128,
    },
    {
        "id": "NC_PA_BNBD_bf16_dim0",
        "B": 2,
        "N_q": 8,
        "N_kv": 1,
        "q_seq": 64,
        "kv_seq": 1024,
        "d": 128,
        "dtype": torch.bfloat16,
        "ql": "BSND",
        "layout": "BNBD",
        "nc": 0,
        "bs": 128,
    },
    {
        "id": "NC_PA_BNBD_bs256_dim0",
        "B": 2,
        "N_q": 8,
        "N_kv": 1,
        "q_seq": 64,
        "kv_seq": 1024,
        "d": 128,
        "dtype": torch.float16,
        "ql": "BSND",
        "layout": "BNBD",
        "nc": 0,
        "bs": 256,
    },
    # FIA NEW-BnNBsD-dim0 (TND + GQA + varlen q)
    {
        "id": "NEW-BnNBsD-dim0",
        "B": 2,
        "N_q": 2,
        "N_kv": 1,
        "q_seq": 6,
        "kv_seq": 1536,
        "d": 128,
        "dtype": torch.float16,
        "ql": "TND",
        "layout": "BNBD",
        "nc": 0,
        "bs": 256,
        "act_q": [3, 6],
    },
    # FIA NEW-BnBsH-dim0 rewritten as 4-D BBND
    {
        "id": "NEW-BnBsH-dim0",
        "B": 3,
        "N_q": 6,
        "N_kv": 2,
        "q_seq": 4,
        "kv_seq": 2304,
        "d": 128,
        "dtype": torch.float16,
        "ql": "TND",
        "layout": "BBND",
        "nc": 0,
        "bs": 384,
        "act_q": [2, 4, 3],
    },
    # FIA NEW-d192-dim0
    {
        "id": "NEW-d192-dim0",
        "B": 1,
        "N_q": 2,
        "N_kv": 1,
        "q_seq": 4,
        "kv_seq": 1536,
        "d": 192,
        "dtype": torch.float16,
        "ql": "TND",
        "layout": "BNBD",
        "nc": 0,
        "bs": 384,
    },
    # FIA OLD-d256-dim0
    {
        "id": "OLD-d256-dim0",
        "B": 2,
        "N_q": 4,
        "N_kv": 2,
        "q_seq": 8,
        "kv_seq": 1024,
        "d": 256,
        "dtype": torch.float16,
        "ql": "BSND",
        "layout": "BNBD",
        "nc": 0,
        "bs": 128,
    },
    # FIA OLD-d64BnBsH-dim0 rewritten as 4-D BBND
    {
        "id": "OLD-d64BnBsH-dim0",
        "B": 1,
        "N_q": 4,
        "N_kv": 2,
        "q_seq": 4,
        "kv_seq": 1536,
        "d": 64,
        "dtype": torch.float16,
        "ql": "BSND",
        "layout": "BBND",
        "nc": 0,
        "bs": 256,
    },
]


FALLBACK_CASES = [
    # flash_attn NC_REJ_PA_BBND_dim01 / FIA BnBsH dim01 fallback
    {
        "id": "NC_REJ_PA_BBND_dim01",
        "B": 2,
        "N_q": 8,
        "N_kv": 1,
        "q_seq": 64,
        "kv_seq": 1024,
        "d": 128,
        "dtype": torch.float16,
        "ql": "BSND",
        "layout": "BBND",
        "nc": (0, 1),
        "bs": 128,
    },
    # flash_attn NC_PA_BNBD_dim01 is legal in CANN FA; FA4 requires dense inner strides
    {
        "id": "NC_REJ_PA_BNBD_dim01",
        "B": 2,
        "N_q": 8,
        "N_kv": 1,
        "q_seq": 64,
        "kv_seq": 1024,
        "d": 128,
        "dtype": torch.float16,
        "ql": "BSND",
        "layout": "BNBD",
        "nc": (0, 1),
        "bs": 128,
    },
    {
        "id": "NC_REJ_PA_BNBD_D64_dim01",
        "B": 2,
        "N_q": 8,
        "N_kv": 1,
        "q_seq": 64,
        "kv_seq": 1024,
        "d": 64,
        "dtype": torch.float16,
        "ql": "BSND",
        "layout": "BNBD",
        "nc": (0, 1),
        "bs": 128,
    },
    {
        "id": "NC_REJ_PA_BNBD_bs256_dim01",
        "B": 2,
        "N_q": 8,
        "N_kv": 1,
        "q_seq": 64,
        "kv_seq": 1024,
        "d": 128,
        "dtype": torch.float16,
        "ql": "BSND",
        "layout": "BNBD",
        "nc": (0, 1),
        "bs": 256,
    },
    # FIA NEW-BnNBsD-dim01
    {
        "id": "NEW-BnNBsD-dim01",
        "B": 1,
        "N_q": 4,
        "N_kv": 2,
        "q_seq": 4,
        "kv_seq": 2048,
        "d": 128,
        "dtype": torch.float16,
        "ql": "BSND",
        "layout": "BNBD",
        "nc": (0, 1),
        "bs": 512,
    },
    # FIA OLD-d256-dim01
    {
        "id": "OLD-d256-dim01",
        "B": 1,
        "N_q": 4,
        "N_kv": 2,
        "q_seq": 8,
        "kv_seq": 1024,
        "d": 256,
        "dtype": torch.float16,
        "ql": "BSND",
        "layout": "BNBD",
        "nc": (0, 1),
        "bs": 128,
    },
]


def _prepare(cfg):
    batch = cfg["B"]
    q_lens = cfg.get("act_q", [cfg["q_seq"]] * batch)
    kv_lens = cfg.get("act_kv", [cfg["kv_seq"]] * batch)
    page_table, num_pages = _block_table(kv_lens, cfg["bs"])
    dtype = cfg["dtype"]
    if cfg["ql"] == "TND":
        query = torch.randn(sum(q_lens), cfg["N_q"], cfg["d"], dtype=dtype, device="npu")
    else:
        query = torch.randn(batch, cfg["q_seq"], cfg["N_q"], cfg["d"], dtype=dtype, device="npu")
    key = torch.randn(num_pages, cfg["bs"], cfg["N_kv"], cfg["d"], dtype=dtype, device="npu")
    value = torch.randn_like(key)
    key_view = _paged_view(key, cfg["layout"], cfg["nc"])
    value_view = _paged_view(value, cfg["layout"], cfg["nc"])
    return query, key, value, key_view, value_view, page_table.npu(), q_lens, kv_lens


@skip_unless_910
@pytest.mark.parametrize("cfg", PA_CASES, ids=[c["id"] for c in PA_CASES])
def test_paged_noncontiguous_matches_contiguous(cfg):
    query, key, value, key_view, value_view, page_table, q_lens, kv_lens = _prepare(cfg)
    expected = _contiguous_strides(tuple(key_view.shape))
    assert key_view.stride(0) != expected[0], f"{cfg['id']}: expected dim0 non-contiguous"
    assert not key_view.is_contiguous()
    _assert_layout(key_view, cfg["layout"])
    _assert_layout(value_view, cfg["layout"])

    out_nc, lse_nc = _run_paged(
        query, key_view, value_view, page_table=page_table, kv_lens=kv_lens, q_lens=q_lens
    )
    out_ref, lse_ref = _run_paged(
        query, key, value, page_table=page_table, kv_lens=kv_lens, q_lens=q_lens
    )
    torch.testing.assert_close(out_nc, out_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(lse_nc, lse_ref, rtol=0.0, atol=0.0)


@skip_unless_910
@pytest.mark.parametrize("cfg", FALLBACK_CASES, ids=[c["id"] for c in FALLBACK_CASES])
def test_paged_inner_noncontiguous_falls_back_to_contiguous(cfg):
    query, key, value, key_view, value_view, page_table, q_lens, kv_lens = _prepare(cfg)
    assert not key_view.is_contiguous()
    out_nc, lse_nc = _run_paged(
        query, key_view, value_view, page_table=page_table, kv_lens=kv_lens, q_lens=q_lens
    )
    out_ref, lse_ref = _run_paged(
        query, key, value, page_table=page_table, kv_lens=kv_lens, q_lens=q_lens
    )
    torch.testing.assert_close(out_nc, out_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(lse_nc, lse_ref, rtol=0.0, atol=0.0)


# FIA test_fia_mla_d512_pa_dim0_noncontiguous.py STRIDE_PROFILES, without split-RoPE.
STRIDE_PROFILES = (
    ("key-only", 17, 0),
    ("value-only", 0, 29),
    ("key-value-same", 23, 23),
    ("key-greater-than-value", 61, 13),
    ("value-greater-than-key", 13, 61),
    ("all-different", 17, 37),
)


@skip_unless_910
@pytest.mark.parametrize("layout", ["BBND", "BNBD"])
@pytest.mark.parametrize("profile", STRIDE_PROFILES, ids=[p[0] for p in STRIDE_PROFILES])
def test_paged_independent_dim0_stride_profiles(layout, profile):
    _, key_gap, value_gap = profile
    nheads, kv_heads, head_size, page_size = 8, 2, 128, 128
    q_lens = (1, 2)
    kv_lens = (127, 129)
    page_table, num_pages = _block_table(kv_lens, page_size, spare=3)
    query = torch.randn(sum(q_lens), nheads, head_size, dtype=torch.float16, device="npu")
    key = torch.randn(num_pages, page_size, kv_heads, head_size, dtype=torch.float16, device="npu")
    value = torch.randn_like(key)
    page_table_npu = page_table.npu()

    def maybe_gap(cache, gap, poison):
        physical = _packed_physical(cache, layout)
        to_logical = (lambda tensor: tensor) if layout == "BBND" else (lambda tensor: tensor.transpose(1, 2))
        if gap == 0:
            return to_logical(physical), None
        view, storage, logical_block, stride0, poison_val = _dim0_view(physical, gap, poison)
        return to_logical(view), (storage, logical_block, stride0, poison_val)

    key_view, key_meta = maybe_gap(key, key_gap, 37.0)
    value_view, value_meta = maybe_gap(value, value_gap, -29.0)
    _assert_layout(key_view, layout)
    _assert_layout(value_view, layout)
    if key_gap or value_gap:
        assert key_view.stride(0) != value_view.stride(0) or key_gap == value_gap

    out_nc, lse_nc = _run_paged(
        query, key_view, value_view, page_table=page_table_npu, kv_lens=kv_lens, q_lens=q_lens
    )
    out_ref, lse_ref = _run_paged(
        query, key, value, page_table=page_table_npu, kv_lens=kv_lens, q_lens=q_lens
    )
    torch.testing.assert_close(out_nc, out_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(lse_nc, lse_ref, rtol=0.0, atol=0.0)
    for meta in (key_meta, value_meta):
        if meta is not None:
            _assert_poison_untouched(*meta)


@skip_unless_910
def test_paged_d512_dim0_noncontiguous():
    """FIA D512 MLA dim0 case, rewritten without split-RoPE."""
    name = torch_npu.npu.get_device_name()
    if "Ascend950" in name:
        pytest.skip("head_dim=512 is supported only on Ascend910")
    q_lens = (1,)
    kv_lens = (129,)
    page_size, nheads, kv_heads, head_size = 128, 8, 1, 512
    page_table, num_pages = _block_table(kv_lens, page_size, spare=3)
    query = torch.randn(sum(q_lens), nheads, head_size, dtype=torch.float16, device="npu")
    key = torch.randn(num_pages, page_size, kv_heads, head_size, dtype=torch.float16, device="npu")
    value = torch.randn_like(key)
    key_view, *key_meta = _dim0_view(key, 17, 37.0)
    value_view, *value_meta = _dim0_view(value, 37, -29.0)
    page_table_npu = page_table.npu()
    out_nc, lse_nc = _run_paged(
        query, key_view, value_view, page_table=page_table_npu, kv_lens=kv_lens, q_lens=q_lens
    )
    out_ref, lse_ref = _run_paged(
        query, key, value, page_table=page_table_npu, kv_lens=kv_lens, q_lens=q_lens
    )
    torch.testing.assert_close(out_nc, out_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(lse_nc, lse_ref, rtol=0.0, atol=0.0)
    _assert_poison_untouched(*key_meta)
    _assert_poison_untouched(*value_meta)


@skip_unless_910
@pytest.mark.parametrize(
    "targets",
    (("key",), ("value",), ("key", "value")),
    ids=("key", "value", "all-caches"),
)
def test_inner_axis_noncontiguous_cache_falls_back(targets):
    """FIA inner-axis fallback: materialize unsupported paged cache views."""
    q_lens = (1,)
    kv_lens = (129,)
    page_size, nheads, kv_heads, head_size = 128, 8, 2, 128
    page_table, num_pages = _block_table(kv_lens, page_size)
    query = torch.randn(sum(q_lens), nheads, head_size, dtype=torch.float16, device="npu")
    key = torch.randn(num_pages, page_size, kv_heads, head_size, dtype=torch.float16, device="npu")
    value = torch.randn_like(key)

    key_view = _inner_strided(key) if "key" in targets else key
    value_view = _inner_strided(value) if "value" in targets else value
    page_table_npu = page_table.npu()
    out_nc, lse_nc = _run_paged(
        query,
        key_view,
        value_view,
        page_table=page_table_npu,
        kv_lens=kv_lens,
        q_lens=q_lens,
    )
    out_ref, lse_ref = _run_paged(
        query,
        key,
        value,
        page_table=page_table_npu,
        kv_lens=kv_lens,
        q_lens=q_lens,
    )
    torch.testing.assert_close(out_nc, out_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(lse_nc, lse_ref, rtol=0.0, atol=0.0)


@skip_unless_910
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

    assert out_pad.shape[0] == used_q + extra_q
    assert lse_pad.shape == (num_heads, used_q + extra_q)
    torch.testing.assert_close(out_pad[:used_q], out_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(lse_pad[:, :used_q], lse_ref, rtol=2e-3, atol=2e-3, equal_nan=True)


@skip_unless_910
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


@skip_unless_910
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("physical_layout", ["BBND", "BNBD"])
def test_paged_tnd_dim0_noncontiguous_with_zero_lengths(physical_layout, causal, dtype):
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
        num_splits=1,
        return_softmax_lse=True,
    )
    out_view, lse_view, *_ = flash_attn_with_kvcache(query, key_view, value_view, **kwargs)
    out_dense, lse_dense, *_ = flash_attn_with_kvcache(
        query, key_view.contiguous(), value_view.contiguous(), **kwargs
    )

    torch.testing.assert_close(out_view, out_dense, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(lse_view, lse_dense, rtol=2e-3, atol=2e-3, equal_nan=True)
    torch.testing.assert_close(out_view[:3], torch.zeros_like(out_view[:3]), rtol=0, atol=0)
    assert torch.isposinf(lse_view[:, :3]).all()


@skip_unless_910
@pytest.mark.parametrize("head_size,kv_length,num_splits", [(512, 129, 1), (64, 2048, 2)])
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


@skip_unless_910
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


@skip_unless_910
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
    key_view = _inner_strided(key) if "key" in targets else key
    value_view = _inner_strided(value) if "value" in targets else value
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


@skip_unless_910
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
    query_view = _inner_strided(query) if "query" in targets else query
    key_view = _inner_strided(key) if "key" in targets else key
    value_view = _inner_strided(value) if "value" in targets else value

    out_view, lse_view = flash_attn_func(
        query_view, key_view, value_view, num_splits=1, return_attn_probs=True
    )
    out_ref, lse_ref = flash_attn_func(
        query, key, value, num_splits=1, return_attn_probs=True
    )
    torch.testing.assert_close(out_view, out_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(lse_view, lse_ref, rtol=0.0, atol=0.0)


@skip_unless_910
def test_fa3_append_kv_rejects_noncontiguous_dense_cache():
    """A cache fallback must not hide append-KV's in-place update target."""
    batch, q_seqlen, cache_len, new_len = 1, 1, 16, 1
    num_heads, kv_heads, head_size = 4, 2, 64
    query = torch.randn(
        (batch, q_seqlen, num_heads, head_size), dtype=torch.float16, device="npu"
    )
    key_cache = _inner_strided(
        torch.randn((batch, cache_len, kv_heads, head_size), dtype=torch.float16, device="npu")
    )
    value_cache = _inner_strided(torch.randn_like(key_cache.contiguous()))
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
