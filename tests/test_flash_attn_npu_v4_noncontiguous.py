# Copyright (c) 2026, Minghua Shen.

"""FA4 ports of the non-contiguous KV cases from ops-transformer.

Sources:
  * fused_infer_attention_score/tests/pytest/test_fia_pa_noncontiguous.py
  * fused_infer_attention_score/tests/pytest/test_fia_mla_d512_pa_dim0_noncontiguous.py
  * flash_attn/tests/pytests/test_cases/functional_nc.py
  * flash_attn/tests/assets/testcase/flash_attn_nc.csv

flash_attention_score has no dedicated non-contiguous cases (its pytest path
forces ``.contiguous()`` before the NPU call).

FA4 paged KV is a logical ``(num_pages, page_size, nheads_k, headdim)`` view.
BBND and BNBD inner layouts are passed to the kernel without a copy; dimension
0 may have arbitrary padding. Unsupported paged inner strides and non-paged
QKV views are materialized to packed BBND/BSND, matching FIA's OpAPI policy.
NZ cache, MLA split-RoPE, and FIA's 3-D BnBsH physical layout are not FA4 APIs,
so those source cases are dropped or rewritten as 4-D BBND.
"""

import pytest
import torch
import torch_npu

from flash_attn_npu_4 import flash_attn_func, flash_attn_varlen_func
from tests.common.test_utils import make_cu_seqlens

_NPU_910 = torch_npu.npu.is_available() and "Ascend910" in torch_npu.npu.get_device_name()
skip_unless_910 = pytest.mark.skipif(not _NPU_910, reason="Ascend910 only")


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
    """logical BBND cache -> FA4 BBND/BNBD view, optionally non-contiguous."""
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


def _inner_strided(source):
    """Preserve logical values while adding padding after the last dimension."""
    padded = torch.zeros(
        *source.shape[:-1], source.shape[-1] + 1, dtype=source.dtype, device=source.device
    )
    view = padded[..., : source.shape[-1]]
    view.copy_(source)
    assert view.stride(-1) == 1
    assert not view.is_contiguous()
    return view


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
        seqused_k=torch.tensor(kv_lens, dtype=torch.int32, device="npu"),
        page_table=page_table,
        causal=causal,
        num_splits=num_splits,
        return_lse=True,
    )
    if query.dim() == 3:
        kwargs.update(
            cu_seqlens_q=make_cu_seqlens(q_lens).npu(),
            max_seqlen_q=max(q_lens),
        )
    torch.npu.synchronize()
    out, lse = flash_attn_varlen_func(query, key, value, **kwargs)
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
    """FIA D512 MLA dim0 case, rewritten without split-RoPE (FA4 has no rope inputs)."""
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
    "targets",
    (("query",), ("key",), ("value",), ("query", "key", "value")),
    ids=("query", "key", "value", "all-qkv"),
)
def test_dense_outer_noncontiguous_qkv_falls_back(targets):
    """Dense FA4 uses packed strides, so outer-strided QKV must be copied."""
    batch, q_seqlen, kv_seqlen = 2, 16, 32
    nheads, kv_heads, head_size = 4, 2, 64
    query = torch.randn(
        batch, q_seqlen, nheads, head_size, dtype=torch.float16, device="npu"
    )
    key = torch.randn(
        batch, kv_seqlen, kv_heads, head_size, dtype=torch.float16, device="npu"
    )
    value = torch.randn_like(key)
    query_view = _inner_strided(query) if "query" in targets else query
    key_view = _inner_strided(key) if "key" in targets else key
    value_view = _inner_strided(value) if "value" in targets else value

    out_nc, lse_nc = flash_attn_func(
        query_view, key_view, value_view, num_splits=1, return_lse=True
    )
    out_ref, lse_ref = flash_attn_func(
        query, key, value, num_splits=1, return_lse=True
    )
    torch.testing.assert_close(out_nc, out_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(lse_nc, lse_ref, rtol=0.0, atol=0.0)
