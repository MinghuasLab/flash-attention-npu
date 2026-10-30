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

FA3 ``flash_attn_with_kvcache`` cases live in
``test_flash_attn_npu_v3_noncontiguous.py``.
"""

import pytest
import torch
import torch_npu

from flash_attn_npu_4 import flash_attn_func, flash_attn_varlen_func
from tests.common.attention_ref import ref_flash_attention_pair
from tests.common.compare import assert_fa_close
from tests.common.test_utils import (
    make_cu_seqlens,
    make_packed_random_tensor,
    make_random_tensor,
)

_NPU_910 = torch_npu.npu.is_available() and "Ascend910" in torch_npu.npu.get_device_name()
skip_unless_910 = pytest.mark.skipif(not _NPU_910, reason="Ascend910 only")


def _nheads_kv(mha_type, nheads):
    if mha_type == "mha":
        return nheads
    if mha_type == "gqa":
        return 3
    return 1


def _sink_matrix(sink, batch_size, nheads, q_seqlen):
    if sink is None:
        return None
    return sink.view(1, nheads, 1, 1).expand(batch_size, nheads, q_seqlen, 1).contiguous()


def _assert_causal_empty_prefix(out, lse, sink, seqlen_q, seqlen_k):
    """Causal q>kv: empty prefix O is 0, LSE is sink (or +inf without sink)."""
    empty = max(0, seqlen_q - seqlen_k)
    if empty == 0:
        return
    empty_out = out[:, :empty]
    empty_lse = lse[:, :, :empty]
    assert torch.allclose(
        empty_out.float(), torch.zeros_like(empty_out.float()), atol=2e-3
    ), (
        f"empty-prefix O not 0: max={empty_out.float().abs().max().item()} "
        f"prefix={empty}"
    )
    if sink is None:
        assert torch.isposinf(empty_lse).all(), (
            f"empty-prefix LSE without sink must be +inf, "
            f"got finite={(~torch.isinf(empty_lse)).any().item()} "
            f"neg_inf={torch.isneginf(empty_lse).any().item()}"
        )
        return
    expected = sink.detach().float().view(1, -1, 1).expand_as(empty_lse)
    max_diff = (empty_lse.float() - expected).abs().max().item()
    assert max_diff <= 1e-4, (
        f"empty-prefix LSE must equal sink: max|Δ|={max_diff} prefix={empty}"
    )


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

@pytest.mark.golden_api("flash_attn_func")
@skip_unless_910
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("has_learnable_sink", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (384, 256),
        (640, 128),
    ],
)
def test_flash_attn_sink(
    seqlen_q,
    seqlen_k,
    d,
    causal,
    has_learnable_sink,
    mha_type,
    dtype,
):
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if "Ascend910" not in name and "Ascend950" not in name:
        pytest.skip("flash_attn_func only supports Ascend910/Ascend950")
    torch.manual_seed(0)
    batch_size = 2
    nheads = 6
    nheads_kv = _nheads_kv(mha_type, nheads)
    query = make_random_tensor(
        (batch_size, seqlen_q, nheads, d), dtype, device="npu"
    )
    key = make_random_tensor(
        (batch_size, seqlen_k, nheads_kv, d), dtype, device="npu"
    )
    value = make_random_tensor(
        (batch_size, seqlen_k, nheads_kv, d), dtype, device="npu"
    )
    sink = (
        make_random_tensor((nheads,), dtype, device="npu")
        if has_learnable_sink
        else None
    )
    scale = d ** -0.5
    atten_mask = None
    if causal:
        atten_mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k),
            diagonal=(seqlen_k - seqlen_q + 1),
        ).bool()
    query_ref = query.detach().cpu()
    key_ref = key.detach().cpu()
    value_ref = value.detach().cpu()
    sink_ref = None if sink is None else sink.detach().cpu()
    sink_matrix = _sink_matrix(sink_ref, batch_size, nheads, seqlen_q)
    rescale = None if has_learnable_sink else 4.0
    out_ref, lse_ref, out_pt, lse_pt = ref_flash_attention_pair(
        query_ref,
        key_ref,
        value_ref,
        scale,
        atten_mask,
        dtype,
        softcap=0.0,
        rescale_threshold=rescale,
        sink_matrix=sink_matrix,
    )
    if atten_mask is not None:
        fully_masked = atten_mask.all(dim=-1)
        out_ref = out_ref.clone()
        out_pt = out_pt.clone()
        out_ref[:, fully_masked] = 0
        out_pt[:, fully_masked] = 0
        if not has_learnable_sink:
            lse_ref = lse_ref.clone()
            lse_pt = lse_pt.clone()
            lse_ref[:, :, fully_masked] = torch.inf
            lse_pt[:, :, fully_masked] = torch.inf

    # GPU test_flash_attn_output uses num_splits in {1, 3} on the split-KV path.
    for num_splits in (1, 3):
        out, lse = flash_attn_func(
            query,
            key,
            value,
            softmax_scale=scale,
            causal=causal,
            window_size=(-1, -1),
            learnable_sink=sink,
            num_splits=num_splits,
            return_lse=True,
        )
        assert_fa_close(out, out_ref, out_pt, name=f"out splits={num_splits}")
        assert_fa_close(lse, lse_ref, lse_pt, name=f"softmax_lse splits={num_splits}")
        if causal:
            _assert_causal_empty_prefix(out, lse, sink, seqlen_q, seqlen_k)


# Subset of tests/test_flash_attn_npu_v4_generalize.py sink cases: dense BSND,
# packed TND, and paged BBND/BNBD, mixing short and long sequences. Softcap,
# SWA, head-dim 512, and noncontiguous paged caches are left in generalize.
_SINK_PAGE = 128
_SINK_LAYOUT_CASES = [
    # dense BSND: short
    {"id": "bsnd-s-mha", "api": "func", "layout": "BSND", "dtype": torch.bfloat16, "nheads": 6, "kv_heads": 6, "d": 64, "q": [8, 8], "kv": [8, 8], "causal": False},
    {"id": "bsnd-s-mqa", "api": "func", "layout": "BSND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 1, "d": 64, "q": [16, 16], "kv": [16, 16], "causal": False},
    {"id": "bsnd-s-gqa-causal", "api": "func", "layout": "BSND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 2, "d": 128, "q": [32, 32], "kv": [32, 32], "causal": True},
    {"id": "bsnd-decode", "api": "func", "layout": "BSND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 1, "d": 128, "q": [1], "kv": [128], "causal": True},
    # dense BSND: medium / empty prefix
    {"id": "bsnd-m-128", "api": "func", "layout": "BSND", "dtype": torch.bfloat16, "nheads": 6, "kv_heads": 6, "d": 128, "q": [128, 128], "kv": [128, 128], "causal": False},
    {"id": "bsnd-m-256-causal", "api": "func", "layout": "BSND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 4, "d": 64, "q": [256, 256], "kv": [256, 256], "causal": True},
    {"id": "bsnd-empty-prefix", "api": "func", "layout": "BSND", "dtype": torch.bfloat16, "nheads": 6, "kv_heads": 6, "d": 64, "q": [192], "kv": [64], "causal": True},
    # dense BSND: long
    {"id": "bsnd-l-decode-2k", "api": "func", "layout": "BSND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 2, "d": 128, "q": [1], "kv": [2048], "causal": True},
    {"id": "bsnd-l-decode-4k", "api": "func", "layout": "BSND", "dtype": torch.bfloat16, "nheads": 4, "kv_heads": 4, "d": 64, "q": [16], "kv": [4096], "causal": False},
    {"id": "bsnd-l-prefill-1k", "api": "func", "layout": "BSND", "dtype": torch.bfloat16, "nheads": 4, "kv_heads": 1, "d": 128, "q": [1024], "kv": [1024], "causal": True},
    {"id": "bsnd-fp16-mid", "api": "func", "layout": "BSND", "dtype": torch.float16, "nheads": 6, "kv_heads": 3, "d": 64, "q": [64, 64], "kv": [128, 128], "causal": False},
    {"id": "bsnd-fp16-decode", "api": "func", "layout": "BSND", "dtype": torch.float16, "nheads": 8, "kv_heads": 8, "d": 128, "q": [1, 1], "kv": [512, 512], "causal": True},
    {"id": "bsnd-d192", "api": "func", "layout": "BSND", "dtype": torch.bfloat16, "nheads": 4, "kv_heads": 2, "d": 192, "q": [64], "kv": [128], "causal": False},
    {"id": "bsnd-d256", "api": "func", "layout": "BSND", "dtype": torch.bfloat16, "nheads": 4, "kv_heads": 2, "d": 256, "q": [32], "kv": [128], "causal": False},
    # packed TND: short
    {"id": "tnd-s", "api": "varlen", "layout": "TND", "dtype": torch.bfloat16, "nheads": 4, "kv_heads": 2, "d": 64, "q": [4, 7, 1], "kv": [8, 9, 5], "causal": False},
    {"id": "tnd-s-gqa", "api": "varlen", "layout": "TND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 2, "d": 128, "q": [8, 16], "kv": [16, 32], "causal": False},
    {"id": "tnd-uneven-causal", "api": "varlen", "layout": "TND", "dtype": torch.bfloat16, "nheads": 6, "kv_heads": 3, "d": 64, "q": [3, 17, 9], "kv": [20, 40, 16], "causal": True},
    {"id": "tnd-tiny", "api": "varlen", "layout": "TND", "dtype": torch.bfloat16, "nheads": 4, "kv_heads": 4, "d": 64, "q": [2, 3, 4], "kv": [8, 8, 8], "causal": False},
    {"id": "tnd-decode", "api": "varlen", "layout": "TND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 1, "d": 128, "q": [1, 1, 1], "kv": [128, 256, 64], "causal": True},
    # packed TND: medium / long
    {"id": "tnd-m", "api": "varlen", "layout": "TND", "dtype": torch.bfloat16, "nheads": 6, "kv_heads": 3, "d": 64, "q": [64, 128], "kv": [128, 256], "causal": False},
    {"id": "tnd-prefill", "api": "varlen", "layout": "TND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 2, "d": 128, "q": [256], "kv": [256], "causal": True},
    {"id": "tnd-l-kv", "api": "varlen", "layout": "TND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 2, "d": 128, "q": [1, 1], "kv": [2048, 1024], "causal": True},
    {"id": "tnd-l-mix", "api": "varlen", "layout": "TND", "dtype": torch.bfloat16, "nheads": 4, "kv_heads": 2, "d": 64, "q": [8, 1, 32], "kv": [512, 2048, 128], "causal": False},
    {"id": "tnd-qgtkv-causal", "api": "varlen", "layout": "TND", "dtype": torch.bfloat16, "nheads": 6, "kv_heads": 6, "d": 64, "q": [128, 64], "kv": [32, 80], "causal": True},
    {"id": "tnd-fp16", "api": "varlen", "layout": "TND", "dtype": torch.float16, "nheads": 8, "kv_heads": 4, "d": 64, "q": [16, 24], "kv": [32, 48], "causal": False},
    {"id": "tnd-mqa", "api": "varlen", "layout": "TND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 1, "d": 128, "q": [5, 7], "kv": [64, 96], "causal": True},
    # paged BBND
    {"id": "bbnd-decode-128", "api": "paged", "layout": "BBND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 1, "d": 64, "q": [1], "kv": [128], "causal": True},
    {"id": "bbnd-decode-gqa", "api": "paged", "layout": "BBND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 2, "d": 128, "q": [1, 1], "kv": [256, 128], "causal": True},
    {"id": "bbnd-s", "api": "paged", "layout": "BBND", "dtype": torch.bfloat16, "nheads": 4, "kv_heads": 4, "d": 64, "q": [16, 16], "kv": [128, 128], "causal": False},
    {"id": "bbnd-m", "api": "paged", "layout": "BBND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 2, "d": 128, "q": [64], "kv": [256], "causal": False},
    {"id": "bbnd-prefill", "api": "paged", "layout": "BBND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 8, "d": 64, "q": [128], "kv": [128], "causal": True},
    {"id": "bbnd-causal", "api": "paged", "layout": "BBND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 2, "d": 128, "q": [32, 16], "kv": [256, 128], "causal": True},
    {"id": "bbnd-l-2k", "api": "paged", "layout": "BBND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 1, "d": 128, "q": [1], "kv": [2048], "causal": True},
    {"id": "bbnd-l-4k", "api": "paged", "layout": "BBND", "dtype": torch.bfloat16, "nheads": 4, "kv_heads": 2, "d": 64, "q": [8], "kv": [4096], "causal": False},
    {"id": "bbnd-fp16-decode", "api": "paged", "layout": "BBND", "dtype": torch.float16, "nheads": 8, "kv_heads": 2, "d": 64, "q": [1], "kv": [512], "causal": True},
    {"id": "bbnd-d192", "api": "paged", "layout": "BBND", "dtype": torch.bfloat16, "nheads": 4, "kv_heads": 1, "d": 192, "q": [4], "kv": [256], "causal": False},
    {"id": "bbnd-mqa", "api": "paged", "layout": "BBND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 1, "d": 128, "q": [1, 2], "kv": [128, 256], "causal": True},
    {"id": "bbnd-gqa-mid", "api": "paged", "layout": "BBND", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 4, "d": 64, "q": [48, 80], "kv": [192, 320], "causal": False},
    # paged BNBD
    {"id": "bnbd-decode-128", "api": "paged", "layout": "BNBD", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 2, "d": 64, "q": [1], "kv": [128], "causal": True},
    {"id": "bnbd-s", "api": "paged", "layout": "BNBD", "dtype": torch.bfloat16, "nheads": 4, "kv_heads": 2, "d": 128, "q": [8, 12], "kv": [64, 96], "causal": False},
    {"id": "bnbd-m-causal", "api": "paged", "layout": "BNBD", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 2, "d": 128, "q": [32], "kv": [256], "causal": True},
    {"id": "bnbd-prefill", "api": "paged", "layout": "BNBD", "dtype": torch.bfloat16, "nheads": 6, "kv_heads": 3, "d": 64, "q": [64], "kv": [256], "causal": False},
    {"id": "bnbd-l-2k", "api": "paged", "layout": "BNBD", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 1, "d": 128, "q": [1], "kv": [2048], "causal": True},
    {"id": "bnbd-l-4k", "api": "paged", "layout": "BNBD", "dtype": torch.bfloat16, "nheads": 4, "kv_heads": 4, "d": 64, "q": [4], "kv": [4096], "causal": False},
    {"id": "bnbd-fp16", "api": "paged", "layout": "BNBD", "dtype": torch.float16, "nheads": 8, "kv_heads": 2, "d": 64, "q": [16, 8], "kv": [128, 192], "causal": False},
    {"id": "bnbd-d256", "api": "paged", "layout": "BNBD", "dtype": torch.bfloat16, "nheads": 4, "kv_heads": 1, "d": 256, "q": [1], "kv": [256], "causal": True},
    {"id": "bnbd-gqa", "api": "paged", "layout": "BNBD", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 4, "d": 128, "q": [24, 40], "kv": [160, 288], "causal": True},
    {"id": "bnbd-uneven", "api": "paged", "layout": "BNBD", "dtype": torch.bfloat16, "nheads": 4, "kv_heads": 2, "d": 64, "q": [1, 16, 8], "kv": [256, 128, 512], "causal": False},
    {"id": "bnbd-mqa-long", "api": "paged", "layout": "BNBD", "dtype": torch.bfloat16, "nheads": 8, "kv_heads": 1, "d": 128, "q": [1, 1], "kv": [1024, 2048], "causal": True},
    {"id": "bnbd-tiny", "api": "paged", "layout": "BNBD", "dtype": torch.bfloat16, "nheads": 4, "kv_heads": 2, "d": 64, "q": [3, 5], "kv": [64, 128], "causal": False},
]


def _sink_causal_mask(qlen, kvlen):
    if qlen == 0 or kvlen == 0:
        return None
    return torch.triu(torch.ones(qlen, kvlen), diagonal=(kvlen - qlen + 1)).bool()


def _sink_reference_bsnd(query, key, value, scale, dtype, sink, causal, *, api_name=None):
    qlen, kvlen, nheads = query.shape[1], key.shape[1], query.shape[2]
    if qlen == 0:
        empty = query[:, :0]
        empty_lse = query.new_empty((query.shape[0], nheads, 0), dtype=torch.float32)
        return empty, empty_lse, empty.clone(), empty_lse.clone()
    if kvlen == 0:
        zeros = torch.zeros_like(query)
        lse = sink.detach().float().view(1, nheads, 1).expand(query.shape[0], nheads, qlen).contiguous()
        return zeros, lse, zeros.clone(), lse.clone()
    mask = _sink_causal_mask(qlen, kvlen) if causal else None
    sink_matrix = _sink_matrix(sink, query.shape[0], nheads, qlen)
    out_ref, lse_ref, out_pt, lse_pt = ref_flash_attention_pair(
        query, key, value, scale, mask, dtype, rescale_threshold=None,
        sink_matrix=sink_matrix, api_name=api_name,
    )
    if mask is not None:
        fully = mask.all(dim=-1)
        out_ref = out_ref.clone()
        out_pt = out_pt.clone()
        out_ref[:, fully] = 0
        out_pt[:, fully] = 0
    return out_ref, lse_ref, out_pt, lse_pt


def _sink_page_table(kv_lens):
    pages = [max(1, (length + _SINK_PAGE - 1) // _SINK_PAGE) for length in kv_lens]
    width = max(pages)
    table = torch.zeros((len(kv_lens), width), dtype=torch.int32)
    cursor = 0
    for row, count in enumerate(pages):
        table[row, :count] = torch.arange(cursor, cursor + count, dtype=torch.int32)
        cursor += count
    return table, cursor


def _sink_paged_view(logical, layout):
    if layout == "BBND":
        return logical
    pages, page, heads, dim = logical.shape
    physical = torch.empty(pages, heads, page, dim, dtype=logical.dtype, device=logical.device)
    physical.copy_(logical.transpose(1, 2))
    return physical.transpose(1, 2)


def _sink_gather(cache, table_row, kvlen):
    if kvlen == 0:
        return cache.new_empty((0, cache.shape[2], cache.shape[3]))
    positions = torch.arange(kvlen, device=cache.device)
    table = table_row.to(device=cache.device, dtype=torch.long)
    return cache[table[positions // _SINK_PAGE], positions % _SINK_PAGE]


def _sink_packed_reference(query, key, value, q_lens, kv_lens, scale, dtype, sink, causal,
                           table=None, api_name=None):
    pieces = []
    q_off = 0
    k_off = 0
    for row, (qlen, kvlen) in enumerate(zip(q_lens, kv_lens)):
        query_i = query[q_off : q_off + qlen]
        q_off += qlen
        if table is None:
            key_i = key[k_off : k_off + kvlen]
            value_i = value[k_off : k_off + kvlen]
            k_off += kvlen
        else:
            key_i = _sink_gather(key, table[row], kvlen)
            value_i = _sink_gather(value, table[row], kvlen)
        if qlen == 0:
            continue
        out_ref, lse_ref, out_pt, lse_pt = _sink_reference_bsnd(
            query_i.unsqueeze(0),
            key_i.unsqueeze(0),
            value_i.unsqueeze(0),
            scale,
            dtype,
            sink,
            causal,
            api_name=api_name,
        )
        pieces.append((out_ref[0], lse_ref[0], out_pt[0], lse_pt[0]))
    if not pieces:
        empty = query.new_empty((0, query.shape[1], query.shape[2]))
        empty_lse = query.new_empty((query.shape[1], 0), dtype=torch.float32)
        return empty, empty_lse, empty.clone(), empty_lse.clone()
    out_refs, lse_refs, out_pts, lse_pts = zip(*pieces)
    return (
        torch.cat(out_refs, dim=0),
        torch.cat(lse_refs, dim=-1),
        torch.cat(out_pts, dim=0),
        torch.cat(lse_pts, dim=-1),
    )


@skip_unless_910
@pytest.mark.parametrize("cfg", _SINK_LAYOUT_CASES, ids=[c["id"] for c in _SINK_LAYOUT_CASES])
def test_flash_attn_sink_layouts(cfg):
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if "Ascend950" in name:
        pytest.skip("learnable_sink is supported only on Ascend910")
    if "Ascend910" not in name:
        pytest.skip("flash_attn sink layouts require Ascend910")

    torch.manual_seed(0)
    q_lens, kv_lens = cfg["q"], cfg["kv"]
    nheads, kv_heads, head_dim, dtype = cfg["nheads"], cfg["kv_heads"], cfg["d"], cfg["dtype"]
    scale = head_dim ** -0.5
    sink = make_random_tensor((nheads,), torch.bfloat16, device="npu")
    sink_cpu = sink.detach().cpu()
    kwargs = dict(
        softmax_scale=scale,
        causal=cfg["causal"],
        learnable_sink=sink,
        num_splits=1,
        return_lse=True,
    )

    if cfg["api"] == "func":
        batch = len(q_lens)
        query = make_random_tensor((batch, q_lens[0], nheads, head_dim), dtype, device="npu")
        key = make_random_tensor((batch, kv_lens[0], kv_heads, head_dim), dtype, device="npu")
        value = make_random_tensor((batch, kv_lens[0], kv_heads, head_dim), dtype, device="npu")
        out, lse = flash_attn_func(query, key, value, **kwargs)
        out_ref, lse_ref, out_pt, lse_pt = _sink_reference_bsnd(
            query.detach().cpu(),
            key.detach().cpu(),
            value.detach().cpu(),
            scale,
            dtype,
            sink_cpu,
            cfg["causal"],
            api_name="flash_attn_func",
        )
        assert_fa_close(out, out_ref, out_pt, name="out")
        assert_fa_close(lse, lse_ref, lse_pt, name="softmax_lse")
        if cfg["causal"]:
            _assert_causal_empty_prefix(out, lse, sink, q_lens[0], kv_lens[0])
        return

    query = make_packed_random_tensor(q_lens, max(q_lens), nheads, head_dim, dtype, device="npu")
    if cfg["api"] == "varlen":
        key = make_packed_random_tensor(kv_lens, max(kv_lens), kv_heads, head_dim, dtype, device="npu")
        value = make_packed_random_tensor(kv_lens, max(kv_lens), kv_heads, head_dim, dtype, device="npu")
        table = None
        out, lse = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=make_cu_seqlens(q_lens).npu(),
            cu_seqlens_k=make_cu_seqlens(kv_lens).npu(),
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(kv_lens),
            **kwargs,
        )
        key_ref, value_ref = key.detach().cpu(), value.detach().cpu()
    else:
        table, num_pages = _sink_page_table(kv_lens)
        key_logical = make_random_tensor(
            (num_pages, _SINK_PAGE, kv_heads, head_dim), dtype, device="npu"
        )
        value_logical = make_random_tensor(
            (num_pages, _SINK_PAGE, kv_heads, head_dim), dtype, device="npu"
        )
        key = _sink_paged_view(key_logical, cfg["layout"])
        value = _sink_paged_view(value_logical, cfg["layout"])
        out, lse = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=make_cu_seqlens(q_lens).npu(),
            max_seqlen_q=max(q_lens),
            seqused_k=torch.tensor(kv_lens, dtype=torch.int32, device="npu"),
            page_table=table.npu(),
            **kwargs,
        )
        key_ref, value_ref = key_logical.detach().cpu(), value_logical.detach().cpu()

    out_ref, lse_ref, out_pt, lse_pt = _sink_packed_reference(
        query.detach().cpu(),
        key_ref,
        value_ref,
        q_lens,
        kv_lens,
        scale,
        dtype,
        sink_cpu,
        cfg["causal"],
        table=table,
        api_name="flash_attn_varlen_func",
    )
    assert_fa_close(out, out_ref, out_pt, name="out")
    assert_fa_close(lse, lse_ref, lse_pt, name="softmax_lse")



@skip_unless_910
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("physical_layout", ["BBND", "BNBD"])
@pytest.mark.parametrize("disable_scheduler_metadata", [False, True])
def test_fa4_paged_tnd_dim0_noncontiguous_with_zero_lengths(
    physical_layout, causal, dtype, disable_scheduler_metadata
):
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
        cu_seqlens_q=cu_q,
        max_seqlen_q=max(q_lengths),
        seqused_k=cache_seqlens,
        page_table=page_table,
        causal=causal,
        num_splits=1,
        return_lse=True,
        disable_scheduler_metadata=disable_scheduler_metadata,
    )
    out_view, lse_view = flash_attn_varlen_func(query, key_view, value_view, **kwargs)
    out_dense, lse_dense = flash_attn_varlen_func(
        query, key_view.contiguous(), value_view.contiguous(), **kwargs
    )

    torch.testing.assert_close(out_view, out_dense, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(lse_view, lse_dense, rtol=2e-3, atol=2e-3, equal_nan=True)
    # Batch 1 owns packed Q rows [0, 3) and has kvlen=0.
    torch.testing.assert_close(out_view[:3], torch.zeros_like(out_view[:3]), rtol=0, atol=0)
    assert torch.isposinf(lse_view[:, :3]).all()



@skip_unless_910
@pytest.mark.parametrize("head_size,kv_length,num_splits", [(256, 129, 1), (64, 2048, 2)])
def test_fa4_paged_tnd_bnbd_dim0_noncontiguous_split_paths(head_size, kv_length, num_splits):
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
        cu_seqlens_q=cu_q,
        max_seqlen_q=1,
        seqused_k=cache_seqlens,
        page_table=page_table,
        causal=True,
        num_splits=num_splits,
        return_lse=True,
    )

    out_bnbd, lse_bnbd = flash_attn_varlen_func(query, key_bnbd, value_bnbd, **kwargs)
    out_bbnd, lse_bbnd = flash_attn_varlen_func(
        query,
        key_bnbd.contiguous(),
        value_bnbd.contiguous(),
        **kwargs,
    )
    torch.testing.assert_close(out_bnbd, out_bbnd, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(lse_bnbd, lse_bbnd, rtol=2e-3, atol=2e-3)



@skip_unless_910
def test_fa4_paged_tnd_total_q_zero_fast_path():
    batch_size, num_heads, kv_heads, head_size = 2, 4, 2, 64
    query = torch.empty((0, num_heads, head_size), dtype=torch.float16, device="npu")
    cu_q = torch.zeros((batch_size + 1,), dtype=torch.int32, device="npu")
    cache_seqlens = torch.zeros((batch_size,), dtype=torch.int32, device="npu")
    page_table = torch.arange(batch_size, dtype=torch.int32, device="npu").reshape(batch_size, 1)
    key_storage = torch.randn(
        (batch_size * 2, kv_heads, 128, head_size), dtype=torch.float16, device="npu"
    )
    value_storage = torch.randn_like(key_storage)

    out, lse = flash_attn_varlen_func(
        query,
        key_storage[::2].transpose(1, 2),
        value_storage[::2].transpose(1, 2),
        cu_seqlens_q=cu_q,
        max_seqlen_q=0,
        seqused_k=cache_seqlens,
        page_table=page_table,
        num_splits=1,
        return_lse=True,
    )
    assert out.shape == query.shape
    assert lse.shape == (num_heads, 0)
    assert out.numel() == 0 and lse.numel() == 0



@skip_unless_910
@pytest.mark.parametrize("disable_scheduler_metadata", [False, True])
@pytest.mark.parametrize(
    "q_lengths,kv_lengths,extra_q,extra_kv,causal,paged",
    [
        ([4, 7, 1], [8, 9, 5], 6, 0, False, False),
        ([4, 7, 1], [8, 9, 5], 6, 5, True, False),
        ([0, 3, 6], [4, 0, 20], 9, 0, True, True),
        ([2, 5], [16, 12], 4, 0, False, True),
    ],
)
def test_fa4_tnd_total_exceeds_cu_seqlens_q(
    q_lengths, kv_lengths, extra_q, extra_kv, causal, paged, disable_scheduler_metadata
):
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

    kwargs = dict(
        cu_seqlens_q=cu_q,
        max_seqlen_q=max(q_lengths),
        causal=causal,
        num_splits=1,
        return_lse=True,
        disable_scheduler_metadata=disable_scheduler_metadata,
    )
    if paged:
        page_size = 128
        num_pages = len(kv_lengths)
        key = torch.randn((num_pages, page_size, kv_heads, head_size), dtype=dtype, device="npu")
        value = torch.randn_like(key)
        kwargs.update(
            page_table=torch.arange(num_pages, dtype=torch.int32, device="npu").view(num_pages, 1),
            seqused_k=torch.tensor(kv_lengths, dtype=torch.int32, device="npu"),
        )
        out_ref, lse_ref = flash_attn_varlen_func(query, key, value, **kwargs)
        out_pad, lse_pad = flash_attn_varlen_func(query_padded, key, value, **kwargs)
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
        kwargs.update(cu_seqlens_k=cu_k, max_seqlen_k=max(kv_lengths))
        out_ref, lse_ref = flash_attn_varlen_func(query, key, value, **kwargs)
        out_pad, lse_pad = flash_attn_varlen_func(query_padded, key, value, **kwargs)

    assert out_pad.shape[0] == used_q + extra_q
    assert lse_pad.shape == (num_heads, used_q + extra_q)
    torch.testing.assert_close(out_pad[:used_q], out_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(lse_pad[:, :used_q], lse_ref, rtol=2e-3, atol=2e-3, equal_nan=True)



@skip_unless_910
@pytest.mark.parametrize("paged", [False, True])
def test_fa4_tnd_total_shorter_than_cu_seqlens_q_errors(paged):
    """Packed T smaller than cu_seqlens_q[-1] would read past q."""
    num_heads, kv_heads, head_size = 4, 2, 64
    dtype = torch.float16
    q_lengths = [4, 7, 1]
    kv_lengths = [8, 9, 5]
    cu_q = make_cu_seqlens(q_lengths).npu()
    used_q = int(cu_q[-1])
    query = torch.randn((used_q - 1, num_heads, head_size), dtype=dtype, device="npu")
    assert query.shape[0] < used_q
    kwargs = dict(cu_seqlens_q=cu_q, max_seqlen_q=max(q_lengths), causal=True, num_splits=1)
    if paged:
        page_size = 128
        num_pages = len(kv_lengths)
        key = torch.randn((num_pages, page_size, kv_heads, head_size), dtype=dtype, device="npu")
        value = torch.randn_like(key)
        kwargs.update(
            page_table=torch.arange(num_pages, dtype=torch.int32, device="npu").view(num_pages, 1),
            seqused_k=torch.tensor(kv_lengths, dtype=torch.int32, device="npu"),
        )
    else:
        used_k = sum(kv_lengths)
        key = torch.randn((used_k, kv_heads, head_size), dtype=dtype, device="npu")
        value = torch.randn_like(key)
        kwargs.update(
            cu_seqlens_k=make_cu_seqlens(kv_lengths).npu(),
            max_seqlen_k=max(kv_lengths),
        )
    with pytest.raises(RuntimeError, match="cu_seqlens_q"):
        flash_attn_varlen_func(query, key, value, **kwargs)
