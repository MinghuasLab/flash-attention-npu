# Copyright (c) 2026, Minghua Shen.
# Window normalization follows hopper/flash_api.cpp at d15f1531a460ba456f41b01a774f33ab2db8febf.

"""Public MLA forward tests. Run with pytest, like test_flash_attn_npu_v3.py."""

from contextlib import nullcontext
import gc
import inspect
import math

import pytest
import torch
import torch_npu  # noqa: F401
from torch._subclasses.fake_tensor import FakeTensorMode

import flash_attn_npu_3 as fa3
from tests.common.compare import assert_fa_close
from tests.common.golden_cache import get_or_compute_golden
from tests.common.test_utils import make_cu_seqlens


DTYPES = [torch.float16, torch.bfloat16]
REPRESENTATIONS = ["compressed", "prefill"]
LAYOUTS = ["bsnd", "tnd", "mixed", "paged_bsnd", "paged_tnd"]
CACHE_LAYOUTS = ["bsnd", "mixed", "paged_bsnd", "paged_tnd"]


@pytest.fixture(scope="module", autouse=True)
def _reference_threads():
    if not torch.npu.is_available():
        pytest.skip("MLA requires Ascend 910B/C")
    if not torch.npu.get_device_name(0).startswith(("Ascend910B", "Ascend910C")):
        pytest.skip("MLA requires Ascend 910B/C")
    # These small CPU references cost much more with a machine-wide thread pool.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_flash_attn_metadata_signature():
    # Fixed FA3 source: hopper/flash_attn_interface.py at d15f1531a460ba456f41b01a774f33ab2db8febf.
    parameters = inspect.signature(fa3.get_scheduler_metadata).parameters
    assert tuple(parameters) == (
        "batch_size",
        "max_seqlen_q",
        "max_seqlen_k",
        "num_heads_q",
        "num_heads_kv",
        "headdim",
        "cache_seqlens",
        "qkv_dtype",
        "headdim_v",
        "cu_seqlens_q",
        "cu_seqlens_k_new",
        "cache_leftpad",
        "page_size",
        "max_seqlen_k_new",
        "causal",
        "window_size",
        "attention_chunk",
        "has_softcap",
        "num_splits",
        "pack_gqa",
        "sm_margin",
    )
    assert all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for p in parameters.values())


@pytest.mark.parametrize(
    "func,names",
    [
        pytest.param(
            fa3.flash_attn_func,
            "q k v softmax_scale causal qv q_descale k_descale v_descale window_size attention_chunk softcap num_splits pack_gqa deterministic sm_margin return_attn_probs",
            id="flash_attn_func",
        ),
        pytest.param(
            fa3.flash_attn_varlen_func,
            "q k v cu_seqlens_q cu_seqlens_k max_seqlen_q max_seqlen_k seqused_q seqused_k softmax_scale causal qv q_descale k_descale v_descale window_size attention_chunk softcap num_splits pack_gqa deterministic sm_margin return_attn_probs",
            id="flash_attn_varlen_func",
        ),
        pytest.param(
            fa3.flash_attn_with_kvcache,
            "q k_cache v_cache k v qv rotary_cos rotary_sin cache_seqlens cache_batch_idx cache_leftpad page_table cu_seqlens_q cu_seqlens_k_new max_seqlen_q rotary_seqlens q_descale k_descale v_descale softmax_scale causal window_size attention_chunk softcap rotary_interleaved scheduler_metadata num_splits pack_gqa sm_margin return_softmax_lse",
            id="flash_attn_with_kvcache",
        ),
    ],
)
def test_flash_attn_forward_signature(func, names):
    # The public parameter names/order follow the fixed official FA3 interface.
    parameters = inspect.signature(func).parameters
    assert tuple(parameters) == tuple(names.split())
    assert all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for p in parameters.values())


@pytest.mark.parametrize("data_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_softcap", [False, True])
def test_flash_attn_metadata_real_fake_contract(data_type, has_softcap):
    lengths = torch.full((2,), 128, dtype=torch.int32, device="npu")
    args = (
        2,
        3,
        128,
        4,
        2,
        128,
        lengths,
        data_type,
        128,
        None,
        None,
        None,
        None,
        0,
        False,
        (-1, -1),
        0,
        has_softcap,
        1,
        None,
        0,
    )
    actual = fa3.get_scheduler_metadata(*args)
    with FakeTensorMode(allow_non_fake_inputs=True):
        fake = fa3.get_scheduler_metadata(*args)
    assert actual.shape == fake.shape == (2376,)
    assert actual.dtype == fake.dtype == torch.uint8
    assert actual.device == fake.device == lengths.device
    assert actual.stride() == fake.stride() == (1,)
    for keyword in ("softcap", "softmax_scale"):
        with pytest.raises(TypeError, match=keyword):
            fa3.get_scheduler_metadata(*args, **{keyword: 0.23})


def _view(tensor, kind):
    """Allocate poisoned backing storage, preserving legal outer strides."""
    if kind == "contiguous" or tensor.numel() == 0:
        return tensor.clone()
    shape = list(tensor.shape)
    if kind == "broadcast":
        return tensor.narrow(0, 0, 1).expand_as(tensor)
    axis = -1 if kind == "scale-head" else -2 if kind == "head-gap" and tensor.ndim >= 3 else 0
    shape[axis] = 2 * shape[axis] + 2
    fill = float("nan") if tensor.is_floating_point() else 117
    storage = torch.full(shape, fill, dtype=tensor.dtype, device=tensor.device)
    selection = [slice(None)] * tensor.ndim
    selection[axis] = slice(1, -1, 2)
    result = storage[tuple(selection)]
    result.copy_(tensor)
    return result


def _make_case(
    layout="bsnd",
    representation="compressed",
    dtype=torch.bfloat16,
    *,
    q=(7, 3),
    k=(65, 33),
    used_q=None,
    used_k=None,
    heads=8,
    kv_heads=2,
    splits=1,
    causal=False,
    window=(-1, -1),
    cap=0.0,
    scale=None,
    pack=None,
    margin=0,
    page=16,
    view="contiguous",
    repeated_pages=False,
    pattern=None,
):
    packed = layout in ("tnd", "mixed", "paged_tnd")
    paged = layout.startswith("paged")
    batch, sq, sk = len(q), max(q, default=0), max(k, default=0)
    qa = list(q) if packed else [sq] * batch
    ka = list(k) if layout == "tnd" else [sk] * batch
    if paged:
        sk = math.ceil(sk / page) * page
        ka = [sk] * batch
    if layout == "func":
        used_k = ka
    qused = list(used_q if used_q is not None else qa)
    kused = list(used_k if used_k is not None else k)
    if layout == "func":
        kused = ka
    hk = 1 if representation == "compressed" else kv_heads
    dq, dv = (64, 512) if representation == "compressed" else (192, 128)
    raw = {
        name: torch.randn(batch, seq, nh, dim).mul_(0.1).to(dtype)
        for name, seq, nh, dim in (("q", sq, heads, dq), ("k", sk, hk, dq), ("v", sk, hk, dv))
    }
    if representation == "compressed":
        raw["qv"] = torch.randn(batch, sq, heads, dv).mul_(0.1).to(dtype)
    if pattern == "block-cancellation":
        for value in raw.values():
            value.zero_()
        raw["qv"][..., 0] = 1
        raw["v"][:, 512, :, 0] = 1
        raw["v"][:, 510:513, :, 1] = torch.tensor([1, -1, 1], dtype=dtype).view(1, 3, 1)
    elif pattern == "softcap-normalization":
        for value in raw.values():
            value.zero_()
        raw["qv"][..., 0] = 1
        raw["v"][..., 0] = 0.234375
        raw["v"][:, 64, :, 0] = 0.25
        raw["v"][..., 1] = 1.75
    elif pattern == "rounding-ties":
        raw["q"].zero_()
        if "qv" in raw:
            raw["qv"].zero_()
        eps = torch.finfo(dtype).eps
        raw["v"].zero_()
        raw["v"][:, 31] = torch.tensor([1, -1, 1 + eps, -1 - eps], dtype=dtype).repeat(dv // 4)
        raw["v"][:, 32] = torch.tensor(
            [1 + eps, -1 - eps, 1 + 2 * eps, -1 - 2 * eps], dtype=dtype
        ).repeat(dv // 4)
    elif pattern in ("saturation", "cancellation"):
        qc = raw["qv"] if "qv" in raw else raw["q"][..., :128]
        qr = raw["q"] if "qv" in raw else raw["q"][..., 128:]
        kc = raw["v"] if "qv" in raw else raw["k"][..., :128]
        kr = raw["k"] if "qv" in raw else raw["k"][..., 128:]
        qc.zero_()
        qr.zero_()
        qc[..., 0] = 32
        values = torch.linspace(-24, 24, sk).to(dtype)[None, :, None]
        kc[..., 0] = values
        if pattern == "cancellation":
            qr[..., 0] = 32
            kr[..., 0] = -values + 0.5
    physical = {}
    table = None
    if paged:
        npages = batch * (sk // page)
        table = torch.arange(npages - 1, -1, -1, dtype=torch.int32).reshape(batch, sk // page)
        for name in ("k", "v"):
            pool = torch.empty(npages, page, hk, raw[name].shape[-1], dtype=dtype)
            for b in range(batch):
                pool[table[b].long()] = raw[name][b].reshape(-1, page, hk, raw[name].shape[-1])
            physical[name] = pool
        if repeated_pages and table.shape[1] > 1:
            table[:, -1] = table[:, 0]
    for name, value in raw.items():
        if name in ("q", "qv") and packed:
            value = (
                torch.cat([value[b, :n] for b, n in enumerate(qa)])
                if batch
                else value.reshape(0, heads, value.shape[-1])
            )
        elif name in ("k", "v") and layout == "tnd":
            value = (
                torch.cat([value[b, :n] for b, n in enumerate(ka)])
                if batch
                else value.reshape(0, hk, value.shape[-1])
            )
        physical.setdefault(name, value)
    tensors = {name: _view(value.to("npu"), view) for name, value in physical.items()}
    # Broadcast views change logical input values as well as physical storage.
    if view == "broadcast":
        physical = {name: _view(value, view) for name, value in physical.items()}
    if paged:
        for name in ("k", "v"):
            for b in range(batch):
                raw[name][b] = physical[name][table[b].long()].reshape(sk, hk, raw[name].shape[-1])
    if view == "broadcast":
        for name, value in physical.items():
            if name in ("k", "v") and paged:
                continue
            if value.ndim == 4:
                raw[name] = value.clone()
            else:
                offset = 0
                for b, n in enumerate(qa if name in ("q", "qv") else ka):
                    raw[name][b, :n] = value[offset : offset + n]
                    offset += n
    options = dict(
        softmax_scale=scale,
        causal=causal,
        window_size=window,
        softcap=cap,
        num_splits=splits,
        pack_gqa=pack,
        sm_margin=margin,
    )
    lengths = torch.tensor(kused, dtype=torch.int32, device="npu")
    prefix = make_cu_seqlens(qa).to("npu") if packed else None
    return dict(
        layout=layout,
        representation=representation,
        raw=raw,
        t=tensors,
        q=qa,
        k=ka,
        qused=qused,
        kused=kused,
        sq=sq,
        sk=sk,
        page=page,
        lengths=lengths,
        prefix=prefix,
        options=options,
        used_q=torch.tensor(qused, dtype=torch.int32, device="npu") if used_q is not None else None,
        prefix_k=make_cu_seqlens(ka).to("npu") if layout == "tnd" else None,
        table=_view(table.to("npu"), "offset") if table is not None else None,
    )


def _metadata(case):
    if case["layout"] in ("func", "tnd"):
        return None
    t, o = case["t"], case["options"]
    return fa3.get_scheduler_metadata(
        len(case["q"]),
        case["sq"],
        case["sk"],
        t["q"].shape[-2],
        t["k"].shape[-2],
        t["q"].shape[-1],
        case["lengths"],
        qkv_dtype=t["q"].dtype,
        headdim_v=t["v"].shape[-1],
        cu_seqlens_q=case["prefix"],
        page_size=case["page"] if case["table"] is not None else None,
        has_softcap=o["softcap"] > 0,
        **{key: o[key] for key in ("causal", "window_size", "num_splits", "pack_gqa", "sm_margin")},
    )


def _call(case, record=None, *, lse=True, **extra):
    t = case["t"]
    args = dict(t, **case["options"])
    args.update(extra)
    if case["layout"] == "func":
        return fa3.flash_attn_func(**args, return_attn_probs=lse)
    if case["layout"] == "tnd":
        return fa3.flash_attn_varlen_func(
            **args,
            cu_seqlens_q=case["prefix"],
            cu_seqlens_k=case["prefix_k"],
            max_seqlen_q=case["sq"],
            max_seqlen_k=case["sk"],
            seqused_q=case["used_q"],
            seqused_k=case["lengths"],
            return_attn_probs=lse,
        )
    args["k_cache"], args["v_cache"] = args.pop("k"), args.pop("v")
    args.setdefault("cache_seqlens", case["lengths"])
    args.setdefault("page_table", case["table"])
    return fa3.flash_attn_with_kvcache(
        **args,
        cu_seqlens_q=case["prefix"],
        max_seqlen_q=case["sq"] if case["prefix"] is not None else None,
        scheduler_metadata=record,
        return_softmax_lse=lse,
    )


def _attention_reference(q, k, v, qv, options, upcast):
    # Preserve the independent FP64/input-dtype equations used by the MLA E2Es.
    sq, sk = q.shape[1], k.shape[1]
    dtype = torch.float64 if upcast else q.dtype
    scale = options["softmax_scale"]
    if scale is None:
        scale = (q.shape[-1] + (0 if qv is None else qv.shape[-1])) ** -0.5
    left, right = options["window_size"]
    if left >= sk - 1:
        left = -1
    if right >= sq - 1:
        right = -1
    causal = options["causal"]
    if sq == 1 and left == right == -1:
        causal = False
    if causal:
        right = 0
    center = torch.arange(sq)[:, None] + sk - sq
    position = torch.arange(sk)[None, :]
    mask = torch.zeros(sq, sk, dtype=torch.bool)
    if left >= 0:
        mask |= position < center - left
    if right >= 0:
        mask |= position > center + right
    groups = q.shape[-2] // k.shape[-2]
    query = q.to(dtype).transpose(1, 2)
    key = k.to(dtype).repeat_interleave(groups, dim=2).transpose(1, 2)
    value = v.to(dtype).repeat_interleave(groups, dim=2).transpose(1, 2)
    scores = (query if upcast else query * scale) @ key.transpose(-1, -2)
    if qv is not None:
        content = qv.to(dtype).transpose(1, 2)
        scores = scores + (content if upcast else content * scale) @ value.transpose(-1, -2)
    scores = scores * scale if upcast else scores.float()
    cap = options["softcap"]
    if cap:
        scores = cap * torch.tanh(scores / cap)
    scores.masked_fill_(mask, -torch.inf)
    lse = scores.logsumexp(-1).float()
    empty = mask.all(-1)
    safe = scores.masked_fill(empty[:, None], 0)
    if upcast:
        weights = (safe - safe.amax(-1, keepdim=True)).exp().masked_fill(empty[:, None], 0)
        out = (weights @ value) / weights.sum(-1, keepdim=True).clamp_min(1)
    else:
        weights = safe.softmax(-1).masked_fill(empty[:, None], 0)
        out = weights.to(dtype) @ value
    return out.transpose(1, 2).to(q.dtype), lse


def _check(case, outputs):
    out, lse = (value.cpu() for value in outputs[:2])
    t = case["t"]
    assert out.shape == (*t["q"].shape[:-1], t["v"].shape[-1])
    assert out.dtype == t["q"].dtype and lse.dtype == torch.float32
    offset = 0
    for b, (allocated, nq, nk) in enumerate(zip(case["q"], case["qused"], case["kused"])):
        packed = case["prefix"] is not None
        actual = out[offset : offset + nq].unsqueeze(0) if packed else out[b : b + 1, :nq]
        logs = lse[:, offset : offset + nq].unsqueeze(0) if packed else lse[b : b + 1, :, :nq]
        offset += allocated
        if nq == 0:
            continue
        if nk == 0:
            assert actual.count_nonzero() == 0
            expected = math.inf if case["table"] is None and case["sk"] == 0 else -math.inf
            assert (logs == expected).all()
            continue
        raw = {
            name: value[b : b + 1, : nq if name in ("q", "qv") else nk]
            for name, value in case["raw"].items()
        }

        def compute():
            values = []
            for upcast in (True, False):
                values.extend(
                    _attention_reference(
                        raw["q"], raw["k"], raw["v"], raw.get("qv"), case["options"], upcast
                    )
                )
            return dict(zip(("out", "lse", "out_pt", "lse_pt"), values))

        refs = get_or_compute_golden(
            nodeid="mla-public-attention",
            metadata=case["options"],
            inputs=raw,
            compute_fn=compute,
            expected_keys=("out", "lse", "out_pt", "lse_pt"),
            source_files=[__file__],
        )
        assert_fa_close(
            actual, refs["out"], refs["out_pt"], softcap=case["options"]["softcap"], name="MLA out"
        )
        assert_fa_close(
            logs, refs["lse"], refs["lse_pt"], softcap=case["options"]["softcap"], name="MLA LSE"
        )
    if len(outputs) == 4:
        _check_partials(
            outputs,
            case["options"]["num_splits"],
            case["prefix"] is not None,
            empty_kv=case["table"] is None and case["sk"] == 0,
        )


def _check_partials(outputs, splits, packed, *, empty_kv=False):
    out, lse, partial, partial_lse = (value.cpu() for value in outputs)
    assert partial.dtype == partial_lse.dtype == torch.float32
    if splits == 1:
        assert partial.numel() == partial_lse.numel() == 0
        return
    assert partial.shape == (splits or 26, *lse.shape, out.shape[-1])
    assert partial_lse.shape == (splits or 26, *lse.shape)
    # FA3 epilogue_fwd::store_zero does not initialize O for LSE=-inf.
    # Mask before multiplication: an unused slot may contain NaN.
    partial = partial.masked_fill(torch.isneginf(partial_lse)[..., None], 0)
    assert not partial.isnan().any() and not partial_lse.isnan().any()
    if empty_kv:
        # Global nonpaged emptiness has +inf LSE but unused split slots stay -inf.
        assert partial.count_nonzero() == 0
        assert torch.isneginf(partial_lse).all()
        return
    weights = partial_lse.softmax(0).nan_to_num()
    merged = (
        (weights[..., None] * partial)
        .sum(0)
        .transpose(0 if packed else 1, 1 if packed else 2)
        .to(out.dtype)
    )
    if splits == 0:
        # Dynamic single-split rows write final O/LSE directly, as in FA3.
        # Only rows with actual partials participate in the split combine.
        split_rows = partial_lse.isfinite().any(0)
        output_rows = split_rows.transpose(0 if packed else 1, 1 if packed else 2)
        out = out.masked_fill(~output_rows[..., None], 0)
        lse = lse.masked_fill(~split_rows, -torch.inf)
    torch.testing.assert_close(
        merged,
        out,
        rtol=0.008 if out.dtype == torch.bfloat16 else 0.001,
        atol=0.004 if out.dtype == torch.bfloat16 else 0.0005,
    )
    torch.testing.assert_close(partial_lse.logsumexp(0), lse, rtol=2e-5, atol=2e-5)


def _equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        actual, expected = (actual,), (expected,)
    assert len(actual) == len(expected)
    for index, (a, b) in enumerate(zip(actual, expected)):
        assert (a.shape, a.stride(), a.dtype, a.device) == (b.shape, b.stride(), b.dtype, b.device)
        if len(actual) == 4 and index == 2 and a.numel():
            a = a.masked_fill(torch.isneginf(actual[3])[..., None], 0)
            b = b.masked_fill(torch.isneginf(expected[3])[..., None], 0)
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize(
    "causal,window,cap,splits,layout,representation,dtype",
    [
        pytest.param(
            causal,
            window,
            cap,
            splits,
            layout,
            representation,
            dtype,
            id=f"{mask}-{cap_name}-{split_name}-{layout}-{representation}-{dtype_name}",
        )
        for mask, causal, window in [
            ("unmasked", False, (-1, -1)),
            ("causal", True, (-1, -1)),
            ("window", False, (2, 1)),
        ]
        for cap_name, cap in [("no-softcap", 0), ("softcap", 15)]
        for split_name, splits in [("auto", 0), ("single", 1), ("explicit", 3)]
        for layout in LAYOUTS
        for representation in REPRESENTATIONS
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]
        # Exclude the confirmed CANN zero-cancellation configuration.
        if (dtype, representation, layout, splits, cap, causal, window)
        != (torch.bfloat16, "compressed", "mixed", 1, 0, False, (2, 1))
    ],
)
def test_mla_forward(dtype, representation, layout, splits, cap, causal, window):
    # Keep one input per combination outside the documented CANN source defect.
    # KV spans multiple tiles; softcap inputs must make clipping observable.
    case = _make_case(
        layout,
        representation,
        dtype,
        q=(3, 1),
        k=(1057, 513),
        splits=splits,
        cap=cap,
        causal=causal,
        window=window,
        scale=0.23 if cap else None,
        pattern="saturation" if cap else None,
    )
    full = _call(case)
    _check(case, full)
    torch.testing.assert_close(_call(case, lse=False), full[0], rtol=0, atol=0)
    if layout in CACHE_LAYOUTS:
        record = _metadata(case)
        reused = _call(case, record)
        _check(case, reused)
        torch.testing.assert_close(_call(case, record, lse=False), reused[0], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("splits", [0, 3], ids=["auto", "explicit"])
@pytest.mark.parametrize("cap", [0, 15], ids=["no-softcap", "softcap"])
def test_mla_block_cancellation(dtype, layout, splits, cap):
    # The first two visible values cancel exactly within a KV block. The next
    # block raises the score maximum and contributes a nonzero output. Auto and
    # explicit splitting keep these contributions in separate fragments; the
    # documented CANN single-fragment defect is excluded from this parameter table.
    case = _make_case(
        layout,
        "compressed",
        dtype,
        q=(1,),
        k=(513,),
        window=(2, 1),
        splits=splits,
        cap=cap,
        scale=1.0,
        pattern="block-cancellation",
    )
    _check(case, _call(case))
    if layout in CACHE_LAYOUTS:
        _check(case, _call(case, _metadata(case)))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("splits", [0, 1, 3], ids=["auto", "single", "explicit"])
def test_mla_softcap_normalization(dtype, layout, splits):
    # A constant V channel must stay constant after attention. These logits
    # expose bias from dividing rounded P @ V by the sum of unrounded P.
    case = _make_case(
        layout,
        "compressed",
        dtype,
        q=(1,),
        k=(65,),
        splits=splits,
        cap=15,
        scale=1.0,
        pattern="softcap-normalization",
    )
    _check(case, _call(case))
    if layout in CACHE_LAYOUTS:
        _check(case, _call(case, _metadata(case)))


# Extra values for combinations already covered above. Keep one relevant layout
# per boundary, both numerical representations/dtypes, and all layouts where
# empty-output contracts or packed scheduling differ. Fields: name, layouts, options.
BOUNDARY_CASES = [
    ("causal-empty-rows", ["bsnd"], dict(q=(17, 3), k=(3, 1), causal=True)),
    ("negative-window", ["tnd"], dict(window=(-128, 864))),
    ("left-window", ["mixed"], dict(q=(3, 1), k=(513, 7), window=(256, -1))),
    ("right-window", ["paged_bsnd"], dict(q=(7, 3), k=(513, 7), window=(-1, 2))),
    ("single-key-window", ["paged_tnd"], dict(q=(33, 3), k=(1, 0), window=(0, 0))),
    ("overlong-window", ["bsnd"], dict(window=(16384, 16384), causal=True)),
    ("causal-empty-splits", ["tnd"], dict(q=(33, 3), k=(3, 1), splits=8, causal=True)),
    (
        "window-used-stride",
        ["paged_tnd"],
        dict(q=(17, 3), k=(1057, 513), used_k=(513, 7), splits=4, window=(256, 2), view="head-gap"),
    ),
    ("long-auto", ["mixed"], dict(q=(3, 1), k=(4097, 1057), splits=0)),
    ("compact-auto", ["tnd"], dict(q=(7, 3), k=(4097, 513), splits=0)),
    ("long-query-auto", ["tnd"], dict(q=(2048, 512), k=(33, 1), heads=2, splits=0)),
    (
        "continuous-auto",
        ["bsnd", "tnd", "paged_tnd"],
        dict(q=(17, 3, 9, 7) * 8, k=(3, 0, 513, 1025) + (3,) * 28, splits=0, cap=15),
    ),
    (
        "auto-causal",
        ["paged_bsnd"],
        dict(q=(65, 3), k=(1057, 513), used_k=(1057, 0), splits=0, causal=True),
    ),
    ("softcap-negative-scale", ["bsnd"], dict(cap=30, scale=-0.23, window=(2, 1))),
    (
        "rounding-ties",
        ["tnd"],
        dict(q=(1,), k=(33,), cap=15, splits=2, scale=0.0, window=(1, 0), pattern="rounding-ties"),
    ),
    ("saturation", ["paged_bsnd"], dict(cap=15, scale=0.23, splits=2, pattern="saturation")),
    ("cancellation", ["paged_tnd"], dict(cap=15, scale=0.23, splits=2, pattern="cancellation")),
    ("large-softcap", ["mixed"], dict(cap=1e8)),
    ("small-softcap", ["bsnd"], dict(cap=1e-4, splits=0)),
    ("unpacked-auto", LAYOUTS, dict(pack=False, margin=1, splits=0, causal=True)),
    ("packed-explicit", LAYOUTS, dict(pack=True, margin=2, splits=3, window=(2, 1), cap=15)),
    (
        "max-splits",
        ["bsnd"],
        dict(q=(1,), k=(33,), heads=2, splits=256, scale=0.0, pack=False, margin=1),
    ),
    (
        "long-kv",
        ["bsnd", "paged_tnd"],
        dict(q=(1, 1), k=(16384, 513), heads=2, splits=3, causal=True, page=1024),
    ),
    (
        "query-tile-tail",
        ["tnd"],
        dict(q=(513, 3), k=(33, 1), heads=4, splits=3, window=(2, 1), pack=False),
    ),
    (
        "kv-tile-tail",
        ["paged_bsnd"],
        dict(
            q=(17, 3), k=(1057, 513), page=48, splits=0, cap=15, pack=False, margin=1, causal=True
        ),
    ),
    ("empty-query", LAYOUTS, dict(q=(0, 0), splits=0)),
    ("empty-kv", LAYOUTS, dict(k=(0, 0), splits=3)),
    ("empty-kv-auto", ["bsnd"], dict(q=(3, 3), k=(0, 0), heads=1, kv_heads=1, splits=0)),
    *[
        (
            f"used-empty-tail-{splits}-{used_k}",
            ["bsnd"],
            dict(q=(3, 3), k=(129, 0), heads=1, kv_heads=1, used_k=used_k, splits=splits),
        )
        for splits in (0, 1, 3)
        for used_k in ((0, 0), (129, 0))
    ],
    ("empty-batch", LAYOUTS, dict(q=(), k=())),
]


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize(
    "layout,spec",
    [
        pytest.param(layout, spec, id=f"{name}-{layout}")
        for name, layouts, spec in BOUNDARY_CASES
        for layout in layouts
    ],
)
def test_mla_boundaries(dtype, representation, layout, spec):
    case = _make_case(layout, representation, dtype, **spec)
    full = _call(case)
    _check(case, full)
    torch.testing.assert_close(_call(case, lse=False), full[0], rtol=0, atol=0)
    if layout in CACHE_LAYOUTS:
        record = _metadata(case)
        reused = _call(case, record)
        _check(case, reused)
        torch.testing.assert_close(_call(case, record, lse=False), reused[0], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize(
    "splits,causal,window,cap",
    [
        (1, False, (-1, -1), 0),
        (0, True, (-1, -1), 0),
        (3, False, (2, 1), 15),
        (1, True, (0, 0), 30),
    ],
)
def test_mla_func(dtype, representation, splits, causal, window, cap):
    case = _make_case(
        "func", representation, dtype, splits=splits, causal=causal, window=window, cap=cap
    )
    outputs = _call(case)
    assert len(outputs) == 2
    _check(case, outputs)
    _equal(_call(case, lse=False), outputs[0])
    _equal(
        _call(case, softmax_scale=(576 if representation == "compressed" else 192) ** -0.5), outputs
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("sq,sk", [(0, 3), (3, 0), (0, 0)])
def test_mla_func_empty(dtype, representation, sq, sk):
    case = _make_case("func", representation, dtype, q=(sq,), k=(sk,))
    _check(case, _call(case))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("view", ["offset", "head-gap", "broadcast"])
def test_mla_views_and_reuse(dtype, representation, layout, view):
    case = _make_case(
        layout, representation, dtype, view=view, repeated_pages=True, splits=3, causal=True
    )
    record = _metadata(case)
    before = {name: value.clone() for name, value in case["t"].items()}
    _check(case, _call(case, record))
    for name, original in before.items():
        _equal(case["t"][name].contiguous(), original.contiguous())
    # Geometry records remain reusable across tensor allocations and new content.
    other = _make_case(
        layout, representation, dtype, view=view, repeated_pages=True, splits=3, causal=True
    )
    _check(other, _call(other, record))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("layout", CACHE_LAYOUTS)
@pytest.mark.parametrize("pack", [None, False], ids=["default", "unpacked"])
def test_mla_accumulator_buffer_reuse(dtype, representation, layout, pack):
    case = _make_case(layout, representation, dtype, q=(33, 3), k=(513, 0), splits=0, pack=pack)
    record = _metadata(case)
    outputs = _call(case, record)
    outputs[2].fill_(float("nan"))
    outputs[3].fill_(float("nan"))
    torch.npu.synchronize()
    del outputs
    outputs = _call(case, record)
    _check(case, outputs)
    # Unused public partials must be ignored even when their storage is poisoned.
    outputs[2].masked_fill_(torch.isneginf(outputs[3])[..., None], float("nan"))
    _check_partials(outputs, 0, case["prefix"] is not None)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("layout", CACHE_LAYOUTS)
@pytest.mark.parametrize(
    "query,keys,split",
    [(257, 1, False), (1, 16384, True)],
    ids=["direct", "split-kv"],
)
def test_mla_auto_partial_outputs(dtype, representation, layout, query, keys, split):
    # One KV tile cannot split; a single short query with long KV must split.
    case = _make_case(layout, representation, dtype, q=(query,), k=(keys,), heads=2, splits=0)
    outputs = _call(case)
    _check(case, outputs)
    count = outputs[3].isfinite().sum(0)
    assert ((count > 1) if split else (count == 0)).all()


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("splits", [0, 1, 3])
def test_mla_varlen_allocation_used(dtype, representation, splits):
    case = _make_case(
        "tnd",
        representation,
        dtype,
        q=(7, 0, 17),
        k=(513, 31, 3),
        used_q=(3, 0, 1),
        used_k=(511, 0, 2),
        view="offset",
        splits=splits,
        window=(2, 1),
        cap=15,
    )
    _check(case, _call(case))
    case["used_q"].copy_(case["used_q"].new_tensor([1, 0, 2]))
    case["lengths"].copy_(case["lengths"].new_tensor([7, 0, 1]))
    case["qused"], case["kused"] = [1, 0, 2], [7, 0, 1]
    _check(case, _call(case))


@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("layout", CACHE_LAYOUTS)
@pytest.mark.parametrize("form", ["none", "integer", "clone", "offset"])
def test_mla_cache_length_and_record_forms(representation, layout, form):
    # Length and record storage are int32; the forward matrix covers both data dtypes.
    case = _make_case(layout, representation, q=(3, 3), k=(32, 32), splits=0)
    record = _metadata(case)
    expected = _call(case, record)
    _check(case, expected)
    if form in ("none", "integer"):
        result = _call(case, record, cache_seqlens=None if form == "none" else 32)
    elif form == "clone":
        result = _call(case, record.clone())
    else:
        storage = torch.empty(record.numel() + 8, dtype=record.dtype, device=record.device)
        storage[4:-4].copy_(record)
        result = _call(case, storage[4:-4])
    _equal(result, expected)
    _equal(_call(case, record, lse=False), expected[0])


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("layout", CACHE_LAYOUTS)
@pytest.mark.parametrize("fake", [False, True], ids=["real", "fake"])
def test_mla_empty_accumulators_default_dtype(dtype, representation, layout, fake):
    case = _make_case(layout, representation, dtype)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with FakeTensorMode() if fake else nullcontext() as mode:
            if fake:
                case = {
                    key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                    for key, value in case.items()
                }
                case["t"] = {key: mode.from_tensor(value) for key, value in case["t"].items()}
            out, lse, *empty = _call(case, _metadata(case))
            assert out.dtype == dtype and lse.dtype == torch.float32
            for tensor in empty:
                assert tensor.shape == (0,) and tensor.dtype == torch.float64
                assert tensor.device == case["t"]["q"].device
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("layout", CACHE_LAYOUTS)
def test_mla_metadata_scale_and_softcap_reuse(dtype, representation, layout):
    case = _make_case(layout, representation, dtype, cap=15, scale=0.125, splits=0)
    record = _metadata(case)
    assert record.untyped_storage().data_ptr() != _metadata(case).untyped_storage().data_ptr()
    for scale, cap in ((0.125, 15), (-0.23, 30), (0.0, 1.0)):
        case["options"].update(softmax_scale=scale, softcap=cap)
        result = _call(case, record)
        _equal(result, _call(case))
        _check(case, result)


@pytest.mark.parametrize("batch", [(2**32 - 1 - 380) // 10 + 1, 2**32])
def test_mla_fake_metadata_overflow(batch):
    with FakeTensorMode():
        lengths = torch.empty(batch, dtype=torch.int32, device="npu")
        with pytest.raises(RuntimeError, match="overflow|shape bounds"):
            fa3.get_scheduler_metadata(batch, 0, 0, 1, 1, 64, lengths, headdim_v=512, num_splits=0)


@pytest.mark.parametrize("layout", ["bsnd", "mixed", "paged_bsnd", "paged_tnd"])
def test_mla_fake_workspace_overflow(layout):
    # Output-only public calls still allocate the native split workspace.
    with FakeTensorMode():
        batch, heads = 1 << 24, 1 << 20
        tnd = layout in ("mixed", "paged_tnd")
        paged = layout.startswith("paged")
        query = (batch, heads) if tnd else (batch, 1, heads)
        cache = (1, 16, 1) if paged else (batch, 1, 1)
        options = dict(dtype=torch.float16, device="npu")
        with pytest.raises(RuntimeError, match="overflow|bounds"):
            fa3.flash_attn_with_kvcache(
                torch.empty((*query, 64), **options),
                torch.empty((*cache, 64), **options),
                torch.empty((*cache, 512), **options),
                qv=torch.empty((*query, 512), **options),
                cu_seqlens_q=torch.empty(batch + 1, dtype=torch.int32, device="npu")
                if tnd
                else None,
                max_seqlen_q=1 if tnd else None,
                page_table=torch.empty((batch, 1), dtype=torch.int32, device="npu")
                if paged
                else None,
                num_splits=256,
            )


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
def test_mla_fake_wide_cache(dtype):
    # Physical cache spans are 64-bit even when the Q prefixes are int32.
    with FakeTensorMode():
        batch, capacity = (1 << 18) + 1, 16384
        out, lse, *_ = fa3.flash_attn_with_kvcache(
            torch.empty((1, 8, 64), dtype=dtype, device="npu"),
            torch.empty((batch, capacity, 1, 64), dtype=dtype, device="npu"),
            torch.empty((batch, capacity, 1, 512), dtype=dtype, device="npu"),
            qv=torch.empty((1, 8, 512), dtype=dtype, device="npu"),
            cu_seqlens_q=torch.empty(batch + 1, dtype=torch.int32, device="npu"),
            max_seqlen_q=1,
            num_splits=1,
            return_softmax_lse=True,
        )
        assert out.shape == (1, 8, 512) and lse.shape == (8, 1)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
# MQA minimum/maximum, non-power-of-two GQA, and MHA. The main matrix adds 8/2.
@pytest.mark.parametrize("heads,kv_heads", [(1, 1), (6, 3), (8, 8), (128, 1)])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_mla_prefill_gqa(dtype, heads, kv_heads, layout):
    case = _make_case(
        layout,
        "prefill",
        dtype,
        q=(3, 1),
        k=(129, 7),
        heads=heads,
        kv_heads=kv_heads,
        splits=0,
        causal=True,
    )
    _check(case, _call(case))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("layout", ["paged_bsnd", "paged_tnd"])
# Minimum, non-power-of-two, standard and large pages, each with a partial tail.
@pytest.mark.parametrize("page", [16, 48, 128, 1024])
def test_mla_page_boundaries(dtype, representation, layout, page):
    case = _make_case(
        layout,
        representation,
        dtype,
        q=(3, 1),
        k=(page + 1, page - 1),
        page=page,
        repeated_pages=True,
        view="offset",
    )
    _check(case, _call(case, _metadata(case)))


@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("layout", ["func", *LAYOUTS])
@pytest.mark.parametrize(
    "splits,pack,margin,cap", [(1, None, 0, 0), (0, False, 1, 15), (3, True, 2, 30)]
)
def test_mla_fake_compile(representation, layout, splits, pack, margin, cap):
    case = _make_case(
        layout, representation, splits=splits, pack=pack, margin=margin, cap=cap, window=(2, 1)
    )
    record = _metadata(case)
    expected = _call(case, record)
    _check(case, expected)
    with FakeTensorMode() as mode:
        fake = {
            key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
            for key, value in case.items()
        }
        fake["t"] = {key: mode.from_tensor(value) for key, value in case["t"].items()}
        outputs = _call(fake, _metadata(fake))
        for actual, ref in zip(outputs, expected):
            assert (actual.shape, actual.stride(), actual.dtype, actual.device) == (
                ref.shape,
                ref.stride(),
                ref.dtype,
                ref.device,
            )
        assert _call(fake, lse=False).shape == expected[0].shape
    torch._dynamo.reset()
    compiled = torch.compile(
        lambda: (_call(case, record), _call(case, _metadata(case), lse=False)),
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )
    full, output = compiled()
    _equal(full, expected)
    _equal(output, expected[0])
    case["t"]["q"].neg_()
    case["raw"]["q"].neg_()
    full, output = compiled()
    _check(case, full)
    _equal(output, full[0])


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("layout", ["func", *LAYOUTS])
@pytest.mark.parametrize("splits", [0, 3], ids=["auto", "split"])
def test_mla_stream_lifetime(dtype, representation, layout, splits):
    case = _make_case(
        layout, representation, dtype, splits=splits, cap=15, window=(2, 1), view="head-gap"
    )
    expected = _call(case, _metadata(case))
    _check(case, expected)
    ready = torch.npu.Event()
    ready.record()
    work = []
    for _ in range(3):
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            stream.wait_event(ready)
            temporary = dict(case, t={key: value.clone() for key, value in case["t"].items()})
            output = tuple(value.clone() for value in _call(temporary, _metadata(temporary)))
            work.append((stream, output))
            del temporary
    del case
    gc.collect()
    pressure = [torch.full((262144,), 99.0, device="npu") for _ in range(4)]
    for stream, output in work:
        stream.synchronize()
        _equal(output, expected)
    assert len({output[0].data_ptr() for _, output in work}) == 3
    del pressure


# (name, layout, query allocation, heads, KV used lengths, causal)
# The fixed-source migration probes are retired. These are public shape/length
# boundaries, including direct/FD and the BSND/TND trailing-empty regressions.
INT8_CASES = [
    ("decode-min", "bsnd", (1,), 1, (1,), False),
    ("decode-h2-tail", "bsnd", (1,), 2, (127,), False),
    ("decode-h4-page", "bsnd", (1,), 4, (128,), False),
    ("decode-h8-page-tail", "bsnd", (1,), 8, (129,), False),
    ("decode-h16", "bsnd", (1,), 16, (512,), False),
    ("decode-h32", "bsnd", (1,), 32, (512,), False),
    ("decode-h64-fd", "bsnd", (1,), 64, (4097,), False),
    ("decode-h128", "bsnd", (1,), 128, (129,), False),
    ("bsnd-two-query", "bsnd", (2, 2), 8, (129, 127), True),
    ("bsnd-four-query", "bsnd", (4,), 32, (512,), True),
    ("bsnd-sixteen-query", "bsnd", (16,), 128, (512,), True),
    ("tnd-single-query", "tnd", (1, 1), 16, (127, 129), False),
    ("tnd-ragged", "tnd", (2, 4, 1), 8, (129, 512, 127), True),
    ("tnd-sixteen-query", "tnd", (16, 2), 64, (4097, 129), True),
    ("bsnd-q32", "bsnd", (32,), 128, (512,), True),
    ("tnd-q32", "tnd", (32,), 8, (512,), True),
    ("bsnd-fd-balance", "bsnd", (4,), 8, (4097,), True),
    ("decode-empty-kv", "bsnd", (1,), 4, (0,), False),
    ("tnd-empty-q", "tnd", (0, 2), 8, (127, 129), True),
    ("tnd-partial-tail", "tnd", (1, 0, 2), 1, (0, 129, 127), True),
    ("bsnd-causal-empty-rows", "bsnd", (4,), 8, (2,), True),
    ("tnd-empty-kv", "tnd", (2, 1), 8, (0, 129), True),
    ("bsnd-mtp-fd", "bsnd", (2,), 128, (4097,), True),
    ("tnd-fast-fd", "tnd", (4,), 8, (4097,), True),
    ("decode-auto-tail", "bsnd", (1, 1, 1), 8, (129, 0, 0), False),
    ("causal-auto-tail", "bsnd", (2, 2, 2), 8, (129, 0, 0), True),
    ("decode-long-tail", "bsnd", (1, 1, 1), 8, (16385, 0, 0), False),
    ("causal-long-tail", "bsnd", (2, 2, 2), 8, (16385, 0, 0), True),
    ("tnd-disjoint", "tnd", (0, 2, 1), 8, (129, 0, 0), True),
    ("tnd-tail", "tnd", (2, 0, 1), 8, (129, 0, 0), True),
    ("tnd-long-tail", "tnd", (2, 0, 1), 8, (16385, 0, 0), True),
]


def _make_int8_case(
    layout="bsnd",
    *,
    q=(2, 2),
    k=(129, 127),
    heads=8,
    causal=True,
    splits=0,
    view="contiguous",
    pattern="uniform",
):
    batch = len(q)
    qshape = (sum(q), heads) if layout == "tnd" else (batch, max(q, default=0), heads)
    capacity = max(1, math.ceil(max(k, default=0) / 128))
    pages = batch * capacity
    raw = dict(
        qv=torch.randint(-127, 128, (*qshape, 512), dtype=torch.int8),
        q=torch.randn(*qshape, 64).to(torch.bfloat16),
        v=torch.randint(-127, 128, (pages, 128, 1, 512), dtype=torch.int8),
        k=torch.randn(pages, 128, 1, 64).to(torch.bfloat16),
        qs=torch.rand(*qshape) * 0.015 + 0.005,
        ks=torch.tensor([0.0125]),
        vs=torch.tensor([0.0175]),
        table=torch.arange(pages - 1, -1, -1, dtype=torch.int32).reshape(batch, capacity),
    )
    if pattern == "uniform":
        raw["q"].zero_()
        raw["qv"].zero_()
    elif pattern == "constant":
        # Equal keys make P quantization exact. Nonzero, distinct query/head
        # scales still affect LSE, including the source's RoPE scaling rule.
        raw["v"].copy_(torch.tensor([-8, -1, 1, 8], dtype=torch.int8).repeat(128))
        raw["k"].fill_(0.25)
        raw["q"].fill_(0.125)
        raw["qv"].copy_(torch.tensor([-2, -1, 1, 2], dtype=torch.int8).repeat(128))
        raw["qs"].copy_(torch.linspace(-0.02, 0.02, math.prod(qshape)).reshape(qshape))
        raw["ks"].fill_(0.125)
        raw["vs"].fill_(0.0625)
    elif pattern != "random":
        raise ValueError(pattern)
    device = {}
    for name, value in raw.items():
        kind = view
        if view == "query-gap":
            kind = "head-gap" if name in ("q", "qv") else "contiguous"
        elif view == "scale-gap":
            kind = "scale-head" if name == "qs" else "contiguous"
        elif view == "allocation-gap":
            kind = "offset" if name in ("q", "qv", "qs") else "contiguous"
        elif view == "page-gap":
            kind = "offset" if name in ("k", "v") else "contiguous"
        elif view == "table-gap":
            kind = "offset" if name == "table" else "contiguous"
        elif view in ("row-gap", "wide-row") and name in ("k", "v"):
            dim = value.shape[-1]
            stride = dim + 32 if view == "row-gap" else 65536
            size = max(0, pages - 1) * (128 * stride + 32) + 127 * stride + dim + 32
            storage = torch.full((size,), 117, dtype=value.dtype, device="npu")
            target = torch.as_strided(storage, value.shape, (128 * stride + 32, stride, dim, 1), 32)
            target.copy_(value)
            device[name] = target
            continue
        elif view in ("row-gap", "wide-row"):
            kind = "contiguous"
        elif view == "broadcast":
            kind = "broadcast" if name in ("k", "v") else "contiguous"
        device[name] = _view(value.to("npu"), kind)
        if kind == "broadcast":
            raw[name] = _view(value, kind)
    args = dict(
        q=device["q"],
        qv=device["qv"],
        k_cache=device["k"],
        v_cache=device["v"],
        page_table=device["table"],
        q_descale=device["qs"],
        k_descale=device["ks"],
        v_descale=device["vs"],
        cache_seqlens=torch.tensor(k, dtype=torch.int32, device="npu"),
        cu_seqlens_q=make_cu_seqlens(q).to("npu") if layout == "tnd" else None,
        max_seqlen_q=max(q, default=0) if layout == "tnd" else None,
        causal=causal,
        num_splits=splits,
    )
    return dict(
        layout=layout, q=list(q), k=list(k), heads=heads, raw=raw, args=args, pattern=pattern
    )


def _int8_metadata(case):
    args = case["args"]
    return fa3.get_scheduler_metadata(
        len(case["q"]),
        max(case["q"], default=0),
        args["page_table"].shape[1] * 128,
        case["heads"],
        1,
        64,
        args["cache_seqlens"],
        qkv_dtype=torch.int8,
        headdim_v=512,
        page_size=128,
        cu_seqlens_q=args["cu_seqlens_q"],
        causal=args["causal"],
        num_splits=args["num_splits"],
    )


def _int8_call(case, record=None, *, lse=True, **extra):
    return fa3.flash_attn_with_kvcache(
        **dict(case["args"], **extra), scheduler_metadata=record, return_softmax_lse=lse
    )


def _int8_check(case, outputs):
    # Arbitrary dynamic-P quantization has documented fixed-source math limits.
    # The numerical E2Es use uniform logits/equal keys, where the independent
    # FP64/FP32 equations remain valid without a compiled source oracle.
    assert case["pattern"] in ("uniform", "constant")
    offset = 0
    tnd = case["layout"] == "tnd"
    out, lse = (value.cpu() for value in outputs[:2])
    assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
    assert out.shape == case["args"]["qv"].shape
    raw = case["raw"]
    for b, (nq, nk) in enumerate(zip(case["q"], case["k"])):
        selection = slice(offset, offset + nq) if tnd else b
        offset += nq
        actual = out[selection]
        logs = lse[:, selection].T if tnd else lse[b].T
        if nk == 0:
            assert actual.count_nonzero() == 0
            assert torch.isneginf(logs).all()
            continue
        if nq == 0:
            continue
        refs = []
        pages = raw["table"][b].long()
        for dtype in (torch.float64, torch.float32):
            q = raw["q"][selection].to(dtype).flatten(0, 1)
            qv = raw["qv"][selection].to(dtype).flatten(0, 1)
            k = raw["k"][pages].reshape(-1, 64)[:nk].to(dtype)
            v = raw["v"][pages].reshape(-1, 512)[:nk].to(dtype)
            # Fixed CANN applies Q/K scales to the combined content/RoPE score.
            scores = (qv @ v.T + q @ k.T) * raw["qs"][selection].to(dtype).reshape(-1, 1)
            scores *= raw["ks"].to(dtype) * (576**-0.5)
            if case["args"]["causal"] and nq > 1:
                allowed = (
                    torch.arange(nk)[None, :]
                    <= torch.arange(nq).repeat_interleave(case["heads"])[:, None] + nk - nq
                )
                scores.masked_fill_(~allowed, -torch.inf)
            valid = torch.isfinite(scores).any(-1)
            expected = torch.zeros(nq * case["heads"], 512, dtype=dtype)
            expected_lse = torch.full((nq * case["heads"],), -torch.inf, dtype=dtype)
            if valid.any():
                weights = (scores[valid] - scores[valid].amax(-1, keepdim=True)).exp()
                expected[valid] = (weights @ (v * raw["vs"].to(dtype))) / weights.sum(
                    -1, keepdim=True
                )
                expected_lse[valid] = scores[valid].logsumexp(-1)
            refs.append(
                (
                    expected.reshape(nq, case["heads"], 512).to(torch.bfloat16),
                    expected_lse.reshape(nq, case["heads"]).float(),
                )
            )
        assert_fa_close(actual, refs[0][0], refs[1][0], name="INT8 MLA out")
        assert_fa_close(logs, refs[0][1], refs[1][1], name="INT8 MLA LSE")
    _check_partials(outputs, case["args"]["num_splits"], tnd)


@pytest.mark.parametrize("splits", [0, 1])
@pytest.mark.parametrize(
    "layout,q,heads,k,causal", [pytest.param(*row[1:], id=row[0]) for row in INT8_CASES]
)
def test_mla_int8_forward(splits, layout, q, heads, k, causal):
    case = _make_int8_case(layout, q=q, k=k, heads=heads, causal=causal, splits=splits)
    outputs = _int8_call(case, _int8_metadata(case))
    _int8_check(case, outputs)
    _equal(outputs, _int8_call(case))
    _equal(_int8_call(case, lse=False), outputs[0])


@pytest.mark.parametrize("layout", ["bsnd", "tnd"])
@pytest.mark.parametrize(
    "view",
    [
        "query-gap",
        "scale-gap",
        "allocation-gap",
        "page-gap",
        "row-gap",
        "wide-row",
        "broadcast",
        "table-gap",
    ],
)
def test_mla_int8_views(layout, view):
    case = _make_int8_case(layout, q=(2, 2) if layout == "bsnd" else (2, 4), view=view)
    _int8_check(case, _int8_call(case, _int8_metadata(case)))


@pytest.mark.parametrize("layout", ["bsnd", "tnd"])
@pytest.mark.parametrize("scale", ["qs", "ks", "vs", "head-scale"])
def test_mla_int8_scales(layout, scale):
    case = _make_int8_case(layout, pattern="constant", view="scale-gap")
    record = _int8_metadata(case)
    assert record.untyped_storage().data_ptr() != _int8_metadata(case).untyped_storage().data_ptr()
    _int8_check(case, _int8_call(case, record))
    name = dict(qs="q_descale", ks="k_descale", vs="v_descale", **{"head-scale": "q_descale"})[
        scale
    ]
    raw = "qs" if scale == "head-scale" else scale
    if scale == "head-scale":
        case["raw"][raw].mul_(-2)
        case["args"][name].mul_(-2)
    else:
        case["raw"][raw].zero_()
        case["args"][name].zero_()
    _int8_check(case, _int8_call(case, record))
    _equal(_int8_call(case, record), _int8_call(case))


@pytest.mark.parametrize("layout", ["bsnd", "tnd"])
@pytest.mark.parametrize("form", ["none", "integer", "clone", "offset"])
def test_mla_int8_length_and_record_forms(layout, form):
    case = _make_int8_case(layout, q=(1, 1), k=(128, 128), causal=False)
    record = _int8_metadata(case)
    expected = _int8_call(case, record)
    _int8_check(case, expected)
    if form in ("none", "integer"):
        result = _int8_call(case, record, cache_seqlens=None if form == "none" else 128)
    elif form == "clone":
        result = _int8_call(case, record.clone())
    else:
        storage = torch.empty(record.numel() + 8, dtype=record.dtype, device=record.device)
        storage[4:-4].copy_(record)
        result = _int8_call(case, storage[4:-4])
    _equal(result, expected)


@pytest.mark.parametrize("layout", ["bsnd", "tnd"])
def test_mla_int8_failed_metadata_can_be_copied(layout):
    # Invalid device lengths publish a failure record. Generation and copying
    # stay usable; consumption rejects the failure status before attention.
    case = _make_int8_case(layout)
    case["args"]["cache_seqlens"][0] = -1
    record = _int8_metadata(case)
    copied = record.clone()
    torch.npu.synchronize()
    assert record.dtype == copied.dtype == torch.int32
    assert record.shape == copied.shape and copied.device == case["args"]["q"].device


@pytest.mark.parametrize("layout", ["bsnd", "tnd"])
@pytest.mark.parametrize("empty", ["query", "batch", "pool"])
def test_mla_int8_empty(layout, empty):
    case = _make_int8_case(
        layout,
        q=() if empty == "batch" else (0, 0) if empty == "query" else (2, 2),
        k=() if empty == "batch" else (0, 0) if empty == "pool" else (129, 127),
    )
    if empty == "pool":
        case["args"]["k_cache"] = case["args"]["k_cache"][:0]
        case["args"]["v_cache"] = case["args"]["v_cache"][:0]
    outputs = _int8_call(case)
    _int8_check(case, outputs)


@pytest.mark.parametrize("layout", ["bsnd", "tnd"])
@pytest.mark.parametrize("lse", [False, True])
@pytest.mark.parametrize("metadata", [False, True], ids=["implicit", "generated"])
def test_mla_int8_fake_compile(layout, lse, metadata):
    case = _make_int8_case(layout, pattern="constant")
    _int8_check(case, _int8_call(case))
    args = dict(case["args"], return_softmax_lse=lse)

    def forward(**kwargs):
        if metadata:
            kwargs["scheduler_metadata"] = _int8_metadata(dict(case, args=kwargs))
        return fa3.flash_attn_with_kvcache(**kwargs)

    expected = forward(**args)
    metadata_shape = _int8_metadata(case).shape
    with FakeTensorMode() as mode:
        fake = {
            key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
            for key, value in args.items()
        }
        result = forward(**fake)
        actuals, refs = (result, expected) if lse else ((result,), (expected,))
        for a, b in zip(actuals, refs):
            assert (a.shape, a.stride(), a.dtype, a.device) == (
                b.shape,
                b.stride(),
                b.dtype,
                b.device,
            )
        fake_case = dict(
            case,
            args={
                key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                for key, value in case["args"].items()
            },
        )
        assert _int8_metadata(fake_case).shape == metadata_shape
    torch._dynamo.reset()
    compiled = torch.compile(forward, backend="eager", fullgraph=True, dynamic=False)
    _equal(compiled(**args), expected)
    args["v_descale"].mul_(2)
    _equal(compiled(**args), forward(**args))


@pytest.mark.parametrize("layout", ["bsnd", "tnd"])
def test_mla_int8_stream_lifetime(layout):
    case = _make_int8_case(layout, view="query-gap")
    record = _int8_metadata(case)
    expected = _int8_call(case, record)
    _int8_check(case, expected)
    ready = torch.npu.Event()
    ready.record()
    work = []
    for _ in range(3):
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            stream.wait_event(ready)
            work.append((stream, _int8_call(case, record)))
    del case, record
    gc.collect()
    pressure = [torch.empty(262144, device="npu") for _ in range(4)]
    for stream, result in work:
        stream.synchronize()
        _equal(result, expected)
    assert len({result[0].data_ptr() for _, result in work}) == 3
    del pressure


INT8_INVALID = [
    "missing-scale",
    "missing-key-scale",
    "missing-value-scale",
    "rope-dtype",
    "content-dtype",
    "scale-dtype",
    "scale-shape",
    "key-scale-shape",
    "value-scale-shape",
    "fp8-scale-shape",
    "q-scale-cpu",
    "k-scale-cpu",
    "v-scale-cpu",
    "q-scale-grad",
    "k-scale-grad",
    "v-scale-grad",
    "page-size",
    "heads",
    "query-limit",
    "noncausal",
    "split",
    "softcap",
    "window",
    "pack",
    "margin",
    "append",
    "rotary",
    "grad",
    "metadata-size",
    "length-dtype",
    "q-last-stride",
    "prefix",
]


@pytest.mark.parametrize("layout", ["bsnd", "tnd"])
@pytest.mark.parametrize("fake", [False, True], ids=["real", "fake"])
@pytest.mark.parametrize("fault", INT8_INVALID)
def test_mla_int8_rejections(layout, fake, fault):
    case = _make_int8_case(layout)
    args = dict(case["args"])
    if fault == "missing-scale":
        args["q_descale"] = None
    elif fault == "missing-key-scale":
        args["k_descale"] = None
    elif fault == "missing-value-scale":
        args["v_descale"] = None
    elif fault == "rope-dtype":
        args["q"] = args["q"].half()
    elif fault == "content-dtype":
        args["qv"] = args["qv"].bfloat16()
    elif fault == "scale-dtype":
        args["q_descale"] = args["q_descale"].bfloat16()
    elif fault == "scale-shape":
        args["q_descale"] = args["q_descale"][..., :1]
    elif fault == "key-scale-shape":
        args["k_descale"] = args["k_descale"].expand(2)
    elif fault == "value-scale-shape":
        args["v_descale"] = args["v_descale"].expand(2)
    elif fault == "fp8-scale-shape":
        args["q_descale"] = torch.ones(len(case["q"]), 1, device="npu")
    elif fault.endswith("-scale-cpu"):
        name = fault[0] + "_descale"
        args[name] = args[name].cpu()
    elif fault.endswith("-scale-grad"):
        args[fault[0] + "_descale"].requires_grad_(True)
    elif fault == "page-size":
        args["k_cache"], args["v_cache"] = args["k_cache"][:, :64], args["v_cache"][:, :64]
    elif fault == "heads":
        for key in ("q", "qv"):
            args[key] = args[key][..., :3, :]
        args["q_descale"] = args["q_descale"][..., :3]
    elif fault == "query-limit":
        if layout == "tnd":
            args["max_seqlen_q"] = 33
        else:
            for key in ("q", "qv", "q_descale"):
                shape = list(args[key].shape)
                shape[1] = 33
                args[key] = torch.empty(shape, dtype=args[key].dtype, device="npu")
    elif fault == "noncausal":
        args["causal"] = False
    elif fault in ("split", "softcap", "window", "pack", "margin"):
        name, value = dict(
            split=("num_splits", 2),
            softcap=("softcap", 2),
            window=("window_size", (64, 0)),
            pack=("pack_gqa", False),
            margin=("sm_margin", 1),
        )[fault]
        args[name] = value
    elif fault == "append":
        args["k"] = args["k_cache"]
    elif fault == "rotary":
        args["rotary_cos"] = args["q"]
    elif fault == "grad":
        args["q"].requires_grad_(True)
    elif fault == "metadata-size":
        args["scheduler_metadata"] = _int8_metadata(case)[:-1]
    elif fault == "length-dtype":
        args["cache_seqlens"] = args["cache_seqlens"].long()
    elif fault == "q-last-stride":
        q = args["q"]
        storage = torch.empty((*q.shape[:-1], 2 * q.shape[-1]), dtype=q.dtype, device=q.device)
        args["q"] = storage[..., ::2].copy_(q)
    elif fault == "prefix":
        args["cu_seqlens_q"] = (
            None if layout == "tnd" else torch.tensor([0, 2, 4], dtype=torch.int32, device="npu")
        )
    with FakeTensorMode() if fake else nullcontext() as mode:
        if fake:
            args = {
                key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                for key, value in args.items()
            }
        match = "q rank must match" if fault == "prefix" else "MLA"
        with pytest.raises(RuntimeError, match=match):
            fa3.flash_attn_with_kvcache(**args)


@pytest.mark.parametrize("fake", [False, True], ids=["real", "fake"])
@pytest.mark.parametrize(
    "fault", ["missing-max", "prefix-dtype", "prefix-shape", "prefix-stride", "nonpaged"]
)
def test_mla_int8_tnd_rejections(fake, fault):
    case = _make_int8_case("tnd")
    args = dict(case["args"])
    if fault == "missing-max":
        args["max_seqlen_q"] = None
    elif fault == "prefix-dtype":
        args["cu_seqlens_q"] = args["cu_seqlens_q"].long()
    elif fault == "prefix-shape":
        args["cu_seqlens_q"] = args["cu_seqlens_q"][:-1]
    elif fault == "prefix-stride":
        args["cu_seqlens_q"] = torch.zeros(6, dtype=torch.int32, device="npu")[::2]
    else:
        args["page_table"] = None
    with FakeTensorMode() if fake else nullcontext() as mode:
        if fake:
            args = {
                key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                for key, value in args.items()
            }
        with pytest.raises(RuntimeError, match="MLA"):
            fa3.flash_attn_with_kvcache(**args)


@pytest.mark.parametrize("layout", ["bsnd", "tnd"])
@pytest.mark.parametrize("fault", ["missing-scale", "q", "q_descale", "k_descale", "v_descale"])
def test_mla_int8_compile_rejection(layout, fault):
    case = _make_int8_case(layout)
    if fault == "missing-scale":
        case["args"]["q_descale"] = None
    else:
        case["args"][fault].requires_grad_(True)
    match = "MLA INT8.*descale" if fault == "missing-scale" else "grad"
    with pytest.raises(RuntimeError, match=match):
        _int8_call(case)
    torch._dynamo.reset()
    compiled = torch.compile(lambda: _int8_call(case), backend="eager", fullgraph=True)
    if fault != "missing-scale":
        case["args"][fault].requires_grad_(False)
        compiled()
        case["args"][fault].requires_grad_(True)
    with pytest.raises(RuntimeError, match=match):
        compiled()


@pytest.mark.parametrize("name", ["q_descale", "k_descale", "v_descale"])
def test_mla_int8_fake_scale_device(name):
    case = _make_int8_case()
    with FakeTensorMode() as mode:
        args = {
            key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
            for key, value in case["args"].items()
        }
        args[name] = torch.empty(args[name].shape, device="npu:1")
        with pytest.raises(RuntimeError, match="same NPU device"):
            fa3.flash_attn_with_kvcache(**args)


@pytest.mark.parametrize("layout", ["bsnd", "tnd"])
@pytest.mark.parametrize("fake", [False, True], ids=["real", "fake"])
def test_mla_int8_empty_accumulators_default_dtype(layout, fake):
    case = _make_int8_case(layout, splits=1)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with FakeTensorMode() if fake else nullcontext() as mode:
            if fake:
                case["args"] = {
                    key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                    for key, value in case["args"].items()
                }
            out, lse, *empty = _int8_call(case)
            assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
            for tensor in empty:
                assert tensor.shape == (0,) and tensor.dtype == torch.float64
                assert tensor.device == case["args"]["q"].device
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("layout", ["func", "tnd", "paged_bsnd"])
@pytest.mark.parametrize(
    "name", ["mla_q_scale", "mla_k_scale", "mla_v_scale", "return_accumulators"]
)
def test_mla_nonstandard_parameters_rejected(layout, name):
    case = _make_case(layout)
    with pytest.raises(TypeError, match=name):
        _call(case, **{name: None})


@pytest.mark.parametrize("layout", ["func", "tnd", "paged_bsnd", "paged_tnd"])
@pytest.mark.parametrize("mixed", [False, True], ids=["int8", "mixed-int8"])
@pytest.mark.parametrize("fake", [False, True], ids=["real", "fake"])
def test_mla_int8_unconfigured_rejected(layout, mixed, fake):
    case = _make_case(layout, dtype=torch.bfloat16)
    for name in ("qv", "v") if mixed else ("q", "k", "v", "qv"):
        case["t"][name] = case["t"][name].to(torch.int8)
    with FakeTensorMode(allow_non_fake_inputs=True) if fake else nullcontext():
        with pytest.raises(RuntimeError, match="MLA"):
            _call(case)
        if layout in CACHE_LAYOUTS:
            with pytest.raises(RuntimeError, match="MLA INT8"):
                fa3.get_scheduler_metadata(
                    len(case["q"]),
                    case["sq"],
                    case["sk"],
                    case["t"]["q"].shape[-2],
                    1,
                    64,
                    case["lengths"],
                    qkv_dtype=torch.int8,
                    headdim_v=512,
                )


@pytest.mark.parametrize("layout", ["func", "tnd", "paged_bsnd", "paged_tnd"])
def test_mla_int8_unconfigured_compile_rejected(layout):
    case = _make_case(layout, dtype=torch.bfloat16)
    case["t"]["qv"] = case["t"]["qv"].to(torch.int8)
    case["t"]["v"] = case["t"]["v"].to(torch.int8)
    torch._dynamo.reset()
    with pytest.raises(RuntimeError, match="MLA"):
        torch.compile(lambda: _call(case), backend="eager", fullgraph=True)()


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("fake", [False, True], ids=["real", "fake"])
@pytest.mark.parametrize(
    "fault",
    [
        "q-rank",
        "k-rank",
        "qv-rank",
        "qk-dim",
        "kv-shape",
        "qv-shape",
        "dtype",
        "qv-dtype",
        "v-dtype",
        "qv-device",
        "v-device",
        "q-stride",
        "k-stride",
        "v-stride",
        "qv-stride",
        "zero-heads",
        "compressed-heads",
        "qv-equal-dims",
        "missing-qv",
    ],
)
def test_mla_invalid_inputs(dtype, fake, fault):
    case = _make_case(dtype=dtype)
    args = dict(case["t"])
    if fault.endswith("-rank"):
        name = fault[:-5]
        args[name] = args[name][0]
    elif fault == "qk-dim":
        args["k"] = args["k"][..., :32]
    elif fault == "kv-shape":
        args["v"] = args["v"][:, :-1]
    elif fault == "qv-shape":
        args["qv"] = args["qv"][:, :-1]
    elif fault == "dtype":
        args = {name: value.float() for name, value in args.items()}
    elif fault.endswith("-dtype"):
        name = fault[:-6]
        args[name] = args[name].to(torch.bfloat16 if dtype == torch.float16 else torch.float16)
    elif fault.endswith("-device"):
        name = fault[:-7]
        args[name] = args[name].cpu()
    elif fault.endswith("-stride"):
        name = fault[:-7]
        value = args[name]
        args[name] = torch.empty(
            (*value.shape[:-1], value.shape[-1] * 2), dtype=dtype, device="npu"
        )[..., ::2]
    elif fault == "zero-heads":
        args["k"], args["v"] = args["k"][..., :0, :], args["v"][..., :0, :]
    elif fault == "compressed-heads":
        args["k"], args["v"] = args["k"].expand(-1, -1, 2, -1), args["v"].expand(-1, -1, 2, -1)
    elif fault == "qv-equal-dims":
        args["v"], args["qv"] = args["v"][..., :64], args["qv"][..., :64]
    else:
        args["qv"] = None
    with FakeTensorMode() if fake else nullcontext() as mode:
        if fake:
            args = {
                key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                for key, value in args.items()
            }
        with pytest.raises(RuntimeError):
            fa3.flash_attn_func(**args)


INVALID_OPTIONS = [
    ("attention_chunk", 1),
    ("num_splits", -1),
    ("num_splits", 257),
    ("pack_gqa", "invalid"),
    ("sm_margin", -1),
    ("q_descale", 1),
    ("softcap", -1.0),
    ("softcap", float("nan")),
    ("softcap", float("inf")),
    ("softcap", 1e100),
    ("softcap", 1e-100),
    ("softmax_scale", float("nan")),
    ("softmax_scale", float("inf")),
    ("softmax_scale", -float("inf")),
    ("softmax_scale", 1e100),
]


@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("fake", [False, True], ids=["real", "fake"])
@pytest.mark.parametrize(
    "layout,name,value",
    # All values reach the shared forward validation. Also guard both
    # rejection branches (scalar options / optional tensors) at every entry.
    [("bsnd", name, value) for name, value in INVALID_OPTIONS]
    + [
        ("func", "attention_chunk", 1),
        ("tnd", "attention_chunk", 1),
        ("func", "q_descale", 1),
        ("tnd", "q_descale", 1),
    ],
)
def test_mla_unsupported_options(representation, layout, fake, name, value):
    case = _make_case(layout, representation)
    with FakeTensorMode() if fake else nullcontext() as mode:
        if fake:
            case = {
                key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                for key, value in case.items()
            }
            case["t"] = {key: mode.from_tensor(value) for key, value in case["t"].items()}
        with pytest.raises(RuntimeError):
            _call(case, **{name: value})


@pytest.mark.parametrize("fake", [False, True], ids=["real", "fake"])
@pytest.mark.parametrize(
    "layout,name",
    # Unsupported cache updates are rejected before layout dispatch. Keep every
    # argument and one append rejection for each additional cache layout.
    [
        ("bsnd", "k"),
        ("bsnd", "v"),
        ("bsnd", "rotary_cos"),
        ("bsnd", "rotary_sin"),
        ("bsnd", "cache_batch_idx"),
        ("bsnd", "cache_leftpad"),
        ("bsnd", "cu_seqlens_k_new"),
        ("mixed", "k"),
        ("paged_bsnd", "k"),
        ("paged_tnd", "k"),
    ],
)
def test_mla_cache_update_rejected(layout, fake, name):
    case = _make_case(layout)
    args = dict(
        q=case["t"]["q"],
        qv=case["t"]["qv"],
        k_cache=case["t"]["k"],
        v_cache=case["t"]["v"],
        cache_seqlens=case["lengths"],
        page_table=case["table"],
        cu_seqlens_q=case["prefix"],
        max_seqlen_q=case["sq"] if case["prefix"] is not None else None,
    )
    args[name] = case["t"]["q"]
    with FakeTensorMode() if fake else nullcontext() as mode:
        if fake:
            args = {
                key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                for key, value in args.items()
            }
        with pytest.raises(RuntimeError, match=name):
            fa3.flash_attn_with_kvcache(**args)


@pytest.mark.parametrize("execution", ["real", "fake", "compile"])
@pytest.mark.parametrize("layout", ["func", "tnd", "bsnd"])
@pytest.mark.parametrize(
    "representation,name",
    [("compressed", name) for name in ("q", "qv", "k", "v")]
    + [("prefill", name) for name in ("q", "k", "v")],
)
@pytest.mark.parametrize("inference", [False, True])
def test_mla_grad_rejected(execution, layout, representation, name, inference):
    # requires_grad rejection is dtype-independent, including inference mode.
    case = _make_case(layout, representation)
    case["t"][name].requires_grad_(True)
    with FakeTensorMode() if execution == "fake" else nullcontext() as mode:
        if execution == "fake":
            case = {
                key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                for key, value in case.items()
            }
            case["t"] = {key: mode.from_tensor(value) for key, value in case["t"].items()}
        with torch.inference_mode(inference):
            if execution == "compile":
                torch._dynamo.reset()
                case["t"][name].requires_grad_(False)
                compiled = torch.compile(lambda: _call(case), backend="eager", fullgraph=True)
                compiled()
                case["t"][name].requires_grad_(True)
            with pytest.raises(RuntimeError, match="grad"):
                if execution == "compile":
                    compiled()
                else:
                    _call(case)


@pytest.mark.parametrize("layout", CACHE_LAYOUTS)
@pytest.mark.parametrize("fake", [False, True], ids=["real", "fake"])
@pytest.mark.parametrize(
    "fault",
    [
        "metadata-size",
        "metadata-dtype",
        "metadata-stride",
        "length-size",
        "length-dtype",
        "length-rank",
        "length-stride",
        "length-device",
    ],
)
def test_mla_metadata_and_length_rejections(layout, fake, fault):
    case = _make_case(layout)
    record = _metadata(case)
    if fault == "metadata-size":
        record = record[:-1]
    elif fault == "metadata-dtype":
        record = record.long()
    elif fault == "metadata-stride":
        record = record.repeat_interleave(2)[::2]
    elif fault == "length-size":
        case["lengths"] = case["lengths"][:1]
    elif fault == "length-dtype":
        case["lengths"] = case["lengths"].long()
    elif fault == "length-rank":
        case["lengths"] = case["lengths"].unsqueeze(0)
    elif fault == "length-stride":
        case["lengths"] = case["lengths"].repeat_interleave(2)[::2]
    else:
        case["lengths"] = case["lengths"].cpu()
    with FakeTensorMode() if fake else nullcontext() as mode:
        if fake:
            case = {
                key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                for key, value in case.items()
            }
            case["t"] = {key: mode.from_tensor(value) for key, value in case["t"].items()}
            record = mode.from_tensor(record)
        with pytest.raises(RuntimeError, match="metadata|length|cache_seqlens|seqused_k"):
            _call(case, record)


@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("layout", ["tnd", "mixed", "paged_tnd"])
@pytest.mark.parametrize("fake", [False, True], ids=["real", "fake"])
@pytest.mark.parametrize("fault", ["dtype", "size", "stride", "device", "maximum"])
def test_mla_prefix_rejections(representation, layout, fake, fault):
    case = _make_case(layout, representation)
    if fault == "dtype":
        case["prefix"] = case["prefix"].long()
    elif fault == "size":
        case["prefix"] = case["prefix"][:-1]
    elif fault == "stride":
        case["prefix"] = case["prefix"].repeat_interleave(2)[::2]
    elif fault == "device":
        case["prefix"] = case["prefix"].cpu()
    else:
        case["sq"] = -1
    with FakeTensorMode() if fake else nullcontext() as mode:
        if fake:
            case = {
                key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                for key, value in case.items()
            }
            case["t"] = {key: mode.from_tensor(value) for key, value in case["t"].items()}
        with pytest.raises(RuntimeError):
            _call(case)


@pytest.mark.parametrize("layout", CACHE_LAYOUTS)
def test_mla_metadata_device_only(layout, monkeypatch):
    case = _make_case(layout)

    def forbidden(*args, **kwargs):
        raise AssertionError("MLA read a device tensor back to the host")

    with monkeypatch.context() as patch:
        for name in ("cpu", "item", "tolist", "numpy"):
            patch.setattr(torch.Tensor, name, forbidden)
        record = _metadata(case)
        outputs = _call(case, record)
    _check(case, outputs)


@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("layout", ["func", "tnd", "bsnd"])
def test_mla_default_options(representation, layout):
    case = _make_case(layout, representation, splits=0 if layout == "bsnd" else 1)
    expected = _call(case)
    _check(case, expected)
    case["options"].pop("num_splits")
    case["options"].pop("softmax_scale")
    _equal(_call(case), expected)
    torch._dynamo.reset()
    _equal(torch.compile(lambda: _call(case), backend="eager", fullgraph=True)(), expected)


@pytest.mark.parametrize("representation", ["compressed", "prefill", "int8"])
@pytest.mark.parametrize("fake", [False, True], ids=["real", "fake"])
@pytest.mark.parametrize(
    "fault",
    [
        "length-dtype",
        "length-rank",
        "length-size",
        "length-stride",
        "length-device",
        "prefix-dtype",
        "prefix-size",
        "prefix-stride",
        "prefix-device",
        "negative-batch",
        "negative-query",
        "large-kv",
        "zero-heads",
        "kv-heads",
        "dtype",
        "page-zero",
        "page-negative",
        "split",
    ],
)
def test_mla_metadata_input_rejections(representation, fake, fault):
    args = dict(
        batch_size=2,
        max_seqlen_q=1,
        max_seqlen_k=128,
        num_heads_q=4,
        num_heads_kv=1,
        headdim=192 if representation == "prefill" else 64,
        headdim_v=128 if representation == "prefill" else 512,
        qkv_dtype=torch.int8 if representation == "int8" else torch.bfloat16,
        cache_seqlens=torch.full((2,), 128, dtype=torch.int32, device="npu"),
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32, device="npu"),
        page_size=128,
        num_splits=1,
    )
    if fault.startswith(("length-", "prefix-")):
        key = "cache_seqlens" if fault.startswith("length-") else "cu_seqlens_q"
        tensor = args[key]
        args[key] = (
            tensor.float()
            if fault.endswith("dtype")
            else tensor[None]
            if fault.endswith("rank")
            else tensor[:-1]
            if fault.endswith("size")
            else tensor.repeat_interleave(2)[::2]
            if fault.endswith("stride")
            else tensor.cpu()
        )
    else:
        name, value = {
            "negative-batch": ("batch_size", -1),
            "negative-query": ("max_seqlen_q", -1),
            "large-kv": ("max_seqlen_k", 2**32),
            "zero-heads": ("num_heads_q", 0),
            "kv-heads": ("num_heads_kv", 3),
            "dtype": ("qkv_dtype", torch.float32),
            "page-zero": ("page_size", 0),
            "page-negative": ("page_size", -1),
            "split": ("num_splits", -1),
        }[fault]
        args[name] = value
    with FakeTensorMode() if fake else nullcontext() as mode:
        if fake:
            args = {
                name: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
                for name, value in args.items()
            }
        match = "(cache_seqlens|seqlens_q).*NPU" if fault.endswith("-device") else None
        with pytest.raises(RuntimeError, match=match):
            fa3.get_scheduler_metadata(**args)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["func", "bsnd"], ids=["func", "kvcache"])
def test_mla_prefill_bnsd_storage(dtype, layout):
    case = _make_case(layout, "prefill", dtype, q=(7, 3), k=(65, 33))
    # Keep BSND logical axes while matching ordinary FA3's BNSD storage pattern.
    for name, tensor in case["t"].items():
        tensor = tensor.transpose(1, 2).contiguous().transpose(1, 2)
        assert not tensor.is_contiguous()
        case["t"][name] = tensor
    _check(case, _call(case))
