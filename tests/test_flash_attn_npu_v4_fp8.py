# Copyright (c) 2026, Minghua Shen.
"""950 V4 FP8: 137 cases; all check O and LSE.

Run from the repository root:
    python -m pytest tests/test_flash_attn_npu_v4_fp8.py -v
"""

from dataclasses import replace

import pytest
import torch
import torch_npu

if not torch_npu.npu.is_available() or "Ascend950" not in torch_npu.npu.get_device_name():
    pytest.skip("FP8 forward tests require Ascend950", allow_module_level=True)

from tests.common.compare import assert_fa_close
from tests.common.attention_ref_fp8 import (
    read_inputs,
    ref_flash_attention_pair,
    visible,
)

pytestmark = pytest.mark.usefixtures("fp8_cpu_threads")


# dense: MHA/GQA/MQA, three masks, softcap, full and tail tiles.
# mask, q_lengths, kv_lengths, kv_heads, group, softcap, check_no_lse, layout
test_cases_func = [
    ("none", (64, 64), (256, 256), 2, 1, 0.0, True, "BSND"),
    ("none", (64, 64), (256, 256), 2, 3, 2.0, False, "BSND"),
    ("none", (64, 64), (256, 256), 1, 4, 0.0, False, "BSND"),
    ("causal", (64, 64), (256, 256), 2, 1, 0.0, True, "BSND"),
    ("causal", (64, 64), (256, 256), 2, 3, 2.0, False, "BSND"),
    ("causal", (64, 64), (256, 256), 1, 4, 0.0, False, "BSND"),
    ("window", (64, 64), (256, 256), 2, 1, 0.0, True, "BSND"),
    ("window", (64, 64), (256, 256), 2, 3, 2.0, False, "BSND"),
    ("window", (64, 64), (256, 256), 1, 4, 0.0, False, "BSND"),
    ("none", (128, 128), (512, 512), 2, 2, 10.0, False, "BSND"),
    ("causal", (128, 128), (512, 512), 2, 2, 10.0, False, "BSND"),
    ("window", (128, 128), (512, 512), 2, 2, 10.0, False, "BSND"),
    ("none", (63,), (127,), 2, 1, 0.0, False, "BNSD"),
    ("none", (65, 65), (129, 129), 2, 3, 2.0, False, "BNSD"),
    ("none", (129, 129), (257, 257), 1, 4, 0.0, False, "BNSD"),
    ("none", (33, 33), (257, 257), 3, 2, 10.0, False, "BNSD"),
    ("none", (64, 64), (255, 255), 2, 1, 0.0, False, "BNSD"),
    ("none", (127, 127), (256, 256), 2, 3, 2.0, False, "BNSD"),
    ("none", (128, 128), (257, 257), 1, 4, 0.0, False, "BNSD"),
    ("none", (129, 129), (385, 385), 2, 2, 10.0, False, "BNSD"),
    ("causal", (63,), (127,), 2, 1, 0.0, False, "BNSD"),
    ("causal", (65, 65), (129, 129), 2, 3, 2.0, False, "BNSD"),
    ("causal", (129, 129), (257, 257), 1, 4, 0.0, False, "BNSD"),
    ("causal", (33, 33), (257, 257), 3, 2, 10.0, False, "BNSD"),
    ("causal", (64, 64), (255, 255), 2, 1, 0.0, False, "BNSD"),
    ("causal", (127, 127), (256, 256), 2, 3, 2.0, False, "BNSD"),
    ("causal", (128, 128), (257, 257), 1, 4, 0.0, False, "BNSD"),
    ("causal", (129, 129), (385, 385), 2, 2, 10.0, False, "BNSD"),
    ("window", (63,), (127,), 2, 1, 0.0, False, "BNSD"),
    ("window", (65, 65), (129, 129), 2, 3, 2.0, False, "BNSD"),
    ("window", (129, 129), (257, 257), 1, 4, 0.0, False, "BNSD"),
    ("window", (33, 33), (257, 257), 3, 2, 10.0, False, "BNSD"),
    ("window", (64, 64), (255, 255), 2, 1, 0.0, False, "BNSD"),
    ("window", (127, 127), (256, 256), 2, 3, 2.0, False, "BNSD"),
    ("window", (128, 128), (257, 257), 1, 4, 0.0, False, "BNSD"),
    ("window", (129, 129), (385, 385), 2, 2, 10.0, False, "BNSD"),
]

# varlen: MHA/GQA/MQA, three masks, softcap and full tiles.
# mask, q_lengths, kv_lengths, kv_heads, group, softcap, check_no_lse
test_cases_varlen = [
    ("none", (33, 65), (129, 257), 2, 1, 0.0, True),
    ("none", (33, 65), (129, 257), 2, 3, 2.0, False),
    ("none", (33, 65), (129, 257), 1, 4, 0.0, False),
    ("causal", (33, 65), (129, 257), 2, 1, 0.0, True),
    ("causal", (33, 65), (129, 257), 2, 3, 2.0, False),
    ("causal", (33, 65), (129, 257), 1, 4, 0.0, False),
    ("window", (33, 65), (129, 257), 2, 1, 0.0, True),
    ("window", (33, 65), (129, 257), 2, 3, 2.0, False),
    ("window", (33, 65), (129, 257), 1, 4, 0.0, False),
    ("none", (128, 128), (512, 512), 2, 2, 10.0, False),
    ("causal", (128, 128), (512, 512), 2, 2, 10.0, False),
    ("window", (128, 128), (512, 512), 2, 2, 10.0, False),
]

# cache: MHA/GQA/MQA, three masks, softcap and full tiles.
# mask, q_lengths, kv_lengths, kv_heads, group, softcap, check_no_lse, layout
test_cases_kvcache = [
    ("none", (33, 65), (129, 257), 2, 1, 0.0, True, "BSND"),
    ("none", (33, 65), (129, 257), 2, 3, 2.0, False, "BSND"),
    ("none", (33, 65), (129, 257), 1, 4, 0.0, False, "BSND"),
    ("causal", (33, 65), (129, 257), 2, 1, 0.0, True, "BSND"),
    ("causal", (33, 65), (129, 257), 2, 3, 2.0, False, "BSND"),
    ("causal", (33, 65), (129, 257), 1, 4, 0.0, False, "BSND"),
    ("window", (33, 65), (129, 257), 2, 1, 0.0, True, "BSND"),
    ("window", (33, 65), (129, 257), 2, 3, 2.0, False, "BSND"),
    ("window", (33, 65), (129, 257), 1, 4, 0.0, False, "BSND"),
    ("none", (128, 128), (512, 512), 2, 2, 10.0, False, "BSND"),
    ("causal", (128, 128), (512, 512), 2, 2, 10.0, False, "BSND"),
    ("window", (128, 128), (512, 512), 2, 2, 10.0, False, "BSND"),
]

# Four page sizes, three masks, host and metadata scheduling.
# page_size, mask, metadata, q_lengths, kv_lengths, softcap
test_cases_paged = [
    (128, "none", False, (65, 129), (129, 257), 0.0),
    (128, "none", True, (65, 129), (129, 257), 2.0),
    (128, "causal", False, (65, 129), (129, 257), 0.0),
    (128, "causal", True, (65, 129), (129, 257), 2.0),
    (128, "window", False, (65, 129), (129, 257), 0.0),
    (128, "window", True, (65, 129), (129, 257), 2.0),
    (256, "none", False, (65, 129), (257, 513), 0.0),
    (256, "none", True, (65, 129), (257, 513), 2.0),
    (256, "causal", False, (65, 129), (257, 513), 0.0),
    (256, "causal", True, (65, 129), (257, 513), 2.0),
    (256, "window", False, (65, 129), (257, 513), 0.0),
    (256, "window", True, (65, 129), (257, 513), 2.0),
    (512, "none", False, (65, 129), (513, 1025), 0.0),
    (512, "none", True, (65, 129), (513, 1025), 2.0),
    (512, "causal", False, (65, 129), (513, 1025), 0.0),
    (512, "causal", True, (65, 129), (513, 1025), 2.0),
    (512, "window", False, (65, 129), (513, 1025), 0.0),
    (512, "window", True, (65, 129), (513, 1025), 2.0),
    (1024, "none", False, (65, 129), (1025, 2049), 0.0),
    (1024, "none", True, (65, 129), (1025, 2049), 2.0),
    (1024, "causal", False, (65, 129), (1025, 2049), 0.0),
    (1024, "causal", True, (65, 129), (1025, 2049), 2.0),
    (1024, "window", False, (65, 129), (1025, 2049), 0.0),
    (1024, "window", True, (65, 129), (1025, 2049), 2.0),
]

# Short Q, long KV, automatic and explicit splits; compare with splits=1.
# mask, q_lengths, kv_lengths, group, num_splits, softcap
test_cases_fd = [
    ("none", (1, 1), (2048, 4096), 4, 0, 0.0),
    ("none", (1, 1), (2048, 4096), 4, 3, 0.0),
    ("none", (8, 8), (2048, 4096), 4, 0, 2.0),
    ("none", (8, 8), (2048, 4096), 4, 3, 2.0),
    ("none", (16, 16), (2048, 4096), 2, 0, 0.0),
    ("none", (16, 16), (2048, 4096), 2, 3, 0.0),
    ("causal", (1, 1), (2048, 4096), 4, 0, 0.0),
    ("causal", (1, 1), (2048, 4096), 4, 3, 0.0),
    ("causal", (8, 8), (2048, 4096), 4, 0, 2.0),
    ("causal", (8, 8), (2048, 4096), 4, 3, 2.0),
    ("causal", (16, 16), (2048, 4096), 2, 0, 0.0),
    ("causal", (16, 16), (2048, 4096), 2, 3, 0.0),
]

# Q/KV tile boundaries, with unmasked and causal/window coverage.
# q_lengths, kv_lengths, mask
test_cases_tail = [
    ((63,), (127,), "none"),
    ((63,), (127,), "causal"),
    ((64,), (128,), "none"),
    ((64,), (128,), "causal"),
    ((65,), (129,), "none"),
    ((65,), (129,), "causal"),
    ((127,), (255,), "none"),
    ((127,), (255,), "window"),
    ((128,), (256,), "none"),
    ((128,), (256,), "window"),
    ((129,), (257,), "none"),
    ((129,), (257,), "window"),
]

# Optional scales, per-batch/head scales, and zero scales.
# q_lengths, kv_lengths, omit_descales, zero_descale
test_cases_descale = [
    ((65, 129), (257, 385), ("q_descale",), ""),
    ((65, 129), (257, 385), ("k_descale",), ""),
    ((65, 129), (257, 385), ("v_descale",), ""),
    ((65, 129), (257, 385), ("q_descale", "k_descale", "v_descale"), ""),
    ((33, 65, 129), (129, 257, 513), (), ""),
    ((65, 129), (257, 385), (), "q_descale"),
    ((65, 129), (257, 385), (), "k_descale"),
    ((65, 129), (257, 385), (), "v_descale"),
]

# Single key, zero V, empty rows, and asymmetric window behavior.
# entry, mask, q_lengths, kv_lengths, window_size, zero_v
test_cases_semantics = [
    ("dense", "none", (1,), (1,), (33, 17), False),
    ("cache", "window", (65, 129), (257, 385), (33, 17), True),
    ("varlen", "causal", (129, 65), (1, 33), (33, 17), False),
    ("dense", "window", (8,), (4,), (-1, 4), False),
    ("varlen", "window", (512, 257), (32, 17), (8, -1), False),
    ("dense", "window", (65, 65), (257, 257), (-1, 8), False),
]


def prepare_v4_forward(kwargs, entry, metadata=False):
    from flash_attn_npu_4 import flash_attn_npu_interface_950 as fa

    q = kwargs["q"]
    k = kwargs["k_cache" if entry == "cache" else "k"]
    v = kwargs["v_cache" if entry == "cache" else "v"]
    cuq, cuk = kwargs.get("cu_seqlens_q"), kwargs.get("cu_seqlens_k")
    table = kwargs.get("page_table")
    original_shape = None
    used_k = kwargs.get("cache_seqlens") if entry == "cache" else None
    if entry == "cache" and q.dim() == 4:
        original_shape = q.shape
        batch, q_len, heads, dim = q.shape
        q = q.reshape(-1, heads, dim)
        cuq = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * q_len
        if table is None and k.dim() == 4:
            kv_len = k.shape[1]
            k, v = k.flatten(0, 1), v.flatten(0, 1)
            cuk = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * kv_len
    return_lse = (
        kwargs.get("return_softmax_lse", False)
        if entry == "cache"
        else kwargs.get("return_attn_probs", False)
    )
    window_left, window_right = kwargs.get("window_size", (-1, -1))
    used_q = None
    if cuq is None:
        q_max, k_max = q.shape[1], k.shape[1]
        q_heads, kv_heads = q.shape[2], k.shape[2]
        layout, q, k, v, _ = fa._detect_layout(q, k, v, table, cuk)
        used_k = torch.full((q.shape[0],), k_max, dtype=torch.int32, device=q.device)
    else:
        layout = fa._LAYOUT_TND
        used_q = cuq[1:] - cuq[:-1]
        q_max = kwargs.get("max_seqlen_q") or int(used_q.max().item())
        k_max = kwargs.get("max_seqlen_k")
        if used_k is None and cuk is not None:
            used_k = cuk[1:] - cuk[:-1]
        if k_max is None and metadata and used_k is not None:
            k_max = int(used_k.max().item())
        q_heads, kv_heads = q.shape[-2], k.shape[-2]

    common = dict(
        q=q,
        k=k,
        v=v,
        layout=layout,
        cu_seqlens_q=cuq,
        cu_seqlens_k=cuk,
        max_seqlen_q=q_max if cuq is not None else None,
        max_seqlen_k=k_max if cuq is not None else None,
        seqused_q=used_q,
        seqused_k=used_k,
        page_table=table,
        softmax_scale=kwargs.get("softmax_scale"),
        causal=kwargs.get("causal", False),
        window_size_left=fa._window_to_npu(window_left),
        window_size_right=fa._window_to_npu(window_right),
        softcap=kwargs.get("softcap", 0.0),
        num_splits=kwargs.get("num_splits", 1),
        q_descale=kwargs.get("q_descale"),
        k_descale=kwargs.get("k_descale"),
        v_descale=kwargs.get("v_descale"),
        return_lse=return_lse,
    )
    if metadata:
        common["scheduler_metadata"] = fa.get_scheduler_metadata(
            max_seqlen_q=q_max,
            max_seqlen_k=k_max,
            nheads=q_heads,
            nheads_kv=kv_heads,
            headdim=q.shape[-1],
            headdim_v=v.shape[-1],
            num_splits=common["num_splits"],
            causal=common["causal"],
            window_size_left=window_left,
            window_size_right=window_right,
            cu_seqlens_q=cuq,
            seqused_q=used_q,
            seqused_k=used_k,
            qkv_dtype=q.dtype,
        )

    # Restore the caller's layout and legacy output convention.
    def normalize(result):
        out, lse = result[0], result[1] if return_lse else None
        if original_shape is not None:
            batch, q_len, heads, dim = original_shape
            out = out.reshape(batch, q_len, heads, dim)
            if lse is not None:
                lse = lse.reshape(heads, batch, q_len).permute(1, 0, 2)
        elif entry in ("dense", "varlen") and lse is not None:
            lse = lse.transpose(-1, -2)
        return (out, lse) if return_lse else out

    return fa._flash_attn_forward, common, normalize


def run_forward(case, inputs, device):
    kwargs = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in inputs["native"].items()
    }
    flag = "return_softmax_lse" if case.entry == "cache" else "return_attn_probs"
    kwargs[flag] = case.return_lse
    kwargs["num_splits"] = case.num_splits
    function, prepared, normalize = prepare_v4_forward(kwargs, case.entry, case.metadata)
    result = normalize(function(**prepared))
    output = (result[0] if case.return_lse else result).cpu()
    lse = result[1].cpu() if case.return_lse else None
    if lse is not None and case.entry in ("dense", "varlen"):
        lse = lse.transpose(-1, -2)
    if case.tnd:
        return output, lse
    output = torch.cat([output[b, :n] for b, n in enumerate(case.q_lengths)])
    if lse is not None:
        lse = torch.cat([lse[b, :, :n] for b, n in enumerate(case.q_lengths)], dim=1)
    return output, lse


def make_inputs(
    entry,
    mask,
    q_lengths,
    kv_lengths,
    kv_heads,
    group,
    page_size,
    softcap,
    num_splits,
    window_size,
    omit_descales,
    zero_descale,
    zero_v,
):
    batch = len(q_lengths)
    q_heads = kv_heads * group
    generator = torch.Generator().manual_seed(20261028)

    def random_fp8(shape):
        return (torch.rand(shape, generator=generator) * 10 - 5).to(torch.float8_e4m3fn)

    tnd = entry == "varlen" or page_size != 0 or (entry == "cache" and len(set(q_lengths)) > 1)
    q_shape = (sum(q_lengths), q_heads, 128) if tnd else (batch, q_lengths[0], q_heads, 128)
    q = random_fp8(q_shape)
    table = None
    if page_size:
        pages_per_batch = (max(kv_lengths) + page_size - 1) // page_size
        pages = batch * pages_per_batch
        kv_shape = (pages, page_size, kv_heads, 128)
        table = torch.arange(pages - 1, -1, -1, dtype=torch.int32).reshape(batch, pages_per_batch)
    elif tnd:
        kv_shape = (sum(kv_lengths), kv_heads, 128)
    else:
        kv_shape = (batch, kv_lengths[0], kv_heads, 128)
    k, v = random_fp8(kv_shape), random_fp8(kv_shape)
    if zero_v:
        v.zero_()

    def cumulative(lengths):
        return torch.tensor((0, *lengths), dtype=torch.int32).cumsum(0).int()

    kwargs = dict(
        q=q,
        softmax_scale=128**-0.5,
        softcap=softcap,
        causal=mask == "causal",
        window_size=window_size if mask == "window" else (-1, -1),
        num_splits=num_splits,
        return_attn_probs=True,
        return_softmax_lse=True,
        cu_seqlens_q=cumulative(q_lengths) if tnd else None,
        cu_seqlens_k=cumulative(kv_lengths) if entry == "varlen" else None,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(kv_lengths),
        page_table=table,
        cache_seqlens=torch.tensor(kv_lengths, dtype=torch.int32),
    )
    kwargs.update(
        {
            "k_cache" if entry == "cache" else "k": k,
            "v_cache" if entry == "cache" else "v": v,
        }
    )
    for name in ("q_descale", "k_descale", "v_descale"):
        value = torch.rand(batch, kv_heads, generator=generator)
        if name == zero_descale:
            value.zero_()
        kwargs[name] = None if name in omit_descales else value
    return kwargs


def run_case(
    entry="cache",
    mask="none",
    q_lengths=(65, 129),
    kv_lengths=(257, 385),
    kv_heads=2,
    group=2,
    page_size=0,
    metadata=False,
    num_splits=1,
    window_size=(33, 17),
    softcap=0.0,
    omit_descales=(),
    zero_descale="",
    zero_v=False,
    check_no_lse=False,
    layout=None,
):
    kwargs = make_inputs(
        entry,
        mask,
        q_lengths,
        kv_lengths,
        kv_heads,
        group,
        page_size,
        softcap,
        num_splits,
        window_size,
        omit_descales,
        zero_descale,
        zero_v,
    )
    case, inputs = read_inputs(kwargs, entry, metadata)
    if layout == "BNSD":
        expected = run_forward(case, inputs, "npu:0")
        for name in (
            "q",
            "k_cache" if entry == "cache" else "k",
            "v_cache" if entry == "cache" else "v",
        ):
            tensor = inputs["native"][name]
            inputs["native"][name] = tensor.transpose(1, 2).contiguous().transpose(1, 2)
        from flash_attn_npu_4 import flash_attn_npu_interface_950 as interface

        backend = interface.flash_attn_npu_4_950
        forward = backend.fwd
        layouts = []

        def capture(*args, **kwargs):
            layouts.append(args[-4])
            return forward(*args, **kwargs)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(backend, "fwd", capture)
            actual = run_forward(case, inputs, "npu:0")
        assert layouts and all(value == interface._LAYOUT_BNSD for value in layouts)
        for result, reference in zip(actual, expected):
            assert torch.equal(result, reference)
    else:
        actual = run_forward(case, inputs, "npu:0")
    high, low = ref_flash_attention_pair(case, inputs)
    total_q = sum(q_lengths)
    assert actual[0].shape == (total_q, kv_heads * group, 128)
    assert actual[0].dtype == torch.bfloat16
    assert actual[1].shape == (kv_heads * group, total_q)
    assert actual[1].dtype == torch.float32
    valid_rows = torch.cat(
        [
            visible(case, b, 0, q, 0, kv).any(-1)
            for b, (q, kv) in enumerate(zip(q_lengths, kv_lengths))
        ]
    )
    assert not actual[0][~valid_rows].count_nonzero()
    if zero_v or zero_descale == "v_descale":
        assert not actual[0].count_nonzero()
    for name, result, reference, benchmark in zip(("O", "LSE"), actual, high, low):
        assert_fa_close(result, reference, benchmark, softcap=case.softcap, name=name)
    if num_splits != 1:
        # Check the requested split path against an unsplit invocation as well.
        unsplit = run_forward(replace(case, num_splits=1), inputs, "npu:0")
        for name, result, reference, benchmark in zip(("O", "LSE"), unsplit, high, low):
            assert_fa_close(
                result, reference, benchmark, softcap=case.softcap, name=f"unsplit {name}"
            )
        for name, result, reference, benchmark in zip(("O", "LSE"), actual, high, unsplit):
            assert_fa_close(
                result, reference, benchmark, softcap=case.softcap, name=f"split vs unsplit {name}"
            )
    if metadata:
        direct = run_forward(replace(case, metadata=False), inputs, "npu:0")
        for result, reference in zip(actual, direct):
            assert torch.equal(result, reference)
    if check_no_lse:
        without_lse = run_forward(replace(case, return_lse=False), inputs, "npu:0")[0]
        assert torch.equal(without_lse, actual[0])


@pytest.mark.parametrize(
    "mask, q_lengths, kv_lengths, kv_heads, group, softcap, check_no_lse, layout",
    test_cases_func,
)
def test_fp8_func(mask, q_lengths, kv_lengths, kv_heads, group, softcap, check_no_lse, layout):
    run_case(
        entry="dense",
        mask=mask,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        kv_heads=kv_heads,
        group=group,
        softcap=softcap,
        check_no_lse=check_no_lse,
        layout=layout,
    )


@pytest.mark.parametrize(
    "mask, q_lengths, kv_lengths, kv_heads, group, softcap, check_no_lse",
    test_cases_varlen,
)
def test_fp8_varlen(mask, q_lengths, kv_lengths, kv_heads, group, softcap, check_no_lse):
    run_case(
        entry="varlen",
        mask=mask,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        kv_heads=kv_heads,
        group=group,
        softcap=softcap,
        check_no_lse=check_no_lse,
    )


@pytest.mark.parametrize(
    "mask, q_lengths, kv_lengths, kv_heads, group, softcap, check_no_lse, layout",
    test_cases_kvcache,
)
def test_fp8_kvcache(mask, q_lengths, kv_lengths, kv_heads, group, softcap, check_no_lse, layout):
    run_case(
        entry="cache",
        mask=mask,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        kv_heads=kv_heads,
        group=group,
        softcap=softcap,
        check_no_lse=check_no_lse,
        layout=layout,
    )


@pytest.mark.parametrize(
    "page_size, mask, metadata, q_lengths, kv_lengths, softcap",
    test_cases_paged,
)
def test_fp8_paged(page_size, mask, metadata, q_lengths, kv_lengths, softcap):
    run_case(
        page_size=page_size,
        mask=mask,
        metadata=metadata,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        softcap=softcap,
    )


@pytest.mark.parametrize(
    "mask, q_lengths, kv_lengths, group, num_splits, softcap",
    test_cases_fd,
)
@pytest.mark.parametrize("metadata", [False, True])
def test_fp8_flash_decode(mask, q_lengths, kv_lengths, group, num_splits, softcap, metadata):
    run_case(
        mask=mask,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        group=group,
        num_splits=num_splits,
        softcap=softcap,
        kv_heads=1,
        page_size=128,
        metadata=metadata,
        check_no_lse=True,
    )


@pytest.mark.parametrize(
    "q_lengths, kv_lengths, mask",
    test_cases_tail,
)
def test_fp8_tail(q_lengths, kv_lengths, mask):
    run_case(
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        mask=mask,
        entry="varlen",
        kv_heads=1,
    )


@pytest.mark.parametrize(
    "q_lengths, kv_lengths, omit_descales, zero_descale",
    test_cases_descale,
)
def test_fp8_descale(q_lengths, kv_lengths, omit_descales, zero_descale):
    run_case(
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        omit_descales=omit_descales,
        zero_descale=zero_descale,
    )


@pytest.mark.parametrize(
    "entry, mask, q_lengths, kv_lengths, window_size, zero_v",
    test_cases_semantics,
)
def test_fp8_semantics(entry, mask, q_lengths, kv_lengths, window_size, zero_v):
    run_case(
        entry=entry,
        mask=mask,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        window_size=window_size,
        zero_v=zero_v,
    )


@pytest.mark.parametrize("view_kind", ["strided_head", "strided_batch", "broadcast_batch"])
def test_fp8_descale_views(view_kind):
    from flash_attn_npu_4.flash_attn_npu_interface_950 import _flash_attn_forward

    torch.manual_seed(0)
    q = torch.randn(2, 65, 4, 128, device="npu:0").to(torch.float8_e4m3fn)
    k = torch.randn(2, 129, 2, 128, device="npu:0").to(torch.float8_e4m3fn)
    v = torch.randn_like(k.float()).to(torch.float8_e4m3fn)
    scales = torch.rand(4, 4, device="npu:0")
    views = {
        "strided_head": scales[::2, ::2],
        "strided_batch": scales[::2, :2],
        "broadcast_batch": scales[:1, :2].expand(2, 2),
    }
    descale = views[view_kind]
    assert not descale.is_contiguous()
    seqused_k = torch.full((2,), 129, dtype=torch.int32, device=q.device)
    actual = _flash_attn_forward(
        q,
        k,
        v,
        seqused_k=seqused_k,
        q_descale=descale,
        k_descale=descale,
        v_descale=descale,
        return_lse=True,
    )
    contiguous = descale.contiguous()
    expected = _flash_attn_forward(
        q,
        k,
        v,
        seqused_k=seqused_k,
        q_descale=contiguous,
        k_descale=contiguous,
        v_descale=contiguous,
        return_lse=True,
    )
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
