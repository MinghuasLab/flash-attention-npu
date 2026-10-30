# Copyright (c) 2026, Minghua Shen.
"""96 fixed 950 V3 FP8 forward cases; all check O and LSE.

Run from the repository root:
    python -m pytest tests/test_flash_attn_npu_v3_fp8.py -v
"""

from dataclasses import replace
import inspect

import pytest
import torch
import torch_npu

if not torch_npu.npu.is_available() or "Ascend950" not in torch_npu.npu.get_device_name():
    pytest.skip("FP8 forward tests require Ascend950", allow_module_level=True)

from tests.common.compare import assert_fa_close
from tests.common.attention_ref_fp8 import (
    high_reference,
    low_reference,
    read_inputs,
    visible,
)


# dense: MHA/GQA/MQA, three masks, softcap and full tiles.
# mask, q_lengths, kv_lengths, kv_heads, group, softcap, check_no_lse
test_cases_func = [
    ("none", (64, 64), (256, 256), 2, 1, 0.0, True),
    ("none", (64, 64), (256, 256), 2, 3, 2.0, False),
    ("none", (64, 64), (256, 256), 1, 4, 0.0, False),
    ("causal", (64, 64), (256, 256), 2, 1, 0.0, True),
    ("causal", (64, 64), (256, 256), 2, 3, 2.0, False),
    ("causal", (64, 64), (256, 256), 1, 4, 0.0, False),
    ("window", (64, 64), (256, 256), 2, 1, 0.0, True),
    ("window", (64, 64), (256, 256), 2, 3, 2.0, False),
    ("window", (64, 64), (256, 256), 1, 4, 0.0, False),
    ("none", (128, 128), (512, 512), 2, 2, 10.0, False),
    ("causal", (128, 128), (512, 512), 2, 2, 10.0, False),
    ("window", (128, 128), (512, 512), 2, 2, 10.0, False),
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
# mask, q_lengths, kv_lengths, kv_heads, group, softcap, check_no_lse
test_cases_kvcache = [
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

# Single key, zero V, empty rows, and the retained legacy window behavior.
# entry, mask, q_lengths, kv_lengths, window_size, zero_v
test_cases_semantics = [
    ("dense", "none", (1,), (1,), (33, 17), False),
    ("cache", "window", (65, 129), (257, 385), (33, 17), True),
    ("varlen", "causal", (129, 65), (1, 33), (33, 17), False),
    ("dense", "window", (8,), (4,), (-1, 4), False),
]


def run_forward(case, inputs, device):
    """直接传递生成的接口参数，不在这里打包 TND、分页或创建 page_table。"""
    import flash_attn_npu_3 as fa

    kwargs = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in inputs["native"].items()
    }
    if case.metadata:
        # scheduler_metadata 必须在目标 NPU 调用官方 planner 生成，不能填随机字节。
        if case.page_size:
            kv_seqlen_bound = kwargs["page_table"].shape[1] * case.page_size
        else:
            kv_seqlen_bound = kwargs["k_cache"].shape[0 if case.tnd else 1]
        kwargs["scheduler_metadata"] = fa.get_scheduler_metadata(
            len(case.q_lengths),
            max(case.q_lengths),
            kv_seqlen_bound,
            case.q_heads,
            case.kv_heads,
            128,
            kwargs["cache_seqlens"],
            qkv_dtype=kwargs["q"].dtype,
            cu_seqlens_q=kwargs["cu_seqlens_q"],
            page_size=case.page_size or None,
            causal=kwargs["causal"],
            window_size=kwargs["window_size"],
            num_splits=case.num_splits,
        )
    function = {
        "dense": fa.flash_attn_func,
        "varlen": fa.flash_attn_varlen_func,
        "cache": fa.flash_attn_with_kvcache,
    }[case.entry]
    return_flag = "return_softmax_lse" if case.entry == "cache" else "return_attn_probs"
    kwargs[return_flag] = case.return_lse
    kwargs["num_splits"] = case.num_splits
    parameters = inspect.signature(function).parameters
    result = function(**{name: value for name, value in kwargs.items() if name in parameters})
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
    actual = run_forward(case, inputs, "npu:0")
    high = high_reference(case, inputs)
    low = low_reference(case, inputs)
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
    "mask, q_lengths, kv_lengths, kv_heads, group, softcap, check_no_lse",
    test_cases_func,
)
def test_fp8_func(mask, q_lengths, kv_lengths, kv_heads, group, softcap, check_no_lse):
    run_case(
        entry="dense",
        mask=mask,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        kv_heads=kv_heads,
        group=group,
        softcap=softcap,
        check_no_lse=check_no_lse,
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
    "mask, q_lengths, kv_lengths, kv_heads, group, softcap, check_no_lse",
    test_cases_kvcache,
)
def test_fp8_kvcache(mask, q_lengths, kv_lengths, kv_heads, group, softcap, check_no_lse):
    run_case(
        entry="cache",
        mask=mask,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        kv_heads=kv_heads,
        group=group,
        softcap=softcap,
        check_no_lse=check_no_lse,
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
def test_fp8_flash_decode(mask, q_lengths, kv_lengths, group, num_splits, softcap):
    run_case(
        mask=mask,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        group=group,
        num_splits=num_splits,
        softcap=softcap,
        kv_heads=1,
        page_size=128,
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
