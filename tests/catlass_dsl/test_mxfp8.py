"""MXFP8 forward tests: golden, RNE, layouts, masks, runtime and benchmarks.

NPU cases carry explicit markers; CPU cases can be selected with ``-m "not npu"``.
The independent golden remains in mxfp8_reference.py for benchmark reuse.

Device parameters use representative combinations instead of Cartesian products:
keep each numerical boundary, layout and execution mode, then exercise their
interactions in targeted regressions. Repeated launches are reserved for cache,
stream and slot-reuse checks; the 24-head stress case retains 20 launches.
"""

import copy
import csv
import importlib.util
import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from catlass import tla
from torch._subclasses.fake_tensor import FakeTensorMode

import flash_attn_npu_4 as fa4
from flash_attn_npu_4.catlass_dsl import (
    compute_block_sparsity,
    flash_attn_func_with_quant as flash_attn_func,
    flash_attn_varlen_func_with_quant as flash_attn_varlen_func,
    graph,
    interface,
    runtime,
    simd,
    simt,
)
from flash_attn_npu_4.catlass_dsl.block_sparsity import BlockSparseTensorsTorch
from flash_attn_npu_4.catlass_dsl.builtin import causal
from flash_attn_npu_4.catlass_dsl.cache_utils import get_jit_cache
from flash_attn_npu_4.catlass_dsl.kernels.flash_fwd_mxfp8_arch35 import (
    _CPP_SHORT_QUANTIZE,
    _mx_common_label_interval,
    _mx_exact_exp_pack_update,
    _mx_exact_label_interval_quant,
    _mx_exact_label_quant,
    _mx_fill_ready,
    _mx_fill_scales,
    _mx_group_scale,
    _mx_refresh_guard,
    _mx_label_ranges,
    _mx_scatter,
    _mx_short_quantize,
    _mx_trace_affine_mask,
    online_quantize_probabilities,
    pack_scaled_rne,
    prepare_group_scales,
    scaled_group_max_readonly,
)

from .mxfp8_reference import (
    SCALE_NAMES,
    check_quantized_lse,
    decode_scale,
    decode_sequences,
    decoded_inputs,
    error_metrics,
    make_case,
    quantize_group32,
    quantize_v,
    reference,
    to_device,
)
from .test_utils import check

# Independent CPU golden and scale encoding.


@pytest.fixture
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.usefixtures("single_cpu_thread")
def test_quantized_denominator_preserves_constant_values():
    case = make_case(q_lengths=(3,), k_lengths=(129,), q_heads=1, kv_heads=1)
    q, k, v = case["qkv"]
    case["qkv"] = q, k, torch.ones(v.shape).to(v.dtype)
    case["kwargs"]["v_descale"].fill_(127)
    normalized, lse = reference(case, normalize_quantized=True)
    unnormalized, original_lse = reference(case)
    torch.testing.assert_close(normalized, torch.ones_like(normalized), atol=2e-7, rtol=0)
    assert (unnormalized - 1).abs().max() > 1e-5
    torch.testing.assert_close(lse, original_lse, atol=0, rtol=0)


@pytest.mark.usefixtures("single_cpu_thread")
@pytest.mark.parametrize("length", [0, 31, 32, 33, 63, 64, 65, 127, 128, 129])
def test_v_scales_pad_each_sequence_to64(length):
    source = torch.linspace(-1, 1, max(1, length * 128))[: length * 128].reshape(length, 1, 128)
    encoded, scale = quantize_v(source)
    assert encoded.shape == source.shape
    assert scale.shape == ((length + 63) // 64, 1, 128, 2)
    factors = decode_scale(scale).permute(0, 3, 1, 2).flatten(0, 1).repeat_interleave(32, 0)
    restored = encoded.float() * factors[:length]
    torch.testing.assert_close(restored, source, atol=0.04, rtol=0.07)
    if length % 64 <= 32 and length % 64:
        assert torch.equal(scale[-1, :, :, 1], torch.full((1, 128), 127, dtype=torch.uint8))


@pytest.mark.usefixtures("single_cpu_thread")
@pytest.mark.parametrize("exponent", [-127, -120, -20, -8, 0, 100])
def test_probability_exponent_threshold_neighbors(exponent):
    threshold = torch.tensor(448.0, dtype=torch.float32) * 2.0**exponent
    values = torch.stack(
        (
            torch.nextafter(threshold, torch.tensor(0.0)),
            threshold,
            torch.nextafter(threshold, torch.tensor(torch.inf)),
        )
    )[:, None].expand(3, 32)
    _, scales = quantize_group32(values)
    assert scales[:, 0].tolist() == [exponent + 127, exponent + 127, exponent + 128]


@pytest.mark.usefixtures("single_cpu_thread")
def test_zero_groups_and_rne_halfway_bytes():
    values = torch.zeros((3, 32))
    values[1, 0], values[2, 0] = 1, 1
    values[1, 1], values[2, 1] = 1.0625 / 256, 1.1875 / 256
    encoded, scale = quantize_group32(values)
    assert scale[:, 0].tolist() == [127, 119, 119]
    assert not encoded[0].view(torch.uint8).count_nonzero()
    assert encoded[1, 1].float() == 1.0  # tie chooses even low mantissa
    assert encoded[2, 1].float() == 1.25  # tie chooses even high mantissa


@pytest.mark.usefixtures("single_cpu_thread")
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_input_quantizer_rejects_nonfinite(value):
    with pytest.raises(ValueError, match="finite"):
        quantize_group32(torch.tensor([value]))


@pytest.mark.usefixtures("single_cpu_thread")
@pytest.mark.parametrize(
    "layout,q_packed,k_packed",
    [
        ("BSND", False, False),
        ("BNSD", False, False),
        ("BSND", True, False),
        ("BNSD", True, False),
        ("BSND", False, True),
        ("BNSD", False, True),
        ("BSND", True, True),  # TND has no BSND/BNSD axis ordering.
    ],
)
def test_online_lse_matches_independent_dense_math(layout, q_packed, k_packed):
    case = make_case(
        q_lengths=(17, 33) if q_packed else (33, 33),
        k_lengths=(65, 129) if k_packed else (129, 129),
        q_heads=2,
        kv_heads=1,
        q_packed=q_packed,
        k_packed=k_packed,
        layout=layout,
        v_scale_capacity=7 if k_packed else None,
    )
    output, lse = reference(case, scale=-0.125, quantize_p=False)
    decoded = decode_sequences(case)
    restored = decoded_inputs(case, dtype=torch.float32)
    qoff = koff = 0
    for batch, (q, k, v) in enumerate(decoded):
        for head in range(2):
            scores = q[:, head] @ k[:, 0].T * -0.125
            expected = scores.softmax(-1) @ v[:, 0]
            expected_lse = scores.logsumexp(-1)
            actual = (
                output[qoff : qoff + len(q), head]
                if q_packed
                else output[batch, :, head]
                if layout == "BSND"
                else output[batch, head]
            )
            actual_lse = lse[head, qoff : qoff + len(q)] if q_packed else lse[batch, head]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(actual_lse, expected_lse, atol=1e-6, rtol=1e-5)
        for index, value in enumerate((q, k, v)):
            packed = q_packed if index == 0 else k_packed
            offset = qoff if index == 0 else koff
            actual = (
                restored[index][offset : offset + len(value)] if packed else restored[index][batch]
            )
            if not packed and layout == "BNSD":
                actual = actual.transpose(0, 1)
            torch.testing.assert_close(actual, value, atol=0, rtol=0)
        qoff += len(q)
        koff += len(k)


@pytest.mark.usefixtures("single_cpu_thread")
def test_p_quantization_preserves_denominator_and_changes_numerator():
    case = make_case(q_lengths=(17,), k_lengths=(129,), q_heads=2, kv_heads=1)
    quantized, lse = reference(case)
    precise, precise_lse = reference(case, quantize_p=False)
    assert torch.equal(lse, precise_lse)
    assert not torch.equal(quantized, precise)
    assert torch.isfinite(quantized).all()


@pytest.mark.usefixtures("single_cpu_thread")
@pytest.mark.parametrize("first,second", [((65, 63), (64, 64)), ((31, 33), (32, 32))])
def test_packed_v_capacity_is_independent_of_segment_boundaries(first, second):
    cases = [
        make_case(
            q_lengths=(1, 1),
            k_lengths=lengths,
            q_heads=2,
            kv_heads=1,
            k_packed=True,
            v_scale_capacity=5,
        )
        for lengths in (first, second)
    ]
    assert cases[0]["kwargs"][SCALE_NAMES[-1]].shape == cases[1]["kwargs"][SCALE_NAMES[-1]].shape
    for case in cases:
        scale = case["kwargs"][SCALE_NAMES[-1]]
        used = sum((length + 63) // 64 for length in case["k_lengths"])
        assert (scale[used:] == 255).all()
        output, lse = reference(case)
        assert torch.isfinite(output).all() and torch.isfinite(lse).all()
        scale[used:] = 0  # Unused capacity must have no effect on the oracle.
        updated, updated_lse = reference(case)
        assert torch.equal(output, updated) and torch.equal(lse, updated_lse)


@pytest.mark.usefixtures("single_cpu_thread")
def test_sampled_reference_only_writes_selected_rows():
    case = make_case(q_lengths=(129,), k_lengths=(65,), q_heads=2, kv_heads=1)
    full, full_lse = reference(case)
    rows = [0, 31, 127, 128]
    sampled, sampled_lse = reference(case, query_rows=[rows])
    torch.testing.assert_close(sampled[:, rows], full[:, rows])
    torch.testing.assert_close(sampled_lse[:, :, rows], full_lse[:, :, rows])
    assert torch.isnan(sampled[:, 1]).all() and torch.isnan(sampled_lse[:, :, 1]).all()


# Input validation, public API and graph schema.


def inputs(length=17, *, packed=False, layout="BSND"):
    qshape = (length, 4, 128) if packed else (1, length, 4, 128)
    kshape = (length + 3, 2, 128) if packed else (1, length + 3, 2, 128)
    if layout == "BNSD" and not packed:
        qshape = (1, 4, length, 128)
        kshape = (1, 2, length + 3, 128)
    q, k = (torch.empty(shape, dtype=torch.float8_e4m3fn) for shape in (qshape, kshape))
    v = torch.empty_like(k)
    qs = torch.full(q.shape[:-1] + (2, 2), 127, dtype=torch.uint8)
    ks = torch.full(k.shape[:-1] + (2, 2), 127, dtype=torch.uint8)
    groups = (length + 3 + 63) // 64
    vshape = (groups, 2, 128, 2) if packed else (1, groups, 2, 128, 2)
    if layout == "BNSD" and not packed:
        vshape = (1, 2, groups, 128, 2)
    vs = torch.full(vshape, 127, dtype=torch.uint8)
    return q, k, v, qs, ks, vs


@pytest.mark.parametrize("packed,layout", [(False, "BSND"), (False, "BNSD"), (True, "BSND")])
def test_scale_metadata_layouts(packed, layout):
    q, k, v, *scales = inputs(packed=packed, layout=layout)
    result = interface._resolve_mxfp8_scales(q, k, v, scales, layout=layout)
    assert all(actual is expected for actual, expected in zip(result, scales))
    assert interface._resolve_mxfp8_scales(q, k, v, (None, None, None)) is None


@pytest.mark.parametrize("missing", [0, 1, 2])
def test_all_three_scales_are_required(missing):
    q, k, v, *scales = inputs()
    scales[missing] = None
    with pytest.raises(ValueError, match="together"):
        interface._resolve_mxfp8_scales(q, k, v, scales)


@pytest.mark.parametrize("invalid", ["dtype", "shape", "device", "q_dtype", "head_dim"])
def test_invalid_mxfp8_metadata(invalid):
    q, k, v, qs, ks, vs = inputs()
    if invalid == "dtype":
        vs = vs.float()
    elif invalid == "shape":
        qs = qs[..., 0]
    elif invalid == "device":
        qs = torch.empty(qs.shape, dtype=qs.dtype, device="meta")
    elif invalid == "q_dtype":
        q = torch.empty(q.shape, dtype=torch.float16)
    else:
        q, k, v = (t[..., :64] for t in (q, k, v))
    with pytest.raises(ValueError):
        interface._resolve_mxfp8_scales(q, k, v, (qs, ks, vs))


@pytest.mark.parametrize("native_e8m0", [False, True])
def test_scale_canonicalization_preserves_bytes_and_alignment(native_e8m0):
    q, k, v, qs, ks, vs = inputs()
    qs = torch.full((qs.numel() + 1,), 127, dtype=torch.uint8)[1:].reshape(qs.shape)
    ks = torch.full((*ks.shape[:-1], 4), 125, dtype=torch.uint8)[..., ::2]
    expected = [t.clone() for t in (qs, ks, vs)]
    supplied = tuple(t.view(torch.float8_e8m0fnu) if native_e8m0 else t for t in (qs, ks, vs))
    canonical = interface._resolve_mxfp8_scales(q, k, v, supplied, canonical=True)
    for actual, wanted in zip(canonical, expected):
        assert actual.dtype == torch.uint8 and actual.is_contiguous()
        assert actual.data_ptr() % 16 == 0
        assert torch.equal(actual, wanted)


@pytest.mark.parametrize("no_grad", [False, True])
@pytest.mark.parametrize("operand", ["q", "k", "v", "q_scale"])
def test_mxfp8_gradients_are_rejected(no_grad, operand):
    q, k, v, qs, ks, vs = inputs()
    if operand == "q_scale":
        qs = qs.view(torch.float8_e8m0fnu).requires_grad_()
    else:
        {"q": q, "k": k, "v": v}[operand].requires_grad_()
    with torch.no_grad() if no_grad else nullcontext():
        with pytest.raises(NotImplementedError, match="inference forward only"):
            interface._resolve_mxfp8_scales(q, k, v, (qs, ks, vs))


def test_packed_v_scale_capacity_and_mixed_layouts():
    q, _, _, qs, _, _ = inputs(packed=False)
    _, k, v, _, ks, vs = inputs(packed=True)
    padded = torch.full((vs.shape[0] + 3, *vs.shape[1:]), 127, dtype=torch.uint8)
    interface._resolve_mxfp8_scales(q, k, v, (qs, ks, padded))
    with pytest.raises(ValueError, match="capacity"):
        interface._resolve_mxfp8_scales(q, k, v, (qs, ks, vs[:0]))


def op_arguments(data, *, packed=False, layout="BSND"):
    q, k, v, qs, ks, vs = data
    cq = torch.tensor([0, q.shape[0]], dtype=torch.int32) if packed else None
    ck = torch.tensor([0, k.shape[0]], dtype=torch.int32) if packed else None
    return (
        q,
        k,
        v,
        cq,
        ck,
        [],
        [],
        [],
        0,
        None,
        False,
        None,
        None,
        0.0,
        [],
        None,
        None,
        layout,
        True,
        False,
        qs,
        ks,
        vs,
    )


@pytest.mark.parametrize("packed,layout", [(False, "BSND"), (False, "BNSD"), (True, "BSND")])
def test_mxfp8_fake_output_contract(packed, layout):
    data = inputs(packed=packed, layout=layout)
    args = op_arguments(data, packed=packed, layout=layout)
    with FakeTensorMode() as mode:
        args = tuple(mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for t in args)
        out, lse = graph._attention_fwd_mxfp8(*args)
    assert out.shape == data[0].shape and out.dtype == torch.bfloat16
    assert lse.dtype == torch.float32
    assert lse.shape == ((4, 17) if packed else (1, 4, 17))


@pytest.mark.parametrize("packed,layout", [(False, "BSND"), (False, "BNSD"), (True, "BNSD")])
@pytest.mark.parametrize("hint", [None, 42])
def test_graph_runtime_resolves_sequence_axis(monkeypatch, packed, layout, hint):
    data = inputs(packed=packed, layout=layout)
    args = list(op_arguments(data, packed=packed, layout=layout))
    args[7] = [0, -1, -1, -1, -1, -1, -1] * 2
    args[15] = args[16] = hint
    monkeypatch.setattr(graph, "_CALLBACKS", {0: (None,) * 6})
    monkeypatch.setattr(interface, "_prepare_config", lambda *args, **kwargs: kwargs)
    _, _, _, config = graph._runtime_attention(*args)
    assert config["max_q"] == (hint if packed and hint is not None else 17)
    assert config["max_k"] == (hint if packed and hint is not None else 20)
    assert config["q_descale"] is data[3]


@pytest.fixture
def cpu_mxfp8_runtime(monkeypatch):
    """Substitute numerical execution; preserve the actual graph/operator ABI."""

    def prepare(*args):
        q, k, v = args[:3]
        scales = interface._resolve_mxfp8_scales(
            q, k, v, args[-3:], layout=args[17], canonical=True
        )
        return q, k, v, {"layout": args[17], "mxfp8_scales": scales}

    def launch(q, k, v, out, lse, config):
        out.fill_(config["mxfp8_scales"][0].flatten()[0].float())
        lse.zero_()

    monkeypatch.setattr(graph, "_runtime_attention", prepare)
    monkeypatch.setattr(runtime, "launch_attention", launch)


@pytest.mark.parametrize(
    "public,packed", [(False, False), (False, True), (True, False), (True, True)]
)
def test_mxfp8_fullgraph_dynamic_scale_rebinding(cpu_mxfp8_runtime, public, packed):
    if public:
        entry = fa4.flash_attn_varlen_func if packed else fa4.flash_attn_func
    else:
        entry = (
            interface.flash_attn_varlen_func_with_quant
            if packed
            else interface.flash_attn_func_with_quant
        )

    def call(q, k, v, qs, ks, vs, cq, ck):
        options = dict(q_descale=qs, k_descale=ks, v_descale=vs, return_lse=True)
        if packed:
            options.update(
                cu_seqlens_q=cq, cu_seqlens_k=ck, max_seqlen_q=q.shape[0], max_seqlen_k=k.shape[0]
            )
        return entry(q, k, v, **options)

    compiled = torch.compile(call, backend="aot_eager", fullgraph=True, dynamic=True)
    for length, code in ((17, 127), (31, 129)):
        data = inputs(length, packed=packed)
        data[3].fill_(code)
        cq = torch.tensor([0, length], dtype=torch.int32)
        ck = torch.tensor([0, length + 3], dtype=torch.int32)
        result = compiled(*data, cq, ck)
        out, lse = result[:2]
        assert out.shape == data[0].shape and out.dtype == torch.bfloat16
        assert torch.all(out == code)
        assert lse.shape == ((4, length) if packed else (1, 4, length))
        if public and packed:
            assert len(result) == 4
            assert all(t.numel() == 0 and t.dtype == torch.float32 for t in result[2:])
        else:
            assert len(result) == 2


def test_mxfp8_custom_op_schema_and_fake(cpu_mxfp8_runtime):
    args = op_arguments(inputs())
    operator = torch.ops.flash_attn_npu_4_catlass.forward_mxfp8.default
    assert [arg.name for arg in operator._schema.arguments[-3:]] == [
        "q_descale",
        "k_descale",
        "v_descale",
    ]
    tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
    before = [tensor.view(torch.uint8).clone() for tensor in tensors]
    out, lse = operator(
        *args[:-3],
        q_descale=args[-3],
        k_descale=args[-2],
        v_descale=args[-1],
    )
    # SchemaCheckMode uses CPU allclose, which does not implement FP8. Check
    # the no-mutation/no-alias contract bytewise, then run fake-tensor opcheck.
    for tensor, original in zip(tensors, before):
        assert torch.equal(tensor.view(torch.uint8), original)
        assert tensor.untyped_storage().data_ptr() not in (
            out.untyped_storage().data_ptr(),
            lse.untyped_storage().data_ptr(),
        )
    result = torch.library.opcheck(
        graph._attention_fwd_mxfp8,
        args,
        test_utils=("test_faketensor",),
    )
    assert all(value == "SUCCESS" for value in result.values())


# Prepared plans, fresh bindings and cache ownership.


def make_launch(*, packed=False, code=127, boundaries=(0, 65, 128), length=128):
    qshape = (length, 4, 128) if packed else (1, length, 4, 128)
    kshape = (length, 2, 128) if packed else (1, length, 2, 128)
    q, k, v = (
        torch.full(shape, code - 80, dtype=torch.uint8).view(torch.float8_e4m3fn)
        for shape in (qshape, kshape, kshape)
    )
    qs, ks = (
        torch.full(shape[:-1] + (2, 2), code, dtype=torch.uint8) for shape in (qshape, kshape)
    )
    vs = torch.full(
        (3, 2, 128, 2) if packed else (1, (length + 63) // 64, 2, 128, 2), code, dtype=torch.uint8
    )
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty((4, length) if packed else (1, 4, length), dtype=torch.float32)
    config = dict(
        B=2 if packed else 1,
        H=4,
        KH=2,
        Sq=length,
        Sk=length,
        layout="BSND",
        mxfp8_scales=(qs, ks, vs),
        score_mod=None,
        mask_mod=None,
        block_sparse_tensors=None,
        scale=0.125,
        cu_q=torch.tensor([0, 64, 128], dtype=torch.int32) if packed else None,
        cu_k=torch.tensor(boundaries, dtype=torch.int32) if packed else None,
        aux_tensors=(torch.full((3,), float(code)),),
        aux_scalars=(float(code),),
    )
    return q, k, v, out, lse, config


@pytest.mark.parametrize(
    "length,capacity,masked,expected",
    [
        (128, None, False, ("general", 16)),
        (1024, None, False, ("continuous", 4)),
        (2048, None, False, ("continuous", 4)),
        (2049, None, False, ("continuous", 4)),
        (1024, 1, False, ("continuous", 1)),
        (1025, 2, False, ("continuous", 2)),
        (2048, 16, False, ("continuous", 4)),
        (2048, 16, True, ("continuous", 4)),
    ],
)
def test_unified_plan_preserves_pipeline_and_repair_selection(
    monkeypatch,
    length,
    capacity,
    masked,
    expected,
):
    from flash_attn_npu_4.catlass_dsl import builtin
    from flash_attn_npu_4.catlass_dsl.block_sparsity import BlockSparseTensorsTorch

    data = make_launch(length=length)
    config = data[-1]
    blocks = (length + 127) // 128
    if capacity is not None:
        counts = torch.zeros((1, 1, blocks), dtype=torch.int32)
        indices = torch.full((1, 1, blocks, capacity), -1, dtype=torch.int32)
        config["block_sparse_tensors"] = BlockSparseTensorsTorch(
            counts,
            indices,
            counts.clone(),
            indices.clone(),
            block_size=(128, 128),
        )
    if masked:
        config["mask_mod"] = builtin.causal()
    compiler = Mock(side_effect=lambda *args: SimpleNamespace())
    monkeypatch.setattr(runtime, "compile_cache", get_jit_cache())
    monkeypatch.setattr(runtime, "_compile_mxfp8", compiler)
    monkeypatch.setattr(runtime, "_cores", lambda device: 28)
    monkeypatch.setattr(
        torch.npu, "get_device_properties", lambda device: SimpleNamespace(L2_cache_size=128 << 20)
    )
    plan = runtime._make_mxfp8_plan(*data, offsets=None)

    pipeline, group_size = expected
    assert compiler.call_args_list[0].args[0][-8:-5] == (False, pipeline, group_size)
    assert compiler.call_count == 1
    if pipeline == "continuous":
        assert plan.guard_elements == (4 * blocks + plan.block_num) * 16
    else:
        assert plan.guard_elements == 0


def test_output_cache_policy_depends_on_capacity_and_working_set():
    # Equal storage with different sequence/batch/head arrangements must make
    # the same choice. Meta tensors avoid allocating benchmark-sized inputs.
    small = torch.empty((1, 8192, 4, 128), dtype=torch.bfloat16, device="meta")
    reshaped = torch.empty((4, 2048, 4, 128), dtype=torch.bfloat16, device="meta")
    assert not runtime._mxfp8_stream_output((small,), 32 << 20)
    assert not runtime._mxfp8_stream_output((reshaped,), 32 << 20)
    assert runtime._mxfp8_stream_output((small, small, small), 32 << 20)
    assert not runtime._mxfp8_stream_output((small, small, small), 64 << 20)
    assert not runtime._mxfp8_stream_output((small,), 0)


@pytest.fixture
def mocked_launch(monkeypatch):
    cache = get_jit_cache()

    compiler, launcher = Mock(side_effect=lambda *args: SimpleNamespace()), Mock()
    monkeypatch.setattr(runtime, "compile_cache", cache)
    monkeypatch.setattr(runtime, "_compile_mxfp8", compiler)
    monkeypatch.setattr(runtime, "_cores", lambda device: 28)
    monkeypatch.setattr(torch.npu, "get_device_properties", lambda device: SimpleNamespace(L2_cache_size=128 << 20))
    monkeypatch.setattr(runtime, "_launch", launcher)
    return cache, compiler, launcher


def test_warm_plan_rebinds_data_scales_outputs_and_aux(mocked_launch):
    cache, compiler, launcher = mocked_launch
    first, second = make_launch(code=127), make_launch(code=129)
    for data in (first, second):
        runtime.launch_mxfp8(*data)
        actual = launcher.call_args.args[1]
        q, k, v, out, lse, config = data
        expected = (q, k, v, *config["mxfp8_scales"])
        assert all(a is e for a, e in zip(actual[:6], expected))
        assert actual[7] is out and actual[8] is lse
        assert actual[15][0] is config["aux_tensors"][0]
        assert actual[16][:len(config["aux_scalars"])] == config["aux_scalars"]
    assert compiler.call_count == 1 and len(cache) == 1
    old_args, new_args = (call.args[1] for call in launcher.call_args_list)
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(old_args[:6], new_args[:6]))
    assert int(new_args[3].flatten()[0]) == 129
    # Changing values within the same allocations must also remain visible.
    second[0].view(torch.uint8).fill_(44)
    second[-1]["mxfp8_scales"][0].fill_(130)
    second[-1]["scale"] = -0.25
    runtime.launch_mxfp8(*second)
    args = launcher.call_args.args[1]
    assert int(args[0].view(torch.uint8).flatten()[0]) == 44
    assert int(args[3].flatten()[0]) == 130 and args[-4] == -0.25
    assert compiler.call_count == 1


def test_packed_boundaries_rebind_without_offset_preparation(mocked_launch, monkeypatch):
    cache, compiler, launcher = mocked_launch
    monkeypatch.setattr(
        torch, "cumsum", Mock(side_effect=AssertionError("offsets belong inside the kernel"))
    )
    data = [make_launch(packed=True, boundaries=cu) for cu in ((0, 65, 128), (0, 64, 128))]
    for tensors in data:
        runtime.launch_mxfp8(*tensors)
        args = launcher.call_args.args[1]
        assert args[6].dtype == torch.int32 and args[6].numel() >= 1
        assert args[14] is tensors[-1]["cu_k"]
        assert args[5].shape[0] == 3
    assert compiler.call_count == 1 and len(cache) == 1
    old, new = (call.args[1][14] for call in launcher.call_args_list)
    assert old.data_ptr() != new.data_ptr()
    # The device derives each prefix from these freshly bound boundaries.
    data[-1][-1]["cu_k"][1] = 65
    runtime.launch_mxfp8(*data[-1])
    assert launcher.call_args.args[1][14].tolist() == [0, 65, 128]
    assert compiler.call_count == 1


def test_eviction_and_clear_discard_mxfp8_plan(mocked_launch):
    cache, compiler, launcher = mocked_launch
    for step, compilations in enumerate((1, 1, 2, 3)):
        if step == 2:
            cache.pop(next(iter(cache)))
        elif step == 3:
            runtime.clear_compile_cache()
            assert not cache
        runtime.launch_mxfp8(*make_launch(code=127 + step))
        assert compiler.call_count == compilations
    executors = [call.args[0] for call in launcher.call_args_list]
    assert executors[0] is executors[1]
    assert executors[1] is not executors[2] and executors[2] is not executors[3]


@pytest.mark.parametrize("output", ["out", "lse"])
def test_invalid_output_dtype_is_rejected_before_compile(mocked_launch, output):
    _, compiler, launcher = mocked_launch
    data = list(make_launch())
    index = 3 if output == "out" else 4
    data[index] = data[index].to(torch.float16)
    with pytest.raises(ValueError, match="BF16 with FP32 LSE"):
        runtime.launch_mxfp8(*data)
    compiler.assert_not_called()
    launcher.assert_not_called()


@pytest.mark.parametrize("operand", ["q", "k", "v", "q_scale", "aux"])
def test_gradients_are_rejected_before_compile(mocked_launch, operand):
    _, compiler, launcher = mocked_launch
    data = make_launch()
    if operand == "q_scale":
        qs, ks, vs = data[-1]["mxfp8_scales"]
        data[-1]["mxfp8_scales"] = (qs.view(torch.float8_e8m0fnu).requires_grad_(), ks, vs)
    elif operand == "aux":
        data[-1]["aux_tensors"][0].requires_grad_()
    else:
        data[("q", "k", "v").index(operand)].requires_grad_()
    with pytest.raises(NotImplementedError, match="gradients|inference forward only"):
        runtime.launch_mxfp8(*data)
    compiler.assert_not_called()
    launcher.assert_not_called()


def test_guard_storage_is_fresh_for_each_fused_launch(mocked_launch, monkeypatch):
    from dataclasses import replace

    _, compiler, launcher = mocked_launch
    make_plan = runtime._make_mxfp8_plan
    monkeypatch.setattr(runtime, "_make_mxfp8_plan", lambda *args: replace(make_plan(*args), guard_elements=64))
    for _ in range(2):
        runtime.launch_mxfp8(*make_launch())
    calls = launcher.call_args_list
    launches = 1
    assert len(calls) == 2 * launches and compiler.call_count == 1
    first, second = calls[0].args[1][6], calls[launches].args[1][6]
    assert first.shape == second.shape == (64,)
    assert first.dtype == second.dtype == torch.int32
    assert first.data_ptr() != second.data_ptr()


@pytest.mark.npu
def test_bf16_and_mxfp8_share_cache_without_aliasing(device, monkeypatch):
    from flash_attn_npu_4.catlass_dsl import (
        flash_attn_varlen_func,
        flash_attn_varlen_func_with_quant,
    )

    from .mxfp8_reference import SCALE_NAMES, decoded_inputs, make_case, to_device
    from .test_utils import check

    cache = get_jit_cache()
    monkeypatch.setattr(runtime, "compile_cache", cache)
    ordinary = Mock(wraps=runtime._compile_kernel)
    quantized = Mock(wraps=runtime._compile_mxfp8)
    monkeypatch.setattr(runtime, "_compile_kernel", ordinary)
    monkeypatch.setattr(runtime, "_compile_mxfp8", quantized)

    for step, expected_compiles in enumerate((1, 1, 2)):
        if step == 2:
            runtime.clear_compile_cache()
            assert not cache
        case = make_case(q_lengths=(128,), k_lengths=(128,), q_heads=4, kv_heads=2, seed=80 + step)
        inputs, kwargs = to_device(case, device)
        baseline = tuple(t.to(device) for t in decoded_inputs(case))
        baseline_kwargs = {name: value for name, value in kwargs.items() if name not in SCALE_NAMES}
        expected, expected_lse = flash_attn_varlen_func(
            *baseline,
            **baseline_kwargs,
            return_lse=True,
        )
        out, lse = flash_attn_varlen_func_with_quant(*inputs, **kwargs, return_lse=True)
        check(out, expected)
        check(lse.to(torch.float16), expected_lse)
        assert ordinary.call_count == quantized.call_count == expected_compiles
        assert len(cache) == 2
        assert {key[0] for key in cache} == {"fwd", "mxfp8"}
        assert all(callable(executor) for executor in cache.values())


def test_kv_pack_policy_requires_reuse_beyond_cache_capacity():
    assert not runtime._mxfp8_should_pack_kv(32 << 20, 128, 128 << 20)
    assert not runtime._mxfp8_should_pack_kv(512 << 20, 2, 128 << 20)
    assert runtime._mxfp8_should_pack_kv(512 << 20, 128, 128 << 20)
    assert not runtime._mxfp8_should_pack_kv(512 << 20, 128, 0)


def test_mask_neighborhood_ignores_padded_sparse_capacity():
    from flash_attn_npu_4.catlass_dsl.kernels.flash_fwd_mxfp8_arch35 import (
        _mx_mask_span,
        _MxAffineIndex,
    )

    q, k = _MxAffineIndex(q=1), _MxAffineIndex(k=1)
    qlen, klen = _MxAffineIndex(qlen=1), _MxAffineIndex(klen=1)
    assert _mx_mask_span(((q >= k) & (q - k < 128)).tree) == 128
    center = q + klen - qlen
    assert _mx_mask_span(((k >= center - 128) & (k <= center + 128)).tree) == 257
    assert _mx_mask_span(((q // 128) == (k // 128)).tree) == 255
    assert _mx_mask_span((((q - k) // 128) == -1).tree) == 128
    assert _mx_mask_span((q >= k).tree) is None
    assert _mx_mask_span(((q % 128) >= (k % 128)).tree) is None


@pytest.mark.parametrize("fallback", [False, True])
def test_exact_auxiliary_plan_omits_guards_and_restores_them_on_fallback(monkeypatch, fallback):
    from flash_attn_npu_4.catlass_dsl.block_sparsity import BlockSparseTensorsTorch

    data = make_launch(packed=True, length=129)
    config = data[-1]
    config["mask_mod"] = global_labels
    config["aux_tensors"] = (torch.arange(129, dtype=torch.int32),) * 2
    config["aux_scalars"] = ()
    counts = torch.ones((2, 1, 2), dtype=torch.int32)
    indices = torch.zeros((2, 1, 2, 2), dtype=torch.int32)
    config["block_sparse_tensors"] = BlockSparseTensorsTorch(
        counts, indices, torch.zeros_like(counts), indices.clone(), block_size=(128, 128)
    )
    calls = []

    def compiler(arguments, *rest):
        calls.append(arguments)
        if fallback and len(calls) == 1:
            raise NotImplementedError("unsupported cached callback")
        return SimpleNamespace()

    monkeypatch.setattr(runtime, "compile_cache", get_jit_cache())
    monkeypatch.setattr(runtime, "_compile_mxfp8", compiler)
    monkeypatch.setattr(runtime, "_cores", lambda device: 28)
    monkeypatch.setattr(
        torch.npu, "get_device_properties", lambda device: SimpleNamespace(L2_cache_size=128 << 20)
    )
    plan = runtime._make_mxfp8_plan(*data, offsets=None)
    assert calls[0][-7] == "exact_aux" and calls[0][-3]
    if fallback:
        assert len(calls) == 2 and calls[1][-7] == "continuous" and not calls[1][-3]
        assert plan.guard_elements == (2 * 4 * 2 + plan.block_num) * 16
    else:
        assert len(calls) == 1 and plan.guard_elements == 0


# Device probability quantization and RNE bytes.


def _flat_bindings(tensors):
    return [
        tla.from_dlpack(t, origin_shape=(t.numel(),), layout_tag=tla.arch.RowMajor) for t in tensors
    ]


def _unpack_probabilities(packed):
    # FULL_STORE: [KV//4, Q//32, KV%4, Q%32], with every byte consumed.
    return packed.cpu().reshape(32, 2, 4, 32).permute(1, 3, 0, 2).reshape(64, 128)


def _unpack_scales(scales):
    return scales.cpu().reshape(4, 2, 16, 2).permute(0, 2, 1, 3).reshape(64, 4)


def probability_cases():
    values = torch.zeros((64, 128), dtype=torch.float32)
    # RNE ties between adjacent E4M3FN values. A maximum of 1 fixes scale=2^-8.
    numbers = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    finite = numbers[torch.isfinite(numbers)]
    ties = (finite[:-1] + finite[1:]) / 512.0
    for row in range(32):
        values[row, ::32] = 1.0
        values[row, 1:32] = ties[(row * 3 + torch.arange(31)) % len(ties)]
    # Power-selection threshold is 1.75 * 2^k, tested on both adjacent floats.
    for row, exponent in enumerate(range(-31, 0), start=32):
        boundary = torch.tensor(1.75 * 2.0**exponent, dtype=torch.float32)
        values[row, 0:32] = torch.nextafter(boundary, torch.tensor(0.0))
        values[row, 32:64] = boundary
        values[row, 64:96] = torch.nextafter(boundary, torch.tensor(float("inf")))
        # Last group remains all zero and must use scale byte 127.
    values[63, :32] = torch.finfo(torch.float32).tiny
    values[63, 32:64] = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    values[63, 64:96] = 0.0
    values[63, 96:] = 2.0**-120
    return values


def test_cpu_probability_quantizer_edges():
    values = probability_cases()
    encoded, scales = quantize_group32(values)
    decoded = encoded.float() * decode_scale(scales).repeat_interleave(32, -1)
    assert torch.isfinite(decoded).all()
    assert (scales[32:63, 3] == 127).all()
    assert (scales[63, :2] == 0).all()
    assert not encoded[63, 64:96].view(torch.uint8).count_nonzero()
    # Exactly halfway between 1 and 1.125 rounds down to the even encoding.
    tied = torch.zeros(32)
    tied[0], tied[1], tied[2] = 1.0, 1.0625 / 256, 1.1875 / 256
    got, scale = quantize_group32(tied)
    assert int(scale[0]) == 119
    assert float(got[1]) == 1.0 and float(got[2]) == 1.25


@tla.jit
def _prepare_probability_scales(scores, ph0, ph1, bad, inverses):
    """Exercise the production E8M0 encoder on supplied probabilities."""
    previous = tla.full(0, tla.Int32)
    invalid = bad.load()
    for group in tla.range_constexpr(4):
        maximum = tla.full(0.0, tla.Float32)
        for row in tla.range(32):
            item = tla.tile_view(scores, tla.make_shape(1, 64), tla.make_coord(group * 32 + row, 0))
            value = item.load()
            finite = tla.cmp(value, float("inf"), mode="lt")
            invalid = tla.where(finite, invalid, tla.full(1.0, tla.Float32))
            safe = tla.where(finite, value, tla.full(0.0, tla.Float32))
            maximum = tla.max(maximum, safe)
            item.store(safe)
        previous = _mx_group_scale(maximum, previous, group, inverses, ph0, ph1)
    bad.store(invalid)
    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


@tla.kernel
def _probability_probe(
    source: tla.Tensor, normalized: tla.Tensor, scales: tla.Tensor, packed: tla.Tensor
):
    p = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
    s = tla.allocate(256, tla.UInt8, tla.AddressSpace.ub, 256)
    b = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    r = tla.allocate(8192, tla.UInt8, tla.AddressSpace.ub, 256)
    matrix = tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1))
    pt = tla.make_tensor(p, matrix)
    bad = tla.make_tensor(b, tla.make_layout(tla.make_shape(64), tla.make_stride(1)))
    # The SIMD bridge needs captured tensor views; it cannot lower derived
    # pointer/recast-based tensor construction within the vector region.
    scale_layout = tla.make_layout(tla.make_shape(128), tla.make_stride(1))
    ph0 = tla.make_tensor(tla.recast_ptr(s, dtype=tla.Int16), scale_layout)
    ph1 = tla.make_tensor(tla.recast_ptr(s + 32, dtype=tla.Int16), scale_layout)
    iv = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    inverses = tla.make_tensor(iv, tla.make_layout(tla.make_shape(4, 64), tla.make_stride(64, 1)))
    ready = tla.flag("probe_ready", tla.arch.MTE2, tla.arch.VECTOR)
    stored = tla.flag("probe_stored", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(pt, tla.make_tensor(source.ptr, matrix))
        tla.set_flag(ready)
        tla.wait_flag(ready)
        with tla.vec.func(mode="simd"):
            bad.store(tla.full(0.0, tla.Float32))
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            _prepare_probability_scales(pt, ph0, ph1, bad, inverses)
        tla.pipe_barrier(tla.pipes.ALL)
        pack_scaled_rne(p, r, iv)
        tla.set_flag(stored)
        tla.wait_flag(stored)
        tla.copy(tla.make_tensor(normalized.ptr, matrix), pt)
        sl = tla.make_layout(tla.make_shape(256), tla.make_stride(1))
        pl = tla.make_layout(tla.make_shape(8192), tla.make_stride(1))
        tla.copy(tla.make_tensor(scales.ptr, sl), tla.make_tensor(s, sl))
        tla.copy(tla.make_tensor(packed.ptr, pl), tla.make_tensor(r, pl))


@pytest.mark.npu
def test_device_probability_quantization_matches_rne_bytes(device):
    values = probability_cases()
    expected, expected_scales = quantize_group32(values)
    source = values.T.contiguous().to(device)
    normalized = torch.empty_like(source)
    scales = torch.empty(256, dtype=torch.uint8, device=device)
    packed = torch.empty(8192, dtype=torch.uint8, device=device)
    tensors = (source, normalized, scales, packed)
    bindings = [
        tla.from_dlpack(t, origin_shape=(t.numel(),), layout_tag=tla.arch.RowMajor) for t in tensors
    ]
    executor = tla.compile(_probability_probe, *bindings, options="--npu-arch 3510")
    executor(*bindings, block_num=1)
    torch.npu.synchronize()
    actual_scales = _unpack_scales(scales)
    actual = _unpack_probabilities(packed)
    assert torch.equal(actual_scales, expected_scales)
    assert torch.equal(actual, expected.view(torch.uint8)), (
        "MXFP8 P bytes differ from the independent CPU RNE quantizer"
    )


@tla.kernel
def _scaled_pack_probe(source: tla.Tensor, inverse_scales: tla.Tensor, packed: tla.Tensor):
    p = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
    iv = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    r = tla.allocate(8192, tla.UInt8, tla.AddressSpace.ub, 256)
    matrix = tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1))
    inverse_layout = tla.make_layout(tla.make_shape(4, 64), tla.make_stride(64, 1))
    ready = tla.flag("scaled_ready", tla.arch.MTE2, tla.arch.VECTOR)
    stored = tla.flag("scaled_stored", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(tla.make_tensor(p, matrix), tla.make_tensor(source.ptr, matrix))
        tla.copy(
            tla.make_tensor(iv, inverse_layout), tla.make_tensor(inverse_scales.ptr, inverse_layout)
        )
        tla.set_flag(ready)
        tla.wait_flag(ready)
        pack_scaled_rne(p, r, iv)
        tla.set_flag(stored)
        tla.wait_flag(stored)
        packed_layout = tla.make_layout(tla.make_shape(8192), tla.make_stride(1))
        tla.copy(tla.make_tensor(packed.ptr, packed_layout), tla.make_tensor(r, packed_layout))


@pytest.mark.npu
def test_device_scaled_probability_pack_matches_rne_bytes(device):
    values = probability_cases()
    expected, scales = quantize_group32(values)
    inverse = torch.ldexp(torch.ones_like(scales, dtype=torch.float32), 127 - scales.int())
    tensors = (
        values.T.contiguous().to(device),
        inverse.T.contiguous().to(device),
        torch.empty(8192, dtype=torch.uint8, device=device),
    )
    bindings = [
        tla.from_dlpack(t, origin_shape=(t.numel(),), layout_tag=tla.arch.RowMajor) for t in tensors
    ]
    executor = tla.compile(_scaled_pack_probe, *bindings, options="--npu-arch 3510")
    executor(*bindings, block_num=1)
    torch.npu.synchronize()
    actual = _unpack_probabilities(tensors[2])
    assert torch.equal(actual, expected.view(torch.uint8))


@tla.jit
def _legacy_online_probability(scores, mstate, lstate, astate, blockm):
    """Frozen unfused online step, independent of the optimized helper."""
    zero = tla.full(0.0, tla.Float32)
    oldm, oldl = mstate.load(), lstate.load()
    newm = tla.max(oldm, blockm)
    safem = tla.where(tla.cmp(newm, float("-inf"), mode="eq"), zero, newm)
    alpha = tla.where(tla.cmp(oldl, 0.0, mode="gt"), tla.exp(oldm - safem), zero)
    blockl = zero
    for row in tla.range(128):
        item = tla.tile_view(scores, tla.make_shape(1, 64), tla.make_coord(row, 0))
        probability = tla.exp(item.load() - safem)
        blockl = blockl + probability
        item.store(probability)
    mstate.store(newm)
    lstate.store(oldl * alpha + blockl)
    astate.store(alpha)


@tla.kernel
def _online_probability_probe(
    source: tla.Tensor,
    input_state: tla.Tensor,
    packed: tla.Tensor,
    scales: tla.Tensor,
    output_state: tla.Tensor,
    fused: tla.Constexpr,
):
    p = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
    s = tla.allocate(256, tla.UInt8, tla.AddressSpace.ub, 256)
    r = tla.allocate(8192, tla.UInt8, tla.AddressSpace.ub, 256)
    u = tla.allocate(320, tla.Float32, tla.AddressSpace.ub, 256)
    iv = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    matrix = tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1))
    states = tla.make_layout(tla.make_shape(5, 64), tla.make_stride(64, 1))
    pt = tla.make_tensor(p, matrix)
    st = tla.make_tensor(u, states)
    m = tla.tile_view(st, tla.make_shape(1, 64), tla.make_coord(0, 0))
    denominator = tla.tile_view(st, tla.make_shape(1, 64), tla.make_coord(1, 0))
    blockm = tla.tile_view(st, tla.make_shape(1, 64), tla.make_coord(2, 0))
    bad = tla.tile_view(st, tla.make_shape(1, 64), tla.make_coord(3, 0))
    alpha = tla.tile_view(st, tla.make_shape(1, 64), tla.make_coord(4, 0))
    inverses = tla.make_tensor(iv, tla.make_layout(tla.make_shape(4, 64), tla.make_stride(64, 1)))
    scale_layout = tla.make_layout(tla.make_shape(128), tla.make_stride(1))
    ph0 = tla.make_tensor(tla.recast_ptr(s, dtype=tla.Int16), scale_layout)
    ph1 = tla.make_tensor(tla.recast_ptr(s + 32, dtype=tla.Int16), scale_layout)
    ready = tla.flag("online_ready", tla.arch.MTE2, tla.arch.VECTOR)
    stored = tla.flag("online_stored", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(pt, tla.make_tensor(source.ptr, matrix))
        tla.copy(st, tla.make_tensor(input_state.ptr, states))
        tla.set_flag(ready)
        tla.wait_flag(ready)
        with tla.vec.func(mode="simd"):
            if tla.const_expr(fused):
                online_quantize_probabilities(
                    pt, m, denominator, alpha, ph0, ph1, bad, inverses, blockm.load()
                )
            else:
                _legacy_online_probability(pt, m, denominator, alpha, blockm.load())
                tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                _prepare_probability_scales(pt, ph0, ph1, bad, inverses)
        tla.pipe_barrier(tla.pipes.ALL)
        pack_scaled_rne(p, r, iv)
        tla.set_flag(stored)
        tla.wait_flag(stored)
        scale_bytes = tla.make_layout(tla.make_shape(256), tla.make_stride(1))
        packed_bytes = tla.make_layout(tla.make_shape(8192), tla.make_stride(1))
        tla.copy(tla.make_tensor(scales.ptr, scale_bytes), tla.make_tensor(s, scale_bytes))
        tla.copy(tla.make_tensor(packed.ptr, packed_bytes), tla.make_tensor(r, packed_bytes))
        tla.copy(tla.make_tensor(output_state.ptr, states), st)


@pytest.mark.npu
@pytest.mark.parametrize("rescale", [False, True])
def test_device_fused_online_quantization_preserves_probabilities(device, rescale):
    generator = torch.Generator().manual_seed(220)
    scores = torch.randn((128, 64), generator=generator) * 4
    # Different group maxima, masked lanes, all-masked rows and tiny P values.
    scores[:32, 4:8] -= 90
    scores[32:64, 4:8] -= 30
    scores[64:96, 4:8] = -torch.inf
    scores[:, 0] = -torch.inf
    scores[:, 1] = -torch.inf
    scores[1, 1] = 0
    scores[0, 2] = torch.nan
    scores[0, 3] = torch.inf
    oldm = torch.linspace(-4, 14, 64) if rescale else torch.full((64,), -torch.inf)
    oldl = torch.linspace(1, 32, 64) if rescale else torch.zeros(64)
    bad = torch.zeros(64)
    bad[-1] = 1  # Invalid flags remain sticky across subsequent valid tiles.
    state = torch.stack((oldm, oldl, scores.max(0).values, bad, torch.zeros(64)))
    source, input_state = scores.to(device), state.to(device)
    results = []
    for fused in (False, True):
        packed = torch.empty(8192, dtype=torch.uint8, device=device)
        scales = torch.empty(256, dtype=torch.uint8, device=device)
        output_state = torch.empty_like(input_state)
        tensors = (source, input_state, packed, scales, output_state)
        bindings = [
            tla.from_dlpack(t, origin_shape=(t.numel(),), layout_tag=tla.arch.RowMajor)
            for t in tensors
        ]
        executor = tla.compile(
            _online_probability_probe, *bindings, fused, options="--npu-arch 3510"
        )
        executor(*bindings, block_num=1)
        torch.npu.synchronize()
        pbytes = _unpack_probabilities(packed)
        results.append((pbytes, scales.cpu(), output_state.cpu()))
    legacy_p, legacy_scale, legacy_state = results[0]
    fused_p, fused_scale, fused_state = results[1]
    assert torch.equal(fused_p, legacy_p)
    assert torch.equal(fused_scale, legacy_scale)
    torch.testing.assert_close(fused_state, legacy_state, rtol=2e-6, atol=2e-5, equal_nan=True)
    assert fused_state[3, 2] == fused_state[3, 3] == fused_state[3, -1] == 1
    assert fused_state[3, 0] == fused_state[3, 1] == 0
    # Independent FP32 denominator: the MX quantized numerator must not feed l.
    newm = torch.maximum(oldm, scores.max(0).values)
    safem = torch.where(torch.isneginf(newm), 0, newm)
    probability = torch.exp(scores - safem)
    expected_alpha = torch.where(oldl > 0, torch.exp(oldm - safem), 0)
    expected_l = oldl * expected_alpha + probability.sum(0)
    finite = torch.isfinite(expected_l)
    torch.testing.assert_close(fused_state[1, finite], expected_l[finite], rtol=2e-6, atol=2e-5)
    torch.testing.assert_close(fused_state[4, finite], expected_alpha[finite], rtol=2e-6, atol=2e-5)


@tla.jit
def _scaled_group_max_reference(scores, scale, groupmax):
    """Freeze multiply-before-max ordering, independent of optimized reduction."""
    blockm = tla.full(float("-inf"), tla.Float32)
    for group in tla.range_constexpr(4):
        maximum = tla.full(float("-inf"), tla.Float32)
        for row in tla.range(32):
            item = tla.tile_view(scores, tla.make_shape(1, 64), tla.make_coord(group * 32 + row, 0))
            maximum = tla.max(maximum, item.load() * scale)
        tla.tile_view(groupmax, tla.make_shape(1, 64), tla.make_coord(group, 0)).store(maximum)
        blockm = tla.max(blockm, maximum)
    return blockm


@tla.kernel
def _optimized_probability_probe(
    source: tla.Tensor,
    input_state: tla.Tensor,
    packed: tla.Tensor,
    scales: tla.Tensor,
    output_state: tla.Tensor,
    output_groupmax: tla.Tensor,
    scale: tla.Float32,
    mode: tla.Constexpr,
    positive: tla.Constexpr,
):
    p = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
    r = tla.allocate(8192, tla.UInt8, tla.AddressSpace.ub, 256)
    s = tla.allocate(256, tla.UInt8, tla.AddressSpace.ub, 256)
    u = tla.allocate(320, tla.Float32, tla.AddressSpace.ub, 256)
    iv = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    gp = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    sm = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    matrix = tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1))
    states = tla.make_layout(tla.make_shape(5, 64), tla.make_stride(64, 1))
    groups = tla.make_layout(tla.make_shape(4, 64), tla.make_stride(64, 1))
    pt, st = tla.make_tensor(p, matrix), tla.make_tensor(u, states)
    m = tla.tile_view(st, tla.make_shape(1, 64), tla.make_coord(0, 0))
    denominator = tla.tile_view(st, tla.make_shape(1, 64), tla.make_coord(1, 0))
    blockm = tla.tile_view(st, tla.make_shape(1, 64), tla.make_coord(2, 0))
    bad = tla.tile_view(st, tla.make_shape(1, 64), tla.make_coord(3, 0))
    alpha = tla.tile_view(st, tla.make_shape(1, 64), tla.make_coord(4, 0))
    inverses, groupmax = tla.make_tensor(iv, groups), tla.make_tensor(gp, groups)
    safem = tla.make_tensor(sm, tla.make_layout(tla.make_shape(64), tla.make_stride(1)))
    half_layout = tla.make_layout(tla.make_shape(128), tla.make_stride(1))
    ph0 = tla.make_tensor(tla.recast_ptr(s, dtype=tla.Int16), half_layout)
    ph1 = tla.make_tensor(tla.recast_ptr(s + 32, dtype=tla.Int16), half_layout)
    ready = tla.flag("optimized_ready", tla.arch.MTE2, tla.arch.VECTOR)
    stored = tla.flag("optimized_stored", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(pt, tla.make_tensor(source.ptr, matrix))
        tla.copy(st, tla.make_tensor(input_state.ptr, states))
        tla.set_flag(ready)
        tla.wait_flag(ready)
        with tla.vec.func(mode="simd"):
            if tla.const_expr(mode == 0):
                maximum = _scaled_group_max_reference(pt, scale, groupmax)
            else:
                maximum = scaled_group_max_readonly(pt, scale, groupmax, positive)
            blockm.store(maximum)
            if tla.const_expr(mode == 2):
                prepare_group_scales(
                    m, denominator, alpha, safem, ph0, ph1, inverses, maximum, groupmax
                )
            else:
                online_quantize_probabilities(
                    pt, m, denominator, alpha, ph0, ph1, bad, inverses, maximum, scale, True, False
                )
        if tla.const_expr(mode == 2):
            _mx_exact_exp_pack_update(p, r, iv, sm, u + 64, u + 256, u + 192, scale)
        else:
            pack_scaled_rne(p, r, iv)
        tla.set_flag(stored)
        tla.wait_flag(stored)
        packed_layout = tla.make_layout(tla.make_shape(8192), tla.make_stride(1))
        scale_layout = tla.make_layout(tla.make_shape(256), tla.make_stride(1))
        tla.copy(tla.make_tensor(packed.ptr, packed_layout), tla.make_tensor(r, packed_layout))
        tla.copy(tla.make_tensor(scales.ptr, scale_layout), tla.make_tensor(s, scale_layout))
        tla.copy(tla.make_tensor(output_state.ptr, states), st)
        tla.copy(tla.make_tensor(output_groupmax.ptr, groups), groupmax)


def _optimized_score_case(case):
    generator = torch.Generator().manual_seed(7421)
    scores = torch.randn((128, 64), generator=generator) * 12
    oldm, oldl = torch.zeros(64), torch.full((64,), 32.0)
    if case == "first":
        oldm.fill_(-torch.inf)
        oldl.zero_()
    elif case == "rescale":
        oldm = torch.linspace(-30, 30, 64)
        oldl = torch.linspace(1, 60, 64)
        scores[:32, 4:8] -= 90
        scores[32:64, 4:8] -= 30
        scores[64:96, 4:8] = -torch.inf
        scores[:, 8:12] *= 1e30
    elif case.startswith("threshold"):
        exponent = -(torch.arange(256).reshape(4, 64) % 127 + 1)
        maximum = torch.ldexp(torch.full((4, 64), 1.75), exponent).log()
        if case != "threshold":
            direction = -torch.inf if case.endswith("below") else torch.inf
            maximum = torch.nextafter(maximum, torch.full_like(maximum, direction))
        grouped = maximum[:, None, :].expand(4, 32, 64).clone()
        for row in range(1, 32):
            grouped[:, row] = torch.nextafter(
                grouped[:, row - 1], torch.full_like(maximum, -torch.inf)
            )
        scores = grouped.reshape(128, 64)
    elif case == "halfway":
        scores = probability_cases().T.contiguous().log()
    elif case == "offset":
        scores[:, :32] -= 1048576.0
        scores[:, 32:] += 1048576.0
        oldm = scores.amax(0) + 3
    elif case == "nonfinite":
        scores[:, 0] = -torch.inf
        scores[:, 1] = -torch.inf
        scores[1, 1] = 0
        scores[0, 2] = torch.nan
        scores[0, 3] = torch.inf
        scores[:, 4] = torch.nan
        scores[::2, 5] = -torch.inf
    bad = torch.zeros(64)
    bad[-1] = 1
    return scores, torch.stack((oldm, oldl, torch.zeros(64), bad, torch.zeros(64)))


@pytest.mark.npu
@pytest.mark.parametrize(
    "case,scale",
    [
        ("first", 0.0),
        ("first", -0.0),
        ("first", 0.125),
        ("first", -0.125),
        ("first", 1e-38),
        ("first", -1e-38),
        ("rescale", 1.7),
        ("rescale", -1.7),
        ("threshold_below", 1.0),
        ("threshold_below", -1.0),
        ("threshold", 1.0),
        ("threshold", -1.0),
        ("threshold_above", 1.0),
        ("threshold_above", -1.0),
        ("halfway", 1.0),
        ("halfway", -1.0),
        ("offset", 1.7),
        ("offset", -1.7),
        ("nonfinite", 0.0),
        ("nonfinite", -0.0),
        ("nonfinite", 0.125),
        ("nonfinite", -0.125),
    ],
)
def test_device_optimized_probability_paths_preserve_bytes(device, case, scale):
    scores, state = _optimized_score_case(case)
    source, input_state = scores.to(device), state.to(device)
    # The production dispatcher obtains this sign outside SIMD. Keep the probe
    # constexpr to isolate reduction and probability arithmetic from dispatch.
    positive = not bool(torch.signbit(torch.tensor(scale)))
    results = []
    for mode in (0, 1, 2):
        packed = torch.empty(8192, dtype=torch.uint8, device=device)
        scales = torch.empty(256, dtype=torch.uint8, device=device)
        output_state = torch.empty_like(input_state)
        groupmax = torch.empty((4, 64), device=device)
        tensors = (source, input_state, packed, scales, output_state, groupmax)
        bindings = [
            tla.from_dlpack(t, origin_shape=(t.numel(),), layout_tag=tla.arch.RowMajor)
            for t in tensors
        ]
        executor = tla.compile(
            _optimized_probability_probe,
            *bindings,
            tla.Float32(scale),
            mode,
            positive,
            options="--npu-arch 3510",
        )
        executor(*bindings, tla.Float32(scale), block_num=1)
        torch.npu.synchronize()
        results.append(
            (
                _unpack_probabilities(packed),
                _unpack_scales(scales),
                output_state.cpu(),
                groupmax.cpu(),
            )
        )
    reference_p, reference_scales, reference_state, reference_groupmax = results[0]
    finite = torch.isfinite(reference_state[1])
    for p, s, st, gm in results[1:]:
        assert torch.equal(p[finite], reference_p[finite])
        assert torch.equal(s[finite], reference_scales[finite])
        torch.testing.assert_close(st[:, finite], reference_state[:, finite], rtol=0, atol=0)
        torch.testing.assert_close(gm[:, finite], reference_groupmax[:, finite], rtol=0, atol=0)
        assert torch.equal(st[3], reference_state[3])
        assert torch.equal(torch.isnan(st[1]), torch.isnan(reference_state[1]))
    # A bad row's internal probabilities may vary, but its sticky flag and NaN
    # classification must survive. All valid denominators use unquantized P.
    scaled = scores * scale
    maximum = torch.where(torch.isnan(scaled), -torch.inf, scaled).amax(0)
    newm = torch.maximum(state[0], maximum)
    safem = torch.where(torch.isneginf(newm), 0, newm)
    probability = torch.exp(scaled - safem)
    alpha = torch.where(state[1] > 0, torch.exp(state[0] - safem), 0)
    expected_l = state[1] * alpha + probability.sum(0)
    finite = torch.isfinite(expected_l)
    torch.testing.assert_close(results[2][2][1, finite], expected_l[finite], rtol=2e-6, atol=2e-5)
    expected_bad = torch.where(torch.isfinite(probability).all(0), state[3], 1.0)
    assert torch.equal(results[2][2][3], expected_bad)


@tla.kernel
def _exact_label_probability_probe(
    source: tla.Tensor,
    input_state: tla.Tensor,
    labels: tla.Tensor,
    packed: tla.Tensor,
    scales: tla.Tensor,
    output_state: tla.Tensor,
    rows: tla.Int32,
    cols: tla.Int32,
):
    p = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
    r = tla.allocate(8192, tla.UInt8, tla.AddressSpace.ub, 256)
    s = tla.allocate(256, tla.UInt8, tla.AddressSpace.ub, 256)
    u = tla.allocate(320, tla.Float32, tla.AddressSpace.ub, 256)
    ids = tla.allocate(192, tla.Int32, tla.AddressSpace.ub, 256)
    matrix = tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1))
    states = tla.make_layout(tla.make_shape(5, 64), tla.make_stride(64, 1))
    label_layout = tla.make_layout(tla.make_shape(192), tla.make_stride(1))
    ready = tla.flag("label_probe_ready", tla.arch.MTE2, tla.arch.VECTOR)
    stored = tla.flag("label_probe_stored", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(tla.make_tensor(p, matrix), tla.make_tensor(source.ptr, matrix))
        tla.copy(tla.make_tensor(u, states), tla.make_tensor(input_state.ptr, states))
        tla.copy(tla.make_tensor(ids, label_layout), tla.make_tensor(labels.ptr, label_layout))
        tla.set_flag(ready)
        tla.wait_flag(ready)
        _mx_exact_label_quant(
            p,
            r,
            s,
            u,
            u + 64,
            u + 256,
            u + 192,
            ids,
            ids + 64,
            tla.Float32(1.0),
            rows,
            cols,
            tla.Int32(0),
            tla.Int32(0),
            tla.Int32(0),
        )
        tla.set_flag(stored)
        tla.wait_flag(stored)
        pl = tla.make_layout(tla.make_shape(8192), tla.make_stride(1))
        sl = tla.make_layout(tla.make_shape(256), tla.make_stride(1))
        tla.copy(tla.make_tensor(packed.ptr, pl), tla.make_tensor(r, pl))
        tla.copy(tla.make_tensor(scales.ptr, sl), tla.make_tensor(s, sl))
        tla.copy(tla.make_tensor(output_state.ptr, states), tla.make_tensor(u, states))


@pytest.mark.npu
@pytest.mark.parametrize(
    "case,tail",
    [
        ("first", False),
        ("rescale", True),
        ("threshold_below", True),
        ("threshold", True),
        ("threshold_above", True),
        ("halfway", False),
        ("offset", True),
        ("nonfinite", False),
        ("rescale", False),
        ("nonfinite", True),
    ],
)
def test_fused_label_quantization_matches_generic_bytes(device, case, tail):
    scores, state = _optimized_score_case(case)
    rows, cols = (31, 65) if tail else (64, 128)
    ids = (
        torch.cat((torch.arange(64) // 5, torch.arange(128) // 5)).int()
        if tail
        else torch.zeros(192, dtype=torch.int32)
    )
    visible = (
        (ids[:64, None] == ids[None, 64:])
        & (torch.arange(64)[:, None] < rows)
        & (torch.arange(128)[None, :] < cols)
    )
    masked = torch.where(visible.T, scores, -torch.inf).contiguous()
    result = []
    for fused in (False, True):
        packed = torch.empty(8192, dtype=torch.uint8, device=device)
        scales = torch.empty(256, dtype=torch.uint8, device=device)
        output = torch.empty_like(state, device=device)
        if fused:
            tensors = (scores.to(device), state.to(device), ids.to(device), packed, scales, output)
            args = _flat_bindings(tensors)
            fn = tla.compile(
                _exact_label_probability_probe,
                *args,
                tla.Int32(rows),
                tla.Int32(cols),
                options="--npu-arch 3510",
            )
            fn(*args, tla.Int32(rows), tla.Int32(cols), block_num=1)
        else:
            groupmax = torch.empty((4, 64), device=device)
            tensors = (masked.to(device), state.to(device), packed, scales, output, groupmax)
            args = _flat_bindings(tensors)
            fn = tla.compile(
                _optimized_probability_probe,
                *args,
                tla.Float32(1.0),
                0,
                True,
                options="--npu-arch 3510",
            )
            fn(*args, tla.Float32(1.0), block_num=1)
        torch.npu.synchronize()
        result.append((_unpack_probabilities(packed), _unpack_scales(scales), output.cpu()))
    (rp, rs, rt), (ap, as_, at) = result
    finite = torch.isfinite(rt[1])
    assert torch.equal(ap[finite], rp[finite])
    assert torch.equal(as_[finite], rs[finite])
    torch.testing.assert_close(
        at[[0, 1, 3, 4]][:, finite],
        rt[[0, 1, 3, 4]][:, finite],
        rtol=2e-6,
        atol=2e-5,
        equal_nan=True,
    )
    assert torch.equal(at[3], rt[3])
    assert torch.equal(torch.isnan(at[1]), torch.isnan(rt[1]))


_SOURCE = (
    _CPP_SHORT_QUANTIZE.split('extern "C"')[0]
    + r"""
__simd_vf__ inline void mx_pack_probe_vf(__ubuf__ float* source,__ubuf__ uint8_t* target) {
 MaskReg mf=CreateMask<float,MaskPattern::ALL>(),mb=CreateMask<uint8_t,MaskPattern::ALL>();
 for(uint16_t group=0;group<4;group++) {
  RegTensor<float> a,b,c,d;RegTensor<uint8_t> bytes;
  LoadAlign(a,source+group*256);LoadAlign(b,source+group*256+64);
  LoadAlign(c,source+group*256+128);LoadAlign(d,source+group*256+192);
  mx_short_pack(a,b,c,d,bytes,mf);StoreAlign(target+group*256,bytes,mb);
 }
}
extern "C" [aicore] __attribute__((always_inline)) void mx_pack_probe(uint64_t s,uint64_t p) {
 mx_pack_probe_vf((__ubuf__ float*)s,(__ubuf__ uint8_t*)p);
}
"""
)


@tla.extern(name="mx_pack_probe", source=_SOURCE)
def _pack(
    s: tla.Pointer[tla.Float32, tla.AddressSpace.ub], p: tla.Pointer[tla.UInt8, tla.AddressSpace.ub]
) -> None: ...


@tla.kernel
def _probe(source: tla.Tensor, target: tla.Tensor):
    s = tla.allocate(1024, tla.Float32, tla.AddressSpace.ub, 256)
    p = tla.allocate(1024, tla.UInt8, tla.AddressSpace.ub, 256)
    layout = tla.make_layout(tla.make_shape(1024), tla.make_stride(1))
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    stored = tla.flag("stored", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(tla.make_tensor(s, layout), tla.make_tensor(source.ptr, layout))
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        _pack(s, p)
        tla.set_flag(stored)
        tla.wait_flag(stored)
        tla.copy(tla.make_tensor(target.ptr, layout), tla.make_tensor(p, layout))


@pytest.mark.npu
def test_four_fp8_fields_match_independent_rne_bytes(device):
    numbers = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    ties = (numbers[:-1] + numbers[1:]) * 0.5
    values = torch.cat(
        (
            numbers,
            ties,
            torch.nextafter(ties, torch.zeros_like(ties)),
            torch.nextafter(ties, torch.full_like(ties, float("inf"))),
        )
    )
    values = values.repeat(3)[:1024].contiguous()
    expected = values.to(torch.float8_e4m3fn).view(torch.uint8)
    expected = expected.reshape(4, 4, 64).permute(0, 2, 1).contiguous().flatten()
    tensors = (values.to(device), torch.empty(1024, dtype=torch.uint8, device=device))
    bindings = [
        tla.from_dlpack(t, origin_shape=(t.numel(),), layout_tag=tla.arch.RowMajor) for t in tensors
    ]
    executor = tla.compile(_probe, *bindings, options="--npu-arch 3510")
    executor(*bindings, block_num=1)
    torch.npu.synchronize()
    assert torch.equal(tensors[1].cpu(), expected)


@tla.kernel
def _short_pack_probe(
    source: tla.Tensor, echo: tla.Tensor, packed: tla.Tensor, state: tla.Tensor, cols: tla.Int32
):
    # Allocate scores first: a valid UB pointer may have numeric address zero.
    score = tla.allocate(8192, tla.Float16, tla.AddressSpace.ub, 256)
    prob = tla.allocate(8192, tla.UInt8, tla.AddressSpace.ub, 256)
    accum = tla.allocate(128, tla.Float32, tla.AddressSpace.ub, 256)
    matrix = tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1))
    byte_layout = tla.make_layout(tla.make_shape(8192), tla.make_stride(1))
    state_layout = tla.make_layout(tla.make_shape(128), tla.make_stride(1))
    ready = tla.flag("short_ready", tla.arch.MTE2, tla.arch.VECTOR)
    stored = tla.flag("short_stored", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(tla.make_tensor(score, matrix), tla.make_tensor(source.ptr, matrix))
        tla.set_flag(ready)
        tla.wait_flag(ready)
        for tile in tla.range(4):
            _mx_short_quantize(
                score,
                prob,
                accum,
                accum + 64,
                tile,
                cols,
                tla.recast_ptr(accum, dtype=tla.Int32),
                accum,
                tla.Int32(0),
                tla.Int32(-1),
                tla.Int32(-1),
                tla.Int32(128),
                False,
            )
        tla.set_flag(stored)
        tla.wait_flag(stored)
        tla.copy(tla.make_tensor(echo.ptr, matrix), tla.make_tensor(score, matrix))
        tla.copy(tla.make_tensor(packed.ptr, byte_layout), tla.make_tensor(prob, byte_layout))
        tla.copy(tla.make_tensor(state.ptr, state_layout), tla.make_tensor(accum, state_layout))


@pytest.mark.npu
@pytest.mark.parametrize("cols", [1, 31, 32, 33, 63, 64, 65, 127, 128])
def test_short_pack_repeated_tiles_preserve_scores(device, cols):
    source = torch.full((128, 64), 0.125, dtype=torch.float16, device=device)
    echo = torch.empty_like(source)
    packed = torch.empty(8192, dtype=torch.uint8, device=device)
    state = torch.empty(128, device=device)
    args = [
        tla.from_dlpack(t, origin_shape=(t.numel(),), layout_tag=tla.arch.RowMajor)
        for t in (source, echo, packed, state)
    ]
    executor = tla.compile(_short_pack_probe, *args, tla.Int32(cols), options="--npu-arch 3510")
    executor(*args, tla.Int32(cols), block_num=1)
    torch.npu.synchronize()
    assert torch.equal(echo.cpu(), source.cpu()), "quantization wrote through an absent offset"
    probability = torch.exp(torch.tensor(0.125 + 1.9462890625))
    expected = torch.zeros((64, 128), dtype=torch.float32)
    expected[:, :cols] = probability
    assert torch.equal(
        _unpack_probabilities(packed), expected.to(torch.float8_e4m3fn).view(torch.uint8)
    )
    torch.testing.assert_close(
        state.cpu()[64:], torch.full((64,), probability * cols / 8 * 4), rtol=2e-6, atol=2e-5
    )


@tla.kernel
def _interval_probability_probe(
    source: tla.Tensor,
    input_state: tla.Tensor,
    labels: tla.Tensor,
    packed: tla.Tensor,
    scales: tla.Tensor,
    output_state: tla.Tensor,
    bounds: tla.Tensor,
    rows: tla.Int32,
    cols: tla.Int32,
    begin: tla.Int32,
    end: tla.Int32,
):
    p = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
    r = tla.allocate(8192, tla.UInt8, tla.AddressSpace.ub, 256)
    s = tla.allocate(256, tla.UInt8, tla.AddressSpace.ub, 256)
    u = tla.allocate(320, tla.Float32, tla.AddressSpace.ub, 256)
    ids = tla.allocate(192, tla.Int32, tla.AddressSpace.ub, 256)
    ranges = tla.allocate(192, tla.Int32, tla.AddressSpace.ub, 256)
    matrix = tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1))
    states = tla.make_layout(tla.make_shape(5, 64), tla.make_stride(64, 1))
    label_layout = tla.make_layout(tla.make_shape(192), tla.make_stride(1))
    ready = tla.flag("label_probe_ready", tla.arch.MTE2, tla.arch.VECTOR)
    stored = tla.flag("label_probe_stored", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(tla.make_tensor(p, matrix), tla.make_tensor(source.ptr, matrix))
        tla.copy(tla.make_tensor(u, states), tla.make_tensor(input_state.ptr, states))
        tla.copy(tla.make_tensor(ids, label_layout), tla.make_tensor(labels.ptr, label_layout))
        tla.set_flag(ready)
        tla.wait_flag(ready)
        _mx_label_ranges(ids, ids + 64, ranges, ranges + 128, rows, cols)
        _mx_common_label_interval(ranges, ranges + 128, rows)
        _mx_exact_label_interval_quant(
            p,
            r,
            s,
            u,
            u + 64,
            u + 256,
            u + 192,
            ids,
            ids + 64,
            tla.Float32(1.0),
            rows,
            cols,
            tla.Int32(0),
            tla.Int32(0),
            tla.Int32((rows == 64) & (cols == 128)),
            begin,
            end,
        )
        tla.set_flag(stored)
        tla.wait_flag(stored)
        pl = tla.make_layout(tla.make_shape(8192), tla.make_stride(1))
        sl = tla.make_layout(tla.make_shape(256), tla.make_stride(1))
        tla.copy(tla.make_tensor(packed.ptr, pl), tla.make_tensor(r, pl))
        tla.copy(tla.make_tensor(scales.ptr, sl), tla.make_tensor(s, sl))
        tla.copy(tla.make_tensor(output_state.ptr, states), tla.make_tensor(u, states))
        bl = tla.make_layout(tla.make_shape(1), tla.make_stride(1))
        tla.copy(tla.make_tensor(bounds.ptr, bl), tla.make_tensor(ranges + 128, bl))


@pytest.mark.npu
@pytest.mark.parametrize(
    "case,begin,end,rows,cols",
    [
        ("first", 0, 64, 64, 128),
        ("rescale", 17, 81, 64, 128),
        ("threshold_below", 63, 65, 31, 65),
        ("threshold", 63, 65, 31, 65),
        ("threshold_above", 63, 65, 31, 65),
        ("halfway", 0, 128, 64, 128),
        ("offset", 127, 128, 63, 128),
        ("nonfinite", 0, 128, 64, 128),
        ("nonfinite", 63, 65, 31, 65),
        ("rescale", 127, 128, 63, 128),
    ],
)
def test_label_interval_matches_generic_bytes(device, case, begin, end, rows, cols):
    scores, state = _optimized_score_case(case)
    keys = torch.where(
        torch.arange(128) < begin, 0, torch.where(torch.arange(128) < end, 7, 9)
    ).int()
    queries = torch.full((64,), 7, dtype=torch.int32)
    queries[rows:] = 999
    ids = torch.cat((queries, keys))
    visible = (
        (ids[:64, None] == ids[None, 64:])
        & (torch.arange(64)[:, None] < rows)
        & (torch.arange(128)[None, :] < cols)
    )
    masked = torch.where(visible.T, scores, -torch.inf).contiguous()
    result = []
    for fused in (False, True):
        packed = torch.empty(8192, dtype=torch.uint8, device=device)
        scales = torch.empty(256, dtype=torch.uint8, device=device)
        output = torch.empty_like(state, device=device)
        if fused:
            bound = torch.empty(1, dtype=torch.int32, device=device)
            tensors = (
                scores.to(device),
                state.to(device),
                ids.to(device),
                packed,
                scales,
                output,
                bound,
            )
            args = _flat_bindings(tensors)
            fn = tla.compile(
                _interval_probability_probe,
                *args,
                tla.Int32(rows),
                tla.Int32(cols),
                tla.Int32(begin),
                tla.Int32(end),
                options="--npu-arch 3510",
            )
            fn(
                *args,
                tla.Int32(rows),
                tla.Int32(cols),
                tla.Int32(begin),
                tla.Int32(end),
                block_num=1,
            )
        else:
            groupmax = torch.empty((4, 64), device=device)
            tensors = (masked.to(device), state.to(device), packed, scales, output, groupmax)
            args = _flat_bindings(tensors)
            fn = tla.compile(
                _optimized_probability_probe,
                *args,
                tla.Float32(1.0),
                0,
                True,
                options="--npu-arch 3510",
            )
            fn(*args, tla.Float32(1.0), block_num=1)
        torch.npu.synchronize()
        if fused:
            assert int(bound.cpu()[0]) == begin + (end << 8)
        result.append((_unpack_probabilities(packed), _unpack_scales(scales), output.cpu()))
    (rp, rs, rt), (ap, as_, at) = result
    finite = torch.isfinite(rt[1])
    assert torch.equal(ap[finite], rp[finite])
    assert torch.equal(as_[finite], rs[finite])
    torch.testing.assert_close(
        at[[0, 1, 3, 4]][:, finite],
        rt[[0, 1, 3, 4]][:, finite],
        rtol=2e-6,
        atol=2e-5,
        equal_nan=True,
    )
    assert torch.equal(at[3], rt[3])
    assert torch.equal(torch.isnan(at[1]), torch.isnan(rt[1]))


@tla.kernel
def _sparse_label_probability_probe(
    source: tla.Tensor,
    input_state: tla.Tensor,
    labels: tla.Tensor,
    packed: tla.Tensor,
    scales: tla.Tensor,
    output_state: tla.Tensor,
    rows: tla.Int32,
    cols: tla.Int32,
):
    p = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
    r = tla.allocate(8192, tla.UInt8, tla.AddressSpace.ub, 256)
    s = tla.allocate(256, tla.UInt8, tla.AddressSpace.ub, 256)
    u = tla.allocate(320, tla.Float32, tla.AddressSpace.ub, 256)
    ids = tla.allocate(192, tla.Int32, tla.AddressSpace.ub, 256)
    ranges = tla.allocate(192, tla.Int32, tla.AddressSpace.ub, 256)
    matrix = tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1))
    states = tla.make_layout(tla.make_shape(5, 64), tla.make_stride(64, 1))
    label_layout = tla.make_layout(tla.make_shape(192), tla.make_stride(1))
    ready = tla.flag("label_probe_ready", tla.arch.MTE2, tla.arch.VECTOR)
    stored = tla.flag("label_probe_stored", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(tla.make_tensor(p, matrix), tla.make_tensor(source.ptr, matrix))
        tla.copy(tla.make_tensor(u, states), tla.make_tensor(input_state.ptr, states))
        tla.copy(tla.make_tensor(ids, label_layout), tla.make_tensor(labels.ptr, label_layout))
        tla.set_flag(ready)
        tla.wait_flag(ready)
        _mx_label_ranges(ids, ids + 64, ranges, ranges + 128, rows, cols)
        _mx_scatter(
            p,
            r,
            s,
            u,
            u + 64,
            u + 256,
            u + 192,
            ranges,
            tla.Int32(0),
            rows,
            cols,
            tla.Int32(0),
            tla.Int32(16),
            tla.Int32(1),
            tla.Float32(1.0),
        )
        tla.set_flag(stored)
        tla.wait_flag(stored)
        pl = tla.make_layout(tla.make_shape(8192), tla.make_stride(1))
        sl = tla.make_layout(tla.make_shape(256), tla.make_stride(1))
        tla.copy(tla.make_tensor(packed.ptr, pl), tla.make_tensor(r, pl))
        tla.copy(tla.make_tensor(scales.ptr, sl), tla.make_tensor(s, sl))
        tla.copy(tla.make_tensor(output_state.ptr, states), tla.make_tensor(u, states))


@pytest.mark.npu
@pytest.mark.parametrize(
    "case,tail",
    [
        ("first", False),
        ("rescale", True),
        ("threshold_below", True),
        ("threshold", True),
        ("threshold_above", True),
        ("halfway", False),
        ("offset", True),
        ("nonfinite", False),
        ("rescale", False),
        ("nonfinite", True),
    ],
)
def test_sparse_label_quantization_matches_generic_bytes(device, case, tail):
    scores, state = _optimized_score_case(case)
    rows, cols = (31, 65) if tail else (64, 128)
    ids = (
        torch.cat(((torch.arange(64) + (0 if tail else 64)) // 5, torch.arange(128) // 5)).int()
        - 10
    )
    visible = (
        (ids[:64, None] == ids[None, 64:])
        & (torch.arange(64)[:, None] < rows)
        & (torch.arange(128)[None, :] < cols)
    )
    masked = torch.where(visible.T, scores, -torch.inf).contiguous()
    result = []
    for fused in (False, True):
        packed = torch.empty(8192, dtype=torch.uint8, device=device)
        scales = torch.empty(256, dtype=torch.uint8, device=device)
        output = torch.empty_like(state, device=device)
        if fused:
            tensors = (scores.to(device), state.to(device), ids.to(device), packed, scales, output)
            args = _flat_bindings(tensors)
            fn = tla.compile(
                _sparse_label_probability_probe,
                *args,
                tla.Int32(rows),
                tla.Int32(cols),
                options="--npu-arch 3510",
            )
            fn(*args, tla.Int32(rows), tla.Int32(cols), block_num=1)
        else:
            groupmax = torch.empty((4, 64), device=device)
            tensors = (masked.to(device), state.to(device), packed, scales, output, groupmax)
            args = _flat_bindings(tensors)
            fn = tla.compile(
                _optimized_probability_probe,
                *args,
                tla.Float32(1.0),
                0,
                True,
                options="--npu-arch 3510",
            )
            fn(*args, tla.Float32(1.0), block_num=1)
        torch.npu.synchronize()
        result.append((_unpack_probabilities(packed), _unpack_scales(scales), output.cpu()))
    (rp, rs, rt), (ap, as_, at) = result
    finite = torch.isfinite(rt[1])
    assert torch.equal(ap[finite], rp[finite])
    assert torch.equal(as_[finite], rs[finite])
    torch.testing.assert_close(
        at[[0, 1, 3, 4]][:, finite],
        rt[[0, 1, 3, 4]][:, finite],
        rtol=2e-6,
        atol=2e-5,
        equal_nan=True,
    )
    assert torch.equal(at[3], rt[3])
    assert torch.equal(torch.isnan(at[1]), torch.isnan(rt[1]))


@pytest.mark.npu
def test_sparse_label_dispatch_handles_rebound_sorted_unsorted_and_wide_labels(device, monkeypatch):
    from flash_attn_npu_4.catlass_dsl import (
        flash_attn_varlen_func_with_quant as flash_attn_varlen_func,
        runtime,
    )
    from tests.catlass_dsl.mxfp8_reference import (
        check_quantized_lse,
        make_case,
        reference,
        to_device,
    )
    from tests.catlass_dsl.test_utils import check

    case = make_case(
        q_lengths=(129, 65),
        k_lengths=(193, 31),
        q_heads=4,
        kv_heads=2,
        q_packed=True,
        k_packed=True,
        seed=1187,
    )
    inputs, kwargs = to_device(case, device)
    qa = torch.empty(224, dtype=torch.int32)
    ka = torch.empty_like(qa)
    aux = (qa.to(device), ka.to(device))

    def mask(b, h, q, k, info, aux, scalars):
        return aux[0][q + info[2]] == aux[1][k + info[3]]

    original = runtime._make_mxfp8_plan

    def capture(*args, **options):
        plan = original(*args, **options)
        assert plan.guard_elements == 0, "must use exact_aux without compiler fallback"
        return plan

    monkeypatch.setattr(runtime, "_make_mxfp8_plan", capture)
    runtime.clear_compile_cache()
    try:
        keys = None
        for mode in ("sorted", "shared_range", "nonuniform_wide", "unsorted", "wide", "empty"):
            qa.copy_(torch.arange(224, dtype=torch.int32) // 5 - 20)
            ka.copy_(qa)
            qa[129:194].copy_(torch.arange(65, dtype=torch.int32) % 7 + 18)
            if mode == "shared_range":
                qa.fill_(7)
                ka.fill_(99)
                ka[:32] = 0
                ka[32:96] = 7
            elif mode == "nonuniform_wide":
                qa.copy_(torch.arange(224, dtype=torch.int32) // 32)
                ka.copy_(qa)
            elif mode == "unsorted":
                ka[:193].copy_(
                    ka[:193].clone()[
                        torch.randperm(193, generator=torch.Generator().manual_seed(1188))
                    ]
                )
            elif mode == "wide":
                qa.zero_()
                ka.zero_()
            elif mode == "empty":
                qa.fill_(2**31 - 1)
                ka.fill_(-(2**31))
            aux[0].copy_(qa.to(device))
            aux[1].copy_(ka.to(device))
            sparse = _blocks(case, mask, (qa, ka))
            # Keep the callback active on fully visible blocks to exercise
            # the wide-range fallback as well as the full-block bypass tests.
            for b in range(2):
                for q in range(2):
                    n = int(sparse.full_block_cnt[b, 0, q])
                    start = int(sparse.mask_block_cnt[b, 0, q])
                    if n:
                        sparse.mask_block_idx[b, 0, q, start : start + n].copy_(
                            sparse.full_block_idx[b, 0, q, :n]
                        )
            sparse.mask_block_cnt.add_(sparse.full_block_cnt)
            sparse.full_block_cnt.zero_()
            expected, logs = reference(
                case, mask_mod=mask, sparse=sparse, aux=(qa, ka), scale=-0.125
            )
            actual, lse = flash_attn_varlen_func(
                *inputs,
                **kwargs,
                mask_mod=global_labels,
                aux_tensors=aux,
                softmax_scale=-0.125,
                block_sparse_tensors=_device_sparse(sparse, device),
                return_lse=True,
            )
            check(actual, expected)
            check_quantized_lse(lse.cpu(), logs)
            if keys is not None:
                assert set(runtime.compile_cache) == keys
            keys = set(runtime.compile_cache)
    finally:
        runtime.clear_compile_cache()


@tla.kernel
def _window_probability_probe(
    source: tla.Tensor,
    input_state: tla.Tensor,
    radii: tla.Tensor,
    packed: tla.Tensor,
    scales: tla.Tensor,
    output_state: tla.Tensor,
    rows: tla.Int32,
    cols: tla.Int32,
    relative: tla.Int32,
    maximum: tla.Int32,
):
    p = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
    r = tla.allocate(8192, tla.UInt8, tla.AddressSpace.ub, 256)
    s = tla.allocate(256, tla.UInt8, tla.AddressSpace.ub, 256)
    u = tla.allocate(320, tla.Float32, tla.AddressSpace.ub, 256)
    ids = tla.allocate(64, tla.Int32, tla.AddressSpace.ub, 256)
    matrix = tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1))
    states = tla.make_layout(tla.make_shape(5, 64), tla.make_stride(64, 1))
    label_layout = tla.make_layout(tla.make_shape(64), tla.make_stride(1))
    ready = tla.flag("label_probe_ready", tla.arch.MTE2, tla.arch.VECTOR)
    stored = tla.flag("label_probe_stored", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(tla.make_tensor(p, matrix), tla.make_tensor(source.ptr, matrix))
        tla.copy(tla.make_tensor(u, states), tla.make_tensor(input_state.ptr, states))
        tla.copy(tla.make_tensor(ids, label_layout), tla.make_tensor(radii.ptr, label_layout))
        tla.set_flag(ready)
        tla.wait_flag(ready)
        _mx_scatter(
            p,
            r,
            s,
            u,
            u + 64,
            u + 256,
            u + 192,
            ids,
            relative,
            rows,
            cols,
            tla.Int32(0),
            maximum,
            tla.Int32(0),
            tla.Float32(1.0),
        )
        tla.set_flag(stored)
        tla.wait_flag(stored)
        pl = tla.make_layout(tla.make_shape(8192), tla.make_stride(1))
        sl = tla.make_layout(tla.make_shape(256), tla.make_stride(1))
        tla.copy(tla.make_tensor(packed.ptr, pl), tla.make_tensor(r, pl))
        tla.copy(tla.make_tensor(scales.ptr, sl), tla.make_tensor(s, sl))
        tla.copy(tla.make_tensor(output_state.ptr, states), tla.make_tensor(u, states))


@pytest.mark.npu
@pytest.mark.parametrize(
    "case,relative,rows,cols",
    [
        ("first", -64, 63, 128),
        ("rescale", 64, 64, 128),
        ("threshold_below", 64, 64, 128),
        ("threshold", 64, 64, 128),
        ("threshold_above", 64, 64, 128),
        ("halfway", 0, 64, 128),
        ("offset", 64, 64, 128),
        ("nonfinite", 0, 64, 128),
        ("nonfinite", 128, 31, 13),
        ("rescale", 128, 31, 13),
    ],
)
def test_window_diagonal_matches_generic_bytes(device, case, relative, rows, cols):
    scores, state = _optimized_score_case(case)
    radius = torch.arange(64, dtype=torch.int32) % 20 - 3
    radius[3] = 7
    q = torch.arange(64)[:, None] + relative
    k = torch.arange(128)[None, :]
    visible = (
        (q >= k) & (q - k <= radius[:, None]) & (torch.arange(64)[:, None] < rows) & (k < cols)
    )
    masked = torch.where(visible.T, scores, -torch.inf).contiguous()
    result = []
    for fused in (False, True):
        packed = torch.empty(8192, dtype=torch.uint8, device=device)
        scales = torch.empty(256, dtype=torch.uint8, device=device)
        output = torch.empty_like(state, device=device)
        if fused:
            args = _flat_bindings(
                (scores.to(device), state.to(device), radius.to(device), packed, scales, output)
            )
            scalars = [tla.Int32(v) for v in (rows, cols, relative, 16)]
            fn = tla.compile(_window_probability_probe, *args, *scalars, options="--npu-arch 3510")
            fn(*args, *scalars, block_num=1)
        else:
            groupmax = torch.empty((4, 64), device=device)
            args = _flat_bindings(
                (masked.to(device), state.to(device), packed, scales, output, groupmax)
            )
            fn = tla.compile(
                _optimized_probability_probe,
                *args,
                tla.Float32(1.0),
                0,
                True,
                options="--npu-arch 3510",
            )
            fn(*args, tla.Float32(1.0), block_num=1)
        torch.npu.synchronize()
        result.append((_unpack_probabilities(packed), _unpack_scales(scales), output.cpu()))
    (rp, rs, rt), (ap, as_, at) = result
    finite = torch.isfinite(rt[1])
    assert torch.equal(ap[finite], rp[finite])
    assert torch.equal(as_[finite], rs[finite])
    torch.testing.assert_close(
        at[[0, 1, 3, 4]][:, finite],
        rt[[0, 1, 3, 4]][:, finite],
        rtol=2e-6,
        atol=2e-5,
        equal_nan=True,
    )
    assert torch.equal(at[3], rt[3])
    assert torch.equal(torch.isnan(at[1]), torch.isnan(rt[1]))


@simt
@tla.jit
def restricted_window(b, h, q, k, info, aux, scalars):
    return (q >= k) & (q - k <= aux[0][q + info[2]]) & (k >= 5)


@pytest.mark.npu
@pytest.mark.parametrize("restricted", [False, True])
def test_window_dispatch_rebinds_radius_and_preserves_extra_conditions(
    device, monkeypatch, restricted
):
    case = make_case(
        q_lengths=(129, 65),
        k_lengths=(193, 31),
        q_heads=4,
        kv_heads=2,
        q_packed=True,
        k_packed=True,
        seed=1185,
    )
    inputs, kwargs = to_device(case, device)
    cpu_radius = torch.empty(224, dtype=torch.int32)
    radius = cpu_radius.to(device)
    native = restricted_window if restricted else global_window

    def mask(b, h, q, k, info, aux, scalars):
        visible = (q >= k) & (q - k <= aux[0][q + info[2]])
        return visible & (k >= 5) if restricted else visible

    original = runtime._make_mxfp8_plan

    def capture(*args, **options):
        plan = original(*args, **options)
        assert plan.guard_elements == 0, "must compile the exact auxiliary path without fallback"
        return plan

    monkeypatch.setattr(runtime, "_make_mxfp8_plan", capture)
    runtime.clear_compile_cache()
    try:
        keys = None
        for mode in ("wide", "narrow", "empty"):
            if mode == "wide":
                values = torch.tensor([1, 16, 17, 127, 2**31 - 1, -(2**31)], dtype=torch.int32)
                cpu_radius.copy_(values[torch.arange(224) % 6])
            elif mode == "narrow":
                cpu_radius.copy_(torch.arange(224, dtype=torch.int32) % 17)
            else:
                cpu_radius.fill_(-(2**31))
            radius.copy_(cpu_radius.to(device))
            sparse = _blocks(case, mask, (cpu_radius,))
            expected, logs = reference(
                case, mask_mod=mask, sparse=sparse, aux=(cpu_radius,), scale=-0.125
            )
            actual, lse = flash_attn_varlen_func(
                *inputs,
                **kwargs,
                mask_mod=native,
                aux_tensors=(radius,),
                softmax_scale=-0.125,
                block_sparse_tensors=_device_sparse(sparse, device),
                return_lse=True,
            )
            check(actual, expected)
            check_quantized_lse(lse.cpu(), logs)
            if keys is not None:
                assert set(runtime.compile_cache) == keys
            keys = set(runtime.compile_cache)
    finally:
        runtime.clear_compile_cache()


@pytest.mark.npu
def test_window_diagonal_valid_and_masked_nan_rows(device):
    case = make_case(
        q_lengths=(129,),
        k_lengths=(129,),
        q_heads=4,
        kv_heads=2,
        q_packed=True,
        k_packed=True,
        seed=1186,
    )
    case["qkv"][0].view(torch.uint8)[[3, 9], 0, :] = 127
    radii = torch.full((129,), 7, dtype=torch.int32)
    radii[9] = -1

    def mask(b, h, q, k, info, aux, scalars):
        return (q >= k) & (q - k <= aux[0][q + info[2]])

    sparse = _blocks(case, mask, (radii,))
    inputs, kwargs = to_device(case, device)
    actual, lse = flash_attn_varlen_func(
        *inputs,
        **kwargs,
        mask_mod=global_window,
        aux_tensors=(radii.to(device),),
        block_sparse_tensors=_device_sparse(sparse, device),
        return_lse=True,
    )
    actual, lse = actual.cpu(), lse.cpu()
    assert torch.isnan(actual[3, 0]).all()
    assert torch.isnan(lse[0, 3])
    assert not actual[9].count_nonzero()
    assert torch.isneginf(lse[:, 9]).all()


# Fixed and packed attention, callbacks and nonfinite values.


@simt
@tla.jit
def scalar_score(s, b, h, q, k, info, aux, scalars):
    return s * scalars[0] + s * s * 0.1


@simd
@tla.jit
def vector_score(s, b, h, q, k, info, aux, scalars):
    return tla.add(tla.mul(s, scalars[0]), tla.mul(tla.mul(s, s), 0.1))


@simt
@tla.jit
def scalar_mask(b, h, q, k, info, aux, scalars):
    return (k <= q + info[1] - info[0]) & (q != 0)


@simd
@tla.jit
def vector_mask(b, h, q, k, info, aux, scalars):
    return tla.bitwise_and(
        tla.cmp(k, tla.add(q, info[1] - info[0]), mode="le"), tla.cmp(q, 0, mode="ne")
    )


@simt
@tla.jit
def aux_score(s, b, h, q, k, info, aux, scalars):
    return s * scalars[0] + aux[0][k + info[3]]


@simt
@tla.jit
def nan_score(s, b, h, q, k, info, aux):
    z = s
    if q == 0:
        z = tla.Float32(float("nan"))
    return z


@simt
@tla.jit
def positive_inf_score(s, b, h, q, k, info, aux):
    z = s
    if q == 0:
        z = tla.Float32(float("inf"))
    return z


@simt
@tla.jit
def negative_inf_score(s, b, h, q, k, info, aux):
    z = s
    if q == 0:
        z = tla.Float32(float("-inf"))
    return z


def score_reference(s, b, h, q, k, info, aux, scalars):
    return s * scalars[0] + s * s * 0.1


def mask_reference(b, h, q, k, info, aux, scalars):
    return (k <= q + info[1] - info[0]) & (q != 0)


def _device_sparse(sparse, device):
    return sparse._replace(
        **{
            name: value.to(device)
            for name, value in sparse._asdict().items()
            if isinstance(value, torch.Tensor)
        }
    )


def sparse_example(case, *, empty=False):
    qblocks = (max(case["q_lengths"], default=0) + 127) // 128
    counts = torch.zeros((1, 1, qblocks), dtype=torch.int32)
    partial = torch.full((1, 1, qblocks, 1), -137, dtype=torch.int32)
    full = partial.clone()
    if not empty:
        counts.fill_(1)
        partial.fill_(0)
        full.fill_(1)
    return BlockSparseTensorsTorch(counts, partial, counts.clone(), full, block_size=(128, 128))


def verify(case, device, *, options=None, reference_options=None, record_property=None):
    options, reference_options = options or {}, reference_options or {}
    inputs, kwargs = to_device(case, device)
    output, lse = flash_attn_varlen_func(*inputs, **kwargs, **options, return_lse=True)
    expected, expected_lse = reference(case, **reference_options)
    check(output, expected)
    check(lse, expected_lse)
    assert output.dtype == torch.bfloat16 and lse.dtype == torch.float32
    assert not output.requires_grad and not lse.requires_grad
    if record_property is not None:
        unquantized, exact_lse = reference(case, quantize_p=False, **reference_options)
        check(lse, exact_lse)
        record_property("mxfp8_output_quantization_error", error_metrics(output, unquantized))
    return output, lse


@pytest.mark.npu
@pytest.mark.parametrize(
    "sq,sk,kv_heads,options",
    [
        (128, 128, 4, {}),
        (17, 23, 1, {}),
        (31, 33, 2, {}),
        (32, 64, 2, {}),
        (63, 65, 2, {}),
        (127, 129, 2, {}),
        (31, 31, 2, {}),
        (32, 32, 2, {}),
        (64, 63, 2, {}),
        (65, 127, 2, {}),
        (129, 257, 2, {"causal": True}),
        (33, 257, 2, {"window_size": (17, 29)}),
        (33, 257, 2, {"softcap": 2.0}),
        (17, 129, 2, {"softmax_scale": 0.0}),
        (17, 129, 2, {"softmax_scale": -0.125}),
        # Static full tiles exercise the optimized path and its built-in fallbacks.
        (128, 256, 2, {"softmax_scale": 0.0}),
        (128, 256, 2, {"softmax_scale": -0.125}),
        (128, 256, 2, {"causal": True}),
        (128, 256, 2, {"softcap": 2.0}),
    ],
)
def test_mxfp8_forward(device, record_property, sq, sk, kv_heads, options):
    case = make_case(q_lengths=(sq,), k_lengths=(sk,), kv_heads=kv_heads)
    ref = dict(options)
    if "softmax_scale" in ref:
        ref["scale"] = ref.pop("softmax_scale")
    verify(case, device, options=options, reference_options=ref, record_property=record_property)


@pytest.mark.npu
@pytest.mark.parametrize(
    "q_packed,k_packed,native_scales",
    [(False, False, False), (False, True, True), (True, False, False), (True, True, True)],
)
def test_mxfp8_packed_and_mixed_layouts(device, q_packed, k_packed, native_scales):
    case = make_case(
        q_lengths=(0, 33) if q_packed else (33, 33),
        k_lengths=(63, 65) if k_packed else (65, 65),
        q_packed=q_packed,
        k_packed=k_packed,
        v_scale_capacity=7 if k_packed else None,
        native_scales=native_scales,
    )
    verify(case, device, options={"causal": True}, reference_options={"causal": True})


@pytest.mark.npu
@pytest.mark.parametrize("q_packed", [False, True])
def test_mxfp8_packed_empty_kv_segment(device, q_packed):
    case = make_case(
        q_lengths=(33, 33), k_lengths=(0, 65), q_packed=q_packed, k_packed=True, v_scale_capacity=4
    )
    output, lse = verify(case, device)
    empty_out = output[:33] if q_packed else output[0]
    empty_lse = lse[:, :33] if q_packed else lse[0]
    assert not empty_out.count_nonzero() and torch.isneginf(empty_lse).all()


@pytest.mark.npu
@pytest.mark.parametrize(
    "mode,packed,sq,sk",
    [
        ("simt", False, 129, 385),
        ("simd", True, 129, 385),
        ("simt", True, 129, 385),
        ("simd", False, 128, 384),
    ],
)
def test_mxfp8_partial_full_sparse_callbacks(device, mode, packed, sq, sk):
    case = make_case(q_lengths=(sq,), k_lengths=(sk,), q_packed=packed, k_packed=packed)
    sparse = sparse_example(case)
    score, mask = (scalar_score, scalar_mask) if mode == "simt" else (vector_score, vector_mask)
    verify(
        case,
        device,
        options=dict(
            score_mod=score,
            mask_mod=mask,
            aux_scalars=(0.7,),
            block_sparse_tensors=_device_sparse(sparse, device),
        ),
        reference_options=dict(
            score_mod=score_reference, mask_mod=mask_reference, scalars=(0.7,), sparse=sparse
        ),
    )


@pytest.mark.npu
@pytest.mark.parametrize("kind", ["empty_q", "empty_kv", "empty_batch", "omitted", "masked"])
def test_mxfp8_empty_cases(device, kind):
    qlens = () if kind == "empty_batch" else (0,) if kind == "empty_q" else (1,)
    klens = () if kind == "empty_batch" else (0,) if kind == "empty_kv" else (257,)
    case = make_case(q_lengths=qlens, k_lengths=klens)
    options = {}
    ref = {}
    if kind == "omitted":
        sparse = sparse_example(case, empty=True)
        options["block_sparse_tensors"] = _device_sparse(sparse, device)
        ref["sparse"] = sparse
    if kind == "masked":
        options["mask_mod"] = scalar_mask
        ref["mask_mod"] = mask_reference
    output, lse = verify(case, device, options=options, reference_options=ref)
    assert not output.count_nonzero() and torch.isneginf(lse).all()


@pytest.mark.npu
def test_mxfp8_internal_bnsd(device):
    case = make_case(q_lengths=(33,), k_lengths=(65,), layout="BNSD")
    qkv, kwargs = to_device(case, device)
    output, lse = interface._flash(
        *qkv,
        layout="BNSD",
        softmax_scale=None,
        causal=False,
        window_size=(None, None),
        softcap=0.0,
        score_mod=None,
        score_mod_bwd=None,
        mask_mod=None,
        aux_tensors=None,
        aux_scalars=None,
        block_sparse_tensors=None,
        block_sparse_tensors_bwd=None,
        return_lse=True,
        **{name: kwargs[name] for name in SCALE_NAMES},
    )
    expected, el = reference(case)
    check(output, expected)
    check(lse, el)


@pytest.mark.npu
def test_mxfp8_cached_plan_rebinds_scale_aux_and_stream(device):
    producer, consumer = torch.npu.Stream(device=device), torch.npu.Stream(device=device)
    keys = None
    previous = None
    for seed, factor in ((703, 0.3), (704, 1.2)):
        case = make_case(q_lengths=(17,), k_lengths=(129,), seed=seed)
        data = torch.linspace(-factor, factor, 129)
        with torch.npu.stream(producer):
            inputs, kwargs = to_device(case, device)
            aux = data.to(device)
        with torch.npu.stream(consumer):
            consumer.wait_stream(producer)
            out, lse = flash_attn_varlen_func(
                *inputs,
                **kwargs,
                score_mod=aux_score,
                aux_tensors=[aux],
                aux_scalars=(factor,),
                return_lse=True,
            )
        torch.npu.current_stream(device).wait_stream(consumer)
        wanted, el = reference(
            case,
            score_mod=lambda s, b, h, q, k, i, a, c: s * c[0] + a[0][k + i[3]],
            aux=(data,),
            scalars=(factor,),
        )
        check(out, wanted)
        check(lse, el)
        if keys is not None:
            assert set(runtime.compile_cache) == keys
            assert all(
                old.data_ptr() != kwargs[name].data_ptr()
                for old, name in zip(previous, SCALE_NAMES)
            )
        keys = set(runtime.compile_cache)
        previous = tuple(kwargs[name] for name in SCALE_NAMES)


@pytest.mark.npu
@pytest.mark.parametrize("compiled", [False, True])
def test_mxfp8_rebinds_packed_boundaries_with_capacity(device, compiled):
    def call(q, k, v, qs, ks, vs, cq, ck):
        return flash_attn_varlen_func(
            q,
            k,
            v,
            q_descale=qs,
            k_descale=ks,
            v_descale=vs,
            cu_seqlens_q=cq,
            cu_seqlens_k=ck,
            max_seqlen_q=64,
            max_seqlen_k=128,
            return_lse=True,
        )

    invoke = (
        torch.compile(call, backend="inductor", fullgraph=True, dynamic=True) if compiled else call
    )
    keys = None
    specifications = [((32, 32), (65, 63)), ((32, 32), (64, 64))]
    if compiled:
        specifications.append(((31, 49), (65, 128)))
    for step, (qlens, klens) in enumerate(specifications):
        case = make_case(
            q_lengths=qlens,
            k_lengths=klens,
            q_packed=True,
            k_packed=True,
            v_scale_capacity=8,
            seed=900 + step,
        )
        inputs, kwargs = to_device(case, device)
        out, lse = invoke(
            *inputs,
            *(kwargs[name] for name in SCALE_NAMES),
            kwargs["cu_seqlens_q"],
            kwargs["cu_seqlens_k"],
        )
        wanted, el = reference(case)
        check(out, wanted)
        check(lse, el)
        if step == 1:
            assert set(runtime.compile_cache) == keys
        keys = set(runtime.compile_cache)


@pytest.mark.npu
@pytest.mark.parametrize(
    "score,empty", [(nan_score, False), (positive_inf_score, False), (negative_inf_score, True)]
)
@pytest.mark.parametrize("masked", [False, True])
def test_mxfp8_nonfinite_callback_classification(device, score, empty, masked):
    case = make_case(q_lengths=(17,), k_lengths=(33,))
    inputs, kwargs = to_device(case, device)
    out, lse = flash_attn_varlen_func(
        *inputs,
        **kwargs,
        score_mod=score,
        mask_mod=scalar_mask if masked else None,
        return_lse=True,
    )
    if empty or masked:
        assert not out[:, 0].count_nonzero()
        assert torch.isneginf(lse[:, :, 0]).all()
    else:
        assert torch.isnan(out[:, 0]).all()
        assert torch.isnan(lse[:, :, 0]).all()
    assert torch.isfinite(out[:, 1:]).all()
    assert torch.isfinite(lse[:, :, 1:]).all()


@pytest.mark.npu
@pytest.mark.parametrize("operand", ["q", "k"])
@pytest.mark.parametrize("scale", [0.0, 0.125, -0.125])
def test_mxfp8_aligned_input_nan_isolated_to_affected_rows(device, operand, scale):
    case = make_case(q_lengths=(128,), k_lengths=(256,))
    expected, expected_lse = reference(case, scale=scale)
    affected = torch.zeros((1, 128, 4), dtype=torch.bool)
    # E4M3FN 0x7f is NaN. The identity-score path must retain its row flag,
    # including when multiplying a non-finite QK score by a zero scale.
    if operand == "q":
        case["qkv"][0].view(torch.uint8)[0, 5, 0, 0] = 0x7F
        affected[0, 5, 0] = True
    else:
        case["qkv"][1].view(torch.uint8)[0, 32, 1, 17] = 0x7F
        affected[:, :, 2:] = True  # Both query heads sharing KV head 1.
    inputs, kwargs = to_device(case, device)
    out, lse = flash_attn_varlen_func(*inputs, **kwargs, softmax_scale=scale, return_lse=True)
    out, lse = out.cpu(), lse.cpu()
    lse_affected = affected.transpose(1, 2)
    assert torch.isnan(out[affected]).all()
    assert torch.isnan(lse[lse_affected]).all()
    check(out[~affected], expected[~affected])
    check(lse[~lse_affected], expected_lse[~lse_affected])


@pytest.mark.npu
@pytest.mark.parametrize(
    "kind",
    [
        "masked_single",
        "masked_pair",
        "visible_full",
        "packed_empty_kv",
        "sparse_empty",
    ],
)
def test_mxfp8_value_nan_preserves_empty_rows_and_lse(device, kind):
    options, reference_options = {}, {}
    if kind == "packed_empty_kv":
        # A NaN in the next sequence must not contaminate an empty KV segment.
        case = make_case(
            q_lengths=(17, 17), k_lengths=(0, 33), k_packed=True, v_scale_capacity=3, seed=221
        )
        case["qkv"][2].view(torch.uint8)[0, 0, 0] = 0x7F
    else:
        sq = 129 if kind == "sparse_empty" else (128 if kind == "visible_full" else 17)
        sk = 129 if kind == "masked_pair" else (256 if kind == "visible_full" else 33)
        case = make_case(q_lengths=(sq,), k_lengths=(sk,), seed=221)
        case["qkv"][2].view(torch.uint8)[0, 0, 0, 0] = 0x7F
    if kind.startswith("masked_"):
        options["mask_mod"] = scalar_mask
        reference_options["mask_mod"] = mask_reference
    elif kind == "sparse_empty":
        sparse = BlockSparseTensorsTorch(
            torch.zeros((1, 1, 2), dtype=torch.int32),
            torch.zeros((1, 1, 2, 1), dtype=torch.int32),
            torch.tensor([[[0, 1]]], dtype=torch.int32),
            torch.zeros((1, 1, 2, 1), dtype=torch.int32),
            block_size=(128, 128),
        )
        options["block_sparse_tensors"] = _device_sparse(sparse, device)
        reference_options["sparse"] = sparse
    inputs, kwargs = to_device(case, device)
    out, lse = flash_attn_varlen_func(*inputs, **kwargs, **options, return_lse=True)
    out, lse = out.cpu(), lse.cpu()
    expected, expected_lse = reference(case, **reference_options)

    # V does not participate in QK or softmax statistics. Its NaN propagates
    # into visible output channels while visible-row LSE remains finite.
    assert torch.equal(torch.isnan(out), torch.isnan(expected))
    finite = torch.isfinite(expected)
    check(out[finite], expected[finite])
    empty = torch.isneginf(expected_lse)
    assert torch.equal(torch.isneginf(lse), empty)
    assert torch.isfinite(lse[~empty]).all()
    check(lse[~empty], expected_lse[~empty])
    assert torch.isnan(out).any()
    if kind != "visible_full":
        empty_output = empty.transpose(1, 2).unsqueeze(-1).expand_as(out)
        assert empty_output.any()
        assert torch.count_nonzero(out[empty_output]) == 0


@pytest.mark.npu
@pytest.mark.parametrize("packed,native_scales", [(False, False), (True, True)])
def test_mxfp8_noncontiguous_data_and_scales(device, packed, native_scales):
    case = make_case(
        q_lengths=(17,),
        k_lengths=(65,),
        q_packed=packed,
        k_packed=packed,
        native_scales=native_scales,
    )
    inputs, kwargs = to_device(case, device)

    def strided(value):
        storage = torch.empty(
            (*value.shape[:-1], 2 * value.shape[-1]), dtype=torch.uint8, device=device
        )
        result = storage[..., ::2]
        result.copy_(value.view(torch.uint8))
        assert not result.is_contiguous()
        return result.view(value.dtype)

    inputs = tuple(strided(value) for value in inputs)
    kwargs.update({name: strided(kwargs[name]) for name in SCALE_NAMES})
    out, lse = flash_attn_varlen_func(*inputs, **kwargs, return_lse=True)
    expected, el = reference(case)
    check(out, expected)
    check(lse, el)


@pytest.mark.npu
def test_mxfp8_fixed_return_shape(device):
    case = make_case(q_lengths=(17,), k_lengths=(23,))
    inputs, kwargs = to_device(case, device)
    scales = {name: kwargs[name] for name in SCALE_NAMES}
    out, lse = flash_attn_func(*inputs, **scales, return_lse=False)
    assert out.shape == inputs[0].shape and out.dtype == torch.bfloat16 and lse is None


# Sparse traversal, online state and repair.


@pytest.mark.npu
@pytest.mark.parametrize(
    "q_packed,k_packed,scale",
    [(False, False, None), (False, True, 0.0), (True, False, -0.125), (True, True, None)],
)
def test_complete_tiles_in_ragged_sequences(device, q_packed, k_packed, scale):
    case = make_case(
        q_lengths=(129, 256) if q_packed else (129, 129),
        k_lengths=(385, 129) if k_packed else (257, 257),
        q_packed=q_packed,
        k_packed=k_packed,
    )
    verify(
        case,
        device,
        options={"softmax_scale": scale},
        reference_options={"scale": scale},
    )


@simt
@tla.jit
def _partial_only_mask(b, h, q, k, info, aux, scalars):
    # All explicitly full blocks have index >= 1 and must bypass this mask.
    return (k < 128) & ((q + k) % 3 != 0) & (q % 17 != 0)


def _partial_mask_reference(b, h, q, k, info, aux, scalars):
    return (k < 128) & ((q + k) % 3 != 0) & (q % 17 != 0)


@pytest.fixture(scope="module")
def pair_case():
    return make_case(
        q_lengths=(641,),
        k_lengths=(641,),
        q_heads=4,
        kv_heads=2,
        seed=20261105,
    )


def _pair_sparse(kind):
    total = torch.arange(6, dtype=torch.int32).reshape(1, 1, 6).expand(1, 4, 6).clone()
    partial = (total > 0).to(torch.int32) if kind == "mixed" else torch.zeros_like(total)
    full = total - partial
    partial_indices = torch.full((1, 4, 6, 1), -193, dtype=torch.int32)
    full_indices = torch.full((1, 4, 6, 5), -193, dtype=torch.int32)
    for head in range(4):
        # Different head orders exercise online updates without sorting the
        # authoritative lists. KV block 5 contains only the final tail element.
        if head % 2:
            order = [5, 3, 1, 4, 2]
        elif kind == "mixed":
            order = [2, 3, 4, 5, 1]
        else:
            order = [1, 2, 3, 4, 5]
        for query_block in range(6):
            if partial[0, head, query_block]:
                partial_indices[0, head, query_block, 0] = 0
            count = int(full[0, head, query_block])
            full_indices[0, head, query_block, :count] = torch.tensor(order[:count])
    return BlockSparseTensorsTorch(
        partial, partial_indices, full, full_indices, block_size=(128, 128)
    )


@pytest.mark.npu
@pytest.mark.parametrize("kind,scale", [("full", None), ("mixed", 0.0), ("mixed", -0.125)])
def test_pair_pv_counts_0_to_5_with_tail(device, pair_case, kind, scale):
    sparse = _pair_sparse(kind)
    output, lse = verify(
        pair_case,
        device,
        options={
            "mask_mod": _partial_only_mask,
            "block_sparse_tensors": _device_sparse(sparse, device),
            "softmax_scale": scale,
        },
        reference_options={"scale": scale, "mask_mod": _partial_mask_reference, "sparse": sparse},
    )
    output, lse = output.cpu(), lse.cpu()
    assert not output[:, :128].count_nonzero()
    assert torch.isneginf(lse[..., :128]).all()
    assert torch.isfinite(output[:, 640:641]).all()
    assert torch.isfinite(lse[..., 640:641]).all()
    if kind == "mixed":
        # Query block 1 selects only partial block 0; these rows are masked.
        rows = torch.arange(128, 256)
        empty_rows = rows[rows % 17 == 0]
        assert not output[:, empty_rows].count_nonzero()
        assert torch.isneginf(lse[..., empty_rows]).all()


@tla.kernel
def _refresh_mxfp8_guard_cache(flags: tla.Tensor, result: tla.Tensor):
    sub = tla.arch.sub_block_idx()
    # Cache separate task/core flag lines before DMA overwrites them.
    before = flags[0] + flags[128]
    preloaded = tla.cross_flag("preloaded")
    begin = tla.cross_flag("begin")
    written = tla.cross_flag("written")
    ready = tla.cross_flag("ready")
    transfer = tla.flag("transfer", tla.arch.VECTOR, tla.arch.MTE3)
    ptr = tla.allocate(256, tla.Int32, tla.AddressSpace.ub, 512)
    ub = tla.make_tensor(ptr, tla.make_layout(tla.make_shape(64), tla.make_stride(1)))
    with tla.cube():
        result[0] = before
        tla.cross_core_wait_flag(preloaded, tla.arch.SCALAR)
        tla.cross_core_set_flag(begin, tla.arch.SCALAR)
    with tla.vector():
        result[(sub + 1) * 128] = before
        tla.cross_core_set_flag(preloaded, tla.arch.SCALAR)
        tla.cross_core_wait_flag(begin, tla.arch.SCALAR)
        with tla.vec.func():
            ub.store(tla.add(tla.full(2, tla.Int32), sub))
        tla.set_flag(transfer)
        tla.wait_flag(transfer)
        gm = tla.make_tensor(
            flags.ptr + sub * 128, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
        )
        tla.copy(gm, ub)
        tla.cross_core_set_flag(written, tla.arch.MTE3)
        tla.cross_core_wait_flag(ready, tla.arch.SCALAR)
        _mx_refresh_guard(flags.ptr)
        result[(sub + 1) * 128 + 1] = flags[0]
        result[(sub + 1) * 128 + 2] = flags[128]
    with tla.cube():
        tla.cross_core_wait_flag(written, tla.arch.SCALAR)
        tla.cross_core_set_flag(ready, tla.arch.SCALAR)
        _mx_refresh_guard(flags.ptr)
        result[1] = flags[0]
        result[2] = flags[128]


@pytest.mark.npu
def test_mxfp8_dma_guard_cache_visibility(device):
    flags = torch.ones(256, dtype=torch.int32, device=device)
    result = torch.zeros(384, dtype=torch.int32, device=device)
    bindings = [
        tla.from_dlpack(t, origin_shape=t.shape, layout_tag=tla.arch.RowMajor)
        for t in (flags, result)
    ]
    executor = tla.compile(_refresh_mxfp8_guard_cache, *bindings, options="--npu-arch 3510")
    executor(*bindings, block_num=1, stream=torch.npu.current_stream(device))
    # Cube and both vector sub-blocks must see the DMA-updated values.
    assert result.cpu().reshape(3, 128)[:, :3].tolist() == [[2, 2, 3]] * 3


@tla.kernel
def _poison_mxfp8_l1(marker: tla.Tensor):
    # Cover the whole attention arena without depending on its scale offsets.
    arena = tla.allocate(512 * 1024, tla.UInt8, tla.AddressSpace.l1, 512)
    with tla.cube():
        _mx_fill_scales(arena, tla.Int32(512 * 1024), tla.Int32(0xFFFF))
        _mx_fill_ready()
        # Keep each core's completion marker on a separate cache line.
        marker[tla.arch.block_idx() * 16] = tla.arch.block_idx() + 1


@pytest.fixture(scope="module")
def mxfp8_l1_poisoner(device):
    cores = runtime._cores(device)
    marker = torch.zeros(cores * 16, dtype=torch.int32, device=device)
    binding = tla.from_dlpack(marker, origin_shape=marker.shape, layout_tag=tla.arch.RowMajor)
    executor = tla.compile(_poison_mxfp8_l1, binding, options="--npu-arch 3510")

    def poison():
        stream = torch.npu.current_stream(device)
        marker.record_stream(stream)
        executor(binding, block_num=cores, stream=stream)

    yield poison
    assert marker.cpu().reshape(cores, 16)[:, 0].tolist() == list(range(1, cores + 1))


@pytest.fixture
def poisoned_mxfp8_l1(monkeypatch, mxfp8_l1_poisoner):
    """Expose uninitialized reads even when a previous launch left valid scales."""
    launch = runtime._launch

    def with_poison(executor, payload, **options):
        mxfp8_l1_poisoner()
        return launch(executor, payload, **options)

    monkeypatch.setattr(runtime, "_launch", with_poison)


@pytest.fixture(scope="module")
def sparse_case(request):
    return make_case(
        q_lengths=(2048,),
        k_lengths=(4096,),
        q_heads=4,
        kv_heads=2,
        layout=getattr(request, "param", "BSND"),
        seed=53,
    )


def _counted_sparse_blocks(capacity=32):
    counts = torch.arange(16, dtype=torch.int32).reshape(1, 1, 16).repeat(1, 4, 1)
    counts[0, 1] = 20 - counts[0, 1]
    counts[0, 2] = 0
    counts[0, 3] = capacity
    counts.clamp_(max=capacity)
    pc, fc = counts // 3, counts - counts // 3
    pi = torch.full((1, 4, 16, capacity), -193, dtype=torch.int32)
    fi = pi.clone()
    order = torch.arange(31, -1, -1, dtype=torch.int32)
    for h in range(4):
        for qb in range(16):
            partial, total = int(pc[0, h, qb]), int(counts[0, h, qb])
            pi[0, h, qb, :partial] = order[:partial]
            fi[0, h, qb, : total - partial] = order[partial:total]
    return BlockSparseTensorsTorch(pc, pi, fc, fi, block_size=(128, 128))


def _run_sparse_case(case, sparse, device, scale):
    qkv, kwargs = to_device(case, device)
    kwargs.pop("max_seqlen_q")
    kwargs.pop("max_seqlen_k")
    device_sparse = sparse._replace(
        **{
            name: t.to(device)
            for name, t in sparse._asdict().items()
            if isinstance(t, torch.Tensor)
        }
    )
    return interface._flash(
        *qkv,
        **kwargs,
        layout=case["layout"],
        softmax_scale=scale,
        causal=False,
        window_size=(None, None),
        softcap=0.0,
        score_mod=None,
        score_mod_bwd=None,
        mask_mod=None,
        aux_tensors=None,
        aux_scalars=None,
        block_sparse_tensors=device_sparse,
        block_sparse_tensors_bwd=None,
        return_lse=True,
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "sparse_case,kind,capacity",
    [
        ("BSND", "default", 6),
        ("BSND", "zero", 6),
        ("BSND", "negative", 6),
        ("BSND", "large", 6),
        ("BSND", "huge", 6),
        ("BSND", "qnan", 6),
        ("BSND", "qnan_negative", 6),
        ("BSND", "knan", 6),
        ("BSND", "knan_negative", 6),
        ("BSND", "vnan", 6),
        ("BSND", "all_negative", 6),
        ("BNSD", "default", 32),
        ("BNSD", "zero", 32),
        ("BNSD", "negative", 32),
        ("BNSD", "large", 32),
        ("BNSD", "huge", 32),
        ("BNSD", "qnan", 32),
        ("BNSD", "qnan_negative", 32),
        ("BNSD", "knan", 32),
        ("BNSD", "knan_negative", 32),
        ("BNSD", "vnan", 32),
        ("BNSD", "all_negative", 32),
    ],
    indirect=["sparse_case"],
)
@pytest.mark.usefixtures("poisoned_mxfp8_l1")
def test_sparse_counts_scales_and_nonfinite(device, sparse_case, kind, capacity):
    case = copy.deepcopy(sparse_case)
    scale = {"zero": 0.0, "negative": -0.125}.get(kind)
    if kind in ("large", "huge"):
        case["kwargs"]["q_descale"] += 12 if kind == "large" else 32
    if "nan" in kind:
        which = {"q": 0, "k": 1, "v": 2}[kind[0]]
        position = 128 if which == 0 else 4095
        index = (0, position, 0, 0) if case["layout"] == "BSND" else (0, 0, position, 0)
        case["qkv"][which].view(torch.uint8)[index] = 255 if kind.endswith("negative") else 127
    if kind == "all_negative":
        case["qkv"][0].view(torch.uint8).fill_(0x38)
        case["qkv"][1].view(torch.uint8).fill_(0xB8)
        case["kwargs"]["q_descale"].fill_(127)
        case["kwargs"]["k_descale"].fill_(127)
    sparse = _counted_sparse_blocks(capacity)
    actual, lse = _run_sparse_case(case, sparse, device, scale)
    rows = [128 * block + offset for block in range(16) for offset in (0, 127)]
    expected, expected_lse = reference(case, sparse=sparse, scale=scale, query_rows=[rows])
    actual = actual.cpu()
    actual = actual[:, rows] if case["layout"] == "BSND" else actual[:, :, rows]
    expected = expected[:, rows] if case["layout"] == "BSND" else expected[:, :, rows]
    lse, expected_lse = lse.cpu()[:, :, rows], expected_lse[:, :, rows]
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isnan(lse), torch.isnan(expected_lse))
    assert torch.equal(torch.isneginf(lse), torch.isneginf(expected_lse))
    assert torch.equal(torch.isfinite(lse), torch.isfinite(expected_lse))
    check(actual[torch.isfinite(expected)], expected[torch.isfinite(expected)])
    finite = torch.isfinite(expected_lse)
    check(lse[finite].to(torch.float16), expected_lse[finite])
    assert actual.dtype == torch.bfloat16 and lse.dtype == torch.float32


@pytest.mark.npu
@pytest.mark.usefixtures("poisoned_mxfp8_l1")
def test_post_quantized_empty_rows_with_stale_l1(device, sparse_case):
    # Packed masks with capacity >= 8 also use PV to calculate the denominator.
    # All mask bits are visible; the explicit empty/full/partial lists win.
    qlen, klen = sparse_case["q_lengths"][0], sparse_case["k_lengths"][0]
    words = torch.full((qlen, (klen + 31) // 32), -1, dtype=torch.int32).view(torch.uint32)

    def mask_ref(b, h, q, k, info, aux, scalars):
        return torch.ones_like(q + k, dtype=torch.bool)

    _verify(
        sparse_case,
        device,
        packed_visibility,
        mask_ref,
        _device_sparse(_counted_sparse_blocks(8), device),
        auxiliary=(words,),
        normalize_quantized=True,
    )


@pytest.mark.npu
@pytest.mark.parametrize("score", [4.12, 4.125, -1.252, -1.254, -1.945, -1.947])
def test_short_probability_guard_boundaries(device, sparse_case, score):
    """Cross both repair thresholds with a constant-score, analytically stable row."""
    case = copy.deepcopy(sparse_case)
    for operand in case["qkv"][:2]:
        operand.view(torch.uint8).fill_(0x38)  # E4M3FN 1.0: QK is exactly 128.
    for name in ("q_descale", "k_descale"):
        case["kwargs"][name].fill_(127)
    sparse = _counted_sparse_blocks(6)
    scale = score / 128
    actual, lse = _run_sparse_case(case, sparse, device, scale)
    rows = [0, 127, 128, 1024, 2047]
    expected, expected_lse = reference(case, sparse=sparse, scale=scale, query_rows=[rows])
    if case["layout"] == "BSND":
        check(actual[:, rows], expected[:, rows])
    else:
        check(actual[:, :, rows], expected[:, :, rows])
    check(lse[:, :, rows].to(torch.float16), expected_lse[:, :, rows])


@pytest.mark.npu
@pytest.mark.parametrize("capacity", [None, 6])
def test_sparse_compile_rebinds_scales(device, capacity):
    from flash_attn_npu_4.catlass_dsl import (
        flash_attn_func_with_quant as flash_attn_func,
        runtime,
    )

    from .mxfp8_reference import SCALE_NAMES

    sparse = None
    device_sparse = None
    if capacity is not None:
        counts = torch.full((1, 1, 16), capacity, dtype=torch.int32)
        indices = torch.arange(capacity, dtype=torch.int32).expand(1, 1, 16, capacity).contiguous()
        sparse = BlockSparseTensorsTorch(
            torch.zeros_like(counts),
            torch.zeros_like(indices),
            counts,
            indices,
            block_size=(128, 128),
        )
        device_sparse = sparse._replace(
            **{
                name: t.to(device)
                for name, t in sparse._asdict().items()
                if isinstance(t, torch.Tensor)
            }
        )

    def call(q, k, v, qs, ks, vs):
        return flash_attn_func(
            q,
            k,
            v,
            q_descale=qs,
            k_descale=ks,
            v_descale=vs,
            block_sparse_tensors=device_sparse,
            return_lse=True,
        )

    invoke = torch.compile(call, backend="inductor", fullgraph=True, dynamic=True)
    case = make_case(q_lengths=(2048,), k_lengths=(2048,), q_heads=2, kv_heads=1)
    rows = [0, 127, 128, 1024, 2047]
    keys = None
    for power in (0, 12):
        current = copy.deepcopy(case)
        current["kwargs"]["q_descale"] += power
        inputs, kwargs = to_device(current, device)
        out, lse = invoke(*inputs, *(kwargs[n] for n in SCALE_NAMES))
        wanted, el = reference(current, sparse=sparse, query_rows=[rows])
        check(out[:, rows], wanted[:, rows])
        check(lse[:, :, rows].to(torch.float16), el[:, :, rows])
        if keys is not None:
            assert set(runtime.compile_cache) == keys
        keys = set(runtime.compile_cache)


@pytest.mark.npu
@pytest.mark.parametrize("capacity", [6, 32])
def test_sparse_concurrent_streams_own_repair_flags(device, capacity):
    streams = [torch.npu.Stream(device=device) for _ in range(4)]
    pending = []
    sparse = _counted_sparse_blocks(capacity)
    for index, stream in enumerate(streams * 2):
        case = make_case(
            q_lengths=(2048,), k_lengths=(4096,), q_heads=4, kv_heads=2, seed=910 + index
        )
        if index % 2:
            case["kwargs"]["q_descale"] += 12
        with torch.npu.stream(stream):
            out, lse = _run_sparse_case(case, sparse, device, None)
        pending.append((case, out, lse))
    for stream in streams:
        torch.npu.current_stream(device).wait_stream(stream)
    rows = [0, 128, 256, 1024, 2047]
    for case, out, lse in pending:
        wanted, el = reference(case, sparse=sparse, query_rows=[rows])
        check(out[:, rows], wanted[:, rows])
        check(lse[:, :, rows].to(torch.float16), el[:, :, rows])


@pytest.mark.npu
def test_short_phase_handoff_across_layouts(device):
    # Keep the previously deadlocking sequence in one test so its coverage
    # does not depend on pytest's ordering of parametrized cases.
    # The same rows also cover one/two-slot scale reuse across query tiles.
    rows = [128 * block + offset for block in range(16) for offset in (0, 127)]
    for layout, capacity in [("BSND", 8), ("BNSD", 1), ("BSND", 2), ("BNSD", 2), ("BNSD", 8)]:
        case = make_case(
            q_lengths=(2048,), k_lengths=(4096,), q_heads=4, kv_heads=2, layout=layout, seed=53
        )
        sparse = _counted_sparse_blocks(capacity)
        out, lse = _run_sparse_case(case, sparse, device, None)
        out, lse = out.cpu(), lse.cpu()
        wanted, expected_lse = reference(case, sparse=sparse, query_rows=[rows])
        if layout == "BSND":
            check(out[:, rows], wanted[:, rows])
        else:
            check(out[:, :, rows], wanted[:, :, rows])
        check(lse[:, :, rows].to(torch.float16), expected_lse[:, :, rows])


@pytest.mark.npu
@pytest.mark.parametrize(
    "layout,tail",
    [
        ("BSND", 1),
        ("BNSD", 31),
        ("BSND", 32),
        ("BNSD", 33),
        ("BSND", 63),
        ("BNSD", 64),
        ("BSND", 65),
        ("BNSD", 127),
    ],
)
def test_continuous_sparse_tails(device, layout, tail):
    """Tail Q lanes, local V group32 boundaries, and empty/partial/full lists."""
    qlen, klen = 1024 + tail, 2048 + tail
    case = make_case(
        q_lengths=(qlen,), k_lengths=(klen,), q_heads=4, kv_heads=2, layout=layout, seed=72 + tail
    )
    blocks, capacity = (qlen + 127) // 128, (klen + 127) // 128
    count = torch.arange(blocks, dtype=torch.int32).reshape(1, 1, blocks)
    count = count.expand(1, 4, blocks).clone()
    count[:, 1] = 1
    count[:, 2] = capacity
    pc, fc = count // 2, count - count // 2
    pi = torch.full((1, 4, blocks, capacity), -1, dtype=torch.int32)
    fi = pi.clone()
    order = torch.arange(capacity - 1, -1, -1, dtype=torch.int32)
    for head in range(4):
        for qb in range(blocks):
            partial, total = int(pc[0, head, qb]), int(count[0, head, qb])
            pi[0, head, qb, :partial] = order[:partial]
            fi[0, head, qb, : total - partial] = order[partial:total]
    sparse = BlockSparseTensorsTorch(pc, pi, fc, fi, block_size=(128, 128))
    rows = sorted(set([0, 127, 128, 1023, 1024, qlen - 1]))
    # Reuse the same compiled shape with normal, overflow and nonfinite data.
    for variant in ("normal", "overflow", "nan"):
        current = copy.deepcopy(case)
        if variant == "overflow":
            current["kwargs"]["q_descale"] += 12
        elif variant == "nan":
            index = (0, klen - 1, 0, 0) if layout == "BSND" else (0, 0, klen - 1, 0)
            current["qkv"][1].view(torch.uint8)[index] = 127
        out, lse = _run_sparse_case(current, sparse, device, None)
        expected, expected_lse = reference(current, sparse=sparse, query_rows=[rows])
        out, lse = out.cpu(), lse.cpu()[:, :, rows]
        out = out[:, rows] if layout == "BSND" else out[:, :, rows]
        expected = expected[:, rows] if layout == "BSND" else expected[:, :, rows]
        expected_lse = expected_lse[:, :, rows]
        assert torch.equal(torch.isnan(out), torch.isnan(expected))
        assert torch.equal(torch.isnan(lse), torch.isnan(expected_lse))
        assert torch.equal(torch.isneginf(lse), torch.isneginf(expected_lse))
        finite = torch.isfinite(expected)
        check(out[finite], expected[finite])
        finite = torch.isfinite(expected_lse)
        check(lse[finite].to(torch.float16), expected_lse[finite])


# Mask callbacks and auxiliary storage.


@simt
@tla.jit
def local_window(b, h, q, k, info, aux, scalars):
    return (q >= k) & (q - k < 128)


@simt
@tla.jit
def packed_visibility(b, h, q, k, info, aux, scalars):
    word = aux[0][q, k // 32]
    return ((word >> (k % 32).to(tla.Int64)) & tla.Int64(1)) != tla.Int64(0)


@simt
@tla.jit
def partial_visibility(b, h, q, k, info, aux, scalars):
    return (q % 17 != 0) & ((q + k) % 3 != 0)


def _cpu_sparse(sparse):
    return sparse._replace(
        **{
            name: value.cpu()
            for name, value in sparse._asdict().items()
            if isinstance(value, torch.Tensor)
        }
    )


def _verify(
    case,
    device,
    mask,
    mask_ref,
    sparse,
    auxiliary=(),
    scale=None,
    scalars=(),
    normalize_quantized=False,
):
    inputs, scales = to_device(case, device)
    scales = {name: scales[name] for name in SCALE_NAMES}
    length = case["q_lengths"][0]
    rows = sorted({0, 31, 63, 64, 127, 128, 601, length - 1} | {i for i in range(0, length, 67)})
    cpu_aux = tuple(t.long() if t.dtype == torch.uint32 else t for t in auxiliary)
    expected, expected_lse = reference(
        case,
        mask_mod=mask_ref,
        sparse=_cpu_sparse(sparse),
        aux=cpu_aux,
        scalars=scalars,
        scale=scale,
        query_rows=[rows],
        normalize_quantized=normalize_quantized,
    )
    previous = None
    for _ in range(3):
        out, lse = flash_attn_func(
            *inputs,
            **scales,
            mask_mod=mask,
            block_sparse_tensors=sparse,
            softmax_scale=scale,
            aux_tensors=tuple(t.to(device) for t in auxiliary),
            aux_scalars=scalars,
            return_lse=True,
        )
        out, lse = out.cpu(), lse.cpu()
        assert torch.isfinite(out).all()
        assert not torch.isnan(lse).any()
        check(out[:, rows], expected[:, rows])
        if normalize_quantized:
            check_quantized_lse(lse[:, :, rows], expected_lse[:, :, rows])
        else:
            check(lse[:, :, rows].to(torch.float16), expected_lse[:, :, rows])
        if previous is not None:
            assert torch.equal(out, previous)
        previous = out


@pytest.mark.npu
@pytest.mark.parametrize("length,kind", [(1025, "window"), (4096, "causal"), (1025, "packed")])
def test_masked_pipeline_repeated_and_tail(device, length, kind):
    case = make_case(q_lengths=(length,), k_lengths=(length,), q_heads=4, kv_heads=2, seed=7)
    auxiliary = ()
    if kind == "window":
        mask = local_window

        def mask_ref(b, h, q, k, info, aux, scalars):
            return (q >= k) & (q - k < 128)
    elif kind == "causal":
        mask = causal()

        def mask_ref(b, h, q, k, info, aux, scalars):
            return q >= k
    else:
        mask = packed_visibility
        words = torch.randint(
            0,
            2**32 - 1,
            (length, (length + 31) // 32),
            dtype=torch.int64,
            generator=torch.Generator().manual_seed(818),
        )
        auxiliary = (words.to(torch.uint32),)

        def mask_ref(b, h, q, k, info, aux, scalars):
            return ((aux[0][q, k // 32] >> (k % 32)) & 1) != 0

    sparse = compute_block_sparsity(
        128, 128, 1, 1, length, length, mask, tuple(t.to(device) for t in auxiliary), device
    )
    _verify(case, device, mask, mask_ref, sparse, auxiliary, normalize_quantized=kind == "packed")


@pytest.mark.npu
@pytest.mark.parametrize("key_length,scale", [(4096, None), (4225, 0.0), (4225, -0.125)])
@pytest.mark.usefixtures("poisoned_mxfp8_l1")
def test_masked_pipeline_authoritative_lists(device, scale, key_length):
    # Keep seed 51 and the first-launch assertion: 4096/4096 has exposed NaNs.
    case = make_case(q_lengths=(4096,), k_lengths=(key_length,), q_heads=4, kv_heads=2, seed=51)
    counts = torch.arange(32, dtype=torch.int32)[None, None, :] % 6
    pc, fc = (counts > 0).int(), torch.clamp(counts - 1, min=0)
    pi = torch.full((1, 1, 32, 1), 1, dtype=torch.int32)
    fi = (
        torch.tensor([(key_length - 1) // 128, 2, 0, 7], dtype=torch.int32)
        .expand(1, 1, 32, 4)
        .contiguous()
    )
    sparse = BlockSparseTensorsTorch(
        pc.to(device), pi.to(device), fc.to(device), fi.to(device), block_size=(128, 128)
    )

    def mask_ref(b, h, q, k, info, aux, scalars):
        return (q % 17 != 0) & ((q + k) % 3 != 0)

    _verify(case, device, partial_visibility, mask_ref, sparse, scale=scale)


@pytest.mark.npu
@pytest.mark.parametrize(
    "kind,length",
    [
        ("reverse_window", 1025),
        ("prefix", 1024),
        ("head", 1024),
        ("branch", 1025),
        ("scalar", 1024),
    ],
)
def test_masked_pipeline_affine_and_generic_callbacks(device, kind, length):
    """Fused bounds and unsupported expressions preserve the callback semantics."""

    @simt
    @tla.jit
    def mask(b, h, q, k, info, aux, scalars):
        visible = tla.Bool(False)
        if tla.const_expr(kind == "reverse_window"):
            visible = (k - q <= 63) & (k - q > -193)
        elif tla.const_expr(kind == "prefix"):
            visible = ((q < 128) & (k < 128)) | (q >= k)
        elif tla.const_expr(kind == "head"):
            visible = ((h == 0) & (q >= k)) | ((h != 0) & (k >= q))
        elif tla.const_expr(kind == "branch"):
            if q >= k:
                visible = q - k < 193
        else:
            visible = (q >= k) & (q - k < scalars[0])
        return visible

    def mask_ref(b, h, q, k, info, aux, scalars):
        if kind == "reverse_window":
            return (k - q <= 63) & (k - q > -193)
        if kind == "prefix":
            return ((q < 128) & (k < 128)) | (q >= k)
        if kind == "head":
            return (q >= k) if h == 0 else (k >= q)
        return (q >= k) & (q - k < 193)

    scalars = (193,) if kind == "scalar" else ()
    case = make_case(q_lengths=(length,), k_lengths=(length,), q_heads=2, kv_heads=1, seed=91)
    sparse = compute_block_sparsity(
        128, 128, 1, 2, length, length, mask, (), device, aux_scalars=scalars
    )
    _verify(case, device, mask, mask_ref, sparse, scalars=scalars)


@simt
@tla.jit
def labels_rank1(b, h, q, k, info, aux, scalars):
    return aux[0][q] == aux[1][k]


def _static_label_callback(kind):
    @tla.jit
    def load_label(tensor, indices):
        return tensor[indices]

    @simt
    @tla.jit
    def callback(b, h, q, k, info, aux, scalars):
        visible = tla.Bool(False)
        if tla.const_expr(kind == "labels"):
            visible = load_label(aux[0], (b, h, q)) == load_label(aux[1], (b, h, k))
        return visible

    return callback


@simt
@tla.jit
def shifted_labels(b, h, q, k, info, aux, scalars):
    return aux[0][(q + 1) % info[0]] == aux[1][k]


@simt
@tla.jit
def branching_labels(b, h, q, k, info, aux, scalars):
    visible = tla.Bool(False)
    if aux[0][q] == aux[1][k]:
        visible = q >= k
    return visible


@pytest.mark.npu
@pytest.mark.parametrize("rank", [1, 2, 3, "indirect", "branch", "repair"])
def test_masked_pipeline_label_aux_updates(device, rank):
    """Label values and offsets remain device inputs on a prepared-plan hit."""
    length, heads = 2048, 2
    case = make_case(q_lengths=(length,), k_lengths=(length,), q_heads=heads, kv_heads=1, seed=71)
    inputs, scales = to_device(case, device)
    scales = {name: scales[name] for name in SCALE_NAMES}
    # The low 32 bits deliberately collide; signed high bits distinguish IDs.
    labels = ((torch.arange(length) // 64) - 16) * (1 << 35) + 7
    if rank == 2:
        labels = labels[None, :]
    elif rank == 3:
        labels = torch.stack((labels, labels ^ (1 << 50)))[None, :, :]
    cpu_aux = (labels.contiguous().clone(), labels.contiguous().clone())
    aux = tuple(t.to(device) for t in cpu_aux)
    mask = {
        1: labels_rank1,
        2: batch_labels,
        3: head_labels,
        "indirect": shifted_labels,
        "branch": branching_labels,
        "repair": labels_rank1,
    }[rank]

    def mask_ref(b, h, q, k, info, tensors, scalars):
        if rank in (1, "repair"):
            return tensors[0][q] == tensors[1][k]
        if rank == "branch":
            return (tensors[0][q] == tensors[1][k]) & (q >= k)
        if rank == 2:
            return tensors[0][b, q] == tensors[1][b, k]
        if rank == 3:
            return tensors[0][b, h, q] == tensors[1][b, h, k]
        return tensors[0][(q + 1) % info[0]] == tensors[1][k]

    sparse = compute_block_sparsity(
        128, 128, 1, heads if rank == 3 else 1, length, length, mask, aux, device
    )
    rows = [0, 1, 31, 32, 63, 64, 127, 128, 511, 1023, length - 1]
    scale = 100.0 if rank == "repair" else None
    for shift in (0, 32):
        cpu_aux[0].copy_(torch.roll(labels, shift, dims=-1))
        aux[0].copy_(cpu_aux[0])
        expected, expected_lse = reference(
            case,
            mask_mod=mask_ref,
            sparse=_cpu_sparse(sparse),
            aux=cpu_aux,
            query_rows=[rows],
            scale=scale,
        )
        out, lse = flash_attn_func(
            *inputs,
            **scales,
            mask_mod=mask,
            aux_tensors=aux,
            block_sparse_tensors=sparse,
            softmax_scale=scale,
            return_lse=True,
        )
        out, lse = out.cpu(), lse.cpu()
        assert torch.isfinite(out).all()
        assert not torch.isnan(lse).any()
        check(out[:, rows], expected[:, rows])
        check(lse[:, :, rows].to(torch.float16), expected_lse[:, :, rows])


@pytest.mark.npu
@pytest.mark.parametrize(
    "qn,kn,rank,dtype,heads,kv_heads,batch,use_helper",
    [
        (16384, 16513, 1, torch.int32, 4, 2, 1, False),
        (4097, 4225, 3, torch.int64, 6, 2, 2, False),
        (1025, 1153, 3, torch.int32, 4, 2, 2, True),
    ],
)
def test_masked_pipeline_tiled_aux_cache(
    device, monkeypatch, qn, kn, rank, dtype, heads, kv_heads, batch, use_helper
):
    """Auxiliary staging must scale with a tile, across tails and GQA batches."""
    from flash_attn_npu_4.catlass_dsl import runtime

    compiled_cache_modes = []
    compile_original = runtime._compile_mxfp8

    def checked_compile(args, index_shape):
        compiled_cache_modes.append(args[-3])
        return compile_original(args, index_shape)

    monkeypatch.setattr(runtime, "_compile_mxfp8", checked_compile)
    case = make_case(
        q_lengths=(qn,) * batch, k_lengths=(kn,) * batch, q_heads=heads, kv_heads=kv_heads, seed=89
    )
    inputs, scales = to_device(case, device)
    scales = {name: scales[name] for name in SCALE_NAMES}
    labels = torch.arange(max(qn, kn)) // 64
    if dtype == torch.int64:
        labels = (labels - 64) * (1 << 35) + 7
    if rank == 3:
        labels = labels[None, None, :] + torch.arange(batch * heads).reshape(batch, heads, 1)
    labels = labels.to(dtype).contiguous()
    cpu_aux = (labels.clone(), labels.clone())
    aux = tuple(t.to(device) for t in cpu_aux)
    mask = labels_rank1 if rank == 1 else head_labels
    if use_helper:
        mask = _static_label_callback("labels")

    def mask_ref(b, h, q, k, info, tensors, scalars):
        if rank == 1:
            return tensors[0][q] == tensors[1][k]
        return tensors[0][b, h, q] == tensors[1][b, h, k]

    sparse = compute_block_sparsity(
        128, 128, batch, heads if rank == 3 else 1, qn, kn, mask, aux, device
    )
    rows = sorted({0, 31, 63, 64, 127, 128, qn // 2, qn - 2, qn - 1})
    for shift in (0, 17):
        # Keep the sparse lists authoritative while changing device data on a
        # prepared-plan hit; some previously visible rows become empty.
        cpu_aux[0].copy_(torch.roll(labels, shift, dims=-1))
        aux[0].copy_(cpu_aux[0])
        expected, expected_lse = reference(
            case,
            mask_mod=mask_ref,
            sparse=_cpu_sparse(sparse),
            aux=cpu_aux,
            query_rows=[rows] * batch,
        )
        out, lse = flash_attn_func(
            *inputs,
            **scales,
            mask_mod=mask,
            aux_tensors=aux,
            block_sparse_tensors=sparse,
            return_lse=True,
        )
        out, lse = out.cpu(), lse.cpu()
        assert torch.isfinite(out).all()
        assert not torch.isnan(lse).any()
        check(out[:, rows], expected[:, rows])
        check(lse[:, :, rows].to(torch.float16), expected_lse[:, :, rows])
    # A successful generic fallback would hide a broken tiled specialization.
    assert compiled_cache_modes and all(meta and meta[0][0] < 0 for meta in compiled_cache_modes)


# Packed sequence boundaries and tails.


@simt
@tla.jit
def shifted_window(b, h, q, k, info, aux, scalars):
    center = q + info[1] - info[0]
    return (k <= center) & (k > center - 128)


def _blocks(case, mask, auxiliary):
    rows = (max(case["q_lengths"]) + 127) // 128
    columns = (max(case["k_lengths"]) + 127) // 128
    shape = (len(case["q_lengths"]), 1, rows)
    pc = torch.zeros(shape, dtype=torch.int32)
    fc = torch.zeros_like(pc)
    pi = torch.full((*shape, columns), -111, dtype=torch.int32)
    fi = torch.full_like(pi, -111)
    qoff = koff = 0
    for batch, (qn, kn) in enumerate(zip(case["q_lengths"], case["k_lengths"])):
        info = (qn, kn, qoff, koff)
        for qb in range((qn + 127) // 128):
            queries = torch.arange(qb * 128, min(qn, (qb + 1) * 128))[:, None]
            for kb in reversed(range((kn + 127) // 128)):
                keys = torch.arange(kb * 128, min(kn, (kb + 1) * 128))[None, :]
                visible = mask(batch, 0, queries, keys, info, auxiliary, ())
                if visible.any():
                    counts, indices = (fc, fi) if visible.all() else (pc, pi)
                    indices[batch, 0, qb, counts[batch, 0, qb]] = kb
                    counts[batch, 0, qb] += 1
        qoff += qn
        koff += kn
    return BlockSparseTensorsTorch(pc, pi, fc, fi, block_size=(128, 128))


@pytest.mark.npu
@pytest.mark.parametrize(
    "kind,scale", [("window", None), ("global_documents", 0.0), ("global_documents", -0.125)]
)
def test_packed_sparse_offsets_empty_segments_and_local_tails(device, kind, scale):
    case = make_case(
        q_lengths=(0, 65, 129, 257),
        k_lengths=(33, 129, 0, 385),
        q_packed=True,
        k_packed=True,
        q_heads=4,
        kv_heads=2,
        v_scale_capacity=20,
        seed=617,
    )
    auxiliary = ()
    if kind == "window":
        native = shifted_window

        def mask(b, h, q, k, info, aux, scalars):
            center = q + info[1] - info[0]
            return (k <= center) & (k > center - 128)
    else:
        native = global_labels
        auxiliary = (
            torch.arange(sum(case["q_lengths"]), dtype=torch.int32) // 73,
            torch.arange(sum(case["k_lengths"]), dtype=torch.int32) // 73,
        )

        def mask(b, h, q, k, info, aux, scalars):
            return aux[0][q + info[2]] == aux[1][k + info[3]]

    sparse = _blocks(case, mask, auxiliary)
    expected, logs = reference(case, mask_mod=mask, sparse=sparse, aux=auxiliary, scale=scale)
    inputs, kwargs = to_device(case, device)
    for _ in range(2):
        output, lse = flash_attn_varlen_func(
            *inputs,
            **kwargs,
            mask_mod=native,
            block_sparse_tensors=_device_sparse(sparse, device),
            aux_tensors=tuple(t.to(device) for t in auxiliary),
            softmax_scale=scale,
            return_lse=True,
        )
        check(output, expected)
        check_quantized_lse(lse.cpu(), logs)
        assert not output[65:194].count_nonzero()
        assert torch.isneginf(lse[:, 65:194]).all()


@pytest.mark.npu
def test_many_packed_segments_advance_over_empty_queries(device):
    q_lengths = tuple((0, 1, 31, 32, 33)[i % 5] for i in range(65))
    k_lengths = tuple((0, 31, 32, 33, 63, 64, 65)[i % 7] for i in range(65))
    case = make_case(
        q_lengths=q_lengths,
        k_lengths=k_lengths,
        q_packed=True,
        k_packed=True,
        q_heads=2,
        kv_heads=1,
        v_scale_capacity=130,
        seed=925,
    )
    expected, logs = reference(case)
    inputs, kwargs = to_device(case, device)
    output, lse = flash_attn_varlen_func(*inputs, **kwargs, return_lse=True)
    check(output, expected)
    check_quantized_lse(lse.cpu(), logs)


@pytest.mark.npu
@pytest.mark.parametrize("key_sign", [-1.0, 1.0])
def test_packed_tail_repairs_probability_underflow_and_saturation(device, key_sign):
    case = make_case(
        q_lengths=(129, 65),
        k_lengths=(257, 31),
        q_packed=True,
        k_packed=True,
        q_heads=2,
        kv_heads=1,
        seed=926,
    )
    for tensor, value in zip(case["qkv"][:2], (1.0, key_sign)):
        code = torch.tensor(value).to(torch.float8_e4m3fn).view(torch.uint8)
        tensor.view(torch.uint8).fill_(int(code))
    case["kwargs"]["q_descale"].fill_(127)
    case["kwargs"]["k_descale"].fill_(127)
    expected, logs = reference(case)
    inputs, kwargs = to_device(case, device)
    output, lse = flash_attn_varlen_func(*inputs, **kwargs, return_lse=True)
    check(output, expected)
    check_quantized_lse(lse.cpu(), logs)


@pytest.mark.npu
def test_packed_dense_long_pipeline_lookahead(device):
    case = make_case(
        q_lengths=(2177, 65),
        k_lengths=(2305, 129),
        q_packed=True,
        k_packed=True,
        q_heads=2,
        kv_heads=1,
        seed=928,
    )
    expected, logs = reference(case)
    inputs, kwargs = to_device(case, device)
    output, lse = flash_attn_varlen_func(*inputs, **kwargs, return_lse=True)
    check(output, expected)
    check_quantized_lse(lse.cpu(), logs)


@pytest.mark.npu
def test_packed_sparse_order_and_remainder(device):
    case = make_case(
        q_lengths=(1153, 129),
        k_lengths=(1281, 65),
        q_heads=4,
        kv_heads=2,
        q_packed=True,
        k_packed=True,
        seed=935,
    )

    def mask(b, h, q, k, info, aux, scalars):
        center = q + info[1] - info[0]
        return (k <= center) & (k > center - 128)

    sparse = _blocks(case, mask, ())
    sparse.mask_block_cnt[0, 0, 2] = 0
    sparse.full_block_cnt[0, 0, 2] = 0
    expected, logs = reference(case, mask_mod=mask, sparse=sparse)
    inputs, kwargs = to_device(case, device)
    output, lse = flash_attn_varlen_func(
        *inputs,
        **kwargs,
        mask_mod=shifted_window,
        block_sparse_tensors=_device_sparse(sparse, device),
        return_lse=True,
    )
    check(output, expected)
    check_quantized_lse(lse.cpu(), logs)


@pytest.mark.npu
@pytest.mark.parametrize(
    "kind", ["mini_causal", "block_diagonal", "dilated", "negative_floor", "negative_mod"]
)
def test_packed_periodic_masks(device, kind):
    @simt
    @tla.jit
    def mask(b, h, q, k, info, aux, scalars):
        if tla.const_expr(kind == "mini_causal"):
            return q % 128 >= k % 128
        elif tla.const_expr(kind == "block_diagonal"):
            return q // 128 == k // 128
        elif tla.const_expr(kind == "negative_floor"):
            return (q - k) // 128 == -1
        elif tla.const_expr(kind == "negative_mod"):
            return (q - k) % 8 == 3
        else:
            return (q >= k) & (q - k < 256) & ((q - k) % 2 == 0)

    def expected_mask(b, h, q, k, info, aux, scalars):
        if kind == "mini_causal":
            return q % 128 >= k % 128
        if kind == "block_diagonal":
            return q // 128 == k // 128
        if kind == "negative_floor":
            return (q - k) // 128 == -1
        if kind == "negative_mod":
            return (q - k) % 8 == 3
        return (q >= k) & (q - k < 256) & ((q - k) % 2 == 0)

    traced = _mx_trace_affine_mask(mask.simt, 1024, 1089, 0, packed=True)
    assert traced is not None
    span = traced[2]
    assert (span is None) == (kind in ("mini_causal", "negative_mod"))
    case = make_case(
        q_lengths=(513, 65),
        k_lengths=(641, 129),
        q_packed=True,
        k_packed=True,
        q_heads=4,
        kv_heads=2,
        seed=282,
    )
    sparse = _blocks(case, expected_mask, ())
    expected, logs = reference(case, mask_mod=expected_mask, sparse=sparse)
    inputs, kwargs = to_device(case, device)
    output, lse = flash_attn_varlen_func(
        *inputs,
        **kwargs,
        mask_mod=mask,
        block_sparse_tensors=_device_sparse(sparse, device),
        return_lse=True,
    )
    check(output, expected)
    check_quantized_lse(lse.cpu(), logs)


@simt
@tla.jit
def bitmask(b, h, q, k, info, aux, scalars):
    return ((aux[0][b, q, k // 32] >> (k % 32).to(tla.Int64)) & tla.Int64(1)) != tla.Int64(0)


@pytest.mark.npu
@pytest.mark.parametrize("qlens,klens", [((257, 65), (385, 129)), ((1025, 129), (257, 65))])
def test_tnd_packed_bits_pairs_singletons_and_local_tails(device, qlens, klens):
    case = make_case(
        q_lengths=qlens,
        k_lengths=klens,
        q_packed=True,
        k_packed=True,
        q_heads=4,
        kv_heads=2,
        seed=933,
    )
    rng = torch.Generator().manual_seed(933)
    words = torch.randint(
        0, 2**32, (2, max(qlens), (max(klens) + 31) // 32), generator=rng, dtype=torch.int64
    )
    words[:, 7::11] = 0

    def mask(b, h, q, k, info, aux, scalars):
        return ((aux[0][b, q, k // 32] >> (k % 32)) & 1) != 0

    sparse = _blocks(case, mask, (words,))
    # Exercise one- and two-tile partial lists as well as their reversed order.
    sparse.mask_block_cnt[0, 0, 0] = 1
    expected, logs = reference(case, mask_mod=mask, aux=(words,), sparse=sparse)
    inputs, kwargs = to_device(case, device)
    for _ in range(3):
        output, lse = flash_attn_varlen_func(
            *inputs,
            **kwargs,
            mask_mod=bitmask,
            aux_tensors=(words.to(torch.uint32).to(device),),
            block_sparse_tensors=_device_sparse(sparse, device),
            return_lse=True,
        )
        check(output, expected)
        check_quantized_lse(lse.cpu(), logs)


@pytest.mark.npu
def test_repair_gaps_cross_ring_boundary_and_rebind_scales(device):
    # Head-major scheduling gives each core 16 tasks. The selected heads are
    # separated by a full eight-slot state ring; using physical task ordinals
    # for the compacted pipeline would overwrite a still-live row state.
    case = make_case(
        q_lengths=(4096,),
        k_lengths=(128,),
        q_heads=16,
        kv_heads=16,
        q_packed=True,
        k_packed=True,
        seed=5289,
    )
    inputs, options = to_device(case, device)
    baseline, baseline_lse = flash_attn_varlen_func(*inputs, **options, return_lse=True)
    baseline, baseline_lse = baseline.cpu(), baseline_lse.cpu()
    rows = [0, 31, 63, 64, 127, 128, 1023, 4095]
    for heads in ((0, 8, 15), (1, 9, 14)):
        changed = copy.deepcopy(case)
        changed["kwargs"]["q_descale"][:, heads] = 140
        expected, expected_lse = reference(changed, query_rows=[rows])
        # Reuse the same device scale storage: task selection must be rebuilt.
        options["q_descale"].copy_(changed["kwargs"]["q_descale"])
        output, lse = flash_attn_varlen_func(*inputs, **options, return_lse=True)
        output, lse = output.cpu(), lse.cpu()
        check(output[rows][:, heads], expected[rows][:, heads])
        check_quantized_lse(lse[list(heads)][:, rows], expected_lse[list(heads)][:, rows])
        untouched = [head for head in range(16) if head not in heads]
        assert torch.equal(output[:, untouched], baseline[:, untouched])
        assert torch.equal(lse[untouched], baseline_lse[untouched])


# Global auxiliary indices and cached metadata.


@pytest.fixture
def output_cache_policy(request, monkeypatch):
    """Exercise both output DMA policies without requiring large test tensors."""
    runtime.clear_compile_cache()
    monkeypatch.setattr(runtime, "_mxfp8_stream_output", lambda *args: request.param)
    original = runtime._compile_mxfp8

    def native_only(arguments, *rest, **options):
        assert arguments[-7] == "exact_aux"
        assert arguments[-4].stream_output == request.param
        try:
            return original(arguments, *rest, **options)
        except Exception as exc:
            pytest.fail(f"Native compilation failed: {exc}")

    monkeypatch.setattr(runtime, "_compile_mxfp8", native_only)
    yield request.param
    runtime.clear_compile_cache()


@simt
@tla.jit
def global_labels(b, h, q, k, info, aux, scalars):
    return aux[0][q + info[2]] == aux[1][k + info[3]]


@simt
@tla.jit
def global_window(b, h, q, k, info, aux, scalars):
    return (q >= k) & (q - k <= aux[0][q + info[2]])


@simt
@tla.jit
def global_window_64(b, h, q, k, info, aux, scalars):
    return (q >= k) & ((q - k).to(tla.Int64) <= aux[0][q + info[2]])


@simt
@tla.jit
def head_labels(b, h, q, k, info, aux, scalars):
    return aux[0][b, h, q] == aux[1][b, h, k]


@simt
@tla.jit
def global_key_filter(b, h, q, k, info, aux, scalars):
    return k >= aux[0][k + info[3]]


@pytest.mark.npu
@pytest.mark.parametrize("kind,dtype", [("labels", torch.int64), ("window", torch.int32)])
def test_tnd_global_aux_cache_with_different_boundaries(device, monkeypatch, kind, dtype):
    case = make_case(
        q_lengths=(0, 65, 129, 257),
        k_lengths=(31, 129, 65, 226),
        q_heads=4,
        kv_heads=2,
        q_packed=True,
        k_packed=True,
        seed=931,
    )
    positions = torch.arange(451, dtype=dtype)
    if kind == "labels":
        aux = (positions // 13, positions // 13)
        if dtype == torch.int64:
            aux = tuple(t + 2**40 for t in aux)
        native = global_labels

        def mask(b, h, q, k, info, aux, scalars):
            return aux[0][q + info[2]] == aux[1][k + info[3]]
    else:
        aux = (positions % 7 + 1,)
        native = global_window_64 if dtype == torch.int64 else global_window

        def mask(b, h, q, k, info, aux, scalars):
            return (q >= k) & (q - k <= aux[0][q + info[2]])

    sparse = _blocks(case, mask, aux)
    expected, logs = reference(case, mask_mod=mask, sparse=sparse, aux=aux)
    inputs, kwargs = to_device(case, device)
    original = runtime._compile_mxfp8
    attempts = []

    def compile_cached(args, *rest, **options):
        attempts.append(args[-3])
        return original(args, *rest, **options)

    monkeypatch.setattr(runtime, "_compile_mxfp8", compile_cached)
    runtime.clear_compile_cache()
    try:
        for _ in range(2):
            actual, lse = flash_attn_varlen_func(
                *inputs,
                **kwargs,
                mask_mod=native,
                aux_tensors=tuple(t.to(device) for t in aux),
                block_sparse_tensors=_device_sparse(sparse, device),
                return_lse=True,
            )
            check(actual, expected)
            check_quantized_lse(lse.cpu(), logs)
        if kind == "window" and dtype == torch.int64:
            assert len(attempts) == 2 and attempts[0] and not attempts[1]
        else:
            assert len(attempts) == 1 and attempts[0], (
                "must compile the cache path without fallback"
            )
    finally:
        runtime.clear_compile_cache()


@pytest.mark.npu
def test_tnd_short_documents_cross_block_boundaries(device):
    """Few visible keys must not use an uncertified fixed-exponent result."""
    case = make_case(
        q_lengths=(513, 65, 129),
        k_lengths=(513, 65, 129),
        q_heads=4,
        kv_heads=4,
        q_packed=True,
        k_packed=True,
        seed=7,
    )
    positions = torch.arange(707, dtype=torch.int32)
    aux = (positions // 5, positions // 5)

    def mask(b, h, q, k, info, aux, scalars):
        return aux[0][q + info[2]] == aux[1][k + info[3]]

    sparse = _blocks(case, mask, aux)
    expected, logs = reference(case, mask_mod=mask, sparse=sparse, aux=aux)
    inputs, kwargs = to_device(case, device)
    actual, lse = flash_attn_varlen_func(
        *inputs,
        **kwargs,
        mask_mod=global_labels,
        aux_tensors=tuple(t.to(device) for t in aux),
        block_sparse_tensors=_device_sparse(sparse, device),
        return_lse=True,
    )
    check(actual, expected)
    check_quantized_lse(lse.cpu(), logs)


@pytest.mark.npu
@pytest.mark.parametrize(
    "q_packed,k_packed,scale,output_cache_policy",
    [
        (False, True, None, False),
        (True, False, 0.0, True),
        (True, True, -0.125, False),
        (True, True, None, True),
    ],
    indirect=["output_cache_policy"],
)
def test_global_labels_mixed_layouts_full_and_omitted_blocks(
    device, q_packed, k_packed, scale, output_cache_policy
):
    case = make_case(
        q_lengths=(129, 65) if q_packed else (129, 129),
        k_lengths=(193, 65) if k_packed else (193, 193),
        q_heads=4,
        kv_heads=2,
        q_packed=q_packed,
        k_packed=k_packed,
        seed=1147,
    )
    # Both arrays cover the cache's shared Q/K capacity. Used labels retain
    # the same values when the two packed token counts differ.
    positions = (
        torch.arange(max(sum(case["q_lengths"]), sum(case["k_lengths"])), dtype=torch.int32) // 7
    )
    aux = (positions, positions.clone())

    def mask(b, h, q, k, info, aux, scalars):
        return aux[0][q + (info[2] if q_packed else 0)] == aux[1][k + (info[3] if k_packed else 0)]

    sparse = _blocks(case, mask, aux)
    # Explicit full/omitted entries remain authoritative over the label mask.
    sparse.mask_block_cnt[0, 0, 0] = 0
    sparse.full_block_cnt[0, 0, 0] = 1
    sparse.full_block_idx[0, 0, 0, 0] = 0
    sparse.mask_block_cnt[1, 0, 0] = 0
    sparse.full_block_cnt[1, 0, 0] = 0
    expected, logs = reference(case, mask_mod=mask, sparse=sparse, aux=aux, scale=scale)
    inputs, kwargs = to_device(case, device)
    actual, lse = flash_attn_varlen_func(
        *inputs,
        **kwargs,
        mask_mod=global_labels,
        aux_tensors=tuple(t.to(device) for t in aux),
        block_sparse_tensors=_device_sparse(sparse, device),
        softmax_scale=scale,
        return_lse=True,
    )
    check(actual, expected)
    check_quantized_lse(lse.cpu(), logs)


@pytest.mark.npu
def test_global_labels_valid_and_masked_nan_rows(device):
    case = make_case(
        q_lengths=(129,),
        k_lengths=(129,),
        q_heads=4,
        kv_heads=2,
        q_packed=True,
        k_packed=True,
        seed=1148,
    )
    case["qkv"][0].view(torch.uint8)[[3, 9], 0, :] = 127
    ids = torch.arange(129, dtype=torch.int32) // 5
    query_ids = ids.clone()
    query_ids[9] = 100000
    aux = (query_ids, ids)

    def mask(b, h, q, k, info, aux, scalars):
        return aux[0][q + info[2]] == aux[1][k + info[3]]

    sparse = _blocks(case, mask, aux)
    inputs, kwargs = to_device(case, device)
    actual, lse = flash_attn_varlen_func(
        *inputs,
        **kwargs,
        mask_mod=global_labels,
        aux_tensors=tuple(t.to(device) for t in aux),
        block_sparse_tensors=_device_sparse(sparse, device),
        return_lse=True,
    )
    actual, lse = actual.cpu(), lse.cpu()
    assert torch.isnan(actual[3, 0]).all() and torch.isnan(lse[0, 3])
    assert not actual[9].count_nonzero() and torch.isneginf(lse[:, 9]).all()
    finite = torch.ones(actual.shape[:-1], dtype=torch.bool)
    finite[3, 0] = False
    assert torch.isfinite(actual[finite]).all()


@pytest.mark.npu
@pytest.mark.parametrize(
    "compiled,output_cache_policy", [(False, True), (True, False)], indirect=["output_cache_policy"]
)
def test_global_labels_rebind_boundaries_scales_and_stream(device, compiled, output_cache_policy):
    from .mxfp8_reference import SCALE_NAMES

    def call(q, k, v, qs, ks, vs, cq, ck, aq, ak, blocks):
        return flash_attn_varlen_func(
            q,
            k,
            v,
            q_descale=qs,
            k_descale=ks,
            v_descale=vs,
            cu_seqlens_q=cq,
            cu_seqlens_k=ck,
            max_seqlen_q=65,
            max_seqlen_k=65,
            mask_mod=global_labels,
            aux_tensors=(aq, ak),
            block_sparse_tensors=blocks,
            return_lse=True,
        )

    invoke = (
        torch.compile(call, backend="inductor", fullgraph=True, dynamic=True) if compiled else call
    )
    producer, consumer = torch.npu.Stream(device=device), torch.npu.Stream(device=device)
    keys = None
    for step, lengths in enumerate(((65, 63), (64, 64))):
        case = make_case(
            q_lengths=lengths,
            k_lengths=lengths,
            q_packed=True,
            k_packed=True,
            q_heads=4,
            kv_heads=2,
            v_scale_capacity=8,
            seed=1150 + step,
        )
        ids = torch.arange(128, dtype=torch.int32) // (5 + step)
        aux = (ids, ids.clone())

        def mask(b, h, q, k, info, aux, scalars):
            return aux[0][q + info[2]] == aux[1][k + info[3]]

        sparse = _blocks(case, mask, aux)
        expected, logs = reference(case, mask_mod=mask, sparse=sparse, aux=aux)
        with torch.npu.stream(producer):
            inputs, kwargs = to_device(case, device)
            aq, ak = (t.to(device) for t in aux)
            blocks = _device_sparse(sparse, device)
        with torch.npu.stream(consumer):
            consumer.wait_stream(producer)
            actual, lse = invoke(
                *inputs,
                *(kwargs[name] for name in SCALE_NAMES),
                kwargs["cu_seqlens_q"],
                kwargs["cu_seqlens_k"],
                aq,
                ak,
                blocks,
            )
        torch.npu.current_stream(device).wait_stream(consumer)
        check(actual, expected)
        check_quantized_lse(lse.cpu(), logs)
        if keys is not None:
            assert set(runtime.compile_cache) == keys
        keys = set(runtime.compile_cache)


@pytest.mark.npu
@pytest.mark.parametrize("scale", [None, -0.125])
def test_packed_head_specific_labels_and_sparse_lists(device, scale):
    from flash_attn_npu_4.catlass_dsl.block_sparsity import BlockSparseTensorsTorch

    lengths = (0, 257, 65)
    case = make_case(
        q_lengths=lengths,
        k_lengths=lengths,
        q_packed=True,
        k_packed=True,
        q_heads=4,
        kv_heads=4,
        seed=1157,
    )
    ids = torch.stack([torch.arange(257, dtype=torch.int32) // (5 + h) for h in range(4)])
    ids = ids[None].expand(3, -1, -1).contiguous()
    keys = ids.clone()
    keys[1, 1, 120:137] = -1
    aux = (ids, keys)

    def mask(b, h, q, k, info, aux, scalars):
        return aux[0][b, h, q] == aux[1][b, h, k]

    parts = [
        _blocks(case, lambda b, h, q, k, i, a, s: mask(b, head, q, k, i, a, s), aux)
        for head in range(4)
    ]
    fields = ("mask_block_cnt", "mask_block_idx", "full_block_cnt", "full_block_idx")
    sparse = BlockSparseTensorsTorch(
        *(torch.cat([getattr(p, f) for p in parts], dim=1) for f in fields), block_size=(128, 128)
    )
    sparse.mask_block_cnt[1, 2, 1] = 0
    sparse.full_block_cnt[1, 2, 1] = 0
    expected, logs = reference(case, mask_mod=mask, sparse=sparse, aux=aux, scale=scale)
    inputs, kwargs = to_device(case, device)
    actual, lse = flash_attn_varlen_func(
        *inputs,
        **kwargs,
        mask_mod=head_labels,
        aux_tensors=tuple(t.to(device) for t in aux),
        block_sparse_tensors=_device_sparse(sparse, device),
        softmax_scale=scale,
        return_lse=True,
    )
    check(actual, expected)
    check_quantized_lse(lse.cpu(), logs)


@pytest.mark.npu
def test_query_window_extreme_radii_and_cached_updates(device):
    case = make_case(
        q_lengths=(65, 257),
        k_lengths=(129, 65),
        q_packed=True,
        k_packed=True,
        q_heads=4,
        kv_heads=2,
        seed=1163,
    )
    full = _blocks(case, lambda b, h, q, k, i, a, s: torch.ones_like(q + k, dtype=torch.bool), ())
    # A fixed superset of partial blocks allows radii to change without
    # updating metadata; negative radii and empty visible rows remain legal.
    sparse = full._replace(
        mask_block_cnt=full.full_block_cnt,
        mask_block_idx=full.full_block_idx,
        full_block_cnt=torch.zeros_like(full.full_block_cnt),
        full_block_idx=torch.full_like(full.full_block_idx, -111),
    )
    inputs, kwargs = to_device(case, device)
    blocks = _device_sparse(sparse, device)
    radius_device = torch.empty(322, dtype=torch.int32, device=device)
    cache_keys = None
    for values in ((-(2**31), -1, 0, 1, 7, 31, 64, 128, 2**31 - 1), (0, 1, 7)):
        radius = torch.tensor(values, dtype=torch.int32).repeat(322)[:322]
        radius_device.copy_(radius)

        def mask(b, h, q, k, info, aux, scalars):
            return (q >= k) & (q - k <= aux[0][q + info[2]])

        expected, logs = reference(case, mask_mod=mask, sparse=sparse, aux=(radius,))
        actual, lse = flash_attn_varlen_func(
            *inputs,
            **kwargs,
            mask_mod=global_window,
            aux_tensors=(radius_device,),
            block_sparse_tensors=blocks,
            return_lse=True,
        )
        check(actual, expected)
        check_quantized_lse(lse.cpu(), logs)
        if cache_keys is not None:
            assert set(runtime.compile_cache) == cache_keys
        cache_keys = set(runtime.compile_cache)


@pytest.mark.npu
@pytest.mark.parametrize(
    "q_packed,k_packed", [(False, False), (False, True), (True, False), (True, True)]
)
def test_query_independent_mask_head_major_schedule(device, q_packed, k_packed):
    case = make_case(
        q_lengths=(1024, 129) if q_packed else (1024, 1024),
        k_lengths=(1024, 65) if k_packed else (1024, 1024),
        q_packed=q_packed,
        k_packed=k_packed,
        q_heads=4,
        kv_heads=2,
        seed=1168,
    )
    data = torch.arange(max(sum(case["q_lengths"]), sum(case["k_lengths"])), dtype=torch.int32) % 13

    def mask(b, h, q, k, info, aux, scalars):
        return k >= aux[0][k + (info[3] if k_packed else 0)]

    sparse = _blocks(case, mask, (data,))
    # Change one query's authoritative selection despite the query-independent
    # callback, so traversal cannot incorrectly reuse sparse lists across rows.
    sparse.full_block_cnt[0, 0, 1] = max(0, int(sparse.full_block_cnt[0, 0, 1]) - 1)
    rows = [
        sorted({min(n - 1, q + delta) for q in range(0, n, 128) for delta in (0, 63, 127)})
        for n in case["q_lengths"]
    ]
    expected, logs = reference(case, mask_mod=mask, sparse=sparse, aux=(data,), query_rows=rows)
    inputs, kwargs = to_device(case, device)
    actual, lse = flash_attn_varlen_func(
        *inputs,
        **kwargs,
        mask_mod=global_key_filter,
        aux_tensors=(data.to(device),),
        block_sparse_tensors=_device_sparse(sparse, device),
        return_lse=True,
    )
    actual, lse = actual.cpu(), lse.cpu()
    offset = 0
    for b, indices in enumerate(rows):
        if q_packed:
            selected = torch.tensor(indices) + offset
            check(actual[selected], expected[selected])
            check_quantized_lse(lse[:, selected], logs[:, selected])
        else:
            check(actual[b, indices], expected[b, indices])
            check_quantized_lse(lse[b, :, indices], logs[b, :, indices])
        offset += case["q_lengths"][b]


@pytest.mark.npu
@pytest.mark.parametrize(
    "q_packed,k_packed", [(False, False), (False, True), (True, False), (True, True)]
)
def test_query_independent_mask_prepacked_layouts(device, monkeypatch, q_packed, k_packed):
    monkeypatch.setattr(runtime, "_mxfp8_should_pack_kv", lambda *args: True)
    original = runtime._make_mxfp8_plan
    packed = []

    def capture(*args, **kwargs):
        plan = original(*args, **kwargs)
        packed.append(plan.pack_kv)
        return plan

    monkeypatch.setattr(runtime, "_make_mxfp8_plan", capture)
    runtime.clear_compile_cache()
    try:
        test_query_independent_mask_head_major_schedule(device, q_packed, k_packed)
        assert packed and all(packed)
    finally:
        runtime.clear_compile_cache()


@pytest.mark.parametrize("kind", ["labels", "window", "keys", "head_labels", "shifted", "branch"])
@pytest.mark.parametrize("packed", [False, True])
def test_auxiliary_cache_roles_are_conservative(kind, packed):
    from types import SimpleNamespace

    from flash_attn_npu_4.catlass_dsl.kernels.flash_fwd_mxfp8_arch35 import _mx_aux_cache_roles

    def labels(s, b, h, q, k, a, c, i, checked):
        return a[0][q + i[2]] == a[1][k + i[3]]

    def window(s, b, h, q, k, a, c, i, checked):
        return (q >= k) & (q - k <= a[0][q + i[2]])

    def keys(s, b, h, q, k, a, c, i, checked):
        return k >= a[0][k + i[3]]

    def head_labels(s, b, h, q, k, a, c, i, checked):
        return a[0][b, h, q] == a[1][b, h, k]

    def shifted(s, b, h, q, k, a, c, i, checked):
        return q >= a[0][q + i[2] + 1]

    def branch(s, b, h, q, k, a, c, i, checked):
        if b == 0:
            return a[0][q + i[2]] == a[1][k + i[3]]
        return a[0][k + i[3]] == a[1][q + i[2]]

    masks = {
        "labels": labels,
        "window": window,
        "keys": keys,
        "head_labels": head_labels,
        "shifted": shifted,
        "branch": branch,
    }
    expected = {
        "labels": ((True, False), (False, True)),
        "window": ((True, False), (False, False)),
        "keys": ((False, True), (False, False)),
        "head_labels": ((True, False), (False, True)),
        "shifted": ((True, True), (True, True)),
        "branch": ((True, True), (True, True)),
    }
    aux = (SimpleNamespace(dtype=tla.Int32),) * 2
    shape, strides = ((2, 4, 128), (512, 128, 1)) if kind == "head_labels" else ((1024,), (1,))
    metadata = ((-1, shape, strides, False),) * 2
    assert _mx_aux_cache_roles(masks[kind], aux, (), metadata, packed) == expected[kind]


@pytest.mark.npu
@pytest.mark.parametrize(
    "q_packed,k_packed,scale", [(False, True, None), (True, False, 0.0), (True, True, -0.125)]
)
def test_query_only_aux_cache_full_omitted_and_partial_blocks(device, q_packed, k_packed, scale):
    case = make_case(
        q_lengths=(129, 65) if q_packed else (129, 129),
        k_lengths=(193, 65) if k_packed else (193, 193),
        q_heads=4,
        kv_heads=4,
        q_packed=q_packed,
        k_packed=k_packed,
        seed=1167,
    )
    radii = torch.arange(sum(case["q_lengths"]), dtype=torch.int32) % 7 + 1
    aux = (radii,)

    def mask(b, h, q, k, info, aux, scalars):
        return (q >= k) & (q - k <= aux[0][q + (info[2] if q_packed else 0)])

    sparse = _blocks(case, mask, aux)
    sparse.mask_block_cnt[0, 0, 0] = 0
    sparse.full_block_cnt[0, 0, 0] = 1
    sparse.full_block_idx[0, 0, 0, 0] = 0
    sparse.mask_block_cnt[1, 0, 0] = 0
    sparse.full_block_cnt[1, 0, 0] = 0
    expected, logs = reference(case, mask_mod=mask, sparse=sparse, aux=aux, scale=scale)
    inputs, kwargs = to_device(case, device)
    actual, lse = flash_attn_varlen_func(
        *inputs,
        **kwargs,
        mask_mod=global_window,
        aux_tensors=(radii.to(device),),
        block_sparse_tensors=_device_sparse(sparse, device),
        softmax_scale=scale,
        return_lse=True,
    )
    check(actual, expected)
    check_quantized_lse(lse.cpu(), logs)


@pytest.mark.npu
@pytest.mark.parametrize("kind,scale", [("labels", None), ("window", -0.125)])
def test_deep_online_pipeline_rebinds_ragged_boundaries(device, monkeypatch, kind, scale):
    # Two cores give each producer more than twelve query tasks, exercising
    # multiple live query/alpha/scale slots without a benchmark-sized golden.
    monkeypatch.setattr(runtime, "_cores", lambda device: 2)
    original_plan = runtime._make_mxfp8_plan

    def capture_plan(*args, **options):
        plan = original_plan(*args, **options)
        assert plan.block_num == 2 and plan.guard_elements == 0, (
            "must exercise the exact online pipeline"
        )
        return plan

    monkeypatch.setattr(runtime, "_make_mxfp8_plan", capture_plan)
    case = make_case(
        q_lengths=(0, 513, 65, 129),
        k_lengths=(31, 641, 0, 193),
        q_heads=4,
        kv_heads=4,
        q_packed=True,
        k_packed=True,
        v_scale_capacity=20,
        seed=1179,
    )
    if kind == "labels":
        aux = (torch.arange(865, dtype=torch.int32) // 5, torch.arange(865, dtype=torch.int32) // 5)
        native = global_labels

        def mask(b, h, q, k, info, aux, scalars):
            return aux[0][q + info[2]] == aux[1][k + info[3]]
    else:
        aux = (torch.arange(865, dtype=torch.int32) % 65,)
        native = global_window

        def mask(b, h, q, k, info, aux, scalars):
            return (q >= k) & (q - k <= aux[0][q + info[2]])

    inputs, kwargs = to_device(case, device)
    runtime.clear_compile_cache()
    try:
        cache_keys = None
        for qlengths, klengths in [
            ((0, 513, 65, 129), (31, 641, 0, 193)),
            ((65, 0, 513, 129), (0, 31, 641, 193)),
        ]:
            case["q_lengths"], case["k_lengths"] = qlengths, klengths
            for name, lengths in [("cu_seqlens_q", qlengths), ("cu_seqlens_k", klengths)]:
                offsets = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0).to(torch.int32)
                case["kwargs"][name] = offsets
                kwargs[name].copy_(offsets.to(device))
            sparse = _blocks(case, mask, aux)
            # A full tile must skip the callback even after partial tiles.
            sparse.mask_block_cnt[3, 0, 0] = 0
            sparse.full_block_cnt[3, 0, 0] = 1
            sparse.full_block_idx[3, 0, 0, 0] = 0
            expected, logs = reference(case, mask_mod=mask, sparse=sparse, aux=aux, scale=scale)
            actual, lse = flash_attn_varlen_func(
                *inputs,
                **kwargs,
                mask_mod=native,
                aux_tensors=tuple(t.to(device) for t in aux),
                softmax_scale=scale,
                block_sparse_tensors=_device_sparse(sparse, device),
                return_lse=True,
            )
            check(actual, expected)
            check_quantized_lse(lse.cpu(), logs)
            if cache_keys is not None:
                assert set(runtime.compile_cache) == cache_keys
            cache_keys = set(runtime.compile_cache)
    finally:
        runtime.clear_compile_cache()


@pytest.mark.npu
@pytest.mark.parametrize(
    "kind,global_index,layout,dtype",
    [
        ("labels", False, "BSND", torch.int32),
        ("labels", True, "BSND", torch.int32),
        ("labels", True, "BNSD", torch.int64),
        ("window", False, "BSND", torch.int32),
        ("window", True, "BNSD", torch.int32),
    ],
)
def test_fixed_aux_indices_rebind_without_fallback(
    device, monkeypatch, kind, global_index, layout, dtype
):
    @simt
    @tla.jit
    def native(b, h, q, k, info, aux, scalars):
        qi = q + info[2] if tla.const_expr(global_index) else q
        ki = k + info[3] if tla.const_expr(global_index) else k
        if tla.const_expr(kind == "labels"):
            return aux[0][qi] == aux[1][ki]
        elif tla.const_expr(kind == "window"):
            return (q >= k) & (q - k <= aux[0][qi])
        else:
            return k >= aux[0][ki]

    def mask(b, h, q, k, info, aux, scalars):
        qi, ki = (q + info[2], k + info[3]) if global_index else (q, k)
        if kind == "labels":
            return aux[0][qi] == aux[1][ki]
        if kind == "window":
            return (q >= k) & (q - k <= aux[0][qi])
        return k >= aux[0][ki]

    case = make_case(
        q_lengths=(1024, 1024),
        k_lengths=(1153, 1153),
        q_heads=4,
        kv_heads=2,
        layout=layout,
        seed=1573,
    )
    inputs, kwargs = to_device(case, device)
    positions = torch.arange(1153, dtype=dtype)
    aux_device = None
    compiled = []
    original = runtime._compile_mxfp8

    def native_only(arguments, *args, **options):
        compiled.append(arguments[-7])
        assert arguments[-3], "supported direct token indices must retain their auxiliary caches"
        try:
            return original(arguments, *args, **options)
        except Exception as exc:
            pytest.fail(f"Cached auxiliary compilation failed: {exc}")

    monkeypatch.setattr(runtime, "_compile_mxfp8", native_only)
    runtime.clear_compile_cache()
    try:
        for update in range(2):
            if kind == "labels":
                labels = positions // (5 + update) + (2**40 if dtype == torch.int64 else 0)
                aux = (labels, labels.clone())
            else:
                aux = (positions % (7 + update) + 1,)
            if aux_device is None:
                aux_device = tuple(t.to(device) for t in aux)
            else:
                for dst, src in zip(aux_device, aux):
                    dst.copy_(src)
            # The TND block builder accumulates offsets across batches. Fixed
            # tensors expose zero offsets to both the callback and its golden.
            sparse = _blocks(
                case,
                lambda b, h, q, k, info, a, s: mask(b, h, q, k, (info[0], info[1], 0, 0), a, s),
                aux,
            )
            sparse.mask_block_cnt[0, 0, 0] = 0
            sparse.full_block_cnt[0, 0, 0] = 1
            sparse.full_block_idx[0, 0, 0, 0] = 0
            sparse.mask_block_cnt[1, 0, 0] = 0
            sparse.full_block_cnt[1, 0, 0] = 0
            expected, logs = reference(case, mask_mod=mask, sparse=sparse, aux=aux, scale=-0.125)
            actual, lse = interface._flash(
                *inputs,
                layout=layout,
                softmax_scale=-0.125,
                causal=False,
                window_size=(None, None),
                softcap=0.0,
                score_mod=None,
                score_mod_bwd=None,
                mask_mod=native,
                aux_tensors=aux_device,
                aux_scalars=None,
                block_sparse_tensors=_device_sparse(sparse, device),
                block_sparse_tensors_bwd=None,
                return_lse=True,
                **{name: kwargs[name] for name in SCALE_NAMES},
            )
            check(actual, expected)
            check_quantized_lse(lse.cpu(), logs)
        assert len(compiled) == 1, "updated auxiliary contents must rebind the prepared plan"
    finally:
        runtime.clear_compile_cache()


# Head reuse, authoritative sparse lists and DMA lifetimes.


@simt
@tla.jit
def neighborhood(b, h, q, k, info, aux, scalars):
    return (k >= q - 128) & (k < q + 128)


@pytest.mark.npu
@pytest.mark.parametrize(
    "layouts,repair",
    [((True, True), False), ((True, False), True), ((False, True), False), ((False, False), True)],
)
def test_head_pair_cache_sparse_tails_and_repair(device, monkeypatch, repair, layouts):
    qp, kp = layouts
    qlens = (385, 129) if qp else (385, 385)
    klens = (513, 65) if kp else (513, 513)
    if not qp and not kp:
        qlens, klens = (1025, 1025), (1153, 1153)
    case = make_case(
        q_lengths=qlens, k_lengths=klens, q_heads=4, kv_heads=4, q_packed=qp, k_packed=kp, seed=936
    )
    if repair:
        case["qkv"][0].view(torch.uint8).fill_(0x38)
        case["qkv"][1].view(torch.uint8).fill_(0xB8)
        case["kwargs"]["q_descale"].fill_(127)
        case["kwargs"]["k_descale"].fill_(127)

    def mask(b, h, q, k, info, aux, scalars):
        return (k >= q - 128) & (k < q + 128)

    sparse = _blocks(case, mask, ())
    sparse = sparse._replace(
        **{
            name: t.repeat_interleave(4, dim=1).clone()
            for name, t in sparse._asdict().items()
            if isinstance(t, torch.Tensor)
        }
    )
    sparse.mask_block_cnt[0, 1, 1] = 0
    sparse.full_block_cnt[0, 1, 1] = 0
    for counts, indices in (
        (sparse.mask_block_cnt, sparse.mask_block_idx),
        (sparse.full_block_cnt, sparse.full_block_idx),
    ):
        for batch in range(2):
            for block in range(counts.shape[-1]):
                n = int(counts[batch, 2, block])
                indices[batch, 2, block, :n] = indices[batch, 2, block, :n].flip(0)
    expected, logs = reference(case, mask_mod=mask, sparse=sparse)
    inputs, kwargs = to_device(case, device)
    original = runtime._compile_mxfp8
    used = []

    def capture(values, *rest, **options):
        if values[-7] == "continuous":
            used.append(values[-4].paired_heads)
        return original(values, *rest, **options)

    monkeypatch.setattr(runtime, "_compile_mxfp8", capture)
    runtime.clear_compile_cache()
    try:
        for _ in range(2):
            output, lse = flash_attn_varlen_func(
                *inputs,
                **kwargs,
                mask_mod=neighborhood,
                block_sparse_tensors=_device_sparse(sparse, device),
                return_lse=True,
            )
            check(output, expected)
            check_quantized_lse(lse.cpu(), logs)
        assert used and all(used)
    finally:
        runtime.clear_compile_cache()


@pytest.mark.npu
def test_head_pair_cache_skips_empty_sequences_and_rebinds_boundaries(device):
    case = make_case(
        q_lengths=(0, 129, 65, 0),
        k_lengths=(65, 0, 193, 0),
        q_heads=4,
        kv_heads=4,
        q_packed=True,
        k_packed=True,
        seed=937,
    )
    inputs, kwargs = to_device(case, device)

    def mask(b, h, q, k, info, aux, scalars):
        return (k >= q - 128) & (k < q + 128)

    for qlens, klens in [((0, 129, 65, 0), (65, 0, 193, 0)), ((65, 0, 129, 0), (0, 193, 0, 65))]:
        case["q_lengths"], case["k_lengths"] = qlens, klens
        for name, lengths in [("cu_seqlens_q", qlens), ("cu_seqlens_k", klens)]:
            offsets = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0).to(torch.int32)
            case["kwargs"][name] = offsets
            kwargs[name].copy_(offsets.to(device))
        sparse = _blocks(case, mask, ())
        expected, logs = reference(case, mask_mod=mask, sparse=sparse)
        output, lse = flash_attn_varlen_func(
            *inputs,
            **kwargs,
            mask_mod=neighborhood,
            block_sparse_tensors=_device_sparse(sparse, device),
            return_lse=True,
        )
        check(output, expected)
        check_quantized_lse(lse.cpu(), logs)


@simt
@tla.jit
def full_diagonal_tiles(b, h, q, k, info, aux, scalars):
    return q // 128 == k // 128


@pytest.mark.npu
@pytest.mark.parametrize(
    "mode", ["normal", "zero_scale", "negative_scale", "underflow", "saturation"]
)
def test_single_full_kv_tile_quantization_and_repair(device, monkeypatch, mode):
    case = make_case(
        q_lengths=(1153, 129, 0),
        k_lengths=(1280, 256, 128),
        q_heads=8,
        kv_heads=8,
        q_packed=True,
        k_packed=True,
        seed=1168,
    )
    scale = 0.0 if mode == "zero_scale" else -0.125 if mode == "negative_scale" else None
    if mode in ("underflow", "saturation"):
        case["qkv"][0].view(torch.uint8).fill_(0x38)
        case["qkv"][1].view(torch.uint8).fill_(0xB8 if mode == "underflow" else 0x38)
        case["kwargs"]["q_descale"].fill_(127)
        case["kwargs"]["k_descale"].fill_(127)

    def mask(b, h, q, k, info, aux, scalars):
        return q // 128 == k // 128

    sparse = _blocks(case, mask, ())
    sparse.full_block_cnt[0, 0, 1] = 0
    expected, logs = reference(case, mask_mod=mask, sparse=sparse, scale=scale)
    inputs, kwargs = to_device(case, device)
    original = runtime._launch
    work = []

    def capture(executor, payload, **options):
        work.append(payload[6])
        return original(executor, payload, **options)

    monkeypatch.setattr(runtime, "_launch", capture)
    actual, lse = flash_attn_varlen_func(
        *inputs,
        **kwargs,
        softmax_scale=scale,
        mask_mod=full_diagonal_tiles,
        block_sparse_tensors=_device_sparse(sparse, device),
        return_lse=True,
    )
    check(actual, expected)
    check_quantized_lse(lse.cpu(), logs)
    capacity = 3 * 8 * ((max(case["q_lengths"]) + 127) // 128)
    aggregates = work[-1].cpu()[capacity * 16 :].reshape(-1, 16)[:, [0, 8]]
    assert bool((aggregates > 0).any()) == (mode in ("underflow", "saturation"))


@simt
@tla.jit
def per_head_window(b, h, q, k, info, aux, scalars):
    return (q >= k) & (q - k <= aux[0][b, h, q])


@pytest.mark.npu
@pytest.mark.parametrize(
    "q_packed,k_packed,heads,repeats",
    [(True, True, 4, 1), (True, False, 8, 1), (False, True, 16, 3), (True, True, 24, 20)],
)
def test_head_groups_keep_distinct_sparse_lists(
    device, monkeypatch, q_packed, k_packed, heads, repeats
):
    monkeypatch.setattr(runtime, "_cores", lambda device: 2)
    case = make_case(
        q_lengths=(257, 65) if q_packed else (257, 257),
        k_lengths=(193, 31) if k_packed else (193, 193),
        q_heads=heads,
        kv_heads=heads,
        q_packed=q_packed,
        k_packed=k_packed,
        v_scale_capacity=10 if k_packed else None,
        seed=1191,
    )
    radii = (
        torch.arange(257, dtype=torch.int32)[None, None, :]
        + torch.arange(heads, dtype=torch.int32)[None, :, None]
    ) % 17
    radii = radii.expand(2, -1, -1).contiguous()

    def mask(b, h, q, k, info, aux, scalars):
        return (q >= k) & (q - k <= aux[0][b, h, q])

    counts = torch.full((2, heads, 3), 2, dtype=torch.int32)
    indices = torch.tensor([1, 0], dtype=torch.int32).expand(2, heads, 3, 2).contiguous()
    full = torch.zeros_like(counts)
    full_indices = torch.full_like(indices, -111)
    if k_packed:
        counts[1] = 1
        indices[1, :, :, 0] = 0
    # Every group includes an omitted head and an authoritative full-block head.
    counts[:, 0::4] = 0
    counts[:, 1::4] = 0
    full[:, 1::4] = 1
    full_indices[:, 1::4, :, 0] = 0
    sparse = BlockSparseTensorsTorch(counts, indices, full, full_indices, block_size=(128, 128))
    expected, logs = reference(case, mask_mod=mask, sparse=sparse, aux=(radii,), scale=-0.125)
    original = runtime._make_mxfp8_plan

    def capture(*args, **options):
        plan = original(*args, **options)
        assert plan.guard_elements == 0 and plan.block_num == 2
        return plan

    monkeypatch.setattr(runtime, "_make_mxfp8_plan", capture)
    runtime.clear_compile_cache()
    try:
        data, kwargs = to_device(case, device)
        aux = (radii.to(device),)
        blocks = _device_sparse(sparse, device)
        # Repeated full-to-tail slot reuse exposes Fill/DMA overlap that a
        # single launch with fewer heads may miss.
        for _ in range(repeats):
            actual, lse = flash_attn_varlen_func(
                *data,
                **kwargs,
                mask_mod=per_head_window,
                aux_tensors=aux,
                softmax_scale=-0.125,
                block_sparse_tensors=blocks,
                return_lse=True,
            )
            check_quantized_lse(lse.cpu(), logs)
            check(actual, expected)
    finally:
        runtime.clear_compile_cache()


@pytest.mark.npu
@pytest.mark.parametrize(
    "layouts,mode",
    [
        ((True, True), "negative"),
        ((True, False), "zero"),
        ((False, True), "underflow"),
        ((False, False), "negative"),
    ],
)
def test_short_queue_with_variable_kv_counts(device, monkeypatch, layouts, mode):
    qp, kp = layouts
    q_lengths = (641, 129, 0) if qp else (641,) * 3
    k_lengths = (641, 193, 65) if kp else (641,) * 3
    if not qp and not kp:
        q_lengths, k_lengths = (1025,) * 3, (1153,) * 3
    case = make_case(
        q_lengths=q_lengths,
        k_lengths=k_lengths,
        q_heads=8,
        kv_heads=8,
        q_packed=qp,
        k_packed=kp,
        seed=1452,
    )
    scale = -0.125 if mode == "negative" else 0.0 if mode == "zero" else 0.125
    if mode == "underflow":
        case["qkv"][0].view(torch.uint8).fill_(0x38)
        case["qkv"][1].view(torch.uint8).fill_(0xB8)
        case["kwargs"]["q_descale"].fill_(127)
        case["kwargs"]["k_descale"].fill_(127)

    def mask(b, h, q, k, info, aux, scalars):
        return (k >= q - 128) & (k < q + 128)

    sparse = _blocks(case, mask, ())
    sparse = sparse._replace(
        **{
            name: tensor.repeat_interleave(8, dim=1).clone()
            for name, tensor in sparse._asdict().items()
            if isinstance(tensor, torch.Tensor)
        }
    )
    sparse.mask_block_cnt.zero_()
    sparse.full_block_cnt.zero_()
    for batch, length in enumerate(case["k_lengths"]):
        blocks = (length + 127) // 128
        for head in range(8):
            for query in range(sparse.mask_block_cnt.shape[-1]):
                count = min((query + head) % 6, blocks)
                if not count:
                    continue
                partial = query % blocks
                sparse.mask_block_cnt[batch, head, query] = 1
                sparse.mask_block_idx[batch, head, query, 0] = partial
                # Full entries remain authoritative even outside the callback
                # neighborhood; reverse order tests the online state lifetime.
                full = [i for i in reversed(range(blocks)) if i != partial][: count - 1]
                sparse.full_block_cnt[batch, head, query] = len(full)
                if full:
                    sparse.full_block_idx[batch, head, query, : len(full)] = torch.tensor(full)
    original = runtime._compile_mxfp8
    plans = []

    def capture(values, *args, **kwargs):
        plans.append((values[-7], values[-4].paired_heads))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(runtime, "_compile_mxfp8", capture)
    runtime.clear_compile_cache()
    try:
        verify(
            case,
            device,
            options={
                "mask_mod": neighborhood,
                "softmax_scale": scale,
                "block_sparse_tensors": _device_sparse(sparse, device),
            },
            reference_options={"mask_mod": mask, "scale": scale, "sparse": sparse},
        )
        assert plans and all(pipeline == "continuous" and paired for pipeline, paired in plans)
    finally:
        runtime.clear_compile_cache()


# Single-sequence capacities and graph rebinding.


@pytest.mark.npu
@pytest.mark.parametrize(
    "layouts,kv_heads,kind",
    [
        ((True, True), 1, "dense"),
        ((True, True), 4, "labels"),
        ((True, False), 4, "dense"),
        ((True, False), 1, "labels"),
        ((False, True), 1, "dense"),
        ((False, True), 4, "labels"),
    ],
)
def test_single_segment_aligned_tight_bounds(device, monkeypatch, layouts, kv_heads, kind):
    """Shape-proven full tiles retain mixed layouts and authoritative sparse lists."""
    from flash_attn_npu_4.catlass_dsl import runtime

    qp, kp = layouts
    case = make_case(
        q_lengths=(256,),
        k_lengths=(384,),
        q_heads=4,
        kv_heads=kv_heads,
        q_packed=qp,
        k_packed=kp,
        v_scale_capacity=9 if kp else None,
        seed=1197,
    )
    aux = (torch.arange(384, dtype=torch.int32) // 5,) * 2 if kind == "labels" else ()

    def mask(b, h, q, k, info, aux, scalars):
        return aux[0][q + info[2]] == aux[1][k + info[3]]

    sparse = _blocks(case, mask, aux) if kind == "labels" else None
    if sparse is not None:
        sparse = sparse._replace(
            **{
                n: t.repeat_interleave(4, dim=1).clone()
                for n, t in sparse._asdict().items()
                if isinstance(t, torch.Tensor)
            }
        )
        sparse.mask_block_cnt[0, 1, 0] = 0
        sparse.full_block_cnt[0, 1, 0] = 1
        sparse.full_block_idx[0, 1, 0, 0] = 2
        sparse.mask_block_cnt[0, 2, 1] = 0
        sparse.full_block_cnt[0, 2, 1] = 0
    expected, logs = reference(
        case, mask_mod=mask if aux else None, sparse=sparse, aux=aux, scale=-0.125
    )
    original = runtime._compile_mxfp8

    def native_only(arguments, *rest, **options):
        try:
            return original(arguments, *rest, **options)
        except Exception as exc:
            pytest.fail(f"Native compilation failed: {exc}")

    monkeypatch.setattr(runtime, "_compile_mxfp8", native_only)
    runtime.clear_compile_cache()
    try:
        inputs, kwargs = to_device(case, device)
        actual, lse = flash_attn_varlen_func(
            *inputs,
            **kwargs,
            mask_mod=global_labels if aux else None,
            aux_tensors=tuple(t.to(device) for t in aux),
            softmax_scale=-0.125,
            block_sparse_tensors=_device_sparse(sparse, device) if sparse is not None else None,
            return_lse=True,
        )
        check(actual, expected)
        check_quantized_lse(lse.cpu(), logs)
    finally:
        runtime.clear_compile_cache()


@pytest.mark.npu
@pytest.mark.parametrize(
    "layouts,kind,kv_heads",
    [
        ((True, True), "dense", 1),
        ((True, True), "window", 2),
        ((True, True), "labels", 4),
        ((True, False), "dense", 2),
        ((True, False), "window", 4),
        ((True, False), "labels", 1),
        ((False, True), "dense", 4),
        ((False, True), "window", 1),
        ((False, True), "labels", 2),
    ],
)
def test_single_segment_capacity_hints_and_mixed_layouts(device, layouts, kind, kv_heads):
    """Nine cases cover every pair of layout, mask and MQA/GQA/MHA head count."""
    qp, kp = layouts
    case = make_case(
        q_lengths=(129,),
        k_lengths=(193,),
        q_heads=4,
        kv_heads=kv_heads,
        q_packed=qp,
        k_packed=kp,
        v_scale_capacity=20 if kp else None,
        seed=1173,
    )
    inputs, kwargs = to_device(case, device)
    # Sparse row capacity can exceed the number of scheduled query tiles.
    kwargs["max_seqlen_q"] = 513 if qp else 129
    kwargs["max_seqlen_k"] = 1025 if kp else 193
    auxiliary = ()
    mask = native = None
    if kind == "labels":
        auxiliary = (torch.arange(1025, dtype=torch.int32) // 7,) * 2
        native = global_labels

        def mask(b, h, q, k, info, aux, scalars):
            return aux[0][q + info[2]] == aux[1][k + info[3]]
    elif kind == "window":
        native = shifted_window

        def mask(b, h, q, k, info, aux, scalars):
            center = q + info[1] - info[0]
            return (k <= center) & (k > center - 128)

    sparse = None if native is None else _blocks(case, mask, auxiliary)
    if sparse is not None:
        sparse = sparse._replace(
            **{
                name: value.repeat_interleave(4, dim=1).clone()
                for name, value in sparse._asdict().items()
                if isinstance(value, torch.Tensor)
            }
        )
        # Independent heads and explicit full/omitted entries remain authoritative.
        sparse.mask_block_cnt[0, 1, 0] = 0
        sparse.full_block_cnt[0, 1, 0] = 1
        sparse.full_block_idx[0, 1, 0, 0] = 1
        sparse.mask_block_cnt[0, 2, 1] = 0
        sparse.full_block_cnt[0, 2, 1] = 0
    expected, logs = reference(case, mask_mod=mask, sparse=sparse, aux=auxiliary, scale=-0.125)
    if sparse is not None and qp:
        padding = (kwargs["max_seqlen_q"] + 127) // 128 - sparse.mask_block_cnt.shape[-1]
        sparse = sparse._replace(
            **{
                name: torch.nn.functional.pad(
                    value,
                    (0, padding) if value.ndim == 3 else (0, 0, 0, padding),
                    value=0 if value.ndim == 3 else -111,
                )
                for name, value in sparse._asdict().items()
                if isinstance(value, torch.Tensor)
            }
        )
    actual, lse = flash_attn_varlen_func(
        *inputs,
        **kwargs,
        mask_mod=native,
        aux_tensors=tuple(t.to(device) for t in auxiliary),
        block_sparse_tensors=None if sparse is None else _device_sparse(sparse, device),
        softmax_scale=-0.125,
        return_lse=True,
    )
    check(actual, expected)
    check_quantized_lse(lse.cpu(), logs)


@pytest.mark.npu
@pytest.mark.parametrize("key_sign", [-1.0, 1.0])
def test_single_segment_repair_uses_actual_lengths(device, key_sign):
    case = make_case(
        q_lengths=(129,),
        k_lengths=(193,),
        q_heads=4,
        kv_heads=4,
        q_packed=True,
        k_packed=True,
        seed=1174,
    )
    for tensor, value in zip(case["qkv"][:2], (1.0, key_sign)):
        tensor.view(torch.uint8).fill_(
            int(torch.tensor(value).to(torch.float8_e4m3fn).view(torch.uint8))
        )
    case["kwargs"]["q_descale"].fill_(127)
    case["kwargs"]["k_descale"].fill_(127)
    expected, logs = reference(case)
    inputs, kwargs = to_device(case, device)
    kwargs.update(max_seqlen_q=513, max_seqlen_k=1025)
    actual, lse = flash_attn_varlen_func(*inputs, **kwargs, return_lse=True)
    check(actual, expected)
    check_quantized_lse(lse.cpu(), logs)


@pytest.mark.npu
@pytest.mark.parametrize("lengths", [(0, 193), (129, 0)])
def test_single_segment_empty_inputs_with_capacity_hints(device, lengths):
    qn, kn = lengths
    case = make_case(
        q_lengths=(qn,),
        k_lengths=(kn,),
        q_heads=4,
        kv_heads=1,
        q_packed=True,
        k_packed=True,
        v_scale_capacity=20,
        seed=1175,
    )
    inputs, kwargs = to_device(case, device)
    kwargs.update(max_seqlen_q=513, max_seqlen_k=1025)
    actual, lse = flash_attn_varlen_func(*inputs, **kwargs, return_lse=True)
    assert actual.shape == (qn, 4, 128)
    assert not actual.count_nonzero()
    assert torch.isneginf(lse).all()


@pytest.mark.npu
def test_single_segment_compiled_dynamic_lengths_and_scale_rebinding(device, monkeypatch):
    from flash_attn_npu_4.catlass_dsl import runtime

    from .mxfp8_reference import SCALE_NAMES

    def call(q, k, v, qs, ks, vs, cq, ck):
        return flash_attn_varlen_func(
            q,
            k,
            v,
            q_descale=qs,
            k_descale=ks,
            v_descale=vs,
            cu_seqlens_q=cq,
            cu_seqlens_k=ck,
            max_seqlen_q=513,
            max_seqlen_k=1025,
            return_lse=True,
        )

    from torch._dynamo.backends.registry import lookup_backend
    graphs = []

    def backend(gm, inputs):
        result = lookup_backend("inductor")(gm, inputs)
        graphs.append(gm)
        return result

    invoke = torch.compile(call, backend=backend, fullgraph=True, dynamic=True)
    original = runtime._compile_mxfp8
    launches = []

    def capture(arguments, *rest, **options):
        launches.append(arguments[-4])
        return original(arguments, *rest, **options)

    monkeypatch.setattr(runtime, "_compile_mxfp8", capture)
    runtime.clear_compile_cache()
    try:
        for step, qn in enumerate((129, 257, 257)):
            case = make_case(
                q_lengths=(qn,),
                k_lengths=(193,),
                q_heads=4,
                kv_heads=2,
                q_packed=True,
                k_packed=True,
                v_scale_capacity=20,
                seed=1176 + step,
            )
            expected, logs = reference(case)
            inputs, kwargs = to_device(case, device)
            actual, lse = invoke(
                *inputs,
                *(kwargs[n] for n in SCALE_NAMES),
                kwargs["cu_seqlens_q"],
                kwargs["cu_seqlens_k"],
            )
            check(actual, expected)
            check_quantized_lse(lse.cpu(), logs)
        assert len(graphs) == 1, "changing lengths must reuse the PyTorch graph"
        assert len(launches) == 1, "sequence lengths and scales must rebind without recompiling"
    finally:
        runtime.clear_compile_cache()


@pytest.mark.npu
@pytest.mark.parametrize(
    "mode,kv_heads", [("mixed", 2), ("empty", 4), ("full", 2), ("value_nan", 4), ("score_nan", 2)]
)
def test_direct_single_tile_preserves_empty_and_nonfinite_rows(device, monkeypatch, mode, kv_heads):
    case = make_case(
        q_lengths=(129, 0, 65),
        k_lengths=(129, 33, 65),
        q_heads=4,
        kv_heads=kv_heads,
        q_packed=True,
        k_packed=True,
        seed=1295,
    )
    qa = torch.arange(227, dtype=torch.int32) // 5
    ka = qa.clone()
    qa[9] = 100000
    if mode == "empty":
        qa.fill_(100000)
    elif mode == "full":
        qa.zero_()
        ka.zero_()
    elif mode == "value_nan":
        case["qkv"][2].view(torch.uint8)[0, 0, 0] = 127
    elif mode == "score_nan":
        case["qkv"][0].view(torch.uint8)[[3, 9], 0, :] = 127

    def mask(b, h, q, k, info, aux, scalars):
        return aux[0][q + info[2]] == aux[1][k + info[3]]

    aux = (qa, ka)
    sparse = _blocks(case, mask, aux)
    # Force an empty row to share a selected tile with active rows, and also
    # exercise an entirely omitted tile and a two-tile row in the same launch.
    if mode != "full":
        sparse.mask_block_cnt[0, 0, 1] = 0
        sparse.full_block_cnt[0, 0, 1] = 0
    expected, logs = reference(case, mask_mod=mask, aux=aux, sparse=sparse, scale=-0.125)
    data, kwargs = to_device(case, device)
    orig = runtime._make_mxfp8_plan

    def capture(*args, **options):
        plan = orig(*args, **options)
        assert plan.guard_elements == 0, "native exact_aux is required"
        return plan

    monkeypatch.setattr(runtime, "_make_mxfp8_plan", capture)
    runtime.clear_compile_cache()
    try:
        for _ in range(20 if mode == "mixed" else 3):
            out, lse = flash_attn_varlen_func(
                *data,
                **kwargs,
                mask_mod=global_labels,
                aux_tensors=tuple(t.to(device) for t in aux),
                block_sparse_tensors=_device_sparse(sparse, device),
                softmax_scale=-0.125,
                return_lse=True,
            )
            out, lse = out.cpu(), lse.cpu()
            assert torch.equal(torch.isnan(out), torch.isnan(expected))
            finite = torch.isfinite(expected)
            if finite.any():
                check(out[finite], expected[finite])
            assert torch.equal(torch.isnan(lse), torch.isnan(logs))
            good_lse = ~torch.isnan(logs)
            check_quantized_lse(lse[good_lse], logs[good_lse])
            empty = torch.isneginf(logs).T.unsqueeze(-1).expand_as(out)
            assert not out[empty].count_nonzero()
    finally:
        runtime.clear_compile_cache()


@pytest.mark.npu
@pytest.mark.parametrize("heads,kv_heads", [(4, 2), (8, 8)])
def test_wide_labels_rebind_uniform_nonuniform_and_unsorted_halves(
    device, monkeypatch, heads, kv_heads
):
    case = make_case(
        q_lengths=(257, 65),
        k_lengths=(257, 129),
        q_heads=heads,
        kv_heads=kv_heads,
        q_packed=True,
        k_packed=True,
        seed=1298,
    )
    qa = torch.empty(386, dtype=torch.int64)
    ka = torch.empty_like(qa)
    aux = (qa.to(device), ka.to(device))
    data, kwargs = to_device(case, device)
    counts = torch.tensor([[[3, 3, 3]], [[2, 2, 2]]], dtype=torch.int32)
    indices = torch.arange(3, dtype=torch.int32).expand(2, 1, 3, 3).contiguous()
    # All listed blocks stay partial, including fully visible and empty ones.
    sparse = BlockSparseTensorsTorch(
        counts,
        indices,
        torch.zeros_like(counts),
        torch.full_like(indices, -111),
        block_size=(128, 128),
    )
    blocks = _device_sparse(sparse, device)

    def mask(b, h, q, k, info, aux, scalars):
        return aux[0][q + info[2]] == aux[1][k + info[3]]

    orig = runtime._make_mxfp8_plan

    def capture(*args, **options):
        plan = orig(*args, **options)
        assert plan.guard_elements == 0
        return plan

    monkeypatch.setattr(runtime, "_make_mxfp8_plan", capture)
    runtime.clear_compile_cache()
    try:
        cache_keys = None
        for mode in ("halves", "high_words", "unsorted", "nonuniform_queries", "wide", "empty"):
            qa.copy_(torch.arange(386, dtype=torch.int64) // 64 + 2**40)
            ka.copy_(qa)
            if mode == "high_words":
                qa.copy_((torch.arange(386) // 64) * 2**40 + 7)
                ka.copy_(qa)
            elif mode == "unsorted":
                ka[:256].copy_(
                    ka[:256].clone()[
                        torch.randperm(256, generator=torch.Generator().manual_seed(98))
                    ]
                )
            elif mode == "nonuniform_queries":
                qa[:256].copy_(
                    qa[:256].clone()[
                        torch.randperm(256, generator=torch.Generator().manual_seed(99))
                    ]
                )
            elif mode == "wide":
                qa.fill_(-(2**40))
                ka.fill_(-(2**40))
            elif mode == "empty":
                qa.fill_(-(2**40))
                ka.fill_(2**40)
            aux[0].copy_(qa.to(device))
            aux[1].copy_(ka.to(device))
            expected, logs = reference(
                case, mask_mod=mask, sparse=sparse, aux=(qa, ka), scale=-0.125
            )
            out, lse = flash_attn_varlen_func(
                *data,
                **kwargs,
                mask_mod=global_labels,
                aux_tensors=aux,
                block_sparse_tensors=blocks,
                softmax_scale=-0.125,
                return_lse=True,
            )
            check(out, expected)
            check_quantized_lse(lse.cpu(), logs)
            if cache_keys is not None:
                assert set(runtime.compile_cache) == cache_keys
            cache_keys = set(runtime.compile_cache)
    finally:
        runtime.clear_compile_cache()


# Label geometry and source rebinding.


@simt
@tla.jit
def batch_labels(b, h, q, k, info, aux, scalars):
    return aux[0][b, q] == aux[1][b, k]


@simt
@tla.jit
def last_head_labels(b, h, q, k, info, aux, scalars):
    return aux[2][b, h, q] == aux[3][b, h, k]


@pytest.mark.npu
@pytest.mark.parametrize(
    "dtype,per_head,padding",
    [
        (torch.int32, False, 0),
        (torch.int32, True, 0),
        (torch.int64, False, 0),
        (torch.int64, True, 0),
        (torch.int64, True, 2),
    ],
)
def test_label_geometry_is_local_to_sources_and_launch(
    device, monkeypatch, per_head, dtype, padding
):
    case = make_case(
        q_lengths=(256, 129),
        k_lengths=(256, 193),
        q_heads=8,
        kv_heads=8,
        q_packed=True,
        k_packed=True,
        seed=1409,
    )
    wide = dtype == torch.int64
    base = (
        (torch.arange(256, dtype=dtype) // 64) * 2**40 + 7
        if wide
        else torch.arange(256, dtype=dtype) // 5 + 7
    )
    qa = (
        base[None, None].expand(2, 8, 256).clone()
        if per_head
        else base[None].expand(2, 256).clone()
    )
    ka = qa.clone()
    host_aux = (qa.clone(), ka.clone()) * bool(padding) + (qa, ka)
    aux = tuple(t.to(device) for t in host_aux)
    inputs, kwargs = to_device(case, device)
    native = last_head_labels if padding else head_labels if per_head else batch_labels

    def mask(b, h, q, k, info, aux, scalars):
        return (
            aux[padding][b, h, q] == aux[padding + 1][b, h, k]
            if per_head
            else aux[0][b, q] == aux[1][b, k]
        )

    monkeypatch.setattr(runtime, "_cores", lambda *args: 2)
    runtime.clear_compile_cache()
    try:
        keys = None
        for update in range(3):
            # Each full KV tile has different int64 labels in its two halves.
            # Alternate heads select opposite halves, with equal low words.
            qa.copy_(base + (update + 1) * (2**48 if wide else 1024))
            ka.copy_(qa)
            if not wide:
                if per_head:
                    for head in range(8):
                        ka[:, head] += (head + update) % 2
                else:
                    ka += update % 2
            elif per_head:
                for head in range(8):
                    if (head + update) % 2:
                        ka[:, head].copy_(
                            ka[:, head].reshape(2, 2, 128).roll(64, -1).reshape(2, 256)
                        )
            elif update % 2:
                ka.copy_(ka.reshape(2, 2, 128).roll(64, -1).reshape(2, 256))
            parts = [
                _blocks(case, lambda b, h, q, k, i, a, s: mask(b, head, q, k, i, a, s), host_aux)
                for head in range(8)
            ]
            fields = ("mask_block_cnt", "mask_block_idx", "full_block_cnt", "full_block_idx")
            sparse = BlockSparseTensorsTorch(
                *(torch.cat([getattr(p, f) for p in parts], dim=1) for f in fields),
                block_size=(128, 128),
            )
            # An authoritative full block and an omitted row interrupt reuse.
            sparse.mask_block_cnt[0, 3, 0] = 0
            sparse.full_block_cnt[0, 3, 0] = 1
            sparse.full_block_idx[0, 3, 0, 0] = 1
            sparse.mask_block_cnt[0, 5, 1] = 0
            sparse.full_block_cnt[0, 5, 1] = 0
            expected, logs = reference(
                case, mask_mod=mask, sparse=sparse, aux=host_aux, scale=-0.125
            )
            aux[-2].copy_(qa.to(device))
            aux[-1].copy_(ka.to(device))
            out, lse = flash_attn_varlen_func(
                *inputs,
                **kwargs,
                mask_mod=native,
                aux_tensors=aux,
                block_sparse_tensors=_device_sparse(sparse, device),
                softmax_scale=-0.125,
                return_lse=True,
            )
            check(out, expected)
            check_quantized_lse(lse.cpu(), logs)
            if keys is not None:
                assert set(runtime.compile_cache) == keys
            keys = set(runtime.compile_cache)
    finally:
        runtime.clear_compile_cache()


# Internal K/V packing.


@pytest.mark.npu
@pytest.mark.parametrize(
    "layouts,repair", [((True, True), False), ((False, True), True), ((True, False), False)]
)
def test_internal_kv_prepack_mixed_layouts_and_repair(device, monkeypatch, layouts, repair):
    qp, kp = layouts
    case = make_case(
        q_lengths=(65, 129) if qp else (129, 129),
        k_lengths=(129, 31) if kp else (129, 129),
        q_packed=qp,
        k_packed=kp,
        q_heads=4,
        kv_heads=2,
        seed=929,
    )
    if repair:
        for tensor, value in zip(case["qkv"][:2], (1.0, -1.0)):
            code = torch.tensor(value).to(torch.float8_e4m3fn).view(torch.uint8)
            tensor.view(torch.uint8).fill_(int(code))
        case["kwargs"]["q_descale"].fill_(127)
        case["kwargs"]["k_descale"].fill_(127)
    expected, logs = reference(case)
    inputs, kwargs = to_device(case, device)
    monkeypatch.setattr(runtime, "_mxfp8_should_pack_kv", lambda *args: True)
    runtime.clear_compile_cache()
    original = runtime._launch
    copied = []

    def launch(executor, payload, **options):
        copied.append(payload[1].data_ptr() != inputs[1].data_ptr())
        return original(executor, payload, **options)

    monkeypatch.setattr(runtime, "_launch", launch)
    try:
        for _ in range(2):
            output, lse = flash_attn_varlen_func(*inputs, **kwargs, return_lse=True)
            check(output, expected)
            check_quantized_lse(lse.cpu(), logs)
        assert copied and all(copied)
    finally:
        runtime.clear_compile_cache()


@pytest.mark.npu
@pytest.mark.parametrize("compiled", [False, True])
def test_prepacked_boundaries_and_scales_rebind_in_graph(device, monkeypatch, compiled):
    monkeypatch.setattr(runtime, "_mxfp8_should_pack_kv", lambda *args: True)
    runtime.clear_compile_cache()
    try:
        test_mxfp8_rebinds_packed_boundaries_with_capacity(device, compiled)
    finally:
        runtime.clear_compile_cache()


# Benchmark timing and report integrity.

path = Path(__file__).resolve().parents[2] / "benchmarks/benchmark_catlass_mxfp8.py"
spec = importlib.util.spec_from_file_location("mxfp8_benchmark", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_statistics_keep_outliers_and_report_dispersion():
    values = [1.0, 2.0, 1000.0]
    measured = benchmark.statistics_us(values)
    assert measured["samples_us"] == values
    assert measured["mean_us"] == pytest.approx(1003 / 3)
    assert measured["median_us"] == 2.0
    assert measured["p95_us"] == 1000.0
    assert measured["coefficient_of_variation"] > 1.0


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [-1.0]])
def test_invalid_timing_samples(values):
    with pytest.raises(ValueError, match="nonempty, finite and nonnegative"):
        benchmark.statistics_us(values)


def test_visible_device_maps_logical_to_physical(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "6,3")
    assert benchmark.physical_device("npu:0") == 6
    assert benchmark.physical_device("npu:1") == 3
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES")
    assert benchmark.physical_device("npu:7") == 7


def test_process_detection_ignores_other_devices_and_headers():
    status = """
| NPU ID | Process id | Process name | Process memory(MB) |
| 6 | 123 | python | 100 |
| 6 | 456 | python | 101 |
| 7 | 789 | python | 102 |
| No running processes found in NPU 0 |
"""
    assert benchmark.device_processes(status, 6) == [123, 456]
    assert benchmark.device_processes(status, 0) == []


def test_default_case_matrix():
    args = SimpleNamespace(
        lengths=[1024, 8192, 16384, 65536, 65531], sparsities=[0.5, 0.9], no_extra=False
    )
    cases = list(benchmark.specifications(args))
    assert len(cases) == 14
    assert len({case["name"] for case in cases}) == 14
    assert [case["kind"] for case in cases[-4:]] == ["dense", "causal", "custom", "tnd"]
    args.no_extra = True
    assert len(list(benchmark.specifications(args))) == 10


def fake_profiler(monkeypatch):
    calls = []

    class Profile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def step(self):
            calls.append("step")

    profiler = SimpleNamespace(
        _ExperimentalConfig=lambda **kw: kw,
        ExportType=SimpleNamespace(Text="text"),
        ProfilerLevel=SimpleNamespace(Level1=1),
        ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
        profile=lambda **kw: Profile(),
        tensorboard_trace_handler=lambda directory: directory,
        schedule=lambda **kw: kw,
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=profiler))
    return calls


def write_trace(directory, rows):
    directory.mkdir()
    with (directory / "kernel_details.csv").open("w") as out:
        writer = csv.DictWriter(
            out, fieldnames=["Name", "Start Time(us)", "Duration(us)", "Device_id"]
        )
        writer.writeheader()
        for name, start, duration, device in rows:
            writer.writerow(
                {
                    "Name": name,
                    "Start Time(us)": start,
                    "Duration(us)": duration,
                    "Device_id": device,
                }
            )


def test_profiler_sums_auxiliary_kernels_per_call(tmp_path, monkeypatch):
    fake_profiler(monkeypatch)
    path = tmp_path / "trace"
    write_trace(
        path,
        [
            ("metadata", 0, 2, 6),
            ("attention", 1, 10, 6),
            ("metadata", 2, 3, 6),
            ("attention", 3, 11, 6),
        ],
    )
    called = []
    result = benchmark.profile_call(
        lambda: called.append(1), path, 2, SimpleNamespace(check=lambda: None)
    )
    assert len(called) == 3  # profiler warmup plus two measured invocations
    assert result["samples_us"] == [12.0, 14.0]
    assert result["mean_us"] == 13.0
    assert result["kernels_per_call"] == 2
    assert result["kernel_counts"] == {"metadata": 2, "attention": 2}


@pytest.mark.parametrize(
    "rows,match",
    [
        ([("a", 0, 1, 6), ("b", 1, 1, 6), ("a", 2, 1, 6)], "does not divide"),
        ([("a", 0, 1, 6), ("b", 1, 1, 6)], "sequence changed"),
        ([("a", 0, 0, 6), ("a", 1, 1, 6)], "nonpositive"),
        ([("a", 0, 1, 6), ("a", 1, 1, 7)], "multiple devices"),
    ],
)
def test_profiler_rejects_incomplete_or_inconsistent_data(tmp_path, monkeypatch, rows, match):
    fake_profiler(monkeypatch)
    path = tmp_path / "trace"
    write_trace(path, rows)
    with pytest.raises(RuntimeError, match=match):
        benchmark.profile_call(lambda: None, path, 2, SimpleNamespace(check=lambda: None))


def test_summary_includes_dispersion_and_partial_run_error(tmp_path):
    measured = benchmark.statistics_us([1.0, 2.0, 6.0])
    report = dict(
        status="failed",
        error="later case failed",
        git_head="123abc",
        catlass_version="2.2.0",
        cases=[
            dict(
                spec=dict(name="test-case"),
                aggregate={
                    role: {kind: measured for kind in ("device", "end_to_end")}
                    for role in ("mxfp8", "bf16")
                },
                device_speedup=1.0,
                end_to_end_speedup=1.0,
            )
        ],
    )
    benchmark.save_report(report, tmp_path)
    with (tmp_path / "summary.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 4
    assert all(float(row["median_us"]) == 2.0 for row in rows)
    assert all(float(row["p95_us"]) == 6.0 for row in rows)
    assert all(float(row["coefficient_of_variation"]) > 0 for row in rows)
    markdown = (tmp_path / "report.md").read_text()
    assert "123abc" in markdown and "later case failed" in markdown
    assert "Median (us)" in markdown and "CV" in markdown
    assert json.loads((tmp_path / "result.json").read_text())["cases"][0]["aggregate"]["mxfp8"][
        "device"
    ]["samples_us"] == [1, 2, 6]


def test_environment_metadata_reads_installed_versions(tmp_path, monkeypatch):
    root = tmp_path / "cann"
    version = root / "share/info/runtime/version.info"
    version.parent.mkdir(parents=True)
    version.write_text("Version=9.2.0\n")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(root))
    monkeypatch.setenv("CATLASS_DSL_CACHE", "1")
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="abc123\n", stderr=""),
    )
    result = benchmark.environment_metadata()
    assert result["git_head"] == "abc123"
    assert result["cann_version_files"][str(version)] == "Version=9.2.0\n"
    assert result["compile_environment"]["CATLASS_DSL_CACHE"] == "1"


def test_compiler_records_actual_options_and_preserves_manifest(tmp_path, monkeypatch):
    @dataclass
    class Options:
        arch_scope: str
        cache_dir: Path

    monkeypatch.setitem(
        sys.modules,
        "catlass.execution",
        SimpleNamespace(compile_option_from_kwargs=lambda kw: Options("c310-aiv", tmp_path)),
    )
    binary = tmp_path / "kernel.o"
    binary.write_bytes(b"test binary")
    (tmp_path / "manifest.json").write_text('{"arch_scope":"c310-aic"}')
    seen = []

    def compile_fn(kernel, *args, **kwargs):
        seen.append((kernel, args, kwargs))
        return SimpleNamespace(kernel_binary_path=binary)

    recorder = benchmark.CompileRecorder(SimpleNamespace(compile=compile_fn), tmp_path)
    result = recorder("kernel", "sample", options="--npu-arch 3510")
    assert result.kernel_binary_path == binary
    assert seen == [("kernel", ("sample",), {"options": "--npu-arch 3510"})]
    entry = recorder.records[0]
    assert entry["requested_options"] == "--npu-arch 3510"
    assert entry["sdk_options"]["cache_dir"] == tmp_path
    assert len(entry["binary_sha256"]) == 64
    assert json.loads((tmp_path / entry["manifest"]).read_text())["arch_scope"] == "c310-aic"


def thermal_text(temperature=75, frequency=1650, nominal=1650):
    return (
        f"\tNPU ID : 5\n\tAicore Freq(MHZ) : {nominal}\n"
        f"\tAicore curFreq(MHZ) : {frequency}\n\tTemperature(C) : {temperature}\n"
    )


def test_parse_temperature_and_frequency_from_common():
    assert benchmark.parse_thermal(thermal_text(99, 400)) == dict(
        temperature_c=99.0, frequency_mhz=400.0, nominal_frequency_mhz=1650.0
    )


@pytest.mark.parametrize(
    "common", ["Temperature(C) : 75", thermal_text("nan"), thermal_text(75, 0)]
)
def test_thermal_parser_rejects_missing_or_invalid_fields(common):
    with pytest.raises(RuntimeError, match="temperature/frequency"):
        benchmark.parse_thermal(common)


def test_thermal_pacing_hysteresis_batching_and_frequency_recording(tmp_path, monkeypatch):
    states = iter([thermal_text(85), thermal_text(86), thermal_text(84), thermal_text(75, 100)])
    reads, waits, checks = [], [], []

    def read(physical):
        reads.append(physical)
        return next(states)

    clock = [0.0]

    def wait(seconds):
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(benchmark, "npu_common", read)
    monkeypatch.setattr(benchmark.time, "sleep", wait)
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: clock[0])
    pacer = benchmark.ThermalPacer(5, tmp_path, SimpleNamespace(check=lambda: checks.append(1)))
    pacer.checkpoint("profile", 0)
    pacer.checkpoint("profile", 5)  # no query between batch boundaries
    pacer.checkpoint("profile", 10)
    assert reads == [5] * 4 and waits == [3.0] * 2
    assert len(checks) == 4
    entries = [
        json.loads(line) for line in (tmp_path / "thermal_pacing.jsonl").read_text().splitlines()
    ]
    assert [entry["event"] for entry in entries].count("cooling_start") == 1
    assert entries[-1]["event"] == "cooling_end"
    assert entries[-1]["temperature_c"] == 75 and entries[-1]["frequency_mhz"] == 100
    assert entries[-1]["elapsed_seconds"] == 6
    assert pacer.metadata()["cooldown_seconds"] == 6


def test_low_idle_frequency_does_not_block_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "npu_common", lambda physical: thermal_text(67, 100))
    monkeypatch.setattr(
        benchmark.time, "sleep", lambda seconds: pytest.fail("idle DVFS must not cause cooling")
    )
    pacer = benchmark.ThermalPacer(5, tmp_path, SimpleNamespace(check=lambda: None))
    pacer.checkpoint("warmup", 0)
    assert pacer.metadata()["cooldown_count"] == 0


def test_thermal_pacing_disabled_does_not_query_device(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark, "npu_common", lambda physical: pytest.fail("unexpected device query")
    )
    pacer = benchmark.ThermalPacer(5, tmp_path, SimpleNamespace(check=lambda: None), enabled=False)
    pacer.checkpoint("profile", 0)
    assert not (tmp_path / "thermal_pacing.jsonl").exists()


def test_thermal_interruption_is_logged_and_run_marked_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "npu_common", lambda physical: thermal_text(90))

    def interrupted(seconds):
        raise KeyboardInterrupt()

    monkeypatch.setattr(benchmark.time, "sleep", interrupted)
    monitor = SimpleNamespace(check=lambda: None, contended=SimpleNamespace(is_set=lambda: False))
    pacer = benchmark.ThermalPacer(5, tmp_path, monitor)
    with pytest.raises(KeyboardInterrupt) as caught:
        pacer.checkpoint("end_to_end", 10)
    report = dict(cases=[], active_case=dict(spec=dict(name="partial"), rounds=[]))
    benchmark.record_failure(report, caught.value, monitor)
    report["thermal_pacing"] = pacer.metadata()
    benchmark.save_report(report, tmp_path)
    entries = [
        json.loads(line) for line in (tmp_path / "thermal_pacing.jsonl").read_text().splitlines()
    ]
    assert entries[-1]["event"] == "cooling_interrupted"
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["status"] == "interrupted" and saved["active_case"]["spec"]["name"] == "partial"
    assert "KeyboardInterrupt" in saved["error"]


def test_pacing_wait_is_outside_end_to_end_sample_window(monkeypatch):
    clock = [0]
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
    )
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", lambda: clock[0])
    stages = []

    def checkpoint(stage, sample):
        stages.append((stage, sample))
        clock[0] += 3_000_000_000

    def kernel():
        clock[0] += 25_000

    result = benchmark.end_to_end(
        kernel,
        3,
        SimpleNamespace(check=lambda: None),
        SimpleNamespace(checkpoint=checkpoint),
        "trial",
    )
    assert result["samples_us"] == [25.0] * 3
    assert stages == [("trial", i) for i in range(3)]


def test_warmup_is_also_synchronized_and_paced(monkeypatch):
    events = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: events.append("sync"))),
    )
    pacer = SimpleNamespace(checkpoint=lambda stage, sample: events.append((stage, sample)))
    benchmark.warmup(
        lambda: events.append("kernel"), 20, SimpleNamespace(check=lambda: None), pacer
    )
    assert events.count("kernel") == 20 and events.count("sync") == 21
    assert events[1:4] == [("warmup", 0), "kernel", "sync"]
    assert events[-3:] == [("warmup", 19), "kernel", "sync"]


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
suite = importlib.import_module("benchmark_catlass_mxfp8_suite")


def test_reference_report_matrix_and_ragged_local_boundaries():
    settings = suite.report_settings()
    cases = suite.make_specs(settings, ["BSND", "TND", "TND-ragged"])
    assert len(cases) == 204 and len({c["id"] for c in cases}) == 204
    assert len(settings["masks"]) == 17
    for c in cases:
        assert sum(c["q_lengths"]) == sum(c["k_lengths"]) == c["length"]
        assert c["heads"] == c["kv_heads"] == 24
        if c["scene"] == "TND-ragged":
            assert len(set(c["q_lengths"])) == 3
            assert c["q_lengths"][1:] == (65, 129)


def test_packed_metadata_decode_keeps_sequence_and_head_axes():
    # Rows per sequence 1,2; columns 2,1. Index-slot offsets 0,2,4.
    sparse = BlockSparseTensorsTorch(
        torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.int32),
        torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int32),
        torch.zeros((2, 3), dtype=torch.int32),
        torch.zeros((2, 4), dtype=torch.int32),
        cu_total_m_blocks=torch.tensor([0, 1, 3], dtype=torch.int32),
        cu_block_idx_offsets=torch.tensor([0, 2, 4], dtype=torch.int32),
        block_size=(128, 128),
    )
    out = suite.cpu_blocks(sparse, dict(q_lengths=(65, 129), k_lengths=(129, 65)))
    assert out.mask_block_cnt.tolist() == [[[1, 0], [0, 0]], [[1, 0], [0, 1]]]
    assert out.mask_block_idx[0, 0, 0, 0] == 1
    assert out.cu_total_m_blocks is None


@pytest.mark.parametrize("error", [0.0, 0.005, 0.2, float("nan")])
def test_streaming_mixed_tolerance_agrees_with_existing_check(error):
    expected = torch.zeros(4097, dtype=torch.bfloat16)
    actual = expected.clone()
    actual[0] = error
    try:
        check(actual, expected)
        passed = True
    except AssertionError:
        passed = False
    assert suite.compare_full(actual, expected)["passed"] == passed


def test_report_retains_invalid_attempts_without_counting_them_as_passed(tmp_path):
    settings = dict(suite.report_settings(), warmup=20, trials=100, rounds=3)
    spec = suite.make_specs(settings, ["TND"])[0]
    folder = tmp_path / "cases" / spec["id"] / "attempt-01"
    folder.mkdir(parents=True)
    suite.write_json(
        folder / "result.json", dict(spec=spec, status="contention", error="another process")
    )
    results = suite.render_report(tmp_path, settings, [spec])
    assert results[0]["status"] == "contention"
    assert results[0]["attempt_count"] == 1 and "speedup" not in results[0]
    assert (tmp_path / "report.html").is_file()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["results"][0]["record"].endswith("attempt-01/result.json")


def test_golden_contract_reports_bf16_error_without_changing_raw_record(tmp_path):
    settings = dict(
        suite.report_settings(),
        warmup=20,
        trials=100,
        rounds=3,
        accuracy_contract="mxfp8_online_golden",
    )
    spec = suite.make_specs(settings, ["TND"])[0]
    folder = tmp_path / "cases" / spec["id"] / "attempt-01"
    folder.mkdir(parents=True)
    accuracy = dict(
        passed=False,
        sampled_reference=dict(passed=True),
        full_output_vs_bf16=dict(
            passed=False, nonfinite_classes_match=True, matched_fraction=0.979
        ),
    )
    raw = dict(spec=spec, status="accuracy_failed", accuracy=accuracy)
    suite.write_json(folder / "result.json", raw)
    recorded = (folder / "result.json").read_bytes()
    result = suite.render_report(tmp_path, settings, [spec])[0]
    assert result["status"] == "passed" and result["recorded_status"] == "accuracy_failed"
    assert result["accuracy"]["full_output_vs_bf16"]["matched_fraction"] == 0.979
    assert (folder / "result.json").read_bytes() == recorded
    accuracy["sampled_reference"]["passed"] = False
    assert not suite.golden_accuracy_passed(accuracy)
    accuracy["sampled_reference"]["passed"] = True
    accuracy["full_output_vs_bf16"]["nonfinite_classes_match"] = False
    assert not suite.golden_accuracy_passed(accuracy)


def test_revision_report_requires_complete_paired_coverage(tmp_path):
    settings = dict(suite.report_settings(), warmup=3, trials=10, rounds=1)
    specs = suite.make_specs(settings, ["BSND"])[:2]
    for spec, previous in zip(specs, (20.0, 5.0)):
        folder = tmp_path / "cases" / spec["id"] / "attempt-01"
        folder.mkdir(parents=True)
        measurements = {
            name: dict(mean_us=value, p95_us=value, coefficient_of_variation=0)
            for name, value in (("bf16", 30.0), ("mxfp8", 10.0), ("previous_mxfp8", previous))
        }
        suite.write_json(
            folder / "result.json",
            dict(
                spec=spec,
                status="passed",
                accuracy=dict(passed=True),
                speedup=3.0,
                revision_speedup=previous / 10,
                measurements=measurements,
            ),
        )
        suite.render_report(tmp_path, settings, specs)
        report = json.loads((tmp_path / "summary.json").read_text())["revision_comparison"]
        assert report["complete"] == (spec == specs[-1])
    assert report["geometric_mean"] == pytest.approx(1.0)
    assert report["total_time_ratio"] == pytest.approx(1.25)
    assert report["mean_nonregression"]
    assert report["regressions_over_5_percent"] == [specs[1]["id"]]
    assert report["by_scene"]["BSND"]["geometric_mean"] == pytest.approx(1.0)
    assert report["by_scene"]["BSND"]["measured"] == 2


def test_previous_revision_binds_planning_helpers_and_restores_on_error(monkeypatch):
    from types import ModuleType, SimpleNamespace

    from mxfp8_revision import PreviousRevision

    name = "flash_attn_npu_4.catlass_dsl.kernels.flash_fwd_mxfp8_arch35"
    package_name = name.rsplit(".", 1)[0]
    current, previous = ModuleType(name), ModuleType("saved_kernel")
    package = ModuleType(package_name)
    package.flash_fwd_mxfp8_arch35 = current
    monkeypatch.setitem(sys.modules, name, current)
    monkeypatch.setitem(sys.modules, package_name, package)
    binder = PreviousRevision.__new__(PreviousRevision)
    binder.current_runtime = SimpleNamespace(launch_mxfp8=object())
    binder.runtime = SimpleNamespace(launch_mxfp8=object())
    binder.kernel = previous
    launch = binder.current_runtime.launch_mxfp8
    with pytest.raises(ValueError), binder.bind():
        from flash_attn_npu_4.catlass_dsl.kernels import flash_fwd_mxfp8_arch35

        assert flash_fwd_mxfp8_arch35 is previous
        assert sys.modules[name] is previous
        assert binder.current_runtime.launch_mxfp8 is binder.runtime.launch_mxfp8
        raise ValueError("benchmark error")
    assert binder.current_runtime.launch_mxfp8 is launch
    assert sys.modules[name] is current
    assert package.flash_fwd_mxfp8_arch35 is current


def test_report_keeps_reused_measurements_distinct_from_new_runs(tmp_path):
    settings = dict(suite.report_settings(), warmup=3, trials=10, rounds=1)
    spec = suite.make_specs(settings, ["BSND"])[0]
    folder = tmp_path / "cases" / spec["id"] / "attempt-01"
    folder.mkdir(parents=True)
    measurements = {
        name: dict(mean_us=value, p95_us=value, coefficient_of_variation=0)
        for name, value in (("bf16", 30.0), ("mxfp8", 10.0), ("previous_mxfp8", 12.0))
    }
    reuse = dict(source_record="previous/case/result.json", audit="binary-audit/case.json")
    raw = dict(
        spec=spec,
        status="passed",
        created_utc="original timestamp",
        accuracy=dict(passed=True),
        speedup=3.0,
        revision_speedup=1.2,
        measurements=measurements,
        measurement_reuse=reuse,
    )
    suite.write_json(folder / "result.json", raw)
    saved = (folder / "result.json").read_bytes()
    suite.render_report(tmp_path, settings, [spec])
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["measurement_provenance"]["newly_measured"] == 0
    assert report["measurement_provenance"]["binary_verified_reused"] == 1
    assert report["results"][0]["created_utc"] == "original timestamp"
    assert (folder / "result.json").read_bytes() == saved
    assert "binary_verified_reuse" in (tmp_path / "summary.csv").read_text()
    assert "本轮实测 0 组" in (tmp_path / "report.html").read_text()
