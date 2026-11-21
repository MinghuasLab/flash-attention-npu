# Copyright (c) 2026, Minghua Shen.

"""MXFP8 forward tests: golden, RNE, layouts, masks, and runtime.

NPU cases carry explicit markers; CPU cases can be selected with ``-m "not npu"``.
The independent golden remains in mxfp8_reference.py for benchmark reuse.

Pure CPU boundary tables retain all values in one test per contract. NPU tests
cover representative layouts, numerical edges, callbacks and runtime lifetimes;
avoid multiplying these by every mask, head count and tail length. The ordered
phase-handoff regression remains intact because it previously exposed a hang.
Repeated launches are reserved for cache, stream and slot-reuse checks.
"""

import copy
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from catlass import tla
from torch._subclasses.fake_tensor import FakeTensorMode

import flash_attn_npu_4 as fa4
from flash_attn_npu_4.catlass_dsl import (
    BlockScales,
    flash_attn_func,
    flash_attn_varlen_func,
    graph,
    interface,
    runtime,
    simd,
    simt,
)
from flash_attn_npu_4.catlass_dsl.block_sparsity import BlockSparseTensorsTorch
from flash_attn_npu_4.catlass_dsl.cache_utils import get_jit_cache
from flash_attn_npu_4.catlass_dsl.kernels.flash_fwd_mxfp8_arch35 import (
    _mx_fill_ready,
    _mx_fill_scales,
    _mx_group_scale,
    pack_scaled_rne,
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
def test_v_scales_pad_each_sequence_to64():
    for length in [0, 31, 32, 33, 63, 64, 65, 127, 128, 129]:
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
def test_probability_exponent_threshold_neighbors():
    for exponent in [-127, -120, -20, -8, 0, 100]:
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
def test_input_quantizer_rejects_nonfinite():
    for value in [float("nan"), float("inf"), -float("inf")]:
        with pytest.raises(ValueError, match="finite"):
            quantize_group32(torch.tensor([value]))


@pytest.mark.usefixtures("single_cpu_thread")
def test_online_lse_matches_independent_dense_math():
    for layout, q_packed, k_packed in [
        ("BSND", False, False),
        ("BNSD", False, False),
        ("BSND", True, False),
        ("BNSD", True, False),
        ("BSND", False, True),
        ("BNSD", False, True),
        ("BSND", True, True),
    ]:
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
                    restored[index][offset : offset + len(value)]
                    if packed
                    else restored[index][batch]
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
def test_packed_v_capacity_is_independent_of_segment_boundaries():
    for first, second in [((65, 63), (64, 64)), ((31, 33), (32, 32))]:
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
        assert (
            cases[0]["kwargs"][SCALE_NAMES[-1]].shape == cases[1]["kwargs"][SCALE_NAMES[-1]].shape
        )
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


def test_scale_metadata_layouts():
    for packed, layout in [(False, "BSND"), (False, "BNSD"), (True, "BSND")]:
        q, k, v, *scales = inputs(packed=packed, layout=layout)
        result = interface._resolve_mxfp8_scales(q, k, v, scales, layout=layout)
        assert all(actual is expected for actual, expected in zip(result, scales))
        assert interface._resolve_mxfp8_scales(q, k, v, None) is None


def test_all_three_scales_are_required():
    for missing in [0, 1, 2]:
        q, k, v, *scales = inputs()
        scales[missing] = None
        with pytest.raises(ValueError, match="all three"):
            interface._resolve_mxfp8_scales(q, k, v, scales)


def test_invalid_mxfp8_metadata():
    for invalid in ["dtype", "shape", "device", "q_dtype", "head_dim"]:
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


def test_scale_canonicalization_preserves_bytes_and_alignment():
    for native_e8m0 in [False, True]:
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


def test_mxfp8_gradients_are_rejected():
    for no_grad in [False, True]:
        for operand in ["q", "k", "v", "q_scale"]:
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


def test_mxfp8_fake_output_contract():
    for packed, layout in [(False, "BSND"), (False, "BNSD"), (True, "BSND")]:
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
    assert config["scales"][0] is data[3]


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
        entry = interface.flash_attn_varlen_func if packed else interface.flash_attn_func

    def call(q, k, v, qs, ks, vs, cq, ck):
        options = dict(scales=BlockScales(qs, ks, vs), return_lse=True)
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


@pytest.fixture
def isolated_compile_cache(monkeypatch):
    """Keep compile-count/policy checks from evicting unrelated warm kernels."""
    cache = get_jit_cache()
    monkeypatch.setattr(runtime, "compile_cache", cache)
    yield cache
    cache.clear()


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
        (1024, None, True, ("continuous", 4)),
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
    monkeypatch.setattr(
        torch.npu, "get_device_properties", lambda device: SimpleNamespace(L2_cache_size=128 << 20)
    )
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
        assert actual[16][: len(config["aux_scalars"])] == config["aux_scalars"]
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
    monkeypatch.setattr(
        runtime, "_make_mxfp8_plan", lambda *args: replace(make_plan(*args), guard_elements=64)
    )
    for _ in range(2):
        runtime.launch_mxfp8(*make_launch())
    calls = launcher.call_args_list
    launches = 1
    assert len(calls) == 2 * launches and compiler.call_count == 1
    first, second = calls[0].args[1][6], calls[launches].args[1][6]
    assert first.shape == second.shape == (64,)
    assert first.dtype == second.dtype == torch.int32
    assert first.data_ptr() != second.data_ptr()


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


def score_reference(s, b, h, q, k, info, aux, scalars):
    return s * scalars[0] + s * s * 0.1


def mask_reference(b, h, q, k, info, aux, scalars):
    return (k <= q + info[1] - info[0]) & (q != 0)


def _device_sparse(sparse, device):
    if sparse is None:
        return None
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
        (17, 23, 1, {}),
        (31, 33, 2, {}),
        (33, 257, 2, {"window_size": (17, 29)}),
        (33, 257, 2, {"softcap": 2.0}),
        (128, 256, 2, {"softmax_scale": 0.0}),
        (128, 256, 2, {"softmax_scale": -0.125}),
    ],
)
def test_mxfp8_forward(device, record_property, sq, sk, kv_heads, options):
    case = make_case(q_lengths=(sq,), k_lengths=(sk,), kv_heads=kv_heads)
    ref = dict(options)
    if "softmax_scale" in ref:
        ref["scale"] = ref.pop("softmax_scale")
    verify(case, device, options=options, reference_options=ref, record_property=record_property)


@pytest.mark.npu
@pytest.mark.parametrize("q_packed,k_packed,native_scales", [(True, True, True)])
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
@pytest.mark.parametrize("mode,packed,sq,sk", [("simd", True, 129, 385)])
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
@pytest.mark.parametrize("kind", ["empty_q", "empty_kv", "empty_batch"])
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
                old.data_ptr() != current.data_ptr()
                for old, current in zip(previous, kwargs["scales"])
            )
        keys = set(runtime.compile_cache)
        previous = kwargs["scales"]


@pytest.mark.npu
@pytest.mark.parametrize("compiled", [True])
def test_mxfp8_rebinds_packed_boundaries_with_capacity(device, compiled):
    def call(q, k, v, qs, ks, vs, cq, ck):
        return flash_attn_varlen_func(
            q,
            k,
            v,
            scales=BlockScales(qs, ks, vs),
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
            *kwargs["scales"],
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
@pytest.mark.parametrize("operand", ["q", "k"])
@pytest.mark.parametrize("scale", [0.0])
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
@pytest.mark.parametrize("kind", ["visible_full", "packed_empty_kv"])
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
@pytest.mark.parametrize("packed,native_scales", [(True, True)])
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
    kwargs["scales"] = BlockScales(*(strided(scale) for scale in kwargs["scales"]))
    out, lse = flash_attn_varlen_func(*inputs, **kwargs, return_lse=True)
    expected, el = reference(case)
    check(out, expected)
    check(lse, el)


@pytest.mark.npu
def test_mxfp8_fixed_return_shape(device):
    case = make_case(q_lengths=(17,), k_lengths=(23,))
    inputs, kwargs = to_device(case, device)
    scales = {"scales": kwargs["scales"]}
    out, lse = flash_attn_func(*inputs, **scales, return_lse=False)
    assert out.shape == inputs[0].shape and out.dtype == torch.bfloat16 and lse is None


# Sparse traversal, online state and repair.


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


@simt
@tla.jit
def packed_visibility(b, h, q, k, info, aux, scalars):
    word = aux[0][q, k // 32]
    return ((word >> (k % 32).to(tla.Int64)) & tla.Int64(1)) != tla.Int64(0)


def _cpu_sparse(sparse):
    if sparse is None:
        return None
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
    scales = {"scales": scales["scales"]}
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
@pytest.mark.parametrize("capacity", [8])
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


# Sparse metadata and conservative callback proofs.


def test_periodic_prefix_proof_requires_exact_local_predicate():
    for kind, expected in [
        ("direct", True),
        ("reversed", True),
        ("strict", False),
        ("shifted", False),
        ("period64", False),
        ("extra", False),
        ("global", False),
    ]:
        from flash_attn_npu_4.catlass_dsl.kernels.flash_fwd_mxfp8_arch35 import (
            _mx_is_periodic_prefix,
        )

        @simt
        @tla.jit
        def mask(b, h, q, k, info, aux, scalars):
            if tla.const_expr(kind == "direct"):
                return q % 128 >= k % 128
            elif tla.const_expr(kind == "reversed"):
                return k % 128 <= q % 128
            elif tla.const_expr(kind == "strict"):
                return q % 128 > k % 128
            elif tla.const_expr(kind == "shifted"):
                return (q + 1) % 128 >= k % 128
            elif tla.const_expr(kind == "period64"):
                return q % 64 >= k % 64
            elif tla.const_expr(kind == "extra"):
                return (q % 128 >= k % 128) & (k < 128)
            else:
                return (q + info[2]) % 128 >= (k + info[3]) % 128

        assert _mx_is_periodic_prefix(mask.simt, 0, packed=True) is expected


# Global auxiliary indices and cached metadata.


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
def head_labels(b, h, q, k, info, aux, scalars):
    return aux[0][b, h, q] == aux[1][b, h, k]


@simt
@tla.jit
def global_key_filter(b, h, q, k, info, aux, scalars):
    return k >= aux[0][k + info[3]]


@simt
@tla.jit
def local_key_filter(b, h, q, k, info, aux, scalars):
    return k >= aux[0][k]


def test_key_aux_trace_distinguishes_local_and_global_indices():
    from flash_attn_npu_4.catlass_dsl.kernels.flash_fwd_mxfp8_arch35 import _mx_trace_key_aux

    assert _mx_trace_key_aux(local_key_filter.simt) == (False,)
    assert _mx_trace_key_aux(global_key_filter.simt) == (True,)
    assert _mx_trace_key_aux(global_labels.simt) == ()
    assert _mx_trace_key_aux(global_window.simt) == ()


def test_auxiliary_cache_roles_are_conservative():
    for kind in ["labels", "window", "keys", "head_labels", "shifted", "branch"]:
        for packed in [False, True]:
            from types import SimpleNamespace

            from flash_attn_npu_4.catlass_dsl.kernels.flash_fwd_mxfp8_arch35 import (
                _mx_aux_cache_roles,
            )

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
            shape, strides = (
                ((2, 4, 128), (512, 128, 1)) if kind == "head_labels" else ((1024,), (1,))
            )
            metadata = ((-1, shape, strides, False),) * 2
            assert _mx_aux_cache_roles(masks[kind], aux, (), metadata, packed) == expected[kind]


# Single-sequence capacities and graph rebinding.


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


# Internal K/V packing.


@pytest.mark.npu
@pytest.mark.parametrize("layouts,repair", [((False, True), True)])
@pytest.mark.usefixtures("isolated_compile_cache")
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
