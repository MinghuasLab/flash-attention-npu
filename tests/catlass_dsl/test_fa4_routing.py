# Copyright (c) 2026, Minghua Shen.

"""FA4 dispatch/ABI tests. CPU kernel substitution is explicit in cpu_runtime."""

import ast
import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Tuple
from unittest.mock import Mock

import pytest
import torch
import torch_npu

import flash_attn_npu_4 as fa4
from flash_attn_npu_4 import catlass_dsl, flash_attn_func, flash_attn_varlen_func
from flash_attn_npu_4 import flash_attn_npu_interface_950 as legacy
from flash_attn_npu_4.catlass_dsl import builtin, utils as dsl_utils

from .test_graph import cpu_runtime  # noqa: F401
from .test_utils import make_inputs


ENTRIES = (flash_attn_func, flash_attn_varlen_func)
TRIGGERS = (
    "score_mod",
    "score_mod_bwd",
    "mask_mod",
    "block_sparse_tensors",
    "block_sparse_tensors_bwd",
    "aux_tensors",
    "aux_scalars",
)


def trigger(name):
    if name == "aux_tensors":
        # Multi-element tensors must never be evaluated for truthiness.
        return [torch.ones(3)]
    if name == "aux_scalars":
        return (0.0,)
    return object()


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("name", TRIGGERS)
def test_each_flex_trigger_precedes_legacy(monkeypatch, entry, name):
    if name not in inspect.signature(entry).parameters:
        pytest.skip("The existing varlen signature has no backward sparse argument")
    backend = Mock(return_value=object())
    monkeypatch.setattr(dsl_utils, "attention", backend)
    monkeypatch.setattr(legacy.FlashAttnFunc, "apply", Mock(side_effect=AssertionError("legacy")))
    monkeypatch.setattr(
        legacy.FlashAttnVarlenFunc, "apply", Mock(side_effect=AssertionError("legacy"))
    )
    value = trigger(name)
    result = entry(None, None, None, **{name: value})
    assert result is backend.return_value
    assert backend.call_args.kwargs[name] is value
    assert backend.call_args.kwargs["varlen"] == (entry is flash_attn_varlen_func)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"scales": None},
        {"config": None},
        {"aux_tensors": [], "aux_scalars": ()},
        {"aux_tensors": (), "aux_scalars": []},
        {"causal": True},
        {"window_size": (-1, 3)},
        {"softcap": 2.0},
    ],
)
def test_default_backend_and_empty_aux(monkeypatch, entry, options):
    backend = Mock(return_value=object())
    cls = legacy.FlashAttnFunc if entry is flash_attn_func else legacy.FlashAttnVarlenFunc
    monkeypatch.setattr(cls, "apply", backend)
    monkeypatch.setattr(dsl_utils, "attention", Mock(side_effect=AssertionError("DSL")))
    q = torch.empty(0, dtype=torch.float16)
    assert entry(q, q, q, **options) is backend.return_value
    arguments = dict(zip(inspect.signature(entry).parameters, backend.call_args.args))
    assert arguments["aux_tensors"] is arguments["aux_scalars"] is None


@pytest.mark.parametrize("entry", ENTRIES)
def test_optional_scales_signature_and_forwarding(monkeypatch, entry):
    parameters = dict(inspect.signature(entry).parameters)
    scales_parameter = parameters.pop("scales")
    assert scales_parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert scales_parameter.default is None
    assert not hasattr(fa4, f"{entry.__name__}_with_quant")

    backend = Mock(return_value=object())
    monkeypatch.setattr(dsl_utils, "attention", backend)
    native = legacy.FlashAttnFunc if entry is flash_attn_func else legacy.FlashAttnVarlenFunc
    monkeypatch.setattr(native, "apply", Mock(side_effect=AssertionError("native backend")))
    arguments = {name: object() for name in parameters}
    arguments["config"] = None
    # gather_bwd_* are not supported on NPU in the fixed-length interface; pass defaults.
    if "gather_bwd_recompute_p" in arguments:
        arguments["gather_bwd_recompute_p"] = False
    if "gather_bwd_token_chunk" in arguments:
        arguments["gather_bwd_token_chunk"] = None
    arguments.update({name: torch.empty(0, dtype=torch.float8_e4m3fn) for name in ("q", "k", "v")})
    scale = torch.empty(0, dtype=torch.uint8)
    scales = fa4.BlockScales(q=scale, k=scale, v=scale)

    qkv = tuple(arguments.pop(name) for name in ("q", "k", "v"))
    assert entry(*qkv, *arguments.values(), scales) is backend.return_value
    assert entry(*qkv, **arguments, scales=scales) is backend.return_value
    for call in backend.call_args_list:
        assert call.args == qkv
        forwarded = {
            name: value
            for name, value in arguments.items()
            if name not in ("config", "gather_bwd_recompute_p", "gather_bwd_token_chunk")
        }
        assert call.kwargs == dict(forwarded, scales=scales, varlen=entry is flash_attn_varlen_func)


@pytest.mark.parametrize("entry", ENTRIES)
def test_native_float_backward_argument_count_is_unchanged(monkeypatch, entry):
    q = torch.zeros((2, 7, 4, 128), dtype=torch.float16, requires_grad=True)
    k = torch.zeros((2, 9, 2, 128), dtype=q.dtype, requires_grad=True)
    v = k.detach().clone().requires_grad_()
    out = torch.zeros_like(q)
    monkeypatch.setattr(legacy, "get_scheduler_metadata", Mock(return_value=None))
    monkeypatch.setattr(
        legacy,
        "_flash_attn_forward",
        Mock(return_value=(out, torch.zeros((2, 4, 7)), torch.empty(0), torch.empty(0))),
    )
    monkeypatch.setattr(
        legacy,
        "_flash_attn_backward",
        Mock(return_value=tuple(torch.ones_like(t) for t in (q, k, v))),
    )
    entry(q, k, v).sum().backward()
    for t in (q, k, v):
        torch.testing.assert_close(t.grad, torch.ones_like(t))


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("missing", [None, "q", "all"])
def test_tensorwise_descales_reach_native_forward(monkeypatch, entry, missing):
    q = torch.zeros((2, 7, 4, 128), dtype=torch.float8_e4m3fn)
    k = torch.zeros((2, 9, 2, 128), dtype=q.dtype)
    values = [torch.full((2, 2), value) for value in (0.5, 0.75, 1.25)]
    if missing == "q":
        values[0] = None
    elif missing == "all":
        values = [None] * 3
    out = torch.zeros_like(q, dtype=torch.bfloat16)
    lse = torch.zeros((2, 4, 7))
    backend = Mock(return_value=(out, lse, torch.empty(0), torch.empty(0)))
    monkeypatch.setattr(legacy, "_flash_attn_forward", backend)
    monkeypatch.setattr(legacy, "get_scheduler_metadata", Mock(return_value=None))
    monkeypatch.setattr(dsl_utils, "attention", Mock(side_effect=AssertionError("Flex")))
    result = entry(q, k, k, scales=fa4.TensorwiseScales(*values), return_lse=True)
    assert result[0] is out
    assert len(result) == (2 if entry is flash_attn_func else 4)
    call = backend.call_args
    signature = inspect.signature(legacy._flash_attn_forward_fake)
    forwarded = signature.bind(*call.args, **call.kwargs).arguments
    for name, expected in zip(("q_descale", "k_descale", "v_descale"), values, strict=True):
        assert forwarded[name] is expected


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("name", TRIGGERS)
def test_tensorwise_rejects_flex_without_native_fallback(monkeypatch, entry, name):
    if name not in inspect.signature(entry).parameters:
        pytest.skip("No backward sparse parameter")
    native = legacy.FlashAttnFunc if entry is flash_attn_func else legacy.FlashAttnVarlenFunc
    backend = Mock(side_effect=AssertionError("native fallback"))
    monkeypatch.setattr(native, "apply", backend)
    monkeypatch.setattr(dsl_utils, "attention", Mock(side_effect=AssertionError("Flex launch")))
    with pytest.raises(NotImplementedError, match="Flex.*TensorwiseScales"):
        entry(None, None, None, scales=fa4.TensorwiseScales(), **{name: trigger(name)})
    backend.assert_not_called()


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("invalid", ["shape", "dtype", "grad", "member"])
def test_tensorwise_validation_before_launch(monkeypatch, entry, invalid):
    q = torch.empty((2, 7, 4, 128), dtype=torch.float8_e4m3fn)
    k = torch.empty((2, 9, 2, 128), dtype=q.dtype)
    scale = torch.ones((2, 2))
    if invalid == "shape":
        scale = scale[:1]
    elif invalid == "dtype":
        scale = scale.to(torch.float16)
    elif invalid == "grad":
        scale.requires_grad_()
    else:
        scale = 1.0
    native = legacy.FlashAttnFunc if entry is flash_attn_func else legacy.FlashAttnVarlenFunc
    backend = Mock(side_effect=AssertionError("native launch"))
    monkeypatch.setattr(native, "apply", backend)
    error = (
        NotImplementedError
        if invalid == "grad"
        else TypeError
        if invalid == "member"
        else ValueError
    )
    with pytest.raises(error):
        entry(q, k, k, scales=fa4.TensorwiseScales(q=scale))
    backend.assert_not_called()


@pytest.mark.npu
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("missing", [None, "q", "all"])
def test_tensorwise_native_precision(device, packed, missing):
    from tests.common.attention_ref_fp8 import read_inputs, ref_flash_attention_pair
    from tests.common.compare import assert_fa_close

    rng = torch.Generator().manual_seed(42)

    def shape(length, heads):
        return (2 * length, heads, 128) if packed else (2, length, heads, 128)

    q, k, v = [
        (torch.randn(shape(length, heads), generator=rng) * 0.2).to(torch.float8_e4m3fn)
        for length, heads in ((17, 4), (23, 2), (23, 2))
    ]
    base = torch.tensor([[1.0, 0.0, 0.5, 0.0], [0.75, 0.0, 1.25, 0.0]])
    scales = [(base * value)[:, ::2] for value in (0.5, 0.75, 1.25)]
    if missing == "q":
        scales[0] = None
    elif missing == "all":
        scales = [None] * 3
    options = dict(
        softmax_scale=128**-0.5, causal=True, window_size=(-1, -1), softcap=0.0, num_splits=1
    )
    if packed:
        options.update(
            cu_seqlens_q=torch.tensor([0, 17, 34], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 23, 46], dtype=torch.int32),
            max_seqlen_q=17,
            max_seqlen_k=23,
        )
    cpu = dict(
        q=q, k=k, v=v, **options, **dict(zip(("q_descale", "k_descale", "v_descale"), scales))
    )
    cpu["return_attn_probs"] = True
    case, inputs = read_inputs(cpu, "varlen" if packed else "dense")
    high, low = ref_flash_attention_pair(case, inputs)
    options = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in options.items()
    }
    entry = flash_attn_varlen_func if packed else flash_attn_func
    actual = entry(
        q.to(device),
        k.to(device),
        v.to(device),
        scales=fa4.TensorwiseScales(*(t.to(device) if t is not None else None for t in scales)),
        return_lse=True,
        **options,
    )
    out, lse = actual[:2]
    if not packed:
        out = out.reshape(-1, 4, 128)
        lse = lse.transpose(0, 1).reshape(4, -1)
    for name, result, reference, benchmark in zip(("O", "LSE"), (out, lse), high, low):
        assert_fa_close(result, reference, benchmark, softcap=0.0, name=name)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("invalid", [None, "qv", "q", "k", "v"])
def test_native_fp8_preserves_upstream_guards(monkeypatch, entry, invalid):
    backend = Mock(return_value=object())
    native = legacy.FlashAttnFunc if entry is flash_attn_func else legacy.FlashAttnVarlenFunc
    monkeypatch.setattr(native, "apply", backend)
    monkeypatch.setattr(dsl_utils, "attention", Mock(side_effect=AssertionError("CATLASS backend")))
    inputs = {
        name: torch.empty(0, dtype=torch.float8_e4m3fn, requires_grad=name == invalid)
        for name in ("q", "k", "v")
    }
    if invalid is None:
        assert entry(**inputs) is backend.return_value
        return
    if invalid == "qv":
        inputs["qv"] = torch.empty(0)
    message = "q/k/v inputs only" if invalid == "qv" else "inference forward only"
    with pytest.raises(NotImplementedError, match=message):
        entry(**inputs)
    backend.assert_not_called()


def test_combine_remains_public():
    assert "flash_attn_combine" in fa4.__all__
    assert fa4.flash_attn_combine is legacy.flash_attn_combine


def test_910_package_does_not_export_quantized_attention(monkeypatch):
    names = (
        "flash_attn_func",
        "flash_attn_varlen_func",
        "flash_attn_combine",
        "get_scheduler_metadata",
    )
    backend = SimpleNamespace(**{name: getattr(fa4, name) for name in names})
    monkeypatch.setitem(sys.modules, "flash_attn_npu_4.flash_attn_npu_interface", backend)
    monkeypatch.setattr(torch_npu.npu, "get_device_name", lambda: "Ascend910B")

    spec = importlib.util.spec_from_file_location(fa4.__name__, fa4.__file__)
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)

    assert package.flash_attn_func is backend.flash_attn_func
    assert package.flash_attn_varlen_func is backend.flash_attn_varlen_func
    for name in (
        "TensorwiseScales",
        "BlockScales",
        "flash_attn_func_with_quant",
        "flash_attn_varlen_func_with_quant",
    ):
        assert not hasattr(package, name)
        assert name not in package.__all__


def test_aux_container_validation():
    with pytest.raises(TypeError, match="aux_tensors"):
        flash_attn_func(None, None, None, aux_tensors=torch.ones(3))


@pytest.fixture(scope="module")
def entry_910():
    # Execute the actual public Python routing functions without loading a
    # second device extension/custom-op library into the same process.
    path = Path(legacy.__file__).with_name("flash_attn_npu_interface.py")
    module = ast.parse(path.read_text())
    module.body = [
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("flash_attn_func", "flash_attn_varlen_func")
    ]
    backend = SimpleNamespace(apply=Mock(return_value="legacy910"))
    namespace = dict(
        _catlass_utils=dsl_utils,
        torch=torch,
        Optional=Optional,
        Tuple=Tuple,
        Callable=Callable,
        FlashAttnFunc=backend,
        FlashAttnVarlenFunc=backend,
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("name", TRIGGERS)
@pytest.mark.parametrize("entry", ["flash_attn_func", "flash_attn_varlen_func"])
def test_910_rejects_flex(entry_910, entry, name):
    fn = entry_910[entry]
    if name not in inspect.signature(fn).parameters:
        pytest.skip("No backward sparse argument on varlen")
    with pytest.raises(NotImplementedError, match="Ascend 950.*Ascend 910"):
        fn(None, None, None, **{name: trigger(name)})
    assert fn(None, None, None, aux_tensors=[], aux_scalars=()) == "legacy910"


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize(
    "name,value",
    [
        ("scheduler_metadata", torch.empty(0)),
        ("disable_scheduler_metadata", True),
        ("num_splits", 2),
        ("pack_gqa", False),
        ("pack_gqa", True),
        ("deterministic", True),
        ("qv", torch.empty(0)),
        ("gather_kv_indices", torch.empty(0)),
        ("learnable_sink", torch.empty(0)),
        ("page_table", torch.empty(0)),
        ("seqused_q", torch.empty(0)),
        ("seqused_k", torch.empty(0)),
        ("min_seqlen_k", 1),
    ],
)
def test_unsupported_flex_parameters_fail_before_launch(monkeypatch, entry, name, value):
    if name not in inspect.signature(entry).parameters:
        pytest.skip("Parameter belongs to varlen only")
    monkeypatch.setattr(legacy.FlashAttnFunc, "apply", Mock(side_effect=AssertionError("legacy")))
    monkeypatch.setattr(
        legacy.FlashAttnVarlenFunc, "apply", Mock(side_effect=AssertionError("legacy"))
    )
    with pytest.raises(NotImplementedError, match=name):
        entry(None, None, None, aux_scalars=(1.0,), **{name: value})


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("num_splits", [0, 1])
def test_public_return_contract(cpu_runtime, entry, return_lse, training, num_splits):
    _, inputs, _, _, _ = make_inputs("cpu", torch.float16)
    with torch.set_grad_enabled(training):
        actual = entry(*inputs, aux_scalars=(1.0,), return_lse=return_lse, num_splits=num_splits)
    out = actual[0] if return_lse else actual
    assert isinstance(out, torch.Tensor) and out.shape == inputs[0].shape
    assert out.requires_grad == training
    if return_lse:
        assert len(actual) == (2 if entry is flash_attn_func else 4)
        assert actual[1].shape == (2, 4, 17)
        for empty in actual[2:]:
            assert empty.shape == (0,) and empty.dtype == torch.float32
            assert empty.device == out.device and empty.requires_grad == training
        if training and len(actual) == 4:
            grads = torch.autograd.grad(sum(t.sum() for t in actual[2:]), inputs)
            assert all(not g.count_nonzero() for g in grads)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("window", [(-1, 3), (3, -1), (-7, -2), (None, 0)])
def test_window_compatibility(cpu_runtime, entry, window):
    _, inputs, _, _, _ = make_inputs("cpu", torch.float16)
    normalized = tuple(None if w is None or w < 0 else w for w in window)
    actual = entry(*inputs, window_size=window, aux_scalars=(1.0,), return_lse=True)
    expected = catlass_dsl.flash_attn_func(*inputs, window_size=normalized, return_lse=True)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("loss", ["all", "lse", "accum"])
def test_public_fullgraph_abi(cpu_runtime, packed, loss):
    if loss == "accum" and not packed:
        pytest.skip("The fixed-length API has no accumulation outputs")
    mask = builtin.causal()

    def fn(q, k, v, cq, ck):
        if packed:
            return flash_attn_varlen_func(
                q, k, v, cu_seqlens_q=cq, cu_seqlens_k=ck, mask_mod=mask, return_lse=True
            )
        return flash_attn_func(q, k, v, mask_mod=mask, return_lse=True)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="aot_eager", fullgraph=True, dynamic=True)
    for length in (17, 29):
        _, inputs, cq, ck, _ = make_inputs("cpu", torch.float16, qp=packed, kp=packed, sq=length)
        result = compiled(*inputs, cq, ck)
        expected = fn(*inputs, cq, ck)
        assert len(result) == (4 if packed else 2)
        for a, b in zip(result, expected):
            torch.testing.assert_close(a, b)
        indices = range(len(result)) if loss == "all" else (1,) if loss == "lse" else (2, 3)
        outputs = tuple(result[i] for i in indices)
        references = tuple(expected[i] for i in indices)
        tangents = tuple(torch.randn_like(t) for t in outputs)
        actual_grads = torch.autograd.grad(outputs, inputs, tangents)
        wanted_grads = torch.autograd.grad(references, inputs, tangents)
        for a, b in zip(actual_grads, wanted_grads):
            torch.testing.assert_close(a, b)
        if loss == "accum":
            assert all(not g.count_nonzero() for g in actual_grads)
    torch._dynamo.reset()
