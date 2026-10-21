"""FA4 dispatch/ABI tests. CPU kernel substitution is explicit in cpu_runtime."""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Tuple
from unittest.mock import Mock

import pytest
import torch

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
    assert entry(None, None, None, **options) is backend.return_value
    arguments = dict(zip(inspect.signature(entry).parameters, backend.call_args.args))
    assert arguments["aux_tensors"] is arguments["aux_scalars"] is None


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
def test_public_fullgraph_abi(cpu_runtime, packed):
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
        tangents = tuple(torch.randn_like(t) for t in result)
        actual_grads = torch.autograd.grad(result, inputs, tangents)
        wanted_grads = torch.autograd.grad(expected, inputs, tangents)
        for a, b in zip(actual_grads, wanted_grads):
            torch.testing.assert_close(a, b)
    torch._dynamo.reset()
