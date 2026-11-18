"""Conservative lowering and ownership checks for separable mask helpers."""

import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import catlass.tla as tla
import pytest
import torch

from .test_utils import DTYPES, check_attention, copy_aux_to_device, make_inputs

from flash_attn_npu_4.catlass_dsl import compute_block_sparsity, flash_attn_func
from flash_attn_npu_4.catlass_dsl.builtin import pipeline_adapter
from flash_attn_npu_4.catlass_dsl.kernels import flash_fwd_arch35 as attention_kernel
from flash_attn_npu_4.catlass_dsl import mask as lowering
from flash_attn_npu_4.catlass_dsl.mods import MaskMod


@tla.jit
def integer_mask(b, h, q, k, info, aux, scalars):
    return (aux[0][q] == aux[1][k]) & (q >= k)


@tla.jit
def float_mask(b, h, q, k, info, aux, scalars):
    return aux[0][q].to(tla.Float32) * 0.5 >= aux[1][k].to(tla.Float32)


@tla.jit
def mixed_load_mask(b, h, q, k, info, aux, scalars):
    return aux[0][q + k] != 0


@tla.jit
def scalar_mask(b, h, q, k, info, aux, scalars):
    return (aux[0][q] == aux[1][k]) & (q >= scalars[0])


@tla.jit
def logical_integer_mask(b, h, q, k, info, aux, scalars):
    return (aux[0][q] and aux[1][k]) > 0


@tla.jit
def mixed_boolean_mask(b, h, q, k, info, aux, scalars):
    return (True & aux[0][q]) == aux[1][k]


@tla.jit
def integer_condition_mask(b, h, q, k, info, aux, scalars):
    visible = tla.Bool(False)
    if aux[0][q]:
        visible = aux[1][k] >= 0
    return visible


class _TensorOwner:
    def __init__(self, dtype=tla.Int32):
        self.ptr = SimpleNamespace(dtype=dtype)


_SPECS = (((None,), (1,), False, True),) * 2
_ADAPTED = {
    callback: pipeline_adapter(MaskMod(callback), "mask", _SPECS, "simt")
    for callback in (
        integer_mask,
        float_mask,
        mixed_load_mask,
        scalar_mask,
        integer_condition_mask,
        logical_integer_mask,
        mixed_boolean_mask,
    )
}


def _plan(callback=integer_mask, scalar=None):
    owners = (_TensorOwner(), _TensorOwner())
    adapted = _ADAPTED[callback]
    metadata = (tla.Int32(512), tla.Int32(1)) * 2
    scalars = metadata if scalar is None else (scalar, *metadata)
    plan = lowering.build(adapted, owners, scalars)
    return plan, tuple(weakref.ref(owner) for owner in owners)


def test_integer_plan_does_not_retain_binding_owners():
    enabled = gc.isenabled()
    gc.disable()
    try:
        plan, refs = _plan()
        assert plan is not None
        assert all(ref() is None for ref in refs)
        source = Path(plan.source).read_text()
        assert "tla.arange(0, dtype=tla.Int32)" in source
        assert "MaskStoreParams" in source
        assert "extern" not in source
        assert lowering._compile_source.cache_info().maxsize == 128
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize(
    "callback",
    [float_mask, mixed_load_mask, integer_condition_mask, logical_integer_mask, mixed_boolean_mask],
)
def test_unsupported_expression_uses_original_mask(callback):
    plan, refs = _plan(callback)
    assert plan is None
    assert all(ref() is None for ref in refs)


def test_scalar_width_is_preserved_or_rejected():
    assert _plan(scalar_mask, tla.Int32(7))[0] is not None
    assert _plan(scalar_mask, tla.Float32(0.5))[0] is None
    assert _plan(scalar_mask, tla.Bool(True))[0] is None
    assert _plan(scalar_mask, tla.Int64(2**40))[0] is None


def test_left_constant_preserves_wide_arithmetic():
    query = lowering.Expr("var", ("q",), "Int64")
    expression = lowering.expr("+", 1, query)
    assert expression.dtype == "Int64"
    assert all(a.dtype == "Int64" for a in expression.args)
    assert _evaluate(expression, 2**40, 0) == 2**40 + 1


def _evaluate(expr, q, k):
    if expr.op == "var":
        return {"q": q, "k": k}[expr.args[0]]
    if expr.op == "const":
        return expr.args[0]
    values = [_evaluate(a, q, k) for a in expr.args]
    if expr.op == "cast":
        value = values[0]
        return (value + 2**31) % 2**32 - 2**31 if expr.dtype == "Int32" else value
    return lowering.OPS[expr.op](*values)


@pytest.mark.parametrize("guarded", [False, True])
def test_comparison_narrowing_requires_independent_bounds(guarded):
    q = lowering.Expr("var", ("q",), "Int64")
    k = lowering.Expr("var", ("k",), "Int64")
    comparison = lowering.expr(">=", q, k)
    root = comparison
    if guarded:
        for value in (q, k):
            root = lowering.expr("&", root, lowering.expr(">=", value, 0))
            root = lowering.expr("&", root, lowering.expr("<", value, 1000))
    transformed = lowering.narrow_guarded_comparisons(root)
    assert (transformed != root) is guarded
    values = (-(2**40), -1, 0, 37, 999, 1000, 2**31 + 7, 2**32 + 37, 2**40)
    for query in values:
        for key in values:
            assert _evaluate(transformed, query, key) == _evaluate(root, query, key)


@tla.jit
def _mapped_mask(b, h, q, k, info, aux, scalars):
    query = aux[0][q]
    key = aux[1][k]
    length = tla.Int64(aux[2].shape[0])
    visible = tla.Bool(False)
    if (query >= tla.Int64(0)) & (key >= tla.Int64(0)) & (query < length) & (key < length):
        query_label = aux[2][query]
        key_label = aux[2][key]
        visible = (
            ((query >= key) | ((query_label >= 0) & (query_label == key_label)))
            & (aux[3][b, query] != 0)
            & (aux[3][b, key] != 0)
        )
    return visible


_mapped_mask_program = MaskMod(_mapped_mask)


@pytest.mark.npu
@pytest.mark.parametrize("dtype", DTYPES)
def test_mapped_mask_with_wide_indices_and_strided_aux(device, dtype, monkeypatch):
    # Compare with an independent Torch predicate, including output/LSE gradients.
    # Invalid wide indices must remain invalid after any guarded narrowing.
    batch, heads, sq, sk, domain = 2, 2, 513, 1031, 2049
    generator = torch.Generator().manual_seed(744)
    query_map = torch.randperm(domain, generator=generator)[:sq].to(torch.int64)
    key_map = torch.randperm(domain, generator=generator)[:sk].to(torch.int64)
    query_map[::137] = -1
    key_map[::149] = -3
    query_map[65] = 2**32 + 37
    key_map[129] = 2**40 + 9
    query_map[131] = 2**31 + 31
    key_map[261] = -(2**40)
    labels = (torch.arange(domain) // 173).to(torch.int32)
    labels[::113] = -1
    valid = (torch.rand(batch, domain * 2, generator=generator) > 0.2).to(torch.int32)[:, ::2]
    cpu_aux = (query_map, key_map, labels, valid)
    aux = tuple(copy_aux_to_device(tensor, device) for tensor in cpu_aux)
    assert aux[3].stride() == valid.stride()
    planned_sources = []
    original = attention_kernel._separable_plan

    def track_plan(*args):
        plan = original(*args)
        if plan is not None:
            planned_sources.append(plan.source)
        return plan

    monkeypatch.setattr(attention_kernel, "_separable_plan", track_plan)
    cpu, inputs, _, _, _ = make_inputs(device, dtype, sq=sq, sk=sk, heads=heads, kh=heads)
    blocks = compute_block_sparsity(
        128, 128, batch, heads, sq, sk, _mapped_mask_program, aux, device
    )
    out, lse = flash_attn_func(
        *inputs,
        mask_mod=_mapped_mask_program,
        aux_tensors=aux,
        block_sparse_tensors=blocks,
        return_lse=True,
    )
    assert planned_sources, "The test must exercise the optimized path, not only fallback"

    def reference(b, h, q, k, info, tensors, scalars):
        query, key = tensors[0][q], tensors[1][k]
        in_bounds = (query >= 0) & (key >= 0) & (query < domain) & (key < domain)
        qi, ki = query.clamp(0, domain - 1), key.clamp(0, domain - 1)
        visible = (query >= key) | ((tensors[2][qi] >= 0) & (tensors[2][qi] == tensors[2][ki]))
        return in_bounds & visible & (tensors[3][b, qi] != 0) & (tensors[3][b, ki] != 0)

    check_attention(cpu, inputs, out, lse, mask_mod=reference, aux_tensors=cpu_aux, loss="both")
