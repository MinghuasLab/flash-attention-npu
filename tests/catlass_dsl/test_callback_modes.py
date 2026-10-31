"""Callback mode decorators, native dispatch and independent device VJPs."""

import inspect
from dataclasses import FrozenInstanceError

import catlass.tla as tla
import pytest
import torch
from .test_utils import DTYPES, check, check_attention, make_inputs

from flash_attn_npu_4.catlass_dsl import (
    compute_block_sparsity,
    flash_attn_func,
    flash_attn_varlen_func,
    simd,
    simt,
)
from flash_attn_npu_4.catlass_dsl.callbacks import prepare_mod
from flash_attn_npu_4.catlass_dsl.mods import UnsupportedModError, callback_mode, normalize_mod
from .reference import reference_attention


@tla.jit
def _default_mask(b, h, q, k, info, aux, scalars):
    center = q + info.seqlen_k - info.seqlen_q
    return (k <= center) & (q != 0)


@simt
@tla.jit
def _scalar_mask(b, h, q, k, info, aux, scalars):
    center = q + info.seqlen_k - info.seqlen_q
    return (k <= center) & (q != 0)


@simd
@tla.jit
def _vector_mask(b, h, q, k, info, aux, scalars):
    center = tla.add(q, info.seqlen_k - info.seqlen_q)
    return tla.bitwise_and(tla.cmp(k, center, mode="le"), tla.cmp(q, 0, mode="ne"))


@simd(simt=_scalar_mask)
@tla.jit
def _paired_mask(b, h, q, k, info, aux, scalars):
    center = tla.add(q, info.seqlen_k - info.seqlen_q)
    return tla.bitwise_and(tla.cmp(k, center, mode="le"), tla.cmp(q, 0, mode="ne"))


@simt(simd=_vector_mask)
@tla.jit
def _scalar_paired_mask(b, h, q, k, info, aux, scalars):
    center = q + info.seqlen_k - info.seqlen_q
    return (k <= center) & (q != 0)


@simt
@tla.jit
def _scalar_score(s, b, h, *, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
    return s * s * aux_scalars[0] + s * 0.7


@simd
@tla.jit
def _vector_score(s, b, h, *, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
    return tla.add(tla.mul(tla.mul(s, s), aux_scalars[0]), tla.mul(s, 0.7))


@simt
@tla.jit
def _scalar_bwd(g, s, b, h, q, k, info, aux, scalars):
    return g * (s * (2.0 * scalars[0]) + 0.7)


@simd
@tla.jit
def _vector_bwd(g, s, b, h, q, k, info, aux, scalars):
    return tla.mul(g, tla.add(tla.mul(s, 2.0 * scalars[0]), 0.7))


@pytest.mark.parametrize(
    "mask", [_default_mask, _scalar_mask, _vector_mask, _paired_mask, _scalar_paired_mask]
)
def test_mode_binding_and_cache(mask):
    mod = prepare_mod(mask, "mask", True)
    assert prepare_mod(mask, "mask", True) is mod
    assert (mod.simt is not None) == (mask is not _vector_mask)
    assert (mod.simd is not None) == (mask not in (_default_mask, _scalar_mask))
    assert callback_mode(mod) == ("simt" if mod.simd is None else "simd")
    if mask is not _default_mask:
        assert inspect.signature(mask) == inspect.signature(mask.__wrapped__)
        assert getattr(mask.__wrapped__, "_tla_jit", False)
        assert normalize_mod(mask, "mask").simt is prepare_mod(mask, "mask").simt
        with pytest.raises(FrozenInstanceError):
            mask.mode = "simt"


@pytest.mark.parametrize("decorate", [simt, simd, simt(), simd()])
def test_decorator_preserves_signature_and_host_call(decorate, monkeypatch):
    @tla.jit
    def callback(x, *, y=2):
        return x + y

    def unexpected_jit(*args, **kwargs):
        pytest.fail("mode markers must not invoke tla.jit")

    monkeypatch.setattr(tla, "jit", unexpected_jit)
    decorated = decorate(callback)
    assert decorated(3, y=4) == 7
    assert inspect.signature(decorated) == inspect.signature(callback)
    assert decorated.__wrapped__ is callback


@pytest.mark.parametrize("decorate", [simt, simd])
def test_invalid_callback_modes_fail_at_binding(decorate):
    with pytest.raises(TypeError, match="callable"):
        decorate(3)
    with pytest.raises(TypeError, match="outside @tla.jit"):
        decorate(lambda *args: True)
    with pytest.raises(TypeError, match="do not stack"):
        decorate(_scalar_mask)
    with pytest.raises(TypeError, match="do not stack"):
        decorate(_vector_mask)


def test_pair_binding_rejects_missing_mode_and_duplicate_vector():
    with pytest.raises(TypeError, match="no SIMT"):
        simd(simt=_vector_mask)(_vector_mask.__wrapped__)
    with pytest.raises(TypeError, match="no SIMD"):
        simt(simd=_scalar_mask)(_scalar_mask.__wrapped__)
    with pytest.raises(ValueError, match="both"):
        prepare_mod(_vector_mask, "mask", simd=_vector_mask)


def test_mixed_mode_requires_explicit_equivalent_implementation():
    scalar = prepare_mod(_scalar_mask, "mask")
    vector = prepare_mod(_vector_score, "score", True)
    paired = prepare_mod(_paired_mask, "mask")
    assert callback_mode(paired, scalar) == "simt"
    assert callback_mode(paired, vector) == "simd"
    with pytest.raises(UnsupportedModError, match="common implementation"):
        callback_mode(scalar, vector)
    # Fail before entering device validation/compilation, including backward.
    q = torch.empty(1, 7, 2, 128, requires_grad=True)
    with pytest.raises(UnsupportedModError, match="common implementation"):
        flash_attn_func(
            q, q, q, mask_mod=_vector_mask, score_mod=_scalar_score, score_mod_bwd=_scalar_bwd
        )
    with pytest.raises(UnsupportedModError, match="common implementation"):
        flash_attn_func(
            q, q, q, mask_mod=_vector_mask, score_mod=_vector_score, score_mod_bwd=_scalar_bwd
        )


def _mask_ref(b, h, q, k, info, aux, scalars):
    return (k <= q + info[1] - info[0]) & (q != 0)


def test_sparse_builder_requires_npu():
    with pytest.raises(ValueError, match="requires an NPU"):
        compute_block_sparsity(
            128, 128, 1, 2, 137, 143, _vector_mask, aux_tensors=None, device="cpu"
        )


def test_custom_score_requires_explicit_vjp():
    q = torch.empty(1, 7, 2, 128, requires_grad=True)
    with pytest.raises(ValueError, match="score_mod_bwd is required"):
        flash_attn_func(q, q, q, score_mod=_scalar_score, aux_scalars=(0.3,))
    with pytest.raises(ValueError, match="requires score_mod"):
        flash_attn_func(q, q, q, score_mod_bwd=_scalar_bwd)


_MODES = {
    "default": (_default_mask, _scalar_score, _scalar_bwd),
    "simt": (_scalar_mask, _scalar_score, _scalar_bwd),
    "simd": (_vector_mask, _vector_score, _vector_bwd),
    "paired-simt": (_paired_mask, _scalar_score, _scalar_bwd),
    "paired-simd": (_scalar_paired_mask, _vector_score, _vector_bwd),
}


@pytest.mark.npu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "mode,qp,kp,custom_score",
    [
        ("default", False, False, False),
        ("simt", True, True, True),
        ("simd", False, False, True),
        ("simd", True, False, False),
        ("paired-simd", False, True, True),
        ("paired-simt", True, True, True),
    ],
)
def test_decorated_callbacks_device_vjp(device, dtype, mode, qp, kp, custom_score):
    sq, sk, kh = (128, 128, 4) if mode == "simd" and not qp and not kp else (23, 17, 2)
    cpu, inputs, cq, ck, kwargs = make_inputs(device, dtype, qp=qp, kp=kp, sq=sq, sk=sk, kh=kh)
    mask, score, backward = _MODES[mode]
    entry = flash_attn_varlen_func if qp or kp else flash_attn_func
    out, lse = entry(
        *inputs,
        **(kwargs if qp or kp else {}),
        mask_mod=mask,
        score_mod=score if custom_score else None,
        score_mod_bwd=backward if custom_score else None,
        aux_scalars=(0.3,),
        causal=True,
        window_size=(0, 0),
        return_lse=True,
    )
    reference = (lambda s, b, h, q, k: s.square() * 0.3 + s * 0.7) if custom_score else None
    check_attention(
        cpu, inputs, out, lse, cq=cq, ck=ck, mask_mod=_mask_ref, score_mod=reference, loss="both"
    )


@pytest.mark.npu
@pytest.mark.parametrize("mask", [_vector_mask, _paired_mask])
def test_decorated_sparse_mask_device_builder_and_attention(device, mask):
    generator = torch.Generator().manual_seed(4102)
    cpu = [
        (torch.randn(s, generator=generator) * 0.2).half().requires_grad_()
        for s in ((1, 137, 4, 128), (1, 143, 2, 128), (1, 143, 2, 128))
    ]
    inputs = [t.detach().to(device).requires_grad_() for t in cpu]
    builder = _scalar_mask if mask is _vector_mask else mask
    blocks = compute_block_sparsity(
        128, 128, 1, 4, 137, 143, builder, aux_tensors=None, device=device
    )
    out, lse = flash_attn_func(*inputs, mask_mod=mask, block_sparse_tensors=blocks, return_lse=True)
    expected, logsum = reference_attention(
        *(t.transpose(1, 2) for t in cpu), mask_mod=_mask_ref, return_lse=True
    )
    expected = expected.transpose(1, 2)
    grads = torch.autograd.grad(lse, inputs, torch.ones_like(lse))
    expected_grads = torch.autograd.grad(logsum, cpu, torch.ones_like(logsum))
    for actual, wanted in zip((out, lse, *grads), (expected, logsum, *expected_grads)):
        check(actual, wanted)
    assert not grads[2].count_nonzero()


@pytest.mark.npu
def test_simd_only_mask_device_builder_requires_scalar_variant(device):
    with pytest.raises(UnsupportedModError, match="requires a SIMT implementation"):
        compute_block_sparsity(
            128, 128, 1, 2, 17, 19, _vector_mask, aux_tensors=None, device=device
        )
