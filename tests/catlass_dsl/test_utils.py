"""Public BSND/TND inputs and independent Torch output/LSE/gradient checks."""

from types import SimpleNamespace
import math

import torch
from .reference import reference_attention

from flash_attn_npu_4.catlass_dsl import flash_attn_varlen_func

DTYPES = (torch.float16, torch.bfloat16)
LAYOUTS = ((False, False), (True, False), (False, True), (True, True))
LAYOUT_IDS = ("bsnd-bsnd", "tnd-bsnd", "bsnd-tnd", "tnd-tnd")

# CANN opbase mixed_tolerance_standard.md, section 2.2: (atol=rtol, fixed limit).
# Select by the actual output dtype before promoting values for comparison.
_MIXED_TOLERANCES = {
    torch.float16: (2**-10, 0.1),
    torch.bfloat16: (2**-7, 0.1),
    torch.float32: (2**-13, 0.01),
}


def aux_for(name, batch=2, heads=4, length=257):
    if name == "document":
        return (
            torch.arange(length)
            .div(73, rounding_mode="floor")
            .expand(batch, heads, -1)
            .contiguous()
            .int(),
        )
    if name == "global_packed_doc":
        return tuple(
            torch.arange(batch * length).div(73, rounding_mode="floor").int() for _ in range(2)
        )
    if name == "packed_aux":
        return (
            torch.randint(
                0,
                2**32 - 1,
                (batch, length, (length + 31) // 32),
                dtype=torch.int64,
                generator=torch.Generator().manual_seed(818),
            ).to(torch.uint32),
        )
    if name in ("ima", "global_ima", "global_causal_window"):
        return (torch.zeros(batch * length, dtype=torch.int32),)
    if name == "batch_bias":
        return (torch.linspace(-0.1, 0.1, batch),)
    if name == "dual_buffer":
        return (torch.linspace(-0.1, 0.1, heads), torch.linspace(-0.1, 0.1, length))
    if name == "global_q_and_kv_bias":
        return tuple(torch.linspace(-0.1, 0.1, batch * length) for _ in range(2))
    if name == "stress_multi_buffer":
        return (
            torch.ones(batch) * 0.01,
            torch.ones(heads) * 0.8,
            torch.linspace(-0.1, 0.1, batch * length),
            torch.linspace(0.1, -0.1, batch * length),
            torch.ones(1025) * 0.03,
        )
    if name in (
        "scaled_squared",
        "global_kv_bias",
        "global_q_bias",
        "global_rel_plus_kv_bias",
        "global_logical_rel_plus_kv_bias",
        "stress_complex_arithmetic",
        "stress_conditional_mask",
        "stress_global_offset",
        "stress_xor_pattern",
    ):
        return (torch.linspace(0.05, 0.15, batch * length),)
    return ()


def check(actual, expected):
    """CANN mixed tolerance: 99% matched and a dynamic maximum-error bound.

    https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
    ULP is evaluated at the narrowed golden value at the maximum finite-error
    point, including the subnormal floor and the largest finite binade.
    """
    dtype = actual.dtype
    tolerance, fixed_limit = _MIXED_TOLERANCES[dtype]
    actual = actual.detach().cpu().double()
    expected = expected.detach().cpu().double()
    assert actual.shape == expected.shape
    assert not torch.isnan(actual).any() and not torch.isnan(expected).any()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    matching_inf = torch.isinf(actual) & (actual == expected.to(dtype).double())
    assert (finite | matching_inf).all(), (
        "nonfinite values must match after narrowing the golden to the output dtype"
    )
    if actual.numel() == 0:
        return
    golden = expected[finite]
    error = (actual[finite] - golden).abs()
    matched = int((error <= tolerance + tolerance * golden.abs()).count_nonzero()) + int(
        matching_inf.count_nonzero()
    )
    ratio = matched / actual.numel()
    assert ratio >= 0.99, {
        "dtype": dtype,
        "matched_ratio": ratio,
        "atol": tolerance,
        "rtol": tolerance,
    }
    if error.numel():
        index = error.argmax()
        anchor = abs(float(golden[index].to(dtype)))
        limit = fixed_limit
        if math.isfinite(anchor):
            info = torch.finfo(dtype)
            # frexp returns one plus the binade exponent. Clamp subnormals to
            # the smallest normal exponent; unlike nextafter(max, inf), this
            # keeps ULP finite at the dtype's largest representable value.
            exponent = math.frexp(max(anchor, info.tiny))[1] - 1
            limit = max(limit, 32 * math.ldexp(info.eps, exponent))
        maximum = float(error[index])
        assert maximum <= limit, {
            "dtype": dtype,
            "max_abs_error": maximum,
            "max_abs_error_limit": limit,
        }


def copy_aux_to_device(value, device):
    """Preserve the entire backing storage, logical strides and storage offset.

    In particular, a zero stride may repeat a nonconstant slice. Filling the
    storage with one scalar would silently change such a test input.
    """
    size = value.untyped_storage().nbytes() // value.element_size()
    storage = torch.empty(0, dtype=value.dtype).set_(value.untyped_storage(), 0, (size,), (1,))
    return storage.to(device).as_strided(value.shape, value.stride(), value.storage_offset())


class _MetadataNPU(torch.Tensor):
    """CPU storage with NPU metadata for host validation only; never launched."""

    @property
    def device(self):
        return SimpleNamespace(type="npu", index=0)


def metadata_tensor(shape, dtype=torch.float16):
    return torch.empty(shape, dtype=dtype).as_subclass(_MetadataNPU)


def make_inputs(
    device,
    dtype,
    *,
    qp=False,
    kp=False,
    sq=17,
    sk=23,
    heads=4,
    kh=2,
    q_lengths=None,
    k_lengths=None,
    seed=1909,
):
    q_lengths = (sq - 3, sq) if q_lengths is None else q_lengths
    k_lengths = (sk, sk - 5) if k_lengths is None else k_lengths
    assert len(q_lengths) == len(k_lengths)
    batch = len(q_lengths)
    cq = (
        torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32)
        if qp
        else None
    )
    ck = (
        torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], dtype=torch.int32)
        if kp
        else None
    )
    qs = (sum(q_lengths), heads, 128) if qp else (batch, sq, heads, 128)
    ks = (sum(k_lengths), kh, 128) if kp else (batch, sk, kh, 128)
    generator = torch.Generator().manual_seed(seed)
    cpu = [
        (torch.randn(shape, generator=generator) * 0.2).to(dtype).requires_grad_()
        for shape in (qs, ks, ks)
    ]
    inputs = [t.detach().to(device).requires_grad_() for t in cpu]
    kwargs = dict(
        cu_seqlens_q=None if cq is None else cq.to(device),
        cu_seqlens_k=None if ck is None else ck.to(device),
    )
    return cpu, inputs, cq, ck, kwargs


def oracle(inputs, *, cu_q=None, cu_k=None, **kwargs):
    """Adapt public layouts around the independent, BNSD-based Torch oracle."""
    q = inputs[0]
    tensors = [t.transpose(1, 2) if t.ndim == 4 else t for t in inputs]
    out, lse = reference_attention(
        *tensors, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, return_lse=True, **kwargs
    )
    return (out.transpose(1, 2) if q.ndim == 4 else out), (lse if q.ndim == 4 else lse.T)


def check_attention(cpu, inputs, out, lse, *, cq=None, ck=None, loss="out", **kwargs):
    """Check output, LSE and all Q/K/V adjoints against Torch autograd."""
    expected, el = oracle(cpu, cu_q=cq, cu_k=ck, **kwargs)
    assert out.dtype == inputs[0].dtype and out.device == inputs[0].device
    assert lse.dtype == torch.float32 and lse.is_contiguous()
    generator = torch.Generator().manual_seed(2902)
    dout = torch.randn(expected.shape, generator=generator).to(expected.dtype) * 0.2
    dlse = torch.randn(el.shape, generator=generator) * 0.2
    dout = torch.zeros_like(dout) if loss == "lse" else dout
    dlse = torch.zeros_like(dlse) if loss == "out" else torch.where(torch.isfinite(el), dlse, 0.0)
    targets = (lse,) if loss == "lse" else (out,) if loss == "out" else (out, lse)
    tangents = (dlse,) if loss == "lse" else (dout,) if loss == "out" else (dout, dlse)
    actual_grads = torch.autograd.grad(targets, inputs, tuple(t.to(out.device) for t in tangents))
    expected_grads = torch.autograd.grad((expected, el), cpu, (dout, dlse))
    for actual, wanted in zip((out, lse, *actual_grads), (expected, el, *expected_grads)):
        check(actual, wanted)
    if loss == "lse":
        assert not actual_grads[2].count_nonzero()
    return actual_grads


def run(
    device,
    dtype,
    *,
    score=None,
    bwd=None,
    mask=None,
    aux=(),
    scalars=(),
    qpacked=False,
    kpacked=False,
    heads=4,
    kh=2,
    sq=17,
    sk=23,
):
    """Run one callback case through the public API, without implicit sparse construction."""
    cpu, inputs, cq, ck, lengths = make_inputs(
        device, dtype, qp=qpacked, kp=kpacked, heads=heads, kh=kh, sq=sq, sk=sk
    )
    kwargs = dict(
        **lengths,
        score_mod=score,
        score_mod_bwd=bwd,
        mask_mod=mask,
        aux_tensors=tuple(copy_aux_to_device(t, device) for t in aux),
        aux_scalars=scalars,
        return_lse=True,
    )
    out, lse = flash_attn_varlen_func(*inputs, **kwargs)
    check_attention(
        cpu,
        inputs,
        out,
        lse,
        cq=cq,
        ck=ck,
        score_mod=score,
        mask_mod=mask,
        aux_tensors=aux,
        aux_scalars=scalars,
    )
    return inputs, kwargs
