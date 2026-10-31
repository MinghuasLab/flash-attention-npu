"""Fixed attention, LSE losses, empty inputs and strided views."""

import pytest
import torch
from .test_utils import DTYPES, check, check_attention, make_inputs, oracle

from flash_attn_npu_4.catlass_dsl import flash_attn_func, flash_attn_varlen_func, runtime

pytestmark = pytest.mark.npu
ENTRIES = (flash_attn_func, flash_attn_varlen_func)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "sq,sk,kh,options",
    [
        (128, 128, 4, {}),
        (129, 143, 4, {"causal": True}),
        (17, 23, 1, {}),
        (256, 128, 2, {"window_size": (19, 31)}),
        (23, 17, 2, {"softcap": -2.0}),
        (128, 129, 2, {"softcap": 2.0, "causal": True}),
        (17, 23, 2, {"softmax_scale": 0.0}),
        (17, 23, 2, {"softmax_scale": -0.125}),
    ],
)
def test_flash_attn_output(device, dtype, sq, sk, kh, options):
    cpu, inputs, _, _, _ = make_inputs(device, dtype, sq=sq, sk=sk, kh=kh)
    out, lse = flash_attn_func(*inputs, **options, return_lse=True)
    reference = {"scale": options.get("softmax_scale")}
    if options.get("causal"):
        reference["mask_mod"] = lambda b, h, q, k, info, aux, scalars: k <= q + info[1] - info[0]
    if "window_size" in options:
        left, right = options["window_size"]
        reference["mask_mod"] = lambda b, h, q, k, info, aux, scalars: (
            (k >= q + info[1] - info[0] - left) & (k <= q + info[1] - info[0] + right)
        )
    if "softcap" in options:
        cap = options["softcap"]
        reference["score_mod"] = lambda s, b, h, q, k: cap * torch.tanh(s / cap)
    check_attention(cpu, inputs, out, lse, **reference)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("loss", ["out", "lse", "both"])
def test_flash_attn_lse_grad(device, entry, loss):
    cpu, inputs, _, _, _ = make_inputs(device, torch.float16)
    out, lse = entry(*inputs, return_lse=True)
    check_attention(cpu, inputs, out, lse, loss=loss)


@pytest.mark.npu
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("sq,sk,batch", [(0, 7, 2), (7, 0, 2), (0, 0, 2), (7, 7, 0)])
def test_npu_empty_inputs(device, packed, sq, sk, batch):
    shapes = [(batch * s, 2, 128) if packed else (batch, s, 2, 128) for s in (sq, sk, sk)]
    inputs = [
        torch.zeros(s, dtype=torch.float16, device=device, requires_grad=True) for s in shapes
    ]
    cq = torch.arange(batch + 1, dtype=torch.int32, device=device) * sq if packed else None
    ck = torch.arange(batch + 1, dtype=torch.int32, device=device) * sk if packed else None
    out, lse = flash_attn_varlen_func(*inputs, cu_seqlens_q=cq, cu_seqlens_k=ck, return_lse=True)
    assert not out.count_nonzero() and torch.isneginf(lse).all()
    for grad in torch.autograd.grad(
        (out, lse), inputs, (torch.ones_like(out), torch.ones_like(lse))
    ):
        assert not grad.count_nonzero()


@pytest.mark.npu
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("kind", ["transpose", "slice", "expand", "offset"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_noncontiguous_gradients(device, packed, kind, dtype):
    shape = (9, 2, 128) if packed else (2, 7, 2, 128)
    size = 9 * 2 * 128 if packed else 2 * 7 * 2 * 128
    generator = torch.Generator().manual_seed(903)
    cpu = [
        (torch.randn(size + 1, generator=generator) * 0.2).to(dtype).requires_grad_()
        for _ in range(3)
    ]
    inputs = [t.detach().to(device).requires_grad_() for t in cpu]

    def view(t):
        if kind == "transpose":
            return (
                t[:-1].view(2, 9, 128).transpose(0, 1)
                if packed
                else t[:-1].view(2, 2, 7, 128).transpose(1, 2)
            )
        if kind == "slice":
            return t[:-1].view(shape)[::2] if packed else t[:-1].view(shape)[:, ::2]
        if kind == "expand":
            return t[:128].view((1,) * (len(shape) - 1) + (128,)).expand(shape)
        return t[1:].view(shape)

    qkv, refs = [view(t) for t in inputs], [view(t) for t in cpu]
    total = qkv[0].shape[0]
    cu = torch.tensor([0, total // 2, total], dtype=torch.int32) if packed else None
    kwargs = dict(cu_seqlens_q=cu.to(device), cu_seqlens_k=cu.to(device)) if packed else {}
    entry = flash_attn_varlen_func if packed else flash_attn_func
    out, lse = entry(*qkv, **kwargs, return_lse=True)
    wanted, el = oracle(refs, cu_q=cu, cu_k=cu)
    grads = torch.autograd.grad((out, lse), inputs, (torch.ones_like(out), torch.ones_like(lse)))
    expected = torch.autograd.grad(
        (wanted, el), cpu, (torch.ones_like(wanted), torch.ones_like(el))
    )
    for a, b in zip((out, lse, *grads), (wanted, el, *expected)):
        check(a, b)


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("window", [(-1, -1), (-2, -3), (-1, 3), (3, -1), (None, -1)])
def test_npu_negative_windows(device, dtype, window):
    cpu, inputs, _, _, _ = make_inputs(device, dtype, kh=4)
    left, right = window
    if left is not None and right is not None and left < 0 and right < 0:
        left = right = None

    def mask(b, h, q, k, i, a, c):
        center = q + i[1] - i[0]
        return ((center - k <= left) if left is not None else True) & (
            (k - center <= right) if right is not None else True
        )

    out, lse = flash_attn_func(*inputs, window_size=window, return_lse=True)
    wanted, el = oracle(cpu, mask_mod=mask)
    grads = torch.autograd.grad((out, lse), inputs, (torch.ones_like(out), torch.ones_like(lse)))
    expected = torch.autograd.grad(
        (wanted, el), cpu, (torch.ones_like(wanted), torch.ones_like(el))
    )
    for a, b in zip((out, lse, *grads), (wanted, el, *expected)):
        check(a, b)


@pytest.mark.npu
@pytest.mark.parametrize("entry", ENTRIES)
def test_npu_lse_tangent_rebinding_and_unused(device, entry):
    cpu, inputs, _, _, _ = make_inputs(device, torch.float16, kh=4)
    gradients = []
    for requested in (False, True):
        out, _ = entry(*inputs, return_lse=requested)
        gradients.append(torch.autograd.grad(out, inputs, torch.ones_like(out)))
    assert all(torch.equal(a, b) for a, b in zip(*gradients))
    cached = None
    for weight in (0.1, 0.7):
        out, lse = entry(*inputs, return_lse=True)
        tangent = torch.full_like(lse, weight)
        actual = torch.autograd.grad(lse, inputs, tangent)
        _, el = oracle(cpu)
        expected = torch.autograd.grad(el, cpu, torch.full_like(el, weight))
        for a, b in zip(actual, expected):
            check(a, b)
        if cached is not None:
            assert set(runtime.compile_cache) == cached
        cached = set(runtime.compile_cache)
    _, lse = entry(*inputs, return_lse=False)
    for grad in torch.autograd.grad(lse, inputs, torch.ones_like(lse)):
        assert not grad.count_nonzero()
