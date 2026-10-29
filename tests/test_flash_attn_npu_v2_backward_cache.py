"""Backward mask reuse preserves gradients across prefixes and streams."""
from itertools import accumulate

import pytest
import torch
import torch_npu  # noqa: F401

from flash_attn_npu.flash_attn_npu_interface import flash_attn_varlen_func
from tests.common.compare import assert_fa_close


def reference(q, k, v, qlens, klens, causal, scale):
    chunks = []
    qi = ki = 0
    group = q.shape[1] // k.shape[1]
    for nq, nk in zip(qlens, klens, strict=True):
        qb = q[qi:qi + nq]
        kb = k[ki:ki + nk].repeat_interleave(group, dim=1)
        vb = v[ki:ki + nk].repeat_interleave(group, dim=1)
        if nq == 0 or nk == 0:
            chunks.append(qb * 0)
        else:
            scores = torch.einsum('qhd,khd->hqk', qb, kb) * scale
            if causal:
                keep = torch.arange(nk)[None, :] <= torch.arange(nq)[:, None] + nk - nq
                valid = keep.any(dim=-1, keepdim=True)
                scores = scores.masked_fill(~keep[None], float('-inf'))
                scores = torch.where(valid[None], scores, 0.)
                probs = scores.softmax(-1) * valid[None]
            else:
                probs = scores.softmax(-1)
            chunks.append(torch.einsum('hqk,khd->qhd', probs, vb))
        qi += nq
        ki += nk
    return torch.cat(chunks) + 0 * (k.sum() + v.sum())


def check(inputs, cuq, cuk, qlens, klens, causal, scale):
    out = flash_attn_varlen_func(*inputs, cuq, cuk, 64, 64, causal=causal, softmax_scale=scale)
    dout = torch.randn_like(out)
    gradients = torch.autograd.grad(out, inputs, dout)
    cpu = [t.detach().cpu().float().requires_grad_() for t in inputs]
    expected = reference(*cpu, qlens, klens, causal, scale)
    ref_gradients = torch.autograd.grad(expected, cpu, dout.cpu().float())
    low = [t.detach().to(inputs[0].dtype).requires_grad_() for t in cpu]
    low_output = reference(*low, qlens, klens, causal, scale)
    low_gradients = torch.autograd.grad(low_output, low, dout.cpu())
    # Use the repository's FP32 / low-precision dual-reference criterion.
    for name, actual, target, rounded in zip(
        ('out', 'dq', 'dk', 'dv'), (out, *gradients), (expected, *ref_gradients),
        (low_output, *low_gradients), strict=True,
    ):
        assert_fa_close(actual, target.to(actual.dtype), rounded, name=name)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('kv_heads', [2, 4])
def test_general_backward_dynamic_prefix_and_stream(dtype, causal, kv_heads):
    torch.manual_seed(88)
    inputs = [torch.randn(t, h, 128, device='npu', dtype=dtype).requires_grad_()
              for t, h in ((22, 4), (28, kv_heads), (28, kv_heads))]
    cuq, cuk = [torch.empty(5, dtype=torch.int32, device='npu') for _ in range(2)]
    streams = [torch.npu.current_stream(), torch.npu.Stream()]
    # Same tensor addresses and totals, different values and empty segments.
    # Declared maxima are bounds (64), not the observed maxima (17/19).
    for stream, qlens, klens, scale in (
        (streams[0], [17, 0, 5, 0], [4, 9, 15, 0], 0.125),
        (streams[0], [0, 13, 5, 4], [1, 0, 8, 19], 0.125),
        (streams[1], [17, 0, 5, 0], [4, 9, 15, 0], 0.25),
        (streams[0], [0, 13, 5, 4], [1, 0, 8, 19], 0.25),
    ):
        torch.npu.synchronize()
        with torch.npu.stream(stream):
            cuq.copy_(torch.tensor([0, *accumulate(qlens)], dtype=torch.int32))
            cuk.copy_(torch.tensor([0, *accumulate(klens)], dtype=torch.int32))
            check(inputs, cuq, cuk, qlens, klens, causal, scale)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('alias', [False, True])
def test_optimized_backward_prefix_identity_and_mask_reuse(dtype, causal, alias):
    torch.manual_seed(89)
    inputs = [torch.randn(22, h, 128, device='npu', dtype=dtype).requires_grad_() for h in (4, 2, 2)]
    cuq = torch.empty(4, dtype=torch.int32, device='npu')
    cuk = cuq if alias else torch.empty_like(cuq)
    for lengths, scale in (([7, 6, 9], 0.125), ([11, 3, 8], 0.125), ([7, 6, 9], 0.25)):
        prefix = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32)
        cuq.copy_(prefix)
        if not alias:
            cuk.copy_(prefix)
        check(inputs, cuq, cuk, lengths, lengths, causal, scale)


def test_backward_fake_tensor_dispatch():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from flash_attn_npu import flash_attn_npu_interface as interface
    with FakeTensorMode():
        inputs = [torch.empty(16, 2, 128, dtype=torch.bfloat16, device='npu', requires_grad=True)
                  for _ in range(3)]
        prefix = torch.empty(3, dtype=torch.int32, device='npu')
        output = interface.flash_attn_varlen_func(*inputs, prefix, prefix, 8, 8, causal=True)
        grads = torch.autograd.grad(output, inputs, torch.empty_like(output))
        assert all(g.shape == x.shape for g, x in zip(grads, inputs, strict=True))


def test_backward_aot_eager_fullgraph():
    # Use the module entry to avoid conftest's Python timing wrapper.
    from flash_attn_npu import flash_attn_npu_interface as interface
    torch.manual_seed(90)
    inputs = [torch.randn(16, 2, 128, dtype=torch.bfloat16, device='npu', requires_grad=True)
              for _ in range(3)]
    prefix = torch.tensor([0, 7, 16], dtype=torch.int32, device='npu')
    def attention(q, k, v):
        return interface.flash_attn_varlen_func(q, k, v, prefix, prefix, 9, 9, causal=True)
    eager = attention(*inputs)
    dout = torch.randn_like(eager)
    expected = torch.autograd.grad(eager, inputs, dout)
    compiled = torch.compile(attention, backend='aot_eager', fullgraph=True)
    output = compiled(*inputs)
    actual = torch.autograd.grad(output, inputs, dout)
    for a, b in zip((output, *actual), (eager, *expected), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
