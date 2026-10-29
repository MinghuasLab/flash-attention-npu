"""CPU dual-reference regressions for the imported Opt scheduler and cross attention."""
from itertools import accumulate

import pytest
import torch
import torch_npu  # noqa: F401

from flash_attn_npu.flash_attn_npu_interface import flash_attn_varlen_func
from tests.common.compare import assert_fa_close
from tests.test_flash_attn_npu_v2_backward_cache import reference


def check_case(qlens, klens, heads, kv_heads, causal, dtype):
    torch.manual_seed(67120)
    cpu = [(torch.randn(total, count, 128) * .25).to(dtype).requires_grad_()
           for total, count in ((sum(qlens), heads), (sum(klens), kv_heads), (sum(klens), kv_heads))]
    xs = [x.detach().npu().requires_grad_() for x in cpu]
    cq = torch.tensor([0, *accumulate(qlens)], dtype=torch.int32, device='npu')
    ck = torch.tensor([0, *accumulate(klens)], dtype=torch.int32, device='npu')
    scale = 128 ** -.5 if causal else 1.
    out = flash_attn_varlen_func(*xs, cq, ck, max(1, max(qlens)), max(1, max(klens)),
                                causal=causal, softmax_scale=scale)
    dout = torch.randn_like(cpu[0]) * .25
    grads = torch.autograd.grad(out, xs, dout.npu())
    high = [x.detach().float().requires_grad_() for x in cpu]
    high_out = reference(*high, qlens, klens, causal, scale)
    low_out = reference(*cpu, qlens, klens, causal, scale)
    high_grad = torch.autograd.grad(high_out, high, dout.float())
    low_grad = torch.autograd.grad(low_out, cpu, dout)
    for name, actual, expected, low in zip(('out', 'dq', 'dk', 'dv'), (out, *grads),
                                         (high_out, *high_grad), (low_out, *low_grad)):
        if actual.numel():
            assert_fa_close(actual, expected.to(dtype), low, name=name)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('qlens,klens,heads,kv_heads,causal', [
    ([279, 257], [279, 257], 4, 4, True),
    ([279, 257], [279, 257], 4, 2, True),
    ([3, 0, 1], [276, 0, 257], 4, 4, False),
    ([3, 0, 1], [276, 0, 257], 4, 2, False),
    ([257, 1], [3, 1], 4, 4, False),
    ([513, 129], [513, 129], 2, 2, False),
])
def test_basic_opt_tile_boundaries(qlens, klens, heads, kv_heads, causal, dtype):
    check_case(qlens, klens, heads, kv_heads, causal, dtype)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_basic_opt_cache_distinguishes_total_k(dtype):
    # Same Q/B/H/D/scale key, different K totals and required gradient workspace.
    # Repeat the first key to cover both allocation and cache reuse.
    for klens in ([1, 3], [257, 276], [1, 3]):
        check_case([3, 3], klens, 2, 2, False, dtype)
