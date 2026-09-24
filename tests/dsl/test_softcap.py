"""Softcap math and full-block execution through the public forward API."""

from importlib.util import find_spec
import os

import pytest

if find_spec("catlass") is None:
    pytest.skip("CATLASS DSL runtime is not installed", allow_module_level=True)
if find_spec("torch") is None:
    pytest.skip("PyTorch is not installed", allow_module_level=True)

import catlass.tla as tla
import torch

from flash_attn_npu_dsl import compute_block_sparsity, flash_attn_func, simd


def scalar_mask(b, h, q, kv, info, tensors):
    return (kv < 128) | (kv == 256)


@simd
def vector_mask(b, h, q, kv, info, tensors):
    return tla.bitwise_or(tla.cmp(kv, 128, "lt"), tla.cmp(kv, 256, "eq"))


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"),
    reason="set FLASH_ATTN_DSL_TEST_DEVICE to opt into NPU execution",
)
@pytest.mark.parametrize(
    "dtype,dim,cap,mask",
    [
        (torch.float16, 8, 50.0, None),
        (torch.bfloat16, 32, 2.0, scalar_mask),
        (torch.float16, 80, 2.0, vector_mask),
    ],
)
def test_softcap_near_zero_saturated_scores_and_full_tail_blocks(dtype, dim, cap, mask):
    import torch_npu  # noqa: F401

    device = int(os.environ["FLASH_ATTN_DSL_TEST_DEVICE"])
    torch.npu.set_device(device)
    q = torch.zeros(1, 7, 3, dim, dtype=dtype)
    k = torch.zeros(1, 257, 1, dim, dtype=dtype)
    q[..., 0] = torch.tensor([-80, -2, -0.001, 0, 0.001, 2, 80]).reshape(1, 7, 1)
    k[..., 0] = ((torch.arange(257) % 3) - 1).reshape(1, 257, 1)
    v = torch.randn(k.shape, generator=torch.Generator().manual_seed(17)).to(dtype)
    scores = q.float().permute(0, 2, 1, 3) @ k.float().permute(0, 2, 3, 1)
    scores = cap * torch.tanh(scores / cap)
    if mask is not None:
        keep = (torch.arange(257) < 128) | (torch.arange(257) == 256)
        scores.masked_fill_(~keep, -torch.inf)
    expected_lse = scores.logsumexp(-1)
    expected = (scores.softmax(-1) @ v.float().permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    inputs = tuple(t.to(f"npu:{device}") for t in (q, k, v))
    schedules = [None]
    if mask is not None:
        # The first block and one-element tail are full. Both must still cap scores.
        schedules.append(
            compute_block_sparsity(128, 128, 1, 3, 7, 257, mask, None, inputs[0].device)
        )
    for blocks in schedules:
        for return_lse in (False, True):
            out, lse = flash_attn_func(
                *inputs,
                softmax_scale=1.0,
                softcap=cap,
                mask_mod=mask,
                block_sparse_tensors=blocks,
                return_lse=return_lse,
            )
            torch.npu.synchronize()
            torch.testing.assert_close(out.cpu().float(), expected, rtol=0, atol=0.05)
            assert (lse is not None) == return_lse
            if return_lse:
                torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0, atol=0.05)
