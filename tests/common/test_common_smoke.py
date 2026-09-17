# Copyright (c) 2026, Minghua Shen.
# Basic tests for the shared attention reference implementation.
# These CPU tests cover batched computation, GQA, masks, sinks, and backward.
# The baseline uses independent PyTorch operators so it does not share the
# implementation logic under test.

import math

import torch

from tests.common.attention_ref import ref_flash_attention, ref_flash_attention_pair
from tests.common.compare import assert_fa_close


def make_inputs(batch, q_len, kv_len, num_heads, kv_heads, head_size, *, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query = torch.randn(
        batch, q_len, num_heads, head_size, generator=generator, dtype=torch.float32
    )
    key = torch.randn(
        batch, kv_len, kv_heads, head_size, generator=generator, dtype=torch.float32
    )
    value = torch.randn(
        batch, kv_len, kv_heads, head_size, generator=generator, dtype=torch.float32
    )
    return query, key, value


def independent_attention(query, key, value, scale, mask=None, softcap=0.0, sink=None):
    """Compute an independent attention baseline with simple PyTorch operators."""
    batch, q_len, num_heads, head_size = query.shape
    kv_heads = key.shape[2]
    assert num_heads % kv_heads == 0
    group_size = num_heads // kv_heads

    key = key.repeat_interleave(group_size, dim=2)
    value = value.repeat_interleave(group_size, dim=2)
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 3, 1)
    v = value.permute(0, 2, 1, 3)
    scores = torch.matmul(q, k) * scale
    if softcap > 0.0:
        scores = torch.tanh(scores / softcap) * softcap

    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        scores = scores.masked_fill(mask[:, None].to(torch.bool), -1e4)

    if sink is None:
        lse = torch.logsumexp(scores.float(), dim=-1)
        probability = torch.softmax(scores, dim=-1)
    else:
        sink = sink.to(device=scores.device, dtype=scores.dtype)
        if sink.dim() == 3:
            sink = sink.unsqueeze(0)
        assert sink.shape == (batch, num_heads, q_len, 1)
        row_max = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
        row_max_high = row_max.to(torch.float64)
        score_exp = torch.exp(scores.to(torch.float64) - row_max_high)
        sink_exp = torch.exp(sink.to(torch.float64) - row_max_high)
        denominator = score_exp.sum(dim=-1, keepdim=True) + sink_exp
        probability = (score_exp / denominator).to(value.dtype)
        lse = (torch.log(denominator) + row_max_high).squeeze(-1)

    output = torch.matmul(probability.to(value.dtype), v)
    return output.permute(0, 2, 1, 3), lse


def causal_mask(batch, q_len, kv_len):
    row = torch.arange(q_len)[None, :, None]
    col = torch.arange(kv_len)[None, None, :]
    offset = kv_len - q_len
    return (col - row >= offset + 1).expand(batch, -1, -1).clone()


def apply_fully_masked_contract(output, lse, mask):
    fully_masked = mask.all(dim=-1)
    output = output.clone()
    lse = lse.clone()
    output[fully_masked] = 0
    lse = lse.masked_fill(fully_masked[:, None, :], torch.inf)
    return output, lse


def test_ref_flash_attention_forward_and_batch_equivalence():
    """Verify that batched and per-batch computations produce identical results."""
    query, key, value = make_inputs(3, 5, 7, 4, 2, 8, seed=1234)
    scale = 1.0 / math.sqrt(query.shape[-1])

    actual, actual_lse = ref_flash_attention(
        query, key, value, scale, None, query.dtype
    )
    expected, expected_lse = independent_attention(query, key, value, scale)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-5)

    per_batch = []
    per_batch_lse = []
    for index in range(query.shape[0]):
        output, lse = independent_attention(
            query[index:index + 1], key[index:index + 1], value[index:index + 1], scale
        )
        per_batch.append(output)
        per_batch_lse.append(lse)
    torch.testing.assert_close(actual, torch.cat(per_batch), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_lse, torch.cat(per_batch_lse), rtol=1e-5, atol=1e-5)


def test_ref_flash_attention_gqa_masks_and_fully_masked_rows():
    """Verify GQA, causal masks, and the output contract for fully masked rows."""
    query, key, value = make_inputs(2, 4, 6, 4, 2, 8, seed=2345)
    scale = 1.0 / math.sqrt(query.shape[-1])
    mask = causal_mask(query.shape[0], query.shape[1], key.shape[1])
    mask[0, 1] = True

    actual, actual_lse = ref_flash_attention(
        query, key, value, scale, mask, query.dtype
    )
    expected, expected_lse = independent_attention(query, key, value, scale, mask)
    actual, actual_lse = apply_fully_masked_contract(actual, actual_lse, mask)
    expected, expected_lse = apply_fully_masked_contract(expected, expected_lse, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-5)
    assert torch.equal(actual[0, 1], torch.zeros_like(actual[0, 1]))
    assert torch.isinf(actual_lse[0, :, 1]).all()


def test_ref_flash_attention_sink_and_pair():
    """Verify sink normalization and the dual-reference wrapper."""
    query, key, value = make_inputs(2, 3, 5, 4, 2, 8, seed=3456)
    scale = 1.0 / math.sqrt(query.shape[-1])
    mask = torch.zeros(query.shape[0], query.shape[1], key.shape[1], dtype=torch.bool)
    mask[1, 0, 3:] = True
    sink = torch.randn(2, 4, 3, 1, generator=torch.Generator().manual_seed(4567))

    actual, actual_lse = ref_flash_attention(
        query, key, value, scale, mask, query.dtype, sink_matrix=sink
    )
    expected, expected_lse = independent_attention(
        query, key, value, scale, mask, sink=sink
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-5)

    out_ref, lse_ref, out_pt, lse_pt = ref_flash_attention_pair(
        query, key, value, scale, mask, query.dtype, sink_matrix=sink
    )
    direct_ref = ref_flash_attention(
        query, key, value, scale, mask, query.dtype,
        upcast=True, reorder_ops=False, sink_matrix=sink,
    )
    direct_pt = ref_flash_attention(
        query, key, value, scale, mask, query.dtype,
        upcast=False, reorder_ops=True, sink_matrix=sink,
    )
    torch.testing.assert_close(out_ref, direct_ref[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lse_ref, direct_ref[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_pt, direct_pt[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lse_pt, direct_pt[1], rtol=1e-5, atol=1e-5)


def test_ref_flash_attention_backward():
    """Verify that the shared reference supports gradients for Q, K, and V."""
    query, key, value = make_inputs(2, 3, 4, 4, 2, 8, seed=5678)
    query.requires_grad_()
    key.requires_grad_()
    value.requires_grad_()
    scale = 1.0 / math.sqrt(query.shape[-1])
    mask = causal_mask(query.shape[0], query.shape[1], key.shape[1])

    actual, _ = ref_flash_attention(query, key, value, scale, mask, query.dtype)
    grad_generator = torch.Generator(device="cpu").manual_seed(6789)
    grad = torch.randn(actual.shape, generator=grad_generator, dtype=actual.dtype)
    actual_grads = torch.autograd.grad(actual, (query, key, value), grad)

    query_ref, key_ref, value_ref = [tensor.detach().clone().requires_grad_()
                                     for tensor in (query, key, value)]
    expected, _ = independent_attention(query_ref, key_ref, value_ref, scale, mask)
    expected_grads = torch.autograd.grad(expected, (query_ref, key_ref, value_ref), grad)

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)


def test_assert_fa_close_detects_perturbed_output():
    """Verify that the comparator catches output errors caused by input perturbations.

    Both ref and pt are computed by ``ref_flash_attention``, while the manual
    baseline uses the independent ``independent_attention`` implementation.
    The assertion passes when all three agree. After adding 1 to one element of
    V and rerunning the independent implementation, the output must diverge
    enough for the assertion to fail. NaN outputs must also be rejected so an
    overly loose or broken comparator cannot silently pass them.
    """
    query, key, value = make_inputs(2, 3, 4, 4, 2, 8, seed=42)
    scale = 1.0 / math.sqrt(query.shape[-1])
    ref, _, pt, _ = ref_flash_attention_pair(query, key, value, scale, None, query.dtype)
    manual, _ = independent_attention(query, key, value, scale)
    assert_fa_close(manual, ref, pt, name="out")  # Manual and dual references agree.

    # Add 1 to one element of V and rerun the independent implementation. The
    # output should shift by roughly its softmax weight, far beyond tolerance,
    # proving that the comparator catches errors from real input perturbations.
    bad_value = value.clone()
    bad_value[0, 0, 0, 0] += 1.0
    bad, _ = independent_attention(query, key, bad_value, scale)
    caught = False
    try:
        assert_fa_close(bad, ref, pt, name="out")
    except AssertionError:
        caught = True
    assert caught, "assert_fa_close 未捕获输入扰动导致的输出偏差"

    nan_output = manual.clone()
    nan_output[0, 0, 0, 0] = float("nan")
    caught = False
    try:
        assert_fa_close(nan_output, ref, pt, name="out")
    except AssertionError:
        caught = True
    assert caught, "assert_fa_close 未捕获 NaN 输出"
