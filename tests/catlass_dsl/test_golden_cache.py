"""Regression coverage for CATLASS DSL golden-cache keys and warm autograd checks."""

from unittest.mock import Mock

import pytest
import torch

from . import mxfp8_reference, test_utils
from .mask_mod_definitions import get_mask_pair


@pytest.fixture
def golden_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_REFRESH", "0")
    return tmp_path


def test_attention_warm_cache_checks_fresh_gradients(golden_cache_dir, monkeypatch):
    cpu, inputs, cq, ck, _ = test_utils.make_inputs(
        "cpu", torch.float16, qp=True, q_lengths=(4, 5), sk=7
    )
    oracle = test_utils.oracle
    reference_calls = Mock(wraps=oracle)
    monkeypatch.setattr(test_utils, "oracle", reference_calls)
    mask = get_mask_pair("causal", window_size=7)

    def run(loss):
        out, lse = oracle(inputs, cu_q=cq, cu_k=ck, mask_mod=mask)
        return test_utils.check_attention(
            cpu, inputs, out, lse.contiguous(), cq=cq, ck=ck, loss=loss, mask_mod=mask
        )

    cold = run("both")
    warm = run("both")
    reference_calls.assert_called_once()
    for actual, expected in zip(warm, cold):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    dq, dk, dv = run("lse")
    assert dv.count_nonzero() == 0
    assert reference_calls.call_count == 2


def test_mxfp8_cache_keys_include_callback_capture_and_sampled_rows(golden_cache_dir, monkeypatch):
    case = mxfp8_reference.make_case(q_lengths=(4,), k_lengths=(5,), q_heads=1, kv_heads=1)
    compute = Mock(wraps=mxfp8_reference._reference)
    monkeypatch.setattr(mxfp8_reference, "_reference", compute)

    def mask(cutoff):
        def visible(b, h, q, k, info, aux, scalars):
            return k <= cutoff

        return visible

    cold = mxfp8_reference.reference(case, mask_mod=mask(torch.tensor(1)), query_rows=[[0]])
    warm = mxfp8_reference.reference(case, mask_mod=mask(torch.tensor(1)), query_rows=[[0]])
    for actual, expected in zip(warm, cold):
        torch.testing.assert_close(actual, expected, equal_nan=True, rtol=0, atol=0)
    compute.assert_called_once()

    changed = mxfp8_reference.reference(case, mask_mod=mask(torch.tensor(2)), query_rows=[[0]])
    assert not torch.equal(changed[1][..., 0], cold[1][..., 0])
    sampled = mxfp8_reference.reference(case, mask_mod=mask(torch.tensor(1)), query_rows=[[1]])
    assert sampled[1][..., 0].isnan().all() and sampled[1][..., 1].isfinite().all()
    assert compute.call_count == 3
