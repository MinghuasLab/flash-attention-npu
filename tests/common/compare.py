# Copyright (c) 2026, Minghua Shen.

"""Numerical comparison rules shared by the attention tests."""

import torch
from tests.common.golden_cache import retry_cached_value


def _assert_fa_close(actual, ref, pt, *, softcap=0.0, name="out"):
    """Compare implementation results using Tri Dao's dual-reference rule.

    ``ref`` is the high-precision reference and ``pt`` is the PyTorch baseline.
    Their maximum difference estimates the numerical error range:
    ``max|actual - ref| <= rtol * max|pt - ref| + 2 * ULP(ref)``.
    """
    actual = actual.detach().cpu()
    ref = ref.detach().cpu()
    pt = pt.detach().cpu()
    assert actual.shape == ref.shape == pt.shape, (
        f"{name}: shape mismatch actual={tuple(actual.shape)} "
        f"ref={tuple(ref.shape)} pt={tuple(pt.shape)}"
    )
    if actual.numel() == 0:
        assert ref.numel() == 0 and pt.numel() == 0, f"{name}: empty shape mismatch"
        return

    # Always reject NaNs. Compare matching infinities semantically before any
    # subtraction because infinities cannot participate in the error metric.
    assert not torch.isnan(actual).any(), f"{name}: actual contains NaN"
    assert not torch.isnan(ref).any(), f"{name}: ref contains NaN"
    assert not torch.isnan(pt).any(), f"{name}: pt contains NaN"

    actual_inf = torch.isinf(actual)
    ref_inf = torch.isinf(ref)
    pt_inf = torch.isinf(pt)
    assert torch.equal(actual_inf, ref_inf), f"{name}: actual/ref inf mask mismatch"
    assert torch.equal(pt_inf, ref_inf), f"{name}: pt/ref inf mask mismatch"
    if ref_inf.any():
        assert torch.equal(actual[ref_inf], ref[ref_inf]), (
            f"{name}: actual/ref inf value mismatch"
        )
        assert torch.equal(pt[ref_inf], ref[ref_inf]), f"{name}: pt/ref inf value mismatch"

    # For tensors mixing finite values and infinities, compare numerical error
    # only over the finite elements.
    finite = torch.isfinite(ref)
    if not finite.any():
        return
    actual = actual[finite]
    ref = ref[finite]
    pt = pt[finite]

    rtol = 3.0 if softcap != 0.0 else 2.0
    # Compute ULP in ref's original dtype. Converting to float32 first would
    # lose the fp16/bf16 ULP that this check must measure.
    ulp = (ref + 0.3 - 0.3 - ref).abs().max().item()
    diff = (actual - ref).abs()
    max_diff = diff.max().item()
    pt_diff = (pt - ref).abs().max().item()
    # When both references match exactly (pt_diff=0) and ref is near zero, the
    # tolerance above collapses to ~0 and cannot cover the implementation's own
    # ULP noise (for example, dQ around 1e-6). Add an absolute lower bound.
    tolerance = max(rtol * pt_diff + 2.0 * ulp, 1e-5)
    # Temporary diagnostics: report the number of failures (isolated element
    # versus a full row), their locations, and the surrounding window.
    if max_diff > tolerance:
        idx = (diff == max_diff).nonzero()
        fi = idx[0].item()
        num_bad = int((diff > tolerance).sum())
        num_loose = int((diff > max(0.5, tolerance)).sum())
        total = actual.numel()
        lo = max(0, fi - 3)
        hi = min(total, fi + 4)
        print(f"  [DEBUG] {name}: shape={tuple(actual.shape)} "
              f"num_bad(>{tolerance:.3g})={num_bad} num_loose(>0.5)={num_loose}")
        print(f"    max_diff={max_diff} flat={fi}/{total} ({100.0*fi/total:.1f}%) "
              f"actual={actual[fi].item()} ref={ref[fi].item()} pt={pt[fi].item()}")
        print(f"    actual[{lo}:{hi}]={actual[lo:hi].tolist()}")
        print(f"    ref   [{lo}:{hi}]={ref[lo:hi].tolist()}")
    assert max_diff <= tolerance, (
        f"{name}: max|actual-ref|={max_diff} exceeds "
        f"{rtol} * max|pt-ref|={pt_diff} + 2*ULP(ref)={2.0 * ulp} "
        f"(softcap={softcap})"
    )


def assert_fa_close(actual, ref, pt, *, softcap=0.0, name="out"):
    """Compare results, refreshing a cached golden once after a mismatch."""
    try:
        _assert_fa_close(actual, ref, pt, softcap=softcap, name=name)
    except AssertionError:
        if not retry_cached_value(ref) and not retry_cached_value(pt):
            raise
        print(f"[golden-cache] mismatch for {name}; recomputed golden and retrying")
        _assert_fa_close(actual, ref, pt, softcap=softcap, name=name)
