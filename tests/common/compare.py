# Copyright (c) 2026, Minghua Shen.

"""Numerical comparison rules shared by the attention tests."""

import functools
import os
import torch
import time
from tests.common.golden_cache import retry_cached_value
from tests.common.timing import add


@functools.lru_cache(maxsize=1)
def _is_ascend950() -> bool:
    try:
        import torch_npu
        return "Ascend950" in torch_npu.npu.get_device_name()
    except Exception:
        return False


def _npu_compare_allowed(actual) -> bool:
    """NPU fast path gate, default all CPU while diagnosing, env FA_TEST_COMPARE is auto, npu or cpu."""
    if not (hasattr(actual, "is_npu") and actual.is_npu):
        return False
    mode = os.environ.get("FA_TEST_COMPARE", "cpu").strip().lower()
    if mode == "cpu":
        return False
    if mode == "auto" and _is_ascend950():
        return False
    return True


def _assert_fa_close_npu(actual, ref, pt, *, softcap=0.0, name="out"):
    """NPU fast path, verdict and diagnostics computed on device, one scalar back on pass."""
    device = actual.device
    if not ref.is_npu:
        ref = ref.detach().to(device)
    if not pt.is_npu:
        pt = pt.detach().to(device)

    rtol = 3.0 if softcap != 0.0 else 2.0
    # Same predicate as the CPU path, all tensors until the single verdict sync.
    nan_bad = actual.isnan().any() | ref.isnan().any() | pt.isnan().any()
    ref_inf = ref.isinf()
    inf_ok = (
        ((actual.isinf() == ref_inf).all() & (pt.isinf() == ref_inf).all())
        & (~ref_inf | (actual == ref)).all()
        & (~ref_inf | (pt == ref)).all()
    )
    # Inf positions cannot enter the metric, zero them like the CPU path masks.
    diff = (actual.float() - ref.float()).abs_()
    pt_diff_all = (pt.float() - ref.float()).abs_()
    ulp_all = (ref + 0.3 - 0.3 - ref).abs()
    diff.masked_fill_(ref_inf, 0)
    pt_diff_all.masked_fill_(ref_inf, 0)
    ulp_all.masked_fill_(ref_inf, 0)
    max_diff = diff.max()
    pt_diff = pt_diff_all.max()
    ulp = ulp_all.max()
    tolerance = torch.clamp(rtol * pt_diff + 2.0 * ulp, min=1e-5)
    passed = (
        ~nan_bad & inf_ok
        & ((max_diff <= 1e-5) | (max_diff <= rtol * pt_diff) | (max_diff <= tolerance))
    )
    if bool(passed):
        return

    # Failure only: worst spots with samples plus metric scalars, pulled once.
    flat_actual, flat_ref, flat_pt = actual.flatten(), ref.flatten(), pt.flatten()
    k = min(4, diff.numel())
    top_vals, top_idx = diff.flatten().topk(k)
    samples = torch.stack([
        top_vals,
        flat_actual[top_idx].float(),
        flat_ref[top_idx].float(),
        flat_pt[top_idx].float(),
    ]).t().reshape(-1)
    metrics = torch.stack([
        nan_bad.float(), (~inf_ok).float(),
        max_diff.float(), pt_diff.float(), ulp.float(),
        (diff > tolerance).sum().float(),
        (diff > torch.clamp(tolerance, min=0.5)).sum().float(),
    ])
    idx_cpu = top_idx.cpu()
    samples = samples.cpu()
    metrics = metrics.cpu()
    nan_bad_v, inf_bad_v, max_diff_v, pt_diff_v, ulp_v, num_bad, num_loose = metrics.tolist()
    if nan_bad_v:
        raise AssertionError(f"{name}: actual/ref/pt contains NaN")
    if inf_bad_v:
        raise AssertionError(f"{name}: actual/pt inf mask or inf value mismatch vs ref")
    spots = [samples[i * 4:(i + 1) * 4].tolist() for i in range(k)]
    total = actual.numel()
    print(f"  [DEBUG] {name}: shape={tuple(actual.shape)} "
          f"num_bad(>{float(tolerance):.3g})={int(num_bad)} "
          f"num_loose(>0.5)={int(num_loose)}")
    for fi, (d, av, rv, pv) in zip(idx_cpu.tolist(), spots):
        print(f"    diff={d} flat={fi}/{total} ({100.0 * fi / total:.1f}%) "
              f"actual={av} ref={rv} pt={pv}")
    raise AssertionError(
        f"{name}: max|actual-ref|={max_diff_v} exceeds "
        f"{rtol} * max|pt-ref|={pt_diff_v} + 2*ULP(ref)={2.0 * ulp_v} "
        f"(softcap={softcap})")


def _assert_fa_close(actual, ref, pt, *, softcap=0.0, name="out"):
    """Compare implementation results using Tri Dao's dual-reference rule.

    NPU fast path for the medium size band, CPU path otherwise.
    """
    if _npu_compare_allowed(actual) and ref.numel() > 0:
        n = actual.numel()
        if 1_000_000 <= n <= 50_000_000:
            try:
                _assert_fa_close_npu(actual, ref, pt, softcap=softcap, name=name)
                return
            except RuntimeError:
                pass

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

    # Every NaN has been rejected above, so "not infinite" is exactly "finite".
    ref_inf = torch.isinf(ref)
    if ref_inf.any():
        assert torch.equal(torch.isinf(actual), ref_inf), (
            f"{name}: actual/ref inf mask mismatch"
        )
        assert torch.equal(torch.isinf(pt), ref_inf), f"{name}: pt/ref inf mask mismatch"
        assert torch.equal(actual[ref_inf], ref[ref_inf]), (
            f"{name}: actual/ref inf value mismatch"
        )
        assert torch.equal(pt[ref_inf], ref[ref_inf]), f"{name}: pt/ref inf value mismatch"
        # Compare numerical error only over the finite elements.
        finite = ~ref_inf
        if not finite.any():
            return
        actual = actual[finite]
        ref = ref[finite]
        pt = pt[finite]
    else:
        # ref is entirely finite, so the inf masks agree exactly when the two
        # other tensors have no infinity either.  Check that without building
        # their (all-false) masks.
        assert not torch.isinf(actual).any(), f"{name}: actual/ref inf mask mismatch"
        assert not torch.isinf(pt).any(), f"{name}: pt/ref inf mask mismatch"

    rtol = 3.0 if softcap != 0.0 else 2.0
    diff = actual - ref
    diff.abs_()
    max_diff = diff.max().item()
    # The tolerance is max(rtol * pt_diff + 2 * ULP, 1e-5); below 1e-5 it always passes.
    if max_diff <= 1e-5:
        return
    pt_diff = (pt - ref).abs_().max().item()
    # 2 * ULP >= 0, so the full tolerance is at least rtol * pt_diff.
    if max_diff <= rtol * pt_diff:
        return
    # Compute ULP in ref's original dtype. Converting to float32 first would
    # lose the fp16/bf16 ULP that this check must measure.
    ulp = (ref + 0.3 - 0.3 - ref).abs().max().item()
    # When both references match exactly (pt_diff=0) and ref is near zero, the
    # tolerance above collapses to ~0 and cannot cover the implementation's own
    # ULP noise (for example, dQ around 1e-6). Add an absolute lower bound.
    tolerance = max(rtol * pt_diff + 2.0 * ulp, 1e-5)
    # Temporary diagnostics: report the number of failures (isolated element
    # versus a full row), their locations, and the surrounding window.
    if max_diff > tolerance:
        # Flat index of the worst element: nonzero() returns one row of
        # coordinates per hit, so it must be flattened before item().
        idx = (diff == max_diff).nonzero().flatten()
        fi = int(idx[0])
        num_bad = int((diff > tolerance).sum())
        num_loose = int((diff > max(0.5, tolerance)).sum())
        # fi is flat, so index through flat views rather than actual[fi].
        flat_actual, flat_ref, flat_pt = actual.flatten(), ref.flatten(), pt.flatten()
        total = flat_actual.numel()
        lo = max(0, fi - 3)
        hi = min(total, fi + 4)
        print(f"  [DEBUG] {name}: shape={tuple(actual.shape)} "
              f"num_bad(>{tolerance:.3g})={num_bad} num_loose(>0.5)={num_loose}")
        print(f"    max_diff={max_diff} flat={fi}/{total} ({100.0*fi/total:.1f}%) "
              f"actual={flat_actual[fi].item()} ref={flat_ref[fi].item()} "
              f"pt={flat_pt[fi].item()}")
        print(f"    actual[{lo}:{hi}]={flat_actual[lo:hi].tolist()}")
        print(f"    ref   [{lo}:{hi}]={flat_ref[lo:hi].tolist()}")
    assert max_diff <= tolerance, (
        f"{name}: max|actual-ref|={max_diff} exceeds "
        f"{rtol} * max|pt-ref|={pt_diff} + 2*ULP(ref)={2.0 * ulp} "
        f"(softcap={softcap})"
    )


def assert_fa_close(actual, ref, pt, *, softcap=0.0, name="out"):
    """Compare results, refreshing a cached golden once after a mismatch."""
    state = getattr(assert_fa_close, "_ci_timing_state", None)
    started = time.perf_counter()
    try:
        _assert_fa_close(actual, ref, pt, softcap=softcap, name=name)
    except AssertionError:
        refreshed = retry_cached_value(ref) or retry_cached_value(pt)
        if not refreshed:
            raise
        state = getattr(assert_fa_close, "_ci_timing_state", None)
        if state is not None:
            state["events"].append("fallback=mismatch_refresh")
        print(f"[golden-cache] mismatch for {name}; recomputed golden and retrying")
        _assert_fa_close(actual, ref, pt, softcap=softcap, name=name)
    finally:
        if state is not None:
            add(state, "compare", time.perf_counter() - started)
