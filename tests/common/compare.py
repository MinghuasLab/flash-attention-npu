# Copyright (c) 2026, Minghua Shen.

"""Numerical comparison rules shared by the attention tests."""

import os
import torch
import time
from tests.common.golden_cache import retry_cached_value
from tests.common.timing import add


def _npu_compare_allowed(actual) -> bool:
    """NPU fast path gate; FA_TEST_COMPARE accepts auto, npu, or cpu."""
    if not (hasattr(actual, "is_npu") and actual.is_npu):
        return False
    mode = os.environ.get("FA_TEST_COMPARE", "auto").strip().lower()
    if mode == "cpu":
        return False
    return True


def _format_compare_diagnostics(
    name, shape, total, max_diff, tolerance, num_bad, num_loose, spots, *, pt_values=True
):
    """Format CPU/NPU mismatch details in one stable, Catlass-style layout."""
    lines = [
        "-" * 79,
        f"[COMPARE FAILED] {name}",
        f"shape={tuple(shape)} total={total} max_diff={max_diff:.6e} "
        f"tolerance={tolerance:.6e} num_bad={num_bad} num_loose={num_loose}",
        "flat_index | diff | actual | ref" + (" | pt" if pt_values else ""),
    ]
    for spot in spots:
        fi, diff, actual, ref = spot[:4]
        line = f"{fi:10d} | {diff:.6e} | {actual!r} | {ref!r}"
        if pt_values:
            line += f" | {spot[4]!r}"
        lines.append(line)
    lines.append("-" * 79)
    return "\n".join(lines)


def _top_compare_spots(actual, ref, diff, pt=None, k=4):
    flat_actual, flat_ref = actual.flatten(), ref.flatten()
    top_vals, top_idx = diff.flatten().topk(min(k, diff.numel()))
    flat_pt = pt.flatten() if pt is not None else None
    spots = []
    for index, value in zip(top_idx.tolist(), top_vals.tolist()):
        spot = [
            int(index),
            float(value),
            flat_actual[index].item(),
            flat_ref[index].item(),
        ]
        if flat_pt is not None:
            spot.append(flat_pt[index].item())
        spots.append(spot)
    return spots


def _assert_fa_close_npu(actual, ref, pt, *, softcap=0.0, name="out"):
    """NPU fast path, verdict and diagnostics computed on device, one scalar back on pass."""
    if actual.shape != ref.shape or actual.shape != pt.shape:
        raise AssertionError(
            f"{name}: shape mismatch actual={tuple(actual.shape)} "
            f"ref={tuple(ref.shape)} pt={tuple(pt.shape)}"
        )

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
        ~nan_bad
        & inf_ok
        & ((max_diff <= 1e-5) | (max_diff <= rtol * pt_diff) | (max_diff <= tolerance))
    )
    if bool(passed):
        return

    # Failure only: worst spots with samples plus metric scalars, pulled once.
    flat_actual, flat_ref, flat_pt = actual.flatten(), ref.flatten(), pt.flatten()
    k = min(4, diff.numel())
    top_vals, top_idx = diff.flatten().topk(k)
    index_base = 2**24
    sample_rows = torch.stack(
        [
            (top_idx // index_base).float(),
            (top_idx % index_base).float(),
            top_vals,
            flat_actual[top_idx].float(),
            flat_ref[top_idx].float(),
            flat_pt[top_idx].float(),
        ]
    ).t()
    metrics = torch.stack(
        [
            nan_bad.float(),
            (~inf_ok).float(),
            max_diff.float(),
            pt_diff.float(),
            ulp.float(),
            (diff > tolerance).sum().float(),
            (diff > torch.clamp(tolerance, min=0.5)).sum().float(),
            tolerance.float(),
        ]
    )
    diagnostics = torch.cat((sample_rows.flatten(), metrics)).cpu()
    sample_size = sample_rows.numel()
    samples = diagnostics[:sample_size].reshape(k, 6).tolist()
    (
        nan_bad_v,
        inf_bad_v,
        max_diff_v,
        pt_diff_v,
        ulp_v,
        num_bad,
        num_loose,
        tolerance_v,
    ) = diagnostics[sample_size:].tolist()
    if nan_bad_v:
        raise AssertionError(f"{name}: actual/ref/pt contains NaN")
    if inf_bad_v:
        raise AssertionError(f"{name}: actual/pt inf mask or inf value mismatch vs ref")
    spots = [
        [int(index_hi) * index_base + int(index_lo), d, av, rv, pv]
        for index_hi, index_lo, d, av, rv, pv in samples
    ]
    print(
        _format_compare_diagnostics(
            name,
            actual.shape,
            actual.numel(),
            max_diff_v,
            tolerance_v,
            int(num_bad),
            int(num_loose),
            spots,
        )
    )
    raise AssertionError(
        f"{name}: max|actual-ref|={max_diff_v} exceeds "
        f"{rtol} * max|pt-ref|={pt_diff_v} + 2*ULP(ref)={2.0 * ulp_v} "
        f"(softcap={softcap})"
    )


def _assert_fa_close(actual, ref, pt, *, softcap=0.0, name="out"):
    """Compare implementation results using Tri Dao's dual-reference rule.

    NPU fast path for the medium size band, CPU path otherwise.
    """
    if actual.shape != ref.shape or actual.shape != pt.shape:
        raise AssertionError(
            f"{name}: shape mismatch actual={tuple(actual.shape)} "
            f"ref={tuple(ref.shape)} pt={tuple(pt.shape)}"
        )

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
    assert not (torch.isnan(actual).any() or torch.isnan(ref).any() or torch.isnan(pt).any()), (
        f"{name}: actual/ref/pt contains NaN"
    )

    # Every NaN has been rejected above, so "not infinite" is exactly "finite".
    ref_inf = torch.isinf(ref)
    if ref_inf.any():
        assert torch.equal(torch.isinf(actual), ref_inf) and torch.equal(
            torch.isinf(pt), ref_inf
        ), f"{name}: actual/pt inf mask or inf value mismatch vs ref"
        assert torch.equal(actual[ref_inf], ref[ref_inf]) and torch.equal(
            pt[ref_inf], ref[ref_inf]
        ), f"{name}: actual/pt inf mask or inf value mismatch vs ref"
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
        assert not torch.isinf(actual).any() and not torch.isinf(pt).any(), (
            f"{name}: actual/pt inf mask or inf value mismatch vs ref"
        )

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
    # Report the same top-mismatch layout as the NPU path.
    if max_diff > tolerance:
        num_bad = int((diff > tolerance).sum())
        num_loose = int((diff > max(0.5, tolerance)).sum())
        print(
            _format_compare_diagnostics(
                name,
                actual.shape,
                actual.numel(),
                max_diff,
                tolerance,
                num_bad,
                num_loose,
                _top_compare_spots(actual, ref, diff, pt),
            )
        )
    assert max_diff <= tolerance, (
        f"{name}: max|actual-ref|={max_diff} exceeds "
        f"{rtol} * max|pt-ref|={pt_diff} + 2*ULP(ref)={2.0 * ulp} "
        f"(softcap={softcap})"
    )


def _assert_single_close_npu(actual, ref, *, rtol, atol, name):
    if actual.shape != ref.shape:
        raise AssertionError(
            f"{name}: shape mismatch actual={tuple(actual.shape)} ref={tuple(ref.shape)}"
        )
    if not ref.is_npu:
        ref = ref.detach().to(actual.device)

    nan_bad = actual.isnan().any() | ref.isnan().any()
    ref_inf = ref.isinf()
    inf_ok = (actual.isinf() == ref_inf).all() & (~ref_inf | (actual == ref)).all()
    diff = (actual.float() - ref.float()).abs_()
    limit = atol + rtol * ref.float().abs()
    diff.masked_fill_(ref_inf, 0)
    limit.masked_fill_(ref_inf, 0)
    max_diff = diff.max()
    passed = ~nan_bad & inf_ok & (diff <= limit).all()
    if bool(passed):
        return

    flat_actual, flat_ref = actual.flatten(), ref.flatten()
    k = min(4, diff.numel())
    top_vals, top_idx = diff.flatten().topk(k)
    index_base = 2**24
    sample_rows = torch.stack(
        [
            (top_idx // index_base).float(),
            (top_idx % index_base).float(),
            top_vals,
            flat_actual[top_idx].float(),
            flat_ref[top_idx].float(),
        ]
    ).t()
    diagnostics = torch.cat(
        (
            sample_rows.flatten(),
            torch.stack(
                [
                    nan_bad.float(),
                    (~inf_ok).float(),
                    max_diff.float(),
                    (diff > limit).sum().float(),
                    limit.max().float(),
                ]
            ),
        )
    ).cpu()
    sample_size = sample_rows.numel()
    samples = diagnostics[:sample_size].reshape(k, 5).tolist()
    nan_bad_v, inf_bad_v, max_diff_v, num_bad, limit_v = diagnostics[sample_size:].tolist()
    if nan_bad_v:
        raise AssertionError(f"{name}: actual/ref contains NaN")
    if inf_bad_v:
        raise AssertionError(f"{name}: actual/ref inf mask or inf value mismatch")
    spots = [
        [int(index_hi) * index_base + int(index_lo), diff_v, actual_v, ref_v]
        for index_hi, index_lo, diff_v, actual_v, ref_v in samples
    ]
    print(
        _format_compare_diagnostics(
            name,
            actual.shape,
            actual.numel(),
            max_diff_v,
            limit_v,
            int(num_bad),
            0,
            spots,
            pt_values=False,
        )
    )
    raise AssertionError(f"{name}: max|actual-ref|={max_diff_v} exceeds torch.testing tolerance")


def assert_fa_close_single(actual, ref, *, rtol, atol, name="out"):
    """Compare one result against one reference, using NPU diagnostics when possible."""
    if actual.shape != ref.shape:
        raise AssertionError(
            f"{name}: shape mismatch actual={tuple(actual.shape)} ref={tuple(ref.shape)}"
        )
    if _npu_compare_allowed(actual) and ref.numel() > 0:
        n = actual.numel()
        if 1_000_000 <= n <= 50_000_000:
            try:
                _assert_single_close_npu(actual, ref, rtol=rtol, atol=atol, name=name)
                return
            except RuntimeError:
                pass
    actual_cpu = actual.detach().cpu()
    ref_cpu = ref.detach().cpu()
    try:
        torch.testing.assert_close(actual_cpu, ref_cpu, rtol=rtol, atol=atol, msg=name)
    except AssertionError:
        diff = (actual_cpu.float() - ref_cpu.float()).abs()
        diff = torch.nan_to_num(diff, nan=float("inf"))
        limit = atol + rtol * ref_cpu.float().abs()
        max_diff = diff.max().item()
        num_bad = int((diff > limit).sum())
        print(
            _format_compare_diagnostics(
                name,
                actual_cpu.shape,
                actual_cpu.numel(),
                max_diff,
                limit.max().item(),
                num_bad,
                int((diff > max(0.5, limit.max().item())).sum()),
                _top_compare_spots(actual_cpu, ref_cpu, diff),
                pt_values=False,
            )
        )
        raise


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
