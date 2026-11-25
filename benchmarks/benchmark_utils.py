# Copyright (c) 2026, Minghua Shen.

"""Shared profiler and reporting utilities for FlashAttention-NPU benchmarks.

Timing protocol:
    Each benchmark case uses a single profiler session with scheduled warmup
    and active collection phases. During each iteration, FlashAttention-NPU
    and CANN FusionAttention are executed alternately under the same profiler.

    After profiling, kernel_details.csv is parsed once. Kernel durations are
    attributed to each implementation using configurable kernel-name patterns.

    The reported mean latency is the sum of matched device-kernel durations
    divided by the number of active iterations. For FWD+BWD mode, the
    measurement includes matched forward and backward kernels.

    These measurements represent aggregated device-kernel execution time,
    not Python API wall time or end-to-end latency. Since reliable iteration
    boundaries are not reconstructed from kernel_details.csv, per-iteration
    minimum, maximum, and standard deviation are not reported.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Callable, Any

import torch
import torch_npu
from torch_npu.profiler.profiler import analyse

AUX_PATTERN = re.compile(r"(?:^|[_./])(Fill|ReduceSum|Cast)(?:$|[_./])", re.I)
DEFAULT_FA_PATTERN = r"SplitFuse|FAInfer|FAG"
DEFAULT_BASE_PATTERN = r"FlashAttention"

L2_MIB = 1024**2


class L2CacheEvictor:
    """
    Evict cached data by scanning a device buffer
    larger than the detected L2 cache.
    """

    def __init__(self, device_index: int, flush_bytes: int):
        if flush_bytes <= 0:
            raise ValueError("flush_bytes must be positive")
        self.device = f"npu:{device_index}"
        # Use FP32: better supported for NPU ReduceSum than uint8->int64.
        # Round up the allocation to at least flush_bytes bytes.
        numel = (flush_bytes + 3) // 4
        self.buffer = torch.empty(numel, dtype=torch.float32, device=self.device)
        self.buffer.fill_(1)
        self.sink = None
        torch.npu.synchronize()

    @torch.no_grad()
    def flush(self):
        # Cache eviction by streaming over a large input buffer; this is NOT
        # a guaranteed hardware cache invalidate.
        self.sink = self.buffer.sum()
        torch.npu.synchronize()


def query_l2_cache_info(device_index: int, override_mb=None):
    """
    Query the current NPU's L2 cache size.

    Priority:
      1. Explicit CLI override
      2. torch.npu.get_device_properties().L2_cache_size
    """
    props = torch.npu.get_device_properties(device_index)

    device_name = props.name

    if override_mb is not None:
        if override_mb <= 0:
            raise ValueError("--l2-cache-mb must be positive")

        l2_cache_bytes = int(override_mb * L2_MIB)
        source = "cli_override"

    else:
        l2_cache_bytes = int(getattr(props, "L2_cache_size", 0) or 0)

        if l2_cache_bytes <= 0:
            raise RuntimeError(
                f"Cannot detect L2 cache size for {device_name}. "
                "Please provide --l2-cache-mb explicitly."
            )

        source = "torch.npu.get_device_properties"

    return {
        "device_name": device_name,
        "l2_cache_bytes": l2_cache_bytes,
        "l2_cache_mib": round(l2_cache_bytes / L2_MIB, 2),
        "l2_cache_source": source,
    }


def wrap_with_l2_flush(step, evictor):

    if evictor is None:
        return step

    def wrapped():
        evictor.flush()
        return step()

    return wrapped


def clear_grads(tensors):
    for tensor in tensors:
        if tensor is not None and tensor.is_leaf:
            tensor.grad = None


def step_forward(fn: Callable[[], torch.Tensor]) -> None:
    out = fn()
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"backend returned non-tensor: {type(out)!r}")


def step_fwd_bwd(
    fn,
    inputs,
    grad,
    *,
    before_fwd=None,
    before_bwd=None,
):
    clear_grads(inputs)

    # L2 Flush before Forward
    if before_fwd is not None:
        before_fwd()

    out = fn()

    if isinstance(out, (tuple, list)):
        out = out[0]

    if not isinstance(out, torch.Tensor) or not out.requires_grad:
        raise NotImplementedError("backend forward output does not support autograd")

    # L2 Flush before Backward
    if before_bwd is not None:
        before_bwd()

    out.backward(grad)


def _event_step(fn, inputs, grad, mode, evictor, start_event, end_event):
    """One torch.npu.Event measurement; L2 flush is outside timed intervals."""

    def flush_l2():
        if evictor is not None:
            evictor.flush()

    def unwrap_output(out, require_grad=False):
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"backend returned non-tensor: {type(out)!r}")
        if require_grad and not out.requires_grad:
            raise NotImplementedError("backend forward output does not support autograd")
        return out

    if mode == "fwd":
        flush_l2()
        start_event.record()
        out = fn()
        end_event.record()
        end_event.synchronize()
        fwd_ms = start_event.elapsed_time(end_event)
        unwrap_output(out)
        return {"fwd_ms": fwd_ms, "bwd_ms": None, "total_ms": fwd_ms}

    if mode != "fwd_bwd":
        raise ValueError(f"unsupported Event timing mode: {mode!r}")

    clear_grads(inputs)
    flush_l2()
    start_event.record()
    out = fn()
    end_event.record()
    end_event.synchronize()
    fwd_ms = start_event.elapsed_time(end_event)
    out = unwrap_output(out, require_grad=True)

    # User-requested cold-cache protocol: flush once more before Backward.
    flush_l2()
    start_event.record()
    out.backward(grad)
    end_event.record()
    end_event.synchronize()
    bwd_ms = start_event.elapsed_time(end_event)
    return {"fwd_ms": fwd_ms, "bwd_ms": bwd_ms, "total_ms": fwd_ms + bwd_ms}


def measure_event_pair(
    fa_fn,
    baseline_fn,
    *,
    mode,
    fa_inputs,
    baseline_inputs,
    fa_grad,
    baseline_grad,
    evictor,
    warmup: int,
    repeat: int,
):
    """Measure FA and CANN separately with NPU Events, alternating sample order.

    Event intervals contain the operator's queued device work; L2 eviction,
    input-grad clearing, and host synchronization are outside timed intervals.
    For FWD+BWD, forward and backward Event durations are summed per sample.
    """
    if warmup < 0 or repeat < 1:
        raise ValueError("Event warmup must be >= 0 and repeat must be >= 1")

    fa_start = torch.npu.Event(enable_timing=True)
    fa_end = torch.npu.Event(enable_timing=True)
    base_start = torch.npu.Event(enable_timing=True)
    base_end = torch.npu.Event(enable_timing=True)

    def fa_event_step():
        return _event_step(fa_fn, fa_inputs, fa_grad, mode, evictor, fa_start, fa_end)

    def baseline_event_step():
        return _event_step(
            baseline_fn, baseline_inputs, baseline_grad, mode, evictor, base_start, base_end
        )

    for _ in range(warmup):
        fa_event_step()
        baseline_event_step()
    torch.npu.synchronize()

    fa_samples = []
    baseline_samples = []
    for index in range(repeat):
        if index % 2 == 0:
            fa_sample = fa_event_step()
            baseline_sample = baseline_event_step()
        else:
            baseline_sample = baseline_event_step()
            fa_sample = fa_event_step()
        fa_samples.append(fa_sample)
        baseline_samples.append(baseline_sample)

    def summarize(samples, prefix):
        totals = [item["total_ms"] for item in samples]
        fwd = [item["fwd_ms"] for item in samples]
        bwd = [item["bwd_ms"] for item in samples if item["bwd_ms"] is not None]
        out = {
            f"{prefix}_event_mean_ms": statistics.mean(totals),
            f"{prefix}_event_median_ms": statistics.median(totals),
            f"{prefix}_event_min_ms": min(totals),
            f"{prefix}_event_max_ms": max(totals),
            f"{prefix}_event_std_ms": statistics.pstdev(totals),
            f"{prefix}_event_fwd_mean_ms": statistics.mean(fwd),
            f"{prefix}_event_fwd_median_ms": statistics.median(fwd),
            f"{prefix}_event_bwd_mean_ms": statistics.mean(bwd) if bwd else None,
            f"{prefix}_event_bwd_median_ms": statistics.median(bwd) if bwd else None,
        }
        return out

    result = {**summarize(fa_samples, "fa"), **summarize(baseline_samples, "baseline")}
    fa_mean = result["fa_event_mean_ms"]
    baseline_mean = result["baseline_event_mean_ms"]
    result.update(
        event_delta_ms=fa_mean - baseline_mean,
        event_speedup_fa_over_baseline=(baseline_mean / fa_mean if fa_mean > 0 else None),
        event_warmup=warmup,
        event_repeat=repeat,
        event_timing_scope="npu_event_operator_elapsed_excluding_l2_flush; fwd_bwd_is_sum_of_two_events",
        event_l2_flush_excluded=True,
        event_status="ok",
    )
    return result


def build_profiler(root: Path, warmup: int, repeat: int):
    """One session with warmup and active iterations."""
    root.mkdir(parents=True, exist_ok=True)

    config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        data_simplification=False,
        l2_cache=False,
    )

    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup, active=repeat, repeat=1),
        record_shapes=False,
        profile_memory=False,
        with_modules=False,
        experimental_config=config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(root), analyse_flag=False),
    )


def _find_profile(root: Path) -> Path:
    candidates = list(root.glob("*_ascend_pt"))
    if not candidates:
        raise RuntimeError(f"No *_ascend_pt generated under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def _load_kernels(root: Path) -> tuple[list[dict[str, str]], Path]:
    """解析 Profiler 结果，失败时提供原始诊断信息。"""
    import inspect

    ascend_pt = _find_profile(root)

    params = inspect.signature(analyse).parameters
    kwargs = {"profiler_path": str(ascend_pt)}

    if "max_process_number" in params:
        kwargs["max_process_number"] = 4

    if "export_type" in params:
        kwargs["export_type"] = "text"

    def diagnostic():
        log_dir = ascend_pt / "logs"
        parts = [
            f"profile={ascend_pt}",
            f"logs_dir={log_dir}",
        ]

        if log_dir.is_dir():
            files = sorted(
                (p for p in log_dir.rglob("*") if p.is_file()),
                key=lambda p: p.stat().st_mtime_ns,
                reverse=True,
            )[:8]

            for path in files:
                parts.append(f"log_file={path}")

                try:
                    lines = path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    ).splitlines()

                    hits = [
                        line.strip()[:400]
                        for line in lines
                        if any(
                            term in line.lower()
                            for term in (
                                "error",
                                "failed",
                                "sqlite",
                                "task",
                                "exception",
                            )
                        )
                    ]

                    parts.extend(hits[-8:])

                except OSError:
                    pass

        return " | ".join(parts)[:5500]

    try:
        analyse(**kwargs)

    except Exception as exc:
        raise RuntimeError(
            f"Profiler analyse failed: {type(exc).__name__}: {exc}; {diagnostic()}"
        ) from exc

    files = list(ascend_pt.rglob("kernel_details.csv"))

    if not files:
        raise RuntimeError(f"kernel_details.csv not found after profiler analyse; {diagnostic()}")

    csv_file = max(
        files,
        key=lambda p: p.stat().st_mtime_ns,
    )

    with csv_file.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"kernel_details.csv has no kernel rows: {csv_file}; {diagnostic()}")

    return rows, csv_file


def _select_kernel(rows, pattern: re.Pattern[str]):
    hits = []
    for row in rows:
        name = row.get("Name", "") or ""
        kind = row.get("Type", "") or ""
        # Match by Name or Type, as done by the reference BSND profiler.
        key = name if name else kind
        if AUX_PATTERN.search(name) or AUX_PATTERN.search(kind):
            continue
        if pattern.search(name) or pattern.search(kind):
            try:
                d = float(row["Duration(us)"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid Duration(us) for kernel {key!r}") from exc
            if not math.isfinite(d) or d < 0:
                raise RuntimeError(f"Invalid kernel duration {d}: {key}")
            hits.append((name, d / 1000.0))
    return hits


def profile_pair(
    fa_step: Callable[[], None],
    base_step: Callable[[], None],
    *,
    warmup: int,
    repeat: int,
    prof_root: Path,
    fa_pattern: str = DEFAULT_FA_PATTERN,
    base_pattern: str = DEFAULT_BASE_PATTERN,
    sync: Callable[[], None] = torch.npu.synchronize,
) -> dict[str, Any]:
    """Alternating FA/CANN in one profiler session.

    Mean time is total matched kernel duration divided by repeat.
    Individual iteration boundaries are not inferred from CSV rows.
    """
    if warmup < 1 or repeat < 1:
        raise ValueError("warmup and repeat must be >= 1")
    if prof_root.exists():
        raise FileExistsError(f"Profiler directory already exists: {prof_root}")

    fa_re = re.compile(fa_pattern, re.I)
    base_re = re.compile(base_pattern, re.I)

    sync()
    with build_profiler(prof_root, warmup, repeat) as prof:
        for _ in range(warmup + repeat):
            fa_step()
            base_step()
            sync()
            prof.step()
        prof.step()

    rows, kernel_csv = _load_kernels(prof_root)
    fa_hits = _select_kernel(rows, fa_re)
    base_hits = _select_kernel(rows, base_re)

    if not fa_hits or not base_hits:
        inventory = sorted({f"{r.get('Name', '')} | {r.get('Type', '')}" for r in rows})
        raise RuntimeError(
            f"Kernel matching failed: FA={len(fa_hits)}, "
            f"CANN={len(base_hits)}; CSV={kernel_csv}; "
            f"inventory={inventory[:80]}"
        )

    results = {}
    for key, hits in (("fa", fa_hits), ("baseline", base_hits)):
        total_ms = sum(duration for _, duration in hits)
        results[key] = {
            "mean_ms": total_ms / repeat,
            "kernels": len(hits),
            "kernels_per_iter": len(hits) / repeat,
        }

    results["kernel_csv_last"] = str(kernel_csv)
    return results


def approx_attention_flops(
    batch: int,
    seqlens_q: list[int],
    seqlens_k: list[int],
    heads: int,
    dim: int,
    causal: bool,
    mode: str,
) -> float:
    """Approximate QK^T + PV FLOPs, assuming square causal attention.

    For equal-length sequences, causal triangle has s*(s+1)/2 valid pairs.
    For general cross-attention, noncausal approximation uses s_q*s_k.
    """
    if len(seqlens_q) != len(seqlens_k):
        raise ValueError("Sequence lists have different sizes")
    pairs = 0
    for sq, sk in zip(seqlens_q, seqlens_k):
        if causal:
            if sq != sk:
                raise ValueError("causal cross-attention FLOPs not supported")
            pairs += sq * (sq + 1) / 2
        else:
            pairs += sq * sk
    forward = 4.0 * heads * dim * pairs
    return forward * (3.5 if mode == "fwd_bwd" else 1.0)


def metrics(flops: float, duration_ms: float | None) -> float | None:
    return flops / (duration_ms * 1e-3) / 1e12 if duration_ms and duration_ms > 0 else None


def write_results(
    rows: list[dict[str, Any]], csv_path: str | None, json_path: str | None, config: dict
):
    if csv_path:
        target = Path(csv_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
        if not fieldnames:
            fieldnames = ["status", "error"]
        with target.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    if json_path:
        target = Path(json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(
                {"config": config, "rows": rows}, f, indent=2, ensure_ascii=False, allow_nan=False
            )
