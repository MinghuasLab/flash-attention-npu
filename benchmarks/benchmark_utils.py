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
from pathlib import Path
from typing import Callable, Any

import torch
import torch_npu
from torch_npu.profiler.profiler import analyse

AUX_PATTERN = re.compile(r"(?:^|[_./])(Fill|ReduceSum|Cast)(?:$|[_./])", re.I)
DEFAULT_FA_PATTERN = r"SplitFuse|FAInfer|FAG"
DEFAULT_BASE_PATTERN = r"FlashAttention"


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


def step_fwd_bwd(fn: Callable[[], torch.Tensor], inputs, grad: torch.Tensor) -> None:
    clear_grads(inputs)
    out = fn()
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not isinstance(out, torch.Tensor) or not out.requires_grad:
        raise NotImplementedError("backend forward output does not support autograd")
    out.backward(grad)


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
    ascend_pt = _find_profile(root)
    # torch_npu's analyse signature varies by version; use supported args only.
    import inspect

    params = inspect.signature(analyse).parameters
    kwargs = {"profiler_path": str(ascend_pt)}
    if "max_process_number" in params:
        kwargs["max_process_number"] = 4
    if "export_type" in params:
        kwargs["export_type"] = "text"
    analyse(**kwargs)
    files = list(ascend_pt.rglob("kernel_details.csv"))
    if not files:
        raise RuntimeError(f"kernel_details.csv not found under {ascend_pt}")
    csv_file = max(files, key=lambda p: p.stat().st_mtime_ns)
    with csv_file.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f)), csv_file


def _select_kernel(rows, pattern: re.Pattern[str]):
    hits = []
    for row in rows:
        name = row.get("Name", "") or ""
        kind = row.get("Type", "") or ""
        # Prefer kernel Name. Type is a fallback when Name is not available.
        key = name if name else kind
        if AUX_PATTERN.search(name) or AUX_PATTERN.search(kind):
            continue
        if pattern.search(key):
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
            "min_ms": None,
            "max_ms": None,
            "stdev_ms": None,
            "kernels": len(hits),
            "kernels_per_iter": len(hits) / repeat,
            "samples_ms": None,
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
