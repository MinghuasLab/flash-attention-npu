#!/usr/bin/env python3
"""FA2/FA3/FA4 vs CANN FusionAttention on Ascend NPU.

BSND/TND, causal/noncausal, training FWD and FWD+BWD.
Each sample profiles one FA step and one CANN step in the same profiler window.
Kernel duration sums are device-kernel work estimates, not API wall latency.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import math
import os
import platform
import subprocess
import sys
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
import torch_npu

from benchmark_utils import (
    DEFAULT_FA_PATTERN,
    DEFAULT_BASE_PATTERN,
    step_forward,
    step_fwd_bwd,
    profile_pair,
    approx_attention_flops,
    metrics,
    write_results,
)

MODULES = {"fa2": "flash_attn_npu", "fa3": "flash_attn_npu_3", "fa4": "flash_attn_npu_4"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fa-version", choices=["all", *MODULES], default="all")
    p.add_argument("--layout", choices=["all", "bsnd", "tnd"], default="all")
    p.add_argument("--mode", choices=["all", "fwd", "fwd_bwd"], default="all")
    p.add_argument(
        "--causal",
        choices=["both", "true", "false"],
        default="both",
        help="Run both causal and noncausal by default",
    )
    p.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    p.add_argument("--seqlen", type=int, nargs="+", default=[128, 512])
    p.add_argument(
        "--min-seqlen",
        type=int,
        default=0,
        help="TND: minimum per-sequence length, 0 means equal seqlen",
    )
    p.add_argument("--nheads", type=int, default=24)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--headdim", type=int, default=128)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--profile-root", default="/tmp/fa_npu_benchmark_profiles")
    p.add_argument("--fa-kernel-pattern", default=DEFAULT_FA_PATTERN)
    p.add_argument("--base-kernel-pattern", default=DEFAULT_BASE_PATTERN)
    p.add_argument("--commit-id", default=os.getenv("CI_BENCHMARK_COMMIT_SHA", ""))
    p.add_argument("--out", default="/tmp/fa_npu_benchmark.csv")
    p.add_argument("--json", default="/tmp/fa_npu_benchmark.json")
    args = p.parse_args()
    if any(
        v <= 0
        for v in [*args.batch, *args.seqlen, args.nheads, args.headdim, args.warmup, args.repeat]
    ):
        p.error("batch, seqlen, nheads, headdim, warmup, repeat must be positive")
    if args.kv_heads <= 0 or args.nheads % args.kv_heads:
        p.error("--kv-heads must be positive and divide --nheads")
    if args.min_seqlen < 0:
        p.error("--min-seqlen must be >= 0")
    return args


def _call_supported(fn, *positional, **kwargs):
    """Adapt optional signature differences; never silently discard causal."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return fn(*positional, **kwargs)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*positional, **kwargs)
    accepted = {key: val for key, val in kwargs.items() if key in params}
    if "causal" in kwargs and "causal" not in accepted:
        raise NotImplementedError(f"{fn} lacks causal parameter; refusing invalid comparison")
    return fn(*positional, **accepted)


@dataclass
class Backend:
    version: str
    dense: object
    varlen: object


def load_backend(version):
    module = importlib.import_module(MODULES[version])
    dense = getattr(module, "flash_attn_func", None)
    varlen = getattr(module, "flash_attn_varlen_func", None)
    if dense is None:
        raise NotImplementedError(f"{version} has no flash_attn_func")
    return Backend(version, dense, varlen)


def backend_forward(backend: Backend, data, causal: bool):
    scale = 1.0 / math.sqrt(data["dim"])
    if data["layout"] == "bsnd":
        return _call_supported(
            backend.dense, *data["fa_qkv"], causal=causal, dropout_p=0.0, softmax_scale=scale
        )
    if backend.varlen is None:
        raise NotImplementedError(f"{backend.version} has no flash_attn_varlen_func")
    q, k, v = data["fa_qkv"]
    return _call_supported(
        backend.varlen,
        q,
        k,
        v,
        data["cu_q"],
        data["cu_k"],
        data["max_seq"],
        data["max_seq"],
        causal=causal,
        dropout_p=0.0,
        softmax_scale=scale,
    )


def baseline_forward(data, causal: bool):
    q, k, v = data["base_qkv"]
    scale = 1.0 / math.sqrt(data["dim"])
    kwargs = dict(scale=scale, keep_prob=1.0)
    if causal:
        if data["layout"] == "tnd":
            # TND requires actual maximum sequence length.
            mask_size = max(data["lengths"])
        else:
            mask_size = data["max_seq"]

        kwargs["atten_mask"] = torch.triu(
            torch.ones((mask_size, mask_size), dtype=torch.bool, device=q.device), diagonal=1
        )
    if data["layout"] == "tnd":
        kwargs.update(actual_seq_qlen=data["cumul"], actual_seq_kvlen=data["cumul"])
    out = torch_npu.npu_fusion_attention(q, k, v, data["heads"], data["layout"].upper(), **kwargs)
    return out[0] if isinstance(out, (tuple, list)) else out


def _gen_seq_lengths(batch, seqlen, min_seqlen, layout, seed):
    if layout != "tnd" or not min_seqlen:
        return [seqlen] * batch
    if min_seqlen > seqlen:
        raise ValueError("--min-seqlen cannot exceed seqlen")
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(min_seqlen, seqlen + 1, (batch,), generator=g).tolist()


def make_inputs(args, batch, seqlen, layout, causal):
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = f"npu:{args.device}"
    torch.manual_seed(args.seed)
    lengths = _gen_seq_lengths(batch, seqlen, args.min_seqlen, layout, args.seed)
    total = sum(lengths)
    shape_q = (
        (batch, seqlen, args.nheads, args.headdim)
        if layout == "bsnd"
        else (total, args.nheads, args.headdim)
    )
    shape_k = (
        (batch, seqlen, args.kv_heads, args.headdim)
        if layout == "bsnd"
        else (total, args.kv_heads, args.headdim)
    )
    sources = [torch.randn(s, device=device, dtype=dtype) for s in (shape_q, shape_k, shape_k)]
    fa_qkv = tuple(t.detach().clone().requires_grad_(True) for t in sources)
    base_qkv = tuple(t.detach().clone().requires_grad_(True) for t in sources)
    grad = torch.randn(shape_q, device=device, dtype=dtype)
    cumul = []
    acc = 0
    for length in lengths:
        acc += length
        cumul.append(acc)
    cu = torch.tensor([0] + cumul, dtype=torch.int32, device=device)
    return {
        "layout": layout,
        "heads": args.nheads,
        "dim": args.headdim,
        "fa_qkv": fa_qkv,
        "base_qkv": base_qkv,
        "grad": grad,
        "max_seq": seqlen,
        "lengths": lengths,
        "cumul": cumul,
        "cu_q": cu,
        "cu_k": cu,
        "causal": causal,
    }


def _run_one(args, backend, batch, seqlen, layout, causal, mode, case_root):
    data = make_inputs(args, batch, seqlen, layout, causal)

    def fa_call():
        return backend_forward(backend, data, causal)

    def base_call():
        return baseline_forward(data, causal)

    if mode == "fwd":

        def fa_step():
            return step_forward(fa_call)

        def base_step():
            return step_forward(base_call)
    else:

        def fa_step():
            return step_fwd_bwd(fa_call, data["fa_qkv"], data["grad"])

        def base_step():
            return step_fwd_bwd(base_call, data["base_qkv"], data["grad"])

    # Fail early before starting profiler; unsupported backward is tested by execution.
    fa_step()
    base_step()
    torch.npu.synchronize()
    prof = profile_pair(
        fa_step,
        base_step,
        warmup=args.warmup,
        repeat=args.repeat,
        prof_root=case_root,
        fa_pattern=args.fa_kernel_pattern,
        base_pattern=args.base_kernel_pattern,
    )
    approx = approx_attention_flops(
        batch, data["lengths"], data["lengths"], args.nheads, args.headdim, causal, mode
    )
    fa, base = prof["fa"], prof["baseline"]
    out = {
        "approx_flops": approx,
        "fa_tflops": metrics(approx, fa["mean_ms"]),
        "baseline_tflops": metrics(approx, base["mean_ms"]),
        "speedup_fa_over_baseline": base["mean_ms"] / fa["mean_ms"] if fa["mean_ms"] > 0 else None,
        "kernel_csv_last": prof["kernel_csv_last"],
    }
    for key, dat in [("fa", fa), ("baseline", base)]:
        for field in ["mean_ms", "min_ms", "max_ms", "stdev_ms", "kernels", "kernels_per_iter"]:
            out[f"{key}_{field}"] = dat[field]
    return out


def main():
    args = parse_args()
    torch.npu.set_device(args.device)
    device_name = torch.npu.get_device_name(args.device)
    commit = args.commit_id
    if not commit:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            commit = "unknown"
    versions = list(MODULES) if args.fa_version == "all" else [args.fa_version]
    layouts = ["bsnd", "tnd"] if args.layout == "all" else [args.layout]
    modes = ["fwd", "fwd_bwd"] if args.mode == "all" else [args.mode]
    causals = [False, True] if args.causal == "both" else [args.causal == "true"]
    run_root = Path(args.profile_root) / f"run_{uuid.uuid4().hex[:12]}"
    rows = []
    for version in versions:
        try:
            backend = load_backend(version)
            backend_error = None
        except Exception as exc:
            backend = None
            backend_error = f"{type(exc).__name__}: {exc}"
        for layout in layouts:
            for causal in causals:
                for batch in args.batch:
                    for seqlen in args.seqlen:
                        for mode in modes:
                            row = {
                                "commit_id": commit,
                                "device_name": device_name,
                                "host_arch": platform.machine(),
                                "torch_version": torch.__version__,
                                "torch_npu_version": torch_npu.__version__,
                                "cann_home": os.getenv("ASCEND_HOME_PATH", ""),
                                "fa_version": version,
                                "layout": layout,
                                "mode": mode,
                                "causal": causal,
                                "batch": batch,
                                "seqlen": seqlen,
                                "nheads": args.nheads,
                                "kv_heads": args.kv_heads,
                                "headdim": args.headdim,
                                "dtype": args.dtype,
                                "warmup": args.warmup,
                                "repeat": args.repeat,
                                "timing_scope": "sum_of_matched_device_kernel_durations",
                                "status": "ok",
                                "error": "",
                            }
                            try:
                                if backend_error:
                                    raise RuntimeError(backend_error)
                                slug = f"{version}_{layout}_{mode}_causal{int(causal)}_b{batch}_s{seqlen}"
                                result = _run_one(
                                    args,
                                    backend,
                                    batch,
                                    seqlen,
                                    layout,
                                    causal,
                                    mode,
                                    run_root / slug,
                                )
                                row.update(result)
                            except NotImplementedError as exc:
                                row.update(status="unsupported", error=str(exc))
                            except Exception as exc:
                                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                                traceback.print_exc()
                            rows.append(row)
                            print(
                                f"[{row['status']}] {version} {layout} causal={causal} "
                                f"{mode} B={batch} S={seqlen} "
                                f"fa_ms={row.get('fa_mean_ms')} base_ms={row.get('baseline_mean_ms')} "
                                f"{row['error']}",
                                flush=True,
                            )
    write_results(rows, args.out, args.json, vars(args))
    print(f"CSV: {args.out}; JSON: {args.json}; profiles: {run_root}")
    if any(row["status"] == "error" for row in rows):
        return 1
    return 0 if any(row["status"] == "ok" for row in rows) else 2


if __name__ == "__main__":
    sys.exit(main())
