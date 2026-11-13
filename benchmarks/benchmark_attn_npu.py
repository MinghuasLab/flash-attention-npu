#!/usr/bin/env python3
# Copyright (c) 2026, Minghua Shen.

"""FA2/FA3/FA4 vs CANN FusionAttention on Ascend NPU.

BSND/TND, causal/noncausal, training FWD and FWD+BWD.
Default mode runs the original BSND grid and original TND random-varlen cases,
merging results into a single CSV/JSON. Optional style flags run only one suite.
Kernel duration sums are device-kernel work estimates, not API wall latency.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import math
import os
import platform
import random
import subprocess
import sys
import traceback
import uuid
import time
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
    _load_kernels,
    _select_kernel,
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
    p.add_argument("--nheads", type=int, default=24)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--headdim", type=int, default=128)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tnd-test-style",
        action="store_true",
        help="Use tnd_test.py workload and third-iteration profiling protocol",
    )
    p.add_argument("--num-cases", type=int, default=30, help="Balanced random TND cases")
    p.add_argument(
        "--bsnd-test-style",
        action="store_true",
        help="Run the BSND reference grid with BNSD CANN baseline",
    )
    p.add_argument("--bsnd-batch", type=int, nargs="+", default=[1, 4, 16, 64, 256, 320, 1024])
    p.add_argument("--bsnd-seqlen", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    p.add_argument(
        "--bsnd-softcap",
        type=float,
        default=0.0,
        help="FA softcap (must be supported for fair comparison)",
    )
    p.add_argument("--profile-root", default="/tmp/fa_npu_benchmark_profiles")
    p.add_argument("--fa-kernel-pattern", default=DEFAULT_FA_PATTERN)
    p.add_argument("--base-kernel-pattern", default=DEFAULT_BASE_PATTERN)
    p.add_argument("--commit-id", default=os.getenv("CI_BENCHMARK_COMMIT_SHA", ""))
    p.add_argument("--out", default="/tmp/fa_npu_benchmark.csv")
    p.add_argument("--json", default="/tmp/fa_npu_benchmark.json")
    args = p.parse_args()
    if any(v <= 0 for v in (args.nheads, args.headdim, args.warmup, args.repeat)):
        p.error("nheads, headdim, warmup, repeat must be positive")
    if args.kv_heads <= 0 or args.nheads % args.kv_heads:
        p.error("--kv-heads must be positive and divide --nheads")
    if args.tnd_test_style and args.bsnd_test_style:
        p.error("--tnd-test-style and --bsnd-test-style are mutually exclusive")
    if any(x <= 0 for x in (*args.bsnd_batch, *args.bsnd_seqlen)):
        p.error("--bsnd-batch and --bsnd-seqlen must be positive")
    if args.bsnd_softcap != 0.0:
        p.error(
            "--bsnd-softcap requires an equivalent CANN softcap implementation; only 0 supported"
        )
    if args.num_cases < 3:
        p.error("--num-cases must be at least 3")
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
    scale = data.get("scale", 1.0 / math.sqrt(data["dim"]))
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
        cu_seqlens_q=data["cu_q"],
        cu_seqlens_k=data["cu_k"],
        max_seqlen_q=data["max_seq"],
        max_seqlen_k=data["max_seq"],
        causal=causal,
        dropout_p=0.0,
        softmax_scale=scale,
        **({"window_size": (-1, -1)} if data.get("tnd_test_style") else {}),
    )


def baseline_forward(data, causal: bool):
    q, k, v = data["base_qkv"]
    scale = data.get("scale", 1.0 / math.sqrt(data["dim"]))
    kwargs = dict(scale=scale, keep_prob=1.0)
    if causal:
        if data["layout"] == "tnd":
            # TND requires actual maximum sequence length.
            mask_size = max(data["lengths"])
        else:
            mask_size = data["max_seq"]

        kwargs["atten_mask"] = data.get("atten_mask")
        if kwargs["atten_mask"] is None:
            kwargs["atten_mask"] = torch.triu(
                torch.ones((mask_size, mask_size), dtype=torch.bool, device=q.device), diagonal=1
            )
    if data["layout"] == "tnd":
        kwargs.update(actual_seq_qlen=data["cumul"], actual_seq_kvlen=data["cumul"])
    if data.get("bsnd_test_style"):
        q, k, v = data["base_qkv"]
        kwargs["atten_mask"] = data["atten_mask"] if causal else None
        out = torch_npu.npu_fusion_attention(q, k, v, data["heads"], "BNSD", **kwargs)
        return out[0] if isinstance(out, (tuple, list)) else out
    if data.get("tnd_test_style"):
        # sparse_mode=3 with next_tockens=0 is the original causal TND setup.
        # The noncausal path must use dense/unmasked attention instead.
        kwargs.update(
            pse=None,
            padding_mask=None,
            pre_tockens=65536,
            next_tockens=0 if causal else 65536,
            inner_precise=0,
            sparse_mode=3 if causal else 0,
        )
    out = torch_npu.npu_fusion_attention(q, k, v, data["heads"], data["layout"].upper(), **kwargs)
    return out[0] if isinstance(out, (tuple, list)) else out


# tnd_test.py-inspired, opt-in analysis protocol. Does not affect default CI.
TND_NUM_SEQS = 320
TND_MIN_SL = 32
TND_MAX_SL = 512
TND_AVG_SL = 150
TND_MASK_SIZE = 2048


def _tnd_random_lengths():
    return [
        max(TND_MIN_SL, min(TND_MAX_SL, int(random.gauss(TND_AVG_SL, TND_AVG_SL * 0.25))))
        for _ in range(TND_NUM_SEQS)
    ]


def _tnd_balanced_cases(n):
    """Draw candidate batches, then sample evenly by aggregate token count."""
    pool = []
    for _ in range(max(n * 50, 2000)):
        lengths = _tnd_random_lengths()
        total = sum(lengths)
        pool.append(
            {
                "lengths": lengths,
                "total_tokens": total,
                "min_sl": min(lengths),
                "max_sl": max(lengths),
                "avg_sl": round(total / TND_NUM_SEQS, 1),
            }
        )
    pool.sort(key=lambda case: case["total_tokens"])
    indexes = [round(i * (len(pool) - 1) / (n - 1)) for i in range(n)]
    cases = [pool[int(i)] for i in indexes]
    for idx, case in enumerate(cases, 1):
        case["case_id"] = idx
        case["label"] = f"am_{idx:02d}"
    return cases


def _tnd_cutoffs(cases):
    totals = sorted(c["total_tokens"] for c in cases)
    return totals[len(totals) // 3 - 1], totals[2 * len(totals) // 3 - 1]


def _tnd_bucket(total, cutoffs):
    return "small" if total <= cutoffs[0] else "medium" if total <= cutoffs[1] else "large"


def _tnd_inputs(lengths, device, atten_mask):
    """Same TND shape and sequence-offset convention as tnd_test.py."""
    total = sum(lengths)
    shapes = [(total, 24, 128), (total, 4, 128), (total, 4, 128)]
    sources = [torch.randn(shape, device=device, dtype=torch.bfloat16) for shape in shapes]
    # Keep independent inputs for future backward/correctness checks.
    fa_qkv = tuple(t.detach().clone().requires_grad_(True) for t in sources)
    base_qkv = tuple(t.detach().clone().requires_grad_(True) for t in sources)
    cumul = []
    for length in lengths:
        cumul.append((cumul[-1] if cumul else 0) + length)
    cu = torch.tensor([0, *cumul], dtype=torch.int32, device=device)
    return {
        "layout": "tnd",
        "heads": 24,
        "dim": 128,
        "fa_qkv": fa_qkv,
        "base_qkv": base_qkv,
        "max_seq": max(lengths),
        "lengths": lengths,
        "cumul": tuple(cumul),
        "cu_q": cu,
        "cu_k": cu,
        "scale": 1.0,
        "atten_mask": atten_mask,
        "tnd_test_style": True,
    }


def _profile_tnd_third(fa_step, base_step, root, fa_pattern, base_pattern):
    """Two unprofiled rounds, then profile the third pair only."""
    import re

    for _ in range(2):
        fa_step()
        base_step()
    torch.npu.synchronize()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        data_simplification=False,
        l2_cache=False,
    )
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=0,
            active=1,
            repeat=1,
        ),
        record_shapes=False,
        profile_memory=False,
        with_modules=False,
        experimental_config=config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(root),
            analyse_flag=False,
        ),
    )

    t0 = time.perf_counter()

    with prof:
        t1 = time.perf_counter()

        fa_step()
        t2 = time.perf_counter()

        base_step()
        t3 = time.perf_counter()

        torch.npu.synchronize()
        t4 = time.perf_counter()

        prof.step()
        t5 = time.perf_counter()

    t6 = time.perf_counter()

    print(
        "[profiler-stage] "
        f"enter={t1 - t0:.3f}s "
        f"fa_submit={t2 - t1:.3f}s "
        f"cann_submit={t3 - t2:.3f}s "
        f"sync={t4 - t3:.3f}s "
        f"step={t5 - t4:.3f}s "
        f"exit={t6 - t5:.3f}s",
        flush=True,
    )
    t_parse_start = time.perf_counter()

    try:
        rows, kernel_csv = _load_kernels(root)
    finally:
        print(
            f"[profiler-stage] parse={time.perf_counter() - t_parse_start:.3f}s",
            flush=True,
        )

    fa_hits = _select_kernel(rows, re.compile(fa_pattern, re.I))
    base_hits = _select_kernel(rows, re.compile(base_pattern, re.I))
    if not fa_hits or not base_hits:
        raise RuntimeError(
            f"No matching kernels: FA={len(fa_hits)}, CANN={len(base_hits)}; see {kernel_csv}"
        )
    return {
        "fa_mean_ms": sum(duration for _, duration in fa_hits),
        "baseline_mean_ms": sum(duration for _, duration in base_hits),
        "fa_kernels": len(fa_hits),
        "baseline_kernels": len(base_hits),
        "fa_kernels_per_iter": len(fa_hits),
        "baseline_kernels_per_iter": len(base_hits),
        "kernel_csv_last": str(kernel_csv),
    }


def _is_npu_oom(exc):
    """识别 PyTorch / torch_npu 显存不足异常。"""
    if isinstance(exc, (torch.OutOfMemoryError, MemoryError)):
        return True

    message = str(exc).lower()

    return any(
        token in message
        for token in (
            "out of memory",
            "memory allocation failed",
            "npu memory is insufficient",
            "acl error 207001",
            "rt_error_memory_allocation",
        )
    )


def _bsnd_precheck_oom(device_name, batch, seqlen, mode):
    """跳过日志中已确认显存不足的 BSND 组合。"""

    if mode != "fwd_bwd":
        return None

    # Ascend 950
    if "Ascend950" in device_name:
        if batch >= 1024 and seqlen >= 2048:
            return "Known Ascend950 OOM: B>=1024, S>=2048"

    # Ascend 910
    if "Ascend910" in device_name:
        if batch >= 1024 and seqlen >= 1024:
            return "Known Ascend910 OOM: B>=1024, S>=1024"

        if batch >= 320 and seqlen >= 2048:
            return "Known Ascend910 OOM: B>=320, S>=2048"

    return None


def _mark_case_exception(row, exc):
    """显存不足跳过，其他异常仍记录为 error。"""

    if _is_npu_oom(exc):
        row.update(
            status="skipped_oom",
            error=f"{type(exc).__name__}: {exc}",
        )

        try:
            torch.npu.empty_cache()
        except Exception:
            pass

    else:
        row.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
        traceback.print_exc()


def _run_tnd_test_style(args, versions, device_name, commit, run_root, rows_out=None):
    """Balanced varlen cases, crossed with requested causal and training modes.

    Each result measures the third FA+CANN invocation pair (two warmups),
    preserving the original TND profiling protocol for comparability.
    """
    case_start = int(os.environ.get("TND_CASE_START", "0"))
    case_stop = int(os.environ.get("TND_CASE_STOP", str(args.num_cases)))

    if not (0 <= case_start < case_stop <= args.num_cases):
        raise ValueError(
            f"Invalid TND case range: [{case_start}, {case_stop}), num_cases={args.num_cases}"
        )

    device = f"npu:{args.device}"

    random.seed(args.seed)

    # 每个进程先生成相同的完整 Case 集合
    cases = _tnd_balanced_cases(args.num_cases)

    # 分桶仍基于完整集合，保持不同批次的分类一致
    cutoffs = _tnd_cutoffs(cases)

    # 只执行当前批次
    cases = cases[case_start:case_stop]

    print(
        f"[benchmark] TND batch: am_{case_start + 1:02d} to am_{case_stop:02d}",
        flush=True,
    )
    modes = ["fwd", "fwd_bwd"] if args.mode == "all" else [args.mode]
    causals = [False, True] if args.causal == "both" else [args.causal == "true"]
    rows = []
    for version in versions:
        try:
            backend, backend_error = load_backend(version), None
        except Exception as exc:
            backend, backend_error = None, f"{type(exc).__name__}: {exc}"
        for causal in causals:
            # Reuse the same balanced case shapes across all variants.
            mask = (
                torch.triu(
                    torch.ones((TND_MASK_SIZE, TND_MASK_SIZE), dtype=torch.bool, device=device),
                    diagonal=1,
                )
                if causal
                else None
            )
            for mode in modes:
                for case in cases:
                    row = {
                        "commit_id": commit,
                        "device_name": device_name,
                        "host_arch": platform.machine(),
                        "torch_version": torch.__version__,
                        "torch_npu_version": torch_npu.__version__,
                        "cann_home": os.getenv("ASCEND_HOME_PATH", ""),
                        "suite": "tnd_reference",
                        "fa_version": version,
                        "layout": "tnd",
                        "baseline_layout": "tnd",
                        "mode": mode,
                        "causal": causal,
                        "batch": TND_NUM_SEQS,
                        "seqlen": TND_MAX_SL,
                        "nheads": 24,
                        "kv_heads": 4,
                        "headdim": 128,
                        "dtype": "bf16",
                        "warmup": 2,
                        "repeat": 1,
                        "case_id": case["case_id"],
                        "label": case["label"],
                        "total_tokens": case["total_tokens"],
                        "min_sl": case["min_sl"],
                        "max_sl": case["max_sl"],
                        "avg_sl": case["avg_sl"],
                        "bucket": _tnd_bucket(case["total_tokens"], cutoffs),
                        "timing_scope": "matched_device_kernel_work_at_third_iteration",
                        "status": "ok",
                        "error": "",
                    }
                    try:
                        if backend_error:
                            raise RuntimeError(backend_error)
                        data = _tnd_inputs(case["lengths"], device, mask)

                        def fa_call():
                            return backend_forward(backend, data, causal)

                        def base_call():
                            return baseline_forward(data, causal)

                        if mode == "fwd":

                            def fa_step():
                                step_forward(fa_call)

                            def base_step():
                                step_forward(base_call)
                        else:
                            # Same upstream gradient for FA and CANN, separate leaf tensors.
                            grad = torch.randn(
                                (case["total_tokens"], 24, 128),
                                device=device,
                                dtype=torch.bfloat16,
                            )

                            def fa_step():
                                step_fwd_bwd(fa_call, data["fa_qkv"], grad)

                            def base_step():
                                step_fwd_bwd(base_call, data["base_qkv"], grad)

                        slug = f"{version}_{mode}_c{int(causal)}_{case['label']}"
                        # 只对 Ascend 910B 启用 TND Profiler 有限重试
                        is_910 = "Ascend910" in device_name
                        max_attempts = 2 if is_910 else 1

                        for attempt in range(max_attempts):
                            # 910B 每次采集使用独立目录，避免复用失败数据
                            # 950 保持原来的目录结构与调用次数
                            profile_root = (
                                run_root / slug / f"attempt_{attempt + 1}"
                                if is_910
                                else run_root / slug
                            )

                            try:
                                measured = _profile_tnd_third(
                                    fa_step,
                                    base_step,
                                    profile_root,
                                    args.fa_kernel_pattern,
                                    args.base_kernel_pattern,
                                )

                                if attempt > 0:
                                    print(
                                        f"[tnd-profiler-retry] recovered: "
                                        f"{slug}, attempt={attempt + 1}",
                                        flush=True,
                                    )

                                break

                            except RuntimeError as exc:
                                error_message = str(exc)

                                # 仅允许 Profiler 数据缺失触发一次重试
                                is_profiler_missing = (
                                    "kernel_details.csv not found" in error_message
                                )

                                if is_910 and attempt == 0 and is_profiler_missing:
                                    print(
                                        f"[tnd-profiler-retry] missing kernel data: "
                                        f"{slug}; retrying once",
                                        flush=True,
                                    )

                                    torch.npu.synchronize()
                                    time.sleep(1)
                                    continue

                                # 第二次仍失败，或属于其他异常，则维持原来的错误处理
                                raise

                        # 记录实际重试次数：0 表示第一次成功，1 表示重试成功
                        row["profiler_retry_count"] = attempt
                        fa_ms = measured["fa_mean_ms"]
                        base_ms = measured["baseline_mean_ms"]
                        row.update(measured)
                        row.update(
                            delta_ms=fa_ms - base_ms,
                            ratio_fa_over_cann=fa_ms / base_ms if base_ms > 0 else None,
                            degrade_pct=(fa_ms / base_ms - 1.0) * 100 if base_ms > 0 else None,
                            speedup_fa_over_baseline=base_ms / fa_ms if fa_ms > 0 else None,
                        )
                    except NotImplementedError as exc:
                        row.update(status="unsupported", error=str(exc))
                    except Exception as exc:
                        _mark_case_exception(row, exc)
                    rows.append(row)
                    print(
                        f"[{row['status']}] {version} {mode} causal={causal} "
                        f"{case['label']} tokens={case['total_tokens']} "
                        f"fa_ms={row.get('fa_mean_ms')} "
                        f"cann_ms={row.get('baseline_mean_ms')} {row['error']}",
                        flush=True,
                    )
    if rows_out is not None:
        rows_out.extend(rows)
    else:
        write_results(rows, args.out, args.json, vars(args))
        print(f"CSV: {args.out}; JSON: {args.json}; profiles: {run_root}")
    if any(r["status"] == "error" for r in rows):
        return 1
    return 0 if any(r["status"] == "ok" for r in rows) else 2


def _run_bsnd_test_style(args, versions, device_name, commit, run_root, rows_out=None):
    """Reference BSND test grid; CANN receives BNSD transposed inputs.

    Preserve the shared profile_pair single-session aggregate timing protocol.
    The reference script's mean/min over *individual kernels* is not used,
    because it is not an iteration-level FWD+BWD metric.
    """
    rows = []
    device = f"npu:{args.device}"
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    for version in versions:
        try:
            backend, backend_error = load_backend(version), None
        except Exception as exc:
            backend, backend_error = None, f"{type(exc).__name__}: {exc}"
        for causal in [False, True] if args.causal == "both" else [args.causal == "true"]:
            for batch in args.bsnd_batch:
                for seqlen in args.bsnd_seqlen:
                    for mode in ["fwd", "fwd_bwd"] if args.mode == "all" else [args.mode]:
                        row = {
                            "commit_id": commit,
                            "device_name": device_name,
                            "host_arch": platform.machine(),
                            "torch_version": torch.__version__,
                            "torch_npu_version": torch_npu.__version__,
                            "cann_home": os.getenv("ASCEND_HOME_PATH", ""),
                            "suite": "bsnd_reference",
                            "fa_version": version,
                            "layout": "bsnd",
                            "baseline_layout": "bnsd",
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
                            "softcap": 0.0,
                            "timing_scope": "mean_matched_device_kernel_work_per_iteration",
                            "status": "ok",
                            "error": "",
                        }
                        try:
                            skip_reason = _bsnd_precheck_oom(device_name, batch, seqlen, mode)

                            if skip_reason:
                                row.update(
                                    status="skipped_oom",
                                    error=skip_reason,
                                )

                                rows.append(row)

                                print(
                                    f"[skipped_oom] BSND {version} {mode} "
                                    f"c={causal} B={batch} S={seqlen}: {skip_reason}",
                                    flush=True,
                                )

                                continue

                            if backend_error:
                                raise RuntimeError(backend_error)
                            # Match reference: FA is BSHD, CANN receives BNSD.
                            torch.manual_seed(args.seed)
                            qs = (batch, seqlen, args.nheads, args.headdim)
                            ks = (batch, seqlen, args.kv_heads, args.headdim)
                            sources = [
                                torch.randn(shape, device=device, dtype=dtype).contiguous()
                                for shape in (qs, ks, ks)
                            ]
                            fa_qkv = tuple(t.detach().clone().requires_grad_(True) for t in sources)
                            base_qkv = tuple(
                                t.transpose(1, 2).contiguous().detach().requires_grad_(True)
                                for t in sources
                            )
                            mask = (
                                torch.triu(
                                    torch.ones((seqlen, seqlen), device=device, dtype=torch.bool),
                                    diagonal=1,
                                )
                                if causal
                                else None
                            )
                            data = {
                                "layout": "bsnd",
                                "bsnd_test_style": True,
                                "heads": args.nheads,
                                "dim": args.headdim,
                                "fa_qkv": fa_qkv,
                                "base_qkv": base_qkv,
                                "scale": args.headdim**-0.5,
                                "atten_mask": mask,
                                "lengths": [seqlen] * batch,
                                "max_seq": seqlen,
                            }
                            grad = torch.randn(qs, device=device, dtype=dtype)
                            grad_base = grad.transpose(1, 2).contiguous()

                            def fa_call():
                                return backend_forward(backend, data, causal)

                            def base_call():
                                return baseline_forward(data, causal)

                            if mode == "fwd":

                                def fa_step():
                                    step_forward(fa_call)

                                def base_step():
                                    step_forward(base_call)
                            else:

                                def fa_step():
                                    step_fwd_bwd(fa_call, data["fa_qkv"], grad)

                                def base_step():
                                    step_fwd_bwd(base_call, data["base_qkv"], grad_base)

                            base_step()
                            torch.npu.synchronize()
                            # 每个 BSND Case 的 Profiler 根目录
                            profile_path = (
                                run_root
                                / f"{version}_bsnd_{mode}_c{int(causal)}_b{batch}_s{seqlen}"
                            )

                            # Profiler 数据缺失时最多重试一次
                            max_attempts = 2 if "Ascend910" in device_name else 1

                            for attempt in range(max_attempts):
                                try:
                                    prof = profile_pair(
                                        fa_step,
                                        base_step,
                                        warmup=args.warmup,
                                        repeat=args.repeat,
                                        prof_root=profile_path / f"attempt_{attempt + 1}",
                                        fa_pattern=args.fa_kernel_pattern,
                                        base_pattern=args.base_kernel_pattern,
                                    )

                                    if attempt > 0:
                                        print(
                                            f"[profiler-retry] recovered: "
                                            f"{version} {mode} causal={causal} "
                                            f"B={batch} S={seqlen}",
                                            flush=True,
                                        )

                                    break

                                except RuntimeError as exc:
                                    error_message = str(exc)

                                    # 只对 Profiler Kernel CSV 缺失进行重试
                                    if (
                                        attempt == 0
                                        and "kernel_details.csv not found" in error_message
                                    ):
                                        print(
                                            f"[profiler-retry] missing kernel data: "
                                            f"{version} {mode} causal={causal} "
                                            f"B={batch} S={seqlen}; "
                                            f"retrying once",
                                            flush=True,
                                        )

                                        torch.npu.synchronize()
                                        time.sleep(1)
                                        continue

                                    # 第二次仍失败，或属于其他错误，则交给外层处理
                                    raise

                            # 保留原有性能计算逻辑
                            fa, base = prof["fa"], prof["baseline"]
                            row["profiler_retry_count"] = attempt
                            flops = approx_attention_flops(
                                batch,
                                data["lengths"],
                                data["lengths"],
                                args.nheads,
                                args.headdim,
                                causal,
                                mode,
                            )
                            row.update(
                                {
                                    "approx_flops": flops,
                                    "fa_tflops": metrics(flops, fa["mean_ms"]),
                                    "baseline_tflops": metrics(flops, base["mean_ms"]),
                                    "speedup_fa_over_baseline": (
                                        base["mean_ms"] / fa["mean_ms"]
                                        if fa["mean_ms"] > 0
                                        else None
                                    ),
                                    "kernel_csv_last": prof["kernel_csv_last"],
                                }
                            )
                            for label, vals in (("fa", fa), ("baseline", base)):
                                for field in ("mean_ms", "kernels", "kernels_per_iter"):
                                    row[f"{label}_{field}"] = vals[field]
                        except NotImplementedError as exc:
                            row.update(status="unsupported", error=str(exc))
                        except Exception as exc:
                            _mark_case_exception(row, exc)
                        rows.append(row)
                        print(
                            f"[{row['status']}] BSND {version} {mode} c={causal} "
                            f"B={batch} S={seqlen} FA={row.get('fa_mean_ms')} "
                            f"CANN={row.get('baseline_mean_ms')} {row['error']}",
                            flush=True,
                        )
    if rows_out is not None:
        rows_out.extend(rows)
    else:
        write_results(rows, args.out, args.json, vars(args))
        print(f"CSV: {args.out}; JSON: {args.json}; profiles: {run_root}")
    return (
        1
        if any(r["status"] == "error" for r in rows)
        else (0 if any(r["status"] == "ok" for r in rows) else 2)
    )


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
    # The current FA2 entry point rejects Ascend 950; exclude it from "all".
    if "Ascend950" in device_name:
        if args.fa_version == "fa2":
            raise SystemExit("FA2 is disabled on Ascend 950 in this CI configuration")
        versions = [version for version in versions if version != "fa2"]
        print(f"[benchmark] Ascend 950: selected FA versions: {', '.join(versions)}")
    run_root = Path(args.profile_root) / f"run_{uuid.uuid4().hex[:12]}"
    rows = []
    do_bsnd = args.layout in ("all", "bsnd") and not args.tnd_test_style
    do_tnd = args.layout in ("all", "tnd") and not args.bsnd_test_style
    if not do_bsnd and not do_tnd:
        raise SystemExit("selected --layout conflicts with the selected --*-test-style option")

    if do_bsnd:
        print("[benchmark] BSND reference: fixed B x S grid", flush=True)
        _run_bsnd_test_style(args, versions, device_name, commit, run_root / "bsnd", rows_out=rows)
    if do_tnd:
        print(
            f"[benchmark] TND reference: {args.num_cases} balanced random-varlen cases", flush=True
        )
        _run_tnd_test_style(args, versions, device_name, commit, run_root / "tnd", rows_out=rows)

    write_results(rows, args.out, args.json, vars(args))
    counts = {
        status: sum(r["status"] == status for r in rows)
        for status in (
            "ok",
            "unsupported",
            "skipped_oom",
            "error",
        )
    }
    print(f"[benchmark] cases={len(rows)} status={counts}", flush=True)
    print(f"CSV: {args.out}; JSON: {args.json}; profiles: {run_root}", flush=True)
    return 1 if counts["error"] else (0 if counts["ok"] else 2)


if __name__ == "__main__":
    sys.exit(main())
