"""Benchmark npu_fa / npu_fa_v1 on the exact OneRec inputs from bench.py.

Default: all five cases, dropout 0/0.1, forward and backward, 5 warmups and
50 measured iterations. Calls ops_benchmark.do_ops_bench without changing its
Event timing or 256 MiB cache-flush policy. Times include the Python wrappers'
device work and synchronization gaps; they are not profiler kernel durations.
Only the attention output participates in backward (not fusion auxiliary data).

Examples:
    python bench_onerec_ops.py
    python bench_onerec_ops.py --scenario t5_decoder --rep 5 --warmup 2
    python bench_onerec_ops.py --forward-only --dropout 0

Outputs: <output-prefix>.csv, .json (including every sample), and .md.
UniAttention must be installed, supplied via --uniattention-root, or present at
../uni-attn/UniAttention. Its dependencies include rich and tabulate.
No packages are installed by this script. For an isolated dependency directory:
    python -m pip install --target /tmp/onerec_ops_deps rich tabulate
    PYTHONPATH=/tmp/onerec_ops_deps python bench_onerec_ops.py
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import importlib
import importlib.util
import json
from pathlib import Path
from statistics import mean, median
import sys
import traceback


ROOT = Path(__file__).resolve().parent
BACKENDS = ("npu_fa", "npu_fa_v1")
INTERFACE_LABELS = {
    "npu_fa": "torch FA",
    "npu_fa_v1": "FA2",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", nargs="+", help="case names from bench.CASES; default: all")
    parser.add_argument("--backend", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--dropout", nargs="+", type=float, default=[0.0, 0.1])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--device", type=int, default=0, help="logical NPU index")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--uniattention-root", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "batchtest/onerec_ops")
    args = parser.parse_args()
    if args.rep < 1 or args.warmup < 0:
        parser.error("--rep must be >= 1 and --warmup must be >= 0")
    if any(not 0 <= p < 1 for p in args.dropout):
        parser.error("--dropout must be in [0, 1)")
    for name in ("backend", "dropout", "scenario"):
        values = getattr(args, name)
        if values is not None:
            setattr(args, name, list(dict.fromkeys(values)))
    return args


def configure_uniattention(root):
    if root is not None:
        root = root.expanduser().resolve()
        if not (root / "uniattention").is_dir():
            raise ValueError(f"{root} must contain the uniattention package")
        sys.path.insert(0, str(root))
    elif importlib.util.find_spec("uniattention") is None:
        sibling = ROOT.parent / "uni-attn/UniAttention"
        if not (sibling / "uniattention").is_dir():
            raise RuntimeError("UniAttention not found; provide --uniattention-root")
        sys.path.insert(0, str(sibling))


def sample_stats(samples):
    if not samples:
        return {key: None for key in ("min", "mean", "median", "max")}
    return {"min": min(samples), "mean": mean(samples), "median": median(samples), "max": max(samples)}


def run_measurement(torch, do_ops_bench, func, inputs, args):
    # Both wrappers return a tuple. Fusion also returns auxiliary tensors and
    # scalar metadata, which must not become backward targets in the utility.
    def attention_output(**kwargs):
        return func(**kwargs)[0]

    def clear_grads():
        for name in ("query_states", "key_states", "value_states"):
            inputs[name].grad = None

    try:
        with torch.set_grad_enabled(not args.forward_only):
            return do_ops_bench(
                attention_output, inputs, warmup=args.warmup, rep=args.rep,
                test_backward=not args.forward_only, pre_hook=clear_grads,
            )
    finally:
        clear_grads()


def add_stats(row, events):
    fwd = events["fwd_elapsed_time"]
    bwd = events["bwd_elapsed_time"]
    samples = {"fwd": fwd, "bwd": bwd, "total": [f + b for f, b in zip(fwd, bwd, strict=False)]}
    for phase, values in samples.items():
        for stat, value in sample_stats(values).items():
            row[f"{phase}_{stat}_ms"] = value


def add_speedups(rows):
    refs = {(r["scenario"], r["dropout"]): r for r in rows if r["backend"] == "npu_fa" and r["status"] == "ok"}
    for row in rows:
        ref = refs.get((row["scenario"], row["dropout"]))
        for phase in ("fwd", "bwd", "total"):
            value = row.get(f"{phase}_mean_ms")
            row[f"{phase}_speedup_vs_npu_fa"] = (
                ref[f"{phase}_mean_ms"] / value
                if ref and row["status"] == "ok" and value else None
            )


def render_report(metadata, rows):
    def fmt(value):
        return "—" if value is None else f"{value:.4f}"

    lines = [
        "# OneRec Attention 性能测试", "",
        f"设备：{metadata['device_name']}，逻辑设备 {metadata['device']}；"
        f"PyTorch {metadata['torch']}；torch_npu {metadata['torch_npu']}。",
        f"输入：复用 bench.py 的 CASES/build_inputs，BF16、TND、head_dim=128；"
        f"每组预热 {metadata['warmup']} 次，测量 {metadata['rep']} 次。",
        "计时：ops_benchmark.do_ops_bench，设备 Event，前向前冲刷 256 MiB 缓存；"
        "包含封装中的设备操作及同步间隙，不是 profiler 的纯 kernel Duration。",
        "反向仅对 Attention 主输出施加随机梯度；本脚本不校验数值精度。",
        "total 为配对前向与反向耗时之和，不包含两者之间生成梯度的时间。",
        "加速比 = torch FA 平均耗时 / 当前实现平均耗时；大于 1 表示当前实现更快。", "",
        "| 名称 | 脚本封装 | 实际调用接口 |",
        "|---|---|---|",
        "| torch FA | `npu_fa.flash_attn` | `torch_npu.npu_fusion_attention` |",
        "| FA2 | `npu_fa_v1.flash_attn` | `flash_attn_npu.flash_attn_npu_interface.flash_attn_varlen_func` |", "",
        "本次使用 eager 模式；FA2 对应 FlashAttention v2 的变长接口。", "",
        "| 场景 | 实现 | dropout | 状态 | 前向均值 ms | 反向均值 ms | 合计均值 ms | 合计加速比 |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {INTERFACE_LABELS[row['backend']]} | {row['dropout']:g} | {row['status']} | "
            f"{fmt(row.get('fwd_mean_ms'))} | {fmt(row.get('bwd_mean_ms'))} | "
            f"{fmt(row.get('total_mean_ms'))} | {fmt(row.get('total_speedup_vs_npu_fa'))} |"
        )
    for row in rows:
        if row["error"]:
            lines.extend(["", f"{row['scenario']} / {row['backend']} / dropout={row['dropout']}: {row['error']}"])
    return "\n".join(lines) + "\n"


def save_results(prefix, metadata, rows, raw):
    add_speedups(rows)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with Path(f"{prefix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    Path(f"{prefix}.json").write_text(
        json.dumps({"metadata": metadata, "summary": rows, "raw": raw}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = render_report(metadata, rows)
    Path(f"{prefix}.md").write_text(report, encoding="utf-8")
    return report


def main():
    args = parse_args()
    import torch
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("No accessible NPU; run in an environment with Ascend device access")
    torch.npu.set_device(args.device)
    import bench

    # bench.py changes sys.path at import time; add UniAttention afterwards.
    configure_uniattention(args.uniattention_root)
    from ops_benchmark import do_ops_bench

    valid = {case["name"] for case in bench.CASES}
    if args.scenario and set(args.scenario) - valid:
        raise ValueError(f"Unknown scenarios: {set(args.scenario) - valid}; choose from {sorted(valid)}")
    cases = [case for case in bench.CASES if not args.scenario or case["name"] in args.scenario]
    functions = {name: importlib.import_module(name).flash_attn for name in args.backend}
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device, "device_name": torch.npu.get_device_name(args.device),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "warmup": args.warmup, "rep": args.rep, "test_backward": not args.forward_only,
        "dtype": str(bench.DTYPE), "head_dim": bench.HEAD_DIM, "seed": bench.SEED,
        "backends": args.backend, "dropout": args.dropout,
        "cases": cases, "timing": "ops_benchmark.do_ops_bench; Event; 256 MiB flush before forward",
        "uniattention": importlib.util.find_spec("uniattention").origin,
    }
    rows, raw = [], []
    for case in cases:
        seed = bench.SEED + sum(map(ord, case["name"]))
        q, k, v, cuq, cuk, maxq, maxk = bench.build_inputs(case, seed)
        # Preserve the exact generated length distribution, not just totals.
        q_prefix, k_prefix = cuq.cpu().tolist(), cuk.cpu().tolist()
        for dropout in args.dropout:
            for name, func in functions.items():
                row = {
                    "scenario": case["name"], "backend": name, "dropout": dropout,
                    "batch": case["batch"], "total_q": case["total_q"], "total_kv": case["total_kv"],
                    "heads_q": case["heads_q"], "heads_kv": case["heads_kv"],
                    "head_dim": bench.HEAD_DIM, "dtype": "bfloat16", "max_q": maxq, "max_kv": maxk,
                    "causal": case["causal"], "scale": case["scale"],
                    "warmup": args.warmup, "rep": args.rep, "test_backward": not args.forward_only,
                    "status": "error", "error": "",
                }
                events = {"fwd_elapsed_time": [], "bwd_elapsed_time": []}
                add_stats(row, events)
                inputs = {
                    "query_states": q.detach(), "key_states": k.detach(), "value_states": v.detach(),
                    "cu_seqlens_q": cuq, "cu_seqlens_k": cuk,
                    "max_seqlen_q": maxq, "max_seqlen_k": maxk,
                    "softmax_scaling": case["scale"], "dropout_ratio": dropout,
                    "attention_mask": "causal" if case["causal"] else "full",
                }
                print(f"Running {case['name']} / {name} / dropout={dropout:g}", flush=True)
                try:
                    torch.manual_seed(seed)
                    torch.npu.manual_seed_all(seed)
                    events = run_measurement(torch, do_ops_bench, func, inputs, args)
                    add_stats(row, events)
                    row["status"] = "ok"
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                finally:
                    del inputs
                    gc.collect()
                    torch.npu.empty_cache()
                rows.append(row)
                raw.append({"scenario": case["name"], "backend": name, "dropout": dropout,
                            "cu_seqlens_q": q_prefix, "cu_seqlens_k": k_prefix, **events})
                # Save after each group, preserving completed measurements.
                save_results(args.output_prefix, metadata, rows, raw)
                print(f"  {row['status']}: fwd={row['fwd_mean_ms']} ms, bwd={row['bwd_mean_ms']} ms", flush=True)
        del q, k, v, cuq, cuk
        gc.collect()
        torch.npu.empty_cache()
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    print(save_results(args.output_prefix, metadata, rows, raw), flush=True)
    print(f"Results: {args.output_prefix}.{{csv,json,md}}", flush=True)
    return int(any(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
