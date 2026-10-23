#!/usr/bin/env python3
"""Compute the true L2 double-benchmark ratios from two ATK runs.

ATK's `cv_fused_double_benchmark` compares:
    Actual_*    : candidate (fused kernel) vs CPU golden
    Benchmark_* : second implementation vs CPU golden
    ratio       : Actual / max(Benchmark, error_thd)      error_thd = 2^-14

`atk pytorch` only wires one benchmark node (cpu = golden), so Benchmark_*
degenerates to 0. This script reads the candidate run and the small-op
reference run produced by `execute_flash_attn_combine.py` and applies the
L2 formula (max 2 / avg 1.2 / rmse 1.2).
"""

import argparse
import ast
import glob
import sys

import pandas as pd

ERROR_THD = 6.103515625e-05  # ATK DoubleBenchmarkCompareStandard.error_thd = 2^-14
L2_LIMITS = {"max": 2.0, "avg": 1.2, "rmse": 1.2}


def _load_report(pattern: str) -> dict:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"report not found: {pattern}")
    df = pd.read_excel(files[-1], sheet_name="statistic")
    metrics = {}
    for _, row in df.iterrows():
        detail = row.get("精度详情")
        if not isinstance(detail, str):
            continue
        for outputs in ast.literal_eval(detail).values():
            for item in outputs:
                ind = item.get("new_benchmark_indicate") or {}
                key = (row["编号"], item.get("filename"))
                metrics[key] = {
                    "max": float(ind.get("Actual_最大相对误差", 0.0)),
                    "avg": float(ind.get("Actual_平均相对误差", 0.0)),
                    "rmse": float(ind.get("Actual_均方根误差", 0.0)),
                    "result": bool(item.get("result", True)),
                }
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="glob of the fused run report")
    ap.add_argument("--benchmark", required=True, help="glob of the small-op run report")
    args = ap.parse_args()

    cand = _load_report(args.candidate)
    bench = _load_report(args.benchmark)
    common = sorted(set(cand) & set(bench))
    if not common:
        raise SystemExit("no overlapping (case id, output) between the two runs")

    worst = {"max": (0.0, None), "avg": (0.0, None), "rmse": (0.0, None)}
    fail = []
    for key in common:
        ratios = {}
        for metric in ("max", "avg", "rmse"):
            denom = max(bench[key][metric], ERROR_THD)
            ratios[metric] = cand[key][metric] / denom
            if ratios[metric] > worst[metric][0]:
                worst[metric] = (ratios[metric], key)
        if any(ratios[m] > L2_LIMITS[m] for m in L2_LIMITS):
            fail.append((key, ratios))

    print(f"matched outputs: {len(common)}")
    for metric in ("max", "avg", "rmse"):
        print(
            f"{metric:>4}: worst={worst[metric][0]:.4f} (case {worst[metric][1]}) "
            f"limit={L2_LIMITS[metric]}"
        )
    print(f"L2 verdict: {'PASS' if not fail else 'FAIL'} ({len(fail)} failing outputs)")
    for key, ratios in fail[:10]:
        print("  FAIL", key, {k: round(v, 4) for k, v in ratios.items()})
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
