"""OneRec TND shapes: FA2 / FA3 / FA4 forward performance.

Shapes follow the OneRec table (bf16, packed TND, PE empty):

    Qwen3 3B    B=32  Q=K=V=O [8928,8,128]               max 279/279  causal  scale=1/sqrt(128)
    Qwen3 4B    B=16  Q=O [4464,32,128] K=V [4464,8,128] max 279/279  causal  scale=1/sqrt(128)
    T5 Encoder  B=512 Q=K=V=O [131475,8,128]             max 276/276  full    scale=1
    T5 Decoder  B=512 Q=K=V=O [1536,16,128]              max 3/3      causal  scale=1
    T5 Cross    B=512 Q=O [1536,16,128]
                      K=V [131475,16,128]                max 3/276    full    scale=1

Each row reports the median end-to-end launch time and the summed attention
kernel Duration from the 3rd iteration. FA2 is measured twice, with
dropout_p=0 and dropout_p=0.1. FA3 and FA4 have no dropout argument.

Run from a version tree (v0.4.0, v0.3.0, or pre-zj):

    python bench_onerec_fa.py
    python bench_onerec_fa.py --fa fa2 --scenario qwen3_3b
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib
import inspect
import os
import random
import re
import shutil
import sys
import types
from statistics import mean, median

script_dir = sys.path.pop(0)
import torch
import torch_npu
from torch_npu.profiler.profiler import analyse

_FA_ROOT = os.path.dirname(os.path.abspath(__file__))
if _FA_ROOT not in sys.path:
    sys.path.insert(0, _FA_ROOT)

DTYPE = torch.bfloat16
HEAD_DIM = 128
SEED = 20261024
WARMUP = 2
E2E_ITERS = 10
PROFILE_WARMUP = 2
RESULT_CSV = os.path.join(_FA_ROOT, "batchtest", "onerec_fa_perf.csv")
PROFILE_ROOT = os.path.join(_FA_ROOT, "batchtest", "profile_onerec_fa")

# Attention device kernels. Metadata / fill / cast tasks are not the FA kernel.
FA_KERNEL = re.compile(
    r"SplitFuse|FAInfer|FlashAttention|FlashAttn|FlashDecode|FACombine|fa_infer|ascendc_fa|"
    r"FusedInfer|PromptFlash|IncreFlash",
    re.I,
)
AUX_KERNEL = re.compile(r"Fill|ReduceSum|Cast|metadata", re.I)

# OneRec table. Qwen3 4B is GQA (Q heads 32, K/V heads 8). Other rows are MHA.
CASES = [
    {
        "name": "qwen3_3b",
        "batch": 32,
        "total_q": 8928,
        "total_kv": 8928,
        "heads_q": 8,
        "heads_kv": 8,
        "max_q": 279,
        "max_kv": 279,
        "causal": True,
        "scale": HEAD_DIM ** -0.5,
    },
    {
        "name": "qwen3_4b",
        "batch": 16,
        "total_q": 4464,
        "total_kv": 4464,
        "heads_q": 32,
        "heads_kv": 8,
        "max_q": 279,
        "max_kv": 279,
        "causal": True,
        "scale": HEAD_DIM ** -0.5,
    },
    {
        "name": "t5_encoder",
        "batch": 512,
        "total_q": 131475,
        "total_kv": 131475,
        "heads_q": 8,
        "heads_kv": 8,
        "max_q": 276,
        "max_kv": 276,
        "causal": False,
        "scale": 1.0,
    },
    {
        "name": "t5_decoder",
        "batch": 512,
        "total_q": 1536,
        "total_kv": 1536,
        "heads_q": 16,
        "heads_kv": 16,
        "max_q": 3,
        "max_kv": 3,
        "causal": True,
        "scale": 1.0,
    },
    {
        "name": "t5_cross",
        "batch": 512,
        "total_q": 1536,
        "total_kv": 131475,
        "heads_q": 16,
        "heads_kv": 16,
        "max_q": 3,
        "max_kv": 276,
        "causal": False,
        "scale": 1.0,
    },
]

FA_SPECS = [
    ("fa2", "flash_attn_npu", 0.0),
    ("fa2", "flash_attn_npu", 0.1),
    ("fa3", "flash_attn_npu_3", None),
    ("fa4", "flash_attn_npu_4", None),
]


def gen_seq_lens(total, batch, max_len, seed):
    """Integer lengths in [1, max_len] that sum to ``total``."""
    if total == batch * max_len:
        return [max_len] * batch
    rng = random.Random(seed)
    if not batch <= total <= batch * max_len:
        raise ValueError(f"infeasible total={total} batch={batch} max_len={max_len}")
    lens = []
    remain = total
    for i in range(batch):
        slots_left = batch - i - 1
        lo = max(1, remain - max_len * slots_left)
        hi = min(max_len, remain - slots_left)
        val = rng.randint(lo, hi)
        lens.append(val)
        remain -= val
    rng.shuffle(lens)
    return lens


def cu_seqlens(lengths):
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    for i, length in enumerate(lengths, start=1):
        cu[i] = cu[i - 1] + int(length)
    return cu.npu()


def load_varlen(package):
    """Load ``flash_attn_varlen_func`` from this version tree.

    v0.3.0's FA2/FA4 ``__init__`` only accepts a device name containing
    ``Ascend910B`` or ``Ascend910C``. This host reports ``Ascend910_9382``,
    so that gate is skipped and the interface module is loaded directly.
    """
    interface = f"{package}.flash_attn_npu_interface"
    try:
        return importlib.import_module(interface).flash_attn_varlen_func
    except RuntimeError as exc:
        if "Unsupported Ascend device" not in str(exc):
            raise
        sys.modules.pop(package, None)
        stub = types.ModuleType(package)
        stub.__path__ = [os.path.join(_FA_ROOT, package)]
        stub.__package__ = package
        sys.modules[package] = stub
        return importlib.import_module(interface).flash_attn_varlen_func


def bind_call(fn, dropout):
    params = inspect.signature(fn).parameters

    def call(q, k, v, cu_q, cu_k, max_q, max_k, scale, causal):
        kwargs = {
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": max_q,
            "max_seqlen_k": max_k,
            "softmax_scale": scale,
            "causal": causal,
        }
        if dropout is not None:
            if "dropout_p" not in params:
                raise RuntimeError(f"{fn.__module__} does not accept dropout_p")
            kwargs["dropout_p"] = dropout
        kwargs = {key: value for key, value in kwargs.items() if key in params}
        return fn(q, k, v, **kwargs)

    return call


def build_inputs(case, seed):
    q_lens = gen_seq_lens(case["total_q"], case["batch"], case["max_q"], seed)
    k_lens = (
        q_lens
        if case["total_q"] == case["total_kv"] and case["max_q"] == case["max_kv"]
        else gen_seq_lens(case["total_kv"], case["batch"], case["max_kv"], seed + 1)
    )
    if max(q_lens) != case["max_q"] or max(k_lens) != case["max_kv"]:
        # The generator only reaches max_len when the total forces it. Pin one
        # sequence to the table's max and rebuild the remainder.
        q_lens = _pin_max(case["total_q"], case["batch"], case["max_q"], seed)
        k_lens = (
            q_lens
            if case["total_q"] == case["total_kv"] and case["max_q"] == case["max_kv"]
            else _pin_max(case["total_kv"], case["batch"], case["max_kv"], seed + 1)
        )
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(
        sum(q_lens), case["heads_q"], HEAD_DIM, generator=generator, dtype=torch.float32
    ).to(DTYPE).npu()
    generator.manual_seed(seed + 2)
    k = torch.randn(
        sum(k_lens), case["heads_kv"], HEAD_DIM, generator=generator, dtype=torch.float32
    ).to(DTYPE).npu()
    generator.manual_seed(seed + 3)
    v = torch.randn(
        sum(k_lens), case["heads_kv"], HEAD_DIM, generator=generator, dtype=torch.float32
    ).to(DTYPE).npu()
    return q, k, v, cu_seqlens(q_lens), cu_seqlens(k_lens), max(q_lens), max(k_lens)


def _pin_max(total, batch, max_len, seed):
    if batch == 1:
        return [total]
    rest = gen_seq_lens(total - max_len, batch - 1, max_len, seed)
    lens = [max_len] + rest
    random.Random(seed).shuffle(lens)
    return lens


def e2e_median_ms(fn):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    samples = []
    for _ in range(E2E_ITERS):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return median(samples)


def parse_kernel_ms(ascend_pt):
    csv_paths = sorted(glob.glob(os.path.join(ascend_pt, "**/kernel_details.csv"), recursive=True))
    if not csv_paths:
        raise RuntimeError(f"kernel_details.csv missing under {ascend_pt}")
    with open(csv_paths[-1], newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    matched = []
    others = []
    for row in rows:
        name = row.get("Name", "")
        kind = row.get("Type", "")
        raw = row.get("Duration(us)", "")
        if not raw:
            continue
        duration_ms = float(raw) / 1000.0
        label = name or kind
        if AUX_KERNEL.search(name) or AUX_KERNEL.search(kind):
            continue
        if FA_KERNEL.search(name) or FA_KERNEL.search(kind):
            matched.append((label, duration_ms))
        else:
            others.append((label, duration_ms))
    chosen = matched or others
    total = sum(duration for _, duration in chosen)
    names = ";".join(f"{label}:{duration:.4f}" for label, duration in chosen[:8])
    if not matched and others:
        names = "unfiltered " + names
    return total, names


def profile_kernel_ms(fn, profile_dir):
    if os.path.isdir(profile_dir):
        shutil.rmtree(profile_dir)
    os.makedirs(profile_dir, exist_ok=True)
    for _ in range(PROFILE_WARMUP):
        fn()
    torch.npu.synchronize()
    experimental = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        data_simplification=False,
        l2_cache=False,
    )
    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=False,
        profile_memory=False,
        with_modules=False,
        experimental_config=experimental,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_dir, analyse_flag=False),
    )
    profiler.start()
    fn()
    profiler.step()
    profiler.step()
    profiler.stop()
    torch.npu.synchronize()
    traces = glob.glob(os.path.join(profile_dir, "*_ascend_pt"))
    if len(traces) != 1:
        raise RuntimeError(f"expected one profiler trace in {profile_dir}, found {len(traces)}")
    analyse(profiler_path=traces[0], max_process_number=8, export_type="text")
    kernel_ms, names = parse_kernel_ms(traces[0])
    shutil.rmtree(profile_dir, ignore_errors=True)
    return kernel_ms, names


def make_fusion_launch(case, q, k, v, cu_q, cu_k):
    """TND npu_fusion_attention. Causal uses sparse_mode=3 and a fixed triu mask."""
    actual_q = tuple(int(x) for x in cu_q.detach().cpu().tolist()[1:])
    actual_k = tuple(int(x) for x in cu_k.detach().cpu().tolist()[1:])
    if case["causal"]:
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device="npu"), diagonal=1)
        sparse_mode = 3
    else:
        mask = None
        sparse_mode = 0

    def launch():
        torch_npu.npu_fusion_attention(
            q, k, v, case["heads_q"],
            "TND",
            pse=None,
            padding_mask=None,
            atten_mask=mask,
            scale=case["scale"],
            keep_prob=1.0,
            pre_tockens=2147483647,
            next_tockens=2147483647,
            inner_precise=0,
            sparse_mode=sparse_mode,
            actual_seq_qlen=actual_q,
            actual_seq_kvlen=actual_k,
        )

    return launch


def _fmt_stats(values):
    return min(values), mean(values), median(values), max(values)


def _blank_stats():
    return {
        "e2e_min": "",
        "e2e_mean": "",
        "e2e_median": "",
        "e2e_max": "",
        "kernel_min": "",
        "kernel_mean": "",
        "kernel_median": "",
        "kernel_max": "",
    }


def _quad(min_v, mean_v, median_v, max_v):
    return f"{min_v:.4f}/{mean_v:.4f}/{median_v:.4f}/{max_v:.4f}"


def _format_table(rows, columns):
    widths = []
    for key, header in columns:
        width = len(header)
        for row in rows:
            width = max(width, len(str(row.get(key, ""))))
        widths.append(width)

    def line(values):
        return "  ".join(str(value).ljust(width) for value, width in zip(values, widths))

    header = line([title for _, title in columns])
    rule = "  ".join("-" * width for width in widths)
    body = [line([row.get(key, "") for key, _ in columns]) for row in rows]
    return "\n".join([header, rule, *body])


def _print_summary_table(summaries):
    display = []
    for row in summaries:
        if row["n_pass"]:
            e2e = _quad(
                float(row["e2e_min"]), float(row["e2e_mean"]),
                float(row["e2e_median"]), float(row["e2e_max"]),
            )
            kernel = _quad(
                float(row["kernel_min"]), float(row["kernel_mean"]),
                float(row["kernel_median"]), float(row["kernel_max"]),
            )
        else:
            e2e = "-"
            kernel = "-"
        display.append({
            "scenario": row["scenario"],
            "fa": row["fa"],
            "drop": row["dropout"] if row["dropout"] != "" else "0",
            "n": f"{row['n_pass']}/{row['n_run']}",
            "e2e": e2e,
            "kernel": kernel,
        })
    print("\nOneRec FA summary (ms). e2e / kernel = min/mean/median/max over passing repeats", flush=True)
    print(
        _format_table(
            display,
            [
                ("scenario", "scenario"),
                ("fa", "fa"),
                ("drop", "dropout"),
                ("n", "pass/run"),
                ("e2e", "e2e ms"),
                ("kernel", "kernel ms"),
            ],
        ),
        flush=True,
    )
    failed = [
        {
            **row,
            "dropout": row["dropout"] if row["dropout"] != "" else "0",
            "error": row["error"][:160],
        }
        for row in summaries
        if row.get("error")
    ]
    if failed:
        print("\nFailed runs", flush=True)
        print(
            _format_table(
                failed,
                [("scenario", "scenario"), ("fa", "fa"), ("dropout", "dropout"), ("error", "error")],
            ),
            flush=True,
        )


def run_one(case, fa_name, package, dropout, version, repeat_idx=0):
    row = {
        "version": version,
        "scenario": case["name"],
        "fa": fa_name,
        "dropout": "" if dropout is None else dropout,
        "repeat": repeat_idx,
        "batch": case["batch"],
        "heads_q": case["heads_q"],
        "heads_kv": case["heads_kv"],
        "total_q": case["total_q"],
        "total_kv": case["total_kv"],
        "max_q": case["max_q"],
        "max_kv": case["max_kv"],
        "causal": int(case["causal"]),
        "scale": case["scale"],
        "e2e_ms": "",
        "kernel_ms": "",
        "kernel_names": "",
        "status": "fail",
        "error": "",
    }
    try:
        q, k, v, cu_q, cu_k, max_q, max_k = build_inputs(
            case, SEED + sum(ord(ch) for ch in case["name"])
        )
        if fa_name == "fusion":
            launch = make_fusion_launch(case, q, k, v, cu_q, cu_k)
        else:
            fn = bind_call(load_varlen(package), dropout)

            def launch():
                fn(q, k, v, cu_q, cu_k, max_q, max_k, case["scale"], case["causal"])

        row["e2e_ms"] = f"{e2e_median_ms(launch):.4f}"
        label = f"{fa_name}_{case['name']}"
        if dropout:
            label += f"_drop{dropout}"
        if repeat_idx:
            label += f"_r{repeat_idx}"
        kernel_ms, names = profile_kernel_ms(launch, os.path.join(PROFILE_ROOT, label))
        row["kernel_ms"] = f"{kernel_ms:.4f}"
        row["kernel_names"] = names
        row["status"] = "pass"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        torch.npu.empty_cache()
    return row


def _print_group_stats(scenario, fa_name, dropout, rows):
    passed = [row for row in rows if row["status"] == "pass"]
    failed = [row["error"] for row in rows if row["status"] != "pass" and row["error"]]
    summary = {
        "scenario": scenario,
        "fa": fa_name,
        "dropout": "" if dropout is None else dropout,
        "n_pass": len(passed),
        "n_run": len(rows),
        "error": failed[0] if failed else "",
        **_blank_stats(),
    }
    if not passed:
        return summary
    e2e = [float(row["e2e_ms"]) for row in passed]
    kernel = [float(row["kernel_ms"]) for row in passed]
    e2e_min, e2e_mean, e2e_median, e2e_max = _fmt_stats(e2e)
    k_min, k_mean, k_median, k_max = _fmt_stats(kernel)
    summary.update({
        "e2e_min": f"{e2e_min:.4f}",
        "e2e_mean": f"{e2e_mean:.4f}",
        "e2e_median": f"{e2e_median:.4f}",
        "e2e_max": f"{e2e_max:.4f}",
        "kernel_min": f"{k_min:.4f}",
        "kernel_mean": f"{k_mean:.4f}",
        "kernel_median": f"{k_median:.4f}",
        "kernel_max": f"{k_max:.4f}",
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description="OneRec FA2/FA3/FA4 e2e and kernel benchmark")
    parser.add_argument("--fa", nargs="*", choices=["fa2", "fa3", "fa4", "fusion"], default=["fa2", "fa3", "fa4"])
    parser.add_argument("--scenario", nargs="*", default=[case["name"] for case in CASES])
    parser.add_argument("--csv", default=RESULT_CSV)
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="repeat each case this many times and report min/mean/median/max",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    version = os.path.basename(_FA_ROOT)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    fields = [
        "version", "scenario", "fa", "dropout", "repeat", "batch", "heads_q", "heads_kv",
        "total_q", "total_kv", "max_q", "max_kv", "causal", "scale",
        "e2e_ms", "kernel_ms", "kernel_names", "status", "error",
    ]
    selected = [case for case in CASES if case["name"] in set(args.scenario)]
    specs = [spec for spec in FA_SPECS if spec[0] in set(args.fa)]
    if "fusion" in args.fa:
        specs.append(("fusion", "", None))
    summaries = []
    with open(args.csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in selected:
            for fa_name, package, dropout in specs:
                group_rows = []
                for repeat_idx in range(args.repeat):
                    row = run_one(case, fa_name, package, dropout, version, repeat_idx)
                    writer.writerow(row)
                    handle.flush()
                    print(
                        f"[{version}] {row['scenario']} {row['fa']} dropout={row['dropout'] or 0} "
                        f"r={repeat_idx + 1}/{args.repeat} {row['status']} "
                        f"e2e={row['e2e_ms']} kernel={row['kernel_ms']} {row['error'][:180]}",
                        flush=True,
                    )
                    group_rows.append(row)
                summaries.append(_print_group_stats(case["name"], fa_name, dropout, group_rows))
    if summaries:
        stats_csv = os.path.splitext(args.csv)[0] + "_stats.csv"
        stats_fields = [
            "scenario", "fa", "dropout", "n_pass", "n_run",
            "e2e_min", "e2e_mean", "e2e_median", "e2e_max",
            "kernel_min", "kernel_mean", "kernel_median", "kernel_max",
            "error",
        ]
        with open(stats_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=stats_fields)
            writer.writeheader()
            writer.writerows(summaries)
        _print_summary_table(summaries)
        print(f"\n[stats] wrote {stats_csv}", flush=True)


if __name__ == "__main__":
    main()
