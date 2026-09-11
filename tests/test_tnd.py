"""
am_flash_attn.txt 严格规格 · 50 组 TND varlen · 侵入式 profiler · 第 3 次迭代

严格参数：320 seqs | GQA 24/4/128 | bf16 | random seqlen [32,512] avg150
按 total_tokens 大中小均衡采样，分析 flash_attn_v2 相对 fusion 劣化。
不使用固定 seed / 确定性计算。
"""
import argparse
import csv
import glob
import os
import random
import re
import sys
from statistics import mean, median

import torch
import torch_npu
from torch_npu.profiler.profiler import analyse

_FA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flash-attention-npu")
if _FA_ROOT not in sys.path:
    sys.path.insert(0, _FA_ROOT)
from flash_attn_npu import flash_attn_varlen_func

# am_flash_attn.txt 严格规格
NUM_SEQS = 320
NUM_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 128
MIN_SEQ_LEN = 32
MAX_SEQ_LEN = 512
AVG_SEQ_LEN = 150
SEQLEN_MODE = "random"
DTYPE = torch.bfloat16
SCALE = HEAD_DIM ** -0.5

TARGET_ITER = 3
NUM_CASES = 50
PROFILE_ROOT = "/data00/zghe2e/batchtest/profile_am_varlen_50"
RESULT_CSV = "/data00/zghe2e/batchtest/compare_am_attn_step3_results.csv"

ATTN_MASK_SIZE = 2048
SPARSE_MODE = 3
AUX_PAT = re.compile(r"Fill|ReduceSum|Cast", re.I)


def gen_varlen_seqlens() -> list:
    seqlens = []
    for _ in range(NUM_SEQS):
        length = int(random.gauss(AVG_SEQ_LEN, AVG_SEQ_LEN * 0.25))
        seqlens.append(max(MIN_SEQ_LEN, min(MAX_SEQ_LEN, length)))
    return seqlens


def select_balanced_cases(n: int = NUM_CASES) -> list:
    """非确定性采样：生成候选 seqlens，按 total_tokens 均匀选取 n 组。"""
    oversample = max(n * 50, 2000)
    pool = []
    while len(pool) < oversample:
        seqlens = gen_varlen_seqlens()
        total = sum(seqlens)
        pool.append({
            "seqlens": seqlens,
            "total_tokens": total,
            "min_sl": min(seqlens),
            "max_sl": max(seqlens),
            "avg_sl": round(total / NUM_SEQS, 1),
        })
    pool.sort(key=lambda x: x["total_tokens"])
    if n >= len(pool):
        picked = pool
    else:
        indices = [int(round(i * (len(pool) - 1) / (n - 1))) for i in range(n)]
        picked = []
        seen = set()
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                picked.append(pool[idx])
        i = 0
        while len(picked) < n and i < len(pool):
            if i not in seen:
                picked.append(pool[i])
                seen.add(i)
            i += 1
        picked = picked[:n]
    for j, item in enumerate(picked, 1):
        item["case_id"] = j
        item["label"] = f"am_{j:02d}"
    return picked


def tercile_cutoffs(cases: list) -> tuple:
    tokens = sorted(c["total_tokens"] for c in cases)
    n = len(tokens)
    if n < 3:
        return tokens[0], tokens[-1]
    return tokens[n // 3 - 1], tokens[2 * n // 3 - 1]


def token_bucket(total_tokens: int, cutoffs: tuple) -> str:
    small_max, medium_max = cutoffs
    if total_tokens <= small_max:
        return "small"
    if total_tokens <= medium_max:
        return "medium"
    return "large"


def get_cu_seqlens(seqlens):
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    return cu


def make_atten_mask(device: str) -> torch.Tensor:
    return torch.triu(
        torch.ones(ATTN_MASK_SIZE, ATTN_MASK_SIZE, dtype=torch.bool, device=device),
        diagonal=1,
    )


def build_inputs(seqlens: list, device: str):
    total = sum(seqlens)
    q = torch.randn(total, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    v = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    cu_cpu = get_cu_seqlens(seqlens)
    cu = torch.tensor(cu_cpu, dtype=torch.int32, device=device)
    actual_seq = tuple(cu_cpu[1:])
    return q, k, v, cu, max(seqlens), actual_seq


def run_flash(q, k, v, cu, max_seqlen):
    flash_attn_varlen_func(
        q, k, v, cu, cu, max_seqlen, max_seqlen,
        dropout_p=0.0, softmax_scale=SCALE, causal=True, window_size=(-1, -1),
    )


def run_fusion(q, k, v, atten_mask, actual_seq):
    torch_npu.npu_fusion_attention(
        q, k, v, NUM_HEADS,
        input_layout="TND",
        pse=None, padding_mask=None, atten_mask=atten_mask,
        scale=SCALE, keep_prob=1.0,
        pre_tockens=65536, next_tockens=0,
        inner_precise=0, sparse_mode=SPARSE_MODE,
        actual_seq_qlen=actual_seq, actual_seq_kvlen=actual_seq,
    )


def parse_iter3_ms(ascend_pt):
    csv_path = sorted(glob.glob(os.path.join(ascend_pt, "**/kernel_details.csv"), recursive=True))[-1]
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    dur_col = "Duration(us)"
    flash_ms = fusion_ms = None
    for r in rows:
        name, typ = r["Name"], r["Type"]
        if AUX_PAT.search(name) or AUX_PAT.search(typ):
            continue
        dur = float(r[dur_col]) / 1000.0
        if "SplitFuse" in name or "FAInfer" in name:
            flash_ms = dur
        elif "FlashAttention" in name or "FlashAttention" in typ:
            fusion_ms = dur
    return flash_ms, fusion_ms


def run_case(case: dict, device: str, profile_root: str, atten_mask: torch.Tensor, cutoffs: tuple):
    seqlens = case["seqlens"]
    q, k, v, cu, max_seqlen, actual_seq = build_inputs(seqlens, device)

    prof_dir = os.path.join(profile_root, case["label"])
    os.makedirs(prof_dir, exist_ok=True)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
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
        record_shapes=False,
        profile_memory=False,
        with_modules=False,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir, analyse_flag=False),
    )

    for stepprof in range(1, TARGET_ITER + 2):
        if stepprof == TARGET_ITER:
            prof.start()
            run_flash(q, k, v, cu, max_seqlen)
            run_fusion(q, k, v, atten_mask, actual_seq)
            prof.step()
            prof.stop()
            break
        run_flash(q, k, v, cu, max_seqlen)
        run_fusion(q, k, v, atten_mask, actual_seq)

    ascend_pt = sorted(glob.glob(os.path.join(prof_dir, "*_ascend_pt")))[-1]
    analyse(profiler_path=ascend_pt, max_process_number=16, export_type="text")
    flash_ms, fusion_ms = parse_iter3_ms(ascend_pt)

    delta_ms = flash_ms - fusion_ms if flash_ms and fusion_ms else None
    ratio = flash_ms / fusion_ms if flash_ms and fusion_ms else None
    degrade_pct = (ratio - 1.0) * 100 if ratio else None

    return {
        **case,
        "bucket": token_bucket(case["total_tokens"], cutoffs),
        "flash_v2_ms": flash_ms,
        "fusion_ms": fusion_ms,
        "delta_ms": delta_ms,
        "ratio": ratio,
        "degrade_pct": degrade_pct,
    }


def analyze_results(results: list):
    ok = [r for r in results if r.get("flash_v2_ms") and r.get("fusion_ms")]
    if not ok:
        print("无有效结果")
        return

    def stat(vals):
        return mean(vals), median(vals), min(vals), max(vals)

    ratios = [r["ratio"] for r in ok]
    deltas = [r["delta_ms"] for r in ok]
    r_mean, r_med, r_min, r_max = stat(ratios)
    d_mean, d_med, d_min, d_max = stat(deltas)

    print("\n" + "=" * 90)
    print("flash_attn_v2 相对 npu_fusion_attention 劣化分析（第3次迭代 kernel Duration）")
    print("=" * 90)
    print(f"有效用例: {len(ok)}/{len(results)}")
    print(f"ratio  (v2/fusion): mean={r_mean:.3f}x  median={r_med:.3f}x  min={r_min:.3f}x  max={r_max:.3f}x")
    print(f"delta  (v2-fusion): mean={d_mean:.3f}ms median={d_med:.3f}ms min={d_min:.3f}ms max={d_max:.3f}ms")
    print(f"劣化百分比 (ratio-1): mean={(r_mean-1)*100:.1f}%  median={(r_med-1)*100:.1f}%")

    for bucket in ("small", "medium", "large"):
        sub = [r for r in ok if r["bucket"] == bucket]
        if not sub:
            continue
        br = [r["ratio"] for r in sub]
        bd = [r["delta_ms"] for r in sub]
        tok = [r["total_tokens"] for r in sub]
        print(
            f"\n[{bucket}] n={len(sub)} tokens=[{min(tok)},{max(tok)}] "
            f"ratio mean={mean(br):.3f}x median={median(br):.3f}x | "
            f"delta mean={mean(bd):.3f}ms median={median(bd):.3f}ms"
        )

    worst = sorted(ok, key=lambda r: r["ratio"], reverse=True)[:5]
    best = sorted(ok, key=lambda r: r["ratio"])[:5]
    print("\n劣化最严重 TOP5 (ratio):")
    for r in worst:
        print(f"  {r['label']} tokens={r['total_tokens']} "
              f"v2={r['flash_v2_ms']:.3f} fusion={r['fusion_ms']:.3f} ratio={r['ratio']:.3f}x")
    print("\n劣化最小 TOP5 (ratio):")
    for r in best:
        print(f"  {r['label']} tokens={r['total_tokens']} "
              f"v2={r['flash_v2_ms']:.3f} fusion={r['fusion_ms']:.3f} ratio={r['ratio']:.3f}x")
    print("=" * 90)


def save_csv(results: list, path: str):
    fields = [
        "case_id", "label", "num_seqs", "total_tokens", "min_sl", "max_sl", "avg_sl",
        "bucket", "flash_v2_ms", "fusion_ms", "delta_ms", "ratio", "degrade_pct",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fields}
            row["num_seqs"] = NUM_SEQS
            w.writerow(row)
    print(f"\n结果 CSV: {path}")


def main():
    parser = argparse.ArgumentParser(description="am_flash_attn 50-case v2 vs fusion profile compare")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--num-cases", type=int, default=NUM_CASES)
    parser.add_argument("--profile-root", default=PROFILE_ROOT)
    parser.add_argument("--result-csv", default=RESULT_CSV)
    args = parser.parse_args()

    if not torch.npu.is_available():
        raise RuntimeError("未检测到可用 NPU 设备")

    torch.npu.set_device(args.device)
    atten_mask = make_atten_mask(args.device)

    cases = select_balanced_cases(args.num_cases)
    cutoffs = tercile_cutoffs(cases)
    print(f"am_flash_attn 严格规格 | {len(cases)} 用例 | TND 320 seqs random [32,512] avg150")
    print(f"token 范围: {cases[0]['total_tokens']} ~ {cases[-1]['total_tokens']}")
    print(f"分桶(三分位): small<={cutoffs[0]} medium<={cutoffs[1]} large>{cutoffs[1]}")

    results = []
    for case in cases:
        print(
            f"\n>>> [{case['case_id']:02d}/{len(cases)}] {case['label']} "
            f"tokens={case['total_tokens']} "
            f"seqlen=[{case['min_sl']},{case['max_sl']}] bucket={token_bucket(case['total_tokens'], cutoffs)}"
        )
        try:
            r = run_case(case, args.device, args.profile_root, atten_mask, cutoffs)
            results.append(r)
            print(
                f"    v2={r['flash_v2_ms']:.3f}ms fusion={r['fusion_ms']:.3f}ms "
                f"delta={r['delta_ms']:+.3f}ms ratio={r['ratio']:.3f}x degrade={r['degrade_pct']:.1f}%"
            )
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({**case, "flash_v2_ms": None, "fusion_ms": None})

    save_csv(results, args.result_csv)

    print("\n" + "-" * 86)
    print(f"{'id':>3} {'tokens':>7} {'bucket':>6} {'v2(ms)':>8} {'fusion':>8} {'delta':>8} {'ratio':>7}")
    print("-" * 86)
    for r in results:
        if not r.get("flash_v2_ms"):
            continue
        print(
            f"{r['case_id']:>3} {r['total_tokens']:>7} {r['bucket']:>6} "
            f"{r['flash_v2_ms']:>8.3f} {r['fusion_ms']:>8.3f} {r['delta_ms']:>+8.3f} {r['ratio']:>7.3f}x"
        )

    analyze_results(results)


if __name__ == "__main__":
    main()
