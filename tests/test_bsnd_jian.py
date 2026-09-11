
"""
am_flash_attn.txt 规格（BSND 放宽）· flash_attn_func vs npu_fusion_attention · 50 组

am 严格参数：320 seqs | GQA 24/4/128 | bf16 | random seqlen [32,512] avg150
BSND 放宽：layout=BSND | causal=False | fusion sparse_mode=0 atten_mask=None
          batch 固定 320，seq_len 按 am 高斯分布采样（矩形输入代替 TND varlen）
按 total_tokens 大中小均衡采样，第 3 次迭代 kernel Duration。

The --fixed-seq-len mode is intentionally provided for profiler comparisons:
it removes the default random-S sampling and prints the logical GM->L1 data
copy breakdown at the QK-Q/QK-K/PV-P/PV-V boundaries. The breakdown is a
source-level byte model (not a replacement for the hardware counters): layout
padding and Catlass copy lowering can make profiler values larger or smaller.
"""
import argparse
import csv
import glob
import inspect
import json
import os
import random
import re
import shutil
import sys
from statistics import mean, median

import torch
import torch_npu
from torch_npu.profiler.profiler import analyse

# This script lives in <repo>/tests. Import the package from the repository,
# rather than relying on the process working directory.
_FA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FA_ROOT not in sys.path:
    sys.path.insert(0, _FA_ROOT)
from flash_attn_npu import flash_attn_func, get_scheduler_metadata

# scheduler_metadata is available only in some flash_attn_npu interface
# revisions. Keep this benchmark usable with both interface variants.
FLASH_ATTN_SUPPORTS_SCHEDULER_METADATA = (
    "scheduler_metadata" in inspect.signature(flash_attn_func).parameters
)

# am_flash_attn.txt 规格（BSND 仅放宽 layout / causal / fusion mask）
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
INPUT_LAYOUT = "BSND"
SPARSE_MODE = 0
Q_TILE_CEIL = 128
N_SPLIT_HELPER = 2
QK_K_L1_TILE = 128
PV_P_L1_TILE = 256

TARGET_ITER = 3
NUM_CASES = 50
PROFILE_ROOT = "./batchtest/profile_am_bsnd_50"
RESULT_CSV = "./batchtest/compare_bsnd_attn_step3_results.csv"
MIN_PROFILE_FREE_BYTES = 1 << 30

AUX_PAT = re.compile(r"Fill|ReduceSum|Cast", re.I)


def free_space(path: str) -> int:
    """Return free bytes on the filesystem containing *path*."""
    probe = os.path.abspath(path)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe).free


def resolve_output_paths(profile_root: str | None, result_csv: str | None):
    """Use a writable local filesystem when the default data disk is full."""
    resolved_profile_root = profile_root or PROFILE_ROOT
    resolved_result_csv = result_csv or RESULT_CSV
    if profile_root is not None or free_space(resolved_profile_root) >= MIN_PROFILE_FREE_BYTES:
        return resolved_profile_root, resolved_result_csv

    for fallback in ("/dev/shm/profile_am_bsnd_50", "/tmp/profile_am_bsnd_50"):
        if free_space(fallback) >= MIN_PROFILE_FREE_BYTES:
            print(
                f"警告: profiler 默认目录所在文件系统仅剩 "
                f"{free_space(resolved_profile_root) / (1 << 30):.2f} GiB，"
                f"自动切换到 {fallback}"
            )
            if result_csv is None:
                resolved_result_csv = os.path.join(
                    os.path.dirname(fallback), os.path.basename(RESULT_CSV)
                )
            return fallback, resolved_result_csv

    raise RuntimeError(
        f"profiler 输出目录 {resolved_profile_root} 所在文件系统空间不足，"
        "且 /dev/shm、/tmp 均没有足够可用空间；请清理磁盘或显式指定 --profile-root"
    )


def gen_seq_len() -> int:
    """am random 模式：高斯分布 avg=150，截断到 [32,512]。"""
    length = int(random.gauss(AVG_SEQ_LEN, AVG_SEQ_LEN * 0.25))
    return max(MIN_SEQ_LEN, min(MAX_SEQ_LEN, length))


def select_balanced_cases(n: int = NUM_CASES) -> list:
    """非确定性采样：固定 batch=320，候选 seq_len 按 total_tokens 均匀选取 n 组。"""
    pool_map = {}
    attempts = 0
    max_unique = MAX_SEQ_LEN - MIN_SEQ_LEN + 1
    while len(pool_map) < max_unique and attempts < max_unique * 200:
        attempts += 1
        seq_len = gen_seq_len()
        if seq_len in pool_map:
            continue
        pool_map[seq_len] = {
            "seq_len": seq_len,
            "total_tokens": NUM_SEQS * seq_len,
        }
    for edge in (MIN_SEQ_LEN, MAX_SEQ_LEN):
        if edge not in pool_map:
            pool_map[edge] = {
                "seq_len": edge,
                "total_tokens": NUM_SEQS * edge,
            }
    pool = sorted(pool_map.values(), key=lambda x: x["total_tokens"])
    if n <= 1:
        picked = pool[-n:] if n == 1 else []
    elif n >= len(pool):
        picked = pool
    else:
        indices = [int(round(i * (len(pool) - 1) / (n - 1))) for i in range(n)]
        picked = []
        picked_seen = set()
        for idx in indices:
            if idx not in picked_seen:
                picked_seen.add(idx)
                picked.append(pool[idx])
        i = 0
        while len(picked) < n and i < len(pool):
            if i not in picked_seen:
                picked.append(pool[i])
                picked_seen.add(i)
            i += 1
        picked = picked[:n]
    for j, item in enumerate(picked, 1):
        item["case_id"] = j
        item["label"] = f"am_bsnd_{j:02d}"
    return picked


def fixed_case(seq_len: int) -> list:
    """Build one deterministic full-shape case for a profiler A/B run."""
    if not MIN_SEQ_LEN <= seq_len <= MAX_SEQ_LEN:
        raise ValueError(f"--fixed-seq-len must be in [{MIN_SEQ_LEN}, {MAX_SEQ_LEN}], got {seq_len}")
    return [{
        "case_id": 1,
        "label": f"fixed_b{NUM_SEQS}_s{seq_len}_h{NUM_HEADS}_kv{NUM_KV_HEADS}_d{HEAD_DIM}",
        "seq_len": seq_len,
        "total_tokens": NUM_SEQS * seq_len,
    }]


def current_tiling(seq_len: int) -> dict:
    """Mirror GetQSBlockTile/GetQNBlockTile for the BSND forward kernel."""
    group_size = NUM_HEADS // NUM_KV_HEADS
    q_s_block = min(Q_TILE_CEIL, seq_len)
    q_n_block = ((Q_TILE_CEIL // seq_len) // N_SPLIT_HELPER) * N_SPLIT_HELPER
    q_n_block = min(q_n_block, group_size)
    q_n_block = max(q_n_block, 1)
    return {
        "group_size": group_size,
        "q_s_block": q_s_block,
        "q_n_block": q_n_block,
        "q_s_blocks": (seq_len + q_s_block - 1) // q_s_block,
        "q_n_blocks_per_group": (group_size + q_n_block - 1) // q_n_block,
    }


def transfer_model(seq_len: int) -> dict:
    """Logical per-task GM->L1 bytes at the four source DataCopy boundaries.

    This mirrors qk_matmul.hpp::loadQGM/operator() and pv_matmul.hpp::operator()
    for the fixed, non-paged BSND path. It deliberately counts payload bytes;
    the profiler's GM_to_L1_datas is an implementation-level metric and may
    include alignment/lowering effects not visible at this boundary.
    """
    tiling = current_tiling(seq_len)
    # if seq_len % tiling["q_s_block"] or NUM_HEADS % NUM_KV_HEADS:
    #     raise ValueError("transfer model currently requires full Q-S tiles and integral GQA groups")
    row_num = tiling["q_s_block"] * tiling["q_n_block"]
    element_bytes = torch.tensor([], dtype=DTYPE).element_size()
    stage_bytes = {
        "QK-Q": row_num * HEAD_DIM * element_bytes,
        "QK-K": seq_len * HEAD_DIM * element_bytes,
        "PV-P": row_num * seq_len * element_bytes,
        "PV-V": seq_len * HEAD_DIM * element_bytes,
    }
    tasks_per_batch = tiling["q_s_blocks"] * tiling["q_n_blocks_per_group"] * NUM_KV_HEADS
    return {
        **tiling,
        "row_num": row_num,
        "qk_k_l1_copies": (seq_len + QK_K_L1_TILE - 1) // QK_K_L1_TILE,
        "pv_p_l1_copies": (seq_len + PV_P_L1_TILE - 1) // PV_P_L1_TILE,
        "tasks_per_batch": tasks_per_batch,
        "total_tasks": NUM_SEQS * tasks_per_batch,
        "stage_bytes": stage_bytes,
        "logical_bytes_per_task": sum(stage_bytes.values()),
    }


def print_transfer_model(seq_len: int):
    model = transfer_model(seq_len)
    print("\nGM->L1 source-level transfer model (payload bytes; profiler may differ after lowering):")
    print(
        f"  tiling: qS={model['q_s_block']} x {model['q_s_blocks']}, "
        f"qN={model['q_n_block']} x {model['q_n_blocks_per_group']} per KV group, "
        f"group={model['group_size']}, tasks={model['total_tasks']}"
    )
    print(
        f"  DataCopy count/task: QK-Q=1, QK-K={model['qk_k_l1_copies']}, "
        f"PV-P={model['pv_p_l1_copies']}, PV-V=1"
    )
    for stage, nbytes in model["stage_bytes"].items():
        print(f"  {stage:>4}: {nbytes / 1024:.1f} KB/task, {nbytes * model['total_tasks'] / 1024:.1f} KB/run")
    print(
        f"  total: {model['logical_bytes_per_task'] / 1024:.1f} KB/task, "
        f"{model['logical_bytes_per_task'] * model['total_tasks'] / 1024:.1f} KB/run"
    )



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


def build_inputs(seq_len: int, device: str):
    q = torch.randn(NUM_SEQS, seq_len, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    k = torch.randn(NUM_SEQS, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    v = torch.randn(NUM_SEQS, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    return q, k, v


def make_scheduler_metadata(q, k, v):
    batch_size, seqlen_q, num_heads, head_size = q.shape
    seqlen_k, num_heads_k = k.shape[1], k.shape[2]
    cache_seqlens = torch.full(
        (batch_size,), seqlen_k, dtype=torch.int32, device=q.device
    )
    return get_scheduler_metadata(
        batch_size, seqlen_q, seqlen_k, num_heads, num_heads_k, head_size,
        cache_seqlens,
        qkv_dtype=q.dtype,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=SCALE,
    )


def save_scheduler_metadata(meta, path):
    params = getattr(meta, "_fa_scheduler_params", {})
    torch.save({"meta": meta.cpu().contiguous(), "params": params}, path)
    print(f"  saved scheduler_metadata -> {path} "
          f"shape={tuple(meta.shape)} nbytes={meta.nbytes} params={json.dumps(params)}")


def load_scheduler_metadata(path, device):
    loaded = torch.load(path, map_location="cpu")
    meta = loaded["meta"].to(device).contiguous()
    meta._fa_scheduler_params = loaded["params"]
    print(f"  loaded scheduler_metadata -> {path} "
          f"shape={tuple(meta.shape)} nbytes={meta.nbytes}")
    return meta


def prepare_scheduler_metadata(q, k, v, save_path, load_path):
    if save_path is not None:
        save_scheduler_metadata(make_scheduler_metadata(q, k, v), save_path)
    if load_path is not None:
        return load_scheduler_metadata(load_path, q.device)
    return None


def run_flash(q, k, v, scheduler_metadata=None):
    kwargs = dict(
        causal=False,
        window_size=[-1, -1],
        alibi_slopes=None,
        return_attn_probs=False,
    )
    if FLASH_ATTN_SUPPORTS_SCHEDULER_METADATA:
        if scheduler_metadata is None:
            scheduler_metadata = make_scheduler_metadata(q, k, v)
        kwargs["scheduler_metadata"] = scheduler_metadata
    flash_attn_func(
        q, k, v, 0.0,
        **kwargs,
    )
    return 1


def run_fusion(q, k, v):
    torch_npu.npu_fusion_attention(
        q, k, v, NUM_HEADS,
        INPUT_LAYOUT,
        pse=None, padding_mask=None, atten_mask=None,
        scale=SCALE, keep_prob=1.0,
        pre_tockens=65536, next_tockens=65536,
        inner_precise=0,
    )
    return 1


def parse_iter3_ms(ascend_pt):
    csv_paths = sorted(glob.glob(os.path.join(ascend_pt, "**/kernel_details.csv"), recursive=True))
    if not csv_paths:
        raise RuntimeError(
            f"profiler analyse 未生成 kernel_details.csv: {ascend_pt}; "
            "请检查原始采集目录、磁盘空间和 profiler 日志"
        )
    csv_path = csv_paths[-1]
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    dur_col = "Duration(us)"
    flash_ms = fusion_ms = None
    for r in rows:
        name, typ = r["Name"], r["Type"]
        if AUX_PAT.search(name) or AUX_PAT.search(typ):
            continue
        if not r.get(dur_col):
            continue
        dur = float(r[dur_col]) / 1000.0
        if "SplitFuse" in name or "FAInfer" in name:
            flash_ms = dur
        elif "FlashAttention" in name or "FlashAttention" in typ:
            fusion_ms = dur
    if flash_ms is None or fusion_ms is None:
        raise RuntimeError(
            f"kernel_details.csv 中未找到目标算子: flash={flash_ms}, fusion={fusion_ms}; "
            f"文件: {csv_path}"
        )
    return flash_ms, fusion_ms


def run_case(case: dict, profile_root: str, cutoffs: tuple, l2_cache: bool,
             save_metadata: str = None, load_metadata: str = None):
    q, k, v = build_inputs(case["seq_len"], case["device"])
    scheduler_metadata = prepare_scheduler_metadata(q, k, v, save_metadata, load_metadata)

    prof_dir = os.path.join(profile_root, case["label"])
    os.makedirs(prof_dir, exist_ok=True)
    existing_ascend_pts = set(glob.glob(os.path.join(prof_dir, "*_ascend_pt")))

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        data_simplification=False,
        l2_cache=True,
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
        # schedule=torch_npu.profiler.schedule(
        #     wait=0, warmup=0, active=1, repeat=1, skip_first=0
        # ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir, analyse_flag=False),
    )

    for stepprof in range(1, TARGET_ITER + 2):
        if stepprof == TARGET_ITER:
            prof.start()
            run_flash(q, k, v, scheduler_metadata)
            run_fusion(q, k, v)
            prof.step()
            prof.step()  # advance past RECORD so the trace is finalized cleanly
            prof.stop()
            break
        run_flash(q, k, v, scheduler_metadata)
        run_fusion(q, k, v)

    new_ascend_pts = set(glob.glob(os.path.join(prof_dir, "*_ascend_pt"))) - existing_ascend_pts
    if len(new_ascend_pts) != 1:
        raise RuntimeError(
            f"expected one new profiler trace under {prof_dir}, found {len(new_ascend_pts)}"
        )
    ascend_pt = new_ascend_pts.pop()
    analyse(profiler_path=ascend_pt, max_process_number=16, export_type="text")
    flash_ms, fusion_ms = parse_iter3_ms(ascend_pt)

    delta_ms = flash_ms - fusion_ms if flash_ms and fusion_ms else None
    ratio = flash_ms / fusion_ms if flash_ms and fusion_ms else None
    degrade_pct = (ratio - 1.0) * 100 if ratio else None

    return {
        **case,
        "bucket": token_bucket(case["total_tokens"], cutoffs),
        "flash_ms": flash_ms,
        "fusion_ms": fusion_ms,
        "delta_ms": delta_ms,
        "ratio": ratio,
        "degrade_pct": degrade_pct,
    }


def analyze_results(results: list):
    ok = [r for r in results if r.get("flash_ms") and r.get("fusion_ms")]
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
    print("flash_attn_func 相对 npu_fusion_attention 劣化分析（第3次迭代 kernel Duration）")
    print("=" * 90)
    print(f"有效用例: {len(ok)}/{len(results)}")
    print(f"ratio  (flash/fusion): mean={r_mean:.3f}x  median={r_med:.3f}x  min={r_min:.3f}x  max={r_max:.3f}x")
    print(f"delta  (flash-fusion): mean={d_mean:.3f}ms median={d_med:.3f}ms min={d_min:.3f}ms max={d_max:.3f}ms")
    print(f"劣化百分比 (ratio-1): mean={(r_mean-1)*100:.1f}%  median={(r_med-1)*100:.1f}%")

    for bucket in ("small", "medium", "large"):
        sub = [r for r in ok if r["bucket"] == bucket]
        if not sub:
            continue
        br = [r["ratio"] for r in sub]
        bd = [r["delta_ms"] for r in sub]
        tok = [r["total_tokens"] for r in sub]
        sl = [r["seq_len"] for r in sub]
        print(
            f"\n[{bucket}] n={len(sub)} seq_len=[{min(sl)},{max(sl)}] tokens=[{min(tok)},{max(tok)}] "
            f"ratio mean={mean(br):.3f}x median={median(br):.3f}x | "
            f"delta mean={mean(bd):.3f}ms median={median(bd):.3f}ms"
        )

    worst = sorted(ok, key=lambda r: r["ratio"], reverse=True)[:5]
    best = sorted(ok, key=lambda r: r["ratio"])[:5]
    print("\n劣化最严重 TOP5 (ratio):")
    for r in worst:
        print(f"  {r['label']} S={r['seq_len']} tokens={r['total_tokens']} "
              f"flash={r['flash_ms']:.3f} fusion={r['fusion_ms']:.3f} ratio={r['ratio']:.3f}x")
    print("\n劣化最小 TOP5 (ratio):")
    for r in best:
        print(f"  {r['label']} S={r['seq_len']} tokens={r['total_tokens']} "
              f"flash={r['flash_ms']:.3f} fusion={r['fusion_ms']:.3f} ratio={r['ratio']:.3f}x")
    print("=" * 90)


def save_csv(results: list, path: str):
    fields = [
        "case_id", "label", "num_seqs", "seq_len", "total_tokens",
        "bucket", "flash_ms", "fusion_ms", "delta_ms", "ratio", "degrade_pct",
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
    parser = argparse.ArgumentParser(
        description="am_flash_attn BSND 放宽规格 · flash_attn_func 50-case vs fusion profile"
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--num-cases", type=int, default=NUM_CASES)
    parser.add_argument(
        "--fixed-seq-len", type=int,
        help="profile exactly one deterministic S value; use 512 for the GM->L1 A/B case",
    )
    parser.add_argument("--seed", type=int, help="seed Python and Torch RNGs for reproducible random-S runs")
    parser.add_argument("--l2-cache", action="store_true", help="export L2Cache.csv with the profiler output")
    parser.add_argument("--save-metadata", default=None,
                        help="path to save precomputed scheduler_metadata (real NPU only)")
    parser.add_argument("--load-metadata", default=None,
                        help="path to load precomputed scheduler_metadata (avoids AICPU launch)")
    parser.add_argument("--profile-root", default=None)
    parser.add_argument("--result-csv", default=None)
    args = parser.parse_args()

    profile_root, result_csv = resolve_output_paths(args.profile_root, args.result_csv)

    if not torch.npu.is_available():
        raise RuntimeError("未检测到可用 NPU 设备")

    torch.npu.set_device(args.device)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.npu.manual_seed_all(args.seed)

    cases = fixed_case(args.fixed_seq_len) if args.fixed_seq_len is not None else select_balanced_cases(args.num_cases)
    cutoffs = tercile_cutoffs(cases)
    for case in cases:
        case["device"] = args.device

    case_mode = f"fixed S={args.fixed_seq_len}" if args.fixed_seq_len is not None else \
        f"random seq_len [{MIN_SEQ_LEN},{MAX_SEQ_LEN}] avg{AVG_SEQ_LEN}"
    print(f"am_flash_attn BSND 放宽 | {len(cases)} 用例 | batch={NUM_SEQS} {case_mode}")
    print(f"GQA {NUM_HEADS}/{NUM_KV_HEADS}/{HEAD_DIM} bf16 | flash causal=False | fusion sparse_mode=0")
    print(f"seq_len 范围: {cases[0]['seq_len']} ~ {cases[-1]['seq_len']} | tokens: {cases[0]['total_tokens']} ~ {cases[-1]['total_tokens']}")
    print(f"分桶(三分位): small<={cutoffs[0]} medium<={cutoffs[1]} large>{cutoffs[1]}")
    print(f"profiler L2Cache.csv: {'enabled' if args.l2_cache else 'disabled'}")
    if args.fixed_seq_len is not None:
        print_transfer_model(args.fixed_seq_len)

    results = []
    for case in cases:
        print(
            f"\n>>> [{case['case_id']:02d}/{len(cases)}] {case['label']} "
            f"num_seqs={NUM_SEQS} S={case['seq_len']} tokens={case['total_tokens']} "
            f"bucket={token_bucket(case['total_tokens'], cutoffs)}"
        )
        try:
            r = run_case(case, profile_root, cutoffs, args.l2_cache,
                         args.save_metadata, args.load_metadata)
            results.append(r)
            print(
                f"    flash={r['flash_ms']:.3f}ms fusion={r['fusion_ms']:.3f}ms "
                f"delta={r['delta_ms']:+.3f}ms ratio={r['ratio']:.3f}x degrade={r['degrade_pct']:.1f}%"
            )
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({**case, "flash_ms": None, "fusion_ms": None})

    save_csv(results, result_csv)

    print("\n" + "-" * 96)
    print(f"{'id':>3} {'seq':>5} {'tokens':>7} {'bucket':>6} {'flash':>8} {'fusion':>8} {'delta':>8} {'ratio':>7}")
    print("-" * 96)
    for r in results:
        if not r.get("flash_ms"):
            continue
        print(
            f"{r['case_id']:>3} {r['seq_len']:>5} {r['total_tokens']:>7} {r['bucket']:>6} "
            f"{r['flash_ms']:>8.3f} {r['fusion_ms']:>8.3f} {r['delta_ms']:>+8.3f} {r['ratio']:>7.3f}x"
        )

    analyze_results(results)


if __name__ == "__main__":
    main()

