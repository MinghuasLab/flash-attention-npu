#!/usr/bin/env python3
"""Extend the bf16/fp16 flash_attn_combine case files from 182 to 200 cases.

The 182 existing cases are the bf16/fp16 subset of the 200-case GPU baseline
report (18 fp32-output rows removed) and are kept byte-identical.  This script
appends 18 new bf16/fp16 cases (ids 200..217) that close the coverage gaps of
the original set:

  * num_splits 6 / 7 / 9 / 15 / 64 / 128 / 256  (接口上限 256，压 Kahan 累加)
  * head_dim 32 / 48 / 80 / 113 / 160 / 200    (非 8/16 对齐 -> pad 路径)
  * T / S 2 / 7 / 65 / 129 / 513 / 1000        (尾块与长序列)
  * heads 3 / 12 / 16、batch 6 / 12

New ids start at 200 so they never collide with the existing GPU baseline rows;
the new cases need a GPU baseline refresh (see the printout at the end) before
they can take part in the L2 double benchmark.

Usage:
    python extend_cases_to_200.py                 # rewrite both files in place
    python extend_cases_to_200.py --dry-run       # print the plan only
"""

import argparse
import json
import os

BASE_DIR = "/data/w00896777/ATK/flash_attn_combine"
FA3_FILE = os.path.join(BASE_DIR, "all_test_flash_attn_combine_bf16fp16.json")
FA4_FILE = os.path.join(BASE_DIR, "all_test_flash_attn_combine_v4_bf16fp16.json")

ND_RANGE = {"name": "nd", "mean": [-100.0, 100.0], "std": [1.0, 25.0], "range": None}
UNIFORM_RANGE = [-5, 5]

# (rank, shape, out_partial range, out_dtype, 说明)
NEW_SPECS = [
    # ---- TND (rank 4, shape = [num_splits, total_q, nheads, head_dim]) ----
    (4, [64, 128, 8, 128], ND_RANGE, "fp16", "splits=64 中等序列"),
    (4, [256, 16, 2, 64], ND_RANGE, "bf16", "splits=256 接口上限"),
    (4, [7, 65, 16, 80], UNIFORM_RANGE, "fp16", "splits=7 + D=80 非对齐"),
    (4, [15, 7, 3, 48], UNIFORM_RANGE, "bf16", "splits=15 + 尾块 T=7"),
    (4, [32, 1000, 1, 256], ND_RANGE, "bf16", "长序列 T=1000 + D=256"),
    (4, [9, 2, 2, 32], UNIFORM_RANGE, "bf16", "splits=9 + D=32 最小值"),
    (4, [6, 129, 64, 96], ND_RANGE, "fp16", "splits=6 + H=64"),
    (4, [128, 4, 8, 192], UNIFORM_RANGE, "bf16", "splits=128 + 短序列"),
    (4, [17, 513, 5, 113], ND_RANGE, "fp16", "D=113 + T=513 尾块"),
    (4, [3, 257, 12, 160], UNIFORM_RANGE, "bf16", "H=12 + D=160"),
    # ---- BSND (rank 5, shape = [num_splits, batch, seqlen, nheads, head_dim]) ----
    (5, [64, 2, 8, 4, 128], ND_RANGE, "fp16", "splits=64 dense"),
    (5, [256, 1, 4, 2, 64], UNIFORM_RANGE, "bf16", "splits=256 dense"),
    (5, [8, 6, 33, 16, 96], ND_RANGE, "fp16", "B=6 + H=16"),
    (5, [9, 1, 1, 12, 32], UNIFORM_RANGE, "bf16", "decode S=1 + D=32"),
    (5, [128, 1, 2, 8, 256], UNIFORM_RANGE, "fp16", "splits=128 + D=256"),
    (5, [5, 12, 7, 3, 200], ND_RANGE, "bf16", "B=12 + D=200 非对齐"),
    (5, [15, 4, 65, 5, 48], UNIFORM_RANGE, "fp16", "splits=15 + S=65"),
    (5, [64, 1, 128, 8, 128], ND_RANGE, "bf16", "splits=64 + S=128"),
]


def build_case(case_id, template, rank, shape, out_range, dtype):
    lse_shape = shape[:-1]
    if rank not in (4, 5):
        raise ValueError(f"rank must be 4 or 5, got {rank}")
    if len(shape) != rank:
        raise ValueError(f"shape {shape} does not match rank {rank}")
    if dtype not in ("fp16", "bf16"):
        raise ValueError(f"out_dtype must be fp16/bf16, got {dtype}")
    if not 1 <= shape[0] <= 256:
        raise ValueError(f"num_splits {shape[0]} out of [1, 256]")
    case = {
        "id": case_id,
        "name": template["name"],
        "api": template["api"],
        "api_type": template["api_type"],
        "backward": False,
        "standard": template["standard"],
        "inputs": [
            {
                "name": "out_partial",
                "type": "tensor",
                "required": True,
                "dtype": "fp32",
                "shape": list(shape),
                "range_values": out_range,
                "backward": False,
                "align_32B": None,
                "outlier_values": None,
            },
            {
                "name": "lse_partial",
                "type": "tensor",
                "required": True,
                "dtype": "fp32",
                "shape": list(lse_shape),
                "range_values": [-5, 5],
                "backward": False,
                "align_32B": None,
                "outlier_values": None,
            },
            {
                "name": "out_dtype",
                "type": "attr",
                "required": True,
                "dtype": "string",
                "shape": None,
                "range_values": dtype,
                "backward": False,
                "align_32B": None,
                "outlier_values": None,
            },
        ],
    }
    return case


def extend(path, dry_run):
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    # Drop a previous run of this script so the operation is idempotent.
    kept = [c for c in cases if int(c["id"]) < 200]
    dropped = len(cases) - len(kept)
    template = kept[0]
    added = []
    for offset, (rank, shape, out_range, dtype, note) in enumerate(NEW_SPECS):
        case = build_case(200 + offset, template, rank, shape, out_range, dtype)
        added.append(case)
        print(f"  id {case['id']:3d} rank{rank} shape={shape} {dtype:4s}  # {note}")

    ids = [int(c["id"]) for c in kept + added]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicated case ids")
    if len(kept) != 182 or len(added) != 18:
        raise ValueError(f"{path}: expected 182 kept + 18 new, got {len(kept)} + {len(added)}")

    print(f"{os.path.basename(path)}: {len(cases)} -> {len(kept) + len(added)} "
          f"(dropped {dropped} stale id>=200 rows)")
    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(kept + added, ensure_ascii=False))
        print(f"  written: {path}")
    return kept + added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for path in (FA3_FILE, FA4_FILE):
        print(f"== {path}")
        extend(path, args.dry_run)
    if not args.dry_run:
        print(
            "\n下一步（在 GPU 机器上重跑一次基线，让新 18 条也有分母）：\n"
            "  atk node --backend gpu --devices 0 --name gpu_base \\\n"
            "    task -c all_test_flash_attn_combine_bf16fp16.json \\\n"
            "    -p execute_flash_attn_combine.py --task accuracy\n"
            "然后把新生成的 *_reports_*.xlsx 作为 --bm_file 与同一份 JSON 配对使用。"
        )


if __name__ == "__main__":
    main()
