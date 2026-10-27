#!/usr/bin/env python3
"""Renumber the ATK case ids to 0..N-1 in file order.

ATK aligns benchmark rows by ``编号`` (case id), so renumbering invalidates any
existing GPU baseline report that was produced from the previous ids. The script
always writes a ``*.pre_renumber.bak`` next to each file and prints the
old -> new mapping so the baselines can be regenerated or re-keyed.

Usage:
    python renumber_case_ids.py --dry-run
    python renumber_case_ids.py
"""

import argparse
import json
import os
import shutil

BASE_DIR = "/data/w00896777/ATK/flash_attn_combine"
FILES = (
    os.path.join(BASE_DIR, "all_test_flash_attn_combine_bf16fp16.json"),
    os.path.join(BASE_DIR, "all_test_flash_attn_combine_v4_bf16fp16.json"),
)


def renumber(path, dry_run):
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    old_ids = [int(c["id"]) for c in cases]
    mapping = {old: new for new, old in enumerate(old_ids)}
    changed = [c for c, old in zip(cases, old_ids) if old != mapping[old]]

    print(f"== {os.path.basename(path)}: {len(cases)} cases, "
          f"{len(changed)} ids change, range {min(old_ids)}..{max(old_ids)} -> 0..{len(cases)-1}")

    for index, case in enumerate(cases):
        case["id"] = index

    if dry_run:
        return mapping

    backup = path + ".pre_renumber.bak"
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
        print(f"   backup: {backup}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(cases, ensure_ascii=False))
    print(f"   written: {path}")
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mappings = []
    for path in FILES:
        mappings.append(renumber(path, args.dry_run))

    if not args.dry_run and mappings:
        same = mappings[0] == mappings[1]
        print(f"\nFA3/FA4 old->new mapping identical: {same}")
        for old in (70, 78, 99, 208, 215, 217):
            if old in mappings[0]:
                print(f"  old id {old:>3} -> new id {mappings[0][old]:>3}")


if __name__ == "__main__":
    main()
