#!/usr/bin/env python3
"""Re-key an existing ATK report (xlsx) onto renumbered case ids.

After ``renumber_case_ids.py`` the case ids are 0..N-1 in file order, but every
report produced earlier is keyed by the old ids. This script maps the old report
rows onto the new ids **by case payload** (shapes + dtypes + ranges), rewrites
the ``编号`` column and drops rows that no longer exist (e.g. the removed fp32
output cases).

Usage:
    python rekey_report_ids.py <old_report.xlsx> <new_json> <old_json_backup> [-o out.xlsx]
"""

import argparse
import os

import pandas as pd


def payload_key(case):
    inputs = case["inputs"]
    return (
        tuple(tuple(i["shape"]) if i.get("shape") else None for i in inputs),
        tuple(i["dtype"] for i in inputs),
        tuple(str(i.get("range_values")) for i in inputs),
    )


def rekey(report, new_json, old_json, out):
    import json

    with open(old_json, encoding="utf-8") as f:
        old_cases = json.load(f)
    with open(new_json, encoding="utf-8") as f:
        new_cases = json.load(f)

    old_id_to_key = {int(c["id"]): payload_key(c) for c in old_cases}
    key_to_new_id = {payload_key(c): int(c["id"]) for c in new_cases}

    df = pd.read_excel(report, sheet_name="statistic")
    df["编号"] = df["编号"].astype(int)
    mapped, dropped = [], []
    for case_id in df["编号"]:
        key = old_id_to_key.get(int(case_id))
        new_id = key_to_new_id.get(key) if key is not None else None
        mapped.append(new_id)
        if new_id is None:
            dropped.append(int(case_id))

    df["编号"] = mapped
    out_df = df[df["编号"].notna()].copy()
    out_df["编号"] = out_df["编号"].astype(int)
    out_df = out_df.sort_values("编号").reset_index(drop=True)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="statistic", index=False)

    print(f"{os.path.basename(report)}: {len(df)} rows -> {len(out_df)} kept, "
          f"{len(dropped)} dropped {dropped[:8]}{'...' if len(dropped) > 8 else ''}")
    print(f"  written: {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("new_json")
    ap.add_argument("old_json")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()
    out = args.output or os.path.join(
        os.path.dirname(args.report), "rekeyed_" + os.path.basename(args.report)
    )
    rekey(args.report, args.new_json, args.old_json, out)


if __name__ == "__main__":
    main()
