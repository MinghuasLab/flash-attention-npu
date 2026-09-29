#!/usr/bin/env python3
"""Parse and validate the release build matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = (
    "name",
    "base_image",
    "py_tag",
    "torch_version",
    "torch_npu_version",
    "torch_npu_release",
    "cann_version",
    "npu",
    "build_version",
    "abi",
    "arch",
    "image",
)


def runner_labels(arch: str) -> list[str]:
    machine = {"aarch64": "arm64", "x86_64": "x64"}.get(arch)
    if machine is None:
        raise ValueError(f"unsupported matrix architecture: {arch!r}")
    return ["self-hosted", "linux", machine, "npu", "flash-attention-npu"]


def load_matrix(path: Path, matrix_filter: str = "") -> list[dict[str, str]]:
    rows = []
    identities = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split("|")
        if len(values) != len(FIELDS):
            raise ValueError(
                f"{path}:{line_number}: expected {len(FIELDS)} fields, got {len(values)}"
            )
        row = dict(zip(FIELDS, values))
        if matrix_filter and matrix_filter not in row["name"]:
            continue
        if row["npu"] not in {"910", "950", "all"}:
            raise ValueError(f"{path}:{line_number}: invalid npu {row['npu']!r}")
        if row["build_version"] not in {"v2", "v3", "v4", "all"}:
            raise ValueError(
                f"{path}:{line_number}: invalid build_version {row['build_version']!r}"
            )
        if row["npu"] == "950" and row["build_version"] == "v2":
            raise ValueError(f"{path}:{line_number}: v2 has no 950 backend")
        if row["abi"].upper() not in {"TRUE", "FALSE", "DETECTED"}:
            raise ValueError(f"{path}:{line_number}: invalid abi {row['abi']!r}")
        row["image"] = row["image"] or f"fa-npu-ci:{row['name']}"
        row["runs_on"] = json.dumps(runner_labels(row["arch"]), separators=(",", ":"))
        identity = tuple(row[field] for field in FIELDS[2:11])
        if identity in identities:
            raise ValueError(f"{path}:{line_number}: duplicate release artifact configuration")
        identities.add(identity)
        rows.append(row)
    if not rows:
        raise ValueError(f"no release combinations selected from {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=Path("ci/build_matrix.tsv"))
    parser.add_argument("--filter", default="")
    parser.add_argument("--github-output")
    args = parser.parse_args()
    payload = json.dumps({"include": load_matrix(args.file, args.filter)}, separators=(",", ":"))
    print(payload)
    if args.github_output:
        with Path(args.github_output).open("a", encoding="utf-8") as output:
            output.write(f"matrix={payload}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
