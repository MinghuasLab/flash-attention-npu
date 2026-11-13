#!/usr/bin/env python3
# Copyright (c) 2026, Minghua Shen.

"""Parse and validate the release build matrix."""

from __future__ import annotations

import argparse
import json
import re
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

ASCENDHUB_CANN = "swr.cn-south-1.myhuaweicloud.com/ascendhub/cann"

# Keep this allowlist aligned with Ascend's COMPATIBILITY.md and the CANN
# images that are available on AscendHub. Each CANN family accepts the
# officially paired PyTorch/torch_npu versions below.
SUPPORTED_STACKS = {
    "8.5": {
        "torch_versions": {
            "2.6.0": ("2.6.0.post5", "v7.3.0-pytorch2.6.0"),
            "2.7.1": ("2.7.1.post2", "v7.3.0-pytorch2.7.1"),
            "2.8.0": ("2.8.0.post2", "v7.3.0-pytorch2.8.0"),
            "2.9.0": ("2.9.0", "v7.3.0-pytorch2.9.0"),
        },
        "python_images": {"cp310": "8.5.0-py3.10-ops", "cp311": "8.5.0-py3.11"},
        "npu": {"910"},
    },
    "9.0": {
        "torch_versions": {
            "2.7.1": ("2.7.1.post4", "v26.0.0-pytorch2.7.1"),
            "2.8.0": ("2.8.0.post4", "v26.0.0-pytorch2.8.0"),
            "2.9.0": ("2.9.0.post2", "v26.0.0-pytorch2.9.0"),
            "2.10.0": ("2.10.0", "v26.0.0-pytorch2.10.0"),
        },
        "python_images": {"cp311": "9.0.0-py3.11-devel", "cp312": "9.0.1-py3.12-devel"},
        "npu": {"910", "950"},
    },
    "9.1": {
        "torch_versions": {
            "2.7.1": ("2.7.1.post10", "v26.1.1-pytorch2.7.1"),
            "2.9.0": ("2.9.0.post8", "v26.1.1-pytorch2.9.0"),
            "2.10.0": ("2.10.0.post6", "v26.1.1-pytorch2.10.0"),
            "2.11.0": ("2.11.0.post2", "v26.1.1-pytorch2.11.0"),
            "2.12.0": ("2.12.0.post2", "v26.1.1-pytorch2.12.0"),
        },
        "python_images": {
            "cp310": "9.1.0-py3.10-devel",
            "cp311": "9.1.0-py3.11-devel",
            "cp312": "9.1.0-py3.12-devel",
        },
        "npu": {"910", "950"},
    },
    "9.2": {
        "torch_versions": {
            "2.12.0": ("2.12.0.post3", "v26.2.0-pytorch2.12.0"),
        },
        "python_images": {
            "cp310": "9.2.0-beta.2-py3.10-devel",
            "cp311": "9.2.0-beta.2-py3.11-devel",
            "cp312": "9.2.0-beta.2-py3.12-devel",
        },
        "npu": {"910", "950"},
    },
}


def version_family(value: str) -> str:
    match = re.match(r"^(\d+)\.(\d+)", value)
    if match is None:
        raise ValueError(f"invalid CANN version {value!r}")
    return f"{match.group(1)}.{match.group(2)}"


def validate_stack(row: dict[str, str], location: str) -> str:
    family = version_family(row["cann_version"])
    stack = SUPPORTED_STACKS.get(family)
    if stack is None:
        raise ValueError(f"{location}: unsupported CANN version {row['cann_version']!r}")

    torch_pair = stack["torch_versions"].get(row["torch_version"])
    if torch_pair is None:
        supported = ", ".join(sorted(stack["torch_versions"]))
        raise ValueError(
            f"{location}: PyTorch {row['torch_version']!r} is unavailable for CANN {family}; "
            f"choose {supported}"
        )
    expected_npu, expected_release = torch_pair
    if row["torch_npu_version"] != expected_npu:
        raise ValueError(
            f"{location}: torch_npu {row['torch_npu_version']!r} is incompatible with "
            f"PyTorch {row['torch_version']}; expected {expected_npu!r}"
        )
    if row["torch_npu_release"] != expected_release:
        raise ValueError(
            f"{location}: TorchNPU release {row['torch_npu_release']!r} is incompatible "
            f"with PyTorch {row['torch_version']}; expected {expected_release!r}"
        )

    image_suffix = stack["python_images"].get(row["py_tag"])
    if image_suffix is None:
        supported = ", ".join(sorted(stack["python_images"]))
        raise ValueError(
            f"{location}: Python tag {row['py_tag']!r} is unavailable for CANN {family}; "
            f"choose {supported}"
        )
    if row["npu"] not in stack["npu"]:
        raise ValueError(f"{location}: NPU {row['npu']!r} is unsupported with CANN {family}")

    product = "910b" if row["npu"] == "910" else "950"
    version, python_suffix = image_suffix.split("-py", 1)
    expected_image = f"{ASCENDHUB_CANN}:{version}-{product}-ubuntu22.04-py{python_suffix}"
    accepted_images = {expected_image, expected_image.replace("-devel", "")}
    if row["base_image"] not in accepted_images:
        raise ValueError(
            f"{location}: base_image does not match CANN/Python/NPU selection; "
            f"expected {expected_image!r}"
        )
    if row["base_image"] != expected_image:
        row["base_image"] = expected_image
    return version


def runner_labels(arch: str) -> list[str]:
    machine = {"aarch64": "arm64", "x86_64": "x64"}.get(arch)
    if machine is None:
        raise ValueError(f"unsupported matrix architecture: {arch!r}")
    return ["self-hosted", "linux", machine, "flash-attention-npu"]


def load_matrix(path: Path, matrix_filter: str = "") -> list[dict[str, str]]:
    rows = []
    identities = set()
    names = set()
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
        location = f"{path}:{line_number}"
        for field in FIELDS[:-1]:
            if not row[field]:
                raise ValueError(f"{location}: {field} must not be empty")
        if row["name"] in names:
            raise ValueError(f"{location}: duplicate matrix name {row['name']!r}")
        names.add(row["name"])
        if row["npu"] not in {"910", "950", "all"}:
            raise ValueError(f"{location}: invalid npu {row['npu']!r}")
        if row["build_version"] not in {"v2", "v3", "v4", "all"}:
            raise ValueError(f"{location}: invalid build_version {row['build_version']!r}")
        if row["npu"] == "950" and row["build_version"] == "v2":
            raise ValueError(f"{location}: v2 has no 950 backend")
        if row["abi"].upper() not in {"TRUE", "FALSE", "DETECTED"}:
            raise ValueError(f"{location}: invalid abi {row['abi']!r}")
        cann_install_version = validate_stack(row, location)
        python_version = row["py_tag"].removeprefix("cp")
        row["conda_env"] = (
            f"py{python_version}-torch{row['torch_version']}-cann{cann_install_version}"
        )
        row["cann_install_version"] = cann_install_version
        row["image"] = row["image"] or f"fa-npu-matrix:{row['arch']}"
        row["runs_on"] = json.dumps(runner_labels(row["arch"]), separators=(",", ":"))
        identity = tuple(row[field] for field in FIELDS[2:11])
        if identity in identities:
            raise ValueError(f"{location}: duplicate release artifact configuration")
        identities.add(identity)
        if matrix_filter and matrix_filter not in row["name"]:
            continue
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
