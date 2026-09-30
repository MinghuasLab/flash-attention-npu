#!/usr/bin/env python3
"""Validate that a release tag matches the package's single public version."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def load_package_version(repo_root: Path) -> str:
    version_path = repo_root / "flash_attn_npu" / "_version.py"
    spec = importlib.util.spec_from_file_location("_release_package_version", version_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load package version from {version_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.__version__)


def validate_release_version(tag: str, version: str) -> None:
    if not tag.startswith("v") or len(tag) == 1:
        raise ValueError(f"release tag must match v<version>, got {tag!r}")
    if tag[1:] != version:
        raise ValueError(f"release tag {tag!r} does not match package version {version!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    version = load_package_version(args.repo_root)
    if args.tag:
        validate_release_version(args.tag, version)
    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
