#!/usr/bin/env python3
# Copyright (c) 2026, Minghua Shen.

"""Validate release wheel naming, metadata, and selected native extensions."""

from __future__ import annotations

import argparse
import email.parser
import importlib.util
import sys
import zipfile
from pathlib import Path


def load_metadata_module(repo_root: Path):
    module_path = repo_root / "flash_attn_npu" / "_wheel_metadata.py"
    spec = importlib.util.spec_from_file_location("_validate_wheel_metadata", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load wheel metadata module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rename_release_wheel(wheel_path: Path, repo_root: Path) -> Path:
    wheel_path = wheel_path.resolve()
    metadata_module = load_metadata_module(repo_root)
    config = metadata_module.get_build_config()
    public_name = (
        f"{metadata_module.PACKAGE_NAME}-{config.version}-{config.python_tag}-"
        f"{config.abi_tag}-{config.platform_tag}.whl"
    )
    if wheel_path.name != public_name:
        raise ValueError(
            f"built wheel name mismatch: expected {public_name}, got {wheel_path.name}"
        )
    release_path = wheel_path.with_name(metadata_module.get_wheel_filename(config))
    wheel_path.rename(release_path)
    return release_path


def validate_wheel(wheel_path: Path, repo_root: Path) -> None:
    metadata_module = load_metadata_module(repo_root)
    config = metadata_module.get_build_config()
    expected_name = metadata_module.get_wheel_filename(config)
    if wheel_path.name != expected_name:
        raise ValueError(f"wheel name mismatch: expected {expected_name}, got {wheel_path.name}")

    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError(f"expected one METADATA file, found {len(metadata_names)}")
        expected_metadata_name = (
            f"{metadata_module.PACKAGE_NAME}-{config.version}.dist-info/METADATA"
        )
        if metadata_names[0] != expected_metadata_name:
            raise ValueError(
                f"wheel metadata path mismatch: {metadata_names[0]} != {expected_metadata_name}"
            )
        metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_names[0]))
        if metadata["Version"] != config.version:
            raise ValueError(
                f"wheel metadata version mismatch: {metadata['Version']} != {config.version}"
            )
        if "flash_attn_npu/__init__.py" not in names:
            raise ValueError("wheel does not contain flash_attn_npu package")

        shared_objects = [name for name in names if name.endswith(".so")]
        if config.npu == "910" and any("_950" in name for name in shared_objects):
            raise ValueError("910 wheel contains a 950 extension")
        if config.npu == "950" and any(
            name.startswith(("flash_attn_npu/", "flash_attn_npu_3/", "flash_attn_npu_4/"))
            for name in shared_objects
        ):
            raise ValueError("950 wheel contains a 910 extension")

        required = []
        if config.npu in {"910", "all"}:
            if config.api in {"v2", "all"}:
                required.append("flash_attn_npu/flash_attn_npu")
            if config.api in {"v3", "all"}:
                required.append("flash_attn_npu_3/flash_attn_npu_3")
            if config.api in {"v4", "all"}:
                required.append("flash_attn_npu_4/flash_attn_npu_4")
        if config.npu in {"950", "all"}:
            if config.api in {"v3", "all"}:
                required.append("flash_attn_npu_3_950")
            if config.api in {"v4", "all"}:
                required.append("flash_attn_npu_4_950")
        missing = [
            prefix
            for prefix in required
            if not any(name.startswith(prefix) for name in shared_objects)
        ]
        if missing:
            raise ValueError(f"wheel is missing required extensions: {', '.join(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rename", action="store_true")
    args = parser.parse_args()
    wheel_path = rename_release_wheel(args.wheel, args.repo_root) if args.rename else args.wheel
    validate_wheel(wheel_path, args.repo_root)
    print(wheel_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
