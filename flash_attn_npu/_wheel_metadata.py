"""Build and release metadata shared by setup.py and CI tooling.

This module intentionally has no PyTorch or torch_npu dependency.  Importing it
must be safe on a plain packaging runner, while a build environment can still
use the installed frameworks to fill in missing values.
"""

from __future__ import annotations

import os
import platform
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from packaging.version import Version


def _load_public_version() -> str:
    version_path = Path(__file__).with_name("_version.py")
    spec = importlib.util.spec_from_file_location("_flash_attn_npu_version", version_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load version from {version_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.__version__)


PUBLIC_VERSION = _load_public_version()

PACKAGE_NAME = "flash_attn_npu"
DEFAULT_REPOSITORY = "MinghuasLab/flash-attention-npu"


def normalize_token(value: object, *, fallback: str = "unknown") -> str:
    """Return a compact, wheel-local-version-safe token."""

    text = str(value or fallback).strip().lower()
    chars = [char if char.isalnum() else "" for char in text]
    token = "".join(chars)
    return token or fallback


def major_minor_token(value: object) -> str:
    try:
        version = Version(str(value))
    except Exception:
        return normalize_token(value)
    return f"{version.major}{version.minor}"


def _env_or_detect(env_name: str, detector, fallback: str) -> str:
    value = os.getenv(env_name)
    if value:
        return value
    try:
        value = detector()
    except Exception:
        value = None
    return str(value) if value else fallback


def _torch_info() -> tuple[str, str, str]:
    try:
        import torch

        torch_version = str(torch.__version__)
        abi = str(bool(torch._C._GLIBCXX_USE_CXX11_ABI)).upper()
    except Exception:
        torch_version, abi = "unknown", "DETECTED"
    try:
        import torch_npu

        torch_npu_version = str(torch_npu.__version__)
    except Exception:
        torch_npu_version = "unknown"
    return torch_version, torch_npu_version, abi


def _detect_cann() -> str:
    try:
        import torch_npu

        return str(torch_npu.utils.get_cann_version())
    except Exception:
        return "unknown"


def _detect_npu() -> str:
    try:
        import torch_npu

        device_name = str(torch_npu.npu.get_device_name())
    except Exception:
        return "all"
    if "Ascend950" in device_name:
        return "950"
    if "Ascend910" in device_name:
        return "910"
    return "all"


@dataclass(frozen=True)
class BuildConfig:
    version: str
    npu: str
    cann: str
    torch: str
    torch_npu: str
    api: str
    abi: str
    python_tag: str
    abi_tag: str
    platform_tag: str
    repository: str

    @property
    def local_version(self) -> str:
        return (
            f"npu{normalize_token(self.npu)}"
            f"cann{major_minor_token(self.cann)}"
            f"torch{major_minor_token(self.torch)}"
            f"torchnpu{normalize_token(self.torch_npu)}"
            f"api{normalize_token(self.api)}"
            f"abi{normalize_token(self.abi)}"
        )

    @property
    def wheel_version(self) -> str:
        separator = "." if "+" in self.version else "+"
        return f"{self.version}{separator}{self.local_version}"


def get_build_config(*, version: Optional[str] = None) -> BuildConfig:
    torch_version, torch_npu_version, detected_abi = _torch_info()
    public_version = version or PUBLIC_VERSION
    local_version = os.getenv("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        public_version = f"{public_version}+{normalize_token(local_version)}"
    return BuildConfig(
        version=public_version,
        npu=os.getenv("FLASH_ATTN_BUILD_NPU", _detect_npu()).lower(),
        cann=_env_or_detect("FLASH_ATTN_CANN_VERSION", _detect_cann, "unknown"),
        torch=_env_or_detect("FLASH_ATTN_TORCH_VERSION", lambda: torch_version, "unknown"),
        torch_npu=_env_or_detect(
            "FLASH_ATTN_TORCH_NPU_VERSION", lambda: torch_npu_version, "unknown"
        ),
        api=os.getenv("FLASH_ATTN_WHEEL_VARIANT", os.getenv("FLASH_ATTN_BUILD_VERSION", "all")),
        abi=os.getenv("FLASH_ATTN_FORCE_CXX11_ABI", detected_abi).upper(),
        python_tag=os.getenv(
            "FLASH_ATTN_PYTHON_TAG", f"cp{sys.version_info.major}{sys.version_info.minor}"
        ),
        abi_tag=os.getenv(
            "FLASH_ATTN_PYTHON_ABI_TAG", f"cp{sys.version_info.major}{sys.version_info.minor}"
        ),
        platform_tag=os.getenv("FLASH_ATTN_PLATFORM_TAG", f"linux_{platform.machine()}"),
        repository=os.getenv(
            "FLASH_ATTN_WHEEL_REPOSITORY", os.getenv("GITHUB_REPOSITORY", DEFAULT_REPOSITORY)
        ),
    )


def get_wheel_filename(config: Optional[BuildConfig] = None) -> str:
    config = config or get_build_config()
    return (
        f"{PACKAGE_NAME}-{config.wheel_version}-{config.python_tag}-"
        f"{config.abi_tag}-{config.platform_tag}.whl"
    )


def get_release_url(config: Optional[BuildConfig] = None, *, tag: Optional[str] = None) -> str:
    config = config or get_build_config()
    tag_name = tag or f"v{config.version.split('+', 1)[0]}"
    return f"https://github.com/{config.repository}/releases/download/{tag_name}/{get_wheel_filename(config)}"


def public_version() -> str:
    return PUBLIC_VERSION


__all__ = [
    "BuildConfig",
    "DEFAULT_REPOSITORY",
    "get_build_config",
    "get_release_url",
    "get_wheel_filename",
    "major_minor_token",
    "normalize_token",
    "public_version",
]
