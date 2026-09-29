import importlib.util
from pathlib import Path

import torch_npu

_version_path = Path(__file__).parents[1] / "flash_attn_npu" / "_version.py"
_version_spec = importlib.util.spec_from_file_location("_flash_attn_npu_3_version", _version_path)
if _version_spec is None or _version_spec.loader is None:
    raise RuntimeError(f"Cannot load package version from {_version_path}")
_version_module = importlib.util.module_from_spec(_version_spec)
_version_spec.loader.exec_module(_version_module)
__version__ = _version_module.__version__

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "get_scheduler_metadata",
]


def is_ascend910() -> bool:
    """Return True if the current device belongs to Ascend 910B/C."""
    device_name = torch_npu.npu.get_device_name()
    return "Ascend910" in device_name


def is_ascend950() -> bool:
    """Return True if the current device belongs to Ascend 950."""
    device_name = torch_npu.npu.get_device_name()
    return "Ascend950" in device_name


if is_ascend910():
    from .flash_attn_npu_interface import (
        flash_attn_func,
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
        get_scheduler_metadata,
    )
elif is_ascend950():
    from .flash_attn_npu_interface_950 import (
        flash_attn_func,
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
        get_scheduler_metadata,
    )
else:
    raise RuntimeError(f"Unsupported Ascend device: {torch_npu.npu.get_device_name()}")
