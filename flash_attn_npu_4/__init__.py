__version__ = "0.3.0"

import torch_npu


def is_ascend910() -> bool:
    """Return True if the current device belongs to Ascend 910B/C."""
    device_name = torch_npu.npu.get_device_name()
    return "Ascend910B" in device_name or "Ascend910C" in device_name


def is_ascend950() -> bool:
    """Return True if the current device belongs to Ascend 950."""
    device_name = torch_npu.npu.get_device_name()
    return "Ascend950" in device_name


if is_ascend910():
    from .flash_attn_npu_interface import (
        flash_attn_func,
        flash_attn_varlen_func,
        get_scheduler_metadata,
    )
elif is_ascend950():
    from .flash_attn_npu_interface_950 import flash_attn_varlen_func
else:
    raise RuntimeError(f"Unsupported Ascend device: {torch_npu.npu.get_device_name()}")
