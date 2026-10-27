__version__ = "0.4.2"

import torch_npu


def _device_name() -> str:
    try:
        return torch_npu.npu.get_device_name() or ""
    except Exception:
        return ""


def is_ascend910() -> bool:
    """Return True if the current device belongs to Ascend 910B/C."""
    return "Ascend910" in _device_name()


def is_ascend950() -> bool:
    """Return True if the current device belongs to Ascend 950."""
    return "Ascend950" in _device_name()


_device = _device_name()
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
    from .flash_attn_quant_npu import (
        QuantMode,
        flash_attn_quant,
    )
elif _device:
    raise RuntimeError(f"Unsupported Ascend device: {_device}")
