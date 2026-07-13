__version__ = "0.1.1"

import torch_npu


def is_arch22() -> bool:
    """Return True if the current device belongs to arch22."""
    device_name = torch_npu.npu.get_device_name()
    return "Ascend910B" in device_name or "Ascend910C" in device_name


if is_arch22():
    from .interface_arch22 import (
        flash_attn_func,
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_with_kvcache,
    )
else:
    raise RuntimeError(f"Unsupported Ascend device: {torch_npu.npu.get_device_name()}")
