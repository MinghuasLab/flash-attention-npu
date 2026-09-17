__version__ = "0.3.0"

import torch_npu


def is_ascend910() -> bool:
    """Return True if the current device belongs to Ascend 910B/C."""
    device_name = torch_npu.npu.get_device_name()
    return "Ascend910" in device_name


if is_ascend910():
    from .flash_attn_npu_interface import (
        flash_attn_func,
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_with_kvcache,
        get_scheduler_metadata,
    )
else:
    raise RuntimeError(f"Unsupported Ascend device: {torch_npu.npu.get_device_name()}")
