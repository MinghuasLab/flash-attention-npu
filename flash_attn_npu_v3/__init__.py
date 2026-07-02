__version__ = "0.1.1"

import torch_npu

if "Ascend950" in torch_npu.npu.get_device_name():
    from .flash_attn_interface_950 import flash_attn_with_kvcache
else:
    from flash_attn_npu_v3.flash_attn_interface import (
        flash_attn_func,
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
        get_scheduler_metadata,
    )
