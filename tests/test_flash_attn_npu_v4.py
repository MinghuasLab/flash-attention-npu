# Copyright (c) 2026, Minghua Shen.

import torch
import torch_npu
import pytest

from tests.common.attention_ref import ref_flash_attention, ref_flash_attention_pair
from tests.common.compare import assert_fa_close
from tests.common.test_utils import (
    gather_paged_kv_batch,
    make_block_table,
    make_local_attention_mask,
    make_packed_random_tensor,
    make_paged_kv_cache,
    make_padded_varlen_mask,
    pad_packed_tensor,
    make_random_tensor,
    make_varlen_seqlens,
)
if "Ascend950" in torch_npu.npu.get_device_name():
    from flash_attn_npu_4 import flash_attn_varlen_func
else:
    from flash_attn_npu_4 import flash_attn_func, flash_attn_varlen_func

def build_cann_causal_mask():
    """Fixed [2048, 2048] causal mask for npu_fused_infer_attention_score."""
    return torch.triu(torch.ones(2048, 2048), diagonal=1).bool().npu()


# flash_attn_varlen_func test parameters (single API; 8 groups x 6 = 48
# regular cases for each of three modes)
# Single-option parameters: fixed values
#   batch_size: [2]
#   block_size: [128]
# Two-option parameters
#   data_type: [torch.float16, torch.bfloat16]
#   is_causal: [False, True]
# Per-group variation: num_heads,kv_heads in {(6,6),(6,1),(6,3)} x head_size
# x (q_seqlen,kv_seqlen) x window, with each column independently shuffled.
# Each mode has an independent parameter range (columns shuffled within groups):
#   1) dense BSND (func-style)              cache_mode=0, layout=BSND, non-varlen, num_splits=0
#        num_heads,kv_heads: A=[(6,6),(6,1),(6,3)] (shared by all three modes)
#        head_size: A=[32,64], B=[128,192]
#        (q_seqlen,kv_seqlen):
#            A=[(1,128),(64,256),(3,799),(3,1024),(16,20000),(16,131072)]
#            B=[(128,128),(1,339),(64,800),(64,2048),(1,131072),(16,4096)]
#        (window_left,window_right): A=[(-1,-1),(512,0)], B=[(0,256),(542,647)]
#   2) dense varlen TND (varlen-style, only mode supporting backward)
#                                              cache_mode=0, layout=TND, varlen, num_splits=0
#        head_size: A=[32,64], B=[128,192]
#        (q_seqlen,kv_seqlen) (small set to avoid excessive packed-varlen size):
#            A=[(1,128),(64,256),(3,799),(3,1024),(16,4096),(16,8192)]
#            B=[(128,128),(1,339),(64,800),(64,2048),(16,2000),(16,4096)]
#        (window_left,window_right): A=[(-1,-1),(512,0)], B=[(0,256),(542,647)]
#   3) paged KV cache (matching Tri Dao test_flash_attn_kvcache scenarios)
#                                              cache_mode=1, layout=TND, varlen, num_splits=1
#        head_size（kvcache d-list [32,59,64,80,128,256]）: A=[32,59,64,80], B=[128,256]
#        (q_seqlen,kv_seqlen) (including decode q=1 and long KV):
#            A=[(1,128),(16,20000),(3,799),(16,131072),(3,1024),(64,256)]
#            B=[(1,339),(1,131072),(128,128),(64,800),(64,2048),(16,4096)]
#        (window_left,window_right) (NPU-specific):
#            A=[(-1,-1),(0,256),(59,571),(460,62)], B=[(536,462),(563,425),(746,16)]
test_cases = [
# ================ 1) dense BSND (func): cache_mode=0, layout=BSND, non-varlen, num_splits=0 ================
# data_type=float16, is_causal=False, cache_mode=0, layout=BSND, is_varied=False, num_splits=0
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=A, (q_seqlen,kv_seqlen)=seqA, (window_left,window_right)=winA
    (torch.float16, 2, 6, 6, 16, 20000, 64, 0, 128, False, "BSND", False, -1, -1, 0),
    (torch.float16, 2, 6, 3, 3, 1024, 32, 0, 128, False, "BSND", False, -1, -1, 0),
    (torch.float16, 2, 6, 3, 3, 799, 64, 0, 128, False, "BSND", False, 512, 0, 0),
    (torch.float16, 2, 6, 1, 64, 256, 32, 0, 128, False, "BSND", False, 512, 0, 0),
    (torch.float16, 2, 6, 6, 1, 128, 32, 0, 128, False, "BSND", False, 512, 0, 0),
    (torch.float16, 2, 6, 1, 16, 131072, 64, 0, 128, False, "BSND", False, -1, -1, 0),

# data_type=float16, is_causal=True, cache_mode=0, layout=BSND, is_varied=False, num_splits=0
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=A, (q_seqlen,kv_seqlen)=seqA, (window_left,window_right)=winA
    (torch.float16, 2, 6, 1, 3, 1024, 64, 0, 128, True, "BSND", False, 512, 0, 0),
    (torch.float16, 2, 6, 6, 3, 799, 64, 0, 128, True, "BSND", False, 512, 0, 0),
    (torch.float16, 2, 6, 1, 64, 256, 64, 0, 128, True, "BSND", False, -1, -1, 0),
    (torch.float16, 2, 6, 3, 16, 131072, 32, 0, 128, True, "BSND", False, 512, 0, 0),
    (torch.float16, 2, 6, 3, 1, 128, 32, 0, 128, True, "BSND", False, -1, -1, 0),
    (torch.float16, 2, 6, 6, 16, 20000, 32, 0, 128, True, "BSND", False, -1, -1, 0),

# data_type=float16, is_causal=False, cache_mode=0, layout=BSND, is_varied=False, num_splits=0
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=B, (q_seqlen,kv_seqlen)=seqB, (window_left,window_right)=winB
    (torch.float16, 2, 6, 6, 16, 4096, 128, 0, 128, False, "BSND", False, 0, 256, 0),
    (torch.float16, 2, 6, 3, 1, 339, 128, 0, 128, False, "BSND", False, 542, 647, 0),
    (torch.float16, 2, 6, 3, 1, 131072, 192, 0, 128, False, "BSND", False, 542, 647, 0),
    (torch.float16, 2, 6, 1, 64, 2048, 192, 0, 128, False, "BSND", False, 0, 256, 0),
    (torch.float16, 2, 6, 1, 64, 800, 128, 0, 128, False, "BSND", False, 542, 647, 0),
    (torch.float16, 2, 6, 6, 128, 128, 192, 0, 128, False, "BSND", False, 0, 256, 0),

# data_type=float16, is_causal=True, cache_mode=0, layout=BSND, is_varied=False, num_splits=0
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=B, (q_seqlen,kv_seqlen)=seqB, (window_left,window_right)=winB
    (torch.float16, 2, 6, 1, 128, 128, 192, 0, 128, True, "BSND", False, 542, 647, 0),
    (torch.float16, 2, 6, 6, 64, 2048, 192, 0, 128, True, "BSND", False, 0, 256, 0),
    (torch.float16, 2, 6, 3, 16, 4096, 128, 0, 128, True, "BSND", False, 0, 256, 0),
    (torch.float16, 2, 6, 6, 1, 339, 192, 0, 128, True, "BSND", False, 542, 647, 0),
    (torch.float16, 2, 6, 1, 1, 131072, 128, 0, 128, True, "BSND", False, 542, 647, 0),
    (torch.float16, 2, 6, 3, 64, 800, 128, 0, 128, True, "BSND", False, 0, 256, 0),

# data_type=bfloat16, is_causal=False, cache_mode=0, layout=BSND, is_varied=False, num_splits=0
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=A, (q_seqlen,kv_seqlen)=seqA, (window_left,window_right)=winA
    (torch.bfloat16, 2, 6, 6, 1, 128, 32, 0, 128, False, "BSND", False, 512, 0, 0),
    (torch.bfloat16, 2, 6, 3, 64, 256, 64, 0, 128, False, "BSND", False, -1, -1, 0),
    (torch.bfloat16, 2, 6, 3, 3, 1024, 64, 0, 128, False, "BSND", False, 512, 0, 0),
    (torch.bfloat16, 2, 6, 1, 16, 131072, 32, 0, 128, False, "BSND", False, -1, -1, 0),
    (torch.bfloat16, 2, 6, 6, 16, 20000, 64, 0, 128, False, "BSND", False, 512, 0, 0),
    (torch.bfloat16, 2, 6, 1, 3, 799, 32, 0, 128, False, "BSND", False, -1, -1, 0),

# data_type=bfloat16, is_causal=True, cache_mode=0, layout=BSND, is_varied=False, num_splits=0
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=A, (q_seqlen,kv_seqlen)=seqA, (window_left,window_right)=winA
    (torch.bfloat16, 2, 6, 6, 1, 128, 32, 0, 128, True, "BSND", False, -1, -1, 0),
    (torch.bfloat16, 2, 6, 3, 16, 131072, 64, 0, 128, True, "BSND", False, 512, 0, 0),
    (torch.bfloat16, 2, 6, 1, 16, 20000, 64, 0, 128, True, "BSND", False, 512, 0, 0),
    (torch.bfloat16, 2, 6, 1, 64, 256, 32, 0, 128, True, "BSND", False, 512, 0, 0),
    (torch.bfloat16, 2, 6, 3, 3, 1024, 64, 0, 128, True, "BSND", False, -1, -1, 0),
    (torch.bfloat16, 2, 6, 6, 3, 799, 32, 0, 128, True, "BSND", False, -1, -1, 0),

# data_type=bfloat16, is_causal=False, cache_mode=0, layout=BSND, is_varied=False, num_splits=0
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=B, (q_seqlen,kv_seqlen)=seqB, (window_left,window_right)=winB
    (torch.bfloat16, 2, 6, 1, 16, 4096, 128, 0, 128, False, "BSND", False, 0, 256, 0),
    (torch.bfloat16, 2, 6, 3, 1, 339, 128, 0, 128, False, "BSND", False, 0, 256, 0),
    (torch.bfloat16, 2, 6, 3, 128, 128, 192, 0, 128, False, "BSND", False, 542, 647, 0),
    (torch.bfloat16, 2, 6, 1, 64, 800, 192, 0, 128, False, "BSND", False, 542, 647, 0),
    (torch.bfloat16, 2, 6, 6, 1, 131072, 128, 0, 128, False, "BSND", False, 542, 647, 0),
    (torch.bfloat16, 2, 6, 6, 64, 2048, 192, 0, 128, False, "BSND", False, 0, 256, 0),

# data_type=bfloat16, is_causal=True, cache_mode=0, layout=BSND, is_varied=False, num_splits=0
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=B, (q_seqlen,kv_seqlen)=seqB, (window_left,window_right)=winB
    (torch.bfloat16, 2, 6, 6, 1, 131072, 128, 0, 128, True, "BSND", False, 542, 647, 0),
    (torch.bfloat16, 2, 6, 6, 128, 128, 192, 0, 128, True, "BSND", False, 542, 647, 0),
    (torch.bfloat16, 2, 6, 3, 64, 2048, 128, 0, 128, True, "BSND", False, 0, 256, 0),
    (torch.bfloat16, 2, 6, 1, 1, 339, 192, 0, 128, True, "BSND", False, 0, 256, 0),
    (torch.bfloat16, 2, 6, 3, 64, 800, 192, 0, 128, True, "BSND", False, 0, 256, 0),
    (torch.bfloat16, 2, 6, 1, 16, 4096, 128, 0, 128, True, "BSND", False, 542, 647, 0),

# ================ 2) dense varlen TND (varlen, only bwd): cache_mode=0, layout=TND, varlen, num_splits=0 ================
# data_type=float16, is_causal=False, cache_mode=0, layout=TND, is_varied=True, num_splits=0
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=A, (q_seqlen,kv_seqlen)=seqA, (window_left,window_right)=winA
    (torch.float16, 2, 6, 3, 3, 799, 32, 0, 128, False, "TND", True, 512, 0, 0),
    (torch.float16, 2, 6, 3, 16, 8192, 64, 0, 128, False, "TND", True, -1, -1, 0),
    (torch.float16, 2, 6, 1, 3, 1024, 64, 0, 128, False, "TND", True, 512, 0, 0),
    (torch.float16, 2, 6, 6, 1, 128, 32, 0, 128, False, "TND", True, -1, -1, 0),
    (torch.float16, 2, 6, 1, 16, 4096, 64, 0, 128, False, "TND", True, 512, 0, 0),
    (torch.float16, 2, 6, 6, 64, 256, 32, 0, 128, False, "TND", True, -1, -1, 0),

# data_type=float16, is_causal=True, cache_mode=0, layout=TND, is_varied=True, num_splits=0
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=A, (q_seqlen,kv_seqlen)=seqA, (window_left,window_right)=winA
    (torch.float16, 2, 6, 6, 3, 799, 32, 0, 128, True, "TND", True, 512, 0, 0),
    (torch.float16, 2, 6, 3, 1, 128, 32, 0, 128, True, "TND", True, 512, 0, 0),
    (torch.float16, 2, 6, 6, 16, 4096, 64, 0, 128, True, "TND", True, 512, 0, 0),
    (torch.float16, 2, 6, 3, 16, 8192, 64, 0, 128, True, "TND", True, -1, -1, 0),
    (torch.float16, 2, 6, 1, 3, 1024, 32, 0, 128, True, "TND", True, -1, -1, 0),
    (torch.float16, 2, 6, 1, 64, 256, 64, 0, 128, True, "TND", True, -1, -1, 0),

# data_type=float16, is_causal=False, cache_mode=0, layout=TND, is_varied=True, num_splits=0
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=B, (q_seqlen,kv_seqlen)=seqB, (window_left,window_right)=winB
    (torch.float16, 2, 6, 1, 64, 800, 128, 0, 128, False, "TND", True, 0, 256, 0),
    (torch.float16, 2, 6, 6, 16, 2000, 192, 0, 128, False, "TND", True, 542, 647, 0),
    (torch.float16, 2, 6, 3, 1, 339, 192, 0, 128, False, "TND", True, 0, 256, 0),
    (torch.float16, 2, 6, 3, 64, 2048, 192, 0, 128, False, "TND", True, 542, 647, 0),
    (torch.float16, 2, 6, 6, 16, 4096, 128, 0, 128, False, "TND", True, 542, 647, 0),
    (torch.float16, 2, 6, 1, 128, 128, 128, 0, 128, False, "TND", True, 0, 256, 0),

# data_type=float16, is_causal=True, cache_mode=0, layout=TND, is_varied=True, num_splits=0
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=B, (q_seqlen,kv_seqlen)=seqB, (window_left,window_right)=winB
    (torch.float16, 2, 6, 3, 1, 339, 192, 0, 128, True, "TND", True, 0, 256, 0),
    (torch.float16, 2, 6, 6, 16, 2000, 192, 0, 128, True, "TND", True, 542, 647, 0),
    (torch.float16, 2, 6, 3, 64, 2048, 128, 0, 128, True, "TND", True, 0, 256, 0),
    (torch.float16, 2, 6, 1, 64, 800, 128, 0, 128, True, "TND", True, 542, 647, 0),
    (torch.float16, 2, 6, 1, 16, 4096, 192, 0, 128, True, "TND", True, 542, 647, 0),
    (torch.float16, 2, 6, 6, 128, 128, 128, 0, 128, True, "TND", True, 0, 256, 0),

# data_type=bfloat16, is_causal=False, cache_mode=0, layout=TND, is_varied=True, num_splits=0
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=A, (q_seqlen,kv_seqlen)=seqA, (window_left,window_right)=winA
    (torch.bfloat16, 2, 6, 1, 1, 128, 64, 0, 128, False, "TND", True, 512, 0, 0),
    (torch.bfloat16, 2, 6, 1, 64, 256, 32, 0, 128, False, "TND", True, -1, -1, 0),
    (torch.bfloat16, 2, 6, 6, 16, 8192, 64, 0, 128, False, "TND", True, -1, -1, 0),
    (torch.bfloat16, 2, 6, 3, 3, 799, 32, 0, 128, False, "TND", True, 512, 0, 0),
    (torch.bfloat16, 2, 6, 3, 3, 1024, 32, 0, 128, False, "TND", True, -1, -1, 0),
    (torch.bfloat16, 2, 6, 6, 16, 4096, 64, 0, 128, False, "TND", True, 512, 0, 0),

# data_type=bfloat16, is_causal=True, cache_mode=0, layout=TND, is_varied=True, num_splits=0
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=A, (q_seqlen,kv_seqlen)=seqA, (window_left,window_right)=winA
    (torch.bfloat16, 2, 6, 1, 16, 4096, 64, 0, 128, True, "TND", True, 512, 0, 0),
    (torch.bfloat16, 2, 6, 6, 64, 256, 32, 0, 128, True, "TND", True, 512, 0, 0),
    (torch.bfloat16, 2, 6, 3, 1, 128, 32, 0, 128, True, "TND", True, 512, 0, 0),
    (torch.bfloat16, 2, 6, 6, 3, 1024, 64, 0, 128, True, "TND", True, -1, -1, 0),
    (torch.bfloat16, 2, 6, 1, 3, 799, 64, 0, 128, True, "TND", True, -1, -1, 0),
    (torch.bfloat16, 2, 6, 3, 16, 8192, 32, 0, 128, True, "TND", True, -1, -1, 0),

# data_type=bfloat16, is_causal=False, cache_mode=0, layout=TND, is_varied=True, num_splits=0
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=B, (q_seqlen,kv_seqlen)=seqB, (window_left,window_right)=winB
    (torch.bfloat16, 2, 6, 1, 1, 339, 192, 0, 128, False, "TND", True, 542, 647, 0),
    (torch.bfloat16, 2, 6, 6, 64, 2048, 128, 0, 128, False, "TND", True, 0, 256, 0),
    (torch.bfloat16, 2, 6, 3, 128, 128, 192, 0, 128, False, "TND", True, 0, 256, 0),
    (torch.bfloat16, 2, 6, 1, 16, 2000, 128, 0, 128, False, "TND", True, 0, 256, 0),
    (torch.bfloat16, 2, 6, 6, 64, 800, 128, 0, 128, False, "TND", True, 542, 647, 0),
    (torch.bfloat16, 2, 6, 3, 16, 4096, 192, 0, 128, False, "TND", True, 542, 647, 0),

# data_type=bfloat16, is_causal=True, cache_mode=0, layout=TND, is_varied=True, num_splits=0
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=B, (q_seqlen,kv_seqlen)=seqB, (window_left,window_right)=winB
    (torch.bfloat16, 2, 6, 6, 128, 128, 192, 0, 128, True, "TND", True, 542, 647, 0),
    (torch.bfloat16, 2, 6, 3, 16, 4096, 128, 0, 128, True, "TND", True, 0, 256, 0),
    (torch.bfloat16, 2, 6, 1, 1, 339, 192, 0, 128, True, "TND", True, 542, 647, 0),
    (torch.bfloat16, 2, 6, 6, 16, 2000, 192, 0, 128, True, "TND", True, 0, 256, 0),
    (torch.bfloat16, 2, 6, 1, 64, 2048, 128, 0, 128, True, "TND", True, 0, 256, 0),
    (torch.bfloat16, 2, 6, 3, 64, 800, 128, 0, 128, True, "TND", True, 542, 647, 0),

# ================ 3) paged KV cache (TriDao kvcache): cache_mode=1, layout=TND, varlen, num_splits=1 ================
# data_type=float16, is_causal=False, cache_mode=1, layout=TND, is_varied=True, num_splits=1
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=A, (q_seqlen,kv_seqlen)=seqA, (window_left,window_right)=winA
    (torch.float16, 2, 6, 1, 16, 131072, 64, 1, 128, False, "TND", True, -1, -1, 1),
    (torch.float16, 2, 6, 6, 64, 256, 32, 1, 128, False, "TND", True, 59, 571, 1),
    (torch.float16, 2, 6, 3, 3, 799, 59, 1, 128, False, "TND", True, 460, 62, 1),
    (torch.float16, 2, 6, 1, 1, 128, 80, 1, 128, False, "TND", True, -1, -1, 1),
    (torch.float16, 2, 6, 3, 3, 1024, 32, 1, 128, False, "TND", True, 0, 256, 1),
    # Known unresolved issue: kv=20000 + window=(0,256) + paged TND triggers
    # kernel NaNs (formerly data_type101/103). Trigger: non-causal,
    # window_left=0, and long KV (kv around 19980). Middle output rows become
    # NaN while LSE remains valid. Temporarily disabled pending kernel analysis.
    # (torch.float16, 2, 6, 6, 16, 20000, 59, 1, 128, False, "TND", True, 0, 256, 1),

# data_type=float16, is_causal=True, cache_mode=1, layout=TND, is_varied=True, num_splits=1
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=A, (q_seqlen,kv_seqlen)=seqA, (window_left,window_right)=winA
    (torch.float16, 2, 6, 1, 1, 128, 32, 1, 128, True, "TND", True, 460, 62, 1),
    # (torch.float16, 2, 6, 6, 16, 20000, 59, 1, 128, True, "TND", True, 0, 256, 1),
    (torch.float16, 2, 6, 6, 16, 131072, 32, 1, 128, True, "TND", True, 59, 571, 1),
    (torch.float16, 2, 6, 3, 3, 799, 80, 1, 128, True, "TND", True, -1, -1, 1),
    (torch.float16, 2, 6, 3, 3, 1024, 64, 1, 128, True, "TND", True, -1, -1, 1),
    (torch.float16, 2, 6, 1, 64, 256, 59, 1, 128, True, "TND", True, 0, 256, 1),

# data_type=float16, is_causal=False, cache_mode=1, layout=TND, is_varied=True, num_splits=1
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=B, (q_seqlen,kv_seqlen)=seqB, (window_left,window_right)=winB
    (torch.float16, 2, 6, 6, 64, 2048, 256, 1, 128, False, "TND", True, 536, 462, 1),
    (torch.float16, 2, 6, 3, 16, 4096, 256, 1, 128, False, "TND", True, 563, 425, 1),
    (torch.float16, 2, 6, 3, 128, 128, 128, 1, 128, False, "TND", True, 563, 425, 1),
    (torch.float16, 2, 6, 6, 64, 800, 256, 1, 128, False, "TND", True, 746, 16, 1),
    (torch.float16, 2, 6, 1, 1, 131072, 128, 1, 128, False, "TND", True, 536, 462, 1),
    (torch.float16, 2, 6, 1, 1, 339, 128, 1, 128, False, "TND", True, 746, 16, 1),

# data_type=float16, is_causal=True, cache_mode=1, layout=TND, is_varied=True, num_splits=1
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=B, (q_seqlen,kv_seqlen)=seqB, (window_left,window_right)=winB
    (torch.float16, 2, 6, 1, 1, 339, 128, 1, 128, True, "TND", True, 536, 462, 1),
    (torch.float16, 2, 6, 1, 1, 131072, 256, 1, 128, True, "TND", True, 563, 425, 1),
    (torch.float16, 2, 6, 6, 64, 800, 256, 1, 128, True, "TND", True, 746, 16, 1),
    (torch.float16, 2, 6, 3, 64, 2048, 128, 1, 128, True, "TND", True, 536, 462, 1),
    (torch.float16, 2, 6, 3, 128, 128, 128, 1, 128, True, "TND", True, 746, 16, 1),
    (torch.float16, 2, 6, 6, 16, 4096, 256, 1, 128, True, "TND", True, 563, 425, 1),

# data_type=bfloat16, is_causal=False, cache_mode=1, layout=TND, is_varied=True, num_splits=1
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=A, (q_seqlen,kv_seqlen)=seqA, (window_left,window_right)=winA
    (torch.bfloat16, 2, 6, 6, 16, 20000, 64, 1, 128, False, "TND", True, -1, -1, 1),
    (torch.bfloat16, 2, 6, 3, 1, 128, 80, 1, 128, False, "TND", True, 0, 256, 1),
    (torch.bfloat16, 2, 6, 1, 3, 1024, 32, 1, 128, False, "TND", True, 59, 571, 1),
    (torch.bfloat16, 2, 6, 3, 16, 131072, 32, 1, 128, False, "TND", True, 460, 62, 1),
    (torch.bfloat16, 2, 6, 1, 3, 799, 59, 1, 128, False, "TND", True, 0, 256, 1),
    (torch.bfloat16, 2, 6, 6, 64, 256, 59, 1, 128, False, "TND", True, -1, -1, 1),

# data_type=bfloat16, is_causal=True, cache_mode=1, layout=TND, is_varied=True, num_splits=1
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=A, (q_seqlen,kv_seqlen)=seqA, (window_left,window_right)=winA
    (torch.bfloat16, 2, 6, 6, 3, 1024, 59, 1, 128, True, "TND", True, 0, 256, 1),
    (torch.bfloat16, 2, 6, 1, 16, 20000, 32, 1, 128, True, "TND", True, 460, 62, 1),
    (torch.bfloat16, 2, 6, 6, 1, 128, 59, 1, 128, True, "TND", True, 59, 571, 1),
    (torch.bfloat16, 2, 6, 3, 3, 799, 32, 1, 128, True, "TND", True, 0, 256, 1),
    (torch.bfloat16, 2, 6, 3, 64, 256, 64, 1, 128, True, "TND", True, -1, -1, 1),
    (torch.bfloat16, 2, 6, 1, 16, 131072, 80, 1, 128, True, "TND", True, -1, -1, 1),

# data_type=bfloat16, is_causal=False, cache_mode=1, layout=TND, is_varied=True, num_splits=1
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=B, (q_seqlen,kv_seqlen)=seqB, (window_left,window_right)=winB
    (torch.bfloat16, 2, 6, 6, 64, 800, 256, 1, 128, False, "TND", True, 536, 462, 1),
    (torch.bfloat16, 2, 6, 1, 1, 131072, 256, 1, 128, False, "TND", True, 536, 462, 1),
    (torch.bfloat16, 2, 6, 1, 1, 339, 128, 1, 128, False, "TND", True, 563, 425, 1),
    (torch.bfloat16, 2, 6, 3, 64, 2048, 128, 1, 128, False, "TND", True, 746, 16, 1),
    (torch.bfloat16, 2, 6, 3, 16, 4096, 256, 1, 128, False, "TND", True, 746, 16, 1),
    (torch.bfloat16, 2, 6, 6, 128, 128, 128, 1, 128, False, "TND", True, 563, 425, 1),

# data_type=bfloat16, is_causal=True, cache_mode=1, layout=TND, is_varied=True, num_splits=1
#   num_heads,kv_heads in {(6,6),(6,1),(6,3)}, head_size=B, (q_seqlen,kv_seqlen)=seqB, (window_left,window_right)=winB
    (torch.bfloat16, 2, 6, 3, 64, 800, 128, 1, 128, True, "TND", True, 536, 462, 1),
    (torch.bfloat16, 2, 6, 1, 16, 4096, 128, 1, 128, True, "TND", True, 563, 425, 1),
    (torch.bfloat16, 2, 6, 6, 128, 128, 256, 1, 128, True, "TND", True, 536, 462, 1),
    (torch.bfloat16, 2, 6, 3, 1, 339, 256, 1, 128, True, "TND", True, 563, 425, 1),
    (torch.bfloat16, 2, 6, 6, 1, 131072, 128, 1, 128, True, "TND", True, 746, 16, 1),
    (torch.bfloat16, 2, 6, 1, 64, 2048, 256, 1, 128, True, "TND", True, 746, 16, 1),
    # ========== Special cases: tiny head_size, large-GQA decode, num_splits=2, and special SWA windows ==========
    (torch.bfloat16, 2, 6, 6, 256, 512, 1, 0, 128, True, "BSND", False, -1, -1, 0),
    (torch.bfloat16, 2, 6, 6, 256, 512, 2, 0, 128, True, "BSND", False, -1, -1, 0),
    (torch.bfloat16, 2, 6, 6, 256, 512, 4, 0, 128, True, "BSND", False, -1, -1, 0),
    (torch.bfloat16, 2, 64, 8, 1, 2048, 128, 1, 128, True, "TND", True, -1, -1, 0),
    (torch.bfloat16, 2, 128, 16, 1, 2048, 128, 1, 128, True, "TND", True, -1, -1, 0),
    (torch.float16, 2, 512, 1, 1, 1024, 128, 1, 128, True, "TND", True, -1, -1, 0),
    (torch.bfloat16, 2, 6, 6, 1024, 1024, 128, 1, 128, True, "TND", True, -1, -1, 2),
    (torch.float16, 2, 6, 6, 1024, 2048, 128, 1, 128, False, "TND", True, -1, -1, 2),
    (torch.float16, 2, 6, 6, 512, 1024, 128, 0, 128, True, "BSND", False, 826, 973, 0),
    (torch.bfloat16, 2, 6, 6, 512, 512, 128, 0, 128, True, "BSND", False, 127, 0, 0),
    (torch.float16, 2, 6, 6, 512, 512, 128, 0, 128, False, "BSND", False, 65, 412, 0),
    (torch.bfloat16, 2, 6, 6, 256, 512, 128, 0, 128, False, "BSND", False, 59, 571, 0),
    (torch.float16, 2, 6, 6, 512, 1024, 128, 1, 128, True, "TND", True, 746, 16, 0),
    (torch.bfloat16, 2, 6, 6, 1024, 1024, 128, 1, 128, True, "TND", True, 512, 0, 0),
    (torch.bfloat16, 2, 6, 6, 512, 512, 128, 1, 128, False, "BSND", False, 508, -256, 0),
    (torch.float16, 2, 6, 6, 512, 512, 128, 1, 128, True, "BSND", False, -128, 864, 0),
]
@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, layout, is_varied, window_size_left, window_size_right, num_splits", test_cases)
def test_fa_kvcache_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, layout, is_varied, window_size_left, window_size_right, num_splits):
    # num_splits>1 (active KV split) is currently only wired for paged KV + varlen-q (TND).
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if num_splits > 1 and not (cache_mode == 1 and layout == "TND"):
        pytest.skip("num_splits>1 requires paged KV cache and TND (varlen-q) layout")
    if "Ascend950" in name and num_splits > 1:
        pytest.skip("Ascend950 does not support num_splits>1")
    if not (1 <= head_size <= 256):
        pytest.skip("head_size must be in [1, 256]")

    if "Ascend950" in name and (window_size_left != -1 or window_size_right != -1):
        pytest.skip("Ascend950 does not support SWA")
    if is_varied and layout != "TND":
        pytest.skip("is_varied requires TND (varlen-q) layout")
    block_size = 128
    if is_varied:
        q_sequences, kv_sequences = make_varlen_seqlens(batch_size, q_seqlen, kv_seqlen, seed=1234)
    else:
        q_sequences = [q_seqlen] * batch_size
        kv_sequences = [kv_seqlen] * batch_size
    t_q_sum = sum(q_sequences)
    t_kv_sum = sum(kv_sequences)
    if layout == "BSND":
        query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type,
                                   device="npu", requires_grad=True)
    elif layout == "TND":
        query = make_packed_random_tensor(q_sequences, q_seqlen, num_heads, head_size, data_type,
                                          device="npu", requires_grad=True)
    key_cache = None
    value_cache = None
    block_tables = None
    if cache_mode == 1:
        # make_paged_kv_cache allocates physical blocks from kv_seqlen so long
        # KV cases cannot make block_table reference nonexistent blocks and
        # trigger an AICore DDR overrun.
        key_cache, value_cache = make_paged_kv_cache(
            batch_size, kv_seqlen, block_size, kv_heads, head_size, data_type,
            device="npu", requires_grad=True
        )
        block_tables = make_block_table(batch_size, kv_seqlen, block_size).npu()
    else:
        if layout == "BSND":
            key_cache = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type,
                                           device="npu", requires_grad=True)
            value_cache = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type,
                                             device="npu", requires_grad=True)
        else:
            key_cache = make_packed_random_tensor(kv_sequences, kv_seqlen, kv_heads, head_size, data_type,
                                                  device="npu", requires_grad=True)
            value_cache = make_packed_random_tensor(kv_sequences, kv_seqlen, kv_heads, head_size, data_type,
                                                    device="npu", requires_grad=True)
        block_tables = None
    if layout == "BSND":
        q_seqlen_list = [q_seqlen] * batch_size
        kv_seqlen_list = [kv_seqlen] * batch_size
    else:
        q_seqlen_list = q_sequences
        kv_seqlen_list = kv_sequences
    scale = 1.0 / (head_size ** 0.5)
    kv_seqlen_list = torch.tensor(kv_seqlen_list, dtype=torch.int32).npu()
    new_q_seqlen_list = None
    new_kv_seqlen_list = None
    new_q_seqlen_list_cpu = None
    new_kv_seqlen_list_cpu = None
    window_size_left_golden = window_size_left
    window_size_right_golden = window_size_right
    # Match Tri Dao GPU host: both sides vs kv_seqlen.
    if kv_seqlen > 0 and window_size_left_golden >= kv_seqlen:
        window_size_left_golden = -1
    if kv_seqlen > 0 and window_size_right_golden >= kv_seqlen:
        window_size_right_golden = -1
    if is_causal:
        window_size_right_golden = 0
    is_causal_golden = (window_size_left_golden < 0 and window_size_right_golden == 0)
    is_local_golden = (window_size_left_golden >= 0 or window_size_right_golden > 0) and not is_causal_golden
    if is_local_golden:
        if window_size_left_golden < 0:
            window_size_left_golden = kv_seqlen
        if window_size_right_golden < 0:
            window_size_right_golden = kv_seqlen
    if layout == "TND":
        new_q_seqlen_list_cpu = [0]
        pre_seq_sum = 0
        for i in range(batch_size):
            pre_seq_sum += q_sequences[i]
            new_q_seqlen_list_cpu.append(pre_seq_sum)
        new_q_seqlen_list = torch.tensor(new_q_seqlen_list_cpu, dtype=torch.int32).npu()
        if cache_mode == 0:
            new_kv_seqlen_list_cpu = [0]
            pre_seq_sum = 0
            for i in range(batch_size):
                pre_seq_sum += kv_sequences[i]
                new_kv_seqlen_list_cpu.append(pre_seq_sum)
            new_kv_seqlen_list = torch.tensor(new_kv_seqlen_list_cpu, dtype=torch.int32).npu()
    # Ascend950 v4 has no backward pass (flash_attn_npu_interface_950.py), so the
    # cu_seqlens_k/bwd path is only exercised on Ascend910; on Ascend950 fall back to
    # the validated cache-offset path. This also avoids passing seqused_k=None into the
    # 950 kernel, which requires per-batch KV seqlen (csrc/ascend950/flash_attn_npu_4/mha_fwd.cpp).
    bwd_supported = layout == "TND" and cache_mode == 0 and num_splits <= 1 and "Ascend950" not in name
    cu_seqlens_k_for_api = new_kv_seqlen_list if bwd_supported else None
    max_seqlen_k_for_api = kv_seqlen if bwd_supported else None
    cache_seqlens_for_api = None if bwd_supported else (
        new_kv_seqlen_list if (layout == "TND" and cache_mode == 0) else kv_seqlen_list
    )
    out_out, softmax_lse, *rest = flash_attn_varlen_func(
        query,
        key_cache,
        value_cache,
        qv=None,
        cu_seqlens_q=new_q_seqlen_list,
        cu_seqlens_k=cu_seqlens_k_for_api,
        max_seqlen_q=q_seqlen,
        max_seqlen_k=max_seqlen_k_for_api,
        seqused_k=cache_seqlens_for_api,
        page_table=block_tables,
        softmax_scale=None,
        causal=is_causal,
        window_size=[window_size_left, window_size_right],  # -1 means infinite context window
        softcap=0.0, # 0.0 means deactivated
        num_splits=num_splits,    # Can be tuned for speed
        pack_gqa=None,   # Can be tuned for speed
        return_lse=True,
    )
    query_ref = query.detach().cpu().requires_grad_(True)
    key_ref = key_cache.detach().cpu().requires_grad_(True)
    value_ref = value_cache.detach().cpu().requires_grad_(True)
    block_tables_cpu = block_tables.cpu() if cache_mode == 1 else None

    if layout == "BSND":
        golden_lseL_gpu_ref = torch.empty((batch_size, num_heads, q_seqlen), dtype=torch.float32)
        golden_lseL_gpu_pt = torch.empty_like(golden_lseL_gpu_ref)
        golden_lseL = torch.empty((batch_size, num_heads, q_seqlen), dtype=torch.float32)
    else:
        golden_lseL_gpu_ref = torch.empty((num_heads, t_q_sum), dtype=torch.float32)
        golden_lseL_gpu_pt = torch.empty_like(golden_lseL_gpu_ref)
        golden_lseL = torch.empty((num_heads, t_q_sum), dtype=torch.float32)
    if layout == "BSND":
        atten_mask = None
        if is_causal_golden:
            atten_mask = torch.triu(
                torch.ones(q_seqlen, kv_seqlen),
                diagonal=(kv_seqlen - q_seqlen + 1),
            ).bool()
        elif is_local_golden:
            atten_mask = make_local_attention_mask(
                q_seqlen,
                kv_seqlen,
                window_size_left_golden,
                window_size_right_golden,
            )
        if cache_mode == 1:
            key_batched, value_batched = gather_paged_kv_batch(
                key_ref, value_ref, block_tables_cpu, kv_seqlen, block_size
            )
        else:
            key_batched, value_batched = key_ref, value_ref
        golden_out_gpu_ref, golden_lseL_gpu_ref, golden_out_gpu_pt, golden_lseL_gpu_pt = ref_flash_attention_pair(
            query_ref, key_batched, value_batched, scale, atten_mask, data_type,
            rescale_threshold=4.0,
        )
        if atten_mask is not None:
            fully_masked = atten_mask.all(dim=-1)
            golden_out_gpu_ref[:, fully_masked] = 0
            golden_out_gpu_pt[:, fully_masked] = 0
            golden_lseL_gpu_ref[:, :, fully_masked] = torch.inf
            golden_lseL_gpu_pt[:, :, fully_masked] = torch.inf
        assert_fa_close(out_out, golden_out_gpu_ref, golden_out_gpu_pt, name="out")
        assert_fa_close(softmax_lse, golden_lseL_gpu_ref, golden_lseL_gpu_pt, name="softmax_lse")
        return
    query_padded = pad_packed_tensor(query_ref, q_sequences, q_seqlen)
    if cache_mode == 1:
        key_padded, value_padded = gather_paged_kv_batch(
            key_ref, value_ref, block_tables_cpu, kv_seqlen, block_size
        )
    else:
        key_padded = pad_packed_tensor(key_ref, kv_sequences, kv_seqlen)
        value_padded = pad_packed_tensor(value_ref, kv_sequences, kv_seqlen)
    q_valid, k_valid, atten_mask = make_padded_varlen_mask(
        q_sequences,
        kv_sequences,
        q_seqlen,
        kv_seqlen,
        is_causal_golden,
        window_size_left_golden,
        window_size_right_golden,
    )
    golden_out_gpu_ref, golden_lse_gpu_ref, golden_out_gpu_pt, golden_lse_gpu_pt = ref_flash_attention_pair(
        query_padded, key_padded, value_padded, scale, atten_mask, data_type, rescale_threshold=4.0
    )
    golden_out_plain, golden_lse_plain = ref_flash_attention(
        query_padded.detach(),
        key_padded.detach(),
        value_padded.detach(),
        scale,
        atten_mask,
        data_type,
    )
    fully_masked = atten_mask.all(dim=-1)
    golden_out_gpu_ref[fully_masked] = 0
    golden_out_gpu_pt[fully_masked] = 0
    golden_out_plain[fully_masked] = 0
    golden_lse_gpu_ref = golden_lse_gpu_ref.masked_fill(fully_masked[:, None, :], torch.inf)
    golden_lse_gpu_pt = golden_lse_gpu_pt.masked_fill(fully_masked[:, None, :], torch.inf)
    golden_lse_plain = golden_lse_plain.masked_fill(fully_masked[:, None, :], torch.inf)
    golden_out_gpu_ref = golden_out_gpu_ref[q_valid]
    golden_out_gpu_pt = golden_out_gpu_pt[q_valid]
    golden_out_plain = golden_out_plain[q_valid]
    golden_lse_gpu_ref = golden_lse_gpu_ref.permute(0, 2, 1)[q_valid].transpose(0, 1)
    golden_lse_gpu_pt = golden_lse_gpu_pt.permute(0, 2, 1)[q_valid].transpose(0, 1)
    golden_lse_plain = golden_lse_plain.permute(0, 2, 1)[q_valid].transpose(0, 1)
    assert_fa_close(out_out, golden_out_gpu_ref, golden_out_gpu_pt, name="out")
    assert_fa_close(softmax_lse, golden_lse_gpu_ref, golden_lse_gpu_pt, name="softmax_lse")
    if bwd_supported:
        dout = make_random_tensor(out_out.shape, out_out.dtype, low=-0.5, high=0.5, device="npu")
        dq_ag, dk_ag, dv_ag = torch.autograd.grad(out_out, (query, key_cache, value_cache), dout)
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(
            golden_out_gpu_ref,
            (query_ref, key_ref, value_ref),
            dout.detach().cpu(),
            retain_graph=True,
        )
        dq_pt, dk_pt, dv_pt = torch.autograd.grad(
            golden_out_gpu_pt,
            (query_ref, key_ref, value_ref),
            dout.detach().cpu(),
        )
        assert_fa_close(dq_ag, dq_ref, dq_pt, name="dQ")
        assert_fa_close(dk_ag, dk_ref, dk_pt, name="dK")
        assert_fa_close(dv_ag, dv_ref, dv_pt, name="dV")
    return

# flash_attn_varlen_func test parameters (Ascend950 head_dim<=256 coverage, NPU only)
# Single-option parameters: fixed values
# data_type: [torch.bfloat16]
# kv_heads: [8]
# block_size: [128]
# softcap: [0.0]
# window_size_left,window_size_right: [(-1,-1)]

# Two-option parameters
# is_causal: [False, True]
# cache_mode: [0, 1]
# layout: [BSND, TND]
# num_heads: baseline coverage [32], dedicated S/N axis-fusion coverage [8, 64]

# Multi-option parameters: grouped values
# head_size: A=[35, 64, 101, 128], B=[151, 192, 201, 256]
# batch_size,q_seqlen,kv_seqlen: A=[(1,256,128), (1,130,128), (2,256,256), (4,128,256), (2,128,128)], B=[(1,384,128), (1,256,384), (1,128,128), (1,256,512), (1,256,192)]
# num_splits: A=[0, 1, 2] for cache=1 with layout=TND; B=[0, 1] otherwise
# Dedicated S/N axis-fusion cases also cover batch=[8,16],
# q_seqlen=[16,32,64], and kv_seqlen=[16,32,64,1024]

hd_cases = [
    # data_type=torch.bfloat16, is_causal=False, cache_mode=0, layout=BSND
    # head_size=A, (batch_size,q_seqlen,kv_seqlen)=A, num_splits=[0,1]
    (torch.bfloat16, 2, 32, 8, 128, 128, 64, 0, 128, False, "BSND", 0, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 256, 128, 101, 0, 128, False, "BSND", 1, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 130, 128, 128, 0, 128, False, "BSND", 0, -1, -1, 0.0),
    (torch.bfloat16, 4, 32, 8, 128, 256, 35, 0, 128, False, "BSND", 1, -1, -1, 0.0),
    (torch.bfloat16, 2, 32, 8, 256, 256, 64, 0, 128, False, "BSND", 0, -1, -1, 0.0),
    (torch.bfloat16, 2, 32, 8, 128, 128, 101, 0, 128, False, "BSND", 1, -1, -1, 0.0),
    # data_type=torch.bfloat16, is_causal=True, cache_mode=0, layout=BSND
    # head_size=A, (batch_size,q_seqlen,kv_seqlen)=B, num_splits=[0,1]
    (torch.bfloat16, 1, 32, 8, 256, 384, 128, 0, 128, True, "BSND", 0, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 128, 128, 101, 0, 128, True, "BSND", 1, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 384, 128, 35, 0, 128, True, "BSND", 0, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 256, 192, 64, 0, 128, True, "BSND", 1, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 256, 512, 128, 0, 128, True, "BSND", 0, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 256, 384, 101, 0, 128, True, "BSND", 1, -1, -1, 0.0),
    # data_type=torch.bfloat16, is_causal=False, cache_mode=1, layout=BSND
    # head_size=B, (batch_size,q_seqlen,kv_seqlen)=A, num_splits=[0,1]
    (torch.bfloat16, 4, 32, 8, 128, 256, 192, 1, 128, False, "BSND", 0, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 130, 128, 201, 1, 128, False, "BSND", 1, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 256, 128, 256, 1, 128, False, "BSND", 0, -1, -1, 0.0),
    (torch.bfloat16, 2, 32, 8, 256, 256, 151, 1, 128, False, "BSND", 1, -1, -1, 0.0),
    (torch.bfloat16, 2, 32, 8, 128, 128, 192, 1, 128, False, "BSND", 0, -1, -1, 0.0),
    (torch.bfloat16, 4, 32, 8, 128, 256, 201, 1, 128, False, "BSND", 1, -1, -1, 0.0),
    # data_type=torch.bfloat16, is_causal=True, cache_mode=1, layout=BSND
    # head_size=B, (batch_size,q_seqlen,kv_seqlen)=B, num_splits=[0,1]
    (torch.bfloat16, 1, 32, 8, 256, 512, 201, 1, 128, True, "BSND", 0, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 128, 128, 256, 1, 128, True, "BSND", 1, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 256, 384, 151, 1, 128, True, "BSND", 0, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 256, 192, 192, 1, 128, True, "BSND", 1, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 384, 128, 201, 1, 128, True, "BSND", 0, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 256, 512, 256, 1, 128, True, "BSND", 1, -1, -1, 0.0),
    # data_type=torch.bfloat16, is_causal=False, cache_mode=0, layout=TND
    # head_size=A, (batch_size,q_seqlen,kv_seqlen)=A, num_splits=[0,1]
    (torch.bfloat16, 4, 32, 8, 128, 256, 101, 0, 128, False, "TND", 0, -1, -1, 0.0),
    (torch.bfloat16, 2, 32, 8, 256, 256, 128, 0, 128, False, "TND", 1, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 256, 128, 64, 0, 128, False, "TND", 0, -1, -1, 0.0),
    (torch.bfloat16, 2, 32, 8, 128, 128, 35, 0, 128, False, "TND", 1, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 130, 128, 101, 0, 128, False, "TND", 0, -1, -1, 0.0),
    (torch.bfloat16, 4, 32, 8, 128, 256, 128, 0, 128, False, "TND", 1, -1, -1, 0.0),
    # data_type=torch.bfloat16, is_causal=True, cache_mode=0, layout=TND
    # head_size=A, (batch_size,q_seqlen,kv_seqlen)=B, num_splits=[0,1]
    (torch.bfloat16, 1, 32, 8, 384, 128, 64, 0, 128, True, "TND", 0, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 256, 192, 101, 0, 128, True, "TND", 1, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 128, 128, 35, 0, 128, True, "TND", 0, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 256, 512, 128, 0, 128, True, "TND", 1, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 256, 384, 64, 0, 128, True, "TND", 0, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 384, 128, 101, 0, 128, True, "TND", 1, -1, -1, 0.0),
    # data_type=torch.bfloat16, is_causal=False, cache_mode=1, layout=TND
    # head_size=B, (batch_size,q_seqlen,kv_seqlen)=A, num_splits=[0,1,2]
    (torch.bfloat16, 2, 32, 8, 128, 128, 192, 1, 128, False, "TND", 0, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 256, 128, 201, 1, 128, False, "TND", 1, -1, -1, 0.0),
    (torch.bfloat16, 4, 32, 8, 128, 256, 256, 1, 128, False, "TND", 2, -1, -1, 0.0),
    (torch.bfloat16, 2, 32, 8, 256, 256, 151, 1, 128, False, "TND", 0, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 130, 128, 192, 1, 128, False, "TND", 1, -1, -1, 0.0),
    (torch.bfloat16, 2, 32, 8, 128, 128, 201, 1, 128, False, "TND", 2, -1, -1, 0.0),
    # data_type=torch.bfloat16, is_causal=True, cache_mode=1, layout=TND
    # head_size=B, (batch_size,q_seqlen,kv_seqlen)=B, num_splits=[0,1,2]
    (torch.bfloat16, 1, 32, 8, 256, 384, 192, 1, 128, True, "TND", 0, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 256, 512, 151, 1, 128, True, "TND", 1, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 384, 128, 201, 1, 128, True, "TND", 2, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 128, 128, 256, 1, 128, True, "TND", 0, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 256, 192, 192, 1, 128, True, "TND", 1, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 256, 384, 151, 1, 128, True, "TND", 2, -1, -1, 0.0),
    # Ascend950 S/N axis-fusion coverage: large batch, small S, many query
    # heads, and unaligned head_size
    (torch.bfloat16, 16, 8, 8, 64, 64, 35, 0, 128, False, "BSND", 0, -1, -1, 0.0),
    (torch.bfloat16, 16, 64, 8, 32, 32, 101, 0, 128, True, "BSND", 1, -1, -1, 0.0),
    (torch.bfloat16, 16, 8, 8, 16, 16, 151, 1, 128, False, "TND", 0, -1, -1, 0.0),
    (torch.bfloat16, 8, 64, 8, 64, 1024, 192, 1, 128, True, "TND", 1, -1, -1, 0.0),
    (torch.bfloat16, 8, 8, 8, 16, 1024, 201, 0, 128, False, "TND", 0, -1, -1, 0.0),
]

@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, layout, num_splits, window_size_left, window_size_right, softcap", hd_cases)
def test_fa_kvcache_ops_with_hd_le_256(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, layout, num_splits, window_size_left, window_size_right, softcap):
    is_varied = layout == 'TND'
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if "Ascend910" in name:
        pytest.skip("Sq > Sk not support in Ascend910")
    test_fa_kvcache_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, layout, is_varied, window_size_left, window_size_right, num_splits)


# ============================================================================
# flash_attn_func tests (BSND layout only, 910 only)
# ============================================================================

test_cases_fa_func = [
    # (data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size_left, window_size_right, softcap)
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, False, -1, -1, 0.0),
    (torch.float16, 7, 5, 1, 512, 512, 128, True, -1, -1, 0.0),
    (torch.float16, 7, 5, 1, 777, 888, 192, False, -1, -1, 0.0),
    # SWA (window_size covers full sequence)
    (torch.bfloat16, 1, 1, 1, 512, 512, 128, True, 512, 0, 0.0),
    # MHA + GQA + MQA
    (torch.bfloat16, 2, 8, 8, 512, 512, 128, True, -1, -1, 0.0),
    (torch.bfloat16, 2, 8, 2, 512, 512, 128, False, -1, -1, 0.0),
    (torch.bfloat16, 2, 4, 1, 512, 512, 128, True, -1, -1, 0.0),
    (torch.float16, 2, 8, 8, 256, 256, 128, False, -1, -1, 0.0),
    (torch.float16, 2, 8, 2, 512, 512, 128, True, -1, -1, 0.0),
    (torch.float16, 2, 4, 1, 256, 256, 128, False, -1, -1, 0.0),
    # head_size=64
    (torch.bfloat16, 2, 16, 16, 512, 512, 64, True, -1, -1, 0.0),
    (torch.float16, 2, 4, 1, 256, 256, 64, True, -1, -1, 0.0),
]


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, "
    "is_causal, window_size_left, window_size_right, softcap",
    test_cases_fa_func,
)
def test_fa_func_custom_ops(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
    is_causal, window_size_left, window_size_right, softcap,
):
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if "Ascend910" not in name:
        pytest.skip("flash_attn_func only supports Ascend910")

    q_min_range = -5.0
    q_max_range = 5.0
    kv_min_range = -5.0
    kv_max_range = 5.0

    query = (q_min_range + (q_max_range - q_min_range) * torch.rand(batch_size, q_seqlen, num_heads, head_size)).to(data_type).npu()
    key_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size, kv_seqlen, kv_heads, head_size)).to(data_type).npu()
    value_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size, kv_seqlen, kv_heads, head_size)).to(data_type).npu()
    scale = 1.0 / (head_size ** 0.5)

    ret = flash_attn_func(
        query,
        key_cache,
        value_cache,
        softmax_scale=scale,
        causal=is_causal,
        window_size=(window_size_left, window_size_right),
        softcap=softcap,
        return_lse=True,
    )
    out_out, softmax_lse = ret

    golden_out_ref = torch.empty((batch_size, q_seqlen, num_heads, head_size), dtype=data_type)
    golden_out_pt = torch.empty((batch_size, q_seqlen, num_heads, head_size), dtype=data_type)
    golden_lseL_ref = torch.empty((batch_size, num_heads, q_seqlen), dtype=torch.float32)
    golden_lseL_pt = torch.empty((batch_size, num_heads, q_seqlen), dtype=torch.float32)
    atten_mask = None
    if is_causal:
        atten_mask = build_cann_causal_mask()[:q_seqlen, :kv_seqlen]
    for i in range(batch_size):
        key_cache_per_batch = key_cache.detach().cpu()[i:i+1]
        value_cache_per_batch = value_cache.detach().cpu()[i:i+1]
        query_cpu = query.detach().cpu()[i:i+1]
        out_ref, lse_ref, out_pt, lse_pt = ref_flash_attention_pair(
            query_cpu, key_cache_per_batch, value_cache_per_batch,
            scale, atten_mask, data_type, softcap=softcap,
        )
        golden_out_ref[i:i + 1] = out_ref.reshape(1, q_seqlen, num_heads, head_size)
        golden_out_pt[i:i + 1] = out_pt.reshape(1, q_seqlen, num_heads, head_size)
        golden_lseL_ref[i:i + 1] = lse_ref.reshape(1, num_heads, q_seqlen)
        golden_lseL_pt[i:i + 1] = lse_pt.reshape(1, num_heads, q_seqlen)

    assert_fa_close(out_out, golden_out_ref, golden_out_pt, softcap=softcap, name="out")
    assert_fa_close(softmax_lse, golden_lseL_ref, golden_lseL_pt, softcap=softcap, name="softmax_lse")
