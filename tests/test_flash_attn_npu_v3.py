# Copyright (c) 2026, Minghua Shen.

import os
import torch
import torch_npu
import pytest
from tests.common.attention_ref import cached_autograd_grads, ref_flash_attention_pair
from tests.common.compare import assert_fa_close
from tests.common.test_utils import (
    gather_paged_kv,
    gather_paged_kv_batch,
    make_attention_inputs,
    make_block_table,
    make_cu_seqlens,
    make_golden_attention_mask,
    make_local_attention_mask,
    make_packed_random_tensor,
    make_paged_kv_cache,
    make_padded_varlen_mask,
    pad_packed_tensor,
    make_random_tensor,
    make_varlen_seqlens,
    check_kvcache_inplace
)
from flash_attn_npu_3 import flash_attn_with_kvcache, flash_attn_func, flash_attn_varlen_func

# flash_attn_with_kvcache test parameters
# Single-option parameters: fixed values
# batch_size: [2]
# block_size: [128]

# Two-option parameters
# data_type: [torch.float16, torch.bfloat16]
# is_causal: [False, True]
# cache_mode: [0, 1]
# layout: [BSND, TND]
# is_varied: [False, True]
# num_splits: [0, 1]

# Multi-option parameters: grouped values
# softcap,num_heads,kv_heads: A=[(0.0,6,6), (0.0,6,1), (0.0,6,3), (2.0,6,6), (2.0,6,1), (2.0,6,3)]
# head_size: A=[32, 64, 128], B=[59, 80, 256]
# q_seqlen,kv_seqlen: A=[(1,128), (64,256), (3,799), (3,1024), (16,20000), (16,131072)], B=[(128,128), (1,339), (64,800), (64,2048), (1,131072)]
# window_size_left,window_size_right: A=[(-1,-1), (512,0)], B=[(0,256), (542,647)]

# Additional coverage: tiny head sizes 1/2/4, large-GQA decode, num_splits=2,
# and special SWA windows
test_cases = [
    # data_type=torch.float16, is_causal=False, cache_mode=0
    # softcap,num_heads,kv_heads=A, head_size=A, (q_seqlen,kv_seqlen)=A, (window_size_left,window_size_right)=A
    (torch.float16, 2, 6, 6, 3, 799, 64, 0, 128, False, "BSND", False, -1, -1, 0.0, 0, False),
    (torch.float16, 2, 6, 6, 16, 131072, 64, 0, 128, False, "BSND", False, 512, 0, 2.0, 0, False),
    (torch.float16, 2, 6, 3, 64, 256, 128, 0, 128, False, "BSND", False, 512, 0, 2.0, 0, False),
    (torch.float16, 2, 6, 1, 3, 1024, 32, 0, 128, False, "BSND", False, 512, 0, 0.0, 0, False),
    (torch.float16, 2, 6, 1, 16, 20000, 32, 0, 128, False, "BSND", False, -1, -1, 2.0, 0, False),
    (torch.float16, 2, 6, 3, 1, 128, 128, 0, 128, False, "BSND", False, -1, -1, 0.0, 0, False),
    # data_type=torch.bfloat16, is_causal=False, cache_mode=0
    # softcap,num_heads,kv_heads=A, head_size=A, (q_seqlen,kv_seqlen)=A, (window_size_left,window_size_right)=B
    (torch.bfloat16, 2, 6, 1, 3, 1024, 32, 0, 128, False, "BSND", False, 542, 647, 0.0, 0, False),
    (torch.bfloat16, 2, 6, 3, 1, 128, 128, 0, 128, False, "BSND", False, 0, 256, 0.0, 0, False),
    (torch.bfloat16, 2, 6, 1, 16, 131072, 32, 0, 128, False, "BSND", False, 0, 256, 2.0, 0, False),
    (torch.bfloat16, 2, 6, 6, 16, 20000, 64, 0, 128, False, "BSND", False, 0, 256, 0.0, 0, False),
    (torch.bfloat16, 2, 6, 3, 64, 256, 128, 0, 128, False, "BSND", False, 542, 647, 2.0, 0, False),
    (torch.bfloat16, 2, 6, 6, 3, 799, 64, 0, 128, False, "BSND", False, 542, 647, 2.0, 0, False),
    # data_type=torch.float16, is_causal=True, cache_mode=0
    # softcap,num_heads,kv_heads=A, head_size=A, (q_seqlen,kv_seqlen)=B, (window_size_left,window_size_right)=A
    (torch.float16, 2, 6, 3, 64, 2048, 64, 0, 128, True, "BSND", False, -1, -1, 0.0, 0, False),
    (torch.float16, 2, 6, 1, 1, 339, 128, 0, 128, True, "BSND", False, 512, 0, 0.0, 0, False),
    (torch.float16, 2, 6, 3, 64, 800, 64, 0, 128, True, "BSND", False, 512, 0, 2.0, 0, False),
    (torch.float16, 2, 6, 6, 64, 800, 32, 0, 128, True, "BSND", False, -1, -1, 0.0, 0, False),
    (torch.float16, 2, 6, 6, 1, 131072, 32, 0, 128, True, "BSND", False, 512, 0, 2.0, 0, False),
    (torch.float16, 2, 6, 1, 128, 128, 128, 0, 128, True, "BSND", False, -1, -1, 2.0, 0, False),
    # data_type=torch.bfloat16, is_causal=True, cache_mode=0
    # softcap,num_heads,kv_heads=A, head_size=A, (q_seqlen,kv_seqlen)=B, (window_size_left,window_size_right)=B
    (torch.bfloat16, 2, 6, 3, 1, 131072, 128, 0, 128, True, "BSND", False, 542, 647, 0.0, 0, False),
    (torch.bfloat16, 2, 6, 1, 1, 339, 64, 0, 128, True, "BSND", False, 542, 647, 2.0, 0, False),
    (torch.bfloat16, 2, 6, 3, 64, 2048, 128, 0, 128, True, "BSND", False, 0, 256, 2.0, 0, False),
    (torch.bfloat16, 2, 6, 1, 128, 128, 64, 0, 128, True, "BSND", False, 0, 256, 0.0, 0, False),
    (torch.bfloat16, 2, 6, 6, 64, 800, 32, 0, 128, True, "BSND", False, 0, 256, 2.0, 0, False),
    (torch.bfloat16, 2, 6, 6, 64, 2048, 32, 0, 128, True, "BSND", False, 542, 647, 0.0, 0, False),
    # data_type=torch.float16, is_causal=False, cache_mode=1
    # softcap,num_heads,kv_heads=A, head_size=B, (q_seqlen,kv_seqlen)=A, (window_size_left,window_size_right)=A
    (torch.float16, 2, 6, 3, 3, 1024, 256, 1, 128, False, "TND", True, 512, 0, 2.0, 1, False),
    (torch.float16, 2, 6, 6, 16, 131072, 80, 1, 128, False, "TND", True, 512, 0, 2.0, 1, False),
    (torch.float16, 2, 6, 1, 3, 799, 59, 1, 128, False, "TND", True, 512, 0, 0.0, 1, False),
    (torch.float16, 2, 6, 6, 1, 128, 80, 1, 128, False, "TND", True, -1, -1, 0.0, 1, False),
    (torch.float16, 2, 6, 3, 64, 256, 256, 1, 128, False, "TND", True, -1, -1, 0.0, 1, False),
    (torch.float16, 2, 6, 1, 16, 20000, 59, 1, 128, False, "TND", True, -1, -1, 2.0, 1, False),
    # data_type=torch.bfloat16, is_causal=False, cache_mode=1
    # softcap,num_heads,kv_heads=A, head_size=B, (q_seqlen,kv_seqlen)=A, (window_size_left,window_size_right)=B
    (torch.bfloat16, 2, 6, 3, 3, 1024, 256, 1, 128, False, "TND", True, 542, 647, 2.0, 1, False),
    (torch.bfloat16, 2, 6, 1, 1, 128, 80, 1, 128, False, "TND", True, 0, 256, 2.0, 1, False),
    (torch.bfloat16, 2, 6, 3, 64, 256, 256, 1, 128, False, "TND", True, 0, 256, 0.0, 1, False),
    (torch.bfloat16, 2, 6, 6, 16, 131072, 59, 1, 128, False, "TND", True, 542, 647, 2.0, 1, False),
    (torch.bfloat16, 2, 6, 6, 3, 799, 59, 1, 128, False, "TND", True, 0, 256, 0.0, 1, False),
    (torch.bfloat16, 2, 6, 1, 16, 20000, 80, 1, 128, False, "TND", True, 542, 647, 0.0, 1, False),
    # data_type=torch.float16, is_causal=True, cache_mode=1
    # softcap,num_heads,kv_heads=A, head_size=B, (q_seqlen,kv_seqlen)=B, (window_size_left,window_size_right)=A
    (torch.float16, 2, 6, 3, 64, 800, 256, 1, 128, True, "TND", True, 512, 0, 0.0, 1, False),
    (torch.float16, 2, 6, 1, 128, 128, 80, 1, 128, True, "TND", True, 512, 0, 2.0, 1, False),
    (torch.float16, 2, 6, 3, 1, 131072, 256, 1, 128, True, "TND", True, -1, -1, 2.0, 1, False),
    (torch.float16, 2, 6, 6, 1, 131072, 59, 1, 128, True, "TND", True, 512, 0, 0.0, 1, False),
    (torch.float16, 2, 6, 6, 64, 2048, 59, 1, 128, True, "TND", True, -1, -1, 2.0, 1, False),
    (torch.float16, 2, 6, 1, 1, 339, 80, 1, 128, True, "TND", True, -1, -1, 0.0, 1, False),
    # data_type=torch.bfloat16, is_causal=True, cache_mode=1
    # softcap,num_heads,kv_heads=A, head_size=B, (q_seqlen,kv_seqlen)=B, (window_size_left,window_size_right)=B
    (torch.bfloat16, 2, 6, 1, 64, 2048, 59, 1, 128, True, "TND", True, 0, 256, 0.0, 1, False),
    (torch.bfloat16, 2, 6, 6, 128, 128, 80, 1, 128, True, "TND", True, 542, 647, 0.0, 1, False),
    (torch.bfloat16, 2, 6, 3, 128, 128, 256, 1, 128, True, "TND", True, 0, 256, 2.0, 1, False),
    (torch.bfloat16, 2, 6, 6, 64, 800, 80, 1, 128, True, "TND", True, 0, 256, 2.0, 1, False),
    (torch.bfloat16, 2, 6, 1, 1, 131072, 59, 1, 128, True, "TND", True, 542, 647, 2.0, 1, False),
    (torch.bfloat16, 2, 6, 3, 1, 339, 256, 1, 128, True, "TND", True, 542, 647, 0.0, 1, False),
    # Tiny head sizes: 1, 2, and 4
    (torch.bfloat16, 2, 6, 6, 256, 512, 1, 0, 128, True, "BSND", False, -1, -1, 0.0, 0, False),
    (torch.bfloat16, 2, 6, 6, 256, 512, 2, 0, 128, True, "BSND", False, -1, -1, 0.0, 0, False),
    (torch.bfloat16, 2, 6, 6, 256, 512, 4, 0, 128, True, "BSND", False, -1, -1, 0.0, 0, False),
    # Large num_heads/GQA decode: (64,8), (128,16), and (512,1)
    (torch.bfloat16, 2, 64, 8, 1, 2048, 128, 1, 128, True, "TND", True, -1, -1, 0.0, 0, False),
    (torch.bfloat16, 2, 128, 16, 1, 2048, 128, 1, 128, True, "TND", True, -1, -1, 0.0, 0, False),
    (torch.float16, 2, 512, 1, 1, 1024, 128, 1, 128, True, "TND", True, -1, -1, 0.0, 0, False),
    # num_splits=2（paged+TND）
    (torch.bfloat16, 2, 6, 6, 1024, 1024, 128, 1, 128, True, "TND", True, -1, -1, 0.0, 2, False),
    (torch.float16, 2, 6, 6, 1024, 2048, 128, 1, 128, False, "TND", True, -1, -1, 0.0, 2, False),
    # Special SWA windows: (826,973), (127,0), (65,412), (59,571), (746,16), and (512,0)
    (torch.float16, 2, 6, 6, 512, 1024, 128, 0, 128, True, "BSND", False, 826, 973, 0.0, 0, False),
    (torch.bfloat16, 2, 6, 6, 512, 512, 128, 0, 128, True, "BSND", False, 127, 0, 0.0, 0, False),
    (torch.float16, 2, 6, 6, 512, 512, 128, 0, 128, False, "BSND", False, 65, 412, 0.0, 0, False),
    (torch.bfloat16, 2, 6, 6, 256, 512, 128, 0, 128, False, "BSND", False, 59, 571, 0.0, 0, False),
    (torch.float16, 2, 6, 6, 512, 1024, 128, 1, 128, True, "TND", True, 746, 16, 0.0, 0, False),
    (torch.bfloat16, 2, 6, 6, 1024, 1024, 128, 1, 128, True, "TND", True, 512, 0, 0.0, 0, False),
    # Additional negative-side windows: (508,-256) and (-128,864)
    (torch.bfloat16, 2, 6, 6, 512, 512, 128, 1, 128, False, "BSND", False, 508, -256, 0.0, 0, False),
    (torch.float16, 2, 6, 6, 512, 512, 128, 1, 128, True, "BSND", False, -128, 864, 0.0, 0, False),
    # AppendKV
    (torch.bfloat16, 1, 32, 4, 1, 2048, 128, 1, 128, False, "BSND", False, -1, -1, 0.0, 0, True),
    (torch.bfloat16, 2, 16, 2, 1, 4096, 128, 1, 128, True, "BSND", False, -1, -1, 0.0, 0, True),
    (torch.bfloat16, 1, 16, 2, 1024, 2048, 128, 1, 128, True, "BSND", False, -1, -1, 0.0, 0, True),
    (torch.bfloat16, 2, 4, 2, 513, 2048, 128, 1, 128, False, "BSND", False, -1, -1, 0.0, 0, True),
    (torch.bfloat16, 1, 16, 2, 128, 2048, 128, 0, 128, False, "BSND", False, -1, -1, 0.0, 0, True),
    (torch.bfloat16, 1, 8, 2, 512, 1024, 128, 0, 128, True, "BSND", False, -1, -1, 0.0, 0, True),
]

@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, layout, is_varied, window_size_left, window_size_right, softcap, num_splits, new_kv", test_cases)
def test_fa_kvcache_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, layout, is_varied, window_size_left, window_size_right, softcap, num_splits, new_kv):
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if num_splits > 1 and not (cache_mode == 1 and layout == "TND"):
        pytest.skip("num_splits>1 requires paged KV cache and TND (varlen-q) layout")
    if "Ascend950" in name and num_splits > 1:
        pytest.skip("Ascend950 does not support num_splits>1")
    if not (1 <= head_size <= 256):
        pytest.skip("head_size must be in [1, 256]")
    if "Ascend950" in name and (softcap != 0.0):
        pytest.skip("Ascend950 support softcap")
    if is_varied and layout != "TND":
        pytest.skip("is_varied requires TND (varlen-q) layout")
    if new_kv:
        if "Ascend950" in name:
            pytest.skip("Ascend950 does not support append-KV")
        if head_size % 16 != 0:
            pytest.skip("append-KV requires head dim % 16 == 0")
        if window_size_left >= 0 or window_size_right >= 0:
            pytest.skip("append-KV does not support SWA")

    block_size = 128
    gen = torch.Generator().manual_seed(1234)
    if is_varied:
        q_sequences, kv_sequences = make_varlen_seqlens(batch_size, q_seqlen, kv_seqlen, seed=1234)
    else:
        q_sequences = [q_seqlen] * batch_size
        kv_sequences = [kv_seqlen] * batch_size
    t_q_sum = sum(q_sequences)
    t_kv_sum = sum(kv_sequences)
    if layout == "BSND":
        query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type, generator=gen, device="npu")
    elif layout == "TND":
        query = make_packed_random_tensor(q_sequences, q_seqlen, num_heads, head_size, data_type, generator=gen, device="npu")
    key_cache = None
    value_cache = None
    block_tables = None
    if cache_mode == 1:
        # make_paged_kv_cache allocates physical blocks from kv_seqlen so long
        # KV cases cannot make block_table reference nonexistent blocks and
        # trigger an AICore DDR overrun.
        key_cache, value_cache = make_paged_kv_cache(
            batch_size, kv_seqlen, block_size, kv_heads, head_size, data_type,
            generator=gen, device="npu"
        )
        block_tables = make_block_table(batch_size, kv_seqlen, block_size).npu()
    else:
        if layout == "BSND":
            key_cache = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type, generator=gen, device="npu")
            value_cache = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type, generator=gen, device="npu")
        else:
            if new_kv:
                # append-KV uses the capacity-aligned per-batch cache layout (same as BSND).
                kv_min_range, kv_max_range = -5.0, 5.0
                key_cache = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type, low=kv_min_range, high=kv_max_range, generator=gen, device="npu")
                value_cache = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type, low=kv_min_range, high=kv_max_range, generator=gen, device="npu")
            else:
                key_cache = make_packed_random_tensor(kv_sequences, kv_seqlen, kv_heads, head_size, data_type, generator=gen, device="npu")
                value_cache = make_packed_random_tensor(kv_sequences, kv_seqlen, kv_heads, head_size, data_type, generator=gen, device="npu")
        block_tables = None
    if layout == "BSND":
        q_seqlen_list = [q_seqlen] * batch_size
        kv_seqlen_list = [kv_seqlen] * batch_size
    else:
        q_seqlen_list = q_sequences
        kv_seqlen_list = kv_sequences
    scale = 1.0 / (head_size ** 0.5)
    is_rotary_interleaved = False
    if new_kv:
        # Append-KV: per-batch old length (causal: old % 512 == 0).
        new_seqlen = min(q_seqlen, max(1, kv_seqlen // 2))
        capacity = ((kv_seqlen + block_size - 1) // block_size) * block_size if cache_mode == 1 else kv_seqlen
        gen = torch.Generator().manual_seed(2026)
        # causal: the kernel mask assumes kv_total = old + new >= q; keep old aligned to 512.
        old_min = ((max(0, q_seqlen - new_seqlen) + 511) // 512) * 512 if is_causal else 0
        if old_min > capacity - new_seqlen:
            pytest.skip("causal append-KV needs capacity for old >= q - new")
        old_lens = (torch.randint(0, (capacity - new_seqlen - old_min) // 512 + 1,
                                  (batch_size,), generator=gen) * 512 + old_min) \
            if is_causal else torch.randint(0, capacity - new_seqlen + 1, (batch_size,), generator=gen)
        cache_seqlens = old_lens.to(torch.int32).npu()
        k_new = torch.randn(batch_size, new_seqlen, kv_heads, head_size, dtype=data_type, generator=gen).npu()
        v_new = torch.randn(batch_size, new_seqlen, kv_heads, head_size, dtype=data_type, generator=gen).npu()
        key_cache_orig = key_cache.detach().clone()
        value_cache_orig = value_cache.detach().clone()
    else:
        cache_seqlens = torch.tensor(kv_seqlen_list, dtype=torch.int32).npu()
        k_new = None
        v_new = None
    rotary_cos = None
    rotary_sin = None
    cache_batch_idx = None
    leftpad_k = None
    alibi_slopes = None
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
    out_out, softmax_lse, *rest = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        k_new,
        v_new,
        None,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=cache_seqlens,
        page_table=block_tables,
        cu_seqlens_q=new_q_seqlen_list,
        cu_seqlens_k_new=None,
        max_seqlen_q=q_seqlen,
        rotary_seqlens=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=None,
        causal=is_causal,
        window_size=[window_size_left, window_size_right],
        attention_chunk=0,
        softcap=softcap,
        rotary_interleaved=is_rotary_interleaved,
        scheduler_metadata=None,
        num_splits=num_splits,
        pack_gqa=None,
        sm_margin=0,
        return_softmax_lse=True
    )

    if new_kv:
        # Append-KV golden: per-batch kv = old + new. Reconstruct the linear KV
        # (cache [0, old_i) + k_new/v_new) and derive the mask against kv_len_i.
        k_new_cpu = k_new.detach().cpu()
        v_new_cpu = v_new.detach().cpu()
        cache_seqlens_cpu = cache_seqlens.detach().cpu()
        query_cpu = query.detach().cpu()
        key_cache_cpu = key_cache.detach().cpu()
        value_cache_cpu = value_cache.detach().cpu()
        block_tables_cpu = block_tables.cpu() if cache_mode == 1 else None
        golden_out_ref = torch.empty(
            (batch_size, q_seqlen, num_heads, head_size) if layout == "BSND"
            else (t_q_sum, num_heads, head_size), dtype=data_type)
        golden_out_pt = torch.empty_like(golden_out_ref)
        golden_lseL_ref = torch.empty(
            (batch_size, num_heads, q_seqlen) if layout == "BSND"
            else (num_heads, t_q_sum), dtype=torch.float32)
        golden_lseL_pt = torch.empty_like(golden_lseL_ref)
        for i in range(batch_size):
            q_seqlen_per_batch = q_sequences[i]
            old_i = int(cache_seqlens_cpu[i])
            kv_len_i = old_i + new_seqlen
            if layout == "BSND":
                query_cpu_per_batch = query_cpu[i : i + 1]
            else:
                query_cpu_per_batch = query_cpu[new_q_seqlen_list_cpu[i] : new_q_seqlen_list_cpu[i + 1]].unsqueeze(0)
            if cache_mode == 1:
                key_per_batch, value_per_batch = gather_paged_kv(
                    key_cache_cpu, value_cache_cpu, block_tables_cpu[i], old_i, block_size)
            elif layout == "BSND":
                key_per_batch = key_cache_cpu[i][:old_i]
                value_per_batch = value_cache_cpu[i][:old_i]
            else:
                key_per_batch = key_cache_cpu[i][:old_i]
                value_per_batch = value_cache_cpu[i][:old_i]
            key_per_batch = torch.cat([key_per_batch, k_new_cpu[i]], dim=0).unsqueeze(0)
            value_per_batch = torch.cat([value_per_batch, v_new_cpu[i]], dim=0).unsqueeze(0)
            atten_mask_i, is_causal_i, is_local_i = make_golden_attention_mask(
                q_seqlen_per_batch, kv_len_i, is_causal, window_size_left, window_size_right)
            out_ref, lse_ref, out_pt, lse_pt = ref_flash_attention_pair(
                query_cpu_per_batch, key_per_batch, value_per_batch, scale,
                atten_mask_i if (is_causal_i or is_local_i) else None,
                data_type, softcap,
            )
            out_ref, out_pt = out_ref[0], out_pt[0]
            lse_ref, lse_pt = lse_ref[0], lse_pt[0]
            if atten_mask_i is not None:
                fully_masked_i = atten_mask_i.all(dim=-1)
                out_ref[fully_masked_i] = 0
                out_pt[fully_masked_i] = 0
                lse_ref[:, fully_masked_i] = torch.inf
                lse_pt[:, fully_masked_i] = torch.inf
            if layout == "BSND":
                golden_out_ref[i] = out_ref
                golden_out_pt[i] = out_pt
                golden_lseL_ref[i] = lse_ref
                golden_lseL_pt[i] = lse_pt
            else:
                golden_out_ref[new_q_seqlen_list_cpu[i] : new_q_seqlen_list_cpu[i + 1]] = out_ref
                golden_out_pt[new_q_seqlen_list_cpu[i] : new_q_seqlen_list_cpu[i + 1]] = out_pt
                golden_lseL_ref[:, new_q_seqlen_list_cpu[i] : new_q_seqlen_list_cpu[i + 1]] = lse_ref
                golden_lseL_pt[:, new_q_seqlen_list_cpu[i] : new_q_seqlen_list_cpu[i + 1]] = lse_pt
        assert_fa_close(out_out, golden_out_ref, golden_out_pt, softcap=softcap, name="out")
        assert_fa_close(softmax_lse, golden_lseL_ref, golden_lseL_pt, softcap=softcap, name="softmax_lse")
        check_kvcache_inplace(key_cache_orig, value_cache_orig, key_cache, value_cache,
                              k_new, v_new, cache_seqlens, block_tables, block_size)
        return

    golden_out_ref = None
    golden_out_pt = None
    if layout == "BSND":
        golden_out_ref = torch.empty((batch_size, q_seqlen, num_heads, head_size), dtype=data_type)
        golden_out_pt = torch.empty_like(golden_out_ref)
        golden_lseL_ref = torch.empty((batch_size, num_heads, q_seqlen), dtype=torch.float32)
        golden_lseL_pt = torch.empty_like(golden_lseL_ref)
    else:
        golden_out_ref = torch.empty((t_q_sum, num_heads, head_size), dtype=data_type)
        golden_out_pt = torch.empty_like(golden_out_ref)
        golden_lseL_ref = torch.empty((num_heads, t_q_sum), dtype=torch.float32)
        golden_lseL_pt = torch.empty_like(golden_lseL_ref)
    query_cpu = query.detach().cpu()
    key_cache_cpu = key_cache.detach().cpu()
    value_cache_cpu = value_cache.detach().cpu()
    block_tables_cpu = block_tables.cpu() if cache_mode == 1 else None
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
                key_cache_cpu, value_cache_cpu, block_tables_cpu, kv_seqlen, block_size
            )
        else:
            key_batched, value_batched = key_cache_cpu, value_cache_cpu
        golden_out_ref, golden_lseL_ref, golden_out_pt, golden_lseL_pt = ref_flash_attention_pair(
            query_cpu, key_batched, value_batched, scale, atten_mask, data_type, softcap
        )
        if atten_mask is not None:
            fully_masked = atten_mask.all(dim=-1)
            golden_out_ref[:, fully_masked] = 0
            golden_out_pt[:, fully_masked] = 0
            golden_lseL_ref[:, :, fully_masked] = torch.inf
            golden_lseL_pt[:, :, fully_masked] = torch.inf
        assert_fa_close(out_out, golden_out_ref, golden_out_pt, softcap=softcap, name="out")
        assert_fa_close(softmax_lse, golden_lseL_ref, golden_lseL_pt, softcap=softcap, name="softmax_lse")
        return
    query_padded = pad_packed_tensor(query_cpu, q_sequences, q_seqlen)
    if cache_mode == 1:
        key_padded, value_padded = gather_paged_kv_batch(
            key_cache_cpu, value_cache_cpu, block_tables_cpu, kv_seqlen, block_size
        )
    else:
        key_padded = pad_packed_tensor(key_cache_cpu, kv_sequences, kv_seqlen)
        value_padded = pad_packed_tensor(value_cache_cpu, kv_sequences, kv_seqlen)
    q_valid = torch.arange(q_seqlen) < torch.tensor(q_sequences)[:, None]
    k_valid = torch.arange(kv_seqlen) < torch.tensor(kv_sequences)[:, None]
    atten_mask = (~q_valid[:, :, None]) | (~k_valid[:, None, :])
    row = torch.arange(q_seqlen)[None, :, None]
    col = torch.arange(kv_seqlen)[None, None, :]
    if is_causal_golden:
        atten_mask = atten_mask | (col - row >= (torch.tensor(kv_sequences) - torch.tensor(q_sequences))[:, None, None] + 1)
    elif is_local_golden:
        diff = col - row
        left = (torch.tensor(kv_sequences) - torch.tensor(q_sequences))[:, None, None] - window_size_left_golden
        right = (torch.tensor(kv_sequences) - torch.tensor(q_sequences))[:, None, None] + window_size_right_golden
        atten_mask = atten_mask | (diff < left) | (diff > right)
    golden_out_ref, golden_lse_ref, golden_out_pt, golden_lse_pt = ref_flash_attention_pair(
        query_padded, key_padded, value_padded, scale, atten_mask, data_type, softcap
    )
    fully_masked = atten_mask.all(dim=-1)
    golden_out_ref[fully_masked] = 0
    golden_out_pt[fully_masked] = 0
    golden_lse_ref = golden_lse_ref.masked_fill(fully_masked[:, None, :], torch.inf)
    golden_lse_pt = golden_lse_pt.masked_fill(fully_masked[:, None, :], torch.inf)
    golden_out_ref = golden_out_ref[q_valid]
    golden_out_pt = golden_out_pt[q_valid]
    golden_lse_ref = golden_lse_ref.permute(0, 2, 1)[q_valid].transpose(0, 1)
    golden_lse_pt = golden_lse_pt.permute(0, 2, 1)[q_valid].transpose(0, 1)
    assert_fa_close(out_out, golden_out_ref, golden_out_pt, softcap=softcap, name="out")
    assert_fa_close(softmax_lse, golden_lse_ref, golden_lse_pt, softcap=softcap, name="softmax_lse")
    return

# flash_attn_func test parameters
# Single-option parameters: fixed values
# batch_size: [4]

# Two-option parameters
# data_type: [torch.float16, torch.bfloat16]
# return_attn_probs: [False, True]
# is_causal: [False, True]

# Multi-option parameters: grouped values
# softcap,num_heads,kv_heads: A=[(0.0,6,6), (0.0,6,1), (0.0,6,2), (2.0,4,4), (2.0,4,1), (2.0,4,2)]
# head_size: A=[32,40,59,64,96,111], B=[128,160,192,224,256]
# q_seqlen,kv_seqlen: A=[(113,203),(128,217),(113,211),(108,256),(256,512)], B=[(512,256),(1024,1024),(1023,1024),(1024,1023),(2048,2048)]
# window_size_left,window_size_right: A=[(-1,-1),(0,256),(64,128)], B=[(512,0),(542,647),(826,973)]
func_cases = [
    # data_type=torch.float16, return_attn_probs=False, is_causal=False
    # softcap,num_heads,kv_heads=A, head_size=A, (q_seqlen,kv_seqlen)=A, (window_size_left,window_size_right)=A
    (torch.float16, 4, 6, 6, 256, 512, 40, False, False, -1, -1, 0.0),
    (torch.float16, 4, 6, 1, 108, 256, 96, False, False, 0, 256, 0.0),
    (torch.float16, 4, 6, 2, 113, 211, 59, False, False, 64, 128, 0.0),
    (torch.float16, 4, 4, 4, 113, 203, 111, False, False, -1, -1, 2.0),
    (torch.float16, 4, 4, 1, 128, 217, 64, False, False, 0, 256, 2.0),
    (torch.float16, 4, 4, 2, 256, 512, 32, False, False, 64, 128, 2.0),
    # data_type=torch.bfloat16, return_attn_probs=False, is_causal=False
    # softcap,num_heads,kv_heads=A, head_size=A, (q_seqlen,kv_seqlen)=A, (window_size_left,window_size_right)=B
    (torch.bfloat16, 4, 6, 6, 113, 203, 40, False, False, 512, 0, 0.0),
    (torch.bfloat16, 4, 6, 1, 128, 217, 64, False, False, 542, 647, 0.0),
    (torch.bfloat16, 4, 6, 2, 113, 211, 96, False, False, 826, 973, 0.0),
    (torch.bfloat16, 4, 4, 4, 256, 512, 32, False, False, 512, 0, 2.0),
    (torch.bfloat16, 4, 4, 1, 108, 256, 59, False, False, 542, 647, 2.0),
    (torch.bfloat16, 4, 4, 2, 113, 203, 111, False, False, 826, 973, 2.0),
    # data_type=torch.float16, return_attn_probs=False, is_causal=True
    # softcap,num_heads,kv_heads=A, head_size=A, (q_seqlen,kv_seqlen)=B, (window_size_left,window_size_right)=A
    (torch.float16, 4, 6, 6, 1024, 1024, 111, False, True, 0, 256, 0.0),
    (torch.float16, 4, 6, 1, 1024, 1023, 59, False, True, -1, -1, 0.0),
    (torch.float16, 4, 6, 2, 512, 256, 32, False, True, 64, 128, 0.0),
    (torch.float16, 4, 4, 4, 2048, 2048, 96, False, True, 0, 256, 2.0),
    (torch.float16, 4, 4, 1, 1023, 1024, 40, False, True, -1, -1, 2.0),
    (torch.float16, 4, 4, 2, 1024, 1024, 64, False, True, 64, 128, 2.0),
    # data_type=torch.bfloat16, return_attn_probs=False, is_causal=True
    # softcap,num_heads,kv_heads=A, head_size=A, (q_seqlen,kv_seqlen)=B, (window_size_left,window_size_right)=B
    (torch.bfloat16, 4, 6, 6, 1024, 1023, 64, False, True, 512, 0, 0.0),
    (torch.bfloat16, 4, 6, 1, 1023, 1024, 59, False, True, 826, 973, 0.0),
    (torch.bfloat16, 4, 6, 2, 1024, 1024, 40, False, True, 542, 647, 0.0),
    (torch.bfloat16, 4, 4, 4, 2048, 2048, 32, False, True, 512, 0, 2.0),
    (torch.bfloat16, 4, 4, 1, 512, 256, 96, False, True, 826, 973, 2.0),
    (torch.bfloat16, 4, 4, 2, 1024, 1023, 111, False, True, 542, 647, 2.0),
    # data_type=torch.float16, return_attn_probs=True, is_causal=False
    # softcap,num_heads,kv_heads=A, head_size=B, (q_seqlen,kv_seqlen)=A, (window_size_left,window_size_right)=A
    (torch.float16, 4, 6, 6, 256, 512, 256, True, False, -1, -1, 0.0),
    (torch.float16, 4, 6, 1, 113, 203, 128, True, False, 64, 128, 0.0),
    (torch.float16, 4, 6, 2, 108, 256, 224, True, False, 0, 256, 0.0),
    (torch.float16, 4, 4, 4, 113, 211, 192, True, False, -1, -1, 2.0),
    (torch.float16, 4, 4, 1, 128, 217, 160, True, False, 64, 128, 2.0),
    (torch.float16, 4, 4, 2, 256, 512, 256, True, False, 0, 256, 2.0),
    # data_type=torch.bfloat16, return_attn_probs=True, is_causal=False
    # softcap,num_heads,kv_heads=A, head_size=B, (q_seqlen,kv_seqlen)=A, (window_size_left,window_size_right)=B
    (torch.bfloat16, 4, 6, 6, 256, 512, 256, True, False, 512, 0, 0.0),
    (torch.bfloat16, 4, 6, 1, 128, 217, 160, True, False, 542, 647, 0.0),
    (torch.bfloat16, 4, 6, 2, 113, 211, 192, True, False, 826, 973, 0.0),
    (torch.bfloat16, 4, 4, 4, 108, 256, 224, True, False, 512, 0, 2.0),
    (torch.bfloat16, 4, 4, 1, 113, 203, 128, True, False, 542, 647, 2.0),
    (torch.bfloat16, 4, 4, 2, 256, 512, 256, True, False, 826, 973, 2.0),
    # data_type=torch.float16, return_attn_probs=True, is_causal=True
    # softcap,num_heads,kv_heads=A, head_size=B, (q_seqlen,kv_seqlen)=B, (window_size_left,window_size_right)=A
    (torch.float16, 4, 6, 6, 1024, 1024, 160, True, True, -1, -1, 0.0),
    (torch.float16, 4, 6, 1, 1023, 1024, 192, True, True, 64, 128, 0.0),
    (torch.float16, 4, 6, 2, 512, 256, 128, True, True, 0, 256, 0.0),
    (torch.float16, 4, 4, 4, 2048, 2048, 256, True, True, -1, -1, 2.0),
    (torch.float16, 4, 4, 1, 1024, 1023, 224, True, True, 64, 128, 2.0),
    (torch.float16, 4, 4, 2, 1024, 1024, 160, True, True, 0, 256, 2.0),
    # data_type=torch.bfloat16, return_attn_probs=True, is_causal=True
    # softcap,num_heads,kv_heads=A, head_size=B, (q_seqlen,kv_seqlen)=B, (window_size_left,window_size_right)=B
    (torch.bfloat16, 4, 6, 6, 1023, 1024, 192, True, True, 512, 0, 0.0),
    (torch.bfloat16, 4, 6, 1, 2048, 2048, 256, True, True, 826, 973, 0.0),
    (torch.bfloat16, 4, 6, 2, 512, 256, 128, True, True, 542, 647, 0.0),
    (torch.bfloat16, 4, 4, 4, 1024, 1023, 224, True, True, 512, 0, 2.0),
    (torch.bfloat16, 4, 4, 1, 1024, 1024, 160, True, True, 826, 973, 2.0),
    (torch.bfloat16, 4, 4, 2, 1023, 1024, 192, True, True, 542, 647, 2.0),
]

@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, return_attn_probs, is_causal, window_size_left, window_size_right, softcap", func_cases)
def test_fa_func_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, return_attn_probs, is_causal, window_size_left, window_size_right, softcap):
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if "Ascend910" not in name and "Ascend950" not in name:
        pytest.skip("flash_attn_func only supports Ascend910/Ascend950")
    if "Ascend950" in name and (softcap != 0.0):
        pytest.skip("Ascend950 does not support softcap")
    query, key_cache, value_cache, dout = make_attention_inputs(
        (batch_size, q_seqlen, num_heads, head_size),
        (batch_size, kv_seqlen, kv_heads, head_size),
        (batch_size, kv_seqlen, kv_heads, head_size),
        (batch_size, q_seqlen, num_heads, head_size),
        data_type,
        device="npu",
    )
    scale = 1.0 / (head_size ** 0.5)
    ret = flash_attn_func(
        query,
        key_cache,
        value_cache,
        softmax_scale=scale,
        causal=is_causal,
        window_size=[window_size_left, window_size_right],
        softcap=softcap,
        return_attn_probs=return_attn_probs)
    if not return_attn_probs:
        out_out = ret
    else:
        out_out, softmax_lse = ret

    query_ref = query.detach().cpu().requires_grad_(True)
    key_ref = key_cache.detach().cpu().requires_grad_(True)
    value_ref = value_cache.detach().cpu().requires_grad_(True)
    atten_mask, _, _ = make_golden_attention_mask(
        q_seqlen,
        kv_seqlen,
        is_causal,
        window_size_left,
        window_size_right,
    )
    golden_out_ref, golden_lseL_ref, golden_out_pt, golden_lseL_pt = ref_flash_attention_pair(
        query_ref, key_ref, value_ref, scale, atten_mask, data_type, softcap
    )
    if atten_mask is not None:
        fully_masked = atten_mask.all(dim=-1)
        golden_out_ref[:, fully_masked] = 0
        golden_out_pt[:, fully_masked] = 0
        golden_lseL_ref[:, :, fully_masked] = torch.inf
        golden_lseL_pt[:, :, fully_masked] = torch.inf

    assert_fa_close(out_out, golden_out_ref, golden_out_pt, softcap=softcap, name="out")
    if return_attn_probs and "Ascend910" in name:
        assert_fa_close(
            softmax_lse,
            golden_lseL_ref,
            golden_lseL_pt,
            softcap=softcap,
            name="softmax_lse",
        )
    if "Ascend910" in name:
        dq_ag, dk_ag, dv_ag = torch.autograd.grad(out_out, (query, key_cache, value_cache), dout)
        dout_ref = dout.detach().cpu()
        dq_ref, dk_ref, dv_ref, dq_pt, dk_pt, dv_pt = cached_autograd_grads(
            os.environ.get("GOLDEN_CACHE_NODEID", "v3"),
            (golden_out_ref, golden_out_pt),
            (query_ref, key_ref, value_ref),
            dout_ref,
            metadata={"version": 3, "kind": "bsnd"},
        )
        assert_fa_close(dq_ag, dq_ref, dq_pt, softcap=softcap, name="dQ")
        assert_fa_close(dk_ag, dk_ref, dk_pt, softcap=softcap, name="dK")
        assert_fa_close(dv_ag, dv_ref, dv_pt, softcap=softcap, name="dV")


# flash_attn_varlen_func test parameters
# Single-option parameters: fixed values
# batch_size: [4]
# block_size: [128]

# Two-option parameters
# data_type: [torch.float16, torch.bfloat16]
# is_causal: [False, True]
# cache_mode: [0, 1]

# Multi-option parameters: grouped values
# softcap,num_heads,kv_heads: A=[(0.0,6,6), (0.0,6,1), (0.0,6,2), (2.0,4,4), (2.0,4,1), (2.0,4,2)]
# head_size: A=[32,59,64,80,96,111], B=[128,160,192,224,256]
# q_seqlen,kv_seqlen: A=[(1,147),(113,203),(128,217),(113,211),(108,256),(256,512)], B=[(512,256),(1024,1024),(1023,1024),(1024,1023),(2048,2048)]
# window_size_left,window_size_right: A=[(-1,-1),(0,256),(64,128)], B=[(512,0),(542,647),(826,973)]
varlen_cases = [
    # data_type=torch.float16, is_causal=False, cache_mode=0
    # softcap,num_heads,kv_heads=A, head_size=A, (q_seqlen,kv_seqlen)=A, (window_size_left,window_size_right)=A
    (torch.float16, 4, 6, 6, 113, 203, 59, 0, 128, False, -1, -1, 0.0),
    (torch.float16, 4, 6, 1, 108, 256, 96, 0, 128, False, 0, 256, 0.0),
    (torch.float16, 4, 6, 2, 128, 217, 64, 0, 128, False, 64, 128, 0.0),
    (torch.float16, 4, 4, 4, 256, 512, 111, 0, 128, False, -1, -1, 2.0),
    (torch.float16, 4, 4, 1, 113, 211, 80, 0, 128, False, 0, 256, 2.0),
    (torch.float16, 4, 4, 2, 1, 147, 32, 0, 128, False, 64, 128, 2.0),
    # data_type=torch.bfloat16, is_causal=False, cache_mode=0
    # softcap,num_heads,kv_heads=A, head_size=A, (q_seqlen,kv_seqlen)=A, (window_size_left,window_size_right)=B
    (torch.bfloat16, 4, 6, 6, 113, 203, 59, 0, 128, False, 512, 0, 0.0),
    (torch.bfloat16, 4, 6, 1, 113, 211, 80, 0, 128, False, 542, 647, 0.0),
    (torch.bfloat16, 4, 6, 2, 108, 256, 96, 0, 128, False, 826, 973, 0.0),
    (torch.bfloat16, 4, 4, 4, 1, 147, 32, 0, 128, False, 512, 0, 2.0),
    (torch.bfloat16, 4, 4, 1, 128, 217, 64, 0, 128, False, 542, 647, 2.0),
    (torch.bfloat16, 4, 4, 2, 256, 512, 111, 0, 128, False, 826, 973, 2.0),
    # data_type=torch.float16, is_causal=True, cache_mode=0
    # softcap,num_heads,kv_heads=A, head_size=A, (q_seqlen,kv_seqlen)=B, (window_size_left,window_size_right)=A
    (torch.float16, 4, 6, 6, 1024, 1024, 111, 0, 128, True, 0, 256, 0.0),
    (torch.float16, 4, 6, 1, 1024, 1023, 64, 0, 128, True, -1, -1, 0.0),
    (torch.float16, 4, 6, 2, 512, 256, 32, 0, 128, True, 64, 128, 0.0),
    (torch.float16, 4, 4, 4, 2048, 2048, 96, 0, 128, True, 0, 256, 2.0),
    (torch.float16, 4, 4, 1, 1023, 1024, 59, 0, 128, True, -1, -1, 2.0),
    (torch.float16, 4, 4, 2, 1024, 1024, 80, 0, 128, True, 64, 128, 2.0),
    # data_type=torch.bfloat16, is_causal=True, cache_mode=0
    # softcap,num_heads,kv_heads=A, head_size=A, (q_seqlen,kv_seqlen)=B, (window_size_left,window_size_right)=B
    (torch.bfloat16, 4, 6, 6, 1024, 1023, 80, 0, 128, True, 512, 0, 0.0),
    (torch.bfloat16, 4, 6, 1, 1023, 1024, 64, 0, 128, True, 826, 973, 0.0),
    (torch.bfloat16, 4, 6, 2, 1024, 1024, 59, 0, 128, True, 542, 647, 0.0),
    (torch.bfloat16, 4, 4, 4, 2048, 2048, 32, 0, 128, True, 512, 0, 2.0),
    (torch.bfloat16, 4, 4, 1, 512, 256, 96, 0, 128, True, 826, 973, 2.0),
    (torch.bfloat16, 4, 4, 2, 1024, 1023, 111, 0, 128, True, 542, 647, 2.0),
    # data_type=torch.float16, is_causal=False, cache_mode=1
    # softcap,num_heads,kv_heads=A, head_size=B, (q_seqlen,kv_seqlen)=A, (window_size_left,window_size_right)=A
    (torch.float16, 4, 6, 6, 113, 203, 256, 1, 128, False, -1, -1, 0.0),
    (torch.float16, 4, 6, 1, 108, 256, 128, 1, 128, False, 64, 128, 0.0),
    (torch.float16, 4, 6, 2, 113, 211, 224, 1, 128, False, 0, 256, 0.0),
    (torch.float16, 4, 4, 4, 1, 147, 192, 1, 128, False, -1, -1, 2.0),
    (torch.float16, 4, 4, 1, 128, 217, 160, 1, 128, False, 64, 128, 2.0),
    (torch.float16, 4, 4, 2, 256, 512, 256, 1, 128, False, 0, 256, 2.0),
    # data_type=torch.bfloat16, is_causal=False, cache_mode=1
    # softcap,num_heads,kv_heads=A, head_size=B, (q_seqlen,kv_seqlen)=A, (window_size_left,window_size_right)=B
    (torch.bfloat16, 4, 6, 6, 1, 147, 256, 1, 128, False, 512, 0, 0.0),
    (torch.bfloat16, 4, 6, 1, 256, 512, 160, 1, 128, False, 542, 647, 0.0),
    (torch.bfloat16, 4, 6, 2, 108, 256, 192, 1, 128, False, 826, 973, 0.0),
    (torch.bfloat16, 4, 4, 4, 113, 211, 224, 1, 128, False, 512, 0, 2.0),
    (torch.bfloat16, 4, 4, 1, 113, 203, 128, 1, 128, False, 542, 647, 2.0),
    (torch.bfloat16, 4, 4, 2, 128, 217, 256, 1, 128, False, 826, 973, 2.0),
    # data_type=torch.float16, is_causal=True, cache_mode=1
    # softcap,num_heads,kv_heads=A, head_size=B, (q_seqlen,kv_seqlen)=B, (window_size_left,window_size_right)=A
    (torch.float16, 4, 6, 6, 1024, 1024, 160, 1, 128, True, -1, -1, 0.0),
    (torch.float16, 4, 6, 1, 1023, 1024, 192, 1, 128, True, 64, 128, 0.0),
    (torch.float16, 4, 6, 2, 512, 256, 128, 1, 128, True, 0, 256, 0.0),
    (torch.float16, 4, 4, 4, 2048, 2048, 256, 1, 128, True, -1, -1, 2.0),
    (torch.float16, 4, 4, 1, 1024, 1023, 224, 1, 128, True, 64, 128, 2.0),
    (torch.float16, 4, 4, 2, 1024, 1024, 160, 1, 128, True, 0, 256, 2.0),
    # data_type=torch.bfloat16, is_causal=True, cache_mode=1
    # softcap,num_heads,kv_heads=A, head_size=B, (q_seqlen,kv_seqlen)=B, (window_size_left,window_size_right)=B
    (torch.bfloat16, 4, 6, 6, 1023, 1024, 192, 1, 128, True, 512, 0, 0.0),
    (torch.bfloat16, 4, 6, 1, 2048, 2048, 256, 1, 128, True, 826, 973, 0.0),
    (torch.bfloat16, 4, 6, 2, 512, 256, 128, 1, 128, True, 542, 647, 0.0),
    (torch.bfloat16, 4, 4, 4, 1024, 1023, 224, 1, 128, True, 512, 0, 2.0),
    (torch.bfloat16, 4, 4, 1, 1024, 1024, 160, 1, 128, True, 826, 973, 2.0),
    (torch.bfloat16, 4, 4, 2, 1023, 1024, 192, 1, 128, True, 542, 647, 2.0),
]

@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, window_size_left, window_size_right, softcap", varlen_cases)
def test_fa_varlen_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, window_size_left, window_size_right, softcap):
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if "Ascend910" not in name and "Ascend950" not in name:
        pytest.skip("flash_attn_varlen_func only supports Ascend910/Ascend950")
    if "Ascend950" in name and (softcap != 0.0):
        pytest.skip("Ascend950 does not support softcap")
    seqlens_q, seqlens_k = make_varlen_seqlens(batch_size, q_seqlen, kv_seqlen)
    cu_q = make_cu_seqlens(seqlens_q)
    cu_k = make_cu_seqlens(seqlens_k)
    total_q = int(cu_q[-1].item())
    total_k = int(cu_k[-1].item())
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)
    query = make_packed_random_tensor(seqlens_q, max_seqlen_q, num_heads, head_size, data_type, device="npu", requires_grad=True)
    key = make_packed_random_tensor(seqlens_k, max_seqlen_k, kv_heads, head_size, data_type, device="npu", requires_grad=True)
    value = make_packed_random_tensor(seqlens_k, max_seqlen_k, kv_heads, head_size, data_type, device="npu", requires_grad=True)
    actual_seq_len = cu_q.npu()
    actual_kv_len = cu_k.npu()

    scale = 1.0 / (head_size ** 0.5)

    output_npu, softmax_lse = flash_attn_varlen_func(
        query,
        key,
        value,
        actual_seq_len,
        actual_kv_len,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=scale,
        causal=is_causal,
        window_size=(window_size_left, window_size_right),
        softcap=softcap,
        return_attn_probs=True,
    )
    query_ref = query.detach().cpu().requires_grad_(True)
    key_ref = key.detach().cpu().requires_grad_(True)
    value_ref = value.detach().cpu().requires_grad_(True)
    query_padded = pad_packed_tensor(query_ref, seqlens_q, max_seqlen_q)
    key_padded = pad_packed_tensor(key_ref, seqlens_k, max_seqlen_k)
    value_padded = pad_packed_tensor(value_ref, seqlens_k, max_seqlen_k)
    q_valid, k_valid, atten_mask = make_padded_varlen_mask(
        seqlens_q,
        seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal,
        window_size_left,
        window_size_right,
    )
    golden_out_ref, golden_lse_ref, golden_out_pt, golden_lse_pt = ref_flash_attention_pair(
        query_padded, key_padded, value_padded, scale, atten_mask, data_type, softcap
    )
    fully_masked = atten_mask.all(dim=-1)
    golden_out_ref[fully_masked] = 0
    golden_out_pt[fully_masked] = 0
    golden_lse_ref = golden_lse_ref.masked_fill(fully_masked[:, None, :], torch.inf)
    golden_lse_pt = golden_lse_pt.masked_fill(fully_masked[:, None, :], torch.inf)
    golden_out_ref = golden_out_ref[q_valid]
    golden_out_pt = golden_out_pt[q_valid]
    golden_lseL_ref = golden_lse_ref.permute(0, 2, 1)[q_valid].transpose(0, 1)
    golden_lseL_pt = golden_lse_pt.permute(0, 2, 1)[q_valid].transpose(0, 1)
    assert_fa_close(output_npu, golden_out_ref, golden_out_pt, softcap=softcap, name="out")
    if "Ascend910" in name:
        assert_fa_close(softmax_lse, golden_lseL_ref, golden_lseL_pt, softcap=softcap, name="softmax_lse")
        dout = make_random_tensor(output_npu.shape, output_npu.dtype, low=-0.5, high=0.5, device="npu")
        dq_ag, dk_ag, dv_ag = torch.autograd.grad(output_npu, (query, key, value), dout)
        dq_ref, dk_ref, dv_ref, dq_pt, dk_pt, dv_pt = cached_autograd_grads(
            os.environ.get("GOLDEN_CACHE_NODEID", "v3-varlen"),
            (golden_out_ref, golden_out_pt),
            (query_ref, key_ref, value_ref),
            dout.detach().cpu(),
            metadata={"version": 3, "kind": "varlen"},
        )
        assert_fa_close(dq_ag, dq_ref, dq_pt, softcap=softcap, name="dQ")
        assert_fa_close(dk_ag, dk_ref, dk_pt, softcap=softcap, name="dK")
        assert_fa_close(dv_ag, dv_ref, dv_pt, softcap=softcap, name="dV")

# flash_attn_with_kvcache test parameters (Ascend950 head_dim<=256 coverage, NPU only)
# Single-option parameters: fixed values
# data_type: [torch.float16]
# kv_heads: [2]
# block_size: [128]
# softcap: [0.0]
# window_size_left,window_size_right: [(-1,-1)]

# Two-option parameters
# is_causal: [False, True]
# cache_mode: [0, 1]
# layout: [BSND, TND]
# num_heads: baseline coverage [16], dedicated S/N axis-fusion coverage [2, 64]

# Multi-option parameters: grouped values
# head_size: A=[35, 64, 101, 128], B=[151, 192, 201, 256]
# batch_size,q_seqlen,kv_seqlen: A=[(1,256,128), (1,136,128), (2,256,256), (4,128,256), (2,128,128)], B=[(1,1024,128), (1,256,384), (1,128,128), (1,256,512), (1,256,192)]
# num_splits: A=[0, 1, 2] for cache=1 with layout=TND; B=[0, 1] otherwise
# Dedicated S/N axis-fusion cases also cover batch=[8,16],
# q_seqlen=[16,32,64], and kv_seqlen=[16,32,64,1024]

hd_cases = [
    # data_type=torch.float16, is_causal=False, cache_mode=0, layout=BSND
    # head_size=A, (batch_size,q_seqlen,kv_seqlen)=A, num_splits=[0,1]
    (torch.float16, 2, 16, 2, 128, 128, 64, 0, 128, False, "BSND", 0, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 256, 128, 101, 0, 128, False, "BSND", 1, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 136, 128, 128, 0, 128, False, "BSND", 0, -1, -1, 0.0),
    (torch.float16, 4, 16, 2, 128, 256, 35, 0, 128, False, "BSND", 1, -1, -1, 0.0),
    (torch.float16, 2, 16, 2, 256, 256, 64, 0, 128, False, "BSND", 0, -1, -1, 0.0),
    (torch.float16, 2, 16, 2, 128, 128, 101, 0, 128, False, "BSND", 1, -1, -1, 0.0),
    # data_type=torch.float16, is_causal=True, cache_mode=0, layout=BSND
    # head_size=A, (batch_size,q_seqlen,kv_seqlen)=B, num_splits=[0,1]
    (torch.float16, 1, 16, 2, 256, 384, 128, 0, 128, True, "BSND", 0, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 128, 128, 101, 0, 128, True, "BSND", 1, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 1024, 128, 35, 0, 128, True, "BSND", 0, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 256, 192, 64, 0, 128, True, "BSND", 1, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 256, 512, 128, 0, 128, True, "BSND", 0, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 256, 384, 101, 0, 128, True, "BSND", 1, -1, -1, 0.0),
    # data_type=torch.float16, is_causal=False, cache_mode=1, layout=BSND
    # head_size=B, (batch_size,q_seqlen,kv_seqlen)=A, num_splits=[0,1]
    (torch.float16, 4, 16, 2, 128, 256, 192, 1, 128, False, "BSND", 0, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 136, 128, 201, 1, 128, False, "BSND", 1, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 256, 128, 256, 1, 128, False, "BSND", 0, -1, -1, 0.0),
    (torch.float16, 2, 16, 2, 256, 256, 151, 1, 128, False, "BSND", 1, -1, -1, 0.0),
    (torch.float16, 2, 16, 2, 128, 128, 192, 1, 128, False, "BSND", 0, -1, -1, 0.0),
    (torch.float16, 4, 16, 2, 128, 256, 201, 1, 128, False, "BSND", 1, -1, -1, 0.0),
    # data_type=torch.float16, is_causal=True, cache_mode=1, layout=BSND
    # head_size=B, (batch_size,q_seqlen,kv_seqlen)=B, num_splits=[0,1]
    (torch.float16, 1, 16, 2, 256, 512, 201, 1, 128, True, "BSND", 0, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 128, 128, 256, 1, 128, True, "BSND", 1, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 256, 384, 151, 1, 128, True, "BSND", 0, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 256, 192, 192, 1, 128, True, "BSND", 1, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 1024, 128, 201, 1, 128, True, "BSND", 0, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 256, 512, 256, 1, 128, True, "BSND", 1, -1, -1, 0.0),
    # data_type=torch.float16, is_causal=False, cache_mode=0, layout=TND
    # head_size=A, (batch_size,q_seqlen,kv_seqlen)=A, num_splits=[0,1]
    (torch.float16, 4, 16, 2, 128, 256, 101, 0, 128, False, "TND", 0, -1, -1, 0.0),
    (torch.float16, 2, 16, 2, 256, 256, 128, 0, 128, False, "TND", 1, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 256, 128, 64, 0, 128, False, "TND", 0, -1, -1, 0.0),
    (torch.float16, 2, 16, 2, 128, 128, 35, 0, 128, False, "TND", 1, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 136, 128, 101, 0, 128, False, "TND", 0, -1, -1, 0.0),
    (torch.float16, 4, 16, 2, 128, 256, 128, 0, 128, False, "TND", 1, -1, -1, 0.0),
    # data_type=torch.float16, is_causal=True, cache_mode=0, layout=TND
    # head_size=A, (batch_size,q_seqlen,kv_seqlen)=B, num_splits=[0,1]
    (torch.float16, 1, 16, 2, 1024, 128, 64, 0, 128, True, "TND", 0, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 256, 192, 101, 0, 128, True, "TND", 1, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 128, 128, 35, 0, 128, True, "TND", 0, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 256, 512, 128, 0, 128, True, "TND", 1, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 256, 384, 64, 0, 128, True, "TND", 0, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 1024, 128, 101, 0, 128, True, "TND", 1, -1, -1, 0.0),
    # data_type=torch.float16, is_causal=False, cache_mode=1, layout=TND
    # head_size=B, (batch_size,q_seqlen,kv_seqlen)=A, num_splits=[0,1,2]
    (torch.float16, 2, 16, 2, 128, 128, 192, 1, 128, False, "TND", 0, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 256, 128, 201, 1, 128, False, "TND", 1, -1, -1, 0.0),
    (torch.float16, 4, 16, 2, 128, 256, 256, 1, 128, False, "TND", 2, -1, -1, 0.0),
    (torch.float16, 2, 16, 2, 256, 256, 151, 1, 128, False, "TND", 0, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 136, 128, 192, 1, 128, False, "TND", 1, -1, -1, 0.0),
    (torch.float16, 2, 16, 2, 128, 128, 201, 1, 128, False, "TND", 2, -1, -1, 0.0),
    # data_type=torch.float16, is_causal=True, cache_mode=1, layout=TND
    # head_size=B, (batch_size,q_seqlen,kv_seqlen)=B, num_splits=[0,1,2]
    (torch.float16, 1, 16, 2, 256, 384, 192, 1, 128, True, "TND", 0, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 256, 512, 151, 1, 128, True, "TND", 1, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 1024, 128, 201, 1, 128, True, "TND", 2, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 128, 128, 256, 1, 128, True, "TND", 0, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 256, 192, 192, 1, 128, True, "TND", 1, -1, -1, 0.0),
    (torch.float16, 1, 16, 2, 256, 384, 151, 1, 128, True, "TND", 2, -1, -1, 0.0),
    # Ascend950 S/N axis-fusion coverage: large batch, small S, many query
    # heads, and unaligned head_size
    (torch.float16, 16, 2, 2, 64, 64, 35, 0, 128, False, "BSND", 0, -1, -1, 0.0),
    (torch.float16, 16, 64, 2, 32, 32, 101, 0, 128, True, "BSND", 1, -1, -1, 0.0),
    (torch.float16, 16, 2, 2, 16, 16, 151, 1, 128, False, "TND", 0, -1, -1, 0.0),
    (torch.float16, 8, 64, 2, 64, 1024, 192, 1, 128, True, "TND", 1, -1, -1, 0.0),
    (torch.float16, 8, 2, 2, 16, 1024, 201, 0, 128, False, "TND", 0, -1, -1, 0.0),
]

@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, layout, num_splits, window_size_left, window_size_right, softcap", hd_cases)
def test_fa_kvcache_ops_with_hd_le_256(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, layout, num_splits, window_size_left, window_size_right, softcap):
    is_varied = layout == 'TND'
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if "Ascend910" in name:
        pytest.skip("Sq > Sk not support in Ascend910")
    test_fa_kvcache_ops(
        data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
        cache_mode, block_size, is_causal, layout, is_varied,
        window_size_left, window_size_right, softcap, num_splits, new_kv=False,
    )
