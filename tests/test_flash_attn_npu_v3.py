# Copyright (c) 2026, Minghua Shen.

import sys
import os
import torch
import torch_npu
import pytest
if "Ascend950" in torch_npu.npu.get_device_name():
    from flash_attn_npu_v3 import flash_attn_with_kvcache
else:
    from flash_attn_npu_v3 import flash_attn_with_kvcache, flash_attn_func, flash_attn_varlen_func

def group_matmul(head, kv_head, left, right, high_prec = 1):
    group_num = head // kv_head
    score = None
    for i in range(kv_head):
        if high_prec == 0:
            group_score = torch.matmul(left[i * group_num:(i + 1) * group_num, :, :].to(torch.float32),
                                        right[i:(i + 1), :, :].to(torch.float32)).to(torch.float32)
        else:
            group_score = torch.matmul(left[i * group_num:(i + 1) * group_num, :, :].to(torch.float32),
                                        right[i:(i + 1), :, :].to(torch.float32))
        if score is None:
            score = group_score
        else:
            score = torch.cat((score, group_score), 0)
    return score

def softmax1(
    qk_result,
    is_first,
    gm,
    interm_dtype = torch.float16
    ):
    sim = qk_result.to(interm_dtype)
    lm = torch.max(sim, dim=-1, keepdims=True)[0]
    if is_first:
        hm = lm
        dm = 0
    else:
        hm = torch.maximum(gm, lm)
        dm = gm - hm
    gm = hm
    sim_sub = sim - hm
    sim_sub = torch.exp(sim_sub.to(interm_dtype))
    row_sum = torch.sum(sim_sub, dim=-1, keepdims=True)
    return sim_sub, row_sum, dm, gm

def qkMM1(
    query,
    key
    ):
    result = None
    qk_k = key.shape[1]
    qk_k_split = 128
    qk_k_loop = (qk_k + 127) // 128
    for qk_k_loop_idx in range(qk_k_loop):
        sub_k = 128 if qk_k_loop_idx != (qk_k_loop - 1) else (qk_k - qk_k_loop_idx * 128)
        partial_Query = query[:, :, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k]
        partial_Key = key[:, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k, :]
        result_split = group_matmul(partial_Query.shape[0], partial_Key.shape[0], partial_Query, partial_Key, 0)
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result

def pvMM2(
    p,
    value
    ):
    result = None
    pv_k = value.shape[1]
    pv_k_split = 128
    pv_k_loop = (pv_k + 127) // 128
    for pv_k_loop_idx in range(pv_k_loop):
        sub_k = 128 if pv_k_loop_idx != (pv_k_loop - 1) else (pv_k - pv_k_loop_idx * 128)
        partial_P = p[:, :, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k]
        partial_Value = value[:, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k, :]
        result_split = group_matmul(partial_P.shape[0], partial_Value.shape[0], partial_P, partial_Value, 0)
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result

def ref_flash_attention(
    query,
    key,
    value,
    scale,
    mask,
    data_type,
    ):
    inner_prec = 0
    interm_dtype = torch.float16 if inner_prec == 1 else torch.float32
    query = query.permute(1, 0, 2)
    key = key.permute(1, 2, 0)
    value = value.permute(1, 0, 2)
    scale = torch.tensor(scale)
    scale = scale.to(torch.float16) if inner_prec == 1 else scale.to(torch.float32)
    context_len = key.shape[2]
    context_size = 512
    group_num = query.shape[0] // key.shape[0]
    gl = None
    gl_high = None
    go = None
    go_high = None
    if mask is not None:
        mask = mask.cpu()
    for kv_start in range(0, context_len, context_size):
        sub_len = context_size
        if kv_start + context_size > context_len:
            sub_len = context_len - kv_start
        sub_key = key[:, :, kv_start: kv_start + sub_len]
        sub_mask = None
        if mask is not None:
            sub_mask = mask[:query.shape[1], kv_start : kv_start + sub_len].to(interm_dtype) * (-1e4)
        sub_value = value[:, kv_start: kv_start + sub_len, :]
        qk_result = qkMM1(query, sub_key).to(interm_dtype)
        qk_result = qk_result * scale
        if mask is not None:
            qk_result += sub_mask
        if kv_start == 0:
            gm = None
        p_result, row_sum, dm, gm = softmax1(qk_result, kv_start == 0, gm, interm_dtype)
        p_result = p_result.to(data_type)
        if kv_start == 0:
            gm_high = None
        lo = pvMM2(p_result, sub_value).to(interm_dtype)
        if kv_start == 0:
            gl = row_sum
            go = lo
        else:
            dm = torch.exp(dm)
            gl = gl * dm
            gl = gl + row_sum
            go = go * dm
            go = go + lo
    go = go / gl
    go = go.permute(1, 0, 2)
    lse = torch.squeeze((torch.log(gl) + gm), dim=-1).to(torch.float32)
    return go.to(data_type), lse

test_cases = [
    # (data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode,
    #  block_size, is_causal, layout, is_varied, window_size_left, window_size_right)
    (torch.bfloat16, 2, 6, 2, 2, 1024, 128, 1, 128, False, "BSND", False, -1, -1),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 1, 128, True, "TND", False, -1, -1),
    (torch.float16, 7, 1, 1, 512, 512, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 1, 128, False, "BSND", False, -1, -1),
    (torch.bfloat16, 2, 1, 1, 1024, 1024, 128, 1, 128, False, "BSND", False, -1, -1),
    (torch.bfloat16, 2, 1, 1, 1024, 1024, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 1, 128, True, "BSND", False, -1, -1),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 1, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, 1, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, 1, 128, True, "BSND", False, -1, -1),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, 1, 128, False, "BSND", False, -1, -1),
    # kv=4096 -> 8 S2 blocks: num_splits=2 -> 2 segs (4 blk each), num_splits=4 -> 4 segs (2 blk each).
    (torch.bfloat16, 1, 1, 1, 1, 4096, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 2, 1, 1, 1, 2048, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.float16, 2, 2, 1, 128, 128, 128, 1, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 2, 6, 2, 2, 1024, 128, 1, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 2, 1, 1, 16, 1024, 128, 1, 128, False, "TND", True, -1, -1),
    (torch.bfloat16, 2, 6, 2, 16, 1024, 128, 1, 128, False, "TND", True, -1, -1),
    (torch.bfloat16, 2, 6, 2, 16, 1024, 128, 1, 128, True, "TND", True, -1, -1),
    (torch.bfloat16, 1, 64, 1, 2, 1024, 256, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 2, 1, 1, 16, 1024, 256, 1, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 2, 1, 1, 16, 10240, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 2, 6, 2, 16, 10240, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 6, 1, 1, 16, 10240, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 1, 128, True, "BSND", False, 512, 0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 1, 128, True, "TND", False, 512, 0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 1, 128, False, "TND", False, 0, 256),
    (torch.float16, 2, 1, 1, 512, 512, 128, 1, 128, False, "TND", False, 508, -256),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 1, 128, False, "BSND", False, -128, 1024),
    (torch.float16, 2, 2, 2, 512, 512, 128, 0, 128, False, "TND", False, 64, 128),
]

@pytest.mark.parametrize("num_splits", [0, 1, 2])
@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, layout, is_varied, window_size_left, window_size_right", test_cases)
def test_fa_custom_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, layout, is_varied, num_splits, window_size_left, window_size_right):
    # num_splits>1 (active KV split) is currently only wired for paged KV + varlen-q (TND).

    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if num_splits > 1 and not (cache_mode == 1 and layout == "TND"):
        pytest.skip("num_splits>1 requires paged KV cache and TND (varlen-q) layout")
    if "Ascend950" in name and num_splits > 1:
        pytest.skip("Ascend950 does not support num_splits>1")
    if "Ascend950" in name and head_size > 128:
        pytest.skip("Ascend950 does not support head_size>128")
    if is_varied and layout != "TND":
        pytest.skip("is_varied requires TND (varlen-q) layout")
    q_min_range = -1.0
    q_max_range = 1.0
    kv_min_range = -1.0
    kv_max_range = 1.0
    block_size = 128
    max_num_blocks_per_seq = (kv_seqlen + block_size - 1) // block_size
    num_blocks = max(64, max_num_blocks_per_seq * batch_size)
    if is_varied:
        # Per-batch q in [1, q_seqlen], kv in [q, kv_seqlen] (kv>=q so q>kv never occurs).
        # Seeded for reproducibility; does not perturb the query/key/value RNG streams above.
        gen = torch.Generator().manual_seed(1234)
        q_sequences = torch.randint(low=1, high=q_seqlen + 1, size=(batch_size,), generator=gen).tolist()
        kv_sequences = [int(torch.randint(low=q, high=kv_seqlen + 1, size=(1,), generator=gen))
                        for q in q_sequences]
    else:
        q_sequences = [q_seqlen] * batch_size
        kv_sequences = [kv_seqlen] * batch_size
    t_q_sum = sum(q_sequences)
    t_kv_sum = sum(kv_sequences)
    if layout == "BSND":
        query = (q_min_range + (q_max_range - q_min_range) * torch.rand(batch_size, q_seqlen, num_heads, head_size)).to(data_type).npu()
    elif layout == "TND":
        query = (q_min_range + (q_max_range - q_min_range) * torch.rand(t_q_sum, num_heads, head_size)).to(data_type).npu()
    key_cache = None
    value_cache = None
    block_tables = []
    if cache_mode == 1:
        key_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(num_blocks, block_size, kv_heads, head_size)).to(data_type).npu()
        value_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(num_blocks, block_size, kv_heads, head_size)).to(data_type).npu()
        for i in range(batch_size):
            block_table = [
                max_num_blocks_per_seq * i + j
                for j in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
        block_tables = torch.tensor(block_tables, dtype=torch.int32).npu()
    else:
        if layout == "BSND":
            key_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size, kv_seqlen, kv_heads, head_size)).to(data_type).npu()
            value_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size, kv_seqlen, kv_heads, head_size)).to(data_type).npu()
        else:
            key_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(t_kv_sum, kv_heads, head_size)).to(data_type).npu()
            value_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(t_kv_sum, kv_heads, head_size)).to(data_type).npu()
        block_tables = None
    if layout == "BSND":
        q_seqlen_list = [q_seqlen] * batch_size
        kv_seqlen_list = [kv_seqlen] * batch_size
    else:
        q_seqlen_list = q_sequences
        kv_seqlen_list = kv_sequences
    scale = 1.0 / (head_size ** 0.5)
    is_rotary_interleaved = False
    softcap = 0
    kv_seqlen_list = torch.tensor(kv_seqlen_list, dtype=torch.int32).npu()
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
    if kv_seqlen > 0 and window_size_left_golden >= kv_seqlen - 1:
        window_size_left_golden = -1
    if q_seqlen > 0 and window_size_right_golden >= q_seqlen - 1:
        window_size_right_golden = -1
    if is_causal:
        window_size_right_golden = 0
    is_causal_golden = (window_size_left_golden < 0 and window_size_right_golden == 0)
    is_local_golden = (window_size_left_golden >= 0 or window_size_right_golden > 0) and not is_causal_golden
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
    cache_seqlens_for_api = new_kv_seqlen_list if (layout == "TND" and cache_mode == 0) else kv_seqlen_list
    out_out, softmax_lse, *rest = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        None,
        None,
        None,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=cache_seqlens_for_api,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=leftpad_k,
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
        softcap=0.0,
        rotary_interleaved=is_rotary_interleaved,
        scheduler_metadata=None,
        num_splits=num_splits,
        pack_gqa=None,
        sm_margin=0,
        return_softmax_lse=True
    )

    def create_binary_matrix(qSeqlen, kvSeqlen, preToken, nextToken):
        preToken = kvSeqlen - qSeqlen - preToken
        nextToken = kvSeqlen - qSeqlen + nextToken
        matrix = [[0 for _ in range(kvSeqlen)] for _ in range(qSeqlen)]
        for i in range(qSeqlen):
            for j in range(kvSeqlen):
                is_below_pretoken_line = (-i + j) < preToken
                is_above_nexttoken_line = (-i + j) > nextToken
                if is_below_pretoken_line or is_above_nexttoken_line:
                    matrix[i][j] = 1
        return torch.tensor(matrix, dtype=torch.bool)

    golden_out = None
    golden_out = None
    if layout == "BSND":
        golden_out = torch.empty((batch_size, q_seqlen, num_heads, head_size), dtype=data_type)
        golden_lseL = torch.empty((batch_size, num_heads, q_seqlen), dtype=torch.float32)
    else:
        golden_out = torch.empty((t_q_sum, num_heads, head_size), dtype=data_type)
        golden_lseL = torch.empty((num_heads, t_q_sum), dtype=torch.float32)
    for i in range(batch_size):
        q_seqlen_per_batch = q_sequences[i]
        kv_seqlen_per_batch = kv_sequences[i]
        key_cache_per_batch = None
        value_cache_per_batch = None
        query_cpu_per_batch = None
        atten_mask = None
        if is_causal_golden:
            atten_mask = torch.triu(
                torch.ones(q_seqlen_per_batch, kv_seqlen_per_batch),
                diagonal=(kv_seqlen_per_batch - q_seqlen_per_batch + 1),
            ).bool()
        elif is_local_golden:
            atten_mask = create_binary_matrix(q_seqlen_per_batch, kv_seqlen_per_batch, window_size_left_golden, window_size_right_golden)
        if layout == "BSND":
            query_cpu_per_batch = query.detach().cpu()[i]
            if cache_mode == 1:
                keys = []
                values = []
                block_table = block_tables.cpu()[i]
                for j in range(kv_seqlen_per_batch):
                    block_number = int(block_table[j // block_size])
                    block_offset = j % block_size
                    k = key_cache.detach().cpu()[block_number, block_offset, :, :]
                    k = k.reshape(kv_heads, head_size)
                    keys.append(k)
                    v = value_cache.detach().cpu()[block_number, block_offset, :, :]
                    v = v.reshape(kv_heads, head_size)
                    values.append(v)
                key_cache_per_batch = torch.stack(keys, dim=0)
                value_cache_per_batch = torch.stack(values, dim=0)
            else:
                key_cache_per_batch = key_cache.detach().cpu()[i]
                value_cache_per_batch = value_cache.detach().cpu()[i]
        else:
            query_cpu_per_batch = query.detach().cpu()[new_q_seqlen_list_cpu[i] : new_q_seqlen_list_cpu[i + 1]]
            if cache_mode == 0:
                key_cache_per_batch = key_cache.detach().cpu()[new_kv_seqlen_list_cpu[i] : new_kv_seqlen_list_cpu[i + 1]]
                value_cache_per_batch = value_cache.detach().cpu()[new_kv_seqlen_list_cpu[i] : new_kv_seqlen_list_cpu[i + 1]]
            else:
                keys = []
                values = []
                block_table = block_tables.cpu()[i]
                for j in range(kv_seqlen_per_batch):
                    block_number = int(block_table[j // block_size])
                    block_offset = j % block_size
                    k = key_cache.detach().cpu()[block_number, block_offset, :, :]
                    k = k.reshape(kv_heads, head_size)
                    keys.append(k)
                    v = value_cache.detach().cpu()[block_number, block_offset, :, :]
                    v = v.reshape(kv_heads, head_size)
                    values.append(v)
                key_cache_per_batch = torch.stack(keys, dim=0)
                value_cache_per_batch = torch.stack(values, dim=0)
        if atten_mask is not None:
            output, golden_lse = ref_flash_attention(query_cpu_per_batch, key_cache_per_batch, value_cache_per_batch, scale, atten_mask, data_type)
        else:
            output, golden_lse = ref_flash_attention(query_cpu_per_batch, key_cache_per_batch, value_cache_per_batch, scale, None, data_type)
        out = output.reshape(q_seqlen_per_batch, num_heads, head_size)
        if is_local_golden:
            preTokens = window_size_left_golden
            nextTokens = window_size_right_golden
            preTokensChange = preTokens - kv_seqlen_per_batch + q_seqlen_per_batch
            nextTokensChange = nextTokens + kv_seqlen_per_batch - q_seqlen_per_batch
            nextTokensError = -nextTokensChange if nextTokensChange < 0 else 0
            preTokensError = (
                q_seqlen_per_batch - kv_seqlen_per_batch - preTokensChange
            ) if q_seqlen_per_batch > kv_seqlen_per_batch + preTokensChange else 0
            actualSeq = q_seqlen_per_batch
            actualSeq -= nextTokensError
            actualSeq -= preTokensError
            if actualSeq != q_seqlen_per_batch:
                if nextTokensError != 0:
                    actualSeq = q_seqlen_per_batch - actualSeq
                    out[ :actualSeq, :, :] = 0
                    golden_lse[:, :actualSeq] = torch.inf
                elif preTokensError != 0:
                    out[actualSeq:, :, :] = 0
                    golden_lse[:, actualSeq:] = torch.inf
        if layout == "BSND":
            golden_out[i:i+1] = out
            golden_lseL[i:i+1] = golden_lse.reshape(1, num_heads, q_seqlen_per_batch)
        else:
            golden_out[new_q_seqlen_list[i] : new_q_seqlen_list[i + 1]] = out
            golden_lseL[:, new_q_seqlen_list[i] : new_q_seqlen_list[i + 1]] = golden_lse.reshape(num_heads, q_seqlen_per_batch)
    rtol = 1e-2
    atol = 1e-2
    torch.testing.assert_close(out_out.cpu(), golden_out.cpu(), rtol=rtol, atol=atol)
    if "Ascend910" in name:
        torch.testing.assert_close(softmax_lse.cpu(), golden_lseL.cpu(), rtol=rtol, atol=atol)

test_cases = [
    # (data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, return_attn_probs, is_causal)
    (torch.float16, 1, 1, 1, 1024, 1024, 128, True, False),
    (torch.float16, 5, 4, 4, 1024, 1024, 128, True, True),
    (torch.float16, 7, 1, 1, 512, 512, 128, True, False),
    (torch.float16, 1, 1, 1, 1024, 1024, 128, False, False),
    (torch.float16, 5, 4, 4, 1024, 1024, 128, False, True),
    (torch.float16, 7, 1, 1, 512, 512, 128, False, False),
    (torch.float16, 4, 2, 1, 513, 513, 128, False, False),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, False),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, True, True),
    (torch.bfloat16, 7, 1, 1, 512, 512, 128, True, False),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, False, False),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, False, True),
    (torch.bfloat16, 7, 1, 1, 512, 512, 128, False, False),
    (torch.float16, 4, 2, 1, 513, 513, 128, False, False),
]
@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, return_attn_probs, is_causal", test_cases)
def test_fa_fwd_custom_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, return_attn_probs, is_causal):
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if "Ascend910" not in name:
        pytest.skip("flash_attn_func only support Ascend910")
    q_min_range = -5.0
    q_max_range = 5.0
    kv_min_range = -5.0
    kv_max_range = 5.0
    query = (q_min_range + (q_max_range - q_min_range) * torch.rand(batch_size, q_seqlen, num_heads, head_size)).to(data_type).npu()
    key_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size, kv_seqlen, kv_heads, head_size)).to(data_type).npu()
    value_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size, kv_seqlen, kv_heads, head_size)).to(data_type).npu()
    scale = 1.0 / (head_size ** 0.5)
    window_size_left = -1
    window_size_right = -1
    softcap = 0.0

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

    golden_out = torch.empty((batch_size, q_seqlen, num_heads, head_size), dtype=data_type)
    golden_lseL = torch.empty((batch_size, num_heads, q_seqlen), dtype=torch.float32)
    atten_mask = None
    if is_causal:
        atten_mask = torch.triu(torch.ones(q_seqlen, kv_seqlen), diagonal=1).bool()
    for i in range(batch_size):
        key_cache_per_batch = key_cache.detach().cpu()[i]
        value_cache_per_batch = value_cache.detach().cpu()[i]
        query_cpu = query.detach().cpu()[i]
        if is_causal:
            output, golden_lse = ref_flash_attention(query_cpu, key_cache_per_batch, value_cache_per_batch, scale, atten_mask, data_type)
        else:
            output, golden_lse = ref_flash_attention(query_cpu, key_cache_per_batch, value_cache_per_batch, scale, None, data_type)
        out = output.reshape(q_seqlen, num_heads, head_size)
        golden_out[i:i+1] = out
        golden_lseL[i:i+1] = golden_lse.reshape(num_heads, q_seqlen)
    rtol = 1e-2
    atol = 1e-2
    torch.testing.assert_close(out_out.cpu(), golden_out.cpu(), rtol=rtol, atol=atol)
    if return_attn_probs:
        torch.testing.assert_close(softmax_lse.cpu(), golden_lseL.cpu(), rtol=rtol, atol=atol)


test_cases = [
    # (data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal)
    (torch.bfloat16, 1, 1, 1, 512, 1024, 128, True),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, False),
    (torch.float16, 7, 5, 1, 512, 512, 128, True),
    (torch.float16, 7, 5, 1, 777, 888, 192, False),
    (torch.float16, 7, 5, 1, 1777, 1888, 256, True),
    (torch.bfloat16, 1, 1, 1, 7777, 8192, 64, True),
    (torch.bfloat16, 7, 5, 1, 711, 8192, 111, True),
]

@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal", test_cases)
def test_fa_varlen_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal):
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if "Ascend910" not in name:
        pytest.skip("flash_attn_varlen_func only support Ascend910")
    q_min_range = -5.0
    q_max_range = 5.0
    kv_min_range = -5.0
    kv_max_range = 5.0
    query = (q_min_range + (q_max_range - q_min_range) * torch.rand(batch_size * q_seqlen, num_heads, head_size)).to(data_type).npu()
    key = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size * kv_seqlen, kv_heads, head_size)).to(data_type).npu()
    value = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size * kv_seqlen, kv_heads, head_size)).to(data_type).npu()
    actual_seq_len = torch.tensor([q_seqlen * i for i in range(batch_size + 1)], dtype=torch.int32).npu()
    actual_kv_len = torch.tensor([kv_seqlen * i for i in range(batch_size + 1)], dtype=torch.int32).npu()

    max_seqlen_q = q_seqlen
    max_seqlen_k = kv_seqlen
    scale = 1.0 / (head_size ** 0.5)
    window_size_left = -1
    window_size_right = -1
    softcap = 0.0

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
    golden_out = torch.empty((batch_size * q_seqlen, num_heads, head_size), dtype=data_type)
    golden_lseL = torch.empty((num_heads, batch_size * q_seqlen), dtype=torch.float32)
    atten_mask = None
    if is_causal:
        atten_mask = (torch.triu(torch.ones(q_seqlen, kv_seqlen), diagonal=kv_seqlen - q_seqlen + 1)).to(torch.bool)
    for i in range(1, batch_size + 1):
        key_per_batch = key.detach().cpu()[(i - 1) * kv_seqlen : i * kv_seqlen]
        value_per_batch = value.detach().cpu()[(i - 1) * kv_seqlen : i * kv_seqlen]
        query_cpu = query.detach().cpu()[(i - 1) * q_seqlen : i * q_seqlen]
        if is_causal:
            output, golden_lse = ref_flash_attention(query_cpu, key_per_batch, value_per_batch, scale, atten_mask, data_type)
        else:
            output, golden_lse = ref_flash_attention(query_cpu, key_per_batch, value_per_batch, scale, None, data_type)
        out = output.reshape(q_seqlen, num_heads, head_size)
        golden_out[(i - 1) * q_seqlen : i * q_seqlen] = out
        golden_lseL[:, (i - 1) * q_seqlen : i * q_seqlen] = golden_lse.reshape(num_heads, q_seqlen)
    rtol = 1e-2
    atol = 1e-2
    torch.testing.assert_close(output_npu.cpu(), golden_out.cpu(), rtol=rtol, atol=atol)
    torch.testing.assert_close(softmax_lse.cpu(), golden_lseL.cpu(), rtol=rtol, atol=atol)