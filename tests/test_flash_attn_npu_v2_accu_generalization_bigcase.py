# Copyright (c) 2026, Minghua Shen.
"""移植自 ATK 用例: flash-attention-npu_test/FlashAttnWithKVCache/FlashAttnWithKVCache_Accu_Generalization.json

每个用例的字段(与 test_fa_custom_ops 风格一致):
    data_type, batch_size, q_seqlen, num_heads, kv_heads, kv_seqlen,
    head_size, cache_mode, num_blocks_per_seq, is_causal,
    window_size_left, window_size_right, return_softmax_lse, range_mode, range_a, range_b

用例分类: 大用例 = batch_size / q_seqlen / num_heads / kv_heads / kv_seqlen 任一 >= 4096
说明:
- cache_mode: 0=非分页, 1=分页(块大小128)
- kv_seqlen = key_cache.shape[1] (与 function_FlashAttnWithKVCache.py 一致);
  cache_mode=1 时该维度为 block_size=128, block_table 按连续块重建;
  cache_mode=0 时 num_blocks_per_seq=0, block_table=None
- window_size_left/right 在 [-1,1000] 内按 case id 随机采样(确定性), -1 表示无限窗口
- range_mode: uniform=[range_a,range_b] 均匀采样; nd=高斯 range_a=mean, range_b=std
- softmax_scale=0 表示自动取 1/sqrt(head_size); softcap=0.0; num_splits=0;
  rotary_interleaved=False; k/v/rotary_cos/rotary_sin/cache_batch_idx/cache_leftpad/alibi_slopes=None
- 精度比对使用两种方式:
  1) torch.testing.assert_close(rtol=1e-2, atol=1e-2) 逐元素容差
  2) ATK cv_fused_double_benchmark 相对误差比(CANN cal_relative_diff 鲁棒形式):
     diff < 0.005 按绝对误差, 否则 diff/max(|a|,|g|,floor); max<=5%, avg<=1.5%, rms<=1.5%
"""

import torch
import torch_npu
import pytest

from flash_attn_npu import flash_attn_with_kvcache

def group_matmul(head, kv_head, left, right, high_prec=1):
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


def softmax1(qk_result, is_first, gm, interm_dtype=torch.float16):
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


def qkMM1(query, key):
    result = None
    qk_k = key.shape[1]
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


def pvMM2(p, value):
    result = None
    pv_k = value.shape[1]
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


def ref_flash_attention(query, key, value, scale, mask, data_type):
    interm_dtype = torch.float32
    query = query.permute(1, 0, 2)
    key = key.permute(1, 2, 0)
    value = value.permute(1, 0, 2)
    scale = torch.tensor(scale).to(torch.float32)
    context_len = key.shape[2]
    context_size = 512
    gm = None
    if mask is not None:
        mask = mask.cpu()
    for kv_start in range(0, context_len, context_size):
        sub_len = context_size
        if kv_start + context_size > context_len:
            sub_len = context_len - kv_start
        sub_key = key[:, :, kv_start: kv_start + sub_len]
        sub_mask = None
        if mask is not None:
            sub_mask = mask[:query.shape[1], kv_start: kv_start + sub_len].to(interm_dtype) * (-1e4)
        sub_value = value[:, kv_start: kv_start + sub_len, :]
        qk_result = qkMM1(query, sub_key).to(interm_dtype)
        qk_result = qk_result * scale
        if mask is not None:
            qk_result += sub_mask
        p_result, row_sum, dm, gm = softmax1(qk_result, kv_start == 0, gm, interm_dtype)
        p_result = p_result.to(data_type)
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


def create_binary_matrix(q_seqlen, kv_seqlen, window_size_left, window_size_right):
    pre_token = kv_seqlen - q_seqlen - window_size_left
    next_token = kv_seqlen - q_seqlen + window_size_right
    matrix = [[0 for _ in range(kv_seqlen)] for _ in range(q_seqlen)]
    for i in range(q_seqlen):
        for j in range(kv_seqlen):
            is_below_pretoken_line = (-i + j) < pre_token
            is_above_nexttoken_line = (-i + j) > next_token
            if is_below_pretoken_line or is_above_nexttoken_line:
                matrix[i][j] = 1
    return torch.tensor(matrix, dtype=torch.bool)


def gen_tensor(range_mode, range_a, range_b, shape, data_type):
    if range_mode == "uniform":
        data = range_a + (range_b - range_a) * torch.rand(shape)
    else:
        data = range_a + range_b * torch.randn(shape)
    return data.to(data_type).npu()


def compare_cv_fused_double_benchmark(golden, actual, max_re_ratio=5.0, avg_re_ratio=1.5, root_mean_squared_ratio=1.5, diff_thd=0.005):
    """移植 ATK 的 cv_fused_double_benchmark 精度比对 (对应 YAML standard.acc)。

    逐元素相对误差采用 CANN 标准(与 tests/precision_compare.py 的 cal_relative_diff 一致),
    对接近 0 的 golden 值不敏感(attention 输出大量元素天然接近 0):
        diff = |actual - golden|
        ratio = diff                               若 diff < diff_thd (绝对误差容差, 视为足够接近)
              = diff / max(|actual|, |golden|, b2) 否则 (b2 为下限 floor)
    再对 ratio 取 max/avg/rms 并转为百分比, 判决条件(与 standard 一致, 单位 %):
        max(ratio) <= max_re_ratio               (默认 5.0%)
        mean(ratio) <= avg_re_ratio              (默认 1.5%)
        rms(ratio) <= root_mean_squared_ratio    (默认 1.5%)
    golden 为 inf 的位置要求 actual 也为相同的 inf(用于 local window 的 padding 行)。
    返回 (是否通过, 统计dict)。
    """
    golden = golden.detach().to(torch.float32)
    actual = actual.detach().to(torch.float32)

    inf_mask = torch.isinf(golden)
    if inf_mask.any():
        if not (actual[inf_mask] == golden[inf_mask]).all():
            return False, {"max_re": float("inf"), "avg_re": float("inf"), "rms_re": float("inf")}
        golden = golden[~inf_mask]
        actual = actual[~inf_mask]

    diff = torch.abs(actual - golden)
    b1 = torch.maximum(torch.abs(actual), torch.abs(golden))
    b2 = (1.0 / (1 << 14)) / diff_thd
    b = torch.maximum(b1, torch.tensor(b2, dtype=torch.float32)) + 1e-10
    ratio = torch.where(diff < diff_thd, diff, diff / b)
    max_re = ratio.max().item() * 100.0
    avg_re = ratio.mean().item() * 100.0
    rms_re = torch.sqrt(torch.mean(ratio ** 2)).item() * 100.0
    stats = {"max_re": max_re, "avg_re": avg_re, "rms_re": rms_re}
    ok = (max_re <= max_re_ratio) and (avg_re <= avg_re_ratio) and (rms_re <= root_mean_squared_ratio)
    return ok, stats

test_cases = [
    # (data_type, batch_size, q_seqlen, num_heads, kv_heads, kv_seqlen, head_size, cache_mode, num_blocks_per_seq, is_causal, window_size_left, window_size_right, return_softmax_lse, range_mode, range_a, range_b)
    (torch.bfloat16, 256, 1, 8192, 4, 128, 1, 1, 1, False, 692, 43, True, "uniform", -1.0, 1.0),
    (torch.float16, 32, 4096, 4, 4, 1, 16, 0, 0, True, 167, 427, False, "uniform", -1.0, 1.0),
    (torch.bfloat16, 4, 2048, 4, 2, 131072, 16, 0, 0, True, 728, 391, True, "uniform", -1.0, 1.0),
    (torch.bfloat16, 4, 346, 128, 64, 4096, 1, 0, 0, True, 468, 592, True, "uniform", 0.0, 1.0),
    (torch.bfloat16, 511, 248, 4, 1, 4096, 1, 0, 0, False, 277, 426, True, "uniform", 0.0, 1.0),
    (torch.float16, 16, 4096, 2, 2, 2, 128, 0, 0, False, 67, 351, True, "uniform", 0.0, 1.0),
    (torch.bfloat16, 64, 2, 131072, 1, 128, 2, 1, 2, True, 248, 857, False, "uniform", 0.0, 1.0),
    (torch.bfloat16, 1, 511, 32, 32, 4096, 172, 0, 0, True, 630, 872, True, "uniform", 0.0, 1.0),
    (torch.bfloat16, 1, 4096, 2, 1, 2, 8, 0, 0, False, 314, 289, False, "uniform", -0.01, 0.01),
    (torch.bfloat16, 2, 131072, 1, 1, 8, 1, 0, 0, False, 423, 294, True, "uniform", -0.01, 0.01),
    (torch.float16, 1, 1024, 8192, 4, 16, 2, 0, 0, True, 147, 447, False, "uniform", -0.01, 0.01),
    (torch.bfloat16, 2, 8, 8192, 4, 1024, 128, 0, 0, False, 277, 402, True, "uniform", -0.01, 0.01),
    (torch.float16, 8192, 4, 2, 1, 128, 16, 1, 1, True, 147, 786, False, "uniform", -0.01, 0.01),
    (torch.bfloat16, 64, 2, 512, 1, 4096, 8, 0, 0, True, 981, 19, True, "uniform", -0.001, 0.001),
    (torch.float16, 2048, 512, 1, 1, 131072, 1, 0, 0, True, 846, 970, False, "uniform", -0.001, 0.001),
    (torch.bfloat16, 2, 16, 8192, 4, 16, 8, 0, 0, True, 585, 60, True, "uniform", -1.0, 1.0),
    (torch.float16, 1, 32, 128, 1, 131072, 2, 0, 0, False, 842, 333, True, "uniform", 0.0, 1.0),
    (torch.float16, 2, 128, 1024, 4, 4096, 4, 0, 0, False, 593, 513, True, "uniform", 0.0, 1.0),
    (torch.bfloat16, 1, 16, 4, 1, 4096, 192, 0, 0, True, 498, 426, False, "uniform", 0.0, 1.0),
    (torch.bfloat16, 577, 2, 4096, 1, 512, 8, 0, 0, True, 182, 646, False, "uniform", 0.0, 1.0),
    (torch.float16, 4, 4096, 203, 1, 4, 2, 0, 0, True, 807, 722, False, "uniform", -0.01, 0.01),
    (torch.bfloat16, 8, 512, 511, 1, 131072, 2, 0, 0, True, 494, 838, True, "uniform", -0.01, 0.01),
    (torch.float16, 2, 4, 32, 1, 8192, 32, 0, 0, False, 517, 673, False, "uniform", -0.001, 0.001),
    (torch.float16, 1, 512, 128, 64, 4096, 2, 0, 0, True, 569, 671, True, "uniform", -0.001, 0.001),
    (torch.bfloat16, 1, 4, 8192, 4, 4, 128, 0, 0, False, 528, 745, False, "uniform", -0.001, 0.001),
    (torch.bfloat16, 1, 4, 4, 4, 8192, 128, 0, 0, True, 44, 645, False, "uniform", -0.001, 0.001),
    (torch.float16, 2, 1, 128, 16, 4096, 32, 0, 0, False, 878, 211, False, "uniform", -0.001, 0.001),
    (torch.float16, 64, 128, 4, 2, 4096, 8, 0, 0, False, 306, 190, False, "uniform", -0.001, 0.001),
    (torch.float16, 16, 4096, 8, 4, 128, 2, 1, 1, True, 354, 741, False, "nd", 0.0, 1.0),
    (torch.bfloat16, 2, 4, 131072, 2, 1024, 4, 0, 0, True, 883, 630, False, "nd", 0.0, 1.0),
    (torch.bfloat16, 4096, 8, 1, 1, 128, 32, 1, 1, True, 855, 424, False, "nd", 0.0, 1.0),
    (torch.bfloat16, 511, 512, 8, 8, 4096, 4, 0, 0, True, 395, 410, False, "nd", 0.0, 1.0),
    (torch.bfloat16, 1, 8192, 1, 1, 128, 1, 1, 64, True, 55, 331, False, "nd", 1.0, 1.0),
    (torch.float16, 4, 1, 256, 1, 4096, 4, 0, 0, False, 323, 420, False, "nd", 1.0, 1.0),
    (torch.float16, 1, 1024, 16, 1, 8192, 16, 0, 0, False, 442, 30, True, "nd", 1.0, 1.0),
    (torch.bfloat16, 1, 4, 8192, 4, 2048, 128, 0, 0, True, 39, 25, False, "nd", 1.0, 1.0),
    (torch.float16, 8192, 1, 139, 1, 128, 4, 0, 0, True, 487, 366, True, "nd", 1.0, 1.0),
    (torch.float16, 8192, 64, 32, 1, 32, 2, 0, 0, True, 146, 468, False, "nd", 1.0, 1.0),
    (torch.bfloat16, 1, 128, 4, 4, 131072, 128, 0, 0, False, 922, 90, True, "nd", 1.0, 1.0),
    (torch.bfloat16, 64, 4, 511, 1, 131072, 8, 0, 0, True, 927, 299, False, "nd", 1.0, 1.0),
    (torch.float16, 4096, 256, 8, 8, 513, 4, 0, 0, True, 419, 362, True, "nd", 1.0, 1.0),
    (torch.float16, 32, 4096, 2, 2, 2, 4, 0, 0, True, 673, 483, True, "nd", 0.0, 0.001),
    (torch.float16, 64, 2, 2, 1, 8192, 8, 0, 0, False, 407, 640, True, "nd", 0.0, 0.001),
    (torch.bfloat16, 8, 4096, 512, 1, 131072, 4, 0, 0, False, 439, 937, False, "nd", 0.0, 0.001),
    (torch.float16, 1, 8, 16, 2, 8192, 32, 0, 0, False, 418, 707, True, "nd", 0.0, 1.0),
    (torch.float16, 1, 511, 4096, 1, 32, 2, 0, 0, False, 281, 399, False, "nd", 0.0, 1.0),
    (torch.bfloat16, 16, 32, 4096, 32, 1, 16, 0, 0, False, 848, 486, True, "nd", 0.0, 1.0),
    (torch.float16, 4096, 128, 8, 8, 128, 8, 1, 1, True, 854, 748, False, "nd", 0.0, 1.0),
    (torch.bfloat16, 4096, 4, 16, 8, 128, 16, 0, 0, False, 945, 251, True, "nd", 0.0, 1.0),
    (torch.bfloat16, 8192, 8, 1, 1, 128, 4, 1, 32, True, 147, -1, True, "nd", 0.0, 1.0),
    (torch.bfloat16, 2, 2, 64, 4, 4096, 16, 0, 0, False, 418, 513, True, "nd", 1.0, 1.0),
    (torch.float16, 256, 1, 4096, 64, 128, 4, 1, 4, True, 784, 990, False, "nd", 1.0, 1.0),
    (torch.float16, 8, 4, 4096, 16, 1, 1, 0, 0, False, 126, 640, False, "nd", 0.0, 0.001),
    (torch.bfloat16, 1, 16, 8192, 4, 511, 2, 0, 0, False, 317, 612, True, "nd", 0.0, 0.001),
    (torch.bfloat16, 2, 128, 64, 2, 8192, 4, 0, 0, True, 33, 118, True, "nd", 0.0, 0.001),
    (torch.float16, 32, 4, 4096, 32, 4, 1, 0, 0, False, 238, 372, False, "nd", 0.0, 0.001),
    (torch.float16, 27, 4096, 4, 4, 128, 8, 1, 1, True, 527, 882, False, "nd", 0.0, 0.001),
    (torch.bfloat16, 511, 8, 16, 2, 8192, 4, 0, 0, True, 683, 652, True, "nd", 0.0, 0.001),
    (torch.float16, 32, 8, 8, 2, 4096, 128, 0, 0, False, 579, 598, True, "nd", 0.0, 0.001),
    (torch.bfloat16, 256, 32, 4096, 64, 511, 2, 0, 0, False, 503, 172, True, "nd", 0.0, 0.001),
    (torch.bfloat16, 1, 1, 4096, 1, 64, 1, 0, 0, False, 254, 638, True, "uniform", -5.0, 5.0),
    (torch.float16, 2, 4096, 1, 1, 16, 2, 0, 0, False, 181, 941, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4, 8192, 1, 1, 128, 1, 1, 1, False, 921, 753, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4, 2, 4096, 2, 128, 1, 1, 1, False, 917, 257, True, "uniform", -5.0, 5.0),
    (torch.float16, 4, 4, 4, 2, 4096, 1, 0, 0, False, 845, 557, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 16, 4096, 1, 1, 2, 1, 0, 0, False, 156, 217, True, "uniform", -5.0, 5.0),
    (torch.float16, 4, 32, 1, 1, 8192, 1, 0, 0, True, 1000, 951, False, "uniform", -5.0, 5.0),
    (torch.float16, 1, 8192, 2, 1, 128, 4, 1, 1, False, 143, 884, False, "uniform", -5.0, 5.0),
    (torch.float16, 4096, 8, 1, 1, 4, 1, 0, 0, True, 491, 114, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 1, 512, 16, 8, 4096, 1, 0, 0, True, 858, 482, True, "uniform", -5.0, 5.0),
    (torch.float16, 4, 8, 513, 1, 4096, 2, 0, 0, True, 311, 764, True, "uniform", -5.0, 5.0),
    (torch.float16, 4096, 1, 1, 1, 1, 8, 0, 0, False, 535, 528, True, "uniform", -5.0, 5.0),
    (torch.float16, 2, 1, 8192, 1, 2, 8, 0, 0, False, 912, 724, False, "uniform", -5.0, 5.0),
    (torch.float16, 2, 1, 196, 4, 8192, 1, 0, 0, True, 122, 345, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 2, 8, 4096, 4, 128, 2, 1, 1, True, 723, 33, False, "uniform", -5.0, 5.0),
    (torch.float16, 2, 4096, 1, 1, 128, 16, 0, 0, False, 409, 188, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 8, 513, 32, 1, 8192, 1, 0, 0, False, 496, 323, False, "uniform", -5.0, 5.0),
    (torch.float16, 1, 4096, 2, 1, 128, 32, 1, 8, True, 17, 107, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4, 8192, 2, 2, 128, 2, 1, 64, False, 338, 745, True, "uniform", -5.0, 5.0),
    (torch.float16, 1, 8, 8192, 2, 256, 8, 0, 0, False, 895, 745, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4, 8, 4096, 2, 128, 4, 0, 0, True, 86, 508, False, "uniform", -5.0, 5.0),
    (torch.float16, 1, 2, 512, 2, 4096, 32, 0, 0, True, 871, 166, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 8, 1, 8192, 16, 16, 8, 0, 0, False, 424, 990, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 8, 4, 8192, 64, 128, 1, 1, 5, False, 686, 507, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 1, 131072, 8, 4, 128, 1, 1, 2, False, 31, 405, True, "uniform", -5.0, 5.0),
    (torch.float16, 2, 2, 513, 1, 131072, 2, 0, 0, False, 916, 12, False, "uniform", -5.0, 5.0),
    (torch.float16, 1, 32, 4096, 8, 128, 8, 1, 1, True, 940, 115, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4, 256, 1, 1, 4096, 32, 0, 0, True, 107, 703, True, "uniform", -5.0, 5.0),
    (torch.float16, 1, 512, 1024, 2, 8192, 2, 0, 0, False, 896, 67, True, "uniform", -5.0, 5.0),
    (torch.float16, 1, 256, 4, 1, 4096, 128, 0, 0, True, 592, 411, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 1, 513, 32, 16, 4096, 8, 0, 0, True, 540, 525, True, "uniform", -5.0, 5.0),
    (torch.float16, 32, 256, 87, 1, 8192, 1, 0, 0, True, 577, 763, True, "uniform", -5.0, 5.0),
    (torch.float16, 2, 1, 4096, 4, 128, 128, 1, 2, False, 514, 266, True, "uniform", -5.0, 5.0),
    (torch.float16, 4, 20, 4096, 8, 128, 4, 1, 16, False, 452, 504, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 2, 4, 4096, 16, 511, 32, 0, 0, False, 42, 687, False, "uniform", -5.0, 5.0),
    (torch.float16, 1, 128, 1024, 8, 8192, 8, 0, 0, True, 0, 274, False, "uniform", -5.0, 5.0),
    (torch.float16, 8, 1024, 16, 1, 8192, 8, 0, 0, False, 661, 34, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 8, 32, 1, 1, 131072, 1, 0, 0, True, 637, 627, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 8, 4096, 2, 1, 128, 16, 1, 32, False, 467, 398, False, "uniform", -5.0, 5.0),
    (torch.float16, 1, 4096, 64, 1, 128, 8, 1, 1, True, 700, 597, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 8192, 16, 4, 1, 32, 2, 0, 0, False, 692, 531, False, "uniform", -5.0, 5.0),
    (torch.float16, 1, 8192, 16, 4, 128, 16, 1, 1, True, 915, 94, True, "uniform", -5.0, 5.0),
    (torch.float16, 32, 64, 2, 1, 8192, 4, 0, 0, True, 660, 429, True, "uniform", -5.0, 5.0),
    (torch.float16, 32, 8192, 4, 1, 128, 2, 1, 2, True, 973, 531, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4096, 2, 2, 1, 128, 2, 1, 1, False, 528, 838, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 2, 4096, 8, 4, 128, 32, 1, 1, True, 841, 701, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 32, 4096, 4, 2, 128, 4, 1, 1, True, 366, 968, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 8192, 4, 1, 1, 64, 2, 0, 0, True, 444, 344, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 2, 64, 512, 32, 8192, 2, 0, 0, True, 202, 707, True, "uniform", -5.0, 5.0),
    (torch.float16, 2, 4096, 1, 1, 128, 256, 1, 1, False, 533, 360, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 1, 513, 16, 8, 8192, 16, 0, 0, False, 116, 689, True, "uniform", -5.0, 5.0),
    (torch.float16, 1, 131072, 8, 8, 8192, 2, 0, 0, True, 472, 184, True, "uniform", -5.0, 5.0),
    (torch.float16, 4, 16, 4096, 4, 1024, 8, 0, 0, False, 485, 192, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4096, 1, 32, 8, 8, 4, 0, 0, False, 478, 189, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 128, 2, 4096, 4, 513, 2, 0, 0, True, 185, 838, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4096, 4, 16, 16, 1, 8, 0, 0, True, 936, 453, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4096, 8, 16, 1, 32, 4, 0, 0, True, 475, 457, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 64, 16, 458, 1, 4096, 4, 0, 0, True, 374, 258, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 1, 1, 8192, 32, 128, 256, 1, 1, True, 372, 746, False, "uniform", -5.0, 5.0),
    (torch.float16, 64, 2, 8192, 64, 128, 2, 0, 0, True, 402, 656, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 2, 4, 1, 1, 131072, 8, 0, 0, False, 34, 339, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 8, 128, 4096, 8, 1, 1, 0, 0, True, 542, 519, True, "uniform", -5.0, 5.0),
    (torch.float16, 1, 512, 8192, 1, 128, 1, 1, 1, True, 134, 421, True, "uniform", -5.0, 5.0),
    (torch.float16, 2, 8, 131072, 4, 128, 1, 1, 1024, True, 815, 18, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4, 16, 8192, 8, 8, 8, 0, 0, True, 19, 118, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 1, 4096, 256, 1, 4096, 4, 0, 0, False, 452, 163, True, "uniform", -5.0, 5.0),
    (torch.float16, 64, 64, 2, 2, 4096, 4, 0, 0, False, 254, 148, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 1, 32, 256, 1, 131072, 16, 0, 0, False, 328, 908, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 2, 511, 2048, 8, 8192, 2, 0, 0, True, 354, 736, False, "uniform", -5.0, 5.0),
    (torch.float16, 8192, 8, 2, 1, 64, 4, 0, 0, True, 548, 753, True, "uniform", -5.0, 5.0),
    (torch.float16, 8192, 1, 671, 1, 2, 1, 0, 0, True, 747, 374, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 2, 2048, 64, 32, 4096, 8, 0, 0, False, 249, 775, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 32, 2, 4096, 4, 128, 8, 1, 16, False, 221, 201, True, "uniform", -5.0, 5.0),
    (torch.float16, 1, 1, 8, 4, 4096, 192, 0, 0, False, 27, 61, False, "uniform", -5.0, 5.0),
    (torch.float16, 332, 8192, 1, 1, 128, 2, 1, 8, True, 609, 719, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 511, 4, 2, 2, 4096, 1, 0, 0, True, 272, 64, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 16, 128, 4096, 4, 4, 1, 0, 0, True, 554, 376, False, "uniform", -5.0, 5.0),
    (torch.float16, 4, 2, 2, 2, 4096, 128, 0, 0, True, 967, 2, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 128, 16, 2, 2, 8192, 2, 0, 0, False, 556, 467, False, "uniform", -5.0, 5.0),
    (torch.float16, 2, 4096, 513, 1, 1, 2, 0, 0, True, 407, 589, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 256, 8192, 1, 1, 8, 4, 0, 0, True, 294, 790, False, "uniform", -5.0, 5.0),
    (torch.float16, 2, 64, 2, 2, 8192, 128, 0, 0, False, 411, 143, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 2, 4096, 1024, 2, 128, 1, 1, 32, True, 681, 659, True, "uniform", -5.0, 5.0),
    (torch.float16, 4, 131072, 16, 1, 128, 1, 1, 32, False, 521, 882, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4096, 513, 2, 2, 2, 2, 0, 0, False, 583, 234, True, "uniform", -5.0, 5.0),
    (torch.float16, 16, 8192, 64, 32, 128, 1, 1, 1, False, 140, 428, True, "uniform", -5.0, 5.0),
    (torch.float16, 16, 8192, 64, 32, 128, 1, 1, 5, False, 985, 984, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 32, 139, 256, 16, 8192, 1, 0, 0, True, 621, 993, False, "uniform", -5.0, 5.0),
    (torch.float16, 4096, 8, 256, 2, 128, 1, 1, 1, True, 401, 282, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 128, 1, 8192, 16, 511, 4, 0, 0, True, 393, 692, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 128, 8192, 1, 1, 128, 4, 1, 64, True, 380, 763, False, "uniform", -5.0, 5.0),
    (torch.float16, 131072, 1, 32, 16, 2, 1, 0, 0, False, 742, 331, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 2, 511, 8192, 2, 32, 2, 0, 0, False, 422, 934, True, "uniform", -5.0, 5.0),
    (torch.float16, 4, 4, 256, 8, 131072, 2, 0, 0, True, 311, 481, False, "uniform", -5.0, 5.0),
    (torch.float16, 4, 2, 32, 8, 8192, 32, 0, 0, True, 169, 256, False, "uniform", -5.0, 5.0),
    (torch.float16, 8, 4, 131072, 64, 4, 4, 0, 0, False, 323, 626, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 2, 16, 16, 16, 4096, 64, 0, 0, False, 976, 313, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 2, 32, 4, 2, 8192, 256, 0, 0, True, 832, 208, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4, 4096, 513, 1, 128, 2, 1, 16, False, 667, 416, True, "uniform", -5.0, 5.0),
    (torch.float16, 16, 4, 108, 4, 8192, 16, 0, 0, True, 28, 67, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 2, 128, 8192, 4, 2048, 8, 0, 0, False, 543, 125, True, "uniform", -5.0, 5.0),
    (torch.float16, 64, 64, 4096, 16, 128, 1, 1, 1, True, 878, 961, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 511, 32, 32, 2, 8192, 1, 0, 0, False, 970, 761, False, "uniform", -5.0, 5.0),
    (torch.float16, 256, 511, 1, 1, 8192, 4, 0, 0, True, 466, 767, False, "uniform", -5.0, 5.0),
    (torch.float16, 8, 2, 4096, 16, 8192, 8, 0, 0, True, 627, 886, True, "uniform", -5.0, 5.0),
    (torch.float16, 16, 256, 2, 1, 8192, 64, 0, 0, True, 124, 329, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 32, 4096, 2, 1, 128, 2, 1, 1024, True, 320, 307, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 8, 1024, 1, 1, 8192, 128, 0, 0, False, 929, 704, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 128, 32, 4096, 8, 128, 1, 1, 4, True, 25, 503, True, "uniform", -5.0, 5.0),
    (torch.float16, 8192, 4, 511, 1, 128, 1, 1, 1, True, 280, 582, False, "uniform", -5.0, 5.0),
    (torch.float16, 1, 2, 4096, 64, 128, 256, 1, 4, False, 550, 538, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4096, 2, 174, 1, 1024, 2, 0, 0, False, 148, 402, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 128, 1, 131072, 2, 128, 1, 1, 64, True, 396, 369, False, "uniform", -5.0, 5.0),
    (torch.float16, 16, 2, 8192, 4, 128, 32, 1, 32, False, 5, 113, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 6, 4, 131072, 16, 1024, 8, 0, 0, False, 65, 513, False, "uniform", -5.0, 5.0),
    (torch.float16, 511, 4096, 16, 2, 16, 1, 0, 0, True, 755, 375, True, "uniform", -5.0, 5.0),
    (torch.float16, 4, 1, 256, 32, 131072, 1, 0, 0, True, 230, 529, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 16, 4, 4, 1, 131072, 8, 0, 0, True, 886, 674, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 1, 8, 32, 2, 131072, 64, 0, 0, False, 119, 742, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 32, 8, 1, 1, 8192, 64, 0, 0, True, 667, 753, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 8, 131072, 16, 2, 1024, 2, 0, 0, True, 61, 667, True, "uniform", -5.0, 5.0),
    (torch.float16, 131072, 2, 32, 4, 8, 2, 0, 0, False, 909, 755, False, "uniform", -5.0, 5.0),
    (torch.float16, 4, 32, 128, 32, 8192, 16, 0, 0, True, 341, 13, False, "uniform", -5.0, 5.0),
    (torch.float16, 4, 128, 28, 1, 131072, 32, 0, 0, False, 868, 694, True, "uniform", -5.0, 5.0),
    (torch.float16, 4, 2, 4096, 32, 128, 16, 1, 64, False, 308, 763, False, "uniform", -5.0, 5.0),
    (torch.float16, 2, 512, 8, 8, 8192, 128, 0, 0, True, 448, 308, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 32, 1, 4096, 4, 128, 16, 1, 64, True, 803, 813, True, "uniform", -5.0, 5.0),
    (torch.float16, 550, 1, 512, 4, 4096, 2, 0, 0, True, 509, 502, False, "uniform", -5.0, 5.0),
    (torch.float16, 256, 64, 184, 1, 8192, 8, 0, 0, False, 410, 496, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 64, 2, 4, 4, 4096, 32, 0, 0, False, 680, 469, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4, 8192, 256, 64, 8, 8, 0, 0, True, 751, 540, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4, 4096, 513, 1, 128, 8, 1, 1, True, 860, 64, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 16, 32, 4, 1, 8192, 256, 0, 0, True, 305, 716, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4096, 8, 1, 1, 128, 16, 1, 4, False, 987, 398, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 16, 4096, 16, 4, 128, 64, 1, 1, True, 164, 107, False, "uniform", -5.0, 5.0),
    (torch.float16, 64, 511, 32, 2, 131072, 2, 0, 0, False, 12, 701, False, "uniform", -5.0, 5.0),
    (torch.float16, 16, 8192, 16, 16, 131072, 1, 0, 0, True, 968, 482, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 128, 4, 8192, 2, 128, 16, 1, 2, False, 116, 261, False, "uniform", -5.0, 5.0),
    (torch.bfloat16, 8, 1011, 4096, 32, 128, 2, 1, 32, True, 835, 51, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 4096, 2, 64, 1, 128, 16, 1, 4, True, 221, 206, True, "uniform", -5.0, 5.0),
    (torch.float16, 128, 4, 4, 2, 8192, 32, 0, 0, True, 891, 591, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 513, 1, 128, 1, 131072, 1, 0, 0, True, 266, 919, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 2, 256, 4, 1, 131072, 256, 0, 0, False, 384, 771, True, "uniform", -5.0, 5.0),
    (torch.float16, 8192, 1, 8, 8, 128, 8, 1, 1, False, 675, 104, True, "uniform", -5.0, 5.0),
    (torch.bfloat16, 1, 109, 32, 32, 8192, 256, 0, 0, True, 73, 504, True, "uniform", -5.0, 5.0),
    (torch.float16, 4096, 16, 1, 1, 1024, 16, 0, 0, False, 532, 887, True, "uniform", -5.0, 5.0),]

@pytest.mark.parametrize("data_type, batch_size, q_seqlen, num_heads, kv_heads, kv_seqlen, head_size, cache_mode, num_blocks_per_seq, is_causal, window_size_left, window_size_right, return_softmax_lse, range_mode, range_a, range_b",
                         test_cases, ids=[f"case_{i}" for i in range(len(test_cases))])
def test_fa_kvcache_accu_generalization(data_type, batch_size, q_seqlen, num_heads, kv_heads, kv_seqlen, head_size, cache_mode, num_blocks_per_seq, is_causal, window_size_left, window_size_right, return_softmax_lse, range_mode, range_a, range_b):
    block_size = 128
    softmax_scale = head_size ** (-0.5)

    query = gen_tensor(range_mode, range_a, range_b, (batch_size, q_seqlen, num_heads, head_size), data_type)
    if cache_mode == 1:
        num_blocks = num_blocks_per_seq * batch_size
        key_cache = gen_tensor(range_mode, range_a, range_b, (num_blocks, block_size, kv_heads, head_size), data_type)
        value_cache = gen_tensor(range_mode, range_a, range_b, (num_blocks, block_size, kv_heads, head_size), data_type)
        block_tables = torch.tensor(
            [[num_blocks_per_seq * i + j for j in range(num_blocks_per_seq)] for i in range(batch_size)],
            dtype=torch.int32,
        ).npu()
    else:
        key_cache = gen_tensor(range_mode, range_a, range_b, (batch_size, kv_seqlen, kv_heads, head_size), data_type)
        value_cache = gen_tensor(range_mode, range_a, range_b, (batch_size, kv_seqlen, kv_heads, head_size), data_type)
        block_tables = None

    kv_seqlen_list = torch.tensor([kv_seqlen] * batch_size, dtype=torch.int32).npu()

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

    result = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        None,
        None,
        rotary_cos=None,
        rotary_sin=None,
        cache_seqlens=kv_seqlen_list,
        cache_batch_idx=None,
        cache_leftpad=None,
        block_table=block_tables,
        softmax_scale=softmax_scale,
        causal=is_causal,
        window_size=[window_size_left, window_size_right],
        rotary_interleaved=False,
        alibi_slopes=None,
        num_splits=0,
        return_softmax_lse=return_softmax_lse,
    )
    if return_softmax_lse:
        out_out, softmax_lse = result
    else:
        out_out = result

    golden_out = torch.empty((batch_size, q_seqlen, num_heads, head_size), dtype=data_type)
    golden_lseL = torch.empty((batch_size, num_heads, q_seqlen), dtype=torch.float32)
    atten_mask = None
    if is_causal_golden:
        atten_mask = torch.triu(torch.ones(q_seqlen, kv_seqlen), diagonal=kv_seqlen - q_seqlen + 1).bool()
    elif is_local_golden:
        atten_mask = create_binary_matrix(q_seqlen, kv_seqlen, window_size_left_golden, window_size_right_golden)

    for i in range(batch_size):
        key_cache_per_batch = None
        value_cache_per_batch = None
        if cache_mode == 1:
            keys = []
            values = []
            block_table = block_tables.cpu()[i]
            key_cache_cpu = key_cache.detach().cpu()
            value_cache_cpu = value_cache.detach().cpu()
            for j in range(kv_seqlen):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size
                keys.append(key_cache_cpu[block_number, block_offset, :, :].reshape(kv_heads, head_size))
                values.append(value_cache_cpu[block_number, block_offset, :, :].reshape(kv_heads, head_size))
            key_cache_per_batch = torch.stack(keys, dim=0)
            value_cache_per_batch = torch.stack(values, dim=0)
        else:
            key_cache_per_batch = key_cache.detach().cpu()[i]
            value_cache_per_batch = value_cache.detach().cpu()[i]
        query_cpu = query.detach().cpu()[i]
        if is_causal_golden or is_local_golden:
            output, golden_lse = ref_flash_attention(query_cpu, key_cache_per_batch, value_cache_per_batch, softmax_scale, atten_mask, data_type)
        else:
            output, golden_lse = ref_flash_attention(query_cpu, key_cache_per_batch, value_cache_per_batch, softmax_scale, None, data_type)
        out = output.reshape(q_seqlen, num_heads, head_size)
        if is_local_golden:
            pre_tokens_change = window_size_left_golden - kv_seqlen + q_seqlen
            next_tokens_change = window_size_right_golden + kv_seqlen - q_seqlen
            next_tokens_error = -next_tokens_change if next_tokens_change < 0 else 0
            pre_tokens_error = (q_seqlen - kv_seqlen - pre_tokens_change) if q_seqlen > kv_seqlen + pre_tokens_change else 0
            actual_seq = q_seqlen - next_tokens_error - pre_tokens_error
            if actual_seq != q_seqlen:
                if next_tokens_error != 0:
                    out[:q_seqlen - actual_seq, :, :] = 0
                    golden_lse[:, :q_seqlen - actual_seq] = torch.inf
                elif pre_tokens_error != 0:
                    out[actual_seq:, :, :] = 0
                    golden_lse[:, actual_seq:] = torch.inf
        golden_out[i:i+1] = out
        golden_lseL[i:i+1] = golden_lse.reshape(num_heads, q_seqlen)

    rtol = 1e-2
    atol = 1e-2
    torch.testing.assert_close(out_out.cpu(), golden_out.cpu(), rtol=rtol, atol=atol)
    if return_softmax_lse:
        torch.testing.assert_close(softmax_lse.cpu(), golden_lseL.cpu(), rtol=rtol, atol=atol)

    ok_out, stats_out = compare_cv_fused_double_benchmark(golden_out, out_out.cpu())
    print("cv_fused_double_benchmark out: max_re=%.4f%% avg_re=%.4f%% rms_re=%.4f%% ok=%s" % (
        stats_out["max_re"], stats_out["avg_re"], stats_out["rms_re"], ok_out))
    assert ok_out, "cv_fused_double_benchmark out failed: %s" % stats_out
    if return_softmax_lse:
        ok_lse, stats_lse = compare_cv_fused_double_benchmark(golden_lseL, softmax_lse.cpu())
        print("cv_fused_double_benchmark lse: max_re=%.4f%% avg_re=%.4f%% rms_re=%.4f%% ok=%s" % (
            stats_lse["max_re"], stats_lse["avg_re"], stats_lse["rms_re"], ok_lse))
        assert ok_lse, "cv_fused_double_benchmark lse failed: %s" % stats_lse

    torch.npu.synchronize()
