# Copyright (c) 2026, Minghua Shen.
"""
FlashAttention v3 反向 pytest。

- 正向：flash_attn_func / flash_attn_varlen_func 得到 out / softmax_lse，不与标杆比较
- 标杆：小算子 fa_small_op_golden（golden_*_bwd_from_fwd，传入 FA out/lse，仅算反传）
- 被测：torch.autograd.grad（FlashAttnFunc / FlashAttnVarlenFunc）

用法:
  pytest tests/test_flash_attn_npu_v3_bwd.py -v
  pytest tests/test_flash_attn_npu_v3_bwd.py -k swa -v
"""

import gc
import os
import random
import sys

import pytest
import torch
import torch_npu

if "Ascend950" in (torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""):
    pytest.skip("flash_attn_func / flash_attn_varlen_func not on Ascend950", allow_module_level=True)

from flash_attn_npu_3 import flash_attn_func, flash_attn_varlen_func

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from fa_small_op_golden import golden_bsnd_bwd_from_fwd, golden_tnd_bwd_from_fwd

RTOL_GOLDEN = 1e-2
ATOL_GOLDEN = 1e-2
INPUT_LIMIT = 2.0
DROPOUT_P = 0.0
GTYPE = torch.float64
CASE_SEED = 42


def golden_tolerance(_data_type):
    return RTOL_GOLDEN, ATOL_GOLDEN


def assert_grad_close(fa_grad, golden_grad, data_type, name=""):
    rtol, atol = golden_tolerance(data_type)
    torch.testing.assert_close(
        fa_grad.cpu(), golden_grad.cpu(), rtol=rtol, atol=atol,
        msg=f"{name} grad mismatch (rtol={rtol}, atol={atol})",
    )


@pytest.fixture(autouse=True)
def _npu_test_hygiene():
    if torch.npu.is_available():
        torch.npu.set_device(0)
    torch.manual_seed(CASE_SEED)
    random.seed(CASE_SEED)
    if hasattr(torch.npu, "manual_seed"):
        torch.npu.manual_seed(CASE_SEED)
    yield
    if torch.npu.is_available():
        torch.npu.synchronize()
    gc.collect()
    if hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()


def get_cu_seqlens(seqlens_list):
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int32)
    for i in range(len(seqlens_list) + 1):
        cu[i] = int(sum(seqlens_list[:i]))
    return cu


def rand_seqlens_qk(batch_size, max_seqlen_q, max_seqlen_k, seed=CASE_SEED):
    """Per-batch random (seqlens_q, seqlens_k) with seqlen_q[i] <= seqlen_k[i]."""
    min_q = max(1, max_seqlen_q // 4)
    min_k = max(1, max_seqlen_k // 4)
    rng = random.Random(seed)
    seqlens_q, seqlens_k = [], []
    for _ in range(batch_size):
        sk = rng.randint(min_k, max_seqlen_k)
        q_hi = min(max_seqlen_q, sk)
        q_lo = min(min_q, q_hi)
        seqlens_q.append(rng.randint(q_lo, q_hi))
        seqlens_k.append(sk)
    return seqlens_q, seqlens_k


def rand_inputs(shape, data_type, device):
    return (INPUT_LIMIT * (torch.rand(shape) - 0.5)).to(data_type).to(device)


def run_bsnd_bwd(
    query, key, value, dout, data_type,
    num_heads, kv_heads, scale, softcap, is_causal, window_size=(-1, -1),
):
    q_ag = query.detach().clone().requires_grad_(True)
    k_ag = key.detach().clone().requires_grad_(True)
    v_ag = value.detach().clone().requires_grad_(True)
    torch.npu.synchronize()
    out_fa, lse_fa = flash_attn_func(
        q_ag, k_ag, v_ag,
        softmax_scale=scale,
        causal=is_causal,
        window_size=list(window_size),
        softcap=softcap,
        return_attn_probs=True,
    )
    dq_ag, dk_ag, dv_ag = torch.autograd.grad(out_fa, (q_ag, k_ag, v_ag), dout)
    torch.npu.synchronize()

    dq_golden, dk_golden, dv_golden = golden_bsnd_bwd_from_fwd(
        query, key, value, dout, out_fa.detach(), lse_fa.detach(),
        num_heads, kv_heads, scale, softcap, DROPOUT_P,
        is_causal, window_size[0], window_size[1],
        gtype=GTYPE,
    )
    return data_type, dq_ag, dk_ag, dv_ag, dq_golden, dk_golden, dv_golden


def run_varlen_bwd(
    query, key, value, dout, data_type,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    seqlens_q, seqlens_k,
    scale, softcap, is_causal, window_size=(-1, -1),
):
    num_heads = query.shape[1]
    kv_heads = key.shape[1]

    q_ag = query.detach().clone().requires_grad_(True)
    k_ag = key.detach().clone().requires_grad_(True)
    v_ag = value.detach().clone().requires_grad_(True)
    torch.npu.synchronize()
    out_fa, lse_fa = flash_attn_varlen_func(
        q_ag, k_ag, v_ag,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        softmax_scale=scale,
        causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        return_attn_probs=True,
    )
    dq_ag, dk_ag, dv_ag = torch.autograd.grad(out_fa, (q_ag, k_ag, v_ag), dout)
    torch.npu.synchronize()

    dq_golden, dk_golden, dv_golden = golden_tnd_bwd_from_fwd(
        query, key, value, dout, out_fa.detach(), lse_fa.detach(),
        num_heads, kv_heads, seqlens_q, seqlens_k, scale, softcap, DROPOUT_P,
        is_causal, window_size[0], window_size[1],
        gtype=GTYPE,
    )
    return data_type, dq_ag, dk_ag, dv_ag, dq_golden, dk_golden, dv_golden


def assert_bwd_results(data_type, dq_ag, dk_ag, dv_ag, dq_golden, dk_golden, dv_golden):
    assert_grad_close(dq_ag, dq_golden, data_type, "dq")
    assert_grad_close(dk_ag, dk_golden, data_type, "dk")
    assert_grad_close(dv_ag, dv_golden, data_type, "dv")


test_cases_bsnd = [
    (torch.float16, 1, 1, 1, 1024, 1024, 128, False, 0.0),
    (torch.float16, 5, 4, 4, 1024, 1024, 128, True, 0.0),
    (torch.float16, 7, 1, 1, 512, 512, 128, False, 0.0),
    (torch.float16, 4, 2, 1, 513, 513, 128, False, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, False, 0.0),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, True, 0.0),
    (torch.bfloat16, 7, 1, 1, 512, 512, 128, False, 0.0),
    (torch.bfloat16, 4, 2, 1, 513, 513, 128, False, 0.0),
    (torch.float16, 1, 1, 1, 1024, 1024, 128, False, 30.0),
    (torch.float16, 5, 4, 4, 1024, 1024, 128, True, 30.0),
    (torch.float16, 7, 1, 1, 512, 512, 128, False, 30.0),
    (torch.float16, 4, 2, 1, 513, 513, 128, False, 30.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, False, 30.0),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, True, 30.0),
    (torch.bfloat16, 7, 1, 1, 512, 512, 128, False, 30.0),
    (torch.bfloat16, 4, 2, 1, 513, 513, 128, False, 30.0),
    (torch.float16, 24, 5, 5, 9216, 9216, 64, False, 0.0),
    (torch.float16, 24, 10, 10, 2304, 2304, 64, False, 0.0),
    (torch.float16, 24, 20, 20, 144, 144, 64, False, 0.0),
    (torch.float16, 24, 10, 10, 2304, 80, 64, False, 0.0),
    (torch.float16, 24, 20, 20, 576, 80, 64, False, 0.0),
    (torch.float16, 24, 20, 20, 144, 80, 64, False, 0.0),
    (torch.bfloat16, 4, 10, 10, 2048, 2048, 128, True, 0.0),
    (torch.bfloat16, 8, 10, 10, 2048, 2048, 128, True, 0.0),
    (torch.bfloat16, 2, 10, 10, 4096, 4096, 128, True, 0.0),
    (torch.bfloat16, 4, 10, 10, 4096, 4096, 128, True, 0.0),
    (torch.float16, 1, 20, 20, 576, 77, 64, False, 0.0),
    (torch.float16, 32, 20, 20, 576, 576, 64, False, 0.0),
    (torch.float16, 32, 20, 20, 144, 144, 64, False, 0.0),
    (torch.float16, 32, 20, 20, 576, 77, 64, False, 0.0),
    (torch.float16, 8, 20, 20, 576, 576, 64, False, 0.0),
    (torch.float16, 16, 20, 20, 144, 144, 64, False, 0.0),
    (torch.float16, 16, 20, 20, 144, 77, 64, False, 0.0),
    (torch.bfloat16, 2, 8, 8, 4096, 4096, 128, True, 0.0),
    (torch.bfloat16, 4, 8, 8, 4096, 4096, 128, True, 0.0),
    (torch.float16, 4, 10, 10, 3952, 3952, 64, False, 0.0),
    (torch.float16, 4, 10, 10, 3952, 77, 64, False, 0.0),
    (torch.float16, 1, 10, 10, 3952, 512, 64, False, 0.0),
    (torch.float16, 1, 10, 10, 4032, 4032, 64, False, 0.0),
    (torch.float16, 2, 10, 10, 4032, 4032, 64, False, 0.0),
    (torch.float16, 4, 10, 10, 4032, 4032, 64, False, 0.0),
    (torch.float16, 1, 10, 10, 4096, 512, 64, False, 0.0),
    (torch.float16, 2, 10, 10, 4096, 512, 64, False, 0.0),
    (torch.float16, 1, 20, 20, 1008, 512, 64, False, 0.0),
    (torch.float16, 1, 20, 20, 1024, 1024, 64, False, 0.0),
    (torch.float16, 2, 20, 20, 1024, 1024, 64, False, 0.0),
    (torch.float16, 4, 20, 20, 1024, 1024, 64, False, 0.0),
    (torch.float16, 4, 20, 20, 1024, 77, 64, False, 0.0),
    (torch.float16, 1, 20, 20, 1024, 512, 64, False, 0.0),
    (torch.float16, 2, 20, 20, 1024, 512, 64, False, 0.0),
    (torch.float16, 1, 10, 10, 3712, 3712, 64, False, 0.0),
    (torch.float16, 1, 10, 10, 3744, 3744, 64, False, 0.0),
    (torch.float16, 1, 10, 10, 3840, 3840, 64, False, 0.0),
    (torch.float16, 2, 10, 10, 3840, 3840, 64, False, 0.0),
    (torch.float16, 4, 10, 10, 3840, 3840, 64, False, 0.0),
    (torch.float16, 4, 10, 10, 3840, 77, 64, False, 0.0),
    (torch.float16, 1, 10, 10, 3840, 512, 64, False, 0.0),
    (torch.float16, 2, 10, 10, 3840, 512, 64, False, 0.0),
    (torch.float16, 4, 10, 10, 3872, 3872, 64, False, 0.0),
    (torch.float16, 4, 10, 10, 3888, 3888, 64, False, 0.0),
    (torch.float16, 4, 10, 10, 3888, 77, 64, False, 0.0),
    (torch.float16, 2, 10, 10, 3888, 512, 64, False, 0.0),
    (torch.float16, 1, 10, 10, 4000, 4000, 64, False, 0.0),
    (torch.float16, 4, 10, 10, 4000, 77, 64, False, 0.0),
    (torch.float16, 1, 10, 10, 4048, 4048, 64, False, 0.0),
    (torch.float16, 1, 10, 10, 4080, 4080, 64, False, 0.0),
    (torch.float16, 2, 10, 10, 4080, 4080, 64, False, 0.0),
    (torch.float16, 2, 20, 20, 1012, 1012, 64, False, 0.0),
    (torch.float16, 4, 20, 20, 1012, 1012, 64, False, 0.0),
    (torch.float16, 2, 20, 20, 1012, 512, 64, False, 0.0),
    (torch.float16, 4, 20, 20, 1020, 1020, 64, False, 0.0),
    (torch.float16, 1, 20, 20, 1020, 512, 64, False, 0.0),
    (torch.float16, 2, 20, 20, 928, 928, 64, False, 0.0),
    (torch.float16, 4, 20, 20, 928, 928, 64, False, 0.0),
    (torch.float16, 4, 20, 20, 936, 936, 64, False, 0.0),
    (torch.float16, 1, 20, 20, 960, 960, 64, False, 0.0),
    (torch.float16, 4, 20, 20, 960, 77, 64, False, 0.0),
    (torch.float16, 2, 20, 20, 960, 512, 64, False, 0.0),
    (torch.float16, 2, 20, 20, 968, 968, 64, False, 0.0),
    (torch.float16, 2, 20, 20, 972, 972, 64, False, 0.0),
    (torch.float16, 4, 20, 20, 972, 972, 64, False, 0.0),
    (torch.float16, 24, 5, 5, 9216, 77, 64, False, 0.0),
    (torch.float16, 24, 10, 10, 2304, 77, 64, False, 0.0),
    (torch.bfloat16, 702, 16, 16, 16, 16, 72, False, 0.0),
    (torch.bfloat16, 140, 16, 16, 96, 96, 72, False, 0.0),
    (torch.bfloat16, 420, 16, 16, 32, 32, 72, False, 0.0),
    (torch.bfloat16, 782, 16, 16, 128, 128, 72, False, 0.0),
    (torch.bfloat16, 1820, 16, 16, 64, 64, 72, False, 0.0),
    (torch.float16, 24, 5, 5, 9216, 80, 64, False, 0.0),
    (torch.bfloat16, 1, 12, 12, 4096, 4096, 128, True, 0.0),
    (torch.float16, 1, 12, 12, 4096, 4096, 128, True, 0.0),
    (torch.float16, 1, 4, 4, 4096, 4096, 256, True, 0.0),
    (torch.float16, 1, 10, 10, 4096, 4096, 64, False, 0.0),
    (torch.float16, 2, 10, 10, 4096, 4096, 64, False, 0.0),
    (torch.float16, 4, 10, 10, 4096, 4096, 64, False, 0.0),
    (torch.float16, 4, 10, 10, 4096, 77, 64, False, 0.0),
    (torch.float16, 24, 10, 10, 77, 77, 64, False, 0.0),
    (torch.float16, 24, 20, 20, 114, 114, 64, False, 0.0),
    (torch.float16, 24, 20, 20, 576, 576, 64, False, 0.0),
    (torch.float16, 24, 20, 20, 77, 77, 64, False, 0.0),
    (torch.float16, 24, 5, 5, 77, 77, 64, False, 0.0),
]


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, softcap",
    test_cases_bsnd,
)
def test_fa_bsnd_bwd(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, softcap,
):
    query = rand_inputs((batch_size, q_seqlen, num_heads, head_size), data_type, "npu")
    key = rand_inputs((batch_size, kv_seqlen, kv_heads, head_size), data_type, "npu")
    value = rand_inputs((batch_size, kv_seqlen, kv_heads, head_size), data_type, "npu")
    dout = rand_inputs((batch_size, q_seqlen, num_heads, head_size), data_type, "npu")

    scale = 1.0 / (head_size ** 0.5)
    results = run_bsnd_bwd(
        query, key, value, dout, data_type,
        num_heads, kv_heads, scale, softcap, is_causal, (-1, -1),
    )
    assert_bwd_results(*results)


test_cases_bsnd_swa = [
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, 512, 0, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, 512, 256, 0.0),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, True, -128, 864, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, False, 0, 256, 0.0),
    (torch.float16, 2, 2, 2, 512, 512, 128, False, 64, 128, 0.0),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, True, 512, 0, 0.0),
    (torch.bfloat16, 2, 6, 2, 2, 1024, 128, True, -1, -1, 0.0),
    (torch.bfloat16, 2, 6, 2, 2, 1024, 128, True, 256, 0, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, 512, 0, 30.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, 512, 256, 30.0),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, True, -128, 864, 30.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, False, 0, 256, 30.0),
    (torch.float16, 2, 2, 2, 512, 512, 128, False, 64, 128, 30.0),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, True, 512, 0, 30.0),
    (torch.bfloat16, 2, 6, 2, 2, 1024, 128, True, -1, -1, 30.0),
    (torch.bfloat16, 2, 6, 2, 2, 1024, 128, True, 256, 0, 30.0),
    (torch.bfloat16, 144, 14, 14, 16, 16, 120, False, -11, 14, 0.0),
]


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, "
    "is_causal, window_size_left, window_size_right, softcap",
    test_cases_bsnd_swa,
)
def test_fa_bsnd_bwd_swa(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
    is_causal, window_size_left, window_size_right, softcap
):
    query = rand_inputs((batch_size, q_seqlen, num_heads, head_size), data_type, "npu")
    key = rand_inputs((batch_size, kv_seqlen, kv_heads, head_size), data_type, "npu")
    value = rand_inputs((batch_size, kv_seqlen, kv_heads, head_size), data_type, "npu")
    dout = rand_inputs((batch_size, q_seqlen, num_heads, head_size), data_type, "npu")

    scale = 1.0 / (head_size ** 0.5)
    window_size = (window_size_left, window_size_right)
    results = run_bsnd_bwd(
        query, key, value, dout, data_type,
        num_heads, kv_heads, scale, softcap, is_causal, window_size,
    )
    assert_bwd_results(*results)


test_cases_varlen = [
    (torch.bfloat16, 3, 1, 1, 512, 1024, 128, True, 0.0),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, False, 0.0),
    (torch.float16, 7, 5, 1, 512, 512, 128, True, 0.0),
    (torch.float16, 7, 5, 1, 777, 888, 192, False, 0.0),
    (torch.float16, 7, 5, 1, 1777, 1888, 256, True, 0.0),
    (torch.bfloat16, 3, 1, 1, 7777, 8192, 64, True, 0.0),
    (torch.bfloat16, 7, 5, 1, 711, 8192, 111, True, 0.0),
    (torch.bfloat16, 3, 16, 16, 562, 562, 96, False, 0.0),
    (torch.bfloat16, 3, 1, 1, 512, 1024, 128, True, 30.0),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, False, 30.0),
    (torch.float16, 7, 5, 1, 512, 512, 128, True, 30.0),
    (torch.float16, 7, 5, 1, 777, 888, 192, False, 30.0),
    (torch.float16, 7, 5, 1, 1777, 1888, 256, True, 30.0),
    (torch.bfloat16, 3, 1, 1, 7777, 8192, 64, True, 30.0),
    (torch.bfloat16, 7, 5, 1, 711, 8192, 111, True, 30.0),
    (torch.bfloat16, 3, 16, 16, 562, 562, 96, False, 30.0),
]


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, softcap",
    test_cases_varlen,
)
def test_fa_varlen_bwd(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, softcap
):
    seqlens_q, seqlens_k = rand_seqlens_qk(batch_size, q_seqlen, kv_seqlen)
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    query = rand_inputs((total_q, num_heads, head_size), data_type, "npu")
    key = rand_inputs((total_k, kv_heads, head_size), data_type, "npu")
    value = rand_inputs((total_k, kv_heads, head_size), data_type, "npu")
    dout = rand_inputs((total_q, num_heads, head_size), data_type, "npu")

    cu_seqlens_q = get_cu_seqlens(seqlens_q).to(dtype=torch.int32).npu()
    cu_seqlens_k = get_cu_seqlens(seqlens_k).to(dtype=torch.int32).npu()
    scale = 1.0 / (head_size ** 0.5)

    results = run_varlen_bwd(
        query, key, value, dout, data_type,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        seqlens_q, seqlens_k,
        scale, softcap, is_causal, (-1, -1),
    )
    assert_bwd_results(*results)


test_cases_varlen_swa = [
    (torch.bfloat16, 3, 4, 4, 1024, 1024, 128, True, 512, 0, 0.0),
    (torch.bfloat16, 3, 1, 1, 512, 1024, 128, True, 512, 0, 0.0),
    (torch.bfloat16, 3, 1, 1, 512, 1024, 128, False, 0, 256, 0.0),
    (torch.float16, 3, 2, 2, 512, 512, 128, False, 64, 128, 0.0),
    (torch.bfloat16, 3, 4, 4, 1024, 1024, 128, True, -128, 864, 0.0),
    (torch.bfloat16, 3, 4, 4, 1024, 1024, 128, True, 512, 0, 30.0),
    (torch.bfloat16, 3, 1, 1, 512, 1024, 128, True, 512, 0, 30.0),
    (torch.bfloat16, 3, 1, 1, 512, 1024, 128, False, 0, 256, 30.0),
    (torch.float16, 3, 2, 2, 512, 512, 128, False, 64, 128, 30.0),
    (torch.bfloat16, 3, 4, 4, 1024, 1024, 128, True, -128, 864, 30.0),
]


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, "
    "is_causal, window_size_left, window_size_right, softcap",
    test_cases_varlen_swa,
)
def test_fa_varlen_bwd_swa(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
    is_causal, window_size_left, window_size_right, softcap
):
    seqlens_q, seqlens_k = rand_seqlens_qk(batch_size, q_seqlen, kv_seqlen)
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    query = rand_inputs((total_q, num_heads, head_size), data_type, "npu")
    key = rand_inputs((total_k, kv_heads, head_size), data_type, "npu")
    value = rand_inputs((total_k, kv_heads, head_size), data_type, "npu")
    dout = rand_inputs((total_q, num_heads, head_size), data_type, "npu")

    cu_seqlens_q = get_cu_seqlens(seqlens_q).to(dtype=torch.int32).npu()
    cu_seqlens_k = get_cu_seqlens(seqlens_k).to(dtype=torch.int32).npu()
    scale = 1.0 / (head_size ** 0.5)
    window_size = (window_size_left, window_size_right)

    results = run_varlen_bwd(
        query, key, value, dout, data_type,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        seqlens_q, seqlens_k,
        scale, softcap, is_causal, window_size,
    )
    assert_bwd_results(*results)
