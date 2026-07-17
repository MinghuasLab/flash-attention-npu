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
import sys

import pytest
import torch
import torch_npu

if "Ascend950" in (torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""):
    pytest.skip("flash_attn_func / flash_attn_varlen_func not on Ascend950", allow_module_level=True)

from flash_attn_npu_v3 import flash_attn_func, flash_attn_varlen_func

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
    (torch.bfloat16, 1, 1, 1, 512, 1024, 128, True, 0.0),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, False, 0.0),
    (torch.float16, 7, 5, 1, 512, 512, 128, True, 0.0),
    (torch.float16, 7, 5, 1, 777, 888, 192, False, 0.0),
    (torch.float16, 7, 5, 1, 1777, 1888, 256, True, 0.0),
    (torch.bfloat16, 1, 1, 1, 7777, 8192, 64, True, 0.0),
    (torch.bfloat16, 7, 5, 1, 711, 8192, 111, True, 0.0),
    (torch.bfloat16, 1, 16, 16, 562, 562, 96, False, 0.0),
    (torch.bfloat16, 1, 1, 1, 512, 1024, 128, True, 30.0),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, False, 30.0),
    (torch.float16, 7, 5, 1, 512, 512, 128, True, 30.0),
    (torch.float16, 7, 5, 1, 777, 888, 192, False, 30.0),
    (torch.float16, 7, 5, 1, 1777, 1888, 256, True, 30.0),
    (torch.bfloat16, 1, 1, 1, 7777, 8192, 64, True, 30.0),
    (torch.bfloat16, 7, 5, 1, 711, 8192, 111, True, 30.0),
    (torch.bfloat16, 1, 16, 16, 562, 562, 96, False, 30.0),
]


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, softcap",
    test_cases_varlen,
)
def test_fa_varlen_bwd(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, softcap
):
    total_q = batch_size * q_seqlen
    total_k = batch_size * kv_seqlen
    query = rand_inputs((total_q, num_heads, head_size), data_type, "npu")
    key = rand_inputs((total_k, kv_heads, head_size), data_type, "npu")
    value = rand_inputs((total_k, kv_heads, head_size), data_type, "npu")
    dout = rand_inputs((total_q, num_heads, head_size), data_type, "npu")

    seqlens_q = [q_seqlen] * batch_size
    seqlens_k = [kv_seqlen] * batch_size
    cu_seqlens_q = get_cu_seqlens(seqlens_q).to(dtype=torch.int32).npu()
    cu_seqlens_k = get_cu_seqlens(seqlens_k).to(dtype=torch.int32).npu()
    scale = 1.0 / (head_size ** 0.5)

    results = run_varlen_bwd(
        query, key, value, dout, data_type,
        cu_seqlens_q, cu_seqlens_k,
        q_seqlen, kv_seqlen,
        seqlens_q, seqlens_k,
        scale, softcap, is_causal, (-1, -1),
    )
    assert_bwd_results(*results)


test_cases_varlen_swa = [
    (torch.bfloat16, 1, 4, 4, 1024, 1024, 128, True, 512, 0, 0.0),
    (torch.bfloat16, 1, 1, 1, 512, 1024, 128, True, 512, 0, 0.0),
    (torch.bfloat16, 1, 1, 1, 512, 1024, 128, False, 0, 256, 0.0),
    (torch.float16, 1, 2, 2, 512, 512, 128, False, 64, 128, 0.0),
    (torch.bfloat16, 1, 4, 4, 1024, 1024, 128, True, -128, 864, 0.0),
    (torch.bfloat16, 1, 4, 4, 1024, 1024, 128, True, 512, 0, 30.0),
    (torch.bfloat16, 1, 1, 1, 512, 1024, 128, True, 512, 0, 30.0),
    (torch.bfloat16, 1, 1, 1, 512, 1024, 128, False, 0, 256, 30.0),
    (torch.float16, 1, 2, 2, 512, 512, 128, False, 64, 128, 30.0),
    (torch.bfloat16, 1, 4, 4, 1024, 1024, 128, True, -128, 864, 30.0),
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
    total_q = batch_size * q_seqlen
    total_k = batch_size * kv_seqlen
    query = rand_inputs((total_q, num_heads, head_size), data_type, "npu")
    key = rand_inputs((total_k, kv_heads, head_size), data_type, "npu")
    value = rand_inputs((total_k, kv_heads, head_size), data_type, "npu")
    dout = rand_inputs((total_q, num_heads, head_size), data_type, "npu")

    seqlens_q = [q_seqlen] * batch_size
    seqlens_k = [kv_seqlen] * batch_size
    cu_seqlens_q = get_cu_seqlens(seqlens_q).to(dtype=torch.int32).npu()
    cu_seqlens_k = get_cu_seqlens(seqlens_k).to(dtype=torch.int32).npu()
    scale = 1.0 / (head_size ** 0.5)
    window_size = (window_size_left, window_size_right)

    results = run_varlen_bwd(
        query, key, value, dout, data_type,
        cu_seqlens_q, cu_seqlens_k,
        q_seqlen, kv_seqlen,
        seqlens_q, seqlens_k,
        scale, softcap, is_causal, window_size,
    )
    assert_bwd_results(*results)
