# Copyright (c) 2026, Huawei Technologies Co., Ltd.
"""
FlashAttention v3 (Ascend950) 反向矩阵测试：直接调用 _flash_attn_backward。

Ascend950 的 autograd 前向 lse 布局尚不可用，故本测试不经过 autograd：
- out / softmax_lse 由 torch 小算子参考前向（CPU fp32）算出再搬上 NPU；
- 直接调用 _flash_attn_backward(deterministic=det) 跑反向。

覆盖矩阵（每个 case 都跑 det=True / det=False 两遍）：
- layout: BSND、TND(varlen)
- 头型: MHA、GQA、MQA
- 因果: causal（右下对齐，支持 sq != sk）、non-causal
- 规模: small(<=512)、mid(1k~2k)、large(4k)、pack（多序列）、非对齐尾块

SWA 与普通用例共用 BSND_CASES / VARLEN_CASES 和现有测试入口。
window_size_left / window_size_right 同时传给参考前向、反向算子和标杆；
(-1, -1) 表示无限窗口，causal 为右下对齐。全掩码行使用 out=0、lse=+inf。

检查项（任一失败即断言失败）：
- golden: dq/dk/dv 与 fa_small_op_golden 对比（rtol=atol=1e-2）；
- det=True: 重复 REPEAT 次逐位一致 (torch.equal)；
- det=False: 首跑与第二次运行的 max|diff| 仅打印（非确定性允许，不做断言）；
- det vs nondet: 两种模式的梯度互相接近（同一容差）。

矩阵仅适用于 Ascend950；其他设备（如 Ascend910）整文件 skip。

用法（仓库根目录）:
  python -m pytest tests/test_flash_attn_npu_v3_bwd_det.py -v
  python -m pytest tests/test_flash_attn_npu_v3_bwd_det.py -k bsnd        # 只跑 BSND
  python -m pytest tests/test_flash_attn_npu_v3_bwd_det.py -k tnd         # 只跑 TND
  python -m pytest tests/test_flash_attn_npu_v3_bwd_det.py -k small       # 只跑小规模
  python -m pytest tests/test_flash_attn_npu_v3_bwd_det.py \
      -k "not large and not long and not eq and not pack8"                # 跳过大规模
"""

import pytest
import torch
import torch_npu

# 950 后端扩展在分机编译 (FLASH_ATTN_BUILD_NPU=910) 的 910 机器上不存在,
# 模块级 import 会在 collection 阶段报错, 故 import 前整文件跳过 (同 v2 模式)。
if "Ascend950" not in (torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""):
    pytest.skip("flash_attn_npu_3 950 backward tests require Ascend950", allow_module_level=True)

from flash_attn_npu_3.flash_attn_npu_interface_950 import _flash_attn_backward
from tests.fa_small_op_golden import (
    golden_bsnd_bwd_from_fwd, golden_tnd_bwd_from_fwd, make_window_atten_mask,
)

INPUT_LIMIT = 2.0
DTYPE = torch.bfloat16
DEVICE = "npu"
RTOL_GOLDEN = 1e-2
ATOL_GOLDEN = 1e-2
DROPOUT_P = 0.0
SOFTCAP = 0.0
GTYPE = torch.float64
REPEAT = 20  # det 模式逐位一致的重复次数

# (name, bsz, seqlen_q, seqlen_k, nheads_q, nheads_kv, headdim, causal, large, window_size_left, window_size_right)
BSND_CASES = [
    # ---- small ----
    ("bsnd_small_mha_causal", 1, 128, 128, 2, 2, 128, True, False, -1, -1),
    ("bsnd_small_mha_nc", 1, 128, 128, 2, 2, 128, False, False, -1, -1),
    ("bsnd_small_gqa_tail_causal", 2, 200, 300, 4, 2, 64, True, False, -1, -1),
    ("bsnd_small_mqa_tail_nc", 2, 300, 500, 4, 1, 64, False, False, -1, -1),
    ("bsnd_tiny_unaligned_causal", 2, 33, 77, 4, 4, 64, True, False, -1, -1),
    # ---- mid ----
    ("bsnd_mid_mha_square_causal", 1, 1024, 1024, 8, 8, 128, True, False, -1, -1),
    ("bsnd_mid_mha_square_nc", 1, 1024, 1024, 8, 8, 128, False, False, -1, -1),
    ("bsnd_mid_mha_rect_causal", 2, 333, 777, 4, 4, 64, True, False, -1, -1),
    ("bsnd_mid_mha_rect_nc", 2, 777, 333, 4, 4, 64, False, False, -1, -1),
    ("bsnd_mid_gqa_square_causal", 1, 1024, 1024, 8, 2, 128, True, False, -1, -1),
    ("bsnd_mid_gqa_square_nc", 1, 1024, 1024, 8, 2, 128, False, False, -1, -1),
    ("bsnd_mid_mqa_causal", 2, 300, 500, 4, 1, 64, True, False, -1, -1),
    ("bsnd_mid_mqa_nc", 2, 300, 500, 4, 1, 64, False, False, -1, -1),
    ("bsnd_mid_gqa_rect_causal", 2, 500, 900, 6, 3, 128, True, False, -1, -1),
    ("bsnd_mid_gqa_rect_nc", 2, 500, 900, 6, 3, 128, False, False, -1, -1),
    ("bsnd_mid_mha_hd64_nc", 2, 777, 333, 6, 3, 64, False, False, -1, -1),
    # ---- long / large ----
    ("bsnd_long_mha_causal", 1, 2048, 2048, 8, 8, 128, True, True, -1, -1),
    ("bsnd_long_mha_nc", 1, 2048, 2048, 8, 8, 128, False, True, -1, -1),
    ("bsnd_long_gqa_causal", 1, 2048, 2048, 8, 2, 128, True, True, -1, -1),
    ("bsnd_large_mha_causal", 1, 4096, 4096, 4, 4, 128, True, True, -1, -1),
    ("bsnd_large_mha_nc", 1, 4096, 4096, 4, 4, 128, False, True, -1, -1),
    # SWA: same entry point and golden as the ordinary rows.
    ("bsnd_swa_zero", 1, 257, 257, 2, 2, 64, False, False, 0, 0),
    ("bsnd_swa_edges", 1, 193, 321, 4, 2, 64, False, False, 65, 129),
    ("bsnd_swa_causal_skipped_kv", 1, 129, 513, 2, 2, 64, True, False, 63, 129),
    ("bsnd_swa_v2_right", 2, 129, 513, 4, 2, 64, False, False, 0, 256),
    ("bsnd_swa_v2_bidir", 1, 113, 211, 4, 1, 128, False, False, 64, 128),
    ("bsnd_swa_v2_left", 1, 513, 1025, 2, 2, 64, False, False, 512, 0),
    ("bsnd_swa_v2_asymmetric", 1, 257, 1025, 4, 2, 64, False, False, 542, 647),
    ("bsnd_swa_v2_wide", 1, 513, 1537, 4, 1, 128, False, False, 826, 973),
    ("bsnd_swa_left_unbounded", 1, 257, 129, 4, 2, 64, False, False, -1, 1),
    ("bsnd_swa_right_unbounded", 1, 129, 513, 2, 2, 64, False, False, 64, -1),
    ("bsnd_swa_negative_right", 1, 129, 513, 4, 2, 64, False, False, 508, -256),
    ("bsnd_swa_overlong_right", 1, 129, 257, 4, 2, 64, False, False, 64, 864),
    ("bsnd_swa_causal_boundary", 1, 129, 513, 4, 2, 64, True, False, 127, 256),
    ("bsnd_swa_mqa", 1, 131, 259, 4, 1, 128, False, False, 17, 33),
]

# (name, cu_seqlens_q, cu_seqlens_k, nheads_q, nheads_kv, headdim, causal, large, window_size_left, window_size_right)
VARLEN_CASES = [
    # ---- small / ragged ----
    ("tnd_small_mha_causal", [0, 256, 640], [0, 256, 640], 4, 4, 128, True, False, -1, -1),
    ("tnd_small_mha_nc", [0, 256, 640], [0, 256, 640], 4, 4, 128, False, False, -1, -1),
    ("tnd_ragged_mqa_causal", [0, 100, 900], [0, 300, 1600], 4, 1, 64, True, False, -1, -1),
    ("tnd_ragged_mqa_nc", [0, 100, 900], [0, 300, 700], 4, 1, 64, False, False, -1, -1),
    ("tnd_ragged_mha_nc", [0, 100, 900], [0, 300, 700], 4, 4, 64, False, False, -1, -1),
    ("tnd_ragged_gqa_causal", [0, 512, 1536], [0, 768, 2048], 4, 2, 128, True, False, -1, -1),
    ("tnd_ragged_gqa_nc", [0, 512, 1536], [0, 768, 2048], 4, 2, 128, False, False, -1, -1),
    # ---- equal-length (left-up causal schedule) ----
    ("tnd_eq_mha_causal", [0, 2048, 4096], [0, 2048, 4096], 8, 8, 128, True, True, -1, -1),
    ("tnd_eq_mha_nc", [0, 2048, 4096], [0, 2048, 4096], 8, 8, 128, False, True, -1, -1),
    ("tnd_eq_gqa_causal", [0, 1024, 2048], [0, 1024, 2048], 8, 2, 128, True, True, -1, -1),
    ("tnd_eq_gqa_nc", [0, 1024, 2048], [0, 1024, 2048], 8, 2, 128, False, True, -1, -1),
    # ---- pack (multi-sequence) ----
    (
        "tnd_pack8_mha_causal",
        [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096],
        [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096],
        8,
        8,
        128,
        True,
        True,
        -1,
        -1,
    ),
    (
        "tnd_pack8_mha_nc",
        [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096],
        [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096],
        8,
        8,
        128,
        False,
        True,
        -1,
        -1,
    ),
    # ---- large single ----
    (
        "tnd_large4_mha_causal",
        [0, 2048, 4096, 6144, 8192],
        [0, 2048, 4096, 6144, 8192],
        4,
        4,
        128,
        True,
        True,
        -1,
        -1,
    ),
    ("tnd_large_mha_nc", [0, 4096], [0, 4096], 4, 4, 128, False, True, -1, -1),
    (
        "tnd_large4_gqa_causal",
        [0, 1024, 2048, 3072, 4096],
        [0, 1024, 2048, 3072, 4096],
        8,
        2,
        128,
        True,
        True,
        -1,
        -1,
    ),
    (
        "tnd_large4_mqa_causal",
        [0, 2048, 4096, 6144, 8192],
        [0, 2048, 4096, 6144, 8192],
        4,
        1,
        64,
        True,
        True,
        -1,
        -1,
    ),
    # SWA: same entry point and golden as the ordinary rows.
    ("tnd_swa_zero", [0, 257, 386], [0, 257, 386], 2, 2, 64, False, False, 0, 0),
    ("tnd_swa_edges", [0, 193, 324], [0, 321, 580], 4, 2, 64, False, False, 65, 129),
    ("tnd_swa_causal_skipped_kv", [0, 129, 194], [0, 513, 772], 2, 2, 64, True, False, 63, 129),
    ("tnd_swa_v2_right", [0, 129, 194], [0, 513, 770], 4, 2, 64, False, False, 0, 256),
    ("tnd_swa_v2_bidir", [0, 113, 178], [0, 211, 340], 4, 1, 128, False, False, 64, 128),
    ("tnd_swa_v2_left", [0, 513, 642], [0, 1025, 1282], 2, 2, 64, False, False, 512, 0),
    ("tnd_swa_v2_asymmetric", [0, 257, 388], [0, 1025, 1538], 4, 2, 64, False, False, 542, 647),
    ("tnd_swa_v2_wide", [0, 513, 770], [0, 1537, 2562], 4, 1, 128, False, False, 826, 973),
    ("tnd_swa_left_unbounded", [0, 257, 388], [0, 129, 388], 4, 2, 64, False, False, -1, 1),
    ("tnd_swa_right_unbounded", [0, 129, 386], [0, 513, 770], 2, 2, 64, False, False, 64, -1),
    ("tnd_swa_negative_right", [0, 129, 194], [0, 513, 770], 4, 2, 64, False, False, 508, -256),
    ("tnd_swa_overlong_right", [0, 129, 194], [0, 257, 386], 4, 2, 64, False, False, 64, 864),
    ("tnd_swa_causal_boundary", [0, 129, 194], [0, 513, 770], 4, 2, 64, True, False, 127, 256),
    ("tnd_swa_mqa", [0, 131, 196], [0, 259, 388], 4, 1, 128, False, False, 17, 33),
]


def _is_ascend950():
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    return "Ascend950" in name


def rand_inputs(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return (INPUT_LIMIT * (torch.rand(shape, generator=g) - 0.5)).to(DTYPE).to(DEVICE)


def torch_ref_fwd_bsnd(q, k, v, scale, causal, window_size_left, window_size_right):
    """torch 小算子参考前向（CPU fp32）。q:(B,Sq,Hq,D) k/v:(B,Sk,Hkv,D)。

    返回 out:(B,Sq,Hq,D) 同 q dtype, lse:(B,Hq,Sq) fp32（自然对数）。
    """
    qb = q.detach().cpu().permute(0, 2, 1, 3).float()
    kb = k.detach().cpu().permute(0, 2, 1, 3).float()
    vb = v.detach().cpu().permute(0, 2, 1, 3).float()
    hq, hkv = qb.shape[1], kb.shape[1]
    if hq != hkv:
        g = hq // hkv
        kb = kb.repeat_interleave(g, dim=1)
        vb = vb.repeat_interleave(g, dim=1)
    sq, sk = qb.shape[2], kb.shape[2]
    s = torch.matmul(qb, kb.transpose(-1, -2)) * scale
    mask = make_window_atten_mask(sq, sk, causal, window_size_left, window_size_right)
    if mask.dim() > 0:
        s = s.masked_fill(mask.bool(), float("-inf"))
    lse = torch.logsumexp(s, dim=-1)  # (B,Hq,Sq) fp32
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, vb)  # (B,Hq,Sq,D)
    out = o.permute(0, 2, 1, 3).to(q.dtype)
    if mask.dim() > 0:
        fully_masked = mask.bool().all(dim=-1)
        out[:, fully_masked] = 0
        lse[:, :, fully_masked] = torch.inf
    return out.to(q.device), lse.to(q.device)


def torch_ref_fwd_tnd(q, k, v, cu_q, cu_k, scale, causal, window_size_left, window_size_right):
    """varlen 参考前向：逐 batch 切片复用 BSND 逻辑。

    返回 out:(total_q,Hq,D), lse:(Hq,total_q) fp32。
    """
    outs, lses = [], []
    for i in range(len(cu_q) - 1):
        qi = q[cu_q[i] : cu_q[i + 1]].unsqueeze(0)
        ki = k[cu_k[i] : cu_k[i + 1]].unsqueeze(0)
        vi = v[cu_k[i] : cu_k[i + 1]].unsqueeze(0)
        oi, li = torch_ref_fwd_bsnd(
            qi, ki, vi, scale, causal, window_size_left, window_size_right
        )
        outs.append(oi.squeeze(0))
        lses.append(li.squeeze(0))
    return torch.cat(outs, dim=0), torch.cat(lses, dim=1)


def run_bwd_bsnd(q, k, v, dout, out, lse, scale, causal, det, window_size_left, window_size_right):
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        None,
        None,
        None,
        None,
        None,
        None,
        dq,
        dk,
        dv,
        scale,
        causal,
        window_size_left,
        window_size_right,
        SOFTCAP,
        det,
        0,
    )
    torch.npu.synchronize()
    return dq, dk, dv


def run_bwd_varlen(q, k, v, dout, out, lse, cu_q_t, cu_k_t, max_sq, max_sk, scale, causal, det, window_size_left, window_size_right):
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        cu_q_t,
        cu_k_t,
        None,
        None,
        max_sq,
        max_sk,
        dq,
        dk,
        dv,
        scale,
        causal,
        window_size_left,
        window_size_right,
        SOFTCAP,
        det,
        0,
    )
    torch.npu.synchronize()
    return dq, dk, dv


def max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def _check_case(name, run_bwd, golden_fn):
    """golden 对齐 + det 逐位一致 + det/nd 互一致；不满足即断言失败。

    run_bwd(det) -> (dq,dk,dv); golden_fn() -> (dq,dk,dv)。
    """
    golden = golden_fn()
    grads = {"det": run_bwd(True), "nd": run_bwd(False)}

    for tag, grad in grads.items():
        for gname, got, gold in zip(("dq", "dk", "dv"), grad, golden):
            torch.testing.assert_close(
                got.cpu(),
                gold.cpu(),
                rtol=RTOL_GOLDEN,
                atol=ATOL_GOLDEN,
                msg=f"{name} [{tag}] {gname} vs golden",
            )

    for it in range(REPEAT):
        cur = run_bwd(True)
        for gname, a, b in zip(("dq", "dk", "dv"), cur, grads["det"]):
            assert torch.equal(a, b), (
                f"{name} [det] iter {it + 1}/{REPEAT}: {gname} 不一致, "
                f"max|diff|={max_diff(a, b):.6e}"
            )

    nd_again = run_bwd(False)
    d = max(max_diff(a, b) for a, b in zip(nd_again, grads["nd"]))
    print(f"[INFO] {name} [nd] 两次运行 max|diff|={d:.6e}")

    for gname, a, b in zip(("dq", "dk", "dv"), grads["det"], grads["nd"]):
        torch.testing.assert_close(
            a.cpu(),
            b.cpu(),
            rtol=RTOL_GOLDEN,
            atol=ATOL_GOLDEN,
            msg=f"{name}: det vs nondet {gname}",
        )

    print(f"[PASS] {name}: golden 一致 + {REPEAT} 次 det 逐位一致 + det/nondet 互相一致")


@pytest.mark.parametrize(
    "name, bsz, sq, sk, hq, hkv, hd, causal, large, window_size_left, window_size_right",
    BSND_CASES,
    ids=[c[0] for c in BSND_CASES],
)
@pytest.mark.skipif(not _is_ascend950(), reason="Ascend950 only")
def test_fa_bwd_det_bsnd(name, bsz, sq, sk, hq, hkv, hd, causal, large, window_size_left, window_size_right):
    scale = hd ** (-0.5)
    q = rand_inputs((bsz, sq, hq, hd), 42)
    k = rand_inputs((bsz, sk, hkv, hd), 43)
    v = rand_inputs((bsz, sk, hkv, hd), 44)
    dout = rand_inputs((bsz, sq, hq, hd), 45)
    out, lse = torch_ref_fwd_bsnd(
        q, k, v, scale, causal, window_size_left, window_size_right
    )

    def run_bwd(det):
        return run_bwd_bsnd(
            q, k, v, dout, out, lse, scale, causal, det,
            window_size_left, window_size_right,
        )

    def golden_fn():
        return golden_bsnd_bwd_from_fwd(
            q, k, v, dout, out, lse, hq, hkv, scale, SOFTCAP, DROPOUT_P, causal,
            window_size_left, window_size_right, gtype=GTYPE
        )

    _check_case(name, run_bwd, golden_fn)


@pytest.mark.parametrize(
    "name, cu_q, cu_k, hq, hkv, hd, causal, large, window_size_left, window_size_right",
    VARLEN_CASES,
    ids=[c[0] for c in VARLEN_CASES],
)
@pytest.mark.skipif(not _is_ascend950(), reason="Ascend950 only")
def test_fa_bwd_det_varlen(name, cu_q, cu_k, hq, hkv, hd, causal, large, window_size_left, window_size_right):
    scale = hd ** (-0.5)
    total_q, total_k = cu_q[-1], cu_k[-1]
    q = rand_inputs((total_q, hq, hd), 42)
    k = rand_inputs((total_k, hkv, hd), 43)
    v = rand_inputs((total_k, hkv, hd), 44)
    dout = rand_inputs((total_q, hq, hd), 45)
    cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=DEVICE)
    cu_k_t = torch.tensor(cu_k, dtype=torch.int32, device=DEVICE)
    max_sq = max(cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1))
    max_sk = max(cu_k[i + 1] - cu_k[i] for i in range(len(cu_k) - 1))
    out, lse = torch_ref_fwd_tnd(
        q, k, v, cu_q, cu_k, scale, causal, window_size_left, window_size_right
    )

    def run_bwd(det):
        return run_bwd_varlen(
            q, k, v, dout, out, lse, cu_q_t, cu_k_t, max_sq, max_sk, scale, causal, det,
            window_size_left, window_size_right,
        )

    def golden_fn():
        seqlens_q = [cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1)]
        seqlens_k = [cu_k[i + 1] - cu_k[i] for i in range(len(cu_k) - 1)]
        return golden_tnd_bwd_from_fwd(
            q,
            k,
            v,
            dout,
            out,
            lse,
            hq,
            hkv,
            seqlens_q,
            seqlens_k,
            scale,
            SOFTCAP,
            DROPOUT_P,
            causal,
            window_size_left,
            window_size_right,
            gtype=GTYPE,
        )

    _check_case(name, run_bwd, golden_fn)
