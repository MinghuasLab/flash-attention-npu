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

注意: causal 语义为右下对齐（j <= i + (sk - sq)），要求每个 batch sq <= sk；
sq > sk 时前 sq-sk 行没有任何可见 key，参考前向/标杆自身 NaN（数学退化，
非 kernel 问题），因此 causal case 均取 sq <= sk。

检查项：
- golden: dq/dk/dv 与 fa_small_op_golden 对比（rtol=atol=1e-2）；
- det=True: 重复 REPEAT 次逐位一致 (torch.equal)；
- det=False: 首跑与第二次运行的 max|diff| 仅打印（非确定性允许，不做断言）；
- det vs nondet: 两种模式的梯度互相接近（同一容差）。

用法:
  python tests/test_flash_attn_npu_v3_bwd_det.py                 # 全部
  python tests/test_flash_attn_npu_v3_bwd_det.py --repeat 20
  python tests/test_flash_attn_npu_v3_bwd_det.py --only bsnd_gqa
  python tests/test_flash_attn_npu_v3_bwd_det.py --mode det      # 只跑 det
  python tests/test_flash_attn_npu_v3_bwd_det.py --quick         # 跳过 large
"""

import argparse
import os
import sys

import torch
import torch_npu  # noqa: F401

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from flash_attn_npu_3.flash_attn_npu_interface_950 import _flash_attn_backward
from fa_small_op_golden import golden_bsnd_bwd_from_fwd, golden_tnd_bwd_from_fwd

INPUT_LIMIT = 2.0
DTYPE = torch.bfloat16
DEVICE = "npu"
RTOL_GOLDEN = 1e-2
ATOL_GOLDEN = 1e-2
DROPOUT_P = 0.0
SOFTCAP = 0.0
GTYPE = torch.float64

# (name, bsz, seqlen_q, seqlen_k, nheads_q, nheads_kv, headdim, causal, large)
BSND_CASES = [
    # ---- small ----
    ("bsnd_small_mha_causal",      1, 128, 128, 2, 2, 128, True,  False),
    ("bsnd_small_mha_nc",          1, 128, 128, 2, 2, 128, False, False),
    ("bsnd_small_gqa_tail_causal", 2, 200, 300, 4, 2, 64,  True,  False),
    ("bsnd_small_mqa_tail_nc",     2, 300, 500, 4, 1, 64,  False, False),
    ("bsnd_tiny_unaligned_causal", 2, 33,  77,  4, 4, 64,  True,  False),
    # ---- mid ----
    ("bsnd_mid_mha_square_causal", 1, 1024, 1024, 8, 8, 128, True,  False),
    ("bsnd_mid_mha_square_nc",     1, 1024, 1024, 8, 8, 128, False, False),
    ("bsnd_mid_mha_rect_causal",   2, 333,  777,  4, 4, 64,  True,  False),
    ("bsnd_mid_mha_rect_nc",       2, 777,  333,  4, 4, 64,  False, False),
    ("bsnd_mid_gqa_square_causal", 1, 1024, 1024, 8, 2, 128, True,  False),
    ("bsnd_mid_gqa_square_nc",     1, 1024, 1024, 8, 2, 128, False, False),
    ("bsnd_mid_mqa_causal",        2, 300,  500,  4, 1, 64,  True,  False),
    ("bsnd_mid_mqa_nc",            2, 300,  500,  4, 1, 64,  False, False),
    ("bsnd_mid_gqa_rect_causal",   2, 500,  900,  6, 3, 128, True,  False),
    ("bsnd_mid_gqa_rect_nc",       2, 500,  900,  6, 3, 128, False, False),
    ("bsnd_mid_mha_hd64_nc",       2, 777,  333,  6, 3, 64,  False, False),
    # ---- long / large ----
    ("bsnd_long_mha_causal",       1, 2048, 2048, 8, 8, 128, True,  True),
    ("bsnd_long_mha_nc",           1, 2048, 2048, 8, 8, 128, False, True),
    ("bsnd_long_gqa_causal",       1, 2048, 2048, 8, 2, 128, True,  True),
    ("bsnd_large_mha_causal",      1, 4096, 4096, 4, 4, 128, True,  True),
    ("bsnd_large_mha_nc",          1, 4096, 4096, 4, 4, 128, False, True),
]

# (name, cu_seqlens_q, cu_seqlens_k, nheads_q, nheads_kv, headdim, causal, large)
VARLEN_CASES = [
    # ---- small / ragged ----
    ("tnd_small_mha_causal",   [0, 256, 640],          [0, 256, 640],         4, 4, 128, True,  False),
    ("tnd_small_mha_nc",       [0, 256, 640],          [0, 256, 640],         4, 4, 128, False, False),
    ("tnd_ragged_mqa_causal",  [0, 100, 900],          [0, 300, 1600],        4, 1, 64,  True,  False),
    ("tnd_ragged_mqa_nc",      [0, 100, 900],          [0, 300, 700],         4, 1, 64,  False, False),
    ("tnd_ragged_mha_nc",      [0, 100, 900],          [0, 300, 700],         4, 4, 64,  False, False),
    ("tnd_ragged_gqa_causal",  [0, 512, 1536],         [0, 768, 2048],        4, 2, 128, True,  False),
    ("tnd_ragged_gqa_nc",      [0, 512, 1536],         [0, 768, 2048],        4, 2, 128, False, False),
    # ---- equal-length (left-up causal schedule) ----
    ("tnd_eq_mha_causal",      [0, 2048, 4096],        [0, 2048, 4096],       8, 8, 128, True,  True),
    ("tnd_eq_mha_nc",          [0, 2048, 4096],        [0, 2048, 4096],       8, 8, 128, False, True),
    ("tnd_eq_gqa_causal",      [0, 1024, 2048],        [0, 1024, 2048],       8, 2, 128, True,  True),
    ("tnd_eq_gqa_nc",          [0, 1024, 2048],        [0, 1024, 2048],       8, 2, 128, False, True),
    # ---- pack (multi-sequence) ----
    ("tnd_pack8_mha_causal",   [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096],
                               [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096], 8, 8, 128, True,  True),
    ("tnd_pack8_mha_nc",       [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096],
                               [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096], 8, 8, 128, False, True),
    # ---- large single ----
    ("tnd_large4_mha_causal",  [0, 2048, 4096, 6144, 8192],
                               [0, 2048, 4096, 6144, 8192],                       4, 4, 128, True,  True),
    ("tnd_large_mha_nc",       [0, 4096],              [0, 4096],             4, 4, 128, False, True),
    ("tnd_large4_gqa_causal",  [0, 1024, 2048, 3072, 4096],
                               [0, 1024, 2048, 3072, 4096],                       8, 2, 128, True,  True),
    ("tnd_large4_mqa_causal",  [0, 2048, 4096, 6144, 8192],
                               [0, 2048, 4096, 6144, 8192],                       4, 1, 64,  True,  True),
]


def rand_inputs(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return (INPUT_LIMIT * (torch.rand(shape, generator=g) - 0.5)).to(DTYPE).to(DEVICE)


def torch_ref_fwd_bsnd(q, k, v, scale, causal):
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
    if causal:
        i = torch.arange(sq).unsqueeze(1)
        j = torch.arange(sk).unsqueeze(0)
        allow = j <= i + (sk - sq)
        s = s.masked_fill(~allow, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)                     # (B,Hq,Sq) fp32
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, vb)                              # (B,Hq,Sq,D)
    out = o.permute(0, 2, 1, 3).to(q.dtype)
    return out.to(q.device), lse.to(q.device)


def torch_ref_fwd_tnd(q, k, v, cu_q, cu_k, scale, causal):
    """varlen 参考前向：逐 batch 切片复用 BSND 逻辑。

    返回 out:(total_q,Hq,D), lse:(Hq,total_q) fp32。
    """
    outs, lses = [], []
    for i in range(len(cu_q) - 1):
        qi = q[cu_q[i]:cu_q[i + 1]].unsqueeze(0)
        ki = k[cu_k[i]:cu_k[i + 1]].unsqueeze(0)
        vi = v[cu_k[i]:cu_k[i + 1]].unsqueeze(0)
        oi, li = torch_ref_fwd_bsnd(qi, ki, vi, scale, causal)
        outs.append(oi.squeeze(0))
        lses.append(li.squeeze(0))
    return torch.cat(outs, dim=0), torch.cat(lses, dim=1)


def run_bwd_bsnd(q, k, v, dout, out, lse, scale, causal, det):
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_backward(
        dout, q, k, v, out, lse,
        None, None, None, None, None, None,
        dq, dk, dv,
        scale, causal, -1, -1, SOFTCAP, det, 0,
    )
    torch.npu.synchronize()
    return dq, dk, dv


def run_bwd_varlen(q, k, v, dout, out, lse, cu_q_t, cu_k_t, max_sq, max_sk,
                   scale, causal, det):
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_backward(
        dout, q, k, v, out, lse,
        cu_q_t, cu_k_t, None, None,
        max_sq, max_sk,
        dq, dk, dv,
        scale, causal, -1, -1, SOFTCAP, det, 0,
    )
    torch.npu.synchronize()
    return dq, dk, dv


def max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def check_golden(name, mode, actual, golden):
    try:
        torch.testing.assert_close(
            actual.cpu(), golden.cpu(), rtol=RTOL_GOLDEN, atol=ATOL_GOLDEN)
        return True
    except AssertionError:
        diff = (actual.float().cpu() - golden.float().cpu()).abs()
        print(f"[FAIL] {name} [{mode}]: grad 与 golden 不符, "
              f"max|diff|={diff.max().item():.6e}")
        return False


def check_case(name, run_bwd, golden_fn, repeat, modes):
    """run_bwd(det) -> (dq,dk,dv); golden_fn() -> (dq,dk,dv)。"""
    results = {}
    ok = True
    golden = golden_fn()

    for det in modes:
        tag = "det" if det else "nd"
        grads = run_bwd(det)
        results[tag] = grads
        for gname, g, gold in zip(("dq", "dk", "dv"), grads, golden):
            ok &= check_golden(name, tag, g, gold)

    if "det" in results:
        base = results["det"]
        for it in range(repeat):
            cur = run_bwd(True)
            for gname, a, b in zip(("dq", "dk", "dv"), cur, base):
                if not torch.equal(a, b):
                    print(f"[FAIL] {name} [det] iter {it}: {gname} 不一致, "
                          f"max|diff|={max_diff(a, b):.6e}")
                    ok = False
                    break
            if not ok:
                break

    if "nd" in results:
        cur = run_bwd(False)
        d = max(max_diff(a, b) for a, b in zip(cur, results["nd"]))
        print(f"[INFO] {name} [nd] 两次运行 max|diff|={d:.6e}")

    if "det" in results and "nd" in results:
        for gname, a, b in zip(("dq", "dk", "dv"),
                               results["det"], results["nd"]):
            try:
                torch.testing.assert_close(
                    a.cpu(), b.cpu(), rtol=RTOL_GOLDEN, atol=ATOL_GOLDEN)
            except AssertionError:
                print(f"[FAIL] {name}: det vs nondet {gname} 不符, "
                      f"max|diff|={max_diff(a, b):.6e}")
                ok = False

    if ok:
        print(f"[PASS] {name}: golden 一致"
              + (f" + {repeat} 次 det 逐位一致" if "det" in results else "")
              + (" + det/nondet 互相一致" if len(results) == 2 else ""))
    return ok


def run_bsnd_case(case, repeat, modes):
    name, bsz, sq, sk, hq, hkv, hd, causal = case[:8]
    scale = hd ** (-0.5)
    q = rand_inputs((bsz, sq, hq, hd), 42)
    k = rand_inputs((bsz, sk, hkv, hd), 43)
    v = rand_inputs((bsz, sk, hkv, hd), 44)
    dout = rand_inputs((bsz, sq, hq, hd), 45)
    out, lse = torch_ref_fwd_bsnd(q, k, v, scale, causal)

    def run_bwd(det):
        return run_bwd_bsnd(q, k, v, dout, out, lse, scale, causal, det)

    def golden_fn():
        return golden_bsnd_bwd_from_fwd(
            q, k, v, dout, out, lse, hq, hkv, scale, SOFTCAP, DROPOUT_P,
            causal, -1, -1, gtype=GTYPE)

    return check_case(name, run_bwd, golden_fn, repeat, modes)


def run_varlen_case(case, repeat, modes):
    name, cu_q, cu_k, hq, hkv, hd, causal = case[:7]
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
    out, lse = torch_ref_fwd_tnd(q, k, v, cu_q, cu_k, scale, causal)

    def run_bwd(det):
        return run_bwd_varlen(
            q, k, v, dout, out, lse, cu_q_t, cu_k_t, max_sq, max_sk,
            scale, causal, det)

    def golden_fn():
        seqlens_q = [cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1)]
        seqlens_k = [cu_k[i + 1] - cu_k[i] for i in range(len(cu_k) - 1)]
        return golden_tnd_bwd_from_fwd(
            q, k, v, dout, out, lse, hq, hkv, seqlens_q, seqlens_k,
            scale, SOFTCAP, DROPOUT_P, causal, -1, -1, gtype=GTYPE)

    return check_case(name, run_bwd, golden_fn, repeat, modes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=20,
                        help="det 模式重复次数（逐位一致检查）")
    parser.add_argument("--only", type=str, default=None,
                        help="只跑名字包含该子串的用例")
    parser.add_argument("--mode", type=str, default="both",
                        choices=("both", "det", "nd"),
                        help="跑 det / nd / 两者")
    parser.add_argument("--quick", action="store_true",
                        help="跳过 large 用例（4096 级）")
    args = parser.parse_args()

    modes = {"both": (True, False), "det": (True,), "nd": (False,)}[args.mode]

    def selected(name, large):
        if args.only and args.only not in name:
            return False
        if args.quick and large:
            return False
        return True

    ok = True
    cases = [c for c in BSND_CASES if selected(c[0], c[8])]
    cases += [c for c in VARLEN_CASES if selected(c[0], c[7])]
    for case in cases:
        if case[0].startswith("bsnd"):
            ok &= run_bsnd_case(case, args.repeat, modes)
        else:
            ok &= run_varlen_case(case, args.repeat, modes)
        sys.stdout.flush()

    print("=" * 40)
    print("ALL PASS" if ok else "SOME CASES FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
