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
import time
import torch
import torch_npu

# 950 后端扩展在分机编译 (FLASH_ATTN_BUILD_NPU=910) 的 910 机器上不存在,
# 模块级 import 会在 collection 阶段报错, 故 import 前整文件跳过 (同 v2 模式)。
if "Ascend950" not in (torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""):
    pytest.skip("flash_attn_npu_3 950 backward tests require Ascend950", allow_module_level=True)

from flash_attn_npu_3.flash_attn_npu_interface_950 import _flash_attn_backward
from tests import fa_small_op_golden
from tests.fa_small_op_golden import golden_bsnd_bwd_from_fwd, golden_tnd_bwd_from_fwd
from tests.common.golden_cache import get_or_compute_golden

INPUT_LIMIT = 2.0
DTYPE = torch.bfloat16
DEVICE = "npu"
RTOL_GOLDEN = 1e-2
ATOL_GOLDEN = 1e-2
DROPOUT_P = 0.0
SOFTCAP = 0.0
GTYPE = torch.float64
REPEAT = 20  # det 模式逐位一致的重复次数

from tests.common import timing as _timing

# 归因计时, kernel 调用记入 backward, 逐位与容差校验记入 compare
_raw_flash_attn_backward = _flash_attn_backward


def _flash_attn_backward(*args, **kwargs):
    started = time.perf_counter()
    out = _raw_flash_attn_backward(*args, **kwargs)
    _timing.record("backward", time.perf_counter() - started)
    return out


def _timed_equal(a, b):
    started = time.perf_counter()
    out = torch.equal(a, b)
    _timing.record("compare", time.perf_counter() - started)
    return out


def _gold_close(got, gold, *, msg):
    started = time.perf_counter()
    try:
        torch.testing.assert_close(got.cpu(), gold.cpu(), rtol=RTOL_GOLDEN, atol=ATOL_GOLDEN, msg=msg)
    finally:
        _timing.record("compare", time.perf_counter() - started)

# (name, bsz, seqlen_q, seqlen_k, nheads_q, nheads_kv, headdim, causal, large)
BSND_CASES = [
    # ---- small ----
    ("bsnd_small_mha_causal", 1, 128, 128, 2, 2, 128, True, False),
    ("bsnd_small_mha_nc", 1, 128, 128, 2, 2, 128, False, False),
    ("bsnd_small_gqa_tail_causal", 2, 200, 300, 4, 2, 64, True, False),
    ("bsnd_small_mqa_tail_nc", 2, 300, 500, 4, 1, 64, False, False),
    ("bsnd_tiny_unaligned_causal", 2, 33, 77, 4, 4, 64, True, False),
    # ---- mid ----
    ("bsnd_mid_mha_square_causal", 1, 1024, 1024, 8, 8, 128, True, False),
    ("bsnd_mid_mha_square_nc", 1, 1024, 1024, 8, 8, 128, False, False),
    ("bsnd_mid_mha_rect_causal", 2, 333, 777, 4, 4, 64, True, False),
    ("bsnd_mid_mha_rect_nc", 2, 777, 333, 4, 4, 64, False, False),
    ("bsnd_mid_gqa_square_causal", 1, 1024, 1024, 8, 2, 128, True, False),
    ("bsnd_mid_gqa_square_nc", 1, 1024, 1024, 8, 2, 128, False, False),
    ("bsnd_mid_mqa_causal", 2, 300, 500, 4, 1, 64, True, False),
    ("bsnd_mid_mqa_nc", 2, 300, 500, 4, 1, 64, False, False),
    ("bsnd_mid_gqa_rect_causal", 2, 500, 900, 6, 3, 128, True, False),
    ("bsnd_mid_gqa_rect_nc", 2, 500, 900, 6, 3, 128, False, False),
    ("bsnd_mid_mha_hd64_nc", 2, 777, 333, 6, 3, 64, False, False),
    # ---- long / large ----
    ("bsnd_long_mha_causal", 1, 2048, 2048, 8, 8, 128, True, True),
    ("bsnd_long_mha_nc", 1, 2048, 2048, 8, 8, 128, False, True),
    ("bsnd_long_gqa_causal", 1, 2048, 2048, 8, 2, 128, True, True),
    ("bsnd_large_mha_causal", 1, 4096, 4096, 4, 4, 128, True, True),
    ("bsnd_large_mha_nc", 1, 4096, 4096, 4, 4, 128, False, True),
]

# (name, cu_seqlens_q, cu_seqlens_k, nheads_q, nheads_kv, headdim, causal, large)
VARLEN_CASES = [
    # ---- small / ragged ----
    ("tnd_small_mha_causal", [0, 256, 640], [0, 256, 640], 4, 4, 128, True, False),
    ("tnd_small_mha_nc", [0, 256, 640], [0, 256, 640], 4, 4, 128, False, False),
    ("tnd_ragged_mqa_causal", [0, 100, 900], [0, 300, 1600], 4, 1, 64, True, False),
    ("tnd_ragged_mqa_nc", [0, 100, 900], [0, 300, 700], 4, 1, 64, False, False),
    ("tnd_ragged_mha_nc", [0, 100, 900], [0, 300, 700], 4, 4, 64, False, False),
    ("tnd_ragged_gqa_causal", [0, 512, 1536], [0, 768, 2048], 4, 2, 128, True, False),
    ("tnd_ragged_gqa_nc", [0, 512, 1536], [0, 768, 2048], 4, 2, 128, False, False),
    # ---- equal-length (left-up causal schedule) ----
    ("tnd_eq_mha_causal", [0, 2048, 4096], [0, 2048, 4096], 8, 8, 128, True, True),
    ("tnd_eq_mha_nc", [0, 2048, 4096], [0, 2048, 4096], 8, 8, 128, False, True),
    ("tnd_eq_gqa_causal", [0, 1024, 2048], [0, 1024, 2048], 8, 2, 128, True, True),
    ("tnd_eq_gqa_nc", [0, 1024, 2048], [0, 1024, 2048], 8, 2, 128, False, True),
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
    ),
    ("tnd_large_mha_nc", [0, 4096], [0, 4096], 4, 4, 128, False, True),
    (
        "tnd_large4_gqa_causal",
        [0, 1024, 2048, 3072, 4096],
        [0, 1024, 2048, 3072, 4096],
        8,
        2,
        128,
        True,
        True,
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
    ),
]


def _is_ascend950():
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    return "Ascend950" in name


def rand_inputs(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return (INPUT_LIMIT * (torch.rand(shape, generator=g) - 0.5)).to(DTYPE).to(DEVICE)


def _cached_case_golden(nodeid, params, q, k, v, dout, compute):
    """按 case 缓存 CPU 参考前向 out/lse 与 golden 梯度, 命中时跳过全部 CPU 计算。"""
    def compute_fn():
        out, lse, dq, dk, dv = compute()
        keys = ("out", "lse", "dq", "dk", "dv")
        return {name: t.detach().to("cpu") for name, t in zip(keys, (out, lse, dq, dk, dv))}

    return get_or_compute_golden(
        nodeid=nodeid,
        metadata=params,
        inputs={"q": q, "k": k, "v": v, "dout": dout},
        compute_fn=compute_fn,
        expected_keys=("out", "lse", "dq", "dk", "dv"),
        source_files=(fa_small_op_golden.__file__,),
        test_source_files=(__file__,),
    )


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
    lse = torch.logsumexp(s, dim=-1)  # (B,Hq,Sq) fp32
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, vb)  # (B,Hq,Sq,D)
    out = o.permute(0, 2, 1, 3).to(q.dtype)
    return out.to(q.device), lse.to(q.device)


def torch_ref_fwd_tnd(q, k, v, cu_q, cu_k, scale, causal):
    """varlen 参考前向：逐 batch 切片复用 BSND 逻辑。

    返回 out:(total_q,Hq,D), lse:(Hq,total_q) fp32。
    """
    outs, lses = [], []
    for i in range(len(cu_q) - 1):
        qi = q[cu_q[i] : cu_q[i + 1]].unsqueeze(0)
        ki = k[cu_k[i] : cu_k[i + 1]].unsqueeze(0)
        vi = v[cu_k[i] : cu_k[i + 1]].unsqueeze(0)
        oi, li = torch_ref_fwd_bsnd(qi, ki, vi, scale, causal)
        outs.append(oi.squeeze(0))
        lses.append(li.squeeze(0))
    return torch.cat(outs, dim=0), torch.cat(lses, dim=1)


def run_bwd_bsnd(q, k, v, dout, out, lse, scale, causal, det):
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
        -1,
        -1,
        SOFTCAP,
        det,
        0,
    )
    torch.npu.synchronize()
    return dq, dk, dv


def run_bwd_varlen(q, k, v, dout, out, lse, cu_q_t, cu_k_t, max_sq, max_sk, scale, causal, det):
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
        -1,
        -1,
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
            _gold_close(got, gold, msg=f"{name} [{tag}] {gname} vs golden")

    for it in range(REPEAT):
        cur = run_bwd(True)
        for gname, a, b in zip(("dq", "dk", "dv"), cur, grads["det"]):
            assert _timed_equal(a, b), (
                f"{name} [det] iter {it + 1}/{REPEAT}: {gname} 不一致, "
                f"max|diff|={max_diff(a, b):.6e}"
            )

    nd_again = run_bwd(False)
    d = max(max_diff(a, b) for a, b in zip(nd_again, grads["nd"]))
    print(f"[INFO] {name} [nd] 两次运行 max|diff|={d:.6e}")

    for gname, a, b in zip(("dq", "dk", "dv"), grads["det"], grads["nd"]):
        _gold_close(a, b, msg=f"{name}: det vs nondet {gname}")

    print(f"[PASS] {name}: golden 一致 + {REPEAT} 次 det 逐位一致 + det/nondet 互相一致")


@pytest.mark.parametrize(
    "name, bsz, sq, sk, hq, hkv, hd, causal, large",
    BSND_CASES,
    ids=[c[0] for c in BSND_CASES],
)
@pytest.mark.skipif(not _is_ascend950(), reason="Ascend950 only")
def test_fa_bwd_det_bsnd(name, bsz, sq, sk, hq, hkv, hd, causal, large, request):
    scale = hd ** (-0.5)
    q = rand_inputs((bsz, sq, hq, hd), 42)
    k = rand_inputs((bsz, sk, hkv, hd), 43)
    v = rand_inputs((bsz, sk, hkv, hd), 44)
    dout = rand_inputs((bsz, sq, hq, hd), 45)

    def compute():
        out, lse = torch_ref_fwd_bsnd(q, k, v, scale, causal)
        dq, dk, dv = golden_bsnd_bwd_from_fwd(
            q, k, v, dout, out, lse, hq, hkv, scale, SOFTCAP, DROPOUT_P, causal, -1, -1, gtype=GTYPE
        )
        return out, lse, dq, dk, dv

    values = _cached_case_golden(
        request.node.nodeid,
        {"layout": "bsnd", "bsz": bsz, "sq": sq, "sk": sk, "hq": hq, "hkv": hkv,
         "hd": hd, "causal": causal, "seeds": [42, 43, 44, 45]},
        q, k, v, dout, compute,
    )
    out = values["out"].to(DEVICE)
    lse = values["lse"].to(DEVICE)
    golden = (values["dq"], values["dk"], values["dv"])

    def run_bwd(det):
        return run_bwd_bsnd(q, k, v, dout, out, lse, scale, causal, det)

    _check_case(name, run_bwd, lambda: golden)


@pytest.mark.parametrize(
    "name, cu_q, cu_k, hq, hkv, hd, causal, large",
    VARLEN_CASES,
    ids=[c[0] for c in VARLEN_CASES],
)
@pytest.mark.skipif(not _is_ascend950(), reason="Ascend950 only")
def test_fa_bwd_det_varlen(name, cu_q, cu_k, hq, hkv, hd, causal, large, request):
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

    def compute():
        out, lse = torch_ref_fwd_tnd(q, k, v, cu_q, cu_k, scale, causal)
        seqlens_q = [cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1)]
        seqlens_k = [cu_k[i + 1] - cu_k[i] for i in range(len(cu_k) - 1)]
        dq, dk, dv = golden_tnd_bwd_from_fwd(
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
            -1,
            -1,
            gtype=GTYPE,
        )
        return out, lse, dq, dk, dv

    values = _cached_case_golden(
        request.node.nodeid,
        {"layout": "tnd", "cu_q": list(cu_q), "cu_k": list(cu_k), "hq": hq,
         "hkv": hkv, "hd": hd, "causal": causal, "seeds": [42, 43, 44, 45]},
        q, k, v, dout, compute,
    )
    out = values["out"].to(DEVICE)
    lse = values["lse"].to(DEVICE)
    golden = (values["dq"], values["dk"], values["dv"])

    def run_bwd(det):
        return run_bwd_varlen(
            q, k, v, dout, out, lse, cu_q_t, cu_k_t, max_sq, max_sk, scale, causal, det
        )

    _check_case(name, run_bwd, lambda: golden)
