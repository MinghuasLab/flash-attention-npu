"""
Golden：npu_fusion_attention 前向 + autograd 反传（参考梯度）
FlashAttention：仅 _flash_attn_backward（不走 autograd、不调 flash_attn_varlen_func）

bwd 输入：
  out         <- golden 前向输出
  softmax_lse <- golden 的 x_max + log(x_sum)；golden 为 NT8，转为 FA bwd 的 TN

用法（与 test_flash_attn_npu_v3_bwd.py 相同，改底部调用参数即可）:
  test_tnd_bwd_only_npu(3, 3, 128, [512, 33, 1111])
  test_bsnd_bwd_only_npu(16, 16, 128, 1, 99, 99)
  批跑：RUN_TND_BATCH / RUN_BSND_BATCH；确定性：DETERMINISTIC_SWITCH
"""

import gc
import json
import os
import random
import subprocess
import sys

import numpy as np
import torch
import torch_npu

from flash_attn_npu_v3.interface_arch22 import _flash_attn_backward

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)
from precision_compare import data_compare

torch.npu.set_device(3)
np.random.seed(3)
torch.manual_seed(3)

# --- 全局开关（TND / BSND 单次与批跑共用）---
# DETERMINISTIC_SWITCH = True
DETERMINISTIC_SWITCH = False
BATCH_CASE_SEED = 3


def npu_between_runs_cleanup():
    torch.npu.synchronize()
    gc.collect()
    if hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()


def set_case_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.npu, "manual_seed"):
        torch.npu.manual_seed(seed)
    if hasattr(torch.npu, "manual_seed_all"):
        torch.npu.manual_seed_all(seed)


def get_cu_seqlens(seqlens_list):
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int32)
    for i in range(len(seqlens_list) + 1):
        cu[i] = int(sum(seqlens_list[:i]))
    return cu


def make_compressed_atten_mask_npu(sparse_mode):
    """sparse 2/3/4：CANN 侧 2048 压缩上三角 bool。"""
    if sparse_mode in (2, 3, 4):
        return torch.triu(torch.ones([2048, 2048]), diagonal=1).to(torch.bool).npu()
    return None


def sample_band_pre_next(seq_q, seq_k):
    """band(sparse=4)：pre < seq_q，next < seq_k；seq 为 1 时取 0。"""
    pre_tocken = random.randint(0, seq_q - 1) if seq_q > 1 else 0
    next_tocken = random.randint(0, seq_k - 1) if seq_k > 1 else 0
    return pre_tocken, next_tocken


def get_pre_next_for_sparse(sparse_mode, seq_q, seq_k):
    if sparse_mode == 4:
        return sample_band_pre_next(seq_q, seq_k)
    if sparse_mode == 3:
        return 65536, 0
    return 65536, 65536


def resolve_fa_bwd_window(sparse_mode, pre_tocken, next_tocken):
    """
    返回 (is_causal, window_left, window_right) 供 _flash_attn_backward 使用。

    flash_api 会规范化 window 并**重新推导** is_causal（flash_api.cpp）：
      is_causal := (window_left < 0 and window_right == 0)
    因此 sparse=3 时传 False+(-1,0) 与 True+(-1,0) 效果相同；这里对 sparse=3 返回
    True，仅表示语义上的 causal，与 golden sparse_mode=3 一致。
    """
    if sparse_mode == 4:
        return False, pre_tocken, next_tocken
    if sparse_mode == 3:
        return True, -1, 0
    return False, -1, -1


def lse_from_golden_max_sum(x_max, x_sum, layout):
    """TND: NT8 -> TN; BSND: BNS8/BNS -> BNS."""
    lse = x_max.to(torch.float32) + torch.log(x_sum.to(torch.float32).clamp_min(1e-20))
    if layout == "tnd":
        if lse.dim() != 3:
            raise ValueError(
                f"TND 期望 x_max/x_sum 为 NT8 (N,T,8)，实际 dim={lse.dim()} shape={tuple(lse.shape)}"
            )
        lse = lse[..., 0].transpose(0, 1)
    else:
        if lse.dim() == 4:
            lse = lse[..., 0]
        elif lse.dim() == 3:
            pass
        else:
            raise ValueError(
                f"BSND 期望 x_max/x_sum 为 BNS8 或 BNS，实际 dim={lse.dim()} shape={tuple(lse.shape)}"
            )
    return lse.contiguous()


def _to_numpy_f32(tensor):
    return np.array(
        tensor.detach().float().cpu().contiguous().numpy(),
        dtype=np.float32,
        copy=True,
    )


def _compare_grad(name, fa_grad, golden_grad):
    fa_np = _to_numpy_f32(fa_grad)
    gold_np = _to_numpy_f32(golden_grad)
    print(f"\n{'=' * 72}")
    print(f"[精度] {name}  (FA bwd vs golden autograd)")
    print(
        f"  FA numpy: type={type(fa_np).__name__}, dtype={fa_np.dtype}, shape={fa_np.shape}; "
        f"golden numpy: type={type(gold_np).__name__}, dtype={gold_np.dtype}, shape={gold_np.shape}"
    )
    fa_finite_mask = np.isfinite(fa_np)
    gold_finite_mask = np.isfinite(gold_np)
    fa_nonfinite = ~fa_finite_mask
    gold_nonfinite = ~gold_finite_mask
    if fa_nonfinite.any() or gold_nonfinite.any():
        print(
            f"  [非有限值] FA: nan={np.isnan(fa_np).sum()} inf={np.isinf(fa_np).sum()}  "
            f"golden: nan={np.isnan(gold_np).sum()} inf={np.isinf(gold_np).sum()}"
        )
    result, pct, max_err = data_compare(fa_np, gold_np)
    if fa_nonfinite.any() or gold_nonfinite.any():
        result = "failed"
    print(f"  => {result}, pass_rate={pct:.4f}%, max_rel_err={max_err:.6g}")
    fa_finite = np.abs(fa_np[fa_finite_mask])
    gold_finite = np.abs(gold_np[gold_finite_mask])
    fa_max = fa_finite.max() if fa_finite.size else np.nan
    gold_max = gold_finite.max() if gold_finite.size else np.nan
    print(f"  finite |max| FA={fa_max:.6g} golden={gold_max:.6g}")
    return result


def test_tnd_bwd_only_npu(nheads, nheads_k, headdim, list_seq, sparse_mode=0):
    """
    TND 变长：仅测 FA bwd，out / softmax_lse 来自 golden 前向。

    Args:
        nheads: Q head 数
        nheads_k: KV head 数（需满足 nheads % nheads_k == 0）
        headdim: head 维
        list_seq: 每个 batch 的真实序列长度，如 [512, 33, 1111]
    """
    scale = 1 / (headdim ** 0.5)
    seqlens_list_q = np.array(list_seq)
    seqlens_list_k = np.array(list_seq)

    max_seqlen_q = int(np.max(seqlens_list_q))
    max_seqlen_k = int(np.max(seqlens_list_k))
    cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
    cu_seqlens_k = get_cu_seqlens(seqlens_list_k)
    total_q = int(seqlens_list_q.sum())
    total_k = int(seqlens_list_k.sum())
    cu_seq_len_list = cu_seqlens_q[1:].cpu().numpy().tolist()
    cu_seq_kvlen_list = cu_seqlens_k[1:].cpu().numpy().tolist()
    print("total_q: ", total_q)
    print("total_k: ", total_k)
    print("cu_seq_len_list is ", cu_seq_len_list)
    print("cu_seq_kvlen_list is ", cu_seq_kvlen_list)

    if DETERMINISTIC_SWITCH:
        torch.use_deterministic_algorithms(True)

    pttype = torch.bfloat16
    limit = 2
    q = (limit * (torch.rand([total_q, nheads, headdim]) - 0.5)).to(pttype)
    k = (limit * (torch.rand([total_k, nheads_k, headdim]) - 0.5)).to(pttype)
    v = (limit * (torch.rand([total_k, nheads_k, headdim]) - 0.5)).to(pttype)
    dout = (limit * (torch.rand([total_q, nheads, headdim]) - 0.5)).to(pttype)
    print("q.shape ", q.shape)
    print("k.shape ", k.shape)
    print("v.shape ", v.shape)
    print("dout.shape ", dout.shape)

    pre_tocken, next_tocken = get_pre_next_for_sparse(
        sparse_mode, max_seqlen_q, max_seqlen_k
    )
    print(
        f"pre_tockens={pre_tocken} (<maxSq={max_seqlen_q}), "
        f"next_tockens={next_tocken} (<maxSkv={max_seqlen_k})"
    )
    causal_switch, window_left, window_right = resolve_fa_bwd_window(
        sparse_mode, pre_tocken, next_tocken
    )

    atten_mask_npu = make_compressed_atten_mask_npu(sparse_mode)

    q = q.npu()
    k = k.npu()
    v = v.npu()
    dout = dout.npu()

    # --- golden fwd + autograd bwd ---
    q_g = q.detach().clone().requires_grad_(True)
    k_g = k.detach().clone().requires_grad_(True)
    v_g = v.detach().clone().requires_grad_(True)
    torch.npu.synchronize()
    npu_rst = torch_npu.npu_fusion_attention(
        q_g, k_g, v_g, nheads,
        pse=None,
        padding_mask=None,
        atten_mask=atten_mask_npu,
        scale=scale,
        keep_prob=1.0,
        input_layout="TND",
        actual_seq_qlen=tuple(cu_seq_len_list),
        actual_seq_kvlen=tuple(cu_seq_kvlen_list),
        pre_tockens=pre_tocken,
        next_tockens=next_tocken,
        inner_precise=0,
        sparse_mode=sparse_mode,
        prefix=None,
    )
    out_npu = npu_rst[0]
    x_max_npu = npu_rst[1]
    x_sum_npu = npu_rst[2]
    print("golden x_max.shape ", tuple(x_max_npu.shape), " x_sum.shape ", tuple(x_sum_npu.shape))
    softmax_lse = lse_from_golden_max_sum(x_max_npu, x_sum_npu, "tnd")
    print("softmax_lse.shape ", tuple(softmax_lse.shape))
    torch.npu.synchronize()
    out_npu.backward(dout)
    dq_golden_npu = q_g.grad
    dk_golden_npu = k_g.grad
    dv_golden_npu = v_g.grad
    torch.npu.synchronize()

    cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32).npu()
    cu_seqlens_k = cu_seqlens_k.to(dtype=torch.int32).npu()
    print("cu_seqlens_q is ", cu_seqlens_q)
    print("cu_seqlens_k is ", cu_seqlens_k)

    # --- FA bwd only（out / lse 来自 golden）---
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _flash_attn_backward(
        dout, q, k, v, out_npu.detach(), softmax_lse,
        cu_seqlens_q, cu_seqlens_k,
        None, None, max_seqlen_q, max_seqlen_k,
        dq, dk, dv,
        scale, causal_switch, window_left, window_right,
        0.0, DETERMINISTIC_SWITCH, 0,
    )
    torch.npu.synchronize()

    dq_cmp = _compare_grad("dq", dq, dq_golden_npu)
    dk_cmp = _compare_grad("dk", dk, dk_golden_npu)
    dv_cmp = _compare_grad("dv", dv, dv_golden_npu)

    cmp_results = {"dq": dq_cmp, "dk": dk_cmp, "dv": dv_cmp}
    overall_success = all(result == "success" for result in cmp_results.values())
    print("=" * 72)
    print(
        f"TND bwd-only overall: {'PASS' if overall_success else 'FAILED'}  "
        f"deterministic={DETERMINISTIC_SWITCH}  {cmp_results}"
    )
    print("=" * 72)
    return {"overall_success": overall_success, "cmp_results": cmp_results}


def test_bsnd_bwd_only_npu(nheads, nheads_k, headdim, batch, seq_q, seq_k, sparse_mode=0):
    """
    BSND 定长：仅测 FA bwd，out / softmax_lse 来自 golden 前向。

    Args:
        nheads, nheads_k, headdim: 同 TND
        batch: B
        seq_q, seq_k: Q/KV 序列长度
    """
    scale = 1 / (headdim ** 0.5)

    if DETERMINISTIC_SWITCH:
        torch.use_deterministic_algorithms(True)

    pttype = torch.float16
    limit = 2
    q = (limit * (torch.rand([batch, seq_q, nheads, headdim]) - 0.5)).to(pttype)
    k = (limit * (torch.rand([batch, seq_k, nheads_k, headdim]) - 0.5)).to(pttype)
    v = (limit * (torch.rand([batch, seq_k, nheads_k, headdim]) - 0.5)).to(pttype)
    dout = (limit * (torch.rand([batch, seq_q, nheads, headdim]) - 0.5)).to(pttype)
    print("q.shape ", q.shape)
    print("k.shape ", k.shape)
    print("v.shape ", v.shape)
    print("dout.shape ", dout.shape)

    pre_tocken, next_tocken = get_pre_next_for_sparse(sparse_mode, seq_q, seq_k)
    print(f"pre_tockens={pre_tocken} (<S1={seq_q}), next_tockens={next_tocken} (<S2={seq_k})")
    causal_switch, window_left, window_right = resolve_fa_bwd_window(
        sparse_mode, pre_tocken, next_tocken
    )

    atten_mask_npu = make_compressed_atten_mask_npu(sparse_mode)

    q = q.npu()
    k = k.npu()
    v = v.npu()
    dout = dout.npu()

    q_g = q.detach().clone().requires_grad_(True)
    k_g = k.detach().clone().requires_grad_(True)
    v_g = v.detach().clone().requires_grad_(True)
    torch.npu.synchronize()
    npu_rst = torch_npu.npu_fusion_attention(
        q_g, k_g, v_g, nheads,
        pse=None,
        padding_mask=None,
        atten_mask=atten_mask_npu,
        scale=scale,
        keep_prob=1.0,
        input_layout="BSND",
        pre_tockens=pre_tocken,
        next_tockens=next_tocken,
        inner_precise=0,
        sparse_mode=sparse_mode,
        prefix=None,
    )
    out_npu = npu_rst[0]
    x_max_npu = npu_rst[1]
    x_sum_npu = npu_rst[2]
    print("golden x_max.shape ", tuple(x_max_npu.shape), " x_sum.shape ", tuple(x_sum_npu.shape))
    softmax_lse = lse_from_golden_max_sum(x_max_npu, x_sum_npu, "bsnd")
    print("softmax_lse.shape ", tuple(softmax_lse.shape))
    torch.npu.synchronize()
    out_npu.backward(dout)
    dq_golden_npu = q_g.grad
    dk_golden_npu = k_g.grad
    dv_golden_npu = v_g.grad
    torch.npu.synchronize()

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _flash_attn_backward(
        dout, q, k, v, out_npu.detach(), softmax_lse,
        None, None,
        None, None, None, None,
        dq, dk, dv,
        scale, causal_switch, window_left, window_right,
        0.0, DETERMINISTIC_SWITCH, 0,
    )
    torch.npu.synchronize()

    dq_cmp = _compare_grad("dq", dq, dq_golden_npu)
    dk_cmp = _compare_grad("dk", dk, dk_golden_npu)
    dv_cmp = _compare_grad("dv", dv, dv_golden_npu)

    cmp_results = {"dq": dq_cmp, "dk": dk_cmp, "dv": dv_cmp}
    overall_success = all(result == "success" for result in cmp_results.values())
    print("=" * 72)
    print(
        f"BSND bwd-only overall: {'PASS' if overall_success else 'FAILED'}  "
        f"deterministic={DETERMINISTIC_SWITCH}  {cmp_results}"
    )
    print("=" * 72)
    return {"overall_success": overall_success, "cmp_results": cmp_results}

def run_single_from_argv():
    if len(sys.argv) < 3 or sys.argv[1] not in ("--single-tnd", "--single-bsnd"):
        return
    params = json.loads(sys.argv[2])
    set_case_seed(BATCH_CASE_SEED)
    if sys.argv[1] == "--single-tnd":
        result = test_tnd_bwd_only_npu(
            params["nheads"],
            params["nheads_k"],
            params["headdim"],
            params["list_seq"],
            params.get("sparse_mode", 0),
        )
        print("SINGLE_TND_RESULT_JSON:", json.dumps(result, sort_keys=True))
    else:
        result = test_bsnd_bwd_only_npu(
            params["nheads"],
            params["nheads_k"],
            params["headdim"],
            params["batch"],
            params["seq_q"],
            params["seq_k"],
            params.get("sparse_mode", 0),
        )
        print("SINGLE_BSND_RESULT_JSON:", json.dumps(result, sort_keys=True))
    sys.exit(0 if result["overall_success"] else 2)


def run_case_subprocess(mode, params):
    result_prefix = f"SINGLE_{mode.upper()}_RESULT_JSON:"
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), f"--single-{mode}", json.dumps(params)],
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(result_prefix):
            return json.loads(line.split(":", 1)[1].strip())
    raise RuntimeError(f"subprocess failed with returncode={proc.returncode}")


run_single_from_argv()


# --- 单次用例（与 test_flash_attn_npu_v3_bwd.py 一致，改参数后取消注释）---

# test_tnd_bwd_only_npu(3, 3, 128, [512, 33, 1111])
# test_tnd_bwd_only_npu(3, 3, 128, [512, 33, 1111], sparse_mode=3)
# test_tnd_bwd_only_npu(28, 7, 256, [550, 1498, 400, 147, 557, 2028, 889, 1057])
# test_tnd_bwd_only_npu(8, 2, 128, [1204])

# test_bsnd_bwd_only_npu(2, 2, 256, 17, 9025, 3840)
# test_bsnd_bwd_only_npu(2, 2, 256, 17, 9025, 3840, sparse_mode=3)
# test_bsnd_bwd_only_npu(10, 10, 64, 10, 61, 61)
# test_bsnd_bwd_only_npu(32, 8, 256, 26, 9453, 7285)
# test_bsnd_bwd_only_npu(9, 3, 64, 3, 2228, 8109)
# test_bsnd_bwd_only_npu(36, 9, 192, 35, 9586, 6678)


# test_bsnd_bwd_only_npu(4, 2, 64, 20, 509, 4415)
# test_bsnd_bwd_only_npu(16, 8, 192, 44, 7147, 4933)
# test_bsnd_bwd_only_npu(10, 5, 192, 35, 9371, 9325)
# test_bsnd_bwd_only_npu(40, 10, 256, 45, 6986, 4203)
# test_bsnd_bwd_only_npu(3, 1, 128, 3, 3679, 986)


# --- TND 批跑：随机 nheads / headdim / 变长 list_seq ---
RUN_TND_BATCH = True
# RUN_TND_BATCH = False
total_runs_tnd = 1
tnd_batch_seed = BATCH_CASE_SEED
TND_BATCH_SPARSE_MODE = 4

if RUN_TND_BATCH:
    param_rng = random.Random(tnd_batch_seed)

    success_count_tnd = 0
    failed_runs_tnd = []
    for i in range(total_runs_tnd):
        run_idx = i + 1
        nheads_k = param_rng.randint(1, 10)
        nheads = param_rng.randint(1, 4) * nheads_k
        headdim = 64 * param_rng.randint(1, 4)
        num_batches = param_rng.randint(1, 50)
        list_seq = [param_rng.randint(1, 10000) for _ in range(num_batches)]
        run_params = {
            "nheads": nheads,
            "nheads_k": nheads_k,
            "headdim": headdim,
            "list_seq": list_seq,
            "total_q": int(sum(list_seq)),
            "max_seqlen": max(list_seq),
            "sparse_mode": TND_BATCH_SPARSE_MODE,
        }
        try:
            npu_between_runs_cleanup()
            print(f"\n[TND batch {run_idx}/{total_runs_tnd}] params: {run_params}")
            run_result = run_case_subprocess("tnd", run_params)
            if run_result["overall_success"]:
                success_count_tnd += 1
                print(f"[TND batch {run_idx}/{total_runs_tnd}] success")
            else:
                failed_runs_tnd.append((run_idx, run_params, run_result["cmp_results"]))
                print(f"[TND batch {run_idx}/{total_runs_tnd}] failed: {run_result['cmp_results']}")
        except Exception as exc:
            failed_runs_tnd.append((run_idx, run_params, {"exception": str(exc)}))
            print(f"[TND batch {run_idx}/{total_runs_tnd}] failed with exception: {exc}")
        finally:
            npu_between_runs_cleanup()

    failed_count_tnd = total_runs_tnd - success_count_tnd
    print("=" * 80)
    print(
        f"TND batch summary: total={total_runs_tnd}, "
        f"success={success_count_tnd}, failed={failed_count_tnd}"
    )
    if failed_runs_tnd:
        print("Failed runs detail:")
        for run_idx, params, detail in failed_runs_tnd:
            print(f"  - run {run_idx}, params={params}, detail={detail}")
    print("=" * 80)

# --- BSND 批跑：随机 nheads / headdim / batch / seq_q / seq_k ---
# RUN_BSND_BATCH = True
RUN_BSND_BATCH = False
total_runs_bsnd = 5
bsnd_batch_seed = BATCH_CASE_SEED
BSND_BATCH_SPARSE_MODE = 0

if RUN_BSND_BATCH:
    param_rng = random.Random(bsnd_batch_seed)

    success_count_bsnd = 0
    failed_runs_bsnd = []
    for i in range(total_runs_bsnd):
        run_idx = i + 1
        nheads_k = param_rng.randint(1, 10)
        nheads = param_rng.randint(1, 4) * nheads_k
        headdim = 64 * param_rng.randint(1, 4)
        batch = param_rng.randint(1, 50)
        seq_q = param_rng.randint(1, 10000)
        seq_k = param_rng.randint(1, 10000)
        run_params = {
            "nheads": nheads,
            "nheads_k": nheads_k,
            "headdim": headdim,
            "batch": batch,
            "seq_q": seq_q,
            "seq_k": seq_k,
            "sparse_mode": BSND_BATCH_SPARSE_MODE,
        }
        try:
            npu_between_runs_cleanup()
            print(f"\n[BSND batch {run_idx}/{total_runs_bsnd}] params: {run_params}")
            run_result = run_case_subprocess("bsnd", run_params)
            if run_result["overall_success"]:
                success_count_bsnd += 1
                print(f"[BSND batch {run_idx}/{total_runs_bsnd}] success")
            else:
                failed_runs_bsnd.append((run_idx, run_params, run_result["cmp_results"]))
                print(f"[BSND batch {run_idx}/{total_runs_bsnd}] failed: {run_result['cmp_results']}")
        except Exception as exc:
            failed_runs_bsnd.append((run_idx, run_params, {"exception": str(exc)}))
            print(f"[BSND batch {run_idx}/{total_runs_bsnd}] failed with exception: {exc}")
        finally:
            npu_between_runs_cleanup()

    failed_count_bsnd = total_runs_bsnd - success_count_bsnd
    print("=" * 80)
    print(
        f"BSND batch summary: total={total_runs_bsnd}, "
        f"success={success_count_bsnd}, failed={failed_count_bsnd}"
    )
    if failed_runs_bsnd:
        print("Failed runs detail:")
        for run_idx, params, detail in failed_runs_bsnd:
            print(f"  - run {run_idx}, params={params}, detail={detail}")
    print("=" * 80)
