#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Varlen (TND) FlashAttention calling script via torch_npu.

Two entry points can be benchmarked on identical inputs:

* raw  -- torch_npu.npu_fusion_attention(_v2) with input_layout="TND" and
  actual_seq_qlen/actual_seq_kvlen (aclnnFlashAttentionVarLenScore).
* func -- flash_attn_varlen_func from flash_attn_npu/__init__.py, which wraps
  the same kernel behind the packaged flash-attn interface (incl. AICPU
  scheduler metadata generation, contiguous/padding handling and the
  autograd Function).

Backward dispatches to aclnnFlashAttentionUnpaddingScoreGrad via either
autograd (--grad auto) or explicit npu_fusion_attention_grad(_v2) (--grad v2,
raw impl only).

Usage:
    python fa_varlen_call.py --device 0 --dtype float16 --check
    python fa_varlen_call.py --impl both --grad auto --check
    python fa_varlen_call.py --impl func --grad auto --check
"""

import argparse
import math
import random
import time

import torch
import torch_npu

CASES = [
    {"name": "tnd_T131475_H8_nocausal_dropout0.1", "total_q": 128, "heads_q": 1,
     "total_kv": 1024, "heads_kv": 1, "head_dim": 128, "scale": 1.0,
     "causal": False, "batch": 4, "max_q": 32, "max_kv": 256, "dropout": 0.0},
    {"name": "tnd_T1536_H16_causal_dropout0.1", "total_q": 1536, "heads_q": 16,
     "total_kv": 1536, "heads_kv": 16, "head_dim": 128, "scale": 1.0,
     "causal": True, "batch": 512, "max_q": 279, "max_kv": 279, "dropout": 0.0},
    {"name": "tnd_Q1536_KV131475_H16_causal_dropout0.1", "total_q": 1536, "heads_q": 16,
     "total_kv": 131475, "heads_kv": 16, "head_dim": 128, "scale": 1.0,
     "causal": True, "batch": 512, "max_q": 3, "max_kv": 279, "dropout": 0.0},
    {"name": "tnd_T8928_H8_causal", "total_q": 8928, "heads_q": 8,
     "total_kv": 8928, "heads_kv": 8, "head_dim": 128, "scale": None,
     "causal": True, "batch": 32, "max_q": 279, "max_kv": 279, "dropout": 0.0},
    {"name": "tnd_T4464_GQA32x8_causal", "total_q": 4464, "heads_q": 32,
     "total_kv": 4464, "heads_kv": 8, "head_dim": 128, "scale": None,
     "causal": True, "batch": 16, "max_q": 279, "max_kv": 279, "dropout": 0.0},
    {"name": "ZJ_non_causal", "total_q": 10240, "heads_q": 24,
     "total_kv": 10240, "heads_kv": 4, "head_dim": 128, "scale": None,
     "causal": False, "batch": 320, "max_q": 32, "max_kv": 32, "dropout": 0.0},
    {"name": "ZJ_causal", "total_q": 10240, "heads_q": 24,
     "total_kv": 10240, "heads_kv": 4, "head_dim": 128, "scale": None,
     "causal": True, "batch": 320, "max_q": 32, "max_kv": 32, "dropout": 0.0},
]

COMPARE_TOL = 1e-2  # per-batch allclose tolerance used in precision checks


def gen_seq_lens(total, batch, max_len, seed, lower=None):
    rng = random.Random(seed)
    if lower is None:
        lower = [1] * batch
    assert len(lower) == batch and min(lower) >= 1 and max(lower) <= max_len
    assert sum(lower) <= total <= batch * max_len, \
        f"infeasible: total={total}, batch={batch}, max_len={max_len}"
    suf_lower = [0] * (batch + 1)
    for i in range(batch - 1, -1, -1):
        suf_lower[i] = suf_lower[i + 1] + lower[i]
    lens, remain = [], total
    for i in range(batch):
        slots_left = batch - i - 1
        lo = max(lower[i], remain - max_len * slots_left)
        hi = min(max_len, remain - suf_lower[i + 1])
        val = rng.randint(lo, hi)
        lens.append(val)
        remain -= val
    if lower == [1] * batch:
        rng.shuffle(lens)
    return lens


def make_inputs(case, dtype, device, seed):
    lens_q = gen_seq_lens(case["total_q"], case["batch"], case["max_q"], seed)
    if case["causal"] and case["total_kv"] == case["total_q"]:
        lens_kv = list(lens_q)
    elif case["causal"]:
        lens_kv = gen_seq_lens(case["total_kv"], case["batch"], case["max_kv"],
                               seed + 10000, lower=lens_q)
    else:
        lens_kv = gen_seq_lens(case["total_kv"], case["batch"], case["max_kv"], seed + 10000)
    cu_q = [0]
    for x in lens_q:
        cu_q.append(cu_q[-1] + x)
    cu_kv = [0]
    for x in lens_kv:
        cu_kv.append(cu_kv[-1] + x)

    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(case["total_q"], case["heads_q"], case["head_dim"], generator=g)
    k = torch.randn(case["total_kv"], case["heads_kv"], case["head_dim"], generator=g)
    v = torch.randn(case["total_kv"], case["heads_kv"], case["head_dim"], generator=g)
    q = q.to(dtype).to(device).contiguous()
    k = k.to(dtype).to(device).contiguous()
    v = v.to(dtype).to(device).contiguous()
    return {
        "q": q, "k": k, "v": v,
        "lens_q": lens_q, "lens_kv": lens_kv,
        "cu_q": torch.tensor(cu_q, dtype=torch.int32, device=device),
        "cu_kv": torch.tensor(cu_kv, dtype=torch.int32, device=device),
        "actual_q": tuple(cu_q[1:]), "actual_kv": tuple(cu_kv[1:]),
        "max_q": max(lens_q), "max_kv": max(lens_kv),
    }


def make_atten_mask(size, device):
    return torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1).to(device)


def call_fa_raw(q, k, v, case, actual_q, actual_kv, atten_mask, scale_value):
    common = {
        "pse": None,
        "atten_mask": atten_mask if case["causal"] else None,
        "keep_prob": 1.0 - case["dropout"],
        "inner_precise": 0,
        "actual_seq_qlen": actual_q,
        "actual_seq_kvlen": actual_kv,
        "sparse_mode": 3 if case["causal"] else 0,
    }
    if scale_value is not None:
        common["scale"] = scale_value
    # if hasattr(torch_npu, "npu_fusion_attention_v2"):
    #     try:
    #         res = torch_npu.npu_fusion_attention_v2(
    #             q, k, v, case["heads_q"], "TND",
    #             pre_tokens=2147483647, next_tokens=2147483647, **common)
    #         return res
    #     except TypeError:
    #         pass
    res = torch_npu.npu_fusion_attention(
        q, k, v, case["heads_q"], "TND",
        pre_tockens=2147483647, next_tockens=2147483647, **common)
    return res


def call_fa_func(q, k, v, case, cu_q, cu_kv, max_q, max_kv, scale_value):
    from flash_attn_npu import flash_attn_varlen_func
    print("lch func !!!!!!!!!!")
    return flash_attn_varlen_func(
        q, k, v,
        cu_q, cu_kv,
        max_q, max_kv,
        dropout_p=case["dropout"],
        softmax_scale=scale_value,
        causal=case["causal"],
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        block_table=None,
    )


def bench_once(fn):
    """Run fn exactly once and return (result, latency_ms)."""
    torch.npu.synchronize()
    start = time.perf_counter()
    res = fn()
    torch.npu.synchronize()
    return res, (time.perf_counter() - start) * 1000.0


def naive_reference(q, k, v, lens_q, lens_kv, scale, causal):
    qf, kf, vf = q.float().cpu(), k.float().cpu(), v.float().cpu()
    if kf.shape[1] != qf.shape[1]:
        rep = qf.shape[1] // kf.shape[1]
        kf = kf.repeat_interleave(rep, dim=1)
        vf = vf.repeat_interleave(rep, dim=1)
    out = torch.zeros_like(qf)
    qo = ko = 0
    for lq, lkv in zip(lens_q, lens_kv):
        qb = qf[qo:qo + lq].transpose(0, 1)
        kb = kf[ko:ko + lkv].transpose(0, 1)
        vb = vf[ko:ko + lkv].transpose(0, 1)
        score = torch.bmm(qb, kb.transpose(1, 2)) * scale
        if causal:
            rows = torch.arange(lq).unsqueeze(1)
            cols = torch.arange(lkv).unsqueeze(0)
            score = score.masked_fill(cols > rows + (lkv - lq), float("-inf"))
        prob = torch.softmax(score, dim=-1)
        out[qo:qo + lq] = torch.bmm(prob, vb).transpose(0, 1)
        qo += lq
        ko += lkv
    return out


def flash_attention_reference(q, k, v, lens_q, lens_kv, scale, causal,
                              block_size=512):
    """CPU golden simulating the flash-attention algorithm (online softmax).

    Unlike naive_reference (full softmax over the whole KV sequence), this
    processes KV in blocks of block_size and maintains running row-max /
    row-sum: for each KV block the block scores are computed, m/l/O are
    rescaled by exp(m_old - m_new) and accumulated, and O is divided by the
    final rowsum at the end. Results are bitwise-independent of the NPU
    kernel (different accumulation order), so it serves as a third-party
    reference check of the flash-attention numerics.
    """
    qf, kf, vf = q.float().cpu(), k.float().cpu(), v.float().cpu()
    if kf.shape[1] != qf.shape[1]:
        rep = qf.shape[1] // kf.shape[1]
        kf = kf.repeat_interleave(rep, dim=1)
        vf = vf.repeat_interleave(rep, dim=1)
    out = torch.zeros_like(qf)
    qo = ko = 0
    for lq, lkv in zip(lens_q, lens_kv):
        qb = qf[qo:qo + lq].transpose(0, 1)   # [H, Lq, D]
        kb = kf[ko:ko + lkv].transpose(0, 1)  # [H, Lkv, D]
        vb = vf[ko:ko + lkv].transpose(0, 1)  # [H, Lkv, D]
        causal_mask = None
        if causal:
            rows = torch.arange(lq).unsqueeze(1)
            cols = torch.arange(lkv).unsqueeze(0)
            # Same masking convention as naive_reference: query row i may
            # attend to KV positions i + (lkv - lq) and earlier.
            causal_mask = cols > rows + (lkv - lq)
        m = torch.full((qb.shape[0], lq), float("-inf"))  # running row max
        l = torch.zeros((qb.shape[0], lq))                # running row sum
        o = torch.zeros((qb.shape[0], lq, qb.shape[2]))   # running output
        for s in range(0, lkv, block_size):
            kc = kb[:, s:s + block_size]
            vc = vb[:, s:s + block_size]
            sc = torch.bmm(qb, kc.transpose(1, 2)) * scale
            if causal_mask is not None:
                sc = sc.masked_fill(causal_mask[:, s:s + block_size],
                                    float("-inf"))
            m_new = torch.maximum(m, sc.max(dim=-1).values)
            alpha = torch.exp(m - m_new)
            p = torch.exp(sc - m_new.unsqueeze(-1))
            l = l * alpha + p.sum(dim=-1)
            o = o * alpha.unsqueeze(-1) + torch.bmm(p, vc)
            m = m_new
        out[qo:qo + lq] = (o / l.unsqueeze(-1)).transpose(0, 1)
        qo += lq
        ko += lkv
    return out


def run_fwd_bwd_auto(fwd, q, k, v, dy):
    qa = q.detach().requires_grad_()
    ka = k.detach().requires_grad_()
    va = v.detach().requires_grad_()
    out = fwd(qa, ka, va)
    if isinstance(out, tuple):
        out = out[0]
    out.backward(dy)
    return qa.grad, ka.grad, va.grad


def call_fa_grad(q, k, v, dy, fwd, case, lens_q, lens_kv, actual_q, actual_kv,
                 atten_mask, scale_value):
    common = {
        "pse": None,
        "padding_mask": None,
        "atten_mask": atten_mask if case["causal"] else None,
        "softmax_max": fwd[1].float(),
        "softmax_sum": fwd[2].float(),
        "softmax_in": None,
        "attention_in": fwd[0],
        "keep_prob": 1.0 - case["dropout"],
        "input_layout": "TND",
        "actual_seq_qlen": actual_q,
        "actual_seq_kvlen": actual_kv,
        "seed": fwd[4],
        "offset": fwd[5],
        "numels": sum(a * b for a, b in zip(lens_q, lens_kv)) * case["heads_q"],
        "inner_precise": 0,
        "sparse_mode": 3 if case["causal"] else 0,
    }
    if scale_value is not None:
        common["scale_value"] = scale_value
    if hasattr(torch_npu, "npu_fusion_attention_grad_v2"):
        common["pre_tokens"] = 2147483647
        common["next_tokens"] = 2147483647
        res = torch_npu.npu_fusion_attention_grad_v2(q, k, v, dy, case["heads_q"], **common)
    else:
        common["pre_tockens"] = 2147483647
        common["next_tockens"] = 2147483647
        res = torch_npu.npu_fusion_attention_grad(q, k, v, dy, case["heads_q"], **common)
    return res[0], res[1], res[2]


def naive_reference_backward(q, k, v, dy, lens_q, lens_kv, scale, causal):
    qf = q.float().cpu()
    kf = k.float().cpu()
    vf = v.float().cpu()
    df = dy.float().cpu()
    hq, hkv = qf.shape[1], kf.shape[1]
    rep = hq // hkv
    if rep > 1:
        kf = kf.repeat_interleave(rep, dim=1)
        vf = vf.repeat_interleave(rep, dim=1)
    dq = torch.zeros_like(qf)
    dk = torch.zeros_like(kf)
    dv = torch.zeros_like(vf)
    qo = ko = 0
    for lq, lkv in zip(lens_q, lens_kv):
        qb = qf[qo:qo + lq].transpose(0, 1)
        kb = kf[ko:ko + lkv].transpose(0, 1)
        vb = vf[ko:ko + lkv].transpose(0, 1)
        dob = df[qo:qo + lq].transpose(0, 1)
        score = torch.bmm(qb, kb.transpose(1, 2)) * scale
        if causal:
            rows = torch.arange(lq).unsqueeze(1)
            cols = torch.arange(lkv).unsqueeze(0)
            score = score.masked_fill(cols > rows + (lkv - lq), float("-inf"))
        prob = torch.softmax(score, dim=-1)
        out_b = torch.bmm(prob, vb)
        ds = prob * (torch.bmm(dob, vb.transpose(1, 2))
                     - (dob * out_b).sum(-1, keepdim=True))
        dq[qo:qo + lq] = torch.bmm(ds, kb).transpose(0, 1) * scale
        dk[ko:ko + lkv] += torch.bmm(ds.transpose(1, 2), qb).transpose(0, 1) * scale
        dv[ko:ko + lkv] += torch.bmm(prob.transpose(1, 2), dob).transpose(0, 1)
        qo += lq
        ko += lkv
    if rep > 1:
        d = dk.shape[2]
        dk = dk.reshape(kf.shape[0], hkv, rep, d).sum(2)
        dv = dv.reshape(vf.shape[0], hkv, rep, d).sum(2)
    return dq, dk, dv


def split_by_batch(tensor, lens):
    """Split a packed [total, ...] tensor into per-batch tensors along dim 0.

    lens gives the sequence length of each batch in packed (TND) order.
    """
    batches, start = [], 0
    for ln in lens:
        batches.append(tensor[start:start + ln])
        start += ln
    assert start == tensor.shape[0], \
        f"sum(lens)={start} != tensor.shape[0]={tensor.shape[0]}"
    return batches


def compare_per_batch(got, ref, lens, tol, label):
    """Compare got vs ref batch-by-batch (packed dim-0 split by lens).

    Prints per-batch error stats and PASS/FAIL, then a summary line.
    Returns (max_abs_err, max_rel_err, n_fail).
    """
    max_abs = max_rel = 0.0
    n_fail = 0
    for i, (g, r) in enumerate(
            zip(split_by_batch(got, lens), split_by_batch(ref, lens))):
        print(f"batch out g: {g}")
        print(f"batch out r: {r}")
        diff = (g - r).abs()
        abs_err = diff.max().item()
        rel_err = (diff / (r.abs() + 1e-6)).max().item()
        ok = torch.allclose(g, r, atol=tol, rtol=tol)
        n_fail += 0 if ok else 1
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        print(f"    batch {i:>4} (len={lens[i]}): max_abs_err={abs_err:.6g}, "
              f"max_rel_err={rel_err:.6g}  {'PASS' if ok else 'FAIL'}")
    verdict = "PASS" if n_fail == 0 else "FAIL"
    print(f"  {label}: per-batch {verdict} "
          f"({len(lens) - n_fail}/{len(lens)} batches ok, tol={tol}), "
          f"max_abs_err={max_abs:.6g}, max_rel_err={max_rel:.6g}")
    return max_abs, max_rel, n_fail


def main():
    parser = argparse.ArgumentParser(description="benchmark varlen (TND) flash-attention on NPU")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--impl", choices=["raw", "func", "both"], default="both",
                        help="entry point: raw=torch_npu.npu_fusion_attention(_v2), "
                             "func=flash_attn_varlen_func, both=compare (default)")
    parser.add_argument("--check", action="store_true",
                        help="compare with naive reference and flash-attention CPU "
                             "golden (kv chunk=512; dropout cases skipped)")
    parser.add_argument("--grad", choices=["none", "auto", "v2", "both"], default="auto",
                        help="backward mode: auto=autograd fwd+bwd, v2=npu_fusion_attention_grad(_v2) (raw only)")
    parser.add_argument("--only", type=int, nargs="*", default=None,
                        help="indices of cases to run")
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    dtype = getattr(torch, args.dtype)
    torch.npu.manual_seed(args.seed)

    selected = CASES if args.only is None else [CASES[i] for i in args.only]
    impls = ["raw", "func"] if args.impl == "both" else [args.impl]
    summary = []

    for idx, case in enumerate(selected):
        inp = make_inputs(case, dtype, device, args.seed + idx)
        scale_value = case["scale"]
        effective_scale = scale_value if scale_value is not None else 1.0
        atten_mask = make_atten_mask(2048, device)

        print(f"\n=== [{idx}] {case['name']} ===")
        print(f"  layout=TND, q: {tuple(inp['q'].shape)} {inp['q'].dtype}, "
              f"kv: {tuple(inp['k'].shape)}, B={case['batch']}, causal={case['causal']}, "
              f"dropout={case['dropout']}, scale={effective_scale}")
        print(f"  cu_seqlens_q: len={len(inp['cu_q'])}, max_q={inp['max_q']}, "
              f"cu_seqlens_k: len={len(inp['cu_kv'])}, max_kv={inp['max_kv']}")

        results = {}
        fwds = {}
        res_raw = None
        outs = {}
        for impl in impls:
            if impl == "raw":
                tag = "raw:aclnnVarLenScore"
                fwds[impl] = lambda qa, ka, va: call_fa_raw(
                    qa, ka, va, case, inp["actual_q"], inp["actual_kv"], atten_mask, scale_value)
            else:
                tag = "func:flash_attn_varlen_func"
                fwds[impl] = lambda qa, ka, va: call_fa_func(
                    qa, ka, va, case, inp["cu_q"], inp["cu_kv"], inp["max_q"], inp["max_kv"],
                    effective_scale)

            print(f"\n  --- [{impl}] {tag} ---")
            try:
                res, latency_ms = bench_once(
                    lambda: fwds[impl](inp["q"], inp["k"], inp["v"]))
            except RuntimeError as e:
                if "dropout" in str(e):
                    print(f"  skipped: {impl} impl does not support dropout "
                          f"(case dropout={case['dropout']}): {e}")
                    results.setdefault("fwd", {})[impl] = None
                    continue
                raise
            if impl == "raw":
                res_raw = res
            out = res[0] if isinstance(res, tuple) else res
            outs[impl] = out
            print(f"  out[{impl}]: {tuple(out.shape)} {out.dtype}, "
                  f"mean={out.float().mean().item():.6f}, std={out.float().std().item():.6f}")
            torch.set_printoptions(precision=6, linewidth=200)
            print(f"  out[{impl}] tensor (full view; large tensors truncated by torch print):")
            print(out)
            results.setdefault("fwd", {})[impl] = latency_ms
            print(f"  fwd latency: {latency_ms:.3f} ms (single call)")

            if args.check:
                if case["dropout"] > 0:
                    print("  check: skipped (dropout > 0)")
                else:
                    ref = naive_reference(inp["q"], inp["k"], inp["v"], inp["lens_q"], inp["lens_kv"],
                                          effective_scale, case["causal"])
                    got = out.float().cpu()
                    print("  check: per-batch comparison vs naive reference "
                          "(split by lens_q):")
                    compare_per_batch(got, ref, inp["lens_q"], COMPARE_TOL,
                                      "check vs naive reference")
                    fa_ref = flash_attention_reference(
                        inp["q"], inp["k"], inp["v"], inp["lens_q"],
                        inp["lens_kv"], effective_scale, case["causal"],
                        block_size=512)
                    print("  check: per-batch comparison vs flash-attention CPU "
                          "golden (kv chunk=512, split by lens_q):")
                    compare_per_batch(got, fa_ref, inp["lens_q"], COMPARE_TOL,
                                      "check vs flash-attn golden")

        if len(outs) == 2:
            print("\n  --- raw vs func: precision comparison of outputs ---")
            r_out = outs["raw"].float().cpu()
            f_out = outs["func"].float().cpu()
            print("  per-batch comparison (split by lens_q):")
            compare_per_batch(r_out, f_out, inp["lens_q"], COMPARE_TOL,
                              "raw vs func")
            diff = (r_out - f_out).abs()
            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            max_rel = (diff / (f_out.abs() + 1e-6)).max().item()
            n_mismatch = (diff > COMPARE_TOL).sum().item()
            allc = torch.allclose(r_out, f_out, atol=COMPARE_TOL, rtol=COMPARE_TOL)
            flat_r = r_out.reshape(-1)
            flat_f = f_out.reshape(-1)
            n = min(32, flat_r.numel())
            print(f"  raw  out[:{n}] = {flat_r[:n].tolist()}")
            print(f"  func out[:{n}] = {flat_f[:n].tolist()}")
            print(f"  diff out[:{n}] = {diff.reshape(-1)[:n].tolist()}")
            print(f"  global: max_abs_err={max_abs:.6g}, mean_abs_err={mean_abs:.6g}, "
                  f"max_rel_err={max_rel:.6g}")
            print(f"  global allclose(atol={COMPARE_TOL}, rtol={COMPARE_TOL})={allc}, "
                  f"elems with |diff|>{COMPARE_TOL}: {n_mismatch}/{diff.numel()}")

        def check_grads(tag, grads):
            if case["dropout"] > 0:
                print(f"  [{tag}] check: skipped (dropout > 0)")
                return
            refs = naive_reference_backward(inp["q"], inp["k"], inp["v"], dy,
                                            inp["lens_q"], inp["lens_kv"],
                                            effective_scale, case["causal"])
            for nm, got_t, ref_t in zip(("dq", "dk", "dv"), grads, refs):
                g = got_t.float().cpu()
                lens = inp["lens_q"] if nm == "dq" else inp["lens_kv"]
                print(f"  [{tag}] check {nm}: per-batch comparison "
                      f"(split by {'lens_q' if nm == 'dq' else 'lens_kv'}):")
                compare_per_batch(g, ref_t, lens, COMPARE_TOL, f"{tag} {nm}")

        def print_grad_stats(tag, grads):
            print(f"  [{tag}] |dq|max={grads[0].abs().max().item():.6f}, "
                  f"|dk|max={grads[1].abs().max().item():.6f}, "
                  f"|dv|max={grads[2].abs().max().item():.6f}")

        if args.grad == "none":
            summary.append((case["name"], results))
            continue

        active = [i for i in impls if results.get("fwd", {}).get(i) is not None]
        if not active:
            summary.append((case["name"], results))
            continue

        dy = torch.randn_like(out)

        if args.grad in ("auto", "both"):
            for impl in active:
                tag = f"grad:auto:{impl}"
                fb = lambda: run_fwd_bwd_auto(fwds[impl], inp["q"], inp["k"], inp["v"], dy)
                grads, latency_ms = bench_once(fb)
                print_grad_stats(tag, grads)
                results.setdefault("fwd+bwd", {})[impl] = latency_ms
                print(f"  [{tag}] fwd+bwd latency: {latency_ms:.3f} ms (single call)")
                if args.check:
                    check_grads(tag, grads)

        if args.grad in ("v2", "both") and "raw" in active:
            tag = "grad:v2:raw"
            grads, latency_ms = bench_once(
                lambda: call_fa_grad(inp["q"], inp["k"], inp["v"], dy, res_raw, case,
                                     inp["lens_q"], inp["lens_kv"], inp["actual_q"],
                                     inp["actual_kv"], atten_mask, scale_value))
            print_grad_stats(tag, grads)
            results.setdefault("bwd", {})["raw"] = latency_ms
            print(f"  [{tag}] bwd latency: {latency_ms:.3f} ms (single call)")
            if args.check:
                check_grads(tag, grads)

        summary.append((case["name"], results))

    def _fmt(v):
        return f"{v:.3f}" if v is not None else "    n/a"

    for metric in ("fwd", "fwd+bwd", "bwd"):
        rows = [(name, r.get(metric, {})) for name, r in summary if r.get(metric)]
        if not rows:
            continue
        print(f"\n=== Summary: {metric} latency (ms) ===")
        header = f"{'case':<42} {'raw':>10} {'func':>10} {'func/raw':>10}"
        print(header)
        print("-" * len(header))
        for name, m in rows:
            raw_ms = m.get("raw")
            func_ms = m.get("func")
            ratio = f"{func_ms / raw_ms:.3f}x" if (raw_ms and func_ms) else ""
            print(f"{name:<42} {_fmt(raw_ms):>10} {_fmt(func_ms):>10} {ratio:>10}")


if __name__ == "__main__":
    main()