#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small-shape func-only scenario derived from fa_varlen_call.py case2.

Keeps the scalar-bound triggers of case2:
  * Q ≪ KV per batch (max_q=3, KV avg 32)
  * batch = 512, heads_q = 16, head_dim = 128
  * causal mask (per-task window computation)
  * same task decomposition: batch x heads = 512 x 16 = 8192 tasks

Total KV is reduced from 131475 to 16384 so the end-to-end wall time and the
msprof op simulator runtime stay small, while the scalar/setup overhead per
task remains the dominant term (scalar-bound symptom is preserved).
"""

import argparse
import json
import random
import time

import torch
import torch_npu

from flash_attn_npu import flash_attn_varlen_func, get_scheduler_metadata
from flash_attn_npu.flash_attn_npu_interface import _flash_attn_varlen_forward

CASES = [
    {
        "name": "mini_Q384_KV2048_H16_causal",
        "total_q": 20480, "heads_q": 24,
        "total_kv": 20480, "heads_kv": 4,
        "head_dim": 128, "scale": 1.0, "causal": True,
        "batch": 320, "max_q": 64, "max_kv": 64, "dropout": 0.0,
    },
]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--impl", choices=["func", "raw"], default="func")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--save-meta", "--save-metadata", dest="save_metadata", default=None,
                        help="path to save precomputed scheduler_metadata (--save-metadata alias; "
                             "real NPU only)")
    parser.add_argument("--load-meta", "--load-metadata", dest="load_metadata", default=None,
                        help="path to load precomputed scheduler_metadata and pass it to the kernel "
                             "without recomputing (--load-metadata alias)")
    parser.add_argument("--no-check", action="store_true",
                        help="skip the output abs-max check (fewer kernels in simulation)")
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    dtype = getattr(torch, args.dtype)
    case = CASES[0]

    lens_q = gen_seq_lens(case["total_q"], case["batch"], case["max_q"], args.seed)
    lens_kv = gen_seq_lens(case["total_kv"], case["batch"], case["max_kv"],
                           args.seed + 10000,
                           lower=lens_q if case["causal"] else None)
    cu_q = [0]
    for x in lens_q:
        cu_q.append(cu_q[-1] + x)
    cu_kv = [0]
    for x in lens_kv:
        cu_kv.append(cu_kv[-1] + x)

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    q = torch.randn(case["total_q"], case["heads_q"], case["head_dim"], generator=g)
    k = torch.randn(case["total_kv"], case["heads_kv"], case["head_dim"], generator=g)
    v = torch.randn(case["total_kv"], case["heads_kv"], case["head_dim"], generator=g)
    q = q.to(dtype).to(device).contiguous()
    k = k.to(dtype).to(device).contiguous()
    v = v.to(dtype).to(device).contiguous()
    cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_kv_t = torch.tensor(cu_kv, dtype=torch.int32, device=device)

    print(f"case: {case['name']}")
    print(f"  q={tuple(q.shape)} kv={tuple(k.shape)} B={case['batch']} "
          f"causal={case['causal']} max_q={max(lens_q)} max_kv={max(lens_kv)} "
          f"avg_kv={case['total_kv'] // case['batch']} tasks={case['batch'] * case['heads_q']}")

    def make_metadata():
        cache_seqlens = torch.tensor(lens_kv, dtype=torch.int32, device=device)
        return get_scheduler_metadata(
            case["batch"], max(lens_q), max(lens_kv),
            case["heads_q"], case["heads_kv"], case["head_dim"],
            cache_seqlens,
            qkv_dtype=dtype,
            cu_seqlens_q=cu_q_t,
            page_size=None,
            causal=case["causal"],
            window_size=(-1, -1),
            softcap=0.0,
            softmax_scale=case["scale"],
        )

    meta = None
    if args.save_metadata is not None:
        meta = make_metadata()
        params = getattr(meta, "_fa_scheduler_params", {})
        torch.save({"meta": meta.cpu().contiguous(), "params": params}, args.save_metadata)
        print(f"  saved scheduler_metadata -> {args.save_metadata} "
              f"shape={tuple(meta.shape)} nbytes={meta.nbytes} params={json.dumps(params)}")
    if args.load_metadata is not None:
        loaded = torch.load(args.load_metadata, map_location="cpu")
        meta = loaded["meta"].to(device).contiguous()
        print(f"  loaded scheduler_metadata -> {args.load_metadata} "
              f"shape={tuple(meta.shape)} nbytes={meta.nbytes}")

    def fwd():
        if args.impl == "raw":
            atten_mask = None
            if case["causal"]:
                atten_mask = torch.triu(
                    torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).to(device)
            return torch_npu.npu_fusion_attention(
                q, k, v, case["heads_q"], "TND",
                pre_tockens=2147483647, next_tockens=2147483647,
                pse=None,
                atten_mask=atten_mask,
                keep_prob=1.0 - case["dropout"],
                inner_precise=0,
                actual_seq_qlen=tuple(cu_q[1:]),
                actual_seq_kvlen=tuple(cu_kv[1:]),
                sparse_mode=3 if case["causal"] else 0,
                scale=case["scale"],
            )
        if meta is not None:
            out, softmax_lse, s_dmask, rng_state = _flash_attn_varlen_forward(
                q, k, v,
                cu_q_t, cu_kv_t,
                max(lens_q), max(lens_kv),
                case["dropout"], case["scale"],
                causal=case["causal"],
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                return_softmax=False,
                block_table=None,
                leftpad_k=None,
                seqused_k=None,
                zero_tensors=False,
                scheduler_metadata=meta,
            )
            return out
        return flash_attn_varlen_func(
            q, k, v,
            cu_q_t, cu_kv_t,
            max(lens_q), max(lens_kv),
            dropout_p=case["dropout"],
            softmax_scale=case["scale"],
            causal=case["causal"],
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
            block_table=None,
        )

    out = fwd()
    out = out[0] if isinstance(out, tuple) else out
    torch.npu.synchronize()
    if not args.no_check:
        print(f"  out={tuple(out.shape)} {out.dtype} max_abs={out.float().abs().max().item():.6f}")

    for _ in range(args.warmup):
        fwd()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.repeat):
        fwd()
    torch.npu.synchronize()
    avg_ms = (time.perf_counter() - t0) / args.repeat * 1000.0
    print(f"  {args.impl} fwd: {avg_ms:.3f} ms/iter (warmup={args.warmup}, repeat={args.repeat})")


if __name__ == "__main__":
    main()
