#!/usr/bin/env python3
"""Performance reference for flash_attn_combine and the FD path.

The requirement states "性能在 FD 体现，此处不要求性能", so this script is a
*reference* measurement, not a gate:

  --mode combine : fused combine kernel only (latency + effective bandwidth)
  --mode fd      : end-to-end FlashDecode A/B (num_splits=1 vs 2/4/8)
  --mode both    : both of the above (default)

Usage:
  source /data/w00896777/ops/0930/ascend-toolkit/set_env.sh
  source /data/miniconda3/etc/profile.d/conda.sh && conda activate wk
  cd /data/w00896777/code/flash-attention-npu
  PYTHONPATH=$PWD python tests/st/atk_flash_attn_combine/bench_combine.py --mode both
"""

import argparse
import statistics
import sys

import torch
import torch_npu


def _sync():
    torch.npu.synchronize()


def _timeit(fn, warmup: int, iters: int) -> float:
    """Return the median latency in milliseconds."""
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        _sync()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def bench_combine(args) -> None:
    from flash_attn_npu_3 import flash_attn_combine

    ns, b, s, h, d = args.ns, args.batch, args.seqlen, args.heads, args.head_dim
    dtype = getattr(torch, args.dtype)
    op = torch.randn(ns, b, s, h, d, device="npu", dtype=torch.float32)
    lp = torch.randn(ns, b, h, s, device="npu", dtype=torch.float32).transpose(-1, -2)

    def run():
        flash_attn_combine(op, lp, out_dtype=dtype)

    ms = _timeit(run, args.warmup, args.iters)
    out_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    total_bytes = (
        ns * b * s * h * d * 4          # out_partial (fp32)
        + ns * b * s * h * 4            # lse_partial (fp32)
        + b * s * h * d * out_bytes     # out
        + b * s * h * 4                 # lse out
    )
    gbps = total_bytes / (ms * 1e-3) / 1e9
    print(
        f"[combine] ns={ns} B={b} S={s} H={h} D={d} dtype={args.dtype} "
        f"| median={ms:.3f} ms | traffic={total_bytes / 2**20:.1f} MiB "
        f"| effective={gbps:.1f} GB/s"
    )


def bench_fd(args) -> None:
    from flash_attn_npu_3 import flash_attn_with_kvcache
    from tests.common.test_utils import make_block_table, make_paged_kv_cache

    dtype = getattr(torch, args.fd_dtype)
    q_len, kv_len, block_size = args.fd_qlen, args.fd_kvlen, 128
    b, h, h_kv, d = args.fd_batch, args.fd_heads, args.fd_kv_heads, args.head_dim
    total_q = b * q_len

    q = torch.randn(total_q, h, d, device="npu", dtype=dtype)
    k_cache, v_cache = make_paged_kv_cache(
        b, kv_len, block_size, h_kv, d, dtype, device="npu"
    )
    block_table = make_block_table(b, kv_len, block_size).npu()
    cache_seqlens = torch.full((b,), kv_len, dtype=torch.int32, device="npu")
    cu_q = torch.tensor(
        [i * q_len for i in range(b + 1)], dtype=torch.int32, device="npu"
    )

    print(
        f"[fd] B={b} H={h} Hkv={h_kv} D={d} q_len={q_len} kv_len={kv_len} "
        f"dtype={args.fd_dtype} block={block_size}"
    )
    base = None
    for splits in args.fd_splits:
        def run(splits=splits):
            flash_attn_with_kvcache(
                q, k_cache, v_cache, None, None, None,
                cache_seqlens=cache_seqlens,
                page_table=block_table,
                cu_seqlens_q=cu_q,
                max_seqlen_q=q_len,
                causal=False,
                num_splits=splits,
                return_softmax_lse=True,
            )

        ms = _timeit(run, args.warmup, args.iters)
        if splits in (0, 1):
            base = ms
        gain = (base / ms) if base else 1.0
        print(f"  num_splits={splits:<3} median={ms:.3f} ms  speedup_vs_base={gain:.2f}x")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("combine", "fd", "both"), default="both")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    # combine shape
    ap.add_argument("--ns", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    # FD shape (decode gate: paged KV + TND, q_len <= 16, kv_len >= 1024)
    ap.add_argument("--fd-batch", type=int, default=8)
    ap.add_argument("--fd-heads", type=int, default=32)
    ap.add_argument("--fd-kv-heads", type=int, default=8)
    ap.add_argument("--fd-qlen", type=int, default=1)
    ap.add_argument("--fd-kvlen", type=int, default=8192)
    ap.add_argument("--fd-dtype", default="float16")
    ap.add_argument("--fd-splits", type=int, nargs="+", default=[1, 2, 4, 8])
    args = ap.parse_args()

    if args.mode in ("combine", "both"):
        bench_combine(args)
    if args.mode in ("fd", "both"):
        bench_fd(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
