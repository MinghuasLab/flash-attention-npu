"""Compare mixed KV and BF16 under the same workload; optionally export NPU traces.

Run on 950: python benchmarks/benchmark_mixed_kv_950.py --trace-dir /tmp/mixed-traces
Event latency includes host scheduling; use exported kernel durations for device comparisons.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
extension_dir = os.environ.get("FA950_EXTENSION_DIR")
if extension_dir:
    sys.path.insert(0, str(Path(extension_dir).resolve()))
import torch
import torch_npu
import flash_attn_npu_3_950 as extension
extension_path = Path(extension.__file__).resolve()
if extension_dir and extension_path.parent != Path(extension_dir).resolve():
    raise RuntimeError(f"Expected extension in {extension_dir}, loaded {extension_path}")
print(json.dumps({"extension_path": str(extension_path),
                  "extension_sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest()}), flush=True)
from flash_attn_npu_3.flash_attn_npu_interface_950 import _flash_attn_forward_impl, flash_attn_with_kvcache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--entry", choices=("impl", "public"), default="impl",
                        help="Time the existing low-level baseline or the public flash_attn_with_kvcache entry")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cache-protocol", choices=("same", "distinct"), default="distinct",
                        help="By default clone inputs and stream 512 MiB before each sample; use same for warm addresses")
    parser.add_argument("--case", type=int, choices=range(6), help="Run one indexed workload for hardware profiling")
    parser.add_argument("--mode", choices=("bf16", "mixed"), help="Run only one storage mode")
    parser.add_argument("--window-size", type=int, nargs=2, default=(-1, -1), metavar=("LEFT", "RIGHT"),
                        help="Inclusive left and right attention-window radii; use 511 0 for causal SWA-512")
    parser.add_argument("--sk", type=int, help="Override KV length for the selected workloads")
    parser.add_argument("--sq", type=int, help="Override query length for a single workload")
    parser.add_argument("--kv-heads", type=int, choices=(1, 2, 4), help="Set per-device KV heads for TP4, TP2 or TP1")
    parser.add_argument("--batch", type=int, help="Override batch size for a single workload")
    parser.add_argument("--q-layout", choices=("TND", "BSND"), default="TND",
                        help="Use packed token rows or a dense per-batch query")
    parser.add_argument("--cache-layout", choices=("BBND", "BNBD"), default="BBND",
                        help="Physical KV layout used throughout warmup and measurement")
    parser.add_argument("--splits", type=int, choices=range(33), metavar="0..32",
                        help="Use the public API's automatic scheduling (0) or request a split count")
    args = parser.parse_args()
    if min(args.window_size) < -1:
        parser.error("window sizes must be -1 or nonnegative")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.sk is not None and args.sk < 1:
        parser.error("--sk must be positive")
    if args.sq is not None and args.sq < 1:
        parser.error("--sq must be positive")
    if args.batch is not None and args.batch < 1:
        parser.error("--batch must be positive")
    if any(x is not None for x in (args.sq, args.kv_heads, args.batch, args.splits)) and args.case is None:
        parser.error("select --case when overriding query shape or split-KV scheduling")
    if args.trace_dir:
        args.trace_dir = args.trace_dir.resolve()
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(args.trace_dir)  # Keep raw profiler files out of the source checkout.
    torch.set_num_threads(4)
    torch.manual_seed(950)
    torch.npu.set_device(0)
    cases = [(1, 1, 4096, 4), (1, 8, 4096, 2),
                                  (1, 1, 65536, 4), (1, 8, 65536, 2),
                                  (1, 512, 4096, 4), (4, 512, 4096, 2)]
    scrub = (torch.zeros(128 * 1024 * 1024, dtype=torch.float32, device="npu")
             if args.cache_protocol == "distinct" else None)
    def run(sample):
        if args.entry == "impl":
            return _flash_attn_forward_impl(**sample)
        return flash_attn_with_kvcache(
            sample["q"], sample["k"], sample["v"], cache_seqlens=sample["seqused_k"],
            cu_seqlens_q=sample["cu_seqlens_q"], max_seqlen_q=sample["max_seqlen_q"],
            page_table=sample["page_table"], k_descale=sample["k_descale"], v_descale=sample["v_descale"],
            softmax_scale=sample["softmax_scale"], causal=sample["causal"], num_splits=sample["num_splits"],
            window_size=(sample["window_size_left"], sample["window_size_right"]),
        )

    for batch, sq, sk, heads in (cases if args.case is None else [cases[args.case]]):
        sq = args.sq or sq
        heads = args.kv_heads or heads
        batch = args.batch or batch
        splits = 0 if args.splits is None else args.splits
        if args.sk is not None:
            sk = args.sk
        q = torch.randn(batch * sq, heads * 16, 192).bfloat16().npu()
        cu_q = None
        if args.q_layout == "TND":
            cu_q = (torch.arange(batch + 1, dtype=torch.int32) * sq).npu()
        else:
            q = q.reshape(batch, sq, heads * 16, 192)
        shape = (batch * ((sk + 127) // 128), 128, heads, 192)
        k = torch.randn(shape).to(torch.float8_e4m3fn)
        v = torch.randn(shape).to(torch.float8_e4m3fn)
        ks = 0.5 + torch.rand(shape[:-1])
        vs = 0.5 + torch.rand(shape[:-1])
        kb = (k.float() * ks[..., None]).bfloat16().npu()
        vb = (v.float() * vs[..., None]).bfloat16().npu()
        k, v, ks, vs = [x.npu() for x in (k, v, ks, vs)]
        if args.cache_layout == "BNBD":
            k, v, kb, vb = [x.transpose(1, 2).contiguous().transpose(1, 2) for x in (k, v, kb, vb)]
        kw = {name: None for name in inspect.signature(_flash_attn_forward_impl).parameters}
        kw.update(q=q, cu_seqlens_q=cu_q,
                  seqused_k=torch.full((batch,), sk, dtype=torch.int32).npu(),
                  max_seqlen_q=sq, max_seqlen_k=sk,
                  page_table=torch.arange(shape[0], dtype=torch.int32).reshape(batch, -1).npu(),
                  softmax_scale=192 ** -0.5, causal=True, window_size_left=args.window_size[0],
                  window_size_right=args.window_size[1], attention_chunk=0, softcap=0.0,
                  rotary_interleaved=True, num_splits=splits,
                  sm_margin=0, return_softmax_lse=False, layout=0 if args.q_layout == "TND" else 1)
        for mode in (("bf16", "mixed") if args.mode is None else (args.mode,)):
            kw.update(k=k if mode == "mixed" else kb, v=v if mode == "mixed" else vb,
                      k_descale=ks if mode == "mixed" else None,
                      v_descale=vs if mode == "mixed" else None)
            with torch.inference_mode():
                for _ in range(3):
                    run(kw)
                torch.npu.synchronize()
                if scrub is None:
                    samples = [kw] * args.repeats
                else:
                    samples = [{key: value.clone() if isinstance(value, torch.Tensor) else value
                                for key, value in kw.items()} for _ in range(args.repeats)]
                    for key in ("k", "v"):
                        assert len({item[key].data_ptr() for item in [kw, *samples]}) == len(samples) + 1
                times = []
                wall_times = []
                for sample in samples:
                    if scrub is not None:
                        scrub.add_(1)
                        torch.npu.synchronize()
                    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    begin_ns = time.perf_counter_ns()
                    run(sample)
                    end.record()
                    end.synchronize()
                    wall_times.append((time.perf_counter_ns() - begin_ns) / 1000.0)
                    times.append(start.elapsed_time(end) * 1000)
                trace = None
                kernel_times = []
                if args.trace_dir:
                    args.trace_dir.mkdir(parents=True, exist_ok=True)
                    trace = args.trace_dir / f"{mode}_b{batch}_q{sq}_k{sk}_h{heads}.json"
                    if tuple(args.window_size) != (-1, -1):
                        trace = trace.with_stem(f"{trace.stem}_window{args.window_size[0]}_{args.window_size[1]}")
                    if args.entry == "public":
                        trace = trace.with_stem(f"{trace.stem}_public")
                    if scrub is not None:
                        trace = trace.with_stem(f"{trace.stem}_distinct")
                    if args.splits is not None:
                        trace = trace.with_stem(f"{trace.stem}_splits{splits}")
                    if args.cache_layout == "BNBD":
                        trace = trace.with_stem(f"{trace.stem}_bnbd")
                    if args.q_layout == "BSND":
                        trace = trace.with_stem(f"{trace.stem}_bsnd")
                    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                                                torch_npu.profiler.ProfilerActivity.NPU]) as prof:
                        for sample in samples:
                            if scrub is not None:
                                scrub.add_(1)
                                torch.npu.synchronize()
                            run(sample)
                        torch.npu.synchronize()
                    prof.export_chrome_trace(str(trace))
                    data = json.loads(trace.read_text())
                    events = data if isinstance(data, list) else data["traceEvents"]
                    kernel_times = [event["dur"] for event in events
                                    if event.get("ph") == "X" and "FAInfer" in event.get("name", "")]
                    if len(kernel_times) != args.repeats:
                        raise RuntimeError(f"Expected {args.repeats} FA kernels, found {len(kernel_times)} in {trace}")
            print(json.dumps(dict(mode=mode, B=batch, Sq=sq, Sk=sk, Hkv=heads, Hq=heads * 16, requested_splits=splits,
                                  entry=args.entry, q_layout=args.q_layout, window_size=args.window_size, synchronized_wall_median_us=statistics.median(wall_times),
                                  synchronized_wall_samples_us=wall_times, D=192, cache_layout=args.cache_layout, cache_protocol=args.cache_protocol, event_median_us=statistics.median(times),
                                  event_samples_us=times, kernel_samples_us=kernel_times,
                                  kernel_median_us=statistics.median(kernel_times) if kernel_times else None,
                                  trace=str(trace) if trace else None)), flush=True)
            del samples, sample


if __name__ == "__main__":
    main()
