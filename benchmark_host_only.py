"""Paired training forward/backward wall timing and separate attention profiling."""
import argparse
import csv
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
import subprocess
import time


def save(output, metadata, rows):
    (output / 'results.json').write_text(json.dumps(dict(metadata=metadata, rows=rows), indent=2) + '\n')
    flat = []
    for row in rows:
        item = {key: row[key] for key in ('scenario', 'dropout', 'phase', 'wall_ratio', 'kernel_ratio', 'relative_ratio_gap_percent')}
        for name, prefix in [('FA2', 'fa2'), ('torch FA', 'torch_fa')]:
            backend = row['backends'][name]
            wall, kernel = backend['wall_median_us'], backend['attention_kernel_median_us']
            item.update({prefix + '_wall_ms': wall / 1000, prefix + '_kernel_ms': kernel / 1000,
                         prefix + '_wall_minus_kernel_ms': (wall - kernel) / 1000,
                         prefix + '_wall_over_kernel': wall / kernel,
                         prefix + '_kernel_share_percent': kernel / wall * 100})
        flat.append(item)
    if flat:
        with (output / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)


def measure(calls, args, destination, phase, torch, torch_npu):
    names = list(calls)
    for _ in range(10):
        for call in calls.values():
            call()
    torch.npu.synchronize()
    samples = {name: [] for name in names}
    for round_id in range(args.rounds):
        current = {name: [] for name in names}
        for i in range(args.rep):
            for name in names if (i + round_id) % 2 == 0 else names[::-1]:
                start = time.perf_counter_ns()
                calls[name]()
                torch.npu.synchronize()
                current[name].append((time.perf_counter_ns() - start) / 1000)
        for name in names:
            samples[name].append(current[name])
    print('  wall complete; profiling', phase, flush=True)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=2, active=args.profile_iters * 2, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(destination)),
    ) as profiler:
        for i in range(2 + args.profile_iters * 2):
            order = names if (i // 2) % 2 == 0 else names[::-1]
            name = order[i % 2]
            with torch.autograd.profiler.record_function(name + ' ' + phase):
                calls[name]()
                torch.npu.synchronize()
            profiler.step()
    details = list(destination.glob('**/kernel_details.csv'))
    assert len(details) == 1, details
    tasks = list(csv.DictReader(details[0].open()))
    backends = {}
    for name in names:
        if phase == 'backward':
            selected = [t for t in tasks if ('FAG' in t['Name'] or 'short_query' in t['Name'])] if name == 'FA2' else [t for t in tasks if 'FlashAttentionScoreGrad' in t['Name']]
        else:
            selected = [t for t in tasks if any(s in t['Name'] for s in ('SplitFuse', 'FAInfer', 'fa_infer', 'ascendc_fa'))] if name == 'FA2' else [t for t in tasks if 'FlashAttentionScore' in t['Name'] and 'Grad' not in t['Name']]
        assert len(selected) == args.profile_iters, (phase, name, len(selected), args.profile_iters, sorted(set(t['Name'] for t in tasks)))
        kernel = [float(t['Duration(us)']) for t in selected]
        wall = [t for sample in samples[name] for t in sample]
        backends[name] = dict(wall_samples_us=samples[name], wall_median_us=median(wall),
                              wall_round_medians_us=[median(s) for s in samples[name]],
                              attention_kernel_samples_us=kernel, attention_kernel_median_us=median(kernel),
                              attention_tasks=selected)
    return backends, str(details[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', nargs='+')
    parser.add_argument('--dropout', type=float, nargs='+', default=[0., .1])
    parser.add_argument('--rep', type=int, default=80)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--profile-iters', type=int, default=15)
    parser.add_argument('--device', type=int, default=2)
    args = parser.parse_args()
    assert min(args.rep, args.rounds, args.profile_iters) > 0
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch_npu
    torch.set_num_threads(8)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    import bench
    import bench_onerec_ops as helper
    helper.configure_uniattention(None)
    import npu_fa
    import npu_fa_v1
    from flash_attn_npu import flash_attn_npu as native
    print('Loaded interfaces; NPU', args.device, flush=True)
    root = Path(__file__).resolve().parent
    binary = Path(native.__file__).resolve()
    assert binary.is_relative_to(root), (binary, root)
    files = [binary, root / 'flash_attn_npu/flash_attn_npu_interface.py', root / 'npu_fa_v1.py', root / 'npu_fa.py', root / 'bench.py', Path(__file__)]
    metadata = dict(device=args.device, device_name=torch.npu.get_device_name(), torch=torch.__version__, torch_npu=torch_npu.__version__,
                    started_at_utc=datetime.now(timezone.utc).isoformat(),
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
                    rep=args.rep, rounds=args.rounds, profile_iters=args.profile_iters, threads=8,
                    files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    interfaces={'FA2': 'npu_fa_v1.flash_attn -> flash_attn_npu.flash_attn_npu_interface.flash_attn_varlen_func',
                                'torch FA': 'npu_fa.flash_attn -> torch_npu.npu_fusion_attention'},
                    timing={'forward_wall': 'grad-enabled training API call and synchronize; returned temporary graph released; inputs prepared outside timing',
                            'backward_wall': 'retained-graph autograd.grad and synchronize; forward and dout generation excluded',
                            'kernel': 'independent profiler main attention kernel only; exactly one per API invocation checked',
                            'aggregation': 'median of all 3x80 wall samples and median of 15 independently profiled kernel samples',
                            'cache': '10 warmup calls; no cache flush; paired alternating backend order',
                            'wall_minus_kernel': 'includes Python/autograd/Host/synchronization/auxiliary kernels; not pure Host overhead'},
                    manifest=json.loads((root / 'host_only_manifest.json').read_text()))
    metadata['timing']['aggregation'] = f'median of all {args.rounds}x{args.rep} wall samples and median of {args.profile_iters} independently profiled kernel samples'
    rows = []
    for case in bench.CASES:
        if args.case and case['name'] not in args.case:
            continue
        seed = bench.SEED + sum(map(ord, case['name']))
        print('Preparing', case['name'], flush=True)
        q, k, v, cq, ck, mq, mk = bench.build_inputs(case, seed)
        for dropout in args.dropout:
            forward_calls, backward_calls = {}, {}
            torch.manual_seed(seed)
            torch.npu.manual_seed_all(seed)
            dout = torch.randn_like(q)
            for name, api in [('FA2', npu_fa_v1.flash_attn), ('torch FA', npu_fa.flash_attn)]:
                print('  Preparing graph', name, 'dropout', dropout, flush=True)
                xs = tuple(t.detach().requires_grad_() for t in (q, k, v))
                def forward(api=api, xs=xs):
                    return api(query_states=xs[0], key_states=xs[1], value_states=xs[2], cu_seqlens_q=cq, cu_seqlens_k=ck,
                               max_seqlen_q=mq, max_seqlen_k=mk, softmax_scaling=case['scale'], dropout_ratio=dropout,
                               attention_mask='causal' if case['causal'] else 'full')[0]
                torch.manual_seed(seed)
                torch.npu.manual_seed_all(seed)
                out = forward()
                def backward(out=out, xs=xs, dout=dout):
                    return torch.autograd.grad(out, xs, dout, retain_graph=True)
                forward_calls[name], backward_calls[name] = forward, backward
            # Release loop-local references; closures retain only the current configuration.
            del out, xs, forward, backward
            for phase, calls in [('forward', forward_calls), ('backward', backward_calls)]:
                print(case['name'], 'dropout', dropout, phase, flush=True)
                dest = args.output / 'profiles' / case['name'] / f'dropout_{dropout:g}' / phase
                backends, detail = measure(calls, args, dest, phase, torch, torch_npu)
                row = dict(scenario=case['name'], config=case, dropout=dropout, phase=phase, kernel_details=detail, backends=backends,
                           input_shapes=[list(t.shape) for t in (q, k, v)], max_lengths=[mq, mk])
                a, b = backends['FA2'], backends['torch FA']
                row['wall_ratio'] = a['wall_median_us'] / b['wall_median_us']
                row['kernel_ratio'] = a['attention_kernel_median_us'] / b['attention_kernel_median_us']
                row['relative_ratio_gap_percent'] = (row['wall_ratio'] / row['kernel_ratio'] - 1) * 100
                rows.append(row)
                save(args.output, metadata, rows)
                print('  E2E/kernel ms', {n: [round(v['wall_median_us']/1000, 4), round(v['attention_kernel_median_us']/1000, 4)] for n,v in backends.items()},
                      'FA2/Fusion', round(row['wall_ratio'], 4), round(row['kernel_ratio'], 4), flush=True)
            del calls, forward_calls, backward_calls, dout
            gc.collect()
            torch.npu.empty_cache()
        del q, k, v, cq, ck
        gc.collect()
        torch.npu.empty_cache()
    print('Completed', len(rows), 'scenario/dropout/phase rows:', args.output / 'summary.csv', flush=True)


if __name__ == '__main__':
    main()
