"""Validate completed samples and write the Chinese host-only measurement report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = json.loads((args.output / 'results.json').read_text())
    rows, meta = data['rows'], data['metadata']
    validation = json.loads((root / 'validation.json').read_text())
    assert validation['binary_sha256'] == meta['manifest']['binary_sha256']
    (args.output / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    expected = {(case, p, phase) for case in ('qwen3_3b', 'qwen3_4b', 't5_encoder', 't5_decoder', 't5_cross')
                for p in (0., .1) for phase in ('forward', 'backward')}
    assert len(rows) == 20 and {(r['scenario'], r['dropout'], r['phase']) for r in rows} == expected
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for p, digest in meta['files'].items():
        assert sha(p) == digest, p
    original = Path('/home/hxb/flash-attention-npu')
    manifest = meta['manifest']
    assert subprocess.check_output(['git', 'status', '--short'], cwd=original, text=True) == manifest['original_status']
    for p, digest in manifest['original_hashes'].items():
        assert sha(original / p) == digest, p
    spread = []
    kernel_names = {}
    for row in rows:
        for name, b in row['backends'].items():
            assert len(b['wall_samples_us']) == meta['rounds']
            assert all(len(a) == meta['rep'] for a in b['wall_samples_us'])
            assert len(b['attention_kernel_samples_us']) == meta['profile_iters']
            assert b['wall_median_us'] == median(t for a in b['wall_samples_us'] for t in a)
            assert b['attention_kernel_median_us'] == median(b['attention_kernel_samples_us'])
            values = b['wall_round_medians_us']
            spread.append(dict(scenario=row['scenario'], dropout=row['dropout'], phase=row['phase'], backend=name,
                               round_median_spread_percent=(max(values)-min(values))/median(values)*100))
            kernel_names.setdefault(row['phase'] + '/' + name, set()).update(t['Name'] for t in b['attention_tasks'])
    audit = dict(rows=len(rows), timing_records=len(rows)*2, wall_samples=len(rows)*2*meta['rounds']*meta['rep'],
                 attention_samples=len(rows)*2*meta['profile_iters'], original_unchanged=True, runtime_hashes_match=True,
                 wall_round_spreads=spread, main_kernel_names={k:sorted(v) for k,v in kernel_names.items()})
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    labels = {'qwen3_3b': 'Qwen3 3B', 'qwen3_4b': 'Qwen3 4B', 't5_encoder': 'T5 Encoder', 't5_decoder': 'T5 Decoder', 't5_cross': 'T5 Cross'}
    lines = ['# 与原项目 Host 优化一致的实验结果', '',
             '实验副本：`' + str(root) + '`。原项目及旧版 host_only 结果保持不变。', '',
             '## 代码范围', '',
             '- 以修复后的 baseline 为设备基线，恢复原项目的 backward Host/Python 优化。',
             '- `flash_api.cpp` 和 `flash_attn_npu_interface.py` 与本轮开始时原项目逐字一致：均匀前缀证明、同一 Tensor 快速判断、Opt tiling 缓存、mask 复用、eager 快路、padding/slice 优化均保留。',
             '- `fag_general_host.cpp` 采用原项目 General 均匀前缀重建、tiling 缓存和 mask 复用，仅删除 singleTile/direct-output 的状态及专用 kernel 调用，改用通用设备 kernel。',
             '- 保留原项目 `preferSingleTileFull` 路由选择。因此 T5 Encoder dropout=0 仍选择 General 路径，但执行通用 kernel；与旧版 host_only 的 Opt 路由不同。',
             '- 未叠加上一版独有的 static tiling 实验。原项目非均匀 General 前缀仍按原实现 D2H。',
             '- 保留共同容量和 MMAD 尾块正确性修复。未引入 singleTile/direct-output、dual_aiv、head_fusion 设备优化。前向维持实验基线 47fec0c。',
             '- 相对修复后的 baseline，92/94 个原有编译对象相同，仅 flash_api.o 和 fag_general_host.o 两个 Host 对象不同，无新增设备对象。',
             '- 二进制 SHA256：`' + manifest['binary_sha256'] + '`。源码/对象依据见副本 `host_only_manifest.json`。', '',
             '## 验证与计时', '',
             '- 已重新执行 backward cache 回归（含动态前缀、跨流、FakeTensor、torch.compile）、CPU FP32/低精度双参考及五场景完整梯度对照；结果见 `validation.json`。',
             '- FA2：`npu_fa_v1.flash_attn` → `flash_attn_npu.flash_attn_npu_interface.flash_attn_varlen_func`。Fusion / torch FA：`npu_fa.flash_attn` → `torch_npu.npu_fusion_attention`。',
             '- BF16、TND、D=128，直接使用 bench.py 的五个场景与实际变长前缀，dropout=0/0.1。',
             '- 前向为开启梯度的训练 API 调用到 synchronize；反向为预建图和 dout 后的 autograd.grad(retain_graph=True) 到 synchronize，不含前向。',
             '- 每项每接口预热10次，E2E交替测量3轮×80次并取全部样本中位数；kernel独立profiling15次并取中位数。热缓存，无cache flush。',
             '- kernel仅主attention算子，每项断言每次调用一个主kernel。E2E-kernel还包含Python、Host、辅助算子和同步，不等于纯CPU耗时。',
             f"- NPU {meta['device']}，{meta['device_name']}；torch {meta['torch']}，torch_npu {meta['torch_npu']}；CANN 9.2.0，CPU线程8。设备快照保存在副本 npu_before.txt / npu_after.txt。",
             '- FA2/Fusion倍率小于1表示FA2更快；E2E/kernel为同一接口两项独立测量中位数之比；耗时单位ms。', '']
    for phase, title in [('forward', '前向'), ('backward', '反向')]:
        lines += ['## ' + title, '',
                  '| 场景 | dropout | FA2 E2E | Fusion E2E | FA2 kernel | Fusion kernel | E2E FA2/Fusion | kernel FA2/Fusion | FA2 E2E/kernel | Fusion E2E/kernel |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            if row['phase'] != phase:
                continue
            a,b = row['backends']['FA2'], row['backends']['torch FA']
            aw,bw,ak,bk = a['wall_median_us'],b['wall_median_us'],a['attention_kernel_median_us'],b['attention_kernel_median_us']
            lines.append(f"| {labels[row['scenario']]} | {row['dropout']:g} | {aw/1000:.4f} | {bw/1000:.4f} | {ak/1000:.4f} | {bk/1000:.4f} | {aw/bw:.3f} | {ak/bk:.3f} | {aw/ak:.3f} | {bw/bk:.3f} |")
        subset = [r for r in rows if r['phase'] == phase]
        faster_wall = sum(r['wall_ratio'] < 1 for r in subset)
        faster_kernel = sum(r['kernel_ratio'] < 1 for r in subset)
        lines += ['', f'本次 {title} 的 10 项配置中，FA2 E2E 快于 Fusion 的有 {faster_wall} 项，主 kernel 快于 Fusion 的有 {faster_kernel} 项。', '']
    widest = max(spread, key=lambda r:r['round_median_spread_percent'])
    lines += ['## 数据校验与解读边界', '',
              f"- 共 {audit['timing_records']} 条接口计时记录，{audit['wall_samples']} 个 E2E 样本，{audit['attention_samples']} 个主 kernel 样本；数量、中位数和运行时文件 SHA256 校验通过。",
              f"- 各项三轮 E2E 中位数的最大极差/中位数为 {widest['round_median_spread_percent']:.2f}%（{widest['scenario']}，dropout={widest['dropout']:g}，{widest['phase']}，{widest['backend']}），逐项见 `audit.json`。",
              '- 本报告为本轮重测；与旧版 host_only 或原项目历史值对照时需考虑采集批次不同，不能精确归因到单项修改。',
              '- E2E/kernel 和 kernel 占比来自独立测量，不能用作流水重叠或纯 CPU 开销的精确拆分。', '',
              '## 产物与复现', '',
              '- `summary.csv`：20 行，包含双方 E2E、kernel、E2E-kernel、E2E/kernel、kernel 占比、FA2/Fusion 和倍率额外差距。',
              '- `results.json`：完整 E2E 与 kernel 样本、实际主 kernel 名称、配置、运行环境和文件哈希。',
              '- `profiles/`：全部原始 profiler 结果。`audit.json`：校验与轮间波动。',
              '- 新副本 `host_only_manifest.json`：源码差异、二进制/对象依据和原项目快照。', '',
              '```bash', 'cd ' + str(root),
              'bash run_host_only.sh /home/hxb/experiments/fa2_backward_20261029/results/host_original_rerun',
              'python write_host_only_report.py /home/hxb/experiments/fa2_backward_20261029/results/host_original_rerun', '```', '']
    old_data = json.loads((root.parent / 'results/host_only/results.json').read_text())
    original_data = json.loads(Path('/home/hxb/flash-attention-npu/benchmark_results/backward_goal_20261028/final_large_paired/results.json').read_text())
    old = {(r['scenario'], r['dropout'], r['phase']): r for r in old_data['rows']}
    original = {(r['scenario'], r['dropout'], 'backward'): r for r in original_data['rows']}
    comparison = []
    for row in rows:
        key = row['scenario'], row['dropout'], row['phase']
        item = dict(scenario=key[0], dropout=key[1], phase=key[2])
        for source, record in [('current', row), ('previous_host_only', old[key]), ('original_archived', original.get(key))]:
            for name, label in [('FA2', 'fa2'), ('torch FA', 'fusion')]:
                for timing, metric in [('wall', 'wall_median_us'), ('kernel', 'attention_kernel_median_us')]:
                    item[f'{source}_{label}_{timing}_ms'] = record['backends'][name][metric]/1000 if record else ''
        comparison.append(item)
    with (args.output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    lines += ['## 与上一版及原项目历史结果对照', '',
              '以下为反向 dropout=0 的 FA2 耗时，单位 ms。旧版 host_only 和原项目列来自历史采集，本轮只重新采集 current；计时定义相同，采集时间和样本数不同。所有 dropout/phase 的对照及各批次 Fusion 数值保存在 `comparison.csv`。', '',
              '| 场景 | 旧版 host_only E2E | 本轮 E2E | 原项目历史 E2E | 旧版 kernel | 本轮 kernel | 原项目历史 kernel |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for item in comparison:
        if item['phase'] != 'backward' or item['dropout'] != 0:
            continue
        values = [item[f'{source}_fa2_{timing}_ms'] for timing in ('wall', 'kernel')
                  for source in ('previous_host_only', 'current', 'original_archived')]
        lines.append('| ' + labels[item['scenario']] + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['',
              'Qwen 和 Decoder 恢复的是原项目 Host/Python 路径；Encoder/Cross 仍没有原项目 singleTile/direct-output 设备优化。本轮保留了原项目 Encoder full-attention 的 General 路由，故它与旧版 host_only 的 kernel 路由也可能不同。以上对照不能作为单项 Host 修改的精确收益消融。', '']
    (args.output / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in audit.items() if k not in ('wall_round_spreads','main_kernel_names')}, indent=2))
    print('Report:', args.output / 'REPORT.md')


if __name__ == '__main__':
    main()
