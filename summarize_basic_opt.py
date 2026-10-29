"""Compare fresh Host-baseline and imported-kernel runs, separately for each dropout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--ported', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    runs = {name: json.loads((path/'results.json').read_text())
            for name, path in [('host_baseline', args.baseline), ('ported', args.ported)]}
    index = {}
    expected = {(case, p, phase) for case in ('qwen3_3b', 'qwen3_4b', 't5_encoder', 't5_decoder', 't5_cross')
                for p in (0., .1) for phase in ('forward', 'backward')}
    audit = {}
    for variant, data in runs.items():
        meta = data['metadata']
        index[variant] = {(r['scenario'], r['dropout'], r['phase']): r for r in data['rows']}
        assert len(data['rows']) == 20 and set(index[variant]) == expected
        for path, digest in meta['files'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        spreads = []
        for row in data['rows']:
            for name, b in row['backends'].items():
                assert len(b['wall_samples_us']) == meta['rounds']
                assert all(len(a) == meta['rep'] for a in b['wall_samples_us'])
                assert len(b['attention_kernel_samples_us']) == meta['profile_iters']
                assert b['wall_median_us'] == median(t for a in b['wall_samples_us'] for t in a)
                assert b['attention_kernel_median_us'] == median(b['attention_kernel_samples_us'])
                values = b['wall_round_medians_us']
                spreads.append(dict(scenario=row['scenario'], dropout=row['dropout'], phase=row['phase'], backend=name,
                                    round_spread_percent=(max(values)-min(values))/median(values)*100))
        audit[variant] = dict(records=40, wall_samples=40*meta['rounds']*meta['rep'],
                              kernel_samples=40*meta['profile_iters'], wall_round_spreads=spreads)
    provenance = json.loads((root/'basic_opt_port.json').read_text())
    validation = json.loads((root/'basic_opt_validation.json').read_text())
    assert validation['binary_sha256'] == provenance['binary_sha256']
    labels = {'qwen3_3b':'Qwen3 3B', 'qwen3_4b':'Qwen3 4B', 't5_encoder':'T5 Encoder',
              't5_decoder':'T5 Decoder', 't5_cross':'T5 Cross'}
    lines = ['# fag_v2_basic_opt 移植实测', '',
             '本报告分别统计 dropout=0 和 dropout=0.1，不跨 dropout 汇总性能。Host 基线与移植版均为本轮重新采集。', '',
             '## 代码与验证', '',
             f"- Host 基线：`{provenance['host_base']}`；来源提交：`{provenance['source_commit']}`。",
             '- [上游对比](' + provenance['compare_url'] + ')。保留原项目 Host/Python 缓存及快路，移植反向 Opt 双地址调度、workspace 复用、流水同步调整与 full Cross 路由。',
             '- 上游 ROW_4 分支存在但默认 lengths={2,1}；本次未额外启用 ROW_4。dropout>0 仍走 General。',
             '- 为不等长 Q/K 补充实际 K 总长度和 tiling cache key；保留单行 MMAD padding=16。未引入上游无关 BNSD 前向回退及平台专用 libstdc++ 构建设置。',
             '- 正确性、编译对象和二进制身份见 `validation.json`、`provenance.json`，原始 profiler 数据保存在两个运行目录的 profiles/。', '',
             f"- 验证通过：{validation['regression_tests_passed']} 项回归、{len(validation['cpu_reference_cases'])} 项 CPU 双参考、10 项实际场景梯度对照；最大相对 L2 差异 {validation['maximum_relative_l2_vs_corrected_baseline']:.3g}。", '',
             '## 接口与口径', '',
             '- FA2 = `npu_fa_v1.flash_attn` → `flash_attn_npu.flash_attn_npu_interface.flash_attn_varlen_func`。',
             '- Fusion / torch FA = `npu_fa.flash_attn` → `torch_npu.npu_fusion_attention`。',
             '- BF16、TND、D128，bench.py 的五个场景及实际变长前缀。前向为训练前向；反向为预建图后 autograd.grad(retain_graph=True)，不包含前向。',
             '- E2E：每接口预热10次，交替测量3×80次 API+同步，取240个样本中位数；kernel：独立 profiler 15次主 attention kernel 的中位数。热缓存，无cache flush。',
             '- E2E-kernel 包括 Python/Host、分配、辅助设备任务及同步，不能解释为纯 CPU 时间。两者独立测量。',
             '- 耗时单位 ms；FA2/Fusion 和移植版/Host基线均以小于1为更快；E2E/kernel 为同一接口两种测量的比例。',
             '- 汇总同时列出五场景倍率的算术均值和耗时求和后的倍率，二者不混用；不将前向/反向或不同 dropout 混算。', '']
    for p in (0., .1):
        dest = args.ported/f'dropout_{p:g}'
        dest.mkdir(exist_ok=True)
        flat = []
        lines += [f'## dropout={p:g}', '']
        for phase, title in [('forward','前向'),('backward','反向')]:
            lines += [f'### {title}', '',
                      '| 场景 | FA2 E2E | Fusion E2E | FA2 kernel | Fusion kernel | E2E FA2/Fusion | kernel FA2/Fusion | FA2 E2E/kernel | Fusion E2E/kernel | E2E 移植/Host | kernel 移植/Host |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for scenario in labels:
                key = scenario, p, phase
                row, base = index['ported'][key], index['host_baseline'][key]
                item = dict(scenario=scenario, dropout=p, phase=phase)
                for variant, record in [('ported',row), ('host_baseline',base)]:
                    for backend, prefix in [('FA2','fa2'),('torch FA','fusion')]:
                        b = record['backends'][backend]
                        wall, kernel = b['wall_median_us']/1000, b['attention_kernel_median_us']/1000
                        item.update({f'{variant}_{prefix}_wall_ms':wall, f'{variant}_{prefix}_kernel_ms':kernel,
                                     f'{variant}_{prefix}_wall_minus_kernel_ms':wall-kernel,
                                     f'{variant}_{prefix}_wall_over_kernel':wall/kernel})
                aw,bw,ak,bk = [item['ported_'+k] for k in ('fa2_wall_ms','fusion_wall_ms','fa2_kernel_ms','fusion_kernel_ms')]
                wr,kr = aw/bw,ak/bk
                hw,hk = aw/item['host_baseline_fa2_wall_ms'],ak/item['host_baseline_fa2_kernel_ms']
                item.update(wall_fa2_over_fusion=wr,kernel_fa2_over_fusion=kr,
                            wall_ported_over_host=hw,kernel_ported_over_host=hk,
                            wall_reduction_vs_host_percent=(1-hw)*100,kernel_reduction_vs_host_percent=(1-hk)*100,
                            relative_ratio_gap_percent=(wr/kr-1)*100)
                flat.append(item)
                lines.append(f'| {labels[scenario]} | {aw:.4f} | {bw:.4f} | {ak:.4f} | {bk:.4f} | {wr:.3f} | {kr:.3f} | {aw/ak:.3f} | {bw/bk:.3f} | {hw:.3f} | {hk:.3f} |')
            lines += ['']
        write_csv(dest/'summary.csv', flat)
        aggregates = []
        for phase in ('forward','backward'):
            selected = [r for r in flat if r['phase']==phase]
            item = dict(dropout=p,phase=phase,scenarios=len(selected))
            for metric in ('wall','kernel'):
                total = lambda key: sum(r[key] for r in selected)
                item[f'{metric}_fa2_over_fusion_arithmetic_mean'] = mean(r[f'{metric}_fa2_over_fusion'] for r in selected)
                item[f'{metric}_fa2_over_fusion_ratio_of_sums'] = total(f'ported_fa2_{metric}_ms')/total(f'ported_fusion_{metric}_ms')
                item[f'{metric}_ported_over_host_arithmetic_mean'] = mean(r[f'{metric}_ported_over_host'] for r in selected)
                item[f'{metric}_ported_over_host_ratio_of_sums'] = total(f'ported_fa2_{metric}_ms')/total(f'host_baseline_fa2_{metric}_ms')
            aggregates.append(item)
        write_csv(dest/'aggregates.csv', aggregates)
        (dest/'samples.json').write_text(json.dumps({v:dict(metadata=d['metadata'],rows=[r for r in d['rows'] if r['dropout']==p]) for v,d in runs.items()},indent=2)+'\n')
        lines += ['| 汇总 | E2E FA2/Fusion 均值 | E2E FA2/Fusion 总和比 | kernel FA2/Fusion 均值 | kernel FA2/Fusion 总和比 |',
                  '|---|---:|---:|---:|---:|']
        for a in aggregates:
            lines.append(f"| {a['phase']} | {a['wall_fa2_over_fusion_arithmetic_mean']:.3f} | {a['wall_fa2_over_fusion_ratio_of_sums']:.3f} | {a['kernel_fa2_over_fusion_arithmetic_mean']:.3f} | {a['kernel_fa2_over_fusion_ratio_of_sums']:.3f} |")
        lines += ['', f'独立文件：`dropout_{p:g}/summary.csv`、`dropout_{p:g}/aggregates.csv`、`dropout_{p:g}/samples.json`。', '']
    lines += ['## 测量波动', '']
    for variant, result in audit.items():
        worst = max(result['wall_round_spreads'],key=lambda r:r['round_spread_percent'])
        lines.append(f"- {variant}：轮间 E2E 中位数最大极差/中位数为 {worst['round_spread_percent']:.2f}%（{worst['scenario']}，dropout={worst['dropout']:g}，{worst['phase']}，{worst['backend']}）。详见 audit.json。")
    (args.ported/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for filename, payload in [('audit.json',audit),('provenance.json',provenance),('validation.json',validation)]:
        (args.ported/filename).write_text(json.dumps(payload,indent=2)+'\n')
    print('Wrote separate dropout reports:',args.ported)


if __name__ == '__main__':
    main()
