# 保持原项目 Host 优化的实验副本

此文记录 `bf5eaed` 的纯 Host 基线。当前工作树随后按用户要求移植了上游设备优化，详见 [BASIC_OPT_PORT.md](BASIC_OPT_PORT.md)；纯 Host 快照位于 `../host_original_before_basic_opt`。

本副本替代旧 `host_only` 作为“保留原项目 Host 优化”的实验版本。
旧副本及历史结果保留不动，原项目 `/home/hxb/flash-attention-npu` 保持不动。

保留原项目的全部反向 Host/Python 优化：

- `flash_api.cpp` 与原项目一致：均匀前缀证明、前缀同一对象判断、Opt tiling 缓存、causal mask 复用及完整注意力路由。
- `flash_attn_npu_interface.py` 与原项目一致：eager 快路、编译/变换模式回退、padding 检查和 slice 优化。
- `fag_general_host.cpp` 保留原项目的均匀前缀重建、按流隔离的 tiling 缓存和 mask 复用。

仅移除与设备专用实现绑定的 `singleTile/direct-output` 状态及调用入口，使用通用设备 kernel。
保留 `preferSingleTileFull` 的 Host 路由条件，所以 T5 Encoder dropout=0 与原项目一样走 General，但执行通用版本。
未叠加旧 `host_only` 独有的 static tiling 实验；非均匀前缀行为遵循原项目。

设备代码基于实验 corrected baseline：`47fec0c` 加容量取整、MMAD 单行尾块共同正确性修复。
前向保持该实验基线，不引入原项目另行进行的 KV-cache 设备修改。
相对 corrected baseline 的 94 个编译对象，仅两个 Host 对象不同，无新增设备对象。

- [与原项目的必要适配差异](excluded_device_wiring.patch)
- [源码及二进制校验清单](host_only_manifest.json)
- [完整测量报告](../results/host_original/REPORT.md)
- [前向/反向耗时与比例](../results/host_original/summary.csv)
- [与旧版及原项目历史数据对照](../results/host_original/comparison.csv)
- [正确性验证结果](../results/host_original/validation.json)

FA2 = `flash_attn_varlen_func`；Fusion / torch FA = `torch_npu.npu_fusion_attention`。
测量五个场景、dropout=0/0.1、训练前向及独立反向；E2E 和主 kernel 分开采样。

```bash
# 输出目录必须不存在。
bash run_host_only.sh ../results/host_original_rerun
python write_host_only_report.py ../results/host_original_rerun
```
