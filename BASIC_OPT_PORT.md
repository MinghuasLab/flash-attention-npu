# fag_v2_basic_opt 移植

Host 基线已提交为 `bf5eaed`，共同正确性修复单独提交为 `1705f43`。
当前工作树在该基线上移植反向设备优化；纯 Host 版本仍可由基线 commit 和 `../host_original_before_basic_opt` 快照复现。
不执行 git push，原项目 `/home/hxb/flash-attention-npu` 未修改。

来源：[main...fag_v2_basic_opt](https://github.com/MinghuasLab/flash-attention-npu/compare/main...caihuali373:flash-attention-npu:fag_v2_basic_opt)。
固定上游 base `68b4cc7f7438509fbedd8f974ccbe28275604821`，head `67120b74754f8ec7cfae76707bd15403c7693e7b`。

移植内容：

- Cube1/2 与 Cube3 使用双地址列表；Cube/Vector 的 full/causal 调度配套更新。上游默认 `lengths={2,1}`，未额外启用 ROW_4。
- P/dS 复用 mm2/mm1 float workspace，并配套调整半精度地址、Vector 写回和 Cube 读取。
- Cube2 等待 Vector 的位置后移，使首次 K 搬运与 Vector 计算重叠；更新 Cube3 的搬运及 dk/dv 执行顺序。
- 允许 dropout=0、D128、非局部且非 causal 的 Cross 走 Opt；dropout>0 继续走 General。

本地适配：

- 保留 Host 均匀前缀证明、同一 Tensor 快速判断、按流隔离的 tiling/mask 缓存和 Python eager 快路。
- Opt tiler 使用实际 Q/K 总长度，cache key 加入 K 总长度，避免 Cross 重用不匹配的 workspace。
- 保留已验证的单行 MMAD padding=16；新 full 调度按剩余容量向下取整。
- 不引入上游撤销 BNSD 前向正确性修复的两处改动。
- 不引入上游为其他主机 GCC/libstdc++ 环境添加的 setup.py 设置，本机沿用已经验证的构建配置。

验证与测量：

- 既有 backward cache/动态前缀/多流/FakeTensor/torch.compile 测试。
- 新增跨分块边界、不等长 Q/K、MHA/GQA、空段和变化 K 总长度缓存测试，采用 CPU FP32/低精度双参考。
- 共同容量/尾块回归，以及五场景两种 dropout 的完整梯度对照。
- 正确性通过后，依次重新测量 Host 基线和移植版；每接口每项 240 个 E2E 样本、15 个独立主 kernel 样本。
- FA2 = `flash_attn_varlen_func`；Fusion = `torch_npu.npu_fusion_attention`。
- 按 dropout=0 和 dropout=0.1 分目录输出 CSV、样本 JSON、前向/反向表及各自汇总，不跨 dropout 混算。

```bash
bash run_basic_opt_comparison.sh ../results/basic_opt_comparison
```

[测量报告](../results/basic_opt_comparison/ported/REPORT.md) ·
[dropout=0](../results/basic_opt_comparison/ported/dropout_0/summary.csv) ·
[dropout=0.1](../results/basic_opt_comparison/ported/dropout_0.1/summary.csv)
