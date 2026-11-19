# Ascend 950 FA3 混合精度 KV 调优方法

`flash_attn_with_kvcache` 使用 BF16 Q 和 E4M3FN KV。K、V 的 FP32 scale 分别按物理 token 和 KV head 索引。KV 在片上反量化为 BF16，QK 和 PV 均使用 BF16 矩阵乘。

## 分析调度与流水

- 分别测量非切分和 Split-KV，从读 Q/K/V 到写出最终 O 和 LSE。比较每个活跃核处理的有效页数，Split-KV 还要计入 partial 的读写、同步和归并。两条路径各自稳定后，用同形状的端到端交点确定自动切分边界。
- SWA 按每块 Q 实际可见的 KV 页估算工作量。低占用 MTP 可以沿 Q head 分工；比较最忙核缩短的时间和重复反量化的成本。Prefill 中，共用同一 KV head 的相邻 Q head 可以共享反量化后的页，各自维护 Softmax、O 和 LSE。
- 检查窗口起点、奇数起始页和尾页。生产者与消费者都按窗口内的相对页号轮转，最后一个消费者读完才能释放 bank。边界 mask 借用 NZ 缓冲区时，核对 MTE3 读完成和 MTE2 写开始之间的依赖。
- 同时看设备时长、Cube 和 Vector 各流水的活跃时间、活跃核数与单核峰值。各流水可以重叠。GM 读量还包含 Q、scale 和 mask，应结合有效 KV 字节与重复读取次数解释。

## 配对验证 full 和 SWA

使用同一扩展、同一组输入和相同的页表对照 FP8 KV 与 BF16 KV。BF16 对照由相同 KV 乘以各自的 scale 后舍入得到。精度同时检查 O 和 LSE，覆盖整页、窗口边界、空行、不等长 batch 以及长序列地址。

测冷缓存时，每次调用使用不同的 KV 地址，在同一 stream 上完成 L2 驱逐并同步，从 trace 读取设备 kernel 时长。比较 split 路径时纳入归并 kernel。采集 PMU 时，每次 application replay 都要先完成驱逐。测量样本与 profiling 报告保存在本地。

在配置好的 950 环境中使用同一基准入口分别运行 full 与 causal SWA-512。窗口包含当前 token。

```bash
python benchmarks/benchmark_mixed_kv_950.py --entry public --cache-protocol distinct --window-size -1 -1 --trace-dir /tmp/fa950-full
python benchmarks/benchmark_mixed_kv_950.py --entry public --cache-protocol distinct --window-size 511 0 --trace-dir /tmp/fa950-swa512
```
