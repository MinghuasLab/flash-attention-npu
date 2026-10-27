# `flash_attn_combine` 内存验收口径（对齐《算子内存标准》）

> 依据：《算子内存标准》（2026-02-05，胡碧霞）——L0/L1/L2 分级看护体系。
> 本文把该标准落到 `flash_attn_combine`（FA3/FA4，A2/A3/A5）上，作为内存验收的唯一口径。

## 1. 标准要点（原文语义）

标准管的是**算子自身引入的额外内存**，不是调用方本来就持有的输入/输出。
按"通过标准"列，每一级都是**两个条件满足其一**即可：

| 等级 | 定义 | 条件 A（相对） | 条件 B（绝对） | 使用场景 |
|---|---|---|---|---|
| **L2** | 关键大模型算子 | 输入输出 **> 500MB** 时，较 GPU 内存 **≤ 5%** | 方案固有 workspace **≤ L2 cache**，且**不随规格等比膨胀** | 业界 top 开源大模型 / 客户关键大模型算子，输入输出 dtype 为模型中的场景 |
| L1 | 大模型算子 | 输入输出 > 500MB 时，较 GPU **≤ 50%** | 同上 | 除 L2 外的大模型算子 |
| L0 | 其他算子 | 输入输出 > 500MB 时，较 GPU **≤ 100%** | 方案固有 workspace ≤ L2 cache 容量 | 小模型算子 |

关键理解：

1. **"其一"意味着 workspace 达标即可，不必再比 GPU**；只有 workspace 超标时才需要走相对判据。
2. **"方案固有 workspace"** 指算子方案本身申请的、不随输入规格放大的缓冲；按规格线性增长的部分不算固有 workspace（也不该出现）。
3. 主流 dtype 场景：`FP16+FP32 / BF16+FP32 / INT32+INT64`。本算子 `out_partial/lse_partial` 为 fp32、`out` 为 fp16/bf16，属主流场景。

### 例外（不计入内存膨胀）

| 类别 | 例外 | 处理 |
|---|---|---|
| 通用 | 非连续 → 连续（前一个 API 的 view 类操作导致） | 不计入 |
| 通用 | 非主流混合 dtype 由 aclnn 插入的 cast | 不计入 |
| A2/A3 | sort 非尾轴排序插入的 transpose | 不计入 |
| A2/A3 | conv2d 为 cube 实现插入的 transdata | 不计入 |
| A5 | sort 非尾轴排序插入的 transpose | 不计入 |

> 本算子的 ATK 执行器为满足 `lse_partial` "seqlen 维连续"契约做了 `transpose+contiguous`，
> 属"非连续转连续"例外；算子内部不做任何布局转换，也不会因此产生额外拷贝。

## 2. 本算子的定级与判定路径

**定级：L2（关键大模型算子）** —— 需求写明"融合算子 L2 双标杆 + 内存不超 L2 阈值"，且它是 top 大模型 attention 的关键子算子。

判定走**条件 B**即可成立：

```
方案固有 workspace = 0  ≤  L2 cache      （对任何 SoC 都成立，因为 0 恒小于任意正数）
```

依据（源码级 + 实测级）：

1. `mha_combine` 内只有两处 `at::empty`：`out`、`lse_out`；**没有 workspace 张量**，`CombineLaunchArgs` 也没有 workspace 字段。
2. kernel 只用 UB（固定 ≤150KiB tile 预算，与规格无关），不申请 GM 中间缓冲。
3. 实测（`NS=8,B=1,S=1024,H=32,D=128`, fp16 out）：

```
inputs = 128.0 + 1.00 MB
before = 130.00  after = 138.13  delta = 8.13 MB   peak_delta = 8.13 MB
out = 8.00 MB   lse_out = 0.12 MB   →  delta 恰等于 out + lse_out
```

即：算子引入的显存 = 输出本身，**workspace 0、无瞬时峰值**。

## 3. 测试方法（三层）

### 3.1 内存正确性（越界 / 多核踩踏 / 未初始化）：msSanitizer

```bash
# 910：使能静态插桩编译
FLASH_ATTN_ENABLE_MSSANITIZER=TRUE FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_BUILD_VERSION=v3 \
  python setup.py build_ext --inplace

# 950（以及本环境的 910）：运行时注入才是可靠路径
mssanitizer --tool=memcheck -- python -m pytest -q tests/test_flash_attn_npu_v3_combine.py
```

本环境实测（CANN 9.3.0）：静态插桩版直接运行会**段错误**（既有 `test_flash_attn_npu_v3.py` 同样崩），
必须用 `mssanitizer --tool=memcheck` 包装运行；`FaCombine` kernel 结论 **No error detected**，
仅有 `FFTS_BASE_ADDR` 未复位告警（`SetSyncBaseAddr` 的寄存器状态告警，非访存错误）。

> 插桩版会覆盖 in-place `.so`，测完必须重编或还原为正常版，否则 ATK/pytest 会崩。

### 3.2 资源量化（峰值 / workspace）

```python
torch.npu.synchronize(); torch_npu.npu.reset_peak_memory_stats()
base = torch_npu.npu.memory_allocated()
out, lse = flash_attn_npu_3.flash_attn_combine(op, lse, out_dtype=torch.float16)
torch.npu.synchronize()
print((torch_npu.npu.memory_allocated() - base) / 2**20,
      (torch_npu.npu.max_memory_allocated() - base) / 2**20)
```

ATK 侧用内存任务（`accuracy` 任务的三列内存字段为空，不能用它出结论）：

```bash
atk node --backend npu --devices 0 \
  node --backend gpu --devices 0 --name gpu_base \
  --bm_file ./all_test_flash_attn_combine_reports_2026-10-20-07-10-25.xlsx \
  task -c ./verify_182_fa3.json \
  -p ./execute_flash_attn_combine.py --task memory_device
```

期望：`Workspace大小(MB) = 0`，`分配内存` 与 `out + lse_out` 理论值一致。

### 3.3 判定

```
条件 B（优先）：固有 workspace ≤ L2 cache 且不随规格膨胀   → 本算子 0，满足
条件 A（补充）：I/O > 500MB 的 shape 上  (NPU峰值−GPU峰值)/GPU峰值 ≤ 5%
```

## 4. 当前证据与缺口

| 项 | 状态 |
|---|---|
| workspace = 0（源码 + torch 计数器实测） | ✅ |
| msSanitizer memcheck 无访存错误 | ✅（运行时注入路径） |
| 200 条 ATK 用例的 I/O 规模 | 最大 48.2MB（id 182/199），**未触发条件 A** |
| 业务大 shape（I/O > 500MB）NPU vs GPU 峰值对照 | ❌ 待补，需要 GPU 机器同 shape 采集 |
| A2/A3/A5 的 L2 cache 具体容量 | ❌ 标准未给数值，`npu-smi` 也不输出；需需求方/系统规格提供（本算子结论不受影响，0 恒满足） |

## 5. 待需求方确认

1. "输入输出超 500M" 的单位是否即 500MB；
2. "绝对值不超 L2 cache" 的判定对象是否就是"方案固有 workspace"；
3. A2 / A3 / A5 各自的 L2 cache 容量数值；
4. 内存峰值统计口径（最大用例 / P99 / 典型业务 shape）。
