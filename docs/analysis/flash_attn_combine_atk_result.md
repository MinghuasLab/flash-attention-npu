# `flash_attn_combine` ATK 精度测试结果（A2/A3，L2 双标杆）

> 环境（按要求重新验证）：
> `source /data/w00896777/ops/0930/ascend-toolkit/set_env.sh`（CANN 9.3.0 weekly 20260930）
> `conda activate wk`（Python 3.10.18，torch 2.7.1，torch_npu 2.7.1.post2，ATK 26.9.24）
> 硬件：8× Ascend910B3（测试固定在 device 3）。
> 用例工程：`tests/st/atk_flash_attn_combine`；用例 JSON 与 GPU 基线在 `/data/w00896777/ATK/flash_attn_combine`。

## 0. 结论

| 后端 | 用例数 | 执行成功 | 精度通过 | 通过率 | `acc_pass_result` |
|---|---|---|---|---|---|
| FA3（双标杆，有 GPU 基线的 182 条） | 182 | 182 | 182 | **100.0%** | **Pass** |
| FA4（双标杆，有 GPU 基线的 182 条） | 182 | 182 | 182 | **100.0%** | **Pass** |
| FA3 / FA4（新增 18 条，单标杆 npu vs cpu） | 18 | 18 | 18 | **100.0%** | **Pass** |

> 新增 18 条（id 182..199）在 GPU 基线刷新前只能用单标杆验证；刷新后即可并入双标杆统计。

三项 L2 比值的全局最大值（两个后端相同，因为 910 v3/v4 kernel 现在是同一份实现）：

| 指标 | L2 阈值 | 实测最大值 |
|---|---|---|
| 最大相对误差比 MARE | ≤ 2 | **1.7311** |
| 平均相对误差比 MERE | ≤ 1.2 | **0.0023** |
| 均方根误差比 RMSE | ≤ 1.2 | **0.8047** |

报告：

- FA3：`/data/w00896777/ATK/flash_attn_combine/atk_output/all_test_flash_attn_combine_bf16fp16_2026-10-20-17-20-30-934622/report/all_test_flash_attn_combine_bf16fp16_reports_2026-10-20-17-20-32.xlsx`
- FA4：`/data/w00896777/ATK/flash_attn_combine/atk_output/all_test_flash_attn_combine_v4_bf16fp16_2026-10-20-17-13-36-902844/report/all_test_flash_attn_combine_v4_bf16fp16_reports_2026-10-20-17-13-38.xlsx`

## 1. 用例设计

### 1.1 算子事实表（来源：`flash_attn_npu_3/flash_attn_npu_interface.py` + `flash_api.cpp::mha_combine`）

| 维度 | 事实 |
|---|---|
| 接口 | Python API，无 aclnn 封装（`api: pytorch` + 自定义 executor） |
| `out_partial` | fp32；rank 5 `(NS,B,S,H,D)` 或 rank 4 `(NS,T,H,D)`；`stride(-1)==1` |
| `lse_partial` | fp32；shape == `out_partial.shape[:-1]`；`stride(-2)==1`（seqlen 维连续） |
| `out_dtype` | fp16 / bf16（本需求范围）；fp32 亦支持但不在本需求用例内 |
| 输出 | `(out, lse)`；`out` 按 `out_dtype`，`lse` 恒 fp32 |
| 约束 | `1 ≤ num_splits ≤ 256`；两 tensor 必须 fp32；shape 强耦合 |

### 1.2 三个产物

| 文件 | 作用 |
|---|---|
| `test_flash_attn_combine.yaml`（FA3）/ `test_flash_attn_combine_v4.yaml`（FA4） | 参数空间 + `standard.acc.cv_fused_double_benchmark`（**L2: 2 / 1.2 / 1.2**）+ `perf: not_key` |
| `flash_attn_combine_constraint.py` | 跨输入修正（`lse_partial` shape 跟随 `out_partial`、`out_dtype` 限定 bf16/fp16、`-inf` 短路场景） |
| `execute_flash_attn_combine.py` | CPU fp64 golden；NPU 调 `flash_attn_npu_3/4.flash_attn_combine`；`function_flash_attn_combine` / `..._v4` 两个注册名 |

### 1.3 用例规模：每后端 200 条

原始 GPU 基线报告是每后端 200 条（98 bf16 + 84 fp16 + 18 fp32）。按"需求只覆盖 bf16/fp16"剔除
18 条 fp32 输出后得到 **182 条基线可比值**，再用 `extend_cases_to_200.py` 补 18 条（id 182..199）凑满 **200 条**。
新增 18 条只补覆盖缺口，不复用被剔除的 fp32 id；它们需要一次 GPU 基线刷新才能进入双标杆统计。

| 维度 | 覆盖 |
|---|---|
| 布局 | BSND(5-D) / TND(4-D) 各半 |
| 输出 dtype | fp16 / bf16 |
| num_splits | 1 / 2 / 3 / 5 / 8 / 17 / 32 |
| head_dim | 64 / 96 / 128 / 192 / 256（+113 边界） |
| seqlen/token | 1 / 16 / 33 / 113 / 257 / 512，含空 tensor(0) 边界 |
| batch / heads | 1/2/4；1/2/4/8 |
| 值域 | 均匀 [-5,5]、正态 μ∈[-100,100] σ∈[1,25]（覆盖 fp16 大值域 ULP 敏感区） |
| `-inf` 短路 | 全 split 掩码行 |

新增 18 条（两个后端相同，均 bf16/fp16）：

| id | 布局 | shape | 覆盖点 |
|---|---|---|---|
| 182 | TND | `[64,128,8,128]` | splits=64 |
| 183 | TND | `[256,16,2,64]` | splits=256（接口上限） |
| 184 | TND | `[7,65,16,80]` | splits=7 + D=80 非对齐 |
| 185 | TND | `[15,7,3,48]` | splits=15 + 尾块 T=7 |
| 186 | TND | `[32,1000,1,256]` | 长序列 T=1000 + D=256（bf16） |
| 187 | TND | `[9,2,2,32]` | D=32 最小值 |
| 188 | TND | `[6,129,64,96]` | splits=6 + H=64 |
| 189 | TND | `[128,4,8,192]` | splits=128 |
| 190 | TND | `[17,513,5,113]` | D=113 + T=513 尾块 |
| 191 | TND | `[3,257,12,160]` | H=12 + D=160 |
| 192 | BSND | `[64,2,8,4,128]` | splits=64 dense |
| 193 | BSND | `[256,1,4,2,64]` | splits=256 dense |
| 194 | BSND | `[8,6,33,16,96]` | B=6 + H=16 |
| 195 | BSND | `[9,1,1,12,32]` | decode S=1 + D=32 |
| 196 | BSND | `[128,1,2,8,256]` | splits=128 + D=256 |
| 197 | BSND | `[5,12,7,3,200]` | B=12 + D=200 非对齐 |
| 198 | BSND | `[15,4,65,5,48]` | splits=15 + S=65 |
| 199 | BSND | `[64,1,128,8,128]` | splits=64 + S=128 |

## 2. 精度标准（L2 双标杆）

`standard.acc = cv_fused_double_benchmark`，按 **L2** 取值：`ratio = NPU指标 / max(GPU指标, 2^-14)`。

| 等级 | 最大相对误差比 | 平均相对误差比 | 均方根误差比 |
|---|---|---|---|
| L0 | ≤ 10 | ≤ 2.0 | ≤ 2.0 |
| L1 | ≤ 5 | ≤ 1.5 | ≤ 1.5 |
| **L2（本次）** | **≤ 2** | **≤ 1.2** | **≤ 1.2** |

比对拓扑：`atk node --backend npu` + `node --backend gpu --bm_file <GPU 报告>`，
即 NPU 实测 vs CPU golden，GPU 基线 vs 同一 CPU golden，再取比值。每个用例比较 `out` 与 `lse` 两个输出。

## 3. 从 180/182 到 182/182：误差根因与修复

### 3.1 现象

首轮（顺序 fp32 累加 + `w_s = exp(lse_s - lse)`）：**180/182**，两条 fp16 用例超限——

| 用例 | shape | 指标 | NPU | GPU | 比值 |
|---|---|---|---|---|---|
| id 59（原 70） | `[8,8,1,32,128]` | RMSE | 7.72e-4 | 5.98e-4 | **1.291** |
| id 71（原 78） | `[16,4,32,128]` | RMSE | 8.46e-4 | 6.90e-4 | **1.225** |

### 3.2 根因

这两条用例的输出量级 ~100（`mean ∈ [-100,100]`），fp16 的 ULP 是 0.0625，而 golden 也是"fp32 参考 → cast fp16"。
于是 RMSE 不再由"实现有多准"决定，而由**有多少元素恰好跨过舍入边界**决定：

```
翻转概率 ≈ |pre-cast 相对误差| / fp16 相对 ULP        (相对 ULP ≈ 2^-11)
RMSE ≈ sqrt(翻转数 / 元素数) × 1 ULP
```

`extra-info/probe_ways.py` 的 CPU 仿真（20 组随机数据）给出误差地板：

| 累加方式 | vs golden 的 RMSE 均值 |
|---|---|
| 与 CPU 完全同序 fp32 累加 | **0**（逐位一致） |
| 2 路 | 1.93e-4 |
| 4 路 | 1.78e-4 |
| 8 路 | 2.08e-4 |
| float64 精确 | 1.70e-4 |

即：任何与 CPU 不同的累加顺序都要付 ~2e-4 的代价，这是本指标的噪声地板；而 GPU 侧当时是 6.9e-4、NPU 是 8.5e-4~1.2e-3，
说明两边都离地板还远，NPU 侧的 pre-cast 相对误差（~7.7e-7 ≈ 6 ULP）才是主因。

### 3.3 修复（两处，均为 kernel 内数值路径）

1. **权重改成"先归一化、后除"**：`w_s = exp(lse_s - max) / Σ exp(lse_s - max)`，
   取代 `w_s = exp(lse_s - lse_out)`。exp 的舍入在分子分母中同源抵消，且省掉了 `ln → exp` 往返带来的 1 ULP LSE 误差；
   该形式顺带少做一次 `(ns × rows)` 的 Exp，只在最后做一次 `(rows × D)` 的 Div。
2. **累加改 Kahan 补偿求和**：`y = t - c; s = acc + y; c = (s - acc) - y`。
   2 路累加在 `num_splits` 增大（接口上限 256）后误差随段长增长；Kahan 与段长无关，恒在 ~1 ULP。

两处叠加后 pre-cast 相对误差从 ~7.7e-7 降到 ~5.8e-7，`id 59`（原 70）的 RMSE 比从 1.291 降到 1.155，
`id 71`（原 78）在最终版本通过，**182/182 全过**。

> 中间验证过程（可用于回归对照）：
> 顺序累加 180/182 → 2 路累加 181/182（33 条用例数值变化，两条超限用例只解决一条）→
> 归一化权重 + Kahan **182/182**。

### 3.4 边界语义保持

- 全 `-inf` 行：`max` 被 clamp 到 `lowest()`，`e_s = 0`，`den = 0` → clamp 到 1 → `out = 0`（与 NV 一致、无数值异常）。
- 正常行 `den ≥ 1`（max 分片贡献 `exp(0)=1`），clamp 不生效。
- `lse` 输出仍走 `max + ln(Σexp)`，全掩码行仍由标量修正回 `-inf`。

### 3.5 新增用例暴露并修复的真缺陷：head_dim 非 16 对齐

补齐到 200 条时新增了 `D=113`（id 190，原 208）和 `D=200`（id 197，原 215）两条，首跑直接出现 `MARE 18816 / 520`。
独立探针 `extra-info/probe_unaligned.py` 复现：

| shape | 结果 |
|---|---|
| `[17,513,5,113]` fp16 | 每个 lane 相对 RMS 误差 ≈ 1.3（全错） |
| `[5,12,7,3,200]` bf16 | 每个 lane 相对 RMS 误差 ≈ 1.3（全错） |
| `[17,513,5,112]` fp16（对照） | 精确 |
| `[5,12,7,3,192]` bf16（对照） | 精确 |

根因：UB 侧行距是 `dA = AlignUp(headDim, 16)`（输出 staging 需要 32B 对齐），
但 `DataCopyPad` 在 `dstStride=0` 时按 **32B 对齐的 blockLen** 连续打包，
实际行距是 `AlignUp(headDim, 8)`。两者只在 `head_dim % 16 == 0` 时相等，
否则每行按 `dA - AlignUp(head_dim,8)` 个 lane 漂移，整块数据错位。

修复（910/950 四份 kernel 同步）：

```cpp
const uint32_t oDstStride =
    (dA * sizeof(float) - AlignUp(headDim * sizeof(float), 32U)) / 32U;
DataCopyExtParams(nrows, headDim * sizeof(float), srcGapBytes, oDstStride, 0U);
```

该值在 `head_dim % 16 == 0` 时恒为 0，所以原有 182 条用例的数值**逐位不变**
（复测 182/182 Pass 已验证）；非对齐时按 32B 单位补齐行距。

> 由此得到一条硬件约束（已写入实现文档 §2.5）：**只要 UB 内的行距与
> `AlignUp(blockLen, 32B)` 不一致，就必须显式传 `dstStride`**，不能依赖默认打包。

## 4. 过程中修的两处用例/接口适配（非算子缺陷）

1. **`lse_partial` 布局**：ATK 生成 (NS,B,S,H) 连续张量，接口要求 seqlen 维连续。执行器显式分配 `(NS,B,H,S)` 缓冲再转置——
   `S=1` 时 `transpose+contiguous` 是空操作（size-1 维不参与连续性判断），必须用缓冲拷贝。
2. **`+inf` 输入**：boundary 默认 `has_infnan=True`，会生成 `lse_partial=+inf`；该输入在 NV 契约下未定义（全 `-inf` 才是合法短路）。
   关闭 `lse_partial` 的 infnan 边界，保留 `-inf`。

## 5. 复现

```bash
source /data/w00896777/ops/0930/ascend-toolkit/set_env.sh
source /data/miniconda3/etc/profile.d/conda.sh && conda activate wk
export ASCEND_RT_VISIBLE_DEVICES=3
export FLASH_ATTN_NPU_ROOT=/data/w00896777/code/flash-attention-npu
cd /data/w00896777/ATK/flash_attn_combine

# FA3（182 条，GPU 基线做分母）
atk node --backend npu --devices 0 \
  node --backend gpu --devices 0 --name gpu_base \
  --bm_file ./all_test_flash_attn_combine_reports_2026-10-20-07-10-25.xlsx \
  task -c ./all_test_flash_attn_combine_bf16fp16.json \
  -p ./execute_flash_attn_combine.py --task accuracy

# FA4（182 条）
atk node --backend npu --devices 0 \
  node --backend gpu --devices 0 --name gpu_base \
  --bm_file ./all_test_flash_attn_combine_v4_reports_2026-10-20-07-08-39.xlsx \
  task -c ./all_test_flash_attn_combine_v4_bf16fp16.json \
  -p ./execute_flash_attn_combine.py --task accuracy
```

注意：

- 主节点必须用 `--backend npu`；用 `--backend pyaclnn` 会因无 aclnn 封装报 `请在用例yaml中补充aclnn_name字段`。
- `--bm_file` 的 case 必须与 `-c` 的 JSON **逐条同 shape**，否则 ATK 退化成回放，比值无意义
  （反建脚本：`/data/w00896777/ATK/flash_attn_combine/rebuild_cases_from_report.py`）。
### 5.1 200 条用例与 GPU 基线刷新

两个用例文件现在各有 200 条（182 条有旧基线 + 18 条新增 id 182..199）。
新增 18 条在旧 `--bm_file` 里没有分母，要跑完整双标杆必须先刷新基线：

```bash
# 在 GPU 机器上（NV 环境）对同一份 200 条 JSON 跑一次基线
atk node --backend gpu --devices 0 --name gpu_base \
  task -c all_test_flash_attn_combine_bf16fp16.json \
  -p execute_flash_attn_combine.py --task accuracy

# 回到 NPU 侧，用新生成的报告做分母
atk node --backend npu --devices 0 \
  node --backend gpu --devices 0 --name gpu_base \
  --bm_file ./<新生成的>_reports_*.xlsx \
  task -c ./all_test_flash_attn_combine_bf16fp16.json \
  -p ./execute_flash_attn_combine.py --task accuracy
```

刷新前如需复现"182 条全过"，直接用备份文件（`.182.bak`，已核验与旧基线逐条对齐）。

### 5.2 用例 id 重排（0..199）与旧基线重映射

两个用例文件的 `id` 已按文件顺序重排为 **0..199**：

- 脚本：`tests/st/atk_flash_attn_combine/renumber_case_ids.py`
- 备份：`*.json.pre_renumber.bak`（重排前的原始 id，旧报告就是按它产生的）
- 注意：FA3 与 FA4 两个文件的**用例顺序不同**（来自两次不同的 GPU 生成），所以同一个旧 id 在两个文件里对应的新 id 也不一定相同

ATK 是按 `编号` 对齐基线的，因此重排后旧报告需要按内容重映射：

- 脚本：`tests/st/atk_flash_attn_combine/rekey_report_ids.py`
- 产物（`编号` 已换成新 id，可用 `--bm_file` 直接配 200 条 json）：
  - `rekeyed_all_test_flash_attn_combine_bf16fp16_reports_2026-10-21-01-02-43.xlsx`（FA3 accuracy，200/200 对齐）
  - `rekeyed_all_test_flash_attn_combine_v4_bf16fp16_reports_2026-10-21-01-04-52.xlsx`（FA4 accuracy，200/200）
  - `rekeyed_memory-all_test_flash_attn_combine_bf16fp16_reports_2026-10-21-01-59-15.xlsx`（FA3 memory，199/199；原报告缺 1 条）
  - `rekeyed_memory-all_test_flash_attn_combine_v4_bf16fp16_reports_2026-10-21-02-08-20.xlsx`（FA4 memory，200/200）

## 6. 环境与构建

- `wk` 环境为 Python 3.10，仓库原有 `.so` 为 cp312，需在 `wk` 内重编：
  `FLASH_ATTN_BUILD_VERSION=v3|v4 FLASH_ATTN_BUILD_NPU=910|950 python setup.py build_ext --inplace`
- 950 产物名为 `flash_attn_npu_{3,4}_950.cpython-310-*.so`，本机无 A5 设备，仅编译/链接验证。
- ATK 26.9.24 的 `cv_fused_double_benchmark_compare` 是编译插件，源包中不可见；本环境已注册可用。
