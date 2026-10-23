# `flash_attn_combine` 交付件清单

> 需求：Flash Attention 3/4 支持 `flash_attn_combine` 接口（A2/A3/A5，bf16/fp16，BSND/TND，融合算子 L2 双标杆）。
> 状态：代码与自测试已完成；A2/A3 上板验证通过，A5 编译验证通过（本机无 950 设备）。
> PR：https://github.com/MinghuasLab/flash-attention-npu/pull/349
> 分支：`AnnYE131:codex/flash-attn-combine`（单 commit，已 rebase 到 `main`）

## 0. 交付摘要

| 类别 | 数量 | 状态 |
|---|---|---|
| C++ kernel（新增） | 8 个文件（4 backend × kernel + host ABI） | 910 编译/上板通过；950 编译/链接通过 |
| host 注册（修改） | 4 个 `flash_api.cpp` | 完成 |
| Python 接口（修改） | 6 个文件（v3/v4 × 910/950 + 2 个 `__init__`） | 完成 |
| 构建脚本 | `setup.py`（4 个 source list） | 完成 |
| 文档 | README 双语 + 4 份分析/结果文档 | 完成 |
| pytest 用例 | 2 个文件，**127 例** | **127 passed** |
| ATK 用例 | 6 个用例/执行文件 + 1 个扩容脚本，**400 条**（FA3 200 + FA4 200，bf16/fp16） | 182×2 双标杆 **Pass**；新增 18×2 单标杆 **Pass**（待 GPU 基线刷新并入双标杆） |

## 1. 代码交付件

### 1.1 新增 C++ kernel 与 host ABI

| 文件 | 行数 | 作用 | 编译验证 |
|---|---|---|---|
| `csrc/ascend910/flash_attn_npu_3/combine_kernel.cpp` | 325 | 910 版 combine Vector kernel（归一化权重 + Kahan fp32 累加、BSND/TND 由 stride 表达、无 GM workspace） | ✅ 910 build+上板 |
| `csrc/ascend910/flash_attn_npu_3/combine_dispatch.hpp` | 59 | host ABI（`CombineLaunchArgs` + 3 个 dtype launcher） | ✅ |
| `csrc/ascend910/flash_attn_npu_4/combine_kernel.cpp` | 325 | FA4 910 后端（与 v3 逐字节相同） | ✅ |
| `csrc/ascend910/flash_attn_npu_4/combine_dispatch.hpp` | 59 | 同上 | ✅ |
| `csrc/ascend950/flash_attn_npu_3/combine_kernel.cpp` | 299 | 950 版（`CATLASS_GLOBAL` + `__DAV_VEC__`，per-row Muls 权重广播 + Kahan） | ✅ dav-3510 编译/链接 |
| `csrc/ascend950/flash_attn_npu_3/combine_dispatch.hpp` | 52 | host ABI（无 ffts） | ✅ |
| `csrc/ascend950/flash_attn_npu_4/combine_kernel.cpp` | 299 | FA4 950 后端（与 v3 逐字节相同） | ✅ |
| `csrc/ascend950/flash_attn_npu_4/combine_dispatch.hpp` | 52 | 同上 | ✅ |

### 1.2 修改：host 注册（4 个）

| 文件 | 改动 |
|---|---|
| `csrc/ascend910/flash_attn_npu_3/flash_api.cpp` | 新增 `mha_combine`（校验/形状推导/tiling/launch）+ `m.def("fwd_combine", ...)`（`py::arg` 默认值对齐 NV） |
| `csrc/ascend910/flash_attn_npu_4/flash_api.cpp` | 同上 |
| `csrc/ascend950/flash_attn_npu_3/flash_api.cpp` | 同上（无 `rtGetC2cCtrlAddr`） |
| `csrc/ascend950/flash_attn_npu_4/flash_api.cpp` | 同上 |

校验覆盖：partial/LSE 必须 fp32、`stride(-1)==1` / `stride(-2)==1`、`num_splits ≤ 256`、`out` dtype/shape、`out_dtype ∈ {fp16,bf16,fp32}`（默认 fp32）、`cu_seqlens/seqused` int32、`return_lse`。

### 1.3 修改：Python 接口（6 个）

| 文件 | 改动 |
|---|---|
| `flash_attn_npu_3/flash_attn_npu_interface.py` | `flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None)` + docstring（NV FA3 签名） |
| `flash_attn_npu_3/flash_attn_npu_interface_950.py` | 950 后端同名接口 |
| `flash_attn_npu_3/__init__.py` | 910/950 两个分支均导出 |
| `flash_attn_npu_4/flash_attn_npu_interface.py` | FA4 签名（含 `cu_seqlens/seqused/virtual_batch_idx/return_lse`；`virtual_batch_idx` 显式 `NotImplementedError`） |
| `flash_attn_npu_4/flash_attn_npu_interface_950.py` | FA4 950 后端 |
| `flash_attn_npu_4/__init__.py` | 两个分支均导出 |

### 1.4 修改：构建

| 文件 | 改动 |
|---|---|
| `setup.py` | `src_ascend910_v3/v4`、`src_ascend950_v3/v4` 各补 `combine_kernel.cpp` |

### 1.5 文档交付

| 文件 | 内容 |
|---|---|
| `README.md` / `README.zh.md` | v3/v4 的 `flash_attn_combine` 用法章节 + Features 表 |
| `docs/analysis/flash_attn_combine_requirement_analysis.md` | 需求维度分析（L2/性能/内存口径） |
| `docs/analysis/flash_attn_combine_interface_spec.md` | 接口规格（签名/形状/dtype/报错/验收） |
| `docs/analysis/flash_attn_combine_interface_implementation.md` | 实现分析（切分/UB/文件清单/实施结果/硬件约束） |
| `docs/analysis/flash_attn_combine_atk_result.md` | ATK 364 条与 L2 双标杆结果（含误差根因与两处数值修复） |
| `docs/analysis/flash_attn_combine_A5_handoff.md` | **A5（Ascend 950）验证交付件**：需求原文、接口契约、实现说明、验证手册、判据、已知边界 |
| `tests/st/atk_flash_attn_combine/run_a5_verify.sh` | A5 一键验证脚本（环境检查 + 950 编译 + 冒烟 + pytest，可选 ATK） |
| `tests/st/atk_flash_attn_combine/package_a5_delivery.sh` | 打包脚本（源码 + ATK 用例/基线 + 文档 + 一键脚本 + MANIFEST） |

## 2. 自测试交付件

### 2.1 pytest（op 级，2 个文件 / 127 例）

| 文件 | 例数 | 内容 |
|---|---|---|
| `tests/test_flash_attn_npu_v3_combine.py` | **94** | BSND 72（dtype×num_splits×head_dim×seqlen，含 `-inf` 段）+ TND 18 + 4 个专项（默认 dtype/原地写/全 mask/错误契约） |
| `tests/test_flash_attn_npu_v4_combine.py` | **33** | `seqused` dense 24 + `cu_seqlens+seqused` varlen 6 + `return_lse=False`/原地写 + `virtual_batch_idx` not-supported + 错误契约 |

判据：golden = `lse=logsumexp(lse_partial,0)`、`out=Σ exp(lse_s−lse)·O_s`；`lse` atol/rtol 1e-4；`out` 偏差 ≤ 2× 一次 cast 误差 + 1e-3。

### 2.2 ATK（接口级，7 个文件 / 400 条）

| 文件 | 作用 |
|---|---|
| `tests/st/atk_flash_attn_combine/test_flash_attn_combine.yaml` | FA3 用例设计（`standard.acc.cv_fused_double_benchmark` = 2/1.2/1.2，`perf: not_key`） |
| `tests/st/atk_flash_attn_combine/test_flash_attn_combine_v4.yaml` | FA4 用例设计（同输入空间） |
| `tests/st/atk_flash_attn_combine/flash_attn_combine_constraint.py` | 跨输入约束 + 确定性场景遍历 |
| `tests/st/atk_flash_attn_combine/execute_flash_attn_combine.py` | CPU fp64 golden + NPU 执行器（FA3/FA4 两个注册名） |
| `tests/st/atk_flash_attn_combine/run_atk.sh` | 一键复现（`atk case` ×2 + `atk pytorch` ×2） |
| `tests/st/atk_flash_attn_combine/extend_cases_to_200.py` | 把 182 条基线可比值扩到 200 条（补 id 182..199，只补 bf16/fp16，幂等） |
| `tests/st/atk_flash_attn_combine/renumber_case_ids.py` | 把用例 `id` 重排为 0..199（写 `*.pre_renumber.bak` 备份） |
| `tests/st/atk_flash_attn_combine/rekey_report_ids.py` | 按用例内容把旧 GPU 报告（xlsx）的 `编号` 重映射到新 id，供 `--bm_file` 直接使用 |
| `tests/st/atk_flash_attn_combine/.gitignore` | 忽略 `atk_output/`、`result/`、`atk.log` |

覆盖矩阵（每后端 182 条，全部 bf16/fp16 且逐条有 GPU 基线）：BSND/TND 各半；num_splits 1/2/3/5/8/17/32；head_dim 64/96/128/192/256（+113 边界）；seqlen 1/16/33/113/257/512 + 空 tensor；batch 1/2/4；heads 1/2/4/8；值域含均匀 [-5,5] 与正态 μ∈[-100,100] σ∈[1,25]；`-inf` 短路行。

> 用例文件构成：182 条与旧 GPU 基线逐条对齐 + 18 条新增（id 182..199，splits 6/7/9/15/64/128/256、
> head_dim 32/48/80/113/160/200、T/S 2/7/65/129/513/1000、H 3/12/16、B 6/12）。
> 新增 18 条需在 GPU 机器上重跑一次基线才能并入双标杆；此前用单标杆（npu vs cpu）验证，18/18 Pass。

### 2.3 运行产物（不入库）

| 产物 | 位置 |
|---|---|
| ATK 精度报告 | `tests/st/atk_flash_attn_combine/atk_output/<run>/report/*.xlsx` |
| ATK 用例 JSON | `tests/st/atk_flash_attn_combine/result/<tag>/json/all_<tag>.json` |
| ATK 日志 | `tests/st/atk_flash_attn_combine/atk_output/<run>/log/atk.log` |

## 3. 验证证据

| 项 | 命令 | 结果 |
|---|---|---|
| 910 编译 | `FLASH_ATTN_BUILD_VERSION=v3\|v4 FLASH_ATTN_BUILD_NPU=910 python setup.py build_ext --inplace` | ✅ 4 个扩展链接成功 |
| 910 安装 | `python setup.py install`（v3/v4） | ✅ 导出 `flash_attn_combine`，原有接口不受影响 |
| 950 编译 | `FLASH_ATTN_BUILD_NPU=950`（`--npu-arch=dav-3510`） | ✅ v3/v4 编译+链接成功（未上板） |
| pytest | `pytest tests/test_flash_attn_npu_v{3,4}_combine.py` | ✅ **127 passed**（Kahan 版本，wk 环境 20s） |
| ATK FA3 | `atk node --backend npu ... node --backend gpu --bm_file <FA3 GPU 报告> ... --task accuracy` | ✅ **182/182，100%，Pass** |
| ATK FA4 | 同上（v4 json + v4 GPU 报告） | ✅ **182/182，100%，Pass** |
| ATK 新增 18 条 | `atk pytorch <200 条 json> -s 182 -e 200 --devices 3`（npu vs cpu） | ✅ FA3/FA4 各 **18/18，100%，Pass** |
| L2 双标杆 | `cv_fused_double_benchmark` 2 / 1.2 / 1.2 | 实测 max_re **1.7311**、avg_re **0.0023**、rmse **0.8047**（全局最大） |
| 主算子回归 | FA3 `flash_attn_func` / FA4 `flash_attn_varlen_func` vs fp32 参考 | max diff 2.5e-4（fp16 正常量级） |

## 4. 已知边界与未覆盖项

| 项 | 说明 |
|---|---|
| A5 上板 | 本机无 950 设备，仅完成 `dav-3510` 编译/链接；需在 950 CI 镜像上补跑 pytest/ATK |
| 性能 | 需求明确"性能在 FD 体现，此处不要求性能"，未设性能门禁。参考数据（A2 单卡，Kahan 版本）：combine 单算子 `ns=8 B=1 S=1024 H=8 D=128 bf16` **0.111 ms**（34.3 MiB，有效 322 GB/s）；FD 链路 `B=8 H=32 Hkv=8 D=128 q=1 kv=8192 fp16` splits 1/2/4 = 0.79/0.79/0.78 ms（两次重复测量，差异在 ±4% 噪声内，即 combine 不作为瓶颈） |
| 内存 | 算子自身零额外 GM workspace；峰值占用来自调用者持有的 fp32 partial，未做实测阈值核对 |
| L2 边界 | fp16 输出量级 ~100 时 RMSE 比值受"跨舍入边界元素数"支配（见 `flash_attn_combine_atk_result.md` §3），实测最大 0.8047，余量约 33% |
| fp32 输出 | kernel 支持 fp32（`out_dtype=fp32`，pytest 覆盖），但本需求验收只取 bf16/fp16，ATK 用例已剔除 18 条 fp32 输出 |
| 端到端 | fwd `num_splits>1` 产出 dense fp32 partial 属二期，本期只交付 combine 接口 |

## 5. 复现

```bash
# A2/A3：pytest（9.1 容器或 wk 环境均可，需先 build_ext --inplace）
pytest tests/test_flash_attn_npu_v3_combine.py tests/test_flash_attn_npu_v4_combine.py

# A2/A3：ATK（L2 双标杆，NPU 实测 vs GPU 基线，182 条/后端）
source /data/w00896777/ops/0930/ascend-toolkit/set_env.sh
source /data/miniconda3/etc/profile.d/conda.sh && conda activate wk
export ASCEND_RT_VISIBLE_DEVICES=3 FLASH_ATTN_NPU_ROOT=/data/w00896777/code/flash-attention-npu
cd /data/w00896777/ATK/flash_attn_combine

atk node --backend npu --devices 0 \
  node --backend gpu --devices 0 --name gpu_base \
  --bm_file ./all_test_flash_attn_combine_reports_2026-10-20-07-10-25.xlsx \
  task -c ./all_test_flash_attn_combine_bf16fp16.json \
  -p ./execute_flash_attn_combine.py --task accuracy

atk node --backend npu --devices 0 \
  node --backend gpu --devices 0 --name gpu_base \
  --bm_file ./all_test_flash_attn_combine_v4_reports_2026-10-20-07-08-39.xlsx \
  task -c ./all_test_flash_attn_combine_v4_bf16fp16.json \
  -p ./execute_flash_attn_combine.py --task accuracy
```
