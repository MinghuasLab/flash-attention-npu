# `flash_attn_combine` 交付件（A5 / Ascend 950 验证用）

> 版本：2026-10-22
> 适用：Flash Attention 3/4 支持 `flash_attn_combine` 接口（A2/A3/A5，bf16/fp16，BSND/TND，融合算子 L2 双标杆）
> 读者：在 A5（Ascend 950）服务器上执行验证的 agent / 工程师
> 本文是**唯一入口**：环境、编译、安装、用例、命令、判据、已知坑都在这里。

---

## 0. 一页摘要（先读这一页）

**需求要做的事**：在 A2/A3/A5 上，为 FA3、FA4 两套 Python API 各提供一个与 NV（Dao-AILab flash-attention）**同名同参同返回**的
`flash_attn_combine(out_partial, lse_partial, ...)`：把多份 split-KV 的 partial 输出按 LSE 加权合并成最终结果。

**当前状态**：代码已完成，A2/A3 上板验证通过；A5（950）本机只完成了**编译/链接**，**没有上板跑过**——本次交付就是让 A5 上的 agent 把最后这一步跑完。

| A5 上要确认的事 | 期望结果 |
|---|---|
| 950 后端能编译出 v3/v4 两个 `.so` | 成功，无链接错误 |
| `import flash_attn_npu_3/4` 并能调用 `flash_attn_combine` | 成功，返回 `(out, lse)` |
| pytest 组合用例 | **127 passed** |
| ATK 200 条精度双标杆 | **198/200 通过**（另有 2 条已知边界，见 §7） |
| msSanitizer 内存检测 | **无访存错误** |
| 内存（workspace / 相对 GPU） | workspace **0**；NPU/GPU 分配内存比 ≤ **1.05** |

**20 分钟快速跑通**（详细步骤见 §6）：

```bash
cd flash-attention-npu
source /usr/local/Ascend/cann/set_env.sh          # 按 A5 实际路径调整
conda activate <你的 python3.10 + torch_npu 环境>
FLASH_ATTN_BUILD_NPU=950 FLASH_ATTN_BUILD_VERSION=v3 python setup.py build_ext --inplace
FLASH_ATTN_BUILD_NPU=950 FLASH_ATTN_BUILD_VERSION=v4 python setup.py build_ext --inplace
python -m pytest tests/test_flash_attn_npu_v3_combine.py tests/test_flash_attn_npu_v4_combine.py -q
```

### 0.1 从 PR 获取代码（推荐）

PR：**https://github.com/MinghuasLab/flash-attention-npu/pull/349**
分支：`AnnYE131:codex/flash-attn-combine`（单 commit，已 rebase 到 `main`）

```bash
# 1) clone 官方仓 + 子模块（catlass 必需）
git clone https://github.com/MinghuasLab/flash-attention-npu.git
cd flash-attention-npu
git submodule update --init --recursive

# 2) 取 PR（GitHub 会在官方仓暴露 refs/pull/<id>/head）
git fetch origin pull/349/head:pr-349
git checkout pr-349

# 或者挑到自己的分支：
# git checkout -b combine-a5 main
# git cherry-pick pr-349

# 3) 编译（A5 只需要 950 后端）
export FLASH_ATTN_BUILD_NPU=950
FLASH_ATTN_BUILD_VERSION=v3 python setup.py build_ext --inplace
FLASH_ATTN_BUILD_VERSION=v4 python setup.py build_ext --inplace

# 4) 安装（二选一）
python setup.py install                      # 方式 A：装进环境
export FLASH_ATTN_NPU_ROOT=$PWD              # 方式 B：in-place
export PYTHONPATH=$PWD:$PYTHONPATH

# 5) 一键自检（环境 + 冒烟 + pytest）
bash tests/st/atk_flash_attn_combine/run_a5_verify.sh --no-build
```

> 若网络访问 `github.com` 受限，可用 `ssh.github.com:443`：
> `git clone ssh://git@ssh.github.com:443/MinghuasLab/flash-attention-npu.git`

---

## 1. 需求原文

```
1. 需求背景
   对标业界 Attention 接口，在接口一致性上对齐 NV 使用方式，模型迁移对齐 NV 接口
2. 需求价值
   生态易用性扩展，模型无感迁移适配
3. 交付质量目标
   商用交付
4. 需求详情
   描述：Flash Attention 3/4 支持 flash_attn_combine 接口
   硬件型号：A2/A3/A5
   数据类型：bf16/fp16
   数据格式：BSND / TND
   确定性计算：不涉及
   资料：涉及
   aclnnfuzz：不涉及
   精度目标：融合算子 L2 双标杆
   性能目标：性能在 FD 体现，此处不要求性能
   内存目标：不超过 L2 阈值大小
5. 需求验收
   按 FA3/4 flash_attn_combine 接口测试
```

### 1.1 精度/内存判据（已确认口径）

| 项 | 判据 |
|---|---|
| 精度（L2 双标杆） | `MARE_npu / max(MARE_gpu, err) ≤ 2`、`MERE_npu / max(MERE_gpu, err) ≤ 1.2`、`RMSE_npu / max(RMSE_gpu, err) ≤ 1.2` |
| 内存（L2） | 满足其一：① I/O ≥ 500MB 时，较 GPU 内存 ≤ 5%；② 方案固有 workspace ≤ L2 cache（910B3=192MB，950 系列 96~128MB）且不随规格膨胀 |
| 性能 | 本需求不设门禁（在 FD 体现） |

---

## 2. 交付物清单

### 2.1 代码（本仓相对路径）

| 文件 | 作用 |
|---|---|
| `csrc/ascend910/flash_attn_npu_3/combine_kernel.cpp` | A2/A3 的 FA3 combine kernel（332 行） |
| `csrc/ascend910/flash_attn_npu_4/combine_kernel.cpp` | A2/A3 的 FA4 kernel（与 v3 逐字节相同） |
| `csrc/ascend950/flash_attn_npu_3/combine_kernel.cpp` | **A5 的 FA3 kernel（306 行，本次验证重点）** |
| `csrc/ascend950/flash_attn_npu_4/combine_kernel.cpp` | A5 的 FA4 kernel（与 v3 逐字节相同） |
| `csrc/ascend*/flash_attn_npu_*/combine_dispatch.hpp` | host↔kernel ABI（`CombineLaunchArgs` + 3 个 dtype launcher） |
| `csrc/ascend*/flash_attn_npu_*/flash_api.cpp` | 新增 `mha_combine`（校验/tiling/launch）+ `m.def("fwd_combine", ...)` |
| `flash_attn_npu_3/flash_attn_npu_interface.py` | FA3 910 接口 `flash_attn_combine` |
| `flash_attn_npu_3/flash_attn_npu_interface_950.py` | **FA3 950 接口** |
| `flash_attn_npu_4/flash_attn_npu_interface.py` | FA4 910 接口（多 `cu_seqlens/seqused/return_lse`） |
| `flash_attn_npu_4/flash_attn_npu_interface_950.py` | **FA4 950 接口** |
| `flash_attn_npu_{3,4}/__init__.py` | 按 `torch_npu.npu.get_device_name()` 自动选 910/950 实现 |
| `setup.py` | 4 个 source list 补 kernel；`FLASH_ATTN_BUILD_NPU=950` 走 `dav-3510` |

### 2.2 测试

| 文件 | 内容 |
|---|---|
| `tests/test_flash_attn_npu_v3_combine.py` | FA3 组合用例 94 条（BSND/TND、dtype、num_splits、head_dim、`-inf` 短路、原地写、错误契约） |
| `tests/test_flash_attn_npu_v4_combine.py` | FA4 组合用例 33 条（`seqused`、`return_lse=False`、错误契约等） |
| `tests/st/atk_flash_attn_combine/execute_flash_attn_combine.py` | ATK 执行器：CPU golden + NPU 调用（FA3/FA4 两个注册名） |
| `tests/st/atk_flash_attn_combine/flash_attn_combine_constraint.py` | ATK 跨输入约束（lse shape、dtype、`-inf` 场景） |
| `tests/st/atk_flash_attn_combine/test_flash_attn_combine{,_v4}.yaml` | ATK 用例生成规格（L2 标准 2/1.2/1.2，`perf: not_key`） |
| `tests/st/atk_flash_attn_combine/{extend_cases_to_200,renumber_case_ids,rekey_report_ids}.py` | 用例扩到 200 条、id 重排、旧报告 id 重映射 |
| `tests/st/atk_flash_attn_combine/bench_combine.py` | combine 单算子 + FD 链路性能参考 |
| `tests/st/atk_flash_attn_combine/run_atk.sh` | ATK 一键脚本（npu vs cpu 冒烟） |

### 2.3 ATK 用例与基线（交付包 `atk/` 目录）

| 文件 | 说明 |
|---|---|
| `all_test_flash_attn_combine_bf16fp16.json` | FA3 用例 **200 条**（id 0..199，bf16 108 / fp16 92） |
| `all_test_flash_attn_combine_v4_bf16fp16.json` | FA4 用例 **200 条**（同 id 规则） |
| `accuracy-all_test_flash_attn_combine_bf16fp16_reports_*.xlsx` | FA3 GPU 精度基线（200 行，与 json 逐条对齐） |
| `accuracy-all_test_flash_attn_combine_v4_bf16fp16_reports_*.xlsx` | FA4 GPU 精度基线 |
| `memory-all_test_flash_attn_combine{,_v4}_bf16fp16_reports_*.xlsx` | GPU 内存基线（FA3 199 行，缺 1 条；FA4 200 行） |

### 2.4 文档

`docs/analysis/` 下 7 份：需求分析、接口规格、实现分析、ATK 结果、内存验收口径、交付清单、验证指南，
以及本文件（`flash_attn_combine_A5_handoff.md`，A5 验证入口）。

---

## 3. 接口规格（A5 上要验证的契约）

### 3.1 FA3（`flash_attn_npu_3`）

```python
flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None) -> (out, softmax_lse)
```

| 参数 | 契约 |
|---|---|
| `out_partial` | **fp32**；dense `(num_splits, B, S, H, D)` 或 varlen `(num_splits, T, H, D)`；`stride(-1)==1` |
| `lse_partial` | **fp32**；shape = `out_partial.shape[:-1]`；**`stride(-2)==1`（seqlen 维连续）** |
| `out` | 可选，预分配输出（原地写） |
| `out_dtype` | `fp16` / `bf16`（需求范围）；`fp32` 也支持但不在本需求验收内；默认 = `out_partial.dtype` |
| 返回 | `out`（`(B,S,H,D)` 或 `(T,H,D)`）+ `softmax_lse`（fp32，`(B,S,H)` 或 `(T,H)`，seqlen 连续） |

### 3.2 FA4（`flash_attn_npu_4`）

```python
flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None,
                   cu_seqlens=None, seqused=None, return_lse=True, ...)
```

- 语义同 FA3；`cu_seqlens/seqused` 走 varlen 分段；`return_lse=False` 时不物化 lse；
- `virtual_batch_idx` 显式 `NotImplementedError`（与 NV 现状一致）。

### 3.3 约束与报错（pytest 覆盖）

- `num_splits ∈ [1, 256]`；`out_partial/lse_partial` 必须 fp32；`out_dtype ∈ {fp16,bf16,fp32}`；
- `lse_partial` 不满足 seqlen 连续 → `RuntimeError: LSE tensor must be contiguous in the seqlen dimension`；
- `out` 预分配时 dtype/shape 必须匹配。

> **A5 agent 注意**：`S=1` 时 `transpose+contiguous` 是空操作（size-1 维不参与连续性判断），
> 需要显式分配缓冲再转置，否则会误报不连续。ATK 执行器里已有正确写法。

---

## 4. 实现说明（为什么这么写）

### 4.1 数据流

```
lse   = logsumexp_s(lse_partial)
e_s   = exp(lse_s - max)
w_s   = e_s / Σ_s e_s
out   = Σ_s w_s * O_s        # fp32 Kahan 补偿累加
out   = Cast(out, out_dtype) # fp16 / bf16
lse_out = max + ln(Σ e_s)    # 仅用于返回
```

### 4.2 关键设计

| 设计点 | 说明 |
|---|---|
| 纯 Vector kernel | 不申请 GM workspace；每个 AIV 负责整块 (batch/head × row-block) tile，无跨核同步 |
| UB 预算 | 单 tile ≤150KB（UB 192KB 留余量）；按 `num_splits/head_dim` 自适应 `tile_rows ∈ {32,24,16,8}` |
| 权重归一化 | `e_s = exp(lse_s - max)`，分子分母同源 → exp 舍入抵消；比 `exp(lse_s - lse)` 少一次 `ln→exp` 往返 |
| 累加 | **Kahan 补偿求和**（910 复用 4 tile；950 多一块 `yTile`），对 `num_splits` 到 256 都稳定在 ~1 ULP |
| 全 `-inf` 行 | `den=0` clamp 到 1 → `out=0`，不出 NaN |
| 非 16 对齐 `head_dim` | 搬入必须显式给 `dstStride`（见 §8.1），否则整块按行漂移 |
| 910/950 差异 | 910 用 `BroadCast` 广播权重；950 用 per-row `Muls + GetValue`（arch35 API 约束） |

### 4.3 已修复的两个真问题（回归时要留着）

1. **数值路径**：`exp(lse_s-lse)` → `e_s/Σe_s` + Kahan，解决 L2 双标杆下 fp16 输出 RMSE 比超 1.2（原 180/182 → 182/182）。
2. **非对齐 head_dim**：`DataCopyPad(dstStride=0)` 按 32B 打包，行距 = `AlignUp(D,8)`，而 UB 其余部分按 `dA=AlignUp(D,16)` 访问；
   `D=113/200` 时整块错位。修复：`oDstStride = (dA*4 - AlignUp(D*4,32))/32`（对 16 对齐 D 恒为 0，不改变既有数值）。

---

## 5. A2/A3 已验证结果（A5 的对照基准）

| 项 | 结果 |
|---|---|
| A2/A3 编译 | 910 v3/v4 均编译成功并上板 |
| pytest | **127 passed**（v3 94 + v4 33） |
| ATK FA3 精度（200 条） | **198/200**，失败 2 条为已知边界（§7） |
| ATK FA4 精度（200 条） | 同上（910 v4 与 v3 kernel 相同） |
| 精度比值（除边界） | MARE ≤ 1.73、MERE ≤ 0.0023、RMSE ≤ 0.80（阈值 2 / 1.2 / 1.2） |
| 内存 | workspace **0**；NPU/GPU 分配内存比 mean 0.51、max 0.70（阈值 1.05） |
| msSanitizer | 运行时注入下 `FaCombine` kernel **No error detected**（仅 FFTS_BASE_ADDR 寄存器告警） |
| 性能参考 | combine 单算子 `ns=8 B=1 S=1024 H=8 D=128 bf16` ≈ **0.11ms / 322 GB/s**；FD 链路 splits 1/2/4 无额外劣化 |

---

## 6. A5 验证手册（按顺序执行）

### 6.1 环境检查

```bash
source /usr/local/Ascend/cann/set_env.sh        # 按实际 CANN 路径
conda activate <a5_env>                          # 需要 python3.10 + torch + torch_npu（版本匹配）
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__); print(torch_npu.npu.get_device_name())"
npu-smi info | head -12
```

判据：`get_device_name()` 含 `Ascend950`（`flash_attn_npu_3/__init__.py` 据此选择 950 接口）。

### 6.2 编译（v3 + v4，950 后端）

```bash
cd flash-attention-npu
FLASH_ATTN_BUILD_NPU=950 FLASH_ATTN_BUILD_VERSION=v3 python setup.py build_ext --inplace
FLASH_ATTN_BUILD_NPU=950 FLASH_ATTN_BUILD_VERSION=v4 python setup.py build_ext --inplace
```

产物（仓库根目录）：

```
flash_attn_npu_3_950.cpython-310-aarch64-linux-gnu.so
flash_attn_npu_4_950.cpython-310-aarch64-linux-gnu.so
```

编译参数应为 `bisheng --npu-arch=dav-3510`。若失败，先看 §8.2。

### 6.3 安装 / 让 ATK 找得到

```bash
# 方式 A：安装到环境（推荐 A5 上做一次）
python setup.py install
# 方式 B：in-place + 环境变量（改代码后免重装）
export FLASH_ATTN_NPU_ROOT=$PWD
export PYTHONPATH=$PWD:$PYTHONPATH
```

ATK 的 NPU 执行器通过 `FLASH_ATTN_NPU_ROOT` 把仓库根加入 `sys.path`。

### 6.4 冒烟（3 分钟）

```bash
python - <<'PY'
import torch, torch_npu
import flash_attn_npu_3, flash_attn_npu_4
NS, B, S, H, D = 4, 1, 32, 2, 128
op  = torch.randn(NS, B, S, H, D)
lse = torch.rand(NS, B, S, H) * -5
lse_c = lse.transpose(2, 3).contiguous().transpose(2, 3)
for mod in (flash_attn_npu_3, flash_attn_npu_4):
    out, l = mod.flash_attn_combine(op.npu(), lse_c.npu(), out_dtype=torch.float16)
    torch.npu.synchronize()
    print(mod.__name__, out.shape, out.dtype, l.shape, l.dtype)
PY
```

期望：两个模块都打印 `(1, 32, 2, 128) float16 (1, 32, 2) float32`。

### 6.5 pytest（约 1~10 分钟）

```bash
python -m pytest tests/test_flash_attn_npu_v3_combine.py \
                 tests/test_flash_attn_npu_v4_combine.py -q
```

判据：**127 passed**。失败先看 §8.4。

### 6.6 ATK 单标杆（无 GPU 也能跑通 200 条）

```bash
export FLASH_ATTN_NPU_ROOT=$PWD
cd atk
PYTHONPATH=$FLASH_ATTN_NPU_ROOT atk pytorch ./all_test_flash_attn_combine_bf16fp16.json \
  -p ./execute_flash_attn_combine.py --task accuracy --devices 0
PYTHONPATH=$FLASH_ATTN_NPU_ROOT atk pytorch ./all_test_flash_attn_combine_v4_bf16fp16.json \
  -p ./execute_flash_attn_combine.py --task accuracy --devices 0
```

期望：200 条 SUCCESS；单标杆下个别 fp16 大值域用例可能因 1 ULP 舍入差异超阈值（正常）。

### 6.7 ATK 双标杆（L2 精度验收，推荐）

GPU 基线是 NVIDIA GPU 产出的报告，与 NPU 硬件无关，A5 可直接复用。

```bash
atk node --backend npu --devices 0 \
  node --backend gpu --devices 0 --name gpu_base \
  --bm_file ./accuracy-all_test_flash_attn_combine_bf16fp16_reports_2026-10-21-03-30-08.xlsx \
  task -c ./all_test_flash_attn_combine_bf16fp16.json \
  -p ./execute_flash_attn_combine.py --task accuracy
```

FA4：`--bm_file` 换 `accuracy-all_test_flash_attn_combine_v4_bf16fp16_reports_2026-10-21-03-27-53.xlsx`，`-c` 换 v4 json。

判据：**≥198/200 通过**；另外 2 条已知边界见 §7。

### 6.8 内存（L2 内存判据）

```bash
atk node --backend npu --devices 0 \
  node --backend gpu --devices 0 --name gpu_base \
  --bm_file ./memory-all_test_flash_attn_combine_bf16fp16_reports_2026-10-21-03-26-21.xlsx \
  task -c ./all_test_flash_attn_combine_bf16fp16.json \
  -p ./execute_flash_attn_combine.py --task memory_device
```

判据：`Workspace大小(MB)` 在 PTA 路径下为空属正常（源码层 workspace = 0）；NPU/GPU 分配内存比 ≤ **1.05**；
FA3 GPU 内存基线只有 199 行，缺的那条显示"无基准"，属基线缺失而非算子问题。

### 6.9 msSanitizer 内存检测（A5 只能运行时注入）

```bash
mssanitizer --tool=memcheck -- python -m pytest -q tests/test_flash_attn_npu_v3_combine.py
```

判据：`FaCombine` kernel 输出 `No error detected`；`FFTS_BASE_ADDR was not reset` 是寄存器告警，不是访存错误。
A5（dav-3510）没有 `libsanitizer_stub`，静态插桩不可用，必须运行时注入。

---

## 7. 已知边界（A5 上大概率同样出现，不是缺陷）

### 7.1 两条"GPU 逐位精确"导致的精度边界

| id | shape | dtype | 现象 |
|---|---|---|---|
| 56 | `[32,16,2,256]` TND | bf16 | GPU 输出与 CPU golden **逐位相同**（MARE/MERE/RMSE=0）；NPU 仅个别元素差 1 ULP，RMSE 比 1.414 > 1.2 |
| 90 | `[8,8,1,32,128]` BSND | fp16 | 同上，RMSE 比 1.414 |

原因：双标杆公式 `ratio = NPU / max(GPU, ε)`，GPU=0 时分母塌缩为 dtype epsilon（bf16 2^-8、fp16 2^-11），
NPU 的 1 ULP 差异被放大。NPU 侧平均相对误差仅 6.3e-7（bf16）/1.2e-7（fp16），无系统性偏差。

判定：**不是算子缺陷**；A5 若同样只在这两条出现 1.4 量级比值，按"指标边界"记录。

### 7.2 ATK 工具侧已知问题

| 问题 | 说明 / 规避 |
|---|---|
| `node.yaml` 用 `pyaclnn` 报 `请在用例yaml中补充aclnn_name字段` | 本算子无 aclnn 封装，必须 `--backend npu` |
| `npu + cpu` 拓扑出不了内存数据 | CPU 节点不支持 PTA perf，内存对比必须 `npu + gpu`（真实 GPU 或 `--bm_file`） |
| `get_ascend_chip_model()` 返回 `Unknown` → L2 判据失效 | `npu-smi info -t board` 不带 `-c` 时无 `Chip Name`；只影响 ≥500MB 大用例的 L2 分支 |
| 用例 id 与基线错位 | ATK 按 `编号` 匹配；FA3/FA4 用例顺序不同，**不能交叉配基线** |

---

## 8. 常见失败与排查

### 8.1 非对齐 head_dim 输出整块错位

现象：`D=113/200` 时每个 lane 都错（MARE 上千）。
根因：`DataCopyPad(dstStride=0)` 行距 = `AlignUp(D,8)`，kernel 其余按 `dA=AlignUp(D,16)`。
已修：`oDstStride = (dA*4 - AlignUp(D*4,32))/32`。若被改掉，症状会重现。

### 8.2 编译失败

| 报错 | 处理 |
|---|---|
| `unable to find library -lsanitizer_stub_dav-c310` | 950 不支持静态插桩，去掉 `FLASH_ATTN_ENABLE_MSSANITIZER`，改用 §6.9 |
| `dav-3510` 相关头文件缺失 | 确认 CANN ≥ 9.x 且已 source `set_env.sh` |
| `cp310` / `cp312` 不匹配 | 在目标环境 `build_ext --inplace` 重编，不要跨版本拷 `.so` |

### 8.3 import 失败

| 报错 | 处理 |
|---|---|
| `ModuleNotFoundError: flash_attn_npu_3` | 未安装也未设 `FLASH_ATTN_NPU_ROOT` / `PYTHONPATH` |
| `Unsupported Ascend device` | `get_device_name()` 不含 Ascend950/910，检查驱动与设备可见性 |
| 加载到旧 `.so` | 确认加载的是 `*950*cpython-<当前版本>*.so` |

### 8.4 A5 测试失败分流

| 现象 | 先查 |
|---|---|
| pytest 编译期失败 | §8.2 |
| pytest 数值失败 | 是否只涉及 fp16 大值域/边界 shape；用 `extra-info/probe_ulp_bias.py` 对比 ULP 分布 |
| ATK 200 条大面积失败 | `-c` 与 `--bm_file` 是否同一后端、id 是否逐条对齐 |
| ATK 单条"无基准" | GPU 基线缺该 id，补跑 GPU 基线即可 |
| 内存比异常 | 确认 `--task memory_device`、两侧都是 PTA 路径、用同一 case 集 |

---

## 9. 验收清单（A5 agent 照此出报告）

| # | 检查项 | 命令/证据 | 通过标准 |
|---|---|---|---|
| 1 | 设备识别 | `torch_npu.npu.get_device_name()` | 含 `Ascend950` |
| 2 | 950 编译 | §6.2 | v3/v4 两个 `.so` 生成 |
| 3 | import + 冒烟 | §6.4 | 两模块均可调用、shape/dtype 正确 |
| 4 | pytest | §6.5 | 127 passed |
| 5 | ATK 精度双标杆 | §6.7 | ≥198/200，余量为 §7.1 两条 |
| 6 | ATK 内存 | §6.8 | workspace 0；NPU/GPU ≤1.05（缺基准条目单列） |
| 7 | msSanitizer | §6.9 | No error detected |
| 8 | 性能（可选） | `bench_combine.py --mode both` | 仅记录，不设门禁 |

报告建议包含：环境版本、编译日志尾部、pytest 汇总、ATK 汇总表、失败用例清单（id/shape/三项比值）、msan 结论、内存比值统计。

---

## 10. 交付包结构与传输

```
flash_attn_combine_a5_delivery_<date>/
├── README_FIRST.md                 # 本文件
├── flash-attention-npu/            # 源码（含 910/950 kernel、接口、pytest、脚本）
├── atk/                            # 用例 json + 执行器 + yaml + GPU 基线 xlsx
├── docs/analysis/                  # 7 份分析文档
├── run_a5_verify.sh                # 一键环境检查 + 编译 + pytest
└── MANIFEST.txt                    # 文件清单与 sha256
```

传输后第一步：

```bash
tar xzf flash_attn_combine_a5_delivery_<date>.tar.gz
cd flash_attn_combine_a5_delivery_<date>
bash run_a5_verify.sh
```
