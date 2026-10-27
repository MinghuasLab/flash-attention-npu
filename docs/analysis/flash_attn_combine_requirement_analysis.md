# 【A2/A3/A5】【生态】Flash Attention 3/4 支持 `flash_attn_combine` 接口 —— 需求详细分析

> 分析基线：本仓 `main` @ `e378fb3`（`flash_attn_npu` v0.4.2）
> 对标基线：Dao-AILab/flash-attention `main`（`hopper/flash_attn_interface.py`、`flash_attn/cute/interface.py`、`hopper/flash_api.cpp`、`tests/cute/test_flash_attn_combine.py`）
> 结论口径：以"接口一致性 = 能直接跑通 NV 侧同名同参调用"为唯一判定标准。

---

## 0. 结论速览（TL;DR）

1. **现状比"缺接口"更微妙**：FA3(910) 已经有 `flash_attn_combine` 的 **Python 空壳**，但它调用的 C++ 算子 `flash_attn_npu_3.fwd_combine` **在所有 backend 中都不存在**（pybind 只注册了 `fwd`/`bwd`/`get_scheduler_metadata`）。也就是说这个接口现在是"**能 import、一调用就 AttributeError**"的死代码。详见 §3。
2. **缺口不止一层**：即使把 C++ 算子补上，还缺"**partial 产出侧**"——NV 的 combine 需要调用者手里有 fp32 的 `out_partial/lse_partial`。本仓现有 FD（FlashDecode）路径的 partial 是 **kernel 内部 workspace 里的 task-packed 布局**，且被 gate 在 `paged KV + TND + q_seqlen ≤ 16` 的 decode 场景，**既拿不到、也不是 NV 的 dense 布局**。
3. **A5(950) 还多一个精度对齐点**：910 的 partial O 已是 fp32（`ElementLse = float`），950 的 partial O 是 **16-bit**（`SIZE_OF_16BIT`），而 NV 强制要求 `out_partial` 为 fp32。
4. **工作量重心不是"合并公式"**：合并数学只是纯 Vector 的 LSE 加权归约，仓内已有两套可参考实现（910 `combine_scale.hpp`、950 `fd_combine.hpp`）。真正的成本在 **ABI 对齐 + dense 布局 + 950 移植 + 端到端打通 + 文档/测试**。
5. **最需要先拍板的三件事**（会影响方案形状）：① **"L2 双标杆"中 L2 的定义**（精度等级，还是"融合算子"级别）与判定细则；② "不超过 L2 阈值"的**具体数值与统计口径**；③ 端到端（fwd 产出 partial）**是否在一期范围内**。

---

## 1. 需求解读：把验收项翻译成本仓口径

| 需求项 | 解读 | 落到本仓的具体口径 |
|---|---|---|
| 硬件 A2 / A3 / A5 | A2=910B、A3=910C、A5=950 | A2/A3 在代码上**共用** `csrc/ascend910/**`（`Arch::AtlasA2`，dav-2201）；A5 走 `csrc/ascend950/**`（dav-3510/arch35）。本机实测为 8×910B3，即 A2 可本地验证，A3/A5 需异地上板 |
| FA3 / FA4 | 两套 API 世代 | `flash_attn_npu_3/**`（FA3）与 `flash_attn_npu_4/**`（FA4）；FA4 与 FA3 的 NV 签名**不同**（见 §2） |
| 数据类型 bf16/fp16 | 输出类型 | `out_dtype ∈ {fp16, bf16}`；**partial 为 fp32**（NV 强制），这本身就是需求的一部分 |
| 数据格式 BSND / TND | dense 与 varlen | BSND：`out_partial (num_splits,B,S,H,D)`；TND：`(num_splits,T,H,D)` + `cu_seqlens/seqused`（FA4）/ "batch 视为 1"（FA3） |
| 确定性计算：不涉及 | combine 是确定性归约 | LSE 加权求和天然确定，无需额外处理；但 NumPy/torch 参考实现要固定归约顺序以做 bit 级对比（可选） |
| 资料：涉及 | 文档交付 | `README.md` / `README.zh.md`（v3/v4 各加 `flash_attn_combine` 章节 + Features 表列）、Python docstring |
| aclnfuzz：不涉及 | — | 本仓为 torch 扩展直调（pybind + `torch.library.custom_op`），无 aclnn/GEIR 路径 |
| 精度目标：融合算子 **L2 双标杆** | 双参考比对 | 双标杆 = CPU 高精度 golden + NPU 小算子拼接参考；ATK 判据 `cv_fused_double_benchmark`：`max_re_ratio ≤ 5`、`avg_re_ratio ≤ 1.5`、`root_mean_squared_ratio ≤ 1.5`。本仓 `assert_fa_close`（`max_diff ≤ 2×pt_diff`，softcap 时 3×）可作开发期快速判据，但不替代 ATK 双标杆 |
| 性能目标：**性能在 FD 体现，此处不要求** | 不设独立性能门禁 | combine 只是 FD 的收尾步骤，收益来自"KV 切分并行"带来的端到端耗时下降；本接口只交付功能与精度，ATK 的 `perf` 字段不参与验收 |
| 内存目标：**不超过 L2 阈值** | 峰值占用受限 | combine 本身**零额外 GM workspace**（只读 partial + 写 out）；占用来自调用者持有的 fp32 partial（`num_splits×B×S×H×D×4B` + LSE `num_splits×B×S×H×4B`）与输出；阈值数值与统计口径待确认 |

---

## 2. 对齐基准：NV 的 `flash_attn_combine` 到底长什么样

### 2.1 FA3（Hopper）

- Python（`hopper/flash_attn_interface.py:938`）：
  ```python
  def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None):
      return flash_attn_3_gpu.fwd_combine(out_partial, lse_partial, out, out_dtype)
  ```
- op schema（`hopper/flash_api.cpp:1732`）：
  `fwd_combine(Tensor out_partial, Tensor lse_partial, Tensor(out!)? out=None, ScalarType? out_dtype=None) -> (Tensor(out!), Tensor)`
- 语义与约束（`hopper/flash_api.cpp::mha_combine`，即 `flash_attn_3::fwd_combine`）：
  - `lse = logsumexp_s(lse_partial)`；`w_s = exp(lse_s - lse)`（`inf/nan → 0`）；`out = Σ_s w_s · O_s`
  - **`out_partial` 必须 fp32**、`lse_partial` 必须 fp32；`out_partial.stride(-1)==1`；`lse_partial.stride(-2)==1`（**S 维连续**）
  - `num_splits ≤ 256`；`out_dtype ∈ {fp32, fp16, bf16}`；head_dim 会补齐到 4 的倍数再切回
  - 返回 `(out, softmax_lse)`；varlen 不额外传 `cu_seqlens`，直接把 batch 视为 1（走同一 kernel）

### 2.2 FA4（cute）

- Python（`flash_attn/cute/interface.py:4468`）：
  ```python
  def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None,
                         cu_seqlens=None, seqused=None, virtual_batch_idx=None, return_lse=True):
  ```
- 形状约定（`_flash_attn_fwd_combine` docstring）：
  - dense：`out_partial (num_splits,B,S,H,D)`、`lse_partial (num_splits,B,S,H)`、`out (B,S,H,D)`、`lse (B,S,H)`
  - varlen：`out_partial (num_splits,T,H,D)`、`lse_partial (num_splits,T,H)`、`out (T,H,D)`、`lse (T,H)`
  - 内部把 LSE transpose 成 **S 连续** 再给 kernel（与本仓 `softmax_lse` 的 `(...,H,S)` 布局一致）
- FA4 的**端到端用法**已在 NV 代码里固化：`interface.py:1063` `is_split_kv = num_splits > 1` → 分配 fp32 `out_partial/lse_partial` → fwd 写 partial → 立即调 `_flash_attn_fwd_combine` 出最终 `out`。`flash_attn_combine` 是给"**自己做 split 的外部框架**"（如上下文并行、跨 rank split-KV）用的公开入口。

### 2.3 上游测试矩阵 = 我们的验收参考（`tests/cute/test_flash_attn_combine.py`）

| 维度 | 取值 |
|---|---|
| dtype（out_dtype） | fp32 / fp16 / bf16 |
| head_dim | 64 / 96 / 128 / 192 / 256 / 512 |
| seqlen | 1 / 2 / 3 / 32 / 64 / 113 / 108 / 256 / 640 / 1024 |
| num_splits | 1 / 2 / 3 / 5 / 17 / 32 / 55 / 97 / 133 |
| varlen 模式 | `cu_seqlens` / `seqused` / `cu_seqlens + seqused` |
| 特殊构造 | 后半段 split 置 `-inf`（短路）、非连续输入、`virtual_batch_idx` 置换、`return_lse=False` |
| 判据 | `lse`：`atol=rtol=1e-5`；`out`：`max_diff ≤ 2×` (cast 后 PyTorch 结果与 fp32 参考之差)，或 `allclose(1e-5)` |

---

## 3. 本仓现状盘点（含证据）

### 3.1 接口层

| backend | Python API | C++ op 注册 | partial 返回 |
|---|---|---|---|
| FA3 910（`flash_attn_npu_3/flash_attn_npu_interface.py`） | ✅ 有（`:1165`）→ 调 `flash_attn_npu_3.fwd_combine` | ❌ 无（`csrc/ascend910/flash_attn_npu_3/flash_api.cpp:1070-1076` 只注册 `fwd`/`bwd`/`get_scheduler_metadata`） | `out_accum`/`lse_accum` 恒为 `None→空 tensor`（`flash_api.cpp:202` 声明后从未赋值，`:688` 原样返回） |
| FA3 950（`_950.py`） | ❌ 无 | ❌ 无（`csrc/ascend950/flash_attn_npu_3/flash_api.cpp:128-134`） | 同上（`_flash_attn_forward` 解包 4 值但恒空） |
| FA4 910（`flash_attn_npu_4/flash_attn_npu_interface.py`） | ❌ 无 | ❌ 无（`csrc/ascend910/flash_attn_npu_4/flash_api.cpp:1013-1019`） | **只返回 2 值**（`:154` `return out, softmax_lse`） |
| FA4 950（`_950.py`） | ❌ 无 | ❌ 无（`csrc/ascend950/flash_attn_npu_4/flash_api.cpp:128-134`） | 解包 4 值，`out_accum` 恒空（`:283`） |

另外 `flash_attn_npu_3/__init__.py:18-24`、`flash_attn_npu_4/__init__.py:18-23` **都没有导出** `flash_attn_combine`（NV 侧是从 interface 模块直接 import 的），即便 FA3-910 有定义，用户 `from flash_attn_npu_3 import flash_attn_combine` 也会 `ImportError`。

### 3.2 内核层：其实有两套"内部 combine"，但都不是公开接口

| 位置 | 形态 | 输入布局 | partial dtype | 触发 gate |
|---|---|---|---|---|
| `csrc/ascend910/flash_attn_npu_3/combine_scale.hpp`（`CombineScale`，v4 同名文件一致） | **融合 epilogue**，被 `mha_fwd_kvcache.cpp:551` 在同一个 FD kernel 末尾调用 | workspace 内 **task-packed**：按 `splitInfo[].lseTaskOffset/oTaskOffset/splitNum` 寻址（`mha_fwd_kvcache.cpp:1017-1018`） | LSE fp32、**O fp32**（`using ElementLse = float`，`mha_fwd_kvcache.cpp:1117`） | 910 v3 `flash_api.cpp:537`：`num_splits ≤ 1 || (paged_KV && is_varlen_q)`；再加 `flashDecodeFlag` 条件 |
| `csrc/ascend950/flash_attn_npu_3/fd_combine.hpp`（`FlashDecodeCombine950`，v4 同名） | **独立 kernel**，由 `fwd_dispatch_impl.hpp:87` 二次 launch | workspace 内 task-packed（`fdPartialLseOffset/fdPartialOOffset/fdPartialCapacity/fdLseSubStride/fdRowCapacity`） | LSE fp32、**O 16-bit**（`tiling.cpp:433-434` 用 `SIZE_OF_16BIT`；`fwd_kernel.cpp:135` `GlobalTensor<ElementO>`） | 950 v4 `mha_fwd.cpp:99`：`num_splits>1` 需 `paged_KV && is_varlen_q`；FD gate 再叠加 `q≤16`、`kv≥1024`、core 容量条件；950 v3 训练接口直接 `num_splits ∈ {0,1}` |

**关键差异**：NV 的公开接口吃的是**调用者可见的 dense fp32 张量**，本仓的两套实现吃的是**内部 workspace 的 task-packed 缓冲**。两者不是"同一个东西换个名字"，中间差了一层布局/所有权语义。

---

## 4. 差距（Gap）清单

| # | 差距 | 影响 | 证据 |
|---|---|---|---|
| G1 | 4 个 backend 全部缺 `fwd_combine` op（FA3-910 有壳无体，其余连壳都没有） | 接口不可用 | §3.1 |
| G2 | 缺 **partial 产出**通道：fwd 无法把 dense fp32 partial 交给调用者 | 接口"能用但没数据"，无法端到端 | `flash_api.cpp:202/688`（910 v3 返回空）、FA4-910 只返回 2 值 |
| G3 | partial 布局语义不一致：内部 task-packed vs NV dense | 需要新 kernel 或 repack，不能直接复用 `CombineScale` | §3.2 |
| G4 | 950 partial O 为 16-bit，NV 强制 fp32 | 精度风险 + 接口不兼容（fp32 输入会被拒） | `tiling.cpp:433` |
| G5 | 现有 split 路径 gate 极窄（paged KV + TND + `q_len ≤ 16`） | NV 的 combine 面向任意 BSND/TND 与任意 seqlen；端到端对齐需要放开 gate 或另建 split 路径 | `flash_api.cpp:537`、`mha_fwd.cpp:99` |
| G6 | 文档、测试全缺（README 无该接口章节，tests 无 combine 用例） | 需求"资料：涉及"未满足 | §1 |
| G7 | 内存口径未定义 | fp32 partial 体积随 `num_splits×S` 线性增长，长序列下轻易超阈值 | §5.4 |
| G8 | FA4 两 backend 返回签名不一致（910 2 值 / 950 4 值） | combine 若依赖 `out_accum`，必须先统一 | `:154` vs `:283` |

---

## 5. 方案设计建议

### 5.1 接口层（Python）——严格照抄 NV 签名

```python
# FA3（910 & 950），与 hopper/flash_attn_interface.py 完全一致
def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None): ...

# FA4（910 & 950），与 flash_attn/cute/interface.py 对齐（一期去掉 virtual_batch_idx）
def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None,
                       cu_seqlens=None, seqused=None, return_lse=True): ...
```

- 一并在 `flash_attn_npu_3/__init__.py`、`flash_attn_npu_4/__init__.py` 增加导出。
- 按仓内既有约定补 `torch.library.custom_op` + `register_fake`（tracing/torch.compile 可用），FA4-950 的 `virtual_batch_idx` 建议**一期不实现**（本仓无 persistent tile scheduler，语义无处落地），文档里显式标注为 not supported。
- v2 **不做**：NV 的 v2 也没有 `flash_attn_combine`，强行加反而破坏"接口一致性"。

### 5.2 算子层（C++）——新增独立 Vector kernel

- host：`mha_combine(out_partial, lse_partial, out*, out_dtype*) -> (out, lse)`，校验项与 NV 逐条对齐（fp32 partial / `stride(-1)==1` / `stride(-2)==1` / `num_splits≤256` / shape / `out_dtype`），报错信息语义对齐。
- device：**不复用** `CombineScale`（它绑死 `splitInfo`）；建议以 950 `FlashDecodeCombine950` 的 UB 预算与 reduce 流程为骨架，改造成 dense 输入（或直接用 packed 入口做一期，二期切 dense）。计算流程：
  ```
  for each (b, s-block, h-block):
      Load LSE[num_splits]  -> ReduceMax -> exp -> ReduceSum -> lse = log(sum)+max
      for s in 0..num_splits-1:
          Load O_s[s, b, s-block, h-block, :] (fp32)
          acc += exp(lse_s - lse) * O_s        // Muls + Add
      Cast -> Store out (out_dtype);  可选 Store lse
  ```
  全流程 **UB-only**，无额外 GM 中间量（满足"算子自身不超内存阈值"）。
- 编译组织：沿用 `autogen/generate_kernels.py` 的 **per-(dtype, layout) TU** 模式，新增 `combine_<dtype>_<layout>.cpp`（每 backend 4 个 TU），`setup.py:607-643` 的 `src_ascend910_v3/v4`、`src_ascend950_v3/v4` 需补 glob。
- 注意两代核差异：910 侧 `CombineScale` 依赖 `GetSubBlockNum()` 双 subblock 分工，950 侧 `MAX_FD_ACTIVE_CORE_NUM=32`/`MAX_HEAD_DIM=256` 等常量需按各自 UB 预算重算（`fd_combine.hpp:21-31` 的偏移表可直接作为起点）。

### 5.3 partial 产出侧（决定这件事能不能"端到端"）

- **一期（接口交付）**：只做 combine op + 上层 API，验收用**构造的随机 partial**（与上游测试一致）。这一期就能满足"按 FA3/4 `flash_attn_combine` 接口测试"的字面验收。
- **二期（生态可用）**：
  - 910 侧：把 fwd `num_splits>1` 的 partial 落成调用者可见的 **dense fp32 tensor**（`out_accum (num_splits,B,H,S,D)` / `lse_accum`，正好与 `flash_attn_npu_3/flash_attn_npu_interface.py:309-326` 的 fake 形状一致——fake 已经按 NV 语义写好了，属于"实现追上声明"）。
  - 950 侧：partial O 改 fp32 或提供 fp32 变体。
  - 布局：优先"split kernel 直接写 dense 布局"；若改造成本过高，退路是"写 packed + combine kernel 直读 packed（内部 repack）"，但那时对外仍必须是 dense。
  - gate：一期允许仍限 `paged KV + TND`（复用现有 FD 门控，风险低）；放开到任意 BSND/seqlen 放二期。

### 5.4 内存与性能口径

- 显存公式（调用者视角，与 NV 同）：
  `partial_bytes ≈ num_splits × B × S × H × D × 4   (O)   +   num_splits × B × S × H × 4   (LSE)`
  例：`B=1, S=8192, H=32, D=128, num_splits=16` → 约 2.1 GB(O) + 16 MB(LSE)。
  长序列下该项增长很快 ⇒ 建议在**接口文档中给出推荐 `num_splits` 上限与显存公式**，并要求需求方明确 A2/A3 各自 L2 阈值数值与判定方式（是"单算子峰值 ≤ L2"还是"workspace ≤ L2"）。
- 性能：combine 是 **memory-bound**（每元素读 4B + 1×scaler 计算），理论下限 ≈ `partial_bytes / HBM 带宽`；建议给 CI 增加一个 combine-only 的 profiler 采集（kernel 耗时 + 带宽利用率 + 多核负载），并明确对标基线。

---

## 6. 实施分解（WBS，粗估 ±50%）

| # | 工作项 | 主要文件 | 依赖 | 粗估 |
|---|---|---|---|---|
| W1 | 接口/ABI 设计冻结（schema、错误语义、docstring、`num_splits` 上限、`out_dtype` 集合） | 设计文档 | 需求方拍板 §8 的开放问题 | 2 人日 |
| W2 | 910 combine kernel + host + pybind（BSND/TND × fp16/bf16） | `csrc/ascend910/flash_attn_npu_3/{flash_api.cpp, combine_*.hpp/cpp, autogen/*}`、`setup.py` | W1 | 8~12 人日 |
| W3 | 910 v4 复用（host 适配 + 注册） | `csrc/ascend910/flash_attn_npu_4/**` | W2 | 3 人日 |
| W4 | 950 v3/v4（`fd_combine.hpp` → 独立 dense combine kernel + 注册） | `csrc/ascend950/flash_attn_npu_{3,4}/**` | W1（不依赖 W2） | 8~12 人日 |
| W5 | Python 接口（FA3-910 修真、FA3-950/FA4-910/FA4-950 新增、`__init__` 导出、fake/meta） | `flash_attn_npu_{3,4}/*.py` | W2~W4 | 4 人日 |
| W6 | 端到端 partial 产出（fwd 返回 dense fp32 `out_accum/lse_accum`；FA4-910 返回签名统一） | fwd host + split epilogue | W5 | 10~15 人日 |
| W7 | 测试：op-level 精度矩阵 + 端到端 + 异常/边界 + 性能 + 内存 + torch.compile | `tests/test_flash_attn_npu_v{3,4}_combine.py`、`tests/common/**` | W5 | 8~10 人日 |
| W8 | 文档：README/README.zh 章节 + Features 表 + docstring | `README*.md` | W5 | 2 人日 |

一期（W1~W5+W7+W8，即"接口 + 算子 + 测试 + 文档"）≈ **28~40 人日**；二期端到端（W6）另需 **10~15 人日**，且需回归 FD 既有性能。

---

## 7. 测试与验收方案

**T1 单算子精度（op-level，必做）**：随机 fp32 partial 构造，覆盖
dtype `{fp16,bf16}`（可加 fp32）× head_dim `{64,96,128,192,256,512}` × seqlen `{1,2,32,113,256,1024}` × num_splits `{1,2,3,5,17,32,256}` × 布局 `{BSND,TND}` × 特殊构造 `{-inf 短路、非连续输入、S 尾部非对齐、H=1 与 GQA}`。
判据：`lse` `atol=rtol=1e-5`；`out` `max_diff ≤ 2×pt_diff`（复用 `assert_fa_close`）。

**T2 端到端一致性（二期，必做）**：`fwd(num_splits=k)` → partial → `flash_attn_combine` 结果 vs `fwd(num_splits=1)` 全量结果，做 L2 双标杆比对；覆盖 causal / SWA / softcap / GQA / paged KV。

**T3 异常与边界**：partial 非 fp32、`num_splits>256`、shape 不匹配、stride 不满足、`out_dtype` 不支持、`out` 传入但 dtype 不符 → 与 NV 相同的报错语义；`cu_seqlens`/`seqused` 组合（含 `seqused<seqlen` 只保证有效段正确）。

**T4 性能**：combine-only profiler（耗时、带宽利用率、多核负载），对同一 shape 与 CANN 基线对比；另测"端到端 fwd(num_splits=k)+combine vs fwd(num_splits=1)"总时长拐点，给出推荐 `num_splits` 启发式。

**T5 内存**：记录 peak HBM 随 `num_splits` 的变化，核对 L2 阈值口径；文档中固化公式与建议上限。

**T6 torch.compile**：`_flash_attn_forward` 已支持 fake/meta（仓内 `test_torch_compile_fa{2,3,4}.py`），combine 需达到同等可 trace 水平。

**CI 路径**：910B 可在本机/CI（`ci/` docker 矩阵，CANN 9.1.0-beta.1 + torch 2.9）验证；950 需 CI 的 950 镜像（`ci/build_matrix.tsv:29`），迭代周期天然更长。

---

## 8. 风险与需拍板的开放问题

| # | 风险/问题 | 建议处理 |
|---|---|---|
| R1 | **"L2 阈值"与"L2 双标杆"中的 L2 定义未落地** | 需求方给出 L2 的定义（精度等级 or 融合算子级别）与阈值数值、统计口径（单算子峰值/workspace/整网）；否则精度与内存验收都无法判定 |
| R2 | **性能基线不明确**（CANN 无对外 combine 算子） | 明确基线为"FD 内部 combine 同 shape 耗时"或"torch 加权求和"，并约定对比方法/精度口径 |
| R3 | A5 无本地机器 | 依赖 950 CI 镜像；建议 combine 的 950 实现与 910 同步开发，避免串行等待 |
| R4 | 端到端（二期）放开 FD gate 可能影响**现有 FD 性能/精度** | 二期做 A/B 回归，FD 原路径保持 gate 不变，新增能力走独立分支逻辑 |
| R5 | FA4 的 `virtual_batch_idx` 依赖 persistent tile scheduler（本仓无） | 一期显式 not-supported 并在文档标注，二期再评估 |
| R6 | `num_splits` 上限：NV combine 允许 256，本仓 FD 受 `blockDim`/FD core 容量限制 | 接口层按 NV 定 256，实现层对超出能力范围给清晰报错；或文档给出实际上限 |
| R7 | 950 partial O 16-bit 与 NV fp32 冲突 | 二期确定：为公开接口增加 fp32 partial 变体（推荐），或明确降级并记录精度影响 |
| R8 | 范围蔓延（fp8 / ALiBi / dropout / attention_chunk 等） | 需求只覆盖 bf16/fp16 + BSND/TND，其余显式不做 |

---

## 9. 需求验收对照

| 验收项 | 达成方式 |
|---|---|
| 按 FA3/4 `flash_attn_combine` 接口测试 | T1（op-level，全矩阵）+ T3（异常/边界）；接口签名与 NV 逐字对齐（含参数名） |
| 生态易用性 / 模型无感迁移 | 同名同参 + 同名异常 + `__init__` 导出 + torch.compile 可 trace；端到端（T2）二期补齐"能真正拿到 partial" |
| 商用交付 | 文档（README 双语 + docstring）、CI 用例、内存/性能基线数据齐备 |

---

## 附：一句话回答"这个需求是什么"

把 NV 侧用于 **split-KV / 上下文并行** 的"**按 LSE 加权合并多份 partial attention 输出**"接口（FA3 4 参数版、FA4 带 `cu_seqlens/seqused/return_lse` 版）在 A2/A3/A5 三代的 FA3、FA4 上补齐并与之逐字对齐；本仓当前只有 FA3-910 的一个**不可用的壳**，其余全缺，且即使补上算子也还缺"调用者可见的 fp32 partial 产出"这一环才能端到端用起来。
