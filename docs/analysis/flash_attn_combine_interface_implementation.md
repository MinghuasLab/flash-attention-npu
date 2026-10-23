# `flash_attn_combine` 接口实现分析

> 前提：FA3 / FA4 的基础算子（`fwd` / `bwd` / `get_scheduler_metadata`）已在 A2/A3(910) 与 A5(950) 上实现完毕，本次只需补齐 `flash_attn_combine`。
> 基线：`main` @ `e378fb3`（分析时仓库无未提交改动，`csrc/catlass` 子模块未初始化）。
> 关联文档：需求层面的整体分析见 `docs/analysis/flash_attn_combine_requirement_analysis.md`。

---

## 0. 一句话结论

这条路是可行的、而且**不需要动 fwd/bwd 主流程**：新增一个独立的 **Vector 算子**（2 套 arch 实现）＋ 4 处 pybind 注册 ＋ 4 处 Python 包装 ＋ 导出/文档/测试即可。

但有两个必须提前明确的边界：

1. **接口的输入是 partial**。既然不改 fwd，`out_partial/lse_partial` 只能由调用者提供（NV 官方测试同样用随机 partial 验证）。所以本次验收 = **op-level 接口/精度/性能**，端到端（fwd 产出 partial → combine）不在范围内。
2. **NV 的默认语义包含 fp32 输出**：`out_dtype=None` 时 NV 返回 **fp32**（取 `out_partial.dtype`）。若我们只支持 fp16/bf16 输出，"同名同参调用"在默认参数下结果类型就与 NV 不一致 —— 这是"接口一致性"的硬点，建议顺手支持 fp32（成本≈跳过 cast）。

---

## 1. 要冻结的接口契约

### 1.1 签名（逐字对齐 NV）

```python
# FA3（910 与 950），与 hopper/flash_attn_interface.py 一致
def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None): ...

# FA4（910 与 950），与 flash_attn/cute/interface.py 对齐（去掉本次不实现的参数）
def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None,
                       cu_seqlens=None, seqused=None, return_lse=True): ...
```

FA4 的 `virtual_batch_idx`（依赖 persistent tile scheduler，本仓无）、`num_splits_dynamic`、`seqlen_k_per_split` 建议显式标注 not supported 并 `NotImplementedError`，与仓内既有做法一致（如 FA4 docstring 里的 `pack_gqa`/`score_mod`）。

### 1.2 形状与布局

| | FA3 | FA4 |
|---|---|---|
| dense `out_partial` | `(num_splits, B, S, H, D)` | `(num_splits, B, S, H, D)` |
| dense `lse_partial` | `(num_splits, B, S, H)` | `(num_splits, B, S, H)` |
| varlen `out_partial` | `(num_splits, T, H, D)` | `(num_splits, T, H, D)` |
| varlen `lse_partial` | `(num_splits, T, H)` | `(num_splits, T, H)` |
| `out` / `lse` 输出 | `(B,S,H,D)` / `(B,H,S)`；varlen `(T,H,D)` / `(H,T)` | 同上（`return_lse=False` 时 lse 为 `None`） |

**实现建议**：内部**按 stride 解释，不假设维度顺序**，只强制 NV 的两条硬约束
`out_partial.stride(-1) == 1`、`lse_partial.stride(-2) == 1`（seqlen 维连续）。
理由：NV 自己的 FA3 fwd 产出的 partial 实际是 `(num_splits, B, H, S, D)`（见 `flash_attn_npu_3/flash_attn_npu_interface.py:309-326` 的 fake），而公开 doc 写的是 `(num_splits, B, S, H, D)`；NV 的 host 也用 stride 寻址。我们用 stride 就能同时吃下两种，避免"文档对齐了、实际调用报错"。

### 1.3 dtype

| 张量 | 要求（对齐 NV） |
|---|---|
| `out_partial` | **必须 fp32**（否则报错，与 NV 一致） |
| `lse_partial` | **必须 fp32** |
| `out_dtype` | `None` → 取 `out_partial.dtype`（=fp32）；显式传入可为 fp16 / bf16 / fp32 |

### 1.4 校验与报错（逐条对齐 NV 语义）

- `out_partial.dim() ∈ {4,5}`（4D=varlen，5D=dense，FA4）
- shape 自洽：`lse_partial` 与 `out_partial` 去掉 D 后一致；`num_splits ≤ 256`
- `out`（若传入）dtype/shape 必须匹配，且 `stride(-1)==1`
- `out_dtype` 非法 → 报错
- `cu_seqlens`/`seqused` 必须 int32；`seqused ≤ cu` 的段长；与 `out_partial` 同 device

### 1.5 边界行为

| 场景 | 期望行为 |
|---|---|
| 某些 split 的 LSE = `-inf`（该 split 未覆盖这段 KV） | 权重置 0，不产生 NaN（NV 显式做 `exp(lse_s - lse) → 0`） |
| 全部 split 的 LSE = `-inf` | 输出 0、`lse = -inf`（与 attention 空行语义一致） |
| `num_splits == 1` | 退化为"直接 cast"，必须仍走上限校验之外的快路径或正常路径 |
| `head_size` 非 4/8 对齐（如 96、113） | 尾块补齐后写回原始 D（NV 把 `head_size` 补齐到 4 的倍数再切回） |
| `S == 0` / `H == 0` | 不 launch，直接返回 |

---

## 2. 实现方案

### 2.1 总体结构

```
Python 包装（4 处） → pybind op <v3|v4>(fwd_combine) → host: mha_combine()
                                                        ├─ 校验 / dtype / shape / stride
                                                        ├─ tiling（纯 host 计算，无需 AICPU）
                                                        └─ launch 独立 combine kernel
kernel：UB-only，读 partial → 加权归约 → 写 out(+lse)
```

- **不引入 GM workspace**：算子自身只读 `out_partial/lse_partial`、只写 `out/lse`。这对"内存不超阈值"是天然友好的（真正的内存占用在调用者的 partial 上）。
- **不需要 AICPU metadata**：combine 没有 mask/tiling 依赖，tiling 在 host 一次算完（区别于 fwd 的 `GetSchedulerMetadataImpl`）。
- **不需要 Cube**：纯 Vector 算子。

### 2.2 切分与多核调度

- tile 粒度：`rows`（同一 (b) 内连续 query 行）× `Dc`（head_dim 切块，`Dc ≤ 256` 且通常取 128/256）
- 任务空间：`B × ceil(S/rows) × ceil(H/Hc)`（Hc = 每次处理的 head 数，建议先把 H 折进 rows 或取 1，见 §2.3）
- 分核：按任务 round-robin 到 AIV 核；每核尽量处理"同一 b、同一行块"的连续任务，命中 L2
- 核内：沿用两代器件一致的"双 subblock 半分"写法（910 `CombineScale` 把行/头拆两半给两个 vector subblock；950 `FlashDecodeCombine950` 用 `GetBlockIdx()/GetSubBlockNum()` 定位任务），保证两个 AIV 都被用满

### 2.3 计算流程与 UB 预算

```
for 每个 (b, row-block, head-block) tile:
    # LSE 段
    Load  LSE[num_splits, rows]         -> ll_ub        (fp32)
    ReduceMax  over splits              -> lse_max
    Sub / Exp / ReduceSum               -> lse_sum
    lse = ln(lse_sum) + lse_max
    w   = exp(lse_s - lse)   (inf/nan -> 0)
    # O 段（按 split 串行累加，可 ping-pong 预取）
    acc = 0
    for j in 0..num_splits-1:
        Load O[j, b, row-block, head-block, :]   (fp32, rows×Dc)
        acc += w[j][rows] * O_tile
    Cast(acc -> out_dtype)
    Store out
    if return_lse: Store lse
```

UB 预算（字节，fp32 缓冲）：

```
UB ≈ num_splits × rows × 8        # LSE + 权重，各按 8 元素对齐
   + rows × Dc × 4 × 3            # O tile ×2(ping-pong) + acc
   + rows × Dc × 2                # 输出 cast 暂存
```

| rows | Dc | num_splits | UB 估算 | 结论 |
|---|---|---|---|---|
| 16 | 128 | 32 | ≈ 28 KB | 轻松 |
| 16 | 256 | 64 | ≈ 56 KB | 轻松 |
| 64 | 128 | 16 | ≈ 104 KB | 可行但偏紧 |
| 64 | 256 | 32 | ≈ 200 KB | 超预算 → 必须拆 tile |

⇒ host 侧用"给定 UB 预算反推 `rows/Dc`"的方式定 tiling（预算建议按器件留出冗余，如 ≤ 160 KB），并对 `num_splits=256` 的极端情况自动缩小 `rows`。参考常数：950 `fd_combine.hpp` 的 `MAX_ROWS_PER_SUBBLOCK=16`、`MAX_HEAD_DIM=256`、`MAX_SPLIT_NUM=32` 就是同一套思路的现成取值。

### 2.4 BSND / TND 索引映射

- BSND：`off = j·(B·S·H·D) + b·(S·H·D) + s·(H·D) + h·D + d`（LSE 同理，末两维按 stride 取）
- TND：把 `(b,s)` 折叠成全局 token `t`；`cu_seqlens` 给出每段起点与段长，`seqused` 给出"有效段"（超出部分**不保证**输出正确，与 NV 一致）
- FA3 的 TND 走"batch 视为 1"的等价路径（无 `cu_seqlens` 参数），即把 T 维当行维即可

### 2.5 尾块与对齐

- `D` 尾块：`DataCopyPad` 按元素粒度搬，UB 内 pad 到 8 元素（32B）再做 Vector 运算
- `rows` 尾块：`lse_block` 可能不足一个 tile，需按有效行数截断（`CombineScale` 里已有同款处理）
- `num_splits` 非 8 对齐：LSE 归约的 shape 需 pad（`splitNumAlign = ceil(ns,8)*8`），pad 值对 `ReduceMax` 必须是 `-inf`、对权重必须是 0（`CombineScale` 用 `Duplicate(lowest())` 起到同样作用）
- **UB 行距 vs `DataCopyPad` 默认打包**：UB 内行距是 `dA = AlignUp(head_dim, 16)`（输出 staging 需 32B 对齐），
  而 `DataCopyPad(dstStride=0)` 按 32B 对齐的 blockLen 连续打包，行距其实是 `AlignUp(head_dim, 8)`。
  两者只在 `head_dim % 16 == 0` 时相等，否则整块按行漂移。因此 O 段搬入必须显式给
  `dstStride = (dA*4 - AlignUp(head_dim*4,32))/32`（32B 单位）。已在 910/950 四份 kernel 统一修复，
  对齐场景该值为 0、数值逐位不变。

### 2.6 精度处理（对齐"L2 双标杆"）

- 全程 fp32；只在最后 `Cast` 到 `out_dtype`
- LSE 用 max-shift 稳定式：`lse = ln(Σ exp(lse_s - max)) + max`（仅用于**返回的 lse**）
- **权重不再走 `exp(lse_s - lse)`**，改成 `w_s = e_s / Σ e_s`，`e_s = exp(lse_s - max)`：
  - exp 的舍入误差在分子分母同源、直接抵消；省掉 `ln → exp` 往返带来的 1 ULP LSE 误差；
    实测 pre-cast 相对误差从 ~7.7e-7 降到 ~5.8e-7。
  - 代价：末尾多做一次 `(rows × D)` 的 `Div`，少做一次 `(ns × rows)` 的 `Exp`。
- **split 累加用 Kahan 补偿求和**：`y = t - c; s = acc + y; c = (s - acc) - y`。
  接口允许 `num_splits ≤ 256`，顺序/2 路累加的误差随段长增长，Kahan 与段长无关（恒 ~1 ULP）。
  910 复用 4 个 `rows×D` tile（acc/comp/oTile/wTile）；950 额外多一块 `yTile`（per_row 12→16 字节/tile）。
- 全 `-inf` 行：`max` clamp 到 `lowest()` → `e_s = 0` → `den = 0`；`den` clamp 到 **1**
  （正常行 `den ≥ 1`，clamp 不生效）→ `out = 0`，不会出现 `nan`/`inf`。
- 判据（pytest）：`out` 偏差 ≤ 2× 一次 cast 误差 + 1e-3；ATK 走 `cv_fused_double_benchmark` 的 L2 三比值。

### 2.7 910 与 950 的差异（同一算法、两套落地）

| | 910（A2/A3） | 950（A5） |
|---|---|---|
| 现成参考 | `combine_scale.hpp`（融合 epilogue，双 subblock 分半） | `fd_combine.hpp`（**已是独立 kernel**，UB 偏移表 + Reduce/BroadCast 流程可直接搬） |
| 复用程度 | 只有 UB 规划与尾块处理可参考（它绑定 `splitInfo`，不能直接用） | 骨架可复用，需把"packed 输入"换成"dense 输入"、把 O 的 16-bit 换成 fp32 |
| launch 约定 | `rtGetC2cCtrlAddr` 取 `fftsAddr`，`at_npu::native::OpCommand::RunOpApiV2("ascendc_fa_combine", lambda)`（`flash_api.cpp:633-687`） | 同款 `RunOpApiV2` 包裹 `launch_combine`（`mha_fwd.cpp:514`），kernel 侧无 ffts/`SetSyncBaseAddr` 需求 |
| 编译依赖 | CATLASS 头（`csrc/catlass` 子模块） | 同 |

### 2.8 编译组织

沿用仓内既有模式：`autogen/generate_kernels.py` 为 **每个 (dtype, layout)** 生成一个 TU（避免模板串行编译），combine 新增 `combine_<dtype>_<layout>.cpp`（每 backend 4 个），并在 `setup.py` 的 `src_ascend910_v3/v4`、`src_ascend950_v3/v4`（`setup.py:607-643`）补 glob。若嫌 TU 过多，退路是单个 TU 内部做 dtype/layout `switch`，代价是编译并行度下降。

### 2.9 torch.compile

按仓内约定给 `fwd_combine` 配 `torch.library.custom_op` + `register_fake`（与 `_flash_attn_forward` 同款），保证 fake tensor / 图模式下可 trace；`out`/`lse` 作为可选输出时注意 fake 侧的形状与 dtype 推导。

---

## 3. 文件级落地清单

| 类别 | 文件 | 改动 |
|---|---|---|
| Python（FA3-910） | `flash_attn_npu_3/flash_attn_npu_interface.py` | 修真 `:1165`，改调真实 op，补校验/docstring/fake |
| Python（FA3-950） | `flash_attn_npu_3/flash_attn_npu_interface_950.py` | 新增 `flash_attn_combine` |
| Python（FA4-910） | `flash_attn_npu_4/flash_attn_npu_interface.py` | 新增（含 `cu_seqlens/seqused/return_lse`） |
| Python（FA4-950） | `flash_attn_npu_4/flash_attn_npu_interface_950.py` | 新增 |
| 导出 | `flash_attn_npu_3/__init__.py`、`flash_attn_npu_4/__init__.py` | 两个分支都加 `flash_attn_combine` |
| C++ host + 注册 | `csrc/ascend910/flash_attn_npu_3/flash_api.cpp`、`csrc/ascend910/flash_attn_npu_4/flash_api.cpp`、`csrc/ascend950/flash_attn_npu_3/flash_api.cpp`、`csrc/ascend950/flash_attn_npu_4/flash_api.cpp` | 各加 `m.def("fwd_combine", &mha_combine, ...)` + host 实现（校验/tiling/launch） |
| C++ kernel（新增） | `csrc/ascend910/flash_attn_npu_{3,4}/combine_*.{hpp,cpp}`、`csrc/ascend950/flash_attn_npu_{3,4}/combine_*.{hpp,cpp}` | 独立 Vector kernel（每 backend 一套，v3/v4 可共享同一份实现） |
| C++ codegen | `csrc/ascend{910,950}/flash_attn_npu_{3,4}/autogen/generate_kernels.py` | 增加 combine 的 per-(dtype,layout) TU 生成 |
| 构建 | `setup.py` | 4 处 source glob 补 combine TU |
| 文档 | `README.md`、`README.zh.md` | v3/v4 各加 `#### flash_attn_combine` 章节 + Features 表 |
| 测试 | `tests/test_flash_attn_npu_v3_combine.py`、`tests/test_flash_attn_npu_v4_combine.py` | 新增 |

---

## 4. 验证方案（接口验收）

**T1 精度矩阵（主验收）**：随机 fp32 partial（含 `-inf` 短路段、非连续输入）
· `out_dtype ∈ {fp16, bf16}`（+ fp32 若采纳 §0.2）
· `D ∈ {64, 96, 128, 192, 256, 512}`
· `S ∈ {1, 2, 32, 113, 256, 1024}`
· `num_splits ∈ {1, 2, 3, 5, 17, 32, 256}`
· 布局 `{BSND, TND}`；TND 再分 `cu_seqlens` / `seqused` / 两者同时
判据：golden 用 NV 同款参考式
`lse = logsumexp(lse_partial, 0)`；`out = Σ exp(lse_j - lse) · O_j`；
比对 `lse`：`atol=rtol=1e-5`；`out`：`max_diff ≤ 2 × (cast 后参考与之差)`，并接入 `tests/common/compare.py::assert_fa_close` 的双标杆规则。

**T2 异常/边界**：partial 非 fp32、`num_splits>256`、shape/stride 不匹配、`out` dtype 不符、`out_dtype` 非法、`S=0`、`H=0`、`D` 非对齐（96/113）。

**T3 torch.compile**：fake/meta 下可 trace，与 fwd 同级别。

**T4 性能（非验收项）**：本次需求明确"性能在 FD 体现，此处不要求性能"，因此只记录 combine 单算子的耗时与带宽利用率（memory-bound，理论上限≈`partial_bytes / HBM 带宽`）作为参考数据，不设门禁。

**T5 环境**：本机 8×910B3 可做 A2 验证；需先 `git submodule update --init csrc/catlass`（当前子模块目录为空）并安装 torch/torch_npu（当前默认 Python 无 torch）。A5(950) 需走 CI 的 950 镜像（`ci/build_matrix.tsv`）。

---

## 5. 工作量估算与实施顺序

| 阶段 | 内容 | 粗估 |
|---|---|---|
| S1 | 契约冻结（签名/布局/dtype/报错、是否支持 fp32 输出、`num_splits` 上限） | 1~2 人日 |
| S2 | 910 combine kernel + host + pybind（BSND/TND × fp16/bf16） | 6~10 人日 |
| S3 | 950 combine kernel + host + pybind（骨架来自 `fd_combine.hpp`） | 6~10 人日 |
| S4 | v3/v4 两侧 Python 接口 + 导出 + fake/meta | 3~4 人日 |
| S5 | 测试（T1~T4） | 6~8 人日 |
| S6 | 文档（README 双语 + docstring） | 2 人日 |

**最小可交付**：S1+S2+S4(FA3-910)+S5 的一部分 ≈ **10~14 人日**即可让 A2 上 `flash_attn_combine` 可用并可测；全量（4 backend，覆盖 A2/A3/A5）≈ **24~36 人日**。建议顺序：**910 → 950**，FA3 → FA4，避免一开始就并行四路。

---

## 6. 决策点与风险

| # | 事项 | 建议 |
|---|---|---|
| D1 | `out_dtype=None` 的默认值 | 对齐 NV（= fp32），因此建议支持 fp32 输出；否则接口一致性会被默认调用打破 |
| D2 | 布局：按 stride 解释 vs 只认一种 dim 序 | 按 stride 解释（NV 自身两种 dim 序都出现过），并强制 `O.stride(-1)==1`、`LSE.stride(-2)==1` |
| D3 | partial 是否必须 fp32 | 是（NV 硬约束）。这也意味着未来若想让 fwd 产出 partial，950 需把 16-bit 的 partial O 改成 fp32 |
| D4 | 端到端（fwd 产出 partial）是否要做 | 本期不做；若要做，另立一项（涉及 split 路径 dense 化与 FD gate 放开） |
| D5 | 性能基线 | CANN 无对外 combine 算子，需明确基线（FD 内部 combine 同 shape / torch 加权求和） |
| D6 | 内存阈值口径 | 建议明确为"调用者 partial 峰值 + 算子 UB"，文档给出公式与推荐 `num_splits` |
| D7 | `virtual_batch_idx` / 动态 split 数 | 本期显式 not supported；本仓无 persistent scheduler 与动态 split 计数消费方 |
| R1 | 950 无法本地验证 | 依赖 CI 950 镜像，迭代周期长；建议 950 与 910 并行开发但 910 先行验证算法正确性 |
| R2 | 编译模板膨胀 | per-(dtype,layout) TU 会新增 16 个 TU（4 backend × 4），编译时间上升；必要时合并为单 TU + switch |
| R3 | `num_splits` 大 + `S` 大时带宽放大 | 读流量 = `num_splits × partial_bytes`，需在文档中给出推荐范围，避免用户误用 |

---

## 7. 实施结果（本次交付）

### 7.1 代码落地

| 层 | 文件 |
|---|---|
| 910 kernel | `csrc/ascend910/flash_attn_npu_{3,4}/combine_kernel.cpp`、`combine_dispatch.hpp` |
| 950 kernel | `csrc/ascend950/flash_attn_npu_{3,4}/combine_kernel.cpp`、`combine_dispatch.hpp` |
| host + pybind | 4 个 `flash_api.cpp` 各新增 `mha_combine` 与 `fwd_combine` 注册（`py::arg` 默认值与 NV 对齐） |
| Python | `flash_attn_npu_3/flash_attn_npu_interface{,_950}.py`、`flash_attn_npu_4/flash_attn_npu_interface{,_950}.py`；两个 `__init__.py` 导出 |
| 构建 | `setup.py` 的 4 个 source list 各补 `combine_kernel.cpp` |
| 文档 | `README.md` / `README.zh.md` 新增 `flash_attn_combine` 章节 + Features 表 |
| 测试 | `tests/test_flash_attn_npu_v3_combine.py`（94 例）、`tests/test_flash_attn_npu_v4_combine.py`（33 例） |

### 7.2 验证结果

| 项 | 结果 |
|---|---|
| A2/A3（910B3, CANN 9.1.0, torch_npu 2.7.1） | FA3 与 FA4 后端 `build_ext` 成功；`pytest` **127 passed**（94 + 33），连续多轮稳定 |
| A5（950, dav-3510） | FA3 与 FA4 后端 **编译 + 链接成功**（`--npu-arch=dav-3510`）；本机无 950 设备，**未上板运行** |
| 精度 | 与 `logsumexp` 参考实现对比：`lse` atol/rtol 1e-4；`out` 偏差 ≤ 2× 一次 cast 误差 + 1e-3 |
| 接口契约 | 覆盖 fp32 partial 强制、`stride(-1)/stride(-2)`、`num_splits ≤ 256`、`out` 原地写、`out_dtype` 默认 fp32、`return_lse=False`、`virtual_batch_idx` not-supported |
| ATK 精度（A2/A3） | FA3 **100/100**、FA4 **100/100** 通过（`cv_fused_double_benchmark` **L2: 2/1.2/1.2**，实测 max_re 2.0、avg_re 0.002、rmse 0.050）。详见 `docs/analysis/flash_attn_combine_atk_result.md` |

### 7.3 实现中确认的硬件约束（重要，后续复用）

1. **MTE2 `DataCopyPad` 的 blockLen 不能小于 32B**：`seqlen=1` 时 4B 的 LSE 搬运会直接触发 aicore exception。实现改为"按 8 元素对齐搬运，尾部不安全时退化为标量读取"。
2. **`rightPad` 不能跨过 32B 块（≤ 7 个元素）**：把 LSE/O 的 pad 从"补齐到 tile 行宽"改为"只补到 8 元素对齐"，其余尾列由预清零保证。
3. **`Ln(0)` 返回最低有限浮点而不是 `-inf`**：全 mask 行的 LSE 需要标量修正为 IEEE `-inf`，否则与 NV 语义不一致。
4. **buffer 复用需要 MTE3→V 同步**：只做 V→MTE3 时，下一 tile 的 `Cast` 会与上一 tile 的 GM 写回竞争，表现为随机的 aicore exception。
5. **VEC 侧 `GetBlockIdx()` 是扁平 AIV 编号**：任务分配用 `for (t = GetBlockIdx(); t < taskCount; t += GetBlockNum() * GetSubBlockNum())`。
6. 910 用 `BroadCast<float,2,1>` 做"每行权重广播"；950 改用 `fd_combine` 已验证的 per-row `Muls + GetValue`（避免 arch35 上的 API 差异）。

### 7.4 剩余工作

- A5 上板精度/性能验证（需 950 CI 镜像）；
- 性能数据仅作记录（本次无性能门禁，收益在 FD 端到端体现）；
- 内存峰值实测与 L2 阈值口径核对；
- 端到端（fwd `num_splits>1` 产出 dense fp32 partial）仍为二期范围。

