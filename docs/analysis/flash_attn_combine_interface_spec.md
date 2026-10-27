# `flash_attn_combine` 接口规格（接口维度分析）

> 本文只从**接口维度**回答两件事：要交付哪些接口、每个接口长成什么样（签名 / 参数 / 形状 / dtype / 返回值 / 报错 / 文档 / 验收）。
> 实现侧（kernel 切分、UB 预算、文件清单）见 `flash_attn_combine_interface_implementation.md`；需求整体分析见 `flash_attn_combine_requirement_analysis.md`。
> 对齐基准：`Dao-AILab/flash-attention` — FA3 `hopper/flash_attn_interface.py:938` + `hopper/flash_api.cpp:1571-1736`；FA4 `flash_attn/cute/interface.py:4468-4556`。

---

## 1. 要交付哪些接口

**两个函数签名、四个可 import 的入口**（因为 v3/v4 在 910/950 上是不同的包与 backend）：

| # | import 路径 | 对应 NV 的模块 | 使用签名 |
|---|---|---|---|
| 1 | `from flash_attn_npu_3 import flash_attn_combine` | `flash_attn_interface`（FA3/Hopper） | FA3 版 |
| 2 | `from flash_attn_npu_3 import flash_attn_combine`（设备为 950 时自动走 `_interface_950`） | 同上 | FA3 版 |
| 3 | `from flash_attn_npu_4 import flash_attn_combine` | `flash_attn.cute.interface`（FA4） | FA4 版 |
| 4 | `from flash_attn_npu_4 import flash_attn_combine`（950 自动走 `_950`） | 同上 | FA4 版 |

**接口维度的目标形态**：一个用 NV 写好的模型/框架，**只改 import 那一行**就能跑通——函数名、参数名、参数顺序、默认值、返回顺序、张量契约全部一致。

> 注：包名不可能与 NV 相同（我们是 `flash_attn_npu_3/4`，且 910/950 在同一个包里按设备分流），这是唯一不可避免的差异，需要在 README 里给出"迁移对照表"。

---

## 2. 签名规格（逐字对齐）

### 2.1 FA3 版

```python
# NV: hopper/flash_attn_interface.py:938
def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None):
    return flash_attn_3_gpu.fwd_combine(out_partial, lse_partial, out, out_dtype)
```

```python
# 本仓（910 / 950 各一份实现，签名完全相同）
def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None):
    """把多份 partial attention 结果按 LSE 加权合并成最终结果。..."""
    return <backend>.fwd_combine(out_partial, lse_partial, out, out_dtype)
```

底层 op schema（FA3 对标 `flash_attn_3::fwd_combine`）：

```
fwd_combine(Tensor out_partial, Tensor lse_partial, Tensor(out!)? out=None,
            ScalarType? out_dtype=None) -> (Tensor(out!), Tensor)
```

### 2.2 FA4 版

```python
# NV: flash_attn/cute/interface.py:4468
def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    virtual_batch_idx: Optional[torch.Tensor] = None,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
```

本仓建议保留 `virtual_batch_idx` 参数名但显式 `NotImplementedError`（仓内已有"参数保留 + Not supported on NPU"的惯例，如 FA4 docstring 里的 `pack_gqa`/`score_mod`）。

### 2.3 参数名也是接口契约

下游迁移代码可能用关键字调用，例如
`flash_attn_combine(out_partial=p, lse_partial=l, out_dtype=torch.bfloat16)`，
所以 **参数名、顺序、默认值都不能自由发挥**；`out_dtype` 不能改叫 `dtype`，`return_lse` 不能改叫 `return_lse_`。

---

## 3. 参数逐项规格

| 参数 | 类型 | 默认 | 形状 | dtype | stride 约束 | 说明 |
|---|---|---|---|---|---|---|
| `out_partial` | Tensor | 必填 | dense：`(num_splits,B,S,H,D)`；varlen：`(num_splits,T,H,D)` | **必须 fp32** | `stride(-1) == 1` | 各分片的未归一化输出。NV 的 FA3 fwd 实际产出为 `(num_splits,B,H,S,D)`，因此实现**按 stride 解释**，两种维序都要能吃 |
| `lse_partial` | Tensor | 必填 | dense：`(num_splits,B,S,H)`；varlen：`(num_splits,T,H)` | **必须 fp32** | **`stride(-2) == 1`**（seqlen 维连续） | 各分片的 logsumexp。NV 的报错文案就是 "LSE tensor must be contiguous in the seqlen dimension" |
| `out` | Tensor? | `None` | dense：`(B,S,H,D)`；varlen：`(T,H,D)` | 必须等于 `out_dtype` | `stride(-1) == 1` | 传入即**原地写**，返回值就是同一个张量（NV 的 `Tensor(out!)`） |
| `out_dtype` | torch.dtype? | `None` | — | `fp16 / bf16 / fp32` | — | **`None` → 取 `out_partial.dtype`（即 fp32）**，与 NV 一致 |
| `cu_seqlens` (FA4) | Tensor? | `None` | `(B+1,)` | int32 | 连续 | packed varlen 的分段边界；与 4D `out_partial` 搭配 |
| `seqused` (FA4) | Tensor? | `None` | `(B,)` | int32 | 连续 | 每段有效长度；只在有效段内保证正确（与 NV 一致），dense 模式下按 batch 对齐 |
| `virtual_batch_idx` (FA4) | Tensor? | `None` | `(B,)` | int32 | 连续 | **不支持**（依赖 persistent tile scheduler）→ `NotImplementedError` |
| `return_lse` (FA4) | bool | `True` | — | — | — | `False` 时返回 `(out, None)` |

补充说明：

- FA3 版**没有** `cu_seqlens/seqused`，varlen 场景按"batch 视为 1、T 维当作行维"处理（与 NV 一致）。
- `num_splits` 不是参数，它由 `out_partial.shape[0]` 隐式给出；上限 **256**。
- 不要求 partial 张量连续（NV 测试特意用非连续输入验证），只要求上面两条 stride 约束。

---

## 4. 返回值规格（这里有一个必须提前知道的坑）

返回值恒为二元组 `(out, lse)`；`return_lse=False` 时 `lse is None`。

| 返回项 | dense | varlen | dtype |
|---|---|---|---|
| `out` | `(B, S, H, D)` | `(T, H, D)` | 按 `out_dtype` |
| `lse` | **`(B, S, H)`** | **`(T, H)`** | fp32 |

**坑**：NV 的 combine 返回的 LSE 是 **`(B,S,H)` / `(T,H)`，且 seqlen 维连续**（FA3 `flash_api.cpp:1633` 用 `empty({B,H,S}).transpose(1,2)`；FA4 `interface.py:4544` 用 `empty(B,H,S).transpose(-1,-2)`），而**本仓 attention fwd 返回的 `softmax_lse` 是 `(B,H,S)` / `(H,T)`**（README 与 python fake 都是这个形状）。

也就是说：`flash_attn_combine` 的 LSE 与本仓 `flash_attn_func` 的 LSE **形状约定不同**。这是 NV 自身两代都存在的现象，不是我们的实现选择。处理方式见 §12 的 D2。

另外：

- 传入 `out` 时返回的就是同一个张量（原地），调用者不应依赖"返回新张量"。
- `lse` 恒为 fp32，与 `out_dtype` 无关。

---

## 5. 支持矩阵（接口承诺要覆盖的范围）

| 维度 | 取值 |
|---|---|
| 硬件 | A2 / A3（910 路径）、A5（950 路径） |
| API 代际 | FA3、FA4 |
| 布局 | dense（BSND）、varlen（TND，含 `cu_seqlens`、`seqused`、两者同时） |
| `out_partial` / `lse_partial` | fp32（唯一允许） |
| `out_dtype` | fp16、bf16（建议一并支持 fp32，理由见 D1） |
| `num_splits` | 1 ~ 256 |
| `head_dim` | 64 / 96 / 128 / 192 / 256 / 512，含非 8 元素对齐值 |
| `seqlen` | 1 起，到大序列（受调用者显存约束，接口不额外设限） |
| GQA / MQA | 由 `H` 维度自然覆盖（combine 只按 head 索引，不关心 Q/KV head 比） |

---

## 6. 校验与报错清单（接口契约的一部分）

下游可能 `try/except` 或依赖错误文案定位问题，所以报错类型与关键字要与 NV 对齐：

| # | 检查 | 触发条件 | 报错类型 | 建议文案（对齐 NV） |
|---|---|---|---|---|
| 1 | partial dtype | `out_partial` / `lse_partial` 非 fp32 | `RuntimeError` | `Attention combine function only support fp32 data type` |
| 2 | O 连续性 | `out_partial.stride(-1) != 1` | `RuntimeError` | `Input tensor must have contiguous last dimension` |
| 3 | LSE 连续性 | `lse_partial.stride(-2) != 1` | `RuntimeError` | `LSE tensor must be contiguous in the seqlen dimension` |
| 4 | `num_splits` 上限 | `num_splits > 256` | `RuntimeError` | `FlashAttention combine only supports num_splits at most 256` |
| 5 | 输出 dtype 合法性 | `out_dtype ∉ {fp16,bf16,fp32}` | `RuntimeError` | `Output type must be FP32, FP16 or BF16` |
| 6 | `out` dtype/shape | 与 `out_dtype`/目标形状不符 | `RuntimeError` | 明确列出期望与实际 |
| 7 | shape 自洽 | `lse_partial` 与 `out_partial` 去掉 D 后不一致 | `RuntimeError` | 明确列出两侧形状 |
| 8 | 维度数 | `out_partial.dim()` 不是 4/5（FA4） | `RuntimeError`（NV 是 `AssertionError`） | `out_partial must have 4 or 5 dimensions` |
| 9 | 设备 | 非 NPU 张量 | `RuntimeError` | 提示需在 NPU 上 |
| 10 | 不支持参数 | FA4 传 `virtual_batch_idx` | `NotImplementedError` | `virtual_batch_idx is not supported on NPU` |

> 注意第 8 条：NV 的 FA3 用 `TORCH_CHECK`（→`RuntimeError`），FA4 用 `assert`（→`AssertionError`）。建议我们统一成 `RuntimeError/ValueError` 并在文档中注明这一差异，而不是刻意复刻 `AssertionError`。

---

## 7. 行为语义（接口层面承诺什么）

数学定义（与 NV 完全一致）：

```
lse = logsumexp_s(lse_partial)                      # s 为 split 维
w_s = exp(lse_s - lse)                              # inf / nan → 0
out = Σ_s w_s · O_s                                 # 累加在 fp32 中进行
```

边界行为：

| 场景 | 承诺 |
|---|---|
| 某些 split 的 `lse_s = -inf`（该分片没覆盖这段 KV） | 该分片权重为 0，不产生 NaN |
| 所有 split 的 `lse_s = -inf` | `out = 0`，`lse = -inf` |
| `num_splits == 1` | 等价于"直接 cast 到 `out_dtype`"，结果与 `exp(0)·O` 一致 |
| `S == 0` 或 `H == 0` | 不 launch，返回空张量（不报错） |
| 输入含 NaN | 按 IEEE 传播，不做额外清洗（与 NV 一致） |
| `seqused` 段外位置 | **不保证正确**（与 NV 一致）；文档需写明 |
| 确定性 | 固定 split 累加顺序，结果确定；不涉及确定性开关 |
| 与"一次算完"的等价性 | 与 `num_splits=1` 的完整结果相比，偏差应不超过"一次 dtype cast"的量级 |

---

## 8. 与 NV 的一致 / 有意差异 / 不支持

| 类别 | 项 |
|---|---|
| **完全一致** | 函数名、参数名/顺序/默认值、返回顺序、`out!` 原地语义、partial 必须 fp32、两条 stride 约束、`num_splits ≤ 256`、`out_dtype` 默认 = `out_partial.dtype`、返回 LSE 的形状与连续性约定、数学定义、`-inf` 短路语义 |
| **有意差异（需文档化）** | ① 包名/import 路径不同（唯一无法对齐项）；② 不支持 `virtual_batch_idx` / 动态 split 数 / `semaphore_to_reset`；③ 报错类型统一为 `RuntimeError/ValueError`（NV FA4 用 `assert`）；④ `out` 传入且 `head_dim` 非对齐时的写回行为（见 D3） |
| **不支持** | fp8 partial、`score_mod`/`mask_mod` 等 FA4 扩展、CUDA 专有语义（如 PDL/semaphore） |

---

## 9. 使用姿势（接口视角的三个场景）

**场景 1：同设备内 split-KV 合并**（最典型）

```python
from flash_attn_npu_3 import flash_attn_combine

# 4 份分片各自算完的中间结果（例如 KV 被切成 4 段分别做 attention）
out_partial = torch.randn(4, B, S, H, D, dtype=torch.float32, device="npu")
lse_partial = torch.randn(4, B, S, H, dtype=torch.float32, device="npu")

out, lse = flash_attn_combine(out_partial, lse_partial, out_dtype=torch.bfloat16)
# out: (B,S,H,D) bf16；lse: (B,S,H) fp32
```

**场景 2：上下文并行 / 跨 rank 合并**（各 rank 算自己 KV 段的 partial，最后在某一 rank 合并）

```python
# 每个 rank 产出 rank 维度的 partial，all-gather 后维度即为 num_splits
out, _ = flash_attn_combine(partials_gathered, lse_gathered,
                            out=preallocated_out, out_dtype=torch.bfloat16)
```

**场景 3：varlen（TND）**

```python
from flash_attn_npu_4 import flash_attn_combine

out, lse = flash_attn_combine(
    out_partial,          # (num_splits, total_q, H, D) fp32
    lse_partial,          # (num_splits, total_q, H) fp32，token 维连续
    cu_seqlens=cu_q,      # (B+1,) int32
    seqused=seq_used,     # (B,) int32，可选
    out_dtype=torch.bfloat16,
)
```

**场景 4（当前不可用，二期）**：`flash_attn_func(..., num_splits=k)` 产出 partial → combine。本仓 fwd 目前返回的 `out_accum/lse_accum` 恒为空张量，所以这条链路要等二期把 partial 暴露出来才能闭环。

---

## 10. 资料交付形态

| 交付物 | 内容要点 |
|---|---|
| Python docstring | 参数表（含形状/dtype/约束）、Returns、`-inf` 短路说明、**与 fwd 的 LSE 形状差异提示**、不支持项 |
| `README.md` / `README.zh.md` | 在 `### FlashAttention v3` 与 `### FlashAttention v4` 下各加 `#### flash_attn_combine` 章节，体例与既有 `flash_attn_func` 章节一致（签名 + 参数说明 + Returns） |
| Features 表 | 三个现有表格（kvcache / func / varlen_func）之外，新增"Split-KV Combine"一节，v3/v4 各标 ✅ 与不支持项 |
| 迁移对照表 | NV import → 本仓 import 的一行替换说明（FA3/FA4 各自一行） |
| 不支持项 | `virtual_batch_idx` 等显式标注 "Not supported on NPU" |

---

## 11. 接口维度的验收用例

| # | 用例 | 判定 |
|---|---|---|
| V1 | 4 条 import 路径 | 均可 `from ... import flash_attn_combine` |
| V2 | 签名一致性 | `inspect.signature(...)` 与 NV 签名（硬编码为期望值）逐参比对：参数名、顺序、默认值 |
| V3 | 形状/dtype/stride 矩阵 | §5 的矩阵逐组合跑通，返回值 shape/dtype 断言 |
| V4 | `out` 原地语义 | 传入 `out` 后 `data_ptr()` 不变且数值被写入 |
| V5 | 非连续输入 | 用 `transpose` 构造的非连续 partial 能正确计算（对应 NV 测试里的 non-contiguous 用例） |
| V6 | 报错契约 | §6 的 10 条各一个用例，断言异常类型 + 消息关键字 |
| V7 | `return_lse=False` | 返回 `(out, None)`，且 `out` 与 `True` 时逐元素一致 |
| V8 | `-inf` 短路 | 后半段 split 置 `-inf`，输出与参考一致、无 NaN |
| V9 | torch.compile | fake/meta 下可 trace（与 fwd 同级别） |
| V10 | 数值正确性 | 与 §7 的数学参考比对（`lse` 1e-5 级、`out` 不超过一次 cast 量级） |

---

## 12. 待拍板的接口决策（5 条，都会写进接口规格）

| # | 决策点 | 选项 | 建议 |
|---|---|---|---|
| D1 | `out_dtype=None` 的默认 | (a) 对齐 NV = fp32；(b) 取 fp16/bf16 | **(a)**；因此需要支持 fp32 输出，否则默认调用返回的 dtype 与 NV 不一致 |
| D2 | 返回 LSE 的形状 | (a) 严格对齐 NV：`(B,S,H)`/`(T,H)`（seqlen 连续）；(b) 与本仓 fwd 统一为 `(B,H,S)`/`(H,T)` | **(a) + 文档显著提示**；若选 (b) 就破坏了"接口一致性"，迁移代码里 `lse.transpose(-1,-2)` 之类的写法会失效 |
| D3 | 传入 `out` 且 `head_dim` 非对齐时 | (a) 写回用户张量；(b) 报错；(c) 复刻 NV 的"静默忽略" | **(a)**，并在文档标注与 NV 的差异（NV 此时会新分配一个张量、用户传入的 `out` 不会被写） |
| D4 | 不支持参数的容纳方式 | (a) 保留参数名 + `NotImplementedError`；(b) 不放进签名（调用即 `TypeError`） | **(a)**：报错信息更明确，且签名与 NV 完全一致 |
| D5 | `num_splits` 上限 | (a) 256（对齐 NV）；(b) 按本仓规模收窄 | **(a)**，实现侧若确有资源约束，用清晰报错暴露而不是改签名 |

