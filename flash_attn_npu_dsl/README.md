# CATLASS DSL Flex Attention

本包聚焦通过 `score_mod` 和 `mask_mod` 定制注意力前向：用户提供回调，
修改 softmax 前的分数或定义可见位置，无需为每种注意力规则重写完整 kernel。

- `score_mod`：对缩放后的 QK 分数施加自定义变换。
- `mask_mod`：判断每个 Q/KV 位置是否可见，可表达因果、滑窗等规则。
- 回调支持普通 Python 函数和 `@tla.jit` helper，按声明使用 SIMT 或显式 SIMD，
  内联到 attention 内核；需要跳过空块时，可显式建块并传入稀疏元数据。

当前实现面向 Ascend 950 的推理前向，支持定长 BSND、变长 TND、只读分页 KV、
FP16/BF16、1～256维 head dimension 和 GQA。
Q/K 使用相同的 Dqk；V 可以独立选择1～256维的 Dv，O 的最后一维随 V。
通过独立的 `flash_attn_npu_dsl` 包使用，不替换现有 V2/V3/V4 的 C++ 后端。

静态 modifier／块表路径的 Tensor 使用连续 ND/NCHW 物理存储；特殊分块存储格式
须由调用方显式转换。这与逻辑上的 BSND 排布是不同的概念。
编译时构造 DSL Tensor 样例，复用编译对象时直接绑定本次 Torch Tensor 的地址，
不复用旧地址或输出。跨 stream 使用仍须由调用方建立依赖并保证输入的生命周期。

## 安装、运行与最小调用

本模块需要 Ascend 950、Linux、Python `>=3.10,<3.14` 和 CANN `>=9.1.0`。
PyTorch 与 torch_npu 版本须匹配，并与所安装的 CATLASS DSL 二进制运行时兼容。
先按[本仓安装指南](../README.zh.md)准备 CANN 环境变量、PyTorch 和 torch_npu；
在本仓源码根目录安装时使用：

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install . --no-build-isolation
```

源码安装显式强制本地构建，以确保分发包包含当前源码中的 Python 模块。
此命令沿用本仓现有 C++ 构建流程，不是独立的纯 Python 安装模式。

CATLASS DSL 是额外依赖：使用 CATLASS `master` 分支中 `79f0ac18` 及之后的版本，按
`python/tla_dsl/docs/zh/dsl_development/index.md` 构建 Release 模式的
`ascend-catlass-dsl` wheel，再按其 `python/tla_dsl/docs/zh/quick_start.md` 安装。
运行时需支持递归结构化参数、callable/JIT helper 和 SIMD。
其 Python 导入名是 `catlass`；本仓的 `csrc/catlass` C++ 头文件子模块不能替代该运行时，
安装本仓也不会自动构建或安装 DSL。将 `CATLASS_DSL_WHEEL` 设置为匹配当前环境的 wheel
文件路径后，安装并预编译数据搬运等 bitcode 模板：

```bash
python -m pip install "$CATLASS_DSL_WHEEL"
python -c "import catlass"
python -m catlass.bc_compile
python -m flash_attn_npu_dsl.example --device 0
python -m flash_attn_npu_dsl.example --simd --sparse --dtype bf16 --head-dim 96
python -m flash_attn_npu_dsl.example --simd-score --sparse --q-len 257 --kv-len 129
python -m flash_attn_npu_dsl.example --head-dim 256 --value-dim 192 --simd --sparse
```

DSL 的 Python 包、二进制扩展和 bitcode 必须配套。不要从未构建的 CATLASS Python
源码目录运行示例，以免同名源码包遮蔽已安装的 wheel。更换 DSL 或 CANN 后，运行
`python -m catlass.bc_compile --force` 重新生成 bitcode；attention 内核在调用时 JIT 编译。

### 运行示例

示例默认使用 SIMT 回调，演示 GQA、右对齐 causal mask 和 `0.75` 分数缩放，
并用小规模 CPU FP32 结果检查 O/LSE。`--simd-mask` / `--simd-score` 分别切换 mask / score，
`--simd` 等价于同时指定这两个选项；`--sparse` 启用显式建块，
`--no-lse` 只计算 O；`--q-len 257 --kv-len 129` 可覆盖前部空行。

安装完成后，可直接传入一个自定义右对齐 causal `mask_mod`，无需修改 `PYTHONPATH`：

```python
import torch
import torch_npu
from flash_attn_npu_dsl import flash_attn_func


def causal_mask(b, h, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars=None):
    return kv_idx <= q_idx + seqlen_info.seqlen_k - seqlen_info.seqlen_q


torch.npu.set_device(0)
B, Sq, Sk, Hq, Hkv, D = 1, 129, 257, 2, 1, 128
q = torch.randn(B, Sq, Hq, D, device="npu:0", dtype=torch.float16)
k = torch.randn(B, Sk, Hkv, D, device=q.device, dtype=q.dtype)
v = torch.randn_like(k)
out, lse = flash_attn_func(q, k, v, mask_mod=causal_mask, return_lse=True)
```

## 输入、返回值与常用参数

Q/K/V 为同设备、同 dtype、连续的 FP16 或 BF16 NPU 张量，采用 BSND 布局。
Q 为 `(B, Sq, Hq, Dqk)`，K 为 `(B, Sk, Hkv, Dqk)`，V 为 `(B, Sk, Hkv, Dv)`；所有维度为正，
`Dqk` 和 `Dv` 各支持1～256，`Hq` 必须是 `Hkv` 的整数倍（GQA）。
仅支持推理前向，不支持梯度或 KV cache 写入；未实现的选项必须保持默认值。

`flash_attn_func` 始终返回 `(out, lse)`。O 为 `(B, Sq, Hq, Dv)`，dtype 与 Q 相同；
`return_lse=True` 返回 FP32 自然对数 LSE，形状为 `(B, Hq, Sq)`，否则 LSE 为 `None`。
完全被 mask 的行返回精确零 O 和 `-inf` LSE。

### 变长 TND

`flash_attn_varlen_func` 接受连续的 Q `[total_q,Hq,Dqk]`、K `[total_k,Hkv,Dqk]`
和 V `[total_k,Hkv,Dv]`。
`cu_seqlens_q/k` 是同设备连续 int32 `[B+1]` 累计存储偏移；可选的
`seqused_q/k` 是 int32 `[B]` 有效长度，可以小于存储段长度。O 为 `[total_q,Hq,Dv]`，
LSE 为 `[Hq,total_q]`，仍返回 `(out,lse)`。非分页输入接受 Q/K/V 全部为 TND
或全部为 BSND，不混用两种存储形式。

```python
from flash_attn_npu_dsl import flash_attn_varlen_func

# 两条序列：Q 长度 65、129，KV 长度 128、257。
cu_q = torch.tensor([0, 65, 194], dtype=torch.int32, device=q.device)
cu_k = torch.tensor([0, 128, 385], dtype=torch.int32, device=q.device)
q_packed = torch.randn(194, Hq, D, dtype=q.dtype, device=q.device)
k_packed = torch.randn(385, Hkv, D, dtype=q.dtype, device=q.device)
v_packed = torch.randn_like(k_packed)
out, lse = flash_attn_varlen_func(
    q_packed,
    k_packed,
    v_packed,
    cu_seqlens_q=cu_q,
    cu_seqlens_k=cu_k,
    max_seqlen_q=129,
    max_seqlen_k=257,
    causal=True,
    return_lse=True,
)
```

`max_seqlen_q/k` 是调度上界；省略时使用总 token 容量，不在热调用中回读长度。
callback 的 q/kv 坐标仍是序列内坐标，`seqlen_info` 同时提供当前有效长度与
`offset_q/k` 累计存储偏移。右对齐 causal/window 按各序列的有效长度计算。
空序列合法；有效 Q 行没有可见 KV 时 O=0、LSE=-inf；`seqused_q` 以外的
存储行没有数值保证。

调用方保证累计偏移单调、从零开始且终值匹配存储容量，有效长度不超过存储段，
并且调度上界覆盖实际有效长度。接口只检查元数据的形状、dtype、设备和连续性。
元数据原地更新或地址替换会在下次调用生效；地址和内容不进入编译 key，
总容量或编译期策略变化则可能产生新编译对象。

### BSND 存储中的有效长度

无需将带 Padding 的 BSND 输入重新打包成 TND。使用同一 varlen 入口，
不传 `cu_seqlens_q/k`，通过连续 int32 `[B]` 的 `seqused_q/k` 指定有效长度：

```python
# 两条序列保留相同存储容量，但 KV 有效长度不同。
q_bsnd = torch.randn(2, 65, Hq, D, dtype=q.dtype, device=q.device)
k_bsnd = torch.randn(2, 257, Hkv, D, dtype=q.dtype, device=q.device)
v_bsnd = torch.randn_like(k_bsnd)
used_k = torch.tensor([129, 257], dtype=torch.int32, device=q.device)
out, lse = flash_attn_varlen_func(
    q_bsnd, k_bsnd, v_bsnd, seqused_k=used_k, causal=True, return_lse=True,
)
```

未提供的有效长度按相应存储容量处理。O 保持 `[B,Sq,Hq,Dv]`，LSE 保持
`[B,Hq,Sq]`；`seqused_q` 之外的存储行没有数值保证。有效 Q 行没有可见 KV 时，
O=0、LSE=-inf。`max_seqlen_q/k` 只限制调度范围，不能改变物理存储跨度；
调用方保证有效长度不超过存储容量和给定上界，入口不回读元数据。

回调使用当前序列的有效长度和局部坐标；相同容量下更新有效长度或替换元数据地址
可复用编译对象。显式建块时传入相同的 `seqused_q/k`，`seqlen_q/k` 使用 BSND
存储容量；各 batch 长度不同时按真实 batch 数建表。有效长度或辅助数据改变可见性后，
须更新或重新生成块表。没有有效长度参数的 BSND 调用复用普通定长入口的执行路径。

### 分页 KV

`flash_attn_varlen_func(..., page_table=..., seqused_k=...)` 直接读取
`[page,token,Hkv,Dqk或Dv]` 的 K/V；页长支持128、256、512、1024。
Q 可使用 TND（提供 `cu_seqlens_q`）或 BSND（不传 `cu_seqlens_q`）。
`page_table` 是同设备连续 int32 `[B,max_pages]`，`seqused_k` 是 int32 `[B]`。
分页输入不能同时传 `cu_seqlens_k`，不在 host 重排或拼接 KV。

```python
# q_packed 与 cu_q 沿用上例；两个序列共享物理页1，页表无需排序。
k_pages = torch.randn(4, 128, Hkv, D, dtype=q.dtype, device=q.device)
v_pages = torch.randn_like(k_pages)
pages = torch.tensor([[2, 1, 0], [1, 3, 0]], dtype=torch.int32, device=q.device)
used_k = torch.tensor([128, 257], dtype=torch.int32, device=q.device)
out, lse = flash_attn_varlen_func(
    q_packed,
    k_pages,
    v_pages,
    cu_seqlens_q=cu_q,
    page_table=pages,
    seqused_k=used_k,
    max_seqlen_q=129,
    max_seqlen_k=257,
    causal=True,
    return_lse=True,
)
```

callback 接收序列内的逻辑 q/kv 坐标，不接收物理页号。现有 score/mask、
auxiliary 和显式块表保持可用；建块时提供相同的 `cu_seqlens_q`、`seqused_q/k`，
不传 `cu_seqlens_k`，classifier 不读取 K/V 或页表内容。
页索引、长度及上界内容由调用方保证合法。页表原地修改和地址替换会在下次调用生效，
不影响编译 key；长度或 auxiliary 改变可见性时，须重新生成或更新块表。

### Split-KV decode

分页 TND 且 `max_seqlen_q <= 16` 时，可用 `num_splits>1` 显式拆分 KV 工作。
分割数不超过设备 Cube 核数；`num_splits=1`（默认）沿用单 kernel。
`num_splits=0` 自动判断：只有低占用、至少65536个有效候选 KV token 的非稀疏短 Q
才拆分；窄窗口按可访问范围计算工作量，块表自动模式保持1。

各分段使用相同的 modifier、辅助数据和块表遍历，另一个 kernel 归并 FP32 部分结果。
计时须包含分段和归并，不能只看分段 attention。
短 KV 或较高占用时，额外分配、提交及归并可能抵消并行收益。
O-only 仍需内部 max/sum，但不分配最终 LSE；确定性仅要求相同分割方案的重复运行，
不同分割数允许正常舍入差异。

### 常用选项

- `softmax_scale`：基础分数缩放，默认 `1 / sqrt(Dqk)`。
- `softcap`：默认 0（关闭）；正值 `c` 将缩放后分数变为 `c * tanh(score / c)`，
  然后再应用 mask。内部使用 SIMD score helper，不可与显式 `score_mod` 同用；
  full 稀疏块仍执行 softcap。
- `causal=True`：右对齐因果掩码，保留 `kv <= q + Sk - Sq`。
- `window_size=(left, right)`：保留 `[q + Sk - Sq - left, q + Sk - Sq + right]`；
  每侧的 `None` 或负值均表示该侧无界；`causal=True` 将右边界设为 0。
- `score_mod` / `mask_mod`：自定义分数与掩码回调。自定义 `mask_mod` 覆盖上述内置掩码。
- `aux_tensors` / `aux_scalars`：传给回调的辅助张量与运行时标量。
- `block_sparse_tensors`：显式块元数据，见下文。
- `pack_gqa`：None 自动选择 Q-head 合并，False 明确关闭；True 尚不支持。
- `deterministic`：接受 True/False，不改变推理前向的计算或编译对象；该选项对应
  反向算法选择，本包仍不支持 backward。同一运行环境、编译产物、输入及块表顺序下，
  确定性回调的重复调用应得到逐位一致的结果；不同块表顺序或计算路径之间不要求逐位一致。

## 回调与 SIMD / SIMT

计算顺序为：基础缩放 → `score_mod` → `mask_mod` → softmax。
mask 返回是否保留当前位置；score 返回修改后的分数。在上面的 mask 基础上，
可叠加 `score_mod`，并通过 `aux_scalars` 传入运行时缩放系数：

```python
import catlass.tla as tla
from flash_attn_npu_dsl import simd


def scale_score(score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
    return score * aux_scalars[0]


out, lse = flash_attn_func(
    q,
    k,
    v,
    mask_mod=causal_mask,
    score_mod=scale_score,
    aux_scalars=(0.75,),
    return_lse=True,
)
```

回调可以是普通 Python 函数或 `@tla.jit` helper，函数体使用 DSL 支持的运算。
`b`、`h` 分别为 batch 和 Q-head 下标，
`q_idx`、`kv_idx` 为序列内下标，`seqlen_info` 提供 `seqlen_q` / `seqlen_k`。
mask 按位置传参；score 的前三项按位置传入，其余按参数名传入。

非空 `aux_scalars` 为 mask 增加第七个位置参数，为 score 增加 `aux_scalars` 关键字；
为 `None` 或空容器时不传该参数，回调可用默认值兼容。`aux_tensors` 为只读、连续、
同设备张量的 tuple/list；未提供时 mask 收到 `None`，score 收到 `()`。

未标记的回调使用标量 SIMT。显式添加 `@simd` 后，回调处理 64-lane 向量：
坐标为 Int32，score 为 Float32，mask 返回谓词向量。等价的 SIMD 回调为：

```python
@simd
def vector_causal_mask(b, h, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars=None):
    return tla.cmp(kv_idx, q_idx + seqlen_info.seqlen_k - seqlen_info.seqlen_q, "le")


@simd
def vector_scale_score(score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
    return score * aux_scalars[0]
```

mask 和 score 的模式独立选择，四种组合均受支持：

| 示例选项 | mask 模式 | score 模式 |
| --- | --- | --- |
| 无 | SIMT | SIMT |
| `--simd-mask` | SIMD | SIMT |
| `--simd-score` | SIMT | SIMD |
| `--simd` 或 `--simd-mask --simd-score` | SIMD | SIMD |

直接调用时，按需将 `mask_mod`、`score_mod` 分别替换为对应的向量版本。
`@simd` 只标记原函数，不自动改写函数体，可置于 `@tla.jit` 前后。
两个 score 函数独立定义：未标记的版本使用 SIMT，带 `@simd` 的版本使用 SIMD。
SIMD 组合条件使用 `tla.bitwise_and/or`、`tla.where`，不用 Python `and/or`；
辅助张量访问仍受 DSL 索引能力限制，不保证任意 SIMD gather 可用。

每个 SIMD lane 对应一个 `(b, h, q, kv)` 分数位置，回调按位置逐元素计算。
内部可以沿 KV 或 Q/head 组织 lane；不要假定 `q`、`h` 在所有 lane 中相同，
也不要依赖相邻 lane 必须对应连续 KV。回调的数学定义、参数及输出布局不变。

## 显式两步块稀疏

仅传 `mask_mod` 仍遍历所有 KV 块；跳过空块需要先分类，再将元数据交给前向：

```python
from flash_attn_npu_dsl import compute_block_sparsity

blocks = compute_block_sparsity(
    128,
    128,
    1,  # causal_mask 与 batch 无关，共享一份分类
    1,  # causal_mask 与 head 无关
    Sq,
    Sk,
    causal_mask,
    None,
    q.device,
    aux_scalars=(0.75,),
)
out, lse = flash_attn_func(
    q,
    k,
    v,
    mask_mod=causal_mask,
    score_mod=scale_score,
    aux_scalars=(0.75,),
    block_sparse_tensors=blocks,
    return_lse=True,
)
```

分类在 NPU 上只按 mask 的 SIMT/SIMD 模式精确执行，与 score 模式无关，仅支持 `(128, 128)` 块：
空块跳过，partial 块执行 mask，full 块跳过自定义 mask，但仍执行 score 修改和尾块边界检查。
分类与前向必须使用相同的 mask 及其输入；mask、序列长度或所依赖的辅助数据变化后需重新分类。

### 按 mask 的实际依赖选择建块维度

上面的右对齐 causal mask 只依赖 Q/KV 坐标和共同的序列长度，因此可以用
`batch_size=1, num_heads=1` 建表，再传给实际为 `B × Hq` 的 attention。
前向通过广播步长直接读取共享表，不需要 `expand().contiguous()` 或复制元数据。
这会减少重复分类及块表存储，不改变 attention 的实际 batch/head 数量。

| mask 的依赖 | 建块的 `batch_size` | 建块的 `num_heads` |
| --- | ---: | ---: |
| 所有 batch/head 共享 | `1` | `1` |
| 随 batch 变化、head 共享 | `B` | `1` |
| 随 head 变化、batch 共享 | `1` | `Hq` |
| batch/head 均可能变化 | `B` | `Hq` |

共享由调用方显式选择，库不会推断 callback 是否依赖 `b`、`h` 或辅助数据。
例如按 head 读取不同阈值的 mask 应按 `Hq` 建表，不能复用 head 0 的分类。
`score_mod` 可以随 batch/head 变化；只要 mask 的可见性相同，仍可共享块表。
若 mask 及其输入不变，块表还可跨多次前向复用；需要重新建块时，也可以继续使用上述广播维度。

### 自行提供块表

也可自行构造 `BlockSparseTensorsTorch`，建议显式设置 `block_size=(128, 128)`：

- `mask_block_cnt` / `mask_block_idx` 为必需的 partial 列表；
  `full_block_cnt` / `full_block_idx` 必须同时提供或同时为 `None`。
- 四个张量均为同设备、连续的 NPU Int32 张量。计数形状为
  `(B或1, Hq或1, ceil(Sq/128))`；索引形状为
  `(B或1, Hq或1, ceil(Sq/128), capacity)`，每个张量的 B/H 广播独立选择。
  partial/full 容量可以不同，均可取 `0` 到 `ceil(Sk/128)`，无需填充到 KV 块数。
- 调用方保证各计数位于 `[0, capacity]`，有效索引位于 `[0, ceil(Sk/128))`，
  每行 partial/full 有效索引在各自列表内无重复、两表间无交叉，且分类与当前 mask 一致。
  零容量对应的计数必须为零；计数以外的槽位忽略。

前向只检查类型、形状、dtype、设备、连续性、容量和可选字段配对，不读取或校验元数据内容。
内核直接读取这四个公开张量，使用各自的广播步长和实际容量寻址，不展开、拼接、排序或生成
中间调度表，也不做设备内容同步。每行按 partial 列表原顺序处理，再按 full 列表原顺序处理，
两阶段共享连续的流水编号；无须对输入索引排序。full 为 `None` 或任一列表容量为零时，
编译期移除相应索引读取，不创建占位索引张量。重新调用时会绑定当前张量和内容；
调用方必须保证张量及其内容在该次异步执行完成前有效且不被修改。

### TND 块表

TND 使用相同的两步接口；向建块函数传入与 attention 相同的累计长度和有效长度：

```python
blocks = compute_block_sparsity(
    128,
    128,
    2,
    1,
    129,
    257,
    causal_mask,
    None,
    q_packed.device,
    cu_seqlens_q=cu_q,
    cu_seqlens_k=cu_k,
)
out, lse = flash_attn_varlen_func(
    q_packed,
    k_packed,
    v_packed,
    cu_seqlens_q=cu_q,
    cu_seqlens_k=cu_k,
    max_seqlen_q=129,
    max_seqlen_k=257,
    mask_mod=causal_mask,
    block_sparse_tensors=blocks,
    return_lse=True,
)
```

counts 为 `[H,total_m_blocks]`，indices 为 `[H,total_indices]`，每个张量可独立使用
`H=1` 广播。`cu_total_m_blocks[b]` 定位该序列的 count 段，
`cu_block_idx_offsets[b] + local_q_block * ceil(effective_k/128)` 定位 index 行。
与 BSND 一样，只读取 count 指定的有效前缀，按 partial→full、列表原顺序消费，
不排序或物化另一套调度表。显式表须提供 `block_size=(128,128)`。

建块入口可通过一个前缀计算 kernel 生成缺少的累计块偏移，以存储段预留容量；
只合并回读一次终值用于分配。已提供两个偏移时不再计算前缀。
有效长度小于存储长度时允许段间留空隙。自行提供累计块偏移时，调用方保证容量、
互不重叠的段和有效索引合法。省略 index 累计偏移时使用
`cu_total_m_blocks[b] * ceil(effective_k/128)`，仅适用于该公式能正确定位各段的表。
不同 batch 长度可能不同，因此 TND 不支持跨 batch 广播。

固定表 attention 不扫描、回读或转换块表。改变有效长度、mask 数据或辅助数据后，
必须先重建或更新表，再调用 attention；同形状表的原地修改和地址替换不要求重新编译。

## torch.compile

使用 PyTorch 2.7 与匹配的 TorchNPU，可将两个 attention 入口作为外部算子接入
Inductor 图。DSL 仍负责内核编译；这不意味着 Inductor 会重写 DSL 内部计算。

```python
compiled_attention = torch.compile(
    lambda q, k, v: flash_attn_func(q, k, v, causal=True, return_lse=True),
    backend="inductor",
    fullgraph=True,
)
out, lse = compiled_attention(q, k, v)
```

固定的 callback 可放在函数闭包中，逻辑在复用期间保持不变。辅助 Tensor、序列长度、
页表及预先准备的块表应作为编译函数的实参传入；每次调用绑定当前值，不捕获旧地址。
图内 `aux_scalars` 使用 Python bool/int/float，沿用 Dynamo 的标量守卫；eager 调用仍可
使用 TLA Numeric。需要容量回读的 `compute_block_sparsity` 在图外执行。
本接入不包含 backward、自动求导、NPUGraph 或跨进程导出已捕获图。

## 代码模块与执行流程

| 文件 | 职责 |
| --- | --- |
| [__init__.py](__init__.py) | 导出 `flash_attn_func`、`flash_attn_varlen_func`、`compute_block_sparsity`、`BlockSparseTensorsTorch`、`simd`。 |
| [example.py](example.py) | 可运行入口：创建输入和回调，按需建块，调用前向并检查 O/LSE。 |
| [interface.py](interface.py) | 校验 Q/K/V、辅助输入和参数，分配输出、转换运行时参数，管理编译缓存并编译、复用及启动内核。 |
| [flash_fwd.py](flash_fwd.py) | 前向主内核：调度 Q/KV 块，组织 QK、回调、softmax、PV 的流水、搬运与同步。 |
| [modifiers.py](modifiers.py) | 定义 SIMD/SIMT 回调契约，绑定参数，执行分数修改与掩码。 |
| [softmax.py](softmax.py) | 在线 softmax、跨块输出重缩放与累加、最终归一化和 LSE，处理空行。 |
| [dn_softmax.py](dn_softmax.py) | 沿 Q/head 组织 SIMD lane 的回调和在线 softmax，处理 KV 尾块与空行。 |
| [dn_full.py](dn_full.py) | 无回调、无边界裁剪的满 tile softmax，直接读取原始 score，省去修改后的 score 写回。 |
| [block_sparsity.py](block_sparsity.py) | 定义公开块元数据，只校验元数据规格并记录四个张量各自的广播步长和容量。 |
| [compute_block_sparsity.py](compute_block_sparsity.py) | 根据 mask 在设备上分类空块、partial 块和 full 块，返回公开块元数据。 |
| [seqlen_info.py](seqlen_info.py) | 构造 attention 与 classifier 共用的序列信息：有效长度、累计存储偏移和块偏移；检查元数据结构。 |
| [split_kv.py](split_kv.py) | 按 max/sum 稳定归并分段 FP32 输出；只在显式或自动拆分时启动。 |
| [torch_compile.py](torch_compile.py) | 私有 custom-op 与 fake 输出规格；静态 callback 和当前 Tensor 输入分开传递。 |

1. 可选建块：`compute_block_sparsity` 调用 mask，产生 partial/full 元数据。
2. 调用前向：`interface.py` 校验输入规格，直接绑定公开块张量，按编译缓存编译或复用后启动内核。
3. 内核按 Q 块遍历有效 KV 块：QK → 基础缩放 → score → mask → 在线 softmax → PV → 输出校正；
   所有 KV 块完成后归一化并写回 O 和可选 LSE。

Cube 负责 QK/PV 矩阵乘；AIV 上的回调按声明使用 SIMT 或 SIMD，softmax 和输出处理使用 SIMD。
`process_score_tile_simt/simd` 负责一轮 score tile 的读取、缩放、可选回调及边界处理；
`call_score_mod` 和 `call_mask_mod` 分别绑定并调用用户函数。
同模式回调保留融合遍历；混合模式仍按 score→mask 执行，但需要额外的 UB 遍历与同步，
并非零开销切换。各路径保留逐线程/逐向量的边界处理。
计算 helper 内联到前向内核中；显式建块另行启动 classifier，Split-KV 另行启动归并内核。

进程内缓存只保存编译结果，不保存启动参数、块内容或输出。无回调、`aux_tensors=None`、
无辅助标量且未提供块表时，
BSND 的 dense 和内置 causal/window 使用动态 Q/K 长度：固定 batch、head、D、dtype、窗口配置及调度策略，
不同正序列长度可复用同一编译对象。输出及 LSE 每次按当前长度分配，空行和尾块按当前长度处理。
O-only 与 O+LSE 使用不同变体；其他参数组合继续按完整张量规格特化。

Q 长度为 1 时，右对齐中心位于最后一个 KV 位置，内置右侧窗口限制被消除。
因此无左侧窗口的单行 causal 可复用 dense 变体；左侧窗口和用户回调仍按原语义执行。
ND 路径的内置窗口在整块可见且没有 score 修改时，直接由 softmax 完成缩放与 KV 尾部保护，
不再额外改写 score；边界块仍执行逐元素窗口判断。

对 Q 长度小于所选 Q tile（常规128行、宽 D 64行）的调用，内核可将共用同一 KV head 的 Q heads 合并，
以减少重复读取 KV。MQA 直接使用连续数据；一般 GQA 在设备上分段读取 Q、写回 O，
不在 host 上转置或复制。传入块表时，要求所有已提供元数据张量的 head 维均为 1；
合并后的计算块直接读取原来的单行 Q 块表，不复制元数据，仍允许按 batch 使用不同的列表。
仅当任务数减少且不降低可占用核心数时选择此策略；其他情况
保持逐 head 调度。用户 callback 仍接收原始的 head、Q 坐标和序列长度，O 与 LSE 的布局不变。
涉及 SIMT 回调时，合并进一步限定为 Q 长度不超过 32，避免逐行坐标处理抵消调度收益。
内置 causal/window 沿用上述合并条件，其 KV 块范围和逐元素判断均使用合并前的 Q 坐标。
合并后的 SIMT 线程按逻辑 Q/head 坐标递增，不要求一个任务内的 head 保持一致。
两种调度策略使用独立编译变体，长度变化导致策略切换时会匹配对应变体。

score 布局由入口在编译前选择。没有 SIMT 回调或显式 auxiliary，且满足以下任一条件时，
采用 DN：合并后的 Q 行数为 128 的整数倍、KV 长度至少 256；或没有合并 head，
Q/KV 长度均至少 512。后者允许长序列带 Q 尾块，避免因尾块而让整段计算退回 ND。
DN 直接计算 `K @ Qᵀ`，让 SIMD lane 沿 Q/head 排列。合并内置窗口保留 ND，以避免该组合的开销；
其他不满足 DN 条件的调用也保持 ND 的 `Q @ Kᵀ` 与 KV-lane 路径。
无 score 修改的 SIMT 稀疏调用也可复用 DN。TND 额外根据可知的平均 Q 长度
约束此选择；仅有最大长度上界不足以证明利用率，指定 `seqused_q` 时不作这一推断。
DN 中，Q/KV 均对齐 128、没有 mask/score/内置窗口且缩放为正的调用使用满 tile 计算体。
其他调用按当前块判断：有效位置全部可见、没有 score 修改且缩放为正时，
也可复用满 tile 计算体；可见性由 full 块标记或内置窗口边界确定。
若物理 tile 有 Q/KV 尾部，先将无效位置置为 `-inf`，再执行带空行保护的同一计算体。
部分可见块仍执行逐元素判断；softcap 属于 score 修改，不会被此路径跳过。
TND 的编译期上界不用于证明满块；判断使用当前任务的实际有效行列与可见性。
这些选择不增加设备 kernel，也不改变块表顺序与 QK/PV 流水；不同选择使用独立编译变体，
同一变体内仍可复用动态长度。用户不需要指定执行后端。

D≤64 的 PV/O 缓冲每行使用 64 列跨度，与一个 FP32 向量匹配；较大 D 保留 128 列。
小 D 的最后一个有效 KV 块在寄存器中合并输出更新与归一化，避免再读写一次 O。
BSND/TND、ND/DN 共享这部分输出处理，不依赖用户选择内置规则或回调。

动态长度路径仅在编译时构造 DSL Tensor；缓存命中时直接从当前 Torch Tensor 绑定地址和
动态描述符，不重复执行 DLPack 转换。它不缓存数据或省略输入规格检查；跨 stream 的生产者
依赖及缓冲生命周期仍由调用方管理，kernel 提交到当前设备的 current stream。
张量静态规格、回调或编译期配置变化时重新匹配；
块容量、四个张量各自的 B/H 广播方式、full 可选状态和零容量状态均参与编译特化。
辅助张量、块张量的内容/地址和运行时标量值变化不触发重新编译。复用期间回调代码及捕获的静态值应保持不变，
修改逻辑时传入新的回调函数；块元数据仍需按上面的规则单独更新。

## 基本验证

在仓库根目录确认运行时依赖，再运行接口与前端 lowering 测试：

```bash
python -c "import catlass.tla, torch, flash_attn_npu_dsl" &&
python -m pytest --confcutdir=tests/dsl -q tests/dsl
```

完成依赖检查后，指定设备可同时运行真机用例：

```bash
FLASH_ATTN_DSL_TEST_DEVICE=0 python -m pytest --confcutdir=tests/dsl -q tests/dsl
```

检查已构建的本仓 wheel 时，为 pytest 命令设置
`FLASH_ATTN_DSL_TEST_WHEEL=/path/to/flash_attn_npu.whl`；未设置时跳过分发检查。
