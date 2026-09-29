# CATLASS DSL Flex Attention

本包聚焦通过 `score_mod` 和 `mask_mod` 定制注意力前向：用户提供回调，
修改 softmax 前的分数或定义可见位置，无需为每种注意力规则重写完整 kernel。

- `score_mod`：对缩放后的 QK 分数施加自定义变换。
- `mask_mod`：判断每个 Q/KV 位置是否可见，可表达因果、滑窗等规则。
- 回调支持普通 Python 函数和 `@tla.jit` helper，按声明使用 SIMT 或显式 SIMD，
  内联到 attention 内核；需要跳过空块时，可显式建块并传入稀疏元数据。

当前实现面向 Ascend 950 的定长推理前向，支持 BSND、FP16/BF16、D8/16/32/64/80/96/128 和 GQA。
通过独立的 `flash_attn_npu_dsl` 包使用，不替换现有 V2/V3/V4 的 C++ 后端。

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
Q 为 `(B, Sq, Hq, D)`，K/V 为 `(B, Sk, Hkv, D)`；所有维度为正，
`D` 支持 8/16/32/64/80/96/128，`Hq` 必须是 `Hkv` 的整数倍（GQA）。
仅支持推理前向，不支持梯度、packed/变长序列或 paged KV；未实现的选项必须保持默认值。

`flash_attn_func` 始终返回 `(out, lse)`。O 与 Q 的形状和 dtype 相同；
`return_lse=True` 返回 FP32 自然对数 LSE，形状为 `(B, Hq, Sq)`，否则 LSE 为 `None`。
完全被 mask 的行返回精确零 O 和 `-inf` LSE。

- `softmax_scale`：基础分数缩放，默认 `1 / sqrt(D)`。
- `softcap`：默认 0（关闭）；正值 `c` 将缩放后分数变为 `c * tanh(score / c)`，
  然后再应用 mask。内部使用 SIMD score helper，不可与显式 `score_mod` 同用；
  full 稀疏块仍执行 softcap。
- `causal=True`：右对齐因果掩码，保留 `kv <= q + Sk - Sq`。
- `window_size=(left, right)`：保留 `[q + Sk - Sq - left, q + Sk - Sq + right]`；
  每侧的 `None` 或负值均表示该侧无界；`causal=True` 将右边界设为 0。
- `score_mod` / `mask_mod`：自定义分数与掩码回调。自定义 `mask_mod` 覆盖上述内置掩码。
- `aux_tensors` / `aux_scalars`：传给回调的辅助张量与运行时标量。
- `block_sparse_tensors`：显式块元数据，见下文。

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

## 显式两步块稀疏

仅传 `mask_mod` 仍遍历所有 KV 块；跳过空块需要先分类，再将元数据交给前向：

```python
from flash_attn_npu_dsl import compute_block_sparsity

blocks = compute_block_sparsity(
    128,
    128,
    B,
    Hq,
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

## 代码模块与执行流程

| 文件 | 职责 |
| --- | --- |
| [__init__.py](__init__.py) | 导出 `flash_attn_func`、`compute_block_sparsity`、`BlockSparseTensorsTorch`、`simd`。 |
| [example.py](example.py) | 可运行入口：创建输入和回调，按需建块，调用前向并检查 O/LSE。 |
| [interface.py](interface.py) | 校验 Q/K/V、辅助输入和参数，分配输出、转换运行时参数，管理编译缓存并编译、复用及启动内核。 |
| [flash_fwd.py](flash_fwd.py) | 前向主内核：调度 Q/KV 块，组织 QK、回调、softmax、PV 的流水、搬运与同步。 |
| [modifiers.py](modifiers.py) | 定义 SIMD/SIMT 回调契约，绑定参数，执行分数修改与掩码。 |
| [softmax.py](softmax.py) | 在线 softmax、跨块输出重缩放与累加、最终归一化和 LSE，处理空行。 |
| [block_sparsity.py](block_sparsity.py) | 定义公开块元数据，只校验元数据规格并记录四个张量各自的广播步长和容量。 |
| [compute_block_sparsity.py](compute_block_sparsity.py) | 根据 mask 在设备上分类空块、partial 块和 full 块，返回公开块元数据。 |
| [seqlen_info.py](seqlen_info.py) | 构造回调共享的定长序列信息，包含长度、零偏移和 KV 块数。 |

1. 可选建块：`compute_block_sparsity` 调用 mask，产生 partial/full 元数据。
2. 调用前向：`interface.py` 校验输入规格，直接绑定公开块张量，按编译缓存编译或复用后启动内核。
3. 内核按 Q 块遍历有效 KV 块：QK → 基础缩放 → score → mask → 在线 softmax → PV → 输出校正；
   所有 KV 块完成后归一化并写回 O 和可选 LSE。

Cube 负责 QK/PV 矩阵乘；AIV 上的回调按声明使用 SIMT 或 SIMD，softmax 和输出处理使用 SIMD。
`process_score_tile_simt/simd` 负责一轮 score tile 的读取、缩放、可选回调及边界处理；
`call_score_mod` 和 `call_mask_mod` 分别绑定并调用用户函数。
同模式回调保留融合遍历；混合模式仍按 score→mask 执行，但需要额外的 UB 遍历与同步，
并非零开销切换。各路径保留逐线程/逐向量的边界处理。
这些 helper 内联到前向内核中；只有显式建块会另行启动 classifier 内核。

进程内缓存只保存编译结果，不保存启动参数、块内容或输出。无回调、`aux_tensors=None`、
无辅助标量且未提供块表时，
dense 和内置 causal/window 使用动态 Q/K 长度：固定 batch、head、D、dtype 和窗口配置，
不同正序列长度可复用同一编译对象。输出及 LSE 每次按当前长度分配，空行和尾块按当前长度处理。
O-only 与 O+LSE 使用不同变体；其他参数组合继续按完整张量规格特化。
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
