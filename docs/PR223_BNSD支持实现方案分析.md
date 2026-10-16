# PR #223 "support BNSD" 实现方案分析

> 分析对象：[MinghuasLab/flash-attention-npu#223](https://github.com/MinghuasLab/flash-attention-npu/pull/223)
> 分析视角：NPU 算子开发
> 分析日期：2026-10-14

---

## 1. PR 概述

| 项目 | 内容 |
|---|---|
| 标题 | support BNSD |
| 作者 | jianshuqiang-cpu |
| 状态 | open（创建于 2026-09-09） |
| 分支 | `support_BNSD`（head sha `5d8d020`）→ `MinghuasLab:main` |
| 变更规模 | 51 个文件，覆盖 Python 接口层、C++ Host 层、Kernel 层、测试层 |
| 核心目标 | **Kernel 原生支持 BNSD 输入布局（即"方案B"）：零拷贝消费 `BNSD.transpose(1, 2)` 产生的非连续 BSND 视图** |

### 1.1 背景与动机

- 昇腾生态常用 **BNSD** `(B, N, S, D)` 布局，而本仓库 kernel 原生消费 **BSND** `(B, S, N, D)`；
- 与 NV Dao-AILab/flash-attention 的接口对齐要求兼容两种布局；
- 此前的"方案A"（接口层 `transpose(1,2).contiguous()` 物化）需要 **2~4 次全量搬运**，长序列下性能劣化可能超过 20%；
- 本 PR 落地"方案B"：`BNSD.transpose(1, 2)` 得到的视图 stride 为 `(H*S*D, D, S*D, 1)`，kernel 直接按此 stride 寻址，**不物化 contiguous 副本，零额外拷贝**。

### 1.2 核心思想一句话

> 把 Q/K/V 的元素级 stride 三元组 `{batchStride, seqStride, headStride}` 作为 **kernel launch 参数**逐次传入，kernel 内所有 batch/seq/head 偏移全部参数化，从而支持任意"最后一维连续"的 4D 视图直接进 kernel。

---

## 2. 总体设计

### 2.1 三层改造架构

```
┌─────────────────────────────────────────────────────────────┐
│ Python 接口层（5 个 interface 文件）                          │
│  · _is_bnsd_transpose_view：识别 BNSD.transpose(1,2) 视图     │
│  · maybe_contiguous_last_dim：只保证最后一维连续，保留外层 stride │
│  · 反向：maybe_contiguous_bwd(_output)，kernel 缓冲 + copyback  │
├─────────────────────────────────────────────────────────────┤
│ C++ Host 层（3 个 910 flash_api.cpp + 2 个 950 mha_fwd.cpp）  │
│  · GetQkvElementStrides：校验 + 提取 stride 三元组            │
│  · IsSupportedBsndBackwardLayout：反向布局白名单             │
│  · SetBsndStrides：反向路径覆写 tiling 中的 stride            │
│  · data_ptr() 修复：支持带 storage_offset 的视图              │
├─────────────────────────────────────────────────────────────┤
│ Kernel 层（910 ×3 套 + 950 ×2 套）                            │
│  · 前向：qBatchStride/qSeqStride/qHeadStride 等参数化寻址     │
│  · 反向：tiling 携带 8 组 stride，epilogue 散射写出            │
│  · 偏移量 U32 → U64 防溢出；分页 KV 引入 blockStride           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 关键设计决策：stride 与 scheduler_metadata 解耦

`kernel_common.hpp` 中的注释明确表达了这一决策：

```cpp
// Element strides are launch-time address metadata, deliberately kept
// outside scheduler_metadata so the same scheduler buffer can be
// reused by equal-shaped tensors with different views.
```

- **scheduler_metadata（tiling）**：只依赖形状（B/S/H/D），与具体视图无关；
- **stride**：属于"本次 launch 的地址元数据"，随每次 kernel 启动单独传参；
- **收益**：同一份缓存的 scheduler_metadata 可被**同形状、不同 stride 视图**复用（测试 `test_paged_kv_native_strides_reuse_scheduler_metadata` 专门验证这一点），保住了 CUDA Graph / 异步场景下 metadata 缓存复用的性能优势。

### 2.3 数据流对比

| | 方案A（接口层适配） | 方案B（本 PR） |
|---|---|---|
| Python 层 | `transpose(1,2).contiguous()` 物化 | 仅校验最后一维连续，视图原样下发 |
| 搬运次数 | 额外 2~4 次全量 D2D 拷贝 | 0 次额外拷贝 |
| Host 层 | 无感知（看到的总是连续张量） | 提取 stride 三元组并传参 |
| Kernel | 按 BSND 连续布局硬编码寻址 | batch/seq/head 偏移全部参数化 |
| scheduler_metadata | 正常 | 可被不同视图复用（解耦设计） |

---

## 3. Python 接口层实现

涉及文件：`flash_attn_npu/flash_attn_npu_interface.py`（+97/-31）、`flash_attn_npu_3/`（+70/-14）、`flash_attn_npu_3/..._950.py`（+19/-16）、`flash_attn_npu_4/`（+56/-12）、`flash_attn_npu_4/..._950.py`（+8/-6）。

### 3.1 新增辅助函数

四个 helper 在所有接口文件中保持一致：

```python
def _is_bnsd_transpose_view(x):
    """Return whether ``x`` is a BSND view made by BNSD.transpose(1, 2)."""
    if x is None or x.dim() != 4 or x.stride(-1) != 1:
        return False
    _, seqlen, nheads, headdim = x.shape
    return x.stride() == (
        nheads * seqlen * headdim,   # batch stride
        headdim,                      # seq stride  ← BNSD 特征
        seqlen * headdim,             # head stride
        1,
    )

def maybe_contiguous_last_dim(x):
    """Keep Q/K/V outer strides; only the vector dimension must be contiguous."""
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

def maybe_contiguous_bwd(x):
    """Keep native BSND and BNSD-transpose layouts zero-copy for FAG backward."""
    if x is None or x.is_contiguous() or _is_bnsd_transpose_view(x):
        return x
    return x.contiguous()

def maybe_contiguous_bwd_output(x):
    """Return an output accepted by the native strided backward fast path."""
    if x is None or x.is_contiguous() or _is_bnsd_transpose_view(x):
        return x, None
    return torch.empty_like(x, memory_format=torch.contiguous_format), x
```

要点：

1. **`_is_bnsd_transpose_view`** 通过 stride 特征值 `(H*S*D, D, S*D, 1)` 精确识别"标准 BNSD 转置视图"，而非泛化的非连续判断——保证只对 kernel 确实支持的布局走零拷贝快路径，其余布局仍回退到物化，**不会误放行任意 stride**；
2. **`maybe_contiguous_last_dim`** 替代了原来的全量 `maybe_contiguous`：只要最后一维（向量维）连续就放行，外层 stride 原样保留给 kernel。这是前向零拷贝的入口；
3. **`maybe_contiguous_bwd_output`** 返回 `(kernel缓冲, copyback目标)` 二元组：梯度先写入连续 kernel 缓冲，计算完成后 `dq_copyback.copy_(dq_kernel)` 回写到用户期望的布局，保证 autograd 语义下梯度布局与输入视图一致。

### 3.2 各路径的接入方式

- **前向**：`q, k, v = [maybe_contiguous_last_dim(x) for x in (q, k, v)]`（原为全量物化）；
- **反向（FAG）**：dq/dk/dv 用 `maybe_contiguous_bwd_output`；q/k/v/dout 用 `maybe_contiguous_bwd` 放行两种原生布局；
- **kvcache 路径**：`k_cache_kernel = maybe_contiguous_last_dim(k_cache)`；发生 appendKV 物化时通过 `k_cache.copy_(k_cache_kernel)` 回写，保证 cache 语义不变；
- **fake 函数（torch library 注册）**：`out = torch.empty(q.shape, dtype=..., device=..., layout=q.layout)` 替代 `torch.empty_like(q)`——`empty_like` 在某些非连续视图下 fake 求值会丢失 layout 信息，显式传 `q.layout` 修复；
- **950 接口差异**：`qv = _maybe_contiguous(qv)`（qv 组合张量仍要求全连续）；enableDN 判定增加 `q.is_contiguous() and k.is_contiguous() and v.is_contiguous()`——**非连续输入禁用 DN 路径**（950 的 DN 依赖稠密存储假设）；同时删除了 `k/v stride(-1)==1` 的前置 assert（校验下沉到 host 层 TORCH_CHECK）。

---

## 4. C++ Host 层实现

涉及文件：`csrc/ascend910/flash_attn_npu/flash_api.cpp`（+106/-15）、`flash_attn_npu_3/`（+112/-14）、`flash_attn_npu_4/`（+97/-15）、`csrc/ascend950/flash_attn_npu_3/mha_fwd.cpp`（+34/-5）、`flash_attn_npu_4/mha_fwd.cpp`（+34/-5）。

### 4.1 stride 提取与校验：GetQkvElementStrides

```cpp
struct QkvElementStrides {
    uint64_t batch;
    uint64_t seq;
    uint64_t head;
};

static QkvElementStrides GetQkvElementStrides(const at::Tensor &x, const char *name)
{
    TORCH_CHECK(x.dim() == 3 || x.dim() == 4, name,
                " must be a 3D TND or 4D BSND/paged tensor");
    TORCH_CHECK(x.stride(x.dim() - 1) == 1, name,
                " must be contiguous in the last dimension");
    for (int64_t i = 0; i < x.dim(); ++i) {
        TORCH_CHECK(x.stride(i) >= 0, name, " does not support negative strides");
    }
    TORCH_CHECK(static_cast<uint64_t>(x.stride(x.dim() - 2))
                    <= std::numeric_limits<uint32_t>::max(),
                name, " head stride exceeds the kernel DMA limit");
    const int64_t seqDim = x.dim() == 3 ? 0 : 1;
    TORCH_CHECK(static_cast<uint64_t>(x.stride(seqDim))
                    <= std::numeric_limits<uint32_t>::max(),
                name, " sequence stride exceeds the kernel DMA limit");
    if (x.dim() == 3) {   // TND varlen：batch 维不存在，batch stride 置 0
        return {0, static_cast<uint64_t>(x.stride(0)),
                static_cast<uint64_t>(x.stride(1))};
    }
    return {static_cast<uint64_t>(x.stride(0)),
            static_cast<uint64_t>(x.stride(1)),
            static_cast<uint64_t>(x.stride(2))};
}
```

设计细节：

- **TND（3D varlen）** 与 4D 统一走同一提取函数：3D 时 `batch=0`，kernel 侧用 seqlen 前缀和寻址，4D 时用 batch stride 寻址，二者在 kernel 内以 `INPUT_LAYOUT` 分支收敛；
- **DMA 限制前置校验**：head/seq stride 必须 ≤ `uint32_max`（昇腾 DMA 搬运单步跨度限制），超限直接报错而非静默错误；
- **拒绝负 stride**（翻转视图），明确报错边界。

### 4.2 反向布局白名单：IsSupportedBsndBackwardLayout

```cpp
static bool IsSupportedBsndBackwardLayout(const at::Tensor &tensor)
{
    if (tensor.dim() != 4 || tensor.stride(3) != 1) {
        return false;
    }
    if (tensor.is_contiguous()) {
        return true;
    }
    const int64_t seqlen  = tensor.size(1);
    const int64_t nheads  = tensor.size(2);
    const int64_t headdim = tensor.size(3);
    return tensor.stride(0) == nheads * seqlen * headdim &&
           tensor.stride(1) == headdim &&
           tensor.stride(2) == seqlen * headdim;
}
```

与 Python 侧 `_is_bnsd_transpose_view` 逻辑镜像：**C++ 层独立再校验一次**，防止绕过 Python 包装直接调用底层 op 时传入不支持的布局（防御性设计）。

### 4.3 关键 bugfix：`storage().data()` → `data_ptr()`

原代码使用 `q.storage().data()` 提取设备指针，返回的是 **storage 起始地址**。当张量是带 `storage_offset` 的视图（如 `transpose` 后再 `slice`）时，指针错误。本 PR 对 q/k/v/out 等 8 个张量统一改为 `q.data_ptr()`，这是**非连续视图能正确进 kernel 的前提**，也是独立于 BNSD 特性的通用正确性修复。

### 4.4 其他 Host 层改动

- **前向**：stride 三元组填入 `fwd_args.qBatchStride / qSeqStride / qHeadStride` 等（910 v2 含 kNew/vNew 共 15 个字段，_3/_4 为 9 个）；
- **反向**：`GetFAGTilingParam` 之后按布局覆写 tiling：

```cpp
if (!is_varlen_q) {
    SetBsndStrides(fagTilingData.qStrides, q);
    SetBsndStrides(fagTilingData.kStrides, k);
    // ... v/dout 同理
}
```

- **输出分配**：`torch::empty_like(q)` → `torch::empty(q.sizes(), q.options())` + `TORCH_CHECK(out.is_contiguous())`——输出始终为连续缓冲，非连续视图的梯度回写交给 Python 层 copyback；
- **varlen 反向（mha_varlen_bwd / mha_bwd varlen 分支）**：仍要求全部张量 contiguous，**反向 strided 支持仅限 padded batch 维场景**（边界明确，文档化在 TORCH_CHECK 中）。

---

## 5. Kernel 层实现（前向）

### 5.1 参数结构扩展：kernel_common.hpp

```cpp
// 910 三个版本一致；v2 另有 kNew/vNew 各 3 个字段（appendKV 场景），共 15 个
uint64_t qBatchStride;  uint64_t qSeqStride;  uint64_t qHeadStride;
uint64_t kBatchStride;  uint64_t kSeqStride;  uint64_t kHeadStride;
uint64_t vBatchStride;  uint64_t vSeqStride;  uint64_t vHeadStride;
```

配套修改：`fwd_dispatch.hpp` / `fwd_dispatch_impl.hpp`（全部版本）的 `FwdLaunchArgs` 增加 stride 字段，launch 宏追加实参，例如 950 v4：

```cpp
<<<a.block_dim, nullptr, a.stream>>>(
    a.q_device, a.k_device, a.v_device, a.mask_device,
    a.block_table_device, a.o_device, a.lse_device, a.q_seq_device,
    a.kv_seq_device, a.workspace_device, a.tiling_device,
    a.q_batch_stride, a.q_seq_stride, a.q_head_stride,
    a.k_batch_stride, a.k_seq_stride, a.k_head_stride,
    a.v_batch_stride, a.v_seq_stride, a.v_head_stride)
```

### 5.2 寻址参数化：mha_fwd_kvcache.cpp（910 v2 +80/-23、v3 +79/-22、v4 +45/-14）

核心偏移逻辑（以 Q 为例）：

```cpp
qBatchStride = params.qBatchStride;
qHeadStride = params.qHeadStride;
strideQ     = params.qSeqStride;

// batch 基址：TND 用 seqlen 前缀和，BSND 用 batch stride
uint64_t qBOffset = (INPUT_LAYOUT == FaiKenel::inputLayout::TND)
    ? static_cast<uint64_t>(prevQSeqlenSum) * strideQ
    : static_cast<uint64_t>(BIdx) * qBatchStride;

// head 基址：原来按 embed(=H*D) 硬编码推导，现在直接用 headStride
uint64_t gmOffsetQ = qBOffset
    + (qSBlockIdx * curQSBlockTile) * strideQ
    + qNStartIdx * qHeadStride;

uint64_t gmOffsetK = kBOffset + kvNIdx * kHeadStride;
uint64_t gmOffsetV = vBOffset + kvNIdx * vHeadStride;
```

原来 kernel 中 head 基址隐含 `n2Idx * headNum * headDim`（即假设 head 维外层连续）的推导，本 PR 全部替换为显式 `headStride` 参数；batch 基址同理。**这是零拷贝的核心：同一个 kernel，连续输入时 stride 恰好等于原硬编码值，行为完全不变；BNSD 视图时 stride 反映真实布局。**

### 5.3 matmul 加载改造：qk_matmul.hpp / pv_matmul.hpp（910 三版本）

`loadQGM` 签名扩展为接收双 stride，DataCopy 布局显式构造：

```cpp
// One ND matrix is a group of heads for one token. The two explicit
// strides let a BHSD.transpose(1, 2) view enter the kernel unchanged.
LayoutA layoutSingleANd(singleGroupHeads, embed, qHeadStride);
copyGmToL1A(l1ATensor, gA, layoutAInL1, layoutSingleANd,
    tokenNumPerGroup, qSeqStride, tokenNumPerGroup, BLOCK_SIZE, rowNumRound);
```

同时 `getKVOffset` 的偏移类型 **uint32 → uint64**（大 batch × 大 seq 下 `B*S*H*D` 容易超 32 位），分页版本改为：

```cpp
// 原来硬编码 blockId * blockSize * strideKV，现在 blockStride 参数化
kOffset = blockId * blockStrideKV + startOffset * strideKV;
```

`blockStrideKV` 使得**分页 cache 的物理块本身也可以是非连续视图**（块内 stride ≠ blockSize*stride）。

### 5.4 appendKV 的双 stride 语义（v2/v3）

```cpp
qkSeqStride = isAppendBlock ? kNewSeqStride : strideK;   // 读取按块选择 layout
```

- **读 kNew/vNew**：用 kNew/vNew 自身 stride（新写入部分可能是任意布局）；
- **写回 cache**：用 cache 自身 stride（`appendCacheStride / appendCacheBlockStride`）——读写两侧 stride 独立，避免"新数据布局"污染"cache 布局"。

---

## 6. Kernel 层实现（反向 FAG）

### 6.1 tiling 与 DBParams 扩展：fag_kernel_common.hpp

```cpp
struct FAGTensorStrides {
    int64_t batch;
    int64_t seq;
    int64_t head;
};
// FAGTilingData 新增 8 组：
FAGTensorStrides qStrides, kStrides, vStrides, doutStrides,
                 outStrides, dqStrides, dkStrides, dvStrides;

// DBParams（Double Buffer 流水参数）新增：
int64_t qTensorOffsetCv, kTensorOffsetCv, vTensorOffsetCv, doutTensorOffsetCv;
int64_t qSeqStride, kSeqStride, vSeqStride, doutSeqStride;
```

FAG 反向的 stride 走 **tiling 结构**而非独立 launch 参数——因为反向 kernel 的 DB 流水参数本来就随 tiling 一起拷贝，搭便车即可。

### 6.2 真实元素偏移计算：fag_kernel.cpp（910 _3/_4，各 +74/-30）

```cpp
int64_t qHeadIdx = dbParam.n2Idx * g + dbParam.gIdx;   // GQA 展开后的实际 head

dbParam.qTensorOffsetCv = dbParam.bIdx * qBatchStride
                       + dbParam.s1oIdx * s1CvInner * qSeqStride
                       + qHeadIdx * qHeadStride;
dbParam.kTensorOffsetCv = dbParam.bIdx * kBatchStride
                       + dbParam.s2oIdx * s2CvInner * kSeqStride
                       + dbParam.n2Idx * kHeadStride;
// v/dout 同理

dbParam.qSeqStride = qSeqStride;   // 替代原 s1Stride = n2 * g * d 的硬编码推导
```

全部 7 处 mmad（mm-dyv、mm-qk、dq、dk、dv 及确定性算法分支）统一改用这组偏移与 stride。GQA 场景下 q 的 head 索引需要 `n2Idx * g + gIdx` 展开，k/v 直接用 `n2Idx`——保持与原推导在连续布局下等价。

### 6.3 梯度散射写出：fag_epilogue_post.hpp（+92/-3）

dq 输出是写到 GM 的非连续区域，无法用整块 DataCopy，改为**按逻辑行号解码 → 分段 DataCopyPad 散射**：

```cpp
// strided 判定：stride 三元组非零且与连续布局推导值不一致
dqIsStrided = dqStrides.seq != 0 &&
    (seq != qHeadNum * qHeadDim ||
     head != qHeadDim ||
     batch != qSeqlen * qHeadNum * qHeadDim);

CATLASS_DEVICE void CopyOutBsnd(dst, src, logicalOffset, dataSize,
                                strides, seqlen, nheads, headdim)
{
    uint64_t flatRow = logicalOffset / headdim;
    uint64_t remainingRows = dataSize / headdim;
    while (remainingRows > 0) {
        int64_t  nIdx  = flatRow % nheads;              // head 索引
        uint64_t bsIdx = flatRow / nheads;
        int64_t  sIdx  = bsIdx % seqlen;                // seq 索引
        int64_t  bIdx  = bsIdx / seqlen;                // batch 索引
        uint64_t rows  = min(remainingRows, nheads - nIdx);  // 本段可连续写的行数
        int64_t dstOffset = bIdx * strides.batch
                          + sIdx * strides.seq
                          + nIdx * strides.head;
        DataCopyPad(dst[dstOffset], src[ubRow * headdim],
            DataCopyExtParams(rows, headdim * sizeof(ElementVecDtype), 0,
                (strides.head - headdim) * sizeof(ElementVecDtype), 0));
        // 推进 flatRow / ubRow / remainingRows ...
    }
}
```

巧妙之处：连续布局下 `(b, s, n)` 行在 GM 中按 `n` 最内层连续排列，散射算法按"head 段"切分——**同一段内行连续，段间跳到新的 stride 目标地址**，把随机散射退化为少量段级 DataCopyPad，而不是逐行搬运。

### 6.4 sfmg 分支：fag_epilogue_sfmg.hpp（+37/-10）

softmax 梯度路径中 dout/out 各自维护 batch/seq/head stride 与独立 srcOffset：

```cpp
doutNStride = (doutSeqStride - headdim) * sizeof(ElementVecDtype);
```

---

## 7. Ascend950 适配（_3 / _4 两套）

### 7.1 tla 布局显式化：fwd_kernel.cpp（各 +54/-20）

950 kernel 使用 tla（Tensor Level Abstraction）描述 GM 布局，原来依赖默认构造（隐含连续 stride），现在显式构造：

```cpp
strideQ = params.qSeqStride;
// qBOffset 分 TND / BSND 两个分支（与 910 相同模式）
lseOffset = curBatch * maxQSeqlen * qHeads;   // 修复：原来用除法推导，改为直接乘法

// Q tile 布局：行 stride 用真实 seqStride
tla::MakeLayout(tla::MakeShape(qBaseTile_, embed_),
                tla::MakeStride(strideQ, tla::Int<1>{}));

// loadQGM 增加 qHeadStride 参数
tla::MakeStride(qHeadStride, tla::Int<1>{});
```

### 7.2 分页读取的 pageOffset 模式：qk_matmul.hpp（+11/-5）/ pv_matmul.hpp

```cpp
uint64_t pageOffset = static_cast<uint64_t>(blockIdx) * kvBlockStride +
                      static_cast<uint64_t>(blockOffset) * kvSeqStride;
auto gBPageTensor = tla::MakeTensor(
    gBTensor.data()[pageOffset], gBTensor.layout(), Arch::PositionGM{});
auto gBTensorTlaTile = GetTile(gBPageTensor,
    tla::MakeCoord(0, 0), tla::MakeShape(embed, l0TileNAct));
```

原来分页读取把页内偏移折进 `GetTile` 的 coord，要求页内连续；现在先按 `blockStride + seqStride` 算出页基址，**在偏移处重建 tensor**，页内布局同样参数化。

### 7.3 enableDN 限制

950 的 DN（Dynamic N？GQA 折叠优化）路径要求稠密存储，接口层在输入非连续时禁用 DN，回退常规路径——功能正确优先于性能。

---

## 8. 关键设计决策与风险分析

### 8.1 设计亮点

| 决策 | 理由 |
|---|---|
| stride 放 launch 参数而非 scheduler_metadata | 保住 metadata 缓存复用（同形状不同视图共享 tiling），CUDA Graph 友好 |
| C++ 层独立白名单校验 | 防绕过 Python 包装直接调底层 op；与 Python 侧逻辑镜像 |
| `data_ptr()` 修复 | 通用正确性修复，带 storage_offset 的任何视图受益 |
| U32 → U64 偏移 | 大 batch/长序列下 `B*S*H*D` 超 32 位是必然场景 |
| appendKV 读写双 stride | 新数据布局与 cache 布局解耦，互不污染 |
| epilogue 段级散射 | 把 O(rows) 逐行写出降为 O(rows/headsPerSegment) 段级 DataCopyPad |
| DMA 上限前置 TORCH_CHECK | 超限报可读错误，避免静默数据损坏 |

### 8.2 限制与风险

1. **DMA 限制**：head/seq stride 必须 ≤ `uint32_max`，超大 H*D（如 H*D > 4G 元素）被显式拒绝；
2. **反向 varlen 不支持 strided**：`mha_varlen_bwd` 仍要求全连续，BNSD 反向仅支持 padded batch 维；
3. **950 禁 DN**：非连续输入在 950 上丢失 DN 优化路径，存在潜在性能回退窗口；
4. **`_is_bnsd_transpose_view` 只识别标准转置视图**：`BNSD.transpose(1,2)` 后再 `slice`/`narrow` 的复合视图会落到物化路径（正确但非零拷贝）；
5. **测试覆盖**：v2/v3/v4 均有 BNSD 用例，950 的 FAG 反向 strided 路径明确跳过（`pytest.skip`），说明该路径在 950 上尚未支持/验证。

---

## 9. 测试验证

### 9.1 新增测试（v2 +143/-16、v3 +187/-15、v4 +140/-13）

三个版本一致的测试模式：

1. **`test_flash_attn_func_accepts_noncontiguous_bshd`**
   BNSD `transpose(1,2)` 视图 vs 连续版，前向 + 反向 `rtol=0, atol=0` **逐位一致**——证明 strided 路径与连续路径数值等价；
2. **`test_paged_kv_native_strides_reuse_scheduler_metadata` / `test_paged_kv_accepts_native_outer_strides`**
   分页 KV cache 与 q 均为非连续视图，先断言 `keep_strides(x) is x`（零拷贝放行），再对比连续版输出逐位一致；
3. **`test_flash_attn_bnsd_transpose_backward_strides`**
   `assert grad.stride() == source.stride()`——**梯度保形**：BNSD 视图反传得到的梯度本身就是 BNSD 布局（copyback 生效），且数值对比连续版通过（rtol=2e-2）。

### 9.2 既有 kvcache 测试改造

`test_fa_kvcache_ops` 等用例的 BSND 构造全部改为"BNSD 存储 + `transpose(1,2)` 暴露 BSND 视图"：

```python
# Allocate every physical block referenced by block_tables. Build the
# storage as BNSD, then expose the paged-cache BSND view to the API.
key_cache = make_random_tensor(
    (num_blocks, kv_heads, block_size, head_size), ...
).transpose(1, 2)
```

即：**测试输入从"天生连续"改为"真实用户场景的转置视图"**，使整条 strided 链路被默认回归覆盖。同时删除了 `make_paged_kv_cache` helper（其分配逻辑无法构造非连续分页 cache）。

### 9.3 ATK 精度泛化用例（新增 2 个文件）

- `tests/test_flash_attn_npu_v2_accu_generalization.py`（+1516 行）
- `tests/test_flash_attn_npu_v2_accu_generalization_bigcase.py`（+528 行）

从 ATK（昇腾测试套件）移植的精度泛化测试，采用 CANN `cal_relative_diff` 判定标准（max ≤ 5% / avg ≤ 1.5% / rms ≤ 1.5%），含 `compare_cv_fused_double_benchmark` 双精度基准对比，覆盖大形状边界用例。

---

## 10. 文件变更清单

### Python 接口层（5 文件）

| 文件 | 变更 | 内容 |
|---|---|---|
| flash_attn_npu/flash_attn_npu_interface.py | +97/-31 | 4 个 helper + 各路径接入 + fake 修复 |
| flash_attn_npu_3/flash_attn_npu_interface.py | +70/-14 | 同上 |
| flash_attn_npu_3/flash_attn_npu_interface_950.py | +19/-16 | qv 全连续 + enableDN 条件 |
| flash_attn_npu_4/flash_attn_npu_interface.py | +56/-12 | 同 v3 |
| flash_attn_npu_4/flash_attn_npu_interface_950.py | +8/-6 | 同 950 v3 |

### C++ Host 层（5 文件）

| 文件 | 变更 | 内容 |
|---|---|---|
| csrc/ascend910/flash_attn_npu/flash_api.cpp | +106/-15 | GetQkvElementStrides、15 个 stride 字段、data_ptr 修复 |
| csrc/ascend910/flash_attn_npu_3/flash_api.cpp | +112/-14 | 同上（9 字段）+ 反向 tiling 覆写 |
| csrc/ascend910/flash_attn_npu_4/flash_api.cpp | +97/-15 | 同上 |
| csrc/ascend950/flash_attn_npu_3/mha_fwd.cpp | +34/-5 | stride 提取与传参 |
| csrc/ascend950/flash_attn_npu_4/mha_fwd.cpp | +34/-5 | 同上 |

### Kernel 层 — 参数与 dispatch（全部版本）

| 文件 | 内容 |
|---|---|
| kernel_common.hpp（910 ×3） | stride 三元组字段（v2 含 kNew/vNew 共 15 个） |
| ascend950 kernel_common.hpp（×2） | 同上（构造函数默认值 0） |
| fwd_dispatch.hpp / fwd_dispatch_impl.hpp（全部） | FwdLaunchArgs 扩展 + launch 宏追加实参 |
| fag_kernel_common.hpp（910） | FAGTensorStrides + tiling 8 组 + DBParams 扩展 |

### Kernel 层 — 前向实现

| 文件 | 变更 | 内容 |
|---|---|---|
| mha_fwd_kvcache.cpp（910 v2） | +80/-23 | 偏移参数化 + appendKV 双 stride |
| mha_fwd_kvcache.cpp（910 v3） | +79/-22 | 同上 |
| mha_fwd_kvcache.cpp（910 v4） | +45/-14 | 偏移参数化 |
| qk_matmul.hpp（910 ×3） | — | loadQGM 双 stride + U64 化 + blockStrideKV |
| pv_matmul.hpp（910 ×3） | — | 同上 + appendCacheStride 写回 |
| fwd_kernel.cpp（950 _3/_4） | 各 +54/-20 | tla 布局显式构造 + pageOffset |
| qk_matmul.hpp / pv_matmul.hpp（950 ×2） | +11/-5 等 | pageOffset 重建 tensor 模式 |

### Kernel 层 — 反向实现（910）

| 文件 | 变更 | 内容 |
|---|---|---|
| fag_kernel.cpp（_3/_4） | 各 +74/-30 | DBParams 真实元素偏移 + 7 处 mmad 改造 |
| fag_epilogue_post.hpp | +92/-3 | CopyOutBsnd 段级散射写出 |
| fag_epilogue_sfmg.hpp | +37/-10 | dout/out 独立 stride 与 srcOffset |

### 测试层（5 文件）

| 文件 | 变更 | 内容 |
|---|---|---|
| tests/test_flash_attn_npu_v2.py | +143/-16 | 3 个新测试 + kvcache BNSD 化 |
| tests/test_flash_attn_npu_v3.py | +187/-15 | 同上 |
| tests/test_flash_attn_npu_v4.py | +140/-13 | 同上（950 反向 skip） |
| tests/test_flash_attn_npu_v2_accu_generalization.py | +1516（新增） | ATK 精度泛化用例 |
| tests/test_flash_attn_npu_v2_accu_generalization_bigcase.py | +528（新增） | ATK 大形状用例 |

---

## 11. 与方案A的对比及工程意义

| 维度 | 方案A（接口层 transpose+contiguous） | 方案B（本 PR，kernel 原生） |
|---|---|---|
| 额外搬运 | 2~4 次全量 D2D | 0 |
| 长序列性能 | 劣化可能 >20% | 与连续输入等价（逐位一致） |
| 接口侵入性 | 新增 layout 参数，改动面小 | 无新参数，接口与 NV 天然兼容 |
| kernel 改动 | 无 | 全链路参数化，5 套代码同步 |
| scheduler_metadata | 正常 | 与 stride 解耦，缓存可复用 |
| 实现风险 | 低 | 中（偏移推导逐处改造，靠逐位一致测试兜底） |
| autograd 语义 | 梯度经 autograd 自动转回 | copyback 显式保证梯度保形 |

**工程意义**：

1. 本 PR 是项目记忆中"方案B 作为长期解决方案"判断的完整落地，一次性覆盖 910（v2/v3/v4）与 950（v3/v4）共 5 套代码副本；
2. `data_ptr()` 修复与 U64 偏移是随 PR 修掉的**存量正确性隐患**，与 BNSD 无关也独立成立；
3. 反向 epilogue 的"段级散射写出"模式，为后续支持更多非连续输出布局（如 varlen 反向、任意 stride）提供了可复用的基础设施；
4. 测试策略值得借鉴：**逐位一致（rtol=0）证明 strided 与连续路径等价**、**梯度保形断言**、**metadata 复用验证**三层覆盖，比单纯数值容差对比更有说服力。
