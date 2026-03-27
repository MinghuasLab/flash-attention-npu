# FlashAttention for Ascend NPU

<div align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-README.md-blue?style=flat-square" alt="English"></a> <a href="#"><img src="https://img.shields.io/badge/中文-README.zh.md-green?style=flat-square" alt="中文"></a>
</div>

## 介绍
FlashAttention 通过分块计算和内存感知算法提升训练和推理效率。当前其主流实现为面向 NVIDIA GPU 架构的 [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)。在昇腾平台迁移过程中，我们发现缺少与[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)接口兼容的实现，增加了适配难度。为此，本仓库参照 [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) 的核心设计，基于 [CANN/CATLASS](https://gitcode.com/cann/catlass) 框架及其样例代码，实现了适配昇腾 NPU 的 FlashAttention 算法。我们提供与 [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) 一致的调用接口，便于模型迁移，并支持后续面向昇腾 NPU 的大模型注意力算法改进和优化。

**本项目正在活跃开发中，欢迎参与讨论与贡献！**


## 准备

### 环境要求

- 硬件: 昇腾 910B / 910C NPU
- 系统: Linux
- 软件: 
  - CANN >= 8.5.0
  - PyTorch >= 2.1.0
  - torch_npu >= 2.1.0 (与Pytorch版本相同)
- Python 依赖包
```bash
pip install packaging psutil ninja
```

### 安装步骤

1. 设置环境变量：
```bash
source /usr/local/Ascend/cann/set_env.sh
```
2. 拉取源码：
```bash
git clone https://github.com/MinghuasLab/flash-attention-npu.git
cd flash-attention-npu
git submodule update --init --recursive
```

3. 编译安装：

```bash
python setup.py install
```

  编译特定版本：

```bash
# 仅编译 v2
FLASH_ATTN_BUILD_VERSION=v2 python setup.py install

# 仅编译 v3
FLASH_ATTN_BUILD_VERSION=v3 python setup.py install
```

## 测试

运行测试脚本：

```bash
# 测试 FlashAttention v2
pytest -q -s tests/test_flash_attn_npu.py

# 测试 FlashAttention v3
pytest -q -s tests/test_flash_attn_npu_v3.py
```

## 使用方法

### FlashAttention v2

#### flash_attn_with_kvcache

```python
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 表示无限上下文窗口
    rotary_interleaved=True,
    alibi_slopes=None,
):
    """
    如果 k 和 v 不为 None，k_cache 和 v_cache 将被 *原地更新* 为 k 和 v 的新值。
    这对于增量解码非常有用：你可以传入上一步缓存的 key/value，
    用当前步骤的新 key/value 更新它们，并在一个内核中完成对更新后缓存的注意力计算。

    如果你传入 k / v，必须确保缓存足够大以容纳新值。
    例如，KV 缓存可以预分配最大序列长度，你可以使用 cache_seqlens 来跟踪批次中每个序列的当前序列长度。

    如果传入了 rotary_cos 和 rotary_sin，还会应用旋转位置编码。
    key @k 将在 cache_seqlens, cache_seqlens + 1 等位置被 rotary_cos 和 rotary_sin 旋转。
    如果是 causal 或 local（即 window_size != (-1, -1)），query @q 将在 cache_seqlens, cache_seqlens + 1 等位置被旋转。
    如果既不是 causal 也不是 local，query @q 将仅在 cache_seqlens 位置被旋转
    （即我们认为 @q 中的所有 token 都在位置 cache_seqlens）。

    支持多查询注意力和分组查询注意力（MQA/GQA），通过传入比 Q 头数少的 KV 来实现。
    注意 Q 的头数必须能被 KV 的头数整除。
    例如，如果 Q 有 6 个头，K、V 有 2 个头，那么 Q 的头 0、1、2 将关注 K、V 的头 0，
    Q 的头 3、4、5 将关注 K、V 的头 1。

    如果 causal=True，因果掩码对齐到注意力矩阵的右下角。
    例如，如果 seqlen_q = 2 且 seqlen_k = 5，因果掩码（1 = 保留，0 = 掩码）为：
        1 1 1 1 0
        1 1 1 1 1
    如果 seqlen_q = 5 且 seqlen_k = 2，因果掩码为：
        0 0
        0 0
        0 0
        1 0
        1 1
    如果掩码的某一行全为零，输出将为零。

    如果 window_size != (-1, -1)，实现滑动窗口局部注意力。
    位置 i 的 query 只会关注 [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] 范围内的 key。

    注意：不支持反向传播。

    参数：
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: 如果没有 block_table，形状为 (batch_size_cache, seqlen_cache, nheads_k, headdim)；
            如果有 block_table（即分页 KV 缓存），形状为 (num_blocks, page_block_size, nheads_k, headdim)
            page_block_size 必须是 256 的倍数。
        v_cache: 如果没有 block_table，形状为 (batch_size_cache, seqlen_cache, nheads_k, headdim)；
            如果有 block_table（即分页 KV 缓存），形状为 (num_blocks, page_block_size, nheads_k, headdim)
        k [可选]: (batch_size, seqlen_new, nheads_k, headdim)。如果不为 None，我们将 k 从 cache_seqlens 指定的位置开始拼接到 k_cache。
        v [可选]: (batch_size, seqlen_new, nheads_k, headdim)。与 k 类似。
        rotary_cos [可选]: (seqlen_ro, rotary_dim / 2)。如果不为 None，我们对 k 和 q 应用旋转位置编码。
            仅在传入 k 和 v 时适用。rotary_dim 必须能被 16 整除。
        rotary_sin [可选]: (seqlen_ro, rotary_dim / 2)。与 rotary_cos 类似。
        cache_seqlens: int 或 (batch_size,)，dtype 为 torch.int32。KV 缓存的序列长度。
        block_table [可选]: (batch_size, max_num_blocks_per_seq)，dtype 为 torch.int32。
        cache_batch_idx: (batch_size,)，dtype 为 torch.int32。用于索引 KV 缓存的索引。
            如果为 None，我们假设批次索引为 [0, 1, 2, ..., batch_size - 1]。
            如果索引不唯一，且提供了 k 和 v，缓存中更新的值可能来自任何重复索引。
        softmax_scale: float。softmax 前对 QK^T 的缩放。默认为 1 / sqrt(headdim)。
        causal: bool。是否应用因果注意力掩码（例如用于自回归建模）。
        window_size: (left, right)。如果 != (-1, -1)，实现滑动窗口局部注意力。
        rotary_interleaved: bool。仅在传入 rotary_cos 和 rotary_sin 时适用。
            如果为 True，旋转位置编码将组合维度 0 & 1、2 & 3 等。如果为 False，
            旋转位置编码将组合维度 0 & rotary_dim / 2、1 & rotary_dim / 2 + 1
            （即 GPT-NeoX 风格）。
        alibi_slopes: (nheads,) 或 (batch_size, nheads)，fp32。
            将 (-alibi_slope * |i + seqlen_k - seqlen_q - j|) 的偏置加到
            query i 和 key j 的注意力分数上。

    返回：
        out: (batch_size, seqlen, nheads, headdim)。
    """
```

### FlashAttention v3

#### flash_attn_with_kvcache

```python
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
):
    """
    v3 版本的 KV 缓存接口，相比 v2 增加了更多功能。

    如果 k 和 v 不为 None，k_cache 和 v_cache 将被 *原地更新* 为 k 和 v 的新值。
    这对于增量解码非常有用。

    支持多查询注意力和分组查询注意力（MQA/GQA）。

    如果 causal=True，因果掩码对齐到注意力矩阵的右下角。

    如果 window_size != (-1, -1)，实现滑动窗口局部注意力。

    注意：不支持反向传播。

    参数：
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: 如果没有 page_table，形状为 (batch_size_cache, seqlen_cache, nheads_k, headdim)；
            如果有 page_table（即分页 KV 缓存），形状为 (num_blocks, page_block_size, nheads_k, headdim)
            page_block_size 可以是任意值（如 1, 2, 3, 64 等）。
        v_cache: 如果没有 page_table，形状为 (batch_size_cache, seqlen_cache, nheads_k, headdim_v)；
            如果有 page_table，形状为 (num_blocks, page_block_size, nheads_k, headdim_v)。
        k [可选]: (batch_size, seqlen_new, nheads_k, headdim)。如果不为 None，从 cache_seqlens 指定的位置开始拼接到 k_cache。
        v [可选]: (batch_size, seqlen_new, nheads_k, headdim_v)。与 k 类似。
        qv [可选]: (batch_size, seqlen, nheads, headdim_v)。
        rotary_cos [可选]: (seqlen_ro, rotary_dim / 2)。旋转位置编码的 cos 值。
        rotary_sin [可选]: (seqlen_ro, rotary_dim / 2)。旋转位置编码的 sin 值。
        cache_seqlens: int 或 (batch_size,)，dtype 为 torch.int32。KV 缓存的序列长度。
        cache_batch_idx: (batch_size,)，dtype 为 torch.int32。用于索引 KV 缓存的索引。
        cache_leftpad: (batch_size,)，dtype 为 torch.int32。KV 缓存起始索引。
        page_table [可选]: (batch_size, max_num_blocks_per_seq)，dtype 为 torch.int32。
        cu_seqlens_q [可选]: 变长模式下的 query 累积序列长度。
        cu_seqlens_k_new [可选]: 变长模式下的新 key 累积序列长度。
        max_seqlen_q [可选]: 变长模式下的最大 query 序列长度。
        rotary_seqlens [可选]: 旋转位置编码的序列长度。
        q_descale, k_descale, v_descale: 可选，用于 FP8 量化的反缩放因子。
        softmax_scale: float。softmax 前对 QK^T 的缩放。默认为 1 / sqrt(headdim)。
        causal: bool。是否应用因果注意力掩码。
        window_size: (left, right)。如果 != (-1, -1)，实现滑动窗口局部注意力。
        attention_chunk: int。注意力分块大小。
        softcap: float。大于 0 时激活 softcapping 注意力。
        rotary_interleaved: bool。旋转位置编码模式。
        scheduler_metadata: 可选，调度器元数据。
        num_splits: int。如果 > 1，将 key/value 在序列维度上分割成这么多块。
            如果 num_splits == 1，不分割。如果 num_splits == 0，自动选择。
        pack_gqa: bool。是否打包 GQA 以提高性能。
        sm_margin: int。SM 边际，用于调优。
        return_softmax_lse: bool。是否返回注意力分数的 logsumexp。

    返回：
        out: (batch_size, seqlen, nheads, headdim)。
        softmax_lse [可选]: (batch_size, nheads, seqlen)。QK^T * scaling 的每行 logsumexp。
    """
```

## 特性

#### flash_attn_with_kvcache
| 特性 | v2 | v3 |
|------|----|----|
| FP16 (float16) | ✅ | ✅ |
| BF16 (bfloat16) | ✅ | ✅ |
| 因果注意力 (Causal) | ✅ | ✅ |
| 滑动窗口注意力 | - | - |
| MQA/GQA | ✅ | ✅ |
| 分页 KV 缓存 | ✅ | ✅ |
| 旋转位置编码 (RoPE) | - | - |
| ALiBi | - | - |
| Softcapping | - | - |
| FP8 量化 | - | - |
| 变长序列 | ✅ | ✅ |



## 许可证

本项目采用 BSD 3-Clause License 开源协议。详情请参阅 [LICENSE](./LICENSE) 文件。
