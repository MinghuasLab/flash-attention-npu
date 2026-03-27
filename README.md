# FlashAttention for Ascend NPU

<div align="center">
  <a href="#"><img src="https://img.shields.io/badge/English-README.md-blue?style=flat-square" alt="English"></a> <a href="README.zh.md"><img src="https://img.shields.io/badge/中文-README.zh.md-green?style=flat-square" alt="中文"></a>
</div>

## Introduction

FlashAttention significantly improves training and inference efficiency for modern large language models through tiling and memory-aware algorithms. The current mainstream implementation is [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention), which is primarily designed for NVIDIA GPU architectures. During Ascend platform migration, we found the lack of an API-compatible implementation with [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) increased adaptation complexity. To address this gap, this repository implements FlashAttention algorithms adapted for Ascend NPU by following the core design of [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) and building upon the [CANN/CATLASS](https://gitcode.com/cann/catlass) framework and its sample code. We provide an API consistent with [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) to facilitate model migration and enable future attention algorithm optimizations for Ascend NPU.

**This project is under active development, and discussions and contributions are highly welcome.**

## Getting Started

### Prerequisites

- Hardware: Ascend 910B / 910C NPU
- OS: Linux
- Software:
  - CANN >= 8.5.0
  - PyTorch >= 2.1.0
  - torch_npu >= 2.1.0 (same version with PyTorch)
- Python Dependencies
```bash
pip install packaging psutil ninja
```

### Installation 

1. Set environment variables:
```bash
source /usr/local/Ascend/cann/set_env.sh
```

2. Clone the repository:
```bash
git clone https://github.com/MinghuasLab/flash-attention-npu.git
cd flash-attention-npu
git submodule update --init --recursive
```

3. Build and install:

```bash
python setup.py install
```

  Build specific version:

```bash
# Build v2 only
FLASH_ATTN_BUILD_VERSION=v2 python setup.py install

# Build v3 only
FLASH_ATTN_BUILD_VERSION=v3 python setup.py install
```

## Testing

Run test scripts:

```bash
# Test FlashAttention v2
pytest -q -s tests/test_flash_attn_npu.py

# Test FlashAttention v3
pytest -q -s tests/test_flash_attn_npu_v3.py
```

## Usage

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
    window_size=(-1, -1),  # -1 means infinite context window
    rotary_interleaved=True,
    alibi_slopes=None,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *in-place* with the new values
    from k and v. This is useful for incremental decoding: you can pass in the cached key/value
    from the previous step, update them with the new key/value from the current step, and in
    the same kernel perform attention with the updated cache.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache can be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence length for each sequence in the batch.

    If rotary_cos and rotary_sin are passed in, rotary positional embedding will be applied.
    key @k will be rotated by rotary_cos and rotary_sin at positions cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), query @q will be rotated at positions
    cache_seqlens, cache_seqlens + 1, etc.
    If neither causal nor local, query @q will be rotated only at position cache_seqlens
    (i.e., we assume that all tokens in @q are at position cache_seqlens).

    Multi-query and grouped-query attention (MQA/GQA) are supported by passing in fewer KV heads
    than Q heads. Q head count must be divisible by KV head count.
    For example, if Q has 6 heads and K, V have 2 heads, then Q heads 0, 1, 2 will attend to
    K, V head 0, and Q heads 3, 4, 5 will attend to K, V head 1.

    If causal=True, the causal mask is aligned to the bottom-right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = mask) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If a row of the mask is all zeros, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention.
    Query at position i will only attend to keys in [i + seqlen_k - seqlen_q - window_size[0],
    i + seqlen_k - seqlen_q + window_size[1]].

    Warning: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: If no block_table, shape (batch_size_cache, seqlen_cache, nheads_k, headdim);
            if block_table (i.e., paged KV cache), shape (num_blocks, page_block_size, nheads_k, headdim)
            page_block_size must be a multiple of 256.
        v_cache: If no block_table, shape (batch_size_cache, seqlen_cache, nheads_k, headdim);
            if block_table (i.e., paged KV cache), shape (num_blocks, page_block_size, nheads_k, headdim)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate k to k_cache
            starting at the position specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary positional
            embedding to k and q. Only applies if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Same as rotary_cos.
        cache_seqlens: int or (batch_size,), dtype torch.int32. The sequence length of the KV cache.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        cache_batch_idx: (batch_size,), dtype torch.int32. Indices to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If indices are not unique and k and v are provided, the updated values in the cache
            might be from any of the duplicate indices.
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for autoregressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        rotary_interleaved: bool. Only applies if rotary_cos and rotary_sin are passed in.
            If True, rotary positional embedding combines dimensions 0 & 1, 2 & 3, etc.
            If False, rotary positional embedding combines dimensions 0 & rotary_dim / 2,
            1 & rotary_dim / 2 + 1 (i.e., GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32.
            Add bias to the attention scores of query i and key j of (-alibi_slope * |i + seqlen_k - seqlen_q - j|).

    Returns:
        out: (batch_size, seqlen, nheads, headdim).
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
    v3 version of the KV cache interface, with more features compared to v2.

    If k and v are not None, k_cache and v_cache will be updated *in-place* with the new values
    from k and v. This is useful for incremental decoding.

    Multi-query and grouped-query attention (MQA/GQA) are supported.

    If causal=True, the causal mask is aligned to the bottom-right corner of the attention matrix.

    If window_size != (-1, -1), implements sliding window local attention.

    Warning: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: If no page_table, shape (batch_size_cache, seqlen_cache, nheads_k, headdim);
            if page_table (i.e., paged KV cache), shape (num_blocks, page_block_size, nheads_k, headdim)
            page_block_size can be any value (e.g., 1, 2, 3, 64, etc).
        v_cache: If no page_table, shape (batch_size_cache, seqlen_cache, nheads_k, headdim_v);
            if page_table, shape (num_blocks, page_block_size, nheads_k, headdim_v).
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, concatenate k to k_cache
            starting at the position specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim_v). Similar to k.
        qv [optional]: (batch_size, seqlen, nheads, headdim_v).
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). Cosine values for rotary positional embedding.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Sine values for rotary positional embedding.
        cache_seqlens: int or (batch_size,), dtype torch.int32. The sequence length of the KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. Indices to index into the KV cache.
        cache_leftpad: (batch_size,), dtype torch.int32. KV cache starting index.
        page_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        cu_seqlens_q [optional]: Cumulative sequence lengths of queries in ragged mode.
        cu_seqlens_k_new [optional]: Cumulative sequence lengths of new keys in ragged mode.
        max_seqlen_q [optional]: Maximum query sequence length in ragged mode.
        rotary_seqlens [optional]: Sequence lengths for rotary positional embedding.
        q_descale, k_descale, v_descale: Optional dequantization scales for FP8 quantization.
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask.
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        attention_chunk: int. Attention chunk size.
        softcap: float. Activates softcapping attention if > 0.
        rotary_interleaved: bool. Rotary positional embedding mode.
        scheduler_metadata: Optional scheduler metadata.
        num_splits: int. If > 1, split key/value along the sequence dimension into this many chunks.
            If num_splits == 1, no splitting. If num_splits == 0, automatically selected.
        pack_gqa: bool. Whether to pack GQA for better performance.
        sm_margin: int. SM margin for tuning.
        return_softmax_lse: bool. Whether to return logsumexp of attention scores.

    Returns:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional]: (batch_size, nheads, seqlen). The logsumexp of each row of QK^T * scaling.
    """
```

## Features

#### flash_attn_with_kvcache
| Feature | v2 | v3 |
|---------|----|----|
| FP16 (float16) | ✅ | ✅ |
| BF16 (bfloat16) | ✅ | ✅ |
| Causal Attention | ✅ | ✅ |
| Sliding Window Attention | - | - |
| MQA/GQA | ✅ | ✅ |
| Paged KV Cache | ✅ | ✅ |
| Rotary Positional Embedding (RoPE) | - | - |
| ALiBi | - | - |
| Softcapping | - | - |
| FP8 Quantization | - | - |
| Variable-length Sequences | ✅ | ✅ |



## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](./LICENSE) file for details.
