# FA4 CATLASS DSL Usage Examples

Run these examples in order in a Python file on Ascend 950. Later examples reuse the imports, callbacks, and tensors defined earlier.

## Fixed-length attention with a custom mask

```python
import torch
import torch_npu
from catlass import tla

from flash_attn_npu_4 import flash_attn_func, flash_attn_varlen_func
from flash_attn_npu_4.catlass_dsl import compute_block_sparsity, simt
from flash_attn_npu_4.catlass_dsl.block_sparsity import transpose_block_sparsity


torch.npu.set_device(0)


@simt
@tla.jit
def local_mask(b, h, q_idx, kv_idx, info, aux):
    # info[0] and info[1] are the segment Q/KV lengths; align bottom-right.
    center = q_idx + info[1] - info[0]
    return (kv_idx <= center) & (kv_idx >= center - 128)


q = torch.randn(2, 256, 4, 128, device="npu", dtype=torch.float16, requires_grad=True)
k = torch.randn(2, 256, 2, 128, device="npu", dtype=torch.float16, requires_grad=True)
v = torch.randn_like(k, requires_grad=True)

out, lse = flash_attn_func(q, k, v, mask_mod=local_mask, return_lse=True)
(out.float().square().mean() + lse.mean()).backward()
```

## Custom score and backward

```python
@simt
@tla.jit
def scale_score(s, b, h, q_idx, kv_idx, info, aux, scalars):
    return s * scalars[0]


@simt
@tla.jit
def scale_score_bwd(g, s, b, h, q_idx, kv_idx, info, aux, scalars):
    return g * scalars[0]


out = flash_attn_func(
    q, k, v,
    score_mod=scale_score,
    score_mod_bwd=scale_score_bwd,
    aux_scalars=(0.5,),
)
out.float().square().mean().backward()
```

## Packed attention

```python
cu = torch.tensor([0, 128, 320], dtype=torch.int32, device=q.device)
pq = torch.randn(320, 4, 128, device=q.device, dtype=q.dtype, requires_grad=True)
pk = torch.randn(320, 2, 128, device=q.device, dtype=q.dtype, requires_grad=True)
pv = torch.randn_like(pk, requires_grad=True)

pout, plse, out_accum, lse_accum = flash_attn_varlen_func(
    pq, pk, pv,
    cu_seqlens_q=cu,
    cu_seqlens_k=cu,
    max_seqlen_q=192,
    max_seqlen_k=192,
    mask_mod=local_mask,
    return_lse=True,
)
pout.float().square().mean().backward()
assert out_accum.numel() == lse_accum.numel() == 0
```

## Block-sparse attention

```python
def sparse_attention(q, k, v):
    batch, sq, heads, _ = q.shape
    sk = k.shape[1]
    blocks = compute_block_sparsity(
        128, 128, batch, heads, sq, sk,
        local_mask, None, q.device,
    )
    blocks_bwd = transpose_block_sparsity(
        blocks,
        batch_size=batch,
        num_heads=heads,
        seqlen_q=sq,
        seqlen_k=sk,
    )
    return flash_attn_func(
        q, k, v,
        mask_mod=local_mask,
        block_sparse_tensors=blocks,
        block_sparse_tensors_bwd=blocks_bwd,
        return_lse=True,
    )


out, lse = sparse_attention(q, k, v)
out.float().square().mean().backward()
```

## torch.compile

```python
# For packed sparse construction, also enable:
# torch._dynamo.config.capture_dynamic_output_shape_ops = True

compiled_attention = torch.compile(
    sparse_attention, backend="inductor", fullgraph=True, dynamic=True,
)
out, lse = compiled_attention(q, k, v)
(out.float().square().mean() + lse.mean()).backward()
```

## Prequantized MXFP8 inference

MXFP8 entry points take a required `TensorwiseScales` argument immediately
after Q/K/V. Pass it positionally or as `scales=...`.

- `flash_attn_func_with_quant(...)` — fixed-length BSND
- `flash_attn_varlen_func_with_quant(...)` — variable-length / mixed BSND/TND

The non-quantized `flash_attn_func` / `flash_attn_varlen_func` remain FP16/BF16
only. All three scales are required and Q/K/V must already be
`torch.float8_e4m3fn`; this API does not quantize floating-point inputs. MXFP8
is supported on Ascend 950 with head dimension
128 and logical sparse blocks of 128x128. Q/K/V and scale tensors must not
require gradients.

Scales contain E8M0 bytes, exposed as `torch.uint8` or
`torch.float8_e8m0fnu`. Byte `e` represents `2**(e-127)`, so byte 127 is scale
1.0; byte 255 is invalid. Q/K group channels by 32; V groups sequence rows by
32. Two adjacent groups are stored in the final dimension of size 2.

| Input layout | Q scale | K scale | V scale |
| --- | --- | --- | --- |
| BSND | `[B,Sq,Hq,2,2]` | `[B,Sk,Hkv,2,2]` | `[B,ceil(Sk/64),Hkv,128,2]` |
| Packed TND | `[Tq,Hq,2,2]` | `[Tk,Hkv,2,2]` | `[capacity,Hkv,128,2]` |

The public fixed-length API uses BSND. The variable-length API permits Q and
KV to independently use BSND or TND. Packed cumulative token offsets retain
the existing int32 `[B+1]` contract, including the leading zero. Packed V-scale
groups restart at each sequence boundary, and their storage is concatenated
with each segment padded to 64 rows. The caller must provide capacity of at
least `sum(ceil(segment_kv_length/64))`; extra trailing capacity is allowed.

```python
# q_fp8/k_fp8/v_fp8 and their scale tensors are supplied by the caller.
from flash_attn_npu_4.catlass_dsl import TensorwiseScales, flash_attn_func_with_quant

out, lse = flash_attn_func_with_quant(
    q_fp8, k_fp8, v_fp8,
    scales=TensorwiseScales(q=q_scale, k=k_scale, v=v_scale),
    mask_mod=local_mask,
    return_lse=True,
)
assert out.dtype == torch.bfloat16
assert lse.dtype == torch.float32
```

MXFP8 supports score/mask callbacks, GQA/MQA, causal/window masks, softcap,
full/partial sparse blocks, and `torch.compile` for forward inference.
Output is BF16 and LSE is FP32. Fully masked rows produce zero output and
`-inf` LSE. MXFP8 backward and the separate
`torch.nn.attention.flex_attention` frontend are not supported.

The direct `catlass_dsl` API returns an `(out, lse_or_none)` pair. The top-level
FA4 API follows the fixed-length and variable-length return conventions shown
above.

## Compilation cache

Compiled kernels are cached automatically within the current process.
Different sequence lengths reuse a cached kernel when the compilation
specialization remains the same; changes to alignment, layout, callbacks,
or execution strategy can require another compilation.

To reuse compiled kernels across processes with CATLASS 2.2, set these
variables before importing FA4:

```bash
export FLASH_ATTENTION_CATLASS_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CATLASS_DSL_CACHE_DIR=/path/to/attention-cache
```

Persistent caching is disabled by default. The cache directory is optional
and defaults to `/tmp/$USER/flash_attention_catlass_dsl_cache` under the
platform temporary directory.

To clear the cache:

```python
from flash_attn_npu_4.catlass_dsl.runtime import clear_compile_cache

clear_compile_cache()                 # Clear memory; retain disk artifacts.
clear_compile_cache(persistent=True)  # Also clear the active disk cache namespace.
```
