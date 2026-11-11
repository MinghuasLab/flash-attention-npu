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


## MXFP8 inference

Fixed-length and variable-length FA4 accept keyword-only `q_descale`,
`k_descale`, `v_descale`; all three are required together. Q/K/V must already be
`torch.float8_e4m3fn` (no in-API quantization), have no gradients, and use
Ascend 950, head dimension 128 and 128x128 sparse blocks.

Scales are E8M0 bytes (`torch.uint8` or `torch.float8_e8m0fnu`): byte `e` means
`2**(e-127)`, so 127 is 1.0 and 255 is invalid. Q/K group channels by 32, V
groups rows by 32; adjacent groups occupy a final dimension of 2.

| Layout | Q scale | K scale | V scale |
| --- | --- | --- | --- |
| BSND | `[B,Sq,Hq,2,2]` | `[B,Sk,Hkv,2,2]` | `[B,ceil(Sk/64),Hkv,128,2]` |
| Packed TND | `[Tq,Hq,2,2]` | `[Tk,Hkv,2,2]` | `[capacity,Hkv,128,2]` |

Packed TND uses int32 `cu_seqlens` with a leading zero. V-scale storage is
concatenated per sequence, each padded to 64 rows; capacity must be at least
`sum(ceil(segment_kv_length/64))`. Q and KV may independently use BSND or TND.

```python
out, lse = flash_attn_func(
    q_fp8, k_fp8, v_fp8,
    q_descale=q_scale, k_descale=k_scale, v_descale=v_scale,
    mask_mod=local_mask, return_lse=True,
)
# out: BF16, lse: FP32
```

Supported: score/mask callbacks, GQA/MQA, causal/window, softcap, sparse blocks
and `torch.compile` forward. Fully masked rows give zero output and `-inf` LSE.
MXFP8 backward and the `flex_attention` frontend are not supported.

## Compile reuse

Default is baseline. Broad forward reuse is opt-in:

```python
with forward_compile_mode("broad"):
    ...  # preparation and execution
```

`get_forward_compile_mode()` returns the effective mode. Without a context,
`FLASH_ATTENTION_CATLASS_DSL_DYNAMIC_SHAPES=1` enables broad; unset/0 keeps
baseline. Modes use separate caches. Broad accepts varying B, Hq/Hkv, Sq/Sk,
alignment, packed lengths, auxiliary strides and sparse capacities; D stays 128.
Callback identity, dtype, layout, sparse mode and auxiliary types still form
separate variants.

Backward has an independent switch, set before the forward call:

```python
with forward_compile_mode("broad"), backward_compile_mode("broad"):
    precompile_attention(sample_q, sample_k, sample_v, mask_mod=local_mask)
    precompile_backward_attention(
        sample_q, sample_k, sample_v, mask_mod=local_mask, lse_gradient=True,
    )
    out, lse = flash_attn_func(q, k, v, mask_mod=local_mask, return_lse=True)
    (out.float().square().mean() + lse.mean()).backward()
```

`FLASH_ATTENTION_CATLASS_DSL_BACKWARD_DYNAMIC_SHAPES=1` is the environment
equivalent. The selected backward policy is saved by forward, so it survives
Autograd worker dispatch. Precompile helpers do not execute attention or
backward; `lse_gradient=True` selects the dLSE variant. MXFP8 is forward-only.
For TND, pass `varlen=True`; prototypes must be nonempty.

`compile_cache_info()` reports hits, misses, compilations, failures, size and
disk hits. Persistent reuse is opt-in:

```bash
export FLASH_ATTENTION_CATLASS_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CATLASS_DSL_CACHE_DIR=/path/to/cache
```

`clear_compile_cache()` clears memory; `clear_compile_cache(persistent=True)`
also clears the active disk cache. `backward_compile_diagnostics()` records
opt-in backward capture/execution events. Training additionally requires the
supported schema-v4 mixed executor and cooperative batch launch; see
`BACKWARD_BATCH_SYNC_FIX.md` for the baseline launch fix.
