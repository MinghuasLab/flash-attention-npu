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
