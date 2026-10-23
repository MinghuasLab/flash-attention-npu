"""Test-only Torch attention and sparse-metadata oracles."""

import inspect
import torch

from flash_attn_npu_4.catlass_dsl.block_sparsity import BlockSparseTensorsTorch, _check_cumulative
from flash_attn_npu_4.catlass_dsl.mods import Mod, reference_callable


def _zero_dependency(*tensors):
    # Empty reductions connect zero adjoints without reading NaN/Inf values.
    return sum(t.narrow(0, 0, 0).sum(dtype=torch.float32) for t in tensors)


def _call(fn, values, aux, scalars, info=(0, 0, 0, 0)):
    if isinstance(fn, Mod):
        callback = reference_callable(fn, "mask" if len(values) == 4 else "score")
        return callback(*values, info, aux, scalars)
    # Independent Torch-only oracles may retain their original signatures.
    count = len(inspect.signature(fn).parameters)
    if count == len(values) + 3:
        return fn(*values, info, aux, scalars)
    return fn(*values, aux, scalars) if count == len(values) + 2 else fn(*values)


def reference_block_sparsity(
    batch_size,
    num_heads,
    seqlen_q,
    seqlen_k,
    mask_mod=None,
    *,
    aux_tensors=(),
    aux_scalars=(),
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    compute_full_blocks=True,
    use_fast_sampling=False,
):
    """Classify 128x128 tiles on CPU using an independent Torch predicate.

    Fixed and packed results use the public sparse tuple. Classification and
    packing are independent of the production builder and normalization code.
    Only valid elements participate; an empty segment has no visible blocks.
    """
    batches, heads = batch_size or 1, num_heads or 1
    nq, nk = (seqlen_q + 127) // 128, (seqlen_k + 127) // 128
    pc = torch.zeros((batches, heads, nq), dtype=torch.int32)
    fc = torch.zeros_like(pc)
    pi = torch.zeros((batches, heads, nq, nk), dtype=torch.int32)
    fi = torch.zeros_like(pi)
    q_offsets = [0] * (batches + 1) if cu_seqlens_q is None else cu_seqlens_q.tolist()
    k_offsets = [0] * (batches + 1) if cu_seqlens_k is None else cu_seqlens_k.tolist()
    q_lengths = [seqlen_q] * batches if cu_seqlens_q is None else torch.diff(cu_seqlens_q).tolist()
    k_lengths = [seqlen_k] * batches if cu_seqlens_k is None else torch.diff(cu_seqlens_k).tolist()
    for b, (qlen, klen) in enumerate(zip(q_lengths, k_lengths)):
        info = (qlen, klen, q_offsets[b], k_offsets[b])
        for h in range(heads):
            for qb in range((qlen + 127) // 128):
                m = min(128, qlen - qb * 128)
                for kb in range((klen + 127) // 128):
                    n = min(128, klen - kb * 128)
                    if use_fast_sampling:
                        qi = qb * 128 + torch.tensor([0, 0, m - 1, m - 1, m // 2])
                        ki = kb * 128 + torch.tensor([0, n - 1, 0, n - 1, n // 2])
                        shape = (5,)
                    else:
                        qi = torch.arange(qb * 128, qb * 128 + m)[:, None]
                        ki = torch.arange(kb * 128, kb * 128 + n)[None, :]
                        shape = (m, n)
                    keep = (
                        True
                        if mask_mod is None
                        else _call(
                            mask_mod,
                            (torch.tensor(b), torch.tensor(h), qi, ki),
                            aux_tensors,
                            aux_scalars,
                            info,
                        )
                    )
                    keep = torch.as_tensor(keep)
                    if keep.dtype != torch.bool:
                        raise ValueError("mask_mod must return boolean visibility")
                    keep = keep.expand(shape)
                    if bool(keep.all()) and compute_full_blocks:
                        fi[b, h, qb, int(fc[b, h, qb])] = kb
                        fc[b, h, qb] += 1
                    elif bool(keep.any()):
                        pi[b, h, qb, int(pc[b, h, qb])] = kb
                        pc[b, h, qb] += 1
    if not compute_full_blocks:
        fc, fi = None, None
    cu_rows = cu_slots = None
    if cu_seqlens_q is not None:
        rows = [(n + 127) // 128 for n in q_lengths]
        cols = [(n + 127) // 128 for n in k_lengths]
        cu_rows = torch.tensor([0, *torch.tensor(rows).cumsum(0).tolist()], dtype=torch.int32)
        slots = [m * n for m, n in zip(rows, cols)]
        cu_slots = torch.tensor([0, *torch.tensor(slots).cumsum(0).tolist()], dtype=torch.int32)

        def pack(counts, indices):
            counts = torch.cat([counts[b, :, :m] for b, m in enumerate(rows)], dim=1)
            indices = torch.cat(
                [
                    indices[b, :, :m, :n].reshape(heads, m * n)
                    for b, (m, n) in enumerate(zip(rows, cols))
                ],
                dim=1,
            )
            return counts.contiguous(), indices.contiguous()

        pc, pi = pack(pc, pi)
        if fc is not None:
            fc, fi = pack(fc, fi)
    return BlockSparseTensorsTorch(pc, pi, fc, fi, cu_rows, cu_slots, (128, 128))


class _ExplicitScore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, forward, backward, values, info, aux, scalars):
        ctx.save_for_backward(x)
        ctx.callback, ctx.values, ctx.info, ctx.aux, ctx.scalars = (
            backward,
            values,
            info,
            aux,
            scalars,
        )
        return _call(forward, (x, *values), aux, scalars, info)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        callback = (
            reference_callable(ctx.callback, "score_bwd")
            if isinstance(ctx.callback, Mod)
            else ctx.callback
        )
        dx = callback(grad, x, *ctx.values, ctx.info, ctx.aux, ctx.scalars)
        return dx, None, None, None, None, None, None


def reference_attention(
    q,
    k,
    v,
    score_mod=None,
    mask_mod=None,
    *,
    aux_tensors=(),
    aux_scalars=(),
    scale=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    return_lse=False,
    score_mod_bwd=None,
):
    """Dense differentiable oracle, including zero-gradient fully masked rows."""
    if (
        q.ndim not in (3, 4)
        or k.ndim not in (3, 4)
        or v.shape != k.shape
        or q.shape[-1] != k.shape[-1]
    ):
        raise ValueError("expected compatible BNSD or packed TND q/k/v")
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    varlen_q, varlen_k = q.ndim == 3, k.ndim == 3
    if (varlen_q and cu_seqlens_q is None) or (varlen_k and cu_seqlens_k is None):
        raise ValueError("each packed reference side needs its cumulative length array")
    B = cu_seqlens_q.numel() - 1 if varlen_q else q.shape[0]
    BK = cu_seqlens_k.numel() - 1 if varlen_k else k.shape[0]
    if B < 0 or B != BK:
        raise ValueError("nonnegative, matching q/k batch dimensions required")
    for t, cu, packed in ((q, cu_seqlens_q, varlen_q), (k, cu_seqlens_k, varlen_k)):
        if packed:
            _check_cumulative(cu, B, t.shape[0], t.device, total=t.shape[0])
        elif cu is not None:
            raise ValueError("fixed-layout side cannot have cumulative lengths")
    qh, kh = (q.shape[1], k.shape[1])
    if kh < 1 or qh < 1 or qh % kh:
        raise ValueError("Hq must be a positive multiple of Hkv")
    if B == 0:
        dependency = _zero_dependency(q, k, v)
        output = torch.zeros_like(q) + dependency.to(q.dtype)
        shape = (q.shape[0], q.shape[1]) if varlen_q else q.shape[:-1]
        lse = torch.full(shape, -torch.inf, dtype=torch.float32, device=q.device) + dependency
        return (output, lse) if return_lse else output
    outputs, log_sums = [], []
    for b in range(B):
        qs, ks = 0, 0
        if varlen_q:
            qs, qe = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
            qb = q[qs:qe].transpose(0, 1)
        else:
            qb = q[b]
        if varlen_k:
            ks, ke = int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1])
            kb, vb = k[ks:ke].transpose(0, 1), v[ks:ke].transpose(0, 1)
        else:
            kb, vb = k[b], v[b]
        H, Sq, D = qb.shape
        Sk = kb.shape[1]
        group = H // kb.shape[0]
        # Keep the oracle's indexing and arithmetic independent of device code.
        # Promote before expanding heads so autograd also sums GQA head
        # gradients in FP32, matching the specified FP32 dK/dV accumulation.
        # Promoting afterwards silently rounds every head's adjoint first.
        expanded_k = kb.float().repeat_interleave(group, dim=0)
        expanded_v = vb.float().repeat_interleave(group, dim=0)
        x = torch.matmul(qb.float(), expanded_k.transpose(-1, -2)) * scale
        hi = torch.arange(H, device=q.device)[:, None, None]
        qi = torch.arange(Sq, device=q.device)[None, :, None]
        ki = torch.arange(Sk, device=q.device)[None, None, :]
        bi = torch.tensor(b, device=q.device)
        info = (Sq, Sk, qs, ks)
        if score_mod_bwd is not None:
            z = _ExplicitScore.apply(
                x,
                score_mod,
                score_mod_bwd,
                (bi, hi, qi, ki),
                info,
                aux_tensors,
                aux_scalars,
            )
        else:
            z = (
                x
                if score_mod is None
                else _call(score_mod, (x, bi, hi, qi, ki), aux_tensors, aux_scalars, info)
            )
        # Keep constant score callbacks connected with a zero Q/K derivative.
        z = z + torch.zeros_like(x) + _zero_dependency(x)
        if mask_mod is not None:
            keep = _call(mask_mod, (bi, hi, qi, ki), aux_tensors, aux_scalars, info)
            keep = torch.as_tensor(keep, device=q.device)
            if keep.dtype != torch.bool:
                raise ValueError("mask_mod must return boolean visibility")
            z = torch.where(keep, z, -torch.inf)
        if Sk == 0:
            result = torch.zeros_like(qb, dtype=torch.float32) + _zero_dependency(qb, kb, vb)
            logsum = torch.full((H, Sq), -torch.inf, device=q.device) + _zero_dependency(qb, kb)
        else:
            nonempty = (z != -torch.inf).any(dim=-1, keepdim=True)
            safe_z = torch.where(nonempty, z, torch.zeros_like(z))
            prob = torch.where(nonempty, safe_z.softmax(dim=-1), torch.zeros_like(z))
            result = prob @ expanded_v
            # Constant score mods still have well-defined zero dQ/dK. Keep
            # these inputs in the oracle graph without using compiler AD.
            result = result + _zero_dependency(qb, expanded_k)
            logsum = torch.where(nonempty.squeeze(-1), safe_z.logsumexp(-1), -torch.inf)
        # LSE is independent of V, but a V-only request should return zeros.
        logsum = logsum + _zero_dependency(vb)
        result = result.to(q.dtype)
        outputs.append(result.transpose(0, 1) if varlen_q else result)
        log_sums.append(logsum.transpose(0, 1) if varlen_q else logsum)
    output = torch.cat(outputs) if varlen_q else torch.stack(outputs)
    lse = torch.cat(log_sums) if varlen_q else torch.stack(log_sums)
    return (output, lse) if return_lse else output
