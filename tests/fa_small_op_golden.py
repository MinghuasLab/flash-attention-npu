# Copyright (c) 2026, Minghua Shen.
"""
小算子 FlashAttention 反向标杆（传入 FA 前向 out/lse，仅算反传）。

供 test_flash_attn_npu_v2_bwd.py / test_flash_attn_npu_v3_bwd.py 使用。
"""

import torch


def broadcast_kv_single(num_heads, num_kv_heads, kv_tensor, dtype):
    factor = num_heads // num_kv_heads
    b, _, s, d = kv_tensor.shape
    kv_res = torch.zeros([b, num_heads, s, d], dtype=dtype)
    for i in range(num_heads):
        j = i // factor
        kv_res[:, i : i + 1, :, :] = kv_tensor[:, j : j + 1, :, :]
    return kv_res


def normalize_window(seq_q, seq_k, is_causal, window_size_left, window_size_right):
    wl = window_size_left
    wr = window_size_right
    if wl >= seq_k - 1:
        wl = -1
    if wr >= seq_q - 1:
        wr = -1
    if is_causal:
        wr = 0
    is_causal_out = wl < 0 and wr == 0
    is_local = (wl >= 0 or wr >= 0) and not is_causal_out
    return wl, wr, is_causal_out, is_local


def make_window_atten_mask(
    seq_q,
    seq_k,
    is_causal=False,
    window_size_left=-1,
    window_size_right=-1,
):
    wl, wr, is_causal_out, is_local = normalize_window(
        seq_q, seq_k, is_causal, window_size_left, window_size_right
    )
    if not is_causal_out and not is_local:
        return torch.tensor(0)

    offset = seq_k - seq_q
    shape = (seq_q, seq_k)
    if is_causal_out:
        return torch.triu(torch.ones(shape), diagonal=offset + 1)

    mask = torch.zeros(shape)
    if wr >= 0:
        mask = mask + torch.triu(torch.ones(shape), diagonal=wr + 1 + offset)
    if wl >= 0:
        mask = mask + torch.tril(torch.ones(shape), diagonal=-wl - 1 + offset)
    return mask


def _resolve_atten_mask(q, k, atten_mask, is_causal, window_size_left, window_size_right):
    if atten_mask is not None and len(atten_mask.shape) != 0:
        return atten_mask
    seq_q = q.shape[2]
    seq_k = k.shape[2]
    return make_window_atten_mask(
        seq_q, seq_k, is_causal, window_size_left, window_size_right
    )


def tsoftmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = muls.sum(dim=-1, keepdims=True)
    return (dp - muls_r) * softmax_res


def _keep_prob(dropout_p):
    return 1.0 - dropout_p


def _softcap_backward_factor(q, k, scale, softcap, compute_dtype, *, gtype=torch.float64):
    if softcap <= 0.0:
        return None
    qb = q.to(compute_dtype)
    kb = k.to(compute_dtype)
    qk = torch.matmul(qb, kb.permute(0, 1, 3, 2)).to(torch.float32).mul(scale)
    tanh_qk = torch.tanh(qk.to(gtype) / softcap)
    return 1.0 - tanh_qk * tanh_qk


def sum_gqa_grad(dk_or_dv, nheads, nheads_k, batch, seq_k, headdim):
    if nheads == nheads_k:
        return dk_or_dv
    g = nheads // nheads_k
    return (
        torch.sum(
            dk_or_dv.reshape(batch, nheads_k, g, seq_k, headdim),
            dim=2,
            keepdim=True,
        ).reshape(batch, nheads_k, seq_k, headdim)
    )


def softmax_res_from_fa_lse_bsnd(
    q_bn,
    k_bn,
    softmax_lse,
    scale,
    softcap,
    is_causal,
    window_size_left,
    window_size_right,
    compute_dtype,
    *,
    gtype=torch.float64,
):
    """q_bn / k_bn: (B, N, S, D)。softmax_lse: FA BSND (B, N, S_q)。"""
    atten_mask = _resolve_atten_mask(
        q_bn, k_bn, None, is_causal, window_size_left, window_size_right
    )
    qb = q_bn.to(compute_dtype)
    kb = k_bn.to(compute_dtype)
    qk = torch.matmul(qb, kb.permute(0, 1, 3, 2)).to(torch.float32).mul(scale)
    if softcap > 0.0:
        qk = softcap * torch.tanh(qk / softcap)
    if atten_mask is not None and len(atten_mask.shape) != 0:
        qk = qk + atten_mask.to(torch.float32) * (-40000.0)
    lse = softmax_lse.to(torch.float32).to(gtype)
    if lse.dim() != 3:
        raise ValueError(f"BSND softmax_lse 期望 3 维 (B,N,S)，实际 {tuple(lse.shape)}")
    lse = lse.unsqueeze(-1)
    softmax_res = torch.exp(qk.to(gtype) - lse)
    if atten_mask is not None and len(atten_mask.shape) != 0:
        softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = 0
    return softmax_res


def softmax_res_from_fa_lse_tnd_slice(
    q_bn,
    k_bn,
    softmax_lse_nt_slice,
    scale,
    softcap,
    is_causal,
    window_size_left,
    window_size_right,
    compute_dtype,
    *,
    gtype=torch.float64,
):
    """TND 单 batch 切片：FA varlen softmax_lse 切片 (N, S_q) NT。"""
    atten_mask = _resolve_atten_mask(
        q_bn, k_bn, None, is_causal, window_size_left, window_size_right
    )
    qb = q_bn.to(compute_dtype)
    kb = k_bn.to(compute_dtype)
    qk = torch.matmul(qb, kb.permute(0, 1, 3, 2)).to(torch.float32).mul(scale)
    if softcap > 0.0:
        qk = softcap * torch.tanh(qk / softcap)
    if atten_mask is not None and len(atten_mask.shape) != 0:
        qk = qk + atten_mask.to(torch.float32) * (-40000.0)
    lse = softmax_lse_nt_slice.to(torch.float32).to(gtype).unsqueeze(0).unsqueeze(-1)
    softmax_res = torch.exp(qk.to(gtype) - lse)
    if atten_mask is not None and len(atten_mask.shape) != 0:
        softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = 0
    return softmax_res


def tbackward_bsnd(dx, q, k, v, softmax_res, drop_mask, scale, softcap, dropout_p):
    keep_prob = _keep_prob(dropout_p)
    dp = torch.matmul(dx, v.permute(0, 1, 3, 2))
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res.permute(0, 1, 3, 2)
        dp_drop = dp
    else:
        drop_res = softmax_res.mul(drop_mask).mul(1.0 / keep_prob).permute(0, 1, 3, 2)
        dp_drop = dp * drop_mask * (1.0 / keep_prob)
    dv = torch.matmul(drop_res, dx)
    softmax_grad_res = tsoftmax_grad(dp_drop, softmax_res) * scale
    softcap_factor = _softcap_backward_factor(q, k, scale, softcap, q.dtype)
    if softcap_factor is not None:
        softmax_grad_res = softmax_grad_res * softcap_factor
    dq = torch.matmul(softmax_grad_res, k)
    dk = torch.matmul(softmax_grad_res.permute(0, 1, 3, 2), q)
    return dq, dk, dv


def tbackward_tnd(dx, q, k, v, softmax_res, drop_mask, scale, softcap, dropout_p):
    keep_prob = _keep_prob(dropout_p)
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res.permute(0, 1, 3, 2)
        dp_drop = torch.matmul(dx, v.permute(0, 1, 3, 2))
    else:
        drop_res = softmax_res.mul(drop_mask).mul(1.0 / keep_prob).permute(0, 1, 3, 2)
        dp = torch.matmul(dx, v.permute(0, 1, 3, 2))
        dp_drop = dp * drop_mask * (1.0 / keep_prob)
    dv = torch.matmul(drop_res, dx)
    softmax_grad_res = tsoftmax_grad(dp_drop, softmax_res) * scale
    softcap_factor = _softcap_backward_factor(q, k, scale, softcap, q.dtype)
    if softcap_factor is not None:
        softmax_grad_res = softmax_grad_res * softcap_factor
    dq = torch.matmul(softmax_grad_res, k)
    dk = torch.matmul(softmax_grad_res.permute(0, 1, 3, 2), q)
    return dq, dk, dv


def golden_bsnd_bwd_from_fwd(
    q,
    k,
    v,
    dout,
    out,
    softmax_lse,
    nheads,
    nheads_k,
    scale,
    softcap,
    dropout_p,
    is_causal,
    window_size_left,
    window_size_right,
    *,
    gtype=torch.float64,
):
    """BSND 反传标杆。softmax_lse layout: FA (B, N, S_q)。"""
    del out
    batch, seq_q, _, headdim = q.shape
    seq_k = k.shape[1]
    compute_dtype = q.dtype

    q_bn = q.detach().cpu().permute(0, 2, 1, 3).to(gtype)
    k_bn = k.detach().cpu().permute(0, 2, 1, 3).to(gtype)
    v_bn = v.detach().cpu().permute(0, 2, 1, 3).to(gtype)
    dx_bn = dout.detach().cpu().permute(0, 2, 1, 3).to(gtype)
    lse_cpu = softmax_lse.detach().cpu().to(torch.float32)

    if nheads == nheads_k:
        k_new, v_new = k_bn, v_bn
    else:
        k_new = broadcast_kv_single(nheads, nheads_k, k_bn, gtype)
        v_new = broadcast_kv_single(nheads, nheads_k, v_bn, gtype)

    softmax_res = softmax_res_from_fa_lse_bsnd(
        q_bn,
        k_new,
        lse_cpu,
        scale,
        softcap,
        is_causal,
        window_size_left,
        window_size_right,
        compute_dtype,
        gtype=gtype,
    )
    drop_mask = torch.tensor(1)
    dq_bn, dk_bn, dv_bn = tbackward_bsnd(
        dx_bn, q_bn, k_new, v_new, softmax_res, drop_mask, scale, softcap, dropout_p
    )
    dk_bn = sum_gqa_grad(dk_bn, nheads, nheads_k, batch, seq_k, headdim)
    dv_bn = sum_gqa_grad(dv_bn, nheads, nheads_k, batch, seq_k, headdim)

    dq = dq_bn.permute(0, 2, 1, 3).to(compute_dtype)
    dk = dk_bn.permute(0, 2, 1, 3).to(compute_dtype)
    dv = dv_bn.permute(0, 2, 1, 3).to(compute_dtype)
    return dq, dk, dv


def golden_tnd_bwd_from_fwd(
    q,
    k,
    v,
    dout,
    out,
    softmax_lse,
    nheads,
    nheads_k,
    seqlens_q,
    seqlens_k,
    scale,
    softcap,
    dropout_p,
    is_causal,
    window_size_left,
    window_size_right,
    *,
    gtype=torch.float64,
):
    """TND 反传标杆。softmax_lse layout: FA varlen (N, total_q) NT。"""
    del out
    seqlens_q = list(seqlens_q)
    seqlens_k = list(seqlens_k)
    cu_q = [0]
    cu_k = [0]
    for sq, sk in zip(seqlens_q, seqlens_k):
        cu_q.append(cu_q[-1] + int(sq))
        cu_k.append(cu_k[-1] + int(sk))
    headdim = q.shape[-1]
    compute_dtype = q.dtype
    lse_nt = softmax_lse.detach().cpu().to(torch.float32)
    if lse_nt.dim() != 2:
        raise ValueError(f"TND softmax_lse 期望 NT (N, total_q)，实际 {tuple(lse_nt.shape)}")

    dq_golden = torch.empty_like(q, dtype=compute_dtype)
    dk_golden = torch.empty_like(k, dtype=compute_dtype)
    dv_golden = torch.empty_like(v, dtype=compute_dtype)
    drop_mask = torch.tensor(1)

    for i, sq in enumerate(seqlens_q):
        sk = seqlens_k[i]
        if sq == 0 or sk == 0:
            continue

        qi = q[cu_q[i] : cu_q[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(gtype)
        ki = k[cu_k[i] : cu_k[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(gtype)
        vi = v[cu_k[i] : cu_k[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(gtype)
        dxi = dout[cu_q[i] : cu_q[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(gtype)
        lse_i = lse_nt[:, cu_q[i] : cu_q[i + 1]]

        if nheads == nheads_k:
            ki_new, vi_new = ki, vi
        else:
            ki_new = broadcast_kv_single(nheads, nheads_k, ki, gtype)
            vi_new = broadcast_kv_single(nheads, nheads_k, vi, gtype)

        softmax_res_i = softmax_res_from_fa_lse_tnd_slice(
            qi,
            ki_new,
            lse_i,
            scale,
            softcap,
            is_causal,
            window_size_left,
            window_size_right,
            compute_dtype,
            gtype=gtype,
        )
        dqi, dki, dvi = tbackward_tnd(
            dxi, qi, ki_new, vi_new, softmax_res_i, drop_mask, scale, softcap, dropout_p
        )
        dki = sum_gqa_grad(dki, nheads, nheads_k, 1, sk, headdim)
        dvi = sum_gqa_grad(dvi, nheads, nheads_k, 1, sk, headdim)

        dq_golden[cu_q[i] : cu_q[i + 1]] = (
            dqi.permute(0, 2, 1, 3).reshape(sq, nheads, headdim).to(compute_dtype)
        )
        dk_golden[cu_k[i] : cu_k[i + 1]] = (
            dki.permute(0, 2, 1, 3).reshape(sk, nheads_k, headdim).to(k.dtype)
        )
        dv_golden[cu_k[i] : cu_k[i + 1]] = (
            dvi.permute(0, 2, 1, 3).reshape(sk, nheads_k, headdim).to(v.dtype)
        )

    return dq_golden, dk_golden, dv_golden