# Copyright (c) 2026, Minghua Shen.

import sys
import os
import torch
import torch_npu
import pytest
import numpy as np
from flash_attn import flash_attn_varlen_func, flash_attn_func

torch.npu.set_device(1)
np.random.seed(3)
torch.manual_seed(3)


def get_cu_seqlens(seqlens_list):
    cu = torch.zeros(len(seqlens_list) + 1, dtype = torch.int32)
    for i in range(len(seqlens_list) + 1):
        cu[i] = sum(seqlens_list[:i])
    return cu

def test_tnd_bwd_npu(nheads, nheads_k, headdim, list_seq):
    g = nheads / nheads_k
    scale = 1 / (headdim ** 0.5)
    seqlens_list_q = np.array(list_seq)
    seqlens_list_k = np.array(list_seq)
    B = len(list_seq)

    keep_prob = 1.0
    max_seqlen_q = np.max(seqlens_list_q)
    max_seqlen_k = np.max(seqlens_list_k)
    cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
    cu_seqlens_k = get_cu_seqlens(seqlens_list_k)
    total_q = seqlens_list_q.sum()
    total_k = seqlens_list_k.sum()
    cu_seq_len_list = cu_seqlens_q[1:].cpu().numpy().tolist()
    cu_seq_kvlen_list = cu_seqlens_k[1:].cpu().numpy().tolist()
    print("total_q: ", total_q)
    print("total_k: ", total_k)
    print("cu_seq_len_list is ", cu_seq_len_list)
    print("cu_seq_kvlen_list is ", cu_seq_kvlen_list)
    
    pttype = torch.bfloat16
    limit = 2
    q = limit * (torch.rand([total_q, nheads, headdim]) - 0.5).to(pttype)
    k = limit * (torch.rand([total_k, nheads_k, headdim]) - 0.5).to(pttype)
    v = limit * (torch.rand([total_k, nheads_k, headdim]) - 0.5).to(pttype)
    dout = limit * (torch.rand([total_q, nheads, headdim]) - 0.5).to(pttype)
    print("q.shape ", q.shape)
    print("k.shape ", k.shape)
    print("v.shape ", v.shape)
    print("dout.shape ", dout.shape)

    
    # npu attention mask args
    pre_tocken = 65536
    next_tocken = 0
    sparse_mode = 3

    # gpu attention mask args
    causal_switch = True
    window_left = 65536
    window_right = 0
    window_left = -1
    window_right = -1

    # call npu_fusion_attention golden
    q = q.npu()
    k = k.npu()
    v = v.npu()
    dout = dout.npu()
    atten_mask_npu = (torch.triu(torch.ones([2048, 2048]), diagonal=1)).to(torch.bool).npu()

    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    torch.npu.synchronize()
    # TODO golden replace
    npu_rst = torch_npu.npu_fusion_attention(
        q, k, v, nheads,
        pse=None,
        padding_mask=None,
        atten_mask=atten_mask_npu,
        scale=scale,
        keep_prob=keep_prob,
        input_layout="TND",
        actual_seq_qlen=tuple(cu_seq_len_list),
        actual_seq_kvlen=tuple(cu_seq_kvlen_list),
        pre_tockens=pre_tocken,
        next_tockens=next_tocken,
        inner_precise=0,
        sparse_mode=sparse_mode,
        prefix=None
    )
    out_npu = npu_rst[0]
    x_max_npu = npu_rst[1]
    x_sum_npu = npu_rst[2]
    torch.npu.synchronize()
    out_npu.backward(dout)
    dq_golden_npu = q.grad
    dk_golden_npu = k.grad
    dv_golden_npu = v.grad
    torch.npu.synchronize()

    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32).npu()
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32).npu()

    # call tridao npu interface
    print("cu_seqlens_q is ", cu_seqlens_q)
    print("cu_seqlens_k is ", cu_seqlens_k)

    q_fa = q.detach().clone()
    k_fa = k.detach().clone()
    v_fa = v.detach().clone()

    q_fa.requires_grad = True
    k_fa.requires_grad = True
    v_fa.requires_grad = True

    out_fa = flash_attn_varlen_func(
        q_fa, k_fa, v_fa,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=(1 - keep_prob),
        softmax_scale=scale,
        causal=causal_switch,
        window_size=(window_left, window_right),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        block_table=None
    )
    (
        dq_tridao,
        dk_tridao,
        dv_tridao,
    ) = torch.autograd.grad(out_fa, (q_fa, k_fa, v_fa), dout)
    torch.npu.synchronize()

    from precision_compare import data_compare
    data_compare(dq_tridao.cpu().float().numpy(), dq_golden_npu.cpu().float().numpy())
    data_compare(dk_tridao.cpu().float().numpy(), dk_golden_npu.cpu().float().numpy())
    data_compare(dv_tridao.cpu().float().numpy(), dv_golden_npu.cpu().float().numpy())

def test_bsnd_bwd_npu(nheads, nheads_k, headdim, batch, seq_q, seq_k):
    g = nheads / nheads_k
    scale = 1 / (headdim ** 0.5)

    pttype = torch.bfloat16
    limit = 2
    q = limit * (torch.rand([batch, seq_q, nheads, headdim]) - 0.5).to(pttype)
    k = limit * (torch.rand([batch, seq_k, nheads, headdim]) - 0.5).to(pttype)
    v = limit * (torch.rand([batch, seq_k, nheads, headdim]) - 0.5).to(pttype)
    dout = limit * (torch.rand([batch, seq_q, nheads, headdim]) - 0.5).to(pttype)
    print("q.shape ", q.shape)
    print("k.shape ", k.shape)
    print("v.shape ", v.shape)
    print("dout.shape ", dout.shape)
    keep_prob = 1.0
    
    # npu attention mask args
    pre_tocken = 65536
    next_tocken = 0
    sparse_mode = 3

    # gpu attention mask args
    causal_switch = True
    window_left = 65536
    window_right = 0
    window_left = -1
    window_right = -1

    # call npu_fusion_attention golden
    q = q.npu()
    k = k.npu()
    v = v.npu()
    dout = dout.npu()
    atten_mask_npu = (torch.triu(torch.ones([2048, 2048]), diagonal=1)).to(torch.bool).npu()

    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    torch.npu.synchronize()
    # TODO golden replace
    npu_rst = torch_npu.npu_fusion_attention(
        q, k, v, nheads,
        pse=None,
        padding_mask=None,
        atten_mask=atten_mask_npu,
        scale=scale,
        keep_prob=keep_prob,
        input_layout="BSND",
        pre_tockens=pre_tocken,
        next_tockens=next_tocken,
        inner_precise=0,
        sparse_mode=sparse_mode,
        prefix=None
    )
    out_npu = npu_rst[0]
    x_max_npu = npu_rst[1]
    x_sum_npu = npu_rst[2]
    torch.npu.synchronize()
    out_npu.backward(dout)
    dq_golden_npu = q.grad
    dk_golden_npu = k.grad
    dv_golden_npu = v.grad
    torch.npu.synchronize()

    q_fa = q.detach().clone()
    k_fa = k.detach().clone()
    v_fa = v.detach().clone()

    q_fa.requires_grad = True
    k_fa.requires_grad = True
    v_fa.requires_grad = True

    out_fa = flash_attn_func(
        q_fa, k_fa, v_fa,
        dropout_p=(1 - keep_prob),
        softmax_scale=scale,
        causal=causal_switch,
        window_size=(window_left, window_right),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False
    )
    (
        dq_tridao,
        dk_tridao,
        dv_tridao,
    ) = torch.autograd.grad(out_fa, (q_fa, k_fa, v_fa), dout)
    torch.npu.synchronize()

    from precision_compare import data_compare
    data_compare(dq_tridao.cpu().float().numpy(), dq_golden_npu.cpu().float().numpy())
    data_compare(dk_tridao.cpu().float().numpy(), dk_golden_npu.cpu().float().numpy())
    data_compare(dv_tridao.cpu().float().numpy(), dv_golden_npu.cpu().float().numpy())


test_tnd_bwd_npu(16, 1, 128, [512, 33, 1111])
test_bsnd_bwd_npu(16, 4, 128, 1, 999, 999)
