# Copyright (c) 2026, Minghua Shen.
"""Backward precision when QK headdim is strictly larger than V headdim.

The public v3/v4 backward accepts qk headdim in [1, 512] with a strictly smaller
v headdim. Widths up to 256 pad to a multiple of 64; widths above 256 pad to a
multiple of 16. Then:

* qk bucket 192 with v <= 128 keeps v at 128 and uses the MLA kernel
* every other pair (for example 21/5, 80/64, 193/192, 300/128, 512/256) pads v
  up to the qk bucket and uses the equal-headdim kernel, including Aligned512
  when the padded width is above 256

Forward still requires equal headdims, so each case builds ``out`` and ``lse``
from the CPU golden and calls the same host padding plus backward entry that
``FlashAttnFunc.backward`` uses.

Each row below is one discrete shape. v3 and v4, BSND and TND, fp16 and bf16,
causal on and off, and MHA/GQA are all represented.
"""

import os

import pytest
import torch
import torch_npu

import flash_attn_npu_3.flash_attn_npu_interface as iface_v3
import flash_attn_npu_4.flash_attn_npu_interface as iface_v4
from tests.common.attention_ref import cached_autograd_grads, ref_flash_attention_pair
from tests.common.compare import assert_fa_close
from tests.common.test_utils import (
    make_attention_inputs,
    make_cu_seqlens,
    make_golden_attention_mask,
    make_packed_random_tensor,
    make_padded_varlen_mask,
    make_random_tensor,
    pad_packed_tensor,
)

# api, dtype, causal, num_heads, kv_heads, batch, q_seqlen, kv_seqlen, qk_headdim, v_headdim
_BSND_CASES = [
    # Same 64-bucket. These used to miss the MLA kernel when launched unpadded.
    ("v3", torch.float16, True, 4, 4, 2, 33, 65, 21, 5),
    ("v4", torch.float16, True, 4, 4, 2, 33, 65, 21, 5),
    ("v3", torch.bfloat16, False, 4, 1, 2, 17, 40, 60, 20),
    ("v4", torch.bfloat16, False, 4, 1, 2, 17, 40, 60, 20),
    ("v3", torch.bfloat16, False, 2, 2, 2, 64, 64, 57, 41),
    ("v4", torch.float16, True, 2, 1, 2, 64, 80, 38, 10),
    ("v3", torch.float16, False, 4, 4, 2, 1, 8, 64, 63),
    ("v4", torch.bfloat16, True, 4, 2, 2, 65, 129, 7, 3),
    ("v3", torch.float16, True, 8, 2, 1, 9, 33, 32, 16),
    ("v4", torch.bfloat16, False, 4, 4, 2, 128, 128, 63, 1),
    ("v3", torch.bfloat16, True, 2, 2, 2, 16, 32, 193, 192),
    ("v4", torch.float16, False, 2, 1, 2, 16, 32, 200, 193),
    ("v3", torch.float16, False, 2, 2, 2, 32, 48, 255, 200),
    ("v4", torch.bfloat16, True, 2, 2, 2, 32, 64, 256, 193),
    # MLA: QK bucket is 128 or 192 and strictly larger than the V bucket.
    ("v3", torch.float16, False, 4, 4, 2, 32, 48, 192, 128),
    ("v4", torch.bfloat16, True, 4, 1, 2, 32, 48, 192, 128),
    ("v3", torch.bfloat16, True, 6, 2, 2, 40, 70, 192, 64),
    ("v4", torch.float16, False, 6, 3, 2, 40, 70, 192, 64),
    ("v3", torch.float16, True, 4, 4, 2, 16, 32, 128, 64),
    ("v4", torch.bfloat16, False, 4, 1, 2, 16, 32, 128, 64),
    ("v3", torch.float16, False, 4, 4, 2, 33, 65, 165, 22),
    ("v4", torch.bfloat16, True, 4, 2, 2, 33, 65, 93, 27),
    ("v3", torch.bfloat16, False, 2, 2, 2, 128, 160, 69, 26),
    ("v4", torch.float16, True, 2, 1, 2, 128, 160, 100, 17),
    ("v3", torch.float16, True, 4, 1, 2, 48, 96, 80, 16),
    ("v4", torch.bfloat16, False, 4, 4, 2, 24, 48, 129, 128),
    ("v3", torch.bfloat16, False, 4, 4, 2, 65, 129, 65, 1),
    ("v4", torch.float16, True, 4, 2, 2, 65, 129, 128, 1),
    ("v3", torch.float16, True, 1, 1, 2, 1, 1, 2, 1),
    ("v4", torch.bfloat16, False, 1, 1, 2, 3, 5, 2, 1),
    # Above 256: 16-aligned qk, v padded up to that bucket, Aligned512 kernel.
    ("v3", torch.float16, False, 2, 2, 2, 16, 32, 300, 128),
    ("v4", torch.bfloat16, True, 2, 1, 2, 16, 32, 320, 64),
    ("v3", torch.bfloat16, True, 4, 2, 2, 8, 16, 400, 192),
    ("v4", torch.float16, False, 4, 4, 2, 8, 16, 512, 256),
    ("v3", torch.float16, False, 2, 1, 2, 4, 8, 257, 1),
    ("v4", torch.bfloat16, True, 2, 2, 2, 4, 8, 511, 496),
]

# api, dtype, causal, num_heads, kv_heads, seqlens_q, seqlens_k, qk_headdim, v_headdim
_TND_CASES = [
    ("v3", torch.float16, True, 4, 4, (17, 33), (40, 65), 21, 5),
    ("v4", torch.bfloat16, False, 4, 1, (8, 17), (20, 40), 60, 20),
    ("v3", torch.bfloat16, False, 2, 2, (1, 9), (8, 33), 32, 16),
    ("v4", torch.float16, True, 4, 2, (64, 65), (80, 129), 7, 3),
    ("v3", torch.float16, False, 4, 4, (1, 64), (8, 64), 64, 63),
    ("v4", torch.bfloat16, True, 2, 2, (16, 32), (32, 64), 256, 193),
    ("v3", torch.float16, False, 2, 2, (8, 16), (16, 32), 384, 128),
    ("v4", torch.bfloat16, True, 4, 1, (4, 8), (8, 16), 512, 192),
    ("v3", torch.bfloat16, True, 4, 1, (16, 32), (32, 48), 192, 128),
    ("v4", torch.float16, False, 6, 2, (24, 40), (40, 70), 192, 64),
    ("v3", torch.float16, True, 4, 4, (8, 16), (16, 32), 128, 64),
    ("v4", torch.bfloat16, False, 4, 2, (33, 48), (65, 96), 165, 22),
    ("v3", torch.bfloat16, False, 2, 1, (64, 128), (128, 160), 100, 17),
    ("v4", torch.float16, True, 4, 4, (24, 65), (48, 129), 129, 64),
    ("v3", torch.float16, False, 1, 1, (1, 3), (1, 5), 2, 1),
    ("v4", torch.bfloat16, True, 8, 2, (9, 33), (33, 40), 57, 41),
]


def _require_npu():
    if torch_npu.npu.device_count() <= 0:
        pytest.skip("no visible NPU")
    name = torch_npu.npu.get_device_name()
    if "Ascend910" not in name and "Ascend950" not in name:
        pytest.skip(f"unequal-headdim backward is not enabled on {name}")


def _run_backward(
    api, dout, q, k, v, out, lse, scale, causal, cu_q=None, cu_k=None, max_q=None, max_k=None
):
    """Match FlashAttnFunc.backward: pad headdim, launch, slice grads back."""
    iface = iface_v3 if api == "v3" else iface_v4
    dout, q, k, v, out, qk_og, v_og = iface._pad_bwd_headdim(dout, q, k, v, out, q.size(-1))
    if api == "v3":
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        iface._flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            dq=dq,
            dk=dk,
            dv=dv,
            softmax_scale=scale,
            is_causal=causal,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            deterministic=False,
            original_qk_gt_v=qk_og > v_og,
            sm_margin=0,
        )
    else:
        dq, dk, dv = iface._flash_attn_backward(
            q,
            k,
            v,
            out,
            dout,
            lse,
            softmax_scale=scale,
            causal=causal,
            window_size_left=None,
            window_size_right=None,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            deterministic=False,
            original_qk_gt_v=qk_og > v_og,
        )
    return dq[..., :qk_og], dk[..., :qk_og], dv[..., :v_og]


def _check_grads(
    api, label, query, key, value, dout, out_ref, out_pt, lse_ref, reference_pair, refs, **launch
):
    scale = 1.0 / (query.shape[-1] ** 0.5)
    out = out_ref.detach().to(device=query.device, dtype=query.dtype).contiguous()
    lse = lse_ref.detach().to(device=query.device, dtype=torch.float32).contiguous()
    dq, dk, dv = _run_backward(api, dout, query, key, value, out, lse, scale, **launch)
    grads = cached_autograd_grads(
        os.environ.get("GOLDEN_CACHE_NODEID", label),
        (out_ref.detach(), out_pt.detach()),
        refs,
        dout.detach().cpu(),
        metadata={"kind": label, "api": api},
        rebuild=reference_pair,
    )
    for name, actual, ref, pt in zip(("dQ", "dK", "dV"), (dq, dk, dv), grads[0:3], grads[3:6]):
        assert_fa_close(actual, ref, pt, name=f"{label}/{name}")


def _bsnd_pair(query, key, value, scale, mask, dtype):
    def reference_pair(query_ref, key_ref, value_ref, differentiable=False):
        out_ref, lse_ref, out_pt, _ = ref_flash_attention_pair(
            query_ref,
            key_ref,
            value_ref,
            scale,
            mask,
            dtype,
            differentiable=differentiable,
        )
        return out_ref, out_pt, lse_ref

    refs = tuple(tensor.detach().cpu().requires_grad_(True) for tensor in (query, key, value))
    out_ref, out_pt, lse_ref = reference_pair(*refs)
    return reference_pair, refs, out_ref, out_pt, lse_ref


@pytest.mark.golden_api("flash_attn_func")
@pytest.mark.parametrize(
    (
        "api, data_type, causal, num_heads, kv_heads, batch, "
        "q_seqlen, kv_seqlen, qk_headdim, v_headdim"
    ),
    _BSND_CASES,
    ids=[
        f"{api}-{'fp16' if dtype == torch.float16 else 'bf16'}-c{int(causal)}-{qk}x{vd}"
        for api, dtype, causal, _, _, _, _, _, qk, vd in _BSND_CASES
    ],
)
def test_unequal_headdim_bwd_bsnd(
    api,
    data_type,
    causal,
    num_heads,
    kv_heads,
    batch,
    q_seqlen,
    kv_seqlen,
    qk_headdim,
    v_headdim,
):
    _require_npu()
    query, key, value, dout = make_attention_inputs(
        (batch, q_seqlen, num_heads, qk_headdim),
        (batch, kv_seqlen, kv_heads, qk_headdim),
        (batch, kv_seqlen, kv_heads, v_headdim),
        (batch, q_seqlen, num_heads, v_headdim),
        data_type,
        device="npu",
    )
    scale = 1.0 / (qk_headdim**0.5)
    mask, _, _ = make_golden_attention_mask(q_seqlen, kv_seqlen, causal, -1, -1)
    reference_pair, refs, out_ref, out_pt, lse_ref = _bsnd_pair(
        query, key, value, scale, mask, data_type
    )
    _check_grads(
        api,
        f"{api}/BSND/qk{qk_headdim}/v{v_headdim}",
        query,
        key,
        value,
        dout,
        out_ref,
        out_pt,
        lse_ref,
        reference_pair,
        refs,
        causal=causal,
    )


def _pack_lse(lse, q_valid):
    """(B, H, S) golden LSE -> (H, total_q) layout read by varlen backward."""
    return lse.permute(0, 2, 1)[q_valid].transpose(0, 1).contiguous()


def _tnd_pair(query, key, value, seqlens_q, seqlens_k, scale, mask, q_valid, dtype):
    def reference_pair(query_ref, key_ref, value_ref, differentiable=False):
        out_ref, lse_ref, out_pt, _ = ref_flash_attention_pair(
            pad_packed_tensor(query_ref, seqlens_q, max(seqlens_q)),
            pad_packed_tensor(key_ref, seqlens_k, max(seqlens_k)),
            pad_packed_tensor(value_ref, seqlens_k, max(seqlens_k)),
            scale,
            mask,
            dtype,
            differentiable=differentiable,
        )
        return out_ref[q_valid], out_pt[q_valid], _pack_lse(lse_ref, q_valid)

    refs = tuple(tensor.detach().cpu().requires_grad_(True) for tensor in (query, key, value))
    out_ref, out_pt, lse_ref = reference_pair(*refs)
    return reference_pair, refs, out_ref, out_pt, lse_ref


@pytest.mark.golden_api("flash_attn_varlen_func")
@pytest.mark.parametrize(
    "api, data_type, causal, num_heads, kv_heads, seqlens_q, seqlens_k, qk_headdim, v_headdim",
    _TND_CASES,
    ids=[
        f"{api}-{'fp16' if dtype == torch.float16 else 'bf16'}-c{int(causal)}-{qk}x{vd}"
        for api, dtype, causal, _, _, _, _, qk, vd in _TND_CASES
    ],
)
def test_unequal_headdim_bwd_tnd(
    api,
    data_type,
    causal,
    num_heads,
    kv_heads,
    seqlens_q,
    seqlens_k,
    qk_headdim,
    v_headdim,
):
    _require_npu()
    seqlens_q = list(seqlens_q)
    seqlens_k = list(seqlens_k)
    max_q = max(seqlens_q)
    max_k = max(seqlens_k)
    query = make_packed_random_tensor(
        seqlens_q, max_q, num_heads, qk_headdim, data_type, device="npu", requires_grad=True
    )
    key = make_packed_random_tensor(
        seqlens_k, max_k, kv_heads, qk_headdim, data_type, device="npu", requires_grad=True
    )
    value = make_packed_random_tensor(
        seqlens_k, max_k, kv_heads, v_headdim, data_type, device="npu", requires_grad=True
    )
    scale = 1.0 / (qk_headdim**0.5)
    q_valid, _, mask = make_padded_varlen_mask(seqlens_q, seqlens_k, max_q, max_k, causal, -1, -1)
    reference_pair, refs, out_ref, out_pt, lse_ref = _tnd_pair(
        query, key, value, seqlens_q, seqlens_k, scale, mask, q_valid, data_type
    )
    dout = make_random_tensor(tuple(out_ref.shape), data_type, low=-1.0, high=1.0, device="npu")
    _check_grads(
        api,
        f"{api}/TND/qk{qk_headdim}/v{v_headdim}",
        query,
        key,
        value,
        dout,
        out_ref,
        out_pt,
        lse_ref,
        reference_pair,
        refs,
        causal=causal,
        cu_q=make_cu_seqlens(seqlens_q).npu(),
        cu_k=make_cu_seqlens(seqlens_k).npu(),
        max_q=max_q,
        max_k=max_k,
    )


@pytest.mark.skipif("Ascend950" in torch_npu.npu.get_device_name(), reason="910 padding semantics")
@pytest.mark.parametrize("api", ["v3", "v4"])
@pytest.mark.parametrize("layout", ["BSND", "TND"])
@pytest.mark.parametrize("dims", [(31, 14), (96, 21), (186, 140), (225, 214)])
def test_padded_unequal_backward_consumes_actual_out(api, layout, dims):
    """The autograd padding route must retain O dependence in every equal tile."""
    qk_dim, v_dim = dims
    q = torch.randn(1, 5, 2, qk_dim, device="npu", dtype=torch.float16) * 0.2
    k = torch.randn_like(q) * 0.2
    v = torch.randn(1, 5, 2, v_dim, device=q.device, dtype=q.dtype) * 0.2
    dout = torch.randn_like(v) * 0.2
    scale = qk_dim**-0.5
    q_cpu, k_cpu, v_cpu = (x.cpu().double().transpose(1, 2) for x in (q, k, v))
    scores = (q_cpu @ k_cpu.transpose(-1, -2)) * scale
    out = (scores.softmax(-1) @ v_cpu).transpose(1, 2).to(device=q.device, dtype=q.dtype)
    lse = scores.logsumexp(-1).to(device=q.device, dtype=torch.float32)
    kwargs = {}
    if layout == "TND":
        q, k, v, out, dout = (x[0] for x in (q, k, v, out, dout))
        lse = lse[0]
        cu = torch.tensor([0, 5], dtype=torch.int32, device=q.device)
        kwargs = dict(cu_q=cu, cu_k=cu, max_q=5, max_k=5)
    changed_out = out.clone()
    changed_out[..., 0] += 1
    original = _run_backward(api, dout, q, k, v, out, lse, scale, False, **kwargs)
    changed = _run_backward(api, dout, q, k, v, changed_out, lse, scale, False, **kwargs)
    assert not torch.equal(original[0], changed[0]), "padding discarded actual-O backward semantics"
    torch.testing.assert_close(original[2], changed[2], atol=0, rtol=0)
