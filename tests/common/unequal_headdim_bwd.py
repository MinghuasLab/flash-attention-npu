# Copyright (c) 2026, Minghua Shen.
"""Shared unequal-headdim backward cases and runners for FA3/FA4 on 910 and 950.

One shape matrix covers both chips. Host launch dispatches on the visible NPU:
910 pads via ``_pad_bwd_headdim`` then the 910 interface; 950 lets
``flash_attn_npu_interface_950`` pad inside ``_flash_attn_backward``. Gradient
checks use the repo ``assert_fa_close`` dual-reference rule on both.

``segment`` / ``check`` / ``TOLERANCES`` remain for optional MLA helper reuse.
"""

from __future__ import annotations

import importlib
import os

import torch

from tests.common import attention_ref, test_utils
from tests.common.attention_ref import (
    cached_autograd_grads,
    ref_flash_attention,
    ref_flash_attention_pair,
)
from tests.common.compare import assert_fa_close
from tests.common.golden_cache import get_or_compute_golden
from tests.common.test_utils import (
    make_attention_inputs,
    make_cu_seqlens,
    make_golden_attention_mask,
    make_packed_random_tensor,
    make_padded_varlen_mask,
    make_random_tensor,
    pad_packed_tensor,
)

# Frozen ratio check used by Ascend 950 MLA helper callers.
TOLERANCES = {torch.float16: (0.00195, 0.1), torch.bfloat16: (0.0156, 1.0)}

# dtype, causal, num_heads, kv_heads, batch, q_seqlen, kv_seqlen, qk_headdim, v_headdim
# Lean matrix: one or two rows per pad path (same-bucket, MLA, equal-pad, >256).
BSND_CASES = [
    # Same 64-bucket pad.
    (torch.float16, True, 4, 4, 2, 33, 65, 21, 5),
    (torch.bfloat16, False, 4, 1, 2, 17, 40, 60, 20),
    # MLA tile: qk bucket 128/192 with v <= 128.
    (torch.float16, False, 4, 4, 2, 32, 48, 192, 128),
    (torch.bfloat16, True, 6, 2, 2, 40, 70, 192, 64),
    (torch.float16, True, 4, 4, 2, 16, 32, 128, 64),
    # Equal-headdim pad (qk > v but not MLA).
    (torch.bfloat16, True, 2, 2, 2, 16, 32, 193, 192),
    (torch.float16, False, 4, 4, 2, 33, 65, 165, 22),
    (torch.bfloat16, False, 4, 4, 2, 24, 48, 129, 128),
    # Above 256: Aligned512 equal-pad.
    (torch.float16, False, 2, 2, 2, 16, 32, 300, 128),
    (torch.float16, False, 4, 4, 2, 8, 16, 512, 256),
]

# dtype, causal, num_heads, kv_heads, seqlens_q, seqlens_k, qk_headdim, v_headdim
TND_CASES = [
    (torch.float16, True, 4, 4, (17, 33), (40, 65), 21, 5),
    (torch.bfloat16, True, 4, 1, (16, 32), (32, 48), 192, 128),
    (torch.float16, True, 4, 4, (8, 16), (16, 32), 128, 64),
    (torch.bfloat16, True, 2, 2, (16, 32), (32, 64), 256, 193),
    (torch.float16, False, 2, 2, (8, 16), (16, 32), 384, 128),
]

# Softcap / sliding-window: one MLA and one equal-pad row.
# (dtype, causal, nheads, kv_heads, batch, sq, sk, qk, v, wl, wr, softcap, scale_or_none)
WINDOW_SOFTCAP_CASES = [
    (torch.float16, False, 4, 2, 1, 65, 97, 128, 64, 31, 17, 0.0, 0.11),
    (torch.float16, True, 4, 2, 1, 65, 97, 192, 128, -1, -1, 0.0, 0.14467962580268923),
]

PADDED_OUT_DIMS = [(31, 14), (186, 140)]

# Public autograd smoke dims (950 forward accepts unequal headdim; 910 skips).
PUBLIC_AUTOGRAD_DIMS = [(96, 64), (192, 128)]


def device_kind() -> str:
    name = torch.npu.get_device_name() if torch.npu.is_available() else ""
    if "950" in name:
        return "950"
    if "Ascend910" in name:
        return "910"
    return "other"


def iface(api: str):
    """Load the FA3/FA4 interface for the visible NPU. Lazy to keep collect safe."""
    # CI groups by literal flash_attn_npu_3 / flash_attn_npu_4 markers.
    ver = 3 if api == "v3" else 4
    suffix = "_950" if device_kind() == "950" else ""
    return importlib.import_module(f"flash_attn_npu_{ver}.flash_attn_npu_interface{suffix}")


def run_backward(
    api,
    dout,
    q,
    k,
    v,
    out,
    lse,
    scale,
    causal,
    cu_q=None,
    cu_k=None,
    max_q=None,
    max_k=None,
    window_size_left=-1,
    window_size_right=-1,
    softcap=0.0,
):
    """Match FlashAttnFunc.backward on the visible chip; slice grads to raw widths."""
    mod = iface(api)
    kind = device_kind()
    qk_og, v_og = q.size(-1), v.size(-1)
    if kind == "910":
        dout, q, k, v, out, qk_og, v_og = mod._pad_bwd_headdim(dout, q, k, v, out, qk_og)
        wl = window_size_left
        wr = window_size_right
        if api == "v3":
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            mod._flash_attn_backward(
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
                window_size_left=wl,
                window_size_right=wr,
                softcap=softcap,
                deterministic=False,
                original_qk_gt_v=qk_og > v_og,
                sm_margin=0,
            )
        else:
            dq, dk, dv = mod._flash_attn_backward(
                q,
                k,
                v,
                out,
                dout,
                lse,
                softmax_scale=scale,
                causal=causal,
                window_size_left=None if wl < 0 else wl,
                window_size_right=None if wr < 0 else wr,
                softcap=softcap,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=max_q,
                max_seqlen_k=max_k,
                deterministic=False,
                original_qk_gt_v=qk_og > v_og,
            )
        return dq[..., :qk_og], dk[..., :qk_og], dv[..., :v_og]

    # 950 pads inside the interface.
    if api == "v3":
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        mod._flash_attn_backward(
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
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            softcap=softcap,
            deterministic=False,
            sm_margin=0,
        )
        return dq, dk, dv
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    returned = mod._flash_attn_backward(
        q,
        k,
        v,
        out,
        dout,
        lse,
        softmax_scale=scale,
        causal=causal,
        softcap=softcap,
        window_size_left=None if window_size_left < 0 else window_size_left,
        window_size_right=None if window_size_right < 0 else window_size_right,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        deterministic=False,
        dq=dq,
        dk=dk,
        dv=dv,
    )
    assert all(got is buf for got, buf in zip(returned, (dq, dk, dv)))
    return dq, dk, dv


def check_grads(
    api, label, query, key, value, dout, out_ref, out_pt, lse_ref, reference_pair, refs, **launch
):
    scale = launch.pop("scale", None)
    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)
    softcap = launch.get("softcap", 0.0)
    out = out_ref.detach().to(device=query.device, dtype=query.dtype).contiguous()
    lse = lse_ref.detach().to(device=query.device, dtype=torch.float32).contiguous()
    dq, dk, dv = run_backward(api, dout, query, key, value, out, lse, scale, **launch)
    grads = cached_autograd_grads(
        os.environ.get("GOLDEN_CACHE_NODEID", label),
        (out_ref.detach(), out_pt.detach()),
        refs,
        dout.detach().cpu(),
        metadata={"kind": label, "api": api},
        rebuild=reference_pair,
    )
    for name, actual, ref, pt in zip(("dQ", "dK", "dV"), (dq, dk, dv), grads[0:3], grads[3:6]):
        assert_fa_close(actual, ref, pt, softcap=softcap, name=f"{label}/{name}")


def bsnd_pair(query, key, value, scale, mask, dtype, softcap=0.0):
    def reference_pair(query_ref, key_ref, value_ref, differentiable=False):
        out_ref, lse_ref, out_pt, _ = ref_flash_attention_pair(
            query_ref,
            key_ref,
            value_ref,
            scale,
            mask,
            dtype,
            softcap=softcap,
            differentiable=differentiable,
        )
        return out_ref, out_pt, lse_ref

    refs = tuple(tensor.detach().cpu().requires_grad_(True) for tensor in (query, key, value))
    out_ref, out_pt, lse_ref = reference_pair(*refs)
    return reference_pair, refs, out_ref, out_pt, lse_ref


def pack_lse(lse, q_valid):
    """(B, H, S) golden LSE -> (H, total_q) layout read by varlen backward."""
    return lse.permute(0, 2, 1)[q_valid].transpose(0, 1).contiguous()


def tnd_pair(query, key, value, seqlens_q, seqlens_k, scale, mask, q_valid, dtype, softcap=0.0):
    def reference_pair(query_ref, key_ref, value_ref, differentiable=False):
        out_ref, lse_ref, out_pt, _ = ref_flash_attention_pair(
            pad_packed_tensor(query_ref, seqlens_q, max(seqlens_q)),
            pad_packed_tensor(key_ref, seqlens_k, max(seqlens_k)),
            pad_packed_tensor(value_ref, seqlens_k, max(seqlens_k)),
            scale,
            mask,
            dtype,
            softcap=softcap,
            differentiable=differentiable,
        )
        return out_ref[q_valid], out_pt[q_valid], pack_lse(lse_ref, q_valid)

    refs = tuple(tensor.detach().cpu().requires_grad_(True) for tensor in (query, key, value))
    out_ref, out_pt, lse_ref = reference_pair(*refs)
    return reference_pair, refs, out_ref, out_pt, lse_ref


def run_bsnd_case(
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
    window_size_left=-1,
    window_size_right=-1,
    softcap=0.0,
    scale=None,
):
    query, key, value, dout = make_attention_inputs(
        (batch, q_seqlen, num_heads, qk_headdim),
        (batch, kv_seqlen, kv_heads, qk_headdim),
        (batch, kv_seqlen, kv_heads, v_headdim),
        (batch, q_seqlen, num_heads, v_headdim),
        data_type,
        device="npu",
    )
    if scale is None:
        scale = 1.0 / (qk_headdim**0.5)
    mask, _, _ = make_golden_attention_mask(
        q_seqlen, kv_seqlen, causal, window_size_left, window_size_right
    )
    reference_pair, refs, out_ref, out_pt, lse_ref = bsnd_pair(
        query, key, value, scale, mask, data_type, softcap=softcap
    )
    check_grads(
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
        scale=scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        softcap=softcap,
    )


def run_tnd_case(
    api, data_type, causal, num_heads, kv_heads, seqlens_q, seqlens_k, qk_headdim, v_headdim
):
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
    reference_pair, refs, out_ref, out_pt, lse_ref = tnd_pair(
        query, key, value, seqlens_q, seqlens_k, scale, mask, q_valid, data_type
    )
    dout = make_random_tensor(tuple(out_ref.shape), data_type, low=-1.0, high=1.0, device="npu")
    check_grads(
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
        scale=scale,
        cu_q=make_cu_seqlens(seqlens_q).npu(),
        cu_k=make_cu_seqlens(seqlens_k).npu(),
        max_q=max_q,
        max_k=max_k,
    )


def run_padded_out_dependence(api, layout, qk_dim, v_dim):
    """The autograd padding route must retain O dependence in every equal tile."""
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
    original = run_backward(api, dout, q, k, v, out, lse, scale, False, **kwargs)
    changed = run_backward(api, dout, q, k, v, changed_out, lse, scale, False, **kwargs)
    assert not torch.equal(original[0], changed[0]), "padding discarded actual-O backward semantics"
    torch.testing.assert_close(original[2], changed[2], atol=0, rtol=0)


def run_public_autograd(api, dtype, qdim, vdim, layout):
    """Public ``flash_attn_*_func`` unequal backward. 910 forward still needs equal dims."""
    if device_kind() != "950":
        raise RuntimeError("public unequal autograd is Ascend 950 only")
    mod = iface(api)
    scale = qdim ** (-0.5)
    values, refs = segment(qdim, vdim, dtype, 65, 97, True, -1, -1, 0.0, scale)
    q, k, v, _, do, _ = values
    if layout == "TND":
        q, k, v, do = [x.squeeze(0) for x in (q, k, v, do)]
        refs = [x.squeeze(0) for x in refs]
    inputs = [x.npu().detach().requires_grad_() for x in (q, k, v)]
    if layout == "TND":
        cuq = torch.tensor([0, 65], dtype=torch.int32, device="npu")
        cuk = torch.tensor([0, 97], dtype=torch.int32, device="npu")
        result = mod.flash_attn_varlen_func(
            *inputs,
            cu_seqlens_q=cuq,
            cu_seqlens_k=cuk,
            max_seqlen_q=65,
            max_seqlen_k=97,
            causal=True,
        )
    else:
        result = mod.flash_attn_func(*inputs, causal=True)
    if isinstance(result, tuple):
        result = result[0]
    result.backward(do.npu())
    torch.npu.synchronize()
    assert result.shape == do.shape
    for inp, ref in zip(inputs, refs):
        assert inp.grad.shape == inp.shape
        check(inp.grad, ref, dtype)


def segment(qdim, vdim, dtype, sq, sk, causal, wl, wr, softcap, scale, heads=4, kv_heads=2):
    """CPU golden segment used by MLA pipeline tests and public-autograd smoke."""
    inputs = [
        torch.randn(1, sq, heads, qdim, dtype=dtype),
        torch.randn(1, sk, kv_heads, qdim, dtype=dtype),
        torch.randn(1, sk, kv_heads, vdim, dtype=dtype),
    ]
    do = torch.randn(1, sq, heads, vdim, dtype=dtype)
    names = ("out", "lse", "dq", "dk", "dv")

    def compute():
        xs = [x.float().requires_grad_() for x in inputs]
        mask = make_golden_attention_mask(sq, sk, causal, wl, wr)[0]
        out, lse = ref_flash_attention(*xs, scale, mask, torch.float32, softcap=softcap)
        refs = torch.autograd.grad(out, xs, do.float())
        return dict(zip(names, (out.detach().to(dtype), lse.detach(), *refs)))

    values = get_or_compute_golden(
        nodeid=os.environ.get("GOLDEN_CACHE_NODEID", "standalone"),
        api_name="unequal_headdim_backward.segment",
        metadata={
            "scale": scale,
            "causal": causal,
            "window": (wl, wr),
            "softcap": softcap,
            "cpu_threads": torch.get_num_threads(),
            "matmul_precision": torch.get_float32_matmul_precision(),
        },
        inputs=(*inputs, do),
        compute_fn=compute,
        expected_keys=names,
        source_files=(__file__, attention_ref.__file__, test_utils.__file__),
    )
    return (*inputs, values["out"], do, values["lse"]), tuple(values[name] for name in names[2:])


_segment = segment


def check(actual, expected, dtype):
    tolerance, max_abs_limit = TOLERANCES[dtype]
    diff = (actual.cpu().float() - expected.detach()).abs()
    matched = diff <= tolerance + tolerance * expected.detach().abs()
    assert torch.isfinite(actual).all()
    assert matched.float().mean().item() >= 0.999
    assert diff.max().item() < max_abs_limit


_check = check
