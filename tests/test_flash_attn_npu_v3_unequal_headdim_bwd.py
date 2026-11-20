# Copyright (c) 2026, Minghua Shen.
"""FA3 unequal-headdim backward on Ascend 910 and Ascend 950.

Shared shape matrix and runners live in ``tests.common.unequal_headdim_bwd``.
FA4 twin: ``test_flash_attn_npu_v4_unequal_headdim_bwd.py``.
"""

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
if torch_npu.npu.device_count() == 0:
    pytest.skip("unequal-headdim backward requires an Ascend NPU", allow_module_level=True)

from tests.common import unequal_headdim_bwd as uh

API = "v3"
_KIND = uh.device_kind()
if _KIND not in ("910", "950"):
    pytest.skip("unequal-headdim backward requires Ascend 910 or 950", allow_module_level=True)


@pytest.mark.golden_api("flash_attn_func")
@pytest.mark.parametrize(
    ("data_type, causal, num_heads, kv_heads, batch, q_seqlen, kv_seqlen, qk_headdim, v_headdim"),
    uh.BSND_CASES,
    ids=[
        f"{'fp16' if dtype == torch.float16 else 'bf16'}-c{int(causal)}-{qk}x{vd}"
        for dtype, causal, _, _, _, _, _, qk, vd in uh.BSND_CASES
    ],
)
def test_unequal_headdim_bwd_bsnd(
    data_type, causal, num_heads, kv_heads, batch, q_seqlen, kv_seqlen, qk_headdim, v_headdim
):
    uh.run_bsnd_case(
        API, data_type, causal, num_heads, kv_heads, batch, q_seqlen, kv_seqlen, qk_headdim, v_headdim
    )


@pytest.mark.golden_api("flash_attn_varlen_func")
@pytest.mark.parametrize(
    "data_type, causal, num_heads, kv_heads, seqlens_q, seqlens_k, qk_headdim, v_headdim",
    uh.TND_CASES,
    ids=[
        f"{'fp16' if dtype == torch.float16 else 'bf16'}-c{int(causal)}-{qk}x{vd}"
        for dtype, causal, _, _, _, _, qk, vd in uh.TND_CASES
    ],
)
def test_unequal_headdim_bwd_tnd(
    data_type, causal, num_heads, kv_heads, seqlens_q, seqlens_k, qk_headdim, v_headdim
):
    uh.run_tnd_case(
        API, data_type, causal, num_heads, kv_heads, seqlens_q, seqlens_k, qk_headdim, v_headdim
    )


@pytest.mark.golden_api("flash_attn_func")
@pytest.mark.parametrize(
    "data_type, causal, num_heads, kv_heads, batch, q_seqlen, kv_seqlen, qk_headdim, v_headdim, "
    "wl, wr, softcap, scale",
    uh.WINDOW_SOFTCAP_CASES,
    ids=[
        f"{'fp16' if dtype == torch.float16 else 'bf16'}-c{int(causal)}-{qk}x{vd}-w{wl}_{wr}-sc{softcap}"
        for dtype, causal, _, _, _, _, _, qk, vd, wl, wr, softcap, _ in uh.WINDOW_SOFTCAP_CASES
    ],
)
def test_unequal_headdim_bwd_window_softcap(
    data_type,
    causal,
    num_heads,
    kv_heads,
    batch,
    q_seqlen,
    kv_seqlen,
    qk_headdim,
    v_headdim,
    wl,
    wr,
    softcap,
    scale,
):
    uh.run_bsnd_case(
        API,
        data_type,
        causal,
        num_heads,
        kv_heads,
        batch,
        q_seqlen,
        kv_seqlen,
        qk_headdim,
        v_headdim,
        window_size_left=wl,
        window_size_right=wr,
        softcap=softcap,
        scale=scale,
    )


@pytest.mark.parametrize("layout", ["BSND", "TND"])
@pytest.mark.parametrize("dims", uh.PADDED_OUT_DIMS)
def test_padded_unequal_backward_consumes_actual_out(layout, dims):
    uh.run_padded_out_dependence(API, layout, *dims)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("qdim,vdim", uh.PUBLIC_AUTOGRAD_DIMS)
@pytest.mark.parametrize("layout", ["BSND", "TND"])
def test_public_autograd_unequal_headdim(dtype, qdim, vdim, layout):
    if _KIND != "950":
        pytest.skip("910 forward still requires equal headdims")
    uh.run_public_autograd(API, dtype, qdim, vdim, layout)
