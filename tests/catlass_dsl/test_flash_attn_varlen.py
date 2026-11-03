"""Ragged and uniform attention across independently packed Q/KV layouts."""

import pytest
from .test_utils import DTYPES, LAYOUT_IDS, LAYOUTS, check_attention, make_inputs

from flash_attn_npu_4.catlass_dsl import flash_attn_varlen_func

pytestmark = pytest.mark.npu


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("qp,kp", LAYOUTS[1:], ids=LAYOUT_IDS[1:])
@pytest.mark.parametrize("kh", [4, 2, 1], ids=["mha", "gqa", "mqa"])
def test_varlen(device, dtype, qp, kp, kh):
    # Three batches jointly cover empty Q, empty KV, ragged tails and nonzero offsets.
    cpu, inputs, cq, ck, kwargs = make_inputs(
        device,
        dtype,
        qp=qp,
        kp=kp,
        sq=129,
        sk=143,
        kh=kh,
        q_lengths=(0, 17, 129),
        k_lengths=(7, 0, 143),
    )
    out, lse = flash_attn_varlen_func(*inputs, **kwargs, causal=True, return_lse=True)
    mask = lambda b, h, q, k, info, aux, scalars: k <= q + info[1] - info[0]
    check_attention(cpu, inputs, out, lse, cq=cq, ck=ck, mask_mod=mask, loss="both")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("qp,kp", LAYOUTS[1:], ids=LAYOUT_IDS[1:])
def test_uniform_packed(device, dtype, qp, kp):
    cpu, inputs, cq, ck, kwargs = make_inputs(
        device,
        dtype,
        qp=qp,
        kp=kp,
        sq=128,
        sk=128,
        kh=4,
        q_lengths=(128, 128),
        k_lengths=(128, 128),
    )
    out, lse = flash_attn_varlen_func(
        *inputs, **kwargs, max_seqlen_q=256, max_seqlen_k=256, return_lse=True
    )
    check_attention(cpu, inputs, out, lse, cq=cq, ck=ck)
