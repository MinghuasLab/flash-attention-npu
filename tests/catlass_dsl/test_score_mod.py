"""Fixed-layout score callbacks and their explicit device VJPs."""

import catlass.tla as tla
import pytest
from .score_mod_definitions import get_score_pair, make_score_aux
from .test_utils import DTYPES, run

from flash_attn_npu_4.catlass_dsl.mods import Mod

pytestmark = pytest.mark.npu


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "name", ("identity", "causal", "alibi", "squared", "batch_bias", "dual_buffer")
)
def test_score_mod_forward_backward(device, dtype, name):
    score, backward = get_score_pair(name)
    run(device, dtype, score=score, bwd=backward, aux=make_score_aux(name))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "name",
    ("rel_bias_clamped", "sliding_window", "block_diagonal", "stress_multi_buffer"),
)
def test_score_boundaries(device, dtype, name):
    score, backward = get_score_pair(name)
    run(
        device,
        dtype,
        score=score,
        bwd=backward,
        aux=make_score_aux(name, length=641),
        sq=641,
        sk=577,
    )


@tla.jit
def _constant_score(s, b, h, q, k, info, aux, scalars):
    return tla.Float32(2.0)


@tla.jit
def _constant_bwd(g, s, b, h, q, k, info, aux, scalars):
    return tla.Float32(0.0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_constant_score_zero_qk_gradients(device, dtype):
    forward = Mod(_constant_score, torch_ref=lambda s, b, h, q, k, info, aux, scalars: 2.0)
    run(device, dtype, score=forward, bwd=Mod(_constant_bwd))
