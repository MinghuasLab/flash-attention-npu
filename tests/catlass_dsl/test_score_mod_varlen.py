"""Global score offsets, multiple auxiliary tensors and explicit VJPs."""

import pytest
from .score_mod_definitions import get_score_pair, make_score_aux
from .test_utils import DTYPES, run

pytestmark = pytest.mark.npu


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "name,qpacked,kpacked",
    [
        ("scaled_squared", True, True),
        ("global_q_bias", True, False),
        ("global_kv_bias", False, True),
        ("global_q_and_kv_bias", True, True),
        ("stress_multi_buffer", True, True),
        ("stress_conditional_mask", True, False),
    ],
)
def test_varlen_scores(device, dtype, name, qpacked, kpacked):
    score, backward = get_score_pair(name)
    run(
        device,
        dtype,
        score=score,
        bwd=backward,
        aux=make_score_aux(name),
        qpacked=qpacked,
        kpacked=kpacked,
    )
