"""Representative local/global masks over packed and mixed layouts."""

import pytest
from .mask_mod_definitions import get_mask_pair, make_mask_aux
from .test_utils import DTYPES, run

pytestmark = pytest.mark.npu


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "name,qpacked,kpacked",
    [
        ("causal", True, True),
        ("document", True, False),
        ("packed_aux", False, True),
        ("global_packed_doc", True, True),
        ("global_ima", False, True),
        ("global_causal_window", True, False),
    ],
)
def test_varlen_masks(device, dtype, name, qpacked, kpacked):
    run(
        device,
        dtype,
        mask=get_mask_pair(name, window_size=7),
        aux=make_mask_aux(name),
        qpacked=qpacked,
        kpacked=kpacked,
    )
