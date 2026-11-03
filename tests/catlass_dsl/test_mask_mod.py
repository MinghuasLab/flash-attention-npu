"""Fixed-layout mask output and gradient checks."""

import pytest
from .mask_mod_definitions import (
    GLOBAL_MASK_NAMES,
    PARAMETERIZED_MASK_NAMES,
    STATIC_MASK_NAMES,
    get_mask_pair,
    make_mask_aux,
)
from .test_utils import DTYPES, run

pytestmark = pytest.mark.npu
SEQLEN_CONFIGS = ((137, 143),)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("sq,sk", SEQLEN_CONFIGS)
@pytest.mark.parametrize("name", STATIC_MASK_NAMES)
def test_static_masks(device, dtype, sq, sk, name):
    run(device, dtype, mask=get_mask_pair(name), aux=make_mask_aux(name), sq=sq, sk=sk)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("sq,sk", ((143, 137),))
@pytest.mark.parametrize("name", PARAMETERIZED_MASK_NAMES)
def test_parameterized_masks(device, dtype, sq, sk, name):
    run(
        device,
        dtype,
        mask=get_mask_pair(name, window_size=17, offset=sk - sq),
        sq=sq,
        sk=sk,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", GLOBAL_MASK_NAMES)
def test_global_masks_fixed_layout(device, dtype, name):
    run(device, dtype, mask=get_mask_pair(name), aux=make_mask_aux(name), sq=137, sk=143)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", ("prefix_lm", "dilated_sliding_window"))
def test_mask_boundaries(device, dtype, name):
    # Cross the actual 512-prefix / 256-window boundaries, not just one tile.
    run(device, dtype, mask=get_mask_pair(name), sq=513, sk=519)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", ("sliding_window", "block_causal"))
def test_original_asymmetric_masks(device, dtype, name):
    """Retain the original left-only window and zero-offset block causal cases."""
    from . import mask_reference as ref

    from . import examples
    from flash_attn_npu_4.catlass_dsl.mods import MaskMod

    native = examples.mask(name)
    eager = (
        ref.get_flex_sliding_window_mask(128, 0, 143 - 137)
        if name == "sliding_window"
        else ref.get_flex_block_causal_mask(0)
    )
    mask = MaskMod(native.simt, torch_ref=lambda b, h, q, k, info, aux, scalars: eager(b, h, q, k))
    run(device, dtype, mask=mask, sq=137, sk=143)
