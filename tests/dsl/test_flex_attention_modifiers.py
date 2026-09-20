# Adapted from CATLASS DSL tests.
"""Callback binding and native frontend lowering; no device execution."""

from importlib.util import find_spec

import pytest

# Only an absent optional runtime is skipped; a broken installed runtime fails.
if find_spec("catlass") is None:
    pytest.skip("CATLASS DSL runtime is not installed", allow_module_level=True)

import catlass.tla as tla
from catlass.base_dsl import BaseDSL
from catlass.execution_lowering import UnsupportedExecutionLowering
from catlass.tla.runtime import make_fake_tensor
from flash_attn_npu_dsl import simd
from flash_attn_npu_dsl.modifiers import (
    uses_simd,
    validate_modifiers,
)


def mask6(b, h, q, kv, info, tensors):
    return q >= kv


def mask7(b, h, q, kv, info, tensors, scalars):
    return q >= kv + scalars[0]


def score7(value, b, h, *, q_idx, kv_idx, seqlen_info, aux_tensors):
    return value * 0.75


def score8(value, b, h, *, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
    return value * aux_scalars[-1]


@simd
@tla.jit
def vector_mask(b, h, q, kv, info, tensors):
    return tla.cmp(q, kv, "ge")


@tla.jit
@simd
def vector_score(value, b, h, *, q_idx, kv_idx, seqlen_info, aux_tensors):
    return value * 0.75


def test_simd_is_an_identity_marker_and_supports_both_jit_orders():
    def callback():
        pass

    assert simd(callback) is callback
    assert callback._use_simd is True
    assert not uses_simd(mask6) and not uses_simd(None)
    assert uses_simd(vector_mask) and uses_simd(vector_score)


@pytest.mark.parametrize("mask,score", [(object(), None), (None, object())])
def test_callback_validation_rejects_nonfunctions(mask, score):
    with pytest.raises(TypeError, match="function"):
        validate_modifiers(mask, score)


def _lower(
    mask=None,
    score=None,
    *,
    dtype=tla.Float16,
    dim=128,
    lse=True,
    tensors=None,
    scalars=None,
    schedule=False,
):
    from flash_attn_npu_dsl.flash_fwd import flex_attention_kernel

    def fake(element, size):
        return make_fake_tensor(element, (size,), (1,), layout_tag=tla.arch.RowMajor)

    sparse, sparse_strides = None, None
    if schedule:
        partial_capacity, full_capacity = (2, 2) if schedule is True else schedule
        sparse = (
            fake(tla.Int32, 2),
            fake(tla.Int32, 2 * partial_capacity) if partial_capacity else None,
            fake(tla.Int32, 2) if full_capacity is not None else None,
            fake(tla.Int32, 2 * full_capacity) if full_capacity else None,
        )
        sparse_strides = (
            (0, 1, 1),
            (0, partial_capacity, partial_capacity),
            (0, 1, 1) if full_capacity is not None else None,
            (0, full_capacity, full_capacity) if full_capacity is not None else None,
        )

    return BaseDSL()._lower(
        flex_attention_kernel.fn,
        kind="kernel",
        options=dict(flex_attention_kernel.options),
        location=flex_attention_kernel.decorator_location,
        type_args=(
            fake(dtype, 17 * 2 * dim),
            fake(dtype, 129 * dim),
            fake(dtype, 129 * dim),
            fake(dtype, 17 * 2 * dim),
            fake(tla.Float32, 34) if lse else None,
            sparse,
            tensors,
            scalars,
            1,
            2,
            1,
            17,
            129,
            dim**-0.5,
            dtype == tla.Float16,
            mask,
            score,
            None,
            None,
            *validate_modifiers(mask, score),
            dim,
            sparse_strides,
        ),
    )


@pytest.mark.parametrize(
    "mask,score,tensors,scalars,dtype,dim",
    [
        (mask6, score7, None, None, tla.Float16, 64),
        (tla.jit(mask7), tla.jit(score8), [], (1, 0.5), tla.BFloat16, 96),
        (vector_mask, vector_score, None, None, tla.Float16, 128),
        (mask6, vector_score, None, None, tla.BFloat16, 96),
        (vector_mask, score7, None, None, tla.Float16, 64),
        (mask6, None, None, None, tla.Float16, 64),
        (None, score7, None, None, tla.BFloat16, 96),
        (vector_mask, None, None, None, tla.Float16, 128),
        (None, vector_score, None, None, tla.Float16, 128),
    ],
)
def test_attention_lowers_independent_modes_and_supported_dimensions(
    mask, score, tensors, scalars, dtype, dim
):
    text = _lower(mask, score, tensors=tensors, scalars=scalars, dtype=dtype, dim=dim).asm()
    assert text.count("tla.func") == 1
    assert "flex_attention_kernel" in text
    assert ("simt" in text) == any(
        callback is not None and not uses_simd(callback) for callback in (mask, score)
    )


def test_o_only_eliminates_lse_storage_and_log():
    with_lse, without_lse = _lower(), _lower(lse=False)
    assert without_lse.argument_trees[4].runtime_leaf_count == 0
    with_text, without_text = with_lse.asm(), without_lse.asm()
    assert "tla.log" in with_text and "tla.log" not in without_text
    assert with_text.count("tla.alloc_ptr") == without_text.count("tla.alloc_ptr") + 1
    assert with_text.count("tla.copy") == without_text.count("tla.copy") + 1


@pytest.mark.parametrize(
    "capacities,leaf_count",
    [
        ((0, None), 1),
        ((0, 0), 2),
        ((2, None), 2),
        ((0, 2), 3),
        ((2, 0), 3),
        ((2, 2), 4),
    ],
)
def test_sparse_optional_pairs_and_zero_capacities_lower_without_dummy_buffers(
    capacities, leaf_count
):
    lowered = _lower(mask6, score7, dim=64, schedule=capacities)
    assert lowered.argument_trees[5].runtime_leaf_count == leaf_count
    assert "flex_attention_kernel" in lowered.asm()


@pytest.mark.parametrize("mask", [mask6, vector_mask])
@pytest.mark.parametrize("score", [score7, vector_score])
def test_full_sparse_blocks_skip_mask_but_keep_score_scale_and_tail(mask, score):
    lowered = _lower(mask, score, dim=64, schedule=True)

    def stages(operation):
        if operation.name == "tla.vec.func":
            text = str(operation)
            if "tla.arange" in text or "simt" in text:
                yield operation
            return
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    yield from stages(child.operation)

    first, second = list(stages(lowered.module.operation))
    first_text, second_text = str(first), str(second)
    mask_cmp = 'tla.cmp "ge"' if uses_simd(mask) else 'tla.simt_cmp "ge"'
    mask_count = 2 if uses_simd(mask) else 1
    score_count = 2 if uses_simd(score) else 1
    mixed = uses_simd(mask) != uses_simd(score)
    scale_value, score_value = "1.250000e-01", "7.500000e-01"

    if mixed:
        # Score runs once for either block kind; only partial blocks run mask.
        dispatch = second.parent
        assert dispatch.parent == first.parent
        assert len(dispatch.regions[1].blocks) == 0
        siblings = [op.operation for op in first.parent.regions[0].blocks[0].operations]
        first_index, dispatch_index = siblings.index(first), siblings.index(dispatch)
        assert first_index < dispatch_index
        assert any(
            op.name == "tla.pipe_barrier" for op in siblings[first_index + 1 : dispatch_index]
        )
        assert mask_cmp not in first_text
        assert second_text.count(mask_cmp) == mask_count
        assert first_text.count(score_value) == score_count
        assert score_value not in second_text and scale_value not in second_text
    else:
        # Same-mode partial/full branches retain a single fused callback stage.
        dispatch = first.parent
        assert dispatch == second.parent
        assert first_text.count(mask_cmp) == mask_count
        assert mask_cmp not in second_text
        assert first_text.index(score_value) < first_text.index(mask_cmp)
        for text in (first_text, second_text):
            assert text.count(score_value) == score_count
            assert text.count(scale_value) == 1

    assert dispatch.name == "scf.if"
    condition = dispatch.operands[0].owner
    assert "arith.cmpi eq" in str(condition)
    assert "arith.constant 0 : i32" in str(condition.operands[1].owner)
    assert first_text.index(scale_value) < first_text.index(score_value)
    # Also catch a second scale inside online softmax after the callback stage.
    assert lowered.asm().count(scale_value) == (1 if mixed else 2)
    for text in (first_text, second_text):
        if 'mode = "simd"' in text:
            assert text.count('tla.cmp "lt"') == 2
            # Padded callback KV lanes clamp to a valid coordinate before masking.
            assert (
                sum(
                    "tla.where" in line and "-> !tla.vector<64xi32>" in line
                    for line in text.splitlines()
                )
                == 2
            )
        else:
            assert text.count('tla.simt_cmp "lt"') == 2
            assert text.index('tla.simt_cmp "lt"') < text.index("tla.simt_load")
            assert "scf.if" in text and "0xFF800000" in text


@simd
def invalid_mask(b, h, q, kv, info, tensors):
    return True


@simd
def invalid_score(value, b, h, *, q_idx, kv_idx, seqlen_info, aux_tensors):
    return 1.0


@simd
def short_score(value, b, h, *, q_idx, kv_idx, seqlen_info, aux_tensors):
    mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
    return value.reduce(tla.ReductionOp.ADD, mask=mask)


@pytest.mark.parametrize(
    "mask,score,error",
    [
        (invalid_mask, None, "MaskSSA"),
        (invalid_mask, score7, "MaskSSA"),
        (None, invalid_score, "Float32 VectorSSA"),
        (mask6, invalid_score, "Float32 VectorSSA"),
        (None, short_score, "Float32 VectorSSA"),
    ],
)
def test_simd_rejects_scalar_or_short_results(mask, score, error):
    with pytest.raises(UnsupportedExecutionLowering, match=error):
        _lower(mask, score)
