"""Sparse attention with independent predicates across packed mask words."""

import catlass.tla as tla
import pytest
import torch

from .test_utils import DTYPES, check_attention, copy_aux_to_device, make_inputs

from flash_attn_npu_4.catlass_dsl import compute_block_sparsity, flash_attn_func


@tla.jit
def _table_mask(b, h, q, k, info, aux, scalars):
    return aux[0][b, h, q, k]


@pytest.mark.npu
@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_mask_words_with_strided_aux(device, dtype):
    # Full, empty and independent partial tiles exercise both single and paired
    # predicate packing. Batch/head differences and tails forbid broadcast or
    # alignment assumptions; the auxiliary table also has a non-unit key stride.
    batch, heads, sq, sk = 2, 4, 385, 519
    generator = torch.Generator().manual_seed(3419)
    table = (torch.rand(batch, heads, sq, sk * 2, generator=generator) < 0.37)[..., ::2]
    table[:, :, 128:256, 128:256] = True
    table[:, :, :, 256:384] = False
    table[:, :, :7, :] = False
    table[:, :, 256:384, :] = False
    aux = (copy_aux_to_device(table, device),)
    assert aux[0].stride() == table.stride()
    cpu, inputs, _, _, _ = make_inputs(device, dtype, sq=sq, sk=sk, heads=heads, kh=heads)
    blocks = compute_block_sparsity(128, 128, batch, heads, sq, sk, _table_mask, aux, device)
    out, lse = flash_attn_func(
        *inputs,
        mask_mod=_table_mask,
        aux_tensors=aux,
        block_sparse_tensors=blocks,
        return_lse=True,
    )

    def reference(b, h, q, k, info, tensors, scalars):
        return tensors[0][b, h, q, k]

    check_attention(cpu, inputs, out, lse, mask_mod=reference, aux_tensors=(table,), loss="both")
