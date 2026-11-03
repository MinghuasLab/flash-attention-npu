"""Regression coverage for D512 16-row GO workspace and KV accumulation."""
import pytest
import torch
import torch_npu
from flash_attn_npu_3 import flash_attn_with_kvcache
from tests.common.attention_ref import ref_flash_attention_pair
from tests.common.compare import assert_fa_close


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('q_len', [30, 32, 34, 128])
@pytest.mark.parametrize('kv_len', [128, 257])
def test_d512_go_row_chunks(dtype, q_len, kv_len):
    generator = torch.Generator().manual_seed(512)
    q = torch.randn((1, q_len, 1, 512), dtype=dtype, generator=generator)
    k = torch.randn((1, kv_len, 1, 512), dtype=dtype, generator=generator)
    v = torch.randn((1, kv_len, 1, 512), dtype=dtype, generator=generator)
    ref, _, baseline, _ = ref_flash_attention_pair(q, k, v, 512 ** -0.5, None, dtype, differentiable=True)
    actual = flash_attn_with_kvcache(
        q.npu(), k.npu(), v.npu(),
        cache_seqlens=torch.tensor([kv_len], dtype=torch.int32).npu(),
        max_seqlen_q=q_len, num_splits=1, return_softmax_lse=False,
    )
    assert_fa_close(actual, ref, baseline, name=f'D512 UB q={q_len} kv={kv_len}')
