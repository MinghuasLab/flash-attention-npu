"""Opt-in 1M-context checks against a chunked CPU attention reference.

Run on 950 with FA950_RUN_LONG_KV=1; ordinary test runs skip this memory-heavy suite.
"""
import math
import os

import pytest
import torch

if os.environ.get("FA950_RUN_LONG_KV") != "1":
    pytest.skip("set FA950_RUN_LONG_KV=1 for the long-context checks", allow_module_level=True)

from test_flash_attn_npu_v3_mixed_kv import make_case
from flash_attn_npu_3.flash_attn_npu_interface_950 import flash_attn_with_kvcache


@pytest.mark.parametrize("sq,sk", [(1, 1049600), (8, 1049600), (512, 1049600),
                                  (16385, 16385), (1048576, 1048576)],
                         ids=["1", "8", "512", "square-16k-tail", "square-1m"])
def test_mixed_kv_long_context(sq, sk):
    torch.set_num_threads(4)
    heads, dim = 2, 192
    q, k, v, ks, vs, table = make_case(sq, sk, heads)
    out, lse, *_ = flash_attn_with_kvcache(
        q.reshape(sq, heads * 16, dim).npu(), k.npu(), v.npu(),
        k_descale=ks.npu(), v_descale=vs.npu(), page_table=table.npu(),
        cache_seqlens=torch.tensor([sk], dtype=torch.int32, device="npu"),
        cu_seqlens_q=torch.tensor([0, sq], dtype=torch.int32, device="npu"),
        max_seqlen_q=sq, causal=True, num_splits=0, return_softmax_lse=True,
    )
    out, lse = out.cpu(), lse.cpu()
    # Check the full output without allocating an additional FP32 copy of
    # the million-row tensor. Only the rows used by the reference need FP32.
    for block in out.split(4096):
        assert torch.isfinite(block).all()
    assert torch.isfinite(lse).all()
    # Check every decode row. For chunked prefill, cross query-tile boundaries
    # without multiplying the work of the million-token CPU reference by 512.
    rows = torch.arange(sq) if sq <= 8 else torch.tensor([0, 63, 64, 127, 128, 255, 256, sq - 1])
    if sq > 2048:
        rows = torch.tensor([0, 127, 128, 2047, 2048, sq // 2, sq - 2, sq - 1])
    first_large_offset = (2**31 + heads * 16 * dim - 1) // (heads * 16 * dim)
    if first_large_offset < sq:
        rows = torch.cat((rows, torch.tensor([first_large_offset - 1, first_large_offset]))).unique()
    checked = rows.numel()

    # Independent CPU reference, retaining only a small score block at a time.
    # Grouping Q heads avoids materializing sixteen copies of the KV cache.
    qg = q[0, rows].permute(1, 0, 2).reshape(heads, 16 * checked, dim).float()
    running_max = torch.full((heads, 16 * checked), -float("inf"))
    running_sum = torch.zeros_like(running_max)
    accumulator = torch.zeros(heads, 16 * checked, dim)
    query_end = (sk - sq + rows).repeat(16)
    for start in range(0, sk, 4096):
        count = min(4096, sk - start)
        pages = table[0, start // 128:(start + count + 127) // 128].long()
        kp = k.view(torch.uint8)[pages].view(torch.float8_e4m3fn).float()
        vp = v.view(torch.uint8)[pages].view(torch.float8_e4m3fn).float()
        kd = (kp * ks[pages][..., None]).bfloat16().reshape(-1, heads, dim)[:count].float()
        vd = (vp * vs[pages][..., None]).bfloat16().reshape(-1, heads, dim)[:count].float()
        scores = torch.bmm(qg, kd.permute(1, 2, 0)) / math.sqrt(dim)
        if start + count > sk - sq:
            mask = torch.arange(start, start + count)[None, :] > query_end[:, None]
            scores.masked_fill_(mask[None], -float("inf"))
        new_max = torch.maximum(running_max, scores.amax(-1))
        correction = (running_max - new_max).exp()
        probability = (scores - new_max[..., None]).exp()
        accumulator.mul_(correction[..., None]).add_(torch.bmm(probability, vd.permute(1, 0, 2)))
        running_sum = running_sum * correction + probability.sum(-1)
        running_max = new_max
    reference = (accumulator / running_sum[..., None]).reshape(heads * 16, checked, dim).permute(1, 0, 2)
    reference_lse = (running_max + running_sum.log()).reshape(heads * 16, checked)
    delta = out[rows].float() - reference
    assert delta.abs().max() <= 0.0002 + 0.01 * reference.abs().max()
    assert delta.square().mean().sqrt() <= 0.01 * reference.square().mean().sqrt() + 0.00002
    torch.testing.assert_close(lse[:, rows], reference_lse, atol=0.003, rtol=0.0003)
