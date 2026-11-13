"""Split-KV consumes the same callbacks and active-block order as normal attention."""

from importlib.util import find_spec
import os

import pytest

if find_spec("catlass") is None or find_spec("torch") is None:
    pytest.skip("optional runtimes absent", allow_module_level=True)

import torch
import catlass.tla as tla
from flash_attn_npu_dsl import flash_attn_varlen_func, compute_block_sparsity, simd


def test_split_requires_paged_decode_before_compilation(monkeypatch):
    from flash_attn_npu_dsl import interface

    monkeypatch.setattr(interface, "_validate_inputs", lambda *args, **kwargs: None)
    q, k = torch.empty(3, 4, 64), torch.empty(129, 2, 64)
    with pytest.raises(NotImplementedError, match="paged TND"):
        flash_attn_varlen_func(
            q,
            k,
            k,
            cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 129], dtype=torch.int32),
            max_seqlen_q=3,
            num_splits=2,
        )


@pytest.mark.parametrize(
    "kv_len,tasks,sparse,expected",
    [
        (65535, 4, None, 1),
        (65536, 4, None, 4),
        (65537, 4, None, 4),
        (65536, 16, None, 1),
        (65536, 4, object(), 1),
        (65536, 0, None, 1),
    ],
)
def test_automatic_split_uses_only_known_long_low_occupancy_work(kv_len, tasks, sparse, expected):
    from flash_attn_npu_dsl.interface import _select_num_splits

    assert _select_num_splits(0, tasks, 16, kv_len, True, sparse, 16, 128) == expected


def _mask(b, h, q, kv, info, aux, aux_scalars=None):
    return (h != 0) & (kv <= q + info.seqlen_k - info.seqlen_q)


@simd
def _vector_mask(b, h, q, kv, info, aux, aux_scalars=None):
    return tla.bitwise_and(
        tla.cmp(h, 0, "ne"), tla.cmp(kv, q + info.seqlen_k - info.seqlen_q, "le")
    )


def _score(s, b, h, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
    return s * aux_scalars[0] + h.to(tla.Float32) * 0.01


@simd
def _vector_score(s, b, h, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
    return s * aux_scalars[0] + h.to(tla.Float32, tla.params.CastParams()) * 0.01


@pytest.mark.skipif(not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"), reason="explicit NPU opt-in")
@pytest.mark.parametrize(
    "batch,heads,kv_heads,mode,sparse,dtype,dq,dv",
    [
        (2, 4, 2, "builtin", False, torch.float16, 64, 96),
        (2, 4, 2, "scalar", True, torch.bfloat16, 64, 96),
        (2, 4, 2, "vector", True, torch.float16, 64, 96),
        (16, 6, 2, "mixed", True, torch.bfloat16, 64, 96),
        (32, 8, 1, "builtin", False, torch.float16, 64, 96),
        (2, 4, 2, "scalar", True, torch.bfloat16, 192, 255),
        (16, 6, 2, "vector", True, torch.float16, 59, 256),
        (2, 4, 2, "builtin", False, torch.float16, 256, 8),
        (2, 4, 2, "mixed_reverse", True, torch.float16, 127, 193),
        (2, 4, 2, "softcap", True, torch.bfloat16, 192, 129),
    ],
)
def test_split_kv_callbacks_and_empty_segments(batch, heads, kv_heads, mode, sparse, dtype, dq, dv):
    import torch_npu  # noqa: F401

    torch.set_num_threads(1)
    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(17)
    sq, sk, page_size = 3, 385, 128
    q = torch.randn(batch, sq, heads, dq, generator=generator).to(dtype)
    k = torch.randn(6, page_size, kv_heads, dq, generator=generator).to(dtype)
    v = torch.randn(6, page_size, kv_heads, dv, generator=generator).to(dtype)
    table = torch.tensor([[3, 1, 4, 0]], dtype=torch.int32).expand(batch, -1).clone()
    used = torch.full((batch,), sk, dtype=torch.int32)
    used[-1] = 0
    cuq = (torch.arange(batch + 1, dtype=torch.int32) * sq).to(device)
    used_npu = used.to(device)
    qn, kn, vn, tn = (x.to(device) for x in (q.flatten(0, 1), k, v, table))
    mask = (
        None
        if mode == "builtin"
        else _mask
        if mode in ("scalar", "mixed_reverse")
        else _vector_mask
    )
    score = (
        None
        if mode in ("builtin", "softcap")
        else _vector_score
        if mode in ("vector", "mixed_reverse")
        else _score
    )
    softcap = 6.0 if mode == "softcap" else 0.0
    auxiliary = (0.75,) if score is not None else None
    blocks = None
    if sparse:
        blocks = compute_block_sparsity(
            128,
            128,
            batch,
            heads,
            sq,
            sk,
            mask,
            None,
            device,
            aux_scalars=auxiliary,
            cu_seqlens_q=cuq,
            seqused_k=used_npu,
        )
        # Keep the active set, but make physical indices deliberately unordered.
        for counts, indices in (
            (blocks.mask_block_cnt, blocks.mask_block_idx),
            (blocks.full_block_cnt, blocks.full_block_idx),
        ):
            current, offsets = indices.cpu(), blocks.cu_block_idx_offsets.cpu()
            sizes = counts.cpu()
            for b in range(batch):
                for h in range(heads):
                    start, count = int(offsets[b]), int(sizes[h, b])
                    current[h, start : start + count] = current[h, start : start + count].flip(0)
            indices.copy_(current)
    kr = k[table.long()].flatten(1, 2)[:, :sk].float().repeat_interleave(heads // kv_heads, dim=2)
    vr = v[table.long()].flatten(1, 2)[:, :sk].float().repeat_interleave(heads // kv_heads, dim=2)
    logits = torch.einsum("bmhd,bnhd->bhmn", q.float(), kr) * dq**-0.5
    if score is not None:
        logits = logits * 0.75 + torch.arange(heads)[None, :, None, None] * 0.01
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    if mask is not None:
        logits[:, 0] = -torch.inf
    keep = torch.arange(sk)[None, :] <= torch.arange(sq)[:, None] + sk - sq
    logits.masked_fill_(~keep, -torch.inf)
    logits[-1] = -torch.inf
    expected_lse = logits.logsumexp(-1)
    expected = torch.einsum("bhmn,bnhd->bmhd", torch.nan_to_num(logits.softmax(-1)), vr)
    empty = torch.isneginf(expected_lse)
    upper = int(torch.npu.get_device_properties(device).cube_core_num)
    for splits in (1, 2, 3, 4, upper, 0):
        for return_lse in (True, False):
            out, lse = flash_attn_varlen_func(
                qn,
                kn,
                vn,
                cu_seqlens_q=cuq,
                seqused_k=used_npu,
                page_table=tn,
                max_seqlen_q=sq,
                max_seqlen_k=sk,
                num_splits=splits,
                causal=True,
                mask_mod=mask,
                score_mod=score,
                softcap=softcap,
                aux_scalars=auxiliary,
                block_sparse_tensors=blocks,
                return_lse=return_lse,
            )
            actual = out.cpu().float().reshape(batch, sq, heads, dv)
            torch.testing.assert_close(actual, expected, atol=0.05, rtol=0)
            assert torch.count_nonzero(actual.permute(0, 2, 1, 3)[empty]) == 0
            if return_lse:
                actual_lse = lse.cpu().reshape(heads, batch, sq).permute(1, 0, 2)
                torch.testing.assert_close(actual_lse, expected_lse, atol=0.05, rtol=0)
