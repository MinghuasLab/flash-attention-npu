"""Logical callback coordinates and direct reads from shared, reordered KV pages."""

from importlib.util import find_spec
import os

import pytest

if find_spec("catlass") is None or find_spec("torch") is None:
    pytest.skip("optional CATLASS DSL / PyTorch runtime is absent", allow_module_level=True)

import torch
import catlass.tla as tla
from flash_attn_npu_dsl import compute_block_sparsity, flash_attn_varlen_func, interface, simd


@pytest.mark.parametrize(
    "kind,match",
    [
        ("dtype", "int32"),
        ("batch", "matching B"),
        ("contiguous", "contiguous"),
        ("missing_length", "seqused_k"),
        ("cu_k", "cu_seqlens_k"),
        ("page_size", "page size"),
    ],
)
def test_invalid_page_metadata_is_rejected_before_compile(monkeypatch, kind, match):
    monkeypatch.setattr(interface, "_validate_inputs", lambda *args, **kwargs: None)
    q = torch.empty(2, 3, 4, 64)
    k = torch.empty(4, 128 if kind != "page_size" else 64, 2, 64)
    table = torch.zeros(2, 3, dtype=torch.int32)
    used = torch.ones(2, dtype=torch.int32)
    kwargs = {}
    if kind == "dtype":
        table = table.long()
    elif kind == "batch":
        table = table[:1]
    elif kind == "contiguous":
        table = torch.zeros(3, 2, dtype=torch.int32).T
    elif kind == "missing_length":
        used = None
    elif kind == "cu_k":
        kwargs["cu_seqlens_k"] = torch.zeros(3, dtype=torch.int32)
    with pytest.raises((TypeError, ValueError), match=match):
        flash_attn_varlen_func(q, k, k, page_table=table, seqused_k=used, **kwargs)


def _scalar_mask(b, h, q, kv, info, aux):
    return (kv <= q + info.seqlen_k - info.seqlen_q) & (h != 0)


@simd
def _vector_mask(b, h, q, kv, info, aux):
    return tla.bitwise_and(
        tla.cmp(kv, q + info.seqlen_k - info.seqlen_q, "le"), tla.cmp(h, 0, "ne")
    )


@pytest.mark.skipif(not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"), reason="explicit NPU opt-in")
@pytest.mark.parametrize(
    "page_size,packed,mode,sparse,dq,dv,dtype",
    [
        (128, True, "scalar", True, 64, 96, torch.float16),
        (256, True, "vector", True, 96, 64, torch.bfloat16),
        (512, False, "builtin", False, 8, 80, torch.float16),
        (1024, False, "vector", True, 128, 32, torch.bfloat16),
        (128, False, "scalar", False, 128, 64, torch.float16),
        (256, True, "builtin", False, 32, 128, torch.bfloat16),
        (128, True, "vector", True, 192, 255, torch.bfloat16),
        (512, False, "scalar", True, 59, 256, torch.float16),
    ],
)
def test_paged_forward(page_size, packed, mode, sparse, dq, dv, dtype):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(17)
    q_lens, k_lens = [128, 0, 65], [page_size + 1, 0, 2 * page_size - 1]
    q_offsets = torch.tensor([0, 128, 128, 193], dtype=torch.int32)
    q = torch.randn((193, 4, dq) if packed else (3, 128, 4, dq), generator=generator).to(dtype)
    k = torch.randn(6, page_size, 2, dq, generator=generator).to(dtype)
    v = torch.randn(6, page_size, 2, dv, generator=generator).to(dtype)
    table = torch.tensor([[3, 1, 4], [4, 2, 0], [1, 5, 3]], dtype=torch.int32)
    qn, kn, vn = (x.to(device) for x in (q, k, v))
    tn = table.to(device)
    used_q = torch.tensor(q_lens, dtype=torch.int32, device=device)
    used_k = torch.tensor(k_lens, dtype=torch.int32, device=device)
    cuq = q_offsets.to(device) if packed else None
    mask = None if mode == "builtin" else _scalar_mask if mode == "scalar" else _vector_mask
    blocks = None
    if sparse:
        blocks = compute_block_sparsity(
            128,
            128,
            3,
            4,
            128,
            3 * page_size,
            mask,
            None,
            device,
            cu_seqlens_q=cuq,
            seqused_q=used_q,
            seqused_k=used_k,
        )
    for change in range(3):
        if change == 1:
            table = table.flip(1).contiguous()
            tn.copy_(table.to(device))
        elif change == 2:
            tn = tn.clone()
            qn, kn, vn = qn.clone(), kn.clone(), vn.clone()
        for return_lse in (True, False):
            out, lse = flash_attn_varlen_func(
                qn,
                kn,
                vn,
                cu_seqlens_q=cuq,
                seqused_q=used_q,
                seqused_k=used_k,
                page_table=tn,
                max_seqlen_q=128,
                max_seqlen_k=3 * page_size,
                causal=True,
                mask_mod=mask,
                block_sparse_tensors=blocks,
                return_lse=return_lse,
            )
            assert out.shape == (*q.shape[:-1], dv)
            actual = out.cpu().float()
            actual_lse = lse.cpu() if lse is not None else None
            for b, (sq, sk) in enumerate(zip(q_lens, k_lens)):
                if not sq:
                    continue
                qr = q[q_offsets[b] : q_offsets[b + 1]] if packed else q[b, :sq]
                kr = k[table[b].long()].flatten(0, 1)[:sk].float().repeat_interleave(2, dim=1)
                vr = v[table[b].long()].flatten(0, 1)[:sk].float().repeat_interleave(2, dim=1)
                scores = torch.einsum("mhd,nhd->hmn", qr.float(), kr) * dq**-0.5
                keep = torch.arange(sk)[None, :] <= torch.arange(sq)[:, None] + sk - sq
                scores.masked_fill_(~keep, -torch.inf)
                if mask is not None:
                    scores[0] = -torch.inf
                ref_lse = scores.logsumexp(-1)
                ref = torch.einsum("hmn,nhd->mhd", torch.nan_to_num(scores.softmax(-1)), vr)
                chosen = actual[q_offsets[b] : q_offsets[b + 1]] if packed else actual[b, :sq]
                torch.testing.assert_close(chosen, ref, atol=0.05, rtol=0)
                empty = torch.isneginf(ref_lse)
                assert torch.count_nonzero(chosen.permute(1, 0, 2)[empty]) == 0
                if return_lse:
                    chosen_lse = (
                        actual_lse[:, q_offsets[b] : q_offsets[b + 1]]
                        if packed
                        else actual_lse[b, :, :sq]
                    )
                    torch.testing.assert_close(chosen_lse, ref_lse, atol=0.05, rtol=0)
