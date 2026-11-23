# Copyright (c) 2026, Minghua Shen.

"""Regression tests for Ascend950 scheduler metadata generated inside an NPUGraph."""

import importlib

import pytest
import torch
import torch_npu

pytestmark = pytest.mark.skipif(
    not torch_npu.npu.is_available() or "Ascend950" not in torch_npu.npu.get_device_name(),
    reason="Ascend950 native kernels",
)


@pytest.mark.parametrize("version", [3, 4], ids=["fa3", "fa4"])
@torch.no_grad()
def test_dense_graph_builds_metadata(version):
    """The AICPU metadata producer must record its events during graph capture."""
    fa = importlib.import_module(f"flash_attn_npu_{version}")
    torch.manual_seed(830)
    q, k, v = (
        torch.randn(2, heads, tokens, 64, dtype=torch.float16, device="npu").transpose(1, 2)
        for heads, tokens in ((4, 17), (2, 67), (2, 67))
    )
    opts = {"return_attn_probs": True} if version == 3 else {"return_lse": True}

    def forward(q, k, v):
        return fa.flash_attn_func(q, k, v, num_splits=1, **opts)

    forward(q, k, v)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = forward(q, k, v)
    for _ in range(2):
        q.copy_(torch.randn_like(q))
        expected = forward(q.contiguous(), k.contiguous(), v.contiguous())
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)


@torch.no_grad()
def test_fa3_paged_graph_builds_metadata_from_device_lengths():
    """Reading cache_seqlens.max().item() during capture used to raise ACL error 107027."""
    from flash_attn_npu_3 import flash_attn_with_kvcache

    torch.manual_seed(832)
    q = torch.randn(4, 4, 128, dtype=torch.bfloat16, device="npu")
    k = torch.randn(4, 2, 128, 128, dtype=q.dtype, device="npu").transpose(1, 2)
    v = torch.randn_like(k).contiguous()
    cu_q = torch.tensor([0, 2, 4], dtype=torch.int32, device="npu")
    table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="npu")
    used = torch.tensor([193, 145], dtype=torch.int32, device="npu")

    def forward(key):
        return flash_attn_with_kvcache(
            q,
            key,
            v,
            cache_seqlens=used,
            cu_seqlens_q=cu_q,
            max_seqlen_q=2,
            page_table=table,
            num_splits=1,
            return_softmax_lse=True,
        )[:2]

    forward(k)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = forward(k)

    packed_k = k.contiguous()
    for lengths in ([193, 145], [97, 121]):
        used.copy_(torch.tensor(lengths, dtype=torch.int32))
        expected = forward(packed_k)
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
