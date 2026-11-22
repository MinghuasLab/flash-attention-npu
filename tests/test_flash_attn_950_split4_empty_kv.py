"""Paged TND FlashDecode must write every live Q row when its KV length becomes zero."""

import importlib
import math

import pytest
import torch
import torch_npu


@pytest.mark.skipif(
    not torch_npu.npu.is_available() or "Ascend950" not in torch_npu.npu.get_device_name(),
    reason="Ascend950 native FlashDecode",
)
@pytest.mark.parametrize(
    "route,execution",
    [
        ("fa3-cache", "eager"),
        ("fa4-auto", "eager"),
        ("fa4-external", "eager"),
        ("fa4-auto", "graph"),
        ("fa4-external", "graph"),
    ],
)
@torch.no_grad()
def test_split4_overwrites_empty_kv_rows(route, execution, monkeypatch):
    version = 3 if route == "fa3-cache" else 4
    fa = importlib.import_module(f"flash_attn_npu_{version}")
    backend = importlib.import_module(f"flash_attn_npu_{version}_950")
    q = torch.zeros((32, 4, 128), dtype=torch.float16, device="npu")
    k = q.new_zeros((18, 128, 2, 128))
    v = torch.ones_like(k)
    v[9:].fill_(2)
    cu_q = torch.tensor([0, 16, 32], dtype=torch.int32, device=q.device)
    lengths = cu_q.new_tensor([1025, 1025])
    table = torch.arange(18, dtype=torch.int32, device=q.device).reshape(2, 9)
    kwargs = dict(cu_seqlens_q=cu_q, max_seqlen_q=16, page_table=table, num_splits=4)
    if version == 3:
        forward = fa.flash_attn_with_kvcache
        kwargs.update(cache_seqlens=lengths, return_softmax_lse=True)
    else:
        forward = fa.flash_attn_varlen_func
        kwargs.update(seqused_k=lengths, max_seqlen_k=1152, return_lse=True)

    def plan():
        return fa.get_scheduler_metadata(
            16, 1152, 4, 2, 128, 4, cu_seqlens_q=cu_q, seqused_k=lengths, qkv_dtype=q.dtype
        )

    if route == "fa4-external":
        kwargs["scheduler_metadata"] = plan()

    # Supply the native out buffer so a missing write leaves a known value.
    out_buffer = torch.empty_like(q)
    native_forward = backend.fwd
    out_index = 6 if version == 3 else 4

    def forward_into_buffer(*args):
        args = list(args)
        assert args[out_index] is None
        args[out_index] = out_buffer
        return native_forward(*args)

    monkeypatch.setattr(backend, "fwd", forward_into_buffer)
    if execution == "graph":
        forward(q, k, v, **kwargs)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out, lse, *_ = forward(q, k, v, **kwargs)

    for kv_lens in ([0, 1025], [1025, 0]):
        lengths.copy_(lengths.new_tensor(kv_lens))
        if route == "fa4-external":
            kwargs["scheduler_metadata"].copy_(plan())
        out_buffer.fill_(37)
        if execution == "eager":
            out, lse, *_ = forward(q, k, v, **kwargs)
        else:
            graph.replay()
        assert out.data_ptr() == out_buffer.data_ptr()
        for batch, (rows, logsumexp) in enumerate(zip(out.cpu().chunk(2), lse.cpu().chunk(2, 1))):
            if kv_lens[batch] == 0:
                assert torch.count_nonzero(rows).item() == 0
                assert torch.isposinf(logsumexp).all()
            else:
                torch.testing.assert_close(rows, torch.full_like(rows, batch + 1))
                expected_lse = torch.full_like(logsumexp, math.log(kv_lens[batch]))
                torch.testing.assert_close(logsumexp, expected_lse, atol=1e-5, rtol=1e-5)
