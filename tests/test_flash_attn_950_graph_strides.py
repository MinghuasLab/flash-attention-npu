# Copyright (c) 2026, Minghua Shen.

"""Ascend950 graph replay with scheduler metadata and strided Q/K/V."""

import importlib
import math

import pytest
import torch
import torch_npu

pytestmark = pytest.mark.skipif(
    not torch_npu.npu.is_available() or "Ascend950" not in torch_npu.npu.get_device_name(),
    reason="Ascend950 native kernels",
)


@pytest.mark.parametrize("version", [3, 4], ids=["fa3", "fa4"])
@pytest.mark.parametrize("execution", ["npugraph", "npugraph_with_planner", "aot_eager"])
@pytest.mark.parametrize(
    "paged,dtype,window_left",
    [(False, torch.bfloat16, -1), (True, torch.bfloat16, 128), (True, torch.float8_e4m3fn, -1)],
    ids=["dense-bnsd", "paged-bnbd-window", "paged-bnbd-fp8"],
)
@torch.no_grad()
def test_graph_strides_with_scheduler_metadata(version, execution, paged, dtype, window_left):
    fa = importlib.import_module(f"flash_attn_npu_{version}")
    torch.manual_seed(829)

    def rand(shape):
        return torch.randn(shape, dtype=torch.float16, device="npu").to(dtype)

    def i32(values):
        return torch.tensor(values, dtype=torch.int32, device="npu")

    used = i32([193, 145])
    if paged:
        cu_q = i32([0, 2, 5])
        table = i32([[3, 1], [0, 2]])
        q = rand((10, 4, 128))[1::2]
        # K and V have different physical axis orders, offsets and page pitches.
        k = rand((8, 2, 128, 128))[1::2].transpose(1, 2)
        v = rand((12, 128, 2, 128))[2::3]
    else:
        cu_q = table = None
        q = rand((2, 4, 3, 128)).transpose(1, 2)
        k = rand((4, 2, 256, 128))[1::2].transpose(1, 2)
        v = rand((6, 256, 2, 128))[2::3]

    def plan():
        opts = dict(cu_seqlens_q=cu_q, qkv_dtype=dtype, causal=window_left < 0, num_splits=1)
        if version == 3:
            return fa.get_scheduler_metadata(
                2,
                3,
                256,
                4,
                2,
                128,
                used,
                page_size=128 if paged else None,
                window_size=(window_left, -1),
                **opts,
            )
        return fa.get_scheduler_metadata(
            3, 256, 4, 2, 128, seqused_k=used, window_size_left=window_left, **opts
        )

    def forward(q, k, v, metadata):
        opts = dict(
            cu_seqlens_q=cu_q,
            page_table=table,
            max_seqlen_q=3,
            causal=window_left < 0,
            window_size=(window_left, -1),
            num_splits=1,
            scheduler_metadata=metadata,
        )
        if version == 3:
            return fa.flash_attn_with_kvcache(
                q, k, v, cache_seqlens=used, return_softmax_lse=True, **opts
            )[:2]
        return fa.flash_attn_varlen_func(
            q, k, v, max_seqlen_k=256, seqused_k=used, return_lse=True, **opts
        )[:2]

    metadata = plan()
    forward(q, k, v, metadata)  # Warm up the same launch before capture.
    torch.npu.synchronize()
    if execution == "aot_eager":
        torch._dynamo.reset()  # Isolate dtype/layout specializations between test cases.
        compiled = torch.compile(forward, backend="aot_eager", fullgraph=True)
    else:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            if execution == "npugraph_with_planner":
                # Paged calls let the public wrapper plan from device lengths.
                captured = forward(q, k, v, None if paged else plan())
            else:
                captured = forward(q, k, v, metadata)

    previous_out = None
    for step in range(3):
        if step == 1:
            # Change only the lengths and refresh the same metadata buffer.
            # Both lengths now fit in one page, while the window is shorter than cache capacity.
            used.copy_(i32([97, 121]))
            refreshed = plan()
            assert not torch.equal(metadata, refreshed)
            metadata.copy_(refreshed)
        if step == 2:
            # Keep the schedule and reuse the captured input addresses.
            for view in (q, k, v):
                view.copy_(rand(view.shape))
            if paged:
                table.copy_(i32([[1, 3], [2, 0]]))

        before = metadata.clone()
        packed = tuple(view.contiguous() for view in (q, k, v))
        expected = forward(*packed, plan())
        # Alternate physical layouts while passing the exact same metadata tensor.
        if execution == "aot_eager":
            packed_out = compiled(*packed, metadata)
            actual = compiled(q, k, v, metadata)
        else:
            packed_out = forward(*packed, metadata)
            graph.replay()
            actual = captured
        for result in (packed_out, actual):
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
        assert torch.equal(metadata, before)
        if previous_out is not None:
            assert not torch.equal(expected[0], previous_out)
        previous_out = expected[0]


@torch.no_grad()
def test_fa4_paged_graph_reschedules_split_kv():
    from flash_attn_npu_4 import flash_attn_varlen_func

    torch.manual_seed(831)
    q = torch.randn(8, 4, 128, dtype=torch.bfloat16, device="npu")[1::2]
    k = torch.randn(32, 2, 128, 128, dtype=q.dtype, device="npu")[1::2].transpose(1, 2)
    v = torch.randn(16, 128, 2, 128, dtype=q.dtype, device="npu")
    table = torch.randperm(16, dtype=torch.int32).reshape(2, 8).npu()
    cu_q = torch.tensor([0, 2, 4], dtype=torch.int32, device="npu")
    used = torch.tensor([1024, 1009], dtype=torch.int32, device="npu")

    def forward(q, k, v, host_tiling=False):
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            seqused_k=used,
            page_table=table,
            max_seqlen_q=2,
            max_seqlen_k=1024,
            num_splits=2,
            return_lse=True,
            disable_scheduler_metadata=host_tiling,
        )[:2]

    forward(q, k, v)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = forward(q, k, v)

    def check_replay(lengths):
        used.copy_(torch.tensor(lengths, dtype=torch.int32))
        packed = (q.contiguous(), k.contiguous(), v.contiguous())
        expected = forward(*packed)
        host_out, host_lse = forward(*packed, host_tiling=True)
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
        torch.testing.assert_close(captured[0], host_out)
        torch.testing.assert_close(captured[1], host_lse)

    # The captured planner must switch between long and short KV schedules.
    for lengths in ([1024, 1009], [111, 97], [981, 1024]):
        check_replay(lengths)

    # Alternate which sample is empty at the same captured output address.
    # Zero logits and unit values give a closed-form result for every row.
    q.zero_()
    v.fill_(1)
    for lengths in ([0, 1009], [981, 0]):
        check_replay(lengths)
        for batch, length in enumerate(lengths):
            rows = slice(2 * batch, 2 * batch + 2)
            out, lse = captured[0][rows], captured[1][:, rows]
            torch.testing.assert_close(out, torch.full_like(out, int(length > 0)), atol=0, rtol=0)
            expected_lse = torch.full_like(lse, math.log(length) if length else math.inf)
            torch.testing.assert_close(lse, expected_lse, atol=1e-5, rtol=1e-5)
