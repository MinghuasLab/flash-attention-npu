# Copyright (c) 2026, Minghua Shen.

"""Logical tensor strides at the public Ascend950 FA3/FA4 interfaces."""

import importlib
import math

import pytest
import torch
import torch_npu

pytestmark = pytest.mark.skipif(
    not torch_npu.npu.is_available() or "Ascend950" not in torch_npu.npu.get_device_name(),
    reason="Ascend950 native kernels",
)


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("causal,fallback", [(False, False), (True, False), (False, True)])
def test_dense_strided_autograd(version, causal, fallback):
    fa = importlib.import_module(f"flash_attn_npu_{version}")
    torch.manual_seed(731)
    q = torch.randn(2, 6, 37, 64, device="npu", dtype=torch.float16).transpose(1, 2)
    k = torch.randn(2, 73, 2, 64, device="npu", dtype=q.dtype)
    v = torch.randn(4, 2, 73, 64, device="npu", dtype=q.dtype)[1::2].transpose(1, 2)
    if fallback:
        q = torch.randn(2, 37, 6, 128, device="npu", dtype=q.dtype)[..., ::2]
        k = k[:, :1].expand_as(k)
    dout = torch.randn_like(q)
    kwargs = {"return_attn_probs": True} if version == 3 else {"return_lse": True}

    def run(inputs):
        inputs = tuple(x.detach().requires_grad_() for x in inputs)
        out, lse = fa.flash_attn_func(*inputs, causal=causal, **kwargs)
        assert out.shape == q.shape and out.is_contiguous()
        assert lse.shape == (2, 6, 37)
        return out, lse, *torch.autograd.grad(out, inputs, dout)

    actual = run((q, k, v))
    packed = run(x.contiguous() for x in (q, k, v))
    for a, b in zip(actual, packed):
        torch.testing.assert_close(a, b, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize(
    "dtype,splits",
    [(torch.float16, 1), (torch.bfloat16, 1), (torch.float8_e4m3fn, 1), (torch.float8_e4m3fn, 2)],
)
def test_mixed_paged_strides(version, dtype, splits):
    fa = importlib.import_module(f"flash_attn_npu_{version}")
    torch.manual_seed(818)

    # Two batches cross independently mapped 128-token pages. K is head-major
    # with gaps between pages. V is token-major with a different page pitch.
    def rand(shape):
        return torch.randn(shape, device="npu", dtype=torch.float16).to(dtype)

    pages_per_batch = 3 if splits == 1 else 9
    pages = 2 * pages_per_batch
    q = rand((11, 4, 128))[::2]  # Last two allocated tokens are outside the cu-Q region.
    k = rand((pages * 2, 2, 128, 128))[1::2].transpose(1, 2)
    v = rand((pages * 3, 128, 2, 128))[2::3]
    cu = torch.tensor([0, 1, 4], device="npu", dtype=torch.int32)
    capacity = 128 * pages_per_batch
    used = torch.tensor([capacity - 7, capacity - 21], device="npu", dtype=torch.int32)
    table = torch.randperm(pages, dtype=torch.int32).reshape(2, pages_per_batch).npu()
    kwargs = dict(cu_seqlens_q=cu, max_seqlen_q=3, page_table=table, num_splits=splits)
    if version == 3:
        forward = fa.flash_attn_with_kvcache
        kwargs.update(cache_seqlens=used, return_softmax_lse=True)
    else:
        forward = fa.flash_attn_varlen_func
        kwargs.update(seqused_k=used, return_lse=True)

    actual = forward(q, k, v, **kwargs)
    expected = forward(q[:4].contiguous(), k.contiguous(), v.contiguous(), **kwargs)
    assert actual[1].shape == (4, 6)
    torch.testing.assert_close(actual[0][:4], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1][:, :4], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize("version", [3, 4])
def test_legacy_native_dense_layout(version, monkeypatch):
    fa = importlib.import_module(f"flash_attn_npu_{version}")
    backend = importlib.import_module(f"flash_attn_npu_{version}_950")
    native_fwd = backend.fwd
    raw_q = torch.randn(2, 4, 17, 64, device="npu", dtype=torch.float16)
    raw_k = torch.randn(2, 2, 67, 64, device="npu", dtype=raw_q.dtype)
    raw_v = torch.randn_like(raw_k)
    logical = tuple(x.transpose(1, 2) for x in (raw_q, raw_k, raw_v))
    kwargs = {"return_attn_probs": True} if version == 3 else {"return_lse": True}
    reference = fa.flash_attn_func(*logical, **kwargs)

    def legacy(*args):
        # The public wrapper passes the original logical views to the native op.
        args = list(args)
        for view, operand in zip(logical, args[:3]):
            assert operand.data_ptr() == view.data_ptr() and operand.stride() == view.stride()
        args[:3] = raw_q, raw_k, raw_v
        args[-1 if version == 3 else -4] = 2
        return native_fwd(*args)

    monkeypatch.setattr(backend, "fwd", legacy)
    actual = fa.flash_attn_func(*logical, **kwargs)
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "varlen"])
def test_autograd_with_gaps_on_every_outer_axis(version, packed):
    fa = importlib.import_module(f"flash_attn_npu_{version}")
    torch.manual_seed(615)

    def strided(shape):
        steps = (2, 3) if packed else (2, 3, 2)
        backing_shape = [length * step for length, step in zip(shape[:-1], steps)] + [shape[-1] + 3]
        backing = torch.randn(backing_shape, device="npu", dtype=torch.float16)
        slices = tuple(slice(1, None, step) for step in steps) + (slice(1, shape[-1] + 1),)
        result = backing[slices]
        assert result.shape == shape and result.stride(-1) == 1 and result.storage_offset() > 0
        return result

    q_shape = (17, 4, 64) if packed else (2, 17, 4, 64)
    kv_shape = (41, 2, 64) if packed else (2, 41, 2, 64)
    q, k, v, dout = (strided(shape) for shape in (q_shape, kv_shape, kv_shape, q_shape))
    opts = {"return_attn_probs": True} if version == 3 else {"return_lse": True}
    func = fa.flash_attn_func
    if packed:
        func = fa.flash_attn_varlen_func
        opts.update(
            cu_seqlens_q=torch.tensor([0, 6, 17], dtype=torch.int32, device="npu"),
            cu_seqlens_k=torch.tensor([0, 16, 41], dtype=torch.int32, device="npu"),
            max_seqlen_q=11,
            max_seqlen_k=25,
        )

    def run(inputs, grad):
        inputs = tuple(x.detach().requires_grad_() for x in inputs)
        out, lse = func(*inputs, **opts)[:2]
        return out, lse, *torch.autograd.grad(out, inputs, grad)

    actual = run((q, k, v), dout)
    expected = run((q.contiguous(), k.contiguous(), v.contiguous()), dout.contiguous())
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("paged", [False, True], ids=["packed", "paged"])
def test_fa4_varlen_empty_query_returns(paged):
    from flash_attn_npu_4 import flash_attn_varlen_func

    q = torch.empty((0, 4, 64), dtype=torch.bfloat16, device="npu")
    k = q.new_zeros((1, 128, 2, 64) if paged else (31, 2, 64))
    cu_q = torch.tensor([0, 0], dtype=torch.int32, device=q.device)
    lengths = torch.tensor([31], dtype=torch.int32, device=q.device)
    kv_args = (
        {"page_table": lengths.new_zeros((1, 1)), "seqused_k": lengths}
        if paged
        else {"cu_seqlens_k": torch.cat((lengths.new_zeros(1), lengths))}
    )
    out, lse, out_accum, lse_accum = flash_attn_varlen_func(
        q, k, k, cu_seqlens_q=cu_q, max_seqlen_q=0, max_seqlen_k=31, return_lse=True, **kv_args
    )
    assert out.shape == q.shape and out.dtype == q.dtype
    assert lse.shape == (4, 0) and lse.dtype == torch.float32
    for accum in (out_accum, lse_accum):
        assert accum.shape == (0,) and accum.dtype == torch.float32 and accum.device == q.device


@pytest.mark.parametrize(
    "version,metadata",
    [(3, "host"), (4, "auto"), (4, "external"), (4, "host")],
    ids=["fa3-cache-host", "fa4-varlen-auto", "fa4-varlen-external", "fa4-varlen-host"],
)
@torch.no_grad()
def test_paged_flashdecode_writes_empty_kv_rows(version, metadata, monkeypatch):
    fa = importlib.import_module(f"flash_attn_npu_{version}")
    backend = importlib.import_module(f"flash_attn_npu_{version}_950")
    q = torch.zeros((32, 4, 128), dtype=torch.float16, device="npu")
    k = q.new_zeros((9, 2, 128, 128)).transpose(1, 2)
    v = torch.ones_like(k)
    cu_q = torch.tensor([0, 16, 32], dtype=torch.int32, device=q.device)
    kv_lens = cu_q.new_tensor([0, 1025])
    table = cu_q.new_tensor([[0] * 9, list(range(9))])
    kwargs = dict(cu_seqlens_q=cu_q, max_seqlen_q=16, page_table=table, num_splits=4)
    if version == 3:
        forward = fa.flash_attn_with_kvcache
        kwargs.update(cache_seqlens=kv_lens, return_softmax_lse=True)
    else:
        forward = fa.flash_attn_varlen_func
        kwargs.update(seqused_k=kv_lens, max_seqlen_k=1152, return_lse=True)
        if metadata == "external":
            kwargs["scheduler_metadata"] = fa.get_scheduler_metadata(
                16, 1152, 4, 2, 128, 4, cu_seqlens_q=cu_q, seqused_k=kv_lens, qkv_dtype=q.dtype
            )
        elif metadata == "host":
            kwargs["disable_scheduler_metadata"] = True

    # An omitted store must fail even if the allocator would have returned zeroed memory.
    prefilled = torch.full_like(q, 37)
    native_fwd = backend.fwd
    out_slot = 6 if version == 3 else 4

    def forward_with_prefilled_out(*args):
        args = list(args)
        assert args[out_slot] is None
        args[out_slot] = prefilled
        return native_fwd(*args)

    monkeypatch.setattr(backend, "fwd", forward_with_prefilled_out)
    out, lse, *_ = forward(q, k, v, **kwargs)
    assert out.data_ptr() == prefilled.data_ptr()
    out, lse = out.cpu(), lse.cpu()
    torch.testing.assert_close(out[:16], torch.zeros_like(out[:16]), rtol=0, atol=0)
    assert torch.isposinf(lse[:, :16]).all()
    torch.testing.assert_close(out[16:], torch.ones_like(out[16:]), rtol=2e-3, atol=2e-3)
    expected_lse = torch.full_like(lse[:, 16:], math.log(1025))
    torch.testing.assert_close(lse[:, 16:], expected_lse, rtol=1e-5, atol=1e-5)
