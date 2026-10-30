"""FA3/FA4 direct backward tests. No repository forward kernel is called."""

import importlib

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
if torch_npu.npu.device_count() == 0:
    pytest.skip("BNSD backward requires an Ascend NPU", allow_module_level=True)

IS_950 = "Ascend950" in torch_npu.npu.get_device_name()


def _backward(
    version,
    q,
    k,
    v,
    out,
    dout,
    lse,
    *,
    deterministic=False,
    causal=False,
    softcap=0.0,
    window=(-1, -1),
    outputs=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
):
    suffix = "_950" if IS_950 else ""
    module = importlib.import_module(f"flash_attn_npu_{version}.flash_attn_npu_interface{suffix}")
    dq, dk, dv = outputs or tuple(
        torch.empty(x.shape, dtype=x.dtype, device=x.device) for x in (q, k, v)
    )
    if version == 3:
        module._flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            dq=dq,
            dk=dk,
            dv=dv,
            is_causal=causal,
            softcap=softcap,
            deterministic=deterministic,
            window_size_left=window[0],
            window_size_right=window[1],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
    else:
        module._flash_attn_backward(
            q,
            k,
            v,
            out,
            dout,
            lse,
            dq=dq,
            dk=dk,
            dv=dv,
            causal=causal,
            softcap=softcap,
            deterministic=deterministic,
            window_size_left=window[0],
            window_size_right=window[1],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
    torch_npu.npu.synchronize()
    return dq, dk, dv


def _reference(q, k, v, dout, causal=False, softcap=0.0, window=(-1, -1)):
    """CPU float64 reference from the actual quantized inputs, including GQA."""
    q, k, v = [x.detach().cpu().double().requires_grad_() for x in (q, k, v)]
    group = q.shape[1] // k.shape[1]
    kk = k.repeat_interleave(group, dim=1)
    vv = v.repeat_interleave(group, dim=1)
    score = (q @ kk.transpose(-1, -2)) * q.shape[-1] ** -0.5
    if softcap:
        score = softcap * torch.tanh(score / softcap)
    sq, sk = q.shape[2], k.shape[2]
    row = torch.arange(sq)[:, None] + sk - sq
    col = torch.arange(sk)[None, :]
    visible = torch.ones((sq, sk), dtype=torch.bool)
    if causal:
        visible &= col <= row
    if window[0] >= 0:
        visible &= col >= row - window[0]
    if window[1] >= 0:
        visible &= col <= row + window[1]
    score = score.masked_fill(~visible, float("-inf"))
    lse = score.logsumexp(-1)
    out = score.softmax(-1) @ vv
    grads = torch.autograd.grad(out, (q, k, v), dout.cpu().double())
    return out.detach(), lse.detach(), grads


def _offset_tensor(x):
    # Contiguous tensor with a nonzero storage offset, including output buffers.
    storage = torch.empty(x.numel() + 16, device=x.device, dtype=x.dtype)
    view = storage[16:].view_as(x)
    view.copy_(x)
    assert view.is_contiguous() and view.storage_offset() == 16
    return view


def _gradient_error_summary(got, golden, bsnd, tol):
    parts = []
    for name, actual, ref, other in zip(("dq", "dk", "dv"), got, golden, bsnd):
        actual_cpu = actual.transpose(1, 2).cpu().double()
        other_cpu = other.transpose(1, 2).cpu().double()
        error = (actual_cpu - ref).abs()
        bad = (error > tol + tol * ref.abs()).sum().item()
        parts.append(
            f"{name}: golden_bad={bad}/{error.numel()}, "
            f"golden_max_abs={error.max().item():.6g}, "
            f"bsnd_max_abs={(actual_cpu - other_cpu).abs().max().item():.6g}"
        )
    return "; ".join(parts)


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 4, 128, 128, 64),  # MHA
        (2, 6, 2, 137, 259, 128),  # GQA, rectangular/tail tiles
        (1, 4, 1, 65, 129, 80),  # MQA, non-bucket head dim
        (1, 1, 1, 1, 1, 32),
        (1, 2, 1, 129, 65, 192),  # Sq > Sk, noncausal
        (1, 2, 2, 65, 129, 256),
    ],
)
def test_bnsd_backward(version, dtype, deterministic, shape):
    b, nq, nk, sq, sk, d = shape
    torch.manual_seed(123)
    q, k, v = [
        torch.randn(s, dtype=dtype, device="npu") * 0.3
        for s in ((b, nq, sq, d), (b, nk, sk, d), (b, nk, sk, d))
    ]
    dout = torch.randn_like(q) * 0.3
    out, lse, golden = _reference(q, k, v, dout)
    out, lse = out.to(dtype=q.dtype).to(device=q.device), lse.float().to(device="npu")
    q, k, v, out, dout, lse = map(_offset_tensor, (q, k, v, out, dout, lse))
    q, k, v, out, dout = [x.transpose(1, 2) for x in (q, k, v, out, dout)]
    outputs = tuple(
        _offset_tensor(torch.zeros(x.shape, dtype=x.dtype, device=x.device)) for x in (q, k, v)
    )
    got = _backward(version, q, k, v, out, dout, lse, deterministic=deterministic, outputs=outputs)
    assert all(x.is_contiguous() for x in got)
    assert [x.shape for x in got] == [x.shape for x in (q, k, v)]
    # BSND is a second oracle using the same saved out and LSE.
    bsnd_inputs = [x.contiguous() for x in (q, k, v, out, dout)]
    bsnd = _backward(version, *bsnd_inputs, lse, deterministic=deterministic)
    tol = 0.015 if dtype == torch.bfloat16 else 0.003
    for name, actual, ref, other in zip(("dq", "dk", "dv"), got, golden, bsnd):
        try:
            torch.testing.assert_close(
                actual.transpose(1, 2).cpu().double(), ref, atol=tol, rtol=tol
            )
            torch.testing.assert_close(actual, other, atol=tol, rtol=tol)
        except AssertionError as error:
            summary = _gradient_error_summary(got, golden, bsnd, tol)
            raise AssertionError(f"{name} failed; {summary}\n{error}") from error
    if deterministic:
        repeat = _backward(version, q, k, v, out, dout, lse, deterministic=True)
        for a, b in zip(got, repeat):
            assert torch.equal(a, b)


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(
    "causal,softcap,window",
    [
        (True, 0.0, (-1, -1)),
        (True, 2.0, (-1, -1)),
        (False, 2.0, (-1, -1)),
        (False, 0.0, (32, 16)),
    ],
)
def test_bnsd_backward_features(version, deterministic, causal, softcap, window):
    if IS_950 and window != (-1, -1):
        pytest.skip("950 backward does not support local windows")
    torch.manual_seed(456)
    q = torch.randn(2, 4, 137, 64, device="npu", dtype=torch.bfloat16) * 0.3
    k = torch.randn(2, 2, 259, 64, device="npu", dtype=torch.bfloat16) * 0.3
    v, dout = torch.randn_like(k) * 0.3, torch.randn_like(q) * 0.3
    out, lse, golden = _reference(q, k, v, dout, causal, softcap, window)
    got = _backward(
        version,
        *[x.transpose(1, 2) for x in (q, k, v, out.to(dtype=q.dtype).to(device=q.device), dout)],
        lse.float().to(device="npu"),
        deterministic=deterministic,
        causal=causal,
        softcap=softcap,
        window=window,
    )
    for actual, ref in zip(got, golden):
        torch.testing.assert_close(
            actual.transpose(1, 2).cpu().double(), ref, atol=0.015, rtol=0.015
        )


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("invalid", ["lse", "output_shape", "output_stride"])
def test_bnsd_backward_rejects_invalid(version, invalid):
    q = torch.randn(2, 4, 17, 64, device="npu", dtype=torch.float16)
    k, v, out, dout = (torch.randn_like(q) for _ in range(4))
    lse = torch.randn(2, 4, 17, device="npu", dtype=torch.float32)
    kwargs = {}
    q, k, v, out, dout = [x.transpose(1, 2) for x in (q, k, v, out, dout)]
    if invalid == "output_stride":
        kwargs["outputs"] = (
            torch.empty_like(q.transpose(1, 2)).transpose(1, 2),
            torch.empty_like(k),
            torch.empty_like(v),
        )
    elif invalid == "lse":
        lse = lse[:, :, :-1]
    else:
        kwargs["outputs"] = (
            torch.empty_like(q[:, :, :-1]),
            torch.empty_like(k),
            torch.empty_like(v),
        )
    with pytest.raises((RuntimeError, ValueError), match="layout|contiguous|LSE|lse|shape"):
        _backward(version, q, k, v, out, dout, lse, **kwargs)


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("storage", ["mixed", "padded_bsnd", "padded_bnsd"])
def test_backward_uses_independent_input_strides(version, storage):
    torch.manual_seed(731)
    q = torch.randn(2, 4, 37, 64, device="npu", dtype=torch.float16) * 0.3
    k = torch.randn(2, 2, 73, 64, device="npu", dtype=torch.float16) * 0.3
    v, dout = torch.randn_like(k) * 0.3, torch.randn_like(q) * 0.3
    out, lse, golden = _reference(q, k, v, dout)

    def logical_view(x, index):
        view = x.transpose(1, 2)
        if storage == "mixed":
            return view.contiguous() if index % 2 else view
        if storage == "padded_bsnd":
            shape = list(view.shape)
            shape[1] *= 2
            result = torch.empty(shape, device=x.device, dtype=x.dtype)[:, ::2]
        else:
            shape = list(x.shape)
            shape[2] *= 2
            result = torch.empty(shape, device=x.device, dtype=x.dtype)[:, :, ::2].transpose(1, 2)
        result.copy_(view)
        return result

    inputs = [
        logical_view(x, i)
        for i, x in enumerate((q, k, v, out.to(dtype=q.dtype).to(device=q.device), dout))
    ]
    got = _backward(version, *inputs, lse.float().to(device="npu"))
    for actual, ref in zip(got, golden):
        assert actual.is_contiguous()
        torch.testing.assert_close(
            actual.transpose(1, 2).cpu().double(), ref, atol=0.003, rtol=0.003
        )


@pytest.mark.parametrize("version", [3, 4])
def test_tnd_backward_padded_token_stride(version):
    torch.manual_seed(975)
    qs, ks, vs, outs, dos, lses = [], [], [], [], [], []
    golden = [[], [], []]
    for sq, sk in [(17, 29), (31, 43)]:
        q = torch.randn(1, 4, sq, 64, device="npu", dtype=torch.float16) * 0.3
        k = torch.randn(1, 2, sk, 64, device="npu", dtype=torch.float16) * 0.3
        v, dout = torch.randn_like(k) * 0.3, torch.randn_like(q) * 0.3
        out, lse, grads = _reference(q, k, v, dout)
        for collection, x in zip(
            (qs, ks, vs, outs, dos), (q, k, v, out.to(dtype=q.dtype).to(device=q.device), dout)
        ):
            collection.append(x.squeeze(0).transpose(0, 1))
        lses.append(lse.squeeze(0))
        for collection, grad in zip(golden, grads):
            collection.append(grad.squeeze(0).transpose(0, 1))

    def padded(collection):
        packed = torch.cat(collection)
        shape = list(packed.shape)
        shape[0] *= 2
        result = torch.empty(shape, device=packed.device, dtype=packed.dtype)[::2]
        result.copy_(packed)
        return result

    inputs = [padded(xs) for xs in (qs, ks, vs, outs, dos)]
    cuq = torch.tensor([0, 17, 48], device="npu", dtype=torch.int32)
    cuk = torch.tensor([0, 29, 72], device="npu", dtype=torch.int32)
    lse = torch.cat(lses, dim=1).float().to(device="npu")
    got = _backward(
        version, *inputs, lse, cu_seqlens_q=cuq, cu_seqlens_k=cuk, max_seqlen_q=31, max_seqlen_k=43
    )
    for actual, chunks in zip(got, golden):
        torch.testing.assert_close(actual.cpu().double(), torch.cat(chunks), atol=0.003, rtol=0.003)


@pytest.mark.parametrize("version", [3, 4])
def test_bnsd_backward_distinct_value_dim_950(version):
    if not IS_950:
        pytest.skip("910 backward requires equal QK and V dimensions")
    q = torch.randn(2, 4, 37, 64, device="npu", dtype=torch.float16) * 0.3
    k = torch.randn(2, 2, 73, 64, device="npu", dtype=torch.float16) * 0.3
    v = torch.randn(2, 2, 73, 128, device="npu", dtype=torch.float16) * 0.3
    dout = torch.randn(2, 4, 37, 128, device="npu", dtype=torch.float16) * 0.3
    out, lse, golden = _reference(q, k, v, dout)
    inputs = [x.transpose(1, 2) for x in (q, k, v, out.to(dtype=q.dtype).to(device=q.device), dout)]
    got = _backward(version, *inputs, lse.float().to(device="npu"))
    for actual, ref in zip(got, golden):
        torch.testing.assert_close(
            actual.transpose(1, 2).cpu().double(), ref, atol=0.003, rtol=0.003
        )
