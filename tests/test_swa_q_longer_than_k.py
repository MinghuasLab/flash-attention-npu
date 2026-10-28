"""Check bottom-right SWA against the original, unnormalized API windows."""

import importlib

import pytest
import torch
import torch_npu


@pytest.mark.parametrize("version", [2, 3, 4])
@pytest.mark.parametrize(
    "nq,nk,window",
    [(8, 4, (4, 4)), (8, 4, (1, 6)), (8, 4, (1, -1)), (257, 65, (65, 65))],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout,unused", [
    ("TND", "none"), ("TND", "kv"), ("TND", "qkv"), ("BSND", "none"),
])
def test_swa_q_longer_than_k(version, nq, nk, window, dtype, unused, layout):
    name = torch_npu.npu.get_device_name()
    if version == 2 and "Ascend910" not in name:
        pytest.skip("FA2 requires Ascend910")
    if version == 2 and unused != "none":
        pytest.skip("FA2 varlen API does not expose seqused_q/k")
    module = importlib.import_module("flash_attn_npu" + (f"_{version}" if version != 2 else ""))
    torch.manual_seed(2026)
    q_lengths = [nq, nq - 1 if layout == "TND" else nq]
    k_lengths = [nk, nk - 1 if layout == "TND" else nk]
    q_alloc = [n + 2 * (unused == "qkv") for n in q_lengths]
    k_alloc = [n + 2 * (unused != "none") for n in k_lengths]
    q = torch.randn(sum(q_alloc), 2, 64, dtype=dtype).npu().requires_grad_()
    k = torch.randn(sum(k_alloc), 2, 64, dtype=dtype).npu().requires_grad_()
    v = torch.randn_like(k).requires_grad_()
    cu_q = torch.tensor([0, q_alloc[0], sum(q_alloc)], dtype=torch.int32, device="npu")
    cu_k = torch.tensor([0, k_alloc[0], sum(k_alloc)], dtype=torch.int32, device="npu")
    kwargs = dict(window_size=window, causal=False)
    if unused != "none":
        kwargs.update(
            seqused_q=(torch.tensor(q_lengths, dtype=torch.int32, device="npu")
                       if unused == "qkv" else None),
            seqused_k=torch.tensor(k_lengths, dtype=torch.int32, device="npu"),
        )
    if layout == "TND":
        out = module.flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_alloc), max_seqlen_k=max(k_alloc), **kwargs,
        )
    else:
        out = module.flash_attn_func(
            q.view(2, nq, 2, 64), k.view(2, nk, 2, 64), v.view(2, nk, 2, 64), **kwargs,
        )
    if isinstance(out, tuple):
        out = out[0]
    out = out.reshape(-1, 2, 64)

    # Independent float64 reference: do not reuse host normalization or cached Golden.
    qr, kr, vr = [x.detach().cpu().double().requires_grad_() for x in (q, k, v)]
    refs, actuals = [], []
    q_start = k_start = 0
    for nq, nk, aq, ak in zip(q_lengths, k_lengths, q_alloc, k_alloc):
        scores = torch.einsum(
            "ihd,jhd->hij", qr[q_start:q_start + nq], kr[k_start:k_start + nk]
        ) / 8
        center = torch.arange(nq, device="cpu")[:, None] + nk - nq
        col = torch.arange(nk, device="cpu")[None, :]
        allowed = torch.ones(nq, nk, dtype=torch.bool, device="cpu")
        if window[0] >= 0:
            allowed &= col >= center - window[0]
        if window[1] >= 0:
            allowed &= col <= center + window[1]
        valid = allowed.any(-1)
        scores = scores.masked_fill(~allowed, -torch.inf)
        scores = torch.where(valid[None, :, None], scores, 0)
        probs = scores.softmax(-1) * valid[None, :, None]
        refs.append(torch.einsum("hij,jhd->ihd", probs, vr[k_start:k_start + nk]))
        actuals.append(out[q_start:q_start + nq])
        q_start += aq
        k_start += ak
    expected, actual = torch.cat(refs), torch.cat(actuals)
    tol = 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual.cpu().double(), expected, atol=tol, rtol=tol)
    # 950 currently supports forward SWA only.
    if "Ascend910" in name:
        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad.cpu().double())
        for got, ref in zip((q.grad, k.grad, v.grad), (qr.grad, kr.grad, vr.grad)):
            grad_tol = 1e-2 if dtype == torch.float16 else 5e-2
            torch.testing.assert_close(got.cpu().double(), ref, atol=grad_tol, rtol=grad_tol)
