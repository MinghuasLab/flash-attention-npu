"""Small end-to-end checks for the Ascend950 FA3 D512 forward path."""

import pytest
import torch
import torch_npu

from flash_attn_npu_3 import flash_attn_with_kvcache
from tests.common.attention_ref import ref_flash_attention_pair
from tests.common.compare import assert_fa_close


@pytest.mark.parametrize("head_dim", [64, 128, 192, 256, 512])
@pytest.mark.parametrize(
    "dtype,layout,causal",
    [
        (torch.float16, "BSND", False),
        (torch.bfloat16, "BSND", True),
        (torch.float16, "TND", False),
        (torch.bfloat16, "TND", True),
    ],
)
def test_fa3_d512_kvcache(dtype, layout, causal, head_dim):
    if torch_npu.npu.device_count() == 0 or "Ascend950" not in torch_npu.npu.get_device_name():
        pytest.skip("requires Ascend950")
    gen = torch.Generator().manual_seed(512)
    q = torch.randn((1, 17, 4, head_dim), dtype=dtype, generator=gen)
    k = torch.randn((1, 129, 2, head_dim), dtype=dtype, generator=gen)
    v = torch.randn((1, 129, 2, head_dim), dtype=dtype, generator=gen)
    mask = (
        torch.triu(torch.ones((17, 129), dtype=torch.bool), diagonal=1 + 129 - 17)
        if causal
        else None
    )
    ref, _, baseline, _ = ref_flash_attention_pair(
        q, k, v, head_dim**-0.5, mask, dtype, differentiable=True
    )
    if layout == "TND":
        q_dev, k_dev, v_dev = q[0].npu(), k[0].npu(), v[0].npu()
        cu_q = torch.tensor([0, 17], dtype=torch.int32).npu()
    else:
        q_dev, k_dev, v_dev = q.npu(), k.npu(), v.npu()
        cu_q = None
    result = flash_attn_with_kvcache(
        q_dev,
        k_dev,
        v_dev,
        cache_seqlens=torch.tensor([129], dtype=torch.int32).npu(),
        cu_seqlens_q=cu_q,
        max_seqlen_q=17,
        causal=causal,
        num_splits=1,
        return_softmax_lse=False,
    )
    if layout == "TND":
        result = result.unsqueeze(0)
    assert_fa_close(result, ref, baseline, name=f"D{head_dim} {layout} causal={causal}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fa3_d512_paged_kvcache(dtype):
    if torch_npu.npu.device_count() == 0 or "Ascend950" not in torch_npu.npu.get_device_name():
        pytest.skip("requires Ascend950")
    gen = torch.Generator().manual_seed(513)
    q = torch.randn((1, 16, 4, 512), dtype=dtype, generator=gen)
    k = torch.randn((1, 129, 2, 512), dtype=dtype, generator=gen)
    v = torch.randn((1, 129, 2, 512), dtype=dtype, generator=gen)
    ref, _, baseline, _ = ref_flash_attention_pair(
        q, k, v, 512**-0.5, None, dtype, differentiable=True
    )

    def paged(x):
        padded = torch.zeros((256, 2, 512), dtype=dtype)
        padded[:129] = x[0]
        return padded.reshape(2, 128, 2, 512).flip(0).contiguous().npu()

    out = flash_attn_with_kvcache(
        q[0].npu(),
        paged(k),
        paged(v),
        cache_seqlens=torch.tensor([129], dtype=torch.int32).npu(),
        page_table=torch.tensor([[1, 0]], dtype=torch.int32).npu(),
        cu_seqlens_q=torch.tensor([0, 16], dtype=torch.int32).npu(),
        max_seqlen_q=16,
        num_splits=1,
        return_softmax_lse=False,
    )
    assert_fa_close(out.unsqueeze(0), ref, baseline, name="D512 paged TND")


def test_fa3_d512_varlen_kvcache():
    if torch_npu.npu.device_count() == 0 or "Ascend950" not in torch_npu.npu.get_device_name():
        pytest.skip("requires Ascend950")
    gen = torch.Generator().manual_seed(514)
    q = torch.randn((33, 16, 512), dtype=torch.bfloat16, generator=gen)
    k = torch.randn((257, 2, 512), dtype=torch.bfloat16, generator=gen)
    v = torch.randn((257, 2, 512), dtype=torch.bfloat16, generator=gen)
    out = flash_attn_with_kvcache(
        q.npu(),
        k.npu(),
        v.npu(),
        cache_seqlens=torch.tensor([129, 128], dtype=torch.int32).npu(),
        cu_seqlens_q=torch.tensor([0, 17, 33], dtype=torch.int32).npu(),
        max_seqlen_q=17,
        num_splits=1,
        return_softmax_lse=False,
    )
    for qstart, qend, kstart, kend in [(0, 17, 0, 129), (17, 33, 129, 257)]:
        ref, _, baseline, _ = ref_flash_attention_pair(
            q[qstart:qend].unsqueeze(0),
            k[kstart:kend].unsqueeze(0),
            v[kstart:kend].unsqueeze(0),
            512**-0.5,
            None,
            torch.bfloat16,
            differentiable=True,
        )
        assert_fa_close(out[qstart:qend].unsqueeze(0), ref, baseline, name=f"D512 varlen {qstart}")


def test_fa3_d512_paged_decode_auto_splits():
    if torch_npu.npu.device_count() == 0 or "Ascend950" not in torch_npu.npu.get_device_name():
        pytest.skip("requires Ascend950")
    gen = torch.Generator().manual_seed(515)
    q = torch.randn((1, 1, 4, 512), dtype=torch.float16, generator=gen)
    k = torch.randn((1, 1024, 2, 512), dtype=torch.float16, generator=gen)
    v = torch.randn((1, 1024, 2, 512), dtype=torch.float16, generator=gen)
    ref, _, baseline, _ = ref_flash_attention_pair(
        q, k, v, 512**-0.5, None, q.dtype, differentiable=True
    )
    k_pages = k.reshape(8, 128, 2, 512).flip(0).contiguous().npu()
    v_pages = v.reshape(8, 128, 2, 512).flip(0).contiguous().npu()
    out = flash_attn_with_kvcache(
        q[0].npu(),
        k_pages,
        v_pages,
        cache_seqlens=torch.tensor([1024], dtype=torch.int32).npu(),
        page_table=torch.tensor([list(range(7, -1, -1))], dtype=torch.int32).npu(),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32).npu(),
        max_seqlen_q=1,
        num_splits=0,
        return_softmax_lse=False,
    )
    assert_fa_close(out.unsqueeze(0), ref, baseline, name="D512 paged decode num_splits=0")


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"num_splits": 2}, "split-KV"),
    ],
)
def test_fa3_d512_rejects_unsupported_paths(kwargs, message):
    if torch_npu.npu.device_count() == 0 or "Ascend950" not in torch_npu.npu.get_device_name():
        pytest.skip("requires Ascend950")
    q = torch.zeros((1, 1, 1, 512), dtype=torch.float16).npu()
    kv = torch.zeros((1, 128, 1, 512), dtype=torch.float16).npu()
    extra = {}
    if kwargs["num_splits"] == 2:
        q = q[0]
        extra = {
            "page_table": torch.tensor([[0]], dtype=torch.int32).npu(),
            "cu_seqlens_q": torch.tensor([0, 1], dtype=torch.int32).npu(),
            "max_seqlen_q": 1,
        }
    with pytest.raises(RuntimeError, match=message):
        flash_attn_with_kvcache(
            q, kv, kv, cache_seqlens=torch.tensor([128], dtype=torch.int32).npu(), **extra, **kwargs
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "layout,causal,paged,batch,sq,sk,heads,kv_heads",
    [
        ("BSND", False, False, 1, 17, 128, 4, 2),
        ("BSND", True, False, 1, 17, 129, 4, 2),
        ("TND", True, False, 1, 17, 129, 4, 2),
        ("BSND", False, False, 2, 257, 640, 4, 2),
        ("TND", False, True, 1, 16, 129, 4, 2),
        ("TND", True, True, 1, 1, 1024, 4, 2),
        ("TND", False, False, 2, 17, 129, 16, 2),
        ("BSND", True, False, 1, 17, 8, 4, 2),
        ("BSND", True, False, 1, 257, 1, 4, 2),
        ("TND", True, False, 1, 257, 1, 4, 2),
    ],
)
def test_fa3_d512_lse(dtype, layout, causal, paged, batch, sq, sk, heads, kv_heads):
    if torch_npu.npu.device_count() == 0 or "Ascend950" not in torch_npu.npu.get_device_name():
        pytest.skip("requires Ascend950")
    gen = torch.Generator().manual_seed(516)
    q = torch.randn((batch, sq, heads, 512), dtype=dtype, generator=gen)
    k = torch.randn((batch, sk, kv_heads, 512), dtype=dtype, generator=gen)
    v = torch.randn((batch, sk, kv_heads, 512), dtype=dtype, generator=gen)
    q_lengths = [sq - i if layout == "TND" else sq for i in range(batch)]
    k_lengths = [sk - i if layout == "TND" else sk for i in range(batch)]
    cu_q = [0]
    for length in q_lengths:
        cu_q.append(cu_q[-1] + length)
    q_dev = q.npu() if layout == "BSND" else torch.cat([q[i, :n] for i, n in enumerate(q_lengths)]).npu()
    page_table = None
    if paged:
        blocks = (sk + 127) // 128
        def pack_pages(x):
            padded = torch.zeros((batch, blocks * 128, kv_heads, 512), dtype=dtype)
            padded[:, :sk] = x
            return padded.reshape(batch * blocks, 128, kv_heads, 512).flip(0).contiguous().npu()
        k_dev, v_dev = pack_pages(k), pack_pages(v)
        page_table = torch.arange(batch * blocks - 1, -1, -1, dtype=torch.int32).reshape(batch, blocks).npu()
    elif layout == "TND":
        k_dev = torch.cat([k[i, :n] for i, n in enumerate(k_lengths)]).npu()
        v_dev = torch.cat([v[i, :n] for i, n in enumerate(k_lengths)]).npu()
    else:
        k_dev, v_dev = k.npu(), v.npu()
    out, lse, *_ = flash_attn_with_kvcache(
        q_dev, k_dev, v_dev,
        cache_seqlens=torch.tensor(k_lengths, dtype=torch.int32).npu(),
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32).npu() if layout == "TND" else None,
        max_seqlen_q=max(q_lengths), page_table=page_table,
        causal=causal, num_splits=1, return_softmax_lse=True,
    )
    assert lse.dtype == torch.float32
    assert tuple(lse.shape) == ((batch, heads, sq) if layout == "BSND" else (heads, sum(q_lengths)))
    assert tuple(out.shape) == tuple(q_dev.shape)
    for i, (nq, nk) in enumerate(zip(q_lengths, k_lengths)):
        mask = torch.triu(torch.ones((1, nq, nk), dtype=torch.bool), diagonal=1 + nk - nq) if causal else None
        ref, lse_ref, baseline, lse_baseline = ref_flash_attention_pair(
            q[i:i + 1, :nq], k[i:i + 1, :nk], v[i:i + 1, :nk],
            512**-0.5, mask, dtype, differentiable=True,
        )
        if mask is not None:
            fully_masked = mask.all(dim=-1)
            ref[fully_masked] = 0
            baseline[fully_masked] = 0
            lse_ref = lse_ref.masked_fill(fully_masked[:, None, :], torch.inf)
            lse_baseline = lse_baseline.masked_fill(fully_masked[:, None, :], torch.inf)
        if layout == "TND":
            actual_o = out[cu_q[i]:cu_q[i + 1]].unsqueeze(0)
            actual_lse = lse[:, cu_q[i]:cu_q[i + 1]].unsqueeze(0)
        else:
            actual_o, actual_lse = out[i:i + 1], lse[i:i + 1]
        assert_fa_close(actual_o, ref, baseline, name=f"D512 LSE out batch{i}")
        assert_fa_close(actual_lse, lse_ref, lse_baseline, name=f"D512 LSE batch{i}")
