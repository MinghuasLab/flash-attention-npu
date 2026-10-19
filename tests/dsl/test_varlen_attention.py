"""Packed-storage contracts, local callback coordinates and runtime rebinding."""

from importlib.util import find_spec
from contextlib import nullcontext
import os
from types import SimpleNamespace

import pytest

if find_spec("catlass") is None or find_spec("torch") is None:
    pytest.skip("optional CATLASS DSL / PyTorch runtime is absent", allow_module_level=True)

import torch
import catlass.tla as tla
from flash_attn_npu_dsl import compute_block_sparsity, flash_attn_varlen_func, interface, simd
from flash_attn_npu_dsl.seqlen_info import validate_sequence_tensor
from flash_attn_npu_dsl.block_sparsity import BlockSparseTensorsTorch, validate_packed_block_sparse


def _causal(b, h, q, kv, info, aux):
    return kv <= q + info.seqlen_k - info.seqlen_q


@simd
def _vector_causal(b, h, q, kv, info, aux):
    return tla.cmp(kv, q + info.seqlen_k - info.seqlen_q, "le")


@tla.jit
def _offset_score(s, b, h, *, q_idx, kv_idx, seqlen_info, aux_tensors):
    return s * 0.75 + (q_idx + seqlen_info.offset_q).to(tla.Float32) * 0.01


@pytest.mark.skipif(not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"), reason="explicit NPU opt-in")
@pytest.mark.parametrize(
    "dtype,dq,dv,batch,q_capacity,k_capacity,mode,used_kind",
    [
        (torch.float16, 64, 96, 3, 129, 257, "builtin", "k"),
        (torch.bfloat16, 128, 64, 3, 129, 257, "simt", "both"),
        (torch.float16, 256, 192, 3, 129, 257, "simd", "q"),
        (torch.bfloat16, 32, 80, 32, 65, 129, "mixed", "both"),
    ],
)
def test_bsnd_effective_lengths_preserve_storage_and_rebind(
    dtype, dq, dv, batch, q_capacity, k_capacity, mode, used_kind
):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(17)
    host = [
        torch.randn(s, generator=gen).to(dtype)
        for s in (
            (batch, q_capacity, 8, dq),
            (batch, k_capacity, 2, dq),
            (batch, k_capacity, 2, dv),
        )
    ]
    inputs = [x.to(device) for x in host]
    q_bound = 5 if mode == "mixed" else 65
    q_lengths = [([q_bound, 0, 3][b % 3] if used_kind != "k" else q_capacity) for b in range(batch)]
    k_lengths = [([129, 0, 1][b % 3] if used_kind != "q" else k_capacity) for b in range(batch)]
    used_q = torch.tensor(q_lengths, dtype=torch.int32, device=device) if used_kind != "k" else None
    used_k = torch.tensor(k_lengths, dtype=torch.int32, device=device) if used_kind != "q" else None
    q_bound = q_bound if used_q is not None else q_capacity
    k_bound = 129 if used_k is not None else k_capacity
    mask = _causal if mode == "simt" else _vector_causal
    options = {"causal": True} if mode == "builtin" else {"mask_mod": mask}
    if mode == "mixed":
        options["score_mod"] = _offset_score
    interface._compiled_kernels.clear()
    try:
        for update in range(3):
            if update == 1:
                if used_q is not None:
                    q_lengths[0] = 1
                    used_q.copy_(torch.tensor(q_lengths, dtype=torch.int32, device=device))
                if used_k is not None:
                    k_lengths[0] = 0
                    used_k.copy_(torch.tensor(k_lengths, dtype=torch.int32, device=device))
            elif update == 2:
                inputs = [x.clone() for x in inputs]
                used_q = used_q.clone() if used_q is not None else None
                used_k = used_k.clone() if used_k is not None else None
            if mode != "builtin":
                options["block_sparse_tensors"] = compute_block_sparsity(
                    128,
                    128,
                    batch,
                    1,
                    q_capacity,
                    k_capacity,
                    mask,
                    None,
                    device,
                    seqused_q=used_q,
                    seqused_k=used_k,
                )
            previous = None
            for return_lse in (True, False):
                out, lse = flash_attn_varlen_func(
                    *inputs,
                    seqused_q=used_q,
                    seqused_k=used_k,
                    max_seqlen_q=q_bound,
                    max_seqlen_k=k_bound,
                    return_lse=return_lse,
                    **options,
                )
                actual, actual_lse = out.cpu().float(), lse.cpu() if lse is not None else None
                assert out.shape == (batch, q_capacity, 8, dv)
                for b, (sq, sk) in enumerate(zip(q_lengths, k_lengths)):
                    if not sq:
                        continue
                    q = host[0][b, :sq].float().transpose(0, 1)
                    k, v = [
                        x[b, :sk].float().repeat_interleave(4, 1).transpose(0, 1) for x in host[1:]
                    ]
                    scores = q @ k.transpose(-1, -2) * dq**-0.5
                    if mode == "mixed":
                        scores = scores * 0.75 + torch.arange(sq)[None, :, None] * 0.01
                    keep = torch.arange(sk)[None, :] <= torch.arange(sq)[:, None] + sk - sq
                    scores.masked_fill_(~keep, -torch.inf)
                    expected_lse = scores.logsumexp(-1)
                    expected = (scores.softmax(-1).nan_to_num() @ v).transpose(0, 1)
                    torch.testing.assert_close(actual[b, :sq], expected, atol=0.05, rtol=0)
                    empty = torch.isneginf(expected_lse)
                    assert torch.count_nonzero(actual[b, :sq].transpose(0, 1)[empty]) == 0
                    if return_lse:
                        torch.testing.assert_close(
                            actual_lse[b, :, :sq], expected_lse, atol=0.05, rtol=0
                        )
                    else:
                        assert lse is None
                        assert torch.equal(actual[b, :sq], previous[b, :sq])
                previous = actual
            if update == 0:
                compiled_count = len(interface._compiled_kernels)
            else:
                assert len(interface._compiled_kernels) == compiled_count
    finally:
        interface._compiled_kernels.clear()


@pytest.mark.parametrize(
    "option,value",
    [
        ("scheduler_metadata", object()),
        ("min_seqlen_k", 1),
        ("qv", object()),
    ],
)
def test_varlen_unsupported_options_fail_before_device_work(option, value):
    with pytest.raises(NotImplementedError, match=option):
        flash_attn_varlen_func(None, None, None, **{option: value})


@pytest.mark.parametrize(
    "shape,dtype,match",
    [
        ((2, 2), torch.int32, "shape"),
        ((4,), torch.int64, "int32"),
        ((3,), torch.int32, "shape"),
    ],
)
def test_sequence_metadata_structure(shape, dtype, match):
    with pytest.raises((ValueError, TypeError), match=match):
        validate_sequence_tensor(
            torch.empty(shape, dtype=dtype), "cu_seqlens_q", 4, torch.device("cpu")
        )


def test_packed_table_independent_head_broadcast_and_zero_capacity(monkeypatch):
    blocks = BlockSparseTensorsTorch(
        torch.zeros(4, 3, dtype=torch.int32),
        torch.empty(1, 0, dtype=torch.int32),
        torch.ones(1, 3, dtype=torch.int32),
        torch.zeros(4, 7, dtype=torch.int32),
        torch.tensor([0, 1, 3], dtype=torch.int32),
        torch.tensor([0, 1, 7], dtype=torch.int32),
        (128, 128),
    )

    def no_read(*args, **kwargs):
        raise AssertionError("metadata validation must not read tensor contents")

    monkeypatch.setattr(torch.Tensor, "item", no_read)
    metadata = validate_packed_block_sparse(
        blocks, batch_size=2, num_heads_q=4, device=torch.device("cpu")
    )
    assert metadata.tensors[1] is None
    assert [s[1] for s in metadata.strides] == [3, 0, 0, 7]
    with pytest.raises(ValueError, match="both"):
        validate_packed_block_sparse(
            blocks._replace(full_block_idx=None),
            batch_size=2,
            num_heads_q=4,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="block_size"):
        validate_packed_block_sparse(
            blocks._replace(block_size=None),
            batch_size=2,
            num_heads_q=4,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    "average_q,max_q,max_k,has_used,expected_dn",
    [
        (127, 512, 512, False, False),
        (128, 512, 512, False, True),
        (129, 512, 512, False, True),
        (128, 512, 512, True, False),
        (128, 128, 256, False, True),
        (128, 129, 512, False, False),
    ],
)
def test_varlen_scalar_sparse_dn_uses_known_occupancy(
    monkeypatch, average_q, max_q, max_k, has_used, expected_dn
):
    captured = []

    def compile_kernel(kernel, key, arguments):
        captured.append(arguments())
        return lambda *args, **kwargs: None

    monkeypatch.setattr(interface, "_validate_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(interface, "_check_launch_tensors", lambda tensors: None)
    monkeypatch.setattr(interface, "from_dlpack", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(interface, "_compile_kernel", compile_kernel)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            device=lambda device: nullcontext(),
            get_device_properties=lambda device: SimpleNamespace(cube_core_num=24),
        ),
        raising=False,
    )
    q = torch.empty(2 * average_q, 4, 64, dtype=torch.float16)
    k = torch.empty(2 * max_k, 2, 64, dtype=torch.float16)
    metadata = torch.zeros(3, dtype=torch.int32)
    blocks = BlockSparseTensorsTorch(
        torch.zeros(1, 8, dtype=torch.int32),
        torch.zeros(1, 32, dtype=torch.int32),
        cu_total_m_blocks=metadata,
        cu_block_idx_offsets=metadata,
        block_size=(128, 128),
    )
    flash_attn_varlen_func(
        q,
        k,
        k,
        cu_seqlens_q=metadata,
        cu_seqlens_k=metadata,
        seqused_q=torch.zeros(2, dtype=torch.int32) if has_used else None,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        mask_mod=_causal,
        block_sparse_tensors=blocks,
    )
    assert captured[0][23:26] == (1, expected_dn, False)


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"), reason="explicit NPU opt-in required"
)
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    "dtype,dim,mode",
    [
        (torch.float16, 8, "builtin"),
        (torch.bfloat16, 16, "simt"),
        (torch.float16, 32, "simd"),
        (torch.bfloat16, 64, "mixed"),
        (torch.float16, 80, "simt"),
        (torch.bfloat16, 96, "builtin"),
        (torch.float16, 128, "simd"),
    ],
)
def test_varlen_offsets_empty_sequences_and_rebinding(dtype, dim, mode, sparse):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(17)
    q_lengths, k_lengths = [3, 0, 65, 129], [5, 17, 0, 257]
    cuq = torch.tensor([0, 3, 3, 68, 197], dtype=torch.int32)
    cuk = torch.tensor([0, 5, 22, 22, 279], dtype=torch.int32)
    host = [
        torch.randn(shape, generator=generator).to(dtype)
        for shape in [(197, 4, dim), (279, 2, dim), (279, 2, dim)]
    ]
    inputs = [t.to(device) for t in host]
    metadata = [t.to(device) for t in (cuq, cuk)]
    used = [torch.tensor(x, dtype=torch.int32, device=device) for x in (q_lengths, k_lengths)]
    options = (
        {"causal": True}
        if mode == "builtin"
        else {"mask_mod": _causal if mode == "simt" else _vector_causal}
    )
    if mode == "mixed":
        options["score_mod"] = _offset_score
    interface._compiled_kernels.clear()
    try:
        for update in range(3):
            if update == 1:
                q_lengths[3], k_lengths[3] = 63, 65
                for dst, value in zip(used, (q_lengths, k_lengths)):
                    dst.copy_(torch.tensor(value, dtype=torch.int32, device=device))
            if update == 2:
                inputs, metadata, used = ([t.clone() for t in x] for x in (inputs, metadata, used))
            if sparse:
                callback = _causal if mode == "simt" else _vector_causal
                options["mask_mod"] = callback
                options["block_sparse_tensors"] = compute_block_sparsity(
                    128,
                    128,
                    4,
                    1,
                    129,
                    257,
                    callback,
                    None,
                    device,
                    cu_seqlens_q=metadata[0],
                    cu_seqlens_k=metadata[1],
                    seqused_q=used[0],
                    seqused_k=used[1],
                )
            expected = torch.zeros_like(host[0], dtype=torch.float32)
            expected_lse = torch.full((4, 197), -torch.inf)
            valid = torch.zeros(197, dtype=torch.bool)
            for b, (sq, sk) in enumerate(zip(q_lengths, k_lengths)):
                qs, ks = int(cuq[b]), int(cuk[b])
                valid[qs : qs + sq] = True
                if not sq or not sk:
                    continue
                q = host[0][qs : qs + sq].float().permute(1, 0, 2)
                k, v = [
                    t[ks : ks + sk].float().repeat_interleave(2, 1).permute(1, 0, 2)
                    for t in host[1:]
                ]
                scores = q @ k.transpose(-1, -2) * dim**-0.5
                if mode == "mixed":
                    scores = scores * 0.75 + (torch.arange(sq) + qs)[None, :, None] * 0.01
                keep = torch.arange(sk)[None, :] <= torch.arange(sq)[:, None] + sk - sq
                scores.masked_fill_(~keep, -torch.inf)
                expected[qs : qs + sq] = (scores.softmax(-1).nan_to_num() @ v).permute(1, 0, 2)
                expected_lse[:, qs : qs + sq] = scores.logsumexp(-1)
            for return_lse in (False, True):
                out, lse = flash_attn_varlen_func(
                    *inputs,
                    cu_seqlens_q=metadata[0],
                    cu_seqlens_k=metadata[1],
                    seqused_q=used[0],
                    seqused_k=used[1],
                    max_seqlen_q=129,
                    max_seqlen_k=257,
                    return_lse=return_lse,
                    **options,
                )
                torch.npu.synchronize()
                actual = out.cpu().float()
                torch.testing.assert_close(actual[valid], expected[valid], atol=0.05, rtol=0)
                assert torch.count_nonzero(actual[3:68]) == 0
                if return_lse:
                    torch.testing.assert_close(
                        lse.cpu()[:, valid], expected_lse[:, valid], atol=0.05, rtol=0
                    )
                else:
                    assert lse is None
            if update == 0:
                compiled_count = len(interface._compiled_kernels)
            else:
                assert len(interface._compiled_kernels) == compiled_count
    finally:
        interface._compiled_kernels.clear()


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"), reason="explicit NPU opt-in required"
)
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_packed_head_writeback_matches_bsnd_and_fp32(kv_heads):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    batch, sq, sk, heads, dim = 32, 5, 17, 33 * kv_heads, 64
    generator = torch.Generator().manual_seed(29)
    host = [
        torch.randn(shape, generator=generator).to(torch.bfloat16)
        for shape in (
            (batch, sq, heads, dim),
            (batch, sk, kv_heads, dim),
            (batch, sk, kv_heads, dim),
        )
    ]
    q = host[0].float().permute(0, 2, 1, 3)
    k, v = [t.float().repeat_interleave(33, 2).permute(0, 2, 1, 3) for t in host[1:]]
    scores = q @ k.transpose(-1, -2) / 8
    scores.masked_fill_(torch.arange(sk)[None, :] > torch.arange(sq)[:, None] + sk - sq, -torch.inf)
    expected = (scores.softmax(-1) @ v).permute(0, 2, 1, 3).reshape(-1, heads, dim)
    expected_lse = scores.logsumexp(-1).permute(1, 0, 2).reshape(heads, -1)
    inputs = [t.to(device) for t in host]
    fixed = interface.flash_attn_func(*inputs, causal=True, return_lse=True)
    packed = flash_attn_varlen_func(
        *(t.flatten(0, 1) for t in inputs),
        cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32, device=device) * sq,
        cu_seqlens_k=torch.arange(batch + 1, dtype=torch.int32, device=device) * sk,
        max_seqlen_q=sq,
        max_seqlen_k=sk,
        causal=True,
        return_lse=True,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(packed[0].cpu().float(), expected, atol=0.05, rtol=0)
    torch.testing.assert_close(packed[1].cpu(), expected_lse, atol=0.05, rtol=0)
    torch.testing.assert_close(packed[0], fixed[0].flatten(0, 1), atol=0.05, rtol=0)
    torch.testing.assert_close(
        packed[1], fixed[1].permute(1, 0, 2).reshape(heads, -1), atol=0.05, rtol=0
    )


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"), reason="explicit NPU opt-in required"
)
@pytest.mark.parametrize(
    "provide_m,provide_indices", [(False, False), (True, False), (False, True), (True, True)]
)
def test_packed_builder_prefixes_preserve_caller_capacity(provide_m, provide_indices):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    # Cross the 128-thread scan boundary; zero-length sequences reserve no rows
    # unless the caller explicitly supplies extra count capacity.
    batch = 129
    q_lengths = torch.arange(batch, dtype=torch.int32) % 3
    k_lengths = torch.arange(batch, dtype=torch.int32) % 5
    cuq = torch.cat((torch.zeros(1, dtype=torch.int32), q_lengths.cumsum(0, dtype=torch.int32)))
    cuk = torch.cat((torch.zeros(1, dtype=torch.int32), k_lengths.cumsum(0, dtype=torch.int32)))
    m_rows = (q_lengths + 127) // 128 + int(provide_m)
    cm = torch.cat((torch.zeros(1, dtype=torch.int32), m_rows.cumsum(0, dtype=torch.int32)))
    capacity = m_rows * ((k_lengths + 127) // 128)
    ci = torch.cat((torch.zeros(1, dtype=torch.int32), capacity.cumsum(0, dtype=torch.int32)))
    blocks = compute_block_sparsity(
        128,
        128,
        batch,
        1,
        2,
        4,
        _vector_causal,
        None,
        device,
        cu_seqlens_q=cuq.to(device),
        cu_seqlens_k=cuk.to(device),
        cu_total_m_blocks=cm.to(device) if provide_m else None,
        cu_block_idx_offsets=ci.to(device) if provide_indices else None,
    )
    assert torch.equal(blocks.cu_total_m_blocks.cpu(), cm)
    assert torch.equal(blocks.cu_block_idx_offsets.cpu(), ci)
    partial, full = blocks.mask_block_cnt.cpu(), blocks.full_block_cnt.cpu()
    for b, (sq, sk) in enumerate(zip(q_lengths.tolist(), k_lengths.tolist())):
        begin, end = int(cm[b]), int(cm[b + 1])
        if not sq or not sk:
            assert torch.count_nonzero(partial[:, begin:end]) == 0
            assert torch.count_nonzero(full[:, begin:end]) == 0
        else:
            mask = torch.arange(sk)[None, :] <= torch.arange(sq)[:, None] + sk - sq
            assert int(full[0, begin]) == int(mask.all())
            assert int(partial[0, begin]) == int(mask.any() and not mask.all())
            assert torch.count_nonzero(partial[:, begin + 1 : end]) == 0
            assert torch.count_nonzero(full[:, begin + 1 : end]) == 0


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"), reason="explicit NPU opt-in required"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_varlen_softcap_retains_scaled_score_before_mask(dtype):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    qlens, klens = [127, 129], [65, 257]
    cuq, cuk = (
        torch.tensor([0, 127, 256], dtype=torch.int32),
        torch.tensor([0, 65, 322], dtype=torch.int32),
    )
    generator = torch.Generator().manual_seed(29)
    host = [
        torch.randn(shape, generator=generator).to(dtype)
        for shape in ((256, 4, 32), (322, 2, 32), (322, 2, 32))
    ]
    expected, expected_lse = torch.zeros(256, 4, 32), torch.full((4, 256), -torch.inf)
    for b, (sq, sk) in enumerate(zip(qlens, klens)):
        qs, ks = int(cuq[b]), int(cuk[b])
        q = host[0][qs : qs + sq].float().permute(1, 0, 2)
        k, v = [t[ks : ks + sk].float().repeat_interleave(2, 1).permute(1, 0, 2) for t in host[1:]]
        scores = 2 * torch.tanh((q @ k.transpose(-1, -2)) * (32**-0.5 / 2))
        scores.masked_fill_(
            torch.arange(sk)[None, :] > torch.arange(sq)[:, None] + sk - sq, -torch.inf
        )
        expected[qs : qs + sq] = (scores.softmax(-1).nan_to_num() @ v).permute(1, 0, 2)
        expected_lse[:, qs : qs + sq] = scores.logsumexp(-1)
    inputs = [t.to(device) for t in host]
    for want_lse in (False, True):
        out, lse = flash_attn_varlen_func(
            *inputs,
            cu_seqlens_q=cuq.to(device),
            cu_seqlens_k=cuk.to(device),
            max_seqlen_q=129,
            max_seqlen_k=257,
            causal=True,
            softcap=2,
            return_lse=want_lse,
        )
        torch.testing.assert_close(out.cpu().float(), expected, atol=0.05, rtol=0)
        if want_lse:
            torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.05, rtol=0)


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"), reason="explicit NPU opt-in required"
)
@pytest.mark.parametrize("packed", [False, True], ids=["bsnd", "tnd"])
@pytest.mark.parametrize(
    "dtype,dim,sq,sk,window",
    [(torch.float16, 64, 193, 258, (32, 0)), (torch.bfloat16, 80, 513, 518, (512, 0))],
    ids=["nd-65-column-end", "dn-5-column-end"],
)
def test_window_clips_compute_and_copy_extents_with_dynamic_reuse(
    packed, dtype, dim, sq, sk, window
):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(29)
    capacity = sq + 32
    host = [
        torch.randn(shape, generator=generator).to(dtype)
        for shape in ((2, capacity, 4, dim), (2, sk, 2, dim), (2, sk, 2, dim))
    ]
    cuq = torch.tensor([0, capacity, 2 * capacity], dtype=torch.int32, device=device)
    cuk = torch.tensor([0, sk, 2 * sk], dtype=torch.int32, device=device)
    used_q = torch.empty(2, dtype=torch.int32, device=device)
    used_k = torch.empty_like(used_q)
    interface._compiled_kernels.clear()
    executables = {}
    try:
        for update in range(3):
            current_q = capacity if update == 1 else sq
            qlens = [current_q, 0 if update == 1 else sq - 1] if packed else [current_q] * 2
            klens = [sk, sq - 129] if packed else [sk] * 2
            current = host if packed else [host[0][:, :current_q].contiguous(), *host[1:]]
            inputs = [t.flatten(0, 1).to(device) if packed else t.to(device) for t in current]
            options = {}
            if packed:
                used_q.copy_(torch.tensor(qlens, dtype=torch.int32, device=device))
                used_k.copy_(torch.tensor(klens, dtype=torch.int32, device=device))
                if update == 2:
                    cuq, cuk, used_q, used_k = (t.clone() for t in (cuq, cuk, used_q, used_k))
                options = dict(
                    cu_seqlens_q=cuq,
                    cu_seqlens_k=cuk,
                    seqused_q=used_q,
                    seqused_k=used_k,
                    max_seqlen_q=capacity,
                    max_seqlen_k=sk,
                )
            expected, expected_lse, empty = [], [], []
            for b, (nq, nk) in enumerate(zip(qlens, klens)):
                q = host[0][b, :nq].float().permute(1, 0, 2)
                k, v = [
                    t[b, :nk].float().repeat_interleave(2, 1).permute(1, 0, 2) for t in host[1:]
                ]
                center = torch.arange(nq)[:, None] + nk - nq
                keys = torch.arange(nk)[None, :]
                keep = (keys >= center - window[0]) & (keys <= center + window[1])
                scores = (q @ k.transpose(-1, -2)) * dim**-0.5
                scores.masked_fill_(~keep, -torch.inf)
                expected.append((scores.softmax(-1).nan_to_num() @ v).permute(1, 0, 2))
                expected_lse.append(scores.logsumexp(-1))
                empty.append(~keep.any(-1))
            call = flash_attn_varlen_func if packed else interface.flash_attn_func
            for return_lse in (False, True):
                out, lse = call(*inputs, **options, window_size=window, return_lse=return_lse)
                torch.npu.synchronize()
                actual = out.cpu().float().reshape(2, -1, 4, dim)
                actual_lse = None
                if return_lse:
                    actual_lse = lse.cpu()
                    if packed:
                        actual_lse = actual_lse.reshape(4, 2, capacity).permute(1, 0, 2)
                for b, nq in enumerate(qlens):
                    torch.testing.assert_close(actual[b, :nq], expected[b], atol=0.05, rtol=0)
                    assert torch.count_nonzero(actual[b, :nq][empty[b]]) == 0
                    if return_lse:
                        torch.testing.assert_close(
                            actual_lse[b, :, :nq], expected_lse[b], atol=0.05, rtol=0
                        )
                    else:
                        assert lse is None
                compiled = next(reversed(interface._compiled_kernels.values()))
                if return_lse not in executables:
                    executables[return_lse] = compiled
                assert compiled is executables[return_lse]
        assert len(interface._compiled_kernels) == 2
    finally:
        interface._compiled_kernels.clear()
