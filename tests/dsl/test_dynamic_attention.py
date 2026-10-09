"""Cross-length executable reuse through the public attention interface."""

from importlib.util import find_spec
import os

import pytest

if find_spec("catlass") is None:
    pytest.skip("CATLASS DSL runtime is not installed", allow_module_level=True)
if find_spec("torch") is None:
    pytest.skip("PyTorch is not installed", allow_module_level=True)

import torch

import catlass.tla as tla
from flash_attn_npu_dsl import BlockSparseTensorsTorch, interface, simd


def _head_mask(b, h, q, kv, info, aux):
    return (h != 0) & (kv <= q + info.seqlen_k - info.seqlen_q)


@tla.jit
def _scale_score(score, b, h, *, q_idx, kv_idx, seqlen_info, aux_tensors):
    return score * 0.75


@tla.jit
def _coordinate_score(score, b, h, *, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars=None):
    return (
        score * 0.75
        + h.to(tla.Float32) * 0.01
        + q_idx.to(tla.Float32) * 0.02
        - kv_idx.to(tla.Float32) * 0.03
    )


@simd
def _vector_head_mask(b, h, q, kv, info, aux):
    return tla.bitwise_and(
        tla.cmp(h, 0, "ne"), tla.cmp(kv, q + info.seqlen_k - info.seqlen_q, "le")
    )


@simd
def _vector_scale_score(score, b, h, *, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars=None):
    return score * 0.75


def _causal_mask(b, h, q, kv, info, aux, aux_scalars):
    return (
        (b > 0)
        & (kv <= q + info.seqlen_k - info.seqlen_q)
        & ((kv < 256) | (kv >= 384))
        & ((b != 1) | (aux_scalars[0] == 0) | (kv >= 128))
    )


@simd
def _vector_causal_mask(b, h, q, kv, info, aux, aux_scalars):
    return tla.bitwise_and(
        tla.bitwise_and(tla.cmp(b, 0, "gt"), tla.cmp(kv, q + info.seqlen_k - info.seqlen_q, "le")),
        tla.bitwise_and(
            tla.bitwise_or(tla.cmp(kv, 256, "lt"), tla.cmp(kv, 384, "ge")),
            tla.bitwise_or(
                tla.bitwise_or(tla.cmp(b, 1, "ne"), tla.cmp(kv * 0 + aux_scalars[0], 0, "eq")),
                tla.cmp(kv, 128, "ge"),
            ),
        ),
    )


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"),
    reason="set FLASH_ATTN_DSL_TEST_DEVICE to opt into NPU execution",
)
@pytest.mark.parametrize(
    "dtype,dim,mask,score",
    [
        (torch.float16, 64, _causal_mask, _coordinate_score),
        (torch.bfloat16, 96, _vector_causal_mask, _vector_scale_score),
        (torch.float16, 128, _causal_mask, _vector_scale_score),
        (torch.bfloat16, 64, _vector_causal_mask, _coordinate_score),
    ],
)
@pytest.mark.parametrize("kv_heads", [1, 3])
def test_packed_shared_blocks_preserve_coordinates_tail_and_rebinding(
    dtype, dim, mask, score, kv_heads
):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(17)
    batch, sq, heads, sk = 32, 5, 33 * kv_heads, 389
    host = tuple(
        torch.randn(shape, generator=generator).to(dtype)
        for shape in (
            (batch, sq, heads, dim),
            (batch, sk, kv_heads, dim),
            (batch, sk, kv_heads, dim),
        )
    )
    inputs = tuple(t.to(device) for t in host)
    q, k, v = (t.float().permute(0, 2, 1, 3) for t in host)
    k = k.repeat_interleave(heads // kv_heads, dim=1)
    v = v.repeat_interleave(heads // kv_heads, dim=1)
    scores = q @ k.transpose(-1, -2) * dim**-0.5 * 0.75
    if score is _coordinate_score:
        scores += (
            torch.arange(heads)[:, None, None] * 0.01
            + torch.arange(sq)[None, :, None] * 0.02
            - torch.arange(sk)[None, None, :] * 0.03
        )
    # Partial tail first, then two full blocks in reverse order. Counts vary
    # by batch; indices broadcast. 165 packed rows must share the one Q list.
    pc = torch.ones(batch, 1, 1, dtype=torch.int32)
    fc = torch.full_like(pc, 2)
    pc[0] = 0
    fc[0] = 0
    blocks = BlockSparseTensorsTorch(
        pc.to(device),
        torch.tensor([[[[3]]]], dtype=torch.int32, device=device),
        fc.to(device),
        torch.tensor([[[[1, 0]]]], dtype=torch.int32, device=device),
        block_size=(128, 128),
    )
    interface._compiled_kernels.clear()
    try:
        for iteration in range(3):
            if iteration == 1:
                fc[1] = 1
                blocks.full_block_cnt.copy_(fc)
            elif iteration == 2:
                blocks = blocks._replace(
                    **{name: tensor.clone() for name, tensor in zip(blocks._fields[:4], blocks[:4])}
                )
                inputs = tuple(t.clone() for t in inputs)
            visible = torch.zeros(batch, 1, sq, sk, dtype=torch.bool)
            for b in range(batch):
                if pc[b, 0, 0]:
                    visible[b, 0, :, 384:] = torch.arange(5)[None, :] <= torch.arange(sq)[:, None]
                if fc[b, 0, 0] >= 1:
                    visible[b, 0, :, 128:256] = True
                if fc[b, 0, 0] == 2:
                    visible[b, 0, :, :128] = True
            masked = scores.masked_fill(~visible, -torch.inf)
            expected_lse = masked.logsumexp(-1)
            expected = (masked.softmax(-1).nan_to_num() @ v).permute(0, 2, 1, 3)
            for return_lse in (False, True):
                out, lse = interface.flash_attn_func(
                    *inputs,
                    mask_mod=mask,
                    score_mod=score,
                    aux_scalars=(int(iteration > 0),),
                    block_sparse_tensors=blocks,
                    return_lse=return_lse,
                )
                torch.npu.synchronize()
                torch.testing.assert_close(out.cpu().float(), expected, rtol=0, atol=0.05)
                assert torch.count_nonzero(out.cpu()[0]) == 0
                if return_lse:
                    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0, atol=0.05)
                else:
                    assert lse is None
            assert len(interface._compiled_kernels) == 2
    finally:
        interface._compiled_kernels.clear()


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"),
    reason="set FLASH_ATTN_DSL_TEST_DEVICE to opt into NPU execution",
)
@pytest.mark.parametrize(
    "dtype,mask,score,heads,kv_heads",
    [
        (torch.float16, None, None, 33, 1),
        (torch.bfloat16, _vector_head_mask, _coordinate_score, 33, 1),
        (torch.float16, _head_mask, _vector_scale_score, 33, 1),
        (torch.bfloat16, _vector_head_mask, _vector_scale_score, 33, 1),
        (torch.float16, _head_mask, _coordinate_score, 3, 1),
        (torch.float16, _head_mask, None, 2, 1),
        (torch.float16, None, _coordinate_score, 5, 1),
        (torch.float16, _head_mask, _scale_score, 4, 1),
        (torch.bfloat16, _vector_head_mask, _coordinate_score, 66, 2),
        (torch.float16, _head_mask, _vector_scale_score, 66, 2),
        (torch.float16, _head_mask, _coordinate_score, 6, 2),
    ],
)
def test_short_queries_preserve_callback_coordinates_and_lse(dtype, mask, score, heads, kv_heads):
    import torch_npu  # noqa: F401

    device = int(os.environ["FLASH_ATTN_DSL_TEST_DEVICE"])
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(29)
    interface._compiled_kernels.clear()
    try:
        lengths = [(5, 17), (7, 129)]
        if mask is _head_mask and score is _vector_scale_score:
            # Cross the packing boundary, then reuse the earlier packed variant.
            lengths.extend(((33, 65), (5, 17)))
        for sq, sk in lengths:
            q = torch.randn(32, sq, heads, 64, generator=generator).to(dtype)
            k = torch.randn(32, sk, kv_heads, 64, generator=generator).to(dtype)
            v = torch.randn(32, sk, kv_heads, 64, generator=generator).to(dtype)
            ref_k = k.float().repeat_interleave(heads // kv_heads, dim=2)
            ref_v = v.float().repeat_interleave(heads // kv_heads, dim=2)
            scores = q.float().permute(0, 2, 1, 3) @ ref_k.permute(0, 2, 3, 1) / 8
            if score is not None:
                scores *= 0.75
            if score is _coordinate_score:
                scores += (
                    torch.arange(heads)[:, None, None] * 0.01
                    + torch.arange(sq)[None, :, None] * 0.02
                    - torch.arange(sk)[None, None, :] * 0.03
                )
            if mask is not None:
                keep = (torch.arange(heads)[:, None, None] != 0) & (
                    torch.arange(sk)[None, None, :] <= torch.arange(sq)[None, :, None] + sk - sq
                )
                scores = scores.masked_fill(~keep, float("-inf"))
            expected_lse = scores.logsumexp(-1)
            expected = (scores.softmax(-1).nan_to_num() @ ref_v.permute(0, 2, 1, 3)).permute(
                0, 2, 1, 3
            )
            inputs = tuple(t.to(f"npu:{device}") for t in (q, k, v))
            for return_lse in (False, True):
                out, lse = interface.flash_attn_func(
                    *inputs, mask_mod=mask, score_mod=score, return_lse=return_lse
                )
                torch.npu.synchronize()
                assert out.is_contiguous()
                torch.testing.assert_close(out.cpu().float(), expected, rtol=0, atol=0.05)
                if return_lse:
                    assert lse.is_contiguous()
                    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0, atol=0.05)
                else:
                    assert lse is None
                if mask is not None:
                    assert torch.count_nonzero(out.cpu()[:, :, 0]) == 0
            if mask is None and score is None:
                assert len(interface._compiled_kernels) == 2
    finally:
        interface._compiled_kernels.clear()


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"),
    reason="set FLASH_ATTN_DSL_TEST_DEVICE to opt into NPU execution",
)
@pytest.mark.parametrize(
    "dtype,dim,heads,kv_heads,lengths,window,score",
    [
        (torch.float16, 8, 48, 2, ((16, 257), (32, 389)), (None, 0), None),
        (torch.float16, 16, 13, 1, ((1, 129), (5, 257)), (None, 0), None),
        (torch.bfloat16, 64, 33, 1, ((1, 2), (2, 2)), (None, 0), None),
        (torch.float16, 64, 128, 1, ((1, 256), (2, 257)), (None, 0), None),
        (torch.float16, 32, 66, 2, ((1, 2), (2, 2)), (0, 0), _vector_scale_score),
        (torch.bfloat16, 96, 34, 2, ((9, 1), (7, 129)), (0, 0), None),
        (torch.float16, 80, 33, 1, ((5, 389), (7, 129)), (128, 32), None),
        (torch.float16, 64, 33, 1, ((5, 1024), (7, 1025)), (128, 32), None),
        (torch.bfloat16, 64, 33, 1, ((5, 257), (7, 129)), (0, None), _coordinate_score),
        (torch.float16, 128, 32, 1, ((4, 257), (8, 389)), (None, 0), _vector_scale_score),
        (torch.float16, 128, 16, 4, ((33, 257), (45, 389)), (None, 0), _vector_scale_score),
        (torch.bfloat16, 96, 30, 2, ((35, 1), (45, 129)), (0, 0), None),
    ],
)
def test_packed_builtin_windows_preserve_query_coordinates(
    dtype, dim, heads, kv_heads, lengths, window, score
):
    import torch_npu  # noqa: F401

    device = int(os.environ["FLASH_ATTN_DSL_TEST_DEVICE"])
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(29)
    interface._compiled_kernels.clear()
    try:
        # A packed tile can begin/end within one Q's head group. Changing lengths
        # also checks current dynamic metadata, including empty tasks before live ones.
        for sq, sk in (*lengths, lengths[0]):
            q = torch.randn(32, sq, heads, dim, generator=generator).to(dtype)
            k = torch.randn(32, sk, kv_heads, dim, generator=generator).to(dtype)
            v = torch.randn(32, sk, kv_heads, dim, generator=generator).to(dtype)
            ref_k = k.float().repeat_interleave(heads // kv_heads, dim=2)
            ref_v = v.float().repeat_interleave(heads // kv_heads, dim=2)
            scores = q.float().permute(0, 2, 1, 3) @ ref_k.permute(0, 2, 3, 1) * dim**-0.5
            if score is not None:
                scores *= 0.75
            if score is _coordinate_score:
                scores = (
                    scores
                    + torch.arange(heads)[:, None, None] * 0.01
                    + torch.arange(sq)[None, :, None] * 0.02
                    - torch.arange(sk)[None, None, :] * 0.03
                )
            qi, ki = torch.arange(sq)[:, None], torch.arange(sk)[None, :]
            keep = torch.ones(sq, sk, dtype=torch.bool)
            left, right = window
            if left is not None:
                keep &= ki >= qi + sk - sq - left
            if right is not None:
                keep &= ki <= qi + sk - sq + right
            scores.masked_fill_(~keep, float("-inf"))
            expected_lse = scores.logsumexp(-1)
            expected = (scores.softmax(-1).nan_to_num() @ ref_v.permute(0, 2, 1, 3)).permute(
                0, 2, 1, 3
            )
            inputs = tuple(t.to(f"npu:{device}") for t in (q, k, v))
            options = {"causal": True} if window == (None, 0) else {"window_size": window}
            for return_lse in (False, True):
                out, lse = interface.flash_attn_func(
                    *inputs, score_mod=score, return_lse=return_lse, **options
                )
                torch.npu.synchronize()
                assert out.shape == q.shape and out.is_contiguous()
                torch.testing.assert_close(out.cpu().float(), expected, rtol=0, atol=0.05)
                assert torch.count_nonzero(out.cpu()[:, ~keep.any(-1)]) == 0
                if return_lse:
                    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0, atol=0.05)
                else:
                    assert lse is None
    finally:
        interface._compiled_kernels.clear()


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"),
    reason="set FLASH_ATTN_DSL_TEST_DEVICE to opt into NPU execution",
)
@pytest.mark.parametrize(
    "mode,dtype,dim",
    [
        ("dense", torch.float16, 128),
        ("causal", torch.bfloat16, 96),
        ("window", torch.float16, 64),
        ("left_window", torch.bfloat16, 96),
        ("right_window", torch.float16, 64),
        ("zero_window", torch.float16, 128),
        ("packed_dense", torch.float16, 8),
        ("packed_dense", torch.bfloat16, 96),
    ],
)
def test_public_attention_reuses_dynamic_lengths(monkeypatch, mode, dtype, dim):
    import torch_npu  # noqa: F401

    device = int(os.environ["FLASH_ATTN_DSL_TEST_DEVICE"])
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(17)
    original_compile = interface.tla.compile
    compiled = []

    def counted_compile(*args, **kwargs):
        result = original_compile(*args, **kwargs)
        compiled.append(result)
        return result

    monkeypatch.setattr(interface.tla, "compile", counted_compile)
    monkeypatch.setenv("CATLASS_DSL_CACHE", "1")
    monkeypatch.setenv("CATLASS_DSL_FORCE_RECOMPILE", "0")
    interface._compiled_kernels.clear()
    windows = {
        "window": (64, 16),
        "left_window": (0, -1),
        "right_window": (-1, 0),
        "zero_window": (0, 0),
    }
    options = (
        {}
        if mode in ("dense", "packed_dense")
        else {"causal": True}
        if mode == "causal"
        else {"window_size": windows[mode]}
    )
    try:
        batch, heads, kv_heads = (32, 34, 2) if mode == "packed_dense" else (2, 2, 1)
        lengths = (
            [(5, 65), (7, 129), (5, 65)]
            if mode == "packed_dense"
            else [(512, 512), (768, 1024), (1024, 768), (512, 512), (1, 257), (257, 1)]
        )
        for index, (sq, sk) in enumerate(lengths):
            q = torch.randn(batch, sq, heads, dim, generator=generator).to(dtype)
            k = torch.randn(batch, sk, kv_heads, dim, generator=generator).to(dtype)
            v = torch.randn(batch, sk, kv_heads, dim, generator=generator).to(dtype)
            runtime_inputs = tuple(t.to(f"npu:{device}") for t in (q, k, v))
            out, lse = interface.flash_attn_func(*runtime_inputs, return_lse=True, **options)
            torch.npu.synchronize()
            # Long aligned queries use DN. Short Q / KV calls use ND, but a
            # single Q removes the right bound and cannot supply the later
            # multi-row window variant.
            expected_compilations = 1 if mode == "packed_dense" or index < 4 else 2
            if index == 5 and mode in ("causal", "window", "right_window", "zero_window"):
                expected_compilations = 3
            assert len(compiled) == expected_compilations

            # The oracle does not call a modifier or use the device block schedule.
            ref_k = k.float().repeat_interleave(heads // kv_heads, dim=2)
            ref_v = v.float().repeat_interleave(heads // kv_heads, dim=2)
            scores = q.float().permute(0, 2, 1, 3) @ ref_k.permute(0, 2, 3, 1)
            scores *= dim**-0.5
            qi, ki = torch.arange(sq)[:, None], torch.arange(sk)[None, :]
            keep = torch.ones(sq, sk, dtype=torch.bool)
            if mode == "causal":
                keep = ki <= qi + sk - sq
            elif mode in windows:
                left, right = windows[mode]
                if left >= 0:
                    keep &= ki >= qi + sk - sq - left
                if right >= 0:
                    keep &= ki <= qi + sk - sq + right
            scores.masked_fill_(~keep, float("-inf"))
            expected_lse = torch.logsumexp(scores, dim=-1)
            expected = (scores.softmax(-1).nan_to_num(0) @ ref_v.permute(0, 2, 1, 3)).permute(
                0, 2, 1, 3
            )
            torch.testing.assert_close(out.cpu().float(), expected, rtol=0, atol=0.05)
            torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0, atol=0.05)
            assert torch.count_nonzero(out.cpu()[:, ~keep.any(-1)]) == 0
    finally:
        interface._compiled_kernels.clear()


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"),
    reason="set FLASH_ATTN_DSL_TEST_DEVICE to opt into NPU execution",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_len,group", [(32, 2), (33, 3), (5, 8), (1, 13), (7, 8), (5, 33)])
def test_packed_lse_preserves_head_and_query_order(dtype, q_len, group):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    batch, kv_heads, kv_len, dim = 64, 2, 65, 16
    generator = torch.Generator().manual_seed(29)
    host_q = torch.randn(batch, q_len, kv_heads * group, dim, generator=generator).to(dtype)
    q = host_q.to(device)
    k = torch.ones(batch, kv_len, kv_heads, dim, dtype=dtype, device=device)
    v = torch.ones_like(k)
    out, lse = interface.flash_attn_func(q, k, v, return_lse=True)
    torch.npu.synchronize()
    # Identical K rows make LSE analytic, but every batch/head/query is distinct.
    expected_lse = host_q.float().sum(-1).permute(0, 2, 1) / 4 + torch.log(torch.tensor(65.0))
    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0, atol=0.05)
    torch.testing.assert_close(out.cpu(), torch.ones_like(out.cpu()), rtol=0, atol=0.05)


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"),
    reason="set FLASH_ATTN_DSL_TEST_DEVICE to opt into NPU execution",
)
def test_dynamic_binding_matches_dlpack_and_uses_current_buffers(monkeypatch):
    import torch_npu  # noqa: F401
    from catlass.base_dsl.jit_executor import JitCompiledFunction
    from catlass.tla.runtime import from_dlpack

    device = int(os.environ["FLASH_ATTN_DSL_TEST_DEVICE"])
    torch.npu.set_device(device)
    launch = JitCompiledFunction.__call__
    executables = set()
    check_payload = True

    def checked_launch(compiled, *args, **kwargs):
        executables.add(id(compiled))
        if check_payload:
            expected = (
                tuple(
                    from_dlpack(
                        arg.tensor.reshape(-1, arg.columns), layout_tag=interface.tla.arch.RowMajor
                    ).mark_compact_shape_dynamic(0)
                    if arg is not None
                    else None
                    for arg in args[:5]
                )
                + args[5:]
            )
            assert compiled.execution_args.generate_launch_payload(
                args
            ) == compiled.execution_args.generate_launch_payload(expected)
        return launch(compiled, *args, **kwargs)

    def filled(shape, fill):
        # A contiguous view with nonzero storage offset must not add that offset
        # a second time when constructing the descriptor's data address.
        storage = torch.full(
            (7 + shape[1] * shape[2] * shape[3],), fill, dtype=torch.float16, device=f"npu:{device}"
        )
        return storage[7:].view(shape)

    def call(sq, sk, fill):
        return interface.flash_attn_func(
            filled((1, sq, 2, 64), 0),
            filled((1, sk, 1, 64), 0),
            filled((1, sk, 1, 64), fill),
            return_lse=True,
        )

    monkeypatch.setenv("CATLASS_DSL_CACHE", "1")
    monkeypatch.setenv("CATLASS_DSL_FORCE_RECOMPILE", "0")
    monkeypatch.setattr(JitCompiledFunction, "__call__", checked_launch)
    interface._compiled_kernels.clear()
    try:
        for sq, sk in ((1, 1), (1, 257), (63, 65), (129, 1)):
            call(sq, sk, 0)
        torch.npu.synchronize()
        # No comparison wrappers may retain the stress inputs or run DLPack
        # exporter side effects during the direct-only lifetime/stream test.
        check_payload = False

        def unexpected_conversion(*args, **kwargs):
            raise AssertionError("cache hit reconstructed a TLA Tensor")

        monkeypatch.setattr(interface, "from_dlpack", unexpected_conversion)
        monkeypatch.setattr(interface.tla, "compile", unexpected_conversion)
        import catlass.tla.runtime as tensor_runtime

        monkeypatch.setattr(tensor_runtime, "export_dlpack_capsule", unexpected_conversion)
        for stream in (torch.npu.current_stream(), torch.npu.Stream(device=device)):
            outputs = []
            with torch.npu.stream(stream):
                for index in range(50):
                    sq, sk = ((1, 257), (63, 65), (129, 1))[index % 3]
                    fill = index % 4 + 1
                    outputs.append((call(sq, sk, fill), fill, sk))
            stream.synchronize()
            assert len({out.data_ptr() for (out, _), _, _ in outputs}) == 50
            for (out, lse), fill, sk in outputs:
                assert torch.equal(out.cpu(), torch.full_like(out.cpu(), fill))
                torch.testing.assert_close(
                    lse.cpu(),
                    torch.full_like(lse.cpu(), float(torch.tensor(sk).log())),
                    rtol=0,
                    atol=1e-5,
                )
        assert len(executables) == 1
    finally:
        interface._compiled_kernels.clear()
