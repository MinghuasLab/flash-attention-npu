"""Cross-length executable reuse through the public attention interface."""

from importlib.util import find_spec
import os

import pytest

if find_spec("catlass") is None:
    pytest.skip("CATLASS DSL runtime is not installed", allow_module_level=True)
if find_spec("torch") is None:
    pytest.skip("PyTorch is not installed", allow_module_level=True)

import torch

from flash_attn_npu_dsl import interface


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"),
    reason="set FLASH_ATTN_DSL_TEST_DEVICE to opt into NPU execution",
)
@pytest.mark.parametrize(
    "mode,dtype,dim",
    [("dense", torch.float16, 128), ("causal", torch.bfloat16, 96), ("window", torch.float16, 64)],
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
    options = (
        {}
        if mode == "dense"
        else {"causal": True}
        if mode == "causal"
        else {"window_size": (64, 16)}
    )
    try:
        for sq, sk in [(512, 512), (768, 1024), (1024, 768), (512, 512), (1, 257), (257, 1)]:
            q = torch.randn(2, sq, 2, dim, generator=generator).to(dtype)
            k = torch.randn(2, sk, 1, dim, generator=generator).to(dtype)
            v = torch.randn(2, sk, 1, dim, generator=generator).to(dtype)
            runtime_inputs = tuple(t.to(f"npu:{device}") for t in (q, k, v))
            out, lse = interface.flash_attn_func(*runtime_inputs, return_lse=True, **options)
            torch.npu.synchronize()
            assert len(compiled) == 1

            # The oracle does not call a modifier or use the device block schedule.
            scores = q.float().permute(0, 2, 1, 3) @ k.float().permute(0, 2, 3, 1)
            scores *= dim**-0.5
            qi, ki = torch.arange(sq)[:, None], torch.arange(sk)[None, :]
            keep = torch.ones(sq, sk, dtype=torch.bool)
            if mode == "causal":
                keep = ki <= qi + sk - sq
            elif mode == "window":
                keep = (ki >= qi + sk - sq - 64) & (ki <= qi + sk - sq + 16)
            scores.masked_fill_(~keep, float("-inf"))
            expected_lse = torch.logsumexp(scores, dim=-1)
            expected = (scores.softmax(-1).nan_to_num(0) @ v.float().permute(0, 2, 1, 3)).permute(
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
