from contextlib import nullcontext

import pytest

from tests.common.attention_ref import ref_flash_attention_pair
from tests.common.compare import assert_fa_close

from torch_compile_utils import (
    run_fixed_compile_test,
    load_api,
    require_soc,
    run_metadata_compile_test,
    run_varlen_compile_test,
    torch,
)


def test_fa3_910_scheduler_metadata_torch_compile_correctness():
    require_soc("910")

    api = load_api("flash_attn_npu_3.flash_attn_npu_interface")

    run_metadata_compile_test(
        api,
        expected_sizes={
            "NO_MASK": 2376,
            "CAUSAL": 2376,
            "LOCAL_LEFT": 2376,
            "LOCAL_RIGHT": 2376,
            "FULL_WINDOW_COLLAPSE": 2376,
        },
        tiling_only_metadata=True,
    )


def test_fa3_910_varlen_torch_compile_correctness():
    require_soc("910")

    api = load_api("flash_attn_npu_3.flash_attn_npu_interface")

    run_varlen_compile_test(
        api,
        backward=True,
    )


def test_fa3_950_scheduler_metadata_torch_compile_correctness():
    require_soc("950")

    api = load_api("flash_attn_npu_3.flash_attn_npu_interface_950")

    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        run_metadata_compile_test(
            api,
            expected_sizes=None,
            tiling_only_metadata=True,
        )


def test_fa3_950_varlen_torch_compile_correctness():
    require_soc("950")

    api = load_api("flash_attn_npu_3.flash_attn_npu_interface_950")

    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        run_varlen_compile_test(
            api,
            backward=False,
        )


def test_fa3_910_fixed_torch_compile_correctness():
    """
    Verify FA3 fixed-length API correctness.
    """

    require_soc("910")

    api = load_api("flash_attn_npu_3.flash_attn_npu_interface")

    run_fixed_compile_test(
        api,
        backward=True,
    )


class TestForwardRegistration:
    @pytest.mark.parametrize("execution", ["eager", "aot_eager"])
    @pytest.mark.parametrize("grad_input", ["q", "k", "v"])
    def test_kvcache_backward_rejected(self, execution, grad_input):
        require_soc("910")
        api = load_api("flash_attn_npu_3.flash_attn_npu_interface")

        inputs = {
            name: torch.ones((1, 4, 2, 64), dtype=torch.float16, device="npu")
            for name in ("q", "k", "v")
        }
        inputs[grad_input].requires_grad_(True)
        forward = api.flash_attn_with_kvcache
        if execution == "aot_eager":
            forward = torch.compile(forward, backend="aot_eager", fullgraph=True)
        # Unlike func/varlen, the cache API has no backward formula. AOT may
        # request the formula while compiling the forward graph.
        with pytest.raises(RuntimeError, match="no autograd formula was registered"):
            out = forward(inputs["q"], inputs["k"], inputs["v"], num_splits=1)
            out.sum().backward()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("execution", ["eager", "no_grad", "inference"])
    def test_kvcache_fresh_outputs(self, dtype, execution):
        require_soc("910")
        api = load_api("flash_attn_npu_3.flash_attn_npu_interface")
        q = torch.zeros((1, 4, 2, 64), dtype=dtype, device="npu")
        k = torch.zeros_like(q)
        v = torch.ones_like(q)
        versions = [t._version for t in (q, k, v)]
        context = {
            "eager": nullcontext,
            "no_grad": torch.no_grad,
            "inference": torch.inference_mode,
        }[execution]
        with context():
            first = api.flash_attn_with_kvcache(q, k, v, num_splits=1, return_softmax_lse=True)
            second = api.flash_attn_with_kvcache(q, k, v, num_splits=1, return_softmax_lse=True)
        tensors = (q, k, v, *first, *second)
        assert len({t.untyped_storage()._cdata for t in tensors}) == len(tensors)
        assert [t._version for t in (q, k, v)] == versions
        torch.testing.assert_close(first[0], v, rtol=0, atol=0)
        torch.testing.assert_close(second[0], v, rtol=0, atol=0)
        assert torch.count_nonzero(q) == torch.count_nonzero(k) == 0
        assert torch.all(v == 1)

    @pytest.mark.parametrize("invalid_dtype", [False, True])
    def test_forward_preallocated_output_rejected(self, invalid_dtype):
        require_soc("910")
        api = load_api("flash_attn_npu_3.flash_attn_npu_interface")

        q = torch.ones((1, 4, 2, 64), dtype=torch.float16, device="npu")
        out = torch.empty_like(q, dtype=torch.float32 if invalid_dtype else q.dtype)
        error = "output must have the same dtype" if invalid_dtype else "may not alias"
        with pytest.raises(RuntimeError, match=error):
            api._flash_attn_forward(q, q, q, out_=out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
def test_fa3_forward_schema_and_fake(dtype, strided):
    require_soc("910")
    api = load_api("flash_attn_npu_3.flash_attn_npu_interface")
    shape = (1, 2, 2, 128 if strided else 64)
    q = torch.zeros(shape, dtype=dtype, device="npu")
    k = torch.zeros_like(q)
    v = torch.ones_like(q)
    if strided:
        q, k, v = (tensor[..., ::2] for tensor in (q, k, v))
    torch.library.opcheck(
        api._flash_attn_forward,
        (q, k, v),
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("entry", ["func", "kvcache"])
@pytest.mark.parametrize("execution", ["eager", "aot_eager"])
@pytest.mark.parametrize("causal", [False, True], ids=["unmasked", "causal"])
def test_fa3_bnsd_public_forward(dtype, entry, execution, causal):
    require_soc("910")
    api = load_api("flash_attn_npu_3.flash_attn_npu_interface")
    batch, heads, sq, sk, dim = 2, 4, 7, 19, 64
    # The public inputs retain BSND shapes and expose contiguous BNSD storage.
    q, k, v = (
        torch.randn((batch, heads, seq, dim), dtype=dtype).to("npu").transpose(1, 2)
        for seq in (sq, sk, sk)
    )
    scale = dim**-0.5
    mask = (
        torch.triu(torch.ones(sq, sk, dtype=torch.bool), diagonal=sk - sq + 1) if causal else None
    )
    out_ref, lse_ref, out_pt, lse_pt = ref_flash_attention_pair(
        q.cpu(), k.cpu(), v.cpu(), scale, mask, dtype
    )
    forward = api.flash_attn_func if entry == "func" else api.flash_attn_with_kvcache
    if execution == "aot_eager":
        forward = torch.compile(forward, backend="aot_eager", fullgraph=True)
    options = {"return_attn_probs": True} if entry == "func" else {"return_softmax_lse": True}
    out, lse, *_ = forward(q, k, v, softmax_scale=scale, causal=causal, num_splits=1, **options)
    assert out.shape == q.shape
    assert_fa_close(out.cpu(), out_ref, out_pt)
    assert_fa_close(lse.cpu(), lse_ref, lse_pt, name="lse")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fa3_bnsd_schema_and_fake(dtype):
    require_soc("910")
    api = load_api("flash_attn_npu_3.flash_attn_npu_interface")
    q = torch.zeros((2, 4, 7, 64), dtype=dtype, device="npu")
    k = torch.zeros((2, 4, 19, 64), dtype=dtype, device="npu")
    v = torch.ones_like(k)
    torch.library.opcheck(
        api._flash_attn_forward,
        (q, k, v),
        {"layout": api._LAYOUT_BNSD},
        test_utils=("test_schema", "test_faketensor"),
    )
