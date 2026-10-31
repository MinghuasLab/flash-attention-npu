import pytest

from torch_compile_utils import (
    run_fixed_compile_test,
    load_api,
    require_soc,
    run_metadata_compile_test,
    run_varlen_compile_test,
    torch,
)


@pytest.mark.parametrize("causal", [False, True])
def test_fa3_910_kvcache_metadata_copy_torch_compile(causal):
    require_soc("910")
    api = load_api("flash_attn_npu_3.flash_attn_npu_interface")
    query = torch.randn(16, 4, 128, dtype=torch.bfloat16, device="npu")
    key = torch.randn(1, 128, 2, 128, dtype=query.dtype, device=query.device)
    value = torch.randn_like(key)
    lengths = torch.tensor([128], dtype=torch.int32, device=query.device)
    cu_q = torch.tensor([0, 16], dtype=torch.int32, device=query.device)
    pages = torch.tensor([[0]], dtype=torch.int32, device=query.device)
    metadata_buffer = torch.empty(2376, dtype=torch.uint8, device=query.device)

    def fn(q, k, v, cache_lengths, metadata):
        fresh = api.get_scheduler_metadata(
            batch_size=1, max_seqlen_q=16, max_seqlen_k=128,
            num_heads_q=4, num_heads_kv=2, headdim=128,
            cache_seqlens=cache_lengths, qkv_dtype=q.dtype,
            cu_seqlens_q=cu_q, page_size=128, causal=causal, num_splits=1,
        )
        metadata.copy_(fresh.clone())
        return api.flash_attn_with_kvcache(
            q, k, v, cache_seqlens=cache_lengths, cu_seqlens_q=cu_q,
            max_seqlen_q=16, page_table=pages, causal=causal,
            scheduler_metadata=metadata, num_splits=1,
        )

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)
    for length in (128, 64, 96):
        lengths.fill_(length)
        expected = fn(query, key, value, lengths, torch.empty_like(metadata_buffer))
        actual = compiled(query, key, value, lengths, metadata_buffer)
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected)


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
