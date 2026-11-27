# Copyright (c) 2026, Minghua Shen.

from torch_compile_utils import (
    run_fixed_compile_test,
    load_api,
    require_soc,
    run_metadata_compile_test,
    run_varlen_compile_test,
)


def test_fa2_910_scheduler_metadata_torch_compile_correctness():
    require_soc("910")

    api = load_api("flash_attn_npu.flash_attn_npu_interface")

    run_metadata_compile_test(
        api,
        expected_sizes={
            "NO_MASK": 2448,
            "CAUSAL": 2448,
            "LOCAL_LEFT": 2448,
            "LOCAL_RIGHT": 2448,
            "FULL_WINDOW_COLLAPSE": 2448,
        },
        tiling_only_metadata=True,
    )


def test_fa2_910_varlen_torch_compile_correctness():
    require_soc("910")

    api = load_api("flash_attn_npu.flash_attn_npu_interface")

    run_varlen_compile_test(
        api,
        backward=True,
    )


def test_fa2_910_fixed_torch_compile_correctness():
    """
    Verify FA2 fixed-length API correctness.

    Compare eager execution with torch.compile execution.
    """

    require_soc("910")

    api = load_api("flash_attn_npu.flash_attn_npu_interface")

    run_fixed_compile_test(
        api,
        backward=True,
    )
