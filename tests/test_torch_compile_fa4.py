import pytest
from torch_compile_utils import (
    run_fixed_compile_test,
    load_api,
    require_soc,
    run_metadata_compile_test,
    run_varlen_compile_test,
)


def test_fa4_910_scheduler_metadata_torch_compile_correctness():
    require_soc("910")

    api = load_api(
        "flash_attn_npu_4.flash_attn_npu_interface"
    )

    run_metadata_compile_test(
        api,
        expected_sizes={
            "NO_MASK": 2376,
            "CAUSAL": 4196680,
            "LOCAL_LEFT": 4196680,
            "LOCAL_RIGHT": 4196680,
            "FULL_WINDOW_COLLAPSE": 2376,
        },
    )


def test_fa4_910_varlen_torch_compile_correctness():
    require_soc("910")

    api = load_api(
        "flash_attn_npu_4.flash_attn_npu_interface"
    )

    run_varlen_compile_test(
        api,
        backward=True,
    )


def test_fa4_950_scheduler_metadata_torch_compile_correctness():
    pytest.skip("Ascend950 does not support scheduler metadata currently")
    require_soc("950")

    api = load_api(
        "flash_attn_npu_4.flash_attn_npu_interface_950"
    )

    run_metadata_compile_test(
        api,
        expected_sizes=None,
    )


def test_fa4_950_varlen_torch_compile_correctness():
    require_soc("950")

    api = load_api(
        "flash_attn_npu_4.flash_attn_npu_interface_950"
    )

    run_varlen_compile_test(
        api,
        backward=False,
    )


def test_fa4_910_fixed_torch_compile_correctness():

    """
    Verify FA4 fixed-length API correctness.
    """

    require_soc("910")

    api = load_api(
        "flash_attn_npu_4.flash_attn_npu_interface"
    )

    run_fixed_compile_test(
        api,
        backward=True,
    )

