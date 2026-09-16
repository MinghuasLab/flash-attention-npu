from torch_compile_utils import (
    run_fixed_compile_test,
    load_api,
    require_soc,
    run_metadata_compile_test,
    run_varlen_compile_test,
)


def test_fa2_910_scheduler_metadata_torch_compile_correctness():
    require_soc("910")

    api = load_api(
        "flash_attn_npu.flash_attn_npu_interface"
    )

    run_metadata_compile_test(
        api,
        expected_sizes={
            "NO_MASK": 2416,
            "CAUSAL": 4196720,
            "LOCAL_LEFT": 4196720,
            "LOCAL_RIGHT": 4196720,
            "FULL_WINDOW_COLLAPSE": 2416,
        },
    )


def test_fa2_910_varlen_torch_compile_correctness():
    require_soc("910")

    api = load_api(
        "flash_attn_npu.flash_attn_npu_interface"
    )

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

    api = load_api(
        "flash_attn_npu.flash_attn_npu_interface"
    )

    run_fixed_compile_test(
        api,
        backward=True,
    )

