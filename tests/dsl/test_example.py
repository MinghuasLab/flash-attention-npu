"""Package CLI coverage for dtypes, head dimensions and modifier execution modes."""

from importlib.util import find_spec
import os
import subprocess
import sys

import pytest

if find_spec("catlass") is None:
    pytest.skip("CATLASS DSL runtime is not installed", allow_module_level=True)


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"),
    reason="set FLASH_ATTN_DSL_TEST_DEVICE to opt into NPU execution",
)
@pytest.mark.parametrize(
    "dtype,dim,q_len,kv_len,extra",
    [
        ("fp16", 64, 1, 257, ["--no-lse"]),
        ("bf16", 96, 129, 257, []),
        ("fp16", 128, 257, 129, ["--sparse"]),
        ("bf16", 64, 129, 257, ["--simd", "--sparse"]),
        ("fp16", 96, 257, 129, ["--simd", "--sparse", "--no-lse"]),
        ("bf16", 128, 257, 513, ["--simd"]),
        ("fp16", 64, 129, 257, ["--simd-mask"]),
        ("bf16", 96, 257, 129, ["--simd-score", "--sparse"]),
    ],
)
def test_package_example(dtype, dim, q_len, kv_len, extra):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flash_attn_npu_dsl.example",
            "--device",
            str(int(os.environ["FLASH_ATTN_DSL_TEST_DEVICE"])),
            "--dtype",
            dtype,
            "--head-dim",
            str(dim),
            "--q-len",
            str(q_len),
            "--kv-len",
            str(kv_len),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "绝对误差 ≤ 0.05" in result.stdout
