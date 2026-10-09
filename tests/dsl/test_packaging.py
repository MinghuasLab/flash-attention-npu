"""Inspect an existing wheel without importing the DSL or its dependencies.

Run with ``FLASH_ATTN_DSL_TEST_WHEEL=/path/to/package.whl python -m pytest
--confcutdir=tests/dsl tests/dsl/test_packaging.py``. No wheel is built or installed.
"""

import ast
import os
from pathlib import Path
import zipfile

import pytest


_WHEEL = os.environ.get("FLASH_ATTN_DSL_TEST_WHEEL")
if not _WHEEL:
    pytest.skip("Set FLASH_ATTN_DSL_TEST_WHEEL to inspect a built wheel", allow_module_level=True)

_PACKAGE = "flash_attn_npu_dsl"
_MODULES = {
    "__init__.py",
    "block_sparsity.py",
    "compute_block_sparsity.py",
    "example.py",
    "flash_fwd.py",
    "interface.py",
    "modifiers.py",
    "seqlen_info.py",
    "softmax.py",
}
_FORBIDDEN_ROOTS = {
    "examples",
    "flash_attn_npu",
    "flash_attn_npu_3",
    "flash_attn_npu_3_950",
    "flash_attn_npu_4",
    "flash_attn_npu_4_950",
}


def test_built_wheel_contains_standalone_dsl_package():
    wheel_path = Path(_WHEEL)
    assert wheel_path.is_file(), f"Wheel does not exist: {wheel_path}"
    assert wheel_path.suffix == ".whl", f"Expected a wheel artifact: {wheel_path}"

    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
        assert len(names) == len(set(names)), "Wheel contains duplicate archive members"
        package_files = {
            name.removeprefix(f"{_PACKAGE}/")
            for name in names
            if name.startswith(f"{_PACKAGE}/") and not name.endswith("/")
        }
        assert {name for name in package_files if name.endswith(".py")} == _MODULES
        assert "README.md" in package_files

        for module in sorted(_MODULES):
            member = f"{_PACKAGE}/{module}"
            tree = ast.parse(wheel.read(member), filename=member)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    imported = [node.module or ""]
                else:
                    continue
                for name in imported:
                    assert name.split(".", 1)[0] not in _FORBIDDEN_ROOTS, (
                        f"{member}:{node.lineno} imports external source package {name!r}"
                    )
