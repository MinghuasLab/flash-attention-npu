"""Shared fixtures for CATLASS DSL tests."""

import pytest
import torch

try:
    import torch_npu
except (ImportError, OSError):
    torch_npu = None


class _UnsupportedHardwareModule(pytest.Module):
    def collect(self):
        # Skip before importing tests that load CATLASS or the FA4 extension.
        pytest.skip("CATLASS DSL tests require Ascend950")


def pytest_pycollect_makemodule(module_path, parent):
    if (
        torch_npu is None
        or not torch_npu.npu.is_available()
        or "Ascend950" not in torch_npu.npu.get_device_name()
    ):
        return _UnsupportedHardwareModule.from_parent(parent, path=module_path)


@pytest.fixture(autouse=True)
def _seed_per_case():
    # Override the repository timing fixture, which wraps attention callables.
    torch.manual_seed(0)
    yield


@pytest.fixture(scope="module")
def device():
    pytest.importorskip("torch_npu")
    device_count = torch.npu.device_count()
    if device_count < 1:
        pytest.skip("Ascend NPU unavailable")
    return torch.device("npu:0")
