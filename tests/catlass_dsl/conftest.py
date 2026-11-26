# Copyright (c) 2026, Minghua Shen.

"""Shared fixtures for CATLASS DSL tests."""

import os

import pytest
import torch

from tests.common.golden_cache import get_or_compute_golden
from tests.common.timing import begin, finish

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
def _seed_per_case(request, monkeypatch):
    # Override the repository timing fixture, which wraps attention callables.
    torch.manual_seed(0)
    nodeid = request.node.nodeid
    monkeypatch.setenv("GOLDEN_CACHE_NODEID", nodeid)
    monkeypatch.setenv("GOLDEN_CACHE_TEST_FILE", str(request.path))
    state = begin()
    monkeypatch.setattr(get_or_compute_golden, "_ci_timing_state", state, raising=False)
    yield
    request.node.user_properties.append(("ci_timing", finish(state, nodeid)))


@pytest.fixture
def cpu_graph_threads():
    """Cap intra-op threads for CPU graph tests and restore the worker's pool."""
    limit = int(os.environ.get("CI_GRAPH_CPU_THREADS") or "0")
    previous = torch.get_num_threads()
    if not 0 < limit < previous:
        yield
        return

    torch.set_num_threads(limit)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def device():
    pytest.importorskip("torch_npu")
    device_count = torch.npu.device_count()
    if device_count < 1:
        pytest.skip("Ascend NPU unavailable")
    return torch.device("npu:0")
