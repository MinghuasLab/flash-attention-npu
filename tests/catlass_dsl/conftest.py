"""Shared fixtures for CATLASS DSL tests."""

import os

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_per_case():
    # Override the repository timing fixture, which wraps attention callables.
    torch.manual_seed(0)
    yield


@pytest.fixture(scope="module")
def device():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    dev = torch.device("npu", int(os.environ.get("FLEX_NPU_DEVICE", "4")))
    torch.npu.set_device(dev)
    return dev
