# Copyright (c) 2026, Minghua Shen.

import torch

from tests.common.attention_ref import cached_ref_flash_attention_pair
from tests.common.golden_cache import get_or_compute_golden


def test_golden_cache_is_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("GOLDEN_CACHE_MODE", raising=False)
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return {"out": torch.ones(1)}

    kwargs = {
        "nodeid": "tests/example.py::test_disabled",
        "metadata": {},
        "inputs": {"q": torch.ones(1)},
        "compute_fn": compute,
        "expected_keys": ("out",),
    }
    get_or_compute_golden(**kwargs)
    get_or_compute_golden(**kwargs)

    assert calls["count"] == 2
    assert not list(tmp_path.iterdir())


def test_golden_cache_miss_hit_and_refresh(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return {"out": torch.tensor([calls["count"]], dtype=torch.float32)}

    kwargs = dict(
        nodeid="tests/example.py::test_case[x]",
        metadata={"seed": 7, "shape": [1]},
        inputs={"q": torch.ones(1)},
        compute_fn=compute,
        expected_keys=("out",),
    )
    assert get_or_compute_golden(**kwargs)["out"].item() == 1
    assert get_or_compute_golden(**kwargs)["out"].item() == 1
    assert calls["count"] == 1

    monkeypatch.setenv("GOLDEN_CACHE_REFRESH", "1")
    assert get_or_compute_golden(**kwargs)["out"].item() == 2
    assert calls["count"] == 2


def test_golden_cache_input_change_and_corruption_recompute(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return {"out": torch.tensor([3])}

    base = dict(
        nodeid="case",
        metadata={"seed": 0},
        compute_fn=compute,
        expected_keys=("out",),
    )
    get_or_compute_golden(inputs={"q": torch.zeros(1)}, **base)
    artifact = next(tmp_path.rglob("case_*.tar.gz"))
    artifact.write_bytes(b"broken")
    get_or_compute_golden(inputs={"q": torch.zeros(1)}, **base)
    assert calls["count"] == 2
    get_or_compute_golden(inputs={"q": torch.ones(1)}, **base)
    assert calls["count"] == 3


def test_golden_cache_groups_cases_by_test_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")

    def compute():
        return {"out": torch.ones(1)}

    for case in ("case_a", "case_b"):
        get_or_compute_golden(
            nodeid=f"tests/example.py::test_attention[{case}]",
            metadata={"case": case},
            inputs={"q": torch.tensor([case == "case_b"])},
            compute_fn=compute,
            expected_keys=("out",),
        )

    test_dirs = list(tmp_path.glob("common_*/test_*"))
    assert len(test_dirs) == 1
    assert len(list(test_dirs[0].glob("case_*.tar.gz"))) == 2


def test_cached_reference_supports_dropout(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    query = torch.ones((1, 2, 1, 2), dtype=torch.float32)
    key = torch.ones_like(query)
    value = torch.arange(4, dtype=torch.float32).reshape(1, 2, 1, 2)
    drop_mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    kwargs = {
        "query": query,
        "key": key,
        "value": value,
        "scale": 1.0,
        "mask": None,
        "data_type": torch.float32,
        "drop_mask": drop_mask,
        "dropout_p": 0.5,
        "nodeid": "tests/example.py::test_dropout",
    }

    first = cached_ref_flash_attention_pair(**kwargs)
    second = cached_ref_flash_attention_pair(**kwargs)
    for first_tensor, second_tensor in zip(first, second):
        torch.testing.assert_close(first_tensor, second_tensor)
