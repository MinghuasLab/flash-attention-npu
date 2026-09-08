# Copyright (c) 2026, Minghua Shen.

import torch

from tests.common.attention_ref import cached_ref_flash_attention_pair
from tests.common.compare import assert_fa_close
from tests.common.golden_cache import get_or_compute_golden, register_retry


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


def test_golden_cache_records_events_outside_worker_stdout(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path / "cache"))
    stats_file = tmp_path / "events.tsv"
    monkeypatch.setenv("GOLDEN_CACHE_STATS_FILE", str(stats_file))
    monkeypatch.setenv("GOLDEN_CACHE_TEST_FILE", "tests/test_example.py")
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")

    kwargs = {
        "nodeid": "tests/test_example.py::test_case[x]",
        "metadata": {"seed": 0},
        "inputs": {"q": torch.ones(1)},
        "compute_fn": lambda: {"out": torch.ones(1)},
        "expected_keys": ("out",),
    }
    get_or_compute_golden(**kwargs)
    get_or_compute_golden(**kwargs)

    events = [line.split("\t", 2)[:2] for line in stats_file.read_text().splitlines()]
    assert [event for event, scope in events if scope == "test"] == [
        "miss", "write_ok", "hit"
    ]


def test_golden_cache_source_change_recomputes(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    source_file = tmp_path / "reference.py"
    source_file.write_text("VERSION = 1\n")
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return {"out": torch.tensor([calls["count"]])}

    kwargs = {
        "nodeid": "tests/example.py::test_case",
        "metadata": {"seed": 0},
        "inputs": {"q": torch.ones(1)},
        "compute_fn": compute,
        "expected_keys": ("out",),
        "source_files": [str(source_file)],
    }
    assert get_or_compute_golden(**kwargs)["out"].item() == 1
    source_file.write_text("VERSION = 2\n")
    assert get_or_compute_golden(**kwargs)["out"].item() == 2
    assert calls["count"] == 2


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


def test_cached_mismatch_recomputes_once(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return {"out": torch.tensor([2.0 if calls["count"] > 1 else 1.0])}

    kwargs = dict(
        nodeid="tests/example.py::test_retry",
        metadata={"seed": 0}, inputs={"q": torch.ones(1)},
        compute_fn=compute, expected_keys=("out",),
    )
    get_or_compute_golden(**kwargs)
    values, status = get_or_compute_golden(**kwargs, return_status=True)
    assert status == "hit"
    register_retry(values, lambda: get_or_compute_golden(**kwargs, force_refresh=True))
    assert_fa_close(torch.tensor([2.0]), values["out"], values["out"], name="out")
    assert calls["count"] == 2
    assert not __import__("tests.common.golden_cache", fromlist=["retry_cached_value"]).retry_cached_value(values["out"])
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
