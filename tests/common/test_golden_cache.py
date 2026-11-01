# Copyright (c) 2026, Minghua Shen.

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import threading
import time

import pytest
import torch

from tests.common.attention_ref import cached_autograd_grads, cached_ref_flash_attention_pair
from tests.common import attention_ref_fp8
from tests.common.compare import assert_fa_close
from tests.common.golden_cache import get_or_compute_golden, register_retry


def test_fp8_reference_uses_v2_cache_api(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    calls = {"high": 0, "low": 0}

    @dataclass(frozen=True)
    class Case:
        entry: str = "varlen"

    def reference(name):
        def compute(case, inputs):
            calls[name] += 1
            return torch.ones(1), torch.zeros(1)

        return compute

    monkeypatch.setattr(attention_ref_fp8, "high_reference", reference("high"))
    monkeypatch.setattr(attention_ref_fp8, "low_reference", reference("low"))
    inputs = {name: torch.ones(1) for name in ("q", "k", "v", "descales")}

    first = attention_ref_fp8.ref_flash_attention_pair(Case(), inputs)
    second = attention_ref_fp8.ref_flash_attention_pair(Case(), inputs)

    assert calls == {"high": 1, "low": 1}
    assert first[0][0].equal(second[0][0])
    assert list((tmp_path / "v2").rglob("api_flash_attn_varlen_func_*/case_*.tar.gz"))


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
    assert [event for event, scope in events if scope == "test"] == ["miss", "write_ok", "hit"]


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
        metadata={"seed": 0},
        inputs={"q": torch.ones(1)},
        compute_fn=compute,
        expected_keys=("out",),
    )
    get_or_compute_golden(**kwargs)
    values, status = get_or_compute_golden(**kwargs, return_status=True)
    assert status == "hit"
    register_retry(values, lambda: get_or_compute_golden(**kwargs, force_refresh=True))
    assert_fa_close(torch.tensor([2.0]), values["out"], values["out"], name="out")
    assert calls["count"] == 2
    assert not __import__(
        "tests.common.golden_cache", fromlist=["retry_cached_value"]
    ).retry_cached_value(values["out"])
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


def test_golden_cache_keeps_existing_cases_after_test_file_grows(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    test_file = tmp_path / "test_example.py"
    test_file.write_text("def test_attention_a(): pass\n")
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return {"out": torch.ones(1)}

    def run(case):
        _, status = get_or_compute_golden(
            nodeid=f"{test_file}::test_attention[{case}]",
            api_name="flash_attn_func",
            metadata={"case": case},
            inputs={"q": torch.tensor([case == "case_b"])},
            compute_fn=compute,
            expected_keys=("out",),
            return_status=True,
        )
        return status

    assert run("case_a") == "miss"
    original_source = test_file.read_text()
    test_file.write_text(original_source + "def test_attention_b(): pass\n")
    assert run("case_a") == "hit"
    assert run("case_b") == "miss"
    test_file.write_text(original_source)  # another PR with the old file revision
    assert run("case_a") == "hit"
    assert calls["count"] == 2

    assert len(list(tmp_path.glob("v2/common_*/test_*"))) == 1
    assert len(list(tmp_path.rglob("case_*.tar.gz"))) == 2


def test_golden_cache_separates_interfaces_for_one_case(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return {"out": torch.tensor([calls["count"]])}

    base = dict(
        nodeid="tests/example.py::test_case[x]",
        metadata={},
        inputs={"q": torch.ones(1)},
        compute_fn=compute,
        expected_keys=("out",),
    )
    first = get_or_compute_golden(api_name="flash_attn_func", **base)["out"].item()
    second = get_or_compute_golden(api_name="flash_attn_varlen_func", **base)["out"].item()
    repeated = get_or_compute_golden(api_name="flash_attn_func", **base)["out"].item()
    assert (first, second, repeated) == (1, 2, 1)
    assert len(list(tmp_path.glob("v2/common_*/test_*/api_*"))) == 2


def test_npu_golden_requires_interface_name(monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_TEST_FILE", "tests/test_flash_attn_npu_new.py")
    monkeypatch.delenv("GOLDEN_CACHE_API_NAME", raising=False)
    query = torch.ones((1, 1, 1, 1))
    with pytest.raises(ValueError, match="golden_api"):
        cached_ref_flash_attention_pair(query, query, query, 1.0, None, torch.float32)


def test_golden_cache_same_case_concurrent_miss_computes_once(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    calls = {"count": 0}
    calls_lock = threading.Lock()

    def compute():
        with calls_lock:
            calls["count"] += 1
        time.sleep(0.05)
        return {"out": torch.ones(1)}

    kwargs = dict(
        nodeid="tests/example.py::test_case[x]",
        api_name="flash_attn_func",
        metadata={},
        inputs={"q": torch.ones(1)},
        compute_fn=compute,
        expected_keys=("out",),
        return_status=True,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: get_or_compute_golden(**kwargs), range(2)))
    assert calls["count"] == 1
    assert sorted(status for _, status in results) == ["hit", "miss"]


def test_golden_cache_hit_updates_retention_time(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    monkeypatch.setenv("GOLDEN_CACHE_MAX_AGE_DAYS", "1")
    monkeypatch.setenv("GOLDEN_CACHE_EVICT_INTERVAL_SECONDS", "0")
    base = dict(
        api_name="flash_attn_func",
        metadata={},
        inputs={"q": torch.ones(1)},
        compute_fn=lambda: {"out": torch.ones(1)},
        expected_keys=("out",),
    )
    get_or_compute_golden(nodeid="tests/example.py::test_old", **base)
    old_artifact = next(tmp_path.rglob("case_*.tar.gz"))
    stale_time = time.time() - 2 * 86400
    os.utime(old_artifact, (stale_time, stale_time))
    get_or_compute_golden(nodeid="tests/example.py::test_old", **base)
    get_or_compute_golden(nodeid="tests/example.py::test_new", **base)
    assert old_artifact.is_file()


def test_golden_cache_size_limit_evicts_old_case(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    monkeypatch.setenv("GOLDEN_CACHE_MAX_BYTES", "1")
    monkeypatch.setenv("GOLDEN_CACHE_EVICT_INTERVAL_SECONDS", "0")
    base = dict(
        api_name="flash_attn_func",
        metadata={},
        inputs={"q": torch.ones(1)},
        compute_fn=lambda: {"out": torch.ones(1)},
        expected_keys=("out",),
    )
    get_or_compute_golden(nodeid="tests/example.py::test_a", **base)
    old_artifact = next(tmp_path.rglob("case_*.tar.gz"))
    get_or_compute_golden(nodeid="tests/example.py::test_b", **base)
    assert not old_artifact.exists()
    assert len(list(tmp_path.rglob("case_*.tar.gz"))) == 1


def test_gradient_cache_tracks_rebuild_function_source(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    refs = tuple(torch.ones(1) for _ in range(3))
    outputs = (torch.ones(1), torch.ones(1))
    dout = torch.ones(1)

    def original(q, k, v, differentiable=False):
        value = q * k * v
        return value, value

    def changed(q, k, v, differentiable=False):
        value = q * k * v
        return value + (q - 1) * 0.5, value

    kwargs = dict(
        nodeid="tests/example.py::test_grad",
        outputs=outputs,
        refs=refs,
        dout=dout,
        api_name="flash_attn_func",
    )
    first = cached_autograd_grads(rebuild=original, **kwargs)
    second = cached_autograd_grads(rebuild=changed, **kwargs)
    assert first[0].item() == 1.0
    assert second[0].item() == 1.5
    assert len(list(tmp_path.rglob("case_*.tar.gz"))) == 2


@pytest.mark.golden_api("flash_attn_func")
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
    assert len(list(tmp_path.glob("v2/common_*/test_*/api_flash_attn_func_*"))) == 1
