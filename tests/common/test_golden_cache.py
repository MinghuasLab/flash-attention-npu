# Copyright (c) 2026, Minghua Shen.

import torch

from tests.common.attention_ref import (
    _ref_flash_attention_pair,
    cached_autograd_grads,
    cached_ref_flash_attention_pair,
    ref_flash_attention_pair,
)
from tests.common.compare import assert_fa_close
from tests.common.golden_cache import get_or_compute_golden, input_digest, register_retry


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


def test_large_tensor_digest_samples_instead_of_full_hash():
    """Lookups must not SHA256 hundreds of MB, but same-shape slices must differ."""
    small = torch.ones(16)
    # 65 * 256 * 4 = 66560 bytes, just above _MAX_DIGEST_BYTES (64KiB).
    large_a = torch.zeros(65, 256)
    large_b = torch.zeros(65, 256)
    large_b[-1, -1] = 1
    small_digest = input_digest(small)
    digest_a = input_digest(large_a)
    digest_b = input_digest(large_b)
    assert "sha256" in small_digest
    assert "sha256" in digest_a
    assert digest_a["shape"] == [65, 256]
    assert digest_a["numel"] == 65 * 256
    assert digest_a["sha256"] != digest_b["sha256"]


def test_ref_pair_caches_requires_grad_inputs(tmp_path, monkeypatch):
    """Backward/v4 cases pass requires_grad tensors; they must still cache."""
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    stats_file = tmp_path / "events.tsv"
    monkeypatch.setenv("GOLDEN_CACHE_STATS_FILE", str(stats_file))
    monkeypatch.setenv("GOLDEN_CACHE_TEST_FILE", "tests/test_example.py")
    query = torch.ones((1, 2, 1, 2), dtype=torch.float32, requires_grad=True)
    key = torch.ones_like(query)
    value = torch.arange(4, dtype=torch.float32).reshape(1, 2, 1, 2)
    first = ref_flash_attention_pair(query, key, value, 1.0, None, torch.float32)
    second = ref_flash_attention_pair(query, key, value, 1.0, None, torch.float32)
    for left, right in zip(first, second):
        torch.testing.assert_close(left.detach(), right.detach())
    events = [line.split("\t", 1)[0] for line in stats_file.read_text().splitlines()]
    assert events == ["miss", "write_ok", "hit"]
    assert first[0].grad_fn is not None
    assert second[0].grad_fn is None


def test_cached_autograd_grads_after_forward_hit(tmp_path, monkeypatch):
    """A forward HIT is detached; grads must use cached_autograd_grads + recompute."""
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    query = torch.ones((1, 2, 1, 2), dtype=torch.float32, requires_grad=True)
    key = torch.ones((1, 2, 1, 2), dtype=torch.float32, requires_grad=True)
    value = torch.arange(4, dtype=torch.float32).reshape(1, 2, 1, 2).requires_grad_(True)
    ref_flash_attention_pair(query, key, value, 1.0, None, torch.float32)
    out_ref, _, out_pt, _ = ref_flash_attention_pair(query, key, value, 1.0, None, torch.float32)
    assert out_ref.grad_fn is None
    dout = torch.ones_like(out_ref)
    try:
        torch.autograd.grad(out_ref, (query, key, value), dout)
        raise AssertionError("detached HIT golden must not backward")
    except RuntimeError as exc:
        assert "does not require grad" in str(exc)

    recomputes = {"count": 0}

    def recompute():
        recomputes["count"] += 1
        live_ref, _, live_pt, _ = _ref_flash_attention_pair(
            query, key, value, 1.0, None, torch.float32
        )
        return live_ref, live_pt

    first = cached_autograd_grads(
        "tests/example.py::test_grad",
        (out_ref, out_pt),
        (query, key, value),
        dout,
        metadata={"kind": "func"},
        recompute_fn=recompute,
    )
    second = cached_autograd_grads(
        "tests/example.py::test_grad",
        (out_ref, out_pt),
        (query, key, value),
        dout,
        metadata={"kind": "func"},
        recompute_fn=recompute,
    )
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right)
    assert recomputes["count"] == 1


def test_large_input_cache_lookup_does_not_hash_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    # 8 MiB, well above the 64KiB digest cap.  A full SHA256 of this on every
    # lookup is what made warm CI runs as slow as cold ones.
    payload = torch.randn(32, 1024, 8, 8)
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return {"out": torch.ones(1)}

    kwargs = dict(
        nodeid="tests/example.py::test_large",
        metadata={"seed": 0},
        inputs={"q": payload},
        compute_fn=compute,
        expected_keys=("out",),
    )
    get_or_compute_golden(**kwargs)
    import time

    started = time.perf_counter()
    get_or_compute_golden(**kwargs)
    elapsed = time.perf_counter() - started
    assert calls["count"] == 1
    assert elapsed < 0.5, f"large-input cache hit took {elapsed:.3f}s"


def test_large_same_shape_different_values_do_not_collide(tmp_path, monkeypatch):
    """new_kv goldens call the cache once per batch with the same nodeid/shape."""
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return {"out": torch.tensor([calls["count"]], dtype=torch.float32)}

    base = dict(
        nodeid="tests/example.py::test_new_kv",
        metadata={"seed": 0},
        compute_fn=compute,
        expected_keys=("out",),
    )
    first = get_or_compute_golden(inputs={"q": torch.zeros(65, 256)}, **base)
    second = get_or_compute_golden(inputs={"q": torch.ones(65, 256)}, **base)
    assert first["out"].item() == 1
    assert second["out"].item() == 2
    assert calls["count"] == 2


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
