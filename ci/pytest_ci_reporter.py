# Copyright (c) 2026, Minghua Shen.

"""Structured pytest events consumed by the CI progress watcher."""

import base64
from collections import defaultdict
from functools import wraps
import json
import os
from pathlib import Path
from time import perf_counter

import pytest


def _encoded(value):
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _write(event):
    path = os.environ.get("CI_PROGRESS_FILE")
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(event + "\n")
        stream.flush()


def pytest_configure(config):
    path = os.environ.get("CI_PROGRESS_FILE")
    if path and not os.environ.get("PYTEST_XDIST_WORKER"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("", encoding="utf-8")


def pytest_collection_finish(session):
    manifest = os.environ.get("CI_COLLECT_MANIFEST")
    if manifest and session.config.option.collectonly:
        Path(manifest).write_text(
            "".join(f"{item.path.resolve()}\t{_encoded(item.nodeid)}\n" for item in session.items),
            encoding="utf-8",
        )
    if not os.environ.get("PYTEST_XDIST_WORKER"):
        count = len(session.items)
        if count > 0:
            _write("total|%d" % count)


def pytest_runtest_logstart(nodeid, location):
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    _write("start|%s" % _encoded(nodeid))


def _timed(function, phase, timings):
    @wraps(function)
    def measure(*args, **kwargs):
        started = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            entry = timings[phase]
            entry["calls"] += 1
            entry["seconds"] += perf_counter() - started

    return measure


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if not os.environ.get("CI_DET_BWD_TIMING_FILE") or item.path.name not in (
        "test_flash_attn_npu_v3_bwd_det.py",
        "test_flash_attn_npu_v4_bwd_det.py",
    ):
        yield
        return

    import torch

    timings = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    original_cpu = torch.Tensor.cpu
    timed_cpu = _timed(original_cpu, "d2h", timings)

    @wraps(original_cpu)
    def tensor_cpu(tensor, *args, **kwargs):
        transfer = timed_cpu if tensor.device.type == "npu" else original_cpu
        return transfer(tensor, *args, **kwargs)

    # Observe the real calls, including the existing syncs, without inserting
    # device waits into the test or timing the CPU-side Golden tensor copies.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", tensor_cpu)
        for owner, name, phase in (
            (item.module, "_flash_attn_backward", "backward_call"),
            (torch.npu, "synchronize", "synchronize"),
            (torch, "equal", "equal"),
            (torch.testing, "assert_close", "assert_close"),
            (item.module, "max_diff", "max_diff"),
        ):
            patch.setattr(owner, name, _timed(getattr(owner, name), phase, timings))
        yield

    item.user_properties.append(("ci_det_bwd_timing", dict(timings)))


def _write_det_bwd_timing(nodeid, duration, phases):
    path = os.environ.get("CI_DET_BWD_TIMING_FILE")
    if not path:
        return
    row = dict(nodeid=nodeid, duration_s=duration, phases=phases)
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def pytest_runtest_logreport(report):
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    if not hasattr(pytest_runtest_logreport, "outcomes"):
        pytest_runtest_logreport.outcomes = {}
    if report.when == "call":
        pytest_runtest_logreport.outcomes[report.nodeid] = (report.outcome, report.duration)
        return
    if report.when != "teardown":
        return
    outcome, duration = pytest_runtest_logreport.outcomes.pop(
        report.nodeid, (report.outcome, report.duration)
    )
    timing = {}
    for key, value in getattr(report, "user_properties", ()):
        if key == "ci_timing":
            timing = value
        if key == "ci_det_bwd_timing":
            _write_det_bwd_timing(report.nodeid, duration, value)
    parts = timing.get("parts", {})
    events = ",".join(dict.fromkeys(timing.get("events", [])))
    # All displayed fields are mutually exclusive time segments.
    # "input" is the parent of rand+h2d, so only children are counted to avoid overlap.
    # "ref" and "golden" are disjoint (golden_cache mutes the reference helper's
    # own timer while it runs compute_fn), so summing them double-counts nothing.
    tracked = sum(
        parts.get(name, 0)
        for name in (
            "input_cpu",
            "h2d",
            "pack",
            "kv",
            "index",
            "postprocess",
            "forward",
            "backward",
            "ref",
            "golden",
            "compare",
        )
    )
    other = max(0.0, timing.get("total", duration) - tracked)
    summary = (
        "total=%.2fs rand=%.2fs h2d=%.2fs pack=%.2fs cache=%.2fs index=%.2fs postprocess=%.2fs "
        "forward=%.2fs backward=%.2fs ref=%.2fs golden=%.2fs compare=%.2fs other=%.2fs%s"
        % (
            timing.get("total", duration),
            parts.get("input_cpu", 0),
            parts.get("h2d", 0),
            parts.get("pack", 0),
            parts.get("kv", 0),
            parts.get("index", 0),
            parts.get("postprocess", 0),
            parts.get("forward", 0),
            parts.get("backward", 0),
            parts.get("ref", 0),
            parts.get("golden", 0),
            parts.get("compare", 0),
            other,
            (" " + events) if events else "",
        )
    )
    _write("result|%s|%.6f|%s|%s" % (outcome, duration, _encoded(report.nodeid), _encoded(summary)))
