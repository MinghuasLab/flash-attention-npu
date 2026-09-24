"""Structured pytest events consumed by the CI progress watcher."""

import base64
import os
from pathlib import Path


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
    if not os.environ.get("PYTEST_XDIST_WORKER"):
        _write("total|%d" % len(session.items))


def pytest_runtest_logstart(nodeid, location):
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    _write("start|%s" % _encoded(nodeid))


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
        report.nodeid, (report.outcome, report.duration))
    timing = {}
    for key, value in getattr(report, "user_properties", ()):
        if key == "ci_timing":
            timing = value
    parts = timing.get("parts", {})
    events = ",".join(dict.fromkeys(timing.get("events", [])))
    tracked = sum(parts.get(name, 0) for name in (
        "input", "input_cpu", "h2d", "pack", "cache", "index",
        "postprocess", "forward", "backward", "ref", "golden", "compare"
    ))
    if any(parts.get(name, 0) for name in ("input_cpu", "h2d", "pack")):
        tracked -= parts.get("input", 0)
    other = max(0.0, timing.get("total", duration) - tracked)
    summary = (
        "total=%.2fs input=%.2fs cpu=%.2fs h2d=%.2fs pack=%.2fs cache=%.2fs index=%.2fs postprocess=%.2fs "
        "forward=%.2fs backward=%.2fs ref=%.2fs golden=%.2fs compare=%.2fs other=%.2fs%s"
        % (timing.get("total", duration), parts.get("input", 0),
           parts.get("input_cpu", 0), parts.get("h2d", 0), parts.get("pack", 0),
           parts.get("cache", 0), parts.get("index", 0),
           parts.get("postprocess", 0), parts.get("forward", 0),
           parts.get("backward", 0), parts.get("ref", 0),
           parts.get("golden", 0), parts.get("compare", 0), other,
           (" " + events) if events else "")
    )
    _write("result|%s|%.6f|%s|%s" % (
        outcome, duration, _encoded(report.nodeid), _encoded(summary)))
