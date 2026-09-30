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
        count = len(session.items)
        if count > 0:
            _write("total|%d" % count)


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
    # All displayed fields are mutually exclusive time segments.
    # "input" is the parent of rand+h2d, so only children are counted to avoid overlap.
    # "ref" and "golden" are disjoint (golden_cache mutes the reference helper's
    # own timer while it runs compute_fn), so summing them double-counts nothing.
    tracked = sum(parts.get(name, 0) for name in (
        "input_cpu", "h2d", "pack",  "kv", "index",
        "postprocess", "forward", "backward", "ref", "golden", "compare"
    ))
    other = max(0.0, timing.get("total", duration) - tracked)
    summary = (
        "total=%.2fs rand=%.2fs h2d=%.2fs pack=%.2fs cache=%.2fs index=%.2fs postprocess=%.2fs "
        "forward=%.2fs backward=%.2fs ref=%.2fs golden=%.2fs compare=%.2fs other=%.2fs%s"
        % (timing.get("total", duration),
           parts.get("input_cpu", 0), parts.get("h2d", 0), parts.get("pack", 0),
           parts.get( "kv", 0), parts.get("index", 0),
           parts.get("postprocess", 0), parts.get("forward", 0),
           parts.get("backward", 0), parts.get("ref", 0),
           parts.get("golden", 0), parts.get("compare", 0), other,
           (" " + events) if events else "")
    )
    _write("result|%s|%.6f|%s|%s" % (
        outcome, duration, _encoded(report.nodeid), _encoded(summary)))
