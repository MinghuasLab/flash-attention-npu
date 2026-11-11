"""Opt-in, process-wide observations of captured backward compilation modes.

Collectors include Autograd worker threads; ContextVar values in those workers
are deliberately not compared with the saved forward policy. Each collector
owns a bounded event window, retains no tensors and has no effect on planning.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock, get_ident
from time import monotonic_ns


@dataclass(frozen=True)
class BackwardCompileEvent:
    """One successful forward capture or backward submission observation.

    ``kernel_submitted=False`` means the zero/unused-gradient path. A true
    value records submission, not asynchronous device completion. Events from
    concurrent calls can interleave: adjacency does not identify a single
    attention invocation.
    """

    phase: str
    mode: str
    source: str
    thread_id: int
    timestamp_ns: int
    kernel_submitted: bool | None = None


class BackwardCompileDiagnostics:
    """Thread-safe bounded observations from one process-wide collection window.

    Use :func:`backward_compile_diagnostics` to activate a collector. Overlapping
    windows each receive matching events from all threads in this process;
    they do not isolate calls from one Python task or correlate invocations.
    ``events`` returns an immutable snapshot and ``dropped_events`` reports
    entries discarded after the configured capacity was reached.
    """

    def __init__(self, max_events=1024):
        if type(max_events) is not int or max_events <= 0:
            raise ValueError("max_events must be a positive integer")
        self._events = deque(maxlen=max_events)
        self._dropped = 0
        self._lock = RLock()

    @property
    def events(self):
        with self._lock:
            return tuple(self._events)

    @property
    def dropped_events(self):
        with self._lock:
            return self._dropped

    def _append(self, event):
        with self._lock:
            if len(self._events) == self._events.maxlen:
                self._dropped += 1
            self._events.append(event)


_COLLECTOR_LOCK = RLock()
_COLLECTORS = ()


@contextmanager
def backward_compile_diagnostics(*, max_events=1024):
    """Observe the modes actually captured and used, without changing policy.

    The window is process-wide so NPU Autograd worker events are visible.
    Select ``backward_compile_mode`` before forward; wrapping backward alone
    cannot change that saved choice. No warning is inferred from the worker's
    context. Collection is disabled by default and keeps no Tensor references.
    """
    global _COLLECTORS
    collector = BackwardCompileDiagnostics(max_events)
    with _COLLECTOR_LOCK:
        _COLLECTORS = (*_COLLECTORS, collector)
    try:
        yield collector
    finally:
        with _COLLECTOR_LOCK:
            _COLLECTORS = tuple(item for item in _COLLECTORS if item is not collector)


def record_backward_capture(mode, *, source):
    if _COLLECTORS:
        _record("capture", mode, source)


def record_backward_execution(config, *, kernel_submitted):
    if not _COLLECTORS:
        return
    mode = config.get("backward_compile_mode")
    if mode is None:
        from .runtime import get_backward_compile_mode

        mode = get_backward_compile_mode()
    _record(
        "execute",
        mode,
        config.get("backward_diagnostic_source", "eager"),
        kernel_submitted=kernel_submitted,
    )


def _record(phase, mode, source, *, kernel_submitted=None):
    # Hold the collector lock through appends so exiting a window waits for
    # already-observed events and its final snapshot cannot grow afterwards.
    with _COLLECTOR_LOCK:
        if not _COLLECTORS:
            return
        event = BackwardCompileEvent(
            phase, mode, source, get_ident(), monotonic_ns(), kernel_submitted
        )
        for collector in _COLLECTORS:
            collector._append(event)
