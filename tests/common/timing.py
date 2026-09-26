"""Per-test timing and golden-cache diagnostics for CI output."""

from __future__ import annotations

import time
from collections import defaultdict


_CURRENT_STATE = None


def begin():
    return {"started": time.perf_counter(), "parts": defaultdict(float), "events": []}


def set_current_state(state):
    global _CURRENT_STATE
    _CURRENT_STATE = state


def record(part, seconds):
    if _CURRENT_STATE is not None:
        add(_CURRENT_STATE, part, seconds)


def add(state, part, seconds, detail=""):
    state["parts"][part] += seconds
    if detail:
        state["events"].append(detail)


def finish(state, node):
    parts = {key: round(value, 4) for key, value in state["parts"].items()}
    return {
        "node": node,
        "total": round(time.perf_counter() - state["started"], 4),
        "parts": parts,
        "events": state["events"],
    }
