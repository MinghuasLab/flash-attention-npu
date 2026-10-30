#!/usr/bin/env python3
"""Measure golden-cache reuse with pytest on a local NPU machine.

Run each revision from the same checkout path and reuse --cache-dir.  The
cache key includes source paths, so separate clone/worktree paths do not model
the CI runner's shared checkout path.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def cache_size(cache_dir: Path) -> tuple[int, int]:
    artifacts = cache_dir / "v2"
    count = size = 0
    if artifacts.is_dir():
        for path in artifacts.rglob("case_*.tar.gz"):
            try:
                size += path.stat().st_size
                count += 1
            except OSError:
                pass
    return count, size


def cache_events(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t", 3)
        if len(fields) >= 2 and fields[1] == "test":
            counts[fields[0]] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--label", required=True, help="Unique run name, such as cold or warm")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    pytest_args = args.pytest_args
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    if not pytest_args:
        parser.error("provide pytest targets after --")
    if not args.label or args.label in {".", ".."} or Path(args.label).name != args.label:
        parser.error("--label must be one directory name")

    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.report_dir.resolve() / args.label
    run_dir.mkdir(parents=True, exist_ok=False)
    stats_file = run_dir / "golden_cache_events.tsv"
    stats_file.touch()
    log_file = run_dir / "pytest.log"
    before_count, before_bytes = cache_size(cache_dir)
    env = os.environ.copy()
    env.update({
        "GOLDEN_CACHE_MODE": "cache",
        "GOLDEN_CACHE_DIR": str(cache_dir),
        "GOLDEN_CACHE_STATS_FILE": str(stats_file),
        "GOLDEN_CACHE_REFRESH": "0",
    })
    command = [sys.executable, "-m", "pytest", *pytest_args]
    print(f"[{args.label}] running {' '.join(command)}; log={log_file}", flush=True)
    started = time.monotonic()
    with log_file.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                check=False)
    elapsed = time.monotonic() - started
    after_count, after_bytes = cache_size(cache_dir)
    events = cache_events(stats_file)
    accesses = events["hit"] + events["miss"]
    report = {
        "label": args.label,
        "command": command,
        "exit_code": result.returncode,
        "duration_seconds": round(elapsed, 2),
        "cache_dir": str(cache_dir),
        "torch_seed_override": env.get("CI_TORCH_SEED"),
        "events": dict(events),
        "hit_rate": round(events["hit"] / accesses, 4) if accesses else None,
        "artifacts_before": before_count,
        "artifacts_after": after_count,
        "bytes_before": before_bytes,
        "bytes_after": after_bytes,
        "bytes_added": after_bytes - before_bytes,
        "pytest_log": str(log_file),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[{args.label}] pytest exit={result.returncode} duration={elapsed:.1f}s")
    print(f"[{args.label}] hit={events['hit']} miss={events['miss']} "
          f"refresh={events['refresh']} read_error={events['read_error']} "
          f"write_error={events['write_error']} hit_rate={report['hit_rate']}")
    print(f"[{args.label}] v2 artifacts={before_count}->{after_count} "
          f"size={before_bytes / 1024**3:.3f}->{after_bytes / 1024**3:.3f} GiB "
          f"added={(after_bytes - before_bytes) / 1024**3:.3f} GiB")
    print(f"[{args.label}] report={run_dir / 'report.json'} log={log_file}")
    if result.returncode:
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"[{args.label}] pytest log (last {min(40, len(lines))} lines):")
        for line in lines[-40:]:
            print(line)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
