#!/usr/bin/env bash
# Copyright (c) 2026, Minghua Shen.

# CI NPU Benchmark Phase 2. This file must run inside the NPU-selected container.
set -euo pipefail

REPO_ROOT="$(pwd)"
DEVICE="${CI_CONTAINER_DEVICE:-0}"
FA_VERSION="${CI_BENCHMARK_FA_VERSION:-all}"
LAYOUT="${CI_BENCHMARK_LAYOUT:-all}"
MODE="${CI_BENCHMARK_MODE:-all}"
CAUSAL="${CI_BENCHMARK_CAUSAL:-both}"
LOG_DIR="${CI_TEST_LOG_DIR:-/tmp/ci_test_logs}"
BENCHMARK_SCRIPT="${CI_BENCHMARK_SCRIPT:-benchmarks/benchmark_attn_npu.py}"
CSV_FILE="$LOG_DIR/benchmark_results.csv"
JSON_FILE="$LOG_DIR/benchmark_results.json"

log() { printf '[CI-benchmark] %s\n' "$*"; }
die() { printf '[CI-benchmark][ERROR] %s\n' "$*" >&2; exit 1; }

mkdir -p "$LOG_DIR"
case "$FA_VERSION" in all|fa2|fa3|fa4) ;; *) die "unsupported FA version: $FA_VERSION" ;; esac
case "$LAYOUT" in all|bsnd|tnd) ;; *) die "unsupported layout: $LAYOUT" ;; esac
[ -f "$BENCHMARK_SCRIPT" ] || die "benchmark script not found: $BENCHMARK_SCRIPT"

# Explicit single-device guard. Never silently proceed with all cards visible.
[ "$DEVICE" = 0 ] || die "single-card benchmark requires CI_CONTAINER_DEVICE=0 (got $DEVICE)"
[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ] || die "ASCEND_RT_VISIBLE_DEVICES is unset"
[[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || die "expected one physical NPU ID, got: $ASCEND_RT_VISIBLE_DEVICES"
export ASCEND_RT_VISIBLE_DEVICES

log "benchmark phase start: $(date '+%Y-%m-%d %H:%M:%S')"
log "repo=$REPO_ROOT"
log "config: physical_npu=$ASCEND_RT_VISIBLE_DEVICES logical_npu=$DEVICE fa_version=$FA_VERSION layout=$LAYOUT mode=$MODE causal=$CAUSAL"
command -v python3 >/dev/null 2>&1 || die "python3 not found in container"

# The mask must be installed before importing torch_npu, and the runtime must
# recognize exactly one device. Fail early rather than recording spurious 507033.
python3 - "$DEVICE" <<'PY' || die "single-NPU check failed; inspect CANN/torch_npu device visibility"
import os
import sys
import torch
import torch_npu

idx = int(sys.argv[1])
mask = os.environ['ASCEND_RT_VISIBLE_DEVICES']
count = torch_npu.npu.device_count()
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('ASCEND_RT_VISIBLE_DEVICES:', mask)
print('torch_npu device_count:', count)
if count != 1:
    raise RuntimeError(f'expected one visible NPU, got {count}; physical mask={mask}')
if idx != 0:
    raise RuntimeError('single-card logical NPU must be 0')
torch_npu.npu.set_device(idx)
x = torch.ones((8,), device=f'npu:{idx}')
torch_npu.npu.synchronize()
print('selected logical NPU:', idx)
print('device name:', torch_npu.npu.get_device_name(idx))
print('single-NPU smoke test OK:', float(x.sum().cpu()))
PY

ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_TOOLKIT_HOME
log "python setup.py install --skip-build (reuse Phase 1 build artifacts)"
python3 setup.py install --skip-build

log "FlashAttention import check"
# Keep one combined benchmark_results.csv/json output; skip FA2 on 950.
python3 - "$FA_VERSION" "$DEVICE" <<'PY'
import sys
import torch_npu

version = sys.argv[1]
device = int(sys.argv[2])
name = torch_npu.npu.get_device_name(device)
is_950 = 'Ascend950' in name
print('Detected NPU:', name)
if is_950 and version == 'fa2':
    raise SystemExit('FA2 is disabled on Ascend 950 in this CI configuration')
if version in ('all', 'fa2') and not is_950:
    import flash_attn_npu
    print('FA2 import OK:', flash_attn_npu.__version__)
if version in ('all', 'fa3'):
    import flash_attn_npu_3
    print('FA3 import OK:', flash_attn_npu_3.__version__)
if version in ('all', 'fa4'):
    import flash_attn_npu_4
    print('FA4 import OK:', flash_attn_npu_4.__version__)
PY

# Diagnostic preflight for 950 profiling parser. Non-fatal; findings go to CI logs.
log "950 Profiler environment preflight"
python3 - <<'PYDIAG'
import ctypes
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

print('[profiler-diag] python:', sys.version.split()[0], flush=True)
print('[profiler-diag] python sqlite:', sqlite3.sqlite_version, flush=True)
for lib in ('libsqlite3.so.0', 'libsqlite3.so'):
    try:
        obj = ctypes.CDLL(lib)
        print('[profiler-diag] ctypes load OK:', lib, flush=True)
    except OSError as exc:
        print('[profiler-diag] ctypes load FAILED:', lib, str(exc), flush=True)
for cmd in (('ldconfig', '-p'), ('which', 'msprof')):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        output = result.stdout + result.stderr
        if cmd[0] == 'ldconfig':
            output = '\n'.join(x for x in output.splitlines() if 'sqlite' in x.lower())
        print(f'[profiler-diag] {cmd[0]} exit={result.returncode}: {output[:3000]}', flush=True)
    except Exception as exc:
        print(f'[profiler-diag] {cmd[0]} failed: {exc}', flush=True)
for name in ('ASCEND_TOOLKIT_HOME', 'ASCEND_HOME_PATH', 'LD_LIBRARY_PATH'):
    print(f'[profiler-diag] {name}={os.environ.get(name, "<unset>")}', flush=True)
for root in ('/usr/local/Ascend/cann-9.2.0-beta.2', '/usr/local/Ascend/cann-9.1.0'):
    print(f'[profiler-diag] CANN path {root} exists={Path(root).exists()}', flush=True)
PYDIAG
DEVICE_NAME="$(python3 - <<'PYDEVICE'
import torch
import torch_npu

print(torch.npu.get_device_name(0))
PYDEVICE
)"
# 仅用于验证 SQLite 库名问题
if [[ "$DEVICE_NAME" == *Ascend950* ]]; then
  if ! ldconfig -p | grep -q 'libsqlite3\.so '; then
    SQLITE_LIB="$(ldconfig -p | awk '/libsqlite3\.so\.0 / {print $NF; exit}')"

    if [[ -n "$SQLITE_LIB" ]]; then
      mkdir -p /tmp/fa950-profiler-libs
      ln -sf "$SQLITE_LIB" /tmp/fa950-profiler-libs/libsqlite3.so
      export LD_LIBRARY_PATH="/tmp/fa950-profiler-libs:${LD_LIBRARY_PATH:-}"
      log "950 SQLite compatibility test: $SQLITE_LIB"
    fi
  fi
fi
log "running benchmark"

BATCH_DIR="$LOG_DIR/benchmark_batches"
mkdir -p "$BATCH_DIR"

# 保留所有批次的输出路径
CSV_PARTS=()
JSON_PARTS=()

# 记录是否出现真实测试失败
BENCHMARK_RC=0

run_benchmark_part() {
  local name="$1"
  local layout="$2"
  local csv="$BATCH_DIR/${name}.csv"
  local json="$BATCH_DIR/${name}.json"
  local rc=0

  shift 2

  if python3 "$BENCHMARK_SCRIPT" \
    --fa-version "$FA_VERSION" \
    --layout "$layout" \
    --mode "$MODE" \
    --causal "$CAUSAL" \
    --device "$DEVICE" \
    --out "$csv" \
    --json "$json" \
    "$@"; then
    :
  else
    rc=$?
    BENCHMARK_RC=1
    log "benchmark part $name failed: exit=$rc"
  fi

  [ -f "$csv" ] && CSV_PARTS+=("$csv")
  [ -f "$json" ] && JSON_PARTS+=("$json")
}

# BSND 维持一次完整运行
if [[ "$LAYOUT" == "all" || "$LAYOUT" == "bsnd" ]]; then
  log "running BSND benchmark"
  run_benchmark_part "bsnd" "bsnd"
fi

# TND 每 10 个 Case 一个独立 Python 进程
if [[ "$LAYOUT" == "all" || "$LAYOUT" == "tnd" ]]; then
  for start in 0 10 20; do
    stop=$((start + 10))

    log "running TND batch: $start to $stop"

    TND_CASE_START="$start" \
    TND_CASE_STOP="$stop" \
    run_benchmark_part "tnd_${start}_${stop}" "tnd" \
      --num-cases 30
  done
fi
log "merging benchmark results"

python3 - "$CSV_FILE" "$JSON_FILE" \
  "${CSV_PARTS[@]}" -- "${JSON_PARTS[@]}" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

args = sys.argv[1:]
out_csv = Path(args[0])
out_json = Path(args[1])

sep = args.index("--")
csv_files = [Path(p) for p in args[2:sep]]
json_files = [Path(p) for p in args[sep + 1:]]

if not csv_files or not json_files:
    raise RuntimeError("No benchmark results to merge")

if len(csv_files) != len(json_files):
    raise RuntimeError("CSV/JSON batch count mismatch")

# 合并 CSV
fields = []
csv_rows = []

for path in csv_files:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for field in reader.fieldnames or []:
            if field not in fields:
                fields.append(field)
        csv_rows.extend(reader)

with out_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(csv_rows)

# 合并 JSON，保留原始行中的数字类型
documents = [json.loads(p.read_text(encoding="utf-8"))
             for p in json_files]

def locate_rows(document):
    if isinstance(document, list):
        return None, document
    if isinstance(document, dict):
        for key in ("results", "rows", "cases"):
            value = document.get(key)
            if isinstance(value, list):
                return key, value
    raise RuntimeError(
        "Unrecognized benchmark JSON structure; "
        "check benchmark_utils.write_results"
    )

merged_rows = []

for document in documents:
    _, part_rows = locate_rows(document)
    merged_rows.extend(part_rows)

key, _ = locate_rows(documents[0])
counts = dict(Counter(row.get("status", "unknown")
                      for row in merged_rows))

if key is None:
    merged_json = merged_rows
else:
    merged_json = documents[0].copy()
    merged_json[key] = merged_rows

    # 更新常见统计字段
    if "status_counts" in merged_json:
        merged_json["status_counts"] = counts
    if "counts" in merged_json:
        merged_json["counts"] = counts

out_json.write_text(
    json.dumps(merged_json, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(
    f"[CI] merged {len(csv_rows)} CSV rows, "
    f"{len(merged_rows)} JSON rows; "
    f"status={counts}",
    flush=True,
)
PY

log "benchmark completed"

if [ "$BENCHMARK_RC" -ne 0 ]; then
  die "one or more benchmark batches failed"
fi
log "benchmark completed"
if [ -f "$CSV_FILE" ]; then
  log "CSV result: $CSV_FILE"
  echo '========== Benchmark CSV =========='
  cat "$CSV_FILE"
  echo '==================================='
else
  log "warning: CSV result was not generated"
fi
if [ -f "$JSON_FILE" ]; then
  log "JSON result: $JSON_FILE"
else
  log "warning: JSON result was not generated"
fi
log "benchmark phase end: $(date '+%Y-%m-%d %H:%M:%S')"
