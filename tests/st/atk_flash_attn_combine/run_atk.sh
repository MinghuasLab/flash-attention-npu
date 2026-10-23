#!/usr/bin/env bash
# ATK closed loop for flash_attn_combine (FA3 + FA4).
#
#   1. generate cases        : atk case
#   2. NPU vs CPU accuracy   : atk pytorch (same two-benchmark topology as
#                              `atk node --backend npu node --backend cpu`)
#
# Usage:
#   bash run_atk.sh [start] [end] [device]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

START="${1:-0}"
END="${2:-100}"
DEVICE="${3:-0}"

atk case -f test_flash_attn_combine.yaml -p flash_attn_combine_constraint.py
atk case -f test_flash_attn_combine_v4.yaml -p flash_attn_combine_constraint.py

for tag in test_flash_attn_combine test_flash_attn_combine_v4; do
  CASE_JSON="$HERE/result/$tag/json/all_$tag.json"
  test -f "$CASE_JSON" || { echo "missing case json: $CASE_JSON"; exit 1; }

  echo "== $tag (npu vs cpu) =="
  PYTHONPATH="$HERE:$REPO_ROOT" atk pytorch \
    "$CASE_JSON" \
    -p "$HERE/execute_flash_attn_combine.py" \
    --task accuracy \
    --devices "$DEVICE" \
    -s "$START" -e "$END"
done

# Explicit two-node topology (equivalent to the pyaclnn + cpu ST layout):
#   atk node --backend npu --devices 0 -o "$HERE" \
#     node --backend cpu \
#     task -c result/test_flash_attn_combine/json/all_test_flash_attn_combine.json \
#          --task accuracy -p execute_flash_attn_combine.py -s 0 -e 2
