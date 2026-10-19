#!/usr/bin/env bash
#
# NPU Benchmark 阶段
#
# 由 ci/run_ci_container.sh 的 Phase 2 调用
# 此时：
#   - Phase 1 已完成 build
#   - 已动态选择一张 NPU
#   - ASCEND_RT_VISIBLE_DEVICES 已映射宿主机物理 NPU
#   - 容器内逻辑设备默认为 CI_CONTAINER_DEVICE=0
#
# 流程：
#   1. NPU 自检
#   2. 安装 Phase 1 build 产物
#   3. FlashAttention import 校验
#   4. 执行 benchmark
#   5. 输出 CSV / JSON
#
# 环境变量：
#   CI_BENCHMARK_FA_VERSION    all|fa2|fa3|fa4
#   CI_BENCHMARK_LAYOUT        all|bsnd|tnd
#   CI_CONTAINER_DEVICE        容器内逻辑 NPU id，默认 0
#   CI_TEST_LOG_DIR            输出目录，默认 /tmp/ci_test_logs


set -euo pipefail


REPO_ROOT="$(pwd)"

DEVICE="${CI_CONTAINER_DEVICE:-0}"

FA_VERSION="${CI_BENCHMARK_FA_VERSION:-all}"
LAYOUT="${CI_BENCHMARK_LAYOUT:-all}"

LOG_DIR="${CI_TEST_LOG_DIR:-/tmp/ci_test_logs}"

BENCHMARK_SCRIPT="${CI_BENCHMARK_SCRIPT:-benchmarks/benchmark_attn_npu.py}"

CSV_FILE="$LOG_DIR/benchmark_results.csv"
JSON_FILE="$LOG_DIR/benchmark_results.json"


log() {
  printf '[CI-benchmark] %s\n' "$*"
}


die() {
  printf '[CI-benchmark][ERROR] %s\n' "$*" >&2
  exit 1
}


# ============================================================================
# 0. 基础检查
# ============================================================================

mkdir -p "$LOG_DIR"


case "$FA_VERSION" in
  all|fa2|fa3|fa4)
    ;;
  *)
    die "unsupported FA version: $FA_VERSION"
    ;;
esac


case "$LAYOUT" in
  all|bsnd|tnd)
    ;;
  *)
    die "unsupported layout: $LAYOUT"
    ;;
esac


if [ ! -f "$BENCHMARK_SCRIPT" ]; then
  die "benchmark script not found: $BENCHMARK_SCRIPT"
fi


log "benchmark phase start: $(date '+%Y-%m-%d %H:%M:%S')"

log "repo=$REPO_ROOT"

log \
  "config: device=$DEVICE fa_version=$FA_VERSION layout=$LAYOUT"

log \
  "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"


# ============================================================================
# 1. NPU 自检
# ============================================================================

command -v python3 >/dev/null 2>&1 \
  || die "python3 not found in container"


python3 - "$DEVICE" <<'PY' \
  || exit_code=$?

import sys

import torch
import torch_npu


device = int(sys.argv[1])

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("torch_npu device_count:", torch_npu.npu.device_count())

assert torch_npu.npu.device_count() >= 1, (
    "device_count == 0; "
    "check --privileged / driver mount / ASCEND_RT_VISIBLE_DEVICES"
)

torch_npu.npu.set_device(device)

print("selected logical NPU:", device)
print("device name:", torch_npu.npu.get_device_name(device))

PY


if [ "${exit_code:-0}" -ne 0 ]; then
  die "torch_npu not functional inside benchmark container"
fi


# ============================================================================
# 2. 安装 Phase 1 build 产物
# ============================================================================

ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_TOOLKIT_HOME


log "python setup.py install --skip-build (reuse Phase 1 build artifacts)"

python3 setup.py install --skip-build


# ============================================================================
# 3. Import 校验
# ============================================================================

log "FlashAttention import check"


python3 - "$FA_VERSION" <<'PY'
import sys


version = sys.argv[1]


if version in ("all", "fa2"):
    import flash_attn_npu
    print("FA2 import OK:", flash_attn_npu.__version__)


if version in ("all", "fa3"):
    import flash_attn_npu_3
    print("FA3 import OK:", flash_attn_npu_3.__version__)


if version in ("all", "fa4"):
    import flash_attn_npu_4
    print("FA4 import OK:", flash_attn_npu_4.__version__)

PY


# ============================================================================
# 4. 执行 Benchmark
# ============================================================================

log "running benchmark"

log \
  "python3 $BENCHMARK_SCRIPT \
--fa-version $FA_VERSION \
--layout $LAYOUT \
--device $DEVICE"


python3 "$BENCHMARK_SCRIPT" \
  --fa-version "$FA_VERSION" \
  --layout "$LAYOUT" \
  --device "$DEVICE" \
  --out "$CSV_FILE" \
  --json "$JSON_FILE"


# ============================================================================
# 5. 输出结果
# ============================================================================

echo ""

log "benchmark completed"


if [ -f "$CSV_FILE" ]; then

  log "CSV result: $CSV_FILE"

  echo ""
  echo "========== Benchmark CSV =========="
  cat "$CSV_FILE"
  echo "==================================="

else

  log "warning: CSV result was not generated"

fi


if [ -f "$JSON_FILE" ]; then

  log "JSON result: $JSON_FILE"

else

  log "warning: JSON result was not generated"

fi


log "benchmark phase end: $(date '+%Y-%m-%d %H:%M:%S')"
