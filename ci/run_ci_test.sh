#!/usr/bin/env bash
#
# 阶段2: NPU 自检 + 安装 + 测试 (容器内执行, 需要 NPU, 已加锁)
#   1. NPU 可用性自检
#   2. python setup.py install (复用阶段1 build/ 产物, 快速安装)
#   3. import 校验
#   4. pytest tests/ (quick 每测试函数随机采样至多 N 个, full 全量)
#
# 由 ci/run_ci_container.sh 阶段2通过 docker run 调用 (绑卡 + 加锁)。
#
# 环境变量 (由 run_ci_container.sh 注入):
#   ASCEND_RT_VISIBLE_DEVICES   宿主机物理卡号
#   CI_MODE                     quick|full
#   CI_RUN_EXAMPLE_ST           true|false
#   CI_TEST_WORKERS             xdist -n (默认 2)
#   CI_QUICK_SAMPLE             quick 每函数采样数 (默认 30)
#   CI_RANDOM_SEED              采样 seed (默认 0, 可复现)
#   CI_TEST_DIRECT_FILE         指定时只跑该文件 (绕过 tests/)
#   CI_TEST_DIRECT_FILTER       直接模式的 -k 过滤
#   CI_CONTAINER_DEVICE         容器内逻辑设备号 (默认 0)
#   GOLDEN_CACHE_MODE/DIR       golden reference cache mode and container path
#   GOLDEN_CACHE_STATS_FILE     xdist-safe tab-separated cache event output

set -euo pipefail

REPO_ROOT="$(pwd)"
DEVICE="${CI_CONTAINER_DEVICE:-0}"

# git safe.directory (容器内 root 操作宿主机 runner 用户的目录, 会触发 dubious ownership)
git config --global --add safe.directory "$REPO_ROOT"

log() { printf '[CI-test] %s\n' "$*"; }
die() { printf '[CI-test][ERROR] %s\n' "$*" >&2; exit 1; }

LOG_DIR="${CI_TEST_LOG_DIR:-/tmp/ci_test_logs}"
mkdir -p "$LOG_DIR"
export GOLDEN_CACHE_STATS_FILE="${GOLDEN_CACHE_STATS_FILE:-$LOG_DIR/golden_cache_events.tsv}"
CACHE_STATS_FILE="$GOLDEN_CACHE_STATS_FILE"

cache_artifact_count() {
  if [ -n "${GOLDEN_CACHE_DIR:-}" ] && [ -d "$GOLDEN_CACHE_DIR" ]; then
    { find "$GOLDEN_CACHE_DIR" -type f -name 'case_*.tar.gz' 2>/dev/null || true; } | wc -l | tr -d ' '
  else
    printf '0\n'
  fi
}

CACHE_ARTIFACTS_BEFORE=0
if [ "${GOLDEN_CACHE_MODE:-off}" != "off" ] && [ "${GOLDEN_CACHE_MODE:-off}" != "0" ]; then
  : > "$CACHE_STATS_FILE"
  CACHE_ARTIFACTS_BEFORE="$(cache_artifact_count)"
  log "golden cache: mode=${GOLDEN_CACHE_MODE} dir=${GOLDEN_CACHE_DIR:-<unset>} artifacts_before=$CACHE_ARTIFACTS_BEFORE stats=$CACHE_STATS_FILE"
else
  : > "$CACHE_STATS_FILE"
  log "golden cache: disabled"
fi

log "repo=$REPO_ROOT device=$DEVICE mode=${CI_MODE:-quick}"
log "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
log "test phase start: $(date '+%Y-%m-%d %H:%M:%S')"

# ---------- 1. NPU 自检 (需要卡) ----------
command -v python3 >/dev/null 2>&1 || die "python3 not found in container"
python3 - <<'PY' || die "torch_npu not functional inside container (check --privileged / driver mount)"
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("torch_npu device_count:", torch_npu.npu.device_count())
assert torch_npu.npu.device_count() >= 1, "device_count==0; --privileged or driver mount missing?"
PY

# ---------- 2. 安装 (复用 build/ 产物, 不重新编译) ----------
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
log "python setup.py install --skip-build (reuse build/ artifacts)"
python3 setup.py install --skip-build

log "import check"
python3 - <<'PY'
import flash_attn_npu_3
print("flash_attn_npu_3", flash_attn_npu_3.__version__)
PY

# ---------- 3. pytest tests/ ----------
if [ "${CI_RUN_EXAMPLE_ST:-true}" != "true" ]; then
  log "CI_RUN_EXAMPLE_ST!=true, skip tests"
  exit 0
fi

command -v pytest >/dev/null 2>&1 || pip install pytest --quiet
python3 -c "import xdist" 2>/dev/null || pip install pytest-xdist --quiet

MODE="${CI_MODE:-quick}"
TEST_WORKERS="${CI_TEST_WORKERS:-2}"
# quick 模式: 每个测试函数随机采样至多 CI_QUICK_SAMPLE 个 item (固定 seed 可复现,
# 采样逻辑在 tests/conftest.py; CI_RANDOM_SEED 改 seed 可换一组子集)
export CI_RANDOM_SEED="${CI_RANDOM_SEED:-0}"
CI_QUICK_SAMPLE="${CI_QUICK_SAMPLE:-30}"

# quick 采样, full 全量
SAMPLE_ARG=""
if [ "$MODE" = "quick" ]; then
  SAMPLE_ARG="--random-sample=${CI_QUICK_SAMPLE}"
fi

FAILED_FILE="$LOG_DIR/failed_cases.txt"
: > "$FAILED_FILE"

run_pytest() {
  local target="$1" logfile="$2"; shift 2
  log ">>> pytest $target mode=$MODE workers=$TEST_WORKERS sample=${SAMPLE_ARG:-<none>} (log=$logfile)"
  set +e
  # shellcheck disable=SC2086
  python3 -m pytest "$target" -vs -n "$TEST_WORKERS" --dist=loadscope $SAMPLE_ARG "$@" >"$logfile" 2>&1
  local rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    log "<<< FAILED (pytest rc=$rc), tail of $logfile:"
    tail -n 30 "$logfile" 2>/dev/null | sed 's/^/    /'
    echo "$target" >> "$FAILED_FILE"
  else
    log "<<< OK ($target)"
  fi
}

summarize_golden_cache() {
  local artifacts_after
  artifacts_after="$(cache_artifact_count)"
  if [ ! -s "$CACHE_STATS_FILE" ]; then
    log "[golden-cache-summary] no events recorded artifacts_before=$CACHE_ARTIFACTS_BEFORE artifacts_after=$artifacts_after"
    return
  fi
  awk -F '\t' '
    $2 == "test" { count[$1]++ }
    END {
      printf "[CI-test] [golden-cache-summary] hit=%d miss=%d refresh=%d read_error=%d write_ok=%d write_error=%d disabled=%d\n",
        count["hit"], count["miss"], count["refresh"], count["read_error"],
        count["write_ok"], count["write_error"], count["disabled"]
    }
  ' "$CACHE_STATS_FILE"
  log "[golden-cache-summary] artifacts_before=$CACHE_ARTIFACTS_BEFORE artifacts_after=$artifacts_after"
}

log "running pytest (mode=$MODE workers=$TEST_WORKERS sample=${SAMPLE_ARG:-<none>})"

# 直接模式: CI_TEST_DIRECT_FILE 指定时, 只跑指定文件 (绕过 tests/)
if [ -n "${CI_TEST_DIRECT_FILE:-}" ]; then
  run_pytest "$CI_TEST_DIRECT_FILE" "$LOG_DIR/direct.log" ${CI_TEST_DIRECT_FILTER:+-k "$CI_TEST_DIRECT_FILTER"}
else
  run_pytest "tests/" "$LOG_DIR/all_tests.log"
fi

summarize_golden_cache

FAILED_CASES="$(tr '\n' ' ' < "$FAILED_FILE" 2>/dev/null || true)"
if [ -n "$FAILED_CASES" ]; then
  die "pytest FAILED targets:$FAILED_CASES"
fi

log "all tests passed"
log "test phase end: $(date '+%Y-%m-%d %H:%M:%S')"
