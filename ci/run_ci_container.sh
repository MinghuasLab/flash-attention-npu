#!/usr/bin/env bash
#
# CI 容器入口 (两阶段):
#   阶段1 (编译, 不加锁): docker run 不绑卡, 跑 ci/run_ci_build.sh
#     - git submodule update --init
#     - python setup.py build  (产物在 build/, 通过 volume 持久化)
#   阶段2 (测试, 动态选卡): 探测 task_count<阈值的空闲卡 + docker run 绑卡, 跑 ci/run_ci_test.sh
#     - NPU 自检
#     - python setup.py install (复用 build/ 产物)
#     - pytest Example ST
#
# 环境变量:
#   CI_MODE                  (默认 quick)  quick|full
#   CI_RUN_EXAMPLE_ST        (默认 true)   是否跑 Example ST
#   CI_CONTAINER_DEVICE      (默认 0)      容器内逻辑设备号
#   CI_DOCKER_PRIVILEGED     (默认 true)   是否带 --privileged
#   CI_DOCKER_IMAGE          (默认 fa-npu-ci:910b-cann9.1-torch2.9)
#   CI_SKIP_BUILD            (默认 false)  true=跳过阶段1 (已有 build/ 产物)
#   CI_NPU_WAIT_MAX_SEC      (默认 600)    所有卡忙时最长等待秒数
#   CI_NPU_WAIT_INTERVAL_SEC (默认 30)     所有卡忙时重探间隔秒数
#   ASCEND_RT_VISIBLE_DEVICES               手动指定宿主机物理卡时跳过自动选卡

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CI_MODE="${CI_MODE:-quick}"
CI_RUN_EXAMPLE_ST="${CI_RUN_EXAMPLE_ST:-true}"
CI_CONTAINER_DEVICE="${CI_CONTAINER_DEVICE:-0}"
CI_DOCKER_PRIVILEGED="${CI_DOCKER_PRIVILEGED:-true}"
CI_DOCKER_IMAGE="${CI_DOCKER_IMAGE:-fa-npu-ci:910b-cann9.1-torch2.9}"
CI_SKIP_BUILD="${CI_SKIP_BUILD:-false}"
# 宿主机日志目录: 挂载进容器, 容器内写的日志直接落到宿主机, 避免 --rm 删除容器后日志丢失。
# 同时让 workflow 的 upload-artifact (path: /tmp/ci_test_logs) 能读到日志。
CI_TEST_LOG_DIR_HOST="${CI_TEST_LOG_DIR_HOST:-/tmp/ci_test_logs}"

log() { printf '[CI] %s\n' "$*"; }
die() { printf '[CI][ERROR] %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker not found"
docker image inspect "$CI_DOCKER_IMAGE" >/dev/null 2>&1 || die "docker image $CI_DOCKER_IMAGE not found; load or build it first"

# ---------- docker run 公共参数 ----------
docker_mount_args() {
  local mount_args=()
  for path in \
    /usr/local/dcmi \
    /usr/local/bin/npu-smi \
    /usr/local/Ascend/driver/lib64 \
    /usr/local/Ascend/driver/version.info \
    /etc/ascend_install.info; do
    if [ -e "$path" ]; then
      mount_args+=(-v "$path:$path")
    fi
  done
  printf '%s\n' "${mount_args[@]}"
}

privileged_args=()
if [ "$CI_DOCKER_PRIVILEGED" = "true" ]; then
  privileged_args+=(--privileged)
fi

# ---------- 取消信号处理 (self-hosted runner 兜底) ----------
# GitHub 取消 workflow 时, self-hosted runner 只杀 runner 进程, 不会停 docker 容器,
# 导致 NPU 被孤儿容器持续占用。用 trap 捕获 SIGTERM/SIGINT, kill 当前 docker client
# (PID), docker client 向容器主进程转发 SIGTERM, 容器退出后 --rm 自动清理。
# 配合 workflow 的 cleanup step (if: always()) 双重兜底。
CURRENT_DOCKER_PID=""
cleanup_on_signal() {
  log "received cancel/terminate signal, killing docker container (pid=${CURRENT_DOCKER_PID:-<none>})..."
  if [ -n "$CURRENT_DOCKER_PID" ]; then
    kill -TERM "$CURRENT_DOCKER_PID" 2>/dev/null || true
    wait "$CURRENT_DOCKER_PID" 2>/dev/null || true
    CURRENT_DOCKER_PID=""
  fi
  exit 130
}
trap cleanup_on_signal SIGTERM SIGINT

# ---------- 阶段1: 编译 (不加锁, 不绑卡) ----------
run_build_phase() {
  log "=== Phase 1: build (no NPU lock) ==="
  docker run --rm \
    "${privileged_args[@]}" \
    --network host \
    --ipc host \
    -v "$REPO_ROOT:/workspace/flash-attention-npu" \
    -e FLASH_ATTN_BUILD_VERSION="${FLASH_ATTN_BUILD_VERSION:-all}" \
    -e GIT_CONFIG_GLOBAL=/tmp/gitconfig \
    -w /workspace/flash-attention-npu \
    "$CI_DOCKER_IMAGE" \
    bash -lc 'git config --global --add safe.directory "*" && bash ci/run_ci_build.sh' &
  CURRENT_DOCKER_PID=$!
  wait "$CURRENT_DOCKER_PID"
}

# ---------- 阶段2: 测试 (加锁 + 绑卡) ----------
# NPU 候选 id 列表
get_candidates() {
  bash "$SCRIPT_DIR/detect_npu.sh" --candidates 2>/dev/null \
    | sed -n 's/^  - id=\([0-9]\+\) .*/\1/p'
}

run_docker_test() {
  local device_id="$1"
  local mount_args=()
  while IFS= read -r m; do
    [ -n "$m" ] && mount_args+=("$m")
  done < <(docker_mount_args)

  log "starting test container: image=$CI_DOCKER_IMAGE physical_device=$device_id -> container_device=$CI_CONTAINER_DEVICE mode=$CI_MODE"
  # 创建宿主机日志目录并开放权限 (容器内 root 写, 宿主机 runner 用户读)
  mkdir -p "$CI_TEST_LOG_DIR_HOST"
  chmod 777 "$CI_TEST_LOG_DIR_HOST" 2>/dev/null || true
  docker run --rm \
    "${privileged_args[@]}" \
    --network host \
    --ipc host \
    -v "$REPO_ROOT:/workspace/flash-attention-npu" \
    -v "$CI_TEST_LOG_DIR_HOST:/tmp/ci_test_logs" \
    "${mount_args[@]}" \
    -e ASCEND_RT_VISIBLE_DEVICES="$device_id" \
    -e CI_MODE="$CI_MODE" \
    -e CI_RUN_EXAMPLE_ST="$CI_RUN_EXAMPLE_ST" \
    -e CI_TEST_WORKERS="${CI_TEST_WORKERS:-}" \
    -e CI_TEST_DIRECT_FILE="${CI_TEST_DIRECT_FILE:-}" \
    -e CI_TEST_DIRECT_FILTER="${CI_TEST_DIRECT_FILTER:-}" \
    -e CI_RANDOM_SEED="${CI_RANDOM_SEED:-0}" \
    -e CI_QUICK_SAMPLE="${CI_QUICK_SAMPLE:-30}" \
    -e CI_TEST_LOG_DIR="/tmp/ci_test_logs" \
    -e CI_CONTAINER_DEVICE="$CI_CONTAINER_DEVICE" \
    -e FLASH_ATTN_BUILD_VERSION="${FLASH_ATTN_BUILD_VERSION:-all}" \
    -e GIT_CONFIG_GLOBAL=/tmp/gitconfig \
    -w /workspace/flash-attention-npu \
    "$CI_DOCKER_IMAGE" \
    bash -lc 'git config --global --add safe.directory "*" && bash ci/run_ci_test.sh' &
  CURRENT_DOCKER_PID=$!
  wait "$CURRENT_DOCKER_PID"
}

acquire_lock_and_run_test() {
  # 动态选卡模式 (不持 flock 长期锁, 允许多 runner 共享卡, 靠 task_count 阈值软限流)。
  # 返回码: 0=测试成功; 1=无可用卡 (全部超阈值, 不计重试, 等待后重探); 2=测试失败 (直接报错)
  local selected line
  # 用 --env 模式取卡号 (输出 "export NPU_SELECTED_DEVICE=<id>"), 避免解析给人看的 --candidates 文本
  line="$(bash "$SCRIPT_DIR/detect_npu.sh" --env 2>/dev/null || true)"
  selected="${line#export NPU_SELECTED_DEVICE=}"
  if [ -z "$selected" ]; then
    return 1   # 无可用卡 (所有卡 task_count >= MAX_TASKS 或 free 不足)
  fi
  log "selected NPU device=$selected (re-probed, no lock held)"
  if run_docker_test "$selected"; then
    return 0
  fi
  return 2
}

# ---------- 主流程 ----------
main() {
  local total_start
  total_start="$(date +%s)"
  log "CI start: $(date '+%Y-%m-%d %H:%M:%S')"

  # 阶段1: 编译
  if [ "$CI_SKIP_BUILD" = "true" ]; then
    log "CI_SKIP_BUILD=true, skip phase 1 (assume build/ exists)"
  else
    run_build_phase
    log "=== Phase 1 done ==="
  fi

  # 阶段2: 动态选卡 + 测试 (无卡时等待重试, 测试失败直接报错)
  log "=== Phase 2: test (dynamic device selection) ==="

  # 首次探测, 确认机器上有 NPU
  cands="$(get_candidates)"
  if [ -z "$cands" ]; then
    die "no candidate NPU detected; run 'bash ci/detect_npu.sh --summary' to check"
  fi

  local wait_max="${CI_NPU_WAIT_MAX_SEC:-600}"
  local wait_interval="${CI_NPU_WAIT_INTERVAL_SEC:-30}"
  local test_passed=false rc

  while [ "$test_passed" != "true" ]; do
    set +e
    acquire_lock_and_run_test
    rc=$?
    set -e
    case $rc in
      0)
        test_passed=true
        ;;
      1)
        # 无可用卡 (全部超阈值), 等待后重探
        if [ "$wait_max" -le 0 ]; then
          die "no available NPU (all cards have >= ${CI_NPU_MAX_TASKS:-4} tasks or insufficient free) after waiting; aborting CI"
        fi
        log "all NPU busy (tasks >= ${CI_NPU_MAX_TASKS:-4} or free <= ${CI_NPU_MIN_FREE_MB:-1024}MB), waiting ${wait_interval}s (remaining wait budget: ${wait_max}s)..."
        sleep "$wait_interval"
        wait_max=$((wait_max - wait_interval))
        ;;
      2)
        # 测试失败, 不重试, 直接报错
        die "test failed; aborting CI"
        ;;
    esac
  done

  total_end="$(date +%s)"
  log "CI end: $(date '+%Y-%m-%d %H:%M:%S') (total=$((total_end - total_start))s)"
}

main "$@"
