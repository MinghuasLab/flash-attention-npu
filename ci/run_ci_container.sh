#!/usr/bin/env bash
#
# CI 容器入口 (两阶段):
#
#   阶段1 (编译, 不加锁):
#     docker run 不绑卡, 跑 ci/run_ci_build.sh
#     - bash ci/init_submodules.sh (浅拉 + runner 磁盘缓存)
#     - python setup.py build
#     - 产物保存在 build/, 通过 volume 持久化
#
#   阶段2 (NPU task, 动态选卡):
#     探测 task_count < 阈值的空闲卡 + docker run 绑卡
#
#     CI_TASK=test:
#       跑 ci/run_ci_test.sh
#       - NPU 自检
#       - python setup.py install --skip-build
#       - pytest correctness tests
#
#     CI_TASK=benchmark:
#       跑 ci/run_ci_benchmark.sh
#       - NPU 自检
#       - python setup.py install --skip-build
#       - FlashAttention benchmark
#
# 环境变量:
#   CI_TASK                  (默认 test)    test|benchmark
#   CI_MODE                  (默认 quick)   quick|full
#   CI_BENCHMARK_FA_VERSION  (默认 all)     all|fa2|fa3|fa4
#   CI_BENCHMARK_LAYOUT      (默认 all)     all|bsnd|tnd
#   CI_RUN_EXAMPLE_ST        (默认 true)    是否跑 Example ST
#   CI_CONTAINER_DEVICE      (默认 0)       容器内逻辑设备号
#   CI_DOCKER_PRIVILEGED     (默认 true)    是否带 --privileged
#   CI_DOCKER_IMAGE          (默认 fa-npu-ci:910b-cann9.1-torch2.9)
#   CI_SKIP_BUILD            (默认 false)   true=跳过阶段1 (已有 build/ 产物)
#   CI_CONTAINER_SCOPE       当前 CI job 的唯一容器归属标识
#   FLASH_ATTN_BUILD_NPU     (默认 auto) 按 torch_npu 设备名选择 910/950; all=双架构
#   CI_NPU_WAIT_MAX_SEC      (默认 600)     所有卡忙时最长等待秒数
#   CI_NPU_WAIT_INTERVAL_SEC (默认 30)      所有卡忙时重探间隔秒数
#   ASCEND_RT_VISIBLE_DEVICES               宿主机物理卡
#
#   GOLDEN_CACHE_HOST_DIR
#       CI 固定为 /home/FA_NPU_CI_DATA
#
#   SUBMODULE_CACHE_DIR
#       默认 $GOLDEN_CACHE_HOST_DIR/submodule-cache
#       runner 用户写不了数据盘时, 仍挂载已存在的父目录,
#       由 root 编译容器创建并回填缓存; 宿主机只读恢复。
#
#   GOLDEN_CACHE_DIR
#       默认 /var/cache/flash-attention-npu/golden_cache
#
#   GOLDEN_CACHE_MODE
#       默认 cache, cache|off
#
#   GOLDEN_CACHE_STATS_FILE
#       容器内固定为 /tmp/ci_test_logs/golden_cache_events.tsv
#
#   GOLDEN_CACHE_MAX_BYTES
#       默认 40 GiB, 缓存总容量上限; 0 为不限
#   GOLDEN_CACHE_MAX_AGE_DAYS
#       默认 30, 最长未命中天数; 0 为不限
#   GOLDEN_CACHE_EVICT_INTERVAL_SECONDS
#       默认 60, 清理扫描最短间隔
#
#   PROXY_CONFIG_FILE
#       默认 $GOLDEN_CACHE_HOST_DIR/proxy.conf
#
#   CI_GIT_PROXY / CI_GIT_PROXY_WHITELIST
#       可选, 覆盖 git 代理和白名单

set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"


# ============================================================================
# 基础配置
# ============================================================================

CI_MODE="${CI_MODE:-quick}"

# test / benchmark
CI_TASK="${CI_TASK:-test}"

# benchmark 参数
CI_BENCHMARK_FA_VERSION="${CI_BENCHMARK_FA_VERSION:-all}"
CI_BENCHMARK_LAYOUT="${CI_BENCHMARK_LAYOUT:-all}"
CI_BENCHMARK_MODE="${CI_BENCHMARK_MODE:-all}"
CI_BENCHMARK_CAUSAL="${CI_BENCHMARK_CAUSAL:-both}"
CI_BENCHMARK_COMMIT_SHA="${CI_BENCHMARK_COMMIT_SHA:-unknown}"

CI_RUN_EXAMPLE_ST="${CI_RUN_EXAMPLE_ST:-true}"
CI_CONTAINER_DEVICE="${CI_CONTAINER_DEVICE:-0}"
# ASCEND_RT_VISIBLE_DEVICES masks host physical devices; with one visible card,
# the application must use logical NPU 0.
if [ "$CI_CONTAINER_DEVICE" != "0" ]; then
  printf '[CI][ERROR] single-card mode requires CI_CONTAINER_DEVICE=0 (got %s)\n' "$CI_CONTAINER_DEVICE" >&2
  exit 1
fi
CI_DOCKER_PRIVILEGED="${CI_DOCKER_PRIVILEGED:-true}"
CI_DOCKER_IMAGE="${CI_DOCKER_IMAGE:-fa-npu-ci:910b-cann9.1-torch2.9}"
CI_SKIP_BUILD="${CI_SKIP_BUILD:-false}"

CI_CONTAINER_SCOPE="${CI_CONTAINER_SCOPE:-local-$(id -u)-$$}"
CI_LOG_SCOPE="${CI_CONTAINER_SCOPE//[^A-Za-z0-9_.-]/_}"
CI_TEST_LOG_DIR_HOST="${CI_TEST_LOG_DIR_HOST:-/tmp/ci_test_logs/$CI_LOG_SCOPE}"

GOLDEN_CACHE_HOST_DIR="${GOLDEN_CACHE_HOST_DIR:-/home/FA_NPU_CI_DATA}"
GOLDEN_CACHE_DIR="${GOLDEN_CACHE_DIR:-/var/cache/flash-attention-npu/golden_cache}"
GOLDEN_CACHE_MODE="${GOLDEN_CACHE_MODE:-cache}"


# shellcheck source=ci/docker_proxy.sh
# shellcheck disable=SC1091
source "$SCRIPT_DIR/docker_proxy.sh"
git_proxy_init "$GOLDEN_CACHE_HOST_DIR"


log() {
  printf '[CI] %s\n' "$*"
}


die() {
  printf '[CI][ERROR] %s\n' "$*" >&2
  exit 1
}


# ============================================================================
# 参数合法性检查
# ============================================================================

case "$CI_TASK" in
  test|benchmark)
    ;;
  *)
    die "unsupported CI_TASK=$CI_TASK (expected: test|benchmark)"
    ;;
esac


if [ "$CI_TASK" = "benchmark" ]; then

  case "$CI_BENCHMARK_FA_VERSION" in
    all|fa2|fa3|fa4)
      ;;
    *)
      die \
        "unsupported CI_BENCHMARK_FA_VERSION=$CI_BENCHMARK_FA_VERSION \
(expected: all|fa2|fa3|fa4)"
      ;;
  esac

  case "$CI_BENCHMARK_LAYOUT" in
    all|bsnd|tnd)
      ;;
    *)
      die \
        "unsupported CI_BENCHMARK_LAYOUT=$CI_BENCHMARK_LAYOUT \
(expected: all|bsnd|tnd)"
      ;;
  esac

fi


# ============================================================================
# Docker 基础检查
# ============================================================================

command -v docker >/dev/null 2>&1 \
  || die "docker not found"

docker image inspect "$CI_DOCKER_IMAGE" >/dev/null 2>&1 \
  || die "docker image $CI_DOCKER_IMAGE not found; load or build it first"


# ============================================================================
# docker run 公共参数
# ============================================================================

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


# ============================================================================
# 取消信号处理
# ============================================================================

# GitHub 取消 workflow 时, self-hosted runner 可能只停止 runner 进程,
# 不会自动停止 docker container。
#
# 因此这里按照当前 job 的 CI_CONTAINER_SCOPE 清理所有本 job 容器，
# 并配合 workflow cleanup step 做双重兜底。

CURRENT_DOCKER_PID=""


cleanup_on_signal() {

  log \
    "received cancel/terminate signal, cleaning containers for scope=$CI_CONTAINER_SCOPE..."

  if [ -n "$CURRENT_DOCKER_PID" ]; then

    kill -KILL "$CURRENT_DOCKER_PID" 2>/dev/null || true
    wait "$CURRENT_DOCKER_PID" 2>/dev/null || true

    CURRENT_DOCKER_PID=""

  fi

  CI_CONTAINER_SCOPE="$CI_CONTAINER_SCOPE" \
    bash "$SCRIPT_DIR/cleanup_ci_containers.sh" || true

  exit 130
}


trap cleanup_on_signal SIGTERM SIGINT


# ============================================================================
# Phase 1: Build
# ============================================================================

run_build_phase() {

  local rc
  local cache_dir="${SUBMODULE_CACHE_DIR:-${GOLDEN_CACHE_HOST_DIR}/submodule-cache}"
  local cache_parent
  local cache_mount_args=()
  local mount_args=()
  while IFS= read -r m; do
    [ -n "$m" ] && mount_args+=("$m")
  done < <(docker_mount_args)

  cache_parent="$(dirname -- "$cache_dir")"

  log "=== Phase 1: build (no NPU lock) ==="


  # 宿主机优先只读恢复缓存。
  # runner 用户无法写数据盘时，只负责浅拉工作区。
  #
  # 编译容器以 root 挂载数据盘并回填缓存。
  # 容器内再次执行 init_submodules 作为兜底。

  bash "$SCRIPT_DIR/init_submodules.sh" "$REPO_ROOT"


  if [ -d "$cache_dir" ]; then

    cache_mount_args=(
      -v "$cache_dir:$cache_dir"
      -e "SUBMODULE_CACHE_DIR=$cache_dir"
    )

    log "submodule cache mount: $cache_dir"

  elif [ -d "$cache_parent" ]; then

    cache_mount_args=(
      -v "$cache_parent:$cache_parent"
      -e "SUBMODULE_CACHE_DIR=$cache_dir"
    )

    log \
      "submodule cache mount: $cache_parent \
(container will create $(basename -- "$cache_dir"))"

  else

    log \
      "warning: submodule cache parent missing ($cache_parent); cache disabled"

  fi


  docker run --rm \
    --label "com.flash-attention-npu.ci.scope=$CI_CONTAINER_SCOPE" \
    "${privileged_args[@]}" \
    "${GIT_PROXY_DOCKER_ARGS[@]}" \
    --network host \
    --ipc host \
    -v "$REPO_ROOT:/workspace/flash-attention-npu" \
    "${cache_mount_args[@]}" \
    "${mount_args[@]}" \
    -e FLASH_ATTN_BUILD_VERSION="${FLASH_ATTN_BUILD_VERSION:-all}" \
    -e FLASH_ATTN_BUILD_NPU="${FLASH_ATTN_BUILD_NPU:-auto}" \
    -e FLASH_ATTN_SKIP_SUBMODULE_INIT=1 \
    -e GIT_CONFIG_GLOBAL=/tmp/gitconfig \
    -e CI_BENCHMARK_COMMIT_SHA="$CI_BENCHMARK_COMMIT_SHA" \
    -w /workspace/flash-attention-npu \
    "$CI_DOCKER_IMAGE" \
    bash -lc \
      'git config --global --add safe.directory /workspace/flash-attention-npu && bash ci/run_ci_build.sh' &


  CURRENT_DOCKER_PID=$!


  if wait "$CURRENT_DOCKER_PID"; then
    rc=0
  else
    rc=$?
  fi


  CURRENT_DOCKER_PID=""

  return "$rc"
}


# ============================================================================
# Phase 2: NPU task
# ============================================================================

# NPU 候选 id 列表
get_candidates() {

  bash "$SCRIPT_DIR/detect_npu.sh" --candidates 2>/dev/null \
    | sed -n 's/^  - id=\([0-9]\+\) .*/\1/p' \
    || true
}


# ----------------------------------------------------------------------------
# 在选定 NPU 上启动 task container
#
# CI_TASK=test:
#     ci/run_ci_test.sh
#
# CI_TASK=benchmark:
#     ci/run_ci_benchmark.sh
# ----------------------------------------------------------------------------

run_docker_task() {

  local device_id="$1"
  # Fail closed if the NPU detector does not return one numeric physical ID.
  if ! [[ "$device_id" =~ ^[0-9]+$ ]]; then
    die "invalid single physical NPU id: $device_id"
  fi
  local rc
  local entry_script
  local mount_args=()


  case "$CI_TASK" in

    test)
      entry_script="ci/run_ci_test.sh"
      ;;

    benchmark)
      entry_script="ci/run_ci_benchmark.sh"
      ;;

  esac


  # 提前检查脚本是否存在
  if [ ! -f "$REPO_ROOT/$entry_script" ]; then
    die "CI entry script not found: $entry_script"
  fi


  while IFS= read -r m; do

    [ -n "$m" ] && mount_args+=("$m")

  done < <(docker_mount_args)


  log \
    "starting $CI_TASK container: \
image=$CI_DOCKER_IMAGE \
physical_device=$device_id \
-> container_device=$CI_CONTAINER_DEVICE"


  # 创建宿主机日志目录并开放权限
  # 容器内 root 写，宿主机 runner 用户读

  mkdir -p "$CI_TEST_LOG_DIR_HOST"
  chmod 777 "$CI_TEST_LOG_DIR_HOST" 2>/dev/null || true


  local golden_mount_args=()


  # Golden cache 主要用于 correctness。
  #
  # 为保持现有 CI 行为兼容，这里仍保留原有挂载逻辑。
  # benchmark 脚本本身可以不使用 golden cache。

  if [ "$GOLDEN_CACHE_MODE" != "off" ] &&
     [ "$GOLDEN_CACHE_MODE" != "0" ]; then

    if mkdir -p "$GOLDEN_CACHE_HOST_DIR" 2>/dev/null &&
       [ -d "$GOLDEN_CACHE_HOST_DIR" ]; then

      golden_mount_args=(
        -v "$GOLDEN_CACHE_HOST_DIR:$GOLDEN_CACHE_DIR:rw"
      )

      log \
        "golden cache enabled: \
host=$GOLDEN_CACHE_HOST_DIR \
container=$GOLDEN_CACHE_DIR"

    else

      log \
        "warning: golden cache directory is unavailable \
($GOLDEN_CACHE_HOST_DIR); disabling cache"

      GOLDEN_CACHE_MODE=off

    fi

  fi
  docker run --rm \
    --label "com.flash-attention-npu.ci.scope=$CI_CONTAINER_SCOPE" \
    "${privileged_args[@]}" \
    "${GIT_PROXY_DOCKER_ARGS[@]}" \
    --network host \
    --ipc host \
    -v "$REPO_ROOT:/workspace/flash-attention-npu" \
    -v "$CI_TEST_LOG_DIR_HOST:/tmp/ci_test_logs" \
    "${golden_mount_args[@]}" \
    "${mount_args[@]}" \
    \
    -e ASCEND_RT_VISIBLE_DEVICES="$device_id" \
    \
    -e CI_TASK="$CI_TASK" \
    -e CI_ENTRY_SCRIPT="$entry_script" \
    \
    -e CI_BENCHMARK_FA_VERSION="$CI_BENCHMARK_FA_VERSION" \
    -e CI_BENCHMARK_LAYOUT="$CI_BENCHMARK_LAYOUT" \
    -e CI_BENCHMARK_MODE="$CI_BENCHMARK_MODE" \
    -e CI_BENCHMARK_CAUSAL="$CI_BENCHMARK_CAUSAL" \
    \
    -e CI_MODE="$CI_MODE" \
    -e CI_RUN_EXAMPLE_ST="$CI_RUN_EXAMPLE_ST" \
    \
    -e CI_TEST_WORKERS="${CI_TEST_WORKERS:-}" \
    -e CI_TEST_DIRECT_FILE="${CI_TEST_DIRECT_FILE:-}" \
    -e CI_TEST_DIRECT_FILTER="${CI_TEST_DIRECT_FILTER:-}" \
    -e CI_RANDOM_SEED="${CI_RANDOM_SEED:-0}" \
    -e CI_QUICK_SAMPLE="${CI_QUICK_SAMPLE:-30}" \
    \
    -e CI_TEST_LOG_DIR="/tmp/ci_test_logs" \
    -e CI_CONTAINER_DEVICE="$CI_CONTAINER_DEVICE" \
    \
    -e GOLDEN_CACHE_MODE="$GOLDEN_CACHE_MODE" \
    -e GOLDEN_CACHE_DIR="$GOLDEN_CACHE_DIR" \
    -e GOLDEN_CACHE_STATS_FILE="/tmp/ci_test_logs/golden_cache_events.tsv" \
    -e GOLDEN_CACHE_REFRESH="${GOLDEN_CACHE_REFRESH:-0}" \
    -e GOLDEN_CACHE_MAX_BYTES="${GOLDEN_CACHE_MAX_BYTES:-42949672960}" \
    -e GOLDEN_CACHE_MAX_AGE_DAYS="${GOLDEN_CACHE_MAX_AGE_DAYS:-30}" \
    -e GOLDEN_CACHE_EVICT_INTERVAL_SECONDS="${GOLDEN_CACHE_EVICT_INTERVAL_SECONDS:-60}" \
    \
    -e FLASH_ATTN_BUILD_VERSION="${FLASH_ATTN_BUILD_VERSION:-all}" \
    -e FLASH_ATTN_BUILD_NPU="${FLASH_ATTN_BUILD_NPU:-auto}" \
    -e GIT_CONFIG_GLOBAL=/tmp/gitconfig \
    \
    -w /workspace/flash-attention-npu \
    "$CI_DOCKER_IMAGE" \
    bash -lc \
      'set -euo pipefail
       : "${ASCEND_RT_VISIBLE_DEVICES:?NPU visibility must be set}"
       [ "$CI_CONTAINER_DEVICE" = 0 ] || { echo "[CI][ERROR] logical NPU must be 0" >&2; exit 1; }
       echo "[CI] NPU mask: physical=$ASCEND_RT_VISIBLE_DEVICES logical=$CI_CONTAINER_DEVICE"
       git config --global --add safe.directory /workspace/flash-attention-npu
       bash "$CI_ENTRY_SCRIPT"' &


  CURRENT_DOCKER_PID=$!


  if wait "$CURRENT_DOCKER_PID"; then
    rc=0
  else
    rc=$?
  fi


  CURRENT_DOCKER_PID=""

  return "$rc"
}


# ----------------------------------------------------------------------------
# 动态选卡 + 执行 task
#
# 返回码:
#   0 = task 成功
#   1 = 当前无可用 NPU
#   2 = task 执行失败
# ----------------------------------------------------------------------------

acquire_lock_and_run_task() {

  # 动态选卡模式:
  #
  # 不持 flock 长期锁，
  # 允许多 runner 共享卡，
  # 靠 task_count 阈值进行软限流。

  local selected
  local line


  line="$(
    bash "$SCRIPT_DIR/detect_npu.sh" --env 2>/dev/null || true
  )"


  selected="${line#export NPU_SELECTED_DEVICE=}"


  if [ -z "$selected" ]; then
    return 1
  fi


  log \
    "selected NPU device=$selected (re-probed, no lock held)"


  if run_docker_task "$selected"; then
    return 0
  fi


  return 2
}


# ============================================================================
# 主流程
# ============================================================================

main() {

  local total_start
  local total_end

  local build_start
  local build_seconds=0
  local build_skipped=false

  local task_start
  local task_seconds=0

  local wait_start
  local wait_seconds=0


  total_start="$(date +%s)"


  log \
    "CI start: $(date '+%Y-%m-%d %H:%M:%S') \
runner=${RUNNER_NAME:-<unset>} \
runner_host=$(hostname) \
task=$CI_TASK"


  # --------------------------------------------------------------------------
  # Phase 1: Build
  # --------------------------------------------------------------------------

  if [ "$CI_SKIP_BUILD" = "true" ]; then

    build_skipped=true

    log \
      "CI_SKIP_BUILD=true, skip phase 1 (assume build/ exists)"

  else

    build_start="$(date +%s)"

    run_build_phase

    build_seconds=$(( $(date +%s) - build_start ))

    log \
      "=== Phase 1 done (build=${build_seconds}s) ==="

  fi


  # --------------------------------------------------------------------------
  # Phase 2: NPU task
  # --------------------------------------------------------------------------

  if [ "$CI_TASK" = "benchmark" ]; then

    log \
      "=== Phase 2: benchmark (dynamic device selection) ==="

    log \
      "benchmark config: \
fa_version=$CI_BENCHMARK_FA_VERSION \
layout=$CI_BENCHMARK_LAYOUT"

  else

    log \
      "=== Phase 2: test (dynamic device selection) ==="

  fi


  task_start="$(date +%s)"


  # 首次探测，确认机器上至少存在候选 NPU

  local cands

  cands="$(get_candidates)"


  if [ -z "$cands" ]; then

    log \
      "no NPU passed the selection filter, diagnostic output:"

    bash "$SCRIPT_DIR/detect_npu.sh" --summary 2>&1 || true

    die \
      "no candidate NPU with \
health=OK \
free>${CI_NPU_MIN_FREE_MB:-1024}MB \
tasks<${CI_NPU_MAX_TASKS:-4}"

  fi


  local wait_max="${CI_NPU_WAIT_MAX_SEC:-600}"
  local wait_interval="${CI_NPU_WAIT_INTERVAL_SEC:-30}"

  local task_passed=false
  local rc


  while [ "$task_passed" != "true" ]; do

    set +e

    acquire_lock_and_run_task
    rc=$?

    set -e


    case "$rc" in

      0)

        task_passed=true

        ;;


      1)

        # 当前无可用卡：
        # 全部超过 task_count 阈值或者空闲显存不足。
        #
        # 等待后重新探测。

        if [ "$wait_max" -le 0 ]; then

          die \
            "no available NPU \
(all cards have >= ${CI_NPU_MAX_TASKS:-4} tasks \
or insufficient free memory) \
after waiting; aborting CI"

        fi


        log \
          "all NPU busy \
(tasks >= ${CI_NPU_MAX_TASKS:-4} \
or free <= ${CI_NPU_MIN_FREE_MB:-1024}MB), \
waiting ${wait_interval}s \
(remaining wait budget: ${wait_max}s)..."


        wait_start="$(date +%s)"

        sleep "$wait_interval"

        now="$(date +%s)"
        wait_seconds=$((wait_seconds + now - wait_start))

        wait_max=$((wait_max - wait_interval))

        ;;


      2)

        # task 本身失败时不重试。
        # 避免 correctness / benchmark 真实错误被重复执行掩盖。

        die "$CI_TASK failed; aborting CI"

        ;;

    esac

  done


  # --------------------------------------------------------------------------
  # Timing
  # --------------------------------------------------------------------------

  total_end="$(date +%s)"


  task_seconds=$((total_end - task_start - wait_seconds))


  log \
    "CI end: $(date '+%Y-%m-%d %H:%M:%S')"


  if [ "$build_skipped" = "true" ]; then

    log \
      "CI timing: \
build=skipped \
task=${CI_TASK}:${task_seconds}s \
npu_wait=${wait_seconds}s \
total=$((total_end - total_start))s"

  else

    log \
      "CI timing: \
build=${build_seconds}s \
task=${CI_TASK}:${task_seconds}s \
npu_wait=${wait_seconds}s \
total=$((total_end - total_start))s"

  fi
}


main "$@"
