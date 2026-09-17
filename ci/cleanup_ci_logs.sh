#!/usr/bin/env bash
#
# CI 容器启动时调用，清理历史日志，默认保留 7 天。
#
# 环境变量:
#   CI_LOG_CLEANUP_ENABLED  (默认 true)  是否清理
#   CI_LOG_RETENTION_DAYS   (默认 7)     保留天数
#   CI_LOG_CLEANUP_DIRS     (默认空)     额外清理目录，多个用 ":" 分隔
#
# 默认清理仓库内: log/ logs/ log_ut/ output/log*/ build/log*/ build_out/log*/
# 以及容器内常见 Ascend/NPU 日志目录。

set -euo pipefail

ENABLED="${CI_LOG_CLEANUP_ENABLED:-true}"
RETENTION_DAYS="${CI_LOG_RETENTION_DAYS:-7}"
EXTRA_DIRS="${CI_LOG_CLEANUP_DIRS:-}"

if [ "$ENABLED" != "true" ]; then
  echo "[cleanup_ci_logs] disabled (CI_LOG_CLEANUP_ENABLED!=true)"
  exit 0
fi

if ! [[ "$RETENTION_DAYS" =~ ^[0-9]+$ ]] || [ "$RETENTION_DAYS" -lt 1 ]; then
  echo "[cleanup_ci_logs] invalid CI_LOG_RETENTION_DAYS=$RETENTION_DAYS, fallback to 7"
  RETENTION_DAYS=7
fi

REPO_ROOT="${REPO_ROOT:-$(pwd)}"

# 仓库内日志目录 (相对 REPO_ROOT)
in_repo_dirs=(
  log logs log_ut
  output/log output/logs
  build/log build/logs build/log_ut
  build_out/log build_out/logs build_out/log_ut
)

# 容器内常见 Ascend/NPU 日志目录
system_dirs=(
  /var/log/ascend
  /var/log/npu
  /home/HwHiAiUser/ascend/log
  /usr/slog
  ~/ascend/log
)

echo "[cleanup_ci_logs] retaining logs for $RETENTION_DAYS day(s)"

clean_dir() {
  local dir="$1"
  if [ -d "$dir" ]; then
    # 删除修改时间超过 RETENTION_DAYS 天的文件
    find "$dir" -type f -mtime "+$RETENTION_DAYS" -print -delete 2>/dev/null || true
    # 删除空目录
    find "$dir" -type d -empty -print -delete 2>/dev/null || true
  fi
}

for rel in "${in_repo_dirs[@]}"; do
  clean_dir "$REPO_ROOT/$rel"
done

for dir in "${system_dirs[@]}"; do
  clean_dir "$dir"
done

# 额外目录
if [ -n "$EXTRA_DIRS" ]; then
  IFS=':' read -ra _extra <<< "$EXTRA_DIRS"
  for dir in "${_extra[@]}"; do
    clean_dir "$dir"
  done
fi

echo "[cleanup_ci_logs] done"
