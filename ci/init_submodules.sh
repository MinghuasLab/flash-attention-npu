#!/usr/bin/env bash
# Copyright (c) 2026, Minghua Shen.

#
# 浅拉 csrc/catlass (只要 include/ 头文件)。FlashAttention-NPU 编译不依赖
# catlass 嵌套的 googletest / llvm-project / triton (TLA DSL), 也不 --recursive,
# 避免 gitcode 上 llvm 大仓 clone 失败 (remote transport reported error)。
#
# 缓存目录 (默认 /home/FA_NPU_CI_DATA/submodule-cache/<catlass-sha>/):
#   只存工作区文件, 不含 .git。命中后直接 rsync 回 csrc/catlass。
#   未命中则 git submodule update --depth 1 (带重试 + HTTP/1.1), 成功后再写入缓存。
#   工作区用 .ci-submodule-sha 记录 gitlink, 缓存恢复后没有 .git 也能感知版本变化。
#
# 权限:
#   runner 用户常常写不了 $GOLDEN_CACHE_HOST_DIR (root 预建)。可读即可从缓存恢复;
#   只有可写时才回填/清理缓存。写缓存交给 root 编译容器 (见 run_ci_container.sh)。
#
# 环境变量:
#   SUBMODULE_CACHE_DIR   缓存根目录 (默认 $GOLDEN_CACHE_HOST_DIR/submodule-cache)
#   SUBMODULE_CACHE_KEEP  保留的 SHA 目录数 (默认 2)
#   SUBMODULE_CLONE_ATTEMPTS  浅克隆失败重试次数 (默认 3)
#   GOLDEN_CACHE_HOST_DIR 默认 /home/FA_NPU_CI_DATA
#
# 用法: bash ci/init_submodules.sh [repo_root]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${1:-$SCRIPT_DIR/..}" && pwd)"
SUBMODULE_PATH="csrc/catlass"
CACHE_ROOT="${SUBMODULE_CACHE_DIR:-${GOLDEN_CACHE_HOST_DIR:-/home/FA_NPU_CI_DATA}/submodule-cache}"
CACHE_KEEP="${SUBMODULE_CACHE_KEEP:-2}"
CLONE_ATTEMPTS="${SUBMODULE_CLONE_ATTEMPTS:-3}"
STAMP_NAME=".ci-submodule-sha"

log() { printf '[CI-submodule] %s\n' "$*"; }
die() { printf '[CI-submodule][ERROR] %s\n' "$*" >&2; exit 1; }

cd "$REPO_ROOT"
git config --global --add safe.directory "$REPO_ROOT" 2>/dev/null || true

expected_sha="$(git rev-parse "HEAD:${SUBMODULE_PATH}" 2>/dev/null || true)"
if [ -z "$expected_sha" ]; then
  expected_sha="$(git ls-tree HEAD "$SUBMODULE_PATH" 2>/dev/null | awk '{print $3}')"
fi
[ -n "$expected_sha" ] || die "cannot resolve gitlink SHA for ${SUBMODULE_PATH}"

# Headers used by csrc/ascend910 and csrc/ascend950. Nested TLA/llvm trees
# are not part of the FlashAttention-NPU compile.
REQUIRED_RELPATHS=(
  "include/catlass/catlass.hpp"
)

tree_complete() {
  local root="$1" rel
  for rel in "${REQUIRED_RELPATHS[@]}"; do
    [ -e "${root}/${rel}" ] || return 1
  done
  return 0
}

# Stamp is the source of truth after a cache restore (no .git in the tree).
# Fall back to a real submodule checkout HEAD only when this directory has
# its own .git; otherwise `git -C` would walk to the parent repo.
workspace_recorded_sha() {
  local stamp="${REPO_ROOT}/${SUBMODULE_PATH}/${STAMP_NAME}"
  local recorded=""
  if [ -f "$stamp" ]; then
    recorded="$(tr -d '[:space:]' < "$stamp")"
    printf '%s' "$recorded"
    return 0
  fi
  if [ -e "${REPO_ROOT}/${SUBMODULE_PATH}/.git" ]; then
    git -C "${REPO_ROOT}/${SUBMODULE_PATH}" rev-parse HEAD 2>/dev/null || true
  fi
}

workspace_ready() {
  tree_complete "${REPO_ROOT}/${SUBMODULE_PATH}" || return 1
  [ "$(workspace_recorded_sha)" = "$expected_sha" ]
}

write_stamp() {
  mkdir -p "${REPO_ROOT}/${SUBMODULE_PATH}"
  printf '%s\n' "$expected_sha" > "${REPO_ROOT}/${SUBMODULE_PATH}/${STAMP_NAME}"
}

sync_tree() {
  local src="$1" dst="$2"
  mkdir -p "$dst"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete --exclude='.git' --exclude='.ci-submodule-complete' --info=stats2 \
      "${src}/" "${dst}/"
  else
    local tmp
    tmp="$(mktemp -d "${dst}.tmp.XXXXXX")"
    cp -a "${src}/." "$tmp/"
    find "$tmp" \( -name .git -o -name .ci-submodule-complete \) -prune -exec rm -rf {} +
    rm -rf "$dst"
    mv "$tmp" "$dst"
  fi
}

save_cache() {
  log "save runner cache ${CACHE_TREE}"
  sync_tree "${REPO_ROOT}/${SUBMODULE_PATH}" "$CACHE_TREE"
  touch "${CACHE_TREE}/.ci-submodule-complete"
  if ! command -v find >/dev/null 2>&1; then
    return 0
  fi
  # Keep the newest CACHE_KEEP SHA directories; drop the rest.
  mapfile -t stale < <(
    find "$CACHE_ROOT" -mindepth 1 -maxdepth 1 -type d -name '[0-9a-f]*' -printf '%T@ %p\n' \
      | sort -nr \
      | awk -v keep="$CACHE_KEEP" 'NR > keep { print $2 }'
  )
  if [ "${#stale[@]}" -gt 0 ]; then
    for stale_dir in "${stale[@]}"; do
      [ -n "$stale_dir" ] || continue
      log "prune old cache $(basename "$stale_dir")"
      rm -rf "$stale_dir"
    done
  fi
}

cache_readable=0
cache_writable=0
if mkdir -p "$CACHE_ROOT" 2>/dev/null || [ -d "$CACHE_ROOT" ]; then
  if [ -d "$CACHE_ROOT" ]; then
    [ -r "$CACHE_ROOT" ] && cache_readable=1
    [ -w "$CACHE_ROOT" ] && cache_writable=1
  fi
fi
if [ "$cache_readable" != 1 ]; then
  log "runner cache unavailable ($CACHE_ROOT); shallow clone only"
fi

CACHE_TREE=""
if [ "$cache_readable" = 1 ]; then
  CACHE_TREE="${CACHE_ROOT}/${expected_sha}"
fi

# Serialize cache restore/save across jobs on the same runner.
# Exclusive lock when we can write; shared lock is enough for a read-only restore
# (runner user often cannot create the lock file on a root-owned cache).
if [ "$cache_writable" = 1 ]; then
  exec 9>"${CACHE_ROOT}/.lock"
  flock 9
elif [ "$cache_readable" = 1 ] && [ -e "${CACHE_ROOT}/.lock" ]; then
  exec 9<"${CACHE_ROOT}/.lock"
  flock -s 9
fi

if workspace_ready; then
  log "workspace already has ${SUBMODULE_PATH} @ ${expected_sha}"
  write_stamp
  if [ "$cache_writable" = 1 ] && [ ! -f "${CACHE_TREE}/.ci-submodule-complete" ]; then
    save_cache
  fi
  exit 0
fi

if [ "$cache_readable" = 1 ] && [ -f "${CACHE_TREE}/.ci-submodule-complete" ] && tree_complete "$CACHE_TREE"; then
  log "cache hit ${expected_sha} -> ${SUBMODULE_PATH}"
  start="$(date +%s)"
  rm -rf "${REPO_ROOT}/${SUBMODULE_PATH}"
  sync_tree "$CACHE_TREE" "${REPO_ROOT}/${SUBMODULE_PATH}"
  write_stamp
  log "restored from runner cache in $(( $(date +%s) - start ))s"
  workspace_ready || die "cache restore incomplete for ${expected_sha}"
  exit 0
fi

log "cache miss ${expected_sha}; shallow clone catlass only (--depth 1, no recursive llvm/triton)"
# Leftover files from a previous cache restore are a regular directory (no .git).
# Clear them so `git submodule update` can recreate the gitlink checkout.
start="$(date +%s)"
clone_ok=0
attempt=1
while [ "$attempt" -le "$CLONE_ATTEMPTS" ]; do
  rm -rf "${REPO_ROOT}/${SUBMODULE_PATH}"
  log "clone attempt ${attempt}/${CLONE_ATTEMPTS}"
  if git -c advice.detachedHead=false \
      -c http.postBuffer=524288000 \
      -c http.version=HTTP/1.1 \
      -c http.lowSpeedLimit=1000 \
      -c http.lowSpeedTime=60 \
      submodule update --init --depth 1 --recommend-shallow \
      "$SUBMODULE_PATH"; then
    clone_ok=1
    break
  fi
  log "clone attempt ${attempt}/${CLONE_ATTEMPTS} failed; retry in 15s"
  attempt=$((attempt + 1))
  [ "$attempt" -le "$CLONE_ATTEMPTS" ] && sleep 15
done
[ "$clone_ok" = 1 ] || die "shallow clone of ${SUBMODULE_PATH} failed after ${CLONE_ATTEMPTS} attempts"
log "shallow clone done in $(( $(date +%s) - start ))s"
write_stamp
workspace_ready || die "shallow clone missing catlass headers"

if [ "$cache_writable" = 1 ]; then
  save_cache
fi
