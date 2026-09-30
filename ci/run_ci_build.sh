#!/usr/bin/env bash
#
# 阶段1: 编译 (容器内执行, 不需要 NPU, 不加锁)
#   1. bash ci/init_submodules.sh  (浅拉 + runner 磁盘缓存)
#   2. python setup.py build  (产物在 build/, 通过 volume 持久化供阶段2复用)
#
# 由 ci/run_ci_container.sh 阶段1通过 docker run 调用 (不绑卡)。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(pwd)"

log() { printf '[CI-build] %s\n' "$*"; }
die() { printf '[CI-build][ERROR] %s\n' "$*" >&2; exit 1; }

# git safe.directory (容器内 root 操作宿主机 runner 用户的目录, 会触发 dubious ownership)
git config --global --add safe.directory "$REPO_ROOT"

log "repo=$REPO_ROOT (build phase, no NPU needed)"
log "build phase start: $(date '+%Y-%m-%d %H:%M:%S')"

command -v python3 >/dev/null 2>&1 || die "python3 not found in container"

# ---------- 1. 子模块 (浅拉 + runner 缓存) ----------
log "init submodules: csrc/catlass (shallow + runner cache)"
bash "$SCRIPT_DIR/init_submodules.sh" "$REPO_ROOT"
export FLASH_ATTN_SKIP_SUBMODULE_INIT=1

# ---------- 2. 编译 (python setup.py build_ext --inplace) ----------
# 用 --inplace 把 .so 直接放到源码目录, 避免从仓库根 import 时源码目录
# 遮蔽 site-packages 里的安装包导致找不到 .so。
export FLASH_ATTN_FORCE_BUILD=TRUE
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
log "python setup.py build_ext --inplace (FLASH_ATTN_BUILD_VERSION=${FLASH_ATTN_BUILD_VERSION:-all})"
# 折叠编译输出, 成功时只留一行状态, 失败时把尾部展开打印
build_log="/tmp/ci_build.log"
echo "::group::build_ext output (live, expand to watch)"
set +e
python3 setup.py build_ext --inplace 2>&1 | tee "$build_log"
build_rc="${PIPESTATUS[0]}"
echo "::endgroup::"
set -e
if [ "$build_rc" -ne 0 ]; then
  tail -n 40 "$build_log" || true
  die "build_ext failed rc=$build_rc"
fi
log "build_ext ok"

log "build phase done (artifacts in build/)"
log "build phase end: $(date '+%Y-%m-%d %H:%M:%S')"
