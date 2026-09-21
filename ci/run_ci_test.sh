#!/usr/bin/env bash
set -euo pipefail
# CI test 入口（最小复现模式）: 编译 → 同进程先后触发 v3/v4 → 设备侧软链 + 固件版本
# 全部输出直出终端; 详细判读逻辑与说明见 ci/repro_min.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
exec bash "$REPO_DIR/ci/repro_min.sh"
