#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${1:-$(git rev-parse --show-toplevel)}"
SUBMODULE_PATH="${2:-csrc/catlass}"

log() { printf '[CI-submodules] %s\n' "$*"; }
die() { printf '[CI-submodules][ERROR] %s\n' "$*" >&2; exit 1; }

[ -d "$REPO_ROOT" ] || die "repository root not found: $REPO_ROOT"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
[ -f "$REPO_ROOT/.gitmodules" ] || die "root .gitmodules not found: $REPO_ROOT"

# CI containers may run as root against a workspace owned by the runner user.
git config --global --add safe.directory "$REPO_ROOT"
git config --global --add safe.directory "$REPO_ROOT/*"

log "init top-level submodule: $SUBMODULE_PATH"
git -C "$REPO_ROOT" submodule update --init -- "$SUBMODULE_PATH"

submodule_root="$REPO_ROOT/$SUBMODULE_PATH"
[ -d "$submodule_root" ] || die "submodule directory not found: $submodule_root"

# Keep the top-level checkout complete, but shallow-clone every dependency below
# it. Git reads each nested repository's .gitmodules while traversing, so paths
# do not need to be duplicated here when a dependency layout changes.
log "init nested submodules recursively (depth=1): $SUBMODULE_PATH"
git -C "$submodule_root" submodule update --init --recursive --depth=1
