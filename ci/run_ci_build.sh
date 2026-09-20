#!/usr/bin/env bash
set -euo pipefail
log() { printf '[CI-build] %s\n' "$*"; }
log "init catlass submodule"
git submodule update --init csrc/catlass
log "done"
