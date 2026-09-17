#!/usr/bin/env bash
# Stop and remove only containers owned by the current GitHub Actions job.

set -euo pipefail

SCOPE="${CI_CONTAINER_SCOPE:-}"
STOP_TIMEOUT="${CI_CONTAINER_STOP_TIMEOUT:-10}"

log() { printf '[cleanup-ci-containers] %s\n' "$*"; }
die() { printf '[cleanup-ci-containers][ERROR] %s\n' "$*" >&2; exit 1; }

[ -n "$SCOPE" ] || die "CI_CONTAINER_SCOPE is required; refusing an unscoped cleanup"
[[ "$STOP_TIMEOUT" =~ ^[0-9]+$ ]] || die "CI_CONTAINER_STOP_TIMEOUT must be a non-negative integer"
command -v docker >/dev/null 2>&1 || die "docker not found"

mapfile -t container_ids < <(
  docker ps -aq \
    --filter "label=com.flash-attention-npu.ci.scope=$SCOPE"
)

if [ "${#container_ids[@]}" -eq 0 ]; then
  log "no containers found for scope=$SCOPE"
  exit 0
fi

log "stopping ${#container_ids[@]} container(s) for scope=$SCOPE: ${container_ids[*]}"
docker stop --time "$STOP_TIMEOUT" "${container_ids[@]}" >/dev/null 2>&1 || true

# --rm normally removes stopped containers. Force-remove anything left behind.
mapfile -t remaining_ids < <(
  docker ps -aq \
    --filter "label=com.flash-attention-npu.ci.scope=$SCOPE"
)
if [ "${#remaining_ids[@]}" -gt 0 ]; then
  log "force-removing remaining container(s): ${remaining_ids[*]}"
  docker rm -f "${remaining_ids[@]}" >/dev/null 2>&1 || true
fi

log "done"
