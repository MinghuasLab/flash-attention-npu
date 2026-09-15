#!/usr/bin/env bash
# Build the hardware-independent quality image and run the common checker in it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
QUALITY_IMAGE="${QUALITY_IMAGE:-flash-attention-npu-quality:local}"
QUALITY_BASE_IMAGE="${QUALITY_BASE_IMAGE:-python:3.11-slim}"

source "$REPO_ROOT/ci/docker_proxy.sh"
docker_proxy_init "${GOLDEN_CACHE_HOST_DIR:-/home/FA_NPU_CI_DATA}"

if ! command -v docker >/dev/null 2>&1; then
    echo "[code-quality] docker is required for the container entry point" >&2
    exit 1
fi

docker build --network host \
    "${DOCKER_PROXY_BUILD_ARGS[@]}" \
    --build-arg "BASE_IMAGE=$QUALITY_BASE_IMAGE" \
    -f "$REPO_ROOT/ci/Dockerfile.code_quality" \
    -t "$QUALITY_IMAGE" \
    "$REPO_ROOT"

docker run --rm \
    --network host \
    "${DOCKER_PROXY_ENV_ARGS[@]}" \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp/quality-home \
    -v "$REPO_ROOT:/workspace" \
    -w /workspace \
    "$QUALITY_IMAGE" \
    ci/code_quality.sh "$@"
