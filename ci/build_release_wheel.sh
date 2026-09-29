#!/usr/bin/env bash
# Build and validate one release-matrix combination inside its CANN image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODE="${RELEASE_BUILD_MODE:-wheel}"

log() { printf '[release-build] %s\n' "$*"; }
die() { printf '[release-build][ERROR] %s\n' "$*" >&2; exit 1; }
require() { [ -n "${!1:-}" ] || die "$1 is required"; }

for variable in MATRIX_NAME MATRIX_PY_TAG MATRIX_TORCH_VERSION MATRIX_TORCH_NPU_VERSION \
  MATRIX_CANN_VERSION MATRIX_NPU MATRIX_BUILD_VERSION MATRIX_ABI MATRIX_ARCH; do
  require "$variable"
done

command -v python3 >/dev/null 2>&1 || die "python3 not found"
command -v bisheng >/dev/null 2>&1 || die "bisheng not found"

cd "$REPO_ROOT"
if [ "${FLASH_ATTN_SKIP_SUBMODULE_INIT:-0}" != "1" ]; then
  git submodule update --init --recursive csrc/catlass
fi
[ -f csrc/catlass/include/catlass/catlass.hpp ] || die "CATLASS headers are missing"

export FLASH_ATTN_FORCE_BUILD=TRUE
export FLASH_ATTN_BUILD_NPU="$MATRIX_NPU"
export FLASH_ATTN_BUILD_VERSION="$MATRIX_BUILD_VERSION"
export FLASH_ATTN_WHEEL_VARIANT="$MATRIX_BUILD_VERSION"
export FLASH_ATTN_CANN_VERSION="$MATRIX_CANN_VERSION"
export FLASH_ATTN_TORCH_VERSION="$MATRIX_TORCH_VERSION"
export FLASH_ATTN_TORCH_NPU_VERSION="$MATRIX_TORCH_NPU_VERSION"
export FLASH_ATTN_PYTHON_TAG="$MATRIX_PY_TAG"
export FLASH_ATTN_PYTHON_ABI_TAG="$MATRIX_PY_TAG"
export FLASH_ATTN_PLATFORM_TAG="linux_${MATRIX_ARCH}"
if [ "${MATRIX_ABI^^}" = "DETECTED" ]; then
  unset FLASH_ATTN_FORCE_CXX11_ABI || true
else
  export FLASH_ATTN_FORCE_CXX11_ABI="${MATRIX_ABI^^}"
fi

detected_abi="$(python3 -c 'import torch; print(str(bool(torch._C._GLIBCXX_USE_CXX11_ABI)).upper())')"
if [ "${MATRIX_ABI^^}" != "DETECTED" ] && [ "${MATRIX_ABI^^}" != "$detected_abi" ]; then
  die "requested ABI ${MATRIX_ABI^^} does not match the image ABI $detected_abi"
fi
export FLASH_ATTN_FORCE_CXX11_ABI="$detected_abi"

log "[build-config] name=$MATRIX_NAME npu=$MATRIX_NPU api=$MATRIX_BUILD_VERSION cann=$MATRIX_CANN_VERSION torch=$MATRIX_TORCH_VERSION torch_npu=$MATRIX_TORCH_NPU_VERSION abi=$detected_abi jobs=${FLASH_ATTN_BUILD_JOBS:-auto}"

if [ "$MODE" = "build-only" ]; then
  python3 setup.py build --build-base="${RELEASE_BUILD_BASE:-/tmp/flash-attn-build-$MATRIX_NAME}"
  exit 0
fi
[ "$MODE" = "wheel" ] || die "unsupported RELEASE_BUILD_MODE=$MODE"

dist_dir="${RELEASE_DIST_DIR:-$REPO_ROOT/dist/$MATRIX_NAME}"
mkdir -p "$dist_dir"
find "$dist_dir" -maxdepth 1 -type f -name '*.whl' -delete
python3 setup.py bdist_wheel --dist-dir="$dist_dir"

mapfile -t wheels < <(find "$dist_dir" -maxdepth 1 -type f -name '*.whl' -print)
[ "${#wheels[@]}" -eq 1 ] || die "expected one wheel in $dist_dir, found ${#wheels[@]}"
wheel_path="${wheels[0]}"
python3 ci/validate_wheel.py "$wheel_path"
install_dir="$(mktemp -d)"
trap 'rm -rf "$install_dir"' EXIT
python3 -m pip install --no-deps --no-index --target "$install_dir" "$wheel_path"
PYTHONPATH="$install_dir" python3 -c \
  'import importlib.metadata; print(importlib.metadata.version("flash-attn-npu"))'
wheel_sha256="$(sha256sum "$wheel_path" | awk '{print $1}')"
log "wheel=$wheel_path"
log "sha256=$wheel_sha256"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  printf 'wheel_path=%s\nwheel_name=%s\nwheel_sha256=%s\n' \
    "$wheel_path" "$(basename "$wheel_path")" "$wheel_sha256" >> "$GITHUB_OUTPUT"
fi
