#!/usr/bin/env bash
#
# 读取 ci/build_matrix.tsv, 为每个 combo (CANN×torch/torch_npu) 构建专用镜像
# fa-npu-ci:<name> (由 ci/Dockerfile 参数化构建)。
# 构建一次后, ci/run_ci_matrix.sh 直接 docker run 这些镜像并发编译。
#
# 用法:
#   bash ci/build_matrix_images.sh                 # 构建全部 combo
#   bash ci/build_matrix_images.sh <name1> <name2> # 只构建指定 combo
#
# 环境变量:
#   MATRIX_FILE           (默认 ci/build_matrix.tsv)
#   IMAGE_PREFIX          (默认 fa-npu-ci)
#   BUILD_MATRIX_PARALLEL (默认 false) true=并发构建

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MATRIX_FILE="${MATRIX_FILE:-$REPO_ROOT/ci/build_matrix.tsv}"
IMAGE_PREFIX="${IMAGE_PREFIX:-fa-npu-ci}"
PARALLEL="${BUILD_MATRIX_PARALLEL:-false}"

# shellcheck source=ci/docker_proxy.sh
source "$SCRIPT_DIR/docker_proxy.sh"
docker_proxy_init "${GOLDEN_CACHE_HOST_DIR:-/home/FA_NPU_CI_DATA}"

# wheel 缓存: 构建前在 WHEEL_SEARCH_PATHS 里全局查找目标 wheel, 找到则复制到
# WHEEL_CACHE_DIR (作为构建上下文一部分), Dockerfile 里 COPY 到 /wheels/ 并优先使用。
# 这样 wheel 放在宿主机任意位置都能被复用, 省去每次构建重复下载几百 MB。
# 文件名需与 Dockerfile 中拼出的完全一致 (torch 含 '+', 如 torch-2.9.0+cpu-...whl)。
WHEEL_CACHE_DIR="${WHEEL_CACHE_DIR:-$SCRIPT_DIR/wheels}"
WHEEL_SEARCH_PATHS="${WHEEL_SEARCH_PATHS:-$HOME /opt /data /tmp /workspace $REPO_ROOT}"
mkdir -p "$WHEEL_CACHE_DIR"

# 在搜索路径里查找指定文件名, 找到则打印首个匹配路径 (找不到返回空)。
find_wheel() {
  local fname="$1" path
  # WHEEL_SEARCH_PATHS is intentionally a space-separated list of roots.
  # shellcheck disable=SC2086
  while IFS= read -r path; do
    [ -f "$path" ] && { printf '%s\n' "$path"; return 0; }
  done < <(find $WHEEL_SEARCH_PATHS -maxdepth 6 -type f -name "$fname" 2>/dev/null)
  return 1
}

log() { printf '[matrix-images] %s\n' "$*"; }
die() { printf '[matrix-images][ERROR] %s\n' "$*" >&2; exit 1; }

[ -f "$MATRIX_FILE" ] || die "matrix file not found: $MATRIX_FILE"
command -v docker >/dev/null 2>&1 || die "docker not found"

WANT=("$@")

# 架构过滤: 非空时只构建 tsv 第 7 列 (arch) 匹配的 combo。用于多机器分架构构建:
# 950 机器设 ARCH_FILTER=x86_64, 910B 机器设 ARCH_FILTER=aarch64, 各建各的。
ARCH_FILTER="${ARCH_FILTER:-}"

# 输出 combo 行 (跳过 # 和空行); ARCH_FILTER 非空时按第 7 列过滤
read_combos() {
  if [ -n "$ARCH_FILTER" ]; then
    awk -F'|' -v arch="$ARCH_FILTER" '
      /^[[:space:]]*#/ || /^[[:space:]]*$/ {next}
      NF >= 7 && $7 == arch {print}
      NF < 7 {print}
    ' "$MATRIX_FILE"
  else
    awk -F'|' '/^[[:space:]]*#/ || /^[[:space:]]*$/ {next} {print}' "$MATRIX_FILE"
  fi
}

build_one() {
  local line="$1" name base_image py_tag torch_ver torch_npu_ver torch_npu_rel arch image
  IFS='|' read -r name base_image py_tag torch_ver torch_npu_ver torch_npu_rel arch image <<< "$line"
  [ -n "$name" ] || return 0
  arch="${arch:-x86_64}"
  if [ "${#WANT[@]}" -gt 0 ]; then
    local hit=false w
    for w in "${WANT[@]}"; do [ "$w" = "$name" ] && { hit=true; break; }; done
    [ "$hit" = "true" ] || return 0
  fi
  # 第 8 列 (image) 非空: 该 combo 用预构建镜像, 跳过构建
  if [ -n "$image" ]; then
    log "skip $name: using prebuilt image '$image' (no build needed)"
    return 0
  fi
  log "building $IMAGE_PREFIX:$name (base=$base_image torch=$torch_ver torch_npu=$torch_npu_ver py=$py_tag arch=$arch)"

  # 全局查找 torch / torch_npu wheel, 命中则复制到缓存目录供 Dockerfile COPY 使用
  local torch_fname npu_fname found
  torch_fname="torch-${torch_ver}+cpu-${py_tag}-${py_tag}-manylinux_2_28_${arch}.whl"
  npu_fname="torch_npu-${torch_npu_ver}-${py_tag}-${py_tag}-manylinux_2_28_${arch}.whl"
  for pair in "torch:${torch_fname}" "torch_npu:${npu_fname}"; do
    local tag="${pair%%:*}" fname="${pair#*:}"
    if [ -f "$WHEEL_CACHE_DIR/$fname" ]; then
      log "  $tag: cached in $WHEEL_CACHE_DIR/$fname"
      continue
    fi
    found="$(find_wheel "$fname" || true)"
    if [ -n "$found" ]; then
      cp -f "$found" "$WHEEL_CACHE_DIR/$fname"
      log "  $tag: found $found -> cached"
    else
      log "  $tag: not found, will download in build"
    fi
  done

  if docker build -t "$IMAGE_PREFIX:$name" \
        "${DOCKER_PROXY_BUILD_ARGS[@]}" \
        --build-arg BASE_IMAGE="$base_image" \
        --build-arg PY_TAG="$py_tag" \
        --build-arg TORCH_VER="$torch_ver" \
        --build-arg TORCH_NPU_VER="$torch_npu_ver" \
        --build-arg TORCH_NPU_RELEASE="$torch_npu_rel" \
        --build-arg ARCH="$arch" \
        -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"; then
    log "  -> $IMAGE_PREFIX:$name OK"
    return 0
  else
    log "  -> $IMAGE_PREFIX:$name FAILED"
    return 1
  fi
}

overall=0
if [ "$PARALLEL" = "true" ]; then
  pids=()
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    build_one "$line" &
    pids+=($!)
  done < <(read_combos)
  for p in "${pids[@]}"; do wait "$p" || overall=1; done
else
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    build_one "$line" || overall=1
  done < <(read_combos)
fi

if [ "$overall" -ne 0 ]; then
  die "one or more matrix images failed to build"
fi
log "done. built images:"
docker images --format '  {{.Repository}}:{{.Tag}}' | grep "^  ${IMAGE_PREFIX}:" || true
