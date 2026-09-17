#!/usr/bin/env bash
# Load an optional, Git-native proxy configuration into Docker containers.
#
# Usage:
#   source "$SCRIPT_DIR/git_proxy.sh"
#   git_proxy_init "$GOLDEN_CACHE_HOST_DIR"
#   docker run "${DOCKER_GIT_PROXY_ARGS[@]}" ...
#
# The configuration defaults to /home/FA_NPU_CI_DATA/git-proxy.conf and can be
# overridden with GIT_PROXY_CONFIG_FILE.
#
# The file uses normal Git config syntax, for example:
#   [http "https://github.com"]
#       proxy = http://127.0.0.1:17890
#
# Only Git reads this configuration.

git_proxy_init() {
  local data_dir="${1:-${GOLDEN_CACHE_HOST_DIR:-/home/FA_NPU_CI_DATA}}"
  local config_file="${GIT_PROXY_CONFIG_FILE:-$data_dir/git-proxy.conf}"
  local config_dir config_name resolved_config
  local container_config="/tmp/flash-attention-npu-git-proxy.conf"

  DOCKER_GIT_PROXY_ARGS=()

  # A missing file means direct access. This keeps proxy use opt-in and lets a
  # runner disable it simply by moving or removing its local configuration.
  [ -f "$config_file" ] || return 0
  [ -r "$config_file" ] || {
    printf '[git-proxy][ERROR] config file is not readable: %s\n' "$config_file" >&2
    return 1
  }

  command -v git >/dev/null 2>&1 || {
    printf '[git-proxy][ERROR] git is required to validate: %s\n' "$config_file" >&2
    return 1
  }
  if ! git config --file "$config_file" --list >/dev/null; then
    printf '[git-proxy][ERROR] invalid Git config: %s\n' "$config_file" >&2
    return 1
  fi

  config_dir="$(cd "$(dirname "$config_file")" && pwd -P)"
  config_name="$(basename "$config_file")"
  resolved_config="$config_dir/$config_name"

  # shellcheck disable=SC2034  # Public array consumed by sourcing callers.
  DOCKER_GIT_PROXY_ARGS=(
    -v "$resolved_config:$container_config:ro"
    -e "GIT_CONFIG_COUNT=1"
    -e "GIT_CONFIG_KEY_0=include.path"
    -e "GIT_CONFIG_VALUE_0=$container_config"
  )
}
