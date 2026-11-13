#!/usr/bin/env bash
# Copyright (c) 2026, Minghua Shen.

# Git-only proxy for CI. This helper never exports HTTP_PROXY/HTTPS_PROXY/NO_PROXY.
#
# Git uses http.<url>.proxy so only whitelist hosts are proxied (default: GitHub).
# Other remotes (for example gitcode.com submodules) stay direct.
#
# Usage:
#   source "$SCRIPT_DIR/docker_proxy.sh"
#   git_proxy_init "$GOLDEN_CACHE_HOST_DIR"
#   docker run "${GIT_PROXY_DOCKER_ARGS[@]}" ...
#
# Config file (default /home/FA_NPU_CI_DATA/proxy.conf), KEY=VALUE lines:
#   GIT_PROXY=http://127.0.0.1:18790
#   GIT_PROXY_WHITELIST=github.com,gist.github.com,api.github.com
# HTTP_PROXY/HTTPS_PROXY keys in the file are accepted as aliases for GIT_PROXY
# so existing runner configs keep working. Environment HTTP_PROXY is ignored.
# Blank GIT_PROXY disables the proxy. Blank GIT_PROXY_WHITELIST proxies nothing.
#
# One-off overrides: CI_GIT_PROXY, CI_GIT_PROXY_WHITELIST.
# Debug: bash ci/docker_proxy.sh [data_dir]

GIT_PROXY_DEFAULT_WHITELIST="github.com,gist.github.com,api.github.com,raw.githubusercontent.com,codeload.github.com,objects.githubusercontent.com"

git_proxy_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

git_proxy_clear_http_env() {
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
}

git_proxy_append_url() {
  local file="$1" url="$2" proxy="$3"
  printf '[http "%s"]\n\tproxy = %s\n' "$url" "$proxy" >> "$file"
}

git_proxy_write_config() {
  local file="$1" host
  : > "$file"
  [ -n "${GIT_PROXY_URL:-}" ] || return 0

  local hosts="${GIT_PROXY_WHITELIST:-}"
  hosts="${hosts//,/ }"
  for host in $hosts; do
    host="$(git_proxy_trim "$host")"
    [ -n "$host" ] || continue
    if [[ "$host" == *"://"* ]]; then
      git_proxy_append_url "$file" "$host" "$GIT_PROXY_URL"
    else
      git_proxy_append_url "$file" "https://$host" "$GIT_PROXY_URL"
      git_proxy_append_url "$file" "https://$host/" "$GIT_PROXY_URL"
      git_proxy_append_url "$file" "http://$host" "$GIT_PROXY_URL"
      git_proxy_append_url "$file" "http://$host/" "$GIT_PROXY_URL"
    fi
  done
}

git_proxy_init() {
  local data_dir="${1:-${GOLDEN_CACHE_HOST_DIR:-/home/FA_NPU_CI_DATA}}"
  local config_file="${PROXY_CONFIG_FILE:-$data_dir/proxy.conf}"
  local host_ips=""
  local proxy_value=""
  local whitelist_value="$GIT_PROXY_DEFAULT_WHITELIST"
  local whitelist_set=0
  local config_key config_value

  git_proxy_clear_http_env

  # Host networking makes 127.0.0.1 inside the container refer to the host proxy.
  host_ips="$(hostname -I 2>/dev/null || true)"
  case " $host_ips " in
    *" 192.168.13.241 "*)
      proxy_value="http://127.0.0.1:18790"
      ;;
    *" 192.168.9.226 "*)
      proxy_value="http://127.0.0.1:17890"
      ;;
  esac

  # Parsed as data, not sourced as shell. Blank GIT_PROXY disables the proxy.
  if [ -f "$config_file" ]; then
    while IFS='=' read -r config_key config_value || [ -n "$config_key" ]; do
      config_key="$(git_proxy_trim "$config_key")"
      [ -z "$config_key" ] && continue
      [[ "$config_key" == \#* ]] && continue
      config_value="$(git_proxy_trim "$config_value")"
      case "$config_key" in
        GIT_PROXY|HTTPS_PROXY|HTTP_PROXY|https_proxy|http_proxy)
          proxy_value="$config_value"
          ;;
        GIT_PROXY_WHITELIST|PROXY_WHITELIST)
          whitelist_value="$config_value"
          whitelist_set=1
          ;;
      esac
    done < "$config_file"
  fi

  [ "${CI_GIT_PROXY+x}" = x ] && proxy_value="$CI_GIT_PROXY"
  if [ "${CI_GIT_PROXY_WHITELIST+x}" = x ]; then
    whitelist_value="$CI_GIT_PROXY_WHITELIST"
    whitelist_set=1
  fi

  GIT_PROXY_URL="$proxy_value"
  if [ "$whitelist_set" = 1 ]; then
    GIT_PROXY_WHITELIST="$whitelist_value"
  else
    GIT_PROXY_WHITELIST="$GIT_PROXY_DEFAULT_WHITELIST"
  fi

  GIT_PROXY_CONFIG_FILE="$(mktemp "${TMPDIR:-/tmp}/ci-git-proxy.XXXXXX.conf")"
  git_proxy_write_config "$GIT_PROXY_CONFIG_FILE"
  chmod 644 "$GIT_PROXY_CONFIG_FILE" 2>/dev/null || true

  GIT_PROXY_DOCKER_ARGS=()
  if [ -n "$GIT_PROXY_URL" ] && [ -s "$GIT_PROXY_CONFIG_FILE" ]; then
    GIT_PROXY_DOCKER_ARGS+=(
      -v "$GIT_PROXY_CONFIG_FILE:/tmp/ci-git-proxy.conf:ro"
      -e GIT_CONFIG_COUNT=1
      -e GIT_CONFIG_KEY_0=include.path
      -e GIT_CONFIG_VALUE_0=/tmp/ci-git-proxy.conf
    )
    export GIT_CONFIG_COUNT=1
    export GIT_CONFIG_KEY_0=include.path
    export GIT_CONFIG_VALUE_0="$GIT_PROXY_CONFIG_FILE"
    printf '[git-proxy] enabled url=%s whitelist=%s\n' "$GIT_PROXY_URL" "$GIT_PROXY_WHITELIST"
  else
    unset GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0
    printf '[git-proxy] disabled\n'
  fi
}

# Old name kept so existing `source ci/docker_proxy.sh` call sites keep working.
docker_proxy_init() {
  git_proxy_init "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  git_proxy_init "${1:-${GOLDEN_CACHE_HOST_DIR:-/home/FA_NPU_CI_DATA}}"
  printf 'GIT_PROXY_URL=%s\n' "${GIT_PROXY_URL:-}"
  printf 'GIT_PROXY_WHITELIST=%s\n' "${GIT_PROXY_WHITELIST:-}"
  printf 'GIT_PROXY_CONFIG_FILE=%s\n' "${GIT_PROXY_CONFIG_FILE:-}"
  if [ -s "${GIT_PROXY_CONFIG_FILE:-}" ]; then
    cat "$GIT_PROXY_CONFIG_FILE"
  fi
fi
