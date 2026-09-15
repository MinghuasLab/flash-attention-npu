#!/usr/bin/env bash
# Shared proxy resolution for Docker entry points.
#
# Usage:
#   source "$SCRIPT_DIR/docker_proxy.sh"
#   docker_proxy_init "$GOLDEN_CACHE_HOST_DIR"
#   docker run "${DOCKER_PROXY_ENV_ARGS[@]}" ...
#   docker build "${DOCKER_PROXY_BUILD_ARGS[@]}" ...
#
# The optional config file defaults to /home/FA_NPU_CI_DATA/proxy.conf and uses
# simple KEY=VALUE lines for HTTP_PROXY, HTTPS_PROXY, and NO_PROXY. CI_HTTP_PROXY,
# CI_HTTPS_PROXY, and CI_NO_PROXY provide one-off overrides.

docker_proxy_init() {
  local data_dir="${1:-${GOLDEN_CACHE_HOST_DIR:-/home/FA_NPU_CI_DATA}}"
  local config_file="${PROXY_CONFIG_FILE:-$data_dir/proxy.conf}"
  local host_ips=""
  local http_proxy_value="${HTTP_PROXY:-}"
  local https_proxy_value="${HTTPS_PROXY:-}"
  local no_proxy_value="${NO_PROXY:-}"
  local config_key config_value

  # These are the defaults for the two supported CI hosts. Host networking makes
  # 127.0.0.1 inside the container refer to the host's proxy service.
  host_ips="$(hostname -I 2>/dev/null || true)"
  case " $host_ips " in
    *" 192.168.13.241 "*)
      http_proxy_value="http://127.0.0.1:18790"
      https_proxy_value="http://127.0.0.1:18790"
      no_proxy_value="localhost,127.0.0.1,::1"
      ;;
    *" 192.168.9.226 "*)
      http_proxy_value="http://127.0.0.1:17890"
      https_proxy_value="http://127.0.0.1:17890"
      no_proxy_value="localhost,127.0.0.1,::1"
      ;;
  esac

  # The config is deliberately parsed as data instead of sourced as shell code.
  # Blank values are accepted so a local config can explicitly disable a proxy.
  if [ -f "$config_file" ]; then
    while IFS='=' read -r config_key config_value || [ -n "$config_key" ]; do
      config_key="${config_key#"${config_key%%[![:space:]]*}"}"
      config_key="${config_key%"${config_key##*[![:space:]]}"}"
      [ -z "$config_key" ] && continue
      [[ "$config_key" == \#* ]] && continue
      config_value="${config_value#"${config_value%%[![:space:]]*}"}"
      config_value="${config_value%"${config_value##*[![:space:]]}"}"
      case "$config_key" in
        HTTP_PROXY|http_proxy) http_proxy_value="$config_value" ;;
        HTTPS_PROXY|https_proxy) https_proxy_value="$config_value" ;;
        NO_PROXY|no_proxy) no_proxy_value="$config_value" ;;
      esac
    done < "$config_file"
  fi

  # CI_* variables are useful for a one-off override without editing the
  # persistent file. They also make the resolver straightforward to test.
  [ "${CI_HTTP_PROXY+x}" = x ] && http_proxy_value="$CI_HTTP_PROXY"
  [ "${CI_HTTPS_PROXY+x}" = x ] && https_proxy_value="$CI_HTTPS_PROXY"
  [ "${CI_NO_PROXY+x}" = x ] && no_proxy_value="$CI_NO_PROXY"

  DOCKER_PROXY_ENV_ARGS=()
  DOCKER_PROXY_BUILD_ARGS=()
  if [ -n "$http_proxy_value" ]; then
    DOCKER_PROXY_ENV_ARGS+=(-e "HTTP_PROXY=$http_proxy_value")
    DOCKER_PROXY_BUILD_ARGS+=(--build-arg "HTTP_PROXY=$http_proxy_value")
  fi
  if [ -n "$https_proxy_value" ]; then
    DOCKER_PROXY_ENV_ARGS+=(-e "HTTPS_PROXY=$https_proxy_value")
    DOCKER_PROXY_BUILD_ARGS+=(--build-arg "HTTPS_PROXY=$https_proxy_value")
  fi
  if [ -n "$no_proxy_value" ]; then
    DOCKER_PROXY_ENV_ARGS+=(-e "NO_PROXY=$no_proxy_value")
    DOCKER_PROXY_BUILD_ARGS+=(--build-arg "NO_PROXY=$no_proxy_value")
  fi
}
