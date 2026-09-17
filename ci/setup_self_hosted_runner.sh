#!/usr/bin/env bash
#
# 注册 GitHub Actions self-hosted runner。
#
# 用法:
#   bash ci/setup_self_hosted_runner.sh \
#     --url https://github.com/<org>/<repo> \
#     --token <registration_token>
#
# 环境变量:
#   RUNNER_ROOT        (默认 /workspace/actions-runner/flash-attention-npu)
#   RUNNER_NAME        (默认 <hostname>-flash-attention-npu)
#   RUNNER_LABELS      (默认 linux,arm64,npu,flash-attention-npu)
#   RUNNER_VERSION     (默认 2.319.0)  runner 版本
#
# 注册 token 由 GitHub 仓库管理员在
#   Settings -> Actions -> Runners -> New self-hosted runner
# 生成, 时效短, 过期后重新生成即可。

set -euo pipefail

RUNNER_ROOT="${RUNNER_ROOT:-/workspace/actions-runner/flash-attention-npu}"
RUNNER_NAME="${RUNNER_NAME:-$(hostname)-flash-attention-npu}"
# 标签按本机架构自动设: x86_64 -> x64, aarch64 -> arm64; 都带 linux,npu,flash-attention-npu。
# workflow 用这些标签把 job 调度到对应架构的 runner。
case "$(uname -m)" in
  x86_64)        RUNNER_LABELS="${RUNNER_LABELS:-linux,x64,npu,flash-attention-npu}" ;;
  aarch64|arm64) RUNNER_LABELS="${RUNNER_LABELS:-linux,arm64,npu,flash-attention-npu}" ;;
  *)             RUNNER_LABELS="${RUNNER_LABELS:-linux,npu,flash-attention-npu}" ;;
esac
RUNNER_VERSION="${RUNNER_VERSION:-2.319.0}"

REPO_URL=""
TOKEN=""

log() { printf '[runner-setup] %s\n' "$*"; }
die() { printf '[runner-setup][ERROR] %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --url) REPO_URL="$2"; shift 2 ;;
    --token) TOKEN="$2"; shift 2 ;;
    --name) RUNNER_NAME="$2"; shift 2 ;;
    --labels) RUNNER_LABELS="$2"; shift 2 ;;
    --root) RUNNER_ROOT="$2"; shift 2 ;;
    *) die "unknown arg: $1" ;;
  esac
done

[ -n "$REPO_URL" ] || die "--url is required"
[ -n "$TOKEN" ] || die "--token is required"

arch="$(uname -m)"
case "$arch" in
  aarch64|arm64) runner_arch="arm64" ;;
  x86_64)        runner_arch="x64" ;;
  *) die "unsupported arch: $arch" ;;
esac

log "repo=$REPO_URL"
log "runner root=$RUNNER_ROOT"
log "runner name=$RUNNER_NAME"
log "runner labels=$RUNNER_LABELS"

# GitHub Actions Runner 拒绝以 root 身份运行 config.sh
if [ "$(id -u)" -eq 0 ]; then
  die "must not run with sudo/root. GitHub runner refuses root. Re-run as a normal user (with sudo available for svc install)."
fi

mkdir -p "$RUNNER_ROOT"
cd "$RUNNER_ROOT"

# 1. 下载并解压 runner (已解压则跳过; 已有 tar 包则直接解压; 否则下载)
if [ -x "./run.sh" ]; then
  log "runner already extracted at $RUNNER_ROOT, skip download"
else
  pkg="actions-runner-linux-${runner_arch}-${RUNNER_VERSION}.tar.gz"
  url="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${pkg}"
  if [ -f "$pkg" ]; then
    log "found existing $pkg, skip download"
  else
    log "downloading $url"
    if ! curl -fL -o "$pkg" "$url"; then
      die "download failed; check network/proxy to github.com"
    fi
  fi
  tar xzf "$pkg"
  # 保留 tar 包, 下次重装/换目录时可复用 (不删)
  log "extracted runner to $RUNNER_ROOT"
fi

# 2. 配置 (带 --replace 替换同名 runner)
log "configuring runner"
./config.sh --url "$REPO_URL" \
  --token "$TOKEN" \
  --name "$RUNNER_NAME" \
  --labels "$RUNNER_LABELS" \
  --replace \
  --unattended \
  --work "_work"

log "runner configured"

# 3. 安装 systemd 服务 (用 sudo, 但 config.sh 已经用普通用户跑完了)
if sudo -n true 2>/dev/null; then
  log "installing systemd service (sudo)"
  sudo ./svc.sh install
  sudo ./svc.sh start
  sudo ./svc.sh status || true
  log "svc installed and started; check GitHub Settings -> Actions -> Runners for green Idle"
else
  log "sudo not available non-interactively; install svc manually:"
  log "  cd $RUNNER_ROOT && sudo ./svc.sh install && sudo ./svc.sh start"
fi

log "done"
