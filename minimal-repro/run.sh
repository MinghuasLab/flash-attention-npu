#!/usr/bin/env bash
set -euo pipefail
# One-shot: build payloads -> build host DSOs -> run -> show device-side softlinks.
# 前置: 已安装 CANN(>=9.1)。若 bisheng 不在 PATH 会自动 source 常见 set_env。
cd "$(dirname "$0")"
if ! command -v bisheng >/dev/null 2>&1; then
    for env_sh in /usr/local/Ascend/ascend-toolkit/set_env.sh \
                  /usr/local/Ascend/cann-9.1.0-beta.1/bin/setenv.bash \
                  /usr/local/Ascend/cann/set_env.sh; do
        [ -f "$env_sh" ] && { echo "[env] sourcing $env_sh"; set +u; source "$env_sh"; set -u; break; }
    done
    command -v bisheng >/dev/null 2>&1 || { echo "ERROR: bisheng not in PATH - source your CANN env first"; exit 1; }
fi
TOOLKIT="${ASCEND_HOME_PATH:-}"
[ -z "$TOOLKIT" ] && for c in /usr/local/Ascend/cann-* /usr/local/Ascend/ascend-toolkit/latest; do [ -d "$c/aarch64-linux" ] && TOOLKIT="$c" && break; done
bash build.sh
bash build_host.sh
echo
echo "===== run ====="
./host
RC=$?
echo
echo "===== device-side softlinks (fold = both soNames -> same target) ====="
for root in "$HOME/ascend/log" /root/ascend/log /var/log/ascend; do
  LOG=$(ls -t "$root"/run/device-*/device-*.log 2>/dev/null | head -1 || true)
  [ -n "${LOG:-}" ] && break
done
if [ -n "${LOG:-}" ] && grep -aq CreateSoftLinkToSoFile "$LOG" 2>/dev/null; then
  grep -a CreateSoftLinkToSoFile "$LOG" | tail -2 \
    | sed -e 's/.*Create soft link so success, //' -e 's/, targetPath=/  ->  /'
else
  echo "(device log not found; search ~/ascend/log/run/device-*/ for the run above)"
fi
exit $RC
