#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# Ascend 910 自定义 AICPU kernel 部署折叠 bug 最小复现脚本
#
# 用法:   bash ci/repro_min.sh
# 现象:   同进程先后部署两个【编译产物字节数完全相同】的自定义 AICPU kernel,
#         第二个 kernel 执行报 errcode 11003 (get kernel failed)
# 判读:   看最后 [3/3] 的两条软链 ——
#         两条 linkPath 指向同一 targetPath  = 折叠复现 (BUG)
#         两条 linkPath 指向不同 targetPath = 环境健康
# 说明:   build 输出直接打印到终端(需数分钟, 有持续输出即正常);
#         无需 root; 设备日志在 $HOME/ascend 或 /root/ascend 下自动查找
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"
T0=$(date +%s)
log() { printf '\n===== [%s] %s (elapsed %ss) =====\n' "$(date +%H:%M:%S)" "$*" "$(( $(date +%s) - T0 ))"; }

log "0/3 环境信息"
head -1 /usr/local/Ascend/driver/version.info 2>/dev/null || echo "driver version.info 未找到"
echo "firmware: $(head -1 /usr/local/Ascend/firmware/version.info 2>/dev/null || echo 'version.info 未找到')"
npu-smi info -t board -i 0 2>/dev/null | grep -iE "Software Version|Firmware Version" || true
python3 -c "import torch,torch_npu; print('torch_npu device_count:', torch_npu.npu.device_count())" || exit 1

log "1/3 编译 twin v3/v4（build 输出直出终端，持续滚动属正常，请等待）"
FLASH_ATTN_BUILD_VERSION=all FLASH_ATTN_BUILD_NPU=910 python3 setup.py install

log "2/3 同进程先后触发 v3 → v4"
# 在 /tmp 下运行: 避免 cwd 里的源码目录遮蔽已安装的编译包(循环导入)
(cd /tmp && python3 - <<'PYEOF'
import os, torch, torch_npu
import flash_attn_npu_3 as fa3
import flash_attn_npu_4 as fa4
pid = os.getpid()
with open("/tmp/repro_pid", "w") as f: f.write(str(pid))
print(f"[REPRO] pid={pid}")
skc = torch.tensor([0,128,256], dtype=torch.int32, device="npu")
try:
    fa3.get_scheduler_metadata(2,256,256,8,8,128,None,skc,is_seqlens_k_cumulative=True)
    torch.npu.synchronize(); print("[REPRO] v3 : OK")
except Exception as e: print(f"[REPRO] v3 : FAILED -- {e}")
try:
    fa4.get_scheduler_metadata(2,256,256,8,8,128,None,skc,is_seqlens_k_cumulative=True)
    torch.npu.synchronize(); print("[REPRO] v4 : OK")
except Exception:
    print("[REPRO] v4 : FAILED")
    print("[REPRO] BUG REPRODUCED (errcode 11003 get kernel failed)")
PYEOF
)

log "3/3 设备侧软链信息（折叠证据）"
TEST_PID=$(cat /tmp/repro_pid 2>/dev/null || echo "")
DEVLOG=""
for _ in 1 2 3 4 5 6 7 8; do
  for root in "$HOME/ascend/log" /root/ascend/log /var/log/ascend; do
    DEVLOG=$(find "$root" -name "device-${TEST_PID}_*.log" -type f 2>/dev/null | tail -1)
    [ -n "$DEVLOG" ] && break
  done
  if [ -n "$DEVLOG" ] && grep -aq CreateSoftLinkToSoFile "$DEVLOG" 2>/dev/null; then break; fi
  sleep 2   # 设备日志落盘有延迟
done
if [ -z "$DEVLOG" ]; then
  echo "(device log 未找到: pid=${TEST_PID}; 手动查 ~/ascend/log/run/device-${TEST_PID}_*.log)"
  exit 0
fi
echo "device log: $DEVLOG"
grep -am1 -aoE "socVersion is [A-Za-z0-9]+|version is [0-9]+" "$DEVLOG" || true
grep -a "CreateSoftLinkToSoFile" "$DEVLOG" \
  | sed -e 's/.*CreateSoftLinkToSoFile\] Create soft link so success, //' \
        -e 's/, targetPath=/  ->  /' || true
# 自动判读: 两条软链的 target 是否相同
mapfile -t TARGETS < <(grep -a "CreateSoftLinkToSoFile" "$DEVLOG" | grep -oE "targetPath=[^ ]+" | sort -u || true)
if [ "${#TARGETS[@]}" -eq 1 ]; then
  echo ""
  echo ">>> VERDICT: 两个 soName 指向同一 store 文件 —— 折叠复现 (BUG)"
elif [ "${#TARGETS[@]}" -ge 2 ]; then
  echo ""
  echo ">>> VERDICT: 两个 soName 指向不同 store 文件 —— 环境健康"
fi
