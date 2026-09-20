#!/usr/bin/env bash
set -euo pipefail
# Resolve to an ABSOLUTE path NOW, while cwd is still the invocation dir:
# later we cd /tmp for the cross-launch test and relative paths would break.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${CI_TEST_LOG_DIR:-/tmp/ci_test_logs}"
mkdir -p "$LOG_DIR"
log() { printf '[CI-test] %s\n' "$*"; }
log "driver: $(head -1 /usr/local/Ascend/driver/version.info)"
python3 -c "import torch,torch_npu; assert torch_npu.npu.device_count()>=1; print('NPU OK')" || exit 1

# ---- FAST COLLECTION MODE -----------------------------------------------
# Repro + P1-P5 behavioral matrix: DONE (results in prior runs, summarized
# in the Huawei ticket). Old full flow kept below the P6 section, disabled.
# This run only needs: build -> P6 ftrace live capture -> driver binaries.

log "=== building twin v3/v4 (single all-pass) ==="
FLASH_ATTN_BUILD_VERSION=all FLASH_ATTN_BUILD_NPU=910 python3 setup.py install > "$LOG_DIR/build_all.log" 2>&1
log "twin build done"

# ---- P6 ftrace: live capture of the driver submission path ----
if [ -f "$REPO_DIR/ci/ftrace_probe.sh" ]; then
  log "=== P6 ftrace driver-path injection (uprobe+kprobe) ==="
  bash "$REPO_DIR/ci/ftrace_probe.sh" 2>&1 | tee "$LOG_DIR/ftrace.log" || true
  cp -r "$REPO_DIR/tmp/ftrace_out" "$LOG_DIR/ftrace_out" 2>/dev/null || true
  chmod -R a+rX "$LOG_DIR/ftrace_out" 2>/dev/null || true
else
  log "WARN: ftrace_probe.sh missing"
fi

# ---- driver-binary collection for offline 25.5.0 vs 25.5.2 diff ----
log "=== driver-binary collection ==="
mkdir -p "$LOG_DIR/drv_bin"
for f in /usr/local/Ascend/driver/lib64/driver/libascend_hal.so \
         /lib/modules/"$(uname -r)"/updates/drv_devmm_host.ko \
         /lib/modules/"$(uname -r)"/updates/ascend_queue.ko \
         /usr/local/Ascend/driver/version.info; do
  cp "$f" "$LOG_DIR/drv_bin/" 2>/dev/null && log "collected: $(basename "$f")"
done || true
chmod -R a+rX "$LOG_DIR/drv_bin" 2>/dev/null || true

# ============================================================================
# DISABLED: original full repro + forensics + P1-P5 (kept for reference;
# re-enable by deleting the leading colon on the OFF block)
# ============================================================================
if false; then OFF=1
log "=== cross-launch test ==="
cd /tmp
export ASCEND_GLOBAL_LOG_LEVEL=0
if ! command -v strace >/dev/null 2>&1; then
  (apt-get update >/dev/null 2>&1 && apt-get install -y strace >/dev/null 2>&1) || true
fi
STRACE_PREFIX=""
command -v strace >/dev/null 2>&1 && STRACE_PREFIX="strace -f -s 64 -e trace=ioctl,write,writev,symlink,symlinkat,link,linkat -o $LOG_DIR/strace.log"
$STRACE_PREFIX python3 << 'PYEOF' > "$LOG_DIR/repro_output.log" 2>&1 || true
import os, torch, torch_npu
import flash_attn_npu_3 as fa3
import flash_attn_npu_4 as fa4
with open("/tmp/repro_pid","w") as f: f.write(str(os.getpid()))
skc = torch.tensor([0,128,256], dtype=torch.int32, device="npu")
try:
    fa3.get_scheduler_metadata(2,256,256,8,8,128,None,skc,is_seqlens_k_cumulative=True)
    torch.npu.synchronize(); print("[REPRO] v3 : OK")
except Exception as e: print(f"[REPRO] v3 : FAILED -- {e}")
try:
    fa4.get_scheduler_metadata(2,256,256,8,8,128,None,skc,is_seqlens_k_cumulative=True)
    torch.npu.synchronize(); print("[REPRO] v4 : OK"); print("[REPRO] PASS")
except Exception as e:
    print(f"[REPRO] v4 : FAILED"); print(f"[REPRO]   {e}"); print("[REPRO] BUG REPRODUCED")
PYEOF
cat "$LOG_DIR/repro_output.log"

TEST_PID=$(cat /tmp/repro_pid 2>/dev/null || echo "")
if [ -n "$TEST_PID" ]; then
  DEVLOG=$(find /root/ascend/log/run -name "device-${TEST_PID}_*.log" -type f 2>/dev/null | tail -1)
  if [ -z "$DEVLOG" ]; then
    DEVLOG=$(find /root/ascend/log -name "device-${TEST_PID}_*.log" -type f 2>/dev/null | tail -1)
  fi
  if [ -n "$DEVLOG" ]; then
    cp "$DEVLOG" "$LOG_DIR/device_full.log"; chmod 644 "$LOG_DIR/device_full.log" 2>/dev/null || true
    log "--- deployment ---"
    grep -a "WriteBufToSoFile\|CreateSoftLinkToSoFile" "$DEVLOG" | sed -e 's/.*\[tid:[0-9]*\] //' -e 's/^\[INFO\] CCECPU([0-9]*,[^)]*):[0-9:.\-]* //' | tee -a "$LOG_DIR/forensics.log"
    log "--- API lookups ---"
    grep -a "GetApi\|Get api" "$DEVLOG" | sed -e 's/.*\[tid:[0-9]*\] //' -e 's/^\[INFO\] CCECPU([0-9]*,[^)]*):[0-9:.\-]* //' | tee -a "$LOG_DIR/forensics.log"
    if [ -f "$LOG_DIR/strace.log" ]; then
      log "--- strace summary ---"
      echo "syscalls traced: $(wc -l < "$LOG_DIR/strace.log"); write() of 22864B: $(grep -ac ', 22864)' "$LOG_DIR/strace.log" || true)" | tee -a "$LOG_DIR/forensics.log"
    fi
    PLOG=$(find /root/ascend/log/debug/plog -name "plog-${TEST_PID}_*.log" -type f 2>/dev/null | tail -1)
    if [ -z "$PLOG" ]; then
      PLOG=$(find /root/ascend/log -name "plog-${TEST_PID}_*.log" -type f 2>/dev/null | tail -1)
    fi
    if [ -n "$PLOG" ] && [ -f "$PLOG" ]; then
      cp "$PLOG" "$LOG_DIR/plog_full.log"; chmod 644 "$LOG_DIR/plog_full.log" 2>/dev/null || true
      log "--- host transmission chains (healthy=2, fold=1/truncated) ---"
      grep -aE "aclrtBinaryLoadFromDataImpl|SetCpuBinInfo|halMemAllocInner.*22864|_drvMemcpyInner.*22864" "$PLOG" | \
        sed -e 's/^\[DEBUG\] //' -e 's/^\[INFO\] //' | cut -c1-190 | tee -a "$LOG_DIR/forensics.log"
      echo "_drvMemcpyInner ByteCount=22864 count: $(grep -ac '_drvMemcpyInner.*ByteCount=22864' "$PLOG" || true)" | tee -a "$LOG_DIR/forensics.log"
    fi
    log "both soNames -> SAME store target = FOLD (bug); different targets = healthy"
  fi
fi
if [ -f "$REPO_DIR/ci/probe.sh" ]; then
  log "=== behavioral probes P1-P5 (DISABLED in fast mode) ==="
  bash "$REPO_DIR/ci/probe.sh" 2>&1 | tee "$LOG_DIR/probes.log" || true
fi
fi # END DISABLED BLOCK
log "done (fast collection mode)"
