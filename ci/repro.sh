#!/bin/bash
# ============================================================
# CANN 9.1.0-beta.1 aicpu store fold bug -- minimal reproducer
# with full device-log forensics
#
# Bug: when a single process launches TWO custom aicpu kernels whose
# .aicpu_binary payloads have IDENTICAL byte sizes, the second kernel's
# bytes are never written to the device-side store. The scheduler folds
# both soNames onto the first kernel's file -> every launch of the
# second kernel fails with errcode:11003 "get kernel failed".
#
# Environment: driver 25.5.0 (910B) -- does NOT reproduce on 25.5.2
# ============================================================
set -e
cd "$(dirname "$0")"

LOG_DIR="./repro_logs"
mkdir -p "$LOG_DIR"

echo "=== driver: $(head -1 /usr/local/Ascend/driver/version.info)"
echo "=== CANN:  $(ls /usr/local/Ascend/ | grep cann | head -1)"

if [ "$1" != "--run" ]; then
  echo ""
  echo "=== [1/3] building twin kernels (v3=22864B, v4=22864B) ==="
  FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 \
    python3 setup.py install > "$LOG_DIR/build_v3.log" 2>&1
  echo "    v3 built OK"
  FLASH_ATTN_BUILD_VERSION=v4 FLASH_ATTN_BUILD_NPU=910 \
    python3 setup.py install > "$LOG_DIR/build_v4.log" 2>&1
  echo "    v4 built OK"
fi

echo ""
echo "=== [2/3] cross-launch test (v3 then v4 in ONE process) ==="
export ASCEND_GLOBAL_LOG_LEVEL=0

cd /tmp  # avoid importing from source dir
TEST_PID_FILE="/tmp/repro_test_pid"

if ! command -v strace >/dev/null 2>&1; then
    # CI image ships without strace; best-effort install (verified working
    # inside fa-npu-ci:910b-cann9.1-torch2.9)
    (apt-get update >/dev/null 2>&1 && apt-get install -y strace >/dev/null 2>&1) || true
fi

RUN_PY="python3"
if command -v strace >/dev/null 2>&1; then
    # Host-side syscall trace: the payload travels via driver ioctl + shared
    # memory (NO write() of 22864B exists -- verified), so strace cannot show
    # the bytes; its value is proving that negative + leaving a raw record.
    RUN_PY="strace -f -s 64 -e trace=ioctl,write,writev,symlink,symlinkat,link,linkat -o $OLDPWD/$LOG_DIR/strace.log python3"
    echo "    (strace attached -> $LOG_DIR/strace.log)"
else
    echo "    (WARN: strace unavailable -- syscall trace skipped)"
fi

$RUN_PY << 'PYEOF' > /tmp/repro_test_output.log 2>&1 || true &
import os, sys, time
import torch
import torch_npu
import flash_attn_npu_3 as fa3
import flash_attn_npu_4 as fa4

# Record PID for device-log correlation
with open(os.environ.get("TEST_PID_FILE", "/tmp/repro_test_pid"), "w") as f:
    f.write(str(os.getpid()))

print(f"[REPRO] process pid: {os.getpid()}")
print(f"[REPRO] NPU devices: {torch_npu.npu.device_count()}")

skc = torch.tensor([0, 128, 256], dtype=torch.int32, device="npu")

# --- kernel #1: v3 (first custom aicpu kernel in this process) ---
try:
    fa3.get_scheduler_metadata(2, 256, 256, 8, 8, 128, None, skc,
                               is_seqlens_k_cumulative=True)
    torch.npu.synchronize()
    print("[REPRO] v3 (kernel #1) : OK")
except Exception as e:
    print(f"[REPRO] v3 (kernel #1) : FAILED -- {e}")

# --- kernel #2: v4 (second, SAME payload size as v3) ---
try:
    fa4.get_scheduler_metadata(2, 256, 256, 8, 8, 128, None, skc,
                               is_seqlens_k_cumulative=True)
    torch.npu.synchronize()
    print("[REPRO] v4 (kernel #2) : OK")
    print("[REPRO] PASS -- no fold (this driver does NOT have the bug)")
except Exception as e:
    print(f"[REPRO] v4 (kernel #2) : FAILED")
    print(f"[REPRO]   {e}")
    print("[REPRO] BUG REPRODUCED -- second same-size kernel folded onto first")
PYEOF
TEST_RC=$?
wait

cat /tmp/repro_test_output.log

echo ""
echo "=== [3/3] device-log forensics ==="
TEST_PID=$(cat "$TEST_PID_FILE" 2>/dev/null || echo "")
if [ -z "$TEST_PID" ]; then
    echo "(WARN: test PID not captured)"
else
    echo "--- Full device log for hostpid=$TEST_PID ---"
    # Evidence lines (CreateSoftLink/WriteBuf/GetApi) live in the run/ log;
    # the debug/ copy of the same pid has none -- prefer run/, fall back to
    # a whole-tree search for environments that only produce one file.
    DEVLOG=$(find /root/ascend/log/run -name "device-${TEST_PID}_*.log" -type f 2>/dev/null | tail -1)
    if [ -z "$DEVLOG" ]; then
        DEVLOG=$(find /root/ascend/log -name "device-${TEST_PID}_*.log" -type f 2>/dev/null | tail -1)
    fi
    if [ -n "$DEVLOG" ] && [ -f "$DEVLOG" ]; then
        cp "$DEVLOG" "$OLDPWD/$LOG_DIR/device_full.log"
        echo "    captured: $DEVLOG -> $LOG_DIR/device_full.log"

        echo ""
        echo "--- KEY EVIDENCE: deployment operations ---"
        grep -a "WriteBufToSoFile\|CreateSoftLinkToSoFile" "$DEVLOG" | \
          sed -e 's/.*\[tid:[0-9]*\] //' -e 's/^\[INFO\] CCECPU([0-9]*,[^)]*):[0-9:.\-]* //' | \
          sed 's|/home/CustAiCpuUser/cust_aicpu_[0-9_]*/|soName=|' | \
          sed 's|, targetPath=/home/CustAiCpuUser/lib/|-> store=|' | \
          sed 's|$|.|'

        echo ""
        echo "--- KEY EVIDENCE: kernel API lookups ---"
        grep -a "GetApi\|Get api" "$DEVLOG" | \
          sed -e 's/.*\[tid:[0-9]*\] //' -e 's/^\[INFO\] CCECPU([0-9]*,[^)]*):[0-9:.\-]* //' | sed 's/$/./'

        if [ -f "$OLDPWD/$LOG_DIR/strace.log" ]; then
            echo ""
            echo "--- HOST-SIDE (strace): syscall summary ---"
            echo "    syscalls traced: $(wc -l < "$OLDPWD/$LOG_DIR/strace.log")"
            echo "    write() calls of 22864 bytes: $(grep -ac ', 22864)' "$OLDPWD/$LOG_DIR/strace.log" || true)"
            echo "    (0 is EXPECTED on all drivers: the payload travels through"
            echo "     driver ioctl + shared memory, not write(); raw log kept"
            echo "     in $LOG_DIR/strace.log for reference)"
        fi

        echo ""
        echo "--- KEY EVIDENCE: host-side transmission chain (plog) ---"
        PLOG=$(find /root/ascend/log/debug/plog -name "plog-${TEST_PID}_*.log" -type f 2>/dev/null | tail -1)
        if [ -z "$PLOG" ]; then
            PLOG=$(find /root/ascend/log -name "plog-${TEST_PID}_*.log" -type f 2>/dev/null | tail -1)
        fi
        if [ -n "$PLOG" ] && [ -f "$PLOG" ]; then
            cp "$PLOG" "$OLDPWD/$LOG_DIR/plog_full.log"
            echo "    captured: $PLOG -> $LOG_DIR/plog_full.log"
            echo "    (per-kernel host chain: aclrtBinaryLoadFromData(22864) ->"
            echo "     SetCpuBinInfo(soName) -> DevMemAlloc(22864) ->"
            echo "     _drvMemcpyInner(ByteCount=22864))"
            echo ""
            grep -aE "aclrtBinaryLoadFromDataImpl|SetCpuBinInfo|halMemAllocInner.*22864|_drvMemcpyInner.*22864" "$PLOG" | \
              sed -e 's/^\[DEBUG\] //' -e 's/^\[INFO\] //' | cut -c1-190
            echo ""
            echo "    _drvMemcpyInner ByteCount=22864 count: $(grep -ac '_drvMemcpyInner.*ByteCount=22864' "$PLOG" || true)"
            echo "    (healthy driver: 2, one full chain per kernel. Fold case:"
            echo "     1 -- or a truncated second chain -- showing exactly where"
            echo "     the host transmission path drops the second payload)"
        else
            echo "    (host plog not found for pid=$TEST_PID)"
        fi

        echo ""
        echo "--- INTERPRETATION ---"
        echo "  Decisive checks:"
        echo "   1. device log, CreateSoftLink lines: healthy = two soNames ->"
        echo "      two DIFFERENT store files; folded = both -> the SAME file."
        echo "   2. host plog, transmission chains: healthy = TWO complete"
        echo "      BinaryLoad->Alloc->Memcpy(22864) chains; folded = the second"
        echo "      chain truncated/absent -> bytes never copied to the device."
        echo "   3. symptom: folded => second kernel errcode 11003"
        echo "      'get kernel failed' (its GetApi never resolves)."
        echo "  Note: WriteBufToSoFile lines show up on driver 25.5.0 (fold"
        echo "  case: only ONE write for two kernels) but may be absent on"
        echo "  fixed drivers; judge by 1+2, not the writes."
    else
        echo "    (device log not found for pid=$TEST_PID)"
        echo "    Searching recent logs..."
        find /root/ascend/log -name "device-*.log" -mmin -5 2>/dev/null | head -5
    fi
fi

echo ""
echo "=== done. artifacts in $LOG_DIR/ ==="
