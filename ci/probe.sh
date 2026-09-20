#!/bin/bash
# ============================================================
# Store-fold behavioral probes -- run on the BUGGY driver (25.5.0).
# Characterizes the fold key of the device-side deploy path
# (batchLoadsoFrombuf -> aicpusd_cust_so_manager):
#   P1 order-swap  : v4 first, then v3   -> is the fold positional?
#   P2 relaunch    : v3, v4, v3 again    -> does kernel #1 keep working
#                                           after #2 was folded?
#   P3 interleave  : v3, v2(different payload size), v4
#                                       -> does v4 fold onto the SAME-SIZE
#                                          v3 file, or onto the most
#                                          recent (v2) file?
#   P4 sleep       : v3, sleep 2s, v4   -> is it a race between deploys?
# Each probe runs in its own process; the device log for that pid is
# then harvested (Write/Link/GetApi lines show which store file each
# soName ended up on).
# Prereq: v3/v4 twin packages installed (ci/run_ci_test.sh or repro.sh).
# ============================================================
set -u
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_DIR/probe_logs"; mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

python3 -c "import flash_attn_npu_3, flash_attn_npu_4" 2>/dev/null || {
  echo "(twin packages missing -- building)"
  FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py install > "$LOG_DIR/build_v3.log" 2>&1
  FLASH_ATTN_BUILD_VERSION=v4 FLASH_ATTN_BUILD_NPU=910 python3 setup.py install > "$LOG_DIR/build_v4.log" 2>&1
}
python3 -c "import flash_attn_npu" 2>/dev/null || \
  FLASH_ATTN_BUILD_VERSION=v2 FLASH_ATTN_BUILD_NPU=910 python3 setup.py install > "$LOG_DIR/build_v2.log" 2>&1

echo "=== payload sizes (.aicpu_binary bytes) ==="
find build -name "fa_metadata.o" | while read -r o; do
  printf "%-60s " "${o#*/build/}"; objdump -h "$o" | awk '/aicpu_binary/{print $3}'
done

export ASCEND_GLOBAL_LOG_LEVEL=0

harvest() {
  local name="$1"
  local pid dev
  pid=$(cat "/tmp/probe_${name}_pid" 2>/dev/null)
  dev=$(find /root/ascend/log/run -name "device-${pid}_*.log" -type f 2>/dev/null | tail -1)
  if [ -n "$dev" ] && [ -f "$dev" ]; then
    cp "$dev" "$LOG_DIR/${name}_device.log"
    echo "-- device deploy sequence:"
    grep -a "WriteBufToSoFile\|CreateSoftLinkToSoFile\|Get api\|GetApi" "$dev" | \
      sed -e 's/.*\[tid:[0-9]*\] //' | cut -c1-190
  else
    echo "-- (device log not found for pid=$pid)"
  fi
}

run_probe() {
  local name="$1" body="$2"
  echo ""
  echo "########## PROBE $name ##########"
  cat > "/tmp/probe_${name}.py" <<PYEOF
import os
with open("/tmp/probe_${name}_pid", "w") as f: f.write(str(os.getpid()))
import torch, torch_npu
$body
PYEOF
  (cd /tmp && python3 "/tmp/probe_${name}.py" 2>&1 | grep -a "\[PROBE\]")
  harvest "$name"
}

HDR=$(cat <<'EOF'
skc = torch.tensor([0, 128, 256], dtype=torch.int32, device="npu")
def call(tag, fn):
    try:
        fn(); torch.npu.synchronize(); print(f"[PROBE] {tag} : OK")
    except Exception as e:
        print(f"[PROBE] {tag} : FAILED -- {type(e).__name__}")
import flash_attn_npu_3 as fa3
import flash_attn_npu_4 as fa4
def v3(): return fa3.get_scheduler_metadata(2,256,256,8,8,128,None,skc,is_seqlens_k_cumulative=True)
def v4(): return fa4.get_scheduler_metadata(2,256,256,8,8,128,None,skc,is_seqlens_k_cumulative=True)
EOF
)

run_probe P1_order_swap "$HDR
call('v4(#1)', v4)
call('v3(#2)', v3)"

run_probe P2_relaunch "$HDR
call('v3(#1)', v3)
call('v4(#2)', v4)
call('v3(#3-again)', v3)"

run_probe P3_interleave "
import flash_attn_npu as fa2
$HDR
def v2(): return fa2.get_scheduler_metadata(2,256,256,8,8,128,skc)
call('v3(#1,sz-A)', v3)
call('v2(#2,sz-B?)', v2)
call('v4(#3,sz-A)', v4)"

run_probe P4_sleep "$HDR
call('v3(#1)', v3)
import time; time.sleep(2.0)
call('v4(#2-after-2s)', v4)"

run_probe P5_copy_loop "
import struct, importlib.util, os
def expected(pkg):
    spec = importlib.util.find_spec(pkg)
    o = os.path.join(os.path.dirname(spec.origin), 'fa_metadata.o')
    blob = open(o, 'rb').read()
    e_shoff, = struct.unpack_from('<Q', blob, 0x28)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from('<HHH', blob, 0x3a)
    shstr_off, = struct.unpack_from('<Q', blob, e_shoff + e_shstrndx*e_shentsize + 0x18)
    for k in range(e_shnum):
        base = e_shoff + k*e_shentsize
        name_off, _, _, _, off, size = struct.unpack_from('<IIQQQQ', blob, base)
        end = blob.index(b'\0', shstr_off+name_off)
        if blob[shstr_off+name_off:end] == b'.aicpu_binary':
            return blob[off:off+size]
    raise RuntimeError('no .aicpu_binary in ' + o)
p3, p4 = expected('flash_attn_npu_3'), expected('flash_attn_npu_4')
assert len(p3) == len(p4) and p3 != p4
h3 = torch.frombuffer(bytearray(p3), dtype=torch.uint8)
h4 = torch.frombuffer(bytearray(p4), dtype=torch.uint8)
stale = 0
N = 300
for i in range(N):
    d3 = h3.to('npu'); d4 = h4.to('npu')
    del d3, d4
    torch.npu.empty_cache()
    d3 = h3.to('npu'); d4 = h4.to('npu')
    b3, b4 = d3.cpu(), d4.cpu()
    if not torch.equal(b4, h4) and torch.equal(b4, h3):
        stale += 1; print(f'[PROBE] P5 iter {i}: SECOND BUFFER CAME BACK AS v3 CONTENT (STALE)')
    elif not torch.equal(b3, h3) and torch.equal(b3, h4):
        stale += 1; print(f'[PROBE] P5 iter {i}: FIRST BUFFER CAME BACK AS v4 CONTENT (CROSSED)')
    del d3, d4, b3, b4
    torch.npu.empty_cache()
print(f'[PROBE] P5 copy-loop: {N} iters stale={stale} ' + ('-> DRIVER COPY PATH BUG REPRODUCED IN ISOLATION' if stale else '-> driver copy path clean; bug is specific to cust-so deploy path'))"

echo ""
echo "=== done. artifacts in $LOG_DIR/ ==="
