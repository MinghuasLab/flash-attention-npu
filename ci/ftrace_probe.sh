#!/bin/bash
# ============================================================
# P6: ftrace dynamic-probe injection into the driver's
# cust-so submission path. No binary patching, no attach:
# uprobe (userspace libascend_hal.so) + kprobe (drv_devmm_host.ko)
# watch every pool get/put, svm ioctl and kernel DMA submit
# while the twin-kernel repro runs.
# Requires: privileged container, root, tracefs mounted.
# ============================================================
set -u
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO_DIR/tmp/ftrace_out"; mkdir -p "$OUT"
HAL=/usr/local/Ascend/driver/lib64/driver/libascend_hal.so
TR=/sys/kernel/tracing

have_tracefs() { [ -d "$TR/events" ] || mount -t tracefs tracefs "$TR" 2>/dev/null; [ -d "$TR/events" ]; }

if ! have_tracefs; then
  echo "[P6] tracefs unavailable -- skip (need privileged + root)"
  exit 0
fi
cd "$TR"
echo 0 > tracing_on
echo > uprobe_events 2>/dev/null
echo > kprobe_events 2>/dev/null
echo > trace

# ---- uprobe on host-side submission (offsets from disassembly) ----
# devmm_host_mem_pool_get(devid w0, size x1, out x2) @0x109928
# devmm_host_mem_pool_put                          @0x109bf0 (verify below)
PUT_OFF=$(readelf -Ws $HAL | awk '/devmm_host_mem_pool_put/{print strtonum("0x"$2); exit}')
[ -n "$PUT_OFF" ] && [ "$PUT_OFF" != "0" ] || PUT_OFF=0x109bf0
cat >> uprobe_events <<EOF
p:pool_get $HAL:0x109928 devid=%x0:u32 size=%x1:u64
r:pool_get_ret $HAL:0x109928 blkptr=%x0
p:svm_ioctl $HAL:0xd8630 cmd=%x1:u64
EOF
# ---- kprobe on kernel-side DMA submission ----
for s in devmm_ioctl_memcpy_proc devmm_memcpy_set_buf_info devmm_alloc_copy_res devmm_ioctl_async_memcpy_proc; do
  if grep -qw "$s" /proc/kallsyms 2>/dev/null; then
    echo "p:k_$s $s" >> kprobe_events
  fi
done
echo "uprobe/kprobe events registered:"; cat uprobe_events kprobe_events | sed 's/^/    /'

echo 1 > events/uprobes/enable 2>/dev/null
echo 1 > events/kprobes/enable 2>/dev/null
echo 1 > tracing_on

# ---- run the twin-kernel repro (the traced workload) ----
echo "[P6] running twin cross-launch under probes..."
python3 - << 'PYEOF' > "$OUT/repro_under_trace.log" 2>&1
import os, torch, torch_npu
try:
    import flash_attn_npu_3 as fa3, flash_attn_npu_4 as fa4
except ImportError:
    import importlib, glob, sys
    b = glob.glob("/workspace/flash-attention-npu/build/lib*/")
    if b: sys.path.insert(0, b[0])
    import flash_attn_npu_3 as fa3, flash_attn_npu_4 as fa4
print(f"[P6] pid={os.getpid()}")
skc = torch.tensor([0,128,256], dtype=torch.int32, device="npu")
def call(tag, fn):
    try: fn(); torch.npu.synchronize(); print(f"[P6] {tag}: OK")
    except Exception as e: print(f"[P6] {tag}: FAILED -- {type(e).__name__}")
call("v3", lambda: fa3.get_scheduler_metadata(2,256,256,8,8,128,None,skc,is_seqlens_k_cumulative=True))
call("v4", lambda: fa4.get_scheduler_metadata(2,256,256,8,8,128,None,skc,is_seqlens_k_cumulative=True))
PYEOF
cat "$OUT/repro_under_trace.log"

echo 0 > tracing_on
cp trace "$OUT/trace.raw"
chmod -R a+rwX "$OUT" 2>/dev/null || true   # tracefs files copy as root-600; runner user must read them
echo "[P6] raw trace events: $(wc -l < trace)"

# ---- analysis: the submission sequence during the two loads ----
echo "--- pool_get + returned block ptr (interleaved) ---"
grep -a "pool_get:\|pool_get_ret:" trace | tail -30 | sed 's/^[^:]*-[^ ]* *//' | cut -c1-160 | tee "$OUT/pool_get.txt"
echo "--- per-load kernel submission groups ---"
grep -a "k_devmm_ioctl_memcpy_proc:" trace | wc -l
echo "[P6] verdict key: two 'pool_get ... size=22864' with SAME blkptr => host-pool reuse confirmed; zero pool_get => kernel-side (alloc_copy_res/set_buf_info/dma chain), dump trace.raw for Huawei"
