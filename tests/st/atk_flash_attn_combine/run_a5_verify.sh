#!/usr/bin/env bash
# One-click A5 (Ascend 950) verification for flash_attn_combine.
#
#   bash run_a5_verify.sh              # env check + build v3/v4 + pytest
#   bash run_a5_verify.sh --no-build   # skip compilation (already built)
#   bash run_a5_verify.sh --atk        # additionally run ATK single-benchmark
#
# Layout A (delivery package): ./flash-attention-npu + ./atk next to this script
# Layout B (inside the repo):  this script lives in tests/st/atk_flash_attn_combine
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$HERE/flash-attention-npu" ]; then
  PKG_ROOT="$HERE"
  REPO="$HERE/flash-attention-npu"
  ATK_DIR="$HERE/atk"
else
  PKG_ROOT="$(cd "$HERE/../.." && pwd)"
  REPO="$PKG_ROOT"
  ATK_DIR="$HERE"
fi

BUILD=1
RUN_ATK=0
for arg in "$@"; do
  case "$arg" in
    --no-build) BUILD=0 ;;
    --atk) RUN_ATK=1 ;;
    -h|--help)
      sed -n '2,12p' "${BASH_SOURCE[0]}"
      exit 0 ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

FAIL=0
step() { echo; echo "===== $* ====="; }
check() { if [ "$1" -eq 0 ]; then echo "[PASS] $2"; else echo "[FAIL] $2"; FAIL=1; fi; }

step "0. layout"
echo "package root : $PKG_ROOT"
echo "repo         : $REPO"
echo "atk assets   : $ATK_DIR"

step "1. python / torch_npu / device"
python - <<'PY'
import sys
print("python", sys.version.split()[0])
try:
    import torch, torch_npu
    print("torch", torch.__version__, "| torch_npu", torch_npu.__version__)
    name = torch_npu.npu.get_device_name()
    print("device:", name)
    print("DEVICE_OK" if "Ascend950" in name else "DEVICE_NOT_950")
except Exception as exc:  # noqa: BLE001
    print("ENV_ERROR:", exc)
    print("DEVICE_NOT_950")
PY
DEV_OUT=$(python -c "import torch_npu;print(torch_npu.npu.get_device_name())" 2>/dev/null)
case "$DEV_OUT" in
  *Ascend950*) check 0 "device is Ascend950" ;;
  *) check 1 "device is NOT Ascend950 (got: ${DEV_OUT:-none})" ;;
esac

if [ "$BUILD" -eq 1 ]; then
  step "2. build 950 (v3 + v4)"
  cd "$REPO" || exit 2
  for ver in v3 v4; do
    echo "--- build $ver ---"
    FLASH_ATTN_BUILD_NPU=950 FLASH_ATTN_BUILD_VERSION=$ver \
      python setup.py build_ext --inplace 2>&1 | tail -3
    check "${PIPESTATUS[0]}" "build $ver"
  done
else
  step "2. build skipped (--no-build)"
fi

step "3. artifacts"
ls -l "$REPO"/flash_attn_npu_3_950*.so "$REPO"/flash_attn_npu_4_950*.so 2>/dev/null \
  || echo "(no 950 .so found; expected flash_attn_npu_{3,4}_950.cpython-*.so in repo root)"

step "4. import + smoke"
cd "$REPO" || exit 2
export FLASH_ATTN_NPU_ROOT="$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
python - <<'PY'
import torch, torch_npu
import flash_attn_npu_3, flash_attn_npu_4
ns, b, s, h, d = 4, 1, 32, 2, 128
op = torch.randn(ns, b, s, h, d)
lse = torch.rand(ns, b, s, h) * -5
lse_c = lse.transpose(2, 3).contiguous().transpose(2, 3)
for mod in (flash_attn_npu_3, flash_attn_npu_4):
    out, l = mod.flash_attn_combine(op.npu(), lse_c.npu(), out_dtype=torch.float16)
    torch.npu.synchronize()
    print(mod.__name__, tuple(out.shape), out.dtype, tuple(l.shape), l.dtype)
    assert tuple(out.shape) == (b, s, h, d) and out.dtype == torch.float16
    assert tuple(l.shape) == (b, s, h) and l.dtype == torch.float32
print("SMOKE_OK")
PY
check $? "smoke (FA3 + FA4)"

step "5. pytest combine"
python -m pytest tests/test_flash_attn_npu_v3_combine.py \
                 tests/test_flash_attn_npu_v4_combine.py -q 2>&1 | tail -5
check "${PIPESTATUS[0]}" "pytest combine (expect 127 passed)"

if [ "$RUN_ATK" -eq 1 ]; then
  step "6. ATK single-benchmark (npu vs cpu)"
  cd "$ATK_DIR" || exit 2
  PYTHONPATH="$REPO:${PYTHONPATH:-}" atk pytorch ./all_test_flash_attn_combine_bf16fp16.json \
    -p ./execute_flash_attn_combine.py --task accuracy --devices 0 2>&1 | tail -8
  check "${PIPESTATUS[0]}" "ATK FA3 200 cases"
  PYTHONPATH="$REPO:${PYTHONPATH:-}" atk pytorch ./all_test_flash_attn_combine_v4_bf16fp16.json \
    -p ./execute_flash_attn_combine.py --task accuracy --devices 0 2>&1 | tail -8
  check "${PIPESTATUS[0]}" "ATK FA4 200 cases"
fi

step "summary"
if [ "$FAIL" -eq 0 ]; then
  echo "ALL CHECKS PASSED"
else
  echo "SOME CHECKS FAILED - see [FAIL] lines above and README.zh.md"
fi
exit "$FAIL"
