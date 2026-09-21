#!/usr/bin/env bash
set -euo pipefail
# Build the two AICPU payloads and extract .aicpu_binary sections.
# Asserts both payloads have EXACTLY equal byte size (the bug's trigger).
cd "$(dirname "$0")"

TOOLKIT="${ASCEND_HOME_PATH:-}"
[ -z "$TOOLKIT" ] && for c in /usr/local/Ascend/cann-* /usr/local/Ascend/ascend-toolkit/latest; do [ -d "$c/aarch64-linux" ] && TOOLKIT="$c" && break; done
ARCH_DIR="$TOOLKIT/aarch64-linux"
AICPU_INC="$ARCH_DIR/asc/include/aicpu_api"
AICPU_LIB="$ARCH_DIR/lib64/device/lib64"
HCC="$TOOLKIT/toolkit/toolchain/hcc"
[ -d "$HCC" ] || HCC=/usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc

build_one() {  # $1=src $2=out.o
    bisheng -O2 -std=c++17 -fvisibility=default -fvisibility-inlines-hidden \
        -D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=2 -D_GNU_SOURCE \
        -I"$AICPU_INC" -I"$(dirname "$1")" \
        --cce-aicpu-L"$AICPU_LIB" --cce-aicpu-laicpu_api \
        --cce-aicpu-toolkit-path="$HCC/bin" --cce-aicpu-sysroot="$HCC/sysroot" \
        -isystem "$HCC/aarch64-target-linux-gnu/include" \
        -isystem "$HCC/aarch64-target-linux-gnu/include/c++/7.3.0" \
        -isystem "$HCC/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu" \
        -isystem "$HCC/aarch64-target-linux-gnu/include/c++/7.3.0/backward" \
        -c -o "$2" -x aicpu "$1"
}

echo "[build] kernel A"
build_one kernels/min_repro_a.aicpu aicpu_a.o
echo "[build] kernel B"
build_one kernels/min_repro_b.aicpu aicpu_b.o

echo "[extract] .aicpu_binary sections"
objcopy -O binary --only-section=.aicpu_binary aicpu_a.o min_repro_a.bin
objcopy -O binary --only-section=.aicpu_binary aicpu_b.o min_repro_b.bin

SA=$(stat -c%s min_repro_a.bin); SB=$(stat -c%s min_repro_b.bin)
echo "payload A = $SA bytes, payload B = $SB bytes"
if [ "$SA" != "$SB" ]; then
    echo "ERROR: payload sizes differ ($SA vs $SB) — trigger condition not met, do NOT proceed"
    exit 1
fi
echo "[ok] equal payload size ($SA) — trigger condition satisfied"
