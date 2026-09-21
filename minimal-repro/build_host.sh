#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
TOOLKIT="${ASCEND_HOME_PATH:-}"
[ -z "$TOOLKIT" ] && for c in /usr/local/Ascend/cann-* /usr/local/Ascend/ascend-toolkit/latest; do [ -d "$c/aarch64-linux" ] && TOOLKIT="$c" && break; done
ACL_INC="$TOOLKIT/aarch64-linux/include"
ACL_LIB="$TOOLKIT/aarch64-linux/lib64"

for kv in "A:aicpu_a.o:liblaunch_a.so" "B:aicpu_b.o:liblaunch_b.so"; do
    K=${kv%%:*}; rest=${kv#*:}; OBJ=${rest%%:*}; SO=${rest##*:}
    # compile glue (host asc mode, compile-only)
    bisheng -O2 -x asc --npu-arch=dav-2201 --cce-auto-infer-kernel-type=false \
        -fPIC -std=c++17 -DKERNEL_SYM=MinReproKernel$K -I"$ACL_INC" \
        -c glue.cpp -o glue_$K.o
    # link glue + aicpu object into one DSO (link step, no -x)
    bisheng --npu-arch=dav-2201 -shared -fPIC glue_$K.o "$OBJ" -o "$SO" \
        -L"$ACL_LIB" -Wl,-rpath,"$ACL_LIB" -lascendcl
    echo "[host] built $SO (glue_$K.o + $OBJ)"
done
gcc -O2 host.cpp -o host -ldl
echo "[host] built host"
