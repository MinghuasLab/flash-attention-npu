/* Per-kernel DSO glue: compiled once per kernel (-DKERNEL_SYM=...), linked
 * with that kernel's aicpu object. Mirrors flash-attention-npu's per-version
 * extension module: the aicpu object embeds .aicpu_binary/.aicpuBinRec, and
 * the first <<<>>> launch makes the CANN runtime lazily deploy it. */
#include <cstdio>
#include <cstdint>
#include "acl/acl.h"

#define STR_(x) #x
#define STR(x) STR_(x)

#ifndef KERNEL_SYM
#error "compile with -DKERNEL_SYM=MinReproKernelA|MinReproKernelB"
#endif

extern __global__ __aicpu__ uint32_t KERNEL_SYM(void *args);

extern "C" int launch_kernel()
{
    static bool inited = false;
    if (!inited) { aclError ei = aclInit(NULL); printf("[glue] aclInit -> %d\n", (int)ei); inited = true; }
    aclrtSetDevice(0);
    uint64_t args = 0;
    aclrtStream s = nullptr;
    aclError e0 = aclrtCreateStream(&s);
    KERNEL_SYM<<<1, nullptr, s>>>(&args, sizeof(args));
    aclError e = aclrtSynchronizeStream(s);
    printf("[glue] launch+sync %s -> create=%d sync=%d (%s)\n",
           STR(KERNEL_SYM), (int)e0, (int)e, e == 0 ? "OK" : "FAIL");
    aclrtDestroyStream(s);
    return (int)e;
}
