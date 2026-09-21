/* Minimal host reproducer: dlopen two kernel DSOs in ONE process (same as
 * importing both flash_attn extension modules), launch A then B.
 * Driver 25.5.0 (910B3): B fails 507018 / errcode 11003 get kernel failed.
 * Driver 25.5.2: both OK. */
#include <cstdio>
#include <initializer_list>
#include <cstdlib>
#include <dlfcn.h>
#include <unistd.h>

int main()
{
    printf("[repro] pid=%d\n", (int)getpid());
    int rc = 0;
    for (const char *name : {"liblaunch_a.so", "liblaunch_b.so"}) {
        printf("[repro] dlopen %s\n", name);
        char path[256];
        snprintf(path, sizeof(path), "./%s", name);
        void *h = dlopen(path, RTLD_NOW);
        if (!h) { printf("[repro] dlopen failed: %s\n", dlerror()); return 1; }
        auto launch = (int (*)())dlsym(h, "launch_kernel");
        if (!launch) { printf("[repro] dlsym failed: %s\n", dlerror()); return 1; }
        if (launch() != 0) rc = 2;
    }
    printf(rc ? "[repro] BUG REPRODUCED: second equal-size kernel failed\n"
              : "[repro] both kernels OK (healthy driver)\n");
    return rc;
}
