// Copyright (c) 2026, Minghua Shen.
//
// ASC TU: AICPU kernel launchers extracted from the flash_api.cpp files so
// the torch-facing host TUs never need ASC launch syntax or the aicpu kernel
// declarations. The kernels themselves are compiled from each family's
// fa_metadata.aicpu via `bisheng -x aicpu` and linked into the same
// extension.
//
// This single source is compiled once per extension with exactly one
// -DFA_AICPU_FAMILY_* define (set by setup.py from the extension name), so
// each .so only references the aicpu kernels it actually ships — python
// dlopens extension modules with RTLD_NOW, so every undefined reference in
// the .so must exist at import time.

#include "acl/acl.h"
#include "fa_aicpu_launch.hpp"

#if defined(FA_AICPU_FAMILY_V2)
extern __global__ __aicpu__ uint32_t ComputeFAMetadataV2(void* args);
#elif defined(FA_AICPU_FAMILY_V3)
extern __global__ __aicpu__ uint32_t ComputeFAMetadataV3(void* args);
#elif defined(FA_AICPU_FAMILY_V4)
extern __global__ __aicpu__ uint32_t ComputeFAMetadataV4(void* args);
extern __global__ __aicpu__ uint32_t WriteFARuntime(void* args);
#elif defined(FA_AICPU_FAMILY_V3_950)
extern __global__ __aicpu__ uint32_t ComputeFAMetadata(void* args);
#elif defined(FA_AICPU_FAMILY_V4_950)
extern __global__ __aicpu__ uint32_t ComputeFAMetadata(void* args);
extern __global__ __aicpu__ uint32_t WriteFARuntime(void* args);
#endif

namespace fa_aicpu {
#if defined(FA_AICPU_FAMILY_V2)
void fa_metadata_v2_launch(void* args, unsigned long size, void* aicpu_stream)
{
    ComputeFAMetadataV2<<<1, nullptr, static_cast<aclrtStream>(aicpu_stream)>>>(args, size);
}
#elif defined(FA_AICPU_FAMILY_V3)
void fa_metadata_v3_launch(void* args, unsigned long size, void* aicpu_stream)
{
    ComputeFAMetadataV3<<<1, nullptr, static_cast<aclrtStream>(aicpu_stream)>>>(args, size);
}
#elif defined(FA_AICPU_FAMILY_V4)
void fa_metadata_v4_launch(void* args, unsigned long size, void* aicpu_stream)
{
    ComputeFAMetadataV4<<<1, nullptr, static_cast<aclrtStream>(aicpu_stream)>>>(args, size);
}

void fa_write_runtime_launch(void* args, unsigned long size, void* rt_stream)
{
    WriteFARuntime<<<1, nullptr, static_cast<aclrtStream>(rt_stream)>>>(args, size);
}
#elif defined(FA_AICPU_FAMILY_V3_950)
void fa_metadata_950_v3_launch(void* args, unsigned long size, void* aicpu_stream)
{
    ComputeFAMetadata<<<1, nullptr, static_cast<aclrtStream>(aicpu_stream)>>>(args, size);
}
#elif defined(FA_AICPU_FAMILY_V4_950)
void fa_metadata_950_v4_launch(void* args, unsigned long size, void* aicpu_stream)
{
    ComputeFAMetadata<<<1, nullptr, static_cast<aclrtStream>(aicpu_stream)>>>(args, size);
}

void fa_write_runtime_launch(void* args, unsigned long size, void* rt_stream)
{
    WriteFARuntime<<<1, nullptr, static_cast<aclrtStream>(rt_stream)>>>(args, size);
}
#endif
} // namespace fa_aicpu
