// Copyright (c) 2026, Minghua Shen.
//
// ASC TU: this extension's AICPU kernel launchers, extracted from flash_api.cpp
// so the torch-facing host TU never needs ASC launch syntax or the aicpu
// kernel declarations. The kernels themselves are compiled from
// fa_metadata.aicpu via `bisheng -x aicpu` and linked into the same extension.

#include "acl/acl.h"
#include "aicpu_launch.hpp"

extern __global__ __aicpu__ uint32_t ComputeFAMetadata(void* args);
extern __global__ __aicpu__ uint32_t WriteFARuntime(void* args);

namespace fa_aicpu {
void fa_metadata_950_v4_launch(void* args, unsigned long size, void* aicpu_stream)
{
    ComputeFAMetadata<<<1, nullptr, static_cast<aclrtStream>(aicpu_stream)>>>(args, size);
}

void fa_write_runtime_launch(void* args, unsigned long size, void* rt_stream)
{
    WriteFARuntime<<<1, nullptr, static_cast<aclrtStream>(rt_stream)>>>(args, size);
}
} // namespace fa_aicpu
