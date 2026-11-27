// Copyright (c) 2026, Minghua Shen.
//
// ASC TU: this extension's AICPU kernel launcher, extracted from flash_api.cpp
// so the torch-facing host TU never needs ASC launch syntax or the aicpu
// kernel declarations. The kernel itself is compiled from fa_metadata.aicpu
// via `bisheng -x aicpu` and linked into the same extension.

#include "acl/acl.h"
#include "aicpu_launch.hpp"

extern __global__ __aicpu__ uint32_t ComputeFAMetadata(void* args);

namespace fa_aicpu {
void fa_metadata_950_v3_launch(void* args, unsigned long size, void* aicpu_stream)
{
    ComputeFAMetadata<<<1, nullptr, static_cast<aclrtStream>(aicpu_stream)>>>(args, size);
}
} // namespace fa_aicpu
