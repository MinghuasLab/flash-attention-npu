// Copyright (c) 2026, Minghua Shen.
//
// Host-side declarations of this extension's AICPU kernel launchers, defined
// in aicpu_launch.cpp. flash_api.cpp is a host TU - torch >= 2.14 headers
// need C++20, which the bisheng ASC frontend cannot parse - so it calls these
// instead of using ASC launch syntax or the aicpu kernel declarations
// directly.
#pragma once

namespace fa_aicpu {
// ComputeFAMetadata<<<1, nullptr, aicpu_stream>>>(args, size)
void fa_metadata_950_v4_launch(void* args, unsigned long size, void* aicpu_stream);
// WriteFARuntime<<<1, nullptr, rt_stream>>>(args, size)
void fa_write_runtime_launch(void* args, unsigned long size, void* rt_stream);
} // namespace fa_aicpu
