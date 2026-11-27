// Copyright (c) 2026, Minghua Shen.
//
// Host-side declaration of this extension's AICPU kernel launcher, defined in
// aicpu_launch.cpp. flash_api.cpp is a host TU - torch >= 2.14 headers need
// C++20, which the bisheng ASC frontend cannot parse - so it calls this
// instead of using ASC launch syntax or the aicpu kernel declarations
// directly.
#pragma once

namespace fa_aicpu {
// ComputeFAMetadataV2<<<1, nullptr, aicpu_stream>>>(args, size)
void fa_metadata_v2_launch(void* args, unsigned long size, void* aicpu_stream);
} // namespace fa_aicpu
