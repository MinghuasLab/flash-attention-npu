// Copyright (c) 2026, Minghua Shen.
//
// Host-side declarations of the AICPU kernel launchers, defined in
// csrc/fa_aicpu_launch.cpp. flash_api.cpp files are host TUs - torch >= 2.14
// headers need C++20, which the bisheng ASC frontend cannot parse - so they
// call these instead of using ASC launch syntax or the aicpu kernel
// declarations directly.
//
// Only the launchers of the extension being built are declared: setup.py
// defines one FA_AICPU_FAMILY_* for the host and the device TUs of an
// extension alike, so calling another family's launcher does not compile
// here instead of failing at import time.
#pragma once

namespace fa_aicpu {
#if defined(FA_AICPU_FAMILY_V2)
// ComputeFAMetadataV2<<<1, nullptr, aicpu_stream>>>(args, size)  — v2, 910
void fa_metadata_v2_launch(void* args, unsigned long size, void* aicpu_stream);
#elif defined(FA_AICPU_FAMILY_V3)
// ComputeFAMetadataV3<<<1, nullptr, aicpu_stream>>>(args, size)  — v3, 910
void fa_metadata_v3_launch(void* args, unsigned long size, void* aicpu_stream);
#elif defined(FA_AICPU_FAMILY_V4)
// ComputeFAMetadataV4<<<1, nullptr, aicpu_stream>>>(args, size)  — v4, 910
void fa_metadata_v4_launch(void* args, unsigned long size, void* aicpu_stream);
// WriteFARuntime<<<1, nullptr, rt_stream>>>(args, size)          — v4, 910
void fa_write_runtime_launch(void* args, unsigned long size, void* rt_stream);
#elif defined(FA_AICPU_FAMILY_V3_950)
// ComputeFAMetadata<<<1, nullptr, aicpu_stream>>>(args, size)    — v3, 950
void fa_metadata_950_v3_launch(void* args, unsigned long size, void* aicpu_stream);
#elif defined(FA_AICPU_FAMILY_V4_950)
// ComputeFAMetadata<<<1, nullptr, aicpu_stream>>>(args, size)    — v4, 950
void fa_metadata_950_v4_launch(void* args, unsigned long size, void* aicpu_stream);
// WriteFARuntime<<<1, nullptr, rt_stream>>>(args, size)          — v4, 950
void fa_write_runtime_launch(void* args, unsigned long size, void* rt_stream);
#endif
} // namespace fa_aicpu
