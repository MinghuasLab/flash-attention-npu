/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// MXFP4 dispatch, dense x bf16 x BNSD x OCP variant.
// One explicit instantiation per translation unit so the MXFP4 kernel
// templates compile in parallel across cores.

#include "../mxfp4_kernel_dispatch_impl.hpp"

template void launch_mxfp4_impl<bfloat16_t, Mxfp4Layout::BNSD, Mxfp4QuantMode::OCP, false>(
    const Mxfp4LaunchArgs &a);
