/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// MXFP4 dispatch, sparse x fp16 x BSND x OCP variant.
// One explicit instantiation per translation unit so the MXFP4 kernel
// templates compile in parallel across cores.

#include "../mxfp4_kernel_dispatch_impl.hpp"

template void launch_mxfp4_impl<half, Mxfp4Layout::BSND, Mxfp4QuantMode::OCP, true>(
    const Mxfp4LaunchArgs &a);
