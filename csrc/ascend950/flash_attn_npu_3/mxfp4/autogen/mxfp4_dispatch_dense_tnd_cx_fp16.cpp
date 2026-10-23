/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// MXFP4 dispatch, dense x fp16 x TND x CX variant.
// One explicit instantiation per translation unit so the MXFP4 kernel
// templates compile in parallel across cores.

#include "../mxfp4_kernel_dispatch_impl.hpp"

template void launch_mxfp4_impl<half, Mxfp4Layout::TND, Mxfp4QuantMode::CX, false>(
    const Mxfp4LaunchArgs &a);
