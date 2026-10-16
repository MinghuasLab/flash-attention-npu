/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* Modified by Minghua Shen, 2026. */

#ifndef MXFP4_DISPATCH_POLICY_HPP
#define MXFP4_DISPATCH_POLICY_HPP

#include "catlass/arch/arch.hpp"

namespace Catlass::Gemm {

struct MmadAtlasA5 {
    using ArchTag = Catlass::Arch::Ascend950;
};

template <bool transposedMm1_ = false>
struct MmadAtlasA5QkMx : public MmadAtlasA5 {};

template <bool transposedMm1_ = false>
struct MmadAtlasA5PvMx : public MmadAtlasA5 {};

struct CopyGlobalMaxL1ToUb : public MmadAtlasA5 {};

} // namespace Catlass::Gemm

#endif // MXFP4_DISPATCH_POLICY_HPP
