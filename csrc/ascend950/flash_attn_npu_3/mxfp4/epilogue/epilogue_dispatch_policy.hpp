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

#ifndef MXFP4_EPILOGUE_DISPATCH_POLICY_HPP
#define MXFP4_EPILOGUE_DISPATCH_POLICY_HPP

#include "catlass/numeric_size.hpp"
#include "catlass/arch/arch.hpp"

namespace Catlass::Epilogue {

enum class LseMode
{
    NONE = 0
};
enum class LseFormat
{
    TN1 = 0,
    BNS1 = 1,
    BSN1 = 2
};
enum class MXQuantMode
{
    NONE = 0,
    OCP = 1,
    CX = 2
};

struct EpilogueMask2Idx {
    static constexpr uint32_t IO_STAGES = 2;
    using ArchTag = Catlass::Arch::Ascend950;
};

template <bool transposedMm1_ = false, MXQuantMode MX_QUANT_MODE_ = MXQuantMode::NONE>
struct EpilogueOnlineSoftmaxMx {
    using ArchTag = Catlass::Arch::Ascend950;
    static constexpr MXQuantMode MX_QUANT_MODE = MX_QUANT_MODE_;
};

template <bool transposedMm1_ = false, MXQuantMode MX_QUANT_MODE_ = MXQuantMode::NONE>
struct EpilogueComputePScaleMx {
    using ArchTag = Catlass::Arch::Ascend950;
    static constexpr MXQuantMode MX_QUANT_MODE = MX_QUANT_MODE_;
};

struct EpilogueCopyGlobalMaxUbToL1Mx {
    using ArchTag = Catlass::Arch::Ascend950;
};

template <LseMode LSE_MODE_ = LseMode::NONE, LseFormat LSE_FORMAT_ = LseFormat::BNS1, bool transposedMm1_ = false>
struct EpilogueAtlasA5RescaleOMx {
    using ArchTag = Catlass::Arch::Ascend950;
};

} // namespace Catlass::Epilogue

#endif // EPILOGUE_DISPATCH_POLICY_HPP
