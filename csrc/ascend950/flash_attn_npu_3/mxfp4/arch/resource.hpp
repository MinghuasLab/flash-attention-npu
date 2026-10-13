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

#ifndef MXFP4_RESOURCE_HPP
#define MXFP4_RESOURCE_HPP

#include "catlass/numeric_size.hpp"
#include "../arch/local_tensor_buffer.hpp"

namespace Catlass::Arch {

template <class ArchTag>
struct Resource {};

template <>
struct Resource<Catlass::Arch::Ascend950> {
  public:
    LocalTensorBuffer<Catlass::Arch::Ascend950, AscendC::TPosition::A1> l1Buf;
    LocalTensorBuffer<Catlass::Arch::Ascend950, AscendC::TPosition::A2> l0ABuf;
    LocalTensorBuffer<Catlass::Arch::Ascend950, AscendC::TPosition::B2> l0BBuf;
    LocalTensorBuffer<Catlass::Arch::Ascend950, AscendC::TPosition::CO1> l0CBuf;
    LocalTensorBuffer<Catlass::Arch::Ascend950, AscendC::TPosition::VECCALC> ubBuf;

    __aicore__ inline Resource()
    {
        AscendC::InitSocState();
    }

    __aicore__ inline ~Resource()
    {
        AscendC::InitSocState();
    }
};

} // namespace Catlass::Arch

#endif // MXFP4_RESOURCE_HPP
