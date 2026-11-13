/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Modified by Minghua Shen, 2026.
 */
#ifndef FA950_FWD_RESOURCE_HPP
#define FA950_FWD_RESOURCE_HPP
#include "catlass/arch/resource.hpp"

namespace Catlass::Arch {
// Only initialize Cube-only buffers on AIC; AIV never consumes them.
template <AscendC::TPosition Position>
struct ForwardCubeBuffer950 : LocalTensorBufferBase {
    __aicore__ inline ForwardCubeBuffer950()
    {
#ifdef __DAV_CUBE__
        LocalTensorBuffer<Ascend950, Position> buffer;
        tensor = buffer.template GetBufferByByte<uint8_t>(0);
#endif
    }
};

struct ForwardResource950 {
    AscendC::TPipe pipe;
    LocalTensorBuffer<Ascend950, AscendC::TPosition::A1> l1Buf;
    LocalTensorBuffer<Ascend950, AscendC::TPosition::A2> l0ABuf;
    LocalTensorBuffer<Ascend950, AscendC::TPosition::B2> l0BBuf;
    LocalTensorBuffer<Ascend950, AscendC::TPosition::C2> btBuf;
    ForwardCubeBuffer950<AscendC::TPosition::CO1> l0CBuf;
    LocalTensorBuffer<Ascend950, AscendC::TPosition::VECCALC> ubBuf;
    ForwardCubeBuffer950<AscendC::TPosition::C2PIPE2GM> fpBuf;

    __aicore__ inline ForwardResource950()
    {
        pipe.Destroy();
    }
};
} // namespace Catlass::Arch
#endif
