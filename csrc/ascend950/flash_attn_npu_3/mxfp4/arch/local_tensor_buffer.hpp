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

#ifndef MXFP4_LOCAL_TENSOR_BUFFER_HPP
#define MXFP4_LOCAL_TENSOR_BUFFER_HPP

#include "catlass/numeric_size.hpp"
#include "catlass/arch/arch.hpp"

namespace Catlass::Arch {

struct LocalTensorBufferBase {
public:
    template <class Element = half>
    __aicore__ inline AscendC::LocalTensor<Element> GetBufferByByte(const uint32_t offset) const
    {
        return tensor[offset].template ReinterpretCast<Element>();
    }

protected:
    __aicore__ inline LocalTensorBufferBase() = default;

    AscendC::LocalTensor<uint8_t> tensor;
};

template <class ArchTag, AscendC::TPosition Position>
struct LocalTensorBuffer {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported local tensor buffer, can not find the specialization.");
};

template <>
struct LocalTensorBuffer<Catlass::Arch::Ascend950, AscendC::TPosition::A1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::A1;

    __aicore__ inline LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::A1, 0, Catlass::Arch::Ascend950::L1_SIZE);
    }
};

template <>
struct LocalTensorBuffer<Catlass::Arch::Ascend950, AscendC::TPosition::A2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::A2;

    __aicore__ inline LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::A2, 0, Catlass::Arch::Ascend950::L0A_SIZE);
    }
};

template <>
struct LocalTensorBuffer<Catlass::Arch::Ascend950, AscendC::TPosition::B2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::B2;

    __aicore__ inline LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::B2, 0, Catlass::Arch::Ascend950::L0B_SIZE);
    }
};

template <>
struct LocalTensorBuffer<Catlass::Arch::Ascend950, AscendC::TPosition::CO1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::CO1;

    __aicore__ inline LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::CO1, 0, Catlass::Arch::Ascend950::L0C_SIZE);
    }
};

template <>
struct LocalTensorBuffer<Catlass::Arch::Ascend950, AscendC::TPosition::VECCALC> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECCALC;

    __aicore__ inline LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::VECCALC, 0, Catlass::Arch::Ascend950::UB_SIZE);
    }
};
} // namespace Catlass::Arch

#endif // INCLUDE_ARCH_MEMORY_H
