/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* Modified by Minghua Shen, 2026. */

#ifndef VF_COMMON_DEF_H_
#define VF_COMMON_DEF_H_
#include "kernel_tensor.h"

namespace Catlass::Epilogue::Block::Mxfp4VF {
using namespace AscendC;
using namespace Reg;

constexpr static AscendC::Reg::CastTrait h2iZero = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::Reg::CastTrait h2iOne = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::Reg::CastTrait castTraitZero = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::Reg::CastTrait castTraitOne = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::Reg::CastTrait castTraitTwo = {
    AscendC::Reg::RegLayout::TWO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::Reg::CastTrait castTraitThree = {
    AscendC::Reg::RegLayout::THREE,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr half NUM_127 = static_cast<half>(127.0f);
constexpr half NUM_NEG_127 = static_cast<half>(-127.0f);
constexpr half NUM_NEG_125 = static_cast<half>(-125.0f);
constexpr half ZERO_VALUE = static_cast<half>(0.0f);
constexpr int16_t SHIFT_VALUE = 23;

constexpr uint8_t NUM_128 = static_cast<uint8_t>(128);
constexpr int16_t NUM_2 = static_cast<int16_t>(2);
constexpr int8_t indexSubLength = static_cast<int8_t>(32);

constexpr half LN2 = static_cast<half>(0.6931471806f);
constexpr half INV_LN2 = static_cast<half>(1.4426950409f);
constexpr half NEG_TWO_VALE = static_cast<half>(-2.0f);
constexpr half TWO_VALE = static_cast<half>(2.0f);
constexpr half MIN_VALUE = static_cast<half>(-65504.0f);

} // namespace Catlass::Epilogue::Block::Mxfp4VF
#endif // VF_COMMON_DEF_H_
