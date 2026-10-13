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

#ifndef GEMM_TILE_COPY_L0C_TO_DST_HPP
#define GEMM_TILE_COPY_L0C_TO_DST_HPP

#include "catlass/numeric_size.hpp"

namespace Catlass::Gemm::Tile {

enum class ScaleGranularity {
    NO_QUANT = 0,
    PER_TENSOR,
    PER_GROUP
};

enum class CopyL0CToUBMode {
    NO_SPLIT = 0,
    SPLIT_M,
    SPLIT_N
};

template <class ArchTag, class ElementSrc, class ElementDst,
          ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT>
struct CopyL0CToDstQuantMode {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

template <class ArchTag>
struct CopyL0CToDstQuantMode<ArchTag, float, float, ScaleGranularity::NO_QUANT> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

template <class ArchTag>
struct CopyL0CToDstQuantMode<ArchTag, float, half, ScaleGranularity::PER_TENSOR> {
    static constexpr auto VALUE = QuantMode_t::QF322F16_PRE;
};

template <class ArchTag, class TensorSrc, class TensorDst, CopyL0CToUBMode CopyMode = CopyL0CToUBMode::NO_SPLIT,
          ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT, bool ReluEnable = false,
          class Enable = void>
struct CopyL0CToUBTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to ub, can not find the specialization.");
};

} // namespace Catlass::Gemm::Tile

#endif // GEMM_TILE_COPY_L0C_TO_DST_HPP
