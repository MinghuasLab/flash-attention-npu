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

#ifndef MXFP4_TILE_COPY_HPP
#define MXFP4_TILE_COPY_HPP

#include "catlass/numeric_size.hpp"
#include "catlass/gemm/helper.hpp"
#include "../../gemm/tile_common/copy_l0c_to_dst.hpp"
#if (__CCE_AICORE__ == 310)
#include "../../gemm/tile_common/copy_gm_to_l1.hpp"
#include "../../gemm/tile_common/copy_l0c_to_ub.hpp"
#endif

namespace Catlass::Gemm::Tile {

// Extra template parameters are kept so kernel_interface instantiations stay stable.
template <class ArchTag, class ElementA_, class LayoutTagA_, class ElementB_, class LayoutTagB_, class ElementC_,
          class LayoutTagC_, class ElementBias = void, bool ReluEnable_ = false,
          ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT, class L0CCopyMode = void>
struct PackedTileCopyTla {
    using LayoutTagL1A = typename Catlass::Gemm::helper::L1ATypeSelector<
        Catlass::Gemm::GemmType<ElementA_, LayoutTagA_>>::L1AType::Layout;
    using LayoutTagL1B = typename Catlass::Gemm::helper::L1BTypeSelector<
        Catlass::Gemm::GemmType<ElementB_, LayoutTagB_>>::L1BType::Layout;
    using LayoutTagL0A = typename Catlass::Gemm::helper::L0ALayoutSelector<ArchTag>::Layout;
    using LayoutTagL0B = Catlass::layout::nZ;
};

#if (__CCE_AICORE__ == 310)
template <class ArchTag, class ElementA_, class LayoutTagA, class ElementB_, class LayoutTagB, class ElementC_,
          class LayoutTagC, class ElementBias = void, CopyL0CToUBMode CopyMode_ = CopyL0CToUBMode::NO_SPLIT,
          bool ReluEnable = false, ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT>
struct PackedTileCopyTlaToUB : public PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA, ElementB_, LayoutTagB,
                                                        ElementC_, LayoutTagC, ElementBias> {};
#endif

} // namespace Catlass::Gemm::Tile

#endif // MXFP4_TILE_COPY_HPP
