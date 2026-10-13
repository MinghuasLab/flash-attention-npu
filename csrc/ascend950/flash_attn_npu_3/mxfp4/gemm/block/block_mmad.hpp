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

#ifndef MXFP4_BLOCK_MMAD_HPP
#define MXFP4_BLOCK_MMAD_HPP

#include "catlass/numeric_size.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "../../gemm/tile_common/gemm_tile_copy.hpp"

namespace Catlass::Gemm::Block {

#if (__CCE_AICORE__ == 310)
template <class DispatchPolicy, class L1TileShape, class L0TileShape, class ElementA, class ElementB, class ElementC,
          class ElementBias = void,
          class TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
              typename DispatchPolicy::ArchTag, ElementA, Catlass::layout::RowMajor, ElementB,
              Catlass::layout::RowMajor, ElementC, Catlass::layout::RowMajor, ElementBias>,
          class TileMmad = Catlass::Gemm::Tile::TileMmadTla<typename DispatchPolicy::ArchTag, ElementA,
                                                            typename TileCopy::LayoutTagL1A>>
struct BlockMmadTla {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadTla is not implemented for this DispatchPolicy");
};
#endif

} // namespace Catlass::Gemm::Block
#if (__CCE_AICORE__ == 310)
#include "../../gemm/block/block_mmad_qk.hpp"
#include "../../gemm/block/block_mmad_pv.hpp"
#include "../../gemm/block/block_mmad_copy_global_max_l1_to_ub.hpp"
#endif
#endif // MXFP4_BLOCK_MMAD_HPP
