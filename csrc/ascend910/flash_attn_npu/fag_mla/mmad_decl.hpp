#pragma once
#include "fag_block.h"
namespace Catlass::Gemm::Block::Mla {
template <class DispatchPolicy, class BlockTileShape, class L1TileShape, class L0TileShape, class AType, class BType,
          class CType, class BiasType = void,
          class TileCopy = Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
          class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>>
struct BlockMmadFagSdp {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFagSdp is not implemented for this DispatchPolicy");
};

template <class DispatchPolicy, class BlockTileShape, class L1ATileShape, class L1BTileShape, class L0TileShape,
          class AType, class BType, class CType, class BiasType = void,
          class TileCopy = Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
          class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>>
struct BlockMmadFAG {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFAG is not implemented for this DispatchPolicy");
};
} // namespace Catlass::Gemm::Block::Mla
