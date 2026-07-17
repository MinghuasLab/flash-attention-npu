/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef FAG_BLOCK_HPP
#define FAG_BLOCK_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"

using namespace Catlass;

namespace Catlass::Gemm::Block {
template <
    class DispatchPolicy,
    class BlockTileShape,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFagSdp {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFagSdp is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class BlockTileShape,
    class L1ATileShape,
    class L1BTileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmadFAG {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFAG is not implemented for this DispatchPolicy");
};
}

namespace Catlass::Epilogue {
// For AtlasA2, MLAG Pre
struct EpilogueAtlasA2FAGPre {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, MLAG Op
template <bool HAS_SOFTCAP_>
struct EpilogueAtlasA2FAGOp {
    using ArchTag = Arch::AtlasA2;
    static constexpr bool HAS_SOFTCAP = HAS_SOFTCAP_;
};

// For AtlasA2, MLAG Sfmg
template <uint32_t INPUT_LAYOUT_>
struct EpilogueAtlasA2FAGSfmg {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, MLAG Op
template <uint32_t INPUT_LAYOUT_, bool IS_DROP_, bool IS_ATTEN_MASK_,
bool HAS_SOFTCAP_>
struct EpilogueAtlasA2SameAbVec {
    using ArchTag = Arch::AtlasA2;
    static constexpr bool HAS_SOFTCAP = HAS_SOFTCAP_;
};

// For AtlasA2, MLAG Post
struct EpilogueAtlasA2FAGPost {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, MLAG DeterministicAdd
template <uint32_t INPUT_LAYOUT_>
struct EpilogueAtlasA2FAGDtmAdd {
    using ArchTag = Arch::AtlasA2;
};
}

namespace Catlass::Gemm {
struct MmadAtlasA2FAGCube1 : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2FAGCube2 : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2FAGCube3 : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

template <uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FagSdp : public MmadAtlasA2 {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0AB_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = 1;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FagdQKV : public MmadAtlasA2 {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0AB_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = 1;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};
}

#endif // FAG_BLOCK_HPP
