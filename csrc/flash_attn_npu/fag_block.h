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

using namespace Catlass;

namespace Catlass::Epilogue {
    // For AtlasA2, MLAG Pre
 	struct EpilogueAtlasA2FAGPre {
 	    using ArchTag = Arch::AtlasA2;
 	};
 	 
    // For AtlasA2, MLAG Op
    struct EpilogueAtlasA2FAGOp {
        using ArchTag = Arch::AtlasA2;
    };
    
    // For AtlasA2, MLAG Sfmg
    struct EpilogueAtlasA2FAGSfmg {
        using ArchTag = Arch::AtlasA2;
    };
    
    // For AtlasA2, MLAG Post
    struct EpilogueAtlasA2FAGPost {
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
}
#endif // FAG_BLOCK_HPP