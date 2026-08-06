/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Ascend950 FA-V4 local epilogue dispatch tags.
 * Mirrors ascend910 fa_block.h LSE_MODE (compile-time OUT_ONLY vs NONE).
 */

#ifndef FAI_BLOCK_950_V4_HPP
#define FAI_BLOCK_950_V4_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

namespace Catlass::Epilogue {

enum class LseModeT { NONE = 0, OUT_ONLY = 1 };

// Templated rescale-O dispatch: LSE write is compile-time gated via LSE_MODE.
template <LseModeT LSE_MODE_>
struct EpilogueFARescaleOT {
    using ArchTag = Arch::Ascend950;
    static constexpr LseModeT LSE_MODE = LSE_MODE_;
};

}  // namespace Catlass::Epilogue

#endif  // FAI_BLOCK_950_V4_HPP
