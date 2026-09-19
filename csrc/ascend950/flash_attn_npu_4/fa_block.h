/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef FAI_BLOCK_HPP
#define FAI_BLOCK_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

namespace Catlass::Epilogue {

// Dispatch policy for the fwd online-softmax epilogue. HAS_SOFTCAP_ carries the
// softcap / no-softcap kernel selection as a compile-time flag: the per-tile
// tanh softcap is then either folded into the scale+max pass or compiled out
// entirely. EpilogueFAOnlineSoftmax (catlass) stays the no-softcap spelling.
template <bool HAS_SOFTCAP_>
struct EpilogueFAOnlineSoftmaxT {
    using ArchTag = Arch::Ascend950;
    static constexpr bool HAS_SOFTCAP = HAS_SOFTCAP_;
};

} // namespace Catlass::Epilogue

#endif // FAI_BLOCK_HPP
