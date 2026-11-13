// Copyright (c) 2026, Minghua Shen.

#ifndef FAI_BLOCK_HPP
#define FAI_BLOCK_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "kernel_common.hpp"

namespace QuantMask {
inline constexpr MaskCategory NO_MASK = MaskCategory::NO_MASK;
inline constexpr MaskCategory CAUSAL_MASK = MaskCategory::MASK_CAUSAL;
inline constexpr MaskCategory SLIDING_WINDOW_MASK = MaskCategory::MASK_SWA;
} // namespace QuantMask

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

template <bool HAS_SOFTCAP_, MaskCategory MASK_CATEGORY_>
struct EpilogueFAQuantOnlineSoftmax {
    using ArchTag = Arch::Ascend950;
    static constexpr MaskCategory MASK_CATEGORY = MASK_CATEGORY_;
    static constexpr bool HAS_SOFTCAP = HAS_SOFTCAP_;
};

// Byte offsets shared by the kernel, OnlineSoftmax and RescaleO.
struct UBufTileHelper {
    uint32_t sStartOffset, pStartOffset, loStartOffset, goStartOffset;
    uint32_t lmStartOffset, gmStartOffset, dmStartOffset, glStartOffset;
    uint32_t lseStartOffset, indexStartOffset, totalBytes;
    uint32_t sBufferBytes, pBufferBytes, loBufferBytes;
    uint32_t otmpStages;
};

} // namespace Catlass::Epilogue

#endif // FAI_BLOCK_HPP
