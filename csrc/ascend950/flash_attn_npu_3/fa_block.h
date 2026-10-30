/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

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
