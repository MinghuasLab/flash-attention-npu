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

// Runtime D512 layout. GM stack slots and L1 staging slots have distinct lifetimes.
struct Stack512Layout {
    static constexpr uint32_t QRows = 128;
    static constexpr uint32_t KvRows = 512;
    static constexpr uint32_t Slots = 3;
    static constexpr uint32_t VectorRows = 64;
    static constexpr uint32_t BatchRows = 16;
    static constexpr uint32_t QOffset = 0;
    static constexpr uint32_t KPOffset = 128 * 1024;
    static constexpr uint32_t VOffset = 256 * 1024;
    static constexpr uint32_t L1Bytes = 384 * 1024;
    static constexpr uint32_t SOffset = 0;
    static constexpr uint32_t POffset = 64 * 1024;
    static constexpr uint32_t LoOffset = 96 * 1024;
    static constexpr uint32_t GoOffset = 128 * 1024;
    static constexpr uint32_t MaxOffset = 173568;
    static constexpr uint32_t SumOffset = 174336;
    static constexpr uint32_t AlphaOffset = 175104;
    static constexpr uint32_t LseOffset = 177920;
    static constexpr uint32_t UbBytes = 216 * 1024;
    static constexpr uint64_t SlotElements = QRows * KvRows;
    static constexpr uint64_t SBytesPerCore = Slots * SlotElements * sizeof(float);
    static constexpr uint64_t PBytesPerCore = Slots * SlotElements * sizeof(half);
    static constexpr uint64_t OBytesPerCore = Slots * SlotElements * sizeof(float);
    static constexpr uint64_t UpdateBytesPerCore = Slots * SlotElements * sizeof(float);
    static constexpr uint64_t WorkspaceBytesPerCore =
        SBytesPerCore + PBytesPerCore + OBytesPerCore + UpdateBytesPerCore;
};
// D512 GM ring ownership. Ready and free travel in opposite directions;
// a slot has at most one outstanding token in either direction.
struct Stack512Sync {
    static constexpr uint16_t S = 0;
    static constexpr uint16_t P = 3;
    static constexpr uint16_t O = 6;
    template <pipe_t Pipe>
    __aicore__ static inline void Set(uint16_t id)
    {
#ifdef __DAV_CUBE__
        AscendC::CrossCoreSetFlag<4, Pipe>(id);
        AscendC::CrossCoreSetFlag<4, Pipe>(id + 16);
#endif
#ifdef __DAV_VEC__
        AscendC::CrossCoreSetFlag<4, Pipe>(id);
#endif
    }
    template <pipe_t Pipe>
    __aicore__ static inline void Wait(uint16_t id)
    {
#ifdef __DAV_CUBE__
        AscendC::CrossCoreWaitFlag<4, Pipe>(id);
        AscendC::CrossCoreWaitFlag<4, Pipe>(id + 16);
#endif
#ifdef __DAV_VEC__
        AscendC::CrossCoreWaitFlag<4, Pipe>(id);
#endif
    }
    __aicore__ static inline void Init()
    {
        for (uint16_t slot = 0; slot < Stack512Layout::Slots; ++slot) {
#ifdef __DAV_CUBE__
            Set<PIPE_MTE1>(P + slot);
#endif
#ifdef __DAV_VEC__
            Set<PIPE_V>(S + slot);
            Set<PIPE_V>(O + slot);
#endif
        }
    }
    // Call only after all real stacks and the two delayed rounds finish.
    __aicore__ static inline void Drain()
    {
        for (uint16_t slot = 0; slot < Stack512Layout::Slots; ++slot) {
#ifdef __DAV_CUBE__
            Wait<PIPE_FIX>(S + slot);
            Wait<PIPE_FIX>(O + slot);
#endif
#ifdef __DAV_VEC__
            Wait<PIPE_MTE3>(P + slot);
#endif
        }
    }
};
static_assert(Stack512Layout::UbBytes <= Arch::Ascend950::UB_SIZE);
static_assert(Stack512Layout::L1Bytes <= Arch::Ascend950::L1_SIZE);

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
