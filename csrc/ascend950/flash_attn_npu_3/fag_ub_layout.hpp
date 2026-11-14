/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Shared AIV UB layout for the Ascend950 FAG backward pipeline.
 */

#ifndef FLASH_ATTN_NPU_FAG_UB_LAYOUT_HPP
#define FLASH_ATTN_NPU_FAG_UB_LAYOUT_HPP

#include <cstdint>

// Host compilation keeps a normal internal-linkage inline.
#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
#define FAG_UB_LAYOUT_API __forceinline__ __aicore__
#else
#define FAG_UB_LAYOUT_API static inline
#endif

namespace FagUbLayout {

// Physical UB row width in 32-bit lanes. The VF kernels always operate on
// 128-element rows even when the L1 task width is 64, so this is a software
// constant, not a hardware one.
constexpr uint32_t ROW_STRIDE = 128;

// Byte alignment used between consecutive UB slots.
constexpr uint32_t SLOT_ALIGN_BYTES = 32;

// The FAG backward kernels are fp16/bf16 only, hence 2-byte P/dS elements.
constexpr uint32_t P_ELEMENT_BYTES = sizeof(uint16_t);

FAG_UB_LAYOUT_API uint64_t RoundUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

// Rows of one L1 task tile handled by a single AIV sub-block.
FAG_UB_LAYOUT_API uint32_t RowsPerSubBlock(uint32_t qTile)
{
    return static_cast<uint32_t>(RoundUp(qTile, 16) / 2);
}

/*
 * One ping (or pong) half of the main pipeline:
 *   [mm1Res][mm2Res][attenMask][LSE][P(NZ, M+1 padding)][dS(NZ, M+1 padding)][delta]
 *
 * mm1Res / mm2Res hold fp32 S and dP, attenMask is uint8, P and dS are stored
 * NZ-padded with one extra row, delta is one fp32 per row.
 */
struct PipelineSlots {
    uint32_t mm1 = 0;
    uint32_t mm2 = 0;
    uint32_t attenMask = 0;
    uint32_t lse = 0;
    uint32_t p = 0;
    uint32_t dS = 0;
    uint32_t delta = 0;
    uint32_t halfBytes = 0; // size of one ping (or pong) half
};

FAG_UB_LAYOUT_API PipelineSlots MakePipelineSlots(uint32_t qTile)
{
    const uint32_t rowsPerSub = RowsPerSubBlock(qTile);
    const uint32_t mmResBytes = rowsPerSub * ROW_STRIDE * sizeof(float);
    const uint32_t attenMaskBytes = rowsPerSub * ROW_STRIDE * sizeof(uint8_t);
    const uint32_t lseBytes = static_cast<uint32_t>(RoundUp(rowsPerSub, 8) * 8 * sizeof(float));
    const uint32_t pBytes = (rowsPerSub + 1) * ROW_STRIDE * P_ELEMENT_BYTES;
    const uint32_t deltaBytes = rowsPerSub * 8 * sizeof(float);

    PipelineSlots slots;
    slots.mm1 = 0;
    slots.mm2 = slots.mm1 + mmResBytes;
    slots.attenMask = slots.mm2 + mmResBytes;
    slots.lse = static_cast<uint32_t>(RoundUp(slots.attenMask + attenMaskBytes, SLOT_ALIGN_BYTES));
    slots.p = static_cast<uint32_t>(RoundUp(slots.lse + lseBytes, SLOT_ALIGN_BYTES));
    slots.dS = static_cast<uint32_t>(RoundUp(slots.p + pBytes, SLOT_ALIGN_BYTES));
    slots.delta = static_cast<uint32_t>(RoundUp(slots.dS + pBytes, SLOT_ALIGN_BYTES));
    slots.halfBytes = static_cast<uint32_t>(RoundUp(slots.delta + deltaBytes, SLOT_ALIGN_BYTES));
    return slots;
}

// Total UB bytes of the two ping/pong halves.
FAG_UB_LAYOUT_API uint32_t PipelineBytes(uint32_t qTile)
{
    return 2 * MakePipelineSlots(qTile).halfBytes;
}

/*
 * delta pass (V1 of the epilogue): a separate double-buffered region that runs
 * to completion before RunTasks, so it reuses the pipeline bytes instead of
 * adding to them. D>256 halves the rows to keep the slot under the budget.
 */
FAG_UB_LAYOUT_API uint32_t DeltaTileRows(uint64_t vHeadDim)
{
    return vHeadDim > 256 ? 32U : 64U;
}

FAG_UB_LAYOUT_API uint32_t DeltaAlignedD(uint64_t vHeadDim)
{
    return vHeadDim > 256 ? static_cast<uint32_t>(RoundUp(vHeadDim, 16)) : 256U;
}

// Bytes of one delta slot: in/out b16 tiles plus the fp32 delta row.
FAG_UB_LAYOUT_API uint32_t DeltaSlotBytes(uint64_t vHeadDim)
{
    const uint32_t rows = DeltaTileRows(vHeadDim);
    const uint32_t alignedD = DeltaAlignedD(vHeadDim);
    return 2 * rows * alignedD * P_ELEMENT_BYTES + rows * sizeof(float);
}

} // namespace FagUbLayout

#undef FAG_UB_LAYOUT_API

#endif // FLASH_ATTN_NPU_FAG_UB_LAYOUT_HPP
