/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */
#ifndef FLASH_ATTN_NPU_3_FWD_POLICY_HPP
#define FLASH_ATTN_NPU_3_FWD_POLICY_HPP

#include <cstdint>

namespace flash_attn::d512 {
constexpr uint32_t HEAD_DIM = 512;
constexpr uint32_t Q_TILE = 128;
constexpr uint32_t KV_TILE = 128;
constexpr uint32_t VECTOR_ROWS = Q_TILE / 2;
constexpr uint32_t L1_DIM_SLICE = 128;
constexpr uint32_t Q_STAGES = 1;
constexpr uint32_t K_STAGES = 2;
constexpr uint32_t V_STAGES = 1;
constexpr uint32_t S_STAGES = 2;
constexpr uint32_t UB_P_STAGES = 2;
constexpr uint32_t S_STAGE_BYTES = VECTOR_ROWS * KV_TILE * sizeof(float);
constexpr uint32_t P_STAGE_BYTES = VECTOR_ROWS * KV_TILE * 2;
constexpr uint32_t P_STAGES = 3;
constexpr uint32_t LO_STAGES = 2;
constexpr uint32_t GO_ROWS = 16;
constexpr uint32_t LO_COLS = 128;
constexpr uint32_t GO_CORE_BYTES = Q_TILE * HEAD_DIM * sizeof(float);
constexpr uint32_t UB_BYTES = 248 * 1024;
constexpr uint32_t LO_TILE_BYTES = Q_TILE * HEAD_DIM * sizeof(float);
constexpr uint32_t LO_CORE_BYTES = LO_TILE_BYTES * LO_STAGES;
constexpr uint32_t L1_BYTES =
    Q_TILE * HEAD_DIM * 2 * Q_STAGES + KV_TILE * L1_DIM_SLICE * 2 * K_STAGES +
    Q_TILE * KV_TILE * 2 * P_STAGES + KV_TILE * HEAD_DIM * 2 * V_STAGES;
static_assert(L1_BYTES == 416 * 1024);

// All offsets are per AIV, in bytes. The shared auxiliary area is budgeted
// here; its concrete mask/scratch lifetimes belong to the D512 epilogue.
struct UbLayout {
    uint32_t sOffset;
    uint32_t pOffset;
    uint32_t goOffset;
    uint32_t lmOffset;
    uint32_t gmOffset;
    uint32_t dmOffset;
    uint32_t llOffset;
    uint32_t glOffset;
    uint32_t lseOffset;
    uint32_t auxOffset;
    uint32_t loOffset;
    uint32_t loRows;
    uint32_t loStageBytes;
    uint32_t endOffset;
};

constexpr UbLayout MakeUbLayout(uint32_t auxBytes = VECTOR_ROWS * KV_TILE * 2, uint32_t scratchReserve = 2 * 1024)
{
    UbLayout b{};
    b.sOffset = 0;
    b.pOffset = S_STAGE_BYTES * S_STAGES;
    b.goOffset = b.pOffset + P_STAGE_BYTES * UB_P_STAGES;
    b.lmOffset = b.goOffset + GO_ROWS * HEAD_DIM * sizeof(float);
    b.gmOffset = b.lmOffset + VECTOR_ROWS * sizeof(float);
    b.dmOffset = b.gmOffset + VECTOR_ROWS * sizeof(float);
    b.llOffset = b.dmOffset + VECTOR_ROWS * sizeof(float) * P_STAGES;
    b.glOffset = b.llOffset + VECTOR_ROWS * sizeof(float);
    b.lseOffset = b.glOffset + VECTOR_ROWS * sizeof(float);
    // Final LSE reuses the output epilogue workspace, as in the existing
    // FA3 implementation. Keep both causal/window mask work areas resident.
    b.auxOffset = b.lseOffset;
    b.loOffset = b.auxOffset + auxBytes;
    const uint32_t available = b.loOffset + scratchReserve <= UB_BYTES ? UB_BYTES - b.loOffset - scratchReserve : 0;
    b.loRows = available / (HEAD_DIM * sizeof(float) * LO_STAGES);
    b.loRows = b.loRows < GO_ROWS ? b.loRows : GO_ROWS;
    b.loStageBytes = b.loRows * HEAD_DIM * sizeof(float);
    b.endOffset = b.loOffset + b.loStageBytes * LO_STAGES;
    return b;
}
constexpr UbLayout DEFAULT_UB = MakeUbLayout();
static_assert(DEFAULT_UB.loRows == 16);
static_assert(VECTOR_ROWS * LO_COLS * sizeof(float) == DEFAULT_UB.loStageBytes);
static_assert(DEFAULT_UB.endOffset + 2 * 1024 <= UB_BYTES);

// Logical tile sequence, not the outer prelaunch loop index.
constexpr uint32_t PSlot(uint32_t sequence)
{
    return sequence % P_STAGES;
}
constexpr uint32_t LoSlot(uint32_t sequence)
{
    return sequence % LO_STAGES;
}
constexpr uint64_t LoByteOffset(uint32_t core, uint32_t sequence)
{
    return static_cast<uint64_t>(core) * LO_CORE_BYTES + static_cast<uint64_t>(LoSlot(sequence)) * LO_TILE_BYTES;
}

// Match the existing GQA task packing, with a 128-row budget for D512.
constexpr uint32_t QHeadTile(uint32_t qLength, uint32_t groupSize)
{
    uint32_t heads = qLength == 0 ? Q_TILE : (Q_TILE / qLength) / 2 * 2;
    heads = heads < groupSize ? heads : groupSize;
    if (heads > 2) {
        heads = heads / 2 * 2;
    }
    return heads == 0 ? 1 : heads;
}

template <class Tiling>
inline void ApplyTiling(Tiling& t)
{
    t.qBaseTile = Q_TILE;
    t.kvBaseTile = KV_TILE;
    t.qkL1TileM = Q_TILE;
    t.qkL1TileN = KV_TILE;
    t.qkL1TileKLeft = HEAD_DIM;
    t.qkL1TileKRight = L1_DIM_SLICE;
    t.pvL1TileM = Q_TILE;
    t.pvL1TileN = HEAD_DIM;
    t.pvL1TileKLeft = KV_TILE;
    t.pvL1TileKRight = KV_TILE;
    t.qL1BufNum = Q_STAGES;
    t.kL1BufNum = K_STAGES;
    t.pL1BufNum = P_STAGES;
    t.vL1BufNum = V_STAGES;
}
} // namespace flash_attn::d512
#endif
