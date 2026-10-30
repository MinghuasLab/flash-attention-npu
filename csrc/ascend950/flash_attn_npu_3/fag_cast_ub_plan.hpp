#ifndef FLASH_ATTN_NPU_FAG_CAST_UB_PLAN_HPP
#define FLASH_ATTN_NPU_FAG_CAST_UB_PLAN_HPP

#include <cstdint>

namespace FAGTiling950 {

struct CastUbPlan {
    uint64_t pipelineBytes = 0;
    uint64_t inputBytes = 0;
    uint64_t outputBytes = 0;
    uint32_t chunkRows = 0;
};

inline uint64_t CastUbRoundUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

// Host planner. Mirrors the kernel's two pipeline slots and separately aligned
// float-input / 16-bit-output tail buffers. Kernel consumes chunkRows via tiling.
inline CastUbPlan PlanBn2s2CastUb(uint64_t qTile, uint64_t kvTile, uint64_t qkDim, uint64_t vDim, uint64_t ubBytes)
{
    CastUbPlan plan;
    if (qTile == 0 || kvTile == 0 || qkDim == 0 || vDim == 0) {
        return plan;
    }
    const uint64_t rowsPerSub = CastUbRoundUp(qTile, 16) / 2;
    const uint64_t mmBytes = rowsPerSub * kvTile * sizeof(float);
    const uint64_t maskBytes = rowsPerSub * kvTile;
    const uint64_t lseBytes = CastUbRoundUp(rowsPerSub, 8) * 8 * sizeof(float);
    const uint64_t pBytes = (rowsPerSub + 1) * kvTile * 2;
    const uint64_t deltaBytes = rowsPerSub * 8 * sizeof(float);
    const uint64_t pOffset = CastUbRoundUp(CastUbRoundUp(2 * mmBytes + maskBytes, 32) + lseBytes, 32);
    const uint64_t dsOffset = CastUbRoundUp(pOffset + pBytes, 32);
    const uint64_t deltaOffset = CastUbRoundUp(dsOffset + pBytes, 32);
    plan.pipelineBytes = 2 * CastUbRoundUp(deltaOffset + deltaBytes, 32);
    if (plan.pipelineBytes >= ubBytes) {
        return plan;
    }
    const uint64_t maxDim = qkDim > vDim ? qkDim : vDim;
    const uint64_t alignedDim = CastUbRoundUp(maxDim, 8);
    const uint64_t available = ubBytes - plan.pipelineBytes;
    const uint32_t maxRows = static_cast<uint32_t>(kvTile < 32 ? kvTile : 32);
    for (uint32_t rows = maxRows; rows > 0; --rows) {
        const uint64_t inBytes = CastUbRoundUp(rows * alignedDim * sizeof(float), 32);
        const uint64_t outBytes = CastUbRoundUp(rows * maxDim * 2, 32);
        if (inBytes + outBytes <= available) {
            plan.chunkRows = rows;
            plan.inputBytes = inBytes;
            plan.outputBytes = outBytes;
            break;
        }
    }
    return plan;
}

} // namespace FAGTiling950
#endif
