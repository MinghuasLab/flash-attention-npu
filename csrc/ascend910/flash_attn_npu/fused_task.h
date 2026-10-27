/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 *
 * Shared Host/AICPU/AICore task geometry, adapted from minFA's grouped tasks.
 */
#pragma once

#include <cstdint>

#ifdef __CCE__
#define FA_TASK_INLINE __forceinline__[host, aicore]
#else
#define FA_TASK_INLINE inline
#endif

namespace fa_fused {
constexpr uint32_t Q_ROWS = 128;
constexpr uint32_t SUB_CORE_ROWS = Q_ROWS / 2;

FA_TASK_INLINE uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
FA_TASK_INLINE uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

FA_TASK_INLINE uint32_t HeadTile(uint32_t qRows, uint32_t groupSize, uint32_t nCap, uint32_t qHeads)
{
    uint32_t tile = Min(nCap, Q_ROWS / (CeilDiv(qRows, 16) * 16));
    tile = tile > 0 ? tile : 1;
    if (tile >= groupSize) {
        tile = tile / groupSize * groupSize;
    } else {
        while (tile > 1 && groupSize % tile != 0) {
            --tile;
        }
    }
    // v2 splits whole heads between vectors (except a single head). Each
    // vector owns 64 state rows; an odd head count must fit the larger half.
    while (tile > 1 && CeilDiv(tile, 2) * qRows > SUB_CORE_ROWS) {
        if (tile > groupSize) {
            tile -= groupSize;
        } else {
            do {
                --tile;
            } while (tile > 1 && groupSize % tile != 0);
        }
    }
    // Keep the same number of tasks while balancing the groups per task.
    if (tile >= groupSize) {
        uint32_t kvHeads = qHeads / groupSize;
        uint32_t groupsPerTask = tile / groupSize;
        if (kvHeads > groupsPerTask) {
            tile = Min(tile, CeilDiv(kvHeads, CeilDiv(kvHeads, groupsPerTask)) * groupSize);
        }
    }
    return tile;
}

struct TaskShape {
    uint32_t normQBlocks;
    uint32_t normHeadTile;
    uint32_t normHeadBlocks;
    uint32_t normTasks;
    uint32_t tailRows;
    uint32_t tailHeadTile;
    uint32_t tailHeadBlocks;
    uint32_t totalTasks;
};

FA_TASK_INLINE TaskShape MakeTaskShape(uint32_t qSeqlen, uint32_t qHeads, uint32_t kvHeads,
                                      uint32_t batch, uint32_t coreNum, bool fixedLayout)
{
    TaskShape shape{};
    uint32_t groupSize = qHeads / kvHeads;
    uint32_t nCap = batch >= coreNum ? qHeads : groupSize;
    shape.normQBlocks = qSeqlen / Q_ROWS;
    shape.normHeadTile = 1;
    shape.normHeadBlocks = qHeads;
    shape.normTasks = shape.normQBlocks * shape.normHeadBlocks;
    shape.tailRows = qSeqlen % Q_ROWS;
    shape.tailHeadTile = shape.tailRows ? HeadTile(shape.tailRows, groupSize, nCap, qHeads) : 1;
    shape.tailHeadBlocks = shape.tailRows ? CeilDiv(qHeads, shape.tailHeadTile) : 0;
    // Avoid repeatedly assigning the expensive MHA tail to the same cores
    // when one batch barely fills a core wave (minFA 1881c78).
    if (fixedLayout && groupSize == 1 && shape.tailHeadBlocks > 0 &&
        shape.normTasks + shape.tailHeadBlocks <= coreNum + 2) {
        shape.tailHeadTile = Min(shape.tailHeadTile, 2);
        shape.tailHeadBlocks = CeilDiv(qHeads, shape.tailHeadTile);
    }
    shape.totalTasks = shape.normTasks + shape.tailHeadBlocks;
    return shape;
}
} // namespace fa_fused

#undef FA_TASK_INLINE
