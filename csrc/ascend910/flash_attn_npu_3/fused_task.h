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

FA_TASK_INLINE uint32_t Min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
FA_TASK_INLINE uint32_t Max(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}
FA_TASK_INLINE uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

// Task rotation with taskIdx += coreNum: one core revisits the same task class
// every coreNum / gcd(T, coreNum) tasks, so the number of tasks per batch T
// decides whether a heavy tail stays pinned to a few cores.
FA_TASK_INLINE uint32_t Gcd(uint32_t a, uint32_t b)
{
    while (b != 0) {
        uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

// Number of head blocks (tasks per Q row-block) for a head tile.
//   tile >= groupSize: a task covers whole KV groups, blocks = ceil(qHeads / tile)
//   tile <  groupSize: a task stays inside one KV group and covers the
//                      tile-th slice of it, so each group yields
//                      ceil(groupSize / tile) blocks (e.g. group 3, tile 2 -> 2+1)
FA_TASK_INLINE uint32_t HeadBlocks(uint32_t headTile, uint32_t groupSize, uint32_t qHeads)
{
    if (headTile >= groupSize) {
        return CeilDiv(qHeads, headTile);
    }
    return (qHeads / groupSize) * CeilDiv(groupSize, headTile);
}

// Head count a task may fuse, limited by the L1 Q budget (Q_ROWS rows per task).
// Q is copied into L1 as 16-row ND2NZ blocks, and the kernel pays that rounding
// once per *segment*, not once per head:
//     segL1Stride = RoundUp<16>(qSBlockSize * segN) * embedRound,  segN = min(tile, groupSize)
// (mha_fwd_kvcache.cpp, "segL1Stride"; budget = Q_ROWS * embedRound, see maxQL1Size).
//
// Decode (qRows small) must stop at ONE KV group: once tile >= groupSize the number
// of KV scans per batch is qHeads/groupSize = kvHeads regardless of tile, so fusing
// more groups saves no KV traffic; it only adds wave-quantisation loss (task count
// collapses, e.g. H64/kv8/B64: 512 tasks -> 64, util 98% -> 80%) and serialises
// several KV heads inside one core. Measured: v3 +18~26%, v2 +12.8% on that shape.
// Going from ceil(groupSize/8) in-group slices to one whole group is still a win
// (JYXH q=1: 8 -> 12 halves the KV scans, -21%/-43%), so the tile must never drop
// below one group when the group fits in L1.
FA_TASK_INLINE uint32_t QNCapacity(uint32_t qRows, uint32_t groupSize)
{
    if (qRows == 0 || groupSize == 0) {
        return 1;
    }
    if (groupSize == 1) {
        // MHA: every head owns its own KV head, so the 16-row segment rounding is
        // paid per head (unchanged from the pre-fuse rule).
        return Q_ROWS / (CeilDiv(qRows, 16) * 16);
    }
    // GQA: one KV group shares one L1 segment, so a whole group costs a single
    // RoundUp16(qRows*groupSize) once. Stop at one group (see above).
    if (CeilDiv(qRows * groupSize, 16) * 16 <= Q_ROWS) {
        return groupSize;
    }
    // The group does not fit: fall back to (and never go below) the in-group even
    // split that the unfused GetQNBlockTile uses.
    return Min(groupSize, Max(Q_ROWS / (CeilDiv(qRows, 16) * 16), (Q_ROWS / qRows) / 2 * 2));
}

FA_TASK_INLINE uint32_t HeadTile(uint32_t qRows, uint32_t groupSize, uint32_t nCap, uint32_t qHeads)
{
    uint32_t tile = Min(nCap, QNCapacity(qRows, groupSize));
    tile = tile > 0 ? tile : 1;
    if (tile >= groupSize) {
        tile = tile / groupSize * groupSize;
    }
    // A tile below groupSize no longer has to divide groupSize: the kernel
    // addresses "the subIdx-th slice of KV group kvIdx" (see HeadBlocks), so
    // e.g. groupSize=3, tile=2 is a valid 2+1 split. Forcing divisibility
    // collapsed prime group sizes to tile=1, doubling the task count and
    // re-reading every KV head per single-head task.
    // v2 splits whole heads between vectors (except a single head). Each
    // vector owns 64 state rows; an odd head count must fit the larger half.
    while (tile > 1 && CeilDiv(tile, 2) * qRows > SUB_CORE_ROWS) {
        if (tile > groupSize) {
            tile -= groupSize;
        } else {
            --tile; // In-group slicing: any tile is a legal slice width.
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

FA_TASK_INLINE TaskShape MakeTaskShape(uint32_t qSeqlen, uint32_t qHeads, uint32_t kvHeads, uint32_t batch,
                                       uint32_t coreNum, bool fixedLayout)
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
    shape.tailHeadBlocks = shape.tailRows ? HeadBlocks(shape.tailHeadTile, groupSize, qHeads) : 0;
    // Avoid repeatedly assigning the expensive MHA tail to the same cores.
    // Only shrink the tail tile when the shrink actually breaks the rotation
    // (gcd(T, coreNum) drops); a fixed "T <= coreNum + 2" window also split
    // tails whose rotation was already fine, costing 5-20% on short sequences
    // (minFA 0e7aa71).
    if (fixedLayout && groupSize == 1 && shape.tailHeadBlocks > 0) {
        const uint32_t curTasks = shape.normTasks + shape.tailHeadBlocks;
        const uint32_t newTile = Min(shape.tailHeadTile, 2);
        if (newTile < shape.tailHeadTile) {
            const uint32_t newBlocks = HeadBlocks(newTile, groupSize, qHeads);
            const uint32_t newTasks = shape.normTasks + newBlocks;
            if (Gcd(newTasks, coreNum) < Gcd(curTasks, coreNum)) {
                shape.tailHeadTile = newTile;
                shape.tailHeadBlocks = newBlocks;
            }
        }
    }
    shape.totalTasks = shape.normTasks + shape.tailHeadBlocks;
    return shape;
}
} // namespace fa_fused

#undef FA_TASK_INLINE
