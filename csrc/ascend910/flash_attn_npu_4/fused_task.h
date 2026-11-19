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
// Geometry that only the rare branches need is kept out of line: the AICore
// inlines this header into its per-batch geometry call, and any growth of that
// instruction stream costs 7-45% on hot shapes (see MakeMhaTaskShape).
#define FA_TASK_NOINLINE static __attribute__((noinline))[host, aicore]
#else
#define FA_TASK_INLINE inline
#define FA_TASK_NOINLINE static inline
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
        // Exception, decode (a single Q row per task): the fixed per-task cost
        // dominates there, and k whole groups keep the per-group L1 segment and the
        // KV scan count while needing 1/k the tasks. Measured H16/kv4 q1 kv4096 B32:
        // 2 groups 0.314 ms vs 1 group 0.342 ms. With more Q rows the vector split
        // and the pipeline favour one group ("one KV group per task" invariant).
        if (qRows == 1U) {
            const uint32_t groups = (Q_ROWS / (CeilDiv(qRows, 16) * 16)) / groupSize;
            return groups > 0U ? groups * groupSize : groupSize;
        }
        if (qRows <= 16U) {
            // Short-q GQA (decode / cross-attention with a few Q rows): one task does
            // so little work that its prologue/epilogue dominates, so it should keep
            // swallowing whole KV groups as long as the Q row budget allows even
            // though that saves no KV traffic. A task of k groups costs
            // k * RoundUp16(qRows * groupSize) rows in L1 Q. Measured T5 cross
            // (q3 kv<=276 H16/kv8 B512, TND): 1 group 0.950 ms -> 8 groups 0.62 ms
            // (-35%, level with minFA); the same tile is used by HeadTile's caller
            // only when the batch keeps the cores busy (see MakeTaskShape).
            const uint32_t segRows = CeilDiv(qRows * groupSize, 16) * 16;
            const uint32_t groups = Q_ROWS / segRows;
            if (groups > 1U) {
                return groups * groupSize;
            }
        }
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

// groupSize == 1 (MHA) head tile: the pre-fuse upstream rule, kept as its own
// small function.
// The generic HeadTile() above is written for GQA (whole KV groups, in-group
// slices, multi-group packing) and every one of those rules is a no-op when
// groupSize == 1, but the code is still inlined into every AICore call site.
FA_TASK_INLINE uint32_t HeadTileMha(uint32_t qRows, uint32_t nCap, uint32_t qHeads)
{
    uint32_t tile = Min(nCap, Q_ROWS / (CeilDiv(qRows, 16) * 16));
    tile = tile > 0 ? tile : 1;
    // v2 splits whole heads between vectors (except a single head). Each
    // vector owns 64 state rows; an odd head count must fit the larger half.
    // groupSize == 1 makes the group divisibility guard a no-op.
    while (tile > 1 && CeilDiv(tile, 2) * qRows > SUB_CORE_ROWS) {
        --tile;
    }
    // Keep the same number of tasks while balancing the groups per task.
    if (tile >= 1U) {
        const uint32_t kvHeads = qHeads;
        const uint32_t groupsPerTask = tile;
        if (kvHeads > groupsPerTask) {
            tile = Min(tile, CeilDiv(kvHeads, CeilDiv(kvHeads, groupsPerTask)));
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

// Fixed-layout MHA tail rotation: rare, gcd based, kept out of line so the
// inline per-batch geometry of the AICore stays at the upstream size (the call
// is unexecuted for varlen). Returns 0 to keep the wide tail tile, otherwise
// newTile | (newHeadBlocks << 16).
FA_TASK_NOINLINE uint32_t MhaTailShrinkPick(uint32_t qHeads, uint32_t batch, uint32_t coreNum, uint32_t normTasks,
                                            uint32_t tailHeadTile, uint32_t tailHeadBlocks)
{
    const uint32_t curTasks = normTasks + tailHeadBlocks;
    const uint32_t newTile = Min(tailHeadTile, 2U);
    if (newTile >= tailHeadTile) {
        return 0U;
    }
    const uint32_t newBlocks = CeilDiv(qHeads, newTile);
    const uint32_t newTasks = normTasks + newBlocks;
    const bool starved = batch * curTasks < 8U * coreNum;
    if (starved || Gcd(newTasks, coreNum) < Gcd(curTasks, coreNum)) {
        return newTile | (newBlocks << 16);
    }
    return 0U;
}

// groupSize == 1 (MHA) geometry: the pre-fuse upstream rule, in its own small
// inline function.
// The generic HeadTile/MakeTaskShape pair above is written for GQA (whole KV
// groups, in-group slices, multi-group packing). Every one of those rules is a
// no-op when groupSize == 1, but the AICore still inlines the code into its
// per-batch geometry call, which it runs once per *batch* inside the TND task
// loop.
FA_TASK_INLINE TaskShape MakeMhaTaskShape(uint32_t qSeqlen, uint32_t qHeads, uint32_t batch, uint32_t coreNum,
                                          bool fixedLayout)
{
    TaskShape shape{};
    const uint32_t nCap = batch >= coreNum ? qHeads : 1U;
    shape.normQBlocks = qSeqlen / Q_ROWS;
    shape.normHeadTile = 1;
    shape.normHeadBlocks = qHeads;
    shape.normTasks = shape.normQBlocks * shape.normHeadBlocks;
    shape.tailRows = qSeqlen % Q_ROWS;
    shape.tailHeadTile = shape.tailRows ? HeadTileMha(shape.tailRows, nCap, qHeads) : 1;
    shape.tailHeadBlocks = shape.tailRows ? CeilDiv(qHeads, shape.tailHeadTile) : 0;
    // Avoid repeatedly assigning the expensive MHA tail to the same cores.
    // Only shrink the tail tile when the shrink actually breaks the rotation
    // (gcd(T, coreNum) drops); a fixed "T <= coreNum + 2" window also split
    // tails whose rotation was already fine, costing 5-20% on short sequences
    if (fixedLayout && shape.tailHeadBlocks > 0U) {
        const uint32_t pick =
            MhaTailShrinkPick(qHeads, batch, coreNum, shape.normTasks, shape.tailHeadTile, shape.tailHeadBlocks);
        if (pick != 0U) {
            shape.tailHeadTile = pick & 0xFFFFU;
            shape.tailHeadBlocks = pick >> 16;
        }
    }
    shape.totalTasks = shape.normTasks + shape.tailHeadBlocks;
    return shape;
}

// groupSize > 1 (GQA) geometry: never inlined into the AICore task loop, the
// per-batch call is negligible there.
FA_TASK_NOINLINE TaskShape MakeGqaTaskShape(uint32_t qSeqlen, uint32_t qHeads, uint32_t kvHeads, uint32_t batch,
                                            uint32_t coreNum, bool fixedLayout)
{
    const uint32_t groupSize = qHeads / kvHeads;
    const uint32_t nCap = batch >= coreNum ? qHeads : groupSize;
    TaskShape shape{};
    shape.normQBlocks = qSeqlen / Q_ROWS;
    shape.normHeadTile = 1;
    shape.normHeadBlocks = qHeads;
    shape.normTasks = shape.normQBlocks * shape.normHeadBlocks;
    shape.tailRows = qSeqlen % Q_ROWS;
    shape.tailHeadTile = shape.tailRows ? HeadTile(shape.tailRows, groupSize, nCap, qHeads) : 1;
    shape.tailHeadBlocks = shape.tailRows ? HeadBlocks(shape.tailHeadTile, groupSize, qHeads) : 0;
    // Short-q GQA may fuse several whole KV groups into one task (QNCapacity) to
    // amortise the per-task prologue/epilogue. That only pays while the batch
    // dimension still fills the grid: keep the wide tile only when batch * (tasks
    // per batch) stays above a few core waves, otherwise fall back to the
    // "one KV group per task" invariant (a single task holding several KV heads
    // serialises their scans and the wave quantisation eats the gain).
    if (shape.tailRows > 1U && shape.tailHeadBlocks > 0U && shape.tailHeadTile > groupSize) {
        if (batch * shape.tailHeadBlocks < 8U * coreNum) {
            shape.tailHeadTile = groupSize;
            shape.tailHeadBlocks = HeadBlocks(groupSize, groupSize, qHeads);
        }
    }
    shape.totalTasks = shape.normTasks + shape.tailHeadBlocks;
    return shape;
}

FA_TASK_INLINE TaskShape MakeTaskShape(uint32_t qSeqlen, uint32_t qHeads, uint32_t kvHeads, uint32_t batch,
                                       uint32_t coreNum, bool fixedLayout)
{
    // MHA (groupSize == 1, incl. kvHeads == 0) keeps the upstream rule in its own
    // small function so the AICore's per-batch call does not inline the GQA code.
    if (kvHeads == 0U || kvHeads == qHeads) {
        return MakeMhaTaskShape(qSeqlen, qHeads, batch, coreNum, fixedLayout);
    }
    return MakeGqaTaskShape(qSeqlen, qHeads, kvHeads, batch, coreNum, fixedLayout);
}
} // namespace fa_fused

#undef FA_TASK_INLINE
