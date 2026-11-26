// Copyright (c) 2026, Minghua Shen.
#pragma once

#include <cstdint>

#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
#define FA950_TASK_INLINE __forceinline__[host, aicore]
#else
#define FA950_TASK_INLINE inline
#endif

namespace fa950_task {
// Nonempty FP16/BF16 segments are bounded in both metadata producers.
constexpr uint32_t MAX_Q_ROWS = 128U;
constexpr uint32_t KV_TILE = 128U;

struct QkL1Layout {
    uint32_t qSlots;
    uint32_t kRows;
    uint32_t kDepth;
};

// Each K slot holds one KV tile. For equal-width 16-bit inputs at D=256,
// Q/K/V use 128 KiB each and P uses 96 KiB (480 KiB of the 512 KiB L1).
inline QkL1Layout MakeQkL1Layout(uint32_t dq, uint32_t dv, uint32_t kvTile, bool fp8)
{
    if (dq == dv && dq > 0U && dq <= 256U) {
        return {2U, kvTile, (dq + 15U) / 16U * 16U};
    }
    return {1U, 256U, fp8 ? 128U : 192U};
}

FA950_TASK_INLINE uint32_t CeilDiv(uint32_t n, uint32_t d)
{
    return (n + d - 1U) / d;
}

FA950_TASK_INLINE uint32_t HeadTile(uint32_t rows, uint32_t groupSize, uint32_t qTile)
{
    uint32_t tile = rows == 0U ? MAX_Q_ROWS : (MAX_Q_ROWS / rows) / 2U * 2U;
    if (qTile < MAX_Q_ROWS && rows != 0U) {
        const uint32_t capacity = qTile / rows;
        tile = tile < capacity ? tile : capacity;
    }
    tile = tile < groupSize ? tile : groupSize;
    if (tile > 2U) {
        tile = tile / 2U * 2U;
    }
    return tile == 0U ? 1U : tile;
}

struct Shape {
    uint32_t fullBlocks;
    uint32_t headsPerFullTask;
    uint32_t fullHeadTasks;
    uint32_t fullTasks;
    uint32_t headsPerTailTask;
    uint32_t totalTasks;
};

// Cap the number of distinct KV heads one task may own without losing a core wave.
FA950_TASK_INLINE uint32_t HeadGroupBudget(uint32_t batch, uint32_t kvHeads, uint32_t cores)
{
    const uint64_t parallel = static_cast<uint64_t>(batch) * kvHeads;
    const uint32_t groups = cores == 0U ? 1U : static_cast<uint32_t>(parallel / cores);
    return groups < 2U ? 1U : (groups < kvHeads ? groups : kvHeads);
}

inline uint32_t MultiTileBudget(uint32_t budget, uint32_t groupSize, bool multipleKvTiles)
{
    if (groupSize != 1U || !multipleKvTiles || budget <= 4U) {
        return budget;
    }
    // Retain four independent waves of head work on multi-panel MHA. Each
    // distinct KV head in a task is issued serially, even when its Q is short.
    const uint32_t perWave = CeilDiv(budget, 4U);
    return perWave < 4U ? perWave : 4U;
}

FA950_TASK_INLINE Shape MakeShape(uint32_t qLength, uint32_t groupSize, uint32_t kvHeads, uint32_t qTile,
                                  uint32_t headGroupBudget)
{
    const uint32_t tailRows = qLength % qTile;
    const bool separateTail = tailRows != 0U && (headGroupBudget > 1U || (headGroupBudget == 1U && qLength > qTile));
    Shape shape{};
    shape.fullBlocks = separateTail ? qLength / qTile : CeilDiv(qLength, qTile);
    shape.headsPerFullTask = HeadTile(qLength, groupSize, qTile);
    shape.fullHeadTasks = kvHeads * CeilDiv(groupSize, shape.headsPerFullTask);
    shape.fullTasks = shape.fullBlocks * shape.fullHeadTasks;
    shape.totalTasks = shape.fullTasks;
    if (separateTail) {
        shape.headsPerTailTask = HeadTile(tailRows, groupSize, qTile);
        // Q and P in L1 need an independently aligned slice per KV head.
        // Groups which do not fit together keep the original within-head schedule.
        uint32_t groups = 1U;
        if (headGroupBudget > 1U && groupSize <= qTile / tailRows) {
            const uint32_t paddedRows = CeilDiv(tailRows * groupSize, 16U) * 16U;
            const uint32_t capacity = qTile / paddedRows;
            const uint32_t limit = capacity < headGroupBudget ? capacity : headGroupBudget;
            groups = CeilDiv(kvHeads, CeilDiv(kvHeads, limit));
        }
        if (groups > 1U) {
            shape.headsPerTailTask = groups * groupSize;
            shape.totalTasks += CeilDiv(kvHeads, groups);
        } else {
            shape.totalTasks += kvHeads * CeilDiv(groupSize, shape.headsPerTailTask);
        }
    }
    return shape;
}

// Old cached metadata stores a boolean fusedCrossGroup in this word. Its
// producer used a different tail policy, even when no KV heads were fused.
namespace legacy {
FA950_TASK_INLINE uint32_t Min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
FA950_TASK_INLINE uint32_t Align16(uint32_t a)
{
    return CeilDiv(a, 16U) * 16U;
}

FA950_TASK_INLINE uint32_t LegacyHeadTile(uint32_t rows, uint32_t group, uint32_t qTile)
{
    uint32_t tile = rows == 0U ? 128U : (128U / rows) / 2U * 2U;
    if (qTile < 128U && rows != 0U) {
        uint32_t cap = 64U / rows;
        tile = Min(tile, cap > 0U ? cap : 1U);
    }
    tile = Min(tile, group);
    if (tile > 2U)
        tile = tile / 2U * 2U;
    return tile > 0U ? tile : 1U;
}

struct Shape {
    uint32_t fullBlocks, fullHeadTile, fullHeadTasks, fullTasks;
    uint32_t tailRows, tailHeadTile, tailHeadTasks, totalTasks;
};

FA950_TASK_INLINE uint32_t HeadTasks(uint32_t heads, uint32_t group, uint32_t tile)
{
    // tile >= group merges whole KV groups (or several of them) into one task, so
    // every task owns `tile` consecutive heads and the group arithmetic drops
    // out. This also keeps the per-task decode free of the extra divisions on the
    // single-KV-head-tile path.
    return tile >= group ? CeilDiv(heads, tile) : (heads / group) * CeilDiv(group, tile);
}

FA950_TASK_INLINE Shape MakeShape(uint32_t q, uint32_t heads, uint32_t kvHeads, uint32_t qTile, uint32_t batch,
                                  uint32_t cores, bool fixedLayout, bool quantized = false, bool allowCrossGroup = true,
                                  bool tinyKv = false)
{
    const uint32_t group = heads / kvHeads;
    Shape s{};
    s.fullBlocks = quantized ? CeilDiv(q, qTile) : q / qTile;
    s.fullHeadTile = quantized ? LegacyHeadTile(q, group, qTile) : 1U;
    s.fullHeadTasks = HeadTasks(heads, group, s.fullHeadTile);
    s.fullTasks = s.fullBlocks * s.fullHeadTasks;
    s.tailRows = quantized ? 0U : q % qTile;
    s.tailHeadTile = s.tailRows ? LegacyHeadTile(s.tailRows, group, qTile) : 1U;
    // On 950, serial segment GEMMs can outweigh vector fusion for a long-Q
    // tail. Keep cross-group fusion for short Q; GQA tails still merge within
    // their KV group. Cross-group fusion merges *distinct* KV heads, so for
    // MHA (group == 1) it only pays off when a task carries a single KV tile:
    // then the per-task fixed cost (offsets, mask setup, task bookkeeping) is
    // what dominates, and merging heads cuts the task count instead of turning
    // a load-bound kernel into a slower serial-segment one (measured 0.73 ->
    // 1.02 ms for short-Q x long-KV MHA). Long-KV MHA keeps one head per task.
    if (s.tailRows && q < qTile && batch >= cores && allowCrossGroup && (group > 1U || tinyKv)) {
        // Each KV segment has its own 16-row-aligned Q/P storage. Effective
        // rows remain compact in UB; both AIVs are bounded by 64 state rows.
        const uint32_t groups = Min(kvHeads, qTile / Align16(s.tailRows * group));
        if (groups > 1U) {
            // Task size trades per-task fixed cost against last-wave fill: a
            // task always reads every KV head it owns exactly once, so tile
            // granularity changes nothing but quantisation. Walk from the
            // largest tile the 128-row budget allows down to one KV group and
            // keep the first size whose launch fills at least 90% of the slots.
            // Ragged tiles (heads not a multiple of the tile) are skipped: the
            // short last task leaves its core idle for a whole wave.
            const uint32_t maxGroups = CeilDiv(kvHeads, CeilDiv(kvHeads, groups));
            uint32_t tile = group;
            for (uint32_t k = maxGroups; k >= 1U; --k) {
                const uint32_t cand = k * group;
                if (k > 1U && heads % cand != 0U) {
                    continue;
                }
                const uint32_t tasks = batch * CeilDiv(heads, cand);
                const uint32_t waves = CeilDiv(tasks, cores);
                if (static_cast<uint64_t>(tasks) * 10U >= static_cast<uint64_t>(waves) * cores * 9U) {
                    tile = cand;
                    break;
                }
            }
            s.tailHeadTile = tile;
        }
    }
    s.tailHeadTasks = s.tailRows ? HeadTasks(heads, group, s.tailHeadTile) : 0U;
    if (fixedLayout && group == 1U && s.tailHeadTasks && s.fullTasks + s.tailHeadTasks <= cores + 2U) {
        s.tailHeadTile = Min(s.tailHeadTile, 2U);
        s.tailHeadTasks = HeadTasks(heads, group, s.tailHeadTile);
    }
    s.totalTasks = s.fullTasks + s.tailHeadTasks;
    return s;
}

} // namespace legacy

FA950_TASK_INLINE Shape DecodeShape(uint32_t q, uint32_t group, uint32_t kvHeads, uint32_t qTile,
                                    uint32_t encodedBudget, uint32_t batch, uint32_t cores, bool fixedLayout,
                                    bool tinyKv)
{
    if (encodedBudget > 1U) {
        return MakeShape(q, group, kvHeads, qTile, encodedBudget - 2U);
    }
    const auto s = legacy::MakeShape(q, group * kvHeads, kvHeads, qTile, batch, cores, fixedLayout, false,
                                     encodedBudget != 0U, tinyKv);
    return {s.fullBlocks, s.fullHeadTile, s.fullHeadTasks, s.fullTasks, s.tailHeadTile, s.totalTasks};
}

struct Task {
    uint32_t qTileIndex, firstHead, heads;
};

FA950_TASK_INLINE Task Decode(const Shape& shape, uint32_t index, uint32_t group, uint32_t kvHeads)
{
    const bool tail = index >= shape.fullTasks;
    const uint32_t qTileIndex = tail ? shape.fullBlocks : index / shape.fullHeadTasks;
    const uint32_t tile = tail ? shape.headsPerTailTask : shape.headsPerFullTask;
    const uint32_t headIndex = tail ? index - shape.fullTasks : index % shape.fullHeadTasks;
    if (tile > group) {
        const uint32_t first = headIndex * tile;
        const uint32_t remaining = kvHeads * group - first;
        return {qTileIndex, first, remaining < tile ? remaining : tile};
    }
    const uint32_t tasksPerGroup = CeilDiv(group, tile);
    const uint32_t offset = headIndex % tasksPerGroup * tile;
    const uint32_t remaining = group - offset;
    return {qTileIndex, headIndex / tasksPerGroup * group + offset, remaining < tile ? remaining : tile};
}
} // namespace fa950_task

#undef FA950_TASK_INLINE
