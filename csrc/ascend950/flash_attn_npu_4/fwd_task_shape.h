// Copyright (c) 2026, Minghua Shen.
#pragma once

#include "fwd_resource_plan.h"

#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
#define FA950_TASK_INLINE __forceinline__[host, aicore]
#else
#define FA950_TASK_INLINE inline
#endif

namespace fa950_task {
FA950_TASK_INLINE uint32_t CeilDiv(uint32_t n, uint32_t d)
{
    return (n + d - 1U) / d;
}

FA950_TASK_INLINE uint32_t HeadTile(uint32_t rows, uint32_t groupSize, uint32_t qTile)
{
    uint32_t tile = rows == 0U ? fa950_plan::MAX_Q_ROWS : (fa950_plan::MAX_Q_ROWS / rows) / 2U * 2U;
    if (qTile < fa950_plan::MAX_Q_ROWS && rows != 0U) {
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

FA950_TASK_INLINE Shape MakeShape(uint32_t qLength, uint32_t groupSize, uint32_t kvHeads, uint32_t qTile,
                                  bool useGqaTailGrouping)
{
    const uint32_t tailRows = qLength % qTile;
    // Group short tails within each KV head. The existing single-block layout
    // already groups heads by its Q length and is also used by FlashDecode.
    const bool separateTail = useGqaTailGrouping && qLength > qTile && tailRows != 0U;
    Shape shape{};
    shape.fullBlocks = separateTail ? qLength / qTile : CeilDiv(qLength, qTile);
    shape.headsPerFullTask = HeadTile(qLength, groupSize, qTile);
    shape.fullHeadTasks = kvHeads * CeilDiv(groupSize, shape.headsPerFullTask);
    shape.fullTasks = shape.fullBlocks * shape.fullHeadTasks;
    shape.headsPerTailTask = separateTail ? HeadTile(tailRows, groupSize, qTile) : 1U;
    shape.totalTasks = shape.fullTasks;
    if (separateTail) {
        shape.totalTasks += kvHeads * CeilDiv(groupSize, shape.headsPerTailTask);
    }
    return shape;
}
} // namespace fa950_task

#undef FA950_TASK_INLINE
