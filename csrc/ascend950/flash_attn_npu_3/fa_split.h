/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASH_ATTN_NPU_950_V3_FA_SPLIT_H
#define FLASH_ATTN_NPU_950_V3_FA_SPLIT_H

#include <cstdint>

#include "tilingdata.h"

namespace fd_tiling {

constexpr uint32_t Q_TILE_CEIL = 128U;
constexpr uint32_t Q_N_SPLIT_ALIGN = 2U;
constexpr uint32_t BASE_KV_SIZE = 128U;
constexpr uint32_t SIZE_OF_16BIT = 2U;
constexpr uint32_t SIZE_OF_32BIT = 4U;
constexpr uint64_t WORKSPACE_ALIGNMENT = 512U;

enum class PlanStatus : uint32_t {
    ENABLED = 0U,
    FALLBACK = 1U,
    INVALID_INPUT = 2U,
    CAPACITY_EXCEEDED = 3U,
    WORKSPACE_EXCEEDED = 4U,
};

struct PlanInput {
    const int32_t *qSeqlenList;
    const int32_t *kvSeqlenList;
    uint32_t batch;
    uint32_t numHeads;
    uint32_t kvHeads;
    uint32_t embeddingSizeV;
    uint32_t blockDim;
    uint32_t numSplits;
    uint32_t qBaseTile;
    uint32_t kvBaseTile;
    bool isTnd;
    bool pagedCache;
    uint64_t workspaceLimit;
};

struct BaseTask {
    uint32_t kvTiles;
    uint32_t rowNum;
};

struct Candidate {
    uint32_t baseTask;
    uint32_t kvBegin;
    uint32_t kvEnd;
    uint64_t cost;
};

inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

inline uint32_t MaxU32(uint32_t lhs, uint32_t rhs)
{
    return lhs > rhs ? lhs : rhs;
}

inline uint32_t CeilDivU32(uint32_t value, uint32_t divisor)
{
    return divisor == 0U ? 0U : (value + divisor - 1U) / divisor;
}

inline uint64_t AlignUpU64(uint64_t value, uint64_t alignment)
{
    return alignment == 0U ? value :
        (value + alignment - 1U) / alignment * alignment;
}

inline uint32_t GetQSBlockTile(uint32_t embeddingSizeV)
{
    return embeddingSizeV > 128U ? 64U : Q_TILE_CEIL;
}

inline uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize,
                               bool restrictMergedRowsForLargeD)
{
    uint32_t tile = qSeqlen == 0U ? Q_TILE_CEIL :
        (Q_TILE_CEIL / qSeqlen) / Q_N_SPLIT_ALIGN * Q_N_SPLIT_ALIGN;
    if (restrictMergedRowsForLargeD && qSeqlen != 0U) {
        const uint32_t maxTile = MaxU32((Q_TILE_CEIL / 2U) / qSeqlen, 1U);
        tile = MinU32(tile, maxTile);
    }
    tile = MinU32(tile, groupSize);
    if (tile > Q_N_SPLIT_ALIGN) {
        tile = tile / Q_N_SPLIT_ALIGN * Q_N_SPLIT_ALIGN;
    }
    return MaxU32(tile, 1U);
}

inline uint32_t PartialCapacityUpperBound(uint32_t blockDim)
{
    const uint32_t activeCores = MinU32(blockDim, MAX_FD_ACTIVE_CORE_NUM);
    const uint32_t baseTasks = MinU32(
        MAX_FD_COMBINE_TASK_NUM,
        blockDim == 0U ? 0U : (3U * blockDim - 1U) / 10U);
    return baseTasks == 0U || activeCores == 0U ?
        0U : baseTasks + activeCores - 1U;
}

inline uint64_t WorkspaceUpperBound(uint64_t pipelineEnd, uint32_t blockDim,
                                    uint32_t embeddingSizeV)
{
    const uint64_t partialCapacity = PartialCapacityUpperBound(blockDim);
    const uint64_t rowCapacity = Q_TILE_CEIL;
    const uint64_t lseSubStride = AlignUpU64((rowCapacity + 1U) / 2U, 8U);
    const uint64_t lseOffset = AlignUpU64(pipelineEnd, WORKSPACE_ALIGNMENT);
    const uint64_t lseSize = partialCapacity * 2U * lseSubStride * SIZE_OF_32BIT;
    const uint64_t oOffset = AlignUpU64(
        lseOffset + lseSize, WORKSPACE_ALIGNMENT);
    return oOffset + partialCapacity * rowCapacity * embeddingSizeV * SIZE_OF_16BIT;
}

inline void ResetFdData(FAInferTilingData &tiling)
{
    tiling.flashDecodeFlag = 0U;
    tiling.fdActiveCoreNum = 0U;
    tiling.fdBaseTaskNum = 0U;
    tiling.fdCombineTaskNum = 0U;
    tiling.fdPartialTaskNum = 0U;
    tiling.fdPartialCapacity = 0U;
    tiling.fdRowCapacity = 0U;
    tiling.fdLseSubStride = 0U;
    tiling.fdCombineBlockDim = 0U;
    tiling.fdPartialLseOffset = 0U;
    tiling.fdPartialOOffset = 0U;
    tiling.fdWorkspaceEnd = 0U;
    for (uint32_t i = 0U; i < MAX_FD_ACTIVE_CORE_NUM; ++i) {
        tiling.fdDecodeSchedules[i] = {-1, -1, -1, -1};
    }
    for (uint32_t i = 0U; i < MAX_FD_COMBINE_TASK_NUM; ++i) {
        tiling.fdCombineSchedules[i] = {-1, -1, -1, 0};
    }
}

inline bool ReadLengths(const PlanInput &input, uint32_t batchIdx,
                        uint32_t &qLen, uint32_t &kvLen)
{
    int64_t qValue = input.qSeqlenList[batchIdx];
    int64_t kvValue = input.kvSeqlenList[batchIdx];
    if (input.isTnd) {
        qValue = static_cast<int64_t>(input.qSeqlenList[batchIdx + 1U]) -
            input.qSeqlenList[batchIdx];
        if (!input.pagedCache) {
            kvValue = static_cast<int64_t>(input.kvSeqlenList[batchIdx + 1U]) -
                input.kvSeqlenList[batchIdx];
        }
    }
    if (qValue <= 0 || kvValue <= 0 ||
        qValue > static_cast<int64_t>(UINT32_MAX) ||
        kvValue > static_cast<int64_t>(UINT32_MAX)) {
        return false;
    }
    qLen = static_cast<uint32_t>(qValue);
    kvLen = static_cast<uint32_t>(kvValue);
    return true;
}

inline Candidate CandidateAt(const BaseTask *bases,
                             const uint32_t *splitCounts,
                             uint32_t baseCount, uint32_t candidateIdx)
{
    for (uint32_t base = 0U; base < baseCount; ++base) {
        const uint32_t count = splitCounts[base];
        if (candidateIdx < count) {
            const uint32_t begin = static_cast<uint32_t>(
                static_cast<uint64_t>(bases[base].kvTiles) * candidateIdx / count);
            const uint32_t end = static_cast<uint32_t>(
                static_cast<uint64_t>(bases[base].kvTiles) *
                (candidateIdx + 1U) / count);
            return {base, begin, end,
                static_cast<uint64_t>(bases[base].rowNum) * (end - begin)};
        }
        candidateIdx -= count;
    }
    return {baseCount, 0U, 0U, 0U};
}

inline PlanStatus BuildPlan(const PlanInput &input, FAInferTilingData &tiling)
{
    ResetFdData(tiling);
    if (input.qSeqlenList == nullptr || input.kvSeqlenList == nullptr ||
        input.batch == 0U || input.numHeads == 0U || input.kvHeads == 0U ||
        input.numHeads % input.kvHeads != 0U || input.blockDim == 0U ||
        input.qBaseTile == 0U || input.kvBaseTile == 0U) {
        return PlanStatus::INVALID_INPUT;
    }

    const uint32_t activeCoreLimit =
        MinU32(input.blockDim, MAX_FD_ACTIVE_CORE_NUM);
    const uint32_t baseTaskLimit = MinU32(
        MAX_FD_COMBINE_TASK_NUM,
        (3U * input.blockDim - 1U) / 10U);
    tiling.fdPartialCapacity = PartialCapacityUpperBound(input.blockDim);

    BaseTask bases[MAX_FD_COMBINE_TASK_NUM];
    uint32_t baseCount = 0U;
    const uint32_t groupSize = input.numHeads / input.kvHeads;
    for (uint32_t batchIdx = 0U; batchIdx < input.batch; ++batchIdx) {
        uint32_t qLen = 0U;
        uint32_t kvLen = 0U;
        if (!ReadLengths(input, batchIdx, qLen, kvLen)) {
            return PlanStatus::INVALID_INPUT;
        }
        const uint32_t qTileNum = CeilDivU32(qLen, input.qBaseTile);
        const uint32_t kvTileNum = CeilDivU32(kvLen, input.kvBaseTile);
        const uint32_t qNBlockTile = GetQNBlockTile(
            qLen, groupSize, input.embeddingSizeV > 128U);
        const uint32_t qNBlockNumPerGroup = CeilDivU32(groupSize, qNBlockTile);
        const uint32_t qNTaskNum = qNBlockNumPerGroup * input.kvHeads;
        for (uint32_t qTileIdx = 0U; qTileIdx < qTileNum; ++qTileIdx) {
            const uint32_t qRows = MinU32(
                input.qBaseTile, qLen - qTileIdx * input.qBaseTile);
            for (uint32_t qNTask = 0U; qNTask < qNTaskNum; ++qNTask) {
                const uint32_t qNBlockIdxInGroup =
                    qNTask % qNBlockNumPerGroup;
                const uint32_t qNBlockSize = MinU32(
                    qNBlockTile,
                    groupSize - qNBlockIdxInGroup * qNBlockTile);
                if (baseCount >= MAX_FD_COMBINE_TASK_NUM) {
                    tiling.fdBaseTaskNum = baseCount + 1U;
                    return PlanStatus::CAPACITY_EXCEEDED;
                }
                bases[baseCount++] = {kvTileNum, qRows * qNBlockSize};
            }
        }
    }

    tiling.fdBaseTaskNum = baseCount;
    if (baseCount == 0U || baseCount > baseTaskLimit || activeCoreLimit < 2U) {
        return PlanStatus::FALLBACK;
    }

    uint32_t splitCounts[MAX_FD_COMBINE_TASK_NUM] = {};
    uint32_t candidateCount = baseCount;
    for (uint32_t base = 0U; base < baseCount; ++base) {
        splitCounts[base] = 1U;
    }
    if (input.numSplits > 1U) {
        candidateCount = 0U;
        for (uint32_t base = 0U; base < baseCount; ++base) {
            splitCounts[base] = MinU32(input.numSplits, bases[base].kvTiles);
            candidateCount += splitCounts[base];
        }
    } else {
        while (candidateCount < activeCoreLimit) {
            uint32_t bestBase = baseCount;
            uint64_t bestCost = 0U;
            for (uint32_t base = 0U; base < baseCount; ++base) {
                if (splitCounts[base] >= bases[base].kvTiles) {
                    continue;
                }
                const uint64_t cost =
                    static_cast<uint64_t>(bases[base].rowNum) *
                    bases[base].kvTiles / splitCounts[base];
                if (bestBase == baseCount || cost > bestCost) {
                    bestBase = base;
                    bestCost = cost;
                }
            }
            if (bestBase == baseCount) {
                break;
            }
            ++splitCounts[bestBase];
            ++candidateCount;
        }
    }

    const uint32_t activeCores = MinU32(activeCoreLimit, candidateCount);
    if (activeCores <= baseCount) {
        return PlanStatus::FALLBACK;
    }

    uint64_t totalCost = 0U;
    for (uint32_t candidate = 0U; candidate < candidateCount; ++candidate) {
        totalCost += CandidateAt(
            bases, splitCounts, baseCount, candidate).cost;
    }

    uint32_t candidateBegin = 0U;
    uint64_t consumedCost = 0U;
    for (uint32_t core = 0U; core < activeCores; ++core) {
        const uint32_t coresLeft = activeCores - core;
        const uint32_t maxEnd = candidateCount - (coresLeft - 1U);
        uint32_t end = candidateBegin + 1U;
        const uint64_t target = totalCost * (core + 1U) / activeCores;
        while (end < maxEnd &&
               consumedCost + CandidateAt(
                   bases, splitCounts, baseCount, end - 1U).cost < target) {
            consumedCost += CandidateAt(
                bases, splitCounts, baseCount, end - 1U).cost;
            ++end;
        }
        consumedCost += CandidateAt(
            bases, splitCounts, baseCount, end - 1U).cost;
        const Candidate first = CandidateAt(
            bases, splitCounts, baseCount, candidateBegin);
        const Candidate last = CandidateAt(
            bases, splitCounts, baseCount, end - 1U);
        tiling.fdDecodeSchedules[core] = {
            static_cast<int32_t>(first.baseTask),
            static_cast<int32_t>(last.baseTask + 1U),
            static_cast<int32_t>(first.kvBegin),
            static_cast<int32_t>(last.kvEnd)};
        candidateBegin = end;
    }

    uint32_t partialStart = 0U;
    uint32_t combineCount = 0U;
    for (uint32_t base = 0U; base < baseCount; ++base) {
        int32_t firstCore = -1;
        uint32_t partialCount = 0U;
        for (uint32_t core = 0U; core < activeCores; ++core) {
            const FdDecodeSchedule &schedule = tiling.fdDecodeSchedules[core];
            if (schedule.baseTaskStart <= static_cast<int32_t>(base) &&
                static_cast<int32_t>(base) < schedule.baseTaskEnd) {
                if (firstCore < 0) {
                    firstCore = static_cast<int32_t>(core);
                }
                ++partialCount;
            }
        }
        if (partialCount > 1U) {
            if (combineCount >= MAX_FD_COMBINE_TASK_NUM) {
                return PlanStatus::CAPACITY_EXCEEDED;
            }
            tiling.fdCombineSchedules[combineCount++] = {
                static_cast<int32_t>(base), firstCore,
                static_cast<int32_t>(partialStart),
                static_cast<int32_t>(partialCount)};
            partialStart += partialCount;
        }
    }

    if (combineCount == 0U || partialStart > tiling.fdPartialCapacity) {
        return PlanStatus::FALLBACK;
    }

    uint32_t maxRows = 0U;
    for (uint32_t base = 0U; base < baseCount; ++base) {
        maxRows = MaxU32(maxRows, bases[base].rowNum);
    }
    tiling.fdActiveCoreNum = activeCores;
    tiling.fdCombineTaskNum = combineCount;
    tiling.fdPartialTaskNum = partialStart;
    tiling.fdCombineBlockDim = MinU32(input.blockDim, combineCount);
    tiling.fdRowCapacity = static_cast<uint32_t>(AlignUpU64(maxRows, 8U));
    tiling.fdLseSubStride = static_cast<uint32_t>(AlignUpU64(
        (tiling.fdRowCapacity + 1U) / 2U, 8U));

    const uint64_t pipelineEnd = tiling.workSpaceSize;
    tiling.fdPartialLseOffset = AlignUpU64(
        pipelineEnd, WORKSPACE_ALIGNMENT);
    const uint64_t partialLseSize =
        static_cast<uint64_t>(tiling.fdPartialCapacity) * 2U *
        tiling.fdLseSubStride * SIZE_OF_32BIT;
    tiling.fdPartialOOffset = AlignUpU64(
        tiling.fdPartialLseOffset + partialLseSize, WORKSPACE_ALIGNMENT);
    const uint64_t partialOSize =
        static_cast<uint64_t>(tiling.fdPartialCapacity) *
        tiling.fdRowCapacity * input.embeddingSizeV * SIZE_OF_16BIT;
    tiling.fdWorkspaceEnd = tiling.fdPartialOOffset + partialOSize;
    if (input.workspaceLimit != 0U &&
        tiling.fdWorkspaceEnd > input.workspaceLimit) {
        tiling.fdActiveCoreNum = 0U;
        tiling.fdCombineTaskNum = 0U;
        tiling.fdPartialTaskNum = 0U;
        tiling.fdCombineBlockDim = 0U;
        return PlanStatus::WORKSPACE_EXCEEDED;
    }
    tiling.workSpaceSize = tiling.fdWorkspaceEnd;
    tiling.flashDecodeFlag = 1U;
    return PlanStatus::ENABLED;
}

}  // namespace fd_tiling

#endif  // FLASH_ATTN_NPU_950_V3_FA_SPLIT_H
