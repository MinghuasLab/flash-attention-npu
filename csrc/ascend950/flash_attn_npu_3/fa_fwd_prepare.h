// Copyright (c) 2026 Huawei Technologies Co., Ltd.

#ifndef FLASH_ATTN_NPU_950_V3_FA_FWD_PREPARE_H
#define FLASH_ATTN_NPU_950_V3_FA_FWD_PREPARE_H

#include <cstdint>
#include "tilingdata.h"

namespace fa3_fwd {

constexpr uint32_t PLAN_VALID = 0;
constexpr uint32_t PLAN_INVALID_SEQUENCE = 1;
constexpr uint32_t PLAN_INVALID_PAGE = 2;
constexpr uint32_t PLAN_INVALID_METADATA = 3;
constexpr uint32_t PLAN_INVALID_WORKSPACE = 4;
constexpr uint32_t PLAN_INVALID_SCHEDULE = 5;

// Private to forward. The public scheduler metadata starts with the original
// FAInferTilingData followed immediately by the cumulative KV lengths.
struct Plan {
    FAInferTilingData tiling;
    uint32_t status;
};

struct PrepareArgs {
    uint64_t plan;
    uint64_t cuQ;
    uint64_t cuKv;
    uint64_t usedQ;
    uint64_t usedKv;
    uint64_t pageTable;
    uint64_t suppliedMetadata;
    int64_t qCapacity;
    int64_t kvCapacity;
    int64_t maxKv;
    int64_t windowLeft;
    int64_t windowRight;
    uint64_t workspaceBytes;
    uint32_t batch;
    uint32_t maxQ;
    uint32_t numHeads;
    uint32_t numKvHeads;
    uint32_t headDim;
    uint32_t headDimV;
    uint32_t blockDim;
    uint32_t pageSize;
    uint32_t pageCount;
    uint32_t pagesPerRow;
    uint32_t numSplits;
    FwdLayout layout;
    bool causal;
    bool allowFd;
    bool quantized;
};

constexpr uint64_t QScratchOffset()
{
    return sizeof(Plan);
}

constexpr uint64_t KvScratchOffset(uint32_t batch)
{
    return QScratchOffset() + (static_cast<uint64_t>(batch) + 1) * sizeof(int32_t);
}

constexpr uint64_t PlanBytes(uint32_t batch)
{
    return KvScratchOffset(batch) + (static_cast<uint64_t>(batch) + 1) * sizeof(int32_t);
}

constexpr uint64_t FdWorkspaceBound(uint32_t aicCount, uint32_t dimV)
{
    if (aicCount == 0) {
        return 0;
    }
    const uint32_t baseTaskBound = (3U * aicCount - 1U) / 10U;
    const uint32_t decodeCoreBound = aicCount < MAX_FD_ACTIVE_CORE_NUM ? aicCount : MAX_FD_ACTIVE_CORE_NUM;
    const uint32_t combineTaskBound =
        baseTaskBound < MAX_FD_COMBINE_TASK_NUM ? baseTaskBound : MAX_FD_COMBINE_TASK_NUM;
    const uint64_t partials = combineTaskBound == 0 ? 0 : combineTaskBound + decodeCoreBound - 1U;
    constexpr uint64_t rowsPerPartial = 128U;
    constexpr uint64_t alignment = 512U;
    const uint64_t partialLse = partials * rowsPerPartial * sizeof(float);
    const uint64_t partialO = partials * rowsPerPartial * dimV * sizeof(float);
    return ((partialLse + alignment - 1) / alignment) * alignment + partialO;
}
} // namespace fa3_fwd
#endif
