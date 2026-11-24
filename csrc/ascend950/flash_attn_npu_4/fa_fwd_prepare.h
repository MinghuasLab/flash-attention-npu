// Copyright (c) 2026, Minghua Shen.
#ifndef FLASH_ATTN_950_V4_FA_FWD_PREPARE_H
#define FLASH_ATTN_950_V4_FA_FWD_PREPARE_H

#include "tilingdata.h"
#include <cstdint>
#include <type_traits>

// Private, per-execution data. Keep the public FAInferStaticTilingData ABI
// separate so callers can still retain, copy and share scheduler metadata.
struct FA4FwdExecution {
    FAInferRuntimeData runtime;
    uint32_t status;
    uint32_t useDn;
    FAInferStaticTilingData tiling;
};

static_assert(std::is_trivially_copyable_v<FA4FwdExecution>);
static_assert(std::is_standard_layout_v<FA4FwdExecution>);

enum class FA4FwdStatus : uint32_t {
    Pending,
    Valid,
    InvalidOffsets,
    InvalidLengths,
    InvalidPage,
    InvalidMetadata,
    InvalidSchedule,
};

struct FA4FwdPrepareArgs {
    uint64_t qOffsets;
    uint64_t kOffsets;
    uint64_t qUsed;
    uint64_t kUsed;
    uint64_t pageTable;
    uint64_t metadata;
    uint64_t execution;
    uint64_t kvLengths;
    uint64_t kvOffsets;
    uint64_t workspaceSize;
    int64_t qTokens;
    int64_t kvCapacity;
    int64_t maxKv;
    int64_t left;
    int64_t right;
    uint32_t batch;
    uint32_t maxQ;
    uint32_t heads;
    uint32_t kvHeads;
    uint32_t qDim;
    uint32_t vDim;
    uint32_t pageSize;
    uint32_t pageCount;
    uint32_t maxPages;
    uint32_t coreCount;
    uint32_t splits;
    uint32_t quantMode;
    FwdLayout layout;
    float scale;
    float softcap;
    bool causal;
    bool inferMaxKv;
    bool useHostSchedule;
    bool bf16;
    bool lse;
};

#endif
