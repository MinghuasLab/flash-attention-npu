/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once
#include <cstdint>
#include "mla_layout.h"
// Shared device arguments for the compressed and prefill MLA kernels.
// No tensor addresses or scale are stored in the reusable schedule.
namespace flash_attn::mla {
struct MlaDenseKernelSchedule {
    // Same endpoint capacity as MlaSchedule; not a device core count.
    static constexpr uint32_t kEndpointCapacity = 26;
    uint32_t batch, sq, heads, sk;
    uint32_t mBaseSize, s2BaseSize, usedCoreNum, blockDim, cvRatio;
    uint32_t mm1ResSize, mm2ResSize;
    uint32_t bN2End[kEndpointCapacity];
    uint32_t gS1End[kEndpointCapacity];
    uint32_t s2End[kEndpointCapacity];
};
} // namespace flash_attn::mla

// Device-only window parameters after the checked CPU NormalizeWindow step.
namespace flash_attn::mla {
struct MlaWindowKernelParams {
    // Same values as MlaWindowKind: None=0, Causal=1, Local=2.
    uint32_t kind = 0;
    int64_t left = 0, right = 0;
};
} // namespace flash_attn::mla

// Independent 192/128 numerical launch data. Not a serialized metadata ABI.
namespace flash_attn::mla {
struct MlaPrefillParams {
    uint32_t batch, sq, sk, heads, kvHeads, mBase, nBase, cores;
    float scale;
    MlaWindowKernelParams window;
    MlaBsndTensorLayout q, k, v;
    float softcap = 0.0f;
};
// A validated fragment of one S1G block for one KV head. The caller owns
// partitioning and bounds; begin/end are S2 tile indices, rows exclude padding.
struct MlaPrefillTask {
    uint32_t batch, kvHead, rowStart, rows, queryLength, kvLength, begin, end;
};
} // namespace flash_attn::mla

// Independent validated split-KV launch descriptors; no metadata/public ABI.
namespace flash_attn::mla {
// Independent partial launch: one validated M/S2 fragment per physical AIC.
// Slots use the native FD workspace (M rows, 512 FP32 O, 8 sum/max each).
// This is launch data, not production metadata or a public accumulator.
struct MlaSplitKvTask {
    uint32_t batch, m, split, s2Begin, s2End;
};
struct MlaSplitKvKernelParams {
    MlaWindowKernelParams window;
    uint64_t outOffsetBytes, sumOffsetBytes, maxOffsetBytes;
    uint64_t totalTasks;
    MlaSplitKvTask tasks[26];
};
// A validated GQA fragment uses the original allocation coordinates while
// scheduling a subset of query heads. Other callers keep their span reader.
struct MlaTaskLengths {
    uint32_t queryStart, queryLength, kvLength;
    uint64_t kvStart;
};
} // namespace flash_attn::mla

// Independent BSND FD launch data; not a production metadata record.
namespace flash_attn::mla {
struct MlaSplitKvMergeParams {
    uint32_t batch, sq, heads, batchIndex, rowStart, rows, mBase, splits;
    uint32_t kvLength, maskKind;
    int64_t left, right;
    // Logical KV intervals, clipped to KV <=16384. uint16 keeps the by-value
    // launch record below the 1536-byte compiler/runtime parameter limit.
    uint16_t begin[256], end[256];
    // TND specialization uses batch=1, sq=totalQ, batchIndex=allocation token.
    // Production Q-used controls mask geometry; sq remains allocation stride.
    // Independent components leave this zero and use sq (zero-Q has no task).
    uint32_t queryLength = 0;
    // HeadDim=128 BSND specialization: heads is the per-KV-head group size.
    uint32_t kvHeads = 1, kvHead = 0;
};
} // namespace flash_attn::mla
