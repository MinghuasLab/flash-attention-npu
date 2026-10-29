/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once

#include "mla_tiling.h"

// Sources: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0 SplitCore and
// fia_tiling_nonquant_mla.cpp; FA3 d15f1531a460ba456f41b01a774f33ab2db8febf
// hopper/flash_api.cpp and block.h.

namespace flash_attn::mla {

struct MlaNativeSplitKvSchedule {
    uint32_t usedCoreNum;
    std::array<uint32_t, kMlaMaxAic> bN2End, gS1End, s2End;
    int64_t maxCost, totalCost;
    uint64_t totalMBlocks, totalS2Blocks;
    MlaFlashDecodeSchedule fd;
};

enum class MlaAccumulatorQueryLayout {
    Bsnd,
    Tnd
};
struct MlaAccumulatorInput {
    MlaAccumulatorQueryLayout layout;
    int64_t batch, queryCapacity, totalQuery, queryHeads, numSplits;
};
struct MlaAccumulatorTensor {
    uint32_t rank;
    std::array<uint64_t, 5> shape, strides; // contiguous ELEMENT strides, unused entries zero
    uint64_t elements, bytes;
};
struct MlaAccumulatorLayout {
    MlaAccumulatorTensor out, lse;
};

// FA3 allocation capacity, not used rows: BSND [S,B,H,Q,512]/[S,B,H,Q],
// TND [S,H,T,512]/[S,H,T]. FP32 for S>1; S=1 has no accumulator storage.
// BSND requires totalQuery=B*queryCapacity; TND totalQuery<=INT32_MAX, with
// prefix/max/used relationships validated by the existing length parser.
MlaScheduleStatus BuildSplitAccumulatorLayout(const MlaAccumulatorInput& input, MlaAccumulatorLayout& output);

struct MlaGqaOptions {
    int64_t packGqa = -1; // -1 = backend default, 0 = separate query heads, 1 = packed GQA.
    int64_t smMargin = 0; // Reserve this many AICs and twice as many AIVs.
};
struct MlaGqaResources {
    MlaPlatformInfo platform;
    uint32_t headGroups, headsPerGroup;
    bool packed;
};
// The NPU default preserves the existing CANN S1G path. G=1 normalizes to
// unpacked. Validate signed options before narrowing/subtraction. Keep the
// physical platform's blockDim mapping for the remaining C1V2 resources.
MlaScheduleStatus ResolveGqaResources(int64_t heads, int64_t kvHeads, const MlaGqaOptions& options,
                                      const MlaPlatformInfo& physical, MlaGqaResources& output);

struct MlaGqaConfig {
    MlaUsedShape shape;
    int64_t kvHeads, valueDim, mBase, nBase;
    int64_t pageSize = 0; // PA uses logical rectangular KV spans; page lookup remains separate.
};

// Lengths are parsed once by the common metadata producer. Packed and
// unpacked heads differ only in the coordinates passed to CANN SplitCore.
MlaScheduleStatus BuildGqaSchedule(const MlaGqaConfig&, const MlaUsedLengths&, const MlaWindowConfig&,
                                   const MlaGqaOptions&, const MlaPlatformInfo&, MlaNativeSplitKvSchedule&);
} // namespace flash_attn::mla
