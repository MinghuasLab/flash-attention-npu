/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once

#include <array>
#include <cstdint>
#include <algorithm>
#include <vector>

namespace flash_attn::mla {

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.h.
// This is an endpoint capacity, not a hard-coded device core count.
constexpr uint32_t kMlaMaxAic = 26;

// Preserve the source's uint32 S1G ceil-div, dense query stride (Hq*512)
// and LSE DMA stride ((Sq-1)*sizeof(float)) without wrapping any field.
constexpr uint32_t kMlaMaxDenseRows = UINT32_MAX - 255;
constexpr uint32_t kMlaMaxQueryHeads = UINT32_MAX / 512;
constexpr uint32_t kMlaMaxQueryLength = UINT32_MAX / sizeof(float) + 1;
constexpr uint64_t kMlaMaxBatchRows = INT64_MAX / (512 * 2);
// Validated dense KV capacity; larger sequences remain explicitly unsupported.
constexpr uint32_t kMlaMaxKvLength = 16384;

struct MlaDenseShape {
    int64_t batch;
    int64_t queryLength;
    int64_t queryHeads;
    int64_t kvLength;
};

struct MlaLocalMemory {
    uint64_t l1Bytes;
    uint64_t ubBytes;
    uint64_t l0aBytes;
    uint64_t l0bBytes;
    uint64_t l0cBytes;
};

struct MlaPlatformInfo {
    uint32_t aicNum;
    uint32_t aivNum;
    MlaLocalMemory localMemory;
    uint64_t libApiWorkspaceBytes;
    // Populate entry c-1 with PlatformAscendC::CalcTschBlockDim(c*ratio, c, c*ratio).
    // Passing platform results keeps the same computation usable by host/AICPU
    // without a gert dependency or an assumed relationship between cores and blockDim.
    std::array<uint32_t, kMlaMaxAic> blockDims;
};

struct MlaFlashDecodeSchedule {
    uint32_t numOfFdHead;
    uint32_t maxS2SplitNum;
    uint32_t usedVecNumOfFd;
    uint32_t gS1BaseSizeOfFd;
    std::array<uint32_t, kMlaMaxAic> bN2IdxOfFdHead;
    std::array<uint32_t, kMlaMaxAic> gS1IdxOfFdHead;
    std::array<uint32_t, kMlaMaxAic> s2SplitNumOfFdHead;
    std::array<uint32_t, kMlaMaxAic> s2SplitStartIdxOfCore;
    std::array<uint32_t, kMlaMaxAic> gS1SplitNumOfFdHead;
    std::array<uint32_t, kMlaMaxAic> gS1LastPartSizeOfFdHead;
    std::array<uint32_t, kMlaMaxAic * 2> gS1IdxEndOfFdHead;
    std::array<uint32_t, kMlaMaxAic * 2> gS1IdxEndOfFdHeadSplit;
};

struct MlaWorkspace {
    // Source tiling fields: elements per core per slot, NOT bytes.
    uint64_t mm1ResSize;
    uint64_t mm2ResSize;
    // Source fills these even when FD is disabled; no FD workspace is allocated.
    uint64_t fdAccumOutSize;
    uint64_t fdLogSumExpSize;
    // Actual kernel segments use launch blockDim, while the host budget uses aicNum.
    uint64_t qkOffsetBytes;
    uint64_t pOffsetBytes;
    uint64_t pvOffsetBytes;
    uint64_t vec2OffsetBytes;
    uint64_t kernelBytes;
    uint64_t userBytes;
    uint64_t libApiBytes;
    uint64_t totalBytes;
};

struct MlaSchedule {
    uint32_t bSize;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t headDim;
    uint32_t headDimRope;
    uint32_t mBaseSize;
    uint32_t s2BaseSize;
    uint32_t aicNum;
    uint32_t aivNum;
    uint32_t cvRatio;
    uint32_t usedCoreNum;
    uint32_t blockDim;
    uint32_t scheduleMode; // CANN ScheduleMode::BATCH_MODE = 1.
    // Exclusive ends in (batch*Hkv, S1G block, S2 block) coordinates.
    // Core 0 starts at (0,0,0); subsequent cores start at the previous endpoint.
    std::array<uint32_t, kMlaMaxAic> bN2End;
    std::array<uint32_t, kMlaMaxAic> gS1End;
    std::array<uint32_t, kMlaMaxAic> s2End;
    int64_t maxCost;
    MlaFlashDecodeSchedule fd;
    MlaWorkspace workspace;
    MlaLocalMemory requiredLocalMemory;
};

enum class MlaScheduleStatus {
    Success,
    UnsupportedShape,
    FieldOverflow,
    InvalidPlatform,
    InsufficientLocalMemory,
    WorkspaceOverflow,
    UnexpectedFlashDecode,
    InvalidSplit,
    AllocationFailure,
};

// Pure CPU scheduling for contiguous BSND, shared 512+64 KV, Hkv=1,
// Sq*Hq>=0 within the address/field bounds above, 0<=Sk<=kMlaMaxKvLength, B>=0,
// no mask/length arrays, num_splits=1.
// A zero batch/query/KV capacity produces a validation-only schedule: one
// launch core, zero tiles/endpoints/workspace. No attention Kernel consumes it.
// FP16 and BF16 have identical scheduling; callers validate tensor/layout/options.
// No tensor addresses or scale are stored. On failure, output is unchanged.
MlaScheduleStatus BuildDenseSchedule(const MlaDenseShape& shape, const MlaPlatformInfo& platform, MlaSchedule& output);

} // namespace flash_attn::mla

namespace flash_attn::mla {

// Independent CPU entry for contiguous BSND with full query capacity and
// per-sample KV lengths. Same shape/platform bounds and num_splits=1 as dense.
// kvLengths must point to lengthCount readable int32 values on the executing
// processor and remain stable during this call. nullptr/0 means full capacity;
// otherwise lengthCount must equal B, and every value must be in [0, Sk].
// Invalid counts/values return UnsupportedShape before conversion to source
// int64 lengths. Values are copied numerically, never pointer-reinterpreted.
// No addresses/length pointers survive the call. On failure output is unchanged.
// Nonzero KV capacity with all lengths zero preserves the source batch endpoint,
// base sizes and workspace; global empty capacity follows BuildDenseSchedule.
// Compute endpoints can omit a zero-KV suffix, as in the pinned source. The
// device consumer initializes zero-KV outputs independently of these ranges.
MlaScheduleStatus BuildBsndKvLengthSchedule(const MlaDenseShape& shape, const MlaPlatformInfo& platform,
                                            const int32_t* kvLengths, uint64_t lengthCount, MlaSchedule& output);

enum class MlaSequenceLayout {
    Bsnd,
    Tnd
};

struct MlaSequenceAllocation {
    MlaSequenceLayout layout;
    int64_t totalTokens; // Physical allocation, including unused tokens.
    int64_t maxLength;   // BSND capacity; TND upper bound on allocation prefix differences.
};

struct MlaUsedShape {
    int64_t batch, queryHeads;
    MlaSequenceAllocation query, kv;
};

struct MlaLengthArray {
    const int32_t* data = nullptr;
    uint64_t count = 0;
};

struct MlaUsedInputs {
    MlaLengthArray cuQuery, cuKv;
    MlaLengthArray usedQuery, usedKv;
};

struct MlaUsedSequence {
    uint64_t queryStart, kvStart; // Physical token offsets; never cumulative used lengths.
    uint32_t queryCapacity, kvCapacity;
    uint32_t queryLength, kvLength;
};

struct MlaUsedLengths {
    uint32_t maxQueryCapacity = 0, maxKvCapacity = 0;
    uint32_t maxQueryLength = 0, maxKvLength = 0;
    std::vector<MlaUsedSequence> sequences;
};

// Independent CPU component: BSND Q/KV, TND Q/KV or TND Q + BSND KV only.
// TND requires int32[B+1] allocation prefixes, including [0] for B=0; BSND
// requires absent prefixes and totalTokens == B*capacity. Optional used arrays
// are absent iff {nullptr, 0}; otherwise exactly B readable int32 values, each
// in [0, its own allocation length]. Absence means the entire allocated segment.
// All static bounds/counts/values are checked before copying any spans. Inputs
// must remain readable/stable during the call. No pointers are retained, and
// failure leaves the output unchanged.
MlaScheduleStatus ParseUsedLengths(const MlaUsedShape& shape, const MlaUsedInputs& inputs, MlaUsedLengths& output);

// Source tile geometry from an already validated allocation/used snapshot.
MlaScheduleStatus PrepareMlaGeometry(const MlaUsedShape&, const MlaPlatformInfo&, const MlaUsedInputs&,
                                     const MlaUsedLengths&, uint32_t pageSize, MlaSchedule&);

} // namespace flash_attn::mla

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/common/op_host/split_core.cpp::CalcS2Range / CalcNextTokenLeftUp.

namespace flash_attn::mla {

struct MlaCausalBlockInput {
    // Effective lengths of ONE sample, not allocation capacities or batch maxima.
    // Layout/address/used validation belongs to the caller. Rows use S1G order.
    int64_t queryLength;
    int64_t kvLength;
    int64_t queryHeads;
    int64_t mBaseSize;
    int64_t s2BaseSize;
    int64_t mBlock;
};

struct MlaCausalBlock {
    int64_t nextTokenLeftUp; // Sk - Sq; key j is visible iff j <= query i + this value.
    uint32_t rowBegin;
    uint32_t rowEnd;
    uint32_t queryBegin;
    uint32_t queryEnd;
    uint32_t kvBegin;
    uint32_t kvEnd;
    uint32_t s2Begin;
    uint32_t s2End;
    uint32_t leadingEmptyRows;
    int64_t cost;
    int64_t normalS2Cost;
    int64_t lastS2Cost;
};

enum class MlaCausalStatus {
    Success,
    UnsupportedShape,
    FieldOverflow,
    InvalidTile,
    InvalidBlock,
    InvalidPlatform,
    InvalidSplit,
    AllocationFailure,
};

// CPU-only geometry for shared 512+64 KV, Hkv=1, right-down causal, num_splits=1.
// Reuses the pinned source's S2 block range. All intervals are half-open and
// relative to one sample. kvBegin/kvEnd are the union of visible logical keys;
// boundary S2 blocks still need a per-row causal mask in the attention kernel.
// leadingEmptyRows counts flattened S1G rows with NO visible key, including
// all rows when Sk=0. It is not a count of query tokens or unused allocation.
// Sq=0 accepts only mBlock=0 and produces empty intervals. Otherwise mBlock
// must name a real M block. Logical tiles are 1..256 by 1..512; accepting a
// size here does not validate device resources or enable a production kernel.
// Uses the existing per-sample MLA shape bounds. Output is unchanged on failure.
// No tensor, metadata format, core partition, workspace or public entry is created.
MlaCausalStatus BuildCausalBlock(const MlaCausalBlockInput& input, MlaCausalBlock& output);

struct MlaCausalSequence {
    int64_t queryLength, kvLength;
};

struct MlaCausalScheduleInput {
    // Owned effective lengths, obtained after the existing layout/used parsers.
    // No allocation prefix, page ID, address or batch maximum is a mask length.
    std::vector<MlaCausalSequence> sequences;
    int64_t queryHeads, mBaseSize, s2BaseSize;
    int64_t aicNum, cvRatio;
};

struct MlaCausalSchedule {
    uint32_t usedCoreNum;
    std::array<uint32_t, kMlaMaxAic> bN2End, gS1End, s2End;
    std::array<int64_t, kMlaMaxAic> coreCost;
    int64_t maxCost, totalCost;
    uint64_t totalMBlocks, totalS2Blocks;
};

// Independent CPU partition at caller-selected logical tile sizes. Retain the
// pinned SplitCore candidate, round partial S2 endpoints up to whole M blocks,
// coalesce duplicates and recompute each core's mask-dependent cost. The last
// endpoint is always (B,0,0), including trailing zero-cost output blocks.
// Every effective Q row has one owner, even if it sees no key. Empty work uses
// one zero-cost core. Unused array entries are zero; output is unchanged on
// failure. No FD, workspace, platform resource validation or production launch.
MlaCausalStatus BuildCausalSchedule(const MlaCausalScheduleInput& input, MlaCausalSchedule& output);

} // namespace flash_attn::mla

// Source: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0 split_core.cpp
// and official FA3 d15f1531a460ba456f41b01a774f33ab2db8febf hopper/flash_api.cpp.

namespace flash_attn::mla {

struct MlaWindowConfig {
    // Call-level bounds used by FA3 host normalization, not per-sample used lengths.
    int64_t maxQueryLength, maxKvLength;
    int64_t windowLeft, windowRight;
    bool causal;
};

enum class MlaWindowKind {
    None,
    Causal,
    Local
};
struct MlaWindow {
    MlaWindowKind kind;
    // FA3 parameter values AFTER replacing unlimited sides by bound - 1.
    // Inactive values may be -1 for zero bounds; kind determines the mask.
    int64_t left, right;
};

// Fixed FA3 forward/metadata normalization for 512+64, attention_chunk=0.
// Negative window values denote unlimited sides. Large positive int64 windows
// are normalized before narrowing/arithmetic; host-normalized values below
// INT32_MIN reject (FA3 forward narrows to int, unlike metadata).
// The single-query causal optimization follows FA3's exact -1 checks.
// Bounds use the existing MLA limits; failure leaves output unchanged.
MlaCausalStatus NormalizeWindow(const MlaWindowConfig& input, MlaWindow& output);

// Same complete-M ownership/output format as causal, including zero-cost rows.
// Reuses causal scheduling for the causal special case; otherwise adapts the
// pinned band/no-mask SplitCore candidates. No FD, metadata or production launch.
MlaCausalStatus BuildWindowSchedule(const MlaCausalScheduleInput& input, const MlaWindowConfig& window,
                                    MlaCausalSchedule& output);

} // namespace flash_attn::mla
