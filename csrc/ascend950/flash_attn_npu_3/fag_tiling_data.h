// Copyright (c) 2026, Minghua Shen.

// Host/device tiling ABI for the Ascend950 FlashAttention v3 backward path.
// Keep this structure POD-only.  It is copied byte-for-byte from a CPU tensor
// to device global memory and will be consumed by the future arch35 FAG
// kernel.

#ifndef FLASH_ATTN_NPU_ARCH35_V3_FAG_TILING_DATA_H
#define FLASH_ATTN_NPU_ARCH35_V3_FAG_TILING_DATA_H

#include "../../fag_tensor_strides.h"
#include <cstdint>
#include <type_traits>

namespace FAGTiling950 {

constexpr uint64_t GM_ALIGNMENT = 512;
constexpr uint64_t MULTI_CORE_SYNC_BYTES = 64 * 1024;
constexpr uint32_t SOFTMAX_REDUCE_FLOATS = 8;
constexpr uint32_t DEFAULT_CONTINUOUS_BLOCK_NUM = 2;
// TND BN2S2 round prefix table capacity (max batches + 1), matching opst
// TND_SWIZZLE_PREFIX_NUM.
constexpr uint32_t TND_SWIZZLE_PREFIX_NUM = 129;

// VecDTM chunk size: the device epilogue (fag_epilogue_deterministic_add.hpp)
// rebuilds one chunk of a round's task list in a fixed-size on-stack table of
// this many entries, tracked by a 64-bit consumed bitmap — do NOT raise
// above 64 without widening both.  Rounds larger than this are legal and are
// processed in multiple chunks.
constexpr uint32_t MAX_DTM_CHUNK_TASKS = 64;

enum class Layout : uint32_t {
    BSND = 0,
    TND = 1,
    BNSD = 2,
};

enum class MaskType : uint32_t {
    NO_MASK = 0,
    CAUSAL = 1,
    BAND = 2,
};

// Deterministic accumulation strategy:
//   LEGACY  = per-round det slots + streaming VecDTM reduction (det-cmp-v2)
//   BN2S2   = opst arch35 column-private schedule: the schedule fixes which
//             core accumulates each (b, n2, s2) dk/dv column and pins the
//             cross-core dq rounds with a mode-0 fix barrier, so no det slot
//             and no per-round reducer is needed.  Stage 1 accumulates
//             dq/dk/dv atomically into the full fp32 workspaces; the opst
//             per-core private region + early bf16 cast is a follow-up.
enum class DetSchedule : uint32_t {
    LEGACY = 0,
    BN2S2 = 1,
};

struct FAGInfo {
    Layout layout = Layout::BSND;
    MaskType maskType = MaskType::NO_MASK;
    uint32_t deterministic = 0;
    uint32_t dqPostAbsorb = 0;
    uint32_t detSchedule = 0;

    uint64_t batch = 0;
    uint64_t qSeqlen = 0;
    uint64_t kvSeqlen = 0;
    uint64_t totalQ = 0;
    uint64_t totalKv = 0;
    uint64_t qHeadNum = 0;
    uint64_t kvHeadNum = 0;
    uint64_t qkHeadDim = 0;
    uint64_t vHeadDim = 0;
    int64_t window_size_left;
    int64_t window_size_right;

    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint32_t dqVecNum = 0;
    uint32_t dkVecNum = 0;
    uint32_t dvVecNum = 0;
    uint32_t continuousBlockNum = DEFAULT_CONTINUOUS_BLOCK_NUM;
    uint64_t ubSize = 0;
    float scaleValue = 1.0f;
    float softcapValue = 0.0f;

    // TND only: planner-local pointers to per-batch actual sequence lengths.
    const int64_t* actualSeqQ = nullptr;
    const int64_t* actualSeqKv = nullptr;
};

struct FAGGradClearRange {
    uint64_t begin = 0;
    uint64_t count = 0;
};

enum class FAGPlanStatus : uint32_t {
    READY = 0,
    INVALID_LENGTH = 1,
    UNSUPPORTED_TILING = 2,
};

struct FAGTilingData {
    FagInputStrides inputStrides{};
    uint32_t layout = 0;
    uint32_t maskType = 0;
    uint32_t deterministic = 0;
    uint32_t dqPostAbsorb = 0;
    uint32_t detSchedule = 0;
    uint32_t dqVecNum = 0, dkVecNum = 0, dvVecNum = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint32_t usedCoreNum = 0;
    uint32_t continuousBlockNum = DEFAULT_CONTINUOUS_BLOCK_NUM;
    uint32_t detCastChunkRows = 0;
    // Non-deterministic short-S TND (shortQ/shortBiDir) with D<=128: cast each
    // flushed dk/dv tile to bf16/fp16 in-loop and skip FagPost for dk/dv.
    // dq still uses the fp32 workspace + FagPost (atomic accumulate over S2).
    // 0: legacy end-of-kernel cast (BSND and non-short TND).
    uint32_t earlyDkvCast = 0;
    // Non-deterministic short bidirectional (maxS1<=16 && maxS2<=16): isolated
    // schedule that replaces the B×N2 full scan with core-strided task indexing.
    // Must stay 0 for longer S1/S2 so those shapes keep the legacy pipeline.
    uint32_t shortBiDir = 0;
    // 1 when every batch shares the same S1/S2. Enables arithmetic batch
    // offsets without cu_seqlens GetValue in the hot loop.
    uint32_t shortBiDirUniform = 0;
    // Non-deterministic short-Q long-KV (maxS1<=16 && maxS2>16): core-strided
    // schedule over (b,n2) with S2 inner. Mutually exclusive with shortBiDir.
    // Indexing is arithmetic (gid → b,n2 via kvHeadNum); no per-batch prefix table.
    uint32_t shortQ = 0;
    // 1 when every batch shares the same S1 (enables arithmetic Q offsets).
    uint32_t shortQUniformS1 = 0;

    uint64_t ubSize = 0;
    uint64_t batch = 0;
    uint64_t qSeqlen = 0;
    uint64_t kvSeqlen = 0;
    uint64_t totalQ = 0;
    uint64_t totalKv = 0;
    uint64_t qHeadNum = 0;
    uint64_t kvHeadNum = 0;
    uint64_t groupSize = 0;
    uint64_t qkHeadDim = 0;
    uint64_t vHeadDim = 0;
    int64_t s1Token;
    int64_t s2Token;

    float scaleValue = 1.0f;
    float softcapValue = 0.0f;

    uint32_t qTile = 0;
    uint32_t kvTile = 0;
    // 1: D>256 128x256 partial-resident puncture path (required for D>256).
    // 0: legacy 128x128 full-residency schedule for D<=256.
    uint32_t d512Partial = 0;

    // BN2S2 deterministic schedule (detSchedule == BN2S2).
    uint32_t detKind = 0;         // fag_det::Kind
    uint32_t detColumnRounds = 0; // rounds per owned column
    uint32_t detBufNum = 1;       // private dk/dv buffers per core (2 for causal fold)
    uint32_t detPrivDkv = 0;      // 1: per-core private dk/dv + early AIV cast (g == 1)
    uint64_t detMaxRound = 0;     // total schedule rounds
    uint64_t dkPrivOffset = 0;    // per-core dk fp32 accumulation region
    uint64_t dvPrivOffset = 0;    // per-core dv fp32 accumulation region
    // TND BN2S2: exclusive end round of each batch (prefix[B] = detMaxRound).
    int64_t tndPrefix[TND_SWIZZLE_PREFIX_NUM] = {0};
    // TND causal (KIND_TND_CAUSAL): diagonal-band prefix tables p1/p2.
    int64_t tndCausalP1[TND_SWIZZLE_PREFIX_NUM] = {0};
    int64_t tndCausalP2[TND_SWIZZLE_PREFIX_NUM] = {0};

    uint64_t dqOffset = 0;
    uint64_t dkOffset = 0;
    uint64_t dvOffset = 0;
    uint64_t deltaOffset = 0;
    uint64_t dqDetOffset = 0;
    uint64_t dkDetOffset = 0;
    uint64_t dvDetOffset = 0;
    uint64_t workspaceSize = 0;
    FAGPlanStatus planStatus = FAGPlanStatus::READY;
    FAGGradClearRange clearDq{};
    FAGGradClearRange clearDk{};
    FAGGradClearRange clearDv{};
};

static_assert(std::is_standard_layout_v<FAGTilingData>, "FAGTilingData must have a stable host/device layout");
static_assert(std::is_trivially_copyable_v<FAGTilingData>, "FAGTilingData must be byte-copyable to device");

struct FagPrepareArgs {
    FAGInfo info;
    FagInputStrides inputStrides;
    uint64_t tilingAddr = 0;
    uint64_t cuQAddr = 0;
    uint64_t cuKvAddr = 0;
    uint64_t usedQAddr = 0;
    uint64_t usedKvAddr = 0;
    uint64_t totalK = 0;
    uint64_t totalV = 0;
    uint64_t workspaceCapacity = 0;
    uint32_t exactPackedLength = 0;
};

static_assert(std::is_trivially_copyable_v<FagPrepareArgs>);

int64_t GetFAGTilingParam(const FAGInfo& info, FAGTilingData& tiling);

} // namespace FAGTiling950

#endif
