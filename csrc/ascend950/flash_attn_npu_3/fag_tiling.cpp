#include <array>
#include <vector>
// Copyright (c) 2026, Minghua Shen.

// Ascend950-specific host tiling and workspace calculation for FA v3 bwd.

#ifndef FAG_TILING_950_CPP_H
#define FAG_TILING_950_CPP_H

#include "fag_common.h"
#include "fag_ub_layout.hpp"
#include "fag_cast_ub_plan.hpp"
#include "fag_det_schedule_host.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>

namespace FAGTiling950 {
namespace {

constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t FP32_ROW_ALIGN = 8; // 32B alignment in float elements

uint64_t RoundUpU64(uint64_t value, uint64_t align)
{
    return (value + align - 1) / align * align;
}

uint64_t CeilDivU64(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

// The existing greedy assignment is a fallback. For a small number of
// distinct full-mask column lengths, search for a packing at the lower bound.
template <typename Column>
void TryTightFullColumnPacking(const Column* columns, size_t count, uint32_t coreCount,
                               uint32_t totalTasks, uint8_t* owners)
{
    constexpr size_t maxShapes = 4;
    constexpr size_t maxPatterns = 4096;
    constexpr uint32_t searchBudget = 2048;
    const uint32_t capacity = static_cast<uint32_t>(CeilDivU64(totalTasks, coreCount));
    if (capacity > 64U || columns[0].tasks > capacity || columns[count - 1].tasks < 4U) {
        return;
    }

    using Counts = std::array<uint16_t, maxShapes>;
    Counts remaining{};
    std::array<uint32_t, maxShapes> weights{};
    std::array<size_t, maxShapes> nextColumn{};
    size_t shapes = 0;
    for (size_t i = 0; i < count; ++i) {
        if (i == 0 || columns[i].tasks != columns[i - 1].tasks) {
            if (shapes == maxShapes) {
                return;
            }
            weights[shapes] = columns[i].tasks;
            nextColumn[shapes++] = i;
        }
        ++remaining[shapes - 1];
    }

    struct Pattern {
        Counts counts;
        uint32_t load;
    };
    std::vector<Pattern> patterns;
    Counts pattern{};
    const auto enumerate = [&](auto& self, size_t shape, uint32_t load) -> void {
        if (patterns.size() == maxPatterns) {
            return;
        }
        if (shape == shapes) {
            if (load > 0) {
                patterns.push_back({pattern, load});
            }
            return;
        }
        const uint32_t limit = std::min<uint32_t>(remaining[shape], (capacity - load) / weights[shape]);
        for (uint32_t n = 0; n <= limit; ++n) {
            pattern[shape] = static_cast<uint16_t>(n);
            self(self, shape + 1, load + n * weights[shape]);
        }
    };
    enumerate(enumerate, 0, 0);
    if (patterns.size() == maxPatterns) {
        return;
    }
    std::stable_sort(patterns.begin(), patterns.end(), [](const Pattern& a, const Pattern& b) {
        return a.load > b.load;
    });

    Counts chosen[LEGACY_VARLEN_CORES] = {};
    uint32_t searches = 0;
    const auto pack = [&](auto& self, uint32_t core, uint32_t todo) -> bool {
        if (todo == 0) {
            return true;
        }
        if (core == coreCount || todo > capacity * (coreCount - core) || searches++ >= searchBudget) {
            return false;
        }
        const size_t largest = std::find_if(remaining.begin(), remaining.end(),
                                            [](uint16_t n) { return n != 0; }) - remaining.begin();
        const uint32_t roomAfterCore = capacity * (coreCount - core - 1);
        for (const auto& candidate : patterns) {
            if (candidate.load + roomAfterCore < todo) {
                break;
            }
            if (candidate.counts[largest] == 0) {
                continue;
            }
            bool fits = true;
            for (size_t i = 0; i < shapes; ++i) {
                fits &= candidate.counts[i] <= remaining[i];
            }
            if (!fits) {
                continue;
            }
            for (size_t i = 0; i < shapes; ++i) {
                remaining[i] -= candidate.counts[i];
            }
            if (self(self, core + 1, todo - candidate.load)) {
                chosen[core] = candidate.counts;
                return true;
            }
            for (size_t i = 0; i < shapes; ++i) {
                remaining[i] += candidate.counts[i];
            }
        }
        return false;
    };
    if (!pack(pack, 0, totalTasks)) {
        return;
    }

    for (uint32_t core = 0; core < coreCount; ++core) {
        for (size_t shape = 0; shape < shapes; ++shape) {
            for (uint16_t i = 0; i < chosen[core][shape]; ++i) {
                owners[nextColumn[shape]++] = static_cast<uint8_t>(core);
            }
        }
    }
}

void PlanLegacyVarlenColumns(const FAGInfo& info, FAGTilingData& tiling)
{
    const bool shortWide = info.qkHeadDim == 192 && info.vHeadDim == 192 && info.qSeqlen <= 256;
    if (info.layout != Layout::TND || info.hasUsedLengths || info.actualSeqQ == nullptr ||
        info.actualSeqKv == nullptr || tiling.usedCoreNum > LEGACY_VARLEN_CORES ||
        info.batch * info.kvHeadNum > LEGACY_VARLEN_HEADS || tiling.qTile != tiling.kvTile ||
        (!shortWide && (info.qkHeadDim > 128 || info.vHeadDim > 128)) || (info.maskType != MaskType::NO_MASK &&
        info.maskType != MaskType::CAUSAL)) {
        return;
    }

    const bool causal = info.maskType == MaskType::CAUSAL;
    bool mixed = false;
    uint32_t columnsPerHead = 0;
    for (uint64_t b = 0; b < info.batch; ++b) {
        const int64_t length = info.actualSeqQ[b];
        if (length <= 0 || length != info.actualSeqKv[b] || length % tiling.qTile != 0 ||
            static_cast<uint64_t>(length) / tiling.qTile > LEGACY_VARLEN_COLUMNS) {
            return;
        }
        mixed |= length != info.actualSeqQ[0];
        columnsPerHead += static_cast<uint32_t>(length) / tiling.qTile;
    }
    const uint64_t groups = info.qHeadNum / info.kvHeadNum;
    const bool splitSmallGqa = !shortWide && !mixed && causal && info.actualSeqQ[0] == 4 * tiling.qTile &&
        groups >= 4 && groups % 2 == 0 && info.batch * info.kvHeadNum * 2 <= LEGACY_VARLEN_HEADS &&
        info.batch * info.kvHeadNum * 2 * 4 >= tiling.usedCoreNum &&
        info.batch * info.kvHeadNum * 4 <= tiling.usedCoreNum;
    const bool splitWide = shortWide && groups >= 2 && groups % 2 == 0 &&
        info.batch * info.kvHeadNum * 2 <= LEGACY_VARLEN_HEADS &&
        columnsPerHead * info.kvHeadNum < tiling.usedCoreNum &&
        columnsPerHead * info.kvHeadNum * 2 >= tiling.usedCoreNum;
    const bool split = splitSmallGqa || splitWide;
    if (!split && (!mixed || shortWide)) {
        return;
    }
    const uint32_t segments = split ? 2U : 1U;
    const uint32_t logicalHeads = static_cast<uint32_t>(info.kvHeadNum) * segments;

    struct Column {
        uint32_t head;
        uint32_t index;
        uint32_t tasks;
    };
    Column columns[LEGACY_VARLEN_HEADS * LEGACY_VARLEN_COLUMNS];
    size_t count = 0;
    for (uint32_t b = 0; b < info.batch; ++b) {
        const uint32_t tiles = static_cast<uint32_t>(info.actualSeqQ[b]) / tiling.qTile;
        for (uint32_t head = 0; head < logicalHeads; ++head) {
            for (uint32_t column = 0; column < tiles; ++column) {
                columns[count++] = {b * logicalHeads + head, column,
                                    causal ? tiles - column : tiles};
            }
        }
    }
    std::stable_sort(columns, columns + count, [](const Column& a, const Column& b) {
        return a.tasks > b.tasks;
    });
    uint32_t tasks[LEGACY_VARLEN_CORES] = {};
    uint8_t owners[LEGACY_VARLEN_HEADS * LEGACY_VARLEN_COLUMNS];
    uint32_t totalTasks = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t core = static_cast<uint32_t>(std::min_element(tasks, tasks + tiling.usedCoreNum) - tasks);
        tasks[core] += columns[i].tasks;
        owners[i] = static_cast<uint8_t>(core);
        totalTasks += columns[i].tasks;
    }

    // Full columns from shorter batches can pack into the slack left by
    // long columns. Try tight capacities before accepting the greedy peak.
    if (!causal) {
        const uint32_t peak = *std::max_element(tasks, tasks + tiling.usedCoreNum);
        const uint32_t minimum = static_cast<uint32_t>(CeilDivU64(totalTasks, tiling.usedCoreNum));
        for (uint32_t capacity = std::max(minimum, columns[0].tasks); capacity < peak; ++capacity) {
            uint32_t load[LEGACY_VARLEN_CORES] = {};
            uint8_t packed[LEGACY_VARLEN_HEADS * LEGACY_VARLEN_COLUMNS];
            size_t i = 0;
            for (; i < count; ++i) {
                uint32_t core = 0;
                while (core < tiling.usedCoreNum && load[core] + columns[i].tasks > capacity) {
                    ++core;
                }
                if (core == tiling.usedCoreNum) {
                    break;
                }
                load[core] += columns[i].tasks;
                packed[i] = static_cast<uint8_t>(core);
            }
            if (i == count) {
                std::copy_n(packed, count, owners);
                break;
            }
        }
    }
    if (!causal && mixed && groups > 1 && !split) {
        uint32_t peakLoads[LEGACY_VARLEN_CORES] = {};
        for (size_t i = 0; i < count; ++i) {
            peakLoads[owners[i]] += columns[i].tasks;
        }
        const auto currentPeak = *std::max_element(peakLoads, peakLoads + tiling.usedCoreNum);
        if (currentPeak > CeilDivU64(totalTasks, tiling.usedCoreNum)) {
            TryTightFullColumnPacking(columns, count, tiling.usedCoreNum, totalTasks, owners);
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const Column& column = columns[i];
        tiling.legacyVarlenColumns[owners[i]][column.head] |= 1U << column.index;
    }
    tiling.legacyVarlenSchedule = segments;
}

void PlanColumnTasks(const FAGInfo& info, FAGTilingData& tiling)
{
    const bool causal = info.maskType == MaskType::CAUSAL;
    if (info.layout != Layout::TND || (!causal && info.maskType != MaskType::NO_MASK) ||
        info.deterministic || info.softcapValue != 0 || info.hasUsedLengths ||
        info.actualSeqQ == nullptr || info.actualSeqKv == nullptr ||
        info.qkHeadDim != 128 || info.vHeadDim != 128 ||
        tiling.qTile != 128 || tiling.kvTile != 128 || tiling.usedCoreNum == 0 ||
        tiling.usedCoreNum > LEGACY_VARLEN_CORES || info.kvHeadNum > 8U) {
        return;
    }
    const bool encoder = !causal && tiling.groupSize == 1 && info.batch >= 64 && info.batch <= 512 &&
        info.qSeqlen > 128 && info.qSeqlen <= 384 && tiling.legacyVarlenSchedule == 0;
    const bool split = causal && tiling.legacyVarlenSchedule == 2U;
    const uint32_t logicalHeads = static_cast<uint32_t>(info.kvHeadNum) * (split ? 2U : 1U);
    const bool gqa = tiling.groupSize > 1 && info.batch <= 32 && logicalHeads <= 8 &&
        (tiling.legacyVarlenSchedule == 1U || split);
    if (!encoder && !gqa) {
        return;
    }
    uint64_t packed = 0;
    for (uint64_t b = 0; b < info.batch; ++b) {
        const int64_t length = info.actualSeqQ[b];
        if (length <= 0 || length != info.actualSeqKv[b]) {
            return;
        }
        const uint64_t cols = CeilDivU64(static_cast<uint64_t>(length), 128);
        if (encoder && (cols > 8 || cols * info.kvHeadNum > tiling.usedCoreNum)) {
            return;
        }
        packed += static_cast<uint64_t>(length);
    }
    if (packed != info.totalQ || packed != info.totalKv) {
        return;
    }
    uint32_t scheduledWork[LEGACY_VARLEN_CORES] = {};
    for (uint32_t b = 0; b < info.batch; ++b) {
        const auto cols = static_cast<uint32_t>(CeilDivU64(info.actualSeqQ[b], 128));
        for (uint32_t head = 0; head < logicalHeads; ++head) {
            uint32_t headOwner = 0;
            if (encoder) {
                headOwner = static_cast<uint32_t>(std::min_element(scheduledWork,
                    scheduledWork + tiling.usedCoreNum) - scheduledWork);
                scheduledWork[headOwner] += cols * cols;
            }
            for (uint32_t col = 0; col < cols; ++col) {
                uint32_t core = headOwner;
                if (gqa) {
                    core = 0;
                    const uint32_t headId = b * logicalHeads + head;
                    while (core < tiling.usedCoreNum &&
                           (tiling.legacyVarlenColumns[core][headId] & (1U << col)) == 0) {
                        ++core;
                    }
                    if (core == tiling.usedCoreNum) {
                        return;
                    }
                }
                const uint16_t count = tiling.columnPlanCount[core];
                if (count == FAGTilingData::MAX_PLANNED_COLUMNS) {
                    return;
                }
                const uint32_t entry = encoder ? ((b << 6) | (head << 3) | col)
                                               : ((b << 8) | (head << 5) | col);
                tiling.columnPlan[core][count] = static_cast<uint16_t>(entry);
                tiling.columnPlanCount[core] = count + 1;
            }
        }
    }
    tiling.columnPlanKind = encoder ? 3U : (causal ? 4U : 2U);
}

// BN2S2 UB budget: the main pipeline owns two ping/pong halves and the dk/dv
// cast owns a chunked tail region (mirrors the kernel Init formula).
int64_t CheckBn2s2CastUb(FAGTilingData& tiling, const FAGInfo& info)
{
    const CastUbPlan plan = PlanBn2s2CastUb(tiling.qTile, tiling.kvTile, info.qkHeadDim, info.vHeadDim, info.ubSize);
    tiling.detCastChunkRows = plan.chunkRows;
    if (plan.chunkRows == 0) {
        fprintf(stderr, "FAG950 det bn2s2: UB too small for one cast row: pipeline %llu, UB %llu\n",
                (unsigned long long)plan.pipelineBytes, (unsigned long long)info.ubSize);
        return -1;
    }
    return 0;
}

// The legacy AIV pipeline keeps a 128-element physical UB row stride even
// when the L1 task width is 64. This preserves the existing VF kernels.
int64_t CheckLegacyUb(const FAGTilingData& tiling, bool singleVec = false, uint32_t extraBytes = 0, bool packedScalars = false)
{
    const uint32_t pipelineUb = FagUbLayout::PipelineBytes(tiling.qTile, singleVec, packedScalars) + extraBytes;
    const uint32_t deltaUb = 2 * FagUbLayout::DeltaSlotBytes(tiling.vHeadDim);
    if (pipelineUb > tiling.ubSize || deltaUb > tiling.ubSize) {
        fprintf(stderr, "FAG950 bwd: UB too small: pipeline %llu, delta %llu, capacity %llu\n",
                (unsigned long long)pipelineUb, (unsigned long long)deltaUb, (unsigned long long)tiling.ubSize);
        return -1;
    }
    return 0;
}

} // namespace

int64_t GetFAGTilingParam(const FAGInfo& info, FAGTilingData& tiling, bool canWriteDkvDirect)
{
    if (info.batch == 0 || info.qSeqlen == 0 || info.kvSeqlen == 0 || info.totalQ == 0 || info.totalKv == 0 ||
        info.qHeadNum == 0 || info.kvHeadNum == 0 || info.qHeadNum % info.kvHeadNum != 0 || info.qkHeadDim == 0 ||
        info.vHeadDim == 0 || info.aicNum == 0 || info.aivNum == 0 || info.continuousBlockNum == 0 ||
        info.ubSize == 0) {
        return -1;
    }

    tiling = FAGTilingData{};
    tiling.layout = static_cast<uint32_t>(info.layout);
    tiling.s1Token = info.window_size_left;
    tiling.s2Token = info.window_size_right;
    tiling.maskType = static_cast<uint32_t>(info.maskType);

    if (info.maskType == MaskType::NO_MASK) {
        tiling.s1Token = INT32_MAX;
        tiling.s2Token = INT32_MAX;
    } else if (info.maskType == MaskType::CAUSAL) {
        tiling.s1Token = INT32_MAX;
        tiling.s2Token = 0;
    } else if (info.maskType == MaskType::BAND) {
        tiling.s1Token = (info.window_size_left < 0) ? INT32_MAX : info.window_size_left;
        tiling.s2Token = (info.window_size_right < 0) ? INT32_MAX : info.window_size_right;
    }

    tiling.deterministic = info.deterministic;
    tiling.detSchedule = info.deterministic ? info.detSchedule : static_cast<uint32_t>(DetSchedule::LEGACY);
    tiling.aicNum = info.aicNum;
    tiling.aivNum = info.aivNum;
    // tiling.
    tiling.continuousBlockNum = info.continuousBlockNum;
    tiling.ubSize = info.ubSize;
    tiling.batch = info.batch;
    tiling.qSeqlen = info.qSeqlen;
    tiling.kvSeqlen = info.kvSeqlen;
    tiling.totalQ = info.totalQ;   // B * S
    tiling.totalKv = info.totalKv; // B * S
    tiling.qHeadNum = info.qHeadNum;
    tiling.kvHeadNum = info.kvHeadNum;
    tiling.groupSize = info.qHeadNum / info.kvHeadNum;
    tiling.qkHeadDim = info.qkHeadDim;
    tiling.vHeadDim = info.vHeadDim;
    tiling.scaleValue = info.scaleValue;
    tiling.softcapValue = info.softcapValue;

    // Default kvTile=128. shortQ keeps 128 (larger kvTile was tried and dropped).
    // VF epilogues use a 128-col chunk loop with row pitch = kvTile.
    // qTile shrinks for short S1. D>256 uses the 128x256 partial schedule.
    // (A/B: mid-seq qTile=64 deepened lag but slowed S>=1024 ~15–50%.)
    const uint64_t maxHeadDim = std::max(info.qkHeadDim, info.vHeadDim);
    const uint64_t alignedQk = RoundUpU64(info.qkHeadDim, 16);
    const uint64_t alignedV = RoundUpU64(info.vHeadDim, 16);
    tiling.d512Partial = 0;
    tiling.qTile = 128;
    tiling.kvTile = 128;
    tiling.shortBiDir = 0;
    tiling.shortBiDirUniform = 0;
    tiling.shortQ = 0;
    tiling.shortQUniformS1 = 0;
    if (maxHeadDim > 256) {
        // Precondition for the 128x256 partial schedule: Q/Dy keep a 256-wide
        // half in one 64KB slot and the high halves stream through the 128KB
        // scratch (RES_KT + RES_VT). Both budgets hold once the aligned head
        // dim is <= 512:
        //   full K[s2, D]  : 128 * 512 * 2B = 128KB <= scratch
        //   Q/dY high half : 128 * 256 * 2B =  64KB <= slot
        constexpr uint64_t MAX_ALIGNED_D = 512;
        if (alignedQk > MAX_ALIGNED_D || alignedV > MAX_ALIGNED_D) {
            fprintf(stderr,
                    "FAG950 bwd: D>256 partial path requires aligned head dim <= %llu, "
                    "got qk=%llu v=%llu\n",
                    (unsigned long long)MAX_ALIGNED_D, (unsigned long long)alignedQk, (unsigned long long)alignedV);
            return -1;
        }
        tiling.d512Partial = 1;
    }

    // shortQ/shortBiDir only on the non-det full-residency path (not d512Partial).
    if (!info.deterministic && tiling.d512Partial == 0) {
        uint64_t maxS1 = info.qSeqlen;
        uint64_t maxS2 = info.kvSeqlen;
        bool uniformS1 = true;
        bool uniformS2 = true;
        if (info.layout == Layout::TND && info.batch > 0) {
            if (info.actualSeqQ != nullptr) {
                maxS1 = 0;
                for (uint64_t b = 0; b < info.batch; ++b) {
                    const uint64_t s = static_cast<uint64_t>(info.actualSeqQ[b] > 0 ? info.actualSeqQ[b] : 0);
                    if (s > maxS1) {
                        maxS1 = s;
                    }
                    if (b > 0 && s != static_cast<uint64_t>(info.actualSeqQ[0] > 0 ? info.actualSeqQ[0] : 0)) {
                        uniformS1 = false;
                    }
                }
            }
            if (info.actualSeqKv != nullptr) {
                maxS2 = 0;
                for (uint64_t b = 0; b < info.batch; ++b) {
                    const uint64_t s = static_cast<uint64_t>(info.actualSeqKv[b] > 0 ? info.actualSeqKv[b] : 0);
                    if (s > maxS2) {
                        maxS2 = s;
                    }
                    if (b > 0 && s != static_cast<uint64_t>(info.actualSeqKv[0] > 0 ? info.actualSeqKv[0] : 0)) {
                        uniformS2 = false;
                    }
                }
            }
        }
        if (maxS1 > 0 && maxS1 <= 16) {
            tiling.qTile = 16;
        } else if (maxS1 <= 32) {
            tiling.qTile = 32;
        } else if (maxS1 <= 64) {
            tiling.qTile = 64;
        }
        // Uniform shortcuts use packed region lengths directly. Used-length
        // caps require the per-batch decoder even when those regions are equal.
        // Short bidirectional: one S1/S2 tile per (b,n2,g). Isolate the
        // scalar-oriented schedule so longer S1/S2 shapes stay on legacy.
        if (maxS1 > 0 && maxS1 <= 16 && maxS2 > 0 && maxS2 <= 16) {
            tiling.shortBiDir = 1;
            if (!info.hasUsedLengths && uniformS1 && uniformS2 && info.batch > 0 && maxS1 > 0 && maxS2 > 0 &&
                info.totalQ == info.batch * maxS1 && info.totalKv == info.batch * maxS2) {
                tiling.shortBiDirUniform = 1;
            }
        } else if (maxS1 > 0 && maxS1 <= 16 && maxS2 > 16 && info.batch > 0) {
            // Short-Q long-KV scene: core-strided (b,n2); S2 inner.
            // Gate on maxS1/S2 only (from actualSeq or max_seqlen). Runtime
            // maps gid → (b,n2) as gid/kvHeadNum (kvHeadNum is global).
            // kvTile stays 128 (256 N-split wiped the win).
            tiling.shortQ = 1;
            if (!info.hasUsedLengths && uniformS1 && info.totalQ == info.batch * maxS1) {
                tiling.shortQUniformS1 = 1;
            }
        }
    }

    constexpr uint64_t L1_SIZE = 512 * 1024;
    constexpr uint64_t L1_CUBE_SLOT_BYTES = 64 * 1024;
    // Full-residency path budgets the full D column into one cube slot.
    // Partial path only keeps a 256-wide half in Q/Dy; scratch holds the rest.
    const uint64_t residentCols = tiling.d512Partial ? 256ULL : std::max(alignedQk, alignedV);
    const uint64_t maxOperandBytes = residentCols * std::max(tiling.qTile, tiling.kvTile) * sizeof(uint16_t);
    const uint64_t pdsBytes = 4ULL * tiling.qTile * tiling.kvTile * sizeof(uint16_t);
    if (maxOperandBytes > L1_CUBE_SLOT_BYTES || 6 * L1_CUBE_SLOT_BYTES + pdsBytes > L1_SIZE) {
        fprintf(stderr, "FAG950 bwd: invalid L1 tiling: operand %llu, P/dS %llu\n", (unsigned long long)maxOperandBytes,
                (unsigned long long)pdsBytes);
        return -1;
    }
    const bool singleVec = tiling.shortQ != 0 || tiling.shortBiDir != 0;
    if (CheckLegacyUb(tiling, singleVec) != 0) {
        return -1;
    }
    if (!info.deterministic && !singleVec) {
        // Match CANN dense interleaving granularity (fusion arch35 mid-seq).
        // shortQ/shortBiDir keep DEFAULT_CONTINUOUS_BLOCK_NUM; MLA D192/Dv128
        // and plain D<=128 both benefit from coarser core-strided waves.
        if ((info.qkHeadDim == 192 && info.vHeadDim == 128) ||
            (info.qkHeadDim <= 128 && info.vHeadDim <= 128)) {
            tiling.continuousBlockNum = info.qSeqlen < 16384 ? 6 : (info.qSeqlen / 16384) * 8;
        }
    }

    tiling.usedCoreNum = info.aicNum; // TODO: 是否需要和实际task取min
    if (!info.deterministic && !singleVec) {
        PlanLegacyVarlenColumns(info, tiling);
        PlanColumnTasks(info, tiling);
    }

    // Deterministic knobs are only meaningful when the deterministic path is
    // enabled; keep them zeroed otherwise so the legacy layout is untouched.
    tiling.dqPostAbsorb = info.deterministic ? info.dqPostAbsorb : 0;
    tiling.dqVecNum = info.deterministic ? info.dqVecNum : 0;
    tiling.dkVecNum = info.deterministic ? info.dkVecNum : 0;
    tiling.dvVecNum = info.deterministic ? info.dvVecNum : 0;
    tiling.earlyDkvCast = 0;
    tiling.skipDkvClear = 0;
    tiling.detCastChunkRows = 0;

    const uint64_t dAlign = RoundUpU64(info.qkHeadDim, FP32_ROW_ALIGN);
    const uint64_t dvAlign = RoundUpU64(info.vHeadDim, FP32_ROW_ALIGN);

    if (!info.deterministic) {
        // Short-Q schedules avoid a separate dK/dV post pass. On the
        // regular schedule, small dK/dV outputs do not amortize the strided
        // Fixpipe cast/write (A/B: ungated canWriteDkvDirect slowed mid-seq
        // ~5%). Use four full KV tiles per AIC as the crossover for the total
        // packed output, counting KV heads (not query heads).
        constexpr uint64_t MIN_KV_TILES_PER_CORE = 4;
        const uint64_t kvRows = tiling.totalKv * tiling.kvHeadNum;
        // Gate on full AIC count (A/B: ungating earlyDkv slowed mid-seq ~5%).
        const bool largeDkv = kvRows >= MIN_KV_TILES_PER_CORE * info.aicNum * tiling.kvTile;
        const bool directDkvEligible = canWriteDkvDirect && (singleVec || largeDkv) &&
                                       tiling.legacyVarlenSchedule != 2U;
        const bool earlyDkv = directDkvEligible && info.fixpipeDkScaleExact;
        tiling.earlyDkvCast = earlyDkv ? 1 : 0;
        // dV has no scale. Keep its direct store when dK needs FP32 post scaling.
        tiling.wideDvDirect = directDkvEligible && !earlyDkv;
        // D<=128 Fixpipe overwrites each visited column (enAtomicDk/Dv=false).
        // When every column is covered, skip the fp32 dk/dv Pre clear.
        tiling.skipDkvClear =
            (!earlyDkv && canWriteDkvDirect && info.qkHeadDim <= 128 && info.vHeadDim <= 128 &&
             tiling.legacyVarlenSchedule != 2U) ? 1 : 0;

        // A full or square causal TND schedule visits every KV column.
        // For sliced 256D dK/dV the first contribution overwrites both slices.
        bool uniform512 = true;
        if (info.layout == Layout::TND && info.qkHeadDim == 256 && info.vHeadDim == 256 &&
            info.qSeqlen >= 512 &&
            !info.hasUsedLengths && info.actualSeqQ != nullptr && info.actualSeqKv != nullptr &&
            (info.maskType == MaskType::NO_MASK || info.maskType == MaskType::CAUSAL)) {
            bool covered = true;
            uint64_t coveredTokens = 0;
            for (uint64_t b = 0; b < info.batch; ++b) {
                const int64_t qLength = info.actualSeqQ[b];
                if (qLength <= 0 || qLength != info.actualSeqKv[b]) {
                    covered = false;
                    break;
                }
                coveredTokens += static_cast<uint64_t>(qLength);
                uniform512 &= qLength == 512;
            }
            tiling.skipDkvClear = covered && coveredTokens == info.totalQ && coveredTokens == info.totalKv ? 1U : 0U;
        }

        // At least one complete balanced wave. A head owns all its dQ contributions.
        const uint64_t heads = info.batch * info.qHeadNum;
        tiling.wideHeadOwn = uniform512 && tiling.skipDkvClear && !earlyDkv && info.qkHeadDim == 256 &&
            info.vHeadDim == 256 && tiling.groupSize == 1 && info.softcapValue == 0 &&
            info.qSeqlen == 512 && info.kvSeqlen == 512 && info.totalQ == info.batch * 512 &&
            info.totalKv == info.batch * 512 && tiling.legacyVarlenSchedule == 0 &&
            heads >= info.aicNum && heads % info.aicNum == 0;

        if (tiling.wideHeadOwn == 0 && tiling.skipDkvClear && !earlyDkv &&
            info.layout == Layout::TND && info.qkHeadDim == 256 && info.vHeadDim == 256 &&
            info.maskType == MaskType::NO_MASK && info.softcapValue == 0 && tiling.groupSize == 1 &&
            info.qSeqlen == 1024 && info.kvSeqlen == 1024 &&
            heads >= info.aicNum && heads % info.aicNum == 0 &&
            tiling.legacyVarlenSchedule == 0 &&
            std::all_of(info.actualSeqQ, info.actualSeqQ + info.batch,
                [&](int64_t length) { return static_cast<uint64_t>(length) == info.qSeqlen; })) {
            tiling.wideHeadOwn = 1;
        }

        // Each KV column is assigned to a single core and visited in order.
        // The existing clear-elision proof gives full coverage and equal Q/KV lengths.
        tiling.wideDvResident = tiling.skipDkvClear && info.layout == Layout::TND &&
            info.qkHeadDim == 256 && info.vHeadDim == 256 && info.softcapValue == 0 &&
            tiling.groupSize == 1 && info.qSeqlen >= 512 && tiling.legacyVarlenSchedule == 0;

        const bool canKeepD192Dv = info.layout == Layout::TND && info.qkHeadDim == 192 &&
            info.vHeadDim == 192 && info.softcapValue == 0 && !info.hasUsedLengths &&
            (info.maskType == MaskType::NO_MASK || info.maskType == MaskType::CAUSAL) &&
            info.qSeqlen == 256 && info.kvSeqlen == 256 && info.totalQ == info.batch * 256 &&
            info.totalKv == info.batch * 256 && info.actualSeqQ != nullptr && info.actualSeqKv != nullptr &&
            tiling.legacyVarlenSchedule == 0;
        if (canKeepD192Dv) {
            bool exact = true;
            for (uint64_t b = 0; b < info.batch; ++b) {
                exact &= info.actualSeqQ[b] == 256 && info.actualSeqKv[b] == 256;
            }
            tiling.wideDvResident = exact;
        }
        if (info.qkHeadDim == 192 && info.vHeadDim == 192 && tiling.legacyVarlenSchedule == 2U) {
            // Each segment accumulates in L0C and atomically contributes once at flush.
            tiling.wideDvResident = 1;
        }

        // Reserve one 64-row FP32 slab and a 16-row conversion buffer per AIV.
        constexpr uint32_t dkUbBytes = 64U * 128U * sizeof(float) + 16U * 128U * sizeof(uint16_t);
        const bool uniformSmallMha = tiling.wideDvDirect != 0 && tiling.columnPlanKind == 0 &&
            tiling.legacyVarlenSchedule == 0 && tiling.groupSize == 1 && !singleVec &&
            info.qkHeadDim == 128 && info.vHeadDim == 128 && tiling.qTile == 128 && tiling.kvTile == 128 &&
            (info.qSeqlen >= 256 && info.qSeqlen <= 1024) && info.qSeqlen == info.kvSeqlen &&
            info.totalQ == info.batch * info.qSeqlen && info.totalKv == info.totalQ &&
            info.actualSeqQ != nullptr && info.actualSeqKv != nullptr &&
            CheckLegacyUb(tiling, false, dkUbBytes, true) == 0;
        tiling.dkCastInUb = uniformSmallMha &&
            std::all_of(info.actualSeqQ, info.actualSeqQ + info.batch,
                        [&](int64_t length) { return static_cast<uint64_t>(length) == info.qSeqlen; }) &&
            std::all_of(info.actualSeqKv, info.actualSeqKv + info.batch,
                        [&](int64_t length) { return static_cast<uint64_t>(length) == info.kvSeqlen; });

        uint64_t dqWsSize = tiling.totalQ * tiling.qHeadNum * tiling.qkHeadDim * FP32_BYTES;
        uint64_t dkWsSize = (earlyDkv || tiling.dkCastInUb != 0) ? 0ULL : tiling.totalKv * tiling.kvHeadNum * tiling.qkHeadDim * FP32_BYTES;
        uint64_t dvWsSize = (earlyDkv || tiling.wideDvDirect != 0) ? 0ULL : tiling.totalKv * tiling.kvHeadNum * tiling.vHeadDim * FP32_BYTES;
        uint64_t deltaWsSize = tiling.totalQ * tiling.qHeadNum * 8;

        tiling.dqOffset = 0;
        tiling.dkOffset = tiling.dqOffset + dqWsSize;
        tiling.dvOffset = tiling.dkOffset + dkWsSize;
        tiling.deltaOffset = tiling.dvOffset + dvWsSize;
        tiling.workspaceSize = tiling.deltaOffset + deltaWsSize;
        tiling.dkPrivOffset = 0;
        tiling.dvPrivOffset = 0;
        return 0;
    }

    // ------------------------------------------------------------------
    // BN2S2 deterministic schedule (opst arch35 column-private accumulation).
    // ------------------------------------------------------------------
    if (tiling.detSchedule == static_cast<uint32_t>(DetSchedule::BN2S2)) {
        if (info.layout == Layout::TND) {
            // Ragged TND.  MHA uses the column-private swizzle (opst
            // CalTNDDenseSwizzleIndex, per-batch round prefix); GQA uses
            // opst's flat blocked partition (CalTNDDenseIndex !IS_N_EQUAL,
            // cumulative-area prefix) with shared dk/dv workspaces because a
            // column may straddle two lanes' slices.  Causal reuses the same
            // dense schedules with the causal mask (masked blocks add exact
            // zeros), like the rectangular BSND causal path.
            if (info.batch + 1 > TND_SWIZZLE_PREFIX_NUM || info.actualSeqQ == nullptr || info.actualSeqKv == nullptr) {
                fprintf(stderr, "FAG950 det bn2s2 tnd: unsupported TND shape (B=%llu)\n",
                        (unsigned long long)info.batch);
                return -1;
            }
            const bool gqa = tiling.groupSize != 1;
            // Causal TND with per-batch sq == sk: opst's left-up causal schedule
            // (CalTNDCausalIndex) then covers exactly the same block triangle as
            // the right-down element mask, so it can replace dense+mask without
            // changing semantics (the mask epilogue still handles the diagonal
            // tiles).  Any batch with sq != sk keeps the dense schedule + mask,
            // which is the only correct choice under right-down alignment.
            if (info.maskType == MaskType::CAUSAL && !gqa) {
                bool allSquare = true;
                for (uint32_t b = 0; b < info.batch; ++b) {
                    if (info.actualSeqQ[b] != info.actualSeqKv[b]) {
                        allSquare = false;
                        break;
                    }
                }
                if (allSquare) {
                    const fag_det_host::TndCausalParams tc = fag_det_host::ComputeTndCausalParams(
                        static_cast<int64_t>(info.batch), info.actualSeqQ, info.actualSeqKv,
                        static_cast<int64_t>(info.kvHeadNum), static_cast<int64_t>(info.aicNum), tiling.qTile,
                        tiling.kvTile);
                    if (tc.supported) {
                        for (int64_t b = 0; b <= static_cast<int64_t>(info.batch) + 2 &&
                                            b < static_cast<int64_t>(TND_SWIZZLE_PREFIX_NUM);
                             ++b) {
                            tiling.tndPrefix[b] = tc.p0[b];
                        }
                        for (int64_t b = 0; b <= static_cast<int64_t>(info.batch) + 1 &&
                                            b < static_cast<int64_t>(TND_SWIZZLE_PREFIX_NUM);
                             ++b) {
                            tiling.tndCausalP1[b] = tc.p1[b];
                            tiling.tndCausalP2[b] = tc.p2[b];
                        }
                        tiling.detKind = fag_det_host::KIND_TND_CAUSAL;
                        tiling.detColumnRounds = 0;
                        tiling.detBufNum = 1;
                        // Columns may be covered by different lanes in
                        // disjoint rounds, so dk/dv must be shared across lanes
                        // (round barrier supplies the deterministic order).
                        tiling.detPrivDkv = 0;
                        tiling.detMaxRound = static_cast<uint64_t>(tc.maxRound);
                        tiling.dqPostAbsorb = 0;
                        tiling.continuousBlockNum = 1;
                        tiling.dqVecNum = 0;
                        tiling.dkVecNum = 0;
                        tiling.dvVecNum = 0;
                        const uint64_t dqWsSize = tiling.totalQ * tiling.qHeadNum * dAlign * FP32_BYTES;
                        const uint64_t deltaWsSize = tiling.totalQ * tiling.qHeadNum * 8;
                        tiling.dqOffset = MULTI_CORE_SYNC_BYTES;
                        const uint64_t dkWsSize = tiling.totalKv * tiling.kvHeadNum * dAlign * FP32_BYTES;
                        const uint64_t dvWsSize = tiling.totalKv * tiling.kvHeadNum * dvAlign * FP32_BYTES;
                        tiling.dkOffset = tiling.dqOffset + RoundUpU64(dqWsSize, GM_ALIGNMENT);
                        tiling.dvOffset = tiling.dkOffset + RoundUpU64(dkWsSize, GM_ALIGNMENT);
                        tiling.deltaOffset = tiling.dvOffset + RoundUpU64(dvWsSize, GM_ALIGNMENT);
                        tiling.workspaceSize = tiling.deltaOffset + RoundUpU64(deltaWsSize, GM_ALIGNMENT);
                        tiling.dqDetOffset = 0;
                        tiling.dkDetOffset = 0;
                        tiling.dvDetOffset = 0;
                        tiling.dkPrivOffset = 0;
                        tiling.dvPrivOffset = 0;
                        return 0;
                    }
                }
            }
            // MHA swizzle requires s1Outer >= min(k, s2Outer) for every batch
            // (intra-round dq uniqueness).  Ragged MHA shapes that violate it
            // take the same flat blocked partition as GQA with the shared
            // dk/dv workspaces (a column may straddle two lanes' slices).
            const bool flat =
                gqa || !fag_det_host::TndDenseSafe(static_cast<int64_t>(info.batch), info.actualSeqQ, info.actualSeqKv,
                                                   static_cast<int64_t>(info.kvHeadNum),
                                                   static_cast<int64_t>(info.aicNum), tiling.qTile, tiling.kvTile);
            int64_t prefix[TND_SWIZZLE_PREFIX_NUM] = {0};
            int64_t maxRound = 0;
            if (flat) {
                int64_t s1Max = 0;
                int64_t s2Max = 0;
                for (int64_t b = 0; b < static_cast<int64_t>(info.batch); ++b) {
                    const int64_t s1Outer = (info.actualSeqQ[b] + tiling.qTile - 1) / tiling.qTile;
                    const int64_t s2Outer = (info.actualSeqKv[b] + tiling.kvTile - 1) / tiling.kvTile;
                    if (s1Outer <= 0 || s2Outer <= 0) {
                        fprintf(stderr, "FAG950 det bn2s2 tnd: empty sequence\n");
                        return -1;
                    }
                    prefix[b + 1] = prefix[b] + s1Outer * s2Outer;
                    s1Max = s1Outer > s1Max ? s1Outer : s1Max;
                    s2Max = s2Outer > s2Max ? s2Outer : s2Max;
                }
                const int64_t k = static_cast<int64_t>(info.aicNum);
                const int64_t g = static_cast<int64_t>(tiling.groupSize);
                const int64_t total = prefix[info.batch] * static_cast<int64_t>(info.kvHeadNum) * g;
                maxRound = static_cast<int64_t>(
                    std::max({CeilDivU64(total, k), static_cast<uint64_t>(s1Max * g), static_cast<uint64_t>(s2Max)}));
            } else {
                maxRound = fag_det_host::TndDensePrefix(
                    static_cast<int64_t>(info.batch), info.actualSeqQ, info.actualSeqKv,
                    static_cast<int64_t>(info.kvHeadNum), // g == 1
                    static_cast<int64_t>(info.aicNum), tiling.qTile, tiling.kvTile, prefix);
            }
            if (maxRound <= 0) {
                fprintf(stderr, "FAG950 det bn2s2 tnd: empty schedule\n");
                return -1;
            }
            for (uint32_t b = 0; b <= info.batch && b < TND_SWIZZLE_PREFIX_NUM; ++b) {
                tiling.tndPrefix[b] = prefix[b];
            }
            tiling.detKind = flat ? fag_det_host::KIND_TND_GQA_DENSE : fag_det_host::KIND_TND_DENSE;
            tiling.detColumnRounds = 0; // column end uses coordinate comparison
            tiling.detBufNum = 1;
            tiling.detPrivDkv = flat ? 0U : 1U;
            tiling.detMaxRound = static_cast<uint64_t>(maxRound);
            tiling.dqPostAbsorb = 0;
            tiling.continuousBlockNum = 1;
            tiling.dqVecNum = 0;
            tiling.dkVecNum = 0;
            tiling.dvVecNum = 0;

            const uint64_t dqWsSize = tiling.totalQ * tiling.qHeadNum * dAlign * FP32_BYTES;
            const uint64_t deltaWsSize = tiling.totalQ * tiling.qHeadNum * 8;
            tiling.dqOffset = MULTI_CORE_SYNC_BYTES;
            if (flat) {
                // Shared dk/dv workspaces (same as the BSND GQA path).
                const uint64_t dkWsSize = tiling.totalKv * tiling.kvHeadNum * dAlign * FP32_BYTES;
                const uint64_t dvWsSize = tiling.totalKv * tiling.kvHeadNum * dvAlign * FP32_BYTES;
                tiling.dkOffset = tiling.dqOffset + RoundUpU64(dqWsSize, GM_ALIGNMENT);
                tiling.dvOffset = tiling.dkOffset + RoundUpU64(dkWsSize, GM_ALIGNMENT);
                tiling.deltaOffset = tiling.dvOffset + RoundUpU64(dvWsSize, GM_ALIGNMENT);
                tiling.workspaceSize = tiling.deltaOffset + RoundUpU64(deltaWsSize, GM_ALIGNMENT);
                tiling.dqDetOffset = 0;
                tiling.dkDetOffset = 0;
                tiling.dvDetOffset = 0;
                tiling.dkPrivOffset = 0;
                tiling.dvPrivOffset = 0;
                return 0;
            }
            const uint64_t dkPrivSize = static_cast<uint64_t>(info.aicNum) * tiling.kvTile * dAlign * FP32_BYTES;
            const uint64_t dvPrivSize = static_cast<uint64_t>(info.aicNum) * tiling.kvTile * dvAlign * FP32_BYTES;
            tiling.deltaOffset = tiling.dqOffset + RoundUpU64(dqWsSize, GM_ALIGNMENT);
            tiling.dkPrivOffset = tiling.deltaOffset + RoundUpU64(deltaWsSize, GM_ALIGNMENT);
            tiling.dvPrivOffset = tiling.dkPrivOffset + RoundUpU64(dkPrivSize, GM_ALIGNMENT);
            tiling.workspaceSize = tiling.dvPrivOffset + RoundUpU64(dvPrivSize, GM_ALIGNMENT);
            tiling.dkOffset = 0;
            tiling.dvOffset = 0;
            tiling.dqDetOffset = 0;
            tiling.dkDetOffset = 0;
            tiling.dvDetOffset = 0;
            return CheckBn2s2CastUb(tiling, info);
        }
        if (info.layout != Layout::BSND && info.layout != Layout::BNSD) {
            fprintf(stderr, "FAG950 det bn2s2: only BSND/TND are supported\n");
            return -1;
        }
        const int64_t bh = static_cast<int64_t>(info.batch) * static_cast<int64_t>(info.kvHeadNum);
        const int64_t m = (static_cast<int64_t>(info.qSeqlen) + static_cast<int64_t>(tiling.qTile) - 1) /
                          static_cast<int64_t>(tiling.qTile);
        const int64_t n = (static_cast<int64_t>(info.kvSeqlen) + static_cast<int64_t>(tiling.kvTile) - 1) /
                          static_cast<int64_t>(tiling.kvTile);
        const int64_t g = static_cast<int64_t>(tiling.groupSize);
        const int64_t k = static_cast<int64_t>(info.aicNum);

        const bool causal = info.maskType == MaskType::CAUSAL;
        const fag_det_host::Selection sel = fag_det_host::SelectSchedule(causal, bh, m, n, g, k);
        if (!sel.supported) {
            fprintf(stderr,
                    "FAG950 det bn2s2: no schedule for causal=%d B=%llu N2=%llu "
                    "m=%lld n=%lld g=%lld k=%d\n",
                    causal ? 1 : 0, (unsigned long long)info.batch, (unsigned long long)info.kvHeadNum, (long long)m,
                    (long long)n, (long long)g, info.aicNum);
            return -1;
        }
        tiling.detKind = sel.kind;
        tiling.detColumnRounds = static_cast<uint32_t>(fag_det_host::ScheduleColumnRounds(sel.kind, m, n));
        tiling.detBufNum = fag_det_host::ScheduleBufNum(sel.kind);
        // GQA shares one dk/dv column between the g query heads, so the
        // per-core private buffer + early cast only applies to MHA (g == 1);
        // GQA keeps the ordered atomic accumulation in the full workspaces.
        tiling.detPrivDkv = (g == 1) ? 1U : 0U;
        // Idle-core trim: small shapes need far fewer cores than the device
        // has, yet every launched core pays the deterministic prologue
        // (event/flag init, private dk/dv slot setup) plus one grid barrier
        // per round.  Keep the round count identical and launch only the
        // smallest core count that sustains it; the per-core task assignment
        // for the kept cores is unchanged because the decode is a function of
        // (round, core) and the core index stays 0..activeCore-1.
        const int64_t maxRound = fag_det_host::ScheduleMaxRound(sel.kind, bh, m, n, g, k);
        int64_t activeCore = k;
        for (int64_t c = 1; c < k; ++c) {
            if (fag_det_host::ScheduleMaxRound(sel.kind, bh, m, n, g, c) == maxRound) {
                activeCore = c;
                break;
            }
        }
        tiling.usedCoreNum = static_cast<uint32_t>(activeCore);
        tiling.detMaxRound = static_cast<uint64_t>(maxRound);
        tiling.dqPostAbsorb = 0;
        // One scheduled task per cube core per round; the paired AIVs are
        // driven by the existing V1/V2 pipeline plus the per-column dk/dv
        // cast, and the VecDTM group split is unused.
        tiling.continuousBlockNum = 1;
        tiling.dqVecNum = 0;
        tiling.dkVecNum = 0;
        tiling.dvVecNum = 0;

        const uint64_t dqWsSize = tiling.totalQ * tiling.qHeadNum * dAlign * FP32_BYTES;
        const uint64_t deltaWsSize = tiling.totalQ * tiling.qHeadNum * 8;
        if (!tiling.detPrivDkv) {
            // Stage-1 layout (GQA): dq/dk/dv use the full fp32 workspaces.
            const uint64_t dkWsSize = tiling.totalKv * tiling.kvHeadNum * dAlign * FP32_BYTES;
            const uint64_t dvWsSize = tiling.totalKv * tiling.kvHeadNum * dvAlign * FP32_BYTES;
            tiling.dqOffset = MULTI_CORE_SYNC_BYTES;
            tiling.dkOffset = tiling.dqOffset + RoundUpU64(dqWsSize, GM_ALIGNMENT);
            tiling.dvOffset = tiling.dkOffset + RoundUpU64(dkWsSize, GM_ALIGNMENT);
            tiling.deltaOffset = tiling.dvOffset + RoundUpU64(dvWsSize, GM_ALIGNMENT);
            tiling.workspaceSize = tiling.deltaOffset + RoundUpU64(deltaWsSize, GM_ALIGNMENT);
            tiling.dqDetOffset = 0;
            tiling.dkDetOffset = 0;
            tiling.dvDetOffset = 0;
            tiling.dkPrivOffset = 0;
            tiling.dvPrivOffset = 0;
            return 0;
        }

        // Stage-2 layout (MHA): [sync 64KB][dq fp32 full][delta][dk priv][dv
        // priv].  dq still uses the full fp32 workspace (ordered atomic
        // accumulation); dk/dv accumulate in per-core private regions and are
        // converted to the bf16 outputs as soon as a column completes.
        const uint64_t dkPrivSize =
            static_cast<uint64_t>(info.aicNum) * tiling.detBufNum * tiling.kvTile * dAlign * FP32_BYTES;
        const uint64_t dvPrivSize =
            static_cast<uint64_t>(info.aicNum) * tiling.detBufNum * tiling.kvTile * dvAlign * FP32_BYTES;

        tiling.dqOffset = MULTI_CORE_SYNC_BYTES;
        tiling.deltaOffset = tiling.dqOffset + RoundUpU64(dqWsSize, GM_ALIGNMENT);
        tiling.dkPrivOffset = tiling.deltaOffset + RoundUpU64(deltaWsSize, GM_ALIGNMENT);
        tiling.dvPrivOffset = tiling.dkPrivOffset + RoundUpU64(dkPrivSize, GM_ALIGNMENT);
        tiling.workspaceSize = tiling.dvPrivOffset + RoundUpU64(dvPrivSize, GM_ALIGNMENT);
        tiling.dkOffset = 0;
        tiling.dvOffset = 0;
        tiling.dqDetOffset = 0;
        tiling.dkDetOffset = 0;
        tiling.dvDetOffset = 0;

        return CheckBn2s2CastUb(tiling, info);
    }

    // Deterministic layout:
    //   [sync 64KB][dq][dk][dv][delta][dqDet][dkDet][dvDet]
    // dq accumulates into a single rolling tile when dqPostAbsorb=1, into the
    // full S1*N1 region otherwise. det slots reserve two complete issue
    // rounds (aicNum * continuousBlockNum tiles per round) per gradient.
    // Alternating the round bank lets VecDTM(r) read one bank while C345(r+1)
    // writes the other, without a done-counter gate on slot reuse.

    // UB budget check.  The main pipeline owns two ping/pong halves starting
    // at UB+0 (formula mirrors fag_kernel.cpp Init); the VecDTM epilogue owns
    // a dedicated tail region (mirrors fag_epilogue_deterministic_add.hpp
    // Init).  Borrowing is NOT allowed: v2 overlaps VecDTM(r) with the next
    // round's C12 SPLIT_M fixpipe, which writes the ping/pong halves.
    // The VecDTM group mapping assumes the three vec groups all exist as
    // physical AIVs.
    if (info.dqVecNum == 0 || info.dkVecNum == 0 || info.dvVecNum == 0 ||
        info.dqVecNum + info.dkVecNum + info.dvVecNum > info.aivNum) {
        fprintf(stderr, "FAG950 det bwd: bad vec groups %u/%u/%u (aivNum %u)\n", info.dqVecNum, info.dkVecNum,
                info.dvVecNum, info.aivNum);
        return -1;
    }
    const uint64_t rowsPerSub = RoundUpU64(tiling.qTile, 16) / 2;
    const uint64_t mmResBytes = rowsPerSub * tiling.kvTile * FP32_BYTES;
    const uint64_t attenMaskBytes = rowsPerSub * tiling.kvTile;
    const uint64_t lseBytes = RoundUpU64(rowsPerSub, 8) * 8 * FP32_BYTES;
    const uint64_t pBytes = (rowsPerSub + 1) * tiling.kvTile * 2;
    const uint64_t deltaBytes = rowsPerSub * 8 * FP32_BYTES;
    const uint64_t lseOff = RoundUpU64(2 * mmResBytes + attenMaskBytes, 32);
    const uint64_t pOff = RoundUpU64(lseOff + lseBytes, 32);
    const uint64_t dSOff = RoundUpU64(pOff + pBytes, 32);
    const uint64_t deltaOff = RoundUpU64(dSOff + pBytes, 32);
    const uint64_t halfUb = RoundUpU64(deltaOff + deltaBytes, 32);
    const uint64_t pipelineUb = 2 * halfUb; // TASK_PINGPONG
    const uint64_t rowCapDq = CeilDivU64(tiling.qTile, info.dqVecNum);
    const uint64_t rowCapDk = CeilDivU64(tiling.kvTile, info.dkVecNum);
    const uint64_t rowCapDv = CeilDivU64(tiling.kvTile, info.dvVecNum);
    uint64_t rowCap = rowCapDq;
    if (rowCapDk > rowCap) {
        rowCap = rowCapDk;
    }
    if (rowCapDv > rowCap) {
        rowCap = rowCapDv;
    }
    const uint64_t colCap = dAlign > dvAlign ? dAlign : dvAlign;
    const uint64_t accUbBytes = RoundUpU64(rowCap * colCap * FP32_BYTES, 32);
    const uint64_t castUbBytes = RoundUpU64(rowCap * info.qkHeadDim * 2, 32);
    const uint64_t detAddUb = 2 * accUbBytes + castUbBytes;
    if (pipelineUb + detAddUb > info.ubSize) {
        fprintf(stderr, "FAG950 det bwd: UB too small: pipeline %llu + VecDTM %llu > %llu\n",
                (unsigned long long)pipelineUb, (unsigned long long)detAddUb, (unsigned long long)info.ubSize);
        return -1;
    }

    const uint64_t dqWsSize = tiling.dqPostAbsorb ? static_cast<uint64_t>(tiling.qTile) * dAlign * FP32_BYTES
                                                  : tiling.totalQ * tiling.qHeadNum * dAlign * FP32_BYTES;
    const uint64_t dkWsSize = tiling.totalKv * tiling.kvHeadNum * dAlign * FP32_BYTES;
    const uint64_t dvWsSize = tiling.totalKv * tiling.kvHeadNum * dvAlign * FP32_BYTES;
    const uint64_t deltaWsSize = tiling.totalQ * tiling.qHeadNum * 8;

    constexpr uint64_t detSlotBanks = 2;
    const uint64_t slotNum = detSlotBanks * static_cast<uint64_t>(info.aicNum) * info.continuousBlockNum;
    const uint64_t dqDetSize = RoundUpU64(slotNum * tiling.qTile * dAlign * FP32_BYTES, GM_ALIGNMENT);
    const uint64_t dkDetSize = RoundUpU64(slotNum * tiling.kvTile * dAlign * FP32_BYTES, GM_ALIGNMENT);
    const uint64_t dvDetSize = RoundUpU64(slotNum * tiling.kvTile * dvAlign * FP32_BYTES, GM_ALIGNMENT);

    tiling.dqOffset = MULTI_CORE_SYNC_BYTES;
    tiling.dkOffset = tiling.dqOffset + RoundUpU64(dqWsSize, GM_ALIGNMENT);
    tiling.dvOffset = tiling.dkOffset + RoundUpU64(dkWsSize, GM_ALIGNMENT);
    tiling.deltaOffset = tiling.dvOffset + RoundUpU64(dvWsSize, GM_ALIGNMENT);
    tiling.dqDetOffset = tiling.deltaOffset + RoundUpU64(deltaWsSize, GM_ALIGNMENT);
    tiling.dkDetOffset = tiling.dqDetOffset + dqDetSize;
    tiling.dvDetOffset = tiling.dkDetOffset + dkDetSize;
    tiling.workspaceSize = tiling.dvDetOffset + dvDetSize;
    return 0;
}

} // namespace FAGTiling950

#endif // FAG_TILING_950_CPP_H
