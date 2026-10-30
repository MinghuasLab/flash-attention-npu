/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "tilingdata.h"

namespace optiling {
const uint32_t SIZE_OF_16BIT = 2;
const uint32_t SIZE_OF_32BIT = 4;
const uint32_t N_SPLIT_HELPER = 2;
const uint32_t MAX_KV_STACK_LEN = 512;
const uint32_t Q_TILE_CEIL = 128;
const uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
const uint32_t BASE_KV_SIZE = 128;
const uint32_t PRELANCH_NUM = 3;
const uint64_t ASCEND950_L1_SIZE = 512ULL * 1024ULL;
const uint64_t RESCALE_OTMP_STAGE_SIZE = 32ULL * 1024ULL;
const uint64_t FD_WORKSPACE_ALIGNMENT = 512ULL;

enum class MaskType : uint32_t {
    NO_MASK = 0,
    MASK_SPEC = 1,
    MASK_BAND = 2
};

enum class DataType : uint32_t {
    FP16 = 0,
    BF16 = 1,
    FP8 = 2
};

struct FAInferContext {
    int32_t numTokens = 0;
    int32_t numHeads = 0;
    int32_t embeddingSize = 0;
    int32_t embeddingSizeV = 0;
    int32_t numBlocks = 0;
    int32_t blockSize = 0;
    int32_t kvHeads = 0;
    int32_t batch = 0;
    int32_t innerPrecise = 0;
    int64_t maxQSeqlen = 0;
    int64_t maxKvSeqlen = 0;
    int64_t preToken = 0;
    int64_t nextToken = 0;
    int64_t windowSizeLeft = 0;
    int64_t windowSizeRight = 0;
    int32_t sparseMode = 0;
    uint32_t maxNumBlocksPerBatch = 0;
    const int32_t* qSeqlenList{nullptr};
    const int32_t* kvSeqlenList{nullptr};
    float scaleValue = 0.0;
    float softcapValue = 0.0;
    size_t* workspaces{nullptr};
    MaskType maskType = MaskType::MASK_SPEC;
    DataType dataType = DataType::FP16;
    bool pagedCacheFlag = true;
    bool lseFlag = false;
    bool isTilingSink = false;
    bool learnableSinkFlag = false;
    bool flashDecodeFlag = false;
    uint32_t numSplits = 1;
    bool kvcacheNzFlag = false;
    FwdLayout layout = FwdLayout::BSND;
    bool pagedShapeFlag = true;
};

struct BatchParams {
    uint32_t qSeqlen;
    uint32_t kvSeqlen;
    uint32_t curQNBlockTile;
    uint32_t qNBlockNumPerGroup;
    uint32_t curQNBlockNum;
    uint32_t curQSBlockTile;
    uint32_t curQSBlockNum;
    uint32_t curKSBlockTile;
    uint32_t curKSBlockNum;
};

class FAInferTiling {
  public:
    FAInferTiling() = default;
    explicit FAInferTiling(const FAInferContext& faInfo);
    void DoTiling(FAInferTilingData& tilingdata);
    void SetCoreNum(uint32_t blockNum)
    {
        this->blockNum_ = blockNum;
    }
    uint32_t GetCoreNum()
    {
        return this->blockNum_;
    }

  private:
    void FillSplitCoreTilingData(FAInferTilingData& tilingdata);
    void FillWorkSpaceTilingData(FAInferTilingData& faTilingData);
    void FillFlashDecodeTilingData(FAInferTilingData& faTilingData);
    uint32_t GetQSBlockTile(int64_t kvSeqlen);
    uint32_t GetKSBlockTile(int64_t kvSeqlen);
    uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize);
    void FillBasicTilingData(FAInferTilingData& faTilingData);
    BatchParams getBatchParams(uint32_t bIdx, uint32_t groupSize);

  private:
    FAInferContext faInfo_;
    uint32_t blockNum_;
};

FAInferTiling::FAInferTiling(const FAInferContext& faInfo) : faInfo_(faInfo) {}

uint32_t FAInferTiling::GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    uint32_t qRowNumCeil = Q_TILE_CEIL;
    uint32_t qNBlockTile = (qSeqlen != 0) ? (qRowNumCeil / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
    if (faInfo_.embeddingSizeV > 128 && qSeqlen != 0) {
        constexpr uint32_t MAX_M_FOR_LARGE_D = Q_TILE_CEIL / 2U;
        uint32_t maxQNBlockTile = std::max(MAX_M_FOR_LARGE_D / qSeqlen, 1U);
        qNBlockTile = std::min(qNBlockTile, maxQNBlockTile);
    }
    qNBlockTile = std::min(qNBlockTile, groupSize);
    if (qNBlockTile > N_SPLIT_HELPER) {
        qNBlockTile = qNBlockTile / N_SPLIT_HELPER * N_SPLIT_HELPER;
    }
    qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
    return qNBlockTile;
}

uint32_t FAInferTiling::GetQSBlockTile(int64_t kvSeqlen)
{
    uint32_t qSBlockTile = faInfo_.embeddingSizeV > 128 ? 64 : Q_TILE_CEIL;
    return qSBlockTile;
}
uint32_t FAInferTiling::GetKSBlockTile(int64_t kvSeqlen)
{
    uint32_t kSBlockTile = faInfo_.dataType == DataType::FP8 ? 256U : 128U;
    return kSBlockTile;
}

void FAInferTiling::FillBasicTilingData(FAInferTilingData& faTilingData)
{
    faTilingData.set_batch(static_cast<uint32_t>(faInfo_.batch));
    faTilingData.set_numHeads(static_cast<uint32_t>(faInfo_.numHeads));
    faTilingData.set_kvHeads(static_cast<uint32_t>(faInfo_.kvHeads));

    faTilingData.set_embeddingSize(static_cast<uint32_t>(faInfo_.embeddingSize));
    faTilingData.set_embeddingSizeV(static_cast<uint32_t>(faInfo_.embeddingSizeV));

    if (faInfo_.pagedCacheFlag) {
        faTilingData.set_blockSize(static_cast<uint32_t>(faInfo_.blockSize));
    } else {
        faTilingData.set_blockSize(BASE_KV_SIZE);
    }
    faTilingData.set_maxQSeqlen(faInfo_.maxQSeqlen);
    faTilingData.set_maxKvSeqlen(faInfo_.maxKvSeqlen);

    faTilingData.set_maskType(static_cast<uint32_t>(faInfo_.maskType));

    faTilingData.set_sparseMode(faInfo_.sparseMode);
    faTilingData.set_cacheLayout(0U);
    faTilingData.set_flashDecodeFlag(faInfo_.flashDecodeFlag ? 1U : 0U);
    faTilingData.splitMode = faInfo_.numSplits == 0U ? 0U : (faInfo_.numSplits == 1U ? 1U : 2U);
    faTilingData.requestedNumSplits = faInfo_.numSplits;
    faTilingData.set_preToken(static_cast<int64_t>(faInfo_.preToken));
    faTilingData.set_nextToken(static_cast<int64_t>(faInfo_.nextToken));
    faTilingData.set_windowSizeLeft(faInfo_.windowSizeLeft);
    faTilingData.set_windowSizeRight(faInfo_.windowSizeRight);

    auto qBaseTile_ = GetQSBlockTile(faInfo_.maxKvSeqlen);
    const uint32_t quantMode = faInfo_.dataType == DataType::FP8 ? 1U : 0U;
    auto kvBaseTile_ = quantMode == 1 ? 256U : BASE_KV_SIZE;
    faTilingData.quantMode = quantMode;
    auto embeddingSizeAligned16_ = (static_cast<uint32_t>(faInfo_.embeddingSize) + 15U) / 16U * 16U;
    auto embeddingSizeVAligned16_ = (static_cast<uint32_t>(faInfo_.embeddingSizeV) + 15U) / 16U * 16U;
    auto qkL1TileM_ = Q_TILE_CEIL;
    auto qkL1TileKLeft_ = embeddingSizeAligned16_;
    auto qL1BufNum_ = 1;
    // K矩阵开启2buf，D按128分割，S2按256分割
    auto qkL1TileN_ = 256;
    auto qkL1TileKRight_ = quantMode == 1 ? 128 : 192;
    auto kL1BufNum_ = 2;
    // V矩阵开启db，D按128分割，kvBaseTile_不分割，指令同样提前于核间同步下发
    // 如果kvBaseTile_进一步增大，考虑关闭db，使得kvBaseTile_不分割
    auto pvL1TileN_ = embeddingSizeVAligned16_;
    auto pvL1TileKLeft_ = kvBaseTile_;
    auto vL1BufNum_ = 2;
    // P矩阵在950上会常驻L1，由于基块的prelaunch为2，因此最好有3 buf，以免基块间流水阻塞
    auto pvL1TileM_ = Q_TILE_CEIL;
    auto pvL1TileKRight_ = kvBaseTile_;
    auto pL1BufNum_ = 3;

    faTilingData.set_innerPrec(0);
    faTilingData.set_actSeqAval(0);
    faTilingData.set_qBaseTile(qBaseTile_);
    faTilingData.set_kvBaseTile(kvBaseTile_);
    faTilingData.set_qkL1TileM(qkL1TileM_);
    faTilingData.set_qkL1TileN(qkL1TileN_);
    faTilingData.set_qkL1TileKLeft(qkL1TileKLeft_);
    faTilingData.set_qkL1TileKRight(qkL1TileKRight_);
    faTilingData.set_pvL1TileM(pvL1TileM_);
    faTilingData.set_pvL1TileN(pvL1TileN_);
    faTilingData.set_pvL1TileKLeft(pvL1TileKLeft_);
    faTilingData.set_pvL1TileKRight(pvL1TileKRight_);
    faTilingData.set_qL1BufNum(qL1BufNum_);
    faTilingData.set_kL1BufNum(kL1BufNum_);
    faTilingData.set_vL1BufNum(vL1BufNum_);
    faTilingData.set_pL1BufNum(pL1BufNum_);
}

void FAInferTiling::FillWorkSpaceTilingData(FAInferTilingData& faTilingData)
{
    uint64_t qkOutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB * SIZE_OF_32BIT * PRELANCH_NUM;
    uint64_t smOnlineOutSize =
        static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB * SIZE_OF_16BIT * PRELANCH_NUM;
    uint64_t pvOutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB * SIZE_OF_32BIT * PRELANCH_NUM;
    uint64_t UpdateSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB * SIZE_OF_32BIT * PRELANCH_NUM;

    uint64_t splitLseTotalSize = 0;
    uint64_t splitOTotalSize = 0;
    if (faInfo_.isTilingSink) {
        splitLseTotalSize = 2 * static_cast<uint64_t>(blockNum_) * Q_TILE_CEIL * SIZE_OF_32BIT * faInfo_.numHeads;
        uint32_t embeddingSizeV = static_cast<uint32_t>(faInfo_.embeddingSizeV);
        splitOTotalSize =
            2 * static_cast<uint64_t>(blockNum_) * Q_TILE_CEIL * embeddingSizeV * SIZE_OF_32BIT * faInfo_.numHeads;
        faTilingData.set_splitLseTotalSize(splitLseTotalSize);
        faTilingData.set_splitOTotalSize(splitOTotalSize);
        faTilingData.set_needCoreNum(blockNum_);
    } else {
        splitLseTotalSize = faTilingData.get_splitLseTotalSize();
        splitOTotalSize = faTilingData.get_splitOTotalSize();
    }
    uint64_t workSpaceSize = qkOutSize + smOnlineOutSize + pvOutSize + UpdateSize + splitLseTotalSize + splitOTotalSize;
    faTilingData.set_qkOutSize(qkOutSize);
    faTilingData.set_smOnlineOutSize(smOnlineOutSize);
    faTilingData.set_pvOutSize(pvOutSize);
    faTilingData.set_UpdateSize(UpdateSize);
    faTilingData.set_workSpaceSize(workSpaceSize);
}

namespace {
struct FdBaseTaskHost {
    uint32_t kvTiles;
    uint32_t rowNum;
    uint32_t batchIdx;
    uint32_t qStart;
    uint32_t qRows;
    uint32_t qHeadStart;
};

struct FdCandidateHost {
    uint32_t baseTask;
    uint32_t kvBegin;
    uint32_t kvEnd;
    uint64_t cost;
};

inline uint64_t AlignUpU64(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

void FillFdCombineCoreSchedules(FAInferTilingData& tiling)
{
    struct SourcePartition {
        uint32_t task;
        uint32_t subBlock;
        uint32_t rows;
    };
    std::vector<SourcePartition> partitions;
    uint64_t totalLoad = 0U;
    for (uint32_t task = 0U; task < tiling.fdCombineTaskNum; ++task) {
        const uint32_t rows = tiling.fdCombineTaskInfo[task].rowNum;
        const uint32_t sourceSplit = std::min(
            rows, static_cast<uint32_t>(AlignUpU64(rows, tiling.quantMode == 1 ? 32U : 8U)) / FD_SOURCE_SUB_BLOCK_NUM);
        const uint32_t sourceRows[FD_SOURCE_SUB_BLOCK_NUM] = {sourceSplit, rows - sourceSplit};
        for (uint32_t sub = 0U; sub < FD_SOURCE_SUB_BLOCK_NUM; ++sub) {
            if (sourceRows[sub] != 0U) {
                partitions.push_back({task, sub, sourceRows[sub]});
            }
        }
        totalLoad += static_cast<uint64_t>(rows) * tiling.fdCombineSchedules[task].partialCount;
    }

    const uint32_t aivNum = tiling.fdActiveCoreNum * FD_SOURCE_SUB_BLOCK_NUM;
    // FD has fewer base tasks than active AICs, hence every nonempty
    // source partition can reserve at least one AIV.
    if (aivNum == 0U || partitions.size() > aivNum) {
        throw std::runtime_error("FD combine source partitions exceed available AIVs");
    }
    const uint64_t averageLoad = std::max<uint64_t>(1U, (totalLoad + aivNum - 1U) / aivNum);
    uint32_t extraAivs = aivNum - static_cast<uint32_t>(partitions.size());
    uint32_t nextAiv = 0U;
    for (const auto& partition : partitions) {
        const uint64_t load =
            static_cast<uint64_t>(partition.rows) * tiling.fdCombineSchedules[partition.task].partialCount;
        // Reserve one AIV for every remaining partition before deriving
        // the row extent. Recompute the actual count after ceil rounding.
        const uint32_t initialAivs = static_cast<uint32_t>(
            std::min<uint64_t>(std::max<uint64_t>(1U, load / averageLoad), std::min(partition.rows, extraAivs + 1U)));
        const uint32_t rowsPerAiv = (partition.rows + initialAivs - 1U) / initialAivs;
        const uint32_t usedAivs = (partition.rows + rowsPerAiv - 1U) / rowsPerAiv;
        for (uint32_t i = 0U; i < usedAivs; ++i) {
            const uint32_t row = i * rowsPerAiv;
            tiling.fdCombineCoreSchedules[nextAiv++] = {partition.task, partition.subBlock, row,
                                                        std::min(rowsPerAiv, partition.rows - row)};
        }
        extraAivs -= usedAivs - 1U;
    }
    tiling.fdCombineAivNum = nextAiv;

    // One combine task has no inter-task imbalance to remove. For the
    // no-LSE path, also retain uniform dispatch when its busiest AIV's
    // aggregate partial O fits the existing input buffer: extra metadata
    // and per-AIV rounds can cost more than the small reduction in work.
    // Derive this from actual partials/rows, never the requested splits.
    tiling.fdCombineUseUniform = 0U;
    if (tiling.fdCombineTaskNum == 1U) {
        tiling.fdCombineUseUniform = FD_COMBINE_UNIFORM_NO_LSE | FD_COMBINE_UNIFORM_WITH_LSE;
    } else {
        const uint32_t rowTile =
            std::min(FD_COMBINE_ROW_TILE, std::max(1U, (tiling.fdCombineRowNum + aivNum - 1U) / aivNum));
        std::vector<uint64_t> uniformLoads(aivNum, 0U);
        std::vector<uint32_t> uniformRounds(aivNum, 0U);
        uint32_t tile = 0U;
        for (const auto& partition : partitions) {
            for (uint32_t row = 0U; row < partition.rows; row += rowTile) {
                const uint32_t core = tile++ % aivNum;
                ++uniformRounds[core];
                uniformLoads[core] += static_cast<uint64_t>(std::min(rowTile, partition.rows - row)) *
                                      tiling.fdCombineSchedules[partition.task].partialCount;
            }
        }
        const uint64_t maxInputElems =
            *std::max_element(uniformLoads.begin(), uniformLoads.end()) * AlignUpU64(tiling.embeddingSizeV, 8U);
        if (maxInputElems <= FD_COMBINE_O_INPUT_ELEMS) {
            tiling.fdCombineUseUniform = FD_COMBINE_UNIFORM_NO_LSE;
            uint32_t balancedRounds = 0U;
            for (uint32_t core = 0U; core < tiling.fdCombineAivNum; ++core) {
                balancedRounds =
                    std::max(balancedRounds, (tiling.fdCombineCoreSchedules[core].rowCount + FD_COMBINE_ROW_TILE - 1U) /
                                                 FD_COMBINE_ROW_TILE);
            }
            // With no multi-buffer input to remove, adding a whole extra
            // row round can also outweigh balancing the LSE stores.
            if (balancedRounds > *std::max_element(uniformRounds.begin(), uniformRounds.end())) {
                tiling.fdCombineUseUniform |= FD_COMBINE_UNIFORM_WITH_LSE;
            }
        }
    }
}
} // namespace

void FAInferTiling::FillFlashDecodeTilingData(FAInferTilingData& tiling)
{
    tiling.fdCombineAivNum = 0U;
    tiling.fdCombineUseUniform = 0U;
    for (auto& schedule : tiling.fdCombineCoreSchedules) {
        schedule = {};
    }
    for (uint32_t i = 0; i < MAX_FD_ACTIVE_CORE_NUM; ++i) {
        tiling.fdDecodeSchedules[i] = {-1, -1, -1, -1};
    }
    for (uint32_t i = 0; i < MAX_FD_COMBINE_TASK_NUM; ++i) {
        tiling.fdCombineSchedules[i] = {-1, -1, -1, 0};
        tiling.fdCombineTaskInfo[i] = {};
    }

    const uint32_t cfdMax = std::min(blockNum_, MAX_FD_ACTIVE_CORE_NUM);
    const uint32_t bfdMax = std::min(MAX_FD_COMBINE_TASK_NUM, blockNum_ == 0U ? 0U : (3U * blockNum_ - 1U) / 10U);
    tiling.fdPartialCapacity = (bfdMax == 0U || cfdMax == 0U) ? 0U : bfdMax + cfdMax - 1U;

    std::vector<FdBaseTaskHost> bases;
    const uint32_t groupSize = faInfo_.numHeads / faInfo_.kvHeads;
    for (int32_t batchIdx = 0; batchIdx < faInfo_.batch; ++batchIdx) {
        uint32_t qLen = static_cast<uint32_t>(faInfo_.qSeqlenList[batchIdx]);
        uint32_t kvLen = static_cast<uint32_t>(faInfo_.kvSeqlenList[batchIdx]);
        if (faInfo_.layout == FwdLayout::TND) {
            qLen = static_cast<uint32_t>(faInfo_.qSeqlenList[batchIdx + 1] - faInfo_.qSeqlenList[batchIdx]);
            if (!faInfo_.pagedCacheFlag) {
                kvLen = static_cast<uint32_t>(faInfo_.kvSeqlenList[batchIdx + 1] - faInfo_.kvSeqlenList[batchIdx]);
            }
        }
        const uint32_t qTile = GetQSBlockTile(kvLen);
        const uint32_t qTileNum = (qLen + qTile - 1U) / qTile;
        const uint32_t kvTileNum = (kvLen + tiling.kvBaseTile - 1U) / tiling.kvBaseTile;
        const uint32_t qNBlockTile = GetQNBlockTile(qLen, groupSize);
        const uint32_t qNBlockNumPerGroup = (groupSize + qNBlockTile - 1U) / qNBlockTile;
        const uint32_t qNTaskNum = qNBlockNumPerGroup * faInfo_.kvHeads;
        for (uint32_t qTileIdx = 0; qTileIdx < qTileNum; ++qTileIdx) {
            const uint32_t qRows = std::min(qTile, qLen - qTileIdx * qTile);
            for (uint32_t qNTask = 0; qNTask < qNTaskNum; ++qNTask) {
                const uint32_t qNBlockIdxInGroup = qNTask % qNBlockNumPerGroup;
                const uint32_t qNBlockSize = std::min(qNBlockTile, groupSize - qNBlockIdxInGroup * qNBlockTile);
                const uint32_t kvHeadIdx = qNTask / qNBlockNumPerGroup;
                bases.push_back({kvTileNum, qRows * qNBlockSize, static_cast<uint32_t>(batchIdx), qTileIdx * qTile,
                                 qRows, kvHeadIdx * groupSize + qNBlockIdxInGroup * qNBlockTile});
            }
        }
    }

    tiling.fdBaseTaskNum = static_cast<uint32_t>(bases.size());
    if (bases.empty() || bases.size() > bfdMax || cfdMax < 2U) {
        tiling.flashDecodeFlag = 0U;
        return;
    }

    std::vector<uint32_t> splitCounts(bases.size(), 1U);
    if (faInfo_.numSplits > 1U) {
        for (uint32_t base = 0; base < bases.size(); ++base) {
            splitCounts[base] = std::min(faInfo_.numSplits, bases[base].kvTiles);
        }
    } else {
        uint32_t totalSegments = static_cast<uint32_t>(bases.size());
        while (totalSegments < cfdMax) {
            uint32_t bestBase = static_cast<uint32_t>(bases.size());
            uint64_t bestCost = 0U;
            for (uint32_t base = 0; base < bases.size(); ++base) {
                if (splitCounts[base] >= bases[base].kvTiles) {
                    continue;
                }
                const uint64_t cost =
                    static_cast<uint64_t>(bases[base].rowNum) * bases[base].kvTiles / splitCounts[base];
                if (bestBase == bases.size() || cost > bestCost) {
                    bestBase = base;
                    bestCost = cost;
                }
            }
            if (bestBase == bases.size()) {
                break;
            }
            ++splitCounts[bestBase];
            ++totalSegments;
        }
    }

    std::vector<FdCandidateHost> candidates;
    for (uint32_t base = 0; base < bases.size(); ++base) {
        const uint32_t count = splitCounts[base];
        for (uint32_t split = 0; split < count; ++split) {
            const uint32_t begin = static_cast<uint32_t>(static_cast<uint64_t>(bases[base].kvTiles) * split / count);
            const uint32_t end =
                static_cast<uint32_t>(static_cast<uint64_t>(bases[base].kvTiles) * (split + 1U) / count);
            candidates.push_back({base, begin, end, static_cast<uint64_t>(bases[base].rowNum) * (end - begin)});
        }
    }

    const uint32_t activeCores = std::min(cfdMax, static_cast<uint32_t>(candidates.size()));
    if (activeCores <= bases.size()) {
        tiling.flashDecodeFlag = 0U;
        return;
    }

    uint64_t totalCost = 0U;
    for (const auto& candidate : candidates) {
        totalCost += candidate.cost;
    }
    std::vector<uint32_t> coreBegin(activeCores);
    std::vector<uint32_t> coreEnd(activeCores);
    uint32_t candidateBegin = 0U;
    uint64_t consumedCost = 0U;
    for (uint32_t core = 0; core < activeCores; ++core) {
        coreBegin[core] = candidateBegin;
        const uint32_t coresLeft = activeCores - core;
        const uint32_t maxEnd = static_cast<uint32_t>(candidates.size()) - (coresLeft - 1U);
        uint32_t end = candidateBegin + 1U;
        const uint64_t target = totalCost * (core + 1U) / activeCores;
        while (end < maxEnd && consumedCost + candidates[end - 1U].cost < target) {
            consumedCost += candidates[end - 1U].cost;
            ++end;
        }
        consumedCost += candidates[end - 1U].cost;
        coreEnd[core] = end;
        candidateBegin = end;

        const auto& first = candidates[coreBegin[core]];
        const auto& last = candidates[end - 1U];
        tiling.fdDecodeSchedules[core] = {static_cast<int32_t>(first.baseTask),
                                          static_cast<int32_t>(last.baseTask + 1U), static_cast<int32_t>(first.kvBegin),
                                          static_cast<int32_t>(last.kvEnd)};
    }

    uint32_t partialStart = 0U;
    uint32_t combineCount = 0U;
    uint32_t combineRows = 0U;
    for (uint32_t base = 0; base < bases.size(); ++base) {
        int32_t firstCore = -1;
        uint32_t partialCount = 0U;
        for (uint32_t core = 0; core < activeCores; ++core) {
            const auto& schedule = tiling.fdDecodeSchedules[core];
            if (schedule.baseTaskStart <= static_cast<int32_t>(base) &&
                static_cast<int32_t>(base) < schedule.baseTaskEnd) {
                if (firstCore < 0) {
                    firstCore = static_cast<int32_t>(core);
                }
                ++partialCount;
            }
        }
        if (partialCount > 1U) {
            tiling.fdCombineSchedules[combineCount] = {static_cast<int32_t>(base), firstCore,
                                                       static_cast<int32_t>(partialStart),
                                                       static_cast<int32_t>(partialCount)};
            const auto& info = bases[base];
            tiling.fdCombineTaskInfo[combineCount++] = {info.batchIdx, info.qStart, info.qRows, info.qHeadStart,
                                                        info.rowNum};
            partialStart += partialCount;
            combineRows += bases[base].rowNum;
        }
    }

    if (combineCount == 0U || partialStart > tiling.fdPartialCapacity) {
        tiling.flashDecodeFlag = 0U;
        return;
    }

    tiling.fdActiveCoreNum = activeCores;
    tiling.fdCombineTaskNum = combineCount;
    tiling.fdPartialTaskNum = partialStart;
    tiling.fdCombineRowNum = combineRows;
    FillFdCombineCoreSchedules(tiling);
    uint32_t maxFdRows = 0U;
    for (const auto& base : bases) {
        maxFdRows = std::max(maxFdRows, base.rowNum);
    }
    tiling.fdRowCapacity = static_cast<uint32_t>(AlignUpU64(maxFdRows, 8U));
    tiling.fdLseSubStride = tiling.quantMode == 1
                                ? static_cast<uint32_t>(AlignUpU64(tiling.fdRowCapacity, 32U) / 2U)
                                : static_cast<uint32_t>(AlignUpU64((tiling.fdRowCapacity + 1U) / 2U, 8U));

    // The Ascend950 FD pipeline keeps QK, softmax, PV and rescale buffers
    // on chip. Its GM workspace only needs the partial LSE and O regions.
    tiling.set_qkOutSize(0U);
    tiling.set_smOnlineOutSize(0U);
    tiling.set_pvOutSize(0U);
    tiling.set_UpdateSize(0U);
    tiling.fdPartialLseOffset = 0U;
    const uint64_t partialLseSize =
        static_cast<uint64_t>(tiling.fdPartialCapacity) * 2U * tiling.fdLseSubStride * sizeof(float);
    tiling.fdPartialOOffset = AlignUpU64(tiling.fdPartialLseOffset + partialLseSize, FD_WORKSPACE_ALIGNMENT);
    const uint64_t partialOSize =
        static_cast<uint64_t>(tiling.fdPartialCapacity) * tiling.fdRowCapacity * faInfo_.embeddingSizeV * sizeof(float);
    tiling.fdWorkspaceEnd = tiling.fdPartialOOffset + partialOSize;
    tiling.workSpaceSize = tiling.fdWorkspaceEnd;
}

void FAInferTiling::FillSplitCoreTilingData(FAInferTilingData& faTilingData)
{
    uint32_t totalTaskNum = 0;
    uint32_t groupSize = faInfo_.numHeads / faInfo_.kvHeads;
    for (int32_t batchIdx = 0; batchIdx < faInfo_.batch; batchIdx++) {
        uint32_t qSeqlen = *(faInfo_.qSeqlenList + batchIdx);
        uint32_t kvSeqlen = *(faInfo_.kvSeqlenList + batchIdx);
        if (faInfo_.layout == FwdLayout::TND) {
            uint64_t prevQSeqlenSum = *(faInfo_.qSeqlenList + batchIdx);
            qSeqlen = *(faInfo_.qSeqlenList + batchIdx + 1) - prevQSeqlenSum;
            if (!faInfo_.pagedCacheFlag) {
                uint64_t prevKvSeqlenSum = *(faInfo_.kvSeqlenList + batchIdx);
                kvSeqlen = *(faInfo_.kvSeqlenList + batchIdx + 1) - prevKvSeqlenSum;
            }
        }
        uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
        uint32_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
        uint32_t curQNBlockNum = qNBlockNumPerGroup * faInfo_.kvHeads;
        uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
        uint32_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
        uint32_t curTaskNum = curQNBlockNum * curQSBlockNum;
        if (batchIdx == 0) {
            faTilingData.set_firstBatchTaskNum(curTaskNum);
        }
        totalTaskNum += curTaskNum;
    }
    faTilingData.set_totalTaskNum(totalTaskNum);
}
BatchParams FAInferTiling::getBatchParams(uint32_t bIdx, uint32_t groupSize)
{
    BatchParams p;
    p.qSeqlen = *(faInfo_.qSeqlenList + bIdx);
    p.kvSeqlen = *(faInfo_.kvSeqlenList + bIdx);
    if (bIdx > 0) {
        uint64_t prevQSeqlenSum = *(faInfo_.qSeqlenList + bIdx - 1);
        p.qSeqlen = p.qSeqlen - prevQSeqlenSum;
    }
    p.curQNBlockTile = GetQNBlockTile(p.qSeqlen, groupSize);
    p.qNBlockNumPerGroup = (groupSize + p.curQNBlockTile - 1) / p.curQNBlockTile;
    p.curQNBlockNum = p.qNBlockNumPerGroup * faInfo_.kvHeads;
    p.curQSBlockTile = GetQSBlockTile(p.kvSeqlen);
    p.curQSBlockNum = (p.qSeqlen + p.curQSBlockTile - 1) / p.curQSBlockTile;
    p.curKSBlockTile = GetKSBlockTile(p.kvSeqlen);
    p.curKSBlockNum = (p.kvSeqlen + p.curKSBlockTile - 1) / p.curKSBlockTile;
    return p;
}

void FAInferTiling::DoTiling(FAInferTilingData& tilingdata)
{
    FillBasicTilingData(tilingdata);
    if (!faInfo_.isTilingSink) {
        FillSplitCoreTilingData(tilingdata);
    }
    FillWorkSpaceTilingData(tilingdata);
    if (faInfo_.flashDecodeFlag) {
        FillFlashDecodeTilingData(tilingdata);
    }
}
} // namespace optiling
