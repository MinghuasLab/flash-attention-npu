/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026
 *
 * FA v2 forward split (flash-decode) scheduling helpers, shared between the
 * host path (flash_api.cpp) and the AICPU scheduler-metadata kernel
 * (fa_metadata.aicpu). Everything is inline and free of <algorithm> so the
 * same code compiles for the AICPU toolchain; FaMin/FaMax replace
 * std::min/std::max.
 */

#ifndef FA_SPLIT_H
#define FA_SPLIT_H

#include <cstdint>
#include "tilingdata.h"

namespace fa_split {

constexpr uint32_t Q_TILE_CEIL = 128;
constexpr uint32_t N_SPLIT_HELPER = 2;
constexpr uint32_t MAX_KV_STACK_LEN = 512;

inline uint32_t FaMin(uint32_t a, uint32_t b) { return a < b ? a : b; }
inline uint32_t FaMax(uint32_t a, uint32_t b) { return a > b ? a : b; }

inline uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    uint32_t qRowNumCeil = Q_TILE_CEIL;
    uint32_t qNBlockTile = (qSeqlen != 0) ?
        (qRowNumCeil / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
    qNBlockTile = FaMin(qNBlockTile, groupSize);
    qNBlockTile = FaMax(qNBlockTile, static_cast<uint32_t>(1));
    return qNBlockTile;
}

inline uint32_t GetQSBlockTile(int64_t kvSeqlen)
{
    uint32_t qSBlockTile = Q_TILE_CEIL;
    return qSBlockTile;
}

inline uint32_t GetKSBlockTile(uint32_t kvSeqlen)
{
    uint32_t kSBlockTile = MAX_KV_STACK_LEN;
    return kSBlockTile;
}

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
    uint32_t curKSBlockTail;  // rows in the last S2 block (new segment tail, else old tail)
};

struct SplitContext {
    int32_t batch_size;
    int32_t num_heads;
    int32_t num_heads_k;
    int32_t seqlen_q;
    int32_t head_size_v;
    int32_t* cu_seqlen_q_cpu;
    int32_t* seqlens_k_cpu;
    bool is_varlen_q;
    uint32_t blockDim;
    int32_t kvNewSeqlen;
};

inline BatchParams getBatchParams(uint32_t bIdx, uint32_t groupSize, const SplitContext& ctx)
{
    BatchParams p;
    if (ctx.is_varlen_q) {
        p.qSeqlen = static_cast<uint32_t>(ctx.cu_seqlen_q_cpu[bIdx + 1] - ctx.cu_seqlen_q_cpu[bIdx]);
    } else {
        p.qSeqlen = static_cast<uint32_t>(ctx.seqlen_q);
    }
    p.kvSeqlen = static_cast<uint32_t>(ctx.seqlens_k_cpu[bIdx] + ctx.kvNewSeqlen);
    p.curQNBlockTile = GetQNBlockTile(p.qSeqlen, groupSize);
    p.qNBlockNumPerGroup = (groupSize + p.curQNBlockTile - 1) / p.curQNBlockTile;
    p.curQNBlockNum = p.qNBlockNumPerGroup * ctx.num_heads_k;
    p.curQSBlockTile = GetQSBlockTile(p.kvSeqlen);
    p.curQSBlockNum = (p.qSeqlen + p.curQSBlockTile - 1) / p.curQSBlockTile;
    p.curKSBlockTile = GetKSBlockTile(p.kvSeqlen);
    // Per-phase S2 block count: old blocks + new blocks
    uint32_t kvSeqlenOld = static_cast<uint32_t>(p.kvSeqlen - ctx.kvNewSeqlen);
    uint32_t kvSBlockNumOld = (kvSeqlenOld + p.curKSBlockTile - 1) / p.curKSBlockTile;
    uint32_t kvSBlockNumNew = ctx.kvNewSeqlen != 0
        ? (static_cast<uint32_t>(ctx.kvNewSeqlen) + p.curKSBlockTile - 1) / p.curKSBlockTile : 0U;
    p.curKSBlockNum = kvSBlockNumOld + kvSBlockNumNew;
    p.curKSBlockTail = kvSBlockNumNew != 0U
        ? (static_cast<uint32_t>(ctx.kvNewSeqlen) - (kvSBlockNumNew - 1U) * p.curKSBlockTile)
        : (kvSBlockNumOld != 0U ? (kvSeqlenOld - (kvSBlockNumOld - 1U) * p.curKSBlockTile) : 0U);
    return p;
}

inline void fillCoreInfoForFlashDecode(FAInferTilingData* tiling, uint32_t groupSize,
                                       uint64_t perCoreTaskNum, const SplitContext& ctx)
{
    int32_t nowBIdx = 0;
    int32_t nowN1Idx = 0;
    int32_t nowS1Idx = 0;
    int32_t nowS2Idx = 0;

    for (uint32_t coreIdx = 0; coreIdx < ctx.blockDim; coreIdx++) {
        tiling->coreInfo[coreIdx].startBIdx = 0;
        tiling->coreInfo[coreIdx].startN1Idx = 0;
        tiling->coreInfo[coreIdx].startS1Idx = 0;
        tiling->coreInfo[coreIdx].startS2Idx = 0;
        tiling->coreInfo[coreIdx].endBIdx = 0;
        tiling->coreInfo[coreIdx].endN1Idx = 0;
        tiling->coreInfo[coreIdx].endS1Idx = 0;
        tiling->coreInfo[coreIdx].endS2Idx = 0;
    }

    auto finishBatch = [&](uint32_t coreIdx) {
        BatchParams p = getBatchParams(ctx.batch_size - 1, groupSize, ctx);
        tiling->coreInfo[coreIdx].endBIdx = ctx.batch_size - 1;
        tiling->coreInfo[coreIdx].endN1Idx = p.curQNBlockNum - 1;
        tiling->coreInfo[coreIdx].endS1Idx = p.curQSBlockNum - 1;
        tiling->coreInfo[coreIdx].endS2Idx = p.curKSBlockNum;
        tiling->set_needCoreNum(coreIdx + 1);
    };

    for (uint32_t coreIdx = 0; coreIdx < ctx.blockDim; coreIdx++) {
        int32_t resTaskNum = static_cast<int32_t>(perCoreTaskNum);
        tiling->coreInfo[coreIdx].startBIdx = nowBIdx;
        tiling->coreInfo[coreIdx].startN1Idx = nowN1Idx;
        tiling->coreInfo[coreIdx].startS1Idx = nowS1Idx;
        tiling->coreInfo[coreIdx].startS2Idx = nowS2Idx;

        BatchParams p = getBatchParams(nowBIdx, groupSize, ctx);

        auto advanceCounters = [&]() {
            if (nowS2Idx == static_cast<int32_t>(p.curKSBlockNum)) { nowS1Idx++; nowS2Idx = 0; }
            if (nowS1Idx == static_cast<int32_t>(p.curQSBlockNum)) { nowN1Idx++; nowS1Idx = 0; nowS2Idx = 0; }
            if (nowN1Idx == static_cast<int32_t>(p.curQNBlockNum)) { nowBIdx++; nowN1Idx = 0; nowS1Idx = 0; nowS2Idx = 0; }
        };

        while (nowS2Idx < static_cast<int32_t>(p.curKSBlockNum) && resTaskNum > 0) {
            p = getBatchParams(nowBIdx, groupSize, ctx);
            uint32_t remainingQ = (nowS1Idx < static_cast<int32_t>(p.curQSBlockNum) - 1)
                ? p.curQSBlockTile
                : (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
            uint32_t remainingKV = (nowS2Idx < static_cast<int32_t>(p.curKSBlockNum) - 1)
                ? p.curKSBlockTile
                : p.curKSBlockTail;
            uint64_t singleS2Task = static_cast<uint64_t>(remainingQ) * remainingKV;
            resTaskNum -= static_cast<int32_t>(singleS2Task);
            nowS2Idx += 1;
        }

        if (resTaskNum <= 0) {
            tiling->coreInfo[coreIdx].endBIdx = nowBIdx;
            tiling->coreInfo[coreIdx].endN1Idx = nowN1Idx;
            tiling->coreInfo[coreIdx].endS1Idx = nowS1Idx;
            tiling->coreInfo[coreIdx].endS2Idx = nowS2Idx;
        }

        advanceCounters();
        if (nowBIdx < ctx.batch_size && resTaskNum <= 0) continue;
        if (nowBIdx == ctx.batch_size) { finishBatch(coreIdx); break; }

        while (nowBIdx < ctx.batch_size && resTaskNum > 0) {
            p = getBatchParams(nowBIdx, groupSize, ctx);
            uint32_t remainingQ = p.qSeqlen * (ctx.num_heads - p.curQNBlockTile * nowN1Idx) - nowS1Idx * p.curQSBlockTile;
            uint32_t remainingKV = p.kvSeqlen;
            uint32_t remainingInBatch = remainingQ * remainingKV;

            if (resTaskNum >= static_cast<int32_t>(remainingInBatch)) {
                resTaskNum -= remainingInBatch;
                nowBIdx++; nowN1Idx = 0; nowS1Idx = 0; nowS2Idx = 0;
            } else {
                break;
            }
        }

        if (nowBIdx == ctx.batch_size) { finishBatch(coreIdx); break; }
        p = getBatchParams(nowBIdx, groupSize, ctx);

        while (nowN1Idx < static_cast<int32_t>(p.curQNBlockNum) && resTaskNum > 0) {
            uint32_t remainingQ = p.qSeqlen * p.curQNBlockTile - nowS1Idx * p.curQSBlockTile;
            uint32_t remainingInN1 = remainingQ * p.kvSeqlen;
            if (resTaskNum >= static_cast<int32_t>(remainingInN1)) {
                resTaskNum -= static_cast<int32_t>(remainingInN1);
                nowN1Idx++; nowS1Idx = 0; nowS2Idx = 0;
            } else {
                break;
            }
        }

        advanceCounters();
        if (nowBIdx == ctx.batch_size) { finishBatch(coreIdx); break; }
        p = getBatchParams(nowBIdx, groupSize, ctx);

        while (nowS1Idx < static_cast<int32_t>(p.curQSBlockNum) && resTaskNum > 0) {
            uint32_t remainingQ = (nowS1Idx < static_cast<int32_t>(p.curQSBlockNum) - 1)
                ? p.curQSBlockTile
                : (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
            uint64_t remainingInS1 = static_cast<uint64_t>(remainingQ) * p.kvSeqlen;
            if (resTaskNum >= static_cast<int64_t>(remainingInS1)) {
                resTaskNum -= static_cast<int32_t>(remainingInS1);
                nowS1Idx++; nowS2Idx = 0;
            } else {
                break;
            }
        }

        advanceCounters();
        if (nowBIdx == ctx.batch_size) { finishBatch(coreIdx); break; }
        p = getBatchParams(nowBIdx, groupSize, ctx);

        while (nowS2Idx < static_cast<int32_t>(p.curKSBlockNum) && resTaskNum > 0) {
            uint32_t remainingQ = (nowS1Idx < static_cast<int32_t>(p.curQSBlockNum) - 1)
                ? p.curQSBlockTile
                : (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
            uint32_t remainingKV = (nowS2Idx < static_cast<int32_t>(p.curKSBlockNum) - 1)
                ? p.curKSBlockTile
                : p.curKSBlockTail;
            uint64_t singleS2Task = static_cast<uint64_t>(remainingQ) * remainingKV;
            resTaskNum -= static_cast<int32_t>(singleS2Task);
            nowS2Idx += 1;
        }

        if (nowBIdx == ctx.batch_size) { finishBatch(coreIdx); break; }

        tiling->coreInfo[coreIdx].endBIdx = nowBIdx;
        tiling->coreInfo[coreIdx].endN1Idx = nowN1Idx;
        tiling->coreInfo[coreIdx].endS1Idx = nowS1Idx;
        tiling->coreInfo[coreIdx].endS2Idx = nowS2Idx;

        advanceCounters();
    }
}

inline void fillSplitInfoForFlashDecode(FAInferTilingData* tiling, uint32_t groupSize,
                                        const SplitContext& ctx)
{
    constexpr uint32_t SIZE_OF_32BIT = 4;

    for (uint32_t splitIdx = 0; splitIdx < ctx.blockDim + 1; splitIdx++) {
        tiling->splitInfo[splitIdx].batchIdx = 0;
        tiling->splitInfo[splitIdx].headStartIdx = 0;
        tiling->splitInfo[splitIdx].headEndIdx = 0;
        tiling->splitInfo[splitIdx].qStartIdx = 0;
        tiling->splitInfo[splitIdx].qEndIdx = 0;
        tiling->splitInfo[splitIdx].splitNum = 0;
        tiling->splitInfo[splitIdx].lseTaskOffset = 0;
        tiling->splitInfo[splitIdx].oTaskOffset = 0;
    }

    int64_t currentLseTaskOffset = 0;
    int64_t currentOTaskOffset = 0;
    int32_t splitIdx = -1;
    int32_t prevBIdx = -1;
    int32_t prevN1Idx = -1;
    int32_t prevS1Idx = -1;

    for (uint32_t coreIdx = 0; coreIdx < ctx.blockDim; coreIdx++) {
        int32_t startBIdx = tiling->coreInfo[coreIdx].startBIdx;
        int32_t startN1Idx = tiling->coreInfo[coreIdx].startN1Idx;
        int32_t startS1Idx = tiling->coreInfo[coreIdx].startS1Idx;
        int32_t startS2Idx = tiling->coreInfo[coreIdx].startS2Idx;
        int32_t endBIdx = tiling->coreInfo[coreIdx].endBIdx;
        int32_t endN1Idx = tiling->coreInfo[coreIdx].endN1Idx;
        int32_t endS1Idx = tiling->coreInfo[coreIdx].endS1Idx;
        int32_t endS2Idx = tiling->coreInfo[coreIdx].endS2Idx;

        tiling->coreInfo[coreIdx].firstSplitKVTaskLseOffset = 0;
        tiling->coreInfo[coreIdx].firstSplitKVTaskOOffset = 0;

        bool foundFirstSplitKV = false;
        for (int BIdx = startBIdx; BIdx <= endBIdx; BIdx++) {
            BatchParams p = getBatchParams(BIdx, groupSize, ctx);

            int curStartN1 = (BIdx == startBIdx) ? startN1Idx : 0;
            int curEndN1 = (BIdx == endBIdx) ? endN1Idx : static_cast<int>(p.curQNBlockNum) - 1;

            for (int N1Idx = curStartN1; N1Idx <= curEndN1; N1Idx++) {
                int curStartS1 = (BIdx == startBIdx && N1Idx == startN1Idx) ? startS1Idx : 0;
                int curEndS1 = (BIdx == endBIdx && N1Idx == endN1Idx) ? endS1Idx : static_cast<int>(p.curQSBlockNum) - 1;

                for (int S1Idx = curStartS1; S1Idx <= curEndS1; S1Idx++) {
                    int curStartS2 = (BIdx == startBIdx && N1Idx == startN1Idx && S1Idx == startS1Idx) ? startS2Idx : 0;
                    int curEndS2 = (BIdx == endBIdx && N1Idx == endN1Idx && S1Idx == endS1Idx) ? endS2Idx : static_cast<int>(p.curKSBlockNum);

                    int coveredS2 = curEndS2 - curStartS2;
                    bool isSplitKV = (coveredS2 > 0 && coveredS2 < static_cast<int>(p.curKSBlockNum));

                    int64_t tmpLseOffset = currentLseTaskOffset;
                    int64_t tmpOOffset = currentOTaskOffset;

                    uint32_t N1IdxPerGroup = N1Idx % p.qNBlockNumPerGroup;
                    uint32_t kvHeadIdx = N1Idx / p.qNBlockNumPerGroup;
                    uint32_t currentHeadStart = kvHeadIdx * groupSize + N1IdxPerGroup * p.curQNBlockTile;
                    uint32_t currentHeadEnd = FaMin(currentHeadStart + p.curQNBlockTile, (kvHeadIdx + 1) * groupSize);

                    uint32_t currentQStart = S1Idx * p.curQSBlockTile;
                    uint32_t currentQEnd = FaMin(currentQStart + p.curQSBlockTile, p.qSeqlen);

                    uint32_t headLen = currentHeadEnd - currentHeadStart;
                    uint32_t qLen = currentQEnd - currentQStart;

                    if (isSplitKV) {
                        if (BIdx != prevBIdx || N1Idx != prevN1Idx || S1Idx != prevS1Idx) {
                            splitIdx++;
                            if (splitIdx < static_cast<int32_t>(ctx.blockDim) + 1) {
                                tiling->splitInfo[splitIdx].batchIdx = BIdx;
                                tiling->splitInfo[splitIdx].splitNum = 0;
                                tiling->splitInfo[splitIdx].headStartIdx = currentHeadStart;
                                tiling->splitInfo[splitIdx].headEndIdx = currentHeadEnd;
                                tiling->splitInfo[splitIdx].qStartIdx = currentQStart;
                                tiling->splitInfo[splitIdx].qEndIdx = currentQEnd;
                                tiling->splitInfo[splitIdx].lseTaskOffset = currentLseTaskOffset;
                                tiling->splitInfo[splitIdx].oTaskOffset = currentOTaskOffset;
                            }
                            prevBIdx = BIdx;
                            prevN1Idx = N1Idx;
                            prevS1Idx = S1Idx;
                        }
                        if (splitIdx >= 0 && splitIdx < static_cast<int32_t>(ctx.blockDim) + 1) {
                            tiling->splitInfo[splitIdx].splitNum++;
                            currentLseTaskOffset += static_cast<int64_t>(headLen) * qLen;
                            currentOTaskOffset += static_cast<int64_t>(headLen) * qLen * ctx.head_size_v;
                        }

                        if (!foundFirstSplitKV) {
                            foundFirstSplitKV = true;
                            tiling->coreInfo[coreIdx].firstSplitKVTaskLseOffset = tmpLseOffset;
                            tiling->coreInfo[coreIdx].firstSplitKVTaskOOffset = tmpOOffset;
                        }
                    }
                }
            }
        }
    }

    uint32_t actualSplitNum = (splitIdx + 1 > static_cast<int32_t>(ctx.blockDim))
        ? ctx.blockDim : static_cast<uint32_t>(splitIdx + 1);
    tiling->set_totalSplitNodeNum(actualSplitNum);
    tiling->set_splitLseTotalSize(currentLseTaskOffset * SIZE_OF_32BIT);
    tiling->set_splitOTotalSize(currentOTaskOffset * SIZE_OF_32BIT);
}

inline void splitBN2S1GS2(FAInferTilingData* tiling, const SplitContext& ctx)
{
    uint64_t totalTaskNum = 0;
    uint32_t groupSize = ctx.num_heads / ctx.num_heads_k;

    for (int32_t batchIdx = 0; batchIdx < ctx.batch_size; batchIdx++) {
        BatchParams p = getBatchParams(batchIdx, groupSize, ctx);
        totalTaskNum += static_cast<uint64_t>(ctx.num_heads) * p.qSeqlen * p.kvSeqlen;
    }
    uint64_t perCoreTaskNum = (totalTaskNum + ctx.blockDim - 1) / ctx.blockDim;
    fillCoreInfoForFlashDecode(tiling, groupSize, perCoreTaskNum, ctx);
    fillSplitInfoForFlashDecode(tiling, groupSize, ctx);
}

}

#endif
