/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026
 */

#ifndef FA_SPLIT_H
#define FA_SPLIT_H

#include <cstdint>
#include "tilingdata.h"

namespace fa_split {

constexpr uint32_t Q_TILE_CEIL = 128;
constexpr uint32_t N_SPLIT_HELPER = 2;
constexpr uint32_t MAX_KV_STACK_LEN = 512;

inline uint32_t FaMin(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
inline uint32_t FaMax(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

inline uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    uint32_t qRowNumCeil = Q_TILE_CEIL;
    uint32_t qNBlockTile = (qSeqlen != 0) ? (qRowNumCeil / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
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
};

struct SplitContext {
    int32_t batch_size;
    int32_t num_heads;
    int32_t num_heads_k;
    int32_t seqlen_q;
    int32_t head_size_v;
    const int32_t* seqlens_q_cpu;
    const int32_t* seqlens_k_cpu;
    uint32_t blockDim;
    int32_t num_splits;
};

inline BatchParams getBatchParams(uint32_t bIdx, uint32_t groupSize, const SplitContext& ctx)
{
    BatchParams p;
    if (ctx.seqlens_q_cpu != nullptr) {
        p.qSeqlen = static_cast<uint32_t>(ctx.seqlens_q_cpu[bIdx]);
    } else {
        p.qSeqlen = static_cast<uint32_t>(ctx.seqlen_q);
    }
    p.kvSeqlen = static_cast<uint32_t>(ctx.seqlens_k_cpu[bIdx]);
    p.curQNBlockTile = GetQNBlockTile(p.qSeqlen, groupSize);
    p.qNBlockNumPerGroup = (groupSize + p.curQNBlockTile - 1) / p.curQNBlockTile;
    p.curQNBlockNum = p.qNBlockNumPerGroup * ctx.num_heads_k;
    p.curQSBlockTile = GetQSBlockTile(p.kvSeqlen);
    p.curQSBlockNum = (p.qSeqlen + p.curQSBlockTile - 1) / p.curQSBlockTile;
    p.curKSBlockTile = GetKSBlockTile(p.kvSeqlen);
    p.curKSBlockNum = (p.kvSeqlen + p.curKSBlockTile - 1) / p.curKSBlockTile;
    return p;
}

inline void fillCoreInfoForFlashDecode(FAInferStaticTilingData* tiling, uint32_t groupSize, uint64_t perCoreTaskNum,
                                       const SplitContext& ctx)
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
            if (nowS2Idx == static_cast<int32_t>(p.curKSBlockNum)) {
                nowS1Idx++;
                nowS2Idx = 0;
            }
            if (nowS1Idx == static_cast<int32_t>(p.curQSBlockNum)) {
                nowN1Idx++;
                nowS1Idx = 0;
                nowS2Idx = 0;
            }
            if (nowN1Idx == static_cast<int32_t>(p.curQNBlockNum)) {
                nowBIdx++;
                nowN1Idx = 0;
                nowS1Idx = 0;
                nowS2Idx = 0;
            }
        };

        while (nowS2Idx < static_cast<int32_t>(p.curKSBlockNum) && resTaskNum > 0) {
            p = getBatchParams(nowBIdx, groupSize, ctx);
            uint32_t remainingQ = (nowS1Idx < static_cast<int32_t>(p.curQSBlockNum) - 1)
                                      ? p.curQSBlockTile
                                      : (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
            uint32_t remainingKV = (nowS2Idx < static_cast<int32_t>(p.curKSBlockNum) - 1)
                                       ? p.curKSBlockTile
                                       : (p.kvSeqlen - nowS2Idx * p.curKSBlockTile);
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
        if (nowBIdx < ctx.batch_size && resTaskNum <= 0)
            continue;
        if (nowBIdx == ctx.batch_size) {
            finishBatch(coreIdx);
            break;
        }

        while (nowBIdx < ctx.batch_size && resTaskNum > 0) {
            p = getBatchParams(nowBIdx, groupSize, ctx);
            uint32_t remainingQ =
                p.qSeqlen * (ctx.num_heads - p.curQNBlockTile * nowN1Idx) - nowS1Idx * p.curQSBlockTile;
            uint32_t remainingKV = p.kvSeqlen;
            uint32_t remainingInBatch = remainingQ * remainingKV;

            if (resTaskNum >= static_cast<int32_t>(remainingInBatch)) {
                resTaskNum -= remainingInBatch;
                nowBIdx++;
                nowN1Idx = 0;
                nowS1Idx = 0;
                nowS2Idx = 0;
            } else {
                break;
            }
        }

        if (nowBIdx == ctx.batch_size) {
            finishBatch(coreIdx);
            break;
        }
        p = getBatchParams(nowBIdx, groupSize, ctx);

        while (nowN1Idx < static_cast<int32_t>(p.curQNBlockNum) && resTaskNum > 0) {
            uint32_t remainingQ = p.qSeqlen * p.curQNBlockTile - nowS1Idx * p.curQSBlockTile;
            uint32_t remainingInN1 = remainingQ * p.kvSeqlen;
            if (resTaskNum >= static_cast<int32_t>(remainingInN1)) {
                resTaskNum -= remainingInN1;
                nowN1Idx++;
                nowS1Idx = 0;
                nowS2Idx = 0;
            } else {
                break;
            }
        }

        advanceCounters();
        if (nowBIdx == ctx.batch_size) {
            finishBatch(coreIdx);
            break;
        }
        p = getBatchParams(nowBIdx, groupSize, ctx);

        while (nowS1Idx < static_cast<int32_t>(p.curQSBlockNum) && resTaskNum > 0) {
            uint32_t remainingQ = (nowS1Idx < static_cast<int32_t>(p.curQSBlockNum) - 1)
                                      ? p.curQSBlockTile
                                      : (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
            uint64_t remainingInS1 = static_cast<uint64_t>(remainingQ) * p.kvSeqlen;
            if (resTaskNum >= static_cast<int64_t>(remainingInS1)) {
                resTaskNum -= static_cast<int32_t>(remainingInS1);
                nowS1Idx++;
                nowS2Idx = 0;
            } else {
                break;
            }
        }

        advanceCounters();
        if (nowBIdx == ctx.batch_size) {
            finishBatch(coreIdx);
            break;
        }
        p = getBatchParams(nowBIdx, groupSize, ctx);

        while (nowS2Idx < static_cast<int32_t>(p.curKSBlockNum) && resTaskNum > 0) {
            uint32_t remainingQ = (nowS1Idx < static_cast<int32_t>(p.curQSBlockNum) - 1)
                                      ? p.curQSBlockTile
                                      : (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
            uint32_t remainingKV = (nowS2Idx < static_cast<int32_t>(p.curKSBlockNum) - 1)
                                       ? p.curKSBlockTile
                                       : (p.kvSeqlen - nowS2Idx * p.curKSBlockTile);
            uint64_t singleS2Task = static_cast<uint64_t>(remainingQ) * remainingKV;
            resTaskNum -= static_cast<int32_t>(singleS2Task);
            nowS2Idx += 1;
        }

        if (nowBIdx == ctx.batch_size) {
            finishBatch(coreIdx);
            break;
        }

        tiling->coreInfo[coreIdx].endBIdx = nowBIdx;
        tiling->coreInfo[coreIdx].endN1Idx = nowN1Idx;
        tiling->coreInfo[coreIdx].endS1Idx = nowS1Idx;
        tiling->coreInfo[coreIdx].endS2Idx = nowS2Idx;

        advanceCounters();
    }
}

inline void fillSplitInfoForFlashDecode(FAInferStaticTilingData* tiling, uint32_t groupSize, const SplitContext& ctx)
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
                int curEndS1 =
                    (BIdx == endBIdx && N1Idx == endN1Idx) ? endS1Idx : static_cast<int>(p.curQSBlockNum) - 1;

                for (int S1Idx = curStartS1; S1Idx <= curEndS1; S1Idx++) {
                    int curStartS2 = (BIdx == startBIdx && N1Idx == startN1Idx && S1Idx == startS1Idx) ? startS2Idx : 0;
                    int curEndS2 = (BIdx == endBIdx && N1Idx == endN1Idx && S1Idx == endS1Idx)
                                       ? endS2Idx
                                       : static_cast<int>(p.curKSBlockNum);

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

    uint32_t actualSplitNum =
        (splitIdx + 1 > static_cast<int32_t>(ctx.blockDim)) ? ctx.blockDim : static_cast<uint32_t>(splitIdx + 1);
    tiling->set_totalSplitNodeNum(actualSplitNum);
    tiling->set_splitLseTotalSize(currentLseTaskOffset * SIZE_OF_32BIT);
    tiling->set_splitOTotalSize(currentOTaskOffset * SIZE_OF_32BIT);
}

inline uint32_t countForceSplitSegments(uint32_t groupSize, const SplitContext& ctx)
{
    uint32_t totalSegs = 0;
    uint32_t nsplit = static_cast<uint32_t>(ctx.num_splits);
    for (int32_t b = 0; b < ctx.batch_size; b++) {
        BatchParams p = getBatchParams(b, groupSize, ctx);
        uint32_t curKSBlockNum = p.curKSBlockNum;
        if (curKSBlockNum == 0) {
            continue;
        }
        uint32_t blocksPerSplit = (curKSBlockNum + nsplit - 1) / nsplit;
        if (blocksPerSplit == 0) {
            blocksPerSplit = 1;
        }
        uint32_t actualSplits = (curKSBlockNum + blocksPerSplit - 1) / blocksPerSplit;
        totalSegs += p.curQNBlockNum * p.curQSBlockNum * actualSplits;
    }
    return totalSegs;
}

inline void fillCoreInfoForceSplit(FAInferStaticTilingData* tiling, uint32_t groupSize, const SplitContext& ctx)
{
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

    uint32_t nsplit = static_cast<uint32_t>(ctx.num_splits);
    uint32_t coreIdx = 0;
    for (int32_t b = 0; b < ctx.batch_size; b++) {
        BatchParams p = getBatchParams(b, groupSize, ctx);
        uint32_t curKSBlockNum = p.curKSBlockNum;
        if (curKSBlockNum == 0) {
            continue;
        }
        uint32_t blocksPerSplit = (curKSBlockNum + nsplit - 1) / nsplit;
        if (blocksPerSplit == 0) {
            blocksPerSplit = 1;
        }
        for (uint32_t n1 = 0; n1 < p.curQNBlockNum; n1++) {
            for (uint32_t s1 = 0; s1 < p.curQSBlockNum; s1++) {
                for (uint32_t segStart = 0; segStart < curKSBlockNum; segStart += blocksPerSplit) {
                    uint32_t segEnd = FaMin(segStart + blocksPerSplit, curKSBlockNum);
                    if (coreIdx >= ctx.blockDim) {
                        break;
                    }
                    tiling->coreInfo[coreIdx].startBIdx = b;
                    tiling->coreInfo[coreIdx].endBIdx = b;
                    tiling->coreInfo[coreIdx].startN1Idx = static_cast<int>(n1);
                    tiling->coreInfo[coreIdx].endN1Idx = static_cast<int>(n1);
                    tiling->coreInfo[coreIdx].startS1Idx = static_cast<int>(s1);
                    tiling->coreInfo[coreIdx].endS1Idx = static_cast<int>(s1);
                    tiling->coreInfo[coreIdx].startS2Idx = static_cast<int>(segStart);
                    tiling->coreInfo[coreIdx].endS2Idx = static_cast<int>(segEnd);
                    coreIdx++;
                }
            }
        }
    }
    tiling->set_needCoreNum(coreIdx);
}

inline void fillCoreInfoNoSplit(FAInferStaticTilingData* tiling, uint32_t groupSize, const SplitContext& ctx)
{
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

    uint32_t totalTasks = 0;
    for (int32_t b = 0; b < ctx.batch_size; b++) {
        BatchParams p = getBatchParams(b, groupSize, ctx);
        totalTasks += p.curQNBlockNum * p.curQSBlockNum;
    }
    if (totalTasks == 0) {
        tiling->set_needCoreNum(0);
        return;
    }

    uint32_t tasksPerCore = (totalTasks + ctx.blockDim - 1) / ctx.blockDim;

    auto locate = [&](uint32_t gidx, int32_t& B, int32_t& N1, int32_t& S1, uint32_t& ksBlk) {
        uint32_t acc = 0;
        for (int32_t b = 0; b < ctx.batch_size; b++) {
            BatchParams p = getBatchParams(b, groupSize, ctx);
            uint32_t cnt = p.curQNBlockNum * p.curQSBlockNum;
            if (cnt == 0) {
                continue;
            }
            if (gidx < acc + cnt) {
                uint32_t local = gidx - acc;
                B = b;
                N1 = static_cast<int32_t>(local / p.curQSBlockNum);
                S1 = static_cast<int32_t>(local % p.curQSBlockNum);
                ksBlk = p.curKSBlockNum;
                return;
            }
            acc += cnt;
        }
        B = ctx.batch_size - 1;
        BatchParams p = getBatchParams(B, groupSize, ctx);
        N1 = static_cast<int32_t>(p.curQNBlockNum) - 1;
        S1 = static_cast<int32_t>(p.curQSBlockNum) - 1;
        ksBlk = p.curKSBlockNum;
    };

    uint32_t usedCores = 0;
    for (uint32_t coreIdx = 0; coreIdx < ctx.blockDim; coreIdx++) {
        uint32_t taskLo = coreIdx * tasksPerCore;
        if (taskLo >= totalTasks) {
            break;
        }
        uint32_t taskHi = FaMin(taskLo + tasksPerCore, totalTasks);

        int32_t b0, n0, s0;
        uint32_t ks0;
        int32_t b1, n1, s1;
        uint32_t ks1;
        locate(taskLo, b0, n0, s0, ks0);
        locate(taskHi - 1, b1, n1, s1, ks1);

        tiling->coreInfo[coreIdx].startBIdx = b0;
        tiling->coreInfo[coreIdx].startN1Idx = n0;
        tiling->coreInfo[coreIdx].startS1Idx = s0;
        tiling->coreInfo[coreIdx].startS2Idx = 0;
        tiling->coreInfo[coreIdx].endBIdx = b1;
        tiling->coreInfo[coreIdx].endN1Idx = n1;
        tiling->coreInfo[coreIdx].endS1Idx = s1;
        tiling->coreInfo[coreIdx].endS2Idx = static_cast<int32_t>(ks1);
        usedCores = coreIdx + 1;
    }
    tiling->set_needCoreNum(usedCores);
}

// Dedicated gemv absorbed-MLA decode: host sinks only per-batch split rules,
// the kernel re-derives the (batch, split) table; each core takes a slice.
inline void splitKvForGemvDecode(FAInferStaticTilingData* tiling, const SplitContext& ctx, uint32_t kvTile = 128)
{
    constexpr uint32_t SIZE_OF_32BIT = 4;
    // FA4 has no append-KV here, so SplitContext carries no kvNewSeqlen.
    constexpr uint32_t kvNewSeqlen = 0;
    const uint32_t qHeads = static_cast<uint32_t>(ctx.num_heads);
    const uint32_t headSizeV = static_cast<uint32_t>(ctx.head_size_v);

    uint32_t batchWithWork = 0;
    for (int32_t b = 0; b < ctx.batch_size; b++) {
        uint32_t kvLen = static_cast<uint32_t>(ctx.seqlens_k_cpu[b] + kvNewSeqlen);
        if (kvLen > 0) {
            batchWithWork++;
        }
    }
    if (batchWithWork == 0) {
        tiling->set_needCoreNum(0);
        tiling->set_totalTaskNum(0);
        tiling->set_nsplitPerBatch(1U);
        tiling->set_totalSplitNodeNum(0);
        tiling->set_splitLseTotalSize(0);
        tiling->set_splitOTotalSize(0);
        return;
    }

    // Split budget: explicit num_splits wins, else ~SEG_TARGET=4 tiles/split
    // (c~1.4us/split, m~2.4us/tile => 4 optimal); batch*splits<=blockDim.
    constexpr uint32_t SEG_TARGET = 4;
    const uint32_t maxSplitsPerBatch = (ctx.blockDim / batchWithWork) ? (ctx.blockDim / batchWithWork) : 1U;
    uint32_t nsplitPerBatch = maxSplitsPerBatch;
    if (ctx.num_splits > 1) {
        // An explicit budget (num_splits > 1) wins; num_splits <= 1 is not
        // honoured as one core (training passes 1, kvcache 0) -> auto below.
        nsplitPerBatch = static_cast<uint32_t>(ctx.num_splits);
    } else {
        uint32_t maxSegCount = 0;
        for (int32_t b = 0; b < ctx.batch_size; b++) {
            uint32_t kvLen = static_cast<uint32_t>(ctx.seqlens_k_cpu[b] + kvNewSeqlen);
            uint32_t segCount = (kvLen + kvTile - 1) / kvTile;
            if (segCount > maxSegCount) {
                maxSegCount = segCount;
            }
        }
        uint32_t target = (maxSegCount + SEG_TARGET - 1) / SEG_TARGET;
        nsplitPerBatch = (target == 0) ? 1U : target;
    }
    if (nsplitPerBatch == 0) {
        nsplitPerBatch = 1;
    }
    if (nsplitPerBatch > maxSplitsPerBatch) {
        nsplitPerBatch = maxSplitsPerBatch;
    }

    // Rule-only accumulation: device re-derives (b, s, segStart, segEnd) from
    // nsplitPerBatch; host keeps only task/partial totals (no coreInfo).
    uint32_t totalTaskNum = 0;
    int32_t splitIdx = 0;
    int64_t lseTaskOffset = 0;
    int64_t oTaskOffset = 0;
    bool anySplit = false;
    for (int32_t b = 0; b < ctx.batch_size; b++) {
        uint32_t kvLen = static_cast<uint32_t>(ctx.seqlens_k_cpu[b] + kvNewSeqlen);
        uint32_t segCount = (kvLen + kvTile - 1) / kvTile;
        if (segCount == 0) {
            continue;
        }
        uint32_t blocksPerSplit = (segCount + nsplitPerBatch - 1) / nsplitPerBatch;
        if (blocksPerSplit == 0) {
            blocksPerSplit = 1;
        }
        uint32_t actualSplits = (segCount + blocksPerSplit - 1) / blocksPerSplit;
        if (actualSplits > 1) {
            anySplit = true;
        }
        totalTaskNum += actualSplits;
        lseTaskOffset += static_cast<int64_t>(actualSplits) * qHeads;
        oTaskOffset += static_cast<int64_t>(actualSplits) * qHeads * headSizeV;
        splitIdx++;
    }

    tiling->set_needCoreNum(FaMin(totalTaskNum, ctx.blockDim));
    tiling->set_totalTaskNum(totalTaskNum);
    tiling->set_nsplitPerBatch(nsplitPerBatch);
    if (anySplit) {
        // Only split batches write partials and need the combine pass; when
        // all batches are single tasks, cores write output (partials unused).
        tiling->set_totalSplitNodeNum(static_cast<uint32_t>(splitIdx));
        tiling->set_splitLseTotalSize(static_cast<uint64_t>(lseTaskOffset) * SIZE_OF_32BIT);
        tiling->set_splitOTotalSize(static_cast<uint64_t>(oTaskOffset) * SIZE_OF_32BIT);
        // CombineScale group table in fixed splitInfo[25], read by the kernel:
        // rows = batches with work (<= blockDim/2 < 25), row idx = ordinal.
        uint32_t row = 0;
        int64_t lseAcc = 0;
        int64_t oAcc = 0;
        for (int32_t b = 0; b < ctx.batch_size; b++) {
            uint32_t kvLen = static_cast<uint32_t>(ctx.seqlens_k_cpu[b] + kvNewSeqlen);
            uint32_t segCount = (kvLen + kvTile - 1) / kvTile;
            if (segCount == 0) {
                continue;
            }
            uint32_t blocksPerSplit = (segCount + nsplitPerBatch - 1) / nsplitPerBatch;
            if (blocksPerSplit == 0) {
                blocksPerSplit = 1;
            }
            uint32_t actualSplits = (segCount + blocksPerSplit - 1) / blocksPerSplit;
            tiling->splitInfo[row].batchIdx = b;
            tiling->splitInfo[row].headStartIdx = 0;
            tiling->splitInfo[row].headEndIdx = static_cast<int>(qHeads);
            tiling->splitInfo[row].qStartIdx = 0;
            tiling->splitInfo[row].qEndIdx = 1;
            tiling->splitInfo[row].splitNum = static_cast<int>(actualSplits);
            tiling->splitInfo[row].lseTaskOffset = lseAcc;
            tiling->splitInfo[row].oTaskOffset = oAcc;
            lseAcc += static_cast<int64_t>(actualSplits) * qHeads;
            oAcc += static_cast<int64_t>(actualSplits) * qHeads * headSizeV;
            row++;
        }
    } else {
        tiling->set_totalSplitNodeNum(0);
        tiling->set_splitLseTotalSize(0);
        tiling->set_splitOTotalSize(0);
    }
}

inline void splitBN2S1GS2(FAInferStaticTilingData* tiling, const SplitContext& ctx)
{
    uint64_t totalTaskNum = 0;
    uint32_t groupSize = ctx.num_heads / ctx.num_heads_k;

    if (ctx.num_splits >= 1) {
        uint32_t totalSegs = countForceSplitSegments(groupSize, ctx);
        uint32_t coreCap = FaMin(ctx.blockDim, static_cast<uint32_t>(25));
        if (totalSegs > 0 && totalSegs <= coreCap) {
            fillCoreInfoForceSplit(tiling, groupSize, ctx);
            fillSplitInfoForFlashDecode(tiling, groupSize, ctx);
            return;
        }
        if (ctx.num_splits == 1) {
            fillCoreInfoNoSplit(tiling, groupSize, ctx);
            fillSplitInfoForFlashDecode(tiling, groupSize, ctx);
            return;
        }
    }

    for (int32_t batchIdx = 0; batchIdx < ctx.batch_size; batchIdx++) {
        BatchParams p = getBatchParams(batchIdx, groupSize, ctx);
        totalTaskNum += static_cast<uint64_t>(ctx.num_heads) * p.qSeqlen * p.kvSeqlen;
    }
    uint64_t perCoreTaskNum = (totalTaskNum + ctx.blockDim - 1) / ctx.blockDim;
    fillCoreInfoForFlashDecode(tiling, groupSize, perCoreTaskNum, ctx);
    fillSplitInfoForFlashDecode(tiling, groupSize, ctx);
}

} // namespace fa_split

#endif
