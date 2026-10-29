/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 *
 * Migrated from ops-transformer FAG basic non-det vec_addr:
 * ROW_4 / ROW_2 packing aligned with cube_addr dual-list schedule.
 */

#ifndef __VECTORAddr_H__
#define __VECTORAddr_H__

#include "common_header.h"

template <MaskType maskType, InputLayout inputLayout>
class VectorAddr {
public:
    struct VecAddrInfo *globalVecAddr;

    int32_t s1BlockNum;
    int32_t s2BlockNum;
    int32_t s2TailLength;
    int32_t s1TailLength;
    int32_t roundId = 1;
    int32_t g;
    int32_t lastBatchQSum;
    int32_t lastBatchKSum;

    __aicore__ uint64_t getSeqRealLength(int32_t sIdx, int32_t len, int32_t s_block_num, int32_t s_tail)
    {
        if (s_tail > 0 && (sIdx + len == s_block_num)) {
            return (len - 1) * BASE_BLOCK_LENGTH + s_tail;
        } else {
            return len * BASE_BLOCK_LENGTH;
        }
    }

    __aicore__ int64_t getSeqLenQ(int32_t i)
    {
        if constexpr (inputLayout == InputLayout::TND) {
            if (i == 0) {
                return ((__gm__ int32_t *)cu_seq_qlen_addr)[0];
            }
            return ((__gm__ int32_t *)cu_seq_qlen_addr)[i] - ((__gm__ int32_t *)cu_seq_qlen_addr)[i - 1];
        } else {
            return qSeqlenFixed;
        }
    }

    __aicore__ int64_t getSeqLenK(int32_t i)
    {
        if constexpr (inputLayout == InputLayout::TND) {
            if (i == 0) {
                return ((__gm__ int32_t *)cu_seq_kvlen_addr)[0];
            }
            return ((__gm__ int32_t *)cu_seq_kvlen_addr)[i] - ((__gm__ int32_t *)cu_seq_kvlen_addr)[i - 1];
        } else {
            return kSeqlenFixed;
        }
    }

    __aicore__ int64_t getTotalLenQ(int32_t i)
    {
        if constexpr (inputLayout == InputLayout::TND) {
            return ((__gm__ int32_t *)cu_seq_qlen_addr)[i];
        } else {
            return (i + 1) * qSeqlenFixed;
        }
    }

    __aicore__ int64_t getTotalLenK(int32_t i)
    {
        if constexpr (inputLayout == InputLayout::TND) {
            return ((__gm__ int32_t *)cu_seq_kvlen_addr)[i];
        } else {
            return (i + 1) * kSeqlenFixed;
        }
    }

    __aicore__ void getOffset(VecBlockInfo &vecPhyAddr, int32_t s1Id, int32_t s2Id, int32_t blockId, int row,
                              int col)
    {
        vecPhyAddr.batchIdx = batchIdx;
        vecPhyAddr.nheadsIdx = headNumIdx;
        vecPhyAddr.SeqQIdx = s1Id + row;
        vecPhyAddr.SeqKIdx = s2Id + col;
        vecPhyAddr.nheadsKIdx = headNumIdx / g;
        vecPhyAddr.gIdx = headNumIdx % g;
        vecPhyAddr.offset = blockId * BASE_BLOCK_LENGTH * BASE_BLOCK_LENGTH;
        vecPhyAddr.lengthy = BASE_BLOCK_LENGTH;
        vecPhyAddr.lengthx = BASE_BLOCK_LENGTH;

        if ((row + s1Id == s1BlockNum - 1) && s1TailLength > 0) {
            vecPhyAddr.lengthy = s1TailLength;
        }
        if ((col + s2Id == s2BlockNum - 1) && s2TailLength > 0) {
            vecPhyAddr.lengthx = s2TailLength;
        }
    }

    __aicore__ void blockAdd(int ky)
    {
        int col = (VEC_BLOCK_LIMIT - num) / ky;
        int startS1Id = s1Idx;
        int startS2Id = s2Idx;
        bool ur = true;
        int kx;
        int cnt;

        if constexpr (maskType == MaskType::NO_MASK) {
            if (startS2Id + col < s2BlockNum) {
                s2Idx += col;
            } else {
                col = s2BlockNum - startS2Id;
                s2Idx = 0;
                s1Idx += ky;
            }

            record(startS1Id, startS2Id, col, ky, col * ky, false, lastBatchQSum, lastBatchKSum);
            num += col * ky;
            return;
        }

        if (startS2Id + col == startS1Id + 1) {
            col = (VEC_BLOCK_LIMIT + 1 - num) / ky;
        }

        int reachS2Edge = s2BlockNum < startS1Id + ky;
        int kxMax = reachS2Edge ? s2BlockNum : startS1Id + ky;

        if (col + startS2Id >= kxMax) {
            kx = kxMax - startS2Id;
            s2Idx = 0;
            s1Idx += ky;

            ur = (reachS2Edge || ky == 1);
            cnt = kx * ky - !ur;
        } else {
            kx = col;
            s2Idx += kx;
            cnt = kx * ky;
            if (startS1Id + 1 < startS2Id + kx) {
                ur = false;
            }
        }

        record(startS1Id, startS2Id, kx, ky, cnt, ur, lastBatchQSum, lastBatchKSum);
        num += cnt;
    }

    __aicore__ __inline__ void record(int32_t s1Id, int32_t s2Id, int32_t kx, int32_t ky, int32_t cnt, int32_t /*ur*/,
                                      int32_t /*leftLastBatchSum*/, int32_t /*rightLastBatchSum*/)
    {
        if (coreSegmentBlockNum % coreNum != cubeIdx) {
            return;
        }
        int32_t blockId = num;
        for (int x = 0; x < kx; x++) {
            for (int y = 0; y < ky; y++) {
                if constexpr (maskType == MaskType::NO_MASK) {
                    getOffset(globalVecAddr->VecBlkInfo[blockId], s1Id, s2Id, blockId, y, x);
                    blockId++;
                } else {
                    if (s1Id + y >= s2Id + x) {
                        getOffset(globalVecAddr->VecBlkInfo[blockId], s1Id, s2Id, blockId, y, x);
                        blockId++;
                    }
                }
            }
        }
        globalVecAddr->blockLength += cnt;
    }

    __aicore__ int32_t addr_mapping(struct VecAddrInfo *vecAddrInfo)
    {
        globalVecAddr = vecAddrInfo;
        globalVecAddr->blockLength = 0;
        int having = 0;

        while (1) {
            if (s1Idx < s1BlockNum && s2Idx < s2BlockNum) {
                for (int lensIdx = 0; lensIdx < 2; lensIdx++) {
                    int lens = lengths[lensIdx];
                    if ((num + lens) <= VEC_BLOCK_LIMIT && (s1Idx + lens - 1) < s1BlockNum) {
                        if (s2Idx != 0 && lens > s1MaxLength - s1Idx) {
                            continue;
                        }

                        if (s2Idx == 0) {
                            s1MaxLength = s1Idx + lens;
                        }

                        if (lens == ROW_4) {
                            int col = (VEC_BLOCK_LIMIT - num) / lens;
                            int kx;
                            bool ur;

                            int reachS2Edge = s2BlockNum <= s1Idx + 1;
                            int kxMax =
                                (maskType == MaskType::NO_MASK || reachS2Edge) ? s2BlockNum : s1Idx + 1;

                            if ((maskType == MaskType::NO_MASK) || (s2Idx + col < kxMax) || reachS2Edge) {
                                kx = s2Idx + col < kxMax ? col : s2BlockNum - s2Idx;
                                ur = true;
                                record(s1Idx, s2Idx, kx, ROW_2, kx * ROW_2, ur, lastBatchQSum, lastBatchKSum);
                                num += kx * ROW_2;
                                record(s1Idx + ROW_2, s2Idx, kx, ROW_2, kx * ROW_2, ur, lastBatchQSum,
                                       lastBatchKSum);
                                num += kx * ROW_2;
                                s2Idx += kx;

                                if (s2Idx >= s2BlockNum) {
                                    s2Idx = 0;
                                    s1Idx += ROW_4;
                                }
                            } else if (s2Idx + col > s1Idx + 1) {
                                kx = s1Idx + ROW_2 - s2Idx;
                                ur = false;
                                record(s1Idx, s2Idx, kx, ROW_2, kx * ROW_2 - 1, ur, lastBatchQSum, lastBatchKSum);
                                num += kx * ROW_2 - 1;
                                s1Idx += ROW_2;

                                blockAdd(ROW_2);
                            } else {
                                kx = (VEC_BLOCK_LIMIT + 1 - num) / lens;
                                int have_ur_block = kx != col;
                                ur = kx == col;
                                record(s1Idx, s2Idx, kx, ROW_2, kx * ROW_2 - have_ur_block, ur, lastBatchQSum,
                                       lastBatchKSum);
                                num += kx * ROW_2 - have_ur_block;
                                record(s1Idx + ROW_2, s2Idx, kx, ROW_2, kx * ROW_2, true, lastBatchQSum,
                                       lastBatchKSum);
                                num += kx * ROW_2;

                                s2Idx += kx;
                                if (have_ur_block) {
                                    s1Idx += ROW_2;
                                }
                            }
                        } else {
                            blockAdd(lens);
                        }

                        break;
                    }
                }

                if (s1MaxLength - s1Idx + num >= VEC_BLOCK_LIMIT) {
                    num = 0;
                    coreSegmentBlockNum++;
                    if (coreSegmentBlockNum == roundId * coreNum) {
                        break;
                    }
                }

                if (s1Idx >= s1BlockNum && batchIdx >= batchNum - 1 && headNumIdx >= headNum - 1) {
                    return having;
                }
            } else {
                s1Idx = 0;
                s2Idx = 0;
                if (headNumIdx == headNum - 1 || s1 == 0 || s2 == 0) {
                    headNumIdx = 0;

                    if (batchIdx < batchNum) {
                        lastBatchQSum = getTotalLenQ(batchIdx);
                        lastBatchKSum = getTotalLenK(batchIdx);
                        batchIdx += 1;
                        if (batchIdx < batchNum) {
                            s1 = getSeqLenQ(batchIdx);
                            s2 = getSeqLenK(batchIdx);
                        } else {
                            s1 = 0;
                            s2 = 0;
                        }
                        s2TailLength = s2 % BASE_BLOCK_LENGTH;
                        s1TailLength = s1 % BASE_BLOCK_LENGTH;

                        s1BlockNum = (s1 + BASE_BLOCK_LENGTH - 1) / BASE_BLOCK_LENGTH;
                        s2BlockNum = (s2 + BASE_BLOCK_LENGTH - 1) / BASE_BLOCK_LENGTH;
                    } else {
                        break;
                    }
                } else {
                    headNumIdx++;
                }
            }

            if (num > VEC_BLOCK_LIMIT - VEC_COL_LIMIT) {
                num = 0;
                coreSegmentBlockNum++;
                if (coreSegmentBlockNum == roundId * coreNum) {
                    break;
                }
            }
        }
        roundId++;
        return having;
    }

    __aicore__ void init(int32_t batchSize, int32_t heads, int32_t gin, int32_t headDims, uint32_t core_index,
                         uint32_t seq_q_len, uint32_t seq_k_len, __gm__ uint8_t *actual_seq_qlen,
                         __gm__ uint8_t *actual_seq_kvlen, uint32_t cubeNum)
    {
        batchNum = batchSize;
        coreNum = cubeNum;
        headNum = heads;
        headDim = headDims;
        cu_seq_qlen_addr = actual_seq_qlen;
        cu_seq_kvlen_addr = actual_seq_kvlen;
        g = gin;
        qSeqlenFixed = seq_q_len;
        kSeqlenFixed = seq_k_len;
        // AIV cores are paired with AIC; map to cube core id.
        cubeIdx = core_index;
        batchIdx = 0;
        headNumIdx = 0;
        s1Idx = 0;
        s2Idx = 0;
        s1MaxLength = 0;
        num = 0;
        coreSegmentBlockNum = 0;
        roundId = 1;

        if constexpr (inputLayout == InputLayout::TND) {
            s1 = getSeqLenQ(batchIdx);
            s2 = getSeqLenK(batchIdx);
        } else {
            s1 = qSeqlenFixed;
            s2 = kSeqlenFixed;
        }

        s1BlockNum = (s1 + 127) / 128;
        s2BlockNum = (s2 + 127) / 128;
        s2TailLength = s2 % 128;
        s1TailLength = s1 % 128;
        lastBatchQSum = 0;
        lastBatchKSum = 0;
    }

private:
    int32_t batchIdx = 0;
    int32_t headNumIdx = 0;
    int32_t s1Idx = 0;
    int32_t s2Idx = 0;
    int32_t s1MaxLength = 0;
    int32_t s1 = 0;
    int32_t s2 = 0;
    int32_t num = 0;
    int32_t coreSegmentBlockNum = 0;
    int32_t batchNum = 0;
    int32_t coreNum = 0;
    int32_t headNum = 0;
    int32_t headDim = 0;
    int32_t qSeqlenFixed = 0;
    int32_t kSeqlenFixed = 0;

    constexpr static int32_t VEC_BLOCK_LIMIT = 16;
    constexpr static int32_t VEC_COL_LIMIT = 4;
    constexpr static int32_t BASE_BLOCK_LENGTH = 128;
    constexpr static int32_t ROW_2 = 2;
    constexpr static int32_t ROW_4 = 4;

    __gm__ uint8_t *cu_seq_qlen_addr;
    __gm__ uint8_t *cu_seq_kvlen_addr;

    int32_t cubeIdx;
    // Keep in sync with cube_addr lengths (ROW_4 deferred until Cube3 m_loop>2 is ready).
    int32_t lengths[2] = {2, 1};
};
#endif
