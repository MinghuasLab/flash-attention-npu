/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef __CUBEAddr_H__
#define __CUBEAddr_H__

#include "common_header.h"

template<MaskType maskType, InputLayout inputLayout>
class CubeAddr {

public:
    int32_t batch;
    int32_t nheads;
    int32_t headdim;
    int32_t g;
    int32_t nheads_k;
    int32_t coreId = 0;

    int32_t coreSegmentBlockNum = 0;
    int32_t SegmentBlockNum = 0;
    int32_t blockNum = 0;

    int32_t batchIdx;
    int32_t nheadsIdx;
    int32_t qSeqIdx;
    int32_t seqKIdx;
    int32_t qSeqlen;
    int32_t kSeqlen;
    int32_t s1BlockNum;
    int32_t s1TailLength;
    int32_t s2BlockNum;
    int32_t s2TailLength;
    int32_t s1GuardInterval;

    int32_t limit;
    int32_t coreNum;
    int32_t roundId;

    int32_t overFlag = 1;
    int32_t lastBatchSum = 0; 

    struct CubeAddrInfo * globalCubeAddr;

    __gm__ uint8_t *cu_seq_qlen_addr;
    __gm__ uint8_t *cu_seq_kvlen_addr;
    __gm__ uint8_t *q_gm_addr;
    __gm__ uint8_t *k_gm_addr;
    __gm__ uint8_t *v_gm_addr;
    __gm__ uint8_t *dout_gm_addr;
    __gm__ uint8_t *user_gm_addr;

    __aicore__ uint64_t getSeqRealLength(int32_t sIdx, int32_t len, int32_t s_block_num, int32_t s_tail) {
        if (s_tail > 0 && (sIdx + len == s_block_num)) {
            return (len - 1) * 128 + s_tail;
        } else {
            return len * 128;
        }
    }
    
    __aicore__ int64_t getTotalLen(int32_t i) {
        int64_t cuTotalSeqQlen = 0;
        if (inputLayout == InputLayout::TND) {
            cuTotalSeqQlen = ((__gm__ int32_t *)cu_seq_qlen_addr)[i];
        } else {
            cuTotalSeqQlen = (i + 1) * qSeqlen;
        }
        return cuTotalSeqQlen;
    }

    __aicore__ uint64_t getLeftAddr(int32_t batchIdx, int32_t nheadsIdx, int32_t qSeqlen, int32_t qSeqIdx, int32_t headdim) {
        return lastBatchSum * nheads * headdim + (qSeqIdx * 128 * nheads + nheadsIdx) * headdim;
    }

    __aicore__ uint64_t getRightAddr(int32_t batchIdx, int32_t nheadsIdx, int32_t kSeqlen, int32_t seqKIdx, int32_t headdim) {
        return lastBatchSum * nheads_k * headdim + (seqKIdx * 128 * nheads_k + (nheadsIdx / g)) * headdim;
    }

    __aicore__ uint64_t getOutAddr(int32_t workspacePos) {
        return workspacePos * 128 * 128;
    }

    __aicore__ int32_t addr_mapping(struct CubeAddrInfo * cubeAddrInfo) {
        globalCubeAddr = cubeAddrInfo;
        globalCubeAddr->blockLength = 0;

        int32_t loopCnt = 0;
        while (overFlag) {
            int32_t guardLen = qSeqIdx + s1GuardInterval - seqKIdx;
            int32_t reserve = limit - blockNum;
            // TODO sparsemode = 0
            int32_t realLenAlign = (reserve + s1GuardInterval - 1) / s1GuardInterval;
            if (realLenAlign >= guardLen) {
                if (coreSegmentBlockNum % coreNum == coreId) {
                    int32_t index = globalCubeAddr->blockLength;
                    globalCubeAddr->addrInfo[index].left = getLeftAddr(batchIdx, nheadsIdx, qSeqlen, qSeqIdx, headdim);
                    globalCubeAddr->addrInfo[index].right = getRightAddr(batchIdx, nheadsIdx, kSeqlen, seqKIdx, headdim);
                    globalCubeAddr->addrInfo[index].out = getOutAddr(blockNum);
                    globalCubeAddr->addrInfo[index].ky = getSeqRealLength(qSeqIdx, s1GuardInterval, s1BlockNum, s1TailLength);
                    globalCubeAddr->addrInfo[index].kx = getSeqRealLength(seqKIdx, guardLen, s2BlockNum, s2TailLength);
                    globalCubeAddr->addrInfo[index].lowerLeft = 1;
                    // TODO sparsemode = 0
                    globalCubeAddr->addrInfo[index].upperRight = (s1GuardInterval % 2);
                    globalCubeAddr->blockLength ++;
                }
                // TODO sparsemode = 0
                blockNum += s1GuardInterval * guardLen - (s1GuardInterval + 1) % 2;
                qSeqIdx += s1GuardInterval;
                seqKIdx = 0;
                if (qSeqIdx == s1BlockNum - 1) {
                    s1GuardInterval = 1;
                }
            } else {
                int32_t realLen = (reserve / s1GuardInterval);
                if (coreSegmentBlockNum % coreNum == coreId) {
                    int32_t index = globalCubeAddr->blockLength;
                    globalCubeAddr->addrInfo[index].left = getLeftAddr(batchIdx, nheadsIdx, qSeqlen, qSeqIdx, headdim);
                    globalCubeAddr->addrInfo[index].right = getRightAddr(batchIdx, nheadsIdx, kSeqlen, seqKIdx, headdim);
                    globalCubeAddr->addrInfo[index].out = getOutAddr(blockNum);
                    globalCubeAddr->addrInfo[index].ky = getSeqRealLength(qSeqIdx, s1GuardInterval, s1BlockNum, s1TailLength);
                    globalCubeAddr->addrInfo[index].kx = getSeqRealLength(seqKIdx, realLen, s2BlockNum, s2TailLength);
                    globalCubeAddr->addrInfo[index].lowerLeft = 1;
                    globalCubeAddr->addrInfo[index].upperRight = 1;
                    globalCubeAddr->blockLength ++;
                }
                blockNum += s1GuardInterval * realLen;
                seqKIdx += realLen;
            }
            SegmentBlockNum++;
            if ((qSeqIdx == s1BlockNum) && (batchIdx == batch - 1) && (nheadsIdx == nheads - 1)) {
                overFlag = 0;
                break;
            }

            if (qSeqIdx == s1BlockNum) {
                if (nheadsIdx == nheads - 1) {
                    lastBatchSum = getTotalLen(batchIdx);
                    batchIdx++;
                    nheadsIdx = 0;
                    if (inputLayout == InputLayout::TND) { 
                        qSeqlen = getSeqLen(batchIdx);
                        kSeqlen = getSeqLen(batchIdx);
                    }
                    s1BlockNum = (qSeqlen + 127) / 128;
                    s1TailLength = qSeqlen % 128;
                    s2BlockNum = (kSeqlen + 127) / 128;
                    s2TailLength = kSeqlen % 128;
                } else {
                    nheadsIdx++;
                }
                qSeqIdx = 0;
                seqKIdx = 0;
                s1GuardInterval = (s1BlockNum == 1) ? 1 : 2;
            }

            if (blockNum >= 15) {
                coreSegmentBlockNum++;
                SegmentBlockNum = 0;
                blockNum = 0;
                if (coreSegmentBlockNum == roundId * coreNum) {
                    break;
                }
            }
        }

        roundId++;
        return overFlag;
    }

    __aicore__ int64_t getSeqLen(int32_t i) {
        int64_t cuSeqQlen;
        if (i == 0) {
            cuSeqQlen = ((__gm__ int32_t *)cu_seq_qlen_addr)[0];
        } else {
            cuSeqQlen = ((__gm__ int32_t *)cu_seq_qlen_addr)[i] - ((__gm__ int32_t *)cu_seq_qlen_addr)[i - 1];
        }
        return cuSeqQlen;
    }

    __aicore__ void init(int32_t batchIn, int32_t nheadsIn, int32_t gIn, int32_t headDimIn, uint32_t coreIdx, 
        uint32_t seq_q_len, uint32_t seq_k_len,
        __gm__ uint8_t *cu_seq_qlen, __gm__ uint8_t *cu_seq_kvlen, uint32_t totalCoreNum) {
        
        batch = batchIn;
        nheads = nheadsIn;
        g = gIn;
        headdim = headDimIn;
        nheads_k = nheads / g;

        cu_seq_qlen_addr = cu_seq_qlen;
        cu_seq_kvlen_addr = cu_seq_kvlen;

        coreSegmentBlockNum = 0;
        SegmentBlockNum = 0;
        blockNum = 0;

        batchIdx = 0;
        nheadsIdx = 0;
        qSeqIdx = 0;
        seqKIdx = 0;
        if (inputLayout == InputLayout::TND) {     
            qSeqlen = getSeqLen(batchIdx);
            kSeqlen = getSeqLen(batchIdx);
        } else {
            qSeqlen = seq_q_len;
            kSeqlen = seq_k_len;
        }
        s1BlockNum = (qSeqlen + 127) / 128;
        s1TailLength = qSeqlen % 128;
        s2BlockNum = (kSeqlen + 127) / 128;
        s2TailLength = kSeqlen % 128;
        s1GuardInterval = (s1BlockNum == 1) ? 1 : 2;

        limit = 16;
        coreNum = totalCoreNum;
        coreId = coreIdx;

        roundId = 1;
        overFlag = 1;
        lastBatchSum = 0;
    }
};
#endif