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

#ifndef FLASH_ATTENTION_REGULAR_H
#define FLASH_ATTENTION_REGULAR_H

#include <string>

struct coreNode {
    int startBIdx;
    int startN1Idx;
    int startS1Idx;
    int startS2Idx;
    int endBIdx;
    int endN1Idx;
    int endS1Idx;
    int endS2Idx;
    int64_t firstSplitKVTaskLseOffset;
    int64_t firstSplitKVTaskOOffset;
};

struct splitNode {
    int batchIdx;
    int headStartIdx;
    int headEndIdx;
    int qStartIdx;
    int qEndIdx;
    int splitNum;
    int64_t lseTaskOffset;
    int64_t oTaskOffset;
};

struct FAInferTilingData {
    uint32_t numHeads;
    uint32_t embeddingSize;
    uint32_t embeddingSizeV;
    uint32_t numBlocks;
    uint32_t blockSize;
    uint32_t maxQSeqlen;
    uint32_t maxKvSeqlen;
    uint32_t kvHeads;
    uint32_t batch;
    uint32_t maxNumBlocksPerBatch;
    uint32_t firstBatchTaskNum;
    uint32_t totalTaskNum;
    uint32_t maskType;
    uint64_t qkOutSize;
    uint64_t smOnlineOutSize;
    uint64_t pvOutSize;
    uint64_t UpdateSize;
    uint64_t workSpaceSize;
    float scaleValue;
    uint64_t padding1;
    uint64_t padding2;
    uint32_t padding3;
    int64_t preToken = 0;
    int64_t nextToken = 0;
    int32_t sparseMode = 0;
    uint32_t globalWindowSize = 4;
    uint32_t localWindowSize = 0;
    std::string cacheLayout = "nd";
    uint64_t splitLseTotalSize;
    uint64_t splitOTotalSize;
    uint32_t totalSplitNodeNum;
    uint32_t needCoreNum;
    coreNode coreInfo[25];
    splitNode splitInfo[25];

    int64_t qSeqlenAligned;
    int64_t kvSeqlenAligned;
    uint32_t innerPrec;
    uint32_t actSeqAval;
    uint32_t qBaseTile;
    uint32_t kvBaseTile;
    uint32_t qkL1TileM;
    uint32_t qkL1TileN;
    uint32_t qkL1TileKLeft;
    uint32_t qkL1TileKRight;
    uint32_t pvL1TileM;
    uint32_t pvL1TileN;
    uint32_t pvL1TileKLeft;
    uint32_t pvL1TileKRight;
    uint32_t qL1BufNum;
    uint32_t kL1BufNum;
    uint32_t vL1BufNum;
    uint32_t pL1BufNum;

    // Getter functions
    uint32_t get_numHeads() const { return numHeads; }
    uint32_t get_embeddingSize() const { return embeddingSize; }
    uint32_t get_embeddingSizeV() const { return embeddingSizeV; }
    uint32_t get_numBlocks() const { return numBlocks; }
    uint32_t get_blockSize() const { return blockSize; }
    uint32_t get_maxQSeqlen() const { return maxQSeqlen; }
    uint32_t get_maxKvSeqlen() const { return maxKvSeqlen; }
    uint32_t get_kvHeads() const { return kvHeads; }
    uint32_t get_batch() const { return batch; }
    uint32_t get_maxNumBlocksPerBatch() const { return maxNumBlocksPerBatch; }
    uint32_t get_firstBatchTaskNum() const { return firstBatchTaskNum; }
    uint32_t get_totalTaskNum() const { return totalTaskNum; }
    uint32_t get_maskType() const { return maskType; }
    uint64_t get_qkOutSize() const { return qkOutSize; }
    uint64_t get_smOnlineOutSize() const { return smOnlineOutSize; }
    uint64_t get_pvOutSize() const { return pvOutSize; }
    uint64_t get_UpdateSize() const { return UpdateSize; }
    uint64_t get_workSpaceSize() const { return workSpaceSize; }
    float get_scaleValue() const { return scaleValue; }
    uint64_t get_padding1() const { return padding1; }
    uint64_t get_padding2() const { return padding2; }
    uint32_t get_padding3() const { return padding3; }
    uint64_t get_splitLseTotalSize() const { return splitLseTotalSize; }
    uint64_t get_splitOTotalSize() const { return splitOTotalSize; }
    uint32_t get_totalSplitNodeNum() const { return totalSplitNodeNum; }
    uint32_t get_needCoreNum() const { return needCoreNum; }

    // Setter functions
    void set_numHeads(uint32_t value) { numHeads = value; }
    void set_embeddingSize(uint32_t value) { embeddingSize = value; }
    void set_embeddingSizeV(uint32_t value) { embeddingSizeV = value; }
    void set_numBlocks(uint32_t value) { numBlocks = value; }
    void set_blockSize(uint32_t value) { blockSize = value; }
    void set_maxQSeqlen(uint32_t value) { maxQSeqlen = value; }
    void set_maxKvSeqlen(uint32_t value) { maxKvSeqlen = value; }
    void set_kvHeads(uint32_t value) { kvHeads = value; }
    void set_batch(uint32_t value) { batch = value; }
    void set_maxNumBlocksPerBatch(uint32_t value) { maxNumBlocksPerBatch = value; }
    void set_firstBatchTaskNum(uint32_t value) { firstBatchTaskNum = value; }
    void set_totalTaskNum(uint32_t value) { totalTaskNum = value; }
    void set_maskType(uint32_t value) { maskType = value; }
    void set_qkOutSize(uint64_t value) { qkOutSize = value; }
    void set_smOnlineOutSize(uint64_t value) { smOnlineOutSize = value; }
    void set_pvOutSize(uint64_t value) { pvOutSize = value; }
    void set_UpdateSize(uint64_t value) { UpdateSize = value; }
    void set_workSpaceSize(uint64_t value) { workSpaceSize = value; }
    void set_scaleValue(float value) { scaleValue = value; }
    void set_padding1(uint64_t value) { padding1 = value; }
    void set_padding2(uint64_t value) { padding2 = value; }
    void set_padding3(uint32_t value) { padding3 = value; }
    void set_sparseMode(int32_t value) { sparseMode = value; }
    void set_globalWindowSize(uint32_t value) { globalWindowSize = value; }
    void set_localWindowSize(uint32_t value) { localWindowSize = value; }
    void set_cacheLayout(std::string value) { cacheLayout = value; }
    void set_preToken(int64_t value) { preToken = value; }
    void set_nextToken(int64_t value) { nextToken = value; }
    void set_splitLseTotalSize(uint64_t value) { splitLseTotalSize = value; }
    void set_splitOTotalSize(uint64_t value) { splitOTotalSize = value; }
    void set_totalSplitNodeNum(uint32_t value) { totalSplitNodeNum = value; }
    void set_needCoreNum(uint32_t value) { needCoreNum = value; }

    int64_t get_qSeqlenAligned() const { return qSeqlenAligned; }
    void set_qSeqlenAligned(int64_t value) { qSeqlenAligned = value;}
    int64_t get_kvSeqlenAligned() const { return kvSeqlenAligned; }
    void set_kvSeqlenAligned(int64_t value) { kvSeqlenAligned = value;}
    uint32_t get_innerPrec() const { return innerPrec; }
    void set_innerPrec(uint32_t value) { innerPrec = value;}
    uint32_t get_actSeqAval() const { return actSeqAval; }
    void set_actSeqAval(uint32_t value) { actSeqAval = value;}
    uint32_t get_qBaseTile() const { return qBaseTile; }
    uint32_t get_kvBaseTile() const { return kvBaseTile; }
    void set_qBaseTile(uint32_t value) { qBaseTile = value;}
    void set_kvBaseTile(uint32_t value) { kvBaseTile = value;}
    uint32_t get_qkL1TileM() const { return qkL1TileM; }
    uint32_t get_qkL1TileN() const { return qkL1TileN; }
    uint32_t get_qkL1TileKLeft() const { return qkL1TileKLeft; }
    uint32_t get_qkL1TileKRight() const { return qkL1TileKRight; }
    uint32_t get_pvL1TileM() const { return pvL1TileM; }
    uint32_t get_pvL1TileN() const { return pvL1TileN; }
    uint32_t get_pvL1TileKLeft() const { return pvL1TileKLeft; }
    uint32_t get_pvL1TileKRight() const { return pvL1TileKRight; }
    uint32_t get_qL1BufNum() const { return qL1BufNum; }
    uint32_t get_kL1BufNum() const { return kL1BufNum; }
    uint32_t get_vL1BufNum() const { return vL1BufNum; }
    uint32_t get_pL1BufNum() const { return pL1BufNum; }
    void set_qkL1TileM(uint32_t value) { qkL1TileM = value;}
    void set_qkL1TileN(uint32_t value) { qkL1TileN = value;}
    void set_qkL1TileKLeft(uint32_t value) { qkL1TileKLeft = value;}
    void set_qkL1TileKRight(uint32_t value) { qkL1TileKRight = value;}
    void set_pvL1TileM(uint32_t value) { pvL1TileM = value;}
    void set_pvL1TileN(uint32_t value) { pvL1TileN = value;}
    void set_pvL1TileKLeft(uint32_t value) { pvL1TileKLeft = value;}
    void set_pvL1TileKRight(uint32_t value) { pvL1TileKRight = value;}
    void set_qL1BufNum(uint32_t value) { qL1BufNum = value;}
    void set_kL1BufNum(uint32_t value) { kL1BufNum = value;}
    void set_vL1BufNum(uint32_t value) { vL1BufNum = value;}
    void set_pL1BufNum(uint32_t value) { pL1BufNum = value;}
};

#endif