/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef FLASH_ATTENTION_REGULAR_H
#define FLASH_ATTENTION_REGULAR_H

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
    uint64_t mm1OutSize;
    uint64_t smOnlineOutSize;
    uint64_t mm2OutSize;
    uint64_t UpdateSize;
    uint64_t workSpaceSize;
    float scaleValue;
    float softcapValue;
    uint64_t padding1;
    uint64_t padding2;
    uint32_t padding3;
    int64_t windowSizeLeft = 0;
    int64_t windowSizeRight = 0;
    uint64_t splitLseTotalSize;
    uint64_t splitOTotalSize;
    uint32_t totalSplitNodeNum;
    uint32_t needCoreNum;
    coreNode coreInfo[25];
    splitNode splitInfo[25];

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
    int64_t get_windowSizeLeft() const { return windowSizeLeft; }
    int64_t get_windowSizeRight() const { return windowSizeRight; }
    uint64_t get_mm1OutSize() const { return mm1OutSize; }
    uint64_t get_smOnlineOutSize() const { return smOnlineOutSize; }
    uint64_t get_mm2OutSize() const { return mm2OutSize; }
    uint64_t get_UpdateSize() const { return UpdateSize; }
    uint64_t get_workSpaceSize() const { return workSpaceSize; }
    float get_scaleValue() const { return scaleValue; }
    float get_softcapValue() const { return softcapValue; }
    uint64_t get_padding1() const { return padding1; }
    uint64_t get_padding2() const { return padding2; }
    uint32_t get_padding3() const { return padding3; }
    uint64_t get_splitLseTotalSize() const { return splitLseTotalSize; }
    uint64_t get_splitOTotalSize() const { return splitOTotalSize; }
    uint32_t get_totalSplitNodeNum() const { return totalSplitNodeNum; }
    uint32_t get_needCoreNum() const { return needCoreNum; }

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
    void set_windowSizeLeft(int64_t value) { windowSizeLeft = value; }
    void set_windowSizeRight(int64_t value) { windowSizeRight = value; }
    void set_mm1OutSize(uint64_t value) { mm1OutSize = value; }
    void set_smOnlineOutSize(uint64_t value) { smOnlineOutSize = value; }
    void set_mm2OutSize(uint64_t value) { mm2OutSize = value; }
    void set_UpdateSize(uint64_t value) { UpdateSize = value; }
    void set_workSpaceSize(uint64_t value) { workSpaceSize = value; }
    void set_scaleValue(float value) { scaleValue = value; }
    void set_softcapValue(float value) { softcapValue = value; }
    void set_padding1(uint64_t value) { padding1 = value; }
    void set_padding2(uint64_t value) { padding2 = value; }
    void set_padding3(uint32_t value) { padding3 = value; }
    void set_splitLseTotalSize(uint64_t value) { splitLseTotalSize = value; }
    void set_splitOTotalSize(uint64_t value) { splitOTotalSize = value; }
    void set_totalSplitNodeNum(uint32_t value) { totalSplitNodeNum = value; }
    void set_needCoreNum(uint32_t value) { needCoreNum = value; }
};

#endif