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

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "fai_tilingdata.h"

namespace optiling{
    const uint32_t SIZE_OF_16BIT = 2;
    const uint32_t SIZE_OF_32BIT = 4;
    const uint32_t N_SPLIT_HELPER = 2;
    const uint32_t MAX_KV_STACK_LEN = 512;
    const uint32_t Q_TILE_CEIL = 128;
    const uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
    const uint32_t BASE_KV_SIZE = 128;
    const uint32_t PRELANCH_NUM = 3;

    enum class MaskType : uint32_t {
        NO_MASK = 0,
        MASK_SPEC = 1
    };

    enum class DataType : uint32_t {
        FP16 = 0,
        BF16 = 1
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
        int32_t sparseMode = 0;
        std::string cacheLayout = "nd";
        uint32_t maxNumBlocksPerBatch = 0;
        const int32_t *qSeqlenList{nullptr};
        const int32_t *kvSeqlenList{nullptr};
        float scaleValue = 0.0;
        size_t* workspaces{nullptr};
        MaskType maskType = MaskType::MASK_SPEC;
        DataType dataType = DataType::FP16;
        bool pagedCacheFlag = true;
        bool lseFlag = false;
        bool isTilingSink = false;
        bool learnableSinkFlag = false;
        bool flashDecodeFlag = false;
        bool kvcacheNzFlag = false;
        std::string layout;
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
            explicit FAInferTiling(const FAInferContext &faInfo);
            void DoTiling(FAInferTilingData &tilingdata);
            void SetCoreNum(uint32_t blockNum) {
                this->blockNum_ = blockNum;
            }
            uint32_t GetCoreNum() {
                return this->blockNum_; 
            }
            uint64_t GetTilingKey();
        private:
            void FillSplitCoreTilingData(FAInferTilingData &tilingdata);
            void FillWorkSpaceTilingData(FAInferTilingData &faTilingData);
            uint32_t GetQSBlockTile(int64_t kvSeqlen);
            uint32_t GetKSBlockTile(int64_t kvSeqlen);
            uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize);
            void FillBasicTilingData(FAInferTilingData &faTilingData);
            BatchParams getBatchParams(uint32_t bIdx, uint32_t groupSize);
        private:
            FAInferContext faInfo_;
            uint32_t blockNum_;
    };

    FAInferTiling::FAInferTiling(const FAInferContext &faInfo): faInfo_(faInfo) {}

    uint32_t FAInferTiling::GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
    {
        uint32_t qRowNumCeil = Q_TILE_CEIL;
        uint32_t qNBlockTile = (qSeqlen != 0) ?
            (qRowNumCeil / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
        qNBlockTile = std::min(qNBlockTile, groupSize);
        qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
        return qNBlockTile;
    }

    uint32_t FAInferTiling::GetQSBlockTile(int64_t kvSeqlen)
    {
        uint32_t qSBlockTile = Q_TILE_CEIL;
        return qSBlockTile;
    }
    uint32_t FAInferTiling::GetKSBlockTile(int64_t kvSeqlen)
    {
        uint32_t kSBlockTile = 128;
        return kSBlockTile;
    }

    void FAInferTiling::FillBasicTilingData(FAInferTilingData &faTilingData)
    {
        faTilingData.set_batch(static_cast<uint32_t>(faInfo_.batch));
        faTilingData.set_numHeads(static_cast<uint32_t>(faInfo_.numHeads));
        faTilingData.set_kvHeads(static_cast<uint32_t>(faInfo_.kvHeads));

        faTilingData.set_embeddingSize(static_cast<uint32_t>(faInfo_.embeddingSize));
        faTilingData.set_embeddingSizeV(static_cast<uint32_t>(faInfo_.embeddingSizeV));
        faTilingData.set_numBlocks(static_cast<uint32_t>(faInfo_.numBlocks));
        if (faInfo_.pagedCacheFlag) {
            faTilingData.set_blockSize(static_cast<uint32_t>(faInfo_.blockSize));
        } else {
            faTilingData.set_blockSize(BASE_KV_SIZE);
        }
        faTilingData.set_maxQSeqlen(faInfo_.maxQSeqlen);
        faTilingData.set_maxKvSeqlen(faInfo_.maxKvSeqlen);
        faTilingData.set_maxNumBlocksPerBatch(faInfo_.maxNumBlocksPerBatch);
        faTilingData.set_maskType(static_cast<uint32_t>(faInfo_.maskType));
        faTilingData.set_scaleValue(faInfo_.scaleValue);
        faTilingData.set_sparseMode(faInfo_.sparseMode);
        faTilingData.set_cacheLayout(faInfo_.cacheLayout);
        faTilingData.set_preToken(static_cast<int64_t>(faInfo_.preToken));
        faTilingData.set_nextToken(static_cast<int64_t>(faInfo_.nextToken));
        
        auto qkL1TileM_ = 128;
        auto qkL1TileKLeft_ = 192;
        auto qL1BufNum_ = 1;
        // K矩阵开启2buf，D按128分割，S2按256分割
        auto qkL1TileN_ = 256;
        auto qkL1TileKRight_ = 192;
        auto kL1BufNum_ = 2;
        // V矩阵开启db，D按128分割，kvBaseTile_不分割，指令同样提前于核间同步下发
        // 如果kvBaseTile_进一步增大，考虑关闭db，使得kvBaseTile_不分割
        auto pvL1TileN_ = 128;
        auto pvL1TileKLeft_ = 256;
        auto vL1BufNum_ = 2;
        // P矩阵在950上会常驻L1，由于基块的prelaunch为2，因此最好有3 buf，以免基块间流水阻塞
        auto pvL1TileM_ = 128;
        auto pvL1TileKRight_ = 128;
        auto pL1BufNum_ = 3;
        faTilingData.set_innerPrec(0);
        faTilingData.set_actSeqAval(0);
        faTilingData.set_qBaseTile(128);
        faTilingData.set_kvBaseTile(128);
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

    uint64_t FAInferTiling::GetTilingKey() 
    {
        constexpr uint64_t SPLIT_FUSE_BASE_KEY = 5000000000000000000;
        constexpr uint64_t PAGED_CACHE_KEY = 10000000;
        constexpr uint64_t COMP_CAUSAL_MASK_KEY = 3;
         constexpr uint64_t COMP_SWA_MASK_KEY = 5;
        constexpr uint64_t LAYOUTQ_BSND_KEY = 100000;
        constexpr uint64_t LAYOUTQ_TND_KEY = 200000;
        constexpr uint64_t DTYPE_FP16_KEY = 100;
        constexpr uint64_t KVCACHE_NZ_KEY = 10;
        constexpr uint64_t DTYPE_BF16_KEY = 200;
        constexpr uint64_t LSE_OUT_ONLY_KEY = 1000;
        constexpr uint64_t INNER_LOW_PREC_KEY = 10000;
        constexpr uint64_t LEARNABLE_SINK_KEY = 100000000;
        constexpr uint64_t FLASH_DECODE_KEY = 100000000000000000;
        uint64_t tilingKey = SPLIT_FUSE_BASE_KEY;
        std::cout << "faInfo_.pagedCacheFlag:" << faInfo_.pagedCacheFlag << std::endl;
        std::cout << "faInfo_.maskType:" << (uint32_t)faInfo_.maskType << std::endl;
        std::cout << "faInfo_.layout:" << faInfo_.layout << std::endl;
        if (faInfo_.pagedCacheFlag) {
            tilingKey += static_cast<uint64_t>(PAGED_CACHE_KEY);
        }
        if (faInfo_.pagedShapeFlag) { // 20000000 表示 BnNBsD
            tilingKey += static_cast<uint64_t>(PAGED_CACHE_KEY);
        }
        if (faInfo_.maskType == MaskType::MASK_SPEC) {
            tilingKey += static_cast<uint64_t>(COMP_CAUSAL_MASK_KEY);
        }
        if (faInfo_.layout == "TND") {
            tilingKey += static_cast<uint64_t>(LAYOUTQ_TND_KEY);
        } else if (faInfo_.layout == "BSND") {
            tilingKey += static_cast<uint64_t>(LAYOUTQ_BSND_KEY);
        }
        if (faInfo_.dataType == DataType::FP16) {
            std::cout << "faInfo_.dataType:" << "fp16" << std::endl;
            tilingKey += static_cast<uint64_t>(DTYPE_FP16_KEY);
        } else if (faInfo_.dataType == DataType::BF16) {
            std::cout << "faInfo_.dataType:" << "bf16" << std::endl;
            tilingKey += static_cast<uint64_t>(DTYPE_BF16_KEY);
        }
        if (faInfo_.lseFlag) {
            tilingKey += static_cast<uint64_t>(LSE_OUT_ONLY_KEY);
        }
        if (faInfo_.learnableSinkFlag) {
            tilingKey += static_cast<uint64_t>(LEARNABLE_SINK_KEY);
        }
        if (faInfo_.innerPrecise == 1) {
            tilingKey += static_cast<uint64_t>(INNER_LOW_PREC_KEY);
        }
        return tilingKey;
    }

    void FAInferTiling::FillWorkSpaceTilingData(FAInferTilingData &faTilingData)
    {
        uint64_t qkOutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_32BIT * PRELANCH_NUM;
        uint64_t smOnlineOutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_16BIT * PRELANCH_NUM;
        uint64_t pvOutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_32BIT * PRELANCH_NUM;
        uint64_t UpdateSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_32BIT * PRELANCH_NUM;
        
        uint64_t splitLseTotalSize = 0;
        uint64_t splitOTotalSize = 0;
        if (faInfo_.isTilingSink) {
            splitLseTotalSize = 2 * static_cast<uint64_t>(blockNum_) * Q_TILE_CEIL * SIZE_OF_32BIT * faInfo_.numHeads;
            uint32_t embeddingSizeV = static_cast<uint32_t>(faInfo_.embeddingSizeV);
            splitOTotalSize = 2 * static_cast<uint64_t>(blockNum_) * Q_TILE_CEIL * embeddingSizeV * SIZE_OF_32BIT * faInfo_.numHeads;
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

    void FAInferTiling::FillSplitCoreTilingData(FAInferTilingData &faTilingData)
    {
        uint32_t totalTaskNum = 0;
        uint32_t groupSize = faInfo_.numHeads / faInfo_.kvHeads;
        for (int32_t batchIdx = 0; batchIdx < faInfo_.batch; batchIdx++) {
            uint32_t qSeqlen = *(faInfo_.qSeqlenList + batchIdx);
            uint32_t kvSeqlen = *(faInfo_.kvSeqlenList + batchIdx);
            if (faInfo_.layout == "TND") {
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
            uint32_t curTaskNum = faInfo_.numHeads * curQSBlockNum;
            if (batchIdx == 0) {
                faTilingData.set_firstBatchTaskNum(curTaskNum);
            }
            totalTaskNum += curTaskNum;
        }
        faTilingData.set_totalTaskNum(totalTaskNum);
    }
    BatchParams FAInferTiling::getBatchParams(uint32_t bIdx, uint32_t groupSize) {
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

    void FAInferTiling::DoTiling(FAInferTilingData &tilingdata)
    {
        FillBasicTilingData(tilingdata);
        if (!faInfo_.isTilingSink) {
            FillSplitCoreTilingData(tilingdata);
        }
        FillWorkSpaceTilingData(tilingdata);
    }
}