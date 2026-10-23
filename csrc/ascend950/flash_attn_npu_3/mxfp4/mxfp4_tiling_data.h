/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* Modified by Minghua Shen, 2026. */

#ifndef MXFP4_TILING_DATA_H
#define MXFP4_TILING_DATA_H

#include <cstdint>

// Host and device share this POD. Field order is part of the launch ABI.
struct Mask2IdxTiling {
    uint32_t xBlockNumAligned;
    uint32_t yBlockNumAligned;
    uint32_t avgRowPerSubCore;
    uint32_t preActiveSubCoreNum;
};

struct BaseTiling {
    uint32_t qBaseTile;
    uint32_t kvBaseTile;
};

struct MmPhaseL1Tiling {
    uint32_t mm1L1TileM;
    uint32_t mm1L1TileN;
    uint32_t mm1L1TileKLeft;
    uint32_t mm1L1TileKRight;
    uint32_t mm2L1TileM;
    uint32_t mm2L1TileN;
    uint32_t mm2L1TileKLeft;
    uint32_t mm2L1TileKRight;
    uint32_t qL1BufNum;
    uint32_t kL1BufNum;
    uint32_t vL1BufNum;
    uint32_t pL1BufNum;
};

struct Mxfp4TilingData {
    uint32_t batch;
    uint32_t numHeads;
    uint32_t kvHeads;
    uint32_t embeddingSize;
    uint32_t blockSize;
    uint32_t maxNumBlocksPerBatch;
    uint32_t firstBatchTaskNum;
    uint32_t totalTaskNum;
    uint32_t coreTaskNum;
    uint32_t maskType;
    float scaleValue;
    uint32_t totalQBlocks;
    uint32_t firstQBlockNum;

    uint64_t blockShapeX;
    uint64_t blockShapeY;

    uint32_t maxKvBlockNum;
    uint32_t maxQBlockNum;
    uint32_t avgRowNumPerSubCore;
    uint32_t preActivateSubCoreNum;
    uint32_t queryLayout;
    uint32_t kvCacheLayout;
    uint32_t maxQSeqlen;
    uint32_t maxKvSeqlen;
    uint32_t useUniformQSeqlen;
    uint32_t useUniformKvSeqlen;

    uint64_t tilingKey;
    uint64_t selectNumIdxSize;
    uint64_t selectIdxSize;
    uint64_t mm1OutSize;
    uint64_t smOnlineOutSize;
    uint64_t mm2OutSize;
    uint64_t updateSize;
    uint64_t workSpaceSize;

    float log2Cx;
    float log2CxCeil;

    Mask2IdxTiling mask2IdxTileInfo;
    BaseTiling baseTileInfo;
    MmPhaseL1Tiling mmPhaseL1TileInfo;
};

#endif // MXFP4_TILING_DATA_H
