/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#include <acl/acl.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstring>

#include "softmax_tiling.cpp"
#include "kernel_common_fag.hpp"

namespace FAGTiling {
struct FAGInfo {
    float scaleValue;

    int64_t seqQShapeSize;
    int64_t queryShape_0;
    int64_t queryShape_1;
    int64_t queryShape_2;
    int64_t keyShape_0;
    int64_t keyShape_1;
    int64_t valueShape_0;
    int64_t valueShape_1;
};

int32_t GetFATilingParam(const FAGInfo fagInfo, uint32_t &blockDim, int64_t *tilingHost, uint64_t& workspaceSize)
{
    auto *fagV2TilingData = reinterpret_cast<FAGv2TilingData *>(tilingHost);
    std::memset(fagV2TilingData, 0, sizeof(FAGv2TilingData));

    uint64_t g = fagInfo.queryShape_1 / fagInfo.keyShape_1;

    int64_t qSize = fagInfo.queryShape_0 * fagInfo.keyShape_1 * g * fagInfo.queryShape_2;
    int64_t kvSize = fagInfo.keyShape_0 * fagInfo.keyShape_1 * 1 * fagInfo.queryShape_2;
    int64_t sfmgSize = fagInfo.queryShape_0 * fagInfo.queryShape_1 * 8;

    // Softmax tiling
    constexpr uint32_t tmpBufferSize = 33 * 1024;
    constexpr uint32_t s1VecSize = 64;
    constexpr uint32_t s2VecSize = 128;
    std::vector<uint32_t> softmaxShape = {s1VecSize, s2VecSize};

    SoftMaxTiling softmaxTilingData;
    SoftMaxTilingFunc(
        softmaxShape, sizeof(float), tmpBufferSize, softmaxTilingData);

    // softmaxGrad tiling
    constexpr uint32_t inputBufferLen = 24 * 1024;
    constexpr uint32_t castBufferLen = 48 * 1024; // castBuffer 48K*2=96K
    uint32_t outputBufferLen = (castBufferLen + fagInfo.queryShape_2 - 1) /  fagInfo.queryShape_2 * 8;
    uint32_t tempBufferLen = 40 * 1024 - outputBufferLen;

    int64_t singleLoopNBurstNum = inputBufferLen / sizeof(float) / fagInfo.queryShape_2;
    std::vector<int64_t> softmaxGradShape = {singleLoopNBurstNum, fagInfo.queryShape_2};

    SoftMaxTiling softmaxGradTilingData;
    SoftMaxGradTilingFunc(softmaxGradShape, sizeof(float), tempBufferLen, 
        softmaxGradTilingData);

    // put SoftMaxData in Tiling
    uint32_t coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    uint32_t vectorCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    fagV2TilingData->coreNum = vectorCoreNum;
    fagV2TilingData->scaleValue = fagInfo.scaleValue;
    fagV2TilingData->batch = fagInfo.seqQShapeSize;
    fagV2TilingData->t1 = fagInfo.queryShape_0;
    fagV2TilingData->t2 = fagInfo.keyShape_0;
    fagV2TilingData->kvHeadNum = fagInfo.keyShape_1;
    fagV2TilingData->g = g;
    fagV2TilingData->qkHeadDim = fagInfo.queryShape_2;
    fagV2TilingData->qSeqlen = fagV2TilingData->t1 / fagV2TilingData->batch;
    fagV2TilingData->qSize = qSize;
    fagV2TilingData->kvSize = kvSize;

    // TODO set workspace offset 
    constexpr size_t WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
    constexpr size_t GM_ALIGN = 512;
    constexpr size_t DB_NUM = 2;
    constexpr size_t matmulSize = 16 * 128 * 128;

    size_t workspaceOffset = WORKSPACE_RSV_BYTE;
    // matmal3 q
    fagV2TilingData->dqWorkSpaceOffset = workspaceOffset;
    workspaceOffset =
        (workspaceOffset + qSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 k
    fagV2TilingData->dkWorkSpaceOffset = workspaceOffset;
    workspaceOffset =
        (workspaceOffset + kvSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 v
    fagV2TilingData->dvWorkSpaceOffset = workspaceOffset;
    workspaceOffset =
        (workspaceOffset + kvSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // sfmg workspace
    fagV2TilingData->sfmgPreBeginAddr = workspaceOffset;
    workspaceOffset =
        (workspaceOffset + sfmgSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    // matmal1/matmal2 workspace size
    fagV2TilingData->mm1WorkSpaceOffset = workspaceOffset;
    workspaceOffset = 
        (workspaceOffset + coreNum * matmulSize * sizeof(float) * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    fagV2TilingData->mm2WorkSpaceOffset = workspaceOffset;
    workspaceOffset = 
        (workspaceOffset + coreNum * matmulSize * sizeof(float) * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    constexpr uint32_t size_of_half = 2;
    fagV2TilingData->pWorkSpaceOffset = workspaceOffset;
    workspaceOffset = 
        (workspaceOffset + coreNum * matmulSize * size_of_half * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    fagV2TilingData->dsWorkSpaceOffset = workspaceOffset;
    workspaceOffset = 
        (workspaceOffset + coreNum * matmulSize * size_of_half * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    workspaceSize = workspaceOffset;

    std::memcpy(&(fagV2TilingData->softmaxTilingData), &softmaxTilingData, sizeof(SoftMaxTiling));
    std::memcpy(&(fagV2TilingData->softmaxGradTilingData), &softmaxGradTilingData, sizeof(SoftMaxTiling));
    return 0;
}

} // namespace UnpadFATiling