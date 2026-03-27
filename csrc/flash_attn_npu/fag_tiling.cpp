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
#include "fag_common/common_header.h"

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

void printFAGTilingData(int64_t *tilingHost) {
    std::cout << "FAGTilingDATA batch: " << tilingHost[TILING_B] << std::endl;
    std::cout << "FAGTilingDATA: total_q: " << tilingHost[TILING_T1] << std::endl;
    std::cout << "FAGTilingDATA: total_k: " << tilingHost[TILING_T2] << std::endl;
    std::cout << "FAGTilingDATA: nheads: " << tilingHost[TILING_N1] << std::endl;
    std::cout << "FAGTilingDATA: nheads_k: " << tilingHost[TILING_N2] << std::endl;
    std::cout << "FAGTilingDATA: G: " << tilingHost[TILING_G] << std::endl;
    std::cout << "FAGTilingDATA: headdim: " << tilingHost[TILING_D] << std::endl;
    std::cout << "FAGTilingDATA: q size: " << tilingHost[TILING_Q_SIZE] << std::endl;
    std::cout << "FAGTilingDATA: kv size: " << tilingHost[TILING_KV_SIZE] << std::endl;
    std::cout << "FAGTilingDATA: dq_workspace_offset: " << tilingHost[TILING_DQ_WORKSPACE_OFFSET] << std::endl;
    std::cout << "FAGTilingDATA: dk_workspace_offset: " << tilingHost[TILING_DK_WORKSPACE_OFFSET] << std::endl;
    std::cout << "FAGTilingDATA: dv_workspace_offset: " << tilingHost[TILING_DV_WORKSPACE_OFFSET] << std::endl;
    std::cout << "FAGTilingDATA: sfmg_workspace_offset: " << tilingHost[TILING_SFMG_WORKSPACE_OFFSET] << std::endl;
    std::cout << "FAGTilingDATA: mm1_workspace_offset: " << tilingHost[TILING_MM1_WORKSPACE_OFFSET] << std::endl;
    std::cout << "FAGTilingDATA: mm2_workspace_offset: " << tilingHost[TILING_MM2_WORKSPACE_OFFSET] << std::endl;
    std::cout << "FAGTilingDATA: p_workspace_offset: " << tilingHost[TILING_P_WORKSPACE_OFFSET] << std::endl;
    std::cout << "FAGTilingDATA: ds_workspace_offset: " << tilingHost[TILING_DS_WORKSPACE_OFFSET] << std::endl;

    float *tilingHostFp = reinterpret_cast<float *>(tilingHost);
    std::cout << "FAGTilingDATA scale value: " << tilingHostFp[TILING_SCALE_VALUE * CONST_2] << std::endl;
    
    uint32_t *tilingHostU32 = reinterpret_cast<uint32_t *>(tilingHost);
    std::cout << "FAGTilingDATA coreNum: " << tilingHostU32[TILING_CORE_NUM * CONST_2] << std::endl;
    std::cout << "FAGTilingDATA srcM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2] << std::endl;
    std::cout << "FAGTilingDATA srcK: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 1] << std::endl;
    std::cout << "FAGTilingDATA srcSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 2] << std::endl;
    std::cout << "FAGTilingDATA outMaxM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 3] << std::endl;
    std::cout << "FAGTilingDATA outMaxK: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 4] << std::endl;
    std::cout << "FAGTilingDATA outMaxSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 5] << std::endl;
    std::cout << "FAGTilingDATA splitM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 6] << std::endl;
    std::cout << "FAGTilingDATA splitK: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 7] << std::endl;
    std::cout << "FAGTilingDATA SplitSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 8] << std::endl;
    std::cout << "FAGTilingDATA reduceM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 9] << std::endl;
    std::cout << "FAGTilingDATA reduceK: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 10] << std::endl;
    std::cout << "FAGTilingDATA reduceSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 11] << std::endl;
    std::cout << "FAGTilingDATA rangeM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 12] << std::endl;
    std::cout << "FAGTilingDATA tailM: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 13] << std::endl;
    std::cout << "FAGTilingDATA tailSplitSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 14] << std::endl;
    std::cout << "FAGTilingDATA tailReduceSize: " << tilingHostU32[TILING_SOFTMAX_TILING_DATA * CONST_2 + 15] << std::endl;

    // softmax grad data
    std::cout << "FAGTilingDATA srcM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2] << std::endl;
    std::cout << "FAGTilingDATA srcK: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 1] << std::endl;
    std::cout << "FAGTilingDATA srcSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 2] << std::endl;
    std::cout << "FAGTilingDATA outMaxM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 3] << std::endl;
    std::cout << "FAGTilingDATA outMaxK: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 4] << std::endl;
    std::cout << "FAGTilingDATA outMaxSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 5] << std::endl;
    std::cout << "FAGTilingDATA splitM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 6] << std::endl;
    std::cout << "FAGTilingDATA splitK: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 7] << std::endl;
    std::cout << "FAGTilingDATA SplitSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 8] << std::endl;
    std::cout << "FAGTilingDATA reduceM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 9] << std::endl;
    std::cout << "FAGTilingDATA reduceK: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 10] << std::endl;
    std::cout << "FAGTilingDATA reduceSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 11] << std::endl;
    std::cout << "FAGTilingDATA rangeM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 12] << std::endl;
    std::cout << "FAGTilingDATA tailM: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 13] << std::endl;
    std::cout << "FAGTilingDATA tailSplitSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 14] << std::endl;
    std::cout << "FAGTilingDATA tailReduceSize: " << tilingHostU32[TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 15] << std::endl;
}

int32_t GetFATilingParam(const FAGInfo fagInfo, uint32_t &blockDim, int64_t *tilingHost)
{
    float *tilingHostFp = reinterpret_cast<float *>(tilingHost);
    tilingHostFp[TILING_SCALE_VALUE * CONST_2] = fagInfo.scaleValue;

    tilingHost[TILING_B] = fagInfo.seqQShapeSize;
    tilingHost[TILING_T1] = fagInfo.queryShape_0;
    tilingHost[TILING_T2] = fagInfo.keyShape_0;
    tilingHost[TILING_N1] = fagInfo.queryShape_1;
    tilingHost[TILING_N2] = fagInfo.keyShape_1;
    tilingHost[TILING_D] = fagInfo.queryShape_2;

    uint64_t g = fagInfo.queryShape_1 / fagInfo.keyShape_1;
    tilingHost[TILING_G] = fagInfo.queryShape_1 / fagInfo.keyShape_1;

    int64_t qSize = fagInfo.queryShape_0 * fagInfo.keyShape_1 * g * fagInfo.queryShape_2;
    int64_t kvSize = fagInfo.keyShape_0 * fagInfo.keyShape_1 * 1 * fagInfo.queryShape_2;
    int64_t sfmgSize = fagInfo.queryShape_0 * fagInfo.queryShape_1 * 8;

    tilingHost[TILING_Q_SIZE] = qSize;
    tilingHost[TILING_KV_SIZE] = kvSize;

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
    uint32_t *tilingHostU32 = reinterpret_cast<uint32_t *>(tilingHost);
    tilingHostU32[TILING_CORE_NUM * CONST_2] = vectorCoreNum;

    uint32_t* softmaxTilingDataPtr = reinterpret_cast<uint32_t *>(&softmaxTilingData);
    memcpy(tilingHostU32 + TILING_SOFTMAX_TILING_DATA * CONST_2, softmaxTilingDataPtr, TILING_SOFTMAX_SIZE);
    softmaxTilingDataPtr = reinterpret_cast<uint32_t *>(&softmaxGradTilingData);
    memcpy(tilingHostU32 + TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2, softmaxTilingDataPtr, TILING_SOFTMAX_SIZE);

    // TODO set workspace offset 
    constexpr size_t WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
    constexpr size_t GM_ALIGN = 512;
    constexpr size_t DB_NUM = 2;
    constexpr size_t matmulSize = 16 * 128 * 128;

    size_t workspaceOffset = WORKSPACE_RSV_BYTE;
    // matmal3 q
    tilingHost[TILING_DQ_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset =
        (workspaceOffset + qSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 k
    tilingHost[TILING_DK_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset =
        (workspaceOffset + kvSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 v
    tilingHost[TILING_DV_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset =
        (workspaceOffset + kvSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // sfmg workspace
    tilingHost[TILING_SFMG_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset =
        (workspaceOffset + sfmgSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    // matmal1/matmal2 workspace size
    tilingHost[TILING_MM1_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset = 
        (workspaceOffset + coreNum * matmulSize * sizeof(float) * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tilingHost[TILING_MM2_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset = 
        (workspaceOffset + coreNum * matmulSize * sizeof(float) * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    constexpr uint32_t size_of_half = 2;
    tilingHost[TILING_P_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset = 
        (workspaceOffset + coreNum * matmulSize * size_of_half * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tilingHost[TILING_DS_WORKSPACE_OFFSET] = workspaceOffset;
    workspaceOffset = 
        (workspaceOffset + coreNum * matmulSize * size_of_half * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    return 0;
}

} // namespace UnpadFATiling