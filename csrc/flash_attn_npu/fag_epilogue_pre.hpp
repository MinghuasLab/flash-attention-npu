/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "fag_block.h"
#include "kernel_operator.h"
#include "fag_common/common_header.h"

namespace Catlass::Epilogue::Block {

template <
    typename ElementVecDtype
>
class BlockEpilogue<
    EpilogueAtlasA2FAGPre,
    ElementVecDtype>
{
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    AscendC::TPipe *pipe;
    AscendC::GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;

    uint32_t cBlockIdx;
    // query
    uint32_t qPreBlockFactor;
    uint32_t qPreBlockTotal;
    uint32_t qPreBlockTail;
    uint32_t kvPreBlockFactor;
    uint32_t kvPreBlockTotal;
    uint32_t kvPreBlockTail;

    int64_t initdqSize;
    int64_t dqOffset;
    int64_t initdkSize;
    int64_t dkvOffset;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, AscendC::TPipe *pipe_in, __gm__ uint8_t *dq, 
    __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *workspace, __gm__ uint8_t * tiling_in)
    {
        cBlockIdx = AscendC::GetBlockIdx();
        pipe = pipe_in;

        AscendC::GlobalTensor<uint64_t> tilingData;
        tilingData.SetGlobalBuffer((__gm__ uint64_t *)tiling_in);
        int64_t dqWorkSpaceOffset = tilingData.GetValue(TILING_DQ_WORKSPACE_OFFSET);
        int64_t dkWorkSpaceOffset = tilingData.GetValue(TILING_DK_WORKSPACE_OFFSET);
        int64_t dvWorkSpaceOffset = tilingData.GetValue(TILING_DV_WORKSPACE_OFFSET);
        int64_t qSize = tilingData.GetValue(TILING_Q_SIZE);
        int64_t kvSize = tilingData.GetValue(TILING_KV_SIZE);

        AscendC::GlobalTensor<uint32_t> tilingDataU32;
        tilingDataU32.SetGlobalBuffer((__gm__ uint32_t *)tiling_in);;
        uint32_t coreNum = tilingDataU32.GetValue(TILING_CORE_NUM * CONST_2);

        // compute tiling params
        qPreBlockFactor = (qSize + coreNum - 1) / coreNum;
        qPreBlockTotal = (qSize + qPreBlockFactor - 1) / qPreBlockFactor;
        int64_t qPreTailNumTmp = qSize % qPreBlockFactor;
        qPreBlockTail = qPreTailNumTmp == 0 ? qPreBlockFactor : qPreTailNumTmp;

        kvPreBlockFactor = (kvSize + coreNum - 1) / coreNum;
        kvPreBlockTotal = (kvSize + kvPreBlockFactor - 1) / kvPreBlockFactor;
        int64_t kvPreTailNumTmp = kvSize % kvPreBlockFactor;
        kvPreBlockTail = kvPreTailNumTmp == 0 ? kvPreBlockFactor : kvPreTailNumTmp;

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + dqWorkSpaceOffset / sizeof(float));
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + dkWorkSpaceOffset / sizeof(float));
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + dvWorkSpaceOffset / sizeof(float));

        initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
        dqOffset = ((int64_t)cBlockIdx) * qPreBlockFactor;
        initdkSize = cBlockIdx == kvPreBlockTotal - 1 ? kvPreBlockTail : kvPreBlockFactor;
        dkvOffset = ((int64_t)cBlockIdx) * kvPreBlockFactor;
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void operator()()
    {
        // process clear dq dk dv workspace
        if (g_coreType == AscendC::AIV && cBlockIdx < qPreBlockTotal) {
            AscendC::InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
        }

        if (g_coreType == AscendC::AIV && cBlockIdx < kvPreBlockTotal) {
            AscendC::InitOutput<float>(dkWorkSpaceGm[dkvOffset], initdkSize, 0);
            AscendC::InitOutput<float>(dvWorkSpaceGm[dkvOffset], initdkSize, 0);
        }
    }

};

}

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP
