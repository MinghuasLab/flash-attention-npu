/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "fag_block.h"
#include "kernel_operator.h"
#include "fag_kernel_common.hpp"

namespace Catlass::Epilogue::Block {

template <
    class ElementVecDtype,
    class TilingData>
class BlockEpilogue<
    EpilogueAtlasA2FAGPre,
    ElementVecDtype,
    TilingData>
{
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    AscendC::TPipe *pipe;
    AscendC::GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;

    uint32_t cBlockIdx;
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
    __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *drop_mask, __gm__ uint8_t *workspace, __gm__ uint8_t * tiling_in)
    {
        cBlockIdx = AscendC::GetBlockIdx();
        pipe = pipe_in;
        (void)drop_mask;

        __gm__ TilingData *tilingData = reinterpret_cast<__gm__ TilingData *>(tiling_in);
        int64_t dqWorkSpaceOffset = tilingData->dqWorkSpaceOffset;
        int64_t dkWorkSpaceOffset = tilingData->dkWorkSpaceOffset;
        int64_t dvWorkSpaceOffset = tilingData->dvWorkSpaceOffset;
        int64_t qSize = tilingData->qSize;
        int64_t kvSize = tilingData->kvSize;
        uint32_t coreNum = tilingData->coreNum;

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