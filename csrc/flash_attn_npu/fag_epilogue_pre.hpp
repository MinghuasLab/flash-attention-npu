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
#include "kernel_common_fag.hpp"

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
    AscendC::GlobalTensor<uint8_t> drop_maskGm, maskWorkSpaceGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> helpQue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inputQue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> castQue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQue;

    constexpr static uint32_t HELP_LEN = 256;
    constexpr static uint32_t BIT8 = 8;
    constexpr static uint32_t NUMBER_8 = 8;
    constexpr static uint32_t B16_VECTOR_MASK = 128;
    constexpr static uint32_t BOOL_BLOCK_NUMS = 32;
    constexpr static uint32_t SINGLE_UB_PROCESS_NUM = 30 * 1024;

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
    int64_t dropMaskSize;
    int64_t maskSingleCoreNum;
    uint32_t maskUsedCoreNum;
    uint32_t maskUBProcessNum;
    uint32_t maskTailUBProcessNum;
    uint32_t maskUBLoop;
    bool isDropBoolMode;

    AscendC::DataCopyParams copyParams;
    AscendC::BinaryRepeatParams repParams;
    half padValue{1.0};

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, AscendC::TPipe *pipe_in, __gm__ uint8_t *dq,
    __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *drop_mask, __gm__ uint8_t *workspace, __gm__ uint8_t * tiling_in)
    {
        cBlockIdx = AscendC::GetBlockIdx();
        pipe = pipe_in;
        isDropBoolMode = false;
        dropMaskSize = 0;
        maskSingleCoreNum = 0;
        maskUsedCoreNum = 0;
        maskUBProcessNum = 0;
        maskTailUBProcessNum = 0;
        maskUBLoop = 0;

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

        if constexpr (!std::is_same_v<TilingData, FAGv2TilingData>) {
            isDropBoolMode = drop_mask != nullptr && tilingData->dropoutIsDivisibleBy8 == 0;
            if (isDropBoolMode) {
                drop_maskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);
                maskWorkSpaceGm.SetGlobalBuffer((__gm__ uint8_t *)workspace + tilingData->dropBeginAddr);

                constexpr uint32_t inputBufferLen = 4 * 1024;
                constexpr uint32_t castBufferLen = 60 * 1024;
                constexpr uint32_t outputBufferLen = 30 * 1024;
                pipe->InitBuffer(helpQue, 1, HELP_LEN);
                pipe->InitBuffer(inputQue, 1, inputBufferLen);
                pipe->InitBuffer(castQue, 1, castBufferLen);
                pipe->InitBuffer(outQue, 1, outputBufferLen);

                dropMaskSize = tilingData->dropMaskSize;
                int64_t maskSize = (dropMaskSize + BOOL_BLOCK_NUMS - 1) / BOOL_BLOCK_NUMS * BOOL_BLOCK_NUMS;
                maskSingleCoreNum = ((maskSize + tilingData->blockOuter - 1) / tilingData->blockOuter
                                    + BOOL_BLOCK_NUMS - 1) / BOOL_BLOCK_NUMS * BOOL_BLOCK_NUMS;
                maskUsedCoreNum = tilingData->maskCoreNum;

                repParams.src0BlkStride = 1;
                repParams.src0RepStride = 0;
                repParams.src1BlkStride = 0;
                repParams.src1RepStride = 0;
                repParams.dstBlkStride = 1;
                repParams.dstRepStride = NUMBER_8;

                copyParams.blockCount = 1;
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
            }
        }
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

        if (g_coreType == AscendC::AIV && cBlockIdx < maskUsedCoreNum) {
            if (!isDropBoolMode) {
                return;
            }
            maskUBLoop = maskSingleCoreNum == 0 ? 0 :
                (maskSingleCoreNum + SINGLE_UB_PROCESS_NUM - 1) / SINGLE_UB_PROCESS_NUM;
            maskTailUBProcessNum = maskSingleCoreNum - (maskUBLoop - 1) * SINGLE_UB_PROCESS_NUM;
            if (unlikely(cBlockIdx == maskUsedCoreNum - 1)) {
                int64_t maskSize = (dropMaskSize + BOOL_BLOCK_NUMS - 1) / BOOL_BLOCK_NUMS * BOOL_BLOCK_NUMS;
                int64_t tailCoreNum = maskSize - (static_cast<int64_t>(maskUsedCoreNum) - 1) * maskSingleCoreNum;
                tailCoreNum = (tailCoreNum + BOOL_BLOCK_NUMS - 1) / BOOL_BLOCK_NUMS * BOOL_BLOCK_NUMS;
                maskUBLoop = (tailCoreNum + SINGLE_UB_PROCESS_NUM - 1) / SINGLE_UB_PROCESS_NUM;
                maskTailUBProcessNum = tailCoreNum - (maskUBLoop - 1) * SINGLE_UB_PROCESS_NUM;
            }

            auto helpTensor = helpQue.AllocTensor<half>();
            AscendC::Duplicate<half>(helpTensor, padValue, HELP_LEN / sizeof(half));
            AscendC::PipeBarrier<PIPE_V>();

            int64_t outputAddr = static_cast<int64_t>(cBlockIdx) * maskSingleCoreNum;
            int64_t inputAddr = outputAddr / BIT8;

            for (int64_t idx = 0; idx < maskUBLoop; idx++) {
                maskUBProcessNum = SINGLE_UB_PROCESS_NUM;
                int64_t outputOffset = idx * maskUBProcessNum;
                int64_t inputOffset = outputOffset / BIT8;
                if (unlikely(idx == maskUBLoop - 1)) {
                    maskUBProcessNum = maskTailUBProcessNum;
                }

                auto inputTensor = inputQue.AllocTensor<uint8_t>();
                copyParams.blockLen = maskUBProcessNum / BIT8;
                AscendC::DataCopyPad(inputTensor, drop_maskGm[inputAddr + inputOffset], copyParams, {false, 0, 0, 0});
                inputQue.EnQue(inputTensor);
                inputQue.DeQue<uint8_t>();

                auto castTensor = castQue.AllocTensor<half>();
                uint8_t selectRepeat = (maskUBProcessNum + B16_VECTOR_MASK - 1) / B16_VECTOR_MASK;
                AscendC::Select(castTensor, inputTensor, helpTensor, static_cast<half>(0.0),
                                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, B16_VECTOR_MASK, selectRepeat, repParams);
                AscendC::PipeBarrier<PIPE_V>();
                inputQue.FreeTensor(inputTensor);

                auto outputTensor = outQue.AllocTensor<uint8_t>();
                AscendC::Cast(outputTensor, castTensor, AscendC::RoundMode::CAST_ROUND, maskUBProcessNum);
                castQue.FreeTensor(castTensor);

                outQue.EnQue(outputTensor);
                outQue.DeQue<uint8_t>();
                copyParams.blockLen = maskUBProcessNum;
                AscendC::DataCopyPad(maskWorkSpaceGm[outputAddr + outputOffset], outputTensor, copyParams);
                outQue.FreeTensor(outputTensor);
            }
            helpQue.FreeTensor(helpTensor);
        }
    }
};

}

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP