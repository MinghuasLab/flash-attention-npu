/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_POST_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_POST_HPP

#include <type_traits>

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "fag_block.h"
#include "kernel_operator.h"
#include "fag_kernel_common.hpp"

using AscendC::CopyRepeatParams;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::TBuf;
using AscendC::TQue;

namespace Catlass::Epilogue::Block {

template <
    class ElementVecDtype,
    class TilingData>
class BlockEpilogue<
    EpilogueAtlasA2FAGPost,
    ElementVecDtype,
    TilingData>
{
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPost;
    using ArchTag = typename DispatchPolicy::ArchTag;

    constexpr static uint32_t POST_BUFFER_NUM = 1;

    AscendC::TPipe *pipe;
    TBuf<QuePosition::VECIN> inBuffer;
    TBuf<QuePosition::VECOUT> outBuffer;

    // input
    AscendC::GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    // output
    AscendC::GlobalTensor<ElementVecDtype> dqGm, dkGm, dvGm;

    int64_t cBlockIdx;
    int64_t ubBaseSize;
    int64_t qPostBlockFactor;
    uint64_t qPostBlockTotal;
    int64_t qPostBaseNum;
    int64_t qPostTailNum;
    int64_t kvPostBlockFactor;
    uint64_t kvPostBlockTotal;
    int64_t kvPostBaseNum;
    int64_t kvPostTailNum;
    float scaleValue;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, AscendC::TPipe *pipe_in, __gm__ uint8_t *dq,
    __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *workspace, __gm__ uint8_t * tiling_in)
    {
        cBlockIdx = GetBlockIdx();
        pipe = pipe_in;

        __gm__ TilingData *tilingData = reinterpret_cast<__gm__ TilingData *>(tiling_in);
        int64_t dqWorkSpaceOffset = tilingData->dqWorkSpaceOffset;
        int64_t dkWorkSpaceOffset = tilingData->dkWorkSpaceOffset;
        int64_t dvWorkSpaceOffset = tilingData->dvWorkSpaceOffset;
        int64_t qSize = tilingData->qSize;
        int64_t kvSize = tilingData->kvSize;
        uint32_t coreNum = tilingData->coreNum;
        scaleValue = tilingData->scaleValue;

        if constexpr (std::is_same_v<TilingData, FAGTilingData>) {
            qSeqlen = tilingData->qSeqlen;
            kvSeqlen = tilingData->kvSeqlen;
            qHeadNum = tilingData->kvHeadNum * tilingData->g;
            kvHeadNum = tilingData->kvHeadNum;
            qHeadDim = tilingData->qkHeadDim;
            vHeadDim = tilingData->vHeadDim;
            dqStrides.batch = tilingData->dqStrides.batch;
            dqStrides.seq = tilingData->dqStrides.seq;
            dqStrides.head = tilingData->dqStrides.head;
            dkStrides.batch = tilingData->dkStrides.batch;
            dkStrides.seq = tilingData->dkStrides.seq;
            dkStrides.head = tilingData->dkStrides.head;
            dvStrides.batch = tilingData->dvStrides.batch;
            dvStrides.seq = tilingData->dvStrides.seq;
            dvStrides.head = tilingData->dvStrides.head;
            dqIsStrided = dqStrides.seq != 0 &&
                (dqStrides.seq != qHeadNum * qHeadDim || dqStrides.head != qHeadDim ||
                 dqStrides.batch != qSeqlen * qHeadNum * qHeadDim);
            dkIsStrided = dkStrides.seq != 0 &&
                (dkStrides.seq != kvHeadNum * qHeadDim || dkStrides.head != qHeadDim ||
                 dkStrides.batch != kvSeqlen * kvHeadNum * qHeadDim);
            dvIsStrided = dvStrides.seq != 0 &&
                (dvStrides.seq != kvHeadNum * vHeadDim || dvStrides.head != vHeadDim ||
                 dvStrides.batch != kvSeqlen * kvHeadNum * vHeadDim);
        }


        dqGm.SetGlobalBuffer((__gm__ ElementVecDtype *)dq);
        dkGm.SetGlobalBuffer((__gm__ ElementVecDtype *)dk);
        dvGm.SetGlobalBuffer((__gm__ ElementVecDtype *)dv);

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->dqWorkSpaceOffset / sizeof(float));
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->dkWorkSpaceOffset / sizeof(float));
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->dvWorkSpaceOffset / sizeof(float));

        // compute tiling
        constexpr static uint32_t POST_COEX_NODE = 3;
        constexpr static uint32_t WORKSPACE_NUM_ALIGN = 256;
        uint32_t curPostCoexNode =  POST_COEX_NODE;
        uint32_t ubSize = ArchTag::UB_SIZE;
        ubBaseSize = ubSize / curPostCoexNode / POST_BUFFER_NUM;
        ubBaseSize = ubBaseSize / WORKSPACE_NUM_ALIGN * WORKSPACE_NUM_ALIGN;

        // dq
        qPostBaseNum = ubBaseSize / sizeof(float);
        if (dqIsStrided) {
            qPostBaseNum = qPostBaseNum / qHeadDim * qHeadDim;
        }
        qPostBlockTotal = qSize;

        int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
        int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;

        qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
        qPostBlockFactor = (qPostBlockOuterTotal + coreNum - 1) / coreNum;

        // dkv
        kvPostBaseNum = qPostBaseNum;
        if ((dkIsStrided || dvIsStrided) && !dqIsStrided) {
            kvPostBaseNum = kvPostBaseNum / qHeadDim * qHeadDim;
        }
        kvPostBlockTotal = kvSize;

        int64_t kvPostTailNumTmp = kvPostBlockTotal % kvPostBaseNum;
        int64_t kvPostBlockOuterTotal = (kvPostBlockTotal + kvPostBaseNum - 1) / kvPostBaseNum;

        kvPostTailNum = kvPostTailNumTmp == 0 ? kvPostBaseNum : kvPostTailNumTmp;
        kvPostBlockFactor = (kvPostBlockOuterTotal + coreNum - 1) / coreNum;

        pipe->InitBuffer(inBuffer, ubBaseSize * 2);
        pipe->InitBuffer(outBuffer, ubBaseSize);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void CopyOutBsnd(AscendC::GlobalTensor<ElementVecDtype> dst,
                     AscendC::LocalTensor<ElementVecDtype> src,
                     uint64_t logicalOffset, uint64_t dataSize,
                     FAGTensorStrides strides, int64_t seqlen,
                     int64_t nheads, int64_t headdim)
    {
        uint64_t flatRow = logicalOffset / headdim;
        uint64_t remainingRows = dataSize / headdim;
        uint64_t ubRow = 0;
        while (remainingRows > 0) {
            int64_t nIdx = flatRow % nheads;
            uint64_t bsIdx = flatRow / nheads;
            int64_t sIdx = bsIdx % seqlen;
            int64_t bIdx = bsIdx / seqlen;
            uint64_t rows = remainingRows < static_cast<uint64_t>(nheads - nIdx) ?
                                remainingRows : static_cast<uint64_t>(nheads - nIdx);
            int64_t dstOffset = bIdx * strides.batch + sIdx * strides.seq + nIdx * strides.head;
            DataCopyPad(dst[dstOffset], src[ubRow * headdim],
                DataCopyExtParams(static_cast<uint16_t>(rows),
                    static_cast<uint32_t>(headdim * sizeof(ElementVecDtype)), 0,
                    static_cast<uint32_t>((strides.head - headdim) * sizeof(ElementVecDtype)), 0));
            flatRow += rows;
            ubRow += rows;
            remainingRows -= rows;
        }
    }

    CATLASS_DEVICE
    void operator()()
    {
        uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
        uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;

        if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
            qEnd = qPostBlockTotal;
        }
        event_t Mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        for (uint64_t i = qBegin; i < qEnd; i = i + qPostBaseNum) {

            AscendC::LocalTensor<float> vecIn = inBuffer.Get<float>();
            AscendC::LocalTensor<ElementVecDtype> vecOut = outBuffer.Get<ElementVecDtype>();
            uint64_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
            DataCopy(vecIn, dqWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B

            event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            Muls(vecIn, vecIn, scaleValue, dataSize);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            event_t Mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);

            if (dqIsStrided) {
                CopyOutBsnd(dqGm, vecOut, i, dataSize, dqStrides, qSeqlen, qHeadNum, qHeadDim);
            } else {
                DataCopy(dqGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        uint64_t kvBegin = cBlockIdx * kvPostBlockFactor * kvPostBaseNum;
        uint64_t kvEnd = (cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum;
        if (((cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum) > kvPostBlockTotal) {
            kvEnd = kvPostBlockTotal;
        }

        for (uint64_t i = kvBegin; i < kvEnd; i = i + kvPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inBuffer.Get<float>();
            AscendC::LocalTensor<ElementVecDtype> vecOut = outBuffer.Get<ElementVecDtype>();
            uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            DataCopy(vecIn, dkWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);

            Muls(vecIn, vecIn, scaleValue, dataSize);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);

            event_t Mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);

            if (dkIsStrided) {
                CopyOutBsnd(dkGm, vecOut, i, dataSize, dkStrides, kvSeqlen, kvHeadNum, qHeadDim);
            } else {
                DataCopy(dkGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint64_t i = kvBegin; i < kvEnd; i = i + kvPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inBuffer.Get<float>();
            AscendC::LocalTensor<ElementVecDtype> vecOut = outBuffer.Get<ElementVecDtype>();
            uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            DataCopy(vecIn, dvWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);

            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            event_t Mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);

            if (dvIsStrided) {
                CopyOutBsnd(dvGm, vecOut, i, dataSize, dvStrides, kvSeqlen, kvHeadNum, vHeadDim);
            } else {
                DataCopy(dvGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            }
            if (i + kvPostBaseNum < kvEnd) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
            }
        }
    }

private:
    FAGTensorStrides dqStrides;
    FAGTensorStrides dkStrides;
    FAGTensorStrides dvStrides;
    int64_t qSeqlen{0};
    int64_t kvSeqlen{0};
    int64_t qHeadNum{0};
    int64_t kvHeadNum{0};
    int64_t qHeadDim{0};
    int64_t vHeadDim{0};
    bool dqIsStrided{false};
    bool dkIsStrided{false};
    bool dvIsStrided{false};
};

}

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_POST_HPP
