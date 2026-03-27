 /**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_OP_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_OP_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "fag_block.h"
#include "kernel_operator.h"
#include "fag_common/common_header.h"
#include "fag_common/vector_addr.h"

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
    typename ElementVecDtype,
    InputLayout inputLayout
>
class BlockEpilogue<
    EpilogueAtlasA2FAGOp,
    ElementVecDtype,
    std::integral_constant<InputLayout, inputLayout>>
{
public:
    using DispatchPolicy = EpilogueAtlasA2FAGOp;
    using ArchTag = typename DispatchPolicy::ArchTag;

    static constexpr InputLayout getLayout() {
        return std::integral_constant<InputLayout, inputLayout>::value;
    }
    AscendC::TPipe *pipe;
    TBuf<> unifiedBuffer;

    GlobalTensor<uint8_t> attenMaskU8Gm;
    GlobalTensor<float> mm1WorkspaceGm;
    GlobalTensor<float> mm2WorkspaceGm;
    GlobalTensor<ElementVecDtype> dropWorkSpaceGm, mulWorkSpaceGm;
    GlobalTensor<float> rowLseGm;
    GlobalTensor<float> sfmgWorkspaceGm;

    constexpr static uint32_t DTYPE_FACTOR = sizeof(float) / sizeof(ElementVecDtype);
    constexpr static uint32_t cal_block_num = 32 / sizeof(float);
    constexpr static uint32_t cal_repeat_num = 256 / sizeof(float);
    constexpr static uint32_t input_block_num = 32 / sizeof(ElementVecDtype);
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t INPUT_NUMS = 2;
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static int64_t C0_SIZE = 16;
    constexpr static int64_t VEC_REPEAT = 8;

    constexpr static uint32_t T2Begin = 0;
    constexpr static uint32_t T1Begin = 33 * 1024;
    constexpr static uint32_t BoolBegin = 50 * 1024;
    constexpr static uint32_t U8Begin = 58 * 1024;
    constexpr static uint32_t T2BlockBegin = 66 * 1024;

    constexpr static uint32_t DbBegin = 74 * 1024;
    constexpr static int64_t TMP_UB_OFFSET = 148 * 1024;
    constexpr static int64_t SFMG_UB_OFFSET = (148 + 33) * 1024;
    constexpr static int64_t TMP_UB_SIZE = 33 * 1024;
    constexpr static int64_t SFMG_UB_SIZE = 8 * 1024;
    constexpr static int64_t TOTAL_SIZE = 189 * 1024;

    constexpr static  uint32_t AttenMaskDimS2 = 2048;


    uint32_t blockIdx;
    uint32_t cubeBlockIdx;
    uint32_t subIdx;

        
    int32_t taskId = 0;
    int32_t pingpongIdx = 0;
    int32_t blockLen = 0;

    // org shape info
    int64_t b;
    int64_t nheads_k;
    int64_t g;
    int64_t cuQSeqLen;
    int64_t cuKSeqLen;
    int64_t headdim;
    int64_t seq_q;
    int64_t seq_k;

    float scaleValue;

    int32_t cubeBaseMN;

    int32_t s1VecSize;
    int32_t s2VecSize;
    constexpr static int32_t S1_CUBESIZE = 128;
    constexpr static int32_t S2_CUBESIZE = 128;
    
    int32_t s1Extend;
    int32_t s2Extend;
    int32_t s2ExtendAlign;
    int32_t s1CubeExtend;
    int32_t s2CubeExtend;
    
    int32_t curSeqQIdx;
    int32_t curSeqKIdx;

    // offset 
    int32_t sfmgOffset = 0;
    int32_t lseOffset = 0;

    int64_t copyInOffset = 0;
    int64_t copyOutOffset = 0;
    DataCopyParams copyInParam;
    DataCopyParams copyOutParam;

    __gm__ uint8_t *cu_seq_qlen_addr;
    __gm__ uint8_t *cu_seq_kvlen_addr;

    SoftMaxTiling softmaxTilingData;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, AscendC::TPipe *pipe_in, __gm__ uint8_t *row_lse,
    __gm__ uint8_t *atten_mask, __gm__ uint8_t *cu_seq_qlen,
    __gm__ uint8_t *cu_seq_kvlen, __gm__ uint8_t * workspace, int32_t batchIn, __gm__ uint8_t * tiling_in)
    {
        b = batchIn;
        // ub分配
        pipe = pipe_in;

        blockIdx = GetBlockIdx();
        cubeBlockIdx = blockIdx / 2;
        subIdx = blockIdx % 2;
        curSeqQIdx = subIdx;
        curSeqKIdx = 0;

        cubeBaseMN = 16 * 128 * 128;

        cu_seq_qlen_addr = cu_seq_qlen;
        cu_seq_kvlen_addr = cu_seq_kvlen;


        // get tiling info 

        AscendC::GlobalTensor<uint64_t> tilingData;
        tilingData.SetGlobalBuffer((__gm__ uint64_t *)tiling_in);
        b = tilingData.GetValue(TILING_B);
        nheads_k = tilingData.GetValue(TILING_N2);
        g = tilingData.GetValue(TILING_G);
        headdim = tilingData.GetValue(TILING_D);
        seq_q = tilingData.GetValue(TILING_T1) / b;
        seq_k = tilingData.GetValue(TILING_T2) / b;

        int64_t sfmgWorkSpaceOffset = tilingData.GetValue(TILING_SFMG_WORKSPACE_OFFSET);
        int64_t mm1WorkSpaceOffset = tilingData.GetValue(TILING_MM1_WORKSPACE_OFFSET);
        int64_t mm2WorkSpaceOffset = tilingData.GetValue(TILING_MM2_WORKSPACE_OFFSET);
        int64_t pWorkSpaceOffset = tilingData.GetValue(TILING_P_WORKSPACE_OFFSET);
        int64_t dsWorkSpaceOffset = tilingData.GetValue(TILING_DS_WORKSPACE_OFFSET);

        AscendC::GlobalTensor<float> tilingDataFp;
        tilingDataFp.SetGlobalBuffer((__gm__ float *)tiling_in);
        scaleValue = tilingDataFp.GetValue(TILING_SCALE_VALUE * CONST_2);

        AscendC::GlobalTensor<uint32_t> tilingHostU32;
        tilingHostU32.SetGlobalBuffer((__gm__ uint32_t *)tiling_in);
        uint32_t coreNum = tilingHostU32.GetValue(TILING_CORE_NUM * CONST_2);
        softmaxTilingData.srcM = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2);
        softmaxTilingData.srcK = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 1);
        softmaxTilingData.srcSize = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 2);
        softmaxTilingData.outMaxM = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 3);
        softmaxTilingData.outMaxK = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 4);
        softmaxTilingData.outMaxSize = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 5);
        softmaxTilingData.splitM = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 6);
        softmaxTilingData.splitK = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 7);
        softmaxTilingData.splitSize = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 8);
        softmaxTilingData.reduceM = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 9);
        softmaxTilingData.reduceK = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 10);
        softmaxTilingData.reduceSize = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 11);
        softmaxTilingData.rangeM = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 12);
        softmaxTilingData.tailM = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 13);
        softmaxTilingData.tailSplitSize = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 14);
        softmaxTilingData.tailReduceSize = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 15);

        pipe->InitBuffer(unifiedBuffer, TOTAL_SIZE);
        // global tensor
        rowLseGm.SetGlobalBuffer((__gm__ float *)row_lse);
        attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *)atten_mask);

        mm1WorkspaceGm.SetGlobalBuffer((__gm__ float *)(workspace + mm1WorkSpaceOffset));
        mulWorkSpaceGm.SetGlobalBuffer((__gm__ ElementVecDtype *)(workspace + dsWorkSpaceOffset));
        
        mm2WorkspaceGm.SetGlobalBuffer((__gm__ float *)(workspace + mm2WorkSpaceOffset));
        dropWorkSpaceGm.SetGlobalBuffer((__gm__ ElementVecDtype *)(workspace + pWorkSpaceOffset));

        sfmgWorkspaceGm.SetGlobalBuffer((__gm__ float *)(workspace + sfmgWorkSpaceOffset));
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t &cuSeqQlen, int64_t &cuSeqKvlen)
    {
        if (unlikely(bIdx == 0)) {
            cuSeqQlen = ((__gm__ int32_t *)cu_seq_qlen_addr)[0];
            cuSeqKvlen = ((__gm__ int32_t *)cu_seq_kvlen_addr)[0];
        } else {
            cuSeqQlen =
                ((__gm__ int32_t *)cu_seq_qlen_addr)[bIdx] - ((__gm__ int32_t *)cu_seq_qlen_addr)[bIdx - 1];
            cuSeqKvlen =
                ((__gm__ int32_t *)cu_seq_kvlen_addr)[bIdx] - ((__gm__ int32_t *)cu_seq_kvlen_addr)[bIdx - 1];
        }
        return;
    }

    CATLASS_DEVICE
    void CopyInAttenMaskBool(LocalTensor<uint8_t> &dstTensor, int64_t attenMaskOffset, uint32_t s1Extend, uint32_t s2Extend)
    {
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = s1Extend;
        intriParams.blockLen = s2Extend * sizeof(uint8_t);
        intriParams.srcStride = (AttenMaskDimS2 - s2Extend) * sizeof(uint8_t);
        intriParams.dstStride = 0;
        intriParams.rsv = 0;
        DataCopyPad(dstTensor, attenMaskU8Gm[attenMaskOffset], intriParams, {false, 0, 0, 0});
    }

    CATLASS_DEVICE
    void CalcAttenMaskBool(
        LocalTensor<float> &dstTensor,
        LocalTensor<uint8_t> srcTensor,
        uint32_t s1Extend,
        uint32_t s2SrcExtend,
        uint32_t s2MaskExtend = 128,
        uint8_t maskType = 0)
    {
        LocalTensor<uint8_t> tmpUbBuffer = unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);

        float scalar;
        if constexpr (AscendC::IsSameType<float, float>::value) {
            uint32_t tmp = 0xFF7FFFFF;
            scalar = *((float *)&tmp);
        } else {
            uint16_t tmp = 0xFBFF;
            scalar = *((ElementVecDtype *)&tmp);
        }

        AscendC::SelectWithBytesMaskShapeInfo info;
        info.firstAxis = s1Extend;
        info.srcLastAxis = s2SrcExtend;
        // info.maskLastAxis = (s2SrcExtend * sizeof(uint8_t) + 31) / 32 * 32 / sizeof(uint8_t);
        info.maskLastAxis = s2MaskExtend;
        dstTensor.SetSize(info.firstAxis * info.srcLastAxis);
        srcTensor.SetSize(info.firstAxis * info.maskLastAxis);
        AscendC::SelectWithBytesMask<float, uint8_t, false>(dstTensor, dstTensor, scalar, srcTensor, tmpUbBuffer, info);
    }

    CATLASS_DEVICE
    void CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend, uint32_t softMaxOffset)
    {
        int64_t nheads = nheads_k * g;
        AscendC::DataCopyExtParams lseCopyParam;
        lseCopyParam.blockCount = s1Extend;
        lseCopyParam.blockLen = sizeof(float);
        lseCopyParam.srcStride = (nheads - 1) * sizeof(float);
        lseCopyParam.dstStride = 0;
        lseCopyParam.rsv = 0;
        AscendC::DataCopyPad(dstTensor, rowLseGm[softMaxOffset], lseCopyParam, {false, 0, 0, 0});
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
        for (uint32_t i = 0; i < s1Extend; i++) {
            AscendC::Brcb(dstTensor[64 * 8 + i * 8], dstTensor[i * 8], 1, {1, 8});
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(dstTensor, 1.0f, s1Extend * 8);
    }

    CATLASS_DEVICE
    void CalcSoftMax(LocalTensor<float>& dstTensor, LocalTensor<float>& src0Tensor, 
                    LocalTensor<float>& src1Tensor, uint32_t s1Extend, uint32_t s2Extend, uint32_t s2ExtendAlign, const SoftMaxTiling& tiling)
    {
        bool isBasicBlock = (s1Extend % 8 == 0) && (s2Extend % 64 == 0);

        if (isBasicBlock) {
            AscendC::LocalTensor<uint8_t> sharedTmp = unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);
            uint32_t shapeArray1[2];
            shapeArray1[0] = s1Extend;
            shapeArray1[1] = s2Extend;
            dstTensor.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray1, AscendC::DataFormat::ND));
            src0Tensor.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray1, AscendC::DataFormat::ND));
            AscendC::SimpleSoftMax<float, false, true>(dstTensor, src1Tensor, src1Tensor[64 * 8], src0Tensor,
                                        sharedTmp, tiling);
        } else {
            uint32_t sub_block_count = (s2Extend + cal_repeat_num - 1) / cal_repeat_num;

            for(uint32_t subIdx = 0; subIdx < sub_block_count; subIdx++) {
                uint32_t subMaskCount = (subIdx == sub_block_count - 1) ? (s2Extend - subIdx * cal_repeat_num) : cal_repeat_num;
                AscendC::Sub(dstTensor[subIdx * cal_repeat_num], src0Tensor[subIdx * cal_repeat_num], src1Tensor[64 * 8],
                        subMaskCount, s1Extend,
                        {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0,
                        static_cast<uint8_t>(s2ExtendAlign / 8), static_cast<uint8_t>(s2ExtendAlign / 8), 1});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(dstTensor[subIdx * cal_repeat_num], dstTensor[subIdx * cal_repeat_num],
                    subMaskCount, s1Extend,
                        {static_cast<uint8_t>(1), static_cast<uint8_t>(1),
                        static_cast<uint8_t>(s2ExtendAlign / 8), static_cast<uint8_t>(s2ExtendAlign / 8)});
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    CATLASS_DEVICE
    void SubGrapA(int64_t curIdx, const VecBlockInfo &blockInfo, event_t mte2WaitMte3A)
    {
        uint32_t ubBufferOffset = 0;

        if (curIdx > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
        }

        AscendC::LocalTensor<float> vecInBuffer3 =
            unifiedBuffer.GetWithOffset<float>(8 * 1024 / sizeof(float), ubBufferOffset + T2BlockBegin);
        
        CopyInSoftMax(vecInBuffer3, s1Extend, lseOffset);

        AscendC::LocalTensor<float> vecClc2Buffer =
            unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), ubBufferOffset + T2Begin);

        AscendC::DataCopyPad(vecClc2Buffer, mm2WorkspaceGm[copyInOffset], copyInParam, {false, 0, 0, 0});

        event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(vecClc2Buffer, vecClc2Buffer, scaleValue, s1Extend * s2ExtendAlign);
            
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<uint8_t> attenMaskUbuint8 =
            unifiedBuffer.GetWithOffset<uint8_t>(16 * 1024 / sizeof(uint8_t), ubBufferOffset + BoolBegin);
        if (blockInfo.SeqQIdx == blockInfo.SeqKIdx) {
            CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8[curSeqQIdx * s1VecSize * 128], s1Extend, s2ExtendAlign, S2_CUBESIZE, 0);
        }

        ///////////////////////////////////////////////////////////////
        // simpleSoftMax
        ///////////////////////////////////////////////////////////////
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<float>(33 * 1024 / sizeof(float), DbBegin);
        CalcSoftMax(simpleSoftmaxResBuf, vecClc2Buffer, vecInBuffer3, s1Extend, s2Extend, s2ExtendAlign, softmaxTilingData);
        LocalTensor<float> vecDropBuffer = simpleSoftmaxResBuf;

        ///////////////////////////////////////////////////////////////
        // cast fp322bf16
        ///////////////////////////////////////////////////////////////
        LocalTensor<ElementVecDtype> vecCopyOutBuffer = unifiedBuffer.GetWithOffset<ElementVecDtype>(17 * 1024 / sizeof(ElementVecDtype), ubBufferOffset + T1Begin);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(vecCopyOutBuffer, vecDropBuffer, RoundMode::CAST_ROUND, s1Extend * s2ExtendAlign);

        event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

        DataCopyPad(dropWorkSpaceGm[copyOutOffset], vecCopyOutBuffer, copyOutParam);

        if (curIdx < blockLen - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
        }
    }
    
    CATLASS_DEVICE
    void SubGrapB(int64_t curIdx, const VecBlockInfo &blockInfo, event_t mte2WaitMte3B)
    {
        uint32_t ubBufferOffset = DbBegin;

        if (curIdx > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }

        // copyIn sfmg
        LocalTensor<float> sfmgClc3 = unifiedBuffer.GetWithOffset<float>(SFMG_UB_SIZE / sizeof(float), SFMG_UB_OFFSET);
        DataCopy(sfmgClc3, sfmgWorkspaceGm[sfmgOffset], s1Extend * 8);

        LocalTensor<float> vecClc1Buffer = unifiedBuffer.GetWithOffset<float>(33 * 1024 / sizeof(float), ubBufferOffset + T1Begin);
        
        // copyIn cube result
        DataCopyPad(vecClc1Buffer, mm1WorkspaceGm[copyInOffset], copyInParam, {false, 0, 0, 0});

        event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);

        ///////////////////////////////////////////////////////////////
        // sub
        ///////////////////////////////////////////////////////////////
        uint32_t sub_block_cout = (s2ExtendAlign + cal_repeat_num - 1) / cal_repeat_num;
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t subIdx = 0; subIdx < sub_block_cout; subIdx++) {
            uint32_t subMaskCout =
                (subIdx == sub_block_cout - 1) ? (s2ExtendAlign - subIdx * cal_repeat_num) : cal_repeat_num;
            Sub(vecClc1Buffer[subIdx * cal_repeat_num], vecClc1Buffer[subIdx * cal_repeat_num], sfmgClc3,
                subMaskCout, s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                static_cast<uint8_t>(s2ExtendAlign / 8), 1});
        }

        ///////////////////////////////////////////////////////////////
        // mul
        ///////////////////////////////////////////////////////////////
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), DbBegin);
        Mul(vecClc1Buffer, vecClc1Buffer, simpleSoftmaxResBuf, s1Extend * s2ExtendAlign);

        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<ElementVecDtype> vecCopyOutBuffer = unifiedBuffer.GetWithOffset<ElementVecDtype>(17 * 1024 / sizeof(ElementVecDtype), ubBufferOffset + T1Begin);
        Cast(vecCopyOutBuffer, vecClc1Buffer, RoundMode::CAST_ROUND, s1Extend * s2ExtendAlign);

        event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

        // doutv = dp -> ds
        DataCopyPad(mulWorkSpaceGm[copyOutOffset], vecCopyOutBuffer, copyOutParam);

        if (curIdx < blockLen - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }
    }

    CATLASS_DEVICE
    void operator()(const VecAddrInfo &addrs)
    {
        taskId = addrs.taskId;
        pingpongIdx = taskId % 2;
        blockLen = addrs.blockLength;

        event_t mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
        if (taskId == 0) {
            LocalTensor<uint8_t> attenMaskUbuint8 =
                    unifiedBuffer.GetWithOffset<uint8_t>(16 * 1024 / sizeof(uint8_t), BoolBegin);
            CopyInAttenMaskBool(attenMaskUbuint8, 0, 128, 128);
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < blockLen; ++i) {
            
            auto &blockInfo = addrs.VecBlkInfo[i];

            ///////////////////////////////////////////////////////////////
            // do scalar calculate
            ///////////////////////////////////////////////////////////////
            
            if constexpr (getLayout() == InputLayout::TND) {
                GetSeqQlenKvlenByBidx(blockInfo.batchIdx, cuQSeqLen, cuKSeqLen);
            } else {
                cuQSeqLen = seq_q;
                cuKSeqLen = seq_k;
            }

            s1CubeExtend = blockInfo.lengthy;
            s2CubeExtend = 128;

            // split info
            s1VecSize = (s1CubeExtend + 1) / 2;
            s2VecSize = 128;

            s1Extend = subIdx ? s1CubeExtend - s1VecSize : s1VecSize;
            s2Extend = blockInfo.lengthx;
            s2ExtendAlign = (s2Extend + 15) / 16 * 16;

            //offset
            int64_t globalSeqStart = 0;
            if constexpr (getLayout() == InputLayout::TND) {
                globalSeqStart = (blockInfo.batchIdx > 0)
                ? ((__gm__ int32_t *)cu_seq_qlen_addr)[blockInfo.batchIdx - 1]
                : 0;
            } else {
                globalSeqStart = (blockInfo.batchIdx > 0)
                ? seq_q
                : 0;
            }
            int64_t seqOffsetInBlock = blockInfo.SeqQIdx * S1_CUBESIZE + curSeqQIdx * s1VecSize;
            int64_t nheads = nheads_k * g;
            int64_t headIdx = blockInfo.nheadsKIdx * g + blockInfo.gIdx;
            lseOffset = (globalSeqStart + seqOffsetInBlock) * nheads + headIdx;

            sfmgOffset = 0;
            if (blockInfo.batchIdx > 0) {
                if constexpr (getLayout() == InputLayout::TND) {
                    sfmgOffset = ((__gm__ int32_t *)cu_seq_qlen_addr)[blockInfo.batchIdx - 1] * nheads_k * g * 8;
                } else {
                    sfmgOffset = seq_q * nheads_k * g * 8;
                }
            }
            sfmgOffset += ((blockInfo.nheadsKIdx * g + blockInfo.gIdx) * cuQSeqLen + blockInfo.SeqQIdx * S1_CUBESIZE + curSeqQIdx * s1VecSize) * 8;
            
            // copyIn cube_workspace params
            copyInOffset = 
                cubeBlockIdx * cubeBaseMN * 2 + pingpongIdx * cubeBaseMN + blockInfo.offset + curSeqQIdx * s1VecSize * s2CubeExtend;
            copyInParam = {
                static_cast<uint16_t>(s1Extend),
                static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                static_cast<uint16_t>((s2CubeExtend - s2ExtendAlign) * sizeof(float)), 
                0
            };

            // copyOut cube_workspace params
            copyOutOffset = 
                (cubeBlockIdx * cubeBaseMN * 2 + pingpongIdx * cubeBaseMN + blockInfo.offset) +
                (curSeqQIdx * s1VecSize * s2CubeExtend);
            copyOutParam = {
                static_cast<uint16_t>(s1Extend),
                static_cast<uint16_t>(s2ExtendAlign * sizeof(ElementVecDtype)),
                0,
                static_cast<uint16_t>((s2CubeExtend - s2ExtendAlign) * sizeof(ElementVecDtype))
            };

            ///////////////////////////////////////////////////////////////
            // do vector calculate
            ///////////////////////////////////////////////////////////////
            event_t mte2WaitMte3A = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
            event_t mte2WaitMte3B = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
            SubGrapA(i, blockInfo, mte2WaitMte3A);
            SubGrapB(i, blockInfo, mte2WaitMte3B);
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }
    }

};
    
}

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_OP_HPP
