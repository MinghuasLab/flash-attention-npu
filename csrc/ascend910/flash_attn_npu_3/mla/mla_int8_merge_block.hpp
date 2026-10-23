/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0.
// attention/incre_flash_attention/op_kernel/arch22/service_vector_flashdecode_mla.h.
// Only INT8 content / BF16 RoPE and output, Hkv=1, 512+64, ND PA.
#pragma once
#include "mla_int8_ops_tile.hpp"
namespace flash_attn::mla::Int8 {
namespace AttentionCommonFlashDecode {
struct FDparams {
    uint32_t* bN2IdxOfFdHead;
    uint32_t* gS1IdxOfFdHead;
    uint32_t* s2SplitNumOfFdHead;
    uint32_t* gS1SplitNumOfFdHead;
    uint32_t* gS1LastPartSizeOfFdHead;
    uint32_t* gS1IdxEndOfFdHead;
    uint32_t* gS1IdxEndOfFdHeadSplit;
    uint32_t usedVecNumOfFd;
    uint32_t gS1BaseSizeOfFd;
};
struct ConstInfoFD {
    // CUBE与VEC核间同步的模式
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    // FP32的0值和极大值
    static constexpr float FLOAT_ZERO = 0;
    static constexpr float FLOAT_MAX = 3.402823466e+38F;

    // preLoad的总次数
    uint32_t preLoadNum = 0U;
    uint64_t batchSize = 0UL;
    uint64_t gSize = 0UL;
    uint64_t qHeadNum = 0UL;
    uint64_t kvHeadNum;
    uint64_t headDim;

    uint64_t headDimAlign;
    uint64_t qSeqSize = 1UL; // q最大S长度

    uint32_t mBaseSize = 1UL;
};
} // namespace AttentionCommonFlashDecode

using namespace AttentionCommonFlashDecode;
using namespace AscendC;
using AscendC::AIC;
using AscendC::AIV;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::ShapeInfo;
using AscendC::SoftmaxConfig;
using AscendC::WaitFlag;
#define MLA_INT8_ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

struct TaskInfo {
    uint32_t bIdx;
    uint32_t gS1Idx;
    uint32_t actualCombineLoopSize;
};

namespace FaBaseVectorFlashDecode {
// BLOCK和REPEAT的字节数
constexpr uint64_t BYTE_BLOCK = 32UL;
constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
// BLOCK和REPEAT的FP32元素数
constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
// repeat stride不能超过256
constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256U;
template <typename T>
__aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 2:此场景元素个数翻倍
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 2:此场景元素个数翻倍
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)
    // 列方向分dLoop次，每次处理8列数据
    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (dLoop <= dealRowCount) {
            int64_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum,
                    dLoop, columnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, dealRowCount,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8; // 每个repeat为256B数据，正好8个datablock
        repeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t i = 0; i < dealRowCount; i++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                repeatParams);
            //  计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * columnCount + dLoop * repeatElementNum],
                    src0Ub[i * columnCount + dLoop * repeatElementNum], src1Ub[i * blockElementNum], dRemain, 1,
                    repeatParams);
            }
        }
    }
}

__aicore__ inline void MatDivsVecFd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 1;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 0;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    int64_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Div(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsDiv);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount,
            repeatParamsDiv);
    }
}

__aicore__ inline void RowSubFd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsSub;
    repeatParamsSub.src0BlkStride = 1;
    repeatParamsSub.src1BlkStride = 1;
    repeatParamsSub.dstBlkStride = 1;
    repeatParamsSub.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsSub.src1RepStride = 0;
    repeatParamsSub.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    int64_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Sub(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsSub);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Sub(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount,
            repeatParamsSub);
    }
}

__aicore__ inline void ColMaxFd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = 0;
    repeatParamsMax.dstRepStride = 0;
    uint32_t columnRepeatCount = dLoop;
    int64_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Max(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsMax);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Max(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount,
            repeatParamsMax);
    }
}

__aicore__ inline void ColAddFd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsAdd;
    repeatParamsAdd.src0BlkStride = 1;
    repeatParamsAdd.src1BlkStride = 1;
    repeatParamsAdd.dstBlkStride = 1;
    repeatParamsAdd.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsAdd.src1RepStride = 0;
    repeatParamsAdd.dstRepStride = 0;
    uint32_t columnRepeatCount = dLoop;
    int64_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Add(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsAdd);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Add(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount,
            repeatParamsAdd);
    }
}

} // namespace FaBaseVectorFlashDecode

// ServiceFlashDecode类
template <typename IFAT>
class ServiceFlashDecode {
  public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using OUT_T = typename IFAT::outputType;
    static constexpr LAYOUT LAYOUT_T = IFAT::layout;
    __aicore__ inline void InitLseOutput(GlobalTensor<T> output, bool enabled)
    {
        softmaxLseGm = output;
        softmaxLseFlag = enabled;
    }

    __aicore__ inline void InitGlobalTensor(GlobalTensor<T> lseMaxFdGm, GlobalTensor<T> lseSumFdGm,
                                            GlobalTensor<T> accumOutGm, GlobalTensor<OUT_T> attentionOutGm,
                                            QueryLengths actualSeqLengthsGmQ);
    __aicore__ inline void InitParams(const AttentionCommonFlashDecode::ConstInfoFD& constInfo);
    __aicore__ inline void InitDecodeParams();
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void FlashDecode(FDparams& fd);

  protected:
    __aicore__ inline void CopyAccumOutIn(LocalTensor<T>& accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount);
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, int64_t baseOffset, uint32_t cntM);
    __aicore__ inline void ComputeScaleValue(LocalTensor<T>& lseExp, uint32_t startRow, uint32_t dealRowCount,
                                             uint32_t cntM);
    __aicore__ inline void Bmm2DataCopyOut(int64_t attenOutOffset, LocalTensor<OUT_T>& attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ReduceFinalRes(LocalTensor<T>& reduceOut, LocalTensor<T>& mm2Res, LocalTensor<T>& lseLocal,
                                          uint32_t cntKV, uint32_t dealRowCount);
    __aicore__ inline void CopyFinalResOut(LocalTensor<T>& accumOutLocal, uint32_t startRow, uint32_t dealRowCount,
                                           int64_t attenOutOffset);

  private:
    // =================================常量区=================================
    static constexpr uint64_t SYNC_LSE_SUM_BUF1_FLAG = 6;
    static constexpr uint64_t SYNC_LSE_SUM_BUF2_FLAG = 7;
    static constexpr uint64_t SYNC_LSE_MAX_BUF1_FLAG = 8;
    static constexpr uint64_t SYNC_LSE_MAX_BUF2_FLAG = 9;
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 12;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = FaBaseVectorFlashDecode::BYTE_BLOCK / sizeof(T);

  protected:
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    QueryLengths actualSeqLengthsGmQ;
    GlobalTensor<T> softmaxLseGm;
    bool softmaxLseFlag = false;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t blockIdx = 0U;
    AttentionCommonFlashDecode::ConstInfoFD constInfo{};
    TaskInfo taskInfo{};

  private:
    // ================================FD Local Buffer区====================================
    TBuf<> fdSumBuf1;
    TBuf<> fdSumBuf2;
    TBuf<> fdMaxBuf1;
    TBuf<> fdMaxBuf2;
    TBuf<> fdLseExpBuf;
    TBuf<> fdMm2ResBuf1;
    TBuf<> fdMm2ResBuf2;
    TBuf<> fdReduceBuf;
    TBuf<> fdOutputBuf;

    TBuf<> fdLseMaxUbBuf;
    TBuf<> fdLseSumUbBuf;
};

template <typename IFAT>
__aicore__ inline void
ServiceFlashDecode<IFAT>::InitGlobalTensor(GlobalTensor<T> lseMaxFdGm, GlobalTensor<T> lseSumFdGm,
                                           GlobalTensor<T> accumOutGm, GlobalTensor<OUT_T> attentionOutGm,
                                           QueryLengths actualSeqLengthsGmQ)
{
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
    this->accumOutGm = accumOutGm;
    this->attentionOutGm = attentionOutGm;
    this->actualSeqLengthsGmQ = actualSeqLengthsGmQ;
}

template <typename IFAT>
__aicore__ inline void ServiceFlashDecode<IFAT>::InitParams(const AttentionCommonFlashDecode::ConstInfoFD& constInfo)
{
    this->constInfo = constInfo;
}

template <typename IFAT>
__aicore__ inline void ServiceFlashDecode<IFAT>::InitDecodeParams()
{
    this->blockIdx = GetBlockIdx();
}

template <typename IFAT>
__aicore__ inline void ServiceFlashDecode<IFAT>::InitBuffers(TPipe* pipe)
{
    if ASCEND_IS_AIV {
        pipe->Reset();
        pipe->InitBuffer(fdSumBuf1, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_4K +
                                        AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdSumBuf2, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_4K +
                                        AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdMaxBuf1, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_4K +
                                        AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdMaxBuf2, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_4K +
                                        AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdLseExpBuf, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_4K +
                                          AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdMm2ResBuf1, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdMm2ResBuf2, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdReduceBuf, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdOutputBuf, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdLseMaxUbBuf, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_256B);
        pipe->InitBuffer(fdLseSumUbBuf, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_256B);
    }
}

template <typename IFAT>
__aicore__ inline void ServiceFlashDecode<IFAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
}

template <typename IFAT>
__aicore__ inline void ServiceFlashDecode<IFAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
}

template <typename IFAT>
__aicore__ inline void ServiceFlashDecode<IFAT>::CopyAccumOutIn(LocalTensor<T>& accumOutLocal, uint32_t splitKVIndex,
                                                                uint32_t startRow, uint32_t dealRowCount)
{
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = constInfo.headDim * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (constInfo.headDimAlign - constInfo.headDim) / BLOCK_ELEMENT_NUM;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (constInfo.headDimAlign - constInfo.headDim) % BLOCK_ELEMENT_NUM;
    copyInPadParams.paddingValue = 0;
    int64_t combineAccumOutOffset = startRow * constInfo.headDim +                          // taskoffset + g轴offset
                                    splitKVIndex * constInfo.mBaseSize * constInfo.headDim; // 份数offset
    DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
}

template <typename IFAT>
__aicore__ inline void ServiceFlashDecode<IFAT>::CopyLseIn(uint32_t startRow, uint32_t dealRowCount, int64_t baseOffset,
                                                           uint32_t cntM)
{
    // 双缓冲区域
    LocalTensor<T> lseSum = cntM % 2 == 0 ? fdSumBuf1.Get<T>() : fdSumBuf2.Get<T>();
    LocalTensor<T> lseMax = cntM % 2 == 0 ? fdMaxBuf1.Get<T>() : fdMaxBuf2.Get<T>();

    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF1_FLAG + cntM % 2);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF1_FLAG + cntM % 2);

    int64_t combineLseOffset = (baseOffset + startRow) * FaBaseVectorFlashDecode::FP32_BLOCK_ELEMENT_NUM;
    int64_t combineLoopOffset = constInfo.mBaseSize * FaBaseVectorFlashDecode::FP32_BLOCK_ELEMENT_NUM;
    uint64_t dealRowCountAlign = dealRowCount * FaBaseVectorFlashDecode::FP32_BLOCK_ELEMENT_NUM;
    for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
        DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm[combineLseOffset + i * combineLoopOffset],
                 dealRowCountAlign); // 份数offset
        DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm[combineLseOffset + i * combineLoopOffset],
                 dealRowCountAlign);
    }
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_SUM_BUF1_FLAG + cntM % 2);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_BUF1_FLAG + cntM % 2);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_SUM_BUF1_FLAG + cntM % 2);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_BUF1_FLAG + cntM % 2);
}

template <typename IFAT>
__aicore__ inline void ServiceFlashDecode<IFAT>::ComputeScaleValue(LocalTensor<T>& lseExp, uint32_t startRow,
                                                                   uint32_t dealRowCount, uint32_t cntM)
{
    LocalTensor<T> lseSum = cntM % 2 == 0 ? fdSumBuf1.Get<T>() : fdSumBuf2.Get<T>();
    LocalTensor<T> lseMax = cntM % 2 == 0 ? fdMaxBuf1.Get<T>() : fdMaxBuf2.Get<T>();

    LocalTensor<T> lseMaxUb = fdLseMaxUbBuf.Get<T>();
    LocalTensor<T> lseSumUb = fdLseSumUbBuf.Get<T>();
    uint64_t dealRowCountAlign = dealRowCount * FaBaseVectorFlashDecode::FP32_BLOCK_ELEMENT_NUM;
    Duplicate(lseMaxUb, -AttentionCommonFlashDecode::ConstInfoFD::FLOAT_MAX, dealRowCountAlign);
    Duplicate(lseSumUb, AttentionCommonFlashDecode::ConstInfoFD::FLOAT_ZERO, dealRowCountAlign);
    pipe_barrier(PIPE_V);

    FaBaseVectorFlashDecode::ColMaxFd(lseMaxUb, lseMax, lseMaxUb, taskInfo.actualCombineLoopSize, dealRowCountAlign,
                                      dealRowCountAlign);
    pipe_barrier(PIPE_V);

    FaBaseVectorFlashDecode::RowSubFd(lseExp, lseMax, lseMaxUb, taskInfo.actualCombineLoopSize, dealRowCountAlign,
                                      dealRowCountAlign);
    pipe_barrier(PIPE_V);

    Exp(lseExp, lseExp, taskInfo.actualCombineLoopSize * dealRowCountAlign);
    pipe_barrier(PIPE_V);

    Mul(lseExp, lseSum, lseExp, taskInfo.actualCombineLoopSize * dealRowCountAlign);
    pipe_barrier(PIPE_V);

    FaBaseVectorFlashDecode::ColAddFd(lseSumUb, lseExp, lseSumUb, taskInfo.actualCombineLoopSize, dealRowCountAlign,
                                      dealRowCountAlign);
    pipe_barrier(PIPE_V);

    FaBaseVectorFlashDecode::MatDivsVecFd(lseExp, lseExp, lseSumUb, taskInfo.actualCombineLoopSize, dealRowCountAlign,
                                          dealRowCountAlign);
    pipe_barrier(PIPE_V);

    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF1_FLAG + cntM % 2);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF1_FLAG + cntM % 2);
}

template <typename IFAT>
__aicore__ inline void ServiceFlashDecode<IFAT>::Bmm2DataCopyOut(int64_t attenOutOffset, LocalTensor<OUT_T>& attenOutUb,
                                                                 uint32_t startRow, uint32_t dealRowCount,
                                                                 uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride =
        (columnCount - actualColumnCount) / (FaBaseVectorFlashDecode::BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb, dataCopyParams);
}

template <typename IFAT>
__aicore__ inline void ServiceFlashDecode<IFAT>::ReduceFinalRes(LocalTensor<T>& reduceOut, LocalTensor<T>& mm2Res,
                                                                LocalTensor<T>& lseLocal, uint32_t cntKV,
                                                                uint32_t dealRowCount)
{
    uint32_t dealRowCountAlign = dealRowCount * FaBaseVectorFlashDecode::FP32_BLOCK_ELEMENT_NUM;
    LocalTensor<T> tmpRst =
        cntKV == 0 ? reduceOut : mm2Res; // 第一次mul结果直接写入reduceOut，否则在mm2Res原地进行mul，再加到reduceOut

    FaBaseVectorFlashDecode::RowMuls(tmpRst, mm2Res, lseLocal[cntKV * dealRowCountAlign], dealRowCount,
                                     constInfo.headDimAlign, constInfo.headDim);

    if (cntKV != 0) {
        pipe_barrier(PIPE_V);
        Add(reduceOut, reduceOut, tmpRst, dealRowCount * constInfo.headDimAlign);
        pipe_barrier(PIPE_V);
    }
}

template <typename IFAT>
__aicore__ inline void ServiceFlashDecode<IFAT>::CopyFinalResOut(LocalTensor<T>& accumOutLocal, uint32_t startRow,
                                                                 uint32_t dealRowCount, int64_t attenOutOffset)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = fdOutputBuf.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo.headDim};
    tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, dealRowCount * constInfo.headDimAlign);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    Bmm2DataCopyOut(attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, constInfo.headDimAlign,
                    constInfo.headDim);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    if (softmaxLseFlag) {
        // Reuse the completed source reduction's FP32 sum/max. The pinned
        // fast FD service omits LSE; ordinary FD uses this same log(sum)+max.
        // The output buffer's existing MTE3/V handshake also owns this copy.
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
        auto lse = fdOutputBuf.Get<T>();
        auto sum = fdLseSumUbBuf.Get<T>();
        auto max = fdLseMaxUbBuf.Get<T>();
        Log(lse, sum, dealRowCount * BLOCK_ELEMENT_NUM);
        PipeBarrier<PIPE_V>();
        Add(lse, lse, max, dealRowCount * BLOCK_ELEMENT_NUM);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        DataCopyExtParams copy{1, sizeof(T), 0, 0, 0};
        for (uint32_t row = 0; row < dealRowCount; ++row) {
            uint64_t flat = attenOutOffset / constInfo.headDim + startRow + row;
            uint64_t offset = flat;
            if constexpr (LAYOUT_T == LAYOUT::TND) {
                offset = (flat % constInfo.qHeadNum) * actualSeqLengthsGmQ.GetValue(constInfo.batchSize - 1) +
                         flat / constInfo.qHeadNum;
            } else {
                uint64_t batch = flat / (constInfo.qSeqSize * constInfo.qHeadNum);
                uint64_t token = (flat / constInfo.qHeadNum) % constInfo.qSeqSize;
                uint64_t head = flat % constInfo.qHeadNum;
                offset = (batch * constInfo.qHeadNum + head) * constInfo.qSeqSize + token;
            }
            DataCopyPad(softmaxLseGm[offset], lse[row * BLOCK_ELEMENT_NUM], copy);
        }
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    }
}

template <typename IFAT>
__aicore__ inline void ServiceFlashDecode<IFAT>::FlashDecode(FDparams& fd)
{
    if (blockIdx >= fd.usedVecNumOfFd) {
        return;
    }
    uint32_t fdTaskPrevEnd = (blockIdx > 0) ? fd.gS1IdxEndOfFdHead[blockIdx - 1] : 0U; // 上一个核末尾是第几个规约
    uint32_t fdS1gOuterMPrevEnd =
        (blockIdx > 0) ? fd.gS1IdxEndOfFdHeadSplit[blockIdx - 1] : 0U;           //上一个核末尾是该规约的第几个base行
    uint32_t fdTaskEnd = fd.gS1IdxEndOfFdHead[blockIdx];                         // 当前核的末尾是第几个规约任务
    uint32_t fdS1gOuterMEnd = fd.gS1IdxEndOfFdHeadSplit[blockIdx];               // 当前核的末尾是该规约的第几个base行
    uint32_t tmpFdS1gOuterMStart = (blockIdx > 0) ? fdS1gOuterMPrevEnd + 1 : 0U; // 当前核从第几个base行开始
    uint32_t tmpFdS1gOuterMEnd = 0U;
    uint32_t reduceGlobalLoop = 0U;
    uint32_t reduceMLoop = 0U;

    // 并不是每个核都会处理同一规约任务的不同行，以下是一个泛化性说明，自己手动推一下即可
    // 首先针对每个核遍历一下它需要处理的规约任务
    for (uint32_t fdTaskId = fdTaskPrevEnd; fdTaskId <= fdTaskEnd; fdTaskId++) {
        tmpFdS1gOuterMEnd = (fdTaskId == fdTaskEnd)
                                ? fdS1gOuterMEnd
                                : (fd.gS1SplitNumOfFdHead[fdTaskId] - 1);    //针对每个核当前处理的规约任务的尾块索引
        taskInfo.bIdx = fd.bN2IdxOfFdHead[fdTaskId];                         //针对每个核当前处理的规约任务的batch索引
        taskInfo.gS1Idx = fd.gS1IdxOfFdHead[fdTaskId] * constInfo.mBaseSize; //针对每个核当前处理的规约任务的S1G索引
        taskInfo.actualCombineLoopSize = fd.s2SplitNumOfFdHead[fdTaskId];    // 当前规约任务kv方向有几份

        //统计当前处理的规约任务之前所有规约任务的块数，单位是块！！
        uint64_t combineTaskPrefixSum = 0UL;
        for (int i = 0; i < fdTaskId; i++) {
            // 计算此前规约数据的累计份数，每一份的数据大小为 kvHeadNum * constInfo.tndSgBasicSize
            // |Task0-0|Task0-1|Task0-3|Task1-0|Task1-2|...|
            combineTaskPrefixSum += fd.s2SplitNumOfFdHead[i];
        }

        int64_t curAttenOutOffset = 0;

        {
            uint64_t actualSeqQPrefixSum;
            if constexpr (LAYOUT_T == LAYOUT::TND) {
                actualSeqQPrefixSum = (taskInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(taskInfo.bIdx - 1);
            } else { // BSND
                actualSeqQPrefixSum = (taskInfo.bIdx <= 0) ? 0 : taskInfo.bIdx * constInfo.qSeqSize;
            }
            int64_t batchOutOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
            curAttenOutOffset = batchOutOffset + taskInfo.gS1Idx * constInfo.headDim;
        }
        // 计算一个偏移，计算真实偏移量，分块大小（这里的分块是指workspace阶段沿着S1G切分）的实际行数*块数，计算当前规约任务之前，所有规约任务的偏移
        int64_t taskOffset = combineTaskPrefixSum * constInfo.kvHeadNum * constInfo.mBaseSize;
        // 第二级索引，针对每个核的正在处理的规约任务来说，遍历它的行基本块（这里是指除完8之后的）
        for (uint32_t fdS1gOuterMIdx = tmpFdS1gOuterMStart; fdS1gOuterMIdx <= tmpFdS1gOuterMEnd;
             fdS1gOuterMIdx++) { // 左闭右闭

            uint32_t actualGSplitSize = fd.gS1BaseSizeOfFd;
            // 如果处理到每一个规约任务的尾块，要考虑到尾块并不都是8的情况，所以actualGSplitSize被重新赋值为尾块里的实际行数
            if (fdS1gOuterMIdx == fd.gS1SplitNumOfFdHead[fdTaskId] - 1) {
                actualGSplitSize = fd.gS1LastPartSizeOfFdHead[fdTaskId];
            }
            // startRow被定义为8*行基本块（索引）,可以把它理解成正在处理的当前规约任务的行基本块，之前有多少总行数
            uint32_t startRow = fdS1gOuterMIdx * fd.gS1BaseSizeOfFd;

            LocalTensor<T> lseExp = fdLseExpBuf.Get<T>();
            LocalTensor<T> reduceOut = fdReduceBuf.Get<T>();
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);

            LocalTensor<T> mm2Res;
            for (uint32_t preLoadIdx = 0; preLoadIdx < constInfo.preLoadNum; preLoadIdx++) {
                mm2Res = (reduceGlobalLoop + preLoadIdx) % 2 == 0 ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
                WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobalLoop + preLoadIdx) % 2);
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobalLoop + preLoadIdx) % 2);
            }

            ComputeScaleValue(lseExp, startRow, actualGSplitSize, reduceMLoop);

            for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
                mm2Res = reduceGlobalLoop % 2 == 0 ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
                if (i >= constInfo.preLoadNum) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + reduceGlobalLoop % 2);
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + reduceGlobalLoop % 2);
                }

                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + reduceGlobalLoop % 2);
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + reduceGlobalLoop % 2);
                reduceGlobalLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize, curAttenOutOffset);
            reduceMLoop += 1;
        }
        tmpFdS1gOuterMStart = 0;
    }
}

} // namespace flash_attn::mla::Int8
