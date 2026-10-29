/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Source: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0,
// arch22/fia_block_vec_nonquant.h, nonquant/no-PSE/no-sink specialization.
// Preserve generic SoftmaxFlashV2 (BRC/non-BRC),
// cast, rescale/add/divide order and ping-pong workspace. This differs from
// the compressed MLA aMLA update. Mask and empty-row mapping use FA3 semantics.
#pragma once
#include "mla_vector_ops_tile.hpp"
#include "mla_prefill_cube_block.hpp"
namespace flash_attn::mla::Block {
using namespace AscendC;
constexpr SoftmaxConfig PREFILL_SOFTMAX_CFG = {false};
constexpr SoftmaxConfig PREFILL_SOFTMAX_NO_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};
template <typename InputT, bool SOFTMAX_WITH_BRC, bool Partial>
class MlaPrefillVector {
    using T = float;
    using COMPUTE_T = float;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;
    using SOFTMAX_TYPE = float;
    using KV_T = InputT;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_SIZE = 2048, BASE_BLOCK_MAX_ELEMENT_NUM = 8192;
    static constexpr uint16_t brcbNum = 8;
    struct SplitInfo {
        uint32_t nBufferIdx, nBufferStartM, nBufferDealM, vecStartM, vecDealM;
    } mSplitInfo;

  public:
    __aicore__ inline void Init(const MlaPrefillParams& p, TPipe* pipe, GlobalTensor<float> qk,
                                GlobalTensor<InputT> prob, GlobalTensor<float> pv, GlobalTensor<float> previous,
                                GM_ADDR out, GM_ADDR lse, GM_ADDR partialOut, GM_ADDR sum, GM_ADDR maximum)
    {
        params = p;
        constInfo = {p.mBase * Tile::Align(p.nBase, 32), p.mBase * Tile::Align(p.nBase, 32), p.mBase * 128, p.nBase,
                     p.heads / p.kvHeads};
        mm1ResGm = qk;
        vec1ResGm = prob;
        mm2ResGm = pv;
        vec2ResGm = previous;
        output.SetGlobalBuffer((__gm__ InputT*)out);
        outputLse.SetGlobalBuffer((__gm__ float*)lse);
        accumOutGm.SetGlobalBuffer((__gm__ float*)partialOut);
        lseSumFdGm.SetGlobalBuffer((__gm__ float*)sum);
        lseMaxFdGm.SetGlobalBuffer((__gm__ float*)maximum);
        InitBuffers(pipe);
    }
    __aicore__ inline void ComputeVec1(const PrefillInfo& info);
    __aicore__ inline void ComputeVec2(const PrefillInfo& info);
    __aicore__ inline void InitEmpty(const PrefillTaskContext& context)
    {
        PrefillInfo info{};
        info.context = context;
        auto zero = outputQue1.AllocTensor<float>();
        Duplicate(zero, 0.0f, 128);
        Duplicate(zero[128], -float(1.0 / 0.0), 8);
        outputQue1.EnQue(zero);
        outputQue1.DeQue<float>();
        for (uint32_t r = GetSubBlockIdx(); r < context.task.rows; r += 2) {
            if (Partial && context.partial) {
                DataCopy(accumOutGm[(context.partialRow + r) * 128], zero, 128);
                DataCopy(lseSumFdGm[(context.partialRow + r) * 8], zero, 8);
                DataCopy(lseMaxFdGm[(context.partialRow + r) * 8], zero[128], 8);
            } else {
                DataCopy(output[OutputRow(info, r) * 128], zero.ReinterpretCast<InputT>(), 128);
                DataCopyPad(outputLse[LseRow(info, r)], zero[128], DataCopyExtParams{1, 4, 0, 0, 0});
            }
        }
        outputQue1.FreeTensor(zero);
    }

  private:
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void SetMSplitInfo(uint32_t mDealSize);
    __aicore__ inline void ProcessVec1SingleBuf(const PrefillInfo& info);
    __aicore__ inline void ProcessVec2SingleBuf(const PrefillInfo& info);
    __aicore__ inline void DealBmm1ResBaseBlock(const PrefillInfo& info, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void SoftmaxFlashV2Compute(const PrefillInfo& info, LocalTensor<float>& mmResUb,
                                                 LocalTensor<uint8_t>& softmaxTmpUb, uint32_t startRow,
                                                 uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void DealBmm2ResBaseBlock(const PrefillInfo& info, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void StoreStats(const PrefillInfo& info);
    __aicore__ inline void Bmm2ResCopyOut(const PrefillInfo& info, LocalTensor<float>& value, uint32_t row,
                                          uint32_t startRow, uint32_t rows, uint32_t columns, uint32_t actualColumns);
    __aicore__ inline bool Valid(const PrefillInfo& info, uint32_t localRow) const
    {
        const auto& task = info.context.task;
        int64_t lo = 0, hi = task.kvLength;
        int64_t center = (task.rowStart + localRow) / constInfo.gSize + int64_t(task.kvLength) - task.queryLength;
        if (params.window.kind == 1)
            hi = Min(hi, center + 1);
        else if (params.window.kind == 2) {
            lo = Max(lo, center - params.window.left);
            hi = Min(hi, center + params.window.right + 1);
        }
        return lo < hi;
    }
    __aicore__ inline uint64_t OutputRow(const PrefillInfo& info, uint32_t localRow) const
    {
        const auto& task = info.context.task;
        uint32_t flat = task.rowStart + localRow;
        return info.context.outputOffset + (uint64_t(task.batch) * params.sq + flat / constInfo.gSize) * params.heads +
               task.kvHead * constInfo.gSize + flat % constInfo.gSize;
    }
    __aicore__ inline uint64_t LseRow(const PrefillInfo& info, uint32_t localRow) const
    {
        const auto& task = info.context.task;
        uint32_t flat = task.rowStart + localRow;
        return info.context.lseOffset +
               (uint64_t(task.batch) * params.heads + task.kvHead * constInfo.gSize + flat % constInfo.gSize) *
                   params.sq +
               flat / constInfo.gSize;
    }
    MlaPrefillParams params;
    PrefillConstants constInfo;
    GlobalTensor<float> mm1ResGm, mm2ResGm, vec2ResGm, accumOutGm, lseSumFdGm, lseMaxFdGm, outputLse;
    GlobalTensor<InputT> vec1ResGm, output;
    TQue<TPosition::VECIN, 2> inputQue1, inputQue2;
    TQue<TPosition::VECOUT, 1> outputQue1, outputQue2;
    TBuf<> tmpBuff1, softmaxMaxBuff, softmaxExpBuff, softmaxSumBuff, softmaxMaxDefaultBuff, softmaxSumDefaultBuff;
    LocalTensor<float> softmaxMaxUb, softmaxSumUb, softmaxExpUb, softmaxMaxDefaultUb, softmaxSumDefaultUb;
};
template <typename InputT, bool SOFTMAX_WITH_BRC, bool Partial>
__aicore__ inline void MlaPrefillVector<InputT, SOFTMAX_WITH_BRC, Partial>::InitBuffers(TPipe* pipe)
{
    // in queue
    pipe->InitBuffer(inputQue1, 2, 32768); // 2:pingpong
    pipe->InitBuffer(inputQue2, 2, 16384); // 2:pingpong

    // out queue
    pipe->InitBuffer(outputQue1, 1, 32768);
    pipe->InitBuffer(outputQue2, 1, 8192);

    // tmpBuff
    pipe->InitBuffer(tmpBuff1, 32768);
    // 1. [M,8]场景: 2K/32B = 64, 即单个VEC上可以缓存64行, 整个AICORE上有2个VEC，所以此时分核的MBaseSize<=128
    // 2. [M,1]场景: 2K/sizeof(float) = 512, 即单个VEC上可以缓存512行, 所以此时分核的MBaseSize<=512*2=1024
    pipe->InitBuffer(softmaxMaxBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, SOFTMAX_TMP_BUFFER_SIZE);
    pipe->InitBuffer(softmaxSumDefaultBuff, SOFTMAX_TMP_BUFFER_SIZE);

    softmaxMaxUb = softmaxMaxBuff.Get<COMPUTE_T>();
    softmaxSumUb = softmaxSumBuff.Get<COMPUTE_T>();
    softmaxExpUb = softmaxExpBuff.Get<COMPUTE_T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<COMPUTE_T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<COMPUTE_T>();

    Duplicate(softmaxMaxDefaultUb, (-float(1.0 / 0.0)), SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T));
    Duplicate(softmaxSumDefaultUb, (COMPUTE_T)0.0, SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T));
}

template <typename InputT, bool SOFTMAX_WITH_BRC, bool Partial>
__aicore__ inline void MlaPrefillVector<InputT, SOFTMAX_WITH_BRC, Partial>::SetMSplitInfo(uint32_t mDealSize)
{
    mSplitInfo.nBufferIdx = 0U;
    mSplitInfo.nBufferStartM = 0U;
    mSplitInfo.nBufferDealM = mDealSize;
    // VEC0处理的M大小
    if (mSplitInfo.nBufferDealM <= 16) {
        mSplitInfo.vecDealM = mSplitInfo.nBufferDealM;
    } else {
        mSplitInfo.vecDealM = ((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16;
    }
    mSplitInfo.vecStartM = 0;
    if (GetBlockIdx() % 2 == 1) {
        // VEC1处理的M大小
        mSplitInfo.vecStartM = mSplitInfo.vecDealM;
        mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
    }
}

template <typename InputT, bool SOFTMAX_WITH_BRC, bool Partial>
__aicore__ inline void
MlaPrefillVector<InputT, SOFTMAX_WITH_BRC, Partial>::ProcessVec1SingleBuf(const PrefillInfo& info)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    if constexpr (!SOFTMAX_WITH_BRC) {
        uint32_t alignVal = 32U / sizeof(COMPUTE_T);
        // 向下8/16对齐是因为UB操作起始地址需32B对齐
        mSplitSize = mSplitSize / alignVal * alignVal;
    }
    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(info, i * mSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign,
                             info.actualSingleProcessSInnerSize);
    }

    if (info.isLastS2Loop)
        StoreStats(info);
}

template <typename InputT, bool SOFTMAX_WITH_BRC, bool Partial>
__aicore__ inline void MlaPrefillVector<InputT, SOFTMAX_WITH_BRC, Partial>::DealBmm1ResBaseBlock(
    const PrefillInfo& info, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t computeSize = dealRowCount * columnCount;
    uint64_t inOutGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                             (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow) * columnCount;
    LocalTensor<MM1_OUT_T> mmResUb = inputQue1.AllocTensor<MM1_OUT_T>();
    DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);
    inputQue1.EnQue(mmResUb);
    inputQue1.DeQue<MM1_OUT_T>();

    Muls(mmResUb, mmResUb, params.scale, computeSize);
    Tile::Softcap(mmResUb, tmpBuff1.Get<uint8_t>(), computeSize, params.softcap);
    if (params.window.kind) {
        PipeBarrier<PIPE_V>();
        const auto& task = info.context.task;
        const int64_t delta = int64_t(task.kvLength) - task.queryLength;
        const int64_t next = params.window.kind == 1 ? delta : params.window.right + delta;
        const int64_t pre = params.window.kind == 1 ? int64_t(params.sq) + params.sk : params.window.left - delta;
        // Wide rows need independent repeats: the vector repeat stride is uint8.
        if (columnCount >= 2048) {
            for (uint32_t r = 0; r < dealRowCount; ++r)
                Tile::WindowMask(mmResUb[r * columnCount],
                                 (info.gS1Idx + mSplitInfo.vecStartM + startRow + r) / constInfo.gSize, 1, 1, pre, next,
                                 info.s2Idx * params.nBase, columnCount, actualColumnCount);
        } else
            Tile::WindowMask(mmResUb, info.gS1Idx + mSplitInfo.vecStartM + startRow, dealRowCount, constInfo.gSize, pre,
                             next, info.s2Idx * params.nBase, columnCount, actualColumnCount);
    }
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpBuff1.Get<uint8_t>();
    SoftmaxFlashV2Compute(info, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<KV_T> vec1ResUb = outputQue1.AllocTensor<KV_T>();
    Cast(vec1ResUb, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    outputQue1.EnQue(vec1ResUb);
    outputQue1.DeQue<KV_T>();
    DataCopy(vec1ResGm[inOutGmOffset], vec1ResUb, computeSize);
    outputQue1.FreeTensor(vec1ResUb);

    inputQue1.FreeTensor(mmResUb);
}

template <typename InputT, bool SOFTMAX_WITH_BRC, bool Partial>
__aicore__ inline void MlaPrefillVector<InputT, SOFTMAX_WITH_BRC, Partial>::SoftmaxFlashV2Compute(
    const PrefillInfo& info, LocalTensor<MM1_OUT_T>& mmResUb, LocalTensor<uint8_t>& softmaxTmpUb, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling =
        SoftMaxFlashV2TilingFunc(srcShape, sizeof(COMPUTE_T), sizeof(COMPUTE_T), softmaxTmpUb.GetSize(), true, false);

    LocalTensor<COMPUTE_T> inSumTensor;
    LocalTensor<COMPUTE_T> inMaxTensor;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    if constexpr (SOFTMAX_WITH_BRC) {
        baseOffset = baseOffset * this->brcbNum;
    }
    uint32_t outIdx = info.loop % constInfo.preLoadNum;
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % constInfo.preLoadNum;
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset];
    }
    if constexpr (SOFTMAX_WITH_BRC) {
        SoftmaxFlashV2<SOFTMAX_TYPE, true, true, false, false, PREFILL_SOFTMAX_CFG>(
            mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
            softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        SoftmaxFlashV2<SOFTMAX_TYPE, true, true, false, false, PREFILL_SOFTMAX_NO_BRC>(
            mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
            softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    }
}

template <typename InputT, bool SOFTMAX_WITH_BRC, bool Partial>
__aicore__ inline void
MlaPrefillVector<InputT, SOFTMAX_WITH_BRC, Partial>::ProcessVec2SingleBuf(const PrefillInfo& info)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDimAlign;
    // 为防止后续处理函数溢出或异常，将 mSplitSize 的最大值限制在255
    if (mSplitSize > 255U) {
        mSplitSize = 255U;
    }
    if constexpr (!SOFTMAX_WITH_BRC) {
        uint32_t alignVal = 32U / sizeof(COMPUTE_T);
        // 向下8/16对齐是因为UB操作起始地址需32B对齐
        mSplitSize = mSplitSize / alignVal * alignVal;
    }
    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, i * mSplitSize, dealSize, constInfo.headDimAlign, constInfo.headDim);
    }
}

template <typename InputT, bool SOFTMAX_WITH_BRC, bool Partial>
__aicore__ inline void MlaPrefillVector<InputT, SOFTMAX_WITH_BRC, Partial>::DealBmm2ResBaseBlock(
    const PrefillInfo& info, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t mStart = mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    if constexpr (SOFTMAX_WITH_BRC) {
        baseOffset = baseOffset * this->brcbNum;
    }
    uint64_t inOutBaseOffset = mStart * columnCount;
    uint64_t srcGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;

    LocalTensor<MM2_OUT_T> bmm2ResUb = inputQue1.AllocTensor<MM2_OUT_T>();
    DataCopy(bmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);
    inputQue1.EnQue(bmm2ResUb);
    inputQue1.DeQue<MM2_OUT_T>();

    // 除第一个循环外，均需要更新中间计算结果
    if (!info.isFirstSInnerLoop) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        LocalTensor<COMPUTE_T> bmm2ResPreUb = inputQue1.AllocTensor<COMPUTE_T>();
        uint64_t vec2ResGmOffset = ((info.loop - 1) % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset], vec2ComputeSize);
        inputQue1.EnQue(bmm2ResPreUb);

        inputQue1.DeQue<COMPUTE_T>();
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t idx = info.loop % constInfo.preLoadNum;

        if constexpr (SOFTMAX_WITH_BRC) {
            Tile::RowMuls<COMPUTE_T>(bmm2ResPreUb, bmm2ResPreUb,
                                     softmaxExpUb[idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset],
                                     dealRowCount, columnCount, actualColumnCount);
        } else {
            LocalTensor<COMPUTE_T> tmpExpBrcbResUb = tmpBuff1.Get<COMPUTE_T>();
            Brcb(tmpExpBrcbResUb, softmaxExpUb[idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset],
                 (dealRowCount + this->brcbNum - 1) / this->brcbNum, {1, this->brcbNum});
            AscendC::PipeBarrier<PIPE_V>();
            Tile::RowMuls<COMPUTE_T>(bmm2ResPreUb, bmm2ResPreUb, tmpExpBrcbResUb, dealRowCount, columnCount,
                                     actualColumnCount);
        }

        AscendC::PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        inputQue1.FreeTensor(bmm2ResPreUb);
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.isLastS2Loop) {
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t idx = info.loop % constInfo.preLoadNum;

        if constexpr (SOFTMAX_WITH_BRC) {
            Tile::RowDivs<COMPUTE_T>(bmm2ResUb, bmm2ResUb,
                                     softmaxSumUb[idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset],
                                     dealRowCount, columnCount, actualColumnCount);
        } else {
            LocalTensor<COMPUTE_T> tmpSumBrcbResUb = tmpBuff1.Get<COMPUTE_T>();
            Brcb(tmpSumBrcbResUb, softmaxSumUb[idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset],
                 (dealRowCount + this->brcbNum - 1) / this->brcbNum, {1, this->brcbNum});
            AscendC::PipeBarrier<PIPE_V>();
            Tile::RowDivs<COMPUTE_T>(bmm2ResUb, bmm2ResUb, tmpSumBrcbResUb, dealRowCount, columnCount,
                                     actualColumnCount);
        }

        AscendC::PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(info, bmm2ResUb, mStart, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<COMPUTE_T> tmpBmm2Res = outputQue1.AllocTensor<COMPUTE_T>();
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(tmpBmm2Res);
        outputQue1.DeQue<COMPUTE_T>();
        uint64_t vec2ResGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vec2ResGmOffset], tmpBmm2Res, vec2ComputeSize);

        outputQue1.FreeTensor(tmpBmm2Res);
    }

    inputQue1.FreeTensor(bmm2ResUb);
}

template <typename InputT, bool SOFTMAX_WITH_BRC, bool Partial>
__aicore__ inline void MlaPrefillVector<InputT, SOFTMAX_WITH_BRC, Partial>::ComputeVec1(const PrefillInfo& info)
{
    SetMSplitInfo(info.actMBaseSize);
    CrossCoreWaitFlag(constInfo.syncC1V1);
    ProcessVec1SingleBuf(info);
    CrossCoreSetFlag<PrefillConstants::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);
}

template <typename InputT, bool SOFTMAX_WITH_BRC, bool Partial>
__aicore__ inline void MlaPrefillVector<InputT, SOFTMAX_WITH_BRC, Partial>::ComputeVec2(const PrefillInfo& info)
{
    SetMSplitInfo(info.actMBaseSize);
    CrossCoreWaitFlag(constInfo.syncC2V2);
    ProcessVec2SingleBuf(info);
}

template <typename InputT, bool SOFTMAX_WITH_BRC, bool Partial>
__aicore__ inline void MlaPrefillVector<InputT, SOFTMAX_WITH_BRC, Partial>::StoreStats(const PrefillInfo& info)
{
    auto sum = softmaxSumUb[(info.loop % 2) * 512], maximum = softmaxMaxUb[(info.loop % 2) * 512];
    const uint32_t rows = mSplitInfo.vecDealM, start = mSplitInfo.vecStartM;
    auto s = tmpBuff1.Get<float>(), m = s[4096];
    if constexpr (SOFTMAX_WITH_BRC) {
        DataCopy(s, sum, rows * 8);
        DataCopy(m, maximum, rows * 8);
    } else {
        Brcb(s, sum, (rows + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        Brcb(m, maximum, (rows + 7) / 8, {1, 8});
    }
    PipeBarrier<PIPE_V>();
    if (Partial && info.context.partial) {
        auto tmp = outputQue2.AllocTensor<float>();
        DataCopy(tmp, s, rows * 8);
        outputQue2.EnQue(tmp);
        outputQue2.DeQue<float>();
        DataCopy(lseSumFdGm[(info.context.partialRow + start) * 8], tmp, rows * 8);
        outputQue2.FreeTensor(tmp);
        tmp = outputQue2.AllocTensor<float>();
        DataCopy(tmp, m, rows * 8);
        outputQue2.EnQue(tmp);
        outputQue2.DeQue<float>();
        DataCopy(lseMaxFdGm[(info.context.partialRow + start) * 8], tmp, rows * 8);
        outputQue2.FreeTensor(tmp);
    } else {
        auto lse = s[2048];
        Tile::ComputeSoftMaxLse(lse, s, m, rows);
        for (uint32_t r = 0; r < rows; ++r)
            if (!Valid(info, start + r))
                Duplicate(lse[r * 8], -float(1.0 / 0.0), 8);
        PipeBarrier<PIPE_V>();
        auto tmp = outputQue2.AllocTensor<float>();
        DataCopy(tmp, lse, rows * 8);
        outputQue2.EnQue(tmp);
        outputQue2.DeQue<float>();
        for (uint32_t r = 0; r < rows; ++r) {
            const uint64_t offset = LseRow(info, start + r);
            DataCopyPad(outputLse[offset], tmp[r * 8], DataCopyExtParams{1, 4, 0, 0, 0});
        }
        outputQue2.FreeTensor(tmp);
    }
}
template <typename InputT, bool SOFTMAX_WITH_BRC, bool Partial>
__aicore__ inline void
MlaPrefillVector<InputT, SOFTMAX_WITH_BRC, Partial>::Bmm2ResCopyOut(const PrefillInfo& info, LocalTensor<float>& value,
                                                                    uint32_t row, uint32_t startRow, uint32_t rows,
                                                                    uint32_t columns, uint32_t actualColumns)
{
    if (Partial && info.context.partial) {
        auto tmp = outputQue1.AllocTensor<float>();
        DataCopy(tmp, value, rows * 128);
        outputQue1.EnQue(tmp);
        outputQue1.DeQue<float>();
        DataCopy(accumOutGm[(info.context.partialRow + row) * 128], tmp, rows * 128);
        outputQue1.FreeTensor(tmp);
    } else {
        for (uint32_t r = 0; r < rows; ++r)
            if (!Valid(info, row + r))
                Duplicate(value[r * 128], 0.0f, 128);
        PipeBarrier<PIPE_V>();
        auto tmp = outputQue1.AllocTensor<InputT>();
        if constexpr (IsSameType<InputT, bfloat16_t>::value)
            Cast(tmp, value, RoundMode::CAST_RINT, rows * 128);
        else if (params.softcap != 0.0f)
            Cast(tmp, value, RoundMode::CAST_RINT, rows * 128);
        else
            Cast(tmp, value, RoundMode::CAST_ROUND, rows * 128);
        outputQue1.EnQue(tmp);
        outputQue1.DeQue<InputT>();
        for (uint32_t r = 0; r < rows;) {
            uint32_t count = Min(rows - r, constInfo.gSize - (info.context.task.rowStart + row + r) % constInfo.gSize);
            DataCopy(output[OutputRow(info, row + r) * 128], tmp[r * 128], count * 128);
            r += count;
        }
        outputQue1.FreeTensor(tmp);
    }
}
} // namespace flash_attn::mla::Block
