/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0.
// attention/fused_infer_attention_score/op_kernel/arch22/fia_block_vec_nonquant_mla.h
// Contiguous BSND/compact TND FP16/BF16, Hkv=1; optional independent FD partial writeback.
// Optional compile-time causal/band masks; existing specializations retain their defaults.
// TND O is [Tq,Hq,512]; FA3 LSE is [Hq,Tq] (source LSE is [Tq,Hq]). Scale is launch data.
// SoftmaxFlashV2, aMLA, all UB partitions, casts and events follow the source.
// Cross-S2 nUpdate retains the source atomic int32 correction of FP32 PV bits.
// This also retains its documented zero-cancellation limitation.
#pragma once
#include "mla_cube_block.hpp"
#include "mla_vector_ops_tile.hpp"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
namespace flash_attn::mla::Block {
struct VectorParams : CubeParams {
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768, BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096, BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr bool softmaxLseFlag = true;
    uint32_t subBlockNum;
    float scaleValue;
    float softcap = 0.0f;
    bool attenMaskFlag = false;
};
constexpr SoftmaxConfig FIA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

template <typename InputT, bool IsTnd = false, bool IsCausal = false, bool IsWindow = false, bool IsPartial = false>
class MlaVector {
  public:
    using T = float;
    using KV_T = InputT;
    using OUT_T = InputT;
    using COMPUTE_T = float;
    using UPDATE_T = float;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;
    static constexpr bool SOFTMAX_WITH_BRC = false;
    __aicore__ inline void Init(VectorParams params, GM_ADDR output, GM_ADDR lse)
    {
        constInfo = params;
        attentionOutGm.SetGlobalBuffer((__gm__ InputT*)output);
        softmaxLseGm.SetGlobalBuffer((__gm__ float*)lse);
    }
    // Source ComputeLogSumExpAndCopyToGm/Bmm2FDDataCopyOut, with a checked
    // slot supplied by the independent Kernel rather than CalcAccumOffset.
    __aicore__ inline void InitPartial(GM_ADDR out, GM_ADDR sum, GM_ADDR maximum, uint64_t rowOffset)
    {
        partialOut.SetGlobalBuffer((__gm__ float*)out + rowOffset * 512);
        partialSum.SetGlobalBuffer((__gm__ float*)sum + rowOffset * 8);
        partialMax.SetGlobalBuffer((__gm__ float*)maximum + rowOffset * 8);
    }
    __aicore__ inline void CopyPartialStats(const RunInfo& info, const MSplitInfo& split);
    __aicore__ inline void CopyPartialOut(const RunInfo& info, LocalTensor<T>& out, uint32_t row, uint32_t count);
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm,
                                                GlobalTensor<int32_t> mm2ResInt32Gm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<UPDATE_T> vec2ResGm, GlobalTensor<MM2_OUT_T> mm2ResGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo& info, const MSplitInfo& mSplitInfo,
                                                 LocalTensor<T>& mmResUb, LocalTensor<uint8_t>& softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void AmlaVecCompute(const RunInfo& info, const MSplitInfo& mSplitInfo, LocalTensor<T>& mmResUb,
                                          LocalTensor<uint8_t>& softmaxTmpUb, uint32_t startRow, uint32_t dealRowCount,
                                          uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo& info, const MSplitInfo& mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void ProcessAmlaNupdate(const RunInfo& info, const MSplitInfo& mSplitInfo);
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo& info, const MSplitInfo& mSplitInfo);
    __aicore__ inline void ProcessVec1L(const RunInfo& info);
    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo& info, const MSplitInfo& mSplitInfo);
    __aicore__ inline void ProcessVec2L(const RunInfo& info);
    __aicore__ inline void ProcessVec2Inner(const RunInfo& info, const MSplitInfo& mSplitInfo, uint32_t mStartRow,
                                            uint32_t mDealSize);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo& info, const MSplitInfo& mSplitInfo,
                                              LocalTensor<T>& bmm2ResUb, uint32_t wsMStart, uint32_t startRow,
                                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo& info, const MSplitInfo& mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo& info, LocalTensor<OUT_T>& attenOutUb, uint32_t wsMStart,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void CopySoftmaxLseToGmByLayout(const RunInfo& info, LocalTensor<T>& lseSrc, uint32_t mOffset,
                                                      const MSplitInfo& mSplitInfo);

  private:
    VectorParams constInfo;
    uint32_t pingpongFlag = 0;
    GlobalTensor<float> mm1ResGm, mm2ResGm, vec2ResGm, softmaxLseGm;
    GlobalTensor<InputT> vec1ResGm, attentionOutGm;
    GlobalTensor<float> partialOut, partialSum, partialMax;
    GlobalTensor<int32_t> mm2ResInt32Gm;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2, SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4, SYNC_INPUT_BUF2_PONG_FLAG = 5;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4, SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = 32768, INPUT2_BUFFER_OFFSET = 8192;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = 1024, LSE_TMP_BUFFER_SIZE = 8192;
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = 32768 / sizeof(T);
    static constexpr uint16_t brcbNum = 8;
    static constexpr uint64_t headDim = 512, headDimAlign = 512;
    static constexpr T FLOAT_ZERO = 0.0f, FLOAT_E_SCALAR = 8388608;
    static constexpr T LN2 = 0.6931471805599453094172, RECIP_OF_LN2 = 1 / LN2;
    T SOFTMAX_MIN_NUM = T(-1.0 / 0.0);
    TBuf<> inputBuff1, inputBuff2, outputBuff1, outputBuff2, tmpBuff1, attenMaskTmpBuff;
    TBuf<> nValueBuff, cofValueBuff, aMlaSumBuff, softmaxMaxBuff, softmaxExpBuff, softmaxSumBuff;
    TBuf<> softmaxMaxDefaultBuff, softmaxSumDefaultBuff;
    LocalTensor<T> nValueUb, cofValueUb, aMlaSumUb, softmaxMaxUb, softmaxExpUb, softmaxSumUb;
    LocalTensor<T> softmaxMaxDefaultUb, softmaxSumDefaultUb;
};
template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::InitBuffers(TPipe* pipe)
{
    // queue
    pipe->InitBuffer(inputBuff1, VectorParams::BUFFER_SIZE_BYTE_32K * 2); // 2:pingpong
    pipe->InitBuffer(inputBuff2, VectorParams::BUFFER_SIZE_BYTE_8K * 2);  // 2:pingpong
    pipe->InitBuffer(outputBuff1, VectorParams::BUFFER_SIZE_BYTE_32K);
    if (constInfo.subBlockNum == 1) { // CV1:1场景，vecDealM 变大一倍，需要的buffer变大一倍
        pipe->InitBuffer(outputBuff2, VectorParams::BUFFER_SIZE_BYTE_8K);
    } else {
        pipe->InitBuffer(outputBuff2, VectorParams::BUFFER_SIZE_BYTE_4K);
    }

    // tmpBuff
    pipe->InitBuffer(tmpBuff1, VectorParams::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(attenMaskTmpBuff, VectorParams::BUFFER_SIZE_BYTE_8K);

    // M_MAX = 512/2vector = 256, 256 * sizeof(T) * N_Buffer
    pipe->InitBuffer(nValueBuff, VectorParams::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(cofValueBuff, VectorParams::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(aMlaSumBuff, VectorParams::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxBuff, VectorParams::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, VectorParams::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, VectorParams::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, VectorParams::BUFFER_SIZE_BYTE_1K);
    pipe->InitBuffer(softmaxSumDefaultBuff, VectorParams::BUFFER_SIZE_BYTE_1K);

    nValueUb = nValueBuff.Get<T>();
    cofValueUb = cofValueBuff.Get<T>();
    aMlaSumUb = aMlaSumBuff.Get<T>();

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
    Duplicate(softmaxSumDefaultUb, FLOAT_ZERO, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm, GlobalTensor<int32_t> mm2ResInt32Gm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->mm2ResInt32Gm = mm2ResInt32Gm;
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void
MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::InitVec2GlobalTensor(GlobalTensor<UPDATE_T> vec2ResGm,
                                                                              GlobalTensor<MM2_OUT_T> mm2ResGm)
{
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::FreeEventID()
{
    CrossCoreWaitFlag(constInfo.syncC2V1);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::SoftmaxFlashV2Compute(
    const RunInfo& info, const MSplitInfo& mSplitInfo, LocalTensor<T>& mmResUb, LocalTensor<uint8_t>& softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling =
        SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);

    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
    }
    SoftmaxFlashV2<T, true, true, false, false, FIA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
        mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
        softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::AmlaVecCompute(
    const RunInfo& info, const MSplitInfo& mSplitInfo, LocalTensor<T>& mmResUb, LocalTensor<uint8_t>& softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t calCount = dealRowCount;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    // compute n(i)
    LocalTensor<T> nTmp = softmaxTmpUb.template ReinterpretCast<T>();
    LocalTensor<T> nUpdateTmp = nTmp[SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Muls(nTmp, softmaxMaxUb[softmaxOutOffset], ((T)(-1.0)) * RECIP_OF_LN2, calCount);

    AscendC::PipeBarrier<PIPE_V>();
    Cast(nTmp, nTmp, RoundMode::CAST_ROUND, calCount);
    AscendC::PipeBarrier<PIPE_V>();

    uint32_t prOutIdx = (info.loop - 1) % (constInfo.preLoadNum);
    uint32_t PreSoftmaxOutOffset = prOutIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    // n(i) - n(i-1)
    if (info.isFirstSInnerLoop) {
        Duplicate(nUpdateTmp, FLOAT_ZERO, calCount); // n1=n0
    } else {
        Sub(nUpdateTmp, nTmp, nValueUb[PreSoftmaxOutOffset], calCount);
    }
    AscendC::PipeBarrier<PIPE_V>();
    // update n(i), DataCopy not support when calCount is not align 32B, so use Adds
    Adds(nValueUb[softmaxOutOffset], nTmp, FLOAT_ZERO, calCount);
    AscendC::PipeBarrier<PIPE_V>();

    // update softmax res
    LocalTensor<T> nUpdateTmp2 = nTmp[2 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    LocalTensor<KV_T> nTmp_KvT = nTmp[3 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)].template ReinterpretCast<KV_T>();
    LocalTensor<T> tmpCofUb = nTmp[4 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    LocalTensor<T> epsUb = nTmp[5 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Muls(nUpdateTmp2, softmaxMaxUb[softmaxOutOffset], RECIP_OF_LN2, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Add(nTmp, nUpdateTmp2, nTmp, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(nTmp, nTmp, LN2, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(nTmp, nTmp, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(nTmp_KvT, nTmp, RoundMode::CAST_ROUND, calCount); // fp32->fp16/bf16
    AscendC::PipeBarrier<PIPE_V>();
    Cast(nUpdateTmp2, nTmp_KvT, RoundMode::CAST_NONE, calCount); // fp16/bf16->fp32
    AscendC::PipeBarrier<PIPE_V>();
    if (info.isLastS2Loop) {
        Mul(aMlaSumUb[softmaxOutOffset], softmaxSumUb[softmaxOutOffset], nUpdateTmp2, calCount);
    }

    LocalTensor<T> nTmp3 = nTmp[6 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Brcb(nTmp3, nUpdateTmp2, (dealRowCount + 7) / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    Tile::RowMuls(mmResUb, mmResUb, nTmp3, dealRowCount, columnCount, actualColumnCount);

    Div(tmpCofUb, nTmp, nUpdateTmp2, calCount); // cof(i)=tmpS32/tmpS16
    if (info.isFirstSInnerLoop) {
        Duplicate(cofValueUb[softmaxOutOffset], (T)1.0, calCount); // cof_0=1
        AscendC::PipeBarrier<PIPE_V>();
        Div(epsUb, cofValueUb[softmaxOutOffset], tmpCofUb, calCount); // 1 / cof(i)
    } else {
        AscendC::PipeBarrier<PIPE_V>();
        Div(epsUb, cofValueUb[PreSoftmaxOutOffset], tmpCofUb, calCount); // cof(i - 1) / cof(i)
    }
    AscendC::PipeBarrier<PIPE_V>();

    Adds(cofValueUb[softmaxOutOffset], tmpCofUb, FLOAT_ZERO, calCount); // store cof(i)
    Adds(epsUb, epsUb, (T)(-1.0), calCount);                            // cof(i - 1) / cof(i) - 1
    AscendC::PipeBarrier<PIPE_V>();
    Muls(epsUb, epsUb, (T)1.5, calCount); // (cof(i - 1) - cof(i)) / cof(i) * 1.5

    Maxs(nUpdateTmp, nUpdateTmp, (T)(-30.0), calCount); // N = max(n(i) - n(i-1), -30)
    AscendC::PipeBarrier<PIPE_V>();
    // Zero scale gives exact uniform weights. The source epsilon would
    // perturb their sum across KV blocks and flip output midpoint rounding.
    // Keep the original aMLA correction for nonuniform scores and cap=0.
    const T epsilon = constInfo.softcap != 0.0f && constInfo.scaleValue == 0.0f ? FLOAT_ZERO : (T)0.000001;
    Adds(epsUb, epsUb, epsilon, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Add(nUpdateTmp, nUpdateTmp, epsUb, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(nUpdateTmp, nUpdateTmp, FLOAT_E_SCALAR, calCount); // N = N * pow(2, 23)
    AscendC::PipeBarrier<PIPE_V>();

    // nUpdate int32 out
    LocalTensor<int32_t> tmQue = outputBuff2.Get<int32_t>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    LocalTensor<int32_t> nInt32Out = tmQue[startRow]; // 缓存nUpdate

    Cast(nInt32Out, nUpdateTmp, RoundMode::CAST_ROUND, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::DealBmm1ResBaseBlock(
    const RunInfo& info, const MSplitInfo& mSplitInfo, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
    uint32_t actualColumnCount)
{
    uint32_t computeSize = dealRowCount * columnCount;
    uint64_t inOutGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                             (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow) * columnCount;
    LocalTensor<MM1_OUT_T> mmResUb = inputBuff1.Get<MM1_OUT_T>();
    mmResUb = mmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    Muls(mmResUb, mmResUb, constInfo.scaleValue, dealRowCount * columnCount);
    Tile::Softcap(mmResUb, tmpBuff1.Get<uint8_t>(), computeSize, constInfo.softcap);

    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (IsCausal || IsWindow) {
        if (constInfo.attenMaskFlag) {
            const uint32_t row = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
            if constexpr (IsWindow) {
                Tile::WindowMask(mmResUb, row, dealRowCount, constInfo.gSize, info.preToken, info.nextToken,
                                 info.s2BatchOffset, columnCount, actualColumnCount);
            } else {
                Tile::CausalMask(mmResUb, row, dealRowCount, constInfo.gSize, info.nextToken, info.s2BatchOffset,
                                 columnCount, actualColumnCount);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    LocalTensor<T> tmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount,
                          actualColumnCount);

    AscendC::PipeBarrier<PIPE_V>();
    AmlaVecCompute(info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);

    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<KV_T> tmpMMResCastTensor = outputBuff1.Get<KV_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    if (constInfo.softcap != 0.0f) {
        // Normalize the actual input-dtype P consumed by the Cube. Using its
        // pre-cast sum amplifies weight-rounding bias for nearly constant V.
        AscendC::PipeBarrier<PIPE_V>();
        Cast(mmResUb, tmpMMResCastTensor, AscendC::RoundMode::CAST_NONE, computeSize);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t base = mSplitInfo.nBufferStartM / 2 + startRow;
        const uint32_t current = (info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + base;
        LocalTensor<T> sum = aMlaSumUb[current];
        Tile::RowSum(sum, mmResUb, dealRowCount, columnCount, actualColumnCount);
        if (!info.isFirstSInnerLoop) {
            const uint32_t previous =
                ((info.loop - 1) % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + base;
            LocalTensor<T> scaledSum = tmpBuff1.Get<T>();
            LocalTensor<uint8_t> nonzero = attenMaskTmpBuff.Get<uint8_t>();
            AscendC::PipeBarrier<PIPE_V>();
            CompareScalar(nonzero, aMlaSumUb[previous], FLOAT_ZERO, CMPMODE::NE, (dealRowCount + 63) / 64 * 64);
            AscendC::PipeBarrier<PIPE_V>();
            Add(scaledSum.template ReinterpretCast<int32_t>(), aMlaSumUb[previous].template ReinterpretCast<int32_t>(),
                outputBuff2.Get<int32_t>()[startRow], dealRowCount);
            AscendC::PipeBarrier<PIPE_V>();
            Select(scaledSum, nonzero, scaledSum, FLOAT_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, dealRowCount);
            AscendC::PipeBarrier<PIPE_V>();
            Add(sum, sum, scaledSum, dealRowCount);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[inOutGmOffset], tmpMMResCastTensor, computeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void
MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::ProcessAmlaNupdate(const RunInfo& info,
                                                                            const MSplitInfo& mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    if (info.isFirstSInnerLoop) {
        return;
    }

    LocalTensor<int32_t> nUpdateTensor = outputBuff2.Get<int32_t>(); // shape:1/2*s1*g
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);

    constexpr uint32_t dGroupSize = 128U;
    constexpr uint32_t mSplitSize =
        64U; // tmpQue size 32KB，一次只能处理64个N，最大保存的数据大小：64*128*sizeof(int32)
    constexpr uint32_t ONE_BLOCK_SIZE = 32U; // 32B
    uint32_t subMSize = Tile::Align(mSplitInfo.vecDealM, 16U);
    constexpr uint16_t elementPerBlock =
        ONE_BLOCK_SIZE / sizeof(int32_t); // 单个datablock的元素数，int32_t类型的为32/4=8

    uint32_t loopCount = (subMSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = subMSize - (loopCount - 1) * mSplitSize; // 尾块
    for (uint32_t loop = 0, processMSize = mSplitSize; loop < loopCount; loop++) {
        if (loop == (loopCount - 1)) {
            processMSize = tailSplitSize;
        }

        LocalTensor<int32_t> tmpQue = outputBuff1.Get<int32_t>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        // (m,1)单次brcb扩充成(m,8), 重复16次, 扩充为(m,128)
        for (uint32_t i = 0; i < dGroupSize / elementPerBlock; i++) {
            Brcb(tmpQue[i * elementPerBlock], nUpdateTensor[loop * mSplitSize],
                 static_cast<uint8_t>((processMSize + elementPerBlock - 1) / elementPerBlock),
                 {static_cast<uint16_t>(
                      dGroupSize / elementPerBlock),   // 单次迭代内，目的操作数不同datablock间地址步长,单位为datablock
                  static_cast<uint16_t>(dGroupSize)}); // 相邻迭代间，目的操作数相同datablock地址步长
        }

        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);

        uint64_t baseoffset = (info.bn2IdxInCurCore % constInfo.preLoadNum) * constInfo.bmm2ResUbSize +
                              (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + loop * mSplitSize) * headDim;

        SetAtomicAdd<int32_t>();
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = static_cast<uint16_t>(processMSize);
        dataCopyParams.blockLen = dGroupSize * sizeof(int32_t) / ONE_BLOCK_SIZE; // 每个block是128个元素，单位为32B
        dataCopyParams.srcStride = 0; // 前面一个数据块的尾与后面数据块的头的间隔
        dataCopyParams.dstStride =
            static_cast<uint16_t>((headDim - dGroupSize) * sizeof(int32_t) / ONE_BLOCK_SIZE); // 单位为32B
        for (uint32_t i = 0; i < headDim / dGroupSize; i++) {                                 // 4=512/128
            DataCopy(mm2ResInt32Gm[baseoffset + i * dGroupSize], tmpQue, dataCopyParams);
        }

        SetAtomicNone();
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void
MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::ProcessVec1SingleBuf(const RunInfo& info,
                                                                              const MSplitInfo& mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
    mSplitSize = mSplitSize / 8 * 8;

    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }

    const uint32_t MAX_M_SPLIT_SIZE = 128; // CV1:1场景，M轴方向最大支持128
    mSplitSize = (constInfo.subBlockNum == 1 && mSplitSize > MAX_M_SPLIT_SIZE) ? MAX_M_SPLIT_SIZE : mSplitSize;

    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(info, mSplitInfo, i * mSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign,
                             info.actualSingleProcessSInnerSize);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::ProcessVec1L(const RunInfo& info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM
                                                              : (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (constInfo.subBlockNum == 1) { // CV1:1场景
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM;
        } else if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }

        CrossCoreWaitFlag(constInfo.syncC1V1);
        // vec1 compute
        ProcessVec1SingleBuf(info, mSplitInfo);
        CrossCoreSetFlag<VectorParams::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);

        CrossCoreWaitFlag(constInfo.syncC2V1);
        // add nUpdate to mm2ResGm
        ProcessAmlaNupdate(info, mSplitInfo);
        CrossCoreSetFlag<VectorParams::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1NupdateC2);

        // move lse for flash decode
        if (info.isLastS2Loop) {
            uint32_t outIdx = info.loop % (constInfo.preLoadNum);
            LocalTensor sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(COMPUTE_T)];
            LocalTensor maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(COMPUTE_T)];
            if (IsPartial && info.isFd) {
                CopyPartialStats(info, mSplitInfo);
            } else if (constInfo.softmaxLseFlag) {
                if (mSplitInfo.vecDealM == 0) {
                    continue;
                }

                LocalTensor<COMPUTE_T> totalLseUb = tmpBuff1.Get<COMPUTE_T>(LSE_TMP_BUFFER_SIZE);
                if constexpr (!SOFTMAX_WITH_BRC) {
                    LocalTensor<COMPUTE_T> lseSumUb =
                        tmpBuff1.GetWithOffset<COMPUTE_T>(LSE_TMP_BUFFER_SIZE, LSE_TMP_BUFFER_SIZE);
                    LocalTensor<COMPUTE_T> lseMaxUb =
                        tmpBuff1.GetWithOffset<COMPUTE_T>(LSE_TMP_BUFFER_SIZE, LSE_TMP_BUFFER_SIZE * 2);
                    Brcb(lseSumUb, sumTensor[mSplitInfo.nBufferStartM / 2],
                         (mSplitInfo.vecDealM + brcbNum - 1) / brcbNum, {1, brcbNum});
                    AscendC::PipeBarrier<PIPE_V>();
                    Brcb(lseMaxUb, maxTensor[mSplitInfo.nBufferStartM / 2],
                         (mSplitInfo.vecDealM + brcbNum - 1) / brcbNum, {1, brcbNum});
                    AscendC::PipeBarrier<PIPE_V>();
                    Tile::ComputeSoftMaxLse(totalLseUb, lseSumUb, lseMaxUb, mSplitInfo.vecDealM);
                } else {
                    Tile::ComputeSoftMaxLse(totalLseUb, sumTensor, maxTensor, mSplitInfo.vecDealM);
                }

                if constexpr (IsCausal || IsWindow) {
                    AscendC::PipeBarrier<PIPE_V>();
                    const uint32_t row = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;
                    const uint32_t empty =
                        Tile::CausalEmptyRows(row, mSplitInfo.vecDealM, constInfo.gSize, info.nextToken);
                    if (empty > 0) {
                        Duplicate(totalLseUb, -float(1.0 / 0.0), empty * brcbNum);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }
                LocalTensor<COMPUTE_T> tmpLseResCastTensor = outputBuff2.Get<COMPUTE_T>();
                WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
                DataCopy(tmpLseResCastTensor, totalLseUb, mSplitInfo.vecDealM * brcbNum);

                SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
                WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);

                uint32_t mOffset = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;
                CopySoftmaxLseToGmByLayout(info, tmpLseResCastTensor, mOffset, mSplitInfo);
                SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
            }
        }
    }
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void
MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::ProcessVec2SingleBuf(const RunInfo& info,
                                                                              const MSplitInfo& mSplitInfo)
{
    if (!info.isLastS2Loop) {
        return;
    }
    if (mSplitInfo.vecDealM == 0) {
        return;
    }

    ProcessVec2Inner(info, mSplitInfo, 0, mSplitInfo.vecDealM);
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::ProcessVec2L(const RunInfo& info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM
                                                              : (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (constInfo.subBlockNum == 1) { // CV1:1场景
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM;
        } else if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }
        CrossCoreWaitFlag(constInfo.syncC2V2);
        ProcessVec2SingleBuf(info, mSplitInfo);
    }
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::ProcessVec2Inner(
    const RunInfo& info, const MSplitInfo& mSplitInfo, uint32_t mStartRow, uint32_t mDealSize)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
    if (mSplitSize > mDealSize) {
        mSplitSize = mDealSize;
    }

    uint32_t loopCount = (mDealSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mDealSize - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, i * mSplitSize + mStartRow, dealSize, headDimAlign, headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::Bmm2CastAndCopyOut(
    const RunInfo& info, const MSplitInfo& mSplitInfo, LocalTensor<T>& bmm2ResUb, uint32_t wsMStart, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (IsCausal || IsWindow) {
        const uint32_t empty =
            Tile::CausalEmptyRows(info.gS1Idx + wsMStart, dealRowCount, constInfo.gSize, info.nextToken);
        if (empty > 0) {
            Duplicate(bmm2ResUb, FLOAT_ZERO, empty * columnCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else if (constInfo.softcap != 0.0f) {
        // Softcap has no source oracle. Match the mathematical output's
        // ties-to-even cast; retain the pinned source's rounding when off.
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::DealBmm2ResBaseBlock(
    const RunInfo& info, const MSplitInfo& mSplitInfo, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
    uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t mStart = mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    uint64_t srcGmOffset =
        (info.bn2IdxInCurCore % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + mStart * columnCount;
    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);

    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    // 将绝对值大于1e10的数置为0
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);
    LocalTensor<T> absBmm2ResUb = bmm2ResUb.template ReinterpretCast<T>();
    Abs(absBmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> cmpMaskUb = absBmm2ResUb.template ReinterpretCast<uint8_t>();
    CompareScalar(cmpMaskUb, absBmm2ResUb, (T)1e10, CMPMODE::LE, vec2ComputeSize);
    AscendC::PipeBarrier<PIPE_V>();
    Select(tmpBmm2ResUb, cmpMaskUb, tmpBmm2ResUb, FLOAT_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, vec2ComputeSize);
    AscendC::PipeBarrier<PIPE_V>();

    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t idx = info.loop % (constInfo.preLoadNum);
    LocalTensor<T> tmpSumUb = attenMaskTmpBuff.Get<T>(); // sumUb用临时内存 16 * 32B  = 512B
    Brcb(tmpSumUb, aMlaSumUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset], (dealRowCount + 7) / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    Tile::RowDivs(bmm2ResUb, tmpBmm2ResUb, tmpSumUb, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    if (IsPartial && info.isFd) {
        // The pinned FD branch stores RowDivs' FP32 result without the final
        // FP16/BF16 cast or public empty-row adaptation.
        CopyPartialOut(info, bmm2ResUb, mStart, dealRowCount);
    } else {
        Bmm2CastAndCopyOut(info, mSplitInfo, bmm2ResUb, mStart, startRow, dealRowCount, columnCount, actualColumnCount);
    }
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::Bmm2DataCopyOutTrans(
    const RunInfo& info, LocalTensor<OUT_T>& attenOutUb, uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
    uint32_t actualColumnCount)
{
    // Source CopyAttenOutUbToGm<InputT, BSNGD, S1G>, ROW_PAD=true.
    // Preserve the per-query head/middle/tail copies even with Hkv=1.
    uint32_t begin = info.gS1Idx + wsMStart;
    uint32_t sBegin = begin / constInfo.gSize, gBegin = begin % constInfo.gSize;
    uint32_t sEnd = (begin + dealRowCount) / constInfo.gSize;
    uint32_t gEnd = (begin + dealRowCount) % constInfo.gSize;
    const uint64_t tokenStart = IsTnd ? info.queryStart : uint64_t(info.bIdx) * constInfo.qSeqSize;
    uint64_t base = (tokenStart + sBegin) * constInfo.gSize * headDim;
    uint32_t headSize = sBegin == sEnd ? gEnd - gBegin : constInfo.gSize - gBegin;
    uint64_t gmOffset = base + gBegin * headDim, ubOffset = 0;
    DataCopyExtParams params;
    params.blockCount = headSize;
    params.blockLen = actualColumnCount * sizeof(OUT_T);
    params.srcStride = (columnCount - actualColumnCount) / (32 / sizeof(OUT_T));
    params.dstStride = 0;
    DataCopyPad(attentionOutGm[gmOffset], attenOutUb[ubOffset], params);
    if (sEnd - sBegin >= 1) {
        gmOffset = base + constInfo.gSize * headDim;
        ubOffset = uint64_t(headSize) * columnCount;
        for (uint32_t s = sBegin + 1; s < sEnd; ++s) {
            params.blockCount = constInfo.gSize;
            DataCopyPad(attentionOutGm[gmOffset], attenOutUb[ubOffset], params);
            gmOffset += constInfo.gSize * headDim;
            ubOffset += constInfo.gSize * columnCount;
        }
        if (gEnd > 0) {
            params.blockCount = gEnd;
            DataCopyPad(attentionOutGm[gmOffset], attenOutUb[ubOffset], params);
        }
    }
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::CopySoftmaxLseToGmByLayout(
    const RunInfo& info, LocalTensor<T>& lseSrc, uint32_t mOffset, const MSplitInfo& mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t dealCount = mSplitInfo.vecDealM;
    uint64_t bN2Offset = IsTnd ? info.queryStart : uint64_t(info.bIdx) * constInfo.gSize * constInfo.qSeqSize;
    uint32_t startS1Idx = mOffset / constInfo.gSize;
    uint32_t startGIdx = mOffset % constInfo.gSize;
    uint32_t endS1Idx = (mOffset + dealCount - 1) / constInfo.gSize;
    uint32_t endGIdx = (mOffset + dealCount - 1) % constInfo.gSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;
    uint64_t s1LeftPaddingSize = 0;

    for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
        outOffset = bN2Offset + uint64_t(startGIdx) * constInfo.qSeqSize + s1Idx + s1LeftPaddingSize;
        if (s1Idx != endS1Idx) {
            curDealRowCount = constInfo.gSize - startGIdx;
        } else {
            curDealRowCount = endGIdx + 1 - startGIdx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        const uint64_t headGapBytes = uint64_t(constInfo.qSeqSize - 1) * sizeof(float);
        if (IsTnd && headGapBytes > UINT32_MAX) {
            // Tq may reach INT32_MAX; FA3's [Hq,Tq] head stride can exceed
            // the DMA field even though all tensor offsets fit uint64.
            dataCopyParams.blockCount = 1;
            dataCopyParams.dstStride = 0;
            for (uint32_t head = 0; head < curDealRowCount; ++head) {
                DataCopyPad(softmaxLseGm[outOffset + uint64_t(head) * constInfo.qSeqSize], lseSrc[ubOffset + head * 8U],
                            dataCopyParams);
            }
        } else {
            dataCopyParams.dstStride = headGapBytes;
            DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        }
        startGIdx = 0;
        ubOffset += curDealRowCount * 8U;
    }
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void
MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::CopyPartialStats(const RunInfo& info, const MSplitInfo& split)
{
    if (split.vecDealM == 0) {
        return;
    }
    const uint32_t base =
        (info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + split.nBufferStartM / 2;
    const uint64_t offset = (info.partialRow + split.nBufferStartM + split.vecStartM) * 8;
    LocalTensor<T> tmp = outputBuff2.Get<T>();
    WaitFlag<HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    Brcb(tmp, softmaxSumUb[base], (split.vecDealM + 7) / 8, {1, 8});
    SetFlag<HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    DataCopy(partialSum[offset], tmp, split.vecDealM * 8);
    SetFlag<HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    Brcb(tmp, softmaxMaxUb[base], (split.vecDealM + 7) / 8, {1, 8});
    SetFlag<HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    DataCopy(partialMax[offset], tmp, split.vecDealM * 8);
    SetFlag<HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename InputT, bool IsTnd, bool IsCausal, bool IsWindow, bool IsPartial>
__aicore__ inline void
MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial>::CopyPartialOut(const RunInfo& info, LocalTensor<T>& out,
                                                                        uint32_t row, uint32_t count)
{
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, out, count * 512);
    SetFlag<HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopyPad(partialOut[(info.partialRow + row) * 512], tmp,
                DataCopyExtParams{uint16_t(count), 512 * sizeof(T), 0, 0, 0});
    SetFlag<HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

} // namespace flash_attn::mla::Block
