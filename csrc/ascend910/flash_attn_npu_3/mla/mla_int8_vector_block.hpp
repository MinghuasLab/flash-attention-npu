/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0.
// attention/incre_flash_attention/op_kernel/arch22/.
// Only INT8 content / BF16 RoPE and output, Hkv=1, 512+64, ND PA.
#pragma once
#include <limits>
#include "mla_int8_cube_block.hpp"
#include "mla_int8_merge_block.hpp"
#include "adv_api/activation/softmax.h"
namespace flash_attn::mla::Int8 {
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using namespace AttentionCommonFlashDecode;

#define MLA_INT8_ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

namespace {
struct TaskContext {
    uint32_t bidx;
    uint32_t gidx;
    uint32_t s1idx;
    uint32_t s2idx;
    uint32_t s2loops;
    uint32_t s2SizeTail;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t tndIsS2SplitCore; // TND格式，[B, N2, S1]三根轴确定后，S2是否被切到了多核上
    uint32_t tndCoreStartKVSplitPos;
    uint32_t isFirstLoop;
    uint64_t actS1Size = 1;
    bool isValid = false;
    bool isLastTask = false;
};

} // namespace

template <typename IFAT>
class MlaInt8Vector {
  public:
    __aicore__ inline MlaInt8Vector(){};
    __aicore__ inline void SetCacheLayout(const MlaPagedCacheLayout& layout)
    {
        if ASCEND_IS_AIC {
            matmulService.SetCacheLayout(layout);
        }
    }
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR attenMask,
                                GM_ADDR blockTable, GM_ADDR attentionOut, GM_ADDR softmaxLse, GM_ADDR workspace,
                                const Int8TilingParams* tiling, TPipe* tPipe);
    __aicore__ inline void InitQuant(GM_ADDR dequantScaleQuery, GM_ADDR keyAntiquantScale, GM_ADDR valueAntiquantScale);
    // Public FA3 lengths are int32; bind them before Init.
    __aicore__ inline void SetInt32Lengths(GM_ADDR lengths)
    {
        publicLengthsPresent = lengths != nullptr;
        lengths32.SetGlobalBuffer((__gm__ int32_t*)lengths);
    }
    __aicore__ inline void SetPublicQuery(GM_ADDR prefix)
    {
        actualSeqLengthsGmQ.SetPublic(prefix);
    }
    __aicore__ inline void InitPublicEmptyTail()
    {
        if constexpr (BALANCE) {
            if ASCEND_IS_AIV {
                if (aiCoreIdx + 1 == usedCoreNum && GetSubBlockIdx() == 0) {
                    // Source balanced tiling can stop before trailing empty Q/KV
                    // segments in either layout. Initialize after the numerical
                    // pipeline drains; extending its B endpoint suppresses flush.
                    for (uint32_t b = bEnd + 1; b < batchSize; ++b) {
                        GetActualSeqLen(b);
                        if (actS1Size && curActualSeqLen == 0)
                            DealActSeqLenIsZero(b, 0);
                    }
                }
            }
        }
    }
    __aicore__ inline void SetPublicPartials(GM_ADDR out, GM_ADDR lse)
    {
        publicPartials = out != nullptr;
        publicOut.SetGlobalBuffer((__gm__ float*)out);
        publicLse.SetGlobalBuffer((__gm__ float*)lse);
    }

    // 中间计算数据类型为float，高精度模式
    using T = float;

    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    using OUT_T = typename IFAT::outputType;
    using ORIGIN_T = typename IFAT::orginalType;
    static constexpr bool FLASH_DECODE = IFAT::flashDecode;
    static constexpr LAYOUT LAYOUT_T = IFAT::layout;
    static constexpr bool BALANCE = IFAT::isBalance;

    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);

    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;

    using ANTIQ_PARAMS_T = Q_T;

    using Q_ROPE_T = typename AscendC::Conditional<ANTIQUANT, Q_T, ORIGIN_T>::type;
    using K_ROPE_T = typename AscendC::Conditional<ANTIQUANT, KV_T, ORIGIN_T>::type;

    using UPDATE_T = typename AscendC::Conditional<QUANT || ANTIQUANT, half, T>::type;

    using TMP_T = typename AscendC::Conditional<ANTIQUANT, half, T>::type;
    using MM1_OUT_T = typename AscendC::Conditional<QUANT, int32_t, TMP_T>::type;
    using MM2_OUT_T = typename AscendC::Conditional<QUANT, half, TMP_T>::type;
    static constexpr float scaleP1 = 127.0;
    // static constexpr float scaleP2 = 1 / 127.0;
    static constexpr float scaleP2 = 0.0078740157f;
    static constexpr uint32_t PRELOAD_NUM = 2;
    ServiceFlashDecode<IFAT> fdService;
    static constexpr uint32_t mFdBaseSizeMla = 8;
    static constexpr int64_t fdPrefetchLen = 2;

  protected:
    const Int8TilingParams* __restrict tilingData = nullptr;
    TPipe* pipe = nullptr;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<Q_ROPE_T> qRopeGm;
    GlobalTensor<K_ROPE_T> kRopeGm;

    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<int32_t> blockTableGm;

    // atten mask
    GlobalTensor<bool> attenMaskBoolGm;

    // antiquant
    GlobalTensor<ANTIQ_PARAMS_T> keyAntiqOffsetGm;
    GlobalTensor<ANTIQ_PARAMS_T> keyAntiqScaleGm;
    GlobalTensor<ANTIQ_PARAMS_T> keyRopeAntiquantScaleGm;

    GlobalTensor<ANTIQ_PARAMS_T> valueAntiqOffsetGm;
    GlobalTensor<ANTIQ_PARAMS_T> valueAntiqScaleGm;

    QueryLengths actualSeqLengthsGmQ;
    GlobalTensor<int32_t> lengths32;
    GlobalTensor<float> publicOut, publicLse;
    bool publicLengthsPresent = false, publicPartials = false;
    __aicore__ inline uint64_t KvLength(uint32_t b)
    {
        return publicLengthsPresent ? uint64_t(lengths32.GetValue(b)) : kvSeqSize;
    }
    __aicore__ inline uint64_t PublicRow(const ExtraInfoMla& info, uint32_t row)
    {
        uint32_t part = 0;
        if constexpr (FLASH_DECODE)
            part = BALANCE ? (info.tndIsS2SplitCore ? info.tndCoreStartKVSplitPos : 0) : s2IdxFD;
        uint32_t m = mSizeVStart + row;
        if constexpr (LAYOUT_T == LAYOUT::TND) {
            uint64_t total = actualSeqLengthsGmQ.GetValue(batchSize - 1);
            uint64_t begin = info.bIdx ? actualSeqLengthsGmQ.GetValue(info.bIdx - 1) : 0;
            return (uint64_t(part) * qHeadNum + m % info.gSize) * total + begin + info.s1Idx * s1SizeSub +
                   m / info.gSize;
        }
        return ((uint64_t(part) * batchSize + info.bIdx) * qHeadNum + m % info.gSize) * qSeqSize +
               info.s1Idx * s1SizeSub + m / info.gSize;
    }
    __aicore__ inline void ExportPublicLse(const ExtraInfoMla& info, LocalTensor<float> sum, LocalTensor<float> max)
    {
        if (!publicPartials)
            return;
        LocalTensor<float> value = tmpBuff3.Get<float>();
        Log(value, sum, mSizeVector);
        PipeBarrier<PIPE_V>();
        Add(value, value, max, mSizeVector);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> broadcast = outputQue2.AllocTensor<float>();
        Brcb(broadcast, value, (mSizeVector + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < mSizeVector; ++row) {
            uint32_t q = info.s1Idx * s1SizeSub + (mSizeVStart + row) / info.gSize;
            if (attenMaskFlag && uint64_t(q) + info.s2Size < (LAYOUT_T == LAYOUT::TND ? info.actS1Size : qSeqSize))
                Duplicate(broadcast[row * 8], -FLOAT_INF, 8);
        }
        outputQue2.EnQue(broadcast);
        outputQue2.DeQue<float>();
        for (uint32_t row = 0; row < mSizeVector; ++row)
            DataCopyPad(publicLse[PublicRow(info, row)], broadcast[row * 8], DataCopyExtParams{1, 4, 0, 0, 0});
        outputQue2.FreeTensor(broadcast);
    }

    // quant
    GlobalTensor<T> deqScale1Gm;
    T deqScaleKey;
    T deqScale2Val;

    // workspace
    GlobalTensor<KV_T> queryPreProcessResGm; // 存放Q1, Q2
    GlobalTensor<MM1_OUT_T> mm1ResGm;        // 存放S
    GlobalTensor<KV_T> vec1ResGm;            // 存放A1, A2
    GlobalTensor<MM2_OUT_T> mm2ResGm;        // 存放O

    GlobalTensor<UPDATE_T> vec2ResGm;

    GlobalTensor<int32_t> nUpdateGm;
    GlobalTensor<T> softmaxSumGm;

    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    // queue
    TQue<QuePosition::VECIN, 1> inputQue1;   // 32K, inque
    TQue<QuePosition::VECIN, 1> inputQue2;   // 16K, inque
    TQue<QuePosition::VECOUT, 1> outputQue1; // 32K, outque
    TQue<QuePosition::VECOUT, 1> outputQue2; // 8K, outque

    // 临时tbuf
    TBuf<> tmpBuff1; // 32K
    TBuf<> tmpBuff2; // 32K
    TBuf<> tmpBuff3; // 2K

    TBuf<> nValueBuff;
    TBuf<> cofValueBuff;
    TBuf<> aMlaSumBuff;
    TBuf<> softmaxMaxBuff; // MLA_INT8_PRE_LOAD_NUM_MLA * 2K
    TBuf<> softmaxExpBuff; // MLA_INT8_PRE_LOAD_NUM_MLA * 2K
    TBuf<> softmaxSumBuff; // MLA_INT8_PRE_LOAD_NUM_MLA * 2K

    TBuf<> dequantScale1Buff;
    LocalTensor<T> dequantScale1Ub;
    TBuf<> pMaxBuff;
    LocalTensor<T> pMaxUb;

    TBuf<> softmaxMaxDefaultBuff; // 2K
    TBuf<> softmaxSumDefaultBuff; // 2K

    LocalTensor<T> nValueUb;
    LocalTensor<T> cofValueUb;
    LocalTensor<T> aMlaSumUb;
    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    TBuf<> antiqScaleBuff;  // 4K
    TBuf<> antiqOffsetBuff; // 4K
    TBuf<> qAmaxBuff;       // MLA_INT8_PRE_LOAD_NUM_MLA * (2K + 256B)

    // antiquant msd
    LocalTensor<T> qAmaxUb;
    LocalTensor<T> antiqScaleUb;
    LocalTensor<T> antiqOffsetUb;
    AttentionCommonFlashDecode::ConstInfoFD constInfo{};

    uint64_t msdRowMaxSize = 0;
    uint64_t msdRowSumSize = 0;
    uint64_t softmaxMaxSumExpSize = 0;

    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint64_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint64_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint64_t SYNC_C2_V2_FLAG = 9;

    static constexpr int32_t FP32_MAX_MASK_ELEMENT_NUM = 64;
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);                    // 32/4=8
    static constexpr uint32_t REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(T);            // 256/4=64
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = BUFFER_SIZE_BYTE_32K / sizeof(T); // 32768/4=8096

    half antiquantExpandCoeff = 254;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t*)&BOOL_ATTEN_MASK_SCALAR_VALUE);
    static constexpr T LN2 = 0.6931471805599453094172;
    static constexpr T FLOAT_INF = std::numeric_limits<T>::infinity();
    static constexpr uint64_t kvHeadNum = 1ULL;
    static constexpr uint64_t headDim = 512ULL;
    static constexpr uint64_t headDimAlign = 512ULL;
    static constexpr uint64_t headDimRope = 64ULL;
    static constexpr uint32_t n2Idx = 0U;
    // static constexpr float scaleC1  = 1024.0 / 127;
    static constexpr uint32_t msdIterNum = 2U;

    bool antiqOffsetExistFlag = false;
    uint32_t antiquantPerTensorFlag = 0U;

    // for workspace pingpong
    const uint32_t dbWorkspaceRatio = MLA_INT8_PRE_LOAD_NUM_MLA;

    __gm__ uint8_t* keyPtr = nullptr;
    __gm__ uint8_t* valuePtr = nullptr;

    __gm__ uint8_t* key_ = nullptr;
    __gm__ uint8_t* value_ = nullptr;

    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t subBlockNum = 2U; // AICORE 上AIC与AIV的数量默认是1:2

    // tilingdata
    uint64_t singleProcessSInnerSize = 0U;
    uint32_t sInnerLoopTimes = 0U;
    uint64_t singleProcessSInnerSizeTail = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t bIdx = 0U;
    uint32_t s1Idx = 0U;

    uint32_t mmResUbSize = 0U;

    uint32_t vec1ResUbSize = 0U;

    uint32_t bmm2ResUbSize = 0U;

    uint64_t batchSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t gSize = 0ULL;

    uint64_t actS1Size = 1ULL;
    uint64_t gSizeSub = 0ULL;
    uint64_t gSizeTail = 0ULL;
    uint64_t s1SizeSub = 0ULL;
    uint64_t s1SizeTail = 0ULL;
    uint64_t gOuter = 1ULL;
    uint64_t s1Outer = 1ULL;
    uint64_t gIdx = 0ULL;

    uint64_t mSizeVector = 0ULL;
    uint64_t mSizeVStart = 0ULL;
    uint64_t kvSeqSize = 0ULL;
    uint64_t qSeqSize = 1ULL;
    uint64_t tSeqSize = 1ULL; // TND T轴总长度（含padding）

    // pageAttention
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint64_t s2BatchBaseOffset = 0;

    bool softmaxLseFlag = false;

    // attention mask
    bool attenMaskFlag = false;
    uint32_t attenMaskSizeAlign = 0U;
    uint32_t sparseMode = 3U;

    // offset
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorARopeCoreOffset = 0ULL;
    uint64_t tensorBRopeCoreOffset = 0ULL;
    uint64_t tensorBOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;
    uint64_t attenMaskOffset = 0ULL;
    uint64_t attenMaskCoreOffset = 0ULL;
    uint64_t attenMaskSize = 0ULL;

    // splitKV
    uint32_t splitKVNum = 0U;
    uint32_t s2Idx = 0U;
    uint32_t s2IdxFD = 0U;
    uint64_t sInnerLoopSize = 0ULL;
    uint32_t actualCombineLoopSize = 0U;
    uint64_t combineLseOffset = 0ULL;
    uint64_t combineAccumOutOffset = 0ULL;

    uint64_t curActualSeqLen = 0ULL;
    uint64_t actualSingleProcessSInnerSize = 0ULL;
    uint64_t actualSingleProcessSInnerSizeAlign = 0ULL;
    uint32_t beforeBlockSplitBn2Nums = 0U;
    uint32_t bn2LoopTimes = 0U;

    uint32_t actualLenDims = 0U;
    uint32_t gMax = 128U;

    uint32_t tndSgBasicSize; // TND格式，cube M轴的tiling大小
    // TND分核信息
    uint32_t bEnd = 0U;
    static constexpr uint32_t n2End = 0U;
    uint32_t s1OuterIdxEnd = 0U;
    uint32_t s2End = 0U;
    uint32_t bStart = 0U;
    uint32_t n2Start = 0U;
    uint32_t s1OuterIdxStart = 0U;
    uint32_t s2Start = 0U;
    uint32_t bEndPrev = 0U;
    static constexpr uint32_t n2EndPrev = 0U;
    uint32_t s1OuterIdxEndPrev = 0U;
    uint32_t s2EndPrev = 0U;
    uint32_t tndFDCoreArrLen = 0U;
    uint32_t coreStartKVSplitPos = 0U;
    TNDFDSplitInfo tndFDCoreInfo;

    // 输出transpose信息
    // 记录当前轮的bIdx nIdx s2Idx actualLen
    uint32_t bn2IdxInCurCore = 0;
    using MatmulServiceFullQuant = MlaInt8Cube<IFAT>;
    using MatmulServiceQuant = MatmulServiceFullQuant;

    using MatmulServiceType = MatmulServiceQuant;
    MatmulServiceType matmulService;

    __aicore__ inline void InitValueGm(uint32_t bIdx);
    __aicore__ inline void InitKeyGm(uint32_t bIdx);
    __aicore__ inline void CalcParams(uint32_t loop, ExtraInfoMla& info, TaskContext& task);

    __aicore__ inline void ComputeMm1(const ExtraInfoMla& info);
    __aicore__ inline void ProcessVec1L(const ExtraInfoMla& info);
    __aicore__ inline void ComputeMm2(const ExtraInfoMla& info);
    __aicore__ inline void ProcessVec2L(const ExtraInfoMla& info);

    bool curActSeqLenIsZero = false;

    template <typename T>
    __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx = 0);
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx);

    __aicore__ inline void GetBN2Gid(const uint32_t bn2gIdx);

    __aicore__ inline void GetTNDAxisStartId(uint32_t& bStart, uint32_t& n2Start, uint32_t& s1OuterStart,
                                             uint32_t& s2Start, int bEnd, int n2End, int s1OuterIdxEnd, int s2End);
    __aicore__ inline uint64_t GetBalanceActualSeqLengths(QueryLengths& actualSeqLengthsTND, int bIdx);

    __aicore__ inline void AttenMaskCopyIn(uint64_t offset, uint32_t dealRowCount, uint32_t actualColumnCount);
    __aicore__ inline void AttenMaskCopyIn(const ExtraInfoMla& info);

    __aicore__ inline void FlashDecodeCompute();
    __aicore__ inline void FlashDecodeComputeND();

    __aicore__ inline void DealBmm1ResBaseBlock(const ExtraInfoMla& info, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void ProcessVec1Inner(const ExtraInfoMla& info);

    __aicore__ inline void ComputeSoftmaxLse(LocalTensor<T> softmaxlseUb, LocalTensor<T>& lseSumUb,
                                             LocalTensor<T>& lseMaxUb, uint32_t dealRowCountAlign);
    __aicore__ inline void DealSoftmaxLseInvalidRows(LocalTensor<T>& softmaxlseUb, LocalTensor<T>& lseMaxUb,
                                                     uint32_t dealRowCount, uint64_t s1Size, uint32_t curS1Idx);
    __aicore__ inline void SoftmaxLseCopyOut(const ExtraInfoMla& info, LocalTensor<T>& lseSumUb,
                                             LocalTensor<T>& lseMaxUb, uint32_t dealRowCount);

    __aicore__ inline void DealQuantBmm2ResBaseBlock(const ExtraInfoMla& info, uint32_t startRow, uint32_t dealRowCount,
                                                     uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void ProcessVec2Inner(const ExtraInfoMla& info);

    __aicore__ inline void SoftmaxFlashV2Compute(const ExtraInfoMla& info, LocalTensor<T>& mmResUb,
                                                 LocalTensor<uint8_t>& softmaxTmpUb, uint32_t startRow,
                                                 uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline bool IsSkipAttenMask(const ExtraInfoMla& info, uint32_t startRow, uint32_t dealRowCount);
    // 针对BSND/BSH/TND等切G的格式拷贝AttentionMask
    __aicore__ inline void AttenMaskCopyForSplitG(const ExtraInfoMla& info, LocalTensor<bool>& attenMaskUb,
                                                  uint32_t startRow, uint32_t dealRowCount);

    __aicore__ inline void ElewiseCompute(const ExtraInfoMla& info, LocalTensor<T>& mmResUb, TBuf<>& tmpBuf,
                                          uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                          uint32_t actualColumnCount);

    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T>& attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2DataCopyOutTrans(const ExtraInfoMla& info, LocalTensor<OUT_T>& attenOutUb,
                                                uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void Bmm2ResCopyOut(const ExtraInfoMla& info, LocalTensor<T>& bmm2ResUb, uint32_t startRow,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const ExtraInfoMla& info, LocalTensor<T>& bmm2ResUb, uint32_t startRow,
                                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    template <typename RT>
    __aicore__ inline void DealInvalidRows(const ExtraInfoMla& info, LocalTensor<RT>& attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void CombineSplitKVRes(uint64_t baseOffset = 0);
    __aicore__ inline void CopyAccumOutIn(uint32_t splitKVIndex, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset = 0);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const ExtraInfoMla& info, LocalTensor<T>& softmaxMaxUb,
                                                       LocalTensor<T>& softmaxSumUb);
    __aicore__ inline void Bmm2FDDataCopyOut(const ExtraInfoMla& info, LocalTensor<T>& bmm2ResUb, uint32_t startRow,
                                             uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ComputeScaleValue(LocalTensor<T>& lseSum, LocalTensor<T>& lseMax, uint32_t startRow,
                                             uint32_t dealRowCount);
    __aicore__ inline void ReduceFinalRes(LocalTensor<T>& dst, LocalTensor<T>& lseLocal, uint32_t startRow,
                                          uint32_t dealRowCount, uint64_t baseOffset = 0);
    __aicore__ inline void CopyFinalResOut(LocalTensor<T>& accumOutLocal, uint32_t startRow, uint32_t dealRowCount);

    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t n2Idx);
    __aicore__ inline void InitSoftmaxLseAllInfOutput(uint32_t bIdx, uint32_t n2Idx);

    __aicore__ inline void CopyFixedUbToGm(const GlobalTensor<T>& dst, const LocalTensor<T>& src, size_t size);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bIdx, uint32_t s1Idx);
};

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::InitTilingData()
{
    singleProcessSInnerSize = tilingData->increFlashAttentionSingleCoreParams.singleProcessSInnerSize;
    usedCoreNum = tilingData->increFlashAttentionSingleCoreParams.usedCoreNum;
    splitKVNum = tilingData->splitKVParams.s2;
    sInnerLoopSize = tilingData->splitKVParams.sInnerLoopSize;

    mmResUbSize = tilingData->increFlashAttentionSingleCoreTensorSize.mmResUbSize;

    vec1ResUbSize = mmResUbSize * msdIterNum;

    bmm2ResUbSize = tilingData->increFlashAttentionSingleCoreTensorSize.bmm2ResUbSize;

    batchSize = tilingData->baseParams.batchSize;
    qHeadNum = gSize = tilingData->baseParams.nNumOfQInOneGroup;

    gSizeSub = tilingData->increFlashAttentionSingleCoreParams.groupSplitSize; // 切块大小Gi
    gOuter = (gSize + gSizeSub - 1) / gSizeSub;
    gSizeTail = gSize - (gOuter - 1) * gSizeSub;

    kvSeqSize = tilingData->baseParams.seqSize;
    qSeqSize = tilingData->baseParams.qSeqSize;
    tSeqSize = tilingData->baseParams.tSeqSize;

    s1SizeSub = tilingData->increFlashAttentionSingleCoreParams.s1SplitSize; // 切块大小Si
    s1Outer = (qSeqSize + s1SizeSub - 1) / s1SizeSub;
    s1SizeTail = qSeqSize - (s1Outer - 1) * s1SizeSub;

    attenMaskFlag = (tilingData->baseParams.attenMaskFlag != 0) ? true : false;
    attenMaskSize = tilingData->baseParams.attenMaskSize;
    sparseMode = tilingData->baseParams.sparseMode;

    softmaxLseFlag = tilingData->baseParams.softmaxLseFlag;

    maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    kvCacheBlockSize = tilingData->baseParams.blockSize;
    tndSgBasicSize = s1SizeSub * gSize;
    // flashdecode新增结构体
    constInfo.preLoadNum = PRELOAD_NUM;
    constInfo.batchSize = tilingData->baseParams.batchSize;
    constInfo.gSize = tilingData->baseParams.nNumOfQInOneGroup;
    constInfo.qHeadNum = tilingData->baseParams.nNumOfQInOneGroup;
    constInfo.kvHeadNum = kvHeadNum;
    constInfo.headDim = headDim;
    constInfo.headDimAlign = headDimAlign;
    constInfo.qSeqSize = tilingData->baseParams.qSeqSize;
    constInfo.mBaseSize =
        tilingData->baseParams.nNumOfQInOneGroup * tilingData->increFlashAttentionSingleCoreParams.s1SplitSize;
}

#define MLA_INT8_QMAX_UB_SIZE (BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_256B)

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        // queue
        pipe->InitBuffer(inputQue1, 1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(inputQue2, 1, BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(outputQue1, 1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(outputQue2, 1, BUFFER_SIZE_BYTE_4K);

        // tmpBuff
        pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(tmpBuff2, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(tmpBuff3, BUFFER_SIZE_BYTE_2K);

        {
            pipe->InitBuffer(dequantScale1Buff, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_256B); // G_MAX / 2 * sizeof(T)
            pipe->InitBuffer(pMaxBuff, MLA_INT8_QMAX_UB_SIZE * MLA_INT8_PRE_LOAD_NUM_MLA);
        }

        // M_MAX=512, 512 * sizeof(T) * N_Buffer

        pipe->InitBuffer(softmaxMaxBuff, BUFFER_SIZE_BYTE_2K * MLA_INT8_PRE_LOAD_NUM_MLA);
        pipe->InitBuffer(softmaxExpBuff, BUFFER_SIZE_BYTE_2K * MLA_INT8_PRE_LOAD_NUM_MLA);
        pipe->InitBuffer(softmaxSumBuff, BUFFER_SIZE_BYTE_2K * MLA_INT8_PRE_LOAD_NUM_MLA);

        pipe->InitBuffer(softmaxMaxDefaultBuff, BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(softmaxSumDefaultBuff, BUFFER_SIZE_BYTE_2K);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::InitAllZeroOutput(uint32_t bIdx, uint32_t n2Idx)
{
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        uint32_t tSize = actualSeqLengthsGmQ.GetValue(batchSize - 1);
        uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
        uint32_t s1Count = actS1Size;

        for (int s1Idx = 0; s1Idx < s1Count; s1Idx++) {
            uint64_t attenOutOffset = (tBase + s1Idx) * kvHeadNum * gSize * headDim + //T轴、s1轴偏移
                                      n2Idx * gSize * headDim;                        //N2轴偏移
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], gSize * headDim, 0);
        }
    } else {
        for (int s1Idx = 0; s1Idx < qSeqSize; s1Idx++) {
            uint64_t attenOutOffset = bIdx * qSeqSize * kvHeadNum * gSize * headDim +
                                      s1Idx * kvHeadNum * gSize * headDim + //B轴、S1轴偏移
                                      n2Idx * gSize * headDim;              //N2轴偏移
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], gSize * headDim, 0);
        }
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::InitSoftmaxLseAllInfOutput(uint32_t bIdx, uint32_t n2Idx)
{
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        uint32_t tokenPrefixSum = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
        uint64_t total = actualSeqLengthsGmQ.GetValue(batchSize - 1);
        for (uint32_t h = 0; h < gSize; ++h)
            matmul::InitOutput<float>(softmaxLseGm[(n2Idx * gSize + h) * total + tokenPrefixSum], actS1Size,
                                      -FLOAT_INF);
    } else { // BSH BSND
        uint64_t softmaxLseOffset = bIdx * kvHeadNum * gSize * qSeqSize + n2Idx * gSize * qSeqSize;
        matmul::InitOutput<T>(softmaxLseGm[softmaxLseOffset], gSize * qSeqSize, -FLOAT_INF);
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx)
{
    if (actualLenDims == 0) {
        curActualSeqLen = kvSeqSize;
    } else if (actualLenDims == 1) {
        curActualSeqLen = KvLength(0);
    } else {
        curActualSeqLen = KvLength(bIdx);
    }

    if constexpr (BALANCE) {
        actS1Size = GetBalanceActualSeqLengths(actualSeqLengthsGmQ, bIdx);
    } else {
        if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
            actS1Size = (s1Idx == s1Outer - 1) ? s1SizeTail : s1SizeSub;
        } else {
            actS1Size = qSeqSize;
        }
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::GetBN2Gid(const uint32_t bn2gIdx)
{
    if constexpr (FLASH_DECODE) {
        bIdx = aiCoreIdx / (kvHeadNum * splitKVNum);
        // n2Idx = (aiCoreIdx / splitKVNum) % kvHeadNum;
        s2IdxFD = aiCoreIdx % splitKVNum;
    } else if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        uint32_t bs1n2 = beforeBlockSplitBn2Nums + bn2gIdx;
        uint32_t s1n2 = bs1n2 % (kvHeadNum * s1Outer);
        bIdx = bs1n2 / (kvHeadNum * s1Outer);
        s1Idx = s1n2 / kvHeadNum;
    } else {
        uint32_t bn2g = beforeBlockSplitBn2Nums + bn2gIdx;
        uint32_t n2g = bn2g % (kvHeadNum * gOuter);
        bIdx = bn2g / (kvHeadNum * gOuter);
        gIdx = n2g % gOuter;
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx)
{
    if ASCEND_IS_AIV {
        InitAllZeroOutput(bIdx, n2Idx);
        // AscendC InitOutput reuses its stack UB without a completion wait.
        // Finish zero O before the FP32 infinity fill reuses that buffer.
        PipeBarrier<PIPE_ALL>();
        if (softmaxLseFlag) {
            InitSoftmaxLseAllInfOutput(bIdx, n2Idx);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::UpdateInnerLoopCond()
{
    if ((curActualSeqLen == 0) || (actS1Size == 0)) {
        curActSeqLenIsZero = true;
        return;
    }
    curActSeqLenIsZero = false;

    if constexpr (BALANCE) {
        singleProcessSInnerSizeTail = curActualSeqLen % singleProcessSInnerSize;
        singleProcessSInnerSizeTail =
            (singleProcessSInnerSizeTail == 0) ? singleProcessSInnerSize : singleProcessSInnerSizeTail;
        sInnerLoopTimes = 0;
    } else {
        int32_t remainSinnerSize = (int32_t)curActualSeqLen;
        int32_t computeSinnerSize = (int32_t)curActualSeqLen;
        if constexpr (FLASH_DECODE) {
            remainSinnerSize = (int32_t)curActualSeqLen - sInnerLoopSize * s2IdxFD;
            computeSinnerSize = remainSinnerSize >= sInnerLoopSize ? sInnerLoopSize : remainSinnerSize;
            if (aiCoreIdx >= batchSize * kvHeadNum * splitKVNum) {
                remainSinnerSize = 0;
            }
        }
        if (remainSinnerSize > 0) {
            if (computeSinnerSize <= singleProcessSInnerSize) {
                singleProcessSInnerSizeTail = computeSinnerSize;
                sInnerLoopTimes = 1;
            } else {
                sInnerLoopTimes = (computeSinnerSize + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
                singleProcessSInnerSizeTail = computeSinnerSize - (sInnerLoopTimes - 1) * singleProcessSInnerSize;
            }
        } else {
            sInnerLoopTimes = 0;
        }
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::Init(GM_ADDR query, GM_ADDR key, GM_ADDR queryRope, GM_ADDR keyRope,
                                                 GM_ADDR attenMask, GM_ADDR blockTable, GM_ADDR attentionOut,
                                                 GM_ADDR softmaxLse, GM_ADDR workspace, const Int8TilingParams* tiling,
                                                 TPipe* tPipe)
{
    if ASCEND_IS_AIV {
        subBlockNum = GetSubBlockNum(); // CV1:2场景,返回值为2，其他场景返回值为1
        tmpBlockIdx = GetBlockIdx();    // vec:0-47
        aiCoreIdx = tmpBlockIdx / subBlockNum;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    // init tiling data
    tilingData = tiling;

    InitTilingData();
    if ASCEND_IS_AIV {
        if (publicPartials) {
            // Initialize all 26 public slots, including empty rows and unused
            // splits, before numerical writers start. Idle lanes participate.
            const uint64_t rows =
                26ULL * qHeadNum * (LAYOUT_T == LAYOUT::TND ? uint64_t(tSeqSize) : uint64_t(batchSize) * qSeqSize);
            const uint64_t lanes = uint64_t(GetBlockNum()) * GetSubBlockNum();
            const uint64_t chunk = (rows + lanes * 8 - 1) / (lanes * 8) * 8;
            const uint64_t begin = uint64_t(tmpBlockIdx) * chunk;
            const uint64_t end = begin + chunk < rows ? begin + chunk : rows;
            // InitOutput takes uint32 elements; keep large allocations bounded.
            constexpr uint32_t maxRows = (UINT32_MAX / headDim) / 8 * 8;
            for (uint64_t row = begin; row < end;) {
                const uint32_t count = end - row < maxRows ? uint32_t(end - row) : maxRows;
                matmul::InitOutput<float>(publicOut[row * headDim], count * uint32_t(headDim), 0.0f);
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
                // Drain each fill before the next InitOutput/InitBuffers reuses UB.
                matmul::InitOutput<float>(publicLse[row], count, -FLOAT_INF);
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
                row += count;
            }
            SyncAll();
        }
    }
    // Execute binds the public length arrays before Init; balanced
    // traversal needs their dimensions while finding core starts.
    actualLenDims = tilingData->baseParams.actualLenDims;
    // 初始化计算参数
    if (aiCoreIdx < usedCoreNum)
        InitCalcParamsEach();

    pipe = tPipe;
    keyPtr = key;
    valuePtr = key;

    actualSingleProcessSInnerSize = 0ULL;
    actualSingleProcessSInnerSizeAlign = 0ULL;

    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T*)query);
    qRopeGm.SetGlobalBuffer((__gm__ Q_ROPE_T*)queryRope);
    kRopeGm.SetGlobalBuffer((__gm__ K_ROPE_T*)keyRope);

    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T*)attentionOut);
    // batch连续时,只需要初始化一次;不连续时,需要在使用时根据batchIdx初始化
    {
        InitKeyGm(0);
        InitValueGm(0);
    }

    if (pipe != nullptr) {
        InitBuffers();
    }

    if (attenMaskFlag) {
        attenMaskBoolGm.SetGlobalBuffer((__gm__ bool*)attenMask);
    }

    // sparse9 TND padding 场景初始化 output 为 0

    {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t*)blockTable);
    }

    if ASCEND_IS_AIV {

        softmaxMaxUb = softmaxMaxBuff.Get<T>();
        softmaxSumUb = softmaxSumBuff.Get<T>();
        softmaxExpUb = softmaxExpBuff.Get<T>();

        softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
        softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();
        {
            pMaxUb = pMaxBuff.Get<T>();
            dequantScale1Ub = dequantScale1Buff.Get<T>();
        }
    }

    // workspace 内存排布
    // |Q--|mm1ResGm(存S)|vec1ResGm(存A1,A2)|mm2ResGm(存O)|vec2ResGm|nUpdateGm
    // |Core0_Q1-Core0_Q2-Core1_Q1-Core1_Q2....Core32_Q1-Core32_Q2|Core0_mmRes
    uint64_t offset = 0;

    mm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T*)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * mmResUbSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * mmResUbSize * sizeof(MM1_OUT_T);
    {
        vec1ResGm.SetGlobalBuffer(
            (__gm__ KV_T*)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * mmResUbSize * sizeof(KV_T)));
        offset += GetBlockNum() * dbWorkspaceRatio * mmResUbSize * sizeof(KV_T);
    }

    mm2ResGm.SetGlobalBuffer(
        (__gm__ MM2_OUT_T*)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * bmm2ResUbSize * sizeof(MM2_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(MM2_OUT_T);

    {
        vec2ResGm.SetGlobalBuffer(
            (__gm__ UPDATE_T*)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * bmm2ResUbSize * sizeof(UPDATE_T)));
        offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(UPDATE_T);
    }

    if constexpr (FLASH_DECODE) {
        accumOutGm.SetGlobalBuffer((__gm__ float*)(workspace + offset));
        offset = offset + tilingData->splitKVParams.accumOutSize * sizeof(float);
        lseSumFdGm.SetGlobalBuffer((__gm__ float*)(workspace + offset));
        lseMaxFdGm.SetGlobalBuffer((__gm__ float*)(workspace + offset) + tilingData->splitKVParams.logSumExpSize / 2);
        offset = offset + tilingData->splitKVParams.logSumExpSize * sizeof(float);
    }

    if (softmaxLseFlag) {
        softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmaxLse);
    }

    if ASCEND_IS_AIV {
        if constexpr (FLASH_DECODE) {
            fdService.InitParams(constInfo);
            fdService.InitGlobalTensor(lseMaxFdGm, lseSumFdGm, accumOutGm, attentionOutGm, actualSeqLengthsGmQ);
            fdService.InitLseOutput(softmaxLseGm, softmaxLseFlag);
        }
    }

    if ASCEND_IS_AIC {
        matmulService.InitParams(qHeadNum, kvHeadNum, headDim, headDimRope, qSeqSize, mmResUbSize, bmm2ResUbSize);
        {
            matmulService.InitMm1GlobalTensor(queryGm, qRopeGm, keyGm, kRopeGm, mm1ResGm);
            matmulService.InitMm2GlobalTensor(vec1ResGm, valueGm, mm2ResGm, attentionOutGm);
        }

        matmulService.InitPageAttentionInfo(blockTableGm, kvCacheBlockSize, maxBlockNumPerBatch);
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::InitKeyGm(uint32_t bIdx)
{
    key_ = keyPtr;

    keyGm.SetGlobalBuffer((__gm__ KV_T*)key_);
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::InitValueGm(uint32_t bIdx)
{
    value_ = valuePtr;

    valueGm.SetGlobalBuffer((__gm__ KV_T*)value_);
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::InitQuant(GM_ADDR dequantScaleQuery, GM_ADDR keyAntiquantScale,
                                                      GM_ADDR valueAntiquantScale)
{

    {
        deqScale1Gm.SetGlobalBuffer((__gm__ T*)dequantScaleQuery);
        deqScaleKey = *(__gm__ T*)keyAntiquantScale;
        deqScale2Val = *(__gm__ T*)valueAntiquantScale;
        if ASCEND_IS_AIV {
            if (!(deqScaleKey <= FLOAT_MAX && deqScaleKey >= -FLOAT_MAX && deqScale2Val <= FLOAT_MAX &&
                  deqScale2Val >= -FLOAT_MAX)) {
                Trap();
            }
        }
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::InitCalcParamsEach()
{
    // 这里是编译器优化写法，定义一个局部数组变量coreSidxEnd(存在栈上)，使用copy_data_align64接口
    // 可以只从ub中拷贝tiling中coreSidxEnd的内容到栈上，而非将整个increFlashAttentionCoreParams
    // 内容拷贝到栈，减少拷贝时间。

    if constexpr (BALANCE) {
#ifdef ASCENDC_CPU_DEBUG
        const uint32_t* coreSidxEnd = tilingData->increFlashAttentionCoreParams.coreSidxEnd;
        const uint32_t* coreBEnd = tilingData->increFlashAttentionCoreParams.coreBEnd;
        const uint32_t* coreS1OuterEnd = tilingData->increFlashAttentionCoreParams.coreS1OuterEnd;
        const uint32_t* coreS2End = tilingData->increFlashAttentionCoreParams.coreS2End;
        const uint32_t* balanceFDCoreBArr = tilingData->tndSplitCoreParams.balanceFDCoreBArr;
        const uint32_t* balanceFDCoreS1Arr = tilingData->tndSplitCoreParams.balanceFDCoreS1Arr;
        const uint32_t* balanceFDCoreKVSplitArr = tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr;
        const uint32_t* balanceFDCoreStartKVSplitNum = tilingData->tndSplitCoreParams.balanceFDCoreStartKVSplitNum;
#else
        uint32_t coreBEnd[MLA_INT8_ARRAY_SIZE(tilingData->increFlashAttentionCoreParams.coreBEnd)];
        uint32_t coreS1OuterEnd[MLA_INT8_ARRAY_SIZE(tilingData->increFlashAttentionCoreParams.coreS1OuterEnd)];
        uint32_t coreS2End[MLA_INT8_ARRAY_SIZE(tilingData->increFlashAttentionCoreParams.coreS2End)];
        uint32_t balanceFDCoreBArr[MLA_INT8_ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreBArr)];
        uint32_t balanceFDCoreS1Arr[MLA_INT8_ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreS1Arr)];
        uint32_t balanceFDCoreKVSplitArr[MLA_INT8_ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr)];
        uint32_t balanceFDCoreStartKVSplitNum[MLA_INT8_ARRAY_SIZE(
            tilingData->tndSplitCoreParams.balanceFDCoreStartKVSplitNum)];
        copy_data_align64((uint8_t*)coreBEnd, (uint8_t*)(tilingData->increFlashAttentionCoreParams.coreBEnd),
                          sizeof(coreBEnd));
        copy_data_align64((uint8_t*)coreS1OuterEnd,
                          (uint8_t*)(tilingData->increFlashAttentionCoreParams.coreS1OuterEnd), sizeof(coreS1OuterEnd));
        copy_data_align64((uint8_t*)coreS2End, (uint8_t*)(tilingData->increFlashAttentionCoreParams.coreS2End),
                          sizeof(coreS2End));
        copy_data_align64((uint8_t*)balanceFDCoreBArr, (uint8_t*)(tilingData->tndSplitCoreParams.balanceFDCoreBArr),
                          sizeof(balanceFDCoreBArr));
        copy_data_align64((uint8_t*)balanceFDCoreS1Arr, (uint8_t*)(tilingData->tndSplitCoreParams.balanceFDCoreS1Arr),
                          sizeof(balanceFDCoreS1Arr));
        copy_data_align64((uint8_t*)balanceFDCoreKVSplitArr,
                          (uint8_t*)(tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr),
                          sizeof(balanceFDCoreKVSplitArr));
        copy_data_align64((uint8_t*)balanceFDCoreStartKVSplitNum,
                          (uint8_t*)(tilingData->tndSplitCoreParams.balanceFDCoreStartKVSplitNum),
                          sizeof(balanceFDCoreStartKVSplitNum));
#endif
        // TND分核信息
        bEnd = coreBEnd[aiCoreIdx];
        s1OuterIdxEnd = coreS1OuterEnd[aiCoreIdx];
        s2End = coreS2End[aiCoreIdx];
        if (aiCoreIdx != 0) {
            bEndPrev = coreBEnd[aiCoreIdx - 1];
            s1OuterIdxEndPrev = coreS1OuterEnd[aiCoreIdx - 1];
            s2EndPrev = coreS2End[aiCoreIdx - 1];
        }
        GetTNDAxisStartId(bStart, n2Start, s1OuterIdxStart, s2Start, bEnd, n2End, s1OuterIdxEnd, s2End);
        tndFDCoreArrLen = tilingData->tndSplitCoreParams.tndFDCoreArrLen;
        if (tmpBlockIdx < tndFDCoreArrLen) {
            // 当前核要规约的[B,N,S1]
            tndFDCoreInfo.bIdx = balanceFDCoreBArr[tmpBlockIdx];
            tndFDCoreInfo.n2Idx = 0;
            tndFDCoreInfo.s1Idx = balanceFDCoreS1Arr[tmpBlockIdx];
            tndFDCoreInfo.kvSplitNum = balanceFDCoreKVSplitArr[tmpBlockIdx];
            for (int i = 0; i < tmpBlockIdx; i++) {
                tndFDCoreInfo.combineTaskPrefixSum += balanceFDCoreKVSplitArr[i];
            }
        }
        coreStartKVSplitPos = balanceFDCoreStartKVSplitNum[aiCoreIdx];
    } else {
        if constexpr (FLASH_DECODE) {
            bn2LoopTimes = 1U;
        } else {
#ifdef ASCENDC_CPU_DEBUG
            const uint32_t* coreSidxEnd = tilingData->increFlashAttentionCoreParams.coreSidxEnd;
#else
            uint32_t coreSidxEnd[MLA_INT8_ARRAY_SIZE(tilingData->increFlashAttentionCoreParams.coreSidxEnd)];
            copy_data_align64((uint8_t*)coreSidxEnd, (uint8_t*)(tilingData->increFlashAttentionCoreParams.coreSidxEnd),
                              sizeof(coreSidxEnd));
#endif
            bn2LoopTimes = coreSidxEnd[aiCoreIdx + 1] - coreSidxEnd[aiCoreIdx];
            beforeBlockSplitBn2Nums = coreSidxEnd[aiCoreIdx];
        }
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::AttenMaskCopyIn(uint64_t offset, uint32_t dealRowCount,
                                                            uint32_t actualColumnCount)
{
    LocalTensor<bool> maskUb = inputQue2.AllocTensor<bool>();
    attenMaskSizeAlign = Align(actualColumnCount, 32U);
    maskUb.SetSize(dealRowCount * attenMaskSizeAlign);
#if (__CCE_AICORE__ > 200)
    if (actualColumnCount % 32 == 0) {
        DataCopy(maskUb, attenMaskBoolGm[offset], attenMaskSizeAlign);
    } else {
        uint32_t typeElementSize = BYTE_BLOCK / sizeof(bool);
        DataCopyExtParams intriParams;
        intriParams.blockLen = actualColumnCount * sizeof(bool);
        intriParams.blockCount = 1;
        intriParams.dstStride = (attenMaskSizeAlign - actualColumnCount) / typeElementSize;
        intriParams.srcStride = 0;
        DataCopyPadExtParams<bool> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = (attenMaskSizeAlign - actualColumnCount) % typeElementSize;
        padParams.paddingValue = 0;
        DataCopyPad(maskUb, attenMaskBoolGm[offset], intriParams, padParams);
    }
#else
    DataCopy(maskUb, attenMaskBoolGm[offset], attenMaskSizeAlign);
#endif
    inputQue2.template EnQue(maskUb);
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::AttenMaskCopyIn(const ExtraInfoMla& info)
{
#define ATTENMASK_STRIDE 2048U
    uint32_t offset;
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        int32_t delta =
            info.s1Idx * s1SizeSub - info.s2Idx * singleProcessSInnerSize + info.s2Size - info.actS1Size; // s1idx = 0
        if (delta < 0) {
            offset = (-delta) < (int32_t)s1SizeSub ? (-delta) : s1SizeSub; // min (-delta, s1Size)
        } else {
            offset = (delta < (int32_t)singleProcessSInnerSize ? delta : singleProcessSInnerSize) *
                     ATTENMASK_STRIDE; // min(delta, s2inner)
        }
    } else {
        int32_t delta =
            info.s1Idx * s1SizeSub - info.s2Idx * singleProcessSInnerSize + info.s2Size - qSeqSize; // s1idx = 0
        if (delta < 0) {
            offset = (-delta) < (int32_t)info.s1Size ? (-delta) : info.s1Size; // min (-delta, s1Size)
        } else {
            offset = (delta < (int32_t)singleProcessSInnerSize ? delta : singleProcessSInnerSize) *
                     ATTENMASK_STRIDE; // min(delta, s2inner)
        }
    }

    attenMaskSizeAlign = Align(info.actualSingleProcessSInnerSize, 32U);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = info.s1Size;
    dataCopyParams.blockLen = attenMaskSizeAlign * sizeof(bool) / 32;
    dataCopyParams.srcStride = (ATTENMASK_STRIDE - attenMaskSizeAlign) * sizeof(bool) / 32;
    dataCopyParams.dstStride = 0;

    LocalTensor<bool> maskUb = inputQue2.AllocTensor<bool>();
    DataCopy(maskUb, attenMaskBoolGm[offset], dataCopyParams);

    inputQue2.template EnQue(maskUb);
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset)
{
    LocalTensor<T> lseMax = inputQue1.AllocTensor<T>();
    LocalTensor<T> lseSum = inputQue2.AllocTensor<T>();
    uint64_t combineLoopOffset;

    if constexpr (BALANCE) {
        //  { B,   N2, splitKVNum, S1, G，D }
        combineLseOffset = (baseOffset + startRow) * FP32_ONE_BLOCK_SIZE;
        combineLoopOffset = tndSgBasicSize * FP32_ONE_BLOCK_SIZE;
    } else {
        combineLseOffset =
            ((bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum) * gSize + startRow) * FP32_ONE_BLOCK_SIZE;
        combineLoopOffset = qSeqSize * gSize * FP32_ONE_BLOCK_SIZE;
    }
    uint64_t dealRowCountAlign = dealRowCount * FP32_ONE_BLOCK_SIZE;
    for (uint32_t i = 0; i < actualCombineLoopSize; i++) {
        DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm[combineLseOffset + i * combineLoopOffset],
                 dealRowCountAlign); // 份数offset
        DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm[combineLseOffset + i * combineLoopOffset],
                 dealRowCountAlign);
    }
    inputQue2.EnQue(lseSum);
    inputQue1.EnQue(lseMax);
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::CopyAccumOutIn(uint32_t splitKVIndex, uint32_t startRow,
                                                           uint32_t dealRowCount)
{
    LocalTensor<T> accumOutLocal = inputQue1.AllocTensor<T>();

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = headDim * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (headDimAlign - headDim) / BLOCK_ELEMENT_NUM;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (headDimAlign - headDim) % BLOCK_ELEMENT_NUM;
    copyInPadParams.paddingValue = 0;

    if constexpr (BALANCE) {
        combineAccumOutOffset = startRow * headDim +                     // taskoffset + g轴offset
                                splitKVIndex * tndSgBasicSize * headDim; // 份数offset
    } else {
        combineAccumOutOffset =
            (bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum + splitKVIndex) * gSize * headDim + startRow * headDim;
    }

    DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    inputQue1.EnQue(accumOutLocal);
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::ComputeScaleValue(LocalTensor<T>& lseSum, LocalTensor<T>& lseMax,
                                                              uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> lseMaxUb = softmaxMaxUb[0];
    LocalTensor<T> lseSumUb = softmaxSumUb[0];
    LocalTensor<T> lseExpUb = tmpBuff1.Get<T>();

    // lseLocal的shape为[actualCombineLoopSize, dealRowCount * FP32_ONE_BLOCK_SIZE]
    uint32_t dealRowCountAlign = dealRowCount * FP32_ONE_BLOCK_SIZE;
    Duplicate(lseMaxUb, -FLOAT_MAX, dealRowCountAlign);
    Duplicate(lseSumUb, FLOAT_ZERO, dealRowCountAlign);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Max(lseMaxUb, lseMaxUb, lseMax[i * dealRowCountAlign], dealRowCountAlign);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Sub(lseExpUb[i * dealRowCountAlign], lseMax[i * dealRowCountAlign], lseMaxUb, dealRowCountAlign);
    }
    PipeBarrier<PIPE_V>();
    Exp(lseExpUb, lseExpUb, actualCombineLoopSize * dealRowCountAlign);
    PipeBarrier<PIPE_V>();

    Mul(lseSum, lseSum, lseExpUb, actualCombineLoopSize * dealRowCountAlign);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Add(lseSumUb, lseSumUb, lseSum[i * dealRowCountAlign], dealRowCountAlign);
        PipeBarrier<PIPE_V>();
    }

    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Div(lseSum[i * dealRowCountAlign], lseSum[i * dealRowCountAlign], lseSumUb, dealRowCountAlign);
    }
    PipeBarrier<PIPE_V>();

    if (softmaxLseFlag) {
        LocalTensor<T> softmaxlseUb = outputQue2.template AllocTensor<T>();
        ComputeSoftmaxLse(softmaxlseUb, lseSumUb, lseMaxUb, dealRowCountAlign);

        outputQue2.EnQue(softmaxlseUb);
        outputQue2.DeQue<T>();

        uint64_t softmaxLseOutOffset = attenOutOffset / headDim + startRow;
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockLen = sizeof(T);
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        if constexpr (LAYOUT_T == LAYOUT::TND) {
            uint64_t total = actualSeqLengthsGmQ.GetValue(batchSize - 1);
            dataCopyParams.blockCount = 1;
            for (uint32_t row = 0; row < dealRowCount; ++row) {
                uint64_t flat = softmaxLseOutOffset + row;
                DataCopyPad(softmaxLseGm[(flat % qHeadNum) * total + flat / qHeadNum],
                            softmaxlseUb[row * FP32_ONE_BLOCK_SIZE], dataCopyParams);
            }
        } else {
            // Source FD writes flattened [B,S,H]; use the same [B,H,S]
            // output contract as the non-FD path, without a separate transpose.
            dataCopyParams.blockCount = 1;
            for (uint32_t row = 0; row < dealRowCount; ++row) {
                uint64_t flat = softmaxLseOutOffset + row;
                uint64_t batch = flat / (qSeqSize * qHeadNum);
                uint64_t token = (flat / qHeadNum) % qSeqSize;
                uint64_t head = flat % qHeadNum;
                DataCopyPad(softmaxLseGm[(batch * qHeadNum + head) * qSeqSize + token],
                            softmaxlseUb[row * FP32_ONE_BLOCK_SIZE], dataCopyParams);
            }
        }
        outputQue2.FreeTensor(softmaxlseUb);
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::ReduceFinalRes(LocalTensor<T>& dst, LocalTensor<T>& lseLocal,
                                                           uint32_t startRow, uint32_t dealRowCount,
                                                           uint64_t baseOffset)
{
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1RepStride = headDimAlign / FP32_ONE_BLOCK_SIZE;
    repeatParams.dstRepStride = headDimAlign / FP32_ONE_BLOCK_SIZE;
    int32_t dtypeMask = 256 / sizeof(float);
    int32_t mulLoop = headDimAlign / dtypeMask;
    int32_t mulRemain = headDimAlign % dtypeMask;

    // 第一次，mul结果直接放到dst里
    CopyAccumOutIn(0, baseOffset + startRow, dealRowCount);
    LocalTensor<T> accumOutLocal = inputQue1.DeQue<T>();
    for (int i = 0; i < mulLoop; i++) {
        Mul(dst[i * dtypeMask], lseLocal, accumOutLocal[i * dtypeMask], dtypeMask, dealRowCount, repeatParams);
    }
    if (mulRemain > 0) {
        Mul(dst[mulLoop * dtypeMask], lseLocal, accumOutLocal[mulLoop * dtypeMask], mulRemain, dealRowCount,
            repeatParams);
    }
    PipeBarrier<PIPE_V>();
    inputQue1.FreeTensor(accumOutLocal);

    uint32_t dealRowCountAlign = dealRowCount * FP32_ONE_BLOCK_SIZE;
    for (uint32_t j = 1; j < actualCombineLoopSize; ++j) {
        CopyAccumOutIn(j, baseOffset + startRow, dealRowCount);
        LocalTensor<T> accumOutLocal = inputQue1.DeQue<T>();
        for (int i = 0; i < mulLoop; i++) {
            Mul(accumOutLocal[i * dtypeMask], lseLocal[j * dealRowCountAlign], accumOutLocal[i * dtypeMask], dtypeMask,
                dealRowCount, repeatParams);
        }
        if (mulRemain > 0) {
            Mul(accumOutLocal[mulLoop * dtypeMask], lseLocal[j * dealRowCountAlign], accumOutLocal[mulLoop * dtypeMask],
                mulRemain, dealRowCount, repeatParams);
        }
        PipeBarrier<PIPE_V>();
        Add(dst, dst, accumOutLocal, dealRowCount * headDimAlign);
        PipeBarrier<PIPE_V>();
        inputQue1.FreeTensor(accumOutLocal);
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::CopyFinalResOut(LocalTensor<T>& accumOutLocal, uint32_t startRow,
                                                            uint32_t dealRowCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
    uint32_t shapeArray[] = {dealRowCount, (uint32_t)headDim};
    tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, dealRowCount * headDimAlign);
    }

    outputQue1.EnQue(tmpBmm2ResCastTensor);
    outputQue1.DeQue<OUT_T>();
    Bmm2DataCopyOut(attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, headDimAlign, headDim);
    outputQue1.FreeTensor(tmpBmm2ResCastTensor);
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::CombineSplitKVRes(uint64_t baseOffset)
{
    if (curActualSeqLen != 0) {
        uint32_t gSplitSizeLse = BUFFER_SIZE_BYTE_16K / (BYTE_BLOCK * splitKVNum);
        uint32_t gSplitSizeAccumOut = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
        // 取两者较小的，用来切g，保证ub够用
        uint32_t gSplitSize = (gSplitSizeLse < gSplitSizeAccumOut) ? gSplitSizeLse : gSplitSizeAccumOut;
        gSplitSize = (gSplitSize > gSize) ? gSize : gSplitSize; // 最大为gSize
        uint32_t loopCount = (gSize + gSplitSize - 1) / gSplitSize;
        uint32_t tailSplitSize = gSize - (loopCount - 1) * gSplitSize;

        // 尾块与非尾块都使用这些ub，减少处理次数
        for (uint32_t i = 0, actualGSplitSize = gSplitSize; i < loopCount; i++) {
            uint32_t startRow = i * gSplitSize;
            if ((i + 1) == loopCount) {
                actualGSplitSize = tailSplitSize;
            }
            CopyLseIn(startRow, actualGSplitSize, baseOffset);
            LocalTensor<T> lseSum = inputQue2.DeQue<T>();
            LocalTensor<T> lseMax = inputQue1.DeQue<T>();
            ComputeScaleValue(lseSum, lseMax, startRow, actualGSplitSize);
            inputQue1.FreeTensor(lseMax);

            uint32_t gSplitBmm2UbSize = headDimAlign * actualGSplitSize;
            LocalTensor<T> tmp1 = tmpBuff1.Get<T>(gSplitBmm2UbSize);
            ReduceFinalRes(tmp1, lseSum, startRow, actualGSplitSize, baseOffset);
            inputQue2.FreeTensor(lseSum);

            CopyFinalResOut(tmp1, startRow, actualGSplitSize);
        }
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::FlashDecodeComputeND()
{
    mSizeVStart = 0;
    // 根据核id从数组把bns1Idx和份数拿出来
    bIdx = tndFDCoreInfo.bIdx;
    s1Idx = tndFDCoreInfo.s1Idx;
    actualCombineLoopSize = tndFDCoreInfo.kvSplitNum;

    uint32_t actualSeqQ = GetBalanceActualSeqLengths(actualSeqLengthsGmQ, bIdx);
    curActualSeqLen = KvLength(bIdx);
    int sOuterLoopTailIdx = ((actualSeqQ + s1SizeSub - 1) / s1SizeSub) - 1;
    int s1RowSize = s1SizeSub;
    if (s1Idx == sOuterLoopTailIdx) { // 尾块场景
        s1RowSize = (actualSeqQ % s1SizeSub == 0) ? s1SizeSub : (actualSeqQ % s1SizeSub);
    }

    s1Idx = s1Idx * s1SizeSub; // 这里s1Idx 指行号
    uint64_t s1IdxStart = s1Idx;
    uint32_t s1IdxEnd = s1Idx + s1RowSize;
    uint64_t taskOffset = tndFDCoreInfo.combineTaskPrefixSum * kvHeadNum * tndSgBasicSize;

    // 每次循环完成一次GD块规约
    for (; s1Idx < s1IdxEnd; s1Idx++) {
        uint64_t baseOffset = 0;
        if constexpr (BALANCE) {
            uint64_t actualSeqQPrefixSum;
            if constexpr (LAYOUT_T == LAYOUT::TND) {
                actualSeqQPrefixSum = (bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
            } else { // BSND
                actualSeqQPrefixSum = (bIdx <= 0) ? 0 : bIdx * qSeqSize;
            }
            uint64_t batchOutOffset = actualSeqQPrefixSum * qHeadNum * headDim; // 累积计算当前Batch对应的 offset
            baseOffset = taskOffset + (s1Idx - s1IdxStart) * kvHeadNum * gSize;
            attenOutOffset = batchOutOffset + s1Idx * (kvHeadNum * gSize * headDim) + n2Idx * (gSize * headDim);
        } else {
            // { B, S1, N2, G，D} BS1N2分核 GD 规约
            attenOutOffset = bIdx * qSeqSize * kvHeadNum * gSize * headDim + s1Idx * (kvHeadNum * gSize * headDim) +
                             n2Idx * (gSize * headDim);
        }
        CombineSplitKVRes(baseOffset);
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::FlashDecodeCompute()
{
    if constexpr (BALANCE) {
        if (tmpBlockIdx >= tndFDCoreArrLen) {
            return;
        }
        FlashDecodeComputeND();
    } else {
        bIdx = tmpBlockIdx / kvHeadNum;
        attenOutOffset = bIdx * kvHeadNum * gSize * headDim + n2Idx * gSize * headDim;
        mSizeVStart = 0;
        if (tmpBlockIdx >= batchSize * kvHeadNum) {
            return;
        }

        if (actualLenDims == 0) {
            curActualSeqLen = kvSeqSize;
        } else if (actualLenDims == 1) {
            curActualSeqLen = KvLength(0);
        } else {
            curActualSeqLen = KvLength(bIdx);
        }

        actualCombineLoopSize = (curActualSeqLen + sInnerLoopSize - 1) / sInnerLoopSize;

        CombineSplitKVRes();
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::ComputeLogSumExpAndCopyToGm(const ExtraInfoMla& info,
                                                                        LocalTensor<T>& softmaxSumUb,
                                                                        LocalTensor<T>& softmaxMaxUb)
{
    if constexpr (BALANCE) {
        // workspace同步修改
        //  src-Shape  { gsizeV, S1, FP32_ONE_BLOCK_SIZE }
        //  dst-Shape  { B  N2, splitKV s1, G, FP32_ONE_BLOCK_SIZE}
        // 这里的offset计算，后续FD切G改切M时，同步改掉
        s2IdxFD = info.tndCoreStartKVSplitPos;
        size_t size = mSizeVector * FP32_ONE_BLOCK_SIZE;
        uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.s1Idx);
        uint64_t offset = (accumTmpOutNum * kvHeadNum * tndSgBasicSize + // taskoffset
                           s2IdxFD * kvHeadNum * tndSgBasicSize +        // 份数offset
                           mSizeVStart) *
                          FP32_ONE_BLOCK_SIZE; // m轴offset

        LocalTensor<T> tmp = outputQue2.AllocTensor<T>();

        Brcb(tmp, softmaxSumUb, (mSizeVector + 7) / 8, {1, 8});
        outputQue2.EnQue(tmp);
        outputQue2.DeQue<T>();
        DataCopy(lseSumFdGm[offset], tmp, size);
        outputQue2.FreeTensor(tmp);

        tmp = outputQue2.AllocTensor<T>();
        Brcb(tmp, softmaxMaxUb, (mSizeVector + 7) / 8, {1, 8});
        outputQue2.EnQue(tmp);
        outputQue2.DeQue<T>();
        DataCopy(lseMaxFdGm[offset], tmp, size);
        outputQue2.FreeTensor(tmp);
    } else {
        size_t size = mSizeVector * FP32_ONE_BLOCK_SIZE;
        size_t offset = (info.bIdx * kvHeadNum * splitKVNum * gSize + info.n2Idx * splitKVNum * gSize +
                         s2IdxFD * gSize + mSizeVStart) *
                        FP32_ONE_BLOCK_SIZE;
        // lseSumFdGm: batchQ * kvHeadNum * splitKVNum * gSize * FP32_ONE_BLOCK_SIZE
        CopyFixedUbToGm(lseSumFdGm[offset], softmaxSumUb, size);

        CopyFixedUbToGm(lseMaxFdGm[offset], softmaxMaxUb, size);
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::AttenMaskCopyForSplitG(const ExtraInfoMla& info,
                                                                   LocalTensor<bool>& attenMaskUb, uint32_t startRow,
                                                                   uint32_t dealRowCount)
{
    uint32_t s1StartIdx = (mSizeVStart + startRow) / info.gSize;
    uint32_t s1EndIdx = (mSizeVStart + startRow + dealRowCount - 1) / info.gSize;
    uint32_t s1Count = s1EndIdx - s1StartIdx + 1;

#define ATTENMASK_STRIDE 2048U

    uint32_t offset;
    uint32_t actualSeqQ = qSeqSize;
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        actualSeqQ = info.actS1Size;
    }
    int32_t delta =
        (info.s1Idx * s1SizeSub + s1StartIdx) - info.s2Idx * singleProcessSInnerSize + (info.s2Size - actualSeqQ);
    if (delta < 0) {
        offset = (-delta) < (int32_t)s1Count ? (-delta) : s1Count;
    } else {
        offset = (delta < (int32_t)singleProcessSInnerSize ? delta : singleProcessSInnerSize) * ATTENMASK_STRIDE;
    }

    attenMaskSizeAlign = Align(info.actualSingleProcessSInnerSize, 32U);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s1Count;
    dataCopyParams.blockLen = attenMaskSizeAlign * sizeof(bool) / 32;
    dataCopyParams.srcStride = (ATTENMASK_STRIDE - attenMaskSizeAlign) * sizeof(bool) / 32;
    dataCopyParams.dstStride = 0;

    attenMaskUb = inputQue2.AllocTensor<bool>();
    DataCopy(attenMaskUb, attenMaskBoolGm[offset], dataCopyParams);

    inputQue2.template EnQue(attenMaskUb);

    attenMaskUb = inputQue2.DeQue<bool>();
    LocalTensor<int16_t> mask16 = attenMaskUb.template ReinterpretCast<int16_t>();
    LocalTensor<int16_t> attenMaskUbDst = mask16[BUFFER_SIZE_BYTE_16K / 4];

    uint32_t headGCount = 0;
    uint32_t firstGIdx = (mSizeVStart + startRow) % info.gSize;
    if (s1Count > 1) {
        headGCount = info.gSize - firstGIdx;
    } else {
        headGCount = dealRowCount;
    }
    uint32_t dstMaskOffset = 0;
    uint32_t srcMaskBaseOffset = 0;
    // head
    SetMaskCount();
    SetVectorMask<int16_t, MaskMode::COUNTER>(attenMaskSizeAlign / 2);
    Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset], AscendC::MASK_PLACEHOLDER,
                         headGCount, {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
    dstMaskOffset += headGCount * attenMaskSizeAlign / 2;
    srcMaskBaseOffset += attenMaskSizeAlign / 2;
    // mid
    uint32_t reminRowCount = dealRowCount - headGCount;
    uint32_t midS1Count = reminRowCount / info.gSize;
    uint32_t tailGSize = reminRowCount % info.gSize;
    for (uint32_t midIdx = 0; midIdx < midS1Count; midIdx++) {
        Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset], AscendC::MASK_PLACEHOLDER,
                             info.gSize, {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
        dstMaskOffset += info.gSize * attenMaskSizeAlign / 2;
        srcMaskBaseOffset += attenMaskSizeAlign / 2;
    }
    // tail
    if (tailGSize > 0) {
        Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset], AscendC::MASK_PLACEHOLDER,
                             tailGSize, {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
    }
    SetMaskNorm();
    ResetMask();
    attenMaskUb = attenMaskUbDst.template ReinterpretCast<bool>();
}

// 新增sparse9的处理

template <typename IFAT>
__aicore__ inline bool MlaInt8Vector<IFAT>::IsSkipAttenMask(const ExtraInfoMla& info, uint32_t startRow,
                                                            uint32_t dealRowCount)
{
    uint32_t actualSeqQ = qSeqSize;
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        actualSeqQ = info.actS1Size;
    }

    uint32_t s2EndPos = info.s2Idx * singleProcessSInnerSize + info.actualSingleProcessSInnerSize;
    // 新增sparse9的处理，sparse9和sparse3类似，但是sparse9是用户自己传入的mask矩阵，算子无法感知具体数值，所以涉及到基本块都要参与计算

    // s2<s1时，必然走mask
    if (info.s2Size < actualSeqQ) {
        return false;
    }

    // 当前的s2位置不超过需要打标记的位置时，不需要mask
    if (s2EndPos <= (info.s2Size - actualSeqQ + 1)) {
        return true;
    }

    // BSH/BSND/TND格式切G，最后一个s1不需要mask
    {
        uint32_t baseS1StartIdx = info.s1Idx * s1SizeSub;
        uint32_t s1StartIdx = (mSizeVStart + startRow) / info.gSize + baseS1StartIdx;
        uint32_t s1EndIdx = (mSizeVStart + startRow + dealRowCount - 1) / info.gSize + baseS1StartIdx;
        if (s1StartIdx == s1EndIdx && s1StartIdx == actualSeqQ - 1) {
            return true;
        }
    }
    return false;
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::ElewiseCompute(const ExtraInfoMla& info, LocalTensor<T>& mmResUb,
                                                           TBuf<>& tmpBuf, uint32_t startRow, uint32_t dealRowCount,
                                                           uint32_t columnCount, uint32_t actualColumnCount)
{

    // attenMask
    if (attenMaskFlag == 1) {
        LocalTensor<bool> attenMaskUb;
        if ((qSeqSize == 1) && (LAYOUT_T != LAYOUT::TND)) {
            AttenMaskCopyIn(info.attenMaskOffset, dealRowCount, actualColumnCount);
            attenMaskUb = inputQue2.DeQue<bool>();
            for (int i = 1; i < dealRowCount; i++) {
                DataCopy(attenMaskUb[i * attenMaskSizeAlign], attenMaskUb, attenMaskSizeAlign);
            }
        } else {
            if (IsSkipAttenMask(info, startRow, dealRowCount)) {
                return;
            }
            { // BSH/BSND/TND
                {
                    AttenMaskCopyForSplitG(info, attenMaskUb, startRow, dealRowCount);
                }
            }
        }

        PipeBarrier<PIPE_V>();

        LocalTensor<uint8_t> ubWorkSpace = tmpBuf.Get<uint8_t>();
        SelectWithBytesMaskShapeInfo selectWithBytesMaskShapeInfo;
        selectWithBytesMaskShapeInfo.firstAxis = dealRowCount;
        selectWithBytesMaskShapeInfo.srcLastAxis = columnCount;
        selectWithBytesMaskShapeInfo.maskLastAxis = attenMaskSizeAlign;
        attenMaskUb.SetSize(dealRowCount * attenMaskSizeAlign); // Select接口要求mask size与参数匹配
        mmResUb.SetSize(dealRowCount * columnCount);            // Select接口要求src size与参数匹配
        SelectWithBytesMask(mmResUb, mmResUb, BOOL_ATTEN_MASK_SCALAR_VALUE, attenMaskUb, ubWorkSpace,
                            selectWithBytesMaskShapeInfo);
        mmResUb.SetSize(BUFFER_SIZE_BYTE_32K / sizeof(T)); // mmResUb Size复原,mask不用复原,与原来一致
        inputQue2.FreeTensor(attenMaskUb);

        PipeBarrier<PIPE_V>();
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::SoftmaxFlashV2Compute(const ExtraInfoMla& info, LocalTensor<T>& mmResUb,
                                                                  LocalTensor<uint8_t>& softmaxTmpUb, uint32_t startRow,
                                                                  uint32_t dealRowCount, uint32_t columnCount,
                                                                  uint32_t actualColumnCount)
{
    SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling =
        SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);

    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = startRow;
    uint32_t outIdx = info.loop % (MLA_INT8_PRE_LOAD_NUM_MLA);
    uint32_t softmaxOutOffset = outIdx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (MLA_INT8_PRE_LOAD_NUM_MLA);
        inMaxTensor = softmaxMaxUb[inIdx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset];
    }

    LocalTensor<T> tmpRowMaxUb = tmpBuff3.Get<T>();
    SoftmaxFlashV2<T, true, true, false, false, MLA_INT8_IFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
        mmResUb, tmpRowMaxUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
        softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);

    {
        PipeBarrier<PIPE_V>();
        uint32_t calQuantCount = dealRowCount;
        Sub(tmpRowMaxUb, tmpRowMaxUb, softmaxMaxUb[softmaxOutOffset], calQuantCount);
        PipeBarrier<PIPE_V>();
        Exp(tmpRowMaxUb, tmpRowMaxUb, calQuantCount);
        PipeBarrier<PIPE_V>();
        Adds(tmpRowMaxUb, tmpRowMaxUb, FLOAT_EPS, calQuantCount);
        PipeBarrier<PIPE_V>();
    }

    {
        if (info.isBmm2Output) {
            PipeBarrier<PIPE_V>();
            LocalTensor<T> tmpSoftmaxSumUb = softmaxTmpUb.template ReinterpretCast<T>();
            Brcb(tmpSoftmaxSumUb, softmaxSumUb[softmaxOutOffset], (dealRowCount + 7) / 8, {1, 8});
            PipeBarrier<PIPE_V>();
            RowDivs(mmResUb, mmResUb, tmpSoftmaxSumUb, dealRowCount, columnCount, actualColumnCount);
        }
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::Bmm2FDDataCopyOut(const ExtraInfoMla& info, LocalTensor<T>& bmm2ResUb,
                                                              uint32_t startRow, uint32_t dealRowCount,
                                                              uint32_t columnCount, uint32_t actualColumnCount)
{
    DealInvalidRows(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    if constexpr (BALANCE) {
        LocalTensor<T> tmp = outputQue1.AllocTensor<T>();
        DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
        outputQue1.EnQue(tmp);
        outputQue1.DeQue<T>();
        s2IdxFD = info.tndCoreStartKVSplitPos;

        uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.s1Idx);
        uint64_t offset = accumTmpOutNum * kvHeadNum * tndSgBasicSize * headDim +              // taskoffset
                          info.tndCoreStartKVSplitPos * kvHeadNum * tndSgBasicSize * headDim + // 份数offset
                          (mSizeVStart + startRow) * actualColumnCount;                        // m轴offset
        GlobalTensor<T> dst = accumOutGm[offset];
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(T);
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(dst, tmp, dataCopyParams);
        outputQue1.FreeTensor(tmp);
    } else {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(T);
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;

        LocalTensor<T> tmp = outputQue1.AllocTensor<T>();
        DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
        outputQue1.EnQue(tmp);
        outputQue1.DeQue<T>();

        size_t base = (info.bIdx * qHeadNum + info.n2Idx * gSize) * splitKVNum * headDim;
        // accumOutGm: batchQ * kvHeadNum * gSize * kvSplitPart_ * headDimAlign_
        DataCopyPad(accumOutGm[base + s2IdxFD * gSize * actualColumnCount + startRow * actualColumnCount +
                               mSizeVStart * actualColumnCount],
                    tmp, dataCopyParams);
        outputQue1.FreeTensor(tmp);
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T>& attenOutUb,
                                                            uint32_t startRow, uint32_t dealRowCount,
                                                            uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + (mSizeVStart + startRow) * actualColumnCount], attenOutUb,
                dataCopyParams);
}

template <typename IFAT>
__aicore__ inline void
MlaInt8Vector<IFAT>::Bmm2DataCopyOutTrans(const ExtraInfoMla& info, LocalTensor<OUT_T>& attenOutUb, uint32_t startRow,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[info.attenOutOffset + (mSizeVStart + startRow) * actualColumnCount], attenOutUb,
                dataCopyParams);
    return;
}

template <typename IFAT>
template <typename RT>
__aicore__ inline void MlaInt8Vector<IFAT>::DealInvalidRows(const ExtraInfoMla& info, LocalTensor<RT>& attenOutUb,
                                                            uint32_t startRow, uint32_t dealRowCount,
                                                            uint32_t columnCount, uint32_t actualColumnCount)
{
    if constexpr (LAYOUT_T != LAYOUT::TND) {
        if (!attenMaskFlag || (qSeqSize <= info.s2Size)) {
            return;
        }
    } else {
        if (!attenMaskFlag || (info.actS1Size <= info.s2Size)) {
            return;
        }
    }
    PipeBarrier<PIPE_V>();
    uint32_t s1Tok = qSeqSize - info.s2Size;
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        s1Tok = info.actS1Size - info.s2Size;
    }

    // BSH and TND
    uint32_t s1 = info.s1Idx * s1SizeSub + (mSizeVStart + startRow) / info.gSize;
    uint32_t gIdx = (mSizeVStart + startRow) % info.gSize;
    for (uint32_t i = 0; i < dealRowCount;) {
        if (s1 < s1Tok) {
            uint32_t gNum = info.gSize - gIdx;
            if (i + gNum > dealRowCount) {
                gNum = dealRowCount - i;
            }
            Duplicate(attenOutUb[i * columnCount], static_cast<RT>(FLOAT_ZERO), columnCount * gNum);
            i += gNum;
            s1++;
            gIdx = 0;
            continue;
        }
        break;
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::Bmm2ResCopyOut(const ExtraInfoMla& info, LocalTensor<T>& bmm2ResUb,
                                                           uint32_t startRow, uint32_t dealRowCount,
                                                           uint32_t columnCount, uint32_t actualColumnCount)
{
    if (publicPartials) {
        DealInvalidRows(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
        LocalTensor<T> tmp = outputQue1.AllocTensor<T>();
        DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
        outputQue1.EnQue(tmp);
        outputQue1.DeQue<T>();
        for (uint32_t row = 0; row < dealRowCount; ++row)
            DataCopyPad(publicOut[PublicRow(info, startRow + row) * headDim], tmp[row * columnCount],
                        DataCopyExtParams{1, uint32_t(headDim * sizeof(T)), 0, 0, 0});
        outputQue1.FreeTensor(tmp);
    }
    if constexpr (BALANCE) {
        if constexpr (FLASH_DECODE) {
            if (info.tndIsS2SplitCore) {
                Bmm2FDDataCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
            } else {
                Bmm2CastAndCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
            }
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
        }
    } else {
        if constexpr (FLASH_DECODE) {
            Bmm2FDDataCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
        }
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::Bmm2CastAndCopyOut(const ExtraInfoMla& info, LocalTensor<T>& bmm2ResUb,
                                                               uint32_t startRow, uint32_t dealRowCount,
                                                               uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
    { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    }

    DealInvalidRows(info, tmpBmm2ResCastTensor, startRow, dealRowCount, columnCount, actualColumnCount);
    outputQue1.EnQue(tmpBmm2ResCastTensor);
    outputQue1.DeQue<OUT_T>();
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, startRow, dealRowCount, columnCount, actualColumnCount);
    outputQue1.FreeTensor(tmpBmm2ResCastTensor);
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::DealBmm1ResBaseBlock(const ExtraInfoMla& info, uint32_t startRow,
                                                                 uint32_t dealRowCount, uint32_t columnCount,
                                                                 uint32_t actualColumnCount)
{
    uint32_t computeSize = dealRowCount * columnCount;
    LocalTensor<T> mmResUb = tmpBuff1.Get<T>();

    size_t batchBase = 0;

    uint64_t inOutGmOffset =
        (info.loop % MLA_INT8_PRE_LOAD_NUM_MLA) * mmResUbSize + (mSizeVStart + startRow) * columnCount;

    {
        LocalTensor<MM1_OUT_T> mm1ResInt32 = inputQue1.AllocTensor<MM1_OUT_T>();
        DataCopy(mm1ResInt32, mm1ResGm[inOutGmOffset + batchBase], computeSize);
        inputQue1.EnQue(mm1ResInt32);
        inputQue1.DeQue<MM1_OUT_T>();
        LocalTensor<T> mm1ResFp32 = mm1ResInt32.template ReinterpretCast<T>();
        RowMuls(mmResUb, mm1ResFp32, dequantScale1Ub[startRow * BLOCK_ELEMENT_NUM], dealRowCount, columnCount,
                actualColumnCount);
        inputQue1.FreeTensor(mm1ResInt32);
    }

    PipeBarrier<PIPE_V>();
    ElewiseCompute(info, mmResUb, tmpBuff2, startRow, dealRowCount, columnCount, actualColumnCount);

    LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(info, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    LocalTensor<KV_T> tmpMMResCastTensor = outputQue1.AllocTensor<KV_T>();

    {
        uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
        uint32_t idx = info.loop % (MLA_INT8_PRE_LOAD_NUM_MLA);
        LocalTensor<T> maxP = pMaxUb[idx * MLA_INT8_QMAX_UB_SIZE / sizeof(T) + baseOffset];

        LocalTensor<T> tmpRowMaxUb = tmpBuff3.Get<T>();
        LocalTensor<T> tmpSrcUb = tmpBuff2.Get<T>();

        // 防止tmpRowMaxUb 值为 0
        Brcb(maxP, tmpRowMaxUb, (dealRowCount + 7) / 8, {1, 8});

        LocalTensor<T> quantScaleP = tmpSrcUb;
        Duplicate(quantScaleP, scaleP1, dealRowCount * BLOCK_ELEMENT_NUM);
        PipeBarrier<PIPE_V>();
        Div(quantScaleP, quantScaleP, maxP, dealRowCount * BLOCK_ELEMENT_NUM);
        PipeBarrier<PIPE_V>();
        RowMuls(mmResUb, mmResUb, quantScaleP, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();

        // 先cast 到 half
        LocalTensor<half> tmpf16Ub = mmResUb.template ReinterpretCast<half>();
        Cast(tmpf16Ub, mmResUb, AscendC::RoundMode::CAST_RINT, computeSize);
        PipeBarrier<PIPE_V>();
        Cast(tmpMMResCastTensor, tmpf16Ub, AscendC::RoundMode::CAST_RINT, computeSize);
    }

    outputQue1.EnQue(tmpMMResCastTensor);
    outputQue1.DeQue<KV_T>();

    DataCopy(vec1ResGm[inOutGmOffset + batchBase], tmpMMResCastTensor, computeSize);
    outputQue1.FreeTensor(tmpMMResCastTensor);
}

// 由于S在GM上已经完成atomic，此处只需要把数据拷贝出来有fp16转成fp32

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::ProcessVec1Inner(const ExtraInfoMla& info)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
    mSplitSize = mSplitSize / 8 * 8;
    if (mSplitSize > mSizeVector) {
        mSplitSize = mSizeVector;
    }
    uint32_t loopCount = (mSizeVector + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSizeVector - (loopCount - 1) * mSplitSize;

    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        {
            DealBmm1ResBaseBlock(info, i * mSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign,
                                 info.actualSingleProcessSInnerSize);
        }
    }

    if (info.s2Idx == info.curSInnerLoopTimes - 1) {
        uint32_t outIdx = info.loop % (MLA_INT8_PRE_LOAD_NUM_MLA);
        uint32_t softmaxOutOffset = outIdx * BUFFER_SIZE_BYTE_2K / sizeof(T);
        LocalTensor<T> sumTensor = softmaxSumUb[softmaxOutOffset];
        LocalTensor<T> maxTensor = softmaxMaxUb[softmaxOutOffset];
        ExportPublicLse(info, sumTensor, maxTensor);

        if constexpr (BALANCE) {
            if (!info.tndIsS2SplitCore) {
                if (softmaxLseFlag) {
                    LocalTensor<T> lseSumUb = tmpBuff1.Get<T>(BUFFER_SIZE_BYTE_2K);
                    LocalTensor<T> lseMaxUb = tmpBuff1.GetWithOffset<T>(BUFFER_SIZE_BYTE_2K, BUFFER_SIZE_BYTE_2K);
                    Brcb(lseSumUb, sumTensor, (mSizeVector + BLOCK_ELEMENT_NUM - 1) / BLOCK_ELEMENT_NUM,
                         {1, BLOCK_ELEMENT_NUM});
                    PipeBarrier<PIPE_V>();
                    Brcb(lseMaxUb, maxTensor, (mSizeVector + BLOCK_ELEMENT_NUM - 1) / BLOCK_ELEMENT_NUM,
                         {1, BLOCK_ELEMENT_NUM});
                    PipeBarrier<PIPE_V>();
                    SoftmaxLseCopyOut(info, lseSumUb, lseMaxUb, mSizeVector);
                }
                return;
            }
        }

        if constexpr (FLASH_DECODE) {
            ComputeLogSumExpAndCopyToGm(info, sumTensor, maxTensor);
            return;
        }

        if (softmaxLseFlag) {
            LocalTensor<T> lseSumUb = tmpBuff1.Get<T>(BUFFER_SIZE_BYTE_2K);
            LocalTensor<T> lseMaxUb = tmpBuff1.GetWithOffset<T>(BUFFER_SIZE_BYTE_2K, BUFFER_SIZE_BYTE_2K);
            Brcb(lseSumUb, sumTensor, (mSizeVector + BLOCK_ELEMENT_NUM - 1) / BLOCK_ELEMENT_NUM,
                 {1, BLOCK_ELEMENT_NUM});
            PipeBarrier<PIPE_V>();
            Brcb(lseMaxUb, maxTensor, (mSizeVector + BLOCK_ELEMENT_NUM - 1) / BLOCK_ELEMENT_NUM,
                 {1, BLOCK_ELEMENT_NUM});
            PipeBarrier<PIPE_V>();
            SoftmaxLseCopyOut(info, lseSumUb, lseMaxUb, mSizeVector);
        }
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::SoftmaxLseCopyOut(const ExtraInfoMla& info, LocalTensor<T>& lseSumUb,
                                                              LocalTensor<T>& lseMaxUb, uint32_t dealRowCount)
{
    uint64_t dealRowCountAlign = dealRowCount * FP32_ONE_BLOCK_SIZE;
    LocalTensor<T> softmaxlseUb = outputQue2.template AllocTensor<T>();
    ComputeSoftmaxLse(softmaxlseUb, lseSumUb, lseMaxUb, dealRowCountAlign);
    uint64_t curS1Size = LAYOUT_T == LAYOUT::TND ? info.actS1Size : qSeqSize;
    DealSoftmaxLseInvalidRows(softmaxlseUb, lseMaxUb, dealRowCount, curS1Size, info.s1Idx * s1SizeSub);

    outputQue2.EnQue(softmaxlseUb);
    outputQue2.DeQue<T>();

    uint32_t startS1Idx = info.s1Idx * s1SizeSub + mSizeVStart / info.gSize;
    uint32_t startGIdx = mSizeVStart % info.gSize;
    uint32_t endS1Idx = info.s1Idx * s1SizeSub + (mSizeVStart + dealRowCount - 1) / info.gSize;
    uint32_t endGIdx = (mSizeVStart + dealRowCount - 1) % info.gSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    if constexpr (LAYOUT_T == LAYOUT::TND) {
        uint64_t tokenPrefixSum = (info.bIdx == 0) ? 0 : actualSeqLengthsGmQ.GetValue(info.bIdx - 1);
        for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
            if (s1Idx != endS1Idx) {
                curDealRowCount = info.gSize - startGIdx;
            } else {
                curDealRowCount = endGIdx + 1 - startGIdx;
            }
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = curDealRowCount;
            dataCopyParams.blockLen = sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            uint64_t total = actualSeqLengthsGmQ.GetValue(batchSize - 1);
            outOffset = (info.n2Idx * info.gSize + startGIdx) * total + tokenPrefixSum + s1Idx;
            dataCopyParams.dstStride = (total - 1) * sizeof(T);
            DataCopyPad(softmaxLseGm[outOffset], softmaxlseUb[ubOffset], dataCopyParams);
            startGIdx = 0;
            ubOffset += curDealRowCount * FP32_ONE_BLOCK_SIZE;
        }
    } else { // BSH BSND
        uint64_t bN2Offset = info.bIdx * qHeadNum * qSeqSize + info.n2Idx * info.gSize * qSeqSize;
        for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
            outOffset = bN2Offset + startGIdx * qSeqSize + s1Idx;
            if (s1Idx != endS1Idx) {
                curDealRowCount = info.gSize - startGIdx;
            } else {
                curDealRowCount = endGIdx + 1 - startGIdx;
            }
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = curDealRowCount;
            dataCopyParams.blockLen = sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = (qSeqSize - 1) * sizeof(T);
            DataCopyPad(softmaxLseGm[outOffset], softmaxlseUb[ubOffset], dataCopyParams);
            startGIdx = 0;
            ubOffset += curDealRowCount * FP32_ONE_BLOCK_SIZE;
        }
    }
    outputQue2.FreeTensor(softmaxlseUb);
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::ComputeSoftmaxLse(LocalTensor<T> softmaxlseUb, LocalTensor<T>& lseSumUb,
                                                              LocalTensor<T>& lseMaxUb, uint32_t dealRowCountAlign)
{
    Log(softmaxlseUb, lseSumUb, dealRowCountAlign);
    PipeBarrier<PIPE_V>();
    Add(softmaxlseUb, softmaxlseUb, lseMaxUb, dealRowCountAlign);
    PipeBarrier<PIPE_V>();
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::DealSoftmaxLseInvalidRows(LocalTensor<T>& softmaxlseUb,
                                                                      LocalTensor<T>& lseMaxUb, uint32_t dealRowCount,
                                                                      uint64_t s1Size, uint32_t curS1Idx)
{
    if (!attenMaskFlag) {
        return;
    }

    if (s1Size < curActualSeqLen) {
        return;
    }

    uint64_t seqLenGap = s1Size - curActualSeqLen;
    uint64_t startS1Idx = curS1Idx + mSizeVStart / gSize;
    if (startS1Idx < seqLenGap) {
        SoftMaxShapeInfo softmaxShapeInfo{static_cast<uint32_t>(dealRowCount), static_cast<uint32_t>(BLOCK_ELEMENT_NUM),
                                          static_cast<uint32_t>(dealRowCount),
                                          static_cast<uint32_t>(BLOCK_ELEMENT_NUM)};
        AdjustSoftMaxRes<T, T>(softmaxlseUb, lseMaxUb, negativeIntScalar, -FLOAT_INF, softmaxShapeInfo);
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::DealQuantBmm2ResBaseBlock(const ExtraInfoMla& info, uint32_t startRow,
                                                                      uint32_t dealRowCount, uint32_t columnCount,
                                                                      uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
    uint32_t idx = info.loop % (MLA_INT8_PRE_LOAD_NUM_MLA);
    LocalTensor<MM2_OUT_T> bmm2ResUb = tmpBuff1.Get<MM2_OUT_T>();
    bmm2ResUb.SetSize(vec2ComputeSize);

    size_t batchBase = 0;

    uint64_t inOutBaseOffset = (mSizeVStart + startRow) * columnCount;
    uint64_t srcGmOffset = (info.loop % MLA_INT8_PRE_LOAD_NUM_MLA) * bmm2ResUbSize + inOutBaseOffset;
    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputQue1.AllocTensor<MM2_OUT_T>();
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset + batchBase], vec2ComputeSize);
    inputQue1.EnQue(tmpBmm2ResUb);
    inputQue1.DeQue<MM2_OUT_T>();

    LocalTensor<T> maxP = pMaxUb[idx * MLA_INT8_QMAX_UB_SIZE / sizeof(T) + baseOffset];
    // pMaxUb(maxPU32) [dealRowCount, 32B / sizeof(float)] fp32 -> maxPDoubleU32 [dealRowCount, 32B / sizeof(half)] fp32
    LocalTensor<uint32_t> maxPU32 = maxP.template ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> maxPDoubleU32 = tmpBuff3.Get<uint32_t>();
    uint32_t repeatNums = (dealRowCount * BLOCK_ELEMENT_NUM + REPEAT_ELEMENT_NUM - 1) / REPEAT_ELEMENT_NUM;
    ShiftLeft(maxPDoubleU32, maxPU32, static_cast<uint32_t>(0), static_cast<uint64_t>(FP32_MAX_MASK_ELEMENT_NUM),
              repeatNums, {2, 1, 16, 8});
    ShiftLeft(maxPDoubleU32[BLOCK_ELEMENT_NUM], maxPU32, static_cast<uint32_t>(0),
              static_cast<uint64_t>(FP32_MAX_MASK_ELEMENT_NUM), repeatNums, {2, 1, 16, 8});
    PipeBarrier<PIPE_V>();
    LocalTensor<T> maxPFp32 = tmpBuff3.Get<T>();
    LocalTensor<MM2_OUT_T> maxPFp16 = tmpBuff3.Get<MM2_OUT_T>();
    // same buffer fp32 to fp16
    Cast(maxPFp16, maxPFp32, RoundMode::CAST_RINT,
         dealRowCount * BLOCK_ELEMENT_NUM * static_cast<uint32_t>(sizeof(T) / sizeof(MM2_OUT_T))); // fp32 to fp16
    PipeBarrier<PIPE_V>();
    RowMuls(bmm2ResUb, tmpBmm2ResUb, maxPFp16, dealRowCount, columnCount, actualColumnCount); // max(p)
    inputQue1.FreeTensor(tmpBmm2ResUb);
    PipeBarrier<PIPE_V>();

    // 除第一个循环外，均需要更新中间计算结果
    if (!info.isFirstSInnerLoop) {
        if (info.isLastTask) {
            event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        }
        LocalTensor<MM2_OUT_T> bmm2ResPreUb = inputQue2.AllocTensor<MM2_OUT_T>();
        uint64_t vec2ResGmOffset = ((info.loop - 1) % MLA_INT8_PRE_LOAD_NUM_MLA) * bmm2ResUbSize + inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset + batchBase], vec2ComputeSize);
        inputQue2.EnQue(bmm2ResPreUb);
        inputQue2.DeQue<MM2_OUT_T>();

        LocalTensor<MM2_OUT_T> tmpTensor = tmpBuff3.Get<MM2_OUT_T>();
        Cast(tmpTensor, softmaxExpUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + startRow], RoundMode::CAST_RINT,
             dealRowCount);
        PipeBarrier<PIPE_V>();
        LocalTensor<MM2_OUT_T> tmpExpBrcbResUb = tmpBuff2.Get<MM2_OUT_T>();
        Brcb(tmpExpBrcbResUb, tmpTensor, (dealRowCount + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, tmpExpBrcbResUb, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        inputQue2.FreeTensor(bmm2ResPreUb);
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        PipeBarrier<PIPE_V>();
        LocalTensor<T> bmm2ResUb1 = tmpBuff2.Get<T>();
        PipeBarrier<PIPE_V>();
        Cast(bmm2ResUb1, bmm2ResUb, RoundMode::CAST_NONE, vec2ComputeSize);
        PipeBarrier<PIPE_V>();
        Muls(bmm2ResUb1, bmm2ResUb1, (T)(1024.0), dealRowCount * columnCount);

        LocalTensor<T> tmpSumBrcbResUb = tmpBuff1.Get<T>();
        Brcb(tmpSumBrcbResUb, softmaxSumUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + startRow], (dealRowCount + 7) / 8,
             {1, 8});
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUb1, bmm2ResUb1, tmpSumBrcbResUb, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Muls(bmm2ResUb1, bmm2ResUb1, deqScale2Val * scaleP2, dealRowCount * columnCount);
        PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(info, bmm2ResUb1, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        PipeBarrier<PIPE_V>();
        LocalTensor<MM2_OUT_T> tmpBmm2Res = outputQue1.AllocTensor<MM2_OUT_T>();
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(tmpBmm2Res);
        outputQue1.DeQue<MM2_OUT_T>();
        uint64_t vec2ResGmOffset = (info.loop % MLA_INT8_PRE_LOAD_NUM_MLA) * bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vec2ResGmOffset + batchBase], tmpBmm2Res, vec2ComputeSize);
        outputQue1.FreeTensor(tmpBmm2Res);
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::ProcessVec2Inner(const ExtraInfoMla& info)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
    if (mSplitSize > mSizeVector) {
        mSplitSize = mSizeVector;
    }
    uint32_t loopCount = (mSizeVector + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSizeVector - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        {
            DealQuantBmm2ResBaseBlock(info, i * mSplitSize, dealSize, headDimAlign, headDim);
        }
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::CalcParams(uint32_t loop, ExtraInfoMla& info, TaskContext& task)
{
    info.loop = loop;
    info.bIdx = task.bidx;
    info.s1Idx = task.s1idx;
    info.s2Idx = task.s2idx;
    info.curSInnerLoopTimes = task.s2loops;
    info.s1Size = task.s1Size;
    info.s2Size = task.s2Size;
    info.gIdx = task.gidx;
    info.gSize = (task.gidx == gOuter - 1) ? gSizeTail : gSizeSub;
    info.tndIsS2SplitCore = task.tndIsS2SplitCore;
    info.tndCoreStartKVSplitPos = task.tndCoreStartKVSplitPos;
    info.isBmm2Output = false;
    info.actS1Size = task.actS1Size;
    info.isValid = task.isValid;
    info.isLastTask = task.isLastTask;

    if ASCEND_IS_AIV {
        info.mSize = info.gSize * info.s1Size;
        {
            info.mSizeV = (info.mSize + 1) / 2;
        }
        info.mSizeVStart = 0;
        if (subBlockNum == 1) { // CV 1:1
            info.mSizeV = info.mSize;
        } else if (tmpBlockIdx % 2 == 1) {
            info.mSizeVStart = info.mSizeV;
            info.mSizeV = info.mSize - info.mSizeV;
        }
    }

    {
        info.isChangeBatch = false;
    }

    if constexpr (BALANCE) {
        info.isFirstSInnerLoop = task.isFirstLoop;
    } else {
        info.isFirstSInnerLoop = (info.s2Idx == 0);
    }
    if (info.isFirstSInnerLoop) {
        bn2IdxInCurCore++;
    }
    info.bn2IdxInCurCore = bn2IdxInCurCore - 1;
    if (info.isFirstSInnerLoop) {
        if constexpr (BALANCE) {
            uint64_t actualSeqQPrefixSum;
            if constexpr (LAYOUT_T == LAYOUT::TND) {
                actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(info.bIdx - 1);
            } else {
                actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : info.bIdx * qSeqSize;
            }
            info.tndBIdxOffset = actualSeqQPrefixSum * qHeadNum * headDim;
            uint64_t tndBIdxRopeOffset = actualSeqQPrefixSum * qHeadNum * headDimRope;
            tensorACoreOffset = info.tndBIdxOffset + (info.s1Idx * s1SizeSub * qHeadNum * headDim);
            tensorARopeCoreOffset = tndBIdxRopeOffset + (info.s1Idx * s1SizeSub * qHeadNum * headDimRope);
            tensorBCoreOffset = info.bIdx * kvSeqSize * kvHeadNum * headDim + info.n2Idx * headDim;
            tensorBRopeCoreOffset = info.bIdx * kvSeqSize * kvHeadNum * headDimRope + info.n2Idx * headDimRope;
        } else {
            // B,S,N2,G,D
            tensorACoreOffset = info.bIdx * qSeqSize * qHeadNum * headDim +
                                info.s1Idx * s1SizeSub * qHeadNum * headDim + info.n2Idx * gSize * headDim;
            tensorARopeCoreOffset = info.bIdx * qSeqSize * qHeadNum * headDimRope +
                                    info.s1Idx * s1SizeSub * qHeadNum * headDimRope + info.n2Idx * gSize * headDimRope;
            // B,S2,N2,D
            tensorBCoreOffset = info.bIdx * kvSeqSize * kvHeadNum * headDim + info.n2Idx * headDim;
            tensorBRopeCoreOffset = info.bIdx * kvSeqSize * kvHeadNum * headDimRope + info.n2Idx * headDimRope;

            if constexpr (FLASH_DECODE) {
                tensorBCoreOffset += s2IdxFD * sInnerLoopSize * kvHeadNum * headDim;
                tensorBRopeCoreOffset += s2IdxFD * sInnerLoopSize * kvHeadNum * headDimRope;
            }
        }
    }
    info.tensorAOffset = tensorACoreOffset;
    info.tensorARopeOffset = tensorARopeCoreOffset;
    {
        info.tensorBOffset = tensorBCoreOffset + info.s2Idx * singleProcessSInnerSize * kvHeadNum * headDim;
        info.tensorBRopeOffset = tensorBRopeCoreOffset + info.s2Idx * singleProcessSInnerSize * kvHeadNum * headDimRope;
    }
    info.attenOutOffset = tensorACoreOffset;

    if constexpr (BALANCE) {
        info.actualSingleProcessSInnerSize = task.s2SizeTail;
    } else {
        info.actualSingleProcessSInnerSize = singleProcessSInnerSize;
        if (info.s2Idx == info.curSInnerLoopTimes - 1) {
            info.actualSingleProcessSInnerSize = task.s2SizeTail;
        }
    }

    info.actualSingleProcessSInnerSizeAlign = Align((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)BYTE_BLOCK);

    info.antiqKeyParamOffset = info.n2Idx * headDim;
    info.antiqValueParamOffset = info.n2Idx * headDim;
    info.antiqKeyRopeParamOffset = info.n2Idx * headDimRope;

    uint64_t sInnerOffsetDataSize = info.s2Idx * singleProcessSInnerSize;
    if constexpr (FLASH_DECODE) {
        sInnerOffsetDataSize += s2IdxFD * sInnerLoopSize;
    }

    attenMaskCoreOffset = info.bIdx * attenMaskSize;
    info.attenMaskOffset = attenMaskCoreOffset + sInnerOffsetDataSize;

    info.s2BatchOffset = s2BatchBaseOffset + sInnerOffsetDataSize;
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::ProcessVec2L(const ExtraInfoMla& info)
{
    mSizeVector = info.mSizeV;
    mSizeVStart = info.mSizeVStart;

    if (mSizeVector == 0) {
        return;
    }
    {
        if (info.isBmm2Output) {
            return;
        }
    }
    {
        ProcessVec2Inner(info);
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::ProcessVec1L(const ExtraInfoMla& info)
{
    mSizeVector = info.mSizeV;
    mSizeVStart = info.mSizeVStart;

    if (mSizeVector == 0) {
        return;
    }
    {
        if (info.isFirstSInnerLoop) {
            // preToken_Head, Query->deqScale1Gm的3种场景下(BSND->BSN;BNSD->BNS;TND->TN), query的偏移除以D即为deqScale1Gm的偏移
            uint64_t quantScale1Offset = info.tensorAOffset / headDim;
            LocalTensor<T> tmpDequantScale1 = inputQue2.AllocTensor<T>();
            if ((mSizeVector % 8) == 0) { // 32B对齐
                DataCopy(tmpDequantScale1, deqScale1Gm[quantScale1Offset + mSizeVStart], mSizeVector);
            } else {
                DataCopyExtParams intriParams;
                intriParams.blockLen = mSizeVector * sizeof(T);
                intriParams.blockCount = 1;
                intriParams.dstStride = 0;
                intriParams.srcStride = 0;
                DataCopyPadExtParams<T> padParams;
                padParams.isPad = true;
                padParams.leftPadding = 0;
                padParams.rightPadding = 8 - (mSizeVector % 8);
                padParams.paddingValue = 0;
                DataCopyPad(tmpDequantScale1, deqScale1Gm[quantScale1Offset + mSizeVStart], intriParams, padParams);
            }
            inputQue2.EnQue(tmpDequantScale1);
            inputQue2.DeQue<T>();
            for (uint32_t row = 0; row < mSizeVector; ++row) {
                T scale = tmpDequantScale1.GetValue(row);
                if (!(scale <= FLOAT_MAX && scale >= -FLOAT_MAX)) {
                    Trap();
                }
            }
            Muls(tmpDequantScale1, tmpDequantScale1, deqScaleKey, mSizeVector);
            PipeBarrier<PIPE_V>();
            Muls(tmpDequantScale1, tmpDequantScale1, static_cast<T>(tilingData->baseParams.scaleValue),
                 mSizeVector); // 乘以1/sqrt(d)
            PipeBarrier<PIPE_V>();
            Brcb(dequantScale1Ub, tmpDequantScale1, (mSizeVector + 7) / 8, {1, 8});
            inputQue2.FreeTensor(tmpDequantScale1);
            PipeBarrier<PIPE_V>();
        }
    }
    ProcessVec1Inner(info);
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::ComputeMm1(const ExtraInfoMla& info)
{
    if (info.isChangeBatch) {
        InitKeyGm(info.bIdx);
        matmulService.UpdateKey(keyGm);
    }
    matmulService.ComputeMm1(info);
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::ComputeMm2(const ExtraInfoMla& info)
{
    if (info.isChangeBatch) {
        InitValueGm(info.bIdx);
        matmulService.UpdateValue(valueGm);
    }
    matmulService.ComputeMm2(info);
}

template <typename IFAT>
__aicore__ inline uint64_t MlaInt8Vector<IFAT>::GetBalanceActualSeqLengths(QueryLengths& actualSeqLengths, int bIdx)
{
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        if (bIdx > 0) {
            return actualSeqLengths.GetValue(bIdx) - actualSeqLengths.GetValue(bIdx - 1);
        } else if (bIdx == 0) {
            return actualSeqLengths.GetValue(0);
        } else {
            return 0;
        }
    } else {
        return qSeqSize;
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::GetTNDAxisStartId(uint32_t& bStart, uint32_t& n2Start,
                                                              uint32_t& s1OuterStart, uint32_t& s2Start, int bEnd,
                                                              int n2End, int s1OuterIdxEnd, int s2End)
{
    if constexpr (BALANCE) {
        if (aiCoreIdx == 0) {
            bStart = 0;
            n2Start = 0;
            s1OuterStart = 0;
            s2Start = 0;
        } else {
            uint32_t actualSeqQPrev = GetBalanceActualSeqLengths(actualSeqLengthsGmQ, bEndPrev);
            uint32_t actualSeqKVPrev = KvLength(bEndPrev);
            uint32_t s1PrevSplitNum = (actualSeqQPrev + s1SizeSub - 1) / s1SizeSub;
            uint32_t s2PrevSplitNum = (actualSeqKVPrev + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
            bStart = bEndPrev;
            n2Start = n2EndPrev;
            s1OuterStart = s1OuterIdxEndPrev;
            if (s2EndPrev >= s2PrevSplitNum - 1) { // 上个核把S2处理完了
                s2Start = 0;
                if (s1OuterIdxEndPrev >= s1PrevSplitNum - 1) { // 上个核把S1处理完了
                    s1OuterStart = 0;
                    if constexpr (n2EndPrev >= kvHeadNum - 1) { // 上个核把N2处理完了
                        n2Start = 0;
                        bStart++;
                    } else {
                        n2Start++;
                    }
                    if (bEndPrev >= batchSize) {
                    }
                } else {
                    s1OuterStart++;
                }
            } else { // 上个核没有把S2处理完，只需要更新s2Start，其他与上个核保持一致
                s2Start = s2EndPrev + 1;
            }
        }
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Vector<IFAT>::CopyFixedUbToGm(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                                                            size_t size)
{
    LocalTensor<T> tmp = outputQue2.template AllocTensor<T>();
    Brcb(tmp, src, (mSizeVector + 7) / 8, {1, 8});
    outputQue2.EnQue(tmp);
    outputQue2.DeQue();
    DataCopy(dst, tmp, size);
    outputQue2.FreeTensor(tmp);
}

template <typename IFAT>
__aicore__ inline uint64_t MlaInt8Vector<IFAT>::CalcAccumOffset(uint32_t bIdx, uint32_t s1Idx)
{
#ifdef ASCENDC_CPU_DEBUG
    const uint32_t* balanceFDCoreBArr = tilingData->tndSplitCoreParams.balanceFDCoreBArr;
    const uint32_t* balanceFDCoreS1Arr = tilingData->tndSplitCoreParams.balanceFDCoreS1Arr;
    const uint32_t* balanceFDCoreKVSplitArr = tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr;
#else
    uint32_t balanceFDCoreBArr[MLA_INT8_ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreBArr)];
    uint32_t balanceFDCoreS1Arr[MLA_INT8_ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreS1Arr)];
    uint32_t balanceFDCoreKVSplitArr[MLA_INT8_ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr)];
    copy_data_align64((uint8_t*)balanceFDCoreBArr, (uint8_t*)(tilingData->tndSplitCoreParams.balanceFDCoreBArr),
                      sizeof(balanceFDCoreBArr));
    copy_data_align64((uint8_t*)balanceFDCoreS1Arr, (uint8_t*)(tilingData->tndSplitCoreParams.balanceFDCoreS1Arr),
                      sizeof(balanceFDCoreS1Arr));
    copy_data_align64((uint8_t*)balanceFDCoreKVSplitArr,
                      (uint8_t*)(tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr),
                      sizeof(balanceFDCoreKVSplitArr));
#endif
    uint64_t accumTmpOutNum = 0;
    int taskId = 0;
    while (balanceFDCoreBArr[taskId] != bIdx || balanceFDCoreS1Arr[taskId] != s1Idx) {
        accumTmpOutNum += balanceFDCoreKVSplitArr[taskId]; // 计算前面的workspace数
        taskId++;
    }
    return accumTmpOutNum;
}

} // namespace flash_attn::mla::Int8
