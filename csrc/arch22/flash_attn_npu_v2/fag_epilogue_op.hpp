/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_OP_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_OP_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "fag_block.h"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "kernel_operator.h"
#include "kernel_common_fag.hpp"
#include "fag_common/common_header.h"

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
constexpr int64_t CAUSAL_COMPRESS_MODE = 1;
constexpr int64_t BAND_COMPRESS_MODE = 2;

template <
    class OutputType_,
    class InputType_,
    uint32_t INPUT_LAYOUT_,
    uint32_t IS_DROP_,
    uint32_t IS_ATTEN_MASK_,
    class TilingData,
    bool HAS_SOFTCAP_
>
class BlockEpilogue<
    EpilogueAtlasA2SameAbVec<INPUT_LAYOUT_, IS_DROP_, IS_ATTEN_MASK_, HAS_SOFTCAP_>,
    OutputType_,
    InputType_,
    TilingData>
{
public:
    using DispatchPolicy = EpilogueAtlasA2SameAbVec<INPUT_LAYOUT_, IS_DROP_, IS_ATTEN_MASK_, HAS_SOFTCAP_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using T1 = InputType_;
    using T2 = OutputType_;
    static constexpr uint32_t INPUT_LAYOUT = INPUT_LAYOUT_;
    static constexpr bool IS_DROP = IS_DROP_;
    static constexpr bool IS_ATTEN_MASK = IS_ATTEN_MASK_;
    static constexpr bool HAS_SOFTCAP = HAS_SOFTCAP_;

    AscendC::TPipe *pipe;
    TBuf<> unifiedBuffer;

    uint32_t coreNum;
    uint32_t cubeCoreNum;
    uint32_t cBlockIdx;
    uint32_t cCubeBlockIdx;
    uint32_t cSubIdx;

    uint32_t vecBlockNum;

    GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, forwardResGm;
    GlobalTensor<uint8_t> maskWorkSpaceGm, attenMaskU8Gm, dropMaskGm;
    GlobalTensor<float> softmaxLseGm;

    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, sfmgWorkspaceGm;

    GlobalTensor<T1> dropWorkSpaceGm, mulWorkSpaceGm;

    GlobalTensor<T2> mm1WorkspaceGm;
    GlobalTensor<T2> mm2WorkspaceGm;

    TBuf<AscendC::TPosition::A1> queryBufL1;
    LocalTensor<T1> qL1Tensor;
    TBuf<AscendC::TPosition::A1> keyBufL1;
    LocalTensor<T1> vL1Tensor;
    LocalTensor<T1> kL1Tensor;
    TBuf<AscendC::TPosition::A1> dsBufL1;
    LocalTensor<T1> dsL1Tensor;
    LocalTensor<T1> dxL1Tensor;

    __gm__ uint8_t *actual_seq_qlen_addr;
    __gm__ uint8_t *actual_seq_kvlen_addr;

    GlobalTensor<float> dvGm;

    SoftMaxTiling softmaxTilingData;

    constexpr static uint32_t BNGSD = 0;
    constexpr static uint32_t SBNGD = 1;
    constexpr static uint32_t BSNGD = 2;
    constexpr static uint32_t ENABLE = 1;

    constexpr static uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {4, 5, 6};
    static constexpr uint64_t SYNC_C1_V1_FLAG[3] = {1, 2, 3};
    static constexpr uint64_t SYNC_C2_V1_FLAG[3] = {7, 8, 9};

    float keepProb;
    float scaleValue;
    float softcapValue;
    int64_t s1Token;
    int64_t s2Token;
    int64_t actualCalcS1Token;
    int64_t actualCalcS2Token;
    bool dropBitMode;

    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t value_d;
    int64_t dAlign;
    int64_t value_dAlign;
    int64_t attenMaskDimS2;
    int64_t t1;

    uint32_t baseMN;
    uint32_t cubeBaseMN;

    int64_t s1Outer;
    uint32_t s1CvInner;
    uint32_t s1CvTail;
    int64_t s2Outer;
    uint32_t s2CvInner;
    uint32_t s2CvTail;

    int64_t sfmgOffset = 0;
    uint32_t preS1Idx = -1;

    int64_t baseIdx{0};
    int64_t bDimIdx{0};
    int64_t n2DimIdx{0};
    int64_t gDimIdx{0};
    int64_t s1oDimIdx{0};
    int64_t s2oCvDimIdx{0};

    int32_t isStart = 1;
    uint32_t pingpongIdx = 1;
    int32_t vecLoopStart;
    int32_t vecLoopEnd;

    uint32_t s1VecLoop = 0;
    uint32_t s1VecSize = 0;
    uint32_t s1ExtendSubGraph = 0;
    uint32_t s2Extend = 0;
    uint32_t s2ExtendAlign = 0;
    uint32_t s2VecLoop = 0;
    uint32_t s2VecSize = 0;

    int64_t dqOutBase{0};
    int64_t kvOutBase{0};
    int64_t dqOutIdx{0};
    int64_t kvOutIdx{0};
    int64_t dqOutArr[24];
    int64_t kvOutArr[24];

    int64_t blockStartIdx = 0;
    int64_t bandIdx = 0;
    int64_t compressMode = 0;

    constexpr static uint32_t T2Begin = 0;
    constexpr static uint32_t T1Begin = 33 * 1024;
    constexpr static uint32_t BoolBegin = 50 * 1024;
    constexpr static uint32_t T2BlockBegin = 58 * 1024;
    constexpr static uint32_t U8Begin = 66 * 1024;
    constexpr static uint32_t DbBegin = 74 * 1024;

    constexpr static uint32_t DTYPE_FACTOR = sizeof(T2) / sizeof(T1);
    constexpr static uint32_t cal_block_num = 32 / sizeof(T2);
    constexpr static uint32_t cal_repeat_num = 256 / sizeof(T2);
    constexpr static uint32_t input_block_num = 32 / sizeof(T1);
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t INPUT_NUMS = 2;
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static int64_t C0_SIZE = 16;
    constexpr static int64_t VEC_REPEAT = 8;
    constexpr static uint32_t PREFIX_COMPRESS_CAUSAL_S_SIZE = 2048;
    constexpr static uint32_t PREFIX_COMPRESS_ALL_MASK_S1_SIZE = 1024;
    constexpr static int64_t GM_DOUBLE_BUFFER = 2;
    constexpr static int64_t SOFTCAP_UB_OFFSET = 32 * 1024;
    constexpr static int64_t TMP_UB_OFFSET = 148 * 1024;
    constexpr static int64_t SFMG_UB_OFFSET = (148 + 33) * 1024;
    constexpr static int64_t TMP_UB_SIZE = 33 * 1024;
    constexpr static int64_t SFMG_UB_SIZE = 8 * 1024;
    constexpr static int64_t TOTAL_SIZE = 189 * 1024;

    constexpr static uint32_t MMAD_BASE_SIZE = 128;
    constexpr static uint32_t S_BASE_SIZE = 512;
    constexpr static uint32_t S_SPLITT_SIZE = 256;
    constexpr static uint32_t L1_CACHE_CAPACITY_LIMIT = 14;
    constexpr static uint32_t DIM_64 = 64;
    constexpr static uint32_t VEC_S2_LEN = 256;
    constexpr static int8_t OUTIDX= -1;
    enum class AttenMaskCompress {
        Empty = 0,
        PreOnly = 1,
        NextOnly = 2,
        All = 3
    };
    AttenMaskCompress AttenBandMode = AttenMaskCompress::All;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, AscendC::TPipe *pipe_in,
                  __gm__ uint8_t *query,
                  __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx,
                  __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask, __gm__ uint8_t *forward_res,
                  __gm__ uint8_t *softmax_lse,
                  __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,
                  __gm__ uint8_t *dq, __gm__ uint8_t *dk,
                  __gm__ uint8_t *dv,
                  __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_in, TBuf<>& buf)
    {
        keyGm.SetGlobalBuffer((__gm__ T1 *)key);
        valueGm.SetGlobalBuffer((__gm__ T1 *)value);
        dxGm.SetGlobalBuffer((__gm__ T1 *)dx);
        queryGm.SetGlobalBuffer((__gm__ T1 *)query);
        forwardResGm.SetGlobalBuffer((__gm__ T1 *)forward_res);
        attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *)atten_mask);
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmax_lse);

        cBlockIdx = GetBlockIdx();
        cCubeBlockIdx = cBlockIdx / 2;
        cSubIdx = cBlockIdx % 2;

        __gm__ TilingData *tilingData = reinterpret_cast<__gm__ TilingData *>(tiling_in);

        // set softmax tilingdata
        softmaxTilingData.srcM = tilingData->softmaxTilingData.srcM;
        softmaxTilingData.srcK = tilingData->softmaxTilingData.srcK;
        softmaxTilingData.srcSize = tilingData->softmaxTilingData.srcSize;
        softmaxTilingData.outMaxM = tilingData->softmaxTilingData.outMaxM;
        softmaxTilingData.outMaxK = tilingData->softmaxTilingData.outMaxK;
        softmaxTilingData.outMaxSize = tilingData->softmaxTilingData.outMaxSize;
        softmaxTilingData.splitM = tilingData->softmaxTilingData.splitM;
        softmaxTilingData.splitK = tilingData->softmaxTilingData.splitK;
        softmaxTilingData.splitSize = tilingData->softmaxTilingData.splitSize;
        softmaxTilingData.reduceM = tilingData->softmaxTilingData.reduceM;
        softmaxTilingData.reduceK = tilingData->softmaxTilingData.reduceK;
        softmaxTilingData.reduceSize = tilingData->softmaxTilingData.reduceSize;
        softmaxTilingData.rangeM = tilingData->softmaxTilingData.rangeM;
        softmaxTilingData.tailM = tilingData->softmaxTilingData.tailM;
        softmaxTilingData.tailSplitSize = tilingData->softmaxTilingData.tailSplitSize;
        softmaxTilingData.tailReduceSize = tilingData->softmaxTilingData.tailReduceSize;

        coreNum = tilingData->coreNum;
        cubeCoreNum = coreNum / 2;
        vecBlockNum = coreNum / 3;

        b = tilingData->batch;
        n2 = tilingData->kvHeadNum;
        g = tilingData->g;
        s1 = tilingData->qSeqlen;
        s2 = tilingData->kvSeqlen;
        d = tilingData->qkHeadDim;
        value_d = tilingData->vHeadDim;
        t1 = tilingData->t1;
        dAlign = (d + 15) / 16 * 16;
        value_dAlign = (value_d + 15) / 16 * 16;

        attenMaskDimS2 = 2048;

        s1Token = tilingData->s1Token;
        s2Token = tilingData->s2Token;
        actualCalcS1Token = s1Token;
        actualCalcS2Token = s2Token;

        s1Outer = tilingData->s1Outer;
        s1CvInner = tilingData->s1CvInner;
        s1CvTail = tilingData->s1CvTail;
        s2Outer = tilingData->s2Outer;
        s2CvInner = tilingData->s2CvInner;
        s2CvTail = s2 - (s2Outer - 1) * s2CvInner;

        baseMN = tilingData->baseMN;
        cubeBaseMN = s1CvInner * s2CvInner;

        actual_seq_qlen_addr = actual_seq_qlen;
        actual_seq_kvlen_addr = actual_seq_kvlen;
        dropBitMode = tilingData->dropoutIsDivisibleBy8 != 0;
        if constexpr (IS_DROP == ENABLE) {
            __gm__ uint8_t *dropMaskAddr = dropBitMode ? drop_mask : workspace + tilingData->dropBeginAddr;
            dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)dropMaskAddr);
        }
        keepProb = tilingData->keepProb;
        scaleValue = tilingData->scaleValue;
        softcapValue = tilingData->softcapValue;
        compressMode = tilingData->attenMaskCompressMode;

        int64_t sfmgOutputSize = b * n2 * g * s1 * 8;
        if constexpr (INPUT_LAYOUT == TND) {
            int64_t seqS2Len = 0;
            seqS2Len = ((__gm__ int32_t *)actual_seq_kvlen)[0];
            dropBitMode = (seqS2Len % 8 == 0);
            for (int64_t i = 0; i + 1 < b; i++) {
                seqS2Len = ((__gm__ int32_t *)actual_seq_kvlen)[i + 1] - ((__gm__ int32_t *)actual_seq_kvlen)[i];
                dropBitMode = (dropBitMode && (seqS2Len % 8 == 0));
            }
            sfmgOutputSize = ((__gm__ int32_t*)actual_seq_qlen)[b - 1] * n2 * g * 8;
        }

        int64_t dqWorkSpaceOffset = tilingData->dqWorkSpaceOffset;
        int64_t dkWorkSpaceOffset = tilingData->dkWorkSpaceOffset;
        int64_t dvWorkSpaceOffset = tilingData->dvWorkSpaceOffset;
        int64_t sfmgPreBeginAddr = tilingData->sfmgPreBeginAddr;

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + dqWorkSpaceOffset / sizeof(float));
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + dkWorkSpaceOffset / sizeof(float));
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + dvWorkSpaceOffset / sizeof(float));

        sfmgWorkspaceGm.SetGlobalBuffer((__gm__ T2 *)workspace + sfmgPreBeginAddr / sizeof(T2));
        int64_t workspaceOffsets =
            (sfmgPreBeginAddr + sfmgOutputSize * sizeof(float) + ADDR_ALIGN_SIZE) /
            ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;

        uint32_t matmulWorkspaceSize = cubeBaseMN * sizeof(float);
        mm1WorkspaceGm.SetGlobalBuffer((__gm__ T2 *)(workspace + workspaceOffsets +
                                                     cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));
        mm2WorkspaceGm.SetGlobalBuffer(
            (__gm__ T2 *)(workspace + workspaceOffsets + cubeCoreNum * matmulWorkspaceSize * GM_DOUBLE_BUFFER +
                          cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

        dropWorkSpaceGm.SetGlobalBuffer(
            (__gm__ T1 *)(workspace + workspaceOffsets + cubeCoreNum * matmulWorkspaceSize * GM_DOUBLE_BUFFER +
                          cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

        mulWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)(workspace + workspaceOffsets +
                                                     cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

        pipe_in->InitBuffer(buf, TOTAL_SIZE);
        unifiedBuffer = buf;
        AscendC::SyncAll();
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void GetSeqQlenKvlenByBidx(int64_t bIdx, int32_t &actualSeqQlen, int32_t &actualSeqKvlen)
    {
        if (unlikely(bIdx == 0)) {
            actualSeqQlen = ((__gm__ int32_t *)actual_seq_qlen_addr)[0];
            actualSeqKvlen = ((__gm__ int32_t *)actual_seq_kvlen_addr)[0];
        } else {
            actualSeqQlen =
                ((__gm__ int32_t *)actual_seq_qlen_addr)[bIdx] - ((__gm__ int32_t *)actual_seq_qlen_addr)[bIdx - 1];
            actualSeqKvlen =
                ((__gm__ int32_t *)actual_seq_kvlen_addr)[bIdx] - ((__gm__ int32_t *)actual_seq_kvlen_addr)[bIdx - 1];
        }
        return;
    }

    CATLASS_DEVICE
    void UpdateToken(int64_t bIdx)
    {
        if constexpr (IS_ATTEN_MASK != ENABLE) {
            return;
        }
        int32_t actualS1Len = 0;
        int32_t actualS2Len = 0;
        GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
        actualCalcS1Token = s1Token + actualS1Len - actualS2Len;
        actualCalcS2Token = s2Token - actualS1Len + actualS2Len;
    }

    CATLASS_DEVICE
    void CopyInAttenMaskBool(LocalTensor<uint8_t> &dstTensor, int64_t attenMaskOffset,
                                             uint32_t s1Extend, uint32_t s2Extend)
    {
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = s1Extend;
        intriParams.blockLen = s2Extend * sizeof(uint8_t);
        intriParams.srcStride = (attenMaskDimS2 - s2Extend) * sizeof(uint8_t);
        intriParams.dstStride = 0;
        intriParams.rsv = 0;
        AscendC::DataCopyPad(dstTensor, attenMaskU8Gm[attenMaskOffset], intriParams, {false, 0, 0, 0});
    }

    CATLASS_DEVICE
    void CalcAttenMaskBool(LocalTensor<T2> &dstTensor, LocalTensor<uint8_t> srcTensor,
                                             uint32_t s1Extend, uint32_t s2Extend, uint8_t maskType = 0)
    {
        LocalTensor<uint8_t> tmpUbBuffer = unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);

        T2 scalar;
        if constexpr (AscendC::IsSameType<T2, float>::value) {
            uint32_t tmp = 0xFF7FFFFF;
            scalar = *((float *)&tmp);
        } else {
            uint16_t tmp = 0xFBFF;
            scalar = *((half *)&tmp);
        }

        AscendC::SelectWithBytesMaskShapeInfo info;
        info.firstAxis = s1Extend;
        info.srcLastAxis = s2Extend;
        info.maskLastAxis = (s2Extend * sizeof(uint8_t) + 31) / 32 * 32 / sizeof(uint8_t);
        dstTensor.SetSize(info.firstAxis * info.srcLastAxis);
        srcTensor.SetSize(info.firstAxis * info.maskLastAxis);
        if (maskType == 0) {
            AscendC::SelectWithBytesMask(dstTensor, dstTensor, scalar, srcTensor, tmpUbBuffer, info);
        } else {
            AscendC::SelectWithBytesMask(dstTensor, scalar, dstTensor, srcTensor, tmpUbBuffer, info);
        }
    }

    CATLASS_DEVICE
    void CalcAttenMaskOffset(int64_t &attenMaskOffset, const int64_t delta,
                                                         uint32_t s1VSize, uint32_t s2VSize)
    {
        if (delta == 0) {
            attenMaskOffset = 0;
        } else if (delta < 0) {
            if (-delta > s1VSize) {
                attenMaskOffset = s1VSize;
            } else {
                attenMaskOffset = -delta;
            }
        } else {
            if (delta > s2VSize) {
                attenMaskOffset = s2VSize * attenMaskDimS2;
            } else {
                attenMaskOffset = delta * attenMaskDimS2;
            }
        }
    }

    CATLASS_DEVICE
    int64_t GetCausalDelta(int64_t causal_delta, DBParams &dbParam)
    {
        if constexpr (INPUT_LAYOUT == TND) {
            int32_t actualS1Len = 0;
            int32_t actualS2Len = 0;
            GetSeqQlenKvlenByBidx(dbParam.bIdx, actualS1Len, actualS2Len);
            return causal_delta - actualS1Len + actualS2Len;
        } else {
            return causal_delta - s1 + s2;
        }
    }

    CATLASS_DEVICE
    void CalcAttenBandMode(int64_t causal_delta, DBParams &dbParam)
    {
        AttenBandMode = AttenMaskCompress::All;
        if (compressMode != CAUSAL_COMPRESS_MODE && compressMode != BAND_COMPRESS_MODE) {
            return;
        }

        int64_t next_delta = causal_delta;
        int64_t pre_delta = causal_delta - INT32_MAX - 1;
        if (compressMode == CAUSAL_COMPRESS_MODE) {
            next_delta = GetCausalDelta(causal_delta, dbParam);
        } else if (compressMode == BAND_COMPRESS_MODE) {
            next_delta = causal_delta + actualCalcS2Token;
            pre_delta = causal_delta - actualCalcS1Token - 1;
        }

        bool NoNext = (next_delta - s2Extend >= 0);
        bool NoPre = (pre_delta + 1 + s1ExtendSubGraph <= 0);

        if (NoNext && NoPre) {
            AttenBandMode = AttenMaskCompress::Empty;
        } else if (NoNext && !NoPre) {
            AttenBandMode = AttenMaskCompress::PreOnly;
        } else if (!NoNext && NoPre) {
            AttenBandMode = AttenMaskCompress::NextOnly;
        } else {
            AttenBandMode = AttenMaskCompress::All;
        }
    }

    __aicore__ inline void CalcAttenMaskOffsetWithCompressModeForUnpad(
        int64_t &attenMaskOffset, int64_t &attenMaskOffset2,
        uint32_t s1VSize, uint32_t s2VSize, int64_t curS1Idx,
        uint32_t s2VBegin, bool &canSimplify, DBParams &dbParam)
    {
        int64_t causal_delta =
            static_cast<int64_t>(dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) - static_cast<int64_t>(s2VBegin);
        CalcAttenBandMode(causal_delta, dbParam);
        if (compressMode == CAUSAL_COMPRESS_MODE) {
            causal_delta = GetCausalDelta(causal_delta, dbParam);
            CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
            return;
        }

        if (compressMode == BAND_COMPRESS_MODE) { // band
            int64_t next_delta = causal_delta + actualCalcS2Token;
            CalcAttenMaskOffset(attenMaskOffset, next_delta, s1VSize, s2VSize);
            int64_t pre_delta = causal_delta - actualCalcS1Token - 1;
            CalcAttenMaskOffset(attenMaskOffset2, pre_delta, s1VSize, s2VSize);
            return;
        }

        attenMaskDimS2 = (uint32_t)s2;
        attenMaskOffset += (static_cast<int64_t>(dbParam.s1oIdx) * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
    }

    CATLASS_DEVICE
    void CalcAttenMaskOffsetWithCompressMode(int64_t &attenMaskOffset,
        int64_t &attenMaskOffset2, uint32_t s1VSize, uint32_t s2VSize, int64_t curS1Idx, uint32_t s2VBegin,
        bool &canSimplify, DBParams &dbParam)
    {
        int64_t causal_delta =
            static_cast<int64_t>(dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) - static_cast<int64_t>(s2VBegin);
        CalcAttenBandMode(causal_delta, dbParam);
        if (compressMode == CAUSAL_COMPRESS_MODE) {
            causal_delta = GetCausalDelta(causal_delta, dbParam);
            CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
            return;
        }

        if (compressMode == BAND_COMPRESS_MODE) {
            int64_t pre_delta = causal_delta - actualCalcS1Token - 1;
            CalcAttenMaskOffset(attenMaskOffset2, pre_delta, s1VSize, s2VSize);
            int64_t next_delta = causal_delta + actualCalcS2Token;
            CalcAttenMaskOffset(attenMaskOffset, next_delta, s1VSize, s2VSize);
            return;
        }

        attenMaskOffset = (static_cast<int64_t>(dbParam.s1oIdx) * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
    }

    CATLASS_DEVICE
    void CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend, uint32_t softMaxOffset)
    {
        AscendC::DataCopyPad(dstTensor, softmaxLseGm[softMaxOffset],
            {1, static_cast<uint16_t>(s1Extend * 4), 0, 0}, {false, 0, 0, 0});

        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
        AscendC::Brcb(dstTensor[s1Extend * 8], dstTensor, static_cast<uint8_t>((s1Extend + 7) / 8), {1, 8});
    }

    CATLASS_DEVICE
    void CalcSoftMax(LocalTensor<float>& dstTensor, LocalTensor<float>& src0Tensor, LocalTensor<float>& src1Tensor,
                     uint32_t s1Extend, uint32_t s2Extend, uint32_t s2ExtendAlign, const SoftMaxTiling& tiling)
    {
        uint32_t sub_block_count = (s2Extend + cal_repeat_num - 1) / cal_repeat_num;

        for(uint32_t subIdx = 0; subIdx < sub_block_count; subIdx++) {
            uint32_t subMaskCount = (subIdx == sub_block_count - 1) ? (s2Extend - subIdx * cal_repeat_num) : cal_repeat_num;
            AscendC::Sub(dstTensor[subIdx * cal_repeat_num], src0Tensor[subIdx * cal_repeat_num], src1Tensor[s1Extend * 8],
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

    CATLASS_DEVICE
    void DropOutCopy(LocalTensor<uint8_t> &vecInDropBuffer, int64_t curS1Idx, int64_t s2VBegin)
    {
    }

        CATLASS_DEVICE
    int64_t GetDropMaskOffset(const DBParams &dbParam, int64_t curS1Idx, uint32_t s2VBegin, uint32_t dropMaskS2Stride)
    {
        int64_t qHeadNumLocal = n2 * g;
        int64_t headIdx = dbParam.n2Idx * g + dbParam.gIdx;
        int64_t s1Pos = dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize;
        if constexpr (INPUT_LAYOUT == TND) {
            int64_t batchS1S2Prefix = 0;
            for (int64_t bi = 0; bi < dbParam.bIdx; ++bi) {
                int32_t as1 = (bi == 0) ? ((__gm__ int32_t *)actual_seq_qlen_addr)[0] :
                    ((__gm__ int32_t *)actual_seq_qlen_addr)[bi] -
                    ((__gm__ int32_t *)actual_seq_qlen_addr)[bi - 1];
                int32_t as2 = (bi == 0) ? ((__gm__ int32_t *)actual_seq_kvlen_addr)[0] :
                    ((__gm__ int32_t *)actual_seq_kvlen_addr)[bi] -
                    ((__gm__ int32_t *)actual_seq_kvlen_addr)[bi - 1];
                batchS1S2Prefix += as1 * as2;
            }
            return batchS1S2Prefix * qHeadNumLocal +
                headIdx * dbParam.actualS1Len * static_cast<int64_t>(dropMaskS2Stride) +
                s1Pos * static_cast<int64_t>(dropMaskS2Stride) + static_cast<int64_t>(s2VBegin);
        }
        return ((dbParam.bIdx * qHeadNumLocal + headIdx) * s1 + s1Pos) *
            static_cast<int64_t>(dropMaskS2Stride) + static_cast<int64_t>(s2VBegin);
    }

    CATLASS_DEVICE
    void SubGrapA(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx, DBParams& dbParam,
                                event_t mte2WaitMte3A)
    {
        pingpongIdx = dbParam.taskId % 2;
        s2Extend = (curS2Idx == s2VecLoop - 1) ? (dbParam.s2CvExtend - (s2VecLoop - 1) * s2VecSize) : s2VecSize;
        s2ExtendAlign = (s2Extend + 15) / 16 * 16;
        uint32_t s2VBegin = dbParam.s2oIdx * s2CvInner + curS2Idx * s2VecSize;

        uint32_t ubBufferOffset = 0;
        uint32_t ubTmpBufferOffset = 0;

        if (curIdx > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
        }

        LocalTensor<float> vecInBuffer3 =
            unifiedBuffer.GetWithOffset<float>(8 * 1024 / sizeof(float), ubBufferOffset + T2BlockBegin);

        int64_t softMaxOffset = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            if (dbParam.bIdx > 0) {
                softMaxOffset = ((__gm__ int32_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1];
            }
            softMaxOffset += (dbParam.n2Idx * g + dbParam.gIdx) * t1 + dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize;
        } else {
            softMaxOffset = ((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1 + dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize; // bns
        }
        CopyInSoftMax(vecInBuffer3, s1ExtendSubGraph, softMaxOffset);

        LocalTensor<uint8_t> attenMaskUbuint8 =
            unifiedBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + BoolBegin);
        int64_t attenMaskOffsetPre = 0;
        bool prefixCompressCanSimplify = false;
        if constexpr (IS_ATTEN_MASK == ENABLE) {
            int64_t attenMaskOffset = 0;
            if constexpr(INPUT_LAYOUT == TND) {
                UpdateToken(dbParam.bIdx);
                CalcAttenMaskOffsetWithCompressModeForUnpad(attenMaskOffset, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend,
                                                        curS1Idx, s2VBegin, prefixCompressCanSimplify, dbParam);
            } else {
                CalcAttenMaskOffsetWithCompressMode(attenMaskOffset, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend, curS1Idx,
                                                s2VBegin, prefixCompressCanSimplify, dbParam);
            }
            // uint8_t
            if (AttenBandMode == AttenMaskCompress::All || AttenBandMode == AttenMaskCompress::NextOnly) {
                CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffset, s1ExtendSubGraph, s2Extend);
            } else if (AttenBandMode == AttenMaskCompress::PreOnly) {
                CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend);
            }
        }

        LocalTensor<uint8_t> vecInDropBuffer =
            unifiedBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + U8Begin);

        LocalTensor<float> vecClc2Buffer =
            unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), ubBufferOffset + T2Begin);

        if (s2VecLoop == 1) {
            AscendC::DataCopy(vecClc2Buffer, mm2WorkspaceGm[pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * s2ExtendAlign],
                    s1ExtendSubGraph * s2ExtendAlign);
        } else {
            AscendC::DataCopyPad(vecClc2Buffer, mm2WorkspaceGm[pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * dbParam.s2CvExtendAlign + curS2Idx * s2VecSize],
                        {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                            static_cast<uint16_t>((dbParam.s2CvExtendAlign - s2ExtendAlign) * sizeof(float)), 0},
                        {false, 0, 0, 0});
        }
        event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(vecClc2Buffer, vecClc2Buffer, (T2)scaleValue, s1ExtendSubGraph * s2ExtendAlign);
        AscendC::PipeBarrier<PIPE_V>();

        // recompute softcap
        if constexpr (HAS_SOFTCAP) {
            AscendC::Maxs(vecClc2Buffer, vecClc2Buffer, -8.8f, s1ExtendSubGraph * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Muls(vecClc2Buffer, vecClc2Buffer, -2.0f, s1ExtendSubGraph * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(vecClc2Buffer, vecClc2Buffer, s1ExtendSubGraph * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(vecClc2Buffer, vecClc2Buffer, 1.0f, s1ExtendSubGraph * s2ExtendAlign);
            // temp buffer for softcap
            AscendC::LocalTensor<float> softcapBuffer = unifiedBuffer.GetWithOffset<float>(
                cal_repeat_num, SOFTCAP_UB_OFFSET);
            AscendC::Duplicate<float, false>(softcapBuffer, 2 * softcapValue, (uint64_t)0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div<float, false>(vecClc2Buffer, softcapBuffer, vecClc2Buffer, (uint64_t)0, 
                    (s1ExtendSubGraph*s2ExtendAlign)/64, {1, 1, 1, 8, 0, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(vecClc2Buffer, vecClc2Buffer, -softcapValue, s1ExtendSubGraph * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
        }
        ///////////////////////////////////////////////////////////////
        // attenMask
        ///////////////////////////////////////////////////////////////
        if constexpr (IS_ATTEN_MASK == ENABLE) {
            // uint8_t
            if (AttenBandMode == AttenMaskCompress::All || AttenBandMode == AttenMaskCompress::NextOnly) {
                CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign);
            } else if (AttenBandMode == AttenMaskCompress::PreOnly) {
                CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign, 1);
            }

            if (compressMode == BAND_COMPRESS_MODE && AttenBandMode == AttenMaskCompress::All) {
                event_t mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                AscendC::SetFlag<HardEvent::V_MTE2>(static_cast<int32_t>(mte2WaitV));
                AscendC::WaitFlag<HardEvent::V_MTE2>(static_cast<int32_t>(mte2WaitV));
                CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend);
                event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                AscendC::SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
                AscendC::WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
                CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign, 1);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<float>(33 * 1024 / sizeof(T2), DbBegin);
        CalcSoftMax(simpleSoftmaxResBuf, vecClc2Buffer, vecInBuffer3, s1ExtendSubGraph, s2Extend, s2ExtendAlign, softmaxTilingData);
        LocalTensor<T2> vecDropBuffer = simpleSoftmaxResBuf;

        if constexpr (IS_DROP == ENABLE) {
            uint32_t totalElems = s1ExtendSubGraph * s2ExtendAlign;
            uint32_t maskRowStride = (s2ExtendAlign + 31) / 32 * 32;
            uint32_t dropMaskS2Stride;
            if constexpr (INPUT_LAYOUT == TND) {
                dropMaskS2Stride = static_cast<uint32_t>(dbParam.actualS2Len);
            } else {
                dropMaskS2Stride = dropBitMode ? (static_cast<uint32_t>(s2) + 31) / 32 * 32 :
                    static_cast<uint32_t>(s2);
            }
            int64_t dropMaskOffset = GetDropMaskOffset(dbParam, curS1Idx, s2VBegin, dropMaskS2Stride);
            LocalTensor<uint8_t> dropMaskU8 =
                unifiedBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + U8Begin);
            LocalTensor<uint16_t> dropMaskU16 = dropMaskU8.template ReinterpretCast<uint16_t>();
            AscendC::Duplicate<uint16_t>(dropMaskU16, static_cast<uint16_t>(0),
                                         s1ExtendSubGraph * maskRowStride / 2);
            event_t v2Mte2A = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(v2Mte2A);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(v2Mte2A);
            AscendC::DataCopyExtParams dropCopyParams;
            dropCopyParams.blockCount = static_cast<uint16_t>(s1ExtendSubGraph);
            uint32_t dropReadLen = s2Extend;
            if constexpr (INPUT_LAYOUT == TND) {
                if (s2Extend > dropMaskS2Stride) {
                    dropReadLen = dropMaskS2Stride;
                }
            }
            dropCopyParams.blockLen = static_cast<uint32_t>(dropReadLen * sizeof(uint8_t));
            dropCopyParams.srcStride = static_cast<uint32_t>((dropMaskS2Stride - dropReadLen) * sizeof(uint8_t));
            dropCopyParams.dstStride = 0;
            dropCopyParams.rsv = 0;
            AscendC::DataCopyPad(dropMaskU8, dropMaskGm[dropMaskOffset], dropCopyParams,
                                 {false, 0, 0, 0});
            event_t dropMte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(dropMte2WaitV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(dropMte2WaitV);
            vecDropBuffer = unifiedBuffer.GetWithOffset<T2>(TMP_UB_SIZE / sizeof(T2), TMP_UB_OFFSET);
            AscendC::Muls(vecDropBuffer, simpleSoftmaxResBuf,
                          static_cast<float>(1.0f / keepProb), totalElems);
            AscendC::PipeBarrier<PIPE_V>();
            LocalTensor<uint8_t> selScratch = unifiedBuffer.GetWithOffset<uint8_t>(
                17 * 1024 / sizeof(uint8_t), ubBufferOffset + T1Begin);
            AscendC::SelectWithBytesMaskShapeInfo selInfo;
            selInfo.firstAxis = s1ExtendSubGraph;
            selInfo.srcLastAxis = s2ExtendAlign;
            selInfo.maskLastAxis = maskRowStride;
            vecDropBuffer.SetSize(s1ExtendSubGraph * s2ExtendAlign);
            dropMaskU8.SetSize(s1ExtendSubGraph * maskRowStride);
            AscendC::SelectWithBytesMask(vecDropBuffer,
                                         static_cast<float>(0.0f), vecDropBuffer,
                                         dropMaskU8, selScratch, selInfo);
            AscendC::PipeBarrier<PIPE_V>();
        }
        LocalTensor<T1> vecCopyOutBuffer = vecDropBuffer.template ReinterpretCast<T1>();
        if constexpr (!AscendC::IsSameType<T1, float>::value) {
            vecCopyOutBuffer = unifiedBuffer.GetWithOffset<T1>(17 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(vecCopyOutBuffer, vecDropBuffer, RoundMode::CAST_ROUND, s1ExtendSubGraph * s2ExtendAlign);
        }
        int64_t copyOutOffset = 0;
        DataCopyParams copyOutParam;
        copyOutOffset = pingpongIdx * cubeBaseMN * DTYPE_FACTOR +
                        curS1Idx * s1VecSize * dbParam.s2CvExtendAlign * DTYPE_FACTOR + curS2Idx * s2VecSize;
        copyOutParam = {
            static_cast<uint16_t>(s1ExtendSubGraph),
            static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)),
            0,
            static_cast<uint16_t>((dbParam.s2CvExtendAlign * DTYPE_FACTOR - s2ExtendAlign) * sizeof(T1))
        };
        event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        AscendC::DataCopyPad(dropWorkSpaceGm[copyOutOffset], vecCopyOutBuffer, copyOutParam);

        if (curIdx < vecLoopEnd - vecLoopStart - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
        }
    }

    CATLASS_DEVICE
    void SubGrapB(int64_t curIdx, int64_t s1VecLoop, int64_t s2VecLoop,
                                 int64_t curS1Idx, int64_t curS2Idx, DBParams& dbParam, event_t mte2WaitMte3B)
    {
        pingpongIdx = dbParam.taskId % 2;
        uint32_t ubBufferOffset = DbBegin;
        s2Extend = (curS2Idx == s2VecLoop -1) ? (dbParam.s2CvExtend - (s2VecLoop - 1) * s2VecSize) : s2VecSize;
        s2ExtendAlign = (s2Extend + 15) / 16 * 16;

        if (curIdx > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }

        // Save S values before SubGrapA(i+1) MTE overwrites T2Begin
        if constexpr (HAS_SOFTCAP) {
            AscendC::LocalTensor<float> savedS = unifiedBuffer.GetWithOffset<float>(
                32 * 1024 / sizeof(float), TMP_UB_OFFSET);
            AscendC::LocalTensor<float> srcS = unifiedBuffer.GetWithOffset<float>(
                32 * 1024 / sizeof(float), T2Begin);
            AscendC::DataCopy(savedS, srcS, s1ExtendSubGraph * s2ExtendAlign);
        }

        if (preS1Idx != curS1Idx) {
            preS1Idx = curS1Idx;
            LocalTensor<float> sfmgClc3 = unifiedBuffer.GetWithOffset<float>(SFMG_UB_SIZE / sizeof(float), SFMG_UB_OFFSET);
            AscendC::DataCopy(sfmgClc3, sfmgWorkspaceGm[sfmgOffset + curS1Idx * s1VecSize * 8], s1ExtendSubGraph * 8);
        }

        LocalTensor<uint8_t> vecInDropBuffer =
            unifiedBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + U8Begin);

        LocalTensor<T2> vecClc1Buffer = unifiedBuffer.GetWithOffset<T2>(33 * 1024 / sizeof(T2), ubBufferOffset + T1Begin);
        LocalTensor<T2> dyvBuffer = unifiedBuffer.GetWithOffset<T2>(33 * 1024 / sizeof(T2), TMP_UB_OFFSET);
        uint32_t maskRowStrideB = (s2ExtendAlign + 31) / 32 * 32;
        if constexpr (IS_DROP == ENABLE) {
            LocalTensor<uint16_t> vecInDropBufferU16 = vecInDropBuffer.template ReinterpretCast<uint16_t>();
            AscendC::Duplicate<uint16_t>(vecInDropBufferU16, static_cast<uint16_t>(0),
                                         s1ExtendSubGraph * maskRowStrideB / 2);
            AscendC::PipeBarrier<PIPE_V>();
            int64_t s2VBegin = dbParam.s2oIdx * s2CvInner + curS2Idx * s2VecSize;
            uint32_t dropMaskS2StrideB;
            if constexpr (INPUT_LAYOUT == TND) {
                dropMaskS2StrideB = static_cast<uint32_t>(dbParam.actualS2Len);
            } else {
                dropMaskS2StrideB = dropBitMode ? (static_cast<uint32_t>(s2) + 31) / 32 * 32 :
                    static_cast<uint32_t>(s2);
            }
            int64_t dropMaskOffsetB = GetDropMaskOffset(dbParam, curS1Idx, s2VBegin, dropMaskS2StrideB);
            event_t v2Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(v2Mte2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(v2Mte2);
            AscendC::DataCopyExtParams dropCopyParams;
            dropCopyParams.blockCount = static_cast<uint16_t>(s1ExtendSubGraph);
            uint32_t dropReadLenB = s2Extend;
            if constexpr (INPUT_LAYOUT == TND) {
                if (s2Extend > dropMaskS2StrideB) {
                    dropReadLenB = dropMaskS2StrideB;
                }
            }
            dropCopyParams.blockLen = static_cast<uint32_t>(dropReadLenB * sizeof(uint8_t));
            dropCopyParams.srcStride = static_cast<uint32_t>((dropMaskS2StrideB - dropReadLenB) * sizeof(uint8_t));
            dropCopyParams.dstStride = 0;
            dropCopyParams.rsv = 0;
            AscendC::DataCopyPad(vecInDropBuffer, dropMaskGm[dropMaskOffsetB], dropCopyParams,
                                 {false, 0, 0, 0});
        }
        if (s2VecLoop == 1) {
            AscendC::DataCopy(vecClc1Buffer, mm1WorkspaceGm[pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * s2ExtendAlign],
                        s1ExtendSubGraph * s2ExtendAlign);
        } else {
            AscendC::DataCopyPad(vecClc1Buffer, mm1WorkspaceGm[pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * dbParam.s2CvExtendAlign + curS2Idx * s2VecSize],
                        {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                            static_cast<uint16_t>((dbParam.s2CvExtendAlign - s2ExtendAlign) * sizeof(float)), 0},
                        {false, 0, 0, 0});
        }

        event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);

        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (IS_DROP == ENABLE) {
            uint32_t totalElems = s1ExtendSubGraph * s2ExtendAlign;
            AscendC::Muls(vecClc1Buffer, vecClc1Buffer, static_cast<float>(1.0f / keepProb), totalElems);
            AscendC::PipeBarrier<PIPE_V>();
            LocalTensor<uint8_t> selScratchB = unifiedBuffer.GetWithOffset<uint8_t>(
                TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);
            AscendC::SelectWithBytesMaskShapeInfo selInfoB;
            selInfoB.firstAxis = s1ExtendSubGraph;
            selInfoB.srcLastAxis = s2ExtendAlign;
            selInfoB.maskLastAxis = maskRowStrideB;
            vecClc1Buffer.SetSize(s1ExtendSubGraph * s2ExtendAlign);
            vecInDropBuffer.SetSize(s1ExtendSubGraph * maskRowStrideB);
            AscendC::SelectWithBytesMask(vecClc1Buffer,
                                         static_cast<float>(0.0f), vecClc1Buffer,
                                         vecInDropBuffer, selScratchB, selInfoB);
            AscendC::PipeBarrier<PIPE_V>();
        }
        uint32_t sub_block_cout = (s2ExtendAlign + cal_repeat_num - 1) / cal_repeat_num;
        LocalTensor<float> sfmgClc3 = unifiedBuffer.GetWithOffset<float>(SFMG_UB_SIZE / sizeof(float), SFMG_UB_OFFSET);

        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t subIdx = 0; subIdx < sub_block_cout; subIdx++) {
            uint32_t subMaskCout =
                (subIdx == sub_block_cout - 1) ? (s2ExtendAlign - subIdx * cal_repeat_num) : cal_repeat_num;
            AscendC::Sub(vecClc1Buffer[subIdx * cal_repeat_num], vecClc1Buffer[subIdx * cal_repeat_num], sfmgClc3,
                subMaskCout, s1ExtendSubGraph,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8), 1});
        }

        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), DbBegin);

        AscendC::Mul(vecClc1Buffer, vecClc1Buffer, simpleSoftmaxResBuf, s1ExtendSubGraph * s2ExtendAlign);
        AscendC::PipeBarrier<PIPE_V>();

        if constexpr (HAS_SOFTCAP) {
            // Use saved copy from TMP_UB_OFFSET (avoids race with SubGrapA MTE)
            AscendC::LocalTensor<float> vecClc2Buffer =
            unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), TMP_UB_OFFSET);
            // avoid mask -INF
            AscendC::Maxs(vecClc2Buffer, vecClc2Buffer, -softcapValue, s1ExtendSubGraph * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(vecClc2Buffer, vecClc2Buffer, vecClc2Buffer, s1ExtendSubGraph * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            // softcap * (1 - (S/softcap)^2) = softcap * sech^2(x)
            AscendC::Muls(vecClc2Buffer, vecClc2Buffer, -(1.0f/softcapValue), s1ExtendSubGraph * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(vecClc2Buffer, vecClc2Buffer, softcapValue, s1ExtendSubGraph * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(vecClc1Buffer, vecClc1Buffer, vecClc2Buffer, s1ExtendSubGraph * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
        }

        LocalTensor<T1> vecCopyOutBuffer = vecClc1Buffer.template ReinterpretCast<T1>();
        if constexpr (!AscendC::IsSameType<T1, float>::value) {
            vecCopyOutBuffer = unifiedBuffer.GetWithOffset<T1>(17 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
            AscendC::Cast(vecCopyOutBuffer, vecClc1Buffer, RoundMode::CAST_ROUND, s1ExtendSubGraph * s2ExtendAlign);
        }

        int64_t copyOutOffset = 0;
        DataCopyParams copyOutParam;
        copyOutOffset = pingpongIdx * cubeBaseMN * DTYPE_FACTOR +
                        curS1Idx * s1VecSize * dbParam.s2CvExtendAlign * DTYPE_FACTOR + curS2Idx * s2VecSize;
        copyOutParam = {
            static_cast<uint16_t>(s1ExtendSubGraph),
            static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)),
            0,
            static_cast<uint16_t>((dbParam.s2CvExtendAlign * DTYPE_FACTOR - s2ExtendAlign) * sizeof(T1))
        };
        event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

        AscendC::DataCopyPad(mulWorkSpaceGm[copyOutOffset], vecCopyOutBuffer, copyOutParam);

        if (curIdx < vecLoopEnd - vecLoopStart - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }
    }

    CATLASS_DEVICE
    void operator()(DBParams& dbParam)
    {
        int64_t actualS1Len;
        int64_t actualS2Len;

        s2VecSize = dbParam.s2CvExtend > VEC_S2_LEN ? VEC_S2_LEN : dbParam.s2CvExtend;
        s2VecLoop = s2VecSize == 0 ? 0 : CeilDiv(dbParam.s2CvExtend, s2VecSize);

        uint32_t s2AlignFactor = BLOCK_SIZE / 2;
        if constexpr (IS_DROP == ENABLE || IS_ATTEN_MASK == ENABLE) {
            // last dim 32B align
            s2AlignFactor = BLOCK_SIZE / sizeof(uint8_t);
        }

        if (s2AlignFactor == 0) {
            return;
        } else {
            s1VecSize = baseMN / ((s2VecSize + s2AlignFactor - 1) / s2AlignFactor * s2AlignFactor);
        }
        if constexpr (IS_DROP == ENABLE) {
            uint32_t s2VecSizeAlign = (s2VecSize + 15) / 16 * 16;
            uint32_t maskRowStride = (s2VecSizeAlign + 31) / 32 * 32;
            uint32_t maxS1ByMaskUb = maskRowStride == 0 ? s1VecSize : (8 * 1024) / maskRowStride;
            s1VecSize = s1VecSize > maxS1ByMaskUb ? maxS1ByMaskUb : s1VecSize;
        }
        s1VecSize = s1VecSize > dbParam.s1CvExtend ? dbParam.s1CvExtend : s1VecSize;
        s1VecSize = s1VecSize > 128 ? 128 : s1VecSize;
        s1VecLoop = s1VecSize == 0 ? 0 : CeilDiv(dbParam.s1CvExtend, s1VecSize);
        if constexpr (INPUT_LAYOUT == TND) {
            GetSeqQlenKvlenByBidx(dbParam.bIdx, dbParam.actualS1Len, dbParam.actualS2Len);
        }

        sfmgOffset = 0;
        if constexpr(INPUT_LAYOUT == TND) {
            if (dbParam.bIdx > 0) {
                sfmgOffset = n2 * g * ((__gm__ int32_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1] * 8;
            }
            sfmgOffset += ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len + dbParam.s1oIdx * s1CvInner) * 8;
        } else {
            sfmgOffset = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1 + dbParam.s1oIdx * s1CvInner) * 8;
        }

        int32_t loopSize = s1VecLoop * s2VecLoop;
        int32_t halfLoop = 0;

        halfLoop = (s1VecLoop / 2) * s2VecLoop;

        vecLoopStart = cSubIdx ? halfLoop : 0;
        vecLoopEnd = cSubIdx ? loopSize : halfLoop;
        preS1Idx = -1;
        event_t mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
        for (int32_t i = vecLoopStart, loopCnt = 0; i < vecLoopEnd; i++, loopCnt++) {
            int32_t curS1Idx;
            int32_t curS2Idx;
            curS1Idx = i / s2VecLoop;
            curS2Idx = i % s2VecLoop;

            s1ExtendSubGraph = (curS1Idx == s1VecLoop - 1) ? (dbParam.s1CvExtend - (s1VecLoop - 1) * s1VecSize) : s1VecSize;

            event_t mte2WaitMte3A = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
            event_t mte2WaitMte3B = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
            SubGrapA(loopCnt, curS1Idx, curS2Idx, dbParam, mte2WaitMte3A);
            SubGrapB(loopCnt, s1VecLoop, s2VecLoop, curS1Idx, curS2Idx, dbParam, mte2WaitMte3B);

            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }
    }
};

// v2 specialization
template <
    class ElementVecDtype,
    InputLayout inputLayout,
    class TilingData,
    bool HAS_SOFTCAP_>
class BlockEpilogue<
    EpilogueAtlasA2FAGOp<HAS_SOFTCAP_>,
    ElementVecDtype,
    std::integral_constant<InputLayout, inputLayout>,
    TilingData>
{
public:
    using DispatchPolicy = EpilogueAtlasA2FAGOp<HAS_SOFTCAP_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr bool HAS_SOFTCAP = HAS_SOFTCAP_;

    static constexpr InputLayout getLayout()
    {
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
    constexpr static int64_t SOFTCAP_UB_OFFSET = 32 * 1024;
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

    int64_t b;
    int64_t nheads_k;
    int64_t g;
    int64_t cuQSeqLen;
    int64_t cuKSeqLen;
    int64_t headdim;
    int64_t seq_q;
    int64_t seq_k;
    int64_t t1;

    float scaleValue;
    float softcapValue;

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
        pipe = pipe_in;

        blockIdx = GetBlockIdx();
        cubeBlockIdx = blockIdx / 2;
        subIdx = blockIdx % 2;
        curSeqQIdx = subIdx;
        curSeqKIdx = 0;

        cubeBaseMN = 16 * 128 * 128;

        cu_seq_qlen_addr = cu_seq_qlen;
        cu_seq_kvlen_addr = cu_seq_kvlen;

        __gm__ TilingData *tilingData = reinterpret_cast<__gm__ TilingData *>(tiling_in);
        b = tilingData->batch;
        nheads_k = tilingData->kvHeadNum;
        g = tilingData->g;
        headdim = tilingData->qkHeadDim;
        t1 = tilingData->t1;
        seq_q = t1 / b;
        seq_k = tilingData->t2 / b;

        int64_t sfmgWorkSpaceOffset = tilingData->sfmgPreBeginAddr;
        int64_t mm1WorkSpaceOffset = tilingData->mm1WorkSpaceOffset;
        int64_t mm2WorkSpaceOffset = tilingData->mm2WorkSpaceOffset;
        int64_t pWorkSpaceOffset = tilingData->pWorkSpaceOffset;
        int64_t dsWorkSpaceOffset = tilingData->dsWorkSpaceOffset;

        scaleValue = tilingData->scaleValue;
        softcapValue = tilingData->softcapValue;
        softmaxTilingData.srcM = tilingData->softmaxTilingData.srcM;
        softmaxTilingData.srcK = tilingData->softmaxTilingData.srcK;
        softmaxTilingData.srcSize = tilingData->softmaxTilingData.srcSize;
        softmaxTilingData.outMaxM = tilingData->softmaxTilingData.outMaxM;
        softmaxTilingData.outMaxK = tilingData->softmaxTilingData.outMaxK;
        softmaxTilingData.outMaxSize = tilingData->softmaxTilingData.outMaxSize;
        softmaxTilingData.splitM = tilingData->softmaxTilingData.splitM;
        softmaxTilingData.splitK = tilingData->softmaxTilingData.splitK;
        softmaxTilingData.splitSize = tilingData->softmaxTilingData.splitSize;
        softmaxTilingData.reduceM = tilingData->softmaxTilingData.reduceM;
        softmaxTilingData.reduceK = tilingData->softmaxTilingData.reduceK;
        softmaxTilingData.reduceSize = tilingData->softmaxTilingData.reduceSize;
        softmaxTilingData.rangeM = tilingData->softmaxTilingData.rangeM;
        softmaxTilingData.tailM = tilingData->softmaxTilingData.tailM;
        softmaxTilingData.tailSplitSize = tilingData->softmaxTilingData.tailSplitSize;
        softmaxTilingData.tailReduceSize = tilingData->softmaxTilingData.tailReduceSize;

        pipe->InitBuffer(unifiedBuffer, TOTAL_SIZE);
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
    void CopyInAttenMaskBool(LocalTensor<uint8_t> &dstTensor, int64_t attenMaskOffset, uint32_t s1Extend,
        uint32_t s2Extend)
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
        LocalTensor<uint8_t> tmpUbBuffer = unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t),
            TMP_UB_OFFSET);

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
        info.maskLastAxis = s2MaskExtend;
        dstTensor.SetSize(info.firstAxis * info.srcLastAxis);
        srcTensor.SetSize(info.firstAxis * info.maskLastAxis);
        AscendC::SelectWithBytesMask<float, uint8_t, false>(dstTensor, dstTensor, scalar, srcTensor, tmpUbBuffer, info);
    }

    CATLASS_DEVICE
    void CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend, uint32_t softMaxOffset)
    {
        AscendC::DataCopyPad(dstTensor, rowLseGm[softMaxOffset],
            {1, static_cast<uint16_t>(s1Extend * 4), 0, 0}, {false, 0, 0, 0});
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
        AscendC::Brcb(dstTensor[64 * 8], dstTensor, static_cast<uint8_t>((s1Extend + 7) / 8), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(dstTensor, 1.0f, s1Extend * 8);
    }

    CATLASS_DEVICE
    void CalcSoftMax(LocalTensor<float>& dstTensor, LocalTensor<float>& src0Tensor, 
                     LocalTensor<float>& src1Tensor, uint32_t s1Extend, uint32_t s2Extend, uint32_t s2ExtendAlign,
                     const SoftMaxTiling& tiling)
    {
        bool isBasicBlock = (s1Extend % 8 == 0) && (s2Extend % 64 == 0);

        if (isBasicBlock) {
            AscendC::LocalTensor<uint8_t> sharedTmp = unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE /
                sizeof(uint8_t), TMP_UB_OFFSET);
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
                uint32_t subMaskCount = (subIdx == sub_block_count - 1) ? (s2Extend - subIdx *
                    cal_repeat_num) : cal_repeat_num;
                AscendC::Sub(dstTensor[subIdx * cal_repeat_num], src0Tensor[subIdx * cal_repeat_num],
                    src1Tensor[64 * 8],
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

        // recompute softcap
        if constexpr (HAS_SOFTCAP) {
            AscendC::Maxs(vecClc2Buffer, vecClc2Buffer, -8.8f, s1Extend * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Muls(vecClc2Buffer, vecClc2Buffer, -2.0f, s1Extend * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(vecClc2Buffer, vecClc2Buffer, s1Extend * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(vecClc2Buffer, vecClc2Buffer, 1.0f, s1Extend * s2ExtendAlign);

            AscendC::LocalTensor<float> softcapBuffer = unifiedBuffer.GetWithOffset<float>(
                cal_repeat_num, SOFTCAP_UB_OFFSET);
            AscendC::Duplicate<float, false>(softcapBuffer, 2 * softcapValue, (uint64_t)0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div<float, false>(vecClc2Buffer, softcapBuffer, vecClc2Buffer, (uint64_t)0, s1Extend*s2ExtendAlign/64, {1, 1, 1, 8, 0, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(vecClc2Buffer, vecClc2Buffer, -softcapValue, s1Extend * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
        }

        LocalTensor<uint8_t> attenMaskUbuint8 =
            unifiedBuffer.GetWithOffset<uint8_t>(16 * 1024 / sizeof(uint8_t), ubBufferOffset + BoolBegin);
        if (blockInfo.SeqQIdx == blockInfo.SeqKIdx) {
            CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8[curSeqQIdx * s1VecSize * 128], s1Extend, s2ExtendAlign,
                S2_CUBESIZE, 0);
            AscendC::PipeBarrier<PIPE_V>();
        }

        ///////////////////////////////////////////////////////////////
        // simpleSoftMax
        ///////////////////////////////////////////////////////////////
        LocalTensor<float> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<float>(33 * 1024 / sizeof(float), DbBegin);
        CalcSoftMax(simpleSoftmaxResBuf, vecClc2Buffer, vecInBuffer3, s1Extend, s2Extend, s2ExtendAlign,
            softmaxTilingData);
        LocalTensor<float> vecDropBuffer = simpleSoftmaxResBuf;

        ///////////////////////////////////////////////////////////////
        // cast fp322bf16
        ///////////////////////////////////////////////////////////////
        LocalTensor<ElementVecDtype> vecCopyOutBuffer = unifiedBuffer.GetWithOffset<ElementVecDtype>(17 * 1024 /
            sizeof(ElementVecDtype), ubBufferOffset + T1Begin);
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

        // Save S values before SubGrapA(i+1) MTE overwrites T2Begin
        if constexpr (HAS_SOFTCAP) {
            LocalTensor<float> savedS = unifiedBuffer.GetWithOffset<float>(
                32 * 1024 / sizeof(float), TMP_UB_OFFSET);
            LocalTensor<float> srcS = unifiedBuffer.GetWithOffset<float>(
                32 * 1024 / sizeof(float), T2Begin);
            AscendC::DataCopy(savedS, srcS, s1Extend * s2ExtendAlign);
        }

        // copyIn sfmg
        LocalTensor<float> sfmgClc3 = unifiedBuffer.GetWithOffset<float>(SFMG_UB_SIZE / sizeof(float), SFMG_UB_OFFSET);
        DataCopy(sfmgClc3, sfmgWorkspaceGm[sfmgOffset], s1Extend * 8);

        LocalTensor<float> vecClc1Buffer = unifiedBuffer.GetWithOffset<float>(33 * 1024 / sizeof(float),
            ubBufferOffset + T1Begin);
        
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

        if constexpr (HAS_SOFTCAP) {
            // Use saved copy from TMP_UB_OFFSET (avoids race with SubGrapA MTE)
            AscendC::LocalTensor<float> vecClc2Buffer =
            unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), TMP_UB_OFFSET);
            AscendC::Maxs(vecClc2Buffer, vecClc2Buffer, -softcapValue, s1Extend * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(vecClc2Buffer, vecClc2Buffer, vecClc2Buffer, s1Extend * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            // softcap * (1 - (S/softcap)^2) = softcap * sech^2(x)
            AscendC::Muls(vecClc2Buffer, vecClc2Buffer, -(1.0f/softcapValue), s1Extend * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(vecClc2Buffer, vecClc2Buffer, softcapValue, s1Extend * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(vecClc1Buffer, vecClc1Buffer, vecClc2Buffer, s1Extend * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
        }

        LocalTensor<ElementVecDtype> vecCopyOutBuffer = unifiedBuffer.GetWithOffset<ElementVecDtype>(17 * 1024 /
            sizeof(ElementVecDtype), ubBufferOffset + T1Begin);
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

            // offset
            int64_t globalSeqStart = 0;
            int64_t batchOffset = 0;
            if constexpr (getLayout() == InputLayout::TND) {
                globalSeqStart = (blockInfo.batchIdx > 0)
                ? ((__gm__ int32_t *)cu_seq_qlen_addr)[blockInfo.batchIdx - 1]
                : 0;
                batchOffset = t1;
            } else {
                globalSeqStart = blockInfo.batchIdx * nheads_k * g * seq_q;
                batchOffset = seq_q;
            }
            int64_t headIdx = (blockInfo.nheadsKIdx * g + blockInfo.gIdx) * batchOffset;
            int64_t seqOffsetInBlock = blockInfo.SeqQIdx * S1_CUBESIZE + curSeqQIdx * s1VecSize;
            lseOffset = globalSeqStart + headIdx + seqOffsetInBlock;

            sfmgOffset = 0;
            if (blockInfo.batchIdx > 0) {
                if constexpr (getLayout() == InputLayout::TND) {
                    sfmgOffset = ((__gm__ int32_t *)cu_seq_qlen_addr)[blockInfo.batchIdx - 1] * nheads_k * g * 8;
                } else {
                    sfmgOffset = seq_q * nheads_k * g * 8;
                }
            }
            sfmgOffset += ((blockInfo.nheadsKIdx * g + blockInfo.gIdx) * cuQSeqLen + blockInfo.SeqQIdx * S1_CUBESIZE +
                curSeqQIdx * s1VecSize) * 8;
            
            // copyIn cube_workspace params
            copyInOffset = cubeBlockIdx * cubeBaseMN * 2 + pingpongIdx * cubeBaseMN + blockInfo.offset + curSeqQIdx * s1VecSize *
                s2CubeExtend;
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
