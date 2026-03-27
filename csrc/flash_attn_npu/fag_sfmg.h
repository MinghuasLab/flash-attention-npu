/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef FAG_SFMG_H
#define FAG_SFMG_H

constexpr AscendC::RoundMode
FLOAT2HALF_ROUND_MODE = AscendC::RoundMode::CAST_NONE;
using AscendC::B16_BYTE_SIZE;
using AscendC::B32_BYTE_SIZE;
using AscendC::BRCB_BROADCAST_NUMBER;
using AscendC::BrcbRepeatParams;
using AscendC::DEFAULT_BLK_NUM;
using AscendC::DEFAULT_BLK_STRIDE;
using AscendC::DEFAULT_BLOCK_SIZE;
using AscendC::DEFAULT_C0_SIZE;
using AscendC::DEFAULT_REPEAT_STRIDE;
using AscendC::DivCeil;
using AscendC::FLOAT_NUM_PER_BLK;
using AscendC::FLOAT_REPEAT_SIZE;
using AscendC::HALF_FACTOR;
using AscendC::HALF_REPEAT_STRIDE;
using AscendC::HardEvent;
using AscendC::LastAxisShapeND;
using AscendC::LocalTensor;
using AscendC::MASK_PLACEHOLDER;
using AscendC::MaskMode;
using AscendC::MAX_REPEAT_TIMES;
using AscendC::ONE_BLK_SIZE;
using AscendC::ONE_BYTE_BIT_SIZE;
using AscendC::PipeBarrier;
using AscendC::ResetMask;
using AscendC::RoundMode;
using AscendC::SCALAR_STACK_DEPTH;
using AscendC::SetFlag;
using AscendC::SetMaskCount;
using AscendC::SetMaskNorm;
using AscendC::SetVectorMask;
using AscendC::ShapeInfo;
using AscendC::SOFTMAX_BASIC_TILE_NUM;
using AscendC::SOFTMAX_COMPUTE_DIM;
using AscendC::SOFTMAXGRAD_COMPUTE_DIM;
using AscendC::WaitFlag;

struct ReduceLastND {
    uint32_t originalSrcM;
    uint32_t originalSrcK;
    uint32_t srcM;
    uint32_t srcK;
    uint32_t dstM;
    uint32_t dstK;
};

struct SoftMaxShapeInfo {
    uint32_t srcM{0};
    uint32_t srcK{0};
    uint32_t oriSrcM{0};
    uint32_t oriSrcK{0};
};

__aicore__ inline void CustomAlignedReduceSumNDImpl(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                                    const LocalTensor<float> &tmpTensor,
                                                    const struct ReduceLastND &reduceParam, const uint32_t splitCount) {
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * FLOAT_REPEAT_SIZE);
    BlockReduceSum<float, false>(tmpTensor, src, 1, MASK_PLACEHOLDER, 1, 1, reduceParam.srcK / FLOAT_NUM_PER_BLK);
    SetMaskNorm();
    ResetMask();
    PipeBarrier<PIPE_V>();
    DataCopy(dst, tmpTensor, {1, (uint16_t) reduceParam.srcM, 0, 0});
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    for (uint32_t i = 1; i < splitCount; i++) {
        SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * FLOAT_REPEAT_SIZE);
        BlockReduceSum<float, false>(tmpTensor, src[i * FLOAT_REPEAT_SIZE], 1, MASK_PLACEHOLDER, 1, 1,
                                     reduceParam.srcK / FLOAT_NUM_PER_BLK);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * FLOAT_NUM_PER_BLK);
        Add<float, false>(dst, dst, tmpTensor, MASK_PLACEHOLDER, 1,
                          {1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
    }
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * FLOAT_NUM_PER_BLK);
    BlockReduceSum<float, false>(dst, dst, 1, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void CustomReduceSumLastNDSplitImpl(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                                      const struct ReduceLastND &reduceParam, uint64_t mask,
                                                      uint32_t dstRepStride, uint32_t splitNum) {
    uint32_t range = reduceParam.srcM / MAX_REPEAT_TIMES;
    uint32_t tail = reduceParam.srcM % MAX_REPEAT_TIMES;

    for (uint32_t i = 0; i < range; i++) {
        WholeReduceSum(dst[i * MAX_REPEAT_TIMES],
                       src[splitNum * FLOAT_REPEAT_SIZE + i * MAX_REPEAT_TIMES * reduceParam.srcK], mask,
                       MAX_REPEAT_TIMES,
                       dstRepStride, 1,
                       reduceParam.srcK / FLOAT_NUM_PER_BLK);
    }
    if (tail != 0) {
        WholeReduceSum(dst[range * MAX_REPEAT_TIMES],
                       src[splitNum * FLOAT_REPEAT_SIZE + range * MAX_REPEAT_TIMES * reduceParam.srcK], mask, tail,
                       dstRepStride,
                       1, reduceParam.srcK / FLOAT_NUM_PER_BLK);
    }
}


__aicore__ inline void CustomSingleBlockBroadCastImpl(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                                      const struct ReduceLastND &reduceParam) {
    BrcbRepeatParams brcbParams;
    brcbParams.dstBlkStride = 1;
    brcbParams.dstRepStride = BRCB_BROADCAST_NUMBER;
    const uint32_t range = reduceParam.originalSrcM / BRCB_BROADCAST_NUMBER;
    const uint32_t tail = reduceParam.originalSrcM % BRCB_BROADCAST_NUMBER;

    if (range != 0) {
        if (reduceParam.dstK == BRCB_BROADCAST_NUMBER * HALF_FACTOR) {
            brcbParams.dstBlkStride = HALF_FACTOR;
            brcbParams.dstRepStride = BRCB_BROADCAST_NUMBER * HALF_FACTOR;
            Brcb(dst[0], src, range, brcbParams);
            Brcb(dst[BRCB_BROADCAST_NUMBER], src, range, brcbParams);
        } else {
            Brcb(dst, src, range, brcbParams);
        }
    }

    if (tail != 0) {
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        float scalarList[SCALAR_STACK_DEPTH] = {0};
        for (uint32_t j = 0; j < tail; j++) {
            scalarList[j] = src[(range * BRCB_BROADCAST_NUMBER + j)].GetValue(0);
        }

        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        for (uint32_t k = 0; k < tail; k++) {
            Duplicate(dst[(range * SCALAR_STACK_DEPTH + k) * reduceParam.dstK], scalarList[k], reduceParam.dstK, 1,
                      DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        }
    }
}


__aicore__ inline void CustomReduceSumLastNDImpl(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                                 const LocalTensor<float> &tmpTensor,
                                                 const struct ReduceLastND &reduceParam) {
    const uint32_t splitCount = reduceParam.originalSrcK / FLOAT_REPEAT_SIZE;
    const uint32_t tailSrcK = reduceParam.originalSrcK % FLOAT_REPEAT_SIZE;
    if (splitCount > 0) {
        CustomAlignedReduceSumNDImpl(tmpTensor, src, dst, reduceParam, splitCount);
    }

    if (tailSrcK != 0) {
        CustomReduceSumLastNDSplitImpl(dst, src, reduceParam, tailSrcK, 1, splitCount);
        PipeBarrier<PIPE_V>();
        if (splitCount == 0) {
            DataCopy(tmpTensor, dst, {1, (uint16_t) reduceParam.srcM, 0, 0});
        } else {
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * FLOAT_NUM_PER_BLK);
            Add<float, false>(tmpTensor, tmpTensor, dst, MASK_PLACEHOLDER, 1,
                              {1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
            SetMaskNorm();
            ResetMask();
        }
    }

    PipeBarrier<PIPE_V>();
    CustomSingleBlockBroadCastImpl(dst, tmpTensor, reduceParam);
}


template<typename T, bool isBasicBlock = false>
__aicore__ inline void CustomSoftmaxGradFrontNDImpl(const LocalTensor <T> &dstTensor, const LocalTensor <T> &gradTensor,
                                                    const LocalTensor <T> &srcTensor,
                                                    const LocalTensor<float> &workLocal, const SoftMaxTiling &tiling,
                                                    const LastAxisShapeND &originalSrcShape) {
    uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);

    ReduceLastND reduceSumParam = {tiling.splitM, originalSrcShape.k, tiling.splitM,
                                   tiling.splitK, tiling.reduceM, tiling.reduceK};

    if constexpr(sizeof(T) == sizeof(half))
    {
        LocalTensor<float> srcBuffer = workLocal;
        LocalTensor<float> gradBuffer = workLocal[tiling.splitSize];
        LocalTensor<float> dstBuffer = workLocal[tiling.splitSize + tiling.splitSize];

        LocalTensor<float> reduceBuffer = workLocal[tiling.splitSize + tiling.splitSize + tiling.splitSize];
        LocalTensor<float> addBuffer =
                workLocal[tiling.splitSize + tiling.splitSize + tiling.splitSize + tiling.reduceSize];
        const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
        const uint32_t elementNumPerBlk = DEFAULT_C0_SIZE / B32_BYTE_SIZE;
        uint8_t offset = (uint8_t)(splitBlock * elementNumPerBlk);
        const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, FLOAT_NUM_PER_BLK));
        const uint8_t reduceCeilValue = (uint8_t)(DivCeil(tiling.reduceSize, FLOAT_REPEAT_SIZE));
        const uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
        SetMaskNorm();
        ResetMask();
        for (uint32_t i = 0; i < tiling.rangeM; i++) {
            if constexpr(isBasicBlock)
            {
                Cast<float, half, false>(srcBuffer, srcTensor[i * tiling.splitSize], RoundMode::CAST_NONE,
                                         MASK_PLACEHOLDER, repeatTimes,
                                         {1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE});
                Cast<float, half, false>(gradBuffer, gradTensor[i * tiling.splitSize], RoundMode::CAST_NONE,
                                         MASK_PLACEHOLDER, repeatTimes,
                                         {1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE});
                PipeBarrier<PIPE_V>();

                Mul<float, false>(dstBuffer, srcBuffer, gradBuffer, MASK_PLACEHOLDER, repeatTimes,
                                  {1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
                for (uint32_t j = 1; j < splitBlock; ++j) {
                    PipeBarrier<PIPE_V>();
                    Add<float, false>(dstBuffer, dstBuffer, dstBuffer[FLOAT_REPEAT_SIZE * j], MASK_PLACEHOLDER,
                                      (uint8_t)(tiling.splitM), {1, 1, 1, offset, offset, offset});
                }
                PipeBarrier<PIPE_V>();
                BlockReduceSum<float, false>(dstBuffer, dstBuffer, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                                             offset);
                PipeBarrier<PIPE_V>();
                BlockReduceSum<float, false>(reduceBuffer, dstBuffer, splitCeilM, MASK_PLACEHOLDER, 1, 1,
                                             DEFAULT_REPEAT_STRIDE);
                PipeBarrier<PIPE_V>();
                Brcb(dstBuffer, reduceBuffer, splitCeilM, {B16_BYTE_SIZE, DEFAULT_REPEAT_STRIDE * B16_BYTE_SIZE});

                Brcb(dstBuffer[DEFAULT_BLK_NUM], reduceBuffer, splitCeilM,
                     {B16_BYTE_SIZE, DEFAULT_REPEAT_STRIDE * B16_BYTE_SIZE});

                PipeBarrier<PIPE_V>();
                Cast<half, float, false>(dstTensor[i * tiling.reduceSize], dstBuffer, FLOAT2HALF_ROUND_MODE,
                                         MASK_PLACEHOLDER, reduceCeilValue,
                                         {1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
            } else {
                Cast(srcBuffer, srcTensor[i * tiling.splitSize], RoundMode::CAST_NONE, tiling.splitSize);
                Cast(gradBuffer, gradTensor[i * tiling.splitSize], RoundMode::CAST_NONE, tiling.splitSize);
                PipeBarrier<PIPE_V>();
                Mul(dstBuffer, srcBuffer, gradBuffer, tiling.splitSize);
                PipeBarrier<PIPE_V>();
                CustomReduceSumLastNDImpl(addBuffer, dstBuffer, reduceBuffer, reduceSumParam);
                PipeBarrier<PIPE_V>();
                Cast(dstTensor[i * tiling.reduceSize], addBuffer, FLOAT2HALF_ROUND_MODE, tiling.reduceSize);
            }
        }
        if (tiling.tailM != 0) {
            Cast(srcBuffer, srcTensor[tiling.rangeM * tiling.splitSize], RoundMode::CAST_NONE, tiling.tailSplitSize);
            Cast(gradBuffer, gradTensor[tiling.rangeM * tiling.splitSize], RoundMode::CAST_NONE, tiling.tailSplitSize);
            PipeBarrier<PIPE_V>();
            Mul(dstBuffer, srcBuffer, gradBuffer, tiling.tailSplitSize);
            reduceSumParam.srcM = tiling.tailM;
            reduceSumParam.dstM = tiling.tailM;
            reduceSumParam.originalSrcM = tiling.tailM;
            PipeBarrier<PIPE_V>();
            CustomReduceSumLastNDImpl(addBuffer, dstBuffer, reduceBuffer, reduceSumParam);
            PipeBarrier<PIPE_V>();
            Cast(dstTensor[tiling.rangeM * tiling.reduceSize], addBuffer, FLOAT2HALF_ROUND_MODE, tiling.tailReduceSize);
        }
    } else {
        LocalTensor<float> srcBuffer = workLocal;
        LocalTensor<float> reduceBuffer = workLocal[tiling.splitSize];
        uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
        uint32_t offset1 = 0;
        uint32_t offset2 = 0;
        const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
        const uint32_t elementNumPerBlk = DEFAULT_C0_SIZE / B32_BYTE_SIZE;
        uint8_t offset = (uint8_t)(splitBlock * elementNumPerBlk);
        const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, elementNumPerBlk));
        SetMaskNorm();
        ResetMask();
        for (uint32_t i = 0; i < tiling.rangeM; i++) {
            if constexpr(isBasicBlock)
            {
                offset2 = i * tiling.reduceSize;
                offset1 = i * tiling.splitSize;
                PipeBarrier<PIPE_V>();
                Mul<float, false>(srcBuffer, srcTensor[offset1], gradTensor[offset1], MASK_PLACEHOLDER, repeatTimes,
                                  {1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});

                for (uint32_t j = 1; j < splitBlock; ++j) {
                    PipeBarrier<PIPE_V>();
                    Add<float, false>(srcBuffer, srcBuffer, srcBuffer[FLOAT_REPEAT_SIZE * j], MASK_PLACEHOLDER,
                                      (uint8_t)(tiling.splitM), {1, 1, 1, offset, offset, offset});
                }
                PipeBarrier<PIPE_V>();
                BlockReduceSum<float, false>(srcBuffer, srcBuffer, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                                             splitBlock * DEFAULT_REPEAT_STRIDE);
                PipeBarrier<PIPE_V>();
                BlockReduceSum<float, false>(reduceBuffer, srcBuffer, splitCeilM, MASK_PLACEHOLDER, 1, 1,
                                             DEFAULT_REPEAT_STRIDE);
                PipeBarrier<PIPE_V>();

                Brcb(dstTensor[offset2], reduceBuffer, splitCeilM, {1, DEFAULT_REPEAT_STRIDE});

            } else {
                Mul(srcBuffer, srcTensor[i * tiling.splitSize], gradTensor[i * tiling.splitSize], tiling.splitSize);
                PipeBarrier<PIPE_V>();
                CustomReduceSumLastNDImpl(dstTensor[i * tiling.reduceSize], srcBuffer, reduceBuffer, reduceSumParam);
                PipeBarrier<PIPE_V>();
            }
        }

        if (tiling.tailM != 0) {
            Mul(srcBuffer, srcTensor[tiling.rangeM * tiling.splitSize], gradTensor[tiling.rangeM * tiling.splitSize],
                tiling.tailSplitSize);
            PipeBarrier<PIPE_V>();

            reduceSumParam.srcM = tiling.tailM;
            reduceSumParam.dstM = tiling.tailM;
            reduceSumParam.originalSrcM = tiling.tailM;
            CustomReduceSumLastNDImpl(dstTensor[tiling.rangeM * tiling.reduceSize], srcBuffer, reduceBuffer,
                                      reduceSumParam);
            PipeBarrier<PIPE_V>();
        }
    }
}


__aicore__ inline bool CustomSoftMaxGradTilingFunc(const uint32_t workLocalSize, const LastAxisShapeND &ndinfo,
                                                   SoftMaxTiling &softmaxTiling, const uint32_t elementNumPerBlk,
                                                   bool isFront = false, bool isBasicBlock = false,
                                                   bool isDataFormatNZ = false) {
    softmaxTiling.srcM = ndinfo.m;
    softmaxTiling.srcK = ndinfo.k;
    softmaxTiling.srcSize = ndinfo.m * ndinfo.k;

    softmaxTiling.outMaxM = ndinfo.m;
    softmaxTiling.outMaxK = elementNumPerBlk;
    softmaxTiling.outMaxSize = ndinfo.m * elementNumPerBlk;

    if (elementNumPerBlk != ONE_BYTE_BIT_SIZE) {
        softmaxTiling.reduceM = workLocalSize /
                                (elementNumPerBlk * SOFTMAX_COMPUTE_DIM + ndinfo.k * SOFTMAXGRAD_COMPUTE_DIM +
                                 FLOAT_REPEAT_SIZE);
    } else {
        if (isFront && !isDataFormatNZ) {
            softmaxTiling.reduceM = workLocalSize / (elementNumPerBlk + ndinfo.k + FLOAT_REPEAT_SIZE);
        } else {
            softmaxTiling.reduceM =
                    workLocalSize / (ndinfo.k + elementNumPerBlk * SOFTMAX_COMPUTE_DIM + FLOAT_REPEAT_SIZE);
        }
    }
    if (softmaxTiling.reduceM < ndinfo.m && softmaxTiling.reduceM > SOFTMAX_BASIC_TILE_NUM) {
        softmaxTiling.reduceM = softmaxTiling.reduceM / SOFTMAX_BASIC_TILE_NUM * SOFTMAX_BASIC_TILE_NUM;
    }
    softmaxTiling.reduceM = softmaxTiling.reduceM < ndinfo.m ? softmaxTiling.reduceM : ndinfo.m;

    if (isBasicBlock && isFront && (softmaxTiling.reduceM > SOFTMAX_BASIC_TILE_NUM) &&
        (softmaxTiling.srcM % SOFTMAX_BASIC_TILE_NUM == 0)) {
        softmaxTiling.reduceM = softmaxTiling.reduceM / SOFTMAX_BASIC_TILE_NUM * SOFTMAX_BASIC_TILE_NUM;
        while (softmaxTiling.srcM % softmaxTiling.reduceM != 0) {
            softmaxTiling.reduceM -= SOFTMAX_BASIC_TILE_NUM;
        }
        while (softmaxTiling.reduceM * ndinfo.k >= FLOAT_REPEAT_SIZE * DEFAULT_BLOCK_SIZE) {
            softmaxTiling.reduceM = softmaxTiling.reduceM / B16_BYTE_SIZE;
        }
    }

    softmaxTiling.reduceK = elementNumPerBlk;
    softmaxTiling.reduceSize = softmaxTiling.reduceM * elementNumPerBlk;

    softmaxTiling.splitM = softmaxTiling.reduceM;
    softmaxTiling.splitK = ndinfo.k;
    softmaxTiling.splitSize = softmaxTiling.reduceM * ndinfo.k;

    softmaxTiling.rangeM = ndinfo.m / softmaxTiling.reduceM;
    softmaxTiling.tailM = ndinfo.m % softmaxTiling.reduceM;

    softmaxTiling.tailSplitSize = softmaxTiling.tailM * ndinfo.k;
    softmaxTiling.tailReduceSize = softmaxTiling.tailM * elementNumPerBlk;
    return true;
}

template<typename T, bool isBasicBlock = false>
__aicore__ inline void SoftmaxGradFrontImpl(const LocalTensor <T> &dstTensor, const LocalTensor <T> &gradTensor,
                                            const LocalTensor <T> &srcTensor, const LocalTensor<float> &workLocal,
                                            const SoftMaxShapeInfo &softmaxShapeInfo) {
    ShapeInfo srcShape = srcTensor.GetShapeInfo();
    uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);
    LastAxisShapeND srcNDinfo;
    LastAxisShapeND originalSrcShape;

    srcNDinfo = GetLastAxisShapeND(srcShape);
    originalSrcShape = GetLastAxisOriginShapeND(srcShape);

    SoftMaxTiling newTiling{};
    CustomSoftMaxGradTilingFunc(workLocal.GetSize(), srcNDinfo, newTiling, elementNumPerBlk, true, isBasicBlock);
    CustomSoftmaxGradFrontNDImpl<T, isBasicBlock>(dstTensor, gradTensor, srcTensor, workLocal, newTiling,
                                                  originalSrcShape);

}

template<typename T, bool isBasicBlock = false>
__aicore__ inline void SoftmaxGradFrontImpl(const LocalTensor <T> &dstTensor, const LocalTensor <T> &gradTensor,
                                            const LocalTensor <T> &srcTensor,
                                            const LocalTensor <uint8_t> &sharedTmpBuffer,
                                            const SoftMaxShapeInfo &softmaxShapeInfo) {
    auto workLocal = sharedTmpBuffer.ReinterpretCast<float>();
    SoftmaxGradFrontImpl<T, isBasicBlock>(dstTensor, gradTensor, srcTensor, workLocal,
                                          softmaxShapeInfo);
}

template<typename T, bool isBasicBlock = false>
__aicore__ inline void SoftmaxGradFront(const LocalTensor <T> &dstTensor, const LocalTensor <T> &gradTensor,
                                        const LocalTensor <T> &srcTensor, const LocalTensor <uint8_t> &sharedTmpBuffer,
                                        const SoftMaxShapeInfo &softmaxShapeInfo = {}) {

    if ASCEND_IS_AIC{
                return;
        }
    SoftmaxGradFrontImpl<T, isBasicBlock>(dstTensor, gradTensor, srcTensor, sharedTmpBuffer,
                                          softmaxShapeInfo);
}


#endif //OPS_TRANSFORMER_SFMG_FLASH_ATTENTION_GRAD_CUSTOM_SFMG_H