/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef FAI950_ONLINE_SOFTMAX_QUANT_HPP
#define FAI950_ONLINE_SOFTMAX_QUANT_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "kernel_common.hpp"
#include "fa_block.h"
#include <limits>

namespace Catlass::Epilogue::Block {
using QuantMask::CAUSAL_MASK;
using QuantMask::NO_MASK;
using QuantMask::SLIDING_WINDOW_MASK;

template <bool HAS_SOFTCAP, MaskCategory maskCategory, class LayoutP, class LayoutS>
class BlockEpilogue<EpilogueFAQuantOnlineSoftmax<HAS_SOFTCAP, maskCategory>, Gemm::GemmType<float8_e4m3_t, LayoutP>,
                    Gemm::GemmType<float, LayoutS>> {
  public:
    using DispatchPolicy = EpilogueFAQuantOnlineSoftmax<HAS_SOFTCAP, maskCategory>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementInput = float;
    using ElementOutput = float8_e4m3_t;
    using ElementMask = uint8_t;
    static constexpr uint32_t UB_S_P_BUF_STAGES = 2;
    static constexpr uint32_t UB_DM_BUF_MAX_STAGES = 3;
    static constexpr uint32_t Q_BASE_TILE_SIZE_PER_SUBCORE = 64;
    static constexpr uint32_t KV_BASE_TILE_SIZE = 256;
    static constexpr uint32_t UB_BLOCK_BYTES = 32;
    static constexpr uint32_t FOUR_ISSUE_NUM = 4;
    static constexpr uint32_t KV_QUARTER_SIZE = KV_BASE_TILE_SIZE / FOUR_ISSUE_NUM;
    static constexpr uint32_t KV_HALF_SIZE = KV_BASE_TILE_SIZE / 2;
    static constexpr uint32_t KV_THREE_QUARTER_SIZE = KV_QUARTER_SIZE * 3;
    static constexpr uint32_t P_STRIPE_NUM = Q_BASE_TILE_SIZE_PER_SUBCORE * sizeof(ElementInput) / UB_BLOCK_BYTES;
    static constexpr uint32_t P_STRIPE_BLOCK_NUM = KV_BASE_TILE_SIZE / FOUR_ISSUE_NUM;
    static constexpr uint32_t P_STRIPE_BLOCK_STRIDE = P_STRIPE_BLOCK_NUM + 1;
    static constexpr float MIN_VALUE = -std::numeric_limits<float>::max();
    static constexpr float LN_P_SCALE = 5.5451774444795623f;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource, const UBufTileHelper& ub, float softcapValue_)
        : softcapValue(softcapValue_)
    {
        for (uint32_t i = 0; i < UB_S_P_BUF_STAGES; ++i) {
            lsUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(ub.sStartOffset + i * ub.sBufferBytes);
            lpUbTensor[i] =
                resource.ubBuf.template GetBufferByByte<ElementOutput>(ub.pStartOffset + i * ub.pBufferBytes);
        }
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(ub.gmStartOffset);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(ub.glStartOffset);
        for (uint32_t i = 0; i < UB_DM_BUF_MAX_STAGES; ++i) {
            dmUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                ub.dmStartOffset + i * Q_BASE_TILE_SIZE_PER_SUBCORE * sizeof(float));
        }
        pGatherIndexUbTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(ub.indexStartOffset);
        InitPGatherIndex((__ubuf__ uint8_t*)pGatherIndexUbTensor.GetPhyAddr());
    }

    template <class TensorP>
    __aicore__ inline void
    operator()(TensorP& l1PTensorTla, uint32_t rowNum, uint32_t qSBlockSize, uint32_t qSTileStart,
               uint32_t kvSTileStart, uint32_t kvSTileSize, int32_t qSeqlen, int32_t kvSeqlen, int64_t windowLeft,
               int64_t windowRight, float scaleValue, bool isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
               Arch::CrossCoreFlag mm1ToSmFlag, Arch::CrossCoreFlag smToMm2Flag)
    {
        const int32_t windowSizeLeft = static_cast<int32_t>(
            windowLeft < 0 ? kvSeqlen - 1 : min(windowLeft, static_cast<int64_t>(kvSeqlen - 1)));
        const int32_t windowSizeRight = static_cast<int32_t>(
            windowRight < 0 ? qSeqlen - 1 : min(windowRight, static_cast<int64_t>(qSeqlen - 1)));
        constexpr uint32_t rowAlign = 32;
        const uint32_t rowSplit = RoundUp(rowNum, rowAlign) / 2;
        const uint32_t rowNumFirstSubCore = min(rowNum, rowSplit);
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t rowNumCurSubCore = subBlockIdx == 0 ? rowNumFirstSubCore : rowNum - rowNumFirstSubCore;
        const uint32_t rowOffsetCurSubCore = subBlockIdx == 0 ? 0 : rowNumFirstSubCore;
        __ubuf__ float* sUbAddr = (__ubuf__ float*)lsUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ ElementOutput* pUbAddr = (__ubuf__ ElementOutput*)lpUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ float* gmUbAddr = (__ubuf__ float*)gmUbTensor.GetPhyAddr();
        __ubuf__ float* glUbAddr = (__ubuf__ float*)glUbTensor.GetPhyAddr();
        __ubuf__ float* dmUbAddr = (__ubuf__ float*)dmUbTensor[l1PBufId].GetPhyAddr();
        __ubuf__ uint8_t* pGatherIndexUbAddr = (__ubuf__ uint8_t*)pGatherIndexUbTensor.GetPhyAddr();

        Arch::CrossCoreWaitFlag<4, PIPE_V>(mm1ToSmFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + UB_S_P_BUF_STAGES);
        if (isFirstKvSTile) {
            ComputeOnlineSoftmax<false>(sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr, pGatherIndexUbAddr,
                                        rowNumCurSubCore, rowOffsetCurSubCore, qSBlockSize, qSTileStart, kvSTileStart,
                                        kvSTileSize, qSeqlen, kvSeqlen, windowSizeLeft, windowSizeRight, scaleValue);
        } else {
            ComputeOnlineSoftmax<true>(sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr, pGatherIndexUbAddr,
                                       rowNumCurSubCore, rowOffsetCurSubCore, qSBlockSize, qSTileStart, kvSTileStart,
                                       kvSTileSize, qSeqlen, kvSeqlen, windowSizeLeft, windowSizeRight, scaleValue);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        Arch::CrossCoreSetFlag<4, PIPE_V>(mm1ToSmFlag);
        Arch::CrossCoreWaitFlag<4, PIPE_MTE3>(smToMm2Flag);

        // Use complete stripes; PV uses the same aligned P stride in L1.
        const uint32_t pKvRound = RoundUp(kvSTileSize, P_STRIPE_BLOCK_NUM);
        const uint32_t pPartElems = Q_BASE_TILE_SIZE_PER_SUBCORE / 2 * pKvRound;
        const uint32_t pSubCoreOffset = subBlockIdx * 2 * pPartElems;
        AscendC::DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = pKvRound / P_STRIPE_BLOCK_NUM;
        dataCopyParams.blockLen = P_STRIPE_BLOCK_NUM;
        dataCopyParams.srcStride = 2 * P_STRIPE_BLOCK_STRIDE - P_STRIPE_BLOCK_NUM;
        dataCopyParams.dstStride = 0;
        AscendC::DataCopy(l1PTensorTla.data()[pSubCoreOffset], lpUbTensor[ubSBufId], dataCopyParams);
        AscendC::DataCopy(l1PTensorTla.data()[pSubCoreOffset + pPartElems],
                          lpUbTensor[ubSBufId][P_STRIPE_BLOCK_STRIDE * UB_BLOCK_BYTES / sizeof(ElementOutput)],
                          dataCopyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + UB_S_P_BUF_STAGES);
        Arch::CrossCoreSetFlag<4, PIPE_MTE3>(smToMm2Flag);
    }

  private:
    __simd_vf__ static inline void InitPGatherIndex(__ubuf__ uint8_t* pGatherIndexUbAddr)
    {
        using namespace AscendC;
        using namespace AscendC::Reg;
        RegTensor<uint8_t> baseIndexVreg;
        RegTensor<uint8_t> indexVreg;
        RegTensor<uint8_t> indexHighVreg;
        MaskReg indexMask = CreateMask<int8_t, MaskPattern::VL32>();

        Arange((RegTensor<int8_t>&)baseIndexVreg, 0);
        ShiftLefts(baseIndexVreg, baseIndexVreg, static_cast<int16_t>(2), indexMask);
        for (uint16_t i = 0; i < 4; ++i) {
            Adds(indexVreg, baseIndexVreg, i, indexMask);
            StoreAlign(pGatherIndexUbAddr + i * 64, indexVreg, indexMask);
            Adds(indexHighVreg, indexVreg, static_cast<uint8_t>(128), indexMask);
            StoreAlign(pGatherIndexUbAddr + i * 64 + 32, indexHighVreg, indexMask);
        }
    }

    enum class KvRange : uint32_t {
        LOWER_QUARTER,
        LOWER_HALF,
        LOWER_THREE_QUARTERS,
        UPPER_THREE_QUARTERS,
        UPPER_HALF,
        UPPER_QUARTER,
        FULL
    };

    template <KvRange Range>
    struct KvRangeInfo {
        static constexpr uint32_t BEGIN = Range == KvRange::UPPER_QUARTER          ? KV_THREE_QUARTER_SIZE
                                          : Range == KvRange::UPPER_HALF           ? KV_HALF_SIZE
                                          : Range == KvRange::UPPER_THREE_QUARTERS ? KV_QUARTER_SIZE
                                                                                   : 0;
        static constexpr uint32_t END = Range == KvRange::LOWER_QUARTER          ? KV_QUARTER_SIZE
                                        : Range == KvRange::LOWER_HALF           ? KV_HALF_SIZE
                                        : Range == KvRange::LOWER_THREE_QUARTERS ? KV_THREE_QUARTER_SIZE
                                                                                 : KV_BASE_TILE_SIZE;
    };

    float softcapValue;
    AscendC::LocalTensor<float> lsUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<ElementOutput> lpUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<float> dmUbTensor[UB_DM_BUF_MAX_STAGES];
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<uint8_t> pGatherIndexUbTensor;

    template <class Reg, class Mask>
    __simd_callee__ static inline void ApplySoftcap(Reg& valueVreg, float softcapValue, Mask& activeMask)
    {
        using namespace AscendC::Reg;
        Reg numeratorVreg, denominatorVreg;
        Muls(valueVreg, valueVreg, -2.0f, activeMask);
        Mins(valueVreg, valueVreg, 17.6f, activeMask);
        Exp(valueVreg, valueVreg, activeMask);
        Duplicate(numeratorVreg, softcapValue);
        Axpy(numeratorVreg, valueVreg, -softcapValue, activeMask);
        Adds(denominatorVreg, valueVreg, 1.0f, activeMask);
        Div(valueVreg, numeratorVreg, denominatorVreg, activeMask);
    }

    __simd_callee__ static inline void ApplyKvTailMask(AscendC::Reg::RegTensor<float>& sVreg,
                                                       AscendC::Reg::RegTensor<int32_t>& kvEndVreg, int32_t kvPosition,
                                                       AscendC::Reg::MaskReg& validRowB32Mask)
    {
        using namespace AscendC::Reg;
        RegTensor<float> minVreg;
        MaskReg visibleMask;
        Duplicate(minVreg, MIN_VALUE);
        CompareScalar<int32_t, AscendC::CMPMODE::GE>(visibleMask, kvEndVreg, kvPosition, validRowB32Mask);
        Select(sVreg, sVreg, minVreg, visibleMask);
    }

    // Merge KV tail visibility before the attention mask's final Select.
    __simd_callee__ static inline void MergeKvTailMask(AscendC::Reg::MaskReg& visibleMask, uint32_t kvRow,
                                                       uint32_t kvSTileSize, AscendC::Reg::MaskReg& validRowB32Mask)
    {
        using namespace AscendC::Reg;
        uint32_t validRowCount = kvRow < kvSTileSize ? Q_BASE_TILE_SIZE_PER_SUBCORE : 0U;
        MaskReg kvTailB32Mask = UpdateMask<float>(validRowCount);
        MaskAnd(visibleMask, visibleMask, kvTailB32Mask, validRowB32Mask);
    }

    template <bool IsFullKvTile>
    __simd_callee__ static inline void
    ApplyAttentionMask(AscendC::Reg::RegTensor<float>& sVreg, AscendC::Reg::RegTensor<int32_t>& qIndexVreg,
                       uint32_t kvRow, uint32_t kvSTileStart, uint32_t kvSTileSize, int32_t windowSizeLeft,
                       int32_t windowSizeRight, AscendC::Reg::MaskReg& validRowB32Mask)
    {
        using namespace AscendC::Reg;
        RegTensor<float> minVreg;
        MaskReg visibleMask;
        Duplicate(minVreg, MIN_VALUE);
        const int32_t kvPosition = static_cast<int32_t>(kvSTileStart + kvRow);
        if constexpr (maskCategory == CAUSAL_MASK) {
            // qIndexVreg is already bounded by the last valid KV position.
            CompareScalar<int32_t, AscendC::CMPMODE::GE>(visibleMask, qIndexVreg, kvPosition, validRowB32Mask);
        } else if constexpr (maskCategory == SLIDING_WINDOW_MASK) {
            RegTensor<uint32_t> windowDistanceVreg;
            Adds(windowDistanceVreg, (RegTensor<uint32_t>&)qIndexVreg,
                 static_cast<uint32_t>(windowSizeRight - kvPosition), validRowB32Mask);
            CompareScalar<uint32_t, AscendC::CMPMODE::LE>(visibleMask, windowDistanceVreg,
                                                          static_cast<uint32_t>(windowSizeLeft) + static_cast<uint32_t>(windowSizeRight),
                                                          validRowB32Mask);
            if constexpr (!IsFullKvTile) {
                MergeKvTailMask(visibleMask, kvRow, kvSTileSize, validRowB32Mask);
            }
        }
        Select(sVreg, sVreg, minVreg, visibleMask);
    }

    template <bool IsUpdate>
    __aicore__ inline void
    ComputeOnlineSoftmax(__ubuf__ float* sUbAddr, __ubuf__ ElementOutput* pUbAddr, __ubuf__ float* gmUbAddr,
                         __ubuf__ float* glUbAddr, __ubuf__ float* dmUbAddr, __ubuf__ uint8_t* pGatherIndexUbAddr,
                         uint32_t rowNumCurSubCore, uint32_t rowOffsetCurSubCore, uint32_t qSBlockSize,
                         uint32_t qSTileStart, uint32_t kvSTileStart, uint32_t kvSTileSize, int32_t qSeqlen,
                         int32_t kvSeqlen, int32_t windowSizeLeft, int32_t windowSizeRight, float scaleValue)
    {
        const int32_t qTileCenterStart = static_cast<int32_t>(qSTileStart) + kvSeqlen - qSeqlen;
        if constexpr (maskCategory == NO_MASK) {
            ComputeUnmaskedOnlineSoftmax<IsUpdate>(sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr, pGatherIndexUbAddr,
                                                   rowNumCurSubCore, kvSTileStart, kvSTileSize, scaleValue);
        } else if constexpr (maskCategory == CAUSAL_MASK) {
            if (static_cast<int32_t>(kvSTileStart + kvSTileSize) > qTileCenterStart) {
                ComputeMaskedOnlineSoftmax<IsUpdate, false>(
                    sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr, pGatherIndexUbAddr, rowNumCurSubCore,
                    rowOffsetCurSubCore, qSBlockSize, qTileCenterStart, kvSTileStart, kvSTileSize, 0, 0, scaleValue);
            } else {
                ComputeUnmaskedOnlineSoftmax<IsUpdate>(sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr,
                                                       pGatherIndexUbAddr, rowNumCurSubCore, kvSTileStart, kvSTileSize,
                                                       scaleValue);
            }
        } else if constexpr (maskCategory == SLIDING_WINDOW_MASK) {
            const bool fullyVisible =
                static_cast<int32_t>(kvSTileStart) >=
                    qTileCenterStart + static_cast<int32_t>(qSBlockSize) - 1 - windowSizeLeft &&
                static_cast<int32_t>(kvSTileStart + kvSTileSize) - 1 <= qTileCenterStart + windowSizeRight;
            if (kvSTileSize == KV_BASE_TILE_SIZE) {
                if (fullyVisible) {
                    ComputeUnmaskedOnlineSoftmax<IsUpdate>(sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr,
                                                           pGatherIndexUbAddr, rowNumCurSubCore, kvSTileStart,
                                                           kvSTileSize, scaleValue);
                } else {
                    ComputeMaskedOnlineSoftmax<IsUpdate, true>(
                        sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr, pGatherIndexUbAddr, rowNumCurSubCore,
                        rowOffsetCurSubCore, qSBlockSize, qTileCenterStart, kvSTileStart, kvSTileSize, windowSizeLeft,
                        windowSizeRight, scaleValue);
                }
            } else {
                ComputeMaskedOnlineSoftmax<IsUpdate, false>(sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr,
                                                            pGatherIndexUbAddr, rowNumCurSubCore, rowOffsetCurSubCore,
                                                            qSBlockSize, qTileCenterStart, kvSTileStart, kvSTileSize,
                                                            windowSizeLeft, windowSizeRight, scaleValue);
            }
        }
    }

    template <bool IsUpdate>
    __aicore__ inline void ComputeUnmaskedOnlineSoftmax(__ubuf__ float* sUbAddr, __ubuf__ ElementOutput* pUbAddr,
                                                        __ubuf__ float* gmUbAddr, __ubuf__ float* glUbAddr,
                                                        __ubuf__ float* dmUbAddr, __ubuf__ uint8_t* pGatherIndexUbAddr,
                                                        uint32_t rowNumCurSubCore, uint32_t kvSTileStart,
                                                        uint32_t kvSTileSize, float scaleValue)
    {
        if (kvSTileSize == KV_BASE_TILE_SIZE) {
            ComputeUnmaskedOnlineSoftmaxImpl<IsUpdate, true>(sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr,
                                                             pGatherIndexUbAddr, rowNumCurSubCore, kvSTileStart,
                                                             kvSTileSize, scaleValue, softcapValue);
        } else if (kvSTileSize <= KV_QUARTER_SIZE) {
            ComputeUnmaskedOnlineSoftmaxImpl<IsUpdate, false, KvRange::LOWER_QUARTER>(
                sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr, pGatherIndexUbAddr, rowNumCurSubCore, kvSTileStart,
                kvSTileSize, scaleValue, softcapValue);
        } else if (kvSTileSize <= KV_HALF_SIZE) {
            ComputeUnmaskedOnlineSoftmaxImpl<IsUpdate, false, KvRange::LOWER_HALF>(
                sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr, pGatherIndexUbAddr, rowNumCurSubCore, kvSTileStart,
                kvSTileSize, scaleValue, softcapValue);
        } else if (kvSTileSize <= KV_THREE_QUARTER_SIZE) {
            ComputeUnmaskedOnlineSoftmaxImpl<IsUpdate, false, KvRange::LOWER_THREE_QUARTERS>(
                sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr, pGatherIndexUbAddr, rowNumCurSubCore, kvSTileStart,
                kvSTileSize, scaleValue, softcapValue);
        } else {
            ComputeUnmaskedOnlineSoftmaxImpl<IsUpdate, false>(sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr,
                                                              pGatherIndexUbAddr, rowNumCurSubCore, kvSTileStart,
                                                              kvSTileSize, scaleValue, softcapValue);
        }
    }

    template <bool IsUpdate, bool IsFullKvTile>
    __aicore__ inline void
    ComputeMaskedOnlineSoftmax(__ubuf__ float* sUbAddr, __ubuf__ ElementOutput* pUbAddr, __ubuf__ float* gmUbAddr,
                               __ubuf__ float* glUbAddr, __ubuf__ float* dmUbAddr, __ubuf__ uint8_t* pGatherIndexUbAddr,
                               uint32_t rowNumCurSubCore, uint32_t rowOffsetCurSubCore, uint32_t qSBlockSize,
                               int32_t qTileCenterStart, uint32_t kvSTileStart, uint32_t kvSTileSize,
                               int32_t windowSizeLeft, int32_t windowSizeRight, float scaleValue)
    {
        // Select a fixed KV range before VF; inactive chains produce zero P.
        uint32_t qSOffset = rowOffsetCurSubCore % qSBlockSize;
        bool crossQHeads = rowNumCurSubCore > qSBlockSize - qSOffset;
        int32_t firstQCenter = qTileCenterStart + static_cast<int32_t>(crossQHeads ? 0U : qSOffset);
        int32_t lastQCenter =
            qTileCenterStart + static_cast<int32_t>(crossQHeads
                                                        ? qSBlockSize - 1U
                                                        : qSOffset + (rowNumCurSubCore ? rowNumCurSubCore - 1U : 0U));
        int32_t kvQuarterStart = static_cast<int32_t>(kvSTileStart) + static_cast<int32_t>(KV_QUARTER_SIZE);
        int32_t kvHalfStart = static_cast<int32_t>(kvSTileStart) + static_cast<int32_t>(KV_HALF_SIZE);
        int32_t kvThreeQuarterStart = static_cast<int32_t>(kvSTileStart) + static_cast<int32_t>(KV_THREE_QUARTER_SIZE);
        int32_t firstVisibleKv = static_cast<int32_t>(kvSTileStart);
        int32_t lastVisibleKv = static_cast<int32_t>(kvSTileStart + kvSTileSize) - 1;
        if constexpr (maskCategory == CAUSAL_MASK) {
            lastVisibleKv = min(lastQCenter, lastVisibleKv);
        } else if constexpr (maskCategory == SLIDING_WINDOW_MASK) {
            firstVisibleKv = firstQCenter - windowSizeLeft;
            lastVisibleKv = min(lastQCenter + windowSizeRight, lastVisibleKv);
        }
        if (lastVisibleKv < kvQuarterStart) {
            ComputeMaskedOnlineSoftmaxImpl<IsUpdate, IsFullKvTile, KvRange::LOWER_QUARTER>(
                sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr, pGatherIndexUbAddr, rowNumCurSubCore,
                rowOffsetCurSubCore, qSBlockSize, qTileCenterStart, kvSTileStart, kvSTileSize, scaleValue, softcapValue,
                windowSizeLeft, windowSizeRight);
        } else if (firstVisibleKv >= kvThreeQuarterStart) {
            ComputeMaskedOnlineSoftmaxImpl<IsUpdate, IsFullKvTile, KvRange::UPPER_QUARTER>(
                sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr, pGatherIndexUbAddr, rowNumCurSubCore,
                rowOffsetCurSubCore, qSBlockSize, qTileCenterStart, kvSTileStart, kvSTileSize, scaleValue, softcapValue,
                windowSizeLeft, windowSizeRight);
        } else if (lastVisibleKv < kvHalfStart) {
            ComputeMaskedOnlineSoftmaxImpl<IsUpdate, IsFullKvTile, KvRange::LOWER_HALF>(
                sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr, pGatherIndexUbAddr, rowNumCurSubCore,
                rowOffsetCurSubCore, qSBlockSize, qTileCenterStart, kvSTileStart, kvSTileSize, scaleValue, softcapValue,
                windowSizeLeft, windowSizeRight);
        } else if (firstVisibleKv >= kvHalfStart) {
            ComputeMaskedOnlineSoftmaxImpl<IsUpdate, IsFullKvTile, KvRange::UPPER_HALF>(
                sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr, pGatherIndexUbAddr, rowNumCurSubCore,
                rowOffsetCurSubCore, qSBlockSize, qTileCenterStart, kvSTileStart, kvSTileSize, scaleValue, softcapValue,
                windowSizeLeft, windowSizeRight);
        } else if (lastVisibleKv < kvThreeQuarterStart) {
            ComputeMaskedOnlineSoftmaxImpl<IsUpdate, IsFullKvTile, KvRange::LOWER_THREE_QUARTERS>(
                sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr, pGatherIndexUbAddr, rowNumCurSubCore,
                rowOffsetCurSubCore, qSBlockSize, qTileCenterStart, kvSTileStart, kvSTileSize, scaleValue, softcapValue,
                windowSizeLeft, windowSizeRight);
        } else if (firstVisibleKv >= kvQuarterStart) {
            ComputeMaskedOnlineSoftmaxImpl<IsUpdate, IsFullKvTile, KvRange::UPPER_THREE_QUARTERS>(
                sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr, pGatherIndexUbAddr, rowNumCurSubCore,
                rowOffsetCurSubCore, qSBlockSize, qTileCenterStart, kvSTileStart, kvSTileSize, scaleValue, softcapValue,
                windowSizeLeft, windowSizeRight);
        } else {
            ComputeMaskedOnlineSoftmaxImpl<IsUpdate, IsFullKvTile>(
                sUbAddr, pUbAddr, gmUbAddr, glUbAddr, dmUbAddr, pGatherIndexUbAddr, rowNumCurSubCore,
                rowOffsetCurSubCore, qSBlockSize, qTileCenterStart, kvSTileStart, kvSTileSize, scaleValue, softcapValue,
                windowSizeLeft, windowSizeRight);
        }
    }

    template <bool IsUpdate, bool IsFullKvTile, KvRange ValidRange = KvRange::FULL>
    __simd_vf__ static inline void
    ComputeUnmaskedOnlineSoftmaxImpl(__ubuf__ float* sUbAddr, __ubuf__ ElementOutput* pUbAddr, __ubuf__ float* gmUbAddr,
                                     __ubuf__ float* glUbAddr, __ubuf__ float* dmUbAddr,
                                     __ubuf__ uint8_t* pGatherIndexUbAddr, uint32_t rowNumCurSubCore,
                                     uint32_t kvSTileStart, uint32_t kvSTileSize, float scaleValue, float softcapValue)
    {
        using namespace AscendC::Reg;
        constexpr uint32_t kvSegmentSize = KV_QUARTER_SIZE;
        constexpr uint32_t pStripeBlockStride = P_STRIPE_BLOCK_STRIDE;
        constexpr uint32_t kvBegin = KvRangeInfo<ValidRange>::BEGIN;
        constexpr uint32_t kvEnd = KvRangeInfo<ValidRange>::END;
        constexpr bool computeKvSegment0 = kvBegin <= 0U && kvEnd > 0U;
        constexpr bool computeKvSegment1 = kvBegin <= KV_QUARTER_SIZE && kvEnd > KV_QUARTER_SIZE;
        constexpr bool computeKvSegment2 = kvBegin <= KV_HALF_SIZE && kvEnd > KV_HALF_SIZE;
        constexpr bool computeKvSegment3 = kvBegin <= KV_THREE_QUARTER_SIZE && kvEnd > KV_THREE_QUARTER_SIZE;
        static constexpr CastTrait castTraitZeroRound = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait castTraitOneRound = {RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait castTraitTwoRound = {RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait castTraitThreeRound = {RegLayout::THREE, SatMode::SAT, MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};

        MaskReg allB32Mask = CreateMask<float, MaskPattern::ALL>();
        MaskReg validRowB32Mask = UpdateMask<float>(rowNumCurSubCore);
        MaskReg allB8Mask = CreateMask<uint8_t, MaskPattern::ALL>();
        RegTensor<float> s0Vreg, s1Vreg, s2Vreg, s3Vreg;
        RegTensor<float> lmVreg;
        RegTensor<float> p0Vreg, p1Vreg, p2Vreg, p3Vreg;
        RegTensor<float> ll0Vreg, ll1Vreg, ll2Vreg, ll3Vreg;
        RegTensor<float> gmVreg, glVreg, dmVreg, rescaledGlVreg;
        RegTensor<int32_t> kvEndVreg;
        RegTensor<ElementOutput> pB8Vreg, packedPVreg;
        RegTensor<uint8_t> gatherIndexVreg;

        if constexpr (!IsFullKvTile) {
            Duplicate(kvEndVreg, static_cast<int32_t>(kvSTileStart + kvSTileSize) - 1);
        }
        // Max pass: cache transformed S with softcap; otherwise reduce raw S.
        RegTensor<float> lm0Vreg, lm1Vreg, lm2Vreg, lm3Vreg;
        Duplicate(lm0Vreg, MIN_VALUE);
        Duplicate(lm1Vreg, MIN_VALUE);
        Duplicate(lm2Vreg, MIN_VALUE);
        Duplicate(lm3Vreg, MIN_VALUE);
        constexpr uint32_t kvCount = kvEnd - kvBegin;
        for (uint32_t i = 0; i < kvCount / FOUR_ISSUE_NUM; ++i) {
            LoadAlign(s0Vreg, sUbAddr + (kvBegin + i * FOUR_ISSUE_NUM) * Q_BASE_TILE_SIZE_PER_SUBCORE);
            if constexpr (HAS_SOFTCAP) {
                Muls(s0Vreg, s0Vreg, scaleValue, validRowB32Mask);
                ApplySoftcap(s0Vreg, softcapValue, validRowB32Mask);
            }
            if constexpr (!IsFullKvTile) {
                ApplyKvTailMask(s0Vreg, kvEndVreg, static_cast<int32_t>(kvSTileStart + kvBegin + i * FOUR_ISSUE_NUM),
                                validRowB32Mask);
            }
            LoadAlign(s1Vreg, sUbAddr + (kvBegin + i * FOUR_ISSUE_NUM + 1) * Q_BASE_TILE_SIZE_PER_SUBCORE);
            if constexpr (HAS_SOFTCAP) {
                Muls(s1Vreg, s1Vreg, scaleValue, validRowB32Mask);
                ApplySoftcap(s1Vreg, softcapValue, validRowB32Mask);
            }
            if constexpr (!IsFullKvTile) {
                ApplyKvTailMask(s1Vreg, kvEndVreg,
                                static_cast<int32_t>(kvSTileStart + kvBegin + i * FOUR_ISSUE_NUM + 1), validRowB32Mask);
            }
            LoadAlign(s2Vreg, sUbAddr + (kvBegin + i * FOUR_ISSUE_NUM + 2) * Q_BASE_TILE_SIZE_PER_SUBCORE);
            if constexpr (HAS_SOFTCAP) {
                Muls(s2Vreg, s2Vreg, scaleValue, validRowB32Mask);
                ApplySoftcap(s2Vreg, softcapValue, validRowB32Mask);
            }
            if constexpr (!IsFullKvTile) {
                ApplyKvTailMask(s2Vreg, kvEndVreg,
                                static_cast<int32_t>(kvSTileStart + kvBegin + i * FOUR_ISSUE_NUM + 2), validRowB32Mask);
            }
            LoadAlign(s3Vreg, sUbAddr + (kvBegin + i * FOUR_ISSUE_NUM + 3) * Q_BASE_TILE_SIZE_PER_SUBCORE);
            if constexpr (HAS_SOFTCAP) {
                Muls(s3Vreg, s3Vreg, scaleValue, validRowB32Mask);
                ApplySoftcap(s3Vreg, softcapValue, validRowB32Mask);
            }
            if constexpr (!IsFullKvTile) {
                ApplyKvTailMask(s3Vreg, kvEndVreg,
                                static_cast<int32_t>(kvSTileStart + kvBegin + i * FOUR_ISSUE_NUM + 3), validRowB32Mask);
            }

            if constexpr (HAS_SOFTCAP) {
                StoreAlign(sUbAddr + (kvBegin + i * FOUR_ISSUE_NUM) * Q_BASE_TILE_SIZE_PER_SUBCORE, s0Vreg,
                           validRowB32Mask);
                StoreAlign(sUbAddr + (kvBegin + i * FOUR_ISSUE_NUM + 1) * Q_BASE_TILE_SIZE_PER_SUBCORE, s1Vreg,
                           validRowB32Mask);
                StoreAlign(sUbAddr + (kvBegin + i * FOUR_ISSUE_NUM + 2) * Q_BASE_TILE_SIZE_PER_SUBCORE, s2Vreg,
                           validRowB32Mask);
                StoreAlign(sUbAddr + (kvBegin + i * FOUR_ISSUE_NUM + 3) * Q_BASE_TILE_SIZE_PER_SUBCORE, s3Vreg,
                           validRowB32Mask);
            }

            Max(lm0Vreg, lm0Vreg, s0Vreg, validRowB32Mask);
            Max(lm1Vreg, lm1Vreg, s1Vreg, validRowB32Mask);
            Max(lm2Vreg, lm2Vreg, s2Vreg, validRowB32Mask);
            Max(lm3Vreg, lm3Vreg, s3Vreg, validRowB32Mask);
        }
        Max(lm0Vreg, lm0Vreg, lm2Vreg, validRowB32Mask);
        Max(lm1Vreg, lm1Vreg, lm3Vreg, validRowB32Mask);
        Max(lmVreg, lm0Vreg, lm1Vreg, validRowB32Mask);
        if constexpr (HAS_SOFTCAP) {
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        } else {
            Muls(lmVreg, lmVreg, scaleValue, validRowB32Mask);
            Maxs(lmVreg, lmVreg, MIN_VALUE, validRowB32Mask);
        }

        // Update the online max and correction, reusing the local max register.
        auto& hmVreg = lmVreg;
        if constexpr (IsUpdate) {
            LoadAlign(gmVreg, gmUbAddr);
            LoadAlign(glVreg, glUbAddr);
            Max(hmVreg, lmVreg, gmVreg, validRowB32Mask);
            FusedExpSub(dmVreg, gmVreg, hmVreg, validRowB32Mask);
            Mul(rescaledGlVreg, dmVreg, glVreg, validRowB32Mask);
            StoreAlign(dmUbAddr, dmVreg, allB32Mask);
        }
        StoreAlign(gmUbAddr, hmVreg, allB32Mask);

        // Exp/P pass: read cached S with softcap; otherwise scale raw S here.
        // P and its FP32 sum share P_SCALE; the saved online max remains unshifted.
        Adds(hmVreg, hmVreg, -LN_P_SCALE, validRowB32Mask);
        Duplicate(ll0Vreg, 0.0f);
        Duplicate(ll1Vreg, 0.0f);
        Duplicate(ll2Vreg, 0.0f);
        Duplicate(ll3Vreg, 0.0f);
        LoadAlign(gatherIndexVreg, pGatherIndexUbAddr);
        __ubuf__ ElementOutput* pStoreUbAddr = pUbAddr;

        // Cast four FP32 exp vectors into byte lanes, then Gather into the PV nZ stripes.
        for (uint32_t i = 0; i < kvSegmentSize; ++i) {
            if constexpr (computeKvSegment0) {
                LoadAlign(s0Vreg, sUbAddr + i * Q_BASE_TILE_SIZE_PER_SUBCORE);
                if constexpr (!HAS_SOFTCAP) {
                    Muls(s0Vreg, s0Vreg, scaleValue, validRowB32Mask);
                    if constexpr (!IsFullKvTile) {
                        ApplyKvTailMask(s0Vreg, kvEndVreg, static_cast<int32_t>(kvSTileStart + i), validRowB32Mask);
                    }
                }
            }
            if constexpr (computeKvSegment1) {
                LoadAlign(s1Vreg, sUbAddr + (i + kvSegmentSize * 1) * Q_BASE_TILE_SIZE_PER_SUBCORE);
                if constexpr (!HAS_SOFTCAP) {
                    Muls(s1Vreg, s1Vreg, scaleValue, validRowB32Mask);
                    if constexpr (!IsFullKvTile) {
                        ApplyKvTailMask(s1Vreg, kvEndVreg, static_cast<int32_t>(kvSTileStart + i + kvSegmentSize * 1),
                                        validRowB32Mask);
                    }
                }
            }
            if constexpr (computeKvSegment2) {
                LoadAlign(s2Vreg, sUbAddr + (i + kvSegmentSize * 2) * Q_BASE_TILE_SIZE_PER_SUBCORE);
                if constexpr (!HAS_SOFTCAP) {
                    Muls(s2Vreg, s2Vreg, scaleValue, validRowB32Mask);
                    if constexpr (!IsFullKvTile) {
                        ApplyKvTailMask(s2Vreg, kvEndVreg, static_cast<int32_t>(kvSTileStart + i + kvSegmentSize * 2),
                                        validRowB32Mask);
                    }
                }
            }
            if constexpr (computeKvSegment3) {
                LoadAlign(s3Vreg, sUbAddr + (i + kvSegmentSize * 3) * Q_BASE_TILE_SIZE_PER_SUBCORE);
                if constexpr (!HAS_SOFTCAP) {
                    Muls(s3Vreg, s3Vreg, scaleValue, validRowB32Mask);
                    if constexpr (!IsFullKvTile) {
                        ApplyKvTailMask(s3Vreg, kvEndVreg, static_cast<int32_t>(kvSTileStart + i + kvSegmentSize * 3),
                                        validRowB32Mask);
                    }
                }
            }
            if constexpr (computeKvSegment0) {
                FusedExpSub(p0Vreg, s0Vreg, hmVreg, validRowB32Mask);
            }
            if constexpr (computeKvSegment1) {
                FusedExpSub(p1Vreg, s1Vreg, hmVreg, validRowB32Mask);
            } else {
                Duplicate(p1Vreg, 0.0f);
            }
            if constexpr (computeKvSegment2) {
                FusedExpSub(p2Vreg, s2Vreg, hmVreg, validRowB32Mask);
            } else {
                Duplicate(p2Vreg, 0.0f);
            }
            if constexpr (computeKvSegment3) {
                FusedExpSub(p3Vreg, s3Vreg, hmVreg, validRowB32Mask);
            } else {
                Duplicate(p3Vreg, 0.0f);
            }
            if constexpr (computeKvSegment0) {
                Add(ll0Vreg, ll0Vreg, p0Vreg, validRowB32Mask);
            }
            if constexpr (computeKvSegment1) {
                Add(ll1Vreg, ll1Vreg, p1Vreg, validRowB32Mask);
            }
            if constexpr (computeKvSegment2) {
                Add(ll2Vreg, ll2Vreg, p2Vreg, validRowB32Mask);
            }
            if constexpr (computeKvSegment3) {
                Add(ll3Vreg, ll3Vreg, p3Vreg, validRowB32Mask);
            }
            // Reuse FP32 registers after sum and Cast; preserve this alias and Cast order.
            auto& pPack1Vreg = (RegTensor<ElementOutput>&)p0Vreg;
            auto& pPack2Vreg = (RegTensor<ElementOutput>&)p1Vreg;
            auto& pPack3Vreg = (RegTensor<ElementOutput>&)p2Vreg;
            // Cast and pack the four byte lanes before Gather and StoreAlign.
            if constexpr (computeKvSegment0) {
                Cast<ElementOutput, float, castTraitZeroRound>(pB8Vreg, p0Vreg, validRowB32Mask);
            }
            if constexpr (computeKvSegment1) {
                Cast<ElementOutput, float, castTraitOneRound>(pPack1Vreg, p1Vreg, validRowB32Mask);
            }
            if constexpr (computeKvSegment2) {
                Cast<ElementOutput, float, castTraitTwoRound>(pPack2Vreg, p2Vreg, validRowB32Mask);
            }
            if constexpr (computeKvSegment3) {
                Cast<ElementOutput, float, castTraitThreeRound>(pPack3Vreg, p3Vreg, validRowB32Mask);
            }
            if constexpr (computeKvSegment0 && computeKvSegment1) {
                Or((RegTensor<uint8_t>&)pB8Vreg, (RegTensor<uint8_t>&)pB8Vreg, (RegTensor<uint8_t>&)pPack1Vreg,
                   allB8Mask);
            }
            if constexpr (computeKvSegment1 && computeKvSegment3) {
                Or((RegTensor<uint8_t>&)pPack2Vreg, (RegTensor<uint8_t>&)pPack2Vreg, (RegTensor<uint8_t>&)pPack3Vreg,
                   allB8Mask);
                Or((RegTensor<uint8_t>&)pB8Vreg, (RegTensor<uint8_t>&)pB8Vreg, (RegTensor<uint8_t>&)pPack2Vreg,
                   allB8Mask);
            } else if constexpr (computeKvSegment1 && computeKvSegment2) {
                Or((RegTensor<uint8_t>&)pB8Vreg, (RegTensor<uint8_t>&)pB8Vreg, (RegTensor<uint8_t>&)pPack2Vreg,
                   allB8Mask);
            }
            Gather(packedPVreg, pB8Vreg, gatherIndexVreg);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                pStoreUbAddr, packedPVreg, pStripeBlockStride, 1, allB8Mask);
        }
        // Merge local sums, then update the online sum.
        Add(ll0Vreg, ll0Vreg, ll2Vreg, allB32Mask);
        Add(ll1Vreg, ll1Vreg, ll3Vreg, allB32Mask);
        Add(ll0Vreg, ll0Vreg, ll1Vreg, allB32Mask);
        if constexpr (IsUpdate) {
            Add(ll0Vreg, rescaledGlVreg, ll0Vreg, validRowB32Mask);
        }
        StoreAlign(glUbAddr, ll0Vreg, allB32Mask);
    }

    template <bool IsUpdate, bool IsFullKvTile, KvRange ValidRange = KvRange::FULL>
    __simd_vf__ static inline void ComputeMaskedOnlineSoftmaxImpl(
        __ubuf__ float* sUbAddr, __ubuf__ ElementOutput* pUbAddr, __ubuf__ float* gmUbAddr, __ubuf__ float* glUbAddr,
        __ubuf__ float* dmUbAddr, __ubuf__ uint8_t* pGatherIndexUbAddr, uint32_t rowNumCurSubCore,
        uint32_t rowOffsetCurSubCore, uint32_t qSBlockSize, int32_t qTileCenterStart, uint32_t kvSTileStart,
        uint32_t kvSTileSize, float scaleValue, float softcapValue, int32_t windowSizeLeft, int32_t windowSizeRight)
    {
        using namespace AscendC::Reg;
        constexpr uint32_t kvSegmentSize = KV_QUARTER_SIZE;
        constexpr uint32_t pStripeBlockStride = P_STRIPE_BLOCK_STRIDE;
        constexpr uint32_t kvBegin = KvRangeInfo<ValidRange>::BEGIN;
        constexpr uint32_t kvEnd = KvRangeInfo<ValidRange>::END;
        constexpr bool computeKvSegment0 = kvBegin <= 0U && kvEnd > 0U;
        constexpr bool computeKvSegment1 = kvBegin <= KV_QUARTER_SIZE && kvEnd > KV_QUARTER_SIZE;
        constexpr bool computeKvSegment2 = kvBegin <= KV_HALF_SIZE && kvEnd > KV_HALF_SIZE;
        constexpr bool computeKvSegment3 = kvBegin <= KV_THREE_QUARTER_SIZE && kvEnd > KV_THREE_QUARTER_SIZE;
        static constexpr CastTrait castTraitZeroRound = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait castTraitOneRound = {RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait castTraitTwoRound = {RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait castTraitThreeRound = {RegLayout::THREE, SatMode::SAT, MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};

        MaskReg allB32Mask = CreateMask<float, MaskPattern::ALL>();
        MaskReg validRowB32Mask = UpdateMask<float>(rowNumCurSubCore);
        MaskReg allB8Mask = CreateMask<uint8_t, MaskPattern::ALL>();
        RegTensor<float> s0Vreg, s1Vreg, s2Vreg, s3Vreg;
        RegTensor<float> lmVreg;
        RegTensor<float> p0Vreg, p1Vreg, p2Vreg, p3Vreg;
        RegTensor<float> ll0Vreg, ll1Vreg, ll2Vreg, ll3Vreg;
        RegTensor<float> gmVreg, glVreg, dmVreg, rescaledGlVreg;
        RegTensor<int32_t> qIndexVreg;
        RegTensor<ElementOutput> pB8Vreg, packedPVreg;
        RegTensor<uint8_t> gatherIndexVreg;

        // Map each fused GQA row to its Q position before applying the attention mask.
        Arange(qIndexVreg, static_cast<int32_t>(rowOffsetCurSubCore));
        RegTensor<int32_t> divisorVreg, headVreg, productVreg;
        Duplicate(divisorVreg, static_cast<int32_t>(qSBlockSize));
        Div(headVreg, qIndexVreg, divisorVreg, allB32Mask);
        Mul(productVreg, headVreg, divisorVreg, allB32Mask);
        Sub(qIndexVreg, qIndexVreg, productVreg, allB32Mask);
        Adds(qIndexVreg, qIndexVreg, qTileCenterStart, allB32Mask);
        if constexpr (maskCategory == CAUSAL_MASK) {
            RegTensor<int32_t> kvEndVreg;
            Duplicate(kvEndVreg, static_cast<int32_t>(kvSTileStart + kvSTileSize) - 1);
            Min(qIndexVreg, qIndexVreg, kvEndVreg, allB32Mask);
        }
        // Max pass: scale, apply softcap/mask, and cache prepared S.
        RegTensor<float> lm0Vreg, lm1Vreg, lm2Vreg, lm3Vreg;
        Duplicate(lm0Vreg, MIN_VALUE);
        Duplicate(lm1Vreg, MIN_VALUE);
        Duplicate(lm2Vreg, MIN_VALUE);
        Duplicate(lm3Vreg, MIN_VALUE);
        constexpr uint32_t kvCount = kvEnd - kvBegin;
        for (uint32_t i = 0; i < kvCount / FOUR_ISSUE_NUM; ++i) {
            LoadAlign(s0Vreg, sUbAddr + (kvBegin + i * FOUR_ISSUE_NUM) * Q_BASE_TILE_SIZE_PER_SUBCORE);
            Muls(s0Vreg, s0Vreg, scaleValue, validRowB32Mask);
            if constexpr (HAS_SOFTCAP) {
                ApplySoftcap(s0Vreg, softcapValue, validRowB32Mask);
            }
            ApplyAttentionMask<IsFullKvTile>(s0Vreg, qIndexVreg, kvBegin + i * FOUR_ISSUE_NUM, kvSTileStart,
                                             kvSTileSize, windowSizeLeft, windowSizeRight, validRowB32Mask);
            LoadAlign(s1Vreg, sUbAddr + (kvBegin + i * FOUR_ISSUE_NUM + 1) * Q_BASE_TILE_SIZE_PER_SUBCORE);
            Muls(s1Vreg, s1Vreg, scaleValue, validRowB32Mask);
            if constexpr (HAS_SOFTCAP) {
                ApplySoftcap(s1Vreg, softcapValue, validRowB32Mask);
            }
            ApplyAttentionMask<IsFullKvTile>(s1Vreg, qIndexVreg, kvBegin + i * FOUR_ISSUE_NUM + 1, kvSTileStart,
                                             kvSTileSize, windowSizeLeft, windowSizeRight, validRowB32Mask);
            LoadAlign(s2Vreg, sUbAddr + (kvBegin + i * FOUR_ISSUE_NUM + 2) * Q_BASE_TILE_SIZE_PER_SUBCORE);
            Muls(s2Vreg, s2Vreg, scaleValue, validRowB32Mask);
            if constexpr (HAS_SOFTCAP) {
                ApplySoftcap(s2Vreg, softcapValue, validRowB32Mask);
            }
            ApplyAttentionMask<IsFullKvTile>(s2Vreg, qIndexVreg, kvBegin + i * FOUR_ISSUE_NUM + 2, kvSTileStart,
                                             kvSTileSize, windowSizeLeft, windowSizeRight, validRowB32Mask);
            LoadAlign(s3Vreg, sUbAddr + (kvBegin + i * FOUR_ISSUE_NUM + 3) * Q_BASE_TILE_SIZE_PER_SUBCORE);
            Muls(s3Vreg, s3Vreg, scaleValue, validRowB32Mask);
            if constexpr (HAS_SOFTCAP) {
                ApplySoftcap(s3Vreg, softcapValue, validRowB32Mask);
            }
            ApplyAttentionMask<IsFullKvTile>(s3Vreg, qIndexVreg, kvBegin + i * FOUR_ISSUE_NUM + 3, kvSTileStart,
                                             kvSTileSize, windowSizeLeft, windowSizeRight, validRowB32Mask);
            StoreAlign(sUbAddr + (kvBegin + i * FOUR_ISSUE_NUM) * Q_BASE_TILE_SIZE_PER_SUBCORE, s0Vreg,
                       validRowB32Mask);
            StoreAlign(sUbAddr + (kvBegin + i * FOUR_ISSUE_NUM + 1) * Q_BASE_TILE_SIZE_PER_SUBCORE, s1Vreg,
                       validRowB32Mask);
            StoreAlign(sUbAddr + (kvBegin + i * FOUR_ISSUE_NUM + 2) * Q_BASE_TILE_SIZE_PER_SUBCORE, s2Vreg,
                       validRowB32Mask);
            StoreAlign(sUbAddr + (kvBegin + i * FOUR_ISSUE_NUM + 3) * Q_BASE_TILE_SIZE_PER_SUBCORE, s3Vreg,
                       validRowB32Mask);

            Max(lm0Vreg, lm0Vreg, s0Vreg, validRowB32Mask);
            Max(lm1Vreg, lm1Vreg, s1Vreg, validRowB32Mask);
            Max(lm2Vreg, lm2Vreg, s2Vreg, validRowB32Mask);
            Max(lm3Vreg, lm3Vreg, s3Vreg, validRowB32Mask);
        }
        Max(lm0Vreg, lm0Vreg, lm2Vreg, validRowB32Mask);
        Max(lm1Vreg, lm1Vreg, lm3Vreg, validRowB32Mask);
        Max(lmVreg, lm0Vreg, lm1Vreg, validRowB32Mask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        // Update the online max and correction, reusing the local max register.
        auto& hmVreg = lmVreg;
        if constexpr (IsUpdate) {
            LoadAlign(gmVreg, gmUbAddr);
            LoadAlign(glVreg, glUbAddr);
            Max(hmVreg, lmVreg, gmVreg, validRowB32Mask);
            FusedExpSub(dmVreg, gmVreg, hmVreg, validRowB32Mask);
            Mul(rescaledGlVreg, dmVreg, glVreg, validRowB32Mask);
            StoreAlign(dmUbAddr, dmVreg, allB32Mask);
        }
        StoreAlign(gmUbAddr, hmVreg, allB32Mask);

        // Exp/P pass: read prepared S without repeating scale, softcap, or mask.
        // P and its FP32 sum share P_SCALE; the saved online max remains unshifted.
        Adds(hmVreg, hmVreg, -LN_P_SCALE, validRowB32Mask);
        Duplicate(ll0Vreg, 0.0f);
        Duplicate(ll1Vreg, 0.0f);
        Duplicate(ll2Vreg, 0.0f);
        Duplicate(ll3Vreg, 0.0f);
        LoadAlign(gatherIndexVreg, pGatherIndexUbAddr);
        __ubuf__ ElementOutput* pStoreUbAddr = pUbAddr;

        // Cast four FP32 exp vectors into byte lanes, then Gather into the PV nZ stripes.
        for (uint32_t i = 0; i < kvSegmentSize; ++i) {
            if constexpr (computeKvSegment0) {
                LoadAlign(s0Vreg, sUbAddr + i * Q_BASE_TILE_SIZE_PER_SUBCORE);
            }
            if constexpr (computeKvSegment1) {
                LoadAlign(s1Vreg, sUbAddr + (i + kvSegmentSize * 1) * Q_BASE_TILE_SIZE_PER_SUBCORE);
            }
            if constexpr (computeKvSegment2) {
                LoadAlign(s2Vreg, sUbAddr + (i + kvSegmentSize * 2) * Q_BASE_TILE_SIZE_PER_SUBCORE);
            }
            if constexpr (computeKvSegment3) {
                LoadAlign(s3Vreg, sUbAddr + (i + kvSegmentSize * 3) * Q_BASE_TILE_SIZE_PER_SUBCORE);
            }
            if constexpr (computeKvSegment0) {
                FusedExpSub(p0Vreg, s0Vreg, hmVreg, validRowB32Mask);
            } else {
                Duplicate(p0Vreg, 0.0f);
            }
            if constexpr (computeKvSegment1) {
                FusedExpSub(p1Vreg, s1Vreg, hmVreg, validRowB32Mask);
            } else {
                Duplicate(p1Vreg, 0.0f);
            }
            if constexpr (computeKvSegment2) {
                FusedExpSub(p2Vreg, s2Vreg, hmVreg, validRowB32Mask);
            } else {
                Duplicate(p2Vreg, 0.0f);
            }
            if constexpr (computeKvSegment3) {
                FusedExpSub(p3Vreg, s3Vreg, hmVreg, validRowB32Mask);
            } else {
                Duplicate(p3Vreg, 0.0f);
            }
            if constexpr (computeKvSegment0) {
                Add(ll0Vreg, ll0Vreg, p0Vreg, validRowB32Mask);
            }
            if constexpr (computeKvSegment1) {
                Add(ll1Vreg, ll1Vreg, p1Vreg, validRowB32Mask);
            }
            if constexpr (computeKvSegment2) {
                Add(ll2Vreg, ll2Vreg, p2Vreg, validRowB32Mask);
            }
            if constexpr (computeKvSegment3) {
                Add(ll3Vreg, ll3Vreg, p3Vreg, validRowB32Mask);
            }
            // Reuse FP32 registers after sum and Cast; preserve this alias and Cast order.
            auto& pPack1Vreg = (RegTensor<ElementOutput>&)p0Vreg;
            auto& pPack2Vreg = (RegTensor<ElementOutput>&)p1Vreg;
            auto& pPack3Vreg = (RegTensor<ElementOutput>&)p2Vreg;
            // Cast and pack the four byte lanes before Gather and StoreAlign.
            if constexpr (computeKvSegment0) {
                Cast<ElementOutput, float, castTraitZeroRound>(pB8Vreg, p0Vreg, validRowB32Mask);
            }
            if constexpr (computeKvSegment1) {
                if constexpr (kvBegin == KV_QUARTER_SIZE) {
                    Cast<ElementOutput, float, castTraitOneRound>(pB8Vreg, p1Vreg, validRowB32Mask);
                } else {
                    Cast<ElementOutput, float, castTraitOneRound>(pPack1Vreg, p1Vreg, validRowB32Mask);
                }
            }
            if constexpr (computeKvSegment2) {
                if constexpr (kvBegin == KV_HALF_SIZE) {
                    Cast<ElementOutput, float, castTraitTwoRound>(pB8Vreg, p2Vreg, validRowB32Mask);
                } else {
                    Cast<ElementOutput, float, castTraitTwoRound>(pPack2Vreg, p2Vreg, validRowB32Mask);
                }
            }
            if constexpr (computeKvSegment3) {
                if constexpr (kvBegin == KV_THREE_QUARTER_SIZE) {
                    Cast<ElementOutput, float, castTraitThreeRound>(pB8Vreg, p3Vreg, validRowB32Mask);
                } else {
                    Cast<ElementOutput, float, castTraitThreeRound>(pPack3Vreg, p3Vreg, validRowB32Mask);
                }
            }
            if constexpr (computeKvSegment0 && computeKvSegment1) {
                Or((RegTensor<uint8_t>&)pB8Vreg, (RegTensor<uint8_t>&)pB8Vreg, (RegTensor<uint8_t>&)pPack1Vreg,
                   allB8Mask);
            }
            if constexpr (computeKvSegment1 && computeKvSegment3) {
                Or((RegTensor<uint8_t>&)pPack2Vreg, (RegTensor<uint8_t>&)pPack2Vreg, (RegTensor<uint8_t>&)pPack3Vreg,
                   allB8Mask);
                Or((RegTensor<uint8_t>&)pB8Vreg, (RegTensor<uint8_t>&)pB8Vreg, (RegTensor<uint8_t>&)pPack2Vreg,
                   allB8Mask);
            } else if constexpr (computeKvSegment1 && computeKvSegment2) {
                Or((RegTensor<uint8_t>&)pB8Vreg, (RegTensor<uint8_t>&)pB8Vreg, (RegTensor<uint8_t>&)pPack2Vreg,
                   allB8Mask);
            } else if constexpr (kvBegin == KV_HALF_SIZE && computeKvSegment3) {
                Or((RegTensor<uint8_t>&)pB8Vreg, (RegTensor<uint8_t>&)pB8Vreg, (RegTensor<uint8_t>&)pPack3Vreg,
                   allB8Mask);
            }
            Gather(packedPVreg, pB8Vreg, gatherIndexVreg);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                pStoreUbAddr, packedPVreg, pStripeBlockStride, 1, allB8Mask);
        }
        // Merge local sums, then update the online sum.
        Add(ll0Vreg, ll0Vreg, ll2Vreg, allB32Mask);
        Add(ll1Vreg, ll1Vreg, ll3Vreg, allB32Mask);
        Add(ll0Vreg, ll0Vreg, ll1Vreg, allB32Mask);
        if constexpr (IsUpdate) {
            Add(ll0Vreg, rescaledGlVreg, ll0Vreg, validRowB32Mask);
        }
        StoreAlign(glUbAddr, ll0Vreg, allB32Mask);
    }
};
} // namespace Catlass::Epilogue::Block
#endif
