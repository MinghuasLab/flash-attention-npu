/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_RESCALE_O_HPP_T
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_RESCALE_O_HPP_T
#include <limits>

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "kernel_common.hpp"

namespace Catlass::Epilogue::Block {

template <
    class ElementO_,
    class ElementOTmp_,
    class ElementS_,
    class TileCopy_,
    class OTmpSrcPos_ // the src TPosition of pv res, viable configurations: GM/L0C
>
class BlockEpilogue<
    EpilogueFARescaleO,
    ElementO_,
    ElementOTmp_,
    ElementS_,
    TileCopy_,
    OTmpSrcPos_>
{
public:
    using DispatchPolicy = EpilogueFARescaleO;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = ElementO_;
    using ElementOTmp = ElementOTmp_;
    using SMDtype = ElementS_;
    using TileCopy = TileCopy_;
    using OTmpSrcPos = OTmpSrcPos_;

    using CopyUbToGmO = typename TileCopy::CopyUbToGmO;

    static constexpr uint32_t UB_OTMP_BUF_STAGES = 2;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64;
    static constexpr uint32_t RESCALE_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t RESCALE_COL_MAX_ELEM_NUM = 128;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t RESCALE_SKIP_SCRATCH_OFFSET = 250 * 1024;

    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource, bool enableRescaleSkip_ = false)
    {
        enableRescaleSkip = enableRescaleSkip_;
        constexpr uint32_t LO_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 7 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = LM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t DM_UB_TENSOR_OFFSET = GM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t LL_UB_TENSOR_OFFSET = DM_UB_TENSOR_OFFSET + 3 * 64 * sizeof(float);
        constexpr uint32_t GL_UB_TENSOR_OFFSET = LL_UB_TENSOR_OFFSET +  64 * sizeof(float);

        for (uint32_t i = 0; i < UB_OTMP_BUF_STAGES; i++) {
            loUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                LO_UB_TENSOR_OFFSET + i * UB_UINT8_BLOCK_SIZE);
        }
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<ElementOTmp>(GO_UB_TENSOR_OFFSET);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementO>(GO_UB_TENSOR_OFFSET);
        glUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        lmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(LM_UB_TENSOR_OFFSET);
        gmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        dmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        redUbTensor = resource.ubBuf.template GetBufferByByte<float>(RESCALE_SKIP_SCRATCH_OFFSET);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }


    // Clear O rows whose absolute Q-token index falls outside the SWA-valid
    // range [delEndRow, delStartRow).  Logical UB rows are S/N combined:
    //   localS = groupRow % qSBlockSize, absQ = qSTileStart + localS
    // so each head replica of the same token is cleared together when qN>1.
    __aicore__ inline
    void ZeroInvalidSwaRows(
        uint32_t rowNumCurSubCore,
        uint32_t embedRound,
        uint32_t rowOffsetCurSubCore,
        uint32_t qSBlockSize,
        int32_t delStartRow,
        int32_t delEndRow,
        uint32_t qSeqlen,
        uint32_t qSTileStart)
    {
        if (qSeqlen == 0U || qSBlockSize == 0U) {
            return;
        }
        if (delStartRow == 0 && delEndRow == static_cast<int32_t>(qSeqlen)) {
            return;
        }
        for (uint32_t i = 0; i < rowNumCurSubCore; ++i) {
            const uint32_t groupRow = rowOffsetCurSubCore + i;
            const uint32_t localS = groupRow % qSBlockSize;
            const int32_t absQ = static_cast<int32_t>(qSTileStart + localS);
            const bool clearTail = (delStartRow != 0) && (absQ >= delStartRow);
            const bool clearHead = (delEndRow != static_cast<int32_t>(qSeqlen)) && (absQ < delEndRow);
            if (clearTail || clearHead) {
                AscendC::Duplicate(
                    goUbTensor16[i * embedRound],
                    static_cast<ElementO>(0),
                    embedRound);
            }
        }
    }

    // Mirror ZeroInvalidSwaRows for LSE: invalid tokens get +inf (host/emptySpan
    // convention), written into the Brcb-expanded LSE scratch (goUbTensor32).
    __aicore__ inline
    void InvalidSwaLseRows(
        uint32_t rowStart,
        uint32_t rowCount,
        uint32_t qSBlockSize,
        int32_t delStartRow,
        int32_t delEndRow,
        uint32_t qSeqlen,
        uint32_t qSTileStart)
    {
        if (qSeqlen == 0U || qSBlockSize == 0U || rowCount == 0U) {
            return;
        }
        if (delStartRow == 0 && delEndRow == static_cast<int32_t>(qSeqlen)) {
            return;
        }
        for (uint32_t i = 0; i < rowCount; ++i) {
            const uint32_t groupRow = rowStart + i;
            const uint32_t localS = groupRow % qSBlockSize;
            const int32_t absQ = static_cast<int32_t>(qSTileStart + localS);
            const bool clearTail = (delStartRow != 0) && (absQ >= delStartRow);
            const bool clearHead = (delEndRow != static_cast<int32_t>(qSeqlen)) && (absQ < delEndRow);
            if (clearTail || clearHead) {
                AscendC::Duplicate(
                    goUbTensor32[i * FLOAT_BLOCK_SIZE],
                    std::numeric_limits<float>::infinity(),
                    FLOAT_BLOCK_SIZE);
            }
        }
    }

    template <uint32_t ColBlocks, bool HeadDimAligned64, class TensorDst>
    __aicore__ inline
    void SubCoreCompute(TensorDst &gOTensorTlaTile,
                        uint32_t colStride,
                        uint32_t curTileMod,
                        uint32_t ubOTmpBufId,
                        bool isFirstKvSTile,
                        bool isLastKvSTile,
                        Arch::CrossCoreFlag pvReadyFlag,
                        uint32_t zeroRowCount,
                        uint32_t tailCols,
                        bool skipOutput,
                        uint32_t rowOffsetCurSubCore,
                        uint32_t qSBlockSize = 1,
                        int32_t delStartRow = 0,
                        int32_t delEndRow = 0,
                        uint32_t qSeqlen = 0,
                        uint32_t qSTileStart = 0,
                        AscendC::GlobalTensor<ElementO> gPartialO = {},
                        AscendC::GlobalTensor<float> gPartialLse = {},
                        bool writePartial = false,
                        uint32_t partialId = 0,
                        uint32_t fdRowCapacity = 0,
                        uint32_t fdLseSubStride = 0)
    {
        uint32_t rowNumCurSubCore = tla::get<0>(gOTensorTlaTile.shape());
        uint32_t colNumCurSubCore = tla::get<1>(gOTensorTlaTile.shape());

        __ubuf__ ElementOTmp *goUb = (__ubuf__ ElementOTmp *) goUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp *loUb = (__ubuf__ ElementOTmp *) loUbTensor[ubOTmpBufId].GetPhyAddr();
        __ubuf__ ElementOTmp *glUb = ( __ubuf__ ElementOTmp *) glUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp *dmUb =
            (__ubuf__ ElementOTmp *) dmUbTensor32[curTileMod * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();
        
        WaitCrossCoreSync<4, PIPE_V>(pvReadyFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        if (isFirstKvSTile) {
            if (!isLastKvSTile) {
                uint32_t totalCopyElems = rowNumCurSubCore * colStride;
                AscendC::DataCopy(goUbTensor32, loUbTensor[ubOTmpBufId], totalCopyElems);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                DivFuncLastAndFirstGeneric<ElementOTmp, ColBlocks, HeadDimAligned64>(
                    goUb, loUb, glUb, rowNumCurSubCore, colNumCurSubCore, colStride, tailCols);
            }
        } else if (!isLastKvSTile) {
            bool skip = enableRescaleSkip &&
                DmAllOne(dmUbTensor32[curTileMod * DM_UB_GLOBAL_ELEM_NUM], rowNumCurSubCore);
            if (skip) {
                AddFunc<ElementOTmp, ColBlocks, HeadDimAligned64>(
                    goUb, loUb, rowNumCurSubCore, colNumCurSubCore, colStride, tailCols);
            } else {
                RescaleFuncGeneric<ElementOTmp, ColBlocks, HeadDimAligned64>(
                    goUb, loUb, dmUb, rowNumCurSubCore, colNumCurSubCore, colStride, tailCols);
            }
        } else {
            bool skip = enableRescaleSkip &&
                DmAllOne(dmUbTensor32[curTileMod * DM_UB_GLOBAL_ELEM_NUM], rowNumCurSubCore);
            if (skip) {
                AddDivFuncLastNotFirst<ElementOTmp, ColBlocks, HeadDimAligned64>(
                    goUb, loUb, glUb, rowNumCurSubCore, colNumCurSubCore, colStride, tailCols);
            } else {
                RescaleFuncLastNotFirstGeneric<ElementOTmp, ColBlocks, HeadDimAligned64>(
                    goUb, loUb, dmUb, glUb, rowNumCurSubCore, colNumCurSubCore, colStride, tailCols);
            }
        }
        if (zeroRowCount > 0) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(
                goUbTensor32,
                static_cast<ElementOTmp>(0),
                zeroRowCount * colStride);
        }
        // release lo buf
        SetCrossCoreSync<4, PIPE_V>(pvReadyFlag);
        if (isLastKvSTile) {
            AscendC::PipeBarrier<PIPE_V>();
            if (std::is_same<ElementO, bfloat16_t>::value) {
                AscendC::Cast(
                    goUbTensor16, goUbTensor32,
                    AscendC::RoundMode::CAST_RINT,
                    rowNumCurSubCore * colStride
                    );
            } else {
                AscendC::Cast(
                    goUbTensor16, goUbTensor32,
                    AscendC::RoundMode::CAST_NONE,
                    rowNumCurSubCore * colStride
                    );
            }
            AscendC::PipeBarrier<PIPE_V>();
            ZeroInvalidSwaRows(
                rowNumCurSubCore, colStride, rowOffsetCurSubCore, qSBlockSize,
                delStartRow, delEndRow, qSeqlen, qSTileStart);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            auto ubOLayoutTla = tla::MakeLayout(
                tla::MakeShape(rowNumCurSubCore, colNumCurSubCore),
                tla::MakeStride(colStride, tla::Int<1>{})
            );
            auto ubOTensorTla = tla::MakeTensor(goUbTensor16, ubOLayoutTla, Arch::PositionUB{});
            if (writePartial) {
                ComputeLse((__ubuf__ float *)glUbTensor32.GetPhyAddr(),
                    (__ubuf__ float *)gmUbTensor32.GetPhyAddr(),
                    (__ubuf__ float *)lmUbTensor32.GetPhyAddr(), rowNumCurSubCore);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                const uint64_t partialORow =
                    (static_cast<uint64_t>(partialId) * fdRowCapacity +
                        rowOffsetCurSubCore) * colNumCurSubCore;
                AscendC::DataCopyPad(
                    gPartialO[partialORow], goUbTensor16,
                    AscendC::DataCopyExtParams(
                        rowNumCurSubCore,
                        static_cast<uint32_t>(colNumCurSubCore * sizeof(ElementO)),
                        0, 0, 0));
                const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
                const uint64_t partialLseRow =
                    (static_cast<uint64_t>(partialId) * 2U + subBlockIdx) *
                    fdLseSubStride;
                AscendC::DataCopyPad(
                    gPartialLse[partialLseRow], lmUbTensor32,
                    AscendC::DataCopyExtParams(
                        1, static_cast<uint32_t>(rowNumCurSubCore * sizeof(float)),
                        0, 0, 0));
            } else if (!skipOutput) {
                copyUbToGmO(gOTensorTlaTile, ubOTensorTla);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }

    template <class TensorDst>
    __aicore__ inline
    void ScatterGroupedOutput(TensorDst &gOTensor, uint32_t qSBlockSize,
                              uint32_t rowStart, uint32_t rowCount,
                              uint32_t embedV, uint32_t outputStride,
                              uint32_t colStride, uint32_t fullyMaskedRowsPerHead)
    {
        auto gO = gOTensor.data();
        // zero fully-masked Q rows in UB (causal Sq>Sk).
        if (fullyMaskedRowsPerHead != 0U) {
            uint32_t groupRow = rowStart;
            uint32_t ubRowOffset = 0U;
            uint32_t remainingRows = rowCount;
            while (remainingRows > 0U) {
                uint32_t localS = groupRow % qSBlockSize;
                uint32_t rowsThisHead = qSBlockSize - localS;
                rowsThisHead = rowsThisHead < remainingRows ? rowsThisHead : remainingRows;
                if (fullyMaskedRowsPerHead > localS) {
                    uint32_t maskedRows = fullyMaskedRowsPerHead - localS;
                    maskedRows = maskedRows < rowsThisHead ? maskedRows : rowsThisHead;
                    AscendC::Duplicate(goUbTensor16[ubRowOffset * colStride],
                        static_cast<ElementO>(0), maskedRows * colStride);
                }
                groupRow += rowsThisHead;
                ubRowOffset += rowsThisHead;
                remainingRows -= rowsThisHead;
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        }
        // scatter O rows to per-head GM addresses.
        uint32_t groupRow = rowStart;
        uint32_t ubRowOffset = 0U;
        uint32_t remainingRows = rowCount;
        while (remainingRows > 0U) {
            uint32_t head = groupRow / qSBlockSize;
            uint32_t localS = groupRow % qSBlockSize;
            uint32_t rowsThisHead = qSBlockSize - localS;
            rowsThisHead = rowsThisHead < remainingRows ? rowsThisHead : remainingRows;
            AscendC::DataCopyPad(
                gO[head * embedV + localS * outputStride],
                goUbTensor16[ubRowOffset * colStride],
                AscendC::DataCopyExtParams(rowsThisHead, embedV * sizeof(ElementO), 0,
                    (outputStride - embedV) * sizeof(ElementO), 0));
            groupRow += rowsThisHead;
            ubRowOffset += rowsThisHead;
            remainingRows -= rowsThisHead;
        }
    }

    __simd_vf__ static inline
    void ComputeLse(__ubuf__ float *glUb, __ubuf__ float *gmUb,
                           __ubuf__ float *lmUb, uint32_t rowCount)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> sumReg;
        RegTensor<float> maxReg;
        RegTensor<float> lseReg;
        MaskReg mask = rowCount == FLOAT_VECTOR_SIZE
            ? CreateMask<float, MaskPattern::ALL>()
            : UpdateMask<float>(rowCount);
        LoadAlign<float, LoadDist::DIST_NORM>(sumReg, glUb);
        LoadAlign<float, LoadDist::DIST_NORM>(maxReg, gmUb);
        AscendC::Reg::Ln<float>(lseReg, sumReg, mask);
        AscendC::Reg::Add<float>(lseReg, lseReg, maxReg, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(lmUb, lseReg, mask);
    }

    __aicore__ inline
    void WriteGroupedLse(AscendC::GlobalTensor<float> gLse,
                         uint32_t qSBlockSize, uint32_t qNBlockSize,
                         uint32_t lseHeadStride, bool isDN,
                         uint32_t fullyMaskedRowsPerHead,
                         int32_t delStartRow = 0,
                         int32_t delEndRow = 0,
                         uint32_t qSeqlen = 0,
                         uint32_t qSTileStart = 0)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t groupRows = qSBlockSize * qNBlockSize;
        uint32_t rowAlign = isDN ? 32U : 8U;
        uint32_t splitRows = (groupRows + rowAlign - 1U) / rowAlign * rowAlign / subBlockNum;
        uint32_t firstSubBlockRows = splitRows < groupRows ? splitRows : groupRows;
        uint32_t rowStart = subBlockIdx == 0U ? 0U : firstSubBlockRows;
        uint32_t rowCount = subBlockIdx == 0U ? firstSubBlockRows :
            (groupRows > splitRows ? groupRows - splitRows : 0U);
        if (rowCount == 0U) {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::PipeBarrier<PIPE_V>();
        ComputeLse((__ubuf__ float *)glUbTensor32.GetPhyAddr(),
            (__ubuf__ float *)gmUbTensor32.GetPhyAddr(),
            (__ubuf__ float *)lmUbTensor32.GetPhyAddr(), rowCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(goUbTensor32.template ReinterpretCast<uint32_t>(),
            lmUbTensor32.template ReinterpretCast<uint32_t>(),
            CeilDiv(rowCount, FLOAT_BLOCK_SIZE), AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        if (fullyMaskedRowsPerHead != 0U) {
            if (qNBlockSize == 1U) {
                uint32_t firstLocalS = rowStart % qSBlockSize;
                if (firstLocalS < fullyMaskedRowsPerHead) {
                    uint32_t maskedRows = fullyMaskedRowsPerHead - firstLocalS;
                    maskedRows = maskedRows < rowCount ? maskedRows : rowCount;
                    AscendC::Duplicate(goUbTensor32, std::numeric_limits<float>::infinity(),
                        maskedRows * FLOAT_BLOCK_SIZE);
                }
            } else {
                uint32_t groupRow = rowStart;
                uint32_t ubRowOffset = 0U;
                uint32_t remainingRows = rowCount;
                while (remainingRows > 0U) {
                    uint32_t localS = groupRow % qSBlockSize;
                    uint32_t rowsThisHead = qSBlockSize - localS;
                    rowsThisHead = rowsThisHead < remainingRows ? rowsThisHead : remainingRows;
                    if (fullyMaskedRowsPerHead > localS) {
                        uint32_t maskedRows = fullyMaskedRowsPerHead - localS;
                        maskedRows = maskedRows < rowsThisHead ? maskedRows : rowsThisHead;
                        AscendC::Duplicate(goUbTensor32[ubRowOffset * FLOAT_BLOCK_SIZE],
                            std::numeric_limits<float>::infinity(), maskedRows * FLOAT_BLOCK_SIZE);
                    }
                    groupRow += rowsThisHead;
                    ubRowOffset += rowsThisHead;
                    remainingRows -= rowsThisHead;
                }
            }
        }
        InvalidSwaLseRows(rowStart, rowCount, qSBlockSize,
            delStartRow, delEndRow, qSeqlen, qSTileStart);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
        if (qNBlockSize == 1U) {
            AscendC::DataCopyPad(gLse[rowStart], goUbTensor32,
                AscendC::DataCopyExtParams(rowCount, sizeof(float), 0, 0, 0));
        } else {
            uint32_t groupRow = rowStart;
            uint32_t ubRowOffset = 0U;
            uint32_t remainingRows = rowCount;
            while (remainingRows > 0U) {
                uint32_t head = groupRow / qSBlockSize;
                uint32_t localS = groupRow % qSBlockSize;
                uint32_t rowsThisHead = qSBlockSize - localS;
                rowsThisHead = rowsThisHead < remainingRows ? rowsThisHead : remainingRows;
                AscendC::DataCopyPad(gLse[head * lseHeadStride + localS],
                    goUbTensor32[ubRowOffset * FLOAT_BLOCK_SIZE],
                    AscendC::DataCopyExtParams(rowsThisHead, sizeof(float), 0, 0, 0));
                groupRow += rowsThisHead;
                ubRowOffset += rowsThisHead;
                remainingRows -= rowsThisHead;
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }
    
    template <typename T, uint32_t ColBlocks, bool HeadDimAligned64 = true>
    __simd_vf__ static inline void RescaleFuncGeneric(__ubuf__ T *goUb, __ubuf__ T *loUb, __ubuf__ T *dmUb,
                                               uint32_t row, uint32_t col, uint32_t rowStride,
                                               uint32_t tailCols)
    {
        using namespace AscendC::MicroAPI;
        const uint32_t VL_ELEM_NUM = 64;
        RegTensor<float> dm, go0, go1, go2, go3, lo0, lo1, lo2, lo3;
        RegTensor<float> mul0, mul1, mul2, mul3, out0, out1, out2, out3;
        MaskReg p0 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p1 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p2 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p3 = CreateMask<float, MaskPattern::ALL>();
        if constexpr (ColBlocks == 64) {
            p0 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 128) {
            p1 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 192) {
            p2 = UpdateMask<float>(tailCols);
        } else {
            p3 = UpdateMask<float>(tailCols);
        }

        for (uint32_t i = 0; i < row; ++i) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(dm, dmUb + i);
            __ubuf__ T *go;
            __ubuf__ T *lo;
            if constexpr (HeadDimAligned64) {
                go = goUb + i * ColBlocks;
                lo = loUb + i * ColBlocks;
            } else {
                go = goUb + i * rowStride;
                lo = loUb + i * rowStride;
            }
            LoadAlign<T, LoadDist::DIST_NORM>(go0, go);
            LoadAlign<T, LoadDist::DIST_NORM>(lo0, lo);
            if constexpr (ColBlocks >= 128) {
                LoadAlign<T, LoadDist::DIST_NORM>(go1, go + VL_ELEM_NUM);
                LoadAlign<T, LoadDist::DIST_NORM>(lo1, lo + VL_ELEM_NUM);
            }
            if constexpr (ColBlocks >= 192) {
                LoadAlign<T, LoadDist::DIST_NORM>(go2, go + VL_ELEM_NUM * 2);
                LoadAlign<T, LoadDist::DIST_NORM>(lo2, lo + VL_ELEM_NUM * 2);
            }
            if constexpr (ColBlocks >= 256) {
                LoadAlign<T, LoadDist::DIST_NORM>(go3, go + VL_ELEM_NUM * 3);
                LoadAlign<T, LoadDist::DIST_NORM>(lo3, lo + VL_ELEM_NUM * 3);
            }

            Mul(mul0, go0, dm, p0);
            if constexpr (ColBlocks >= 128) {
                Mul(mul1, go1, dm, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Mul(mul2, go2, dm, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Mul(mul3, go3, dm, p3);
            }
            Add(out0, mul0, lo0, p0);
            if constexpr (ColBlocks >= 128) {
                Add(out1, mul1, lo1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Add(out2, mul2, lo2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Add(out3, mul3, lo3, p3);
            }

            StoreAlign<T, StoreDist::DIST_NORM_B32>(go, out0, p0);
            if constexpr (ColBlocks >= 128) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM, out1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 2, out2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 3, out3, p3);
            }
        }
    }

    template <typename T, uint32_t ColBlocks, bool HeadDimAligned64 = true>
    __simd_vf__ static inline void RescaleFuncLastNotFirstGeneric(__ubuf__ T *goUb, __ubuf__ T *loUb,
                                                           __ubuf__ T *dmUb, __ubuf__ T *glUb,
                                                           uint32_t row, uint32_t col, uint32_t rowStride,
                                                           uint32_t tailCols)
    {
        using namespace AscendC::MicroAPI;
        const uint32_t VL_ELEM_NUM = 64;
        RegTensor<float> dm, gl, go0, go1, go2, go3, lo0, lo1, lo2, lo3;
        RegTensor<float> mul0, mul1, mul2, mul3, sum0, sum1, sum2, sum3;
        RegTensor<float> out0, out1, out2, out3;
        MaskReg p0 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p1 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p2 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p3 = CreateMask<float, MaskPattern::ALL>();
        if constexpr (ColBlocks == 64) {
            p0 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 128) {
            p1 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 192) {
            p2 = UpdateMask<float>(tailCols);
        } else {
            p3 = UpdateMask<float>(tailCols);
        }

        for (uint32_t i = 0; i < row; ++i) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(dm, dmUb + i);
            LoadAlign<T, LoadDist::DIST_BRC_B32>(gl, glUb + i);
            __ubuf__ T *go;
            __ubuf__ T *lo;
            if constexpr (HeadDimAligned64) {
                go = goUb + i * ColBlocks;
                lo = loUb + i * ColBlocks;
            } else {
                go = goUb + i * rowStride;
                lo = loUb + i * rowStride;
            }
            LoadAlign<T, LoadDist::DIST_NORM>(go0, go);
            LoadAlign<T, LoadDist::DIST_NORM>(lo0, lo);
            if constexpr (ColBlocks >= 128) {
                LoadAlign<T, LoadDist::DIST_NORM>(go1, go + VL_ELEM_NUM);
                LoadAlign<T, LoadDist::DIST_NORM>(lo1, lo + VL_ELEM_NUM);
            }
            if constexpr (ColBlocks >= 192) {
                LoadAlign<T, LoadDist::DIST_NORM>(go2, go + VL_ELEM_NUM * 2);
                LoadAlign<T, LoadDist::DIST_NORM>(lo2, lo + VL_ELEM_NUM * 2);
            }
            if constexpr (ColBlocks >= 256) {
                LoadAlign<T, LoadDist::DIST_NORM>(go3, go + VL_ELEM_NUM * 3);
                LoadAlign<T, LoadDist::DIST_NORM>(lo3, lo + VL_ELEM_NUM * 3);
            }

            Mul(mul0, go0, dm, p0);
            if constexpr (ColBlocks >= 128) {
                Mul(mul1, go1, dm, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Mul(mul2, go2, dm, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Mul(mul3, go3, dm, p3);
            }
            Add(sum0, mul0, lo0, p0);
            if constexpr (ColBlocks >= 128) {
                Add(sum1, mul1, lo1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Add(sum2, mul2, lo2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Add(sum3, mul3, lo3, p3);
            }
            Div(out0, sum0, gl, p0);
            if constexpr (ColBlocks >= 128) {
                Div(out1, sum1, gl, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Div(out2, sum2, gl, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Div(out3, sum3, gl, p3);
            }

            StoreAlign<T, StoreDist::DIST_NORM_B32>(go, out0, p0);
            if constexpr (ColBlocks >= 128) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM, out1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 2, out2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 3, out3, p3);
            }
        }
    }

    template <typename T, uint32_t ColBlocks, bool HeadDimAligned64 = true>
    __simd_vf__ static inline void DivFuncLastAndFirstGeneric(__ubuf__ T *goUb, __ubuf__ T *loUb,
                                                       __ubuf__ T *glUb, uint32_t row, uint32_t col,
                                                       uint32_t rowStride,
                                                       uint32_t tailCols)
    {
        using namespace AscendC::MicroAPI;
        const uint32_t VL_ELEM_NUM = 64;
        RegTensor<float> gl, in0, in1, in2, in3, out0, out1, out2, out3;
        MaskReg p0 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p1 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p2 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p3 = CreateMask<float, MaskPattern::ALL>();
        if constexpr (ColBlocks == 64) {
            p0 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 128) {
            p1 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 192) {
            p2 = UpdateMask<float>(tailCols);
        } else {
            p3 = UpdateMask<float>(tailCols);
        }

        for (uint32_t i = 0; i < row; ++i) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(gl, glUb + i);
            __ubuf__ T *go;
            __ubuf__ T *lo;
            if constexpr (HeadDimAligned64) {
                go = goUb + i * ColBlocks;
                lo = loUb + i * ColBlocks;
            } else {
                go = goUb + i * rowStride;
                lo = loUb + i * rowStride;
            }
            LoadAlign<T, LoadDist::DIST_NORM>(in0, lo);
            if constexpr (ColBlocks >= 128) {
                LoadAlign<T, LoadDist::DIST_NORM>(in1, lo + VL_ELEM_NUM);
            }
            if constexpr (ColBlocks >= 192) {
                LoadAlign<T, LoadDist::DIST_NORM>(in2, lo + VL_ELEM_NUM * 2);
            }
            if constexpr (ColBlocks >= 256) {
                LoadAlign<T, LoadDist::DIST_NORM>(in3, lo + VL_ELEM_NUM * 3);
            }

            Div(out0, in0, gl, p0);
            if constexpr (ColBlocks >= 128) {
                Div(out1, in1, gl, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Div(out2, in2, gl, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Div(out3, in3, gl, p3);
            }

            StoreAlign<T, StoreDist::DIST_NORM_B32>(go, out0, p0);
            if constexpr (ColBlocks >= 128) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM, out1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 2, out2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 3, out3, p3);
            }
        }
    }

    __aicore__ inline
    void SetMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % FLOAT_VECTOR_SIZE;
        for (int64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }
        if (len == VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    __aicore__ inline
    bool DmAllOne(AscendC::LocalTensor<float> dm, uint32_t row)
    {
        SetMask((int32_t)row);
        AscendC::WholeReduceMin<float, false>(
            redUbTensor, dm, (int32_t)0, 1, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
        float minDm = redUbTensor.GetValue(0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID7);
        // Restore the full vector mask so later legacy-API vector ops are unaffected.
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        return minDm >= 1.0f;
    }

    // Skip variant of RescaleFunc: dm == 1 for all rows, so go = go + lo.
    template <typename T, uint32_t ColBlocks, bool HeadDimAligned64 = true>
    __simd_vf__ static inline void AddFunc(__ubuf__ T *goUb, __ubuf__ T *loUb,
                                    uint32_t row, uint32_t col, uint32_t rowStride,
                                    uint32_t tailCols)
    {
        using namespace AscendC::MicroAPI;
        const uint32_t VL_ELEM_NUM = 64;
        RegTensor<float> go0, go1, go2, go3, lo0, lo1, lo2, lo3;
        RegTensor<float> out0, out1, out2, out3;
        MaskReg p0 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p1 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p2 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p3 = CreateMask<float, MaskPattern::ALL>();
        if constexpr (ColBlocks == 64) {
            p0 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 128) {
            p1 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 192) {
            p2 = UpdateMask<float>(tailCols);
        } else {
            p3 = UpdateMask<float>(tailCols);
        }

        for (uint32_t i = 0; i < row; ++i) {
            __ubuf__ T *go;
            __ubuf__ T *lo;
            if constexpr (HeadDimAligned64) {
                go = goUb + i * ColBlocks;
                lo = loUb + i * ColBlocks;
            } else {
                go = goUb + i * rowStride;
                lo = loUb + i * rowStride;
            }
            LoadAlign<T, LoadDist::DIST_NORM>(go0, go);
            LoadAlign<T, LoadDist::DIST_NORM>(lo0, lo);
            if constexpr (ColBlocks >= 128) {
                LoadAlign<T, LoadDist::DIST_NORM>(go1, go + VL_ELEM_NUM);
                LoadAlign<T, LoadDist::DIST_NORM>(lo1, lo + VL_ELEM_NUM);
            }
            if constexpr (ColBlocks >= 192) {
                LoadAlign<T, LoadDist::DIST_NORM>(go2, go + VL_ELEM_NUM * 2);
                LoadAlign<T, LoadDist::DIST_NORM>(lo2, lo + VL_ELEM_NUM * 2);
            }
            if constexpr (ColBlocks >= 256) {
                LoadAlign<T, LoadDist::DIST_NORM>(go3, go + VL_ELEM_NUM * 3);
                LoadAlign<T, LoadDist::DIST_NORM>(lo3, lo + VL_ELEM_NUM * 3);
            }

            Add(out0, go0, lo0, p0);
            if constexpr (ColBlocks >= 128) {
                Add(out1, go1, lo1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Add(out2, go2, lo2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Add(out3, go3, lo3, p3);
            }

            StoreAlign<T, StoreDist::DIST_NORM_B32>(go, out0, p0);
            if constexpr (ColBlocks >= 128) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM, out1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 2, out2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 3, out3, p3);
            }
        }
    }

    // Skip variant of RescaleFuncLastNotFirst: go = (go + lo) / gl.
    template <typename T, uint32_t ColBlocks, bool HeadDimAligned64 = true>
    __simd_vf__ static inline void AddDivFuncLastNotFirst(__ubuf__ T *goUb, __ubuf__ T *loUb,
                                                   __ubuf__ T *glUb, uint32_t row,
                                                   uint32_t col, uint32_t rowStride,
                                                   uint32_t tailCols)
    {
        using namespace AscendC::MicroAPI;
        const uint32_t VL_ELEM_NUM = 64;
        RegTensor<float> gl, go0, go1, go2, go3, lo0, lo1, lo2, lo3;
        RegTensor<float> sum0, sum1, sum2, sum3, out0, out1, out2, out3;
        MaskReg p0 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p1 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p2 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p3 = CreateMask<float, MaskPattern::ALL>();
        if constexpr (ColBlocks == 64) {
            p0 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 128) {
            p1 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 192) {
            p2 = UpdateMask<float>(tailCols);
        } else {
            p3 = UpdateMask<float>(tailCols);
        }

        for (uint32_t i = 0; i < row; ++i) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(gl, glUb + i);
            __ubuf__ T *go;
            __ubuf__ T *lo;
            if constexpr (HeadDimAligned64) {
                go = goUb + i * ColBlocks;
                lo = loUb + i * ColBlocks;
            } else {
                go = goUb + i * rowStride;
                lo = loUb + i * rowStride;
            }
            LoadAlign<T, LoadDist::DIST_NORM>(go0, go);
            LoadAlign<T, LoadDist::DIST_NORM>(lo0, lo);
            if constexpr (ColBlocks >= 128) {
                LoadAlign<T, LoadDist::DIST_NORM>(go1, go + VL_ELEM_NUM);
                LoadAlign<T, LoadDist::DIST_NORM>(lo1, lo + VL_ELEM_NUM);
            }
            if constexpr (ColBlocks >= 192) {
                LoadAlign<T, LoadDist::DIST_NORM>(go2, go + VL_ELEM_NUM * 2);
                LoadAlign<T, LoadDist::DIST_NORM>(lo2, lo + VL_ELEM_NUM * 2);
            }
            if constexpr (ColBlocks >= 256) {
                LoadAlign<T, LoadDist::DIST_NORM>(go3, go + VL_ELEM_NUM * 3);
                LoadAlign<T, LoadDist::DIST_NORM>(lo3, lo + VL_ELEM_NUM * 3);
            }

            Add(sum0, go0, lo0, p0);
            if constexpr (ColBlocks >= 128) {
                Add(sum1, go1, lo1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Add(sum2, go2, lo2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Add(sum3, go3, lo3, p3);
            }

            Div(out0, sum0, gl, p0);
            if constexpr (ColBlocks >= 128) {
                Div(out1, sum1, gl, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Div(out2, sum2, gl, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Div(out3, sum3, gl, p3);
            }

            StoreAlign<T, StoreDist::DIST_NORM_B32>(go, out0, p0);
            if constexpr (ColBlocks >= 128) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM, out1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 2, out2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 3, out3, p3);
            }
        }
    }

    template <bool LseMode, class TensorDst>
    __aicore__ inline
    void operator()(TensorDst &gOTensor,
                    AscendC::GlobalTensor<float> gLse,
                    GemmCoord actualOriShape,
                    uint32_t curTileMod,
                    uint32_t gatheredKvSTileIdx,
                    bool isFirstKvSTile,
                    bool isLastKvSTile,
                    Arch::CrossCoreFlag pvReadyFlag,
                    bool isDN,
                    uint32_t fullyMaskedRowsPerHead,
                    uint32_t qSBlockSize,
                    uint32_t qNBlockSize,
                    uint32_t lseHeadStride,
                    uint32_t outputStride,
                    int32_t delStartRow = 0,
                    int32_t delEndRow = 0,
                    uint32_t qSeqlen = 0,
                    uint32_t qSTileStart = 0,
                    AscendC::GlobalTensor<ElementO> gPartialO = {},
                    AscendC::GlobalTensor<float> gPartialLse = {},
                    bool writePartial = false,
                    uint32_t partialId = 0,
                    uint32_t fdRowCapacity = 0,
                    uint32_t fdLseSubStride = 0)
    {
        uint32_t rowNumOri = actualOriShape[0];
        uint32_t colNumOri = actualOriShape[1];
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t colNumOriAligned16 = RoundUp(colNumOri, 16);
        uint32_t groupRows = qSBlockSize * qNBlockSize;
        uint32_t rowAlign = isDN ? 32U : 8U;
        uint32_t splitRows = (groupRows + rowAlign - 1U) / rowAlign * rowAlign / subBlockNum;
        uint32_t firstSubBlockRows = splitRows < groupRows ? splitRows : groupRows;
        uint32_t rowOffsetCurSubCore = subBlockIdx == 0U ? 0U : firstSubBlockRows;
        uint32_t rowNumCurSubCore = subBlockIdx == 0U ? firstSubBlockRows :
            (groupRows > splitRows ? groupRows - splitRows : 0U);
        uint32_t colNumCurSubCore = colNumOri;
        uint32_t colStrideCurSubCore = colNumOriAligned16;
        uint32_t zeroRowCount = 0;
        if (qNBlockSize == 1U && rowOffsetCurSubCore < fullyMaskedRowsPerHead) {
            uint32_t remainingRows = fullyMaskedRowsPerHead - rowOffsetCurSubCore;
            zeroRowCount = remainingRows < rowNumCurSubCore ? remainingRows : rowNumCurSubCore;
        }

        // Grouped SN rows belong to different heads in GM, so only use this
        // tile for its shape and scatter the final O rows explicitly.
        auto gOTensorTlaTile = GetTile(gOTensor,
            tla::MakeCoord(qNBlockSize == 1U ? rowOffsetCurSubCore : 0U, 0),
            tla::MakeShape(rowNumCurSubCore, colNumCurSubCore));
        uint32_t ubOTmpBufId = gatheredKvSTileIdx % UB_OTMP_BUF_STAGES;
        uint32_t vlElemNum = AscendC::GetVecLen() / sizeof(ElementOTmp);
        bool headdimAligned64 = (colNumOri % 64 == 0);
        if (rowNumCurSubCore > 0) {
            if (colNumCurSubCore <= vlElemNum) {
                if (headdimAligned64) {
                    SubCoreCompute<64, true>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore, qNBlockSize > 1U, rowOffsetCurSubCore, qSBlockSize, delStartRow, delEndRow, qSeqlen, qSTileStart, gPartialO, gPartialLse, writePartial, partialId, fdRowCapacity, fdLseSubStride);
                } else {
                    SubCoreCompute<64, false>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore, qNBlockSize > 1U, rowOffsetCurSubCore, qSBlockSize, delStartRow, delEndRow, qSeqlen, qSTileStart, gPartialO, gPartialLse, writePartial, partialId, fdRowCapacity, fdLseSubStride);
                }
            } else if (colNumCurSubCore <= 2 * vlElemNum) {
                if (headdimAligned64) {
                    SubCoreCompute<128, true>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - vlElemNum, qNBlockSize > 1U, rowOffsetCurSubCore, qSBlockSize, delStartRow, delEndRow, qSeqlen, qSTileStart, gPartialO, gPartialLse, writePartial, partialId, fdRowCapacity, fdLseSubStride);
                } else {
                    SubCoreCompute<128, false>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - vlElemNum, qNBlockSize > 1U, rowOffsetCurSubCore, qSBlockSize, delStartRow, delEndRow, qSeqlen, qSTileStart, gPartialO, gPartialLse, writePartial, partialId, fdRowCapacity, fdLseSubStride);
                }
            } else if (colNumCurSubCore <= 3 * vlElemNum) {
                if (headdimAligned64) {
                    SubCoreCompute<192, true>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - 2 * vlElemNum, qNBlockSize > 1U, rowOffsetCurSubCore, qSBlockSize, delStartRow, delEndRow, qSeqlen, qSTileStart, gPartialO, gPartialLse, writePartial, partialId, fdRowCapacity, fdLseSubStride);
                } else {
                    SubCoreCompute<192, false>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - 2 * vlElemNum, qNBlockSize > 1U, rowOffsetCurSubCore, qSBlockSize, delStartRow, delEndRow, qSeqlen, qSTileStart, gPartialO, gPartialLse, writePartial, partialId, fdRowCapacity, fdLseSubStride);
                }
            } else {
                if (headdimAligned64) {
                    SubCoreCompute<256, true>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - 3 * vlElemNum, qNBlockSize > 1U, rowOffsetCurSubCore, qSBlockSize, delStartRow, delEndRow, qSeqlen, qSTileStart, gPartialO, gPartialLse, writePartial, partialId, fdRowCapacity, fdLseSubStride);
                } else {
                    SubCoreCompute<256, false>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - 3 * vlElemNum, qNBlockSize > 1U, rowOffsetCurSubCore, qSBlockSize, delStartRow, delEndRow, qSeqlen, qSTileStart, gPartialO, gPartialLse, writePartial, partialId, fdRowCapacity, fdLseSubStride);
                }
            }
        } else {
            Arch::CrossCoreWaitFlag<4, PIPE_V>(pvReadyFlag);
            Arch::CrossCoreSetFlag<4, PIPE_V>(pvReadyFlag);
        }
        if (!writePartial && qNBlockSize > 1U && rowNumCurSubCore > 0U && isLastKvSTile) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            ScatterGroupedOutput(gOTensor, qSBlockSize, rowOffsetCurSubCore,
                rowNumCurSubCore, colNumOri, outputStride, colStrideCurSubCore,
                fullyMaskedRowsPerHead);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        }
        if constexpr (LseMode) {
            if constexpr (std::is_same_v<SMDtype, float>) {
                if (!writePartial && rowNumCurSubCore > 0U && isLastKvSTile) {
                    WriteGroupedLse(gLse, qSBlockSize, qNBlockSize, lseHeadStride,
                        isDN, fullyMaskedRowsPerHead,
                        delStartRow, delEndRow, qSeqlen, qSTileStart);
                }
            }
        }
    }
private:
    AscendC::LocalTensor<ElementOTmp> loUbTensor[UB_OTMP_BUF_STAGES];
    AscendC::LocalTensor<SMDtype> dmUbTensor16;
    AscendC::LocalTensor<SMDtype> glUbTensor16;
    AscendC::LocalTensor<float> dmUbTensor32;
    AscendC::LocalTensor<float> glUbTensor32;
    AscendC::LocalTensor<float> lmUbTensor32;
    AscendC::LocalTensor<float> gmUbTensor32;
    AscendC::LocalTensor<float> redUbTensor;
    bool enableRescaleSkip{false};
    AscendC::LocalTensor<ElementO> goUbTensor16;
    AscendC::LocalTensor<ElementOTmp> goUbTensor32;

    CopyUbToGmO copyUbToGmO;
};
}
#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_RESCALE_O_LOCAL_HPP
