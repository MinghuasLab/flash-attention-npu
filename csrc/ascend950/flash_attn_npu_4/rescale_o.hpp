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

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include <limits>

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
    // Match arch22: fully-masked / zero-sum rows write -inf (not host +inf).
    static constexpr float LSE_NEG_INF = -std::numeric_limits<float>::infinity();

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
        // LSE UB buffer placed after RESCALE_SKIP_SCRATCH (250KB) + 256B scratch.
        // 252KB keeps a 4KB safety margin to the 256KB UB ceiling.
        constexpr uint32_t LSE_UB_TENSOR_OFFSET = RESCALE_SKIP_SCRATCH_OFFSET + 256;

        for (uint32_t i = 0; i < UB_OTMP_BUF_STAGES; i++) {
            loUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                LO_UB_TENSOR_OFFSET + i * UB_UINT8_BLOCK_SIZE);
        }
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<ElementOTmp>(GO_UB_TENSOR_OFFSET);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementO>(GO_UB_TENSOR_OFFSET);
        glUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        dmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        redUbTensor = resource.ubBuf.template GetBufferByByte<float>(RESCALE_SKIP_SCRATCH_OFFSET);
        lseUbTensor = resource.ubBuf.template GetBufferByByte<float>(LSE_UB_TENSOR_OFFSET);
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

    template < uint32_t VHeadSize, class TensorDst>
    __aicore__ inline
    void SubCoreCompute(TensorDst &gOTensorTlaTile,
                        uint32_t curTileMod,
                        uint32_t ubOTmpBufId,
                        bool isFirstKvSTile,
                        bool isLastKvSTile,
                        Arch::CrossCoreFlag pvReadyFlag,
                        uint32_t rowOffsetCurSubCore,
                        bool lseFlag,
                        AscendC::GlobalTensor<float> *gLse,
                        int64_t lseGmRowOffset,
                        int64_t lseGmRowStride)
    {
        uint32_t rowNumCurSubCore = tla::get<0>(gOTensorTlaTile.shape());
        uint32_t colNumCurSubCore = tla::get<1>(gOTensorTlaTile.shape());
        uint32_t vlElemNum = AscendC::GetVecLen() / sizeof(ElementOTmp);
        uint32_t colFullLoop = CeilDiv(colNumCurSubCore, vlElemNum) - 1;
        uint32_t colTail = (colNumCurSubCore - 1) % vlElemNum + 1;

        __ubuf__ ElementOTmp *goUb = (__ubuf__ ElementOTmp *) goUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp *loUb = (__ubuf__ ElementOTmp *) loUbTensor[ubOTmpBufId].GetPhyAddr();
        __ubuf__ ElementOTmp *glUb = ( __ubuf__ ElementOTmp *) glUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp *dmUb =
            (__ubuf__ ElementOTmp *) dmUbTensor32[curTileMod * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();
        
        WaitCrossCoreSync<4, PIPE_V>(pvReadyFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        if (isFirstKvSTile) {
            if (!isLastKvSTile) {
                uint32_t totalCopyElems = rowNumCurSubCore * VHeadSize;
                AscendC::DataCopy(goUbTensor32, loUbTensor[ubOTmpBufId], totalCopyElems);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                DivFuncLastAndFirst<ElementOTmp, VHeadSize>(
                    goUb, loUb, glUb, rowNumCurSubCore, colFullLoop, colTail, vlElemNum);
            }
        } else if (!isLastKvSTile) {
            bool skip = enableRescaleSkip &&
                DmAllOne(dmUbTensor32[curTileMod * DM_UB_GLOBAL_ELEM_NUM], rowNumCurSubCore);
            if (skip) {
                AddFunc<ElementOTmp, VHeadSize>(
                    goUb, loUb, rowNumCurSubCore, colFullLoop, colTail, vlElemNum);
            } else {
                RescaleFunc<ElementOTmp, VHeadSize>(
                    goUb, loUb, dmUb, rowNumCurSubCore, colFullLoop, colTail, vlElemNum);
            }
        } else {
            bool skip = enableRescaleSkip &&
                DmAllOne(dmUbTensor32[curTileMod * DM_UB_GLOBAL_ELEM_NUM], rowNumCurSubCore);
            if (skip) {
                AddDivFuncLastNotFirst<ElementOTmp, VHeadSize>(
                    goUb, loUb, glUb, rowNumCurSubCore, colFullLoop, colTail, vlElemNum);
            } else {
                RescaleFuncLastNotFirst<ElementOTmp, VHeadSize>(
                    goUb, loUb, dmUb, glUb, rowNumCurSubCore, colFullLoop, colTail, vlElemNum);
            }
        }
        // release lo buf
        SetCrossCoreSync<4, PIPE_V>(pvReadyFlag);
        if (isLastKvSTile) {
            AscendC::PipeBarrier<PIPE_V>();
            // Softmax LSE: lse = ln(gl) + gm, aligned with arch22 v2/v4 rescale_o.
            // GM layout is head-major: BSND [B,H,S] / TND [H,total_q], so tokens of one
            // head are contiguous (lseGmRowStride == 1) and are written as one burst.
            if (lseFlag && rowNumCurSubCore > 0 && gLse != nullptr) {
                uint32_t lseVecLoop = CeilDiv(rowNumCurSubCore, FLOAT_VECTOR_SIZE);
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                AscendC::Ln<float, false>(
                    lseUbTensor, glUbTensor32,
                    (uint64_t)0, lseVecLoop,
                    AscendC::UnaryRepeatParams(1, 1, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::Add<float, false>(
                    lseUbTensor, lseUbTensor, gmUbTensor,
                    (uint64_t)0, lseVecLoop,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));

                // Fully-masked rows: gl == 0 → write -inf (design §8.4 / arch22 InvalidLine
                // equivalent for the zero-sum case; SWA delStart/delEnd not needed yet —
                // arch35 rejects window_size != (-1,-1)).
                FixupZeroSumLse(rowNumCurSubCore);

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
                const int64_t lseGmOffset =
                    lseGmRowOffset + static_cast<int64_t>(rowOffsetCurSubCore) * lseGmRowStride;
                if (lseGmRowStride == 1) {
                    // Match arch22 v2 single-head path: contiguous [H,S] / [H,total_q] burst.
                    AscendC::DataCopyPad(
                        (*gLse)[lseGmOffset],
                        lseUbTensor,
                        AscendC::DataCopyExtParams(
                            1, rowNumCurSubCore * sizeof(float), 0, 0, 0));
                } else {
                    AscendC::DataCopyPad(
                        (*gLse)[lseGmOffset],
                        lseUbTensor,
                        AscendC::DataCopyExtParams(
                            rowNumCurSubCore, sizeof(float), 0, (lseGmRowStride - 1) * sizeof(float), 0));
                }
            }

            if (std::is_same<ElementO, bfloat16_t>::value) {
                AscendC::Cast(
                    goUbTensor16, goUbTensor32,
                    AscendC::RoundMode::CAST_RINT,
                    rowNumCurSubCore * VHeadSize
                    );
            } else {
                AscendC::Cast(
                    goUbTensor16, goUbTensor32,
                    AscendC::RoundMode::CAST_NONE,
                    rowNumCurSubCore * VHeadSize
                    );
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            auto ubOLayoutTla = tla::MakeLayout(
                tla::MakeShape(rowNumCurSubCore, colNumCurSubCore),
                tla::MakeStride(VHeadSize, tla::Int<1>{})
            );
            auto ubOTensorTla = tla::MakeTensor(goUbTensor16, ubOLayoutTla, Arch::PositionUB{});
            copyUbToGmO(gOTensorTlaTile, ubOTensorTla);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }
    
    template <typename T, uint32_t VHeadSize>
    __simd_vf__ inline void RescaleFunc(__ubuf__ T *goUb, __ubuf__ T *loUb, __ubuf__ T *dmUb,
                                        uint32_t row, uint32_t colFullLoop, 
                                        uint32_t colTail, uint32_t vlElemNum)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> dmVreg;
        RegTensor<float> goPreVreg;
        RegTensor<float> goPreRollVreg;
        RegTensor<float> loVreg;
        RegTensor<float> loRollVreg;
        RegTensor<float> mulVreg;
        RegTensor<float> mulRollVreg;
        RegTensor<float> goCurVreg;
        RegTensor<float> goCurRollVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint32_t i = 0; i < row; i++) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(dmVreg, dmUb + i);
            LoadAlign<T, LoadDist::DIST_NORM>(goPreVreg, goUb + i * VHeadSize);
            LoadAlign<T, LoadDist::DIST_NORM>(goPreRollVreg, goUb + i * VHeadSize +vlElemNum);
            LoadAlign<T, LoadDist::DIST_NORM>(loVreg, loUb + i * VHeadSize);
            LoadAlign<T, LoadDist::DIST_NORM>(loRollVreg, loUb + i * VHeadSize + vlElemNum);
            Mul(mulVreg, goPreVreg, dmVreg, pregFull);
            Mul(mulRollVreg, goPreRollVreg, dmVreg, pregTail);
            Add(goCurVreg, mulVreg, loVreg, pregFull);
            Add(goCurRollVreg, mulRollVreg, loRollVreg, pregTail);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + i * VHeadSize, goCurVreg, pregFull);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(
                goUb + i * VHeadSize + vlElemNum, goCurRollVreg, pregTail);
        }
    }

    template <typename T, uint32_t VHeadSize>
    __simd_vf__ inline void RescaleFuncLastNotFirst(__ubuf__ T *goUb, __ubuf__ T *loUb,
                                                    __ubuf__ T *dmUb, __ubuf__ T *glUb,
                                                    uint32_t row, uint32_t colFullLoop, 
                                                    uint32_t colTail, uint32_t vlElemNum)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> dmVreg;
        RegTensor<float> goPreVreg;
        RegTensor<float> goPreRollVreg;
        RegTensor<float> loVreg;
        RegTensor<float> loRollVreg;
        RegTensor<float> mulVreg;
        RegTensor<float> mulRollVreg;
        RegTensor<float> goCurVreg;
        RegTensor<float> goCurRollVreg;
        RegTensor<float> glVreg;
        RegTensor<float> divVreg;
        RegTensor<float> divRollVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint32_t i = 0; i < row; i++) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(dmVreg, dmUb + i);
            LoadAlign<T, LoadDist::DIST_BRC_B32>(glVreg, glUb + i);
            LoadAlign<T, LoadDist::DIST_NORM>(goPreVreg, goUb + i * VHeadSize);
            LoadAlign<T, LoadDist::DIST_NORM>(goPreRollVreg, goUb + i * VHeadSize + vlElemNum);
            LoadAlign<T, LoadDist::DIST_NORM>(loVreg, loUb + i * VHeadSize);
            LoadAlign<T, LoadDist::DIST_NORM>(loRollVreg, loUb + i * VHeadSize + vlElemNum);
            Mul(mulVreg, goPreVreg, dmVreg, pregFull);
            Mul(mulRollVreg, goPreRollVreg, dmVreg, pregTail);
            Add(goCurVreg, mulVreg, loVreg, pregFull);
            Add(goCurRollVreg, mulRollVreg, loRollVreg, pregTail);
            Div(divVreg, goCurVreg, glVreg, pregFull);
            Div(divRollVreg, goCurRollVreg, glVreg, pregTail);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + i * VHeadSize, divVreg, pregFull);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + i * VHeadSize + vlElemNum, divRollVreg, pregTail);
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

    // Replace LSE for rows whose running sum gl is non-positive (fully masked).
    // Matches design §8.4: lse = -inf when s == 0.
    __aicore__ inline
    void FixupZeroSumLse(uint32_t rowNum)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
        for (uint32_t i = 0; i < rowNum; ++i) {
            float s = glUbTensor32.GetValue(i);
            if (s <= 0.0f) {
                lseUbTensor.SetValue(i, LSE_NEG_INF);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID6);
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

    // Skip variant of RescaleFunc: dm == 1 for all rows, so go = go + lo (no dm load, no Mul).
    template <typename T, uint32_t VHeadSize>
    __simd_vf__ inline void AddFunc(__ubuf__ T *goUb, __ubuf__ T *loUb,
                                    uint32_t row, uint32_t colFullLoop,
                                    uint32_t colTail, uint32_t vlElemNum)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> goPreVreg;
        RegTensor<float> goPreRollVreg;
        RegTensor<float> loVreg;
        RegTensor<float> loRollVreg;
        RegTensor<float> goCurVreg;
        RegTensor<float> goCurRollVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint32_t i = 0; i < row; i++) {
            LoadAlign<T, LoadDist::DIST_NORM>(goPreVreg, goUb + i * VHeadSize);
            LoadAlign<T, LoadDist::DIST_NORM>(goPreRollVreg, goUb + i * VHeadSize + vlElemNum);
            LoadAlign<T, LoadDist::DIST_NORM>(loVreg, loUb + i * VHeadSize);
            LoadAlign<T, LoadDist::DIST_NORM>(loRollVreg, loUb + i * VHeadSize + vlElemNum);
            Add(goCurVreg, goPreVreg, loVreg, pregFull);
            Add(goCurRollVreg, goPreRollVreg, loRollVreg, pregTail);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + i * VHeadSize, goCurVreg, pregFull);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(
                goUb + i * VHeadSize + vlElemNum, goCurRollVreg, pregTail);
        }
    }

    // Skip variant of RescaleFuncLastNotFirst: go = (go + lo) / gl (no dm load, no Mul).
    template <typename T, uint32_t VHeadSize>
    __simd_vf__ inline void AddDivFuncLastNotFirst(__ubuf__ T *goUb, __ubuf__ T *loUb,
                                                   __ubuf__ T *glUb,
                                                   uint32_t row, uint32_t colFullLoop,
                                                   uint32_t colTail, uint32_t vlElemNum)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> goPreVreg;
        RegTensor<float> goPreRollVreg;
        RegTensor<float> loVreg;
        RegTensor<float> loRollVreg;
        RegTensor<float> goCurVreg;
        RegTensor<float> goCurRollVreg;
        RegTensor<float> glVreg;
        RegTensor<float> divVreg;
        RegTensor<float> divRollVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint32_t i = 0; i < row; i++) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(glVreg, glUb + i);
            LoadAlign<T, LoadDist::DIST_NORM>(goPreVreg, goUb + i * VHeadSize);
            LoadAlign<T, LoadDist::DIST_NORM>(goPreRollVreg, goUb + i * VHeadSize + vlElemNum);
            LoadAlign<T, LoadDist::DIST_NORM>(loVreg, loUb + i * VHeadSize);
            LoadAlign<T, LoadDist::DIST_NORM>(loRollVreg, loUb + i * VHeadSize + vlElemNum);
            Add(goCurVreg, goPreVreg, loVreg, pregFull);
            Add(goCurRollVreg, goPreRollVreg, loRollVreg, pregTail);
            Div(divVreg, goCurVreg, glVreg, pregFull);
            Div(divRollVreg, goCurRollVreg, glVreg, pregTail);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + i * VHeadSize, divVreg, pregFull);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + i * VHeadSize + vlElemNum, divRollVreg, pregTail);
        }
    }

    template <typename T, uint32_t VHeadSize>
    __simd_vf__ inline void DivFuncLastAndFirst(__ubuf__ T *goUb, __ubuf__ T *loUb, __ubuf__ T *glUb,
                                                uint32_t row, uint32_t colFullLoop, 
                                                uint32_t colTail, uint32_t vlElemNum)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> goCurVreg;
        RegTensor<float> goCurRollVreg;
        RegTensor<float> glVreg;
        RegTensor<float> divVreg;
        RegTensor<float> divRollVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint32_t i = 0; i < row; i++) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(glVreg, glUb + i);
            LoadAlign<T, LoadDist::DIST_NORM>(goCurVreg, loUb + i * VHeadSize);
            LoadAlign<T, LoadDist::DIST_NORM>(goCurRollVreg, loUb + i * VHeadSize + vlElemNum);
            Div(divVreg, goCurVreg, glVreg, pregFull);
            Div(divRollVreg, goCurRollVreg, glVreg, pregFull);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + i * VHeadSize, divVreg, pregFull);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + i * VHeadSize + vlElemNum, divRollVreg, pregFull);
        }
    }

    template <class TensorDst>
    __aicore__ inline
    void operator()(TensorDst &gOTensor,
                    GemmCoord actualOriShape,
                    uint32_t curTileMod,
                    uint32_t gatheredKvSTileIdx,
                    bool isFirstKvSTile,
                    bool isLastKvSTile,
                    Arch::CrossCoreFlag pvReadyFlag,
                    bool isDN,
                    bool lseFlag = false,
                    AscendC::GlobalTensor<float> *gLse = nullptr,
                    int64_t lseGmRowOffset = 0,
                    int64_t lseGmRowStride = 0)
    {
        uint32_t rowNumOri = actualOriShape[0];
        uint32_t colNumOri = actualOriShape[1];
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t rowNumOriAligned = isDN ? RoundUp(rowNumOri, 32) : RoundUp(rowNumOri, 8);
        uint32_t colNumOriAligned8 = RoundUp(colNumOri, 8);

        uint32_t rowNumSplit = rowNumOriAligned / subBlockNum;
        rowNumSplit = (rowNumOri < rowNumSplit) ? rowNumOri : rowNumSplit;
        uint32_t rowNumCurSubCore = (subBlockIdx == 0) ? rowNumSplit : (rowNumOri - rowNumSplit);
        uint32_t rowOffsetCurSubCore = rowNumSplit * subBlockIdx;
        uint32_t colNumCurSubCore = colNumOri;
        uint32_t colStrideCurSubCore = colNumOriAligned8;

        auto gOTensorTlaTile = GetTile(gOTensor,
            tla::MakeCoord(rowOffsetCurSubCore, 0), tla::MakeShape(rowNumCurSubCore, colNumCurSubCore));
        uint32_t ubOTmpBufId = gatheredKvSTileIdx % UB_OTMP_BUF_STAGES;
        if (rowNumCurSubCore > 0) {
            if(colStrideCurSubCore == 128){
                SubCoreCompute<128>(
                    gOTensorTlaTile,
                    curTileMod,
                    ubOTmpBufId,
                    isFirstKvSTile,
                    isLastKvSTile,
                    pvReadyFlag,
                    rowOffsetCurSubCore,
                    lseFlag,
                    gLse,
                    lseGmRowOffset,
                    lseGmRowStride);
            } else if(colStrideCurSubCore == 64){
                SubCoreCompute<64>(
                    gOTensorTlaTile,
                    curTileMod,
                    ubOTmpBufId,
                    isFirstKvSTile,
                    isLastKvSTile,
                    pvReadyFlag,
                    rowOffsetCurSubCore,
                    lseFlag,
                    gLse,
                    lseGmRowOffset,
                    lseGmRowStride);
            }
        } else {
            Arch::CrossCoreWaitFlag<4, PIPE_V>(pvReadyFlag);
            Arch::CrossCoreSetFlag<4, PIPE_V>(pvReadyFlag);
        }
    }
private:
    AscendC::LocalTensor<ElementOTmp> loUbTensor[UB_OTMP_BUF_STAGES];
    AscendC::LocalTensor<SMDtype> dmUbTensor16;
    AscendC::LocalTensor<SMDtype> glUbTensor16;
    AscendC::LocalTensor<float> dmUbTensor32;
    AscendC::LocalTensor<float> glUbTensor32;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> lseUbTensor;
    AscendC::LocalTensor<float> redUbTensor;
    bool enableRescaleSkip{false};
    AscendC::LocalTensor<ElementO> goUbTensor16;
    AscendC::LocalTensor<ElementOTmp> goUbTensor32;

    CopyUbToGmO copyUbToGmO;
};
}
#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_RESCALE_O_LOCAL_HPP