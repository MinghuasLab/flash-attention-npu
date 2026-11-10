/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Modified by Minghua Shen, 2026.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_NO_MASK_HPP_T
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_NO_MASK_HPP_T

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "kernel_common.hpp"
#include "fa_block.h"

namespace Catlass::Epilogue::Block {

template <class OutputType_, class InputType_, class MaskType_, LseModeT LSE_MODE_, bool HAS_SOFTCAP_>
class BlockEpilogue<EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, float, HAS_SOFTCAP_>, OutputType_, InputType_, MaskType_> {
  public:
    struct TaskPlan {
        uint32_t subBlockIdx;
        uint32_t rowActualThisSubBlock;
        uint32_t rowOffsetThisSubBlock;
    };

    using DispatchPolicy = EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, float, HAS_SOFTCAP_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;
    using ElementMask = typename MaskType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutMask = typename MaskType_::Layout;

    static constexpr LseModeT LSE_MODE = DispatchPolicy::LSE_MODE;

    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 8192;

    static constexpr uint32_t DIRECT_REDUCED_OFFSET = 6 * UB_UINT8_VECTOR_SIZE / sizeof(float);
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;
    static constexpr uint32_t TASK_STATE_SLOTS = 3;
    static constexpr uint32_t TASK_GM_UB_TENSOR_OFFSET =
        11 * UB_UINT8_BLOCK_SIZE + 4 * UB_UINT8_VECTOR_SIZE + 2 * REPEAT_SIZE_IN_BYTE;
    static constexpr uint32_t TASK_GL_UB_TENSOR_OFFSET =
        TASK_GM_UB_TENSOR_OFFSET + TASK_STATE_SLOTS * MAX_ROW_NUM_SUB_CORE * sizeof(float);

    static constexpr uint32_t CAUSAL_MAX_COLUMN_NUM = KernelCommon::MAX_KV_STACK_LEN;
    static constexpr uint32_t CAUSAL_MAX_ROWS_PER_LOOP = FLOAT_VECTOR_SIZE;
    static constexpr uint32_t CAUSAL_KEEP_STRIDE = CAUSAL_MAX_COLUMN_NUM / 8;
    // Reuse the first 4 KiB of the raw GM-mask scratch; it does not alias P.
    static constexpr uint32_t CAUSAL_KEEP_UB_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;
    static constexpr uint32_t SELECT_SCALAR_UB_OFFSET =
        TASK_GL_UB_TENSOR_OFFSET + TASK_STATE_SLOTS * MAX_ROW_NUM_SUB_CORE * sizeof(float);

    __aicore__ inline BlockEpilogue() {}

    __aicore__ inline ~BlockEpilogue() {}

    __aicore__ inline static TaskPlan MakeTaskPlan(uint32_t rowNum, uint32_t qSBlockSize, uint32_t qNBlockSize)
    {
        TaskPlan plan{};
        plan.subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        const uint32_t rowSplitSubBlock = (qNBlockSize == 1U) ? qSBlockSize / 2U : qSBlockSize * qNSplitSubBlock;
        plan.rowActualThisSubBlock = (plan.subBlockIdx == 1U) ? rowNum - rowSplitSubBlock : rowSplitSubBlock;
        plan.rowOffsetThisSubBlock = plan.subBlockIdx * rowSplitSubBlock;
        return plan;
    }

    __aicore__ inline void init(Arch::Resource<ArchTag>& resource, float scaleValue_, float softcapValue_)
    {
        // Allocate UB space
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;

        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t SOFTCAP_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t LL_UB_TENSOR_OFFSET =
            10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE + REPEAT_SIZE_IN_BYTE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET =
            10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE + 2 * REPEAT_SIZE_IN_BYTE;
        constexpr uint32_t GL_UB_TENSOR_OFFSET =
            10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE + 5 * REPEAT_SIZE_IN_BYTE;
        constexpr uint32_t DM_UB_TENSOR_OFFSET =
            10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE + 8 * REPEAT_SIZE_IN_BYTE;

        scaleValue = scaleValue_;
        softcapValue = softcapValue_;

        lsUbTensor = resource.ubBuf.template GetBufferByByte<float>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        softcapUbTensor = resource.ubBuf.template GetBufferByByte<float>(SOFTCAP_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<float>(LM_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(LL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
    }

    __aicore__ inline void InitArithMask(Arch::Resource<ArchTag>& resource)
    {
        keepUbTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(CAUSAL_KEEP_UB_OFFSET);
        selectScalarUbTensor = resource.ubBuf.template GetBufferByByte<float>(SELECT_SCALAR_UB_OFFSET);
        maskSToVEventId = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V);
        maskVToSEventId = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
        AscendC::Duplicate(selectScalarUbTensor, -3.0e38f, FLOAT_BLOCK_SIZE);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReleaseArithMaskEvents()
    {
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::S_V>(maskSToVEventId);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(maskVToSEventId);
    }

    __aicore__ inline void ApplyArithMask(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNum,
                                          uint32_t columnNumRound, int64_t diagS0, uint32_t rowOffset,
                                          uint32_t qSBlockSize)
    {
        // The previous Select must finish reading keep before Scalar overwrites it.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(maskVToSEventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(maskVToSEventId);
        auto keepWords = keepUbTensor.ReinterpretCast<uint64_t>();
        const uint32_t wordCount = CeilDiv(columnNumRound, 64U);
        uint32_t token = rowOffset % qSBlockSize;
        for (uint32_t r = 0; r < rowNumCurLoop; ++r) {
            const int64_t rowVisible = diagS0 + static_cast<int64_t>(token) + 1;
            const int32_t visible =
                rowVisible <= 0 ? 0
                                : (rowVisible >= static_cast<int64_t>(columnNum) ? static_cast<int32_t>(columnNum)
                                                                                 : static_cast<int32_t>(rowVisible));
            for (uint32_t j = 0; j < wordCount; ++j) {
                const int32_t bits = visible - static_cast<int32_t>(j * 64);
                // LSB-first prefix; guard both endpoints to avoid invalid shifts.
                const uint64_t word =
                    bits >= 64 ? UINT64_MAX : (bits <= 0 ? uint64_t{0} : ((uint64_t{1} << bits) - uint64_t{1}));
                keepWords.SetValue(r * (CAUSAL_KEEP_STRIDE / sizeof(uint64_t)) + j, word);
            }
            if (++token == qSBlockSize) {
                token = 0;
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(maskSToVEventId);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(maskSToVEventId);
        AscendC::SetCmpMask(selectScalarUbTensor);
        AscendC::PipeBarrier<PIPE_V>();

        // Predicate rows are 32B-aligned slots, not a flat rows*W bitstream.
        // Count mode handles W's partial repeat without touching the next row.
        // The low-level overload emits vsel without scalar setup or mask resets.
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(columnNumRound);
        for (uint32_t r = 0; r < rowNumCurLoop; ++r) {
            const uint32_t offset = sUbOffset + r * columnNumRound;
            AscendC::Select(lsUbTensor[offset], keepUbTensor[r * CAUSAL_KEEP_STRIDE], lsUbTensor[offset],
                            static_cast<uint8_t>(1), AscendC::BinaryRepeatParams{});
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ApplySwaArithMask(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNum,
                                             uint32_t columnNumRound, int64_t diagS0, uint32_t rowOffset,
                                             uint32_t qSBlockSize, int64_t leftS0, bool applyLeft, bool applyRight)
    {
        // The previous Select must finish reading keep before Scalar overwrites it.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(maskVToSEventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(maskVToSEventId);
        auto keepWords = keepUbTensor.ReinterpretCast<uint64_t>();
        const uint32_t wordCount = CeilDiv(columnNumRound, 64U);
        uint32_t token = rowOffset % qSBlockSize;
        for (uint32_t r = 0; r < rowNumCurLoop; ++r) {
            const int64_t rowVisible = applyRight ? diagS0 + static_cast<int64_t>(token) + 1 : columnNum;
            const int64_t rowLeft = applyLeft ? leftS0 + static_cast<int64_t>(token) : 0;
            const int32_t visible =
                rowVisible <= 0 ? 0
                                : (rowVisible >= static_cast<int64_t>(columnNum) ? static_cast<int32_t>(columnNum)
                                                                                 : static_cast<int32_t>(rowVisible));
            for (uint32_t j = 0; j < wordCount; ++j) {
                const int32_t bits = visible - static_cast<int32_t>(j * 64);
                // LSB-first prefix; guard both endpoints to avoid invalid shifts.
                const uint64_t word =
                    bits >= 64 ? UINT64_MAX : (bits <= 0 ? uint64_t{0} : ((uint64_t{1} << bits) - uint64_t{1}));
                const int64_t leftBits = rowLeft - static_cast<int64_t>(j * 64);
                const uint64_t leftWord =
                    leftBits >= 64 ? UINT64_MAX
                                   : (leftBits <= 0 ? uint64_t{0} : ((uint64_t{1} << leftBits) - uint64_t{1}));
                keepWords.SetValue(r * (CAUSAL_KEEP_STRIDE / sizeof(uint64_t)) + j, word & ~leftWord);
            }
            if (++token == qSBlockSize) {
                token = 0;
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(maskSToVEventId);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(maskSToVEventId);
        AscendC::SetCmpMask(selectScalarUbTensor);
        AscendC::PipeBarrier<PIPE_V>();

        // Predicate rows are 32B-aligned slots, not a flat rows*W bitstream.
        // Count mode handles W's partial repeat without touching the next row.
        // The low-level overload emits vsel without scalar setup or mask resets.
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(columnNumRound);
        for (uint32_t r = 0; r < rowNumCurLoop; ++r) {
            const uint32_t offset = sUbOffset + r * columnNumRound;
            AscendC::Select(lsUbTensor[offset], keepUbTensor[r * CAUSAL_KEEP_STRIDE], lsUbTensor[offset],
                            static_cast<uint8_t>(1), AscendC::BinaryRepeatParams{});
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SetVecMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % FLOAT_VECTOR_SIZE;
        for (int64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }

        if (len == VECTOR_SIZE || len == 0) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    __aicore__ inline void SetBlockReduceMask(int32_t len)
    {
        if (len > 8 || len < 1) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        uint64_t subMask = ((uint64_t)1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask + (subMask << 56) +
                             (subMask << 40) + (subMask << 24) + (subMask << 8);
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
    }

    __aicore__ inline void Rowsum(const AscendC::LocalTensor<float>& srcUb, const AscendC::LocalTensor<float>& outUb,
                                  const AscendC::LocalTensor<float>& partial, uint32_t numRowsRound, uint32_t numElems,
                                  uint32_t numElemsAligned)
    {
        const uint32_t partialRowBlocks = CeilDiv(numElems, FLOAT_VECTOR_SIZE);
        const auto reduced = partial[DIRECT_REDUCED_OFFSET];
        for (uint32_t v = 0; v < numElems; v += FLOAT_VECTOR_SIZE) {
            const uint32_t valid = AscendC::Std::min(numElems - v, FLOAT_VECTOR_SIZE);
            const uint64_t mask = valid == FLOAT_VECTOR_SIZE ? UINT64_MAX : ((1ULL << valid) - 1ULL);
            AscendC::SetVectorMask<int8_t>(0, mask);
            AscendC::BlockReduceSum<float, false>(partial[v / FLOAT_BLOCK_SIZE], srcUb[v], numRowsRound, 0,
                                                  partialRowBlocks, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // Exclude unwritten tail slots instead of initializing shared scratch.
        const uint32_t validSlots = (partialRowBlocks - 1) * FLOAT_BLOCK_SIZE +
                                    CeilDiv(numElems - (partialRowBlocks - 1) * FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE);
        SetVecMask(validSlots);
        AscendC::BlockReduceSum<float, false>(reduced, partial, numRowsRound, 0, 1, 1, partialRowBlocks);
        AscendC::PipeBarrier<PIPE_V>();
        SetBlockReduceMask(partialRowBlocks);
        AscendC::BlockReduceSum<float, false>(outUb, reduced, numRowsRound / FLOAT_BLOCK_SIZE, 0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    __aicore__ inline void Rowsum512(const AscendC::LocalTensor<float>& srcUb, const AscendC::LocalTensor<float>& outUb,
                                     const AscendC::LocalTensor<float>& scratch, uint32_t numRowsRound,
                                     uint32_t numElemsAligned)
    {
        const auto reduced = scratch[UB_UINT8_VECTOR_SIZE];
        AscendC::BlockReduceSum<float, false>(scratch, srcUb, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE, 0, 1,
                                              1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum<float, false>(
            reduced, scratch, numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE, 0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum<float, false>(
            outUb, reduced, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE, 0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Rowmax(const AscendC::LocalTensor<float>& srcUb, const AscendC::LocalTensor<float>& outUb,
                                  const AscendC::LocalTensor<float>& partial, uint32_t numRowsRound, uint32_t numElems,
                                  uint32_t numElemsAligned)
    {
        const uint32_t partialRowBlocks = CeilDiv(numElems, FLOAT_VECTOR_SIZE);
        const auto reduced = partial[DIRECT_REDUCED_OFFSET];
        for (uint32_t v = 0; v < numElems; v += FLOAT_VECTOR_SIZE) {
            const uint32_t valid = AscendC::Std::min(numElems - v, FLOAT_VECTOR_SIZE);
            const uint64_t mask = valid == FLOAT_VECTOR_SIZE ? UINT64_MAX : ((1ULL << valid) - 1ULL);
            AscendC::SetVectorMask<int8_t>(0, mask);
            AscendC::BlockReduceMax<float, false>(partial[v / FLOAT_BLOCK_SIZE], srcUb[v], numRowsRound, 0,
                                                  partialRowBlocks, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // Exclude unwritten tail slots instead of initializing shared scratch.
        const uint32_t validSlots = (partialRowBlocks - 1) * FLOAT_BLOCK_SIZE +
                                    CeilDiv(numElems - (partialRowBlocks - 1) * FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE);
        SetVecMask(validSlots);
        AscendC::BlockReduceMax<float, false>(reduced, partial, numRowsRound, 0, 1, 1, partialRowBlocks);
        AscendC::PipeBarrier<PIPE_V>();
        SetBlockReduceMask(partialRowBlocks);
        AscendC::BlockReduceMax<float, false>(outUb, reduced, numRowsRound / FLOAT_BLOCK_SIZE, 0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    __aicore__ inline void Rowmax512(const AscendC::LocalTensor<float>& srcUb, const AscendC::LocalTensor<float>& outUb,
                                     const AscendC::LocalTensor<float>& scratch, uint32_t numRowsRound,
                                     uint32_t numElemsAligned)
    {
        const auto reduced = scratch[UB_UINT8_VECTOR_SIZE];
        AscendC::BlockReduceMax<float, false>(scratch, srcUb, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE, 0, 1,
                                              1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float, false>(
            reduced, scratch, numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE, 0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float, false>(
            outUb, reduced, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE, 0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopySGmToUb(AscendC::GlobalTensor<ElementInput> gInput, uint32_t sUbOffset,
                                       uint32_t rowNumCurLoop, uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(lsUbTensor[sUbOffset], gInput,
                          AscendC::DataCopyParams(rowNumCurLoop, columnNumRound / FLOAT_BLOCK_SIZE,
                                                  (columnNumPad - columnNumRound) / FLOAT_BLOCK_SIZE, 0));
    }

    __aicore__ inline void ScaleS(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        AscendC::Muls<float, false>(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], scaleValue, (uint64_t)0,
                                    CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                                    AscendC::UnaryRepeatParams(1, 1, 8, 8));

        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ApplySoftcap(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        uint32_t repeatTimes = CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE);
        AscendC::UnaryRepeatParams repeatParams(1, 1, 8, 8);

        // Clip the input to avoid overflow
        AscendC::Maxs<float, false>(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], -8.8f, (uint64_t)0, repeatTimes,
                                    repeatParams);
        AscendC::PipeBarrier<PIPE_V>();

        // (2 * softcap) / (1 + exp(-2x)) - softcap
        AscendC::Muls<float, false>(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], -2.0f, (uint64_t)0, repeatTimes,
                                    repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp<float, false>(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], (uint64_t)0, repeatTimes,
                                   repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds<float, false>(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], 1.0f, (uint64_t)0, repeatTimes,
                                    repeatParams);

        AscendC::Duplicate<float, false>(softcapUbTensor, 2 * softcapValue, (uint64_t)0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div<float, false>(lsUbTensor[sUbOffset], softcapUbTensor, lsUbTensor[sUbOffset], (uint64_t)0,
                                   repeatTimes, AscendC::BinaryRepeatParams(1, 1, 1, 8, 0, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds<float, false>(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], -softcapValue, (uint64_t)0,
                                    repeatTimes, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CalcLocalRowMax(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum,
                                           uint32_t columnNumRound, uint32_t rowOffset)
    {
        if (columnNum == 512U) {
            Rowmax512(lsUbTensor[sUbOffset], lmUbTensor[rowOffset], tvUbTensor, rowNumCurLoopRound, columnNumRound);
        } else {
            Rowmax(lsUbTensor[sUbOffset], lmUbTensor[rowOffset], tvUbTensor, rowNumCurLoopRound, columnNum,
                   columnNumRound);
        }
    }

    __aicore__ inline void UpdateGlobalRowMax(uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
                                              uint32_t columnNumRound, uint32_t dmUbOffsetCurCycle,
                                              uint32_t stateRowOffset, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            AscendC::DataCopy(hmUbTensor[rowOffset], lmUbTensor[rowOffset],
                              AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::DataCopy(gmUbTensor[stateRowOffset], lmUbTensor[rowOffset],
                              AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
            return;
        } else {
            SetVecMask(rowNumCurLoop);
            // *** hm = vmax(lm, gm)
            AscendC::Max<float, false>(hmUbTensor[rowOffset], lmUbTensor[rowOffset], gmUbTensor[stateRowOffset],
                                       (uint64_t)0, 1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = gm - hm
            AscendC::Sub<float, false>(dmUbTensor[dmUbOffsetCurCycle], gmUbTensor[stateRowOffset],
                                       hmUbTensor[rowOffset], (uint64_t)0, 1,
                                       AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = exp(dm)
            AscendC::Exp<float, false>(dmUbTensor[dmUbOffsetCurCycle], dmUbTensor[dmUbOffsetCurCycle], (uint64_t)0, 1,
                                       AscendC::UnaryRepeatParams(1, 1, 8, 8));
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();
        // *** gm = hm
        AscendC::DataCopy(gmUbTensor[stateRowOffset], hmUbTensor[rowOffset],
                          AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CalcExp(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
                                   uint32_t columnNum, uint32_t columnNumRound, uint32_t rowOffset)
    {
        // *** hm_block = expand_to_block(hm), 存放于 tv
        AscendC::Brcb(tvUbTensor.template ReinterpretCast<uint32_t>(),
                      hmUbTensor[rowOffset].template ReinterpretCast<uint32_t>(), rowNumCurLoopRound / FLOAT_BLOCK_SIZE,
                      AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        // *** ls = ls - hm_block
        for (uint32_t subIdx = 0; subIdx < columnNum / FLOAT_VECTOR_SIZE; ++subIdx) {
            AscendC::Sub<float, false>(lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE],
                                       lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE], tvUbTensor, (uint64_t)0,
                                       rowNumCurLoop,
                                       AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE,
                                                                   columnNumRound / FLOAT_BLOCK_SIZE, 1));
        }
        if (columnNum % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(columnNum % FLOAT_VECTOR_SIZE);
            AscendC::Sub<float, false>(lsUbTensor[sUbOffset][columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                       lsUbTensor[sUbOffset][columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                       tvUbTensor, (uint64_t)0, rowNumCurLoop,
                                       AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE,
                                                                   columnNumRound / FLOAT_BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // *** ls = exp(ls)
        AscendC::Exp<float, false>(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], (uint64_t)0,
                                   CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                                   AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CalcLocalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum,
                                           uint32_t columnNumRound, uint32_t rowOffset)
    {
        if (columnNum == 512U) {
            Rowsum512(lsUbTensor[sUbOffset], llUbTensor[rowOffset], tvUbTensor, rowNumCurLoopRound, columnNumRound);
        } else {
            Rowsum(lsUbTensor[sUbOffset], llUbTensor[rowOffset], tvUbTensor, rowNumCurLoopRound, columnNum,
                   columnNumRound);
        }
    }

    __aicore__ inline void UpdateGlobalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
                                              uint32_t dmUbOffsetCurCycle, uint32_t stateRowOffset, uint32_t rowOffset,
                                              uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // *** gl = ll
            AscendC::DataCopy(glUbTensor[stateRowOffset], llUbTensor[rowOffset],
                              AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(rowNumCurLoop);
            // *** gl = dm * gl
            AscendC::Mul<float, false>(glUbTensor[stateRowOffset], dmUbTensor[dmUbOffsetCurCycle],
                                       glUbTensor[stateRowOffset], (uint64_t)0, 1,
                                       AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** gl = ll + gl
            AscendC::Add<float, false>(glUbTensor[stateRowOffset], glUbTensor[stateRowOffset], llUbTensor[rowOffset],
                                       (uint64_t)0, 1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }

    __aicore__ inline void DownCastP(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        // *** lp = castfp32to16(ls)
        if (std::is_same<ElementOutput, bfloat16_t>::value) {
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[sUbOffset], lsUbTensor[sUbOffset], AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE), AscendC::UnaryRepeatParams(1, 1, 4, 8));
        } else {
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[sUbOffset], lsUbTensor[sUbOffset], AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE), AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }
    }

    __aicore__ inline void CopyPUbToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t sUbOffset,
                                       uint32_t rowNumCurLoop, uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(gOutput, lpUbTensor[sUbOffset],
                          AscendC::DataCopyParams(rowNumCurLoop, columnNumRound / BLOCK_SIZE, 0,
                                                  (columnNumPad - columnNumRound) / BLOCK_SIZE));
    }

    enum class BufferSyncMode : uint32_t {
        NoMask = 0,
        Masked = 1,
    };

    __aicore__ inline void SubCoreCompute(AscendC::GlobalTensor<ElementOutput> gOutput,
                                          const LayoutOutput& layoutOutput, uint32_t rowOffset,
                                          uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile,
                                          uint32_t isFirstRowLoop, uint32_t isLastRowLoop, uint32_t columnNumRound,
                                          uint32_t pingpongFlag, uint32_t curStackTileMod, uint32_t taskStateSlot,
                                          bool isSplitKV, BufferSyncMode syncMode,
                                          bool startsWithMaskThenNomaskFlag = false)
    {
        uint32_t rowNumCurLoop = layoutOutput.shape(0);
        uint32_t rowNumCurLoopRound = RoundUp(rowNumCurLoop, FLOAT_BLOCK_SIZE);
        uint32_t columnNum = layoutOutput.shape(1);
        uint32_t columnNumPad = layoutOutput.stride(0);
        uint32_t sUbOffset = pingpongFlag * MAX_UB_S_ELEM_NUM;
        uint32_t dmUbOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffset;
        uint32_t stateRowOffset = taskStateSlot * 64 + rowOffset;

        if (syncMode == BufferSyncMode::NoMask) {
            if (isLastNoMaskStackTile && isLastRowLoop) {
                if (!startsWithMaskThenNomaskFlag) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
            }
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        }

        CalcLocalRowMax(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        UpdateGlobalRowMax(rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound, dmUbOffsetCurCycle,
                           stateRowOffset, rowOffset, isFirstStackTile);

        CalcExp(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        if (syncMode == BufferSyncMode::NoMask) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
        }

        DownCastP(sUbOffset, rowNumCurLoop, columnNumRound);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);

        CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
        CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);

        if (syncMode == BufferSyncMode::Masked) {
            // Mask conversion aliases both P buffers; release both after the P store.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
        }
        UpdateGlobalRowSum(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, dmUbOffsetCurCycle, stateRowOffset, rowOffset,
                           isFirstStackTile);
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<ElementOutput> gOutput,
                                      AscendC::GlobalTensor<ElementInput> gInput, const LayoutOutput& layoutOutput,
                                      const LayoutInput& layoutInput, GemmCoord actualBlockShape,
                                      const TaskPlan& taskPlan, uint32_t isFirstStackTile,
                                      uint32_t isLastNoMaskStackTile, uint32_t qSBlockSize, uint32_t qNBlockSize,
                                      uint32_t curStackTileMod, uint32_t taskStateSlot, Arch::CrossCoreFlag qkReady,
                                      uint32_t& softmaxPingPongFlag, bool isSplitKV = false,
                                      bool startsWithMaskTile = false, bool startsWithMaskThenNomaskFlag = false)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE);
        uint32_t columnNumPad = layoutInput.stride(0);

        const uint32_t subBlockIdx = taskPlan.subBlockIdx;
        const uint32_t rowActualThisSubBlock = taskPlan.rowActualThisSubBlock;
        const uint32_t rowOffsetThisSubBlock = taskPlan.rowOffsetThisSubBlock;
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        // The idle vector core must still consume this stack's QK-ready flag.
        if (rowActualThisSubBlock == 0) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = softmaxPingPongFlag % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                if (startsWithMaskTile && rowLoopIdx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
                }
                // Delay QK-ready until the first S load to overlap scalar setup.
                if (rowLoopIdx == 0) {
                    Arch::CrossCoreWaitFlag(qkReady);
                }
                CopySGmToUb(gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound,
                            columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = (softmaxPingPongFlag - 1) % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (delayedRowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                if constexpr (HAS_SOFTCAP_) {
                    ApplySoftcap((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                }
                SubCoreCompute(gOutputCurLoop, layoutOutputCurLoop, rowOffsetCurLoop, isFirstStackTile,
                               isLastNoMaskStackTile, (delayedRowLoopIdx == 0), (delayedRowLoopIdx == rowLoopNum - 1),
                               columnNumRound, pingpongFlag, curStackTileMod, taskStateSlot, isSplitKV,
                               BufferSyncMode::NoMask, startsWithMaskThenNomaskFlag);
            }
            if (rowLoopIdx < rowLoopNum) {
                softmaxPingPongFlag++;
            }
        }
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<ElementOutput> gOutput,
                                      AscendC::GlobalTensor<ElementInput> gInput, const LayoutOutput& layoutOutput,
                                      const LayoutInput& layoutInput, GemmCoord actualBlockShape,
                                      const TaskPlan& taskPlan, uint32_t isFirstStackTile, uint32_t qSBlockSize,
                                      uint32_t curStackTileMod, uint32_t taskStateSlot, Arch::CrossCoreFlag qkReady,
                                      uint32_t& softmaxPingPongFlag, int64_t diagS0, bool doTriUMask,
                                      bool isSplitKV = false)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE);
        uint32_t columnNumPad = layoutInput.stride(0);
        const uint32_t subBlockIdx = taskPlan.subBlockIdx;
        const uint32_t rowActualThisSubBlock = taskPlan.rowActualThisSubBlock;
        const uint32_t rowOffsetThisSubBlock = taskPlan.rowOffsetThisSubBlock;

        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        if (rowActualThisSubBlock == 0) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = softmaxPingPongFlag % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                if (rowLoopIdx == 0) {
                    Arch::CrossCoreWaitFlag(qkReady);
                }
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound,
                            columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = (softmaxPingPongFlag - 1) % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowNumCurLoop =
                    (delayedRowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                if constexpr (HAS_SOFTCAP_) {
                    ApplySoftcap((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                }
                if (doTriUMask) {
                    ApplyArithMask(pingpongFlag * MAX_UB_S_ELEM_NUM, rowNumCurLoop, columnNum, columnNumRound, diagS0,
                                   rowOffsetThisSubBlock + rowOffsetCurLoop, qSBlockSize);
                }
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                SubCoreCompute(gOutputCurLoop, layoutOutputCurLoop, rowOffsetCurLoop, isFirstStackTile, 0,
                               (delayedRowLoopIdx == 0), (delayedRowLoopIdx == rowLoopNum - 1), columnNumRound,
                               pingpongFlag, curStackTileMod, taskStateSlot, isSplitKV,
                               // Arithmetic predicates do not alias P; both causal tile modes use ping-pong sync.
                               BufferSyncMode::NoMask);
            }
            if (rowLoopIdx < rowLoopNum) {
                softmaxPingPongFlag++;
            }
        }
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<ElementOutput> gOutput,
                                      AscendC::GlobalTensor<ElementInput> gInput, const LayoutOutput& layoutOutput,
                                      const LayoutInput& layoutInput, GemmCoord actualBlockShape,
                                      const TaskPlan& taskPlan, uint32_t isFirstStackTile, uint32_t qSBlockSize,
                                      uint32_t curStackTileMod, uint32_t taskStateSlot, Arch::CrossCoreFlag qkReady,
                                      uint32_t& softmaxPingPongFlag, int64_t diagS0, bool doTriUMask, bool isSplitKV,
                                      int64_t leftS0, bool applyLeft, bool applyRight)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE);
        uint32_t columnNumPad = layoutInput.stride(0);
        const uint32_t subBlockIdx = taskPlan.subBlockIdx;
        const uint32_t rowActualThisSubBlock = taskPlan.rowActualThisSubBlock;
        const uint32_t rowOffsetThisSubBlock = taskPlan.rowOffsetThisSubBlock;

        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        if (rowActualThisSubBlock == 0) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = softmaxPingPongFlag % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                if (rowLoopIdx == 0) {
                    Arch::CrossCoreWaitFlag(qkReady);
                }
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound,
                            columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = (softmaxPingPongFlag - 1) % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowNumCurLoop =
                    (delayedRowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                if constexpr (HAS_SOFTCAP_) {
                    ApplySoftcap((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                }
                if (doTriUMask) {
                    ApplySwaArithMask(pingpongFlag * MAX_UB_S_ELEM_NUM, rowNumCurLoop, columnNum, columnNumRound,
                                      diagS0, rowOffsetThisSubBlock + rowOffsetCurLoop, qSBlockSize, leftS0, applyLeft,
                                      applyRight);
                }
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                SubCoreCompute(gOutputCurLoop, layoutOutputCurLoop, rowOffsetCurLoop, isFirstStackTile, 0,
                               (delayedRowLoopIdx == 0), (delayedRowLoopIdx == rowLoopNum - 1), columnNumRound,
                               pingpongFlag, curStackTileMod, taskStateSlot, isSplitKV,
                               // Arithmetic predicates do not alias P; both SWA tile modes use ping-pong sync.
                               BufferSyncMode::NoMask);
            }
            if (rowLoopIdx < rowLoopNum) {
                softmaxPingPongFlag++;
            }
        }
    }

  private:
    float scaleValue;
    float softcapValue;

    AscendC::TEventID maskSToVEventId;
    AscendC::TEventID maskVToSEventId;
    AscendC::LocalTensor<uint8_t> keepUbTensor;
    AscendC::LocalTensor<float> selectScalarUbTensor;

    AscendC::LocalTensor<float> lsUbTensor;
    AscendC::LocalTensor<ElementOutput> lpUbTensor;
    AscendC::LocalTensor<float> lmUbTensor;
    AscendC::LocalTensor<float> hmUbTensor;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<float> llUbTensor;
    AscendC::LocalTensor<float> tvUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> softcapUbTensor;
};

} // namespace Catlass::Epilogue::Block

#endif
