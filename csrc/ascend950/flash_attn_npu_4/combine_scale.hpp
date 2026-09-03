/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASH_ATTN_NPU_950_V4_COMBINE_SCALE_HPP
#define FLASH_ATTN_NPU_950_V4_COMBINE_SCALE_HPP

#include <type_traits>

#include "kernel_operator.h"
#include "adv_api/reduce/reduce.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"

#include "tilingdata.h"
#include "kernel_common.hpp"

template <class ElementO>
class FlashDecodeCombine950 {
public:
    static constexpr uint32_t MAX_SPLIT_NUM = MAX_FD_ACTIVE_CORE_NUM;
    // FD is gated to q_seqlen <= 16.  A fused Q-head task is combined one
    // head at a time, so the UB requirement remains bounded by 16 rows.
    static constexpr uint32_t MAX_ROWS_PER_SUBBLOCK = 16U;
    static constexpr uint32_t MAX_HEAD_DIM = 256U;
    static constexpr uint32_t FLOATS_PER_BLOCK = 8U;
    static constexpr uint32_t MAX_LSE_ELEMS =
        MAX_SPLIT_NUM * MAX_ROWS_PER_SUBBLOCK;
    static constexpr uint32_t MAX_O_ELEMS =
        MAX_ROWS_PER_SUBBLOCK * MAX_HEAD_DIM;

    __aicore__ inline FlashDecodeCombine950()
    {
        constexpr uint32_t LSE_OFFSET = 0U;
        constexpr uint32_t LSE_BROADCAST_OFFSET =
            LSE_OFFSET + MAX_LSE_ELEMS * sizeof(float);
        constexpr uint32_t WEIGHT_OFFSET =
            LSE_BROADCAST_OFFSET + MAX_LSE_ELEMS * sizeof(float);
        constexpr uint32_t LSE_MAX_OFFSET =
            WEIGHT_OFFSET + MAX_LSE_ELEMS * sizeof(float);
        constexpr uint32_t LSE_SUM_OFFSET =
            LSE_MAX_OFFSET + MAX_ROWS_PER_SUBBLOCK * sizeof(float);
        constexpr uint32_t GLOBAL_LSE_OFFSET =
            LSE_SUM_OFFSET + MAX_ROWS_PER_SUBBLOCK * sizeof(float);
        constexpr uint32_t O_INPUT_OFFSET =
            GLOBAL_LSE_OFFSET + MAX_ROWS_PER_SUBBLOCK * sizeof(float);
        constexpr uint32_t O_TMP_OFFSET =
            O_INPUT_OFFSET + MAX_O_ELEMS * sizeof(ElementO);
        constexpr uint32_t O_ACC_OFFSET =
            O_TMP_OFFSET + MAX_O_ELEMS * sizeof(float);
        constexpr uint32_t REDUCE_TMP_OFFSET =
            O_ACC_OFFSET + MAX_O_ELEMS * sizeof(float);

        lseUb_ = resource.ubBuf.template GetBufferByByte<float>(LSE_OFFSET);
        broadcastUb_ =
            resource.ubBuf.template GetBufferByByte<float>(LSE_BROADCAST_OFFSET);
        weightUb_ = resource.ubBuf.template GetBufferByByte<float>(WEIGHT_OFFSET);
        lseMaxUb_ = resource.ubBuf.template GetBufferByByte<float>(LSE_MAX_OFFSET);
        lseSumUb_ = resource.ubBuf.template GetBufferByByte<float>(LSE_SUM_OFFSET);
        globalLseUb_ =
            resource.ubBuf.template GetBufferByByte<float>(GLOBAL_LSE_OFFSET);
        oInputUb_ =
            resource.ubBuf.template GetBufferByByte<ElementO>(O_INPUT_OFFSET);
        oTmpUb_ = resource.ubBuf.template GetBufferByByte<float>(O_TMP_OFFSET);
        oAccUb_ = resource.ubBuf.template GetBufferByByte<float>(O_ACC_OFFSET);
        oOutUb_ = resource.ubBuf.template GetBufferByByte<ElementO>(O_INPUT_OFFSET);
        reduceTmpUb_ =
            resource.ubBuf.template GetBufferByByte<uint8_t>(REDUCE_TMP_OFFSET);
    }

    __aicore__ inline void operator()(FAIKernelParams const &params)
    {
#ifdef __DAV_VEC__
        AscendC::SetAtomicNone();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<int8_t>(
            static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));

        __gm__ FAInferTilingData *tiling =
            reinterpret_cast<__gm__ FAInferTilingData *>(params.tiling);
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t combineIdx = AscendC::GetBlockIdx() / subBlockNum;
        if (combineIdx >= tiling->fdCombineTaskNum) {
            return;
        }

        AscendC::GlobalTensor<float> partialLse;
        partialLse.SetGlobalBuffer((__gm__ float *)(
            params.workSpace + tiling->fdPartialLseOffset));
        AscendC::GlobalTensor<ElementO> partialO;
        partialO.SetGlobalBuffer((__gm__ ElementO *)(
            params.workSpace + tiling->fdPartialOOffset));
        AscendC::GlobalTensor<ElementO> output;
        output.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<float> outputLse;
        outputLse.SetGlobalBuffer((__gm__ float *)params.lse);
        AscendC::GlobalTensor<int32_t> actualQ;
        actualQ.SetGlobalBuffer((__gm__ int32_t *)params.actualQseqlen);

        const uint32_t baseTask = static_cast<uint32_t>(
            tiling->fdCombineSchedules[combineIdx].baseTask);
        uint32_t batchIdx = 0U;
        uint32_t previousTasks = 0U;
        uint32_t batchTasks = 0U;
        uint32_t qLen = 0U;
        uint32_t qNBlockTile = 1U;
        uint32_t qNBlockNumPerGroup = 1U;
        uint32_t qNTaskNum = 0U;
        const uint32_t groupSize = tiling->numHeads / tiling->kvHeads;
        while (batchIdx < tiling->batch) {
            qLen = static_cast<uint32_t>(
                actualQ.GetValue(batchIdx + 1U) - actualQ.GetValue(batchIdx));
            qNBlockTile = GetQNBlockTile(
                qLen, groupSize, tiling->embeddingSizeV > 128U);
            qNBlockNumPerGroup = CeilDiv(groupSize, qNBlockTile);
            qNTaskNum = qNBlockNumPerGroup * tiling->kvHeads;
            batchTasks = CeilDiv(qLen, tiling->qBaseTile) * qNTaskNum;
            if (baseTask < previousTasks + batchTasks) {
                break;
            }
            previousTasks += batchTasks;
            ++batchIdx;
        }
        if (batchIdx >= tiling->batch) {
            return;
        }

        const uint32_t localTask = baseTask - previousTasks;
        const uint32_t qTileIdx = localTask / qNTaskNum;
        const uint32_t qNBlockIdx = localTask - qTileIdx * qNTaskNum;
        const uint32_t qNBlockIdxInGroup =
            qNBlockIdx % qNBlockNumPerGroup;
        const uint32_t kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup;
        const uint32_t qHeadStart =
            kvHeadIdx * groupSize + qNBlockIdxInGroup * qNBlockTile;
        const uint32_t qNBlockSize = Min(
            qNBlockTile, groupSize - qNBlockIdxInGroup * qNBlockTile);
        const uint32_t qStart = qTileIdx * tiling->qBaseTile;
        const uint32_t qRows = Min(tiling->qBaseTile, qLen - qStart);
        const uint32_t groupRows = qRows * qNBlockSize;
        const uint32_t sourceRowSplit =
            Min(groupRows, RoundUp(groupRows, FLOATS_PER_BLOCK) / subBlockNum);
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        if (subBlockIdx != 0U || qRows == 0U) {
            return;
        }

        const uint32_t partialCount = static_cast<uint32_t>(
            tiling->fdCombineSchedules[combineIdx].partialCount);
        const uint32_t firstPartial = static_cast<uint32_t>(
            tiling->fdCombineSchedules[combineIdx].partialStart);
        const uint32_t partialCountAlign =
            RoundUp(partialCount, FLOATS_PER_BLOCK);
        const uint32_t headDim = tiling->embeddingSizeV;
        const uint32_t dRound =
            RoundUp(headDim, FLOATS_PER_BLOCK);
        const uint32_t lsePartialStride = 2U * tiling->fdLseSubStride;
        const uint32_t qTokenBase =
            static_cast<uint32_t>(actualQ.GetValue(batchIdx)) + qStart;

        for (uint32_t qHeadOffset = 0U;
             qHeadOffset < qNBlockSize; ++qHeadOffset) {
            const uint32_t rowStart = qHeadOffset * qRows;
            const uint32_t localRows = qRows;
            const uint32_t rowCountAlign =
                RoundUp(localRows, FLOATS_PER_BLOCK);

            PrepareWeights(partialLse, firstPartial, partialCount,
                partialCountAlign, rowStart, sourceRowSplit,
                localRows, rowCountAlign,
                tiling->fdLseSubStride, lsePartialStride);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

            AscendC::Duplicate(oAccUb_, 0.0F, localRows * dRound);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t partial = 0U; partial < partialCount; ++partial) {
                const uint64_t partialOOffset =
                    (static_cast<uint64_t>(firstPartial + partial) *
                        tiling->fdRowCapacity + rowStart) * headDim;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::DataCopyPad(
                    oInputUb_, partialO[partialOOffset],
                    AscendC::DataCopyExtParams(
                        localRows, headDim * sizeof(ElementO), 0, 0, 0),
                    AscendC::DataCopyPadExtParams<ElementO>(
                        true, 0, dRound - headDim, static_cast<ElementO>(0.0F)));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                for (uint32_t row = 0U; row < localRows; ++row) {
                    AscendC::Cast(
                        oTmpUb_[row * dRound], oInputUb_[row * dRound],
                        AscendC::RoundMode::CAST_NONE, headDim);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(
                        oTmpUb_[row * dRound], oTmpUb_[row * dRound],
                        weightUb_.GetValue(partial * rowCountAlign + row),
                        headDim);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(
                        oAccUb_[row * dRound], oAccUb_[row * dRound],
                        oTmpUb_[row * dRound], headDim);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }

            for (uint32_t row = 0U; row < localRows; ++row) {
                if (std::is_same<ElementO, bfloat16_t>::value) {
                    AscendC::Cast(
                        oOutUb_[row * dRound], oAccUb_[row * dRound],
                        AscendC::RoundMode::CAST_RINT, headDim);
                } else {
                    AscendC::Cast(
                        oOutUb_[row * dRound], oAccUb_[row * dRound],
                        AscendC::RoundMode::CAST_NONE, headDim);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            const uint32_t qHeadIdx = qHeadStart + qHeadOffset;
            for (uint32_t row = 0U; row < localRows; ++row) {
                const uint64_t token = qTokenBase + row;
                const uint64_t outputOffset =
                    (token * tiling->numHeads + qHeadIdx) * headDim;
                AscendC::DataCopyPad(
                    output[outputOffset], oOutUb_[row * dRound],
                    AscendC::DataCopyExtParams(
                        1, headDim * sizeof(ElementO), 0, 0, 0));
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }

        // On arch35 scalar GM stores are only reliable from the leading AIV.
        // O remains parallel across combine tasks; AIV0 serializes the small
        // (FD-gated) set of LSE reductions and scalar stores.
        if (AscendC::GetBlockIdx() == 0U) {
            WriteAllLse(tiling, partialLse, actualQ, outputLse);
        }
#endif
    }

private:
#ifdef __DAV_VEC__
    __aicore__ inline void PrepareWeights(
        AscendC::GlobalTensor<float> &partialLse,
        uint32_t firstPartial, uint32_t partialCount,
        uint32_t partialCountAlign, uint32_t sourceRowStart,
        uint32_t sourceRowSplit,
        uint32_t rowCount, uint32_t rowCountAlign,
        uint32_t lseSubStride, uint32_t lsePartialStride)
    {
        const uint32_t calcElems = partialCountAlign * rowCountAlign;
        AscendC::Duplicate(lseUb_, -3.402823466e+38F, calcElems);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        for (uint32_t partial = 0U; partial < partialCount; ++partial) {
            for (uint32_t row = 0U; row < rowCount; ++row) {
                const uint32_t sourceRow = sourceRowStart + row;
                const uint32_t sourceSubBlock =
                    sourceRow < sourceRowSplit ? 0U : 1U;
                const uint32_t sourceLocalRow = sourceRow < sourceRowSplit ?
                    sourceRow : sourceRow - sourceRowSplit;
                const uint64_t lseOffset =
                    static_cast<uint64_t>(firstPartial + partial) *
                        lsePartialStride +
                    sourceSubBlock * lseSubStride + sourceLocalRow;
                lseUb_.SetValue(
                    partial * rowCountAlign + row,
                    partialLse.GetValue(lseOffset));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

        uint32_t reduceShape[] = {partialCountAlign, rowCountAlign};
        AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA, false>(
            lseMaxUb_, lseUb_, reduceTmpUb_, reduceShape, true);
        AscendC::PipeBarrier<PIPE_V>();
        BroadcastRows(lseMaxUb_, rowCountAlign, partialCountAlign);
        AscendC::Sub(weightUb_, lseUb_, broadcastUb_, calcElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(weightUb_, weightUb_, calcElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, false>(
            lseSumUb_, weightUb_, reduceTmpUb_, reduceShape, true);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Ln(globalLseUb_, lseSumUb_, rowCountAlign);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(globalLseUb_, globalLseUb_, lseMaxUb_, rowCountAlign);
        AscendC::PipeBarrier<PIPE_V>();
        BroadcastRows(globalLseUb_, rowCountAlign, partialCountAlign);
        AscendC::Sub(weightUb_, lseUb_, broadcastUb_, calcElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(weightUb_, weightUb_, calcElems);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BroadcastRows(
        const AscendC::LocalTensor<float> &src,
        uint32_t rowCountAlign, uint32_t partialCountAlign)
    {
        for (uint32_t partial = 0U; partial < partialCountAlign; ++partial) {
            AscendC::Adds(
                broadcastUb_[partial * rowCountAlign], src,
                0.0F, rowCountAlign);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void WriteAllLse(
        __gm__ FAInferTilingData *tiling,
        AscendC::GlobalTensor<float> &partialLse,
        AscendC::GlobalTensor<int32_t> &actualQ,
        AscendC::GlobalTensor<float> &outputLse)
    {
        for (uint32_t combine = 0U;
             combine < tiling->fdCombineTaskNum; ++combine) {
            const uint32_t baseTask = static_cast<uint32_t>(
                tiling->fdCombineSchedules[combine].baseTask);
            uint32_t batchIdx = 0U;
            uint32_t previousTasks = 0U;
            uint32_t qLen = 0U;
            uint32_t qNBlockTile = 1U;
            uint32_t qNBlockNumPerGroup = 1U;
            uint32_t qNTaskNum = 0U;
            const uint32_t groupSize = tiling->numHeads / tiling->kvHeads;
            while (batchIdx < tiling->batch) {
                qLen = static_cast<uint32_t>(
                    actualQ.GetValue(batchIdx + 1U) -
                    actualQ.GetValue(batchIdx));
                qNBlockTile = GetQNBlockTile(
                    qLen, groupSize, tiling->embeddingSizeV > 128U);
                qNBlockNumPerGroup = CeilDiv(groupSize, qNBlockTile);
                qNTaskNum = qNBlockNumPerGroup * tiling->kvHeads;
                const uint32_t batchTasks =
                    CeilDiv(qLen, tiling->qBaseTile) * qNTaskNum;
                if (baseTask < previousTasks + batchTasks) {
                    break;
                }
                previousTasks += batchTasks;
                ++batchIdx;
            }
            if (batchIdx >= tiling->batch) {
                continue;
            }

            const uint32_t localTask = baseTask - previousTasks;
            const uint32_t qTileIdx = localTask / qNTaskNum;
            const uint32_t qNBlockIdx = localTask - qTileIdx * qNTaskNum;
            const uint32_t qNBlockIdxInGroup =
                qNBlockIdx % qNBlockNumPerGroup;
            const uint32_t kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup;
            const uint32_t qHeadStart =
                kvHeadIdx * groupSize + qNBlockIdxInGroup * qNBlockTile;
            const uint32_t qNBlockSize = Min(
                qNBlockTile, groupSize - qNBlockIdxInGroup * qNBlockTile);
            const uint32_t qStart = qTileIdx * tiling->qBaseTile;
            const uint32_t qRows = Min(tiling->qBaseTile, qLen - qStart);
            const uint32_t groupRows = qRows * qNBlockSize;
            const uint32_t sourceRowSplit = Min(groupRows,
                RoundUp(groupRows, FLOATS_PER_BLOCK) / 2U);
            const uint32_t partialCount = static_cast<uint32_t>(
                tiling->fdCombineSchedules[combine].partialCount);
            const uint32_t firstPartial = static_cast<uint32_t>(
                tiling->fdCombineSchedules[combine].partialStart);
            const uint32_t partialCountAlign =
                RoundUp(partialCount, FLOATS_PER_BLOCK);
            const uint32_t rowCountAlign =
                RoundUp(qRows, FLOATS_PER_BLOCK);
            const uint32_t qTokenBase =
                static_cast<uint32_t>(actualQ.GetValue(batchIdx)) + qStart;
            const uint32_t totalQ = static_cast<uint32_t>(
                actualQ.GetValue(tiling->batch));
            for (uint32_t qHeadOffset = 0U;
                 qHeadOffset < qNBlockSize; ++qHeadOffset) {
                const uint32_t sourceRowStart = qHeadOffset * qRows;
                PrepareWeights(partialLse, firstPartial, partialCount,
                    partialCountAlign, sourceRowStart, sourceRowSplit,
                    qRows, rowCountAlign, tiling->fdLseSubStride,
                    2U * tiling->fdLseSubStride);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                const uint32_t qHeadIdx = qHeadStart + qHeadOffset;
                for (uint32_t row = 0U; row < qRows; ++row) {
                    outputLse.SetValue(
                        static_cast<uint64_t>(qHeadIdx) * totalQ +
                            qTokenBase + row,
                        globalLseUb_.GetValue(row));
                }
            }
        }
    }

    __aicore__ inline static uint32_t RoundUp(
        uint32_t value, uint32_t alignment)
    {
        return (value + alignment - 1U) / alignment * alignment;
    }

    __aicore__ inline static uint32_t CeilDiv(
        uint32_t value, uint32_t divisor)
    {
        return (value + divisor - 1U) / divisor;
    }

    __aicore__ inline static uint32_t Min(uint32_t lhs, uint32_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }
#endif

    Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource;
    AscendC::LocalTensor<float> lseUb_;
    AscendC::LocalTensor<float> broadcastUb_;
    AscendC::LocalTensor<float> weightUb_;
    AscendC::LocalTensor<float> lseMaxUb_;
    AscendC::LocalTensor<float> lseSumUb_;
    AscendC::LocalTensor<float> globalLseUb_;
    AscendC::LocalTensor<ElementO> oInputUb_;
    AscendC::LocalTensor<float> oTmpUb_;
    AscendC::LocalTensor<float> oAccUb_;
    AscendC::LocalTensor<ElementO> oOutUb_;
    AscendC::LocalTensor<uint8_t> reduceTmpUb_;
};

template <class InDtype>
CATLASS_GLOBAL void FAFlashDecodeCombine(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables,
    GM_ADDR o, GM_ADDR lse, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
    GM_ADDR workspace, GM_ADDR tiling)
{
    FAIKernelParams params{q, k, v, mask, blockTables,
        actualQseqlen, actualKvseqlen, o, lse, workspace, tiling};
    FlashDecodeCombine950<InDtype> combine;
    combine(params);
}

#endif
