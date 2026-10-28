/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASH_ATTN_NPU_950_V3_FD_COMBINE_HPP
#define FLASH_ATTN_NPU_950_V3_FD_COMBINE_HPP

#include <type_traits>

#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"

#include "tilingdata.h"
#include "kernel_common.hpp"

template <class ElementO, bool OutputLse>
class FlashDecodeCombine950 {
  public:
    static constexpr uint32_t MAX_SPLIT_NUM = MAX_FD_ACTIVE_CORE_NUM;
    static constexpr uint32_t MAX_ROW_TILE = FD_COMBINE_ROW_TILE;
    static constexpr uint32_t MAX_HEAD_DIM = 256U;
    static constexpr uint32_t FLOATS_PER_BLOCK = 8U;
    static constexpr uint32_t MAX_LSE_ELEMS = MAX_SPLIT_NUM * MAX_ROW_TILE;
    static constexpr uint32_t MAX_O_ELEMS = MAX_ROW_TILE * MAX_HEAD_DIM;
    static constexpr uint32_t MAX_O_INPUT_ELEMS = FD_COMBINE_O_INPUT_ELEMS;
    using Resource = Catlass::Arch::Resource<Catlass::Arch::Ascend950>;

    __aicore__ inline explicit FlashDecodeCombine950(Resource& resource)
    {
        constexpr uint32_t LSE_OFFSET = 0U;
        constexpr uint32_t WEIGHT_OFFSET = LSE_OFFSET + MAX_LSE_ELEMS * sizeof(float);
        constexpr uint32_t GLOBAL_LSE_OFFSET = WEIGHT_OFFSET + MAX_LSE_ELEMS * sizeof(float);
        constexpr uint32_t LSE_STORE_OFFSET = GLOBAL_LSE_OFFSET + MAX_ROW_TILE * sizeof(float);
        constexpr uint32_t O_INPUT_OFFSET = LSE_STORE_OFFSET + MAX_ROW_TILE * FLOATS_PER_BLOCK * sizeof(float);
        constexpr uint32_t O_ACC_OFFSET = O_INPUT_OFFSET + MAX_O_INPUT_ELEMS * sizeof(float);

        lseUb_ = resource.ubBuf.template GetBufferByByte<float>(LSE_OFFSET);
        weightUb_ = resource.ubBuf.template GetBufferByByte<float>(WEIGHT_OFFSET);
        globalLseUb_ = resource.ubBuf.template GetBufferByByte<float>(GLOBAL_LSE_OFFSET);
        lseStoreUb_ = resource.ubBuf.template GetBufferByByte<float>(LSE_STORE_OFFSET);
        oInputUb_ = resource.ubBuf.template GetBufferByByte<float>(O_INPUT_OFFSET);
        oAccUb_ = resource.ubBuf.template GetBufferByByte<float>(O_ACC_OFFSET);
        oOutUb_ = resource.ubBuf.template GetBufferByByte<ElementO>(O_INPUT_OFFSET);
    }

    __aicore__ inline void operator()(FAIKernelParams const& params)
    {
#ifdef __DAV_VEC__
        __gm__ FAInferTilingData* tiling = reinterpret_cast<__gm__ FAInferTilingData*>(params.tiling);
        constexpr uint32_t uniformMode = OutputLse ? FD_COMBINE_UNIFORM_WITH_LSE : FD_COMBINE_UNIFORM_NO_LSE;
        if ((tiling->fdCombineUseUniform & uniformMode) != 0U) {
            RunUniform(params);
            return;
        }
        AscendC::SetAtomicNone();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));

        const uint32_t aivIdx = AscendC::GetBlockIdx();
        // The caller has already completed partial writes and SyncAll on every
        // launched AIC/AIV, including AIVs with no combine assignment.
        if (aivIdx >= tiling->fdCombineAivNum) {
            return;
        }

        AscendC::GlobalTensor<float> partialLse;
        partialLse.SetGlobalBuffer((__gm__ float*)(params.workSpace + tiling->fdPartialLseOffset));
        AscendC::GlobalTensor<float> partialO;
        partialO.SetGlobalBuffer((__gm__ float*)(params.workSpace + tiling->fdPartialOOffset));
        AscendC::GlobalTensor<ElementO> output;
        output.SetGlobalBuffer((__gm__ ElementO*)params.o);
        AscendC::GlobalTensor<float> outputLse;
        if constexpr (OutputLse) {
            outputLse.SetGlobalBuffer((__gm__ float*)params.lse);
        }
        AscendC::GlobalTensor<int32_t> actualQ;
        actualQ.SetGlobalBuffer((__gm__ int32_t*)params.actualQseqlen);
        const uint32_t headDim = tiling->embeddingSizeV;
        const uint32_t inputStride = RoundUp(headDim, FLOATS_PER_BLOCK);
        // Keep the accumulator padded for the final low-precision output DMA.
        const uint32_t dRound = RoundUp(headDim, 32U / sizeof(ElementO));
        const uint32_t totalQ = OutputLse ? static_cast<uint32_t>(params.storage.totalQTokens) : 0U;
        const uint32_t combineIdx = tiling->fdCombineCoreSchedules[aivIdx].combineTaskIdx;
        const uint32_t sourceSubBlock = tiling->fdCombineCoreSchedules[aivIdx].sourceSubBlock;
        const uint32_t assignedStart = tiling->fdCombineCoreSchedules[aivIdx].sourceLocalRow;
        const uint32_t assignedRows = tiling->fdCombineCoreSchedules[aivIdx].rowCount;
        const uint32_t batchIdx = tiling->fdCombineTaskInfo[combineIdx].batchIdx;
        const uint32_t qRows = tiling->fdCombineTaskInfo[combineIdx].qRows;
        const uint32_t qHeadStart = tiling->fdCombineTaskInfo[combineIdx].qHeadStart;
        const uint32_t groupRows = tiling->fdCombineTaskInfo[combineIdx].rowNum;
        const uint32_t sourceRowSplit = Min(groupRows, RoundUp(groupRows, FLOATS_PER_BLOCK) / FD_SOURCE_SUB_BLOCK_NUM);
        // Host task geometry uses seqused_q, but packed storage follows cu-Q.
        const uint32_t qTokenBase =
            static_cast<uint32_t>(actualQ.GetValue(batchIdx)) + tiling->fdCombineTaskInfo[combineIdx].qStart;
        const uint32_t partialCount = static_cast<uint32_t>(tiling->fdCombineSchedules[combineIdx].partialCount);
        const uint32_t firstPartial = static_cast<uint32_t>(tiling->fdCombineSchedules[combineIdx].partialStart);
        const uint32_t partialCountAlign = RoundUp(partialCount, FLOATS_PER_BLOCK);

        for (uint32_t row = 0U; row < assignedRows; row += MAX_ROW_TILE) {
            const uint32_t sourceLocalRow = assignedStart + row;
            const uint32_t rowCount = Min(MAX_ROW_TILE, assignedRows - row);
            const uint32_t rowStart = (sourceSubBlock == 0U ? 0U : sourceRowSplit) + sourceLocalRow;
            const uint32_t rowCountAlign = RoundUp(rowCount, FLOATS_PER_BLOCK);
            PrepareWeights(partialLse, firstPartial, partialCount, partialCountAlign, sourceSubBlock, sourceLocalRow,
                           rowCount, rowCountAlign, tiling->fdLseSubStride);
            if constexpr (OutputLse) {
                // LSE is ready before O reduction. Its separate UB buffers
                // remain live until the end of this row tile.
                StoreLse(outputLse, rowStart, rowCount, qRows, qHeadStart, qTokenBase, totalQ);
            }
            const uint32_t tileElems = rowCount * dRound;
            AscendC::Duplicate(oAccUb_, 0.0F, tileElems);
            AscendC::PipeBarrier<PIPE_V>();
            const uint32_t inputPartialStride = rowCount * inputStride;
            const uint32_t batchCapacity = MAX_O_INPUT_ELEMS / inputPartialStride;
            for (uint32_t partial = 0U; partial < partialCount; partial += batchCapacity) {
                const uint32_t batchCount = Min(batchCapacity, partialCount - partial);
                const uint64_t partialOOffset =
                    (static_cast<uint64_t>(firstPartial + partial) * tiling->fdRowCapacity + rowStart) * headDim;
                LoadPartials(partialO, partialOOffset, batchCount, rowCount, headDim, inputStride,
                             tiling->fdRowCapacity);
                AccumulatePartials((__ubuf__ float*)oAccUb_.GetPhyAddr(), (__ubuf__ float*)oInputUb_.GetPhyAddr(),
                                   (__ubuf__ float*)weightUb_.GetPhyAddr(partial * rowCountAlign), batchCount, rowCount,
                                   headDim, inputStride, dRound, rowCountAlign);
                AscendC::PipeBarrier<PIPE_V>();
            }
            // Partial O and the weighted reduction stay FP32. Convert only
            // the final result to the public output dtype.
            if (std::is_same<ElementO, bfloat16_t>::value) {
                AscendC::Cast(oOutUb_, oAccUb_, AscendC::RoundMode::CAST_RINT, tileElems);
            } else {
                AscendC::Cast(oOutUb_, oAccUb_, AscendC::RoundMode::CAST_NONE, tileElems);
            }
            AscendC::PipeBarrier<PIPE_V>();
            StoreOutput(output, oOutUb_, rowStart, rowCount, qRows, qHeadStart, qTokenBase, tiling->numHeads, headDim,
                        dRound);
            if constexpr (OutputLse) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
    }

  private:
    // Preserve the original mapping and DMA/VF sequence for small or uniform
    // workloads selected by the host policy. All AIVs have passed SyncAll.
    __aicore__ inline void RunUniform(FAIKernelParams const& params)
    {
#ifdef __DAV_VEC__
        AscendC::SetAtomicNone();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));

        __gm__ FAInferTilingData* tiling = reinterpret_cast<__gm__ FAInferTilingData*>(params.tiling);
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t aivIdx = AscendC::GetBlockIdx();
        const uint32_t aivNum = AscendC::GetBlockNum() * subBlockNum;
        const uint32_t rowTile = Min(MAX_ROW_TILE, Max(1U, CeilDiv(tiling->fdCombineRowNum, aivNum)));

        AscendC::GlobalTensor<float> partialLse;
        partialLse.SetGlobalBuffer((__gm__ float*)(params.workSpace + tiling->fdPartialLseOffset));
        AscendC::GlobalTensor<float> partialO;
        partialO.SetGlobalBuffer((__gm__ float*)(params.workSpace + tiling->fdPartialOOffset));
        AscendC::GlobalTensor<ElementO> output;
        output.SetGlobalBuffer((__gm__ ElementO*)params.o);
        AscendC::GlobalTensor<float> outputLse;
        if constexpr (OutputLse) {
            outputLse.SetGlobalBuffer((__gm__ float*)params.lse);
        }
        AscendC::GlobalTensor<int32_t> actualQ;
        actualQ.SetGlobalBuffer((__gm__ int32_t*)params.actualQseqlen);
        AscendC::GlobalTensor<int32_t> seqUsedQ;
        seqUsedQ.SetGlobalBuffer((__gm__ int32_t*)params.seqUsedQ);

        const uint32_t groupSize = tiling->numHeads / tiling->kvHeads;
        const uint32_t headDim = tiling->embeddingSizeV;
        const uint32_t inputStride = RoundUp(headDim, FLOATS_PER_BLOCK);
        // Keep the accumulator padded for the final low-precision output DMA.
        const uint32_t dRound = RoundUp(headDim, 32U / sizeof(ElementO));
        const uint32_t totalQ = OutputLse ? static_cast<uint32_t>(params.storage.totalQTokens) : 0U;
        uint32_t workBase = 0U;
        for (uint32_t combineIdx = 0U; combineIdx < tiling->fdCombineTaskNum; ++combineIdx) {
            const uint32_t baseTask = static_cast<uint32_t>(tiling->fdCombineSchedules[combineIdx].baseTask);
            uint32_t batchIdx = 0U;
            uint32_t previousTasks = 0U;
            uint32_t qLen = 0U;
            uint32_t qNBlockTile = 1U;
            uint32_t qNBlockNumPerGroup = 1U;
            uint32_t qNTaskNum = 0U;
            while (batchIdx < tiling->batch) {
                // Used lengths select tasks; cu_seqlens_q defines storage offsets.
                qLen = params.seqUsedQ != nullptr
                           ? static_cast<uint32_t>(seqUsedQ.GetValue(batchIdx))
                           : static_cast<uint32_t>(actualQ.GetValue(batchIdx + 1U) - actualQ.GetValue(batchIdx));
                qNBlockTile = GetQNBlockTile(qLen, groupSize, headDim > 128U);
                qNBlockNumPerGroup = CeilDiv(groupSize, qNBlockTile);
                qNTaskNum = qNBlockNumPerGroup * tiling->kvHeads;
                const uint32_t batchTasks = CeilDiv(qLen, tiling->qBaseTile) * qNTaskNum;
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
            const uint32_t qNBlockIdx = localTask % qNTaskNum;
            const uint32_t qNBlockIdxInGroup = qNBlockIdx % qNBlockNumPerGroup;
            const uint32_t kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup;
            const uint32_t qHeadStart = kvHeadIdx * groupSize + qNBlockIdxInGroup * qNBlockTile;
            const uint32_t qNBlockSize = Min(qNBlockTile, groupSize - qNBlockIdxInGroup * qNBlockTile);
            const uint32_t qStart = qTileIdx * tiling->qBaseTile;
            const uint32_t qRows = Min(tiling->qBaseTile, qLen - qStart);
            const uint32_t groupRows = qRows * qNBlockSize;
            const uint32_t sourceRowSplit = Min(groupRows, RoundUp(groupRows, FLOATS_PER_BLOCK) / subBlockNum);
            const uint32_t sub0Tiles = CeilDiv(sourceRowSplit, rowTile);
            const uint32_t tilesPerTask = sub0Tiles + CeilDiv(groupRows - sourceRowSplit, rowTile);
            const uint32_t qTokenBase = static_cast<uint32_t>(actualQ.GetValue(batchIdx)) + qStart;
            const uint32_t partialCount = static_cast<uint32_t>(tiling->fdCombineSchedules[combineIdx].partialCount);
            const uint32_t firstPartial = static_cast<uint32_t>(tiling->fdCombineSchedules[combineIdx].partialStart);
            const uint32_t partialCountAlign = RoundUp(partialCount, FLOATS_PER_BLOCK);

            // Round-robin row tiles across all AIVs, including both subblocks.
            const uint32_t firstTile = (aivIdx + aivNum - workBase % aivNum) % aivNum;
            for (uint32_t tile = firstTile; tile < tilesPerTask; tile += aivNum) {
                const uint32_t sourceSubBlock = tile < sub0Tiles ? 0U : 1U;
                const uint32_t tileInSubBlock = sourceSubBlock == 0U ? tile : tile - sub0Tiles;
                const uint32_t sourceRows = sourceSubBlock == 0U ? sourceRowSplit : groupRows - sourceRowSplit;
                const uint32_t sourceLocalRow = tileInSubBlock * rowTile;
                const uint32_t rowCount = Min(rowTile, sourceRows - sourceLocalRow);
                const uint32_t rowStart = (sourceSubBlock == 0U ? 0U : sourceRowSplit) + sourceLocalRow;
                const uint32_t rowCountAlign = RoundUp(rowCount, FLOATS_PER_BLOCK);
                PrepareWeights(partialLse, firstPartial, partialCount, partialCountAlign, sourceSubBlock,
                               sourceLocalRow, rowCount, rowCountAlign, tiling->fdLseSubStride);
                if constexpr (OutputLse) {
                    StoreLse(outputLse, rowStart, rowCount, qRows, qHeadStart, qTokenBase, totalQ);
                }
                const uint32_t tileElems = rowCount * dRound;
                AscendC::Duplicate(oAccUb_, 0.0F, tileElems);
                AscendC::PipeBarrier<PIPE_V>();
                const uint32_t inputPartialStride = rowCount * inputStride;
                const uint32_t batchCapacity = MAX_O_INPUT_ELEMS / inputPartialStride;
                for (uint32_t partial = 0U; partial < partialCount; partial += batchCapacity) {
                    const uint32_t batchCount = Min(batchCapacity, partialCount - partial);
                    const uint64_t partialOOffset =
                        (static_cast<uint64_t>(firstPartial + partial) * tiling->fdRowCapacity + rowStart) * headDim;
                    LoadPartials(partialO, partialOOffset, batchCount, rowCount, headDim, inputStride,
                                 tiling->fdRowCapacity);
                    AccumulatePartials((__ubuf__ float*)oAccUb_.GetPhyAddr(), (__ubuf__ float*)oInputUb_.GetPhyAddr(),
                                       (__ubuf__ float*)weightUb_.GetPhyAddr(partial * rowCountAlign), batchCount,
                                       rowCount, headDim, inputStride, dRound, rowCountAlign);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                // Partial O and the weighted reduction stay FP32. Convert only
                // the final result to the public output dtype.
                if (std::is_same<ElementO, bfloat16_t>::value) {
                    AscendC::Cast(oOutUb_, oAccUb_, AscendC::RoundMode::CAST_RINT, tileElems);
                } else {
                    AscendC::Cast(oOutUb_, oAccUb_, AscendC::RoundMode::CAST_NONE, tileElems);
                }
                AscendC::PipeBarrier<PIPE_V>();
                StoreOutput(output, oOutUb_, rowStart, rowCount, qRows, qHeadStart, qTokenBase, tiling->numHeads,
                            headDim, dRound);
                if constexpr (OutputLse) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
                }
            }
            workBase += tilesPerTask;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
    }

#ifdef __DAV_VEC__
    __aicore__ inline void LoadPartials(AscendC::GlobalTensor<float>& partialO, uint64_t offset, uint32_t partialCount,
                                        uint32_t rowCount, uint32_t headDim, uint32_t inputStride, uint32_t rowCapacity)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        if (headDim == inputStride) {
            // Copy every split's row tile with one strided DMA. GM skips the
            // other rows of each partial; UB packs the split tiles together.
            AscendC::DataCopyPad(oInputUb_, partialO[offset],
                                 AscendC::DataCopyExtParams(partialCount, rowCount * headDim * sizeof(float),
                                                            (rowCapacity - rowCount) * headDim * sizeof(float), 0, 0),
                                 AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0.0F));
        } else {
            // Non-aligned dimensions need padding between UB rows. Queue all
            // split copies before waiting rather than synchronizing each split.
            for (uint32_t partial = 0U; partial < partialCount; ++partial) {
                const uint64_t srcOffset = offset + static_cast<uint64_t>(partial) * rowCapacity * headDim;
                AscendC::DataCopyPad(oInputUb_[partial * rowCount * inputStride], partialO[srcOffset],
                                     AscendC::DataCopyExtParams(rowCount, headDim * sizeof(float), 0, 0, 0),
                                     AscendC::DataCopyPadExtParams<float>(true, 0, inputStride - headDim, 0.0F));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
    }

    __aicore__ inline void PrepareWeights(AscendC::GlobalTensor<float>& partialLse, uint32_t firstPartial,
                                          uint32_t partialCount, uint32_t partialCountAlign, uint32_t sourceSubBlock,
                                          uint32_t sourceLocalRow, uint32_t rowCount, uint32_t rowCountAlign,
                                          uint32_t lseSubStride)
    {
        const uint32_t calcElems = partialCountAlign * rowCountAlign;
        const uint32_t lsePartialStride = 2U * lseSubStride;
        AscendC::Duplicate(lseUb_, -3.402823466e+38F, calcElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        const uint64_t lseOffset =
            static_cast<uint64_t>(firstPartial) * lsePartialStride + sourceSubBlock * lseSubStride + sourceLocalRow;
        AscendC::DataCopyPad(
            lseUb_, partialLse[lseOffset],
            AscendC::DataCopyExtParams(partialCount, rowCount * sizeof(float),
                                       (lsePartialStride - rowCount) * sizeof(float), 0, 0),
            AscendC::DataCopyPadExtParams<float>(true, 0, rowCountAlign - rowCount, -3.402823466e+38F));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        NormalizeWeights((__ubuf__ float*)lseUb_.GetPhyAddr(), (__ubuf__ float*)weightUb_.GetPhyAddr(),
                         (__ubuf__ float*)globalLseUb_.GetPhyAddr(), partialCountAlign, rowCount, rowCountAlign);
        // Publish VF stores before the O reduction or final LSE DMA reads UB.
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Each active lane is one row. Keep that row's max/sum in registers and
    // reuse them for every split instead of materializing broadcast tensors.
    __simd_vf__ static inline void NormalizeWeights(__ubuf__ float* lse, __ubuf__ float* weights,
                                                    __ubuf__ float* globalLse, uint32_t partialCountAlign,
                                                    uint32_t rowCount, uint32_t rowStride)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> maxReg, sumReg, valueReg;
        uint32_t remaining = rowCount;
        MaskReg mask = UpdateMask<float>(remaining);
        LoadAlign<float, LoadDist::DIST_NORM>(maxReg, lse);
        for (uint32_t partial = 1U; partial < partialCountAlign; ++partial) {
            LoadAlign<float, LoadDist::DIST_NORM>(valueReg, lse + partial * rowStride);
            AscendC::MicroAPI::Max(maxReg, maxReg, valueReg, mask);
        }

        Duplicate<float, MaskMergeMode::ZEROING, float>(sumReg, 0.0F, mask);
        for (uint32_t partial = 0U; partial < partialCountAlign; ++partial) {
            LoadAlign<float, LoadDist::DIST_NORM>(valueReg, lse + partial * rowStride);
            Sub(valueReg, valueReg, maxReg, mask);
            Exp(valueReg, valueReg, mask);
            Add(sumReg, sumReg, valueReg, mask);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(weights + partial * rowStride, valueReg, mask);
        }
        if constexpr (OutputLse) {
            Log(valueReg, sumReg, mask);
            Add(valueReg, valueReg, maxReg, mask);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(globalLse, valueReg, mask);
        }

        // The normalization loop reloads the unnormalized weights just stored.
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        for (uint32_t partial = 0U; partial < partialCountAlign; ++partial) {
            LoadAlign<float, LoadDist::DIST_NORM>(valueReg, weights + partial * rowStride);
            Div(valueReg, valueReg, sumReg, mask);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(weights + partial * rowStride, valueReg, mask);
        }
    }

    // Accumulate a batch of splits in vector registers, preserving split order
    // and separate FP32 Mul/Add instructions. Read/store the accumulator once
    // per batch rather than once per split; weights never pass through scalars.
    __simd_vf__ static inline void AccumulatePartials(__ubuf__ float* acc, __ubuf__ float* input,
                                                      __ubuf__ float* weights, uint32_t partialCount, uint32_t rowCount,
                                                      uint32_t headDim, uint32_t inputStride, uint32_t accStride,
                                                      uint32_t weightStride)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> weightReg, inputReg, accReg;
        for (uint32_t row = 0U; row < rowCount; ++row) {
            uint32_t remaining = headDim;
            for (uint32_t col = 0U; col < headDim; col += 64U) {
                MaskReg mask = UpdateMask<float>(remaining);
                const uint32_t accOffset = row * accStride + col;
                LoadAlign<float, LoadDist::DIST_NORM>(accReg, acc + accOffset);
                for (uint32_t partial = 0U; partial < partialCount; ++partial) {
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(weightReg, weights + partial * weightStride + row);
                    const uint32_t inputOffset = (partial * rowCount + row) * inputStride + col;
                    LoadAlign<float, LoadDist::DIST_NORM>(inputReg, input + inputOffset);
                    Mul(inputReg, inputReg, weightReg, mask);
                    Add(accReg, accReg, inputReg, mask);
                }
                StoreAlign<float, StoreDist::DIST_NORM_B32>(acc + accOffset, accReg, mask);
            }
        }
    }

    __aicore__ inline void StoreOutput(AscendC::GlobalTensor<ElementO>& output,
                                       const AscendC::LocalTensor<ElementO>& src, uint32_t rowStart, uint32_t rowCount,
                                       uint32_t qRows, uint32_t qHeadStart, uint32_t qTokenBase, uint32_t numHeads,
                                       uint32_t headDim, uint32_t dRound)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        if (qRows == 1U) {
            const uint64_t offset = (static_cast<uint64_t>(qTokenBase) * numHeads + qHeadStart + rowStart) * headDim;
            AscendC::DataCopyPad(output[offset], src,
                                 AscendC::DataCopyExtParams(rowCount, headDim * sizeof(ElementO), 0, 0, 0));
        } else {
            // Partial rows are head-major; output is TND. Copy each head's
            // token run with a destination stride, including tiles crossing heads.
            for (uint32_t row = 0U; row < rowCount;) {
                const uint32_t flatRow = rowStart + row;
                const uint32_t tokenInHead = flatRow % qRows;
                const uint32_t count = Min(rowCount - row, qRows - tokenInHead);
                const uint64_t offset =
                    (static_cast<uint64_t>(qTokenBase + tokenInHead) * numHeads + qHeadStart + flatRow / qRows) *
                    headDim;
                AscendC::DataCopyPad(output[offset], src[row * dRound],
                                     AscendC::DataCopyExtParams(count, headDim * sizeof(ElementO), 0,
                                                                (numHeads - 1U) * headDim * sizeof(ElementO), 0));
                row += count;
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    // Submit asynchronously; the caller waits for MTE3_V EVENT_ID1 after O
    // writeback, before either LSE buffer can be reused by the next row tile.
    __aicore__ inline void StoreLse(AscendC::GlobalTensor<float>& outputLse, uint32_t rowStart, uint32_t rowCount,
                                    uint32_t qRows, uint32_t qHeadStart, uint32_t qTokenBase, uint32_t totalQ)
    {
        if (qRows == 1U && totalQ == 1U) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::DataCopyPad(outputLse[qHeadStart + rowStart], globalLseUb_,
                                 AscendC::DataCopyExtParams(1, rowCount * sizeof(float), 0, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            return;
        }

        // Pad each scalar to one UB block so strided DMA can scatter LSE to
        // [head, total_q]. GM scalar stores cannot be used by every AIV on arch35.
        AscendC::Brcb(lseStoreUb_.template ReinterpretCast<uint32_t>(),
                      globalLseUb_.template ReinterpretCast<uint32_t>(), CeilDiv(rowCount, FLOATS_PER_BLOCK),
                      AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        if (qRows == 1U) {
            const uint64_t offset = static_cast<uint64_t>(qHeadStart + rowStart) * totalQ + qTokenBase;
            AscendC::DataCopyPad(
                outputLse[offset], lseStoreUb_,
                AscendC::DataCopyExtParams(rowCount, sizeof(float), 0, (totalQ - 1U) * sizeof(float), 0));
        } else {
            for (uint32_t row = 0U; row < rowCount;) {
                const uint32_t flatRow = rowStart + row;
                const uint32_t tokenInHead = flatRow % qRows;
                const uint32_t count = Min(rowCount - row, qRows - tokenInHead);
                const uint64_t offset =
                    static_cast<uint64_t>(qHeadStart + flatRow / qRows) * totalQ + qTokenBase + tokenInHead;
                AscendC::DataCopyPad(outputLse[offset], lseStoreUb_[row * FLOATS_PER_BLOCK],
                                     AscendC::DataCopyExtParams(count, sizeof(float), 0, 0, 0));
                row += count;
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }

    __aicore__ inline static uint32_t RoundUp(uint32_t value, uint32_t alignment)
    {
        return (value + alignment - 1U) / alignment * alignment;
    }
    __aicore__ inline static uint32_t CeilDiv(uint32_t value, uint32_t divisor)
    {
        return (value + divisor - 1U) / divisor;
    }
    __aicore__ inline static uint32_t Min(uint32_t lhs, uint32_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }
    __aicore__ inline static uint32_t Max(uint32_t lhs, uint32_t rhs)
    {
        return lhs > rhs ? lhs : rhs;
    }
#endif

    AscendC::LocalTensor<float> lseUb_;
    AscendC::LocalTensor<float> weightUb_;
    AscendC::LocalTensor<float> globalLseUb_;
    AscendC::LocalTensor<float> lseStoreUb_;
    AscendC::LocalTensor<float> oInputUb_;
    AscendC::LocalTensor<float> oAccUb_;
    AscendC::LocalTensor<ElementO> oOutUb_;
};

#endif
