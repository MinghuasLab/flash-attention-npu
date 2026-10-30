/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"

#define GEMM_BLOCK_BLOCK_MMAD_FLASH_ATTENTION_QK_HPP
#define GEMM_BLOCK_BLOCK_MMAD_FLASH_ATTENTION_QK_DN_HPP
#define GEMM_BLOCK_BLOCK_MMAD_FLASH_ATTENTION_PV_HPP
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_SOFTMAX_HIGH_PREC_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_RESCALE_O_HPP
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"

#include "tla/tensor.hpp"
#include "tla/layout.hpp"

#include "kernel_common.hpp"
#include "fa_block.h"
#include "rescale_o.hpp"
#include "online_softmax_quant.hpp"
#include "init_outputs.hpp"
#include "qk_matmul.hpp"
#include "qk_matmul_DN.hpp"
#include "pv_matmul.hpp"
#include "tilingdata.h"
#include "fd_combine.hpp"

using namespace Catlass;
using namespace tla;

namespace QuantFA {
using QuantMask::CAUSAL_MASK;
using QuantMask::NO_MASK;
using QuantMask::SLIDING_WINDOW_MASK;

static constexpr uint32_t PRE_LAUNCH = 2;
static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
static constexpr uint32_t UB_OTMP_BUF_STAGES = 1;
static constexpr uint32_t UB_S_BUF_STAGES = 2;
static constexpr uint32_t HEAD_DIM = 128;
static constexpr uint32_t VECTOR_SUBCORE_NUM = 2;
static constexpr uint32_t MATRIX_ALIGNMENT = 32;
static constexpr uint32_t UB_BLOCK_BYTES = 32;
static constexpr uint32_t VECTOR_REGISTER_BYTES = 256;
static constexpr uint32_t CUBE_MATRIX_BLOCK = 16;

template <class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV, class EpilogueRescaleO, Format qFormat,
          Format kvFormat, CacheMode kvcacheType, PageShape kvcacheShape, MaskCategory maskCategory,
          CacheLayout cacheLayout, bool LseMode>
class FAIKernel950 {
  public:
    using ArchTag = typename BlockMmadQK::ArchTag;

    using ElementQ = typename BlockMmadQK::ElementB;
    using ElementK = typename BlockMmadQK::ElementA;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = bfloat16_t;

    using LayoutQ = layout::ColumnMajor;
    using LayoutK = layout::RowMajor;
    using LayoutS = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    bool hasSeqUsedQ = false;
    bool hasSeqUsedKv = false;

    __aicore__ inline FAIKernel950() {}

    // *Used* lengths from seqused when provided; cu stays the region layout.
    __aicore__ inline int64_t GetQSeqLen(uint32_t bIdx, uint32_t maxQSeqlen, AscendC::GlobalTensor<int32_t>& gQSeq,
                                         AscendC::GlobalTensor<int32_t>& gSeqUsedQ) const
    {
        if constexpr (qFormat == Format::TND) {
            if (hasSeqUsedQ) {
                return static_cast<int64_t>(gSeqUsedQ.GetValue(bIdx));
            }
            return static_cast<int64_t>(gQSeq.GetValue(bIdx + 1) - gQSeq.GetValue(bIdx));
        }
        return static_cast<int64_t>(maxQSeqlen);
    }

    __aicore__ inline int64_t GetKvSeqLen(uint32_t bIdx, AscendC::GlobalTensor<int32_t>& gKvSeq,
                                          AscendC::GlobalTensor<int32_t>& gSeqUsedKv) const
    {
        if constexpr (kvFormat == Format::TND) {
            if (hasSeqUsedKv) {
                return static_cast<int64_t>(gSeqUsedKv.GetValue(bIdx));
            }
            if constexpr (kvcacheType == CacheMode::normalCache) {
                return static_cast<int64_t>(gKvSeq.GetValue(bIdx + 1) - gKvSeq.GetValue(bIdx));
            }
        }
        return static_cast<int64_t>(gKvSeq.GetValue(bIdx));
    }

    __aicore__ inline void operator()(FAIKernelParams const& params)
    {
        __gm__ FAInferTilingData* faiTilingData = reinterpret_cast<__gm__ FAInferTilingData*>(params.tiling);
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK*)params.v);
        AscendC::GlobalTensor<int32_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int32_t*)params.actualQseqlen);
        AscendC::GlobalTensor<int32_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t*)params.actualKvseqlen);
        AscendC::GlobalTensor<int32_t> gSeqUsedQ;
        gSeqUsedQ.SetGlobalBuffer((__gm__ int32_t*)params.seqUsedQ);
        AscendC::GlobalTensor<int32_t> gSeqUsedKv;
        gSeqUsedKv.SetGlobalBuffer((__gm__ int32_t*)params.seqUsedKv);
        hasSeqUsedQ = params.seqUsedQ != nullptr;
        hasSeqUsedKv = params.seqUsedKv != nullptr;
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t*)(params.blockTables));
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
        AscendC::GlobalTensor<float> gLse;
        gLse.SetGlobalBuffer((__gm__ float*)params.lse);
        AscendC::GlobalTensor<float> gPartialLse;
        gPartialLse.SetGlobalBuffer((__gm__ float*)(params.workSpace + faiTilingData->fdPartialLseOffset));
        AscendC::GlobalTensor<float> gPartialO;
        gPartialO.SetGlobalBuffer((__gm__ float*)(params.workSpace + faiTilingData->fdPartialOOffset));
#ifdef __DAV_VEC__
        const bool hasQDescale = params.qDescale != nullptr;
        const bool hasKDescale = params.kDescale != nullptr;
        const bool hasVDescale = params.vDescale != nullptr;
        AscendC::GlobalTensor<float> gQDescale;
        AscendC::GlobalTensor<float> gKDescale;
        AscendC::GlobalTensor<float> gVDescale;
        if (hasQDescale) {
            gQDescale.SetGlobalBuffer((__gm__ float*)params.qDescale);
        }
        if (hasKDescale) {
            gKDescale.SetGlobalBuffer((__gm__ float*)params.kDescale);
        }
        if (hasVDescale) {
            gVDescale.SetGlobalBuffer((__gm__ float*)params.vDescale);
        }
#endif
        // tiling data

        batch_ = faiTilingData->batch;
        qHeads_ = faiTilingData->numHeads;
        kvHeads_ = faiTilingData->kvHeads;
        firstBatchTaskNum_ = faiTilingData->firstBatchTaskNum;
        totalTaskNum_ = faiTilingData->totalTaskNum;
        scaleValue_ = params.scaleValue;
        softcapValue_ = params.softcapValue;
        maxNumBlocksPerBatch_ = params.maxNumBlocksPerBatch;
        blockSize_ = faiTilingData->blockSize;
        numBlocks_ = params.numBlocks;
        windowSizeLeft_ = faiTilingData->windowSizeLeft;
        windowSizeRight_ = faiTilingData->windowSizeRight;
        flashDecodeFlag_ = faiTilingData->flashDecodeFlag;
        fdCombineTaskNum_ = faiTilingData->fdCombineTaskNum;
        fdRowCapacity_ = faiTilingData->fdRowCapacity;
        fdLseSubStride_ = faiTilingData->fdLseSubStride;

        AscendC::LocalTensor<ElementP> l1PTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_OTMP_BUF_STAGES];
        auto pvL1AddrStart_ = qkL1TileM_ * qkL1TileKLeft_ * qL1BufNum_ * sizeof(ElementQ) +
                              qkL1TileKRight_ * qkL1TileN_ * kL1BufNum_ * sizeof(ElementK);
        for (uint32_t i = 0; i < pL1BufNum_; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementP>(
                pvL1AddrStart_ + pvL1TileM_ * pvL1TileKLeft_ * sizeof(ElementP) * i);
        }
        CalcUBufTileInfo();
        for (uint32_t i = 0; i < UB_S_BUF_STAGES; ++i) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ub_.sStartOffset + i * ub_.sBufferBytes);
        }
        ubOTmpTensor[0] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(ub_.loStartOffset);

        Gemm::Block::BlockMmadQKTileHelper qkL1TileHelper(qkL1TileM_, qkL1TileN_, qkL1TileKLeft_, qkL1TileKRight_,
                                                          qL1BufNum_, kL1BufNum_);
        Gemm::Block::BlockMmadPVTileHelper pvL1TileHelper(pvL1TileM_, pvL1TileN_, pvL1TileKLeft_, pvL1TileKRight_,
                                                          pL1BufNum_, vL1BufNum_);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        InitSyncFlags<4, 4, 4>();
#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
        BlockMmadQK blockMmadQK(resource, qkL1TileHelper);
        BlockMmadPV blockMmadPV(resource, pvL1AddrStart_, pvL1TileHelper);
#endif

#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, ub_, softcapValue_);
        EpilogueRescaleO epilogueRescaleO(resource, ub_);
        Epilogue::Block::InitOutputs950<ArchTag, ElementO> initOutputs(resource, ub_);
#endif

        int64_t strideQ = 0;
        int64_t strideO = 0;
        int64_t strideK = 0;
        int64_t strideV = 0;
        int64_t kvNumTokens;
        if constexpr (kvFormat == Format::TND && kvcacheType == CacheMode::normalCache) {
            kvNumTokens = gActualKvseqlen.GetValue(batch_);
        } else {
            kvNumTokens = gActualKvseqlen.GetValue(batch_ - 1);
        }
        // int64_t kvNumTokens = gActualKvseqlen.GetValue(batch_ - 1); // used for TND_NZ

        int64_t qHeadStride = embed_;
        int64_t kvHeadStrideK = embed_;
        int64_t kvHeadStrideV = embedV_;
        if constexpr (qFormat == Format::BNSD) {
            strideQ = embed_;
            qHeadStride = static_cast<int64_t>(faiTilingData->maxQSeqlen) * embed_;
        } else {
            strideQ = qHeads_ * embed_;
        }
        if constexpr (kvFormat == Format::BNSD) {
            strideK = embed_;
            strideV = embedV_;
            kvHeadStrideK = static_cast<int64_t>(params.kvSeqlenAllocated) * embed_;
            kvHeadStrideV = static_cast<int64_t>(params.kvSeqlenAllocated) * embedV_;
        } else {
            strideK = kvHeads_ * embed_;
            strideV = kvHeads_ * embedV_;
        }
        strideO = qHeads_ * embedV_;

        uint32_t embedVRound = RoundUp(embedV_, 16);
        uint32_t groupSize = qHeads_ / kvHeads_;
        int64_t qBOffset = 0;
        int64_t kBOffset = 0;
        int64_t vBOffset = 0;
        int64_t oBOffset = 0;
        int64_t blockBOffset = 0;
        uint32_t preTotalTaskNum = 0;
        uint32_t curBatch = 0;
        uint32_t cachedBatch = 0xFFFFFFFFU;
        // Used lengths drive task counts; physical (region) lengths drive offsets.
        int64_t qSeqlen = GetQSeqLen(curBatch, faiTilingData->maxQSeqlen, gActualQseqlen, gSeqUsedQ);
        int64_t qPhysicalSeqlen = faiTilingData->maxQSeqlen;
        int64_t kvSeqlen = 0;
        int64_t kvPhysicalSeqlen = gActualKvseqlen.GetValue(curBatch);
        uint32_t curTotalTaskNum = firstBatchTaskNum_;
        if constexpr (qFormat == Format::TND) {
            qPhysicalSeqlen =
                static_cast<int64_t>(gActualQseqlen.GetValue(curBatch + 1) - gActualQseqlen.GetValue(curBatch));
        }
        if constexpr (kvFormat == Format::TND && kvcacheType == CacheMode::normalCache) {
            kvPhysicalSeqlen =
                static_cast<int64_t>(gActualKvseqlen.GetValue(curBatch + 1) - gActualKvseqlen.GetValue(curBatch));
        }
        uint32_t taskBegin = coreIdx;
        uint32_t taskEnd = totalTaskNum_;
        uint32_t taskStep = coreNum;
        int32_t fdBaseTaskStart = -1;
        int32_t fdBaseTaskEnd = -1;
        int32_t fdFirstKvTileStart = -1;
        int32_t fdLastKvTileEnd = -1;
        if (flashDecodeFlag_ != 0U) {
            fdBaseTaskStart = faiTilingData->fdDecodeSchedules[coreIdx].baseTaskStart;
            fdBaseTaskEnd = faiTilingData->fdDecodeSchedules[coreIdx].baseTaskEnd;
            fdFirstKvTileStart = faiTilingData->fdDecodeSchedules[coreIdx].firstKvTileStart;
            fdLastKvTileEnd = faiTilingData->fdDecodeSchedules[coreIdx].lastKvTileEnd;
            taskBegin = static_cast<uint32_t>(fdBaseTaskStart);
            taskEnd = static_cast<uint32_t>(fdBaseTaskEnd);
            taskStep = 1U;
        }
        uint32_t pipelineTileBase = 0U;
        uint64_t pipelineL0AStageBase = 0U;
        uint64_t pipelineL0BStageBase = 0U;
        for (uint32_t taskIdx = taskBegin; taskIdx < taskEnd; taskIdx += taskStep) {
            while (taskIdx >= curTotalTaskNum) {
                ++curBatch;
                preTotalTaskNum = curTotalTaskNum;
                if constexpr (qFormat == Format::BNSD) {
                    qBOffset += static_cast<int64_t>(qHeads_) * qHeadStride;
                } else {
                    qBOffset += qPhysicalSeqlen * strideQ;
                }
                if constexpr (kvcacheType == CacheMode::normalCache) {
                    if constexpr (kvFormat == Format::BNSD) {
                        kBOffset += static_cast<uint64_t>(kvHeads_) * kvHeadStrideK;
                        vBOffset += static_cast<uint64_t>(kvHeads_) * kvHeadStrideV;
                    } else {
                        kBOffset += static_cast<uint64_t>(kvPhysicalSeqlen * strideK);
                        vBOffset += static_cast<uint64_t>(kvPhysicalSeqlen * strideV);
                    }
                } else {
                    blockBOffset += static_cast<uint64_t>(maxNumBlocksPerBatch_);
                }
                oBOffset += qPhysicalSeqlen * strideO;

                qSeqlen = GetQSeqLen(curBatch, faiTilingData->maxQSeqlen, gActualQseqlen, gSeqUsedQ);
                qPhysicalSeqlen = faiTilingData->maxQSeqlen;
                kvPhysicalSeqlen = gActualKvseqlen.GetValue(curBatch);
                if constexpr (qFormat == Format::TND) {
                    qPhysicalSeqlen =
                        static_cast<int64_t>(gActualQseqlen.GetValue(curBatch + 1) - gActualQseqlen.GetValue(curBatch));
                }
                if constexpr (kvFormat == Format::TND && kvcacheType == CacheMode::normalCache) {
                    kvPhysicalSeqlen = static_cast<int64_t>(gActualKvseqlen.GetValue(curBatch + 1) -
                                                            gActualKvseqlen.GetValue(curBatch));
                }
                uint32_t curQNBlockTile = GetQNBlockTile(static_cast<uint32_t>(qSeqlen), groupSize, embedV_ > 128U);
                uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                uint32_t qNTaskNum = qNBlockNumPerGroup * kvHeads_;
                curTotalTaskNum += (qSeqlen + qBaseTile_ - 1) / qBaseTile_ * qNTaskNum;
            }
            if (curBatch != cachedBatch) {
                kvSeqlen = GetKvSeqLen(curBatch, gActualKvseqlen, gSeqUsedKv);
                cachedBatch = curBatch;
            }
            uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
            uint32_t qSTileIdx;
            uint32_t kvHeadIdx;
            uint32_t qNStartIdx;
            uint32_t qNBlockSize;
            uint32_t curQNBlockTile = GetQNBlockTile(static_cast<uint32_t>(qSeqlen), groupSize, embedV_ > 128U);
            uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
            uint32_t qNTaskNum = qNBlockNumPerGroup * kvHeads_;
            qSTileIdx = taskIdxCurBatch / qNTaskNum;
            uint32_t qNBlockIdx = taskIdxCurBatch - qSTileIdx * qNTaskNum;
            uint32_t qNBlockIdxInGroup = qNBlockIdx % qNBlockNumPerGroup;
            kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup;
            qNStartIdx = kvHeadIdx * groupSize + qNBlockIdxInGroup * curQNBlockTile;
            qNBlockSize = AscendC::Std::min(curQNBlockTile, groupSize - qNBlockIdxInGroup * curQNBlockTile);
#ifdef __DAV_VEC__
            const uint32_t descaleOffset = curBatch * kvHeads_ + kvHeadIdx;
            const float qDequantScale = hasQDescale ? gQDescale.GetValue(descaleOffset) : 1.0f;
            const float kDequantScale = hasKDescale ? gKDescale.GetValue(descaleOffset) : 1.0f;
            const float vDequantScale = hasVDescale ? gVDescale.GetValue(descaleOffset) : 1.0f;
            const float scoreScale = scaleValue_ * qDequantScale * kDequantScale;
#endif
            int64_t gmOffsetQ = 0;
            int64_t gmOffsetK = 0;
            int64_t gmOffsetV = 0;
            int64_t gmOffsetO = 0;
            int64_t qSOffset = qSTileIdx * qBaseTile_;
            gmOffsetQ = qBOffset + qSOffset * strideQ + qNStartIdx * qHeadStride;
            gmOffsetO = oBOffset + qSOffset * strideO + qNStartIdx * embedV_;
            uint32_t lseHeadStride = static_cast<uint32_t>(faiTilingData->maxQSeqlen);
            uint64_t lseOffset = static_cast<uint64_t>(qBOffset / strideQ) * qHeads_ +
                                 static_cast<uint64_t>(qNStartIdx) * lseHeadStride + qSOffset;
            if constexpr (qFormat == Format::TND) {
                lseHeadStride = static_cast<uint32_t>(gActualQseqlen.GetValue(batch_));
                lseOffset = static_cast<uint64_t>(qNStartIdx) * lseHeadStride +
                            static_cast<uint64_t>(qBOffset / strideQ + qSOffset);
            } else if constexpr (qFormat == Format::BNSD) {
                lseOffset = static_cast<uint64_t>(curBatch) * qHeads_ * lseHeadStride +
                            static_cast<uint64_t>(qNStartIdx) * lseHeadStride + qSOffset;
            }
            if constexpr (kvcacheType == CacheMode::pagedCache && kvcacheShape == PageShape::BnNBsD) {
                strideK = embed_;
                strideV = embedV_;
                gmOffsetK = static_cast<uint64_t>(kvHeadIdx) * embed_ * blockSize_;
                gmOffsetV = static_cast<uint64_t>(kvHeadIdx) * embedV_ * blockSize_;
            } else if (cacheLayout == CacheLayout::nd) {
                gmOffsetK = kBOffset + static_cast<uint64_t>(kvHeadIdx) * kvHeadStrideK;
                gmOffsetV = vBOffset + static_cast<uint64_t>(kvHeadIdx) * kvHeadStrideV;
            } else {
                gmOffsetK = kBOffset + static_cast<uint64_t>(kvHeadIdx) * kvHeadStrideK * kvNumTokens;
                gmOffsetV = vBOffset + static_cast<uint64_t>(kvHeadIdx) * kvHeadStrideV * kvNumTokens;
            }
            uint32_t qsBlockNum = (qSeqlen + qBaseTile_ - 1) / qBaseTile_;
            uint32_t qSBlockSize = qSTileIdx == qsBlockNum - 1 ? qSeqlen - (qsBlockNum - 1) * qBaseTile_ : qBaseTile_;
            uint32_t rowNum = qSBlockSize * qNBlockSize;
            uint32_t kvSTileSizeAct = kvBaseTile_;

            uint32_t kvStart = 0;
            uint32_t noSkipKvS = static_cast<uint32_t>(kvSeqlen);
            uint32_t kvSLoopNum = 0;
            int64_t windowSizeLeftStartLen = 0;
            int64_t windowSizeLeftEndLen = 0;
            int64_t windowSizeRightStartLen = 0;
            int64_t windowSizeRightEndLen = 0;
            int32_t delStartRow = 0;
            int32_t delEndRow = static_cast<int32_t>(qSeqlen);
            const int32_t qSTileStart = static_cast<int32_t>(qSTileIdx * qBaseTile_);
            if constexpr (maskCategory == NO_MASK) {
                kvSLoopNum = static_cast<uint32_t>(CeilDiv(noSkipKvS, static_cast<int64_t>(kvBaseTile_)));
            } else if constexpr (maskCategory == CAUSAL_MASK) {
                int64_t causalKvEnd = static_cast<int64_t>(qSTileIdx + 1U) * qBaseTile_ + kvSeqlen - qSeqlen;
                if (causalKvEnd < 0) {
                    causalKvEnd = 0;
                }
                noSkipKvS = AscendC::Std::min((uint32_t)kvSeqlen, (uint32_t)causalKvEnd);
                kvSLoopNum = static_cast<uint32_t>(CeilDiv(noSkipKvS, static_cast<int64_t>(kvBaseTile_)));
            } else if constexpr (maskCategory == SLIDING_WINDOW_MASK) {
                // Use the actual ragged lengths and signed bounds before clipping.
                // Negative windows are unlimited; Q and K have different bounds.
                const int64_t qLength = static_cast<int64_t>(qSeqlen);
                const int64_t kvLength = static_cast<int64_t>(kvSeqlen);
                const int64_t seqDiff = kvLength - qLength;
                const int64_t qBlockBase = static_cast<int64_t>(qSTileIdx) * qBaseTile_;
                const bool notPreMask = windowSizeLeft_ < 0 || windowSizeLeft_ >= kvLength - 1;
                const bool notNextMask = windowSizeRight_ < 0 || windowSizeRight_ >= qLength - 1;
                if (!notPreMask) {
                    const int64_t leftOrigin = seqDiff - windowSizeLeft_;
                    windowSizeLeftStartLen = qBlockBase + leftOrigin;
                    windowSizeLeftEndLen = windowSizeLeftStartLen + qSBlockSize;
                    const int64_t firstKey = AscendC::Std::min(kvLength,
                        AscendC::Std::max(int64_t{0}, windowSizeLeftStartLen));
                    kvStart = static_cast<uint32_t>(firstKey) / kvBaseTile_;
                    if (windowSizeLeftEndLen > kvLength) {
                        delStartRow = static_cast<int32_t>(AscendC::Std::min(qLength,
                            AscendC::Std::max(int64_t{0}, kvLength - leftOrigin)));
                    }
                }
                if (!notNextMask) {
                    const int64_t rightOrigin = seqDiff + windowSizeRight_;
                    windowSizeRightStartLen = qBlockBase + rightOrigin;
                    // Exclusive upper bound for the last Q row in this tile.
                    windowSizeRightEndLen = windowSizeRightStartLen + qSBlockSize;
                    const int64_t endKey = AscendC::Std::min(kvLength,
                        AscendC::Std::max(int64_t{0}, windowSizeRightEndLen));
                    noSkipKvS = static_cast<uint32_t>(endKey);
                    if (windowSizeRightStartLen < 0) {
                        delEndRow = static_cast<int32_t>(AscendC::Std::min(qLength,
                            AscendC::Std::max(int64_t{0}, -rightOrigin)));
                    }
                }
                kvSLoopNum = static_cast<uint32_t>(CeilDiv(noSkipKvS, static_cast<int64_t>(kvBaseTile_)));
            }

            uint32_t fullyMaskedRowsPerHead = 0;
            if constexpr (maskCategory == CAUSAL_MASK) {
                fullyMaskedRowsPerHead = noSkipKvS < qSBlockSize ? qSBlockSize - noSkipKvS : 0;
            }

            bool emptySpan = (kvSLoopNum == 0);
            if constexpr (maskCategory == SLIDING_WINDOW_MASK) {
                emptySpan = emptySpan || (kvStart >= kvSLoopNum);
            }
            if (emptySpan) {
#ifdef __DAV_VEC__
                initOutputs.template operator()<LseMode>(gO[gmOffsetO], gLse[lseOffset], qSBlockSize, qNBlockSize,
                                                         lseHeadStride, embedV_, static_cast<uint32_t>(strideO),
                                                         embedV_);
#endif
                continue;
            }
            uint32_t kvSplitStart = kvStart;
            uint32_t kvSplitEnd = kvSLoopNum;
            bool writePartial = false;
            uint32_t partialId = 0U;
            if (flashDecodeFlag_ != 0U) {
                if (taskIdx == static_cast<uint32_t>(fdBaseTaskStart)) {
                    kvSplitStart = static_cast<uint32_t>(fdFirstKvTileStart);
                }
                if (taskIdx + 1U == static_cast<uint32_t>(fdBaseTaskEnd)) {
                    kvSplitEnd = static_cast<uint32_t>(fdLastKvTileEnd);
                }
                kvSplitEnd = AscendC::Std::min(kvSplitEnd, kvSLoopNum);
                kvSplitStart = AscendC::Std::min(kvSplitStart, kvSplitEnd);
                for (uint32_t combineIdx = 0; combineIdx < fdCombineTaskNum_; ++combineIdx) {
                    const int32_t combineBaseTask = faiTilingData->fdCombineSchedules[combineIdx].baseTask;
                    if (combineBaseTask == static_cast<int32_t>(taskIdx)) {
                        writePartial = true;
                        partialId = static_cast<uint32_t>(faiTilingData->fdCombineSchedules[combineIdx].partialStart +
                                                          static_cast<int32_t>(coreIdx) -
                                                          faiTilingData->fdCombineSchedules[combineIdx].firstCore);
                        break;
                    }
                }
            }
            const uint32_t kvSLoopCount = kvSplitEnd - kvSplitStart;

#ifdef __DAV_CUBE__
            uint32_t qShapeCol = strideQ;
            uint32_t kShapeCol = strideK;
            uint32_t vShapeCol = strideV;
            auto gmQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(qShapeCol, qBaseTile_);
            auto gmQTensorTla = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTla, Arch::PositionGM{});

            GemmCoord actualBlockShapeQ{rowNum, embed_, 0};
            blockMmadQK.loadQGM(gmQTensorTla, actualBlockShapeQ, qSBlockSize, qNBlockSize, qHeadStride);
            uint32_t kShapeRow = 0;
            if constexpr (kvcacheType == CacheMode::pagedCache) {
                kShapeRow = numBlocks_ * blockSize_;
            } else {
                kShapeRow = kvSeqlen;
            }
            auto gmKLayoutTla = tla::MakeLayout<ElementK, LayoutK>(kShapeRow, kShapeCol);
            auto gmKTensorTla = tla::MakeTensor(gK[gmOffsetK], gmKLayoutTla, Arch::PositionGM{});
            auto gmVLayoutTla = tla::MakeLayout<ElementV, LayoutV>(kShapeRow, vShapeCol);
            auto gmVTensorTla = tla::MakeTensor(gV[gmOffsetV], gmVLayoutTla, Arch::PositionGM{});
#endif
#ifdef __DAV_VEC__
            uint32_t oShapeCol = strideO;
            auto gmOLayoutTla = tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, oShapeCol);
            auto gmOTensorTla = tla::MakeTensor(gO[gmOffsetO], gmOLayoutTla, Arch::PositionGM{});
#endif
            for (uint32_t kvSTileRelIdx = 0; kvSTileRelIdx < kvSLoopCount + PRE_LAUNCH; ++kvSTileRelIdx) {
                const uint32_t kvSTileIdx = kvSplitStart + kvSTileRelIdx;
                const uint32_t pipelineTileSeq =
                    flashDecodeFlag_ != 0U ? pipelineTileBase + kvSTileRelIdx : kvSTileRelIdx;
                if (kvSTileRelIdx < kvSLoopCount) {
                    if (kvSTileIdx == kvSLoopNum - 1) {
                        kvSTileSizeAct = noSkipKvS - kvSTileIdx * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};
                    uint32_t ubSBufId = pipelineTileSeq % UB_S_BUF_STAGES;
                    auto ubSLayoutTla = tla::MakeLayout<ElementS, LayoutS>(RoundUp(kvSTileSizeAct, CUBE_MATRIX_BLOCK),
                                                                           qBaseTile_ / VECTOR_SUBCORE_NUM);
                    auto ubSTensorTla = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayoutTla, Arch::PositionUB{});
                    uint32_t qkReadyFlagId = ubSBufId;
                    Arch::CrossCoreFlag qkReadyFlag(qkReadyFlagId);
#ifdef __DAV_CUBE__
                    uint64_t prefixSumL0AStages =
                        CalcCrossqkpvPrefixSumL0ABStages(kvSTileRelIdx, qkL0ATotalStages_, pvL0ATotalStages_,
                                                         kvSLoopCount, true) +
                        (flashDecodeFlag_ != 0U ? pipelineL0AStageBase : 0U);
                    uint64_t prefixSumL0BStages =
                        CalcCrossqkpvPrefixSumL0ABStages(kvSTileRelIdx, qkL0BTotalStages_, pvL0BTotalStages_,
                                                         kvSLoopCount, true) +
                        (flashDecodeFlag_ != 0U ? pipelineL0BStageBase : 0U);
                    blockMmadQK(gmKTensorTla, ubSTensorTla, gBlockTable[blockBOffset], actualBlockShapeQK, blockSize_,
                                kvSTileIdx, kvSTileRelIdx, 0, kvHeads_, kvNumTokens, kvBaseTile_, 0, 0, 0, qSBlockSize,
                                qNBlockSize, qkReadyFlag, prefixSumL0AStages, prefixSumL0BStages);
                    if (kvSTileRelIdx == kvSLoopCount - 1U) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
                    }
#endif
                    uint32_t l1PBufId = pipelineTileSeq % pL1BufNum_;
                    uint32_t softmaxReadyFlagId = l1PBufId + UB_S_BUF_STAGES;
                    Arch::CrossCoreFlag softmaxReadyFlag(softmaxReadyFlagId);
                    auto l1PLayoutTla =
                        tla::MakeLayout<ElementP, typename BlockMmadPV::LayoutTagL1A>(qBaseTile_, kvSTileSizeAct);
                    auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId], l1PLayoutTla, Arch::PositionL1{});
#ifdef __DAV_VEC__
                    epilogueOnlineSoftmax(l1PTensorTla, rowNum, qSBlockSize, qSTileStart, kvSTileIdx * kvBaseTile_,
                                          kvSTileSizeAct, qSeqlen, kvSeqlen, windowSizeLeft_, windowSizeRight_,
                                          scoreScale, kvSTileRelIdx == 0, ubSBufId, l1PBufId, qkReadyFlag,
                                          softmaxReadyFlag);
#endif
                }
                if (kvSTileRelIdx >= PRE_LAUNCH) {
                    uint32_t kvSTileRelIdxNow = kvSTileRelIdx - PRE_LAUNCH;
                    uint32_t kvSTileIdxNow = kvSplitStart + kvSTileRelIdxNow;
                    uint32_t pipelineTileSeqNow =
                        flashDecodeFlag_ != 0U ? pipelineTileBase + kvSTileRelIdxNow : kvSTileRelIdxNow;
                    if (kvSTileIdxNow == kvSLoopNum - 1) {
                        kvSTileSizeAct = noSkipKvS - kvSTileIdxNow * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }
                    GemmCoord actualBlockShapePV{qBaseTile_, embedV_, kvSTileSizeAct};
                    uint32_t ubOTmpBufId = pipelineTileSeqNow % UB_OTMP_BUF_STAGES;
                    uint32_t pvReadyFlagId = ubOTmpBufId + UB_S_BUF_STAGES + pL1BufNum_;
#ifdef __DAV_CUBE__
                    uint32_t l1PBufId = pipelineTileSeqNow % pL1BufNum_;
                    auto ubOTmpLayoutTla = tla::MakeLayout<ElementOTmp, LayoutOTmp>(qBaseTile_, embedVRound);
                    auto ubOTmpTensorTla =
                        tla::MakeTensor(ubOTmpTensor[ubOTmpBufId], ubOTmpLayoutTla, Arch::PositionUB{});
                    uint32_t softmaxReadyFlagId = l1PBufId + UB_S_BUF_STAGES;
                    Arch::CrossCoreFlag softmaxReadyFlag(softmaxReadyFlagId);
                    Arch::CrossCoreFlag pvReadyFlag(pvReadyFlagId);
                    uint64_t prefixSumL0AStages =
                        CalcCrossqkpvPrefixSumL0ABStages(kvSTileRelIdxNow, qkL0ATotalStages_, pvL0ATotalStages_,
                                                         kvSLoopCount, false) +
                        (flashDecodeFlag_ != 0U ? pipelineL0AStageBase : 0U);
                    uint64_t prefixSumL0BStages =
                        CalcCrossqkpvPrefixSumL0ABStages(kvSTileRelIdxNow, qkL0BTotalStages_, pvL0BTotalStages_,
                                                         kvSLoopCount, false) +
                        (flashDecodeFlag_ != 0U ? pipelineL0BStageBase : 0U);
                    blockMmadPV(gmVTensorTla, ubOTmpTensorTla, gBlockTable[blockBOffset], actualBlockShapePV,
                                blockSize_, kvSTileIdxNow, pipelineTileSeqNow, 0, kvHeads_, kvNumTokens, kvBaseTile_, 0,
                                0, 0, softmaxReadyFlag, pvReadyFlag, prefixSumL0AStages, prefixSumL0BStages);
#endif
#ifdef __DAV_VEC__
                    Arch::CrossCoreFlag pvReadyFlag(pvReadyFlagId);
                    uint32_t curTileMod = pipelineTileSeqNow % (PRE_LAUNCH + 1);
                    uint32_t fmRowsPerHead = fullyMaskedRowsPerHead;
                    int32_t swaDelStartRow = 0;
                    int32_t swaDelEndRow = 0;
                    uint32_t swaQSeqlen = 0;
                    uint32_t swaQSTileStart = 0;
                    if constexpr (maskCategory == SLIDING_WINDOW_MASK) {
                        fmRowsPerHead = 0U;
                        swaDelStartRow = delStartRow;
                        swaDelEndRow = delEndRow;
                        swaQSeqlen = static_cast<uint32_t>(qSeqlen);
                        swaQSTileStart = static_cast<uint32_t>(qSTileStart);
                    }
                    epilogueRescaleO.template operator()<LseMode>(
                        gmOTensorTla, gLse[lseOffset], actualBlockShapePV, curTileMod, pipelineTileSeqNow,
                        (kvSTileRelIdxNow == 0), (kvSTileRelIdxNow == kvSLoopCount - 1), pvReadyFlag, true,
                        fmRowsPerHead, qSBlockSize, qNBlockSize, lseHeadStride, static_cast<uint32_t>(strideO), embedV_,
                        swaDelStartRow, swaDelEndRow, swaQSeqlen, swaQSTileStart, gPartialO, gPartialLse, writePartial,
                        partialId, fdRowCapacity_, fdLseSubStride_, vDequantScale);
#endif
                }
            }
            pipelineTileBase += kvSLoopCount;
            pipelineL0AStageBase += static_cast<uint64_t>(kvSLoopCount) * (qkL0ATotalStages_ + pvL0ATotalStages_);
            pipelineL0BStageBase += static_cast<uint64_t>(kvSLoopCount) * (qkL0BTotalStages_ + pvL0BTotalStages_);
        }
        ReleaseSyncFlags<4, 4, 4>();
        if constexpr (qFormat == Format::TND && kvcacheType == CacheMode::pagedCache &&
                      maskCategory != SLIDING_WINDOW_MASK) {
            if (flashDecodeFlag_ != 0U) {
                // Every producer joins before combine reuses the main kernel's UB.
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SyncAll<false>();
#ifdef __DAV_VEC__
                FlashDecodeCombine950<ElementO, LseMode, 1> combine(resource);
                combine(params);
#endif
            }
        }
    }

    template <uint32_t QK_SM_MODE, uint32_t PV_RE_MODE, uint32_t SM_PV_MODE>
    __aicore__ inline void InitSyncFlags()
    {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        // Query
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        // Key
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        // Value
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        // L0A
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        // L0B
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        // L0C
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        // cross core sync
        if constexpr (SM_PV_MODE == 4U) {
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(2);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(18);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(3);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(19);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(4);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(20);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        // mask2index
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        // softmax
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        // mask
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID5);
        // rescale
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (QK_SM_MODE == 4U) {
            AscendC::CrossCoreSetFlag<QK_SM_MODE, PIPE_V>(0);
            AscendC::CrossCoreSetFlag<QK_SM_MODE, PIPE_V>(1);
        }
        if constexpr (PV_RE_MODE == 4U) {
            AscendC::CrossCoreSetFlag<PV_RE_MODE, PIPE_V>(5);
        }
#endif
    }

    template <uint32_t QK_SM_MODE, uint32_t PV_RE_MODE, uint32_t SM_PV_MODE>
    __aicore__ inline void ReleaseSyncFlags()
    {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        if constexpr (QK_SM_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<QK_SM_MODE, PIPE_FIX>(0);
            AscendC::CrossCoreWaitFlag<QK_SM_MODE, PIPE_FIX>(1);
            AscendC::CrossCoreWaitFlag<QK_SM_MODE, PIPE_FIX>(16);
            AscendC::CrossCoreWaitFlag<QK_SM_MODE, PIPE_FIX>(17);
        }
        if constexpr (PV_RE_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<PV_RE_MODE, PIPE_FIX>(5);
            AscendC::CrossCoreWaitFlag<PV_RE_MODE, PIPE_FIX>(21);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (SM_PV_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<SM_PV_MODE, PIPE_MTE3>(2);
            AscendC::CrossCoreWaitFlag<SM_PV_MODE, PIPE_MTE3>(3);
            AscendC::CrossCoreWaitFlag<SM_PV_MODE, PIPE_MTE3>(4);
        }
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CalcUBufTileInfo()
    {
        constexpr uint32_t qBaseTilePerSubCore = qBaseTile_ / VECTOR_SUBCORE_NUM;
        constexpr uint32_t kvBaseTilePerSubCore = CeilDiv(kvBaseTile_, MATRIX_ALIGNMENT) * MATRIX_ALIGNMENT;
        constexpr uint32_t embedPerSubCore = CeilDiv(embedV_, MATRIX_ALIGNMENT) * MATRIX_ALIGNMENT;
        constexpr uint32_t pExtraElemNum =
            qBaseTilePerSubCore > kvBaseTilePerSubCore ? qBaseTilePerSubCore : kvBaseTilePerSubCore;
        constexpr uint32_t sBufferBytes = qBaseTilePerSubCore * kvBaseTilePerSubCore * sizeof(ElementS);
        constexpr uint32_t pBufferBytes =
            (qBaseTilePerSubCore * kvBaseTilePerSubCore + pExtraElemNum) * sizeof(ElementP);
        constexpr uint32_t loBufferBytes = qBaseTilePerSubCore * embedPerSubCore * sizeof(ElementOTmp);

        constexpr uint32_t sStartOffset = 0;
        constexpr uint32_t pStartOffset = sStartOffset + sBufferBytes * UB_S_BUF_STAGES;
        constexpr uint32_t loStartOffset = pStartOffset + pBufferBytes * EpilogueOnlineSoftmax::UB_S_P_BUF_STAGES;
        constexpr uint32_t goStartOffset = loStartOffset + loBufferBytes * UB_OTMP_BUF_STAGES;
        constexpr uint32_t lmStartOffset = goStartOffset + loBufferBytes;
        constexpr uint32_t gmStartOffset = lmStartOffset + qBaseTilePerSubCore * sizeof(ElementS);
        constexpr uint32_t dmStartOffset = gmStartOffset + qBaseTilePerSubCore * sizeof(ElementS);
        constexpr uint32_t glStartOffset =
            dmStartOffset + qBaseTilePerSubCore * MAX_CROSS_CORE_BUF_STAGES * sizeof(ElementS);
        constexpr uint32_t lseStartOffset = glStartOffset + qBaseTilePerSubCore * sizeof(ElementS);
        constexpr uint32_t indexStartOffset = lseStartOffset + qBaseTilePerSubCore * UB_BLOCK_BYTES;
        constexpr uint32_t totalBytes = indexStartOffset + VECTOR_REGISTER_BYTES;
        static_assert(totalBytes <= ArchTag::UB_SIZE, "FP8 UB layout exceeds Ascend950 capacity");

        ub_ = {sStartOffset,  pStartOffset,  loStartOffset, goStartOffset,  lmStartOffset,
               gmStartOffset, dmStartOffset, glStartOffset, lseStartOffset, indexStartOffset,
               totalBytes,    sBufferBytes,  pBufferBytes,  loBufferBytes,  UB_OTMP_BUF_STAGES};
    }

    __aicore__ inline uint64_t CalcCrossqkpvPrefixSumL0ABStages(uint32_t KvSTileIdx, uint32_t singleqkL0Stages,
                                                                uint32_t singlepvL0Stages, uint32_t kvSLoopNum,
                                                                bool isCurPhaseqk)
    {
        uint64_t prefixSumStages;
        if (isCurPhaseqk) {
            prefixSumStages = (KvSTileIdx <= PRE_LAUNCH)
                                  ? KvSTileIdx * singleqkL0Stages
                                  : KvSTileIdx * singleqkL0Stages + (KvSTileIdx - PRE_LAUNCH) * singlepvL0Stages;
        } else {
            prefixSumStages = (KvSTileIdx + PRE_LAUNCH < kvSLoopNum)
                                  ? (KvSTileIdx + 1 + PRE_LAUNCH) * singleqkL0Stages + KvSTileIdx * singlepvL0Stages
                                  : kvSLoopNum * singleqkL0Stages + KvSTileIdx * singlepvL0Stages;
        }
        return prefixSumStages;
    }

  private:
    Arch::Resource<ArchTag> resource;

    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    static constexpr uint32_t embed_ = HEAD_DIM;
    static constexpr uint32_t embedV_ = HEAD_DIM;
    uint32_t firstBatchTaskNum_;
    uint32_t totalTaskNum_;
    float scaleValue_;
    float softcapValue_;
    uint32_t maxNumBlocksPerBatch_;
    uint32_t blockSize_;
    uint32_t numBlocks_;
    uint32_t flashDecodeFlag_;
    uint32_t fdCombineTaskNum_;
    uint32_t fdRowCapacity_;
    uint32_t fdLseSubStride_;

    Epilogue::UBufTileHelper ub_;
    static constexpr uint32_t qBaseTile_ = Q_M_TILE_MAX;
    static constexpr uint32_t kvBaseTile_ = BlockMmadPV::L0_TILE_K;

    static constexpr uint32_t qkL1TileM_ = qBaseTile_;
    static constexpr uint32_t qkL1TileN_ = kvBaseTile_;
    static constexpr uint32_t qkL1TileKLeft_ = embed_;
    static constexpr uint32_t qkL1TileKRight_ = embed_;
    static constexpr uint32_t pvL1TileM_ = qBaseTile_;
    static constexpr uint32_t pvL1TileN_ = embedV_;
    static constexpr uint32_t pvL1TileKLeft_ = kvBaseTile_;
    static constexpr uint32_t pvL1TileKRight_ = kvBaseTile_;
    static constexpr uint32_t qL1BufNum_ = 1;
    static constexpr uint32_t kL1BufNum_ = 2;
    static constexpr uint32_t vL1BufNum_ = 2;
    static constexpr uint32_t pL1BufNum_ = 3;

    int64_t windowSizeLeft_ = 0;
    int64_t windowSizeRight_ = 0;

    static constexpr uint32_t qkL0ATotalStages_ =
        CeilDiv(qBaseTile_, BlockMmadQK::L0_TILE_M) * CeilDiv(embed_, BlockMmadQK::L0_TILE_K);
    static constexpr uint32_t qkL0BTotalStages_ =
        CeilDiv(kvBaseTile_, BlockMmadQK::L0_TILE_N) * CeilDiv(embed_, BlockMmadQK::L0_TILE_K);
    static constexpr uint32_t pvL0ATotalStages_ =
        CeilDiv(qBaseTile_, BlockMmadPV::L0_TILE_M) * CeilDiv(kvBaseTile_, BlockMmadPV::L0_TILE_K);
    static constexpr uint32_t pvL0BTotalStages_ =
        CeilDiv(kvBaseTile_, BlockMmadPV::L0_TILE_K) * CeilDiv(embedV_, BlockMmadPV::L0_TILE_N);
};

template <class InDtype, class SMDtype, Format qFormat, Format kvFormat, CacheMode kvcacheType, PageShape kvcacheShape,
          MaskCategory maskCategory, CacheLayout cacheLayout, bool LseMode, bool HasSoftcap>
CATLASS_GLOBAL void FAInferDnQuant(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o,
                                   GM_ADDR lse, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR seqUsedQ,
                                   GM_ADDR seqUsedKv, GM_ADDR workspace, GM_ADDR tiling, float scaleValue,
                                   float softcapValue, uint32_t numBlocks, uint32_t maxNumBlocksPerBatch,
                                   GM_ADDR qDescale, GM_ADDR kDescale, GM_ADDR vDescale, int64_t kvSeqlenAllocated)
{
    using ArchTag = Arch::Ascend950;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    static_assert(std::is_same_v<SMDtype, float>);
    using ElementP = InDtype;
    using ElementO = bfloat16_t;
    using ElementOTmp = float;
    using LayoutK = layout::RowMajor;
    using LayoutQ = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutPDummy = layout::nZ;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using L1TileShapeQK = Shape<Int<Q_M_TILE_MAX>, Int<Q_BLK>, Int<HEAD_DIM>>;
    using L0TileShapeQK = Shape<Int<Q_M_TILE_MAX>, Int<Q_BLK>, Int<HEAD_DIM>>;
    using DispatchPolicyQK =
        Gemm::MmadFlashAttentionQK<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, true>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementK, LayoutK, ElementQ, LayoutQ, ElementS,
                                                         LayoutS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_N>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementK, ElementQ,
                                                  ElementS, void, TileCopyQK>;
    // Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueFAQuantOnlineSoftmax<HasSoftcap, maskCategory>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<Q_M_TILE_MAX>, Int<HEAD_DIM>, Int<HEAD_DIM>>;
    using L0TileShapePV = Shape<Int<Q_M_TILE_MAX>, Int<HEAD_DIM>, Int<Q_BLK>>;
    using DispatchPolicyPV =
        Gemm::MmadFlashAttentionPV<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, true>;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV>;
    // rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueFARescaleO;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS,
                                                            ElementV, TileCopyRescaleO, Arch::PositionL0C>;

    using Kernel = FAIKernel950<BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, qFormat, kvFormat,
                                kvcacheType, kvcacheShape, maskCategory, cacheLayout, LseMode>;
    FAIKernelParams params{q,
                           k,
                           v,
                           mask,
                           blockTables,
                           actualQseqlen,
                           actualKvseqlen,
                           seqUsedQ,
                           seqUsedKv,
                           o,
                           lse,
                           workspace,
                           tiling,
                           scaleValue,
                           softcapValue,
                           numBlocks,
                           maxNumBlocksPerBatch,
                           kvSeqlenAllocated};
    params.qDescale = qDescale;
    params.kDescale = kDescale;
    params.vDescale = vDescale;
    Kernel faInfer;
    faInfer(params);
}

} // namespace QuantFA
