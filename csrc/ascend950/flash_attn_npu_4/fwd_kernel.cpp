/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Modified by Minghua Shen, 2026.
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
#include "fwd_resource.hpp"
#include "fwd_resource_plan.h"
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_SOFTMAX_HIGH_PREC_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_RESCALE_O_HPP
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"

#include "tla/tensor.hpp"
#include "tla/layout.hpp"

#include "kernel_common.hpp"
#include "fa_block.h"
#include "rescale_o.hpp"
#include "online_softmax.hpp"
#include "init_outputs.hpp"
#include "qk_matmul.hpp"
#include "qk_matmul_DN.hpp"
#include "pv_matmul.hpp"
#include "tilingdata.h"
#include "fd_combine.hpp"

using namespace Catlass;
using namespace tla;

static constexpr uint32_t PRE_LAUNCH = 2;
static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
static constexpr uint32_t UB_S_OTMP_BUF_STAGES = 2;
static constexpr uint32_t UB_S_BUF_STAGES = 2;
// One descriptor per in-flight kv tile: tile i is described when its QK is
// issued and consumed PRE_LAUNCH steps later, possibly inside a later task.
static constexpr uint32_t PV_DESC_SLOTS = PRE_LAUNCH + 2;

template <class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV, class EpilogueRescaleO, Format qFormat,
          Format kvFormat, CacheMode kvcacheType, PageShape kvcacheShape, MaskCategory maskCategory,
          CacheLayout cacheLayout, bool IsDN, bool LseMode>
class FAIKernel950 {
  public:
    using ArchTag = typename BlockMmadQK::ArchTag;

    using ElementQ = std::conditional_t<IsDN, typename BlockMmadQK::ElementB, typename BlockMmadQK::ElementA>;
    using ElementK = std::conditional_t<IsDN, typename BlockMmadQK::ElementA, typename BlockMmadQK::ElementB>;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename BlockMmadQK::ElementA;
    using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;

    using LayoutQ = std::conditional_t<IsDN, layout::ColumnMajor, layout::RowMajor>;
    using LayoutK = std::conditional_t<IsDN, layout::RowMajor, layout::ColumnMajor>;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutMask = layout::RowMajor;

    uint32_t otmpStages_ = 2U;
    bool hasSeqUsedQ = false;
    bool hasSeqUsedKv = false;

    __aicore__ inline FAIKernel950() {}

    // Parameters of one kv tile that must survive PV, RS
    struct PvTileDesc {
        // cube side: V matmul
        uint64_t gmOffsetV;
        uint64_t blockBOffset;
        uint32_t kShapeRow;
        uint32_t vShapeCol;
        uint32_t kvNumTokens;
        // vector side: rescale O and final lse
        uint64_t gmOffsetO;
        uint64_t lseOffset;
        uint32_t strideO;
        uint32_t oHeadStride;
        uint32_t lseHeadStride;
        // shared geometry / bookkeeping
        uint32_t kvSTileIdx;
        uint32_t kvSTileSizeAct;
        uint32_t rowNum;
        uint32_t rowNumRound;
        uint32_t pipelineTileSeq;
        uint32_t relIdx;
        uint32_t kvSLoopCount;
        uint32_t qSBlockSize;
        uint32_t qNBlockSize;
        // Cross-group fused tasks run one serial segment per KV head.
        uint32_t segmentRows;
        uint32_t segmentCount;
        int64_t vHeadStride;
        uint32_t fullyMaskedRowsPerHead;
        uint32_t statsSlot;
        uint32_t partialId;
        int32_t delStartRow;
        int32_t delEndRow;
        int64_t qSeqlen;
        int64_t qSTileStart;
        bool writePartial;
    };

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
        if constexpr (kvFormat == Format::BNSD) {
            // BNSD KV is indexed per (batch, head) with the allocated plane length;
            // never use the TND cumulative-length differencing.
            if (hasSeqUsedKv) {
                return static_cast<int64_t>(gSeqUsedKv.GetValue(bIdx));
            }
            return static_cast<int64_t>(gKvSeq.GetValue(bIdx));
        } else if constexpr (kvFormat == Format::TND) {
            if (hasSeqUsedKv) {
                return static_cast<int64_t>(gSeqUsedKv.GetValue(bIdx));
            }
            if constexpr (kvcacheType == CacheMode::normalCache) {
                return static_cast<int64_t>(gKvSeq.GetValue(bIdx + 1) - gKvSeq.GetValue(bIdx));
            }
            return static_cast<int64_t>(gKvSeq.GetValue(bIdx));
        } else {
            return static_cast<int64_t>(gKvSeq.GetValue(bIdx));
        }
    }

    template <bool FusedGroups = false>
    __aicore__ inline void operator()(FAIKernelParams const& params)
    {
        __gm__ const FAInferStaticTilingData* faiTilingData =
            reinterpret_cast<__gm__ const FAInferStaticTilingData*>(params.tiling);
        __gm__ const FAInferRuntimeData* runtimeTilingData =
            reinterpret_cast<__gm__ const FAInferRuntimeData*>(params.runtimeTiling);
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
        AscendC::GlobalTensor<ElementMask> gMask;
        gMask.SetGlobalBuffer((__gm__ ElementMask*)params.mask);
        //tiling data

        batch_ = faiTilingData->batch;
        qHeads_ = faiTilingData->numHeads;
        kvHeads_ = faiTilingData->kvHeads;
        embed_ = faiTilingData->embeddingSize;
        embedV_ = faiTilingData->embeddingSizeV;
        firstBatchTaskNum_ = faiTilingData->firstBatchTaskNum;
        totalTaskNum_ = faiTilingData->totalTaskNum;
        scaleValue_ = runtimeTilingData->scaleValue;
        softcapValue_ = runtimeTilingData->softcapValue;
        // base tile info
        qBaseTile_ = faiTilingData->qBaseTile;
        kvBaseTile_ = faiTilingData->kvBaseTile;
        // whether actual seqlen is provided
        actSeqAval_ = faiTilingData->actSeqAval;
        // aligned seqlen q & kv
        qSeqlenAligned_ = faiTilingData->qSeqlenAligned;
        kvSeqlenAligned_ = faiTilingData->kvSeqlenAligned;
        maxNumBlocksPerBatch_ = runtimeTilingData->maxNumBlocksPerBatch;
        blockSize_ = runtimeTilingData->blockSize;
        numBlocks_ = runtimeTilingData->numBlocks;
        qkL1TileM_ = faiTilingData->qkL1TileM;
        qkL1TileN_ = faiTilingData->qkL1TileN;
        qkL1TileKLeft_ = faiTilingData->qkL1TileKLeft;
        qkL1TileKRight_ = faiTilingData->qkL1TileKRight;
        pvL1TileM_ = faiTilingData->pvL1TileM;
        pvL1TileN_ = faiTilingData->pvL1TileN;
        pvL1TileKLeft_ = faiTilingData->pvL1TileKLeft;
        pvL1TileKRight_ = faiTilingData->pvL1TileKRight;
        qL1BufNum_ = faiTilingData->qL1BufNum;
        kL1BufNum_ = faiTilingData->kL1BufNum;
        vL1BufNum_ = faiTilingData->vL1BufNum;
        pL1BufNum_ = faiTilingData->pL1BufNum;
        windowSizeLeft_ = runtimeTilingData->windowSizeLeft;
        windowSizeRight_ = runtimeTilingData->windowSizeRight;
        flashDecodeFlag_ = faiTilingData->flashDecodeFlag != 0U && runtimeTilingData->allowFlashDecode != 0U;
        fdCombineTaskNum_ = faiTilingData->fdCombineTaskNum;
        fdRowCapacity_ = faiTilingData->fdRowCapacity;
        fdLseSubStride_ = faiTilingData->fdLseSubStride;

        AscendC::LocalTensor<ElementP> l1PTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_S_OTMP_BUF_STAGES];
        auto pvL1AddrStart_ = qkL1TileM_ * qkL1TileKLeft_ * qL1BufNum_ * sizeof(ElementQ) +
                              qkL1TileKRight_ * qkL1TileN_ * kL1BufNum_ * sizeof(ElementK);
        for (uint32_t i = 0; i < pL1BufNum_; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementP>(
                pvL1AddrStart_ + pvL1TileM_ * pvL1TileKLeft_ * sizeof(ElementP) * i);
        }
        const auto ubPlan = fa950_plan::KernelUbPlan(embedV_, qBaseTile_);
        otmpStages_ = ubPlan.otmpStages;
        for (uint32_t i = 0; i < UB_S_BUF_STAGES; ++i) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ubPlan.s + ubPlan.sBytes * i);
        }
        for (uint32_t i = 0; i < otmpStages_; ++i) {
            ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(ubPlan.lo + ubPlan.loBytes * i);
        }
        const uint32_t ubSRoundTile = 128;

        Gemm::Block::BlockMmadQKTileHelper qkL1TileHelper(qkL1TileM_, qkL1TileN_, qkL1TileKLeft_, qkL1TileKRight_,
                                                          qL1BufNum_, kL1BufNum_);
        Gemm::Block::BlockMmadPVTileHelper pvL1TileHelper(pvL1TileM_, pvL1TileN_, pvL1TileKLeft_, pvL1TileKRight_,
                                                          pL1BufNum_, vL1BufNum_);

        qkL0ATotalStages_ = CeilDiv(Q_M_TILE_MAX, BlockMmadQK::L0_TILE_M) * CeilDiv(embed_, BlockMmadQK::L0_TILE_K);
        qkL0BTotalStages_ = CeilDiv(kvBaseTile_, BlockMmadQK::L0_TILE_N) * CeilDiv(embed_, BlockMmadQK::L0_TILE_K);
        pvL0ATotalStages_ =
            CeilDiv(Q_M_TILE_MAX, BlockMmadPV::L0_TILE_M) * CeilDiv(kvBaseTile_, BlockMmadPV::L0_TILE_K);
        pvL0BTotalStages_ = CeilDiv(kvBaseTile_, BlockMmadPV::L0_TILE_K) * CeilDiv(embedV_, BlockMmadPV::L0_TILE_N);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        InitSyncFlags<4, 4, 4>();
#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
        BlockMmadQK blockMmadQK(resource, qkL1TileHelper);
        BlockMmadPV blockMmadPV(resource, pvL1AddrStart_, pvL1TileHelper);
        blockMmadQK.SetPageStride(params.storage.keyPageStride);
        blockMmadPV.SetPageStride(params.storage.valuePageStride);
#endif

        const bool d512 = fa950_plan::IsD512(embed_, embedV_) && qBaseTile_ == Q_M_TILE_MAX;
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, ubPlan, scaleValue_, softcapValue_, !d512);
        EpilogueRescaleO epilogueRescaleO(resource, ubPlan, flashDecodeFlag_ == 0U);
        // Plan-based buffers only for plans that carry a dedicated LSE scratch
        // (D512); every legacy plan keeps the established fixed offsets.
        Epilogue::Block::InitOutputs950<ArchTag, ElementO> initOutputs(resource, ubPlan);
#endif

        int64_t strideQ = 0;
        int64_t strideO = 0;
        int64_t strideK = 0;
        int64_t strideV = 0;
        // Per-head element distances. For BSND the head plane is D wide and the
        // token stride is H*D; for BNSD each head plane is S_alloc*D wide and
        // the token stride is D.
        int64_t qHeadStride = 0;
        int64_t oHeadStride = 0;
        int64_t kvHeadStrideK = 0;
        int64_t kvHeadStrideV = 0;
        int64_t kvNumTokens;
        if constexpr (kvFormat == Format::TND && kvcacheType == CacheMode::normalCache) {
            kvNumTokens = gActualKvseqlen.GetValue(batch_);
        } else {
            kvNumTokens = gActualKvseqlen.GetValue(batch_ - 1);
        }
        // int64_t kvNumTokens = gActualKvseqlen.GetValue(batch_ - 1); // used for TND_NZ

        if constexpr (qFormat == Format::BNSD) {
            strideQ = embed_;
            qHeadStride = static_cast<int64_t>(faiTilingData->maxQSeqlen) * embed_;
        } else {
            strideQ = qHeads_ * embed_;
            qHeadStride = embed_;
        }
        // Q/K/V are read with the input layout's strides (BNSD stays zero-copy),
        // but O is always addressed with BSND semantics so the returned tensor is
        // a contiguous (B, S, H, D) block regardless of the input layout. The
        // head plane is D wide and the token stride is H*D in BSND.
        strideO = qHeads_ * embedV_;
        oHeadStride = embedV_;
        kvHeadStrideK = embed_;
        kvHeadStrideV = embedV_;
        if constexpr (cacheLayout == CacheLayout::nd) {
            if constexpr (kvFormat == Format::BNSD) {
                strideK = embed_;
                strideV = embedV_;
                kvHeadStrideK = static_cast<int64_t>(params.kvSeqlenAllocated) * embed_;
                kvHeadStrideV = static_cast<int64_t>(params.kvSeqlenAllocated) * embedV_;
            } else {
                strideK = kvHeads_ * embed_;
                strideV = kvHeads_ * embedV_;
            }
        } else {
            strideK = 16;
            strideV = 16;
        }

        if constexpr (kvcacheType == CacheMode::pagedCache && cacheLayout == CacheLayout::nd) {
            strideK = params.storage.keyTokenStride;
            strideV = params.storage.valueTokenStride;
            kvHeadStrideK = params.storage.keyHeadStride;
            kvHeadStrideV = params.storage.valueHeadStride;
        }

        uint32_t embedVRound = RoundUp(embedV_, 16);
        uint32_t groupSize = qHeads_ / kvHeads_;
        // Exact D512 (Dq=Dk=Dv=512): single-slot Q/K/V L1, LO128/GO512 UB plan,
        // positional causal/SWA mask and four-piece PV delivery. Every branch
        // below keys on the tiling fields, which Host and AICPU write jointly.
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
        // Shared host/AICore decomposition: one geometry struct per batch, so
        // the task loop only decodes indexes instead of recomputing the tile
        // rules on every core and every task.
        const uint32_t batchCount = faiTilingData->batch;
        // Short-KV (single KV tile per task) MHA: merging distinct KV heads is
        // what pays off there, so allow cross-group fusion even for group == 1.
        // Must match host tiling / AICPU metadata exactly (task count contract).
        const bool tinyKvFusion = faiTilingData->maxKvSeqlen <= kvBaseTile_;
        fa950_task::Shape taskShape =
            fa950_task::MakeShape(static_cast<uint32_t>(qSeqlen), qHeads_, kvHeads_, qBaseTile_, batchCount, coreNum,
                                  qFormat != Format::TND, false, FusedGroups && !fa950_plan::IsD512(embed_, embedV_), tinyKvFusion);
        if constexpr (qFormat == Format::TND) {
            qPhysicalSeqlen =
                static_cast<int64_t>(gActualQseqlen.GetValue(curBatch + 1) - gActualQseqlen.GetValue(curBatch));
        }
        if constexpr (kvFormat == Format::TND && kvcacheType == CacheMode::normalCache) {
            kvPhysicalSeqlen =
                static_cast<int64_t>(gActualKvseqlen.GetValue(curBatch + 1) - gActualKvseqlen.GetValue(curBatch));
        } else if constexpr (kvFormat == Format::BSND && kvcacheType == CacheMode::normalCache) {
            // Dense BSND is [B, S_alloc, H, D]; S_alloc may exceed cache_seqlens.
            kvPhysicalSeqlen = params.kvSeqlenAllocated;
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
        // Cross-task pipelining state
        uint32_t tileIssued = 0U;
        uint32_t pvIssued = 0U;
        uint32_t taskSeq = 0U;
        PvTileDesc pvDescs[PV_DESC_SLOTS];
        // L0 A/B stage counters in exact issue order
        uint64_t l0AStagesIssued = 0U;
        uint64_t l0BStagesIssued = 0U;
        for (uint32_t taskIdx = taskBegin; taskIdx < taskEnd; taskIdx += taskStep) {
            while (taskIdx >= curTotalTaskNum) {
                ++curBatch;
                preTotalTaskNum = curTotalTaskNum;
                // Batch strides are recomputed on every batch advance because packed
                // TND regions have a per-batch physical seqlen. Dense BSND uses the
                // allocated S (constant). BNSD planes are maxSeqlen sized so they
                // are constant.
                int64_t qBatchStride = 0;
                if constexpr (qFormat == Format::BNSD) {
                    qBatchStride = static_cast<int64_t>(qHeads_) * qHeadStride;
                } else {
                    qBatchStride = qPhysicalSeqlen * strideQ;
                }
                // O always follows BSND: one batch is qPhysicalSeqlen rows of
                // strideO elements each.
                int64_t oBatchStride = qPhysicalSeqlen * strideO;
                int64_t kBatchStride = 0;
                int64_t vBatchStride = 0;
                if constexpr (kvFormat == Format::BNSD) {
                    kBatchStride = static_cast<int64_t>(kvHeads_) * kvHeadStrideK;
                    vBatchStride = static_cast<int64_t>(kvHeads_) * kvHeadStrideV;
                } else {
                    kBatchStride = kvPhysicalSeqlen * strideK;
                    vBatchStride = kvPhysicalSeqlen * strideV;
                }
                qBOffset += qBatchStride;
                if constexpr (kvcacheType == CacheMode::normalCache) {
                    kBOffset += static_cast<uint64_t>(kBatchStride);
                    vBOffset += static_cast<uint64_t>(vBatchStride);
                } else {
                    blockBOffset += static_cast<uint64_t>(maxNumBlocksPerBatch_);
                }
                oBOffset += oBatchStride;

                const int64_t previousQSeqlen = qSeqlen;
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
                } else if constexpr (kvFormat == Format::BSND && kvcacheType == CacheMode::normalCache) {
                    kvPhysicalSeqlen = params.kvSeqlenAllocated;
                }
                if (qSeqlen != previousQSeqlen) {
                    taskShape =
                        fa950_task::MakeShape(static_cast<uint32_t>(qSeqlen), qHeads_, kvHeads_, qBaseTile_, batchCount,
                                              coreNum, qFormat != Format::TND, false, FusedGroups && !fa950_plan::IsD512(embed_, embedV_), tinyKvFusion);
                }
                curTotalTaskNum += taskShape.totalTasks;
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
            uint32_t qRowStart;
            uint32_t qRowsInTask;
            if (flashDecodeFlag_ != 0U) {
                // FD base blocks are enumerated by the classic host schedule.
                uint32_t curQNBlockTile = GetQNBlockTile(static_cast<uint32_t>(qSeqlen), groupSize,
                                                        qBaseTile_ < Q_M_TILE_MAX);
                uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                uint32_t qNTaskNum = qNBlockNumPerGroup * kvHeads_;
                qSTileIdx = taskIdxCurBatch / qNTaskNum;
                uint32_t qNBlockIdx = taskIdxCurBatch - qSTileIdx * qNTaskNum;
                uint32_t qNBlockIdxInGroup = qNBlockIdx % qNBlockNumPerGroup;
                kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup;
                qNStartIdx = kvHeadIdx * groupSize + qNBlockIdxInGroup * curQNBlockTile;
                qNBlockSize = AscendC::Std::min(curQNBlockTile, groupSize - qNBlockIdxInGroup * curQNBlockTile);
                qRowStart = qSTileIdx * qBaseTile_;
                qRowsInTask = qSeqlen - qRowStart > qBaseTile_ ? qBaseTile_ : qSeqlen - qRowStart;
            } else {
                const fa950_task::Task geometry = fa950_task::Decode(
                    taskShape, taskIdxCurBatch, static_cast<uint32_t>(qSeqlen), qHeads_, kvHeads_, qBaseTile_);
                kvHeadIdx = geometry.headStart / groupSize;
                qNStartIdx = geometry.headStart;
                qNBlockSize = geometry.heads;
                qRowStart = geometry.qStart;
                qRowsInTask = geometry.qRows;
                qSTileIdx = geometry.qStart / qBaseTile_;
            }
            int64_t gmOffsetQ = 0;
            int64_t gmOffsetK = 0;
            int64_t gmOffsetV = 0;
            int64_t gmOffsetO = 0;
            int64_t qSOffset = qSTileIdx * qBaseTile_;
            gmOffsetQ = qBOffset + qSOffset * strideQ + qNStartIdx * qHeadStride;
            gmOffsetO = oBOffset + qSOffset * strideO + qNStartIdx * oHeadStride;
            uint32_t lseHeadStride = static_cast<uint32_t>(faiTilingData->maxQSeqlen);
            uint64_t lseOffset = static_cast<uint64_t>(qBOffset / strideQ) * qHeads_ +
                                 static_cast<uint64_t>(qNStartIdx) * lseHeadStride + qSOffset;
            if constexpr (qFormat == Format::BNSD) {
                // LSE is packed as (B, H, maxQSeqlen); the batch index cannot be
                // recovered from qBOffset/strideQ in BNSD, so use curBatch.
                lseHeadStride = static_cast<uint32_t>(faiTilingData->maxQSeqlen);
                lseOffset = static_cast<uint64_t>(curBatch) * qHeads_ * lseHeadStride +
                            static_cast<uint64_t>(qNStartIdx) * lseHeadStride + qSOffset;
            } else if constexpr (qFormat == Format::TND) {
                lseHeadStride = static_cast<uint32_t>(params.storage.totalQTokens);
                lseOffset = static_cast<uint64_t>(qNStartIdx) * lseHeadStride +
                            static_cast<uint64_t>(qBOffset / strideQ + qSOffset);
            }
            int64_t kvSOffset = 0;
            int64_t kHeadStrideL1 = 0;
            int64_t vHeadStrideL1 = 0;
            if constexpr (kvcacheType == CacheMode::pagedCache && kvcacheShape == PageShape::BnNBsD) {
                strideK = embed_;
                strideV = embedV_;
                gmOffsetK = static_cast<uint64_t>(kvHeadIdx) * embed_ * blockSize_;
                gmOffsetV = static_cast<uint64_t>(kvHeadIdx) * embedV_ * blockSize_;
                kHeadStrideL1 = static_cast<int64_t>(embed_) * blockSize_;
                vHeadStrideL1 = static_cast<int64_t>(embedV_) * blockSize_;
            } else if (cacheLayout == CacheLayout::nd) {
                gmOffsetK = kBOffset + static_cast<uint64_t>(kvHeadIdx) * kvHeadStrideK;
                gmOffsetV = vBOffset + static_cast<uint64_t>(kvHeadIdx) * kvHeadStrideV;
                kHeadStrideL1 = kvHeadStrideK;
                vHeadStrideL1 = kvHeadStrideV;
            } else {
                gmOffsetK = kBOffset + static_cast<uint64_t>(kvHeadIdx * embed_) * kvNumTokens;
                gmOffsetV = vBOffset + static_cast<uint64_t>(kvHeadIdx * embedV_) * kvNumTokens;
                kHeadStrideL1 = static_cast<int64_t>(embed_) * kvNumTokens;
                vHeadStrideL1 = static_cast<int64_t>(embedV_) * kvNumTokens;
            }
            uint32_t qSBlockSize = qRowsInTask;
            // Tasks spanning several KV heads are executed as serial segments
            // that reuse the same L1 Q tile and accumulate into the same rows.
            uint32_t segmentCount = 1U;
            uint32_t segmentHeads = qNBlockSize;
            if constexpr (FusedGroups) {
                if (qNBlockSize > groupSize) {
                    segmentCount = qNBlockSize / groupSize;
                    segmentHeads = qNBlockSize / segmentCount;
                }
            }
            const uint32_t segmentRows = qSBlockSize * segmentHeads;
            const int64_t vHeadStride = vHeadStrideL1;
            // P is written back per KV head so each segment starts on a 16-row
            // L1 boundary; single-segment tasks keep the contiguous fast path.
#ifdef __DAV_VEC__
            // Always published (0 == no segmentation): the epilogue object is
            // reused across tasks, so a stale count would corrupt later tasks.
            epilogueOnlineSoftmax.SetSegmentRows(segmentCount > 1U ? segmentRows : 0U);
#endif
            uint32_t rowNum = qSBlockSize * qNBlockSize;
            uint32_t rowNumRound = 0;
            if constexpr (IsDN) {
                rowNumRound = RoundUp(rowNum, 32U);
            } else {
                rowNumRound = RoundUp(rowNum, 16U);
            }
            uint32_t qkRowNumRound = rowNumRound;
            uint32_t kvSTileSizeAct = kvBaseTile_;

            uint32_t kvStart = 0;
            uint32_t noSkipKvS = static_cast<uint32_t>(kvSeqlen);
            uint32_t kvSLoopNum = 0;
            int64_t windowSizeLeftStartLen = 0;
            int64_t windowSizeLeftEndLen = 0;
            int64_t windowSizeRightStartLen = 0;
            int64_t windowSizeRightEndLen = 0;
            bool notPreMask = true;
            bool notNextMask = true;
            int32_t delStartRow = 0;
            int32_t delEndRow = static_cast<int32_t>(qSeqlen);
            const int32_t qSTileStart = static_cast<int32_t>(qSTileIdx * qBaseTile_);
            if constexpr (maskCategory == MaskCategory::MASK_CAUSAL) {
                int64_t causalKvEnd;
                if (d512) {
                    // Bottom-right aligned center of the tile's last row; use
                    // the actual tile rows so tail tiles do not over-extend.
                    causalKvEnd = static_cast<int64_t>(qSTileIdx) * qBaseTile_ + qSBlockSize + kvSeqlen - qSeqlen;
                } else {
                    causalKvEnd = static_cast<int64_t>(qSTileIdx + 1U) * qBaseTile_ + kvSeqlen - qSeqlen;
                }
                if (causalKvEnd < 0) {
                    causalKvEnd = 0;
                }
                noSkipKvS = AscendC::Std::min((uint32_t)kvSeqlen, (uint32_t)causalKvEnd);
                kvSLoopNum = static_cast<uint32_t>(CeilDiv(noSkipKvS, static_cast<int64_t>(kvBaseTile_)));
            } else if constexpr (maskCategory == MaskCategory::MASK_SWA) {
                // Use the actual ragged lengths and signed bounds before clipping.
                // Negative windows are unlimited; Q and K have different bounds.
                const int64_t qLength = static_cast<int64_t>(qSeqlen);
                const int64_t kvLength = static_cast<int64_t>(kvSeqlen);
                const int64_t seqDiff = kvLength - qLength;
                const int64_t qBlockBase = static_cast<int64_t>(qSTileIdx) * qBaseTile_;
                notPreMask = windowSizeLeft_ < 0 || windowSizeLeft_ >= kvLength - 1;
                notNextMask = windowSizeRight_ < 0 || windowSizeRight_ >= qLength - 1;
                if (!notPreMask) {
                    const int64_t leftOrigin = seqDiff - windowSizeLeft_;
                    windowSizeLeftStartLen = qBlockBase + leftOrigin;
                    windowSizeLeftEndLen = windowSizeLeftStartLen + qSBlockSize;
                    const int64_t firstKey =
                        AscendC::Std::min(kvLength, AscendC::Std::max(int64_t{0}, windowSizeLeftStartLen));
                    kvStart = static_cast<uint32_t>(firstKey) / kvBaseTile_;
                    if (windowSizeLeftEndLen > kvLength) {
                        delStartRow = static_cast<int32_t>(
                            AscendC::Std::min(qLength, AscendC::Std::max(int64_t{0}, kvLength - leftOrigin)));
                    }
                }
                if (!notNextMask) {
                    const int64_t rightOrigin = seqDiff + windowSizeRight_;
                    windowSizeRightStartLen = qBlockBase + rightOrigin;
                    // Exclusive upper bound for the last Q row in this tile.
                    windowSizeRightEndLen = windowSizeRightStartLen + qSBlockSize;
                    const int64_t endKey =
                        AscendC::Std::min(kvLength, AscendC::Std::max(int64_t{0}, windowSizeRightEndLen));
                    noSkipKvS = static_cast<uint32_t>(endKey);
                    if (windowSizeRightStartLen < 0) {
                        delEndRow = static_cast<int32_t>(
                            AscendC::Std::min(qLength, AscendC::Std::max(int64_t{0}, -rightOrigin)));
                    }
                }
                kvSLoopNum = static_cast<uint32_t>(CeilDiv(noSkipKvS, static_cast<int64_t>(kvBaseTile_)));
            } else {
                kvSLoopNum = static_cast<uint32_t>(CeilDiv(noSkipKvS, static_cast<int64_t>(kvBaseTile_)));
            }

            uint32_t fullyMaskedRowsPerHead = 0;
            if constexpr (maskCategory == MaskCategory::MASK_CAUSAL) {
                fullyMaskedRowsPerHead = noSkipKvS < qSBlockSize ? qSBlockSize - noSkipKvS : 0;
            }

            bool emptySpan = (kvSLoopNum == 0);
            if constexpr (maskCategory == MaskCategory::MASK_SWA) {
                emptySpan = emptySpan || (kvStart >= kvSLoopNum);
            }
            // An empty span still has to retire the PV/rescale work left pending by
            // earlier tasks: initOutputs reuses the same epilogue UB, L1 P and O-tmp
            // stages. Such a task runs only the drain iterations of the loop below
            // and initializes its O/LSE afterwards.
            uint32_t kvSplitStart = 0U;
            uint32_t kvSplitEnd = 0U;
            bool writePartial = false;
            uint32_t partialId = 0U;
            if (!emptySpan) {
                kvSplitStart = kvStart;
                kvSplitEnd = kvSLoopNum;
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
                            partialId =
                                static_cast<uint32_t>(faiTilingData->fdCombineSchedules[combineIdx].partialStart +
                                                      static_cast<int32_t>(coreIdx) -
                                                      faiTilingData->fdCombineSchedules[combineIdx].firstCore);
                            break;
                        }
                    }
                }
            }
            const uint32_t kvSLoopCount = kvSplitEnd - kvSplitStart;
#ifdef __DAV_CUBE__
            uint32_t qShapeCol = strideQ;
            uint32_t kShapeCol = strideK;
            uint32_t vShapeCol = strideV;
            auto gmQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(qBaseTile_, qShapeCol);
            auto gmQLayoutTlaDN = tla::MakeLayout<ElementQ, LayoutQ>(qShapeCol, qBaseTile_);
            auto gmQTensorTla = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTla, Arch::PositionGM{});
            auto gmQTensorTlaDN = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTlaDN, Arch::PositionGM{});

            GemmCoord actualBlockShapeQ{rowNum, embed_, 0};
            if (kvSLoopCount != 0U) {
                if constexpr (IsDN) {
                    blockMmadQK.loadQGM(gmQTensorTlaDN, actualBlockShapeQ, qSBlockSize, qNBlockSize, qHeadStride);
                } else if (segmentCount == 1U) {
                    blockMmadQK.template loadQGM<false>(gmQTensorTla, actualBlockShapeQ, qSBlockSize, qNBlockSize,
                                                        qHeadStride);
                } else {
                    // One Q tile per KV head; each segment owns a 16-row padded
                    // slice so the fused QK reads stay segment-local.
                    for (uint32_t seg = 0U; seg < segmentCount; ++seg) {
                        auto gmSegmentQ = tla::MakeTensor(gQ[gmOffsetQ + seg * segmentHeads * qHeadStride],
                                                          gmQLayoutTla, Arch::PositionGM{});
                        GemmCoord segmentShape{segmentRows, embed_, 0};
                        blockMmadQK.template loadQGM<true>(gmSegmentQ, segmentShape, qSBlockSize, segmentHeads,
                                                           qHeadStride,
                                                           seg * RoundUp(segmentRows, 16U) * RoundUp(embed_, 16U),
                                                           seg == 0U, seg + 1U == segmentCount);
                    }
                }
            }
            uint32_t kShapeRow = 0;
            if constexpr (kvcacheType == CacheMode::pagedCache) {
                kShapeRow = numBlocks_ * blockSize_;
            } else {
                kShapeRow = kvSeqlen;
            }
            auto gmKLayoutTla = tla::MakeLayout<ElementK, LayoutK>(kShapeCol, kShapeRow);
            auto gmKLayoutTlaDN = tla::MakeLayout<ElementK, LayoutK>(kShapeRow, kShapeCol);
            auto gmKTensorTla = tla::MakeTensor(gK[gmOffsetK], gmKLayoutTla, Arch::PositionGM{});
            auto gmKTensorTlaDN = tla::MakeTensor(gK[gmOffsetK], gmKLayoutTlaDN, Arch::PositionGM{});
#endif
            const uint32_t taskStatsSlot = taskSeq % fa950_stats::STATS_TASK_STAGES;
            // Only the last task flushes the delayed PV queue
            const bool lastTaskOnCore = (taskIdx + taskStep >= taskEnd);
            const bool flushPvQueue = lastTaskOnCore || emptySpan;
            const uint32_t kvLoopEnd = kvSLoopCount + (flushPvQueue ? PV_DESC_SLOTS : 0U);
            for (uint32_t kvSTileRelIdx = 0; kvSTileRelIdx < kvLoopEnd; ++kvSTileRelIdx) {
                const bool inDrain = (kvSTileRelIdx >= kvSLoopCount);
                const uint32_t pipelineTileSeq = tileIssued;
                if (!inDrain) {
                    const uint32_t kvSTileIdx = kvSplitStart + kvSTileRelIdx;
                    if (kvSTileIdx == kvSLoopNum - 1) {
                        kvSTileSizeAct = noSkipKvS - kvSTileIdx * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};
                    uint32_t ubSBufId = pipelineTileSeq % UB_S_OTMP_BUF_STAGES;
                    int64_t stride = 64;
                    auto ubSLayoutTla =
                        tla::MakeLayout<ElementS, LayoutS>(qkRowNumRound, RoundUp(kvSTileSizeAct, ubSRoundTile));
                    auto ubSLayoutTlaDN = tla::MakeLayout<ElementS, LayoutS>(RoundUp(kvSTileSizeAct, 16), 64);
                    auto ubSTensorTla = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayoutTla, Arch::PositionUB{});
                    auto ubSTensorTlaDN = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayoutTlaDN, Arch::PositionUB{});
                    uint32_t qkReadyFlagId = ubSBufId;
                    Arch::CrossCoreFlag qkReadyFlag(qkReadyFlagId);
                    // Snapshot everything the delayed PV/rescale of this tile
                    // needs; the task-local variables may describe another task
                    // by the time it executes.
                    PvTileDesc& pvDesc = pvDescs[tileIssued % PV_DESC_SLOTS];
                    pvDesc.gmOffsetV = static_cast<uint64_t>(gmOffsetV);
                    pvDesc.gmOffsetO = static_cast<uint64_t>(gmOffsetO);
                    pvDesc.lseOffset = lseOffset;
                    pvDesc.blockBOffset = static_cast<uint64_t>(blockBOffset);
                    pvDesc.kvSTileIdx = kvSTileIdx;
                    pvDesc.kvSTileSizeAct = kvSTileSizeAct;
                    pvDesc.rowNum = rowNum;
                    pvDesc.rowNumRound = rowNumRound;
                    pvDesc.pipelineTileSeq = pipelineTileSeq;
                    pvDesc.relIdx = kvSTileRelIdx;
                    pvDesc.kvSLoopCount = kvSLoopCount;
                    pvDesc.qSBlockSize = qSBlockSize;
                    pvDesc.qNBlockSize = qNBlockSize;
                    if constexpr (FusedGroups) {
                        pvDesc.segmentRows = segmentRows;
                        pvDesc.segmentCount = segmentCount;
                        pvDesc.vHeadStride = vHeadStride;
                    }
                    pvDesc.fullyMaskedRowsPerHead = fullyMaskedRowsPerHead;
                    pvDesc.statsSlot = taskStatsSlot;
                    pvDesc.partialId = partialId;
                    pvDesc.delStartRow = delStartRow;
                    pvDesc.delEndRow = delEndRow;
                    pvDesc.qSeqlen = qSeqlen;
                    pvDesc.qSTileStart = qSTileStart;
                    pvDesc.strideO = static_cast<uint32_t>(strideO);
                    pvDesc.oHeadStride = static_cast<uint32_t>(oHeadStride);
                    pvDesc.lseHeadStride = lseHeadStride;
                    pvDesc.kvNumTokens = kvNumTokens;
                    pvDesc.writePartial = writePartial;
#ifdef __DAV_CUBE__
                    uint64_t prefixSumL0AStages = l0AStagesIssued;
                    uint64_t prefixSumL0BStages = l0BStagesIssued;
                    pvDesc.kShapeRow = kShapeRow;
                    pvDesc.vShapeCol = vShapeCol;
                    if constexpr (IsDN) {
                        blockMmadQK(gmKTensorTlaDN, ubSTensorTlaDN, gBlockTable[blockBOffset], actualBlockShapeQK,
                                    blockSize_, kvSTileIdx, pipelineTileSeq, 0, kvHeads_, kvNumTokens, kvBaseTile_, 0,
                                    0, 0, qSBlockSize, qNBlockSize, qkReadyFlag, prefixSumL0AStages,
                                    prefixSumL0BStages);
                    } else if (!FusedGroups || segmentCount == 1U) {
                        blockMmadQK.template operator()<false>(
                            gmKTensorTla, ubSTensorTla, gBlockTable[blockBOffset], actualBlockShapeQK, blockSize_,
                            kvSTileIdx, pipelineTileSeq, 0, kvHeads_, kvNumTokens, kvBaseTile_, 0, 0, 0, qSBlockSize,
                            qNBlockSize, qkReadyFlag, prefixSumL0AStages, prefixSumL0BStages);
                    } else {
                        for (uint32_t seg = 0U; seg < segmentCount; ++seg) {
                            GemmCoord segmentShape{segmentRows, kvSTileSizeAct, embed_};
                            auto gmSegmentK =
                                tla::MakeTensor(gK[gmOffsetK + seg * kHeadStrideL1], gmKLayoutTla, Arch::PositionGM{});
                            blockMmadQK.template operator()<true>(
                                gmSegmentK, ubSTensorTla, gBlockTable[blockBOffset], segmentShape, blockSize_,
                                kvSTileIdx, pipelineTileSeq, 0, kvHeads_, kvNumTokens, kvBaseTile_, 0, 0, 0,
                                qSBlockSize, segmentHeads, qkReadyFlag, prefixSumL0AStages + seg * qkL0ATotalStages_,
                                prefixSumL0BStages + seg * qkL0BTotalStages_,
                                seg * RoundUp(segmentRows, 16U) * RoundUp(embed_, 16U), seg * segmentRows, rowNum,
                                seg == 0U, seg + 1U == segmentCount);
                        }
                    }
                    if (kvSTileRelIdx == kvSLoopCount - 1U) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
                    }
                    l0AStagesIssued += qkL0ATotalStages_ * segmentCount;
                    l0BStagesIssued += qkL0BTotalStages_ * segmentCount;
#endif
                    uint32_t l1PBufId = pipelineTileSeq % pL1BufNum_;
                    uint32_t softmaxReadyFlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag softmaxReadyFlag(softmaxReadyFlagId);
                    auto l1PLayoutTla = tla::MakeLayout<ElementP, Catlass::layout::zN>(rowNum, kvSTileSizeAct);
                    auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId], l1PLayoutTla, Arch::PositionL1{});
#ifdef __DAV_VEC__
                    // The previous task's final lse rescale may still read the
                    // other slot; give this task its own row statistics.
                    epilogueOnlineSoftmax.SetStatsSlot(taskStatsSlot);
                    if constexpr (maskCategory == MaskCategory::MASK_CAUSAL) {
                        if (d512) {
                            // Positional causal mask: center of the tile's
                            // first logical row, bottom-right aligned with the
                            // effective Q/KV lengths.
                            int32_t qTileCenterStart = static_cast<int32_t>(qSTileStart + kvSeqlen - qSeqlen);
                            uint32_t kvSStartIdx = kvSTileIdx * kvBaseTile_;
                            uint32_t kvSEndIdx = kvSStartIdx + kvSTileSizeAct;
                            bool doTriUMask = qTileCenterStart < static_cast<int32_t>(kvSEndIdx - 1);
                            if (doTriUMask) {
                                epilogueOnlineSoftmax.template operator()<false>(
                                    l1PTensorTla, actualBlockShapeQK, (kvSTileRelIdx == 0), ubSBufId, l1PBufId,
                                    qkReadyFlag, softmaxReadyFlag, qTileCenterStart, kvSStartIdx, qSBlockSize,
                                    qNBlockSize);
                            } else {
                                epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, (kvSTileRelIdx == 0), ubSBufId,
                                                      l1PBufId, qkReadyFlag, softmaxReadyFlag, qSBlockSize,
                                                      qNBlockSize);
                            }
                        } else {
                            auto gmMaskLayoutTla = tla::MakeLayout<ElementMask, LayoutMask>(2048, 2048);
                            auto gmMaskTensorTla = tla::MakeTensor(gMask, gmMaskLayoutTla, Arch::PositionGM{});
                            int64_t triUp = static_cast<int64_t>(noSkipKvS) - qSBlockSize;
                            uint32_t triDown = noSkipKvS;
                            uint32_t kvSStartIdx = kvSTileIdx * kvBaseTile_;
                            uint32_t kvSEndIdx = kvSStartIdx + kvSTileSizeAct;
                            bool doTriUMask = triUp < static_cast<int64_t>(kvSEndIdx - 1);
                            if (doTriUMask) {
                                epilogueOnlineSoftmax(l1PTensorTla, gmMaskTensorTla, actualBlockShapeQK,
                                                      (kvSTileRelIdx == 0), ubSBufId, l1PBufId, qkReadyFlag,
                                                      softmaxReadyFlag, triUp, triDown, 0u, 0u, kvSStartIdx, kvSEndIdx,
                                                      1, qSBlockSize, qNBlockSize);
                            } else {
                                epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, (kvSTileRelIdx == 0), ubSBufId,
                                                      l1PBufId, qkReadyFlag, softmaxReadyFlag, qSBlockSize,
                                                      qNBlockSize);
                            }
                        }
                    } else if constexpr (maskCategory == MaskCategory::MASK_SWA) {
                        int64_t kvSStartIdxI = static_cast<int64_t>(kvSTileIdx) * kvBaseTile_;
                        int64_t kvSEndIdxI = kvSStartIdxI + kvSTileSizeAct;
                        bool doTriUPreMask =
                            notPreMask
                                ? false
                                : (windowSizeLeftStartLen >= kvSStartIdxI && windowSizeLeftStartLen < kvSEndIdxI) ||
                                      (windowSizeLeftEndLen > kvSStartIdxI && windowSizeLeftEndLen <= kvSEndIdxI) ||
                                      (windowSizeLeftStartLen <= kvSStartIdxI && windowSizeLeftEndLen >= kvSEndIdxI);
                        bool doTriUNextMask =
                            notNextMask
                                ? false
                                : (windowSizeRightStartLen >= kvSStartIdxI && windowSizeRightStartLen < kvSEndIdxI) ||
                                      (windowSizeRightEndLen > kvSStartIdxI && windowSizeRightEndLen <= kvSEndIdxI) ||
                                      (windowSizeRightStartLen <= kvSStartIdxI && windowSizeRightEndLen >= kvSEndIdxI);
                        if (d512) {
                            if (doTriUPreMask || doTriUNextMask) {
                                // Window bounds are clamped to the effective
                                // lengths (Sk-1 / Sq-1), matching the donor;
                                // on effective positions this clamp behaves
                                // like an unlimited window because no token
                                // outside [0, Sk-1] is ever visited.
                                const int32_t left = static_cast<int32_t>(
                                    windowSizeLeft_ < 0
                                        ? kvSeqlen - 1
                                        : AscendC::Std::min(windowSizeLeft_, static_cast<int64_t>(kvSeqlen - 1)));
                                const int32_t right = static_cast<int32_t>(
                                    windowSizeRight_ < 0
                                        ? qSeqlen - 1
                                        : AscendC::Std::min(windowSizeRight_, static_cast<int64_t>(qSeqlen - 1)));
                                epilogueOnlineSoftmax.template operator()<true>(
                                    l1PTensorTla, actualBlockShapeQK, (kvSTileRelIdx == 0), ubSBufId, l1PBufId,
                                    qkReadyFlag, softmaxReadyFlag,
                                    static_cast<int32_t>(qSTileStart + kvSeqlen - qSeqlen),
                                    static_cast<uint32_t>(kvSStartIdxI), qSBlockSize, qNBlockSize, left, right);
                            } else {
                                epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, (kvSTileRelIdx == 0), ubSBufId,
                                                      l1PBufId, qkReadyFlag, softmaxReadyFlag, qSBlockSize,
                                                      qNBlockSize);
                            }
                        } else {
                            auto gmMaskLayoutTla = tla::MakeLayout<ElementMask, LayoutMask>(2048, 2048);
                            auto gmMaskTensorTla = tla::MakeTensor(gMask, gmMaskLayoutTla, Arch::PositionGM{});
                            if (doTriUPreMask || doTriUNextMask) {
                                epilogueOnlineSoftmax(
                                    l1PTensorTla, gmMaskTensorTla, actualBlockShapeQK, (kvSTileRelIdx == 0), ubSBufId,
                                    l1PBufId, qkReadyFlag, softmaxReadyFlag, kvSStartIdxI, doTriUPreMask,
                                    doTriUNextMask, windowSizeLeftStartLen, windowSizeLeftEndLen,
                                    windowSizeRightStartLen, windowSizeRightEndLen, qSBlockSize, qNBlockSize, 1u);
                            } else {
                                epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, (kvSTileRelIdx == 0), ubSBufId,
                                                      l1PBufId, qkReadyFlag, softmaxReadyFlag, qSBlockSize,
                                                      qNBlockSize);
                            }
                        }
                    } else {
                        if constexpr (IsDN) {
                            epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, (kvSTileRelIdx == 0), ubSBufId,
                                                  l1PBufId, qkReadyFlag, softmaxReadyFlag, 1, qSBlockSize, qNBlockSize);
                        } else {
                            epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, (kvSTileRelIdx == 0), ubSBufId,
                                                  l1PBufId, qkReadyFlag, softmaxReadyFlag, qSBlockSize, qNBlockSize);
                        }
                    }
#endif
                    ++tileIssued;
                }
                if (pvIssued < tileIssued && (inDrain || pvIssued + PRE_LAUNCH < tileIssued)) {
                    // Tile issued PRE_LAUNCH tiles ago (possibly by an earlier task).
                    const PvTileDesc& pvDesc = pvDescs[pvIssued % PV_DESC_SLOTS];
                    const uint32_t kvSTileRelIdxNow = pvDesc.relIdx;
                    const uint32_t pipelineTileSeqNow = pvDesc.pipelineTileSeq;
                    GemmCoord actualBlockShapePV{pvDesc.rowNum, embedV_, pvDesc.kvSTileSizeAct};
                    uint32_t ubOTmpBufId = pipelineTileSeqNow % otmpStages_;
                    uint32_t pvReadyFlagId = ubOTmpBufId + UB_S_OTMP_BUF_STAGES + pL1BufNum_;
#ifdef __DAV_CUBE__
                    uint32_t l1PBufId = pipelineTileSeqNow % pL1BufNum_;
                    // D512 LO stores one 128-column piece per FixPipe round.
                    auto ubOTmpLayoutTla = tla::MakeLayout<ElementOTmp, LayoutOTmp>(
                        pvDesc.rowNumRound, d512 ? 128U : embedVRound);
                    auto ubOTmpTensorTla =
                        tla::MakeTensor(ubOTmpTensor[ubOTmpBufId], ubOTmpLayoutTla, Arch::PositionUB{});
                    uint32_t softmaxReadyFlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag softmaxReadyFlag(softmaxReadyFlagId);
                    Arch::CrossCoreFlag pvReadyFlag(pvReadyFlagId);
                    auto gmVLayoutTlaNow = tla::MakeLayout<ElementV, LayoutV>(pvDesc.kShapeRow, pvDesc.vShapeCol);
                    auto gmVTensorTlaNow = tla::MakeTensor(gV[pvDesc.gmOffsetV], gmVLayoutTlaNow, Arch::PositionGM{});
                    if (!FusedGroups || pvDesc.segmentCount == 1U) {
                        blockMmadPV.template operator()<false>(
                            gmVTensorTlaNow, ubOTmpTensorTla, gBlockTable[pvDesc.blockBOffset], actualBlockShapePV,
                            blockSize_, pvDesc.kvSTileIdx, pipelineTileSeqNow, 0, kvHeads_, pvDesc.kvNumTokens,
                            kvBaseTile_, 0, 0, 0, softmaxReadyFlag, pvReadyFlag, l0AStagesIssued, l0BStagesIssued,
                            d512);
                        l0AStagesIssued += pvL0ATotalStages_;
                        l0BStagesIssued += pvL0BTotalStages_;
                    } else {
                        // One PV per KV head: each segment starts a new 16-row
                        // slice of the P tile and lands on its own O rows.
                        for (uint32_t seg = 0U; seg < pvDesc.segmentCount; ++seg) {
                            auto gmSegmentV = tla::MakeTensor(gV[pvDesc.gmOffsetV + seg * pvDesc.vHeadStride],
                                                              gmVLayoutTlaNow, Arch::PositionGM{});
                            GemmCoord segmentShape{pvDesc.segmentRows, embedV_, pvDesc.kvSTileSizeAct};
                            blockMmadPV.template operator()<true>(
                                gmSegmentV, ubOTmpTensorTla, gBlockTable[pvDesc.blockBOffset], segmentShape, blockSize_,
                                pvDesc.kvSTileIdx, pipelineTileSeqNow, 0, kvHeads_, pvDesc.kvNumTokens, kvBaseTile_, 0,
                                0, 0, softmaxReadyFlag, pvReadyFlag, l0AStagesIssued, l0BStagesIssued, false,
                                seg * RoundUp(pvDesc.segmentRows, 16U) * RoundUp(pvDesc.kvSTileSizeAct, 16U),
                                seg * pvDesc.segmentRows, pvDesc.rowNum, seg == 0U, seg + 1U == pvDesc.segmentCount,
                                seg, pvDesc.segmentCount);
                            l0AStagesIssued += pvL0ATotalStages_;
                            l0BStagesIssued += pvL0BTotalStages_;
                        }
                    }
#endif
#ifdef __DAV_VEC__
                    Arch::CrossCoreFlag pvReadyFlag(pvReadyFlagId);
                    uint32_t curTileMod = pipelineTileSeqNow % (PRE_LAUNCH + 1);
                    // Read the statistics slot of the task that owns this tile.
                    epilogueRescaleO.SetStatsSlot(pvDesc.statsSlot);
                    uint32_t fmRowsPerHead = pvDesc.fullyMaskedRowsPerHead;
                    int32_t swaDelStartRow = 0;
                    int32_t swaDelEndRow = 0;
                    uint32_t swaQSeqlen = 0;
                    uint32_t swaQSTileStart = 0;
                    if constexpr (maskCategory == MaskCategory::MASK_SWA) {
                        fmRowsPerHead = 0U;
                        swaDelStartRow = pvDesc.delStartRow;
                        swaDelEndRow = pvDesc.delEndRow;
                        swaQSeqlen = static_cast<uint32_t>(pvDesc.qSeqlen);
                        swaQSTileStart = static_cast<uint32_t>(pvDesc.qSTileStart);
                    }
                    auto gmOLayoutTlaNow = tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, pvDesc.strideO);
                    auto gmOTensorTlaNow =
                        tla::MakeTensor(gO[pvDesc.gmOffsetO], gmOLayoutTlaNow, Arch::PositionGM{});
                    // FD bypasses scalar Dm reduction; the D512 path consumes
                    // four 128-column LO pieces while keeping its planned UB pitch.
                    if (flashDecodeFlag_ != 0U) {
                        epilogueRescaleO.template operator()<LseMode, false>(
                            gmOTensorTlaNow, gLse[pvDesc.lseOffset], actualBlockShapePV, curTileMod, pipelineTileSeqNow,
                            (kvSTileRelIdxNow == 0), (kvSTileRelIdxNow == pvDesc.kvSLoopCount - 1), pvReadyFlag, IsDN,
                            fmRowsPerHead, pvDesc.qSBlockSize, pvDesc.qNBlockSize, pvDesc.lseHeadStride, pvDesc.strideO,
                            pvDesc.oHeadStride, swaDelStartRow, swaDelEndRow, swaQSeqlen, swaQSTileStart,
                            gPartialO, gPartialLse, pvDesc.writePartial && !IsDN, pvDesc.partialId, fdRowCapacity_,
                            fdLseSubStride_, 1.0f, d512);
                    } else {
                        epilogueRescaleO.template operator()<LseMode, true>(
                            gmOTensorTlaNow, gLse[pvDesc.lseOffset], actualBlockShapePV, curTileMod, pipelineTileSeqNow,
                            (kvSTileRelIdxNow == 0), (kvSTileRelIdxNow == pvDesc.kvSLoopCount - 1), pvReadyFlag, IsDN,
                            fmRowsPerHead, pvDesc.qSBlockSize, pvDesc.qNBlockSize, pvDesc.lseHeadStride, pvDesc.strideO,
                            pvDesc.oHeadStride, swaDelStartRow, swaDelEndRow, swaQSeqlen, swaQSTileStart,
                            gPartialO, gPartialLse, pvDesc.writePartial && !IsDN, pvDesc.partialId, fdRowCapacity_,
                            fdLseSubStride_, 1.0f, d512);
                    }

#endif
                    ++pvIssued;
                }
            }
            if (emptySpan) {
                // Every pending consumer is retired by the drain iterations above,
                // so it is safe to hand the epilogue UB to initOutputs now.
#ifdef __DAV_VEC__
                initOutputs.template operator()<LseMode>(gO[gmOffsetO], gLse[lseOffset], qSBlockSize, qNBlockSize,
                                                         lseHeadStride, embedV_, static_cast<uint32_t>(strideO),
                                                         static_cast<uint32_t>(oHeadStride));
#endif
            }
            // Only tasks that issued softmax work consume a statistics slot.
            if (kvSLoopCount != 0U) {
                ++taskSeq;
            }
        }
        ReleaseSyncFlags<4, 4, 4>();
        constexpr bool denseBsnd = qFormat == Format::BSND && kvcacheType == CacheMode::normalCache;
        constexpr bool pagedTnd = qFormat == Format::TND && kvcacheType == CacheMode::pagedCache;
        if constexpr (!IsDN && (denseBsnd || pagedTnd) && maskCategory != MaskCategory::MASK_SWA) {
            if (flashDecodeFlag_ != 0U) {
                // All launched AIC/AIV blocks join after their partial DMAs
                // complete. Reuse the main kernel's UB only after this barrier.
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SyncAll<false>();
#ifdef __DAV_VEC__
                FlashDecodeCombine950<ElementO, LseMode, 0, denseBsnd> combine(resource);
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
            if (otmpStages_ == 2U) {
                AscendC::CrossCoreSetFlag<PV_RE_MODE, PIPE_V>(6);
            }
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
            if (otmpStages_ == 2U) {
                AscendC::CrossCoreWaitFlag<PV_RE_MODE, PIPE_FIX>(6);
                AscendC::CrossCoreWaitFlag<PV_RE_MODE, PIPE_FIX>(22);
            }
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
    Arch::ForwardResource950 resource;

    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t embedV_;
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

    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    uint32_t actSeqAval_;

    int64_t qSeqlenAligned_;
    int64_t kvSeqlenAligned_;

    uint32_t qkL1TileM_;
    uint32_t qkL1TileN_;
    uint32_t qkL1TileKLeft_;
    uint32_t qkL1TileKRight_;
    uint32_t pvL1TileM_;
    uint32_t pvL1TileN_;
    uint32_t pvL1TileKLeft_;
    uint32_t pvL1TileKRight_;
    uint32_t qL1BufNum_;
    uint32_t kL1BufNum_;
    uint32_t vL1BufNum_;
    uint32_t pL1BufNum_;

    int64_t windowSizeLeft_ = 0;
    int64_t windowSizeRight_ = 0;

    uint32_t qkL0ATotalStages_;
    uint32_t qkL0BTotalStages_;
    uint32_t pvL0ATotalStages_;
    uint32_t pvL0BTotalStages_;
};

template <class InDtype, class SMDtype, Format qFormat, Format kvFormat, CacheMode kvcacheType, PageShape kvcacheShape,
          MaskCategory maskCategory, CacheLayout cacheLayout, bool LseMode, bool HasSoftcap>
// SyncAll<false> in the kernel body requires batch-mode scheduling; without it,
// concurrent streams running other sync kernels can deadlock on core allocation.
__schedmode__(1) CATLASS_GLOBAL
    void FAInfer(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o, GM_ADDR lse,
                 GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR seqUsedQ, GM_ADDR seqUsedKv, GM_ADDR workspace,
                 GM_ADDR tiling, GM_ADDR runtimeTiling, int64_t kvSeqlenAllocated, FwdStorageParams storage)
{
    using ArchTag = Arch::Ascend950;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    static_assert(std::is_same_v<SMDtype, float>);
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = float;
    using ElementMask = uint8_t;
    // layout tags
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    // To cater to the existing TileCopy class, a dummy rowMajor is used
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    // OTmp is rowMajor on UB(dst)
    using LayoutOTmp = layout::RowMajor;
    // mask
    using LayoutMask = layout::RowMajor;
    // 处理单个tile内Q和K的matmul
    using L1TileShapeQK = Shape<Int<Q_BLK>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK =
        Gemm::MmadFlashAttentionQK<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, false>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
                                                         LayoutS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK>;
    // Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueFAOnlineSoftmaxT<HasSoftcap>;
    using TileCopySoftmax = Epilogue::Tile::TileCopySoftmax<ArchTag, ElementMask, ElementP, LayoutMask, LayoutPDummy>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax =
        Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, MaskType, TileCopySoftmax>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV =
        Gemm::MmadFlashAttentionPV<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, false>;
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
                                kvcacheType, kvcacheShape, maskCategory, cacheLayout, false, LseMode>;
    FAIKernelParams params{q,         k, v,   mask,      blockTables, actualQseqlen, actualKvseqlen,    seqUsedQ,
                           seqUsedKv, o, lse, workspace, tiling,      runtimeTiling, kvSeqlenAllocated, storage};
    Kernel faInfer;
    // Cross-KV-head group fusion only exists on the masked path (the unmasked
    // variants use the DN kernel, which has no segment decomposition).
    if constexpr (maskCategory != MaskCategory::NO_MASK) {
        faInfer.template operator()<true>(params);
    } else {
        faInfer.template operator()<false>(params);
    }
}

template <class InDtype, class SMDtype, Format qFormat, Format kvFormat, CacheMode kvcacheType, PageShape kvcacheShape,
          MaskCategory maskCategory, CacheLayout cacheLayout, bool LseMode, bool HasSoftcap>
__schedmode__(1) CATLASS_GLOBAL
    void FAInferDn(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o, GM_ADDR lse,
                   GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR seqUsedQ, GM_ADDR seqUsedKv,
                   GM_ADDR workspace, GM_ADDR tiling, GM_ADDR runtimeTiling, int64_t kvSeqlenAllocated,
                   FwdStorageParams storage)
{
    using ArchTag = Arch::Ascend950;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    static_assert(std::is_same_v<SMDtype, float>);
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = float;
    using ElementMask = uint8_t;
    using LayoutK = layout::RowMajor;
    using LayoutQ = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutMask = layout::RowMajor;
    using L1TileShapeQK = Shape<Int<128>, Int<Q_BLK>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK =
        Gemm::MmadFlashAttentionQK<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, true>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementK, LayoutK, ElementQ, LayoutQ, ElementS,
                                                         LayoutS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_N>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementK, ElementQ,
                                                  ElementS, void, TileCopyQK>;
    // Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueFAOnlineSoftmaxT<HasSoftcap>;
    using TileCopySoftmax = Epilogue::Tile::TileCopySoftmax<ArchTag, ElementMask, ElementP, LayoutMask, LayoutPDummy>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax =
        Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, MaskType, TileCopySoftmax>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
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
                                kvcacheType, kvcacheShape, maskCategory, cacheLayout, true, LseMode>;
    FAIKernelParams params{q,         k, v,   mask,      blockTables, actualQseqlen, actualKvseqlen,    seqUsedQ,
                           seqUsedKv, o, lse, workspace, tiling,      runtimeTiling, kvSeqlenAllocated, storage};
    Kernel faInfer;
    faInfer(params);
}
