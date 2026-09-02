/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "online_softmax.hpp"
#include "rescale_o.hpp"
#include "init_outputs.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "pv_matmul.hpp"
#include "qk_matmul.hpp"
#include "CombineScale.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "fa_block.h"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "kernel_common.hpp"
#include "kernel_operator.h"
#include "tilingdata.h"
using namespace Catlass;
using namespace KernelCommon;

namespace SplitFuse {
    template <
        class BlockMmadQK,
        class BlockMmadPV,
        class EpilogueOnlineSoftmax,
        class EpilogueRescaleO,
        bool PAGED_CACHE_FLAG,
        FaiKenel::MaskType MASK_TYPE = FaiKenel::MaskType::NO_MASK,
        FaiKenel::inputLayout INPUT_LAYOUT = FaiKenel::inputLayout::BSND,
        class CombineScale = void>
    class FAInferKernel {
    public:
        using ArchTag = typename BlockMmadQK::ArchTag;
        using L1TileShape = typename BlockMmadQK::L1TileShape;
        using ElementQ = typename BlockMmadQK::ElementA;
        using LayoutQ = typename BlockMmadQK::LayoutA;
        using ElementK = typename BlockMmadQK::ElementB;
        using LayoutK = typename BlockMmadQK::LayoutB;
        using ElementS = typename BlockMmadQK::ElementC;
        using LayoutS = typename BlockMmadQK::LayoutC;

        using ElementP = typename BlockMmadPV::ElementA;
        using LayoutP = typename BlockMmadPV::LayoutA;
        using ElementV = typename BlockMmadPV::ElementB;
        using LayoutV = typename BlockMmadPV::LayoutB;

        using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;
        using LayoutMask = typename EpilogueOnlineSoftmax::LayoutMask;

        using ElementO = typename EpilogueRescaleO::ElementOutput;
        using LayoutO = typename EpilogueRescaleO::LayoutOutput;

        using ElementOTmp = typename EpilogueRescaleO::ElementInput;
        using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;

        using ElementLse = typename EpilogueRescaleO::ElementLse;
        using LayoutLse = typename EpilogueRescaleO::LayoutLse;

        using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;
        using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;

        static constexpr Epilogue::LseModeT LSE_MODE = EpilogueRescaleO::LSE_MODE;

        // SWA empty-window tiles: init output (0) and lse (inf) without running attention.
        using InitOutDispatchPolicy = Epilogue::EpilogueAtlasA2InitOutWhenZero<LSE_MODE>;
        using OTypeInit = Gemm::GemmType<ElementO, LayoutO>;
        using LseTypeInit = Gemm::GemmType<ElementLse, LayoutLse>;
        using EpilogueInitOut = Epilogue::Block::BlockEpilogue<InitOutDispatchPolicy, OTypeInit, LseTypeInit>;
        
        static constexpr uint32_t STACK_SLOTS = PRE_LAUNCH + 1;
        static constexpr uint32_t MAX_STACK_BLOCKS = MAX_KV_STACK_LEN / 128;
        static constexpr uint32_t SCORE_SLOT_ELEMENTS = Q_TILE_CEIL * MAX_KV_STACK_LEN;

        struct StackDescriptor {
            uint64_t gmOffsetV;
            uint64_t gmOffsetO;
            uint64_t gmOffsetLse;
            uint64_t blockBOffset;
            uint32_t rowNum;
            uint32_t stackSeqTile;
            uint32_t qSeqlen;
            uint32_t qSBlockSize;
            uint32_t qNBlockSize;
            uint32_t kvSIdx;
            uint32_t kvSLoopNumTotal;
            uint32_t noSkipKvS;
            // uint32_t qBlockY;
            // uint32_t curSelectNum;
            // uint32_t kvYBlockNum;
            uint32_t slot;
            uint32_t taskStateSlot;
            uint32_t selectedKvCount;
            uint32_t fineSegmentBits;
            uint32_t selectedKvIndices[MAX_STACK_BLOCKS];
            bool firstStack;
            bool lastStack;
            int32_t delStartRow;
            int32_t delEndRow;
            int32_t qSBlockIdx;
            int32_t curQNBlockTile;
        };


        struct GlobalTensorBundle {
            AscendC::GlobalTensor<ElementQ>& gQ;
            AscendC::GlobalTensor<ElementK>& gK;
            AscendC::GlobalTensor<ElementK>& gV;
            AscendC::GlobalTensor<ElementMask>& gMask;
            AscendC::GlobalTensor<int32_t>& gBlockTable;
            AscendC::GlobalTensor<int32_t>& gActualQseqlen;
            AscendC::GlobalTensor<int32_t>& gActualKvseqlen;
            AscendC::GlobalTensor<ElementO>& gO;
            AscendC::GlobalTensor<ElementLse>& gLse;
            AscendC::GlobalTensor<ElementLse>& gLseFD;
            AscendC::GlobalTensor<ElementLse>& gOFD;
            AscendC::GlobalTensor<ElementS>& gS;
            AscendC::GlobalTensor<ElementP>& gP;
            AscendC::GlobalTensor<ElementOTmp>& gOTmp;
            AscendC::GlobalTensor<ElementOTmp>& gOUpdate;
            AscendC::GlobalTensor<ElementP>& gPret;
            AscendC::GlobalTensor<ElementMask>& gDrop;
        };

        __aicore__ inline
        FAInferKernel() {}

        __aicore__ inline
        void operator()(FAIKernelParams const &params)
        {
            __gm__ FAInferTilingData *fATilingData = reinterpret_cast<__gm__ FAInferTilingData *>(params.tiling);
            mm1OutSize = fATilingData->mm1OutSize;
            smOnlineOutSize = fATilingData->smOnlineOutSize;
            mm2OutSize = fATilingData->mm2OutSize;
            batch = fATilingData->batch;
            qHeads = fATilingData->numHeads;
            kvHeads = fATilingData->kvHeads;
            embed = fATilingData->embeddingSize;
            embedV = fATilingData->embeddingSizeV;
            pagedBlockSize = fATilingData->blockSize;
            maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;
            firstBatchTaskNum = fATilingData->firstBatchTaskNum;
            totalTaskNum = fATilingData->totalTaskNum;
            blockSize = fATilingData->blockSize;
            maskType = fATilingData->maskType;
            windowSizeLeft = fATilingData->windowSizeLeft;
            windowSizeRight = fATilingData->windowSizeRight;
            scaleValue = fATilingData->scaleValue;
            softcapValue = fATilingData->softcapValue;
            dropoutValue = fATilingData->dropoutValue;
            maxQSeqlen = fATilingData->maxQSeqlen;
            maxKvSeqlen = fATilingData->maxKvSeqlen;
            flashDecodeFlag = fATilingData->flashDecodeFlag;

            // FD workspace sizing: reserve head of workspace for gLseFD/gOFD.
            uint64_t Lsesize = 0;
            uint64_t Losize = 0;
            if (flashDecodeFlag != 0U) {
                Lsesize = fATilingData->splitLseTotalSize;
                Losize = fATilingData->splitOTotalSize;
            }

            AscendC::GlobalTensor<ElementQ> gQ;
            gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
            AscendC::GlobalTensor<ElementK> gK;
            gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
            AscendC::GlobalTensor<ElementK> gV;
            gV.SetGlobalBuffer((__gm__ ElementK *)params.v);
            AscendC::GlobalTensor<ElementMask> gMask;
            gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
            AscendC::GlobalTensor<int32_t> gBlockTable;
            gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
            AscendC::GlobalTensor<int32_t> gActualQseqlen;
            gActualQseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualQseqlen);
            AscendC::GlobalTensor<int32_t> gActualKvseqlen;
            gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualKvseqlen);
            AscendC::GlobalTensor<ElementO> gO;
            gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
            AscendC::GlobalTensor<ElementLse> gLse;
            gLse.SetGlobalBuffer((__gm__ ElementLse *)params.lse);

            AscendC::GlobalTensor<ElementLse> gLseFD;
            AscendC::GlobalTensor<ElementLse> gOFD;
            if (flashDecodeFlag != 0U) {
                gLseFD.SetGlobalBuffer((__gm__ ElementLse *)(params.workSpace));
                gOFD.SetGlobalBuffer((__gm__ ElementLse *)(params.workSpace + Lsesize));
            }

            AscendC::GlobalTensor<ElementS> gS;
            gS.SetGlobalBuffer((__gm__ ElementS *)(params.workSpace + Lsesize + Losize));
            AscendC::GlobalTensor<ElementP> gP;
            gP.SetGlobalBuffer((__gm__ ElementP *)(params.workSpace + Lsesize + Losize + mm1OutSize));
            AscendC::GlobalTensor<ElementOTmp> gOTmp;
            gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace + Lsesize + Losize +
                mm1OutSize + smOnlineOutSize));
            AscendC::GlobalTensor<ElementOTmp> gOUpdate;
            gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace + Lsesize + Losize +
                mm1OutSize + smOnlineOutSize + mm2OutSize));
            
            AscendC::GlobalTensor<ElementP> gPret;
            gPret.SetGlobalBuffer((__gm__ ElementP *)(fATilingData->pDevice));
            AscendC::GlobalTensor<ElementMask> gDrop;
            gDrop.SetGlobalBuffer((__gm__ ElementMask *)(fATilingData->dropMaskDevice));

            GlobalTensorBundle globalTensors{
                gQ, gK, gV, gMask, gBlockTable,
                gActualQseqlen, gActualKvseqlen,
                gO, gLse, gLseFD, gOFD,
                gS, gP, gOTmp, gOUpdate, gPret, gDrop
            };

            strideQ = static_cast<uint64_t>(qHeads * embed);
            strideO = static_cast<uint64_t>(qHeads * embedV);
            strideK = static_cast<uint64_t>(kvHeads * embed);
            strideV = static_cast<uint64_t>(kvHeads * embedV);
            stridePret = static_cast<uint64_t>(maxQSeqlen * maxKvSeqlen);
            strideDrop = static_cast<uint64_t>(maxQSeqlen * CeilDiv(maxKvSeqlen, 8));

            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
#ifdef __DAV_C220_CUBE__
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);

            uint32_t kDynNum = RoundUp(embed, NUM_128);
            kDynNum = kDynNum < NUM_256 ? NUM_256 : kDynNum;
            uint32_t maxQKPL1Size = L1_MAX_SIZE - embedV * MAX_KV_STACK_LEN * sizeof(ElementV);
            uint32_t maxQL1Size = Q_TILE_CEIL * kDynNum * sizeof(ElementQ);
            uint32_t maxNDynNum =
                ((maxQKPL1Size - maxQL1Size) / kDynNum / sizeof(ElementV) / DOUBLE_BUFFER) / NUM_32 * NUM_32;

            uint32_t nDynNum = maxNDynNum < L1_MAX_N_NUM ? maxNDynNum : L1_MAX_N_NUM;
            nDynNum = L1_MAX_N_NUM % nDynNum != 0 ? RoundDown((nDynNum - 1), NUM_32) : nDynNum;

            uint32_t L1_QK_SIZE = BlockMmadQK::L1TileShape::M * kDynNum * sizeof(ElementQ);
            blockMmadQK.SetPingPongState(&pingPongState);
            blockMmadPV.SetPingPongState(&pingPongState);
            blockMmadQK.init(resource, nDynNum, kDynNum, MAX_KV_STACK_LEN);
            uint32_t kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M;
            blockMmadPV.init(resource, nDynNum, kPVDynNum, MAX_KV_STACK_LEN, L1_QK_SIZE);
#endif
#ifdef __DAV_C220_VEC__
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);

            epilogueOnlineSoftmax.init(
                resource, scaleValue, softcapValue, gPret, stridePret, gDrop, strideDrop, maxKvSeqlen);
            epilogueRescaleO.init(resource, dropoutValue);

            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif

            embedRound = RoundUp(embed, FaiKenel::BLOCK_SIZE);
            embedRoundV = RoundUp(embedV, FaiKenel::BLOCK_SIZE);
            groupSize = qHeads / kvHeads;

            totalQTokens = 0;
            if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                totalQTokens = static_cast<uint32_t>(gActualQseqlen.GetValue(batch));
            }

            StackDescriptor descriptors[STACK_SLOTS];
            uint32_t issuedStackCount = 0;
            uint32_t taskStateSequence = 0;

            const bool idleCoreFD = (flashDecodeFlag != 0U) &&
                (fATilingData->needCoreNum != 0U) &&
                (coreIdx >= fATilingData->needCoreNum);
            if (flashDecodeFlag != 0U && !idleCoreFD) {
                uint32_t startBIdx = fATilingData->coreInfo[coreIdx].startBIdx;
                uint32_t startN1Idx = fATilingData->coreInfo[coreIdx].startN1Idx;
                uint32_t startS1Idx = fATilingData->coreInfo[coreIdx].startS1Idx;
                uint32_t startS2Idx = fATilingData->coreInfo[coreIdx].startS2Idx;
                uint32_t endBIdx = fATilingData->coreInfo[coreIdx].endBIdx;
                uint32_t endN1Idx = fATilingData->coreInfo[coreIdx].endN1Idx;
                uint32_t endS1Idx = fATilingData->coreInfo[coreIdx].endS1Idx;
                uint32_t endS2Idx = fATilingData->coreInfo[coreIdx].endS2Idx;
                uint64_t gmOffsetLseFD = fATilingData->coreInfo[coreIdx].firstSplitKVTaskLseOffset;
                uint64_t gmOffsetOFD = fATilingData->coreInfo[coreIdx].firstSplitKVTaskOOffset;

                for (uint32_t BIdx = startBIdx; BIdx <= endBIdx; BIdx++) {
                    uint32_t qSeqlenCur = fATilingData->maxQSeqlen;
                    uint32_t kvSeqlenCur = static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx));
                    if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                        uint32_t prevQSeqlenSum = static_cast<uint32_t>(gActualQseqlen.GetValue(BIdx));
                        qSeqlenCur = static_cast<uint32_t>(gActualQseqlen.GetValue(BIdx + 1)) - prevQSeqlenSum;
                        if constexpr (!PAGED_CACHE_FLAG) {
                            uint32_t prevKvSeqlenSum = static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx));
                            kvSeqlenCur = static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx + 1)) - prevKvSeqlenSum;
                        }
                    }

                    uint32_t curQNBlockTileTmp = GetQNBlockTile(qSeqlenCur, groupSize);
                    uint32_t qNBlockNumPerGroupTmp = CeilDiv(groupSize, curQNBlockTileTmp);
                    uint32_t curQNBlockNumTmp = qNBlockNumPerGroupTmp * kvHeads;
                    uint32_t curQSBlockTileTmp = GetQSBlockTile(kvSeqlenCur);
                    uint32_t curQSBlockNumTmp = CeilDiv(qSeqlenCur, curQSBlockTileTmp);
                    uint32_t curKSBlockNumTmp = CeilDiv(kvSeqlenCur, MAX_KV_STACK_LEN);

                    int32_t stN1IdxNow = (BIdx == startBIdx) ? (int32_t)startN1Idx : 0;
                    int32_t enN1IdxNow = (BIdx == endBIdx) ? (int32_t)endN1Idx : (int32_t)curQNBlockNumTmp - 1;

                    for (int32_t n1Idx = stN1IdxNow; n1Idx <= enN1IdxNow; n1Idx++) {
                        int32_t stS1IdxNow = (BIdx == startBIdx && n1Idx == stN1IdxNow) ?
                            (int32_t)startS1Idx : 0;
                        int32_t enS1IdxNow = (BIdx == endBIdx && n1Idx == enN1IdxNow) ?
                            (int32_t)endS1Idx : (int32_t)curQSBlockNumTmp - 1;

                        for (int32_t s1Idx = stS1IdxNow; s1Idx <= enS1IdxNow; s1Idx++) {
                            int32_t stS2IdxNow = (BIdx == startBIdx && n1Idx == stN1IdxNow && s1Idx == stS1IdxNow) ?
                                (int32_t)startS2Idx : 0;
                            int32_t enS2IdxNow = (BIdx == endBIdx && n1Idx == enN1IdxNow && s1Idx == enS1IdxNow) ?
                                (int32_t)endS2Idx : (int32_t)curKSBlockNumTmp;
                            bool isSplitKV = (enS2IdxNow - stS2IdxNow) > 0 &&
                                (enS2IdxNow - stS2IdxNow) < static_cast<int32_t>(curKSBlockNumTmp);
                            
                            uint32_t pipelineDrain = 0; 
                            if(startBIdx == endBIdx && n1Idx == enN1IdxNow && s1Idx == enS1IdxNow) {
                                pipelineDrain = PRE_LAUNCH;
                            }

                            const uint32_t currentTaskStateSlot =  taskStateSequence % STACK_SLOTS;
                            ++taskStateSequence;

                            runMainLoop(
                                coreIdx, BIdx, (uint32_t)n1Idx, (uint32_t)s1Idx,
                                isSplitKV, stS2IdxNow, enS2IdxNow,
                                gmOffsetLseFD, gmOffsetOFD,
                                globalTensors,
                                descriptors,
                                currentTaskStateSlot,
                                issuedStackCount,
                                pipelineDrain
                            );

                            if (isSplitKV) {
                                uint32_t qSBlockSizeTmp = (s1Idx == static_cast<int32_t>(curQSBlockNumTmp - 1U)) ?
                                    (qSeqlenCur - s1Idx * curQSBlockTileTmp) : curQSBlockTileTmp;
                                uint32_t qNBlockIdxCurGroupTmp = n1Idx % qNBlockNumPerGroupTmp;
                                uint32_t qNBlockSizeTmp = (qNBlockIdxCurGroupTmp == (qNBlockNumPerGroupTmp - 1U)) ?
                                    (groupSize - qNBlockIdxCurGroupTmp * curQNBlockTileTmp) : curQNBlockTileTmp;
                                gmOffsetLseFD += qSBlockSizeTmp * qNBlockSizeTmp;
                                gmOffsetOFD += qSBlockSizeTmp * qNBlockSizeTmp * embedV;
                            }
                        }
                    }
                }
            } else if (flashDecodeFlag == 0U) {
                // taskIdx 随 for 单调递增，批次定位游标跨 task 复用，避免每个 task 从 batch 0 重扫
                uint32_t curBatchTmp = 0;
                uint32_t preTotalTaskNumTmp = 0;
                uint32_t curTotalTaskNumTmp = firstBatchTaskNum;
                // 当前 batch 派生量缓存（GM 输入只读，同 batch 内不变），0xFFFFFFFF 表示未缓存
                uint32_t cachedBatch = 0xFFFFFFFFU;
                uint32_t curQNBlockNumCur = 0;
                uint32_t curQSBlockNumCur = 0;
                for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum))  {
                    // AscendC::printf("taskIdx %u\n", taskIdx);
                    while (taskIdx >= curTotalTaskNumTmp) {
                        ++curBatchTmp;
                        preTotalTaskNumTmp = curTotalTaskNumTmp;
                        uint32_t qSeqlenTmp = fATilingData->maxQSeqlen;
                        uint32_t kvSeqlenTmp = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatchTmp));
                        if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                            uint32_t prevQSeqlenSumTmp = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatchTmp));
                            qSeqlenTmp = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatchTmp + 1)) - prevQSeqlenSumTmp;
                            if constexpr (!PAGED_CACHE_FLAG) {
                                uint32_t prevKvSeqlenSumTmp = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatchTmp));
                                kvSeqlenTmp = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatchTmp + 1)) - prevKvSeqlenSumTmp;
                            }
                        }

                        uint32_t curQNBlockTileTmp = GetQNBlockTile(qSeqlenTmp, groupSize);
                        uint32_t qNBlockNumPerGroupTmp = CeilDiv(groupSize, curQNBlockTileTmp);
                        uint32_t curQNBlockNumTmp = qNBlockNumPerGroupTmp * kvHeads;
                        uint32_t curQSBlockTileTmp = GetQSBlockTile(kvSeqlenTmp);
                        uint32_t curQSBlockNumTmp = CeilDiv(qSeqlenTmp, curQSBlockTileTmp);
                        curTotalTaskNumTmp += curQNBlockNumTmp * curQSBlockNumTmp;
                    }

                    if (curBatchTmp != cachedBatch) {
                        uint32_t qSeqlenCur = fATilingData->maxQSeqlen;
                        if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                            uint32_t prevQSeqlenSumCur = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatchTmp));
                            qSeqlenCur = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatchTmp + 1)) - prevQSeqlenSumCur;
                        }
                        uint32_t curQNBlockTileCur = GetQNBlockTile(qSeqlenCur, groupSize);
                        uint32_t qNBlockNumPerGroupCur = CeilDiv(groupSize, curQNBlockTileCur);
                        curQNBlockNumCur = qNBlockNumPerGroupCur * kvHeads;
                        curQSBlockNumCur = CeilDiv(qSeqlenCur, 128);
                        cachedBatch = curBatchTmp;
                    }

                    uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNumTmp;
                    // uint32_t qSBlockIdxCur = taskIdxCurBatch / curQNBlockNumCur;
                    // uint32_t qNBlockIdxCur = taskIdxCurBatch - qSBlockIdxCur * curQNBlockNumCur;

                    uint32_t qNBlockIdxCur = taskIdxCurBatch / curQSBlockNumCur;
                    uint32_t qSBlockIdxCur = taskIdxCurBatch - qNBlockIdxCur * curQSBlockNumCur;
                    

                    uint32_t pipelineDrain = 0; 
                    if(taskIdx + uint32_t(coreNum) >= totalTaskNum) {
                        pipelineDrain = PRE_LAUNCH;
                    }
                    
                    const uint32_t currentTaskStateSlot =  taskStateSequence % STACK_SLOTS;
                    ++taskStateSequence;

                    runMainLoop(
                        coreIdx, curBatchTmp, qNBlockIdxCur, qSBlockIdxCur,
                        false, 0, 0,
                        0, 0,
                        globalTensors,
                        descriptors,
                        currentTaskStateSlot,
                        issuedStackCount,
                        pipelineDrain
                    );
                }
            }

#ifdef __DAV_C220_CUBE__
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif
#ifdef __DAV_C220_VEC__
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif
            AscendC::PipeBarrier<PIPE_ALL>();

            if (flashDecodeFlag != 0U) {
                AscendC::SyncAll();
#ifdef __DAV_C220_VEC__
                CombineScale combineScale;
                combineScale.init(resource);
                combineScale(
                    qHeads,
                    fATilingData->totalSplitNodeNum,
                    embedV,
                    fATilingData->splitInfo,
                    gLseFD,
                    gOFD,
                    gO,
                    gActualQseqlen,
                    INPUT_LAYOUT == FaiKenel::inputLayout::TND,
                    maxQSeqlen,
                    (LSE_MODE == Epilogue::LseModeT::OUT_ONLY),
                    gLse,
                    totalQTokens
                );
#endif
            }
        }

        __aicore__ inline void runMainLoop(
            uint32_t coreIdx,
            uint32_t BIdx,
            uint32_t qNBlockIdx,
            uint32_t qSBlockIdx,
            bool isSplitKV,
            int32_t stS2IdxNow,
            int32_t enS2IdxNow,
            uint64_t gmOffsetLseFD,
            uint64_t gmOffsetOFD,
            GlobalTensorBundle& globalTensors,
            StackDescriptor (&descriptors)[STACK_SLOTS],
            const uint32_t& currentTaskStateSlot,
            uint32_t& issuedStackCount,
            uint32_t pipelineDrain
        ) {
            auto& gQ = globalTensors.gQ;
            auto& gK = globalTensors.gK;
            auto& gV = globalTensors.gV;
            auto& gMask = globalTensors.gMask;
            auto& gBlockTable = globalTensors.gBlockTable;
            auto& gActualQseqlen = globalTensors.gActualQseqlen;
            auto& gActualKvseqlen = globalTensors.gActualKvseqlen;
            auto& gO = globalTensors.gO;
            auto& gLse = globalTensors.gLse;
            auto& gLseFD = globalTensors.gLseFD;
            auto& gOFD = globalTensors.gOFD;
            auto& gS = globalTensors.gS;
            auto& gP = globalTensors.gP;
            auto& gOTmp = globalTensors.gOTmp;
            auto& gOUpdate = globalTensors.gOUpdate;
            auto& gPret = globalTensors.gPret;
            auto& gDrop = globalTensors.gDrop;

            uint32_t qSeqlen = maxQSeqlen;
            uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx));
            uint32_t prevQSeqlenSum = 0;
            uint32_t prevKvSeqlenSum = 0;

            if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                prevQSeqlenSum = static_cast<uint32_t>(gActualQseqlen.GetValue(BIdx));
                qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(BIdx + 1)) - prevQSeqlenSum;
                if constexpr (!PAGED_CACHE_FLAG) {
                    prevKvSeqlenSum = static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx));
                    kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx + 1)) - prevKvSeqlenSum;
                }
            } else {
                // BSND: Q/O/LSE per-batch storage step is maxQSeqlen.
                prevQSeqlenSum = BIdx * maxQSeqlen;
                if constexpr (!PAGED_CACHE_FLAG) {
                    // Mirror mha_fwd_kvcache_2.cpp semantics: per-batch K/V step
                    // uses each batch's actual kvSeqlen (prefix sum across batches).
                    for (uint32_t b = 0; b < BIdx; b++) {
                        prevKvSeqlenSum += static_cast<uint32_t>(gActualKvseqlen.GetValue(b));
                    }
                }
            }

            uint64_t qBOffset = static_cast<uint64_t>(prevQSeqlenSum) * strideQ;
            uint64_t kBOffset = 0;
            uint64_t vBOffset = 0;
            uint64_t blockBOffset = 0;
            if constexpr (!PAGED_CACHE_FLAG) {
                kBOffset = static_cast<uint64_t>(prevKvSeqlenSum) * strideK;
                vBOffset = static_cast<uint64_t>(prevKvSeqlenSum) * strideV;
            } else {
                blockBOffset = static_cast<uint64_t>(BIdx) * static_cast<uint64_t>(maxNumBlocksPerBatch);
            }
            uint64_t oBOffset = static_cast<uint64_t>(prevQSeqlenSum) * strideO;
            // LSE output is head-major (BNS for BSND, NT for TND). The batch base and the
            // per-head stride depend on layout:
            //   TND  -> {num_heads, total_q}:  batch folded into global token, head stride = totalQTokens
            //   BSND -> {batch, num_heads, seqlen_q}: batch base = prevQSeqlenSum*qHeads, head stride = maxQSeqlen
            uint64_t lseBOffset;
            uint32_t lseHeadStride;
            if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                lseBOffset = static_cast<uint64_t>(prevQSeqlenSum);
                lseHeadStride = totalQTokens;
            } else {
                lseBOffset = static_cast<uint64_t>(prevQSeqlenSum) * qHeads;
                lseHeadStride = maxQSeqlen;
            }

            uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
            uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
            uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
            uint32_t curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
            uint32_t curKSBlockNum = CeilDiv(kvSeqlen, MAX_KV_STACK_LEN);

            uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
            uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;
            uint32_t qNStartIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
            uint32_t lseTokenOffset = qSBlockIdx * curQSBlockTile;

            uint64_t gmOffsetQ = qBOffset +
                static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideQ +
                static_cast<uint64_t>(qNStartIdx * embed);
            uint64_t gmOffsetK = kBOffset + static_cast<uint64_t>(kvNIdx * embed);
            uint64_t gmOffsetV = vBOffset + static_cast<uint64_t>(kvNIdx * embedV);
            uint64_t gmOffsetO = oBOffset +
                static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideO +
                static_cast<uint64_t>(qNStartIdx * embedV);
            // head-major: base + head*headStride + qToken
            uint64_t gmOffsetLse = lseBOffset +
                static_cast<uint64_t>(qNStartIdx) * lseHeadStride +
                static_cast<uint64_t>(lseTokenOffset);

            uint64_t gmOffsetPret = static_cast<uint64_t>(BIdx * qHeads + qNStartIdx) * stridePret +
                                    static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * maxKvSeqlen;
            uint64_t gmOffsetDrop = static_cast<uint64_t>(BIdx * qHeads + qNStartIdx) * strideDrop +
                                    static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * CeilDiv(maxKvSeqlen, 8);

            uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1U)) ?
                (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;
            uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1U)) ?
                (groupSize - qNBlockIdxCurGroup * curQNBlockTile) : curQNBlockTile;
            uint32_t rowNum = qSBlockSize * qNBlockSize;

            uint32_t noSkipKvS = kvSeqlen;
            uint32_t kvSLoopNumTotal = 0;
            uint32_t kvStart = 0;
            int32_t windowSizeLeftStartLen = 0;
            int32_t windowSizeLeftEndLen = 0;
            int32_t windowSizeRightStartLen = 0;
            int32_t windowSizeRightEndLen = 0;
            bool notPreMask = true;
            bool notNextMask = true;
            int32_t delStartRow = 0;
            int32_t delEndRow = qSeqlen;
            bool startsWithMaskTile = false;
            bool startsWithMaskThenNomaskFlag = false;
            if (maskType == 1U) {
                int64_t diffS = static_cast<int64_t>(kvSeqlen) - static_cast<int64_t>(qSeqlen);
                int64_t causalKvEnd =
                    static_cast<int64_t>((qSBlockIdx + 1U) * curQSBlockTile) + diffS;
                causalKvEnd = causalKvEnd < 0 ? 0 : causalKvEnd;
                noSkipKvS = AscendC::Std::min(causalKvEnd, static_cast<int64_t>(kvSeqlen));
                kvSLoopNumTotal = CeilDiv(noSkipKvS, MAX_KV_STACK_LEN);
                delEndRow = qSeqlen > kvSeqlen ? static_cast<int32_t>(qSeqlen - kvSeqlen) : delEndRow;
            } else if (maskType == 2U) {
                int32_t leftPointwindowSizeLeft = kvSeqlen;
                int32_t leftPointwindowSizeRight = 0;
                if (windowSizeLeft < 0 && windowSizeLeft * (-1) >= qSeqlen) {
                    kvStart = kvSeqlen / MAX_KV_STACK_LEN + 1;
                } else if (windowSizeLeft != SPARSE_MODE_INT_MAX) {
                    leftPointwindowSizeLeft = kvSeqlen - qSeqlen - windowSizeLeft;
                    windowSizeLeftStartLen = qSBlockIdx * curQSBlockTile + leftPointwindowSizeLeft;
                    windowSizeLeftEndLen = qSBlockIdx * curQSBlockTile + qSBlockSize + leftPointwindowSizeLeft;
                    kvStart = AscendC::Std::max(static_cast<int32_t>(0), windowSizeLeftStartLen) / static_cast<int32_t>(MAX_KV_STACK_LEN);
                    notPreMask = false;
                } else {
                    kvStart = 0;
                }
                if (windowSizeRight < 0 && windowSizeRight * (-1) >= kvSeqlen) {
                    kvSLoopNumTotal = 0;
                } else if (windowSizeRight != SPARSE_MODE_INT_MAX) {
                    leftPointwindowSizeRight = kvSeqlen - qSeqlen + windowSizeRight;
                    windowSizeRightStartLen = qSBlockIdx * curQSBlockTile + leftPointwindowSizeRight;
                    windowSizeRightEndLen = qSBlockIdx * curQSBlockTile + qSBlockSize + leftPointwindowSizeRight;
                    int32_t nsk = AscendC::Std::min(static_cast<int32_t>(kvSeqlen), RoundUp(windowSizeRightEndLen, static_cast<int32_t>(MAX_KV_STACK_LEN)));
                    nsk = nsk <= 0 ? static_cast<int32_t>(kvSeqlen) : nsk;
                    noSkipKvS = static_cast<uint32_t>(nsk);
                    kvSLoopNumTotal = CeilDiv(noSkipKvS, MAX_KV_STACK_LEN);
                    notNextMask = false;
                } else {
                    noSkipKvS = kvSeqlen;
                    kvSLoopNumTotal = CeilDiv(noSkipKvS, MAX_KV_STACK_LEN);
                }
                if (windowSizeLeftEndLen > static_cast<int32_t>(kvSeqlen) && windowSizeLeft != SPARSE_MODE_INT_MAX) {
                    delStartRow = kvSeqlen - leftPointwindowSizeLeft;
                } else if (windowSizeRightStartLen < 0 && windowSizeRight != SPARSE_MODE_INT_MAX) {
                    delEndRow = -leftPointwindowSizeRight;
                }
            } else {
                noSkipKvS = kvSeqlen;
                kvSLoopNumTotal = CeilDiv(noSkipKvS, MAX_KV_STACK_LEN);
            }

            // Match FA GPU split-KV: n_block_min/max = split ∩ window (do not overwrite SWA).
            uint32_t winKvStart = kvStart;
            uint32_t winKvEnd = kvSLoopNumTotal;
            uint32_t kvEnd = winKvEnd;
            if (flashDecodeFlag != 0U) {
                uint32_t fdStart = static_cast<uint32_t>(stS2IdxNow);
                uint32_t fdEnd = (enS2IdxNow == static_cast<int32_t>(curKSBlockNum)) ?
                    winKvEnd : static_cast<uint32_t>(enS2IdxNow);
                kvStart = AscendC::Std::max(fdStart, winKvStart);
                kvEnd = AscendC::Std::min(fdEnd, winKvEnd);
            }

            int32_t stackSeqCount = 0;
            uint32_t preKVNum = PRE_LAUNCH;
            uint32_t blockStackNum = (MAX_KV_STACK_LEN - 1 + pagedBlockSize) / pagedBlockSize;
            uint32_t stackSeqTile = MAX_KV_STACK_LEN;
            uint32_t stackSeqTilePad = MAX_KV_STACK_LEN;
            uint32_t taskStackCount = 0;

            // Empty after split\cap window (GPU early exit). Split partials host-inited to 0/-inf.
            if (kvStart >= kvEnd) {
#ifdef __DAV_C220_VEC__
                if (!isSplitKV) {
                    LayoutO layoutOInit(qSeqlen, embed * qHeads);
                    LayoutLse layoutLseInit(qHeads, lseHeadStride);
                    EpilogueInitOut epilogueInitOut(resource);
                    epilogueInitOut(gO[gmOffsetO], gLse[gmOffsetLse], layoutOInit, layoutLseInit, qSBlockSize, qNBlockSize);
                }
#endif
                return;
            }
#ifdef __DAV_C220_CUBE__
            LayoutQ layoutQTemp(rowNum, embed);
            LayoutK layoutKTemp(strideK, stackSeqTile);
            LayoutV layoutVTemp(stackSeqTile, strideV);
            blockMmadQK.resetBlockStart(kvStart, pagedBlockSize);
            blockMmadPV.resetBlockStart(kvStart, pagedBlockSize);
            blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, qHeads);
#endif
            for (uint32_t kvSIdx = kvStart; kvSIdx < kvEnd + pipelineDrain; kvSIdx++) {
                if (kvSIdx < kvEnd) {
                    const uint32_t curStackTileMod = issuedStackCount % STACK_SLOTS;
                    StackDescriptor &desc = descriptors[curStackTileMod];
                    if (kvSIdx + 1 > kvSLoopNumTotal - 1U) {
                        desc.stackSeqTile = noSkipKvS - kvSIdx * MAX_KV_STACK_LEN;
                    } else {
                        desc.stackSeqTile = MAX_KV_STACK_LEN;
                    }
                    
                    desc.gmOffsetV = gmOffsetV;
                    desc.gmOffsetO = gmOffsetO;
                    desc.gmOffsetLse = gmOffsetLse;
                    desc.blockBOffset = blockBOffset;
                    desc.rowNum = rowNum;
                    desc.qSeqlen = qSeqlen;
                    desc.qSBlockSize = qSBlockSize;
                    desc.qNBlockSize = qNBlockSize;
                    desc.kvSIdx = kvSIdx;
                    desc.kvSLoopNumTotal = kvSLoopNumTotal;
                    desc.noSkipKvS = noSkipKvS;
                    // desc.qBlockY = qBlockY;
                    // desc.curSelectNum = curSelectNum;
                    // desc.kvYBlockNum = kvYBlockNum;
                    desc.slot = curStackTileMod;
                    desc.taskStateSlot = currentTaskStateSlot;
                    desc.firstStack = taskStackCount == 0;
                    desc.lastStack = kvSIdx + 1 >= kvSLoopNumTotal;
                    desc.delStartRow = delStartRow;
                    desc.delEndRow = delEndRow;
                    desc.qSBlockIdx = qSBlockIdx;
                    desc.curQNBlockTile = curQNBlockTile;
                    uint64_t gmOffsetS =
                        static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * STACK_SLOTS +
                        curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                    GemmCoord actualBlockShapeQK{rowNum, desc.stackSeqTile, embed};
                    LayoutS layOutS(rowNum, desc.stackSeqTile, stackSeqTilePad);
#ifdef __DAV_C220_CUBE__
                    if constexpr (PAGED_CACHE_FLAG) {
                        blockMmadQK(
                            gQ[gmOffsetQ],
                            gK[gmOffsetK],
                            gS[gmOffsetS],
                            gBlockTable[blockBOffset],
                            layoutQTemp,
                            layoutKTemp,
                            layOutS,
                            actualBlockShapeQK,
                            kvSIdx,
                            kvSLoopNumTotal,
                            pagedBlockSize,
                            strideK);
                    } else {
                        blockMmadQK(
                            gQ[gmOffsetQ],
                            gK[gmOffsetK],
                            gS[gmOffsetS],
                            gBlockTable,
                            layoutQTemp,
                            layoutKTemp,
                            layOutS,
                            actualBlockShapeQK,
                            kvSIdx,
                            kvSLoopNumTotal,
                            pagedBlockSize,
                            strideK);
                    }
                    if(kvSIdx == kvEnd - 1) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_C220_VEC__
                    LayoutP layOutP(rowNum, desc.stackSeqTile, stackSeqTilePad);
                    LayoutMask layOutMask(COMP_TRIU_MASK_DIM_LEN, COMP_TRIU_MASK_DIM_LEN);
                    uint64_t gmOffsetP = gmOffsetS;
                    uint32_t kvSStartIdx = kvSIdx * MAX_KV_STACK_LEN;
                    uint32_t kvSEndIdx = kvSStartIdx + desc.stackSeqTile;
                    epilogueOnlineSoftmax.set_gmOffsetPret(gmOffsetPret + kvSStartIdx);
                    epilogueOnlineSoftmax.set_gmOffsetDrop(gmOffsetDrop + kvSStartIdx / 8);
                    // TODO mask 部分的参数还未完全 通过 desc传递
                    if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL) {
                        int64_t triUp =
                            static_cast<int64_t>(noSkipKvS) - static_cast<int64_t>(qSBlockSize);
                        uint32_t triDown = static_cast<uint32_t>(noSkipKvS);
                        bool doTriUMask = triUp < static_cast<int64_t>(kvSEndIdx - 1U);
                        if (doTriUMask) {
                            if (flashDecodeFlag != 0U) {
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    gMask,
                                    layOutP,
                                    layOutS,
                                    layOutMask,
                                    actualBlockShapeQK,
                                    desc.firstStack,
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
                                    desc.taskStateSlot,
                                    qkReady,
                                    triUp,
                                    triDown,
                                    kvSStartIdx,
                                    kvSEndIdx,
                                    isSplitKV);
                            } else {
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    gMask,
                                    layOutP,
                                    layOutS,
                                    layOutMask,
                                    actualBlockShapeQK,
                                    desc.firstStack,
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
                                    desc.taskStateSlot,
                                    qkReady,
                                    triUp,
                                    triDown,
                                    kvSStartIdx,
                                    kvSEndIdx,
                                    false);
                            }
                        } else {
                            uint32_t noMaskStackSeqNum = (triUp + 1) / MAX_KV_STACK_LEN;
                            int32_t lastNoMaskStackId;
                            if (flashDecodeFlag != 0U) {
                                lastNoMaskStackId = (int32_t)noMaskStackSeqNum - 1 - (int32_t)kvStart;
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    layOutP,
                                    layOutS,
                                    actualBlockShapeQK,
                                    desc.firstStack,
                                    (stackSeqCount == lastNoMaskStackId),
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
                                    desc.taskStateSlot,
                                    qkReady,
                                    isSplitKV);
                            } else {
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    layOutP,
                                    layOutS,
                                    actualBlockShapeQK,
                                    desc.firstStack,
                                    (stackSeqCount == noMaskStackSeqNum - 1),
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
                                    desc.taskStateSlot,
                                    qkReady,
                                    false);
                            }
                        }
                    } else if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_SWA) {
                        // StartLen can be negative when Sq>>Sk; compare in int32 so
                        // it is not promoted against uint32 kvSStartIdx/kvSEndIdx.
                        int32_t kvStartI = static_cast<int32_t>(kvSStartIdx);
                        int32_t kvEndI = static_cast<int32_t>(kvSEndIdx);
                        bool doTriUPreMask = (maskType != 2 || notPreMask) ? false :
                            (windowSizeLeftStartLen >= kvStartI && windowSizeLeftStartLen < kvEndI) ||
                            (windowSizeLeftEndLen > kvStartI && windowSizeLeftEndLen <= kvEndI) ||
                            (windowSizeLeftStartLen <= kvStartI && windowSizeLeftEndLen >= kvEndI);
                        bool doTriUNextMask = (maskType != 2 || notNextMask) ? false :
                            (windowSizeRightStartLen >= kvStartI && windowSizeRightStartLen < kvEndI) ||
                            (windowSizeRightEndLen > kvStartI && windowSizeRightEndLen <= kvEndI) ||
                            (windowSizeRightStartLen <= kvStartI && windowSizeRightEndLen >= kvEndI);
                        bool doTriUMask = (doTriUPreMask || doTriUNextMask);
                        if (doTriUMask) {
                            startsWithMaskTile = true;
                            startsWithMaskThenNomaskFlag = true;
                            epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    gMask,
                                    layOutP,
                                    layOutS,
                                    layOutMask,
                                    actualBlockShapeQK,
                                    desc.firstStack,
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
                                    desc.taskStateSlot,
                                    qkReady,
                                    kvSStartIdx,
                                    doTriUPreMask,
                                    doTriUNextMask,
                                    windowSizeLeftStartLen,
                                    windowSizeLeftEndLen,
                                    windowSizeRightStartLen,
                                    windowSizeRightEndLen,
                                    (flashDecodeFlag != 0U) ? isSplitKV : false);
                        } else {
                            bool isLastNoMaskStackTile = (windowSizeRightStartLen >= kvSeqlen) || (windowSizeRightStartLen < 0);
                            uint32_t kvSeqlenLimit = isLastNoMaskStackTile ? kvSeqlen : windowSizeRightStartLen;
                            uint32_t alignedKvSeqlenLimit = isLastNoMaskStackTile ? RoundUp(kvSeqlenLimit, MAX_KV_STACK_LEN) : RoundDown(kvSeqlenLimit, MAX_KV_STACK_LEN);
                            uint32_t noMaskStackSeqNum = (alignedKvSeqlenLimit - kvStart * MAX_KV_STACK_LEN) / MAX_KV_STACK_LEN;
                            epilogueOnlineSoftmax(
                                gP[gmOffsetP],
                                gS[gmOffsetS],
                                layOutP,
                                layOutS,
                                actualBlockShapeQK,
                                desc.firstStack,
                                (stackSeqCount == noMaskStackSeqNum - 1),
                                qSBlockSize,
                                qNBlockSize,
                                curStackTileMod,
                                desc.taskStateSlot,
                                qkReady,
                                (flashDecodeFlag != 0U) ? isSplitKV : false,
                                startsWithMaskTile,
                                startsWithMaskThenNomaskFlag);
                            startsWithMaskTile = false;
                        }
                    } else {
                        if (flashDecodeFlag != 0U) {
                            epilogueOnlineSoftmax(
                                gP[gmOffsetP],
                                gS[gmOffsetS],
                                layOutP,
                                layOutS,
                                actualBlockShapeQK,
                                desc.firstStack,
                                0,
                                qSBlockSize,
                                qNBlockSize,
                                curStackTileMod,
                                desc.taskStateSlot,
                                qkReady,
                                isSplitKV);
                        } else {
                            epilogueOnlineSoftmax(
                                gP[gmOffsetP],
                                gS[gmOffsetS],
                                layOutP,
                                layOutS,
                                actualBlockShapeQK,
                                desc.firstStack,
                                0,
                                qSBlockSize,
                                qNBlockSize,
                                curStackTileMod,
                                desc.taskStateSlot,
                                qkReady,
                                false);
                        }
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                }
                if (issuedStackCount >= PRE_LAUNCH) {
                    StackDescriptor &pvDesc = descriptors[(issuedStackCount - PRE_LAUNCH) % STACK_SLOTS];
                    uint64_t gmOffsetOTmp =
                        static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * STACK_SLOTS +
                        pvDesc.slot * WORKSPACE_BLOCK_SIZE_DB);
                    GemmCoord actualBlockShapePV{pvDesc.rowNum, embedV, pvDesc.stackSeqTile};
                    LayoutOTmp layoutOTmp(pvDesc.rowNum, embedV, embedRoundV);
#ifdef __DAV_C220_CUBE__
                    LayoutP layoutPTemp(pvDesc.rowNum, pvDesc.stackSeqTile, stackSeqTilePad);
                    uint64_t gmOffsetP = coreIdx * WORKSPACE_BLOCK_SIZE_DB * STACK_SLOTS +
                        pvDesc.slot * WORKSPACE_BLOCK_SIZE_DB;
                    if constexpr (PAGED_CACHE_FLAG) {
                        blockMmadPV(
                            gP[gmOffsetP],
                            gV[pvDesc.gmOffsetV],
                            gOTmp[gmOffsetOTmp],
                            gBlockTable[pvDesc.blockBOffset],
                            layoutPTemp,
                            layoutVTemp,
                            layoutOTmp,
                            actualBlockShapePV,
                            pvDesc.kvSIdx,
                            pvDesc.kvSLoopNumTotal,
                            pagedBlockSize,
                            pvDesc.noSkipKvS,
                            strideV,
                            blockStackNum,
                            softmaxReady);
                    } else {
                        blockMmadPV(
                            gP[gmOffsetP],
                            gV[pvDesc.gmOffsetV],
                            gOTmp[gmOffsetOTmp],
                            gBlockTable,
                            layoutPTemp,
                            layoutVTemp,
                            layoutOTmp,
                            actualBlockShapePV,
                            pvDesc.kvSIdx,
                            pvDesc.kvSLoopNumTotal,
                            pagedBlockSize,
                            pvDesc.noSkipKvS,
                            strideV,
                            blockStackNum,
                            softmaxReady);
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_C220_VEC__
                    LayoutO layoutO(pvDesc.qSeqlen, embed * qHeads);
                    LayoutUpdate layoutUpdate(pvDesc.rowNum, embed, embedRound);
                    // Head-major LSE: (num_heads, headStride). stride(0)=headStride is
                    // maxQSeqlen (BSND) or totalQTokens (TND) — the correct per-layout head
                    // stride, so rescale_o.hpp's gLse[qNIdx * layoutLse.stride(0)] lands right.
                    LayoutLse layoutLse(qHeads,
                        (INPUT_LAYOUT == FaiKenel::inputLayout::TND) ? totalQTokens : maxQSeqlen);
                    uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * WORKSPACE_BLOCK_SIZE_DB);
                    // Arch::CrossCoreWaitFlag(pvReady);
                    
                    // TODO FD 未适配
                    if (flashDecodeFlag != 0U) {
                        LayoutLse layoutgmLse(pvDesc.qSBlockSize, pvDesc.qNBlockSize);
                        LayoutLse layoutgmLo(pvDesc.qSBlockSize, embed * pvDesc.qNBlockSize);
                        typename EpilogueRescaleO::SplitKVParams splitParams;
                        splitParams.isSplitkv = isSplitKV;
                        splitParams.gCombineLse = gLseFD[gmOffsetLseFD];
                        splitParams.gCombineo = gOFD[gmOffsetOFD];
                        splitParams.layoutgmLse = &layoutgmLse;
                        splitParams.layoutgmLo = &layoutgmLo;

                        epilogueRescaleO(
                            gO[pvDesc.gmOffsetO],
                            gOTmp[gmOffsetOTmp],
                            gOUpdate[gmOffsetUpdate],
                            gLse[pvDesc.gmOffsetLse],
                            layoutO,
                            layoutOTmp,
                            layoutUpdate,
                            layoutLse,
                            actualBlockShapePV,
                            pvReady,
                            pvDesc.qSBlockSize,
                            pvDesc.qNBlockSize,
                            pvDesc.firstStack,
                            pvDesc.lastStack,
                            pvDesc.slot,
                            pvDesc.taskStateSlot,
                            splitParams);
                    } else {
                        epilogueRescaleO(
                            gO[pvDesc.gmOffsetO],
                            gOTmp[gmOffsetOTmp],
                            gOUpdate[gmOffsetUpdate],
                            gLse[pvDesc.gmOffsetLse],
                            layoutO,
                            layoutOTmp,
                            layoutUpdate,
                            layoutLse,
                            actualBlockShapePV,
                            pvReady,
                            pvDesc.qSBlockSize,
                            pvDesc.qNBlockSize,
                            pvDesc.firstStack,
                            pvDesc.lastStack,
                            pvDesc.slot,
                            pvDesc.taskStateSlot,
                            typename EpilogueRescaleO::SplitKVParams(),
                            pvDesc.delStartRow,
                            pvDesc.delEndRow,
                            pvDesc.qSeqlen,
                            pvDesc.qSBlockIdx,
                            pvDesc.curQNBlockTile);
                    }
#endif
                }
                stackSeqCount++;
                ++issuedStackCount;
                ++taskStackCount;
            }
        }

    private:
        uint64_t mm1OutSize;
        uint64_t smOnlineOutSize;
        uint64_t mm2OutSize;
        uint32_t batch;
        uint32_t qHeads;
        uint32_t kvHeads;
        uint32_t embed;
        uint32_t embedV;
        uint32_t pagedBlockSize;
        uint32_t maxNumBlocksPerBatch;
        uint32_t firstBatchTaskNum;
        uint32_t totalTaskNum;
        uint32_t blockSize;
        uint32_t maskType;
        int64_t  windowSizeLeft;
        int64_t  windowSizeRight;
        float    scaleValue;
        float    softcapValue;
        float    dropoutValue;
        uint32_t totalQTokens;
        uint32_t maxQSeqlen;
        uint32_t maxKvSeqlen;
        uint32_t flashDecodeFlag;

        uint64_t strideQ;
        uint64_t strideO;
        uint64_t strideK;
        uint64_t strideV;
        uint64_t stridePret;
        uint64_t strideDrop;
        uint32_t embedRound;
        uint32_t embedRoundV;
        uint32_t groupSize;

        Arch::Resource<ArchTag> resource;
        Arch::CrossCoreFlag qkReady{QK_READY_ID};
        Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
        Arch::CrossCoreFlag pvReady{PV_READY_ID};
        Gemm::Block::BlockPingPongState pingPongState;

        BlockMmadQK blockMmadQK;
        BlockMmadPV blockMmadPV;
        EpilogueOnlineSoftmax epilogueOnlineSoftmax;
        EpilogueRescaleO epilogueRescaleO;
    };
}

namespace SplitFuse {
    template <
        typename InputDtypeQ = half,
        typename InputDtypeKv = half,
        typename IntermCalcPrec = float,
        bool PagedCacheFlag = false,
        FaiKenel::MaskType maskCategory = FaiKenel::MaskType::NO_MASK,
        FaiKenel::inputLayout inLayout = FaiKenel::inputLayout::TND,
        Epilogue::LseModeT lseMode = Epilogue::LseModeT::NONE,
        bool HAS_SOFTCAP = false,
        bool RETURN_SOFTMAX = false,
        bool HAS_DROPOUT = false>
    __global__ __aicore__ void FAInfer(
        uint64_t fftsAddr,
        GM_ADDR q,
        GM_ADDR k,
        GM_ADDR v,
        GM_ADDR mask,
        GM_ADDR blockTables,
        GM_ADDR o,
        GM_ADDR lse,
        GM_ADDR actualQseqlen,
        GM_ADDR actualKvseqlen,
        GM_ADDR workspace,
        GM_ADDR tiling)
    {
        AscendC::SetSyncBaseAddr(fftsAddr);

        using ArchTag = Arch::AtlasA2;
        using ElementQ = InputDtypeQ;
        using LayoutQ = layout::RowMajor;
        using ElementK = InputDtypeKv;
        using LayoutK = layout::ColumnMajor;
        using ElementV = InputDtypeKv;
        using LayoutV = layout::RowMajor;
        using ElementS = IntermCalcPrec;
        using LayoutS = layout::RowMajor;
        using ElementP = InputDtypeQ;
        using LayoutP = layout::RowMajor;
        using ElementO = InputDtypeQ;
        using LayoutO = layout::RowMajor;
        using ElementLse = float;
        using LayoutLse = layout::RowMajor;
        using ElementMask = uint8_t;
        using LayoutMask = layout::RowMajor;
        using ElementOTmp = IntermCalcPrec;
        using LayoutOTmp = layout::RowMajor;
        using ElementUpdate = IntermCalcPrec;
        using LayoutUpdate = layout::RowMajor;

        using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
        using L0TileShapeQK = GemmShape<128, 128, 128>;
        using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQKT<PagedCacheFlag, false>;
        using QType = Gemm::GemmType<ElementQ, LayoutQ>;
        using KType = Gemm::GemmType<ElementK, LayoutK>;
        using SType = Gemm::GemmType<ElementS, LayoutS>;
        using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK,
                                                   QType, KType, SType>;

        using DispatchPolicyOnlineSoftmax =
            Epilogue::EpilogueAtlasA2OnlineSoftmaxT<lseMode, IntermCalcPrec, HAS_SOFTCAP, RETURN_SOFTMAX, HAS_DROPOUT>;
        using PType = Gemm::GemmType<ElementP, LayoutP>;
        using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
        using EpilogueOnlineSoftmax =
            Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType>;

        using L1TileShapePV = GemmShape<128, 128, 256>;
        using L0TileShapePV = GemmShape<128, 128, 128>;
        using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPVT<PagedCacheFlag, false>;
        using VType = Gemm::GemmType<ElementV, LayoutV>;
        using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
        using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV,
                                                   PType, VType, OTmpType>;

        using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleOT<lseMode, IntermCalcPrec, HAS_DROPOUT>;
        using OType = Gemm::GemmType<ElementO, LayoutO>;
        using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
        using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
        using EpilogueRescaleO =
            Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;


        using SplitInfoType = splitNode[25];
        using CombineScale = Epilogue::Block::CombineScale<OType, LseType, SplitInfoType>;
        using FAInferKernelType =
            FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO,
                          PagedCacheFlag, maskCategory, inLayout, CombineScale>;

        FAIKernelParams params{q, k, v, mask, blockTables, actualQseqlen, actualKvseqlen, o, lse, workspace, tiling};
        FAInferKernelType flashAttnInfer;
        flashAttnInfer(params);
    }
}
