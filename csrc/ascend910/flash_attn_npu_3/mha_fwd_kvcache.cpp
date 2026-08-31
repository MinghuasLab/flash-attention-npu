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
#include "combine_scale.hpp"
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
            // Append-KV newkv tensors.
            AscendC::GlobalTensor<ElementK>& gKNew;
            AscendC::GlobalTensor<ElementK>& gVNew;
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
            maxQSeqlen = fATilingData->maxQSeqlen;
            flashDecodeFlag = fATilingData->flashDecodeFlag;
            kvNewSeqlen = fATilingData->kvNewSeqlen;
            kvCacheSeqlen = fATilingData->kvCacheSeqlen;

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

            // Append-KV newkv tensors (nullptr when kvNewSeqlen == 0).
            AscendC::GlobalTensor<ElementK> gKNew;
            gKNew.SetGlobalBuffer((__gm__ ElementK *)params.kNew);
            AscendC::GlobalTensor<ElementK> gVNew;
            gVNew.SetGlobalBuffer((__gm__ ElementK *)params.vNew);

            GlobalTensorBundle globalTensors{
                gQ, gK, gV, gMask, gBlockTable,
                gActualQseqlen, gActualKvseqlen,
                gO, gLse, gLseFD, gOFD,
                gS, gP, gOTmp, gOUpdate,
                gKNew, gVNew
            };

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
            uint32_t kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M;
            // Append-KV QK writeback staging (ND copy of the new-K sub-tile), placed after
            // the QK/PV shared K/P region and the PV V region (the K and P tiles share one
            // L1 region across the QK/PV phases, so it must not be counted twice).
            uint32_t ndCopyL1Offset = L1_QK_SIZE +
                nDynNum * kDynNum * sizeof(ElementK) * DOUBLE_BUFFER +
                embedV * MAX_KV_STACK_LEN * sizeof(ElementV);
            blockMmadQK.init(resource, nDynNum, kDynNum, MAX_KV_STACK_LEN, 0, ndCopyL1Offset);
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

            epilogueOnlineSoftmax.init(resource, scaleValue, softcapValue);
            epilogueRescaleO.init(resource);

            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif
            strideQ = static_cast<uint64_t>(qHeads * embed);
            strideO = static_cast<uint64_t>(qHeads * embedV);
            strideK = static_cast<uint64_t>(kvHeads * embed);
            strideV = static_cast<uint64_t>(kvHeads * embedV);
            embedRound = RoundUp(embed, FaiKenel::BLOCK_SIZE);
            embedRoundV = RoundUp(embedV, FaiKenel::BLOCK_SIZE);
            groupSize = qHeads / kvHeads;

            totalQTokens = 0;
            if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                totalQTokens = static_cast<uint32_t>(gActualQseqlen.GetValue(batch));
            }

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
                    uint32_t kvSeqlenCur = static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx)) + kvNewSeqlen;
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
                    // Append: per-phase S2 block count
                    uint32_t curKSBlockNumTmp = CeilDiv(kvSeqlenCur, MAX_KV_STACK_LEN);
                    if (kvNewSeqlen != 0U) {
                        curKSBlockNumTmp = CeilDiv(
                            static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx)), MAX_KV_STACK_LEN) +
                            CeilDiv(kvNewSeqlen, MAX_KV_STACK_LEN);
                    }

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

                            runMainLoop(
                                coreIdx, BIdx, (uint32_t)n1Idx, (uint32_t)s1Idx,
                                isSplitKV, stS2IdxNow, enS2IdxNow,
                                gmOffsetLseFD, gmOffsetOFD,
                                globalTensors
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
                for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
                    uint32_t curBatchTmp = 0;
                    uint32_t preTotalTaskNumTmp = 0;
                    uint32_t curTotalTaskNumTmp = firstBatchTaskNum;

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

                    uint32_t qSeqlenCur = fATilingData->maxQSeqlen;
                    uint32_t kvSeqlenCur = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatchTmp));
                    if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                        uint32_t prevQSeqlenSumCur = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatchTmp));
                        qSeqlenCur = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatchTmp + 1)) - prevQSeqlenSumCur;
                        if constexpr (!PAGED_CACHE_FLAG) {
                            uint32_t prevKvSeqlenSumCur = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatchTmp));
                            kvSeqlenCur = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatchTmp + 1)) - prevKvSeqlenSumCur;
                        }
                    }
                    uint32_t curQNBlockTileCur = GetQNBlockTile(qSeqlenCur, groupSize);
                    uint32_t qNBlockNumPerGroupCur = CeilDiv(groupSize, curQNBlockTileCur);
                    uint32_t curQNBlockNumCur = qNBlockNumPerGroupCur * kvHeads;

                    uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNumTmp;
                    uint32_t qSBlockIdxCur = taskIdxCurBatch / curQNBlockNumCur;
                    uint32_t qNBlockIdxCur = taskIdxCurBatch - qSBlockIdxCur * curQNBlockNumCur;

                    runMainLoop(
                        coreIdx, curBatchTmp, qNBlockIdxCur, qSBlockIdxCur,
                        false, 0, 0,
                        0, 0,
                        globalTensors
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
            GlobalTensorBundle& globalTensors
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
            auto& gKNew = globalTensors.gKNew;
            auto& gVNew = globalTensors.gVNew;

            // ? maybe change to template later
            const bool appendKVFlag = (kvNewSeqlen != 0U);

            uint32_t qSeqlen = maxQSeqlen;
            // Append KV: total S = old + new.
            uint32_t kvSeqlenOld = static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx));
            uint32_t kvSeqlen = kvSeqlenOld + (appendKVFlag ? kvNewSeqlen : 0U);
            uint32_t prevQSeqlenSum = 0;
            uint32_t prevKvSeqlenSum = 0;

            if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                prevQSeqlenSum = static_cast<uint32_t>(gActualQseqlen.GetValue(BIdx));
                qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(BIdx + 1)) - prevQSeqlenSum;
                if constexpr (!PAGED_CACHE_FLAG) {
                    // gActualKvseqlen holds prefix sums; append uses the capacity-aligned
                    // per-batch cache layout, so the offset is kvCacheSeqlen-stepped.
                    kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx + 1)) -
                        static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx));
                    kvSeqlenOld = kvSeqlen;
                    prevKvSeqlenSum = appendKVFlag ?
                        BIdx * kvCacheSeqlen : static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx));
                }
            } else {
                // BSND: Q/O/LSE per-batch storage step is maxQSeqlen.
                prevQSeqlenSum = BIdx * maxQSeqlen;
                if constexpr (!PAGED_CACHE_FLAG) {
                    if (appendKVFlag) {
                        // Capacity-aligned cache: cache batch occupies [BIdx * kvCacheSeqlen, ...).
                        prevKvSeqlenSum = BIdx * kvCacheSeqlen;
                    } else {
                        // Mirror mha_fwd_kvcache_2.cpp semantics: per-batch K/V step
                        // uses each batch's actual kvSeqlen (prefix sum across batches).
                        for (uint32_t b = 0; b < BIdx; b++) {
                            prevKvSeqlenSum += static_cast<uint32_t>(gActualKvseqlen.GetValue(b));
                        }
                    }
                }
            }
            // TND non-paged read the per-batch old length from prefix sums; re-add new.
            if (appendKVFlag) {
                kvSeqlen = kvSeqlenOld + kvNewSeqlen;
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
            // LSE flat offset depends on the output layout the host allocates:
            //   BSND -> (batch, num_heads, seqlen) batch-major: head stride = maxQSeqlen,
            //           each batch occupies prevQSeqlenSum * qHeads elements.
            //   TND  -> (num_heads, total_q) head-major: head stride = totalQTokens,
            //           per-batch tokens are concatenated along the fast dim, so the
            //           per-batch offset is just prevQSeqlenSum (not scaled by qHeads).
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
            // Append: per-phase S2 block count (same as fa_split getBatchParams),
            if (appendKVFlag) {
                curKSBlockNum = CeilDiv(kvSeqlenOld, MAX_KV_STACK_LEN) +
                    CeilDiv(kvNewSeqlen, MAX_KV_STACK_LEN);
            }

            uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
            uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;
            uint32_t qNStartIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;

            uint64_t gmOffsetQ = qBOffset +
                static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideQ +
                static_cast<uint64_t>(qNStartIdx * embed);
            uint64_t gmOffsetK = kBOffset + static_cast<uint64_t>(kvNIdx * embed);
            uint64_t gmOffsetV = vBOffset + static_cast<uint64_t>(kvNIdx * embedV);
            // newkv are (b, s_new, h_k, d); batch b starts at b * kvNewSeqlen tokens.
            uint64_t gmOffsetKNew = appendKVFlag ?
                static_cast<uint64_t>(BIdx) * kvNewSeqlen * strideK + static_cast<uint64_t>(kvNIdx * embed) : 0;
            uint64_t gmOffsetVNew = appendKVFlag ?
                static_cast<uint64_t>(BIdx) * kvNewSeqlen * strideV + static_cast<uint64_t>(kvNIdx * embedV) : 0;
            uint64_t gmOffsetO = oBOffset +
                static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideO +
                static_cast<uint64_t>(qNStartIdx * embedV);
            uint64_t gmOffsetLse = lseBOffset +
                static_cast<uint64_t>(qNStartIdx) * lseHeadStride +
                static_cast<uint64_t>(qSBlockIdx * curQSBlockTile);

            uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1U)) ?
                (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;
            uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1U)) ?
                (groupSize - qNBlockIdxCurGroup * curQNBlockTile) : curQNBlockTile;
            uint32_t rowNum = qSBlockSize * qNBlockSize;

            int64_t noSkipKvS = static_cast<int64_t>(kvSeqlen);
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
                    noSkipKvS = AscendC::Std::min(static_cast<int32_t>(kvSeqlen), RoundUp(windowSizeRightEndLen, static_cast<int32_t>(MAX_KV_STACK_LEN)));
                    noSkipKvS = noSkipKvS <= 0 ? kvSeqlen : noSkipKvS;
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
                kvSLoopNumTotal = CeilDiv(kvSeqlen, MAX_KV_STACK_LEN);
            }

            // Append-KV phase split: [0, kvStartOld) reads the cache,
            // [kvStartOld, kvEnd) reads newkv and writes it back to the cache
            uint32_t kvStartOld = 0;
            uint32_t kvLoopNumTotalNew = 0;
            uint32_t kvSLoopNumTotalOld = kvSLoopNumTotal;
            if (appendKVFlag) {
                kvStartOld = CeilDiv(kvSeqlenOld, MAX_KV_STACK_LEN);
                kvSLoopNumTotalOld = CeilDiv(
                    AscendC::Std::max((int64_t)0,
                        AscendC::Std::min(noSkipKvS, (int64_t)kvSeqlenOld)),
                    (int64_t)MAX_KV_STACK_LEN);
                kvLoopNumTotalNew = CeilDiv(kvNewSeqlen, MAX_KV_STACK_LEN);
                kvSLoopNumTotal = kvSLoopNumTotalOld + kvLoopNumTotalNew;
            }

            uint32_t kvEnd = kvSLoopNumTotal;
            // Match FA GPU split-KV (flash_fwd_kernel.h compute_attn_1rowblock_splitkv):
            // n_block_min = max(split_lo, window_lo), n_block_max = min(split_hi, window_hi).
            // Do NOT overwrite SWA/causal bounds with the FD S2 range alone.
            uint32_t winKvStart = kvStart;
            uint32_t winKvEnd = kvSLoopNumTotal;
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

            // Empty after split\cap window (GPU: n_block_min >= n_block_max early exit).
            // Non-split: zero final O / +inf LSE. Split partials are host-inited to 0 / -inf.
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
            for (uint32_t kvSIdx = kvStart; kvSIdx < kvEnd + preKVNum; kvSIdx++) {
                if (kvSIdx < kvEnd) {
                    // oldkvcache + newkv, tasks dispatch to each cores
                    const bool isAppendBlock = appendKVFlag && (kvSIdx >= kvStartOld);
                    const bool doCopyback = isAppendBlock && (qSBlockIdx == (BIdx % curQSBlockNum));
                    const uint32_t kvSIdxLocal = isAppendBlock ? (kvSIdx - kvStartOld) : kvSIdx;
                    AscendC::GlobalTensor<ElementK> gQKSrc = isAppendBlock ? gKNew : gK;
                    uint64_t qkSrcOffset = isAppendBlock ? gmOffsetKNew : gmOffsetK;
                    // Row offset within the cache row.
                    uint64_t qkCacheOffset = doCopyback ?
                        (uint64_t)(kvSeqlenOld + kvSIdxLocal * MAX_KV_STACK_LEN) : 0;
                    stackSeqTile = GetStackSeqTile(isAppendBlock, kvSIdx, kvSIdxLocal,
                                                   kvLoopNumTotalNew, kvSLoopNumTotalOld,
                                                   noSkipKvS, kvSeqlenOld, kvNewSeqlen);
                    uint32_t curStackTileMod = stackSeqCount % (PRE_LAUNCH + 1U);
                    uint64_t gmOffsetS =
                        static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                        curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                    LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
#ifdef __DAV_C220_CUBE__
                    if constexpr (PAGED_CACHE_FLAG) {
                        blockMmadQK(
                            gQ[gmOffsetQ],
                            gQKSrc[qkSrcOffset],
                            gS[gmOffsetS],
                            isAppendBlock ? gBlockTable : gBlockTable[blockBOffset],
                            layoutQTemp,
                            layoutKTemp,
                            layOutS,
                            actualBlockShapeQK,
                            kvSIdxLocal,
                            kvSLoopNumTotal,
                            pagedBlockSize,
                            strideK,
                            doCopyback,
                            gK[gmOffsetK],
                            qkCacheOffset,
                            gBlockTable,
                            isAppendBlock ? pagedBlockSize : 0U,
                            static_cast<uint32_t>(blockBOffset));
                    } else {
                        blockMmadQK(
                            gQ[gmOffsetQ],
                            gQKSrc[qkSrcOffset],
                            gS[gmOffsetS],
                            gBlockTable,
                            layoutQTemp,
                            layoutKTemp,
                            layOutS,
                            actualBlockShapeQK,
                            kvSIdxLocal,
                            kvSLoopNumTotal,
                            pagedBlockSize,
                            strideK,
                            doCopyback,
                            gK[gmOffsetK],
                            qkCacheOffset);
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_C220_VEC__
                    LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                    LayoutMask layOutMask(COMP_TRIU_MASK_DIM_LEN, COMP_TRIU_MASK_DIM_LEN);
                    uint64_t gmOffsetP = gmOffsetS;
                    // Append new-phase tiles live at cache rows old + local*512; the mask
                    // window must follow the data (logical kvSIdx*512 would drift past the
                    // diagonal when the cached length is not 512-aligned).
                    uint32_t kvSStartIdx = isAppendBlock ?
                        (kvSeqlenOld + kvSIdxLocal * MAX_KV_STACK_LEN) : kvSIdx * MAX_KV_STACK_LEN;
                    uint32_t kvSEndIdx = kvSStartIdx + stackSeqTile;
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
                                    (stackSeqCount == 0),
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
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
                                    (stackSeqCount == 0),
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
                                    qkReady,
                                    triUp,
                                    triDown,
                                    kvSStartIdx,
                                    kvSEndIdx,
                                    false);
                            }
                        } else {
                            // Last nomask tile before the diagonal tile, append phase
                            // split aware (a tile ending exactly at triUp+1 is nomask).
                            uint32_t lastNoMaskTile;
                            if (triUp + 1U == kvSeqlen) {
                                lastNoMaskTile = kvSLoopNumTotal - 1U;
                            } else if (triUp + 1U == kvSeqlenOld) {
                                lastNoMaskTile = kvStartOld - 1U;
                            } else if (triUp + 1U < kvSeqlenOld) {
                                lastNoMaskTile = (triUp + 1U) / MAX_KV_STACK_LEN - 1U;
                            } else {
                                lastNoMaskTile = kvStartOld +
                                    (triUp + 1U - kvSeqlenOld) / MAX_KV_STACK_LEN - 1U;
                            }
                            Arch::CrossCoreWaitFlag(qkReady);
                            int32_t lastNoMaskStackId;
                            if (flashDecodeFlag != 0U) {
                                lastNoMaskStackId = (int32_t)lastNoMaskTile - (int32_t)kvStart;
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    layOutP,
                                    layOutS,
                                    actualBlockShapeQK,
                                    (stackSeqCount == 0),
                                    (stackSeqCount == lastNoMaskStackId),
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
                                    isSplitKV);
                            } else {
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    layOutP,
                                    layOutS,
                                    actualBlockShapeQK,
                                    (stackSeqCount == 0),
                                    (stackSeqCount == lastNoMaskTile),
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
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
                                    (stackSeqCount == 0),
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
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
                            // Relative to this core's [kvStart, kvEnd) after FD\cap window.
                            uint32_t noMaskStackSeqNum = (alignedKvSeqlenLimit - kvStart * MAX_KV_STACK_LEN) / MAX_KV_STACK_LEN;
                            Arch::CrossCoreWaitFlag(qkReady);
                            epilogueOnlineSoftmax(
                                gP[gmOffsetP],
                                gS[gmOffsetS],
                                layOutP,
                                layOutS,
                                actualBlockShapeQK,
                                (stackSeqCount == 0),
                                (stackSeqCount == noMaskStackSeqNum - 1),
                                qSBlockSize,
                                qNBlockSize,
                                curStackTileMod,
                                (flashDecodeFlag != 0U) ? isSplitKV : false,
                                startsWithMaskTile,
                                startsWithMaskThenNomaskFlag);
                            startsWithMaskTile = false;
                        }
                    } else {
                        Arch::CrossCoreWaitFlag(qkReady);
                        if (flashDecodeFlag != 0U) {
                            epilogueOnlineSoftmax(
                                gP[gmOffsetP],
                                gS[gmOffsetS],
                                layOutP,
                                layOutS,
                                actualBlockShapeQK,
                                (stackSeqCount == 0),
                                0,
                                qSBlockSize,
                                qNBlockSize,
                                curStackTileMod,
                                isSplitKV);
                        } else {
                            epilogueOnlineSoftmax(
                                gP[gmOffsetP],
                                gS[gmOffsetS],
                                layOutP,
                                layOutS,
                                actualBlockShapeQK,
                                (stackSeqCount == 0),
                                0,
                                qSBlockSize,
                                qNBlockSize,
                                curStackTileMod,
                                false);
                        }
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                }
                if (kvSIdx >= kvStart + preKVNum) {
                    uint32_t nowkvSIdx = kvSIdx - preKVNum;
                    // PV-side phase split (nowkvSIdx is preKVNum behind kvSIdx); dedup like QK.
                    const bool isAppendBlockPV = appendKVFlag && (nowkvSIdx >= kvStartOld);
                    const bool doCopybackPV = isAppendBlockPV && (qSBlockIdx == (BIdx % curQSBlockNum));
                    uint32_t nowKvSIdxLocal =
                        isAppendBlockPV ? (nowkvSIdx - kvStartOld) : nowkvSIdx;
                    AscendC::GlobalTensor<ElementK> gPVSrc = isAppendBlockPV ? gVNew : gV;
                    uint64_t pvSrcOffset = isAppendBlockPV ? gmOffsetVNew : gmOffsetV;
                    uint64_t pvCacheOffset = doCopybackPV ?
                        (uint64_t)(kvSeqlenOld + nowKvSIdxLocal * MAX_KV_STACK_LEN) : 0;
                    stackSeqTile = GetStackSeqTile(isAppendBlockPV, nowkvSIdx, nowKvSIdxLocal,
                                                   kvLoopNumTotalNew, kvSLoopNumTotalOld,
                                                   noSkipKvS, kvSeqlenOld, kvNewSeqlen);
                    uint32_t curStackTileMod = (stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1U);
                    uint64_t gmOffsetOTmp =
                        static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                        curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                    GemmCoord actualBlockShapePV{rowNum, embedV, stackSeqTile};
                    LayoutOTmp layoutOTmp(rowNum, embedV, embedRoundV);
#ifdef __DAV_C220_CUBE__
                    LayoutP layoutPTemp(rowNum, stackSeqTile, stackSeqTilePad);
                    uint64_t gmOffsetP = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                        curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                    if constexpr (PAGED_CACHE_FLAG) {
                        blockMmadPV(
                            gP[gmOffsetP],
                            gPVSrc[pvSrcOffset],
                            gOTmp[gmOffsetOTmp],
                            isAppendBlockPV ? gBlockTable : gBlockTable[blockBOffset],
                            layoutPTemp,
                            layoutVTemp,
                            layoutOTmp,
                            actualBlockShapePV,
                            nowKvSIdxLocal,
                            kvSLoopNumTotal,
                            pagedBlockSize,
                            noSkipKvS,
                            strideV,
                            blockStackNum,
                            softmaxReady,
                            doCopybackPV,
                            gV[gmOffsetV],
                            pvCacheOffset,
                            gBlockTable,
                            isAppendBlockPV ? pagedBlockSize : 0U,
                            static_cast<uint32_t>(blockBOffset));
                    } else {
                        blockMmadPV(
                            gP[gmOffsetP],
                            gPVSrc[pvSrcOffset],
                            gOTmp[gmOffsetOTmp],
                            gBlockTable,
                            layoutPTemp,
                            layoutVTemp,
                            layoutOTmp,
                            actualBlockShapePV,
                            nowKvSIdxLocal,
                            kvSLoopNumTotal,
                            pagedBlockSize,
                            noSkipKvS,
                            strideV,
                            blockStackNum,
                            softmaxReady,
                            doCopybackPV,
                            gV[gmOffsetV],
                            pvCacheOffset);
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_C220_VEC__
                    LayoutO layoutO(qSeqlen, embed * qHeads);
                    LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                    LayoutLse layoutLse(qHeads,
                        (INPUT_LAYOUT == FaiKenel::inputLayout::TND) ? totalQTokens : maxQSeqlen);
                    uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * WORKSPACE_BLOCK_SIZE_DB);
                    Arch::CrossCoreWaitFlag(pvReady);

                    if (flashDecodeFlag != 0U) {
                        LayoutLse layoutgmLse(qSBlockSize, qNBlockSize);
                        LayoutLse layoutgmLo(qSBlockSize, embed * qNBlockSize);
                        typename EpilogueRescaleO::SplitKVParams splitParams;
                        splitParams.isSplitkv = isSplitKV;
                        splitParams.gCombineLse = gLseFD[gmOffsetLseFD];
                        splitParams.gCombineo = gOFD[gmOffsetOFD];
                        splitParams.layoutgmLse = &layoutgmLse;
                        splitParams.layoutgmLo = &layoutgmLo;

                        epilogueRescaleO(
                            gO[gmOffsetO],
                            gOTmp[gmOffsetOTmp],
                            gOUpdate[gmOffsetUpdate],
                            gLse[gmOffsetLse],
                            layoutO,
                            layoutOTmp,
                            layoutUpdate,
                            layoutLse,
                            actualBlockShapePV,
                            qSBlockSize,
                            qNBlockSize,
                            (stackSeqCount - PRE_LAUNCH == 0),
                            nowkvSIdx + 1 >= kvEnd,
                            curStackTileMod,
                            splitParams);
                    } else {
                        epilogueRescaleO(
                            gO[gmOffsetO],
                            gOTmp[gmOffsetOTmp],
                            gOUpdate[gmOffsetUpdate],
                            gLse[gmOffsetLse],
                            layoutO,
                            layoutOTmp,
                            layoutUpdate,
                            layoutLse,
                            actualBlockShapePV,
                            qSBlockSize,
                            qNBlockSize,
                            (stackSeqCount - PRE_LAUNCH == 0),
                            nowkvSIdx + 1 >= kvSLoopNumTotal,
                            curStackTileMod,
                            typename EpilogueRescaleO::SplitKVParams(),
                            delStartRow,
                            delEndRow,
                            qSeqlen,
                            qSBlockIdx,
                            curQNBlockTile);
                    }
#endif
                }
                stackSeqCount++;
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
        uint32_t totalQTokens;
        uint32_t maxQSeqlen;
        uint32_t flashDecodeFlag;

        uint32_t kvNewSeqlen;
        uint32_t kvCacheSeqlen;

        uint64_t strideQ;
        uint64_t strideO;
        uint64_t strideK;
        uint64_t strideV;
        uint32_t embedRound;
        uint32_t embedRoundV;
        uint32_t groupSize;

        Arch::Resource<ArchTag> resource;
        Arch::CrossCoreFlag qkReady{QK_READY_ID};
        Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
        Arch::CrossCoreFlag pvReady{PV_READY_ID};

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
        bool HAS_SOFTCAP = false>
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
        GM_ADDR tiling,
        GM_ADDR kNew,
        GM_ADDR vNew)
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

        using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmaxT<lseMode, IntermCalcPrec, HAS_SOFTCAP>;
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

        using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleOT<lseMode, IntermCalcPrec>;
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

        FAIKernelParams params{q, k, v, mask, blockTables, actualQseqlen, actualKvseqlen, o, lse, workspace, tiling,
                               kNew, vNew};
        FAInferKernelType flashAttnInfer;
        flashAttnInfer(params);
    }
}
