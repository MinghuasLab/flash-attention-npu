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
template <class BlockMmadQK, class BlockMmadPV, class EpilogueOnlineSoftmax, class EpilogueRescaleO,
          bool PAGED_CACHE_FLAG, bool FLASH_DECODE_FLAG, FaiKenel::MaskType MASK_TYPE = FaiKenel::MaskType::NO_MASK,
          FaiKenel::inputLayout INPUT_LAYOUT = FaiKenel::inputLayout::BSND, class CombineScale = void>
class FAInferKernel {
  public:
    bool hasSeqUsedQ = false;
    bool hasSeqUsedKv = false;
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
    static constexpr uint32_t SCORE_SLOT_ELEMENTS = Q_TILE_CEIL * MAX_KV_STACK_LEN;

    struct DelayedPVTaskContext {
        uint64_t gmOffsetV;
        uint64_t gmOffsetVNew;
        uint64_t gmOffsetO;
        uint64_t gmOffsetLse;
        // Delayed rescale must retain the producing task's split destination.
        uint64_t gmOffsetLseFD;
        uint64_t gmOffsetOFD;
        uint64_t blockBOffset;
        uint32_t rowNum;
        uint32_t qSeqlen;
        uint32_t qSBlockSize;
        uint32_t qNBlockSize;
        uint32_t kvSLoopNumTotal;
        uint32_t noSkipKvS;
        int32_t invalidSuffixStartRow;
        int32_t invalidPrefixEndRow;
        int32_t qSBlockIdx;
        int32_t curQNBlockTile;
        bool isSplitKV;
        typename EpilogueRescaleO::TaskPlan rescaleTaskPlan;
    };
    struct StackDescriptor {
        uint64_t pvCacheOffset;
        uint32_t stackSeqTile;
        uint32_t kvSIdx;
        uint32_t slot;
        uint32_t taskStateSlot;
        bool isAppendBlock;
        bool doCopyback;
        bool firstStack;
        bool lastStack;
    };

    // Per-batch Q length: seqused_q when provided, else cu prefix diff.
    __aicore__ inline uint32_t QSeqLenOf(uint32_t bIdx, AscendC::GlobalTensor<int32_t>& gCuSeq,
                                         AscendC::GlobalTensor<int32_t>& gSeqUsed) const
    {
        if (hasSeqUsedQ) {
            return static_cast<uint32_t>(gSeqUsed.GetValue(bIdx));
        }
        return static_cast<uint32_t>(gCuSeq.GetValue(bIdx + 1) - gCuSeq.GetValue(bIdx));
    }

    // Per-batch KV length: seqused_k when provided, else cu prefix diff.
    __aicore__ inline uint32_t KvSeqLenOf(uint32_t bIdx, AscendC::GlobalTensor<int32_t>& gCuSeq,
                                          AscendC::GlobalTensor<int32_t>& gSeqUsed) const
    {
        if (hasSeqUsedKv) {
            return static_cast<uint32_t>(gSeqUsed.GetValue(bIdx));
        }
        return static_cast<uint32_t>(gCuSeq.GetValue(bIdx + 1) - gCuSeq.GetValue(bIdx));
    }

    struct GlobalTensorBundle {
        AscendC::GlobalTensor<ElementQ>& gQ;
        AscendC::GlobalTensor<ElementK>& gK;
        AscendC::GlobalTensor<ElementK>& gV;
        AscendC::GlobalTensor<ElementMask>& gMask;
        AscendC::GlobalTensor<int32_t>& gBlockTable;
        AscendC::GlobalTensor<int32_t>& gActualQseqlen;
        AscendC::GlobalTensor<int32_t>& gActualKvseqlen;
        AscendC::GlobalTensor<int32_t>& gSeqUsedQ;
        AscendC::GlobalTensor<int32_t>& gSeqUsedKv;
        AscendC::GlobalTensor<ElementO>& gO;
        AscendC::GlobalTensor<ElementLse>& gLse;
        AscendC::GlobalTensor<ElementLse>& gLseFD;
        AscendC::GlobalTensor<ElementLse>& gOFD;
        AscendC::GlobalTensor<ElementS>& gS;
        AscendC::GlobalTensor<ElementP>& gP;
        AscendC::GlobalTensor<ElementOTmp>& gOTmp;
        AscendC::GlobalTensor<ElementOTmp>& gOUpdate;
        AscendC::GlobalTensor<ElementK>& gKNew;
        AscendC::GlobalTensor<ElementK>& gVNew;
    };

    __aicore__ inline FAInferKernel() {}

    __aicore__ inline void operator()(FAIKernelParams const& params)
    {
        hasSeqUsedQ = params.seqUsedQ != nullptr;
        hasSeqUsedKv = params.seqUsedKv != nullptr;
        __gm__ FAInferTilingData* fATilingData =
            reinterpret_cast<__gm__ FAInferTilingData*>(params.tiling.tilingDataAddr);
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
        scaleValue = params.tiling.scaleValue;
        softcapValue = params.tiling.softcapValue;
        maxQSeqlen = fATilingData->maxQSeqlen;
        maxKvSeqlen = fATilingData->maxKvSeqlen;
        kvNewSeqlen = fATilingData->kvNewSeqlen;
        kvCacheSeqlen = fATilingData->kvCacheSeqlen;
        kvSeqlenAllocated = params.kvSeqlenAllocated;

        // FD workspace sizing: reserve head of workspace for gLseFD/gOFD.
        uint64_t Lsesize = 0;
        uint64_t Losize = 0;
        if constexpr (FLASH_DECODE_FLAG) {
            Lsesize = fATilingData->splitLseTotalSize;
            Losize = fATilingData->splitOTotalSize;
        }

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK*)params.v);
        AscendC::GlobalTensor<ElementMask> gMask;
        gMask.SetGlobalBuffer((__gm__ ElementMask*)params.mask);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t*)(params.blockTables));
        AscendC::GlobalTensor<int32_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int32_t*)params.actualQseqlen);
        AscendC::GlobalTensor<int32_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t*)params.actualKvseqlen);
        AscendC::GlobalTensor<int32_t> gSeqUsedQ;
        gSeqUsedQ.SetGlobalBuffer((__gm__ int32_t*)params.seqUsedQ);
        AscendC::GlobalTensor<int32_t> gSeqUsedKv;
        gSeqUsedKv.SetGlobalBuffer((__gm__ int32_t*)params.seqUsedKv);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse*)params.lse);

        AscendC::GlobalTensor<ElementLse> gLseFD;
        AscendC::GlobalTensor<ElementLse> gOFD;
        if constexpr (FLASH_DECODE_FLAG) {
            gLseFD.SetGlobalBuffer((__gm__ ElementLse*)(params.workSpace));
            gOFD.SetGlobalBuffer((__gm__ ElementLse*)(params.workSpace + Lsesize));
        }

        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS*)(params.workSpace + Lsesize + Losize));
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP*)(params.workSpace + Lsesize + Losize + mm1OutSize));
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer(
            (__gm__ ElementOTmp*)(params.workSpace + Lsesize + Losize + mm1OutSize + smOnlineOutSize));
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer(
            (__gm__ ElementOTmp*)(params.workSpace + Lsesize + Losize + mm1OutSize + smOnlineOutSize + mm2OutSize));

        // Append-KV newkv tensors (nullptr when kvNewSeqlen == 0).
        AscendC::GlobalTensor<ElementK> gKNew;
        gKNew.SetGlobalBuffer((__gm__ ElementK*)params.kNew);
        AscendC::GlobalTensor<ElementK> gVNew;
        gVNew.SetGlobalBuffer((__gm__ ElementK*)params.vNew);
        // Append-KV per-batch cache-batch indices / left padding (GM params).

        GlobalTensorBundle globalTensors{
            gQ,   gK,     gV,   gMask, gBlockTable, gActualQseqlen, gActualKvseqlen, gSeqUsedQ, gSeqUsedKv, gO,
            gLse, gLseFD, gOFD, gS,    gP,          gOTmp,          gOUpdate,        gKNew,     gVNew};

        if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::BNSD) {
            // BNSD: (batch, head, seq, dim). Adjacent tokens within a head are
            // `embed`/`embedV` apart; adjacent heads are maxQSeqlen*embed for Q and
            // kvSeqlenAllocated*embed(V) apart for the KV cache.
            strideQ = static_cast<uint64_t>(embed);
            strideO = static_cast<uint64_t>(embedV);
            strideK = PAGED_CACHE_FLAG ? params.tiling.keyTokenStride : static_cast<uint64_t>(embed);
            strideV = PAGED_CACHE_FLAG ? params.tiling.valueTokenStride : static_cast<uint64_t>(embedV);
            qHeadStride = static_cast<uint64_t>(maxQSeqlen) * embed;
            oHeadStride = static_cast<uint64_t>(maxQSeqlen) * embedV;
            kvHeadStrideK =
                PAGED_CACHE_FLAG ? params.tiling.keyHeadStride : static_cast<uint64_t>(kvSeqlenAllocated) * embed;
            kvHeadStrideV =
                PAGED_CACHE_FLAG ? params.tiling.valueHeadStride : static_cast<uint64_t>(kvSeqlenAllocated) * embedV;
        } else {
            // BSND / TND: (batch|token, seq, head, dim). Adjacent heads are
            // `embed`/`embedV` apart; adjacent tokens are qHeads*embed (Q) or
            // kvHeads*embed(V) (KV).
            strideQ = static_cast<uint64_t>(qHeads * embed);
            strideO = static_cast<uint64_t>(qHeads * embedV);
            strideK = PAGED_CACHE_FLAG ? params.tiling.keyTokenStride : static_cast<uint64_t>(kvHeads * embed);
            strideV = PAGED_CACHE_FLAG ? params.tiling.valueTokenStride : static_cast<uint64_t>(kvHeads * embedV);
            qHeadStride = static_cast<uint64_t>(embed);
            oHeadStride = static_cast<uint64_t>(embedV);
            kvHeadStrideK = PAGED_CACHE_FLAG ? params.tiling.keyHeadStride : static_cast<uint64_t>(embed);
            kvHeadStrideV = PAGED_CACHE_FLAG ? params.tiling.valueHeadStride : static_cast<uint64_t>(embedV);
        }
        // Q/K/V are read with the input layout's strides (BNSD stays zero-copy), but
        // O is always addressed with BSND semantics so the returned tensor is a
        // contiguous (B, S, H, D) block regardless of the input layout. The head
        // plane is D wide and the token stride is H*D in BSND.
        strideO = static_cast<uint64_t>(qHeads * embedV);
        oHeadStride = static_cast<uint64_t>(embedV);

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

        const bool headDimSplit = embed > NUM_256;
        groupSize = qHeads / kvHeads;
        const bool compactDecodeL1 =
            !FLASH_DECODE_FLAG && embed == 512U && embedV == 512U && kvNewSeqlen == 0U && maxQSeqlen * groupSize <= 64U;
        kvStackLen = MAX_KV_STACK_LEN;
        uint32_t kDynNum = NUM_256;
        uint32_t nDynNum = L1_D512_KV_TILE;
        uint32_t qL1DynNum = RoundUp(embed, FaiKenel::BLOCK_SIZE);
        uint32_t kPVDynNum = NUM_256;
        uint32_t vL1Stages = DOUBLE_BUFFER;
        uint32_t vSegLen = NUM_256;
        uint32_t vSliceWidth = L1_HEAD_DIM_SLICE;
        uint32_t qL1Rows = Q_TILE_CEIL;
        uint32_t pL1Rows = BlockMmadPV::L1TileShape::M;
        uint32_t pL1Stages = DOUBLE_BUFFER;
        if (!headDimSplit) {
            // Preserve the production <=256 layout.
            kDynNum = RoundUp(embed, NUM_128);
            kDynNum = kDynNum < NUM_256 ? NUM_256 : kDynNum;
            uint32_t maxQKPL1Size = L1_MAX_SIZE - embedV * kvStackLen * sizeof(ElementV);
            uint32_t maxQL1Size = Q_TILE_CEIL * kDynNum * sizeof(ElementQ);
            uint32_t maxNDynNum =
                ((maxQKPL1Size - maxQL1Size) / kDynNum / sizeof(ElementV) / DOUBLE_BUFFER) / NUM_32 * NUM_32;
            nDynNum = maxNDynNum < L1_MAX_N_NUM ? maxNDynNum : L1_MAX_N_NUM;
            nDynNum = L1_MAX_N_NUM % nDynNum != 0 ? RoundDown((nDynNum - 1), NUM_32) : nDynNum;
            qL1DynNum = kDynNum;
            kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M;
            vL1Stages = 1U;
            vSegLen = kvStackLen;
            vSliceWidth = embedV;
        } else if (compactDecodeL1) {
            qL1Rows = 64U;
            pL1Rows = 64U;
            pL1Stages = 1U;
            kPVDynNum = kvStackLen;
        }

        blockMmadQK.SetPingPongState(&pingPongState);
        blockMmadPV.SetPingPongState(&pingPongState);
        blockMmadQK.SetPageStride(params.tiling.keyPageStride);
        blockMmadPV.SetPageStride(params.tiling.valuePageStride);

        uint32_t L1_QK_SIZE = qL1Rows * qL1DynNum * sizeof(ElementQ);
        uint32_t qkKPoolSize = nDynNum * kDynNum * sizeof(ElementK) * DOUBLE_BUFFER;
        uint32_t kPVPoolSize = pL1Rows * kPVDynNum * sizeof(ElementP) * pL1Stages;
        uint32_t vPoolSize = vL1Stages * vSegLen * vSliceWidth * sizeof(ElementV);
        // Append-KV QK writeback staging (ND copy of the new-K sub-tile), placed after
        // the QK/PV shared K/P region and the PV V region (the K and P tiles share one
        // L1 region across the QK/PV phases, so it must not be counted twice).
        uint32_t pBufOffset = compactDecodeL1 ? (L1_QK_SIZE + qkKPoolSize) : L1_QK_SIZE;
        uint32_t vBufOffset = pBufOffset + kPVPoolSize;
        uint32_t ndCopyL1Offset = compactDecodeL1 ? (vBufOffset + vPoolSize) : (L1_QK_SIZE + kPVPoolSize + vPoolSize);
        blockMmadQK.init(resource, nDynNum, kDynNum, qL1Rows, qL1DynNum, kvStackLen, 0, ndCopyL1Offset);
        if (compactDecodeL1) {
            blockMmadPV.initCompactDecode(resource, pL1Rows, kvStackLen, pBufOffset, vBufOffset);
        } else {
            blockMmadPV.init(resource, nDynNum, kPVDynNum, kvStackLen, L1_QK_SIZE, vL1Stages, vSegLen, vSliceWidth,
                             ndCopyL1Offset);
        }
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
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);

        epilogueOnlineSoftmax.init(resource, scaleValue, softcapValue);
        if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL || MASK_TYPE == FaiKenel::MaskType::MASK_SWA) {
            epilogueOnlineSoftmax.InitArithMask(resource);
        }
        epilogueRescaleO.init(resource);

        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif
        uint32_t softmaxPingPongFlag = 0;
        embedRound = RoundUp(embed, FaiKenel::BLOCK_SIZE);
        embedRoundV = RoundUp(embedV, FaiKenel::BLOCK_SIZE);
        groupSize = qHeads / kvHeads;

        totalQTokens = 0;
        if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
            // LSE is (num_heads, q.size(0)). Host passes that length; it may
            // exceed cu_seqlens_q[-1] when the packed tail is unused.
            totalQTokens = static_cast<uint32_t>(params.tiling.totalQTokens);
        }

        StackDescriptor descriptors[STACK_SLOTS];
        DelayedPVTaskContext taskContexts[STACK_SLOTS];
        uint32_t issuedStackCount = 0;
        uint32_t taskStateSequence = 0;

        if constexpr (FLASH_DECODE_FLAG) {
            const bool idleCoreFD = (fATilingData->needCoreNum != 0U) && (coreIdx >= fATilingData->needCoreNum);
            if (!idleCoreFD) {
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
                        qSeqlenCur = QSeqLenOf(BIdx, gActualQseqlen, gSeqUsedQ);
                        if constexpr (!PAGED_CACHE_FLAG) {
                            kvSeqlenCur = KvSeqLenOf(BIdx, gActualKvseqlen, gSeqUsedKv);
                        }
                    }
                    // Per batch KV length including the appended new KV, reused by every task
                    kvSeqlenCur += (kvNewSeqlen != 0U ? kvNewSeqlen : 0U);

                    uint32_t curQNBlockTileTmp = GetQNBlockTile(qSeqlenCur, groupSize);
                    uint32_t qNBlockNumPerGroupTmp = CeilDiv(groupSize, curQNBlockTileTmp);
                    uint32_t curQNBlockNumTmp = qNBlockNumPerGroupTmp * kvHeads;
                    uint32_t curQSBlockTileTmp = GetQSBlockTile(kvSeqlenCur);
                    uint32_t curQSBlockNumTmp = CeilDiv(qSeqlenCur, curQSBlockTileTmp);
                    uint32_t curKSBlockNumTmp = CeilDiv(kvSeqlenCur, kvStackLen);
                    if (kvNewSeqlen != 0U) {
                        curKSBlockNumTmp = CeilDiv(static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx)), kvStackLen) +
                                           CeilDiv(kvNewSeqlen, kvStackLen);
                    }

                    int32_t stN1IdxNow = (BIdx == startBIdx) ? (int32_t)startN1Idx : 0;
                    int32_t enN1IdxNow = (BIdx == endBIdx) ? (int32_t)endN1Idx : (int32_t)curQNBlockNumTmp - 1;

                    for (int32_t n1Idx = stN1IdxNow; n1Idx <= enN1IdxNow; n1Idx++) {
                        int32_t stS1IdxNow = (BIdx == startBIdx && n1Idx == stN1IdxNow) ? (int32_t)startS1Idx : 0;
                        int32_t enS1IdxNow = (BIdx == endBIdx && n1Idx == enN1IdxNow) ? (int32_t)endS1Idx
                                                                                      : (int32_t)curQSBlockNumTmp - 1;

                        for (int32_t s1Idx = stS1IdxNow; s1Idx <= enS1IdxNow; s1Idx++) {
                            int32_t stS2IdxNow = (BIdx == startBIdx && n1Idx == stN1IdxNow && s1Idx == stS1IdxNow)
                                                     ? (int32_t)startS2Idx
                                                     : 0;
                            int32_t enS2IdxNow = (BIdx == endBIdx && n1Idx == enN1IdxNow && s1Idx == enS1IdxNow)
                                                     ? (int32_t)endS2Idx
                                                     : (int32_t)curKSBlockNumTmp;
                            bool isSplitKV = (enS2IdxNow - stS2IdxNow) > 0 &&
                                             (enS2IdxNow - stS2IdxNow) < static_cast<int32_t>(curKSBlockNumTmp);
                            uint32_t pipelineDrain = 0;
                            if (BIdx == endBIdx && n1Idx == enN1IdxNow && s1Idx == enS1IdxNow) {
                                pipelineDrain = PRE_LAUNCH;
                            }

                            const uint32_t currentTaskStateSlot = taskStateSequence % STACK_SLOTS;
                            const bool hasTaskWork =
                                runMainLoop(coreIdx, BIdx, qSeqlenCur, kvSeqlenCur, (uint32_t)n1Idx, (uint32_t)s1Idx,
                                            isSplitKV, stS2IdxNow, enS2IdxNow, gmOffsetLseFD, gmOffsetOFD,
                                            globalTensors, descriptors, taskContexts, currentTaskStateSlot,
                                            issuedStackCount, pipelineDrain, softmaxPingPongFlag);
                            taskStateSequence += static_cast<uint32_t>(hasTaskWork);

                            if (isSplitKV) {
                                uint32_t qSBlockSizeTmp = (s1Idx == static_cast<int32_t>(curQSBlockNumTmp - 1U))
                                                              ? (qSeqlenCur - s1Idx * curQSBlockTileTmp)
                                                              : curQSBlockTileTmp;
                                uint32_t qNBlockIdxCurGroupTmp = n1Idx % qNBlockNumPerGroupTmp;
                                uint32_t qNBlockSizeTmp = (qNBlockIdxCurGroupTmp == (qNBlockNumPerGroupTmp - 1U))
                                                              ? (groupSize - qNBlockIdxCurGroupTmp * curQNBlockTileTmp)
                                                              : curQNBlockTileTmp;
                                gmOffsetLseFD += qSBlockSizeTmp * qNBlockSizeTmp;
                                gmOffsetOFD += qSBlockSizeTmp * qNBlockSizeTmp * embedV;
                            }
                        }
                    }
                }
            }
        } else {
            // Reuse the batch cursor across monotonically increasing tasks.
            uint32_t curBatchTmp = 0;
            uint32_t preTotalTaskNumTmp = 0;
            uint32_t curTotalTaskNumTmp = firstBatchTaskNum;
            uint32_t cachedBatch = 0xFFFFFFFFU;
            uint32_t curQNBlockNumCur = 0;
            uint32_t curQSBlockNumCur = 0;
            uint32_t qSeqlenCur = 0;
            uint32_t kvSeqlenCur = 0;
            // Resolve batch 0 Q length here since the firstBatchTaskNum seed skips the while advance for batch 0
            qSeqlenCur = fATilingData->maxQSeqlen;
            if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                qSeqlenCur = QSeqLenOf(0, gActualQseqlen, gSeqUsedQ);
            }
            for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
                while (taskIdx >= curTotalTaskNumTmp) {
                    ++curBatchTmp;
                    preTotalTaskNumTmp = curTotalTaskNumTmp;
                    qSeqlenCur = fATilingData->maxQSeqlen;
                    if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                        qSeqlenCur = QSeqLenOf(curBatchTmp, gActualQseqlen, gSeqUsedQ);
                    }

                    // GetQSBlockTile is a constant independent of kvSeqlen so the kv
                    // length is resolved in the cachedBatch gate below, not here. If this
                    // function ever becomes kv dependent move the kv resolution back.
                    uint32_t curQNBlockTileTmp = GetQNBlockTile(qSeqlenCur, groupSize);
                    uint32_t qNBlockNumPerGroupTmp = CeilDiv(groupSize, curQNBlockTileTmp);
                    uint32_t curQNBlockNumTmp = qNBlockNumPerGroupTmp * kvHeads;
                    uint32_t curQSBlockTileTmp = GetQSBlockTile(kvSeqlenCur);
                    uint32_t curQSBlockNumTmp = CeilDiv(qSeqlenCur, curQSBlockTileTmp);
                    curTotalTaskNumTmp += curQNBlockNumTmp * curQSBlockNumTmp;
                }

                if (curBatchTmp != cachedBatch) {
                    kvSeqlenCur = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatchTmp));
                    if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                        if constexpr (!PAGED_CACHE_FLAG) {
                            kvSeqlenCur = KvSeqLenOf(curBatchTmp, gActualKvseqlen, gSeqUsedKv);
                        }
                    }
                    // Per batch KV length including the appended new KV, reused by every task
                    kvSeqlenCur += (kvNewSeqlen != 0U ? kvNewSeqlen : 0U);
                    uint32_t curQNBlockTileCur = GetQNBlockTile(qSeqlenCur, groupSize);
                    uint32_t qNBlockNumPerGroupCur = CeilDiv(groupSize, curQNBlockTileCur);
                    curQNBlockNumCur = qNBlockNumPerGroupCur * kvHeads;
                    curQSBlockNumCur = CeilDiv(qSeqlenCur, 128);
                    cachedBatch = curBatchTmp;
                }

                uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNumTmp;

                uint32_t qSBlockIdxCur = taskIdxCurBatch / curQNBlockNumCur;
                uint32_t qNBlockIdxCur = taskIdxCurBatch - qSBlockIdxCur * curQNBlockNumCur;

                uint32_t pipelineDrain = 0;
                if (taskIdx + uint32_t(coreNum) >= totalTaskNum) {
                    pipelineDrain = PRE_LAUNCH;
                }

                const uint32_t currentTaskStateSlot = taskStateSequence % STACK_SLOTS;
                const bool hasTaskWork =
                    runMainLoop(coreIdx, curBatchTmp, qSeqlenCur, kvSeqlenCur, qNBlockIdxCur, qSBlockIdxCur, false, 0,
                                0, 0, 0, globalTensors, descriptors, taskContexts, currentTaskStateSlot,
                                issuedStackCount, pipelineDrain, softmaxPingPongFlag);
                taskStateSequence += static_cast<uint32_t>(hasTaskWork);
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
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL || MASK_TYPE == FaiKenel::MaskType::MASK_SWA) {
            epilogueOnlineSoftmax.ReleaseArithMaskEvents();
        }
#endif
        AscendC::PipeBarrier<PIPE_ALL>();

        if constexpr (FLASH_DECODE_FLAG) {
            AscendC::SyncAll();
#ifdef __DAV_C220_VEC__
            CombineScale combineScale;
            combineScale.init(resource);
            combineScale(qHeads, fATilingData->totalSplitNodeNum, embedV, fATilingData->splitInfo, gLseFD, gOFD, gO,
                         gActualQseqlen, INPUT_LAYOUT == FaiKenel::inputLayout::TND, maxQSeqlen,
                         (LSE_MODE == Epilogue::LseModeT::OUT_ONLY), gLse, totalQTokens);
#endif
        }
    }

    __aicore__ inline bool runMainLoop(uint32_t coreIdx, uint32_t BIdx, uint32_t qSeqlen, uint32_t kvSeqlenCur,
                                       uint32_t qNBlockIdx, uint32_t qSBlockIdx, bool isSplitKV, int32_t stS2IdxNow,
                                       int32_t enS2IdxNow, uint64_t gmOffsetLseFD, uint64_t gmOffsetOFD,
                                       GlobalTensorBundle& globalTensors, StackDescriptor (&descriptors)[STACK_SLOTS],
                                       DelayedPVTaskContext (&taskContexts)[STACK_SLOTS],
                                       const uint32_t& currentTaskStateSlot, uint32_t& issuedStackCount,
                                       uint32_t pipelineDrain, uint32_t& softmaxPingPongFlag)
    {
        auto& gQ = globalTensors.gQ;
        auto& gK = globalTensors.gK;
        auto& gV = globalTensors.gV;
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

        const bool appendKVFlag = (kvNewSeqlen != 0U);
        // Per batch KV length from the caller, already includes the appended new KV
        uint32_t kvSeqlen = kvSeqlenCur;
        uint32_t kvSeqlenOld = appendKVFlag ? (kvSeqlenCur - kvNewSeqlen) : kvSeqlenCur;
        uint32_t prevQSeqlenSum = 0;
        uint32_t prevKvSeqlenSum = 0;

        if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
            prevQSeqlenSum = static_cast<uint32_t>(gActualQseqlen.GetValue(BIdx));
            if constexpr (!PAGED_CACHE_FLAG) {
                prevKvSeqlenSum =
                    appendKVFlag ? BIdx * kvCacheSeqlen : static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx));
            }
        } else {
            // BSND: Q/O/LSE per-batch storage step is maxQSeqlen.
            prevQSeqlenSum = BIdx * maxQSeqlen;
            if constexpr (!PAGED_CACHE_FLAG) {
                if (appendKVFlag) {
                    // Capacity-aligned cache: cache batch occupies [BIdx * kvCacheSeqlen, ...).
                    prevKvSeqlenSum = BIdx * kvCacheSeqlen;
                } else {
                    // BSND K seqlens are not variable, just multiply by the batch index to get the offset.
                    prevKvSeqlenSum = static_cast<uint32_t>(gActualKvseqlen.GetValue(0)) * BIdx;
                }
            }
        }

        uint64_t qBOffset;
        uint64_t oBOffset;
        if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::BNSD) {
            // BNSD: batch b starts at head 0 of token 0, i.e. (b*qHeads)*S_alloc*D.
            // prevQSeqlenSum == b*maxQSeqlen, so the plain prevQSeqlenSum*strideQ
            // form (BSND/TND) would be missing the qHeads factor here. O is not
            // affected: it is written as contiguous BSND and uses the formula below.
            qBOffset = static_cast<uint64_t>(BIdx) * qHeads * qHeadStride;
        } else {
            qBOffset = static_cast<uint64_t>(prevQSeqlenSum) * strideQ;
        }
        // O is always contiguous BSND: one batch is prevQSeqlenSum rows of strideO.
        oBOffset = static_cast<uint64_t>(prevQSeqlenSum) * strideO;
        uint64_t kBOffset = 0;
        uint64_t vBOffset = 0;
        uint64_t blockBOffset = 0;
        if constexpr (!PAGED_CACHE_FLAG) {
            if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::BNSD) {
                kBOffset = static_cast<uint64_t>(BIdx) * kvHeads * kvHeadStrideK;
                vBOffset = static_cast<uint64_t>(BIdx) * kvHeads * kvHeadStrideV;
            } else {
                kBOffset = static_cast<uint64_t>(prevKvSeqlenSum) * strideK;
                vBOffset = static_cast<uint64_t>(prevKvSeqlenSum) * strideV;
            }
        } else {
            blockBOffset = static_cast<uint64_t>(BIdx) * static_cast<uint64_t>(maxNumBlocksPerBatch);
        }
        // LSE output is head-major (BNS). The batch base and the per-head stride
        // depend on layout:
        //   TND  -> {num_heads, q.size(0)}: batch folded into global token,
        //           head stride = totalQTokens (may exceed cu_seqlens_q[-1])
        //   BSND -> {batch, num_heads, seqlen_q}: batch base = prevQSeqlenSum*qHeads, head stride = maxQSeqlen
        //   BNSD -> same (B, num_heads, seqlen_q) storage as BSND.
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
        uint32_t curKSBlockNum = CeilDiv(kvSeqlen, kvStackLen);
        if (appendKVFlag) {
            curKSBlockNum = CeilDiv(kvSeqlenOld, kvStackLen) + CeilDiv(kvNewSeqlen, kvStackLen);
        }

        uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
        uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;
        uint32_t qNStartIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
        uint32_t lseTokenOffset = qSBlockIdx * curQSBlockTile;

        uint64_t gmOffsetQ = qBOffset + static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideQ +
                             static_cast<uint64_t>(qNStartIdx) * qHeadStride;
        uint64_t gmOffsetK = kBOffset + static_cast<uint64_t>(kvNIdx) * kvHeadStrideK;
        uint64_t gmOffsetV = vBOffset + static_cast<uint64_t>(kvNIdx) * kvHeadStrideV;
        // newkv are (b, s_new, h_k, d); batch b starts at b * kvNewSeqlen tokens.
        uint64_t gmOffsetKNew =
            appendKVFlag ? static_cast<uint64_t>(BIdx) * kvNewSeqlen * strideK + static_cast<uint64_t>(kvNIdx * embed)
                         : 0;
        uint64_t gmOffsetVNew =
            appendKVFlag ? static_cast<uint64_t>(BIdx) * kvNewSeqlen * strideV + static_cast<uint64_t>(kvNIdx * embedV)
                         : 0;
        uint64_t gmOffsetO = oBOffset + static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideO +
                             static_cast<uint64_t>(qNStartIdx) * oHeadStride;
        // head-major: base + head*headStride + qToken
        uint64_t gmOffsetLse =
            lseBOffset + static_cast<uint64_t>(qNStartIdx) * lseHeadStride + static_cast<uint64_t>(lseTokenOffset);

        uint32_t qSBlockSize =
            (qSBlockIdx == (curQSBlockNum - 1U)) ? (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;
        uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1U))
                                   ? (groupSize - qNBlockIdxCurGroup * curQNBlockTile)
                                   : curQNBlockTile;
        uint32_t rowNum = qSBlockSize * qNBlockSize;

        uint32_t noSkipKvS = kvSeqlen;
        uint32_t kvSLoopNumTotal = 0;
        uint32_t kvStart = 0;
        int64_t windowSizeLeftStartLen = 0;
        int64_t windowSizeLeftEndLen = 0;
        int64_t windowSizeRightStartLen = 0;
        int64_t windowSizeRightEndLen = 0;
        bool notPreMask = true;
        bool notNextMask = true;
        int32_t invalidSuffixStartRow = 0;     // 0 disables suffix clearing.
        int32_t invalidPrefixEndRow = qSeqlen; // qSeqlen disables prefix clearing.
        if (maskType == 1U) {
            int64_t diffS = static_cast<int64_t>(kvSeqlen) - static_cast<int64_t>(qSeqlen);
            int64_t causalKvEnd = static_cast<int64_t>((qSBlockIdx + 1U) * curQSBlockTile) + diffS;
            causalKvEnd = causalKvEnd < 0 ? 0 : causalKvEnd;
            noSkipKvS = AscendC::Std::min(causalKvEnd, static_cast<int64_t>(kvSeqlen));
            kvSLoopNumTotal = CeilDiv(noSkipKvS, kvStackLen);
            invalidPrefixEndRow = qSeqlen > kvSeqlen ? static_cast<int32_t>(qSeqlen - kvSeqlen) : invalidPrefixEndRow;
        } else if (maskType == 2U) {
            // Apply the API's bottom-right window to the actual sequence.
            // Do arithmetic in signed 64-bit before clipping or converting to
            // block indices; declared ragged bounds can be much larger than Q/K.
            const int64_t qLength = static_cast<int64_t>(qSeqlen);
            const int64_t kvLength = static_cast<int64_t>(kvSeqlen);
            const int64_t seqDiff = kvLength - qLength;
            const int64_t qBlockBase = static_cast<int64_t>(qSBlockIdx) * curQSBlockTile;
            notPreMask = windowSizeLeft < 0 || windowSizeLeft >= kvLength - 1;
            notNextMask = windowSizeRight < 0 || windowSizeRight >= qLength - 1;
            if (!notPreMask) {
                const int64_t leftOrigin = seqDiff - windowSizeLeft;
                windowSizeLeftStartLen = qBlockBase + leftOrigin;
                windowSizeLeftEndLen = windowSizeLeftStartLen + qSBlockSize;
                const int64_t firstKey = AscendC::Std::min(kvLength,
                    AscendC::Std::max(int64_t{0}, windowSizeLeftStartLen));
                kvStart = static_cast<uint32_t>(firstKey) / kvStackLen;
                if (windowSizeLeftEndLen > kvLength) {
                    invalidSuffixStartRow = static_cast<int32_t>(AscendC::Std::min(qLength,
                        AscendC::Std::max(int64_t{0}, kvLength - leftOrigin)));
                }
            }
            if (!notNextMask) {
                const int64_t rightOrigin = seqDiff + windowSizeRight;
                windowSizeRightStartLen = qBlockBase + rightOrigin;
                // Exclusive upper bound: the inclusive end of the last row + 1.
                windowSizeRightEndLen = windowSizeRightStartLen + qSBlockSize;
                const int64_t endKey = AscendC::Std::min(kvLength,
                    AscendC::Std::max(int64_t{0}, windowSizeRightEndLen));
                noSkipKvS = static_cast<uint32_t>(endKey);
                if (windowSizeRightStartLen < 0) {
                    invalidPrefixEndRow = static_cast<int32_t>(AscendC::Std::min(qLength,
                        AscendC::Std::max(int64_t{0}, -rightOrigin)));
                }
            }
            kvSLoopNumTotal = CeilDiv(noSkipKvS, kvStackLen);
        } else {
            noSkipKvS = kvSeqlen;
            kvSLoopNumTotal = CeilDiv(noSkipKvS, kvStackLen);
        }

        // Match FA GPU split-KV: n_block_min/max = split ∩ window (do not overwrite SWA).
        uint32_t winKvStart = kvStart;
        uint32_t winKvEnd = kvSLoopNumTotal;
        uint32_t kvEnd = winKvEnd;

        // Append-KV phase split: [0, kvStartOld) reads the cache,
        // [kvStartOld, kvEnd) reads newkv and writes it back to the cache
        // (writeback must run even for fully-masked tiles).
        uint32_t kvStartOld = 0;
        uint32_t kvLoopNumTotalNew = 0;
        uint32_t kvSLoopNumTotalOld = kvSLoopNumTotal;
        if (appendKVFlag) {
            kvSLoopNumTotalOld =
                CeilDiv(AscendC::Std::max((int64_t)0, AscendC::Std::min(static_cast<int64_t>(noSkipKvS),
                                                                        static_cast<int64_t>(kvSeqlenOld))),
                        (int64_t)kvStackLen);
            // Phase indices count the old blocks actually issued after causal
            // clipping. The logical new-KV origin remains kvSeqlenOld below.
            // Using ceil(oldLength / tile) here skips new blocks when the
            // writeback owner's Q tile has a fully masked old-KV prefix.
            kvStartOld = kvSLoopNumTotalOld;
            kvLoopNumTotalNew = CeilDiv(kvNewSeqlen, kvStackLen);
            kvSLoopNumTotal = kvSLoopNumTotalOld + kvLoopNumTotalNew;
            kvEnd = kvSLoopNumTotal;
            // winKvEnd must be the per-phase total (v3 captures it after
            // the append update) so the FD fdEnd fallback covers old+new.
            winKvEnd = kvSLoopNumTotal;
        }

        // FD cap runs AFTER the append update (v3 order): the coreInfo S2
        // slice is final and must not be overridden by append's kvEnd.
        if constexpr (FLASH_DECODE_FLAG) {
            uint32_t fdStart = static_cast<uint32_t>(stS2IdxNow);
            uint32_t fdEnd =
                (enS2IdxNow == static_cast<int32_t>(curKSBlockNum)) ? winKvEnd : static_cast<uint32_t>(enS2IdxNow);
            kvStart = AscendC::Std::max(fdStart, winKvStart);
            kvEnd = AscendC::Std::min(fdEnd, winKvEnd);
        }

        int32_t stackSeqCount = 0;
        uint32_t preKVNum = PRE_LAUNCH;
        uint32_t blockStackNum = (kvStackLen - 1 + pagedBlockSize) / pagedBlockSize;
        uint32_t stackSeqTile = kvStackLen;
        uint32_t stackSeqTilePad = kvStackLen;
        uint32_t taskStackCount = 0;
        const bool isEmptyTask = kvStart >= kvEnd;
        if (__builtin_expect(isEmptyTask, false)) {
#ifdef __DAV_C220_VEC__
            if (!isSplitKV) {
                // layoutOInit.shape(1) is the O token stride and also the source of
                // the init epilogue's embedV (shape(1) / qHeads): the contiguous
                // BSND output of every input layout is qHeads*embedV wide.
                LayoutO layoutOInit(qSeqlen, qHeads * embedV);
                LayoutLse layoutLseInit(qHeads, lseHeadStride);
                EpilogueInitOut epilogueInitOut(resource);
                epilogueInitOut(gO[gmOffsetO], gLse[gmOffsetLse], layoutOInit, layoutLseInit, qSBlockSize, qNBlockSize,
                                static_cast<uint32_t>(strideO), static_cast<uint32_t>(oHeadStride));
            }
#endif
            if (pipelineDrain == 0) {
                return false;
            }
            // A final empty task still drains the preceding tasks' delayed PV.
            kvStart = kvEnd;
        }

#ifdef __DAV_C220_CUBE__
        LayoutQ layoutQTemp(rowNum, embed);
        LayoutK layoutKTemp(strideK, stackSeqTile);
        LayoutV layoutVTemp(stackSeqTile, strideV);
        if (!isEmptyTask) {
            blockMmadQK.resetBlockStart(kvStart, pagedBlockSize);
            blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, static_cast<uint32_t>(qHeadStride),
                                static_cast<uint32_t>(strideQ));
        }
#endif
#ifdef __DAV_C220_VEC__
        typename EpilogueOnlineSoftmax::TaskPlan softmaxTaskPlan{};
        typename EpilogueRescaleO::TaskPlan rescaleTaskPlan{};
#endif
        if (!isEmptyTask) {
#ifdef __DAV_C220_VEC__
            softmaxTaskPlan = EpilogueOnlineSoftmax::MakeTaskPlan(rowNum, qSBlockSize, qNBlockSize);
            rescaleTaskPlan = EpilogueRescaleO::MakeTaskPlan(rowNum, qSBlockSize, qNBlockSize, embedRoundV);
#endif
            taskContexts[currentTaskStateSlot] = DelayedPVTaskContext{gmOffsetV,
                                                                      gmOffsetVNew,
                                                                      gmOffsetO,
                                                                      gmOffsetLse,
                                                                      gmOffsetLseFD,
                                                                      gmOffsetOFD,
                                                                      blockBOffset,
                                                                      rowNum,
                                                                      qSeqlen,
                                                                      qSBlockSize,
                                                                      qNBlockSize,
                                                                      kvSLoopNumTotal,
                                                                      noSkipKvS,
                                                                      invalidSuffixStartRow,
                                                                      invalidPrefixEndRow,
                                                                      static_cast<int32_t>(qSBlockIdx),
                                                                      static_cast<int32_t>(curQNBlockTile),
                                                                      isSplitKV
#ifdef __DAV_C220_VEC__
                                                                      ,
                                                                      rescaleTaskPlan
#endif
            };
        }
        for (uint32_t kvSIdx = kvStart; kvSIdx < kvEnd + pipelineDrain; ++kvSIdx) {
            if (kvSIdx < kvEnd) {
                StackDescriptor& desc = descriptors[issuedStackCount % STACK_SLOTS];
                const bool isAppendBlock = appendKVFlag && (kvSIdx >= kvStartOld);
                const bool doCopyback = isAppendBlock && (qSBlockIdx == (BIdx % curQSBlockNum));
                const uint32_t kvSIdxLocal = isAppendBlock ? kvSIdx - kvStartOld : kvSIdx;
                auto gQKSrc = isAppendBlock ? gKNew : gK;
                const uint64_t qkSrcOffset = isAppendBlock ? gmOffsetKNew : gmOffsetK;
                const uint64_t qkCacheOffset =
                    doCopyback ? static_cast<uint64_t>(kvSeqlenOld) + kvSIdxLocal * kvStackLen : 0;
                const uint32_t curStackTileMod = issuedStackCount % STACK_SLOTS;
                // Bound split tails by the fixed QK/PV workspace capacity.
                desc.stackSeqTile = AscendC::Std::min(
                    kvStackLen, GetStackSeqTile(isAppendBlock, kvSIdx, kvSIdxLocal, kvLoopNumTotalNew,
                                                kvSLoopNumTotalOld, noSkipKvS, kvSeqlenOld, kvNewSeqlen, kvStackLen));
                // Delayed PV retains this stack's source, cache destination and writeback owner.
                desc.pvCacheOffset = qkCacheOffset;
                desc.isAppendBlock = isAppendBlock;
                desc.doCopyback = doCopyback;
                desc.kvSIdx = kvSIdxLocal;
                desc.slot = curStackTileMod;
                desc.taskStateSlot = currentTaskStateSlot;
                desc.firstStack = taskStackCount == 0;
                desc.lastStack = kvSIdx + 1 >= kvEnd;
                uint64_t gmOffsetS = static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * STACK_SLOTS +
                                                           curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                GemmCoord actualBlockShapeQK{rowNum, desc.stackSeqTile, embed};
                LayoutS layOutS(rowNum, desc.stackSeqTile, stackSeqTilePad);
#ifdef __DAV_C220_CUBE__
                // Newkv blocks read the contiguous k_new tensor and write back
                // into the cache (page-addressed when paged); old blocks read
                // the cache via the page table.
                if constexpr (PAGED_CACHE_FLAG) {
                    blockMmadQK(gQ[gmOffsetQ], gQKSrc[qkSrcOffset], gS[gmOffsetS],
                                isAppendBlock ? gBlockTable : gBlockTable[blockBOffset], layoutQTemp, layoutKTemp,
                                layOutS, actualBlockShapeQK, kvSIdxLocal, kvSLoopNumTotal, pagedBlockSize, strideK,
                                doCopyback, gK[gmOffsetK], qkCacheOffset, gBlockTable,
                                isAppendBlock ? pagedBlockSize : 0U, static_cast<uint32_t>(blockBOffset));
                } else {
                    blockMmadQK(gQ[gmOffsetQ], gQKSrc[qkSrcOffset], gS[gmOffsetS], gBlockTable, layoutQTemp,
                                layoutKTemp, layOutS, actualBlockShapeQK, kvSIdxLocal, kvSLoopNumTotal, pagedBlockSize,
                                strideK, doCopyback, gK[gmOffsetK], qkCacheOffset);
                }
                if (kvSIdx == kvEnd - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_C220_VEC__
                LayoutP layOutP(rowNum, desc.stackSeqTile, stackSeqTilePad);
                uint64_t gmOffsetP = gmOffsetS;
                uint32_t kvSStartIdx = isAppendBlock ? kvSeqlenOld + kvSIdxLocal * kvStackLen : kvSIdx * kvStackLen;
                uint32_t kvSEndIdx = kvSStartIdx + desc.stackSeqTile;
                if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL) {
                    int64_t triUp = static_cast<int64_t>(noSkipKvS) - static_cast<int64_t>(qSBlockSize);
                    bool doTriUMask = triUp < static_cast<int64_t>(kvSEndIdx - 1U);
                    int64_t diagS0 = triUp - static_cast<int64_t>(kvSStartIdx);
                    epilogueOnlineSoftmax(gP[gmOffsetP], gS[gmOffsetS], layOutP, layOutS, actualBlockShapeQK,
                                          softmaxTaskPlan, desc.firstStack, qSBlockSize, curStackTileMod,
                                          desc.taskStateSlot, qkReady, softmaxPingPongFlag, diagS0, doTriUMask,
                                          isSplitKV);
                } else if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_SWA) {
                    // StartLen can be negative when Sq>>Sk; compare in int64 so
                    // it is not promoted against uint32 kvSStartIdx/kvSEndIdx.
                    int64_t kvStartI = static_cast<int64_t>(kvSStartIdx);
                    int64_t kvEndI = static_cast<int64_t>(kvSEndIdx);
                    bool doTriUPreMask =
                        (maskType != 2 || notPreMask)
                            ? false
                            : (windowSizeLeftStartLen >= kvStartI && windowSizeLeftStartLen < kvEndI) ||
                                  (windowSizeLeftEndLen > kvStartI && windowSizeLeftEndLen <= kvEndI) ||
                                  (windowSizeLeftStartLen <= kvStartI && windowSizeLeftEndLen >= kvEndI);
                    bool doTriUNextMask =
                        (maskType != 2 || notNextMask)
                            ? false
                            : (windowSizeRightStartLen >= kvStartI && windowSizeRightStartLen < kvEndI) ||
                                  (windowSizeRightEndLen > kvStartI && windowSizeRightEndLen <= kvEndI) ||
                                  (windowSizeRightStartLen <= kvStartI && windowSizeRightEndLen >= kvEndI);
                    bool doTriUMask = doTriUPreMask || doTriUNextMask;
                    int64_t leftS0 = static_cast<int64_t>(windowSizeLeftStartLen) - kvSStartIdx;
                    int64_t rightS0 = static_cast<int64_t>(windowSizeRightStartLen) - kvSStartIdx;
                    epilogueOnlineSoftmax(gP[gmOffsetP], gS[gmOffsetS], layOutP, layOutS, actualBlockShapeQK,
                                          softmaxTaskPlan, desc.firstStack, qSBlockSize, curStackTileMod,
                                          desc.taskStateSlot, qkReady, softmaxPingPongFlag, rightS0, doTriUMask,
                                          isSplitKV, leftS0, doTriUPreMask, doTriUNextMask);
                } else {
                    if constexpr (FLASH_DECODE_FLAG) {
                        epilogueOnlineSoftmax(gP[gmOffsetP], gS[gmOffsetS], layOutP, layOutS, actualBlockShapeQK,
                                              softmaxTaskPlan, desc.firstStack, 0, qSBlockSize, qNBlockSize,
                                              curStackTileMod, desc.taskStateSlot, qkReady, softmaxPingPongFlag,
                                              isSplitKV, false, false);
                    } else {
                        epilogueOnlineSoftmax(gP[gmOffsetP], gS[gmOffsetS], layOutP, layOutS, actualBlockShapeQK,
                                              softmaxTaskPlan, desc.firstStack, 0, qSBlockSize, qNBlockSize,
                                              curStackTileMod, desc.taskStateSlot, qkReady, softmaxPingPongFlag, false,
                                              false, false);
                    }
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
            }
            if (issuedStackCount >= PRE_LAUNCH) {
                StackDescriptor& pvDesc = descriptors[(issuedStackCount - PRE_LAUNCH) % STACK_SLOTS];
                DelayedPVTaskContext& task = taskContexts[pvDesc.taskStateSlot];
                uint64_t gmOffsetOTmp = static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * STACK_SLOTS +
                                                              pvDesc.slot * WORKSPACE_BLOCK_SIZE_DB);
                GemmCoord actualBlockShapePV{task.rowNum, embedV, pvDesc.stackSeqTile};
                LayoutOTmp layoutOTmp(task.rowNum, embedV, embedRoundV);
#ifdef __DAV_C220_CUBE__
                auto gPVSrc = pvDesc.isAppendBlock ? gVNew : gV;
                uint64_t gmOffsetVSrc = pvDesc.isAppendBlock ? task.gmOffsetVNew : task.gmOffsetV;
                uint32_t kvSLoopNumTotal = task.kvSLoopNumTotal;
                LayoutP layoutPTemp(task.rowNum, pvDesc.stackSeqTile, stackSeqTilePad);
                uint64_t gmOffsetP =
                    coreIdx * WORKSPACE_BLOCK_SIZE_DB * STACK_SLOTS + pvDesc.slot * WORKSPACE_BLOCK_SIZE_DB;
                if constexpr (PAGED_CACHE_FLAG) {
                    // Delayed PV derives its page cursor from its own stack, not the current QK task.
                    blockMmadPV.resetBlockStart(pvDesc.kvSIdx, pagedBlockSize);
                    blockMmadPV(gP[gmOffsetP], gPVSrc[gmOffsetVSrc], gOTmp[gmOffsetOTmp],
                                pvDesc.isAppendBlock ? gBlockTable : gBlockTable[task.blockBOffset], layoutPTemp,
                                layoutVTemp, layoutOTmp, actualBlockShapePV, pvDesc.kvSIdx, kvSLoopNumTotal,
                                pagedBlockSize, task.noSkipKvS, strideV, blockStackNum, softmaxReady, pvDesc.doCopyback,
                                gV[task.gmOffsetV], pvDesc.pvCacheOffset, gBlockTable,
                                pvDesc.isAppendBlock ? pagedBlockSize : 0U, static_cast<uint32_t>(task.blockBOffset));
                } else {
                    blockMmadPV(gP[gmOffsetP], gPVSrc[gmOffsetVSrc], gOTmp[gmOffsetOTmp], gBlockTable, layoutPTemp,
                                layoutVTemp, layoutOTmp, actualBlockShapePV, pvDesc.kvSIdx, kvSLoopNumTotal,
                                pagedBlockSize, task.noSkipKvS, strideV, blockStackNum, softmaxReady, pvDesc.doCopyback,
                                gV[task.gmOffsetV], pvDesc.pvCacheOffset);
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_C220_VEC__
                LayoutO layoutO(task.qSeqlen, qHeads * embedV);
                LayoutUpdate layoutUpdate(task.rowNum, embed, embedRound);
                // Head-major LSE: (num_heads, headStride). stride(0)=headStride is
                // maxQSeqlen (BSND) or totalQTokens (TND) — the correct per-layout head
                // stride, so rescale_o.hpp's gLse[qNIdx * layoutLse.stride(0)] lands right.
                LayoutLse layoutLse(qHeads, (INPUT_LAYOUT == FaiKenel::inputLayout::TND) ? totalQTokens : maxQSeqlen);
                uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * WORKSPACE_BLOCK_SIZE_DB);

                if constexpr (FLASH_DECODE_FLAG) {
                    LayoutLse layoutgmLse(task.qSBlockSize, task.qNBlockSize);
                    LayoutLse layoutgmLo(task.qSBlockSize, embed * task.qNBlockSize);
                    typename EpilogueRescaleO::SplitKVParams splitParams;
                    splitParams.isSplitkv = task.isSplitKV;
                    splitParams.gCombineLse = gLseFD[task.gmOffsetLseFD];
                    splitParams.gCombineo = gOFD[task.gmOffsetOFD];
                    splitParams.layoutgmLse = &layoutgmLse;
                    splitParams.layoutgmLo = &layoutgmLo;

                    epilogueRescaleO(gO[task.gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate],
                                     gLse[task.gmOffsetLse], layoutO, layoutOTmp, layoutUpdate, layoutLse,
                                     actualBlockShapePV, task.rescaleTaskPlan, pvReady, task.qSBlockSize,
                                     task.qNBlockSize, pvDesc.firstStack, pvDesc.lastStack, pvDesc.slot,
                                     pvDesc.taskStateSlot, splitParams, 0, 0, 0, 0, 1, static_cast<uint32_t>(strideO),
                                     static_cast<uint32_t>(oHeadStride));
                } else {
                    epilogueRescaleO(
                        gO[task.gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gLse[task.gmOffsetLse],
                        layoutO, layoutOTmp, layoutUpdate, layoutLse, actualBlockShapePV, task.rescaleTaskPlan, pvReady,
                        task.qSBlockSize, task.qNBlockSize, pvDesc.firstStack, pvDesc.lastStack, pvDesc.slot,
                        pvDesc.taskStateSlot, typename EpilogueRescaleO::SplitKVParams(), task.invalidSuffixStartRow,
                        task.invalidPrefixEndRow, task.qSeqlen, task.qSBlockIdx, task.curQNBlockTile,
                        static_cast<uint32_t>(strideO), static_cast<uint32_t>(oHeadStride));
                }
#endif
            }
            ++stackSeqCount;
            ++issuedStackCount;
            ++taskStackCount;
        }
        return !isEmptyTask;
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
    int64_t windowSizeLeft;
    int64_t windowSizeRight;
    float scaleValue;
    float softcapValue;
    uint32_t totalQTokens;
    uint32_t maxQSeqlen;
    uint32_t maxKvSeqlen;
    uint32_t kvStackLen = MAX_KV_STACK_LEN;
    // Append-KV: per-batch new length (0 = disabled) and cache capacity.
    uint32_t kvNewSeqlen;
    uint32_t kvCacheSeqlen;
    // BNSD only: allocated KV sequence length (KV head-plane pitch in tokens).
    int64_t kvSeqlenAllocated;

    // Q/K/V row (token) strides and per-head plane pitches. BSND/TND use the packed
    // forms (head stride = embed(V), token stride = H*embed(V)); BNSD swaps them
    // (token stride = embed(V), head stride = S_alloc*embed(V)). O is always
    // contiguous BSND (strideO = qHeads*embedV, oHeadStride = embedV).
    uint64_t strideQ;
    uint64_t strideO;
    uint64_t strideK;
    uint64_t strideV;
    uint64_t qHeadStride;
    uint64_t oHeadStride;
    uint64_t kvHeadStrideK;
    uint64_t kvHeadStrideV;
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
} // namespace SplitFuse

namespace SplitFuse {
template <typename InputDtypeQ = half, typename InputDtypeKv = half, typename IntermCalcPrec = float,
          bool PagedCacheFlag = false, bool FlashDecodeFlag = false,
          FaiKenel::MaskType maskCategory = FaiKenel::MaskType::NO_MASK,
          FaiKenel::inputLayout inLayout = FaiKenel::inputLayout::TND,
          Epilogue::LseModeT lseMode = Epilogue::LseModeT::NONE, bool HAS_SOFTCAP = false>
__global__ __aicore__ void
FAInfer(uint64_t fftsAddr, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o, GM_ADDR lse,
        GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR workspace, FAInferTilingParams tiling, GM_ADDR kNew,
        GM_ADDR vNew, GM_ADDR seqUsedQ = nullptr, GM_ADDR seqUsedKv = nullptr, int64_t kvSeqlenAllocated = 0)
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
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmaxT<lseMode, IntermCalcPrec, HAS_SOFTCAP>;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType>;

    using L1TileShapePV = GemmShape<128, 128, 256>;
    using L0TileShapePV = GemmShape<128, 128, 128>;
    using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPVT<PagedCacheFlag, false>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTmpType>;

    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleOT<lseMode, IntermCalcPrec>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;

    using SplitInfoType = splitNode[25];
    using CombineScale = Epilogue::Block::CombineScale<OType, LseType, SplitInfoType>;
    using FAInferKernelType = FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO,
                                            PagedCacheFlag, FlashDecodeFlag, maskCategory, inLayout, CombineScale>;

    FAIKernelParams params{q,   k,         v,      mask, blockTables, actualQseqlen, actualKvseqlen, o,
                           lse, workspace, tiling, kNew, vNew,        seqUsedQ,      seqUsedKv,      kvSeqlenAllocated};
    FAInferKernelType flashAttnInfer;
    flashAttnInfer(params);
}
} // namespace SplitFuse
