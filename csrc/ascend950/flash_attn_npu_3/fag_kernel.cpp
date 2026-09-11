/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend950 FlashAttention v3 backward device-kernel entrypoint.
 */

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/gemm/block/block_mmad.hpp"

#include "tla/tensor.hpp"
#include "tla/layout.hpp"

#include "fag_common.h"
#include "fag_epilogue_post.hpp"
#include "fag_epilogue_pre.hpp"
#include "fag_block.h"
#include "fag_mmad_sdp.hpp"
#include "fag_mmad_dqkv.hpp"
#include "fag_epilogue_scaled_mask_softmax.hpp"
#include "fag_epilogue_softmax_grad_front.hpp"
#include "fag_epilogue_sub_mul.hpp"
#include "kernel_operator.h"

static constexpr uint32_t TASK_PINGPONG = 2;
static constexpr uint8_t CROSS_CORE_SYNC_MODE = 4;
static constexpr uint16_t V0_V1_FLAG_ID_OFFSET = 16;
static constexpr uint16_t SYNC_C1_TO_V1_FLAG[TASK_PINGPONG] = {0, 1};
static constexpr uint16_t SYNC_C2_TO_V2_FLAG[TASK_PINGPONG] = {2, 3};
static constexpr uint16_t SYNC_V2_TO_C34_FLAG = 4;
static constexpr uint16_t SYNC_V1_TO_C5_FLAG = 5;
static constexpr uint16_t SYNC_C34_TO_V2_FLAG = 8;
static constexpr uint16_t SYNC_C5_TO_V1_FLAG = 9;

/*
 * FAG arch35 task pipeline and L1-buffer ownership:
 *
 *   C1: mm(Q, K^T)   -- K^T resident; Q kept in Q[taskId%2] for C34
 *                       prefetch V^T (group head) overlapping S cube
 *       -- C1_TO_V1[taskId % 2] -->
 *   V1: scaledMaskSoftmax
 *       -- V1_TO_C5 --> C5: mm(P^T, dY)  -- reuse L1 DY[taskId%2] from C2;
 *                                         dV accumulates in L0C until group end
 *                                         D<=128: prefetch RowMajor K into RES_KT
 *                                         upper half (overlaps MTE2 with cube)
 *       <-- C5_TO_V1 -- return the P L1 buffer; release DY
 *
 *   C2: mm(dY, V^T)  -- V^T resident; dY kept in DY[taskId%2] for C5
 *       -- C2_TO_V2[taskId % 2] -->
 *   V2: dS
 *       -- V2_TO_C34 --> C3: mm(dS, K) / C4: mm(dS^T, Q)
 *                       -- D<=128: reuse resident K + L1 Q (no K GM load)
 *                       -- D>128: K-slice via DY scratch; reuse L1 Q
 *       <-- C34_TO_V2 -- return the dS L1 buffer
 *
 * Scheduling: each AIC owns whole (batch, n2, s2Block) groups so contiguous
 * s1 ranges never bounce across cores (keeps KV L1 residency valid).
 *
 * In steady state, C1/C2 of task i overlap V1/V2/C5/C3/C4 of task i - 1.
 */

template <
    typename DataType,
    class BlockMmadSdP_,
    class BlockMmaddQKV_,
    class EpilogueScaledMaskSoftmax_,
    class EpilogueSubMul_,
    FAGTiling950::Layout INPUT_LAYOUT,
    bool IS_ATTEN_MASK,
    bool IS_DTM,
    bool IS_SOFTCAP
>
class FlashAttentionScoreGrad950 {
public:
    using BlockMmadSdP = BlockMmadSdP_;
    using BlockMmaddQKV = BlockMmaddQKV_;
    using EpilogueScaledMaskSoftmax =
        EpilogueScaledMaskSoftmax_;
    using EpilogueSubMul = EpilogueSubMul_;

    // Methods
    CATLASS_DEVICE
    FlashAttentionScoreGrad950() {}

    CATLASS_DEVICE
    ~FlashAttentionScoreGrad950() {}

    CATLASS_DEVICE
    void Init(FAGKernelParams const &params)
    {
        tiling_ = reinterpret_cast<const __gm__ TilingData *>(params.tiling);

        doutGm_.SetGlobalBuffer((__gm__ DataType *)params.dout);
        qGm_.SetGlobalBuffer((__gm__ DataType *)params.q);
        kGm_.SetGlobalBuffer((__gm__ DataType *)params.k);
        vGm_.SetGlobalBuffer((__gm__ DataType *)params.v);
        attenMaskGm_.SetGlobalBuffer((__gm__ uint8_t *)params.attenMask);
        softmaxLseGm_.SetGlobalBuffer((__gm__ float *)params.softmaxLse);
        cuSeqQGm_.SetGlobalBuffer((__gm__ int32_t *)params.cuSeqQlen);
        cuSeqKvGm_.SetGlobalBuffer((__gm__ int32_t *)params.cuSeqKvlen);

        dqWorkspaceGm_.SetGlobalBuffer(
            (__gm__ float *)(params.workspace + tiling_->dqOffset));
        dkWorkspaceGm_.SetGlobalBuffer(
            (__gm__ float *)(params.workspace + tiling_->dkOffset));
        dvWorkspaceGm_.SetGlobalBuffer(
            (__gm__ float *)(params.workspace + tiling_->dvOffset));

        batchNum_ = static_cast<uint32_t>(tiling_->batch);
        qSeqlen_ = static_cast<uint32_t>(tiling_->qSeqlen);
        kvSeqlen_ = static_cast<uint32_t>(tiling_->kvSeqlen);
        qHeadNum_ = tiling_->qHeadNum;
        kvHeadNum_ = tiling_->kvHeadNum;
        groupNum_ = static_cast<uint32_t>(tiling_->groupSize);
        qkHeadDim_ = tiling_->qkHeadDim;
        vHeadDim_ = tiling_->vHeadDim;
        qBlockSize_ = tiling_->qTile;
        kvBlockSize_ = tiling_->kvTile;
        coreNum_ = tiling_->usedCoreNum;
        continuousBlockNum_ = tiling_->continuousBlockNum;
        waveSize_ =
            static_cast<uint64_t>(coreNum_) * continuousBlockNum_;
        scaleValue_ = tiling_->scaleValue;
        softcapValue_ = tiling_->softcapValue;
        softcapInputScale_ = IS_SOFTCAP
            ? scaleValue_ / softcapValue_
            : scaleValue_;

        // L1 layout:
        //   [P ping][P pong][dS ping][dS pong] | cube working via Mm12L1Offset
        //   cube: [RES_KT|RES_K half][RES_VT][Q0][Q1][DY0][DY1]
        //   Q/DY ping-pong by taskId%2; D<=128 packs RowMajor K above K^T.
        const uint32_t l1TileBytes =
            qBlockSize_ * kvBlockSize_ * sizeof(DataType);
        const uint32_t l1PBaseOffset = 0;
        const uint32_t l1dSBaseOffset =
            TASK_PINGPONG * l1TileBytes;
        for (uint32_t i = 0; i < TASK_PINGPONG; ++i) {
            l1PTensor[i] =
                resource.l1Buf.template GetBufferByByte<DataType>(
                    l1PBaseOffset + i * l1TileBytes);
            l1dSTensor[i] =
                resource.l1Buf.template GetBufferByByte<DataType>(
                    l1dSBaseOffset + i * l1TileBytes);
        }

        // Per-AIV UB layout.  Ping and pong each own one complete half of UB:
        //   ping: [mm1Res][mm2Res][attenMask][LSE][P(NZ, M+1 padding)][dS(NZ, M+1 padding)][delta]
        //   pong: [mm1Res][mm2Res][attenMask][LSE][P(NZ, M+1 padding)][dS(NZ, M+1 padding)][delta]
        // Fixpipe SPLIT_M sends half of the logical M rows to each AIV.
        const uint32_t mAligned = RoundUp(qBlockSize_, 16);
        const uint32_t rowsPerSubBlock = mAligned / 2;
        const uint32_t mmResTileBytes =
            rowsPerSubBlock * kvBlockSize_ * sizeof(float);
        const uint32_t attenMaskTileBytes =
            rowsPerSubBlock * kvBlockSize_ * sizeof(uint8_t);
        const uint32_t lseTileBytes =
            RoundUp(rowsPerSubBlock, 8) * 8U * sizeof(float);
        const uint32_t pTileBytes =
            (rowsPerSubBlock + 1) * kvBlockSize_ * sizeof(DataType);
        const uint32_t dSTileBytes = pTileBytes;
        const uint32_t deltaTileBytes =
            rowsPerSubBlock * 8U * sizeof(float);

        const uint32_t mm1SlotOffset = 0;
        const uint32_t mm2SlotOffset = mm1SlotOffset + mmResTileBytes;
        const uint32_t attenMaskSlotOffset = mm2SlotOffset + mmResTileBytes;
        const uint32_t lseSlotOffset = RoundUp(attenMaskSlotOffset + attenMaskTileBytes, 32);
        const uint32_t pSlotOffset = RoundUp(lseSlotOffset + lseTileBytes, 32);
        const uint32_t dSSlotOffset = RoundUp(pSlotOffset + pTileBytes, 32);
        const uint32_t deltaSlotOffset = RoundUp(dSSlotOffset + dSTileBytes, 32);
        const uint32_t pingPongHalfBytes = RoundUp(deltaSlotOffset + deltaTileBytes, 32);

        for (uint32_t i = 0; i < TASK_PINGPONG; ++i) {
            const uint32_t halfBaseOffset = i * pingPongHalfBytes;
            ubMm1ResTensor[i] =
                resource.ubBuf.template GetBufferByByte<float>(
                    halfBaseOffset + mm1SlotOffset);
            ubMm2ResTensor[i] =
                resource.ubBuf.template GetBufferByByte<float>(
                    halfBaseOffset + mm2SlotOffset);
            attenMaskUbTensor[i] =
                resource.ubBuf.template GetBufferByByte<uint8_t>(
                    halfBaseOffset + attenMaskSlotOffset);
            lseUbTensor[i] =
                resource.ubBuf.template GetBufferByByte<float>(
                    halfBaseOffset + lseSlotOffset);
            ubPTensor[i] =
                resource.ubBuf.template GetBufferByByte<DataType>(
                    halfBaseOffset + pSlotOffset);
            ubDSTensor[i] =
                resource.ubBuf.template GetBufferByByte<DataType>(
                    halfBaseOffset + dSSlotOffset);
            deltaUbTensor[i] =
                resource.ubBuf.template GetBufferByByte<float>(
                    halfBaseOffset + deltaSlotOffset);
        }

        if (batchNum_ != 0 && qBlockSize_ != 0 && kvBlockSize_ != 0) {
            LoadDecoderBatch();
        }
        InitEvents();
    }

    CATLASS_DEVICE
    void operator()(FAGKernelParams const &params)
    {
        Init(params);
        uint32_t coreIdx = 0;
        uint32_t subBlockIdx = 0;
#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
#endif
#ifdef __DAV_VEC__
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t vectorBlockIdx = AscendC::GetBlockIdx();
        coreIdx = vectorBlockIdx / subBlockNum;
        subBlockIdx = vectorBlockIdx % subBlockNum;
        epilogueScaledMaskSoftmax_.Init(
            resource,
            params.tiling);
        epilogueSubMul_.Init(
            resource,
            params.workspace,
            params.tiling);
        epiloguePre_.Init(resource, params.workspace, params.tiling);
        epilogueSoftmaxGradFront_.Init(
            resource, params.dout, params.out, params.workspace,
            params.tiling);
        epiloguePost_.Init(
            resource, params.dq, params.dk, params.dv, params.workspace,
            params.tiling);
        const uint32_t vectorCoreNum = coreNum_ * subBlockNum;
        epiloguePre_(
            vectorBlockIdx, vectorCoreNum, vWaitMte3Ping);
        epilogueSoftmaxGradFront_(
            vectorBlockIdx, vectorCoreNum, mte3WaitMte2Ping,
            mte3WaitMte2Pong, vWaitMte2Ping, vWaitMte2Pong,
            vWaitMte3Ping, vWaitMte3Pong);
#endif
        AscendC::SyncAll<false>();
        RunTasks(coreIdx, subBlockIdx);
        AscendC::SyncAll<false>();
#ifdef __DAV_VEC__
        epiloguePost_(
            vectorBlockIdx, vectorCoreNum, mte3WaitMte2Ping,
            mte3WaitMte2Pong, vWaitMte2Ping, vWaitMte2Pong,
            vWaitMte3Ping, vWaitMte3Pong);
#endif
    }

private:
    using TilingData = FAGTiling950::FAGTilingData;

    CATLASS_DEVICE
    void InitEvents()
    {
        vWaitMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>());
        vWaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>());
        vWaitMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>());
        vWaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>());
        mte3WaitMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
        mte3WaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
        lseMte2WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>());
        lseMte2WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>());
        deltaMte2WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>());
        deltaMte2WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>());
        mte3WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>());
        mte3WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>());
        pVWaitMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_V>());
        pVWaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_V>());
        dSVWaitMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_V>());
        dSVWaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_V>());
    }

    CATLASS_DEVICE
    void GetBatchShape(
        uint32_t batchIdx,
        uint64_t &qBatchStart,
        uint64_t &kvBatchStart,
        uint32_t &s1Length,
        uint32_t &s2Length)
    {
        if constexpr (INPUT_LAYOUT == FAGTiling950::Layout::TND) {
            const uint64_t qEnd =
                static_cast<uint64_t>(cuSeqQGm_.GetValue(batchIdx));
            const uint64_t kvEnd =
                static_cast<uint64_t>(cuSeqKvGm_.GetValue(batchIdx));
            qBatchStart =
                batchIdx == 0
                    ? 0
                    : static_cast<uint64_t>(
                          cuSeqQGm_.GetValue(batchIdx - 1));
            kvBatchStart =
                batchIdx == 0
                    ? 0
                    : static_cast<uint64_t>(
                          cuSeqKvGm_.GetValue(batchIdx - 1));
            s1Length = static_cast<uint32_t>(qEnd - qBatchStart);
            s2Length = static_cast<uint32_t>(kvEnd - kvBatchStart);
        } else {
            s1Length = qSeqlen_;
            s2Length = kvSeqlen_;
            qBatchStart = static_cast<uint64_t>(batchIdx) * s1Length;
            kvBatchStart = static_cast<uint64_t>(batchIdx) * s2Length;
        }
    }

    CATLASS_DEVICE
    uint32_t FirstValidS1Block(
        uint32_t s1Length,
        uint32_t s2Length,
        uint32_t s2BlockIdx)
    {
        if constexpr (!IS_ATTEN_MASK) {
            return 0;
        }

        // floor((s2BlockIdx * kvBlockSize - (s2Length - s1Length)) /
        //       qBlockSize), clamped to [0, ceil(s1Length / qBlockSize)].
        const int64_t numerator =
            static_cast<int64_t>(s2BlockIdx) * kvBlockSize_ -
            (static_cast<int64_t>(s2Length) - s1Length);
        if (numerator <= 0) {
            return 0;
        }
        const uint32_t s1BlockNum =
            static_cast<uint32_t>(CeilDiv(s1Length, qBlockSize_));
        return Min(
            static_cast<uint32_t>(numerator / qBlockSize_), s1BlockNum);
    }

    CATLASS_DEVICE
    uint64_t CountValidS1Blocks(
        uint32_t s1Length,
        uint32_t s2Length,
        uint32_t s2BlockNum)
    {
        const uint32_t s1BlockNum =
            static_cast<uint32_t>(CeilDiv(s1Length, qBlockSize_));
        uint64_t count = 0;
        for (uint32_t s2BlockIdx = 0; s2BlockIdx < s2BlockNum;
             ++s2BlockIdx) {
            count += s1BlockNum -
                FirstValidS1Block(
                    s1Length, s2Length, s2BlockIdx);
        }
        return count;
    }

    CATLASS_DEVICE
    void LoadDecoderBatch()
    {
        GetBatchShape(
            decoderBatchIdx_,
            decoderQBatchStart_, decoderKvBatchStart_,
            decoderS1Length_, decoderS2Length_);
        decoderS1BlockNum_ = static_cast<uint32_t>(
            CeilDiv(decoderS1Length_, qBlockSize_));
        decoderS2BlockNum_ = static_cast<uint32_t>(
            CeilDiv(decoderS2Length_, kvBlockSize_));
        decoderValidS1PerN2_ = CountValidS1Blocks(
            decoderS1Length_, decoderS2Length_, decoderS2BlockNum_);
        decoderBatchBlockNum_ =
            kvHeadNum_ * groupNum_ *
            decoderValidS1PerN2_;
        decoderN2Idx_ = 0;
        decoderS2BlockIdx_ = 0;
        decoderS2BlockBegin_ = 0;
    }

    CATLASS_DEVICE
    void FillBlockInfo(
        uint64_t blockId,
        uint32_t n2Idx,
        uint32_t groupIdx,
        uint32_t s1BlockIdx,
        FAGBlockInfo &block)
    {
        const uint32_t s1Start = s1BlockIdx * qBlockSize_;
        const uint32_t s2Start = decoderS2BlockIdx_ * kvBlockSize_;
        const uint64_t totalS1Start =
            decoderQBatchStart_ + s1Start;
        const uint64_t totalS2Start =
            decoderKvBatchStart_ + s2Start;
        const uint64_t qHeadIdx =
            static_cast<uint64_t>(n2Idx) * groupNum_ + groupIdx;

        block.blockId = blockId;
        block.batchIdx = decoderBatchIdx_;
        block.n2Idx = n2Idx;
        block.groupIdx = groupIdx;
        block.s1BlockIdx = s1BlockIdx;
        block.s2BlockIdx = decoderS2BlockIdx_;
        block.s1Start = s1Start;
        block.s2Start = s2Start;
        block.curBatchS1 = decoderS1Length_;
        block.curBatchS2 = decoderS2Length_;
        block.s1Extend =
            Min(qBlockSize_, decoderS1Length_ - s1Start);
        block.s2Extend =
            Min(kvBlockSize_, decoderS2Length_ - s2Start);
        block.totalS1Start = totalS1Start;
        block.totalS2Start = totalS2Start;
        block.qOffset =
            (totalS1Start * qHeadNum_ + qHeadIdx) * qkHeadDim_;
        block.kOffset =
            (totalS2Start * kvHeadNum_ + n2Idx) * qkHeadDim_;
        block.vOffset =
            (totalS2Start * kvHeadNum_ + n2Idx) * vHeadDim_;
        block.doutOffset =
            (totalS1Start * qHeadNum_ + qHeadIdx) * vHeadDim_;
        // TODO: M方向划分和MM1输出一致
        block.firstHalfRealS1 = CeilDiv(block.s1Extend, 2U);
    }

    CATLASS_DEVICE
    bool DecodeBlock(
        uint64_t blockId,
        FAGBlockInfo &block)
    {
        // ------------bIdx------------
        while (decoderBatchIdx_ < batchNum_ &&
               blockId >= decoderBatchBlockBegin_ + decoderBatchBlockNum_) {
            decoderBatchBlockBegin_ += decoderBatchBlockNum_;
            ++decoderBatchIdx_;
            if (decoderBatchIdx_ < batchNum_) {
                LoadDecoderBatch();
            }
        }
        if (decoderBatchIdx_ >= batchNum_) {
            return false;
        }
        // ------------n2Idx------------
        const uint64_t blockNumPerN2 =
            static_cast<uint64_t>(groupNum_) * decoderValidS1PerN2_;
        const uint64_t blockInBatch =
            blockId - decoderBatchBlockBegin_;
        const uint32_t n2Idx =
            static_cast<uint32_t>(blockInBatch / blockNumPerN2);
        const uint64_t blockInN2 = blockInBatch % blockNumPerN2;
        if (n2Idx != decoderN2Idx_) {
            decoderN2Idx_ = n2Idx;
            decoderS2BlockIdx_ = 0;
            decoderS2BlockBegin_ = 0;
        }
        // ------------s2Idx------------
        uint32_t firstValidS1 = 0;
        uint32_t validS1Num = 0;
        while (decoderS2BlockIdx_ < decoderS2BlockNum_) {
            firstValidS1 = FirstValidS1Block(
                decoderS1Length_, decoderS2Length_,
                decoderS2BlockIdx_);
            validS1Num = decoderS1BlockNum_ - firstValidS1;
            const uint64_t s2BlockTasks =
                static_cast<uint64_t>(groupNum_) * validS1Num;
            if (blockInN2 <
                decoderS2BlockBegin_ + s2BlockTasks) {
                break;
            }
            decoderS2BlockBegin_ += s2BlockTasks;
            ++decoderS2BlockIdx_;
        }
        if (decoderS2BlockIdx_ >= decoderS2BlockNum_) {
            return false;
        }
        // ------------gIdx & s1Idx------------
        const uint64_t blockInS2 =
            blockInN2 - decoderS2BlockBegin_;
        const uint32_t groupIdx =
            static_cast<uint32_t>(blockInS2 / validS1Num);
        const uint32_t s1BlockIdx =
            firstValidS1 +
            static_cast<uint32_t>(blockInS2 % validS1Num);
        FillBlockInfo(
            blockId, n2Idx, groupIdx, s1BlockIdx, block);
        return true;
    }

    CATLASS_DEVICE
    void RunTasks(
        uint32_t coreIdx,
        uint32_t subBlockIdx)
    {
        if (coreIdx >= coreNum_ || coreNum_ == 0 ||
            qBlockSize_ == 0 || kvBlockSize_ == 0) {
            return;
        }
#ifdef __DAV_CUBE__
        BlockMmadSdP mm12(resource, Mm12L1Offset());
        BlockMmaddQKV mm345(resource, Mm12L1Offset());
        SetCubeEvents();
#endif
#ifdef __DAV_VEC__
        SetVecEvents();
#endif
        // Assign each (batch, n2, s2Block) group to exactly one AIC so that
        // contiguous s1 ranges stay on the same core (KV L1 residency + local
        // dk/dv accumulate). Round-robin over kvGroupId.
        uint64_t taskId = 0;
        uint64_t kvGroupId = 0;
        bool hasPrev = false;
        bool ktResident = false;
        bool vtResident = false;

        for (uint32_t batchIdx = 0; batchIdx < batchNum_; ++batchIdx) {
            uint64_t qBatchStart = 0;
            uint64_t kvBatchStart = 0;
            uint32_t s1Length = 0;
            uint32_t s2Length = 0;
            GetBatchShape(batchIdx, qBatchStart, kvBatchStart, s1Length, s2Length);
            const uint32_t s1BlockNum =
                static_cast<uint32_t>(CeilDiv(s1Length, qBlockSize_));
            const uint32_t s2BlockNum =
                static_cast<uint32_t>(CeilDiv(s2Length, kvBlockSize_));

            for (uint32_t n2Idx = 0; n2Idx < kvHeadNum_; ++n2Idx) {
                for (uint32_t s2BlockIdx = 0; s2BlockIdx < s2BlockNum;
                     ++s2BlockIdx, ++kvGroupId) {
                    if ((kvGroupId % coreNum_) != coreIdx) {
                        continue;
                    }

                    const uint32_t firstValidS1 = FirstValidS1Block(
                        s1Length, s2Length, s2BlockIdx);
                    if (firstValidS1 >= s1BlockNum) {
                        continue;
                    }

                    const uint32_t validS1Num = s1BlockNum - firstValidS1;
                    const uint32_t tasksInGroup = groupNum_ * validS1Num;
                    if (tasksInGroup == 0) {
                        continue;
                    }

                    uint32_t taskInGroup = 0;
                    for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
                        for (uint32_t s1Ord = 0; s1Ord < validS1Num;
                             ++s1Ord, ++taskInGroup) {
                            FAGBlockInfo block{};
                            const uint32_t s1BlockIdx = firstValidS1 + s1Ord;
                            FillBlockInfoFrom(
                                batchIdx, n2Idx, groupIdx, s1BlockIdx, s2BlockIdx,
                                qBatchStart, kvBatchStart, s1Length, s2Length,
                                block);
                            block.taskId = taskId;
                            block.loadKv = (taskInGroup == 0);
                            block.initDkv = (taskInGroup == 0);
                            block.flushDkv = (taskInGroup + 1 == tasksInGroup);

#ifdef __DAV_CUBE__
                            // New s2 group: free previous KT/VT before reload.
                            if (block.loadKv && ktResident) {
                                mm12.ReleaseResidentKT();
                                ktResident = false;
                            }
                            if (block.loadKv && vtResident) {
                                mm12.ReleaseResidentVT();
                                vtResident = false;
                            }
                            ProcessC1Stage(block, mm12);
                            ProcessC2Stage(block, mm12);
                            if (block.loadKv) {
                                ktResident = true;
                                vtResident = true;
                            }
                            if (hasPrev) {
                                ProcessC5Stage(previousBlock_, true, mm345);
                                ProcessC34Stage(previousBlock_, true, mm345);
                            }
#endif
#ifdef __DAV_VEC__
                            if (hasPrev) {
                                ProcessV1Stage(previousBlock_, subBlockIdx);
                                ProcessV2Stage(previousBlock_, subBlockIdx);
                            }
#endif
                            previousBlock_ = block;
                            hasPrev = true;
                            ++taskId;
                        }
                    }
                }
            }
        }

        if (hasPrev) {
#ifdef __DAV_CUBE__
            ProcessC5Stage(previousBlock_, false, mm345);
            ProcessC34Stage(previousBlock_, false, mm345);
            if (ktResident) {
                mm12.ReleaseResidentKT();
            }
            if (vtResident) {
                mm12.ReleaseResidentVT();
            }
#endif
#ifdef __DAV_VEC__
            ProcessV1Stage(previousBlock_, subBlockIdx);
            ProcessV2Stage(previousBlock_, subBlockIdx);
#endif
        }
#ifdef __DAV_CUBE__
        WaitCubeEvents();
#endif
#ifdef __DAV_VEC__
        WaitVecEvents();
#endif
    }

    CATLASS_DEVICE
    void FillBlockInfoFrom(
        uint32_t batchIdx,
        uint32_t n2Idx,
        uint32_t groupIdx,
        uint32_t s1BlockIdx,
        uint32_t s2BlockIdx,
        uint64_t qBatchStart,
        uint64_t kvBatchStart,
        uint32_t s1Length,
        uint32_t s2Length,
        FAGBlockInfo &block)
    {
        const uint32_t s1Start = s1BlockIdx * qBlockSize_;
        const uint32_t s2Start = s2BlockIdx * kvBlockSize_;
        const uint64_t totalS1Start = qBatchStart + s1Start;
        const uint64_t totalS2Start = kvBatchStart + s2Start;
        const uint64_t qHeadIdx =
            static_cast<uint64_t>(n2Idx) * groupNum_ + groupIdx;

        block.batchIdx = batchIdx;
        block.n2Idx = n2Idx;
        block.groupIdx = groupIdx;
        block.s1BlockIdx = s1BlockIdx;
        block.s2BlockIdx = s2BlockIdx;
        block.s1Start = s1Start;
        block.s2Start = s2Start;
        block.curBatchS1 = s1Length;
        block.curBatchS2 = s2Length;
        block.s1Extend = Min(qBlockSize_, s1Length - s1Start);
        block.s2Extend = Min(kvBlockSize_, s2Length - s2Start);
        block.totalS1Start = totalS1Start;
        block.totalS2Start = totalS2Start;
        block.qOffset =
            (totalS1Start * qHeadNum_ + qHeadIdx) * qkHeadDim_;
        block.kOffset =
            (totalS2Start * kvHeadNum_ + n2Idx) * qkHeadDim_;
        block.vOffset =
            (totalS2Start * kvHeadNum_ + n2Idx) * vHeadDim_;
        block.doutOffset =
            (totalS1Start * qHeadNum_ + qHeadIdx) * vHeadDim_;
        block.firstHalfRealS1 = CeilDiv(block.s1Extend, 2U);
    }

#ifdef __DAV_CUBE__
    CATLASS_DEVICE
    void SetCubeEvents()
    {
        // L0A ping/pong: 0/1
        // L0B ping/pong: 2/3.
        for (uint32_t eventId = 0; eventId < 4; ++eventId) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventId);
        }
        // mm12/mm345 L1 slots + RES_K half-slot token (L1_EVENT_COUNT).
        for (uint32_t eventId = 0;
             eventId < Catlass::Gemm::Ascend950FagL1Layout::L1_EVENT_COUNT;
             ++eventId) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventId);
        }
        // One availability token for every physical 64-KB L0C slot.
        for (uint32_t eventId = 0;
             eventId < Catlass::Gemm::Ascend950FagL0CLayout::L0C_SLOT_NUM;
             ++eventId) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventId);
        }
    }

    CATLASS_DEVICE
    void WaitCubeEvents()
    {
        for (uint32_t eventId = 0; eventId < 4; ++eventId) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventId);
        }
        for (uint32_t eventId = 0;
             eventId < Catlass::Gemm::Ascend950FagL1Layout::L1_EVENT_COUNT;
             ++eventId) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventId);
        }
        for (uint32_t eventId = 0;
             eventId < Catlass::Gemm::Ascend950FagL0CLayout::L0C_SLOT_NUM;
             ++eventId) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventId);
        }
    }

    CATLASS_DEVICE
    void ProcessC1Stage(FAGBlockInfo const &block, BlockMmadSdP &mm12)
    {
        const uint32_t slot = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        const uint16_t flagId = SYNC_C1_TO_V1_FLAG[slot];
        auto q = MakeGmTensor(qGm_, block.qOffset, block.s1Extend,
            qkHeadDim_, qHeadNum_ * qkHeadDim_);
        auto k = MakeGmTensor(kGm_, block.kOffset, block.s2Extend,
            qkHeadDim_, kvHeadNum_ * qkHeadDim_);
        auto v = MakeGmTensor(vGm_, block.vOffset, block.s2Extend,
            vHeadDim_, kvHeadNum_ * vHeadDim_);
        auto sLayout = tla::MakeLayout(
            tla::MakeShape(block.s1Extend, block.s2Extend),
            tla::MakeStride(kvBlockSize_, tla::Int<1>{}));
        auto s = tla::MakeTensor(
            ubMm1ResTensor[slot], sLayout, Catlass::Arch::PositionUB{});
        // Prefetch V^T on the group’s first task so MTE2 overlaps S cube.
        mm12.ComputeS(q, k, s, Catlass::GemmCoord(
            block.s1Extend, block.s2Extend, qkHeadDim_),
            block.loadKv, slot, block.loadKv, v, block.s2Extend, vHeadDim_);
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(flagId);
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(
            flagId + V0_V1_FLAG_ID_OFFSET);
    }

    CATLASS_DEVICE
    void ProcessC2Stage(FAGBlockInfo const &block, BlockMmadSdP &mm12)
    {
        const uint32_t slot = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        const uint16_t flagId = SYNC_C2_TO_V2_FLAG[slot];
        auto dy = MakeGmTensor(doutGm_, block.doutOffset, block.s1Extend,
            vHeadDim_, qHeadNum_ * vHeadDim_);
        auto v = MakeGmTensor(vGm_, block.vOffset, block.s2Extend,
            vHeadDim_, kvHeadNum_ * vHeadDim_);
        auto dpLayout = tla::MakeLayout(
            tla::MakeShape(block.s1Extend, block.s2Extend),
            tla::MakeStride(kvBlockSize_, tla::Int<1>{}));
        auto dp = tla::MakeTensor(
            ubMm2ResTensor[slot], dpLayout, Catlass::Arch::PositionUB{});
        // VT already prefetched in C1 when loadKv; only wait on first use.
        mm12.ComputeDP(dy, v, dp, Catlass::GemmCoord(
            block.s1Extend, block.s2Extend, vHeadDim_),
            /*loadV=*/false, /*waitV=*/block.loadKv, slot);
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(flagId);
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(
            flagId + V0_V1_FLAG_ID_OFFSET);
    }

    CATLASS_DEVICE
    void ProcessC5Stage(
        FAGBlockInfo const &block,
        bool returnL1,
        BlockMmaddQKV &mm345)
    {
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
            SYNC_V1_TO_C5_FLAG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
            SYNC_V1_TO_C5_FLAG + V0_V1_FLAG_ID_OFFSET);

        const uint32_t slot = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        auto dv = MakeGmTensor(dvWorkspaceGm_, block.vOffset,
            block.s2Extend, vHeadDim_, kvHeadNum_ * vHeadDim_);

        // D<=128: prefetch RowMajor K into RES_KT upper half so MTE2 overlaps
        // with dv cube; subsequent C34 tasks in the group reuse it.
        if (block.initDkv && qkHeadDim_ <= 128U) {
            auto k = MakeGmTensor(kGm_, block.kOffset, block.s2Extend,
                qkHeadDim_, kvHeadNum_ * qkHeadDim_);
            mm345.LoadResidentK(k, block.s2Extend, qkHeadDim_);
        }

        // Reuse L1 dY from C2. Single AIC owns (b,n2,s2) → no GM atomic on dv.
        mm345.ComputeDv(l1PTensor[slot], dv,
            Catlass::GemmCoord(block.s1Extend, vHeadDim_, block.s2Extend),
            slot, block.initDkv, block.flushDkv, /*enAtomicDv=*/false);

        if (returnL1) {
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
                SYNC_C5_TO_V1_FLAG);
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
                SYNC_C5_TO_V1_FLAG + V0_V1_FLAG_ID_OFFSET);
        }
    }

    CATLASS_DEVICE
    void ProcessC34Stage(
        FAGBlockInfo const &block,
        bool returnL1,
        BlockMmaddQKV &mm345)
    {
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
            SYNC_V2_TO_C34_FLAG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
            SYNC_V2_TO_C34_FLAG + V0_V1_FLAG_ID_OFFSET);

        const uint32_t slot = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        auto k = MakeGmTensor(kGm_, block.kOffset, block.s2Extend,
            qkHeadDim_, kvHeadNum_ * qkHeadDim_);
        auto dq = MakeGmTensor(dqWorkspaceGm_, block.qOffset,
            block.s1Extend, qkHeadDim_, qHeadNum_ * qkHeadDim_);
        auto dk = MakeGmTensor(dkWorkspaceGm_, block.kOffset,
            block.s2Extend, qkHeadDim_, kvHeadNum_ * qkHeadDim_);

        // D<=128: waitResidentK on first task (C5 prefetched); later tasks hit
        // resident K. D>128: ComputeDqDk streams K via DY scratch.
        mm345.ComputeDqDk(
            l1dSTensor[slot], k, dq, dk,
            Catlass::GemmCoord(block.s1Extend, qkHeadDim_, block.s2Extend),
            slot,
            /*waitResidentK=*/block.initDkv,
            /*enAtomicDq=*/true,
            block.initDkv, block.flushDkv, /*enAtomicDk=*/false);

        if (returnL1) {
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
                SYNC_C34_TO_V2_FLAG);
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
                SYNC_C34_TO_V2_FLAG + V0_V1_FLAG_ID_OFFSET);
        }
    }

    template <class GlobalTensor>
    CATLASS_DEVICE
    auto MakeGmTensor(GlobalTensor &gm, uint64_t offset,
        uint32_t rows, uint64_t cols, uint64_t rowStride)
    {
        auto layout = tla::MakeLayout(
            tla::MakeShape(rows, cols),
            tla::MakeStride(rowStride, tla::Int<1>{}));
        return tla::MakeTensor(gm[offset], layout, Catlass::Arch::PositionGM{});
    }

    CATLASS_DEVICE
    uint32_t Mm12L1Offset() const
    {
        // P/dS ping-pong occupies the front of L1; cube working tiles
        // (RES_KT[+K half]/VT + Q0/Q1 + DY0/DY1) start after that.
        return 2U * TASK_PINGPONG * qBlockSize_ * kvBlockSize_ * sizeof(DataType);
    }
#endif

#ifdef __DAV_VEC__
    CATLASS_DEVICE
    void SetVecEvents()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(lseMte2WaitVPing);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(lseMte2WaitVPong);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(deltaMte2WaitVPing);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(deltaMte2WaitVPong);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pVWaitMte3Ping);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pVWaitMte3Pong);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(dSVWaitMte3Ping);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(dSVWaitMte3Pong);
    }

    CATLASS_DEVICE
    void WaitVecEvents()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(lseMte2WaitVPing);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(lseMte2WaitVPong);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(deltaMte2WaitVPing);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(deltaMte2WaitVPong);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pVWaitMte3Ping);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pVWaitMte3Pong);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(dSVWaitMte3Ping);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(dSVWaitMte3Pong);
    }

    CATLASS_DEVICE
    void ProcessV1Stage(
        FAGBlockInfo const &block,
        uint32_t subBlockIdx)
    {
        const uint32_t flagSlot =
            static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(
            SYNC_C1_TO_V1_FLAG[flagSlot]);

        // Before task 1 and later overwrite P in L1, C5 must return ownership
        // of the buffer used by the preceding task.
        if (block.taskId != 0) {
            AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(
                SYNC_C5_TO_V1_FLAG);
        }

        auto attenMaskLayout = tla::MakeLayout<
            uint8_t, Catlass::layout::RowMajor>(256, 256);
        auto attenMaskGmTensor = tla::MakeTensor(
            attenMaskGm_, attenMaskLayout, Catlass::Arch::PositionGM{});

        uint64_t lseRows = 0;
        uint64_t lseColumns = 0;
        uint64_t lseRowOffset = 0;
        uint64_t lseColOffset = 0;
        const uint64_t n1Idx = static_cast<uint64_t>(block.n2Idx) * groupNum_ + block.groupIdx;
        const uint64_t subBlockS1Offset = subBlockIdx * block.firstHalfRealS1;
        const uint64_t subBlockRealS1 = subBlockIdx ?
            block.s1Extend - block.firstHalfRealS1 :
            block.firstHalfRealS1;
        if constexpr (INPUT_LAYOUT == FAGTiling950::Layout::TND) {
            // TND:  [N1, totalS1]
            lseRows = qHeadNum_;
            lseColumns = tiling_->totalQ;
            lseRowOffset = n1Idx;
            lseColOffset = block.totalS1Start + subBlockS1Offset;
        } else {
            // BSND: [B * N1, S1]
            lseRows = static_cast<uint64_t>(batchNum_) * qHeadNum_;
            lseColumns = qSeqlen_;
            lseRowOffset = static_cast<uint64_t>(block.batchIdx) * qHeadNum_ + n1Idx;
            lseColOffset = block.s1Start + subBlockS1Offset;
        }
        auto softmaxLseLayout = tla::MakeLayout<
            float, Catlass::layout::RowMajor>(lseRows, lseColumns);
        auto softmaxLseGmTensor = tla::MakeTensor(
            softmaxLseGm_, softmaxLseLayout,
            tla::MakeCoord(lseRowOffset, lseColOffset),
            Catlass::Arch::PositionGM{});

        event_t mte2ToVEvent = flagSlot ? vWaitMte2Pong : vWaitMte2Ping;
        event_t vToMte2Event = flagSlot ? lseMte2WaitVPong : lseMte2WaitVPing;
        event_t vToMte3Event = flagSlot ? mte3WaitVPong : mte3WaitVPing;
        event_t mte3ToVEvent = flagSlot ? pVWaitMte3Pong : pVWaitMte3Ping;
        if (subBlockRealS1 != 0U) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event);
            epilogueScaledMaskSoftmax_(
                block,
                attenMaskGmTensor,
                softmaxLseGmTensor,
                attenMaskUbTensor[flagSlot],
                lseUbTensor[flagSlot],
                ubMm1ResTensor[flagSlot],
                ubPTensor[flagSlot],
                l1PTensor[flagSlot],
                mte2ToVEvent,
                vToMte3Event,
                mte3ToVEvent,
                softcapInputScale_,
                softcapValue_,
                subBlockRealS1,
                subBlockIdx);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event);
        }

        // Notify C5 only after this vector sub-block has finished writing P.
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(
            SYNC_V1_TO_C5_FLAG);
    }

    CATLASS_DEVICE
    void ProcessV2Stage(
        FAGBlockInfo const &block,
        uint32_t subBlockIdx)
    {
        const uint32_t flagSlot =
            static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(
            SYNC_C2_TO_V2_FLAG[flagSlot]);

        // Before task 1 and later overwrite dS in L1, C3/C4 must return
        // ownership of the buffer used by the preceding task.
        if (block.taskId != 0) {
            AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(
                SYNC_C34_TO_V2_FLAG);
        }
        const uint32_t bufferId = flagSlot;
        event_t mte2ToVEvent = flagSlot ? vWaitMte2Pong : vWaitMte2Ping;
        event_t vToMte2Event = flagSlot ? deltaMte2WaitVPong : deltaMte2WaitVPing;
        event_t vToMte3Event = flagSlot ? mte3WaitVPong : mte3WaitVPing;
        event_t mte3ToVEvent = flagSlot ? dSVWaitMte3Pong : dSVWaitMte3Ping;

        const uint32_t subBlockRealS1 = subBlockIdx ?
            block.s1Extend - block.firstHalfRealS1 :
            block.firstHalfRealS1;
        if (subBlockRealS1 != 0U) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event);
            epilogueSubMul_(
                block,
                ubMm2ResTensor[bufferId],
                ubPTensor[bufferId],
                ubMm1ResTensor[bufferId],
                ubDSTensor[bufferId],
                deltaUbTensor[bufferId],
                l1dSTensor[bufferId],
                mte2ToVEvent,
                vToMte3Event,
                mte3ToVEvent,
                subBlockIdx);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event);
        }

        // Notify C3/C4 only after this vector sub-block has finished writing dS.
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(
            SYNC_V2_TO_C34_FLAG);
    }
#endif

private:
    Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource;
#ifdef __DAV_VEC__
    Catlass::Epilogue::Block::FagPre<
        Catlass::Arch::Ascend950, TilingData> epiloguePre_;
    Catlass::Epilogue::Block::FagSoftmaxGradFront<
        DataType, Catlass::Arch::Ascend950,
        TilingData> epilogueSoftmaxGradFront_;
    EpilogueScaledMaskSoftmax epilogueScaledMaskSoftmax_;
    EpilogueSubMul epilogueSubMul_;
    Catlass::Epilogue::Block::FagPost<
        DataType, Catlass::Arch::Ascend950, TilingData> epiloguePost_;
#endif

    const __gm__ TilingData *tiling_ = nullptr;

    AscendC::GlobalTensor<DataType> doutGm_;
    AscendC::GlobalTensor<DataType> qGm_;
    AscendC::GlobalTensor<DataType> kGm_;
    AscendC::GlobalTensor<DataType> vGm_;
    AscendC::GlobalTensor<uint8_t> attenMaskGm_;
    AscendC::GlobalTensor<float> softmaxLseGm_;
    AscendC::GlobalTensor<int32_t> cuSeqQGm_;
    AscendC::GlobalTensor<int32_t> cuSeqKvGm_;
    AscendC::GlobalTensor<float> dqWorkspaceGm_;
    AscendC::GlobalTensor<float> dkWorkspaceGm_;
    AscendC::GlobalTensor<float> dvWorkspaceGm_;

    AscendC::LocalTensor<DataType> l1PTensor[TASK_PINGPONG];
    AscendC::LocalTensor<DataType> l1dSTensor[TASK_PINGPONG];
    AscendC::LocalTensor<float> ubMm1ResTensor[TASK_PINGPONG];
    AscendC::LocalTensor<float> ubMm2ResTensor[TASK_PINGPONG];
    AscendC::LocalTensor<uint8_t> attenMaskUbTensor[TASK_PINGPONG];
    AscendC::LocalTensor<float> lseUbTensor[TASK_PINGPONG];
    AscendC::LocalTensor<DataType> ubPTensor[TASK_PINGPONG];
    AscendC::LocalTensor<DataType> ubDSTensor[TASK_PINGPONG];
    AscendC::LocalTensor<float> deltaUbTensor[TASK_PINGPONG];

    event_t vWaitMte2Ping;
    event_t vWaitMte2Pong;
    event_t vWaitMte3Ping;
    event_t vWaitMte3Pong;
    event_t mte3WaitMte2Ping;
    event_t mte3WaitMte2Pong;
    event_t lseMte2WaitVPing;
    event_t lseMte2WaitVPong;
    event_t deltaMte2WaitVPing;
    event_t deltaMte2WaitVPong;
    event_t mte3WaitVPing;
    event_t mte3WaitVPong;
    event_t pVWaitMte3Ping;
    event_t pVWaitMte3Pong;
    event_t dSVWaitMte3Ping;
    event_t dSVWaitMte3Pong;

    uint32_t batchNum_ = 0;
    uint32_t qSeqlen_ = 0;
    uint32_t kvSeqlen_ = 0;
    uint64_t qHeadNum_ = 0;
    uint64_t kvHeadNum_ = 0;
    uint32_t groupNum_ = 0;
    uint64_t qkHeadDim_ = 0;
    uint64_t vHeadDim_ = 0;
    uint32_t qBlockSize_ = 0;
    uint32_t kvBlockSize_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t continuousBlockNum_ = 0;
    uint64_t waveSize_ = 0;
    float scaleValue_ = 1.0f;
    float softcapValue_ = 0.0f;
    float softcapInputScale_ = 1.0f;

    uint32_t decoderBatchIdx_ = 0;
    uint64_t decoderBatchBlockBegin_ = 0;
    uint64_t decoderBatchBlockNum_ = 0;
    uint64_t decoderQBatchStart_ = 0;
    uint64_t decoderKvBatchStart_ = 0;
    uint32_t decoderS1Length_ = 0;
    uint32_t decoderS2Length_ = 0;
    uint32_t decoderS1BlockNum_ = 0;
    uint32_t decoderS2BlockNum_ = 0;
    uint64_t decoderValidS1PerN2_ = 0;

    uint32_t decoderN2Idx_ = 0;
    uint32_t decoderS2BlockIdx_ = 0;
    uint64_t decoderS2BlockBegin_ = 0;

    FAGBlockInfo previousBlock_{};
};

template <
    typename DataType,
    FAGTiling950::Layout INPUT_LAYOUT,
    bool IS_CAUSAL,
    bool IS_DETERMINISTIC,
    bool IS_SOFTCAP>
CATLASS_GLOBAL void FlashAttentionV3Bwd950(
    GM_ADDR dout,
    GM_ADDR q,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR out,
    GM_ADDR mask,
    GM_ADDR softmax_lse,
    GM_ADDR cu_seqlens_q,
    GM_ADDR cu_seqlens_k,
    GM_ADDR dq,
    GM_ADDR dk,
    GM_ADDR dv,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    using ArchTag = Catlass::Arch::Ascend950;
    using L1TileShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<256>>;
    using L0TileShape = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;

    using DispatchPolicySdP = Catlass::Gemm::MmadAscend950FagSdP<2, 2, false>;
    using TileCopySdP = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag,
        DataType, Catlass::layout::RowMajor,
        DataType, Catlass::layout::ColumnMajor,
        float, Catlass::layout::RowMajor,
        void, Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadSdP = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicySdP, L1TileShape, L0TileShape,
        DataType, DataType, float, void, TileCopySdP>;

    using DispatchPolicydQKV = Catlass::Gemm::MmadAscend950FagdQKV<2, 2, false>;
    using TileCopydQKV = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag,
        DataType, Catlass::layout::RowMajor,
        DataType, Catlass::layout::RowMajor,
        float, Catlass::layout::RowMajor>;
    using BlockMmaddQKV = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicydQKV, L1TileShape, L0TileShape,
        DataType, DataType, float, void, TileCopydQKV>;

    using DispatchPolicyScaledMaskSoftmax =
        Catlass::Epilogue::EpilogueAscend950FAGScaledMaskSoftmax<
            INPUT_LAYOUT, IS_CAUSAL, IS_SOFTCAP>;
    using EpilogueScaledMaskSoftmax =
        Catlass::Epilogue::Block::BlockEpilogue<
            DispatchPolicyScaledMaskSoftmax,
            DataType,
            float,
            FAGTiling950::FAGTilingData>;
    using DispatchPolicySubMul =
        Catlass::Epilogue::EpilogueAscend950FAGSubMul<
            INPUT_LAYOUT, IS_SOFTCAP>;
    using EpilogueSubMul =
        Catlass::Epilogue::Block::BlockEpilogue<
            DispatchPolicySubMul,
            DataType,
            DataType,
            float,
            FAGTiling950::FAGTilingData>;
    using FAGKernel950 = FlashAttentionScoreGrad950<
        DataType,
        BlockMmadSdP,
        BlockMmaddQKV,
        EpilogueScaledMaskSoftmax,
        EpilogueSubMul,
        INPUT_LAYOUT,
        IS_CAUSAL,
        IS_DETERMINISTIC,
        IS_SOFTCAP>;
    FAGKernelParams params{dout, q, k, v, out, mask, softmax_lse,
        cu_seqlens_q, cu_seqlens_k, dq, dk, dv, workspace, tiling};
    FAGKernel950 fag;
    fag(params);
}
