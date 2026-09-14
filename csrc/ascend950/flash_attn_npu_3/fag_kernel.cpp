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
#include "fag_det_schedule.hpp"
#include "fag_epilogue_post.hpp"
#include "fag_epilogue_pre.hpp"
#include "fag_block.h"
#include "fag_mmad_sdp.hpp"
#include "fag_mmad_dqkv.hpp"
#include "fag_epilogue_scaled_mask_softmax.hpp"
#include "fag_epilogue_softmax_grad_front.hpp"
#include "fag_epilogue_sub_mul.hpp"
#include "fag_epilogue_deterministic_add.hpp"
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
// BN2S2: mode-0 fix barrier shared by every AIC to pin the atomic dq/dk/dv
// accumulation of one schedule round before the next round is issued.
static constexpr uint8_t DETER_FIX_SYNC_MODE = 0;
static constexpr uint16_t SYNC_DQ_ROUND_FLAG = 10;
// BN2S2 per-column dk/dv conversion: AIC signals a completed private column,
// the paired AIVs cast it to the bf16 output and release the buffer.
static constexpr uint16_t SYNC_C34_TO_V5_FLAG = 11;  // dk column ready
static constexpr uint16_t SYNC_C5_TO_V6_FLAG = 12;   // dv column ready
static constexpr uint16_t SYNC_V5_TO_C34_FLAG = 13;  // dk buffer free
static constexpr uint16_t SYNC_V6_TO_C5_FLAG = 14;   // dv buffer free
static constexpr uint32_t DET_CAST_CHUNK_ROWS = 32;

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

        if constexpr (IS_DTM) {
            dqDetWorkspaceGm_.SetGlobalBuffer(
                (__gm__ float *)(params.workspace + tiling_->dqDetOffset));
            dkDetWorkspaceGm_.SetGlobalBuffer(
                (__gm__ float *)(params.workspace + tiling_->dkDetOffset));
            dvDetWorkspaceGm_.SetGlobalBuffer(
                (__gm__ float *)(params.workspace + tiling_->dvDetOffset));
            dkPrivGm_.SetGlobalBuffer(
                (__gm__ float *)(params.workspace + tiling_->dkPrivOffset));
            dvPrivGm_.SetGlobalBuffer(
                (__gm__ float *)(params.workspace + tiling_->dvPrivOffset));
        }
        if constexpr (IS_DTM) {
            // BN2S2 cast outputs / TND length pointers (unused by the
            // non-deterministic instantiation).
            dkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(params.dk));
            dvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(params.dv));
            cuSeqQPtr_ = reinterpret_cast<__gm__ int32_t *>(params.cuSeqQlen);
            cuSeqKvPtr_ = reinterpret_cast<__gm__ int32_t *>(params.cuSeqKvlen);
        }

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
        if constexpr (IS_DTM) {
            dtmVecCoreNum_ =
                tiling_->dqVecNum + tiling_->dkVecNum + tiling_->dvVecNum;
        }
        waveSize_ =
            static_cast<uint64_t>(coreNum_) * continuousBlockNum_;
        scaleValue_ = tiling_->scaleValue;
        softcapValue_ = tiling_->softcapValue;
        softcapInputScale_ = IS_SOFTCAP
            ? scaleValue_ / softcapValue_
            : scaleValue_;

        if constexpr (IS_DTM) {
            // Det slot geometry must match fag_tiling.cpp: one slot is
            // qTile/kvTile rows of RoundUp(headDim, 8) floats.
            qkHeadDimAlign_ = (qkHeadDim_ + 7) / 8 * 8;
            vHeadDimAlign_ = (vHeadDim_ + 7) / 8 * 8;
            dqDetSlotElems_ =
                static_cast<uint64_t>(qBlockSize_) * qkHeadDimAlign_;
            dkDetSlotElems_ =
                static_cast<uint64_t>(kvBlockSize_) * qkHeadDimAlign_;
            dvDetSlotElems_ =
                static_cast<uint64_t>(kvBlockSize_) * vHeadDimAlign_;
            detBn2s2_ =
                tiling_->detSchedule ==
                static_cast<uint32_t>(FAGTiling950::DetSchedule::BN2S2);
            if (detBn2s2_) {
                // opst arch35 BN2S2 schedule: every cube core runs the same
                // round sequence; the schedule itself provides dq/dk/dv
                // determinism, so no per-round reducer or det slot is used.
                detShape_.batch = static_cast<int64_t>(batchNum_);
                detShape_.qSeqLen = static_cast<int64_t>(qSeqlen_);
                detShape_.kvSeqLen = static_cast<int64_t>(kvSeqlen_);
                detShape_.qHeadNum = static_cast<int64_t>(qHeadNum_);
                detShape_.kvHeadNum = static_cast<int64_t>(kvHeadNum_);
                detShape_.groupNum = static_cast<int64_t>(groupNum_);
                detShape_.qTile = static_cast<int64_t>(qBlockSize_);
                detShape_.kvTile = static_cast<int64_t>(kvBlockSize_);
                detShape_.coreNum = static_cast<int64_t>(coreNum_);
                detKind_ = tiling_->detKind;
                detColumnRounds_ = tiling_->detColumnRounds == 0
                    ? 1 : tiling_->detColumnRounds;
                detBufNum_ = tiling_->detBufNum == 0
                    ? 1 : tiling_->detBufNum;
                detPrivDkv_ = tiling_->detPrivDkv != 0;
                dkPrivSlotElems_ =
                    static_cast<uint64_t>(kvBlockSize_) * qkHeadDimAlign_;
                dvPrivSlotElems_ =
                    static_cast<uint64_t>(kvBlockSize_) * vHeadDimAlign_;
                totalRounds_ = static_cast<uint32_t>(tiling_->detMaxRound);
                totalBlockNum_ =
                    static_cast<uint64_t>(totalRounds_) * waveSize_;

                // AIV dk/dv cast buffers live in a dedicated tail region of
                // UB; fag_tiling.cpp checks the budget against ubSize.
                castChunkRows_ = kvBlockSize_ < DET_CAST_CHUNK_ROWS
                    ? kvBlockSize_ : DET_CAST_CHUNK_ROWS;
                const uint64_t inCol =
                    qkHeadDimAlign_ > vHeadDimAlign_
                    ? qkHeadDimAlign_ : vHeadDimAlign_;
                const uint64_t headCol = qkHeadDim_ > vHeadDim_
                    ? qkHeadDim_ : vHeadDim_;
                const uint64_t castInBytes = RoundUp<32>(
                    static_cast<uint64_t>(castChunkRows_) * inCol *
                    sizeof(float));
                const uint64_t castOutBytes = RoundUp<32>(
                    static_cast<uint64_t>(castChunkRows_) * headCol *
                    sizeof(DataType));
                if (tiling_->ubSize > castInBytes + castOutBytes) {
                    castInUb_ =
                        resource.ubBuf.template GetBufferByByte<float>(
                            tiling_->ubSize - castInBytes - castOutBytes);
                    castOutUb_ =
                        resource.ubBuf.template GetBufferByByte<DataType>(
                            tiling_->ubSize - castOutBytes);
                }
            } else {
                // Every core derives the same total task/round count so that
                // the v1 round-end SyncAll barriers are joined uniformly,
                // independent of how many tasks each core actually issued.
                totalBlockNum_ = 0;
                for (uint32_t b = 0; b < batchNum_; ++b) {
                    uint64_t qStart = 0, kvStart = 0;
                    uint32_t s1Len = 0, s2Len = 0;
                    GetBatchShape(b, qStart, kvStart, s1Len, s2Len);
                    const uint32_t s1BlkNum = static_cast<uint32_t>(
                        CeilDiv(s1Len, qBlockSize_));
                    totalBlockNum_ += kvHeadNum_ * groupNum_ *
                        CountValidS2Blocks(s1Len, s2Len, s1BlkNum);
                }
                totalRounds_ = static_cast<uint32_t>(
                    CeilDiv(totalBlockNum_, waveSize_));
            }
        }

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
        epilogueDetAdd_.Init(
            resource, params.dq, params.cuSeqQlen, params.cuSeqKvlen,
            params.workspace, params.tiling);
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
        if constexpr (IS_DTM) {
            // BN2S2 dk/dv cast (AIV): single alternating set shared by dk/dv.
            castMte3ToMte2_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
            castMte2ToV_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>());
            castVToMte3_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>());
        }
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

    // Deterministic axis order (b -> n2 -> s1 -> g -> s2) scans s1 outer and
    // s2 inner, so the per-s1 valid s2-block count is the segment length.
    CATLASS_DEVICE
    uint32_t ValidS2BlockNum(
        uint32_t s1Length,
        uint32_t s2Length,
        uint32_t s1BlockIdx)
    {
        const uint32_t s2BlockNum =
            static_cast<uint32_t>(CeilDiv(s2Length, kvBlockSize_));
        if constexpr (!IS_ATTEN_MASK) {
            return s2BlockNum;
        }

        // Causal: kv col j is visible to q row i iff
        // j <= i + (s2Length - s1Length).  The last visible col of this s1
        // block is (s1BlockIdx + 1) * qBlockSize - 1 + (s2Length - s1Length).
        const int64_t lastCol =
            static_cast<int64_t>(s1BlockIdx + 1) * qBlockSize_ - 1 +
            (static_cast<int64_t>(s2Length) - s1Length);
        if (lastCol < 0) {
            return 0;
        }
        return Min(
            static_cast<uint32_t>(lastCol / kvBlockSize_),
            s2BlockNum - 1) + 1;
    }

    CATLASS_DEVICE
    uint32_t LastValidS2Block(
        uint32_t s1Length,
        uint32_t s2Length,
        uint32_t s1BlockIdx)
    {
        // Mirror of FirstValidS1Block; requires ValidS2BlockNum > 0.
        return ValidS2BlockNum(s1Length, s2Length, s1BlockIdx) - 1;
    }

    CATLASS_DEVICE
    uint64_t CountValidS2Blocks(
        uint32_t s1Length,
        uint32_t s2Length,
        uint32_t s1BlockNum)
    {
        uint64_t count = 0;
        for (uint32_t s1BlockIdx = 0; s1BlockIdx < s1BlockNum;
             ++s1BlockIdx) {
            count += ValidS2BlockNum(s1Length, s2Length, s1BlockIdx);
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
        if constexpr (IS_DTM) {
            // Axis order b -> n2 -> s1 -> g -> s2: per-s1 valid s2 counts.
            decoderValidS2PerN2_ = CountValidS2Blocks(
                decoderS1Length_, decoderS2Length_, decoderS1BlockNum_);
            decoderBatchBlockNum_ =
                kvHeadNum_ * groupNum_ *
                decoderValidS2PerN2_;
            decoderN2Idx_ = 0;
            decoderS1BlockIdx_ = 0;
            decoderS1BlockBegin_ = 0;
        } else {
            decoderValidS1PerN2_ = CountValidS1Blocks(
                decoderS1Length_, decoderS2Length_, decoderS2BlockNum_);
            decoderBatchBlockNum_ =
                kvHeadNum_ * groupNum_ *
                decoderValidS1PerN2_;
            decoderN2Idx_ = 0;
            decoderS2BlockIdx_ = 0;
            decoderS2BlockBegin_ = 0;
        }
    }

    CATLASS_DEVICE
    void FillBlockInfo(
        uint64_t blockId,
        uint32_t n2Idx,
        uint32_t groupIdx,
        uint32_t s1BlockIdx,
        uint32_t s2BlockIdx,
        FAGBlockInfo &block)
    {
        const uint32_t s1Start = s1BlockIdx * qBlockSize_;
        const uint32_t s2Start = s2BlockIdx * kvBlockSize_;
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
        block.s2BlockIdx = s2BlockIdx;
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

    // Fill one task from the opst BN2S2 schedule coordinate.  BSND uses the
    // uniform global shape; TND reads the per-batch cumulative lengths.
    CATLASS_DEVICE
    void FillBlockInfoBn2s2(
        uint64_t blockId,
        const fag_det::Coord &c,
        FAGBlockInfo &block)
    {
        uint32_t s1Len = qSeqlen_;
        uint32_t s2Len = kvSeqlen_;
        uint64_t qBatchStart = 0;
        uint64_t kvBatchStart = 0;
        if constexpr (INPUT_LAYOUT == FAGTiling950::Layout::TND) {
            qBatchStart = c.batch == 0
                ? 0 : static_cast<uint64_t>(cuSeqQPtr_[c.batch - 1]);
            kvBatchStart = c.batch == 0
                ? 0 : static_cast<uint64_t>(cuSeqKvPtr_[c.batch - 1]);
            s1Len = static_cast<uint32_t>(
                static_cast<uint64_t>(cuSeqQPtr_[c.batch]) - qBatchStart);
            s2Len = static_cast<uint32_t>(
                static_cast<uint64_t>(cuSeqKvPtr_[c.batch]) - kvBatchStart);
        } else {
            qBatchStart = static_cast<uint64_t>(c.batch) * qSeqlen_;
            kvBatchStart = static_cast<uint64_t>(c.batch) * kvSeqlen_;
        }

        const uint32_t s1Start = static_cast<uint32_t>(c.s1) * qBlockSize_;
        const uint32_t s2Start = static_cast<uint32_t>(c.s2) * kvBlockSize_;
        const uint64_t totalS1Start = qBatchStart + s1Start;
        const uint64_t totalS2Start = kvBatchStart + s2Start;
        const uint64_t qHeadIdx =
            static_cast<uint64_t>(c.n2) * groupNum_ + c.g;

        block.blockId = blockId;
        block.batchIdx = static_cast<uint32_t>(c.batch);
        block.n2Idx = static_cast<uint32_t>(c.n2);
        block.groupIdx = static_cast<uint32_t>(c.g);
        block.s1BlockIdx = static_cast<uint32_t>(c.s1);
        block.s2BlockIdx = static_cast<uint32_t>(c.s2);
        block.s1Start = s1Start;
        block.s2Start = s2Start;
        block.curBatchS1 = s1Len;
        block.curBatchS2 = s2Len;
        block.s1Extend = Min(qBlockSize_, s1Len - s1Start);
        block.s2Extend = Min(kvBlockSize_, s2Len - s2Start);
        block.totalS1Start = totalS1Start;
        block.totalS2Start = totalS2Start;
        block.qOffset =
            (totalS1Start * qHeadNum_ + qHeadIdx) * qkHeadDim_;
        block.kOffset =
            (totalS2Start * kvHeadNum_ + c.n2) * qkHeadDim_;
        block.vOffset =
            (totalS2Start * kvHeadNum_ + c.n2) * vHeadDim_;
        block.doutOffset =
            (totalS1Start * qHeadNum_ + qHeadIdx) * vHeadDim_;
        block.firstHalfRealS1 = CeilDiv(block.s1Extend, 2U);
        // BN2S2 treats every task as its own residency/accumulate group: the
        // shared C1/C2 stages reload K^T/V^T each task and dk/dv flush per
        // task (the private/shared workspaces handle cross-task accumulation).
        block.loadKv = true;
        block.initDkv = true;
        block.flushDkv = true;

        // Private dk/dv accumulator layout: which fold buffer this task
        // belongs to and the real column each buffer accumulates.
        block.detParity = c.parity;
        block.detBufNum = detBufNum_;
        for (uint32_t p = 0; p < 2; ++p) {
            block.detFoldValid[p] = c.foldValid[p];
            block.detFoldN2[p] = static_cast<uint32_t>(c.foldN2[p]);
            if (c.foldValid[p] == 0) {
                block.detFoldS2Start[p] = 0;
                block.detFoldS2Extend[p] = 0;
                continue;
            }
            const uint32_t foldS2Start =
                static_cast<uint32_t>(c.foldS2[p]) * kvBlockSize_;
            // Packed row start of the fold column's batch: for TND it is the
            // cumulative length (batches may differ), for BSND it is the
            // uniform batch stride.  Fold buffers other than the task's own
            // column (p == 1) only occur for the BSND causal zip.
            uint64_t foldKvStart = 0;
            if constexpr (INPUT_LAYOUT == FAGTiling950::Layout::TND) {
                foldKvStart = c.foldBatch[p] == 0
                    ? 0
                    : static_cast<uint64_t>(
                        cuSeqKvPtr_[c.foldBatch[p] - 1]);
            } else {
                foldKvStart =
                    static_cast<uint64_t>(c.foldBatch[p]) * kvSeqlen_;
            }
            block.detFoldS2Start[p] = foldKvStart + foldS2Start;
            block.detFoldS2Extend[p] =
                Min(kvBlockSize_, s2Len - foldS2Start);
        }
    }

    CATLASS_DEVICE
    bool DecodeBlock(
        uint64_t blockId,
        FAGBlockInfo &block)
    {
        if constexpr (IS_DTM) {
            if (detBn2s2_) {
                // Schedule-address space: blockId = round * coreNum + core.
                const uint64_t core = blockId % static_cast<uint64_t>(coreNum_);
                const uint64_t round = blockId / static_cast<uint64_t>(coreNum_);
                fag_det::Coord c;
                bool ok = false;
                if (detKind_ == static_cast<uint32_t>(fag_det::KIND_TND_DENSE)) {
                    ok = cuSeqQPtr_ != nullptr && cuSeqKvPtr_ != nullptr &&
                        fag_det::CalTNDDenseSwizzleIndex(
                            detShape_, cuSeqQPtr_, cuSeqKvPtr_,
                            tiling_->tndPrefix,
                            static_cast<int64_t>(core) + 1,
                            static_cast<int64_t>(round) + 1, c);
                } else if (detKind_ ==
                           static_cast<uint32_t>(fag_det::KIND_TND_GQA_DENSE)) {
                    ok = cuSeqQPtr_ != nullptr && cuSeqKvPtr_ != nullptr &&
                        fag_det::CalTNDDenseGqaIndex(
                            detShape_, cuSeqQPtr_, cuSeqKvPtr_,
                            tiling_->tndPrefix,
                            static_cast<int64_t>(core) + 1,
                            static_cast<int64_t>(round) + 1,
                            static_cast<int64_t>(tiling_->detMaxRound), c);
                } else if (detKind_ ==
                           static_cast<uint32_t>(fag_det::KIND_TND_CAUSAL)) {
                    ok = cuSeqQPtr_ != nullptr && cuSeqKvPtr_ != nullptr &&
                        fag_det::CalTNDCausalIndex(
                            detShape_, cuSeqQPtr_, cuSeqKvPtr_,
                            tiling_->tndPrefix, tiling_->tndCausalP1,
                            tiling_->tndCausalP2,
                            static_cast<int64_t>(tiling_->detMaxRound),
                            static_cast<int64_t>(core) + 1,
                            static_cast<int64_t>(round) + 1, c);
                } else {
                    ok = fag_det::Decode(
                        static_cast<fag_det::Kind>(detKind_), detShape_,
                        static_cast<int64_t>(round) + 1,
                        static_cast<int64_t>(core) + 1, c);
                }
                if (!ok) {
                    return false;
                }
                FillBlockInfoBn2s2(blockId, c, block);
                // Causal mask is right-down aligned (j <= i + (sk - sq)); a block
                // entirely above the diagonal contributes exact zeros, so skip it
                // and let the dense schedule pay only for the causal region.
                // Semantic no-op (the epilogue would mask the whole block).
                if (tiling_->maskType != 0 && detBufNum_ <= 1 &&
                    detKind_ != static_cast<uint32_t>(fag_det::KIND_LEFT_UP_CAUSAL_SWIZZLE) &&
                    detKind_ != static_cast<uint32_t>(fag_det::KIND_TND_CAUSAL)) {
                    const int64_t diagOffset =
                        static_cast<int64_t>(block.curBatchS2) -
                        static_cast<int64_t>(block.curBatchS1);
                    if (static_cast<int64_t>(block.s2Start) >
                        static_cast<int64_t>(block.s1Start) +
                            static_cast<int64_t>(block.s1Extend) - 1 + diagOffset) {
                        return false;
                    }
                }
                return true;
            }
        }
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
            static_cast<uint64_t>(groupNum_) *
            (IS_DTM ? decoderValidS2PerN2_ : decoderValidS1PerN2_);
        const uint64_t blockInBatch =
            blockId - decoderBatchBlockBegin_;
        const uint32_t n2Idx =
            static_cast<uint32_t>(blockInBatch / blockNumPerN2);
        const uint64_t blockInN2 = blockInBatch % blockNumPerN2;
        if (n2Idx != decoderN2Idx_) {
            decoderN2Idx_ = n2Idx;
            if constexpr (IS_DTM) {
                decoderS1BlockIdx_ = 0;
                decoderS1BlockBegin_ = 0;
            } else {
                decoderS2BlockIdx_ = 0;
                decoderS2BlockBegin_ = 0;
            }
        }
        if constexpr (IS_DTM) {
            // ------------s1Idx------------
            uint32_t validS2Num = 0;
            while (decoderS1BlockIdx_ < decoderS1BlockNum_) {
                validS2Num = ValidS2BlockNum(
                    decoderS1Length_, decoderS2Length_,
                    decoderS1BlockIdx_);
                const uint64_t s1BlockTasks =
                    static_cast<uint64_t>(groupNum_) * validS2Num;
                if (blockInN2 <
                    decoderS1BlockBegin_ + s1BlockTasks) {
                    break;
                }
                decoderS1BlockBegin_ += s1BlockTasks;
                ++decoderS1BlockIdx_;
            }
            if (decoderS1BlockIdx_ >= decoderS1BlockNum_) {
                return false;
            }
            // ------------gIdx & s2Idx------------
            const uint64_t blockInS1 =
                blockInN2 - decoderS1BlockBegin_;
            const uint32_t groupIdx =
                static_cast<uint32_t>(blockInS1 / validS2Num);
            const uint32_t s2BlockIdx =
                static_cast<uint32_t>(blockInS1 % validS2Num);
            FillBlockInfo(blockId, n2Idx, groupIdx, decoderS1BlockIdx_, s2BlockIdx, block);
            return true;
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
        FillBlockInfo(blockId, n2Idx, groupIdx, s1BlockIdx, decoderS2BlockIdx_, block);
        return true;
    }

    // A scheduled column ends when no later round of the same core produces a
    // task on the same column.  Pure decode lookahead: holes inside a column
    // keep the accumulator open, a later column (or the end of the schedule)
    // closes it.  Both cube and vector sides evaluate it identically.
    CATLASS_DEVICE
    bool IsColumnEndBn2s2(uint32_t coreIdx, const FAGBlockInfo &cur)
    {
        if (detBufNum_ > 1) {
            // Causal fold: the lane's two real columns interleave inside one
            // virtual column, so end the accumulation by round arithmetic.
            const uint32_t colRounds = detColumnRounds_ == 0 ? 1 : detColumnRounds_;
            const uint32_t base = cur.issueRound / colRounds;
            for (uint32_t s = cur.issueRound + 1; s < totalRounds_; ++s) {
                FAGBlockInfo next{};
                if (DecodeBlock(
                        static_cast<uint64_t>(s) * coreNum_ + coreIdx, next)) {
                    return (s / colRounds) != base;
                }
            }
            return true;
        }
        // Single-buffer schedules (dense/GQA/TND): the accumulator closes as
        // soon as the next valid task owns a different column.
        for (uint32_t s = cur.issueRound + 1; s < totalRounds_; ++s) {
            FAGBlockInfo next{};
            if (DecodeBlock(
                    static_cast<uint64_t>(s) * coreNum_ + coreIdx, next)) {
                return next.batchIdx != cur.batchIdx ||
                    next.n2Idx != cur.n2Idx ||
                    next.groupIdx != cur.groupIdx ||
                    next.s2BlockIdx != cur.s2BlockIdx;
            }
        }
        return true;
    }

    // ------------------------------------------------------------------
    // BN2S2 deterministic loop: one scheduled task per cube core per round.
    // The pending task's back-end is flushed one iteration after its
    // front-end, exactly like the non-DTM drain pipeline.  Every iteration
    // ends with a mode-0 fix barrier, so the cross-core atomic accumulation
    // of round r is pinned before round r+1 is issued: this is the only
    // ordering primitive the schedule needs.
    // ------------------------------------------------------------------
    CATLASS_DEVICE
    void RunTasksBn2s2(
        uint32_t coreIdx,
        uint32_t subBlockIdx)
    {
#ifdef __DAV_CUBE__
        BlockMmadSdP mm12(resource, Mm12L1Offset());
        BlockMmaddQKV mm345(resource, Mm12L1Offset());
        SetCubeEvents();
#endif
#ifdef __DAV_VEC__
        SetVecEvents();
#endif
        coreIdxBn2s2_ = coreIdx;
        uint64_t taskId = 0;
        bool hasPending = false;
        bool ktResident = false;
        bool vtResident = false;
        FAGBlockInfo pending{};
        for (uint32_t step = 0; step < totalRounds_; ++step) {
            FAGBlockInfo block{};
            const bool valid = DecodeBlock(
                static_cast<uint64_t>(step) * coreNum_ + coreIdx, block);
            if (valid) {
                block.taskId = taskId;
                block.issueRound = step;
                block.issueLane = 0;
            }
#ifdef __DAV_CUBE__
            // Front-end MM of the new task overlaps the back-end below.
            if (valid) {
                // Per-task residency: release the previous K^T/V^T before C1
                // reloads them (event pairing identical to the main loop).
                if (ktResident) {
                    mm12.ReleaseResidentKT();
                    ktResident = false;
                }
                if (vtResident) {
                    mm12.ReleaseResidentVT();
                    vtResident = false;
                }
                ProcessC1Stage(block, mm12);
                ProcessC2Stage(block, mm12);
                ktResident = true;
                vtResident = true;
            }
#endif
            if (hasPending) {
                const bool colEnd =
                    IsColumnEndBn2s2(coreIdx, pending);
#ifdef __DAV_CUBE__
                ProcessC5StageBn2s2(pending, true, mm345, colEnd);
                ProcessC34StageBn2s2(pending, true, mm345, colEnd);
#endif
#ifdef __DAV_VEC__
                ProcessV1Stage(pending, subBlockIdx);
                ProcessV2Stage(pending, subBlockIdx);
                ProcessDkvCastStage(pending, subBlockIdx, colEnd);
#endif
                hasPending = false;
            }
            if (valid) {
                pending = block;
                hasPending = true;
                ++taskId;
            }
#ifdef __DAV_CUBE__
            AscendC::CrossCoreSetFlag<DETER_FIX_SYNC_MODE, PIPE_FIX>(
                SYNC_DQ_ROUND_FLAG);
            AscendC::CrossCoreWaitFlag<DETER_FIX_SYNC_MODE, PIPE_FIX>(
                SYNC_DQ_ROUND_FLAG);
#endif
        }
        if (hasPending) {
#ifdef __DAV_CUBE__
            ProcessC5StageBn2s2(pending, false, mm345, true);
            ProcessC34StageBn2s2(pending, false, mm345, true);
#endif
#ifdef __DAV_VEC__
            ProcessV1Stage(pending, subBlockIdx);
            ProcessV2Stage(pending, subBlockIdx);
            ProcessDkvCastStage(pending, subBlockIdx, true);
#endif
        }
#ifdef __DAV_CUBE__
        // Must run even when this core has no pending task left: the final
        // WaitCubeEvents consumes one token per L1 event, and the last C1's
        // load is only balanced by this release.
        if (ktResident) {
            mm12.ReleaseResidentKT();
        }
        if (vtResident) {
            mm12.ReleaseResidentVT();
        }
        WaitCubeEvents();
#endif
#ifdef __DAV_VEC__
        WaitVecEvents();
#endif
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
        if constexpr (IS_DTM) {
            if (detBn2s2_) {
                RunTasksBn2s2(coreIdx, subBlockIdx);
                return;
            }
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
                    }                }
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

    // BN2S2 back-end.  dq/dk/dv all accumulate atomically into the full fp32
    // workspaces: under the schedule every dk/dv column is owned by a single
    // core (fixed program order), while dq is ordered by the per-round mode-0
    // fix barrier in RunTasksBn2s2.
    // BN2S2 dv: accumulate the task's partial into the core-private buffer of
    // the task's fold parity.  The first contribution of a column overwrites
    // (stale data from the previous column), the rest atomic-add.  At the
    // column end the buffer is published to the paired AIVs for conversion.
    CATLASS_DEVICE
    void ProcessC5StageBn2s2(
        FAGBlockInfo const &block,
        bool returnL1,
        BlockMmaddQKV &mm345,
        bool colEnd)
    {
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
            SYNC_V1_TO_C5_FLAG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
            SYNC_V1_TO_C5_FLAG + V0_V1_FLAG_ID_OFFSET);

        const uint32_t parity = block.detParity & 1U;
        const uint64_t bufIdx =
            static_cast<uint64_t>(coreIdxBn2s2_) * detBufNum_ + parity;
        if (detPrivDkv_) {
            if (!dvAnyOpen_) {
                if (dvColumnStarted_) {
                    // Reuse gate: the previous column's cast must have drained the
                    // private buffers before the first overwrite.
                    AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(
                        SYNC_V6_TO_C5_FLAG);
                    AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(
                        SYNC_V6_TO_C5_FLAG + V0_V1_FLAG_ID_OFFSET);
                }
                dvAnyOpen_ = true;
                dvColumnStarted_ = true;
            }
        }
        const bool firstOfBuf = detPrivDkv_ && !dvBufOpen_[parity];
        dvBufOpen_[parity] = true;

        const uint32_t slot =
            static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        // Prefetch RowMajor K for the following C34 (D<=128).  Per-task group:
        // the previous C34 released the resident slot on its flush.
        if (qkHeadDim_ <= 128U) {
            auto k = MakeGmTensor(kGm_, block.kOffset, block.s2Extend,
                qkHeadDim_, kvHeadNum_ * qkHeadDim_);
            mm345.LoadResidentK(k, block.s2Extend, qkHeadDim_);
        }
        // dY is already in L1 DY[slot] from C2; release it here.
        if (detPrivDkv_) {
            auto dv = MakeGmTensor(dvPrivGm_, bufIdx * dvPrivSlotElems_,
                block.s2Extend, vHeadDim_, vHeadDimAlign_);
            mm345.ComputeDv(l1PTensor[slot], dv,
                Catlass::GemmCoord(block.s1Extend, vHeadDim_, block.s2Extend),
                slot, /*initDv=*/true, /*flushDv=*/true,
                /*enAtomicDv=*/!firstOfBuf);
        } else {
            auto dv = MakeGmTensor(dvWorkspaceGm_, block.vOffset,
                block.s2Extend, vHeadDim_, kvHeadNum_ * vHeadDim_);
            mm345.ComputeDv(l1PTensor[slot], dv,
                Catlass::GemmCoord(block.s1Extend, vHeadDim_, block.s2Extend),
                slot, /*initDv=*/true, /*flushDv=*/true, /*enAtomicDv=*/true);
        }

        if (detPrivDkv_ && colEnd) {
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(
                SYNC_C5_TO_V6_FLAG);
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(
                SYNC_C5_TO_V6_FLAG + V0_V1_FLAG_ID_OFFSET);
            dvAnyOpen_ = false;
            dvBufOpen_[0] = false;
            dvBufOpen_[1] = false;
        }

        if (returnL1) {
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
                SYNC_C5_TO_V1_FLAG);
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
                SYNC_C5_TO_V1_FLAG + V0_V1_FLAG_ID_OFFSET);
        }
    }

    // BN2S2 dq/dk.  dq accumulates atomically in the full fp32 workspace
    // (round-ordered by the mode-0 barrier); dk follows the same private
    // per-column scheme as dv.
    CATLASS_DEVICE
    void ProcessC34StageBn2s2(
        FAGBlockInfo const &block,
        bool returnL1,
        BlockMmaddQKV &mm345,
        bool colEnd)
    {
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
            SYNC_V2_TO_C34_FLAG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
            SYNC_V2_TO_C34_FLAG + V0_V1_FLAG_ID_OFFSET);

        const uint32_t parity = block.detParity & 1U;
        const uint64_t bufIdx =
            static_cast<uint64_t>(coreIdxBn2s2_) * detBufNum_ + parity;
        if (detPrivDkv_) {
            if (!dkAnyOpen_) {
                if (dkColumnStarted_) {
                    AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(
                        SYNC_V5_TO_C34_FLAG);
                    AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(
                        SYNC_V5_TO_C34_FLAG + V0_V1_FLAG_ID_OFFSET);
                }
                dkAnyOpen_ = true;
                dkColumnStarted_ = true;
            }
        }
        const bool firstOfBuf = detPrivDkv_ && !dkBufOpen_[parity];
        dkBufOpen_[parity] = true;

        const uint32_t slot =
            static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        auto k = MakeGmTensor(kGm_, block.kOffset, block.s2Extend,
            qkHeadDim_, kvHeadNum_ * qkHeadDim_);
        auto dq = MakeGmTensor(dqWorkspaceGm_, block.qOffset, block.s1Extend,
            qkHeadDim_, qHeadNum_ * qkHeadDim_);
        // Q is read from L1 Q[slot] (loaded by C1), K from the resident slot
        // (prefetched by C5) for D<=128; D>128 streams K through DY scratch.
        if (detPrivDkv_) {
            auto dk = MakeGmTensor(dkPrivGm_, bufIdx * dkPrivSlotElems_,
                block.s2Extend, qkHeadDim_, qkHeadDimAlign_);
            mm345.ComputeDqDk(
                l1dSTensor[slot], k, dq, dk,
                Catlass::GemmCoord(block.s1Extend, qkHeadDim_, block.s2Extend),
                slot, /*waitResidentK=*/true, /*enAtomicDq=*/true,
                /*initDk=*/true, /*flushDk=*/true, /*enAtomicDk=*/!firstOfBuf);
        } else {
            auto dk = MakeGmTensor(dkWorkspaceGm_, block.kOffset,
                block.s2Extend, qkHeadDim_, kvHeadNum_ * qkHeadDim_);
            mm345.ComputeDqDk(
                l1dSTensor[slot], k, dq, dk,
                Catlass::GemmCoord(block.s1Extend, qkHeadDim_, block.s2Extend),
                slot, /*waitResidentK=*/true, /*enAtomicDq=*/true,
                /*initDk=*/true, /*flushDk=*/true, /*enAtomicDk=*/true);
        }

        if (detPrivDkv_ && colEnd) {
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(
                SYNC_C34_TO_V5_FLAG);
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(
                SYNC_C34_TO_V5_FLAG + V0_V1_FLAG_ID_OFFSET);
            dkAnyOpen_ = false;
            dkBufOpen_[0] = false;
            dkBufOpen_[1] = false;
        }

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
        if constexpr (IS_DTM) {
            if (detBn2s2_) {
                // One-time arm; subsequent Set/Wait strictly alternate
                // across cast chunks and flush points.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(castMte3ToMte2_);
            }
        }
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

    // One chunk of a private dk/dv column: copy `rows` rows of `headDim`
    // floats into UB, optionally scale (dk), cast to the output dtype and
    // store into the strided output rows.  The MTE3_MTE2 token alternates
    // across chunks and flush points (armed once in SetVecEvents).
    CATLASS_DEVICE
    void CastRegionRows(
        uint64_t totalS2Start,
        uint32_t n2Idx,
        uint32_t s2Extend,
        uint32_t subBlockIdx,
        AscendC::GlobalTensor<float> &src,
        uint64_t srcOffset,
        uint64_t srcRowStride,
        uint64_t headDim,
        AscendC::GlobalTensor<DataType> &dst,
        bool applyScale)
    {
        if (s2Extend == 0U) {
            return;
        }
        const uint32_t firstHalf = (s2Extend + 1U) / 2U;
        const uint32_t rowBegin = subBlockIdx * firstHalf;
        const uint32_t rowLimit =
            (rowBegin + firstHalf < s2Extend) ? rowBegin + firstHalf : s2Extend;
        uint32_t row = rowBegin;
        while (row < rowLimit) {
            const uint32_t rows =
                (rowLimit - row < castChunkRows_) ? rowLimit - row : castChunkRows_;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(castMte3ToMte2_);
            AscendC::DataCopyExtParams inParams{
                static_cast<uint16_t>(rows),
                static_cast<uint32_t>(headDim * sizeof(float)),
                static_cast<int64_t>((srcRowStride - headDim) * sizeof(float)),
                0, 0};
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(
                castInUb_,
                src[srcOffset + static_cast<uint64_t>(row) * srcRowStride],
                inParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(castMte2ToV_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(castMte2ToV_);
            const uint32_t elems = rows * static_cast<uint32_t>(headDim);
            if (applyScale) {
                AscendC::Muls(castInUb_, castInUb_, scaleValue_, elems);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Cast(castOutUb_, castInUb_,
                AscendC::RoundMode::CAST_RINT, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(castVToMte3_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(castVToMte3_);
            const uint64_t outOffset =
                (totalS2Start + row) * kvHeadNum_ * headDim + n2Idx * headDim;
            AscendC::DataCopyExtParams outParams{
                static_cast<uint16_t>(rows),
                static_cast<uint32_t>(headDim * sizeof(DataType)),
                0,
                static_cast<int64_t>(
                    (kvHeadNum_ - 1U) * headDim * sizeof(DataType)),
                0};
            AscendC::DataCopyPad(dst[outOffset], castOutUb_, outParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(castMte3ToMte2_);
            row += rows;
        }
    }

    // BN2S2 dk/dv column conversion.  Called after V1/V2 of the pending task
    // on both AIVs: track the fold buffers seen for this column and, at the
    // column end, wait for the cube-side publish, convert each used private
    // buffer and release it for the next column.
    CATLASS_DEVICE
    void ProcessDkvCastStage(
        FAGBlockInfo const &block,
        uint32_t subBlockIdx,
        bool colEnd)
    {
        if (!detPrivDkv_) {
            return;
        }
        const uint32_t parity = block.detParity & 1U;
        dkHasData_[parity] = true;
        dvHasData_[parity] = true;
        if (!colEnd) {
            return;
        }

        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(
            SYNC_C34_TO_V5_FLAG);
        for (uint32_t p = 0; p < detBufNum_; ++p) {
            if (!dkHasData_[p]) {
                continue;
            }
            dkHasData_[p] = false;
            if (block.detFoldValid[p] == 0U) {
                continue;
            }
            CastRegionRows(
                block.detFoldS2Start[p], block.detFoldN2[p],
                block.detFoldS2Extend[p], subBlockIdx, dkPrivGm_,
                (static_cast<uint64_t>(coreIdxBn2s2_) * detBufNum_ + p) *
                    dkPrivSlotElems_,
                qkHeadDimAlign_, qkHeadDim_, dkGm_, true);
        }
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(
            SYNC_V5_TO_C34_FLAG);

        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(
            SYNC_C5_TO_V6_FLAG);
        for (uint32_t p = 0; p < detBufNum_; ++p) {
            if (!dvHasData_[p]) {
                continue;
            }
            dvHasData_[p] = false;
            if (block.detFoldValid[p] == 0U) {
                continue;
            }
            CastRegionRows(
                block.detFoldS2Start[p], block.detFoldN2[p],
                block.detFoldS2Extend[p], subBlockIdx, dvPrivGm_,
                (static_cast<uint64_t>(coreIdxBn2s2_) * detBufNum_ + p) *
                    dvPrivSlotElems_,
                vHeadDimAlign_, vHeadDim_, dvGm_, false);
        }
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(
            SYNC_V6_TO_C5_FLAG);
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
    
    CATLASS_DEVICE
    void ProcessVecDTMStage(uint32_t issueRound)
    {
        epilogueDetAdd_(
            AscendC::GetBlockIdx(), issueRound, totalBlockNum_
        );
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
    Catlass::Epilogue::Block::FagDeterministicAdd<
        DataType, Catlass::Arch::Ascend950, TilingData> epilogueDetAdd_;
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
    __gm__ int32_t *cuSeqQPtr_ = nullptr;
    __gm__ int32_t *cuSeqKvPtr_ = nullptr;
    AscendC::GlobalTensor<float> dqWorkspaceGm_;
    AscendC::GlobalTensor<float> dkWorkspaceGm_;
    AscendC::GlobalTensor<float> dvWorkspaceGm_;
    
    AscendC::GlobalTensor<float> dqDetWorkspaceGm_;
    AscendC::GlobalTensor<float> dkDetWorkspaceGm_;
    AscendC::GlobalTensor<float> dvDetWorkspaceGm_;
    AscendC::GlobalTensor<float> dkPrivGm_;
    AscendC::GlobalTensor<float> dvPrivGm_;
    AscendC::GlobalTensor<DataType> dkGm_;
    AscendC::GlobalTensor<DataType> dvGm_;
    
    // Det slot geometry (fp32 elements) and uniform round count (IS_DTM).
    uint64_t qkHeadDimAlign_ = 0;
    uint64_t vHeadDimAlign_ = 0;
    uint64_t dqDetSlotElems_ = 0;
    uint64_t dkDetSlotElems_ = 0;
    uint64_t dvDetSlotElems_ = 0;
    uint64_t totalBlockNum_ = 0;
    uint32_t totalRounds_ = 0;

    // BN2S2 deterministic schedule (detSchedule == BN2S2).
    bool detBn2s2_ = false;
    fag_det::Shape detShape_{};
    uint32_t detKind_ = 0;
    uint32_t detColumnRounds_ = 0;
    uint32_t detBufNum_ = 1;
    bool detPrivDkv_ = false;
    uint64_t dkPrivSlotElems_ = 0;
    uint64_t dvPrivSlotElems_ = 0;
    uint32_t coreIdxBn2s2_ = 0;
    uint32_t castChunkRows_ = 0;
    // Cube-side per-column private buffer state (open buffers + reuse gate).
    bool dkAnyOpen_ = false;
    bool dvAnyOpen_ = false;
    bool dkColumnStarted_ = false;
    bool dvColumnStarted_ = false;
    bool dkBufOpen_[2] = {false, false};
    bool dvBufOpen_[2] = {false, false};
    // Vector-side fold buffer tracking for the current column.
    bool dkHasData_[2] = {false, false};
    bool dvHasData_[2] = {false, false};

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
    // BN2S2 dk/dv cast tokens (AIV).
    event_t castMte3ToMte2_{static_cast<event_t>(0)};
    event_t castMte2ToV_{static_cast<event_t>(0)};
    event_t castVToMte3_{static_cast<event_t>(0)};
    AscendC::LocalTensor<float> castInUb_;
    AscendC::LocalTensor<DataType> castOutUb_;

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
    uint32_t dtmVecCoreNum_ = 0;  // IS_DTM: dqVecNum + dkVecNum + dvVecNum
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
    uint32_t decoderS1BlockIdx_ = 0;
    uint64_t decoderS1BlockBegin_ = 0;
    uint64_t decoderValidS2PerN2_ = 0;

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
