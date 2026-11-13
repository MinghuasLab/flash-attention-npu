/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Isolated D>256 128x256 partial-resident bwd kernel. Not included by the
 * legacy fag_kernel.cpp translation unit.
 */
#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_KERNEL_D512_PARTIAL_HPP
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_KERNEL_D512_PARTIAL_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/gemm/block/block_mmad.hpp"

#include "tla/tensor.hpp"
#include "tla/layout.hpp"

#include "fag_common.h"
#include "fag_ub_layout.hpp"
#include "fag_epilogue_post.hpp"
#include "fag_epilogue_pre.hpp"
#include "fag_block.h"
#include "fag_mmad_sdp_partial.hpp"
#include "fag_mmad_dqkv_partial.hpp"
#include "fag_epilogue_scaled_mask_softmax.hpp"
#include "fag_epilogue_softmax_grad_front.hpp"
#include "fag_epilogue_sub_mul.hpp"
#include "kernel_operator.h"

static constexpr uint32_t TASK_PINGPONG = 2;
// Keep the AIV physical UB row stride unchanged for 64-wide tasks. Only the
// valid columns copied to L1 follow tiling kvTile.
static constexpr uint32_t UB_S2_ROW_STRIDE = FagUbLayout::ROW_STRIDE;
static constexpr uint8_t CROSS_CORE_SYNC_MODE = 4;
static constexpr uint16_t V0_V1_FLAG_ID_OFFSET = 16;
static constexpr uint16_t SYNC_C1_TO_V1_FLAG[TASK_PINGPONG] = {0, 1};
static constexpr uint16_t SYNC_C2_TO_V2_FLAG[TASK_PINGPONG] = {2, 3};
static constexpr uint16_t SYNC_V2_TO_C34_FLAG = 4;
static constexpr uint16_t SYNC_V1_TO_C5_FLAG = 5;
static constexpr uint16_t SYNC_C34_TO_V2_FLAG = 8;
static constexpr uint16_t SYNC_C5_TO_V1_FLAG = 9;

/*
 * D>256 partial-resident pipeline (128x256):
 *
 *   C1: S = Q @ K^T   -- Q low half stays in Q[ping]; high + K^T via scratch
 *   C2: dP = dY @ V^T -- Dy low stays in DY[ping]; high + V^T via scratch
 *   C5: dV = P^T @ dY -- Dy low from L1; Dy high streamed from GM into scratch;
 *                       then release DY[ping] for C34
 *   PrefetchC34: Q high -> DY[ping]; full K -> 128KB scratch
 *   C34: dQ = dS @ K; dK = dS^T @ Q (low from Q slot, high from DY slot)
 *
 * Steady state: C1/C2 of task i overlap V1/V2/C5/C34 of task i-1.
 */

template <typename DataType, class BlockMmadSdP_, class BlockMmaddQKV_, class EpilogueScaledMaskSoftmax_,
          class EpilogueSubMul_, FAGTiling950::Layout INPUT_LAYOUT, bool IS_ATTEN_MASK, bool IS_DTM, bool IS_SOFTCAP>
class FlashAttentionScoreGrad950Partial {
  public:
    using BlockMmadSdP = BlockMmadSdP_;
    using BlockMmaddQKV = BlockMmaddQKV_;
    using EpilogueScaledMaskSoftmax = EpilogueScaledMaskSoftmax_;
    using EpilogueSubMul = EpilogueSubMul_;

    // Methods
    CATLASS_DEVICE
    FlashAttentionScoreGrad950Partial() {}

    CATLASS_DEVICE
    ~FlashAttentionScoreGrad950Partial() {}

    CATLASS_DEVICE
    void Init(FAGKernelParams const& params)
    {
        tiling_ = reinterpret_cast<const __gm__ TilingData*>(params.tiling);
        inputStrides_.q = {tiling_->inputStrides.q.batch, tiling_->inputStrides.q.sequence,
                           tiling_->inputStrides.q.head};
        inputStrides_.k = {tiling_->inputStrides.k.batch, tiling_->inputStrides.k.sequence,
                           tiling_->inputStrides.k.head};
        inputStrides_.v = {tiling_->inputStrides.v.batch, tiling_->inputStrides.v.sequence,
                           tiling_->inputStrides.v.head};
        inputStrides_.dout = {tiling_->inputStrides.dout.batch, tiling_->inputStrides.dout.sequence,
                              tiling_->inputStrides.dout.head};
        inputStrides_.out = {tiling_->inputStrides.out.batch, tiling_->inputStrides.out.sequence,
                             tiling_->inputStrides.out.head};
        inputStrides_.enabled = tiling_->inputStrides.enabled;

        doutGm_.SetGlobalBuffer((__gm__ DataType*)params.dout);
        qGm_.SetGlobalBuffer((__gm__ DataType*)params.q);
        kGm_.SetGlobalBuffer((__gm__ DataType*)params.k);
        vGm_.SetGlobalBuffer((__gm__ DataType*)params.v);
        attenMaskGm_.SetGlobalBuffer((__gm__ uint8_t*)params.attenMask);
        softmaxLseGm_.SetGlobalBuffer((__gm__ float*)params.softmaxLse);
        cuSeqQGm_.SetGlobalBuffer((__gm__ int32_t*)params.cuSeqQlen);
        cuSeqKvGm_.SetGlobalBuffer((__gm__ int32_t*)params.cuSeqKvlen);
        // *Used* lengths (seqused); cu tensors above stay the region layout.
        seqUsedQGm_.SetGlobalBuffer((__gm__ int32_t*)params.seqUsedQ);
        seqUsedKvGm_.SetGlobalBuffer((__gm__ int32_t*)params.seqUsedKv);
        hasSeqUsedQ_ = params.seqUsedQ != nullptr;
        hasSeqUsedKv_ = params.seqUsedKv != nullptr;

        dqWorkspaceGm_.SetGlobalBuffer((__gm__ float*)(params.workspace + tiling_->dqOffset));
        dkWorkspaceGm_.SetGlobalBuffer((__gm__ float*)(params.workspace + tiling_->dkOffset));
        dvWorkspaceGm_.SetGlobalBuffer((__gm__ float*)(params.workspace + tiling_->dvOffset));

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
        waveSize_ = static_cast<uint64_t>(coreNum_) * continuousBlockNum_;
        scaleValue_ = tiling_->scaleValue;
        softcapValue_ = tiling_->softcapValue;
        softcapInputScale_ = IS_SOFTCAP ? scaleValue_ / softcapValue_ : scaleValue_;

        // L1 layout:
        //   [P ping][P pong][dS ping][dS pong] | cube working via Mm12L1Offset
        //   cube: [RES_KT|RES_K half][RES_VT][Q0][Q1][DY0][DY1]
        //   Q/DY ping-pong by taskId%2; D<=128 packs RowMajor K above K^T.
        const uint32_t l1TileBytes = qBlockSize_ * kvBlockSize_ * sizeof(DataType);
        const uint32_t l1PBaseOffset = 0;
        const uint32_t l1dSBaseOffset = TASK_PINGPONG * l1TileBytes;
        for (uint32_t i = 0; i < TASK_PINGPONG; ++i) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<DataType>(l1PBaseOffset + i * l1TileBytes);
            l1dSTensor[i] = resource.l1Buf.template GetBufferByByte<DataType>(l1dSBaseOffset + i * l1TileBytes);
        }

        // Per-AIV UB layout.  Ping and pong each own one complete half of UB:
        //   ping: [mm1Res][mm2Res][attenMask][LSE][P(NZ, M+1 padding)][dS(NZ, M+1 padding)][delta]
        //   pong: [mm1Res][mm2Res][attenMask][LSE][P(NZ, M+1 padding)][dS(NZ, M+1 padding)][delta]
        // Fixpipe SPLIT_M sends half of the logical M rows to each AIV.
        // Offsets come from fag_ub_layout.hpp, which is also used by the host
        // budget check in fag_tiling.cpp.
        static_assert(sizeof(DataType) == FagUbLayout::P_ELEMENT_BYTES,
                      "FagUbLayout assumes 2-byte P/dS elements");
        const FagUbLayout::PipelineSlots ubSlots = FagUbLayout::MakePipelineSlots(qBlockSize_);
        const uint32_t mm1SlotOffset = ubSlots.mm1;
        const uint32_t mm2SlotOffset = ubSlots.mm2;
        const uint32_t attenMaskSlotOffset = ubSlots.attenMask;
        const uint32_t lseSlotOffset = ubSlots.lse;
        const uint32_t pSlotOffset = ubSlots.p;
        const uint32_t dSSlotOffset = ubSlots.dS;
        const uint32_t deltaSlotOffset = ubSlots.delta;
        const uint32_t pingPongHalfBytes = ubSlots.halfBytes;

        for (uint32_t i = 0; i < TASK_PINGPONG; ++i) {
            const uint32_t halfBaseOffset = i * pingPongHalfBytes;
            ubMm1ResTensor[i] = resource.ubBuf.template GetBufferByByte<float>(halfBaseOffset + mm1SlotOffset);
            ubMm2ResTensor[i] = resource.ubBuf.template GetBufferByByte<float>(halfBaseOffset + mm2SlotOffset);
            attenMaskUbTensor[i] =
                resource.ubBuf.template GetBufferByByte<uint8_t>(halfBaseOffset + attenMaskSlotOffset);
            lseUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(halfBaseOffset + lseSlotOffset);
            ubPTensor[i] = resource.ubBuf.template GetBufferByByte<DataType>(halfBaseOffset + pSlotOffset);
            ubDSTensor[i] = resource.ubBuf.template GetBufferByByte<DataType>(halfBaseOffset + dSSlotOffset);
            deltaUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(halfBaseOffset + deltaSlotOffset);
        }

        InitEvents();
    }

    CATLASS_DEVICE
    void operator()(FAGKernelParams const& params)
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
        epilogueScaledMaskSoftmax_.Init(resource, params.tiling);
        epilogueSubMul_.Init(resource, params.workspace, params.tiling);
        epiloguePre_.Init(resource, params.workspace, params.tiling);
        epilogueSoftmaxGradFront_.Init(resource, params.dout, params.out, params.workspace, params.tiling);
        epiloguePost_.Init(resource, params.dq, params.dk, params.dv, params.workspace, params.tiling);
        const uint32_t vectorCoreNum = coreNum_ * subBlockNum;
        epiloguePre_(vectorBlockIdx, vectorCoreNum, vWaitMte3Ping);
        epilogueSoftmaxGradFront_(vectorBlockIdx, vectorCoreNum, mte3WaitMte2Ping, mte3WaitMte2Pong, vWaitMte2Ping,
                                  vWaitMte2Pong, vWaitMte3Ping, vWaitMte3Pong);
#endif
        AscendC::SyncAll<false>();
        RunTasks(coreIdx, subBlockIdx);
        AscendC::SyncAll<false>();
#ifdef __DAV_VEC__
        epiloguePost_(vectorBlockIdx, vectorCoreNum, mte3WaitMte2Ping, mte3WaitMte2Pong, vWaitMte2Ping, vWaitMte2Pong,
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
    void GetBatchShape(uint32_t batchIdx, uint64_t& qBatchStart, uint64_t& kvBatchStart, uint32_t& s1Length,
                       uint32_t& s2Length)
    {
        if constexpr (INPUT_LAYOUT == FAGTiling950::Layout::TND) {
            const uint64_t qEnd = static_cast<uint64_t>(cuSeqQGm_.GetValue(batchIdx));
            const uint64_t kvEnd = static_cast<uint64_t>(cuSeqKvGm_.GetValue(batchIdx));
            qBatchStart = batchIdx == 0 ? 0 : static_cast<uint64_t>(cuSeqQGm_.GetValue(batchIdx - 1));
            kvBatchStart = batchIdx == 0 ? 0 : static_cast<uint64_t>(cuSeqKvGm_.GetValue(batchIdx - 1));
            // *Used* lengths from seqused when provided; batch starts stay on cu.
            s1Length = hasSeqUsedQ_ ? static_cast<uint32_t>(seqUsedQGm_.GetValue(batchIdx))
                                    : static_cast<uint32_t>(qEnd - qBatchStart);
            s2Length = hasSeqUsedKv_ ? static_cast<uint32_t>(seqUsedKvGm_.GetValue(batchIdx))
                                     : static_cast<uint32_t>(kvEnd - kvBatchStart);
        } else {
            s1Length = qSeqlen_;
            s2Length = kvSeqlen_;
            qBatchStart = static_cast<uint64_t>(batchIdx) * s1Length;
            kvBatchStart = static_cast<uint64_t>(batchIdx) * s2Length;
        }
    }

    CATLASS_DEVICE
    void GetValidS1BlockRange(uint32_t s1Length, uint32_t s2Length, uint32_t s2BlockIdx,
                              uint32_t& firstValidS1, uint32_t& validS1Num)
    {
        validS1Num = 0;
        // No mask: every S1 block is valid.
        if constexpr (!IS_ATTEN_MASK) {
            validS1Num = static_cast<uint32_t>(CeilDiv(s1Length, qBlockSize_));
            return;
        }
        firstValidS1 = 0;
        const int64_t s2Start = static_cast<int64_t>(s2BlockIdx) * kvBlockSize_;
        const int64_t delta = static_cast<int64_t>(s2Length) - s1Length;
        // Causal: only the first S1 block is restricted; the range extends
        // to the final Q block, without a left-window or KV-tail calculation.
        if (tiling_->maskType == static_cast<uint32_t>(FAGTiling950::MaskType::CAUSAL)) {
            const int64_t s1IdxUp = Max(static_cast<int64_t>(0), s2Start - delta);
            if (s1IdxUp >= s1Length) {
                return;
            }
            firstValidS1 = static_cast<uint32_t>(s1IdxUp / qBlockSize_);
            validS1Num = static_cast<uint32_t>(CeilDiv(s1Length, qBlockSize_)) - firstValidS1;
            return;
        }

        // BAND: both S1 boundaries depend on the window and actual KV tile size.
        const int64_t s1IdxUp = tiling_->s2Token == INT32_MAX ? 0 :
            Max(static_cast<int64_t>(0), s2Start - delta - tiling_->s2Token);
        const int64_t s2RealSize = Min(static_cast<int64_t>(kvBlockSize_), static_cast<int64_t>(s2Length) - s2Start);
        const int64_t s1IdxDown = tiling_->s1Token == INT32_MAX ? s1Length :
            Min(static_cast<int64_t>(s1Length), s2Start + s2RealSize - delta + tiling_->s1Token);
        if (s1IdxUp >= s1IdxDown) {
            return;
        }
        firstValidS1 = static_cast<uint32_t>(s1IdxUp / qBlockSize_);
        validS1Num = static_cast<uint32_t>((s1IdxDown + qBlockSize_ - 1) / qBlockSize_) - firstValidS1;
    }

    CATLASS_DEVICE
    void RunTasks(uint32_t coreIdx, uint32_t subBlockIdx)
    {
        if (coreIdx >= coreNum_ || coreNum_ == 0 || qBlockSize_ == 0 || kvBlockSize_ == 0) {
            return;
        }
        static_assert(!IS_DTM, "d512 partial kernel is non-deterministic only");
#ifdef __DAV_CUBE__
        BlockMmadSdP mm12(resource, Mm12L1Offset());
        BlockMmaddQKV mm345(resource, Mm12L1Offset());
        SetCubeEvents();
#endif
#ifdef __DAV_VEC__
        SetVecEvents();
#endif
        // Each (batch, n2, s2Block) group stays on one AIC for local dk/dv
        // accumulate. Partial reloads K^T/V^T every task (no group residency).
        uint64_t taskId = 0;
        uint64_t kvGroupId = 0;
        bool hasPrev = false;

        for (uint32_t batchIdx = 0; batchIdx < batchNum_; ++batchIdx) {
            uint64_t qBatchStart = 0;
            uint64_t kvBatchStart = 0;
            uint32_t s1Length = 0;
            uint32_t s2Length = 0;
            GetBatchShape(batchIdx, qBatchStart, kvBatchStart, s1Length, s2Length);
            const uint32_t s2BlockNum = static_cast<uint32_t>(CeilDiv(s2Length, kvBlockSize_));

            for (uint32_t n2Idx = 0; n2Idx < kvHeadNum_; ++n2Idx) {
                for (uint32_t s2BlockIdx = 0; s2BlockIdx < s2BlockNum; ++s2BlockIdx, ++kvGroupId) {
                    if ((kvGroupId % coreNum_) != coreIdx) {
                        continue;
                    }

                    uint32_t firstValidS1 = 0;
                    uint32_t validS1Num = 0;
                    GetValidS1BlockRange(s1Length, s2Length, s2BlockIdx, firstValidS1, validS1Num);
                    const uint32_t tasksInGroup = groupNum_ * validS1Num;
                    if (tasksInGroup == 0) {
                        continue;
                    }

                    uint32_t taskInGroup = 0;
                    for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
                        for (uint32_t s1Ord = 0; s1Ord < validS1Num; ++s1Ord, ++taskInGroup) {
                            FAGBlockInfo block{};
                            const uint32_t s1BlockIdx = firstValidS1 + s1Ord;
                            FillBlockInfoFrom(batchIdx, n2Idx, groupIdx, s1BlockIdx, s2BlockIdx, qBatchStart,
                                              kvBatchStart, s1Length, s2Length, block);
                            block.taskId = taskId;
                            block.loadKv = true;  // partial: no KT/VT group residency

#ifdef __DAV_CUBE__
                            ProcessC1Stage(block, mm12);
                            ProcessC2Stage(block, mm12);
                            if (hasPrev) {
                                ProcessC5Stage(previousBlock_, true, mm345);
                                ProcessC34Prefetch(previousBlock_, mm345);
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
            ProcessC34Prefetch(previousBlock_, mm345);
            ProcessC34Stage(previousBlock_, false, mm345);
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
    void FillBlockInfoFrom(uint32_t batchIdx, uint32_t n2Idx, uint32_t groupIdx, uint32_t s1BlockIdx,
                           uint32_t s2BlockIdx, uint64_t qBatchStart, uint64_t kvBatchStart, uint32_t s1Length,
                           uint32_t s2Length, FAGBlockInfo& block)
    {
        const uint32_t s1Start = s1BlockIdx * qBlockSize_;
        const uint32_t s2Start = s2BlockIdx * kvBlockSize_;
        const uint64_t totalS1Start = qBatchStart + s1Start;
        const uint64_t totalS2Start = kvBatchStart + s2Start;
        const uint64_t qHeadIdx = static_cast<uint64_t>(n2Idx) * groupNum_ + groupIdx;

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
        block.qOffset = (totalS1Start * qHeadNum_ + qHeadIdx) * qkHeadDim_;
        block.kOffset = (totalS2Start * kvHeadNum_ + n2Idx) * qkHeadDim_;
        block.vOffset = (totalS2Start * kvHeadNum_ + n2Idx) * vHeadDim_;
        block.doutOffset = (totalS1Start * qHeadNum_ + qHeadIdx) * vHeadDim_;
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
        for (uint32_t eventId = 0; eventId < Catlass::Gemm::Ascend950FagL1Layout::L1_EVENT_COUNT; ++eventId) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventId);
        }
        // One availability token for every physical 64-KB L0C slot.
        for (uint32_t eventId = 0; eventId < Catlass::Gemm::Ascend950FagL0CLayout::L0C_SLOT_NUM; ++eventId) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventId);
        }
    }

    CATLASS_DEVICE
    void WaitCubeEvents()
    {
        for (uint32_t eventId = 0; eventId < 4; ++eventId) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventId);
        }
        for (uint32_t eventId = 0; eventId < Catlass::Gemm::Ascend950FagL1Layout::L1_EVENT_COUNT; ++eventId) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventId);
        }
        for (uint32_t eventId = 0; eventId < Catlass::Gemm::Ascend950FagL0CLayout::L0C_SLOT_NUM; ++eventId) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventId);
        }
    }

    CATLASS_DEVICE
    void ProcessC1Stage(FAGBlockInfo const& block, BlockMmadSdP& mm12)
    {
        const uint32_t slot = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        const uint16_t flagId = SYNC_C1_TO_V1_FLAG[slot];
        auto q = MakeGmTensor(qGm_, InputOffset(inputStrides_.q, block, true), block.s1Extend, qkHeadDim_,
                              inputStrides_.q.sequence);
        auto k = MakeGmTensor(kGm_, InputOffset(inputStrides_.k, block, false), block.s2Extend, qkHeadDim_,
                              inputStrides_.k.sequence);
        auto v = MakeGmTensor(vGm_, InputOffset(inputStrides_.v, block, false), block.s2Extend, vHeadDim_,
                              inputStrides_.v.sequence);
        auto sLayout = tla::MakeLayout(tla::MakeShape(block.s1Extend, block.s2Extend),
                                       tla::MakeStride(UB_S2_ROW_STRIDE, tla::Int<1>{}));
        auto s = tla::MakeTensor(ubMm1ResTensor[slot], sLayout, Catlass::Arch::PositionUB{});
        // Prefetch V^T on the group’s first task so MTE2 overlaps S cube.
        mm12.ComputeS(q, k, s, Catlass::GemmCoord(block.s1Extend, block.s2Extend, qkHeadDim_), block.loadKv, slot,
                      block.loadKv, v, block.s2Extend, vHeadDim_);
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(flagId);
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(flagId + V0_V1_FLAG_ID_OFFSET);
    }

    CATLASS_DEVICE
    void ProcessC2Stage(FAGBlockInfo const& block, BlockMmadSdP& mm12)
    {
        const uint32_t slot = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        const uint16_t flagId = SYNC_C2_TO_V2_FLAG[slot];
        auto dy = MakeGmTensor(doutGm_, InputOffset(inputStrides_.dout, block, true), block.s1Extend, vHeadDim_,
                               inputStrides_.dout.sequence);
        auto v = MakeGmTensor(vGm_, InputOffset(inputStrides_.v, block, false), block.s2Extend, vHeadDim_,
                              inputStrides_.v.sequence);
        auto dpLayout = tla::MakeLayout(tla::MakeShape(block.s1Extend, block.s2Extend),
                                        tla::MakeStride(UB_S2_ROW_STRIDE, tla::Int<1>{}));
        auto dp = tla::MakeTensor(ubMm2ResTensor[slot], dpLayout, Catlass::Arch::PositionUB{});
        // VT already prefetched in C1 when loadKv; only wait on first use.
        mm12.ComputeDP(dy, v, dp, Catlass::GemmCoord(block.s1Extend, block.s2Extend, vHeadDim_),
                       /*loadV=*/false, /*waitV=*/block.loadKv, slot);
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(flagId);
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(flagId + V0_V1_FLAG_ID_OFFSET);
    }

    CATLASS_DEVICE
    void ProcessC5Stage(FAGBlockInfo const& block, bool returnL1, BlockMmaddQKV& mm345)
    {
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_V1_TO_C5_FLAG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_V1_TO_C5_FLAG + V0_V1_FLAG_ID_OFFSET);

        const uint32_t slot = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        auto dy = MakeGmTensor(doutGm_, InputOffset(inputStrides_.dout, block, true), block.s1Extend, vHeadDim_,
                               inputStrides_.dout.sequence);
        auto dv = MakeGmTensor(dvWorkspaceGm_, block.vOffset, block.s2Extend, vHeadDim_, kvHeadNum_ * vHeadDim_);

        // Low-half dY stays in L1 DY[slot] from C2; high half streams from GM.
        // PrefetchC34Operands reuses the released DY slot for Q high half.
        mm345.ComputeDv(l1PTensor[slot], dy, dv, Catlass::GemmCoord(block.s1Extend, vHeadDim_, block.s2Extend), slot,
                        /*enAtomicDv=*/vHeadDim_ > BlockMmaddQKV::BASE);
        if (returnL1) {
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V1_FLAG);
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V1_FLAG + V0_V1_FLAG_ID_OFFSET);
        }
    }

    CATLASS_DEVICE
    void ProcessC34Prefetch(FAGBlockInfo const& block, BlockMmaddQKV& mm345)
    {
        const uint32_t slot = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        auto q = MakeGmTensor(qGm_, InputOffset(inputStrides_.q, block, true), block.s1Extend, qkHeadDim_,
                              inputStrides_.q.sequence);
        auto k = MakeGmTensor(kGm_, InputOffset(inputStrides_.k, block, false), block.s2Extend, qkHeadDim_,
                              inputStrides_.k.sequence);
        mm345.PrefetchC34Operands(q, k, block.s1Extend, block.s2Extend, qkHeadDim_, slot);
    }

    CATLASS_DEVICE
    void ProcessC34Stage(FAGBlockInfo const& block, bool returnL1, BlockMmaddQKV& mm345)
    {
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_V2_TO_C34_FLAG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_V2_TO_C34_FLAG + V0_V1_FLAG_ID_OFFSET);

        const uint32_t slot = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        auto dq = MakeGmTensor(dqWorkspaceGm_, block.qOffset, block.s1Extend, qkHeadDim_, qHeadNum_ * qkHeadDim_);
        auto dk = MakeGmTensor(dkWorkspaceGm_, block.kOffset, block.s2Extend, qkHeadDim_, kvHeadNum_ * qkHeadDim_);

        // PrefetchC34Operands already placed Q-high in DY[slot] and full K in
        // scratch; ComputeDqDk only waits those events and runs the four knives.
        mm345.ComputeDqDk(l1dSTensor[slot], dq, dk, Catlass::GemmCoord(block.s1Extend, qkHeadDim_, block.s2Extend),
                          slot, /*enAtomicDq=*/true,
                          /*enAtomicDk=*/qkHeadDim_ > BlockMmaddQKV::BASE);
        if (returnL1) {
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_C34_TO_V2_FLAG);
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_C34_TO_V2_FLAG + V0_V1_FLAG_ID_OFFSET);
        }
    }


    CATLASS_DEVICE uint64_t InputOffset(const FagTensorStrides& strides, const FAGBlockInfo& block, bool isQ) const
    {
        uint64_t token = isQ ? block.s1Start : block.s2Start;
        uint64_t batchOffset = block.batchIdx * strides.batch;
        if constexpr (INPUT_LAYOUT == FAGTiling950::Layout::TND) {
            token = isQ ? block.totalS1Start : block.totalS2Start;
            batchOffset = 0;
        }
        const uint64_t head = isQ ? block.n2Idx * groupNum_ + block.groupIdx : block.n2Idx;
        return batchOffset + token * strides.sequence + head * strides.head;
    }

    template <class GlobalTensor>
    CATLASS_DEVICE auto MakeGmTensor(GlobalTensor& gm, uint64_t offset, uint32_t rows, uint64_t cols,
                                     uint64_t rowStride)
    {
        auto layout = tla::MakeLayout(tla::MakeShape(rows, cols), tla::MakeStride(rowStride, tla::Int<1>{}));
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
    void ProcessV1Stage(FAGBlockInfo const& block, uint32_t subBlockIdx)
    {
        const uint32_t flagSlot = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(SYNC_C1_TO_V1_FLAG[flagSlot]);

        // Before task 1 and later overwrite P in L1, C5 must return ownership
        // of the buffer used by the preceding task.
        if (block.taskId != 0) {
            AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SYNC_C5_TO_V1_FLAG);
        }

        auto attenMaskLayout = tla::MakeLayout<uint8_t, Catlass::layout::RowMajor>(256, 256);
        auto attenMaskGmTensor = tla::MakeTensor(attenMaskGm_, attenMaskLayout, Catlass::Arch::PositionGM{});

        uint64_t lseRows = 0;
        uint64_t lseColumns = 0;
        uint64_t lseRowOffset = 0;
        uint64_t lseColOffset = 0;
        const uint64_t n1Idx = static_cast<uint64_t>(block.n2Idx) * groupNum_ + block.groupIdx;
        const uint64_t subBlockS1Offset = subBlockIdx * block.firstHalfRealS1;
        const uint64_t subBlockRealS1 = subBlockIdx ? block.s1Extend - block.firstHalfRealS1 : block.firstHalfRealS1;
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
        auto softmaxLseLayout = tla::MakeLayout<float, Catlass::layout::RowMajor>(lseRows, lseColumns);
        auto softmaxLseGmTensor = tla::MakeTensor(
            softmaxLseGm_, softmaxLseLayout, tla::MakeCoord(lseRowOffset, lseColOffset), Catlass::Arch::PositionGM{});

        event_t mte2ToVEvent = flagSlot ? vWaitMte2Pong : vWaitMte2Ping;
        event_t vToMte2Event = flagSlot ? lseMte2WaitVPong : lseMte2WaitVPing;
        event_t vToMte3Event = flagSlot ? mte3WaitVPong : mte3WaitVPing;
        event_t mte3ToVEvent = flagSlot ? pVWaitMte3Pong : pVWaitMte3Ping;
        if (subBlockRealS1 != 0U) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event);
            epilogueScaledMaskSoftmax_(block, attenMaskGmTensor, softmaxLseGmTensor, attenMaskUbTensor[flagSlot],
                                       lseUbTensor[flagSlot], ubMm1ResTensor[flagSlot], ubPTensor[flagSlot],
                                       l1PTensor[flagSlot], mte2ToVEvent, vToMte2Event, vToMte3Event, mte3ToVEvent,
                                       softcapInputScale_, softcapValue_, subBlockRealS1, subBlockIdx);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event);
        }

        // Notify C5 only after this vector sub-block has finished writing P.
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SYNC_V1_TO_C5_FLAG);
    }

    CATLASS_DEVICE
    void ProcessV2Stage(FAGBlockInfo const& block, uint32_t subBlockIdx)
    {
        const uint32_t flagSlot = static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(SYNC_C2_TO_V2_FLAG[flagSlot]);

        // Before task 1 and later overwrite dS in L1, C3/C4 must return
        // ownership of the buffer used by the preceding task.
        if (block.taskId != 0) {
            AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SYNC_C34_TO_V2_FLAG);
        }
        const uint32_t bufferId = flagSlot;
        event_t mte2ToVEvent = flagSlot ? vWaitMte2Pong : vWaitMte2Ping;
        event_t vToMte2Event = flagSlot ? deltaMte2WaitVPong : deltaMte2WaitVPing;
        event_t vToMte3Event = flagSlot ? mte3WaitVPong : mte3WaitVPing;
        event_t mte3ToVEvent = flagSlot ? dSVWaitMte3Pong : dSVWaitMte3Ping;

        const uint32_t subBlockRealS1 = subBlockIdx ? block.s1Extend - block.firstHalfRealS1 : block.firstHalfRealS1;
        if (subBlockRealS1 != 0U) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event);
            epilogueSubMul_(block, ubMm2ResTensor[bufferId], ubPTensor[bufferId], ubMm1ResTensor[bufferId],
                            ubDSTensor[bufferId], deltaUbTensor[bufferId], l1dSTensor[bufferId], mte2ToVEvent,
                            vToMte3Event, mte3ToVEvent, subBlockIdx);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event);
        }

        // Notify C3/C4 only after this vector sub-block has finished writing dS.
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SYNC_V2_TO_C34_FLAG);
    }

#endif

  private:
    Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource;
#ifdef __DAV_VEC__
    Catlass::Epilogue::Block::FagPre<Catlass::Arch::Ascend950, TilingData> epiloguePre_;
    Catlass::Epilogue::Block::FagSoftmaxGradFront<DataType, Catlass::Arch::Ascend950, TilingData>
        epilogueSoftmaxGradFront_;
    EpilogueScaledMaskSoftmax epilogueScaledMaskSoftmax_;
    EpilogueSubMul epilogueSubMul_;
    Catlass::Epilogue::Block::FagPost<DataType, Catlass::Arch::Ascend950, TilingData> epiloguePost_;
#endif

    const __gm__ TilingData* tiling_ = nullptr;

    AscendC::GlobalTensor<DataType> doutGm_;
    AscendC::GlobalTensor<DataType> qGm_;
    AscendC::GlobalTensor<DataType> kGm_;
    AscendC::GlobalTensor<DataType> vGm_;
    AscendC::GlobalTensor<uint8_t> attenMaskGm_;
    AscendC::GlobalTensor<float> softmaxLseGm_;
    AscendC::GlobalTensor<int32_t> cuSeqQGm_;
    AscendC::GlobalTensor<int32_t> cuSeqKvGm_;
    AscendC::GlobalTensor<int32_t> seqUsedQGm_;
    AscendC::GlobalTensor<int32_t> seqUsedKvGm_;
    bool hasSeqUsedQ_ = false;
    bool hasSeqUsedKv_ = false;
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
    FagInputStrides inputStrides_;
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

    FAGBlockInfo previousBlock_{};
};

template <typename DataType, FAGTiling950::Layout INPUT_LAYOUT, bool IS_CAUSAL, bool /*IS_DETERMINISTIC*/, bool IS_SOFTCAP>
CATLASS_GLOBAL void FlashAttentionV3Bwd950Partial(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR mask,
                                           GM_ADDR softmax_lse, GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_k,
                                           GM_ADDR seqUsedQ, GM_ADDR seqUsedKv, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                           GM_ADDR workspace, GM_ADDR tiling)
{
    using ArchTag = Catlass::Arch::Ascend950;
    using L1TileShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<256>>;
    using L0TileShape = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;

    using DispatchPolicySdP = Catlass::Gemm::MmadAscend950FagSdPPartial<2, 2, false>;
    using TileCopySdP =
        Catlass::Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, DataType, Catlass::layout::RowMajor, DataType,
                                                   Catlass::layout::ColumnMajor, float, Catlass::layout::RowMajor, void,
                                                   Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadSdP = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicySdP, L1TileShape, L0TileShape, DataType,
                                                            DataType, float, void, TileCopySdP>;

    using DispatchPolicydQKV = Catlass::Gemm::MmadAscend950FagdQKVPartial<2, 2, false>;
    using TileCopydQKV =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, Catlass::layout::RowMajor, DataType,
                                               Catlass::layout::RowMajor, float, Catlass::layout::RowMajor>;
    using BlockMmaddQKV = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicydQKV, L1TileShape, L0TileShape, DataType,
                                                             DataType, float, void, TileCopydQKV>;

    using DispatchPolicyScaledMaskSoftmax =
        Catlass::Epilogue::EpilogueAscend950FAGScaledMaskSoftmax<INPUT_LAYOUT, IS_CAUSAL, IS_SOFTCAP>;
    using EpilogueScaledMaskSoftmax = Catlass::Epilogue::Block::BlockEpilogue<DispatchPolicyScaledMaskSoftmax, DataType,
                                                                              float, FAGTiling950::FAGTilingData>;
    using DispatchPolicySubMul = Catlass::Epilogue::EpilogueAscend950FAGSubMul<INPUT_LAYOUT, IS_SOFTCAP>;
    using EpilogueSubMul = Catlass::Epilogue::Block::BlockEpilogue<DispatchPolicySubMul, DataType, DataType, float,
                                                                   FAGTiling950::FAGTilingData>;
    using FAGKernel950 =
        FlashAttentionScoreGrad950Partial<DataType, BlockMmadSdP, BlockMmaddQKV, EpilogueScaledMaskSoftmax, EpilogueSubMul,
                                   INPUT_LAYOUT, IS_CAUSAL, /*IS_DTM=*/false, IS_SOFTCAP>;
    FAGKernelParams params{dout,         q,        k,         v,  out, mask, softmax_lse, cu_seqlens_q,
                           cu_seqlens_k, seqUsedQ, seqUsedKv, dq, dk,  dv,   workspace,   tiling};
    FAGKernel950 fag;
    fag(params);
}

#endif // FLASH_ATTN_NPU_ASCEND950_V3_FAG_KERNEL_D512_PARTIAL_HPP
