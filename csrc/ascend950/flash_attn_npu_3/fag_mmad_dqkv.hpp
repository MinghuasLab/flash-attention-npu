/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Modified by Minghua Shen, 2026.
 */

#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_MMAD_DQKV_HPP
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_MMAD_DQKV_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_dst.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "fag_block.h"

namespace Catlass::Gemm::Block {

template <uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, bool ENABLE_UNIT_FLAG_, class L1TileShape_, class L0TileShape_,
          class ElementA_, class ElementB_, class ElementC_, class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadAscend950FagdQKV<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>, L1TileShape_, L0TileShape_,
                    ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_, TileMmad_> {
  public:
    using DispatchPolicy = MmadAscend950FagdQKV<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using TileMmad = TileMmad_;

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A; // zN for RowMajor A
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B; // zN for RowMajor B
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A; // zN on Ascend950
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B; // nZ

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

    using L1AAlignHelper = typename TileCopy::L1AAlignHelper;
    using L1BAlignHelper = typename TileCopy::L1BAlignHelper;

    // ColumnMajor-A TileCopy for dk/dv (dS^T / P^T)
    using TileCopyCol = Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, layout::ColumnMajor, ElementB,
                                                      layout::RowMajor, ElementC, layout::RowMajor>;
    using LayoutTagL1ACol = typename TileCopyCol::LayoutTagL1A; // nZ
    using LayoutTagL0ACol = typename TileCopyCol::LayoutTagL0A; // zN
    using CopyL1ToL0ACol = typename TileCopyCol::CopyL1ToL0A;
    using TileMmadCol = Gemm::Tile::TileMmadTla<ArchTag, ElementA, LayoutTagL1ACol>;

    static_assert(tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
                  "L1TileShape must be tla::tuple and static!");
    static_assert(tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
                  "L0TileShape must be tla::tuple and static!");
    static_assert(std::is_same_v<ArchTag, Arch::Ascend950>, "Requires Ascend950");

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t L0AB_STAGES = DispatchPolicy::L0AB_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static constexpr uint32_t L0C_BUF_SIZE = DispatchPolicy::L0C_BUF_SIZE;
    static constexpr uint32_t BASE = DispatchPolicy::BASE;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0AB_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0AB_STAGES;

    static constexpr uint32_t SLOT_DQ = Ascend950FagL0CLayout::SLOT_DQ;
    static constexpr uint32_t SLOT_DQ_PING = Ascend950FagL0CLayout::SLOT_DQ_PING;
    static constexpr uint32_t SLOT_DK = Ascend950FagL0CLayout::SLOT_DK;
    static constexpr uint32_t SLOT_DV = Ascend950FagL0CLayout::SLOT_DV;

    static constexpr uint32_t L1_TILE_MAX = BASE * 256 * sizeof(ElementA);
    static constexpr uint32_t L1_TILE_HALF = BASE * BASE * sizeof(ElementA);
    static constexpr uint32_t L1_KT_OFFSET = Ascend950FagL1Layout::SLOT_RES_KT * L1_TILE_MAX;
    // RowMajor K packs into the upper half of the RES_KT 64KB slot when D<=BASE
    // (lower half holds K^T for C1). D>BASE keeps streaming K via DY scratch.
    static constexpr uint32_t L1_K_OFFSET = L1_KT_OFFSET + L1_TILE_HALF;
    static constexpr uint32_t L1_EVENT_K = Ascend950FagL1Layout::L1_EVENT_K;
    static_assert(Ascend950FagL1Layout::SLOT_COUNT * L1_TILE_MAX +
                          2 * Ascend950FagL1Layout::TASK_PINGPONG * BASE * BASE * sizeof(ElementA) <=
                      ArchTag::L1_SIZE,
                  "L1 overflow");
    static_assert(L0C_STAGES * L0C_BUF_SIZE <= ArchTag::L0C_SIZE, "L0C overflow");
    static_assert(L1_TILE_HALF * 2 == L1_TILE_MAX, "RES_K half-slot packing assumes 64KB tiles");

    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0, uint32_t eventIdStart = 0,
                 bool depth2 = false)
        : depth2_(depth2)
    {
        l1KT = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_KT_OFFSET);
        l1K = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_K_OFFSET);
        for (uint32_t i = 0; i < Ascend950FagL1Layout::TASK_PINGPONG; ++i) {
            l1Q[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart +
                                                                       Ascend950FagL1Layout::QSlot(i) * L1_TILE_MAX);
            l1Dy[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart +
                                                                        Ascend950FagL1Layout::DySlot(i) * L1_TILE_MAX);
        }
        l1Q[2] = resource.l1Buf.template GetBufferByByte<ElementB>(
            l1BufAddrStart + Ascend950FagL1Layout::SLOT_RES_VT * L1_TILE_MAX + Ascend950FagL1Layout::TILE_BYTES);
        l1Dy[2] = resource.l1Buf.template GetBufferByByte<ElementB>(
            l1BufAddrStart + Ascend950FagL1Layout::SLOT_DY1 * L1_TILE_MAX + Ascend950FagL1Layout::TILE_BYTES);

        for (uint32_t i = 0; i < L0AB_STAGES; i++) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0AEventList[i] = static_cast<int32_t>(i + eventIdStart);
            l0BEventList[i] = static_cast<int32_t>(i + L0AB_STAGES + eventIdStart);
        }
        for (uint32_t i = 0; i < L0C_STAGES; i++) {
            l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_BUF_SIZE * i);
            l0CEventList[i] = static_cast<int32_t>((i + eventIdStart) % 8);
        }
        for (uint32_t i = 0; i < Ascend950FagL1Layout::SLOT_COUNT; i++) {
            l1EventList[i] = static_cast<int32_t>((i + eventIdStart) % 8);
        }
        l1EventK = static_cast<int32_t>((L1_EVENT_K + eventIdStart) % 8);
        l0AListId = 0;
        l0BListId = 0;
    }

    CATLASS_DEVICE
    uint32_t L0ABStage() const
    {
        return l0AListId;
    }

    CATLASS_DEVICE
    void SetL0ABStage(uint32_t stage)
    {
        l0AListId = stage;
        l0BListId = stage;
    }

    CATLASS_DEVICE
    uint32_t QSlotOf(uint32_t taskPing) const
    {
        return depth2_ ? Ascend950FagL1Layout::QSlotDepth2(taskPing) : Ascend950FagL1Layout::QSlot(taskPing);
    }

    CATLASS_DEVICE
    uint32_t DySlotOf(uint32_t taskPing) const
    {
        return depth2_ ? Ascend950FagL1Layout::DySlotDepth2(taskPing) : Ascend950FagL1Layout::DySlot(taskPing);
    }

    CATLASS_DEVICE
    uint32_t QBufOf(uint32_t taskPing) const
    {
        return depth2_ ? taskPing : (taskPing & 1U);
    }

    CATLASS_DEVICE
    uint32_t DyBufOf(uint32_t taskPing) const
    {
        return depth2_ ? taskPing : (taskPing & 1U);
    }

    CATLASS_DEVICE
    ~BlockMmadTla() {}

    /// Prefetch RowMajor K into the RES_KT upper half (D<=BASE only).
    /// Prefer calling from C5 so MTE2 overlaps dv cube work.
    template <class TensorK>
    CATLASS_DEVICE void LoadResidentK(TensorK& k, uint32_t skvActual, uint32_t dActual)
    {
        if (dActual > BASE) {
            return;
        }
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dActual);
        CopyGmToL1B(l1K, k, skvActual, dActual, skvRound, dRound,
                    /*eventIdx=*/0, /*useKEvent=*/true);
    }

    CATLASS_DEVICE
    void ReleaseResidentK()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventK);
    }

    /**
     * dq = dS * K, dk = dS^T * Q.
     * Q is already in L1 Q[taskPing] from C1.
     * D<=BASE: K is resident in RES_K (prefetched at C5 / initDkv).
     * D>BASE: K streams into DY[taskPing] scratch after C5.
     */
    template <class TensorK, class TensorDq, class TensorDk>
    CATLASS_DEVICE void ComputeDqDk(AscendC::LocalTensor<ElementA> dsNz, TensorK& k, TensorDq& dq, TensorDk& dk,
                                    GemmCoord const& actualShape, uint32_t taskPing, bool waitResidentK,
                                    bool enAtomicDq, bool initDk, bool flushDk, bool enAtomicDk,
                                    float fixpipeScaleDk = 1.0f, bool reuseKT = false, bool keepDv = false, bool alternateC = false)
    {
        const uint32_t sqActual = actualShape.m();
        const uint32_t dActual = actualShape.n();
        const uint32_t skvActual = actualShape.k();
        if (dActual <= BASE) {
            ComputeDqDkSmallHeadDim(dsNz, dq, dk, sqActual, skvActual, dActual, taskPing, waitResidentK, enAtomicDq,
                                    initDk, flushDk, enAtomicDk, fixpipeScaleDk);
        } else {
            ComputeDqDkLargeHeadDim(dsNz, k, dq, dk, sqActual, skvActual, dActual, taskPing, enAtomicDq, enAtomicDk, reuseKT, keepDv, alternateC);
        }
    }

    /**
     * dv = P^T * dY. dY is already in L1 DY[taskPing] from C2.
     * After this stage the DY slot is released (D>BASE C34 may reuse as K scratch).
     */
    template <class TensorDv>
    CATLASS_DEVICE void ComputeDv(AscendC::LocalTensor<ElementA> pNz, TensorDv& dv, GemmCoord const& actualShape,
                                  uint32_t taskPing, bool initDv, bool flushDv, bool enAtomicDv,
                                  float fixpipeScaleDv = 1.0f, bool keepDv = false)
    {
        const uint32_t sqActual = actualShape.m();
        const uint32_t dvActual = actualShape.n();
        const uint32_t skvActual = actualShape.k();
        if (dvActual <= BASE) {
            ComputeDvSmallHeadDim(pNz, dv, sqActual, skvActual, dvActual, taskPing, initDv, flushDv, enAtomicDv,
                                  fixpipeScaleDv);
        } else {
            ComputeDvLargeHeadDim(pNz, dv, sqActual, skvActual, dvActual, taskPing, enAtomicDv, initDv, flushDv, keepDv);
        }
    }

    template <class TensorDq, class TensorDk>
    CATLASS_DEVICE void ComputeDqDkToUb(AscendC::LocalTensor<ElementA> dsNz, TensorDq& dq, TensorDk& dk,
                                       GemmCoord const& shape, uint32_t taskPing, bool waitResidentK,
                                       bool enAtomicDq, bool initDk, bool flushDk)
    {
        ComputeDqDkSmallHeadDim<true>(dsNz, dq, dk, shape.m(), shape.k(), shape.n(), taskPing,
                                      waitResidentK, enAtomicDq, initDk, flushDk, false, 1.0f);
    }

  protected:
    template <bool DkToUb = false, class TensorDq, class TensorDk>
    CATLASS_DEVICE void ComputeDqDkSmallHeadDim(AscendC::LocalTensor<ElementA> dsNz, TensorDq& dq, TensorDk& dk,
                                                uint32_t sqActual, uint32_t skvActual, uint32_t dActual,
                                                uint32_t taskPing, bool waitResidentK, bool enAtomicDq, bool initDk,
                                                bool flushDk, bool enAtomicDk, float fixpipeScaleDk)
    {
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dActual);
        const uint32_t qBuf = QBufOf(taskPing);
        const uint32_t qEvent = QSlotOf(taskPing);
        // K resident in RES_KT upper half — no per-task GM load.
        // dq always Fixpipes: OUTER_LAST (0b11). Resident dk matches fusion:
        // accumulate → UNITFLAG_ENABLE (0b10) with no Fixpipe; flush → 0b11 + Fixpipe.
        constexpr uint8_t kUfLast = 0b11;
        constexpr uint8_t kUfCont = 0b10;
        GemmRow(dsNz, l1K, l0CTensorList[SLOT_DQ], sqRound, dRound, skvRound, sqActual, dActual, skvActual, true, kUfLast,
                SLOT_DQ, L1_EVENT_K, /*waitB=*/waitResidentK, /*releaseB=*/false,
                /*useKEvent=*/true);
        FixpipeTla(dq, l0CTensorList[SLOT_DQ], sqActual, dActual, sqRound, enAtomicDq, SLOT_DQ);
        const uint8_t ufDk = flushDk ? kUfLast : kUfCont;
        GemmColA(dsNz, l1Q[qBuf], l0CTensorList[SLOT_DK], skvRound, dRound, sqRound, skvActual, dActual, sqActual,
                 initDk, ufDk, SLOT_DK, qEvent, /*waitB=*/false, /*releaseB=*/true);
        if (flushDk) {
            if constexpr (DkToUb) {
                auto layout = tla::MakeLayoutL0C(skvRound, dActual);
                auto accumulator = tla::MakeTensor(l0CTensorList[SLOT_DK], layout, Arch::PositionL0C{});
                using Copy = Gemm::Tile::CopyL0CToUBTla<ArchTag, decltype(accumulator), TensorDk,
                    Gemm::Tile::CopyL0CToUBMode::SPLIT_M, Gemm::Tile::ScaleGranularity::NO_QUANT, false>;
                Copy copy;
                copy(dk, accumulator, 0b11);
            } else {
                FixpipeTla(dk, l0CTensorList[SLOT_DK], skvActual, dActual, skvRound, enAtomicDk, SLOT_DK, fixpipeScaleDk);
            }
            ReleaseResidentK();
        }
    }

    template <class TensorDv>
    CATLASS_DEVICE void ComputeDvSmallHeadDim(AscendC::LocalTensor<ElementA> pNz, TensorDv& dv, uint32_t sqActual,
                                              uint32_t skvActual, uint32_t dvActual, uint32_t taskPing, bool initDv,
                                              bool flushDv, bool enAtomicDv, float fixpipeScaleDv)
    {
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dvRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dvActual);
        const uint32_t dyBuf = DyBufOf(taskPing);
        const uint32_t dyEvent = DySlotOf(taskPing);
        constexpr uint8_t kUfLast = 0b11;
        constexpr uint8_t kUfCont = 0b10;
        const uint8_t ufDv = flushDv ? kUfLast : kUfCont;
        GemmColA(pNz, l1Dy[dyBuf], l0CTensorList[SLOT_DV], skvRound, dvRound, sqRound, skvActual, dvActual, sqActual,
                 initDv, ufDv, SLOT_DV, dyEvent, /*waitB=*/false, /*releaseB=*/true);
        if (flushDv) {
            FixpipeTla(dv, l0CTensorList[SLOT_DV], skvActual, dvActual, skvRound, enAtomicDv, SLOT_DV, fixpipeScaleDv);
        }
    }

    template <class TensorK, class TensorDq, class TensorDk>
    CATLASS_DEVICE void ComputeDqDkLargeHeadDim(AscendC::LocalTensor<ElementA> dsNz, TensorK& k, TensorDq& dq,
                                                TensorDk& dk, uint32_t sqActual, uint32_t skvActual, uint32_t dActual,
                                                uint32_t taskPing, bool enAtomicDq, bool enAtomicDk, bool reuseKT, bool keepDv, bool alternateC)
    {
        // D>BASE: Fixpipe each D-slice. Q is full in L1 from C1; K streams
        // slice-by-slice into the DY scratch slot. Each GM slice honors
        // overwrite on the first private-buffer contribution and atomic-add thereafter.
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dFullRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dActual);
        const uint32_t qBuf = QBufOf(taskPing);
        const uint32_t dyBuf = DyBufOf(taskPing);
        const uint32_t qEvent = QSlotOf(taskPing);
        const uint32_t dyEvent = DySlotOf(taskPing);
        const uint32_t dLoops = CeilDiv(dActual, BASE);
        for (uint32_t dIdx = 0; dIdx < dLoops; ++dIdx) {
            const uint32_t nOff = dIdx * BASE;
            const uint32_t nAct = Min(BASE, dActual - nOff);
            const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nAct);
            // Keep alternating physical slots across both slices when dV owns the upper slots.
            const bool pingDq = alternateC || (dIdx & 1U) != 0;
            const uint32_t slotDq = pingDq ? SLOT_DQ_PING : SLOT_DQ;
            const uint32_t slotDk = keepDv ? (pingDq ? SLOT_DQ : SLOT_DQ_PING) : SLOT_DK;
            const bool lastSlice = (dIdx + 1 == dLoops);
            auto kTile = GetTile(k, tla::MakeCoord(0u, nOff), tla::MakeShape(skvActual, nAct));
            auto dqTile = GetTile(dq, tla::MakeCoord(0u, nOff), tla::MakeShape(sqActual, nAct));
            auto dkTile = GetTile(dk, tla::MakeCoord(0u, nOff), tla::MakeShape(skvActual, nAct));
            if (reuseKT) {
                // nZ [D, Skv] and zN [Skv, D] have identical storage.
                GemmRow(dsNz, l1KT, l0CTensorList[slotDq], sqRound, nRound, skvRound, sqActual, nAct, skvActual,
                        true, 0b11, slotDq, 0, /*waitB=*/false, /*releaseB=*/false, false, nOff, dFullRound);
            } else {
                CopyGmToL1B(l1Dy[dyBuf], kTile, skvActual, nAct, skvRound, nRound, dyEvent);
                GemmRow(dsNz, l1Dy[dyBuf], l0CTensorList[slotDq], sqRound, nRound, skvRound, sqActual, nAct, skvActual,
                        true, 0b11, slotDq, dyEvent, /*waitB=*/true, /*releaseB=*/true);
            }
            FixpipeTla(dqTile, l0CTensorList[slotDq], sqActual, nAct, sqRound, enAtomicDq, slotDq);
            GemmColA(dsNz, l1Q[qBuf], l0CTensorList[slotDk], skvRound, nRound, sqRound, skvActual, nAct, sqActual, true,
                     0b11, slotDk, qEvent,
                     /*waitB=*/false, /*releaseB=*/lastSlice,
                     /*bNOff=*/nOff, /*bNFull=*/dFullRound);
            FixpipeTla(dkTile, l0CTensorList[slotDk], skvActual, nAct, skvRound, enAtomicDk, slotDk);
        }
    }

    template <class TensorDv>
    CATLASS_DEVICE void ComputeDvLargeHeadDim(AscendC::LocalTensor<ElementA> pNz, TensorDv& dv, uint32_t sqActual,
                                              uint32_t skvActual, uint32_t dvActual, uint32_t taskPing, bool enAtomicDv, bool initDv,
                                              bool flushDv, bool keepDv)
    {
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dvFullRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dvActual);
        const uint32_t dyBuf = DyBufOf(taskPing);
        const uint32_t dyEvent = DySlotOf(taskPing);
        const uint32_t dvLoops = CeilDiv(dvActual, BASE);
        for (uint32_t dIdx = 0; dIdx < dvLoops; ++dIdx) {
            const uint32_t nOff = dIdx * BASE;
            const uint32_t nAct = Min(BASE, dvActual - nOff);
            const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nAct);
            const bool lastSlice = (dIdx + 1 == dvLoops);
            const uint32_t slotDv = keepDv ? (dIdx == 0 ? SLOT_DK : SLOT_DV) : SLOT_DV;
            const bool flush = !keepDv || flushDv;
            const uint8_t unitFlag = flush ? 0b11 : 0b10;
            auto dvTile = GetTile(dv, tla::MakeCoord(0u, nOff), tla::MakeShape(skvActual, nAct));
            GemmColA(pNz, l1Dy[dyBuf], l0CTensorList[slotDv], skvRound, nRound, sqRound, skvActual, nAct, sqActual,
                     !keepDv || initDv, unitFlag, slotDv, dyEvent,
                     /*waitB=*/false, /*releaseB=*/lastSlice,
                     /*bNOff=*/nOff, /*bNFull=*/dvFullRound);
            if (flush) {
                FixpipeTla(dvTile, l0CTensorList[slotDv], skvActual, nAct, skvRound, enAtomicDv, slotDv);
            }
        }
    }

    template <class TensorGm>
    CATLASS_DEVICE void CopyGmToL1B(AscendC::LocalTensor<ElementB> l1Buf, TensorGm& gm, uint32_t kActual,
                                    uint32_t nActual, uint32_t kRound, uint32_t nRound, uint32_t eventIdx,
                                    bool useKEvent = false)
    {
        using CopyGmToL1BOp = typename TileCopy::template CopyGmToL1B<TensorGm>;
        CopyGmToL1BOp copyGmToL1B;

        auto layoutL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(kRound, nRound);
        auto tensorL1 = tla::MakeTensor(l1Buf, layoutL1, Arch::PositionL1{});
        auto tensorTileGm = GetTile(gm, tla::MakeCoord(0u, 0u), tla::MakeShape(kActual, nActual));

        const int32_t ev = useKEvent ? l1EventK : l1EventList[eventIdx];
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ev);
        copyGmToL1B(tensorL1, tensorTileGm);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ev);
    }

    CATLASS_DEVICE
    void GemmRow(AscendC::LocalTensor<ElementA> l1A, AscendC::LocalTensor<ElementB> l1B,
                 AscendC::LocalTensor<ElementAccumulator> l0C, uint32_t mRound, uint32_t nRound, uint32_t kRound,
                 uint32_t mActual, uint32_t nActual, uint32_t kActual, bool initC, uint8_t unitFlag, uint32_t l0cSlot,
                 uint32_t l1BEvent, bool waitB, bool releaseB, bool useKEvent = false,
                 uint32_t bNOff = 0, uint32_t bNFull = 0)
    {
        auto layoutAInL1 = tla::MakeLayout<ElementA, LayoutTagL1A>(mRound, kRound);
        auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mRound, kRound);
        auto layoutBInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(kRound, bNFull == 0 ? nRound : bNFull);
        auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kRound, nRound);
        auto layoutCInL0 = tla::MakeLayoutL0C(mRound, nRound);

        auto tensorL1A = tla::MakeTensor(l1A, layoutAInL1, Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0ATensorList[l0AListId], layoutAInL0, Arch::PositionL0A{});
        auto tensorL1B = tla::MakeTensor(l1B, layoutBInL1, Arch::PositionL1{});
        auto tensorL0B = tla::MakeTensor(l0BTensorList[l0BListId], layoutBInL0, Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutCInL0, Arch::PositionL0C{});
        auto tileL1A = GetTile(tensorL1A, tla::MakeCoord(0u, 0u), tla::MakeShape(mRound, kRound));
        auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(0u, bNOff), tla::MakeShape(kRound, nRound));

        const int32_t evB = useKEvent ? l1EventK : l1EventList[l1BEvent];

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
        copyL1ToL0A(tensorL0A, tileL1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
        if (waitB) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(evB);
        }
        copyL1ToL0B(tensorL0B, tileL1B);
        if (releaseB) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(evB);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);
        if constexpr (!ENABLE_UNIT_FLAG) {
            if (initC) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
            }
        }
        uint32_t mMad = (mActual == 1) ? 2 : mRound;
        uint8_t uf = ENABLE_UNIT_FLAG ? unitFlag : 0;
        tileMmad(tensorL0C, tensorL0A, tensorL0B, mMad, nActual, kActual, initC, uf);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
        l0AListId = (l0AListId + 1 < L0AB_STAGES) ? (l0AListId + 1) : 0;
        l0BListId = (l0BListId + 1 < L0AB_STAGES) ? (l0BListId + 1) : 0;
    }

    CATLASS_DEVICE
    void GemmColA(AscendC::LocalTensor<ElementA> l1A, AscendC::LocalTensor<ElementB> l1B,
                  AscendC::LocalTensor<ElementAccumulator> l0C, uint32_t mRound, uint32_t nRound, uint32_t kRound,
                  uint32_t mActual, uint32_t nActual, uint32_t kActual, bool initC, uint8_t unitFlag, uint32_t l0cSlot,
                  uint32_t l1BEvent, bool waitB, bool releaseB, uint32_t bNOff = 0, uint32_t bNFull = 0)
    {
        const uint32_t nLayout = (bNFull == 0) ? nRound : bNFull;
        auto layoutAInL1 = tla::MakeLayout<ElementA, LayoutTagL1ACol>(mRound, kRound);
        auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0ACol>(mRound, kRound);
        auto layoutBInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(kRound, nLayout);
        auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kRound, nRound);
        auto layoutCInL0 = tla::MakeLayoutL0C(mRound, nRound);

        auto tensorL1A = tla::MakeTensor(l1A, layoutAInL1, Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0ATensorList[l0AListId], layoutAInL0, Arch::PositionL0A{});
        auto tensorL1B = tla::MakeTensor(l1B, layoutBInL1, Arch::PositionL1{});
        auto tensorL0B = tla::MakeTensor(l0BTensorList[l0BListId], layoutBInL0, Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutCInL0, Arch::PositionL0C{});
        auto tileL1A = GetTile(tensorL1A, tla::MakeCoord(0u, 0u), tla::MakeShape(mRound, kRound));
        auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(0u, bNOff), tla::MakeShape(kRound, nRound));

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
        copyL1ToL0ACol(tensorL0A, tileL1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);

        const bool bMutex = Ascend950FagL1Layout::IsMutexSlot(l1BEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
        if (waitB) {
            if (bMutex) {
                AscendC::Mutex::Lock<PIPE_MTE1>(static_cast<int32_t>(Ascend950FagL1Layout::MutexOf(l1BEvent)));
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[l1BEvent]);
            }
        }
        copyL1ToL0B(tensorL0B, tileL1B);
        if (releaseB) {
            if (bMutex) {
                AscendC::Mutex::Unlock<PIPE_MTE1>(static_cast<int32_t>(Ascend950FagL1Layout::MutexOf(l1BEvent)));
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[l1BEvent]);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);
        if constexpr (!ENABLE_UNIT_FLAG) {
            if (initC) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
            }
        }
        uint32_t mMad = (mActual == 1) ? 2 : mRound;
        uint8_t uf = ENABLE_UNIT_FLAG ? unitFlag : 0;
        tileMmadCol(tensorL0C, tensorL0A, tensorL0B, mMad, nActual, kActual, initC, uf);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
        l0AListId = (l0AListId + 1 < L0AB_STAGES) ? (l0AListId + 1) : 0;
        l0BListId = (l0BListId + 1 < L0AB_STAGES) ? (l0BListId + 1) : 0;
    }

    template <class TensorGm>
    CATLASS_DEVICE void FixpipeTla(TensorGm& gmC, AscendC::LocalTensor<ElementAccumulator> l0C, uint32_t mActual,
                                   uint32_t nActual, uint32_t mRound, bool enAtomic, uint32_t l0cSlot,
                                   float scale = 1.0f)
    {
        auto layoutCInL0 = tla::MakeLayoutL0C(mRound, nActual);
        auto tensorL0C = tla::MakeTensor(l0C, layoutCInL0, Arch::PositionL0C{});
        auto tensorGmTile = GetTile(gmC, tla::MakeCoord(0u, 0u), tla::MakeShape(mActual, nActual));

        // Unit flag: hardware orders mmad→fixpipe→next mmad on this L0C.
        // Soft M_FIX/FIX_M would scalar-stall the cube issue queue (fusion skips them).
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0cSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0cSlot]);
        }

        // scale!=1: Fixpipe PER_TENSOR cast (fp32 L0C → bf16/fp16 GM with deqScalar).
        // Used by earlyDkvCast to write dk directly and skip the AIV cast path.
        if (scale != 1.0f) {
            using CopyScaled = Gemm::Tile::CopyL0CToGmTla<ArchTag, decltype(tensorL0C), decltype(tensorGmTile),
                                                          Gemm::Tile::ScaleGranularity::PER_TENSOR, false>;
            CopyScaled copyL0CToDst({scale});
            if constexpr (ENABLE_UNIT_FLAG) {
                copyL0CToDst(tensorGmTile, tensorL0C, 0b11);
            } else {
                copyL0CToDst(tensorGmTile, tensorL0C);
            }
        } else {
            using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<decltype(tensorGmTile)>;
            CopyL0CToDst copyL0CToDst;
            if (enAtomic) {
                AscendC::SetAtomicType<float>();
                if constexpr (ENABLE_UNIT_FLAG) {
                    copyL0CToDst(tensorGmTile, tensorL0C, 0b11);
                } else {
                    copyL0CToDst(tensorGmTile, tensorL0C);
                }
                AscendC::SetAtomicNone();
            } else if constexpr (ENABLE_UNIT_FLAG) {
                copyL0CToDst(tensorGmTile, tensorL0C, 0b11);
            } else {
                copyL0CToDst(tensorGmTile, tensorL0C);
            }
        }
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
        }
    }

    AscendC::LocalTensor<ElementB> l1KT;
    AscendC::LocalTensor<ElementB> l1K;
    AscendC::LocalTensor<ElementB> l1Q[Ascend950FagL1Layout::QDY_DEPTH];
    AscendC::LocalTensor<ElementB> l1Dy[Ascend950FagL1Layout::QDY_DEPTH];
    bool depth2_{false};

    AscendC::LocalTensor<ElementA> l0ATensorList[L0AB_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[L0AB_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];

    int32_t l0AEventList[L0AB_STAGES];
    int32_t l0BEventList[L0AB_STAGES];
    int32_t l0CEventList[L0C_STAGES];
    int32_t l1EventList[Ascend950FagL1Layout::SLOT_COUNT];
    int32_t l1EventK{0};

    uint32_t l0AListId{0};
    uint32_t l0BListId{0};

    TileMmad tileMmad;
    TileMmadCol tileMmadCol;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0ACol copyL1ToL0ACol;
    CopyL1ToL0B copyL1ToL0B;
};

} // namespace Catlass::Gemm::Block

#endif // FLASH_ATTN_NPU_ASCEND950_V3_FAG_MMAD_DQKV_HPP
