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
    BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0, uint32_t eventIdStart = 0)
    {
        l1Kt = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_KT_OFFSET);
        l1K = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_K_OFFSET);
        for (uint32_t i = 0; i < Ascend950FagL1Layout::TASK_PINGPONG; ++i) {
            l1Q[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart +
                                                                       Ascend950FagL1Layout::QSlot(i) * L1_TILE_MAX);
            l1Dy[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart +
                                                                        Ascend950FagL1Layout::DySlot(i) * L1_TILE_MAX);
        }

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
    ~BlockMmadTla() {}

    CATLASS_DEVICE void SetL0AReuse(bool enable) { l0AReuse_ = enable; }

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
                                    float fixpipeScaleDk = 1.0f, bool reuseKt = false)
    {
        const uint32_t sqActual = actualShape.m();
        const uint32_t dActual = actualShape.n();
        const uint32_t skvActual = actualShape.k();
        if (dActual <= BASE) {
            ComputeDqDkSmallHeadDim(dsNz, dq, dk, sqActual, skvActual, dActual, taskPing, waitResidentK, enAtomicDq,
                                    initDk, flushDk, enAtomicDk, fixpipeScaleDk);
        } else {
            ComputeDqDkLargeHeadDim(dsNz, k, dq, dk, sqActual, skvActual, dActual, taskPing, enAtomicDq, initDk,
                                    flushDk, enAtomicDk, reuseKt, fixpipeScaleDk);
        }
    }

    /**
     * dv = P^T * dY. dY is already in L1 DY[taskPing] from C2.
     * After this stage the DY slot is released (D>BASE C34 may reuse as K scratch).
     */
    template <class TensorDv>
    CATLASS_DEVICE void ComputeDv(AscendC::LocalTensor<ElementA> pNz, TensorDv& dv, GemmCoord const& actualShape,
                                  uint32_t taskPing, bool initDv, bool flushDv, bool enAtomicDv,
                                  float fixpipeScaleDv = 1.0f)
    {
        const uint32_t sqActual = actualShape.m();
        const uint32_t dvActual = actualShape.n();
        const uint32_t skvActual = actualShape.k();
        if (dvActual <= BASE) {
            ComputeDvSmallHeadDim(pNz, dv, sqActual, skvActual, dvActual, taskPing, initDv, flushDv, enAtomicDv,
                                  fixpipeScaleDv);
        } else {
            ComputeDvLargeHeadDim(pNz, dv, sqActual, skvActual, dvActual, taskPing, enAtomicDv);
        }
    }

    template <class TensorUb>
    CATLASS_DEVICE void ComputeDvWithUb(AscendC::LocalTensor<ElementA> pNz, TensorUb& low, TensorUb& high,
                                        uint32_t taskPing)
    {
        using UbCopy = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementA, layout::RowMajor, ElementB,
            layout::ColumnMajor, float, layout::RowMajor, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
        typename UbCopy::template CopyL0CToDst<TensorUb> copy;
        const uint32_t dyEvent = Ascend950FagL1Layout::DySlot(taskPing);
        constexpr bool cacheA = false;
        const uint32_t cachedSlot = cacheA ? l0AListId : L0AB_STAGES;
        GemmColA(pNz, l1Dy[taskPing], l0CTensorList[SLOT_DQ], BASE, BASE, BASE, BASE, BASE, BASE,
                 true, 0b11, SLOT_DQ, dyEvent, false, false, 0U, 2U * BASE, false, cacheA, cachedSlot);
        auto cLow = tla::MakeTensor(l0CTensorList[SLOT_DQ], tla::MakeLayoutL0C(BASE, BASE), Arch::PositionL0C{});
        copy(low, cLow, 0b11);
        GemmColA(pNz, l1Dy[taskPing], l0CTensorList[SLOT_DQ_PING], BASE, BASE, BASE, BASE, BASE, BASE,
                 true, 0b11, SLOT_DQ_PING, dyEvent, false, true, BASE, 2U * BASE, cacheA, false, cachedSlot);
        auto cHigh = tla::MakeTensor(l0CTensorList[SLOT_DQ_PING], tla::MakeLayoutL0C(BASE, BASE), Arch::PositionL0C{});
        copy(high, cHigh, 0b11);
    }

  protected:
    template <class TensorDq, class TensorDk>
    CATLASS_DEVICE void ComputeDqDkSmallHeadDim(AscendC::LocalTensor<ElementA> dsNz, TensorDq& dq, TensorDk& dk,
                                                uint32_t sqActual, uint32_t skvActual, uint32_t dActual,
                                                uint32_t taskPing, bool waitResidentK, bool enAtomicDq, bool initDk,
                                                bool flushDk, bool enAtomicDk, float fixpipeScaleDk)
    {
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dActual);
        const uint32_t qEvent = Ascend950FagL1Layout::QSlot(taskPing);
        // K resident in RES_KT upper half — no per-task GM load.
        GemmRow(dsNz, l1K, l0CTensorList[SLOT_DQ], sqRound, dRound, skvRound, sqActual, dActual, skvActual, true, 0b11,
                SLOT_DQ, L1_EVENT_K, /*waitB=*/waitResidentK, /*releaseB=*/false,
                /*useKEvent=*/true);
        FixpipeTla(dq, l0CTensorList[SLOT_DQ], sqActual, dActual, sqRound, enAtomicDq, SLOT_DQ);
        GemmColA(dsNz, l1Q[taskPing], l0CTensorList[SLOT_DK], skvRound, dRound, sqRound, skvActual, dActual, sqActual,
                 initDk, flushDk ? 0b11 : 0b10, SLOT_DK, qEvent, /*waitB=*/false, /*releaseB=*/true);
        if (flushDk) {
            FixpipeTla(dk, l0CTensorList[SLOT_DK], skvActual, dActual, skvRound, enAtomicDk, SLOT_DK, fixpipeScaleDk);
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
        const uint32_t dyEvent = Ascend950FagL1Layout::DySlot(taskPing);
        GemmColA(pNz, l1Dy[taskPing], l0CTensorList[SLOT_DV], skvRound, dvRound, sqRound, skvActual, dvActual, sqActual,
                 initDv, flushDv ? 0b11 : 0b10, SLOT_DV, dyEvent, /*waitB=*/false, /*releaseB=*/true);
        if (flushDv) {
            FixpipeTla(dv, l0CTensorList[SLOT_DV], skvActual, dvActual, skvRound, enAtomicDv, SLOT_DV, fixpipeScaleDv);
        }
    }

    template <class TensorK, class TensorDq, class TensorDk>
    CATLASS_DEVICE void ComputeDqDkLargeHeadDim(AscendC::LocalTensor<ElementA> dsNz, TensorK& k, TensorDq& dq,
                                                TensorDk& dk, uint32_t sqActual, uint32_t skvActual, uint32_t dActual,
                                                uint32_t taskPing, bool enAtomicDq, bool initDk, bool flushDk,
                                                bool enAtomicDk, bool reuseKt, float fixpipeScaleDk)
    {
        // D>BASE: Fixpipe each D-slice of dq -- dq sums over s2, so every task
        // must publish its partial.  dk sums over s1 and one AIC walks the whole
        // s1 range of an (n2, s2Block) group, so the two dk slices accumulate in
        // the two dedicated L0C slots (SLOT_DK, SLOT_DK + 1) and are published
        // once at group end -- this cuts dk GM write traffic by the group length
        // (32x for this case).  K still streams slice-by-slice into the DY
        // scratch slot.  L0C budget: dk accumulator = slots 2,3; dq and dv are
        // transient and ping-pong over slots 0,1 (the SDP pair), dq's slice-d
        // fixpipe overlapping slice-(d+1)'s MMAD.
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dFullRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dActual);
        const uint32_t qEvent = Ascend950FagL1Layout::QSlot(taskPing);
        const uint32_t dyEvent = Ascend950FagL1Layout::DySlot(taskPing);
        const uint32_t dLoops = CeilDiv(dActual, BASE);
        const bool cacheA = l0AReuse_ && reuseKt && dLoops == 2U;
        for (uint32_t dIdx = 0; dIdx < dLoops; ++dIdx) {
            const uint32_t nOff = dIdx * BASE;
            const uint32_t nAct = Min(BASE, dActual - nOff);
            const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nAct);
            // dq ping-pongs over the two transient slots so its fixpipe for one
            // D-slice overlaps the MMAD of the next; dk keeps the two persistent
            // accumulator slots.
            const uint32_t slotDq = (dIdx & 1U) ? SLOT_DQ_PING : SLOT_DQ;
            const uint32_t slotDk = SLOT_DK + dIdx;
            const bool lastSlice = (dIdx + 1 == dLoops);
            const uint8_t dqUnit = 0b11;
            const uint8_t dkUnit = flushDk ? 0b11 : 0b10;
            auto kTile = GetTile(k, tla::MakeCoord(0u, nOff), tla::MakeShape(skvActual, nAct));
            auto dqTile = GetTile(dq, tla::MakeCoord(0u, nOff), tla::MakeShape(sqActual, nAct));
            auto dkTile = GetTile(dk, tla::MakeCoord(0u, nOff), tla::MakeShape(skvActual, nAct));
            if (reuseKt) {
                GemmRow(dsNz, l1Kt, l0CTensorList[slotDq], sqRound, nRound, skvRound, sqActual, nAct, skvActual,
                        true, dqUnit, slotDq, dyEvent, false, false, false, nOff, dFullRound,
                        cacheA && lastSlice, cacheA && !lastSlice);
            } else {
            CopyGmToL1B(l1Dy[taskPing], kTile, skvActual, nAct, skvRound, nRound, dyEvent);
            GemmRow(dsNz, l1Dy[taskPing], l0CTensorList[slotDq], sqRound, nRound, skvRound, sqActual, nAct, skvActual,
                    true, dqUnit, slotDq, dyEvent, /*waitB=*/true, /*releaseB=*/true);
            }
            FixpipeTla(dqTile, l0CTensorList[slotDq], sqActual, nAct, sqRound, enAtomicDq, slotDq);
            // Accumulating MMADs must NOT raise the unit flag (0b10): only the
            // group-end MMAD that is followed by FixpipeTla passes 0b11.  A
            // 0b11 without a consuming fixpipe leaves the MAD/FIXP handshake
            // unfinished and deadlocks the cube pipe (see the MLA dqkv path,
            // which uses flushDk ? 0b11 : 0b10 for the same accumulate pattern).
            GemmColA(dsNz, l1Q[taskPing], l0CTensorList[slotDk], skvRound, nRound, sqRound, skvActual, nAct, sqActual,
                     initDk, dkUnit, slotDk, qEvent,
                     /*waitB=*/false, /*releaseB=*/lastSlice,
                     /*bNOff=*/nOff, /*bNFull=*/dFullRound, cacheA && lastSlice, cacheA && !lastSlice);
            if (flushDk) {
                FixpipeTla(dkTile, l0CTensorList[slotDk], skvActual, nAct, skvRound, enAtomicDk, slotDk, fixpipeScaleDk);
            }
        }
    }

    template <class TensorDv>
    CATLASS_DEVICE void ComputeDvLargeHeadDim(AscendC::LocalTensor<ElementA> pNz, TensorDv& dv, uint32_t sqActual,
                                              uint32_t skvActual, uint32_t dvActual, uint32_t taskPing, bool enAtomicDv)
    {
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dvFullRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dvActual);
        const uint32_t dyEvent = Ascend950FagL1Layout::DySlot(taskPing);
        const uint32_t dvLoops = CeilDiv(dvActual, BASE);
        // dk owns both SLOT_DK and SLOT_DV (SLOT_DK + 1) as its persistent
        // s1-accumulators, so dv moves onto the transient SDP pair (slots 0,1),
        // ping-ponging across them exactly like dq.  C5 (dv) runs before C34
        // (dq/dk) within a task, so the two stages reuse the pair in order.
        for (uint32_t dIdx = 0; dIdx < dvLoops; ++dIdx) {
            const uint32_t nOff = dIdx * BASE;
            const uint32_t nAct = Min(BASE, dvActual - nOff);
            const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nAct);
            const uint32_t slotDv = (dIdx & 1U) ? SLOT_DQ_PING : SLOT_DQ;
            const bool lastSlice = (dIdx + 1 == dvLoops);
            auto dvTile = GetTile(dv, tla::MakeCoord(0u, nOff), tla::MakeShape(skvActual, nAct));
            GemmColA(pNz, l1Dy[taskPing], l0CTensorList[slotDv], skvRound, nRound, sqRound, skvActual, nAct,
                     sqActual, true, 0b11, slotDv, dyEvent,
                     /*waitB=*/false, /*releaseB=*/lastSlice,
                     /*bNOff=*/nOff, /*bNFull=*/dvFullRound);
            FixpipeTla(dvTile, l0CTensorList[slotDv], skvActual, nAct, skvRound, enAtomicDv, slotDv);
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
                 uint32_t bNOff = 0, uint32_t bNFull = 0, bool reuseA = false,
                 bool preserveA = false, uint32_t cachedASlot = L0AB_STAGES)
    {
        auto layoutAInL1 = tla::MakeLayout<ElementA, LayoutTagL1A>(mRound, kRound);
        auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mRound, kRound);
        auto layoutBInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(kRound, bNFull == 0 ? nRound : bNFull);
        auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kRound, nRound);
        auto layoutCInL0 = tla::MakeLayoutL0C(mRound, nRound);

        auto tensorL1A = tla::MakeTensor(l1A, layoutAInL1, Arch::PositionL1{});
        const uint32_t aSlot = cachedASlot < L0AB_STAGES ? cachedASlot : l0AListId;
        auto tensorL0A = tla::MakeTensor(l0ATensorList[aSlot], layoutAInL0, Arch::PositionL0A{});
        auto tensorL1B = tla::MakeTensor(l1B, layoutBInL1, Arch::PositionL1{});
        const bool wholeL0B = nRound * kRound * sizeof(ElementB) > L0B_PINGPONG_BUF_SIZE;
        const uint32_t bSlot = wholeL0B ? 0U : l0BListId;
        if (wholeL0B) AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[1U]);
        auto tensorL0B = tla::MakeTensor(l0BTensorList[bSlot], layoutBInL0, Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutCInL0, Arch::PositionL0C{});
        auto tileL1A = GetTile(tensorL1A, tla::MakeCoord(0u, 0u), tla::MakeShape(mRound, kRound));
        auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(0u, bNOff), tla::MakeShape(kRound, nRound));

        const int32_t evB = useKEvent ? l1EventK : l1EventList[l1BEvent];

        if (!reuseA) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[aSlot]);
            copyL1ToL0A(tensorL0A, tileL1A);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[aSlot]);
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[bSlot]);
        if (waitB) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(evB);
        }
        copyL1ToL0B(tensorL0B, tileL1B);
        if (releaseB) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(evB);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[bSlot]);

        if (!reuseA) AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[aSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[bSlot]);
        if constexpr (!ENABLE_UNIT_FLAG) {
            if (initC) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
            }
        }
        uint32_t mMad = (mActual == 1) ? 2 : mRound;
        uint8_t uf = ENABLE_UNIT_FLAG ? unitFlag : 0;
        tileMmad(tensorL0C, tensorL0A, tensorL0B, mMad, nActual, kActual, initC, uf);

        if (!preserveA) AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[aSlot]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[bSlot]);
        if (wholeL0B) AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[1U]);
        l0AListId = (l0AListId + 1 < L0AB_STAGES) ? (l0AListId + 1) : 0;
        l0BListId = (l0BListId + 1 < L0AB_STAGES) ? (l0BListId + 1) : 0;
    }

    CATLASS_DEVICE
    void GemmColA(AscendC::LocalTensor<ElementA> l1A, AscendC::LocalTensor<ElementB> l1B,
                  AscendC::LocalTensor<ElementAccumulator> l0C, uint32_t mRound, uint32_t nRound, uint32_t kRound,
                  uint32_t mActual, uint32_t nActual, uint32_t kActual, bool initC, uint8_t unitFlag, uint32_t l0cSlot,
                  uint32_t l1BEvent, bool waitB, bool releaseB, uint32_t bNOff = 0, uint32_t bNFull = 0, bool reuseA = false,
                 bool preserveA = false, uint32_t cachedASlot = L0AB_STAGES)
    {
        const uint32_t nLayout = (bNFull == 0) ? nRound : bNFull;
        auto layoutAInL1 = tla::MakeLayout<ElementA, LayoutTagL1ACol>(mRound, kRound);
        auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0ACol>(mRound, kRound);
        auto layoutBInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(kRound, nLayout);
        auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kRound, nRound);
        auto layoutCInL0 = tla::MakeLayoutL0C(mRound, nRound);

        auto tensorL1A = tla::MakeTensor(l1A, layoutAInL1, Arch::PositionL1{});
        const uint32_t aSlot = cachedASlot < L0AB_STAGES ? cachedASlot : l0AListId;
        auto tensorL0A = tla::MakeTensor(l0ATensorList[aSlot], layoutAInL0, Arch::PositionL0A{});
        auto tensorL1B = tla::MakeTensor(l1B, layoutBInL1, Arch::PositionL1{});
        const bool wholeL0B = nRound * kRound * sizeof(ElementB) > L0B_PINGPONG_BUF_SIZE;
        const uint32_t bSlot = wholeL0B ? 0U : l0BListId;
        if (wholeL0B) AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[1U]);
        auto tensorL0B = tla::MakeTensor(l0BTensorList[bSlot], layoutBInL0, Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutCInL0, Arch::PositionL0C{});
        auto tileL1A = GetTile(tensorL1A, tla::MakeCoord(0u, 0u), tla::MakeShape(mRound, kRound));
        auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(0u, bNOff), tla::MakeShape(kRound, nRound));

        if (!reuseA) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[aSlot]);
            copyL1ToL0ACol(tensorL0A, tileL1A);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[aSlot]);
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[bSlot]);
        if (waitB) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[l1BEvent]);
        }
        copyL1ToL0B(tensorL0B, tileL1B);
        if (releaseB) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[l1BEvent]);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[bSlot]);

        if (!reuseA) AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[aSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[bSlot]);
        if constexpr (!ENABLE_UNIT_FLAG) {
            if (initC) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
            }
        }
        uint32_t mMad = (mActual == 1) ? 2 : mRound;
        uint8_t uf = ENABLE_UNIT_FLAG ? unitFlag : 0;
        tileMmadCol(tensorL0C, tensorL0A, tensorL0B, mMad, nActual, kActual, initC, uf);

        if (!preserveA) AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[aSlot]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[bSlot]);
        if (wholeL0B) AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[1U]);
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
                AscendC::SetAtomicType<typename decltype(tensorGmTile)::Element>();
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

    bool l0AReuse_ = false;
    AscendC::LocalTensor<ElementB> l1Kt;
    AscendC::LocalTensor<ElementB> l1K;
    AscendC::LocalTensor<ElementB> l1Q[Ascend950FagL1Layout::TASK_PINGPONG];
    AscendC::LocalTensor<ElementB> l1Dy[Ascend950FagL1Layout::TASK_PINGPONG];

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
