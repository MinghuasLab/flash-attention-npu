/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * D>256 puncture dQKV block. After C5 releases Dy[ping], C34 loads
 * Q[:,HALF_D:D) into that Dy slot and the full K[s2,D] into the 128KB scratch.
 * Only included by fag_bwd_d512_partial.cpp.
 */

#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_MMAD_DQKV_PARTIAL_HPP
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_MMAD_DQKV_PARTIAL_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "fag_block.h"

namespace Catlass::Gemm::Block {

template <uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, bool ENABLE_UNIT_FLAG_, class L1TileShape_, class L0TileShape_,
          class ElementA_, class ElementB_, class ElementC_, class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadAscend950FagdQKVPartial<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>, L1TileShape_, L0TileShape_,
                    ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_, TileMmad_> {
  public:
    using DispatchPolicy = MmadAscend950FagdQKVPartial<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using TileMmad = TileMmad_;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using L1AAlignHelper = typename TileCopy::L1AAlignHelper;
    using L1BAlignHelper = typename TileCopy::L1BAlignHelper;
    using TileCopyCol = Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, layout::ColumnMajor, ElementB, layout::RowMajor,
                                                      ElementC, layout::RowMajor>;
    using LayoutTagL1ACol = typename TileCopyCol::LayoutTagL1A;
    using LayoutTagL0ACol = typename TileCopyCol::LayoutTagL0A;
    using CopyL1ToL0ACol = typename TileCopyCol::CopyL1ToL0A;
    using TileMmadCol = Gemm::Tile::TileMmadTla<ArchTag, ElementA, LayoutTagL1ACol>;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t L0AB_STAGES = DispatchPolicy::L0AB_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static constexpr uint32_t L0C_BUF_SIZE = DispatchPolicy::L0C_BUF_SIZE;
    static constexpr uint32_t BASE = DispatchPolicy::BASE;
    static constexpr uint32_t HALF_D = DispatchPolicy::HALF_D;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0AB_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0AB_STAGES;
    static constexpr uint32_t SLOT_DQ = Ascend950FagL0CLayout::SLOT_DQ;
    static constexpr uint32_t SLOT_DQ_PING = Ascend950FagL0CLayout::SLOT_DQ_PING;
    static constexpr uint32_t SLOT_DK = Ascend950FagL0CLayout::SLOT_DK;
    static constexpr uint32_t SLOT_DV = Ascend950FagL0CLayout::SLOT_DV;
    static constexpr uint32_t L1_TILE_MAX = BASE * 256 * sizeof(ElementA);
    static constexpr uint32_t L1_SCRATCH0_OFFSET = Ascend950FagL1Layout::SLOT_RES_KT * L1_TILE_MAX;
    static constexpr uint32_t L1_EVENT_S0 = Ascend950FagL1Layout::SLOT_RES_KT;

    static_assert(std::is_same_v<ArchTag, Arch::Ascend950>, "Requires Ascend950");

    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0, uint32_t eventIdStart = 0)
    {
        // Contiguous 128KB scratch starts at RES_KT (adjacent to RES_VT).
        l1Scratch = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_SCRATCH0_OFFSET);
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
        l0AListId = 0;
        l0BListId = 0;
    }

    CATLASS_DEVICE ~BlockMmadTla() {}

    // Streams dY high half from GM into scratch for n2/n3; low half stays in Dy[ping].
    template <class TensorDy, class TensorDv>
    CATLASS_DEVICE void ComputeDv(AscendC::LocalTensor<ElementA> pNz, TensorDy& dyGm, TensorDv& dv,
                                  GemmCoord const& actualShape, uint32_t taskPing, bool enAtomicDv)
    {
        const uint32_t sqActual = actualShape.m();
        const uint32_t dvActual = actualShape.n();
        const uint32_t skvActual = actualShape.k();
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dyEvent = Ascend950FagL1Layout::DySlot(taskPing);
        const uint32_t dLow = Min(HALF_D, dvActual);
        const uint32_t dHigh = dvActual > HALF_D ? (dvActual - HALF_D) : 0U;
        const uint32_t dLowRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dLow);
        const uint32_t dHighRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dHigh);
        const uint32_t dvLoops = CeilDiv(dvActual, BASE);
        bool loadedHi = false;

        for (uint32_t dIdx = 0; dIdx < dvLoops; ++dIdx) {
            const uint32_t nOff = dIdx * BASE;
            const uint32_t nAct = Min(BASE, dvActual - nOff);
            const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nAct);
            const uint32_t slotDv = (dIdx & 1U) ? SLOT_DK : SLOT_DV;
            const bool lastSlice = (dIdx + 1 == dvLoops);
            auto dvTile = GetTile(dv, tla::MakeCoord(0u, nOff), tla::MakeShape(skvActual, nAct));

            if (nOff < HALF_D) {
                // Dy low was Waited in C2 GemmABt (releaseA=false); do not re-Wait.
                GemmColA(pNz, l1Dy[taskPing], l0CTensorList[slotDv], skvRound, nRound, sqRound, skvActual, nAct,
                         sqActual, true, 0b11, slotDv, dyEvent, /*waitB=*/false,
                         /*releaseB=*/false, nOff, dLowRound);
            } else {
                if (!loadedHi) {
                    auto dyHi = GetTile(dyGm, tla::MakeCoord(0u, HALF_D), tla::MakeShape(sqActual, dHigh));
                    CopyGmToL1B(l1Scratch, dyHi, sqActual, dHigh, sqRound, dHighRound, L1_EVENT_S0);
                    loadedHi = true;
                }
                // Scratch was just produced by CopyGmToL1B; wait once on first high knife.
                GemmColA(pNz, l1Scratch, l0CTensorList[slotDv], skvRound, nRound, sqRound, skvActual, nAct, sqActual,
                         true, 0b11, slotDv, L1_EVENT_S0, /*waitB=*/(nOff == HALF_D),
                         /*releaseB=*/false, nOff - HALF_D, dHighRound);
            }
            FixpipeTla(dvTile, l0CTensorList[slotDv], skvActual, nAct, skvRound, enAtomicDv, slotDv);
        }
        // Hand Dy[ping] and scratch to PrefetchC34Operands (Q-high + full K).
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[dyEvent]);
        if (loadedHi) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[L1_EVENT_S0]);
        }
    }

    template <class TensorDq, class TensorDk>
    CATLASS_DEVICE void ComputeDqDk(AscendC::LocalTensor<ElementA> dsNz, TensorDq& dq, TensorDk& dk,
                                    GemmCoord const& actualShape, uint32_t taskPing, bool enAtomicDq, bool enAtomicDk)
    {
        // High-half Q must already be in Dy[ping] and full K in scratch
        // (PrefetchC34Operands).
        const uint32_t sqActual = actualShape.m();
        const uint32_t dActual = actualShape.n();
        const uint32_t skvActual = actualShape.k();
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dFullRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dActual);
        const uint32_t dLow = Min(HALF_D, dActual);
        const uint32_t dHigh = dActual > HALF_D ? (dActual - HALF_D) : 0U;
        const uint32_t dLowRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dLow);
        const uint32_t dHighRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dHigh);
        const uint32_t qEvent = Ascend950FagL1Layout::QSlot(taskPing);
        const uint32_t dyEvent = Ascend950FagL1Layout::DySlot(taskPing);
        const uint32_t dLoops = CeilDiv(dActual, BASE);

        for (uint32_t dIdx = 0; dIdx < dLoops; ++dIdx) {
            const uint32_t nOff = dIdx * BASE;
            const uint32_t nAct = Min(BASE, dActual - nOff);
            const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nAct);
            const uint32_t slotDq = (dIdx & 1U) ? SLOT_DQ_PING : SLOT_DQ;
            const uint32_t slotDk = (dIdx & 1U) ? SLOT_DK : SLOT_DV;
            const bool lastSlice = (dIdx + 1 == dLoops);
            auto dqTile = GetTile(dq, tla::MakeCoord(0u, nOff), tla::MakeShape(sqActual, nAct));
            auto dkTile = GetTile(dk, tla::MakeCoord(0u, nOff), tla::MakeShape(skvActual, nAct));

            // dQ = dS @ K[:, nOff:]: K is fully resident in scratch (bNFull=D).
            GemmRow(dsNz, l1Scratch, l0CTensorList[slotDq], sqRound, nRound, skvRound, sqActual, nAct, skvActual, true,
                    0b11, slotDq, L1_EVENT_S0, /*waitB=*/(dIdx == 0), /*releaseB=*/lastSlice, nOff, dFullRound);
            FixpipeTla(dqTile, l0CTensorList[slotDq], sqActual, nAct, sqRound, enAtomicDq, slotDq);

            // dK = dS^T @ Q[:, nOff:]: low from Q slot (C1 left it; waitB=false like
            // ComputeDqDkLargeHeadDim), high from Dy slot after PrefetchC34Operands.
            if (nOff < HALF_D) {
                const bool lastLow = (nOff + BASE >= dLow) || lastSlice;
                GemmColA(dsNz, l1Q[taskPing], l0CTensorList[slotDk], skvRound, nRound, sqRound, skvActual, nAct,
                         sqActual, true, 0b11, slotDk, qEvent, /*waitB=*/false,
                         /*releaseB=*/lastLow, nOff, dLowRound);
            } else {
                GemmColA(dsNz, l1Dy[taskPing], l0CTensorList[slotDk], skvRound, nRound, sqRound, skvActual, nAct,
                         sqActual, true, 0b11, slotDk, dyEvent, /*waitB=*/(nOff == HALF_D),
                         /*releaseB=*/lastSlice, nOff - HALF_D, dHighRound);
            }
            FixpipeTla(dkTile, l0CTensorList[slotDk], skvActual, nAct, skvRound, enAtomicDk, slotDk);
        }
    }

    // Load Q high into released Dy[ping] and full K into scratch before ComputeDqDk.
    template <class TensorQ, class TensorK>
    CATLASS_DEVICE void PrefetchC34Operands(TensorQ& q, TensorK& k, uint32_t sqActual, uint32_t skvActual,
                                            uint32_t dActual, uint32_t taskPing)
    {
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dFullRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dActual);
        const uint32_t dHigh = dActual > HALF_D ? (dActual - HALF_D) : 0U;
        const uint32_t dyEvent = Ascend950FagL1Layout::DySlot(taskPing);

        if (dHigh > 0) {
            const uint32_t dHighRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dHigh);
            auto qHi = GetTile(q, tla::MakeCoord(0u, HALF_D), tla::MakeShape(sqActual, dHigh));
            CopyGmToL1B(l1Dy[taskPing], qHi, sqActual, dHigh, sqRound, dHighRound, dyEvent);
        }
        CopyGmToL1B(l1Scratch, k, skvActual, dActual, skvRound, dFullRound, L1_EVENT_S0);
    }

  protected:
    template <class TensorGm>
    CATLASS_DEVICE void CopyGmToL1B(AscendC::LocalTensor<ElementB> l1Buf, TensorGm& gm, uint32_t kActual,
                                    uint32_t nActual, uint32_t kRound, uint32_t nRound, uint32_t eventIdx)
    {
        using CopyGmToL1BOp = typename TileCopy::template CopyGmToL1B<TensorGm>;
        CopyGmToL1BOp copyGmToL1B;
        auto layoutL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(kRound, nRound);
        auto tensorL1 = tla::MakeTensor(l1Buf, layoutL1, Arch::PositionL1{});
        auto tensorTileGm = GetTile(gm, tla::MakeCoord(0u, 0u), tla::MakeShape(kActual, nActual));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[eventIdx]);
        copyGmToL1B(tensorL1, tensorTileGm);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[eventIdx]);
    }

    // GemmRow with column offset into full-width K in scratch.
    CATLASS_DEVICE void GemmRow(AscendC::LocalTensor<ElementA> l1A, AscendC::LocalTensor<ElementB> l1B,
                                AscendC::LocalTensor<ElementAccumulator> l0C, uint32_t mRound, uint32_t nRound,
                                uint32_t kRound, uint32_t mActual, uint32_t nActual, uint32_t kActual, bool initC,
                                uint8_t unitFlag, uint32_t l0cSlot, uint32_t l1BEvent, bool waitB, bool releaseB,
                                uint32_t bNOff, uint32_t bNFull)
    {
        auto layoutAInL1 = tla::MakeLayout<ElementA, LayoutTagL1A>(mRound, kRound);
        auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mRound, kRound);
        auto layoutBInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(kRound, bNFull);
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
        copyL1ToL0A(tensorL0A, tileL1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
        if (waitB) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[l1BEvent]);
        }
        copyL1ToL0B(tensorL0B, tileL1B);
        if (releaseB) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[l1BEvent]);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);
        if (initC) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
        }
        const uint32_t mMad = (mActual == 1) ? 2 : mRound;
        tileMmad(tensorL0C, tensorL0A, tensorL0B, mMad, nActual, kActual, initC, ENABLE_UNIT_FLAG ? unitFlag : 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
        l0AListId = (l0AListId + 1 < L0AB_STAGES) ? (l0AListId + 1) : 0;
        l0BListId = (l0BListId + 1 < L0AB_STAGES) ? (l0BListId + 1) : 0;
    }

    CATLASS_DEVICE void GemmColA(AscendC::LocalTensor<ElementA> l1A, AscendC::LocalTensor<ElementB> l1B,
                                 AscendC::LocalTensor<ElementAccumulator> l0C, uint32_t mRound, uint32_t nRound,
                                 uint32_t kRound, uint32_t mActual, uint32_t nActual, uint32_t kActual, bool initC,
                                 uint8_t unitFlag, uint32_t l0cSlot, uint32_t l1BEvent, bool waitB, bool releaseB,
                                 uint32_t bNOff = 0, uint32_t bNFull = 0)
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
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
        if (waitB) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[l1BEvent]);
        }
        copyL1ToL0B(tensorL0B, tileL1B);
        if (releaseB) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[l1BEvent]);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);
        if (initC) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
        }
        const uint32_t mMad = (mActual == 1) ? 2 : mRound;
        tileMmadCol(tensorL0C, tensorL0A, tensorL0B, mMad, nActual, kActual, initC, ENABLE_UNIT_FLAG ? unitFlag : 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
        l0AListId = (l0AListId + 1 < L0AB_STAGES) ? (l0AListId + 1) : 0;
        l0BListId = (l0BListId + 1 < L0AB_STAGES) ? (l0BListId + 1) : 0;
    }

    template <class TensorGm>
    CATLASS_DEVICE void FixpipeTla(TensorGm& gmC, AscendC::LocalTensor<ElementAccumulator> l0C, uint32_t mActual,
                                   uint32_t nActual, uint32_t mRound, bool enAtomic, uint32_t l0cSlot)
    {
        auto layoutCInL0 = tla::MakeLayoutL0C(mRound, nActual);
        auto tensorL0C = tla::MakeTensor(l0C, layoutCInL0, Arch::PositionL0C{});
        auto tensorGmTile = GetTile(gmC, tla::MakeCoord(0u, 0u), tla::MakeShape(mActual, nActual));
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<decltype(tensorGmTile)>;
        CopyL0CToDst copyL0CToDst;
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0cSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0cSlot]);
        if (enAtomic) {
            AscendC::SetAtomicType<float>();
            copyL0CToDst(tensorGmTile, tensorL0C);
            AscendC::SetAtomicNone();
        } else {
            copyL0CToDst(tensorGmTile, tensorL0C);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
    }

    AscendC::LocalTensor<ElementB> l1Scratch;
    AscendC::LocalTensor<ElementB> l1Q[Ascend950FagL1Layout::TASK_PINGPONG];
    AscendC::LocalTensor<ElementB> l1Dy[Ascend950FagL1Layout::TASK_PINGPONG];
    AscendC::LocalTensor<ElementA> l0ATensorList[L0AB_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[L0AB_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];
    int32_t l0AEventList[L0AB_STAGES];
    int32_t l0BEventList[L0AB_STAGES];
    int32_t l0CEventList[L0C_STAGES];
    int32_t l1EventList[Ascend950FagL1Layout::SLOT_COUNT];
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};
    TileMmad tileMmad;
    TileMmadCol tileMmadCol;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0ACol copyL1ToL0ACol;
    CopyL1ToL0B copyL1ToL0B;
};

} // namespace Catlass::Gemm::Block

#endif
