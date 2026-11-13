/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * D>256 puncture SdP block: Q/Dy keep columns [0, HALF_D); high half and
 * K^T/V^T stream through the shared 128KB scratch (RES_KT + RES_VT).
 * Only included by fag_bwd_d512_partial.cpp.
 */

#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_MMAD_SDP_PARTIAL_HPP
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_MMAD_SDP_PARTIAL_HPP

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
struct BlockMmadTla<MmadAscend950FagSdPPartial<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>, L1TileShape_, L0TileShape_,
                    ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_, TileMmad_> {
  public:
    using DispatchPolicy = MmadAscend950FagSdPPartial<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
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

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t L0AB_STAGES = DispatchPolicy::L0AB_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static constexpr uint32_t L0C_BUF_SIZE = DispatchPolicy::L0C_BUF_SIZE;
    static constexpr uint32_t BASE = DispatchPolicy::BASE;
    static constexpr uint32_t HALF_D = DispatchPolicy::HALF_D;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0AB_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0AB_STAGES;
    static constexpr uint32_t L1_TILE_MAX = BASE * 256 * sizeof(ElementA);
    static constexpr uint32_t L1_SCRATCH0_OFFSET = Ascend950FagL1Layout::SLOT_RES_KT * L1_TILE_MAX;
    static constexpr uint32_t L1_SCRATCH1_OFFSET = Ascend950FagL1Layout::SLOT_RES_VT * L1_TILE_MAX;
    static constexpr uint32_t L1_EVENT_S0 = Ascend950FagL1Layout::SLOT_RES_KT;
    static constexpr uint32_t L1_EVENT_S1 = Ascend950FagL1Layout::SLOT_RES_VT;

    static_assert(L0C_STAGES == Ascend950FagL0CLayout::L0C_SDP_SLOT_NUM, "SdP must use exactly 2 L0C slots");
    static_assert(std::is_same_v<ArchTag, Arch::Ascend950>, "Requires Ascend950");

    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0, uint32_t eventIdStart = 0)
    {
        AscendC::SetHF32Mode(false);
        l1Scratch0 = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_SCRATCH0_OFFSET);
        l1Scratch1 = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_SCRATCH1_OFFSET);
        for (uint32_t i = 0; i < Ascend950FagL1Layout::TASK_PINGPONG; ++i) {
            l1Q[i] = resource.l1Buf.template GetBufferByByte<ElementA>(
                l1BufAddrStart + Ascend950FagL1Layout::QSlot(i) * L1_TILE_MAX);
            l1Dy[i] = resource.l1Buf.template GetBufferByByte<ElementA>(
                l1BufAddrStart + Ascend950FagL1Layout::DySlot(i) * L1_TILE_MAX);
        }
        for (uint32_t i = 0; i < L0AB_STAGES; i++) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0AEventList[i] = static_cast<int32_t>(i + eventIdStart);
            l0BEventList[i] = static_cast<int32_t>(i + L0AB_STAGES + eventIdStart);
        }
        for (uint32_t i = 0; i < L0C_STAGES; i++) {
            l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(
                L0C_BUF_SIZE * (Ascend950FagL0CLayout::SLOT_SDP_0 + i));
            l0CEventList[i] = static_cast<int32_t>((Ascend950FagL0CLayout::SLOT_SDP_0 + i + eventIdStart) % 8);
        }
        for (uint32_t i = 0; i < Ascend950FagL1Layout::SLOT_COUNT; i++) {
            l1EventList[i] = static_cast<int32_t>((i + eventIdStart) % 8);
        }
        l0AListId = 0;
        l0BListId = 0;
        l0CListId = 0;
    }

    CATLASS_DEVICE ~BlockMmadTla() {}

    /**
     * S = Q * K^T with D split at HALF_D. Low-half Q stays in Q[taskPing].
     * Scratch is released at the end (no group KV residency).
     */
    template <class TensorA, class TensorB, class TensorC, class TensorV>
    CATLASS_DEVICE void ComputeS(TensorA& tensorA, TensorB& tensorB, TensorC& tensorC, GemmCoord const& actualShape,
                                 bool /*loadK*/, uint32_t taskPing, bool /*prefetchV*/, TensorV& /*tensorV*/,
                                 uint32_t /*vSkvActual*/, uint32_t /*vDActual*/)
    {
        const uint32_t sqActual = actualShape.m();
        const uint32_t skvActual = actualShape.n();
        const uint32_t dActual = actualShape.k();
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1BAlignHelper::N_ALIGNED>(skvActual);
        const uint32_t qEvent = Ascend950FagL1Layout::QSlot(taskPing);
        const uint32_t slotC = l0CListId;
        const uint32_t dLow = Min(HALF_D, dActual);
        const uint32_t dHigh = dActual > HALF_D ? (dActual - HALF_D) : 0U;
        const uint32_t dLowRound = RoundUp<L1AAlignHelper::K_ALIGNED>(dLow);

        auto qLow = GetTile(tensorA, tla::MakeCoord(0u, 0u), tla::MakeShape(sqActual, dLow));
        CopyGmToL1A(l1Q[taskPing], qLow, sqActual, dLow, sqRound, dLowRound, qEvent);
        auto kLow = GetTile(tensorB, tla::MakeCoord(0u, 0u), tla::MakeShape(skvActual, dLow));
        CopyGmToL1BT(l1Scratch0, kLow, dLow, skvActual, dLowRound, skvRound, L1_EVENT_S0);
        GemmABt(l1Q[taskPing], l1Scratch0, l0CTensorList[slotC], sqRound, skvRound, dLowRound, sqActual, skvActual,
                dLow, slotC, qEvent, L1_EVENT_S0, /*waitB=*/true, /*releaseB=*/true, /*releaseA=*/false,
                /*initC=*/true);

        if (dHigh > 0) {
            const uint32_t dHighRound = RoundUp<L1AAlignHelper::K_ALIGNED>(dHigh);
            auto qHi = GetTile(tensorA, tla::MakeCoord(0u, HALF_D), tla::MakeShape(sqActual, dHigh));
            // Q high lands in scratch1 as ElementA-viewed buffer; reuse ElementB buffer bytes.
            AscendC::LocalTensor<ElementA> l1QHi =
                l1Scratch1.template ReinterpretCast<ElementA>();
            CopyGmToL1A(l1QHi, qHi, sqActual, dHigh, sqRound, dHighRound, L1_EVENT_S1);
            auto kHi = GetTile(tensorB, tla::MakeCoord(0u, HALF_D), tla::MakeShape(skvActual, dHigh));
            CopyGmToL1BT(l1Scratch0, kHi, dHigh, skvActual, dHighRound, skvRound, L1_EVENT_S0);
            GemmABt(l1QHi, l1Scratch0, l0CTensorList[slotC], sqRound, skvRound, dHighRound, sqActual, skvActual, dHigh,
                    slotC, L1_EVENT_S1, L1_EVENT_S0, /*waitB=*/true, /*releaseB=*/true, /*releaseA=*/true,
                    /*initC=*/false);
        }

        FixpipeUb(tensorC, l0CTensorList[slotC], sqActual, skvActual, sqRound, slotC);
        l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
    }

    template <class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE void ComputeDP(TensorA& tensorA, TensorB& tensorB, TensorC& tensorC, GemmCoord const& actualShape,
                                  bool /*loadV*/, bool /*waitV*/, uint32_t taskPing)
    {
        const uint32_t sqActual = actualShape.m();
        const uint32_t skvActual = actualShape.n();
        const uint32_t dActual = actualShape.k();
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1BAlignHelper::N_ALIGNED>(skvActual);
        const uint32_t dyEvent = Ascend950FagL1Layout::DySlot(taskPing);
        const uint32_t slotC = l0CListId;
        const uint32_t dLow = Min(HALF_D, dActual);
        const uint32_t dHigh = dActual > HALF_D ? (dActual - HALF_D) : 0U;
        const uint32_t dLowRound = RoundUp<L1AAlignHelper::K_ALIGNED>(dLow);

        auto dyLow = GetTile(tensorA, tla::MakeCoord(0u, 0u), tla::MakeShape(sqActual, dLow));
        CopyGmToL1A(l1Dy[taskPing], dyLow, sqActual, dLow, sqRound, dLowRound, dyEvent);
        auto vLow = GetTile(tensorB, tla::MakeCoord(0u, 0u), tla::MakeShape(skvActual, dLow));
        CopyGmToL1BT(l1Scratch0, vLow, dLow, skvActual, dLowRound, skvRound, L1_EVENT_S0);
        GemmABt(l1Dy[taskPing], l1Scratch0, l0CTensorList[slotC], sqRound, skvRound, dLowRound, sqActual, skvActual,
                dLow, slotC, dyEvent, L1_EVENT_S0, /*waitB=*/true, /*releaseB=*/true, /*releaseA=*/false,
                /*initC=*/true);

        if (dHigh > 0) {
            const uint32_t dHighRound = RoundUp<L1AAlignHelper::K_ALIGNED>(dHigh);
            auto dyHi = GetTile(tensorA, tla::MakeCoord(0u, HALF_D), tla::MakeShape(sqActual, dHigh));
            AscendC::LocalTensor<ElementA> l1DyHi = l1Scratch1.template ReinterpretCast<ElementA>();
            CopyGmToL1A(l1DyHi, dyHi, sqActual, dHigh, sqRound, dHighRound, L1_EVENT_S1);
            auto vHi = GetTile(tensorB, tla::MakeCoord(0u, HALF_D), tla::MakeShape(skvActual, dHigh));
            CopyGmToL1BT(l1Scratch0, vHi, dHigh, skvActual, dHighRound, skvRound, L1_EVENT_S0);
            GemmABt(l1DyHi, l1Scratch0, l0CTensorList[slotC], sqRound, skvRound, dHighRound, sqActual, skvActual, dHigh,
                    slotC, L1_EVENT_S1, L1_EVENT_S0, /*waitB=*/true, /*releaseB=*/true, /*releaseA=*/true,
                    /*initC=*/false);
        }

        FixpipeUb(tensorC, l0CTensorList[slotC], sqActual, skvActual, sqRound, slotC);
        l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
    }

    // No-op stubs so the shared task loop can keep the call sites.
    CATLASS_DEVICE void ReleaseResidentKT() {}
    CATLASS_DEVICE void ReleaseResidentVT() {}

  protected:
    template <class TensorGm>
    CATLASS_DEVICE void CopyGmToL1A(AscendC::LocalTensor<ElementA> l1Buf, TensorGm& gm, uint32_t mActual,
                                    uint32_t kActual, uint32_t mRound, uint32_t kRound, uint32_t eventIdx)
    {
        using CopyGmToL1AOp = typename TileCopy::template CopyGmToL1A<TensorGm>;
        CopyGmToL1AOp copyGmToL1A;
        auto layoutL1 = tla::MakeLayout<ElementA, LayoutTagL1A>(mRound, kRound);
        auto tensorL1 = tla::MakeTensor(l1Buf, layoutL1, Arch::PositionL1{});
        auto tensorTileGm = GetTile(gm, tla::MakeCoord(0u, 0u), tla::MakeShape(mActual, kActual));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[eventIdx]);
        copyGmToL1A(tensorL1, tensorTileGm);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[eventIdx]);
    }

    template <class TensorGm>
    CATLASS_DEVICE void CopyGmToL1BT(AscendC::LocalTensor<ElementB> l1Buf, TensorGm& gm, uint32_t dActual,
                                     uint32_t skvActual, uint32_t dRound, uint32_t skvRound, uint32_t eventIdx)
    {
        // gm may be a GetTile view of K/V[:, HALF_D:). GetTile keeps the parent
        // data() pointer and encodes the origin in coord(); rebuilding a
        // ColumnMajor view from data() alone would re-read the low half.
        const int64_t rowStride = static_cast<int64_t>(tla::get<0>(gm.stride()));
        const uint64_t baseOffset = static_cast<uint64_t>(gm.layout()(gm.coord()));
        auto layoutCol = tla::MakeLayout(tla::MakeShape(dActual, skvActual),
                                         tla::MakeStride(tla::Int<1>{}, rowStride),
                                         tla::MakeShape(dActual, skvActual));
        auto gmCol = tla::MakeTensor(gm.data()[baseOffset], layoutCol, Arch::PositionGM{});
        using CopyGmToL1BOp = typename TileCopy::template CopyGmToL1B<decltype(gmCol)>;
        CopyGmToL1BOp copyGmToL1B;
        auto layoutL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(dRound, skvRound);
        auto tensorL1 = tla::MakeTensor(l1Buf, layoutL1, Arch::PositionL1{});
        auto tensorTileGm = GetTile(gmCol, tla::MakeCoord(0u, 0u), tla::MakeShape(dActual, skvActual));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[eventIdx]);
        copyGmToL1B(tensorL1, tensorTileGm);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[eventIdx]);
    }

    CATLASS_DEVICE void GemmABt(AscendC::LocalTensor<ElementA> l1A, AscendC::LocalTensor<ElementB> l1B,
                                AscendC::LocalTensor<ElementAccumulator> l0C, uint32_t mRound, uint32_t nRound,
                                uint32_t kRound, uint32_t mActual, uint32_t nActual, uint32_t kActual, uint32_t l0cSlot,
                                uint32_t l1AEvent, uint32_t l1BEvent, bool waitB, bool releaseB, bool releaseA,
                                bool initCOuter)
    {
        auto layoutAInL1 = tla::MakeLayout<ElementA, LayoutTagL1A>(mRound, kRound);
        auto layoutBInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(kRound, nRound);
        auto tensorL1A = tla::MakeTensor(l1A, layoutAInL1, Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(l1B, layoutBInL1, Arch::PositionL1{});
        auto layoutCInL0 = tla::MakeLayoutL0C(mRound, nRound);
        auto tensorL0C = tla::MakeTensor(l0C, layoutCInL0, Arch::PositionL0C{});

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[l1AEvent]);
        if (waitB) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[l1BEvent]);
        }
        if (initCOuter) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
        }

        const uint32_t kLoops = CeilDiv(kActual, BASE);
        for (uint32_t kIdx = 0; kIdx < kLoops; ++kIdx) {
            const uint32_t kOff = kIdx * BASE;
            const uint32_t tileK = (kIdx + 1 < kLoops) ? BASE : (kActual - kOff);
            const uint32_t kTileRound = RoundUp<L1AAlignHelper::K_ALIGNED>(tileK);
            auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mRound, kTileRound);
            auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kTileRound, nRound);
            auto tensorL0A = tla::MakeTensor(l0ATensorList[l0AListId], layoutAInL0, Arch::PositionL0A{});
            auto tensorL0B = tla::MakeTensor(l0BTensorList[l0BListId], layoutBInL0, Arch::PositionL0B{});
            auto tileL1A = GetTile(tensorL1A, tla::MakeCoord(0u, kOff), tla::MakeShape(mRound, kTileRound));
            auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(kOff, 0u), tla::MakeShape(kTileRound, nRound));

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
            copyL1ToL0A(tensorL0A, tileL1A);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
            copyL1ToL0B(tensorL0B, tileL1B);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);

            const bool initC = initCOuter && (kIdx == 0);
            const uint32_t mMad = (mActual == 1) ? 2 : mRound;
            tileMmad(tensorL0C, tensorL0A, tensorL0B, mMad, nActual, tileK, initC, 0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
            l0AListId = (l0AListId + 1 < L0AB_STAGES) ? (l0AListId + 1) : 0;
            l0BListId = (l0BListId + 1 < L0AB_STAGES) ? (l0BListId + 1) : 0;
        }
        if (releaseA) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[l1AEvent]);
        }
        if (releaseB) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[l1BEvent]);
        }
    }

    template <class TensorUb>
    CATLASS_DEVICE void FixpipeUb(TensorUb& ubC, AscendC::LocalTensor<ElementAccumulator> l0C, uint32_t mActual,
                                  uint32_t nActual, uint32_t mRound, uint32_t l0cSlot)
    {
        auto layoutCInL0 = tla::MakeLayoutL0C(mRound, nActual);
        auto tensorL0C = tla::MakeTensor(l0C, layoutCInL0, Arch::PositionL0C{});
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorUb>;
        CopyL0CToDst copyL0CToDst;
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0cSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0cSlot]);
        copyL0CToDst(ubC, tensorL0C);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
    }

    AscendC::LocalTensor<ElementA> l1Q[Ascend950FagL1Layout::TASK_PINGPONG];
    AscendC::LocalTensor<ElementA> l1Dy[Ascend950FagL1Layout::TASK_PINGPONG];
    AscendC::LocalTensor<ElementB> l1Scratch0;
    AscendC::LocalTensor<ElementB> l1Scratch1;
    AscendC::LocalTensor<ElementA> l0ATensorList[L0AB_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[L0AB_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];
    int32_t l0AEventList[L0AB_STAGES];
    int32_t l0BEventList[L0AB_STAGES];
    int32_t l0CEventList[L0C_STAGES];
    int32_t l1EventList[Ascend950FagL1Layout::SLOT_COUNT];
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};
    uint32_t l0CListId{0};
    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
};

} // namespace Catlass::Gemm::Block

#endif
