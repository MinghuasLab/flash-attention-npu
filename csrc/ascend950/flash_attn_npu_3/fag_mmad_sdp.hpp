/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_MMAD_SDP_HPP
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_MMAD_SDP_HPP

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

template <
    uint32_t L1A_STAGES_,
    uint32_t L1B_STAGES_,
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmadTla<
    MmadAscend950FagSdP<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    ElementA_,
    ElementB_,
    ElementC_,
    ElementBias_,
    TileCopy_,
    TileMmad_
> {
public:
    using DispatchPolicy = MmadAscend950FagSdP<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>;
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

    static constexpr uint32_t SLOT_S = Ascend950FagL0CLayout::SLOT_SDP_0;
    static constexpr uint32_t SLOT_DP = Ascend950FagL0CLayout::SLOT_SDP_1;
    static_assert(L0C_STAGES == Ascend950FagL0CLayout::L0C_SDP_SLOT_NUM, "SdP must use exactly 2 L0C slots");
    static_assert(L0C_STAGES * L0C_BUF_SIZE <= ArchTag::L0C_SIZE / 2, "SdP L0C must stay in low 128KB");

    static constexpr uint32_t L1_TILE_MAX = BASE * 256 * sizeof(ElementA);
    static constexpr uint32_t L1_KT_OFFSET =
        Ascend950FagL1Layout::SLOT_RES_KT * L1_TILE_MAX;
    static constexpr uint32_t L1_VT_OFFSET =
        Ascend950FagL1Layout::SLOT_RES_VT * L1_TILE_MAX;
    static constexpr uint32_t L1_EVENT_KT = Ascend950FagL1Layout::SLOT_RES_KT;
    static constexpr uint32_t L1_EVENT_VT = Ascend950FagL1Layout::SLOT_RES_VT;
    // P/dS (2*2*128*128*2) + 6 cube slots must fit in 512KB L1.
    static_assert(
        Ascend950FagL1Layout::SLOT_COUNT * L1_TILE_MAX +
            2 * Ascend950FagL1Layout::TASK_PINGPONG * BASE * BASE * sizeof(ElementA) <=
            ArchTag::L1_SIZE,
        "L1 overflow");
    static_assert(Ascend950FagL0CLayout::L0C_SLOT_NUM * L0C_BUF_SIZE <= ArchTag::L0C_SIZE, "L0C overflow");

    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0, uint32_t eventIdStart = 0)
    {
        if constexpr (ENABLE_UNIT_FLAG) {
            AscendC::SetMMLayoutTransform(true);
        }
        AscendC::SetHF32Mode(false);

        l1KT = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_KT_OFFSET);
        l1VT = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_VT_OFFSET);
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

    CATLASS_DEVICE
    ~BlockMmadTla() {}

    /**
     * S = Q * K^T. K^T stays resident across s1 of the same s2 group when
     * loadK == false. Q lands in Q[taskPing] and is kept for C34 (releaseA=false).
     * When prefetchV is set, V^T is issued on MTE2 before the S gemm so it can
     * overlap with cube; ComputeDP should then use loadV=false, waitV=true.
     */
    template <class TensorA, class TensorB, class TensorC, class TensorV>
    CATLASS_DEVICE
    void ComputeS(TensorA& tensorA, TensorB& tensorB, TensorC& tensorC,
        GemmCoord const& actualShape, bool loadK, uint32_t taskPing,
        bool prefetchV, TensorV& tensorV, uint32_t vSkvActual, uint32_t vDActual)
    {
        uint32_t sqActual = actualShape.m();
        uint32_t skvActual = actualShape.n();
        uint32_t dActual = actualShape.k();

        uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        uint32_t skvRound = RoundUp<L1BAlignHelper::N_ALIGNED>(skvActual);
        uint32_t dRound = RoundUp<L1AAlignHelper::K_ALIGNED>(dActual);

        const uint32_t qEvent = Ascend950FagL1Layout::QSlot(taskPing);
        uint32_t slotC = l0CListId;
        CopyGmToL1A(l1Q[taskPing], tensorA, sqActual, dActual, sqRound, dRound, qEvent);
        if (loadK) {
            CopyGmToL1BT(l1KT, tensorB, dActual, skvActual, dRound, skvRound, L1_EVENT_KT);
        }
        if (prefetchV) {
            const uint32_t vDRound = RoundUp<L1AAlignHelper::K_ALIGNED>(vDActual);
            const uint32_t vSkvRound = RoundUp<L1BAlignHelper::N_ALIGNED>(vSkvActual);
            CopyGmToL1BT(l1VT, tensorV, vDActual, vSkvActual, vDRound, vSkvRound, L1_EVENT_VT);
        }
        // Keep Q in L1 for C34; do not release A back to MTE2.
        GemmABt(l1Q[taskPing], l1KT, l0CTensorList[slotC],
            sqRound, skvRound, dRound, sqActual, skvActual, dActual,
            slotC, qEvent, L1_EVENT_KT,
            /*waitB=*/loadK, /*releaseB=*/false, /*releaseA=*/false);
        FixpipeUb(tensorC, l0CTensorList[slotC], sqActual, skvActual, sqRound, slotC);
        l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
    }

    /**
     * dP = dY * V^T. V^T stays resident when loadV == false.
     * dY lands in DY[taskPing] and is kept for C5 (releaseA=false).
     * waitV must be true on the first use after a VT prefetch/load.
     */
    template <class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE
    void ComputeDP(TensorA& tensorA, TensorB& tensorB, TensorC& tensorC,
        GemmCoord const& actualShape, bool loadV, bool waitV, uint32_t taskPing)
    {
        uint32_t sqActual = actualShape.m();
        uint32_t skvActual = actualShape.n();
        uint32_t dActual = actualShape.k();

        uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        uint32_t skvRound = RoundUp<L1BAlignHelper::N_ALIGNED>(skvActual);
        uint32_t dRound = RoundUp<L1AAlignHelper::K_ALIGNED>(dActual);

        const uint32_t dyEvent = Ascend950FagL1Layout::DySlot(taskPing);
        uint32_t slotC = l0CListId;
        CopyGmToL1A(l1Dy[taskPing], tensorA, sqActual, dActual, sqRound, dRound, dyEvent);
        if (loadV) {
            CopyGmToL1BT(l1VT, tensorB, dActual, skvActual, dRound, skvRound, L1_EVENT_VT);
        }
        GemmABt(l1Dy[taskPing], l1VT, l0CTensorList[slotC],
            sqRound, skvRound, dRound, sqActual, skvActual, dActual,
            slotC, dyEvent, L1_EVENT_VT,
            /*waitB=*/waitV, /*releaseB=*/false, /*releaseA=*/false);
        FixpipeUb(tensorC, l0CTensorList[slotC], sqActual, skvActual, sqRound, slotC);
        l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
    }

    CATLASS_DEVICE
    void ReleaseResidentKT()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[L1_EVENT_KT]);
    }

    CATLASS_DEVICE
    void ReleaseResidentVT()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[L1_EVENT_VT]);
    }

    // Backward-compatible entry: always reload both operands.
    template <class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE
    void operator()(TensorA& tensorA, TensorB& tensorB, TensorC& tensorC, GemmCoord const& actualShape)
    {
        ComputeS(tensorA, tensorB, tensorC, actualShape, true, /*taskPing=*/0,
            /*prefetchV=*/false, tensorB, actualShape.n(), actualShape.k());
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
            l1EventList[Ascend950FagL1Layout::QSlot(0)]);
        ReleaseResidentKT();
    }

protected:
    template <class TensorGm>
    CATLASS_DEVICE
    void CopyGmToL1A(
        AscendC::LocalTensor<ElementA> l1Buf, TensorGm& gm,
        uint32_t mActual, uint32_t kActual, uint32_t mRound, uint32_t kRound, uint32_t eventIdx)
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

    /// Reload RowMajor GM [Skv, D] as ColumnMajor [D, Skv] (K^T / V^T) into L1 (nZ).
    template <class TensorGm>
    CATLASS_DEVICE
    void CopyGmToL1BT(
        AscendC::LocalTensor<ElementB> l1Buf, TensorGm& gm,
        uint32_t dActual, uint32_t skvActual, uint32_t dRound, uint32_t skvRound, uint32_t eventIdx)
    {
        auto layoutCol = tla::MakeLayout(
            tla::MakeShape(dActual, skvActual),
            tla::MakeStride(
                tla::Int<1>{},
                static_cast<int64_t>(tla::get<0>(gm.stride()))),
            tla::MakeShape(dActual, skvActual));
        auto gmCol = tla::MakeTensor(
            gm.data(), layoutCol, Arch::PositionGM{});

        using CopyGmToL1BOp = typename TileCopy::template CopyGmToL1B<decltype(gmCol)>;
        CopyGmToL1BOp copyGmToL1B;

        auto layoutL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(dRound, skvRound);
        auto tensorL1 = tla::MakeTensor(l1Buf, layoutL1, Arch::PositionL1{});
        auto tensorTileGm = GetTile(gmCol, tla::MakeCoord(0u, 0u), tla::MakeShape(dActual, skvActual));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[eventIdx]);
        copyGmToL1B(tensorL1, tensorTileGm);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[eventIdx]);
    }

    CATLASS_DEVICE
    void GemmABt(
        AscendC::LocalTensor<ElementA> l1A,
        AscendC::LocalTensor<ElementB> l1B,
        AscendC::LocalTensor<ElementAccumulator> l0C,
        uint32_t mRound, uint32_t nRound, uint32_t kRound,
        uint32_t mActual, uint32_t nActual, uint32_t kActual,
        uint32_t l0cSlot, uint32_t l1AEvent, uint32_t l1BEvent,
        bool waitB, bool releaseB, bool releaseA = true)
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
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
        }

        uint32_t kLoops = CeilDiv(kActual, BASE);
        for (uint32_t kIdx = 0; kIdx < kLoops; ++kIdx) {
            uint32_t kOff = kIdx * BASE;
            uint32_t tileK = (kIdx + 1 < kLoops) ? BASE : (kActual - kOff);
            uint32_t kTileRound = RoundUp<L1AAlignHelper::K_ALIGNED>(tileK);

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

            bool initC = (kIdx == 0);
            uint32_t mMad = (mActual == 1) ? 2 : mRound;
            uint8_t unitFlag = 0;
            if constexpr (ENABLE_UNIT_FLAG) {
                unitFlag = (kIdx + 1 == kLoops) ? 0b11 : 0b10;
            }
            tileMmad(tensorL0C, tensorL0A, tensorL0B, mMad, nActual, tileK, initC, unitFlag);

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
    CATLASS_DEVICE
    void FixpipeUb(
        TensorUb& ubC,
        AscendC::LocalTensor<ElementAccumulator> l0C,
        uint32_t mActual, uint32_t nActual, uint32_t mRound,
        uint32_t l0cSlot)
    {
        auto layoutCInL0 = tla::MakeLayoutL0C(mRound, nActual);
        auto tensorL0C = tla::MakeTensor(l0C, layoutCInL0, Arch::PositionL0C{});

        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorUb>;
        CopyL0CToDst copyL0CToDst;

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0cSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0cSlot]);
            copyL0CToDst(ubC, tensorL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
        } else {
            copyL0CToDst(ubC, tensorL0C, 0b11);
        }
    }

    AscendC::LocalTensor<ElementA> l1Q[Ascend950FagL1Layout::TASK_PINGPONG];
    AscendC::LocalTensor<ElementB> l1KT;
    AscendC::LocalTensor<ElementA> l1Dy[Ascend950FagL1Layout::TASK_PINGPONG];
    AscendC::LocalTensor<ElementB> l1VT;

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

#endif // FLASH_ATTN_NPU_ASCEND950_V3_FAG_MMAD_SDP_HPP
