/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
struct BlockMmadTla <
    MmadAscend950FagdQKV<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>,
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
    using TileCopyCol = Gemm::Tile::PackedTileCopyTla<
        ArchTag, ElementA, layout::ColumnMajor, ElementB, layout::RowMajor, ElementC, layout::RowMajor>;
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
    static constexpr uint32_t L1_KT_OFFSET =
        Ascend950FagL1Layout::SLOT_RES_KT * L1_TILE_MAX;
    // RowMajor K packs into the upper half of the RES_KT 64KB slot when D<=BASE
    // (lower half holds K^T for C1). D>BASE keeps streaming K via DY scratch.
    static constexpr uint32_t L1_K_OFFSET = L1_KT_OFFSET + L1_TILE_HALF;
    static constexpr uint32_t L1_EVENT_K = Ascend950FagL1Layout::L1_EVENT_K;
    static_assert(
        Ascend950FagL1Layout::SLOT_COUNT * L1_TILE_MAX +
            2 * Ascend950FagL1Layout::TASK_PINGPONG * BASE * BASE * sizeof(ElementA) <=
            ArchTag::L1_SIZE,
        "L1 overflow");
    static_assert(L0C_STAGES * L0C_BUF_SIZE <= ArchTag::L0C_SIZE, "L0C overflow");
    static_assert(L1_TILE_HALF * 2 == L1_TILE_MAX, "RES_K half-slot packing assumes 64KB tiles");

    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0, uint32_t eventIdStart = 0)
    {
        l1K = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_K_OFFSET);
        for (uint32_t i = 0; i < Ascend950FagL1Layout::TASK_PINGPONG; ++i) {
            l1Q[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1BufAddrStart + Ascend950FagL1Layout::QSlot(i) * L1_TILE_MAX);
            l1Dy[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1BufAddrStart + Ascend950FagL1Layout::DySlot(i) * L1_TILE_MAX);
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

    /// Prefetch RowMajor K into the RES_KT upper half (D<=BASE only).
    /// Prefer calling from C5 so MTE2 overlaps dv cube work.
    template <class TensorK>
    CATLASS_DEVICE
    void LoadResidentK(TensorK& k, uint32_t skvActual, uint32_t dActual)
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
    CATLASS_DEVICE
    void ComputeDqDk(
        AscendC::LocalTensor<ElementA> dsNz,
        TensorK& k,
        TensorDq& dq, TensorDk& dk,
        GemmCoord const& actualShape,
        uint32_t taskPing, bool waitResidentK, bool enAtomicDq,
        bool initDk, bool flushDk, bool enAtomicDk)
    {
        const uint32_t sqActual = actualShape.m();
        const uint32_t dActual = actualShape.n();
        const uint32_t skvActual = actualShape.k();
        if (dActual <= BASE) {
            ComputeDqDkSmallHeadDim(dsNz, dq, dk,
                sqActual, skvActual, dActual, taskPing, waitResidentK, enAtomicDq,
                initDk, flushDk, enAtomicDk);
        } else {
            ComputeDqDkLargeHeadDim(dsNz, k, dq, dk,
                sqActual, skvActual, dActual, taskPing, enAtomicDq);
        }
    }

    /**
     * dv = P^T * dY. dY is already in L1 DY[taskPing] from C2.
     * After this stage the DY slot is released (D>BASE C34 may reuse as K scratch).
     */
    template <class TensorDv>
    CATLASS_DEVICE
    void ComputeDv(
        AscendC::LocalTensor<ElementA> pNz,
        TensorDv& dv,
        GemmCoord const& actualShape,
        uint32_t taskPing, bool initDv, bool flushDv, bool enAtomicDv)
    {
        const uint32_t sqActual = actualShape.m();
        const uint32_t dvActual = actualShape.n();
        const uint32_t skvActual = actualShape.k();
        if (dvActual <= BASE) {
            ComputeDvSmallHeadDim(pNz, dv,
                sqActual, skvActual, dvActual, taskPing, initDv, flushDv, enAtomicDv);
        } else {
            ComputeDvLargeHeadDim(pNz, dv,
                sqActual, skvActual, dvActual, taskPing);
        }
    }

protected:
    template <class TensorDq, class TensorDk>
    CATLASS_DEVICE
    void ComputeDqDkSmallHeadDim(
        AscendC::LocalTensor<ElementA> dsNz,
        TensorDq& dq, TensorDk& dk,
        uint32_t sqActual, uint32_t skvActual, uint32_t dActual,
        uint32_t taskPing, bool waitResidentK, bool enAtomicDq,
        bool initDk, bool flushDk, bool enAtomicDk)
    {
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dActual);
        const uint32_t qEvent = Ascend950FagL1Layout::QSlot(taskPing);
        // K resident in RES_KT upper half — no per-task GM load.
        GemmRow(dsNz, l1K, l0CTensorList[SLOT_DQ],
            sqRound, dRound, skvRound, sqActual, dActual, skvActual,
            true, 0b11, SLOT_DQ, L1_EVENT_K, /*waitB=*/waitResidentK, /*releaseB=*/false,
            /*useKEvent=*/true);
        FixpipeTla(dq, l0CTensorList[SLOT_DQ], sqActual, dActual, sqRound, enAtomicDq, SLOT_DQ);
        GemmColA(dsNz, l1Q[taskPing], l0CTensorList[SLOT_DK],
            skvRound, dRound, sqRound, skvActual, dActual, sqActual,
            initDk, 0b11, SLOT_DK, qEvent, /*waitB=*/false, /*releaseB=*/true);
        if (flushDk) {
            FixpipeTla(dk, l0CTensorList[SLOT_DK], skvActual, dActual, skvRound, enAtomicDk, SLOT_DK);
            ReleaseResidentK();
        }
    }

    template <class TensorDv>
    CATLASS_DEVICE
    void ComputeDvSmallHeadDim(
        AscendC::LocalTensor<ElementA> pNz, TensorDv& dv,
        uint32_t sqActual, uint32_t skvActual, uint32_t dvActual,
        uint32_t taskPing, bool initDv, bool flushDv, bool enAtomicDv)
    {
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dvRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dvActual);
        const uint32_t dyEvent = Ascend950FagL1Layout::DySlot(taskPing);
        GemmColA(pNz, l1Dy[taskPing], l0CTensorList[SLOT_DV],
            skvRound, dvRound, sqRound, skvActual, dvActual, sqActual,
            initDv, 0b11, SLOT_DV, dyEvent, /*waitB=*/false, /*releaseB=*/true);
        if (flushDv) {
            FixpipeTla(dv, l0CTensorList[SLOT_DV], skvActual, dvActual, skvRound, enAtomicDv, SLOT_DV);
        }
    }

    template <class TensorK, class TensorDq, class TensorDk>
    CATLASS_DEVICE
    void ComputeDqDkLargeHeadDim(
        AscendC::LocalTensor<ElementA> dsNz,
        TensorK& k, TensorDq& dq, TensorDk& dk,
        uint32_t sqActual, uint32_t skvActual, uint32_t dActual,
        uint32_t taskPing, bool enAtomicDq)
    {
        // D>BASE: Fixpipe each D-slice. Q is full in L1 from C1; K streams
        // slice-by-slice into the DY scratch slot.
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dFullRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dActual);
        const uint32_t qEvent = Ascend950FagL1Layout::QSlot(taskPing);
        const uint32_t dyEvent = Ascend950FagL1Layout::DySlot(taskPing);
        const uint32_t dLoops = CeilDiv(dActual, BASE);
        for (uint32_t dIdx = 0; dIdx < dLoops; ++dIdx) {
            const uint32_t nOff = dIdx * BASE;
            const uint32_t nAct = Min(BASE, dActual - nOff);
            const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nAct);
            const uint32_t slotDq = (dIdx & 1U) ? SLOT_DQ_PING : SLOT_DQ;
            const bool lastSlice = (dIdx + 1 == dLoops);
            auto kTile = GetTile(k, tla::MakeCoord(0u, nOff), tla::MakeShape(skvActual, nAct));
            auto dqTile = GetTile(dq, tla::MakeCoord(0u, nOff), tla::MakeShape(sqActual, nAct));
            auto dkTile = GetTile(dk, tla::MakeCoord(0u, nOff), tla::MakeShape(skvActual, nAct));
            CopyGmToL1B(l1Dy[taskPing], kTile, skvActual, nAct, skvRound, nRound, dyEvent);
            GemmRow(dsNz, l1Dy[taskPing], l0CTensorList[slotDq],
                sqRound, nRound, skvRound, sqActual, nAct, skvActual,
                true, 0b11, slotDq, dyEvent, /*waitB=*/true, /*releaseB=*/true);
            FixpipeTla(dqTile, l0CTensorList[slotDq],
                sqActual, nAct, sqRound, enAtomicDq, slotDq);
            GemmColA(dsNz, l1Q[taskPing], l0CTensorList[SLOT_DK],
                skvRound, nRound, sqRound, skvActual, nAct, sqActual,
                true, 0b11, SLOT_DK, qEvent,
                /*waitB=*/false, /*releaseB=*/lastSlice,
                /*bNOff=*/nOff, /*bNFull=*/dFullRound);
            FixpipeTla(dkTile, l0CTensorList[SLOT_DK],
                skvActual, nAct, skvRound, /*enAtomic=*/true, SLOT_DK);
        }
    }

    template <class TensorDv>
    CATLASS_DEVICE
    void ComputeDvLargeHeadDim(
        AscendC::LocalTensor<ElementA> pNz, TensorDv& dv,
        uint32_t sqActual, uint32_t skvActual, uint32_t dvActual,
        uint32_t taskPing)
    {
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dvFullRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dvActual);
        const uint32_t dyEvent = Ascend950FagL1Layout::DySlot(taskPing);
        const uint32_t dvLoops = CeilDiv(dvActual, BASE);
        for (uint32_t dIdx = 0; dIdx < dvLoops; ++dIdx) {
            const uint32_t nOff = dIdx * BASE;
            const uint32_t nAct = Min(BASE, dvActual - nOff);
            const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nAct);
            const bool lastSlice = (dIdx + 1 == dvLoops);
            auto dvTile = GetTile(dv, tla::MakeCoord(0u, nOff), tla::MakeShape(skvActual, nAct));
            GemmColA(pNz, l1Dy[taskPing], l0CTensorList[SLOT_DV],
                skvRound, nRound, sqRound, skvActual, nAct, sqActual,
                true, 0b11, SLOT_DV, dyEvent,
                /*waitB=*/false, /*releaseB=*/lastSlice,
                /*bNOff=*/nOff, /*bNFull=*/dvFullRound);
            FixpipeTla(dvTile, l0CTensorList[SLOT_DV],
                skvActual, nAct, skvRound, /*enAtomic=*/true, SLOT_DV);
        }
    }

    template <class TensorGm>
    CATLASS_DEVICE
    void CopyGmToL1B(
        AscendC::LocalTensor<ElementB> l1Buf, TensorGm& gm,
        uint32_t kActual, uint32_t nActual, uint32_t kRound, uint32_t nRound,
        uint32_t eventIdx, bool useKEvent = false)
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
    void GemmRow(
        AscendC::LocalTensor<ElementA> l1A,
        AscendC::LocalTensor<ElementB> l1B,
        AscendC::LocalTensor<ElementAccumulator> l0C,
        uint32_t mRound, uint32_t nRound, uint32_t kRound,
        uint32_t mActual, uint32_t nActual, uint32_t kActual,
        bool initC, uint8_t unitFlag, uint32_t l0cSlot,
        uint32_t l1BEvent, bool waitB, bool releaseB, bool useKEvent = false)
    {
        auto layoutAInL1 = tla::MakeLayout<ElementA, LayoutTagL1A>(mRound, kRound);
        auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mRound, kRound);
        auto layoutBInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(kRound, nRound);
        auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kRound, nRound);
        auto layoutCInL0 = tla::MakeLayoutL0C(mRound, nRound);

        auto tensorL1A = tla::MakeTensor(l1A, layoutAInL1, Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0ATensorList[l0AListId], layoutAInL0, Arch::PositionL0A{});
        auto tensorL1B = tla::MakeTensor(l1B, layoutBInL1, Arch::PositionL1{});
        auto tensorL0B = tla::MakeTensor(l0BTensorList[l0BListId], layoutBInL0, Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutCInL0, Arch::PositionL0C{});
        auto tileL1A = GetTile(tensorL1A, tla::MakeCoord(0u, 0u), tla::MakeShape(mRound, kRound));
        auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(0u, 0u), tla::MakeShape(kRound, nRound));

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
        if (initC) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
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
    void GemmColA(
        AscendC::LocalTensor<ElementA> l1A,
        AscendC::LocalTensor<ElementB> l1B,
        AscendC::LocalTensor<ElementAccumulator> l0C,
        uint32_t mRound, uint32_t nRound, uint32_t kRound,
        uint32_t mActual, uint32_t nActual, uint32_t kActual,
        bool initC, uint8_t unitFlag, uint32_t l0cSlot,
        uint32_t l1BEvent, bool waitB, bool releaseB,
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
        uint32_t mMad = (mActual == 1) ? 2 : mRound;
        uint8_t uf = ENABLE_UNIT_FLAG ? unitFlag : 0;
        tileMmadCol(tensorL0C, tensorL0A, tensorL0B, mMad, nActual, kActual, initC, uf);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
        l0AListId = (l0AListId + 1 < L0AB_STAGES) ? (l0AListId + 1) : 0;
        l0BListId = (l0BListId + 1 < L0AB_STAGES) ? (l0BListId + 1) : 0;
    }

    template <class TensorGm>
    CATLASS_DEVICE
    void FixpipeTla(
        TensorGm& gmC,
        AscendC::LocalTensor<ElementAccumulator> l0C,
        uint32_t mActual, uint32_t nActual, uint32_t mRound,
        bool enAtomic, uint32_t l0cSlot)
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
            if constexpr (ENABLE_UNIT_FLAG) {
                copyL0CToDst(tensorGmTile, tensorL0C, 0b11);
            } else {
                copyL0CToDst(tensorGmTile, tensorL0C);
            }
            AscendC::SetAtomicNone();
        } else {
            if constexpr (ENABLE_UNIT_FLAG) {
                copyL0CToDst(tensorGmTile, tensorL0C, 0b11);
            } else {
                copyL0CToDst(tensorGmTile, tensorL0C);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
    }

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
