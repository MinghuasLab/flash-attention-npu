/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef FLASH_ATTN_NPU_ASCEND950_V4_FAG_MMAD_MLA_SDP_HPP
#define FLASH_ATTN_NPU_ASCEND950_V4_FAG_MMAD_MLA_SDP_HPP

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

#include "fag_block_mla.h"

namespace Catlass::Gemm::Block {

template <uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, class L1TileShape_, class L0TileShape_, class ElementA_,
          class ElementB_, class ElementC_, class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadAscend950FagMlaSdP<L1A_STAGES_, L1B_STAGES_>, L1TileShape_, L0TileShape_, ElementA_, ElementB_,
                    ElementC_, ElementBias_, TileCopy_, TileMmad_> {
  public:
    using DispatchPolicy = MmadAscend950FagMlaSdP<L1A_STAGES_, L1B_STAGES_>;
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

    static constexpr uint32_t SLOT_S = Ascend950FagMlaL0CLayout::SLOT_SDP_0;
    static constexpr uint32_t SLOT_DP = Ascend950FagMlaL0CLayout::SLOT_SDP_0;
    static_assert(L0C_STAGES == Ascend950FagMlaL0CLayout::L0C_SDP_SLOT_NUM, "SdP must use exactly 1 L0C slot");

    static constexpr uint32_t L1_SLOT_QK_BYTES = BASE * Ascend950FagMlaL1Layout::QK_HEAD_DIM * sizeof(ElementA);
    static constexpr uint32_t L1_SLOT_VV_BYTES = BASE * BASE * sizeof(ElementA);

    static constexpr uint32_t L1SlotBytes(uint32_t slot)
    {
        if (slot == Ascend950FagMlaL1Layout::SLOT_RES_KT0 || slot == Ascend950FagMlaL1Layout::SLOT_RES_KT1 ||
            slot == Ascend950FagMlaL1Layout::SLOT_Q0 || slot == Ascend950FagMlaL1Layout::SLOT_Q1 ||
            slot == Ascend950FagMlaL1Layout::SLOT_Q2) {
            return L1_SLOT_QK_BYTES;
        }
        return L1_SLOT_VV_BYTES;
    }

    static constexpr uint32_t L1SlotOffset(uint32_t slot)
    {
        uint32_t off = 0;
        for (uint32_t i = Ascend950FagMlaL1Layout::SLOT_RES_KT0; i < slot; ++i) {
            off += L1SlotBytes(i);
        }
        return off;
    }

    static constexpr uint32_t L1_KT_OFFSET = L1SlotOffset(Ascend950FagMlaL1Layout::SLOT_RES_KT0);
    static constexpr uint32_t L1_VT_OFFSET = L1SlotOffset(Ascend950FagMlaL1Layout::SLOT_RES_VT);
    static constexpr uint32_t L1_EVENT_KT = Ascend950FagMlaL1Layout::SLOT_RES_KT0;
    static constexpr uint32_t L1_EVENT_VT = Ascend950FagMlaL1Layout::SLOT_RES_VT;
    // P/dS (2*2*128*128*2) + 6 cube slots must fit in 512KB L1.
    static_assert(L1SlotOffset(Ascend950FagMlaL1Layout::SLOT_COUNT) +
                          2 * Ascend950FagMlaL1Layout::TASK_PINGPONG * BASE * BASE * sizeof(ElementA) <=
                      ArchTag::L1_SIZE,
                  "MLA L1 overflow");
    static_assert(Ascend950FagMlaL0CLayout::L0COffset(Ascend950FagMlaL0CLayout::L0C_SLOT_NUM - 1) +
                          Ascend950FagMlaL0CLayout::L0CSize(Ascend950FagMlaL0CLayout::L0C_SLOT_NUM - 1) <=
                      ArchTag::L0C_SIZE,
                  "L0C overflow");

    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag>& resource, Ascend950FagMlaL0ABCursor& cursor, uint32_t l1BufAddrStart = 0,
                 uint32_t eventIdStart = 0)
        : l0AListId(cursor.a), l0BListId(cursor.b)
    {
        AscendC::SetMMLayoutTransform(true);
        AscendC::SetHF32Mode(false);

        for (uint32_t i = 0; i < 2; ++i) {
            l1KT[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1BufAddrStart + L1SlotOffset(Ascend950FagMlaL1Layout::SLOT_RES_KT0 + i));
        }
        l1VT = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_VT_OFFSET);
        for (uint32_t i = 0; i < Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH; ++i) {
            l1Q[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart +
                                                                       L1SlotOffset(Ascend950FagMlaL1Layout::QSlot(i)));
            l1Dy[i] = resource.l1Buf.template GetBufferByByte<ElementA>(
                l1BufAddrStart + L1SlotOffset(Ascend950FagMlaL1Layout::DySlot(i)));
        }

        for (uint32_t i = 0; i < L0AB_STAGES; i++) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0AMutexList[i] = static_cast<int32_t>(Ascend950FagMlaL1Layout::L0AMutex(i));
            l0BMutexList[i] = static_cast<int32_t>(Ascend950FagMlaL1Layout::L0BMutex(i));
        }
        for (uint32_t i = 0; i < L0C_STAGES; i++) {
            const uint32_t slot = Ascend950FagMlaL0CLayout::SLOT_SDP_0 + i;
            l0CTensorList[i] =
                resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(Ascend950FagMlaL0CLayout::L0COffset(slot));
        }
        for (uint32_t i = 0; i < Ascend950FagMlaL1Layout::SLOT_COUNT; i++) {
            l1MutexList[i] = static_cast<int32_t>(Ascend950FagMlaL1Layout::L1Mutex(i));
        }
    }

    CATLASS_DEVICE
    ~BlockMmadTla() {}

    /**
     * S = Q * K^T. K^T stays resident across s1 of the same s2 group when
     * loadK == false. Q lands in Q[taskId % QDY_BUFFER_DEPTH] and is kept for C34 (releaseA=false).
     * When prefetchV is set, V^T is issued on MTE2 before the S gemm so it can
     * overlap with cube; ComputeDP should then use loadV=false, waitV=true.
     * If loadQ == false, Q has already been prefetched into its L1 slot.
     */
    template <class TensorA, class TensorB, class TensorC, class TensorV>
    CATLASS_DEVICE void ComputeS(TensorA& tensorA, TensorB& tensorB, TensorC& tensorC, GemmCoord const& actualShape,
                                 bool loadK, uint32_t taskId, bool loadQ, bool prefetchV, TensorV& tensorV,
                                 uint32_t vSkvActual, uint32_t vDActual, uint32_t residentKSlot = 0)
    {
        uint32_t sqActual = actualShape.m();
        uint32_t skvActual = actualShape.n();
        uint32_t dActual = actualShape.k();

        uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        uint32_t skvRound = RoundUp<L1BAlignHelper::N_ALIGNED>(skvActual);
        uint32_t dRound = RoundUp<L1AAlignHelper::K_ALIGNED>(dActual);

        const uint32_t qSlot = Ascend950FagMlaL1Layout::QSlot(taskId);
        const uint32_t qEvent = qSlot;
        const uint32_t ktEvent = L1_EVENT_KT + residentKSlot;
        constexpr uint32_t slotC = 0;
        if (loadQ) {
            CopyGmToL1A(l1Q[qSlot % Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH], tensorA, sqActual, dActual, sqRound,
                        dRound, qEvent);
        }
        if (loadK) {
            CopyGmToL1BT(l1KT[residentKSlot], tensorB, dActual, skvActual, dRound, skvRound, ktEvent);
        }
        if (prefetchV) {
            const uint32_t vDRound = RoundUp<L1AAlignHelper::K_ALIGNED>(vDActual);
            const uint32_t vSkvRound = RoundUp<L1BAlignHelper::N_ALIGNED>(vSkvActual);
            CopyGmToL1BT(l1VT, tensorV, vDActual, vSkvActual, vDRound, vSkvRound, L1_EVENT_VT);
        }
        // Keep Q in L1 for C34; do not release A back to MTE2.
        GemmABt(l1Q[qSlot % Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH], l1KT[residentKSlot], l0CTensorList[slotC],
                sqRound, skvRound, dRound, sqActual, skvActual, dActual, slotC, qEvent, ktEvent,
                /*waitB=*/loadK, /*releaseB=*/false, /*releaseA=*/false);
        FixpipeUb(tensorC, l0CTensorList[slotC], sqActual, skvActual, sqRound, slotC);
    }

    /**
     * dP = dY * V^T. V^T stays resident when loadV == false.
     * dY lands in DY[taskId % QDY_BUFFER_DEPTH] and is kept for C5 (releaseA=false).
     * waitV must be true on the first use after a VT prefetch/load.
     * If loadDY == false, dY has already been prefetched into its L1 slot.
     */
    template <class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE void ComputeDP(TensorA& tensorA, TensorB& tensorB, TensorC& tensorC, GemmCoord const& actualShape,
                                  bool loadV, bool waitV, uint32_t taskId, bool loadDY)
    {
        uint32_t sqActual = actualShape.m();
        uint32_t skvActual = actualShape.n();
        uint32_t dActual = actualShape.k();

        uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        uint32_t skvRound = RoundUp<L1BAlignHelper::N_ALIGNED>(skvActual);
        uint32_t dRound = RoundUp<L1AAlignHelper::K_ALIGNED>(dActual);

        const uint32_t dySlot = Ascend950FagMlaL1Layout::DySlot(taskId);
        const uint32_t dyEvent = dySlot;
        constexpr uint32_t slotC = 0;
        if (loadDY) {
            CopyGmToL1A(l1Dy[dySlot % Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH], tensorA, sqActual, dActual, sqRound,
                        dRound, dyEvent);
        }
        if (loadV) {
            CopyGmToL1BT(l1VT, tensorB, dActual, skvActual, dRound, skvRound, L1_EVENT_VT);
        }
        GemmABt(l1Dy[dySlot % Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH], l1VT, l0CTensorList[slotC], sqRound, skvRound,
                dRound, sqActual, skvActual, dActual, slotC, dyEvent, L1_EVENT_VT,
                /*waitB=*/waitV, /*releaseB=*/false, /*releaseA=*/false);
        FixpipeUb(tensorC, l0CTensorList[slotC], sqActual, skvActual, sqRound, slotC);
    }

    /**
     * Prefetch Q for a future task into its L1 slot. The caller must ensure
     * the slot has been released by the task that used it three tasks earlier.
     */
    template <class TensorA>
    CATLASS_DEVICE void PrefetchQ(TensorA& tensorA, GemmCoord const& actualShape, uint32_t taskId)
    {
        uint32_t sqActual = actualShape.m();
        uint32_t dActual = actualShape.k();
        uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        uint32_t dRound = RoundUp<L1AAlignHelper::K_ALIGNED>(dActual);

        const uint32_t qSlot = Ascend950FagMlaL1Layout::QSlot(taskId);
        const uint32_t qEvent = qSlot;
        CopyGmToL1A(l1Q[qSlot % Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH], tensorA, sqActual, dActual, sqRound, dRound,
                    qEvent);
    }

    /**
     * Prefetch dY for a future task into its L1 slot.
     */
    template <class TensorA>
    CATLASS_DEVICE void PrefetchDY(TensorA& tensorA, GemmCoord const& actualShape, uint32_t taskId)
    {
        uint32_t sqActual = actualShape.m();
        uint32_t dActual = actualShape.k();
        uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        uint32_t dRound = RoundUp<L1AAlignHelper::K_ALIGNED>(dActual);

        const uint32_t dySlot = Ascend950FagMlaL1Layout::DySlot(taskId);
        const uint32_t dyEvent = dySlot;
        CopyGmToL1A(l1Dy[dySlot % Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH], tensorA, sqActual, dActual, sqRound,
                    dRound, dyEvent);
    }

    CATLASS_DEVICE
    void ReleaseResidentKT(uint32_t residentKSlot = 0)
    {
        AscendC::Mutex::Unlock<PIPE_MTE1>(l1MutexList[Ascend950FagMlaL1Layout::SLOT_RES_KT0 + residentKSlot]);
    }

    CATLASS_DEVICE
    void ReleaseResidentVT()
    {
        AscendC::Mutex::Unlock<PIPE_MTE1>(l1MutexList[Ascend950FagMlaL1Layout::SLOT_RES_VT]);
    }

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

        AscendC::Mutex::Lock<PIPE_MTE2>(l1MutexList[eventIdx]);
        copyGmToL1A(tensorL1, tensorTileGm);
        AscendC::Mutex::Unlock<PIPE_MTE2>(l1MutexList[eventIdx]);
    }

    /// Reload RowMajor GM [Skv, D] as ColumnMajor [D, Skv] (K^T / V^T) into L1 (nZ).
    template <class TensorGm>
    CATLASS_DEVICE void CopyGmToL1BT(AscendC::LocalTensor<ElementB> l1Buf, TensorGm& gm, uint32_t dActual,
                                     uint32_t skvActual, uint32_t dRound, uint32_t skvRound, uint32_t eventIdx)
    {
        auto layoutCol = tla::MakeLayout(tla::MakeShape(dActual, skvActual),
                                         tla::MakeStride(tla::Int<1>{}, static_cast<int64_t>(tla::get<0>(gm.stride()))),
                                         tla::MakeShape(dActual, skvActual));
        auto gmCol = tla::MakeTensor(gm.data(), layoutCol, Arch::PositionGM{});

        using CopyGmToL1BOp = typename TileCopy::template CopyGmToL1B<decltype(gmCol)>;
        CopyGmToL1BOp copyGmToL1B;

        auto layoutL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(dRound, skvRound);
        auto tensorL1 = tla::MakeTensor(l1Buf, layoutL1, Arch::PositionL1{});
        auto tensorTileGm = GetTile(gmCol, tla::MakeCoord(0u, 0u), tla::MakeShape(dActual, skvActual));

        AscendC::Mutex::Lock<PIPE_MTE2>(l1MutexList[eventIdx]);
        copyGmToL1B(tensorL1, tensorTileGm);
        AscendC::Mutex::Unlock<PIPE_MTE2>(l1MutexList[eventIdx]);
    }

    CATLASS_DEVICE
    void GemmABt(AscendC::LocalTensor<ElementA> l1A, AscendC::LocalTensor<ElementB> l1B,
                 AscendC::LocalTensor<ElementAccumulator> l0C, uint32_t mRound, uint32_t nRound, uint32_t kRound,
                 uint32_t mActual, uint32_t nActual, uint32_t kActual, uint32_t l0cSlot, uint32_t l1AEvent,
                 uint32_t l1BEvent, bool waitB, bool releaseB, bool releaseA = true)
    {
        (void)l0cSlot;
        auto layoutAInL1 = tla::MakeLayout<ElementA, LayoutTagL1A>(mRound, kRound);
        auto layoutBInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(kRound, nRound);
        auto tensorL1A = tla::MakeTensor(l1A, layoutAInL1, Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(l1B, layoutBInL1, Arch::PositionL1{});
        auto layoutCInL0 = tla::MakeLayoutL0C(mRound, nRound);
        auto tensorL0C = tla::MakeTensor(l0C, layoutCInL0, Arch::PositionL0C{});

        AscendC::Mutex::Lock<PIPE_MTE1>(l1MutexList[l1AEvent]);
        if (waitB) {
            AscendC::Mutex::Lock<PIPE_MTE1>(l1MutexList[l1BEvent]);
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

            AscendC::Mutex::Lock<PIPE_MTE1>(l0AMutexList[l0AListId]);
            copyL1ToL0A(tensorL0A, tileL1A);
            AscendC::Mutex::Unlock<PIPE_MTE1>(l0AMutexList[l0AListId]);

            AscendC::Mutex::Lock<PIPE_MTE1>(l0BMutexList[l0BListId]);
            copyL1ToL0B(tensorL0B, tileL1B);
            AscendC::Mutex::Unlock<PIPE_MTE1>(l0BMutexList[l0BListId]);

            AscendC::Mutex::Lock<PIPE_M>(l0AMutexList[l0AListId]);
            AscendC::Mutex::Lock<PIPE_M>(l0BMutexList[l0BListId]);

            bool initC = (kIdx == 0);
            uint32_t mMad = (mActual == 1) ? 2 : mRound;
            uint8_t unitFlag = (kIdx + 1 == kLoops) ? 0b11 : 0b10;
            tileMmad(tensorL0C, tensorL0A, tensorL0B, mMad, nActual, tileK, initC, unitFlag);

            AscendC::Mutex::Unlock<PIPE_M>(l0AMutexList[l0AListId]);
            AscendC::Mutex::Unlock<PIPE_M>(l0BMutexList[l0BListId]);
            l0AListId = (l0AListId + 1 < L0AB_STAGES) ? (l0AListId + 1) : 0;
            l0BListId = (l0BListId + 1 < L0AB_STAGES) ? (l0BListId + 1) : 0;
        }

        if (releaseA) {
            AscendC::Mutex::Unlock<PIPE_MTE1>(l1MutexList[l1AEvent]);
        }
        if (releaseB) {
            AscendC::Mutex::Unlock<PIPE_MTE1>(l1MutexList[l1BEvent]);
        }
    }

    template <class TensorUb>
    CATLASS_DEVICE void FixpipeUb(TensorUb& ubC, AscendC::LocalTensor<ElementAccumulator> l0C, uint32_t mActual,
                                  uint32_t nActual, uint32_t mRound, uint32_t l0cSlot)
    {
        (void)l0cSlot;
        auto layoutCInL0 = tla::MakeLayoutL0C(mRound, nActual);
        auto tensorL0C = tla::MakeTensor(l0C, layoutCInL0, Arch::PositionL0C{});

        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorUb>;
        CopyL0CToDst copyL0CToDst;

        copyL0CToDst(ubC, tensorL0C, 0b11);
    }

    AscendC::LocalTensor<ElementA> l1Q[Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH];
    AscendC::LocalTensor<ElementB> l1KT[2];
    AscendC::LocalTensor<ElementA> l1Dy[Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH];
    AscendC::LocalTensor<ElementB> l1VT;

    AscendC::LocalTensor<ElementA> l0ATensorList[L0AB_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[L0AB_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];

    int32_t l0AMutexList[L0AB_STAGES];
    int32_t l0BMutexList[L0AB_STAGES];
    int32_t l1MutexList[Ascend950FagMlaL1Layout::SLOT_COUNT];

    // References to the per-AIC cursor shared with the other GEMM block.
    uint32_t& l0AListId;
    uint32_t& l0BListId;

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
};

} // namespace Catlass::Gemm::Block

#endif // FLASH_ATTN_NPU_ASCEND950_V4_FAG_MMAD_MLA_SDP_HPP
