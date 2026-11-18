/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef FLASH_ATTN_NPU_ASCEND950_V4_FAG_MMAD_MLA_DQKV_HPP
#define FLASH_ATTN_NPU_ASCEND950_V4_FAG_MMAD_MLA_DQKV_HPP

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

template <uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, bool ENABLE_UNIT_FLAG_, class L1TileShape_, class L0TileShape_,
          class ElementA_, class ElementB_, class ElementC_, class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadAscend950FagMladQKV<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>, L1TileShape_, L0TileShape_,
                    ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_, TileMmad_> {
  public:
    using DispatchPolicy = MmadAscend950FagMladQKV<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>;
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
    static constexpr uint16_t L1_FREE_FLAG_INVALID = 0xffff;
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

    static constexpr uint32_t SLOT_DQ = Ascend950FagMlaL0CLayout::SLOT_DQ;
    static constexpr uint32_t SLOT_DK = Ascend950FagMlaL0CLayout::SLOT_DK;
    static constexpr uint32_t SLOT_DV = Ascend950FagMlaL0CLayout::SLOT_DV;

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
    static constexpr uint32_t L1_K_OFFSET = L1_KT_OFFSET;
    static constexpr uint32_t L1_EVENT_K = Ascend950FagMlaL1Layout::SLOT_RES_KT0;
    static_assert(L1SlotOffset(Ascend950FagMlaL1Layout::SLOT_COUNT) +
                          2 * Ascend950FagMlaL1Layout::TASK_PINGPONG * BASE * BASE * sizeof(ElementA) <=
                      ArchTag::L1_SIZE,
                  "MLA L1 overflow");
    static_assert(Ascend950FagMlaL0CLayout::L0COffset(Ascend950FagMlaL0CLayout::L0C_SLOT_NUM - 1) +
                          Ascend950FagMlaL0CLayout::L0CSize(Ascend950FagMlaL0CLayout::L0C_SLOT_NUM - 1) <=
                      ArchTag::L0C_SIZE,
                  "dQKV L0C overflow");

    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag>& resource, Ascend950FagMlaL0ABCursor& cursor, uint32_t l1BufAddrStart = 0,
                 uint32_t eventIdStart = 0)
        : l0AListId(cursor.a), l0BListId(cursor.b)
    {
        for (uint32_t i = 0; i < 2; ++i) {
            l1KSlots[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1BufAddrStart + L1SlotOffset(Ascend950FagMlaL1Layout::SLOT_RES_KT0 + i));
        }
        l1K = l1KSlots[0];
        for (uint32_t i = 0; i < Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH; ++i) {
            l1Q[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart +
                                                                       L1SlotOffset(Ascend950FagMlaL1Layout::QSlot(i)));
            l1Dy[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
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
            l0CMutexList[i] = static_cast<int32_t>(Ascend950FagMlaL1Layout::L0CMutex(slot));
        }
        for (uint32_t i = 0; i < Ascend950FagMlaL1Layout::SLOT_COUNT; i++) {
            l1MutexList[i] = static_cast<int32_t>(Ascend950FagMlaL1Layout::L1Mutex(i));
        }
    }

    CATLASS_DEVICE
    ~BlockMmadTla() {}

    CATLASS_DEVICE
    void SelectResidentKSlot(uint32_t slot)
    {
        l1K = l1KSlots[slot];
        residentKSlot_ = slot;
    }

    /**
     * dq = dS * K, dk = dS^T * Q. MLA fixed qk=192: K is resident in L1
     * (loaded by C1); dQ takes D-slices from the resident K, dK accumulates
     * the full head dim in the DK L0C slot.
     */
    template <class TensorDq, class TensorDk>
    CATLASS_DEVICE void ComputeDqDk(AscendC::LocalTensor<ElementA> dsNz, TensorDq& dq, TensorDk& dk,
                                    GemmCoord const& actualShape, uint32_t taskId, bool enAtomicDq, bool initDk,
                                    bool flushDk, bool enAtomicDk, bool releaseKT = false,
                                    uint16_t l1AFreeFlag = L1_FREE_FLAG_INVALID)
    {
        const uint32_t sqActual = actualShape.m();
        const uint32_t skvActual = actualShape.k();
        ComputeDqDkMla(dsNz, dq, dk, sqActual, skvActual, Ascend950FagMlaL1Layout::QK_HEAD_DIM, taskId, enAtomicDq,
                       initDk, flushDk, enAtomicDk, releaseKT, l1AFreeFlag);
    }

    /**
     * dv = P^T * dY. MLA fixed dv=128: small-head-dim path.
     */
    template <class TensorDv>
    CATLASS_DEVICE void ComputeDv(AscendC::LocalTensor<ElementA> pNz, TensorDv& dv, GemmCoord const& actualShape,
                                  uint32_t taskId, bool initDv, bool flushDv, bool enAtomicDv,
                                  uint16_t l1AFreeFlag = L1_FREE_FLAG_INVALID)
    {
        const uint32_t sqActual = actualShape.m();
        const uint32_t skvActual = actualShape.k();
        ComputeDvSmallHeadDim(pNz, dv, sqActual, skvActual, Ascend950FagMlaL1Layout::V_HEAD_DIM, taskId, initDv,
                              flushDv, enAtomicDv, l1AFreeFlag);
    }

  protected:
    template <class TensorDv>
    CATLASS_DEVICE void ComputeDvSmallHeadDim(AscendC::LocalTensor<ElementA> pNz, TensorDv& dv, uint32_t sqActual,
                                              uint32_t skvActual, uint32_t dvActual, uint32_t taskId, bool initDv,
                                              bool flushDv, bool enAtomicDv,
                                              uint16_t l1AFreeFlag = L1_FREE_FLAG_INVALID)
    {
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dvRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dvActual);
        const uint32_t dySlot = Ascend950FagMlaL1Layout::DySlot(taskId);
        const uint32_t dyEvent = dySlot;
        GemmColA(pNz, l1Dy[dySlot % Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH], l0CTensorList[SLOT_DV], skvRound,
                 dvRound, sqRound, skvActual, dvActual, sqActual, initDv, flushDv ? 0b11 : 0b10, SLOT_DV, dyEvent,
                 /*waitB=*/false, /*releaseB=*/true,
                 /*bNOff=*/0, /*bNFull=*/0, /*loadA=*/true, /*releaseA=*/true,
                 /*acquireC=*/true, /*publishC=*/true, l1AFreeFlag);
        if (flushDv) {
            FixpipeTla(dv, l0CTensorList[SLOT_DV], skvActual, dvActual, skvRound, enAtomicDv, SLOT_DV);
        }
    }

    /**
     * MLA qk=192 special path: D-sliced dQ and full-head-dim dK accumulation.
     * The dedicated dK L0C slot is large enough for the 128-wide dK slice plus
     * the 64-wide tail placed at a column offset.  Both slices accumulate
     * across s1 blocks of the same s2 group and are flushed together at group
     * end. dQ is written once for the full head dimension.
     */
    template <class TensorDq, class TensorDk>
    CATLASS_DEVICE void ComputeDqDkMla(AscendC::LocalTensor<ElementA> dsNz, TensorDq& dq, TensorDk& dk,
                                       uint32_t sqActual, uint32_t skvActual, uint32_t dActual, uint32_t taskId,
                                       bool enAtomicDq, bool initDk, bool flushDk, bool enAtomicDk,
                                       bool releaseKT = false, uint16_t l1AFreeFlag = L1_FREE_FLAG_INVALID)
    {
        const uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        const uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        const uint32_t dFullRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dActual);
        const uint32_t qSlot = Ascend950FagMlaL1Layout::QSlot(taskId);
        const uint32_t qEvent = qSlot;
        const uint32_t dLoops = CeilDiv(dActual, BASE);

        // ---- dQ: read K from the resident L1 slot loaded by C1 ----
        // One producer/consumer interval for the full D=192 result, as in
        // CANN MatmulNMutex. Both N slices share the 96 KiB slot with C1/C2.
        for (uint32_t dIdx = 0; dIdx < dLoops; ++dIdx) {
            const uint32_t nOff = dIdx * BASE;
            const uint32_t nAct = Min(BASE, dActual - nOff);
            const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nAct);
            const bool firstSlice = dIdx == 0;
            const bool lastSlice = dIdx + 1 == dLoops;
            auto l1KSlice = l1K;
            l1KSlice.SetAddrWithOffset(l1KSlice, nOff * skvRound);
            auto dqL0C = l0CTensorList[SLOT_DQ];
            dqL0C.SetAddrWithOffset(dqL0C, nOff * sqRound);
            GemmRow(dsNz, l1KSlice, dqL0C, sqRound, nRound, skvRound, sqActual, nAct, skvActual, true, 0b11, SLOT_DQ,
                    L1_EVENT_K + residentKSlot_, /*waitB=*/false,
                    /*releaseB=*/lastSlice && releaseKT,
                    /*loadA=*/firstSlice, /*releaseA=*/lastSlice,
                    /*acquireC=*/firstSlice, /*publishC=*/lastSlice);
        }
        FixpipeTla(dq, l0CTensorList[SLOT_DQ], sqActual, dActual, sqRound, enAtomicDq, SLOT_DQ);

        // ---- dK: accumulate both D-slices in the dedicated dK slot ----
        const uint32_t dkTailOffset = (BASE / Catlass::C0_NUM_PER_FRACTAL) *
                                      RoundUp<Catlass::C0_NUM_PER_FRACTAL>(skvRound) * Catlass::C0_NUM_PER_FRACTAL;
        for (uint32_t dIdx = 0; dIdx < dLoops; ++dIdx) {
            const uint32_t nOff = dIdx * BASE;
            const uint32_t nAct = Min(BASE, dActual - nOff);
            const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nAct);
            const bool lastSlice = (dIdx + 1 == dLoops);
            AscendC::LocalTensor<ElementAccumulator> dkL0C = l0CTensorList[SLOT_DK];
            if (dIdx != 0) {
                dkL0C.SetAddrWithOffset(dkL0C, dkTailOffset);
            }
            // Zero the full dK slot on the first s1 block only; otherwise accumulate.
            const bool initC = initDk;
            GemmColA(dsNz, l1Q[qSlot % Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH], dkL0C, skvRound, nRound, sqRound,
                     skvActual, nAct, sqActual, initC, flushDk ? 0b11 : 0b10, SLOT_DK, qEvent,
                     /*waitB=*/false, /*releaseB=*/lastSlice,
                     /*bNOff=*/nOff, /*bNFull=*/dFullRound,
                     /*loadA=*/dIdx == 0, /*releaseA=*/lastSlice,
                     /*acquireC=*/dIdx == 0, /*publishC=*/lastSlice, l1AFreeFlag);
        }
        if (flushDk) {
            FixpipeTla(dk, l0CTensorList[SLOT_DK], skvActual, dActual, skvRound, enAtomicDk, SLOT_DK);
        }
    }

    CATLASS_DEVICE
    // MLA's 128+64 N slices share A. Hold its consumer ownership until the
    // final slice.
    void GemmRow(AscendC::LocalTensor<ElementA> l1A, AscendC::LocalTensor<ElementB> l1B,
                 AscendC::LocalTensor<ElementAccumulator> l0C, uint32_t mRound, uint32_t nRound, uint32_t kRound,
                 uint32_t mActual, uint32_t nActual, uint32_t kActual, bool initC, uint8_t unitFlag, uint32_t l0cSlot,
                 uint32_t l1BEvent, bool waitB, bool releaseB, bool loadA = true, bool releaseA = true,
                 bool acquireC = true, bool publishC = true)
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

        const int32_t mtxB = l1MutexList[l1BEvent];

        if (loadA) {
            AscendC::Mutex::Lock<PIPE_MTE1>(l0AMutexList[l0AListId]);
            copyL1ToL0A(tensorL0A, tileL1A);
            AscendC::Mutex::Unlock<PIPE_MTE1>(l0AMutexList[l0AListId]);
        }

        AscendC::Mutex::Lock<PIPE_MTE1>(l0BMutexList[l0BListId]);
        if (waitB) {
            AscendC::Mutex::Lock<PIPE_MTE1>(mtxB);
        }
        copyL1ToL0B(tensorL0B, tileL1B);
        if (releaseB) {
            AscendC::Mutex::Unlock<PIPE_MTE1>(mtxB);
        }
        AscendC::Mutex::Unlock<PIPE_MTE1>(l0BMutexList[l0BListId]);

        if (loadA) {
            AscendC::Mutex::Lock<PIPE_M>(l0AMutexList[l0AListId]);
        }
        AscendC::Mutex::Lock<PIPE_M>(l0BMutexList[l0BListId]);
        if (!ENABLE_UNIT_FLAG && acquireC) {
            AscendC::Mutex::Lock<PIPE_M>(l0CMutexList[l0cSlot]);
        }
        uint32_t mMad = (mActual == 1) ? 2 : mRound;
        uint8_t uf = ENABLE_UNIT_FLAG ? unitFlag : 0;
        tileMmad(tensorL0C, tensorL0A, tensorL0B, mMad, nActual, kActual, initC, uf);

        if (!ENABLE_UNIT_FLAG && publishC) {
            AscendC::Mutex::Unlock<PIPE_M>(l0CMutexList[l0cSlot]);
        }
        if (releaseA) {
            AscendC::Mutex::Unlock<PIPE_M>(l0AMutexList[l0AListId]);
        }
        AscendC::Mutex::Unlock<PIPE_M>(l0BMutexList[l0BListId]);
        if (releaseA) {
            l0AListId = (l0AListId + 1 < L0AB_STAGES) ? (l0AListId + 1) : 0;
        }
        l0BListId = (l0BListId + 1 < L0AB_STAGES) ? (l0BListId + 1) : 0;
    }

    CATLASS_DEVICE
    void GemmColA(AscendC::LocalTensor<ElementA> l1A, AscendC::LocalTensor<ElementB> l1B,
                  AscendC::LocalTensor<ElementAccumulator> l0C, uint32_t mRound, uint32_t nRound, uint32_t kRound,
                  uint32_t mActual, uint32_t nActual, uint32_t kActual, bool initC, uint8_t unitFlag, uint32_t l0cSlot,
                  uint32_t l1BEvent, bool waitB, bool releaseB, uint32_t bNOff = 0, uint32_t bNFull = 0,
                  bool loadA = true, bool releaseA = true, bool acquireC = true, bool publishC = true,
                  uint16_t l1AFreeFlag = L1_FREE_FLAG_INVALID)
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
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0u, 0u), tla::MakeShape(mRound, nRound));

        if (loadA) {
            AscendC::Mutex::Lock<PIPE_MTE1>(l0AMutexList[l0AListId]);
            copyL1ToL0ACol(tensorL0A, tileL1A);
            AscendC::Mutex::Unlock<PIPE_MTE1>(l0AMutexList[l0AListId]);
            // L1 A is no longer read after this copy. Return its slot before
            // waiting for or loading B; L0A remains owned through the N slices.
            if (l1AFreeFlag != L1_FREE_FLAG_INVALID) {
                AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(l1AFreeFlag);
                AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(l1AFreeFlag + 16);
            }
        }

        AscendC::Mutex::Lock<PIPE_MTE1>(l0BMutexList[l0BListId]);
        if (waitB) {
            AscendC::Mutex::Lock<PIPE_MTE1>(l1MutexList[l1BEvent]);
        }
        copyL1ToL0B(tensorL0B, tileL1B);
        if (releaseB) {
            AscendC::Mutex::Unlock<PIPE_MTE1>(l1MutexList[l1BEvent]);
        }
        AscendC::Mutex::Unlock<PIPE_MTE1>(l0BMutexList[l0BListId]);

        if (loadA) {
            AscendC::Mutex::Lock<PIPE_M>(l0AMutexList[l0AListId]);
        }
        AscendC::Mutex::Lock<PIPE_M>(l0BMutexList[l0BListId]);
        if (!ENABLE_UNIT_FLAG && acquireC) {
            AscendC::Mutex::Lock<PIPE_M>(l0CMutexList[l0cSlot]);
        }
        uint32_t mMad = (mActual == 1) ? 2 : mRound;
        uint8_t uf = ENABLE_UNIT_FLAG ? unitFlag : 0;
        tileMmadCol(tileL0C, tensorL0A, tensorL0B, mMad, nActual, kActual, initC, uf);

        if (!ENABLE_UNIT_FLAG && publishC) {
            AscendC::Mutex::Unlock<PIPE_M>(l0CMutexList[l0cSlot]);
        }
        if (releaseA) {
            AscendC::Mutex::Unlock<PIPE_M>(l0AMutexList[l0AListId]);
        }
        AscendC::Mutex::Unlock<PIPE_M>(l0BMutexList[l0BListId]);
        if (releaseA) {
            l0AListId = (l0AListId + 1 < L0AB_STAGES) ? (l0AListId + 1) : 0;
        }
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

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::Mutex::Lock<PIPE_FIX>(l0CMutexList[l0cSlot]);
        }

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
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::Mutex::Unlock<PIPE_FIX>(l0CMutexList[l0cSlot]);
        }
    }

    uint32_t residentKSlot_ = 0;
    AscendC::LocalTensor<ElementB> l1K;
    AscendC::LocalTensor<ElementB> l1KSlots[2];
    AscendC::LocalTensor<ElementB> l1Q[Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH];
    AscendC::LocalTensor<ElementB> l1Dy[Ascend950FagMlaL1Layout::QDY_BUFFER_DEPTH];

    AscendC::LocalTensor<ElementA> l0ATensorList[L0AB_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[L0AB_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];

    int32_t l0AMutexList[L0AB_STAGES];
    int32_t l0BMutexList[L0AB_STAGES];
    int32_t l0CMutexList[L0C_STAGES];
    int32_t l1MutexList[Ascend950FagMlaL1Layout::SLOT_COUNT];

    // References to the per-AIC cursor shared with the other GEMM block.
    uint32_t& l0AListId;
    uint32_t& l0BListId;

    TileMmad tileMmad;
    TileMmadCol tileMmadCol;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0ACol copyL1ToL0ACol;
    CopyL1ToL0B copyL1ToL0B;
};

} // namespace Catlass::Gemm::Block

#endif // FLASH_ATTN_NPU_ASCEND950_V4_FAG_MMAD_MLA_DQKV_HPP
