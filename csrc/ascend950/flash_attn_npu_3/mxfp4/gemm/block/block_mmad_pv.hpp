/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* Modified by Minghua Shen, 2026. */

#ifndef GEMM_BLOCK_PV_HPP
#define GEMM_BLOCK_PV_HPP

#include "catlass/numeric_size.hpp"
#include "../../arch/resource.hpp"
#include "../../gemm/gemm_dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "../../gemm/tile_common/gemm_tile_copy.hpp"
#include "../../gemm/tile_common/copy_l1_to_l0_mx.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "block_mmad_utils.hpp"

namespace Catlass::Gemm::Block {
template <
    bool transposedMm1_, class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_,
    class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<
    MmadAtlasA5PvMx<transposedMm1_>, L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_, ElementBias_,
    TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA5PvMx<transposedMm1_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;

    using TileMmad = TileMmad_;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr bool IS_MXFP4 = AscendC::IsSameType<ElementA, fp4x2_e2m1_t>::value;
    static_assert(IS_MXFP4, "BlockMmadTla<MmadAtlasA5PvMx> requires ElementA = fp4x2_e2m1_t");

    // Event ID：QK 用 2-3，PV 用 4-7；同 ID 不同 HardEvent 类型不冲突
    static constexpr uint32_t KV_EVENT0 = 4;
    static constexpr uint32_t KV_EVENT1 = 5;
    // PV L0A/B 双缓冲，与 KV_EVENT 复用（不同 HardEvent 类型，flag 独立）
    static constexpr uint32_t PV_L0AB_EVENT0 = KV_EVENT0;
    static constexpr uint32_t PV_L0AB_EVENT1 = KV_EVENT1;
    // PV L0C 单缓冲
    static constexpr uint32_t PV_L0C_EVENT0 = KV_EVENT0;

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t& kvBufId) : l1ABufId(kvBufId)
    {
        for (uint32_t i = 0; i < MXFP4::L1_P_BUF_CNT; i++) {
            l1BTensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementB>(MXFP4::L1_P_BUF_OFFSET + MXFP4::L1_P_BUF_SIZE * i);
        }

        for (uint32_t i = 0; i < MXFP4::L1_P_SCALE_BUF_CNT; i++) {
            l1BScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(
                MXFP4::L1_P_SCALE_BUF_OFFSET + MXFP4::L1_P_SCALE_BUF_SIZE * i);
        }

        for (uint32_t i = 0; i < MXFP4::L1_KV_BUF_CNT; i++) {
            l1ATensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementA>(MXFP4::L1_KV_BUF_OFFSET + MXFP4::L1_KV_BUF_SIZE * i);
        }

        for (uint32_t i = 0; i < MXFP4::L1_KV_DESCALE_BUF_CNT; i++) {
            l1AScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(
                MXFP4::L1_KV_DESCALE_BUF_OFFSET + MXFP4::L1_KV_DESCALE_BUF_SIZE * i);
        }

        for (uint32_t i = 0; i < MXFP4::L0A_PV_BUF_CNT; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(
                MXFP4::L0A_PV_BUF_OFFSET + MXFP4::L0A_PV_BUF_SIZE * i);
        }

        for (uint32_t i = 0; i < MXFP4::L0B_PV_BUF_CNT; i++) {
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(
                MXFP4::L0B_PV_BUF_OFFSET + MXFP4::L0B_PV_BUF_SIZE * i);
        }

        l0CTensor[0] = resource.l0CBuf.template GetBufferByByte<float>(MXFP4::L0C_PV_BUF_OFFSET);

        AllocEventID();
        InitL0BufferForReduceSum();
    }
    __aicore__ inline ~BlockMmadTla()
    {}

    __aicore__ inline void AllocEventID()
    {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(PV_L0C_EVENT0);
    }

    __aicore__ inline void FreeEventID()
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(PV_L0C_EVENT0);
    }

    // V: nZ L1 + V-scale → zN L0A（转置）
    __aicore__ inline void LoadVToL0(uint32_t l1ABufId, uint32_t l0ABufId, uint32_t embedReal, uint32_t s2Align64)
    {
        copyL1ToL0AMx(
            l0ATensor[l0ABufId], l1ATensor[l1ABufId],
            l1AScaleTensor[l1ABufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(), embedReal, s2Align64);
    }

    // P: zN L1 + P-scale → nZ L0B（转置）
    __aicore__ inline void LoadPToL0(
        uint32_t l1BBufId, uint32_t l0BBufId, uint32_t s2Align64, uint32_t embedReal, uint32_t s1Align64)
    {
        constexpr uint32_t s2Base = MXFP4::S2_BASE_TILE_SIZE;
        constexpr uint32_t scaleSrcStride = s2Base / MXFP4::FP4_C0_ELEMS + 1;
        copyL1ToL0BMx(
            l0BTensor[l0BBufId], l1BTensor[l1BBufId],
            l1BScaleTensor[l1BBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(), s2Align64, embedReal, s2Base,
            scaleSrcStride, s1Align64);
    }

    // C = Vᵀ @ P = Oᵀ，initC 由 isTileGroupFirstTile 控制
    __aicore__ inline void MatmulPV(
        uint32_t l0ABufId, uint32_t l0BBufId, uint32_t m, uint32_t n, uint32_t k, bool initC)
    {
        auto l0ALayout = tla::MakeLayout<ElementA, LayoutTagL0A>(m, k);
        auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayout, Catlass::Arch::PositionL0A{});
        auto l0BLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(k, n);
        auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId], l0BLayout, Catlass::Arch::PositionL0B{});
        auto l0CLayout = tla::MakeLayoutL0C(m, n);
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[0], l0CLayout, Catlass::Arch::PositionL0C{});
        tileMmad(l0CTensorTla, l0ATensorTla, l0BTensorTla, m, n, k, initC);
    }

    // V 数据 + V-scale 融合稀疏 gather：按 gSparseBlockIdx 逐段搬 GM → L1。
    // 两者的分段边界必然一致，故只遍历一遍 index，每段发 V / V-scale 两条 copy。
    // ori 连续递增即代表 GM 上物理相邻，向前偷看合并成单条 Nd2Nz。
    // 段推进天然落在 y-block 边界上（仅首段 innerStart 可能非 0），故循环内无除法/取模。
    // firstYBlockIdx / firstOriYBlockIdx 由调用方提前预取，避免首个 index 的阻塞访存落在此处。
    template <class TensorV, class TensorL1A, class TensorVScale, class TensorL1AScale>
    __aicore__ inline void SparseVFusedBaseTileL1FullLoad(
        TensorV& gVTensor, TensorL1A& l1ATensorTla, TensorVScale& gVScaleTensor, TensorL1AScale& l1AScaleTensorTla,
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx, uint32_t gatheredKvSTileIdx, int64_t kvSeqlen,
        uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
        uint32_t curBaseTileSize, uint32_t embed, uint32_t firstYBlockIdx, uint32_t firstOriYBlockIdx)
    {
        using CopyGmToL1A = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorV, TensorL1A>;
        using CopyGmToL1AScale = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorVScale, TensorL1AScale>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1AScale copyGmToL1AScale;

        const uint32_t kvSeqlenU = static_cast<uint32_t>(kvSeqlen);
        uint32_t gatheredYBlockIdx = firstYBlockIdx;
        uint32_t yBlockInnerStart = gatheredKvSTileIdx * kvSBaseTile - firstYBlockIdx * blockShapeY;
        uint32_t oriYBlockIdx = firstOriYBlockIdx;
        uint32_t dealtLenAccum = 0;

        while (dealtLenAccum < curBaseTileSize && gatheredYBlockIdx < yBlockNumRsvd && oriYBlockIdx < yBlockNumAval) {
            uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
            if (oriStartOffset >= kvSeqlenU) {
                break;
            }

            uint32_t curYBlockSize = blockShapeY;
            if (oriYBlockIdx + 1 == yBlockNumAval) {
                curYBlockSize = kvSeqlenU - oriYBlockIdx * blockShapeY;
            }
            if (curYBlockSize <= yBlockInnerStart) {
                break;
            }
            uint32_t availLen = curYBlockSize - yBlockInnerStart;
            // 尾部不满块之后 gather 流即终止（与原实现的 curDealtLen==0 退出等价）
            bool runEndsShort = (curYBlockSize < blockShapeY);

            // 向前偷看：ori 连续则 GM 地址连续，拼进同一条 Nd2Nz
            uint32_t peekYBlockIdx = gatheredYBlockIdx + 1;
            uint32_t peekOriIdx = oriYBlockIdx + 1;
            while (!runEndsShort && dealtLenAccum + availLen < curBaseTileSize && peekYBlockIdx < yBlockNumRsvd &&
                   peekOriIdx < yBlockNumAval &&
                   static_cast<uint32_t>(gSparseBlockIdx.GetValue(peekYBlockIdx)) == peekOriIdx) {
                uint32_t peekBlockSize = blockShapeY;
                if (peekOriIdx + 1 == yBlockNumAval) {
                    peekBlockSize = kvSeqlenU - peekOriIdx * blockShapeY;
                }
                availLen += peekBlockSize;
                runEndsShort = (peekBlockSize < blockShapeY);
                ++peekYBlockIdx;
                ++peekOriIdx;
            }

            uint32_t curDealtLen = min(availLen, curBaseTileSize - dealtLenAccum);
            uint32_t curScaleLen = CeilDiv(curDealtLen, Catlass::MX_SCALE_GROUP_NUM);

            auto gVTensorTile =
                GetTile(gVTensor, tla::MakeCoord(0, oriStartOffset), tla::MakeShape(embed, curDealtLen));
            auto l1ATensorTile =
                GetTile(l1ATensorTla, tla::MakeCoord(0, dealtLenAccum), tla::MakeShape(embed, curDealtLen));
            copyGmToL1A(l1ATensorTile, gVTensorTile);

            auto gVScaleTile = GetTile(
                gVScaleTensor, tla::MakeCoord(0, oriStartOffset / Catlass::MX_SCALE_GROUP_NUM),
                tla::MakeShape(embed, curScaleLen));
            auto l1AScaleTile = GetTile(
                l1AScaleTensorTla, tla::MakeCoord(0, dealtLenAccum / Catlass::MX_SCALE_GROUP_NUM),
                tla::MakeShape(embed, curScaleLen));
            copyGmToL1AScale(l1AScaleTile, gVScaleTile);

            dealtLenAccum += curDealtLen;
            if (runEndsShort) {
                break;
            }
            // 未退出循环时必有 curDealtLen == availLen，偷看过的块已全部消费
            gatheredYBlockIdx = peekYBlockIdx;
            yBlockInnerStart = 0;
            if (dealtLenAccum < curBaseTileSize) {
                oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
            }
        }
    }

    // L0C → UB fixpipe（SPLIT_N 模式）
    template <class TensorUB>
    __aicore__ inline void FixpipeMm2(uint32_t s1Align64, TensorUB& ubOTmpTensor)
    {
        auto l0CTensorTla = tla::MakeTensor(
            l0CTensor[0], tla::MakeLayoutL0C(MXFP4::PV_MMAD_M_DIM, s1Align64), Catlass::Arch::PositionL0C{});
        using CopyL0CToDst = Catlass::Gemm::Tile::CopyL0CToUBTla<
            ArchTag, decltype(l0CTensorTla), TensorUB, Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_N, Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT,
            false>;
        CopyL0CToDst copyL0CToDst;
        copyL0CToDst(ubOTmpTensor, l0CTensorTla);
    }

    // L0C → UB fixpipe（NO_SPLIT 模式，s1 ≤ 64 时单 vector 核搬出）
    template <class TensorUB>
    __aicore__ inline void FixpipeMm2SingleVect(uint32_t s1Align64, TensorUB& ubOTmpTensor)
    {
        auto l0CTensorTla = tla::MakeTensor(
            l0CTensor[0], tla::MakeLayoutL0C(MXFP4::PV_MMAD_M_DIM, s1Align64), Catlass::Arch::PositionL0C{});
        using CopyL0CToDst = Catlass::Gemm::Tile::CopyL0CToUBTla<
            ArchTag, decltype(l0CTensorTla), TensorUB, Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT,
            Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, false>;
        CopyL0CToDst copyL0CToDst;
        copyL0CToDst(ubOTmpTensor, l0CTensorTla);
    }

    __aicore__ inline void InitL0BufferForReduceSum()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(KV_EVENT0 + l1ABufId);

        // seed fill：V data = fp4 4.0, V scale = e8m0 0.25 → 有效值 1.0
        constexpr uint16_t vFillBlocks = static_cast<uint16_t>(MXFP4::L0A_PV_BUF_SIZE / MXFP4::BLOCK_SIZE);
        AscendC::InitConstValueParams<uint16_t> vFillParams(1, vFillBlocks, 0, MXFP4::SEED_V_DATA_FILL);
        AscendC::Fill(l1ATensor[l1ABufId].template ReinterpretCast<uint16_t>(), vFillParams);
        AscendC::PipeBarrier<PIPE_MTE2>();

        constexpr uint16_t vScaleFillBlocks = static_cast<uint16_t>(MXFP4::V_SCALE_L0A_SIZE / MXFP4::BLOCK_SIZE);
        AscendC::InitConstValueParams<uint16_t> vscaleFillParams(1, vScaleFillBlocks, 0, MXFP4::SEED_V_SCALE_FILL);
        AscendC::Fill(l1AScaleTensor[l1ABufId].template ReinterpretCast<uint16_t>(), vscaleFillParams);
        // AscendC::PipeBarrier<PIPE_MTE2>();

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(KV_EVENT0 + l1ABufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(KV_EVENT0 + l1ABufId);

        // seed 装载：把 V 的 D+16 行（含 rowsum pad）非转置装入 L0A，循环 seed 两个 ping-pong 槽
        constexpr uint32_t mStepVal = MXFP4::PV_MMAD_M_DIM / MXFP4::NZ_C0_ELEMS;
        constexpr uint32_t kStepVal = MXFP4::S2_BASE_TILE_SIZE / MXFP4::FP4_C0_ELEMS;

        AscendC::LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.mStep = mStepVal;
        loadData2DParamsA.kStep = kStepVal;
        loadData2DParamsA.srcStride = mStepVal;
        loadData2DParamsA.dstStride = mStepVal;
        loadData2DParamsA.ifTranspose = false;

        AscendC::LoadData2DMxParams loadData2DMXParamsA;
        loadData2DMXParamsA.xStartPosition = 0;
        loadData2DMXParamsA.yStartPosition = 0;
        loadData2DMXParamsA.xStep = mStepVal;
        loadData2DMXParamsA.yStep = kStepVal;
        loadData2DMXParamsA.srcStride = kStepVal;
        loadData2DMXParamsA.dstStride = kStepVal;

        for (uint32_t i = 0; i < MXFP4::L0A_PV_BUF_CNT; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT0 + pvL0abBufId);
            LoadData(
                l0ATensor[pvL0abBufId].template ReinterpretCast<fp4x2_e2m1_t>(),
                l1ATensor[l1ABufId].template ReinterpretCast<fp4x2_e2m1_t>(),
                l1AScaleTensor[l1ABufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(), loadData2DParamsA,
                loadData2DMXParamsA);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT0 + pvL0abBufId);
            pvL0abBufId = (pvL0abBufId + 1) % MXFP4::L0A_PV_BUF_CNT;
        }
        // AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(KV_EVENT0 + l1ABufId);
        l1ABufId = (l1ABufId + 1) % MXFP4::L1_KV_BUF_CNT;
    }

    __aicore__ inline void PadPS2NotAlign64(uint32_t s2Align64, uint32_t s2Align32, uint32_t embedReal, uint32_t pSlot)
    {
        AscendC::InitConstValueParams<uint16_t> PL1InitParams(
            1, static_cast<uint16_t>(s2Align64 - s2Align32), 0, MXFP4::ZERO_FILL_PATTERN);
        uint32_t planeOffset = s2Align32 * MXFP4::FP4_C0_ELEMS;
        AscendC::Fill(l1BTensor[pSlot][planeOffset].template ReinterpretCast<uint16_t>(), PL1InitParams);
        uint32_t plane2Offset = 2 * planeOffset + (s2Align64 - s2Align32) * MXFP4::FP4_C0_ELEMS;
        AscendC::Fill(l1BTensor[pSlot][plane2Offset].template ReinterpretCast<uint16_t>(), PL1InitParams);
    }

    __aicore__ inline void PadVS2NotAlign64(uint32_t s2Align64, uint32_t curBaseTileSize, uint32_t embedReal)
    {
        AscendC::InitConstValueParams<uint16_t> kvL1InitParams(
            1, static_cast<uint16_t>(s2Align64 - curBaseTileSize), 0, MXFP4::ZERO_FILL_PATTERN);
        uint32_t halfEmbed = MXFP4::ROW_SUM_NUM / 2;
        AscendC::Fill(
            l1ATensor[l1ABufId][halfEmbed * curBaseTileSize].template ReinterpretCast<uint16_t>(), kvL1InitParams);
        AscendC::Fill(
            l1ATensor[l1ABufId][halfEmbed * curBaseTileSize * 2 + (s2Align64 - curBaseTileSize) * halfEmbed]
                .template ReinterpretCast<uint16_t>(),
            kvL1InitParams);
    }

    template <class TensorV, class TensorC, class TaskInfoT, class TileInfoT, class SparsityPolicy>
    __aicore__ inline void operator()(
        TensorV& gV, AscendC::GlobalTensor<uint8_t> gVDequantScale, AscendC::GlobalTensor<int32_t> gSparseIdx,
        TensorC& ubOTmpTensor, Catlass::GemmCoord actualBlockShapePV, TileInfoT& delay20TileInfo,
        TaskInfoT& delay20TaskInfo, uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t kvHeadMul, SparsityPolicy)
    {
        uint32_t rowNum = actualBlockShapePV[0];
        uint32_t embed = actualBlockShapePV[1];
        uint32_t curBaseTileSize = actualBlockShapePV[2];

        uint32_t embedReal = embed;
        AscendC::GlobalTensor<uint8_t> gVScale = gVDequantScale[delay20TaskInfo.gmOffsetVScale];
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx = gSparseIdx[delay20TaskInfo.gmOffsetSparseIdx];
        int64_t kvSeqlen = delay20TaskInfo.kvSeqlen;
        uint32_t yBlockNumAval = delay20TaskInfo.yBlockNumAval;
        uint32_t yBlockNumRsvd = delay20TaskInfo.yBlockNumRsvd;
        uint32_t gatheredKvSTileIdx = delay20TileInfo.pvGatheredKvSTileIdx;
        uint32_t pSlot = delay20TileInfo.loop % MXFP4::L1_P_BUF_CNT;

        // [预取] 首个 sparse index 是阻塞式 GM 标量读，且 gather 的第一条 Nd2Nz 地址依赖它。
        //   在此提前发起，把访存延迟藏进后面的 WaitFlag，并让标量单元能继续往前排 MTE2 指令。
        uint32_t firstYBlockIdx = 0;
        uint32_t firstOriYBlockIdx = 0;
        if constexpr (SparsityPolicy::IS_SPARSE) {
            firstYBlockIdx = (gatheredKvSTileIdx * kvSBaseTile) / blockShapeY;
            firstOriYBlockIdx = static_cast<uint32_t>(gSparseBlockIdx.GetValue(firstYBlockIdx));
        }

        uint32_t mMmad = MXFP4::PV_MMAD_M_DIM;

        // 先等 MTE1 释放 L1A 再 Pad: 对齐 QFA 安全顺序, 避免 Pad(MTE2 Fill) 与上一轮
        // MTE1 LoadData(L1A->L0A) 竞争 l1ATensor[l1ABufId]（仅尾部 kvs 不齐 64 时触发, 符合偶现）。
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(KV_EVENT0 + l1ABufId);

        if (delay20TileInfo.kvsActBaseTileAlign32 != delay20TileInfo.kvsActBaseTileAlign64) {
            PadPS2NotAlign64(
                delay20TileInfo.kvsActBaseTileAlign64, delay20TileInfo.kvsActBaseTileAlign32, embedReal, pSlot);
        }

        if (curBaseTileSize != delay20TileInfo.kvsActBaseTileAlign64) {
            PadVS2NotAlign64(delay20TileInfo.kvsActBaseTileAlign64, curBaseTileSize, embedReal);
        }

        auto l1ATensorTla = tla::MakeTensor(
            l1ATensor[l1ABufId],
            tla::MakeLayout<ElementA, Catlass::layout::nZ>(embedReal, delay20TileInfo.kvsActBaseTileAlign64),
            Catlass::Arch::PositionL1{});

        auto gVScaleTensorTla = tla::MakeTensor(
            gVScale.ReinterpretCast<AscendC::fp8_e8m0_t>(),
            tla::MakeMxScaleLayout<AscendC::fp8_e8m0_t, Catlass::layout::ColumnMajor, false>(
                kvHeadMul * embedReal,
                CeilDiv(delay20TaskInfo.kvSeqlen, static_cast<int64_t>(Catlass::MX_SCALE_GROUP_NUM))),
            Catlass::Arch::PositionGM{});
        auto l1AScaleTensorTla = tla::MakeTensor(
            l1AScaleTensor[l1ABufId].ReinterpretCast<AscendC::fp8_e8m0_t>(),
            tla::MakeMxScaleLayout<AscendC::fp8_e8m0_t, Catlass::layout::zZ, false>(
                embedReal, delay20TileInfo.kvsActBaseTileAlign64 / Catlass::MX_SCALE_GROUP_NUM),
            Catlass::Arch::PositionL1{});

        if constexpr (SparsityPolicy::IS_SPARSE) {
            SparseVFusedBaseTileL1FullLoad(
                gV, l1ATensorTla, gVScaleTensorTla, l1AScaleTensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen,
                kvSBaseTile, blockShapeY, yBlockNumAval, yBlockNumRsvd, curBaseTileSize, embedReal, firstYBlockIdx,
                firstOriYBlockIdx);
        } else {
            // Dense policy: load one contiguous V tile and its matching
            // sequence-group scale tile. No selectIdx/gather path is compiled.
            using CopyGmToL1A = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorV, decltype(l1ATensorTla)>;
            using CopyGmToL1AScale =
                Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(gVScaleTensorTla), decltype(l1AScaleTensorTla)>;
            CopyGmToL1A copyGmToL1A;
            CopyGmToL1AScale copyGmToL1AScale;
            const uint32_t denseStart = gatheredKvSTileIdx * kvSBaseTile;
            auto gVTile = GetTile(gV, tla::MakeCoord(0, denseStart), tla::MakeShape(embedReal, curBaseTileSize));
            auto l1VTile = GetTile(l1ATensorTla, tla::MakeCoord(0, 0), tla::MakeShape(embedReal, curBaseTileSize));
            copyGmToL1A(l1VTile, gVTile);

            const uint32_t scaleStart = denseStart / Catlass::MX_SCALE_GROUP_NUM;
            const uint32_t scaleLen = CeilDiv(curBaseTileSize, Catlass::MX_SCALE_GROUP_NUM);
            auto gVScaleTile =
                GetTile(gVScaleTensorTla, tla::MakeCoord(0, scaleStart), tla::MakeShape(embedReal, scaleLen));
            auto l1VScaleTile = GetTile(l1AScaleTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(embedReal, scaleLen));
            copyGmToL1AScale(l1VScaleTile, gVScaleTile);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(KV_EVENT0 + l1ABufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(KV_EVENT0 + l1ABufId);

        uint32_t l0CEventId = PV_L0C_EVENT0;
        uint32_t l0ABufId = pvL0abBufId;
        uint32_t l0BBufId = pvL0abBufId;
        uint32_t l0ABEventId = PV_L0AB_EVENT0 + pvL0abBufId;

        if (delay20TileInfo.isTileGoupFirstTile) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);

        LoadPToL0(
            pSlot, l0BBufId, delay20TileInfo.kvsActBaseTileAlign64, embedReal, delay20TaskInfo.qsActBaseTileAlign64);
        LoadVToL0(l1ABufId, l0ABufId, embedReal, delay20TileInfo.kvsActBaseTileAlign64);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);

        MatmulPV(
            l0ABufId, l0BBufId, mMmad, delay20TaskInfo.qsActBaseTileAlign64, delay20TileInfo.kvsActBaseTileAlign64,
            delay20TileInfo.isTileGoupFirstTile);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);
        pvL0abBufId = (pvL0abBufId + 1) % MXFP4::L0A_PV_BUF_CNT;

        if (delay20TileInfo.isUpdatePScale) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            if (delay20TaskInfo.qsActBaseTileAlign8 <= MXFP4::CONST_64) {
                FixpipeMm2SingleVect(delay20TaskInfo.qsActBaseTileAlign64, ubOTmpTensor);
            } else {
                FixpipeMm2(delay20TaskInfo.qsActBaseTileAlign64, ubOTmpTensor);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(KV_EVENT0 + l1ABufId);
        l1ABufId = (l1ABufId + 1) % MXFP4::L1_KV_BUF_CNT;

        if (delay20TileInfo.isLastSecondKvsTile) {
            InitL0BufferForReduceSum();
        }
    }

protected:
    AscendC::LocalTensor<ElementA> l1ATensor[MXFP4::L1_KV_BUF_CNT];
    AscendC::LocalTensor<ElementB> l1BTensor[MXFP4::L1_P_BUF_CNT];
    AscendC::LocalTensor<ElementA> l0ATensor[MXFP4::L0A_PV_BUF_CNT];
    AscendC::LocalTensor<ElementB> l0BTensor[MXFP4::L0A_PV_BUF_CNT];
    AscendC::LocalTensor<float> l0CTensor[MXFP4::L0C_PV_BUF_CNT];
    AscendC::LocalTensor<uint8_t> l1AScaleTensor[MXFP4::L1_KV_BUF_CNT];
    AscendC::LocalTensor<uint8_t> l1BScaleTensor[MXFP4::L1_P_BUF_CNT];

    TileMmad tileMmad;
    Catlass::Gemm::Tile::CopyL1ToL0AMxA5 copyL1ToL0AMx;
    Catlass::Gemm::Tile::CopyL1ToL0BMxA5 copyL1ToL0BMx;

    uint32_t pvL0abBufId = 0;
    uint32_t& l1ABufId;
};

} // namespace Catlass::Gemm::Block

#endif // GEMM_BLOCK_PV_HPP
