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

#ifndef GEMM_BLOCK_QK_HPP
#define GEMM_BLOCK_QK_HPP

#include "catlass/numeric_size.hpp"
#include "../../arch/resource.hpp"
#include "../../gemm/gemm_dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "../../gemm/tile_common/gemm_tile_copy.hpp"
#include "block_mmad_utils.hpp"
#include "../../gemm/tile_common/copy_l1_to_l0_mx.hpp"
#include "../../gemm/tile_common/copy_gm_to_l1_mx_scale_dn2nz.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Block {

template <
    class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_, class ElementBias_,
    class TileCopy_, class TileMmad_>
struct BlockMmadTla<
    MmadAtlasA5QkMx<true>, L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_,
    TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA5QkMx<true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;

    using TileMmad = TileMmad_;

    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape_{});

    // Event ID：QK 用 2-3，L1 K 复用 MXFP4::KV_EVENT0..3（与 PV 共享）
    static constexpr uint32_t Q_EVENT0 = 2;
    static constexpr uint32_t Q_EVENT1 = 3;
    static constexpr uint32_t QK_L0AB_EVENT0 = 2;
    static constexpr uint32_t QK_L0AB_EVENT1 = 3;
    static constexpr uint32_t QK_L0C_EVENT0 = 2;
    static constexpr uint32_t QK_L0C_EVENT1 = 3;

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t& kvBufId, uint64_t softmaxScale)
        : kBufId(kvBufId)
    {
        softmaxScale_ = softmaxScale;
        for (uint32_t i = 0; i < MXFP4::L1_Q_BUF_CNT; i++) {
            l1BTensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementA>(MXFP4::L1_Q_BUF_OFFSET + MXFP4::L1_Q_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::L1_Q_BUF_CNT; i++) {
            l1BScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(
                MXFP4::L1_Q_DESCALE_BUF_OFFSET + MXFP4::L1_Q_DESCALE_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::L1_KV_BUF_CNT; i++) {
            l1ATensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementB>(MXFP4::L1_KV_BUF_OFFSET + MXFP4::L1_KV_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::L1_KV_DESCALE_BUF_CNT; i++) {
            l1AScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(
                MXFP4::L1_KV_DESCALE_BUF_OFFSET + MXFP4::L1_KV_DESCALE_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::L0A_QK_BUF_CNT; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementB>(
                MXFP4::L0A_QK_BUF_OFFSET + MXFP4::L0A_QK_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::L0B_QK_BUF_CNT; i++) {
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementA>(
                MXFP4::L0B_QK_BUF_OFFSET + MXFP4::L0B_QK_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::L0C_QK_BUF_CNT; i++) {
            l0CTensor[i] =
                resource.l0CBuf.template GetBufferByByte<float>(MXFP4::L0C_QK_BUF_OFFSET + MXFP4::L0C_QK_BUF_SIZE * i);
        }

        AllocEventID();
    }

    __aicore__ inline void AllocEventID()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT1);
    }

    __aicore__ inline void FreeEventID()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT1);
    }

    __aicore__ inline ~BlockMmadTla()
    {}

    __aicore__ inline uint32_t CeilDivision(uint32_t numerator, uint32_t denominator)
    {
        return (numerator + denominator - 1) / denominator;
    }

    __aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
    {
        return (a < b) ? a : b;
    }

    // Q 稠密 GM → L1（RowMajor → zN）
    template <class TensorQ, class TensorL1Q>
    __aicore__ inline void CopyQGmToL1(
        uint32_t qsActBaseTile, uint32_t embed, TensorQ gQTensorTla, TensorL1Q l1QTensorTla)
    {
        auto l1QTileTla = GetTile(l1QTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(qsActBaseTile, embed));
        // BSND: GetTile 确保 dValue = embed 而非 qHeadMul * embed
        auto gQTileTla = GetTile(gQTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(qsActBaseTile, embed));

        using CopyGmToL1Q = Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(gQTileTla), decltype(l1QTensorTla)>;
        CopyGmToL1Q copyGmToL1Q;
        copyGmToL1Q(l1QTileTla, gQTileTla);
    }

    // Q-scale 稠密 GM → L1（Host 沿 D Dn2Nz）
    __aicore__ inline void CopyQScaleGmToL1(
        uint32_t qsActBaseTile, uint32_t scaleK, AscendC::GlobalTensor<uint8_t>& gQDequantScale, uint32_t qBufId,
        uint32_t qHeadMul)
    {
        copyGmToL1MxScaleDn2Nz(l1BScaleTensor[qBufId], gQDequantScale, qsActBaseTile, scaleK, 0, 0, qHeadMul);
    }

    // K 数据 + K-scale 融合稀疏 gather：按 gSparseBlockIdx 逐段搬 GM → L1。
    // 两者的分段边界必然一致，故只遍历一遍 index，每段发 K(Nd2Nz) / K-scale(Dn2Nz) 两条 copy。
    // ori 连续递增即代表 GM 上物理相邻，向前偷看合并成单条 copy。
    // 段推进天然落在 y-block 边界上（仅首段 innerStart 可能非 0），故循环内无除法/取模。
    // firstYBlockIdx / firstOriYBlockIdx 由调用方提前预取，避免首个 index 的阻塞访存落在此处。
    template <class TensorK, class TensorL1K>
    __aicore__ inline void SparseKFusedBaseTileL1FullLoad(
        TensorK gKTensorTla, TensorL1K l1KTensorTla, AscendC::GlobalTensor<uint8_t> gKScale,
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx, uint32_t gatheredKvSTileIdx, int64_t kvSeqlen,
        uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
        uint32_t kvsActBaseTile, uint32_t embed, uint32_t kBufId, uint32_t kvHeadMul, uint32_t firstYBlockIdx,
        uint32_t firstOriYBlockIdx)
    {
        using CopyGmToL1K = Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(gKTensorTla), decltype(l1KTensorTla)>;
        CopyGmToL1K copyGmToL1K;

        uint32_t scaleK = CeilDivision(embed, MXFP4::FP4_C0_ELEMS) * 2;
        uint32_t scaleRowHalf = scaleK >> 1;
        // headMul * scaleRowHalf 在整段 gather 里不变，提到循环外，避免每段重算并重载 this。
        const uint32_t srcScaleStride = kvHeadMul * scaleRowHalf;
        AscendC::GlobalTensor<half> gKScaleB16;
        gKScaleB16.SetGlobalBuffer((__gm__ half *)(gKScale.GetPhyAddr()), MXFP4::GM_MAX_BUFFER_LEN);
        auto l1KScaleB16 = l1AScaleTensor[kBufId].template ReinterpretCast<half>();
        AscendC::Dn2NzParams scaleDn2Nz;
        scaleDn2Nz.dnNum = 1;
        scaleDn2Nz.nValue = scaleRowHalf;
        scaleDn2Nz.srcDnMatrixStride = 0;
        scaleDn2Nz.srcDValue = srcScaleStride;
        scaleDn2Nz.dstNzC0Stride = scaleRowHalf;
        scaleDn2Nz.dstNzNStride = 1;
        scaleDn2Nz.dstNzMatrixStride = 0;

        const uint32_t kvSeqlenU = static_cast<uint32_t>(kvSeqlen);
        uint32_t gatheredYBlockIdx = firstYBlockIdx;
        uint32_t yBlockInnerStart = gatheredKvSTileIdx * kvSBaseTile - firstYBlockIdx * blockShapeY;
        uint32_t oriYBlockIdx = firstOriYBlockIdx;
        uint32_t dealtLenAccum = 0;

        while (dealtLenAccum < kvsActBaseTile && gatheredYBlockIdx < yBlockNumRsvd && oriYBlockIdx < yBlockNumAval) {
            uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
            if (oriStartOffset >= kvSeqlenU) {
                break;
            }

            // 满块直接用 blockShapeY。尾块才改长度，避免每段都做一次 SEL。
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

            // 向前偷看：ori 连续则 GM 地址连续，拼进同一条 copy
            uint32_t peekYBlockIdx = gatheredYBlockIdx + 1;
            uint32_t peekOriIdx = oriYBlockIdx + 1;
            while (!runEndsShort && dealtLenAccum + availLen < kvsActBaseTile && peekYBlockIdx < yBlockNumRsvd &&
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

            uint32_t curDealtLen = MinU32(availLen, kvsActBaseTile - dealtLenAccum);

            auto gKTile = GetTile(gKTensorTla, tla::MakeCoord(oriStartOffset, 0), tla::MakeShape(curDealtLen, embed));
            auto l1KTile = GetTile(l1KTensorTla, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(curDealtLen, embed));
            copyGmToL1K(l1KTile, gKTile);

            // 原实现每段调用 copyGmToL1MxScaleDn2Nz，在循环内重算
            // srcB16Off/dstB16Off 并重建 Dn2Nz 参数。这里参数、half 视图和
            // srcScaleStride 已在循环外备好，每段只更新 dValue=curDealtLen。
            // 源地址仍是 oriStartOffset * kvHeadMul * scaleRowHalf，目的地址仍是
            // dealtLenAccum * scaleRowHalf。
            scaleDn2Nz.dValue = curDealtLen;
            AscendC::DataCopy(
                l1KScaleB16[dealtLenAccum * scaleRowHalf], gKScaleB16[oriStartOffset * srcScaleStride], scaleDn2Nz);

            dealtLenAccum += curDealtLen;
            if (runEndsShort) {
                break;
            }
            // 未退出循环时必有 curDealtLen == availLen，偷看过的块已全部消费
            gatheredYBlockIdx = peekYBlockIdx;
            yBlockInnerStart = 0;
            if (dealtLenAccum < kvsActBaseTile) {
                oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
            }
        }
    }

    // Q → L0B（非转置，带 mx scale）
    __aicore__ inline void LoadQToL0B(
        uint32_t qsActBaseTileAlign16, uint32_t embed, uint32_t qsActBaseTileAlign16L0, uint32_t scaleMAlign16,
        uint32_t scaleK, uint32_t qBufId, uint32_t qkL0abBufId)
    {
        copyL1ToL0BMxQk(
            l0BTensor[qkL0abBufId].template ReinterpretCast<fp4x2_e2m1_t>(),
            l1BTensor[qBufId].template ReinterpretCast<fp4x2_e2m1_t>(),
            l1BScaleTensor[qBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(), qsActBaseTileAlign16, embed,
            qsActBaseTileAlign16L0, scaleMAlign16, scaleK);
    }

    // K → L0A（非转置，带 mx scale，n 方向子切分）
    __aicore__ inline void LoadKToL0A(
        uint32_t nSubRowStart, uint32_t nCur, uint32_t embed, uint32_t kvsActBaseTileAlign16, uint32_t scaleK,
        uint32_t kBufId, uint32_t qkL0abBufId)
    {
        copyL1ToL0AMxQk(
            l0ATensor[qkL0abBufId].template ReinterpretCast<fp4x2_e2m1_t>(),
            l1ATensor[kBufId].template ReinterpretCast<fp4x2_e2m1_t>(),
            l1AScaleTensor[kBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(), nSubRowStart, nCur, embed,
            kvsActBaseTileAlign16, scaleK);
    }

    // C = K · Qᵀ，每 nSub 独立计算，initC 恒 true
    __aicore__ inline void MatmulQK(
        uint32_t n, uint32_t qsActBaseTileAlign16, uint32_t embed, uint32_t qkL0abBufId, uint32_t qkL0cBufId)
    {
        auto l0ALayout = tla::MakeLayout<ElementB, LayoutTagL0A>(n, embed);
        auto l0ATensorTla = tla::MakeTensor(l0ATensor[qkL0abBufId], l0ALayout, Catlass::Arch::PositionL0A{});
        auto l0BLayout = tla::MakeLayout<ElementA, LayoutTagL0B>(embed, qsActBaseTileAlign16);
        auto l0BTensorTla = tla::MakeTensor(l0BTensor[qkL0abBufId], l0BLayout, Catlass::Arch::PositionL0B{});
        auto l0CLayout = tla::MakeLayoutL0C(n, qsActBaseTileAlign16);
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[qkL0cBufId], l0CLayout, Catlass::Arch::PositionL0C{});
        tileMmad(l0CTensorTla, l0ATensorTla, l0BTensorTla, n, qsActBaseTileAlign16, embed, true);
    }

    // L0C → UB fixpipe（NO_SPLIT 模式，subBlockId 选目标 AIV）
    template <class TensorUB>
    __aicore__ inline void FixpipeMm1(
        uint32_t qsActBaseTileAlign16, uint32_t n, TensorUB& ubSTensorTla, bool subBlockId, uint32_t qkL0cBufId)
    {
        auto l0CTensorTla = tla::MakeTensor(
            l0CTensor[qkL0cBufId], tla::MakeLayoutL0C(n, qsActBaseTileAlign16), Catlass::Arch::PositionL0C{});

        using CopyL0CToUB = Catlass::Gemm::Tile::CopyL0CToUBTla<
            ArchTag, decltype(l0CTensorTla), TensorUB, Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT,
            Catlass::Gemm::Tile::ScaleGranularity::PER_TENSOR, false>;
        CopyL0CToUB copyL0CToUB;
        copyL0CToUB(ubSTensorTla, l0CTensorTla, softmaxScale_, subBlockId);
    }

    // 发一次 nSub 的 fixpipe（含 M_FIX / FIX_M 的成对同步）。
    //   拆出来是为了能延后一拍发射：调用点把它插在 mmad 之后、下一轮的 WaitFlag<MTE1_M> 之前，
    //   用它的发射时间掩盖跨流水 flag 的往返延迟。
    template <class TensorUB>
    __aicore__ inline void EmitFixpipeMm1(
        uint32_t l0cBufId, uint32_t nSubRowStart, uint32_t nCur, uint32_t qsActBaseTileAlign16, TensorUB& ubSTensorTla,
        uint32_t subBlockIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(QK_L0C_EVENT0 + l0cBufId);
        // nCur 偶数对齐用于 Fixpipe mSize
        uint32_t nCurEven = (nCur + 1) >> 1 << 1;
        auto ubSSubTla =
            GetTile(ubSTensorTla, tla::MakeCoord(nSubRowStart, 0), tla::MakeShape(nCurEven, qsActBaseTileAlign16));
        FixpipeMm1(qsActBaseTileAlign16, nCur, ubSSubTla, subBlockIdx != 0, l0cBufId);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT0 + l0cBufId);
    }

    template <class TensorQ, class TensorK, class TensorUB, class TaskInfoT, class TileInfoT, class SparsityPolicy>
    __aicore__ inline void operator()(
        TensorQ& gQTensorTla, TensorK& gKTensorTla, AscendC::GlobalTensor<uint8_t> gQDequantScale,
        AscendC::GlobalTensor<uint8_t> gKDequantScale, AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
        TensorUB& ubSTensorTla, TaskInfoT& curTaskInfo, TileInfoT& curTileInfo, uint32_t embed, uint32_t kvSBaseTile,
        uint32_t blockShapeY, uint32_t subBlockIdx, uint32_t kvHeadMul, uint32_t qHeadMul, SparsityPolicy)
    {
        uint32_t qsActBaseTile = curTaskInfo.qsActBaseTile;
        int64_t kvSeqlen = curTaskInfo.kvSeqlen;
        uint32_t yBlockNumAval = curTaskInfo.yBlockNumAval;
        uint32_t yBlockNumRsvd = curTaskInfo.yBlockNumRsvd;
        uint32_t qsActBaseTileAlign16 = curTaskInfo.qsActBaseTileAlign16;
        uint32_t qsActBaseTileAlign128 = curTaskInfo.qsActBaseTileAlign128;

        uint32_t kvsActBaseTile = curTileInfo.kvsActBaseTile;
        uint32_t gatheredKvSTileIdx = curTileInfo.pvGatheredKvSTileIdx;
        bool isFirstKvsTile = curTileInfo.isFirstKvsTile;
        bool isLastKvsTile = curTileInfo.isLastKvsTile;
        uint32_t kvsActBaseTileAlign16 = curTileInfo.kvsActBaseTileAlign16;

        uint32_t scaleK = CeilDivision(embed, MXFP4::FP4_C0_ELEMS) * 2;
        uint32_t l1KEventId = MXFP4::KV_EVENT0 + kBufId;
        uint32_t l1QEventId = Q_EVENT0 + qBufId;

        // [预取] 首个 sparse index 是阻塞式 GM 标量读，且 gather 的第一条 copy 地址依赖它。
        //   在此提前发起，把访存延迟藏进后面的 Q 搬运 / WaitFlag，
        //   并让标量单元能继续往前排 MTE2 指令。
        uint32_t firstYBlockIdx = 0;
        uint32_t firstOriYBlockIdx = 0;
        if constexpr (SparsityPolicy::IS_SPARSE) {
            firstYBlockIdx = (gatheredKvSTileIdx * kvSBaseTile) / blockShapeY;
            firstOriYBlockIdx = static_cast<uint32_t>(gSparseBlockIdx.GetValue(firstYBlockIdx));
        }

        // Q/QScale 仅在 task 首 KV tile 搬 GM → L1，后续 tile 复用
        if (isFirstKvsTile) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1QEventId);
            auto l1QLayout = tla::MakeLayout<ElementA, Catlass::layout::zN>(qsActBaseTileAlign16, embed);
            auto l1QTensorTla = tla::MakeTensor(l1BTensor[qBufId], l1QLayout, Catlass::Arch::PositionL1{});
            CopyQGmToL1(qsActBaseTile, embed, gQTensorTla, l1QTensorTla);
            CopyQScaleGmToL1(qsActBaseTile, scaleK, gQDequantScale, qBufId, qHeadMul);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1QEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1QEventId);

            // [Q 常驻 L0B] Q 在整个 task 内不变，LoadQToL0B 的实参全是循环不变量；
            //   而 QK 的 L0B(offset 0..16K) 与 PV 的(16K 起)分属不同区间，无人覆盖，
            //   故在此一次灌满两个 ping-pong 槽，nSub 循环里不再重复装载。
            //   原实现每个 nSub 都重装同一份 Q，一个 task 的 518 次装载里 516 次是冗余。
            for (uint32_t i = 0; i < MXFP4::L0A_QK_BUF_CNT; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT0 + qkL0abBufId);
                LoadQToL0B(
                    qsActBaseTileAlign16, embed, qsActBaseTileAlign16, qsActBaseTileAlign16, scaleK, qBufId,
                    qkL0abBufId);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT0 + qkL0abBufId);
                qkL0abBufId = (qkL0abBufId + 1) % MXFP4::L0A_QK_BUF_CNT;
            }
        }

        // 每 tile 稀疏 gather K / K-scale 到 L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KEventId);
        auto l1KLayout = tla::MakeLayout<ElementA, Catlass::layout::zN>(kvsActBaseTile, embed);
        auto l1KTensorTla = tla::MakeTensor(l1ATensor[kBufId], l1KLayout, Catlass::Arch::PositionL1{});
        if constexpr (SparsityPolicy::IS_SPARSE) {
            SparseKFusedBaseTileL1FullLoad(
                gKTensorTla, l1KTensorTla, gKDequantScale, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile,
                blockShapeY, yBlockNumAval, yBlockNumRsvd, kvsActBaseTile, embed, kBufId, kvHeadMul, firstYBlockIdx,
                firstOriYBlockIdx);
        } else {
            // Dense policy: K and K-scale are contiguous in sequence order.
            // No sparse index tensor is read or materialized.
            using CopyGmToL1K = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorK, decltype(l1KTensorTla)>;
            CopyGmToL1K copyGmToL1K;
            const uint32_t denseStart = gatheredKvSTileIdx * kvSBaseTile;
            auto gKTile = GetTile(gKTensorTla, tla::MakeCoord(denseStart, 0), tla::MakeShape(kvsActBaseTile, embed));
            auto l1KTile = GetTile(l1KTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(kvsActBaseTile, embed));
            copyGmToL1K(l1KTile, gKTile);
            const uint32_t scaleK = CeilDivision(embed, MXFP4::FP4_C0_ELEMS) * 2;
            const uint32_t scaleRowHalf = scaleK / 2;
            copyGmToL1MxScaleDn2Nz(
                l1AScaleTensor[kBufId], gKDequantScale, kvsActBaseTile, scaleK, denseStart * kvHeadMul * scaleRowHalf,
                0, kvHeadMul);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KEventId);

        // n 方向按 L0_TILE_N 子切分。
        // [软流水] fixpipe 延后一拍发射：原实现里 SetFlag/WaitFlag 成对紧挨着（M 等 MTE1、
        //   FIX 等 M），两次跨流水往返中间没有任何工作可掩盖，实测每个 nSub 白等约 92 cycle。
        //   改成：mmad(k) 发完只 Set 不 Wait，把 fixpipe(k-1) 插到 LoadK(k) 与 mmad(k) 之间，
        //   于是 MTE1_M 的往返被 fixpipe 的发射盖住、M_FIX 的往返被 LoadK 盖住。
        //   L0C 双缓冲天然够用：c_k 的 FIX_M 由 k+1 轮释放，k+2 轮才复用。
        //   注意不能把 pending 的 fixpipe 带出本次调用——kernel 紧接着会 CrossCoreSetFlag
        //   通知向量核 S 已就绪，所以循环结束必须 flush。
        uint32_t nLoopNum = CeilDivision(kvsActBaseTile, L0_TILE_N);
        bool fixPending = false;
        uint32_t pendL0cBufId = 0;
        uint32_t pendNSubRowStart = 0;
        uint32_t pendNCur = 0;

        // 基址和与 nSub 无关的 LoadData / Mmad 字段在进循环前落到局部变量。
        // 混合 TU 下 this 的成员会溢到栈上，每轮 LD_IO 再重算地址。
        constexpr uint32_t kNzC0 = MXFP4::NZ_C0_ELEMS;
        constexpr uint32_t kFp4C0 = MXFP4::FP4_C0_ELEMS;
        const uint32_t scaleRowHalfL0 = scaleK >> 1;
        const uint32_t kStep = (embed + kFp4C0 - 1) / kFp4C0;
        const uint32_t srcStrideRows = kvsActBaseTileAlign16 / kNzC0;
        auto l1KFp4 = l1ATensor[kBufId].template ReinterpretCast<fp4x2_e2m1_t>();
        auto l1KScaleE8 = l1AScaleTensor[kBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>();
        auto l0AFp4_0 = l0ATensor[0].template ReinterpretCast<fp4x2_e2m1_t>();
        auto l0AFp4_1 = l0ATensor[1].template ReinterpretCast<fp4x2_e2m1_t>();
        auto l0A0 = l0ATensor[0];
        auto l0A1 = l0ATensor[1];
        auto l0B0 = l0BTensor[0];
        auto l0B1 = l0BTensor[1];
        auto l0C0 = l0CTensor[0];
        auto l0C1 = l0CTensor[1];

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.sid = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.kStep = kStep;
        loadDataParams.srcStride = srcStrideRows;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData2DMxParams loadMxDataParams;
        loadMxDataParams.yStartPosition = 0;
        loadMxDataParams.yStep = scaleRowHalfL0;
        loadMxDataParams.srcStride = scaleRowHalfL0;
        loadMxDataParams.dstStride = scaleRowHalfL0;

        AscendC::MmadParams mmadParams;
        mmadParams.n = qsActBaseTileAlign16;
        mmadParams.k = embed;
        mmadParams.unitFlag = 0;
        mmadParams.cmatrixInitVal = true;
#if (__CCE_AICORE__ == 310)
        mmadParams.disableGemv = true;
#endif

        for (uint32_t nSub = 0; nSub < nLoopNum; ++nSub) {
            uint32_t l0ABEventId = QK_L0AB_EVENT0 + qkL0abBufId;
            uint32_t l0CEventId = QK_L0C_EVENT0 + qkL0cBufId;
            uint32_t nSubRowStart = nSub * L0_TILE_N;
            uint32_t nCur = MinU32(L0_TILE_N, kvsActBaseTile - nSubRowStart);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);
            // Q 已在 task 首 tile 常驻两个 L0B 槽，此处只需装 K
            // Q 已常驻 L0B。这里只更新随 nSub 变化的 m 起点和步长，不再重建 layout。
            uint32_t nCurAlign16 = (nCur + kNzC0 - 1) / kNzC0 * kNzC0;
            uint32_t mStep = nCurAlign16 / kNzC0;
            uint32_t mStart = nSubRowStart / kNzC0;
            loadDataParams.mStartPosition = mStart;
            loadDataParams.mStep = mStep;
            loadDataParams.dstStride = mStep;
            loadMxDataParams.xStartPosition = mStart;
            loadMxDataParams.xStep = mStep;
            if (qkL0abBufId == 0) {
                AscendC::LoadData(l0AFp4_0, l1KFp4, l1KScaleE8, loadDataParams, loadMxDataParams);
            } else {
                AscendC::LoadData(l0AFp4_1, l1KFp4, l1KScaleE8, loadDataParams, loadMxDataParams);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);

            // 上一轮的 fixpipe 插在这里，掩盖刚发出的 MTE1_M 往返
            if (fixPending) {
                EmitFixpipeMm1(
                    pendL0cBufId, pendNSubRowStart, pendNCur, qsActBaseTileAlign16, ubSTensorTla, subBlockIdx);
                fixPending = false;
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);
            mmadParams.m = nCur;
            if (qkL0abBufId == 0) {
                if (qkL0cBufId == 0) {
                    AscendC::Mmad(l0C0, l0A0, l0B0, mmadParams);
                } else {
                    AscendC::Mmad(l0C1, l0A0, l0B0, mmadParams);
                }
            } else if (qkL0cBufId == 0) {
                AscendC::Mmad(l0C0, l0A1, l0B1, mmadParams);
            } else {
                AscendC::Mmad(l0C1, l0A1, l0B1, mmadParams);
            }
            if ((nCur / kNzC0) * (qsActBaseTileAlign16 / kNzC0) < 10) {
                AscendC::PipeBarrier<PIPE_M>();
            }
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);
            qkL0abBufId = (qkL0abBufId + 1) % MXFP4::L0A_QK_BUF_CNT;

            // 只 Set 不 Wait，本轮的 fixpipe 推迟到下一轮（或循环外的 flush）
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            pendL0cBufId = qkL0cBufId;
            pendNSubRowStart = nSubRowStart;
            pendNCur = nCur;
            fixPending = true;
            qkL0cBufId = (qkL0cBufId + 1) % MXFP4::L0C_QK_BUF_CNT;
        }
        if (fixPending) {
            EmitFixpipeMm1(pendL0cBufId, pendNSubRowStart, pendNCur, qsActBaseTileAlign16, ubSTensorTla, subBlockIdx);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KEventId);
        kBufId = (kBufId + 1) % MXFP4::L1_KV_BUF_CNT;

        if (unlikely(isLastKvsTile)) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1QEventId);
            qBufId = (qBufId + 1) % MXFP4::L1_Q_BUF_CNT;
        }
    }

protected:
    AscendC::LocalTensor<ElementB> l1ATensor[MXFP4::L1_KV_BUF_CNT];
    AscendC::LocalTensor<ElementA> l1BTensor[MXFP4::L1_Q_BUF_CNT];
    AscendC::LocalTensor<ElementB> l0ATensor[MXFP4::L0A_QK_BUF_CNT];
    AscendC::LocalTensor<ElementA> l0BTensor[MXFP4::L0A_QK_BUF_CNT];
    AscendC::LocalTensor<float> l0CTensor[MXFP4::L0C_QK_BUF_CNT];
    AscendC::LocalTensor<uint8_t> l1AScaleTensor[MXFP4::L1_KV_BUF_CNT];
    AscendC::LocalTensor<uint8_t> l1BScaleTensor[MXFP4::L1_Q_BUF_CNT];

    Catlass::Gemm::Tile::CopyL1ToL0AMxQKA5 copyL1ToL0AMxQk;
    Catlass::Gemm::Tile::CopyL1ToL0BMxQKA5 copyL1ToL0BMxQk;
    TileMmad tileMmad;
    Catlass::Gemm::Tile::CopyGmToL1MxScaleDn2NzA5 copyGmToL1MxScaleDn2Nz;

    uint32_t qBufId = 0;
    uint32_t& kBufId;
    uint32_t qkL0abBufId = 0;
    uint32_t qkL0cBufId = 0;
    uint64_t softmaxScale_;
};

} // namespace Catlass::Gemm::Block

#endif // GEMM_BLOCK_QK_HPP
