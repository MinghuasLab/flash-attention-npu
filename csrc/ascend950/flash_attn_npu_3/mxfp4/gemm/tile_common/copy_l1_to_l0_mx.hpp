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

#ifndef GEMM_TILE_COPY_L1_TO_L0_MX_HPP
#define GEMM_TILE_COPY_L1_TO_L0_MX_HPP

#include "catlass/numeric_size.hpp"
#include "tla/tensor.hpp"
#include "../../gemm/block/block_mmad_utils.hpp"

namespace Catlass::Gemm::Tile {

// MXFP4 专用 L1 → L0 搬运 policy（带 E8M0 scale 的 5 参 mx LoadData）。
// 独立于既有 TileCopyTla dispatch，避免影响其它 kernel。
// 维数由形参显式传入，暂不经 tla 张量 shape 推导。

// mx L1 → L0A 转置搬运：L1 V(nZ, [S2,D]) + V-scale → L0A(zN, Vᵀ=[D,S2])
struct CopyL1ToL0AMxA5 {
    CATLASS_DEVICE
    CopyL1ToL0AMxA5() {}

    // dAct  : embed 维（源 k → 转置后 L0A m）
    // s2Act : seq 维对齐 64（源 m → 转置后 L0A k）
    template <class TensorDst, class TensorSrc, class TensorScale>
    CATLASS_DEVICE void operator()(TensorDst const& dst, TensorSrc const& src, TensorScale const& scale, uint32_t dAct,
                                   uint32_t s2Act)
    {
        constexpr uint32_t NZ_C0 = Catlass::Gemm::Block::MXFP4::NZ_C0_ELEMS;
        constexpr uint32_t FP4_C0 = Catlass::Gemm::Block::MXFP4::FP4_C0_ELEMS;

        AscendC::LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.mStep = s2Act / NZ_C0;
        loadData2DParamsA.kStep = dAct / FP4_C0;
        loadData2DParamsA.srcStride = s2Act / NZ_C0;
        loadData2DParamsA.dstStride = (dAct + NZ_C0 - 1) / NZ_C0 + 1;
        if (dAct <= FP4_C0) {
            loadData2DParamsA.dstStride += FP4_C0 / NZ_C0;
        }
        loadData2DParamsA.ifTranspose = true;

        AscendC::LoadData2DMxParams load2DMxParamsA;
        load2DMxParamsA.xStartPosition = 0;
        load2DMxParamsA.yStartPosition = 0;
        load2DMxParamsA.xStep = (dAct + NZ_C0 - 1) / NZ_C0;
        load2DMxParamsA.yStep = (s2Act + FP4_C0 - 1) / FP4_C0;
        load2DMxParamsA.srcStride = load2DMxParamsA.yStep;
        load2DMxParamsA.dstStride = load2DMxParamsA.yStep;

        AscendC::LoadData(dst, src, scale, loadData2DParamsA, load2DMxParamsA);
    }
};

// mx L1 → L0B 转置搬运：L1 P(zN, [M,S2]) + P-scale → L0B(nZ, Pᵀ=[S2,M])
struct CopyL1ToL0BMxA5 {
    CATLASS_DEVICE
    CopyL1ToL0BMxA5() {}

    // s2Align64         : seq 维对齐 64（源 m → 转置后 L0B k）
    // dAct              : embed 维
    // s2Base            : 整 tile 的 S2 基准尺寸，用于推导 scale srcStride
    // actScaleSrcStride : 实际 scale srcStride（0 表示由 s2Base 推导）
    // s1Align64         : M 维对齐 64（源 k → 转置后 L0B n）
    template <class TensorDst, class TensorSrc, class TensorScale>
    CATLASS_DEVICE void operator()(TensorDst const& dst, TensorSrc const& src, TensorScale const& scale,
                                   uint32_t s2Align64, uint32_t dAct, uint32_t s2Base, uint32_t actScaleSrcStride,
                                   uint32_t s1Align64)
    {
        constexpr uint32_t NZ_C0 = Catlass::Gemm::Block::MXFP4::NZ_C0_ELEMS;
        constexpr uint32_t FP4_C0 = Catlass::Gemm::Block::MXFP4::FP4_C0_ELEMS;

        AscendC::LoadData2DParamsV2 loadData2DParamsB;
        loadData2DParamsB.mStartPosition = 0;
        loadData2DParamsB.kStartPosition = 0;
        loadData2DParamsB.mStep = (s2Align64 + NZ_C0 - 1) / NZ_C0;
        loadData2DParamsB.kStep = (s1Align64 + FP4_C0 - 1) / FP4_C0;
        loadData2DParamsB.srcStride = loadData2DParamsB.mStep;
        loadData2DParamsB.dstStride = s1Align64 / NZ_C0;
        loadData2DParamsB.ifTranspose = true;

        AscendC::LoadData2DMxParams load2DMxParamsB;
        load2DMxParamsB.xStartPosition = 0;
        load2DMxParamsB.yStartPosition = 0;
        load2DMxParamsB.xStep = (s1Align64 + NZ_C0 - 1) / NZ_C0;
        load2DMxParamsB.yStep = (s2Align64 + FP4_C0 - 1) / FP4_C0;
        load2DMxParamsB.srcStride = (actScaleSrcStride == 0) ? (s2Base / FP4_C0 + 1) : actScaleSrcStride;
        load2DMxParamsB.dstStride = load2DMxParamsB.yStep;

        AscendC::LoadData(dst, src, scale, loadData2DParamsB, load2DMxParamsB);
    }
};

// mx L1 → L0A 非转置搬运（QK）：L1 K(nZ, [S,D]) + K-scale(沿 D) → L0A
struct CopyL1ToL0AMxQKA5 {
    CATLASS_DEVICE
    CopyL1ToL0AMxQKA5() {}

    // nSubRowStart  : 本子块在整 tile L1 中的行起点
    // nCur          : 本子块实际 K 行数
    // alignK        : CeilAlign(D, 64)
    // s2Align16Full : 整 tile 行 16 对齐
    // scaleK        : Ceil(D/64)*2，沿 D
    template <class TensorDst, class TensorSrc, class TensorScale>
    CATLASS_DEVICE void operator()(TensorDst const& dst, TensorSrc const& src, TensorScale const& scale,
                                   uint32_t nSubRowStart, uint32_t nCur, uint32_t alignK, uint32_t s2Align16Full,
                                   uint32_t scaleK)
    {
        constexpr uint32_t NZ_C0 = Catlass::Gemm::Block::MXFP4::NZ_C0_ELEMS;
        constexpr uint32_t FP4_C0 = Catlass::Gemm::Block::MXFP4::FP4_C0_ELEMS;
        uint32_t nCurAlign16 = (nCur + NZ_C0 - 1) / NZ_C0 * NZ_C0;
        uint32_t scaleRowHalf = scaleK / 2;

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.sid = 0;
        loadDataParams.mStartPosition = nSubRowStart / NZ_C0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = nCurAlign16 / NZ_C0;
        loadDataParams.kStep = (alignK + FP4_C0 - 1) / FP4_C0;
        loadDataParams.srcStride = s2Align16Full / NZ_C0;
        loadDataParams.dstStride = nCurAlign16 / NZ_C0;
        loadDataParams.ifTranspose = false;

        AscendC::LoadData2DMxParams loadMxDataParams;
        loadMxDataParams.xStartPosition = nSubRowStart / NZ_C0;
        loadMxDataParams.yStartPosition = 0;
        loadMxDataParams.xStep = nCurAlign16 / NZ_C0;
        loadMxDataParams.yStep = scaleRowHalf;
        loadMxDataParams.srcStride = scaleRowHalf;
        loadMxDataParams.dstStride = scaleRowHalf;

        AscendC::LoadData(dst, src, scale, loadDataParams, loadMxDataParams);
    }
};

// mx L1 → L0B 非转置搬运（QK）：L1 Q(nZ, [M,D]) + Q-scale(沿 D) → L0B
struct CopyL1ToL0BMxQKA5 {
    CATLASS_DEVICE
    CopyL1ToL0BMxQKA5() {}

    // mAlignL1      : M 维 L1 对齐（恒 128）
    // alignK        : CeilAlign(D, 64)
    // mAlignL0      : M 维 L0B 对齐
    // scaleMAlignL1 : scale 的 M 维 L1 对齐
    // scaleK        : Ceil(D/64)*2，沿 D
    template <class TensorDst, class TensorSrc, class TensorScale>
    CATLASS_DEVICE void operator()(TensorDst const& dst, TensorSrc const& src, TensorScale const& scale,
                                   uint32_t mAlignL1, uint32_t alignK, uint32_t mAlignL0, uint32_t scaleMAlignL1,
                                   uint32_t scaleK)
    {
        constexpr uint32_t NZ_C0 = Catlass::Gemm::Block::MXFP4::NZ_C0_ELEMS;
        constexpr uint32_t FP4_C0 = Catlass::Gemm::Block::MXFP4::FP4_C0_ELEMS;
        uint32_t scaleRowHalf = scaleK / 2;

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.sid = 0;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = (mAlignL1 + NZ_C0 - 1) / NZ_C0;
        loadDataParams.kStep = (alignK + FP4_C0 - 1) / FP4_C0;
        loadDataParams.srcStride = (mAlignL1 + NZ_C0 - 1) / NZ_C0;
        loadDataParams.dstStride = (mAlignL0 + NZ_C0 - 1) / NZ_C0;
        loadDataParams.ifTranspose = false;

        AscendC::LoadData2DMxParams loadMxDataParams;
        loadMxDataParams.xStartPosition = 0;
        loadMxDataParams.yStartPosition = 0;
        loadMxDataParams.xStep = (scaleMAlignL1 + NZ_C0 - 1) / NZ_C0;
        loadMxDataParams.yStep = scaleRowHalf;
        loadMxDataParams.srcStride = scaleRowHalf;
        loadMxDataParams.dstStride = scaleRowHalf;

        AscendC::LoadData(dst, src, scale, loadDataParams, loadMxDataParams);
    }
};

} // namespace Catlass::Gemm::Tile

#endif // GEMM_TILE_COPY_L1_TO_L0_MX_HPP
