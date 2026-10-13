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

#ifndef GEMM_TILE_COPY_GM_TO_L1_HPP
#define GEMM_TILE_COPY_GM_TO_L1_HPP

#include "catlass/numeric_size.hpp"
#include "catlass/arch/arch.hpp"
#include "../../gemm/tile_common/tile_copy_tla.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Tile {

/// Partial specialization for CopyGmToL1, AtlasA5, RowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Catlass::Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 =
        Catlass::BytesToBits(Catlass::BYTE_PER_C0) / Catlass::SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint32_t ndNum = 1,
                                   uint32_t srcNdMatrixStride = 0, uint32_t dstNzMatrixStride = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and RowMajor, while TensorDst must be L1 and zN");

        const uint32_t nValue = tla::get<0>(srcTensor.shape());
        const uint32_t dValue = tla::get<1>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<0>(srcTensor.stride());
        const uint32_t dstInnerStrideRow = tla::get<0, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = ndNum;
        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        // [fp4 改造] 镜像 catlass 官方 copy_gm_to_l1.hpp(Ascend950) 的 float4 分支：
        //   fp4x2_e2m1_t 的 tla element 是字节(sizeof=1)，但 GM 张量按 fp4 元素(4-bit)索引、
        //   Nd2Nz 的 dValue/srcDValue 需按 fp4x2(字节)单位。layout shape 是 fp4 元素单位(D)，
        //   故对 fp4 源把 dValue/srcDValue CeilDiv/2 还原成 fp4x2 字节单位。
        if constexpr (AscendC::IsSameType<ElementSrc, fp4x2_e2m1_t>::value) {
            intriParams.dValue = CeilDiv(intriParams.dValue, 2u);
        }
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.srcDValue = srcDValue;
        if constexpr (AscendC::IsSameType<ElementSrc, fp4x2_e2m1_t>::value) {
            intriParams.srcDValue = CeilDiv(intriParams.srcDValue, 2u);
        }
        intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = dstInnerStrideRow / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = dstNzMatrixStride;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for CopyGmToL1, AtlasA5, ColumnMajor in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Catlass::Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 =
        Catlass::BytesToBits(Catlass::BYTE_PER_C0) / Catlass::SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint32_t ndNum = 1,
                                   uint32_t srcNdMatrixStride = 0, uint32_t dstNzMatrixStride = 0)
    {
        static_assert(tla::detail::isColumnMajor<typename TensorSrc::Layout>::value &&
                          tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                          TensorSrc::position == AscendC::TPosition::GM &&
                          TensorDst::position == AscendC::TPosition::A1,
                      "The input parameters do not match. TensorSrc must be GM and ColumnMajor, "
                      "while TensorDst must be L1 and nZ");

        const uint32_t nValue = tla::get<1>(srcTensor.shape());
        const uint32_t dValue = tla::get<0>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<1>(srcTensor.stride());
        const uint32_t dstInnerStrideCol = tla::get<1, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = ndNum;
        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        // [fp4 改造] 镜像 catlass 官方 copy_gm_to_l1.hpp(Ascend950) 的 float4 分支（ColumnMajor→nZ 路径）。
        //   V 用此路径：GM ColumnMajor(D,S2) shape 为 fp4 元素单位，Nd2Nz dValue/srcDValue 需 fp4x2 字节单位，
        //   故对 fp4 源 CeilDiv/2。这样 dValue=D/2、srcDValue=D/2，与原 raw gather 的 aColValue=D/2 一致。
        if constexpr (AscendC::IsSameType<ElementSrc, fp4x2_e2m1_t>::value) {
            intriParams.dValue = CeilDiv(intriParams.dValue, 2u);
        }
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.srcDValue = srcDValue;
        if constexpr (AscendC::IsSameType<ElementSrc, fp4x2_e2m1_t>::value) {
            intriParams.srcDValue = CeilDiv(intriParams.srcDValue, 2u);
        }
        intriParams.dstNzC0Stride = dstOuterStrideRow / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = dstInnerStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = dstNzMatrixStride;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// [mx-scale 移植] fp8_e8m0 V-scale GM->L1：MxScaleA(ColumnMajor, trans) -> zZ。
/// 移植自 catlass 官方 copy_gm_to_l1.hpp 的 MxScaleForColumnMajorA->zZ（Ascend950），用 MXFP4 的
/// isMxScaleForColumnMajorA / isMxScaleForzZ。
/// V 是 A 操作数、GM ColumnMajor、L1->L0A 转置 -> V-scale 走 A-scale 的 trans 变体。
/// Nd2Nz 参数取自 memory shape()/stride()（originShape 是逻辑尺寸，不参与寻址）：
///   MxScaleA ColumnMajor 布局 shape=(rows,(2,CeilDiv(cols,2)))、stride=(2,(1,rows*2))，故：
///     nValue    = get<1,1>(shape) (= CeilDiv(cols,2) = scaleRows)
///     dValue    = get<0>(shape)  (= rows = D)
///     srcDValue = CeilDiv(get<1,1>(stride), MX_SCALE_COPY_GROUP_NUM=2) (= rows*2/2 = D)
///     dstNzC0Stride = get<0,1>(dstStride)/BYTE_PER_C0, dstNzNStride = 1
/// 全部与原 raw SparseVScaleBaseTileL1FullLoad 的 Nd2Nz 对齐（待上机验证）。
template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Catlass::Arch::Ascend950,
                   tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
                   tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
                   std::enable_if_t<tla::detail::isMxScaleForColumnMajorA<ElementMx, LayoutSrc>::value &&
                                    tla::detail::isMxScaleForzZ<ElementMx, LayoutDst>::value>> {
    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint32_t ndNum = 1,
                                   uint32_t srcNdMatrixStride = 0, uint32_t dstNzMatrixStride = 0)
    {
        static_assert(
            tla::detail::isMxScaleForColumnMajorA<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::isMxScaleForzZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "MxScaleA(ColumnMajor,trans) GM -> zZ L1: src must be isMxScaleForColumnMajorA/GM, dst isMxScaleForzZ/A1");

        // MxScaleA shape=(rows,(2,CeilDiv(cols,2))) -> 直接取 fractal 字段
        const uint32_t nValue = tla::get<1, 1>(srcTensor.shape()); // = CeilDiv(cols,2)
        const uint32_t dValue = tla::get<0>(srcTensor.shape());    // = rows
        const uint32_t srcDValue = CeilDiv(tla::get<1, 1>(srcTensor.stride()),
                                           Catlass::MX_SCALE_COPY_GROUP_NUM); // = rows*2/2
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());
        const uint32_t dstNzC0Stride = dstOuterStrideRow / Catlass::BYTE_PER_C0;

        AscendC::Nd2NzParams intriParams;
        intriParams.ndNum = ndNum;
        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.srcDValue = srcDValue;
        intriParams.dstNzC0Stride = dstNzC0Stride;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = dstNzMatrixStride;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // fp8_e8m0 scale 按 B16(half) 视图 Nd2Nz（2 fp8/B16），与 raw V-scale gather 一致。
        auto srcHalf = srcTensor.data()[srcOffset].template ReinterpretCast<half>();
        auto dstHalf = dstTensor.data()[dstOffset].template ReinterpretCast<half>();
        AscendC::DataCopy(dstHalf, srcHalf, intriParams);
    }
};

} // namespace Catlass::Gemm::Tile

#endif // GEMM_TILE_COPY_GM_TO_L1_HPP
