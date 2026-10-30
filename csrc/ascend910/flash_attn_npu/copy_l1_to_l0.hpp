/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef COPY_L1_TO_L0_HPP
#define COPY_L1_TO_L0_HPP

#include "catlass/catlass.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

using namespace AscendC;
using namespace KernelCommon;

namespace Catlass::Gemm::Tile {
template <class ArchTag, typename Element>
struct CopyL1ToL0AImpl {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;

    CATLASS_DEVICE
    void operator()(LocalTensor<Element> const& dstTensor, LocalTensor<Element> const& srcTensor,
                    LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
        uint16_t l1M = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint16_t l1K = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint16_t l0M = layoutDst.shape(0) * layoutDst.shape(1);
        uint16_t l0K = layoutDst.shape(2) * layoutDst.shape(3);
        SetFmatrix(1, l1M, PAD_LIST, FmatrixMode::FMATRIX_LEFT);
        static constexpr IsResetLoad3dConfig config = {false, false};
        LoadData3DParamsV2<Element> loadDataParams;
        loadDataParams.kExtension = l0K;
        loadDataParams.mExtension = l0M;
        loadDataParams.channelSize = l1K;

        LoadData<Element, config>(dstTensor, srcTensor, loadDataParams);
    }
};

template <class ArchTag, typename Element>
struct CopyL1ToL0BImpl {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    CATLASS_DEVICE
    void operator()(LocalTensor<Element> const& dstTensor, LocalTensor<Element> const& srcTensor,
                    LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
        uint16_t l1K = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint16_t l1N = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint16_t l0K = layoutDst.shape(0) * layoutDst.shape(1);
        uint16_t l0N = layoutDst.shape(2) * layoutDst.shape(3);
        // K, N need to be 16 aligned
        uint16_t l1KAlign = RoundUp<C0_NUM_PER_FRACTAL>(l1K);
        uint16_t l1NAlign = RoundUp<C0_NUM_PER_FRACTAL>(l1N);
        uint16_t l0KAlign = RoundUp<C0_NUM_PER_FRACTAL>(l0K);
        uint16_t l0NAlign = RoundUp<C0_NUM_PER_FRACTAL>(l0N);
        SetFmatrix(1, l1KAlign, PAD_LIST, FmatrixMode::FMATRIX_RIGHT);
        static constexpr IsResetLoad3dConfig config = {false, false};
        LoadData3DParamsV2<Element> loadDataParams;
        loadDataParams.kExtension = l0NAlign;
        loadDataParams.mExtension = l0KAlign;
        loadDataParams.channelSize = l1NAlign;
        loadDataParams.fMatrixCtrl = true;

        LoadData<Element, config>(dstTensor, srcTensor, loadDataParams);
    }
};

template <class ArchTag>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<half, layout::zN, TPosition::A1>> : CopyL1ToL0AImpl<ArchTag, half> {};

template <class ArchTag>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<bfloat16_t, layout::zN, TPosition::A1>>
    : CopyL1ToL0AImpl<ArchTag, bfloat16_t> {};

template <class ArchTag>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<half, layout::zN, TPosition::A1>> : CopyL1ToL0BImpl<ArchTag, half> {};

template <class ArchTag>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<bfloat16_t, layout::zN, TPosition::A1>>
    : CopyL1ToL0BImpl<ArchTag, bfloat16_t> {};

} // namespace Catlass::Gemm::Tile

#endif // COPY_L1_TO_L0_HPP
