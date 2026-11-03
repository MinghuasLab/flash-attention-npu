/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../fwd_kernel.cpp"

template <class InDtype, class SMDtype, Format qFormat, Format kvFormat, CacheMode kvcacheType, PageShape kvcacheShape,
          MaskCategory maskCategory, CacheLayout cacheLayout, bool LseMode, bool HasSoftcap>
CATLASS_GLOBAL void FAInferD512(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o,
                                GM_ADDR lse, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR seqUsedQ,
                                GM_ADDR seqUsedKv, GM_ADDR workspace, GM_ADDR tiling, float scaleValue,
                                float softcapValue, uint32_t numBlocks, uint32_t maxNumBlocksPerBatch,
                                int64_t kvSeqlenAllocated)
{
    using ArchTag = Arch::Ascend950;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    static_assert(std::is_same_v<SMDtype, float>);
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = float;
    using ElementMask = uint8_t;
    // layout tags
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    // To cater to the existing TileCopy class, a dummy rowMajor is used
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    // OTmp is rowMajor on UB(dst)
    using LayoutOTmp = layout::RowMajor;
    // mask
    using LayoutMask = layout::RowMajor;
    // 处理单个tile内Q和K的matmul
    using L1TileShapeQK = Shape<Int<Q_BLK>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK =
        Gemm::MmadFlashAttentionQK<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, false>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
                                                         LayoutS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadQK = Gemm::Block::QkD512<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                            ElementS, void, TileCopyQK>;
    // Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueFAOnlineSoftmaxT<HasSoftcap>;
    using TileCopySoftmax = Epilogue::Tile::TileCopySoftmax<ArchTag, ElementMask, ElementP, LayoutMask, LayoutPDummy>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax =
        Epilogue::Block::D512Softmax<DispatchPolicyOnlineSoftmax, PType, SType, MaskType, TileCopySoftmax>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV =
        Gemm::MmadFlashAttentionPV<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, false>;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadPV = Gemm::Block::PvD512<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                            ElementOTmp, void, TileCopyPV>;
    // rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueFARescaleO;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::D512Rescale<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS,
                                                          TileCopyRescaleO, Arch::PositionL0C>;

    using Kernel = FAIKernel950<BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, qFormat, kvFormat,
                                kvcacheType, kvcacheShape, maskCategory, cacheLayout, false, LseMode, true>;
    FAIKernelParams params{q,
                           k,
                           v,
                           mask,
                           blockTables,
                           actualQseqlen,
                           actualKvseqlen,
                           seqUsedQ,
                           seqUsedKv,
                           o,
                           lse,
                           workspace,
                           tiling,
                           scaleValue,
                           softcapValue,
                           numBlocks,
                           maxNumBlocksPerBatch,
                           kvSeqlenAllocated};
    Kernel faInfer;
    faInfer(params);
}
