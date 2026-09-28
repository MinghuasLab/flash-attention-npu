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

#ifndef GEMM_TILE_COPY_GM_TO_L1_MX_SCALE_DN2NZ_HPP
#define GEMM_TILE_COPY_GM_TO_L1_MX_SCALE_DN2NZ_HPP

#include "catlass/numeric_size.hpp"
#include "../../gemm/block/block_mmad_utils.hpp"

namespace Catlass::Gemm::Tile {

// QK 专用：Host 沿 D 的 E8M0 scale GM → L1（Dn2Nz）
// Host 布局 [S, D//64, 2]，每 token scaleK = Ceil(D/64)*2 字节；
// 与 PV 的 MxScaleA(ColumnMajor) → zZ（Nd2Nz、沿 S）完全不同，禁止混用。

struct CopyGmToL1MxScaleDn2NzA5 {
    CATLASS_DEVICE
    CopyGmToL1MxScaleDn2NzA5()
    {}

    // rows      : 本次搬的 token 行数
    // scaleK    : Ceil(D/64)*2
    // srcB16Off : 源 half 偏移（稀疏多段拼接时使用，稠密传 0）
    // dstB16Off : 目的 half 偏移
    // headMul   : 头内 token 步长因子（BNSD=1, BSND/TND=N）
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<uint8_t> const& l1Scale, AscendC::GlobalTensor<uint8_t> const& gScale, uint32_t rows,
        uint32_t scaleK, uint32_t srcB16Off = 0, uint32_t dstB16Off = 0, uint32_t headMul = 1)
    {
        uint32_t scaleRowHalf = scaleK / 2;

        AscendC::GlobalTensor<half> gScaleB16;
        gScaleB16.SetGlobalBuffer((__gm__ half*)(gScale.GetPhyAddr()), Catlass::Gemm::Block::MXFP4::GM_MAX_BUFFER_LEN);
        auto l1ScaleB16 = l1Scale.ReinterpretCast<half>();

        AscendC::Dn2NzParams dn2nzParams;
        dn2nzParams.dnNum = 1;
        dn2nzParams.dValue = rows;
        dn2nzParams.nValue = scaleRowHalf;
        dn2nzParams.srcDnMatrixStride = 0;
        dn2nzParams.srcDValue = headMul * scaleRowHalf;
        dn2nzParams.dstNzC0Stride = scaleRowHalf;
        dn2nzParams.dstNzNStride = 1;
        dn2nzParams.dstNzMatrixStride = 0;

        AscendC::DataCopy(l1ScaleB16[dstB16Off], gScaleB16[srcB16Off], dn2nzParams);
    }
};

} // namespace Catlass::Gemm::Tile

#endif // GEMM_TILE_COPY_GM_TO_L1_MX_SCALE_DN2NZ_HPP
