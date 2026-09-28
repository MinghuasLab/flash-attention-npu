/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* Modified by Minghua Shen, 2026. */

/*!
 * \file vf_attn_out_transpose.h
 * \brief
 */

#ifndef VF_ATTN_OUT_TRANSPOSE_H_
#define VF_ATTN_OUT_TRANSPOSE_H_
#include "kernel_tensor.h"
#include "vf_common_def.h"
namespace Catlass::Epilogue::Block::Mxfp4VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace Reg;

#define VMULSCVT false
#define DROPOUT false

template <uint16_t QsBase = 128, typename Otype = bfloat16_t>
__simd_vf__ void transpose_attn_out_vf(__ubuf__ float* atten_out_md, __ubuf__ float* atten_out_dm, __ubuf__ float* rsum)
{
    MaskReg preg_all32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all16 = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<float> vreg_rrsum, vreg_ones;
    Duplicate(vreg_ones, (float)1.0);
    LoadAlign(vreg_rrsum, (__ubuf__ float*)rsum);
    Div(vreg_rrsum, vreg_ones, vreg_rrsum, preg_all32);

    for (uint16_t i = 0; i < QsBase / 16; i++) {
        RegTensor<float> vreg_src1, vreg_src2, vreg_src3, vreg_src4, vreg_src5, vreg_src6, vreg_src7, vreg_src8;
        RegTensor<float> vreg_src9, vreg_src10, vreg_src11, vreg_src12, vreg_src13, vreg_src14, vreg_src15, vreg_src16;
        RegTensor<Otype> vreg_bf16_1, vreg_bf16_2;

        LoadAlign(vreg_src1, (__ubuf__ float*)atten_out_dm + i * 64 * 16 + 64 * 0);
        LoadAlign(vreg_src2, (__ubuf__ float*)atten_out_dm + i * 64 * 16 + 64 * 1);
        LoadAlign(vreg_src3, (__ubuf__ float*)atten_out_dm + i * 64 * 16 + 64 * 2);
        LoadAlign(vreg_src4, (__ubuf__ float*)atten_out_dm + i * 64 * 16 + 64 * 3);
        LoadAlign(vreg_src5, (__ubuf__ float*)atten_out_dm + i * 64 * 16 + 64 * 4);
        LoadAlign(vreg_src6, (__ubuf__ float*)atten_out_dm + i * 64 * 16 + 64 * 5);
        LoadAlign(vreg_src7, (__ubuf__ float*)atten_out_dm + i * 64 * 16 + 64 * 6);
        LoadAlign(vreg_src8, (__ubuf__ float*)atten_out_dm + i * 64 * 16 + 64 * 7);
        LoadAlign(vreg_src9, (__ubuf__ float*)atten_out_dm + i * 64 * 16 + 64 * 8);
        LoadAlign(vreg_src10, (__ubuf__ float*)atten_out_dm + i * 64 * 16 + 64 * 9);
        LoadAlign(vreg_src11, (__ubuf__ float*)atten_out_dm + i * 64 * 16 + 64 * 10);
        LoadAlign(vreg_src12, (__ubuf__ float*)atten_out_dm + i * 64 * 16 + 64 * 11);
        LoadAlign(vreg_src13, (__ubuf__ float*)atten_out_dm + i * 64 * 16 + 64 * 12);
        LoadAlign(vreg_src14, (__ubuf__ float*)atten_out_dm + i * 64 * 16 + 64 * 13);
        LoadAlign(vreg_src15, (__ubuf__ float*)atten_out_dm + i * 64 * 16 + 64 * 14);
        LoadAlign(vreg_src16, (__ubuf__ float*)atten_out_dm + i * 64 * 16 + 64 * 15);

        Mul(vreg_src1, vreg_src1, vreg_rrsum, preg_all32);
        Mul(vreg_src2, vreg_src2, vreg_rrsum, preg_all32);
        Mul(vreg_src3, vreg_src3, vreg_rrsum, preg_all32);
        Mul(vreg_src4, vreg_src4, vreg_rrsum, preg_all32);
        Mul(vreg_src5, vreg_src5, vreg_rrsum, preg_all32);
        Mul(vreg_src6, vreg_src6, vreg_rrsum, preg_all32);
        Mul(vreg_src7, vreg_src7, vreg_rrsum, preg_all32);
        Mul(vreg_src8, vreg_src8, vreg_rrsum, preg_all32);
        Mul(vreg_src9, vreg_src9, vreg_rrsum, preg_all32);
        Mul(vreg_src10, vreg_src10, vreg_rrsum, preg_all32);
        Mul(vreg_src11, vreg_src11, vreg_rrsum, preg_all32);
        Mul(vreg_src12, vreg_src12, vreg_rrsum, preg_all32);
        Mul(vreg_src13, vreg_src13, vreg_rrsum, preg_all32);
        Mul(vreg_src14, vreg_src14, vreg_rrsum, preg_all32);
        Mul(vreg_src15, vreg_src15, vreg_rrsum, preg_all32);
        Mul(vreg_src16, vreg_src16, vreg_rrsum, preg_all32);

        Cast<Otype, float, castTraitZero>(vreg_bf16_1, vreg_src1, preg_all32);
        Cast<Otype, float, castTraitOne>(vreg_bf16_2, vreg_src2, preg_all32);
        Or((RegTensor<uint16_t>&)vreg_src1, (RegTensor<uint16_t>&)vreg_bf16_1, (RegTensor<uint16_t>&)vreg_bf16_2,
           preg_all16);

        Cast<Otype, float, castTraitZero>(vreg_bf16_1, vreg_src3, preg_all32);
        Cast<Otype, float, castTraitOne>(vreg_bf16_2, vreg_src4, preg_all32);
        Or((RegTensor<uint16_t>&)vreg_src2, (RegTensor<uint16_t>&)vreg_bf16_1, (RegTensor<uint16_t>&)vreg_bf16_2,
           preg_all16);

        Cast<Otype, float, castTraitZero>(vreg_bf16_1, vreg_src5, preg_all32);
        Cast<Otype, float, castTraitOne>(vreg_bf16_2, vreg_src6, preg_all32);
        Or((RegTensor<uint16_t>&)vreg_src3, (RegTensor<uint16_t>&)vreg_bf16_1, (RegTensor<uint16_t>&)vreg_bf16_2,
           preg_all16);

        Cast<Otype, float, castTraitZero>(vreg_bf16_1, vreg_src7, preg_all32);
        Cast<Otype, float, castTraitOne>(vreg_bf16_2, vreg_src8, preg_all32);
        Or((RegTensor<uint16_t>&)vreg_src4, (RegTensor<uint16_t>&)vreg_bf16_1, (RegTensor<uint16_t>&)vreg_bf16_2,
           preg_all16);

        Cast<Otype, float, castTraitZero>(vreg_bf16_1, vreg_src9, preg_all32);
        Cast<Otype, float, castTraitOne>(vreg_bf16_2, vreg_src10, preg_all32);
        Or((RegTensor<uint16_t>&)vreg_src5, (RegTensor<uint16_t>&)vreg_bf16_1, (RegTensor<uint16_t>&)vreg_bf16_2,
           preg_all16);

        Cast<Otype, float, castTraitZero>(vreg_bf16_1, vreg_src11, preg_all32);
        Cast<Otype, float, castTraitOne>(vreg_bf16_2, vreg_src12, preg_all32);
        Or((RegTensor<uint16_t>&)vreg_src6, (RegTensor<uint16_t>&)vreg_bf16_1, (RegTensor<uint16_t>&)vreg_bf16_2,
           preg_all16);

        Cast<Otype, float, castTraitZero>(vreg_bf16_1, vreg_src13, preg_all32);
        Cast<Otype, float, castTraitOne>(vreg_bf16_2, vreg_src14, preg_all32);
        Or((RegTensor<uint16_t>&)vreg_src7, (RegTensor<uint16_t>&)vreg_bf16_1, (RegTensor<uint16_t>&)vreg_bf16_2,
           preg_all16);

        Cast<Otype, float, castTraitZero>(vreg_bf16_1, vreg_src15, preg_all32);
        Cast<Otype, float, castTraitOne>(vreg_bf16_2, vreg_src16, preg_all32);
        Or((RegTensor<uint16_t>&)vreg_src8, (RegTensor<uint16_t>&)vreg_bf16_1, (RegTensor<uint16_t>&)vreg_bf16_2,
           preg_all16);

        Interleave(vreg_src1, vreg_src5, vreg_src1, vreg_src5);
        Interleave(vreg_src2, vreg_src6, vreg_src2, vreg_src6);
        Interleave(vreg_src3, vreg_src7, vreg_src3, vreg_src7);
        Interleave(vreg_src4, vreg_src8, vreg_src4, vreg_src8);

        Interleave(vreg_src1, vreg_src3, vreg_src1, vreg_src3);
        Interleave(vreg_src5, vreg_src7, vreg_src5, vreg_src7);
        Interleave(vreg_src2, vreg_src4, vreg_src2, vreg_src4);
        Interleave(vreg_src6, vreg_src8, vreg_src6, vreg_src8);

        Interleave(vreg_src1, vreg_src2, vreg_src1, vreg_src2);
        Interleave(vreg_src3, vreg_src4, vreg_src3, vreg_src4);
        Interleave(vreg_src5, vreg_src6, vreg_src5, vreg_src6);
        Interleave(vreg_src7, vreg_src8, vreg_src7, vreg_src8);

        StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
            (__ubuf__ float*)atten_out_md + 8 * i + (8 + 1) * 8 * 8 * 0, (RegTensor<float>&)vreg_src1, 8 + 1,
            preg_all32);
        StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
            (__ubuf__ float*)atten_out_md + 8 * i + (8 + 1) * 8 * 8 * 1, (RegTensor<float>&)vreg_src2, 8 + 1,
            preg_all32);
        StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
            (__ubuf__ float*)atten_out_md + 8 * i + (8 + 1) * 8 * 8 * 2, (RegTensor<float>&)vreg_src3, 8 + 1,
            preg_all32);
        StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
            (__ubuf__ float*)atten_out_md + 8 * i + (8 + 1) * 8 * 8 * 3, (RegTensor<float>&)vreg_src4, 8 + 1,
            preg_all32);
        StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
            (__ubuf__ float*)atten_out_md + 8 * i + (8 + 1) * 8 * 8 * 4, (RegTensor<float>&)vreg_src5, 8 + 1,
            preg_all32);
        StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
            (__ubuf__ float*)atten_out_md + 8 * i + (8 + 1) * 8 * 8 * 5, (RegTensor<float>&)vreg_src6, 8 + 1,
            preg_all32);
        StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
            (__ubuf__ float*)atten_out_md + 8 * i + (8 + 1) * 8 * 8 * 6, (RegTensor<float>&)vreg_src7, 8 + 1,
            preg_all32);
        StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
            (__ubuf__ float*)atten_out_md + 8 * i + (8 + 1) * 8 * 8 * 7, (RegTensor<float>&)vreg_src8, 8 + 1,
            preg_all32);
    }
}

template <uint16_t QsBase = 128, typename Otype = bfloat16_t>
__aicore__ inline void TransposeAttnOutCallVF(
    const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const LocalTensor<float>& rSum)
{
    __ubuf__ float* atten_out_md = (__ubuf__ float*)dstTensor.GetPhyAddr();
    __ubuf__ float* atten_out_dm = (__ubuf__ float*)srcTensor.GetPhyAddr();
    __ubuf__ float* rsum = (__ubuf__ float*)rSum.GetPhyAddr();

    transpose_attn_out_vf<QsBase, Otype>(atten_out_md, atten_out_dm, rsum);
}

} // namespace Catlass::Epilogue::Block::Mxfp4VF
#endif // VF_ATTN_OUT_TRANSPOSE_H_
