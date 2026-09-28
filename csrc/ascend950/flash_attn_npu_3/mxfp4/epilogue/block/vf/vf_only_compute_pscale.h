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
 * \file vf_only_compute_pscale.h
 * \brief
 */

#ifndef VF_ONLY_COMPUTE_PSCALE_H_
#define VF_ONLY_COMPUTE_PSCALE_H_
#include "kernel_tensor.h"
#include "vf_common_def.h"
namespace Catlass::Epilogue::Block::Mxfp4VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace Reg;

#define VMULSCVT false
#define DROPOUT false

// T: half
template <bool clear_gmax, typename T, uint16_t m = 128>
__simd_vf__ inline void compute_only_pscale_vf(
    __ubuf__ T* umax1, __ubuf__ T* umax2, __ubuf__ T* ugmax_old, __ubuf__ float* urs, uint16_t coreIndex)
{
    MaskReg preg_all16 = CreateMask<uint16_t, MaskPattern::ALL>();

    RegTensor<half> vreg_gmax, vreg_max1, vreg_max2, vreg_rs16, vreg_gmax_old;
    RegTensor<float> vreg_rs1, vreg_rs2;

    if constexpr (clear_gmax) {
        LoadAlign(vreg_max1, umax1);
        LoadAlign(vreg_max2, umax2);

        Max(vreg_gmax, vreg_max1, vreg_max2, preg_all16);
        StoreAlign(ugmax_old, vreg_gmax, preg_all16);
    } else {
        LoadAlign(vreg_max1, umax1);
        LoadAlign(vreg_max2, umax2);
        LoadAlign(vreg_gmax_old, ugmax_old);

        Max(vreg_gmax, vreg_max1, vreg_max2, preg_all16);
        Max(vreg_gmax, vreg_gmax, vreg_gmax_old, preg_all16);
        StoreAlign(ugmax_old, vreg_gmax, preg_all16);

        Sub(vreg_rs16, vreg_gmax_old, vreg_gmax, preg_all16);
        Adds(vreg_rs16, vreg_rs16, NUM_127, preg_all16);
        Maxs(vreg_rs16, vreg_rs16, ZERO_VALUE, preg_all16);
        Cast<int32_t, T, h2iZero>((RegTensor<int32_t>&)vreg_rs1, vreg_rs16, preg_all16);
        Cast<int32_t, T, h2iOne>((RegTensor<int32_t>&)vreg_rs2, vreg_rs16, preg_all16);
        ShiftLefts((RegTensor<int32_t>&)vreg_rs1, (RegTensor<int32_t>&)vreg_rs1, SHIFT_VALUE, preg_all16);
        ShiftLefts((RegTensor<int32_t>&)vreg_rs2, (RegTensor<int32_t>&)vreg_rs2, SHIFT_VALUE, preg_all16);

        Interleave(
            (RegTensor<int32_t>&)vreg_rs1, (RegTensor<int32_t>&)vreg_rs2, (RegTensor<int32_t>&)vreg_rs1,
            (RegTensor<int32_t>&)vreg_rs2);
        for (uint16_t i = 0; i < 1 - coreIndex; ++i) {
            StoreAlign(urs, vreg_rs1, preg_all16);
        }

        for (uint16_t i = 0; i < coreIndex; ++i) {
            StoreAlign(urs, vreg_rs2, preg_all16);
        }
    }
}

template <bool clear_gmax, typename T, uint16_t QsBase = 128>
__aicore__ inline void ComputeOnlyPscaleCallVF(
    const LocalTensor<T>& umax1, const LocalTensor<T>& umax2, const LocalTensor<T>& ugmaxOld,
    const LocalTensor<float>& urs, uint16_t coreIndex)
{
    __ubuf__ T* umax1_buf = (__ubuf__ T*)umax1.GetPhyAddr();
    __ubuf__ T* umax2_buf = (__ubuf__ T*)umax2.GetPhyAddr();
    __ubuf__ T* ugmax_old_buf = (__ubuf__ T*)ugmaxOld.GetPhyAddr();
    __ubuf__ float* urs_buf = (__ubuf__ float*)urs.GetPhyAddr();

    compute_only_pscale_vf<clear_gmax, T, QsBase>(umax1_buf, umax2_buf, ugmax_old_buf, urs_buf, coreIndex);
}
} // namespace Catlass::Epilogue::Block::Mxfp4VF
#endif // VF_ONLY_COMPUTE_PSCALE_H_
