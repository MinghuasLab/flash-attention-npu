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
 * \file vf_compute_pscale_dm_qs64.h
 * \brief
 */

#ifndef VF_COMPUTE_PSCALE_DM_QS64_H_
#define VF_COMPUTE_PSCALE_DM_QS64_H_
#include "kernel_tensor.h"
#include "vf_common_def.h"
#include "../../epilogue_dispatch_policy.hpp"
namespace Catlass::Epilogue::Block::Mxfp4VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace Reg;

#define VMULSCVT false
#define DROPOUT false

// T: half
template <MXQuantMode MX_QUANT_MODE = MXQuantMode::OCP, bool clear_gmax, typename T, uint16_t m = 128>
__simd_vf__ inline void compute_pscale_qs64_vf(
    __ubuf__ uint8_t* mxscale1, __ubuf__ uint8_t* mxscale2, __ubuf__ T* ulmax1, __ubuf__ T* ulmax2, __ubuf__ T* umax1,
    __ubuf__ T* umax2, __ubuf__ T* ugmax_old, __ubuf__ float* urs, uint16_t firstLoop, uint16_t secondLoop,
    uint16_t coreIndex, const T LOG2_CX_CEIL)
{
    MaskReg preg_all_32 = CreateMask<uint32_t, MaskPattern::ALL>();
    MaskReg preg_VL32_32 = CreateMask<uint32_t, MaskPattern::VL32>();
    MaskReg preg_all16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_VL64_16 = CreateMask<uint16_t, MaskPattern::VL64>();
    MaskReg preg_VL128_8 = CreateMask<uint8_t, MaskPattern::VL128>();

    RegTensor<half> vreg_gmax, vreg_max1, vreg_max2, vreg_rs16, vreg_gmax_old;
    RegTensor<float> vreg_rs1, vreg_rs2;

    constexpr uint16_t ulmaxLoopRow = m * 8;

    if constexpr (clear_gmax) {
        LoadAlign(vreg_max1, umax1);
        LoadAlign(vreg_max2, umax2);

        Max(vreg_gmax, vreg_max1, vreg_max2, preg_VL64_16);
        StoreAlign(ugmax_old, vreg_gmax, preg_VL64_16);
    } else {
        LoadAlign(vreg_max1, umax1);
        LoadAlign(vreg_max2, umax2);
        LoadAlign(vreg_gmax_old, ugmax_old);

        Max(vreg_gmax, vreg_max1, vreg_max2, preg_VL64_16);
        Max(vreg_gmax, vreg_gmax, vreg_gmax_old, preg_VL64_16);
        StoreAlign(ugmax_old, vreg_gmax, preg_VL64_16);

        Sub(vreg_rs16, vreg_gmax_old, vreg_gmax, preg_VL64_16);
        Adds(vreg_rs16, vreg_rs16, NUM_127, preg_VL64_16);
        Maxs(vreg_rs16, vreg_rs16, ZERO_VALUE, preg_VL64_16);
        Cast<int32_t, T, h2iZero>((RegTensor<int32_t>&)vreg_rs1, vreg_rs16, preg_VL64_16);
        Cast<int32_t, T, h2iOne>((RegTensor<int32_t>&)vreg_rs2, vreg_rs16, preg_VL64_16);
        ShiftLefts((RegTensor<int32_t>&)vreg_rs1, (RegTensor<int32_t>&)vreg_rs1, SHIFT_VALUE, preg_VL32_32);
        ShiftLefts((RegTensor<int32_t>&)vreg_rs2, (RegTensor<int32_t>&)vreg_rs2, SHIFT_VALUE, preg_VL32_32);

        Interleave(
            (RegTensor<int32_t>&)vreg_rs1, (RegTensor<int32_t>&)vreg_rs2, (RegTensor<int32_t>&)vreg_rs1,
            (RegTensor<int32_t>&)vreg_rs2);
        for (uint16_t i = 0; i < 1 - coreIndex; ++i) {
            StoreAlign(urs, vreg_rs1, preg_all_32);
        }

        for (uint16_t i = 0; i < coreIndex; ++i) {
            StoreAlign(urs, vreg_rs2, preg_all_32);
        }
    }

    Duplicate(vreg_max1, MIN_VALUE);
    StoreAlign(umax1, vreg_max1, preg_all16);
    if constexpr (MX_QUANT_MODE == MXQuantMode::OCP) {
        Adds(vreg_gmax, vreg_gmax, NUM_NEG_125, preg_VL64_16);
    } else {
        Adds(vreg_gmax, vreg_gmax, NUM_NEG_127, preg_VL64_16);
        Adds(vreg_gmax, vreg_gmax, LOG2_CX_CEIL, preg_VL64_16);
    }

    for (uint16_t i = 0; i < firstLoop; i++) {
        RegTensor<half> vreg_scale1, vreg_scale2, vreg_scale3, vreg_scale4, vreg_scale5, vreg_scale6, vreg_scale7,
            vreg_scale8;
        LoadAlign(vreg_scale1, ulmax1 + (i * ulmaxLoopRow));
        LoadAlign(vreg_scale2, ulmax1 + (i * ulmaxLoopRow + 1 * m));
        LoadAlign(vreg_scale3, ulmax1 + (i * ulmaxLoopRow + 2 * m));
        LoadAlign(vreg_scale4, ulmax1 + (i * ulmaxLoopRow + 3 * m));
        LoadAlign(vreg_scale5, ulmax1 + (i * ulmaxLoopRow + 4 * m));
        LoadAlign(vreg_scale6, ulmax1 + (i * ulmaxLoopRow + 5 * m));
        LoadAlign(vreg_scale7, ulmax1 + (i * ulmaxLoopRow + 6 * m));
        LoadAlign(vreg_scale8, ulmax1 + (i * ulmaxLoopRow + 7 * m));

        Sub(vreg_scale1, vreg_scale1, vreg_gmax, preg_VL64_16);
        Sub(vreg_scale2, vreg_scale2, vreg_gmax, preg_VL64_16);
        Sub(vreg_scale3, vreg_scale3, vreg_gmax, preg_VL64_16);
        Sub(vreg_scale4, vreg_scale4, vreg_gmax, preg_VL64_16);
        Sub(vreg_scale5, vreg_scale5, vreg_gmax, preg_VL64_16);
        Sub(vreg_scale6, vreg_scale6, vreg_gmax, preg_VL64_16);
        Sub(vreg_scale7, vreg_scale7, vreg_gmax, preg_VL64_16);
        Sub(vreg_scale8, vreg_scale8, vreg_gmax, preg_VL64_16);

        // SAT Mode, if < 0, then = 0
        RegTensor<uint8_t> vreg_mxscale1, vreg_mxscale2, vreg_mxscale3, vreg_mxscale4, vreg_mxscale5, vreg_mxscale6,
            vreg_mxscale7, vreg_mxscale8;
        Cast<uint8_t, T, castTraitZero>(vreg_mxscale1, vreg_scale1, preg_VL64_16);
        Cast<uint8_t, T, castTraitOne>(vreg_mxscale2, vreg_scale2, preg_VL64_16);
        Cast<uint8_t, T, castTraitZero>(vreg_mxscale3, vreg_scale3, preg_VL64_16);
        Cast<uint8_t, T, castTraitOne>(vreg_mxscale4, vreg_scale4, preg_VL64_16);
        Cast<uint8_t, T, castTraitZero>(vreg_mxscale5, vreg_scale5, preg_VL64_16);
        Cast<uint8_t, T, castTraitOne>(vreg_mxscale6, vreg_scale6, preg_VL64_16);
        Cast<uint8_t, T, castTraitZero>(vreg_mxscale7, vreg_scale7, preg_VL64_16);
        Cast<uint8_t, T, castTraitOne>(vreg_mxscale8, vreg_scale8, preg_VL64_16);

        Or(vreg_mxscale1, vreg_mxscale1, vreg_mxscale2, preg_VL128_8);
        Or(vreg_mxscale3, vreg_mxscale3, vreg_mxscale4, preg_VL128_8);
        Or(vreg_mxscale5, vreg_mxscale5, vreg_mxscale6, preg_VL128_8);
        Or(vreg_mxscale7, vreg_mxscale7, vreg_mxscale8, preg_VL128_8);

        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(
            mxscale1 + i * 32 * 5 * 8 + 32 * 0, vreg_mxscale1, 5, preg_VL128_8);

        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(
            mxscale1 + i * 32 * 5 * 8 + 32 * 1, vreg_mxscale3, 5, preg_VL128_8);

        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(
            mxscale1 + i * 32 * 5 * 8 + 32 * 2, vreg_mxscale5, 5, preg_VL128_8);

        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(
            mxscale1 + i * 32 * 5 * 8 + 32 * 3, vreg_mxscale7, 5, preg_VL128_8);
    }

    for (uint16_t i = 0; i < secondLoop; i++) {
        RegTensor<half> vreg_scale1, vreg_scale2, vreg_scale3, vreg_scale4, vreg_scale5, vreg_scale6, vreg_scale7,
            vreg_scale8;
        LoadAlign(vreg_scale1, ulmax2 + (i * ulmaxLoopRow));
        LoadAlign(vreg_scale2, ulmax2 + (i * ulmaxLoopRow + 1 * m));
        LoadAlign(vreg_scale3, ulmax2 + (i * ulmaxLoopRow + m * 2));
        LoadAlign(vreg_scale4, ulmax2 + (i * ulmaxLoopRow + m * 3));
        LoadAlign(vreg_scale5, ulmax2 + (i * ulmaxLoopRow + m * 4));
        LoadAlign(vreg_scale6, ulmax2 + (i * ulmaxLoopRow + m * 5));
        LoadAlign(vreg_scale7, ulmax2 + (i * ulmaxLoopRow + m * 6));
        LoadAlign(vreg_scale8, ulmax2 + (i * ulmaxLoopRow + m * 7));

        Sub(vreg_scale1, vreg_scale1, vreg_gmax, preg_VL64_16);
        Sub(vreg_scale2, vreg_scale2, vreg_gmax, preg_VL64_16);
        Sub(vreg_scale3, vreg_scale3, vreg_gmax, preg_VL64_16);
        Sub(vreg_scale4, vreg_scale4, vreg_gmax, preg_VL64_16);
        Sub(vreg_scale5, vreg_scale5, vreg_gmax, preg_VL64_16);
        Sub(vreg_scale6, vreg_scale6, vreg_gmax, preg_VL64_16);
        Sub(vreg_scale7, vreg_scale7, vreg_gmax, preg_VL64_16);
        Sub(vreg_scale8, vreg_scale8, vreg_gmax, preg_VL64_16);

        // SAT Mode, if < 0, then = 0
        RegTensor<uint8_t> vreg_mxscale1, vreg_mxscale2, vreg_mxscale3, vreg_mxscale4, vreg_mxscale5, vreg_mxscale6,
            vreg_mxscale7, vreg_mxscale8;
        Cast<uint8_t, half, castTraitZero>(vreg_mxscale1, vreg_scale1, preg_VL64_16);
        Cast<uint8_t, half, castTraitOne>(vreg_mxscale2, vreg_scale2, preg_VL64_16);
        Cast<uint8_t, half, castTraitZero>(vreg_mxscale3, vreg_scale3, preg_VL64_16);
        Cast<uint8_t, half, castTraitOne>(vreg_mxscale4, vreg_scale4, preg_VL64_16);
        Cast<uint8_t, half, castTraitZero>(vreg_mxscale5, vreg_scale5, preg_VL64_16);
        Cast<uint8_t, half, castTraitOne>(vreg_mxscale6, vreg_scale6, preg_VL64_16);
        Cast<uint8_t, half, castTraitZero>(vreg_mxscale7, vreg_scale7, preg_VL64_16);
        Cast<uint8_t, half, castTraitOne>(vreg_mxscale8, vreg_scale8, preg_VL64_16);

        Or(vreg_mxscale1, vreg_mxscale1, vreg_mxscale2, preg_VL128_8);
        Or(vreg_mxscale3, vreg_mxscale3, vreg_mxscale4, preg_VL128_8);
        Or(vreg_mxscale5, vreg_mxscale5, vreg_mxscale6, preg_VL128_8);
        Or(vreg_mxscale7, vreg_mxscale7, vreg_mxscale8, preg_VL128_8);

        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(
            mxscale2 + i * 32 * 5 * 8 + 32 * 0, vreg_mxscale1, 5, preg_VL128_8);

        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(
            mxscale2 + i * 32 * 5 * 8 + 32 * 1, vreg_mxscale3, 5, preg_VL128_8);

        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(
            mxscale2 + i * 32 * 5 * 8 + 32 * 2, vreg_mxscale5, 5, preg_VL128_8);

        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(
            mxscale2 + i * 32 * 5 * 8 + 32 * 3, vreg_mxscale7, 5, preg_VL128_8);
    }
}

template <MXQuantMode MX_QUANT_MODE = MXQuantMode::OCP, bool clear_gmax, typename T, uint16_t QsBase = 128>
__aicore__ inline void ComputePscaleAndDmQS64CallVF(
    const LocalTensor<uint8_t>& mxscale1, const LocalTensor<uint8_t>& mxscale2, const LocalTensor<T>& ulmax1,
    const LocalTensor<T>& ulmax2, const LocalTensor<T>& umax1, const LocalTensor<T>& umax2,
    const LocalTensor<T>& ugmaxOld, const LocalTensor<float>& urs, uint16_t firstLoop, uint16_t secondLoop,
    uint16_t coreIndex, const T LOG2_CX_CEIL)
{
    __ubuf__ uint8_t* mxscale1_buf = (__ubuf__ uint8_t*)mxscale1.GetPhyAddr();
    __ubuf__ uint8_t* mxscale2_buf = (__ubuf__ uint8_t*)mxscale2.GetPhyAddr();
    __ubuf__ T* ulmax1_buf = (__ubuf__ T*)ulmax1.GetPhyAddr();
    __ubuf__ T* ulmax2_buf = (__ubuf__ T*)ulmax2.GetPhyAddr();
    __ubuf__ T* umax1_buf = (__ubuf__ T*)umax1.GetPhyAddr();
    __ubuf__ T* umax2_buf = (__ubuf__ T*)umax2.GetPhyAddr();
    __ubuf__ T* ugmax_old_buf = (__ubuf__ T*)ugmaxOld.GetPhyAddr();
    __ubuf__ float* urs_buf = (__ubuf__ float*)urs.GetPhyAddr();

    compute_pscale_qs64_vf<MX_QUANT_MODE, clear_gmax, T, QsBase>(
        mxscale1_buf, mxscale2_buf, ulmax1_buf, ulmax2_buf, umax1_buf, umax2_buf, ugmax_old_buf, urs_buf, firstLoop,
        secondLoop, coreIndex, LOG2_CX_CEIL);
}
} // namespace Catlass::Epilogue::Block::Mxfp4VF
#endif // VF_COMPUTE_PSCALE_DM_QS64_H_
