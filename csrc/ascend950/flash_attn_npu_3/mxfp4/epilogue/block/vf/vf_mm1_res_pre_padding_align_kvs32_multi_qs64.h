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
 * \file vf_softmax_dn_cast_nz.h
 * \brief
 */

#ifndef VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_QS64_H
#define VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_QS64_H
#include "vf_common_def.h"

namespace Catlass::Epilogue::Block::Mxfp4VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace Reg;

template <typename T, uint16_t QsBase = 128>
__simd_vf__ inline void mm1_res_pre_padding_align_kvs32_multi_qs64_vf(
    __ubuf__ T* s, uint16_t actKvsTile, uint16_t kvsActBaseTileAlign32)
{
    // ====================== 寄存器定义 ======================
    MaskReg mask_VL64_16b = CreateMask<uint16_t, MaskPattern::VL64>();
    uint16_t kvsIdx = 0;
    RegTensor<T> padding_tensor;
    Duplicate(padding_tensor, MIN_VALUE, mask_VL64_16b);
    Muls(padding_tensor, padding_tensor, TWO_VALE, mask_VL64_16b);
    for (kvsIdx = actKvsTile; kvsIdx < kvsActBaseTileAlign32 - 1; kvsIdx += 2) {
        StoreAlign(s + (kvsIdx * QsBase) * 2, padding_tensor, mask_VL64_16b);
        StoreAlign(s + (kvsIdx * QsBase + 1 * QsBase) * 2, padding_tensor, mask_VL64_16b);
    }

    for (uint16_t idx = kvsIdx; idx < kvsActBaseTileAlign32; ++idx) {
        StoreAlign(s + (idx * QsBase) * 2, padding_tensor, mask_VL64_16b);
    }
}

template <typename T>
__aicore__ inline void Mm1ResPrePaddingAlignKvs32MultiQs64CallVF(
    const LocalTensor<T>& srcTensor, uint16_t actKvsTile, uint16_t kvsActBaseTileAlign32)
{
    __ubuf__ T* s_ub = (__ubuf__ T*)srcTensor.GetPhyAddr();

    mm1_res_pre_padding_align_kvs32_multi_qs64_vf<T>(s_ub, actKvsTile, kvsActBaseTileAlign32);
}

} // namespace Catlass::Epilogue::Block::Mxfp4VF
#endif // VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_QS64_H
