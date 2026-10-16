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

#ifndef VF_RESCALE_O_DN_H_
#define VF_RESCALE_O_DN_H_
#include "kernel_tensor.h"
#include "vf_common_def.h"
namespace Catlass::Epilogue::Block::Mxfp4VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace Reg;

template <bool isUpdate, uint16_t KvsBase = 256>
__simd_vf__ void process_update_o_vf(__ubuf__ float* update_res_ub, __ubuf__ float* mm2_res_ub, __ubuf__ float* urs,
                                     __ubuf__ float* rsum, __ubuf__ float* ursum)
{
    // 更新 mm2
    MaskReg preg_all32 = CreateMask<uint32_t, MaskPattern::ALL>();
    if constexpr (isUpdate) {
        RegTensor<float> vreg_rs, vreg_ursum, vreg_rsum;
        LoadAlign(vreg_rs, (__ubuf__ float*)urs);
        LoadAlign(vreg_rsum, (__ubuf__ float*)rsum);
        LoadAlign(vreg_ursum, (__ubuf__ float*)ursum);
        Mul(vreg_rsum, vreg_rsum, vreg_rs, preg_all32);
        Add(vreg_rsum, vreg_rsum, vreg_ursum, preg_all32);

        StoreAlign((__ubuf__ float*)rsum, vreg_rsum, preg_all32);

        for (uint16_t i = 0; i < KvsBase / 2; i += 4) {
            RegTensor<float> vreg_pv1, vreg_pv2, vreg_pv3, vreg_pv4, vreg_out1, vreg_out2, vreg_out3, vreg_out4;
            LoadAlign(vreg_out1, (__ubuf__ float*)update_res_ub + (i + 0) * 64);
            LoadAlign(vreg_out2, (__ubuf__ float*)update_res_ub + (i + 1) * 64);
            LoadAlign(vreg_out3, (__ubuf__ float*)update_res_ub + (i + 2) * 64);
            LoadAlign(vreg_out4, (__ubuf__ float*)update_res_ub + (i + 3) * 64);

            Mul(vreg_out1, vreg_out1, vreg_rs, preg_all32);
            Mul(vreg_out2, vreg_out2, vreg_rs, preg_all32);
            Mul(vreg_out3, vreg_out3, vreg_rs, preg_all32);
            Mul(vreg_out4, vreg_out4, vreg_rs, preg_all32);

            LoadAlign(vreg_pv1, (__ubuf__ float*)mm2_res_ub + (i + 0) * 64);
            LoadAlign(vreg_pv2, (__ubuf__ float*)mm2_res_ub + (i + 1) * 64);
            LoadAlign(vreg_pv3, (__ubuf__ float*)mm2_res_ub + (i + 2) * 64);
            LoadAlign(vreg_pv4, (__ubuf__ float*)mm2_res_ub + (i + 3) * 64);

            Add(vreg_out1, vreg_out1, vreg_pv1, preg_all32);
            Add(vreg_out2, vreg_out2, vreg_pv2, preg_all32);
            Add(vreg_out3, vreg_out3, vreg_pv3, preg_all32);
            Add(vreg_out4, vreg_out4, vreg_pv4, preg_all32);

            StoreAlign((__ubuf__ float*)update_res_ub + (i + 0) * 64, vreg_out1, preg_all32);
            StoreAlign((__ubuf__ float*)update_res_ub + (i + 1) * 64, vreg_out2, preg_all32);
            StoreAlign((__ubuf__ float*)update_res_ub + (i + 2) * 64, vreg_out3, preg_all32);
            StoreAlign((__ubuf__ float*)update_res_ub + (i + 3) * 64, vreg_out4, preg_all32);
        }
    } else {
        RegTensor<float> vreg_grsum;
        LoadAlign(vreg_grsum, (__ubuf__ float*)ursum);
        StoreAlign((__ubuf__ float*)rsum, vreg_grsum, preg_all32);

        for (uint16_t i = 0; i < KvsBase / 2; i += 4) {
            RegTensor<float> vreg_out1, vreg_out2, vreg_out3, vreg_out4;
            LoadAlign(vreg_out1, (__ubuf__ float*)mm2_res_ub + (i + 0) * 64);
            LoadAlign(vreg_out2, (__ubuf__ float*)mm2_res_ub + (i + 1) * 64);
            LoadAlign(vreg_out3, (__ubuf__ float*)mm2_res_ub + (i + 2) * 64);
            LoadAlign(vreg_out4, (__ubuf__ float*)mm2_res_ub + (i + 3) * 64);

            StoreAlign((__ubuf__ float*)update_res_ub + (i + 0) * 64, vreg_out1, preg_all32);
            StoreAlign((__ubuf__ float*)update_res_ub + (i + 1) * 64, vreg_out2, preg_all32);
            StoreAlign((__ubuf__ float*)update_res_ub + (i + 2) * 64, vreg_out3, preg_all32);
            StoreAlign((__ubuf__ float*)update_res_ub + (i + 3) * 64, vreg_out4, preg_all32);
        }
    }
}

template <bool isUpdate, uint16_t KvsBase = 256>
__aicore__ inline void ProcessUpdateOCallVF(const LocalTensor<float>& updateResUb, const LocalTensor<float>& mm2ResUb,
                                            const LocalTensor<float>& urs, const LocalTensor<float>& rsum,
                                            const LocalTensor<float>& ursum)
{
    __ubuf__ float* update_res_ub_buf = (__ubuf__ float*)updateResUb.GetPhyAddr();
    __ubuf__ float* mm2_res_ub_buf = (__ubuf__ float*)mm2ResUb.GetPhyAddr();
    __ubuf__ float* urs_buf = (__ubuf__ float*)urs.GetPhyAddr();
    __ubuf__ float* rsum_buf = (__ubuf__ float*)rsum.GetPhyAddr();
    __ubuf__ float* ursum_buf = (__ubuf__ float*)ursum.GetPhyAddr();

    process_update_o_vf<isUpdate, KvsBase>(update_res_ub_buf, mm2_res_ub_buf, urs_buf, rsum_buf, ursum_buf);
}

} // namespace Catlass::Epilogue::Block::Mxfp4VF
#endif // VF_RESCALE_O_DN_H_
