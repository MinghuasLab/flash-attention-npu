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
 * \file vf_softmax_dn_cast_nz_align_qs128_kvs32_multi.h
 * \brief softmax + mxfp4 quant for qs=128, kvs padding 32 multi (kvsActBaseTileAlign32 >= 64, runtime variable)
 */

#ifndef VF_SOFTMAX_DN_CAST_NZ_ALIGN_QS128_KVS32_MULTI_H_
#define VF_SOFTMAX_DN_CAST_NZ_ALIGN_QS128_KVS32_MULTI_H_
#include "vf_common_def.h"
#include "../../epilogue_dispatch_policy.hpp"

namespace Catlass::Epilogue::Block::Mxfp4VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace Reg;

template <MXQuantMode MX_QUANT_MODE = MXQuantMode::OCP, bool clear_gmax, typename T, typename T2, uint16_t QsBase = 128>
__simd_vf__ inline void softmax_with_group_max_align_qs128_kvs32_multi_vf(
    __ubuf__ T2* p_dest, __ubuf__ T* s, __ubuf__ T* local_group_max, __ubuf__ T* global_max,
    __ubuf__ uint8_t* indexes_ub, const T NEG_LOG2_CX, uint16_t KvsBaseAlign32, uint16_t KvsBaseAlign64)
{
    // ====================== 寄存器定义 ======================
    RegTensor<half> src_c0, src_c1, src_c2, src_c3;
    RegTensor<half> src_n0, src_n1, src_n2, src_n3;
    RegTensor<half> curr_group_max;
    RegTensor<half> next_group_max;
    RegTensor<half> group_gmax;
    RegTensor<half> min_val_reg;
    RegTensor<uint8_t> idx_nd2nz;

    // 量化专用寄存器
    RegTensor<bfloat16_t> src_bf16_0, src_bf16_1, src_bf16_2, src_bf16_3;
    RegTensor<float4_e2m1x2_t> quant_0, quant_1, quant_2, quant_3;

    // ====================== 分块常量 ======================
    const uint16_t ROWS_PER_GROUP = 32;
    const uint16_t GROUP_COUNT = KvsBaseAlign32 / ROWS_PER_GROUP;
    const uint16_t GROUP_COUNT_ALIGN_64 = KvsBaseAlign64 / ROWS_PER_GROUP;
    const uint16_t ROW_SUB_LOOP = 4;
    const uint16_t ITER_PER_GROUP = ROWS_PER_GROUP / ROW_SUB_LOOP;
    uint32_t MID_VALID_CNT = QsBase * (GROUP_COUNT - 1);

    // ====================== 掩码定义 ======================
    MaskReg preg_all_16bit = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_all_8bit = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg preg_vl128 = CreateMask<uint8_t, MaskPattern::VL128>();
    MaskReg preg_vl128_not;
    MaskNot(preg_vl128_not, preg_vl128, preg_all_8bit);

    MaskReg preg_invalid_max;

    // ====================== 全局最大值初始化 ======================
    LoadAlign(group_gmax, global_max);

    LoadAlign(idx_nd2nz, indexes_ub);

    // ====================== 预计算：第一个分组的最大值 ======================
    Duplicate(curr_group_max, MIN_VALUE);
    for (uint16_t iter = 0; iter < ITER_PER_GROUP; ++iter) {
        LoadAlign(src_c0, s + (iter * QsBase * ROW_SUB_LOOP + 0 * QsBase) * 2);
        LoadAlign(src_c1, s + (iter * QsBase * ROW_SUB_LOOP + 1 * QsBase) * 2);
        LoadAlign(src_c2, s + (iter * QsBase * ROW_SUB_LOOP + 2 * QsBase) * 2);
        LoadAlign(src_c3, s + (iter * QsBase * ROW_SUB_LOOP + 3 * QsBase) * 2);

        Max(src_c0, src_c0, src_c1, preg_all_16bit);
        Max(src_c2, src_c2, src_c3, preg_all_16bit);
        Max(curr_group_max, curr_group_max, src_c0, preg_all_16bit);
        Max(curr_group_max, curr_group_max, src_c2, preg_all_16bit);
    }

    Muls(curr_group_max, curr_group_max, INV_LN2, preg_all_16bit);
    if constexpr (MX_QUANT_MODE == MXQuantMode::OCP) {
        Truncate<T, RoundMode::CAST_FLOOR>(curr_group_max, curr_group_max, preg_all_16bit);
    } else {
        // CX: M'_(local,FP16) = ceil(Smax/ln2 - log2(CX))·ln2, 其中 -log2(CX) = NEG_LOG2_CX。
        Adds(curr_group_max, curr_group_max, NEG_LOG2_CX, preg_all_16bit);
        Truncate<T, RoundMode::CAST_CEIL>(curr_group_max, curr_group_max, preg_all_16bit);
    }
    Max(group_gmax, group_gmax, curr_group_max, preg_all_16bit);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>(local_group_max, curr_group_max, preg_all_16bit);
    if constexpr (MX_QUANT_MODE == MXQuantMode::OCP) {
        Adds(curr_group_max, curr_group_max, NEG_TWO_VALE, preg_all_16bit);
    }
    Muls(curr_group_max, curr_group_max, LN2, preg_all_16bit);

    // ====================== 核心：双块流水分组循环 ======================
    for (uint16_t i = 0; i < GROUP_COUNT; i++) {
        MaskReg preg_valid_max = UpdateMask<half>(MID_VALID_CNT);
        MaskNot(preg_invalid_max, preg_valid_max, preg_all_16bit);

        // 初始化下一个块最大值
        Duplicate(next_group_max, MIN_VALUE);

        // ========== 第一个内循环：处理偶数子块 j = 0,2,4,6 ==========
        for (uint16_t j = 0; j < ITER_PER_GROUP; j += 2) {
            uint16_t rowOffset_next = (i + 1) * ROWS_PER_GROUP + j * ROW_SUB_LOOP;
            LoadAlign(src_n0, s + (rowOffset_next * QsBase + 0 * QsBase) * 2);
            LoadAlign(src_n1, s + (rowOffset_next * QsBase + 1 * QsBase) * 2);
            LoadAlign(src_n2, s + (rowOffset_next * QsBase + 2 * QsBase) * 2);
            LoadAlign(src_n3, s + (rowOffset_next * QsBase + 3 * QsBase) * 2);
            Max(src_n0, src_n0, src_n1, preg_valid_max);
            Max(src_n2, src_n2, src_n3, preg_valid_max);
            Max(next_group_max, next_group_max, src_n0, preg_valid_max);
            Max(next_group_max, next_group_max, src_n2, preg_valid_max);

            // 当前块计算：Sub + Exp
            uint16_t rowOffset_cur = i * ROWS_PER_GROUP + j * ROW_SUB_LOOP;
            LoadAlign(src_c0, s + (rowOffset_cur * QsBase + 0 * QsBase) * 2);
            LoadAlign(src_c1, s + (rowOffset_cur * QsBase + 1 * QsBase) * 2);
            LoadAlign(src_c2, s + (rowOffset_cur * QsBase + 2 * QsBase) * 2);
            LoadAlign(src_c3, s + (rowOffset_cur * QsBase + 3 * QsBase) * 2);

            Sub(src_c0, src_c0, curr_group_max, preg_all_16bit);
            Sub(src_c1, src_c1, curr_group_max, preg_all_16bit);
            Sub(src_c2, src_c2, curr_group_max, preg_all_16bit);
            Sub(src_c3, src_c3, curr_group_max, preg_all_16bit);

            Exp(src_c0, src_c0, preg_all_16bit);
            Exp(src_c1, src_c1, preg_all_16bit);
            Exp(src_c2, src_c2, preg_all_16bit);
            Exp(src_c3, src_c3, preg_all_16bit);

            // 4bit 量化压缩
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_0, src_c0, preg_all_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_1, src_c1, preg_all_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_2, src_c2, preg_all_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_3, src_c3, preg_all_16bit);

            Cast<float4_e2m1x2_t, bfloat16_t, castTraitZero>(quant_0, src_bf16_0, preg_all_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitOne>(quant_1, src_bf16_1, preg_all_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitTwo>(quant_2, src_bf16_2, preg_all_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitThree>(quant_3, src_bf16_3, preg_all_16bit);

            // 数据合并
            Or((RegTensor<uint8_t>&)quant_0, (RegTensor<uint8_t>&)quant_0, (RegTensor<uint8_t>&)quant_1, preg_all_8bit);
            Or((RegTensor<uint8_t>&)quant_2, (RegTensor<uint8_t>&)quant_2, (RegTensor<uint8_t>&)quant_3, preg_all_8bit);
            Or((RegTensor<uint8_t>&)quant_0, (RegTensor<uint8_t>&)quant_0, (RegTensor<uint8_t>&)quant_2, preg_all_8bit);
            Gather((RegTensor<uint8_t>&)quant_0, (RegTensor<uint8_t>&)quant_0, idx_nd2nz);

            // 写入内存（偶数块：无+128偏移）
            StoreAlign(
                ((__ubuf__ uint8_t*&)p_dest) + i * 2048 + j * 256, (RegTensor<uint8_t>&)quant_0,
                preg_vl128 // 低128字节掩码
            );

            StoreAlign(
                ((__ubuf__ uint8_t*&)p_dest) + i * 2048 + j * 256 + 16256, (RegTensor<uint8_t>&)quant_0,
                preg_vl128_not // 高128字节掩码
            );
        }

        // ========== 第二个内循环：处理奇数子块 j+1 = 1,3,5,7 ==========
        for (uint16_t j = 0; j < ITER_PER_GROUP; j += 2) {
            // 预计算：下一个块 (i+1) 最大值
            uint16_t rowOffset_next = (i + 1) * ROWS_PER_GROUP + (j + 1) * ROW_SUB_LOOP;
            LoadAlign(src_n0, s + (rowOffset_next * QsBase + 0 * QsBase) * 2);
            LoadAlign(src_n1, s + (rowOffset_next * QsBase + 1 * QsBase) * 2);
            LoadAlign(src_n2, s + (rowOffset_next * QsBase + 2 * QsBase) * 2);
            LoadAlign(src_n3, s + (rowOffset_next * QsBase + 3 * QsBase) * 2);

            Max(src_n0, src_n0, src_n1, preg_valid_max);
            Max(src_n2, src_n2, src_n3, preg_valid_max);
            Max(next_group_max, next_group_max, src_n0, preg_valid_max);
            Max(next_group_max, next_group_max, src_n2, preg_valid_max);

            // 当前块计算：Sub + Exp
            uint16_t rowOffset_cur = i * ROWS_PER_GROUP + (j + 1) * ROW_SUB_LOOP;
            LoadAlign(src_c0, s + (rowOffset_cur * QsBase + 0 * QsBase) * 2);
            LoadAlign(src_c1, s + (rowOffset_cur * QsBase + 1 * QsBase) * 2);
            LoadAlign(src_c2, s + (rowOffset_cur * QsBase + 2 * QsBase) * 2);
            LoadAlign(src_c3, s + (rowOffset_cur * QsBase + 3 * QsBase) * 2);

            Sub(src_c0, src_c0, curr_group_max, preg_all_16bit);
            Sub(src_c1, src_c1, curr_group_max, preg_all_16bit);
            Sub(src_c2, src_c2, curr_group_max, preg_all_16bit);
            Sub(src_c3, src_c3, curr_group_max, preg_all_16bit);

            Exp(src_c0, src_c0, preg_all_16bit);
            Exp(src_c1, src_c1, preg_all_16bit);
            Exp(src_c2, src_c2, preg_all_16bit);
            Exp(src_c3, src_c3, preg_all_16bit);

            // 4bit 量化压缩
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_0, src_c0, preg_all_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_1, src_c1, preg_all_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_2, src_c2, preg_all_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_3, src_c3, preg_all_16bit);

            Cast<float4_e2m1x2_t, bfloat16_t, castTraitZero>(quant_0, src_bf16_0, preg_all_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitOne>(quant_1, src_bf16_1, preg_all_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitTwo>(quant_2, src_bf16_2, preg_all_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitThree>(quant_3, src_bf16_3, preg_all_16bit);

            // 数据合并
            Or((RegTensor<uint8_t>&)quant_0, (RegTensor<uint8_t>&)quant_0, (RegTensor<uint8_t>&)quant_1, preg_all_8bit);
            Or((RegTensor<uint8_t>&)quant_2, (RegTensor<uint8_t>&)quant_2, (RegTensor<uint8_t>&)quant_3, preg_all_8bit);
            Or((RegTensor<uint8_t>&)quant_0, (RegTensor<uint8_t>&)quant_0, (RegTensor<uint8_t>&)quant_2, preg_all_8bit);
            Gather((RegTensor<uint8_t>&)quant_0, (RegTensor<uint8_t>&)quant_0, idx_nd2nz);

            // 奇数块：低128字节写入（基础地址 +128 偏移）
            StoreAlign(
                ((__ubuf__ uint8_t*&)p_dest) + i * 2048 + j * 256 + 128, (RegTensor<uint8_t>&)quant_0, preg_vl128);

            // 奇数块：高128字节写入（远端地址）
            StoreAlign(
                ((__ubuf__ uint8_t*&)p_dest) + i * 2048 + j * 256 + 16384, (RegTensor<uint8_t>&)quant_0,
                preg_vl128_not);
        }

        // ====================== 全局/局部最大值更新 ======================
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>(global_max, group_gmax, preg_invalid_max);

        // 下一块最大值归一化
        Muls(next_group_max, next_group_max, INV_LN2, preg_valid_max);
        if constexpr (MX_QUANT_MODE == MXQuantMode::OCP) {
            Truncate<T, RoundMode::CAST_FLOOR>(next_group_max, next_group_max, preg_valid_max);
        } else {
            // CX: 与当前块一致, ceil(Smax/ln2 - log2(CX))
            Adds(next_group_max, next_group_max, NEG_LOG2_CX, preg_valid_max);
            Truncate<T, RoundMode::CAST_CEIL>(next_group_max, next_group_max, preg_valid_max);
        }
        Max(group_gmax, group_gmax, next_group_max, preg_valid_max);

        // 存储下一块最大值到 ulmax (i+1)*128 偏移
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>(
            local_group_max + ((i + 1) * QsBase), next_group_max, preg_valid_max);

        // 更新当前块最大值，用于下一次循环
        if constexpr (MX_QUANT_MODE == MXQuantMode::OCP) {
            Adds(next_group_max, next_group_max, NEG_TWO_VALE, preg_all_16bit);
        }
        Muls(curr_group_max, next_group_max, LN2, preg_valid_max);
    }

    // padding group_max 到 KvsBaseAlign64 (multi -inf), 使对应的 pscale = 0
    Duplicate(min_val_reg, MIN_VALUE);
    for (uint16_t i = GROUP_COUNT; i < GROUP_COUNT_ALIGN_64; ++i) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>(local_group_max + i * QsBase, min_val_reg, preg_all_16bit);
    }
}

template <MXQuantMode MX_QUANT_MODE = MXQuantMode::OCP, bool clear_gmax, typename T, typename T2, uint16_t QsBase = 128>
__aicore__ inline void SoftmaxWithGroupMaxAlignQs128Kvs32MultiCallVF(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& localGroupMax,
    const LocalTensor<T>& globalMax, const LocalTensor<uint8_t>& indexesBuf, const T NEG_LOG2_CX,
    uint16_t KvsBaseAlign32, uint16_t KvsBaseAlign64)
{
    __ubuf__ T2* p_dest = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T* s_ub = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T* local_group_max = (__ubuf__ T*)localGroupMax.GetPhyAddr();
    __ubuf__ T* global_max = (__ubuf__ T*)globalMax.GetPhyAddr();
    __ubuf__ uint8_t* indexes_ub = (__ubuf__ uint8_t*)indexesBuf.GetPhyAddr();

    softmax_with_group_max_align_qs128_kvs32_multi_vf<MX_QUANT_MODE, clear_gmax, T, T2, QsBase>(
        p_dest, s_ub, local_group_max, global_max, indexes_ub, NEG_LOG2_CX, KvsBaseAlign32, KvsBaseAlign64);
}

} // namespace Catlass::Epilogue::Block::Mxfp4VF
#endif // VF_SOFTMAX_DN_CAST_NZ_ALIGN_QS128_KVS32_MULTI_H_
