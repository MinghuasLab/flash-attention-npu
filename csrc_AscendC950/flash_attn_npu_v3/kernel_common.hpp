/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_COMMON
#define KERNEL_COMMON

#include "kernel_operator.h"

constexpr int32_t NUM1 = 1;
constexpr int32_t NUM4 = 4;

constexpr int32_t NUM64 = 64;
constexpr int32_t NUM512 = 512;
constexpr int32_t NUM576 = 576;
constexpr int32_t BASIC_BLOCK_SIZE = 256;
constexpr int32_t Q_BLK = 256;
constexpr int32_t MAX_STACK_LEN = 512;

constexpr uint32_t FLOAT_VECTOR_SIZE = 64;

constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;

struct FAIKernelParams {
    // Data members
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR blockTables;
    GM_ADDR actualQseqlen;
    GM_ADDR actualKvseqlen;
    GM_ADDR o;
    GM_ADDR lse;
    GM_ADDR workSpace;
    GM_ADDR tiling;

    // Methods
    __aicore__ inline FAIKernelParams() {}

    __aicore__ inline FAIKernelParams(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR blockTables_,
            GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_, GM_ADDR o_, GM_ADDR lse_, GM_ADDR workSpace_, GM_ADDR tiling_)
        : q(q_), k(k_), v(v_), mask(mask_), blockTables(blockTables_), actualQseqlen(actualQseqlen_),
            actualKvseqlen(actualKvseqlen_), o(o_), lse(lse_), workSpace(workSpace_), tiling(tiling_) {}
};

enum class Format
{
    TND = 0,
    BSND = 1
};

enum class CacheMode 
{
    normalCache = 0,
    pagedCache = 1,
};

enum class PageShape 
{
    BnBsND = 0,
    BnNBsD = 1,
    normalShape = 2,
};

enum class MaskCategory 
{
    NO_MASK = 0,
    MASK_CAUSAL = 1,
    MASK_SWA = 4,
};

enum class CacheLayout : uint8_t
{
    nd = 0,
    nz = 1,
};

#endif
