/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */

// Host-side ABI for the split-attention "combine" operator
// (mirrors flash_attn_3::fwd_combine / flash_attn.cute flash_attn_combine).
//
// The kernel itself lives in combine_kernel.cpp so that flash_api.cpp stays
// free of the heavy kernel headers, matching the fwd/bwd dispatch split.

#pragma once

#include <cstdint>

#include "acl/acl.h"

struct CombineLaunchArgs {
    aclrtStream aclStream{nullptr};
    uint64_t fftsAddr{0};
    uint32_t launchBlockDim{0};

    uint32_t isDense{0};  // 1: (num_splits, B, S, H, D), 0: (num_splits, T, H, D)
    uint32_t writeLse{0}; // 1: materialize the combined LSE
    uint32_t numSplits{0};
    uint32_t batch{1};
    uint32_t seqlen{0}; // S for dense, T for varlen
    uint32_t numHeads{0};
    uint32_t headDim{0};
    uint32_t tileRows{0};

    uint8_t* outPartial{nullptr};
    uint8_t* lsePartial{nullptr};
    uint8_t* out{nullptr};
    uint8_t* lseOut{nullptr};

    // Element strides of out_partial (last dim is always contiguous).
    int64_t oSplit{0};
    int64_t oBatch{0};
    int64_t oSeq{0};
    int64_t oHead{0};

    // Element strides of lse_partial.
    int64_t lSplit{0};
    int64_t lBatch{0};
    int64_t lSeq{0};
    int64_t lHead{0};

    // Element strides of out (last dim contiguous) / lse output (seqlen contiguous).
    int64_t outBatch{0};
    int64_t outSeq{0};
    int64_t outHead{0};
    int64_t lseOutBatch{0};
    int64_t lseOutHead{0};
};

void launch_combine_fp16(const CombineLaunchArgs& args);
void launch_combine_bf16(const CombineLaunchArgs& args);
void launch_combine_fp32(const CombineLaunchArgs& args);
