// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once

#include <cstdint>

enum class FwdLayout : uint32_t {
    TND = 0,
    BSND = 1,
    BNSD = 2, // Legacy native forward ABI with raw [B,N,S,D] dimensions.
};

// Element strides of a logical [batch/page, token, head, dim] view.
// Packed TND has no outer axis. The innermost dimension is contiguous.
struct FwdTensorStrides {
    int64_t outer;
    int64_t token;
    int64_t head;
};

struct FwdStorageParams {
    FwdTensorStrides q;
    FwdTensorStrides k;
    FwdTensorStrides v;
    uint64_t totalQTokens;
};
