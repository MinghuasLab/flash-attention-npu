// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once

#include "kernel_operator.h"
#include "fwd_storage.h"

struct FwdBatchOffsets {
    int64_t q;
    int64_t k;
    int64_t v;
    int64_t outputToken;
    uint64_t lse;
    int64_t blockTable;
};

// Dense operands have independent batch strides. Packed operands index tokens
// by their cumulative sequence offsets. Paged K/V resolve the page in the tile copy.
template <bool Packed, bool Paged>
__aicore__ inline FwdBatchOffsets
GetFwdBatchOffsets(const FwdStorageParams& storage, uint32_t batch, uint32_t maxQ, uint32_t heads,
                   uint32_t blocksPerBatch, AscendC::GlobalTensor<int32_t>& cuQ, AscendC::GlobalTensor<int32_t>& cuK)
{
    FwdBatchOffsets offsets{};
    if constexpr (Packed) {
        offsets.outputToken = cuQ.GetValue(batch);
        offsets.q = offsets.outputToken * storage.q.token;
        offsets.lse = offsets.outputToken;
    } else {
        offsets.outputToken = static_cast<int64_t>(batch) * maxQ;
        offsets.q = batch * storage.q.outer;
        offsets.lse = offsets.outputToken * heads;
    }
    if constexpr (Paged) {
        offsets.blockTable = static_cast<int64_t>(batch) * blocksPerBatch;
    } else if constexpr (Packed) {
        const int64_t token = cuK.GetValue(batch);
        offsets.k = token * storage.k.token;
        offsets.v = token * storage.v.token;
    } else {
        offsets.k = batch * storage.k.outer;
        offsets.v = batch * storage.v.outer;
    }
    return offsets;
}
