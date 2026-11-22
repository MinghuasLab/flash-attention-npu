// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once

#include <ATen/ATen.h>
#include "fwd_storage.h"

// The tile copies take a contiguous D row and 32-bit row/matrix strides.
// Materialize a view only when its rows cannot be represented by these copies.
inline at::Tensor PrepareFwdTensor(const at::Tensor& tensor)
{
    auto fits = [&](int axis) {
        return tensor.size(axis) <= 1 || (tensor.stride(axis) >= tensor.size(-1) && tensor.stride(axis) <= UINT32_MAX);
    };
    return tensor.stride(-1) == 1 && fits(-3) && fits(-2) ? tensor : tensor.contiguous();
}

inline FwdTensorStrides GetFwdStrides(const at::Tensor& tensor)
{
    auto stride = [&](int axis) { return tensor.size(axis) == 1 ? tensor.size(-1) : tensor.stride(axis); };
    const int64_t token = stride(-3);
    const int64_t head = stride(-2);
    TORCH_CHECK(tensor.numel() == 0 || (token <= UINT32_MAX && head <= UINT32_MAX),
                "token and head strides must fit the 32-bit tile copy");
    return {tensor.dim() == 4 ? tensor.stride(0) : 0, token, head};
}

inline FwdLayout PrepareFwdViews(int64_t layout, bool packedQ, bool packedK, bool paged, int64_t lengthCount,
                                 at::Tensor& q, at::Tensor& k, at::Tensor& v)
{
    TORCH_CHECK(layout >= 0 && layout <= static_cast<int64_t>(FwdLayout::BNSD), "unsupported layout code");
    TORCH_CHECK((layout == static_cast<int64_t>(FwdLayout::TND)) == packedQ, "layout code conflicts with cu_seqlens_q");
    if (layout == static_cast<int64_t>(FwdLayout::BNSD)) {
        TORCH_CHECK(!paged && !packedK && lengthCount == q.size(0),
                    "legacy dense layout requires per-batch KV lengths and rank-4 dense caches");
        q = q.transpose(1, 2);
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);
    }
    q = PrepareFwdTensor(q);
    k = PrepareFwdTensor(k);
    v = PrepareFwdTensor(v);
    return packedQ ? FwdLayout::TND : FwdLayout::BSND;
}
