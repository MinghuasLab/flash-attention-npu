/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026
 */

#ifndef FLASH_ATTN_CACHED_TRIU_MASK_H
#define FLASH_ATTN_CACHED_TRIU_MASK_H

#include <mutex>
#include <unordered_map>

#include <torch/extension.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

// Process-wide 2048x2048 int8 triu(1). Allocated and filled on device at most
// once (runtime memset + AIV triu_; no H2D / AICPU). Address is stable so ACL
// graphs can capture on one thread and replay on another.
inline at::Tensor CachedCompressedTriuMask()
{
    const c10::DeviceIndex idx = c10_npu::getCurrentNPUStream().device_index();
    static std::mutex mu;
    static std::unordered_map<c10::DeviceIndex, at::Tensor> masks;
    std::lock_guard<std::mutex> lock(mu);
    at::Tensor &mask = masks[idx];
    if (mask.defined()) {
        return mask;
    }
    constexpr int64_t dim = 2048;
    mask = at::empty({dim, dim}, at::dtype(at::kByte).device(at::Device(at::kPrivateUse1, idx)));
    const size_t bytes = static_cast<size_t>(mask.numel()) * mask.itemsize();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    TORCH_CHECK(aclrtMemsetAsync(mask.data_ptr(), bytes, /*value=*/1, bytes, stream.stream(false)) == ACL_SUCCESS,
                "aclrtMemsetAsync failed while filling compressed triu mask");
    mask.triu_(/*diagonal=*/1);
    TORCH_CHECK(aclrtSynchronizeStream(stream.stream(false)) == ACL_SUCCESS,
                "aclrtSynchronizeStream failed while filling compressed triu mask");
    return mask;
}

#endif
