// Copyright (c) 2026, Minghua Shen.

#ifndef FLASH_ATTN_DEVICE_TRIU_MASK_H
#define FLASH_ATTN_DEVICE_TRIU_MASK_H

#include <unordered_map>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

// Initialize and synchronize only the device used by this caller.
// Per-thread device caches avoid touching other workers' devices.
//
// dim=2048: forward compressed mask (mha_fwd).
// dim=256:  backward FAG epilogue layout RowMajor(256,256) (mha_bwd).
inline at::Tensor MakeDeviceTriuMask(int64_t dim = 2048)
{
    TORCH_CHECK(dim > 0, "MakeDeviceTriuMask: dim must be positive");
    const auto idx = c10_npu::getCurrentNPUStream().device_index();
    using Key = uint64_t;
    const Key key =
        (static_cast<uint64_t>(static_cast<uint32_t>(idx)) << 32) | static_cast<uint64_t>(static_cast<uint32_t>(dim));
    static thread_local std::unordered_map<Key, at::Tensor> masks;
    auto found = masks.find(key);
    if (found != masks.end()) {
        return found->second;
    }
    c10_npu::NPUGuard guard(idx);
    at::Tensor mask =
        at::ones({dim, dim}, at::TensorOptions().dtype(at::kByte).device(at::Device(at::kPrivateUse1, idx)));
    mask.triu_(/*diagonal=*/1);
    c10_npu::getCurrentNPUStream().synchronize();
    return masks.emplace(key, std::move(mask)).first->second;
}

#endif
