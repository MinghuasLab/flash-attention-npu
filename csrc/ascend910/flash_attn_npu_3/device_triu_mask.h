// Copyright (c) 2026, Minghua Shen.

#ifndef FLASH_ATTN_DEVICE_TRIU_MASK_H
#define FLASH_ATTN_DEVICE_TRIU_MASK_H

#include <unordered_map>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#if __has_include("torch_npu/csrc/core/npu/NPUGraphsUtils.h")
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#endif

inline std::unordered_map<c10::DeviceIndex, at::Tensor>& DeviceTriuMaskCache()
{
    static thread_local std::unordered_map<c10::DeviceIndex, at::Tensor> masks;
    return masks;
}

// Python atexit runs before sanitizer/runtime global destructors. Release NPU
// tensors there instead of waiting for the main thread's TLS destruction.
inline void ClearDeviceTriuMaskCache()
{
    auto& masks = DeviceTriuMaskCache();
    for (const auto& entry : masks) {
        c10_npu::NPUGuard guard(entry.first);
        c10_npu::device_synchronize();
    }
    masks.clear();
}

// Initialize and synchronize only the device used by this caller.
// Per-thread device caches avoid touching other workers' devices.
inline at::Tensor MakeDeviceTriuMask(bool cacheMask = true)
{
    const auto idx = c10_npu::getCurrentNPUStream().device_index();
    bool capturing = false;
#if __has_include("torch_npu/csrc/core/npu/NPUGraphsUtils.h")
    capturing = c10_npu::currentStreamCaptureStatusMayInitCtx() != c10_npu::CaptureStatus::None;
#endif
    auto& masks = DeviceTriuMaskCache();
    auto found = masks.find(idx);
    if (cacheMask && !capturing && found != masks.end()) {
        return found->second;
    }
    c10_npu::NPUGuard guard(idx);
    constexpr int64_t dim = 2048;
    at::Tensor mask =
        at::ones({dim, dim}, at::TensorOptions().dtype(at::kByte).device(at::Device(at::kPrivateUse1, idx)));
    mask.triu_(/*diagonal=*/1);
    // Captures own their mask in the graph pool, including when this thread
    // has a warm cache. A replay can outlive the thread that created it.
    if (capturing || !cacheMask)
        return mask;
    c10_npu::getCurrentNPUStream().synchronize();
    return masks.emplace(idx, std::move(mask)).first->second;
}

#endif
