/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026
 */

#ifndef FLASH_ATTN_DEVICE_TRIU_MASK_H
#define FLASH_ATTN_DEVICE_TRIU_MASK_H

#include <mutex>
#include <unordered_map>
#include <vector>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#if __has_include("torch_npu/csrc/core/npu/NPUGraphsUtils.h")
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#endif

// Cached 2048x2048 int8 triu(1) mask, one tensor per visible NPU device.
// The first call initializes every device in this process; later calls only
// look up the current device. Callers only read the tensor. No host/device
// copy or AICPU work is involved. Fills are launched on all devices first,
// then each device is synchronized once so other streams cannot consume an
// in-flight fill.
inline at::Tensor MakeDeviceTriuMask()
{
    const auto idx = c10_npu::getCurrentNPUStream().device_index();

    static std::once_flag once;
    static std::vector<at::Tensor> masks;

    std::call_once(once, []() {
        const auto ndev = c10_npu::device_count();
        masks.resize(static_cast<size_t>(ndev));
        constexpr int64_t dim = 2048;
        for (c10::DeviceIndex i = 0; i < ndev; ++i) {
            c10_npu::NPUGuard guard(i);
            at::Tensor mask =
                at::ones({dim, dim}, at::TensorOptions().dtype(at::kByte).device(at::Device(at::kPrivateUse1, i)));
            mask.triu_(/*diagonal=*/1);
            masks[static_cast<size_t>(i)] = std::move(mask);
        }
        for (c10::DeviceIndex i = 0; i < ndev; ++i) {
            c10_npu::NPUGuard guard(i);
            c10_npu::getCurrentNPUStream().synchronize();
        }
    });

    return masks[static_cast<size_t>(idx)];
}

// Initialize and synchronize only the device used by this caller.
// Per-thread device caches avoid touching other workers' devices.
inline at::Tensor MakeDeviceTriuMask(bool cacheMask)
{
    const auto idx = c10_npu::getCurrentNPUStream().device_index();
    bool capturing = false;
#if __has_include("torch_npu/csrc/core/npu/NPUGraphsUtils.h")
    capturing = c10_npu::currentStreamCaptureStatusMayInitCtx() != c10_npu::CaptureStatus::None;
#endif
    static thread_local std::unordered_map<c10::DeviceIndex, at::Tensor> masks;
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
