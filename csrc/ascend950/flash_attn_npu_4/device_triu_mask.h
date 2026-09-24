/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef FLASH_ATTN_NPU_4_DEVICE_TRIU_MASK_H
#define FLASH_ATTN_NPU_4_DEVICE_TRIU_MASK_H

#include <mutex>
#include <vector>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

// Cache the fixed 2048x2048 uint8 causal mask once per visible NPU. Every
// device is initialized before the cache becomes visible to other threads.
inline at::Tensor MakeDeviceTriuMask()
{
    const auto current_device = c10_npu::getCurrentNPUStream().device_index();
    static std::once_flag once;
    static std::vector<at::Tensor> masks;

    std::call_once(once, []() {
        const auto device_count = c10_npu::device_count();
        masks.resize(static_cast<size_t>(device_count));
        constexpr int64_t dim = 2048;
        for (c10::DeviceIndex device = 0; device < device_count; ++device) {
            c10_npu::NPUGuard guard(device);
            auto mask =
                at::ones({dim, dim}, at::TensorOptions().dtype(at::kByte).device(at::Device(at::kPrivateUse1, device)));
            mask.triu_(/*diagonal=*/1);
            masks[static_cast<size_t>(device)] = std::move(mask);
        }
        for (c10::DeviceIndex device = 0; device < device_count; ++device) {
            c10_npu::NPUGuard guard(device);
            c10_npu::getCurrentNPUStream().synchronize();
        }
    });

    return masks[static_cast<size_t>(current_device)];
}

#endif
