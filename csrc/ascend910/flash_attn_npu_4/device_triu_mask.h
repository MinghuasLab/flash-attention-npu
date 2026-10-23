#ifndef FLASH_ATTN_DEVICE_TRIU_MASK_H
#define FLASH_ATTN_DEVICE_TRIU_MASK_H

#include <unordered_map>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

// Initialize and synchronize only the device used by this caller.
// Per-thread device caches avoid touching other workers' devices.
// The map is leaked: destroying it on thread exit, process exit, or in a
// forked child would free NPU storage after the runtime may already be gone.
inline at::Tensor MakeDeviceTriuMask()
{
    const auto idx = c10_npu::getCurrentNPUStream().device_index();
    static thread_local auto* masks = new std::unordered_map<c10::DeviceIndex, at::Tensor>();
    auto found = masks->find(idx);
    if (found != masks->end()) {
        return found->second;
    }
    c10_npu::NPUGuard guard(idx);
    constexpr int64_t dim = 2048;
    at::Tensor mask =
        at::ones({dim, dim}, at::TensorOptions().dtype(at::kByte).device(at::Device(at::kPrivateUse1, idx)));
    mask.triu_(/*diagonal=*/1);
    c10_npu::getCurrentNPUStream().synchronize();
    return masks->emplace(idx, std::move(mask)).first->second;
}

#endif