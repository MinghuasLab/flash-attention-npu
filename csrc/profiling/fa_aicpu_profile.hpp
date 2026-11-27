/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Modified by Minghua Shen, 2026.
 */

#pragma once

#include <utility>

#include "fa_profile.hpp"

namespace fa_profile __attribute__((visibility("hidden"))) {

    // The generated AICPU stub can load and register the binary before
    // reaching aclrtLaunchKernelWithConfig. Keep the descriptors on its host
    // thread and start the profiling range at the actual launch API.
    inline thread_local const TensorInfo* pending_aicpu_profile = nullptr;

    class PendingAicpuProfile {
      public:
        explicit PendingAicpuProfile(const TensorInfo& profile)
            : previous_(std::exchange(pending_aicpu_profile, &profile))
        {}

        PendingAicpuProfile(const PendingAicpuProfile&) = delete;
        PendingAicpuProfile& operator=(const PendingAicpuProfile&) = delete;

        ~PendingAicpuProfile()
        {
            pending_aicpu_profile = previous_;
        }

      private:
        const TensorInfo* previous_;
    };

    template <typename Launch>
    void LaunchAicpuWithProfile(const TensorInfo& profile, Launch&& launch)
    {
        if (IsEnabled()) {
            const PendingAicpuProfile pending(profile);
            launch();
        } else {
            launch();
        }
    }

} // namespace fa_profile

extern "C" aclError __real_aclrtLaunchKernelWithConfig(aclrtFuncHandle function, uint32_t blocks, aclrtStream stream,
                                                       aclrtLaunchKernelCfg* config, aclrtArgsHandle args,
                                                       void* reserve);

// setup.py wraps only calls made from this extension to the ACL function.
extern "C" __attribute__((visibility("hidden"))) aclError
__wrap_aclrtLaunchKernelWithConfig(aclrtFuncHandle function, uint32_t blocks, aclrtStream stream,
                                   aclrtLaunchKernelCfg* config, aclrtArgsHandle args, void* reserve)
{
    const auto launch = [&] {
        return __real_aclrtLaunchKernelWithConfig(function, blocks, stream, config, args, reserve);
    };
    const auto* profile = std::exchange(fa_profile::pending_aicpu_profile, nullptr);
    if (profile == nullptr) {
        return launch();
    }

    aclError status = ACL_SUCCESS;
    (*profile)(stream, blocks, [&] { status = launch(); });
    return status;
}
