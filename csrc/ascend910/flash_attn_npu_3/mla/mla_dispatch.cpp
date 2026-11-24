/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#include "mla_dispatch.h"

namespace flash_attn::mla {
thread_local MlaBatchLaunchState* activeMlaLaunch = nullptr;
}

extern "C" aclError __real_aclrtLaunchKernelWithHostArgs(aclrtFuncHandle, uint32_t, aclrtStream, aclrtLaunchKernelCfg*,
                                                         void*, size_t, aclrtPlaceHolderInfo*, size_t);
extern "C" aclError __real_aclrtLaunchKernelWithArgsArray(void*, uint32_t, aclrtStream, aclrtLaunchKernelCfg*, void**)
    __attribute__((weak));

namespace {
template <typename Launch>
aclError LaunchWithMlaConfig(aclrtLaunchKernelCfg* config, const Launch& launch)
{
    auto* state = flash_attn::mla::activeMlaLaunch;
    if (!state) {
        return launch(config);
    }
    ++state->calls;
    // Preserve runtime attributes, including dynamic UB, engine, timeout,
    // profiling and dump settings, while overriding only the scheduling mode.
    constexpr size_t capacity = 16;
    aclrtLaunchKernelAttr attrs[capacity]{};
    size_t count = 0;
    if (config) {
        if (config->numAttrs >= capacity || (config->numAttrs && !config->attrs)) {
            state->error = ACL_ERROR_INVALID_PARAM;
            return state->error;
        }
        for (size_t i = 0; i < config->numAttrs; ++i) {
            if (config->attrs[i].id != ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE) {
                attrs[count++] = config->attrs[i];
            }
        }
    }
    attrs[count].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attrs[count++].value.schemMode = 1; // fixed CANN MLA ScheduleMode::BATCH_MODE
    aclrtLaunchKernelCfg batchConfig{attrs, count};
    state->error = launch(&batchConfig);
    return state->error;
}
} // namespace

extern "C" __attribute__((visibility("hidden"))) aclError __wrap_aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle function, uint32_t blocks, aclrtStream stream, aclrtLaunchKernelCfg* config, void* args,
    size_t argsSize, aclrtPlaceHolderInfo* placeholders, size_t placeholderCount)
{
    return LaunchWithMlaConfig(config, [&](aclrtLaunchKernelCfg* actual) {
        return __real_aclrtLaunchKernelWithHostArgs(function, blocks, stream, actual, args, argsSize, placeholders,
                                                    placeholderCount);
    });
}

extern "C" __attribute__((visibility("hidden"))) aclError __wrap_aclrtLaunchKernelWithArgsArray(
    void* function, uint32_t blocks, aclrtStream stream, aclrtLaunchKernelCfg* config, void** args)
{
    return LaunchWithMlaConfig(config, [&](aclrtLaunchKernelCfg* actual) {
        if (!__real_aclrtLaunchKernelWithArgsArray)
            return ACL_ERROR_INVALID_PARAM;
        return __real_aclrtLaunchKernelWithArgsArray(function, blocks, stream, actual, args);
    });
}
