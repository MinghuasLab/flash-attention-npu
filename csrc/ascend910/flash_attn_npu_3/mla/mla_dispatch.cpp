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

extern "C" __attribute__((visibility("hidden"))) aclError __wrap_aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle function, uint32_t blocks, aclrtStream stream, aclrtLaunchKernelCfg* config, void* args,
    size_t argsSize, aclrtPlaceHolderInfo* placeholders, size_t placeholderCount)
{
    auto* state = flash_attn::mla::activeMlaLaunch;
    if (!state) {
        return __real_aclrtLaunchKernelWithHostArgs(function, blocks, stream, config, args, argsSize, placeholders,
                                                    placeholderCount);
    }
    ++state->calls;
    // CANN 9.0 defines eight launch attributes. Keep every non-mode attribute,
    // including dynamic UB, engine, timeout and dump settings.
    aclrtLaunchKernelAttr attrs[9]{};
    size_t count = 0;
    if (config) {
        if (config->numAttrs > 8 || (config->numAttrs && !config->attrs)) {
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
    state->error = __real_aclrtLaunchKernelWithHostArgs(function, blocks, stream, &batchConfig, args, argsSize,
                                                        placeholders, placeholderCount);
    return state->error;
}
