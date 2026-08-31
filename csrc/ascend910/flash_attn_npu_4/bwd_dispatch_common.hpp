/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once

#include "bwd_dispatch.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "torch_npu/csrc/framework/OpCommand.h"

#include "fag_kernel.cpp"

#define BWD_KERNEL_LAUNCH(DT, DTYPE, IS_MASK, IS_DTM, IS_SOFTCAP)                                       \
    do {                                                                                         \
        FAGGeneral<DT, DTYPE, kInputLayout, IS_MASK, 0, IS_DTM, IS_SOFTCAP>                      \
            <<<blockDim, nullptr, aclStream>>>(                                                  \
                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice,                      \
                nullptr, attenMaskDevice, softMaxLseDevice,                                      \
                cuSeqQlenDevice, cuSeqKvlenDevice,                                               \
                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);           \
    } while (0)

// Pick the headdim specialization at runtime.
#define BWD_LAUNCH_HD(DTYPE, IS_MASK, IS_DTM, IS_SOFTCAP)                                  \
    do {                                                                                   \
        switch (qk_headdim_kernel) {                                                       \
            case 64:                                                                       \
                BWD_KERNEL_LAUNCH(DTemplateType::Aligned64, DTYPE, IS_MASK, IS_DTM, IS_SOFTCAP);  \
                break;                                                                     \
            case 128:                                                                      \
                BWD_KERNEL_LAUNCH(DTemplateType::Aligned128, DTYPE, IS_MASK, IS_DTM, IS_SOFTCAP); \
                break;                                                                     \
            case 192:                                                                      \
                BWD_KERNEL_LAUNCH(DTemplateType::Aligned192, DTYPE, IS_MASK, IS_DTM, IS_SOFTCAP); \
                break;                                                                     \
            case 256:                                                                      \
                BWD_KERNEL_LAUNCH(DTemplateType::Aligned256, DTYPE, IS_MASK, IS_DTM, IS_SOFTCAP); \
                break;                                                                     \
            default:                                                                       \
                break;                                                                     \
        }                                                                                  \
    } while (0)

#define BWD_BOOL_SWITCH(COND, CONST_NAME, ...)             \
    do {                                                   \
        if (COND) {                                        \
            constexpr bool CONST_NAME = true;              \
            __VA_ARGS__                                    \
        } else {                                           \
            constexpr bool CONST_NAME = false;             \
            __VA_ARGS__                                    \
        }                                                  \
    } while (0)

template <typename DType, uint32_t kInputLayout>
void bwd_dispatch_run(const BwdLaunchArgs &a) {
    const uint32_t blockDim = a.blockDim;
    const aclrtStream aclStream = a.aclStream;
    const uint64_t fftsAddr = a.fftsAddr;
    const bool is_softcap = a.is_softcap;
    const bool has_attn_mask = a.has_attn_mask;
    const bool deterministic = a.deterministic;
    const uint32_t qk_headdim_kernel = a.qk_headdim_kernel;
    uint8_t *dOutDevice = a.dOutDevice;
    uint8_t *qDevice = a.qDevice;
    uint8_t *kDevice = a.kDevice;
    uint8_t *vDevice = a.vDevice;
    uint8_t *outDevice = a.outDevice;
    uint8_t *attenMaskDevice = a.attenMaskDevice;
    uint8_t *softMaxLseDevice = a.softMaxLseDevice;
    uint8_t *cuSeqQlenDevice = a.cuSeqQlenDevice;
    uint8_t *cuSeqKvlenDevice = a.cuSeqKvlenDevice;
    uint8_t *dqDevice = a.dqDevice;
    uint8_t *dkDevice = a.dkDevice;
    uint8_t *dvDevice = a.dvDevice;
    uint8_t *workspaceDevice = a.workspaceDevice;
    uint8_t *tilingDevice = a.tilingDevice;

    auto launch_fag_general_kernel = [=]() -> int {
        BWD_BOOL_SWITCH(is_softcap, HasSoftcap, {
            BWD_BOOL_SWITCH(has_attn_mask, IsAttenMask, {
                BWD_BOOL_SWITCH(deterministic, IsDtm, {
                    BWD_LAUNCH_HD(DType, IsAttenMask, IsDtm, HasSoftcap);
                });
            });
        });
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fag_general", launch_fag_general_kernel);
}

#undef BWD_BOOL_SWITCH
#undef BWD_LAUNCH_HD
#undef BWD_KERNEL_LAUNCH
