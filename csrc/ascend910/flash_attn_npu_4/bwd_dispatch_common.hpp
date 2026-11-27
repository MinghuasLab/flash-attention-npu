// Copyright (c) 2026, Minghua Shen.

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

// The FAGGeneral launch below enters the aicpu op-tracking context through
// fa_bridge::run_op_api_v2(), which the host TU flash_api.cpp implements with
// at_npu::native::OpCommand::RunOpApiV2. This kernel-side header must stay
// free of torch/torch_npu includes: the ASC frontend cannot parse torch >= 2.14's
// C++20 headers.
#include <functional>

namespace fa_bridge {
void run_op_api_v2(const char* op_name, const std::function<int()>& func);
} // namespace fa_bridge

#include "fag_kernel.cpp"

#define BWD_KERNEL_LAUNCH(DT, DTYPE, IS_MASK, IS_DTM, IS_SOFTCAP, ACTUAL_OUT)                                          \
    do {                                                                                                               \
        FAGGeneral<DT, DTYPE, kInputLayout, IS_MASK, 0, IS_DTM, IS_SOFTCAP, ACTUAL_OUT>                                \
            <<<blockDim, nullptr, aclStream>>>(fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr,    \
                                               attenMaskDevice, softMaxLseDevice, cuSeqQlenDevice, cuSeqKvlenDevice,   \
                                               seqUsedQDevice, seqUsedKvDevice, dqDevice, dkDevice, dvDevice,          \
                                               workspaceDevice, tilingDevice);                                         \
    } while (0)

// Pick the headdim specialization at runtime. qk > v uses Mla192Dv128
// (the host guarantees qk <= 192 and v <= 128). Equal widths, including
// those padded up to 512, keep their own bucket.
#define BWD_LAUNCH_HD(DTYPE, IS_MASK, IS_DTM, IS_SOFTCAP, ACTUAL_OUT)                                                  \
    do {                                                                                                               \
        if (qk_headdim_kernel > v_headdim_kernel) {                                                                    \
            BWD_KERNEL_LAUNCH(DTemplateType::Mla192Dv128, DTYPE, IS_MASK, IS_DTM, IS_SOFTCAP, false);                  \
            break;                                                                                                     \
        }                                                                                                              \
        switch (qk_headdim_kernel) {                                                                                   \
        case 64:                                                                                                       \
            BWD_KERNEL_LAUNCH(DTemplateType::Aligned64, DTYPE, IS_MASK, IS_DTM, IS_SOFTCAP, ACTUAL_OUT);               \
            break;                                                                                                     \
        case 128:                                                                                                      \
            BWD_KERNEL_LAUNCH(DTemplateType::Aligned128, DTYPE, IS_MASK, IS_DTM, IS_SOFTCAP, ACTUAL_OUT);              \
            break;                                                                                                     \
        case 192:                                                                                                      \
            BWD_KERNEL_LAUNCH(DTemplateType::Aligned192, DTYPE, IS_MASK, IS_DTM, IS_SOFTCAP, ACTUAL_OUT);              \
            break;                                                                                                     \
        case 256:                                                                                                      \
            BWD_KERNEL_LAUNCH(DTemplateType::Aligned256, DTYPE, IS_MASK, IS_DTM, IS_SOFTCAP, ACTUAL_OUT);              \
            break;                                                                                                     \
        case 512:                                                                                                      \
            BWD_KERNEL_LAUNCH(DTemplateType::Aligned512, DTYPE, IS_MASK, IS_DTM, IS_SOFTCAP, ACTUAL_OUT);              \
            break;                                                                                                     \
        default:                                                                                                       \
            break;                                                                                                     \
        }                                                                                                              \
    } while (0)

#define BWD_BOOL_SWITCH(COND, CONST_NAME, ...)                                                                         \
    do {                                                                                                               \
        if (COND) {                                                                                                    \
            constexpr bool CONST_NAME = true;                                                                          \
            __VA_ARGS__                                                                                                \
        } else {                                                                                                       \
            constexpr bool CONST_NAME = false;                                                                         \
            __VA_ARGS__                                                                                                \
        }                                                                                                              \
    } while (0)

template <typename DType, uint32_t kInputLayout>
void bwd_dispatch_run(const BwdLaunchArgs& a)
{
    const uint32_t blockDim = a.blockDim;
    const aclrtStream aclStream = a.aclStream;
    const uint64_t fftsAddr = a.fftsAddr;
    const bool is_softcap = a.is_softcap;
    const bool has_attn_mask = a.has_attn_mask;
    const bool deterministic = a.deterministic;
    const uint32_t qk_headdim_kernel = a.qk_headdim_kernel;
    const uint32_t v_headdim_kernel = a.v_headdim_kernel;
    uint8_t* dOutDevice = a.dOutDevice;
    uint8_t* qDevice = a.qDevice;
    uint8_t* kDevice = a.kDevice;
    uint8_t* vDevice = a.vDevice;
    uint8_t* outDevice = a.outDevice;
    uint8_t* attenMaskDevice = a.attenMaskDevice;
    uint8_t* softMaxLseDevice = a.softMaxLseDevice;
    uint8_t* cuSeqQlenDevice = a.cuSeqQlenDevice;
    uint8_t* cuSeqKvlenDevice = a.cuSeqKvlenDevice;
    uint8_t* seqUsedQDevice = a.seqUsedQDevice;
    uint8_t* seqUsedKvDevice = a.seqUsedKvDevice;
    uint8_t* dqDevice = a.dqDevice;
    uint8_t* dkDevice = a.dkDevice;
    uint8_t* dvDevice = a.dvDevice;
    uint8_t* workspaceDevice = a.workspaceDevice;
    uint8_t* tilingDevice = a.tilingDevice;

    auto launch_fag_general_kernel = [=]() -> int {
        BWD_BOOL_SWITCH(is_softcap, HasSoftcap, {
            BWD_BOOL_SWITCH(has_attn_mask, IsAttenMask, {
                BWD_BOOL_SWITCH(deterministic, IsDtm, {
                    BWD_BOOL_SWITCH(a.original_qk_gt_v, UseActualOut,
                                    { BWD_LAUNCH_HD(DType, IsAttenMask, IsDtm, HasSoftcap, UseActualOut); });
                });
            });
        });
        return 0;
    };
    fa_bridge::run_op_api_v2("ascendc_fag_general", launch_fag_general_kernel);
}

#undef BWD_BOOL_SWITCH
#undef BWD_LAUNCH_HD
#undef BWD_KERNEL_LAUNCH

// Each autogen TU ends with the explicit instantiation definition of its own
// (dtype, layout) pair. Declaring every pair extern here keeps
// launch_bwd_dispatch() below from instantiating all six of them in every TU
// that includes this header, which would undo the per-TU split.
extern template void bwd_dispatch_run<bfloat16_t, 0>(const BwdLaunchArgs& a);
extern template void bwd_dispatch_run<bfloat16_t, 1>(const BwdLaunchArgs& a);
extern template void bwd_dispatch_run<bfloat16_t, 2>(const BwdLaunchArgs& a);
extern template void bwd_dispatch_run<half, 0>(const BwdLaunchArgs& a);
extern template void bwd_dispatch_run<half, 1>(const BwdLaunchArgs& a);
extern template void bwd_dispatch_run<half, 2>(const BwdLaunchArgs& a);

// Dtype-agnostic host entry (declared in bwd_dispatch.hpp): is_bf16 x layout
// selection happens here, device-side, so host TUs never name the ASC dtype
// types. BSND=0 / TND=1 / BNSD=2 (fag_kernel_common.hpp). gnu::used forces
// emission (no device TU calls it directly).
[[gnu::used]] inline void launch_bwd_dispatch(uint32_t layout, const BwdLaunchArgs& a)
{
    if (a.is_bf16) {
        if (layout == 0) {
            bwd_dispatch_run<bfloat16_t, 0>(a);
        } else if (layout == 1) {
            bwd_dispatch_run<bfloat16_t, 1>(a);
        } else {
            bwd_dispatch_run<bfloat16_t, 2>(a);
        }
    } else {
        if (layout == 0) {
            bwd_dispatch_run<half, 0>(a);
        } else if (layout == 1) {
            bwd_dispatch_run<half, 1>(a);
        } else {
            bwd_dispatch_run<half, 2>(a);
        }
    }
}
