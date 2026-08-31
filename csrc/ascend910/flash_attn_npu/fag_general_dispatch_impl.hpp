/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

//
// Shared implementation of the v2 FAGGeneral backward dispatch. Included by the
// generated autogen/fag_general_dispatch_<dtype>_<layout>.cpp stubs, each of
// which explicitly instantiates one launch_fag_general_dispatch_{bf16,fp16}<TND>
// or <BSND>, so the 256 FAGGeneral instantiations land in four parallel-compiled
// object files (64 each) instead of one.
//
// The launch tree reproduces the exact causal x deterministic x headdim
// combinations of LaunchFAGGeneralKernel in fag_general_launch.hpp; the dtype
// dimension is hoisted to a template parameter so each TU instantiates one
// dtype. Template params of ::FAGGeneral map as:
//   <DTemplateType::AlignedNNN, DType, kInputLayout, IS_CAUSAL, IS_DROP, IS_DTM, IS_SOFTCAP>
// where IS_CAUSAL => IS_ATTEN_MASK, IS_DROP => dropout, IS_DTM => deterministic.
//
// ADAPTATION vs opt_compiler: main wraps the FAGGeneral launch in an
// at_npu::native::OpCommand::RunOpApiV2("ascendc_fag", ...) context (for aicpu
// op tracking). That wrapper is preserved here. FagGeneralLaunchArgs::is_causal
// carries main's has_attn_mask flag (causal OR local), which selects the
// kernel's IS_ATTEN_MASK template param exactly as main's LaunchFAGGeneralKernel
// did — no semantic change.

#pragma once

#include "fag_general_dispatch.hpp"

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

// OpCommand: main wraps the FAGGeneral launch in an OpCommand context for aicpu
// op tracking (RunOpApiV2("ascendc_fag", ...)); preserved through extraction.
#include "torch_npu/csrc/framework/OpCommand.h"

// fag_kernel.cpp provides the ::FAGGeneral kernel template, the DTemplateType
// enum, and the unqualified BSND/TND layout constants (0/1, from
// fag_kernel_common.hpp which it includes). It self-includes fag_tiling.h for
// the FAGTilingData type. Each dtype TU includes it once.
#include "../flash_attn_npu_3/fag_kernel.cpp"

// Launch one FAGGeneral specialization. IS_CAUSAL / IS_DROP / IS_DTM /
// IS_SOFTCAP map to the kernel's template params. Argument order is identical
// to LaunchFAGGeneralKernel in fag_general_launch.hpp; the drop_mask kernel arg
// is a.dropMaskDevice (nullptr when dropout is off).
#define FAG_KERNEL_LAUNCH(DTYPE, IS_CAUSAL, IS_DROP, IS_DTM, IS_SOFTCAP)                             \
    ::FAGGeneral<DTemplateType::DTYPE, DType, kInputLayout, IS_CAUSAL, IS_DROP, IS_DTM, IS_SOFTCAP>  \
        <<<a.blockDim, nullptr, a.aclStream>>>(                                                      \
            a.fftsAddr, a.dOutDevice, a.qDevice, a.kDevice, a.vDevice,                               \
            a.outDevice, a.dropMaskDevice, a.attenMaskDevice, a.softMaxLseDevice,                    \
            a.cuSeqQlenDevice, a.cuSeqKvlenDevice, a.dqDevice, a.dkDevice,                           \
            a.dvDevice, nullptr, a.workspaceDevice, a.tilingDevice)

// Pick the headdim specialization at runtime.
#define FAG_LAUNCH_HD(IS_CAUSAL, IS_DROP, IS_DTM, IS_SOFTCAP)                                          \
    do {                                                                                               \
        switch (a.qk_headdim_kernel) {                                                                 \
            case 64:  FAG_KERNEL_LAUNCH(Aligned64,  IS_CAUSAL, IS_DROP, IS_DTM, IS_SOFTCAP); break;    \
            case 128: FAG_KERNEL_LAUNCH(Aligned128, IS_CAUSAL, IS_DROP, IS_DTM, IS_SOFTCAP); break;    \
            case 192: FAG_KERNEL_LAUNCH(Aligned192, IS_CAUSAL, IS_DROP, IS_DTM, IS_SOFTCAP); break;    \
            case 256: FAG_KERNEL_LAUNCH(Aligned256, IS_CAUSAL, IS_DROP, IS_DTM, IS_SOFTCAP); break;    \
            default: break;                                                                            \
        }                                                                                              \
    } while (0)

#define FAG_BOOL_SWITCH(COND, CONST_NAME, ...)             \
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
void launch_fag_general_dispatch_impl(const FagGeneralLaunchArgs &a) {
    auto fag_general_call = [=]() -> int {
        FAG_BOOL_SWITCH(a.is_softcap, HasSoftcap, {
            FAG_BOOL_SWITCH(a.is_causal, IsAttenMask, {
                FAG_BOOL_SWITCH(a.deterministic, IsDtm, {
                    FAG_BOOL_SWITCH(a.has_dropout, IsDrop, {
                        FAG_LAUNCH_HD(IsAttenMask, IsDrop, IsDtm, HasSoftcap);
                    });
                });
            });
        });
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fag", fag_general_call);
}

template <uint32_t kInputLayout>
void launch_fag_general_dispatch_bf16(const FagGeneralLaunchArgs &a) {
    launch_fag_general_dispatch_impl<bfloat16_t, kInputLayout>(a);
}

template <uint32_t kInputLayout>
void launch_fag_general_dispatch_fp16(const FagGeneralLaunchArgs &a) {
    launch_fag_general_dispatch_impl<half, kInputLayout>(a);
}
