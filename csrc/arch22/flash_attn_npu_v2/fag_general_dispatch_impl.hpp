/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

//
// Shared implementation of the v2 FAGGeneral backward dispatch. Included by the
// generated autogen/fag_general_dispatch_<dtype>_<layout>.cpp stubs, each of
// which explicitly instantiates one launch_fag_general_dispatch_{bf16,fp16}<TND>
// or <BSND>, so the 64 FAGGeneral instantiations land in four parallel-compiled
// object files (16 each) instead of one.
//
// The launch tree reproduces the exact causal x deterministic x headdim
// combinations of LaunchFAGGeneralKernel in fag_general_launch.hpp; the dtype
// dimension is hoisted to a template parameter so each TU instantiates one
// dtype. Template params of ::FAGGeneral map as:
//   <DTemplateType::AlignedNNN, DType, kInputLayout, IS_CAUSAL, 0, IS_DTM>
// where IS_CAUSAL => IS_ATTEN_MASK, IS_DTM => deterministic (IS_DTM).
//
// ADAPTATION vs opt_compiler: main wraps the FAGGeneral launch in an
// at_npu::native::OpCommand::RunOpApiV2("ascendc_fag", ...) context (for aicpu
// op tracking). That wrapper is preserved here. FagGeneralLaunchArgs::is_causal
// carries main's has_attn_mask flag (causal OR local), which selects the
// kernel's IS_ATTEN_MASK template param exactly as main's LaunchFAGGeneralKernel
// did — no semantic change.

#pragma once

#include "fag_general_dispatch.hpp"

// fag_kernel.cpp (and the CATLASS/FAG headers it pulls in, e.g.
// kernel_common_fag.hpp) assume these standard headers are already visible.
// In the original single-TU layout they were supplied transitively by
// fag_tiling.cpp, which is NOT included here (its FAGTiling::* function
// definitions are host-side tiling helpers owned by fag_general_host.cpp;
// pulling them in would create duplicate symbols across the per-dtype TUs).
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
// kernel_common_fag.hpp which it includes). It self-includes fag_tiling.h for
// the FAGTilingData type. Each dtype TU includes it once.
#include "../flash_attn_npu_v3/fag_kernel.cpp"

// Launch one FAGGeneral specialization. IS_CAUSAL / IS_DTM map to the kernel's
// IS_ATTEN_MASK / IS_DTM template params; ALN selects DTemplateType::AlignedNNN.
// Argument order is identical to LaunchFAGGeneralKernel in fag_general_launch.hpp.
#define GEN_LAUNCH(ALN, IS_CAUSAL, IS_DTM, IS_SOFTCAP)                                       \
    ::FAGGeneral<DTemplateType::ALN, DType, kInputLayout, IS_CAUSAL, 0, IS_DTM, IS_SOFTCAP>  \
        <<<a.blockDim, nullptr, a.aclStream>>>(                                              \
            a.fftsAddr, a.dOutDevice, a.qDevice, a.kDevice, a.vDevice,                       \
            a.outDevice, nullptr, a.attenMaskDevice, a.softMaxLseDevice,                     \
            a.cuSeqQlenDevice, a.cuSeqKvlenDevice, a.dqDevice, a.dkDevice,                   \
            a.dvDevice, nullptr, a.workspaceDevice, a.tilingDevice)

template <typename DType, uint32_t kInputLayout>
void launch_fag_general_dispatch_impl(const FagGeneralLaunchArgs &a) {
    // Wrap the launch tree in an OpCommand context, matching main's
    // fag_general_launch.hpp (RunOpApiV2("ascendc_fag", ...)).
    auto fag_general_call = [=]() -> int {
        const uint32_t hd = a.qk_headdim_kernel;
        if (a.is_softcap) {
            if (a.is_causal) {
                if (a.deterministic) {
                    switch (hd) {
                        case 64:  GEN_LAUNCH(Aligned64,  1, 1, 1); break;
                        case 128: GEN_LAUNCH(Aligned128, 1, 1, 1); break;
                        case 192: GEN_LAUNCH(Aligned192, 1, 1, 1); break;
                        case 256: GEN_LAUNCH(Aligned256, 1, 1, 1); break;
                        default: break;
                    }
                } else {
                    switch (hd) {
                        case 64:  GEN_LAUNCH(Aligned64,  1, 0, 1); break;
                        case 128: GEN_LAUNCH(Aligned128, 1, 0, 1); break;
                        case 192: GEN_LAUNCH(Aligned192, 1, 0, 1); break;
                        case 256: GEN_LAUNCH(Aligned256, 1, 0, 1); break;
                        default: break;
                    }
                }
            } else {
                if (a.deterministic) {
                    switch (hd) {
                        case 64:  GEN_LAUNCH(Aligned64,  0, 1, 1); break;
                        case 128: GEN_LAUNCH(Aligned128, 0, 1, 1); break;
                        case 192: GEN_LAUNCH(Aligned192, 0, 1, 1); break;
                        case 256: GEN_LAUNCH(Aligned256, 0, 1, 1); break;
                        default: break;
                    }
                } else {
                    switch (hd) {
                        case 64:  GEN_LAUNCH(Aligned64,  0, 0, 1); break;
                        case 128: GEN_LAUNCH(Aligned128, 0, 0, 1); break;
                        case 192: GEN_LAUNCH(Aligned192, 0, 0, 1); break;
                        case 256: GEN_LAUNCH(Aligned256, 0, 0, 1); break;
                        default: break;
                    }
                }
            }
        } else {
            if (a.is_causal) {
                if (a.deterministic) {
                    switch (hd) {
                        case 64:  GEN_LAUNCH(Aligned64,  1, 1, 0); break;
                        case 128: GEN_LAUNCH(Aligned128, 1, 1, 0); break;
                        case 192: GEN_LAUNCH(Aligned192, 1, 1, 0); break;
                        case 256: GEN_LAUNCH(Aligned256, 1, 1, 0); break;
                        default: break;
                    }
                } else {
                    switch (hd) {
                        case 64:  GEN_LAUNCH(Aligned64,  1, 0, 0); break;
                        case 128: GEN_LAUNCH(Aligned128, 1, 0, 0); break;
                        case 192: GEN_LAUNCH(Aligned192, 1, 0, 0); break;
                        case 256: GEN_LAUNCH(Aligned256, 1, 0, 0); break;
                        default: break;
                    }
                }
            } else {
                if (a.deterministic) {
                    switch (hd) {
                        case 64:  GEN_LAUNCH(Aligned64,  0, 1, 0); break;
                        case 128: GEN_LAUNCH(Aligned128, 0, 1, 0); break;
                        case 192: GEN_LAUNCH(Aligned192, 0, 1, 0); break;
                        case 256: GEN_LAUNCH(Aligned256, 0, 1, 0); break;
                        default: break;
                    }
                } else {
                    switch (hd) {
                        case 64:  GEN_LAUNCH(Aligned64,  0, 0, 0); break;
                        case 128: GEN_LAUNCH(Aligned128, 0, 0, 0); break;
                        case 192: GEN_LAUNCH(Aligned192, 0, 0, 0); break;
                        case 256: GEN_LAUNCH(Aligned256, 0, 0, 0); break;
                        default: break;
                    }
                }
            }
        } 
        
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