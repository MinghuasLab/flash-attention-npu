/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once

//
// Heavy half of the v2 FAGGeneral backward dispatch: the kernel definition
// (via ../flash_attn_npu_3/fag_kernel.cpp), the OpCommand wrapper include and
// the launch / switch macros. Included ONLY by the generated
// autogen/fag_combo_<dtype>_<layout>_headdim<64|128|192|256>_softcap<0|1>.cpp
// combo TUs,
// each of which fixes one (headdim, softcap) pair and instantiates the 8
// causal x deterministic x dropout FAGGeneral variants, so the 64
// instantiations of a (dtype, layout) family compile in 8 parallel TUs
// instead of one. The light half (combo selection) lives in
// fag_general_dispatch_impl.hpp.
//
// ADAPTATION vs opt_compiler: main wraps the FAGGeneral launch in an
// at_npu::native::OpCommand::RunOpApiV2("ascendc_fag", ...) context (for aicpu
// op tracking). That wrapper is preserved per combo launcher. The op name is
// exactly "ascendc_fag" for v2.
//

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
// kernel_common_fag.hpp which it includes). It self-includes fag_tiling.h for
// the FAGTilingData type. Each combo TU includes it once.
#include "../flash_attn_npu_3/fag_kernel.cpp"

// Launch one FAGGeneral specialization. All template axes are macro arguments
// (no free identifiers), and every runtime argument is read from the
// FagGeneralLaunchArgs `a` of the enclosing combo launcher. The drop_mask
// kernel arg is a.dropMaskDevice (nullptr when dropout is off).
#define FAG_KERNEL_LAUNCH(DTYPE, KINPUTLAYOUT, DT_ALIGNED, IS_CAUSAL, IS_DROP, IS_DTM, IS_SOFTCAP)   \
    do {                                                                                             \
        ::FAGGeneral<DTemplateType::DT_ALIGNED, DTYPE, KINPUTLAYOUT, IS_CAUSAL, IS_DROP, IS_DTM, IS_SOFTCAP> \
            <<<a.blockDim, nullptr, a.aclStream>>>(                                                  \
                a.fftsAddr, a.dOutDevice, a.qDevice, a.kDevice, a.vDevice,                           \
                a.outDevice, a.dropMaskDevice, a.attenMaskDevice, a.softMaxLseDevice,                \
                a.cuSeqQlenDevice, a.cuSeqKvlenDevice, a.dqDevice, a.dkDevice,                       \
                a.dvDevice, nullptr, a.workspaceDevice, a.tilingDevice);                             \
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
