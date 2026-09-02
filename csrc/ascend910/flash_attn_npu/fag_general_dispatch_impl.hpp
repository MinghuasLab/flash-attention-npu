/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once

//
// Light half of the v2 FAGGeneral backward dispatch: selects one of the 32
// generated autogen/fag_combo_<dtype>_<layout>_headdim<64|128|192|256>_softcap<0|1>
// combo
// functions. Each combo TU (heavy, via fag_kernel_launch.hpp) fixes one
// (headdim, softcap) pair and instantiates the 8 causal x deterministic x
// dropout FAGGeneral variants, so the 64 instantiations of a (dtype, layout)
// family compile in 8 parallel TUs instead of one. This header stays
// lightweight (no CATLASS / kernel includes) so the autogen stubs and host
// TUs never drag in the kernel templates.
//
// Layout constants come from the self-contained fag_layout.hpp (extracted
// from kernel_common_fag.hpp so the light stubs can use the named constants
// without that header's kernel-header prerequisites). Host TUs keep getting
// the constants from their own heavy include chains.
//
// The OpCommand("ascendc_fag") wrapper moved into the per-combo launchers
// (fag_kernel_launch.hpp users): exactly one combo runs per dispatch, so the
// wrapper still fires once per launch, as before.
//

#include "fag_general_dispatch.hpp"

#include "fag_layout.hpp"

#include <type_traits>

// bfloat16_t must be complete here: the autogen bf16 stub instantiates the
// bf16 wrapper (which names bfloat16_t via launch_fag_general_dispatch_impl)
// with only this header's light include set.
#include "kernel_operator.h"

#include "autogen/fag_combo_decls.hpp"

template <typename DType, uint32_t kInputLayout>
void launch_fag_general_dispatch_impl(const FagGeneralLaunchArgs &a) {
    if constexpr (std::is_same_v<DType, half>) {
        if constexpr (kInputLayout == TND) {
            FAG_SELECT_COMBO_16(fp16, tnd);
        } else {  // BSND
            FAG_SELECT_COMBO_16(bf16, bsnd);
        }
    } else {
        if constexpr (kInputLayout == TND) {
            FAG_SELECT_COMBO_16(bf16, tnd);
        } else {  // BSND
            FAG_SELECT_COMBO_16(bf16, bsnd);
        }
    }
}

template <uint32_t kInputLayout>
void launch_fag_general_dispatch_bf16(const FagGeneralLaunchArgs &a) {
    launch_fag_general_dispatch_impl<bfloat16_t, kInputLayout>(a);
}

template <uint32_t kInputLayout>
void launch_fag_general_dispatch_fp16(const FagGeneralLaunchArgs &a) {
    launch_fag_general_dispatch_impl<half, kInputLayout>(a);
}
