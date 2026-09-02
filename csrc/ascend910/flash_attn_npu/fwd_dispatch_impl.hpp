/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once

//
// Light half of the v2 forward (FAInfer) dispatch: selects one of the 32
// generated autogen/fwd_combo_<dtype>_<layout>_softcap<0|1>_retsoftmax<0|1>_
// dropout<0|1> combo functions. Each combo TU (heavy, via
// fwd_kernel_launch.hpp) fixes one (softcap, return_softmax, dropout) triple
// and instantiates the 6 paged x mask FAInfer variants, so the 48
// instantiations of a (dtype, layout) family compile in 8 parallel TUs
// instead of one. This header stays lightweight (no CATLASS / kernel
// includes) so the autogen stubs and host TUs never drag in the kernel
// templates.
//

#include "fwd_dispatch.hpp"

#include <type_traits>

// bfloat16_t must be complete here: the autogen bf16 stub explicitly
// instantiates launch_fwd_impl<bfloat16_t, ...> with only this header's light
// include set (the heavy fwd_kernel_launch.hpp chain is no longer pulled in).
#include "kernel_operator.h"

#include "autogen/fwd_combo_decls.hpp"

template <typename DType, bool IS_TND>
void launch_fwd_impl(const FwdLaunchArgs &a) {
    if constexpr (std::is_same_v<DType, half>) {
        if constexpr (IS_TND) {
            FWD_SELECT_COMBO_8(fp16, tnd);
        } else {
            FWD_SELECT_COMBO_8(fp16, bsnd);
        }
    } else {
        if constexpr (IS_TND) {
            FWD_SELECT_COMBO_8(bf16, tnd);
        } else {
            FWD_SELECT_COMBO_8(bf16, bsnd);
        }
    }
}
