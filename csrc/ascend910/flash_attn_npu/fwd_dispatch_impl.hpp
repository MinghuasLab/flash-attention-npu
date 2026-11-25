// Copyright (c) 2026, Minghua Shen.

#pragma once

//
// Light half of the v2 forward (FAInfer) dispatch: selects one of the 48
// generated autogen/fwd_combo_<dtype>_<layout>[_softcap][_retsoftmax]
// [_dropout] combo functions. Each combo TU (heavy, via
// fwd_kernel_launch.hpp) fixes one (softcap, return_softmax, dropout) triple
// and instantiates the 6 paged x mask FAInfer variants, so the 48
// instantiations of a (dtype, layout) family compile in 8 parallel TUs
// instead of one. This header stays lightweight (no CATLASS includes) so the
// autogen stubs and host TUs never drag in the kernel templates.
//

#include "fwd_dispatch.hpp"

#include <type_traits>

// bfloat16_t must be complete here: the autogen bf16 stub explicitly
// instantiates launch_fwd_impl<bfloat16_t, ...> with only this header's light
// include set (the heavy fwd_kernel_launch.hpp chain is no longer pulled in).
#include "kernel_operator.h"

#include "autogen/fwd_combo_decls.hpp"

template <typename DType, KernelCommon::FaiKenel::inputLayout kLayout>
void launch_fwd_impl(const FwdLaunchArgs& a)
{
    if constexpr (std::is_same_v<DType, half>) {
        if constexpr (kLayout == KernelCommon::FaiKenel::inputLayout::TND) {
            FWD_SELECT_COMBO_8(fp16, tnd);
        } else if constexpr (kLayout == KernelCommon::FaiKenel::inputLayout::BNSD) {
            FWD_SELECT_COMBO_8(fp16, bnsd);
        } else {
            FWD_SELECT_COMBO_8(fp16, bsnd);
        }
    } else {
        if constexpr (kLayout == KernelCommon::FaiKenel::inputLayout::TND) {
            FWD_SELECT_COMBO_8(bf16, tnd);
        } else if constexpr (kLayout == KernelCommon::FaiKenel::inputLayout::BNSD) {
            FWD_SELECT_COMBO_8(bf16, bnsd);
        } else {
            FWD_SELECT_COMBO_8(bf16, bsnd);
        }
    }
}

// Each autogen TU ends with the explicit instantiation definition of its own
// (dtype, layout) pair. Declaring every pair extern here keeps
// launch_fwd_dispatch() below from instantiating all six of them in every TU
// that includes this header, which would undo the per-TU split.
extern template void launch_fwd_impl<bfloat16_t, KernelCommon::FaiKenel::inputLayout::BSND>(const FwdLaunchArgs& a);
extern template void launch_fwd_impl<bfloat16_t, KernelCommon::FaiKenel::inputLayout::TND>(const FwdLaunchArgs& a);
extern template void launch_fwd_impl<bfloat16_t, KernelCommon::FaiKenel::inputLayout::BNSD>(const FwdLaunchArgs& a);
extern template void launch_fwd_impl<half, KernelCommon::FaiKenel::inputLayout::BSND>(const FwdLaunchArgs& a);
extern template void launch_fwd_impl<half, KernelCommon::FaiKenel::inputLayout::TND>(const FwdLaunchArgs& a);
extern template void launch_fwd_impl<half, KernelCommon::FaiKenel::inputLayout::BNSD>(const FwdLaunchArgs& a);

// Dtype/layout-agnostic host entry (declared in fwd_dispatch.hpp): the
// is_bf16 x layout selection happens here, device-side, so host TUs never
// name the ASC dtype types. gnu::used forces emission: no device TU calls
// it directly, and host TUs only see the declaration.
[[gnu::used]] inline void launch_fwd_dispatch(const FwdLaunchArgs& a)
{
    if (a.is_bf16) {
        if (a.layout == KernelCommon::FaiKenel::inputLayout::TND) {
            launch_fwd_impl<bfloat16_t, KernelCommon::FaiKenel::inputLayout::TND>(a);
        } else if (a.layout == KernelCommon::FaiKenel::inputLayout::BNSD) {
            launch_fwd_impl<bfloat16_t, KernelCommon::FaiKenel::inputLayout::BNSD>(a);
        } else {
            launch_fwd_impl<bfloat16_t, KernelCommon::FaiKenel::inputLayout::BSND>(a);
        }
    } else {
        if (a.layout == KernelCommon::FaiKenel::inputLayout::TND) {
            launch_fwd_impl<half, KernelCommon::FaiKenel::inputLayout::TND>(a);
        } else if (a.layout == KernelCommon::FaiKenel::inputLayout::BNSD) {
            launch_fwd_impl<half, KernelCommon::FaiKenel::inputLayout::BNSD>(a);
        } else {
            launch_fwd_impl<half, KernelCommon::FaiKenel::inputLayout::BSND>(a);
        }
    }
}
