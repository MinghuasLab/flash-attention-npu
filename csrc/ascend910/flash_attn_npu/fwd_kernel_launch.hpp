/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once

//
// Heavy half of the v2 forward (FAInfer) dispatch: the kernel definitions
// (via mha_fwd_kvcache.cpp) plus the launch / switch macros. Included ONLY by
// the generated autogen/fwd_combo_<dtype>_<layout>_softcap<0|1>_retsoftmax<0|1>_
// dropout<0|1>.cpp
// combo TUs, each of which fixes one (softcap, return_softmax, dropout) triple
// and instantiates the 6 paged x mask FAInfer variants, so the 48
// instantiations of a (dtype, layout) family compile in 8 parallel TUs
// instead of one. The light half (combo selection) lives in
// fwd_dispatch_impl.hpp, which host TUs include without pulling in the
// kernel templates.
//

#include "fwd_dispatch.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "mha_fwd_kvcache.cpp"

// FAInfer (no IS_FD template arg — flash-decode moved to tiling). MASK_TYPE and
// LAYOUT are named enum values fixed by the switch macros / generated combo.
// RETURN_SOFTMAX and DROPOUT are the forward's optional epilogue / dropout
// template axes. All runtime arguments are read from the FwdLaunchArgs `a`.
// This repo's FAInfer has no append-KV (kNew/vNew) parameters.
#define FWD_KERNEL_LAUNCH(DTYPE, PAGED, MASK_TYPE, LAYOUT, SOFTCAP, RETURN_SOFTMAX, DROPOUT) \
    SplitFuse::FAInfer<DTYPE, DTYPE, float, PAGED, MASK_TYPE, LAYOUT,                        \
                       Catlass::Epilogue::LseModeT::OUT_ONLY, SOFTCAP, RETURN_SOFTMAX, DROPOUT> \
        <<<a.blockDim, nullptr, a.aclStream>>>(                                              \
            a.fftsAddr, a.qDevice, a.kDevice, a.vDevice, a.maskDevice, a.blockTableDevice,   \
            a.oDevice, a.softmaxLseDevice, a.qSeqDevice, a.kvSeqDevice,                      \
            a.workspaceDevice, a.tilingDevice);

// BOOL_SWITCH-style helper (idea from static_switch.h in flash-attention): each
// branch fixes the runtime bool as a named constexpr flag, so the dispatch
// below reads named flags instead of bare true/false literals. Statement-form
// (no lambda/return) to match the statement-style macros already used here;
// both branches are compiled, so the set of FAInfer instantiations is
// unchanged.
#define FWD_BOOL_SWITCH(COND, CONST_NAME, ...)             \
    do {                                                   \
        if (COND) {                                        \
            constexpr bool CONST_NAME = true;              \
            __VA_ARGS__                                    \
        } else {                                           \
            constexpr bool CONST_NAME = false;             \
            __VA_ARGS__                                    \
        }                                                  \
    } while (0)

// Three-way mask selection with the original precedence: local (MASK_SWA) >
// causal (MASK_CAUSAL) > NO_MASK. Fixes the chosen enum as a named constexpr
// so the launch reads MaskType instead of a bare enumerator token.
#define FWD_MASK_SWITCH(IS_LOCAL, IS_CAUSAL, CONST_NAME, ...)             \
    do {                                                                  \
        if (IS_LOCAL) {                                                   \
            constexpr auto CONST_NAME = FaiKenel::MaskType::MASK_SWA;     \
            __VA_ARGS__                                                   \
        } else if (IS_CAUSAL) {                                           \
            constexpr auto CONST_NAME = FaiKenel::MaskType::MASK_CAUSAL;  \
            __VA_ARGS__                                                   \
        } else {                                                          \
            constexpr auto CONST_NAME = FaiKenel::MaskType::NO_MASK;      \
            __VA_ARGS__                                                   \
        }                                                                 \
    } while (0)
