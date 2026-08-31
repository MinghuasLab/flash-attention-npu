/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once

#include "fwd_dispatch.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "mha_fwd_kvcache.cpp"

// FAInfer (no IS_FD template arg — flash-decode moved to tiling). MASK_TYPE and
// LAYOUT are named enum values fixed by the switch macros below. RETURN_SOFTMAX
// and DROPOUT are the forward's optional epilogue / dropout template axes.
#define FWD_KERNEL_LAUNCH(DTYPE, PAGED, MASK_TYPE, LAYOUT, SOFTCAP, RETURN_SOFTMAX, DROPOUT) \
    SplitFuse::FAInfer<DTYPE, DTYPE, float, PAGED, MASK_TYPE, LAYOUT,                        \
                       Catlass::Epilogue::LseModeT::OUT_ONLY, SOFTCAP, RETURN_SOFTMAX, DROPOUT> \
        <<<blockDim, nullptr, aclStream>>>(                                                  \
            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice,               \
            oDevice, softmaxLseDevice, qSeqDevice, kvSeqDevice,                              \
            workspaceDevice, tilingDevice, kNewDevice, vNewDevice);

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

template <typename DType, bool IS_TND>
void launch_fwd_impl(const FwdLaunchArgs &a) {
    constexpr auto LAYOUT = IS_TND ? FaiKenel::inputLayout::TND : FaiKenel::inputLayout::BSND;

    const uint32_t blockDim = a.blockDim;
    const aclrtStream aclStream = a.aclStream;
    const uint64_t fftsAddr = a.fftsAddr;
    const bool paged_KV = a.paged_KV;
    const bool is_causal = a.is_causal;
    const bool is_local = a.is_local;
    const bool flashDecodeFlag = a.flashDecodeFlag;
    const bool has_softcap = a.has_softcap;
    const bool return_softmax = a.return_softmax;
    const bool has_dropout = a.has_dropout;
    uint8_t *qDevice = a.qDevice;
    uint8_t *kDevice = a.kDevice;
    uint8_t *vDevice = a.vDevice;
    uint8_t *kNewDevice = a.kNewDevice;
    uint8_t *vNewDevice = a.vNewDevice;
    uint8_t *maskDevice = a.maskDevice;
    uint8_t *blockTableDevice = a.blockTableDevice;
    uint8_t *oDevice = a.oDevice;
    uint8_t *softmaxLseDevice = a.softmaxLseDevice;
    uint8_t *qSeqDevice = a.qSeqDevice;
    uint8_t *kvSeqDevice = a.kvSeqDevice;
    uint8_t *workspaceDevice = a.workspaceDevice;
    uint8_t *tilingDevice = a.tilingDevice;
    (void)flashDecodeFlag;

    FWD_BOOL_SWITCH(paged_KV, IsPaged, {
        FWD_MASK_SWITCH(is_local, is_causal, MaskType, {
            FWD_BOOL_SWITCH(has_softcap, HasSoftcap, {
                FWD_BOOL_SWITCH(return_softmax, ReturnSoftmax, {
                    FWD_BOOL_SWITCH(has_dropout, HasDropout, {
                        FWD_KERNEL_LAUNCH(DType, IsPaged, MaskType, LAYOUT, HasSoftcap, ReturnSoftmax, HasDropout);
                    });
                });
            });
        });
    });
}

#undef FWD_MASK_SWITCH
#undef FWD_BOOL_SWITCH
#undef FWD_KERNEL_LAUNCH
