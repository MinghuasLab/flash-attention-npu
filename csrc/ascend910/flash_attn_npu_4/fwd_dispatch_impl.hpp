/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once

#include "fwd_dispatch.hpp"

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

#include "mha_fwd_kvcache.cpp"

// 7-param FAInfer (no IS_FD template arg — main moved flash-decode to tiling).
#define FWD_LAUNCH(DTYPE, PAGED, MASK_TYPE, LAYOUT_TYPE, SOFTCAP)                  \
    SplitFuse::FAInfer<DTYPE, DTYPE, float, PAGED, MASK_TYPE, LAYOUT_TYPE,         \
                       Catlass::Epilogue::LseModeT::OUT_ONLY, SOFTCAP>             \
        <<<launchBlockDim, nullptr, aclStream>>>(                                  \
            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice,     \
            oDevice, softmaxLseDevice, qSeqDevice, kvSeqDevice,                    \
            workspaceDevice, tilingDevice)

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
void launch_fwd_dtype(const FwdLaunchArgs &a) {
    constexpr auto LAYOUT = IS_TND ? FaiKenel::inputLayout::TND : FaiKenel::inputLayout::BSND;

    const uint32_t launchBlockDim = a.launchBlockDim;
    const aclrtStream aclStream = a.aclStream;
    const uint64_t fftsAddr = a.fftsAddr;
    const bool paged_KV = a.paged_KV;
    const bool is_causal = a.is_causal;
    const bool is_local = a.is_local;
    const bool flashDecodeFlag = a.flashDecodeFlag;
    const bool has_softcap = a.has_softcap;
    uint8_t *qDevice = a.qDevice;
    uint8_t *kDevice = a.kDevice;
    uint8_t *vDevice = a.vDevice;
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
                FWD_LAUNCH(DType, IsPaged, MaskType, LAYOUT, HasSoftcap);
            });
        });
    });
}

#undef FWD_MASK_SWITCH
#undef FWD_BOOL_SWITCH
#undef FWD_LAUNCH
