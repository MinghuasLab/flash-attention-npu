/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
//
// RE-DERIVED for main (not ported from opt_compiler): main's v3-910 forward
// unifies TND+BSND FAInfer behind a single host function and selects layout at
// runtime via is_varlen_q (varlen => TND, else BSND). main also dropped the
// FAInfer flash-decode template parameter — FD is now a tiling-axis handled by
// fa_split (flashDecodeFlag only adjusts launchBlockDim / tiling, never a
// template arg). So FAInfer here takes 7 template params
// <DType, DType, float, PAGED, MASK, LAYOUT, OUT_ONLY>, and the per-(dtype,
// layout) TU instantiates 6 variants (paged x {SWA, causal, no-mask}).

#pragma once

#include "fwd_dispatch.hpp"

// Standard headers that the CATLASS/FAG headers (reached via mha_fwd_kvcache.cpp)
// assume to be already visible. In the original single-TU layout these were
// supplied transitively; supply them explicitly here.
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
#define FWD_LAUNCH(DTYPE, PAGED, MASK, LAYOUT)                                     \
    SplitFuse::FAInfer<DTYPE, DTYPE, float, PAGED,                                 \
                       FaiKenel::MaskType::MASK, FaiKenel::inputLayout::LAYOUT,    \
                       Catlass::Epilogue::LseModeT::OUT_ONLY>                      \
        <<<launchBlockDim, nullptr, aclStream>>>(                                  \
            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice,      \
            oDevice, softmaxLseDevice, qSeqDevice, kvSeqDevice,                    \
            workspaceDevice, tilingDevice)

// IS_TND selects layout at compile time (false => BSND, true => TND); each
// (dtype, IS_TND) pair is explicitly instantiated in its own autogen TU, so
// only the matching if constexpr branch is kept. flashDecodeFlag is extracted
// for symmetry with FwdLaunchArgs but is NOT a template axis here (FD is
// resolved in tiling by fa_split before this launch).
template <typename DType, bool IS_TND>
void launch_fwd_dtype(const FwdLaunchArgs &a) {
    const uint32_t launchBlockDim = a.launchBlockDim;
    const aclrtStream aclStream = a.aclStream;
    const uint64_t fftsAddr = a.fftsAddr;
    const bool paged_KV = a.paged_KV;
    const bool is_causal = a.is_causal;
    const bool is_local = a.is_local;
    const bool flashDecodeFlag = a.flashDecodeFlag;
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

    if (paged_KV) {
        if (is_local) {
            if constexpr (IS_TND) {
                FWD_LAUNCH(DType, true, MASK_SWA, TND);
            } else {
                FWD_LAUNCH(DType, true, MASK_SWA, BSND);
            }
        } else if (is_causal) {
            if constexpr (IS_TND) {
                FWD_LAUNCH(DType, true, MASK_CAUSAL, TND);
            } else {
                FWD_LAUNCH(DType, true, MASK_CAUSAL, BSND);
            }
        } else {
            if constexpr (IS_TND) {
                FWD_LAUNCH(DType, true, NO_MASK, TND);
            } else {
                FWD_LAUNCH(DType, true, NO_MASK, BSND);
            }
        }
    } else {
        if (is_local) {
            if constexpr (IS_TND) {
                FWD_LAUNCH(DType, false, MASK_SWA, TND);
            } else {
                FWD_LAUNCH(DType, false, MASK_SWA, BSND);
            }
        } else if (is_causal) {
            if constexpr (IS_TND) {
                FWD_LAUNCH(DType, false, MASK_CAUSAL, TND);
            } else {
                FWD_LAUNCH(DType, false, MASK_CAUSAL, BSND);
            }
        } else {
            if constexpr (IS_TND) {
                FWD_LAUNCH(DType, false, NO_MASK, TND);
            } else {
                FWD_LAUNCH(DType, false, NO_MASK, BSND);
            }
        }
    }
}

#undef FWD_LAUNCH
