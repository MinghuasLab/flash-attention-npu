/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

//
// Shared implementation of the v2 forward FAInfer dispatch. Included by the
// generated autogen/fwd_dispatch_<dtype>_<layout>.cpp stubs, each of which
// explicitly instantiates one launch_fwd_impl<DType, IS_TND>, so the FAInfer
// template instantiations land in separate (parallel-compiled) object files.
//
// The launch tree below reproduces the exact dtype x paged x mask x
// flash-decode x layout combinations of the three original host functions
// (mha_fwd_kvcache: BSND with FD; mha_fwd: BSND non-paged; mha_varlen_fwd: TND).
// mha_fwd and mha_fwd_kvcache share the BSND path; mha_varlen_fwd uses TND.

#pragma once

#include "fwd_dispatch.hpp"

// Standard headers that the CATLASS/FAG headers (reached via mha_fwd_kvcache.cpp)
// assume to be already visible.
#include <algorithm>
#include <cstring>
#include <limits>

// mha_fwd_kvcache.cpp provides the SplitFuse::FAInfer kernel template, the
// FAInferKernel class, FAIKernelParams, and the FaiKenel enum namespace.
#include "mha_fwd_kvcache.cpp"

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
            SplitFuse::FAInfer<DType, DType, float, true, FaiKenel::MaskType::MASK_SWA,
                LAYOUT, Catlass::Epilogue::LseModeT::OUT_ONLY>
                <<<blockDim, nullptr, aclStream>>>(
                    fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                    qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (is_causal) {
            // Flash-decode (8th template param = true) is a BSND-only path
            // (mha_fwd_kvcache); compiled out for TND so no FD+TND combo is
            // instantiated.
            if constexpr (!IS_TND) {
                if (flashDecodeFlag) {
                    SplitFuse::FAInfer<DType, DType, float, true, FaiKenel::MaskType::MASK_CAUSAL,
                        LAYOUT, Catlass::Epilogue::LseModeT::OUT_ONLY, true>
                        <<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    SplitFuse::FAInfer<DType, DType, float, true, FaiKenel::MaskType::MASK_CAUSAL,
                        LAYOUT, Catlass::Epilogue::LseModeT::OUT_ONLY>
                        <<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            } else {
                SplitFuse::FAInfer<DType, DType, float, true, FaiKenel::MaskType::MASK_CAUSAL,
                    LAYOUT, Catlass::Epilogue::LseModeT::OUT_ONLY>
                    <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            }
        } else {
            if constexpr (!IS_TND) {
                if (flashDecodeFlag) {
                    SplitFuse::FAInfer<DType, DType, float, true, FaiKenel::MaskType::NO_MASK,
                        LAYOUT, Catlass::Epilogue::LseModeT::OUT_ONLY, true>
                        <<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    SplitFuse::FAInfer<DType, DType, float, true, FaiKenel::MaskType::NO_MASK,
                        LAYOUT, Catlass::Epilogue::LseModeT::OUT_ONLY>
                        <<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            } else {
                SplitFuse::FAInfer<DType, DType, float, true, FaiKenel::MaskType::NO_MASK,
                    LAYOUT, Catlass::Epilogue::LseModeT::OUT_ONLY>
                    <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            }
        }
    } else {
        if (is_local) {
            SplitFuse::FAInfer<DType, DType, float, false, FaiKenel::MaskType::MASK_SWA,
                LAYOUT, Catlass::Epilogue::LseModeT::OUT_ONLY>
                <<<blockDim, nullptr, aclStream>>>(
                    fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                    qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (is_causal) {
            SplitFuse::FAInfer<DType, DType, float, false, FaiKenel::MaskType::MASK_CAUSAL,
                LAYOUT, Catlass::Epilogue::LseModeT::OUT_ONLY>
                <<<blockDim, nullptr, aclStream>>>(
                    fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                    qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else {
            SplitFuse::FAInfer<DType, DType, float, false, FaiKenel::MaskType::NO_MASK,
                LAYOUT, Catlass::Epilogue::LseModeT::OUT_ONLY>
                <<<blockDim, nullptr, aclStream>>>(
                    fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                    qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        }
    }
}
