/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Standalone translation unit for the D>256 128x256 partial-resident bwd
 * kernel. Compiles independently of fag_kernel.cpp so the legacy D<=256 
 * paths are not recompiled when this file changes.
 */

#include "fag_bwd_d512_partial.hpp"
#include "fag_kernel_d512_partial.hpp"

namespace {

template <typename DType, FAGTiling950::Layout LAYOUT, bool IS_CAUSAL, bool IS_SOFTCAP>
inline void LaunchPartial(const FagD512PartialLaunchArgs& a)
{
    FlashAttentionV3Bwd950Partial<DType, LAYOUT, IS_CAUSAL, /*IS_DETERMINISTIC=*/false, IS_SOFTCAP>
        <<<a.launchBlockDim, nullptr, a.aclStream>>>(
            a.dout, a.q, a.k, a.v, a.out, a.mask, a.softmaxLse, a.cuSeqQ, a.cuSeqKv, a.seqUsedQ, a.seqUsedKv, a.dq,
            a.dk, a.dv, a.workspace, a.tiling);
}

template <typename DType, FAGTiling950::Layout LAYOUT>
inline void DispatchCausalSoftcap(const FagD512PartialLaunchArgs& a)
{
    if (a.isCausal) {
        if (a.isSoftcap) {
            LaunchPartial<DType, LAYOUT, true, true>(a);
        } else {
            LaunchPartial<DType, LAYOUT, true, false>(a);
        }
    } else if (a.isSoftcap) {
        LaunchPartial<DType, LAYOUT, false, true>(a);
    } else {
        LaunchPartial<DType, LAYOUT, false, false>(a);
    }
}

template <typename DType>
inline void DispatchLayout(const FagD512PartialLaunchArgs& a)
{
    switch (static_cast<FAGTiling950::Layout>(a.layoutCode)) {
        case FAGTiling950::Layout::TND:
            DispatchCausalSoftcap<DType, FAGTiling950::Layout::TND>(a);
            break;
        case FAGTiling950::Layout::BNSD:
            DispatchCausalSoftcap<DType, FAGTiling950::Layout::BNSD>(a);
            break;
        case FAGTiling950::Layout::BSND:
        default:
            DispatchCausalSoftcap<DType, FAGTiling950::Layout::BSND>(a);
            break;
    }
}

} // namespace

void launch_fag_d512_partial(const FagD512PartialLaunchArgs& args)
{
    if (args.dtypeCode == 1U) {
        DispatchLayout<bfloat16_t>(args);
    } else {
        DispatchLayout<half>(args);
    }
}
