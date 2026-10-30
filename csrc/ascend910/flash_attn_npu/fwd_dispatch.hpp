/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once

#include <cstdint>
#include "acl/acl.h"

// The layout enum (KernelCommon::FaiKenel::inputLayout) must be visible here so
// FwdLaunchArgs can carry it and launch_fwd can dispatch on it. kernel_common.hpp
// needs kernel_operator.h for __aicore__ / GM_ADDR, both of which the host TUs
// already pull in, so this adds no new CATLASS/kernel-template cost.
#include "kernel_operator.h"
#include "kernel_common.hpp"

struct FwdLaunchArgs {
    uint32_t blockDim;
    aclrtStream aclStream;
    uint64_t fftsAddr;
    bool is_bf16;
    bool paged_KV;
    bool is_causal;
    bool is_local;        // sliding-window attention (MASK_SWA)
    bool flashDecodeFlag; // only meaningful for the BSND (kvcache) path
    bool has_softcap;
    bool return_softmax = false;
    bool has_dropout = false;
    KernelCommon::FaiKenel::inputLayout layout = KernelCommon::FaiKenel::inputLayout::BSND;
    int64_t kv_seqlen_allocated = 0; // Dense cache capacity, or BNSD fixed-attention KV head-plane width.
    uint8_t* alibiSlopesDevice;
    uint8_t* qDevice;
    uint8_t* kDevice;
    uint8_t* vDevice;
    uint8_t* kNewDevice;       // may be nullptr when append-KV is disabled
    uint8_t* vNewDevice;       // may be nullptr when append-KV is disabled
    uint8_t* maskDevice;       // may be nullptr when is_causal is false
    uint8_t* blockTableDevice; // may be nullptr when paged_KV is false
    uint8_t* oDevice;
    uint8_t* softmaxLseDevice;
    uint8_t* qSeqDevice;
    uint8_t* kvSeqDevice;
    uint8_t* workspaceDevice;
    uint8_t* tilingDevice;
};

// Per-(dtype, layout) implementation, defined in autogen/fwd_dispatch_<dtype>_<layout>.cpp.
// Each TU instantiates one launch_fwd_impl<DType, kLayout> (BSND: 6 FAInfer variants,
// TND: 4, BNSD: 6) via fwd_dispatch_impl.hpp.
template <typename DType, KernelCommon::FaiKenel::inputLayout kLayout>
void launch_fwd_impl(const FwdLaunchArgs& a);

// Runtime entry: pick the layout carried by a.layout, then the dtype, then dispatch
// to the matching (dtype, layout) TU.
inline void launch_fwd(const FwdLaunchArgs& a)
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
