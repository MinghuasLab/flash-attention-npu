/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once

#include <cstdint>
#include "acl/acl.h"

struct FwdLaunchArgs {
    uint32_t blockDim;
    aclrtStream aclStream;
    uint64_t fftsAddr;
    bool is_bf16;
    bool paged_KV;
    bool is_causal;
    bool is_local;              // sliding-window attention (MASK_SWA)
    bool flashDecodeFlag;       // only meaningful for the BSND (kvcache) path
    bool has_softcap;
    bool return_softmax = false;
    bool has_dropout = false;
    uint8_t *alibiSlopesDevice;  
    uint8_t *qDevice;
    uint8_t *kDevice;
    uint8_t *vDevice;
    uint8_t *kNewDevice;        // may be nullptr when append-KV is disabled
    uint8_t *vNewDevice;        // may be nullptr when append-KV is disabled
    uint8_t *maskDevice;        // may be nullptr when is_causal is false
    uint8_t *blockTableDevice;  // may be nullptr when paged_KV is false
    uint8_t *oDevice;
    uint8_t *softmaxLseDevice;
    uint8_t *qSeqDevice;
    uint8_t *kvSeqDevice;
    uint8_t *workspaceDevice;
    uint8_t *tilingDevice;
};

// Per-(dtype, layout) implementation, defined in autogen/fwd_dispatch_<dtype>_<layout>.cpp.
// Each TU instantiates one launch_fwd_impl<DType, IS_TND> (BSND: 6 FAInfer variants,
// TND: 4) via fwd_dispatch_impl.hpp.
template <typename DType, bool IS_TND>
void launch_fwd_impl(const FwdLaunchArgs &a);

// Runtime entry: pick dtype, dispatch to the matching dtype TU. IS_TND is chosen
// at the call site (kvcache/mha_fwd => false, varlen_fwd => true).
template <bool IS_TND>
inline void launch_fwd(const FwdLaunchArgs &a) {
    if (a.is_bf16) {
        launch_fwd_impl<bfloat16_t, IS_TND>(a);
    } else {
        launch_fwd_impl<half, IS_TND>(a);
    }
}
