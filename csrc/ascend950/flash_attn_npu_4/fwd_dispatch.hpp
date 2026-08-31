/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef FWD_DISPATCH_HPP
#define FWD_DISPATCH_HPP

#include <cstdint>
#include "acl/acl.h"
#include "kernel_common.hpp"  // Format / CacheMode / PageShape / MaskCategory / CacheLayout

struct FwdLaunchArgs {
    bool is_bf16;
    Format layout;            // BSND or TND
    MaskCategory mask_category;
    bool paged_kv;
    bool enable_dn;           // use the FAInferDn fast path
    bool lse_mode;            // return softmax LSE
    uint32_t block_dim;
    aclrtStream stream;
    uint8_t *q_device;
    uint8_t *k_device;
    uint8_t *v_device;
    uint8_t *mask_device;          // may be nullptr when not masked
    uint8_t *block_table_device;   // may be nullptr when !paged_kv
    uint8_t *o_device;
    uint8_t *lse_device;
    uint8_t *q_seq_device;
    uint8_t *kv_seq_device;
    uint8_t *workspace_device;
    uint8_t *tiling_device;
};

// Per-(dtype, layout) implementation, defined in fwd_dispatch_impl.hpp and
// explicitly instantiated per (dtype, IS_TND) in
// autogen/fwd_dispatch_<dtype>_<layout>.cpp. IS_TND is true for TND (varlen)
// layout, false for BSND.
template <typename DType, bool IS_TND>
void launch_fwd_impl(const FwdLaunchArgs &a);

// Runtime entry: IS_TND is picked from a.layout at runtime, then the matching
// dtype's launcher is selected. launch_fwd_impl is explicitly instantiated
// per (dtype, IS_TND) in the autogen TUs.
inline void launch_fwd(const FwdLaunchArgs &a) {
    const bool is_bsnd = (a.layout == Format::BSND);
    if (a.is_bf16) {
        if (is_bsnd) {
            launch_fwd_impl<bfloat16_t, false>(a);
        } else {
            launch_fwd_impl<bfloat16_t, true>(a);
        }
    } else {
        if (is_bsnd) {
            launch_fwd_impl<half, false>(a);
        } else {
            launch_fwd_impl<half, true>(a);
        }
    }
}

#endif  // FWD_DISPATCH_HPP
