/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef FWD_DISPATCH_HPP
#define FWD_DISPATCH_HPP

#include <cstdint>
#include "acl/acl.h"
#include "kernel_common.hpp" // Format / CacheMode / PageShape / MaskCategory / CacheLayout

struct FwdLaunchArgs {
    bool is_bf16;
    Format layout; // TND, BSND or BNSD
    MaskCategory mask_category;
    bool paged_kv;
    bool enable_dn;   // use the FAInferDn fast path
    bool lse_mode;    // return softmax LSE
    bool has_softcap; // softcap > 0: selects the softcap kernel instance
    uint32_t block_dim;
    aclrtStream stream;
    uint8_t* q_device;
    uint8_t* k_device;
    uint8_t* v_device;
    uint8_t* mask_device;        // may be nullptr when not masked
    uint8_t* block_table_device; // may be nullptr when !paged_kv
    uint8_t* o_device;
    uint8_t* lse_device;
    uint8_t* q_seq_device;
    uint8_t* kv_seq_device;
    uint8_t* seq_used_q_device;  // per-batch used Q lengths (seqused_q); nullptr when absent
    uint8_t* seq_used_kv_device; // per-batch used KV lengths (seqused_k); nullptr when absent
    uint8_t* workspace_device;
    uint8_t* tiling_device;
    float scale_value;
    float softcap_value;
    uint32_t num_blocks;
    uint32_t max_num_blocks_per_batch;
    int64_t kv_seqlen_allocated; // allocated KV sequence length (BNSD head plane)
    FwdStorageParams storage;
};

// Per-(dtype, layout) implementation, defined in fwd_dispatch_impl.hpp and
// explicitly instantiated per (dtype, Format) in
// autogen/fwd_dispatch_<dtype>_<layout>.cpp.
template <typename DType, Format kLayout>
void launch_fwd_impl(const FwdLaunchArgs& a);

// Runtime entry: kLayout is picked from a.layout at runtime, then the matching
// dtype's launcher is selected. launch_fwd_impl is explicitly instantiated
// per (dtype, Format) in the autogen TUs.
inline void launch_fwd(const FwdLaunchArgs& a)
{
    if (a.is_bf16) {
        if (a.layout == Format::TND) {
            launch_fwd_impl<bfloat16_t, Format::TND>(a);
        } else if (a.layout == Format::BNSD) {
            launch_fwd_impl<bfloat16_t, Format::BNSD>(a);
        } else {
            launch_fwd_impl<bfloat16_t, Format::BSND>(a);
        }
    } else {
        if (a.layout == Format::TND) {
            launch_fwd_impl<half, Format::TND>(a);
        } else if (a.layout == Format::BNSD) {
            launch_fwd_impl<half, Format::BNSD>(a);
        } else {
            launch_fwd_impl<half, Format::BSND>(a);
        }
    }
}

#endif // FWD_DISPATCH_HPP
