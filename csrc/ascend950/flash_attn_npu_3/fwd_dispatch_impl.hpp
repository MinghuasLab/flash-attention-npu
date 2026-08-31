/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef FWD_DISPATCH_IMPL_HPP
#define FWD_DISPATCH_IMPL_HPP

#include <torch/extension.h>

#include "fwd_dispatch.hpp"
#include "fwd_kernel.cpp"

// ─── SWITCH / LAUNCH macros ───────────────────────────────────────
// Three flat macros; none of them expands another. They are composed only at
// the use site in launch_fwd_impl(), mirroring the BOOL_SWITCH /
// MASK_SWITCH / LAUNCH style used elsewhere in this repo.

// Convert a runtime bool into a named compile-time flag.
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

// Convert the runtime mask_category into a named compile-time constant.
// The three MaskCategory values are enumerated explicitly.
#define FWD_MASK_SWITCH(MASK_CATEGORY, MASK_CAT, ...)                          \
    do {                                                                       \
        if (MASK_CATEGORY == MaskCategory::NO_MASK) {                          \
            constexpr auto MASK_CAT = MaskCategory::NO_MASK;                   \
            __VA_ARGS__                                                        \
        } else if (MASK_CATEGORY == MaskCategory::MASK_CAUSAL) {               \
            constexpr auto MASK_CAT = MaskCategory::MASK_CAUSAL;               \
            __VA_ARGS__                                                        \
        } else {                                                               \
            constexpr auto MASK_CAT = MaskCategory::MASK_SWA;                  \
            __VA_ARGS__                                                        \
        }                                                                      \
    } while (0)

#define FWD_KERNEL_LAUNCH(KERNEL, MASK_CAT, CACHE_MODE_V, PAGE_SHAPE_V, LSE_MODE)  \
    KERNEL<DType, float, kFormat, kFormat, CACHE_MODE_V, PAGE_SHAPE_V,             \
           MASK_CAT, kCacheLayout, LSE_MODE>                                       \
        <<<a.block_dim, nullptr, a.stream>>>(                                      \
            a.q_device, a.k_device, a.v_device, a.mask_device,                     \
            a.block_table_device, a.o_device, a.lse_device, a.q_seq_device,        \
            a.kv_seq_device, a.workspace_device, a.tiling_device)

template <typename DType, bool IS_TND>
void launch_fwd_impl(const FwdLaunchArgs &a) {
    constexpr Format kFormat = IS_TND ? Format::TND : Format::BSND;
    // The 950 host path only ever feeds ND-layout KV caches; the kernel
    // observes this as a compile-time template parameter, so no runtime
    // field is needed in the tiling data.
    constexpr CacheLayout kCacheLayout = CacheLayout::nd;

    FWD_MASK_SWITCH(a.mask_category, MaskCat, {
        FWD_BOOL_SWITCH(a.lse_mode, LseMode, {
            FWD_BOOL_SWITCH(a.enable_dn, IsDN, {
                FWD_BOOL_SWITCH(a.paged_kv, IsPaged, {
                    constexpr auto CacheModeV =
                        IsPaged ? CacheMode::pagedCache : CacheMode::normalCache;
                    constexpr auto PageShapeV =
                        IsPaged ? PageShape::BnBsND : PageShape::normalShape;
                    // Only NO_MASK has an FAInferDn fast path: masked (causal /
                    // SWA) workloads always use FAInfer. Because this condition
                    // is a compile-time constant, if constexpr also prunes the
                    // masked FAInferDn instantiations from this TU.
                    constexpr bool kUseDnFastPath =
                        (MaskCat == MaskCategory::NO_MASK) && IsDN;
                    if constexpr (kUseDnFastPath) {
                        FWD_KERNEL_LAUNCH(FAInferDn, MaskCat,
                                          CacheModeV, PageShapeV, LseMode);
                    } else {
                        FWD_KERNEL_LAUNCH(FAInfer, MaskCat,
                                          CacheModeV, PageShapeV, LseMode);
                    }
                });
            });
        });
    });
}

#undef FWD_KERNEL_LAUNCH
#undef FWD_MASK_SWITCH
#undef FWD_BOOL_SWITCH

#endif  // FWD_DISPATCH_IMPL_HPP
