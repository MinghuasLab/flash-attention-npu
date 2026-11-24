// Copyright (c) 2026, Minghua Shen.

#ifndef FWD_DISPATCH_IMPL_HPP
#define FWD_DISPATCH_IMPL_HPP

#include <torch/extension.h>

#include "fwd_dispatch.hpp"
#include "fwd_kernel.cpp"
#include "fwd_kernel_quant.cpp"

// ─── SWITCH / LAUNCH macros ───────────────────────────────────────
// Three flat macros; none of them expands another. They are composed only at
// the use site in launch_fwd_impl(), mirroring the BOOL_SWITCH /
// MASK_SWITCH / LAUNCH style used elsewhere in this repo.

// Convert a runtime bool into a named compile-time flag.
#define FWD_BOOL_SWITCH(COND, CONST_NAME, ...)                                                                         \
    do {                                                                                                               \
        if (COND) {                                                                                                    \
            constexpr bool CONST_NAME = true;                                                                          \
            __VA_ARGS__                                                                                                \
        } else {                                                                                                       \
            constexpr bool CONST_NAME = false;                                                                         \
            __VA_ARGS__                                                                                                \
        }                                                                                                              \
    } while (0)

// Convert the runtime mask_category into a named compile-time constant.
// The three MaskCategory values are enumerated explicitly.
#define FWD_MASK_SWITCH(MASK_CATEGORY, MASK_CAT, ...)                                                                  \
    do {                                                                                                               \
        if (MASK_CATEGORY == MaskCategory::NO_MASK) {                                                                  \
            constexpr auto MASK_CAT = MaskCategory::NO_MASK;                                                           \
            __VA_ARGS__                                                                                                \
        } else if (MASK_CATEGORY == MaskCategory::MASK_CAUSAL) {                                                       \
            constexpr auto MASK_CAT = MaskCategory::MASK_CAUSAL;                                                       \
            __VA_ARGS__                                                                                                \
        } else {                                                                                                       \
            constexpr auto MASK_CAT = MaskCategory::MASK_SWA;                                                          \
            __VA_ARGS__                                                                                                \
        }                                                                                                              \
    } while (0)

#define FWD_KERNEL_LAUNCH(MASK_CAT, CACHE_MODE_V, PAGE_SHAPE_V, LSE_MODE, HAS_SOFTCAP, USE_DN, DYNAMIC)          \
    FAInfer<DType, float, kLayout, kLayout, CACHE_MODE_V, PAGE_SHAPE_V, MASK_CAT, kCacheLayout, LSE_MODE,         \
            HAS_SOFTCAP, USE_DN, DYNAMIC>                                                                     \
        <<<a.block_dim, nullptr, a.stream>>>(a.q_device, a.k_device, a.v_device, a.mask_device,                  \
                                             a.block_table_device, a.o_device, a.lse_device, a.q_seq_device,    \
                                             a.kv_seq_device, a.seq_used_q_device, a.seq_used_kv_device,         \
                                             a.workspace_device, a.tiling_device, a.scale_value, a.softcap_value, \
                                             a.num_blocks, a.max_num_blocks_per_batch, a.kv_seqlen_allocated,   \
                                             a.storage, a.enable_dn)

#define FWD_QUANT_LAUNCH(MASK_CAT, CACHE_MODE_V, PAGE_SHAPE_V, LSE_MODE, HAS_SOFTCAP, DYNAMIC)                   \
    QuantFA::FAInferDnQuant<DType, float, kLayout, kLayout, CACHE_MODE_V, PAGE_SHAPE_V, MASK_CAT, kCacheLayout, \
                            LSE_MODE, HAS_SOFTCAP, DYNAMIC><<<a.block_dim, nullptr, a.stream>>>(                \
        a.q_device, a.k_device, a.v_device, a.mask_device, a.block_table_device, a.o_device, a.lse_device,      \
        a.q_seq_device, a.kv_seq_device, a.seq_used_q_device, a.seq_used_kv_device, a.workspace_device,          \
        a.tiling_device, a.scale_value, a.softcap_value, a.num_blocks, a.max_num_blocks_per_batch,               \
        a.q_descale_device, a.k_descale_device, a.v_descale_device, a.kv_seqlen_allocated, a.storage)

template <typename DType, Format kLayout>
void launch_fwd_impl(const FwdLaunchArgs& a)
{
    // The 950 host path only ever feeds ND-layout KV caches; the kernel
    // observes this as a compile-time template parameter, so no runtime
    // field is needed in the tiling data.
    constexpr CacheLayout kCacheLayout = CacheLayout::nd;

    FWD_BOOL_SWITCH(a.lse_mode, LseMode, {
        FWD_BOOL_SWITCH(a.paged_kv, IsPaged, {
            constexpr auto CacheModeV = IsPaged ? CacheMode::pagedCache : CacheMode::normalCache;
            constexpr auto PageShapeV = IsPaged ? PageShape::BnBsND : PageShape::normalShape;
            FWD_BOOL_SWITCH(a.has_softcap, HasSoftcap, {
                if (a.dynamic_dispatch) {
                    constexpr auto Mask = MaskCategory::NO_MASK;
                    if constexpr (std::is_same_v<DType, float8_e4m3_t>) {
                        FWD_QUANT_LAUNCH(Mask, CacheModeV, PageShapeV, LseMode, HasSoftcap, true);
                    } else {
                        FWD_KERNEL_LAUNCH(Mask, CacheModeV, PageShapeV, LseMode, HasSoftcap, false, true);
                    }
                } else {
                    FWD_MASK_SWITCH(a.mask_category, MaskCat, {
                        if constexpr (std::is_same_v<DType, float8_e4m3_t>) {
                            FWD_QUANT_LAUNCH(MaskCat, CacheModeV, PageShapeV, LseMode, HasSoftcap, false);
                        } else if constexpr (MaskCat == MaskCategory::NO_MASK) {
                            FWD_BOOL_SWITCH(a.enable_dn, IsDN, {
                                FWD_KERNEL_LAUNCH(MaskCat, CacheModeV, PageShapeV, LseMode, HasSoftcap, IsDN, false);
                            });
                        } else {
                            FWD_KERNEL_LAUNCH(MaskCat, CacheModeV, PageShapeV, LseMode, HasSoftcap, false, false);
                        }
                    });
                }
            });
        });
    });
}

#undef FWD_KERNEL_LAUNCH
#undef FWD_QUANT_LAUNCH
#undef FWD_MASK_SWITCH
#undef FWD_BOOL_SWITCH

#endif // FWD_DISPATCH_IMPL_HPP
