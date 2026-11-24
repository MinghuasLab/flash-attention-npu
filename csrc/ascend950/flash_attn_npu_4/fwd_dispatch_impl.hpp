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

#define FWD_KERNEL_LAUNCH(MASK_CAT, USE_DN, ADAPTIVE_MASK, ADAPTIVE_DN)                                               \
    FAInfer<DType, float, kLayout, kLayout, CacheModeV, PageShapeV, MASK_CAT, kCacheLayout, LseMode, HasSoftcap,       \
            USE_DN, ADAPTIVE_MASK, ADAPTIVE_DN><<<a.block_dim, nullptr, a.stream>>>(                                 \
        a.q_device, a.k_device, a.v_device, a.mask_device, a.block_table_device, a.o_device, a.lse_device,           \
        a.q_seq_device, a.kv_seq_device, a.seq_used_q_device, a.seq_used_kv_device, a.workspace_device,              \
        a.tiling_device, a.runtime_tiling_device, a.kv_seqlen_allocated, a.storage)

#define FWD_QUANT_LAUNCH(MASK_CAT, ADAPTIVE_MASK)                                                                    \
    QuantFA::FAInferDnQuant<DType, float, kLayout, kLayout, CacheModeV, PageShapeV, MASK_CAT,                         \
                           kCacheLayout, LseMode, HasSoftcap, ADAPTIVE_MASK><<<a.block_dim, nullptr, a.stream>>>(   \
        a.q_device, a.k_device, a.v_device, a.mask_device, a.block_table_device, a.o_device, a.lse_device,           \
        a.q_seq_device, a.kv_seq_device, a.seq_used_q_device, a.seq_used_kv_device, a.workspace_device,              \
        a.tiling_device, a.runtime_tiling_device, a.q_descale_device, a.k_descale_device, a.v_descale_device,        \
        a.kv_seqlen_allocated)

template <typename DType, Format kLayout>
void launch_fwd_impl(const FwdLaunchArgs& a)
{
    constexpr CacheLayout kCacheLayout = CacheLayout::nd;
    FWD_BOOL_SWITCH(a.lse_mode, LseMode, {
        FWD_BOOL_SWITCH(a.paged_kv, IsPaged, {
            constexpr auto CacheModeV = IsPaged ? CacheMode::pagedCache : CacheMode::normalCache;
            constexpr auto PageShapeV = IsPaged ? PageShape::BnBsND : PageShape::normalShape;
            FWD_BOOL_SWITCH(a.has_softcap, HasSoftcap, {
                if constexpr (std::is_same_v<DType, float8_e4m3_t>) {
                    if constexpr (kLayout != Format::BNSD) {
                        if (a.adaptive_mask) {
                            FWD_QUANT_LAUNCH(MaskCategory::NO_MASK, true);
                            return;
                        }
                    }
                    FWD_MASK_SWITCH(a.mask_category, MaskCat, { FWD_QUANT_LAUNCH(MaskCat, false); });
                } else {
                    if constexpr (kLayout != Format::BNSD) {
                        if (a.adaptive_mask) {
                            FWD_KERNEL_LAUNCH(MaskCategory::NO_MASK, false, true, false);
                            return;
                        }
                    }
                    if constexpr (kLayout == Format::TND && IsPaged) {
                        if (a.adaptive_dn) {
                            FWD_KERNEL_LAUNCH(MaskCategory::NO_MASK, false, false, true);
                            return;
                        }
                    }
                    FWD_MASK_SWITCH(a.mask_category, MaskCat, {
                        if constexpr (MaskCat == MaskCategory::NO_MASK) {
                            FWD_BOOL_SWITCH(a.enable_dn, EnableDn, {
                                FWD_KERNEL_LAUNCH(MaskCat, EnableDn, false, false);
                            });
                        } else {
                            FWD_KERNEL_LAUNCH(MaskCat, false, false, false);
                        }
                    });
                }
            });
        });
    });
}

#undef FWD_QUANT_LAUNCH
#undef FWD_KERNEL_LAUNCH
#undef FWD_MASK_SWITCH
#undef FWD_BOOL_SWITCH

#endif // FWD_DISPATCH_IMPL_HPP
