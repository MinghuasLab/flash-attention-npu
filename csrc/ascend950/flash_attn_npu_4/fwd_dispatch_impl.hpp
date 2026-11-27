// Copyright (c) 2026, Minghua Shen.

#ifndef FWD_DISPATCH_IMPL_HPP
#define FWD_DISPATCH_IMPL_HPP

// No torch includes here: this header is compiled into the ASC dispatch TUs,
// and torch >= 2.14 headers require C++20, which the bisheng ASC frontend
// cannot parse. Everything below works on plain pointers from
// fwd_dispatch.hpp's launch-args structs.

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

#define FWD_KERNEL_LAUNCH(KERNEL, MASK_CAT, CACHE_MODE_V, PAGE_SHAPE_V, LSE_MODE, HAS_SOFTCAP)                         \
    KERNEL<DType, float, kLayout, kLayout, CACHE_MODE_V, PAGE_SHAPE_V, MASK_CAT, kCacheLayout, LSE_MODE, HAS_SOFTCAP>  \
        <<<a.block_dim, nullptr, a.stream>>>(                                                                          \
            a.q_device, a.k_device, a.v_device, a.mask_device, a.block_table_device, a.o_device, a.lse_device,         \
            a.q_seq_device, a.kv_seq_device, a.seq_used_q_device, a.seq_used_kv_device, a.workspace_device,            \
            a.tiling_device, a.runtime_tiling_device, a.kv_seqlen_allocated, a.storage)

template <typename DType, Format kLayout>
void launch_fwd_impl(const FwdLaunchArgs& a)
{
    // The 950 host path only ever feeds ND-layout KV caches; the kernel
    // observes this as a compile-time template parameter, so no runtime
    // field is needed in the tiling data.
    constexpr CacheLayout kCacheLayout = CacheLayout::nd;

    if constexpr (std::is_same_v<DType, float8_e4m3_t>) {
        FWD_MASK_SWITCH(a.mask_category, MaskCat, {
            FWD_BOOL_SWITCH(a.lse_mode, LseMode, {
                FWD_BOOL_SWITCH(a.paged_kv, IsPaged, {
                    constexpr auto CacheModeV = IsPaged ? CacheMode::pagedCache : CacheMode::normalCache;
                    constexpr auto PageShapeV = IsPaged ? PageShape::BnBsND : PageShape::normalShape;
                    FWD_BOOL_SWITCH(a.has_softcap, HasSoftcap, {
                        QuantFA::FAInferDnQuant<DType, float, kLayout, kLayout, CacheModeV, PageShapeV, MaskCat,
                                                kCacheLayout, LseMode, HasSoftcap><<<a.block_dim, nullptr, a.stream>>>(
                            a.q_device, a.k_device, a.v_device, a.mask_device, a.block_table_device, a.o_device,
                            a.lse_device, a.q_seq_device, a.kv_seq_device, a.seq_used_q_device, a.seq_used_kv_device,
                            a.workspace_device, a.tiling_device, a.fp8_runtime, a.q_descale_device, a.k_descale_device,
                            a.v_descale_device, a.kv_seqlen_allocated);
                    });
                });
            });
        });
    } else {
        FWD_MASK_SWITCH(a.mask_category, MaskCat, {
            FWD_BOOL_SWITCH(a.lse_mode, LseMode, {
                FWD_BOOL_SWITCH(a.enable_dn, IsDN, {
                    FWD_BOOL_SWITCH(a.paged_kv, IsPaged, {
                        constexpr auto CacheModeV = IsPaged ? CacheMode::pagedCache : CacheMode::normalCache;
                        constexpr auto PageShapeV = IsPaged ? PageShape::BnBsND : PageShape::normalShape;
                        // Only NO_MASK has an FAInferDn fast path: masked (causal /
                        // SWA) workloads always use FAInfer. Because this condition
                        // is a compile-time constant, if constexpr also prunes the
                        // masked FAInferDn instantiations from this TU.
                        constexpr bool kUseDnFastPath = (MaskCat == MaskCategory::NO_MASK) && IsDN;
                        FWD_BOOL_SWITCH(a.has_softcap, HasSoftcap, {
                            if constexpr (kUseDnFastPath) {
                                FWD_KERNEL_LAUNCH(FAInferDn, MaskCat, CacheModeV, PageShapeV, LseMode, HasSoftcap);
                            } else {
                                FWD_KERNEL_LAUNCH(FAInfer, MaskCat, CacheModeV, PageShapeV, LseMode, HasSoftcap);
                            }
                        });
                    });
                });
            });
        });
    }
}

#undef FWD_KERNEL_LAUNCH
#undef FWD_MASK_SWITCH
#undef FWD_BOOL_SWITCH

// Dtype/quant-agnostic host entry (declared in fwd_dispatch.hpp): the
// quant_mode x is_bf16 x layout selection happens here, device-side, so host
// TUs never name the ASC dtype types (float8_e4m3_t / bfloat16_t / half).
// gnu::used forces emission (no device TU calls it directly).
[[gnu::used]] inline void launch_fwd_dispatch(const FwdLaunchArgs& a)
{
    if (a.quant_mode == 1) {
        if (a.layout == Format::TND) {
            launch_fwd_impl<float8_e4m3_t, Format::TND>(a);
        } else if (a.layout == Format::BNSD) {
            launch_fwd_impl<float8_e4m3_t, Format::BNSD>(a);
        } else {
            launch_fwd_impl<float8_e4m3_t, Format::BSND>(a);
        }
    } else if (a.is_bf16) {
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

#endif // FWD_DISPATCH_IMPL_HPP
