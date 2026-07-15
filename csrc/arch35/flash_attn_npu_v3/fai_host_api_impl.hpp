/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef FAI_HOST_API_IMPL_HPP
#define FAI_HOST_API_IMPL_HPP

#include <type_traits>

#include "fai_host_api.hpp"

// fai_kernel.cpp provides the FAInfer / FAInferDn kernel templates. It is
// template-only (CATLASS_GLOBAL void function templates, no non-template global
// symbols), so each autogen TU includes it once and instantiates its own slice
// without producing duplicate symbols.
#include "fai_kernel.cpp"

namespace fai_host
{

// ─── KEY / LAUNCH_CASE macros ─────────────────────────────────────
// Bit layout matches BuildKernelKey() in fai_host_api.hpp:
//   bit 0: cacheLayout  (0=nd, 1=nz)
//   bit 1: dataType     (0=half, 1=bf16)
//   bit 2: maskType     (1 if causal mask)
//   bit 3: maskType     (1 if SWA mask)
//   bit 4: innerPrec    (0=fp32, 1=fp16)
//   bit 5: layout       (0=TND, 1=BSND)
//   bit 6: cacheMode    (0=normalCache, 1=pagedCache)
//   bit 7: pageShape    (0=BnBsND, 1=BnNBsD)
#define KEY(CL_, DT, MT, SW, IP, LO, CM_, PS_)                \
    (((CL_) << 0) | ((DT) << 1) | ((MT) << 2) | ((SW) << 3) | \
     ((IP) << 4) | ((LO) << 5) | ((CM_) << 6) | ((PS_) << 7))

#define LAUNCH_CASE_DN(CL_, DT, MT, SW, IP, LO, CM_, PS_,                              \
                       T, AccT, QF, KVF, CachingMode, PageShapeType,                   \
                       MaskCat, CacheLay)                                              \
    case KEY(CL_, DT, MT, SW, IP, LO, CM_, PS_):                                       \
        if (enableDN)                                                                  \
        {                                                                              \
            FAInferDn<T, AccT, QF, KVF, CachingMode, PageShapeType, MaskCat, CacheLay> \
                <<<blockDim, nullptr, stream>>>(                                        \
                    qDevice, kDevice, vDevice, maskDevice, blockTableDevice,           \
                    oDevice, lseDevice, qSeqDevice, kvSeqDevice,                       \
                    workspaceDevice, tilingDevice);                                    \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            FAInfer<T, AccT, QF, KVF, CachingMode, PageShapeType, MaskCat, CacheLay>   \
                <<<blockDim, nullptr, stream>>>(                                        \
                    qDevice, kDevice, vDevice, maskDevice, blockTableDevice,           \
                    oDevice, lseDevice, qSeqDevice, kvSeqDevice,                       \
                    workspaceDevice, tilingDevice);                                    \
        }                                                                              \
        return ACL_SUCCESS;

#define LAUNCH_CASE(CL_, DT, MT, SW, IP, LO, CM_, PS_,                           \
                    T, AccT, QF, KVF, CachingMode, PageShapeType,                \
                    MaskCat, CacheLay)                                           \
    case KEY(CL_, DT, MT, SW, IP, LO, CM_, PS_):                                 \
        FAInfer<T, AccT, QF, KVF, CachingMode, PageShapeType, MaskCat, CacheLay> \
            <<<blockDim, nullptr, stream>>>(                                     \
                qDevice, kDevice, vDevice, maskDevice, blockTableDevice,         \
                oDevice, lseDevice, qSeqDevice, kvSeqDevice,                     \
                workspaceDevice, tilingDevice);                                  \
        return ACL_SUCCESS;

// Per-(dtype, layout) forward dispatch. launch_fai_dispatch is a primary
// template (NOT explicitly specialized): each autogen TU explicitly instantiates
// one <DType, IS_TND> pair, which the C++ rule [temp.explicit] turns into a
// strong symbol (an explicit instantiation of an already-specialized function
// would instead be ignored, so specialization is deliberately avoided here).
//
// The four (dtype, layout) case sets differ, so the single body selects among
// them at compile time with if constexpr: only the branch matching the
// instantiated <DType, IS_TND> is kept, and the discarded branches are not
// instantiated — so each TU still compiles only its own FAInfer / FAInferDn
// variants, keeping the kernel-template work parallelized across the four TUs.
// The top-level LaunchFAI() router (fai_host_api.cpp) includes only
// fai_host_api.hpp (the primary-template declaration) and emits external
// references that resolve to these instantiations at link time.
template <typename DType, bool IS_TND>
aclError launch_fai_dispatch(uint32_t kernelKey, bool enableDN,
                         uint32_t blockDim, aclrtStream stream,
                         uint8_t *qDevice, uint8_t *kDevice, uint8_t *vDevice,
                         uint8_t *maskDevice, uint8_t *blockTableDevice,
                         uint8_t *oDevice, uint8_t *lseDevice,
                         uint8_t *qSeqDevice, uint8_t *kvSeqDevice,
                         uint8_t *workspaceDevice, uint8_t *tilingDevice) {
    if constexpr (std::is_same_v<DType, half> && !IS_TND)
    {
        // half x BSND (dataType bit=0, layout bit=1)
        switch (kernelKey)
        {
            LAUNCH_CASE_DN(0, 0, 0, 0, 0, 1, 0, 0, half, float, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 0, 1, 0, 0, half, float, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 0, 1, 0, 0, half, float, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 0, 0, 0, 0, 1, 1, 0, half, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 0, 1, 1, 0, half, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 0, 1, 1, 0, half, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 0, 0, 0, 0, 1, 1, 1, half, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 0, 1, 1, 1, half, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 0, 1, 1, 1, half, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 0, 1, 1, 0, 0, half, half, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 1, 1, 0, 0, half, half, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 1, 1, 0, 0, half, half, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 0, 1, 1, 1, 0, half, half, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 1, 1, 1, 0, half, half, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 1, 1, 1, 0, half, half, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 0, 1, 1, 1, 1, half, half, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 1, 1, 1, 1, half, half, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 1, 1, 1, 1, half, half, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)

            default:
                break;
        }
    }
    else if constexpr (std::is_same_v<DType, half> && IS_TND)
    {
        // half x TND (dataType bit=0, layout bit=0)
        switch (kernelKey)
        {
            LAUNCH_CASE_DN(0, 0, 0, 0, 0, 0, 0, 0, half, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 0, 0, 0, 0, half, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 0, 0, 0, 0, half, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 0, 0, 0, 0, 0, 1, 0, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 0, 0, 1, 0, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 0, 0, 1, 0, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 0, 0, 0, 0, 0, 1, 1, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 0, 0, 1, 1, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 0, 0, 1, 1, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 0, 1, 0, 0, 0, half, half, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 1, 0, 0, 0, half, half, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 1, 0, 0, 0, half, half, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 0, 1, 0, 1, 0, half, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 1, 0, 1, 0, half, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 1, 0, 1, 0, half, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 0, 1, 0, 1, 1, half, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 1, 0, 1, 1, half, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 1, 0, 1, 1, half, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)

            default:
                break;
        }
    }
    else if constexpr (std::is_same_v<DType, bfloat16_t> && !IS_TND)
    {
        // bfloat16_t x BSND (dataType bit=1, layout bit=1); bf16 forces fp32 acc
        switch (kernelKey)
        {
            LAUNCH_CASE_DN(0, 1, 0, 0, 0, 1, 0, 0, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 1, 0, 0, 1, 0, 0, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 0, 1, 0, 1, 0, 0, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 1, 0, 0, 0, 1, 1, 0, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 1, 0, 0, 1, 1, 0, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 0, 1, 0, 1, 1, 0, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 1, 0, 0, 0, 1, 1, 1, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 1, 0, 0, 1, 1, 1, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 0, 1, 0, 1, 1, 1, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)

            default:
                break;
        }
    }
    else
    {
        // bfloat16_t x TND (dataType bit=1, layout bit=0); bf16 forces fp32 acc
        switch (kernelKey)
        {
            LAUNCH_CASE_DN(0, 1, 0, 0, 0, 0, 0, 0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 1, 0, 0, 0, 0, 0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 0, 1, 0, 0, 0, 0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 1, 0, 0, 0, 0, 1, 0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 1, 0, 0, 0, 1, 0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 0, 1, 0, 0, 1, 0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 1, 0, 0, 0, 0, 1, 1, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 1, 0, 0, 0, 1, 1, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 0, 1, 0, 0, 1, 1, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)

            default:
                break;
        }
    }
    return ACL_ERROR_INVALID_PARAM;
}

#undef KEY
#undef LAUNCH_CASE
#undef LAUNCH_CASE_DN

}  // namespace fai_host

#endif  // FAI_HOST_API_IMPL_HPP
