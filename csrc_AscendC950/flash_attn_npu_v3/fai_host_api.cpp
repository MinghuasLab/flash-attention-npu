/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "fai_host_api.hpp"

#include "fai_kernel.cpp"

namespace fai_host
{

// ─── KEY / LAUNCH_CASE macros ─────────────────────────────────────
//
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
                <<<blockDim, nullptr, stream>>>(                                       \
                    qDevice, kDevice, vDevice, maskDevice, blockTableDevice,           \
                    oDevice, lseDevice, qSeqDevice, kvSeqDevice,                       \
                    workspaceDevice, tilingDevice);                                    \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            FAInfer<T, AccT, QF, KVF, CachingMode, PageShapeType, MaskCat, CacheLay>   \
                <<<blockDim, nullptr, stream>>>(                                       \
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

    // ─── Public API ────────────────────────────────────────────────────

    uint32_t BuildKernelKey(const std::string &dataType,
                            const std::string &cacheLayout,
                            uint32_t maskType,
                            uint32_t innerPrec,
                            Format layout,
                            CacheMode cacheMode,
                            PageShape pageShape)
    {
        uint32_t key = 0;
        key |= (cacheLayout == "nd" ? 0u : 1u) << 0;
        key |= (dataType == "half" ? 0u : 1u) << 1;
        key |= (maskType == 1u ? 1u : 0u) << 2;
        key |= (maskType == 4u ? 1u : 0u) << 3;
        const uint32_t ipBit = (dataType == "half") ? ((innerPrec == 1u) ? 1u : 0u) : 0u;
        key |= ipBit << 4;
        key |= (layout == Format::TND ? 0u : 1u) << 5;
        key |= (cacheMode == CacheMode::pagedCache ? 1u : 0u) << 6;
        const uint32_t psBit = (cacheMode == CacheMode::pagedCache)
                                   ? ((pageShape == PageShape::BnBsND) ? 0u : 1u)
                                   : 0u;
        key |= psBit << 7;
        return key;
    }

    aclError LaunchFAI(uint32_t kernelKey, bool enableDN,
                       uint32_t blockDim, aclrtStream stream,
                       uint8_t *qDevice, uint8_t *kDevice, uint8_t *vDevice,
                       uint8_t *maskDevice, uint8_t *blockTableDevice,
                       uint8_t *oDevice, uint8_t *lseDevice,
                       uint8_t *qSeqDevice, uint8_t *kvSeqDevice,
                       uint8_t *workspaceDevice, uint8_t *tilingDevice)
    {
        switch (kernelKey)
        {
            // ============ CacheLayout::nd (CL=0) ============
            // half (DT=0) IP=0
            LAUNCH_CASE_DN(0, 0, 0, 0, 0, 0, 0, 0, half, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 0, 0, 0, 0, half, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 0, 0, 0, 0, half, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 0, 0, 0, 0, 1, 0, 0, half, float, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 0, 1, 0, 0, half, float, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 0, 1, 0, 0, half, float, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)

            LAUNCH_CASE_DN(0, 0, 0, 0, 0, 0, 1, 0, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 0, 0, 1, 0, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 0, 0, 1, 0, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 0, 0, 0, 0, 0, 1, 1, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 0, 0, 1, 1, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 0, 0, 1, 1, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 0, 0, 0, 0, 1, 1, 0, half, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 0, 1, 1, 0, half, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 0, 1, 1, 0, half, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 0, 0, 0, 0, 1, 1, 1, half, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 0, 1, 1, 1, half, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 0, 1, 1, 1, half, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)

            // half IP=1
            LAUNCH_CASE(0, 0, 0, 0, 1, 0, 0, 0, half, half, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 1, 0, 0, 0, half, half, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 1, 0, 0, 0, half, half, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 0, 1, 1, 0, 0, half, half, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 1, 1, 0, 0, half, half, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 1, 1, 0, 0, half, half, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)

            LAUNCH_CASE(0, 0, 0, 0, 1, 0, 1, 0, half, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 1, 0, 1, 0, half, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 1, 0, 1, 0, half, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 0, 1, 0, 1, 1, half, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 1, 0, 1, 1, half, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 1, 0, 1, 1, half, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 0, 1, 1, 1, 0, half, half, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 1, 1, 1, 0, half, half, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 1, 1, 1, 0, half, half, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 0, 1, 1, 1, 1, half, half, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 1, 0, 1, 1, 1, 1, half, half, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 0, 0, 1, 1, 1, 1, 1, half, half, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)

            // bf16 (DT=1) IP=0
            LAUNCH_CASE_DN(0, 1, 0, 0, 0, 0, 0, 0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 1, 0, 0, 0, 0, 0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 0, 1, 0, 0, 0, 0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 1, 0, 0, 0, 1, 0, 0, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 1, 0, 0, 1, 0, 0, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 0, 1, 0, 1, 0, 0, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 1, 0, 0, 0, 0, 1, 0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 1, 0, 0, 0, 1, 0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 0, 1, 0, 0, 1, 0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 1, 0, 0, 0, 0, 1, 1, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 1, 0, 0, 0, 1, 1, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 0, 1, 0, 0, 1, 1, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 1, 0, 0, 0, 1, 1, 0, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 1, 0, 0, 1, 1, 0, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 0, 1, 0, 1, 1, 0, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
            LAUNCH_CASE_DN(0, 1, 0, 0, 0, 1, 1, 1, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 1, 0, 0, 1, 1, 1, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
            LAUNCH_CASE(0, 1, 0, 1, 0, 1, 1, 1, bfloat16_t, float, Format::BSND, Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)

        default:
            return ACL_ERROR_INVALID_PARAM;
        }
    }
} // namespace fai_host

#undef KEY
#undef LAUNCH_CASE
#undef LAUNCH_CASE_DN