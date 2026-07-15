/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "fai_host_api.hpp"

namespace fai_host
{
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
        // kernelKey bit layout (see BuildKernelKey):
        //   bit 1: dataType (0=half, 1=bf16)
        //   bit 5: layout   (0=TND, 1=BSND)
        // Route to the (dtype, layout) TU that owns the matching switch slice.
        // The slice returns ACL_SUCCESS on a registered case, or
        // ACL_ERROR_INVALID_PARAM if kernelKey has no matching case in it.
        const bool is_bf16 = (kernelKey >> 1) & 1u;
        const bool is_bsnd = (kernelKey >> 5) & 1u;
        if (is_bf16) {
            if (is_bsnd) {
                return launch_fai_dispatch<bfloat16_t, false>(kernelKey, enableDN,
                    blockDim, stream, qDevice, kDevice, vDevice, maskDevice,
                    blockTableDevice, oDevice, lseDevice, qSeqDevice, kvSeqDevice,
                    workspaceDevice, tilingDevice);
            } else {
                return launch_fai_dispatch<bfloat16_t, true>(kernelKey, enableDN,
                    blockDim, stream, qDevice, kDevice, vDevice, maskDevice,
                    blockTableDevice, oDevice, lseDevice, qSeqDevice, kvSeqDevice,
                    workspaceDevice, tilingDevice);
            }
        } else {
            if (is_bsnd) {
                return launch_fai_dispatch<half, false>(kernelKey, enableDN,
                    blockDim, stream, qDevice, kDevice, vDevice, maskDevice,
                    blockTableDevice, oDevice, lseDevice, qSeqDevice, kvSeqDevice,
                    workspaceDevice, tilingDevice);
            } else {
                return launch_fai_dispatch<half, true>(kernelKey, enableDN,
                    blockDim, stream, qDevice, kDevice, vDevice, maskDevice,
                    blockTableDevice, oDevice, lseDevice, qSeqDevice, kvSeqDevice,
                    workspaceDevice, tilingDevice);
            }
        }
    }
} // namespace fai_host
