/** Copyright (c) 2026 Huawei Technologies Co., Ltd. CANN Open Software License Agreement Version 2.0. */
#ifndef FA950_FWD_RUNTIME_PARAMS_H
#define FA950_FWD_RUNTIME_PARAMS_H
#include <type_traits>
#include "tilingdata.h"
#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
#define FA950_RUNTIME_INLINE __forceinline__[host, aicore]
#else
#define FA950_RUNTIME_INLINE inline
#endif

// Common launch values remain primitive scalars, as in FA3. Page sizes
// 128/256/512/1024 use at most 11 bits; bit31 carries FD permission.
constexpr uint32_t FWD_FD_PERMISSION = 1U << 31;
struct FwdWindowBounds { int64_t left, right; };
static_assert(sizeof(FwdWindowBounds) == 16);
template <bool HasWindow>
using FwdWindowLaunchParams = std::conditional_t<HasWindow, FwdWindowBounds, uint8_t>;

FA950_RUNTIME_INLINE uint32_t PackFwdBlockConfig(const FAInferRuntimeData& runtime)
{
    return runtime.blockSize | (runtime.allowFlashDecode ? FWD_FD_PERMISSION : 0U);
}

template <bool HasWindow>
FA950_RUNTIME_INLINE FwdWindowLaunchParams<HasWindow> PackFwdWindow(const FAInferRuntimeData& runtime)
{
    if constexpr (HasWindow) {
        return {runtime.windowSizeLeft, runtime.windowSizeRight};
    } else {
        return 0U;
    }
}

template <bool HasWindow>
FA950_RUNTIME_INLINE FAInferRuntimeData MakeFwdRuntime(
    float scale, float softcap, uint32_t numBlocks, uint32_t maxBlocks, uint32_t blockConfig,
    FwdWindowLaunchParams<HasWindow> window, int64_t defaultRight)
{
    FAInferRuntimeData runtime{};
    runtime.numBlocks = numBlocks;
    runtime.blockSize = blockConfig & ~FWD_FD_PERMISSION;
    runtime.maxNumBlocksPerBatch = maxBlocks;
    runtime.scaleValue = scale;
    runtime.softcapValue = softcap;
    runtime.allowFlashDecode = (blockConfig & FWD_FD_PERMISSION) != 0U;
    if constexpr (HasWindow) {
        runtime.windowSizeLeft = window.left;
        runtime.windowSizeRight = window.right;
    } else {
        runtime.windowSizeLeft = -1;
        runtime.windowSizeRight = defaultRight;
    }
    runtime.maskType = HasWindow ? 2U : (defaultRight == 0 ? 1U : 0U);
    return runtime;
}
#undef FA950_RUNTIME_INLINE
#endif
