/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef FAI_HOST_API_HPP
#define FAI_HOST_API_HPP

#include <cstdint>
#include <string>

#include "acl/acl.h"
#include "kernel_common.hpp"  // Format / CacheMode / PageShape / MaskCategory / CacheLayout

namespace fai_host {

//   bit 0: cacheLayout  (0=nd, 1=nz)
//   bit 1: dataType     (0=half, 1=bf16)
//   bit 2: maskType     (1 if causal mask)
//   bit 3: maskType     (1 if SWA mask)
//   bit 4: innerPrec    (0=fp32, 1=fp16) — bf16 forces 0
//   bit 5: layout       (0=TND, 1=BSND)
//   bit 6: cacheMode    (0=normalCache, 1=pagedCache)
//   bit 7: pageShape    (0=BnBsND, 1=BnNBsD) — only when cacheMode=paged
uint32_t BuildKernelKey(const std::string& dataType,     // "half" | "bf16"
                        const std::string& cacheLayout,  // "nd" | "nz"
                        uint32_t maskType,               // 0 | 1 | 4
                        uint32_t innerPrec,              // 0 | 1
                        Format layout,
                        CacheMode cacheMode,
                        PageShape pageShape);

aclError LaunchFAI(uint32_t kernelKey, bool enableDN,
                   uint32_t blockDim, aclrtStream stream,
                   uint8_t* qDevice, uint8_t* kDevice, uint8_t* vDevice,
                   uint8_t* maskDevice, uint8_t* blockTableDevice,
                   uint8_t* oDevice, uint8_t* lseDevice,
                   uint8_t* qSeqDevice, uint8_t* kvSeqDevice,
                   uint8_t* workspaceDevice, uint8_t* tilingDevice);

// Per-(dtype, layout) forward dispatch, a primary template. The top-level
// LaunchFAI() extracts the dtype (bit 1) and layout (bit 5) bits from kernelKey
// and routes to the matching instantiation. Each <DType, IS_TND> pair is
// explicitly instantiated in its own autogen translation unit
// (autogen/fai_dispatch_<dtype>_<layout>.cpp) so the FAInfer / FAInferDn
// template instantiations compile in parallel; the generic body (which selects
// the per-(dtype, layout) case set with if constexpr) lives in
// fai_host_api_impl.hpp. Returns ACL_SUCCESS on a registered case, or falls
// through (returns ACL_ERROR_INVALID_PARAM) if kernelKey has no matching case.
template <typename DType, bool IS_TND>
aclError launch_fai_dispatch(uint32_t kernelKey, bool enableDN,
                         uint32_t blockDim, aclrtStream stream,
                         uint8_t* qDevice, uint8_t* kDevice, uint8_t* vDevice,
                         uint8_t* maskDevice, uint8_t* blockTableDevice,
                         uint8_t* oDevice, uint8_t* lseDevice,
                         uint8_t* qSeqDevice, uint8_t* kvSeqDevice,
                         uint8_t* workspaceDevice, uint8_t* tilingDevice);

}  // namespace fai_host

#endif  // FAI_HOST_API_HPP
