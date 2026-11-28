// Copyright (c) 2026, Minghua Shen.

#ifndef FLASH_ATTN_NPU_ASCEND950_FAG_DKV_CAST_HOST_HPP
#define FLASH_ATTN_NPU_ASCEND950_FAG_DKV_CAST_HOST_HPP

#include "fag_common.h"

#include <algorithm>
#include <initializer_list>
#include <vector>
#include <ATen/core/Tensor.h>

namespace FAGTiling950 {

// A direct Fixpipe store must initialize every element of both outputs.
// Dense and right-aligned causal TND visit every KV column for nonempty Q.
// A band, a used-length cap, or an unused packed tail can leave holes.
inline bool CanWriteDkvDirect(const FAGInfo& info,
                              const std::vector<int64_t>& queryLengths, int64_t kvEnd,
                              const at::Tensor& k, const at::Tensor& v, const at::Tensor& dk, const at::Tensor& dv,
                              std::initializer_list<const at::Tensor*> otherTensors)
{
    const bool unsupportedSchedule =
        info.layout != Layout::TND || info.deterministic || info.maskType == MaskType::BAND ||
        info.softcapValue > 0.0f || info.hasUsedLengths || info.qkHeadDim > 128 || info.vHeadDim > 128 ||
        info.qkHeadDim % 16 != 0 || info.vHeadDim % 16 != 0;
    const bool ineligibleForDirectCast =
        unsupportedSchedule || dk.sizes() != k.sizes() || dv.sizes() != v.sizes() ||
        dk.dtype() != k.dtype() || dv.dtype() != v.dtype();
    if (ineligibleForDirectCast || dk.is_alias_of(k) || dk.is_alias_of(v) || dv.is_alias_of(k) ||
        dv.is_alias_of(v) || dk.is_alias_of(dv)) {
        return false;
    }

    const bool coversAllKv =
        kvEnd == k.size(0) && kvEnd == v.size(0) &&
        !std::any_of(queryLengths.begin(), queryLengths.end(), [](int64_t length) { return length == 0; });
    return coversAllKv && std::none_of(otherTensors.begin(), otherTensors.end(), [&](const at::Tensor* tensor) {
        return dk.is_alias_of(*tensor) || dv.is_alias_of(*tensor);
    });
}

} // namespace FAGTiling950

#endif
