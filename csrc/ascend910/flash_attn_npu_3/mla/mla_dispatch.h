/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once
#include "acl/acl.h"
#include "mla_metadata.h"
#include "mla_int8_schedule.h"

namespace flash_attn::mla {
struct MlaLaunchArgs {
    aclrtStream stream;
    uint64_t ffts, q, qv, k, v, out, lse, accum, accumLse, workspace, metadata, checked;
    uint64_t cuQuery, cuKv, usedQuery, usedKv;
    MlaContext context;
    MlaBsndInputLayouts inputs;
    MlaPagedCacheInputs cache;
    float scale, softcap;
};
// Bisheng 9.0's ASC launcher does not expose a scheduling-mode argument.
// The link wrapper changes only the launch inside this thread-local scope.
// Ordinary FA3 calls pass through unchanged; a different compiler path fails
// explicitly instead of silently launching MLA in NORMAL_MODE.
struct MlaBatchLaunchState {
    uint32_t calls = 0;
    aclError error = ACL_SUCCESS;
};
extern thread_local MlaBatchLaunchState* activeMlaLaunch;

class MlaBatchLaunchScope {
  public:
    MlaBatchLaunchScope() : previous(activeMlaLaunch)
    {
        activeMlaLaunch = &state;
    }
    ~MlaBatchLaunchScope()
    {
        activeMlaLaunch = previous;
    }
    MlaBatchLaunchScope(const MlaBatchLaunchScope&) = delete;
    MlaBatchLaunchScope& operator=(const MlaBatchLaunchScope&) = delete;
    aclError Result() const
    {
        return state.calls == 1 ? state.error : ACL_ERROR_INVALID_PARAM;
    }

  private:
    MlaBatchLaunchState state;
    MlaBatchLaunchState* previous;
};

template <typename DType>
aclError LaunchCompressed(const MlaLaunchArgs&);
template <typename DType>
aclError LaunchPrefill(const MlaLaunchArgs&);
template <typename DType>
aclError LaunchGrouped(const MlaLaunchArgs&);
template <typename DType>
aclError LaunchMla(const MlaLaunchArgs& a)
{
    const auto& c = a.context;
    if (c.margin || (!c.packed && c.heads != c.kvHeads) || c.splits > 1)
        return LaunchGrouped<DType>(a);
    return c.valueDim == 512 ? LaunchCompressed<DType>(a) : LaunchPrefill<DType>(a);
}
} // namespace flash_attn::mla
namespace flash_attn::mla::Int8 {
struct LaunchArgs {
    aclrtStream stream;
    uint64_t ffts, q, qv, k, v, qs, ks, vs, table, lengths, metadata, mask, out, lse, accum, accumLse, workspace;
    Config config;
    MlaPagedCacheLayout cache;
    float scale;
    uint64_t checked = 0;
    uint64_t prefix = 0;
    uint32_t totalQuery = 0;
};
aclError Launch(const LaunchArgs&);
} // namespace flash_attn::mla::Int8
