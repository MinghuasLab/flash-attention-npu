/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once
#include "acl/acl.h"
#include "mla_metadata.h"

namespace flash_attn::mla {
// Private, shape-only whole-M plan. Device prefixes are checked on every
// invocation before this plan can reach the numerical task ring.
struct MlaStaticPrefillPlan {
    MlaLayout layout{};
    uint32_t usedCoreNum = 0;
    uint32_t bEnd[26]{}, mEnd[26]{};
};
static_assert(std::is_trivially_copyable_v<MlaStaticPrefillPlan>);
// Shape-only decode geometry and the single-query FD table. Every AIC/AIV
// checks the current KV length and active page IDs before using a schedule.
struct MlaDynamicDecodePlan {
    MlaLayout layouts[4]{};
    uint64_t vectorEnds[17]{};
    uint32_t vectors[17]{};
    uint32_t enabled = 0; // 0: generic, 1: single query, 2: packed Q16 PA.
};
static_assert(std::is_trivially_copyable_v<MlaDynamicDecodePlan>);
struct MlaLaunchArgs {
    aclrtStream stream;
    uint64_t ffts, q, qv, k, v, out, lse, accum, accumLse, workspace, metadata, checked;
    uint64_t cuQuery, cuKv, usedQuery, usedKv, mask;
    MlaContext context;
    MlaBsndInputLayouts inputs;
    MlaPagedCacheInputs cache;
    float scale, softcap;
    MlaStaticPrefillPlan staticPrefill{};
    // Host routing only. LaunchCompressed copies the plan into the dedicated
    // decode argument and clears this pointer before passing it to device.
    const MlaDynamicDecodePlan* dynamicDecode = nullptr;
};
static_assert(sizeof(MlaLaunchArgs) <= 1536);
struct MlaDynamicDecodeLaunchArgs {
    MlaLaunchArgs launch;
    MlaDynamicDecodePlan plan;
};
static_assert(std::is_trivially_copyable_v<MlaDynamicDecodeLaunchArgs>);
static_assert(sizeof(MlaDynamicDecodeLaunchArgs) <= 1536);
// Bisheng's ASC launcher does not expose a scheduling-mode argument.
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
    const bool nativeSingleQueryDecode = c.valueDim == 512 && c.batch == 1 && c.kvHeads == 1 && c.packed && c.splits == 0 &&
                                         (MlaTnd(c) ? c.totalQ : c.sq) == 1 && c.heads <= 256 && c.heads % 16 == 0 &&
                                         c.window.kind == 0 && a.softcap == 0.0f;
    if ((c.margin && !nativeSingleQueryDecode) || (!c.packed && c.heads != c.kvHeads) || c.splits > 1)
        return LaunchGrouped<DType>(a);
    return c.valueDim == 512 ? LaunchCompressed<DType>(a) : LaunchPrefill<DType>(a);
}
} // namespace flash_attn::mla
