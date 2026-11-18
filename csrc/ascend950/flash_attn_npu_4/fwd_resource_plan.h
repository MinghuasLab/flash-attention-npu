/** Copyright (c) 2026 Huawei Technologies Co., Ltd. CANN Open Software License Agreement Version 2.0. */
#ifndef FA950_FWD_RESOURCE_PLAN_H
#define FA950_FWD_RESOURCE_PLAN_H

#include <cstdint>

#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
#define FA950_PLAN_INLINE __forceinline__[host, aicore]
#else
#define FA950_PLAN_INLINE inline
#endif

namespace fa950_plan {
constexpr uint32_t UB_CAPACITY = 248U * 1024U;
constexpr uint32_t L1_CAPACITY = 512U * 1024U;
constexpr uint32_t SCORE_STAGES = 2;
constexpr uint32_t P_STAGES = 3;
constexpr uint32_t PV_FLAG_BASE = SCORE_STAGES + P_STAGES;

struct UbPlan {
    uint32_t s, p, lo, go, lm, gm, dm, ll, gl, mask, lse, total;
    uint32_t sBytes, pBytes, loBytes, goBytes, lseBytes, otmpStages;
    // P UB slots: the legacy plan keeps two slots rotating with the Score
    // slot; the D512 plan keeps a single slot with a fixed copy event.
    uint32_t pStages;
};

// Exact D512 profile (Dq=Dk=Dv=512, Q tile 128). Score keeps two slots and the
// DM ring keeps three slots for the in-task PRE_LAUNCH=2 pipeline; P UB, LO and
// K/V L1 slots are single. LO holds one 64x128 FP32 column piece of the PV
// result; GO holds the full 64x512 FP32 accumulator. No mask handle exists:
// D512 causal/SWA masking is positional. LSE scratch is a dedicated 256 B
// buffer used only by the restricted single-head fast path.
FA950_PLAN_INLINE constexpr UbPlan MakeUbPlanD512()
{
    UbPlan u{};
    u.sBytes = 64U * 128U * 4U;         // Score, two slots of 32768 B
    u.pBytes = (64U + 1U) * 128U * 2U;  // P UB, one slot of 16640 B
    u.loBytes = 64U * 128U * 4U;        // LO, one slot of 32768 B
    u.goBytes = 64U * 512U * 4U;        // GO, one slot of 131072 B
    u.lseBytes = 64U * 4U;              // LSE scratch of 256 B
    u.otmpStages = 1U;
    u.pStages = 1U;
    u.s = 0U;
    u.p = SCORE_STAGES * u.sBytes;      // 65536
    u.lo = u.p + u.pBytes;              // 82176
    u.go = u.lo + u.loBytes;            // 114944
    u.lm = u.go + u.goBytes;            // 246016
    u.gm = u.lm + 64U * 4U;             // 246272
    u.dm = u.gm + 64U * 4U;             // 246528
    u.ll = u.dm + P_STAGES * 64U * 4U;  // 247296
    u.gl = u.ll + 64U * 4U;             // 247552
    u.mask = 0U;
    u.lse = u.gl + 64U * 4U;            // 247808
    u.total = u.lse + u.lseBytes;       // 248064
    return u;
}

// Score/P retain the 64-row capacity needed by merged GQA and DN. The
// reduction vectors also retain 64 elements because VF loads whole registers.
FA950_PLAN_INLINE constexpr UbPlan MakeUbPlan(uint32_t dv, uint32_t otmpStages)
{
    const uint32_t alignedDv = (dv + 15U) / 16U * 16U;
    UbPlan u{};
    u.sBytes = 64U * 128U * 4U;
    u.pBytes = (64U + 1U) * 128U * 2U;
    u.loBytes = 64U * alignedDv * 4U;
    u.goBytes = u.loBytes;
    u.lseBytes = 0U;
    u.otmpStages = otmpStages;
    u.pStages = SCORE_STAGES;
    u.p = SCORE_STAGES * u.sBytes;
    u.lo = u.p + SCORE_STAGES * u.pBytes;
    u.go = u.lo + otmpStages * u.loBytes;
    u.lm = u.go + u.loBytes;
    u.gm = u.lm + 64U * 4U;
    u.dm = u.gm + 64U * 4U;
    u.ll = u.dm + P_STAGES * 64U * 4U;
    u.gl = u.ll + 64U * 4U;
    u.mask = u.gl + 64U * 4U;
    u.lse = 0U;
    u.total = u.mask + 2U * 64U * 128U;
    return u;
}

FA950_PLAN_INLINE constexpr uint32_t OTmpStages(uint32_t dv)
{
    return MakeUbPlan(dv, 2U).total <= UB_CAPACITY ? 2U : 1U;
}

FA950_PLAN_INLINE constexpr bool IsD512(uint32_t d, uint32_t dv)
{
    return d == 512U && dv == 512U;
}

FA950_PLAN_INLINE constexpr UbPlan KernelUbPlan(uint32_t dv, uint32_t qTile)
{
    if (dv == 512U && qTile == 128U) {
        return MakeUbPlanD512();
    }
    if (dv > 128U && qTile == 128U) {
        return MakeUbPlan(dv, OTmpStages(dv));
    }
    // Preserve the established small-query/DN layout and its pipeline depths.
    UbPlan u = MakeUbPlan(128U, 2U);
    u.lo = 128U * 1024U;
    u.go = 192U * 1024U;
    u.lm = 224U * 1024U;
    u.gm = u.lm + 256U;
    u.dm = u.gm + 256U;
    u.ll = u.dm + 768U;
    u.gl = u.ll + 256U;
    u.mask = u.gl + 256U;
    u.total = u.mask + 16384U;
    return u;
}

// D512 packs Q(128x512) + K(512x128) + P(128x128, 3 slots) + V(128x512) into
// one 480 KiB L1 region; the legacy K/V reservations do not apply to it.
FA950_PLAN_INLINE constexpr uint32_t D512L1Bytes()
{
    return 128U * 512U * 2U + 512U * 128U * 2U + 128U * 128U * 2U * P_STAGES + 128U * 512U * 2U;
}

FA950_PLAN_INLINE constexpr uint32_t L1Bytes(uint32_t d, uint32_t dv)
{
    if (IsD512(d, dv)) {
        return D512L1Bytes();
    }
    return 128U * ((d + 15U) / 16U * 16U) * 2U + 192U * 256U * 2U * 2U + 128U * 128U * 2U * P_STAGES +
           128U * ((dv + 15U) / 16U * 16U) * 2U * 2U;
}

FA950_PLAN_INLINE constexpr uint32_t QueryTile(uint32_t d, uint32_t dv, uint32_t maxQ)
{
    // D512 has a single legal profile (Q128). The legacy Q64 fallback would
    // land in the fixed small-query UB layout, which is not a D512 plan.
    if (IsD512(d, dv)) {
        return 128U;
    }
    // Preserve small-query task parallelism and existing decode grouping.
    return dv > 128U &&
                   (maxQ <= 64U || L1Bytes(d, dv) > L1_CAPACITY || MakeUbPlan(dv, OTmpStages(dv)).total > UB_CAPACITY)
               ? 64U
               : 128U;
}

FA950_PLAN_INLINE constexpr uint32_t QueryTileForTasks(uint32_t d, uint32_t dv, uint32_t maxQ, uint64_t fullQ128Tasks,
                                                       uint32_t cores)
{
    if (IsD512(d, dv)) {
        return 128U;
    }
    // Larger tiles reduce independent Q work. Keep several complete waves so
    // tail tasks and the single-PV-buffer latency do not dominate small inputs.
    // Count only full Q tiles; ragged/empty sequences must not inflate occupancy.
    return dv > 128U && fullQ128Tasks < uint64_t(cores) * 4U ? 64U : QueryTile(d, dv, maxQ);
}

static_assert(MakeUbPlanD512().s == 0U && MakeUbPlanD512().p == 65536U && MakeUbPlanD512().lo == 82176U &&
                  MakeUbPlanD512().go == 114944U && MakeUbPlanD512().lm == 246016U &&
                  MakeUbPlanD512().dm == 246528U && MakeUbPlanD512().gl == 247552U &&
                  MakeUbPlanD512().lse == 247808U && MakeUbPlanD512().total == 248064U,
              "D512 UB offsets");
static_assert(MakeUbPlanD512().total <= UB_CAPACITY, "D512 UB capacity");
static_assert(MakeUbPlanD512().loBytes == 32768U && MakeUbPlanD512().goBytes == 131072U, "D512 LO/GO capacities");
static_assert(MakeUbPlanD512().pStages == 1U && MakeUbPlanD512().otmpStages == 1U, "D512 single-slot stages");
static_assert(D512L1Bytes() == 491520U && D512L1Bytes() <= L1_CAPACITY, "D512 L1 capacity");
static_assert(MakeUbPlan(160, 2).total == 239872U, "D160 dual-buffer layout");
static_assert(MakeUbPlan(192, 1).total == 215296U, "D192 single-buffer layout");
static_assert(MakeUbPlan(256, 1).total == 248064U, "D256 single-buffer layout");
static_assert(MakeUbPlan(256, 1).total <= UB_CAPACITY, "FP16/BF16 UB capacity");
static_assert(L1Bytes(256, 256) <= L1_CAPACITY, "FP16/BF16 L1 capacity");
} // namespace fa950_plan

#undef FA950_PLAN_INLINE
#endif
