/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Host-side mirror of the fag_det:: schedule selectors and round counters.
 *
 * The device header fag_det_schedule.hpp is compiled only inside __aicore__
 * kernels, so the host tiler cannot call it.  This header carries the same
 * selection rules and round formulas with plain inline functions.  Keep the
 * two in sync when the schedule definitions change.
 */

#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_DET_SCHEDULE_HOST_HPP
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_DET_SCHEDULE_HOST_HPP

#include <cstdint>

namespace fag_det_host {

enum Kind : uint32_t {
    KIND_NONE = 0,
    KIND_DENSE_SWIZZLE = 1,
    KIND_DENSE_INDEX = 2,
    KIND_CAUSAL_SWIZZLE = 3,
    KIND_LEFT_UP_CAUSAL_SWIZZLE = 4,
    KIND_GQA_DENSE = 5,
    KIND_TND_DENSE = 6,
    KIND_TND_GQA_DENSE = 7,
    KIND_TND_CAUSAL = 8,
};

inline int64_t HMin(int64_t a, int64_t b) { return a < b ? a : b; }
inline int64_t HMax(int64_t a, int64_t b) { return a > b ? a : b; }
inline int64_t HCeil(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline int64_t DenseMaxRound(int64_t k, int64_t m, int64_t n, int64_t b)
{
    return HCeil(n * b, HMin(k, m * b)) * m;
}

inline int64_t GqaDenseMaxRound(int64_t k, int64_t m, int64_t n, int64_t b, int64_t g)
{
    const int64_t kk = HMin(HMin(k, b * g * m), b * n);
    return HMax(HMax(HCeil(b * n * g, kk), HCeil(n, m)), g) * m;
}

inline int64_t LeftUpCausalSwizzleMaxRound(
    int64_t k, int64_t m, int64_t n, int64_t b)
{
    const int64_t pairCount = b >> 1;
    if (k <= 0 || m <= 0 || n <= 0 || pairCount <= 0) {
        return 0;
    }
    int64_t virtualM = m;
    int64_t virtualN = m + 1;
    int64_t activeK = HMin(k, m * pairCount);
    if (m > n) {
        virtualM = 2 * m - n + 1;
        virtualN = n;
        activeK = HMin(k, n * pairCount);
    }
    return virtualM * HCeil(virtualN * pairCount, activeK);
}

inline int64_t CausalSwizzleMaxRound(int64_t k, int64_t m, int64_t n, int64_t b)
{
    return HMax(m * HCeil((n + 1) * (b >> 1), k), n + 1);
}

struct Selection {
    uint32_t kind = KIND_NONE;
    bool supported = false;
};

// k = aicNum (cube cores).  BSND uniform layouts only.
inline Selection SelectSchedule(
    bool causal, int64_t batchBh, int64_t m, int64_t n, int64_t g, int64_t k)
{
    Selection sel;
    if (batchBh <= 0 || m <= 0 || n <= 0 || k <= 0) {
        return sel;
    }
    if (g == 1) {
        if (!causal) {
            // Swizzle needs k <= m for intra-round dq uniqueness; the
            // non-swizzle column rotation is safe for every k.
            sel.kind = (m >= k) ? KIND_DENSE_SWIZZLE : KIND_DENSE_INDEX;
            sel.supported = true;
            return sel;
        }
        // Top-left aligned square causal with an even batch count: the
        // LEFT_UP fold zips adjacent batches, which is the cheapest schedule.
        if (((batchBh & 1) == 0) && m == n) {
            sel.kind = KIND_LEFT_UP_CAUSAL_SWIZZLE;
            sel.supported = true;
            return sel;
        }
        // Rectangular or odd-batch causal MHA falls back to the dense
        // schedule with the causal mask applied in the epilogue: masked
        // blocks add exact zeros, and the epilogue now bounds the 256x256
        // mask window for far upper-triangle blocks.
        sel.kind = (m >= k) ? KIND_DENSE_SWIZZLE : KIND_DENSE_INDEX;
        sel.supported = true;
        return sel;
    }
    // GQA: dense columns with the causal mask applied in the epilogue (the
    // fold-based causal schedule assumes g == 1).  Masked blocks contribute
    // exact zeros; the shared ordered-atomic accumulation stays deterministic.
    sel.kind = KIND_GQA_DENSE;
    sel.supported = true;
    return sel;
}

inline int64_t ScheduleMaxRound(
    uint32_t kind, int64_t batchBh, int64_t m, int64_t n, int64_t g, int64_t k)
{
    switch (kind) {
        case KIND_DENSE_SWIZZLE:
        case KIND_DENSE_INDEX:
            return DenseMaxRound(k, m, n, batchBh);
        case KIND_LEFT_UP_CAUSAL_SWIZZLE:
        case KIND_CAUSAL_SWIZZLE:
            return LeftUpCausalSwizzleMaxRound(k, m, n, batchBh);
        case KIND_GQA_DENSE:
            return GqaDenseMaxRound(k, m, n, batchBh, g);
        default:
            return 0;
    }
}

inline int64_t ScheduleColumnRounds(uint32_t kind, int64_t m, int64_t n)
{
    switch (kind) {
        case KIND_DENSE_SWIZZLE:
        case KIND_DENSE_INDEX:
        case KIND_GQA_DENSE:
            return m;
        case KIND_CAUSAL_SWIZZLE:
        case KIND_LEFT_UP_CAUSAL_SWIZZLE:
            return (m > n) ? (2 * m - n + 1) : m;
        default:
            return 0;
    }
}

// Number of private dk/dv accumulation buffers per core: causal schedules
// fold two real columns into one round span, so both need their own accumulator.
inline uint32_t ScheduleBufNum(uint32_t kind)
{
    return (kind == KIND_CAUSAL_SWIZZLE ||
            kind == KIND_LEFT_UP_CAUSAL_SWIZZLE) ? 2U : 1U;
}

// TND dense safety: within one round the active lanes enumerate consecutive
// (n1, s2) columns whose s1 = (s2 + delta) % s1Outer must stay unique, so each
// batch needs s1Outer >= min(k, s2Outer).  Also rejects empty sequences.
inline bool TndDenseSafe(
    int64_t batch, const int64_t *seqQ, const int64_t *seqKv, int64_t n1,
    int64_t k, int64_t qTile, int64_t kvTile)
{
    if (batch <= 0 || seqQ == nullptr || seqKv == nullptr || n1 <= 0 ||
        k <= 0 || qTile <= 0 || kvTile <= 0) {
        return false;
    }
    for (int64_t b = 0; b < batch; ++b) {
        if (seqQ[b] <= 0 || seqKv[b] <= 0) {
            return false;
        }
        const int64_t s1Outer = HCeil(seqQ[b], qTile);
        const int64_t s2Outer = HCeil(seqKv[b], kvTile);
        const int64_t need = HMin(k, s2Outer);
        if (s1Outer < need) {
            return false;
        }
    }
    return true;
}

// TND dense: exclusive end round of each batch, mirroring opst
// CalcTNDSwizzleParam.  Returns prefix[batch] (the total schedule rounds).
// ---------------------------------------------------------------------------
// opst CalcleTNDCausalDeterPrefix/ParamNormal (MHA, step = 1): three prefix
// tables + tail round counts for the left-up causal schedule.  Only used when
// the shape is covered (batch small, MHA); the caller falls back to the dense
// schedule + causal mask otherwise.
// ---------------------------------------------------------------------------
struct TndCausalParams {
    bool supported = false;
    int64_t maxRound = 0;
    int64_t p0[FAGTiling950::TND_SWIZZLE_PREFIX_NUM] = {0};
    int64_t p1[FAGTiling950::TND_SWIZZLE_PREFIX_NUM] = {0};
    int64_t p2[FAGTiling950::TND_SWIZZLE_PREFIX_NUM] = {0};
};

inline TndCausalParams ComputeTndCausalParams(
    int64_t batch, const int64_t *seqQ, const int64_t *seqKv, int64_t n2,
    int64_t k, int64_t qTile, int64_t kvTile)
{
    TndCausalParams out{};
    // step = 1 layout needs batch + 3 prefix slots (p0 tail has two entries).
    if (batch <= 0 || seqQ == nullptr || seqKv == nullptr || n2 <= 0 ||
        k <= 0 || qTile <= 0 || kvTile <= 0 ||
        batch + 2 >= static_cast<int64_t>(FAGTiling950::TND_SWIZZLE_PREFIX_NUM)) {
        return out;
    }
    int64_t p0[FAGTiling950::TND_SWIZZLE_PREFIX_NUM] = {0};
    int64_t p1[FAGTiling950::TND_SWIZZLE_PREFIX_NUM] = {0};
    int64_t p2[FAGTiling950::TND_SWIZZLE_PREFIX_NUM] = {0};
    int64_t m0Max = 0, m1Max = 0, m2Max = 0;
    const int64_t N12 = (n2 % k) % 2;
    for (int64_t i = 0; i < batch; ++i) {
        if (seqQ[i] <= 0 || seqKv[i] <= 0) {
            return out;
        }
        const int64_t m = HCeil(seqQ[i], qTile);
        int64_t n = HCeil(seqKv[i], kvTile);
        if (m < n) {
            n = m;
        }
        m0Max = HMax(m0Max, 2 * m - n + 1);
        p0[i + 1] = p0[i] + (2 * m - n + 1) * n;
        if (N12 > 0) {
            p1[i + 1] = p1[i] + (m - (n + 1) / 2 + 1) * (n / 2);
            m1Max = HMax(m1Max, m - (n + 1) / 2 + 1);
            p2[i + 1] = p2[i] + (m - n / 2) * ((n + 1) / 2);
            m2Max = HMax(m2Max, m - n / 2);
        }
    }
    const int64_t N11 = (n2 % k) / 2;
    const int64_t N12b = (n2 % k) % 2;
    const int64_t prefix0Max1 = p0[batch] / 2 * (n2 / k);
    const int64_t prefix0Max2 = HMax(HCeil(p0[batch] * N11, k), m0Max);
    int64_t total = prefix0Max1;
    p0[batch + 1] = prefix0Max1;
    if (N11 > 0) {
        p0[batch + 2] = prefix0Max2;
        total += prefix0Max2;
    } else {
        p0[batch + 2] = 0;
    }
    if (N12b > 0) {
        const int64_t r1 = HMax(HCeil(p1[batch], k), m1Max);
        const int64_t r2 = HMax(HCeil(p2[batch], k), m2Max);
        p1[batch + 1] = r1;
        p2[batch + 1] = r2;
        total += r1 + r2;
    }
    // N12b == 0: p1/p2 stay zero-initialized (R1 = R2 = 0).
    for (int64_t i = 0; i < static_cast<int64_t>(FAGTiling950::TND_SWIZZLE_PREFIX_NUM); ++i) {
        out.p0[i] = p0[i];
        out.p1[i] = p1[i];
        out.p2[i] = p2[i];
    }
    out.maxRound = total;
    out.supported = total > 0;
    return out;
}

inline int64_t TndDensePrefix(
    int64_t batch, const int64_t *seqQ, const int64_t *seqKv, int64_t n1,
    int64_t k, int64_t qTile, int64_t kvTile, int64_t *prefix)
{
    if (batch <= 0 || seqQ == nullptr || seqKv == nullptr || prefix == nullptr ||
        n1 <= 0 || k <= 0 || qTile <= 0 || kvTile <= 0) {
        return 0;
    }
    prefix[0] = 0;
    for (int64_t b = 0; b < batch; ++b) {
        const int64_t s1Outer = HCeil(seqQ[b], qTile);
        const int64_t s2Outer = HCeil(seqKv[b], kvTile);
        prefix[b + 1] = prefix[b] + HCeil(n1 * s2Outer, k) * s1Outer;
    }
    return prefix[batch];
}

}  // namespace fag_det_host

#endif  // FLASH_ATTN_NPU_ASCEND950_V3_FAG_DET_SCHEDULE_HOST_HPP
