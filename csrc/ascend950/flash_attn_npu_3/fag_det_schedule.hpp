/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Deterministic BN2S2 (batch/head x kv-block column-private) task schedules
 * ported from ops-transformer arch35 FlashAttentionScoreGrad:
 *   attention/flash_attention_score_grad/op_kernel/arch35/deter.h
 *   op_host/arch35/flash_attention_score_grad_tiling_{common,normal}_regbase.cpp
 *
 * The schedule assigns every (core, round) pair a task coordinate
 * (batch, n2, g, s1Block, s2Block) so that
 *   - one core owns a fixed (batch, n2, g, s2Block) column for a whole
 *     block of rounds and rotates s1 inside it: dk/dv are therefore
 *     accumulated by a single core in program order;
 *   - within one round the active cores never write the same dq tile, so the
 *     only cross-core ordering requirement is across rounds.
 *
 * The functions here are pure scalar integer math and are shared by the
 * host tiler (round counting / validation) and the device kernel (decode).
 * Coordinates follow opst's 1-based convention internally and are converted
 * to the 0-based FAGBlockInfo convention by the Decode() adapter.
 */

#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_DET_SCHEDULE_HPP
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_DET_SCHEDULE_HPP

#include <cstdint>

// Coordinate/decode entry points are device-only and must be force-inlined
// into the kernel.  The scalar helpers and schedule selectors are plain
// inline (device code inlines them the same way catlass tla does), so the
// host tiler can call the selectors directly.
#define FAG_DET_AICORE __forceinline__ __aicore__

namespace fag_det {

// ---------------------------------------------------------------------------
// Scalar helpers (host/device shared; no std/device intrinsic dependency).
// ---------------------------------------------------------------------------

FAG_DET_AICORE int64_t DetMin(int64_t a, int64_t b) { return a < b ? a : b; }
FAG_DET_AICORE int64_t DetMax(int64_t a, int64_t b) { return a > b ? a : b; }

// Positive ceil-division, matching opst's Ceil<int64_t> usage in deter.h.
FAG_DET_AICORE int64_t DetCeil(int64_t a, int64_t b) { return (a + b - 1) / b; }

FAG_DET_AICORE int64_t DetGcd(int64_t a, int64_t b)
{
    int64_t r;
    while (b > 0) {
        r = a % b;
        a = b;
        b = r;
    }
    return a;
}

// ---------------------------------------------------------------------------
// Schedule kinds.  Numeric values are part of the host/device tiling ABI.
// ---------------------------------------------------------------------------

enum Kind : uint32_t {
    KIND_NONE = 0,
    KIND_DENSE_SWIZZLE = 1,       // opst CalDenseSwizzleIndex (MHA)
    KIND_DENSE_INDEX = 2,         // opst CalDenseIndex (MHA)
    KIND_CAUSAL_SWIZZLE = 3,      // opst CalCausalSwizzleIndex (MHA)
    KIND_LEFT_UP_CAUSAL_SWIZZLE = 4,  // opst CalLeftUpCausalSwizzleIndex (MHA)
    KIND_GQA_DENSE = 5,           // opst CalGQADenseIndex
    KIND_TND_DENSE = 6,           // opst CalTNDDenseSwizzleIndex (ragged MHA)
    KIND_TND_GQA_DENSE = 7,       // opst CalTNDDenseIndex !IS_N_EQUAL (ragged GQA)
    KIND_TND_CAUSAL = 8,          // opst CalTNDCausalIndex (ragged causal MHA)
};

struct Shape {
    int64_t batch = 0;
    int64_t qSeqLen = 0;
    int64_t kvSeqLen = 0;
    int64_t qHeadNum = 0;
    int64_t kvHeadNum = 0;
    int64_t groupNum = 1;
    int64_t qTile = 128;
    int64_t kvTile = 128;
    int64_t coreNum = 0;

    FAG_DET_AICORE int64_t M() const { return DetCeil(qSeqLen, qTile); }
    FAG_DET_AICORE int64_t N() const { return DetCeil(kvSeqLen, kvTile); }
    FAG_DET_AICORE int64_t Bh() const { return batch * kvHeadNum; }
};

// opst raw coordinate: w is the 1-based combined (b, n2, g) index, 0 = invalid.
struct RawCoord {
    int64_t w = 0;
    int64_t s1 = 0;  // 1-based q block
    int64_t s2 = 0;  // 1-based kv block
};

// ---------------------------------------------------------------------------
// opst CalDenseIndex: non-swizzle dense column rotation (safe for k > m).
// ---------------------------------------------------------------------------
FAG_DET_AICORE void CalDenseIndex(
    int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r,
    RawCoord &c)
{
    c.w = 0;
    k = DetMin(k, b * m);
    if (j > k) {
        return;
    }
    int64_t p = (DetCeil(r, m) - 1) * k + j;
    int64_t w = p % b;
    w = (w != 0) ? w : b;
    int64_t y = DetCeil(p, b);
    int64_t y1 = y % m;
    y1 = (y1 != 0) ? y1 : m;
    int64_t r1 = r % m;
    r1 = (r1 != 0) ? r1 : m;
    int64_t x = y1 + r1 - 1;
    if (x > m) {
        x -= m;
    }
    if (w >= 1 && w <= b && x >= 1 && x <= m && y >= 1 && y <= n) {
        c.w = w;
        c.s1 = x;
        c.s2 = y;
    }
}

// ---------------------------------------------------------------------------
// opst CalDenseSwizzleIndex: consecutive KV columns per lane, s1 rotates.
// Requires k <= b*m for the intra-round dq safety property.
// ---------------------------------------------------------------------------
FAG_DET_AICORE void CalDenseSwizzleIndex(
    int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r,
    RawCoord &c)
{
    c.w = 0;
    j = j - 1;
    r = r - 1;
    k = DetMin(k, b * m);
    if (j > k) {
        return;
    }
    int64_t p = (r / m) * k + j;
    int64_t w = p / n;
    int64_t y = p % n;
    int64_t x = (y + r) % m;
    if (x >= m) {
        x -= m;
    }
    w += 1;
    x += 1;
    y += 1;
    if (w >= 1 && w <= b && x >= 1 && x <= m && y >= 1 && y <= n) {
        c.w = w;
        c.s1 = x;
        c.s2 = y;
    }
}

// ---------------------------------------------------------------------------
// opst CalCausalSwizzleIndex: two adjacent batches folded into one virtual
// rectangle so both triangles are scheduled without bubbles.
//
// A lane's virtual column is fixed for a whole column while s1 rotates; the
// fold condition flips once per column, so the lane contributes to TWO real
// columns (odd/even batch of the pair).  foldInfo reports both real columns
// and which one this task belongs to, so the kernel can keep a per-parity
// private dk/dv accumulator and flush both at the column end.
// ---------------------------------------------------------------------------
struct FoldInfo {
    int64_t w[2] = {0, 0};   // 1-based combined (b,n2,g) index, 0 = absent
    int64_t s2[2] = {0, 0};  // 1-based kv block, 0 = absent
    int64_t parity = 0;      // accumulation buffer of this task
};

FAG_DET_AICORE void CalCausalSwizzleIndex(
    int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r,
    RawCoord &c, FoldInfo *fold = nullptr)
{
    c.w = 0;
    // Adjacent B or N are concatenated into one full S1S2 rectangle.
    int64_t nNew = n + 1;
    if (m != n) {
        nNew = (n - m + 2) + (n + 1);
    }
    int64_t bNew = b >> 1;
    CalDenseSwizzleIndex(k, m, nNew, bNew, j, r, c);
    if (c.w == 0) {
        return;
    }
    int64_t w = c.w;
    int64_t x = c.s1;
    int64_t y = c.s2;
    if (m == n) {
        if (y >= x + 1) {
            y = (n << 1) - m - y + 2;
            x = m + 1 - x;
            w = (w << 1);
        } else {
            w = (w << 1) - 1;
        }
    } else {
        if (y >= x + (n + 1) - m + 1) {
            y = ((n + 1) << 1) - m - y + 2;
            x = m + 1 - x;
            w = (w << 1);
        } else {
            w = (w << 1) - 1;
        }
    }
    if (w >= 1 && w <= b && x >= 1 && x <= m && y >= 1 && y <= n) {
        c.w = w;
        c.s1 = x;
        c.s2 = y;
        if (fold != nullptr && m == n) {
            // Uniform pair formulas: derive the other column from either one.
            const int64_t oddW = (w & 1) ? w : w - 1;
            if (w & 1) {
                const int64_t yv = y;
                fold->w[0] = oddW;
                fold->s2[0] = yv;
                fold->w[1] = oddW + 1;
                fold->s2[1] = (yv >= 2) ? (m - yv + 2) : 0;
                fold->parity = 0;
            } else {
                const int64_t yv = m - y + 2;
                fold->w[0] = oddW;
                fold->s2[0] = (yv <= m) ? yv : 0;
                fold->w[1] = oddW + 1;
                fold->s2[1] = y;
                fold->parity = 1;
            }
        }
    } else {
        c.w = 0;
    }
}

// ---------------------------------------------------------------------------
// opst CalLeftUpCausalSwizzleIndex (LEFT_UP_CAUSAL / top-left aligned causal).
// ---------------------------------------------------------------------------
FAG_DET_AICORE void CalLeftUpCausalSwizzleIndex(
    int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r,
    RawCoord &c, FoldInfo *fold = nullptr)
{
    c.w = 0;
    int64_t pairCount = b >> 1;
    if (k <= 0 || m <= 0 || n <= 0 || pairCount <= 0 || j < 1 || r < 1) {
        return;
    }

    if (m <= n) {
        int64_t activeK = DetMin(k, m * pairCount);
        if (j > activeK) {
            return;
        }
        // Internally the virtual rectangle is m x (m+1); the fold columns are
        // within the real kv range because n >= m here.
        CalCausalSwizzleIndex(activeK, m, m, b, j, r, c, fold);
        return;
    }

    int64_t virtualM = 2 * m - n + 1;
    int64_t activeK = DetMin(k, n * pairCount);
    if (j > activeK) {
        return;
    }
    int64_t columnId = (r - 1) / virtualM * activeK + j - 1;
    if (columnId >= n * pairCount) {
        return;
    }

    int64_t pairId = columnId / n + 1;
    int64_t virtualS2 = columnId % n + 1;
    int64_t virtualS1 = (r - 1) % virtualM + 1;
    int64_t oddBatchLen = m - virtualS2 + 1;
    if (virtualS1 <= oddBatchLen) {
        c.w = 2 * pairId - 1;
        c.s1 = virtualS2 + virtualS1 - 1;
        c.s2 = virtualS2;
        if (fold != nullptr) {
            fold->w[0] = 2 * pairId - 1;
            fold->s2[0] = virtualS2;
            fold->w[1] = 2 * pairId;
            fold->s2[1] = n - virtualS2 + 1;
            fold->parity = 0;
        }
        return;
    }

    int64_t evenBatchOffset = virtualS1 - oddBatchLen;
    c.w = 2 * pairId;
    c.s1 = m - evenBatchOffset + 1;
    c.s2 = n - virtualS2 + 1;
    if (fold != nullptr) {
        fold->w[0] = 2 * pairId - 1;
        fold->s2[0] = virtualS2;
        fold->w[1] = 2 * pairId;
        fold->s2[1] = n - virtualS2 + 1;
        fold->parity = 1;
    }
}

// ---------------------------------------------------------------------------
// opst CalGQADenseIndex / CalGQADenseIndexNoTune: per-(batch, kv-head) g
// columns are distributed so that each lane rotates s1 inside one group
// column per round span.
// ---------------------------------------------------------------------------
FAG_DET_AICORE void CalGQADenseIndex(
    int64_t k, int64_t m, int64_t n, int64_t b, int64_t coreId, int64_t roundId,
    int64_t g, RawCoord &c, int64_t denseRound = 0)
{
    c.w = 0;
    k = DetMin(DetMin(k, b * g * m), b * n);
    int64_t R = denseRound > 0
        ? denseRound
        : DetMax(DetMax(DetCeil(b * n * g, k), DetCeil(n, m)), g);
    if (coreId < 1 || coreId > k || roundId < 1 || roundId > R * m) {
        return;
    }
    int64_t ID = (coreId - 1) * R + DetCeil(roundId, m);
    int64_t localId = roundId % m;
    localId = (localId != 0) ? localId : m;

    int64_t num = g * n;
    if (ID > num * b) {
        return;
    }

    int64_t N = b * g;
    int64_t bId = ID % N;
    bId = (bId != 0) ? bId : N;
    bId = DetCeil(bId, g);
    int64_t y = DetCeil(ID, N);
    int64_t w = ID % g;
    w = (w != 0) ? w : g;

    int64_t gcd = DetGcd(N, R);
    int64_t t1 = R / gcd;
    int64_t t2 = N / gcd;

    int64_t t1New = t1 * m;
    int64_t y1 = y % t1New;
    y1 = (y1 != 0) ? y1 : t1New;
    int64_t offset = DetCeil(y1, t1);

    if (t1New < n) {
        int64_t n1 = n % t1New;
        n1 = (n1 != 0) ? n1 : t1New;
        if (y <= n - n1) {
            int64_t delta = DetCeil(y, t1New);
            ID += delta;
            if (ID > (delta - 1) * t2 * m * R + offset * t2 * R) {
                ID -= t2 * R;
            }
            bId = ID % N;
            bId = (bId != 0) ? bId : N;
            bId = DetCeil(bId, g);
            w = ID % g;
            w = (w != 0) ? w : g;
            y = DetCeil(ID, N);
        }
    }

    int64_t x = localId + offset - 1;
    if (x > m) {
        x -= m;
    }

    c.w = w + (bId - 1) * g;
    c.s1 = x;
    c.s2 = y;
}

// Decoded 0-based task coordinate.  Fold fields describe the private dk/dv
// accumulator layout for causal schedules (two real columns per round span).
struct Coord {
    int64_t batch = 0;
    int64_t n2 = 0;
    int64_t g = 0;
    int64_t s1 = 0;
    int64_t s2 = 0;
    bool valid = false;

    uint32_t parity = 0;       // accumulation buffer of this task
    uint32_t foldCount = 1;    // number of real columns in this round span
    uint32_t foldValid[2] = {1, 0};
    int64_t foldBatch[2] = {0, 0};
    int64_t foldN2[2] = {0, 0};
    int64_t foldG[2] = {0, 0};
    int64_t foldS2[2] = {0, 0};
};

FAG_DET_AICORE void RawToCoordTriple(
    const Shape &s, int64_t w, int64_t &batchIdx, int64_t &n2Idx,
    int64_t &gIdx)
{
    const int64_t n1 = s.groupNum * s.kvHeadNum;
    batchIdx = (w - 1) / n1;
    const int64_t n1Idx = (w - 1) % n1;
    n2Idx = n1Idx / s.groupNum;
    gIdx = n1Idx % s.groupNum;
}

FAG_DET_AICORE void RawToCoord(
    const Shape &s, const RawCoord &raw, Coord &out)
{
    out.valid = false;
    if (raw.w < 1 || raw.s1 < 1 || raw.s2 < 1) {
        return;
    }
    const int64_t batchIdx = (raw.w - 1) / (s.groupNum * s.kvHeadNum);
    const int64_t n1Idx = (raw.w - 1) % (s.groupNum * s.kvHeadNum);
    const int64_t n2Idx = n1Idx / s.groupNum;
    const int64_t gIdx = n1Idx % s.groupNum;
    if (batchIdx < 0 || batchIdx >= s.batch ||
        n2Idx < 0 || n2Idx >= s.kvHeadNum ||
        gIdx < 0 || gIdx >= s.groupNum ||
        raw.s1 > s.M() || raw.s2 > s.N()) {
        return;
    }
    out.batch = batchIdx;
    out.n2 = n2Idx;
    out.g = gIdx;
    out.s1 = raw.s1 - 1;
    out.s2 = raw.s2 - 1;
    out.valid = true;
}

// ---------------------------------------------------------------------------
// opst CalTNDDenseSwizzleIndex: ragged TND dense MHA.  The host serializes
// the per-batch round counts into prefix[] (exclusive end rounds); within a
// batch a lane owns one (n1, s2) column for s1Outer rounds while s1 rotates,
// and lanes beyond the batch's column count are holes.
// ---------------------------------------------------------------------------
template <typename CuPtr, typename PrefixPtr>
FAG_DET_AICORE bool CalTNDDenseSwizzleIndex(
    const Shape &s, CuPtr cuQ, CuPtr cuK, PrefixPtr prefix,
    int64_t j, int64_t r, Coord &out)
{
    out = Coord{};
    if (cuQ == nullptr || cuK == nullptr || prefix == nullptr ||
        j < 1 || j > s.coreNum || r < 1) {
        return false;
    }
    j -= 1;
    r -= 1;
    const int64_t n1 = s.kvHeadNum * s.groupNum;
    for (int64_t bIdx = 0; bIdx < s.batch; ++bIdx) {
        if (r >= prefix[bIdx + 1]) {
            continue;
        }
        const int64_t qStart =
            bIdx == 0 ? 0 : static_cast<int64_t>(cuQ[bIdx - 1]);
        const int64_t kvStart =
            bIdx == 0 ? 0 : static_cast<int64_t>(cuK[bIdx - 1]);
        const int64_t s1Len = static_cast<int64_t>(cuQ[bIdx]) - qStart;
        const int64_t s2Len = static_cast<int64_t>(cuK[bIdx]) - kvStart;
        if (s1Len <= 0 || s2Len <= 0) {
            return false;
        }
        const int64_t s1Outer = DetCeil(s1Len, s.qTile);
        const int64_t s2Outer = DetCeil(s2Len, s.kvTile);
        const int64_t delta = r - prefix[bIdx];
        const int64_t linearIdx = delta / s1Outer * s.coreNum + j;
        if (linearIdx >= s2Outer * n1) {
            return false;
        }
        const int64_t n1Idx = linearIdx / s2Outer;
        const int64_t s2Idx = linearIdx % s2Outer;
        const int64_t s1Idx = (s2Idx + delta) % s1Outer;
        out.batch = bIdx;
        out.n2 = n1Idx / s.groupNum;
        out.g = n1Idx % s.groupNum;
        out.s1 = s1Idx;
        out.s2 = s2Idx;
        out.valid = true;
        out.parity = 0;
        out.foldCount = 1;
        out.foldValid[0] = 1;
        out.foldValid[1] = 0;
        out.foldBatch[0] = out.batch;
        out.foldN2[0] = out.n2;
        out.foldG[0] = out.g;
        out.foldS2[0] = out.s2;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// opst CalTNDDenseIndex with IS_N_EQUAL=false: flat blocked partition for
// ragged GQA.  areaPrefix[b] holds cumulative s1Outer*s2Outer (NOT rounds);
// lane j owns the contiguous slice [(j-1)*R+1, j*R] of the flattened
// (batch, n2, s2, g, s1) task space and sweeps it in order.  The gcd tweak
// keeps the same-round (batch, n2, g, s1) keys distinct across lanes.
// ---------------------------------------------------------------------------
template <typename CuPtr, typename PrefixPtr>
FAG_DET_AICORE bool CalTNDDenseGqaIndex(
    const Shape &s, CuPtr cuQ, CuPtr cuK, PrefixPtr areaPrefix,
    int64_t j, int64_t r, int64_t maxRound, Coord &out)
{
    out = Coord{};
    if (cuQ == nullptr || cuK == nullptr || areaPrefix == nullptr ||
        j < 1 || j > s.coreNum || r < 1 || r > maxRound) {
        return false;
    }
    const int64_t n1 = s.kvHeadNum * s.groupNum;
    const int64_t ID = (j - 1) * maxRound + r;
    int64_t w = 0;
    while (w + 1 < s.batch && ID > areaPrefix[w + 1] * n1) {
        ++w;
    }
    if (ID > areaPrefix[w + 1] * n1) {
        return false;
    }
    int64_t delta = ID - areaPrefix[w] * n1;
    if (delta < 1) {
        return false;
    }
    const int64_t qStart = w == 0 ? 0 : static_cast<int64_t>(cuQ[w - 1]);
    const int64_t kStart = w == 0 ? 0 : static_cast<int64_t>(cuK[w - 1]);
    const int64_t s1Len = static_cast<int64_t>(cuQ[w]) - qStart;
    const int64_t s2Len = static_cast<int64_t>(cuK[w]) - kStart;
    if (s1Len <= 0 || s2Len <= 0) {
        return false;
    }
    const int64_t m = DetCeil(s1Len, s.qTile);
    const int64_t n = DetCeil(s2Len, s.kvTile);
    const int64_t g = s.groupNum;
    const int64_t base = m * n * g;
    const int64_t deltaN = (delta - 1) / base + 1;
    delta = delta % base;
    if (delta == 0) {
        delta = base;
    }
    const int64_t mNew = m * g;
    const int64_t gd = DetGcd(mNew, maxRound);
    const int64_t t1 = maxRound / gd;
    const int64_t t2 = mNew / gd;
    int64_t x = delta % mNew;
    if (x == 0) {
        x = mNew;
    }
    int64_t y = DetCeil(delta, mNew);
    if (t1 < n) {
        int64_t nTail = n % t1;
        nTail = (nTail != 0) ? nTail : t1;
        if (y <= n - nTail) {
            const int64_t deltaAdj = DetCeil(y, t1);
            delta += deltaAdj;
            if (delta > deltaAdj * t2 * maxRound) {
                delta -= t2 * maxRound;
            }
            x = delta % mNew;
            if (x == 0) {
                x = mNew;
            }
            y = DetCeil(delta, mNew);
        }
    }
    const int64_t n1Id = DetCeil(x, m);
    x = x % m;
    if (x == 0) {
        x = m;
    }
    if (deltaN > s.kvHeadNum || n1Id > g || y > n) {
        return false;
    }
    out.batch = w;
    out.n2 = deltaN - 1;
    out.g = n1Id - 1;
    out.s1 = x - 1;
    out.s2 = y - 1;
    out.valid = true;
    out.parity = 0;
    out.foldCount = 1;
    out.foldValid[0] = 1;
    out.foldValid[1] = 0;
    out.foldBatch[0] = out.batch;
    out.foldN2[0] = out.n2;
    out.foldG[0] = out.g;
    out.foldS2[0] = out.s2;
    return true;
}

// ---------------------------------------------------------------------------
// opst CalTNDCausalIndex (MHA): left-up causal decomposition of a ragged TND
// batch set.  Host serializes three prefix tables (p0/p1/p2) plus the tail
// round counts; each (round, lane) maps to one *real* tile of the causal
// triangle (verified in .docs/scratch/tnd_causal_schedule.py: coverage ==
// exact triangle, no duplicates, same-round dq keys / columns distinct, so
// the shared dk/dv workspace + round barrier is deterministic).
//
// step = 1 only (batch <= 128); the host falls back to dense+mask otherwise.
// ---------------------------------------------------------------------------
FAG_DET_AICORE void CalVirtualIndex(int64_t flag, int64_t &m, int64_t &n)
{
    if (m < n) {
        n = m;
    }
    if (flag == 0) {
        m = 2 * m - n + 1;
    } else if (flag == 1) {
        m = m - (n + 1) / 2 + 1;
        n = n / 2;
    } else {
        m = m - n / 2;
        n = (n + 1) / 2;
    }
}

FAG_DET_AICORE void CalCausalPosWholeBatch(
    int64_t m, int64_t n, int64_t a, int64_t &x, int64_t &y)
{
    const int64_t n1 = n / 2 * 2;
    const int64_t L = 2 * m - n1 + 1;
    const int64_t rmLocal = n1 * L / 2;
    if (a <= rmLocal) {
        y = DetCeil(a, L);
        int64_t r1 = a % L;
        r1 = (r1 != 0) ? r1 : L;
        x = r1 + y - 1;
        if (x > m) {
            x = 2 * m + 1 - x;
            y = n1 + 1 - y;
        }
    } else {
        const int64_t a1 = a - rmLocal;
        y = n;
        x = a1 - 1 + y;
    }
}

// Inner decode (flag picks the virtual rectangle).  Outputs the 1-based
// combined head index comb = batch * n1 + deltaN and the virtual (x, y).
template <typename CuPtr, typename PrefixPtr>
FAG_DET_AICORE bool CalTNDDenseCausalIndex(
    const Shape &s, CuPtr cuQ, CuPtr cuK, PrefixPtr prefix,
    int64_t deterMaxRound, int64_t innerN1, int64_t flag,
    int64_t j, int64_t r, int64_t &comb, int64_t &x, int64_t &y,
    int64_t &realM, int64_t &realN)
{
    if (cuQ == nullptr || cuK == nullptr || prefix == nullptr ||
        j < 1 || j > s.coreNum || r < 1 || r > deterMaxRound) {
        return false;
    }
    const int64_t ID = (j - 1) * deterMaxRound + r;
    int64_t w = 0;
    while ((w + 1) < s.batch && ID > prefix[w + 1] * innerN1) {
        ++w;
    }
    if (w >= s.batch) {
        return false;
    }
    const int64_t qStart = w == 0 ? 0 : static_cast<int64_t>(cuQ[w - 1]);
    const int64_t kvStart = w == 0 ? 0 : static_cast<int64_t>(cuK[w - 1]);
    const int64_t s1Len = static_cast<int64_t>(cuQ[w]) - qStart;
    const int64_t s2Len = static_cast<int64_t>(cuK[w]) - kvStart;
    if (s1Len <= 0 || s2Len <= 0) {
        return false;
    }
    realM = DetCeil(s1Len, s.qTile);
    realN = DetCeil(s2Len, s.kvTile);
    int64_t vm = realM;
    int64_t vn = (realM < realN) ? realM : realN;
    CalVirtualIndex(flag, vm, vn);
    const int64_t base = vm * vn;
    int64_t delta = ID - prefix[w] * innerN1;
    const int64_t deltaN = (delta - 1) / base + 1;
    delta = delta % base;
    if (delta == 0) {
        delta = base;
    }
    const int64_t gd = DetGcd(vm, deterMaxRound);
    const int64_t t1 = deterMaxRound / gd;
    const int64_t t2 = vm / gd;
    x = (delta - 1) % vm + 1;
    y = (delta - 1) / vm + 1;
    if (t1 < vn) {
        int64_t nTail = vn % t1;
        nTail = (nTail != 0) ? nTail : t1;
        if (y <= vn - nTail) {
            const int64_t adj = DetCeil(y, t1);
            delta += adj;
            if (delta > adj * t2 * deterMaxRound) {
                delta -= t2 * deterMaxRound;
            }
            x = (delta - 1) % vm + 1;
            y = (delta - 1) / vm + 1;
        }
    }
    comb = w * innerN1 + deltaN;
    return true;
}

template <typename CuPtr>
FAG_DET_AICORE int64_t TndBatchOuterQ(const Shape &s, CuPtr cuQ, int64_t bIdx)
{
    const int64_t qs = bIdx == 0 ? 0 : static_cast<int64_t>(cuQ[bIdx - 1]);
    return DetCeil(static_cast<int64_t>(cuQ[bIdx]) - qs, s.qTile);
}

template <typename CuPtr>
FAG_DET_AICORE int64_t TndBatchOuterK(const Shape &s, CuPtr cuK, int64_t bIdx)
{
    const int64_t ks = bIdx == 0 ? 0 : static_cast<int64_t>(cuK[bIdx - 1]);
    return DetCeil(static_cast<int64_t>(cuK[bIdx]) - ks, s.kvTile);
}

template <typename CuPtr, typename PrefixPtr>
FAG_DET_AICORE bool CalTNDCausalIndex(
    const Shape &s, CuPtr cuQ, CuPtr cuK, PrefixPtr p0, PrefixPtr p1,
    PrefixPtr p2, int64_t maxRound, int64_t j, int64_t r, Coord &out)
{
    out = Coord{};
    if (p0 == nullptr || maxRound <= 0 ||
        j < 1 || j > s.coreNum || r < 1 || r > maxRound) {
        return false;
    }
    const int64_t N1 = s.kvHeadNum * s.groupNum;   // g == 1 -> kvHeadNum
    const int64_t N11 = (s.kvHeadNum % s.coreNum) / 2;   // inner n2 (segment 2)
    const int64_t maxIdx = s.batch + 1;            // step == 1
    const int64_t R01 = p0[maxIdx];
    const int64_t R02 = p0[maxIdx + 1];
    const int64_t R0 = R01 + R02;
    const int64_t R1 = p1 != nullptr ? p1[maxIdx] : 0;
    const int64_t R2 = p2 != nullptr ? p2[maxIdx] : 0;

    int64_t batch = 0;
    int64_t hn2 = 0;
    int64_t x = 0;
    int64_t y = 0;
    int64_t realM = 0;
    int64_t realN = 0;


    if (r <= R01) {
        const int64_t N10 = N1 / s.coreNum;
        if (N10 <= 0) {
            return false;
        }
        const int64_t aJudge = DetCeil(r * 2, N10);
        int64_t w = 0;
        while ((w + 1) < s.batch && aJudge > p0[w + 1]) {
            ++w;
        }
        if (w >= s.batch) {
            return false;
        }
        int64_t a = r - p0[w] * N10 / 2;
        int64_t m = TndBatchOuterQ(s, cuQ, w);
        int64_t n = TndBatchOuterK(s, cuK, w);
        n = (m < n) ? m : n;
        int64_t roundBatch = (2 * m - n + 1) * n / 2;
        while (a > roundBatch * N10) {
            a -= roundBatch * N10;
            ++w;
            if (w >= s.batch) {
                return false;
            }
            m = TndBatchOuterQ(s, cuQ, w);
            n = TndBatchOuterK(s, cuK, w);
            n = (m < n) ? m : n;
            roundBatch = (2 * m - n + 1) * n / 2;
        }
        int64_t a0 = a % roundBatch;
        a0 = (a0 != 0) ? a0 : roundBatch;
        int64_t cx = 0;
        int64_t cy = 0;
        CalCausalPosWholeBatch(m, n, a0, cx, cy);
        int64_t ww = (a - 1) / roundBatch * s.coreNum + j;
        ww = ((ww - 1) / s.coreNum) * s.coreNum +
             ((cy - 1 + (ww - 1)) % s.coreNum) + 1;
        const int64_t head = w * N1 + ww;
        batch = (head - 1) / N1;
        hn2 = (head - 1) % N1 + 1;
        x = cx;
        y = cy;
        realM = m;
        realN = n;
    } else {
        int64_t comb = 0;
        int64_t cx = 0;
        int64_t cy = 0;
        if (r <= R0) {
            if (!CalTNDDenseCausalIndex(s, cuQ, cuK, p0, R02, N11, 0,
                                        j, r - R01, comb, cx, cy, realM, realN)) {
                return false;
            }
            const int64_t b1 = DetCeil(comb, N11);
            int64_t b2 = comb % N11;
            b2 = (b2 != 0) ? b2 : N11;
            const int64_t m = realM;
            int64_t n = (m < realN) ? m : realN;
            if (cx >= cy + m - n + 1) {
                x = cx - (m - n + 1);
                y = cy;
                b2 = 2 * b2 - 1;
            } else {
                x = m + 1 - cx;
                y = n + 1 - cy;
                b2 = 2 * b2;
            }
            const int64_t head = (b1 - 1) * N1 + (N1 / s.coreNum) * s.coreNum + b2;
            batch = (head - 1) / N1;
            hn2 = (head - 1) % N1 + 1;
        } else if (r <= R0 + R1) {
            if (!CalTNDDenseCausalIndex(s, cuQ, cuK, p1, R1, 1, 1,
                                        j, r - R0, comb, cx, cy, realM, realN)) {
                return false;
            }
            const int64_t m = realM;
            int64_t n = (m < realN) ? m : realN;
            if (cx >= cy + m - n + 1) {
                x = cx - (m - n + 1);
                y = cy;
            } else {
                x = m + 1 - cx;
                y = n + 1 - cy;
            }
            const int64_t head = comb * N1;
            batch = (head - 1) / N1;
            hn2 = (head - 1) % N1 + 1;
        } else {
            if (!CalTNDDenseCausalIndex(s, cuQ, cuK, p2, R2, 1, 2,
                                        j, r - R0 - R1, comb, cx, cy, realM, realN)) {
                return false;
            }
            int64_t n = (realM < realN) ? realM : realN;
            x = cx + n / 2;
            y = cy;
            const int64_t head = comb * N1;
            batch = (head - 1) / N1;
            hn2 = (head - 1) % N1 + 1;
        }
    }

    if (batch < 0 || batch >= s.batch || hn2 < 1 || hn2 > s.kvHeadNum) {
        return false;
    }
    const int64_t chkM = TndBatchOuterQ(s, cuQ, batch);
    const int64_t chkN = TndBatchOuterK(s, cuK, batch);
    if (x < 1 || x > chkM || y < 1 || y > chkN) {
        return false;
    }
    out.batch = batch;
    out.n2 = hn2 - 1;
    out.g = 0;
    out.s1 = x - 1;
    out.s2 = y - 1;
    out.valid = true;
    out.parity = 0;
    out.foldCount = 1;
    out.foldValid[0] = 1;
    out.foldValid[1] = 0;
    out.foldBatch[0] = out.batch;
    out.foldN2[0] = out.n2;
    out.foldG[0] = out.g;
    out.foldS2[0] = out.s2;
    return true;
}

// Decode the task of (round, core).  round and core are 1-based.
FAG_DET_AICORE bool Decode(
    Kind kind, const Shape &s, int64_t round, int64_t core, Coord &out)
{
    out = Coord{};
    if (kind == KIND_NONE || round < 1 || core < 1 || core > s.coreNum) {
        return false;
    }
    RawCoord raw;
    FoldInfo fold;
    const FoldInfo *foldPtr = nullptr;
    switch (kind) {
        case KIND_DENSE_SWIZZLE:
            CalDenseSwizzleIndex(
                s.coreNum, s.M(), s.N(), s.Bh(), core, round, raw);
            break;
        case KIND_DENSE_INDEX:
            CalDenseIndex(
                s.coreNum, s.M(), s.N(), s.Bh(), core, round, raw);
            break;
        case KIND_LEFT_UP_CAUSAL_SWIZZLE:
            foldPtr = &fold;
            CalLeftUpCausalSwizzleIndex(
                s.coreNum, s.M(), s.N(), s.Bh(), core, round, raw, &fold);
            break;
        case KIND_CAUSAL_SWIZZLE:
            foldPtr = &fold;
            CalCausalSwizzleIndex(
                s.coreNum, s.M(), s.N(), s.Bh(), core, round, raw, &fold);
            break;
        case KIND_GQA_DENSE: {
            // opst tunes denseRound on the host; without the tuning pass use
            // the default R formula (CalGQADenseIndexNoTune).
            CalGQADenseIndex(
                s.coreNum, s.M(), s.N(), s.Bh(), core, round, s.groupNum,
                raw);
            break;
        }
        default:
            return false;
    }
    RawToCoord(s, raw, out);
    if (!out.valid) {
        return false;
    }
    if (foldPtr == nullptr) {
        // Dense / GQA: one private buffer, the task's own column.
        out.parity = 0;
        out.foldCount = 1;
        out.foldValid[0] = 1;
        out.foldValid[1] = 0;
        out.foldBatch[0] = out.batch;
        out.foldN2[0] = out.n2;
        out.foldG[0] = out.g;
        out.foldS2[0] = out.s2;
    } else {
        out.parity = static_cast<uint32_t>(fold.parity);
        for (uint32_t p = 0; p < 2; ++p) {
            if (fold.w[p] < 1 || fold.s2[p] < 1) {
                out.foldValid[p] = 0;
                continue;
            }
            int64_t bIdx = 0;
            int64_t n2Idx = 0;
            int64_t gIdx = 0;
            RawToCoordTriple(s, fold.w[p], bIdx, n2Idx, gIdx);
            if (bIdx < 0 || bIdx >= s.batch ||
                n2Idx < 0 || n2Idx >= s.kvHeadNum ||
                gIdx < 0 || gIdx >= s.groupNum ||
                fold.s2[p] > s.N()) {
                out.foldValid[p] = 0;
                continue;
            }
            out.foldValid[p] = 1;
            out.foldBatch[p] = bIdx;
            out.foldN2[p] = n2Idx;
            out.foldG[p] = gIdx;
            out.foldS2[p] = fold.s2[p] - 1;
        }
        out.foldCount = out.foldValid[0] + out.foldValid[1];
    }
    return true;
}

}  // namespace fag_det

#endif  // FLASH_ATTN_NPU_ASCEND950_V3_FAG_DET_SCHEDULE_HPP
