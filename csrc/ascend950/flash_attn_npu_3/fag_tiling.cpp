/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend950-specific host tiling and workspace calculation for FA v3 bwd.
 */

#include "fag_common.h"
#include "fag_det_schedule_host.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>

namespace FAGTiling950 {
namespace {

constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t FP32_ROW_ALIGN = 8;  // 32B alignment in float elements

uint64_t RoundUpU64(uint64_t value, uint64_t align)
{
    return (value + align - 1) / align * align;
}

uint64_t CeilDivU64(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

// BN2S2 UB budget: the main pipeline owns two ping/pong halves and the dk/dv
// cast owns a chunked tail region (mirrors the kernel Init formula).
int64_t CheckBn2s2CastUb(const FAGTilingData &tiling, const FAGInfo &info)
{
    const uint64_t FP32_BYTES = sizeof(float);
    const uint64_t rowsPerSub = RoundUpU64(tiling.qTile, 16) / 2;
    const uint64_t mmResBytes = rowsPerSub * tiling.kvTile * FP32_BYTES;
    const uint64_t attenMaskBytes = rowsPerSub * tiling.kvTile;
    const uint64_t lseBytes = RoundUpU64(rowsPerSub, 8) * 8 * FP32_BYTES;
    const uint64_t pBytes = (rowsPerSub + 1) * tiling.kvTile * 2;
    const uint64_t deltaBytes = rowsPerSub * 8 * FP32_BYTES;
    const uint64_t pOff = RoundUpU64(
        RoundUpU64(2 * mmResBytes + attenMaskBytes, 32) + lseBytes, 32);
    const uint64_t dSOff = RoundUpU64(pOff + pBytes, 32);
    const uint64_t deltaOff = RoundUpU64(dSOff + pBytes, 32);
    const uint64_t halfUb = RoundUpU64(deltaOff + deltaBytes, 32);
    const uint64_t pipelineUb = 2 * halfUb;
    const uint64_t chunkRows = tiling.kvTile < 32 ? tiling.kvTile : 32;
    const uint64_t dAlign = RoundUpU64(info.qkHeadDim, FP32_ROW_ALIGN);
    const uint64_t dvAlign = RoundUpU64(info.vHeadDim, FP32_ROW_ALIGN);
    const uint64_t inCol = dAlign > dvAlign ? dAlign : dvAlign;
    const uint64_t headCol =
        info.qkHeadDim > info.vHeadDim ? info.qkHeadDim : info.vHeadDim;
    const uint64_t castUb =
        chunkRows * inCol * FP32_BYTES + chunkRows * headCol * 2;
    if (pipelineUb + castUb > info.ubSize) {
        fprintf(stderr,
            "FAG950 det bn2s2: UB too small: pipeline %llu + cast %llu > %llu\n",
            (unsigned long long)pipelineUb, (unsigned long long)castUb,
            (unsigned long long)info.ubSize);
        return -1;
    }
    return 0;
}

}  // namespace

int64_t GetFAGTilingParam(const FAGInfo &info, FAGTilingData &tiling)
{
    if (info.batch == 0 || info.qSeqlen == 0 || info.kvSeqlen == 0 ||
        info.totalQ == 0 || info.totalKv == 0 || info.qHeadNum == 0 ||
        info.kvHeadNum == 0 || info.qHeadNum % info.kvHeadNum != 0 ||
        info.qkHeadDim == 0 || info.vHeadDim == 0 || info.aicNum == 0 ||
        info.aivNum == 0 || info.continuousBlockNum == 0 || info.ubSize == 0) {
        return -1;
    }

    tiling = FAGTilingData{};
    tiling.layout = static_cast<uint32_t>(info.layout);
    tiling.maskType = static_cast<uint32_t>(info.maskType);
    tiling.deterministic = info.deterministic;
    tiling.detSchedule =
        info.deterministic ? info.detSchedule : static_cast<uint32_t>(DetSchedule::LEGACY);
    tiling.aicNum = info.aicNum;
    tiling.aivNum = info.aivNum;
    // tiling.
    tiling.continuousBlockNum = info.continuousBlockNum;
    tiling.ubSize = info.ubSize;
    tiling.batch = info.batch;
    tiling.qSeqlen = info.qSeqlen;
    tiling.kvSeqlen = info.kvSeqlen;
    tiling.totalQ = info.totalQ;   // B * S
    tiling.totalKv = info.totalKv; // B * S
    tiling.qHeadNum = info.qHeadNum;
    tiling.kvHeadNum = info.kvHeadNum;
    tiling.groupSize = info.qHeadNum / info.kvHeadNum;
    tiling.qkHeadDim = info.qkHeadDim;
    tiling.vHeadDim = info.vHeadDim;
    tiling.scaleValue = info.scaleValue;
    tiling.softcapValue = info.softcapValue;

    tiling.qTile = 128;
    tiling.kvTile = 128;

    tiling.usedCoreNum = info.aicNum; // TODO: 是否需要和实际task取min

    // Deterministic knobs are only meaningful when the deterministic path is
    // enabled; keep them zeroed otherwise so the legacy layout is untouched.
    tiling.dqPostAbsorb = info.deterministic ? info.dqPostAbsorb : 0;
    tiling.dqVecNum = info.deterministic ? info.dqVecNum : 0;
    tiling.dkVecNum = info.deterministic ? info.dkVecNum : 0;
    tiling.dvVecNum = info.deterministic ? info.dvVecNum : 0;

    const uint64_t dAlign = RoundUpU64(info.qkHeadDim, FP32_ROW_ALIGN);
    const uint64_t dvAlign = RoundUpU64(info.vHeadDim, FP32_ROW_ALIGN);

    if (!info.deterministic) {
        // Legacy layout, byte-identical to the non-deterministic path.
        uint64_t dqWsSize = tiling.totalQ * tiling.qHeadNum * tiling.qkHeadDim * FP32_BYTES;
        uint64_t dkWsSize = tiling.totalKv * tiling.kvHeadNum * tiling.qkHeadDim * FP32_BYTES;
        uint64_t dvWsSize = tiling.totalKv * tiling.kvHeadNum * tiling.vHeadDim * FP32_BYTES;
        uint64_t deltaWsSize = tiling.totalQ * tiling.qHeadNum * 8;

        tiling.dqOffset = 0;
        tiling.dkOffset = tiling.dqOffset + dqWsSize;
        tiling.dvOffset = tiling.dkOffset + dkWsSize;
        tiling.deltaOffset = tiling.dvOffset + dvWsSize;
        tiling.workspaceSize = tiling.deltaOffset + deltaWsSize;
        return 0;
    }

    // ------------------------------------------------------------------
    // BN2S2 deterministic schedule (opst arch35 column-private accumulation).
    // ------------------------------------------------------------------
    if (tiling.detSchedule == static_cast<uint32_t>(DetSchedule::BN2S2)) {
        if (info.layout == Layout::TND) {
            // Ragged TND.  MHA uses the column-private swizzle (opst
            // CalTNDDenseSwizzleIndex, per-batch round prefix); GQA uses
            // opst's flat blocked partition (CalTNDDenseIndex !IS_N_EQUAL,
            // cumulative-area prefix) with shared dk/dv workspaces because a
            // column may straddle two lanes' slices.  Causal reuses the same
            // dense schedules with the causal mask (masked blocks add exact
            // zeros), like the rectangular BSND causal path.
            if (info.batch + 1 > TND_SWIZZLE_PREFIX_NUM ||
                info.actualSeqQ == nullptr || info.actualSeqKv == nullptr) {
                fprintf(stderr,
                    "FAG950 det bn2s2 tnd: unsupported TND shape (B=%llu)\n",
                    (unsigned long long)info.batch);
                return -1;
            }
            const bool gqa = tiling.groupSize != 1;
            // Causal TND with per-batch sq == sk: opst's left-up causal schedule
            // (CalTNDCausalIndex) then covers exactly the same block triangle as
            // the right-down element mask, so it can replace dense+mask without
            // changing semantics (the mask epilogue still handles the diagonal
            // tiles).  Any batch with sq != sk keeps the dense schedule + mask,
            // which is the only correct choice under right-down alignment.
            if (info.maskType == MaskType::CAUSAL && !gqa) {
                bool allSquare = true;
                for (uint32_t b = 0; b < info.batch; ++b) {
                    if (info.actualSeqQ[b] != info.actualSeqKv[b]) {
                        allSquare = false;
                        break;
                    }
                }
                if (allSquare) {
                    const fag_det_host::TndCausalParams tc =
                        fag_det_host::ComputeTndCausalParams(
                            static_cast<int64_t>(info.batch), info.actualSeqQ,
                            info.actualSeqKv,
                            static_cast<int64_t>(info.kvHeadNum),
                            static_cast<int64_t>(info.aicNum), tiling.qTile,
                            tiling.kvTile);
                    if (tc.supported) {
                        for (int64_t b = 0;
                             b <= static_cast<int64_t>(info.batch) + 2 &&
                             b < static_cast<int64_t>(TND_SWIZZLE_PREFIX_NUM);
                             ++b) {
                            tiling.tndPrefix[b] = tc.p0[b];
                        }
                        for (int64_t b = 0;
                             b <= static_cast<int64_t>(info.batch) + 1 &&
                             b < static_cast<int64_t>(TND_SWIZZLE_PREFIX_NUM);
                             ++b) {
                            tiling.tndCausalP1[b] = tc.p1[b];
                            tiling.tndCausalP2[b] = tc.p2[b];
                        }
                        tiling.detKind = fag_det_host::KIND_TND_CAUSAL;
                        tiling.detColumnRounds = 0;
                        tiling.detBufNum = 1;
                        // Columns may be covered by different lanes in
                        // disjoint rounds, so dk/dv must be shared across lanes
                        // (round barrier supplies the deterministic order).
                        tiling.detPrivDkv = 0;
                        tiling.detMaxRound =
                            static_cast<uint64_t>(tc.maxRound);
                        tiling.dqPostAbsorb = 0;
                        tiling.continuousBlockNum = 1;
                        tiling.dqVecNum = 0;
                        tiling.dkVecNum = 0;
                        tiling.dvVecNum = 0;
                        const uint64_t dqWsSize =
                            tiling.totalQ * tiling.qHeadNum * dAlign *
                            FP32_BYTES;
                        const uint64_t deltaWsSize =
                            tiling.totalQ * tiling.qHeadNum * 8;
                        tiling.dqOffset = MULTI_CORE_SYNC_BYTES;
                        const uint64_t dkWsSize = tiling.totalKv *
                            tiling.kvHeadNum * dAlign * FP32_BYTES;
                        const uint64_t dvWsSize = tiling.totalKv *
                            tiling.kvHeadNum * dvAlign * FP32_BYTES;
                        tiling.dkOffset = tiling.dqOffset +
                            RoundUpU64(dqWsSize, GM_ALIGNMENT);
                        tiling.dvOffset = tiling.dkOffset +
                            RoundUpU64(dkWsSize, GM_ALIGNMENT);
                        tiling.deltaOffset = tiling.dvOffset +
                            RoundUpU64(dvWsSize, GM_ALIGNMENT);
                        tiling.workspaceSize = tiling.deltaOffset +
                            RoundUpU64(deltaWsSize, GM_ALIGNMENT);
                        tiling.dqDetOffset = 0;
                        tiling.dkDetOffset = 0;
                        tiling.dvDetOffset = 0;
                        tiling.dkPrivOffset = 0;
                        tiling.dvPrivOffset = 0;
                        return 0;
                    }
                }
            }
            // MHA swizzle requires s1Outer >= min(k, s2Outer) for every batch
            // (intra-round dq uniqueness).  Ragged MHA shapes that violate it
            // take the same flat blocked partition as GQA with the shared
            // dk/dv workspaces (a column may straddle two lanes' slices).
            const bool flat = gqa || !fag_det_host::TndDenseSafe(
                static_cast<int64_t>(info.batch), info.actualSeqQ,
                info.actualSeqKv, static_cast<int64_t>(info.kvHeadNum),
                static_cast<int64_t>(info.aicNum), tiling.qTile,
                tiling.kvTile);
            int64_t prefix[TND_SWIZZLE_PREFIX_NUM] = {0};
            int64_t maxRound = 0;
            if (flat) {
                int64_t s1Max = 0;
                int64_t s2Max = 0;
                for (int64_t b = 0; b < static_cast<int64_t>(info.batch); ++b) {
                    const int64_t s1Outer = (info.actualSeqQ[b] + tiling.qTile - 1) /
                        tiling.qTile;
                    const int64_t s2Outer = (info.actualSeqKv[b] + tiling.kvTile - 1) /
                        tiling.kvTile;
                    if (s1Outer <= 0 || s2Outer <= 0) {
                        fprintf(stderr, "FAG950 det bn2s2 tnd: empty sequence\n");
                        return -1;
                    }
                    prefix[b + 1] = prefix[b] + s1Outer * s2Outer;
                    s1Max = s1Outer > s1Max ? s1Outer : s1Max;
                    s2Max = s2Outer > s2Max ? s2Outer : s2Max;
                }
                const int64_t k = static_cast<int64_t>(info.aicNum);
                const int64_t g = static_cast<int64_t>(tiling.groupSize);
                const int64_t total = prefix[info.batch] *
                    static_cast<int64_t>(info.kvHeadNum) * g;
                maxRound = static_cast<int64_t>(
                    std::max({CeilDivU64(total, k), static_cast<uint64_t>(s1Max * g),
                              static_cast<uint64_t>(s2Max)}));
            } else {
                maxRound = fag_det_host::TndDensePrefix(
                    static_cast<int64_t>(info.batch), info.actualSeqQ,
                    info.actualSeqKv,
                    static_cast<int64_t>(info.kvHeadNum), // g == 1
                    static_cast<int64_t>(info.aicNum), tiling.qTile,
                    tiling.kvTile, prefix);
            }
            if (maxRound <= 0) {
                fprintf(stderr, "FAG950 det bn2s2 tnd: empty schedule\n");
                return -1;
            }
            for (uint32_t b = 0; b <= info.batch && b < TND_SWIZZLE_PREFIX_NUM;
                 ++b) {
                tiling.tndPrefix[b] = prefix[b];
            }
            tiling.detKind = flat ? fag_det_host::KIND_TND_GQA_DENSE
                                  : fag_det_host::KIND_TND_DENSE;
            tiling.detColumnRounds = 0;  // column end uses coordinate comparison
            tiling.detBufNum = 1;
            tiling.detPrivDkv = flat ? 0U : 1U;
            tiling.detMaxRound = static_cast<uint64_t>(maxRound);
            tiling.dqPostAbsorb = 0;
            tiling.continuousBlockNum = 1;
            tiling.dqVecNum = 0;
            tiling.dkVecNum = 0;
            tiling.dvVecNum = 0;

            const uint64_t dqWsSize =
                tiling.totalQ * tiling.qHeadNum * dAlign * FP32_BYTES;
            const uint64_t deltaWsSize = tiling.totalQ * tiling.qHeadNum * 8;
            tiling.dqOffset = MULTI_CORE_SYNC_BYTES;
            if (flat) {
                // Shared dk/dv workspaces (same as the BSND GQA path).
                const uint64_t dkWsSize = tiling.totalKv * tiling.kvHeadNum *
                    dAlign * FP32_BYTES;
                const uint64_t dvWsSize = tiling.totalKv * tiling.kvHeadNum *
                    dvAlign * FP32_BYTES;
                tiling.dkOffset =
                    tiling.dqOffset + RoundUpU64(dqWsSize, GM_ALIGNMENT);
                tiling.dvOffset =
                    tiling.dkOffset + RoundUpU64(dkWsSize, GM_ALIGNMENT);
                tiling.deltaOffset =
                    tiling.dvOffset + RoundUpU64(dvWsSize, GM_ALIGNMENT);
                tiling.workspaceSize =
                    tiling.deltaOffset + RoundUpU64(deltaWsSize, GM_ALIGNMENT);
                tiling.dqDetOffset = 0;
                tiling.dkDetOffset = 0;
                tiling.dvDetOffset = 0;
                tiling.dkPrivOffset = 0;
                tiling.dvPrivOffset = 0;
                return 0;
            }
            const uint64_t dkPrivSize = static_cast<uint64_t>(info.aicNum) *
                tiling.kvTile * dAlign * FP32_BYTES;
            const uint64_t dvPrivSize = static_cast<uint64_t>(info.aicNum) *
                tiling.kvTile * dvAlign * FP32_BYTES;
            tiling.deltaOffset =
                tiling.dqOffset + RoundUpU64(dqWsSize, GM_ALIGNMENT);
            tiling.dkPrivOffset =
                tiling.deltaOffset + RoundUpU64(deltaWsSize, GM_ALIGNMENT);
            tiling.dvPrivOffset =
                tiling.dkPrivOffset + RoundUpU64(dkPrivSize, GM_ALIGNMENT);
            tiling.workspaceSize =
                tiling.dvPrivOffset + RoundUpU64(dvPrivSize, GM_ALIGNMENT);
            tiling.dkOffset = 0;
            tiling.dvOffset = 0;
            tiling.dqDetOffset = 0;
            tiling.dkDetOffset = 0;
            tiling.dvDetOffset = 0;
            return CheckBn2s2CastUb(tiling, info);
        }
        if (info.layout != Layout::BSND) {
            fprintf(stderr, "FAG950 det bn2s2: only BSND/TND are supported\n");
            return -1;
        }
        const int64_t bh =
            static_cast<int64_t>(info.batch) *
            static_cast<int64_t>(info.kvHeadNum);
        const int64_t m = (static_cast<int64_t>(info.qSeqlen) +
                           static_cast<int64_t>(tiling.qTile) - 1) /
            static_cast<int64_t>(tiling.qTile);
        const int64_t n = (static_cast<int64_t>(info.kvSeqlen) +
                           static_cast<int64_t>(tiling.kvTile) - 1) /
            static_cast<int64_t>(tiling.kvTile);
        const int64_t g = static_cast<int64_t>(tiling.groupSize);
        const int64_t k = static_cast<int64_t>(info.aicNum);

        const bool causal = info.maskType == MaskType::CAUSAL;
        const fag_det_host::Selection sel =
            fag_det_host::SelectSchedule(causal, bh, m, n, g, k);
        if (!sel.supported) {
            fprintf(stderr,
                "FAG950 det bn2s2: no schedule for causal=%d B=%llu N2=%llu "
                "m=%lld n=%lld g=%lld k=%d\n",
                causal ? 1 : 0, (unsigned long long)info.batch,
                (unsigned long long)info.kvHeadNum, (long long)m,
                (long long)n, (long long)g, info.aicNum);
            return -1;
        }
        tiling.detKind = sel.kind;
        tiling.detColumnRounds = static_cast<uint32_t>(
            fag_det_host::ScheduleColumnRounds(sel.kind, m, n));
        tiling.detBufNum = fag_det_host::ScheduleBufNum(sel.kind);
        // GQA shares one dk/dv column between the g query heads, so the
        // per-core private buffer + early cast only applies to MHA (g == 1);
        // GQA keeps the ordered atomic accumulation in the full workspaces.
        tiling.detPrivDkv = (g == 1) ? 1U : 0U;
        tiling.detMaxRound = static_cast<uint64_t>(
            fag_det_host::ScheduleMaxRound(sel.kind, bh, m, n, g, k));
        tiling.dqPostAbsorb = 0;
        // One scheduled task per cube core per round; the paired AIVs are
        // driven by the existing V1/V2 pipeline plus the per-column dk/dv
        // cast, and the VecDTM group split is unused.
        tiling.continuousBlockNum = 1;
        tiling.dqVecNum = 0;
        tiling.dkVecNum = 0;
        tiling.dvVecNum = 0;

        const uint64_t dqWsSize =
            tiling.totalQ * tiling.qHeadNum * dAlign * FP32_BYTES;
        const uint64_t deltaWsSize = tiling.totalQ * tiling.qHeadNum * 8;
        if (!tiling.detPrivDkv) {
            // Stage-1 layout (GQA): dq/dk/dv use the full fp32 workspaces.
            const uint64_t dkWsSize =
                tiling.totalKv * tiling.kvHeadNum * dAlign * FP32_BYTES;
            const uint64_t dvWsSize =
                tiling.totalKv * tiling.kvHeadNum * dvAlign * FP32_BYTES;
            tiling.dqOffset = MULTI_CORE_SYNC_BYTES;
            tiling.dkOffset = tiling.dqOffset + RoundUpU64(dqWsSize, GM_ALIGNMENT);
            tiling.dvOffset = tiling.dkOffset + RoundUpU64(dkWsSize, GM_ALIGNMENT);
            tiling.deltaOffset = tiling.dvOffset + RoundUpU64(dvWsSize, GM_ALIGNMENT);
            tiling.workspaceSize =
                tiling.deltaOffset + RoundUpU64(deltaWsSize, GM_ALIGNMENT);
            tiling.dqDetOffset = 0;
            tiling.dkDetOffset = 0;
            tiling.dvDetOffset = 0;
            tiling.dkPrivOffset = 0;
            tiling.dvPrivOffset = 0;
            return 0;
        }

        // Stage-2 layout (MHA): [sync 64KB][dq fp32 full][delta][dk priv][dv
        // priv].  dq still uses the full fp32 workspace (ordered atomic
        // accumulation); dk/dv accumulate in per-core private regions and are
        // converted to the bf16 outputs as soon as a column completes.
        const uint64_t dkPrivSize = static_cast<uint64_t>(info.aicNum) *
            tiling.detBufNum * tiling.kvTile * dAlign * FP32_BYTES;
        const uint64_t dvPrivSize = static_cast<uint64_t>(info.aicNum) *
            tiling.detBufNum * tiling.kvTile * dvAlign * FP32_BYTES;

        tiling.dqOffset = MULTI_CORE_SYNC_BYTES;
        tiling.deltaOffset = tiling.dqOffset + RoundUpU64(dqWsSize, GM_ALIGNMENT);
        tiling.dkPrivOffset = tiling.deltaOffset + RoundUpU64(deltaWsSize, GM_ALIGNMENT);
        tiling.dvPrivOffset = tiling.dkPrivOffset + RoundUpU64(dkPrivSize, GM_ALIGNMENT);
        tiling.workspaceSize = tiling.dvPrivOffset + RoundUpU64(dvPrivSize, GM_ALIGNMENT);
        tiling.dkOffset = 0;
        tiling.dvOffset = 0;
        tiling.dqDetOffset = 0;
        tiling.dkDetOffset = 0;
        tiling.dvDetOffset = 0;

        return CheckBn2s2CastUb(tiling, info);
    }

    // Deterministic layout:
    //   [sync 64KB][dq][dk][dv][delta][dqDet][dkDet][dvDet]
    // dq accumulates into a single rolling tile when dqPostAbsorb=1, into the
    // full S1*N1 region otherwise. det slots reserve two complete issue
    // rounds (aicNum * continuousBlockNum tiles per round) per gradient.
    // Alternating the round bank lets VecDTM(r) read one bank while C345(r+1)
    // writes the other, without a done-counter gate on slot reuse.

    // UB budget check.  The main pipeline owns two ping/pong halves starting
    // at UB+0 (formula mirrors fag_kernel.cpp Init); the VecDTM epilogue owns
    // a dedicated tail region (mirrors fag_epilogue_deterministic_add.hpp
    // Init).  Borrowing is NOT allowed: v2 overlaps VecDTM(r) with the next
    // round's C12 SPLIT_M fixpipe, which writes the ping/pong halves.
    // The VecDTM group mapping assumes the three vec groups all exist as
    // physical AIVs.
    if (info.dqVecNum == 0 || info.dkVecNum == 0 || info.dvVecNum == 0 ||
        info.dqVecNum + info.dkVecNum + info.dvVecNum > info.aivNum) {
        fprintf(stderr,
            "FAG950 det bwd: bad vec groups %u/%u/%u (aivNum %u)\n",
            info.dqVecNum, info.dkVecNum, info.dvVecNum, info.aivNum);
        return -1;
    }
    const uint64_t rowsPerSub = RoundUpU64(tiling.qTile, 16) / 2;
    const uint64_t mmResBytes = rowsPerSub * tiling.kvTile * FP32_BYTES;
    const uint64_t attenMaskBytes = rowsPerSub * tiling.kvTile;
    const uint64_t lseBytes =
        RoundUpU64(rowsPerSub, 8) * 8 * FP32_BYTES;
    const uint64_t pBytes = (rowsPerSub + 1) * tiling.kvTile * 2;
    const uint64_t deltaBytes = rowsPerSub * 8 * FP32_BYTES;
    const uint64_t lseOff = RoundUpU64(2 * mmResBytes + attenMaskBytes, 32);
    const uint64_t pOff = RoundUpU64(lseOff + lseBytes, 32);
    const uint64_t dSOff = RoundUpU64(pOff + pBytes, 32);
    const uint64_t deltaOff = RoundUpU64(dSOff + pBytes, 32);
    const uint64_t halfUb = RoundUpU64(deltaOff + deltaBytes, 32);
    const uint64_t pipelineUb = 2 * halfUb;  // TASK_PINGPONG
    const uint64_t rowCapDq = CeilDivU64(tiling.qTile, info.dqVecNum);
    const uint64_t rowCapDk = CeilDivU64(tiling.kvTile, info.dkVecNum);
    const uint64_t rowCapDv = CeilDivU64(tiling.kvTile, info.dvVecNum);
    uint64_t rowCap = rowCapDq;
    if (rowCapDk > rowCap) { rowCap = rowCapDk; }
    if (rowCapDv > rowCap) { rowCap = rowCapDv; }
    const uint64_t colCap = dAlign > dvAlign ? dAlign : dvAlign;
    const uint64_t accUbBytes =
        RoundUpU64(rowCap * colCap * FP32_BYTES, 32);
    const uint64_t castUbBytes =
        RoundUpU64(rowCap * info.qkHeadDim * 2, 32);
    const uint64_t detAddUb = 2 * accUbBytes + castUbBytes;
    if (pipelineUb + detAddUb > info.ubSize) {
        fprintf(stderr,
            "FAG950 det bwd: UB too small: pipeline %llu + VecDTM %llu > %llu\n",
            (unsigned long long)pipelineUb, (unsigned long long)detAddUb,
            (unsigned long long)info.ubSize);
        return -1;
    }

    const uint64_t dqWsSize = tiling.dqPostAbsorb
        ? static_cast<uint64_t>(tiling.qTile) * dAlign * FP32_BYTES
        : tiling.totalQ * tiling.qHeadNum * dAlign * FP32_BYTES;
    const uint64_t dkWsSize = tiling.totalKv * tiling.kvHeadNum * dAlign * FP32_BYTES;
    const uint64_t dvWsSize = tiling.totalKv * tiling.kvHeadNum * dvAlign * FP32_BYTES;
    const uint64_t deltaWsSize = tiling.totalQ * tiling.qHeadNum * 8;

    constexpr uint64_t detSlotBanks = 2;
    const uint64_t slotNum = detSlotBanks *
        static_cast<uint64_t>(info.aicNum) * info.continuousBlockNum;
    const uint64_t dqDetSize = RoundUpU64(
        slotNum * tiling.qTile * dAlign * FP32_BYTES, GM_ALIGNMENT);
    const uint64_t dkDetSize = RoundUpU64(
        slotNum * tiling.kvTile * dAlign * FP32_BYTES, GM_ALIGNMENT);
    const uint64_t dvDetSize = RoundUpU64(
        slotNum * tiling.kvTile * dvAlign * FP32_BYTES, GM_ALIGNMENT);

    tiling.dqOffset = MULTI_CORE_SYNC_BYTES;
    tiling.dkOffset = tiling.dqOffset + RoundUpU64(dqWsSize, GM_ALIGNMENT);
    tiling.dvOffset = tiling.dkOffset + RoundUpU64(dkWsSize, GM_ALIGNMENT);
    tiling.deltaOffset = tiling.dvOffset + RoundUpU64(dvWsSize, GM_ALIGNMENT);
    tiling.dqDetOffset = tiling.deltaOffset + RoundUpU64(deltaWsSize, GM_ALIGNMENT);
    tiling.dkDetOffset = tiling.dqDetOffset + dqDetSize;
    tiling.dvDetOffset = tiling.dkDetOffset + dkDetSize;
    tiling.workspaceSize = tiling.dvDetOffset + dvDetSize;
    return 0;
}

}  // namespace FAGTiling950
