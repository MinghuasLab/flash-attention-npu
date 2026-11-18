/** Copyright (c) 2026 Huawei Technologies Co., Ltd. CANN Open Software License Agreement Version 2.0. */
#ifndef FA950_FWD_TILING_POLICY_H
#define FA950_FWD_TILING_POLICY_H
#include "fwd_resource_plan.h"
#include "tilingdata.h"

namespace fa950_plan {
// Host/AICPU policy: fully masked query rows do not provide Cube work. With
// bottom-right alignment, a finite right bound makes the first Q-K-right rows
// empty. Unlimited-right windows and unmasked attention keep every query row.
inline uint32_t ActiveQueryRows(uint32_t q, uint32_t kv, bool hasMask, int64_t right)
{
    if (kv == 0U)
        return 0U;
    if (!hasMask || right < 0)
        return q;
    const uint64_t rows = uint64_t(kv) + uint64_t(right);
    return rows < q ? static_cast<uint32_t>(rows) : q;
}

// D512 L1 tiling fields, shared by Host tiling and AICPU metadata so the
// device-side address plan always matches. Q/K/V use one slot each, P keeps
// three; QK N is 128 (one KV tile) and K covers the full D.
template <class TilingData>
inline void ApplyD512ResourceFields(TilingData& t)
{
    t.qkL1TileM = 128U;
    t.qkL1TileN = 128U;
    t.qkL1TileKLeft = 512U;
    t.qkL1TileKRight = 512U;
    t.pvL1TileM = 128U;
    t.pvL1TileN = 512U;
    t.pvL1TileKLeft = 128U;
    t.pvL1TileKRight = 128U;
    t.qL1BufNum = 1U;
    t.kL1BufNum = 1U;
    t.vL1BufNum = 1U;
    t.pL1BufNum = 3U;
    t.kvBaseTile = 128U;
}

inline uint32_t SelectQueryTile(uint32_t d, uint32_t dv, uint32_t maxQ, uint64_t fullTasks, uint32_t cores,
                                bool isLocal, int64_t left, int64_t right)
{
    // D512 has one legal profile (Q128); skip every small-input or windowed
    // fallback that would select the legacy fixed UB layout.
    if (IsD512(d, dv)) {
        return 128U;
    }
    // A narrow finite window adds masked KV work when Q doubles. Single PV
    // storage also serializes Fixpipe/rescale: use a conservative four-KV-tile
    // window floor there, versus two tiles when both PV stages fit in UB.
    const uint32_t minWindowTiles = OTmpStages(dv) == 1U ? 4U : 2U;
    if (dv > 128U && isLocal && left >= 0 && right >= 0 && uint64_t(left) + right + 1U < minWindowTiles * 128U)
        return 64U;
    return QueryTileForTasks(d, dv, maxQ, fullTasks, cores);
}
} // namespace fa950_plan
#endif
