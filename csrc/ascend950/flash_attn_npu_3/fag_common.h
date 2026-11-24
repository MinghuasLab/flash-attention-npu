// Copyright (c) 2026, Minghua Shen.

// Host/device tiling ABI for the Ascend950 FlashAttention v3 backward path.
// Keep this structure POD-only.  It is copied byte-for-byte from a CPU tensor
// to device global memory and will be consumed by the future arch35 FAG
// kernel.

#ifndef FLASH_ATTN_NPU_ARCH35_V3_FAG_COMMON_H
#define FLASH_ATTN_NPU_ARCH35_V3_FAG_COMMON_H

#include "fag_tiling_data.h"

#include "catlass/catlass.hpp"
#include "kernel_operator.h"

CATLASS_DEVICE
bool IsFagPlanReady(GM_ADDR tiling)
{
    const auto* plan = reinterpret_cast<const __gm__ FAGTiling950::FAGTilingData*>(tiling);
    return plan->planStatus == FAGTiling950::FAGPlanStatus::READY;
}

struct FAGBlockInfo {
    uint64_t blockId = 0;
    uint64_t taskId = 0;
    uint32_t issueRound = 0;
    uint32_t issueLane = 0;

    uint32_t batchIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t groupIdx = 0;
    uint32_t s1BlockIdx = 0;
    uint32_t s2BlockIdx = 0;

    uint32_t s1Start = 0;
    uint32_t s2Start = 0;
    uint32_t curBatchS1 = 0;
    uint32_t curBatchS2 = 0;
    uint32_t s1Extend = 0;
    uint32_t s2Extend = 0;
    uint32_t firstHalfRealS1 = 0;

    uint64_t totalS1Start = 0;
    uint64_t totalS2Start = 0;
    uint64_t qOffset = 0;
    uint64_t kOffset = 0;
    uint64_t vOffset = 0;
    uint64_t doutOffset = 0;

    // KV-group residency / local dk,dv accumulate control.
    // One AIC owns a full (batch, n2, s2Block) group so K/V stay in L1 and
    // dk/dv can accumulate in L0C until the last s1 of the group.
    bool loadKv = true;
    bool flushDkv = true;
    bool initDkv = true;
    // shortQ: first S2 tile may overwrite zeroed dq workspace (no atomic).
    bool atomicDq = true;

    // BN2S2 only: fold accumulator index of this task and, for every fold
    // buffer of the current column, the real (batch, n2, s2) column that the
    // private dk/dv region accumulates.  Dense schedules use buffer 0 only.
    uint32_t detParity = 0;
    uint32_t detBufNum = 1;
    uint32_t detFoldValid[2] = {0, 0};
    uint32_t detFoldN2[2] = {0, 0};
    uint64_t detFoldS2Start[2] = {0, 0};
    uint32_t detFoldS2Extend[2] = {0, 0};
};

// Device-kernel argument bundle. Keep it in this header together with the
// tiling ABI so the arch35 FAG host and device paths share one ABI definition.
struct FAGKernelParams {
    GM_ADDR dout;
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR out;
    GM_ADDR attenMask;
    GM_ADDR softmaxLse;
    GM_ADDR cuSeqQlen;
    GM_ADDR cuSeqKvlen;
    // Per-batch *used* lengths (seqused); nullptr when absent, cu arrays stay region.
    GM_ADDR seqUsedQ;
    GM_ADDR seqUsedKv;
    GM_ADDR dq;
    GM_ADDR dk;
    GM_ADDR dv;
    GM_ADDR workspace;
    GM_ADDR tiling;

    CATLASS_DEVICE
    FAGKernelParams() = default;

    CATLASS_DEVICE
    FAGKernelParams(GM_ADDR dout_, GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR out_, GM_ADDR attenMask_,
                    GM_ADDR softmaxLse_, GM_ADDR cuSeqQlen_, GM_ADDR cuSeqKvlen_, GM_ADDR seqUsedQ_, GM_ADDR seqUsedKv_,
                    GM_ADDR dq_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR workspace_, GM_ADDR tiling_)
        : dout(dout_), q(q_), k(k_), v(v_), out(out_), attenMask(attenMask_), softmaxLse(softmaxLse_),
          cuSeqQlen(cuSeqQlen_), cuSeqKvlen(cuSeqKvlen_), seqUsedQ(seqUsedQ_), seqUsedKv(seqUsedKv_), dq(dq_), dk(dk_),
          dv(dv_), workspace(workspace_), tiling(tiling_)
    {}
};

#endif
