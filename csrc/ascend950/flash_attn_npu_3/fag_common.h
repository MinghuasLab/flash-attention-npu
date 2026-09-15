/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Host/device tiling ABI for the Ascend950 FlashAttention v3 backward path.
 *
 * Keep this structure POD-only.  It is copied byte-for-byte from a CPU tensor
 * to device global memory and will be consumed by the future arch35 FAG
 * kernel.
 */

#ifndef FLASH_ATTN_NPU_ARCH35_V3_FAG_COMMON_H
#define FLASH_ATTN_NPU_ARCH35_V3_FAG_COMMON_H

#include <cstdint>
#include <type_traits>

#include "catlass/catlass.hpp"
#include "kernel_operator.h"

namespace FAGTiling950 {

constexpr uint64_t GM_ALIGNMENT = 512;
constexpr uint64_t MULTI_CORE_SYNC_BYTES = 64 * 1024;
constexpr uint32_t SOFTMAX_REDUCE_FLOATS = 8;
constexpr uint32_t DEFAULT_CONTINUOUS_BLOCK_NUM = 2;

enum class Layout : uint32_t {
    BSND = 0,
    TND = 1,
};

enum class MaskType : uint32_t {
    NO_MASK = 0,
    CAUSAL = 1,
};

struct FAGInfo {
    Layout layout = Layout::BSND;
    MaskType maskType = MaskType::NO_MASK;
    uint32_t deterministic = 0;

    uint64_t batch = 0;
    uint64_t qSeqlen = 0;
    uint64_t kvSeqlen = 0;
    uint64_t totalQ = 0;
    uint64_t totalKv = 0;
    uint64_t qHeadNum = 0;
    uint64_t kvHeadNum = 0;
    uint64_t qkHeadDim = 0;
    uint64_t vHeadDim = 0;

    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint32_t continuousBlockNum = DEFAULT_CONTINUOUS_BLOCK_NUM;
    uint64_t ubSize = 0;
    float scaleValue = 1.0f;
    float softcapValue = 0.0f;
};

struct FAGTilingData {
    uint32_t layout = 0;
    uint32_t maskType = 0;
    uint32_t deterministic = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint32_t usedCoreNum = 0;
    uint32_t continuousBlockNum = DEFAULT_CONTINUOUS_BLOCK_NUM;
    uint32_t reserved0 = 0;

    uint64_t ubSize = 0;
    uint64_t batch = 0;
    uint64_t qSeqlen = 0;
    uint64_t kvSeqlen = 0;
    uint64_t totalQ = 0;
    uint64_t totalKv = 0;
    uint64_t qHeadNum = 0;
    uint64_t kvHeadNum = 0;
    uint64_t groupSize = 0;
    uint64_t qkHeadDim = 0;
    uint64_t vHeadDim = 0;
    float scaleValue = 1.0f;
    float softcapValue = 0.0f;

    uint32_t qTile = 0;
    uint32_t kvTile = 0;

    uint64_t dqOffset = 0;
    uint64_t dkOffset = 0;
    uint64_t dvOffset = 0;
    uint64_t deltaOffset = 0;
    uint64_t workspaceSize = 0;
};

static_assert(std::is_standard_layout_v<FAGTilingData>,
              "FAGTilingData must have a stable host/device layout");
static_assert(std::is_trivially_copyable_v<FAGTilingData>,
              "FAGTilingData must be byte-copyable to device");

int64_t GetFAGTilingParam(const FAGInfo &info, FAGTilingData &tiling);

}  // namespace FAGTiling950

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
    GM_ADDR dq;
    GM_ADDR dk;
    GM_ADDR dv;
    GM_ADDR workspace;
    GM_ADDR tiling;

    CATLASS_DEVICE
    FAGKernelParams() = default;

    CATLASS_DEVICE
    FAGKernelParams(
        GM_ADDR dout_,
        GM_ADDR q_,
        GM_ADDR k_,
        GM_ADDR v_,
        GM_ADDR out_,
        GM_ADDR attenMask_,
        GM_ADDR softmaxLse_,
        GM_ADDR cuSeqQlen_,
        GM_ADDR cuSeqKvlen_,
        GM_ADDR dq_,
        GM_ADDR dk_,
        GM_ADDR dv_,
        GM_ADDR workspace_,
        GM_ADDR tiling_)
        : dout(dout_),
          q(q_),
          k(k_),
          v(v_),
          out(out_),
          attenMask(attenMask_),
          softmaxLse(softmaxLse_),
          cuSeqQlen(cuSeqQlen_),
          cuSeqKvlen(cuSeqKvlen_),
          dq(dq_),
          dk(dk_),
          dv(dv_),
          workspace(workspace_),
          tiling(tiling_)
    {
    }
};

#endif
