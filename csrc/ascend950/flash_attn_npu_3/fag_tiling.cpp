/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend950-specific host tiling and workspace calculation for FA v3 bwd.
 */

#include "fag_common.h"

#include <algorithm>
#include <limits>

namespace FAGTiling950 {
namespace {

constexpr uint64_t FP32_BYTES = sizeof(float);

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
    tiling.aicNum = info.aicNum;
    tiling.aivNum = info.aivNum;
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

}  // namespace FAGTiling950
