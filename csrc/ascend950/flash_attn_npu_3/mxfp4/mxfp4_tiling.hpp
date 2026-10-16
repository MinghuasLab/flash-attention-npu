/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef MXFP4_TILING_HPP
#define MXFP4_TILING_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mxfp4_tiling_data.h"

namespace optiling {

struct Mxfp4Context {
    int64_t batch = 0;
    int64_t qHeads = 0;
    int64_t kvHeads = 0;
    int64_t maxQ = 0;
    int64_t maxKv = 0;
    int64_t logicalD = 0;
    int64_t blockX = 128;
    int64_t blockY = 128;
    bool sparse = false;
    bool hasQLengths = false;
    bool hasKvLengths = false;
    std::vector<int64_t> qLengths;
    float scaleValue = 0.0f;
    bool cx = false;
    double dstTypeMax = 0.0;
};

class Mxfp4Tiling {
  public:
    Mxfp4Tiling() = default;
    explicit Mxfp4Tiling(const Mxfp4Context& ctx) : ctx_(ctx) {}

    void SetCoreNum(uint32_t blockNum)
    {
        blockNum_ = blockNum;
    }

    uint32_t GetCoreNum() const
    {
        return blockNum_;
    }

    void DoTiling(Mxfp4TilingData& tilingdata)
    {
        FillBasicTilingData(tilingdata);
        FillWorkSpaceTilingData(tilingdata);
    }

  private:
    static int64_t CeilDiv(int64_t a, int64_t b)
    {
        return (a + b - 1) / b;
    }

    void FillBasicTilingData(Mxfp4TilingData& tilingdata)
    {
        const uint32_t qBaseTile = static_cast<uint32_t>(std::min<int64_t>(ctx_.blockX, 128));
        const uint32_t kvBaseTile = 256;
        uint32_t totalTasks = 0;
        uint32_t firstBatchTasks = 0;
        for (int64_t b = 0; b < ctx_.batch; ++b) {
            const int64_t sq = ctx_.hasQLengths ? ctx_.qLengths[static_cast<size_t>(b)] : ctx_.maxQ;
            const uint32_t fullX = static_cast<uint32_t>(sq / ctx_.blockX);
            const uint32_t tailX = static_cast<uint32_t>(sq % ctx_.blockX);
            const uint32_t tiles = fullX * static_cast<uint32_t>(CeilDiv(ctx_.blockX, qBaseTile)) +
                                   (tailX == 0 ? 0 : static_cast<uint32_t>(CeilDiv(tailX, qBaseTile)));
            const uint32_t batchTasks = tiles * static_cast<uint32_t>(ctx_.qHeads);
            totalTasks += batchTasks;
            if (b == 0) {
                firstBatchTasks = batchTasks;
            }
        }

        const uint32_t xBlocks = static_cast<uint32_t>(CeilDiv(ctx_.maxQ, ctx_.blockX));
        const uint32_t yBlocks = static_cast<uint32_t>(CeilDiv(ctx_.maxKv, ctx_.blockY));
        const uint32_t totalMaskRows = static_cast<uint32_t>(ctx_.batch * ctx_.qHeads) * xBlocks;
        const uint32_t aivNum = blockNum_ * 2;
        const uint32_t avgRows = ctx_.sparse ? static_cast<uint32_t>(CeilDiv(totalMaskRows, aivNum)) : 0;

        tilingdata.batch = static_cast<uint32_t>(ctx_.batch);
        tilingdata.numHeads = static_cast<uint32_t>(ctx_.qHeads);
        tilingdata.kvHeads = static_cast<uint32_t>(ctx_.kvHeads);
        tilingdata.embeddingSize = static_cast<uint32_t>(ctx_.logicalD);
        tilingdata.firstBatchTaskNum = firstBatchTasks;
        tilingdata.totalTaskNum = totalTasks;
        tilingdata.coreTaskNum = blockNum_ == 0 ? 0 : totalTasks / blockNum_;
        tilingdata.scaleValue = ctx_.scaleValue;
        tilingdata.blockShapeX = static_cast<uint64_t>(ctx_.blockX);
        tilingdata.blockShapeY = static_cast<uint64_t>(ctx_.blockY);
        tilingdata.maxKvBlockNum = yBlocks;
        tilingdata.maxQBlockNum = xBlocks;
        tilingdata.maxQSeqlen = static_cast<uint32_t>(ctx_.maxQ);
        tilingdata.maxKvSeqlen = static_cast<uint32_t>(ctx_.maxKv);
        tilingdata.useUniformQSeqlen = ctx_.hasQLengths ? 0 : 1;
        tilingdata.useUniformKvSeqlen = ctx_.hasKvLengths ? 0 : 1;
        tilingdata.mask2IdxTileInfo = {xBlocks, yBlocks, avgRows,
                                       ctx_.sparse ? static_cast<uint32_t>(CeilDiv(totalMaskRows, avgRows)) : 0};
        tilingdata.baseTileInfo = {qBaseTile, kvBaseTile};
        if (ctx_.cx) {
            constexpr float kCxDefaultLog2 = 2.584962500721156f;
            tilingdata.log2Cx =
                ctx_.dstTypeMax == 0.0 ? kCxDefaultLog2 : static_cast<float>(std::log2(ctx_.dstTypeMax));
            tilingdata.log2CxCeil = std::ceil(tilingdata.log2Cx);
        }
    }

    void FillWorkSpaceTilingData(Mxfp4TilingData& tilingdata)
    {
        const uint64_t selectIdxSize = ctx_.sparse
                                           ? static_cast<uint64_t>(ctx_.batch * ctx_.qHeads) * tilingdata.maxQBlockNum *
                                                 tilingdata.maxKvBlockNum * sizeof(int32_t)
                                           : 0;
        const uint64_t selectCountSize =
            ctx_.sparse ? static_cast<uint64_t>(ctx_.batch * ctx_.qHeads) * tilingdata.maxQBlockNum * sizeof(int32_t)
                        : 0;
        tilingdata.selectIdxSize = selectIdxSize;
        tilingdata.workSpaceSize = selectIdxSize + selectCountSize;
    }

    Mxfp4Context ctx_;
    uint32_t blockNum_ = 0;
};

} // namespace optiling

#endif // MXFP4_TILING_HPP
