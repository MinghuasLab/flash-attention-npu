/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026
 */

#ifndef FA_METADATA_ARGS_H
#define FA_METADATA_ARGS_H

#include <cstdint>
#include "tilingdata.h"

namespace fa_metadata {
constexpr uint32_t MASK_DIM = 2048;
constexpr uint64_t MASK_BYTES = static_cast<uint64_t>(MASK_DIM) * MASK_DIM;

inline uint64_t TilingOffset(bool causal) { return causal ? MASK_BYTES : 0; }
inline uint64_t MetadataBytes(bool causal) { return TilingOffset(causal) + sizeof(FAInferTilingData); }

constexpr uint64_t WORKSPACE_BLOCK_SIZE_DB = static_cast<uint64_t>(128) * 512;
constexpr uint64_t PRELAUNCH_NUM = 3;
inline uint64_t Mm1OutSize(uint64_t blockDim) { return blockDim * WORKSPACE_BLOCK_SIZE_DB * 4 * PRELAUNCH_NUM; }
inline uint64_t SmOnlineOutSize(uint64_t blockDim) { return blockDim * WORKSPACE_BLOCK_SIZE_DB * 2 * PRELAUNCH_NUM; }
inline uint64_t Mm2OutSize(uint64_t blockDim) { return blockDim * WORKSPACE_BLOCK_SIZE_DB * 4 * PRELAUNCH_NUM; }
inline uint64_t UpdateOutSize(uint64_t blockDim) { return blockDim * WORKSPACE_BLOCK_SIZE_DB * 4 * PRELAUNCH_NUM; }
inline uint64_t WorkSpaceSize(uint64_t blockDim)
{
    return Mm1OutSize(blockDim) + SmOnlineOutSize(blockDim) + Mm2OutSize(blockDim) + UpdateOutSize(blockDim);
}
}

struct FAMetadataArgs {
    uint64_t cuSeqlensQAddr;
    uint64_t seqlensKAddr;
    uint64_t metaOutAddr;
    uint32_t batch;
    uint32_t numHeads;
    uint32_t numHeadsK;
    uint32_t embeddingSize;
    uint32_t embeddingSizeV;
    uint32_t numBlocks;
    uint32_t blockSize;
    uint32_t maxNumBlocksPerBatch;
    uint32_t maxQSeqlen;
    uint32_t maskType;
    uint32_t blockDim;
    uint32_t isVarlen;
    uint32_t isVarlenKv;
    uint32_t pagedKV;
    uint32_t numSplits;
    float scaleValue;
};

#endif
