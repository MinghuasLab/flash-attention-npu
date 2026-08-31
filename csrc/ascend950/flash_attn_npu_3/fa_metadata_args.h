/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef CSRC_ASCEND950_FLASH_ATTN_NPU_3_FA_METADATA_ARGS_H
#define CSRC_ASCEND950_FLASH_ATTN_NPU_3_FA_METADATA_ARGS_H

#include <cstdint>
#include <type_traits>

#include "tilingdata.h"

static_assert(std::is_trivially_copyable<FAInferTilingData>::value,
              "FAInferTilingData must be trivially copyable to cross the AICPU/device boundary");

namespace fa_metadata {
constexpr uint32_t MASK_DIM = 2048;
constexpr uint64_t MASK_BYTES = static_cast<uint64_t>(MASK_DIM) * MASK_DIM;

constexpr uint64_t WORKSPACE_BLOCK_SIZE_DB = static_cast<uint64_t>(128) * 512;
constexpr uint32_t SIZE_OF_16BIT = 2;
constexpr uint32_t SIZE_OF_32BIT = 4;
constexpr uint32_t PRELAUNCH_NUM = 3;
constexpr uint64_t WS_FLOOR = uint64_t(1024) * 1024 * 32 * 4;  // 128 MiB

// The mask buffer is present whenever the final mask type is not NO_MASK
// (causal or band/SWA); the tiling blob sits right after it, then kvCum.
inline uint64_t TilingOffset(bool hasMask)
{
    return hasMask ? MASK_BYTES : 0;
}

inline uint64_t MetadataBytes(bool hasMask)
{
    return TilingOffset(hasMask) + sizeof(FAInferTilingData);
}

inline uint64_t KvSeqlenOffset(bool hasMask)
{
    return MetadataBytes(hasMask);
}

inline uint64_t KvSeqlenBytes(uint32_t batch)
{
    return (static_cast<uint64_t>(batch) + 1) * sizeof(int32_t);
}

inline uint64_t MetadataBytesWithKv(bool hasMask, uint32_t batch)
{
    return KvSeqlenOffset(hasMask) + KvSeqlenBytes(batch);
}

inline uint64_t Mm1OutSize(uint64_t blockDim)
{
    return blockDim * WORKSPACE_BLOCK_SIZE_DB * SIZE_OF_32BIT * PRELAUNCH_NUM;
}

inline uint64_t SmOnlineOutSize(uint64_t blockDim)
{
    return blockDim * WORKSPACE_BLOCK_SIZE_DB * SIZE_OF_16BIT * PRELAUNCH_NUM;
}

inline uint64_t Mm2OutSize(uint64_t blockDim)
{
    return blockDim * WORKSPACE_BLOCK_SIZE_DB * SIZE_OF_32BIT * PRELAUNCH_NUM;
}

inline uint64_t UpdateOutSize(uint64_t blockDim)
{
    return blockDim * WORKSPACE_BLOCK_SIZE_DB * SIZE_OF_32BIT * PRELAUNCH_NUM;
}

inline uint64_t WorkSpaceSize(uint64_t blockDim)
{
    return Mm1OutSize(blockDim) + SmOnlineOutSize(blockDim) +
           Mm2OutSize(blockDim) + UpdateOutSize(blockDim);
}
}  // namespace fa_metadata

struct FwdMaskDerivation {
    bool is_causal;
    bool is_local;
    int64_t window_left;
    int64_t window_right;
    uint32_t maskType;
};

// Window normalization + mask-type derivation, mirroring the host tiling
// branch in mha_fwd, but bounding the KV side with a host-known upper bound
// (max_seqlen_k / cache capacity) instead of the device-side actual max KV
// seqlen, so it is usable on the scheduler-metadata path where no D2H sync is
// allowed. Both metadata creation and fwd consumption use the same derivation.
inline FwdMaskDerivation DeriveFwdMask(bool causal, int64_t window_left,
                                       int64_t window_right,
                                       int64_t /*max_seqlen_q*/,
                                       int64_t max_seqlen_k_bound)
{
    if (max_seqlen_k_bound > 0 && window_left >= max_seqlen_k_bound) {
        window_left = -1;
    }
    if (max_seqlen_k_bound > 0 && window_right >= max_seqlen_k_bound) {
        window_right = -1;
    }
    if (causal) {
        window_right = 0;
    }
    FwdMaskDerivation derived;
    derived.is_causal = (window_left < 0 && window_right == 0);
    derived.is_local = (window_left >= 0 || window_right >= 0) && !derived.is_causal;
    if (derived.is_local) {
        if (window_left < 0) {
            window_left = max_seqlen_k_bound;
        }
        if (window_right < 0) {
            window_right = max_seqlen_k_bound;
        }
    }
    derived.window_left = window_left;
    derived.window_right = window_right;
    derived.maskType = derived.is_local ? 4u
        : (derived.is_causal ? 1u : 0u);
    return derived;
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
    int32_t windowSizeLeft;
    int32_t windowSizeRight;
    uint32_t blockDim;
    uint32_t isVarlen;
    uint32_t isVarlenKv;
    uint32_t pagedKV;
    float softmaxScale;
};

#endif
