/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* Modified by Minghua Shen, 2026. */

#ifndef MXFP4_KERNEL_UTILS_HPP
#define MXFP4_KERNEL_UTILS_HPP

#include "catlass/numeric_size.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/matrix.hpp"

#include "arch/resource.hpp"

#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "mxfp4_tiling_data.h"
#include "kernel_operator.h"

namespace MXFP4Kernel {

static constexpr uint32_t TILE_GROUP_N = 16;

template <bool Sparse>
struct SparsityPolicy {
    static constexpr bool IS_SPARSE = Sparse;
};

struct TaskInfo {
    uint32_t batchIdx = 0;
    uint32_t qHeadIdx = 0;
    uint32_t kvHeadIdx = 0;
    uint32_t qSTileIdx = 0;
    uint32_t xBlockIdx = 0;
    uint64_t gatheredKvSeqlen = 0;
    uint32_t qsActBaseTile = 0;
    uint32_t qsBaseTile = 0;
    uint32_t kvsBaseTile = 0;
    uint32_t taskTileNum = 0;
    // ===== PV 稀疏 gather 所需（mxfp4）=====
    int64_t gmOffsetV = 0;          // V 数据 batch+head 基址偏移（fp4 元素单位，sizeof=1 即字节）
    int64_t gmOffsetVScale = 0;     // V-scale batch+head 基址偏移（E8M0 字节单位）
    uint32_t gmOffsetSparseIdx = 0; // 本 task 的 sparseIdx 起始 = gmOffsetSparseCount * yBlockNumAligned
    int64_t kvSeqlen = 0;           // 当前 batch 实际 KV 长度
    uint32_t yBlockNumAval = 0;     // CeilDiv(kvSeqlen, blockShapeY)，实际 Y-block 总数
    uint32_t yBlockNumRsvd = 0;     // 本 task 选中 Y-block 数（来自 gSparseCount）
    // ===== QK(Mm1) 所需（mxfp4）=====
    int64_t gmOffsetQ = 0;      // Q 数据偏移（含本 task 的 Q seq 起点，fp4 元素单位）
    int64_t gmOffsetK = 0;      // K 数据 batch+head 基址（seq 由后续稀疏 gather 处理）
    int64_t gmOffsetQScale = 0; // Q-scale 偏移（E8M0 字节；Host 沿 D）
    int64_t gmOffsetKScale = 0; // K-scale batch+head 基址（E8M0 字节；Host 沿 D）
    // O GM 偏移
    int64_t gmOffsetO = 0;
    uint32_t oShapeCol = 0;
    uint32_t qsActBaseTileAlign128 = 0;
    uint32_t qsActBaseTileAlign64 = 0;
    uint32_t qsActBaseTileAlign16 = 0;
    uint32_t qsActBaseTileAlign8 = 0;
};

// CreateTaskInfo 跨调用持久状态：batch 扫描进度 + 各 GM 的 batch 累积偏移。
// 同核 taskIdx (coreTaskId * coreNum + coreIdx) 单调递增，batch 只会前进不会后退，
// 故扫描可从上次结束位置继续，无需每次从 batch 0 重扫。
// curTotalTaskNum 需用 firstBatchTaskNum 初始化（tiling 运行期才知道），在主循环外完成。
struct BatchOffsetInfo {
    uint32_t curBatch = 0;        // 当前扫描到的 batch
    uint32_t preTotalTaskNum = 0; // curBatch 之前所有 batch 的 task 总数
    uint32_t curTotalTaskNum = 0; // 含 curBatch 在内的 task 总数（初始化为 firstBatchTaskNum）
    // TND 下各 GM 的 batch 累积偏移（BNSD/BSND 由 curBatch 直接算，不用这些字段）
    int64_t oBOffset = 0;      // O 输出
    int64_t qBOffset = 0;      // Q 数据
    int64_t qScaleBOffset = 0; // Q-scale（字节）
    int64_t vBOffset = 0;      // V(=K) 数据
    int64_t vScaleBOffset = 0; // V-scale（字节）
    int64_t kScaleBOffset = 0; // K-scale（字节）
};

struct TileInfo {
    uint32_t loop = 0;              // 单核上处理的所有qs对应的第几个kvs基本块(累加递增)
    uint32_t curKvsTileLoopIdx = 0; // 在当前qs对应的Kvs的循环下标

    bool isFirstKvsTile = false;      // 当前qs对应的Kvs的第一个循环
    bool isLastKvsTile = false;       // 当前qs对应的Kvs的最后一个循环
    bool isLastSecondKvsTile = false; // 当前qs对应的Kvs的倒数第二个循环
    bool isUpdatePScale = false;      // TileGroup的最后一个
    bool isTileGoupFirstTile =
        false; // Kvs上16个softmax是一个TileGroup，这是每一个TileGroup(16个kvs基本块)的第一个softmax 任务，isC2Sync
    uint32_t kvsFirstTileStartVecCore = 0; // Kvs上第一个softmax分给哪个vec core
    uint32_t tileMaxIdx = 0;
    uint32_t updateScaleIdx = 0;
    bool isKvsFirstTilePerCore = false; // 16个softmax均分在两个core, 当前任务是否是分在当前core上的第一个，

    uint32_t kvsActBaseTile = 0;
    uint32_t kvsActBaseTileAlign16 = 0;
    uint32_t kvsActBaseTileAlign32 = 0;
    uint32_t kvsActBaseTileAlign64 = 0;
    // ===== PV 稀疏 gather 所需（mxfp4）=====
    uint32_t pvGatheredKvSTileIdx = 0; // 当前 tile 在 task 内的 gather 后 KV base tile 序号，供 PV
    uint16_t pscaleNum = 0;
};
} // namespace MXFP4Kernel

#include "epilogue/block/block_epilogue.hpp"
#include "epilogue/epilogue_dispatch_policy.hpp"
#include "gemm/block/block_mmad.hpp"
#include "gemm/gemm_dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace Mxfp4Kernel {

enum class Format
{
    TND = 0,
    BNSD = 1,
    BSND = 2
};

struct Mxfp4KernelParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR blockSparseMask;
    GM_ADDR actualSeqLengths;
    GM_ADDR actualSeqLengthsKv;
    GM_ADDR qDequantScale;
    GM_ADDR kDequantScale;
    GM_ADDR vDequantScale;
    GM_ADDR attentionOut;
    GM_ADDR workSpace;
    GM_ADDR tiling;

    // Methods
    __aicore__ inline Mxfp4KernelParams()
    {}
    __aicore__ inline Mxfp4KernelParams(
        GM_ADDR query_, GM_ADDR key_, GM_ADDR value_, GM_ADDR blockSparseMask_, GM_ADDR actualSeqLengths_,
        GM_ADDR actualSeqLengthsKv_, GM_ADDR qDequantScale_, GM_ADDR kDequantScale_, GM_ADDR vDequantScale_,
        GM_ADDR attentionOut_, GM_ADDR workSpace_, GM_ADDR tiling_)
        : query(query_),
          key(key_),
          value(value_),
          blockSparseMask(blockSparseMask_),
          actualSeqLengths(actualSeqLengths_),
          actualSeqLengthsKv(actualSeqLengthsKv_),
          qDequantScale(qDequantScale_),
          kDequantScale(kDequantScale_),
          vDequantScale(vDequantScale_),
          attentionOut(attentionOut_),
          workSpace(workSpace_),
          tiling(tiling_)
    {}
};

__aicore__ inline uint32_t GetCurQSTileNum(int64_t curQSeqlen, uint32_t blockShapeX, uint32_t qBaseTile)
{
    uint32_t fullXBlockNum = curQSeqlen / blockShapeX;
    uint32_t tailXBlockSize = curQSeqlen % blockShapeX;
    uint32_t qSTileNumPerFullXBlock = (blockShapeX + qBaseTile - 1) / qBaseTile;
    uint32_t qSTileNumTailXBlock = (tailXBlockSize + qBaseTile - 1) / qBaseTile;
    uint32_t curQSTileNum = qSTileNumPerFullXBlock * fullXBlockNum + qSTileNumTailXBlock;
    return curQSTileNum;
}

} // namespace Mxfp4Kernel

#endif
