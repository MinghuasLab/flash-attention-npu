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

#ifndef MXFP4_KERNEL_HPP
#define MXFP4_KERNEL_HPP

#include "kernel_operator.h"
#include "kernel_utils.hpp"
#include "epilogue/block/block_epilogue_utils.hpp"
#include "gemm/block/block_mmad_utils.hpp"

using namespace Catlass;
using namespace tla;
using namespace MXFP4Kernel;

namespace EpMXFP4 = Epilogue::Block::MXFP4; // UB 常量别名
namespace GmMXFP4 = Gemm::Block::MXFP4;     // L1 常量别名

namespace Mxfp4Kernel {

template <
    class EpilogueMask2Idx, class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV, class EpilogueRescaleO,
    class EpilogueComputePScale, class EpilogueCopyGlobalMaxUbToL1, class BlockMmadCopyGlobalMaxL1ToUB, Format qFormat,
    Format kvFormat, bool transposedMm1 = false, bool sparsePolicy = true>
class Mxfp4FullQuantKernel {
public:
    using ArchTag = typename BlockMmadPV::ArchTag;

    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementDisguiseP = typename EpilogueOnlineSoftmax::ElementDisguiseP;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename EpilogueRescaleO::ElementO;
    using ElementSparseMask = typename EpilogueMask2Idx::ElementSparseMask;
    using ElementSparseIdx = typename EpilogueMask2Idx::ElementSparseIdx;
    using ElementSparseCount = typename EpilogueMask2Idx::ElementSparseCount;
    using ElementGroupMax = typename EpilogueComputePScale::ElementGroupMax;
    using ElementPScale = typename EpilogueComputePScale::ElementPScale;
    using ElementLocalGlobalMax = typename EpilogueCopyGlobalMaxUbToL1::ElementLocalGlobalMax;

    using LayoutQ = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::RowMajor;
    using LayoutS = Catlass::layout::RowMajor;
    using LayoutPL1 = typename EpilogueOnlineSoftmax::LayoutPL1;
    using LayoutV = Catlass::layout::ColumnMajor;
    using LayoutO = Catlass::layout::RowMajor;
    using LayoutOTmp = Catlass::layout::RowMajor;
    using LayoutLocalGlobalMax = typename EpilogueCopyGlobalMaxUbToL1::LayoutLocalGlobalMax;
    using LayoutPScale = typename EpilogueComputePScale::LayoutPScale;
    using LayoutSparseIdx = Catlass::layout::RowMajor;
    using LayoutSparseCount = Catlass::layout::RowMajor;

    using LayoutTagL1P = typename BlockMmadPV::LayoutTagL1A;

    // 核间同步
    static constexpr uint8_t SYNC_MODE_2 = 2;
    static constexpr uint8_t SYNC_MODE_4 = 4;
    static constexpr uint16_t CROSS_CORE_SYNC_V1_C1[2] = {9, 10};
    static constexpr uint16_t CROSS_CORE_SYNC_GMAX_L1_UB_BACK = 8;
    static constexpr uint16_t CROSS_CORE_SYNC_GMAX_UB_TO_L1_BUF1 = 7;
    static constexpr uint16_t CROSS_CORE_SYNC_GMAX_UB_TO_L1_BUF0 = 6;
    static constexpr uint16_t CROSS_CORE_SYNC_GMAX_L1_TO_UB_BUF1 = 5;
    static constexpr uint16_t CROSS_CORE_SYNC_GMAX_L1_TO_UB_BUF0 = 4;
    static constexpr uint16_t CROSS_CORE_SYNC_PSCALE_C2_BUF1 = 3;
    static constexpr uint16_t CROSS_CORE_SYNC_PSCALE_C2_BUF0 = 2;
    static constexpr uint16_t CROSS_CORE_SYNC_C2_V2 = 1;
    static constexpr uint16_t CROSS_CORE_SYNC_V2_C2 = 0;

    static constexpr uint32_t PV_RE_DELAY_N = 20;
    static constexpr uint32_t DELAY_PSCALE_N = 3;

    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PV_RE_DELAY_N + 1;

    // Methods
    __aicore__ inline Mxfp4FullQuantKernel()
    {}

    __aicore__ inline void operator()(Mxfp4KernelParams const& params)
    {
        __gm__ Mxfp4TilingData* tilingData =
            reinterpret_cast<__gm__ Mxfp4TilingData*>(params.tiling);
        FetchBaseShapeInfo(tilingData);
        InitLayoutStride();
        // global buffers
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.query);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.key);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK*)params.value);
        AscendC::GlobalTensor<int64_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int64_t*)params.actualSeqLengths);
        AscendC::GlobalTensor<int64_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t*)params.actualSeqLengthsKv);
        AscendC::GlobalTensor<uint8_t> gBlockSparseMask;
        if constexpr (sparsePolicy) {
            gBlockSparseMask.SetGlobalBuffer((__gm__ uint8_t*)params.blockSparseMask);
        }
        AscendC::GlobalTensor<uint8_t> gQDequantScale;
        gQDequantScale.SetGlobalBuffer((__gm__ uint8_t*)params.qDequantScale);
        AscendC::GlobalTensor<uint8_t> gKDequantScale;
        gKDequantScale.SetGlobalBuffer((__gm__ uint8_t*)params.kDequantScale);
        AscendC::GlobalTensor<uint8_t> gVDequantScale;
        gVDequantScale.SetGlobalBuffer((__gm__ uint8_t*)params.vDequantScale);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO*)params.attentionOut);
        AscendC::GlobalTensor<ElementSparseIdx> gSparseIdx;
        AscendC::GlobalTensor<ElementSparseCount> gSparseCount;
        if constexpr (sparsePolicy) {
            gSparseIdx.SetGlobalBuffer((__gm__ ElementSparseIdx*)params.workSpace);
            gSparseCount.SetGlobalBuffer((__gm__ ElementSparseCount*)(params.workSpace + sparseIdxSize_));
        } else {
            // Dense TUs never read these tensors, but QK/PV still take a
            // GlobalTensor argument. Bind them to tiling so the pointer is valid.
            gSparseIdx.SetGlobalBuffer((__gm__ ElementSparseIdx*)params.tiling);
            gSparseCount.SetGlobalBuffer((__gm__ ElementSparseCount*)params.tiling);
        }

        // cross core data move dst buffers
        AscendC::LocalTensor<ElementS> ubSTensor[EpMXFP4::UB_S_BUF_CNT];
        AscendC::LocalTensor<ElementDisguiseP> l1PTensor[GmMXFP4::L1_P_BUF_CNT];
        AscendC::LocalTensor<ElementPScale> l1PScaleTensor[GmMXFP4::L1_P_SCALE_BUF_CNT];
        AscendC::LocalTensor<ElementS> l1LocalGlobalMaxTensor[GmMXFP4::L1_LOCAL_GLOBAL_MAX_BUF_CNT];
        AscendC::LocalTensor<ElementS> ubPeerGlobalMaxTensor[EpMXFP4::UB_PEER_GLOBAL_MAX_CNT];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor;

        InitCrossCoreDstBuf(
            l1PTensor, l1PScaleTensor, l1LocalGlobalMaxTensor, ubSTensor, ubPeerGlobalMaxTensor, ubOTmpTensor);
        // core idx
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        // set reverse sync flags
        InitSyncFlags();
        // mask preprocess
        if constexpr (sparsePolicy) {
            if ASCEND_IS_AIV {
                EpilogueMask2Idx epilogueMask2Idx(resource);
                uint32_t totalRowNumBlockMask = batch_ * qHeads_ * xBlockNumAligned_;
                epilogueMask2Idx(
                    gBlockSparseMask, gSparseIdx, gSparseCount, totalRowNumBlockMask, yBlockNumAligned_,
                    avgRowPerSubCore_, preActiveSubCoreNum_);
            }
            AscendC::SyncAll<false>();
        }

        // tla define
        auto ubSLayoutTla = tla::MakeLayout<ElementS, LayoutS>(EpMXFP4::KVS_BASE_SIZE, EpMXFP4::QS_BASE_SIZE * 2);
        auto ubSTensorTla = tla::MakeTensor(ubSTensor[0], ubSLayoutTla, Catlass::Arch::PositionUB{});

        auto l1PScaleLayoutTla =
            tla::MakeLayout<ElementPScale, LayoutPScale>(GmMXFP4::L1_P_SCALE_BUF_CNT, GmMXFP4::L1_P_SCALE_BUF_SIZE);
        auto l1PScaleTensorTla = tla::MakeTensor(l1PScaleTensor[0], l1PScaleLayoutTla, Catlass::Arch::PositionL1{});

#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();

        uint32_t kvBufId = 0;
        uint64_t softmaxScale = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&scaleValue_));
        BlockMmadQK blockMmadQK(resource, kvBufId, softmaxScale);
        BlockMmadPV blockMmadPV(resource, kvBufId);
        BlockMmadCopyGlobalMaxL1ToUB blockGMaxL1ToUb(resource);
#endif
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        // host 侧已预算: log2Cx_ = -dstTypeMax, log2CxCeil_ = ceil(dstTypeMax)
        half NEG_LOG2_CX = static_cast<half>(-log2Cx_);
        half LOG2_CX_CEIL = static_cast<half>(log2CxCeil_);

        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, NEG_LOG2_CX);
        EpilogueCopyGlobalMaxUbToL1 epilogueCopyGlobalMaxUbToL1(resource);
        EpilogueComputePScale epilogueComputePScale(resource, LOG2_CX_CEIL);
        EpilogueRescaleO epilogueRescaleO(resource);
#endif

        // runMainLoop
        uint64_t coreTileLoop = 0;

        uint64_t curTaskTileIdx = 0;
        uint64_t delay20TaskTileIdx = 0;
        uint64_t delay3TaskTileIdx = 0;

        uint32_t coreTaskId = 0;
        uint32_t delay20CoreTaskId = 0;
        uint32_t delay3CoreTaskId = 0;

        uint64_t preTaskTileSum = 0;
        uint64_t delay20preTaskTileSum = 0;
        uint64_t delay3preTaskTileSum = 0;

        uint64_t coreTaskNum = coreTaskNum_;
        coreTaskNum = (coreIdx < totalTaskNum_ % coreNum) ? (coreTaskNum + 1) : coreTaskNum;

        TaskInfo taskInfos[PRELOAD_TASK_CACHE_SIZE] = {};
        TileInfo tileInfos[PRELOAD_TASK_CACHE_SIZE] = {};

        // CreateTaskInfo 跨调用 batch 扫描状态：curTotalTaskNum 从首 batch task 数起步，
        // 后续调用在 while 循环内基于该状态续算，不再从 batch 0 重扫
        BatchOffsetInfo batchOffsetInfo;
        batchOffsetInfo.curTotalTaskNum = firstBatchTaskNum_;

        const uint32_t kvHeadMul = (kvFormat == Format::BNSD) ? 1u : kvHeads_;
        const uint32_t qHeadMul = (qFormat == Format::BNSD) ? 1u : qHeads_;

        while (coreTaskId < coreTaskNum || delay20CoreTaskId < coreTaskNum || delay3CoreTaskId < coreTaskNum) {
            curTaskTileIdx = coreTileLoop - preTaskTileSum;
            delay20TaskTileIdx = coreTileLoop - PV_RE_DELAY_N - delay20preTaskTileSum;
            delay3TaskTileIdx = coreTileLoop - DELAY_PSCALE_N - delay3preTaskTileSum;

            TaskInfo& curTaskInfo = taskInfos[coreTaskId % PRELOAD_TASK_CACHE_SIZE];
            TaskInfo& delay20TaskInfo = taskInfos[delay20CoreTaskId % PRELOAD_TASK_CACHE_SIZE];
            TaskInfo& delay3TaskInfo = taskInfos[delay3CoreTaskId % PRELOAD_TASK_CACHE_SIZE];

            TileInfo& curTileInfo = tileInfos[coreTileLoop % PRELOAD_TASK_CACHE_SIZE];
            TileInfo& delay20TileInfo = tileInfos[(coreTileLoop - PV_RE_DELAY_N) % PRELOAD_TASK_CACHE_SIZE];
            TileInfo& delay3TileInfo = tileInfos[(coreTileLoop - DELAY_PSCALE_N) % PRELOAD_TASK_CACHE_SIZE];

            // 创建qs任务
            if (curTaskTileIdx == 0 && coreTaskId < coreTaskNum) {
                CreateTaskInfo(
                    curTaskInfo, coreTaskId, coreIdx, coreNum, gActualQseqlen, gActualKvseqlen, gSparseCount,
                    gSparseIdx, batchOffsetInfo);
            }

            // main process
            if (coreTileLoop >= PV_RE_DELAY_N && delay20CoreTaskId < coreTaskNum) {
                if ASCEND_IS_AIC {
                    if (delay20TileInfo.isTileGoupFirstTile) {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(
                            CROSS_CORE_SYNC_PSCALE_C2_BUF0 + delay20TileInfo.pscaleNum);
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(
                            CROSS_CORE_SYNC_PSCALE_C2_BUF0 + delay20TileInfo.pscaleNum + 16);
                    }
                    if (delay20TileInfo.isUpdatePScale) {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2);
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2 + 16);
                    }

                    auto gmVLayoutTla = tla::MakeLayout<ElementV, LayoutV>(kvHeadMul * embed_, kvBaseTile_);
                    auto gmVTensorTla =
                        tla::MakeTensor(gV[delay20TaskInfo.gmOffsetV], gmVLayoutTla, Catlass::Arch::PositionGM{});

                    // [catlass化] UB Oᵀ 实际存储布局：rows=embed_+1（D 行 Oᵀ + 1 行 rowsum）
                    auto ubOTmpLayoutTla = tla::MakeLayout(
                        tla::MakeShape(GmMXFP4::ROW_SUM_NUM + 1, delay20TaskInfo.qsActBaseTileAlign64),
                        tla::MakeStride(static_cast<int64_t>(64), tla::Int<1>{}));
                    auto ubOTmpTensorTla = tla::MakeTensor(ubOTmpTensor, ubOTmpLayoutTla, Catlass::Arch::PositionUB{});

                    Catlass::GemmCoord actualBlockShapePV{
                        delay20TaskInfo.qsActBaseTile, embed_, delay20TileInfo.kvsActBaseTile};

                    blockMmadPV(
                        gmVTensorTla, gVDequantScale, gSparseIdx, ubOTmpTensorTla, actualBlockShapePV, delay20TileInfo,
                        delay20TaskInfo, kvBaseTile_, blockShapeY_, kvHeadMul, SparsityPolicy<sparsePolicy>{});

                    if (delay20TileInfo.isUpdatePScale) {
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2 + 16);
                    }
                } else {
                    if (delay20TileInfo.isUpdatePScale) {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C2_V2);

                        auto gmOLayoutTla = tla::MakeLayout(
                            tla::MakeShape(qBaseTile_, embed_),
                            tla::MakeStride(delay20TaskInfo.oShapeCol, tla::Int<1>{}));
                        auto gmOTensorTla =
                            tla::MakeTensor(gO[delay20TaskInfo.gmOffsetO], gmOLayoutTla, Catlass::Arch::PositionGM{});
                        Catlass::GemmCoord actualBlockShapeVPT{
                            embed_, delay20TaskInfo.qsActBaseTile, delay20TileInfo.kvsActBaseTile};
                        epilogueRescaleO(gmOTensorTla, actualBlockShapeVPT, delay20TaskInfo, delay20TileInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
                    }
                }
            }
            if (coreTileLoop >= DELAY_PSCALE_N && delay3CoreTaskId < coreTaskNum) {
                if (delay3TileInfo.isUpdatePScale) {
                    if ASCEND_IS_AIC {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(
                            CROSS_CORE_SYNC_GMAX_UB_TO_L1_BUF0 + delay3TileInfo.tileMaxIdx / 2);
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(
                            CROSS_CORE_SYNC_GMAX_UB_TO_L1_BUF0 + delay3TileInfo.tileMaxIdx / 2 + 16);
                        auto l1GMaxLayoutTla =
                            tla::MakeLayout<ElementLocalGlobalMax, LayoutLocalGlobalMax>(1, EpMXFP4::KVS_BASE_SIZE);
                        auto l1GMaxTensorTla = tla::MakeTensor(
                            ubPeerGlobalMaxTensor[delay3TileInfo.tileMaxIdx], l1GMaxLayoutTla,
                            Catlass::Arch::PositionUB{});
                        blockGMaxL1ToUb(l1GMaxTensorTla, delay3TileInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(
                            CROSS_CORE_SYNC_GMAX_L1_TO_UB_BUF0 + delay3TileInfo.tileMaxIdx / 2);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(
                            CROSS_CORE_SYNC_GMAX_L1_TO_UB_BUF0 + delay3TileInfo.tileMaxIdx / 2 + 16);
                        if (delay3TileInfo.tileMaxIdx == 3 ||
                            (delay3CoreTaskId == coreTaskNum - 1 &&
                             delay3TileInfo.isLastKvsTile)) { // tilemax最后一块 或者 单核上的最后一块
                            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK);
                            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK + 16);
                        }
                    } else {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(
                            CROSS_CORE_SYNC_GMAX_L1_TO_UB_BUF0 + delay3TileInfo.tileMaxIdx / 2);
                        epilogueComputePScale(l1PScaleTensorTla, delay3TileInfo, delay3TaskInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(
                            CROSS_CORE_SYNC_PSCALE_C2_BUF0 + delay3TileInfo.pscaleNum);
                    }
                }
            }
            if (coreTaskId < coreTaskNum) {
                CreateTileInfo(
                    curTileInfo, coreTileLoop, static_cast<uint32_t>(curTaskTileIdx),
                    curTaskInfo); // 创建任务

                uint32_t mm1ResBufId = (curTileInfo.loop / 2) % 2;
                uint32_t subBlockIdx = curTileInfo.loop % 2;

                if ASCEND_IS_AIC {
                    AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(
                        CROSS_CORE_SYNC_V1_C1[mm1ResBufId] + subBlockIdx * 16);
                    auto gQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(
                        curTaskInfo.qsActBaseTile, // rows = S1 实际 tile
                        qHeadMul * embed_);        // cols → row-stride（BNSD 时 qHeadMul=1，即 embed_）
                    auto gQTensorTla =
                        tla::MakeTensor(gQ[curTaskInfo.gmOffsetQ], gQLayoutTla, Catlass::Arch::PositionGM{});
                    auto gKLayout = tla::MakeLayout<ElementK, LayoutK>(
                        curTaskInfo.kvSeqlen,
                        kvHeadMul * embed_); // [BSND stride 修复] row-stride=kvHeadMul*embed(BNSD=embed)
                    auto gKTensorTla =
                        tla::MakeTensor(gK[curTaskInfo.gmOffsetK], gKLayout, Catlass::Arch::PositionGM{});
                    auto curUBSTensorTla = GetTile(
                        ubSTensorTla, tla::MakeCoord(0, EpMXFP4::QS_BASE_SIZE * mm1ResBufId),
                        MakeShape(EpMXFP4::KVS_BASE_SIZE, EpMXFP4::QS_BASE_SIZE));

                    // QK：UB S 传当前 ping-pong 槽 TLA；Fixpipe 按 subBlockIdx 写到对应 AIV。
                    blockMmadQK(
                        gQTensorTla, gKTensorTla, gQDequantScale[curTaskInfo.gmOffsetQScale],
                        gKDequantScale[curTaskInfo.gmOffsetKScale], gSparseIdx[curTaskInfo.gmOffsetSparseIdx],
                        curUBSTensorTla, curTaskInfo, curTileInfo, embed_, kvBaseTile_, blockShapeY_, subBlockIdx,
                        kvHeadMul, qHeadMul, SparsityPolicy<sparsePolicy>{});
                    AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(
                        CROSS_CORE_SYNC_V1_C1[mm1ResBufId] + subBlockIdx * 16);
                } else {
#ifdef __DAV_VEC__
                    if (subBlockIdx == AscendC::GetSubBlockIdx()) {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId]);
                        uint32_t l1PBufId = curTileInfo.loop % GmMXFP4::L1_P_BUF_CNT;
                        auto l1PLayoutTla = tla::MakeLayout<ElementDisguiseP, LayoutPL1>(
                            EpMXFP4::KVS_BASE_SIZE, EpMXFP4::QS_BASE_SIZE / 2);
                        auto l1PTensorTla =
                            tla::MakeTensor(l1PTensor[l1PBufId], l1PLayoutTla, Catlass::Arch::PositionL1{});
                        Catlass::GemmCoord actualBlockShapeQK{
                            curTileInfo.kvsActBaseTile, curTaskInfo.qsActBaseTile, embed_};
                        epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, curTileInfo, curTaskInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId]);
                    }
                    if (curTileInfo.isUpdatePScale) {
                        if (curTileInfo.tileMaxIdx == 0) {
                            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK);
                        }
                        auto l1GMaxLayoutTla =
                            tla::MakeLayout<ElementLocalGlobalMax, LayoutLocalGlobalMax>(2, EpMXFP4::QS_BASE_SIZE);
                        auto l1GMaxTensorTla = tla::MakeTensor(
                            l1LocalGlobalMaxTensor[curTileInfo.tileMaxIdx], l1GMaxLayoutTla,
                            Catlass::Arch::PositionL1{});
                        epilogueCopyGlobalMaxUbToL1(l1GMaxTensorTla, curTileInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(
                            CROSS_CORE_SYNC_GMAX_UB_TO_L1_BUF0 + curTileInfo.tileMaxIdx / 2);
                    }
#endif
                }
            }

            // 是否要切换到下一个qs任务, 新的kvs
            if (curTaskTileIdx == curTaskInfo.taskTileNum - 1 && coreTaskId < coreTaskNum) {
                preTaskTileSum += curTaskInfo.taskTileNum;
                coreTaskId++;
            }
            if (coreTileLoop >= PV_RE_DELAY_N && delay20TaskTileIdx == delay20TaskInfo.taskTileNum - 1) {
                delay20preTaskTileSum += delay20TaskInfo.taskTileNum;
                delay20CoreTaskId++;
            }
            if (coreTileLoop >= DELAY_PSCALE_N && delay3TaskTileIdx == delay3TaskInfo.taskTileNum - 1 &&
                delay3CoreTaskId < coreTaskNum) {
                delay3preTaskTileSum += delay3TaskInfo.taskTileNum;
                delay3CoreTaskId++;
            }
            coreTileLoop++;
        }

        // release reverse sync flags
        ReleaseSyncFlags();
    }

    __aicore__ inline void FetchBaseShapeInfo(__gm__ Mxfp4TilingData* tilingData)
    {
        batch_ = tilingData->batch;
        qHeads_ = tilingData->numHeads;
        kvHeads_ = tilingData->kvHeads;
        embed_ = tilingData->embeddingSize;
        firstBatchTaskNum_ = tilingData->firstBatchTaskNum;
        totalTaskNum_ = tilingData->totalTaskNum;
        coreTaskNum_ = tilingData->coreTaskNum;
        blockShapeX_ = tilingData->blockShapeX;
        blockShapeY_ = tilingData->blockShapeY;
        scaleValue_ = tilingData->scaleValue;
        log2Cx_ = tilingData->log2Cx;
        log2CxCeil_ = tilingData->log2CxCeil;
        // mask2idx tile info
        xBlockNumAligned_ = tilingData->mask2IdxTileInfo.xBlockNumAligned;
        yBlockNumAligned_ = tilingData->mask2IdxTileInfo.yBlockNumAligned;
        avgRowPerSubCore_ = tilingData->mask2IdxTileInfo.avgRowPerSubCore;
        preActiveSubCoreNum_ = tilingData->mask2IdxTileInfo.preActiveSubCoreNum;
        // base tile info
        qBaseTile_ = tilingData->baseTileInfo.qBaseTile;
        kvBaseTile_ = tilingData->baseTileInfo.kvBaseTile;
        // whether actual seqlen is provided
        actSeqAval_ = (!tilingData->useUniformQSeqlen) && (!tilingData->useUniformKvSeqlen);
        sparseIdxSize_ = tilingData->selectIdxSize;
        // aligned seqlen q & kv
        qSeqlenAligned_ = tilingData->maxQSeqlen;
        kvSeqlenAligned_ = tilingData->maxKvSeqlen;
    }

    __aicore__ inline void InitLayoutStride()
    {
        // O stride
        if constexpr (qFormat == Format::TND) {
            strideO_ = static_cast<int64_t>(qHeads_) * embed_;
        } else if constexpr (qFormat == Format::BNSD) {
            strideOB_ = static_cast<int64_t>(qHeads_) * qSeqlenAligned_ * embed_;
            strideON_ = qSeqlenAligned_ * embed_;
            strideOS_ = embed_;
        } else if constexpr (qFormat == Format::BSND) {
            strideOB_ = qSeqlenAligned_ * static_cast<int64_t>(qHeads_) * embed_;
            strideOS_ = static_cast<int64_t>(qHeads_) * embed_;
            strideON_ = embed_;
        }
        // V / V-scale / K-scale stride
        const int64_t kScaleBytesPerToken = static_cast<int64_t>(embed_ / 64) * 2;
        if constexpr (kvFormat == Format::TND) {
            strideKVData_ = static_cast<int64_t>(kvHeads_) * embed_;
            vScaleStrideN_ = static_cast<int64_t>(embed_) * 2;
            kScaleStrideN_ = kScaleBytesPerToken;
        } else if constexpr (kvFormat == Format::BNSD) {
            strideKVDataB_ = static_cast<int64_t>(kvHeads_) * kvSeqlenAligned_ * embed_;
            strideKVDataN_ = kvSeqlenAligned_ * embed_;
            vScaleStrideB_ =
                static_cast<int64_t>(kvHeads_) * CeilDiv(kvSeqlenAligned_, static_cast<int64_t>(64)) * embed_ * 2;
            vScaleStrideN_ = CeilDiv(kvSeqlenAligned_, static_cast<int64_t>(64)) * embed_ * 2;
            kScaleStrideB_ = static_cast<int64_t>(kvHeads_) * kvSeqlenAligned_ * kScaleBytesPerToken;
            kScaleStrideN_ = kvSeqlenAligned_ * kScaleBytesPerToken;
        } else if constexpr (kvFormat == Format::BSND) {
            strideKVDataB_ = kvSeqlenAligned_ * static_cast<int64_t>(kvHeads_) * embed_;
            strideKVDataN_ = embed_;
            vScaleStrideB_ =
                CeilDiv(kvSeqlenAligned_, static_cast<int64_t>(64)) * static_cast<int64_t>(kvHeads_) * embed_ * 2;
            vScaleStrideN_ = static_cast<int64_t>(embed_) * 2;
            kScaleStrideB_ = kvSeqlenAligned_ * static_cast<int64_t>(kvHeads_) * kScaleBytesPerToken;
            kScaleStrideN_ = kScaleBytesPerToken;
        }
        // Q 数据 / Q-scale stride（Host 沿 D）
        const int64_t qScaleBytesPerToken = static_cast<int64_t>(embed_ / 64) * 2;
        if constexpr (qFormat == Format::TND) {
            strideQData_ = static_cast<int64_t>(qHeads_) * embed_;
            qScaleStrideN_ = qScaleBytesPerToken;
        } else if constexpr (qFormat == Format::BNSD) {
            strideQDataB_ = static_cast<int64_t>(qHeads_) * qSeqlenAligned_ * embed_;
            strideQDataN_ = qSeqlenAligned_ * embed_;
            qScaleStrideB_ = static_cast<int64_t>(qHeads_) * qSeqlenAligned_ * qScaleBytesPerToken;
            qScaleStrideN_ = qSeqlenAligned_ * qScaleBytesPerToken;
        } else if constexpr (qFormat == Format::BSND) {
            strideQDataB_ = qSeqlenAligned_ * static_cast<int64_t>(qHeads_) * embed_;
            strideQDataN_ = embed_;
            qScaleStrideB_ = qSeqlenAligned_ * static_cast<int64_t>(qHeads_) * qScaleBytesPerToken;
            qScaleStrideN_ = qScaleBytesPerToken;
        }
    }

    __aicore__ inline void InitCrossCoreDstBuf(
        AscendC::LocalTensor<ElementDisguiseP> (&l1PTensor)[GmMXFP4::L1_P_BUF_CNT],
        AscendC::LocalTensor<ElementPScale> (&l1PScaleTensor)[GmMXFP4::L1_P_SCALE_BUF_CNT],
        AscendC::LocalTensor<ElementS> (&l1LocalGlobalMax)[GmMXFP4::L1_LOCAL_GLOBAL_MAX_BUF_CNT],
        AscendC::LocalTensor<ElementS> (&ubSTensor)[EpMXFP4::UB_S_BUF_CNT],
        AscendC::LocalTensor<ElementS> (&ubPeerGlobalMax)[EpMXFP4::UB_PEER_GLOBAL_MAX_CNT],
        AscendC::LocalTensor<ElementOTmp>& ubOTmpTensor)

    {
        // L1 侧 buffer 初始化：OFFSET + i * SIZE 递增
        for (uint32_t i = 0; i < GmMXFP4::L1_P_BUF_CNT; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementDisguiseP>(
                GmMXFP4::L1_P_BUF_OFFSET + GmMXFP4::L1_P_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < GmMXFP4::L1_P_SCALE_BUF_CNT; i++) {
            l1PScaleTensor[i] = resource.l1Buf.template GetBufferByByte<ElementPScale>(
                GmMXFP4::L1_P_SCALE_BUF_OFFSET + GmMXFP4::L1_P_SCALE_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < GmMXFP4::L1_LOCAL_GLOBAL_MAX_BUF_CNT; i++) {
            l1LocalGlobalMax[i] = resource.l1Buf.template GetBufferByByte<ElementS>(
                GmMXFP4::L1_LOCAL_GLOBAL_MAX_BUF_OFFSET + GmMXFP4::L1_LOCAL_GLOBAL_MAX_BUF_SIZE * i);
        }
        // UB 侧 buffer 初始化
        for (uint32_t i = 0; i < EpMXFP4::UB_S_BUF_CNT; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                EpMXFP4::UB_S_BUF_OFFSET + EpMXFP4::UB_S_INNER_BUF_OFFSET * i);
        }
        for (uint32_t i = 0; i < EpMXFP4::UB_PEER_GLOBAL_MAX_CNT; i++) {
            ubPeerGlobalMax[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                EpMXFP4::UB_PEER_GLOBAL_MAX_BUF_OFFSET + EpMXFP4::UB_PEER_GLOBAL_MAX_BUF_SIZE * i);
        }
        // OTmp 单 buffer
        ubOTmpTensor = resource.ubBuf.template GetBufferByByte<ElementOTmp>(EpMXFP4::UB_OTMP_BUF_OFFSET);
    }

    // Qs维度任务
    __aicore__ inline void CreateTaskInfo(
        TaskInfo& info, uint32_t coreTaskId, uint32_t coreIdx, uint32_t coreNum,
        AscendC::GlobalTensor<int64_t>& gActualQseqlen, AscendC::GlobalTensor<int64_t>& gActualKvseqlen,
        AscendC::GlobalTensor<ElementSparseCount>& gSparseCount, AscendC::GlobalTensor<ElementSparseIdx>& gSparseIdx,
        BatchOffsetInfo& batchInfo)
    {
        uint32_t taskIdx = coreTaskId * coreNum + coreIdx; // 反推全局 task 序号

        uint32_t groupSize = qHeads_ / kvHeads_;
        uint32_t qSTileNumPerFullXBlock = CeilDiv(blockShapeX_, qBaseTile_);

        // 反推 taskIdx 所在的 batch：同核 taskIdx 单调递增，
        // 从 batchInfo 保存的上次扫描位置续算，不再从 batch 0 重扫
        int64_t qSeqlen = actSeqAval_ ? gActualQseqlen.GetValue(batchInfo.curBatch) : qSeqlenAligned_;
        int64_t kvSeqlen = actSeqAval_ ? gActualKvseqlen.GetValue(batchInfo.curBatch) : kvSeqlenAligned_;
        uint32_t curQSTileNum = GetCurQSTileNum(qSeqlen, blockShapeX_, qBaseTile_);
        while (taskIdx >= batchInfo.curTotalTaskNum) {
            ++batchInfo.curBatch;
            batchInfo.preTotalTaskNum = batchInfo.curTotalTaskNum;
            // TND: 跨 batch 时累加上一 batch 实际长度（此时 kvSeqlen/qSeqlen 仍为上一 batch 值）
            if constexpr (qFormat == Format::TND) {
                batchInfo.oBOffset += qSeqlen * strideO_;
                batchInfo.qBOffset += qSeqlen * strideQData_;
                batchInfo.qScaleBOffset += qSeqlen * static_cast<int64_t>(qHeads_) * qScaleStrideN_;
            }
            if constexpr (kvFormat == Format::TND) {
                batchInfo.vBOffset += kvSeqlen * strideKVData_;
                batchInfo.vScaleBOffset +=
                    CeilDiv(kvSeqlen, static_cast<int64_t>(64)) * static_cast<int64_t>(kvHeads_) * embed_ * 2;
                batchInfo.kScaleBOffset += kvSeqlen * static_cast<int64_t>(kvHeads_) * kScaleStrideN_;
            }
            qSeqlen = actSeqAval_ ? gActualQseqlen.GetValue(batchInfo.curBatch) : qSeqlenAligned_;
            kvSeqlen = actSeqAval_ ? gActualKvseqlen.GetValue(batchInfo.curBatch) : kvSeqlenAligned_;
            curQSTileNum = GetCurQSTileNum(qSeqlen, blockShapeX_, qBaseTile_);
            batchInfo.curTotalTaskNum += curQSTileNum * qHeads_;
        }
        uint32_t curBatch = batchInfo.curBatch;

        // 拆解 taskIdx 得到 head / qSTile / xBlock 索引
        uint32_t taskIdxCurBatch = taskIdx - batchInfo.preTotalTaskNum;
        uint32_t qHeadIdx = taskIdxCurBatch / curQSTileNum;
        uint32_t kvHeadIdx = qHeadIdx / groupSize;
        uint32_t qSTileIdx = taskIdxCurBatch - qHeadIdx * curQSTileNum;
        uint32_t xBlockIdx = qSTileIdx / qSTileNumPerFullXBlock;
        uint32_t qSTileIdxCurXBlock = qSTileIdx - xBlockIdx * qSTileNumPerFullXBlock;

        // 实际 Q tile 大小 (最后一个 tile 可能不足 qBaseTile_)
        uint32_t xBlockNumAval = static_cast<uint32_t>(CeilDiv(qSeqlen, static_cast<int64_t>(blockShapeX_)));
        uint32_t xBlockSize =
            (xBlockIdx == xBlockNumAval - 1) ? static_cast<uint32_t>(qSeqlen - xBlockIdx * blockShapeX_) : blockShapeX_;
        uint32_t qSTileNumCurXBlock = CeilDiv(xBlockSize, qBaseTile_);
        uint32_t qsActBaseTile = (qSTileIdxCurXBlock == qSTileNumCurXBlock - 1) ?
                                     (xBlockSize - qSTileIdxCurXBlock * qBaseTile_) :
                                     qBaseTile_;

        // Sparse tasks consume mask2idx output. Dense tasks cover the complete
        // KV sequence directly and never read sparse workspace.
        uint32_t gmOffsetSparseCount = 0;
        uint32_t yBlockNumRsvd = 0;
        uint32_t gmOffsetSparseIdx = 0;
        uint32_t yBlockNumAval = static_cast<uint32_t>(CeilDiv(kvSeqlen, static_cast<int64_t>(blockShapeY_)));
        int64_t gatheredKvSeqlen = kvSeqlen;
        if constexpr (sparsePolicy) {
            gmOffsetSparseCount = curBatch * qHeads_ * xBlockNumAligned_ + qHeadIdx * xBlockNumAligned_ + xBlockIdx;
            yBlockNumRsvd = gSparseCount.GetValue(gmOffsetSparseCount);
            gmOffsetSparseIdx = gmOffsetSparseCount * yBlockNumAligned_;
            uint32_t lastIdxOffset = gmOffsetSparseIdx + yBlockNumRsvd - 1;
            uint32_t lastSparseIdx = gSparseIdx.GetValue(lastIdxOffset);
            uint32_t lastYBlockSize = (lastSparseIdx == yBlockNumAval - 1) ?
                                          static_cast<uint32_t>(kvSeqlen - lastSparseIdx * blockShapeY_) :
                                          blockShapeY_;
            gatheredKvSeqlen = (yBlockNumRsvd - 1) * blockShapeY_ + lastYBlockSize;
        } else {
            yBlockNumRsvd = yBlockNumAval;
        }

        // taskTileNum = ceil(gatheredKvSeqlen / kvBaseTile_)
        uint32_t taskTileNum = static_cast<uint32_t>(CeilDiv(gatheredKvSeqlen, static_cast<int64_t>(kvBaseTile_)));

        // O GM offset
        int64_t qSOffset = static_cast<int64_t>(xBlockIdx) * blockShapeX_ + qSTileIdxCurXBlock * qBaseTile_;
        if constexpr (qFormat == Format::TND) {
            info.gmOffsetO = batchInfo.oBOffset + qSOffset * strideO_ + qHeadIdx * embed_;
        } else if constexpr (qFormat == Format::BNSD) {
            int64_t oBOffset = static_cast<int64_t>(curBatch) * strideOB_;
            info.gmOffsetO = oBOffset + qHeadIdx * strideON_ + qSOffset * strideOS_;
        } else if constexpr (qFormat == Format::BSND) {
            int64_t oBOffset = static_cast<int64_t>(curBatch) * strideOB_;
            info.gmOffsetO = oBOffset + qSOffset * strideOS_ + qHeadIdx * strideON_;
        }
        // O shape col
        if constexpr (qFormat == Format::TND) {
            info.oShapeCol = static_cast<uint32_t>(strideO_);
        } else if constexpr (qFormat == Format::BNSD) {
            info.oShapeCol = static_cast<uint32_t>(strideOS_);
        } else if constexpr (qFormat == Format::BSND) {
            info.oShapeCol = static_cast<uint32_t>(strideOS_);
        }

        info.batchIdx = curBatch;
        info.qHeadIdx = qHeadIdx;
        info.kvHeadIdx = kvHeadIdx;
        info.qSTileIdx = qSTileIdx;
        info.xBlockIdx = xBlockIdx;
        info.gatheredKvSeqlen = gatheredKvSeqlen;
        info.qsActBaseTile = qsActBaseTile;

        info.qsActBaseTileAlign8 = CeilDiv(qsActBaseTile, 8u) * 8;
        info.qsActBaseTileAlign16 = CeilDiv(qsActBaseTile, 16u) * 16;
        info.qsActBaseTileAlign64 = CeilDiv(qsActBaseTile, 64u) * 64;
        info.qsActBaseTileAlign128 = CeilDiv(qsActBaseTile, 128u) * 128;

        info.qsBaseTile = qBaseTile_;
        info.kvsBaseTile = kvBaseTile_;
        info.taskTileNum = taskTileNum;
        // ===== PV 稀疏 gather 所需 =====
        info.gmOffsetSparseIdx = gmOffsetSparseIdx; // 本 task sparseIdx 起始
        info.kvSeqlen = kvSeqlen;
        info.yBlockNumAval = yBlockNumAval;
        info.yBlockNumRsvd = yBlockNumRsvd;
        // V 数据 batch+head 基址（seq 由 gather 的 oriStartOffset 处理，基址只含 batch+head）
        // V-scale batch+head 基址（字节单位，布局 [B,N2,S2//64,D,2]）
        if constexpr (kvFormat == Format::TND) {
            // TND V [T2,N2,D]：head 在 token 内 stride = embed_（≠ strideKVData_，后者是 batch-token 步长
            // kvHeads_*embed_）
            info.gmOffsetV = batchInfo.vBOffset + kvHeadIdx * embed_;
            info.gmOffsetVScale = batchInfo.vScaleBOffset + kvHeadIdx * vScaleStrideN_;
            info.gmOffsetK = info.gmOffsetV;
            info.gmOffsetKScale = batchInfo.kScaleBOffset + kvHeadIdx * kScaleStrideN_;
        } else if constexpr (kvFormat == Format::BNSD) {
            info.gmOffsetV = static_cast<int64_t>(curBatch) * strideKVDataB_ + kvHeadIdx * strideKVDataN_;
            info.gmOffsetVScale = static_cast<int64_t>(curBatch) * vScaleStrideB_ + kvHeadIdx * vScaleStrideN_;
            info.gmOffsetK = info.gmOffsetV;
            info.gmOffsetKScale = static_cast<int64_t>(curBatch) * kScaleStrideB_ + kvHeadIdx * kScaleStrideN_;
        } else { // BSND
            info.gmOffsetV = static_cast<int64_t>(curBatch) * strideKVDataB_ + kvHeadIdx * strideKVDataN_;
            info.gmOffsetVScale = static_cast<int64_t>(curBatch) * vScaleStrideB_ + kvHeadIdx * vScaleStrideN_;
            info.gmOffsetK = info.gmOffsetV;
            info.gmOffsetKScale = static_cast<int64_t>(curBatch) * kScaleStrideB_ + kvHeadIdx * kScaleStrideN_;
        }
        // Q / Q-scale：稠密搬需要带上本 task 的 Q seq 起点（与上方 O 的 qSOffset 一致）
        const int64_t qScaleBytesPerTok = static_cast<int64_t>(embed_ / 64) * 2;
        if constexpr (qFormat == Format::TND) {
            info.gmOffsetQ = batchInfo.qBOffset + qSOffset * strideQData_ + qHeadIdx * embed_;
            info.gmOffsetQScale = batchInfo.qScaleBOffset +
                                  qSOffset * static_cast<int64_t>(qHeads_) * qScaleBytesPerTok +
                                  qHeadIdx * qScaleStrideN_;
        } else if constexpr (qFormat == Format::BNSD) {
            info.gmOffsetQ =
                static_cast<int64_t>(curBatch) * strideQDataB_ + qHeadIdx * strideQDataN_ + qSOffset * embed_;
            info.gmOffsetQScale = static_cast<int64_t>(curBatch) * qScaleStrideB_ + qHeadIdx * qScaleStrideN_ +
                                  qSOffset * qScaleBytesPerTok;
        } else { // BSND
            info.gmOffsetQ = static_cast<int64_t>(curBatch) * strideQDataB_ +
                             qSOffset * static_cast<int64_t>(qHeads_) * embed_ + qHeadIdx * embed_;
            info.gmOffsetQScale = static_cast<int64_t>(curBatch) * qScaleStrideB_ +
                                  qSOffset * static_cast<int64_t>(qHeads_) * qScaleBytesPerTok +
                                  qHeadIdx * qScaleStrideN_;
        }
    }

    // 256*128基本块维度
    __aicore__ inline void CreateTileInfo(
        TileInfo& info, uint64_t coreTileLoop, uint32_t curTaskTileIdx, const TaskInfo& taskInfo)
    {
        info.loop = static_cast<uint32_t>(coreTileLoop);
        info.curKvsTileLoopIdx = curTaskTileIdx;
        info.isFirstKvsTile = (curTaskTileIdx == 0);
        info.isLastKvsTile = (curTaskTileIdx + 1 == taskInfo.taskTileNum);
        info.isLastSecondKvsTile =
            ((curTaskTileIdx + 2 == taskInfo.taskTileNum) || (taskInfo.taskTileNum == 1)); // pv rowsum刷1

        info.isUpdatePScale = (info.isLastKvsTile || ((info.curKvsTileLoopIdx + 1) % TILE_GROUP_N == 0));
        info.isTileGoupFirstTile = (info.curKvsTileLoopIdx % TILE_GROUP_N == 0);
        if (info.isFirstKvsTile) {
            kvsFirstTileStartVecCore_ = static_cast<uint32_t>(coreTileLoop % 2);
        }
        info.kvsFirstTileStartVecCore = kvsFirstTileStartVecCore_;

        if (info.isTileGoupFirstTile && !info.isFirstKvsTile) { // 非第一个kvs tileGroup，更新dm id
            updateScaleIdx_ = (updateScaleIdx_ + 1) % 4;
        }

        // task 第一个 tile 或 TileGroup 第一个 tile 时切换 buff (四缓冲)
        if (info.isFirstKvsTile || info.isTileGoupFirstTile) {
            tileMaxIdx_ = (tileMaxIdx_ + 1) % 4;
            pscaleNum_ = (pscaleNum_ + 1) % 20;
        }
        info.pscaleNum = static_cast<uint16_t>(pscaleNum_ / 10);
        info.tileMaxIdx = tileMaxIdx_;
        info.updateScaleIdx = updateScaleIdx_;
        info.isKvsFirstTilePerCore = (info.curKvsTileLoopIdx % TILE_GROUP_N / 2 == 0);

        info.kvsActBaseTile =
            info.isLastKvsTile ? (taskInfo.gatheredKvSeqlen - (taskInfo.taskTileNum - 1) * kvBaseTile_) : kvBaseTile_;
        // PV 稀疏 gather 用：当前 tile 在 task 内 gather 后的 KV base tile 序号
        info.pvGatheredKvSTileIdx = curTaskTileIdx;
        info.kvsActBaseTileAlign16 = (info.kvsActBaseTile + 15) / 16 * 16;
        info.kvsActBaseTileAlign32 = (info.kvsActBaseTile + 31) / 32 * 32;
        info.kvsActBaseTileAlign64 = (info.kvsActBaseTile + 63) / 64 * 64;
    }

    __aicore__ inline void InitSyncFlags()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK);
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK + 16);
        } else {
            // 核间
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[0]);
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[1]);
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
            // mask2index
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);

            // vec
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP4::SYNC_P_BUF0_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP4::SYNC_P_BUF1_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP4::SYNC_ATTNOUT_BUF_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP4::SYNC_GMAX_UB_TO_L1_BUF0_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP4::SYNC_GMAX_UB_TO_L1_BUF1_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP4::SYNC_GMAX_UB_TO_L1_BUF2_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP4::SYNC_GMAX_UB_TO_L1_BUF3_FLAG);
        }
    }

    __aicore__ inline void ReleaseSyncFlags()
    {
        if ASCEND_IS_AIC {
            // 核间
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2 + 16);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[0]);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[0] + 16);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[1]);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[1] + 16);
        } else {
            // 核间
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK);

            // mask2index
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);

            // vec
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP4::SYNC_P_BUF0_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP4::SYNC_P_BUF1_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP4::SYNC_ATTNOUT_BUF_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP4::SYNC_GMAX_UB_TO_L1_BUF0_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP4::SYNC_GMAX_UB_TO_L1_BUF1_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP4::SYNC_GMAX_UB_TO_L1_BUF2_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP4::SYNC_GMAX_UB_TO_L1_BUF3_FLAG);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::Resource<ArchTag> resource;
    /*
    tiling info, which are const in each kernel launch
    */
    // basic shape info
    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t firstBatchTaskNum_;
    uint32_t totalTaskNum_;
    uint32_t coreTaskNum_; // 向下取整: totalTaskNum_ / blockDim
    uint32_t blockShapeX_;
    uint32_t blockShapeY_;
    float scaleValue_;
    float log2Cx_;
    float log2CxCeil_;
    // mask2idx tile info
    uint32_t xBlockNumAligned_;
    uint32_t yBlockNumAligned_;
    uint32_t avgRowPerSubCore_;
    uint32_t preActiveSubCoreNum_;
    // base tile info
    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    // whether actual seqlen is provided
    uint32_t actSeqAval_;
    // workspace size
    uint64_t sparseIdxSize_;
    // aligned seqlen q & kv
    int64_t qSeqlenAligned_;
    int64_t kvSeqlenAligned_;
    // ===== stride 族（O / V 数据 / V-scale）=====
    int64_t strideO_;
    int64_t strideOB_;
    int64_t strideON_;
    int64_t strideOS_;
    // ===== V 数据 / V-scale stride（PV 稀疏 gather 用，operator() 内预算一次）=====
    // V 数据为 fp4，sizeof(fp4x2_e2m1_t)=1，元素偏移即字节偏移
    int64_t strideKVData_ = 0;
    int64_t strideKVDataB_ = 0;
    int64_t strideKVDataN_ = 0;
    // V-scale 为 E8M0(1 字节)，GM 布局 [B,N2,S2//64,D,2]；B16 视图下每 64-S2 行 group 占 D 个 B16 = 2D 字节
    int64_t vScaleStrideB_ = 0;
    int64_t vScaleStrideN_ = 0;
    // ===== K-scale / Q / Q-scale stride（QK Mm1 用；K-scale Host 沿 D）=====
    int64_t kScaleStrideB_ = 0;
    int64_t kScaleStrideN_ = 0;
    int64_t strideQData_ = 0;
    int64_t strideQDataB_ = 0;
    int64_t strideQDataN_ = 0;
    int64_t qScaleStrideB_ = 0;
    int64_t qScaleStrideN_ = 0;

    // ---- CreateTileInfo 持久状态 (跨 tileGroup 保持 ----
    uint32_t kvsFirstTileStartVecCore_ = 0; // task 首 tile 时切换 (loop % 2)
    uint32_t updateScaleIdx_ = 3;           //  dm idx
    uint32_t tileMaxIdx_ = 3;               // 四缓冲索引 (0~3)
    uint32_t pscaleNum_ = 19;               // 20缓冲索引 (0~19)
};

} // namespace Mxfp4Kernel

namespace Mxfp4 {

template <class OType, Mxfp4Kernel::Format Layout, Catlass::Epilogue::MXQuantMode QuantMode, bool Sparse>
__aicore__ inline void Run(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR blockSparseMask, GM_ADDR actualSeqLengths,
    GM_ADDR actualSeqLengthsKv, GM_ADDR qDequantScale, GM_ADDR kDequantScale, GM_ADDR vDequantScale,
    GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR lse, GM_ADDR tiling)
{
    using namespace Catlass;
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementQ = fp4x2_e2m1_t;
    using ElementK = fp4x2_e2m1_t;
    using ElementV = fp4x2_e2m1_t;
    using ElementS = half;
    using ElementP = fp4x2_e2m1_t;
    using ElementScale = uint8_t;
    using ElementOTmp = float;
    using ElementDm = float;
    using ElementGroupMax = half;
    using ElementLocalGlobalMax = half;

    using LayoutS = Catlass::layout::RowMajor;
    using LayoutPDummy = Catlass::layout::zN;
    using LayoutPScale = Catlass::layout::RowMajor;
    using LayoutV = Catlass::layout::RowMajor;
    using LayoutO = Catlass::layout::RowMajor;
    using LayoutOTmp = Catlass::layout::RowMajor;
    using LayoutLocalGlobalMax = Catlass::layout::RowMajor;

    using DispatchPolicyMask2Idx = Epilogue::EpilogueMask2Idx;
    using EpilogueMask2Idx = Epilogue::Block::BlockEpilogue<DispatchPolicyMask2Idx, uint8_t, int32_t, int32_t>;

    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadAtlasA5QkMx<true>;
    using TileCopyQK = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementQ, Catlass::layout::RowMajor, ElementK, Catlass::layout::ColumnMajor, ElementS, LayoutS, void,
        Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_N, false, Catlass::Gemm::Tile::ScaleGranularity::PER_GROUP>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<
        DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK, ElementS, void, TileCopyQK>;

    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueOnlineSoftmaxMx<true, QuantMode>;
    using PType = Catlass::Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Catlass::Gemm::GemmType<ElementS, LayoutS>;
    using PScaleType = Catlass::Gemm::GemmType<ElementScale, LayoutPScale>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, PScaleType>;

    using DispatchPolicyComputePScale = Epilogue::EpilogueComputePScaleMx<true, QuantMode>;
    using EpilogueComputePScale =
        Epilogue::Block::BlockEpilogue<DispatchPolicyComputePScale, ElementGroupMax, ElementDm, PScaleType>;
    using DispatchPolicyCopyGlobalMaxUbToL1 = Epilogue::EpilogueCopyGlobalMaxUbToL1Mx;
    using EpilogueCopyGlobalMaxUbToL1 =
        Epilogue::Block::BlockEpilogue<DispatchPolicyCopyGlobalMaxUbToL1, ElementLocalGlobalMax, LayoutLocalGlobalMax>;

    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::MmadAtlasA5PvMx<true>;
    using TileCopyPV = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp, void,
        Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M, false, Catlass::Gemm::Tile::ScaleGranularity::PER_GROUP>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<
        DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV, ElementOTmp, void, TileCopyPV>;

    using BlockMmadCopyGlobalMaxL1ToUB = Gemm::Block::BlockMmadTla<
        Gemm::CopyGlobalMaxL1ToUb, void, void, ElementLocalGlobalMax, void, void, void, void, void>;
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA5RescaleOMx<
        Epilogue::LseMode::NONE,
        Layout == Mxfp4Kernel::Format::TND ?
            Epilogue::LseFormat::TN1 :
            (Layout == Mxfp4Kernel::Format::BNSD ? Epilogue::LseFormat::BNS1 : Epilogue::LseFormat::BSN1),
        true>;
    using TileCopyRescaleO = Catlass::Epilogue::Tile::TileCopyRescaleO<ArchTag, OType, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<
        DispatchPolicyRescaleO, OType, ElementOTmp, ElementDm, TileCopyRescaleO, Catlass::Arch::PositionL0C>;

    using Kernel = Mxfp4Kernel::Mxfp4FullQuantKernel<
        EpilogueMask2Idx, BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, EpilogueComputePScale,
        EpilogueCopyGlobalMaxUbToL1, BlockMmadCopyGlobalMaxL1ToUB, Layout, Layout, true, Sparse>;

    (void)lse;
    Mxfp4Kernel::Mxfp4KernelParams params{
        query,         key,           value,         blockSparseMask, actualSeqLengths, actualSeqLengthsKv,
        qDequantScale, kDequantScale, vDequantScale, attentionOut,    workspace,        tiling};
    Kernel kernel;
    kernel(params);
}

} // namespace Mxfp4

#endif // MXFP4_KERNEL_HPP
