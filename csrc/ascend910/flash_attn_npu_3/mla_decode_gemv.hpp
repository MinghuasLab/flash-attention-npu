/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

// Absorbed-MLA decode GEMV kernel.
//
// The generic FAInfer path walks the KV cache in 512-token stacks and carries
// per-stack descriptors, append-KV copyback, paged addressing and mask state.
// For the canonical decode shape (Sq == 1, one KV head, QK = 512 latent + 64
// rope, Dv = 512) that machinery dominates: the whole task is a single
// M = numHeads GEMM chain over KV, so the 512-row stack is pure overhead.
//
// This kernel keeps the proven BlockMmadQK / BlockMmadPV / online-softmax /
// rescale epilogues but walks the cache in 128-token tiles and drops the
// descriptor/appends/paged/mask branches. It is selected from
// fwd_dispatch_impl.hpp before the FAInfer dispatch and never affects the
// non-MLA path.

#pragma once

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "combine_scale.hpp"
#include "fa_block.h"
#include "kernel_common.hpp"
#include "kernel_operator.h"
#include "online_softmax.hpp"
#include "pv_matmul.hpp"
#include "qk_matmul.hpp"
#include "rescale_o.hpp"
#include "tilingdata.h"

using namespace Catlass;
using namespace KernelCommon;

namespace SplitFuse {

constexpr uint32_t MLA_DECODE_KV_TILE = 128;

constexpr uint32_t MLA_DECODE_ROPE_DIM = 64;
constexpr uint32_t MLA_DECODE_LATENT_DIM = 512;
constexpr uint32_t MLA_DECODE_QK_DIM = MLA_DECODE_ROPE_DIM + MLA_DECODE_LATENT_DIM;

template <class DispatchPolicyT, class OutputType_, class InputType_, class UpdateType_, class LseType_>
class MlaDecodePersistRescale {
  public:
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;
    using ElementUpdate = typename UpdateType_::Element;
    using ElementLse = typename LseType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutUpdate = typename UpdateType_::Layout;
    using LayoutLse = typename LseType_::Layout;

    using ArchTag = typename DispatchPolicyT::ArchTag;

    static constexpr Catlass::Epilogue::LseModeT LSE_MODE = DispatchPolicyT::LSE_MODE;

    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t MAX_UB_O_ELEM_NUM = 8192;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr uint32_t SIZE_OF_16BIT = 2;

    // AtlasA2（910B）UB 实为 192KB（12×16KB），持久区必须落在 12 块内。
    // 持久 O 累加器：8×16KB 起 32KB（每子核 16 行 × 512 embed × fp32 = 32KB）。
    static constexpr uint32_t PERSIST_GO_UB_OFFSET = 8 * UB_UINT8_BLOCK_SIZE;
    static constexpr uint32_t PERSIST_GO_UB_BYTES = 2 * UB_UINT8_BLOCK_SIZE;
    // lo 双缓冲：4×16KB 起 64KB，每槽 16 行 × 512 embed × fp32 = 32KB。
    static constexpr uint32_t LO_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
    static constexpr uint32_t LO_SLOT_UB_BYTES = 16 * 512 * sizeof(float);
    // 末段 cast 输出（fp16）：8×16KB 起，与 goPersist 同址原地 cast fp32→fp16
    // （共享版 rescale_o.hpp 已验证该同址模式）。
    static constexpr uint32_t GO16_UB_TENSOR_OFFSET = 8 * UB_UINT8_BLOCK_SIZE;

    // 复用 online_softmax 写入的 UB 状态区（与共享 rescale_o.hpp 一致）。
    static constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;
    static constexpr uint32_t GM_UB_TENSOR_OFFSET =
        10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE + 2 * REPEAT_SIZE_IN_BYTE;
    static constexpr uint32_t GL_UB_TENSOR_OFFSET =
        10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE + 5 * REPEAT_SIZE_IN_BYTE;
    static constexpr uint32_t DM_UB_TENSOR_OFFSET =
        10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE + 8 * REPEAT_SIZE_IN_BYTE;
    static constexpr uint32_t LSE_STAGING_ELEMENTS = FLOAT_VECTOR_SIZE * (1 + FLOAT_BLOCK_SIZE);
    static constexpr uint32_t LSE_STAGING_UB_OFFSET = 11 * UB_UINT8_BLOCK_SIZE - LSE_STAGING_ELEMENTS * sizeof(float);

    struct TaskPlan {
        uint32_t rowNumTile;
        uint32_t subBlockIdx;
        uint32_t qNSplitSubBlock;
        uint32_t qNThisSubBlock;
        uint32_t inRowActualThisSubBlock;
        uint32_t inRowOffsetThisSubBlock;
        uint32_t qSThisSubBlock;
        uint32_t rowLoop;
        uint32_t needRowLoop;
    };

    __aicore__ inline MlaDecodePersistRescale() {}

    __aicore__ inline ~MlaDecodePersistRescale() {}

    __aicore__ inline static TaskPlan MakeTaskPlan(uint32_t rowNum, uint32_t qSBlockSize, uint32_t qNBlockSize,
                                                   uint32_t embedRoundV)
    {
        TaskPlan plan{};
        const uint32_t maxRowNumPerLoop = MAX_UB_O_ELEM_NUM / embedRoundV;
        plan.rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        plan.subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        plan.qNSplitSubBlock = qNBlockSize / subBlockNum;
        plan.qNThisSubBlock =
            (qNBlockSize == 1U) ? 0U
                                : (plan.subBlockIdx == 1U ? qNBlockSize - plan.qNSplitSubBlock : plan.qNSplitSubBlock);
        const uint32_t inRowSplitSubBlock =
            (qNBlockSize == 1U) ? qSBlockSize / subBlockNum : qSBlockSize * plan.qNSplitSubBlock;
        plan.inRowActualThisSubBlock = (plan.subBlockIdx == 1U) ? rowNum - inRowSplitSubBlock : inRowSplitSubBlock;
        plan.inRowOffsetThisSubBlock = plan.subBlockIdx * inRowSplitSubBlock;
        plan.qSThisSubBlock = (qNBlockSize == 1U) ? plan.inRowActualThisSubBlock : qSBlockSize;
        plan.rowLoop = plan.inRowActualThisSubBlock == 0U ? 0U : CeilDiv(plan.inRowActualThisSubBlock, plan.rowNumTile);
        plan.needRowLoop = plan.rowLoop > 1U ? 1U : 0U;
        return plan;
    }

    __aicore__ inline void init(Arch::Resource<ArchTag>& resource)
    {
        loUbTensor = resource.ubBuf.template GetBufferByByte<float>(LO_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO16_UB_TENSOR_OFFSET);
        goPersistUbTensor = resource.ubBuf.template GetBufferByByte<float>(PERSIST_GO_UB_OFFSET);
        lseStagingUbTensor = resource.ubBuf.template GetBufferByByte<float>(LSE_STAGING_UB_OFFSET);
        static_assert(PERSIST_GO_UB_OFFSET + PERSIST_GO_UB_BYTES <= 12 * UB_UINT8_BLOCK_SIZE,
                      "persistent O accumulator must fit inside 192KB UB");
        static_assert(LO_UB_TENSOR_OFFSET + 2 * LO_SLOT_UB_BYTES <= PERSIST_GO_UB_OFFSET,
                      "double-buffered lo must not reach the persistent O region");
    }

    __aicore__ inline void SetMask(int32_t len)
    {
        uint64_t temp = static_cast<uint64_t>(len) % static_cast<uint64_t>(FLOAT_VECTOR_SIZE);
        uint64_t mask = (uint64_t{1} << temp) - uint64_t{1};

        if (len == VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    __aicore__ inline void CopyOToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t proTokenIdx,
                                     uint32_t proTokenNum, uint32_t epiTokenNum, uint32_t integralHeadNum,
                                     uint32_t qSThisSubBlock, uint32_t embed, uint32_t embedRound,
                                     uint32_t outputStride, uint32_t oHeadStride)
    {
        uint32_t sourceRow = 0U;
        uint32_t fullHeadIdx = 0U;
        uint32_t remaining = proTokenNum + integralHeadNum * qSThisSubBlock + epiTokenNum;
        while (remaining != 0U) {
            uint32_t copyRows;
            uint32_t gmOffset;
            if (proTokenNum != 0U && sourceRow == 0U) {
                copyRows = proTokenNum;
                gmOffset = proTokenIdx * outputStride;
            } else if (fullHeadIdx < integralHeadNum) {
                copyRows = qSThisSubBlock;
                gmOffset = ((proTokenNum != 0U ? 1U : 0U) + fullHeadIdx) * oHeadStride;
                ++fullHeadIdx;
            } else {
                copyRows = epiTokenNum;
                gmOffset = ((proTokenNum != 0U ? 1U : 0U) + integralHeadNum) * oHeadStride;
            }
            AscendC::DataCopyPad(gOutput[gmOffset], goUbTensor16[sourceRow * embedRound],
                                 AscendC::DataCopyExtParams(copyRows, embed * SIZE_OF_16BIT, 0,
                                                            (outputStride - embed) * SIZE_OF_16BIT, 0));
            sourceRow += copyRows;
            remaining -= copyRows;
        }
    }

    // FD 变体：把未归一化 fp32 持久 O 写入 partial workspace，CombineScale 按 w_s 加权归一。
    // 布局与 fp16 版一致仅元素宽度不同，decode 下每次只写 1 行，目的端 stride 不参与。
    __aicore__ inline void CopyOToGmFp32(AscendC::GlobalTensor<float> gOutput, uint32_t proTokenIdx,
                                         uint32_t proTokenNum, uint32_t epiTokenNum, uint32_t integralHeadNum,
                                         uint32_t qSThisSubBlock, uint32_t embed, uint32_t embedRound,
                                         uint32_t outputStride, uint32_t oHeadStride)
    {
        uint32_t sourceRow = 0U;
        uint32_t fullHeadIdx = 0U;
        uint32_t remaining = proTokenNum + integralHeadNum * qSThisSubBlock + epiTokenNum;
        while (remaining != 0U) {
            uint32_t copyRows;
            uint32_t gmOffset;
            if (proTokenNum != 0U && sourceRow == 0U) {
                copyRows = proTokenNum;
                gmOffset = proTokenIdx * outputStride;
            } else if (fullHeadIdx < integralHeadNum) {
                copyRows = qSThisSubBlock;
                gmOffset = ((proTokenNum != 0U ? 1U : 0U) + fullHeadIdx) * oHeadStride;
                ++fullHeadIdx;
            } else {
                copyRows = epiTokenNum;
                gmOffset = ((proTokenNum != 0U ? 1U : 0U) + integralHeadNum) * oHeadStride;
            }
            AscendC::DataCopyPad(gOutput[gmOffset], goPersistUbTensor[sourceRow * embedRound],
                                 AscendC::DataCopyExtParams(copyRows, embed * sizeof(float), 0, 0, 0));
            sourceRow += copyRows;
            remaining -= copyRows;
        }
    }

    __aicore__ inline void CopyLseToGm(AscendC::GlobalTensor<ElementLse> gLse, const LayoutLse& layoutLse,
                                       uint32_t qNThisSubBlock, uint32_t qSThisSubBlock, uint32_t totalRowNum)
    {
        const uint32_t headCount = qNThisSubBlock == 0U ? 1U : qNThisSubBlock;
        const uint32_t rowsPerHead = qNThisSubBlock == 0U ? totalRowNum : qSThisSubBlock;
        uint32_t headStride = layoutLse.stride(0);
        if (headCount == 1U || headStride == rowsPerHead) {
            AscendC::DataCopyPad(gLse, lseStagingUbTensor,
                                 AscendC::DataCopyExtParams(1, totalRowNum * sizeof(float), 0, 0, 0));
            return;
        }
        // Brcb 已把每个 LSE 标量放到独立 32B UB 块；逐 head 四字节搬运。
        uint32_t sourceOffset = FLOAT_VECTOR_SIZE;
        uint64_t gmOffset = 0U;
        for (uint32_t head = 0U; head < headCount; ++head) {
            AscendC::DataCopyPad(gLse[gmOffset], lseStagingUbTensor[sourceOffset],
                                 AscendC::DataCopyExtParams(rowsPerHead, sizeof(float), 0, 0, 0));
            sourceOffset += rowsPerHead * FLOAT_BLOCK_SIZE;
            gmOffset += headStride;
        }
    }

    __aicore__ inline void
    SubCoreCompute(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementInput> gInput,
                   AscendC::GlobalTensor<ElementLse> gLse, const LayoutOutput& layoutOutput,
                   const LayoutInput& layoutInput, const LayoutLse& layoutLse, uint32_t qNThisSubBlock,
                   uint32_t qSThisSubBlock, uint32_t totalRowNum, uint32_t isFirstStackTile, uint32_t isLastStackTile,
                   uint32_t curStackTileMod, uint32_t taskStateSlot, uint32_t isLastRowLoop, uint32_t rowOffsetLoop,
                   uint32_t outputStride, uint32_t oHeadStride, bool fdMode, AscendC::GlobalTensor<float> gPartialO)
    {
        uint32_t curRowNum = layoutInput.shape(0);
        uint32_t embed = layoutInput.shape(1);
        uint32_t embedRound = layoutInput.stride(0);
        uint32_t curRowNumRound = RoundUp(curRowNum, FLOAT_BLOCK_SIZE);
        uint32_t dmUbOffsetCurStackTile = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffsetLoop;
        uint32_t stateRowOffset = taskStateSlot * 64 + rowOffsetLoop;
        auto lseUbTensor = lseStagingUbTensor;
        auto lseBroadcastUbTensor = lseUbTensor[FLOAT_VECTOR_SIZE];
        const uint32_t loSlotOffset = (curStackTileMod % 2U) * (LO_SLOT_UB_BYTES / sizeof(float));

        if (isFirstStackTile) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            // *** go = lo：首段 OTmp 直接入持久区
            AscendC::DataCopy(goPersistUbTensor, gInput,
                              AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        } else {
            // lo 双缓冲：本段读 lo[slot%2]，与上一段 V 用 lo[(slot-1)%2] 无别名，
            // 故取消共享代码的 V_MTE2 EVENT_ID3 段间串行等待。
            AscendC::DataCopy(loUbTensor[loSlotOffset], gInput,
                              AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                          dmUbTensor[dmUbOffsetCurStackTile].ReinterpretCast<uint32_t>(),
                          curRowNumRound / FLOAT_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** go = go * dm_block（exp(m_old - m_new)）
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vmul_idx = 0; vmul_idx < embed / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                AscendC::Mul<float, false>(goPersistUbTensor[vmul_idx * FLOAT_VECTOR_SIZE],
                                           goPersistUbTensor[vmul_idx * FLOAT_VECTOR_SIZE], tvUbTensor, (uint64_t)0,
                                           curRowNum,
                                           AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                       embedRound / FLOAT_BLOCK_SIZE, 1));
            }
            if (embed % FLOAT_VECTOR_SIZE > 0) {
                SetMask(embed % FLOAT_VECTOR_SIZE);
                AscendC::Mul<float, false>(goPersistUbTensor[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                           goPersistUbTensor[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE], tvUbTensor,
                                           (uint64_t)0, curRowNum,
                                           AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                       embedRound / FLOAT_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // *** go = lo + go
            AscendC::Add<float, false>(goPersistUbTensor, goPersistUbTensor, loUbTensor[loSlotOffset], (uint64_t)0,
                                       (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                       AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (!isLastStackTile) {
            // go 驻留 UB，跳过 gUpdate 的 GM 写回（D1）。
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        } else {
            // 仅末段统一写回：go = go / gl，cast fp16，写 gO；LSE = ln(gl) + gm。
            uint32_t proTokenIdx = rowOffsetLoop % qSThisSubBlock;
            uint32_t proTokenNum = AscendC::Std::min(curRowNum, (qSThisSubBlock - proTokenIdx)) % qSThisSubBlock;
            uint32_t integralHeadNum = (curRowNum - proTokenNum) / qSThisSubBlock;
            uint32_t epiTokenNum = curRowNum - proTokenNum - integralHeadNum * qSThisSubBlock;
            if (fdMode) {
                // *** gl_block = expand_to_block(gl)
                AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                              glUbTensor.ReinterpretCast<uint32_t>()[stateRowOffset], curRowNumRound / FLOAT_BLOCK_SIZE,
                              AscendC::BrcbRepeatParams(1, 8));
                AscendC::PipeBarrier<PIPE_V>();
                // *** go = go / gl_block
                // FD 的 partial O 需 split 内归一化（同 rescale_o.hpp），权重已含全局 1/Σ。
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                for (uint32_t vdiv_idx = 0; vdiv_idx < embed / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
                    AscendC::Div<float, false>(goPersistUbTensor[vdiv_idx * FLOAT_VECTOR_SIZE],
                                               goPersistUbTensor[vdiv_idx * FLOAT_VECTOR_SIZE], tvUbTensor, (uint64_t)0,
                                               curRowNum,
                                               AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                           embedRound / FLOAT_BLOCK_SIZE, 1));
                }
                if (embed % FLOAT_VECTOR_SIZE > 0) {
                    SetMask(embed % FLOAT_VECTOR_SIZE);
                    AscendC::Div<float, false>(goPersistUbTensor[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                               goPersistUbTensor[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                               tvUbTensor, (uint64_t)0, curRowNum,
                                               AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                           embedRound / FLOAT_BLOCK_SIZE, 1));
                    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                CopyOToGmFp32(gPartialO, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum, qSThisSubBlock, embed,
                              embedRound, outputStride, oHeadStride);
            } else {
                // *** gl_block = expand_to_block(gl)
                AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                              glUbTensor.ReinterpretCast<uint32_t>()[stateRowOffset], curRowNumRound / FLOAT_BLOCK_SIZE,
                              AscendC::BrcbRepeatParams(1, 8));
                AscendC::PipeBarrier<PIPE_V>();
                // *** go = go / gl_block
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                for (uint32_t vdiv_idx = 0; vdiv_idx < embed / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
                    AscendC::Div<float, false>(goPersistUbTensor[vdiv_idx * FLOAT_VECTOR_SIZE],
                                               goPersistUbTensor[vdiv_idx * FLOAT_VECTOR_SIZE], tvUbTensor, (uint64_t)0,
                                               curRowNum,
                                               AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                           embedRound / FLOAT_BLOCK_SIZE, 1));
                }
                if (embed % FLOAT_VECTOR_SIZE > 0) {
                    SetMask(embed % FLOAT_VECTOR_SIZE);
                    AscendC::Div<float, false>(goPersistUbTensor[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                               goPersistUbTensor[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                               tvUbTensor, (uint64_t)0, curRowNum,
                                               AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                           embedRound / FLOAT_BLOCK_SIZE, 1));
                    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                }
                AscendC::PipeBarrier<PIPE_V>();
                // *** go = castfp32to16(go)
                if (std::is_same<ElementOutput, bfloat16_t>::value) {
                    AscendC::Cast<ElementOutput, float, false>(
                        goUbTensor16, goPersistUbTensor, AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                        (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                        AscendC::UnaryRepeatParams(1, 1, 4, 8));
                } else {
                    AscendC::Cast<ElementOutput, float, false>(
                        goUbTensor16, goPersistUbTensor, AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                        (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                        AscendC::UnaryRepeatParams(1, 1, 4, 8));
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                // *** 写 O 到 GM
                AscendC::PipeBarrier<PIPE_V>();
                CopyOToGm(gOutput, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum, qSThisSubBlock, embed,
                          embedRound, outputStride, oHeadStride);
            }
            if constexpr (LSE_MODE == Catlass::Epilogue::LseModeT::OUT_ONLY) {
                if (isLastRowLoop) {
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Ln<float, false>(lseUbTensor, glUbTensor[taskStateSlot * 64], (uint64_t)0,
                                              CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                                              AscendC::UnaryRepeatParams(1, 1, 8, 8));

                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<float, false>(lseUbTensor, lseUbTensor, gmUbTensor[taskStateSlot * 64], (uint64_t)0,
                                               CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                                               AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Brcb(lseBroadcastUbTensor.ReinterpretCast<uint32_t>(),
                                  lseUbTensor.ReinterpretCast<uint32_t>(), CeilDiv(totalRowNum, FLOAT_BLOCK_SIZE),
                                  AscendC::BrcbRepeatParams(1, 8));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    CopyLseToGm(gLse, layoutLse, qNThisSubBlock, qSThisSubBlock, totalRowNum);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        }
    }

    __aicore__ inline void
    operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementInput> gInput,
               AscendC::GlobalTensor<ElementUpdate> gUpdate, AscendC::GlobalTensor<ElementLse> gLse,
               const LayoutOutput& layoutOutput, const LayoutInput& layoutInput, const LayoutUpdate& layoutUpdate,
               const LayoutLse& layoutLse, GemmCoord actualBlockShape, const TaskPlan& taskPlan,
               Arch::CrossCoreFlag pvReady, uint32_t qSBlockSize, uint32_t qNBlockSize, uint32_t isFirstStackTile,
               uint32_t isLastStackTile, uint32_t curStackTileMod, uint32_t taskStateSlot, uint32_t outputStride,
               uint32_t oHeadStride, bool fdMode, AscendC::GlobalTensor<float> gPartialO)
    {
        (void)gUpdate;
        (void)layoutUpdate;
        uint32_t rowNum = actualBlockShape.m();
        uint32_t embed = actualBlockShape.n();
        const uint32_t subBlockIdx = taskPlan.subBlockIdx;
        const uint32_t qNSplitSubBlock = taskPlan.qNSplitSubBlock;
        const uint32_t qNThisSubBlock = taskPlan.qNThisSubBlock;
        const uint32_t inRowActualThisSubBlock = taskPlan.inRowActualThisSubBlock;
        const uint32_t inRowOffsetThisSubBlock = taskPlan.inRowOffsetThisSubBlock;
        uint32_t outHeadOffsetThisSubBlock = (qNBlockSize == 1U) ? 0 : subBlockIdx * qNSplitSubBlock;
        uint32_t qSThisSubBlock = taskPlan.qSThisSubBlock;
        int64_t outOffsetSubBlock = static_cast<int64_t>(outHeadOffsetThisSubBlock) * oHeadStride;

        uint32_t outLseRowOffsetThisSubBlock = (qNBlockSize == 1U) ? 0 : subBlockIdx * qNSplitSubBlock;
        uint32_t outLseColOffsetThisSubBlock = (qNBlockSize == 1U) ? inRowOffsetThisSubBlock : 0;
        int64_t offsetLse = layoutLse.GetOffset(MatrixCoord(outLseRowOffsetThisSubBlock, outLseColOffsetThisSubBlock));
        auto gLseThisSubBlock = gLse[offsetLse];
        auto layoutOutLseThisSubBlock = layoutLse;

        // 空闲子核必须消费 pv-ready，避免残留 flag 影响下一任务。
        if (inRowActualThisSubBlock == 0U) {
            Arch::CrossCoreWaitFlag(pvReady);
            return;
        }

        uint32_t rowLoop = taskPlan.rowLoop;
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoop; rowLoopIdx++) {
            uint32_t rowOffsetLoop = rowLoopIdx * taskPlan.rowNumTile;
            uint32_t rowActualCurLoop = (rowLoopIdx == (rowLoop - 1U))
                                            ? inRowActualThisSubBlock - rowLoopIdx * taskPlan.rowNumTile
                                            : taskPlan.rowNumTile;

            int64_t offsetOutput =
                static_cast<int64_t>(rowLoopIdx * taskPlan.rowNumTile / qSThisSubBlock) * oHeadStride +
                outOffsetSubBlock;
            auto gOutputCurLoop = gOutput[offsetOutput];
            auto gPartialOCurLoop = gPartialO[offsetOutput];
            auto layoutOutputCurLoop = layoutOutput;
            int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(inRowOffsetThisSubBlock + rowOffsetLoop, 0));
            auto gInputCurLoop = gInput[offsetInput];
            auto layoutInputCurLoop = layoutInput.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));

            if (rowLoopIdx == 0) {
                Arch::CrossCoreWaitFlag(pvReady);
            }
            SubCoreCompute(gOutputCurLoop, gInputCurLoop, gLseThisSubBlock, layoutOutputCurLoop, layoutInputCurLoop,
                           layoutOutLseThisSubBlock, qNThisSubBlock, qSThisSubBlock, inRowActualThisSubBlock,
                           isFirstStackTile, isLastStackTile, curStackTileMod, taskStateSlot,
                           (rowLoopIdx == rowLoop - 1U), rowOffsetLoop, outputStride, oHeadStride, fdMode,
                           gPartialOCurLoop);
        }
    }

  private:
    AscendC::LocalTensor<float> loUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> tvUbTensor;
    AscendC::LocalTensor<ElementOutput> goUbTensor16;
    AscendC::LocalTensor<float> goPersistUbTensor;
    AscendC::LocalTensor<float> lseStagingUbTensor;
};

template <class BlockMmadQK, class BlockMmadPV, class EpilogueOnlineSoftmax, class EpilogueRescaleO,
          FaiKenel::inputLayout INPUT_LAYOUT = FaiKenel::inputLayout::BSND>
class MlaDecodeGemvKernel {
  public:
    using ArchTag = typename BlockMmadQK::ArchTag;
    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutQ = typename BlockMmadQK::LayoutA;
    using ElementK = typename BlockMmadQK::ElementB;
    using LayoutK = typename BlockMmadQK::LayoutB;
    using ElementS = typename BlockMmadQK::ElementC;
    using LayoutS = typename BlockMmadQK::LayoutC;

    using ElementP = typename BlockMmadPV::ElementA;
    using LayoutP = typename BlockMmadPV::LayoutA;
    using ElementV = typename BlockMmadPV::ElementB;
    using LayoutV = typename BlockMmadPV::LayoutB;

    using ElementO = typename EpilogueRescaleO::ElementOutput;
    using LayoutO = typename EpilogueRescaleO::LayoutOutput;
    using ElementOTmp = typename EpilogueRescaleO::ElementInput;
    using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;
    using ElementLse = typename EpilogueRescaleO::ElementLse;
    using LayoutLse = typename EpilogueRescaleO::LayoutLse;
    using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;
    using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;

    // KV-split reduction (only instantiated in the multi-core path).
    using SplitInfoType = splitNode[25];
    using CombineScaleType =
        Catlass::Epilogue::Block::CombineScale<Gemm::GemmType<ElementO, LayoutO>, Gemm::GemmType<ElementLse, LayoutLse>,
                                               SplitInfoType>;

    static constexpr uint32_t STACK_SLOTS = PRE_LAUNCH + 1;

    __aicore__ inline MlaDecodeGemvKernel() {}

    __aicore__ inline void operator()(FAIKernelParams const& params)
    {
        __gm__ FAInferTilingData* tilingData =
            reinterpret_cast<__gm__ FAInferTilingData*>(params.tiling.tilingDataAddr);
        mm1OutSize = tilingData->mm1OutSize;
        smOnlineOutSize = tilingData->smOnlineOutSize;
        mm2OutSize = tilingData->mm2OutSize;
        updateSize = tilingData->UpdateSize;
        batch = tilingData->batch;
        qHeads = tilingData->numHeads;
        kvHeads = 1;                 // D2: 闸门 IsMlaDecodeGemvShape 保证 numHeadsK==1
        embed = MLA_DECODE_ROPE_DIM; // D2: 闸门保证 headSizeOg+headSizeV==576 && headSizeV==512
        embedV = MLA_DECODE_LATENT_DIM;
        embedQK = MLA_DECODE_QK_DIM;
        maxQSeqlen = tilingData->maxQSeqlen;
        kvNewSeqlen = tilingData->kvNewSeqlen;
        pagedBlockSize = tilingData->blockSize;
        scaleValue = params.tiling.scaleValue;
        softcapValue = params.tiling.softcapValue;
        totalQTokens = static_cast<uint32_t>(params.tiling.totalQTokens);
        kvSeqlenAllocated = params.kvSeqlenAllocated;

        // Both fused layouts are dense (paged input stays on the FAInfer path).
        if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::BNSD) {
            strideQ = embed;
            strideK = embed;
            strideV = embedV;
            qHeadStride = static_cast<uint64_t>(maxQSeqlen) * embed;
            strideQv = embedV;
            qvHeadStride = static_cast<uint64_t>(maxQSeqlen) * embedV;
            kvHeadStrideK = static_cast<uint64_t>(kvSeqlenAllocated) * embed;
            kvHeadStrideV = static_cast<uint64_t>(kvSeqlenAllocated) * embedV;
        } else {
            strideQ = static_cast<uint64_t>(qHeads * embed);
            strideK = static_cast<uint64_t>(kvHeads * embed);
            strideV = static_cast<uint64_t>(kvHeads * embedV);
            qHeadStride = embed;
            strideQv = static_cast<uint64_t>(qHeads * embedV);
            qvHeadStride = embedV;
            kvHeadStrideK = embed;
            kvHeadStrideV = embedV;
        }
        // O is always written as contiguous BSND (B, S, H, Dv).
        strideO = static_cast<uint64_t>(qHeads * embedV);
        oHeadStride = embedV;
        embedRoundV = RoundUp(embedV, FaiKenel::BLOCK_SIZE);

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        AscendC::GlobalTensor<ElementQ> gQv;
        gQv.SetGlobalBuffer((__gm__ ElementQ*)params.qv);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK*)params.v);
        AscendC::GlobalTensor<int32_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t*)params.actualKvseqlen);
        // Packed TND only: cu_seqlens_q / seqused_k (nullptr for BSND/BNSD).
        AscendC::GlobalTensor<int32_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int32_t*)params.actualQseqlen);
        AscendC::GlobalTensor<int32_t> gSeqUsedKv;
        gSeqUsedKv.SetGlobalBuffer((__gm__ int32_t*)params.seqUsedKv);
        const bool hasSeqUsedKv = params.seqUsedKv != nullptr;
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse*)params.lse);

        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS*)params.workSpace);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP*)(params.workSpace + mm1OutSize));
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp*)(params.workSpace + mm1OutSize + smOnlineOutSize));
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp*)(params.workSpace + mm1OutSize + smOnlineOutSize + mm2OutSize));

        // FD partials live past this kernel's scratch tables (see the host's
        // partialBase): [LSE f32][O f32], group-major, written by each core.
        const uint64_t partialBase = mm1OutSize + smOnlineOutSize + mm2OutSize + updateSize;
        AscendC::GlobalTensor<ElementLse> gLseFD;
        gLseFD.SetGlobalBuffer((__gm__ ElementLse*)(params.workSpace + partialBase));
        AscendC::GlobalTensor<ElementLse> gOFD;
        gOFD.SetGlobalBuffer((__gm__ ElementLse*)(params.workSpace + partialBase + tilingData->splitLseTotalSize));

        uint32_t coreIdx = AscendC::GetBlockIdx();
#ifdef __DAV_C220_CUBE__
        // D2：prelaunch 旗标裁剪到必要集合。AIC 侧只消费/置位 M_MTE1、FIX_M、MTE1_MTE2 的必要事件 ID；
        // 未消费的置位与尾部 Wait 成对删除，旗标记账保持平衡。
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);

        // 128-token tiles: Q/K/P keep the dynamic tile shapes the generic
        // kernel uses for the absorbed form, only the KV stack is shorter.
        const uint32_t nDynNum = L1_MAX_N_NUM;
        const uint32_t kDynNum = 2U * L1_MAX_N_NUM;
        const uint32_t qDynNum = RoundUp(embedQK, FaiKenel::BLOCK_SIZE);
        // G1：L1 里 Q 行区必须与 H 无关，常驻行数固定为 QK block 的 L1TileShape::M 而非 qHeads；
        // 否则 K/P/V 基址随 H 增长会把 V/nd 区顶出 L1 上限（512KB）致越界，FA4 同路径对齐。
        const uint32_t qRows = BlockMmadQK::L1TileShape::M;
        const uint32_t pRows = BlockMmadPV::L1TileShape::M;
        const uint32_t kPVDynNum = 2U * L1_MAX_N_NUM;
        const uint32_t vSegLen = 2U * L1_MAX_N_NUM;
        const uint32_t vSliceWidth = L1_HEAD_DIM_SLICE;
        const uint32_t l1QKSize = qRows * qDynNum * sizeof(ElementQ);
        const uint32_t ndCopyL1Offset = l1QKSize + nDynNum * kDynNum * sizeof(ElementK) * DOUBLE_BUFFER +
                                        pRows * kPVDynNum * sizeof(ElementP) * DOUBLE_BUFFER +
                                        DOUBLE_BUFFER * vSegLen * vSliceWidth * sizeof(ElementV);

        blockMmadQK.SetPingPongState(&pingPongState);
        blockMmadPV.SetPingPongState(&pingPongState);
        blockMmadQK.init(resource, nDynNum, kDynNum, qRows, qDynNum, MLA_DECODE_KV_TILE, 0, ndCopyL1Offset);
        blockMmadPV.init(resource, nDynNum, kPVDynNum, MLA_DECODE_KV_TILE, l1QKSize, DOUBLE_BUFFER, vSegLen,
                         vSliceWidth, ndCopyL1Offset);
#endif
#ifdef __DAV_C220_VEC__
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);

        epilogueOnlineSoftmax.init(resource, scaleValue, softcapValue);
        epilogueRescaleO.init(resource);
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif

        // ---- G1：大 H 走 Q 分块滚动路径。现把 head 维切成 <= kMlaDecodeGemvHeadBlock 行的块；
        // 复用与 H 无关的 L1 布局（Q 128 行 + K/P/V 各 2 槽 ≈400KB），每块退化成已验证的 <=32 行流程。
        constexpr uint32_t kMlaDecodeGemvHeadBlock = 32U;
        const uint32_t gemvNumHeadBlocks = (qHeads + kMlaDecodeGemvHeadBlock - 1U) / kMlaDecodeGemvHeadBlock;
        const uint32_t gemvHeadBlock = (qHeads + gemvNumHeadBlocks - 1U) / gemvNumHeadBlocks;
        const uint32_t qSBlockSize = 1U;
        uint32_t softmaxPingPongFlag = 0;
        uint32_t taskStateSequence = 0;

        // ---- D2：循环不变量提升。decode 几何固定，Layout/TaskPlan 只依赖任务级量，与 batch/KV 段无关；
        // K/V 布局与 workspace 基址在 head 块循环外构造一次。
        const uint64_t wsSlotBase = static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * STACK_SLOTS;
        const uint64_t wsUpdateBase = static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB;
#ifdef __DAV_C220_CUBE__
        LayoutK layoutKLatent(strideV, MLA_DECODE_KV_TILE);
        LayoutV layoutVTemp(MLA_DECODE_KV_TILE, strideV);
#endif
#ifdef __DAV_C220_VEC__
        LayoutO layoutO(qSBlockSize, qHeads * embedV);
        // BSND/BNSD LSE is (B,H,maxQSeqlen); TND LSE is (H,total_q).
        // FD partial is [split][head] (lseTaskOffset + s*qHeads), stride 1.
        const bool lseSplitActive = tilingData->splitLseTotalSize > 0U;
        LayoutLse layoutLse(qHeads, (INPUT_LAYOUT == FaiKenel::inputLayout::TND && !lseSplitActive) ? totalQTokens
                                                                                                    : maxQSeqlen);
#endif
        // fdMode is armed only when the host split KV across cores (it sets
        // splitLseTotalSize, else a whole-batch core must NOT use partial path.
        const bool fdMode = tilingData->splitLseTotalSize > 0U;
        // G1：LSE 的每 head 步长。BSND/BNSD 是 (B,H,maxQSeqlen)，TND 是 (H,total_q)，
        // FD partial 是 [split][head] 步长 1（== maxQSeqlen），用于把 head 块基址偏移到正确位置。
        const uint32_t lseHeadStride =
            (INPUT_LAYOUT == FaiKenel::inputLayout::TND && !fdMode) ? totalQTokens : maxQSeqlen;

        // ---- P0 (G2)：就地重算 task 表。宿主只下沉 totalTaskNum/nsplitPerBatch，每核按与宿主
        // splitKvForGemvDecode 逐字一致重算 (b,s,segStart,segEnd)，K/V 由 kvSegBase 读取。
        const uint32_t totalTaskNum = tilingData->totalTaskNum;
        const uint32_t nsplitPerBatch = tilingData->nsplitPerBatch;
        const uint32_t numCores = AscendC::GetBlockNum();
        const uint32_t perCore = (totalTaskNum + numCores - 1U) / numCores;
        const uint32_t taskStart = coreIdx * perCore;
        const uint32_t taskEnd = AscendC::Std::min(taskStart + perCore, totalTaskNum);

        // 每 batch 的 KV 长度由私有成员 KvLenOf 给出（与宿主 seqlens_k_cpu 及通用
        // FAInfer 的 KvSeqLenOf 口径一致）；FA3 追加 kvNewSeqlen 以对齐宿主。

        // ---- G1：head 维分块滚动。每个 head 块独立走完整（task,段）流水，Q 从 GM 按块重装进 128 行 Q 区，
        // 所以 L1 占用与 H 无关。qHeads<=32 时只有一块（headBase==0），所有 head 基址偏移恒为 0。
        for (uint32_t m0 = 0; m0 < qHeads; m0 += gemvHeadBlock) {
            // 本 head 块的块内行数（<= 32）。MakeTaskPlan 都按块内行数重建，使每块
            // 都退化成已验证的 "子核 <= 16 行 / rowLoop == 1" 流程。
            const uint32_t rowNum = AscendC::Std::min(gemvHeadBlock, qHeads - m0);
            // 非 const：blockMmadQK::loadQGM 的 qNBlockSize 形参是非 const 引用。
            uint32_t qNBlockSize = rowNum;
            const uint64_t headBase = static_cast<uint64_t>(m0);
#ifdef __DAV_C220_CUBE__
            LayoutQ layoutQ(rowNum, embedQK);
            LayoutOTmp layoutOTmp(rowNum, embedV, embedRoundV);
#endif
#ifdef __DAV_C220_VEC__
            typename EpilogueOnlineSoftmax::TaskPlan softmaxTaskPlan =
                EpilogueOnlineSoftmax::MakeTaskPlan(rowNum, qSBlockSize, qNBlockSize);
            typename EpilogueRescaleO::TaskPlan rescaleTaskPlan =
                EpilogueRescaleO::MakeTaskPlan(rowNum, qSBlockSize, qNBlockSize, embedRoundV);
            LayoutOTmp layoutOTmp(rowNum, embedV, embedRoundV);
            LayoutUpdate layoutUpdate(rowNum, embedV, embedRoundV);
#endif

            // ---- 定位起点（每核一次，O(batch)）：沿 batch 累加 task 数与 partial 基址，
            // 定位 taskStart 所在 batch 的 split；b/s 与 lseBase/oBase 成 running 状态。
            uint32_t b = 0;
            uint32_t t = 0;
            uint32_t kvLenB = 0;
            uint32_t segCountB = 0;
            uint32_t blocksPerSplit = 1U;
            uint32_t actualSplits = 0U;
            int64_t lseBase = 0;
            int64_t oBase = 0;
            for (; b < batch; ++b) {
                kvLenB = KvLenOf(b, gActualKvseqlen, gSeqUsedKv, hasSeqUsedKv) + kvNewSeqlen;
                segCountB = CeilDiv(kvLenB, MLA_DECODE_KV_TILE);
                if (segCountB == 0U) {
                    continue;
                }
                blocksPerSplit = CeilDiv(segCountB, nsplitPerBatch);
                if (blocksPerSplit == 0U) {
                    blocksPerSplit = 1U;
                }
                actualSplits = CeilDiv(segCountB, blocksPerSplit);
                if (t + actualSplits > taskStart) {
                    break;
                }
                t += actualSplits;
                lseBase += static_cast<int64_t>(actualSplits) * qHeads;
                oBase += static_cast<int64_t>(actualSplits) * qHeads * embedV;
            }
            uint32_t s = taskStart - t;

            for (uint32_t taskIdx = taskStart; taskIdx < taskEnd; ++taskIdx) {
                const uint32_t BIdx = b;
                // 本 task 覆盖本 batch 的 [segStart,segEnd) 段，与宿主
                // splitKvForGemvDecode 的 blocksPerSplit/actualSplits 规则逐字一致。
                const uint32_t segStart = s * blocksPerSplit;
                const uint32_t segEnd = AscendC::Std::min(segStart + blocksPerSplit, segCountB);
                const uint32_t kvSegBase = segStart * MLA_DECODE_KV_TILE;
                const uint32_t kvSeqlenLocal =
                    (segEnd > segStart) ? (AscendC::Std::min(kvLenB, segEnd * MLA_DECODE_KV_TILE) - kvSegBase) : 0U;
                // blockMmadPV 的 nIdx/nLoop/blockSize 形参是非 const 引用，必须传左值；
                // 本 kernel 每段恰好一个 KV tile，仅需满足签名约束。
                uint32_t numSeg = (segEnd > segStart) ? (segEnd - segStart) : 0U;

                uint64_t qBOffset;
                uint64_t qvBOffset;
                uint64_t kBOffset;
                uint64_t vBOffset;
                uint64_t gmOffsetO;
                uint64_t gmOffsetLse;
                if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::BNSD) {
                    qBOffset = static_cast<uint64_t>(BIdx) * qHeads * qHeadStride;
                    qvBOffset = static_cast<uint64_t>(BIdx) * qHeads * qvHeadStride;
                    kBOffset = static_cast<uint64_t>(BIdx) * kvHeads * kvHeadStrideK +
                               static_cast<uint64_t>(kvSegBase) * strideK;
                    vBOffset = static_cast<uint64_t>(BIdx) * kvHeads * kvHeadStrideV +
                               static_cast<uint64_t>(kvSegBase) * strideV;
                    gmOffsetO = static_cast<uint64_t>(BIdx) * maxQSeqlen * strideO;
                    gmOffsetLse = static_cast<uint64_t>(BIdx) * maxQSeqlen * qHeads;
                } else if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                    // TND: out (total_q,H,Dv), LSE (H,total_q), head stride is
                    // totalQTokens; per-seq bases come from cu_seqlens.
                    const uint64_t prevQSeqlenSum = static_cast<uint64_t>(gActualQseqlen.GetValue(BIdx));
                    const uint64_t prevKvSeqlenSum = static_cast<uint64_t>(gActualKvseqlen.GetValue(BIdx));
                    qBOffset = prevQSeqlenSum * strideQ;
                    qvBOffset = prevQSeqlenSum * strideQv;
                    kBOffset = (prevKvSeqlenSum + static_cast<uint64_t>(kvSegBase)) * strideK;
                    vBOffset = (prevKvSeqlenSum + static_cast<uint64_t>(kvSegBase)) * strideV;
                    gmOffsetO = prevQSeqlenSum * strideO;
                    gmOffsetLse = prevQSeqlenSum;
                } else {
                    const uint64_t prevQSeqlenSum = static_cast<uint64_t>(BIdx) * maxQSeqlen;
                    const uint64_t prevKvSeqlenSum = static_cast<uint64_t>(BIdx) * kvSeqlenAllocated;
                    qBOffset = prevQSeqlenSum * strideQ;
                    qvBOffset = prevQSeqlenSum * strideQv;
                    kBOffset = (prevKvSeqlenSum + static_cast<uint64_t>(kvSegBase)) * strideK;
                    vBOffset = (prevKvSeqlenSum + static_cast<uint64_t>(kvSegBase)) * strideV;
                    gmOffsetO = static_cast<uint64_t>(BIdx) * maxQSeqlen * strideO;
                    gmOffsetLse = static_cast<uint64_t>(BIdx) * maxQSeqlen * qHeads;
                }
                const uint64_t gmOffsetQ = qBOffset + headBase * qHeadStride;
                const uint64_t gmOffsetQv = qvBOffset + headBase * qvHeadStride;
                const uint64_t gmOffsetK = kBOffset;
                const uint64_t gmOffsetV = vBOffset;
                // O 的输出由 epilogue 用块内 head 索引 + oHeadStride 写，所以这里把张量
                // 基址抬到本 head 块起点（direct / FD 同一个 oHeadStride）。
                const uint64_t gmOffsetOHead = gmOffsetO + headBase * oHeadStride;

#ifdef __DAV_C220_CUBE__
                blockMmadQK.resetBlockStart(0U, pagedBlockSize);
                blockMmadQK.SetMlaLatentDim(embedV);
                blockMmadQK.SetMlaRopeQ(gQ[gmOffsetQ], static_cast<uint32_t>(qHeadStride),
                                        static_cast<uint32_t>(strideQ));
                blockMmadQK.SetMlaRopeK(gK[gmOffsetK], static_cast<uint32_t>(strideK), 0);
                blockMmadQK.loadQGM(gQv[gmOffsetQv], layoutQ, rowNum, qNBlockSize, static_cast<uint32_t>(qvHeadStride),
                                    static_cast<uint32_t>(strideQv));
                // P0 (G2)：Q L1 槽位回补挪到本 task 段循环之后（见下）。此处立刻 SetFlag 会让
                // loadQGM（MTE2 写 Q）抢在末次 copyL1ToL0A（MTE1 读 Q）前，构成 WAR hazard。
#endif
                // The running max/sum state must persist across all KV tiles
                // of one task, so its slot is per batch and not per tile.
                const uint32_t stateSlot = BIdx % STACK_SLOTS;
                (void)taskStateSequence;

                // 段间软件流水（P1）：cube 的 K/V 搬运与 MMA、vector softmax/rescale 跨段重叠，
                // ws/UB 按 slot=seg%STACK_SLOTS 双缓冲；FD 写 partial，direct 写输出。
                AscendC::GlobalTensor<ElementLse> gLseWrite = fdMode ? gLseFD : gLse;
                const uint64_t gmOffsetLseWrite =
                    (fdMode ? static_cast<uint64_t>(lseBase + static_cast<int64_t>(s) * qHeads) : gmOffsetLse) +
                    headBase * static_cast<uint64_t>(lseHeadStride);
                const uint64_t gmOffsetOWrite =
                    (fdMode ? static_cast<uint64_t>(oBase + static_cast<int64_t>(s) * qHeads * embedV) : gmOffsetO) +
                    headBase * oHeadStride;

                // blockMmadPV 的 blockSize 形参是非 const 引用，需非 const 左值。
                uint32_t blockSize = pagedBlockSize;
                if (numSeg > 0U) {
                    // ---- prologue：MM1(0) + softmax(0) ----
                    const uint32_t stackSeqTile0 = AscendC::Std::min(MLA_DECODE_KV_TILE, kvSeqlenLocal);
                    const uint64_t gmOffsetBlock0 = wsSlotBase;
                    GemmCoord actualBlockShapeQK0{rowNum, stackSeqTile0, embedQK};
                    LayoutS layOutS0(rowNum, stackSeqTile0, MLA_DECODE_KV_TILE);
#ifdef __DAV_C220_CUBE__
                    blockMmadQK(gQ[gmOffsetQ], gV[gmOffsetV], gS[gmOffsetBlock0], AscendC::GlobalTensor<int32_t>(),
                                layoutQ, layoutKLatent, layOutS0, actualBlockShapeQK0, 0U, numSeg, blockSize,
                                static_cast<uint32_t>(strideV), false, gK[gmOffsetK], 0);
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_C220_VEC__
                    LayoutP layOutP0(rowNum, stackSeqTile0, MLA_DECODE_KV_TILE);
                    epilogueOnlineSoftmax(gP[gmOffsetBlock0], gS[gmOffsetBlock0], layOutP0, layOutS0,
                                          actualBlockShapeQK0, softmaxTaskPlan, 1U, 0U, qSBlockSize, qNBlockSize, 0U,
                                          stateSlot, qkReady, softmaxPingPongFlag, false, false, false);
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                }
                for (uint32_t seg = 0; seg < numSeg; ++seg) {
                    const bool hasNext = (seg + 1U < numSeg);
                    const bool firstStack = (seg == 0U);
                    const bool lastStack = (seg + 1U >= numSeg);
                    // 前瞻段（seg+1）参数：MM1 + softmax。
                    const uint32_t kvSStartNext = (seg + 1U) * MLA_DECODE_KV_TILE;
                    const uint32_t stackSeqTileNext =
                        AscendC::Std::min(MLA_DECODE_KV_TILE, kvSeqlenLocal - kvSStartNext);
                    const uint32_t slotNext = (seg + 1U) % STACK_SLOTS;
                    const uint64_t gmOffsetBlockNext =
                        wsSlotBase + static_cast<uint64_t>(slotNext) * WORKSPACE_BLOCK_SIZE_DB;
                    GemmCoord actualBlockShapeQKNext{rowNum, stackSeqTileNext, embedQK};
                    LayoutS layOutSNext(rowNum, stackSeqTileNext, MLA_DECODE_KV_TILE);
                    // 当前段（seg）参数：MM2 + rescale。
                    const uint32_t kvSStart = seg * MLA_DECODE_KV_TILE;
                    const uint32_t stackSeqTile = AscendC::Std::min(MLA_DECODE_KV_TILE, kvSeqlenLocal - kvSStart);
                    const uint32_t slot = seg % STACK_SLOTS;
                    const uint64_t gmOffsetBlock = wsSlotBase + static_cast<uint64_t>(slot) * WORKSPACE_BLOCK_SIZE_DB;
                    GemmCoord actualBlockShapePV{rowNum, embedV, stackSeqTile};

#ifdef __DAV_C220_CUBE__
                    if (hasNext) {
                        blockMmadQK(gQ[gmOffsetQ], gV[gmOffsetV], gS[gmOffsetBlockNext],
                                    AscendC::GlobalTensor<int32_t>(), layoutQ, layoutKLatent, layOutSNext,
                                    actualBlockShapeQKNext, seg + 1U, numSeg, blockSize, static_cast<uint32_t>(strideV),
                                    false, gK[gmOffsetK], 0);
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                    }
                    LayoutP layoutPTemp(rowNum, stackSeqTile, MLA_DECODE_KV_TILE);
                    blockMmadPV(gP[gmOffsetBlock], gV[gmOffsetV], gOTmp[gmOffsetBlock],
                                AscendC::GlobalTensor<int32_t>(), layoutPTemp, layoutVTemp, layoutOTmp,
                                actualBlockShapePV, seg, numSeg, blockSize, kvSeqlenLocal,
                                static_cast<uint32_t>(strideV), 1U, softmaxReady, false, gV[gmOffsetV], 0);
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_C220_VEC__
                    if (hasNext) {
                        LayoutP layOutPNext(rowNum, stackSeqTileNext, MLA_DECODE_KV_TILE);
                        epilogueOnlineSoftmax(gP[gmOffsetBlockNext], gS[gmOffsetBlockNext], layOutPNext, layOutSNext,
                                              actualBlockShapeQKNext, softmaxTaskPlan, 0U, 0U, qSBlockSize, qNBlockSize,
                                              slotNext, stateSlot, qkReady, softmaxPingPongFlag, false, false, false);
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
                    }
                    epilogueRescaleO(gO[gmOffsetOHead], gOTmp[gmOffsetBlock], gOUpdate[wsUpdateBase],
                                     gLseWrite[gmOffsetLseWrite], layoutO, layoutOTmp, layoutUpdate, layoutLse,
                                     actualBlockShapePV, rescaleTaskPlan, pvReady, qSBlockSize, qNBlockSize, firstStack,
                                     lastStack, slot, stateSlot, static_cast<uint32_t>(strideO),
                                     static_cast<uint32_t>(oHeadStride), fdMode, gOFD[gmOffsetOWrite]);
#endif
                }
#ifdef __DAV_C220_CUBE__
                // P0 (G2)：本 task 段循环毕、MTE1 读完 Q L1 区后才归还 MTE1_MTE2 EVENT_ID3，
                // 与 prelaunch 1 枚 Set、尾部 drain 1 枚 Wait 配平，消除 WAR hazard。
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
#endif
                // ---- 推进到下一个 task：split 内先递增 s；batch split 用尽后重算有工作 batch 的规则，
                // lseBase/oBase 按 running 累加推进（与宿主逐字一致），b/s/segCountB 等就地更新。
                ++s;
                if (s >= actualSplits) {
                    lseBase += static_cast<int64_t>(actualSplits) * qHeads;
                    oBase += static_cast<int64_t>(actualSplits) * qHeads * embedV;
                    s = 0;
                    bool found = false;
                    for (uint32_t nb = b + 1U; nb < batch; ++nb) {
                        uint32_t kvLenN = KvLenOf(nb, gActualKvseqlen, gSeqUsedKv, hasSeqUsedKv) + kvNewSeqlen;
                        uint32_t segCountN = CeilDiv(kvLenN, MLA_DECODE_KV_TILE);
                        if (segCountN == 0U) {
                            continue;
                        }
                        b = nb;
                        kvLenB = kvLenN;
                        segCountB = segCountN;
                        blocksPerSplit = CeilDiv(segCountN, nsplitPerBatch);
                        if (blocksPerSplit == 0U) {
                            blocksPerSplit = 1U;
                        }
                        actualSplits = CeilDiv(segCountN, blocksPerSplit);
                        found = true;
                        break;
                    }
                    if (!found) {
                        break;
                    }
                }
            }
        } // ---- G1：结束 head 分块循环

#ifdef __DAV_C220_CUBE__
        // D2：与 prelaunch 成对裁剪（见上），M_MTE1 4..7、MTE1_MTE2 2/7 无对应置位。
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
#endif
#ifdef __DAV_C220_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
#endif
        AscendC::PipeBarrier<PIPE_ALL>();

        // Multi-core KV split: partial O/LSE complete in workspace, so one
        // grid-stride combine reduces them to final O/LSE; direct tasks skip.
        if (fdMode) {
            AscendC::SyncAll();
#ifdef __DAV_C220_VEC__
            CombineScaleType combineScale;
            combineScale.init(resource);
            if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                // Packed TND: the group's owning seq folds cu_seqlens_q prefix
                // into both O and LSE; the LSE head stride is total_q.
                combineScale(qHeads, tilingData->totalSplitNodeNum, embedV, tilingData->splitInfo, gLseFD, gOFD, gO,
                             gActualQseqlen, true, maxQSeqlen,
                             (EpilogueRescaleO::LSE_MODE == Catlass::Epilogue::LseModeT::OUT_ONLY), gLse, totalQTokens);
            } else {
                combineScale(qHeads, tilingData->totalSplitNodeNum, embedV, tilingData->splitInfo, gLseFD, gOFD, gO,
                             AscendC::GlobalTensor<int32_t>(), false, maxQSeqlen,
                             (EpilogueRescaleO::LSE_MODE == Catlass::Epilogue::LseModeT::OUT_ONLY), gLse, 0U);
            }
#endif
        }
    }

  private:
    uint64_t mm1OutSize = 0;
    uint64_t smOnlineOutSize = 0;
    uint64_t mm2OutSize = 0;
    uint64_t updateSize = 0;
    uint32_t batch = 0;
    uint32_t qHeads = 0;
    uint32_t kvHeads = 0;
    uint32_t embed = 0;
    uint32_t embedV = 0;
    uint32_t embedQK = 0;
    uint32_t maxQSeqlen = 0;
    // Packed TND LSE is (num_heads, total_q), so the head stride is the
    // launch's totalQTokens, not maxQSeqlen. Zero for BSND/BNSD.
    uint32_t totalQTokens = 0;
    uint32_t kvNewSeqlen = 0;
    uint32_t pagedBlockSize = 128;
    uint64_t kvSeqlenAllocated = 0;
    float scaleValue = 0.0f;
    float softcapValue = 0.0f;
    uint64_t strideQ = 0;
    uint64_t strideO = 0;
    uint64_t strideK = 0;
    uint64_t strideV = 0;
    uint64_t qHeadStride = 0;
    uint64_t oHeadStride = 0;
    uint64_t strideQv = 0;
    uint64_t qvHeadStride = 0;
    uint64_t kvHeadStrideK = 0;
    uint64_t kvHeadStrideV = 0;
    uint32_t embedRoundV = 0;

    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
    Arch::CrossCoreFlag pvReady{PV_READY_ID};
    Gemm::Block::BlockPingPongState pingPongState;

    BlockMmadQK blockMmadQK;
    BlockMmadPV blockMmadPV;
    EpilogueOnlineSoftmax epilogueOnlineSoftmax;
    EpilogueRescaleO epilogueRescaleO;

    // P0 (G2)：每 batch KV 长度（不含 kvNewSeqlen），口径同宿主 seqlens_k_cpu 及通用 FAInfer 的
    // KvSeqLenOf 一致：BSND/BNSD 用 actualKvseqlen[b]，TND 用前缀差；核内重算须与宿主逐字一致。
    __aicore__ inline uint32_t KvLenOf(uint32_t b, AscendC::GlobalTensor<int32_t> gKvSeq,
                                       AscendC::GlobalTensor<int32_t> gSeqUsed, bool hasSeqUsed)
    {
        if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
            return hasSeqUsed
                       ? static_cast<uint32_t>(gSeqUsed.GetValue(b))
                       : static_cast<uint32_t>(gKvSeq.GetValue(b + 1)) - static_cast<uint32_t>(gKvSeq.GetValue(b));
        } else {
            return static_cast<uint32_t>(gKvSeq.GetValue(b));
        }
    }
};

template <typename InputDtypeQ = half, FaiKenel::inputLayout INPUT_LAYOUT = FaiKenel::inputLayout::BSND>
__global__ __aicore__ void MlaDecodeGemv(uint64_t fftsAddr, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o, GM_ADDR lse,
                                         GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR seqUsedKv,
                                         GM_ADDR workspace, FAInferTilingParams tiling, GM_ADDR qv,
                                         int64_t kvSeqlenAllocated)
{
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Arch::AtlasA2;
    using ElementQ = InputDtypeQ;
    using LayoutQ = layout::RowMajor;
    using ElementK = InputDtypeQ;
    using LayoutK = layout::ColumnMajor;
    using ElementV = InputDtypeQ;
    using LayoutV = layout::RowMajor;
    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using ElementP = InputDtypeQ;
    using LayoutP = layout::RowMajor;
    using ElementO = InputDtypeQ;
    using LayoutO = layout::RowMajor;
    using ElementLse = float;
    using LayoutLse = layout::RowMajor;
    using ElementMask = uint8_t;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;

    using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
    using L0TileShapeQK = GemmShape<128, 128, 128>;
    using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQKT<false, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

    using DispatchPolicyOnlineSoftmax =
        Epilogue::EpilogueAtlasA2OnlineSoftmaxT<Epilogue::LseModeT::OUT_ONLY, float, false>;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, MaskType>;

    using L1TileShapePV = GemmShape<128, 128, 256>;
    using L0TileShapePV = GemmShape<128, 128, 128>;
    using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPVT<false, false>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTmpType>;

    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleOT<Epilogue::LseModeT::OUT_ONLY, float>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
    using EpilogueRescaleO =
        SplitFuse::MlaDecodePersistRescale<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;

    using KernelType =
        MlaDecodeGemvKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, INPUT_LAYOUT>;

    FAIKernelParams params{q,         k,      v,       nullptr, nullptr, actualQseqlen, actualKvseqlen,    o, lse,
                           workspace, tiling, nullptr, nullptr, nullptr, seqUsedKv,     kvSeqlenAllocated, qv};
    KernelType kernel;
    kernel(params);
}

} // namespace SplitFuse
