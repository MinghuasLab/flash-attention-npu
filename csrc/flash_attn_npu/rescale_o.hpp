/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_NO_SPLIT_ROW_HPP_T
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_NO_SPLIT_ROW_HPP_T

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "fa_block.h"

namespace Catlass::Epilogue::Block {

template <
    class OutputType_,
    class InputType_,
    class UpdateType_,
    class LseType_,
    LseModeT LSE_MODE_>
class BlockEpilogue<
    EpilogueAtlasA2RescaleOT<LSE_MODE_, float>,
    OutputType_,
    InputType_,
    UpdateType_,
    LseType_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2RescaleOT<LSE_MODE_, float>;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;
    using ElementUpdate = typename UpdateType_::Element;
    using ElementLse = typename LseType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutUpdate = typename UpdateType_::Layout;
    using LayoutLse = typename LseType_::Layout;

    static constexpr LseModeT LSE_MODE = DispatchPolicy::LSE_MODE;

    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;
    static constexpr uint32_t MULTIPLIER = 2;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr float LSE_OUT_INI = std::numeric_limits<float>::infinity();
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;
    static constexpr uint32_t HALF_DM_UB_SIZE = 64;
    static constexpr uint32_t HALF_LL_UB_SIZE = 256;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t NUM4 = 4;
    static constexpr uint32_t MAX_UB_O_ELEM_NUM = 8192;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr uint32_t SIZE_OF_16BIT = 2;

    struct SplitKVParams {
        bool isSplitkv = false;
        AscendC::GlobalTensor<ElementLse> gCombineLse;
        AscendC::GlobalTensor<ElementLse> gCombineo;
        const LayoutLse *layoutgmLse = nullptr;
        const LayoutInput *layoutgmLo = nullptr;
    };

    __aicore__ inline
    BlockEpilogue() {}

    __aicore__ inline
    ~BlockEpilogue() {}

    __aicore__ inline
    void init(Arch::Resource<ArchTag> &resource)
    {
        // Allocate UB space
        constexpr uint32_t LO_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 8 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;

        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t LSE_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;

        loUbTensor = resource.ubBuf.template GetBufferByByte<float>(LO_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_UB_TENSOR_OFFSET);
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GO_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        lse32_ubuf_tensor = resource.ubBuf.template GetBufferByByte<float>(LSE_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    void SetMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = static_cast<uint64_t>(len) % static_cast<uint64_t>(FLOAT_VECTOR_SIZE);
        for (uint64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }

        if (len == VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    __aicore__ inline
    void InvalidLineLSEProcess(
        uint32_t qNThisSubBlock, int32_t delStartRow, uint32_t qSBlockIdx, uint32_t inRowOffsetThisSubBlock,
        uint32_t totalRowNum, int32_t delEndRow, uint32_t qSeqlen, uint32_t qSThisSubBlock)
    {
        uint32_t qNSubBlockStartOffset = qNThisSubBlock == 0U ? qSBlockIdx * VECTOR_SIZE + inRowOffsetThisSubBlock : qSBlockIdx * VECTOR_SIZE;
        uint32_t qNSubBlockEnbdOffset = totalRowNum + qNSubBlockStartOffset;
        if (qNThisSubBlock == 0U && delStartRow != 0 && qNSubBlockEnbdOffset >= delStartRow) {
            uint32_t start = qNSubBlockStartOffset > delStartRow ? 0 : (delStartRow - qNSubBlockStartOffset);
            uint32_t end = totalRowNum;
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(
                tvUbTensor[start * FLOAT_BLOCK_SIZE],
                LSE_OUT_INI,
                (end - start) * FLOAT_BLOCK_SIZE
            );
        }
        if (qNThisSubBlock == 0U && delEndRow != qSeqlen && qNSubBlockStartOffset < delEndRow) {
            uint32_t rowStart = qNSubBlockStartOffset;
            uint32_t start = 0;
            uint32_t end = rowStart + totalRowNum >= delEndRow ? (delEndRow - rowStart) : totalRowNum;
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(
                tvUbTensor[start * FLOAT_BLOCK_SIZE],
                LSE_OUT_INI,
                (end - start) * FLOAT_BLOCK_SIZE
            );
        }
        if (qNThisSubBlock != 0U && delStartRow != 0 && qNSubBlockEnbdOffset >= delStartRow) {
            uint32_t start = delStartRow - qNSubBlockStartOffset;
            uint32_t end = qSThisSubBlock;
            for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Duplicate(
                    tvUbTensor[(qNIdx * qSThisSubBlock + start) * FLOAT_BLOCK_SIZE],
                    LSE_OUT_INI,
                    (end - start) * FLOAT_BLOCK_SIZE
                );
            }
        }
        if (qNThisSubBlock != 0U && delEndRow != qSeqlen && qNSubBlockStartOffset < delEndRow) {
            uint32_t start = 0;
            uint32_t end = qNSubBlockEnbdOffset >= delEndRow ? (delEndRow - qNSubBlockStartOffset) : totalRowNum;
            for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Duplicate(
                    tvUbTensor[(qNIdx * qSThisSubBlock + start) * FLOAT_BLOCK_SIZE],
                    LSE_OUT_INI,
                    (end - start) * FLOAT_BLOCK_SIZE
                );
            }
        }
    }

    __aicore__ inline
    void CopyOToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t proTokenIdx, uint32_t proTokenNum,
        uint32_t epiTokenNum, uint32_t integralHeadNum, uint32_t qSThisSubBlock, uint32_t embed, uint32_t oHiddenSize)
    {
        uint32_t innerOGmOffset = 0;
        uint32_t innerGOUbOffset = 0;
        if (proTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset + proTokenIdx * oHiddenSize],
                goUbTensor16[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    proTokenNum, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += proTokenNum * embed;
        }
        for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor16[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    qSThisSubBlock, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += qSThisSubBlock * embed;
        }
        if (epiTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor16[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    epiTokenNum, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
        }
    }

    // FD-only: fp32 variant of CopyOToGm used when writing partial O into
    // splitParams.gCombineo (needs higher precision before the combine-scale step).
    __aicore__ inline
    void CopyOToGmFp32(
        AscendC::GlobalTensor<float> gOutput,
        uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t epiTokenNum, uint32_t integralHeadNum,
        uint32_t qSThisSubBlock, uint32_t embed, uint32_t embedRound, uint32_t oHiddenSize, uint32_t oHiddenSize_gmlo)
    {
        uint32_t innerOGmOffset = 0;
        uint32_t innerGOUbOffset = 0;
        uint32_t blockLen = embed * sizeof(float);
        uint32_t blockLenAligned = (blockLen + 31) / 32 * 32;
        uint32_t srcStride = (embedRound * sizeof(float) - blockLenAligned) / 32;
        if (proTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset + proTokenIdx * oHiddenSize],
                goUbTensor32[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    proTokenNum, blockLen, srcStride, (oHiddenSize_gmlo - embed) * sizeof(float), 0));
            innerOGmOffset += embed;
            innerGOUbOffset += proTokenNum * embedRound;
        }
        for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor32[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    qSThisSubBlock, blockLen, srcStride, (oHiddenSize_gmlo - embed) * sizeof(float), 0));
            innerOGmOffset += embed;
            innerGOUbOffset += qSThisSubBlock * embedRound;
        }
        if (epiTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor32[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    epiTokenNum, blockLen, srcStride, (oHiddenSize_gmlo - embed) * sizeof(float), 0));
        }
    }

    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementUpdate> gUpdate,
        AscendC::GlobalTensor<ElementLse> gLse,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        const LayoutUpdate &layoutUpdate,
        const LayoutLse &layoutLse,
        uint32_t qNThisSubBlock, uint32_t qSThisSubBlock, uint32_t totalRowNum,
        uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod,
        uint32_t needRowLoop, uint32_t isLastRowLoop, uint32_t rowOffsetLoop,
        uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t epiTokenNum, uint32_t integralHeadNum,
        const SplitKVParams &splitParams,
        uint32_t rowOffsetCurLoop, int32_t delStartRow, int32_t delEndRow, uint32_t qSeqlen,
        uint32_t qSBlockIdx, uint32_t rowNum, uint32_t inRowOffsetThisSubBlock, uint32_t curQNBlockTile)
    {
        uint32_t curRowNum = layoutInput.shape(0);
        uint32_t embed = layoutInput.shape(1);
        uint32_t embedRound = layoutInput.stride(0);
        uint32_t curRowNumRound = RoundUp(curRowNum, FLOAT_BLOCK_SIZE);
        uint32_t qSBlockSize = layoutOutput.shape(0);
        uint32_t oHiddenSize = layoutOutput.shape(1);
        uint32_t qHeads = layoutLse.shape(0);
        uint32_t dmUbOffsetCurStackTile = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffsetLoop;

        // FD: read partial-O / partial-LSE hidden dims from splitParams layouts.
        uint32_t oHiddenSize_gmlo = 0;
        uint32_t qHeads_gmlse = 0;
        if (splitParams.isSplitkv) {
            oHiddenSize_gmlo = splitParams.layoutgmLo->shape(1);
            qHeads_gmlse = splitParams.layoutgmLse->shape(1);
        }

        if (!isFirstStackTile) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::DataCopy(
                loUbTensor, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        if (!isFirstStackTile) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                dmUbTensor[dmUbOffsetCurStackTile].ReinterpretCast<uint32_t>(),
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            if (needRowLoop) {
                AscendC::DataCopy(
                    goUbTensor32, gUpdate,
                    AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            }
            // *** go = go * dm_block
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vmul_idx = 0; vmul_idx < embed / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                AscendC::Mul<float, false>(
                    goUbTensor32[vmul_idx * FLOAT_VECTOR_SIZE],
                    goUbTensor32[vmul_idx * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
            }
            if (embed % FLOAT_VECTOR_SIZE > 0) {
                SetMask(embed % FLOAT_VECTOR_SIZE);
                AscendC::Mul<float, false>(
                    goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // *** go = lo + go
            AscendC::Add<float, false>(
                goUbTensor32,
                goUbTensor32,
                loUbTensor,
                (uint64_t)0,
                (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        } else {
            // *** go = lo
            AscendC::DataCopy(
                goUbTensor32, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }

        if (isLastStackTile) {
            // *** gl_block = expand_to_block(gl)
            AscendC::Brcb(
                tvUbTensor.ReinterpretCast<uint32_t>(),
                glUbTensor.ReinterpretCast<uint32_t>()[rowOffsetLoop],
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** go = go / gl_block
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vdiv_idx = 0; vdiv_idx < embed / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
                AscendC::Div<float, false>(
                    goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                    goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
            }
            if (embed % FLOAT_VECTOR_SIZE > 0) {
                SetMask(embed % FLOAT_VECTOR_SIZE);
                AscendC::Div<float, false>(
                    goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();

            // *** go = castfp32to16(go)
            // FD: skip the fp32->fp16 cast when writing to splitParams.gCombineo
            // (partial O must stay fp32 for numerically-safe combine).
            if (!splitParams.isSplitkv) {
                if (std::is_same<ElementOutput, bfloat16_t>::value) {
                    AscendC::Cast<ElementOutput, float, false>(
                        goUbTensor16, goUbTensor32,
                        AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                        (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                        AscendC::UnaryRepeatParams(1, 1, 4, 8));
                } else {
                    AscendC::Cast<ElementOutput, float, false>(
                        goUbTensor16, goUbTensor32,
                        AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                        (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                        AscendC::UnaryRepeatParams(1, 1, 4, 8));
                }
            }
            uint32_t rowStart = qSBlockIdx * VECTOR_SIZE + rowOffsetCurLoop ;
            uint32_t innerGOUbOffset = 0;
            uint32_t subBlockStart = (curQNBlockTile == 1U) ? rowStart  : (rowStart >= qSeqlen ? rowStart - rowStart / qSeqlen * qSeqlen : rowStart);
            if (delStartRow != 0) {
                if (proTokenNum != 0U && subBlockStart + proTokenNum >= delStartRow) {
                    uint32_t start = subBlockStart >= delStartRow ? 0 : delStartRow - subBlockStart;
                    uint32_t end = proTokenNum;
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Duplicate<ElementOutput>(
                        goUbTensor16[innerGOUbOffset + start * embed],
                        static_cast<ElementOutput>(0),
                        (end - start) * embed
                    );
                    innerGOUbOffset += proTokenNum * embed;
                }
                if (subBlockStart + qSThisSubBlock >= delStartRow) {
                    for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
                        uint32_t start = subBlockStart >= delStartRow ? 0 : delStartRow - subBlockStart;
                        uint32_t end = qSThisSubBlock;
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Duplicate<ElementOutput>(
                            goUbTensor16[innerGOUbOffset + start  * embed],
                            static_cast<ElementOutput>(0),
                            (end - start) * embed
                        );
                        innerGOUbOffset += qSThisSubBlock * embed;
                    }
                }
                if (epiTokenNum != 0U && subBlockStart + epiTokenNum >= delStartRow) {
                    uint32_t start = subBlockStart >= delStartRow ? 0 : delStartRow - subBlockStart;
                    uint32_t end = epiTokenNum;
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Duplicate<ElementOutput>(
                        goUbTensor16[innerGOUbOffset + start * embed],
                        static_cast<ElementOutput>(0),
                        (end - start) * embed
                    );
                }
            }
            if (delEndRow != qSeqlen) {
                if (proTokenNum != 0U && subBlockStart < delEndRow) {
                    uint32_t start = curQNBlockTile == 1U ? rowStart : 0;
                    uint32_t end = (subBlockStart + proTokenNum >= delEndRow) ?
                                                    (curQNBlockTile == 1U ? delEndRow : delEndRow - subBlockStart)
                                                            : subBlockStart + proTokenNum;
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Duplicate<ElementOutput>(
                        goUbTensor16[innerGOUbOffset],
                        static_cast<ElementOutput>(0),
                        (end - start) * embed
                    );
                    innerGOUbOffset += proTokenNum * embed;
                }
                if (subBlockStart < delEndRow) {
                    for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
                        uint32_t start = curQNBlockTile == 1U ? subBlockStart : proTokenNum;
                        uint32_t end = (subBlockStart + qSThisSubBlock >= delEndRow) ?
                                            (curQNBlockTile == 1U ? delEndRow : delEndRow - subBlockStart)
                                                         : start + qSThisSubBlock;
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Duplicate<ElementOutput>(
                            goUbTensor16[innerGOUbOffset],
                            static_cast<ElementOutput>(0),
                            (end - start) * embed
                        );
                        innerGOUbOffset += qSThisSubBlock * embed;
                    }
                }
                if (epiTokenNum != 0U && subBlockStart < delEndRow) {
                    uint32_t start = curQNBlockTile == 1U ? subBlockStart : proTokenNum + integralHeadNum * qSThisSubBlock + subBlockStart;
                    uint32_t end = curQNBlockTile == 1U ? (subBlockStart + epiTokenNum >= delEndRow ? delEndRow : subBlockStart + epiTokenNum) :
                                            (epiTokenNum >= delEndRow ? start + delEndRow: start + epiTokenNum);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Duplicate<ElementOutput>(
                        goUbTensor16[innerGOUbOffset],
                        static_cast<ElementOutput>(0),
                        (end - start) * embed
                    );
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            // ***move O to GM: FD SplitKV writes partial fp32 O into gCombineo;
            // otherwise writes fp16 O directly to gOutput.
            if (splitParams.isSplitkv) {
                CopyOToGmFp32(
                    splitParams.gCombineo,
                    proTokenIdx,
                    proTokenNum,
                    epiTokenNum,
                    integralHeadNum,
                    qSThisSubBlock,
                    embed,
                    embedRound,
                    oHiddenSize, oHiddenSize_gmlo);
            } else {
                CopyOToGm(
                    gOutput, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum,
                    qSThisSubBlock, embed, oHiddenSize);
            }
            if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
                if (isLastRowLoop) {
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Ln<float, false>(
                        lse32_ubuf_tensor,
                        glUbTensor,
                        (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                        AscendC::UnaryRepeatParams(1, 1, 8, 8));

                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<float, false>(
                        lse32_ubuf_tensor,
                        lse32_ubuf_tensor,
                        gmUbTensor,
                        (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                        AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                    AscendC::PipeBarrier<PIPE_V>();

                    // *** lse_block = expand_to_block(lse)
                    AscendC::Brcb(
                        tvUbTensor.ReinterpretCast<uint32_t>(),
                        lse32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                        CeilDiv(totalRowNum, FLOAT_BLOCK_SIZE),
                        AscendC::BrcbRepeatParams(1, 8));
                    InvalidLineLSEProcess(qNThisSubBlock, delStartRow, qSBlockIdx,
                            inRowOffsetThisSubBlock, totalRowNum, delEndRow, qSeqlen, qSThisSubBlock);
                    if (!splitParams.isSplitkv) {
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    
                    if (qNThisSubBlock == 0U) {
                        if (splitParams.isSplitkv) {
                            AscendC::DataCopyPad(
                                splitParams.gCombineLse, tvUbTensor,
                                AscendC::DataCopyExtParams(
                                    totalRowNum, sizeof(float), 0, (qHeads_gmlse - 1) * sizeof(float), 0));
                        } else {
                            AscendC::DataCopyPad(
                                gLse, tvUbTensor,
                                AscendC::DataCopyExtParams(
                                    totalRowNum, sizeof(float), 0, 0, 0));
                        }
                    } else {
                        for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                            if (splitParams.isSplitkv) {
                                AscendC::DataCopyPad(
                                    splitParams.gCombineLse[qNIdx],
                                    tvUbTensor[qNIdx * qSBlockSize * FLOAT_BLOCK_SIZE],
                                    AscendC::DataCopyExtParams(
                                        qSBlockSize, sizeof(float), 0, (qHeads_gmlse - 1) * sizeof(float), 0));
                            } else {
                                AscendC::DataCopyPad(
                                    gLse[qNIdx * layoutLse.stride(0)],
                                    tvUbTensor[qNIdx * qSBlockSize * FLOAT_BLOCK_SIZE],
                                    AscendC::DataCopyExtParams(
                                        qSBlockSize, sizeof(float), 0, 0, 0));
                            }
                        }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
                }
            } else {
                // FD SplitKV must still write partial LSE even when LSE_MODE != OUT_ONLY,
                // because the combine epilogue requires per-split LSE for rescaling.
                if (splitParams.isSplitkv) {
                    if (isLastRowLoop) {
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Ln<float, false>(
                            lse32_ubuf_tensor,
                            glUbTensor,
                            (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                            AscendC::UnaryRepeatParams(1, 1, 8, 8));

                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add<float, false>(
                            lse32_ubuf_tensor,
                            lse32_ubuf_tensor,
                            gmUbTensor,
                            (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                        AscendC::PipeBarrier<PIPE_V>();

                        AscendC::Brcb(
                            tvUbTensor.ReinterpretCast<uint32_t>(),
                            lse32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                            CeilDiv(totalRowNum, FLOAT_BLOCK_SIZE),
                            AscendC::BrcbRepeatParams(1, 8));
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);

                        if (qNThisSubBlock == 0U) {
                            AscendC::DataCopyPad(
                                splitParams.gCombineLse, tvUbTensor,
                                AscendC::DataCopyExtParams(
                                    totalRowNum, sizeof(float), 0, (qHeads_gmlse - 1) * sizeof(float), 0));
                        } else {
                            for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                                AscendC::DataCopyPad(
                                    splitParams.gCombineLse[qNIdx],
                                    tvUbTensor[qNIdx * qSBlockSize * FLOAT_BLOCK_SIZE],
                                    AscendC::DataCopyExtParams(
                                        qSBlockSize, sizeof(float), 0, (qHeads_gmlse - 1) * sizeof(float), 0));
                            }
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
                    }
                }
            }
        } else if (needRowLoop) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::DataCopy(
                gUpdate, goUbTensor32, AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    }

    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementUpdate> gUpdate,
        AscendC::GlobalTensor<ElementLse> gLse,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        const LayoutUpdate &layoutUpdate,
        const LayoutLse &layoutLse,
        GemmCoord actualBlockShape,
        uint32_t qSBlockSize, uint32_t qNBlockSize,
        uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod,
        const SplitKVParams& splitParams = SplitKVParams(),
        int32_t delStartRow = 0, int32_t delEndRow = 0, uint32_t qSeqlen = 0,
        uint32_t qSBlockIdx = 0, uint32_t curQNBlockTile = 1)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t embed = actualBlockShape.n();
        uint32_t maxRowNumPerLoop = MAX_UB_O_ELEM_NUM / embed;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ? 0
                                  : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock)
                                                       : qNSplitSubBlock;
        uint32_t inRowSplitSubBlock =
            (qNBlockSize == 1U) ? (qSBlockSize / subBlockNum) : (qSBlockSize * qNSplitSubBlock);
        uint32_t inRowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - inRowSplitSubBlock) : inRowSplitSubBlock;
        uint32_t inRowOffsetThisSubBlock = subBlockIdx * inRowSplitSubBlock;
        uint32_t outRowOffsetThisSubBlock = (qNBlockSize == 1U) ? inRowOffsetThisSubBlock : 0;
        uint32_t outColOffsetThisSubBlock = (qNBlockSize == 1U) ? 0 : subBlockIdx * qNSplitSubBlock * embed;
        uint32_t qSThisSubBlock = (qNBlockSize == 1U) ? inRowActualThisSubBlock : qSBlockSize;
        int64_t outOffsetSubBlock =
            layoutOutput.GetOffset(MatrixCoord(outRowOffsetThisSubBlock, outColOffsetThisSubBlock));

        // FD: resolve per-subblock offset for the partial O buffer (gCombineo).
        int64_t gmlooutOffsetSubBlock = 0;
        if (splitParams.isSplitkv) {
            gmlooutOffsetSubBlock =
                splitParams.layoutgmLo->GetOffset(MatrixCoord(outRowOffsetThisSubBlock, outColOffsetThisSubBlock));
        }

        uint32_t outLseRowOffsetThisSubBlock = (qNBlockSize == 1U) ?
            0 : subBlockIdx * qNSplitSubBlock;  // row = heads
        uint32_t outLseColOffsetThisSubBlock = (qNBlockSize == 1U) ?
            inRowOffsetThisSubBlock : 0;  // col = sequence
        int64_t offsetLse =
            layoutLse.GetOffset(MatrixCoord(outLseRowOffsetThisSubBlock, outLseColOffsetThisSubBlock));
        auto gLseThisSubBlock = gLse[offsetLse];
        auto layoutOutLseThisSubBlock = layoutLse;

        // FD: resolve per-subblock offset into the partial LSE buffer (gCombineLse).
        int64_t gmLseoffsetLse = 0;
        if (splitParams.isSplitkv) {
            outLseRowOffsetThisSubBlock = (qNBlockSize == 1U) ? inRowOffsetThisSubBlock : 0;
            outLseColOffsetThisSubBlock = (qNBlockSize == 1U) ? 0 : subBlockIdx * qNSplitSubBlock;
            gmLseoffsetLse =
                splitParams.layoutgmLse->GetOffset(MatrixCoord(outLseRowOffsetThisSubBlock, outLseColOffsetThisSubBlock));
        }

        // Forward a mutable copy of splitParams into SubCoreCompute so we can
        // rewrite gCombineLse/gCombineo to the subblock-local slice.
        SplitKVParams blockParams = splitParams;
        if (splitParams.isSplitkv) {
            blockParams.gCombineLse = splitParams.gCombineLse[gmLseoffsetLse];
        }

        if (inRowActualThisSubBlock > 0U) {
            uint32_t rowLoop = CeilDiv(inRowActualThisSubBlock, rowNumTile);
            uint32_t needRowLoop = (rowLoop > 1U) ? 1 : 0;

            // The rows of each cycle consist of multiple heads with several tokens.
            // There are several integral heads, one prologue head, one epilogue head.
            uint32_t proTokenIdx = 0;      // the token idx of the start token of the prologue part
            uint32_t proTokenIdxPre = 0;   // the token idx of the start token of the pre prologue part
            uint32_t proTokenNum = 0;      // the token num of the prologue part
            uint32_t epiTokenNum = 0;      // the token num of the epilogue part
            uint32_t integralHeadNum = 0;  // the number of integral heads within a cycle
            uint32_t qSRemian = qSThisSubBlock;
            for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoop; rowLoopIdx++) {
                uint32_t rowOffsetLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetCurLoop = inRowOffsetThisSubBlock + rowOffsetLoop;
                uint32_t rowActualCurLoop =
                    (rowLoopIdx == (rowLoop - 1U)) ? inRowActualThisSubBlock - rowLoopIdx * rowNumTile : rowNumTile;

                int64_t offsetOutput =
                    static_cast<int64_t>(rowLoopIdx * rowNumTile / qSThisSubBlock * embed) + outOffsetSubBlock;

                // FD: advance gCombineo to the current row-loop's tile offset.
                int64_t gmloffset = 0;
                if (splitParams.isSplitkv) {
                    gmloffset =
                        static_cast<int64_t>(rowLoopIdx * rowNumTile / qSThisSubBlock * embed) + gmlooutOffsetSubBlock;
                    blockParams.gCombineo = splitParams.gCombineo[gmloffset];
                }

                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput;
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
                auto gInputCurLoop = gInput[offsetInput];
                auto layoutInputCurLoop = layoutInput.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));

                int64_t offsetUpdate = layoutUpdate.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
                auto gUpdateCurLoop = gUpdate[offsetUpdate];
                auto layoutUpdateCurLoop = layoutUpdate.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));

                proTokenIdx = rowOffsetLoop % qSThisSubBlock;
                proTokenNum = AscendC::Std::min(rowActualCurLoop, (qSThisSubBlock - proTokenIdx)) % qSThisSubBlock;
                integralHeadNum = (rowActualCurLoop - proTokenNum) / qSThisSubBlock;
                epiTokenNum = rowActualCurLoop - proTokenNum - integralHeadNum * qSThisSubBlock;

                SubCoreCompute(
                    gOutputCurLoop,
                    gInputCurLoop,
                    gUpdateCurLoop,
                    gLseThisSubBlock,
                    layoutOutputCurLoop,
                    layoutInputCurLoop,
                    layoutUpdateCurLoop,
                    layoutOutLseThisSubBlock,
                    qNThisSubBlock,
                    qSThisSubBlock,
                    inRowActualThisSubBlock,
                    isFirstStackTile,
                    isLastStackTile,
                    curStackTileMod,
                    needRowLoop,
                    (rowLoopIdx == rowLoop - 1U),
                    rowOffsetLoop,
                    proTokenIdx,
                    proTokenNum,
                    epiTokenNum,
                    integralHeadNum,
                    blockParams,
                    rowOffsetCurLoop,
                    delStartRow,
                    delEndRow,
                    qSeqlen,
                    qSBlockIdx,
                    rowNum,
                    inRowOffsetThisSubBlock,
                    curQNBlockTile);
            }
        }
    }

private:
    AscendC::LocalTensor<float> loUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<float> hmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> tvUbTensor;
    AscendC::LocalTensor<ElementOutput> goUbTensor16;
    AscendC::LocalTensor<float> goUbTensor32;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> lse32_ubuf_tensor;
};

}

#endif
