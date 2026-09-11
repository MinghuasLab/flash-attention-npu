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
    LseModeT LSE_MODE_,
    bool HAS_DROPOUT_>
class BlockEpilogue<
    EpilogueAtlasA2RescaleOT<LSE_MODE_, float, HAS_DROPOUT_>,
    OutputType_,
    InputType_,
    UpdateType_,
    LseType_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2RescaleOT<LSE_MODE_, float, HAS_DROPOUT_>;
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

    // Scalar + broadcast LSE: [177920, 180224), between live DM rows and raw masks.
    static constexpr uint32_t LSE_STAGING_ELEMENTS = FLOAT_VECTOR_SIZE * (1 + FLOAT_BLOCK_SIZE);
    static constexpr uint32_t LSE_STAGING_UB_OFFSET =
        11 * UB_UINT8_BLOCK_SIZE - LSE_STAGING_ELEMENTS * sizeof(float);

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
    void init(Arch::Resource<ArchTag> &resource, float dropoutValue_)
    {
        // Allocate UB space
        constexpr uint32_t LO_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 8 * UB_UINT8_BLOCK_SIZE;

        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE + 2 * 256;
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE + 5 * 256;
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE + 8 * 256;

        dropoutValue = dropoutValue_;

        loUbTensor = resource.ubBuf.template GetBufferByByte<float>(LO_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_UB_TENSOR_OFFSET);
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GO_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        static_assert(DM_UB_TENSOR_OFFSET + (2 * MAX_ROW_NUM_SUB_CORE + FLOAT_VECTOR_SIZE) * sizeof(float)
            <= LSE_STAGING_UB_OFFSET, "LSE staging overlaps live DM rows");
        lseStagingUbTensor = resource.ubBuf.template GetBufferByByte<float>(LSE_STAGING_UB_OFFSET);
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
        uint32_t qNThisSubBlock, int32_t invalidSuffixStartRow, uint32_t qSBlockIdx, uint32_t inRowOffsetThisSubBlock,
        uint32_t totalRowNum, int32_t invalidPrefixEndRow, uint32_t qSeqlen, uint32_t qSThisSubBlock,
        const AscendC::LocalTensor<float> &lseUbTensor)
    {
        if (totalRowNum == 0U || (invalidSuffixStartRow == 0 && invalidPrefixEndRow == qSeqlen)) {
            return;
        }
        const uint32_t headCount = qNThisSubBlock == 0U ? 1U : qNThisSubBlock;
        const uint32_t rowsPerHead = qNThisSubBlock == 0U ? totalRowNum : qSThisSubBlock;
        const int64_t tokenStart = static_cast<int64_t>(qSBlockIdx) * VECTOR_SIZE +
            (qNThisSubBlock == 0U ? inRowOffsetThisSubBlock : 0U);
        uint64_t headMask = 0;
        if (invalidPrefixEndRow >= 0 && invalidPrefixEndRow != qSeqlen && invalidPrefixEndRow > tokenStart) {
            const uint32_t prefixRows = AscendC::Std::min(
                static_cast<int64_t>(rowsPerHead), static_cast<int64_t>(invalidPrefixEndRow) - tokenStart);
            headMask |= ~uint64_t{0} >> (FLOAT_VECTOR_SIZE - prefixRows);
        }
        if (invalidSuffixStartRow > 0 && invalidSuffixStartRow < tokenStart + rowsPerHead) {
            const uint32_t suffixStart = invalidSuffixStartRow > tokenStart ? invalidSuffixStartRow - tokenStart : 0U;
            const uint32_t suffixRows = rowsPerHead - suffixStart;
            headMask |= (~uint64_t{0} >> (FLOAT_VECTOR_SIZE - suffixRows)) << suffixStart;
        }
        if (headMask == 0U) {
            return;
        }
        uint64_t invalidMask = 0;
        for (uint32_t headIdx = 0; headIdx < headCount; ++headIdx) {
            invalidMask |= headMask << (headIdx * rowsPerHead);
        }
        // Each vector owns at most 64 rows; mask from the aligned base even for unaligned heads.
        uint64_t mask[2] = {invalidMask, 0};
        AscendC::Duplicate<float>(lseUbTensor, LSE_OUT_INI, mask, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void ClearInvalidOutputRows(
        uint32_t ubRowOffset, uint32_t tokenStart, uint32_t tokenNum,
        int32_t invalidSuffixStartRow, int32_t invalidPrefixEndRow, uint32_t qSeqlen, uint32_t embedRound)
    {
        if (tokenNum == 0U) {
            return;
        }

        // invalidSuffixStartRow marks an invalid suffix [invalidSuffixStartRow, qSeqlen).
        if (invalidSuffixStartRow > 0) {
            uint32_t suffixStart = static_cast<uint32_t>(invalidSuffixStartRow);
            uint32_t localStart = 0U;
            if (tokenStart < suffixStart) {
                uint32_t validPrefix = suffixStart - tokenStart;
                localStart = validPrefix < tokenNum ? validPrefix : tokenNum;
            }
            if (localStart < tokenNum) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Duplicate<ElementOutput>(
                    goUbTensor16[(ubRowOffset + localStart) * embedRound],
                    static_cast<ElementOutput>(0),
                    (tokenNum - localStart) * embedRound);
            }
        }

        // invalidPrefixEndRow marks an invalid prefix [0, invalidPrefixEndRow).
        if (invalidPrefixEndRow >= 0 && invalidPrefixEndRow != static_cast<int32_t>(qSeqlen)) {
            uint32_t prefixEnd = static_cast<uint32_t>(invalidPrefixEndRow);
            uint32_t localEnd = 0U;
            if (tokenStart < prefixEnd) {
                uint32_t invalidPrefix = prefixEnd - tokenStart;
                localEnd = invalidPrefix < tokenNum ? invalidPrefix : tokenNum;
            }
            if (localEnd > 0U) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Duplicate<ElementOutput>(
                    goUbTensor16[ubRowOffset * embedRound],
                    static_cast<ElementOutput>(0),
                    localEnd * embedRound);
            }
        }
    }

    __aicore__ inline
    void CopyOToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t proTokenIdx, uint32_t proTokenNum,
        uint32_t epiTokenNum, uint32_t integralHeadNum, uint32_t qSThisSubBlock, uint32_t embed, uint32_t embedRound, uint32_t oHiddenSize)
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
            innerGOUbOffset += proTokenNum * embedRound;
        }
        for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor16[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    qSThisSubBlock, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += qSThisSubBlock * embedRound;
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
                gOutput[innerOGmOffset + proTokenIdx * oHiddenSize_gmlo],
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
        uint32_t taskStateSlot,
        uint32_t needRowLoop, uint32_t isLastRowLoop, uint32_t rowOffsetLoop,
        uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t epiTokenNum, uint32_t integralHeadNum,
        const SplitKVParams &splitParams,
        uint32_t rowOffsetCurLoop, int32_t invalidSuffixStartRow, int32_t invalidPrefixEndRow, uint32_t qSeqlen,
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
        uint32_t stateRowOffset = taskStateSlot * 64 + rowOffsetLoop;
        auto lseUbTensor = lseStagingUbTensor;
        auto lseBroadcastUbTensor = lseUbTensor[FLOAT_VECTOR_SIZE];

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
                glUbTensor.ReinterpretCast<uint32_t>()[stateRowOffset],
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

            if constexpr (HAS_DROPOUT_) {
                // go = go * (1 / (1 - p))
                AscendC::Muls<float, false>(
                    goUbTensor32, goUbTensor32, dropoutValue, (uint64_t)0,
                    CeilDiv(curRowNum * embedRound, FLOAT_VECTOR_SIZE), AscendC::UnaryRepeatParams());
                AscendC::PipeBarrier<PIPE_V>();
            }

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
            uint32_t subBlockStart = (curQNBlockTile == 1U) ? rowStart  : (rowStart >= qSeqlen ? rowStart - rowStart / qSeqlen * qSeqlen : rowStart);
            if (!splitParams.isSplitkv) {
                if (curQNBlockTile == 1U) {
                    ClearInvalidOutputRows(
                        0U, rowStart, curRowNum, invalidSuffixStartRow, invalidPrefixEndRow, qSeqlen, embedRound);
                } else {
                    uint32_t innerGOUbRowOffset = 0U;
                    uint32_t qBlockStart = qSBlockIdx * VECTOR_SIZE;
                    if (proTokenNum != 0U) {
                        ClearInvalidOutputRows(
                            innerGOUbRowOffset, subBlockStart, proTokenNum,
                            invalidSuffixStartRow, invalidPrefixEndRow, qSeqlen, embedRound);
                        innerGOUbRowOffset += proTokenNum;
                    }
                    for (uint32_t qNIdx = 0U; qNIdx < integralHeadNum; qNIdx++) {
                        ClearInvalidOutputRows(
                            innerGOUbRowOffset, qBlockStart, qSThisSubBlock,
                            invalidSuffixStartRow, invalidPrefixEndRow, qSeqlen, embedRound);
                        innerGOUbRowOffset += qSThisSubBlock;
                    }
                    if (epiTokenNum != 0U) {
                        ClearInvalidOutputRows(
                            innerGOUbRowOffset, qBlockStart, epiTokenNum,
                            invalidSuffixStartRow, invalidPrefixEndRow, qSeqlen, embedRound);
                    }
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
                    qSThisSubBlock, embed, embedRound, oHiddenSize);
            }
            if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
                if (isLastRowLoop) {
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Ln<float, false>(
                        lseUbTensor,
                        glUbTensor[taskStateSlot * 64],
                        (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                        AscendC::UnaryRepeatParams(1, 1, 8, 8));

                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<float, false>(
                        lseUbTensor,
                        lseUbTensor,
                        gmUbTensor[taskStateSlot * 64],
                        (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                        AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                    AscendC::PipeBarrier<PIPE_V>();

                    InvalidLineLSEProcess(qNThisSubBlock, invalidSuffixStartRow, qSBlockIdx,
                            inRowOffsetThisSubBlock, totalRowNum, invalidPrefixEndRow, qSeqlen, qSThisSubBlock,
                            lseUbTensor);
                    AscendC::Brcb(
                        lseBroadcastUbTensor.ReinterpretCast<uint32_t>(),
                        lseUbTensor.ReinterpretCast<uint32_t>(),
                        CeilDiv(totalRowNum, FLOAT_BLOCK_SIZE),
                        AscendC::BrcbRepeatParams(1, 8));
                    if (!splitParams.isSplitkv) {
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);

                    if (splitParams.isSplitkv) {
                        // isSplitkv: per-head strided write to token-major gCombineLse. UNCHANGED.
                        if (qNThisSubBlock == 0U) {
                            AscendC::DataCopyPad(
                                splitParams.gCombineLse, lseBroadcastUbTensor,
                                AscendC::DataCopyExtParams(
                                    totalRowNum, sizeof(float), 0, (qHeads_gmlse - 1) * sizeof(float), 0));
                        } else {
                            for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                                AscendC::DataCopyPad(
                                    splitParams.gCombineLse[qNIdx],
                                    lseBroadcastUbTensor[qNIdx * qSBlockSize * FLOAT_BLOCK_SIZE],
                                    AscendC::DataCopyExtParams(
                                        qSBlockSize, sizeof(float), 0, (qHeads_gmlse - 1) * sizeof(float), 0));
                                }
                        }
                    } else {
                        // BNS/single-batch NT heads are contiguous; multi-batch NT strides by total tokens.
                        uint32_t lseHeadCount = (qNThisSubBlock == 0U) ? 1U : qNThisSubBlock;
                        uint32_t lseSeqLen = totalRowNum / lseHeadCount;
                        uint32_t lseHeadStrideGm = layoutLse.stride(0);
                        bool isLseContiguous = (lseHeadCount == 1U) || (lseHeadStrideGm == lseSeqLen);
                        if (isLseContiguous) {
                            AscendC::DataCopyPad(
                                gLse, lseUbTensor,
                                AscendC::DataCopyExtParams(
                                    1, totalRowNum * sizeof(float), 0, 0, 0));
                        } else if (lseSeqLen % FLOAT_BLOCK_SIZE == 0U) {
                            AscendC::DataCopyPad(
                                gLse, lseUbTensor,
                                AscendC::DataCopyExtParams(
                                    lseHeadCount,
                                    lseSeqLen * sizeof(float),
                                    0,
                                    (lseHeadStrideGm - lseSeqLen) * sizeof(float),
                                    0));
                        } else {
                            // MTE3 rounds UB blocks to 32 B; broadcast rows keep scalar sources aligned.
                            for (uint32_t sIdx = 0; sIdx < lseSeqLen; ++sIdx) {
                                AscendC::DataCopyPad(
                                    gLse[sIdx],
                                    lseBroadcastUbTensor[sIdx * FLOAT_BLOCK_SIZE],
                                    AscendC::DataCopyExtParams(
                                        lseHeadCount,
                                        sizeof(float),
                                        lseSeqLen - 1U,
                                        (lseHeadStrideGm - 1U) * sizeof(float),
                                        0));
                            }
                        }
                    }
                    uint32_t taskStateEventId = taskStateSlot == 0 ? EVENT_ID4 :
                            (taskStateSlot == 1 ? EVENT_ID6 : EVENT_ID7);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(taskStateEventId);
                }
            } else {
                // FD SplitKV must still write partial LSE even when LSE_MODE != OUT_ONLY,
                // because the combine epilogue requires per-split LSE for rescaling.
                if (splitParams.isSplitkv) {
                    if (isLastRowLoop) {
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Ln<float, false>(
                            lseUbTensor,
                            glUbTensor[taskStateSlot * 64],
                            (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                            AscendC::UnaryRepeatParams(1, 1, 8, 8));

                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add<float, false>(
                            lseUbTensor,
                            lseUbTensor,
                            gmUbTensor[taskStateSlot * 64],
                            (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                        AscendC::PipeBarrier<PIPE_V>();

                        AscendC::Brcb(
                            lseBroadcastUbTensor.ReinterpretCast<uint32_t>(),
                            lseUbTensor.ReinterpretCast<uint32_t>(),
                            CeilDiv(totalRowNum, FLOAT_BLOCK_SIZE),
                            AscendC::BrcbRepeatParams(1, 8));
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);

                        if (qNThisSubBlock == 0U) {
                            AscendC::DataCopyPad(
                                splitParams.gCombineLse, lseBroadcastUbTensor,
                                AscendC::DataCopyExtParams(
                                    totalRowNum, sizeof(float), 0, (qHeads_gmlse - 1) * sizeof(float), 0));
                        } else {
                            for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                                AscendC::DataCopyPad(
                                    splitParams.gCombineLse[qNIdx],
                                    lseBroadcastUbTensor[qNIdx * qSBlockSize * FLOAT_BLOCK_SIZE],
                                    AscendC::DataCopyExtParams(
                                        qSBlockSize, sizeof(float), 0, (qHeads_gmlse - 1) * sizeof(float), 0));
                            }
                        }
                        uint32_t taskStateEventId = taskStateSlot == 0 ? EVENT_ID4 :
                            (taskStateSlot == 1 ? EVENT_ID6 : EVENT_ID7);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(taskStateEventId);
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
        Arch::CrossCoreFlag pvReady,
        uint32_t qSBlockSize, uint32_t qNBlockSize,
        uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod,
        uint32_t taskStateSlot,
        const SplitKVParams& splitParams = SplitKVParams(),
        int32_t invalidSuffixStartRow = 0, int32_t invalidPrefixEndRow = 0, uint32_t qSeqlen = 0,
        uint32_t qSBlockIdx = 0, uint32_t curQNBlockTile = 1)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t embed = actualBlockShape.n();
        uint32_t embedRoundV = (layoutInput.stride(0) == 0) ? BLOCK_SIZE : layoutInput.stride(0);
        uint32_t maxRowNumPerLoop = MAX_UB_O_ELEM_NUM / embedRoundV;
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

        // Idle vectors must consume PV-ready to avoid leaving a stale flag for the next task.
        if (inRowActualThisSubBlock == 0U) {
            Arch::CrossCoreWaitFlag(pvReady);
            return;
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

                // Fixed 64-row vector partitions prevent cross-task O-update overwrites.
                int64_t offsetUpdate = layoutUpdate.GetOffset(
                    MatrixCoord(subBlockIdx * FLOAT_VECTOR_SIZE + rowOffsetLoop, 0));
                auto gUpdateCurLoop = gUpdate[offsetUpdate];
                auto layoutUpdateCurLoop = layoutUpdate.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));

                proTokenIdx = rowOffsetLoop % qSThisSubBlock;
                proTokenNum = AscendC::Std::min(rowActualCurLoop, (qSThisSubBlock - proTokenIdx)) % qSThisSubBlock;
                integralHeadNum = (rowActualCurLoop - proTokenNum) / qSThisSubBlock;
                epiTokenNum = rowActualCurLoop - proTokenNum - integralHeadNum * qSThisSubBlock;

                if (rowLoopIdx == 0) {
                    Arch::CrossCoreWaitFlag(pvReady);
                }
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
                    taskStateSlot,
                    needRowLoop,
                    (rowLoopIdx == rowLoop - 1U),
                    rowOffsetLoop,
                    proTokenIdx,
                    proTokenNum,
                    epiTokenNum,
                    integralHeadNum,
                    blockParams,
                    rowOffsetCurLoop,
                    invalidSuffixStartRow,
                    invalidPrefixEndRow,
                    qSeqlen,
                    qSBlockIdx,
                    rowNum,
                    inRowOffsetThisSubBlock,
                    curQNBlockTile);
            }
        }
    }

private:
    float dropoutValue;
    AscendC::LocalTensor<float> loUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<float> hmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> tvUbTensor;
    AscendC::LocalTensor<ElementOutput> goUbTensor16;
    AscendC::LocalTensor<float> goUbTensor32;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> lseStagingUbTensor;
};

}

#endif
