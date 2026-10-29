/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once

#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include <type_traits>
#include "lib/math/tanh.h"

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0.
// attention/common/op_kernel/vector_common.h: RowDivs, RowMuls, RowSum, ComputeSoftMaxLse.
// Preserve source vector instruction order and row/column repetition choices.

namespace flash_attn::mla::Tile {
using namespace AscendC;
constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = 64;
constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = 8;
constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

__aicore__ inline void RowSum(LocalTensor<float>& dstUb, LocalTensor<float> srcUb, uint32_t dealRowCount,
                              uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t blockCount = actualColumnCount / FP32_REPEAT_ELEMENT_NUM;
    uint32_t remain = actualColumnCount % FP32_REPEAT_ELEMENT_NUM;
    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParams.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    if (blockCount > 0 && remain > 0) {
        Add(srcUb, srcUb, srcUb[blockCount * FP32_REPEAT_ELEMENT_NUM], remain, dealRowCount, repeatParams);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t pairs = blockCount / 2; pairs > 0; pairs = blockCount / 2) {
        blockCount = (blockCount + 1) / 2;
        for (uint32_t j = 0; j < pairs; j++) {
            Add(srcUb[j * FP32_REPEAT_ELEMENT_NUM], srcUb[j * FP32_REPEAT_ELEMENT_NUM],
                srcUb[(j + blockCount) * FP32_REPEAT_ELEMENT_NUM], FP32_REPEAT_ELEMENT_NUM, dealRowCount, repeatParams);
        }
        PipeBarrier<PIPE_V>();
    }
    WholeReduceSum(dstUb, srcUb, Min(actualColumnCount, FP32_REPEAT_ELEMENT_NUM), dealRowCount, 1, 1,
                   columnCount / FP32_BLOCK_ELEMENT_NUM);
}

template <typename T>
__aicore__ inline void RowDivs(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM]
    // dstUb:[dealRowCount, columnCount]
    uint32_t repeatNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockNum = FP32_BLOCK_ELEMENT_NUM;
    if constexpr (std::is_same<T, half>::value) {
        repeatNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockNum = FP32_BLOCK_ELEMENT_NUM * 2;
    }
    uint32_t dLoop = actualColumnCount / repeatNum;
    uint32_t dRemain = actualColumnCount % repeatNum;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / blockNum;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = columnCount / blockNum;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, repeatNum, dealRowCount, repeatParamsDiv);
            offset += repeatNum;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * blockNum], repeatNum, columnRepeatCount, columnRepeatParams);
            offset += columnCount;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * repeatNum], src0Ub[dLoop * repeatNum], src1Ub, dRemain, dealRowCount, repeatParamsDiv);
    }
}

template <typename T>
__aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM]
    // dstUb:[dealRowCount, columnCount] dealRowCount is repeat times, must be less 256
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (dLoop <= dealRowCount) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum,
                    dLoop, columnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, dealRowCount,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t i = 0; i < dealRowCount; i++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                repeatParams);
            //  计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * columnCount + dLoop * repeatElementNum],
                    src0Ub[i * columnCount + dLoop * repeatElementNum], src1Ub[i * blockElementNum], dRemain, 1,
                    repeatParams);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ComputeSoftMaxLse(LocalTensor<T>& softmaxlseUb, LocalTensor<T>& softmaxSumUb,
                                         LocalTensor<T>& softmaxMaxUb, uint32_t dealRowCount)
{
    uint32_t blockNum = FP32_BLOCK_ELEMENT_NUM;
    if constexpr (std::is_same<T, half>::value) {
        blockNum = FP32_BLOCK_ELEMENT_NUM * 2;
    }
    uint64_t dealRowCountAlign = dealRowCount * FP32_BLOCK_ELEMENT_NUM;
    Log(softmaxlseUb, softmaxSumUb, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();
    Add(softmaxlseUb, softmaxlseUb, softmaxMaxUb, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();
}

} // namespace flash_attn::mla::Tile

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// arch22/fia_block_vec_nonquant_mla.h::ElewiseCompute / DealInvalidRows.
// Specialize boolean right-down causal/band masks to S1G rows. Generate the mask
// from validated effective lengths instead of loading a public mask tensor.
// Preserve the source finite sentinel and softmax/aMLA arithmetic; adapt only
// fully masked O/LSE to the FA3 contract in the Block epilogue.

namespace flash_attn::mla::Tile {
using namespace AscendC;

__aicore__ inline uint32_t CausalEmptyRows(uint32_t row, uint32_t count, uint32_t heads, int64_t nextToken)
{
    if (nextToken >= 0) {
        return 0;
    }
    const uint64_t end = uint64_t(-nextToken) * heads;
    return end <= row ? 0 : Min(uint64_t(count), end - row);
}

__aicore__ inline void CausalMask(LocalTensor<float> scores, uint32_t row, uint32_t rows, uint32_t heads,
                                  int64_t nextToken, uint32_t keyStart, uint32_t columns, uint32_t actualColumns)
{
    // Operate at aligned vector addresses. One bit mask covers 64 FP32 lanes;
    // repeat along heads belonging to the same query, including head tails.
    for (uint32_t start = 0; start < rows;) {
        const uint32_t query = (row + start) / heads;
        const uint32_t count = Min(rows - start, heads - (row + start) % heads);
        const int64_t visible = int64_t(query) + nextToken + 1 - keyStart;
        if (visible < actualColumns) {
            const uint32_t first = visible > 0 ? uint32_t(visible) : 0;
            for (uint32_t col = first / 64 * 64; col < columns; col += 64) {
                const uint32_t begin = col < first ? first - col : 0;
                const uint32_t end = Min(64U, columns - col);
                uint64_t bits[2] = {((~uint64_t(0)) << begin) & ((~uint64_t(0)) >> (64 - end)), 0};
                Duplicate(scores[start * columns + col], -1000000000000.0f, bits, count, 1, columns / 8);
            }
        }
        start += count;
    }
}

__aicore__ inline void WindowMask(LocalTensor<float> scores, uint32_t row, uint32_t rows, uint32_t heads,
                                  int64_t preToken, int64_t nextToken, uint32_t keyStart, uint32_t columns,
                                  uint32_t actualColumns)
{
    CausalMask(scores, row, rows, heads, nextToken, keyStart, columns, actualColumns);
    // The source band combines the upper triangle with the inverted lower
    // triangle. Preserve the same finite sentinel and aligned FP32 writes.
    for (uint32_t start = 0; start < rows;) {
        const uint32_t query = (row + start) / heads;
        const uint32_t count = Min(rows - start, heads - (row + start) % heads);
        const int64_t hidden = int64_t(query) - preToken - keyStart;
        const uint32_t limit = hidden > 0 ? Min(uint64_t(hidden), uint64_t(columns)) : 0;
        for (uint32_t col = 0; col < limit; col += 64) {
            const uint32_t end = Min(64U, limit - col);
            uint64_t bits[2] = {(~uint64_t(0)) >> (64 - end), 0};
            Duplicate(scores[start * columns + col], -1000000000000.0f, bits, count, 1, columns / 8);
        }
        start += count;
    }
}
} // namespace flash_attn::mla::Tile

// FA3 softcap applies to the complete scaled score, before masking.
// Reuse AscendC's FP32 tanh and the existing softmax scratch allocation.

namespace flash_attn::mla::Tile {
__aicore__ inline void Softcap(AscendC::LocalTensor<float> scores, AscendC::LocalTensor<uint8_t> scratch,
                               uint32_t count, float cap)
{
    if (cap == 0.0f)
        return;
    // Four 8 KiB regions fit in the existing softmax scratch. Tanh's input,
    // output and temporary tensors are disjoint. All scratch contents are
    // dead before SoftmaxFlashV2; no extra UB or event is needed.
    constexpr uint32_t chunkSize = 2048;
    auto normalized = scratch.ReinterpretCast<float>();
    auto temporary = scratch[chunkSize * sizeof(float)];
    temporary.SetSize(chunkSize * sizeof(float));
    auto polynomial = scratch[2 * chunkSize * sizeof(float)].ReinterpretCast<float>();
    auto square = scratch[3 * chunkSize * sizeof(float)].ReinterpretCast<float>();
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t start = 0; start < count; start += chunkSize) {
        uint32_t size = count - start < chunkSize ? count - start : chunkSize;
        AscendC::Duplicate(normalized, cap, size);
        AscendC::PipeBarrier<PIPE_V>();
        // Division also handles very small positive caps without overflowing
        // a precomputed reciprocal (zero scores must remain zero).
        AscendC::Div(normalized, scores[start], normalized, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Tanh(scores[start], normalized, temporary, size);
        AscendC::PipeBarrier<PIPE_V>();
        // AscendC tanh uses exp(2*x)-1. For |x| <= 1/8, evaluate its
        // odd Taylor polynomial through x^7 instead: the relative remainder
        // is < 1.4e-9, below FP32 precision, without cancellation near zero.
        // Reuse tanh's now-dead temporary as the Select bit mask.
        AscendC::Mul(square, normalized, normalized, size);
        AscendC::PipeBarrier<PIPE_V>();
        // On 910B CompareScalar emits complete 64-float repeats only. Include
        // the final repeat so a 32-lane tail does not retain Tanh scratch bits.
        // The rounded read stays within square's 2048 elements; Select below
        // consumes only the valid size lanes, never the unused padding bits.
        AscendC::CompareScalar(temporary, square, 0.015625f, AscendC::CMPMODE::LE, (size + 63) / 64 * 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(square, square, 0.015625f, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(polynomial, square, -17.0f / 315.0f, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(polynomial, polynomial, 2.0f / 15.0f, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(polynomial, polynomial, square, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(polynomial, polynomial, -1.0f / 3.0f, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(polynomial, polynomial, square, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(polynomial, polynomial, 1.0f, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(normalized, normalized, polynomial, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(scores[start], temporary, normalized, scores[start], AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                        size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(scores[start], scores[start], cap, size);
        AscendC::PipeBarrier<PIPE_V>();
    }
}
} // namespace flash_attn::mla::Tile
