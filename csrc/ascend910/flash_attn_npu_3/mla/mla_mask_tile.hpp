/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// CANN attention/common/op_kernel/vector_common.h:
// compressed triangular mask copy, S1G expansion and SelectWithBytesMask.
// FA3 supplies implicit causal/window bounds instead of a public mask tensor.
#pragma once
#include "kernel_operator.h"

namespace flash_attn::mla::Tile {
using namespace AscendC;
constexpr uint32_t kMlaMaskDim = 2048;
constexpr uint32_t kMlaMaskTileBytes = 8192;

__aicore__ inline uint32_t CausalEmptyRows(uint32_t row, uint32_t count, uint32_t heads, int64_t next)
{
    if (next >= 0)
        return 0;
    const uint64_t end = uint64_t(-next) * heads;
    return end <= row ? 0 : Min(uint64_t(count), end - row);
}

__aicore__ inline bool NeedsWindowMask(uint32_t row, uint32_t rows, uint32_t heads, int64_t pre, int64_t next,
                                       uint32_t key, uint32_t columns)
{
    const int64_t first = row / heads, last = (uint64_t(row) + rows - 1) / heads;
    return int64_t(key) + columns - 1 > first + next || int64_t(key) < last - pre;
}

__aicore__ inline void TriangleMask(LocalTensor<float> scores, LocalTensor<uint8_t> storage,
                                    LocalTensor<uint8_t> scratch, uint64_t maskAddress, uint32_t row, uint32_t rows,
                                    uint32_t heads, int64_t next, uint32_t key, uint32_t columns,
                                    uint32_t actualColumns, bool invert)
{
    const int64_t firstQuery = row / heads, lastQuery = (uint64_t(row) + rows - 1) / heads;
    const int64_t firstKey = key, lastKey = int64_t(key) + actualColumns - 1;
    const bool above = firstKey > lastQuery + next;
    const bool below = lastKey <= firstQuery + next;
    if (invert ? above : below)
        return;
    if (invert ? below : above) {
        Duplicate(scores, -1000000000000.0f, rows * columns);
        PipeBarrier<PIPE_V>();
        return;
    }

    // Only a diagonal intersection reaches this copy. Its canonical offset
    // is within the shared 2048-square triangle for the validated M/N tiles.
    const int64_t delta = firstKey - firstQuery - next;
    const uint32_t maskRow = delta < 0 ? uint32_t(-delta) : 0;
    const uint32_t maskColumn = delta > 0 ? uint32_t(delta) : 0;
    GlobalTensor<uint8_t> triangle;
    triangle.SetGlobalBuffer((__gm__ uint8_t*)maskAddress);
    const uint32_t queryRows = lastQuery - firstQuery + 1;
    auto ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(ready);
    WaitFlag<HardEvent::V_MTE2>(ready);
    DataCopyPad(storage, triangle[uint64_t(maskRow) * kMlaMaskDim + maskColumn],
                DataCopyExtParams{uint16_t(queryRows), actualColumns, kMlaMaskDim - actualColumns, 0, 0},
                DataCopyPadExtParams<uint8_t>{true, 0, uint8_t(columns - actualColumns), 0});
    ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(ready);
    WaitFlag<HardEvent::MTE2_V>(ready);

    auto mask = storage;
    if (heads != 1) {
        // The second half is the source S1G expansion scratch. Repeated heads
        // share one query-mask row; copy all heads in one vector repeat.
        auto expanded = storage[kMlaMaskTileBytes].ReinterpretCast<int16_t>();
        auto input = storage.ReinterpretCast<int16_t>();
        SetMaskCount();
        SetVectorMask<int16_t, MaskMode::COUNTER>(columns / sizeof(int16_t));
        for (uint32_t done = 0, query = 0; done < rows; ++query) {
            const uint32_t count = Min(rows - done, heads - (row + done) % heads);
            for (uint32_t copied = 0; copied < count;) {
                const uint32_t repeat = Min(255U, count - copied);
                Copy<int16_t, false>(expanded[(done + copied) * columns / sizeof(int16_t)],
                                     input[query * columns / sizeof(int16_t)], MASK_PLACEHOLDER, repeat,
                                     {1, 1, uint16_t(columns / 32), 0});
                copied += repeat;
            }
            done += count;
        }
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
        mask = expanded.ReinterpretCast<uint8_t>();
    }
    SelectWithBytesMaskShapeInfo shape;
    shape.firstAxis = rows;
    shape.srcLastAxis = columns;
    shape.maskLastAxis = columns;
    const auto sourceSize = scores.GetSize();
    mask.SetSize(rows * columns);
    scores.SetSize(rows * columns);
    if (invert)
        SelectWithBytesMask(scores, -1000000000000.0f, scores, mask, scratch, shape);
    else
        SelectWithBytesMask(scores, scores, -1000000000000.0f, mask, scratch, shape);
    scores.SetSize(sourceSize);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void WindowMask(LocalTensor<float> scores, LocalTensor<uint8_t> mask, LocalTensor<uint8_t> scratch,
                                  uint64_t maskAddress, uint32_t row, uint32_t rows, uint32_t heads, int64_t pre,
                                  int64_t next, uint32_t key, uint32_t columns, uint32_t actualColumns)
{
    TriangleMask(scores, mask, scratch, maskAddress, row, rows, heads, next, key, columns, actualColumns, false);
    TriangleMask(scores, mask, scratch, maskAddress, row, rows, heads, -pre - 1, key, columns, actualColumns, true);
}
// CANN9.3 causal/band BOOL SG source branches.

namespace PrefillMask {
enum SparseMode : uint8_t {
    RIGHT_DOWN_CAUSAL = 3,
    BAND = 4
};
enum LAYOUT_Q {
    GS,
    SG
};
struct MaskInfo {
    uint32_t gs1StartIdx, gs1dealNum, s1Size, gSize, s2StartIdx, s2dealNum, s2Size;
    int64_t preToken = 0, nextToken = 0;
    uint32_t attenMaskStride;
    LAYOUT_Q layout;
    SparseMode sparseMode;
    uint32_t maskValue;
};
__aicore__ inline uint32_t Align(uint32_t n, uint32_t a)
{
    return (n + a - 1) / a * a;
}

__aicore__ inline uint64_t ComputeAttenMaskOffsetCompress(MaskInfo& info, uint32_t s1StartIdx)
{
    int64_t nextToken = 0;
    if (info.sparseMode == RIGHT_DOWN_CAUSAL) {
        nextToken = static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    } else if (info.sparseMode == BAND) {
        nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    }
    uint64_t offset = 0;
    int64_t delta = nextToken + s1StartIdx - info.s2StartIdx;
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    if (delta < 0) {
        offset = (-delta) < static_cast<int64_t>(info.gs1dealNum) ? (-delta) : info.gs1dealNum;
    } else {
        offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) * info.attenMaskStride;
    }
    return offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffsetCompressPre(MaskInfo& info, uint32_t s1StartIdx)
{
    int64_t preToken = info.preToken + static_cast<int64_t>(info.s1Size) - static_cast<int64_t>(info.s2Size);
    int64_t delta = -preToken + static_cast<int64_t>(s1StartIdx) - static_cast<int64_t>(info.s2StartIdx) - 1;
    uint64_t offset = 0;
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    if (delta < 0) {
        offset = (-delta) < static_cast<int64_t>(info.gs1dealNum) ? (-delta) : info.gs1dealNum;
    } else {
        offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) * info.attenMaskStride;
    }
    return offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffset(MaskInfo& info, uint32_t s1StartIdx = 0, bool isPre = false)
{
    if (isPre)
        return ComputeAttenMaskOffsetCompressPre(info, s1StartIdx);
    return ComputeAttenMaskOffsetCompress(info, s1StartIdx);
}

template <typename T>
__aicore__ inline void AttentionmaskDataCopy(LocalTensor<T>& attenMaskUb, GlobalTensor<T>& srcGmAddr, MaskInfo& info,
                                             uint32_t s1StartIdx, uint32_t s1EndIdx, bool isPre = false)
{
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    uint64_t maskOffset = ComputeAttenMaskOffset(info, s1StartIdx, isPre);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
    dataCopyParams.blockLen = info.s2dealNum;
    dataCopyParams.srcStride = info.attenMaskStride - info.s2dealNum;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 0};

    DataCopyPad(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);
}

template <typename T, typename U>
__aicore__ inline void AttentionmaskCopyInForSgLayout(LocalTensor<T>& attenMaskUb, GlobalTensor<T>& srcGmAddr,
                                                      LocalTensor<U>& tmpBuf, MaskInfo& info, bool isPre = false)
{
    uint32_t s1StartIdx = info.gs1StartIdx / info.gSize;
    uint32_t s1EndIdx = (info.gs1StartIdx + info.gs1dealNum - 1) / info.gSize;
    uint32_t s1Count = s1EndIdx - s1StartIdx + 1;
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);

    AttentionmaskDataCopy<T>(attenMaskUb, srcGmAddr, info, s1StartIdx, s1EndIdx + 1, isPre);

    event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(enQueEvtID);
    WaitFlag<HardEvent::MTE2_V>(enQueEvtID);

    LocalTensor<int16_t> attenMaskUbDst = tmpBuf.template ReinterpretCast<int16_t>();
    LocalTensor<int16_t> mask16 = attenMaskUb.template ReinterpretCast<int16_t>();
    uint32_t headGCount = s1Count > 1 ? (info.gSize - info.gs1StartIdx % info.gSize) : info.gs1dealNum;
    uint32_t dstMaskOffset = 0;
    uint32_t srcMaskBaseOffset = 0;

    SetMaskCount();
    SetVectorMask<int16_t, MaskMode::COUNTER>(attenMaskSizeAlign / 2);
    Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset], AscendC::MASK_PLACEHOLDER,
                         headGCount, {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
    dstMaskOffset += headGCount * attenMaskSizeAlign / sizeof(int16_t);
    srcMaskBaseOffset += attenMaskSizeAlign / sizeof(int16_t);

    uint32_t reminRowCount = info.gs1dealNum - headGCount;
    uint32_t midS1Count = reminRowCount / info.gSize;
    uint32_t tailGSize = reminRowCount % info.gSize;

    for (uint32_t midIdx = 0; midIdx < midS1Count; midIdx++) {
        Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset], AscendC::MASK_PLACEHOLDER,
                             info.gSize, {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
        dstMaskOffset += info.gSize * attenMaskSizeAlign / sizeof(int16_t);
        srcMaskBaseOffset += attenMaskSizeAlign / sizeof(int16_t);
    }

    if (tailGSize > 0) {
        Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset], AscendC::MASK_PLACEHOLDER,
                             tailGSize, {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
    }
    SetMaskNorm();
    ResetMask();
    attenMaskUb = attenMaskUbDst.template ReinterpretCast<T>();
}

__aicore__ inline bool IsSkipAttentionmask(MaskInfo& info)
{

    int32_t s1StartIdx = info.layout == GS ? info.gs1StartIdx % info.s1Size : info.gs1StartIdx / info.gSize;
    if (info.layout == GS && s1StartIdx + info.gs1dealNum > info.s1Size) {
        return false;
    }

    int64_t nextToken = 0;
    if (info.sparseMode == RIGHT_DOWN_CAUSAL) {
        nextToken = static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    } else if (info.sparseMode == BAND) {
        nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    }

    if (static_cast<int64_t>(info.s2StartIdx + info.s2dealNum) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
        return true;
    }
    return false;
}

__aicore__ inline bool IsSkipAttentionmaskForPre(MaskInfo& info)
{
    if (info.sparseMode != BAND) {
        return true;
    }

    int32_t s1StartIdx = info.layout == GS ? info.gs1StartIdx % info.s1Size : info.gs1StartIdx / info.gSize;
    if (info.layout == GS && s1StartIdx + info.gs1dealNum > info.s1Size) {
        return false;
    }

    int64_t preToken = info.preToken + static_cast<uint64_t>(info.s1Size) - static_cast<uint64_t>(info.s2Size);
    int32_t s1EndIdx =
        info.layout == GS ? s1StartIdx + info.gs1dealNum : (info.gs1StartIdx + info.gs1dealNum) / info.gSize;

    if (static_cast<int64_t>(info.s2StartIdx) + preToken >= static_cast<int64_t>(s1EndIdx)) {
        return true;
    }
    return false;
}

template <typename T, typename U>
__aicore__ inline void AttentionmaskCopyIn(LocalTensor<T>& attenMaskUb, GlobalTensor<T>& srcGmAddr,
                                           LocalTensor<U>& tmpBuf, MaskInfo& info, bool isPre = false)
{
    if (info.layout == SG) {
        AttentionmaskCopyInForSgLayout(attenMaskUb, srcGmAddr, tmpBuf, info, isPre);
    }
}

template <typename T, typename M, typename U>
__aicore__ inline void AttentionMaskCompute(LocalTensor<T>& dstUb, LocalTensor<T>& srcUb, LocalTensor<M>& attenMaskUb,
                                            LocalTensor<U>& tmpBuf, MaskInfo& info, bool isPre = false)
{
    uint32_t dealRowCount = info.gs1dealNum;
    uint32_t columnCount = Align(info.s2dealNum, 32U);
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);

    SelectWithBytesMaskShapeInfo selectWithBytesMaskShapeInfo;
    selectWithBytesMaskShapeInfo.firstAxis = dealRowCount;
    selectWithBytesMaskShapeInfo.srcLastAxis = columnCount;
    selectWithBytesMaskShapeInfo.maskLastAxis = attenMaskSizeAlign;
    attenMaskUb.SetSize(dealRowCount * attenMaskSizeAlign);
    srcUb.SetSize(dealRowCount * columnCount);
    if (isPre) {
        SelectWithBytesMask(dstUb, *((T*)&info.maskValue), srcUb, attenMaskUb, tmpBuf, selectWithBytesMaskShapeInfo);
    } else {
        SelectWithBytesMask(dstUb, srcUb, *((T*)&info.maskValue), attenMaskUb, tmpBuf, selectWithBytesMaskShapeInfo);
    }
    constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    srcUb.SetSize(BUFFER_SIZE_BYTE_32K / sizeof(T));
}

} // namespace PrefillMask
} // namespace flash_attn::mla::Tile
