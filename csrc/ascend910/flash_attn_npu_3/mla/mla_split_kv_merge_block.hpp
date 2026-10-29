/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Source: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0,
// arch22/fia_block_vec_flashdecode_arch22.h: ComputeScaleValue,
// ReduceFinalRes, CopyFinalResOut. FP16/BF16; 512+64 and 192/128 numerical specializations.
#pragma once
#include "catlass/catlass.hpp"
#include "mla_fd_ops_tile.hpp"
#include "mla_params.h"
namespace flash_attn::mla::Block {
using namespace AscendC;

template <typename OutputT, bool WriteSingleAccumulator = false, bool IsTnd = false, uint32_t HeadDim = 512,
          bool GroupedOutput = HeadDim == 128>
class MlaSplitKvMerge {
    static_assert(HeadDim == 512 || HeadDim == 128);
    static constexpr uint32_t kOutputEvent = 6;
    static constexpr uint32_t kLseEvent = 7;
    static constexpr uint32_t kStatisticsEvent = 8;
    static constexpr uint32_t kPartialEvent = 10;

  public:
    __aicore__ inline void Init(const MlaSplitKvMergeParams& p, GM_ADDR out, GM_ADDR sum, GM_ADDR maximum,
                                GM_ADDR result, GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse, TPipe* pipe,
                                float softcap = 0.0f)
    {
        InitBuffers(pipe);
        Bind(p, out, sum, maximum, result, lse, accum, accumLse, softcap);
    }

    __aicore__ inline void Bind(const MlaSplitKvMergeParams& p, GM_ADDR out, GM_ADDR sum, GM_ADDR maximum,
                                GM_ADDR result, GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse, float softcap = 0.0f)
    {
        params = p;
        roundToEven = softcap != 0.0f;
        writeAccumulators = accum != nullptr;
        partial.SetGlobalBuffer((__gm__ float*)out);
        sums.SetGlobalBuffer((__gm__ float*)sum);
        maxima.SetGlobalBuffer((__gm__ float*)maximum);
        output.SetGlobalBuffer((__gm__ OutputT*)result);
        outputLse.SetGlobalBuffer((__gm__ float*)lse);
        accumulator.SetGlobalBuffer((__gm__ float*)accum);
        accumulatorLse.SetGlobalBuffer((__gm__ float*)accumLse);
    }

    __aicore__ inline void InitBuffers(TPipe* pipe)
    {
        // <=192 split-row pairs use the source 6 KiB bound. 256 explicit
        // splits use one row and 8 KiB. No uint8 vector repeat may be 256.
        pipe->InitBuffer(sumBuf[0], 8192);
        pipe->InitBuffer(sumBuf[1], 8192);
        pipe->InitBuffer(maxBuf[0], 8192);
        pipe->InitBuffer(maxBuf[1], 8192);
        pipe->InitBuffer(weightBuf, 8192);
        pipe->InitBuffer(rowMaxBuf[0], 256);
        pipe->InitBuffer(rowMaxBuf[1], 256);
        pipe->InitBuffer(rowSumBuf[0], 256);
        pipe->InitBuffer(rowSumBuf[1], 256);
        pipe->InitBuffer(lseBuf, 256);
        pipe->InitBuffer(partialBuf[0], 16384);
        pipe->InitBuffer(partialBuf[1], 16384);
        pipe->InitBuffer(reduceBuf, 16384);
        pipe->InitBuffer(castBuf, 8192);
        rowLoop = 0;
        partialLoop = 0;
        // Match native AllocEventID/FreeEventID: events and ping-pong parity
        // live for the whole FD phase, including Bind calls between groups.
        SetFlag<HardEvent::V_MTE2>(kStatisticsEvent);
        SetFlag<HardEvent::V_MTE2>(kStatisticsEvent + 1);
        SetFlag<HardEvent::V_MTE2>(kPartialEvent);
        SetFlag<HardEvent::V_MTE2>(kPartialEvent + 1);
        SetFlag<HardEvent::MTE3_V>(kOutputEvent);
        SetFlag<HardEvent::MTE3_V>(kLseEvent);
    }

    __aicore__ inline void Finish()
    {
        WaitFlag<HardEvent::V_MTE2>(kStatisticsEvent);
        WaitFlag<HardEvent::V_MTE2>(kStatisticsEvent + 1);
        WaitFlag<HardEvent::V_MTE2>(kPartialEvent);
        WaitFlag<HardEvent::V_MTE2>(kPartialEvent + 1);
        WaitFlag<HardEvent::MTE3_V>(kOutputEvent);
        WaitFlag<HardEvent::MTE3_V>(kLseEvent);
    }

    __aicore__ inline uint32_t TileRows() const
    {
        return Max(1U, Min(8U, 192U / params.splits));
    }

    __aicore__ inline bool Valid(uint32_t split, uint32_t row) const
    {
        int64_t lo = params.begin[split], hi = params.end[split];
        const int64_t query = (params.rowStart + row) / params.heads;
        const int64_t center = query + int64_t(params.kvLength) - (params.queryLength ? params.queryLength : params.sq);
        if (params.maskKind == 1) {
            hi = Min(hi, center + 1);
        } else if (params.maskKind == 2) {
            lo = Max(lo, center - params.left);
            hi = Min(hi, center + params.right + 1);
        }
        return lo < hi;
    }

    __aicore__ inline uint64_t RowAddress(uint32_t row) const
    {
        const uint64_t flat = uint64_t(params.rowStart) + row;
        if constexpr (IsTnd)
            return (uint64_t(GroupedOutput ? params.kvHead : 0) * params.heads + flat % params.heads) * params.sq +
                   params.batchIndex + flat / params.heads;
        if constexpr (GroupedOutput)
            return ((uint64_t(params.batchIndex) * params.kvHeads + params.kvHead) * params.heads +
                    flat % params.heads) *
                       params.sq +
                   flat / params.heads;
        return (uint64_t(params.batchIndex) * params.heads + flat % params.heads) * params.sq + flat / params.heads;
    }

    __aicore__ inline void Run(uint32_t start, uint32_t rows)
    {
        const uint32_t statisticsSlot = rowLoop % 2;
        auto sum = sumBuf[statisticsSlot].Get<float>(), maximum = maxBuf[statisticsSlot].Get<float>();
        auto weights = weightBuf.Get<float>();
        auto rowMax = rowMaxBuf[statisticsSlot].Get<float>(), rowSum = rowSumBuf[statisticsSlot].Get<float>();
        auto lse = lseBuf.Get<float>();
        const uint32_t width = rows * 8;
        WaitFlag<HardEvent::V_MTE2>(kStatisticsEvent + statisticsSlot);
        for (uint32_t split = 0; split < params.splits; ++split) {
            const uint64_t offset = (uint64_t(split) * params.mBase + start) * 8;
            DataCopy(sum[split * width], sums[offset], width);
            DataCopy(maximum[split * width], maxima[offset], width);
        }
        SetFlag<HardEvent::MTE2_V>(kStatisticsEvent + statisticsSlot);
        WaitFlag<HardEvent::MTE2_V>(kStatisticsEvent + statisticsSlot);
        // Preserve native FD's two preloaded O slots. The statistics and
        // weight calculation overlap these independent GM-to-UB transfers.
        const uint32_t preload = Min(2U, params.splits);
        for (uint32_t split = 0; split < preload; ++split) {
            LoadPartial(split, start, rows, (partialLoop + split) % 2);
        }
        bool any[8]{};
        for (uint32_t split = 0; split < params.splits; ++split) {
            for (uint32_t row = 0; row < rows; ++row) {
                const bool valid = Valid(split, start + row);
                any[row] = any[row] || valid;
                if (!valid) {
                    // Never infer row validity from the source's finite mask
                    // sentinel: a real logit may legitimately be that small.
                    Duplicate(sum[split * width + row * 8], 0.0f, 8);
                    Duplicate(maximum[split * width + row * 8], -3.402823466e38f, 8);
                }
            }
        }
        Duplicate(rowMax, -3.402823466e38f, width);
        Duplicate(rowSum, 0.0f, width);
        PipeBarrier<PIPE_V>();
        for (uint32_t s = 0; s < params.splits; s += 255) {
            Tile::ColMax(rowMax, maximum[s * width], rowMax, Min(255U, params.splits - s), width, width);
            PipeBarrier<PIPE_V>();
        }
        for (uint32_t s = 0; s < params.splits; s += 255) {
            Tile::RowSub(weights[s * width], maximum[s * width], rowMax, Min(255U, params.splits - s), width, width);
        }
        PipeBarrier<PIPE_V>();
        Exp(weights, weights, params.splits * width);
        PipeBarrier<PIPE_V>();
        Mul(weights, sum, weights, params.splits * width);
        PipeBarrier<PIPE_V>();
        for (uint32_t s = 0; s < params.splits; s += 255) {
            Tile::ColAdd(rowSum, weights[s * width], rowSum, Min(255U, params.splits - s), width, width);
            PipeBarrier<PIPE_V>();
        }
        bool emptyRows = false;
        for (uint32_t row = 0; row < rows; ++row) {
            if (!any[row]) {
                Duplicate(rowSum[row * 8], 1.0f, 8);
                emptyRows = true;
            }
        }
        if (emptyRows)
            PipeBarrier<PIPE_V>();
        for (uint32_t s = 0; s < params.splits; s += 255) {
            Tile::MatDivsVec(weights[s * width], weights[s * width], rowSum, Min(255U, params.splits - s), width,
                             width);
        }
        PipeBarrier<PIPE_V>();
        // Native computes final LSE after normalizing the FD weights. Only
        // wait for the previous LSE DMA when its output buffer is reused.
        WaitFlag<HardEvent::MTE3_V>(kLseEvent);
        Tile::ComputeSoftMaxLse(lse, rowSum, rowMax, rows);
        for (uint32_t row = 0; row < rows; ++row) {
            if (!any[row]) {
                Duplicate(lse[row * 8], -float(1.0 / 0.0), 8);
            }
        }
        SetFlag<HardEvent::V_MTE3>(kLseEvent);
        WaitFlag<HardEvent::V_MTE3>(kLseEvent);
        for (uint32_t row = 0; row < rows;) {
            const uint64_t gap = (uint64_t(params.sq) - 1) * 4;
            const uint32_t count =
                gap > UINT32_MAX ? 1 : Min(rows - row, params.heads - (params.rowStart + start + row) % params.heads);
            DataCopyPad(
                outputLse[RowAddress(start + row)], lse[row * 8],
                DataCopyExtParams{uint16_t(count), 4, 0, count == 1 ? 0 : uint32_t((uint64_t(params.sq) - 1) * 4), 0});
            row += count;
        }
        SetFlag<HardEvent::MTE3_V>(kLseEvent);

        const bool storePartials = writeAccumulators && (params.splits > 1 || WriteSingleAccumulator);
        if (storePartials) {
            // Scaling no longer reads raw sums. Reuse them for all partial
            // LSEs before writeback, retaining the per-row invalid mapping.
            Tile::ComputeSoftMaxLse(sum, sum, maximum, params.splits * rows);
        } else {
            SetFlag<HardEvent::V_MTE2>(kStatisticsEvent + statisticsSlot);
        }
        auto reduced = reduceBuf.Get<float>();
        for (uint32_t split = 0; split < params.splits; ++split) {
            const uint32_t slot = partialLoop % 2;
            if (split >= preload) {
                LoadPartial(split, start, rows, slot);
            }
            WaitFlag<HardEvent::MTE2_V>(kPartialEvent + slot);
            auto current = partialBuf[slot].Get<float>();
            bool invalidPartial = false;
            for (uint32_t row = 0; row < rows; ++row) {
                if (!Valid(split, start + row)) {
                    Duplicate(current[row * HeadDim], 0.0f, HeadDim);
                    invalidPartial = true;
                }
            }
            if (invalidPartial)
                PipeBarrier<PIPE_V>();
            if (storePartials) {
                auto partialLse = sum[split * width];
                for (uint32_t row = 0; row < rows; ++row) {
                    if (!Valid(split, start + row)) {
                        Duplicate(partialLse[row * 8], -float(1.0 / 0.0), 8);
                    }
                }
                if (invalidPartial)
                    PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                for (uint32_t row = 0; row < rows;) {
                    const uint64_t gap = (uint64_t(params.sq) - 1) * HeadDim * 4;
                    const uint32_t count =
                        gap > UINT32_MAX
                            ? 1
                            : Min(rows - row, params.heads - (params.rowStart + start + row) % params.heads);
                    const uint64_t offset = uint64_t(split) * params.batch * params.heads * params.sq *
                                                (GroupedOutput ? params.kvHeads : 1) +
                                            RowAddress(start + row);
                    DataCopyPad(accumulator[offset * HeadDim], current[row * HeadDim],
                                DataCopyExtParams{uint16_t(count), HeadDim * 4, 0, count == 1 ? 0 : uint32_t(gap), 0});
                    DataCopyPad(accumulatorLse[offset], partialLse[row * 8],
                                DataCopyExtParams{uint16_t(count), 4, 0,
                                                  count == 1 ? 0 : uint32_t((uint64_t(params.sq) - 1) * 4), 0});
                    row += count;
                }
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
            auto target = split == 0 ? reduced : current;
            Tile::RowMuls(target, current, weights[split * width], rows, HeadDim, HeadDim);
            if (split != 0) {
                PipeBarrier<PIPE_V>();
                Add(reduced, reduced, current, rows * HeadDim);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE2>(kPartialEvent + slot);
            ++partialLoop;
        }
        // Unlike native FD, raw sums also carry the public partial LSEs.
        // Every partial DMA above has completed before this slot is released.
        if (storePartials)
            SetFlag<HardEvent::V_MTE2>(kStatisticsEvent + statisticsSlot);
        PipeBarrier<PIPE_V>();
        auto cast = castBuf.Get<OutputT>();
        WaitFlag<HardEvent::MTE3_V>(kOutputEvent);
        if constexpr (std::is_same<OutputT, bfloat16_t>::value) {
            Cast(cast, reduced, RoundMode::CAST_RINT, rows * HeadDim);
        } else if (roundToEven) {
            Cast(cast, reduced, RoundMode::CAST_RINT, rows * HeadDim);
        } else {
            Cast(cast, reduced, RoundMode::CAST_ROUND, rows * HeadDim);
        }
        SetFlag<HardEvent::V_MTE3>(kOutputEvent);
        WaitFlag<HardEvent::V_MTE3>(kOutputEvent);
        const uint64_t outRow =
            uint64_t(params.batchIndex) * (IsTnd ? 1 : params.sq) * params.heads + params.rowStart + start;
        if constexpr (GroupedOutput) {
            for (uint32_t row = 0; row < rows;) {
                const uint64_t flat = uint64_t(params.rowStart) + start + row;
                const uint32_t count = Min(rows - row, params.heads - uint32_t(flat % params.heads));
                const uint64_t address =
                    ((uint64_t(params.batchIndex) * (IsTnd ? 1 : params.sq) + flat / params.heads) * params.kvHeads +
                     params.kvHead) *
                        params.heads +
                    flat % params.heads;
                DataCopy(output[address * HeadDim], cast[row * HeadDim], count * HeadDim);
                row += count;
            }
        } else
            DataCopy(output[outRow * HeadDim], cast, rows * HeadDim);
        SetFlag<HardEvent::MTE3_V>(kOutputEvent);
        ++rowLoop;
    }

  private:
    __aicore__ inline void LoadPartial(uint32_t split, uint32_t start, uint32_t rows, uint32_t slot)
    {
        WaitFlag<HardEvent::V_MTE2>(kPartialEvent + slot);
        DataCopyPad(partialBuf[slot].Get<float>(), partial[(uint64_t(split) * params.mBase + start) * HeadDim],
                    DataCopyExtParams{uint16_t(rows), HeadDim * 4, 0, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 0, 0.0f});
        SetFlag<HardEvent::MTE2_V>(kPartialEvent + slot);
    }

    MlaSplitKvMergeParams params;
    bool roundToEven = false;
    bool writeAccumulators = true;
    uint32_t rowLoop = 0;
    uint32_t partialLoop = 0;
    GlobalTensor<float> partial, sums, maxima, outputLse, accumulator, accumulatorLse;
    GlobalTensor<OutputT> output;
    TBuf<> sumBuf[2], maxBuf[2], weightBuf, rowMaxBuf[2], rowSumBuf[2], lseBuf, partialBuf[2], reduceBuf, castBuf;
};
} // namespace flash_attn::mla::Block
