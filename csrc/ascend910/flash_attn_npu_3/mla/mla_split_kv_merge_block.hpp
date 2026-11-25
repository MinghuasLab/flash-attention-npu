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
          bool GroupedOutput = HeadDim == 128, bool NativeFd = false, bool MayHaveEmptyFragments = false,
          typename MergeParams = MlaSplitKvMergeParams>
class MlaSplitKvMerge {
    static_assert(HeadDim == 512 || HeadDim == 128);
    static_assert(!NativeFd || (HeadDim == 512 && !GroupedOutput && !WriteSingleAccumulator));
    static constexpr bool kBoundedSplits = MergeParams::kSplitCapacity <= 192;
    static_assert(!kBoundedSplits || !MayHaveEmptyFragments);
    static constexpr uint32_t kOutputEvent = 6;
    static constexpr uint32_t kLseEvent = 7;
    static constexpr uint32_t kStatisticsEvent = 8;
    static constexpr uint32_t kPartialEvent = 10;
    static constexpr uint32_t kExportEvent = 0;

  public:
    __aicore__ inline void Init(const MergeParams& p, GM_ADDR out, GM_ADDR sum, GM_ADDR maximum, GM_ADDR result,
                                GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse, TPipe* pipe, float softcap = 0.0f)
    {
        InitBuffers(pipe, accumLse != nullptr);
        Bind(p, out, sum, maximum, result, lse, accum, accumLse, softcap);
    }

    __aicore__ inline void Bind(const MergeParams& p, GM_ADDR out, GM_ADDR sum, GM_ADDR maximum, GM_ADDR result,
                                GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse, float softcap = 0.0f)
    {
        params = p;
        // Each split's valid rows form one interval for causal/local masks.
        // Valid group endpoints therefore admit the native FD fast path.
        if constexpr (NativeFd) {
            allRowsValid = true;
        } else {
            allRowsValid = params.rows != 0;
            if constexpr (kBoundedSplits) {
                PrepareValidRows();
            } else {
                for (uint32_t split = 0; split < params.splits && allRowsValid; ++split)
                    allRowsValid = Valid(split, 0) && Valid(split, params.rows - 1);
            }
            roundToEven = softcap != 0.0f;
            writeAccumulators = accumLse != nullptr;
            writeAccumulatorOut = accum != nullptr;
        }
        partial.SetGlobalBuffer((__gm__ float*)out);
        sums.SetGlobalBuffer((__gm__ float*)sum);
        maxima.SetGlobalBuffer((__gm__ float*)maximum);
        output.SetGlobalBuffer((__gm__ OutputT*)result);
        writeLse = lse != nullptr;
        outputLse.SetGlobalBuffer((__gm__ float*)lse);
        accumulator.SetGlobalBuffer((__gm__ float*)accum);
        accumulatorLse.SetGlobalBuffer((__gm__ float*)accumLse);
    }

    __aicore__ inline void InitBuffers(TPipe* pipe, bool exportPublicPartials = true)
    {
        exportBuffersEnabled = !NativeFd && exportPublicPartials;
        // <=192 split-row pairs use the source 6 KiB bound. 256 explicit
        // splits use one row and 8 KiB. No uint8 vector repeat may be 256.
        constexpr uint32_t statisticsBytes = NativeFd || kBoundedSplits ? 6144 : 8192;
        pipe->InitBuffer(sumBuf[0], statisticsBytes);
        pipe->InitBuffer(sumBuf[1], statisticsBytes);
        pipe->InitBuffer(maxBuf[0], statisticsBytes);
        pipe->InitBuffer(maxBuf[1], statisticsBytes);
        pipe->InitBuffer(weightBuf, statisticsBytes);
        pipe->InitBuffer(partialBuf[0], 16384);
        pipe->InitBuffer(partialBuf[1], 16384);
        pipe->InitBuffer(reduceBuf, 16384);
        pipe->InitBuffer(castBuf, NativeFd || kBoundedSplits ? 16384 : 8192);
        pipe->InitBuffer(rowMaxBuf[0], 256);
        pipe->InitBuffer(rowSumBuf[0], 256);
        pipe->InitBuffer(rowMaxBuf[1], 256);
        pipe->InitBuffer(rowSumBuf[1], 256);
        pipe->InitBuffer(lseBuf, 256);
        // The public partial outputs use their own ping-pong payloads. Native
        // FD may overwrite an input O slot immediately after ReduceFinalRes.
        // These buffers keep that lifetime independent of the public DMA.
        if (!NativeFd && exportBuffersEnabled) {
            pipe->InitBuffer(exportOutBuf[0], 16384);
            pipe->InitBuffer(exportOutBuf[1], 16384);
            pipe->InitBuffer(exportLseBuf[0], 256);
            pipe->InitBuffer(exportLseBuf[1], 256);
        }
        // Automatic FD uses 97536 bytes, or 130816 with public export buffers.
        // Grouped FD also fits within the attention stage's 170 KiB.
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
        if (!NativeFd && exportBuffersEnabled) {
            SetFlag<HardEvent::MTE3_V>(kExportEvent);
            SetFlag<HardEvent::MTE3_V>(kExportEvent + 1);
        }
    }

    __aicore__ inline void Finish()
    {
        WaitFlag<HardEvent::V_MTE2>(kStatisticsEvent);
        WaitFlag<HardEvent::V_MTE2>(kStatisticsEvent + 1);
        WaitFlag<HardEvent::V_MTE2>(kPartialEvent);
        WaitFlag<HardEvent::V_MTE2>(kPartialEvent + 1);
        WaitFlag<HardEvent::MTE3_V>(kOutputEvent);
        WaitFlag<HardEvent::MTE3_V>(kLseEvent);
        if (!NativeFd && exportBuffersEnabled) {
            WaitFlag<HardEvent::MTE3_V>(kExportEvent);
            WaitFlag<HardEvent::MTE3_V>(kExportEvent + 1);
        }
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
        const uint32_t width = rows * 8;
        WaitFlag<HardEvent::V_MTE2>(kStatisticsEvent + statisticsSlot);
        for (uint32_t split = 0; split < params.splits; ++split) {
            // The grouped producer skips empty fragments. The validity mask
            // below supplies their statistics without reading workspace.
            if (MayHaveEmptyFragments && params.begin[split] >= params.end[split])
                continue;
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
        const uint32_t fullRows = (1U << rows) - 1;
        uint32_t validRows = NativeFd || allRowsValid ? fullRows : 0;
        uint8_t validBySplit[MergeParams::kSplitCapacity];
        if (!NativeFd && !allRowsValid) {
            for (uint32_t split = 0; split < params.splits; ++split) {
                const uint32_t valid = ValidRowMask(split, start, rows, fullRows);
                validBySplit[split] = valid;
                validRows |= valid;
                if (valid == fullRows)
                    continue;
                for (uint32_t row = 0; row < rows; ++row) {
                    if (!(valid & (1U << row))) {
                        // Never infer row validity from the source's finite
                        // mask sentinel: a real logit may be that small.
                        Duplicate(sum[split * width + row * 8], 0.0f, 8);
                        Duplicate(maximum[split * width + row * 8], -3.402823466e38f, 8);
                    }
                }
            }
        }
        Duplicate(rowMax, -3.402823466e38f, width);
        Duplicate(rowSum, 0.0f, width);
        PipeBarrier<PIPE_V>();
        if constexpr (NativeFd || kBoundedSplits) {
            Tile::ColMax(rowMax, maximum, rowMax, params.splits, width, width);
            PipeBarrier<PIPE_V>();
        } else {
            for (uint32_t s = 0; s < params.splits; s += 255) {
                Tile::ColMax(rowMax, maximum[s * width], rowMax, Min(255U, params.splits - s), width, width);
                PipeBarrier<PIPE_V>();
            }
        }
        if constexpr (NativeFd || kBoundedSplits) {
            Tile::RowSub(weights, maximum, rowMax, params.splits, width, width);
        } else {
            for (uint32_t s = 0; s < params.splits; s += 255) {
                Tile::RowSub(weights[s * width], maximum[s * width], rowMax, Min(255U, params.splits - s), width,
                             width);
            }
        }
        PipeBarrier<PIPE_V>();
        Exp(weights, weights, params.splits * width);
        PipeBarrier<PIPE_V>();
        Mul(weights, sum, weights, params.splits * width);
        PipeBarrier<PIPE_V>();
        if constexpr (NativeFd || kBoundedSplits) {
            Tile::ColAdd(rowSum, weights, rowSum, params.splits, width, width);
            PipeBarrier<PIPE_V>();
        } else {
            for (uint32_t s = 0; s < params.splits; s += 255) {
                Tile::ColAdd(rowSum, weights[s * width], rowSum, Min(255U, params.splits - s), width, width);
                PipeBarrier<PIPE_V>();
            }
        }
        const bool emptyRows = !NativeFd && validRows != fullRows;
        if (emptyRows) {
            for (uint32_t row = 0; row < rows; ++row)
                if (!(validRows & (1U << row)))
                    Duplicate(rowSum[row * 8], 1.0f, 8);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (NativeFd || kBoundedSplits) {
            Tile::MatDivsVec(weights, weights, rowSum, params.splits, width, width);
        } else {
            for (uint32_t s = 0; s < params.splits; s += 255) {
                Tile::MatDivsVec(weights[s * width], weights[s * width], rowSum, Min(255U, params.splits - s), width,
                                 width);
            }
        }
        PipeBarrier<PIPE_V>();
        if (writeLse) {
            // Match FIA's optional final LSE after normalizing the FD weights.
            auto lse = lseBuf.Get<float>();
            WaitFlag<HardEvent::MTE3_V>(kLseEvent);
            Tile::ComputeSoftMaxLse(lse, rowSum, rowMax, rows);
            if (emptyRows)
                for (uint32_t row = 0; row < rows; ++row)
                    if (!(validRows & (1U << row)))
                        Duplicate(lse[row * 8], -float(1.0 / 0.0), 8);
            SetFlag<HardEvent::V_MTE3>(kLseEvent);
            WaitFlag<HardEvent::V_MTE3>(kLseEvent);
            for (uint32_t row = 0; row < rows;) {
                const uint64_t gap = (uint64_t(params.sq) - 1) * sizeof(float);
                const uint32_t count =
                    gap > UINT32_MAX ? 1
                                     : Min(rows - row, params.heads - (params.rowStart + start + row) % params.heads);
                DataCopyPad(outputLse[RowAddress(start + row)], lse[row * 8],
                            DataCopyExtParams{uint16_t(count), 4, 0, count == 1 ? 0 : uint32_t(gap), 0});
                row += count;
            }
            SetFlag<HardEvent::MTE3_V>(kLseEvent);
        }

        const bool storePartials = !NativeFd && writeAccumulators && (params.splits > 1 || WriteSingleAccumulator);
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
            const uint32_t invalid = !NativeFd && !allRowsValid ? fullRows & ~validBySplit[split] : 0;
            if (invalid) {
                for (uint32_t row = 0; row < rows; ++row)
                    if (invalid & (1U << row))
                        Duplicate(current[row * HeadDim], 0.0f, HeadDim);
                PipeBarrier<PIPE_V>();
            }
            if (storePartials) {
                auto partialLse = sum[split * width];
                if (invalid) {
                    for (uint32_t row = 0; row < rows; ++row) {
                        if (invalid & (1U << row)) {
                            Duplicate(partialLse[row * 8], -float(1.0 / 0.0), 8);
                        }
                    }
                    PipeBarrier<PIPE_V>();
                }
                ExportPartial(current, partialLse, split, start, rows, slot);
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
        // The raw sums have been consumed into independent export slots.
        // Their public DMAs may still be pending; no DMA reads this slot.
        if (storePartials)
            SetFlag<HardEvent::V_MTE2>(kStatisticsEvent + statisticsSlot);
        PipeBarrier<PIPE_V>();
        auto cast = castBuf.Get<OutputT>();
        WaitFlag<HardEvent::MTE3_V>(kOutputEvent);
        if constexpr (std::is_same<OutputT, bfloat16_t>::value) {
            Cast(cast, reduced, RoundMode::CAST_RINT, rows * HeadDim);
        } else {
            if constexpr (NativeFd) {
                Cast(cast, reduced, RoundMode::CAST_ROUND, rows * HeadDim);
            } else if (roundToEven) {
                Cast(cast, reduced, RoundMode::CAST_RINT, rows * HeadDim);
            } else {
                Cast(cast, reduced, RoundMode::CAST_ROUND, rows * HeadDim);
            }
        }
        SetFlag<HardEvent::V_MTE3>(kOutputEvent);
        WaitFlag<HardEvent::V_MTE3>(kOutputEvent);
        CopyFinalOut(cast, start, rows);
        SetFlag<HardEvent::MTE3_V>(kOutputEvent);
        ++rowLoop;
    }

  private:
    __aicore__ inline uint32_t ClampRow(int64_t row) const
    {
        return Max(int64_t(0), Min(row, int64_t(params.rows)));
    }

    __aicore__ inline void PrepareValidRows()
    {
        // The checked automatic schedule contains only nonempty fragments.
        if (!params.maskKind)
            return;
        const int64_t queryOffset = int64_t(params.queryLength ? params.queryLength : params.sq) - params.kvLength;
        for (uint32_t split = 0; split < params.splits; ++split) {
            uint32_t first = 0, end = params.rows;
            if (params.begin[split] >= params.end[split]) {
                end = 0;
            } else {
                // A query row sees this fragment iff its window intersects the KV interval.
                const int64_t right = params.maskKind == 2 ? params.right : 0;
                first = ClampRow((params.begin[split] + queryOffset - right) * params.heads - params.rowStart);
                if (params.maskKind == 2)
                    end = ClampRow((params.end[split] + queryOffset + params.left) * params.heads - params.rowStart);
            }
            params.begin[split] = first;
            params.end[split] = end;
            allRowsValid = allRowsValid && !first && end == params.rows;
        }
    }

    __aicore__ inline uint32_t ValidRowMask(uint32_t split, uint32_t start, uint32_t rows, uint32_t fullRows) const
    {
        if constexpr (kBoundedSplits) {
            const uint32_t first = Max(start, uint32_t(params.begin[split]));
            const uint32_t end = Min(start + rows, uint32_t(params.end[split]));
            return first < end ? (1U << (end - start)) - (1U << (first - start)) : 0;
        } else {
            if (Valid(split, start) && Valid(split, start + rows - 1))
                return fullRows;
            uint32_t mask = 0;
            for (uint32_t row = 0; row < rows; ++row)
                if (Valid(split, start + row))
                    mask |= 1U << row;
            return mask;
        }
    }

    __aicore__ inline void CopyFinalOut(LocalTensor<OutputT> cast, uint32_t start, uint32_t rows)
    {
        // CopyAttenOutUbToGm<OUT_T, BSNGD/TNGD, S1G>, ROW_PAD=true.
        const uint32_t begin = params.rowStart + start;
        const uint32_t sBegin = begin / params.heads, gBegin = begin % params.heads;
        const uint32_t sEnd = (begin + rows) / params.heads, gEnd = (begin + rows) % params.heads;
        const uint32_t kvHeads = GroupedOutput ? params.kvHeads : 1;
        const uint64_t tokenStart = uint64_t(params.batchIndex) * (IsTnd ? 1 : params.sq);
        const uint64_t queryStride = uint64_t(kvHeads) * params.heads * HeadDim;
        const uint64_t base =
            (tokenStart + sBegin) * queryStride + uint64_t(GroupedOutput ? params.kvHead : 0) * params.heads * HeadDim;
        const uint32_t headSize = sBegin == sEnd ? gEnd - gBegin : params.heads - gBegin;
        uint64_t gmOffset = base + uint64_t(gBegin) * HeadDim, ubOffset = 0;
        DataCopyExtParams copyParams{uint16_t(headSize), HeadDim * sizeof(OutputT), 0, 0, 0};
        DataCopyPad(output[gmOffset], cast[ubOffset], copyParams);
        if (sEnd - sBegin >= 1) {
            gmOffset = base + queryStride;
            ubOffset = uint64_t(headSize) * HeadDim;
            for (uint32_t s = sBegin + 1; s < sEnd; ++s) {
                copyParams.blockCount = params.heads;
                DataCopyPad(output[gmOffset], cast[ubOffset], copyParams);
                gmOffset += queryStride;
                ubOffset += uint64_t(params.heads) * HeadDim;
            }
            if (gEnd > 0) {
                copyParams.blockCount = gEnd;
                DataCopyPad(output[gmOffset], cast[ubOffset], copyParams);
            }
        }
    }

    __aicore__ inline void ExportPartial(LocalTensor<float> current, LocalTensor<float> partialLse, uint32_t split,
                                         uint32_t start, uint32_t rows, uint32_t slot)
    {
        auto exported = exportOutBuf[slot].Get<float>();
        auto exportedLse = exportLseBuf[slot].Get<float>();
        WaitFlag<HardEvent::MTE3_V>(kExportEvent + slot);
        if (writeAccumulatorOut)
            DataCopy(exported, current, rows * HeadDim);
        DataCopy(exportedLse, partialLse, rows * 8);
        SetFlag<HardEvent::V_MTE3>(kExportEvent + slot);
        WaitFlag<HardEvent::V_MTE3>(kExportEvent + slot);
        for (uint32_t row = 0; row < rows;) {
            const uint64_t gap = (uint64_t(params.sq) - 1) * HeadDim * 4;
            const uint32_t count =
                gap > UINT32_MAX ? 1 : Min(rows - row, params.heads - (params.rowStart + start + row) % params.heads);
            const uint64_t offset =
                uint64_t(split) * params.batch * params.heads * params.sq * (GroupedOutput ? params.kvHeads : 1) +
                RowAddress(start + row);
            if (writeAccumulatorOut)
                DataCopyPad(accumulator[offset * HeadDim], exported[row * HeadDim],
                            DataCopyExtParams{uint16_t(count), HeadDim * 4, 0, count == 1 ? 0 : uint32_t(gap), 0});
            DataCopyPad(
                accumulatorLse[offset], exportedLse[row * 8],
                DataCopyExtParams{uint16_t(count), 4, 0, count == 1 ? 0 : uint32_t((uint64_t(params.sq) - 1) * 4), 0});
            row += count;
        }
        SetFlag<HardEvent::MTE3_V>(kExportEvent + slot);
    }

    __aicore__ inline void LoadPartial(uint32_t split, uint32_t start, uint32_t rows, uint32_t slot)
    {
        WaitFlag<HardEvent::V_MTE2>(kPartialEvent + slot);
        if (!MayHaveEmptyFragments || params.begin[split] < params.end[split]) {
            const uint32_t stride = params.partialOutStride ? params.partialOutStride : params.mBase;
            DataCopyPad(partialBuf[slot].Get<float>(), partial[(uint64_t(split) * stride + start) * HeadDim],
                        DataCopyExtParams{uint16_t(rows), HeadDim * 4, 0, 0, 0},
                        DataCopyPadExtParams<float>{true, 0, 0, 0.0f});
        }
        SetFlag<HardEvent::MTE2_V>(kPartialEvent + slot);
    }

    MergeParams params;
    bool roundToEven = false;
    bool writeAccumulators = true;
    bool writeLse = false;
    bool writeAccumulatorOut = true;
    bool exportBuffersEnabled = true;
    bool allRowsValid = false;
    uint32_t rowLoop = 0;
    uint32_t partialLoop = 0;
    GlobalTensor<float> partial, sums, maxima, outputLse, accumulator, accumulatorLse;
    GlobalTensor<OutputT> output;
    TBuf<> sumBuf[2], maxBuf[2], weightBuf, rowMaxBuf[2], rowSumBuf[2], lseBuf, partialBuf[2], reduceBuf, castBuf;
    TBuf<> exportOutBuf[2], exportLseBuf[2];
};
} // namespace flash_attn::mla::Block
