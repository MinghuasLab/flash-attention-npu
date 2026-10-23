/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */

// Split-attention combine kernel (Ascend 910B/910C).
//
//   lse = logsumexp_s(lse_partial)          s = split index
//   w_s = exp(lse_s - lse)                  inf/nan -> 0
//   out = sum_s w_s * O_s                   accumulate in fp32
//
// The kernel is a pure Vector operator: each AIV owns whole
// (batch/head, row-block) tiles, so no cross-core synchronization is needed and
// no GM workspace is allocated.

#include "combine_dispatch.hpp"

#include <limits>
#include <type_traits>

#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"

namespace FaCombine {

using ArchTag = Catlass::Arch::AtlasA2;

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return (value + align - 1U) / align * align;
}

template <class OutDType>
__global__ __aicore__ void FACombine(uint64_t fftsAddr, GM_ADDR outPartial, GM_ADDR lsePartial, GM_ADDR outGm,
                                     GM_ADDR lseOutGm, uint32_t isDense, uint32_t writeLse, uint32_t numSplits,
                                     uint32_t batch, uint32_t seqlen, uint32_t numHeads, uint32_t headDim,
                                     uint32_t tileRows, int64_t oSplit, int64_t oBatch, int64_t oSeq, int64_t oHead,
                                     int64_t lSplit, int64_t lBatch, int64_t lSeq, int64_t lHead, int64_t outBatch,
                                     int64_t outSeq, int64_t outHead, int64_t lseOutBatch, int64_t lseOutHead)
{
    AscendC::SetSyncBaseAddr(fftsAddr);
#ifdef __DAV_C220_VEC__
    constexpr uint32_t kOutBytes = sizeof(OutDType);
    // Keep every row of the output staging buffer 32B aligned.
    constexpr uint32_t kOutAlign = (kOutBytes == 4U) ? 8U : 16U;

    const uint32_t nsA = AlignUp(numSplits, 8U);
    const uint32_t rA = AlignUp(tileRows, 8U);
    const uint32_t dA = AlignUp(headDim, kOutAlign);
    const uint32_t totalCount = nsA * rA;

    Catlass::Arch::Resource<ArchTag> resource;

    uint32_t off = 0U;
    auto lseUb = resource.ubBuf.template GetBufferByByte<float>(off);
    off += totalCount * sizeof(float);
    auto bcastUb = resource.ubBuf.template GetBufferByByte<float>(off);
    off += totalCount * sizeof(float);
    auto redTmpUb = resource.ubBuf.template GetBufferByByte<float>(off);
    off += totalCount * sizeof(float);
    auto maxUb = resource.ubBuf.template GetBufferByByte<float>(off);
    off += rA * sizeof(float);
    auto sumUb = resource.ubBuf.template GetBufferByByte<float>(off);
    off += rA * sizeof(float);
    auto lseOutUb = resource.ubBuf.template GetBufferByByte<float>(off);
    off += rA * sizeof(float);
    // Denominator of the normalised weights: sum_s exp(lse_s - max).
    auto denUb = resource.ubBuf.template GetBufferByByte<float>(off);
    off += rA * sizeof(float);
    auto accUb = resource.ubBuf.template GetBufferByByte<float>(off);
    off += rA * dA * sizeof(float);
    // Compensation term of the Kahan summation over splits. Compensated
    // summation keeps the fp32 pre-cast sum within ~1 ULP regardless of the
    // number of splits, which matters because the fp16 output rounds away
    // sub-ULP differences and only the residual fp32 error decides whether an
    // element lands on the other side of a rounding boundary.
    auto compUb = resource.ubBuf.template GetBufferByByte<float>(off);
    off += rA * dA * sizeof(float);
    auto oUb = resource.ubBuf.template GetBufferByByte<float>(off);
    off += rA * dA * sizeof(float);
    auto wTileUb = resource.ubBuf.template GetBufferByByte<float>(off);
    off += rA * dA * sizeof(float);
    // Reduce / BroadCast take a uint8_t shared temporary workspace.
    auto sharedTmpUb = resource.ubBuf.template GetBufferByByte<uint8_t>(off);
    off += rA * dA * sizeof(float);
    auto outUb = resource.ubBuf.template GetBufferByByte<OutDType>(off);
    off += rA * dA * kOutBytes;

    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));

    AscendC::GlobalTensor<float> gPartialO;
    AscendC::GlobalTensor<float> gPartialLse;
    AscendC::GlobalTensor<OutDType> gOut;
    AscendC::GlobalTensor<float> gLseOut;
    gPartialO.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outPartial));
    gPartialLse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lsePartial));
    gOut.SetGlobalBuffer(reinterpret_cast<__gm__ OutDType*>(outGm));
    if (writeLse != 0U) {
        gLseOut.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lseOutGm));
    }

    const uint32_t rowBlocks = (seqlen + tileRows - 1U) / tileRows;
    const uint32_t headBatch = (isDense != 0U) ? (batch * numHeads) : numHeads;
    const uint32_t taskCount = headBatch * rowBlocks;
    // On the VEC side GetBlockIdx() enumerates the launched AIVs, so
    // GetBlockNum() * GetSubBlockNum() is the flat AIV count.
    const uint32_t threads = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();

    for (uint32_t task = AscendC::GetBlockIdx(); task < taskCount; task += threads) {
        const uint32_t hb = task / rowBlocks;
        const uint32_t rb = task - hb * rowBlocks;
        const uint32_t bIdx = (isDense != 0U) ? (hb / numHeads) : 0U;
        const uint32_t hIdx = (isDense != 0U) ? (hb % numHeads) : hb;
        const uint32_t s0 = rb * tileRows;
        const uint32_t remain = seqlen - s0;
        const uint32_t nrows = (remain < tileRows) ? remain : tileRows;

        // ---- LSE tile: (numSplits, tileRows), padded rows = -inf ----
        AscendC::Duplicate(lseUb, std::numeric_limits<float>::lowest(), static_cast<int32_t>(totalCount));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

        const int64_t lseBase = static_cast<int64_t>(bIdx) * lBatch + static_cast<int64_t>(hIdx) * lHead +
                                static_cast<int64_t>(s0) * lSeq;
        // MTE2 requires a blockLen of at least 32B, and keeping it 32B-aligned
        // avoids the ragged-tail fast path. Round the transfer up to 8 floats
        // when the row allows it; otherwise fall back to scalar reads.
        const uint32_t loadElems = AlignUp(nrows, 8U);
        const bool vectorLoad = (loadElems <= (seqlen - s0));
        if (vectorLoad) {
            for (uint32_t split = 0U; split < numSplits; ++split) {
                AscendC::DataCopyPad(
                    lseUb[split * rA], gPartialLse[lseBase + static_cast<int64_t>(split) * lSplit],
                    AscendC::DataCopyExtParams(1U, static_cast<uint32_t>(loadElems * sizeof(float)), 0U, 0U, 0U),
                    AscendC::DataCopyPadExtParams<float>(false, 0U, static_cast<uint8_t>(loadElems - nrows),
                                                        std::numeric_limits<float>::lowest()));
            }
        } else {
            for (uint32_t split = 0U; split < numSplits; ++split) {
                const int64_t base = lseBase + static_cast<int64_t>(split) * lSplit;
                for (uint32_t row = 0U; row < nrows; ++row) {
                    lseUb.SetValue(split * rA + row, gPartialLse.GetValue(base + static_cast<int64_t>(row)));
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        uint32_t shapeSplit[2] = {nsA, rA};
        uint32_t dstShapeSplit[2] = {nsA, rA};
        uint32_t srcShapeRow[2] = {1U, rA};
        AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA, false>(maxUb, lseUb, sharedTmpUb, shapeSplit, true);
        AscendC::PipeBarrier<PIPE_V>();
        // A row whose splits are all masked out has max = -inf; clamp it to the
        // lowest finite value so exp(-inf - max) and ln(sum) stay well defined.
        // For rows with at least one finite split this is a no-op.
        AscendC::Maxs(maxUb, maxUb, std::numeric_limits<float>::lowest(), static_cast<int32_t>(rA));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BroadCast<float, 2, 0>(bcastUb, maxUb, dstShapeSplit, srcShapeRow, sharedTmpUb);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(redTmpUb, lseUb, bcastUb, static_cast<int32_t>(totalCount));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(redTmpUb, redTmpUb, static_cast<int32_t>(totalCount));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, false>(sumUb, redTmpUb, sharedTmpUb, shapeSplit, true);
        AscendC::PipeBarrier<PIPE_V>();
        // Keep sum_s exp(lse_s - max) as the weight denominator; ln() below
        // only feeds the returned LSE.
        AscendC::Adds(denUb, sumUb, 0.0f, static_cast<int32_t>(rA));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Ln(sumUb, sumUb, static_cast<int32_t>(rA));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(lseOutUb, sumUb, maxUb, static_cast<int32_t>(rA));
        AscendC::PipeBarrier<PIPE_V>();
        if (writeLse != 0U) {
            // ln(0) is clamped to the lowest finite float by the vector unit;
            // restore the IEEE -inf that logsumexp produces for fully masked
            // rows so the combined LSE matches the NV contract. Done here so the
            // scalar writes are ordered before the trailing vector work.
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
            for (uint32_t row = 0U; row < rA; ++row) {
                if (lseOutUb.GetValue(row) <= std::numeric_limits<float>::lowest()) {
                    lseOutUb.SetValue(row, -std::numeric_limits<float>::infinity());
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
        }

        // Weights are exp(lse_s - max) / sum_s exp(lse_s - max) rather than
        // exp(lse_s - lse): the exp() rounding is common to numerator and
        // denominator and cancels, and no log/exp round trip is involved, so
        // the fp32 pre-cast error drops by roughly an order of magnitude.
        // A fully masked row has den = 0 and is clamped to 1 so that the
        // all-zero numerator yields +0 instead of NaN. Valid rows have
        // den >= 1 (the max split contributes exp(0) == 1), so the clamp is
        // inert whenever any split is finite.
        AscendC::Maxs(denUb, denUb, 1.0f, static_cast<int32_t>(rA));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Duplicate(accUb, 0.0f, static_cast<int32_t>(rA * dA));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(compUb, 0.0f, static_cast<int32_t>(rA * dA));
        AscendC::PipeBarrier<PIPE_V>();
        // Tail lanes of each row are never stored, but zero them once so the
        // scaled accumulation cannot turn stale data into NaN.
        AscendC::Duplicate(oUb, 0.0f, static_cast<int32_t>(rA * dA));
        AscendC::PipeBarrier<PIPE_V>();

        const int64_t oBase = static_cast<int64_t>(bIdx) * oBatch + static_cast<int64_t>(hIdx) * oHead +
                              static_cast<int64_t>(s0) * oSeq;
        // DataCopyPad padding must stay within one 32B block (<= 7 floats).
        const uint32_t oPad = AlignUp(headDim, 8U) - headDim;
        // Rows in UB are laid out with pitch dA (32B aligned for the output
        // staging), but DataCopyPad packs consecutive blocks on a 32B-aligned
        // blockLen. Pass the gap explicitly, otherwise every row drifts by
        // (dA - AlignUp(headDim, 8)) lanes whenever headDim % 16 != 0.
        const uint32_t oDstStride =
            (dA * sizeof(float) - AlignUp(headDim * static_cast<uint32_t>(sizeof(float)), 32U)) / 32U;
        uint32_t dstShapeO[2] = {rA, dA};
        uint32_t srcShapeO[2] = {rA, 1U};
        for (uint32_t split = 0U; split < numSplits; ++split) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::DataCopyPad(
                oUb, gPartialO[oBase + static_cast<int64_t>(split) * oSplit],
                AscendC::DataCopyExtParams(static_cast<uint16_t>(nrows),
                                           static_cast<uint32_t>(headDim * sizeof(float)),
                                           static_cast<uint32_t>((oSeq - static_cast<int64_t>(headDim)) * 4),
                                           oDstStride, 0U),
                AscendC::DataCopyPadExtParams<float>(true, 0U, static_cast<uint8_t>(oPad), 0.0f));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

            AscendC::BroadCast<float, 2, 1>(wTileUb, redTmpUb[split * rA], dstShapeO, srcShapeO, sharedTmpUb);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(oUb, oUb, wTileUb, static_cast<int32_t>(rA * dA));
            AscendC::PipeBarrier<PIPE_V>();
            // Kahan: y = t - c; s = acc + y; c = (s - acc) - y; acc = s.
            // oUb holds t and is recycled for s, wTileUb holds y.
            AscendC::Sub(wTileUb, oUb, compUb, static_cast<int32_t>(rA * dA));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(oUb, accUb, wTileUb, static_cast<int32_t>(rA * dA));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(compUb, oUb, accUb, static_cast<int32_t>(rA * dA));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(compUb, compUb, wTileUb, static_cast<int32_t>(rA * dA));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(accUb, oUb, 0.0f, static_cast<int32_t>(rA * dA));
            AscendC::PipeBarrier<PIPE_V>();
        }
        // Fold the compensation into the sum before normalising.
        AscendC::Add(accUb, accUb, compUb, static_cast<int32_t>(rA * dA));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BroadCast<float, 2, 1>(wTileUb, denUb, dstShapeO, srcShapeO, sharedTmpUb);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(accUb, accUb, wTileUb, static_cast<int32_t>(rA * dA));
        AscendC::PipeBarrier<PIPE_V>();

        const int64_t outBase = static_cast<int64_t>(bIdx) * outBatch + static_cast<int64_t>(hIdx) * outHead +
                                static_cast<int64_t>(s0) * outSeq;
        // UB-side gap between output rows, in 32B units.
        const uint32_t outSrcStride =
            (dA * kOutBytes - AlignUp(headDim * kOutBytes, 32U)) / 32U;
        if constexpr (std::is_same<OutDType, float>::value) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::DataCopyPad(
                gOut[outBase], accUb,
                AscendC::DataCopyExtParams(static_cast<uint16_t>(nrows),
                                           static_cast<uint32_t>(headDim * sizeof(float)), outSrcStride,
                                           static_cast<uint32_t>((outSeq - static_cast<int64_t>(headDim)) * 4), 0U));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        } else {
            if (std::is_same<OutDType, bfloat16_t>::value) {
                AscendC::Cast(outUb, accUb, AscendC::RoundMode::CAST_RINT, static_cast<int32_t>(rA * dA));
            } else {
                AscendC::Cast(outUb, accUb, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(rA * dA));
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::DataCopyPad(
                gOut[outBase], outUb,
                AscendC::DataCopyExtParams(static_cast<uint16_t>(nrows),
                                           static_cast<uint32_t>(headDim * kOutBytes), outSrcStride,
                                           static_cast<uint32_t>((outSeq - static_cast<int64_t>(headDim)) * kOutBytes),
                                           0U));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        }

        if (writeLse != 0U) {
            const int64_t lseOutBase = static_cast<int64_t>(bIdx) * lseOutBatch +
                                       static_cast<int64_t>(hIdx) * lseOutHead + static_cast<int64_t>(s0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::DataCopyPad(gLseOut[lseOutBase], lseOutUb,
                                 AscendC::DataCopyExtParams(1U, static_cast<uint32_t>(nrows * sizeof(float)), 0U, 0U, 0U));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

template <class OutDType>
inline void LaunchCombine(const CombineLaunchArgs& a)
{
    FACombine<OutDType><<<a.launchBlockDim, nullptr, a.aclStream>>>(
        a.fftsAddr, a.outPartial, a.lsePartial, a.out, a.lseOut, a.isDense, a.writeLse, a.numSplits, a.batch,
        a.seqlen, a.numHeads, a.headDim, a.tileRows, a.oSplit, a.oBatch, a.oSeq, a.oHead, a.lSplit, a.lBatch,
        a.lSeq, a.lHead, a.outBatch, a.outSeq, a.outHead, a.lseOutBatch, a.lseOutHead);
}

} // namespace FaCombine

void launch_combine_fp16(const CombineLaunchArgs& args) { FaCombine::LaunchCombine<half>(args); }

void launch_combine_bf16(const CombineLaunchArgs& args) { FaCombine::LaunchCombine<bfloat16_t>(args); }

void launch_combine_fp32(const CombineLaunchArgs& args) { FaCombine::LaunchCombine<float>(args); }
