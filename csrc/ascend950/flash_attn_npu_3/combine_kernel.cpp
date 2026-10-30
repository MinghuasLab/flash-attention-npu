/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */

// Split-attention combine kernel (Ascend 950 / arch35).
//
//   lse = logsumexp_s(lse_partial)
//   w_s = exp(lse_s - lse)        inf/nan -> 0
//   out = sum_s w_s * O_s         accumulate in fp32
//
// Pure Vector operator: each AIV owns whole (batch/head, row-block) tiles.
// Weight broadcasting follows the proven fd_combine pattern (per-row scalar
// multiply after a V->S sync) rather than BroadCast, which keeps the arch35
// path identical to the FlashDecode merge already exercised in production.

#include "combine_dispatch.hpp"

#include <limits>
#include <type_traits>

#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"

namespace FaCombine950 {

using ArchTag = Catlass::Arch::Ascend950;

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return (value + align - 1U) / align * align;
}

template <class OutDType>
CATLASS_GLOBAL void
FACombine(GM_ADDR outPartial, GM_ADDR lsePartial, GM_ADDR outGm, GM_ADDR lseOutGm, uint32_t isDense, uint32_t writeLse,
          uint32_t numSplits, uint32_t batch, uint32_t seqlen, uint32_t numHeads, uint32_t headDim, uint32_t tileRows,
          int64_t oSplit, int64_t oBatch, int64_t oSeq, int64_t oHead, int64_t lSplit, int64_t lBatch, int64_t lSeq,
          int64_t lHead, int64_t outBatch, int64_t outSeq, int64_t outHead, int64_t lseOutBatch, int64_t lseOutHead)
{
#ifdef __DAV_VEC__
    constexpr uint32_t kOutBytes = sizeof(OutDType);
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
    auto wUb = resource.ubBuf.template GetBufferByByte<float>(off);
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
    // Kahan compensation term.
    auto compUb = resource.ubBuf.template GetBufferByByte<float>(off);
    off += rA * dA * sizeof(float);
    auto oUb = resource.ubBuf.template GetBufferByByte<float>(off);
    off += rA * dA * sizeof(float);
    // Kahan y = t - c staging.
    auto yUb = resource.ubBuf.template GetBufferByByte<float>(off);
    off += rA * dA * sizeof(float);
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
    const uint32_t threads = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();

    for (uint32_t task = AscendC::GetBlockIdx(); task < taskCount; task += threads) {
        const uint32_t hb = task / rowBlocks;
        const uint32_t rb = task - hb * rowBlocks;
        const uint32_t bIdx = (isDense != 0U) ? (hb / numHeads) : 0U;
        const uint32_t hIdx = (isDense != 0U) ? (hb % numHeads) : hb;
        const uint32_t s0 = rb * tileRows;
        const uint32_t remain = seqlen - s0;
        const uint32_t nrows = (remain < tileRows) ? remain : tileRows;

        AscendC::Duplicate(lseUb, std::numeric_limits<float>::lowest(), static_cast<int32_t>(totalCount));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

        const int64_t lseBase =
            static_cast<int64_t>(bIdx) * lBatch + static_cast<int64_t>(hIdx) * lHead + static_cast<int64_t>(s0) * lSeq;
        const uint32_t loadElems = AlignUp(nrows, 8U);
        const bool vectorLoad = (lSeq == 1 && loadElems <= (seqlen - s0));
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
                    lseUb.SetValue(split * rA + row, gPartialLse.GetValue(base + static_cast<int64_t>(row) * lSeq));
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        uint32_t shapeSplit[2] = {nsA, rA};
        AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA, false>(maxUb, lseUb, sharedTmpUb, shapeSplit, true);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(maxUb, maxUb, std::numeric_limits<float>::lowest(), static_cast<int32_t>(rA));
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t split = 0U; split < nsA; ++split) {
            AscendC::Adds(bcastUb[split * rA], maxUb, 0.0f, static_cast<int32_t>(rA));
        }
        AscendC::PipeBarrier<PIPE_V>();
        // e_s = exp(lse_s - max) doubles as the unnormalised weight, so the
        // exp() rounding is shared by numerator and denominator and cancels.
        AscendC::Sub(wUb, lseUb, bcastUb, static_cast<int32_t>(totalCount));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(wUb, wUb, static_cast<int32_t>(totalCount));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, false>(sumUb, wUb, sharedTmpUb, shapeSplit, true);
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
        // Clamp den to 1 so an all-masked row yields zero instead of NaN.
        AscendC::Maxs(denUb, denUb, 1.0f, static_cast<int32_t>(rA));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

        AscendC::Duplicate(accUb, 0.0f, static_cast<int32_t>(rA * dA));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(compUb, 0.0f, static_cast<int32_t>(rA * dA));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(oUb, 0.0f, static_cast<int32_t>(rA * dA));
        AscendC::PipeBarrier<PIPE_V>();

        const int64_t oBase =
            static_cast<int64_t>(bIdx) * oBatch + static_cast<int64_t>(hIdx) * oHead + static_cast<int64_t>(s0) * oSeq;
        const uint32_t oPad = AlignUp(headDim, 8U) - headDim;
        // Use explicit dstStride to preserve row pitch dA for unaligned headDim.
        const uint32_t oDstStride =
            (dA * sizeof(float) - AlignUp(headDim * static_cast<uint32_t>(sizeof(float)), 32U)) / 32U;
        for (uint32_t split = 0U; split < numSplits; ++split) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::DataCopyPad(oUb, gPartialO[oBase + static_cast<int64_t>(split) * oSplit],
                                 AscendC::DataCopyExtParams(
                                     static_cast<uint16_t>(nrows), static_cast<uint32_t>(headDim * sizeof(float)),
                                     static_cast<uint32_t>((oSeq - static_cast<int64_t>(headDim)) * 4), oDstStride, 0U),
                                 AscendC::DataCopyPadExtParams<float>(true, 0U, static_cast<uint8_t>(oPad), 0.0f));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            for (uint32_t row = 0U; row < nrows; ++row) {
                AscendC::Muls(oUb[row * dA], oUb[row * dA], wUb.GetValue(split * rA + row), static_cast<int32_t>(dA));
                AscendC::PipeBarrier<PIPE_V>();
                // Kahan: y = t - c; s = acc + y; c = (s - acc) - y; acc = s.
                AscendC::Sub(yUb[row * dA], oUb[row * dA], compUb[row * dA], static_cast<int32_t>(dA));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(oUb[row * dA], accUb[row * dA], yUb[row * dA], static_cast<int32_t>(dA));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(compUb[row * dA], oUb[row * dA], accUb[row * dA], static_cast<int32_t>(dA));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(compUb[row * dA], compUb[row * dA], yUb[row * dA], static_cast<int32_t>(dA));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(accUb[row * dA], oUb[row * dA], 0.0f, static_cast<int32_t>(dA));
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        // comp stores the negated low-order error, so subtract it before normalising.
        AscendC::Sub(accUb, accUb, compUb, static_cast<int32_t>(rA * dA));
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = 0U; row < rA; ++row) {
            AscendC::Muls(accUb[row * dA], accUb[row * dA], 1.0f / denUb.GetValue(row), static_cast<int32_t>(dA));
        }
        AscendC::PipeBarrier<PIPE_V>();

        const int64_t outBase = static_cast<int64_t>(bIdx) * outBatch + static_cast<int64_t>(hIdx) * outHead +
                                static_cast<int64_t>(s0) * outSeq;
        const uint32_t outSrcStride = (dA * kOutBytes - AlignUp(headDim * kOutBytes, 32U)) / 32U;
        if constexpr (std::is_same<OutDType, float>::value) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::DataCopyPad(
                gOut[outBase], accUb,
                AscendC::DataCopyExtParams(static_cast<uint16_t>(nrows), static_cast<uint32_t>(headDim * sizeof(float)),
                                           outSrcStride,
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
            AscendC::DataCopyPad(gOut[outBase], outUb,
                                 AscendC::DataCopyExtParams(
                                     static_cast<uint16_t>(nrows), static_cast<uint32_t>(headDim * kOutBytes),
                                     outSrcStride,
                                     static_cast<uint32_t>((outSeq - static_cast<int64_t>(headDim)) * kOutBytes), 0U));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        }

        if (writeLse != 0U) {
            const int64_t lseOutBase = static_cast<int64_t>(bIdx) * lseOutBatch +
                                       static_cast<int64_t>(hIdx) * lseOutHead + static_cast<int64_t>(s0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::DataCopyPad(
                gLseOut[lseOutBase], lseOutUb,
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
        a.outPartial, a.lsePartial, a.out, a.lseOut, a.isDense, a.writeLse, a.numSplits, a.batch, a.seqlen, a.numHeads,
        a.headDim, a.tileRows, a.oSplit, a.oBatch, a.oSeq, a.oHead, a.lSplit, a.lBatch, a.lSeq, a.lHead, a.outBatch,
        a.outSeq, a.outHead, a.lseOutBatch, a.lseOutHead);
}

} // namespace FaCombine950

void launch_combine_fp16(const CombineLaunchArgs& args)
{
    FaCombine950::LaunchCombine<half>(args);
}

void launch_combine_bf16(const CombineLaunchArgs& args)
{
    FaCombine950::LaunchCombine<bfloat16_t>(args);
}

void launch_combine_fp32(const CombineLaunchArgs& args)
{
    FaCombine950::LaunchCombine<float>(args);
}
