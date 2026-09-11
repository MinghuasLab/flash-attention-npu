/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_SOFTMAX_GRAD_FRONT_HPP
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_SOFTMAX_GRAD_FRONT_HPP

#include "catlass/arch/resource.hpp"
#include "fag_common.h"

namespace Catlass::Epilogue::Block {

template <typename DataType, class ArchTag, class TilingData>
class FagSoftmaxGradFront {
public:
    CATLASS_DEVICE void Init(
        Catlass::Arch::Resource<ArchTag> &resource,
        GM_ADDR dout,
        GM_ADDR out,
        GM_ADDR workspace,
        GM_ADDR tiling)
    {
        tiling_ = reinterpret_cast<const __gm__ TilingData *>(tiling);
        doutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(dout));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(out));
        deltaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + tiling_->deltaOffset));

        constexpr uint32_t tileRows = 64U;
        constexpr uint32_t maxAlignedD = 256U;
        const uint32_t doutBytes = tileRows * maxAlignedD * sizeof(DataType);
        const uint32_t slotBytes = 2U * doutBytes + tileRows * sizeof(float);
        for (uint32_t slot = 0; slot < 2U; ++slot) {
            const uint32_t slotOffset = slot * slotBytes;
            doutUb_[slot] = resource.ubBuf.template GetBufferByByte<DataType>(slotOffset);
            outUb_[slot] = resource.ubBuf.template GetBufferByByte<DataType>(slotOffset + doutBytes);
            deltaUb_[slot] = resource.ubBuf.template GetBufferByByte<float>(slotOffset + 2U * doutBytes);
        }
    }

    CATLASS_DEVICE void operator()(
        uint32_t vectorCoreId,
        uint32_t vectorCoreNum,
        event_t mte3ToMte2Ping,
        event_t mte3ToMte2Pong,
        event_t mte2ToVPing,
        event_t mte2ToVPong,
        event_t vToMte3Ping,
        event_t vToMte3Pong)
    {
        constexpr uint32_t tileRows = 64U;
        const uint32_t realD = static_cast<uint32_t>(tiling_->vHeadDim);
        const uint32_t alignedD = (realD + 15U) / 16U * 16U;
        const uint64_t totalRows = tiling_->totalQ * tiling_->qHeadNum;
        if (vectorCoreNum == 0U) return;
        const uint64_t perCore = (totalRows + vectorCoreNum - 1U) / vectorCoreNum;
        const uint64_t rangeBegin = static_cast<uint64_t>(vectorCoreId) * perCore;
        if (rangeBegin >= totalRows) return;
        const uint64_t rangeCount = totalRows - rangeBegin < perCore ? totalRows - rangeBegin : perCore;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Ping);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Pong);
        uint64_t done = 0;
        uint32_t pingPongIdx = 0;
        while (done < rangeCount) {
            const uint32_t slot = pingPongIdx;
            const event_t mte3ToMte2Event = slot == 0U ? mte3ToMte2Ping : mte3ToMte2Pong;
            const event_t mte2ToVEvent = slot == 0U ? mte2ToVPing : mte2ToVPong;
            const event_t vToMte3Event = slot == 0U ? vToMte3Ping : vToMte3Pong;
            const uint64_t remaining = rangeCount - done;
            const uint32_t rows = static_cast<uint32_t>(
                remaining < tileRows ? remaining : tileRows);
            const uint64_t rowBegin = rangeBegin + done;
            const uint64_t inputOffset = rowBegin * realD;
            AscendC::DataCopyExtParams copyParams{
                static_cast<uint16_t>(rows),
                static_cast<uint32_t>(realD * sizeof(DataType)),
                0, 0, 0};
            AscendC::DataCopyPadExtParams<DataType> padParams{
                false, 0, 0, 0};
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Event);
            AscendC::DataCopyPad(
                doutUb_[slot], doutGm_[inputOffset], copyParams, padParams);
            AscendC::DataCopyPad(
                outUb_[slot], outGm_[inputOffset], copyParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent);

            SoftmaxGradFrontVF(
                reinterpret_cast<__ubuf__ DataType *>(doutUb_[slot].GetPhyAddr()),
                reinterpret_cast<__ubuf__ DataType *>(outUb_[slot].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(deltaUb_[slot].GetPhyAddr()),
                rows, realD, alignedD);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event);
            AscendC::DataCopyExtParams outputParams{
                1, static_cast<uint32_t>(rows * sizeof(float)),
                0, 0, 0};
            AscendC::DataCopyPad(
                deltaGm_[rowBegin], deltaUb_[slot], outputParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Event);
            done += rows;
            pingPongIdx = 1U - pingPongIdx;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Ping);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Pong);
    }

private:
    __simd_vf__ inline static void SoftmaxGradFrontVF(
        __ubuf__ DataType *dout,
        __ubuf__ DataType *out,
        __ubuf__ float *delta,
        uint32_t rows,
        uint32_t realD,
        uint32_t rowStride)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<DataType> doutPacked0, outPacked0;
        RegTensor<DataType> doutPacked1, outPacked1;
        RegTensor<float> doutEven, doutOdd, outEven, outOdd;
        RegTensor<float> productEven, productOdd, productSum;
        RegTensor<float> reduced0, reduced1, reduced;
        UnalignReg storeState;
        MaskReg packedMask = CreateMask<DataType, MaskPattern::ALL>();
        MaskReg fullFp32Mask = CreateMask<float, MaskPattern::ALL>();
        constexpr static CastTrait b16ToFp32Even = {
            RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
        constexpr static CastTrait b16ToFp32Odd = {
            RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};

        for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
            const uint32_t firstValid = realD < 128U ? realD : 128U;
            uint32_t firstPairs = (firstValid + 1U) / 2U;
            uint32_t firstOdd = firstValid / 2U;
            MaskReg firstPairMask = UpdateMask<float>(firstPairs);
            MaskReg firstOddMask = UpdateMask<float>(firstOdd);

            LoadAlign(doutPacked0, dout + row * rowStride);
            LoadAlign(outPacked0, out + row * rowStride);
            Cast<float, DataType, b16ToFp32Even>(doutEven, doutPacked0, packedMask);
            Cast<float, DataType, b16ToFp32Odd>(doutOdd, doutPacked0, packedMask);
            Cast<float, DataType, b16ToFp32Even>(outEven, outPacked0, packedMask);
            Cast<float, DataType, b16ToFp32Odd>(outOdd, outPacked0, packedMask);
            Mul(productEven, doutEven, outEven, firstPairMask);
            Mul(productOdd, doutOdd, outOdd, firstOddMask);
            Add(productSum, productEven, productOdd, firstPairMask);
            Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(
                reduced0, productSum, firstPairMask);

            Duplicate(reduced1, 0.0F);
            if (realD > 128U) {
                const uint32_t secondValid = realD - 128U;
                uint32_t secondPairs = (secondValid + 1U) / 2U;
                uint32_t secondOdd = secondValid / 2U;
                MaskReg secondPairMask = UpdateMask<float>(secondPairs);
                MaskReg secondOddMask = UpdateMask<float>(secondOdd);
                LoadAlign(doutPacked1, dout + row * rowStride + 128U);
                LoadAlign(outPacked1, out + row * rowStride + 128U);
                Cast<float, DataType, b16ToFp32Even>(doutEven, doutPacked1, packedMask);
                Cast<float, DataType, b16ToFp32Odd>(doutOdd, doutPacked1, packedMask);
                Cast<float, DataType, b16ToFp32Even>(outEven, outPacked1, packedMask);
                Cast<float, DataType, b16ToFp32Odd>(outOdd, outPacked1, packedMask);
                Mul(productEven, doutEven, outEven, secondPairMask);
                Mul(productOdd, doutOdd, outOdd, secondOddMask);
                Add(productSum, productEven, productOdd, secondPairMask);
                Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(
                    reduced1, productSum, secondPairMask);
            }
            Add(reduced, reduced0, reduced1, fullFp32Mask);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(
                delta, reduced, storeState, 1);
        }
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(
            delta, storeState, 0);
    }

    const __gm__ TilingData *tiling_ = nullptr;
    AscendC::GlobalTensor<DataType> doutGm_;
    AscendC::GlobalTensor<DataType> outGm_;
    AscendC::GlobalTensor<float> deltaGm_;
    AscendC::LocalTensor<DataType> doutUb_[2];
    AscendC::LocalTensor<DataType> outUb_[2];
    AscendC::LocalTensor<float> deltaUb_[2];
};

}  // namespace Catlass::Epilogue::Block

#endif
