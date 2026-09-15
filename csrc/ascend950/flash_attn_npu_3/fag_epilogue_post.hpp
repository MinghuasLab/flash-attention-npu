/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_POST_HPP
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_POST_HPP

#include "catlass/arch/resource.hpp"
#include "fag_common.h"

namespace Catlass::Epilogue::Block {

template <typename DataType, class ArchTag, class TilingData>
class FagPost {
public:
    CATLASS_DEVICE void Init(
        Catlass::Arch::Resource<ArchTag> &resource,
        GM_ADDR dq,
        GM_ADDR dk,
        GM_ADDR dv,
        GM_ADDR workspace,
        GM_ADDR tiling)
    {
        tiling_ = reinterpret_cast<const __gm__ TilingData *>(tiling);
        dqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(dq));
        dkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(dk));
        dvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(dv));
        dqWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + tiling_->dqOffset));
        dkWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + tiling_->dkOffset));
        dvWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + tiling_->dvOffset));
        constexpr uint32_t tileElements = 20U * 1024U;
        constexpr uint32_t slotBytes = tileElements * (sizeof(float) + sizeof(DataType));
        for (uint32_t slot = 0; slot < 2U; ++slot) {
            const uint32_t slotOffset = slot * slotBytes;
            srcUb_[slot] = resource.ubBuf.template GetBufferByByte<float>(slotOffset);
            dstUb_[slot] = resource.ubBuf.template GetBufferByByte<DataType>(
                slotOffset + tileElements * sizeof(float));
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
        const uint64_t dqCount = static_cast<uint64_t>(tiling_->totalQ) * tiling_->qHeadNum * tiling_->qkHeadDim;
        const uint64_t dkCount = static_cast<uint64_t>(tiling_->totalKv) * tiling_->kvHeadNum * tiling_->qkHeadDim;
        const uint64_t dvCount = static_cast<uint64_t>(tiling_->totalKv) * tiling_->kvHeadNum * tiling_->vHeadDim;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Ping);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Pong);
        uint32_t pingPongIdx = 0;
        ProcessRegion<false>(dvGm_, dvWorkspace_, dvCount, 1.0F,
            vectorCoreId, vectorCoreNum, pingPongIdx, mte3ToMte2Ping, mte3ToMte2Pong,
            mte2ToVPing, mte2ToVPong, vToMte3Ping, vToMte3Pong);
        ProcessRegion<true>(dkGm_, dkWorkspace_, dkCount, tiling_->scaleValue,
            vectorCoreId, vectorCoreNum, pingPongIdx, mte3ToMte2Ping, mte3ToMte2Pong,
            mte2ToVPing, mte2ToVPong, vToMte3Ping, vToMte3Pong);
        ProcessRegion<true>(dqGm_, dqWorkspace_, dqCount, tiling_->scaleValue,
            vectorCoreId, vectorCoreNum, pingPongIdx, mte3ToMte2Ping, mte3ToMte2Pong,
            mte2ToVPing, mte2ToVPong, vToMte3Ping, vToMte3Pong);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Ping);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Pong);
    }

private:
    template <bool ApplyScale>
    CATLASS_DEVICE void ProcessRegion(
        AscendC::GlobalTensor<DataType> &dst,
        AscendC::GlobalTensor<float> &src,
        uint64_t elementCount,
        float scale,
        uint32_t vectorCoreId,
        uint32_t vectorCoreNum,
        uint32_t &pingPongIdx,
        event_t mte3ToMte2Ping,
        event_t mte3ToMte2Pong,
        event_t mte2ToVPing,
        event_t mte2ToVPong,
        event_t vToMte3Ping,
        event_t vToMte3Pong)
    {
        constexpr uint32_t tileElements = 20U * 1024U;
        if (vectorCoreNum == 0U) return;
        const uint64_t perCore = (elementCount + vectorCoreNum - 1U) / vectorCoreNum;
        const uint64_t rangeBegin = static_cast<uint64_t>(vectorCoreId) * perCore;
        if (rangeBegin >= elementCount) return;
        const uint64_t rangeCount = elementCount - rangeBegin < perCore ? elementCount - rangeBegin : perCore;
        uint64_t done = 0;
        while (done < rangeCount) {
            const uint32_t slot = pingPongIdx;
            const event_t mte3ToMte2Event = slot == 0U ? mte3ToMte2Ping : mte3ToMte2Pong;
            const event_t mte2ToVEvent = slot == 0U ? mte2ToVPing : mte2ToVPong;
            const event_t vToMte3Event = slot == 0U ? vToMte3Ping : vToMte3Pong;
            const uint64_t remaining = rangeCount - done;
            const uint32_t current = static_cast<uint32_t>(
                remaining < tileElements ? remaining : tileElements);
            const uint64_t gmOffset = rangeBegin + done;
            AscendC::DataCopyExtParams inputParams{
                1, static_cast<uint32_t>(current * sizeof(float)),
                0, 0, 0};
            AscendC::DataCopyPadExtParams<float> inputPad{false, 0, 0, 0};
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Event);
            AscendC::DataCopyPad(srcUb_[slot], src[gmOffset], inputParams, inputPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent);
            if constexpr (ApplyScale) {
                AscendC::Muls(srcUb_[slot], srcUb_[slot], scale, current);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Cast(
                dstUb_[slot], srcUb_[slot], AscendC::RoundMode::CAST_RINT, current);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event);
            AscendC::DataCopyExtParams outputParams{
                1, static_cast<uint32_t>(current * sizeof(DataType)),
                0, 0, 0};
            AscendC::DataCopyPad(dst[gmOffset], dstUb_[slot], outputParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Event);
            done += current;
            pingPongIdx = 1U - pingPongIdx;
        }
    }

    const __gm__ TilingData *tiling_ = nullptr;
    AscendC::GlobalTensor<DataType> dqGm_;
    AscendC::GlobalTensor<DataType> dkGm_;
    AscendC::GlobalTensor<DataType> dvGm_;
    AscendC::GlobalTensor<float> dqWorkspace_;
    AscendC::GlobalTensor<float> dkWorkspace_;
    AscendC::GlobalTensor<float> dvWorkspace_;
    AscendC::LocalTensor<float> srcUb_[2];
    AscendC::LocalTensor<DataType> dstUb_[2];
};

}  // namespace Catlass::Epilogue::Block

#endif
