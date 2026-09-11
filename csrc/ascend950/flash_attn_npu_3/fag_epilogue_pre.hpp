/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_PRE_HPP
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_PRE_HPP

#include "catlass/arch/resource.hpp"
#include "fag_common.h"

namespace Catlass::Epilogue::Block {

template <class ArchTag, class TilingData>
class FagPre {
public:
    CATLASS_DEVICE void Init(
        Catlass::Arch::Resource<ArchTag> &resource,
        GM_ADDR workspace,
        GM_ADDR tiling)
    {
        tiling_ = reinterpret_cast<const __gm__ TilingData *>(tiling);
        dqWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + tiling_->dqOffset));
        dkWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + tiling_->dkOffset));
        dvWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + tiling_->dvOffset));
        zeroUb_ = resource.ubBuf.template GetBufferByByte<float>(0);
    }

    CATLASS_DEVICE void operator()(
        uint32_t vectorCoreId,
        uint32_t vectorCoreNum,
        event_t vToMte3Event)
    {
        constexpr uint32_t tileElements = 20U * 1024U;
        AscendC::Duplicate(zeroUb_, 0.0F, tileElements);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event);

        const uint64_t dqCount = static_cast<uint64_t>(tiling_->totalQ) * tiling_->qHeadNum * tiling_->qkHeadDim;
        const uint64_t dkCount = static_cast<uint64_t>(tiling_->totalKv) * tiling_->kvHeadNum * tiling_->qkHeadDim;
        const uint64_t dvCount = static_cast<uint64_t>(tiling_->totalKv) * tiling_->kvHeadNum * tiling_->vHeadDim;
        ClearRegion(dqWorkspace_, dqCount, vectorCoreId, vectorCoreNum);
        ClearRegion(dkWorkspace_, dkCount, vectorCoreId, vectorCoreNum);
        ClearRegion(dvWorkspace_, dvCount, vectorCoreId, vectorCoreNum);
    }

private:
    CATLASS_DEVICE void ClearRegion(
        AscendC::GlobalTensor<float> &dst,
        uint64_t elementCount,
        uint32_t vectorCoreId,
        uint32_t vectorCoreNum)
    {
        constexpr uint32_t tileElements = 20U * 1024U;
        if (vectorCoreNum == 0U) return;
        const uint64_t perCore = (elementCount + vectorCoreNum - 1U) / vectorCoreNum;
        const uint64_t rangeBegin = static_cast<uint64_t>(vectorCoreId) * perCore;
        if (rangeBegin >= elementCount) return;
        const uint64_t rangeCount = elementCount - rangeBegin < perCore ? elementCount - rangeBegin : perCore;
        uint64_t done = 0;
        while (done < rangeCount) {
            const uint64_t remaining = rangeCount - done;
            const uint32_t current = static_cast<uint32_t>(
                remaining < tileElements ? remaining : tileElements);
            const uint64_t gmOffset = rangeBegin + done;
            const uint32_t aligned = current / 8U * 8U;
            if (aligned != 0U) {
                AscendC::DataCopy(dst[gmOffset], zeroUb_, aligned);
            }
            const uint32_t tail = current - aligned;
            if (tail != 0U) {
                AscendC::DataCopyExtParams copyParams{
                    1, static_cast<uint32_t>(tail * sizeof(float)),
                    0, 0, 0};
                AscendC::DataCopyPad(dst[gmOffset + aligned], zeroUb_, copyParams);
            }
            done += current;
        }
    }

    const __gm__ TilingData *tiling_ = nullptr;
    AscendC::GlobalTensor<float> dqWorkspace_;
    AscendC::GlobalTensor<float> dkWorkspace_;
    AscendC::GlobalTensor<float> dvWorkspace_;
    AscendC::LocalTensor<float> zeroUb_;
};

}  // namespace Catlass::Epilogue::Block

#endif
