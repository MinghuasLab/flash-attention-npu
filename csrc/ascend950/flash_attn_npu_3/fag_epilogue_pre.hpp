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
    CATLASS_DEVICE void Init(Catlass::Arch::Resource<ArchTag>& resource, GM_ADDR workspace, GM_ADDR tiling)
    {
        tiling_ = reinterpret_cast<const __gm__ TilingData*>(tiling);
        dqWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + tiling_->dqOffset));
        dkWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + tiling_->dkOffset));
        dvWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + tiling_->dvOffset));
        zeroUb_ = resource.ubBuf.template GetBufferByByte<float>(0);
        workspace_ = workspace;
    }

    CATLASS_DEVICE void operator()(uint32_t vectorCoreId, uint32_t vectorCoreNum, event_t vToMte3Event)
    {
        constexpr uint32_t tileElements = 20U * 1024U;
        AscendC::Duplicate(zeroUb_, 0.0F, tileElements);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event);

        // dqPostAbsorb=1 (deterministic): the dq workspace is a single rolling
        // tile (qTile x RoundUp(qkHeadDim,8) floats, see fag_tiling.cpp), NOT
        // the full S1*N1 region.  Clearing the full count here would overrun
        // into the dk/dv/delta regions and race with FagSoftmaxGradFront's
        // delta writes (no barrier between the two epilogue calls).
        const uint64_t dqCount = tiling_->dqPostAbsorb
                                     ? static_cast<uint64_t>(tiling_->qTile) * ((tiling_->qkHeadDim + 7U) / 8U * 8U)
                                     : static_cast<uint64_t>(tiling_->totalQ) * tiling_->qHeadNum * tiling_->qkHeadDim;
        const uint64_t dkCount = static_cast<uint64_t>(tiling_->totalKv) * tiling_->kvHeadNum * tiling_->qkHeadDim;
        const uint64_t dvCount = static_cast<uint64_t>(tiling_->totalKv) * tiling_->kvHeadNum * tiling_->vHeadDim;
        ClearRegion(dqWorkspace_, dqCount, vectorCoreId, vectorCoreNum);
        // BN2S2 with per-core private dk/dv: the private regions' first write
        // of a column overwrites stale data, so only dq is zeroed.
        if (tiling_->detPrivDkv == 0) {
            ClearRegion(dkWorkspace_, dkCount, vectorCoreId, vectorCoreNum);
            ClearRegion(dvWorkspace_, dvCount, vectorCoreId, vectorCoreNum);
        }
        if (vectorCoreId == 0) {
            // GM_ADDR is __gm__ uint8_t*: cast to int64_t to zero the FULL
            // counters, and use atomics so the zeroing is visible at L2 —
            // a scalar store could sit in this core's DCache and never reach
            // the L2 where the v2 AtomicAdd/poll operate.
            AscendC::AtomicExch(reinterpret_cast<__gm__ uint64_t*>(workspace_), (uint64_t)0); // readyCounter
            AscendC::AtomicExch(reinterpret_cast<__gm__ uint64_t*>(workspace_ + sizeof(uint64_t)),
                                (uint64_t)0); // doneCounter
        }
    }

  private:
    CATLASS_DEVICE void ClearRegion(AscendC::GlobalTensor<float>& dst, uint64_t elementCount, uint32_t vectorCoreId,
                                    uint32_t vectorCoreNum)
    {
        constexpr uint32_t tileElements = 20U * 1024U;
        if (vectorCoreNum == 0U)
            return;
        const uint64_t perCore = (elementCount + vectorCoreNum - 1U) / vectorCoreNum;
        const uint64_t rangeBegin = static_cast<uint64_t>(vectorCoreId) * perCore;
        if (rangeBegin >= elementCount)
            return;
        const uint64_t rangeCount = elementCount - rangeBegin < perCore ? elementCount - rangeBegin : perCore;
        uint64_t done = 0;
        while (done < rangeCount) {
            const uint64_t remaining = rangeCount - done;
            const uint32_t current = static_cast<uint32_t>(remaining < tileElements ? remaining : tileElements);
            const uint64_t gmOffset = rangeBegin + done;
            const uint32_t aligned = current / 8U * 8U;
            if (aligned != 0U) {
                AscendC::DataCopy(dst[gmOffset], zeroUb_, aligned);
            }
            const uint32_t tail = current - aligned;
            if (tail != 0U) {
                AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(tail * sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPad(dst[gmOffset + aligned], zeroUb_, copyParams);
            }
            done += current;
        }
    }

    const __gm__ TilingData* tiling_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    AscendC::GlobalTensor<float> dqWorkspace_;
    AscendC::GlobalTensor<float> dkWorkspace_;
    AscendC::GlobalTensor<float> dvWorkspace_;
    AscendC::LocalTensor<float> zeroUb_;
};

} // namespace Catlass::Epilogue::Block

#endif
