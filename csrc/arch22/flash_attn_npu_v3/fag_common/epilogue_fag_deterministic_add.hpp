#ifndef FAG_COMMON_EPILOGUE_FAG_DETERMINISTIC_ADD_HPP
#define FAG_COMMON_EPILOGUE_FAG_DETERMINISTIC_ADD_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "../../flash_attn_npu_v2/fag_block.h"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "kernel_operator.h"
#include "../../flash_attn_npu_v2/kernel_common_fag.hpp"

using AscendC::CopyRepeatParams;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::TBuf;
using AscendC::TQue;
using AscendC::TQue;

namespace Catlass::Epilogue::Block {

struct IndexParams {
    int64_t bIdx;
    int64_t n2Idx;
    int64_t s2oIdx;
    int64_t gIdx;
    int64_t s1oIdx;
};

template <
    uint32_t INPUT_LAYOUT_
>
class BlockEpilogue<
    EpilogueAtlasA2FAGDtmAdd<INPUT_LAYOUT_>
>
{
public:
    using DispatchPolicy = EpilogueAtlasA2FAGDtmAdd<INPUT_LAYOUT_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t INPUT_LAYOUT = INPUT_LAYOUT_;

    TBuf<> unifiedBuffer;

    uint32_t coreNum;
    uint32_t cubeCoreNum;
    uint32_t cBlockIdx;
    uint32_t cCubeBlockIdx;
    uint32_t cSubIdx;
    uint32_t vecBlockNum;

    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t value_d;
    int64_t dAlign;
    int64_t value_dAlign;
    int64_t attenMaskDimS2;

    // split info
    int64_t s1Outer;
    uint32_t s1CvInner;
    uint32_t s1CvTail;
    int64_t s2Outer;
    uint32_t s2CvInner;
    uint32_t s2CvTail;

    uint32_t pingpongIdx = 1;

    __gm__ uint8_t *actual_seq_qlen_addr;
    __gm__ uint8_t *actual_seq_kvlen_addr;

    constexpr static uint32_t ENABLE = 1;

    constexpr static int64_t TOTAL_SIZE = 189 * 1024;
    constexpr static uint32_t INPUT_NUMS = 2;
    constexpr static int64_t GM_DOUBLE_BUFFER = 2;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;

    constexpr static int8_t OUTIDX= -1;

    __aicore__ inline void GetSeqQlenKvlenByBidx(int64_t bIdx, int32_t &actualSeqQlen, int32_t &actualSeqKvlen)
    {
        if (unlikely(bIdx == 0)) {
            actualSeqQlen = ((__gm__ int32_t *)actual_seq_qlen_addr)[0];
            actualSeqKvlen = ((__gm__ int32_t *)actual_seq_kvlen_addr)[0];
        } else {
            actualSeqQlen =
                ((__gm__ int32_t *)actual_seq_qlen_addr)[bIdx] - ((__gm__ int32_t *)actual_seq_qlen_addr)[bIdx - 1];
            actualSeqKvlen =
                ((__gm__ int32_t *)actual_seq_kvlen_addr)[bIdx] - ((__gm__ int32_t *)actual_seq_kvlen_addr)[bIdx - 1];
        }
        return;
    }

    CATLASS_DEVICE
    void GetIndex(int64_t baseIdx, IndexParams& idx)
    {
        if constexpr(INPUT_LAYOUT == TND) {
            int32_t actualSeqQlen = 0;
            int32_t actualSeqKvlen = 0;
            int64_t resbaseIdx = baseIdx;
            for (int64_t bIdx = 0; bIdx < b; bIdx++) {
                GetSeqQlenKvlenByBidx(bIdx, actualSeqQlen, actualSeqKvlen);
                s1Outer = (actualSeqQlen + s1CvInner - 1) / s1CvInner;
                s2Outer = (actualSeqKvlen + s2CvInner - 1) / s2CvInner;
                int64_t totalBaseIdx = n2 * g * s1Outer * s2Outer;
                if (resbaseIdx < totalBaseIdx) {
                    idx.bIdx = bIdx;
                    idx.n2Idx = resbaseIdx / (s2Outer * g * s1Outer);
                    int64_t n2DimTail = resbaseIdx % (s2Outer * g * s1Outer);
                    idx.s2oIdx = n2DimTail / (g * s1Outer);
                    int64_t s2oDimTail = n2DimTail % (g * s1Outer);
                    idx.gIdx = s2oDimTail / s1Outer;
                    idx.s1oIdx = n2DimTail % s1Outer;
                    break;
                } else {
                    resbaseIdx -= totalBaseIdx;
                }
            }
        } else {
            idx.bIdx = baseIdx / (n2 * s2Outer * g * s1Outer);
            int64_t bDimTail = baseIdx % (n2 * s2Outer * g * s1Outer);
            idx.n2Idx  = bDimTail / (s2Outer * g * s1Outer);
            int64_t n2DimTail = baseIdx % (s2Outer * g * s1Outer);
            idx.s2oIdx = n2DimTail / (g * s1Outer);
            int64_t s2oDimTail = n2DimTail % (g * s1Outer);
            idx.gIdx = s2oDimTail / s1Outer;
            idx.s1oIdx = s2oDimTail % s1Outer;
        }
    }

    CATLASS_DEVICE
    void CalcDqReduce(DBParams& dbParam,
    GlobalTensor<float> &srcTensor, GlobalTensor<float> &dstTensor, int64_t d, int64_t dAlign, uint32_t vecCalBlockNum)
    {
        pingpongIdx = dbParam.taskId % 2;
        uint32_t s1CalcInner = (s1CvInner + vecCalBlockNum - 1) / vecCalBlockNum;
        int64_t singleCoreDataNum = s1CalcInner * dAlign;

        LocalTensor<float> dqRes = unifiedBuffer.GetWithOffset<float>(singleCoreDataNum, 0);
        LocalTensor<float> inBuf = unifiedBuffer.GetWithOffset<float>(singleCoreDataNum, singleCoreDataNum * sizeof(float));

        for (int8_t groupId = 0; groupId < cubeCoreNum; groupId++) {
            if (dbParam.blockIdArr[groupId] == -1) {
                break;
            }
            if (dbParam.dqGroupId[groupId] < groupId && dbParam.dqGroupId[groupId] != OUTIDX) {
                continue;
            }
            event_t vWaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
            Duplicate<float>(dqRes, 0.0, singleCoreDataNum);
            AscendC::PipeBarrier<PIPE_V>();
            int64_t blockId = -1;
            int64_t maxS1Extend = 0;
            for (int8_t coreId = groupId; coreId < cubeCoreNum; coreId++) {
                if (dbParam.blockIdArr[coreId] == -1) {
                    break;
                }
                if (groupId == dbParam.dqGroupId[coreId]) {
                    blockId = dbParam.blockIdArr[coreId];
                    uint32_t usedCoreNum = (dbParam.s1CvExtendArr[coreId] + s1CalcInner - 1) / s1CalcInner;
                    if (cBlockIdx % vecCalBlockNum >= usedCoreNum) {
                        continue;
                    }
                    uint32_t s1CalcTail = dbParam.s1CvExtendArr[coreId] - s1CalcInner * (usedCoreNum - 1);
                    uint32_t s1CalcExtend = (cBlockIdx % vecCalBlockNum == usedCoreNum - 1) ? s1CalcTail : s1CalcInner;
                    maxS1Extend = maxS1Extend > s1CalcExtend ? maxS1Extend : s1CalcExtend;

                    //copyOut & add
                    uint64_t srcOffset = pingpongIdx * cubeCoreNum * s1CvInner * dAlign + coreId * s1CvInner * dAlign +
                                    cBlockIdx % vecCalBlockNum * s1CalcInner * d;

                    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMTE2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMTE2);
                    DataCopy(inBuf, srcTensor[srcOffset], s1CalcExtend * d);
                    event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
                    Add(dqRes, dqRes, inBuf, s1CalcExtend * d);
                    event_t mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(mte2WaitV);  // 循环间的反向同步
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(mte2WaitV);
                }
            }
            //copyOut
            if (blockId != -1 && maxS1Extend != 0) {
                IndexParams idx;
                GetIndex(blockId, idx);
                uint64_t dstOffset = 0;
                uint32_t copyOutDstStride = 0;

                if constexpr (INPUT_LAYOUT == TND) {
                    if (idx.bIdx > 0) {
                        dstOffset = ((__gm__ int32_t *)actual_seq_qlen_addr)[idx.bIdx - 1] * n2 * g * d;
                    }
                    dstOffset += (((idx.s1oIdx * s1CvInner + cBlockIdx % vecCalBlockNum * s1CalcInner) * n2 + idx.n2Idx) * g + idx.gIdx) * d;
                    copyOutDstStride = n2 * g * d - d;
                } else if constexpr (INPUT_LAYOUT == BSND) {
                    dstOffset = (((idx.bIdx * s1 + idx.s1oIdx * s1CvInner + cBlockIdx % vecCalBlockNum * s1CalcInner) * n2 + idx.n2Idx) * g + idx.gIdx) * d;
                    copyOutDstStride = n2 * g * d - d;
                }

                event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
                AscendC::SetAtomicAdd<float>();
                DataCopyPad(dstTensor[dstOffset], dqRes,
                            {static_cast<uint16_t>(maxS1Extend), static_cast<uint32_t>(d * sizeof(float)), 0,
                            static_cast<uint32_t>(copyOutDstStride * sizeof(float)), 0});
                AscendC::SetAtomicNone();
            }
        }
    }

    CATLASS_DEVICE
    void CalcDkvReduce(DBParams& dbParam,
    GlobalTensor<float> &srcTensor, GlobalTensor<float> &dstTensor, int64_t d, int64_t dAlign, uint32_t vecCalBlockNum)
    {
        pingpongIdx = dbParam.taskId % 2;
        uint32_t s2CalcInner = (s2CvInner + vecCalBlockNum - 1) / vecCalBlockNum;
        int64_t singleCoreDataNum = s2CalcInner * dAlign;

        LocalTensor<float> resBuf = unifiedBuffer.GetWithOffset<float>(singleCoreDataNum, 0);
        LocalTensor<float> inBuf = unifiedBuffer.GetWithOffset<float>(singleCoreDataNum, singleCoreDataNum * sizeof(float));

        // 按gs1方向连续分核，dk、dv需要累加的数据是连续的
        for (int8_t groupId = 0; groupId < cubeCoreNum; groupId++) {
            if (dbParam.blockIdArr[groupId] == -1) {
                break;
            }
            if (dbParam.kvGroupId[groupId] < groupId && dbParam.kvGroupId[groupId] != OUTIDX) {
                continue;
            }
            event_t vWaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
            Duplicate<float>(resBuf, 0.0, singleCoreDataNum);
            AscendC::PipeBarrier<PIPE_V>();
            int64_t blockId = -1;
            int64_t maxS2Extend = 0;
            for (int8_t coreId = groupId; coreId < cubeCoreNum; coreId++) {
                if (dbParam.blockIdArr[coreId] == -1) {
                    break;
                }
                if (groupId == dbParam.kvGroupId[coreId]) {
                    blockId = dbParam.blockIdArr[coreId];
                    uint32_t usedCoreNum = (dbParam.s2CvExtendArr[coreId] + s2CalcInner - 1) / s2CalcInner;
                    if (cBlockIdx % vecCalBlockNum >= usedCoreNum) {
                        continue;
                    }
                    uint32_t s2CalcTail = dbParam.s2CvExtendArr[coreId] - s2CalcInner * (usedCoreNum - 1);
                    uint32_t s2CalcExtend = (cBlockIdx % vecCalBlockNum == usedCoreNum - 1) ? s2CalcTail : s2CalcInner;
                    maxS2Extend = maxS2Extend > s2CalcExtend ? maxS2Extend : s2CalcExtend;

                    uint64_t srcOffset = pingpongIdx * cubeCoreNum * s2CvInner * dAlign + coreId * s2CvInner * dAlign +
                                    cBlockIdx % vecCalBlockNum * s2CalcInner * d;

                    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMTE2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMTE2);
                    DataCopy(inBuf, srcTensor[srcOffset], s2CalcExtend * d);
                    event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
                    Add(resBuf, resBuf, inBuf, s2CalcExtend * d);
                    event_t mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(mte2WaitV);  // 循环间的反向同步
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(mte2WaitV);
                    
                }
            }
            //copyOut
            if (blockId != -1 && maxS2Extend != 0) {
                IndexParams idx;
                GetIndex(blockId, idx);
                uint64_t dstOffset = 0;
                uint32_t copyOutDstStride = 0;

                if constexpr (INPUT_LAYOUT == TND) {
                    if (idx.bIdx > 0) {
                        dstOffset = ((__gm__ int32_t *)actual_seq_kvlen_addr)[idx.bIdx - 1] * n2 * d;
                    }
                    dstOffset += ((idx.s2oIdx * s2CvInner + cBlockIdx % vecCalBlockNum * s2CalcInner) * n2 + idx.n2Idx) * d;
                    copyOutDstStride = n2 * d - d;
                } else if constexpr (INPUT_LAYOUT == BSND) {
                    dstOffset = ((idx.bIdx * s2 + idx.s2oIdx * s2CvInner + cBlockIdx % vecCalBlockNum * s2CalcInner) * n2 + idx.n2Idx) * d;
                    copyOutDstStride = n2 * d - d;
                }

                event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
                AscendC::SetAtomicAdd<float>();
                DataCopyPad(dstTensor[dstOffset], resBuf, 
                    {static_cast<uint16_t>(maxS2Extend), static_cast<uint32_t>(d * sizeof(float)), 0,
                    static_cast<uint32_t>(copyOutDstStride * sizeof(float)), 0});
                AscendC::SetAtomicNone();
            }
        }
    }

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_in, TBuf<>& buf)
    {
        unifiedBuffer = buf;
        cBlockIdx = GetBlockIdx();
        cCubeBlockIdx = cBlockIdx / 2;
        cSubIdx = cBlockIdx % 2;

        __gm__ FAGTilingData *tilingData = reinterpret_cast<__gm__ FAGTilingData *>(tiling_in);

        coreNum = tilingData->coreNum;
        cubeCoreNum = coreNum / 2;
        vecBlockNum = coreNum / 3;

        b = tilingData->batch;
        n2 = tilingData->kvHeadNum;
        g = tilingData->g;
        s1 = tilingData->qSeqlen;
        s2 = tilingData->kvSeqlen;
        d = tilingData->qkHeadDim;
        value_d = tilingData->vHeadDim;
        dAlign = (d + 15) / 16 * 16;
        value_dAlign = (value_d + 15) / 16 * 16;

        // split info
        s1Outer = tilingData->s1Outer;
        s1CvInner = tilingData->s1CvInner;
        s1CvTail = tilingData->s1CvTail;
        s2Outer = tilingData->s2Outer;
        s2CvInner = tilingData->s2CvInner;
        s2CvTail = s2 - (s2Outer - 1) * s2CvInner;

        actual_seq_qlen_addr = actual_seq_qlen;
        actual_seq_kvlen_addr = actual_seq_kvlen;
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void operator()(DBParams& dbParam, GlobalTensor<float>& dqWorkSpaceGm, GlobalTensor<float>& dkWorkSpaceGm, GlobalTensor<float>& dvWorkSpaceGm,
        GlobalTensor<float>& dqDtmWsGm, GlobalTensor<float>& dkDtmWsGm, GlobalTensor<float>& dvDtmWsGm)
    {
        int64_t s1CalcInner = (s1CvInner + vecBlockNum - 1) / vecBlockNum;
        int64_t s2CalcInner = (s2CvInner + vecBlockNum - 1) / vecBlockNum;

        // When workspace is insufficient, fall back to a full-core reduction, and use
        // coreNum` as vecCalBlockNum to ensure global deterministic accumulation and stable numerical precision.
        // workspace不足，计算降为全核reduction，用coreNum作为vecCalBlockNum来确保全局确定性累加和数值精度稳定
        if (unlikely(s1CalcInner * dAlign * sizeof(float) * 2 > TOTAL_SIZE ||
            s2CalcInner * dAlign * sizeof(float) * 2 > TOTAL_SIZE)) {
            CalcDqReduce(dbParam, dqDtmWsGm, dqWorkSpaceGm, d, dAlign, coreNum);
            CalcDkvReduce(dbParam, dkDtmWsGm, dkWorkSpaceGm, d, dAlign, coreNum);
            CalcDkvReduce(dbParam, dvDtmWsGm, dvWorkSpaceGm, value_d, value_dAlign, coreNum);
        } else {
            if (cBlockIdx < vecBlockNum) {
                CalcDqReduce(dbParam, dqDtmWsGm, dqWorkSpaceGm, d, dAlign, vecBlockNum);
            } else if (cBlockIdx < 2 * vecBlockNum) {
                CalcDkvReduce(dbParam, dkDtmWsGm, dkWorkSpaceGm, d, dAlign, vecBlockNum);
            } else if (cBlockIdx < 3 * vecBlockNum) {
                CalcDkvReduce(dbParam, dvDtmWsGm, dvWorkSpaceGm, value_d, value_dAlign, vecBlockNum);
            }
        }
    }
};

}

#endif // FAG_COMMON_EPILOGUE_FAG_DETERMINISTIC_ADD_HPP
