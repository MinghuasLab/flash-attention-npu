/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0.
// attention/incre_flash_attention/op_kernel/arch22/.
// Only INT8 content / BF16 RoPE and output, Hkv=1, 512+64, ND PA.
#pragma once
#include "mla_int8_vector_block.hpp"
namespace flash_attn::mla::Int8 {
template <typename IFAT>
class MlaInt8Kernel : public MlaInt8Vector<IFAT> {
    using Base = MlaInt8Vector<IFAT>;
    using T = typename Base::T;
    using Base::actS1Size;
    using Base::aiCoreIdx;
    using Base::ANTIQUANT;
    using Base::BALANCE;
    using Base::bEnd;
    using Base::bIdx;
    using Base::bn2LoopTimes;
    using Base::bStart;
    using Base::CalcParams;
    using Base::ComputeMm1;
    using Base::ComputeMm2;
    using Base::coreStartKVSplitPos;
    using Base::curActSeqLenIsZero;
    using Base::curActualSeqLen;
    using Base::DealActSeqLenIsZero;
    using Base::fdPrefetchLen;
    using Base::fdService;
    using Base::FLASH_DECODE;
    using Base::FlashDecodeCompute;
    using Base::GetActualSeqLen;
    using Base::GetBN2Gid;
    using Base::gIdx;
    using Base::InitBuffers;
    using Base::kvHeadNum;
    using Base::matmulService;
    using Base::mFdBaseSizeMla;
    using Base::n2End;
    using Base::n2Idx;
    using Base::n2Start;
    using Base::pipe;
    using Base::ProcessVec1L;
    using Base::ProcessVec2L;
    using Base::QUANT;
    using Base::s1Idx;
    using Base::s1OuterIdxEnd;
    using Base::s1OuterIdxStart;
    using Base::s1SizeSub;
    using Base::s2End;
    using Base::s2Idx;
    using Base::s2Start;
    using Base::singleProcessSInnerSize;
    using Base::singleProcessSInnerSizeTail;
    using Base::sInnerLoopTimes;
    using Base::SOFTMAX_MIN_NUM;
    using Base::softmaxMaxDefaultUb;
    using Base::softmaxSumDefaultUb;
    using Base::SYNC_C1_V1_FLAG;
    using Base::SYNC_C2_V2_FLAG;
    using Base::SYNC_MODE2;
    using Base::SYNC_V0_C1_FLAG;
    using Base::SYNC_V1_C2_FLAG;
    using Base::tilingData;
    using Base::tmpBlockIdx;
    using Base::UpdateInnerLoopCond;
    using Base::usedCoreNum;

  public:
    __aicore__ inline void Process();

  private:
    __aicore__ inline void ProcessBalance();
    __aicore__ inline void ProcessBXXD();

    __aicore__ inline void PreloadPipeline(uint32_t loop, ExtraInfoMla* extraInfo, TaskContext& task);
    __aicore__ inline void FlashDecode();
};
template <typename IFAT>
__aicore__ inline void MlaInt8Kernel<IFAT>::FlashDecode()
{
    if (tmpBlockIdx < tilingData->tndSplitCoreParams.usedVecNumOfFd) {
        FDparams fdParams;
        fdService.InitBuffers(pipe);
        AscendC::ICachePreLoad(fdPrefetchLen);
#ifdef ASCENDC_CPU_DEBUG
        const uint32_t* bN2IdxOfFdHead = tilingData->tndSplitCoreParams.balanceFDCoreBArr;
        const uint32_t* gS1IdxOfFdHead = tilingData->tndSplitCoreParams.balanceFDCoreS1Arr;
        const uint32_t* s2SplitNumOfFdHead = tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr;
        const uint32_t* gS1IdxEndOfFdHead = tilingData->tndSplitCoreParams.gS1IdxEndOfFdHead;
        const uint32_t* gS1IdxEndOfFdHeadSplit = tilingData->tndSplitCoreParams.gS1IdxEndOfFdHeadSplit;
        const uint32_t* gS1SplitNumOfFdHead = tilingData->tndSplitCoreParams.gS1SplitNumOfFdHeadMla;
        const uint32_t* gS1LastPartSizeOfFdHead = tilingData->tndSplitCoreParams.gS1LastPartSizeOfFdHead;
#else
        uint32_t bN2IdxOfFdHead[MLA_INT8_ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreBArr)];
        uint32_t gS1IdxOfFdHead[MLA_INT8_ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreS1Arr)];
        uint32_t s2SplitNumOfFdHead[MLA_INT8_ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr)];
        uint32_t gS1IdxEndOfFdHead[MLA_INT8_ARRAY_SIZE(tilingData->tndSplitCoreParams.gS1IdxEndOfFdHead)];
        uint32_t gS1IdxEndOfFdHeadSplit[MLA_INT8_ARRAY_SIZE(tilingData->tndSplitCoreParams.gS1IdxEndOfFdHeadSplit)];
        uint32_t gS1SplitNumOfFdHead[MLA_INT8_ARRAY_SIZE(tilingData->tndSplitCoreParams.gS1SplitNumOfFdHeadMla)];
        uint32_t gS1LastPartSizeOfFdHead[MLA_INT8_ARRAY_SIZE(tilingData->tndSplitCoreParams.gS1LastPartSizeOfFdHead)];
        copy_data_align64((uint8_t*)bN2IdxOfFdHead, (uint8_t*)(tilingData->tndSplitCoreParams.balanceFDCoreBArr),
                          sizeof(bN2IdxOfFdHead));
        copy_data_align64((uint8_t*)gS1IdxOfFdHead, (uint8_t*)(tilingData->tndSplitCoreParams.balanceFDCoreS1Arr),
                          sizeof(gS1IdxOfFdHead));
        copy_data_align64((uint8_t*)s2SplitNumOfFdHead,
                          (uint8_t*)(tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr),
                          sizeof(s2SplitNumOfFdHead));
        copy_data_align64((uint8_t*)gS1IdxEndOfFdHead, (uint8_t*)(tilingData->tndSplitCoreParams.gS1IdxEndOfFdHead),
                          sizeof(gS1IdxEndOfFdHead));
        copy_data_align64((uint8_t*)gS1IdxEndOfFdHeadSplit,
                          (uint8_t*)(tilingData->tndSplitCoreParams.gS1IdxEndOfFdHeadSplit),
                          sizeof(gS1IdxEndOfFdHeadSplit));
        copy_data_align64((uint8_t*)gS1SplitNumOfFdHead,
                          (uint8_t*)(tilingData->tndSplitCoreParams.gS1SplitNumOfFdHeadMla),
                          sizeof(gS1SplitNumOfFdHead));
        copy_data_align64((uint8_t*)gS1LastPartSizeOfFdHead,
                          (uint8_t*)(tilingData->tndSplitCoreParams.gS1LastPartSizeOfFdHead),
                          sizeof(gS1LastPartSizeOfFdHead));
#endif
        fdParams = {bN2IdxOfFdHead,          gS1IdxOfFdHead,
                    s2SplitNumOfFdHead,      gS1SplitNumOfFdHead,
                    gS1LastPartSizeOfFdHead, gS1IdxEndOfFdHead,
                    gS1IdxEndOfFdHeadSplit,  tilingData->tndSplitCoreParams.usedVecNumOfFd,
                    mFdBaseSizeMla};

        SyncAll();

        fdService.AllocEventID();
        fdService.InitDecodeParams();
        fdService.FlashDecode(fdParams);
        fdService.FreeEventID();
    } else {
        // superkernel 场景，启动核数大于实际运行核数时，未启动的核仅需要保留 SyncAll
        SyncAll();
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Kernel<IFAT>::Process()
{
    //usedCoreNum: 使用的总核数
    if (aiCoreIdx < usedCoreNum) {
        if ASCEND_IS_AIV {
            Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, BUFFER_SIZE_BYTE_2K / sizeof(T));
            Duplicate(softmaxSumDefaultUb, FLOAT_ZERO, BUFFER_SIZE_BYTE_2K / sizeof(T));
        } else {
            matmulService.AllocEventID();
        }

        if constexpr (BALANCE) {
            ProcessBalance();
        } else {
            ProcessBXXD();
        }

        if ASCEND_IS_AIC {
            matmulService.FreeEventID();
        }
    }

    if constexpr (FLASH_DECODE) {
        // 多核同步
        SyncAll();
        if ASCEND_IS_AIV {
            if (tilingData->tndSplitCoreParams.FdBalanceFlag == 1) {
                FlashDecode();
            } else {
                FlashDecodeCompute();
            }
        }
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Kernel<IFAT>::ProcessBalance()
{
    ExtraInfoMla extraInfo[MLA_INT8_IFA_PRELOAD_TASK_CACHE_SIZE];
    TaskContext taskContext[MLA_INT8_PRE_LOAD_NUM_MLA];

    uint32_t gloop = 0;
    uint32_t tasks = 0;

    int tmpN2LoopEnd, tmpS1IdxLoopEnd, tmpS2LoopEnd;

    bool globalLoopStart = true;

    for (int bIdx = bStart; bIdx <= bEnd; bIdx++) {
        GetActualSeqLen(bIdx);
        UpdateInnerLoopCond();
        if (bIdx == bEnd) {
            tmpN2LoopEnd = n2End;
        } else {
            tmpN2LoopEnd = kvHeadNum - 1;
        }

        for (int n2Idx = n2Start; n2Idx <= tmpN2LoopEnd; n2Idx++) {
            if (curActSeqLenIsZero) {
                DealActSeqLenIsZero(bIdx, n2Idx);
                continue;
            }
            if ((bIdx == bEnd) && (n2Idx == n2End)) {
                tmpS1IdxLoopEnd = s1OuterIdxEnd;
            } else {
                tmpS1IdxLoopEnd = (actS1Size + s1SizeSub - 1) / s1SizeSub; // 切分个数
                tmpS1IdxLoopEnd = tmpS1IdxLoopEnd - 1;                     // 循环上界是<=，这里-1
            }
            for (int sOuterLoopIdx = s1OuterIdxStart; sOuterLoopIdx <= tmpS1IdxLoopEnd; sOuterLoopIdx++) {
                int s2SplitNum =
                    (curActualSeqLen + singleProcessSInnerSize - 1) / singleProcessSInnerSize; // S2切分份数
                bool isEnd = (bIdx == bEnd) && (n2Idx == n2End) && (sOuterLoopIdx == s1OuterIdxEnd);
                if (isEnd) {
                    tmpS2LoopEnd = s2End;
                } else {
                    tmpS2LoopEnd = s2SplitNum - 1; // 循环上界是<=，这里-1
                }
                // 当前s2是否被切，决定了输出是否要写到attenOut上
                bool tndIsS2SplitCore = ((s2Start == 0) && (tmpS2LoopEnd == s2SplitNum - 1)) ? false : true;

                uint32_t extraLoop = 0;
                if (isEnd) {
                    extraLoop = 2;
                }

                for (int s2Idx = s2Start; s2Idx <= (tmpS2LoopEnd + extraLoop); s2Idx++) {
                    TaskContext& ctx = taskContext[tasks++];
                    ctx.bidx = bIdx;
                    ctx.s1idx = sOuterLoopIdx;
                    ctx.s2idx = s2Idx;
                    ctx.s2loops = tmpS2LoopEnd + 1;
                    ctx.gidx = gIdx;
                    ctx.s2SizeTail = singleProcessSInnerSize; // 当前任务要处理的S2长度，复用s2SizeTail
                    if (s2Idx == s2SplitNum - 1) {
                        ctx.s2SizeTail = singleProcessSInnerSizeTail;
                    }
                    ctx.tndIsS2SplitCore = tndIsS2SplitCore;
                    ctx.tndCoreStartKVSplitPos = globalLoopStart ? coreStartKVSplitPos : 0;
                    ctx.isFirstLoop = s2Idx == s2Start;
                    int sOuterLoopTailIdx = ((actS1Size + s1SizeSub - 1) / s1SizeSub) - 1; // 尾块idx=切分份数 - 1
                    if (sOuterLoopIdx == sOuterLoopTailIdx) {                              // 尾块场景
                        ctx.s1Size = (actS1Size % s1SizeSub == 0) ? s1SizeSub : (actS1Size % s1SizeSub);
                    } else {
                        ctx.s1Size = s1SizeSub;
                    }
                    ctx.s2Size = curActualSeqLen;
                    ctx.actS1Size = actS1Size;
                    ctx.isValid = (s2Idx <= tmpS2LoopEnd);
                    bool isLast = isEnd && (s2Idx == s2End);
                    ctx.isLastTask = isLast;

                    // PreloadPipeline loop初始值要求为PRE_LOAD_NUM_MLA
                    PreloadPipeline(gloop + MLA_INT8_PRE_LOAD_NUM_MLA, extraInfo, ctx);

                    gloop += tasks;
                    tasks = 0;
                }
                globalLoopStart = false;
                s2Start = 0;
            }
            s1OuterIdxStart = 0;
        }
        n2Start = 0;
    }
}

template <typename IFAT>
__aicore__ inline void MlaInt8Kernel<IFAT>::ProcessBXXD()
{
    // 0,1 拥有不同的ExtraInfoMla 和TaskContext
    ExtraInfoMla extraInfo[MLA_INT8_IFA_PRELOAD_TASK_CACHE_SIZE];
    TaskContext taskContext[MLA_INT8_PRE_LOAD_NUM_MLA];
    uint32_t gloop = 0; // gloop 记录全局已处理了多少个Task
    uint32_t tasks = 0; // 记录每组Task数量（预加载task的数量）

    if constexpr (!FLASH_DECODE) {
        for (uint32_t bn2gIdx = bn2LoopTimes; bn2gIdx > 0; bn2gIdx--) {
            GetBN2Gid(bn2gIdx - 1);
            GetActualSeqLen(bIdx, s1Idx);
            if (curActualSeqLen == 0) {
                DealActSeqLenIsZero(bIdx, n2Idx);
            }
            if (curActualSeqLen != 0) {
                break;
            }
            bn2LoopTimes--;
        }
    }

    for (uint32_t bn2gIdx = 0; bn2gIdx < bn2LoopTimes; bn2gIdx++) {
        GetBN2Gid(bn2gIdx);
        GetActualSeqLen(bIdx, s1Idx);
        UpdateInnerLoopCond();
        if (curActSeqLenIsZero) {
            DealActSeqLenIsZero(bIdx, n2Idx);
            continue;
        }

        uint32_t extraLoop = 0;
        bool isOuterLoopLast = (bn2gIdx == bn2LoopTimes - 1);
        if (isOuterLoopLast) {
            extraLoop = MLA_INT8_PRE_LOAD_NUM_MLA;
        }

        // 遍历当前S2，当前有Task预加载能力，每PRE_LOAD_NUM_MLA（2）个sInner组成一组Task并发
        for (uint32_t sInnerLoopIdx = 0; sInnerLoopIdx < (sInnerLoopTimes + extraLoop); sInnerLoopIdx++) {
            TaskContext& ctx = taskContext[tasks++];
            ctx.bidx = bIdx;
            ctx.s2idx = sInnerLoopIdx;
            ctx.s2loops = sInnerLoopTimes;
            ctx.gidx = gIdx;
            ctx.s2SizeTail = singleProcessSInnerSizeTail;
            ctx.s1Size = actS1Size;
            ctx.s2Size = curActualSeqLen;
            ctx.s1idx = s1Idx;
            ctx.isValid = (sInnerLoopIdx < sInnerLoopTimes);

            bool isLast = isOuterLoopLast && (sInnerLoopIdx == sInnerLoopTimes - 1);
            ctx.isLastTask = isLast;

            {
                // PreloadPipeline loop初始值要求为PRE_LOAD_NUM_MLA
                PreloadPipeline(gloop + MLA_INT8_PRE_LOAD_NUM_MLA, extraInfo, ctx);
            }

            gloop += tasks;
            tasks = 0;
        }
    }
}

template <typename IFAT>
__aicore__ inline void
MlaInt8Kernel<IFAT>::PreloadPipeline(uint32_t loop, ExtraInfoMla extraInfo[MLA_INT8_IFA_PRELOAD_TASK_CACHE_SIZE],
                                     TaskContext& ctx)
{
    ExtraInfoMla& extraInfo2 = extraInfo[loop % MLA_INT8_IFA_PRELOAD_TASK_CACHE_SIZE];       // 本轮任务
    ExtraInfoMla& extraInfo1 = extraInfo[(loop - 1) % MLA_INT8_IFA_PRELOAD_TASK_CACHE_SIZE]; // 上一轮任务
    ExtraInfoMla& extraInfo0 = extraInfo[(loop - 2) % MLA_INT8_IFA_PRELOAD_TASK_CACHE_SIZE]; // 上两轮任务

    CalcParams(loop, extraInfo2, ctx);

    if (extraInfo2.isValid) {
        if ASCEND_IS_AIV {
        }

        if ASCEND_IS_AIC {

            ComputeMm1(extraInfo2);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG);
        }
    }
    if (extraInfo1.isValid) {
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(SYNC_C1_V1_FLAG);
            ProcessVec1L(extraInfo1);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V1_C2_FLAG);
        }
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag(SYNC_V1_C2_FLAG);
            ComputeMm2(extraInfo1);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C2_V2_FLAG);
        }
    }
    if (extraInfo0.isValid) {
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(SYNC_C2_V2_FLAG);
            ProcessVec2L(extraInfo0);
        }
        extraInfo0.isValid = false;
    }
}

} // namespace flash_attn::mla::Int8
