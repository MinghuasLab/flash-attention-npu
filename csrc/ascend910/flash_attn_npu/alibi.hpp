/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */
#ifndef COMMON_ALIBI_BIAS_HPP
#define COMMON_ALIBI_BIAS_HPP

__aicore__ inline void RescaleBiasRow(AscendC::LocalTensor<float> &workUb,
                                      float slope, float preSlope, int32_t count)
{
    AscendC::Muls<float>(workUb, workUb, slope / preSlope, count);  
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void BuildAlibiBiasRow(AscendC::LocalTensor<float> &workUb, 
    int64_t baseColIdx, float slope, int32_t count, 
    AscendC::TEventID eventIdSToV, AscendC::TEventID eventIdVToS)
{
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);

    int n = 8;  // FLOAT_PER_DATABLOCK，只对第 1 个 datablock 中的元素进行标量操作
    for (int i = 0; i < n; i++) {
        workUb.SetValue(i, static_cast<float>(baseColIdx + i) * slope);
    }

    AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);  
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
    
    while (n < count) {
        if (2 * n < count) {
            AscendC::Adds(workUb[n], workUb, static_cast<float>(n) * slope, n);
            AscendC::PipeBarrier<PIPE_V>();
            n *= 2;
        } else {
            AscendC::Adds(workUb[n], workUb, static_cast<float>(n) * slope, count - n);
            AscendC::PipeBarrier<PIPE_V>();
            break;
        }
    }
    
    if (baseColIdx < 0) {
        AscendC::Abs(workUb, workUb, AscendC::Std::min(-baseColIdx, count));
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void SubBiasToScoreRow(AscendC::LocalTensor<float> &scoreUb, uint32_t rowOff,
                                    AscendC::LocalTensor<float> &workUb, int32_t count)
{
    AscendC::Sub<float>(scoreUb[rowOff], scoreUb[rowOff], workUb, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void ApplyAlibi(
    AscendC::LocalTensor<float> &scoreUb, uint32_t scoreOffset,
    uint32_t rowStride, uint32_t columnNum,
    int64_t absRowStart, uint32_t rowNumCurLoop,
    uint32_t qSBlockSize, int64_t qSBlockBaseIdx, 
    int64_t qNBlockBaseIdx, int64_t alibiDiffS, 
    AscendC::GlobalTensor<float> &slopesGm, int64_t slopesBatchOffset,
    AscendC::LocalTensor<float> &workUb, int64_t kvSStartIdx)
{
    if (rowNumCurLoop == 0 || columnNum == 0) {
        return;
    }

    AscendC::TEventID eventIdSToV = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V);
    AscendC::TEventID eventIdVToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);

    for (uint32_t ri = 0; ri < rowNumCurLoop; ++ri) {
        int64_t absRow = absRowStart + static_cast<int64_t>(ri);
        int64_t qNIdx = qNBlockBaseIdx + absRow / qSBlockSize;
        float slope = slopesGm.GetValue(slopesBatchOffset + qNIdx);
        int64_t baseColIdx = kvSStartIdx - (alibiDiffS + qSBlockBaseIdx + (absRow % qSBlockSize));
        BuildAlibiBiasRow(workUb, baseColIdx, slope, columnNum, eventIdSToV, eventIdVToS);
        SubBiasToScoreRow(scoreUb, scoreOffset + ri * rowStride, workUb, columnNum);
    }

    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::S_V>(eventIdSToV);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(eventIdVToS);
}

#endif
