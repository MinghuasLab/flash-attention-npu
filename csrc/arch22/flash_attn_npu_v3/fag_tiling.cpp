#include "fag_tiling.h"
#include "../flash_attn_npu_v2/softmax_tiling.cpp"
#include "../flash_attn_npu_v2/kernel_common_fag.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "catlass/detail/alignment.hpp"

using namespace std;
namespace FAGTiling {

float CalculateMaskRatio(FAGTilingData &fagTilingData)
{
    // calculate ratio of all mask
    float realS1 = 0;
    float realS2 = 0;
    if (fagTilingData.maskType == static_cast<uint32_t>(MaskType::NO_MASK) ||
        fagTilingData.maskType == static_cast<uint32_t>(MaskType::MASK_BAND)) {
        if (fagTilingData.s1Token >= 0 && fagTilingData.s2Token >= 0) {
            realS1 = fagTilingData.s1Token >= fagTilingData.qSeqlen ? static_cast<float>(fagTilingData.qSeqlen) :
                                                                static_cast<float>(fagTilingData.s1Token);
            realS2 = fagTilingData.s2Token >= fagTilingData.kvSeqlen ? static_cast<float>(fagTilingData.kvSeqlen) :
                                                                static_cast<float>(fagTilingData.s2Token);
            return (realS1 + realS2) / static_cast<float>(fagTilingData.qSeqlen + fagTilingData.kvSeqlen);
        } else if (fagTilingData.s1Token < 0 && fagTilingData.s2Token >= 0) {
            realS2 = fagTilingData.s2Token >= fagTilingData.kvSeqlen ? fagTilingData.kvSeqlen : fagTilingData.s2Token;
            return (realS2 + fagTilingData.s1Token) / static_cast<float>(fagTilingData.kvSeqlen);
        } else {
            realS1 = fagTilingData.s1Token >= fagTilingData.qSeqlen ? fagTilingData.qSeqlen : fagTilingData.s1Token;
            return (realS1 + fagTilingData.s2Token) / static_cast<float>(fagTilingData.qSeqlen);
        }
    } else if (fagTilingData.maskType == static_cast<uint32_t>(MaskType::MASK_CAUSUAL)) {
        if (fagTilingData.qSeqlen >= fagTilingData.kvSeqlen) {
            return 1 - static_cast<float>(HALF * fagTilingData.kvSeqlen) / static_cast<float>(fagTilingData.qSeqlen);
        } else {
            return static_cast<float>(HALF * fagTilingData.qSeqlen) / static_cast<float>(fagTilingData.kvSeqlen);
        }
    } else {
        return 1.0f;
    }
}

void AdjustCvInner(FAGTilingData &fagTilingData)
{
    if (fagTilingData.isDeterministic) {
        int64_t vecBlockNum = fagTilingData.coreNum / VEC_SPLIT_NUM;
        int64_t s1CalcInner = (fagTilingData.s1CvInner + vecBlockNum - 1) / vecBlockNum;
        int64_t s2CalcInner = (fagTilingData.s2CvInner + vecBlockNum - 1) / vecBlockNum;
        int64_t dAlign = (fagTilingData.qkHeadDim + INPUT_ALIGN - 1) / INPUT_ALIGN * INPUT_ALIGN;
        if (s1CalcInner * dAlign * sizeof(float) * DB_NUM > TOTAL_SIZE ||
            s2CalcInner * dAlign * sizeof(float) * DB_NUM > TOTAL_SIZE) {
            fagTilingData.s1CvInner = SAMEAB_S1_256;
            fagTilingData.s2CvInner = SAMEAB_S1_256;
        }
        if (fagTilingData.qSeqlen <= fagTilingData.s1CvInner && fagTilingData.kvSeqlen <= fagTilingData.s2CvInner && fagTilingData.g == 1) {
            // s小于基本块大小时，可以走回非确定性模板
            fagTilingData.isDeterministic = false;
        }
        return;
    }
    // calculate best cvInner
    if (fagTilingData.isSparse == false && fagTilingData.kvSeqlen < 4 * SAMEAB_S1_BASE) {
        if (fagTilingData.qkHeadDim == 64 && fagTilingData.qSeqlen >= fagTilingData.s1CvInner && fagTilingData.kvSeqlen >= fagTilingData.s2CvInner) {
            // D=64时，mmbaseN=256, s1/s2足够大，不需要调整基本块
            return;
        }
        // fuzzy for base cvInner
        uint32_t baseS1;
        uint32_t baseS2;
        float threshold = 0.50;
        uint32_t step = 16;
        constexpr uint32_t maxBaseBlockSize = SAMEAB_S1_BASE * SAMEAB_S2_BASE;

        float s2FloatRatio = static_cast<float>(fagTilingData.kvSeqlen) / static_cast<float>(SAMEAB_S2_BASE);
        uint32_t s2Ratio = static_cast<int>(s2FloatRatio);
        if (s2FloatRatio - s2Ratio > threshold) {
            s2Ratio += 1;
        }
        s2Ratio = s2Ratio < 1 ? 1 : s2Ratio;
        baseS2 = (fagTilingData.kvSeqlen + s2Ratio - 1) / s2Ratio;
        baseS2 = (baseS2 + step - 1) / step * step;
        baseS1 = (maxBaseBlockSize / baseS2) / step * step;
        baseS1 = baseS1 > SAMEAB_S1_BASE ? SAMEAB_S1_BASE : baseS1;
        uint32_t s1Ratio = (fagTilingData.qSeqlen + baseS1 - 1) / baseS1;
        if (fagTilingData.qSeqlen % s1Ratio == 0) {
            // adjust s1CvInner smaller for balance
            baseS1 = ((fagTilingData.qSeqlen / s1Ratio) + step - 1) / step * step;
        }

        int64_t alignS1 = (fagTilingData.qSeqlen + step - 1) / step * step;
        if (static_cast<int64_t>(baseS1) > alignS1) {
            baseS1 = alignS1;
            uint32_t baseS2Resize = (maxBaseBlockSize / baseS1) / step * step;
            baseS2 = baseS2Resize > baseS2 ? baseS2Resize : baseS2;
        }
        baseS2 = baseS2 > SAMEAB_S2_BASE ? SAMEAB_S2_BASE : baseS2;

        fagTilingData.s1CvInner = baseS1;
        fagTilingData.s2CvInner = baseS2;
        return;
    }

    uint32_t s1Outer = (fagTilingData.qSeqlen + SAMEAB_S1_BASE - 1) / SAMEAB_S1_BASE;
    uint32_t s2Outer = (fagTilingData.kvSeqlen + SAMEAB_S2_BASE - 1) / SAMEAB_S2_BASE;

    if (fagTilingData.qSeqlen != fagTilingData.kvSeqlen && (CalculateMaskRatio(fagTilingData) - 0.23) <= 1E-6 &&
        (s1Outer * s2Outer) < TOTAL_BLOCK_PIPELINE) { // S不等长且有效基本块个数占全计算比例小于0.23
        fagTilingData.s1CvInner = SAMEAB_S1_256;
        fagTilingData.s2CvInner = SAMEAB_S1_BASE;
        return;
    }
}

bool SetSparseParams(const FAGTilingData &fagTilingData)
{
    if (fagTilingData.layoutType == TND) {
        return true;
    }
    return fagTilingData.maskType != static_cast<uint32_t>(MaskType::NO_MASK);
}

static void ProcessTokensInfo(FAGTilingData &fagTilingData)
{
    if (fagTilingData.layoutType != TND &&
        (fagTilingData.maskType == static_cast<uint32_t>(MaskType::MASK_CAUSUAL) ||
         fagTilingData.maskType == static_cast<uint32_t>(MaskType::MASK_BAND))) {
        fagTilingData.s1Token += fagTilingData.qSeqlen - fagTilingData.kvSeqlen;
        fagTilingData.s2Token += fagTilingData.kvSeqlen - fagTilingData.qSeqlen;
    }
}

void FillWorkSpaceTilingData(FAGTilingData &fagTilingData) {
   // begin position
    size_t workspaceSize = MUL_CORE_SYNC_BUFFER;
    uint32_t s1Inner = std::min(INITIAL_S1_SPLIT_NUM, fagTilingData.s1Align);

    // matmal3 q
    workspaceSize = (workspaceSize + static_cast<size_t>(fagTilingData.qSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 k
    workspaceSize = (workspaceSize + static_cast<size_t>(fagTilingData.kvSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 v
    workspaceSize = (workspaceSize + static_cast<size_t>(fagTilingData.vSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    // mask bool workspace size
    if (fagTilingData.dropoutIsDivisibleBy8 == 0) {
        workspaceSize = (workspaceSize + static_cast<size_t>(fagTilingData.dropMaskSize) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    }
    // sfmg workspace
    workspaceSize = workspaceSize + static_cast<size_t>(AlignTo(
        fagTilingData.sfmgNormalAxisSize * SOFTMAX_REDUCE_SIZE * FP32_BYTES, static_cast<int64_t>(GM_ALIGN)));

    // matmal1/matmal2 workspace size
    size_t vectorCoreNum = fagTilingData.coreNum;
    workspaceSize =
        (workspaceSize + vectorCoreNum * fagTilingData.s1CvInner * fagTilingData.s2CvInner * FP32_BYTES * MATMUL_INPUT_NUM +
         GM_ALIGN) /
        GM_ALIGN * GM_ALIGN;

    workspaceSize += WORKSPACE_BUFFER;

    // 确定性计算
    if (fagTilingData.isDeterministic) {
        int64_t dAlign = (fagTilingData.qkHeadDim + FP16_BLOCK_NUMS - 1) / FP16_BLOCK_NUMS * FP16_BLOCK_NUMS;
        int64_t value_dAlign = (fagTilingData.vHeadDim + FP16_BLOCK_NUMS - 1) / FP16_BLOCK_NUMS * FP16_BLOCK_NUMS;
        int64_t cvS2Inner = fagTilingData.s2CvInner;
        workspaceSize +=
            AlignUp(fagTilingData.s1CvInner * dAlign * FP32_BYTES * fagTilingData.aicNum * DB_NUM, GM_ALIGN);
        workspaceSize += AlignUp(cvS2Inner * dAlign * FP32_BYTES * fagTilingData.aicNum * DB_NUM, GM_ALIGN);
        workspaceSize += AlignUp(cvS2Inner * value_dAlign * FP32_BYTES * fagTilingData.aicNum * DB_NUM, GM_ALIGN);
    }

    fagTilingData.workspaceSize = workspaceSize;
}

bool DoPreTiling(FAGTilingData &fagTilingData)
{
    uint32_t castBufferLen = 60 * 1024;
    uint32_t outputBufferLen = 30 * 1024;
    uint32_t inputBufferLen = 4 * 1024;
    int64_t singleUBProcessNum = castBufferLen / 2;

    int64_t maskSize = AlignTo(fagTilingData.dropMaskSize, static_cast<int64_t>(BOOL_BLOCK_NUMS));
    int64_t singleCoreNum = AlignTo(CeilCommon(maskSize, static_cast<int64_t>(fagTilingData.blockOuter)),
                                    static_cast<int64_t>(BOOL_BLOCK_NUMS));
    int64_t maskUsedCoreNum = static_cast<int64_t>(CeilCommon(maskSize, singleCoreNum));

    int64_t tailCoreNum = maskSize - (maskUsedCoreNum - 1) * singleCoreNum;
    tailCoreNum = AlignTo(tailCoreNum, static_cast<int64_t>(BOOL_BLOCK_NUMS));

    int64_t singleCoreUBLoop = static_cast<int64_t>(CeilCommon(singleCoreNum, singleUBProcessNum));
    int64_t tailCoreUBLoop = static_cast<int64_t>(CeilCommon(tailCoreNum, singleUBProcessNum));

    int64_t singleCoreUBLastLoopNum = static_cast<int64_t>(singleCoreNum - (singleCoreUBLoop - 1) * singleUBProcessNum);
    int64_t tailCoreUBLastLoopNum = static_cast<int64_t>(tailCoreNum - (tailCoreUBLoop - 1) * singleUBProcessNum);

    fagTilingData.maskCoreNum = maskUsedCoreNum;
    fagTilingData.singleUBProcessNum = static_cast<int64_t>(singleUBProcessNum);
    fagTilingData.maskSingleCoreLoop = singleCoreUBLoop;
    fagTilingData.maskLastLoopNum = singleCoreUBLastLoopNum;
    fagTilingData.maskTailCoreLoop = tailCoreUBLoop;
    fagTilingData.maskTailCoreLastLoopNum = tailCoreUBLastLoopNum;

    if (maskUsedCoreNum == 0) {
        cerr << "divisor maskUsedCoreNum is 0." << endl;
        return false;
    }

    int64_t dropBeginAddr = MUL_CORE_SYNC_BUFFER;

    int64_t qSizeReal = fagTilingData.qSize;
    int64_t kvSizeReal = fagTilingData.kvSize;
    int64_t vSizeReal = fagTilingData.vSize;
    dropBeginAddr = (dropBeginAddr + (qSizeReal) * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    dropBeginAddr = (dropBeginAddr + (kvSizeReal) * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    dropBeginAddr = (dropBeginAddr + (vSizeReal) * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    fagTilingData.dropBeginAddr = dropBeginAddr;
    return true;
}

bool DoPostTiling(FAGTilingData &fagTilingData)
{
    int64_t workspaceOffsets = MUL_CORE_SYNC_BUFFER;
    fagTilingData.dqWorkSpaceOffset = workspaceOffsets;

    workspaceOffsets = (workspaceOffsets + fagTilingData.qSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    fagTilingData.dkWorkSpaceOffset = workspaceOffsets;

    workspaceOffsets = (workspaceOffsets + fagTilingData.kvSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    fagTilingData.dvWorkSpaceOffset = workspaceOffsets;

    return true;
}

void DoPreSfmgTiling(FAGTilingData &fagTilingData)
{
    int64_t value_dAlign = (fagTilingData.vHeadDim + FP16_BLOCK_NUMS - 1) / FP16_BLOCK_NUMS * FP16_BLOCK_NUMS;
    uint32_t inputBufferLen = 24 * 1024;                                    // castBuffer 24K*2=48K
    uint32_t castBufferLen = 48 * 1024;                                     // castBuffer 48K*2=96K
    uint32_t outputBufferLen = CeilCommon(castBufferLen, value_dAlign) * 8; // 输出(qSeqlen,8)
    uint32_t tempBufferLen = 40 * 1024 - outputBufferLen;

    int64_t normalAxisSize = 0;
    if (fagTilingData.layoutType == TND) {
        normalAxisSize = fagTilingData.t1 * fagTilingData.kvHeadNum * fagTilingData.g;
    } else {
        normalAxisSize = fagTilingData.batch * fagTilingData.kvHeadNum * fagTilingData.g * fagTilingData.qSeqlen;
    }
    fagTilingData.sfmgNormalAxisSize = normalAxisSize;

    // 计算单loop的计算量及loop次数
    int64_t singleLoopNBurstNum = inputBufferLen / fagTilingData.dataTypeSize / value_dAlign;

    int64_t dropMaskSize = 0;
    if (fagTilingData.dropoutIsDivisibleBy8 == 0) {
        dropMaskSize = (fagTilingData.dropMaskSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN; // 与主kernel的偏移计算保持一致
    }
    int64_t sfmgPreBeginAddr = fagTilingData.dropBeginAddr + dropMaskSize;
    fagTilingData.sfmgPreBeginAddr = sfmgPreBeginAddr;

    // Softmax tiling
    uint32_t cvS2Inner = fagTilingData.s2CvInner;
    uint32_t s2VSize = cvS2Inner > 256 ? 256 : cvS2Inner;
    uint32_t s1VecSize =
        std::min(((INITIAL_S1_SPLIT_NUM * INITIAL_S2_SPLIT_NUM + s2VSize - 1) / s2VSize), fagTilingData.s1Inner);

    std::vector<uint32_t> softmaxShape = {s1VecSize, s2VSize};

    SoftMaxTiling softmaxTilingData;
    SoftMaxTilingFunc(softmaxShape, fagTilingData.calTypeSize, fagTilingData.tmpBufferSize, softmaxTilingData);

    // softmaxGrad tiling
    SoftMaxTiling softmaxGradTilingData;
    std::vector<int64_t> softmaxGradShape = {singleLoopNBurstNum, value_dAlign};
    SoftMaxGradTilingFunc(softmaxGradShape, fagTilingData.calTypeSize, tempBufferLen, softmaxGradTilingData);

    // put SoftMaxData in Tiling
    fagTilingData.softmaxTilingData = softmaxTilingData;
    fagTilingData.softmaxGradTilingData = softmaxGradTilingData;
}

int64_t GetFAGTilingParam(const FAGInfo &fagInfo, uint32_t aicNum, uint32_t aivNum, uint64_t ubSize, FAGTilingData &fagTilingData)
{
    fagTilingData.scaleValue = fagInfo.scaleValue;
    fagTilingData.softcapValue = fagInfo.softcapValue;
    fagTilingData.keepProb = fagInfo.keepProb;
    fagTilingData.batch = fagInfo.batch;
    fagTilingData.qSeqlen = fagInfo.qSeqlen;
    fagTilingData.qHeadNum = fagInfo.qHeadNum;
    fagTilingData.qkHeadDim = fagInfo.qkHeadDim;
    fagTilingData.kvSeqlen = fagInfo.kvSeqlen;
    fagTilingData.kvHeadNum = fagInfo.kvHeadNum;
    fagTilingData.vHeadDim = fagInfo.vHeadDim;
    fagTilingData.isDeterministic = fagInfo.isDeterministic;

    fagTilingData.coreNum = aivNum;
    fagTilingData.aicNum = aicNum;
    fagTilingData.ubSize = ubSize;
    fagTilingData.ubSize -= MATMUL_SIZE;
    if (fagTilingData.ubSize <= 0) {
        cerr << "ubSize is invalid." << endl;
        return -1;
    }

    fagTilingData.g = fagInfo.qHeadNum / fagInfo.kvHeadNum;

    bool isTnd = fagInfo.layout == TND;
    if (isTnd) {
        fagTilingData.layoutType = TND;
        auto actualSeqQlenTensor = fagInfo.qSeqlenList;
        auto actualSeqKvlenTensor = fagInfo.kvSeqlenList;
        if (actualSeqQlenTensor == nullptr || actualSeqKvlenTensor == nullptr) {
            cerr << "actualSeqQlenTensor or actualSeqKvlenTensor is nullptr.\n";
            return -1;
        }
        const int64_t seqQShapeSize = fagInfo.batch;
        const int64_t kvSeqShapeSize = fagInfo.batch;
        if (seqQShapeSize != kvSeqShapeSize) {
            cerr << "actualSeqQlenTensor shapeSize is not equal actualSeqKvlenTensor.\n";
            return -1;
        }
        for (int64_t i = 0; i < seqQShapeSize; ++i) {
            if (i == 0) {
                fagTilingData.actualSeqQlen.push_back(actualSeqQlenTensor[i]);
                fagTilingData.actualSeqKvlen.push_back(actualSeqKvlenTensor[i]);
            } else {
                fagTilingData.actualSeqQlen.push_back(actualSeqQlenTensor[i] - actualSeqQlenTensor[i - 1]);
                fagTilingData.actualSeqKvlen.push_back(actualSeqKvlenTensor[i] - actualSeqKvlenTensor[i - 1]);
            }
            fagTilingData.sumS1S2Product += fagTilingData.actualSeqQlen[i] * fagTilingData.actualSeqKvlen[i];
        }

        uint64_t tailZeroCount = 0;
        for (auto i = seqQShapeSize - 1; i >= 1; --i) {
            if (fagTilingData.actualSeqQlen[i] <= 0 && fagTilingData.actualSeqKvlen[i] <= 0) {
                ++tailZeroCount;
            } else {
                break;
            }
        }
        fagTilingData.batch -= tailZeroCount;
        fagTilingData.t1 = actualSeqQlenTensor[seqQShapeSize - 1];
        fagTilingData.t2 = actualSeqKvlenTensor[kvSeqShapeSize- 1];
        fagTilingData.qSeqlen = *std::max_element(fagTilingData.actualSeqQlen.begin(), fagTilingData.actualSeqQlen.end());
        fagTilingData.kvSeqlen = *std::max_element(fagTilingData.actualSeqKvlen.begin(), fagTilingData.actualSeqKvlen.end());
    } else {
        fagTilingData.layoutType = BSND;
    }

    fagTilingData.s1Align = (fagTilingData.qSeqlen + INPUT_ALIGN - 1) / INPUT_ALIGN * INPUT_ALIGN;
    fagTilingData.s2Align = (fagTilingData.kvSeqlen + INPUT_ALIGN - 1) / INPUT_ALIGN * INPUT_ALIGN;
    fagTilingData.s1Inner = INITIAL_S1_SPLIT_NUM;
    fagTilingData.s2Inner = INITIAL_S2_SPLIT_NUM;

    fagTilingData.s1Token = fagInfo.window_size_left;
    fagTilingData.s2Token = fagInfo.window_size_right;
    fagTilingData.maskType = static_cast<uint32_t>(fagInfo.maskType);

    if (fagInfo.maskType == static_cast<int32_t>(MaskType::NO_MASK)) {
        fagTilingData.attenMaskOptional = EMPTY_TENSOR;
        fagTilingData.s1Token = INT32_MAX;
        fagTilingData.s2Token = INT32_MAX;
    } else if (fagInfo.maskType == static_cast<int32_t>(MaskType::MASK_CAUSUAL)) {
        fagTilingData.attenMaskOptional = NORMAL_TENSOR;
        fagTilingData.attenMaskCompressMode = CAUSAL_COMPRESS_MODE;
        fagTilingData.s1Token = INT32_MAX;
        fagTilingData.s2Token = 0;
    } else if (fagInfo.maskType == static_cast<int32_t>(MaskType::MASK_BAND)) {
        fagTilingData.attenMaskOptional = NORMAL_TENSOR;
        fagTilingData.attenMaskCompressMode = BAND_COMPRESS_MODE;
        fagTilingData.s1Token = (fagInfo.window_size_left < 0) ? INT32_MAX : fagInfo.window_size_left;
        fagTilingData.s2Token = (fagInfo.window_size_right < 0) ? INT32_MAX : fagInfo.window_size_right;
    }

    ProcessTokensInfo(fagTilingData);

    fagTilingData.isSparse = SetSparseParams(fagTilingData);

    fagTilingData.s1CvInner = SAMEAB_S1_BASE;
    fagTilingData.s2CvInner = SAMEAB_S2_BASE;

    AdjustCvInner(fagTilingData);

    fagTilingData.s1Outer = (fagTilingData.qSeqlen + fagTilingData.s1CvInner - 1) / fagTilingData.s1CvInner;
    fagTilingData.s2Outer = (fagTilingData.kvSeqlen + fagTilingData.s2CvInner - 1) / fagTilingData.s2CvInner;

    uint32_t s1TailTmp = fagTilingData.qSeqlen % fagTilingData.s1Inner;
    uint32_t s1CvTailTmp = fagTilingData.qSeqlen % fagTilingData.s1CvInner;
    uint32_t s2TailTmp = fagTilingData.kvSeqlen % fagTilingData.s2Inner;
    uint32_t s2CvTailTmp = fagTilingData.kvSeqlen % fagTilingData.s2CvInner;
    fagTilingData.s1Tail = s1TailTmp == 0 ? fagTilingData.s1Inner : s1TailTmp;
    fagTilingData.s2Tail = s2TailTmp == 0 ? fagTilingData.s2Inner : s2TailTmp;
    fagTilingData.s1CvTail = s1CvTailTmp == 0 ? fagTilingData.s1CvInner : s1CvTailTmp;
    fagTilingData.s2CvTail = s2CvTailTmp == 0 ? fagTilingData.s2CvInner : s2CvTailTmp;
    fagTilingData.sfmgNormalAxisSize = fagTilingData.batch * fagTilingData.kvHeadNum * fagTilingData.g * fagTilingData.qSeqlen;

    fagTilingData.dropoutIsDivisibleBy8 = 1;
    if (fagTilingData.keepProb < 1) {
        if (isTnd) {
            for (int64_t i = 0; i < fagTilingData.batch; i++) {
                if (fagTilingData.actualSeqKvlen[i] % BIT_NUMS != 0) {
                    fagTilingData.dropoutIsDivisibleBy8 = 0;
                    break;
                }
            }
        } else if (fagTilingData.kvSeqlen % BIT_NUMS != 0) {
            fagTilingData.dropoutIsDivisibleBy8 = 0;
        }
    }

    // 目前仅支持fp16/bf16
    fagTilingData.dataTypeSize = FP16_BYTES;
    fagTilingData.dataBlockNum = FP16_BLOCK_NUMS;
    fagTilingData.calTypeSize = FP32_BYTES;
    fagTilingData.calBlockNum = FP32_BLOCK_NUMS;

    fagTilingData.blockOuter = fagTilingData.coreNum;
    int64_t fusedOuter = fagTilingData.batch * fagTilingData.kvHeadNum * fagTilingData.g * fagTilingData.s1Outer * fagTilingData.s2Outer;
    fagTilingData.blockFactor = (fusedOuter + fagTilingData.coreNum - 1) / fagTilingData.coreNum;

    if (fagTilingData.layoutType == TND) {
        fagTilingData.qSize = fagTilingData.t1 * fagTilingData.kvHeadNum * fagTilingData.g * fagTilingData.qkHeadDim;
        fagTilingData.kvSize = fagTilingData.t2 * fagTilingData.kvHeadNum * 1 * fagTilingData.qkHeadDim;
        fagTilingData.vSize = fagTilingData.t2 * fagTilingData.kvHeadNum * 1 * fagTilingData.vHeadDim;
        fagTilingData.dropMaskSize = fagTilingData.kvHeadNum * fagTilingData.g * fagTilingData.sumS1S2Product;
    } else {
        fagTilingData.qSize = fagInfo.batch * fagInfo.qHeadNum * fagInfo.qSeqlen *  fagInfo.qkHeadDim;
        fagTilingData.kvSize = fagInfo.batch * fagInfo.kvHeadNum * fagInfo.kvSeqlen * fagInfo.qkHeadDim;
        fagTilingData.vSize = fagInfo.batch * fagInfo.kvHeadNum * fagInfo.kvSeqlen * fagInfo.vHeadDim;
        fagTilingData.dropMaskSize = fagInfo.batch * fagInfo.qHeadNum * fagInfo.kvSeqlen * fagInfo.qSeqlen;
    }

    fagTilingData.baseMN = fagTilingData.s1Inner * fagTilingData.s2Inner;
    uint32_t tmpBufferSize = (fagTilingData.ubSize - fagTilingData.s1Inner * fagTilingData.s2Inner * BASIC_BLOCK_MULTIPLE -
                              fagTilingData.s1Inner * SHAPE_INFO * fagTilingData.calTypeSize) /
                             BYTE_BLOCK * BYTE_BLOCK;
    fagTilingData.tmpBufferSize = tmpBufferSize;

    // pre tiling_data
    if (!DoPreTiling(fagTilingData)) {
        cerr << "get DoPreTiling fail." << endl;
        return -1;
    }
    DoPreSfmgTiling(fagTilingData);
    if (!DoPostTiling(fagTilingData)) {
        cerr << "get DoPostTiling fail." << endl;
        return -1;
    }
    FillWorkSpaceTilingData(fagTilingData);

    return 0;
}

} // FAGTiling
