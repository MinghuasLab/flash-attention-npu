#ifndef KERNEL_COMMON_FAG_HPP
#define KERNEL_COMMON_FAG_HPP

constexpr static uint32_t BSND = 0;
constexpr static uint32_t TND = 1;

struct FAGTilingData {
    int64_t coreNum;
    int64_t aicNum;
    uint64_t ubSize;

    int64_t batch;
    int64_t qSeqlen;
    int64_t qHeadNum;
    int64_t qkHeadDim;   // query, key dim
    int64_t kvSeqlen;
    int64_t kvHeadNum;
    int64_t vHeadDim;    // value dim
    int64_t g;
    uint32_t s1Align = 0;
    uint32_t s2Align;
    int64_t s1Outer;
    int64_t s2Outer;
    uint32_t s1Inner;
    uint32_t s2Inner;
    uint32_t s1CvInner;
    uint32_t s2CvInner;
    uint32_t s1Tail;
    uint32_t s2Tail;
    uint32_t s1CvTail;
    uint32_t s2CvTail;
    int64_t sfmgNormalAxisSize = 0;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t sumS1S2Product = 0;

    uint32_t attenMaskOptional;
    uint32_t attenMaskCompressMode = 0;
    uint32_t layoutType;
    float scaleValue;
    float softcapValue;
    float keepProb;

    uint32_t dataTypeSize;
    uint32_t dataBlockNum;
    uint32_t calTypeSize;
    uint32_t calBlockNum;
    int64_t s1Token;
    int64_t s2Token;
    uint32_t blockOuter;
    int64_t blockFactor;

    int64_t qSize;
    int64_t kvSize;
    int64_t vSize;
    int64_t dropMaskSize;

    uint32_t baseMN;
    uint32_t maskType = 0;

    std::vector<int32_t> actualSeqQlen;
    std::vector<int32_t> actualSeqKvlen;

    bool isSparse;
    bool isDeterministic;
    uint32_t tmpBufferSize = 0;

    // pre TilingData
    uint32_t maskCoreNum = 0;
    uint32_t singleUBProcessNum = 0;
    uint32_t maskSingleCoreLoop = 0;
    uint32_t maskLastLoopNum = 0;
    uint32_t maskTailCoreLoop = 0;
    uint32_t maskTailCoreLastLoopNum = 0;
    uint32_t dropoutIsDivisibleBy8 = 1;
    uint64_t dropBeginAddr = 0;

    // preSfmg TilingData
    int64_t sfmgPreBeginAddr = 0;

    // post TIlingData
    uint64_t dqWorkSpaceOffset = 0;
    uint64_t dkWorkSpaceOffset = 0;
    uint64_t dvWorkSpaceOffset = 0;

    uint64_t workspaceSize = 0;

    // sfmg tiling data
    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling softmaxGradTilingData;
    int64_t alibiSlopesBatchStride = 0;
};

struct FAGv2TilingData {
    int64_t pWorkSpaceOffset;
    int64_t dsWorkSpaceOffset;
    uint32_t coreNum;
    float scaleValue;
    float softcapValue;
    int64_t alibiSlopesBatchStride = 0;
    int64_t batch;
    int64_t t1;
    int64_t t2;
    int64_t kvHeadNum;
    int64_t g;
    int64_t qSeqlen;
    int64_t qkHeadDim;
    int64_t qSize;
    int64_t kvSize;
    int64_t dqWorkSpaceOffset;
    int64_t dkWorkSpaceOffset;
    int64_t dvWorkSpaceOffset;
    int64_t sfmgPreBeginAddr;
    int64_t mm1WorkSpaceOffset;
    int64_t mm2WorkSpaceOffset;
    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling softmaxGradTilingData;
};

struct DBParams {
    int64_t blockId;
    int64_t taskId;
    int64_t bIdx;
    int64_t n2Idx;
    int64_t s2oIdx;
    int64_t gIdx;
    int64_t s1oIdx;
    int32_t s1CvExtend;
    int32_t s2CvExtend;
    int32_t s1CvExtendAlign;
    int32_t s2CvExtendAlign;
    int64_t aTensorOffsetCv{0};
    int64_t bTensorOffsetCv{0};
    int32_t actualS1Len{0};
    int32_t actualS2Len{0};
    int64_t s1Stride;
    int64_t s2Stride;
    int64_t blockIdArr[24];  //确定性计算预留
    int32_t s1CvExtendArr[24];
    int32_t s2CvExtendArr[24];
    int8_t dqGroupId[24];
    int8_t kvGroupId[24];
};

struct FAGKernelParams {
    GM_ADDR dout;
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR out;
    GM_ADDR drop_mask;
    GM_ADDR atten_mask;
    GM_ADDR softmax_lse;
    GM_ADDR cu_seq_qlen;
    GM_ADDR cu_seq_kvlen;
    GM_ADDR dq;
    GM_ADDR dk;
    GM_ADDR dv;
    GM_ADDR alibi_slopes;
    GM_ADDR workspace;
    GM_ADDR tiling;
    // Methods
    CATLASS_DEVICE
    FAGKernelParams() {
    }
    CATLASS_DEVICE
    FAGKernelParams(GM_ADDR dout_,
                    GM_ADDR q_,
                    GM_ADDR k_,
                    GM_ADDR v_,
                    GM_ADDR out_,
                    GM_ADDR drop_mask_, // nullptr
                    GM_ADDR atten_mask_,
                    GM_ADDR softmax_lse_,
                    GM_ADDR cu_seq_qlen_,
                    GM_ADDR cu_seq_kvlen_,
                    GM_ADDR dq_,
                    GM_ADDR dk_,
                    GM_ADDR dv_,
                    GM_ADDR alibi_slopes_,
                    GM_ADDR workspace_,
                    GM_ADDR tiling_)
        : dout(dout_)
        , q(q_)
        , k(k_)
        , v(v_)
        , out(out_)
        , drop_mask(drop_mask_)
        , atten_mask(atten_mask_)
        , softmax_lse(softmax_lse_)
        , cu_seq_qlen(cu_seq_qlen_)
        , cu_seq_kvlen(cu_seq_kvlen_)
        , dq(dq_)
        , dk(dk_)
        , dv(dv_)
        , alibi_slopes(alibi_slopes_)
        , workspace(workspace_)
        , tiling(tiling_){
    }
};

inline int64_t CeilCommon(int64_t num1, int64_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <class T> inline T AlignTo(const T n, const T alignSize)
{
    if (alignSize == 0) {
        return 0;
    }
    return (n + alignSize - 1) & (~(alignSize - 1));
}

template <typename T> inline T AlignUp(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

#endif
