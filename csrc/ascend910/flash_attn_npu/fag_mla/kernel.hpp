#pragma once

// MLA-only scheduling and pipeline; shared by v3 and v4.
template <class BlockMmad_, class BlockMmad2_, class BlockMmad3_, class EpilogueFAGPre_, class EpilogueFAGSfmg_,
          class EpilogueFAGSabVec_, class EpilogueFAGPost_, const uint32_t INPUT_LAYOUT, const bool IS_ATTEN_MASK>
class FlashAttentionScoreGradMla {
  public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    // cube23 set
    using BlockMmad2 = BlockMmad2_;
    using LayoutA2 = typename BlockMmad2::LayoutA;
    using LayoutB2 = typename BlockMmad2::LayoutB;
    using LayoutC2 = typename BlockMmad2::LayoutC;

    using BlockMmad3 = BlockMmad3_;
    using LayoutA3 = typename BlockMmad3::LayoutA;
    using LayoutB3 = typename BlockMmad3::LayoutB;
    using LayoutC3 = typename BlockMmad3::LayoutC;

    using EpilogueFAGPre = EpilogueFAGPre_;
    using EpilogueFAGSfmg = EpilogueFAGSfmg_;
    using EpilogueFAGSabVec = EpilogueFAGSabVec_;
    using EpilogueFAGPost = EpilogueFAGPost_;

    AscendC::GlobalTensor<ElementA> queryGm;
    AscendC::GlobalTensor<ElementA> keyGm;
    AscendC::GlobalTensor<ElementA> valueGm;
    AscendC::GlobalTensor<ElementA> dxGm;

    AscendC::GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    AscendC::GlobalTensor<float> mm1WorkspaceGm, mm2WorkspaceGm;
    AscendC::GlobalTensor<ElementA> dropWorkSpaceGm, mulWorkSpaceGm;

    __gm__ uint8_t* actual_seq_qlen_addr;
    __gm__ uint8_t* actual_seq_kvlen_addr;
    __gm__ uint8_t* seqUsedQAddr;
    __gm__ uint8_t* seqUsedKvAddr;
    bool hasSeqUsedQ;
    bool hasSeqUsedKv;

    constexpr static uint32_t ENABLE = 1;

    __aicore__ inline void Init(FAGKernelParams const& params, __gm__ FAGTilingData* fagTilingData)
    {
        queryGm.SetGlobalBuffer((__gm__ ElementA*)params.q);
        keyGm.SetGlobalBuffer((__gm__ ElementA*)params.k);
        valueGm.SetGlobalBuffer((__gm__ ElementA*)params.v);
        dxGm.SetGlobalBuffer((__gm__ ElementA*)params.dout);

        coreNum = fagTilingData->coreNum;
        cubeCoreNum = coreNum / 2;

        vecBlockNum = coreNum / 3;

        // shape info
        b = fagTilingData->batch;
        n2 = fagTilingData->kvHeadNum;
        g = fagTilingData->g;
        s1 = fagTilingData->qSeqlen;
        s2 = fagTilingData->kvSeqlen;
        d = fagTilingData->qkHeadDim;
        value_d = fagTilingData->vHeadDim;
        dAlign = (d + 15) / 16 * 16;
        value_dAlign = (value_d + 15) / 16 * 16;
        inputStrides.q = {fagTilingData->inputStrides.q.batch, fagTilingData->inputStrides.q.sequence,
                          fagTilingData->inputStrides.q.head};
        inputStrides.k = {fagTilingData->inputStrides.k.batch, fagTilingData->inputStrides.k.sequence,
                          fagTilingData->inputStrides.k.head};
        inputStrides.v = {fagTilingData->inputStrides.v.batch, fagTilingData->inputStrides.v.sequence,
                          fagTilingData->inputStrides.v.head};
        inputStrides.dout = {fagTilingData->inputStrides.dout.batch, fagTilingData->inputStrides.dout.sequence,
                             fagTilingData->inputStrides.dout.head};
        inputStrides.out = {fagTilingData->inputStrides.out.batch, fagTilingData->inputStrides.out.sequence,
                            fagTilingData->inputStrides.out.head};
        inputStrides.enabled = fagTilingData->inputStrides.enabled;
        if (inputStrides.enabled == 0) {
            inputStrides.q = {s1 * n2 * g * d, n2 * g * d, d};
            inputStrides.k = {s2 * n2 * d, n2 * d, d};
            inputStrides.v = {s2 * n2 * value_d, n2 * value_d, value_d};
            inputStrides.dout = {s1 * n2 * g * value_d, n2 * g * value_d, value_d};
            inputStrides.out = inputStrides.dout;
        }

        s1Token = fagTilingData->s1Token;
        s2Token = fagTilingData->s2Token;
        actualCalcS1Token = s1Token;
        actualCalcS2Token = s2Token;

        // split info
        s1Outer = fagTilingData->s1Outer;
        s1CvInner = fagTilingData->s1CvInner;
        s1CvTail = fagTilingData->s1CvTail;
        s2Outer = fagTilingData->s2Outer;
        s2CvInner = fagTilingData->s2CvInner;
        s2CvTail = s2 - (s2Outer - 1) * s2CvInner;
        cubeBaseMN = s1CvInner * s2CvInner;

        actual_seq_qlen_addr = params.cu_seq_qlen;
        actual_seq_kvlen_addr = params.cu_seq_kvlen;
        seqUsedQAddr = params.seqUsedQ;
        seqUsedKvAddr = params.seqUsedKv;
        hasSeqUsedQ = params.seqUsedQ != nullptr;
        hasSeqUsedKv = params.seqUsedKv != nullptr;
        shortHeadSchedule = INPUT_LAYOUT == TND && b == 1 && g == 1 && n2 >= 4 &&
                            n2 % cubeCoreNum == 0 && d == 192 && value_d == 128 && s1 > 0 && s1 <= 256 && s1 == s2 &&
                            s1CvInner == 128 && fagTilingData->maskType == 1 && !hasSeqUsedQ && !hasSeqUsedKv;
        shortOwnedWrite = shortHeadSchedule && s2Outer == 1 && fagTilingData->fullPackedTensors;
        independentHeadStreams = shortHeadSchedule && n2 == 2 * cubeCoreNum;

        // From tiling: cu[b - 1] would undercount after used-length tail trimming.
        int64_t sfmgOutputSize = static_cast<int64_t>(fagTilingData->sfmgNormalAxisSize) * 8;

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)params.workspace +
                                      fagTilingData->dqWorkSpaceOffset / sizeof(float));
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)params.workspace +
                                      fagTilingData->dkWorkSpaceOffset / sizeof(float));
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float*)params.workspace +
                                      fagTilingData->dvWorkSpaceOffset / sizeof(float));

        int64_t workspaceOffsets =
            (fagTilingData->sfmgPreBeginAddr + sfmgOutputSize * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE *
            ADDR_ALIGN_SIZE;

        // matmul1 and matmul2 workspace size
        uint32_t matmulWorkspaceSize = cubeBaseMN * sizeof(float);
        mm1WorkspaceGm.SetGlobalBuffer((__gm__ float*)(params.workspace + workspaceOffsets +
                                                       cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));
        mm2WorkspaceGm.SetGlobalBuffer((__gm__ float*)(params.workspace + workspaceOffsets +
                                                       cubeCoreNum * matmulWorkspaceSize * GM_DOUBLE_BUFFER +
                                                       cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

        // drop workspace offset 和 mm2WorkspaceGm 地址相同
        dropWorkSpaceGm.SetGlobalBuffer((__gm__ ElementA*)(params.workspace + workspaceOffsets +
                                                           cubeCoreNum * matmulWorkspaceSize * GM_DOUBLE_BUFFER +
                                                           cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

        // mul workspace offset 和 mm1WorkspaceGm 地址相同
        mulWorkSpaceGm.SetGlobalBuffer((__gm__ ElementA*)(params.workspace + workspaceOffsets +
                                                          cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));
    }

    __aicore__ inline void GetSeqQlenKvlenByBidx(int64_t bIdx, int32_t& actualSeqQlen, int32_t& actualSeqKvlen)
    {
        // *Used* lengths (seqused-aware); task partitioning and window math follow.
        if (unlikely(bIdx == 0)) {
            actualSeqQlen =
                hasSeqUsedQ ? ((__gm__ int32_t*)seqUsedQAddr)[0] : ((__gm__ int32_t*)actual_seq_qlen_addr)[0];
            actualSeqKvlen =
                hasSeqUsedKv ? ((__gm__ int32_t*)seqUsedKvAddr)[0] : ((__gm__ int32_t*)actual_seq_kvlen_addr)[0];
        } else {
            actualSeqQlen = hasSeqUsedQ ? ((__gm__ int32_t*)seqUsedQAddr)[bIdx]
                                        : ((__gm__ int32_t*)actual_seq_qlen_addr)[bIdx] -
                                              ((__gm__ int32_t*)actual_seq_qlen_addr)[bIdx - 1];
            actualSeqKvlen = hasSeqUsedKv ? ((__gm__ int32_t*)seqUsedKvAddr)[bIdx]
                                          : ((__gm__ int32_t*)actual_seq_kvlen_addr)[bIdx] -
                                                ((__gm__ int32_t*)actual_seq_kvlen_addr)[bIdx - 1];
        }
        return;
    }

    __aicore__ inline void UpdateIndex()
    {
        s1oDimIdx = 0;
        if (s2oCvDimIdx < s2Outer - 1) {
            s2oCvDimIdx++;
        } else if (gDimIdx < g - 1) {
            s2oCvDimIdx = 0;
            gDimIdx++;
        } else if (n2DimIdx < n2 - 1) {
            gDimIdx = 0;
            s2oCvDimIdx = 0;
            n2DimIdx++;
        } else {
            gDimIdx = 0;
            s2oCvDimIdx = 0;
            n2DimIdx = 0;
            bDimIdx++;
        }
    }

    __aicore__ inline bool CalcValidBlock(int64_t& baseIdx, int64_t& startCoreId, DBParams& dbParam)
    {
        if (shortHeadSchedule) {
            // Match CANN's contiguous per-AIV head ranges: AIV 2c and 2c+1
            // own adjacent heads, each traversing Q blocks in increasing order.
            const int64_t head = independentHeadStreams ? 2 * cCubeBlockIdx + shortTaskCursor % 2
                                                        : cCubeBlockIdx + (shortTaskCursor / s1Outer) * cubeCoreNum;
            if (head >= n2 || (independentHeadStreams && shortTaskCursor >= 2 * s1Outer)) {
                dbParam.blockId = -1;
                baseIdx = -1;
                return true;
            }
            cachedActualSeqQlen = s1;
            cachedActualSeqKvlen = s2;
            dbParam.bIdx = 0;
            dbParam.n2Idx = head;
            dbParam.gIdx = 0;
            dbParam.s2oIdx = 0;
            dbParam.s1oIdx = independentHeadStreams ? shortTaskCursor / 2 : shortTaskCursor % s1Outer;
            dbParam.blockId = head * s1Outer + dbParam.s1oIdx;
            dbParam.s1CvExtend = dbParam.s1oIdx == s1Outer - 1 ? s1CvTail : s1CvInner;
            const int64_t right = dbParam.s1oIdx * s1CvInner + dbParam.s1CvExtend;
            dbParam.s2CvExtend = right < s2 ? right : s2;
            dbParam.s1CvExtendAlign = (dbParam.s1CvExtend + 15) / 16 * 16;
            dbParam.s2CvExtendAlign = (dbParam.s2CvExtend + 15) / 16 * 16;
            ++shortTaskCursor;
            baseIdx = shortTaskCursor;
            return true;
        }
        if (bDimIdx >= b) { // 越界情形
            if (cCubeBlockIdx >= startCoreId) {
                dbParam.blockId = -1; // 没有数据需要处理
            }
            baseIdx = -1;
            return true;
        }
        int32_t actualSeqQlen = s1;
        int32_t actualSeqKvlen = s2;
        if constexpr (INPUT_LAYOUT == TND) {
            if (static_cast<uint32_t>(bDimIdx) != cachedBatch) {
                cachedBatch = static_cast<uint32_t>(bDimIdx);
                GetSeqQlenKvlenByBidx(bDimIdx, cachedActualSeqQlen, cachedActualSeqKvlen);
                if constexpr (IS_ATTEN_MASK == ENABLE) {
                    actualCalcS1Token = s1Token + cachedActualSeqQlen - cachedActualSeqKvlen;
                    actualCalcS2Token = s2Token - cachedActualSeqQlen + cachedActualSeqKvlen;
                }
                s1Outer = (cachedActualSeqQlen + s1CvInner - 1) / s1CvInner;
                s2Outer = (cachedActualSeqKvlen + s2CvInner - 1) / s2CvInner;
                s1CvTail = cachedActualSeqQlen - (s1Outer - 1) * s1CvInner;
                s2CvTail = cachedActualSeqKvlen - (s2Outer - 1) * s2CvInner;
            }
            actualSeqQlen = cachedActualSeqQlen;
            actualSeqKvlen = cachedActualSeqKvlen;
        }

        int64_t s1IdxUp = s2oCvDimIdx * s2CvInner - actualCalcS2Token;
        int64_t s1IdxDown = (s2oCvDimIdx + 1) * s2CvInner + actualCalcS1Token;

        // s2token 保护: 1、sparse场景，基本块无效。2、s1==0 or s2==0 场景，无基本块。跳过
        if (s1IdxUp >= actualSeqQlen || actualSeqKvlen == 0) {
            // 当出现s2==0的空Tensor场景时，不进行基本块编号
            baseIdx += actualSeqKvlen ? (s1Outer - s1oDimIdx) : 0;
            UpdateIndex();
            return false;
        }

        int64_t s1oIdxUp = s1IdxUp / s1CvInner;
        s1oIdxUp = s1oIdxUp > 0 ? s1oIdxUp : 0;

        s1IdxDown = s1IdxDown > actualSeqQlen ? actualSeqQlen : s1IdxDown;
        int64_t s1oIdxDown = (s1IdxDown + s1CvInner - 1) / s1CvInner - 1;

        // 当前s1方向没有有效基本块或者起始位置在preToken下方，跳过当前s1
        if (s1oIdxDown < s1oIdxUp || s1oDimIdx > s1oIdxDown) {
            baseIdx += (s1Outer - s1oDimIdx);
            UpdateIndex();
            return false;
        }
        // 起始在nextToken上方，跳到第一个有效块
        if (s1oDimIdx < s1oIdxUp) {
            baseIdx += s1oIdxUp - s1oDimIdx;
            s1oDimIdx = s1oIdxUp;
        }
        int64_t validNum = s1oIdxDown - s1oDimIdx + 1;
        if (cCubeBlockIdx >= startCoreId && cCubeBlockIdx - startCoreId < validNum) {
            dbParam.blockId = baseIdx + cCubeBlockIdx - startCoreId;
            dbParam.bIdx = bDimIdx;
            dbParam.n2Idx = n2DimIdx;
            dbParam.s2oIdx = s2oCvDimIdx;
            dbParam.gIdx = gDimIdx;
            dbParam.s1oIdx = s1oDimIdx + cCubeBlockIdx - startCoreId;
            dbParam.s1CvExtend = (dbParam.s1oIdx == s1Outer - 1) ? s1CvTail : s1CvInner;
            dbParam.s2CvExtend = (dbParam.s2oIdx == s2Outer - 1) ? s2CvTail : s2CvInner;
            int64_t s2RightIdx = dbParam.s1oIdx * s1CvInner + dbParam.s1CvExtend + actualCalcS2Token;
            s2RightIdx = s2RightIdx > 0 ? s2RightIdx : 0;
            s2RightIdx = (s2RightIdx + 7) / 8 * 8;
            dbParam.s2CvExtend = s2RightIdx > (dbParam.s2oIdx * s2CvInner + dbParam.s2CvExtend)
                                     ? dbParam.s2CvExtend
                                     : s2RightIdx - dbParam.s2oIdx * s2CvInner;
            dbParam.s2CvExtend = dbParam.s2CvExtend > 0 ? dbParam.s2CvExtend : 0;
            dbParam.s1CvExtendAlign = (dbParam.s1CvExtend + 15) / 16 * 16;
            dbParam.s2CvExtendAlign = (dbParam.s2CvExtend + 15) / 16 * 16;
        }

        if (cubeCoreNum - startCoreId <= validNum) {
            baseIdx = baseIdx + cubeCoreNum - startCoreId;
            if ((s1oDimIdx + cubeCoreNum - startCoreId) < s1Outer) {
                s1oDimIdx = s1oDimIdx + cubeCoreNum - startCoreId;
            } else {
                UpdateIndex();
            }
            return true;
        } else {
            baseIdx += (s1Outer - s1oDimIdx);
            startCoreId += validNum;
            UpdateIndex();
            return false;
        }
    }

    __aicore__ inline int64_t InputOffset(const FagTensorStrides& strides, const DBParams& task, bool isQ)
    {
        int64_t token = isQ ? task.s1oIdx * s1CvInner : task.s2oIdx * s2CvInner;
        int64_t batchOffset = task.bIdx * strides.batch;
        if constexpr (INPUT_LAYOUT == TND) {
            batchOffset = 0;
            if (task.bIdx > 0) {
                token += isQ ? ((__gm__ int32_t*)actual_seq_qlen_addr)[task.bIdx - 1]
                             : ((__gm__ int32_t*)actual_seq_kvlen_addr)[task.bIdx - 1];
            }
        }
        const int64_t head = isQ ? task.n2Idx * g + task.gIdx : task.n2Idx;
        return batchOffset + token * strides.sequence + head * strides.head;
    }

    __aicore__ inline void ComputeMM1(DBParams& dbParam)
    {
        pingpongIdx = dbParam.taskId % 2;
        dbParam.actualS1Len = s1;
        dbParam.actualS2Len = s2;
        dbParam.s1Stride = 0;
        dbParam.s2Stride = 0;

        if constexpr (INPUT_LAYOUT == TND) {
            // Lengths from the batch-once cache; offsets read per task like the forward.
            dbParam.actualS1Len = cachedActualSeqQlen;
            dbParam.actualS2Len = cachedActualSeqKvlen;
            dbParam.aTensorOffsetCv = 0;
            dbParam.bTensorOffsetCv = 0;
            if (dbParam.bIdx > 0) {
                dbParam.aTensorOffsetCv = ((__gm__ int32_t*)actual_seq_qlen_addr)[dbParam.bIdx - 1] * n2 * g * d;
                dbParam.bTensorOffsetCv = ((__gm__ int32_t*)actual_seq_kvlen_addr)[dbParam.bIdx - 1] * n2 * d;
            }
            dbParam.aTensorOffsetCv += ((dbParam.s1oIdx * s1CvInner * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * d;
            dbParam.bTensorOffsetCv += (dbParam.s2oIdx * s2CvInner * n2 + dbParam.n2Idx) * d;
            dbParam.s1Stride = n2 * g * d;
            dbParam.s2Stride = n2 * d;
        } else {
            dbParam.aTensorOffsetCv =
                (((dbParam.bIdx * s1 + dbParam.s1oIdx * s1CvInner) * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * d;
            dbParam.bTensorOffsetCv = ((dbParam.bIdx * s2 + dbParam.s2oIdx * s2CvInner) * n2 + dbParam.n2Idx) * d;
            dbParam.s1Stride = n2 * g * d;
            dbParam.s2Stride = n2 * d;
        }

        int64_t s1_size = dbParam.actualS1Len;

        // dy * v need align the d to value_d.
        int64_t specify_for_v_aTensorOffsetCv = InputOffset(inputStrides.dout, dbParam, true);
        int64_t specify_for_v_bTensorOffsetCv = InputOffset(inputStrides.v, dbParam, false);
        int64_t specify_for_v_s1Stride = inputStrides.dout.sequence;
        int64_t specify_for_v_s2Stride = inputStrides.v.sequence;

        // mm-dyv
        GemmCoord actualBlockShape1(dbParam.s1CvExtend, dbParam.s2CvExtend, value_d);
        LayoutA layoutA1{actualBlockShape1.m(), actualBlockShape1.k(), specify_for_v_s1Stride};
        LayoutB layoutB1{actualBlockShape1.k(), actualBlockShape1.n(),
                         specify_for_v_s2Stride}; // ColumnMajor shape=(d,s2)  stride=n2*g*d
        LayoutC layoutC1{actualBlockShape1.m(), actualBlockShape1.n(), dbParam.s2CvExtendAlign};

        {
            BlockMmad blockMmadDyV(resource);
            blockMmadDyV(dxGm[specify_for_v_aTensorOffsetCv], layoutA1, valueGm[specify_for_v_bTensorOffsetCv],
                         layoutB1, mm1WorkspaceGm[pingpongIdx * cubeBaseMN], layoutC1, actualBlockShape1, true,
                         !shortHeadSchedule);
        }

        // L1/L0A/L0B are drained; short tasks carry L0C free flags into QK.
        if (!shortHeadSchedule) {
            AscendC::PipeBarrier<PIPE_MTE3>();
        }

        // mm-qk
        GemmCoord actualBlockShape(dbParam.s1CvExtend, dbParam.s2CvExtend, d);
        LayoutA layoutA{actualBlockShape.m(), actualBlockShape.k(), inputStrides.q.sequence};
        LayoutB layoutB{actualBlockShape.k(), actualBlockShape.n(), inputStrides.k.sequence};
        LayoutC layoutC{actualBlockShape.m(), actualBlockShape.n(), dbParam.s2CvExtendAlign};
        BlockMmad blockMmadQk(resource);
        blockMmadQk(queryGm[InputOffset(inputStrides.q, dbParam, true)], layoutA,
                    keyGm[InputOffset(inputStrides.k, dbParam, false)], layoutB,
                    mm2WorkspaceGm[pingpongIdx * cubeBaseMN], layoutC, actualBlockShape, !shortHeadSchedule, true);
    }

    __aicore__ inline void ComputeMMDqkv(DBParams& dbParam, int64_t nextBlockId)
    {
        pingpongIdx = dbParam.taskId % 2;
        const bool accumulateDq = !shortHeadSchedule;
        const bool accumulateKv = !shortHeadSchedule || dbParam.s1oIdx != 0;
        // Earlier Q blocks initialized only this causal K prefix. New K rows
        // must use ordinary stores when the pre-zero stage is omitted.
        const uint32_t atomicKvRows = shortOwnedWrite ? static_cast<uint32_t>(dbParam.s1oIdx * s1CvInner) : 0xffffffffU;
        int64_t dqOffset = 0;
        int64_t dqRopeOffset = 0;
        int64_t dkvOffset = 0;
        int64_t dkRopeOffset = 0;
        int64_t dvOffset = 0;
        // 默认ND
        dqOffset = dbParam.aTensorOffsetCv;
        dkvOffset = dbParam.bTensorOffsetCv;
        dvOffset = dbParam.bTensorOffsetCv / d * value_d;

        int64_t dqKc = dbParam.s1Stride;
        int64_t dkvKc = dbParam.s2Stride;
        int64_t dvKc = dbParam.s2Stride / d * value_d;
        int64_t s1_size = dbParam.actualS1Len;

        int64_t s2_size = dbParam.s2CvExtendAlign * DTYPE_FACTOR;
        int64_t dvOutMStride = dbParam.actualS2Len;

        // mm4 切分s1
        constexpr int32_t s1SliceSize = S_SPLITT_SIZE;
        int32_t s1SliceNum = (dbParam.s1CvExtend + s1SliceSize - 1) / s1SliceSize;
        int32_t s1SliceTail = dbParam.s1CvExtend - (s1SliceNum - 1) * s1SliceSize;

        // mm3 切分s2
        constexpr int32_t s2SliceSize = S_SPLITT_SIZE;
        int32_t s2SliceNum = (dbParam.s2CvExtend + s2SliceSize - 1) / s2SliceSize;
        int32_t s2SliceTail = dbParam.s2CvExtend - (s2SliceNum - 1) * s2SliceSize;

        // m是s1方向的循环
        uint32_t mSplitSize = MMAD_BASE_SIZE;
        uint32_t mLoops = (dbParam.s1CvExtend + mSplitSize - 1) / mSplitSize;
        uint32_t mTail = dbParam.s1CvExtend - (mLoops - 1) * mSplitSize;
        uint32_t subMSizeAct = mSplitSize;
        uint32_t subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;

        // n 是s2方向的循环
        uint32_t nSplitSize = MMAD_BASE_SIZE;
        uint32_t nLoops = (dbParam.s2CvExtend + nSplitSize - 1) / nSplitSize;
        uint32_t nTail = dbParam.s2CvExtend - (nLoops - 1) * nSplitSize;
        uint32_t subNSizeAct = nSplitSize;
        uint32_t subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;

        // d方向循环
        uint32_t dSplitSize = MMAD_BASE_SIZE;
        uint32_t dLoops = (d + dSplitSize - 1) / dSplitSize;
        uint32_t dTail = d - (dLoops - 1) * dSplitSize;
        uint32_t subDSizeAct = dSplitSize;
        uint32_t subDSizeActAlign = CeilDiv(subDSizeAct, C0_SIZE) * C0_SIZE;

        // value_d方向循环
        uint32_t valueDLoops = (value_d + dSplitSize - 1) / dSplitSize;
        uint32_t valueDTail = value_d - (valueDLoops - 1) * dSplitSize;
        uint32_t gmBaseBlockOffset = 0;

        // ///////////////////////////////////////////////////////////////
        // // Matmal4 dq
        // ///////////////////////////////////////////////////////////////
        // // left [B, N2, G, S1, s2] right [B, N2, 1, S2, D] output [B, N2, G, S1, D]
        GemmCoord actualBlockShape1(dbParam.s1CvExtend, d, dbParam.s2CvExtend);
        LayoutA2 layoutA1{actualBlockShape1.m(), actualBlockShape1.k(), s2_size}; // RowMajor  shape=(s1,s2)  stride=s2
        LayoutB2 layoutB1{actualBlockShape1.k(), actualBlockShape1.n(), inputStrides.k.sequence};
        LayoutC2 layoutC1{actualBlockShape1.m(), actualBlockShape1.n(), dqKc};
        {
            BlockMmad2 blockMmad2(resource);
            blockMmad2(mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], keyGm[InputOffset(inputStrides.k, dbParam, false)],
                       dqWorkSpaceGm[dqOffset], layoutA1, layoutB1, layoutC1, actualBlockShape1, accumulateDq);
        }

        // dq GM write done; blockMmad3 reuses same L1
        AscendC::PipeBarrier<PIPE_MTE3>();

        BlockMmad3 blockMmad3(resource, 0, 4);
        // ///////////////////////////////////////////////////////////////
        // // left [B, N2, G, S1, S2] right [B, N2, 1, S1, D] output [B, N2, G, S2, D]
        GemmCoord actualBlockShape2(dbParam.s2CvExtend, d, dbParam.s1CvExtend);
        LayoutA3 layoutA2{actualBlockShape2.m(), actualBlockShape2.k(),
                          s2_size}; // ColumnMajor  origin shape=(s2,s1), so stride=s2
        LayoutB3 layoutB2{actualBlockShape2.k(), actualBlockShape2.n(), inputStrides.q.sequence};
        LayoutC3 layoutC2{actualBlockShape2.m(), actualBlockShape2.n(), dkvKc};
        blockMmad3(mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], queryGm[InputOffset(inputStrides.q, dbParam, true)],
                   dkWorkSpaceGm[dkvOffset], layoutA2, layoutB2, layoutC2, actualBlockShape2, accumulateKv,
                   atomicKvRows);

        AscendC::PipeBarrier<PIPE_MTE3>();

        // ///////////////////////////////////////////////////////////////
        // // Matmal5 dv
        // ///////////////////////////////////////////////////////////////
        // // left [B, N2, G, S1, S2] right [B, N2, G, S1, D2] output [B, N2, 1, S2, D2]
        GemmCoord actualBlockShape3(dbParam.s2CvExtend, value_d, dbParam.s1CvExtend);
        LayoutA3 layoutA3{actualBlockShape3.m(), actualBlockShape3.k(), s2_size};
        LayoutB3 layoutB3{actualBlockShape3.k(), actualBlockShape3.n(), inputStrides.dout.sequence};
        LayoutC3 layoutC3{actualBlockShape3.m(), actualBlockShape3.n(), dvKc};
        blockMmad3(dropWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], dxGm[InputOffset(inputStrides.dout, dbParam, true)],
                   dvWorkSpaceGm[dvOffset], layoutA3, layoutB3, layoutC3, actualBlockShape3, accumulateKv,
                   atomicKvRows);
    }

    // Methods
    CATLASS_DEVICE
    FlashAttentionScoreGradMla() {}

    CATLASS_DEVICE
    ~FlashAttentionScoreGradMla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(FAGKernelParams const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(FAGKernelParams const& params)
    {
        __gm__ FAGTilingData* fagTilingData = reinterpret_cast<__gm__ FAGTilingData*>(params.tiling);
        cCubeBlockIdx = AscendC::GetBlockIdx();
        cBlockIdx = cCubeBlockIdx * 2;
        Init(params, fagTilingData);

        bool isFinish = false;
        int64_t startCoreId = 0;
        while (!isFinish) {
            isFinish = CalcValidBlock(blockStartIdx, startCoreId, dbParams[0]);
        }

        dbParams[0].taskId = 0;
        if (dbParams[0].blockId != -1) {
            ComputeMM1(dbParams[0]);
            AscendC::CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG[dbParams[0].taskId]);
        }

        int64_t taskId = 1;
        int8_t extraLoopNum = 1;
        while (extraLoopNum >= 0) {
            isFinish = false;
            startCoreId = 0;
            while (!isFinish) {
                isFinish = CalcValidBlock(blockStartIdx, startCoreId, dbParams[taskId % 3]);
            }
            dbParams[taskId % 3].taskId = taskId;
            if (dbParams[taskId % 3].blockId != -1) {
                ComputeMM1(dbParams[taskId % 3]);
                AscendC::CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG[dbParams[taskId % 3].taskId % 2]);
            }
            if (taskId > 0 && dbParams[(taskId - 1) % 3].blockId != -1) {
                AscendC::CrossCoreWaitFlag(SYNC_V1_C2_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
                ComputeMMDqkv(dbParams[(taskId - 1) % 3], dbParams[(taskId) % 3].blockId);
                if (dbParams[(taskId) % 3].blockId == -1) {
                    AscendC::CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(
                        SYNC_C2_V1_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
                }
            }

            taskId++;
            if (blockStartIdx == -1) {
                extraLoopNum -= 1;
            }
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(FAGKernelParams const& params)
    {
        // 给L1custom的变量
        cBlockIdx = AscendC::GetBlockIdx();
        cCubeBlockIdx = cBlockIdx / 2;
        cSubIdx = cBlockIdx % 2;

        __gm__ FAGTilingData* fagTilingData = reinterpret_cast<__gm__ FAGTilingData*>(params.tiling);
        Init(params, fagTilingData);

        if (!shortOwnedWrite) {
            AscendC::TPipe pipePre;
            EpilogueFAGPre epilogueFagPre(resource, &pipePre, params.dq, params.dk, params.dv, params.drop_mask,
                                          params.workspace, params.tiling);
            epilogueFagPre();
            pipePre.Destroy();
            if (shortHeadSchedule) {
                // Padded storage keeps the original zero-before-atomic contract.
                AscendC::SyncAll();
            }
        }

        if (!shortHeadSchedule) {
            // vec SoftmaxGrad
            AscendC::TPipe pipeSoftmaxGrad;
            EpilogueFAGSfmg epilogueFagSfmg(resource, &pipeSoftmaxGrad, params.dout, params.out, params.cu_seq_qlen,
                                            params.workspace, params.tiling);
            epilogueFagSfmg();
            pipeSoftmaxGrad.Destroy();
        }

        // vector process
        int64_t taskId = 1;
        int8_t extraLoopNum = 1;
        AscendC::TPipe pipeVec;
        TBuf<> unifiedBuffer;
        EpilogueFAGSabVec epilogueFAGSabVec(
            resource, &pipeVec, params.q, params.k, params.v, params.dout, params.drop_mask, params.atten_mask,
            params.out, params.softmax_lse, params.cu_seq_qlen, params.cu_seq_kvlen, params.seqUsedQ, params.seqUsedKv,
            params.dq, params.dk, params.dv, params.alibi_slopes, params.workspace, params.tiling, unifiedBuffer);
        // Only initialized shared workspace needs the initial AIV barrier.
        if constexpr (INPUT_LAYOUT != TND) {
            AscendC::SyncAll();
        } else if (!shortOwnedWrite) {
            AscendC::SyncAll();
        }
        epilogueFAGSabVec.taskSfmgEnabled = shortHeadSchedule;
        epilogueFAGSabVec.singleAivTask = independentHeadStreams;

        bool isFinish = false;
        int64_t startCoreId = 0;
        while (!isFinish) {
            isFinish = CalcValidBlock(blockStartIdx, startCoreId, dbParams[0]);
        }
        dbParams[0].taskId = 0;
        while (extraLoopNum >= 0) {
            isFinish = false;
            startCoreId = 0;
            while (!isFinish) {
                isFinish = CalcValidBlock(blockStartIdx, startCoreId, dbParams[taskId % 3]);
            }

            dbParams[taskId % 3].taskId = taskId;
            if (taskId > 0 && dbParams[(taskId - 1) % 3].blockId != -1) {
                const bool ownsTask =
                    !independentHeadStreams || dbParams[(taskId - 1) % 3].n2Idx == 2 * cCubeBlockIdx + cSubIdx;
                if (shortHeadSchedule && ownsTask) {
                    epilogueFAGSabVec.PrepareTaskSfmg(dbParams[(taskId - 1) % 3]);
                }
                AscendC::CrossCoreWaitFlag(SYNC_C1_V1_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);

                if (taskId > 0 && dbParams[(taskId - 1) % 3].blockId != -1) {
                    if (ownsTask) {
                        epilogueFAGSabVec(dbParams[(taskId - 1) % 3]);
                    }
                    // Both AIVs acknowledge every broadcast, including the peer's
                    // task, to preserve the existing mode-2 notification count.
                    AscendC::CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(
                        SYNC_V1_C2_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
                }
                if (dbParams[(taskId) % 3].blockId == -1) {
                    AscendC::CrossCoreWaitFlag(SYNC_C2_V1_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
                }
            }

            taskId++;
            if (blockStartIdx == -1) {
                extraLoopNum -= 1;
            }
        }
        AscendC::SyncAll();
        pipeVec.Destroy();

        // post compute
        AscendC::TPipe pipePost;
        EpilogueFAGPost epilogueFagPost(resource, &pipePost, params.dq, params.dk, params.dv, params.workspace,
                                        params.tiling);
        epilogueFagPost();
        pipePost.Destroy();
    }

  private:
    Arch::Resource<ArchTag> resource;
    constexpr static uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {4, 5, 6};
    static constexpr uint64_t SYNC_C1_V1_FLAG[3] = {1, 2, 3};
    static constexpr uint64_t SYNC_C2_V1_FLAG[3] = {7, 8, 9};
    int64_t blockStartIdx = 0;
    bool shortHeadSchedule = false;
    bool shortOwnedWrite = false;
    bool independentHeadStreams = false;
    int64_t shortTaskCursor = 0;
    DBParams dbParams[3];
    // base info
    int64_t bDimIdx{0};
    int64_t n2DimIdx{0};
    int64_t gDimIdx{0};
    int64_t s1oDimIdx{0};
    int64_t s2oCvDimIdx{0};

    // Batch-once per-batch cache; safe in MM1 because it runs before the next walk.
    uint32_t cachedBatch = 0xFFFFFFFFU;
    int32_t cachedActualSeqQlen = 0;
    int32_t cachedActualSeqKvlen = 0;

    // split info
    int64_t s1Outer;
    uint32_t s1CvInner;
    uint32_t s1CvTail;
    int64_t s2Outer;
    uint32_t s2CvInner;
    uint32_t s2CvTail;

    // org shape info
    FagInputStrides inputStrides;
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t value_d;
    int64_t dAlign;
    int64_t value_dAlign;

    uint32_t coreNum;
    uint32_t cubeCoreNum;
    uint32_t vecBlockNum;
    uint32_t cBlockIdx;
    uint32_t cCubeBlockIdx;
    uint32_t cSubIdx;

    // optional control
    int64_t s1Token;
    int64_t s2Token;
    int64_t actualCalcS1Token;
    int64_t actualCalcS2Token;

    uint32_t pingpongIdx = 1;

    uint32_t cubeBaseMN;
};
