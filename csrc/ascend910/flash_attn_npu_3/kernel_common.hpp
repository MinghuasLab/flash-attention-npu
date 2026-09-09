/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef KERNEL_COMMON
#define KERNEL_COMMON


namespace KernelCommon {
    constexpr uint32_t QK_READY_ID = 1;
    constexpr uint32_t SOFTMAX_READY_ID = 2;
    constexpr uint32_t PV_READY_ID = 3;
    constexpr uint32_t PRE_LAUNCH = 2;
    constexpr uint32_t N_SPLIT_HELPER = 2;
    constexpr uint32_t MAX_KV_STACK_LEN = 512;
    constexpr uint32_t Q_TILE_CEIL = 128;
    constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
    constexpr uint32_t L1_MAX_SIZE = 524288;
    constexpr uint32_t L1_MAX_N_NUM = 128;
    constexpr uint32_t DOUBLE_BUFFER = 2;
    constexpr uint32_t COMP_TRIU_MASK_DIM_LEN = 2048;
    constexpr uint32_t NUM_32 = 32;
    constexpr uint32_t NUM_128 = 128;
    constexpr uint32_t NUM_256 = 256;

    template <typename T>
    __aicore__ inline
    T AlignUp(T a, T b)
    {
        return (b == 0) ? 0 : (a + b - 1) / b * b;
    }

    template <typename T>
    __aicore__ inline
    T Max(T a, T b)
    {
        return (a > b) ? a : b;
    }

    namespace FaiKenel {
        constexpr uint32_t BLOCK_SIZE = 16;

        enum class cvPipeLineType : uint32_t {
            FAI_COMMON_NORMAL = 0,
            FAI_COMMON_CHUNK_MASK = 1,
        };

        enum class MaskType : uint32_t {
            NO_MASK = 0,
            MASK_CAUSAL = 1,
            MASK_BAND = 2,
            MASK_SWA = 4
        };

        enum class inputLayout : uint32_t {
            BSND = 0,
            TND = 1
        };
    };

    struct FAIKernelParams {
        GM_ADDR q;
        GM_ADDR k;
        GM_ADDR v;
        GM_ADDR mask;
        GM_ADDR blockTables;
        GM_ADDR actualQseqlen;
        GM_ADDR actualKvseqlen;
        GM_ADDR o;
        GM_ADDR lse;
        GM_ADDR workSpace;
        GM_ADDR tiling;
        GM_ADDR kNew;
        GM_ADDR vNew;

        // Element strides are launch-time address metadata, deliberately kept
        // outside scheduler_metadata: a cached scheduler buffer is therefore
        // reusable by tensors with the same shape but different views.
        uint64_t qBatchStride;
        uint64_t qSeqStride;
        uint64_t qHeadStride;
        uint64_t kBatchStride;
        uint64_t kSeqStride;
        uint64_t kHeadStride;
        uint64_t vBatchStride;
        uint64_t vSeqStride;
        uint64_t vHeadStride;
        uint64_t kNewBatchStride;
        uint64_t kNewSeqStride;
        uint64_t kNewHeadStride;
        uint64_t vNewBatchStride;
        uint64_t vNewSeqStride;
        uint64_t vNewHeadStride;

        __aicore__ inline FAIKernelParams() {}

        __aicore__ inline FAIKernelParams(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR blockTables_,
                GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_, GM_ADDR o_, GM_ADDR lse_, GM_ADDR workSpace_,
                    GM_ADDR tiling_, GM_ADDR kNew_, GM_ADDR vNew_,
                    uint64_t qBatchStride_, uint64_t qSeqStride_, uint64_t qHeadStride_,
                    uint64_t kBatchStride_, uint64_t kSeqStride_, uint64_t kHeadStride_,
                    uint64_t vBatchStride_, uint64_t vSeqStride_, uint64_t vHeadStride_,
                    uint64_t kNewBatchStride_, uint64_t kNewSeqStride_, uint64_t kNewHeadStride_,
                    uint64_t vNewBatchStride_, uint64_t vNewSeqStride_, uint64_t vNewHeadStride_)
            : q(q_), k(k_), v(v_), mask(mask_), blockTables(blockTables_), actualQseqlen(actualQseqlen_),
                actualKvseqlen(actualKvseqlen_), o(o_), lse(lse_), workSpace(workSpace_), tiling(tiling_),
                kNew(kNew_), vNew(vNew_),
                qBatchStride(qBatchStride_), qSeqStride(qSeqStride_), qHeadStride(qHeadStride_),
                kBatchStride(kBatchStride_), kSeqStride(kSeqStride_), kHeadStride(kHeadStride_),
                vBatchStride(vBatchStride_), vSeqStride(vSeqStride_), vHeadStride(vHeadStride_),
                kNewBatchStride(kNewBatchStride_), kNewSeqStride(kNewSeqStride_), kNewHeadStride(kNewHeadStride_),
                vNewBatchStride(vNewBatchStride_), vNewSeqStride(vNewSeqStride_), vNewHeadStride(vNewHeadStride_) {}
    };

    __aicore__ inline uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
    {
        uint32_t qNBlockTile = (qSeqlen != 0) ?
            (Q_TILE_CEIL / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
        qNBlockTile = qNBlockTile < groupSize ? qNBlockTile : groupSize;
        qNBlockTile = qNBlockTile < 1 ? 1 : qNBlockTile;
        return qNBlockTile;
    }

    __aicore__ inline uint32_t GetQSBlockTile(uint32_t kvSeqlen)
    {
        uint32_t qSBlockTile = Q_TILE_CEIL;
        return qSBlockTile;
    }

    __aicore__ inline
    uint32_t GetStackSeqTile(bool isNewBlock, uint32_t kvSIdx, uint32_t kvSIdxLocal,
                                uint32_t kvLoopNumTotalNew, uint32_t kvSLoopNumTotalOld,
                                int64_t noSkipKvS, uint32_t kvSeqlenOld, uint32_t kvNewSeqlen)
    {
        if (isNewBlock) {
            return (kvSIdxLocal + 1 > kvLoopNumTotalNew - 1U) ?
                (kvNewSeqlen - kvSIdxLocal * MAX_KV_STACK_LEN) : MAX_KV_STACK_LEN;
        }
        if (kvSIdx + 1 > kvSLoopNumTotalOld - 1U) {
            return AscendC::Std::min(noSkipKvS, (int64_t)kvSeqlenOld) - kvSIdx * MAX_KV_STACK_LEN;
        }
        return MAX_KV_STACK_LEN;
    }
}
#endif
