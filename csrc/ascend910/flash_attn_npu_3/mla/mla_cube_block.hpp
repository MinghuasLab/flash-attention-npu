/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0.
// attention/fused_infer_attention_score/op_kernel/arch22/
// fia_block_cube_nonquant_mla.h::ComputeMm1/2, ProcessMm1/2.
// BSND/TND (optional independent strides), Hkv=1, split-RoPE FP16/BF16.
// QK/PV retain the source loops, seven L1 slots and synchronization protocol.
// This is a component, not a forward entry point; its caller validates bounds.
#pragma once
#include "mla_mmad_tile.hpp"
#include "mla_paged_copy_tile.hpp"
namespace flash_attn::mla::Block {
using namespace AscendC;
using Tile::Align;

struct CubeParams {
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    static constexpr uint32_t preLoadNum = 2, nBufferMBaseSize = 256;
    static constexpr uint32_t syncC1V1 = 7, syncV1C2 = 8, syncC2V1 = 4;
    static constexpr uint32_t syncV1NupdateC2 = 5, syncC2V2 = 9;
    static constexpr uint32_t headDim = 512;
    uint32_t qSeqSize, gSize, kvSeqSize;
    uint32_t mmResUbSize, bmm2ResUbSize; // Elements per core per GM slot.
};

// Subset of fia_public_define.h::RunInfo shared by Cube/Vector, same units.
struct RunInfo {
    uint32_t loop = 0, bIdx = 0, n2Idx = 0, gS1Idx = 0;
    uint32_t actMBaseSize = 0, actualSingleProcessSInnerSize = 0;
    uint32_t actualSingleProcessSInnerSizeAlign = 0;
    uint32_t bn2IdxInCurCore = 0, s2BatchOffset = 0;
    uint32_t queryStart = 0; // Compact query token offset, captured per task.
    uint64_t kvStart = 0;    // Compact TND or physical BSND token offset; never truncate b*Sk.
    int64_t nextToken = 0;   // Upper mask offset from this query, using effective lengths.
    int64_t preToken = 0;    // Lower mask offset: first key is query - preToken.
    uint64_t partialRow = 0; // Auto FD row within the persistent partial arrays, captured per task.
    bool isFd = true;        // Source tndIsS2SplitCore; explicit partial kernels always retain FD writeback.
    bool isFirstSInnerLoop = true;
    bool isLastS2Loop = true, isValid = false;
};
struct MSplitInfo {
    uint32_t nBufferStartM, nBufferDealM;
    uint32_t nBufferIdx, vecStartM, vecDealM;
};

template <typename InputT, bool IsTnd = false, bool HasStrides = false, bool IsPaged = false>
class MlaCube {
  public:
    using Q_T = InputT;
    using KV_T = InputT;
    using MM_OUT_T = float;
    __aicore__ inline void Init(TPipe* pipe_, CubeParams params, GM_ADDR query, GM_ADDR queryRope, GM_ADDR sharedKv,
                                GM_ADDR keyRope, const MlaBsndInputLayouts* layouts = nullptr, bool kvIsTnd = IsTnd,
                                GM_ADDR pageTable = nullptr, const MlaPagedCacheLayout* pagedLayout = nullptr)
    {
        pipe = pipe_;
        constInfo = params;
        queryGmTensor.Init(query, params.qSeqSize, params.gSize, 512, HasStrides ? &layouts->qv : nullptr, IsTnd);
        queryRopeGmTensor.Init(queryRope, params.qSeqSize, params.gSize, 64, HasStrides ? &layouts->q : nullptr, IsTnd);
        if constexpr (IsPaged) {
            pagedCache.Init(keyRope, sharedKv, pageTable, *pagedLayout);
        } else {
            keyGmTensor.Init(sharedKv, params.kvSeqSize, 1, 512, HasStrides ? &layouts->v : nullptr, kvIsTnd);
            keyRopeGmTensor.Init(keyRope, params.kvSeqSize, 1, 64, HasStrides ? &layouts->k : nullptr, kvIsTnd);
            valueGmTensor.Init(sharedKv, params.kvSeqSize, 1, 512, HasStrides ? &layouts->v : nullptr, kvIsTnd);
        }
        pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * 4);
        l1QPTensor = bufQPL1.Get<Q_T>();
        pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * 3);
        l1KVTensor = bufKVL1.Get<KV_T>();
        mmL0ABuffers.Init(pipe);
        mmL0BBuffers.Init(pipe);
        pipe->InitBuffer(tmpBufL0C, L0C_BLOCK_SIZE * 2);
        cL0TensorPingPong = tmpBufL0C.Get<MM_OUT_T>();
    }
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<float> qk)
    {
        mm1ResGm = qk;
    }
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<InputT> p, GlobalTensor<float> pv)
    {
        vec1ResGm = p;
        mm2ResGm = pv;
    }
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const RunInfo& info);
    __aicore__ inline void ComputeMm2(const RunInfo& info);

  private:
    template <typename T>
    __aicore__ inline uint32_t BlockAlign(uint32_t s)
    {
        return Align(s, 32U / sizeof(T));
    }
    __aicore__ inline uint32_t GetQPL1RealIdx(uint32_t mIdx, uint32_t kIdx)
    {
        return (mIdx % 2) * 2 + kIdx;
    }
    __aicore__ inline void ProcessMm1(const RunInfo& info, MSplitInfo mSplitInfo);
    __aicore__ inline void ProcessMm2(const RunInfo& info, MSplitInfo mSplitInfo);
    __aicore__ inline void CopyGmToL1(LocalTensor<InputT>& dst, GlobalTensor<InputT>& src, uint32_t rows, uint32_t cols,
                                      uint32_t stride);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<InputT>& dst, const RunInfo& info, uint32_t m, uint32_t rows,
                                          uint32_t cols, uint32_t n);
    static constexpr uint32_t M_L1_SPLIT_SIZE = 128, N_L1_SPLIT_SIZE = 128;
    static constexpr uint32_t MM1_K_L1_SPLIT_SIZE = 288, MM1_K_L0_SPLIT_SIZE = 96, MM2_K_L0_SPLIT_SIZE = 128;
    static constexpr uint32_t L1_BLOCK_SIZE = 128 * 288 * sizeof(InputT), L1_BLOCK_OFFSET = 128 * 288;
    static constexpr uint32_t L0C_BLOCK_SIZE = 65536;
    static constexpr uint32_t L1_EVENT0 = EVENT_ID2, L1_EVENT1 = EVENT_ID3, L1_EVENT2 = EVENT_ID4;
    static constexpr uint32_t L1_EVENT3 = EVENT_ID5, L1_EVENT4 = EVENT_ID6, L1_EVENT5 = EVENT_ID7,
                              L1_EVENT6 = EVENT_ID1;
    static constexpr uint32_t mte21QPIds[4] = {L1_EVENT0, L1_EVENT1, L1_EVENT2, L1_EVENT3};
    static constexpr uint32_t mte21KVIds[3] = {L1_EVENT4, L1_EVENT5, L1_EVENT6};
    CubeParams constInfo;
    TPipe* pipe;
    Tile::DenseTensor<InputT, HasStrides> queryGmTensor, queryRopeGmTensor, keyGmTensor, keyRopeGmTensor, valueGmTensor;
    Tile::CopyQueryGmToL1<InputT, HasStrides> copyQueryGmToL1;
    Tile::CopyKvGmToL1<InputT, HasStrides> copyKvGmToL1;
    Tile::PagedCache<InputT> pagedCache;
    Tile::CopyKKropePAGmToL1<InputT> copyPagedQk;
    Tile::CopyPagedKvGmToL1<InputT> copyPagedV;
    GlobalTensor<float> mm1ResGm, mm2ResGm;
    GlobalTensor<InputT> vec1ResGm;
    TBuf<TPosition::A1> bufQPL1, bufKVL1;
    TBuf<TPosition::CO1> tmpBufL0C;
    LocalTensor<InputT> l1QPTensor, l1KVTensor;
    LocalTensor<float> cL0TensorPingPong;
    Tile::L0Buffers<TPosition::A2> mmL0ABuffers;
    Tile::L0Buffers<TPosition::B2> mmL0BBuffers;
    uint32_t qpL1BufIter = 0, kvL1BufIter = 0, cL0BufIter = 0;
};

template <typename InputT, bool IsTnd, bool HasStrides, bool IsPaged>
__aicore__ inline void MlaCube<InputT, IsTnd, HasStrides, IsPaged>::AllocEventID()
{
    CrossCoreSetFlag<CubeParams::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT2);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT3);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT4);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT5);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT6);
}

template <typename InputT, bool IsTnd, bool HasStrides, bool IsPaged>
__aicore__ inline void MlaCube<InputT, IsTnd, HasStrides, IsPaged>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT2);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT3);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT4);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT5);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT6);

    mmL0ABuffers.Uninit(pipe);
    mmL0BBuffers.Uninit(pipe);
}

template <typename InputT, bool IsTnd, bool HasStrides, bool IsPaged>
__aicore__ inline void
MlaCube<InputT, IsTnd, HasStrides, IsPaged>::CopyGmToL1(LocalTensor<KV_T>& l1Tensor, GlobalTensor<KV_T>& gmSrcTensor,
                                                        uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;

    nd2nzPara.dstNzC0Stride = (srcN + 15) / 16 * 16; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename InputT, bool IsTnd, bool HasStrides, bool IsPaged>
__aicore__ inline void
MlaCube<InputT, IsTnd, HasStrides, IsPaged>::CopyInMm2AToL1(LocalTensor<KV_T>& aL1Tensor, const RunInfo& info,
                                                            uint32_t mSeqIdx, uint32_t subMSizeAct, uint32_t nSize,
                                                            uint32_t nOffset)
{
    // 全量拷贝 确认是否是紧密排布, actualSingleProcessSInnerSize是否需要32B对齐
    auto srcGm = vec1ResGm[(info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                           mSeqIdx * info.actualSingleProcessSInnerSizeAlign + nOffset];
    CopyGmToL1(aL1Tensor, srcGm, subMSizeAct, nSize, info.actualSingleProcessSInnerSizeAlign);
}

template <typename InputT, bool IsTnd, bool HasStrides, bool IsPaged>
__aicore__ inline void MlaCube<InputT, IsTnd, HasStrides, IsPaged>::ComputeMm1(const RunInfo& info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        ProcessMm1(info, mSplitInfo);
        CrossCoreSetFlag<CubeParams::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    }
}

template <typename InputT, bool IsTnd, bool HasStrides, bool IsPaged>
__aicore__ inline void MlaCube<InputT, IsTnd, HasStrides, IsPaged>::ComputeMm2(const RunInfo& info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        CrossCoreWaitFlag(constInfo.syncV1C2);
        ProcessMm2(info, mSplitInfo);
        CrossCoreSetFlag<CubeParams::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);
        CrossCoreSetFlag<CubeParams::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);
    }
}

template <typename InputT, bool IsTnd, bool HasStrides, bool IsPaged>
__aicore__ inline void MlaCube<InputT, IsTnd, HasStrides, IsPaged>::ProcessMm1(const RunInfo& info,
                                                                               const MSplitInfo mSplitInfo)
{
    // 最外层还需要一层m的循环
    uint32_t mSize = mSplitInfo.nBufferDealM;
    uint32_t mL1Size = M_L1_SPLIT_SIZE;
    uint32_t mL1SizeAlign = Align(M_L1_SPLIT_SIZE, 16U);
    uint32_t mL1Loops = (mSize + M_L1_SPLIT_SIZE - 1) / M_L1_SPLIT_SIZE;

    uint32_t nSize = info.actualSingleProcessSInnerSize;
    uint32_t nL1Size = N_L1_SPLIT_SIZE;
    uint32_t nL1SizeAlign = Align(N_L1_SPLIT_SIZE, 16U);
    uint32_t nL1Loops = (nSize + N_L1_SPLIT_SIZE - 1) / N_L1_SPLIT_SIZE;

    uint32_t kSize = 576;
    uint32_t kL1Size = MM1_K_L1_SPLIT_SIZE; // 288
    uint32_t kL1Loops = 2;                  // 2 : 576/288, mla专用 这里不考虑d泛化

    uint32_t kL0Size = 96;
    uint32_t kL0Loops = (kL1Size + kL0Size - 1) / kL0Size; // 288 / 96 = 3 kloops

    LocalTensor<KV_T> bL1Tensor;
    LocalTensor<KV_T> kRopeTensor;
    // ka表示左矩阵4buf选择哪一块buf, kb表示右矩阵3buf选择哪一块buf
    uint32_t ka = 0, kb = 0;

    // L1 切n切k切m
    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) { // L1切n, 512/128
        if (nL1 == (nL1Loops - 1)) {
            // 尾块重新计算size
            nL1Size = nSize - (nL1Loops - 1) * N_L1_SPLIT_SIZE;
            nL1SizeAlign = Align(nL1Size, 16U);
        }

        for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) { // L1切k, 576/288, 这里不考虑d泛化
            uint32_t kb = kvL1BufIter % 3;
            kvL1BufIter++;
            WaitFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]);
            // 从k当中取当前的块
            bL1Tensor = l1KVTensor[kb * L1_BLOCK_OFFSET];
            if (kL1 == 0) {
                Tile::L1Tensor<InputT> dstTensor{.tensor = bL1Tensor, .rowCount = nL1SizeAlign};
                Tile::KvCoord gmCoord{.tokenStart = IsTnd ? info.kvStart : uint64_t(info.bIdx) * constInfo.kvSeqSize,
                                      .n2Idx = info.n2Idx,
                                      .s2Idx = info.s2BatchOffset + nL1 * N_L1_SPLIT_SIZE,
                                      .dIdx = kL1 * 256, // D方向上切256
                                      .s2DealSize = nL1Size,
                                      .dDealSize = 256, // D方向上切256
                                      .batchIdx = info.bIdx};
                Tile::L1Tensor<InputT> dstRopeTensor{
                    .tensor = bL1Tensor[nL1SizeAlign * (BlockAlign<KV_T>(constInfo.headDim) >> 1)],
                    .rowCount = nL1SizeAlign};
                Tile::KvCoord gmCoordKRope{.tokenStart =
                                               IsTnd ? info.kvStart : uint64_t(info.bIdx) * constInfo.kvSeqSize,
                                           .n2Idx = info.n2Idx,
                                           .s2Idx = info.s2BatchOffset + nL1 * N_L1_SPLIT_SIZE,
                                           .dIdx = kL1 * 32U, // D方向上切32
                                           .s2DealSize = nL1Size,
                                           .dDealSize = 32U, // D方向上切32
                                           .batchIdx = info.bIdx};
                if constexpr (IsPaged) {
                    copyPagedQk(dstTensor, dstRopeTensor, pagedCache, gmCoord, gmCoordKRope);
                } else {
                    copyKvGmToL1(dstTensor, keyGmTensor, gmCoord);
                    copyKvGmToL1(dstRopeTensor, keyRopeGmTensor, gmCoordKRope);
                }

            } else {
                Tile::L1Tensor<InputT> dstRopeTensor{.tensor = bL1Tensor, .rowCount = nL1SizeAlign};
                Tile::KvCoord gmCoordKRope{.tokenStart =
                                               IsTnd ? info.kvStart : uint64_t(info.bIdx) * constInfo.kvSeqSize,
                                           .n2Idx = info.n2Idx,
                                           .s2Idx = info.s2BatchOffset + nL1 * N_L1_SPLIT_SIZE,
                                           .dIdx = kL1 * 32U, // D方向上切32
                                           .s2DealSize = nL1Size,
                                           .dDealSize = 32U, // D方向上切32
                                           .batchIdx = info.bIdx};
                Tile::L1Tensor<InputT> dstTensor{.tensor = bL1Tensor[32U * nL1SizeAlign], .rowCount = nL1SizeAlign};
                Tile::KvCoord gmCoord{.tokenStart = IsTnd ? info.kvStart : uint64_t(info.bIdx) * constInfo.kvSeqSize,
                                      .n2Idx = info.n2Idx,
                                      .s2Idx = info.s2BatchOffset + nL1 * N_L1_SPLIT_SIZE,
                                      .dIdx = kL1 * 256U, // D方向上切256
                                      .s2DealSize = nL1Size,
                                      .dDealSize = 256U, // D方向上切256
                                      .batchIdx = info.bIdx};
                if constexpr (IsPaged) {
                    copyPagedQk(dstTensor, dstRopeTensor, pagedCache, gmCoord, gmCoordKRope);
                } else {
                    copyKvGmToL1(dstRopeTensor, keyRopeGmTensor, gmCoordKRope);
                    copyKvGmToL1(dstTensor, keyGmTensor, gmCoord);
                }
            }
            SetFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            WaitFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            mL1Size = M_L1_SPLIT_SIZE;
            mL1SizeAlign = Align(M_L1_SPLIT_SIZE, 16U);
            for (uint32_t mL1 = 0; mL1 < mL1Loops; mL1++) {
                uint32_t aL1PaddingSize = 0; // 用于使左矩阵对齐到尾部, 以保证两块32K内存连续
                if (mL1 == (mL1Loops - 1)) {
                    // 尾块重新计算size
                    mL1Size = mSize - (mL1Loops - 1) * M_L1_SPLIT_SIZE;
                    mL1SizeAlign = Align(mL1Size, 16U);
                    aL1PaddingSize = (M_L1_SPLIT_SIZE - mL1SizeAlign) *
                                     288; // mL1SizeAlign<128 kL1=0时需要偏移, 确保qRope能一半拷贝到当前tensor,
                                          // 一半拷贝到下一个tensor
                }

                // 左矩阵L1选择12块还是34块的index, 由m l1 index决定
                // 左矩阵L1选择12块或34块的前一块还是后一块, 由k l1 index决定
                uint32_t mIdx = qpL1BufIter + mL1;
                ka = GetQPL1RealIdx(mIdx, kL1);
                LocalTensor<Q_T> aL1Tensor =
                    l1QPTensor[ka * L1_BLOCK_OFFSET + (1 - kL1) * aL1PaddingSize]; // kL1=0时需要偏移
                if (nL1 == 0) {                                                    // mL1=0, mL1=1两次
                    if (kL1 == 0) {
                        Tile::L1Tensor<InputT> dstTensor{.tensor = aL1Tensor, .rowCount = Align(mL1Size, 16U)};
                        Tile::QueryCoord gmCoord{
                            .tokenStart = IsTnd ? info.queryStart : uint64_t(info.bIdx) * constInfo.qSeqSize,
                            .n2Idx = info.n2Idx,
                            .gS1Idx = info.gS1Idx + mSplitInfo.nBufferStartM + mL1 * M_L1_SPLIT_SIZE,
                            .dIdx = kL1 * 256,
                            .gS1DealSize = mL1Size,
                            .dDealSize = 256,
                            .batchIdx = info.bIdx};
                        LocalTensor<Q_T> qRopeTensor = aL1Tensor[mL1SizeAlign * 256];
                        Tile::L1Tensor<InputT> dstRopeTensor{.tensor = aL1Tensor[mL1SizeAlign * 256],
                                                             .rowCount = Align(mL1Size, 16U)};
                        Tile::QueryCoord gmCoordQRope{
                            .tokenStart = IsTnd ? info.queryStart : uint64_t(info.bIdx) * constInfo.qSeqSize,
                            .n2Idx = info.n2Idx,
                            .gS1Idx = info.gS1Idx + mSplitInfo.nBufferStartM + mL1 * M_L1_SPLIT_SIZE,
                            .dIdx = 0,
                            .gS1DealSize = mL1Size,
                            .dDealSize = 64,
                            .batchIdx = info.bIdx};
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]);
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka + 1]);
                        copyQueryGmToL1(dstTensor, queryGmTensor, gmCoord);
                        // 由于L1里面是NZ, 这里q rope的偏移为整块q nope切k的后大小, 256为headDim的一半
                        copyQueryGmToL1(dstRopeTensor, queryRopeGmTensor, gmCoordQRope);
                    } else {
                        Tile::L1Tensor<InputT> dstTensor{.tensor =
                                                             aL1Tensor[mL1SizeAlign * 32], // 32 : rope headDim的一半
                                                         .rowCount = Align(mL1Size, 16U)};
                        Tile::QueryCoord gmCoord{
                            .tokenStart = IsTnd ? info.queryStart : uint64_t(info.bIdx) * constInfo.qSeqSize,
                            .n2Idx = info.n2Idx,
                            .gS1Idx = info.gS1Idx + mSplitInfo.nBufferStartM + mL1 * M_L1_SPLIT_SIZE,
                            .dIdx = kL1 * 256,
                            .gS1DealSize = mL1Size,
                            .dDealSize = 256,
                            .batchIdx = info.bIdx};
                        copyQueryGmToL1(dstTensor, queryGmTensor, gmCoord);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
                }

                // 使用unitflag同步
                if (kL1 == 0) {
                    // WaitFlag<HardEvent::FIX_M>(MmFixpEventId(cL0BufIter % 2));
                }
                LocalTensor cL0Tensor =
                    cL0TensorPingPong[(cL0BufIter % 2) *
                                      (L0C_BLOCK_SIZE / sizeof(MM_OUT_T))]; // 需要保证cL0BufIter和m步调一致
                Tile::MMParam param;
                param.singleM = mL1SizeAlign;
                param.singleN = nL1SizeAlign;
                param.singleK = kL1Size;
                param.isLeftTranspose = false;
                param.isRightTranspose = false;
                param.isOutKFisrt = (kL1 == 0) ? true : false;
                param.unitFlag = (kL1 == kL1Loops - 1) ? Tile::UNITFLAG_EN_OUTER_LAST : Tile::UNITFLAG_ENABLE;
                Tile::MatmulKPP<Q_T, KV_T, MM_OUT_T, M_L1_SPLIT_SIZE, N_L1_SPLIT_SIZE, MM1_K_L0_SPLIT_SIZE,
                                Tile::ABLayout::MK, Tile::ABLayout::NK>(aL1Tensor, bL1Tensor, mmL0ABuffers,
                                                                        mmL0BBuffers, cL0Tensor, param);

                if (nL1 == (nL1Loops - 1)) {
                    SetFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]); // 反向同步, 表示L1中的A已经被mte1消费完
                }

                if (kL1 == 1) { // 最后一轮kL1循环
                    // 输出偏移info.loop % (constInfo.preLoadNum)) * mmResUbSize是否在matmul里计算
                    Tile::StoreFp32(
                        mm1ResGm[(info.loop % (constInfo.preLoadNum)) * constInfo.mmResUbSize + nL1 * N_L1_SPLIT_SIZE +
                                 (mSplitInfo.nBufferStartM + mL1 * M_L1_SPLIT_SIZE) *
                                     info.actualSingleProcessSInnerSizeAlign],
                        cL0Tensor, mL1SizeAlign, nL1SizeAlign, info.actualSingleProcessSInnerSizeAlign);
                }
                if (mL1Loops == 2) {
                    cL0BufIter++;
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]); // 反向同步, 表示L1已经被mte1消费完
        }
        if (mL1Loops == 1) {
            cL0BufIter++;
        }
    }
    qpL1BufIter += mL1Loops;
}

template <typename InputT, bool IsTnd, bool HasStrides, bool IsPaged>
__aicore__ inline void MlaCube<InputT, IsTnd, HasStrides, IsPaged>::ProcessMm2(const RunInfo& info,
                                                                               const MSplitInfo mSplitInfo)
{
    uint32_t mSize = mSplitInfo.nBufferDealM;
    uint32_t mSizeAlign = (mSize + 16 - 1) / 16;
    uint32_t mL1Loops = (mSize + M_L1_SPLIT_SIZE - 1) / M_L1_SPLIT_SIZE;
    uint32_t mL1SizeAlign = M_L1_SPLIT_SIZE; // 16对齐
    uint32_t mL1Size = M_L1_SPLIT_SIZE;      // m的实际大小

    uint32_t nSize = BlockAlign<KV_T>(constInfo.headDim);
    uint32_t nL1Loops = (nSize + N_L1_SPLIT_SIZE - 1) / N_L1_SPLIT_SIZE;
    uint32_t nL1SizeAlign = N_L1_SPLIT_SIZE; // 16对齐
    uint32_t nL1Size = N_L1_SPLIT_SIZE;      // n的实际大小

    uint32_t kSize = info.actualSingleProcessSInnerSize;
    uint32_t kL1Size = 256;
    uint32_t kL1SizeAlign = Align(kL1Size, 16U);
    uint32_t kL1Loops = (kSize + kL1Size - 1) / kL1Size;
    uint32_t kL0Size = 128;
    uint32_t kL0Loops = (kL1Size + kL0Size - 1) / kL0Size;
    uint32_t kL0SizeAlign = kL0Size;
    LocalTensor<KV_T> bL1Tensor;  // bL1Tensor需要跨m复用, 需要定义在外面?
    LocalTensor<KV_T> subvTensor; // subvTensor需要跨m复用, 需要定义在外面?

    // ka表示左矩阵4buf选择哪一块buf, kb表示右矩阵3buf选择哪一块buf
    uint32_t ka = 0, kb = 0;
    uint32_t mBaseIdx = qpL1BufIter;
    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) { // n切L1
        if (nL1 == (nL1Loops - 1)) {
            // 尾块
            nL1Size = nSize - (nL1Loops - 1) * N_L1_SPLIT_SIZE;
            nL1SizeAlign = Align(nL1Size, 16U);
        }

        // k l1写成一个循环, 和mm1保持一致
        kL1Size = 256;
        kL1SizeAlign = Align(kL1Size, 16U);
        for (uint32_t k1 = 0; k1 < kL1Loops; k1++) { // k切L1, 这里套了一层l0来操作
            if (k1 == (kL1Loops - 1)) {
                // 尾块
                kL1Size = kSize - (kL1Loops - 1) * 256;
                kL1SizeAlign = Align(kL1Size, 16U);
            }
            uint32_t kb = kvL1BufIter % 3U;
            kvL1BufIter++;
            WaitFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]);
            bL1Tensor = l1KVTensor[kb * L1_BLOCK_OFFSET];
            uint32_t kOffset = k1 * kL0Loops;
            kL0Size = 128U;
            // 此处必须先初始化kL0Size, 再求kL0Loops, 否则由于循环会改变kL0Size大小, 导致kL0Loops错误
            kL0Loops = (kL1Size + kL0Size - 1U) / kL0Size;
            kL0SizeAlign = kL0Size;
            for (uint32_t kL1 = kOffset; kL1 < kL0Loops + kOffset; kL1++) { // 128 循环搬pa
                if (kL1 == kOffset + kL0Loops - 1U) {
                    // 尾块
                    kL0Size = kL1Size - (kL0Loops - 1U) * kL0Size;
                    kL0SizeAlign = Align(kL0Size, 16U);
                }
                Tile::L1Tensor<InputT> dstTensor{.tensor = bL1Tensor[(kL1 - kOffset) * 128U * N_L1_SPLIT_SIZE],
                                                 .rowCount = kL0SizeAlign};
                Tile::KvCoord gmCoord{.tokenStart = IsTnd ? info.kvStart : uint64_t(info.bIdx) * constInfo.kvSeqSize,
                                      .n2Idx = info.n2Idx,
                                      .s2Idx = info.s2BatchOffset + (kL1 - kOffset) * 128U + k1 * 256U,
                                      .dIdx = nL1 * N_L1_SPLIT_SIZE,
                                      .s2DealSize = kL0Size,
                                      .dDealSize = N_L1_SPLIT_SIZE,
                                      .batchIdx = info.bIdx};
                if constexpr (IsPaged) {
                    copyPagedV(dstTensor, pagedCache, gmCoord);
                } else {
                    copyKvGmToL1(dstTensor, valueGmTensor, gmCoord);
                }
            }
            SetFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            WaitFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            mL1SizeAlign = M_L1_SPLIT_SIZE;
            mL1Size = M_L1_SPLIT_SIZE; // m的实际大小
            for (uint32_t mL1 = 0; mL1 < mL1Loops; mL1++) {
                if (mL1 == (mL1Loops - 1)) {
                    // 尾块
                    mL1Size = mSize - (mL1Loops - 1) * M_L1_SPLIT_SIZE;
                    mL1SizeAlign = Align(mL1Size, 16U);
                }

                uint32_t mIdx = mBaseIdx + mL1;
                ka = GetQPL1RealIdx(mIdx, k1);
                LocalTensor<KV_T> aL1Tensor = l1QPTensor[ka * L1_BLOCK_OFFSET];
                if (nL1 == 0) {
                    WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]);
                    CopyInMm2AToL1(aL1Tensor, info, mSplitInfo.nBufferStartM + mL1 * M_L1_SPLIT_SIZE, mL1Size, kL1Size,
                                   256 * k1);
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
                }
                LocalTensor cL0Tensor =
                    cL0TensorPingPong[(cL0BufIter % 2) *
                                      (L0C_BLOCK_SIZE / sizeof(MM_OUT_T))]; // 需要保证cL0BufIter和m步调一致
                uint32_t baseK = 128;
                uint32_t baseN = 128;
                kL0Size = 128;
                kL0SizeAlign = kL0Size;

                Tile::MMParam param;
                param.singleM = mL1SizeAlign;
                param.singleN = nL1SizeAlign;
                param.singleK = kL1Size;
                param.isLeftTranspose = false;
                param.isRightTranspose = false;
                param.isOutKFisrt = (k1 == 0) ? true : false;
                param.unitFlag = (k1 == kL1Loops - 1) ? Tile::UNITFLAG_EN_OUTER_LAST : Tile::UNITFLAG_ENABLE;
                Tile::MatmulKPP<KV_T, KV_T, MM_OUT_T, M_L1_SPLIT_SIZE, N_L1_SPLIT_SIZE, MM2_K_L0_SPLIT_SIZE,
                                Tile::ABLayout::MK, Tile::ABLayout::KN>(aL1Tensor, bL1Tensor, mmL0ABuffers,
                                                                        mmL0BBuffers, cL0Tensor, param);

                if (nL1 == (nL1Loops - 1)) {
                    SetFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]); // 反向同步, 表示L1中的A已经被mte1消费完
                }

                if (k1 == (kL1Loops - 1)) {
                    if (nL1 == 0 && mL1 == 0) { // 第一次Fixpipe前等待
                        CrossCoreWaitFlag(constInfo.syncV1NupdateC2);
                    }

                    if (!info.isFirstSInnerLoop) {
                        SetAtomicAdd<MM_OUT_T>();
                    }
                    // ND
                    uint64_t mm2Offset =
                        (mSplitInfo.nBufferStartM + mL1 * M_L1_SPLIT_SIZE) * nSize + nL1 * N_L1_SPLIT_SIZE;
                    Tile::StoreFp32(
                        mm2ResGm[(info.bn2IdxInCurCore % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + mm2Offset],
                        cL0Tensor, mL1SizeAlign, nL1SizeAlign, nSize);
                    if (!info.isFirstSInnerLoop) {
                        SetAtomicNone();
                    }
                }

                if (mL1Loops == 2) {
                    cL0BufIter++;
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]);
        }

        if (mL1Loops == 1) {
            cL0BufIter++;
        }
    }

    qpL1BufIter += mL1Loops;
}

} // namespace flash_attn::mla::Block
