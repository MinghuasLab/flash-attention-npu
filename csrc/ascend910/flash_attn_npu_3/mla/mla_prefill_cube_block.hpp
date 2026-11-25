/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Source: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0,
// arch22/fia_block_cube_nonquant_gqa.h.
// 192/128 nonquant specialization: source Q2V2KP3 buffers, MMAD order,
// Q/V reuse, L0 transfers and cross-core events. Inputs use independent FA3
// strides; source split/combined RoPE both map to the full 192-wide row.
#pragma once
#include "mla_params.h"
#include "mla_metadata.h"
#include "mla_paged_copy_tile.hpp"
#include "mla_prefill_cube_ops_tile.hpp"
namespace flash_attn::mla::Block {
using namespace AscendC;
// Invocation coordinates belong to each ring entry: Vec1/Vec2 can still be
// consuming the previous batch while Cube starts the next one.
struct PrefillTaskContext {
    MlaPrefillTask task{};
    uint64_t qOffset = 0, kOffset = 0, vOffset = 0;
    uint64_t outputOffset = 0, lseOffset = 0, partialRow = 0;
    uint32_t cacheBatch = 0;
    bool partial = false;
};
struct PrefillInfo {
    PrefillTaskContext context{};
    uint32_t loop = 0, bIdx = 0, n2Idx = 0, gS1Idx = 0, s2Idx = 0;
    uint32_t actMBaseSize = 0, actualSingleProcessSInnerSize = 0, actualSingleProcessSInnerSizeAlign = 0;
    bool isFirstSInnerLoop = true, isLastS2Loop = true, isValid = false;
};
struct PrefillConstants {
    static constexpr uint32_t headDim = 128, headDimRope = 64, headDimAlign = 128;
    static constexpr uint32_t FIA_SYNC_MODE2 = 2, syncC1V1 = 7, syncV1C2 = 8, syncC2V2 = 9, preLoadNum = 2;
    uint32_t mmResUbSize, vec1ResUbSize, bmm2ResUbSize, s2BaseSize, gSize;
};
template <typename InputT, bool Paged = false>
class MlaPrefillCube {
    using T = float;
    using Q_T = InputT;
    using KV_T = InputT;
    using MM_OUT_T = float;
    using Axis = Tile::Prefill::Axis;
    template <typename T1, uint32_t N, uint64_t Stride = 0>
    using Array = Tile::Prefill::Array<T1, N, Stride>;
    struct BufSnapshot {
        uint64_t signature = uint64_t(-1);
        uint64_t base = uint64_t(-1);
        uint32_t batch = uint32_t(-1);
        uint32_t firstBufId = 0, bufCnt = 0;
    };

  public:
    __aicore__ inline void Init(const MlaPrefillParams& p, TPipe* pipe, GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                const MlaPagedCacheInputs* cache = nullptr)
    {
        InitParams({p.mBase * Tile::Align(p.nBase, 32), p.mBase * Tile::Align(p.nBase, 32), p.mBase * 128, p.nBase,
                    p.heads / p.kvHeads});
        queryGmTensor.Init(q, p.sq, p.heads / p.kvHeads, 192, &p.q);
        keyGmTensor.Init(k, p.sk, p.kvHeads, 192, &p.k);
        valueGmTensor.Init(v, p.sk, p.kvHeads, 128, &p.v);
        if constexpr (Paged)
            paged.Init(k, v, (GM_ADDR)cache->table.address, cache->layout);
        InitBuffers(pipe);
    }
    __aicore__ inline void InitParams(const PrefillConstants& constInfo);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<float>& mm1ResGm);
    __aicore__ inline void InitMm2GlobalTensor(const GlobalTensor<InputT>& vec1ResGm,
                                               const GlobalTensor<float>& mm2ResGm);
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const PrefillInfo& info);
    __aicore__ inline void ComputeMm2(const PrefillInfo& info);

  private:
    template <typename T1>
    static __aicore__ inline constexpr T1 AlignToPowerOfTwo(T1 s, T1 a)
    {
        return (s + a - 1) & (~(a - 1));
    }
    template <typename T1>
    __aicore__ inline constexpr uint32_t GetC0Num()
    {
        return 32 / sizeof(T1);
    }
    __aicore__ inline void CopyQToL1(uint32_t dstBufId, const PrefillInfo& info, uint32_t subMStart, uint32_t subMSize);
    __aicore__ inline void CopyKToL1(uint32_t dstBufId, const PrefillInfo& info, uint32_t subNStart, uint32_t subNSize);
    __aicore__ inline void CopyPToL1(uint32_t dstBufId, const PrefillInfo& info, uint32_t gmStride, uint32_t subMStart,
                                     uint32_t subMSize, uint32_t subKStart, uint32_t subKSize);
    __aicore__ inline void CopyVToL1(uint32_t dstBufId, const PrefillInfo& info, uint32_t subKStart, uint32_t subKSize);
    template <uint32_t M_L1_SIZE, bool FORCE = false>
    __aicore__ inline void ResetLoad3DConfig();
    template <uint32_t M_L1_SPLIT_Size, typename T1>
    __aicore__ inline void LoadAToL0(uint32_t dstBufId, const LocalTensor<T1>& l1Tensor, uint32_t mL1Size,
                                     uint32_t subMStart, uint32_t subMSize, uint32_t subKStart, uint32_t subKSize);
    __aicore__ inline void LoadBTransposeToL0(uint32_t dstBufId, const LocalTensor<InputT>& l1Tensor, uint32_t nL1Size,
                                              uint32_t subNStart, uint32_t subNSize, uint32_t subKStart,
                                              uint32_t subKSize);
    __aicore__ inline void LoadBToL0(uint32_t dstBufId, const LocalTensor<InputT>& l1Tensor, uint32_t kL1Size,
                                     uint32_t subKStart, uint32_t subKSize, uint32_t subNStart, uint32_t subNSize);
    __aicore__ inline void FixpipeCToGM(GlobalTensor<float>& mmResGm, uint32_t cL0BufId, uint32_t dstStride,
                                        uint32_t subMStart, uint32_t subMSize, uint32_t subNStart, uint32_t subNSize);
    // =================================L1 Buffer=================================
    static constexpr uint32_t L1_Q_SIZE = 256 * 192;  // 64KB or 96KB
    static constexpr uint32_t L1_V_SIZE = 256 * 128;  // 64KB
    static constexpr uint32_t L1_KP_SIZE = 128 * 256; // 64KB. K: 128 * 192 or 128 * 256; P: 128 * 256

    static constexpr uint32_t L1_Q_BUFCNT = 2;
    static constexpr uint32_t L1_V_BUFCNT = 2;
    static constexpr uint32_t L1_KP_BUFCNT = 3;

    Array<LocalTensor<Q_T>, L1_Q_BUFCNT, L1_Q_SIZE> qL1Tensor;
    Array<LocalTensor<KV_T>, L1_V_BUFCNT, L1_V_SIZE> vL1Tensor;
    Array<LocalTensor<KV_T>, L1_KP_BUFCNT, L1_KP_SIZE> kpL1Tensor;

    BufSnapshot qL1Snapshot;
    BufSnapshot vL1Snapshot;

    uint32_t qL1BufId = 0;
    uint32_t vL1BufId = 0;
    uint32_t kpL1BufId = 0;

    TBuf<TPosition::A1> L1_Q;
    TBuf<TPosition::A1> L1_V;
    TBuf<TPosition::A1> L1_KP;

    // =================================L0 Buffer=================================
    static constexpr uint32_t L0A_PP_SIZE = 128 * 128; // 32KB
    static constexpr uint32_t L0B_PP_SIZE = 128 * 128; // 32KB
    static constexpr uint32_t L0C_PP_SIZE = 128 * 128; // 64KB

    static constexpr uint32_t L0AB_BUFCNT = 2;
    static constexpr uint32_t L0C_BUFCNT = 2;

    Array<LocalTensor<Q_T>, L0AB_BUFCNT, L0A_PP_SIZE> aL0Tensor;
    Array<LocalTensor<KV_T>, L0AB_BUFCNT, L0B_PP_SIZE> bL0Tensor;
    Array<LocalTensor<T>, L0C_BUFCNT, L0C_PP_SIZE> cL0Tensor;

    uint32_t abL0BufId = 0;
    uint32_t cL0BufId = 0;

    TBuf<TPosition::A2> L0_A;
    TBuf<TPosition::B2> L0_B;
    TBuf<TPosition::CO1> L0_C;

    // =================================Event&Buffer ID===========================
    // mte2 <> mte1
    static constexpr uint32_t Q_EVENT0 = EVENT_ID0;
    static constexpr uint32_t V_EVENT0 = Q_EVENT0 + L1_Q_BUFCNT;
    static constexpr uint32_t KP_EVENT0 = V_EVENT0 + L1_V_BUFCNT;

    // m <> mte1
    static constexpr uint32_t L0AB_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0AB_EVENT1 = EVENT_ID4;
    static constexpr uint32_t L0_READY_EVENT = EVENT_ID7;

    // fix <> m
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID4;

    PrefillConstants constInfo{};
    Tile::DenseTensor<InputT, true> queryGmTensor, keyGmTensor, valueGmTensor;
    Tile::CopyQueryGmToL1<InputT, true> copyQueryGmToL1;
    Tile::CopyKvGmToL1<InputT, true> copyKvGmToL1;
    Tile::PagedCache<InputT> paged;
    template <bool Key>
    __aicore__ inline void CopyPaged(Tile::L1Tensor<InputT>& dst, const Tile::KvCoord& coord, uint32_t head)
    {
        const auto& layout = Key ? paged.layout.k : paged.layout.v;
        const auto& tensor = Key ? paged.k : paged.v;
        uint32_t done = 0;
        while (done < coord.s2DealSize) {
            const auto segment = paged.Segment(coord.batchIdx, coord.s2Idx + done, coord.s2DealSize - done);
            const uint64_t offset = (Key ? segment.kOffset : segment.vOffset) + uint64_t(head) * layout.headStride;
            Tile::CopySingleMatrixNDToNZ(dst.tensor[uint64_t(done) * 16], tensor[offset + coord.dIdx], segment.rowCount,
                                         coord.dDealSize, layout.rowStride, dst.rowCount);
            done += segment.rowCount;
        }
    }
    Array<GlobalTensor<float>, 2> mm1ResGm, mm2ResGm;
    Array<GlobalTensor<InputT>, 2> vec1ResGm;
    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {false, false};
    uint32_t load3DL1SizeCfg = 0;
    LoadData3DParamsV2<InputT> loadData3DParams;
    LoadData2DParams mm1LoadDataBTransposeToL0Params, mm2LoadDataBToL0Params;
};
template <typename InputT, bool Paged>
__aicore__ inline void MlaPrefillCube<InputT, Paged>::InitParams(const PrefillConstants& constInfo)
{
    this->constInfo = constInfo;

    loadData3DParams.l1W = GetC0Num<Q_T>(); // Win=M0
    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

    loadData3DParams.mStartPt = 0;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = 0;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = 0;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;

    mm1LoadDataBTransposeToL0Params.startIndex = 0;
    mm1LoadDataBTransposeToL0Params.srcStride = 1;
    mm1LoadDataBTransposeToL0Params.dstGap = 0;
    mm1LoadDataBTransposeToL0Params.ifTranspose = false;

    mm2LoadDataBToL0Params.startIndex = 0;
    mm2LoadDataBToL0Params.repeatTimes = constInfo.headDim / GetC0Num<Q_T>(); // 列（N轴）方向上的分形数
    mm2LoadDataBToL0Params.dstGap = 0;
    mm2LoadDataBToL0Params.ifTranspose = true; // 将小Z格式转置为小N
}

template <typename InputT, bool Paged>
__aicore__ inline void MlaPrefillCube<InputT, Paged>::InitMm1GlobalTensor(const GlobalTensor<MM_OUT_T>& mm1ResGm)
{
    // mm1
    this->mm1ResGm.Init(mm1ResGm, constInfo.mmResUbSize);
}

template <typename InputT, bool Paged>
__aicore__ inline void MlaPrefillCube<InputT, Paged>::InitMm2GlobalTensor(const GlobalTensor<KV_T>& vec1ResGm,
                                                                          const GlobalTensor<MM_OUT_T>& mm2ResGm)
{
    // mm2
    this->vec1ResGm.Init(vec1ResGm, constInfo.vec1ResUbSize);
    this->mm2ResGm.Init(mm2ResGm, constInfo.bmm2ResUbSize);
}

template <typename InputT, bool Paged>
__aicore__ inline void MlaPrefillCube<InputT, Paged>::InitBuffers(TPipe* pipe)
{
    // L1 Q
    pipe->InitBuffer(L1_Q, L1_Q_SIZE * L1_Q_BUFCNT * sizeof(Q_T));
    qL1Tensor.Init(L1_Q.Get<Q_T>());

    // L1 V
    pipe->InitBuffer(L1_V, L1_V_SIZE * L1_V_BUFCNT * sizeof(KV_T));
    vL1Tensor.Init(L1_V.Get<KV_T>());

    // L1 KP
    pipe->InitBuffer(L1_KP, L1_KP_SIZE * L1_KP_BUFCNT * sizeof(KV_T));
    kpL1Tensor.Init(L1_KP.Get<KV_T>());

    // L0A
    pipe->InitBuffer(L0_A, L0A_PP_SIZE * 2 * sizeof(Q_T));
    aL0Tensor.Init(L0_A.Get<Q_T>());

    // L0B
    pipe->InitBuffer(L0_B, L0B_PP_SIZE * 2 * sizeof(KV_T));
    bL0Tensor.Init(L0_B.Get<KV_T>());

    // L0C
    pipe->InitBuffer(L0_C, L0C_PP_SIZE * 2 * sizeof(MM_OUT_T));
    cL0Tensor.Init(L0_C.Get<T>());
}

template <typename InputT, bool Paged>
__aicore__ inline void MlaPrefillCube<InputT, Paged>::AllocEventID()
{
    for (uint32_t i = 0; i < L1_KP_BUFCNT; ++i) {
        SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + i);
    }

    for (uint32_t i = 0; i < L1_V_BUFCNT; ++i) {
        SetFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + i);
    }

    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT1);

    SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename InputT, bool Paged>
__aicore__ inline void MlaPrefillCube<InputT, Paged>::FreeEventID()
{
    for (uint32_t i = 0; i < L1_KP_BUFCNT; ++i) {
        WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + i);
    }

    for (uint32_t i = 0; i < L1_V_BUFCNT; ++i) {
        WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + i);
    }

    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT1);

    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename InputT, bool Paged>
template <uint32_t M_L1_SIZE, bool FORCE>
__aicore__ inline void MlaPrefillCube<InputT, Paged>::ResetLoad3DConfig()
{
    if constexpr (!FORCE) {
        if (this->load3DL1SizeCfg == M_L1_SIZE) {
            return;
        }
    }

    this->load3DL1SizeCfg = M_L1_SIZE;
    constexpr uint8_t padList[] = {0, 0, 0, 255};
    SetFmatrix(M_L1_SIZE / BLOCK_CUBE, GetC0Num<Q_T>(), padList, FmatrixMode::FMATRIX_LEFT);
    SetLoadDataPaddingValue<Q_T>(0);
}

template <typename InputT, bool Paged>
template <uint32_t M_L1_SPLIT_Size, typename T1>
__aicore__ inline void MlaPrefillCube<InputT, Paged>::LoadAToL0(uint32_t dstBufId, const LocalTensor<T1>& l1Tensor,
                                                                uint32_t mL1Size, uint32_t subMStart, uint32_t subMSize,
                                                                uint32_t subKStart, uint32_t subKSize)
{
    auto srcTensor = l1Tensor[mL1Size * subKStart][GetC0Num<T1>() * subMStart];
    auto dstTensor = this->aL0Tensor[dstBufId];
    loadData3DParams.mExtension = subMSize;
    loadData3DParams.kExtension = subKSize;
    loadData3DParams.channelSize = subKSize;

    if (likely(mL1Size == M_L1_SPLIT_Size)) {
        LoadData<T1, LOAD3DV2_CONFIG>(dstTensor, srcTensor, loadData3DParams);
    } else {
        loadData3DParams.l1H = mL1Size / BLOCK_CUBE;
        LoadData(dstTensor, srcTensor, loadData3DParams);

        ResetLoad3DConfig<M_L1_SPLIT_Size, true>();
    }
}

template <typename InputT, bool Paged>
__aicore__ inline void
MlaPrefillCube<InputT, Paged>::LoadBTransposeToL0(uint32_t dstBufId, const LocalTensor<KV_T>& l1Tensor,
                                                  uint32_t nL1Size, uint32_t subNStart, uint32_t subNSize,
                                                  uint32_t subKStart, uint32_t subKSize)
{
    auto srcTensor = l1Tensor[nL1Size * subKStart][GetC0Num<KV_T>() * subNStart];
    auto dstTensor = this->bL0Tensor[dstBufId];
    if (nL1Size == subNSize) {
        mm1LoadDataBTransposeToL0Params.repeatTimes =
            CeilDiv(subKSize, (uint32_t)BLOCK_CUBE) * CeilDiv(subNSize, GetC0Num<KV_T>());
        LoadData(dstTensor, srcTensor, mm1LoadDataBTransposeToL0Params);
    } else {
        mm1LoadDataBTransposeToL0Params.repeatTimes = CeilDiv(subNSize, (uint32_t)BLOCK_CUBE);
        uint32_t kLoops = subKSize / GetC0Num<KV_T>();
        for (uint32_t i = 0; i < kLoops; i++) {
            LoadData(dstTensor[subNSize * i * BLOCK_CUBE], srcTensor[nL1Size * i * GetC0Num<KV_T>()],
                     mm1LoadDataBTransposeToL0Params);
        }
    }
}

template <typename InputT, bool Paged>
__aicore__ inline void MlaPrefillCube<InputT, Paged>::LoadBToL0(uint32_t dstBufId, const LocalTensor<KV_T>& l1Tensor,
                                                                uint32_t kL1Size, uint32_t subKStart, uint32_t subKSize,
                                                                uint32_t subNStart, uint32_t subNSize)
{
    mm2LoadDataBToL0Params.srcStride = kL1Size / BLOCK_CUBE;

    auto srcTensor = l1Tensor[kL1Size * subNStart][GetC0Num<KV_T>() * subKStart];
    auto dstTensor = this->bL0Tensor[dstBufId];

    uint32_t kLoops = subKSize / BLOCK_CUBE;
    for (uint32_t i = 0; i < kLoops; i++) {
        LoadData(dstTensor[i * BLOCK_CUBE * subNSize], srcTensor[i * BLOCK_CUBE * GetC0Num<KV_T>()],
                 mm2LoadDataBToL0Params);
    }
}

template <typename InputT, bool Paged>
__aicore__ inline void MlaPrefillCube<InputT, Paged>::FixpipeCToGM(GlobalTensor<MM_OUT_T>& mmResGm, uint32_t cL0BufId,
                                                                   uint32_t dstStride, uint32_t subMStart,
                                                                   uint32_t subMSize, uint32_t subNStart,
                                                                   uint32_t subNSize)
{
    FixpipeParamsV220 fixParams;
    fixParams.nSize = subNSize;
    fixParams.mSize = subMSize;
    fixParams.srcStride = AlignToPowerOfTwo(subMSize, (uint32_t)BLOCK_CUBE);
    fixParams.ndNum = 1;

    auto cL0Tensor = this->cL0Tensor[cL0BufId];

    fixParams.dstStride = dstStride;
    auto dstTensor = mmResGm[dstStride * subMStart + subNStart];
    Fixpipe<MM_OUT_T, T, CFG_ROW_MAJOR>(dstTensor, cL0Tensor, fixParams);
}

template <typename InputT, bool Paged>
__aicore__ inline void MlaPrefillCube<InputT, Paged>::ComputeMm1(const PrefillInfo& info)
{

    Axis m(info.actMBaseSize);
    Axis n(info.actualSingleProcessSInnerSize);
    Axis k(constInfo.headDim + constInfo.headDimRope);
    auto mmResGm = this->mm1ResGm[info.loop % 2];

    constexpr uint32_t M_SPLIT_SIZE = 256;
    const uint32_t N_SPLIT_SIZE = (k.sizeAct <= 128) ? 256 : 128;

    constexpr uint32_t M_BASE = 128;
    constexpr uint32_t K_BASE = 128;
    constexpr uint32_t N_BASE = 128;

    ResetLoad3DConfig<M_SPLIT_SIZE>();

    auto mSlices = m.Split(M_SPLIT_SIZE);
    auto kL0Slices = k.Split(K_BASE);

    bool canFullLoadQ = (mSlices.size() <= L1_Q_BUFCNT);
    uint64_t qCoord = ((uint64_t)info.n2Idx << 32) | ((uint64_t)info.gS1Idx);
    bool reuseQBuf = canFullLoadQ && qL1Snapshot.signature == qCoord && qL1Snapshot.base == info.context.qOffset &&
                     qL1Snapshot.batch == info.context.cacheBatch;
    if (!reuseQBuf) {
        qL1Snapshot.bufCnt = 0;
        qL1Snapshot.firstBufId = this->qL1BufId;
        qL1Snapshot.signature = qCoord;
        qL1Snapshot.base = info.context.qOffset;
        qL1Snapshot.batch = info.context.cacheBatch;
    }

    for (auto& nL1 : n.Split(N_SPLIT_SIZE)) {
        WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + this->kpL1BufId);
        CopyKToL1(this->kpL1BufId, info, nL1.start, nL1.sizeAct);

        SetFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + this->kpL1BufId);
        WaitFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + this->kpL1BufId);

        int32_t mL1Id = -1;
        for (auto& mL1 : mSlices) {
            ++mL1Id;

            if (unlikely(mL1Id >= qL1Snapshot.bufCnt)) {
                reuseQBuf = false;
            }

            uint32_t qBufId;
            if (unlikely(!reuseQBuf)) {
                qBufId = this->qL1BufId;
                // 在需要搬入Q前才去Set MTE1->MTE2事件，而不是在L0算完后就去Set，是考虑到Q_L1
                // buf的生命周期可能跨越多轮MM1计算， 如果前一次MM1计算还未完成，还在复用Q_L1
                // buf，后一次MM1计算就开始搬运，就会覆盖了前一次计算的数据
                SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + this->qL1BufId);
                WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qBufId);
                CopyQToL1(qBufId, info, mL1.start, mL1.sizeAct);

                SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qBufId);
                WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qBufId);
                qL1Snapshot.bufCnt += static_cast<uint32_t>(canFullLoadQ); // 不能全载Q时，不缓存到快照中
            } else {
                qBufId = (qL1Snapshot.firstBufId + mL1Id) % L1_Q_BUFCNT;
            }

            auto nL0Slices = nL1.Split(N_BASE);
            for (auto& mL0 : mL1.Split(M_BASE)) {
                for (auto& nL0 : nL0Slices) {
                    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + this->cL0BufId);

                    for (auto& kL0 : kL0Slices) {
                        WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + this->abL0BufId);

                        LoadAToL0<M_SPLIT_SIZE>(this->abL0BufId, qL1Tensor[qBufId], mL1.AlignedSize(), mL0.start,
                                                mL0.AlignedSize(), kL0.start, kL0.AlignedSize());
                        LoadBTransposeToL0(this->abL0BufId, kpL1Tensor[this->kpL1BufId], nL1.AlignedSize(), nL0.start,
                                           nL0.AlignedSize(), kL0.start, kL0.AlignedSize());

                        SetFlag<HardEvent::MTE1_M>(L0_READY_EVENT);
                        WaitFlag<HardEvent::MTE1_M>(L0_READY_EVENT);

                        bool isLastK = kL0.IsTailOf(k);
                        MmadParams mmadParams;
                        mmadParams.m = mL0.AlignedSize();
                        mmadParams.n = nL0.sizeAct;
                        mmadParams.k = kL0.sizeAct;
                        mmadParams.cmatrixInitVal = (kL0.start == 0);
                        mmadParams.cmatrixSource = false;

                        Mmad(cL0Tensor[this->cL0BufId], aL0Tensor[this->abL0BufId], bL0Tensor[this->abL0BufId],
                             mmadParams);
                        if (likely(!isLastK)) {
                            AscendC::PipeBarrier<PIPE_M>();
                        }
                        SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + this->abL0BufId);
                        this->abL0BufId = (this->abL0BufId + 1) % L0AB_BUFCNT;
                    }

                    SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + this->cL0BufId);
                    WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + this->cL0BufId);

                    FixpipeCToGM(mmResGm, this->cL0BufId, info.actualSingleProcessSInnerSizeAlign,
                                 mL1.start + mL0.start, mL0.sizeAct, nL1.start + nL0.start, nL0.sizeAct);
                    SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + this->cL0BufId);
                    this->cL0BufId = (this->cL0BufId + 1) % L0C_BUFCNT;
                }
            }
            if (unlikely(!reuseQBuf)) {
                this->qL1BufId = (this->qL1BufId + 1) % L1_Q_BUFCNT;
                reuseQBuf = canFullLoadQ;
            }
        }
        SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + this->kpL1BufId);
        ++this->kpL1BufId;
        if (this->kpL1BufId >= L1_KP_BUFCNT) {
            this->kpL1BufId = 0;
        }
    }
    CrossCoreSetFlag<PrefillConstants::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
}

template <typename InputT, bool Paged>
__aicore__ inline void MlaPrefillCube<InputT, Paged>::ComputeMm2(const PrefillInfo& info)
{

    CrossCoreWaitFlag(constInfo.syncV1C2);
    constexpr uint32_t M_SPLIT_SIZE = 128;
    constexpr uint32_t K_SPLIT_SIZE = 256;
    constexpr uint32_t K_BASE = 128;

    Axis m(info.actMBaseSize);
    Axis n(constInfo.headDim);
    Axis k(info.actualSingleProcessSInnerSize);
    auto mmResGm = this->mm2ResGm[info.loop % 2];

    ResetLoad3DConfig<M_SPLIT_SIZE>();

    auto KL1Slices = k.Split(K_SPLIT_SIZE);

    bool canFullLoadV = (KL1Slices.size() <= L1_V_BUFCNT);
    uint64_t vCoord = ((uint64_t)info.bIdx << 48) | ((uint64_t)info.n2Idx << 32) | ((uint64_t)info.s2Idx);
    bool reuseVBuf = false;
    vL1Snapshot.bufCnt = 0;
    vL1Snapshot.firstBufId = this->vL1BufId;
    vL1Snapshot.signature = vCoord;

    for (auto& mL1 : m.Split(M_SPLIT_SIZE)) {
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + this->cL0BufId);

        int32_t kL1Id = -1;
        for (auto& kL1 : KL1Slices) {
            ++kL1Id;
            WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + this->kpL1BufId);
            CopyPToL1(this->kpL1BufId, info,
                      info.actualSingleProcessSInnerSizeAlign, /* P为ND时，每行元素个数会按32对齐 */
                      mL1.start, mL1.sizeAct, kL1.start, kL1.sizeAct);

            SetFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + this->kpL1BufId);
            WaitFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + this->kpL1BufId);

            if (unlikely(kL1Id >= vL1Snapshot.bufCnt)) {
                reuseVBuf = false;
            }

            uint32_t vBufId;
            if (unlikely(!reuseVBuf)) {
                vBufId = this->vL1BufId;
                WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + vBufId);
                CopyVToL1(vBufId, info, kL1.start, kL1.sizeAct);

                SetFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + vBufId);
                WaitFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + vBufId);
                vL1Snapshot.bufCnt += static_cast<uint32_t>(canFullLoadV); // 不能全载V时，不缓存到快照中
            } else {
                vBufId = (vL1Snapshot.firstBufId + kL1Id) % L1_V_BUFCNT;
            }

            for (auto& kL0 : kL1.Split(K_BASE)) {
                WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + this->abL0BufId);

                LoadAToL0<M_SPLIT_SIZE>(this->abL0BufId, kpL1Tensor[this->kpL1BufId], mL1.AlignedSize(), 0,
                                        mL1.AlignedSize(), kL0.start, kL0.AlignedSize());
                LoadBToL0(this->abL0BufId, vL1Tensor[vBufId], kL1.AlignedSize(), kL0.start, kL0.AlignedSize(), 0,
                          n.AlignedSize());
                SetFlag<HardEvent::MTE1_M>(L0_READY_EVENT);
                WaitFlag<HardEvent::MTE1_M>(L0_READY_EVENT);

                bool isLastK = ((k.sizeAct - kL1.start - kL0.start) <= K_BASE);
                MmadParams mmadParams;
                mmadParams.m = mL1.AlignedSize();
                mmadParams.n = n.sizeAct;
                mmadParams.k = kL0.sizeAct;
                mmadParams.cmatrixInitVal = (kL1.start == 0) && (kL0.start == 0);
                mmadParams.cmatrixSource = false;

                Mmad(cL0Tensor[this->cL0BufId], aL0Tensor[this->abL0BufId], bL0Tensor[this->abL0BufId], mmadParams);
                if (likely(!isLastK)) {
                    AscendC::PipeBarrier<PIPE_M>();
                }
                SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + this->abL0BufId);
                this->abL0BufId = (this->abL0BufId + 1) % L0AB_BUFCNT;
            }
            SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + this->kpL1BufId);
            ++this->kpL1BufId;
            if (this->kpL1BufId >= L1_KP_BUFCNT) {
                this->kpL1BufId = 0;
            }

            if (unlikely(!canFullLoadV || mL1.IsTailOf(m))) {
                // 当可以复用V_L1 buf时，它的生命周期跨越整个mL1的迭代，在mL1迭代完成释放
                SetFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + vBufId);
            }

            if (unlikely(!reuseVBuf)) {
                this->vL1BufId = (this->vL1BufId + 1) % L1_V_BUFCNT;
                reuseVBuf = canFullLoadV;
            }
        }

        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + this->cL0BufId);
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + this->cL0BufId);

        FixpipeCToGM(mmResGm, this->cL0BufId,
                     constInfo.headDimAlign, /* 输出ND时，mm2ResGm两行之间间隔元素个数按32对齐 */
                     mL1.start, mL1.sizeAct, 0, n.sizeAct);
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + this->cL0BufId);
        this->cL0BufId = (this->cL0BufId + 1) % L0C_BUFCNT;
    }
    CrossCoreSetFlag<PrefillConstants::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);
}

template <typename InputT, bool Paged>
__aicore__ inline void MlaPrefillCube<InputT, Paged>::CopyQToL1(uint32_t slot, const PrefillInfo& info, uint32_t row,
                                                                uint32_t count)
{
    auto src = queryGmTensor;
    src.gmTensor = src.gmTensor[info.context.qOffset + uint64_t(info.n2Idx) * constInfo.gSize * src.HeadStride()];
    Tile::L1Tensor<InputT> dst{qL1Tensor[slot], Tile::Align(count, 16)};
    Tile::QueryCoord coord{0, 0, info.gS1Idx + row, 0, count, 192, info.bIdx};
    copyQueryGmToL1(dst, src, coord);
}
template <typename InputT, bool Paged>
__aicore__ inline void MlaPrefillCube<InputT, Paged>::CopyKToL1(uint32_t slot, const PrefillInfo& info, uint32_t col,
                                                                uint32_t count)
{
    auto src = keyGmTensor;
    src.gmTensor = src.gmTensor[info.context.kOffset + uint64_t(info.n2Idx) * src.HeadStride()];
    Tile::L1Tensor<InputT> dst{kpL1Tensor[slot], Tile::Align(count, 16)};
    Tile::KvCoord coord{0, 0, info.s2Idx * constInfo.s2BaseSize + col, 0, count, 192, info.bIdx};
    if constexpr (Paged) {
        coord.batchIdx = info.context.cacheBatch;
        CopyPaged<true>(dst, coord, info.n2Idx);
    } else
        copyKvGmToL1(dst, src, coord);
}
template <typename InputT, bool Paged>
__aicore__ inline void MlaPrefillCube<InputT, Paged>::CopyVToL1(uint32_t slot, const PrefillInfo& info, uint32_t col,
                                                                uint32_t count)
{
    auto src = valueGmTensor;
    src.gmTensor = src.gmTensor[info.context.vOffset + uint64_t(info.n2Idx) * src.HeadStride()];
    Tile::L1Tensor<InputT> dst{vL1Tensor[slot], Tile::Align(count, 16)};
    Tile::KvCoord coord{0, 0, info.s2Idx * constInfo.s2BaseSize + col, 0, count, 128, info.bIdx};
    if constexpr (Paged) {
        coord.batchIdx = info.context.cacheBatch;
        CopyPaged<false>(dst, coord, info.n2Idx);
    } else
        copyKvGmToL1(dst, src, coord);
}
template <typename InputT, bool Paged>
__aicore__ inline void MlaPrefillCube<InputT, Paged>::CopyPToL1(uint32_t slot, const PrefillInfo& info, uint32_t stride,
                                                                uint32_t row, uint32_t rows, uint32_t col,
                                                                uint32_t cols)
{
    Tile::CopySingleMatrixNDToNZ(kpL1Tensor[slot], vec1ResGm[info.loop % 2][stride * row + col], rows, cols, stride,
                                 Tile::Align(rows, 16));
}
} // namespace flash_attn::mla::Block
