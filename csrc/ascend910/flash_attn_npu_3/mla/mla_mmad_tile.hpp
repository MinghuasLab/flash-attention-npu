/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0.
// attention/common/op_kernel/matmul.h::LoadDataToL0A/B, MatmulKPP.
// Dense FP16/BF16 specialization; retain the source L0 loop and paired A/B lifetime.
#pragma once

#include "catlass/arch/arch.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "mla_copy_tile.hpp"
namespace flash_attn::mla::Tile {
using namespace AscendC;

// Source buffers_policy.h / attn_buffer.h: two 32 KiB slots for each operand.
// B's event allocations and initial tokens are retained even though MatmulKPP
// synchronizes the pair through A. TPipe remains the sole resource owner.
template <TPosition Position>
class L0Buffers {
  public:
    struct Buffer {
        LocalTensor<half> tensor;
        TEventID ready;
        TEventID released;
        template <HardEvent Event>
        __aicore__ inline void Wait() const
        {
            WaitFlag<Event>(Event == HardEvent::M_MTE1 ? released : ready);
        }
        template <HardEvent Event>
        __aicore__ inline void Set() const
        {
            SetFlag<Event>(Event == HardEvent::M_MTE1 ? released : ready);
        }
        template <typename T>
        __aicore__ inline LocalTensor<T> GetTensor() const
        {
            return tensor.template ReinterpretCast<T>();
        }
    };
    __aicore__ inline void Init(TPipe* pipe)
    {
        pipe->InitBuffer(storage, 2 * 32768);
        for (uint32_t i = 0; i < 2; ++i) {
            slots[i].tensor = storage.template Get<half>()[i * 16384];
            slots[i].ready = pipe->AllocEventID<HardEvent::MTE1_M>();
            slots[i].released = pipe->AllocEventID<HardEvent::M_MTE1>();
            slots[i].template Set<HardEvent::M_MTE1>();
        }
    }
    __aicore__ inline Buffer Get()
    {
        Buffer result = slots[next];
        next ^= 1;
        return result;
    }
    __aicore__ inline void Uninit(TPipe* pipe)
    {
        for (uint32_t i = 0; i < 2; ++i) {
            slots[i].template Wait<HardEvent::M_MTE1>();
            pipe->ReleaseEventID<HardEvent::MTE1_M>(slots[i].ready);
            pipe->ReleaseEventID<HardEvent::M_MTE1>(slots[i].released);
        }
    }

  private:
    TBuf<Position> storage;
    Buffer slots[2];
    uint32_t next = 0;
};

constexpr uint32_t UNITFLAG_ENABLE = 2;
constexpr uint32_t UNITFLAG_EN_OUTER_LAST = 3;
constexpr uint32_t FP16_ONE_FRACTAL_ELEMENT = 16;
constexpr uint32_t ONE_FRACTAL_H_ELEMENT = 16;
constexpr uint32_t ONE_FRACTAL_W_BYTE = 32;
constexpr uint32_t LOAD3D_L1W_SIZE = 16;
constexpr uint8_t LOAD3D_STRIDE_W = 1, LOAD3D_STRIDE_H = 1;
constexpr uint8_t LOAD3D_FILTER_W = 1, LOAD3D_FILTER_H = 1;
constexpr uint8_t LOAD3D_DILA_FILTER_W = 1, LOAD3D_DILA_FILTER_H = 1;
static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};
enum class ABLayout {
    MK,
    KM,
    KN,
    NK
};
struct MMParam {
    uint32_t singleM, singleN, singleK;
    bool isLeftTranspose, isRightTranspose, isOutKFisrt;
    uint32_t unitFlag;
};
template <typename T, ABLayout AL>
__aicore__ inline void LoadDataToL0A(LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor, const MMParam& mmParam,
                                     uint64_t L1Aoffset, uint32_t kSplitSize, uint32_t mSplitSize)
{
    if constexpr (AL == ABLayout::MK) {
        LoadData3DParamsV2<T> loadData3DParams;
        loadData3DParams.l1H = mSplitSize / LOAD3D_L1W_SIZE; // 源操作数height
        loadData3DParams.l1W = LOAD3D_L1W_SIZE;              // 源操作数weight
        loadData3DParams.padList[0] = 0;
        loadData3DParams.padList[1] = 0;
        loadData3DParams.padList[2] = 0;
        loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

        loadData3DParams.mExtension = mSplitSize; // 在目的操作数height维度的传输长度
        loadData3DParams.kExtension = kSplitSize; // 在目的操作数width维度的传输长度
        loadData3DParams.mStartPt = 0;            // 卷积核在目的操作数width维度的起点
        loadData3DParams.kStartPt = 0;            // 卷积核在目的操作数height维度的起点
        loadData3DParams.strideW = 1;             // 卷积核在源操作数width维度滑动的步长
        loadData3DParams.strideH = 1;             // 卷积核在源操作数height维度滑动的步长
        loadData3DParams.filterW = 1;             // 卷积核width
        loadData3DParams.filterSizeW = false;     // 是否在filterW的基础上将卷积核width增加256个元素
        loadData3DParams.filterH = 1;             // 卷积核height
        loadData3DParams.filterSizeH = false;     // 是否在filterH的基础上将卷积核height增加256个元素
        loadData3DParams.dilationFilterW = 1;     // 卷积核width膨胀系数
        loadData3DParams.dilationFilterH = 1;     // 卷积核height膨胀系数
        loadData3DParams.enTranspose = 0;         // 是否启用转置功能，对整个目标矩阵进行转置
        loadData3DParams.fMatrixCtrl = 0;
        loadData3DParams.channelSize =
            kSplitSize; // 源操作数的通道数。膨胀系数为1时，目的weight为filterW*filterH*channelSize
        LoadData<T, LOAD3DV2_CONFIG>(aL0Tensor, aL1Tensor[L1Aoffset], loadData3DParams);
    } else if constexpr (AL == ABLayout::KM) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0; // 分型矩阵ID，表明搬运起始位置为源操作数中第0个分型
        loadData2DParams.repeatTimes =
            (kSplitSize / ONE_FRACTAL_H_ELEMENT) *
            (mmParam.singleM / (ONE_FRACTAL_W_BYTE / sizeof(T))); // 迭代次数，每个迭代可以处理512B数据
        loadData2DParams.srcStride = 1; // 相邻迭代间，源操作数前一个分型和后一个分型起始地址的间隔（单位512B）
        loadData2DParams.dstGap = 0; // 相邻迭代间，目的操作数前一个分型的结束地址和后一个分型起始地址的间隔（单位512B）
        loadData2DParams.ifTranspose = true;
        LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParams);
    }
}

// L1→L0B + 切K/切N/全载
template <typename T, ABLayout BL>
__aicore__ inline void LoadDataToL0B(LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor, const MMParam& mmParam,
                                     uint64_t L1Boffset, uint32_t kSplitSize, uint32_t nSplitSize)
{
    if constexpr (BL == ABLayout::KN) {
        LoadData3DParamsV2<T> loadData3DParams;
        loadData3DParams.l1H = kSplitSize / LOAD3D_L1W_SIZE; // 源操作数height
        loadData3DParams.l1W = LOAD3D_L1W_SIZE;              // 源操作数weight=16，目的height=l1H*L1W
        loadData3DParams.padList[0] = 0;
        loadData3DParams.padList[1] = 0;
        loadData3DParams.padList[2] = 0;
        loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

        loadData3DParams.mExtension = kSplitSize; // 在目的操作数height维度的传输长度
        loadData3DParams.kExtension = nSplitSize; // 在目的操作数width维度的传输长度
        loadData3DParams.mStartPt = 0;            // 卷积核在目的操作数width维度的起点
        loadData3DParams.kStartPt = 0;            // 卷积核在目的操作数height维度的起点
        loadData3DParams.strideW = LOAD3D_STRIDE_W;
        loadData3DParams.strideH = LOAD3D_STRIDE_H;
        loadData3DParams.filterW = LOAD3D_FILTER_W;
        loadData3DParams.filterSizeW = false; // 是否在filterW的基础上将卷积核width增加256个元素
        loadData3DParams.filterH = LOAD3D_FILTER_H;
        loadData3DParams.filterSizeH = false;                    // 是否在filterH的基础上将卷积核height增加256个元素
        loadData3DParams.dilationFilterW = LOAD3D_DILA_FILTER_W; // 卷积核width膨胀系数
        loadData3DParams.dilationFilterH = LOAD3D_DILA_FILTER_H; // 卷积核height膨胀系数
        loadData3DParams.enTranspose = 1;                        // 是否启用转置功能
        loadData3DParams.fMatrixCtrl =
            0; // 使用FMATRIX_LEFT还是使用FMATRIX_RIGHT，=0使用FMATRIX_LEFT，=1使用FMATRIX_RIGHT 1
        loadData3DParams.channelSize =
            nSplitSize; // 源操作数的通道数。膨胀系数为1时，目的weight为filterW*filterH*channelSize
        LoadData<T, LOAD3DV2_CONFIG>(bL0Tensor, bL1Tensor[L1Boffset], loadData3DParams);
    } else if constexpr (BL == ABLayout::NK) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes =
            (nSplitSize + (ONE_FRACTAL_H_ELEMENT - 1)) / ONE_FRACTAL_H_ELEMENT *
            (kSplitSize / (ONE_FRACTAL_W_BYTE / sizeof(T))); // 迭代次数，每个迭代可以处理512B数据
        loadData2DParams.srcStride = 1;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = false;
        LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParams);
    }
}

template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL>
__aicore__ inline void MatmulKPP(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor,
                                 L0Buffers<TPosition::A2>& aL0BuffsDb, L0Buffers<TPosition::B2>& bL0BuffsDb,
                                 const LocalTensor<C>& cL0Tensor, const MMParam& param)
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t kSplitSize = (kLoops == 1) ? param.singleK : baseK;
    uint32_t kSplitSizeAlign = Align(kSplitSize, FP16_ONE_FRACTAL_ELEMENT);
    uint64_t L1Aoffset = Align(param.singleM, FP16_ONE_FRACTAL_ELEMENT) * kSplitSize;
    uint64_t L1Boffset = Align(param.singleN, FP16_ONE_FRACTAL_ELEMENT) * kSplitSize;
    for (uint32_t k = 0; k < kLoops; k++) {
        if (k == kLoops - 1) {
            kSplitSize = (param.singleK % baseK) ? (param.singleK % baseK) : kSplitSize;
            kSplitSizeAlign = Align(kSplitSize, FP16_ONE_FRACTAL_ELEMENT);
        }
        auto l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul:上一轮matmul完成后才能搬运新数据到L0A
        LocalTensor<A> L0ATensor = l0aBuffer.GetTensor<A>();
        LoadDataToL0A<A, AL>(L0ATensor, aL1Tensor, param, k * L1Aoffset, kSplitSizeAlign, param.singleM);
        auto l0bBuffer = bL0BuffsDb.Get();
        LocalTensor<B> L0BTensor = l0bBuffer.GetTensor<B>();
        LoadDataToL0B<B, BL>(L0BTensor, bL1Tensor, param, k * L1Boffset, kSplitSizeAlign, param.singleN);

        l0aBuffer.Set<HardEvent::MTE1_M>();  // mte1搬运完后，通知可以开始matmul
        l0aBuffer.Wait<HardEvent::MTE1_M>(); // matmul等mte1：L0A数据搬运完成后才能开始matmul

        using AType = Catlass::Gemm::GemmType<A, Catlass::layout::RowMajor>;
        using BType = Catlass::Gemm::GemmType<B, Catlass::layout::RowMajor>;
        Catlass::Gemm::Tile::TileMmad<Catlass::Arch::AtlasA2, AType, BType, void> mma;
        uint8_t unitFlag =
            param.unitFlag == 0 ? 0 : ((param.unitFlag == UNITFLAG_EN_OUTER_LAST && k == kLoops - 1) ? 3 : 2);
        // Callers pass M/N aligned to 16, matching source's GEMV avoidance.
        // CATLASS also issues the source's PIPE_M barrier for <10 fractals.
        mma(cL0Tensor, L0ATensor, L0BTensor, param.singleM, param.singleN, kSplitSize, param.isOutKFisrt && k == 0,
            unitFlag);
        l0aBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0A
    }
}
} // namespace flash_attn::mla::Tile
