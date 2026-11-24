// Copyright (c) 2026, Minghua Shen.

#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_SINK_HPP
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_SINK_HPP

#include "catlass/arch/resource.hpp"
#include "fag_ub_layout.hpp"
#include "kernel_operator.h"

#ifdef __DAV_VEC__
namespace Catlass::Epilogue::Block {

class FagSinkGrad {
  public:
    CATLASS_DEVICE bool Enabled() const { return enabled_; }

    CATLASS_DEVICE void Init(Arch::Resource<Arch::Ascend950>& resource, AscendC::LocalTensor<float> scratch,
                             GM_ADDR sink, GM_ADDR dsink, GM_ADDR workspace,
                             uint32_t vectorCore, uint32_t vectorCores, uint64_t heads)
    {
        enabled_ = true;
        sink_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sink));
        dsink_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dsink));
        partials_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        scratch_ = scratch;
        work_ = resource.ubBuf.template GetBufferByByte<float>(0);
        vectorCore_ = vectorCore;
        vectorCores_ = vectorCores;
        heads_ = heads;
        scratchReady_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_V>());
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(scratchReady_);

        // Each AIV clears one contiguous segment of the packed partial table.
        constexpr uint32_t chunk = 256;
        AscendC::Duplicate(work_, 0.0f, chunk);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint64_t offset = 0; offset < heads_; offset += chunk) {
            const uint32_t count = heads_ - offset < chunk ? heads_ - offset : chunk;
            const AscendC::DataCopyExtParams copy{1, count * 4U, 0, 0, 0};
            AscendC::DataCopyPad(partials_[vectorCore_ * heads_ + offset], work_, copy);
        }
        // The prepass reuses the same UB at offset zero.
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE void Accumulate(AscendC::LocalTensor<float> p, AscendC::LocalTensor<float> dp,
                                   AscendC::LocalTensor<float> lse, uint32_t rows, uint32_t columns,
                                   uint64_t head, event_t vectorToStore)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(scratchReady_);
        TileContribution(Ptr(scratch_), Ptr(p), Ptr(dp), Ptr(lse), rows, columns, sink_.GetValue(head));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vectorToStore);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vectorToStore);
        // No two cores target the same FP32 slot. Atomic mode adds subsequent
        // tiles from this AIV in issue order without a GM-to-UB round trip.
        const AscendC::DataCopyExtParams copy{1, 4, 0, 0, 0};
        AscendC::SetAtomicType<float>();
        AscendC::DataCopyPad(partials_[head * vectorCores_ + vectorCore_], scratch_, copy);
        AscendC::SetAtomicNone();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(scratchReady_);
    }

    CATLASS_DEVICE void Drain()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(scratchReady_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE void Finish(event_t loadToVector, event_t vectorToStore)
    {
        for (uint64_t head = vectorCore_; head < heads_; head += vectorCores_) {
            const AscendC::DataCopyExtParams load{1, vectorCores_ * 4U, 0, 0, 0};
            auto pad = AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0};
            AscendC::DataCopyPad(work_, partials_[head * vectorCores_], load, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(loadToVector);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(loadToVector);
            SumPartials(Ptr(scratch_), Ptr(work_), vectorCores_);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vectorToStore);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vectorToStore);
            const AscendC::DataCopyExtParams store{1, 4, 0, 0, 0};
            AscendC::DataCopyPad(dsink_[head], scratch_, store);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

  private:
    static constexpr uint32_t LANES = 64;

    CATLASS_DEVICE static __ubuf__ float* Ptr(const AscendC::LocalTensor<float>& tensor)
    {
        return reinterpret_cast<__ubuf__ float*>(tensor.GetPhyAddr());
    }

    __simd_vf__ static inline void TileContribution(__ubuf__ float* dst, __ubuf__ float* p, __ubuf__ float* dp,
                                                    __ubuf__ float* lse, uint32_t rows, uint32_t columns, float sink)
    {
        using namespace AscendC::MicroAPI;
        auto all = CreateMask<float, MaskPattern::ALL>();
        uint32_t cols0 = columns < LANES ? columns : LANES;
        uint32_t cols1 = columns - cols0;
        auto mask0 = UpdateMask<float>(cols0);
        auto mask1 = UpdateMask<float>(cols1);
        RegTensor<float> acc, zero, bias, rowLse, sinkProbability, p0, p1, dp0, dp1, row, reduced;
        Duplicate(acc, 0.0f);
        Duplicate(zero, 0.0f);
        Duplicate(bias, sink);
        for (uint32_t i = 0; i < rows; ++i) {
            const uint32_t offset = i * FagUbLayout::ROW_STRIDE;
            LoadAlign(p0, p + offset);
            LoadAlign(p1, p + offset + LANES);
            LoadAlign(dp0, dp + offset);
            LoadAlign(dp1, dp + offset + LANES);
            Select(dp0, dp0, zero, mask0);
            Select(dp1, dp1, zero, mask1);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rowLse, lse + i);
            ExpSub(sinkProbability, bias, rowLse, all);
            Mul(p0, p0, dp0, all);
            Mul(p1, p1, dp1, all);
            Add(row, p0, p1, all);
            Mul(row, row, sinkProbability, all);
            Add(acc, acc, row, all);
        }
        Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(reduced, acc, all);
        Muls(reduced, reduced, -1.0f, all);
        UnalignReg state;
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, reduced, state, 1);
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(dst, state, 0);
    }

    __simd_vf__ static inline void SumPartials(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    {
        using namespace AscendC::MicroAPI;
        auto all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> acc, term, reduced;
        Duplicate(acc, 0.0f);
        for (uint32_t offset = 0; offset < count; offset += LANES) {
            uint32_t remaining = count - offset;
            auto mask = UpdateMask<float>(remaining);
            LoadAlign(term, src + offset);
            Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(reduced, term, mask);
            Add(acc, acc, reduced, all);
        }
        UnalignReg state;
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, acc, state, 1);
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(dst, state, 0);
    }

    bool enabled_ = false;
    uint32_t vectorCore_ = 0;
    uint32_t vectorCores_ = 0;
    uint64_t heads_ = 0;
    event_t scratchReady_{};
    AscendC::LocalTensor<float> scratch_, work_;
    AscendC::GlobalTensor<float> sink_, dsink_, partials_;
};

} // namespace Catlass::Epilogue::Block
#endif
#endif
