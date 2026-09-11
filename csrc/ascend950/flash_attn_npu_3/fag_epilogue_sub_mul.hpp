/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend950 FlashAttention v3 backward dS epilogue.
 */

#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_SUB_MUL_HPP
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_SUB_MUL_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"

#include "fag_block.h"

namespace Catlass::Epilogue::Block {

/**
 * V2 dS implementation hook.
 *
 * Call chain:
 *   FlashAttentionV3Bwd950
 *     -> FlashAttentionScoreGrad950
 *     -> ProcessV2Stage
 *     -> this BlockEpilogue::operator()
 *
 * Intended data path:
 *   C2 dP UB + V1 P UB + delta GM
 *     -> dS = P * (dP - delta) * optional(1 - tanh(x)^2) -> dS L1.
 *
 * Only the interface and type wiring are provided for now. ProcessV2Stage
 * owns the cross-core ready/return handshake around this implementation.
 */
template <
    FAGTiling950::Layout INPUT_LAYOUT_,
    bool IS_SOFTCAP_,
    class ElementDS_,
    class ElementP_,
    class ElementDP_,
    class TilingData_>
class BlockEpilogue<
    EpilogueAscend950FAGSubMul<INPUT_LAYOUT_, IS_SOFTCAP_>,
    ElementDS_,
    ElementP_,
    ElementDP_,
    TilingData_> {
public:
    using DispatchPolicy =
        EpilogueAscend950FAGSubMul<INPUT_LAYOUT_, IS_SOFTCAP_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementDS = ElementDS_;
    using ElementP = ElementP_;
    using ElementDP = ElementDP_;
    using TilingData = TilingData_;

    static constexpr FAGTiling950::Layout INPUT_LAYOUT = INPUT_LAYOUT_;
    static constexpr bool IS_SOFTCAP = IS_SOFTCAP_;

    CATLASS_DEVICE
    BlockEpilogue() = default;

    CATLASS_DEVICE
    void Init(
        Arch::Resource<ArchTag> &resource,
        GM_ADDR workspace,
        GM_ADDR tiling)
    {
        (void)resource;
        tiling_ = reinterpret_cast<const __gm__ TilingData *>(tiling);
        deltaWorkspaceGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(
                workspace + tiling_->deltaOffset));
    }

    template <
        class TensorDP,
        class TensorP,
        class TensorSoftcapGrad,
        class TensorDelta,
        class TensorDS>
    CATLASS_DEVICE
    void operator()(
        FAGBlockInfo const &block,
        TensorDP const &ubDPTensor,
        TensorP const &ubPTensor,
        TensorSoftcapGrad const &ubSoftcapGradTensor,
        TensorDS &ubDSTensor,
        TensorDelta &deltaUbTensor,
        TensorDS &l1DSTensor,
        event_t mte2ToVEvent,
        event_t vToMte3Event,
        event_t mte3ToVEvent,
        uint32_t subBlockIdx)
    {
        using namespace AscendC::MicroAPI;
        constexpr uint32_t rowStride = 128;
        constexpr uint32_t lanes = 64;
        constexpr uint32_t c0 = 32U / sizeof(ElementDS);

        const uint32_t m = subBlockIdx == 0
            ? block.firstHalfRealS1
            : block.s1Extend - block.firstHalfRealS1;
        const uint32_t rowOffset = subBlockIdx * block.firstHalfRealS1;
        const uint64_t qBase = static_cast<uint64_t>(block.totalS1Start) +
            rowOffset;
        const uint64_t head = static_cast<uint64_t>(block.n2Idx) *
            static_cast<uint64_t>(tiling_->groupSize) + block.groupIdx;

        // DeltaGm: [T1, N1] --> DeltaUB: [m, 8]
        const uint64_t deltaOffset = qBase * tiling_->qHeadNum + head;
        AscendC::DataCopyExtParams deltaCopyParams{
            static_cast<uint16_t>(m),
            static_cast<uint32_t>(sizeof(float)),
            static_cast<int64_t>((tiling_->qHeadNum - 1U) * sizeof(float)),
            0, 0};
        AscendC::DataCopyPadExtParams<float> deltaPadParams{
            false, 0, 0, 0};
        AscendC::DataCopyPad(
            deltaUbTensor,
            deltaWorkspaceGm_[deltaOffset],
            deltaCopyParams,
            deltaPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent);

        auto dp = reinterpret_cast<__ubuf__ ElementDP *>(ubDPTensor.GetPhyAddr());
        auto p = reinterpret_cast<__ubuf__ ElementP *>(ubPTensor.GetPhyAddr());
        auto softcapGrad = reinterpret_cast<__ubuf__ float *>(
            ubSoftcapGradTensor.GetPhyAddr());
        auto out = reinterpret_cast<__ubuf__ ElementDS *>(ubDSTensor.GetPhyAddr());
        auto delta = reinterpret_cast<__ubuf__ float *>(deltaUbTensor.GetPhyAddr());

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent);

        ComputeSubMul(dp, p, softcapGrad, delta, out, m);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event);
        const uint32_t l1M = RoundUp(block.s1Extend, c0);
        const uint32_t l1Offset = subBlockIdx == 0
            ? 0 : block.firstHalfRealS1 * c0;
        AscendC::DataCopyParams copyParams{
            static_cast<uint16_t>(rowStride / c0),
            static_cast<uint16_t>(m), 1,
            static_cast<uint16_t>(l1M - m)};
        AscendC::DataCopy(l1DSTensor[l1Offset], ubDSTensor, copyParams);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent);
    }

private:
    __simd_vf__ inline static void ComputeSubMul(
        __ubuf__ ElementDP *dp, __ubuf__ ElementP *p,
        __ubuf__ float *softcapGrad, __ubuf__ float *delta,
        __ubuf__ ElementDS *out, uint32_t m)
    {
        using namespace AscendC::MicroAPI;
        constexpr uint32_t rowStride = 128;
        constexpr uint32_t lanes = 64;
        constexpr uint32_t c0 = 32U / sizeof(ElementP);
        const uint32_t nzBlockStride = m + 1U;
        RegTensor<float> dp0, dp1;
        RegTensor<ElementP> pPacked;
        RegTensor<float> pEven, pOdd, p0f, p1f, deltaV, r0, r1;
        RegTensor<float> softcapGrad0, softcapGrad1;
        RegTensor<float> rEven, rOdd;
        RegTensor<ElementDS> oEven, oOdd, oPacked;
        MaskReg fullDp = CreateMask<float, MaskPattern::ALL>();
        MaskReg fullP = CreateMask<ElementP, MaskPattern::ALL>();
        MaskReg fullOut = CreateMask<ElementDS, MaskPattern::ALL>();
        constexpr static CastTrait b16ToB32Even = {
            RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
        constexpr static CastTrait b16ToB32Odd = {
            RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
        constexpr static CastTrait b32ToB16Even = {
            RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND};
        constexpr static CastTrait b32ToB16Odd = {
            RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND};
        for (uint32_t i = 0; i < m; ++i) {
            LoadAlign(dp0, dp + i * rowStride);
            LoadAlign(dp1, dp + i * rowStride + lanes);
            __ubuf__ ElementP *pRow = p + i * c0;
            LoadAlign<ElementP, DataCopyMode::DATA_BLOCK_COPY,
                PostLiteral::POST_MODE_UPDATE>(
                pPacked, pRow, nzBlockStride, 1, fullP);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(deltaV, delta + i * 8U);
            Cast<float, ElementP, b16ToB32Even>(pEven, pPacked, fullP);
            Cast<float, ElementP, b16ToB32Odd>(pOdd, pPacked, fullP);
            Interleave(p0f, p1f, pEven, pOdd);
            Sub(r0, dp0, deltaV, fullDp);
            Sub(r1, dp1, deltaV, fullDp);
            Mul(r0, r0, p0f, fullDp);
            Mul(r1, r1, p1f, fullDp);
            if constexpr (IS_SOFTCAP) {
                LoadAlign(softcapGrad0, softcapGrad + i * rowStride);
                LoadAlign(softcapGrad1,
                    softcapGrad + i * rowStride + lanes);
                Mul(r0, r0, softcapGrad0, fullDp);
                Mul(r1, r1, softcapGrad1, fullDp);
            }
            DeInterleave(rEven, rOdd, r0, r1);
            Cast<ElementDS, float, b32ToB16Even>(oEven, rEven, fullOut);
            Cast<ElementDS, float, b32ToB16Odd>(oOdd, rOdd, fullOut);
            Or(reinterpret_cast<RegTensor<uint16_t> &>(oPacked),
                reinterpret_cast<RegTensor<uint16_t> &>(oEven),
                reinterpret_cast<RegTensor<uint16_t> &>(oOdd), fullOut);
            __ubuf__ ElementDS *outRow = out + i * c0;
            StoreAlign<ElementDS, DataCopyMode::DATA_BLOCK_COPY,
                PostLiteral::POST_MODE_UPDATE>(
                outRow, oPacked, nzBlockStride, 1, fullOut);
        }
    }

    const __gm__ TilingData *tiling_ = nullptr;
    AscendC::GlobalTensor<float> deltaWorkspaceGm_;
};

}  // namespace Catlass::Epilogue::Block

#endif  // FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_SUB_MUL_HPP
