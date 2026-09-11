/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend950 FlashAttention v3 backward scaled-mask-softmax epilogue.
 */

#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_SCALED_MASK_SOFTMAX_HPP
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_SCALED_MASK_SOFTMAX_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"

#include "fag_block.h"

namespace Catlass::Epilogue::Block {

template <
    FAGTiling950::Layout INPUT_LAYOUT_,
    bool IS_ATTEN_MASK_,
    bool IS_SOFTCAP_,
    class ElementP_,
    class ElementS_,
    class TilingData_>
class BlockEpilogue<
    EpilogueAscend950FAGScaledMaskSoftmax<
        INPUT_LAYOUT_, IS_ATTEN_MASK_, IS_SOFTCAP_>,
    ElementP_,
    ElementS_,
    TilingData_> {
public:
    using DispatchPolicy =
        EpilogueAscend950FAGScaledMaskSoftmax<
            INPUT_LAYOUT_, IS_ATTEN_MASK_, IS_SOFTCAP_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementP = ElementP_;
    using ElementS = ElementS_;
    using TilingData = TilingData_;

    static constexpr FAGTiling950::Layout INPUT_LAYOUT = INPUT_LAYOUT_;
    static constexpr bool IS_ATTEN_MASK = IS_ATTEN_MASK_;
    static constexpr bool IS_SOFTCAP = IS_SOFTCAP_;

    CATLASS_DEVICE
    BlockEpilogue() = default;

    CATLASS_DEVICE
    void Init(
        Arch::Resource<ArchTag> &resource,
        GM_ADDR tiling)
    {
        tiling_ = reinterpret_cast<const __gm__ TilingData *>(tiling);
    }

    template <
        class TensorAttenMaskGm,
        class TensorAttenMaskUb,
        class TensorSoftmaxLseGm,
        class TensorLseUb,
        class TensorS,
        class TensorPUb,
        class TensorPL1>
    CATLASS_DEVICE
    void operator()(
        FAGBlockInfo const &block,
        TensorAttenMaskGm const &attenMaskGm,
        TensorSoftmaxLseGm const &softmaxLseGm,
        TensorAttenMaskUb &attenMaskUbTensor,
        TensorLseUb &lseUbTensor,
        TensorS const &ubMm1Tensor,
        TensorPUb &ubPTensor,
        TensorPL1 &l1PTensor,
        event_t mte2ToVEvent,
        event_t vToMte3Event,
        event_t mte3ToVEvent,
        float scaleValue,
        float softcapValue,
        uint64_t s1RealSize,
        uint32_t subBlockIdx)
    {
        // Intermediate row stride is always 128 elements. s2Extend is the
        // number of valid columns in this basic block.
        // mask: [s1RealSize, 128] * sizeof(uint8)
        // lseUbTensor: [AlignUp(s1RealSize, 8)] * sizeof(fp32)
        // ubMm1Tensor: [s1RealSize, 128] * sizeof(fp32)
        // ubPTensor: [128 / C0, s1RealSize + 1, C0]
        // l1PTensor: [128 / C0, AlignUp(block.s1Extend, 16), C0]

        // 1. Copy in attenMask (only support causal now)
        if constexpr (IS_ATTEN_MASK_) {
            const int64_t diagOffset =
                static_cast<int64_t>(block.curBatchS2) -
                static_cast<int64_t>(block.curBatchS1);
            const int64_t rowStart =
                static_cast<int64_t>(block.s1Start) +
                static_cast<int64_t>(subBlockIdx) * block.firstHalfRealS1;
            const int64_t colEnd =
                static_cast<int64_t>(block.s2Start) + block.s2Extend;
            if (colEnd - 1 > rowStart + diagOffset) {
                // [maskRowStart, maskRowStart + s1RealSize)
                // [maskColStart, maskColStart + s2Extend)
                const int64_t delta =
                    static_cast<int64_t>(block.s2Start) -
                    rowStart - diagOffset;
                const uint32_t maskRowStart = Max(-delta, (int64_t)0);
                const uint32_t maskColStart = Max(delta, (int64_t)0);
                const auto maskGmOffset = attenMaskGm.layout()(
                    tla::MakeCoord(maskRowStart, maskColStart));

                AscendC::DataCopyExtParams maskCopyParams{
                    static_cast<uint16_t>(s1RealSize),
                    static_cast<uint32_t>(S2_ROW_STRIDE * sizeof(uint8_t)),
                    static_cast<int64_t>((256U - S2_ROW_STRIDE) * sizeof(uint8_t)),
                    0, 0};
                AscendC::DataCopyPadExtParams<uint8_t> maskPadParams{
                    false, 0, 0, 0};
                AscendC::DataCopyPad(
                    attenMaskUbTensor,
                    attenMaskGm.data()[maskGmOffset],
                    maskCopyParams,
                    maskPadParams);
            } else {
                // This AIV slice lies completely below the causal diagonal.
                // Clear the reusable ping/pong mask slot so a previous masked
                // tile cannot leak into the current unmasked tile.
                AscendC::Duplicate(
                    attenMaskUbTensor,
                    static_cast<uint8_t>(0),
                    static_cast<uint32_t>(s1RealSize) * S2_ROW_STRIDE);
            }
        }

        // 2. Copy in lse ([B,N1,S1] or [N1,totalS1])
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(s1RealSize * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        const auto lseGmOffset = softmaxLseGm.layout()(softmaxLseGm.coord());
        AscendC::DataCopyPad(
            lseUbTensor,
            softmaxLseGm.data()[lseGmOffset],
            copyParams,
            padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent);

        // 3. scale + optional softcap + mask + exp(S-LSE) + Cast + Nd2Nz
        MulsMaskSimpleSoftmax<IS_ATTEN_MASK, IS_SOFTCAP>(
            reinterpret_cast<__ubuf__ ElementP *>(ubPTensor.GetPhyAddr()),
            reinterpret_cast<__ubuf__ ElementS *>(ubMm1Tensor.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint8_t *>(attenMaskUbTensor.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(lseUbTensor.GetPhyAddr()),
            static_cast<uint16_t>(s1RealSize),
            block.s2Extend,
            scaleValue,
            softcapValue);
        if constexpr (IS_SOFTCAP) {
            AscendC::PipeBarrier<PIPE_V>();
        }

        // 4. UB --> L1
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event);

        const uint32_t l1M = RoundUp(block.s1Extend, NZ_C0_SIZE);
        const uint32_t l1Offset = subBlockIdx == 0 ? 0 : block.firstHalfRealS1 * NZ_C0_SIZE;
        AscendC::DataCopyParams pCopyParams;
        pCopyParams.blockCount = S2_ROW_STRIDE / NZ_C0_SIZE;
        pCopyParams.blockLen = static_cast<uint16_t>(s1RealSize);
        pCopyParams.srcStride = 1;
        pCopyParams.dstStride = static_cast<uint16_t>(l1M - s1RealSize);
        AscendC::DataCopy(l1PTensor[l1Offset], ubPTensor, pCopyParams);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent);
    }

private:
    /**
     * Rebuild P from the QK result and the LSE saved by forward:
     *
     *   logits = scaleValue * S
     *   logits = softcap * tanh(logits)  (when enabled; the kernel passes
     *                                    scaleValue = scale / softcap)
     *   P = exp(logits - LSE)
     *
     * A non-zero byte in maskUb denotes a masked element. S, mask, and P all
     * use a fixed 128-element physical row stride; n is only the number of
     * valid columns in the current basic block.
     */
    template <bool HAS_ATTEN_MASK, bool HAS_SOFTCAP>
    __simd_vf__ inline void MulsMaskSimpleSoftmax(
        __ubuf__ ElementP *dstUb,
        __ubuf__ ElementS *srcUb,
        __ubuf__ uint8_t *maskUb,
        __ubuf__ float *lseUb,
        uint16_t m,
        uint32_t n,
        float scaleValue,
        float softcapValue)
    {
        using namespace AscendC::MicroAPI;

        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };
        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        RegTensor<float> srcVreg0;
        RegTensor<float> srcVreg1;
        RegTensor<float> srcEvenVreg;
        RegTensor<float> srcOddVreg;
        RegTensor<float> lseVreg;
        RegTensor<float> minVreg;
        RegTensor<float> softcapNumeratorVreg;
        RegTensor<float> softcapGradVreg0;
        RegTensor<float> softcapGradVreg1;
        RegTensor<ElementP> dstVreg0;
        RegTensor<ElementP> dstVreg1;
        RegTensor<ElementP> dstVreg;

        MaskReg maskVreg0;
        MaskReg maskVreg1;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregFullP = CreateMask<ElementP, MaskPattern::ALL>();
        uint32_t tail0 = n < FP32_PER_VREG ? n : FP32_PER_VREG;
        uint32_t tail1 = n > FP32_PER_VREG ? n - FP32_PER_VREG : 0;
        MaskReg pregTail0 = UpdateMask<float>(tail0);
        MaskReg pregTail1 = UpdateMask<float>(tail1);

        const uint32_t nzBlockStride = m * NZ_C0_SIZE * sizeof(ElementP) / 32U + 1U;

        Duplicate(minVreg, MASKED_VALUE);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg0, srcUb + i * S2_ROW_STRIDE);
            LoadAlign(srcVreg1, srcUb + i * S2_ROW_STRIDE + FP32_PER_VREG);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(lseVreg, lseUb + i); // broadcast b32

            Muls(srcVreg0, srcVreg0, scaleValue, pregTail0);
            Muls(srcVreg1, srcVreg1, scaleValue, pregTail1);

            if constexpr (HAS_SOFTCAP) {
                // Compute tanh(x) once for both the reconstructed logits and
                // the backward factor 1 - tanh(x)^2. The lower clamp bounds
                // exp(-2*x) without affecting the representable tanh result.
                Duplicate(softcapNumeratorVreg, 2.0f);
                Maxs(srcVreg0, srcVreg0, -8.8f, pregTail0);
                Muls(srcVreg0, srcVreg0, -2.0f, pregTail0);
                Exp(srcVreg0, srcVreg0, pregTail0);
                Adds(srcVreg0, srcVreg0, 1.0f, pregTail0);
                Div(srcVreg0, softcapNumeratorVreg, srcVreg0, pregTail0);
                Adds(srcVreg0, srcVreg0, -1.0f, pregTail0);

                Maxs(srcVreg1, srcVreg1, -8.8f, pregTail1);
                Muls(srcVreg1, srcVreg1, -2.0f, pregTail1);
                Exp(srcVreg1, srcVreg1, pregTail1);
                Adds(srcVreg1, srcVreg1, 1.0f, pregTail1);
                Div(srcVreg1, softcapNumeratorVreg, srcVreg1, pregTail1);
                Adds(srcVreg1, srcVreg1, -1.0f, pregTail1);

                Mul(softcapGradVreg0, srcVreg0, srcVreg0, pregTail0);
                Muls(softcapGradVreg0, softcapGradVreg0, -1.0f, pregTail0);
                Adds(softcapGradVreg0, softcapGradVreg0, 1.0f, pregTail0);
                Mul(softcapGradVreg1, srcVreg1, srcVreg1, pregTail1);
                Muls(softcapGradVreg1, softcapGradVreg1, -1.0f, pregTail1);
                Adds(softcapGradVreg1, softcapGradVreg1, 1.0f, pregTail1);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(
                    srcUb + i * S2_ROW_STRIDE, softcapGradVreg0, pregTail0);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(
                    srcUb + i * S2_ROW_STRIDE + FP32_PER_VREG,
                    softcapGradVreg1, pregTail1);

                Muls(srcVreg0, srcVreg0, softcapValue, pregTail0);
                Muls(srcVreg1, srcVreg1, softcapValue, pregTail1);
            }

            if constexpr (HAS_ATTEN_MASK) {
                __ubuf__ uint8_t *maskRow = maskUb + i * S2_ROW_STRIDE;
                LoadAlign<uint32_t, PostLiteral::POST_MODE_UPDATE,
                    MaskDist::DIST_DS>(maskVreg0,
                    (__ubuf__ uint32_t *&)maskRow, FP32_PER_VREG); // 64Byte(512bit) --> DIST_DS --> 256bit
                LoadAlign<uint32_t, PostLiteral::POST_MODE_UPDATE,
                    MaskDist::DIST_DS>(maskVreg1,
                    (__ubuf__ uint32_t *&)maskRow, FP32_PER_VREG);
                // (dst, src0, src1, mask): 1->src0, 0->src1
                Select(srcVreg0, minVreg, srcVreg0, maskVreg0);
                Select(srcVreg1, minVreg, srcVreg1, maskVreg1);
            }

            // Mask padding elements
            Select(srcVreg0, srcVreg0, minVreg, pregTail0);
            Select(srcVreg1, srcVreg1, minVreg, pregTail1);
            ExpSub(srcVreg0, srcVreg0, lseVreg, pregFull);
            ExpSub(srcVreg1, srcVreg1, lseVreg, pregFull);

            // [0..63], [64..127]
            //   -> even: [0,2,4,...,126]
            //   -> odd:  [1,3,5,...,127]
            DeInterleave(srcEvenVreg, srcOddVreg, srcVreg0, srcVreg1);

            if constexpr (AscendC::IsSameType<ElementP, bfloat16_t>::value) {
                Cast<bfloat16_t, float, castTraitZero>(
                    dstVreg0, srcEvenVreg, pregFull);
                Cast<bfloat16_t, float, castTraitOne>(
                    dstVreg1, srcOddVreg, pregFull);
            } else {
                Cast<half, float, castTraitZero>(
                    dstVreg0, srcEvenVreg, pregFull);
                Cast<half, float, castTraitOne>(
                    dstVreg1, srcOddVreg, pregFull);
            }
            Or(
                reinterpret_cast<RegTensor<uint16_t> &>(dstVreg),
                reinterpret_cast<RegTensor<uint16_t> &>(dstVreg0),
                reinterpret_cast<RegTensor<uint16_t> &>(dstVreg1),
                pregFullP);
            // Fused ND -> NZ scatter:
            //   [m, 128] -> [128 / C0, m + 1, C0].
            StoreAlign<ElementP, DataCopyMode::DATA_BLOCK_COPY,
                PostLiteral::POST_MODE_UPDATE>(
                dstUb, dstVreg, nzBlockStride, 1, pregFullP);
        }
    }
private:
    static constexpr uint32_t S2_ROW_STRIDE = 128;
    static constexpr uint32_t FP32_PER_VREG = 64;
    static constexpr float MASKED_VALUE = -3.0e38f;
    static constexpr uint32_t NZ_C0_SIZE = 32U / sizeof(ElementP);
    const __gm__ TilingData *tiling_ = nullptr;
};

}  // namespace Catlass::Epilogue::Block

#endif  // FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_SCALED_MASK_SOFTMAX_HPP
