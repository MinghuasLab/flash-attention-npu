// Copyright (c) 2026. Mixed-storage KV producer for the Ascend 950 FA3 pipeline.
#ifndef FA950_KV_DEQUANT_HPP
#define FA950_KV_DEQUANT_HPP

#include "kernel_operator.h"
#include "fwd_resource_plan.h"

namespace fa950_mixed {
constexpr uint32_t HEAD_DIM = 192;
constexpr uint32_t CHUNK_ROWS = 64;
constexpr uint32_t CHUNK_ELEMS = CHUNK_ROWS * HEAD_DIM;
constexpr uint32_t NZ_ROWS = CHUNK_ROWS + 1U; // Spread adjacent NZ stripes across UB banks.
constexpr uint32_t DENSE_SCALE_BYTES = CHUNK_ROWS * 4U * sizeof(float);
constexpr uint32_t STAGING_BYTES = CHUNK_ELEMS + NZ_ROWS * HEAD_DIM * 2U + DENSE_SCALE_BYTES;
static_assert(fa950_plan::MakeUbPlan(HEAD_DIM, 1).total + STAGING_BYTES <= fa950_plan::UB_CAPACITY,
              "Mixed D192 staging must fit alongside attention state");

// K[D,N] in nZ and V[N,D] in zN share the physical address
// (d/16)*round_up(N,16)*16 + n*16 + d%16. Each AIV owns 64 KV rows.
// One AIV dequantizes its complete 64-row half tile per VF call. Up to four
// KV heads use compact scales; the generic path aliases scales with NZ stripe zero.
class KvDequant {
public:
    template<class Resource>
    __aicore__ inline KvDequant(Resource& resource, uint32_t ubOffset)
    {
        input_ = resource.ubBuf.template GetBufferByByte<float8_e4m3_t>(ubOffset);
        output_ = resource.ubBuf.template GetBufferByByte<bfloat16_t>(ubOffset + CHUNK_ELEMS);
        // Scale row r occupies the 32-byte NZ output block (C1=0, row=r).
        // DequantNz loads that scale before writing the same row's first block;
        // other rows' scale blocks remain untouched until their own load.
        scale_ = output_.template ReinterpretCast<float>();
        denseScale_ = resource.ubBuf.template GetBufferByByte<float>(
            ubOffset + CHUNK_ELEMS + NZ_ROWS * HEAD_DIM * 2U);
    }

    template<class Resource>
    __aicore__ inline void Produce(Resource& resource, GM_ADDR cache, GM_ADDR scales,
                                    AscendC::GlobalTensor<int32_t> pages, uint32_t head, uint32_t heads,
                                    uint64_t pageStride, uint64_t tokenStride, uint64_t headStride,
                                    uint32_t tile, uint32_t rows, uint32_t l1Offset, uint32_t flag)
    {
        AscendC::GlobalTensor<uint8_t> src;
        AscendC::GlobalTensor<float> scale;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(cache));
        scale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
        auto dst = resource.l1Buf.template GetBufferByByte<bfloat16_t>(l1Offset);
        const uint32_t lane = AscendC::GetSubBlockIdx();
        const uint32_t begin = lane * 64U;
        const uint32_t end = rows < begin + 64U ? rows : begin + 64U;
        const uint32_t paddedRows = (rows + 15U) / 16U * 16U;
        const uint64_t page = static_cast<uint64_t>(pages.GetValue(tile));
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(flag);
        for (uint32_t start = begin; start < end; start += CHUNK_ROWS) {
            const uint32_t count = end - start < CHUNK_ROWS ? end - start : CHUNK_ROWS;
            // Compact scales and FP8 input have separate storage from NZ.
            // Both can be overwritten as soon as Vector has consumed them.
            Drain(heads);
            AscendC::DataCopyExtParams copy{static_cast<uint16_t>(count), HEAD_DIM,
                                           static_cast<uint32_t>(tokenStride - HEAD_DIM), 0, 0};
            AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(input_.template ReinterpretCast<uint8_t>(), src[page * pageStride + head * headStride + start * tokenStride], copy, pad);
            // TP1/TP2/TP4 scales fit in 1 KiB. Read a contiguous token range
            // and select the head in VF instead of issuing 64 four-byte blocks.
            auto scaleValues = scale_;
            uint32_t scaleStride = 8U;
            AscendC::DataCopyPadExtParams<float> scalePad{false, 0, 0, 0.0f};
            if (heads <= 4U) {
                AscendC::DataCopyExtParams scaleCopy{1, count * heads * 4U, 0, 0, 0};
                AscendC::DataCopyPad(denseScale_, scale[(page * 128U + start) * heads], scaleCopy, scalePad);
                scaleValues = denseScale_[head];
                scaleStride = heads;
            } else {
                AscendC::DataCopyExtParams scaleCopy{static_cast<uint16_t>(count), 4U, (heads - 1U) * 4U, 0, 0};
                AscendC::DataCopyPad(scale_, scale[(page * 128U + start) * heads + head], scaleCopy, scalePad);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
            if (heads <= 4U) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            }
            DequantNz(reinterpret_cast<__ubuf__ bfloat16_t*>(output_.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float8_e4m3_t*>(input_.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float*>(scaleValues.GetPhyAddr()), count, scaleStride);
            if (heads <= 4U) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            const uint32_t copyRows = (count + 15U) / 16U * 16U;
            const AscendC::DataCopyParams scatter{HEAD_DIM / 16U, static_cast<uint16_t>(copyRows),
                                                  static_cast<uint16_t>(NZ_ROWS - copyRows),
                                                  static_cast<uint16_t>(paddedRows - copyRows)};
            AscendC::DataCopy(dst[start * 16U], output_, scatter);
            if (heads <= 4U) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            } else {
                // Strided scales alias NZ, so MTE2 must wait for its copy.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            }
            stagingInFlight_ = true;
        }
        // Even a lane with no valid rows participates in the handshake.
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(flag);
    }

    __aicore__ inline void Drain(uint32_t heads)
    {
        if (stagingInFlight_) {
            if (heads <= 4U) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            }
            stagingInFlight_ = false;
        }
    }
private:
    friend class KvDequantBuffered;
    friend class KvDequantWhole;
    // Keep conversion and FP32 scaling in registers. Pair independent fragments
    // and deinterleave the even BF16 lanes before NZ stores.
    __simd_vf__ static inline void DequantNz(__ubuf__ bfloat16_t* dst, __ubuf__ float8_e4m3_t* src,
                                            __ubuf__ float* scales, uint32_t validRows, uint32_t scaleStride, uint32_t nzRows = NZ_ROWS)
    {
        using namespace AscendC::Reg;
        static constexpr CastTrait toFloat = {RegLayout::ZERO, SatMode::NO_SAT,
                                               MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE};
        static constexpr CastTrait toBf16 = {RegLayout::ZERO, SatMode::SAT,
                                              MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait toFloatOdd = {RegLayout::ONE, SatMode::NO_SAT,
                                               MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE};
        static constexpr CastTrait toBf16Odd = {RegLayout::ONE, SatMode::SAT,
                                              MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        RegTensor<float8_e4m3_t> fp8a, fp8b, fp8c;
        RegTensor<float> fa, fb, fc, scale;
        RegTensor<bfloat16_t> ba, bb, bc, packed128, packed64, unused;
        RegTensor<float8_e4m3_t> fp8a_r1, fp8b_r1, fp8c_r1;
        RegTensor<float> fa_r1, fb_r1, fc_r1, scale_r1;
        RegTensor<bfloat16_t> ba_r1, bb_r1, bc_r1, packed128_r1, packed64_r1, unused_r1;
        // Enable both even and odd lanes for the two conversion layouts.
        MaskReg full32 = CreateMask<uint8_t, MaskPattern::ALL>();
        MaskReg mask128 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg mask64 = CreateMask<uint16_t, MaskPattern::VL64>();
        const uint16_t activeRows = static_cast<uint16_t>(validRows);
        // Interleave independent rows to expose conversion/multiply scheduling.
        uint16_t row = 0;
        for (; row + 1U < activeRows; row += 2U) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scale, scales + row * scaleStride);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scale_r1, scales + (row + 1U) * scaleStride);
            LoadAlign<uint8_t, LoadDist::DIST_UNPACK_B16>((RegTensor<uint8_t>&)fp8a,
                (__ubuf__ uint8_t*)(src + row * HEAD_DIM));
            LoadAlign<uint8_t, LoadDist::DIST_UNPACK_B16>((RegTensor<uint8_t>&)fp8a_r1,
                (__ubuf__ uint8_t*)(src + (row + 1U) * HEAD_DIM));
            LoadAlign<float8_e4m3_t, LoadDist::DIST_UNPACK4_B8>(fp8c, src + row * HEAD_DIM + 128U);
            LoadAlign<float8_e4m3_t, LoadDist::DIST_UNPACK4_B8>(fp8c_r1, src + (row + 1U) * HEAD_DIM + 128U);
            Cast<float, float8_e4m3_t, toFloat>(fa, fp8a, full32);
            Cast<float, float8_e4m3_t, toFloat>(fa_r1, fp8a_r1, full32);
            Cast<float, float8_e4m3_t, toFloatOdd>(fb, fp8a, full32);
            Cast<float, float8_e4m3_t, toFloatOdd>(fb_r1, fp8a_r1, full32);
            Cast<float, float8_e4m3_t, toFloat>(fc, fp8c, full32);
            Cast<float, float8_e4m3_t, toFloat>(fc_r1, fp8c_r1, full32);
            Mul(fa, fa, scale, full32);
            Mul(fa_r1, fa_r1, scale_r1, full32);
            Mul(fb, fb, scale, full32);
            Mul(fb_r1, fb_r1, scale_r1, full32);
            Mul(fc, fc, scale, full32);
            Mul(fc_r1, fc_r1, scale_r1, full32);
            Cast<bfloat16_t, float, toBf16>(ba, fa, full32);
            Cast<bfloat16_t, float, toBf16>(ba_r1, fa_r1, full32);
            Cast<bfloat16_t, float, toBf16Odd>(bb, fb, full32);
            Cast<bfloat16_t, float, toBf16Odd>(bb_r1, fb_r1, full32);
            Cast<bfloat16_t, float, toBf16>(bc, fc, full32);
            Cast<bfloat16_t, float, toBf16>(bc_r1, fc_r1, full32);
            Or<uint16_t>((RegTensor<uint16_t>&)packed128, (RegTensor<uint16_t>&)ba,
                         (RegTensor<uint16_t>&)bb, mask128);
            Or<uint16_t>((RegTensor<uint16_t>&)packed128_r1, (RegTensor<uint16_t>&)ba_r1,
                         (RegTensor<uint16_t>&)bb_r1, mask128);
            DeInterleave(packed64, unused, bc, bc);
            DeInterleave(packed64_r1, unused_r1, bc_r1, bc_r1);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(
                dst + row * 16U, packed128, nzRows, mask128);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(
                dst + (row + 1U) * 16U, packed128_r1, nzRows, mask128);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(
                dst + 8U * nzRows * 16U + row * 16U, packed64, nzRows, mask64);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(
                dst + 8U * nzRows * 16U + (row + 1U) * 16U, packed64_r1, nzRows, mask64);
        }
        // Odd tails must not read the next token or its scale.
        for (; row < activeRows; ++row) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scale, scales + row * scaleStride);
            LoadAlign<uint8_t, LoadDist::DIST_UNPACK_B16>((RegTensor<uint8_t>&)fp8a,
                (__ubuf__ uint8_t*)(src + row * HEAD_DIM));
            LoadAlign<float8_e4m3_t, LoadDist::DIST_UNPACK4_B8>(fp8c, src + row * HEAD_DIM + 128U);
            Cast<float, float8_e4m3_t, toFloat>(fa, fp8a, full32);
            Cast<float, float8_e4m3_t, toFloatOdd>(fb, fp8a, full32);
            Cast<float, float8_e4m3_t, toFloat>(fc, fp8c, full32);
            Mul(fa, fa, scale, full32);
            Mul(fb, fb, scale, full32);
            Mul(fc, fc, scale, full32);
            Cast<bfloat16_t, float, toBf16>(ba, fa, full32);
            Cast<bfloat16_t, float, toBf16Odd>(bb, fb, full32);
            Cast<bfloat16_t, float, toBf16>(bc, fc, full32);
            Or<uint16_t>((RegTensor<uint16_t>&)packed128, (RegTensor<uint16_t>&)ba,
                         (RegTensor<uint16_t>&)bb, mask128);
            DeInterleave(packed64, unused, bc, bc);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(
                dst + row * 16U, packed128, nzRows, mask128);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(
                dst + 8U * nzRows * 16U + row * 16U, packed64, nzRows, mask64);
                }
        Duplicate((RegTensor<uint16_t>&)packed128, static_cast<uint16_t>(0));
        const uint16_t paddingRows = static_cast<uint16_t>((validRows + 15U) / 16U * 16U - validRows);
        for (uint16_t pad = 0; pad < paddingRows; ++pad) {
            const uint32_t row = validRows + pad;
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(
                dst + row * 16U, packed128, nzRows, mask128);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(
                dst + 8U * nzRows * 16U + row * 16U, packed128, nzRows, mask64);
        }
    }
    AscendC::LocalTensor<float8_e4m3_t> input_;
    AscendC::LocalTensor<bfloat16_t> output_;
    AscendC::LocalTensor<float> scale_;
    AscendC::LocalTensor<float> denseScale_;
    bool stagingInFlight_ = false;
};

constexpr uint32_t BUFFERED_SCALE_BYTES = CHUNK_ROWS * 32U;
constexpr uint32_t HALF_INPUT_BYTES = CHUNK_ELEMS + BUFFERED_SCALE_BYTES;
constexpr uint32_t OUTPUT_ROWS = CHUNK_ROWS / 2U;
constexpr uint32_t OUTPUT_NZ_ROWS = OUTPUT_ROWS + 1U;
constexpr uint32_t OUTPUT_BANK_BYTES = OUTPUT_NZ_ROWS * HEAD_DIM * sizeof(bfloat16_t);
static_assert(fa950_plan::MakeUbPlan(HEAD_DIM, 1, 64, 64, 1).total +
              2U * (HALF_INPUT_BYTES + OUTPUT_BANK_BYTES) <= fa950_plan::UB_CAPACITY,
              "Two FP8 inputs and two independent NZ outputs must fit at Q tile 128");

static_assert(fa950_plan::MakeUbPlan(HEAD_DIM, 1, 64, 64, 1, 1).mask +
              64U * HEAD_DIM * sizeof(float) + 2U * 64U * sizeof(float) +
              2U * (HALF_INPUT_BYTES + OUTPUT_BANK_BYTES) <= fa950_plan::UB_CAPACITY,
              "Paired Q128 with two 32-row NZ banks must fit");
static_assert(64U * 128U <= OUTPUT_BANK_BYTES, "The first NZ bank must contain the causal mask");
static_assert(2U * 64U * 128U <= 2U * OUTPUT_BANK_BYTES, "The two NZ banks must contain both SWA masks");
// Each AIV stages its operand in two 64-row inputs. Convert in 32-row
// stripes so MTE3 can read one NZ bank while Vector fills the other.
class KvDequantBuffered {
public:
    template<class Resource>
    __aicore__ inline KvDequantBuffered(Resource& resource, uint32_t ubOffset)
    {
        for (uint32_t slot = 0; slot < 2U; ++slot) {
            const uint32_t base = ubOffset + slot * HALF_INPUT_BYTES;
            input_[slot] = resource.ubBuf.template GetBufferByByte<uint8_t>(base);
            scale_[slot] = resource.ubBuf.template GetBufferByByte<float>(base + CHUNK_ELEMS);
        }
        for (uint32_t bank = 0; bank < 2U; ++bank) {
            output_[bank] = resource.ubBuf.template GetBufferByByte<bfloat16_t>(
                ubOffset + 2U * HALF_INPUT_BYTES + bank * OUTPUT_BANK_BYTES);
        }
    }

    __aicore__ inline void Prefetch(GM_ADDR cache, GM_ADDR scales,
        AscendC::GlobalTensor<int32_t> pages, uint32_t head, uint32_t heads,
        uint64_t pageStride, uint64_t tokenStride, uint64_t headStride,
        uint32_t tile, uint32_t rows, uint32_t kvSlot)
    {
        if (AscendC::GetSubBlockIdx() != 1U - kvSlot) { return; }
        const uint64_t page = static_cast<uint64_t>(pages.GetValue(tile));
        for (uint32_t slot = 0; slot < 2U; ++slot) {
            const uint32_t start = slot * CHUNK_ROWS;
            count_[slot] = rows > start ? AscendC::Std::min(rows - start, CHUNK_ROWS) : 0U;
            if (count_[slot] == 0U) { continue; }
            // Input/scales are dead as soon as the previous VF has consumed them.
            // Do not hold the next GM read until the independent NZ output DMA ends.
            DrainInput(slot);
            AscendC::GlobalTensor<uint8_t> src;
            AscendC::GlobalTensor<float> scaleGm;
            src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(cache));
            scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
            AscendC::DataCopyExtParams copy{static_cast<uint16_t>(count_[slot]), HEAD_DIM,
                static_cast<uint32_t>(tokenStride - HEAD_DIM), 0, 0};
            AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(input_[slot], src[page * pageStride + head * headStride + start * tokenStride], copy, pad);
            AscendC::DataCopyPadExtParams<float> scalePad{false, 0, 0, 0.0f};
            if (heads <= 4U) {
                AscendC::DataCopyExtParams scaleCopy{1, count_[slot] * heads * 4U, 0, 0, 0};
                AscendC::DataCopyPad(scale_[slot], scaleGm[(page * 128U + start) * heads], scaleCopy, scalePad);
                scaleOffset_[slot] = head;
                scaleStride_[slot] = heads;
            } else {
                AscendC::DataCopyExtParams scaleCopy{static_cast<uint16_t>(count_[slot]), 4U,
                    (heads - 1U) * 4U, 0, 0};
                AscendC::DataCopyPad(scale_[slot], scaleGm[(page * 128U + start) * heads + head], scaleCopy, scalePad);
                scaleOffset_[slot] = 0U;
                scaleStride_[slot] = 8U;
            }
            // The first half can enter Vector while MTE2 reads the second half.
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6 + slot);
        }
    }

    template<class Resource>
    __aicore__ inline void Produce(Resource& resource, uint32_t rows,
        uint32_t l1Offset, uint32_t flag, uint32_t kvSlot,
        uint32_t firstHalf = 0U, uint32_t endHalf = 2U)
    {
        if (AscendC::GetSubBlockIdx() != 1U - kvSlot) { return; }
        if (firstHalf == 0U) {
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(flag);
        }
        auto dst = resource.l1Buf.template GetBufferByByte<bfloat16_t>(l1Offset);
        const uint32_t paddedRows = (rows + 15U) / 16U * 16U;
        for (uint32_t slot = firstHalf; slot < endHalf; ++slot) {
            if (count_[slot] == 0U) { continue; }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6 + slot);
            for (uint32_t start = 0; start < count_[slot]; start += OUTPUT_ROWS) {
                const uint32_t bank = start / OUTPUT_ROWS;
                const uint32_t n = AscendC::Std::min(count_[slot] - start, OUTPUT_ROWS);
                const uint32_t event = EVENT_ID6 + bank;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
                auto scales = scale_[slot][scaleOffset_[slot] + start * scaleStride_[slot]];
                KvDequant::DequantNz(reinterpret_cast<__ubuf__ bfloat16_t*>(output_[bank].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float8_e4m3_t*>(input_[slot][start * HEAD_DIM].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(scales.GetPhyAddr()), n, scaleStride_[slot], OUTPUT_NZ_ROWS);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
                const uint32_t copyRows = (n + 15U) / 16U * 16U;
                const AscendC::DataCopyParams scatter{HEAD_DIM / 16U, static_cast<uint16_t>(copyRows),
                    static_cast<uint16_t>(OUTPUT_NZ_ROWS - copyRows), static_cast<uint16_t>(paddedRows - copyRows)};
                AscendC::DataCopy(dst[(slot * CHUNK_ROWS + start) * 16U], output_[bank], scatter);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6 + slot);
            inputInFlight_[slot] = true;
        }
        if (endHalf == 2U) {
            AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(flag);
        }
    }

    __aicore__ inline void Drain()
    {
        for (uint32_t slot = 0; slot < 2U; ++slot) {
            DrainInput(slot);
        }
    }
private:
    __aicore__ inline void DrainInput(uint32_t slot)
    {
        if (inputInFlight_[slot]) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6 + slot);
            inputInFlight_[slot] = false;
        }
    }
    AscendC::LocalTensor<uint8_t> input_[2];
    AscendC::LocalTensor<float> scale_[2];
    AscendC::LocalTensor<bfloat16_t> output_[2];
    uint32_t count_[2] = {0U, 0U};
    uint32_t scaleOffset_[2] = {0U, 0U};
    uint32_t scaleStride_[2] = {0U, 0U};
    bool inputInFlight_[2] = {false, false};
};
// Dedicated producers read a whole page tile: AIV0 handles K and AIV1 V.
// Both still run their original Q-row portions of Softmax and Rescale.
constexpr uint32_t WHOLE_ELEMS = 128U * HEAD_DIM;
constexpr uint32_t WHOLE_NZ_ROWS = 129U;
constexpr uint32_t WHOLE_SCALE_BYTES = 128U * 32U;
constexpr uint32_t WHOLE_STAGING_BYTES = WHOLE_ELEMS + WHOLE_SCALE_BYTES + WHOLE_NZ_ROWS * HEAD_DIM * 2U;
static_assert(fa950_plan::MakeUbPlan(HEAD_DIM, 1, 32).total + WHOLE_STAGING_BYTES <=
              fa950_plan::UB_CAPACITY, "Whole-page mixed-KV producer must fit");

constexpr uint32_t WHOLE_INPUT_STAGE_BYTES = WHOLE_ELEMS + WHOLE_SCALE_BYTES;
static_assert(fa950_plan::MakeUbPlan(HEAD_DIM, 1, 16, 16).total +
              3U * WHOLE_INPUT_STAGE_BYTES + 2U * WHOLE_NZ_ROWS * HEAD_DIM * 2U <=
              fa950_plan::UB_CAPACITY, "16 rows per AIV permit three inputs and two NZ outputs");

static_assert(fa950_plan::MakeUbPlan(HEAD_DIM, 1, 32, 32).total +
              3U * WHOLE_INPUT_STAGE_BYTES + WHOLE_NZ_ROWS * HEAD_DIM * 2U <=
              fa950_plan::UB_CAPACITY, "32 rows per AIV permit three inputs and one NZ output");

class KvDequantWhole {
public:
    template<class Resource>
    __aicore__ inline KvDequantWhole(Resource& resource, uint32_t ubOffset, bool deepPipeline, bool twoOutputs)
        : deepPipeline_(deepPipeline), twoOutputs_(twoOutputs)
    {
        for (uint32_t slot = 0; slot < 3U; ++slot) {
            const uint32_t base = ubOffset + (deepPipeline_ ? slot : 0U) * WHOLE_INPUT_STAGE_BYTES;
            input_[slot] = resource.ubBuf.template GetBufferByByte<uint8_t>(base);
            scale_[slot] = resource.ubBuf.template GetBufferByByte<float>(base + WHOLE_ELEMS);
        }
        for (uint32_t slot = 0; slot < 2U; ++slot) {
            output_[slot] = resource.ubBuf.template GetBufferByByte<bfloat16_t>(
                ubOffset + (deepPipeline_ ? 3U : 1U) * WHOLE_INPUT_STAGE_BYTES
                + (twoOutputs_ ? slot : 0U) * WHOLE_NZ_ROWS * HEAD_DIM * 2U);
        }
    }

    __aicore__ inline void Prefetch(GM_ADDR cache, GM_ADDR scales,
        AscendC::GlobalTensor<int32_t> pages, uint32_t head, uint32_t heads,
        uint64_t pageStride, uint64_t tokenStride, uint64_t headStride,
        uint32_t tile, uint32_t rows, uint32_t slot)
    {
        if (AscendC::GetSubBlockIdx() != 1U - slot) { return; }
        const uint32_t bank = deepPipeline_ ? writeSequence_++ % 3U : 0U;
        const uint32_t start = 0U;
        // slot 1 is K (AIV0); slot 0 is V (AIV1). Only the owner signals READY for its tile.
        count_[bank] = rows;
        if (count_[bank] == 0U) { return; }
        // Input/scales are dead as soon as the previous VF has consumed them.
        // Do not hold the next GM read until the independent NZ output DMA ends.
        DrainInput(bank);
        AscendC::GlobalTensor<uint8_t> src;
        AscendC::GlobalTensor<float> scaleGm;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(cache));
        scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
        const uint64_t page = static_cast<uint64_t>(pages.GetValue(tile));
        AscendC::DataCopyExtParams copy{static_cast<uint16_t>(count_[bank]), HEAD_DIM,
            static_cast<uint32_t>(tokenStride - HEAD_DIM), 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(input_[bank], src[page * pageStride + head * headStride + start * tokenStride], copy, pad);
        AscendC::DataCopyPadExtParams<float> scalePad{false, 0, 0, 0.0f};
        if (heads <= 4U) {
            AscendC::DataCopyExtParams scaleCopy{1, count_[bank] * heads * 4U, 0, 0, 0};
            AscendC::DataCopyPad(scale_[bank], scaleGm[(page * 128U + start) * heads], scaleCopy, scalePad);
            scaleOffset_[bank] = head;
            scaleStride_[bank] = heads;
        } else {
            AscendC::DataCopyExtParams scaleCopy{static_cast<uint16_t>(count_[bank]), 4U,
                (heads - 1U) * 4U, 0, 0};
            AscendC::DataCopyPad(scale_[bank], scaleGm[(page * 128U + start) * heads + head], scaleCopy, scalePad);
            scaleOffset_[bank] = 0U;
            scaleStride_[bank] = 8U;
        }
        // Publish before the other slot's GM read is enqueued: V's VF must
        // not depend on K's read merely because both inputs are prefetched.
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(InputEvent(bank));
    }

    template<class Resource>
    __aicore__ inline void Produce(Resource& resource, uint32_t rows,
        uint32_t l1Offset, uint32_t flag, uint32_t slot)
    {
        // Only the actual producer participates in this L1 buffer's handshake.
        if (AscendC::GetSubBlockIdx() != 1U - slot) { return; }
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(flag);
        {
            const uint32_t sequence = readSequence_++;
            const uint32_t bank = deepPipeline_ ? sequence % 3U : 0U;
            const uint32_t outputBank = twoOutputs_ ? sequence % 2U : 0U;
            // InitSyncFlags seeds these credits; each producer (including
            // InitOutputs) consumes one and returns it after its MTE3 read.
            // Wait on Vector, not Scalar, so issue can run ahead of the DMA.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6 + outputBank);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(InputEvent(bank));
            auto scales = scale_[bank][scaleOffset_[bank]];
            KvDequant::DequantNz(reinterpret_cast<__ubuf__ bfloat16_t*>(output_[outputBank].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float8_e4m3_t*>(input_[bank].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(scales.GetPhyAddr()), count_[bank], scaleStride_[bank], WHOLE_NZ_ROWS);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(InputEvent(bank));
            inputInFlight_[bank] = true;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6 + outputBank);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6 + outputBank);
            const uint32_t copyRows = (count_[bank] + 15U) / 16U * 16U;
            const uint32_t paddedRows = (rows + 15U) / 16U * 16U;
            const AscendC::DataCopyParams scatter{HEAD_DIM / 16U, static_cast<uint16_t>(copyRows),
                static_cast<uint16_t>(WHOLE_NZ_ROWS - copyRows), static_cast<uint16_t>(paddedRows - copyRows)};
            auto dst = resource.l1Buf.template GetBufferByByte<bfloat16_t>(l1Offset);
            AscendC::DataCopy(dst, output_[outputBank], scatter);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6 + outputBank);
        }
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(flag);
    }

    __aicore__ inline void Drain()
    {
        for (uint32_t bank = 0; bank < 3U; ++bank) {
            DrainInput(bank);
        }
        // ReleaseSyncFlags consumes the final MTE3->V credits for both outputs.
    }
private:
    // 0/1 and 4/5 are reserved by existing mask/epilogue operations.
    __aicore__ inline uint32_t InputEvent(uint32_t bank) const
    {
        return bank < 2U ? 2U + bank : 4U + bank;
    }
    __aicore__ inline void DrainInput(uint32_t slot)
    {
        if (inputInFlight_[slot]) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(InputEvent(slot));
            inputInFlight_[slot] = false;
        }
    }
    bool deepPipeline_;
    bool twoOutputs_;
    uint32_t writeSequence_ = 0U;
    uint32_t readSequence_ = 0U;
    AscendC::LocalTensor<uint8_t> input_[3];
    AscendC::LocalTensor<float> scale_[3];
    AscendC::LocalTensor<bfloat16_t> output_[2];
    uint32_t count_[3] = {0U, 0U};
    uint32_t scaleOffset_[3] = {0U, 0U};
    uint32_t scaleStride_[3] = {0U, 0U};
    bool inputInFlight_[3] = {false, false};
};
} // namespace fa950_mixed
#endif
