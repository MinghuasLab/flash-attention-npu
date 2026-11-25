// Copyright (c) 2026. Adjacent-head BBND KV producer for single-token FlashDecode.
#pragma once
#include "kv_dequant.hpp"

namespace fa950_mixed {

// Whole pages rotate two input banks and share one NZ output. Half pages keep
// the two halves of a page in separate banks until both heads consume them.
class PairedDecodeKv {
    static constexpr uint32_t PAGE = 128U;
    static constexpr uint32_t WIDTH = 2U * HEAD_DIM;
    static constexpr uint32_t MAX_HEADS = 4U;
    template<uint32_t Rows> static constexpr uint32_t RAW_BYTES = Rows * WIDTH;
    template<uint32_t Rows> static constexpr uint32_t BANK_BYTES = RAW_BYTES<Rows> + Rows * MAX_HEADS * sizeof(float);
    using Resource = Catlass::Arch::ForwardResource950;

    Resource& resource_;
    uint32_t base_;

    template<uint32_t Rows>
    __aicore__ inline void Read(GM_ADDR cache, GM_ADDR descale, uint64_t token, uint32_t head,
                                uint32_t heads, uint32_t rows, uint32_t bank)
    {
        AscendC::GlobalTensor<uint8_t> src;
        AscendC::GlobalTensor<float> scales;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(cache));
        scales.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(descale));
        const uint32_t addr = base_ + bank * BANK_BYTES<Rows>;
        auto raw = resource_.ubBuf.template GetBufferByByte<uint8_t>(addr);
        auto scale = resource_.ubBuf.template GetBufferByByte<float>(addr + RAW_BYTES<Rows>);
        const uint64_t first = token * heads;
        AscendC::DataCopyPad(raw, src[(first + head) * HEAD_DIM],
            AscendC::DataCopyExtParams{static_cast<uint16_t>(rows), WIDTH, (heads - 2U) * HEAD_DIM, 0, 0},
            AscendC::DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
        AscendC::DataCopyPad(scale, scales[first], AscendC::DataCopyExtParams{1, rows * heads * 4U, 0, 0, 0},
                            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0F});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6 + bank);
    }

    template<uint32_t Rows>
    __aicore__ inline void Convert(uint32_t bank, uint32_t member, uint32_t head, uint32_t heads, uint32_t rows,
                                  AscendC::LocalTensor<bfloat16_t> nz)
    {
        const uint32_t addr = base_ + bank * BANK_BYTES<Rows>;
        auto raw = resource_.ubBuf.template GetBufferByByte<float8_e4m3_t>(addr);
        auto sc = resource_.ubBuf.template GetBufferByByte<float>(addr + RAW_BYTES<Rows>);
        KvDequant::DequantNz<WIDTH>(reinterpret_cast<__ubuf__ bfloat16_t*>(nz.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float8_e4m3_t*>(raw.GetPhyAddr()) + member * HEAD_DIM,
            reinterpret_cast<__ubuf__ float*>(sc.GetPhyAddr()) + head + member, rows, heads, Rows + 1U);
    }

    template<uint32_t Rows>
    __aicore__ inline void Write(AscendC::LocalTensor<bfloat16_t> dst, AscendC::LocalTensor<bfloat16_t> nz,
                                uint32_t rows, uint32_t pageRows)
    {
        const uint16_t blocks = (rows + 15U) / 16U * 16U;
        const uint16_t alignedPage = (pageRows + 15U) / 16U * 16U;
        AscendC::DataCopy(dst, nz, AscendC::DataCopyParams{HEAD_DIM / 16U, blocks,
            static_cast<uint16_t>(Rows + 1U - blocks), static_cast<uint16_t>(alignedPage - blocks)});
    }

public:
    __aicore__ inline PairedDecodeKv(Resource& resource, uint32_t base) : resource_(resource), base_(base) {}
    template<bool WholePage>
    __aicore__ inline void Produce(GM_ADDR cache, GM_ADDR descale, AscendC::GlobalTensor<int32_t> pages,
        uint32_t head, uint32_t heads, uint32_t splitStart, uint32_t pageCount, uint32_t kvLength,
        uint32_t seq, uint32_t l1Addr, uint32_t flag)
    {
        const uint32_t member = seq % 2U;
        const uint32_t page = seq / 2U;
        const uint32_t tile = splitStart + page;
        const uint32_t rows = AscendC::Std::min(PAGE, kvLength - tile * PAGE);
        auto dst = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(l1Addr);
        if constexpr (WholePage) {
            const uint32_t bank = page % 2U;
            if (member == 0U) {
                // Prime both banks, then enqueue the next page while this one is being consumed.
                for (uint32_t next = page == 0U ? 0U : page + 1U; next <= page + 1U && next < pageCount; ++next) {
                    const uint32_t input = next % 2U;
                    if (next >= 2U) { AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6 + input); }
                    const uint32_t nextTile = splitStart + next;
                    Read<PAGE>(cache, descale, static_cast<uint64_t>(pages.GetValue(nextTile)) * PAGE,
                               head, heads, AscendC::Std::min(PAGE, kvLength - nextTile * PAGE), input);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6 + bank);
            }
            auto nz = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(base_ + 2U * BANK_BYTES<PAGE>);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            Convert<PAGE>(bank, member, head, heads, rows, nz);
            if (member == 1U && page + 2U < pageCount) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6 + bank);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(flag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            Write<PAGE>(dst, nz, rows, rows);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        } else {
            const uint32_t halves = (rows + CHUNK_ROWS - 1U) / CHUNK_ROWS;
            if (member == 0U) {
                const uint64_t first = static_cast<uint64_t>(pages.GetValue(tile)) * PAGE;
                for (uint32_t i = 0; i < halves; ++i) {
                    Read<CHUNK_ROWS>(cache, descale, first + i * CHUNK_ROWS, head, heads,
                                     AscendC::Std::min(CHUNK_ROWS, rows - i * CHUNK_ROWS), i);
                }
            }
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(flag);
            for (uint32_t i = 0; i < halves; ++i) {
                auto nz = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
                    base_ + 2U * BANK_BYTES<CHUNK_ROWS> + i * NZ_ROWS * HEAD_DIM * sizeof(bfloat16_t));
                const uint32_t n = AscendC::Std::min(CHUNK_ROWS, rows - i * CHUNK_ROWS);
                const uint32_t event = EVENT_ID6 + i;
                if (member == 0U) { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event); }
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
                Convert<CHUNK_ROWS>(i, member, head, heads, n, nz);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
                Write<CHUNK_ROWS>(dst[i * CHUNK_ROWS * 16U], nz, n, rows);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
            }
            if (member == 1U) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
            }
        }
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(flag);
    }
    static constexpr uint32_t STATE_END = fa950_plan::MakeUbPlan(HEAD_DIM, 2, 16, 16).mask +
        16U * HEAD_DIM * sizeof(float) + 2U * 64U * sizeof(float);
    static_assert(STATE_END + 2U * BANK_BYTES<PAGE> + (PAGE + 1U) * HEAD_DIM * sizeof(bfloat16_t) <= fa950_plan::UB_CAPACITY);
    static_assert(STATE_END + 2U * BANK_BYTES<CHUNK_ROWS> + 2U * NZ_ROWS * HEAD_DIM * sizeof(bfloat16_t) <= fa950_plan::UB_CAPACITY);
};
} // namespace fa950_mixed
