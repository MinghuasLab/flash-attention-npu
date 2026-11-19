/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend950 FlashAttention v4 MLA backward block dispatch policies.
 *
 * L1 layout matches MLA设计.md A5:
 *   P  : 2 slots x 128 x 128 x sizeof(DataType)
 *   dS : 2 slots x 128 x 128 x sizeof(DataType)
 *   Q  : 3 slots x 128 x 192 x sizeof(DataType)
 *   K  : 2 slots x 128 x 192 x sizeof(DataType)
 *   dY : 3 slots x 128 x 128 x sizeof(DataType)
 *   V  : 1 slot  x 128 x 128 x sizeof(DataType)
 */

#ifndef FLASH_ATTN_NPU_ASCEND950_V4_FAG_BLOCK_MLA_H
#define FLASH_ATTN_NPU_ASCEND950_V4_FAG_BLOCK_MLA_H

#include "catlass/arch/arch.hpp"

namespace Catlass::Gemm {

// All five GEMMs share the physical L0A/B double buffers and their cursors.
struct Ascend950FagMlaL0ABCursor {
    uint32_t a = 0;
    uint32_t b = 0;
};

struct Ascend950FagMlaL0CLayout {
    // Three fixed slots matching MLA design.md A5:
    //   slot 0: 128 x 192 fp32 = 96 KiB (QK / dYV / dQ)
    //   slot 1: 128 x 192 fp32 = 96 KiB (dK)
    //   slot 2: 128 x 128 fp32 = 64 KiB (dV)
    static constexpr uint32_t L0C_BUF_SIZE = 64 * 1024; // legacy field kept for template compatibility
    static constexpr uint32_t L0C_SLOT_NUM = 3;
    static constexpr uint32_t SLOT_SDP_0 = 0;
    static constexpr uint32_t SLOT_DK = 1;
    static constexpr uint32_t SLOT_DV = 2;
    static constexpr uint32_t SLOT_DQ = SLOT_SDP_0;
    static constexpr uint32_t L0C_SDP_SLOT_NUM = 1;

    static constexpr uint32_t L0COffset(uint32_t slot)
    {
        if (slot == 0)
            return 0;
        if (slot == 1)
            return 96 * 1024;
        if (slot == 2)
            return 192 * 1024;
        return 0;
    }

    static constexpr uint32_t L0CSize(uint32_t slot)
    {
        if (slot == 0 || slot == 1)
            return 96 * 1024;
        if (slot == 2)
            return 64 * 1024;
        return 0;
    }
};

struct Ascend950FagMlaL1Layout {
    static constexpr uint32_t BASE = 128;
    static constexpr uint32_t QK_HEAD_DIM = 192;
    static constexpr uint32_t V_HEAD_DIM = 128;

    static constexpr uint32_t TASK_PINGPONG = 2;    // P/dS and generic kernel ping-pong depth
    static constexpr uint32_t QDY_BUFFER_DEPTH = 3; // Q/dY triple-buffer depth for MLA

    // P / dS ping-pong slots (managed by the kernel wrapper, not by BlockMmadTla).
    static constexpr uint32_t SLOT_P = 0;
    static constexpr uint32_t SLOT_DS = 1;

    // Cube working slots after Mm12L1Offset().
    static constexpr uint32_t SLOT_RES_KT0 = 2; // Complete 128 x 192 resident K buffer 0
    static constexpr uint32_t SLOT_RES_KT1 = 3; // Complete 128 x 192 resident K buffer 1
    static constexpr uint32_t SLOT_RES_VT = 4;  // V^T resident, 128 x 128
    static constexpr uint32_t SLOT_Q0 = 5;
    static constexpr uint32_t SLOT_Q1 = 6;
    static constexpr uint32_t SLOT_Q2 = 7;
    static constexpr uint32_t SLOT_DY0 = 8;
    static constexpr uint32_t SLOT_DY1 = 9;
    static constexpr uint32_t SLOT_DY2 = 10;
    static constexpr uint32_t SLOT_COUNT = 11;

    // Legacy event IDs (kept for API compatibility; inner-core sync now uses mutex).
    static constexpr uint32_t L1_EVENT_KT0 = SLOT_RES_KT0;
    static constexpr uint32_t L1_EVENT_KT1 = SLOT_RES_KT1;
    static constexpr uint32_t L1_EVENT_COUNT = SLOT_COUNT + 1;

    // Mutex ID layout (Ascend950 provides 28 MutexIDs 0..27).
    // L1 slots reuse their slot index as mutex id: one mutex per buffer slot.
    static constexpr uint32_t L1_MUTEX_BASE = 0;
    static constexpr uint32_t L1_MUTEX_COUNT = SLOT_COUNT;
    // L0A/B/C ping-pong mutexes follow the L1 slots.
    static constexpr uint32_t L0A_MUTEX_BASE = L1_MUTEX_BASE + L1_MUTEX_COUNT; // 11
    static constexpr uint32_t L0B_MUTEX_BASE = L0A_MUTEX_BASE + 2;             // 13
    static constexpr uint32_t L0C_MUTEX_BASE = L0B_MUTEX_BASE + 2;             // 15
    static constexpr uint32_t TOTAL_MUTEX_COUNT = L0C_MUTEX_BASE + 3;          // 18
    static_assert(TOTAL_MUTEX_COUNT <= 28, "MLA backward mutex IDs exceed hardware limit");

    CATLASS_DEVICE
    static constexpr uint32_t L1Mutex(uint32_t slot)
    {
        return L1_MUTEX_BASE + slot;
    }

    CATLASS_DEVICE
    static constexpr uint32_t L0AMutex(uint32_t stage)
    {
        return L0A_MUTEX_BASE + stage;
    }

    CATLASS_DEVICE
    static constexpr uint32_t L0BMutex(uint32_t stage)
    {
        return L0B_MUTEX_BASE + stage;
    }

    CATLASS_DEVICE
    static constexpr uint32_t L0CMutex(uint32_t slot)
    {
        return L0C_MUTEX_BASE + slot;
    }

    // Return the L1 slot that holds Q/dY for a given task id (triple-buffer).
    CATLASS_DEVICE
    static constexpr uint32_t QSlot(uint32_t taskId)
    {
        const uint32_t idx = taskId % QDY_BUFFER_DEPTH;
        return (idx == 0) ? SLOT_Q0 : (idx == 1) ? SLOT_Q1 : SLOT_Q2;
    }

    CATLASS_DEVICE
    static constexpr uint32_t DySlot(uint32_t taskId)
    {
        const uint32_t idx = taskId % QDY_BUFFER_DEPTH;
        return (idx == 0) ? SLOT_DY0 : (idx == 1) ? SLOT_DY1 : SLOT_DY2;
    }
};

// Ascend950 MLA backward S / dP: S = Q * K^T, dP = dY * V^T.
// MLA always uses the unit-flag / MMLayoutTransform pipeline.
template <uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2>
struct MmadAscend950FagMlaSdP {
    using ArchTag = Arch::Ascend950;
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0AB_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = Ascend950FagMlaL0CLayout::L0C_SDP_SLOT_NUM;
    static constexpr uint32_t L0C_BUF_SIZE = Ascend950FagMlaL0CLayout::L0C_BUF_SIZE;
    static constexpr uint32_t BASE = 128;
};

// Ascend950 MLA backward dQKV: dq = dS * K, dk = dS^T * Q, dv = P^T * dY.
template <uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2>
struct MmadAscend950FagMladQKV {
    using ArchTag = Arch::Ascend950;
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0AB_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = Ascend950FagMlaL0CLayout::L0C_SLOT_NUM;
    static constexpr uint32_t L0C_BUF_SIZE = Ascend950FagMlaL0CLayout::L0C_BUF_SIZE;
    static constexpr uint32_t BASE = 128;
};

} // namespace Catlass::Gemm

#endif // FLASH_ATTN_NPU_ASCEND950_V4_FAG_BLOCK_MLA_H
