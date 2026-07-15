/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef INCLUDE_COMMON_HEADER_H
#define INCLUDE_COMMON_HEADER_H


#include <limits>
#include <type_traits>
#include <cstdint>
#include "kernel_operator.h"
#include "kernel_event.h"
#include "kernel_tensor.h"
#include "kernel_macros.h"


#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))

const int32_t TILING_PARA_NUM = 35;
const int32_t TILING_CORE_NUM = 3;
const int32_t TILING_SCALE_VALUE = 4;
const int32_t TILING_B = 5;
const int32_t TILING_T1 = 6;
const int32_t TILING_T2 = 7;
const int32_t TILING_N1 = 8;
const int32_t TILING_N2 = 8;
const int32_t TILING_G = 9;
const int32_t TILING_D = 10;
const int32_t TILING_Q_SIZE = 11;
const int32_t TILING_KV_SIZE = 12;

const int32_t TILING_DQ_WORKSPACE_OFFSET = 13;
const int32_t TILING_DK_WORKSPACE_OFFSET = 14;
const int32_t TILING_DV_WORKSPACE_OFFSET = 15;
const int32_t TILING_SFMG_WORKSPACE_OFFSET = 16;
const int32_t TILING_MM1_WORKSPACE_OFFSET = 17;
const int32_t TILING_MM2_WORKSPACE_OFFSET = 18;
const int32_t TILING_P_WORKSPACE_OFFSET = 0;
const int32_t TILING_DS_WORKSPACE_OFFSET = 1;

const int32_t CONST_2 = 2;
const int32_t CONST_8 = 8;
const int32_t CONST_UINT64_SIZE = 8;
const int32_t TILING_SOFTMAX_SIZE = sizeof(SoftMaxTiling);
const int32_t TILING_SOFTMAX_TILING_DATA = 19;
const int32_t TILING_SOFTMAX_GRAD_TILING_DATA = 19 + TILING_SOFTMAX_SIZE / CONST_UINT64_SIZE;

/////////////////////////////////////////////////////
// common struct
/////////////////////////////////////////////////////
constexpr int32_t BASE_BLOCK_LENGTH = 128;

struct AddrInfo {
    uint64_t left;
    uint64_t right;
    uint64_t out;
    int32_t kx = 0;
    int32_t ky = 0;
    int32_t lineStride = 0;
    bool lowerLeft;
    bool upperRight;
};

struct CubeAddrInfo {
    int32_t taskId;
    int32_t blockLength;
    AddrInfo addrInfo[16];
};

struct VecBlockInfo {
    int32_t SeqQIdx;
    int32_t SeqKIdx;
    int32_t batchIdx;
    int32_t nheadsIdx;
    int32_t nheadsKIdx;
    int32_t gIdx = 0;
    int32_t offset;
    int32_t lengthx = 0;
    int32_t lengthy = 0;
    int8_t mask = 0;
};

struct VecAddrInfo {
    int32_t taskId;
    int32_t blockLength = 0;
    VecBlockInfo VecBlkInfo[16];
};

constexpr uint32_t CUBE2VEC = 7;
constexpr uint32_t VEC2CUBE = 8;
constexpr uint32_t CUBE2POST = 9;

template <typename T>
constexpr T T_MAX = std::numeric_limits<T>::max();

template <typename T>
__aicore__ inline T Min(const T lhs, const T rhs)
{
    return lhs < rhs ? lhs : rhs;
}

enum class MaskType : uint32_t {
    NO_MASK = 0,
    MASK_CAUSAL = 1
};

enum class InputLayout : uint32_t {
    BSND = 0,
    TND = 1
};

#endif // COMMON_HEADER_H