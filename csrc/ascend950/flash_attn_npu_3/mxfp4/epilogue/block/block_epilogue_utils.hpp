/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* Modified by Minghua Shen, 2026. */

#ifndef BLOCK_EPILOGUE_UTILS_HPP
#define BLOCK_EPILOGUE_UTILS_HPP

#include "catlass/numeric_size.hpp"

namespace Catlass::Epilogue::Block {

namespace MXFP4 {
static constexpr uint32_t KB_BYTE = 1024;
static constexpr uint32_t QS_BASE_SIZE = 128;
static constexpr uint32_t KVS_BASE_SIZE = 256;
static constexpr uint32_t DATA_BLOCK_BYTE = 32;

// UB 分配
// mm1Res(S): single 64K, BUFCNT=2, FIXPIPE<->V
static constexpr uint32_t UB_S_BUF_CNT = 2;
static constexpr uint32_t UB_S_INNER_BUF_OFFSET = 256;
static constexpr uint32_t UB_S_BUF_OFFSET = 0;

// mm2Res(PV): 32K, BUFCNT=1, FIXPIPE<->V
static constexpr uint32_t UB_OTMP_BUF_OFFSET = 128 * KB_BYTE;

// LocalRowSum: 0.25K, BUFCNT=1,
static constexpr uint32_t UB_LOCAL_ROW_SUM_BUF_OFFSET = 160 * KB_BYTE;

// GlobalRowSum: 0.25K, BUFCNT=1, resident
static constexpr uint32_t UB_GLOBAL_ROW_SUM_BUF_OFFSET = 160 * KB_BYTE + 256;

// vec1Res(P): single 16K, BUFCNT=2, output buffer
static constexpr uint32_t UB_P_BUF_CNT = 2;
static constexpr uint32_t UB_P_INNER_BUF_OFFSET = 256;
static constexpr uint32_t UB_P_INNER_BUF_ELEMENT_OFFSET = 256;
static constexpr uint32_t UB_P_BUF_OFFSET = 160 * KB_BYTE + 512;

// attnTrans(space-reuse P): 18K, BUFCNT=1, output buffer
static constexpr uint32_t UB_O_TRANS_BUF_OFFSET = 160 * KB_BYTE + 512;

// pScale(reuse P): single 1.25K, BUFCNT=8, output buffer
static constexpr uint32_t UB_P_SCALE_CNT = 8;
static constexpr uint32_t UB_P_SCALE_BUF_SIZE = 1280;
static constexpr uint32_t UB_P_SCALE_BUF_OFFSET = 178 * KB_BYTE + 512;

// attentionOut: 32K, BUFCNT=1, output buffer
static constexpr uint32_t UB_O_BUF_OFFSET = 192 * KB_BYTE + 512;

// peerGlobalMax: single 0.25K, BUFCNT=4, L1 <-> UB
static constexpr uint32_t UB_PEER_GLOBAL_MAX_CNT = 4;
static constexpr uint32_t UB_PEER_GLOBAL_MAX_BUF_SIZE = 256;
static constexpr uint32_t UB_PEER_GLOBAL_MAX_BUF_OFFSET = 224 * KB_BYTE + 512;

// softmaxMax: 0.25K, BUFCNT=1, resident
static constexpr uint32_t UB_SOFTMAX_MAX_BUF_OFFSET = 225 * KB_BYTE + 512;

// LocalGroupMax: single 2K, BUFCNT=10, resident
static constexpr uint32_t UB_LOCAL_GROUP_MAX_CNT = 20;
static constexpr uint32_t UB_LOCAL_GROUP_MAX_BUF_SIZE = KB_BYTE;
static constexpr uint32_t UB_LOCAL_GROUP_MAX_BUF_OFFSET = 225 * KB_BYTE + 768;

// LocalGlobalMax: single 0.25K, BUFCNT=4, resident
static constexpr uint32_t UB_LOCAL_GLOBAL_MAX_CNT = 4;
static constexpr uint32_t UB_LOCAL_GLOBAL_MAX_BUF_SIZE = 256;
static constexpr uint32_t UB_LOCAL_GLOBAL_MAX_BUF_OFFSET = 245 * KB_BYTE + 768;

// updateScale(缩放因子dm): single 0.25K, BUFCNT=4, resident
static constexpr uint32_t UB_UPDATE_SCALE_CNT = 4;
static constexpr uint32_t UB_UPDATE_SCALE_BUF_SIZE = 256;
static constexpr uint32_t UB_UPDATE_SCALE_BUF_OFFSET = 246 * KB_BYTE + 768;

// Index: 0.25K, BUFCNT=1, resident
static constexpr uint32_t UB_INDEX_BUF_OFFSET = 247 * KB_BYTE + 768;

// vec同步eventID
static constexpr uint64_t SYNC_P_BUF0_FLAG = 0;
static constexpr uint64_t SYNC_P_BUF1_FLAG = 1;
static constexpr uint64_t SYNC_GMAX_UB_TO_L1_BUF0_FLAG = 2;
static constexpr uint64_t SYNC_GMAX_UB_TO_L1_BUF1_FLAG = 3;
static constexpr uint64_t SYNC_GMAX_UB_TO_L1_BUF2_FLAG = 4;
static constexpr uint64_t SYNC_GMAX_UB_TO_L1_BUF3_FLAG = 5;
static constexpr uint64_t SYNC_ATTNOUT_BUF_FLAG = 6;
} // namespace MXFP4

} // namespace Catlass::Epilogue::Block

#endif // BLOCK_EPILOGUE_UTILS_HPP
