/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef MXFP4_LAUNCH_ARGS_HPP
#define MXFP4_LAUNCH_ARGS_HPP

#include <cstdint>
#include "acl/acl.h"

enum class Mxfp4Layout : uint32_t {
    TND = 0,
    BNSD = 1,
    BSND = 2
};
enum class Mxfp4QuantMode : uint32_t {
    OCP = 0,
    CX = 1
};

struct Mxfp4LaunchArgs {
    uint32_t block_dim;
    aclrtStream stream;
    uint8_t* query;
    uint8_t* key;
    uint8_t* value;
    uint8_t* block_sparse_mask;
    uint8_t* actual_q_lengths;
    uint8_t* actual_kv_lengths;
    uint8_t* q_scale;
    uint8_t* k_scale;
    uint8_t* v_scale;
    uint8_t* output;
    uint8_t* workspace;
    uint8_t* lse;
    uint8_t* tiling;
};

#endif // MXFP4_LAUNCH_ARGS_HPP
