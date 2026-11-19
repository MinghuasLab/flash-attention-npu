/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Host launch ABI for the D>256 128x256 partial-resident bwd kernel.
 * Defined in fag_bwd_d512_partial.cpp; declared for mha_bwd dispatch only.
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include "acl/acl.h"

struct FagD512PartialLaunchArgs {
    aclrtStream aclStream{nullptr};
    uint32_t launchBlockDim{0};

    uint8_t* dout{nullptr};
    uint8_t* q{nullptr};
    uint8_t* k{nullptr};
    uint8_t* v{nullptr};
    uint8_t* out{nullptr};
    uint8_t* mask{nullptr};
    uint8_t* softmaxLse{nullptr};
    uint8_t* cuSeqQ{nullptr};
    uint8_t* cuSeqKv{nullptr};
    uint8_t* seqUsedQ{nullptr};
    uint8_t* seqUsedKv{nullptr};
    uint8_t* dq{nullptr};
    uint8_t* dk{nullptr};
    uint8_t* dv{nullptr};
    uint8_t* workspace{nullptr};
    uint8_t* tiling{nullptr};

    // 0=fp16, 1=bf16
    uint32_t dtypeCode{0};
    // FAGTiling950::Layout underlying value
    uint32_t layoutCode{0};
    bool isCausal{false};
    bool isSoftcap{false};
};

// This struct is the only type crossing the host/partial-TU boundary.  Being
// trivially copyable and standard-layout is what guarantees that the field
// offsets written by mha_bwd.cpp match the ones read by fag_bwd_d512_partial.cpp.
static_assert(std::is_trivially_copyable<FagD512PartialLaunchArgs>::value,
              "FagD512PartialLaunchArgs must stay trivially copyable (cross-TU launch ABI)");
static_assert(std::is_standard_layout<FagD512PartialLaunchArgs>::value,
              "FagD512PartialLaunchArgs must stay standard-layout (field offsets are part of the ABI)");

void launch_fag_d512_partial(const FagD512PartialLaunchArgs& args);
