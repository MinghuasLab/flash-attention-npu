// Copyright (c) 2026, Minghua Shen.
//
// Host/device boundary of the 950 backward launch of the v4 API extension.
// mha_bwd.cpp is compiled as part of the host TU flash_api.cpp
// (torch >= 2.14 headers need C++20, which the bisheng ASC frontend cannot
// parse), so it fills Bwd950LaunchArgs with plain pointers and calls
// bwd_dispatch_950(), defined in bwd_launch.cpp (ASC TU), which owns the
// FlashAttentionV3Bwd950 and FlashAttentionV3BwdMla950 template instantiations.
// The D>256 partial-resident kernel keeps its own boundary in
// fag_bwd_d512_partial_host.hpp and does not use this struct.
#pragma once

#include <cstdint>
#include <type_traits>
#include "acl/acl.h"

namespace fa950 {
enum class Bwd950Layout : int32_t {
    BSND = 0,
    TND = 1,
    BNSD = 2,
};

struct Bwd950LaunchArgs {
    uint32_t launch_cores;
    aclrtStream stream;
    bool is_bf16;
    Bwd950Layout layout;
    bool has_attn_mask;
    bool deterministic;
    bool has_softcap;
    // General kernel only: true selects SINGLE_VEC (NO_SPLIT Fixpipe + AIV0-only
    // handshake), from the tiling's shortQ / shortBiDir flags.
    bool single_vec;
    // Selects the MLA-absorbed kernel family; single_vec is unused there.
    bool is_mla;
    uint8_t* dout;
    uint8_t* q;
    uint8_t* k;
    uint8_t* v;
    uint8_t* out;
    uint8_t* mask; // nullptr when !has_attn_mask
    uint8_t* softmax_lse;
    uint8_t* cu_q;       // nullptr when not varlen
    uint8_t* cu_k;       // nullptr when not varlen
    uint8_t* seq_used_q; // may be nullptr
    uint8_t* seq_used_k; // may be nullptr
    uint8_t* dq;
    uint8_t* dk;
    uint8_t* dv;
    uint8_t* workspace;
    uint8_t* tiling;
};

// The field offsets written by mha_bwd.cpp (host frontend) must match the ones
// read by bwd_launch.cpp (ASC frontend).
static_assert(std::is_trivially_copyable<Bwd950LaunchArgs>::value,
              "Bwd950LaunchArgs must stay trivially copyable (cross-frontend launch ABI)");
static_assert(std::is_standard_layout<Bwd950LaunchArgs>::value,
              "Bwd950LaunchArgs must stay standard-layout (field offsets are part of the ABI)");

void bwd_dispatch_950(const Bwd950LaunchArgs& a);
} // namespace fa950
