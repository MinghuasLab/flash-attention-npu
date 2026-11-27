// Copyright (c) 2026, Minghua Shen.
//
// ASC TU: the FlashAttentionV3Bwd950 / FlashAttentionV3BwdMla950 template
// instantiations and <<<>>> launches of the 950 backward path of the v4 API
// extension. The backward kernel templates live in the v3 directory (main
// keeps the 950 backward kernels in flash_attn_npu_3). mha_bwd.cpp is
// compiled as part of the host TU
// flash_api.cpp (torch >= 2.14 headers need C++20, which the bisheng ASC
// frontend cannot parse), so it marshals the tensors into the plain-pointer
// Bwd950LaunchArgs and calls bwd_dispatch_950(). The macros and the dtype /
// layout dispatch below are the ones that used to live in mha_bwd.cpp; only
// the operands changed from local variables to launch-args fields.

#include "acl/acl.h"
#include "bwd_launch.hpp"
#include "../flash_attn_npu_3/fag_kernel.cpp"
#include "../flash_attn_npu_3/fa_bwd_mla_kernel.cpp"

namespace fa950 {

#define LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, IS_ATTEN_MASK, IS_DETERMINISTIC, IS_SOFTCAP, SINGLE_VEC)                    \
    FlashAttentionV3Bwd950<DTYPE, FAGTiling950::Layout::INPUT_LAYOUT, IS_ATTEN_MASK, IS_DETERMINISTIC, IS_SOFTCAP,     \
                           SINGLE_VEC>                                                                                 \
        <<<a.launch_cores, nullptr, a.stream>>>(a.dout, a.q, a.k, a.v, a.out, a.mask, a.softmax_lse, a.cu_q, a.cu_k,   \
                                                a.seq_used_q, a.seq_used_k, a.dq, a.dk, a.dv, a.workspace, a.tiling)

#define DISPATCH_BWD950_FLAGS_SV(DTYPE, INPUT_LAYOUT, SINGLE_VEC)                                                      \
    do {                                                                                                               \
        if (a.has_attn_mask) {                                                                                         \
            if (a.deterministic) {                                                                                     \
                if (a.has_softcap) {                                                                                   \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, true, true, true, SINGLE_VEC);                                  \
                } else {                                                                                               \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, true, true, false, SINGLE_VEC);                                 \
                }                                                                                                      \
            } else {                                                                                                   \
                if (a.has_softcap) {                                                                                   \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, true, false, true, SINGLE_VEC);                                 \
                } else {                                                                                               \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, true, false, false, SINGLE_VEC);                                \
                }                                                                                                      \
            }                                                                                                          \
        } else {                                                                                                       \
            if (a.deterministic) {                                                                                     \
                if (a.has_softcap) {                                                                                   \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, false, true, true, SINGLE_VEC);                                 \
                } else {                                                                                               \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, false, true, false, SINGLE_VEC);                                \
                }                                                                                                      \
            } else {                                                                                                   \
                if (a.has_softcap) {                                                                                   \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, false, false, true, SINGLE_VEC);                                \
                } else {                                                                                               \
                    LAUNCH_BWD950(DTYPE, INPUT_LAYOUT, false, false, false, SINGLE_VEC);                               \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

#define DISPATCH_BWD950_FLAGS(DTYPE, INPUT_LAYOUT)                                                                     \
    do {                                                                                                               \
        if (a.single_vec) {                                                                                            \
            DISPATCH_BWD950_FLAGS_SV(DTYPE, INPUT_LAYOUT, true);                                                       \
        } else {                                                                                                       \
            DISPATCH_BWD950_FLAGS_SV(DTYPE, INPUT_LAYOUT, false);                                                      \
        }                                                                                                              \
    } while (0)

#define LAUNCH_BWD950_MLA(DTYPE, INPUT_LAYOUT, IS_ATTEN_MASK, IS_SOFTCAP)                                              \
    FlashAttentionV3BwdMla950<DTYPE, FAGTiling950::Layout::INPUT_LAYOUT, IS_ATTEN_MASK, IS_SOFTCAP>                    \
        <<<a.launch_cores, nullptr, a.stream>>>(a.dout, a.q, a.k, a.v, a.out, a.mask, a.softmax_lse, a.cu_q, a.cu_k,   \
                                                a.seq_used_q, a.seq_used_k, a.dq, a.dk, a.dv, a.workspace, a.tiling)

#define DISPATCH_BWD950_MLA_FLAGS(DTYPE, INPUT_LAYOUT)                                                                 \
    do {                                                                                                               \
        if (a.has_attn_mask) {                                                                                         \
            if (a.has_softcap) {                                                                                       \
                LAUNCH_BWD950_MLA(DTYPE, INPUT_LAYOUT, true, true);                                                    \
            } else {                                                                                                   \
                LAUNCH_BWD950_MLA(DTYPE, INPUT_LAYOUT, true, false);                                                   \
            }                                                                                                          \
        } else {                                                                                                       \
            if (a.has_softcap) {                                                                                       \
                LAUNCH_BWD950_MLA(DTYPE, INPUT_LAYOUT, false, true);                                                   \
            } else {                                                                                                   \
                LAUNCH_BWD950_MLA(DTYPE, INPUT_LAYOUT, false, false);                                                  \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

void bwd_dispatch_950(const Bwd950LaunchArgs& a)
{
    if (a.is_mla) {
        if (a.is_bf16) {
            if (a.layout == Bwd950Layout::TND) {
                DISPATCH_BWD950_MLA_FLAGS(bfloat16_t, TND);
            } else if (a.layout == Bwd950Layout::BNSD) {
                DISPATCH_BWD950_MLA_FLAGS(bfloat16_t, BNSD);
            } else {
                DISPATCH_BWD950_MLA_FLAGS(bfloat16_t, BSND);
            }
        } else {
            if (a.layout == Bwd950Layout::TND) {
                DISPATCH_BWD950_MLA_FLAGS(half, TND);
            } else if (a.layout == Bwd950Layout::BNSD) {
                DISPATCH_BWD950_MLA_FLAGS(half, BNSD);
            } else {
                DISPATCH_BWD950_MLA_FLAGS(half, BSND);
            }
        }
    } else if (a.is_bf16) {
        if (a.layout == Bwd950Layout::TND) {
            DISPATCH_BWD950_FLAGS(bfloat16_t, TND);
        } else if (a.layout == Bwd950Layout::BNSD) {
            DISPATCH_BWD950_FLAGS(bfloat16_t, BNSD);
        } else {
            DISPATCH_BWD950_FLAGS(bfloat16_t, BSND);
        }
    } else {
        if (a.layout == Bwd950Layout::TND) {
            DISPATCH_BWD950_FLAGS(half, TND);
        } else if (a.layout == Bwd950Layout::BNSD) {
            DISPATCH_BWD950_FLAGS(half, BNSD);
        } else {
            DISPATCH_BWD950_FLAGS(half, BSND);
        }
    }
}

} // namespace fa950

#undef DISPATCH_BWD950_FLAGS
#undef DISPATCH_BWD950_FLAGS_SV
#undef LAUNCH_BWD950
#undef DISPATCH_BWD950_MLA_FLAGS
#undef LAUNCH_BWD950_MLA
