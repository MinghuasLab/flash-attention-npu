/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Public ND BSND/TND adapter around the migrated fixed-source numerical kernel.
#include "kernel_operator.h"
#include "mla_dispatch.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla_int8_kernel.hpp"
#include "mla_metadata_consumer.h"
using namespace AscendC;

namespace flash_attn::mla::Int8 {
template <bool Tnd, bool Fd, bool Balance>
__aicore__ inline void Execute(const LaunchArgs& a, const Int8TilingParams& p)
{
    TPipe pipe;
    MlaInt8Kernel<Int8Type<Tnd, Fd, Balance>> op;
    op.SetInt32Lengths((GM_ADDR)a.lengths);
    if constexpr (Tnd)
        op.SetPublicQuery((GM_ADDR)a.prefix);
    op.SetPublicPartials((GM_ADDR)a.accum, (GM_ADDR)a.accumLse);
    op.Init((GM_ADDR)a.qv, (GM_ADDR)a.v, (GM_ADDR)a.q, (GM_ADDR)a.k, (GM_ADDR)a.mask, (GM_ADDR)a.table, (GM_ADDR)a.out,
            (GM_ADDR)a.lse, (GM_ADDR)a.workspace, &p, &pipe);
    op.SetCacheLayout(a.cache);
    op.InitQuant((GM_ADDR)a.qs, (GM_ADDR)a.ks, (GM_ADDR)a.vs);
    op.Process();
    op.InitPublicEmptyTail();
}
template <bool Tnd>
__aicore__ inline void Dispatch(const LaunchArgs& a)
{
    SetSyncBaseAddr(a.ffts);
    Int8TilingParams p{};
    uint32_t mode = 0;
    if (!LoadCheckedSchedule<CheckedSchedule>(a.checked, 3, p, &mode))
        return;
    if (!a.config.batch || !p.baseParams.qSeqSize)
        return;
    p.baseParams.scaleValue = a.scale;
    p.baseParams.softmaxLseFlag = a.lse != 0;
    switch (mode & ~1U) {
    case 0:
        Execute<Tnd, false, false>(a, p);
        break;
    case 2:
        Execute<Tnd, true, false>(a, p);
        break;
    case 4:
        Execute<Tnd, false, true>(a, p);
        break;
    case 6:
        Execute<Tnd, true, true>(a, p);
        break;
    default:
        Trap();
    }
}
} // namespace flash_attn::mla::Int8
#endif

__global__ __aicore__ __mix__(1, 2) void mla_int8_bsnd(flash_attn::mla::Int8::LaunchArgs a)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    flash_attn::mla::Int8::Dispatch<false>(a);
#endif
}
__global__ __aicore__ __mix__(1, 2) void mla_int8_tnd(flash_attn::mla::Int8::LaunchArgs a)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    flash_attn::mla::Int8::Dispatch<true>(a);
#endif
}

namespace flash_attn::mla::Int8 {
aclError Launch(const LaunchArgs& a)
{
    if (a.prefix)
        mla_int8_tnd<<<a.config.cores, nullptr, a.stream>>>(a);
    else
        mla_int8_bsnd<<<a.config.cores, nullptr, a.stream>>>(a);
    return ACL_SUCCESS;
}
} // namespace flash_attn::mla::Int8
