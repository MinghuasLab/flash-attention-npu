/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef MXFP4_KERNEL_DISPATCH_IMPL_HPP
#define MXFP4_KERNEL_DISPATCH_IMPL_HPP

#include "mxfp4_kernel_dispatch.hpp"
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "mxfp4_kernel.hpp"
#endif

template <typename OType, Mxfp4Layout Layout, Mxfp4QuantMode QuantMode, bool Sparse>
__global__ __aicore__ __mix__(1,
                              2) void mxfp4_kernel(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR blockSparseMask,
                                                   GM_ADDR actualSeqLengths, GM_ADDR actualSeqLengthsKv,
                                                   GM_ADDR qDequantScale, GM_ADDR kDequantScale, GM_ADDR vDequantScale,
                                                   GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR lse, GM_ADDR tiling)
{
#if defined(__CCE_AICORE__)
    constexpr Mxfp4Kernel::Format kLayout = Layout == Mxfp4Layout::TND    ? Mxfp4Kernel::Format::TND
                                            : Layout == Mxfp4Layout::BNSD ? Mxfp4Kernel::Format::BNSD
                                                                          : Mxfp4Kernel::Format::BSND;
    constexpr auto kQuant =
        QuantMode == Mxfp4QuantMode::OCP ? Catlass::Epilogue::MXQuantMode::OCP : Catlass::Epilogue::MXQuantMode::CX;
    Mxfp4::Run<OType, kLayout, kQuant, Sparse>(query, key, value, blockSparseMask, actualSeqLengths, actualSeqLengthsKv,
                                               qDequantScale, kDequantScale, vDequantScale, attentionOut, workspace,
                                               lse, tiling);
#endif
}

template <typename OType, Mxfp4Layout Layout, Mxfp4QuantMode QuantMode, bool Sparse>
void launch_mxfp4_impl(const Mxfp4LaunchArgs& a)
{
    mxfp4_kernel<OType, Layout, QuantMode, Sparse><<<a.block_dim, nullptr, a.stream>>>(
        a.query, a.key, a.value, a.block_sparse_mask, a.actual_q_lengths, a.actual_kv_lengths, a.q_scale, a.k_scale,
        a.v_scale, a.output, a.workspace, a.lse, a.tiling);
}

#endif // MXFP4_KERNEL_DISPATCH_IMPL_HPP
