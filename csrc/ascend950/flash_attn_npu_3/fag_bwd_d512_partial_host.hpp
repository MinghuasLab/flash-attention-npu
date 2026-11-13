/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Host-side helper for launching the D>256 128x256 partial-resident bwd kernel.
 */

#pragma once

#include <cstdint>

#include <ATen/ATen.h>

#include "fag_bwd_d512_partial.hpp"

namespace fag_d512 {
inline uint8_t* D512PartialPtr(const at::Tensor& t)
{
    return reinterpret_cast<uint8_t*>(t.data_ptr());
}

// Builds the launch ABI from the host dispatch state. Argument order mirrors FagD512PartialLaunchArgs.
inline FagD512PartialLaunchArgs MakeD512PartialArgs(aclrtStream stream, uint32_t launchBlockDim,
                                                    const at::Tensor& dout, const at::Tensor& q, const at::Tensor& k,
                                                    const at::Tensor& v, const at::Tensor& out, uint8_t* mask,
                                                    const at::Tensor& softmaxLse, uint8_t* cuSeqQ, uint8_t* cuSeqKv,
                                                    uint8_t* seqUsedQ, uint8_t* seqUsedKv, const at::Tensor& dq,
                                                    const at::Tensor& dk, const at::Tensor& dv,
                                                    const at::Tensor& workspace, const at::Tensor& tilingDevice,
                                                    bool isBf16, uint32_t layoutCode, bool isCausal, bool isSoftcap)
{
    FagD512PartialLaunchArgs args{};
    args.aclStream = stream;
    args.launchBlockDim = launchBlockDim;
    args.dout = D512PartialPtr(dout);
    args.q = D512PartialPtr(q);
    args.k = D512PartialPtr(k);
    args.v = D512PartialPtr(v);
    args.out = D512PartialPtr(out);
    args.softmaxLse = D512PartialPtr(softmaxLse);
    // Optional inputs: null when the corresponding feature is disabled.
    args.mask = mask;
    args.cuSeqQ = cuSeqQ;
    args.cuSeqKv = cuSeqKv;
    args.seqUsedQ = seqUsedQ;
    args.seqUsedKv = seqUsedKv;
    args.dq = D512PartialPtr(dq);
    args.dk = D512PartialPtr(dk);
    args.dv = D512PartialPtr(dv);
    args.workspace = D512PartialPtr(workspace);
    args.tiling = D512PartialPtr(tilingDevice);
    args.dtypeCode = isBf16 ? 1U : 0U;
    args.layoutCode = layoutCode;
    args.isCausal = isCausal;
    args.isSoftcap = isSoftcap;
    return args;
}
} // namespace fag_d512
