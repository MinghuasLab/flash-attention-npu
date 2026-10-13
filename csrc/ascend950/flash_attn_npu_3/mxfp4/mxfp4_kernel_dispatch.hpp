/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef MXFP4_KERNEL_DISPATCH_HPP
#define MXFP4_KERNEL_DISPATCH_HPP

#include "kernel_operator.h"
#include "mxfp4_launch_args.hpp"

// Per-(dtype, layout, quant, sparsity) implementation, defined in
// mxfp4_kernel_dispatch_impl.hpp and explicitly instantiated in
// autogen/mxfp4_dispatch_<policy>_<layout>_<quant>_<dtype>.cpp.
template <typename OType, Mxfp4Layout Layout, Mxfp4QuantMode QuantMode, bool Sparse>
void launch_mxfp4_impl(const Mxfp4LaunchArgs& a);

// Runtime entry. Each launch_mxfp4_impl specialization lives in its own
// autogen translation unit.
inline void launch_mxfp4(const Mxfp4LaunchArgs& a, Mxfp4Layout layout, Mxfp4QuantMode quant_mode, bool output_bf16,
                         bool sparse)
{
    if (!sparse) {
        if (layout == Mxfp4Layout::TND) {
            if (quant_mode == Mxfp4QuantMode::OCP) {
                if (output_bf16) {
                    launch_mxfp4_impl<bfloat16_t, Mxfp4Layout::TND, Mxfp4QuantMode::OCP, false>(a);
                } else {
                    launch_mxfp4_impl<half, Mxfp4Layout::TND, Mxfp4QuantMode::OCP, false>(a);
                }
            } else {
                if (output_bf16) {
                    launch_mxfp4_impl<bfloat16_t, Mxfp4Layout::TND, Mxfp4QuantMode::CX, false>(a);
                } else {
                    launch_mxfp4_impl<half, Mxfp4Layout::TND, Mxfp4QuantMode::CX, false>(a);
                }
            }
        } else if (layout == Mxfp4Layout::BNSD) {
            if (quant_mode == Mxfp4QuantMode::OCP) {
                if (output_bf16) {
                    launch_mxfp4_impl<bfloat16_t, Mxfp4Layout::BNSD, Mxfp4QuantMode::OCP, false>(a);
                } else {
                    launch_mxfp4_impl<half, Mxfp4Layout::BNSD, Mxfp4QuantMode::OCP, false>(a);
                }
            } else {
                if (output_bf16) {
                    launch_mxfp4_impl<bfloat16_t, Mxfp4Layout::BNSD, Mxfp4QuantMode::CX, false>(a);
                } else {
                    launch_mxfp4_impl<half, Mxfp4Layout::BNSD, Mxfp4QuantMode::CX, false>(a);
                }
            }
        } else {
            if (quant_mode == Mxfp4QuantMode::OCP) {
                if (output_bf16) {
                    launch_mxfp4_impl<bfloat16_t, Mxfp4Layout::BSND, Mxfp4QuantMode::OCP, false>(a);
                } else {
                    launch_mxfp4_impl<half, Mxfp4Layout::BSND, Mxfp4QuantMode::OCP, false>(a);
                }
            } else {
                if (output_bf16) {
                    launch_mxfp4_impl<bfloat16_t, Mxfp4Layout::BSND, Mxfp4QuantMode::CX, false>(a);
                } else {
                    launch_mxfp4_impl<half, Mxfp4Layout::BSND, Mxfp4QuantMode::CX, false>(a);
                }
            }
        }
    } else {
        if (layout == Mxfp4Layout::TND) {
            if (quant_mode == Mxfp4QuantMode::OCP) {
                if (output_bf16) {
                    launch_mxfp4_impl<bfloat16_t, Mxfp4Layout::TND, Mxfp4QuantMode::OCP, true>(a);
                } else {
                    launch_mxfp4_impl<half, Mxfp4Layout::TND, Mxfp4QuantMode::OCP, true>(a);
                }
            } else {
                if (output_bf16) {
                    launch_mxfp4_impl<bfloat16_t, Mxfp4Layout::TND, Mxfp4QuantMode::CX, true>(a);
                } else {
                    launch_mxfp4_impl<half, Mxfp4Layout::TND, Mxfp4QuantMode::CX, true>(a);
                }
            }
        } else if (layout == Mxfp4Layout::BNSD) {
            if (quant_mode == Mxfp4QuantMode::OCP) {
                if (output_bf16) {
                    launch_mxfp4_impl<bfloat16_t, Mxfp4Layout::BNSD, Mxfp4QuantMode::OCP, true>(a);
                } else {
                    launch_mxfp4_impl<half, Mxfp4Layout::BNSD, Mxfp4QuantMode::OCP, true>(a);
                }
            } else {
                if (output_bf16) {
                    launch_mxfp4_impl<bfloat16_t, Mxfp4Layout::BNSD, Mxfp4QuantMode::CX, true>(a);
                } else {
                    launch_mxfp4_impl<half, Mxfp4Layout::BNSD, Mxfp4QuantMode::CX, true>(a);
                }
            }
        } else {
            if (quant_mode == Mxfp4QuantMode::OCP) {
                if (output_bf16) {
                    launch_mxfp4_impl<bfloat16_t, Mxfp4Layout::BSND, Mxfp4QuantMode::OCP, true>(a);
                } else {
                    launch_mxfp4_impl<half, Mxfp4Layout::BSND, Mxfp4QuantMode::OCP, true>(a);
                }
            } else {
                if (output_bf16) {
                    launch_mxfp4_impl<bfloat16_t, Mxfp4Layout::BSND, Mxfp4QuantMode::CX, true>(a);
                } else {
                    launch_mxfp4_impl<half, Mxfp4Layout::BSND, Mxfp4QuantMode::CX, true>(a);
                }
            }
        }
    }
}

#endif // MXFP4_KERNEL_DISPATCH_HPP
