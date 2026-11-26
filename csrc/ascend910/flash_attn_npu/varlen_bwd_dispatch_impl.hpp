// Copyright (c) 2026, Minghua Shen.

//
// Shared implementation of the v2 varlen backward FAGVarlenOpt dispatch. Included
// by the generated autogen/varlen_bwd_dispatch_<dtype>.cpp stubs, each of which
// explicitly instantiates launch_varlen_bwd_impl<DType> for its dtype only.
//
// The launch tree reproduces the exact dtype x causal combinations of the
// original mha_varlen_bwd (always TND layout), including the dead
// ENABLE_ASCENDC_DUMP path which is compiled out unless that macro is defined.
//
// ADAPTATION vs opt_compiler: main wraps the FAGVarlenOpt launch in an
// at_npu::native::OpCommand::RunOpApiV2("ascendc_fag", ...) context (for aicpu
// op tracking). That wrapper is preserved here — no semantic change.

#pragma once

#include "varlen_bwd_dispatch.hpp"

// Standard headers the CATLASS/FAG headers assume visible.
#include <algorithm>
#include <cstring>
#include <limits>

// The varlen-bwd launch is wrapped in an OpCommand context through the
// host-implemented bridge below; this kernel-side header must stay free of
// torch/torch_npu includes (the ASC frontend cannot parse torch >= 2.14's
// C++20 headers).
#include "../../op_command_bridge.hpp"

// mha_varlen_bwd.cpp provides the FAG::FAGVarlenOpt kernel template, the FAG
// namespace, and MaskType / InputLayout (via fag_common/common_header.h).
#include "mha_varlen_bwd.cpp"

template <typename DType>
void launch_varlen_bwd_impl(const VarlenBwdLaunchArgs& a)
{
    // Wrap the launch tree in an OpCommand context, matching main's
    // mha_varlen_bwd (RunOpApiV2("ascendc_fag", ...)).
    auto fag_varlen_opt_call = [=]() -> int {
        const uint32_t blockDim = a.blockDim;
        const aclrtStream aclStream = a.aclStream;
        const uint64_t fftsAddr = a.fftsAddr;
        const bool is_causal = a.is_causal;
        const bool is_softcap = a.is_softcap;
        const bool has_alibi = a.has_alibi;
        uint8_t* qDevice = a.qDevice;
        uint8_t* kDevice = a.kDevice;
        uint8_t* vDevice = a.vDevice;
        uint8_t* dOutDevice = a.dOutDevice;
        uint8_t* attenMaskDevice = a.attenMaskDevice;
        uint8_t* softMaxLseDevice = a.softMaxLseDevice;
        uint8_t* outDevice = a.outDevice;
        uint8_t* cuSeqQlenDevice = a.cuSeqQlenDevice;
        uint8_t* cuSeqKvlenDevice = a.cuSeqKvlenDevice;
        uint8_t* dqDevice = a.dqDevice;
        uint8_t* dkDevice = a.dkDevice;
        uint8_t* dvDevice = a.dvDevice;
        uint8_t* alibiSlopesDevice = a.alibiSlopesDevice;
        uint8_t* workspaceDevice = a.workspaceDevice;
        uint8_t* tilingDevice = a.tilingDevice;

#if defined(ENABLE_ASCENDC_DUMP)
        uint8_t* ptrDumpDevice{nullptr};
        aclCheck(aclrtMalloc(reinterpret_cast<void**>(&ptrDumpDevice), ALL_DUMPSIZE, ACL_MEM_MALLOC_HUGE_FIRST));
        FAG::FAGVarlenOpt<DType><<<blockDim, nullptr, aclStream>>>(
            fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr, nullptr,
            attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, cuSeqQlenDevice, cuSeqKvlenDevice, nullptr,
            nullptr, dqDevice, dkDevice, dvDevice, alibiSlopesDevice, workspaceDevice, tilingDevice, ptrDumpDevice);
        aclCheck(aclrtSynchronizeStream(aclStream));
        Adx::AdumpPrintWorkSpace(ptrDumpDevice, ALL_DUMPSIZE, aclStream, "device_fag");
        aclCheck(aclrtFree(ptrDumpDevice));
#else
        if (has_alibi) {
            if (is_softcap) {
                if (is_causal) {
                    FAG::FAGVarlenOpt<DType, MaskType::MASK_CAUSAL, InputLayout::TND, true, true>
                        <<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr,
                            nullptr, attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, cuSeqQlenDevice,
                            cuSeqKvlenDevice, nullptr, nullptr, dqDevice, dkDevice, dvDevice, alibiSlopesDevice,
                            workspaceDevice, tilingDevice, nullptr);
                } else {
                    FAG::FAGVarlenOpt<DType, MaskType::NO_MASK, InputLayout::TND, true, true>
                        <<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr,
                            nullptr, attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, cuSeqQlenDevice,
                            cuSeqKvlenDevice, nullptr, nullptr, dqDevice, dkDevice, dvDevice, alibiSlopesDevice,
                            workspaceDevice, tilingDevice, nullptr);
                }
            } else {
                if (is_causal) {
                    FAG::FAGVarlenOpt<DType, MaskType::MASK_CAUSAL, InputLayout::TND, false, true>
                        <<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr,
                            nullptr, attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, cuSeqQlenDevice,
                            cuSeqKvlenDevice, nullptr, nullptr, dqDevice, dkDevice, dvDevice, alibiSlopesDevice,
                            workspaceDevice, tilingDevice, nullptr);
                } else {
                    FAG::FAGVarlenOpt<DType, MaskType::NO_MASK, InputLayout::TND, false, true>
                        <<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr,
                            nullptr, attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, cuSeqQlenDevice,
                            cuSeqKvlenDevice, nullptr, nullptr, dqDevice, dkDevice, dvDevice, alibiSlopesDevice,
                            workspaceDevice, tilingDevice, nullptr);
                }
            }
        } else {
            if (is_softcap) {
                if (is_causal) {
                    FAG::FAGVarlenOpt<DType, MaskType::MASK_CAUSAL, InputLayout::TND, true, false>
                        <<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr,
                            nullptr, attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, cuSeqQlenDevice,
                            cuSeqKvlenDevice, nullptr, nullptr, dqDevice, dkDevice, dvDevice, alibiSlopesDevice,
                            workspaceDevice, tilingDevice, nullptr);
                } else {
                    FAG::FAGVarlenOpt<DType, MaskType::NO_MASK, InputLayout::TND, true, false>
                        <<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr,
                            nullptr, attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, cuSeqQlenDevice,
                            cuSeqKvlenDevice, nullptr, nullptr, dqDevice, dkDevice, dvDevice, alibiSlopesDevice,
                            workspaceDevice, tilingDevice, nullptr);
                }
            } else {
                if (is_causal) {
                    FAG::FAGVarlenOpt<DType, MaskType::MASK_CAUSAL, InputLayout::TND, false, false>
                        <<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr,
                            nullptr, attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, cuSeqQlenDevice,
                            cuSeqKvlenDevice, nullptr, nullptr, dqDevice, dkDevice, dvDevice, alibiSlopesDevice,
                            workspaceDevice, tilingDevice, nullptr);
                } else {
                    FAG::FAGVarlenOpt<DType, MaskType::NO_MASK, InputLayout::TND, false, false>
                        <<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr,
                            nullptr, attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, cuSeqQlenDevice,
                            cuSeqKvlenDevice, nullptr, nullptr, dqDevice, dkDevice, dvDevice, alibiSlopesDevice,
                            workspaceDevice, tilingDevice, nullptr);
                }
            }
        }
#endif
        return 0;
    };
    fa_bridge::run_op_api_v2("ascendc_fag", fag_varlen_opt_call);
}

// Each autogen TU ends with the explicit instantiation definition of its own
// dtype. Declaring both extern here keeps launch_varlen_bwd_dispatch() below from
// instantiating the two of them in every TU that includes this header,
// which would undo the per-TU split.
extern template void launch_varlen_bwd_impl<bfloat16_t>(const VarlenBwdLaunchArgs& a);
extern template void launch_varlen_bwd_impl<half>(const VarlenBwdLaunchArgs& a);

// Dtype-agnostic host entry (declared in varlen_bwd_dispatch.hpp): the
// is_bf16 selection happens here, device-side, so host TUs never name the
// ASC dtype types. gnu::used forces emission (no device TU calls it
// directly); references the per-dtype instantiations in the autogen TUs.
[[gnu::used]] inline void launch_varlen_bwd_dispatch(const VarlenBwdLaunchArgs& a)
{
    if (a.is_bf16) {
        launch_varlen_bwd_impl<bfloat16_t>(a);
    } else {
        launch_varlen_bwd_impl<half>(a);
    }
}
