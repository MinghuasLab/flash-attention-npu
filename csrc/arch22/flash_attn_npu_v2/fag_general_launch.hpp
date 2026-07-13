#ifndef FAG_GENERAL_LAUNCH_HPP
#define FAG_GENERAL_LAUNCH_HPP

#include "runtime/rt_ffts.h"

template <typename LayoutTag>
inline void LaunchFAGGeneralKernel(
    bool is_bf16,
    bool is_causal,
    bool deterministic,
    uint32_t qk_headdim_kernel,
    uint32_t blockDim,
    aclrtStream aclStream,
    uint64_t fftsAddr,
    uint8_t *dOutDevice,
    uint8_t *qDevice,
    uint8_t *kDevice,
    uint8_t *vDevice,
    uint8_t *outDevice,
    uint8_t *attenMaskDevice,
    uint8_t *softMaxLseDevice,
    uint8_t *cuSeqQlenDevice,
    uint8_t *cuSeqKvlenDevice,
    uint8_t *dqDevice,
    uint8_t *dkDevice,
    uint8_t *dvDevice,
    uint8_t *workspaceDevice,
    uint8_t *tilingDevice)
{
    constexpr uint32_t kInputLayout = LayoutTag::value;
    auto fag_general_call = [=]() -> int {
        if (is_bf16) {
            if (is_causal) {
                if (deterministic) {
                    switch (qk_headdim_kernel) {
                        case 64:
                            ::FAGGeneral<DTemplateType::Aligned64, bfloat16_t, kInputLayout, 1, 0, 1><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 128:
                            ::FAGGeneral<DTemplateType::Aligned128, bfloat16_t, kInputLayout, 1, 0, 1><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 192:
                            ::FAGGeneral<DTemplateType::Aligned192, bfloat16_t, kInputLayout, 1, 0, 1><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 256:
                            ::FAGGeneral<DTemplateType::Aligned256, bfloat16_t, kInputLayout, 1, 0, 1><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        default:
                            break;
                    }
                } else {
                    switch (qk_headdim_kernel) {
                        case 64:
                            ::FAGGeneral<DTemplateType::Aligned64, bfloat16_t, kInputLayout, 1, 0, 0><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 128:
                            ::FAGGeneral<DTemplateType::Aligned128, bfloat16_t, kInputLayout, 1, 0, 0><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 192:
                            ::FAGGeneral<DTemplateType::Aligned192, bfloat16_t, kInputLayout, 1, 0, 0><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 256:
                            ::FAGGeneral<DTemplateType::Aligned256, bfloat16_t, kInputLayout, 1, 0, 0><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        default:
                            break;
                    }
                }
            } else {
                if (deterministic) {
                    switch (qk_headdim_kernel) {
                        case 64:
                            ::FAGGeneral<DTemplateType::Aligned64, bfloat16_t, kInputLayout, 0, 0, 1><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 128:
                            ::FAGGeneral<DTemplateType::Aligned128, bfloat16_t, kInputLayout, 0, 0, 1><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 192:
                            ::FAGGeneral<DTemplateType::Aligned192, bfloat16_t, kInputLayout, 0, 0, 1><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 256:
                            ::FAGGeneral<DTemplateType::Aligned256, bfloat16_t, kInputLayout, 0, 0, 1><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        default:
                            break;
                    }
                } else {
                    switch (qk_headdim_kernel) {
                        case 64:
                            ::FAGGeneral<DTemplateType::Aligned64, bfloat16_t, kInputLayout, 0, 0, 0><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 128:
                            ::FAGGeneral<DTemplateType::Aligned128, bfloat16_t, kInputLayout, 0, 0, 0><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 192:
                            ::FAGGeneral<DTemplateType::Aligned192, bfloat16_t, kInputLayout, 0, 0, 0><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 256:
                            ::FAGGeneral<DTemplateType::Aligned256, bfloat16_t, kInputLayout, 0, 0, 0><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        default:
                            break;
                    }
                }
            }
        } else {
            if (is_causal) {
                if (deterministic) {
                    switch (qk_headdim_kernel) {
                        case 64:
                            ::FAGGeneral<DTemplateType::Aligned64, half, kInputLayout, 1, 0, 1><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 128:
                            ::FAGGeneral<DTemplateType::Aligned128, half, kInputLayout, 1, 0, 1><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 192:
                            ::FAGGeneral<DTemplateType::Aligned192, half, kInputLayout, 1, 0, 1><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 256:
                            ::FAGGeneral<DTemplateType::Aligned256, half, kInputLayout, 1, 0, 1><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        default:
                            break;
                    }
                } else {
                    switch (qk_headdim_kernel) {
                        case 64:
                            ::FAGGeneral<DTemplateType::Aligned64, half, kInputLayout, 1, 0, 0><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 128:
                            ::FAGGeneral<DTemplateType::Aligned128, half, kInputLayout, 1, 0, 0><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 192:
                            ::FAGGeneral<DTemplateType::Aligned192, half, kInputLayout, 1, 0, 0><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 256:
                            ::FAGGeneral<DTemplateType::Aligned256, half, kInputLayout, 1, 0, 0><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        default:
                            break;
                    }
                }
            } else {
                if (deterministic) {
                    switch (qk_headdim_kernel) {
                        case 64:
                            ::FAGGeneral<DTemplateType::Aligned64, half, kInputLayout, 0, 0, 1><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 128:
                            ::FAGGeneral<DTemplateType::Aligned128, half, kInputLayout, 0, 0, 1><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 192:
                            ::FAGGeneral<DTemplateType::Aligned192, half, kInputLayout, 0, 0, 1><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 256:
                            ::FAGGeneral<DTemplateType::Aligned256, half, kInputLayout, 0, 0, 1><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        default:
                            break;
                    }
                } else {
                    switch (qk_headdim_kernel) {
                        case 64:
                            ::FAGGeneral<DTemplateType::Aligned64, half, kInputLayout, 0, 0, 0><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 128:
                            ::FAGGeneral<DTemplateType::Aligned128, half, kInputLayout, 0, 0, 0><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 192:
                            ::FAGGeneral<DTemplateType::Aligned192, half, kInputLayout, 0, 0, 0><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        case 256:
                            ::FAGGeneral<DTemplateType::Aligned256, half, kInputLayout, 0, 0, 0><<<blockDim, nullptr, aclStream>>>(
                                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                                cuSeqQlenDevice, cuSeqKvlenDevice,
                                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                            break;
                        default:
                            break;
                    }
                }
            }
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fag", fag_general_call);
}

#endif
