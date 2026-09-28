/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* Modified by Minghua Shen, 2026. */

#ifndef GEMM_BLOCK_COPY_GLOBAL_MAX_L1_TO_UB_HPP
#define GEMM_BLOCK_COPY_GLOBAL_MAX_L1_TO_UB_HPP

#include "catlass/numeric_size.hpp"
#include "../../arch/resource.hpp"
#include "../../gemm/gemm_dispatch_policy.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Block {

template <class ElementLocalGlobalMax_>
struct BlockMmadTla<CopyGlobalMaxL1ToUb, void, void, ElementLocalGlobalMax_, void, void, void, void, void> {
public:
    using DispatchPolicy = CopyGlobalMaxL1ToUb;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementLocalGlobalMax = ElementLocalGlobalMax_;

    static constexpr uint32_t L1_SINGLE_GLOBAL_MAX_SIZE = 128;

    static constexpr uint8_t VEC0 = 0;
    static constexpr uint8_t VEC1 = 1;
    static constexpr uint32_t SUB_VEC_NUM = 2;
    static constexpr uint32_t BLOCK_SIZE = 32;

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag>& resource)
    {
        // L1 区域顺序（与 QFA 一致）：P 数据 | P-scale | Q 数据 | Q-scale | KV(=V) 数据 | KV(=V)-scale
        // localGlobalMaxL1 = 256*4
        for (uint32_t i = 0; i < MXFP4::L1_LOCAL_GLOBAL_MAX_BUF_CNT; i++) {
            localGlobalMaxL1[i] = resource.l1Buf.template GetBufferByByte<ElementLocalGlobalMax>(
                MXFP4::L1_LOCAL_GLOBAL_MAX_BUF_OFFSET + MXFP4::L1_LOCAL_GLOBAL_MAX_BUF_SIZE * i);
        }
    }
    __aicore__ inline ~BlockMmadTla()
    {}

    template <class TensorGMaxUb, class TileInfoT>
    __aicore__ inline void operator()(TensorGMaxUb& gMaxUbTensor, TileInfoT& delay3TileInfo)
    {
        // AscendC::PRINTF("CopyGMaxL1ToUb \n");
        uint32_t blockCount = static_cast<uint32_t>(tla::get<0>(gMaxUbTensor.shape()));
        uint32_t blockLenPerSubVec = static_cast<uint32_t>(tla::get<1>(gMaxUbTensor.shape())) / SUB_VEC_NUM /
                                     BLOCK_SIZE * sizeof(ElementLocalGlobalMax);

        AscendC::LocalTensor<ElementLocalGlobalMax> localGlobalMax0 = localGlobalMaxL1[delay3TileInfo.tileMaxIdx];
        AscendC::LocalTensor<ElementLocalGlobalMax> localGlobalMax1 =
            localGlobalMaxL1[delay3TileInfo.tileMaxIdx][L1_SINGLE_GLOBAL_MAX_SIZE];
        AscendC::LocalTensor<ElementLocalGlobalMax> peerGlobalMax = gMaxUbTensor.data();

        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = static_cast<uint16_t>(blockCount); // 连续数据块个数为1
        intriParams.blockLen =
            static_cast<uint16_t>(blockLenPerSubVec); // 连续数据块长度，单位为DataBlock，此处长度为128个half元素
        intriParams.srcGap = 0;                       // 源操作数做连续搬运
        intriParams.dstGap = 0;                       // 目的操作数连续排布
        // CANN 9.1 does not expose the newer DataCopyL1ToUB wrapper used by
        // ops-transformer. Invoke its underlying Atlas-A5 builtin directly.
        copy_cbuf_to_ubuf(
            (__ubuf__ void*)peerGlobalMax.GetPhyAddr(), (__cbuf__ void*)localGlobalMax0.GetPhyAddr(), VEC0,
            intriParams.blockCount, intriParams.blockLen, intriParams.srcGap, intriParams.dstGap);
        copy_cbuf_to_ubuf(
            (__ubuf__ void*)peerGlobalMax.GetPhyAddr(), (__cbuf__ void*)localGlobalMax1.GetPhyAddr(), VEC1,
            intriParams.blockCount, intriParams.blockLen, intriParams.srcGap, intriParams.dstGap);
        AscendC::PipeBarrier<PIPE_MTE1>();
    }

protected:
    // MXFP4 专用：E8M0 scale 的 L1 buffer（V-scale 4 份，P-scale 20 份）+ MX 版 L1->L0 拷贝器
    AscendC::LocalTensor<half> localGlobalMaxL1[MXFP4::L1_LOCAL_GLOBAL_MAX_BUF_CNT];
};

} // namespace Catlass::Gemm::Block

#endif // GEMM_BLOCK_COPY_GLOBAL_MAX_L1_TO_UB_HPP
