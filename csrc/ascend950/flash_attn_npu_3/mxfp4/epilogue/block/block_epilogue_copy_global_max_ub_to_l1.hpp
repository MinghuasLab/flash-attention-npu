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

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_COPY_GLOBAL_MAX_UB_TO_L1
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_COPY_GLOBAL_MAX_UB_TO_L1

#include "catlass/numeric_size.hpp"
#include "../../arch/resource.hpp"
#include "../../epilogue/epilogue_dispatch_policy.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "block_epilogue_utils.hpp"

namespace Catlass::Epilogue::Block {

using namespace MXFP4Kernel;

template <class ElementLocalGlobalMax_, class LayoutLocalGlobalMax_>
class BlockEpilogue<EpilogueCopyGlobalMaxUbToL1Mx, ElementLocalGlobalMax_, LayoutLocalGlobalMax_> {
public:
    using DispatchPolicy = EpilogueCopyGlobalMaxUbToL1Mx;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementLocalGlobalMax = ElementLocalGlobalMax_;
    using LayoutLocalGlobalMax = LayoutLocalGlobalMax_;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource)
    {
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GLOBAL_MAX_CNT; i++) {
            localGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementLocalGlobalMax>(
                MXFP4::UB_LOCAL_GLOBAL_MAX_BUF_OFFSET + MXFP4::UB_LOCAL_GLOBAL_MAX_BUF_SIZE * i);
        }
    }
    __aicore__ inline ~BlockEpilogue()
    {}

    template <class TensorL1GlobalMax>
    __aicore__ inline void operator()(TensorL1GlobalMax const& l1GlocalMaxTensorTla, TileInfo const& tileInfo)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        AscendC::LocalTensor<ElementLocalGlobalMax>& localGlobalMax = localGlobalMaxUBTensor[tileInfo.tileMaxIdx];
        // 如果当前kvs只有一个softmax，就另外一个核不用做同步操作
        if (tileInfo.kvsFirstTileStartVecCore != subBlockIdx && tileInfo.isTileGoupFirstTile) {
            auto l1TlaTile = tla::GetTile(
                l1GlocalMaxTensorTla, tla::MakeCoord(1 - subBlockIdx, 0), tla::MakeShape(1, MXFP4::QS_BASE_SIZE));
            auto dstOffset = l1TlaTile.layout()(l1TlaTile.coord());
            uint16_t l1TlaTileMSize = static_cast<uint16_t>(tla::get<0>(l1TlaTile.shape()));
            uint32_t l1TlaTileNSize = tla::get<1>(l1TlaTile.shape());
            AscendC::DataCopyParams repeatParams;
            repeatParams.blockCount = l1TlaTileMSize;
            repeatParams.blockLen =
                static_cast<uint16_t>(l1TlaTileNSize * sizeof(ElementLocalGlobalMax) / MXFP4::DATA_BLOCK_BYTE);
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            AscendC::DataCopy(l1GlocalMaxTensorTla.data()[dstOffset], localGlobalMax, repeatParams);
            return;
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_GMAX_UB_TO_L1_BUF0_FLAG + tileInfo.tileMaxIdx);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_GMAX_UB_TO_L1_BUF0_FLAG + tileInfo.tileMaxIdx);

        auto l1TlaTile = tla::GetTile(
            l1GlocalMaxTensorTla, tla::MakeCoord(1 - subBlockIdx, 0),
            tla::MakeShape(1, static_cast<uint32_t>(MXFP4::QS_BASE_SIZE)));
        auto dstOffset = l1TlaTile.layout()(l1TlaTile.coord());

        uint16_t l1TlaTileMSize = static_cast<uint16_t>(tla::get<0>(l1TlaTile.shape()));
        uint32_t l1TlaTileNSize = tla::get<1>(l1TlaTile.shape());
        AscendC::DataCopyParams repeatParams;
        repeatParams.blockCount = l1TlaTileMSize;
        repeatParams.blockLen =
            static_cast<uint16_t>(l1TlaTileNSize * sizeof(ElementLocalGlobalMax) / MXFP4::DATA_BLOCK_BYTE);
        repeatParams.srcStride = 0;
        repeatParams.dstStride = 0;
        AscendC::DataCopy(l1GlocalMaxTensorTla.data()[dstOffset], localGlobalMax, repeatParams);
    }

private:
    AscendC::LocalTensor<ElementLocalGlobalMax>
        localGlobalMaxUBTensor[MXFP4::UB_LOCAL_GLOBAL_MAX_CNT]; // LocalGlobalMax
};

} // namespace Catlass::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_COPY_GLOBAL_MAX_UB_TO_L1
