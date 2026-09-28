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

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_REG_HIGH_PREC
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_REG_HIGH_PREC

#include "catlass/numeric_size.hpp"
#include "../../arch/resource.hpp"
#include "../../epilogue/epilogue_dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "block_epilogue_utils.hpp"
#include "vf/vf_rescale_o_dn.h"
#include "vf/vf_attn_out_transpose.h"

namespace Catlass::Epilogue::Block {
using namespace MXFP4Kernel;
template <
    class ElementO_, class ElementOTmp_, class ElementDm_, class TileCopy_, class OTmpSrcPos_, LseMode LSE_MODE_,
    LseFormat LSE_FORMAT_>
class BlockEpilogue<
    EpilogueAtlasA5RescaleOMx<LSE_MODE_, LSE_FORMAT_, true>, ElementO_, ElementOTmp_, ElementDm_, TileCopy_,
    OTmpSrcPos_> {
public:
    using DispatchPolicy = EpilogueAtlasA5RescaleOMx<LSE_MODE_, LSE_FORMAT_, true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = ElementO_;
    using ElementOTmp = ElementOTmp_;
    using ElementDm = ElementDm_;
    using ElementSum = ElementOTmp;
    using TileCopy = TileCopy_;
    using CopyUbToGmO = typename TileCopy::CopyUbToGmO;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource)
    {
        oTmpUBTensor = resource.ubBuf.template GetBufferByByte<ElementOTmp>(MXFP4::UB_OTMP_BUF_OFFSET);
        localRowSumUBTensor = resource.ubBuf.template GetBufferByByte<ElementSum>(MXFP4::UB_LOCAL_ROW_SUM_BUF_OFFSET);
        globalRowSumUBTensor = resource.ubBuf.template GetBufferByByte<ElementSum>(MXFP4::UB_GLOBAL_ROW_SUM_BUF_OFFSET);
        oTransUBTensor = resource.ubBuf.template GetBufferByByte<ElementO>(MXFP4::UB_O_TRANS_BUF_OFFSET);
        oUBTensor = resource.ubBuf.template GetBufferByByte<ElementOTmp>(MXFP4::UB_O_BUF_OFFSET);
        for (uint32_t i = 0; i < MXFP4::UB_UPDATE_SCALE_CNT; i++) {
            dmUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementDm>(
                MXFP4::UB_UPDATE_SCALE_BUF_OFFSET + MXFP4::UB_UPDATE_SCALE_BUF_SIZE * i);
        }
    }

    __aicore__ inline ~BlockEpilogue()
    {}

    template <class TensorO>
    __aicore__ inline void operator()(
        TensorO& gOTensor, Catlass::GemmCoord& actualOriShape, TaskInfo& taskInfo, TileInfo& tileInfo)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        AscendC::LocalTensor<ElementDm> dm = dmUBTensor[tileInfo.updateScaleIdx];

        if (tileInfo.curKvsTileLoopIdx / TILE_GROUP_N == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_ATTNOUT_BUF_FLAG);
            Mxfp4VF::ProcessUpdateOCallVF<false>(
                oUBTensor, oTmpUBTensor, dm, globalRowSumUBTensor, localRowSumUBTensor);
        } else {
            Mxfp4VF::ProcessUpdateOCallVF<true>(oUBTensor, oTmpUBTensor, dm, globalRowSumUBTensor, localRowSumUBTensor);
        }

        if (tileInfo.isLastKvsTile) {
            const uint32_t actQsTile = actualOriShape.n();
            const uint32_t splitMSizePerCore = (MXFP4::QS_BASE_SIZE + 1) / 2;
            uint32_t actMSizeThisCore = actQsTile < splitMSizePerCore ? actQsTile : splitMSizePerCore;
            if (subBlockIdx != VEC0) {
                actMSizeThisCore = actQsTile < splitMSizePerCore ? 0 : (actQsTile - splitMSizePerCore);
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (actMSizeThisCore != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF0_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF1_FLAG);
                AscendC::LocalTensor<ElementOTmp> outTensorFp32Ub =
                    oTransUBTensor.template ReinterpretCast<ElementOTmp>(); // 空间复用P, bf16 vf写出是强解释fp32
                Mxfp4VF::TransposeAttnOutCallVF<MXFP4::QS_BASE_SIZE, ElementO>(
                    outTensorFp32Ub, oUBTensor, globalRowSumUBTensor);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::LocalTensor<ElementO> oTransUBTensorB16 = oTransUBTensor.template ReinterpretCast<ElementO>();
                AscendC::LocalTensor<ElementO> oUBTensorB16 = oUBTensor.template ReinterpretCast<ElementO>();
                uint32_t embedColumnCnt = EMB_ALIGN128 + MXFP4::DATA_BLOCK_BYTE / sizeof(ElementO);
                // 搬回 oUBTensorB16
                AscendC::DataCopy(oUBTensorB16, oTransUBTensorB16, actMSizeThisCore * embedColumnCnt);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF0_FLAG);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF1_FLAG);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_ATTNOUT_BUF_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_ATTNOUT_BUF_FLAG);

                // copyUBOToGm
                uint32_t embdingSize = actualOriShape.m();
                uint32_t rowOffsetCurSubCore = subBlockIdx * splitMSizePerCore;
                uint32_t colNumCurSubCore = tla::get<1>(gOTensor.shape());
                auto gOTensorTlaTile = GetTile(
                    gOTensor, tla::MakeCoord(rowOffsetCurSubCore, 0),
                    tla::MakeShape(actMSizeThisCore, colNumCurSubCore));
                auto ubOLayoutTla = tla::MakeLayout(
                    tla::MakeShape(actMSizeThisCore, embdingSize), tla::MakeStride(embedColumnCnt, tla::Int<1>{}));
                auto ubOTensorTla = tla::MakeTensor(oUBTensorB16, ubOLayoutTla, Catlass::Arch::PositionUB{});
                copyUbToGmO(gOTensorTlaTile, ubOTensorTla);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_ATTNOUT_BUF_FLAG);
        }
    }

private:
    // UB tensors（FIXPIPE<->V 及常驻 buffer）
    AscendC::LocalTensor<ElementOTmp> oTmpUBTensor;                         // mm2Res(PV)
    AscendC::LocalTensor<ElementSum> localRowSumUBTensor;                   // LocalRowSum
    AscendC::LocalTensor<ElementSum> globalRowSumUBTensor;                  // GlobalRowSum
    AscendC::LocalTensor<ElementO> oTransUBTensor;                          // attnTrans(空间复用P)
    AscendC::LocalTensor<ElementOTmp> oUBTensor;                            // attentionOut
    AscendC::LocalTensor<ElementDm> dmUBTensor[MXFP4::UB_UPDATE_SCALE_CNT]; // updateScale
    CopyUbToGmO copyUbToGmO;
    static constexpr uint32_t VEC0 = 0;
    static constexpr uint32_t EMB_ALIGN128 = 128;
};

} // namespace Catlass::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_REG_HIGH_PREC
