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

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_REG_LOW_PREC_FP16_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_REG_LOW_PREC_FP16_HPP

#include "catlass/numeric_size.hpp"
#include "../../arch/resource.hpp"
#include "../../epilogue/epilogue_dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "block_epilogue_utils.hpp"
#include "vf/vf_init_nd2nz_indexes_duplicate.h"
#include "vf/vf_softmax_dn_cast_nz_qs128_kvs256.h"
#include "vf/vf_softmax_dn_cast_nz_qs64_kvs256.h"
#include "vf/vf_mm1_res_pre_padding_align_kvs32_multi.h"
#include "vf/vf_mm1_res_pre_padding_align_kvs32_multi_qs64.h"
#include "vf/vf_softmax_dn_cast_nz_align_qs128_kvs32_multi.h"
#include "vf/vf_softmax_dn_cast_nz_align_qs128_kvs32.h"
#include "vf/vf_softmax_dn_cast_nz_align_qs64_kvs32_multi.h"
#include "vf/vf_softmax_dn_cast_nz_align_qs64_kvs32.h"

namespace Catlass::Epilogue::Block {

using namespace MXFP4Kernel;

template <MXQuantMode MX_QUANT_MODE_, class OutputType_, class LayoutS_, class ScaleType_>
class BlockEpilogue<EpilogueOnlineSoftmaxMx<true, MX_QUANT_MODE_>, OutputType_, Catlass::Gemm::GemmType<half, LayoutS_>,
                    ScaleType_> {
  public:
    using DispatchPolicy = EpilogueOnlineSoftmaxMx<true, MX_QUANT_MODE_>;
    static constexpr MXQuantMode MX_QUANT_MODE = DispatchPolicy::MX_QUANT_MODE;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementInput = half; // S
    using ElementMax = ElementInput;
    using ElementDisguiseP = uint8_t;
    using ElementIndex = uint8_t;
    using LayoutPL1 = typename OutputType_::Layout;
    using LayoutPUB = Catlass::layout::RowMajor;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource, ElementInput NEG_LOG2_CX)
    {
        // FIXPIPE<->V 区
        for (uint32_t i = 0; i < MXFP4::UB_S_BUF_CNT; i++) {
            sUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(MXFP4::UB_S_BUF_OFFSET +
                                                                                 MXFP4::UB_S_INNER_BUF_OFFSET * i);
        }

        // 输出缓冲区
        for (uint32_t i = 0; i < MXFP4::UB_P_BUF_CNT; i++) {
            pUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementDisguiseP>(MXFP4::UB_P_BUF_OFFSET +
                                                                                     MXFP4::UB_P_INNER_BUF_OFFSET * i);
        }

        // 常驻 buffer
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GROUP_MAX_CNT; i++) {
            localGroupMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementMax>(
                MXFP4::UB_LOCAL_GROUP_MAX_BUF_OFFSET + MXFP4::UB_LOCAL_GROUP_MAX_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GLOBAL_MAX_CNT; i++) {
            localGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementMax>(
                MXFP4::UB_LOCAL_GLOBAL_MAX_BUF_OFFSET + MXFP4::UB_LOCAL_GLOBAL_MAX_BUF_SIZE * i);
        }
        indexUBTensor = resource.ubBuf.template GetBufferByByte<ElementIndex>(MXFP4::UB_INDEX_BUF_OFFSET);

        NEG_LOG2_CX_ = NEG_LOG2_CX;
        //  Init Tensor
        Mxfp4VF::InitIndexesAndDuplicateCallVF<ElementMax>(indexUBTensor, localGlobalMaxUBTensor[0]);
    }
    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorL1P>
    __aicore__ inline void operator()(TensorL1P& l1PTensorTla, Catlass::GemmCoord& actualBlockShape,
                                      TileInfo const& tileInfo, TaskInfo const& taskInfo)
    {
        uint32_t spBuffIdx = GetSPBufferIdx(tileInfo.loop);
        // 跨 tile 同步: 每个 vec core 都需要一个跨 tile 的同步
        if (tileInfo.isKvsFirstTilePerCore) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_GMAX_UB_TO_L1_BUF0_FLAG + tileInfo.tileMaxIdx);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF0_FLAG + spBuffIdx);

        const uint32_t actKvsTile = actualBlockShape.m();
        const uint32_t actQsTile = actualBlockShape.n();
        const uint32_t groupMaxIdx = GetLocalGroupMaxBufIdx(tileInfo.loop);
        if (actKvsTile == MXFP4::KVS_BASE_SIZE) {
            if (taskInfo.qsActBaseTileAlign64 == MXFP4::QS_BASE_SIZE) {
                // qs=128, kvs=256 softmax
                Mxfp4VF::SoftmaxWithGroupMaxQs128Kvs256CallVF<MX_QUANT_MODE, true, ElementInput, ElementDisguiseP,
                                                              MXFP4::KVS_BASE_SIZE, MXFP4::QS_BASE_SIZE>(
                    pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                    localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_);
            } else {
                // qs=64, kvs=256 softmax
                Mxfp4VF::SoftmaxWithGroupMaxQs64Kvs256CallVF<MX_QUANT_MODE, true, ElementInput, ElementDisguiseP,
                                                             MXFP4::KVS_BASE_SIZE, MXFP4::QS_BASE_SIZE>(
                    pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                    localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_);
            }

        } else {
            // kvs padding 32 multi
            if (actKvsTile != tileInfo.kvsActBaseTileAlign32) {
                if (taskInfo.qsActBaseTileAlign64 == MXFP4::QS_BASE_SIZE) {
                    Mxfp4VF::Mm1ResPrePaddingAlignKvs32MultiCallVF<ElementInput>(
                        sUBTensor[spBuffIdx], static_cast<uint16_t>(actKvsTile),
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32));

                } else {
                    // qs=64
                    Mxfp4VF::Mm1ResPrePaddingAlignKvs32MultiQs64CallVF<ElementInput>(
                        sUBTensor[spBuffIdx], static_cast<uint16_t>(actKvsTile),
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32));
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            // softmax
            if (tileInfo.kvsActBaseTileAlign32 == MXFP4::DATA_BLOCK_BYTE) {
                if (taskInfo.qsActBaseTileAlign64 == MXFP4::QS_BASE_SIZE) {
                    // softmax padding 32, qs=128
                    Mxfp4VF::SoftmaxWithGroupMaxAlignQs128Kvs32CallVF<MX_QUANT_MODE, true, ElementInput,
                                                                      ElementDisguiseP>(
                        pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                        localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_);
                } else {
                    // softmax padding 32, qs=64
                    Mxfp4VF::SoftmaxWithGroupMaxAlignQs64Kvs32CallVF<MX_QUANT_MODE, true, ElementInput,
                                                                     ElementDisguiseP>(
                        pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                        localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_);
                }
            } else {
                if (taskInfo.qsActBaseTileAlign64 == MXFP4::QS_BASE_SIZE) {
                    // softmax padding 32 multi >= 64, qs=128
                    Mxfp4VF::SoftmaxWithGroupMaxAlignQs128Kvs32MultiCallVF<MX_QUANT_MODE, true, ElementInput,
                                                                           ElementDisguiseP, MXFP4::QS_BASE_SIZE>(
                        pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                        localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_,
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32),
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign64));
                } else {
                    // softmax padding 32 multi >= 64, qs=64
                    Mxfp4VF::SoftmaxWithGroupMaxAlignQs64Kvs32MultiCallVF<MX_QUANT_MODE, true, ElementInput,
                                                                          ElementDisguiseP, MXFP4::QS_BASE_SIZE>(
                        pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                        localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_,
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32),
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign64));
                }
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_P_BUF0_FLAG + spBuffIdx);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_P_BUF0_FLAG + spBuffIdx);

        // DataCopy UB -> L1 搬运
        auto ubPLayoutTla =
            tla::MakeLayout<ElementDisguiseP, LayoutPUB>(MXFP4::KVS_BASE_SIZE / 4, MXFP4::QS_BASE_SIZE * 4);
        auto ubPTensorTla = tla::MakeTensor(pUBTensor[0], ubPLayoutTla, Catlass::Arch::PositionUB{});

        CopyPUBToPL1(l1PTensorTla, ubPTensorTla, tileInfo, taskInfo, spBuffIdx);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF0_FLAG + spBuffIdx);
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void CopyPUBToPL1(TensorDst const& l1PTensorTla, TensorSrc const& ubPTensorTla,
                                        TileInfo const& tileInfo, TaskInfo const& taskInfo, uint32_t spBuffIdx)
    {
        uint32_t pUBTlaMSize = tla::get<0>(ubPTensorTla.shape());
        uint32_t pUBTlaNSize = tla::get<1>(ubPTensorTla.shape());

        if (tileInfo.kvsActBaseTileAlign32 == MXFP4::KVS_BASE_SIZE) {
            // qs=128, kvs=256 整块搬运
            AscendC::DataCopyParams repeatParams;
            if (taskInfo.qsActBaseTileAlign64 == MXFP4::QS_BASE_SIZE) {
                repeatParams.blockCount = static_cast<uint16_t>(pUBTlaMSize); // UB的物理P一整行256B, 包含P的逻辑视图4行
            } else {
                repeatParams.blockCount =
                    static_cast<uint16_t>(pUBTlaMSize / 2); // UB的物理P一整行256B, 包含P的逻辑视图4行
            }
            repeatParams.blockLen = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
            repeatParams.srcStride = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
            repeatParams.dstStride = 0;

            auto pUBTensorTlaTile =
                GetTile(ubPTensorTla, tla::MakeCoord(0u, spBuffIdx * MXFP4::UB_P_INNER_BUF_ELEMENT_OFFSET),
                        tla::MakeShape(pUBTlaMSize, pUBTlaNSize / 2));
            auto srcOffset = pUBTensorTlaTile.layout()(pUBTensorTlaTile.coord());

            AscendC::DataCopy(l1PTensorTla.data(), pUBTensorTlaTile.data()[srcOffset], repeatParams);

        } else {
            // 消去高位 低位中间的间隙搬运
            // 低位搬运
            auto pUBTensorTlaTile =
                GetTile(ubPTensorTla, tla::MakeCoord(0u, spBuffIdx * MXFP4::UB_P_INNER_BUF_ELEMENT_OFFSET),
                        tla::MakeShape(pUBTlaMSize, pUBTlaNSize / 2));
            auto srcOffset = pUBTensorTlaTile.layout()(pUBTensorTlaTile.coord());

            AscendC::DataCopyParams repeatParams1;
            repeatParams1.blockCount = static_cast<uint16_t>(
                tileInfo.kvsActBaseTileAlign32 /
                (pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE)); // UB的物理P一整行256B, 包含P的逻辑视图低位(高位)地址8行
            repeatParams1.blockLen = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
            repeatParams1.srcStride = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
            repeatParams1.dstStride = 0;

            auto pUBTensorTlaTile1 =
                tla::GetTile(ubPTensorTla, tla::MakeCoord(0u, spBuffIdx * MXFP4::UB_P_INNER_BUF_ELEMENT_OFFSET),
                             tla::MakeShape(pUBTlaMSize / 2, pUBTlaNSize / 2));
            auto srcOffset1 = pUBTensorTlaTile1.layout()(pUBTensorTlaTile1.coord());
            AscendC::DataCopy(l1PTensorTla.data(), ubPTensorTla.data()[srcOffset1], repeatParams1);

            if (taskInfo.qsActBaseTileAlign64 == MXFP4::QS_BASE_SIZE) {
                // 高位搬运
                AscendC::DataCopyParams repeatParams2;
                repeatParams2.blockCount = static_cast<uint16_t>(
                    tileInfo.kvsActBaseTileAlign32 /
                    (pUBTlaNSize / 2 /
                     MXFP4::DATA_BLOCK_BYTE)); // UB的物理P一整行256B, 包含P的逻辑视图高位(低位)地址8行
                repeatParams2.blockLen = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
                repeatParams2.srcStride = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
                repeatParams2.dstStride = 0;

                uint32_t pL1NHalfSize = tla::get<1, 0>(l1PTensorTla.shape());
                uint32_t xL1P = tileInfo.kvsActBaseTileAlign64;
                uint32_t yL1P = 0;
                if (tileInfo.kvsActBaseTileAlign64 == MXFP4::KVS_BASE_SIZE) {
                    xL1P = 0;
                    yL1P = pL1NHalfSize;
                }
                auto pUBTensorTlaTile2 = tla::GetTile(pUBTensorTlaTile1, tla::MakeCoord(pUBTlaMSize / 2, 0u),
                                                      tla::MakeShape(pUBTlaMSize / 2, pUBTlaNSize / 2)); // 遗留
                auto pl1TensorTlaTile = tla::GetTile(l1PTensorTla, tla::MakeCoord(xL1P, yL1P),
                                                     tla::MakeShape(MXFP4::KVS_BASE_SIZE, pL1NHalfSize));

                auto srcOffset2 = pUBTensorTlaTile2.layout()(pUBTensorTlaTile2.coord());
                auto dstOffset2 = pl1TensorTlaTile.layout()(pl1TensorTlaTile.coord());

                AscendC::DataCopy(pl1TensorTlaTile.data()[dstOffset2], pUBTensorTlaTile2.data()[srcOffset2],
                                  repeatParams2);
            }
        }
    }

    // buffer id 获取
    __aicore__ inline uint32_t GetSPBufferIdx(const uint32_t loop)
    {
        return loop / 2 % 2;
    }

    __aicore__ inline uint32_t GetLocalGroupMaxBufIdx(uint32_t loop)
    {
        return loop / 2 * 2 % MXFP4::UB_LOCAL_GROUP_MAX_CNT;
    }

  private:
    // UB tensors（FIXPIPE<->V 及常驻 buffer）
    AscendC::LocalTensor<ElementInput> sUBTensor[MXFP4::UB_S_BUF_CNT];                       // mm1Res(S)
    AscendC::LocalTensor<ElementDisguiseP> pUBTensor[MXFP4::UB_P_BUF_CNT];                   // vec1Res(P)
    AscendC::LocalTensor<ElementMax> localGroupMaxUBTensor[MXFP4::UB_LOCAL_GROUP_MAX_CNT];   // LocalGroupMax
    AscendC::LocalTensor<ElementMax> localGlobalMaxUBTensor[MXFP4::UB_LOCAL_GLOBAL_MAX_CNT]; // LocalGlobalMax
    AscendC::LocalTensor<ElementIndex> indexUBTensor;                                        // Index
    ElementInput NEG_LOG2_CX_;
};

} // namespace Catlass::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_REG_LOW_PREC_FP16_HPP
