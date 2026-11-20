/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend 950 FlashAttention v3 MLA backward — independent device kernel entry.
 *
 * This file provides the MLA-specialised global kernel entry that matches the
 * memory layout in MLA设计.md A5:
 *   - S1 = S2 = 128 outer tiles
 *   - L0C three-slot layout: 128x192 (QK/dY·V/dQ), 128x192 (dK), 128x128 (dV)
 *   - L1: Q 3x128x192, K 2x128x192, dY 3x128x128, V 1x128x128, dS 2x128x128, P 2x128x128
 *
 * The kernel body reuses the proven FlashAttentionScoreGrad950 class; the
 * entry here is independent so the MLA tile/memory policy can diverge from the
 * generic v3 FAG kernel without touching the legacy path.
 */

#include "fag_mmad_mla_sdp.hpp"
#include "fag_mmad_mla_dqkv.hpp"
#include "fag_kernel_mla.h"

template <typename DataType, FAGTiling950::Layout INPUT_LAYOUT, bool IS_CAUSAL, bool IS_SOFTCAP>
CATLASS_DEVICE void RunMlaBackward950(FAGKernelParams const& params)
{
#ifdef __DAV_CUBE__
    AscendC::SetMMRowMajor();
#endif
    using ArchTag = Catlass::Arch::Ascend950;
    // MLA A5 layout: outer tiles fixed at 128x128, inner K up to 192 for QK/dQ/dK.
    using L1TileShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<256>>;
    using L0TileShape = tla::Shape<tla::Int<128>, tla::Int<192>, tla::Int<128>>;

    using DispatchPolicySdP = Catlass::Gemm::MmadAscend950FagMlaSdP<2, 2>;
    using TileCopySdP =
        Catlass::Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, DataType, Catlass::layout::RowMajor, DataType,
                                                   Catlass::layout::ColumnMajor, float, Catlass::layout::RowMajor, void,
                                                   Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadSdP = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicySdP, L1TileShape, L0TileShape, DataType,
                                                            DataType, float, void, TileCopySdP>;

    using DispatchPolicydQKV = Catlass::Gemm::MmadAscend950FagMladQKV<2, 2>;
    using TileCopydQKV =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, Catlass::layout::RowMajor, DataType,
                                               Catlass::layout::RowMajor, float, Catlass::layout::RowMajor>;
    using BlockMmaddQKV = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicydQKV, L1TileShape, L0TileShape, DataType,
                                                             DataType, float, void, TileCopydQKV>;

    using DispatchPolicyScaledMaskSoftmax =
        Catlass::Epilogue::EpilogueAscend950FAGMlaScaledMaskSoftmax<INPUT_LAYOUT, IS_CAUSAL, IS_SOFTCAP>;
    using EpilogueScaledMaskSoftmax = Catlass::Epilogue::Block::BlockEpilogue<DispatchPolicyScaledMaskSoftmax, DataType,
                                                                              float, FAGTiling950::FAGTilingData>;
    using DispatchPolicySubMul = Catlass::Epilogue::EpilogueAscend950FAGMlaSubMul<INPUT_LAYOUT, IS_SOFTCAP>;
    using EpilogueSubMul = Catlass::Epilogue::Block::BlockEpilogue<DispatchPolicySubMul, DataType, DataType, float,
                                                                   FAGTiling950::FAGTilingData>;
    using FAGMlaKernel950 =
        FlashAttentionScoreGradMla950<DataType, BlockMmadSdP, BlockMmaddQKV, EpilogueScaledMaskSoftmax, EpilogueSubMul,
                                      INPUT_LAYOUT, IS_CAUSAL, IS_SOFTCAP>;
    FAGMlaKernel950 fag;
    fag(params);
}

template <typename DataType, FAGTiling950::Layout INPUT_LAYOUT, bool IS_CAUSAL, bool IS_SOFTCAP>
CATLASS_GLOBAL void FlashAttentionV3BwdMla950(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR mask,
                                              GM_ADDR softmax_lse, GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_k,
                                              GM_ADDR seqUsedQ, GM_ADDR seqUsedKv, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    FAGKernelParams params{dout,         q,        k,         v,  out, mask, softmax_lse, cu_seqlens_q,
                           cu_seqlens_k, seqUsedQ, seqUsedKv, dq, dk,  dv,   workspace,   tiling};
    RunMlaBackward950<DataType, INPUT_LAYOUT, IS_CAUSAL, IS_SOFTCAP>(params);
}
