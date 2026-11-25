/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0.
// attention/common/op_kernel/memcopy/attn_copy_gm_to_l1.h:
// CopySingle/MultiMatrixNDToNZ, CopyQueryGmToL1::ProcessS1G,
// CopyKvGmToL1::ProcessContinuousOrTensorlist, specialized to FP16/BF16.
// FA3 BSND/TND adaptation: independent 64-bit batch/row/head strides, including
// read-only overlap. Keep the source S1G row order and ND-to-NZ fallbacks.
#pragma once
#include "catlass/gemm/tile/atlasa2/copy_l0c_to_gm.hpp"
#include "mla_layout.h"
namespace flash_attn::mla::Tile {
using namespace AscendC;
__aicore__ inline uint32_t Align(uint32_t n, uint32_t a)
{
    return (n + a - 1) / a * a;
}
template <typename InputT>
struct L1Tensor {
    LocalTensor<InputT> tensor;
    uint32_t rowCount;
};
struct QueryCoord {
    uint64_t tokenStart;
    uint32_t n2Idx, gS1Idx, dIdx, gS1DealSize, dDealSize;
    uint32_t batchIdx = 0;
};
struct KvCoord {
    uint64_t tokenStart;
    uint32_t n2Idx, s2Idx, dIdx, s2DealSize, dDealSize;
    uint32_t batchIdx = 0;
};
template <typename InputT, bool HasStrides = false>
struct DenseTensor {
    GlobalTensor<InputT> gmTensor;
    uint32_t sequence, heads, dim;
    MlaBsndTensorLayout layout;
    bool tokenAddressing;
    __aicore__ inline void Init(GM_ADDR ptr, uint32_t s, uint32_t h, uint32_t d,
                                const MlaBsndTensorLayout* strides = nullptr, bool isTnd = false)
    {
        gmTensor.SetGlobalBuffer((__gm__ InputT*)ptr);
        sequence = s;
        heads = h;
        dim = d;
        if constexpr (HasStrides) {
            layout = *strides;
            tokenAddressing = isTnd;
        }
    }
    __aicore__ inline uint64_t RowStride() const
    {
        if constexpr (HasStrides) {
            return layout.rowStride;
        }
        return uint64_t(heads) * dim;
    }
    __aicore__ inline uint64_t HeadStride() const
    {
        if constexpr (HasStrides) {
            return layout.headStride;
        }
        return dim;
    }
    __aicore__ inline uint64_t RowOffset(uint64_t tokenStart, uint32_t batch, uint32_t row) const
    {
        if constexpr (HasStrides) {
            // TND starts are allocation prefixes, including unused slots.
            // Mixed cache uses its own BSND batch/row strides instead.
            return tokenAddressing ? (tokenStart + row) * layout.rowStride
                                   : uint64_t(batch) * layout.batchStride + uint64_t(row) * layout.rowStride;
        }
        return (tokenStart + row) * heads * dim;
    }
};

template <typename InputT>
__aicore__ inline void CopySingleMatrixNDToNZ(LocalTensor<InputT> dst, GlobalTensor<InputT> src, uint32_t rows,
                                              uint32_t cols, uint64_t stride, uint32_t dstRows)
{
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = rows;
    p.dValue = cols;
    p.dstNzC0Stride = dstRows;
    p.dstNzNStride = 1;
    p.srcNdMatrixStride = 0;
    p.dstNzMatrixStride = 0;
    if (stride < 65536 && stride != 0) {
        p.srcDValue = static_cast<uint32_t>(stride);
        DataCopy(dst, src, p);
    } else {
        // Source CopySingleMatrixNDToNZ fallback, widened before arithmetic.
        // Also use one-row copies for broadcast, without encoding zero stride.
        p.nValue = 1;
        p.srcDValue = cols;
        for (uint32_t row = 0; row < rows; ++row) {
            DataCopy(dst[uint64_t(row) * (32 / sizeof(InputT))], src[uint64_t(row) * stride], p);
        }
    }
}

template <typename InputT>
__aicore__ inline void CopyMultiMatrixNDToNZ(LocalTensor<InputT> dst, GlobalTensor<InputT> src, uint32_t count,
                                             uint64_t srcMatrixStride, uint32_t dstMatrixStride, uint32_t rows,
                                             uint32_t cols, uint64_t stride, uint32_t dstRows)
{
    // A legal matrix stride does not imply a legal row stride in FA3 views.
    // Unlike the source's >65536 check, include 65536: Atlas A2 Nd2NzParams
    // stores srcNdMatrixStride in uint16_t and encodes that value as zero.
    if (srcMatrixStride >= 65536 || stride >= 65536 || srcMatrixStride == 0 || stride == 0) {
        uint64_t srcOffset = 0, dstOffset = 0;
        for (uint32_t i = 0; i < count; ++i) {
            CopySingleMatrixNDToNZ(dst[dstOffset], src[srcOffset], rows, cols, stride, dstRows);
            srcOffset += srcMatrixStride;
            dstOffset += dstMatrixStride;
        }
    } else {
        Nd2NzParams p;
        p.ndNum = count;
        p.nValue = rows;
        p.dValue = cols;
        p.srcDValue = static_cast<uint32_t>(stride);
        p.dstNzC0Stride = dstRows;
        p.dstNzNStride = 1;
        p.srcNdMatrixStride = static_cast<uint32_t>(srcMatrixStride);
        p.dstNzMatrixStride = dstMatrixStride;
        DataCopy(dst, src, p);
    }
}

template <typename InputT, bool HasStrides = false>
class CopyQueryGmToL1 {
  public:
    __aicore__ inline void operator()(L1Tensor<InputT>& dst, DenseTensor<InputT, HasStrides>& src, QueryCoord& c)
    {
        uint32_t sStart = c.gS1Idx / src.heads, gStart = c.gS1Idx % src.heads;
        uint32_t sEnd = (c.gS1Idx + c.gS1DealSize) / src.heads;
        uint32_t gEnd = (c.gS1Idx + c.gS1DealSize) % src.heads;
        uint64_t base = src.RowOffset(c.tokenStart, c.batchIdx, sStart) + c.dIdx;
        uint64_t strideS = src.RowStride(), strideG = src.HeadStride();
        if (src.heads == 1) {
            CopySingleMatrixNDToNZ(dst.tensor, src.gmTensor[base], sEnd - sStart, c.dDealSize, strideS, dst.rowCount);
            return;
        }
        uint32_t head = sStart == sEnd ? gEnd - gStart : src.heads - gStart;
        CopySingleMatrixNDToNZ(dst.tensor, src.gmTensor[base + gStart * strideG], head, c.dDealSize, strideG,
                               dst.rowCount);
        if (sEnd - sStart >= 1) {
            uint64_t gmOffset = base + strideS;
            uint64_t l1Offset = head * 16;
            if (sEnd - sStart > 1) {
                CopyMultiMatrixNDToNZ(dst.tensor[l1Offset], src.gmTensor[gmOffset], sEnd - sStart - 1, strideS,
                                      src.heads * 16, src.heads, c.dDealSize, strideG, dst.rowCount);
                gmOffset += (sEnd - sStart - 1) * strideS;
                l1Offset += (sEnd - sStart - 1) * src.heads * 16;
            }
            if (gEnd > 0) {
                CopySingleMatrixNDToNZ(dst.tensor[l1Offset], src.gmTensor[gmOffset], gEnd, c.dDealSize, strideG,
                                       dst.rowCount);
            }
        }
    }
};
template <typename InputT, bool HasStrides = false>
class CopyKvGmToL1 {
  public:
    __aicore__ inline void operator()(L1Tensor<InputT>& dst, DenseTensor<InputT, HasStrides>& src, KvCoord& c)
    {
        uint64_t offset = src.RowOffset(c.tokenStart, c.batchIdx, c.s2Idx) + c.dIdx;
        CopySingleMatrixNDToNZ(dst.tensor, src.gmTensor[offset], c.s2DealSize, c.dDealSize, src.RowStride(),
                               dst.rowCount);
    }
};

__aicore__ inline void StoreFp32(GlobalTensor<float> dst, LocalTensor<float> src, uint32_t m, uint32_t n,
                                 uint32_t stride)
{
    using GmType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    Catlass::Gemm::Tile::CopyL0CToGm<Catlass::Arch::AtlasA2, float, GmType> copy;
    copy(dst, src, Catlass::layout::RowMajor(m, n, stride), Catlass::layout::zN::template MakeLayout<float>(m, n), 3);
}
} // namespace flash_attn::mla::Tile
