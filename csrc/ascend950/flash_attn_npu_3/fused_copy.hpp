/*
 * Copyright (c) 2026, Minghua Shen.
 */
#pragma once

namespace Catlass::Gemm::Block {
// A segment can straddle the compact task's AIV boundary. Publish precisely
// its valid rows to each destination; aligned L0C padding is never exposed.
template <class TensorDst, class TensorSrc>
__aicore__ inline void CopyFusedSegmentToUB(TensorDst& dst, TensorSrc& src, uint32_t rowOffset, uint32_t rows,
                                            uint32_t totalRows, uint32_t colOffset, uint32_t cols)
{
    const uint32_t split = RoundUp(totalRows, 8U) / 2U;
    for (uint32_t sub = 0U; sub < 2U; ++sub) {
        const uint32_t base = sub == 0U ? 0U : split;
        const uint32_t limit = sub == 0U ? split : totalRows;
        const uint32_t begin = rowOffset > base ? rowOffset : base;
        const uint32_t end = rowOffset + rows < limit ? rowOffset + rows : limit;
        if (end > begin) {
            auto dstTile = GetTile(dst, tla::MakeCoord(begin - base, colOffset), tla::MakeShape(end - begin, cols));
            auto srcTile = GetTile(src, tla::MakeCoord(begin - rowOffset, 0), tla::MakeShape(end - begin, cols));
            Tile::CopyL0CToUBTla<Arch::Ascend950, decltype(srcTile), decltype(dstTile)> copy;
            copy(dstTile, srcTile, sub != 0U, 0);
        }
    }
}
} // namespace Catlass::Gemm::Block
