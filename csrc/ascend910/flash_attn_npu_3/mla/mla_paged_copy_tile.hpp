/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Source: CANN ops-transformer
// ca18aee610729ca5270a44e2d84c751bc687fbf0, attention/common/op_kernel/memcopy/
// attn_copy_gm_to_l1.h::CopyKKropePAGmToL1 / CopyKvGmToL1::ProcessPageAttention.
// ND FP16/BF16, shared content KV, Hkv=1. Reuse the validated FA3 layout and
// the existing ND-to-NZ tile, including its 64-bit/zero-stride fallbacks.
// The caller validates the full descriptor, lengths and all effective pages
// before constructing buffers/events, and keeps these inputs immutable until
// completion. Coordinates must lie within that batch's effective KV length;
// the paired QK coordinates describe the same rows and may select different D.
// Events and L1 buffer lifetime remain the Block caller's responsibility.
#pragma once
#include "mla_copy_tile.hpp"
#include "mla_layout.h"
namespace flash_attn::mla::Tile {

template <typename InputT>
struct PagedCache {
    GlobalTensor<InputT> k, v;
    GlobalTensor<int32_t> pages;
    MlaPagedCacheLayout layout;

    __aicore__ inline void Init(GM_ADDR keyRope, GM_ADDR sharedKv, GM_ADDR table, const MlaPagedCacheLayout& descriptor)
    {
        static_assert(IsSameType<InputT, half>::value || IsSameType<InputT, bfloat16_t>::value);
        k.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(keyRope));
        v.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(sharedKv));
        pages.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table));
        layout = descriptor;
    }

    __aicore__ inline MlaPagedCopySegment Segment(uint32_t batch, uint32_t start, uint32_t remaining) const
    {
        MlaPagedCopySegment s;
        s.tableOffset = uint64_t(batch) * layout.tableRowStride + start / layout.pageSize;
        // Page IDs have already passed the device consumer's signed check.
        // Lookup once per page segment, shared by content K and RoPE K.
        s.physicalPage = static_cast<uint64_t>(pages.GetValue(s.tableOffset));
        s.pageRow = start % layout.pageSize;
        s.rowCount = layout.pageSize - s.pageRow;
        if (s.rowCount > remaining) {
            s.rowCount = remaining;
        }
        s.kOffset = s.physicalPage * layout.k.batchStride + s.pageRow * layout.k.rowStride;
        s.vOffset = s.physicalPage * layout.v.batchStride + s.pageRow * layout.v.rowStride;
        return s;
    }
};

template <typename InputT>
class CopyKKropePAGmToL1 {
  public:
    __aicore__ inline void operator()(L1Tensor<InputT>& content, L1Tensor<InputT>& rope, const PagedCache<InputT>& src,
                                      const KvCoord& cc, const KvCoord& rc)
    {
        uint32_t done = 0;
        while (done < cc.s2DealSize) {
            auto s = src.Segment(cc.batchIdx, cc.s2Idx + done, cc.s2DealSize - done);
            CopySingleMatrixNDToNZ(content.tensor[uint64_t(done) * 16], src.v[s.vOffset + cc.dIdx], s.rowCount,
                                   cc.dDealSize, src.layout.v.rowStride, content.rowCount);
            CopySingleMatrixNDToNZ(rope.tensor[uint64_t(done) * 16], src.k[s.kOffset + rc.dIdx], s.rowCount,
                                   rc.dDealSize, src.layout.k.rowStride, rope.rowCount);
            done += s.rowCount;
        }
    }
};

template <typename InputT>
class CopyPagedKvGmToL1 {
  public:
    __aicore__ inline void operator()(L1Tensor<InputT>& dst, const PagedCache<InputT>& src, const KvCoord& c)
    {
        uint32_t done = 0;
        while (done < c.s2DealSize) {
            auto s = src.Segment(c.batchIdx, c.s2Idx + done, c.s2DealSize - done);
            CopySingleMatrixNDToNZ(dst.tensor[uint64_t(done) * 16], src.v[s.vOffset + c.dIdx], s.rowCount, c.dDealSize,
                                   src.layout.v.rowStride, dst.rowCount);
            done += s.rowCount;
        }
    }
};
} // namespace flash_attn::mla::Tile
