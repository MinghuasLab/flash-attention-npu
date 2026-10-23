/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once
#include "kernel_operator.h"
#include "mla_dispatch.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla_dispatch_impl.hpp"
namespace flash_attn::mla {
template <bool IsTnd>
__aicore__ inline MlaSplitKvMergeParams AutoMergeParams(const MlaPartitionView<MlaReader>& read,
                                                        const MlaAutoSplitBsndSchedule& s, const MlaContext& c,
                                                        const MlaAutoSplitBsndTask& t)
{
    MlaSplitKvMergeParams p{};
    const auto& a = s.used.tiling;
    p.batch = a.batch;
    p.sq = a.sq;
    p.heads = a.heads;
    p.batchIndex = t.batch;
    p.rowStart = t.rowBegin;
    p.rows = t.rowEnd - t.rowBegin;
    p.mBase = a.mBaseSize;
    p.splits = t.splitCount;
    p.queryLength = read(8 * t.batch + 6);
    p.kvLength = read(8 * t.batch + 7);
    if constexpr (IsTnd) {
        p.batch = 1;
        p.sq = c.totalQ;
        p.batchIndex = read(8 * t.batch);
    }
    p.maskKind = c.window.kind;
    p.left = c.window.left;
    p.right = c.window.right;
    auto geometry = t;
    auto_split_detail::Geometry<false>(read, s, c.window, geometry);
    auto_split_detail::Parts(s, geometry, UINT32_MAX, nullptr, p.begin, p.end);
    for (uint32_t split = 0; split < p.splits; ++split) {
        const uint32_t begin = p.begin[split] * a.s2BaseSize, end = p.end[split] * a.s2BaseSize;
        p.begin[split] = geometry.kvBegin > begin ? geometry.kvBegin : begin;
        p.end[split] = geometry.kvEnd < end ? geometry.kvEnd : end;
    }
    return p;
}
template <typename DType, bool IsTnd>
__aicore__ inline void AutoMerge(const MlaLaunchArgs& a, const MlaAutoSplitBsndSchedule& s,
                                 const MlaPartitionView<MlaReader>& read, AscendC::TPipe* pipe)
{
    const auto& c = a.context;
    if ASCEND_IS_AIV {
        const uint32_t owner = AscendC::GetBlockIdx();
        if (owner >= s.vectors) {
            AscendC::SyncAll();
            return;
        }
        const uint32_t base = auto_split_detail::Base(s.used.tiling.batch);
        const uint32_t firstHead = owner ? read(base + kMlaNativeVectorHeadEnd + owner - 1) : 0;
        const uint32_t firstPart = owner ? read(base + kMlaNativeVectorPartEnd + owner - 1) + 1 : 0;
        const uint32_t lastHead = read(base + kMlaNativeVectorHeadEnd + owner);
        const uint32_t lastPart = read(base + kMlaNativeVectorPartEnd + owner);
        pipe->Reset();
        Block::MlaSplitKvMerge<DType, false, IsTnd> block;
        block.InitBuffers(pipe);
        AscendC::ICachePreLoad(2);
        AscendC::SyncAll();
        // The validated vector endpoints already give this lane's contiguous
        // FD interval. Do not rediscover ownership for every other lane's row.
        for (uint32_t h = firstHead; h <= lastHead; ++h) {
            const uint32_t parts = h == lastHead ? lastPart + 1 : read(base + kMlaNativeFdParts + h);
            const uint32_t first = h == firstHead ? firstPart : 0;
            if (first >= parts)
                continue;
            MlaAutoSplitBsndMerge r{};
            AutoSplitBsndMerge(read, s, h, 0, r);
            MlaAutoSplitBsndTask t{};
            AutoSplitBsndTask(read, s, c.window, r.group, 0, t);
            auto p = AutoMergeParams<IsTnd>(read, s, c, t);
            block.Bind(p, (GM_ADDR)a.workspace + r.outOffset, (GM_ADDR)a.workspace + r.sumOffset,
                       (GM_ADDR)a.workspace + r.maxOffset, (GM_ADDR)a.out, (GM_ADDR)a.lse, (GM_ADDR)a.accum,
                       (GM_ADDR)a.accumLse, a.softcap);
            const uint32_t tile = block.TileRows();
            for (uint32_t part = first; part < parts; ++part) {
                const uint32_t end = p.rows < (part + 1) * 8 ? p.rows : (part + 1) * 8;
                for (uint32_t row = part * 8; row < end; row += tile)
                    block.Run(row, end - row < tile ? end - row : tile);
            }
        }
        block.Finish();
    }
}

} // namespace flash_attn::mla
#endif
namespace flash_attn::mla {
template <typename DType, bool IsTnd, bool Paged>
__global__ __aicore__ __mix__(1, 2) void CompressedAttention(MlaLaunchArgs a)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaReader read;
    MlaLayout l{};
    MlaAutoSplitBsndSchedule s{};
    if (!LoadMla<DType>(a, read, l, s))
        return;
    AscendC::SetSyncBaseAddr(a.ffts);
    AscendC::TPipe pipe;
    InitializeMlaOutputs<DType>(a);
    if (!s.groups || MlaEmptyKv(a.context))
        return;
    if ASCEND_IS_AIV {
        AscendC::SyncAll();
    }
    const MlaPartitionView<MlaReader> view{read, a.context, l};
    Kernel::MlaAutoSplit<DType, IsTnd, Paged> op;
    op(s, view, a.context.window, a.scale, (GM_ADDR)a.qv, (GM_ADDR)a.q, (GM_ADDR)a.v, (GM_ADDR)a.k, (GM_ADDR)a.out,
       (GM_ADDR)a.lse, (GM_ADDR)a.workspace, &pipe, (GM_ADDR)a.metadata + 4 * kMlaRecordHeaderWords, a.context.totalQ,
       a.inputs, a.context.layout == 1, a.softcap, (GM_ADDR)a.cache.table.address, &a.cache.layout);
    if (a.context.splits == 0)
        AutoMerge<DType, IsTnd>(a, s, view, &pipe);
#endif
}
template <typename DType>
aclError LaunchCompressed(const MlaLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    const uint32_t cores = a.context.physicalCores - a.context.margin;
    if (MlaTnd(a.context)) {
        if (a.context.pageSize)
            CompressedAttention<DType, true, true><<<cores, nullptr, a.stream>>>(a);
        else
            CompressedAttention<DType, true, false><<<cores, nullptr, a.stream>>>(a);
    } else if (a.context.pageSize)
        CompressedAttention<DType, false, true><<<cores, nullptr, a.stream>>>(a);
    else
        CompressedAttention<DType, false, false><<<cores, nullptr, a.stream>>>(a);
    return launch.Result();
}
} // namespace flash_attn::mla
