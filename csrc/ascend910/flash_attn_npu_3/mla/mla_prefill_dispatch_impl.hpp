/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once
#include "kernel_operator.h"
#include "mla_dispatch.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "lib/matmul_intf.h"
#include "mla_prefill_kernel.hpp"
#include "mla_kernel.hpp"
#include "mla_dispatch_impl.hpp"
#include "mla_split_kv_merge_block.hpp"
#endif
namespace flash_attn::mla {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
template <typename DType, bool Brc, bool Partial, bool Paged>
__aicore__ inline void RunPrefillImpl(const MlaLaunchArgs& a, const MlaReader& read, const MlaAutoSplitBsndSchedule& s,
                                      const MlaLayout& l, uint32_t core, AscendC::TPipe* pipe)
{
    const auto& c = a.context;
    const bool autoSplit = c.splits == 0;
    uint64_t first = core, last = s.groups * l.capacity;
    if (autoSplit) {
        if (core >= s.usedCoreNum)
            return;
        const MlaPartitionView<MlaReader> view{read, c, l};
        const uint32_t prefix = auto_split_detail::Base(s.used.tiling.batch) + kMlaNativePartitionWords;
        const uint32_t b = core ? s.bEnd[core - 1] : 0;
        first = ReadMetadataWide(view, prefix + 2 * b) + (core ? s.mEnd[core - 1] : 0);
        last = ReadMetadataWide(view, prefix + 2 * s.bEnd[core]) + s.mEnd[core] + (s.nEnd[core] ? 1 : 0);
        if (last > s.groups)
            last = s.groups;
    }
    if (first >= last)
        return;
    Kernel::MlaPrefill<DType, Brc, Partial, Paged> kernel;
    MlaPrefillParams params{c.batch, c.sq,    c.sk,     c.heads,    c.kvHeads,  l.m,        l.n,
                            l.cores, a.scale, c.window, a.inputs.q, a.inputs.k, a.inputs.v, a.softcap};
    if (MlaTnd(c)) {
        params.batch = 1;
        params.sq = c.totalQ;
    }
    kernel.Init(params, core, (GM_ADDR)a.q, (GM_ADDR)a.k, (GM_ADDR)a.v, (GM_ADDR)a.out, (GM_ADDR)a.lse,
                (GM_ADDR)a.workspace, (GM_ADDR)a.workspace + l.outOffset, (GM_ADDR)a.workspace + l.sumOffset,
                (GM_ADDR)a.workspace + l.maxOffset, pipe, &a.cache);
    for (uint64_t index = first; index < last; index += autoSplit ? 1 : l.cores) {
        const uint64_t group = autoSplit ? index : index / l.capacity;
        const uint32_t split = autoSplit ? 0 : index % l.capacity;
        MlaAutoSplitBsndTask t{};
        if (!MlaTask(read, c, l, s, group, split, t, autoSplit ? core : UINT32_MAX) || t.s2Begin == t.s2End)
            continue;
        const uint32_t b = t.batch / c.kvHeads;
        Block::PrefillTaskContext context{};
        context.task = {MlaTnd(c) ? 0U : b,
                        t.batch % c.kvHeads,
                        t.rowBegin,
                        t.rowEnd - t.rowBegin,
                        read((MlaSequenceWord(b) + 6)),
                        read((MlaSequenceWord(b) + 6) + 1),
                        t.s2Begin,
                        t.s2End};
        context.cacheBatch = b;
        context.partial = t.splitCount > 1;
        context.partialRow = (group * l.capacity + t.split) * l.m;
        if (MlaTnd(c)) {
            const uint64_t qStart = read(MlaSequenceWord(b));
            const uint64_t kStart = read(MlaSequenceWord(b) + 2);
            context.qOffset = qStart * a.inputs.q.rowStride;
            if constexpr (!Paged) {
                context.kOffset = c.layout == 1 ? kStart * a.inputs.k.rowStride : uint64_t(b) * a.inputs.k.batchStride;
                context.vOffset = c.layout == 1 ? kStart * a.inputs.v.rowStride : uint64_t(b) * a.inputs.v.batchStride;
            }
            context.outputOffset = qStart * c.heads;
            context.lseOffset = qStart;
        }
        kernel.RunTask(context);
    }
    kernel.Finish();
}
template <typename DType, bool IsTnd>
__aicore__ inline void PrefillMerge(const MlaLaunchArgs& a, const MlaSplitKvMergeParams& p, const MlaLayout& l,
                                    uint64_t row, uint32_t begin, uint32_t end,
                                    Block::MlaSplitKvMerge<DType, true, IsTnd, 128>& merge)
{
    merge.Bind(p, (GM_ADDR)a.workspace + l.outOffset + row * 128 * 4, (GM_ADDR)a.workspace + l.sumOffset + row * 8 * 4,
               (GM_ADDR)a.workspace + l.maxOffset + row * 8 * 4, (GM_ADDR)a.out, (GM_ADDR)a.lse, (GM_ADDR)a.accum,
               (GM_ADDR)a.accumLse, a.softcap);
    for (uint32_t start = begin; start < end; start += merge.TileRows())
        merge.Run(start, merge.TileRows() < end - start ? merge.TileRows() : end - start);
}
template <typename DType, bool IsTnd>
__aicore__ inline void
MergePrefillRows(const MlaLaunchArgs& a, const MlaReader& read, const MlaAutoSplitBsndSchedule& s, const MlaLayout& l,
                 uint64_t group, uint32_t begin, uint32_t end, Block::MlaSplitKvMerge<DType, true, IsTnd, 128>& merge)
{
    const auto& c = a.context;
    MlaAutoSplitBsndTask t{};
    if (!MlaTask(read, c, l, s, group, 0, t)) {
        AscendC::Trap();
        return;
    }
    if (begin >= t.rowEnd - t.rowBegin)
        return;
    MlaSplitKvMergeParams p{};
    p.batch = c.batch;
    p.sq = c.sq;
    p.heads = c.heads / c.kvHeads;
    p.batchIndex = t.batch / c.kvHeads;
    p.kvHeads = c.kvHeads;
    p.kvHead = t.batch % c.kvHeads;
    p.rowStart = t.rowBegin;
    p.rows = t.rowEnd - t.rowBegin;
    p.mBase = l.m;
    p.splits = t.splitCount;
    p.queryLength = read((MlaSequenceWord(p.batchIndex) + 6));
    p.kvLength = read((MlaSequenceWord(p.batchIndex) + 6) + 1);
    p.maskKind = c.window.kind;
    p.left = c.window.left;
    p.right = c.window.right;
    const MlaPartitionView<MlaReader> view{read, c, l};
    auto geometry = t;
    auto_split_detail::Geometry<false>(view, s, c.window, geometry);
    if (auto_split_detail::Parts(s, geometry, UINT32_MAX, nullptr, p.begin, p.end) != p.splits) {
        AscendC::Trap();
        return;
    }
    for (uint32_t split = 0; split < p.splits; ++split) {
        const uint32_t first = p.begin[split] * l.n, last = p.end[split] * l.n;
        p.begin[split] = p.kvLength < first ? p.kvLength : first;
        p.end[split] = p.kvLength < last ? p.kvLength : last;
    }
    const uint64_t row = group * l.capacity * l.m;
    end = end < p.rows ? end : p.rows;
    if constexpr (IsTnd) {
        p.batch = 1;
        p.sq = c.totalQ;
        p.batchIndex = read(MlaSequenceWord(t.batch / c.kvHeads));
    }
    PrefillMerge<DType, IsTnd>(a, p, l, row, begin, end, merge);
}
template <typename DType, bool IsTnd>
__aicore__ inline void MergePrefillOutputsImpl(const MlaLaunchArgs& a, const MlaReader& read,
                                               const MlaAutoSplitBsndSchedule& s, const MlaLayout& l,
                                               AscendC::TPipe* pipe)
{
    const auto& c = a.context;
    const uint32_t owner = AscendC::GetBlockIdx();
    if (owner >= s.vectors) {
        AscendC::SyncAll();
        return;
    }
    pipe->Reset();
    Block::MlaSplitKvMerge<DType, true, IsTnd, 128> merge;
    merge.InitBuffers(pipe);
    AscendC::ICachePreLoad(2);
    AscendC::SyncAll();
    // The native tail assigns only FD groups to vectors. Direct S1G
    // blocks have already written their final O/LSE.
    const MlaPartitionView<MlaReader> view{read, c, l};
    const uint32_t base = auto_split_detail::Base(s.used.tiling.batch);
    const uint32_t firstHead = owner ? view(base + kMlaNativeVectorHeadEnd + owner - 1) : 0;
    const uint32_t firstPart = owner ? view(base + kMlaNativeVectorPartEnd + owner - 1) + 1 : 0;
    const uint32_t lastHead = view(base + kMlaNativeVectorHeadEnd + owner);
    const uint32_t lastPart = view(base + kMlaNativeVectorPartEnd + owner);
    for (uint32_t head = firstHead; head <= lastHead; ++head) {
        const uint32_t parts = head == lastHead ? lastPart + 1 : view(base + kMlaNativeFdParts + head);
        const uint32_t part = head == firstHead ? firstPart : 0;
        if (part >= parts)
            continue;
        MlaAutoSplitBsndMerge r{};
        if (!AutoSplitBsndMerge(view, s, head, part, r)) {
            merge.Finish();
            AscendC::Trap();
            return;
        }
        MergePrefillRows<DType, IsTnd>(a, read, s, l, r.group, part * 8, parts * 8, merge);
    }
    merge.Finish();
}
#endif
template <typename DType, bool Brc, bool Partial, bool Paged, bool IsTnd>
__global__ __aicore__ __mix__(1, 2) void PrefillAttention(MlaLaunchArgs a)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaReader read;
    MlaAutoSplitBsndSchedule s;
    MlaLayout l{};
    if (!LoadMla<DType>(a, read, l, s))
        return;
    AscendC::SetSyncBaseAddr(a.ffts);
    AscendC::TPipe pipe;
    InitializeMlaOutputs<DType>(a);
    if (!s.groups || MlaEmptyKv(a.context))
        return;
    if ASCEND_IS_AIV {
        // Direct output must wait for all vector cores, including idle ones.
        if (a.context.splits <= 1)
            AscendC::SyncAll();
    }
    uint32_t core = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= AscendC::GetSubBlockNum();
    }
    RunPrefillImpl<DType, Brc, Partial, Paged>(a, read, s, l, core, &pipe);
    if ASCEND_IS_AIV {
        if constexpr (Partial) {
            MergePrefillOutputsImpl<DType, IsTnd>(a, read, s, l, &pipe);
        }
    }
#endif
}
// Resolve device specialization on host, as in fwd_dispatch_impl.hpp.
#define MLA_PREFILL_BOOL_SWITCH(COND, CONST_NAME, ...)                                                                 \
    do {                                                                                                               \
        if (COND) {                                                                                                    \
            constexpr bool CONST_NAME = true;                                                                          \
            __VA_ARGS__                                                                                                \
        } else {                                                                                                       \
            constexpr bool CONST_NAME = false;                                                                         \
            __VA_ARGS__                                                                                                \
        }                                                                                                              \
    } while (0)

template <typename DType>
aclError LaunchPrefill(const MlaLaunchArgs& a)
{
    MlaLayout l{};
    if (!MlaRecordLayout(a.context, l))
        return ACL_ERROR_INVALID_PARAM;
    MlaBatchLaunchScope launch;
    MLA_PREFILL_BOOL_SWITCH(l.m <= 128, Brc, {
        MLA_PREFILL_BOOL_SWITCH(a.context.splits != 1, Partial, {
            MLA_PREFILL_BOOL_SWITCH(a.context.pageSize != 0, Paged, {
                MLA_PREFILL_BOOL_SWITCH(MlaTnd(a.context), IsTnd, {
                    PrefillAttention<DType, Brc, Partial, Paged, IsTnd><<<l.cores, nullptr, a.stream>>>(a);
                });
            });
        });
    });
    return launch.Result();
}
#undef MLA_PREFILL_BOOL_SWITCH
} // namespace flash_attn::mla
