/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Reuse the validated partial/FD numerics with packed GQA scheduling coordinates.
#pragma once
#include "kernel_operator.h"
#include "mla_dispatch.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla_dispatch_impl.hpp"
#include "mla_kernel.hpp"
#include "mla_split_kv_kernel.hpp"
#include "mla_prefill_kernel.hpp"
#include "mla_split_kv_merge_block.hpp"
namespace flash_attn::mla {
template <typename DType, bool IsTnd, bool Paged>
__aicore__ inline void GqaCompressed(const MlaLaunchArgs& a, const MlaLayout& l, const MlaAutoSplitBsndTask& t,
                                     const MlaReader& read, uint32_t core, uint64_t row)
{
    const auto& c = a.context;
    const uint32_t b = t.batch / l.headGroups, head = (t.batch % l.headGroups) * l.g;
    MlaDenseKernelSchedule schedule{};
    schedule.batch = c.batch;
    schedule.sq = c.sq;
    schedule.sk = c.sk;
    schedule.heads = l.g;
    schedule.mBaseSize = l.m;
    schedule.s2BaseSize = l.n;
    schedule.blockDim = schedule.usedCoreNum = l.cores;
    schedule.cvRatio = 2;
    schedule.mm1ResSize = l.m * ((l.n + 31) / 32 * 32);
    schedule.mm2ResSize = l.m * 512;
    MlaSplitKvKernelParams p{};
    p.window = c.window;
    p.tasks[core] = {b, t.m, t.split, t.s2Begin, t.s2End};
    p.totalTasks = l.cores;
    p.outOffsetBytes = l.outOffset + row * 512 * 4;
    p.sumOffsetBytes = l.sumOffset + row * 8 * 4;
    p.maxOffsetBytes = l.maxOffset + row * 8 * 4;
    MlaTaskLengths lengths{read(32 + 8 * b), read(32 + 8 * b + 6), read(32 + 8 * b + 7),
                           ReadMetadataWide(read, 32 + 8 * b + 2)};
    AscendC::TPipe pipe;
    Kernel::MlaGqaPartial<DType, IsTnd, Paged> op;
    op.RunFragment(schedule, p, lengths, a.scale, (GM_ADDR)a.qv + uint64_t(head) * a.inputs.qv.headStride * 2,
                   (GM_ADDR)a.q + uint64_t(head) * a.inputs.q.headStride * 2, (GM_ADDR)a.v, (GM_ADDR)a.k,
                   (GM_ADDR)a.workspace, &pipe, a.inputs, c.layout == 1, a.cache, a.softcap);
}
template <typename DType, bool Brc, bool Paged>
__aicore__ inline void GqaPrefill(const MlaLaunchArgs& a, const MlaLayout& l, const MlaAutoSplitBsndTask& t,
                                  const MlaReader& read, uint32_t core, uint64_t row)
{
    const auto& c = a.context;
    const uint32_t b = t.batch / l.headGroups, head = (t.batch % l.headGroups) * l.g;
    const uint32_t kvHead = head / (c.heads / c.kvHeads);
    const auto& in = a.inputs;
    const uint64_t qStart = MlaTnd(c) ? uint64_t(read(32 + 8 * b)) * in.q.rowStride : uint64_t(b) * in.q.batchStride;
    const uint64_t kStart = c.layout == 1 ? uint64_t(read(32 + 8 * b + 2)) * in.k.rowStride
                            : Paged       ? 0
                                          : uint64_t(b) * in.k.batchStride;
    const uint64_t vStart = c.layout == 1 ? uint64_t(read(32 + 8 * b + 2)) * in.v.rowStride
                            : Paged       ? 0
                                          : uint64_t(b) * in.v.batchStride;
    MlaPrefillParams p{1, c.sq, c.sk, l.g, 1, l.m, l.n, l.cores, a.scale, c.window, in.q, in.k, in.v, a.softcap};
    MlaPrefillTask task{
        0, 0, t.rowBegin, t.rowEnd - t.rowBegin, read(32 + 8 * b + 6), read(32 + 8 * b + 7), t.s2Begin, t.s2End};
    auto cache = a.cache;
    if constexpr (Paged)
        cache.table.address += uint64_t(b) * cache.layout.tableRowStride * 4;
    AscendC::TPipe pipe;
    Kernel::MlaPrefill<DType, Brc, true, Paged> op;
    op.Run(p, task, core, (GM_ADDR)a.q + (qStart + uint64_t(head) * in.q.headStride) * 2,
           (GM_ADDR)a.k + (kStart + uint64_t(kvHead) * in.k.headStride) * 2,
           (GM_ADDR)a.v + (vStart + uint64_t(kvHead) * in.v.headStride) * 2, (GM_ADDR)a.out, (GM_ADDR)a.lse,
           (GM_ADDR)a.workspace, (GM_ADDR)a.workspace + l.outOffset + row * 128 * 4,
           (GM_ADDR)a.workspace + l.sumOffset + row * 8 * 4, (GM_ADDR)a.workspace + l.maxOffset + row * 8 * 4, &pipe,
           &cache);
}
template <typename DType, uint32_t Dim, bool IsTnd, bool Accumulator>
__aicore__ inline void GqaMerge(const MlaLaunchArgs& a, const MlaLayout& l, const MlaSplitKvMergeParams& p,
                                uint64_t row, uint32_t begin, uint32_t count)
{
    AscendC::TPipe pipe;
    Block::MlaSplitKvMerge<DType, Accumulator, IsTnd, Dim, true> op;
    op.Init(p, (GM_ADDR)a.workspace + l.outOffset + row * Dim * 4, (GM_ADDR)a.workspace + l.sumOffset + row * 8 * 4,
            (GM_ADDR)a.workspace + l.maxOffset + row * 8 * 4, (GM_ADDR)a.out, (GM_ADDR)a.lse, (GM_ADDR)a.accum,
            (GM_ADDR)a.accumLse, &pipe, a.softcap);
    op.Run(begin, count);
    op.Finish();
}
} // namespace flash_attn::mla
#endif
namespace flash_attn::mla {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
template <typename DType>
__aicore__ inline void RunGqaAttention(const MlaLaunchArgs& a, const MlaReader& read, const MlaLayout& l,
                                       const MlaAutoSplitBsndSchedule& s)
{
    uint32_t core = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV
        core /= AscendC::GetSubBlockNum();
    const auto& c = a.context;
    const uint32_t slots = MlaTaskSlots(c, s);
    for (uint64_t index = core; index < s.groups * slots; index += l.cores) {
        const uint64_t group = index / slots;
        const uint32_t split = index % slots;
        MlaAutoSplitBsndTask t{};
        if (!MlaTask(read, c, l, s, group, split, t) || t.s2Begin == t.s2End)
            continue;
        const uint64_t row = (group * l.capacity + split) * l.m;
        if (c.valueDim == 512) {
            if (MlaTnd(c)) {
                if (c.pageSize)
                    GqaCompressed<DType, true, true>(a, l, t, read, core, row);
                else
                    GqaCompressed<DType, true, false>(a, l, t, read, core, row);
            } else if (c.pageSize)
                GqaCompressed<DType, false, true>(a, l, t, read, core, row);
            else
                GqaCompressed<DType, false, false>(a, l, t, read, core, row);
        } else if (l.m <= 128) {
            if (c.pageSize)
                GqaPrefill<DType, true, true>(a, l, t, read, core, row);
            else
                GqaPrefill<DType, true, false>(a, l, t, read, core, row);
        } else if (c.pageSize)
            GqaPrefill<DType, false, true>(a, l, t, read, core, row);
        else
            GqaPrefill<DType, false, false>(a, l, t, read, core, row);
    }
}
template <typename DType>
__aicore__ inline void RunGqaMerge(const MlaLaunchArgs& a, const MlaReader& read, const MlaLayout& l,
                                   const MlaAutoSplitBsndSchedule& s)
{
    if ASCEND_IS_AIV {
        const auto& c = a.context;
        const uint32_t tile = 192 / l.capacity > 8 ? 8 : 192 / l.capacity ? 192 / l.capacity : 1;
        // A short query cannot populate the remainder of the fixed M tile.
        // Keep partial addresses unchanged while avoiding empty merge tasks.
        const uint64_t rows = uint64_t(c.sq) * l.g;
        const uint32_t rowBound = rows < l.m ? uint32_t(rows) : l.m;
        const uint32_t tiles = (rowBound + tile - 1) / tile;
        for (uint64_t index = AscendC::GetBlockIdx(); index < s.groups * tiles; index += l.cores * 2) {
            const uint64_t group = index / tiles;
            const uint32_t begin = (index % tiles) * tile;
            MlaAutoSplitBsndTask t{};
            if (!MlaTask(read, c, l, s, group, 0, t)) {
                AscendC::Trap();
                return;
            }
            const uint32_t b = t.batch / l.headGroups;
            MlaSplitKvMergeParams p{};
            p.batch = MlaTnd(c) ? 1 : c.batch;
            p.sq = MlaTnd(c) ? c.totalQ : c.sq;
            p.batchIndex = MlaTnd(c) ? read(32 + 8 * b) : b;
            p.heads = l.g;
            p.kvHeads = l.headGroups;
            p.kvHead = t.batch % l.headGroups;
            p.rowStart = t.rowBegin;
            p.rows = t.rowEnd - t.rowBegin;
            if (begin >= p.rows)
                continue;
            p.mBase = l.m;
            p.splits = t.splitCount;
            p.queryLength = read(32 + 8 * b + 6);
            p.kvLength = read(32 + 8 * b + 7);
            p.maskKind = c.window.kind;
            p.left = c.window.left;
            p.right = c.window.right;
            if (!c.splits) {
                auto geometry = t;
                auto_split_detail::Geometry<false>(MlaPartitionView<MlaReader>{read, c, l}, s, c.window, geometry);
                auto_split_detail::Parts(s, geometry, UINT32_MAX, nullptr, p.begin, p.end);
            }
            for (uint32_t split = 0; split < p.splits; ++split) {
                uint32_t begin = p.begin[split], end = p.end[split];
                if (c.splits)
                    SplitBsndFragment(p.queryLength, p.kvLength, l.g, t.m, l.m, l.n, c.window, c.splits, split, begin,
                                      end);
                p.begin[split] = p.kvLength < begin * l.n ? p.kvLength : begin * l.n;
                p.end[split] = p.kvLength < end * l.n ? p.kvLength : end * l.n;
            }
            const uint64_t row = group * l.capacity * l.m;
            const uint32_t count = tile < p.rows - begin ? tile : p.rows - begin;
#define MLA_GQA_MERGE_DIM(D, T)                                                                                        \
    if (c.splits == 1)                                                                                                 \
        GqaMerge<DType, D, T, false>(a, l, p, row, begin, count);                                                      \
    else                                                                                                               \
        GqaMerge<DType, D, T, true>(a, l, p, row, begin, count)
            if (c.valueDim == 512) {
                if (MlaTnd(c)) {
                    MLA_GQA_MERGE_DIM(512, true);
                } else {
                    MLA_GQA_MERGE_DIM(512, false);
                }
            } else if (MlaTnd(c)) {
                MLA_GQA_MERGE_DIM(128, true);
            } else {
                MLA_GQA_MERGE_DIM(128, false);
            }
#undef MLA_GQA_MERGE_DIM
        }
    }
}
#endif
template <typename DType>
__global__ __aicore__ __mix__(1, 2) void GqaAttention(MlaLaunchArgs a)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaReader read;
    MlaLayout l{};
    MlaAutoSplitBsndSchedule s{};
    // Every core validates before initialization, events or synchronization.
    if (!LoadMla<DType>(a, read, l, s))
        return;
    AscendC::SetSyncBaseAddr(a.ffts);
    {
        AscendC::TPipe pipe;
        InitializeMlaOutputs<DType>(a);
    }
    if (MlaEmptyKv(a.context))
        return;
    if ASCEND_IS_AIV {
        AscendC::SyncAll();
    }
    RunGqaAttention<DType>(a, read, l, s);
    if ASCEND_IS_AIV {
        // Match source FiaKernelNonQuantMla::FlashDecode: all vector cores
        // join after partial writeback, including cores with no attention task.
        // This also orders initialization before any final output is written.
        AscendC::SyncAll();
        RunGqaMerge<DType>(a, read, l, s);
    }
#endif
}
template <typename DType>
aclError LaunchGrouped(const MlaLaunchArgs& a)
{
    const uint32_t cores = a.context.physicalCores - a.context.margin;
    MlaBatchLaunchScope launch;
    GqaAttention<DType><<<cores, nullptr, a.stream>>>(a);
    return launch.Result();
}
} // namespace flash_attn::mla
