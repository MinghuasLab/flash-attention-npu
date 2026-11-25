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
template <typename DType, bool Brc, bool Partial, bool Paged, bool HasSoftcap, typename Reader>
__aicore__ inline void RunPrefillImpl(const MlaLaunchArgs& a, const Reader& read, const MlaAutoSplitBsndSchedule& s,
                                      const MlaLayout& l, uint32_t core, AscendC::TPipe* pipe)
{
    const auto& c = a.context;
    uint64_t first = 0, last = 0;
    if (!MlaOwnedGroups(read, c, s, core, first, last))
        return;
    Kernel::MlaPrefill<DType, Brc, Partial, Paged, HasSoftcap> kernel;
    MlaPrefillParams params{c.batch, c.sq,     c.sk,       c.heads,    c.kvHeads,  l.m,       l.n,   l.cores,
                            a.scale, c.window, a.inputs.q, a.inputs.k, a.inputs.v, a.softcap, a.mask};
    if (MlaTnd(c)) {
        params.batch = 1;
        params.sq = c.totalQ;
    }
    kernel.Init(params, core, (GM_ADDR)a.q, (GM_ADDR)a.k, (GM_ADDR)a.v, (GM_ADDR)a.out, (GM_ADDR)a.lse,
                (GM_ADDR)a.workspace, (GM_ADDR)a.workspace + l.outOffset, (GM_ADDR)a.workspace + l.sumOffset,
                (GM_ADDR)a.workspace + l.maxOffset, pipe, &a.cache);
    using Cursor = std::conditional_t<Partial, MlaNativeAutoCursor, MlaWholeMCursor>;
    Cursor cursor;
    const MlaPartitionView<Reader> view{read, c, l};
    if constexpr (Partial)
        cursor.template Init<Paged>(view, s, core, l.headGroups);
    else
        cursor.Init(l, s, core, first, last);
    for (uint64_t group = first; group < last; ++group) {
        MlaAutoSplitBsndTask t{};
        if constexpr (Partial) {
            if (!cursor.Next(view, s, c.window, t))
                break;
            group = t.group;
        } else if (!cursor.Next(read, c, l, s, t)) {
            AscendC::Trap();
            return;
        }
        if (t.s2Begin == t.s2End)
            continue;
        const uint32_t b = cursor.taskBatch;
        Block::PrefillTaskContext context{};
        context.task = {MlaTnd(c) ? 0U : b, cursor.taskHead, t.rowBegin, t.rowEnd - t.rowBegin,
                        cursor.queryLength, cursor.kvLength, t.s2Begin,  t.s2End};
        context.cacheBatch = b;
        if constexpr (Partial) {
            context.partial = t.splitCount > 1;
            if (context.partial) {
                if (t.fdHead >= s.fdHeads || t.split >= t.splitCount || t.slot >= s.partialPrefix[t.fdHead + 1]) {
                    cursor.valid = false;
                    break;
                }
                context.partialRow = t.slot * l.m;
            }
        }
        if (MlaTnd(c)) {
            const uint64_t qStart = cursor.queryStart;
            const uint64_t kStart = cursor.kvStart;
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
    if constexpr (Partial)
        if (!cursor.valid)
            AscendC::Trap();
}

// Only the one physical sequence and its whole-M prefixes are read by the
// direct task cursor. This adapter has no GM metadata or mutable tensor cache.
struct MlaStaticPrefillReader {
    uint32_t queryLength, kvLength, heads, mBlocks;

    __aicore__ inline uint32_t operator()(uint64_t word) const
    {
        constexpr uint32_t sequence = kMlaRecordHeaderWords;
        constexpr uint32_t prefix = sequence + kMlaSequenceWords + kMlaNativePartitionWords;
        if (word >= sequence && word < sequence + kMlaSequenceWords) {
            const uint32_t field = word - sequence;
            return field == 4 || field == 6 ? queryLength : field == 5 || field == 7 ? kvLength : 0;
        }
        if (word >= prefix && word < prefix + 2ULL * (uint64_t(heads) + 1)) {
            const uint64_t value = ((word - prefix) / 2) * mBlocks;
            return (word - prefix) % 2 ? uint32_t(value >> 32) : uint32_t(value);
        }
        AscendC::Trap();
        return 0;
    }
};

__aicore__ inline bool LoadStaticPrefill(const MlaLaunchArgs& a, MlaStaticPrefillReader& read, MlaLayout& l,
                                         MlaAutoSplitBsndSchedule& s)
{
    const auto& c = a.context;
    const auto& plan = a.staticPrefill;
    if (!a.cuQuery || !a.cuKv || a.cuQuery % 4 || a.cuKv % 4 || !plan.usedCoreNum ||
        plan.usedCoreNum > AscendC::GetBlockNum()) {
        AscendC::Trap();
        return false;
    }
    AscendC::GlobalTensor<uint32_t> query, kv;
    query.SetGlobalBuffer((__gm__ uint32_t*)a.cuQuery);
    kv.SetGlobalBuffer((__gm__ uint32_t*)a.cuKv);
    // Every AIC and AIV validates the mutable prefixes before output stores,
    // event allocation or waits. A failing vector can never strand its cube.
    const uint32_t qBegin = query.GetValue(0), qEnd = query.GetValue(1);
    const uint32_t kBegin = kv.GetValue(0), kEnd = kv.GetValue(1);
    if ((qBegin | kBegin | (qEnd ^ c.totalQ) | (kEnd ^ c.totalK)) != 0) {
        AscendC::Trap();
        return false;
    }
    l = plan.layout;
    l.cores = AscendC::GetBlockNum();
    const uint32_t blocks = (uint64_t(c.totalQ) + l.m - 1) / l.m;
    read = {c.totalQ, c.totalK, c.heads, blocks};
    s = {};
    s.aicNum = plan.layout.cores;
    s.usedCoreNum = plan.usedCoreNum;
    s.groups = uint64_t(c.heads) * blocks;
    l.groups = s.groups;
    s.used.query = {1, {c.totalQ, 0}, c.sq};
    s.used.kv = {1, {c.totalK, 0}, c.sk};
    auto& t = s.used.tiling;
    t.batch = c.heads;
    t.sq = c.sq;
    t.sk = c.sk;
    t.heads = 1;
    t.mBaseSize = l.m;
    t.s2BaseSize = l.n;
    t.usedCoreNum = s.usedCoreNum;
    t.blockDim = plan.layout.cores;
    t.cvRatio = 2;
    const uint64_t rows = ((uint64_t(c.totalQ) < l.m ? c.totalQ : l.m) + 15) / 16 * 16;
    t.mm1ResSize = rows * ((l.n + 31) / 32 * 32);
    t.mm2ResSize = rows * 512;
    for (uint32_t core = 0; core < 26; ++core) {
        s.bEnd[core] = t.bN2End[core] = plan.bEnd[core];
        s.mEnd[core] = t.gS1End[core] = plan.mEnd[core];
    }
    s.outOffset = l.outOffset;
    s.sumOffset = l.sumOffset;
    s.maxOffset = l.maxOffset;
    s.bytes = l.bytes;
    return true;
}
template <typename DType, bool IsTnd>
using PrefillFdMerge = Block::MlaSplitKvMerge<DType, true, IsTnd, 128, true, false, false, MlaAutoSplitKvMergeParams>;

template <typename DType, bool IsTnd>
__aicore__ inline void PrefillMerge(const MlaLaunchArgs& a, const MlaAutoSplitKvMergeParams& p, const MlaLayout& l,
                                    uint64_t row, uint32_t begin, uint32_t end, PrefillFdMerge<DType, IsTnd>& merge)
{
    merge.Bind(p, (GM_ADDR)a.workspace + l.outOffset + row * 128 * 4, (GM_ADDR)a.workspace + l.sumOffset + row * 8 * 4,
               (GM_ADDR)a.workspace + l.maxOffset + row * 8 * 4, (GM_ADDR)a.out, (GM_ADDR)a.lse, (GM_ADDR)a.accum,
               (GM_ADDR)a.accumLse, a.softcap);
    for (uint32_t start = begin; start < end; start += merge.TileRows())
        merge.Run(start, merge.TileRows() < end - start ? merge.TileRows() : end - start);
}
template <typename DType, bool IsTnd>
__aicore__ inline void MergePrefillRows(const MlaLaunchArgs& a, const MlaReader& read,
                                        const MlaAutoSplitBsndSchedule& s, const MlaLayout& l,
                                        const MlaAutoSplitBsndMerge& fragment, uint32_t begin, uint32_t end,
                                        PrefillFdMerge<DType, IsTnd>& merge)
{
    const auto& c = a.context;
    const MlaPartitionView<MlaReader> view{read, c, l};
    MlaAutoSplitBsndTask geometry{};
    geometry.batch = fragment.batch;
    geometry.m = fragment.m;
    auto_split_detail::Geometry<false>(view, s, c.window, geometry);
    const uint32_t rows = geometry.rowEnd - geometry.rowBegin;
    if (begin >= rows)
        return;
    MlaAutoSplitKvMergeParams p{};
    p.batch = c.batch;
    p.sq = c.sq;
    p.heads = c.heads / c.kvHeads;
    p.batchIndex = fragment.batch / c.kvHeads;
    p.kvHeads = c.kvHeads;
    p.kvHead = fragment.batch % c.kvHeads;
    p.rowStart = geometry.rowBegin;
    p.rows = rows;
    p.mBase = l.m;
    p.splits = fragment.splitCount;
    p.queryLength = read((MlaSequenceWord(p.batchIndex) + 6));
    p.kvLength = read((MlaSequenceWord(p.batchIndex) + 6) + 1);
    p.maskKind = c.window.kind;
    p.left = c.window.left;
    p.right = c.window.right;
    if (p.splits < 2 || p.splits > p.kSplitCapacity) {
        AscendC::Trap();
        return;
    }
    if (p.maskKind) {
        if (auto_split_detail::Parts(s, geometry, UINT32_MAX, nullptr, p.begin, p.end, p.kSplitCapacity) != p.splits) {
            AscendC::Trap();
            return;
        }
        for (uint32_t split = 0; split < p.splits; ++split) {
            const uint32_t first = p.begin[split] * l.n, last = p.end[split] * l.n;
            p.begin[split] = p.kvLength < first ? p.kvLength : first;
            p.end[split] = p.kvLength < last ? p.kvLength : last;
        }
    }
    const uint64_t row = fragment.firstSlot * l.m;
    end = end < p.rows ? end : p.rows;
    if constexpr (IsTnd) {
        p.batch = 1;
        p.sq = c.totalQ;
        p.batchIndex = read(MlaSequenceWord(p.batchIndex));
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
    PrefillFdMerge<DType, IsTnd> merge;
    merge.InitBuffers(pipe, a.accumLse != 0);
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
        MergePrefillRows<DType, IsTnd>(a, read, s, l, r, part * 8, parts * 8, merge);
    }
    merge.Finish();
}
#endif
template <typename DType, bool Brc, bool Partial, bool Paged, bool IsTnd, bool HasSoftcap>
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
    if ASCEND_IS_AIV {
        // Direct, non-paged tasks write every O and LSE row when the
        // validated lengths and causal window leave no allocation holes.
        const bool directRows =
            !s.fdHeads && !Paged && a.context.window.kind <= 1 && !MlaNeedsOutputInit<true>(read, a.context, s);
        if (directRows) {
            // Inactive public partial LSE slots still carry the -inf sentinel.
            if (a.accumLse)
                InitializeMlaOutputs<DType>(a, false, false);
        } else {
            InitializeMlaOutputs<DType>(a, MlaNeedsOutputInit(read, a.context, s));
            if (a.context.splits <= 1 && s.groups && !MlaEmptyKv(a.context))
                AscendC::SyncAll();
        }
    }
    if (!s.groups || MlaEmptyKv(a.context))
        return;
    uint32_t core = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= AscendC::GetSubBlockNum();
    }
    RunPrefillImpl<DType, Brc, Partial, Paged, HasSoftcap>(a, read, s, l, core, &pipe);
    if ASCEND_IS_AIV {
        if constexpr (Partial) {
            if (s.fdHeads)
                MergePrefillOutputsImpl<DType, IsTnd>(a, read, s, l, &pipe);
        }
    }
#endif
}
template <typename DType>
__global__ __aicore__ __mix__(1, 2) void StaticPrefillAttention(MlaLaunchArgs a)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaStaticPrefillReader read;
    MlaAutoSplitBsndSchedule s;
    MlaLayout l{};
    if (!LoadStaticPrefill(a, read, l, s))
        return;
    AscendC::SetSyncBaseAddr(a.ffts);
    AscendC::TPipe pipe;
    if ASCEND_IS_AIV {
        InitializeMlaOutputs<DType>(a, MlaNeedsOutputInit(read, a.context, s));
        AscendC::SyncAll();
    }
    uint32_t core = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= AscendC::GetSubBlockNum();
    }
    // TND prefill has N <= 512 and M == 512, so its source Brc is false.
    RunPrefillImpl<DType, false, false, false, false>(a, read, s, l, core, &pipe);
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
    const uint32_t cores = MlaStaticLaunchCores(a.context);
    if (a.staticPrefill.usedCoreNum) {
        StaticPrefillAttention<DType><<<cores, nullptr, a.stream>>>(a);
        return launch.Result();
    }
    MLA_PREFILL_BOOL_SWITCH(l.m <= 128, Brc, {
        MLA_PREFILL_BOOL_SWITCH(a.context.splits != 1, Partial, {
            MLA_PREFILL_BOOL_SWITCH(a.context.pageSize != 0, Paged, {
                MLA_PREFILL_BOOL_SWITCH(MlaTnd(a.context), IsTnd, {
                    MLA_PREFILL_BOOL_SWITCH(a.softcap != 0.0f, HasSoftcap, {
                        PrefillAttention<DType, Brc, Partial, Paged, IsTnd, HasSoftcap>
                            <<<cores, nullptr, a.stream>>>(a);
                    });
                });
            });
        });
    });
    return launch.Result();
}
#undef MLA_PREFILL_BOOL_SWITCH
} // namespace flash_attn::mla
