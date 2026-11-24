/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once
#include "kernel_operator.h"
#include "mla_dispatch.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla_dispatch_impl.hpp"
#include "mla_q16_schedule.hpp"
namespace flash_attn::mla {
// External records are only uint32 aligned. Keep every DMA inside the
// declared words; the destination padding belongs exclusively to UB/local.
__aicore__ inline void SnapshotMlaWords(uint64_t address, uint32_t count, uint32_t* destination)
{
    if (!count)
        return;
    const uint32_t prefix = count / 16 * 16;
#if defined(__DAV_CUBE__)
    const auto* source = reinterpret_cast<__gm__ uint32_t*>(address);
    uint32_t first = 0;
    if (!(address % 64)) {
        if (prefix)
            copy_data_align64(reinterpret_cast<uint8_t*>(destination), reinterpret_cast<__gm__ uint8_t*>(address),
                              uint64_t(prefix) * 4);
        first = prefix;
    }
    for (uint32_t word = first; word < count; ++word)
        destination[word] = source[word];
#else
    AscendC::GlobalTensor<uint32_t> source;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    AscendC::LocalTensor<uint32_t> staging(AscendC::TPosition::VECIN, 0, 400);
    AscendC::DataCopyPad(staging, source, AscendC::DataCopyExtParams{1, count * 4, 0, 0, 0},
                         AscendC::DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    auto* words = reinterpret_cast<__ubuf__ uint32_t*>(get_imm(0));
    if (prefix)
        copy_data_align64(reinterpret_cast<uint8_t*>(destination), reinterpret_cast<__ubuf__ uint8_t*>(words),
                          uint64_t(prefix) * 4);
    for (uint32_t word = prefix; word < count; ++word)
        destination[word] = words[word];
#endif
    // UB0 and event0 are reusable before TPipe allocates numerical resources.
    pipe_barrier(PIPE_ALL);
}
struct MlaLocalMetadataReader {
    const uint32_t* words = nullptr;
    __aicore__ inline uint32_t operator()(uint64_t word) const
    {
        return words[word];
    }
};
__aicore__ inline MlaMetadataReadStatus ReadDecodePagesSnapshot(const MlaContext& c, const MlaPagedCacheInputs& cache,
                                                                uint32_t length)
{
    using Status = MlaMetadataReadStatus;
    const auto status = CheckPagedCacheDescriptor(c, cache, 1, 64, 512);
    if (status != Status::Success)
        return status;
    if (length > c.sk)
        return Status::InvalidCacheLengths;
    // Every caller has B1, PA128 and K capacity8192. Inactive IDs stay unread.
    const uint32_t count = (length + 127) / 128;
    uint32_t storage[64 + 16];
    auto* pages = reinterpret_cast<uint32_t*>((reinterpret_cast<uint64_t>(storage) + 63) & ~uint64_t(63));
    SnapshotMlaWords(cache.table.address, count, pages);
    for (uint32_t page = 0; page < count; ++page) {
        const int32_t id = int32_t(pages[page]);
        if (id < 0 || uint64_t(id) >= cache.layout.physicalPages)
            return Status::InvalidPageTable;
    }
    return Status::Success;
}
// The sole physical B1/Q1 span, native partition and M prefix are synthesized
// from the current device KV length and the host's shape-only FD table.
struct MlaDynamicDecodeReader {
    const MlaContext* context = nullptr;
    const MlaLayout* recordLayout = nullptr;
    const MlaAutoSplitBsndSchedule* schedule = nullptr;
    uint64_t vectorEnds = 0;
    uint32_t kvLength = 0;

    __aicore__ inline uint32_t operator()(uint64_t word) const
    {
        const auto& c = *context;
        const auto& l = *recordLayout;
        const auto& s = *schedule;
        if (word < kMlaRecordHeaderWords) {
            const uint32_t kvCapacity = c.pageSize ? (kvLength + 127) / 128 * 128 : c.sk;
            const auto header = MlaHeader(c, l, 1, kvCapacity);
            return reinterpret_cast<const uint32_t*>(&header)[word];
        }
        if (word < MlaTail(c)) {
            const uint32_t field = word - kMlaRecordHeaderWords;
            return field == 4 || field == 6 ? 1 : field == 5 ? 8192 : field == 7 ? kvLength : 0;
        }
        const uint64_t index = word - MlaTail(c);
        if (index >= kMlaNativePartitionWords) {
            if (index < kMlaNativePartitionWords + 4)
                return index == kMlaNativePartitionWords + 2 ? 1 : 0;
            AscendC::Trap();
            return 0;
        }
        if (index == 1)
            return uint32_t(c.window.left);
        if (index == 2)
            return uint32_t(c.window.right);
        if (index == kMlaNativeUsedCoreNum)
            return s.usedCoreNum;
        if (index >= kMlaNativeBEnd && index < kMlaNativeBEnd + 26)
            return s.bEnd[index - kMlaNativeBEnd];
        if (index >= kMlaNativeNEnd && index < kMlaNativeNEnd + 26)
            return s.nEnd[index - kMlaNativeNEnd];
        if (index == kMlaNativeFdHeads)
            return s.fdHeads;
        if (index == kMlaNativeMaxSplits)
            return s.maxSplits;
        if (index == kMlaNativeVectors)
            return s.vectors;
        if (index == kMlaNativeFdRows)
            return 8;
        if (index == kMlaNativeFdSplits)
            return s.maxSplits;
        if (index >= kMlaNativeCoreSplitStart && index < kMlaNativeCoreSplitStart + 26) {
            const uint32_t core = index - kMlaNativeCoreSplitStart;
            return core < s.usedCoreNum ? core : 0;
        }
        if (index == kMlaNativeFdParts)
            return s.fdHeads ? 16 : 0;
        if (index == kMlaNativeFdLastRows)
            return s.fdHeads ? 8 : 0;
        if (index >= kMlaNativeVectorPartEnd && index < kMlaNativeVectorPartEnd + 52) {
            const uint32_t vector = index - kMlaNativeVectorPartEnd;
            return vector < s.vectors ? uint32_t((vectorEnds >> (4 * vector)) & 15) : 0;
        }
        return 0;
    }
};

__aicore__ inline bool LoadDynamicDecode(const MlaLaunchArgs& a, const MlaDynamicDecodePlan& plan,
                                         MlaDynamicDecodeReader& read, MlaLayout& l, MlaAutoSplitBsndSchedule& s)
{
    const auto& c = a.context;
    if (!plan.enabled || c.valueDim != 512 || c.batch != 1 || c.sq != 1 || c.heads != 128 || c.kvHeads != 1 ||
        c.sk != 8192 || c.physicalCores < 16 || c.physicalCores > 26 || c.margin || c.splits || !c.packed ||
        c.window.kind || a.softcap != 0.0f || c.hasSoftcap || (c.layout != 0 && c.layout != 3) ||
        c.pageSize != (c.layout == 3 ? 128U : 0U) || (c.usedFlags != 0 && c.usedFlags != 2) || a.usedQuery ||
        a.cuQuery || a.cuKv || !a.metadata || a.metadata > UINT64_MAX - 4 * kMlaRecordHeaderWords ||
        AscendC::GetBlockNum() != 16) {
        AscendC::Trap();
        return false;
    }
    uint32_t length = 8192;
    if (c.usedFlags & 2) {
        if (!ValidUsedMetadataAddress(a.usedKv, 1)) {
            AscendC::Trap();
            return false;
        }
        AscendC::GlobalTensor<uint32_t> lengths;
        lengths.SetGlobalBuffer((__gm__ uint32_t*)a.usedKv);
        length = lengths.GetValue(0);
        if (length > 8192) {
            AscendC::Trap();
            return false;
        }
    } else if (a.usedKv) {
        AscendC::Trap();
        return false;
    }
    // Each AIC and both AIV lanes validate every active page before any
    // initialization, numerical event allocation or Cube/Vector wait is reached.
    if (c.pageSize) {
        if (ReadDecodePagesSnapshot(c, a.cache, length) != MlaMetadataReadStatus::Success) {
            AscendC::Trap();
            return false;
        }
    }
    const uint32_t blocks = (length + 511) / 512;
    const uint32_t slot = c.pageSize && length ? ((length + 127) / 128 < 4 ? (length + 127) / 128 : 4) - 1 : 3;
    const auto& recordLayout = plan.layouts[slot];
    if (blocks > 1 && (!plan.vectors[blocks] || plan.vectors[blocks] > 16)) {
        AscendC::Trap();
        return false;
    }
    l = recordLayout;
    l.cores = AscendC::GetBlockNum();
    s = {};
    s.aicNum = recordLayout.cores;
    s.usedCoreNum = blocks ? blocks : 1;
    s.fdHeads = blocks > 1;
    s.maxSplits = s.fdHeads ? blocks : 0;
    s.vectors = s.fdHeads ? plan.vectors[blocks] : 0;
    s.groups = l.groups = 1;
    s.partialPrefix[1] = s.maxSplits;
    s.used.query = {0, {1, 0}, 1};
    s.used.kv = {0, {8192, 0}, 8192};
    s.used.usedFlags = c.usedFlags;
    auto& t = s.used.tiling;
    t.batch = t.sq = 1;
    t.sk = 8192;
    t.heads = 128;
    t.mBaseSize = 256;
    t.s2BaseSize = l.n;
    t.usedCoreNum = s.usedCoreNum;
    t.blockDim = recordLayout.cores;
    t.cvRatio = 2;
    t.mm1ResSize = 128 * l.n;
    t.mm2ResSize = 128 * 512;
    for (uint32_t core = 0; core < s.usedCoreNum; ++core) {
        const bool last = core + 1 == s.usedCoreNum;
        s.bEnd[core] = t.bN2End[core] = last;
        s.nEnd[core] = t.s2End[core] = last ? 0 : core + 1;
    }
    s.outOffset = l.outOffset;
    s.sumOffset = l.sumOffset;
    s.maxOffset = l.maxOffset;
    s.bytes = l.bytes;
    read = {&c, &recordLayout, &s, plan.vectorEnds[blocks], length};
    return true;
}

struct MlaQ16DecodeReader {
    const MlaContext* context = nullptr;
    const MlaLayout* recordLayout = nullptr;
    const MlaAutoSplitBsndSchedule* schedule = nullptr;
    detail::MlaQ16PackedState packed{};
    uint32_t kvLength = 0;

    __aicore__ inline uint32_t Byte(uint32_t byte) const
    {
        return uint32_t(packed.word[byte / 8] >> ((byte % 8) * 8)) & 255;
    }
    __aicore__ inline uint32_t FdSplits(uint32_t head) const
    {
        return uint32_t(packed.word[4] >> (head * 6 + 3)) & 7;
    }
    __aicore__ inline uint32_t operator()(uint64_t word) const
    {
        const auto& c = *context;
        const auto& l = *recordLayout;
        const auto& s = *schedule;
        if (word < kMlaRecordHeaderWords) {
            const auto header = MlaHeader(c, l, 16, (kvLength + 127) / 128 * 128);
            return reinterpret_cast<const uint32_t*>(&header)[word];
        }
        if (word < MlaTail(c)) {
            const uint32_t field = word - kMlaRecordHeaderWords;
            return field == 4 || field == 6 ? 16 : field == 5 ? 8192 : field == 7 ? kvLength : 0;
        }
        const uint64_t index = word - MlaTail(c);
        if (index >= kMlaNativePartitionWords) {
            if (index < kMlaNativePartitionWords + 4)
                return index == kMlaNativePartitionWords + 2 ? 8 : 0;
            AscendC::Trap();
            return 0;
        }
        if (index < 3)
            return index == 0 ? c.window.kind : index == 1 ? uint32_t(c.window.left) : uint32_t(c.window.right);
        if (index == kMlaNativeUsedCoreNum)
            return s.usedCoreNum;
        if (index >= kMlaNativeBEnd && index < kMlaNativeBEnd + 26)
            return s.bEnd[index - kMlaNativeBEnd];
        if (index >= kMlaNativeMEnd && index < kMlaNativeMEnd + 26)
            return s.mEnd[index - kMlaNativeMEnd];
        if (index >= kMlaNativeNEnd && index < kMlaNativeNEnd + 26)
            return s.nEnd[index - kMlaNativeNEnd];
        if (index == kMlaNativeFdHeads)
            return s.fdHeads;
        if (index == kMlaNativeMaxSplits)
            return s.maxSplits;
        if (index == kMlaNativeVectors)
            return s.vectors;
        if (index == kMlaNativeFdRows)
            return 8;
        if (index >= kMlaNativeFdM && index < kMlaNativeFdM + 8)
            return uint32_t(packed.word[4] >> ((index - kMlaNativeFdM) * 6)) & 7;
        if (index >= kMlaNativeFdSplits && index < kMlaNativeFdSplits + 8)
            return FdSplits(index - kMlaNativeFdSplits);
        if (index >= kMlaNativeCoreSplitStart && index < kMlaNativeCoreSplitStart + 20)
            return uint32_t(packed.word[3] >> ((index - kMlaNativeCoreSplitStart) * 2)) & 3;
        if (index >= kMlaNativeFdParts && index < kMlaNativeFdParts + s.fdHeads)
            return 32;
        if (index >= kMlaNativeFdLastRows && index < kMlaNativeFdLastRows + s.fdHeads)
            return 8;
        if (index >= kMlaNativeVectorHeadEnd && index < kMlaNativeVectorHeadEnd + 40)
            return Byte(40 + index - kMlaNativeVectorHeadEnd) >> 5;
        if (index >= kMlaNativeVectorPartEnd && index < kMlaNativeVectorPartEnd + 40)
            return Byte(40 + index - kMlaNativeVectorPartEnd) & 31;
        return 0;
    }
};

__aicore__ inline bool LoadQ16Decode(const MlaLaunchArgs& a, const MlaDynamicDecodePlan& plan, MlaQ16DecodeReader& read,
                                     MlaLayout& l, MlaAutoSplitBsndSchedule& s)
{
    const auto& c = a.context;
    if (plan.enabled != 2 || c.valueDim != 512 || c.batch != 1 || c.sq != 16 || c.sk != 8192 || c.heads != 128 ||
        c.kvHeads != 1 || c.splits || c.margin || !c.packed || c.hasSoftcap || a.softcap != 0.0f ||
        c.physicalCores != 20 || c.layout != 3 || c.pageSize != 128 || c.window.kind != 1 || c.window.left != 8191 ||
        c.window.right || (c.usedFlags != 0 && c.usedFlags != 2) || a.usedQuery || a.cuQuery || a.cuKv || !a.metadata ||
        a.metadata > UINT64_MAX - 4 * kMlaRecordHeaderWords || AscendC::GetBlockNum() != 20) {
        AscendC::Trap();
        return false;
    }
    uint32_t length = 8192;
    if (c.usedFlags & 2) {
        if (!ValidUsedMetadataAddress(a.usedKv, 1)) {
            AscendC::Trap();
            return false;
        }
        AscendC::GlobalTensor<uint32_t> lengths;
        lengths.SetGlobalBuffer((__gm__ uint32_t*)a.usedKv);
        length = lengths.GetValue(0);
        if (length > 8192) {
            AscendC::Trap();
            return false;
        }
    } else if (a.usedKv) {
        AscendC::Trap();
        return false;
    }
    if (ReadDecodePagesSnapshot(c, a.cache, length) != MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    const auto packed = detail::LookupMlaQ16Schedule(length);
    const uint32_t control = packed.word[0];
    const auto& recordLayout = plan.layouts[control & 3];
    l = recordLayout;
    l.cores = AscendC::GetBlockNum();
    s = {};
    s.aicNum = recordLayout.cores;
    s.usedCoreNum = (control >> 2) & 31;
    s.fdHeads = (control >> 7) & 15;
    s.maxSplits = (control >> 11) & 7;
    s.vectors = (control >> 14) & 63;
    s.groups = l.groups = 8;
    read = {&c, &recordLayout, &s, packed, length};
    for (uint32_t head = 0; head < s.fdHeads; ++head)
        s.partialPrefix[head + 1] = s.partialPrefix[head] + read.FdSplits(head);
    s.used.query = {0, {16, 0}, 16};
    s.used.kv = {0, {8192, 0}, 8192};
    s.used.usedFlags = c.usedFlags;
    auto& t = s.used.tiling;
    t.batch = 1;
    t.sq = 16;
    t.sk = 8192;
    t.heads = 128;
    t.mBaseSize = 256;
    t.s2BaseSize = l.n;
    t.usedCoreNum = s.usedCoreNum;
    t.blockDim = recordLayout.cores;
    t.cvRatio = 2;
    t.mm1ResSize = 256 * l.n;
    t.mm2ResSize = 256 * 512;
    for (uint32_t core = 0; core < 20; ++core) {
        const uint32_t endpoint = read.Byte(4 + core);
        s.bEnd[core] = t.bN2End[core] = endpoint >> 7;
        s.mEnd[core] = t.gS1End[core] = (endpoint >> 4) & 7;
        s.nEnd[core] = t.s2End[core] = endpoint & 15;
    }
    s.outOffset = l.outOffset;
    s.sumOffset = l.sumOffset;
    s.maxOffset = l.maxOffset;
    s.bytes = l.bytes;
    return true;
}

template <bool HasSoftcap = true, typename Reader>
__aicore__ inline MlaPartialOutputAlias SelectPartialOutputAlias(const MlaLaunchArgs& a, const Reader& read,
                                                                 const MlaAutoSplitBsndSchedule& s)
{
    const auto& c = a.context;
    const auto& t = s.used.tiling;
    // A single query makes the source S1G row order identical to public
    // [split,B,H,Q,D] (or [split,H,T,D]). Every active core owns one positive
    // KV fragment of this sole M group, so its split index is its core index.
    if constexpr (HasSoftcap) {
        if (a.softcap != 0.0f)
            return {};
    }
    if (c.valueDim != 512 || c.batch != 1 || c.kvHeads != 1 || !c.packed || c.splits || c.window.kind ||
        (MlaTnd(c) ? c.totalQ : c.sq) != 1 || c.heads > t.mBaseSize || c.heads % 16 ||
        s.groups != 1 || s.fdHeads != 1 || s.maxSplits <= 1 || s.maxSplits > 26 || s.maxSplits != s.usedCoreNum ||
        read(MlaSequenceWord(0) + 6) != 1)
        return {};
    if (a.accum && a.accumLse)
        return {a.accum, c.heads, a.accumLse};
    // The native workspace reserves one M-base slot per fragment. In this
    // single-group schedule the checked fragment index is the core index.
    return {a.workspace + s.outOffset, t.mBaseSize, 0};
}

template <bool IsTnd, typename Reader>
__aicore__ inline bool FillAutoMergeParams(const MlaPartitionView<Reader>& read, const MlaAutoSplitBsndSchedule& s,
                                          const MlaContext& c, const MlaAutoSplitBsndTask& t,
                                          MlaAutoSplitKvMergeParams& p)
{
    const auto& a = s.used.tiling;
    p.batch = a.batch;
    p.sq = a.sq;
    p.heads = a.heads;
    p.batchIndex = t.batch;
    p.rowStart = t.rowBegin;
    p.rows = t.rowEnd - t.rowBegin;
    p.mBase = a.mBaseSize;
    p.splits = t.splitCount;
    if (p.splits < 2 || p.splits > p.kSplitCapacity)
        return false;
    if constexpr (IsTnd) {
        p.batch = 1;
        p.sq = c.totalQ;
        p.batchIndex = read(8 * t.batch);
    }
    p.maskKind = c.window.kind;
    if (!p.maskKind)
        return true;
    p.queryLength = read(8 * t.batch + 6);
    p.kvLength = read(8 * t.batch + 7);
    p.left = c.window.left;
    p.right = c.window.right;
    auto geometry = t;
    auto_split_detail::Geometry<false>(read, s, c.window, geometry);
    if (auto_split_detail::Parts(s, geometry, UINT32_MAX, nullptr, p.begin, p.end, p.kSplitCapacity) != p.splits)
        return false;
    for (uint32_t split = 0; split < p.splits; ++split) {
        const uint32_t begin = p.begin[split] * a.s2BaseSize, end = p.end[split] * a.s2BaseSize;
        p.begin[split] = geometry.kvBegin > begin ? geometry.kvBegin : begin;
        p.end[split] = geometry.kvEnd < end ? geometry.kvEnd : end;
    }
    return true;
}
template <typename DType, bool IsTnd, bool NativeFd, typename Reader>
__aicore__ inline void AutoMergeImpl(const MlaLaunchArgs& a, const MlaAutoSplitBsndSchedule& s,
                                     const MlaPartitionView<Reader>& read, AscendC::TPipe* pipe,
                                     MlaPartialOutputAlias outputAlias)
{
    const auto& c = a.context;
    if ASCEND_IS_AIV {
        const uint32_t owner = AscendC::GetBlockIdx();
        if (owner >= s.vectors) {
            AscendC::SyncAll();
            return;
        }
        const uint32_t base = auto_split_detail::Base(s.used.tiling.batch);
        const uint32_t firstPart = owner ? read(base + kMlaNativeVectorPartEnd + owner - 1) + 1 : 0;
        const uint32_t lastPart = read(base + kMlaNativeVectorPartEnd + owner);
        pipe->Reset();
        Block::MlaSplitKvMerge<DType, false, IsTnd, 512, false, NativeFd, false, MlaAutoSplitKvMergeParams> block;
        block.InitBuffers(pipe, a.accumLse != 0);
        AscendC::ICachePreLoad(2);
        AscendC::SyncAll();
        if constexpr (NativeFd) {
            // The fast path requires one batch, one query, one FD group and no
            // mask. Its validated split and vector endpoints are sufficient.
            MlaAutoSplitKvMergeParams p{};
            p.batch = p.sq = 1;
            p.heads = p.rows = c.heads;
            p.mBase = s.used.tiling.mBaseSize;
            p.splits = s.maxSplits;
            p.partialOutStride = outputAlias.rows;
            block.Bind(p, (GM_ADDR)outputAlias.out, (GM_ADDR)a.workspace + s.sumOffset,
                       (GM_ADDR)a.workspace + s.maxOffset, (GM_ADDR)a.out, (GM_ADDR)a.lse, nullptr, nullptr);
            const uint32_t tile = block.TileRows();
            for (uint32_t part = firstPart; part <= lastPart; ++part) {
                const uint32_t end = (part + 1) * 8;
                for (uint32_t row = part * 8; row < end; row += tile)
                    block.Run(row, end - row < tile ? end - row : tile);
            }
        } else {
            const uint32_t firstHead = owner ? read(base + kMlaNativeVectorHeadEnd + owner - 1) : 0;
            const uint32_t lastHead = read(base + kMlaNativeVectorHeadEnd + owner);
            // The validated vector endpoints give each lane's FD interval.
            for (uint32_t h = firstHead; h <= lastHead; ++h) {
                const uint32_t parts = h == lastHead ? lastPart + 1 : read(base + kMlaNativeFdParts + h);
                const uint32_t first = h == firstHead ? firstPart : 0;
                if (first >= parts)
                    continue;
                MlaAutoSplitBsndMerge r{};
                MlaAutoSplitBsndTask t{};
                MlaAutoSplitKvMergeParams p{};
                if (!AutoSplitBsndMerge(read, s, h, 0, r) || !AutoSplitBsndTask(read, s, c.window, r.group, 0, t) ||
                    !FillAutoMergeParams<IsTnd>(read, s, c, t, p)) {
                    block.Finish();
                    AscendC::Trap();
                    return;
                }
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
        }
        block.Finish();
    }
}
template <typename DType, bool IsTnd, typename Reader>
__aicore__ inline void AutoMerge(const MlaLaunchArgs& a, const MlaAutoSplitBsndSchedule& s,
                                 const MlaPartitionView<Reader>& read, AscendC::TPipe* pipe,
                                 MlaPartialOutputAlias outputAlias = {})
{
    // Direct groups have already written their outputs and need no FD barrier.
    if (!s.fdHeads)
        return;
    if (outputAlias.out)
        AutoMergeImpl<DType, IsTnd, true>(a, s, read, pipe, outputAlias);
    else
        AutoMergeImpl<DType, IsTnd, false>(a, s, read, pipe, outputAlias);
}

} // namespace flash_attn::mla
#endif
namespace flash_attn::mla {
template <typename DType, bool Paged>
__global__ __aicore__ __mix__(1, 2) void DynamicDecodeAttention(MlaDynamicDecodeLaunchArgs args)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    const auto& a = args.launch;
    MlaDynamicDecodeReader read;
    MlaLayout l{};
    MlaAutoSplitBsndSchedule s;
    if (!LoadDynamicDecode(a, args.plan, read, l, s))
        return;
    AscendC::SetSyncBaseAddr(a.ffts);
    AscendC::TPipe pipe;
    InitializeMlaOutputs<DType>(a, MlaNeedsOutputInit(read, a.context, s));
    if (!read.kvLength)
        return;
    if ASCEND_IS_AIV {
        AscendC::SyncAll();
    }
    const MlaPartitionView<MlaDynamicDecodeReader> view{read, a.context, l};
    const auto outputAlias = SelectPartialOutputAlias<false>(a, read, s);
    Kernel::MlaAutoSplit<DType, false, Paged, false> op;
    // RunAuto only binds spans; its cursor supplies every effective length.
    // Keep a real reserved metadata-tail address for that unused binding.
    op(s, view, a.context.window, a.scale, (GM_ADDR)a.qv, (GM_ADDR)a.q, (GM_ADDR)a.v, (GM_ADDR)a.k, (GM_ADDR)a.out,
       (GM_ADDR)a.lse, (GM_ADDR)a.workspace, &pipe, (GM_ADDR)a.metadata + 4 * kMlaRecordHeaderWords, a.context.totalQ,
       a.inputs, false, 0.0f, (GM_ADDR)a.cache.table.address, &a.cache.layout, a.mask, outputAlias);
    AutoMerge<DType, false>(a, s, view, &pipe, outputAlias);
#endif
}

template <typename DType>
__global__ __aicore__ __mix__(1, 2) void Q16DecodeAttention(MlaDynamicDecodeLaunchArgs args)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    const auto& a = args.launch;
    MlaQ16DecodeReader read;
    MlaLayout l{};
    MlaAutoSplitBsndSchedule s;
    if (!LoadQ16Decode(a, args.plan, read, l, s))
        return;
    AscendC::SetSyncBaseAddr(a.ffts);
    AscendC::TPipe pipe;
    InitializeMlaOutputs<DType>(a, true);
    if (!read.kvLength)
        return;
    if ASCEND_IS_AIV {
        AscendC::SyncAll();
    }
    const MlaPartitionView<MlaQ16DecodeReader> view{read, a.context, l};
    const auto outputAlias = SelectPartialOutputAlias<false>(a, read, s);
    Kernel::MlaAutoSplit<DType, false, true, false> op;
    op(s, view, a.context.window, a.scale, (GM_ADDR)a.qv, (GM_ADDR)a.q, (GM_ADDR)a.v, (GM_ADDR)a.k, (GM_ADDR)a.out,
       (GM_ADDR)a.lse, (GM_ADDR)a.workspace, &pipe, (GM_ADDR)a.metadata + 4 * kMlaRecordHeaderWords, a.context.totalQ,
       a.inputs, false, 0.0f, (GM_ADDR)a.cache.table.address, &a.cache.layout, a.mask, outputAlias);
    AutoMerge<DType, false>(a, s, view, &pipe, outputAlias);
#endif
}

template <typename DType, bool IsTnd, bool Paged, bool HasSoftcap = true>
__global__ __aicore__ __mix__(1, 2) void CompressedAttention(MlaLaunchArgs a)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaReader read;
    MlaLayout l{};
    MlaAutoSplitBsndSchedule s;
    if (!LoadMla<DType>(a, read, l, s))
        return;
    AscendC::SetSyncBaseAddr(a.ffts);
    AscendC::TPipe pipe;
    if ASCEND_IS_AIV {
        // With complete rows, the direct and FD paths write every final O.
        const bool initOutput = MlaNeedsOutputInit<true>(read, a.context, s);
        const bool initLse = initOutput || s.fdHeads;
        InitializeMlaOutputs<DType>(a, initOutput, initLse);
        const bool needsBarrier = initOutput || (initLse && a.lse) || (s.fdHeads && a.accumLse);
        if (needsBarrier && s.groups && !MlaEmptyKv(a.context))
            AscendC::SyncAll();
    }
    if (!s.groups || MlaEmptyKv(a.context))
        return;
    const MlaPartitionView<MlaReader> view{read, a.context, l};
    const auto outputAlias = SelectPartialOutputAlias<HasSoftcap>(a, read, s);
    Kernel::MlaAutoSplit<DType, IsTnd, Paged, HasSoftcap> op;
    op(s, view, a.context.window, a.scale, (GM_ADDR)a.qv, (GM_ADDR)a.q, (GM_ADDR)a.v, (GM_ADDR)a.k, (GM_ADDR)a.out,
       (GM_ADDR)a.lse, (GM_ADDR)a.workspace, &pipe, (GM_ADDR)a.metadata + 4 * kMlaRecordHeaderWords, a.context.totalQ,
       a.inputs, a.context.layout == 1, a.softcap, (GM_ADDR)a.cache.table.address, &a.cache.layout, a.mask,
       outputAlias);
    if (a.context.splits == 0)
        AutoMerge<DType, IsTnd>(a, s, view, &pipe, outputAlias);
#endif
}

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
struct MlaDeviceMetadataArray {
    uint64_t address;
    __aicore__ inline uint32_t operator()(uint64_t word) const
    {
        MlaReader read;
        read.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
        return read(word);
    }
};
struct MlaDeviceMetadataArrays {
    uint64_t address[4];
    __aicore__ inline uint32_t operator()(uint32_t axis, uint32_t word) const
    {
        return MlaDeviceMetadataArray{address[axis]}(word);
    }
};
template <typename DType>
__aicore__ inline bool LoadProvidedDecode(const MlaLaunchArgs& a, MlaLocalMetadataReader& read, uint32_t* snapshot,
                                          MlaLayout& l, MlaAutoSplitBsndSchedule& s)
{
    const auto& c = a.context;
    const uint32_t dtype =
        AscendC::IsSameType<DType, half>::value ? uint32_t(MlaMetadataDtype::Fp16) : uint32_t(MlaMetadataDtype::Bf16);
    MlaLayout allocation{};
    if ((a.checked || c.dtype != dtype || c.valueDim != 512 || c.batch != 1 || c.sq != 1 || c.sk != 8192 ||
         c.heads != 128 || c.kvHeads != 1 || c.splits || c.margin || !c.packed || c.hasSoftcap || a.softcap != 0.0f ||
         c.window.kind || (c.layout != 0 && c.layout != 3) || c.pageSize != (c.layout == 3 ? 128U : 0U) ||
         c.usedFlags != 2 || !MlaRecordLayout(c, allocation) ||
         !ValidUsedMetadataAddress(a.metadata, allocation.words) ||
         AscendC::GetBlockNum() != MlaStaticLaunchCores(c)) &&
        !(a.checked == 0 && c.dtype == dtype && c.valueDim == 512 && c.batch == 1 && c.sq == 16 && c.sk == 8192 &&
          c.heads == 128 && c.kvHeads == 1 && !c.splits && !c.margin && c.packed && !c.hasSoftcap &&
          a.softcap == 0.0f && c.window.kind == 1 && c.window.left == 8191 && c.window.right == 0 && c.layout == 3 &&
          c.pageSize == 128 && c.physicalCores == 20 && c.usedFlags == 2 && MlaRecordLayout(c, allocation) &&
          ValidUsedMetadataAddress(a.metadata, allocation.words) &&
          AscendC::GetBlockNum() == MlaStaticLaunchCores(c))) {
        AscendC::Trap();
        return false;
    }
    const MlaDeviceMetadataArrays arrays{{a.cuQuery, a.cuKv, a.usedQuery, a.usedKv}};
    for (uint32_t i = 0; i < 4; ++i) {
        const uint64_t count = uint64_t(c.batch) + (i < 2), address = arrays.address[i];
        const bool present = i == 0 ? MlaTnd(c) : i == 1 ? c.layout == 1 : c.usedFlags & (1U << (i - 2));
        if ((present && !ValidUsedMetadataAddress(address, count)) || (!present && address)) {
            AscendC::Trap();
            return false;
        }
    }
    SnapshotMlaWords(a.metadata, allocation.words, snapshot);
    read.words = snapshot;
    auto status = ReadMlaMetadata<true>(read, c, arrays, l, s);
    if (status == MlaMetadataReadStatus::Success && c.pageSize)
        status = ReadDecodePagesSnapshot(c, a.cache, read(MlaSequenceWord(0) + 7));
    if (status != MlaMetadataReadStatus::Success || !s.usedCoreNum || s.usedCoreNum > AscendC::GetBlockNum() ||
        s.vectors > 2 * AscendC::GetBlockNum()) {
        AscendC::Trap();
        return false;
    }
    l.cores = AscendC::GetBlockNum();
    return true;
}
#endif

template <typename DType, bool Paged>
__global__ __aicore__ __mix__(1, 2) void ProvidedDecodeAttention(MlaLaunchArgs a)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    // Explicitly align the address: Bisheng may misplace an alignas local array.
    uint32_t storage[400 + 16];
    auto* snapshot = reinterpret_cast<uint32_t*>((reinterpret_cast<uint64_t>(storage) + 63) & ~uint64_t(63));
    MlaLocalMetadataReader read;
    MlaLayout l{};
    MlaAutoSplitBsndSchedule s;
    if (!LoadProvidedDecode<DType>(a, read, snapshot, l, s))
        return;
    AscendC::SetSyncBaseAddr(a.ffts);
    AscendC::TPipe pipe;
    const auto outputAlias = SelectPartialOutputAlias<false>(a, read, s);
    if (outputAlias.out) {
        if ASCEND_IS_AIV {
            // Producers write all active partial rows and FD writes all
            // final rows. Only the unused public LSE slots need filling.
            if (outputAlias.lse) {
                const uint64_t rows = a.context.heads;
                const uint64_t active = rows * s.maxSplits;
                InitializeMla<float>(outputAlias.lse + active * sizeof(float), rows * 26 - active, -float(1.0 / 0.0));
            }
        }
    } else {
        InitializeMlaOutputs<DType>(a, MlaNeedsOutputInit(read, a.context, s));
    }
    if (!s.groups || MlaEmptyKv(a.context))
        return;
    if ASCEND_IS_AIV {
        if (!outputAlias.out)
            AscendC::SyncAll();
    }
    const MlaPartitionView<MlaLocalMetadataReader> view{read, a.context, l};
    Kernel::MlaAutoSplit<DType, false, Paged, false> op;
    op(s, view, a.context.window, a.scale, (GM_ADDR)a.qv, (GM_ADDR)a.q, (GM_ADDR)a.v, (GM_ADDR)a.k, (GM_ADDR)a.out,
       (GM_ADDR)a.lse, (GM_ADDR)a.workspace, &pipe, (GM_ADDR)a.metadata + 4 * kMlaRecordHeaderWords, a.context.totalQ,
       a.inputs, false, 0.0f, (GM_ADDR)a.cache.table.address, &a.cache.layout, a.mask, outputAlias);
    AutoMerge<DType, false>(a, s, view, &pipe, outputAlias);
#endif
}

template <typename DType, bool HasSoftcap>
void LaunchCompressedImpl(const MlaLaunchArgs& a, uint32_t cores)
{
    if (MlaTnd(a.context)) {
        if (a.context.pageSize)
            CompressedAttention<DType, true, true, HasSoftcap><<<cores, nullptr, a.stream>>>(a);
        else
            CompressedAttention<DType, true, false, HasSoftcap><<<cores, nullptr, a.stream>>>(a);
    } else if (a.context.pageSize)
        CompressedAttention<DType, false, true, HasSoftcap><<<cores, nullptr, a.stream>>>(a);
    else
        CompressedAttention<DType, false, false, HasSoftcap><<<cores, nullptr, a.stream>>>(a);
}
template <typename DType>
aclError LaunchCompressed(const MlaLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    const uint32_t cores = MlaStaticLaunchCores(a.context);
    if (a.dynamicDecode) {
        MlaDynamicDecodeLaunchArgs args{a, *a.dynamicDecode};
        args.launch.dynamicDecode = nullptr;
        if (args.plan.enabled == 2)
            Q16DecodeAttention<DType><<<cores, nullptr, a.stream>>>(args);
        else if (a.context.pageSize)
            DynamicDecodeAttention<DType, true><<<cores, nullptr, a.stream>>>(args);
        else
            DynamicDecodeAttention<DType, false><<<cores, nullptr, a.stream>>>(args);
        return launch.Result();
    }
    if (!a.checked) {
        if (a.context.pageSize)
            ProvidedDecodeAttention<DType, true><<<cores, nullptr, a.stream>>>(a);
        else
            ProvidedDecodeAttention<DType, false><<<cores, nullptr, a.stream>>>(a);
        return launch.Result();
    }
    if (a.softcap != 0.0f)
        LaunchCompressedImpl<DType, true>(a, cores);
    else
        LaunchCompressedImpl<DType, false>(a, cores);
    return launch.Result();
}
} // namespace flash_attn::mla
