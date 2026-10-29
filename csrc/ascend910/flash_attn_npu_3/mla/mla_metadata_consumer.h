/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once
#include "mla_metadata.h"
#include "mla_split.h"
#if defined(__NPU_ARCH__)
#define MLA_METADATA_INLINE __aicore__ inline
#else
#define MLA_METADATA_INLINE inline
#endif
namespace flash_attn::mla {
enum class MlaMetadataReadStatus : uint32_t {
    Success = 0,
    InvalidBuffer,
    InvalidHeader,
    GenerationFailed,
    DtypeMismatch,
    ShapeMismatch,
    PlatformMismatch,
    InvalidTiling,
    InvalidEndpoints,
    InvalidWorkspace,
    InsufficientLocalMemory,
    InvalidCacheLengths,
    InvalidPrefixes,
    InvalidUsedLengths,
    InvalidPageTable,
    InvalidPage,
};

template <typename Reader>
MLA_METADATA_INLINE uint64_t ReadMetadataWide(const Reader& read, uint32_t word)
{
    return uint64_t(read(word)) | (uint64_t(read(word + 1)) << 32);
}

struct MlaUsedKernelSchedule {
    MlaDenseKernelSchedule tiling;
    MlaMetadataAllocation query, kv;
    uint32_t usedFlags;
};
static_assert(sizeof(MlaUsedKernelSchedule) == 98 * sizeof(uint32_t));

MLA_METADATA_INLINE uint64_t UsedMetadataTotal(const MlaMetadataAllocation& axis)
{
    return uint64_t(axis.totalTokens.low) | (uint64_t(axis.totalTokens.high) << 32);
}

MLA_METADATA_INLINE bool ValidUsedMetadataAddress(uint64_t address, uint64_t count)
{
    return (address != 0 || count == 0) && address % 4 == 0 && count <= (UINT64_MAX - address) / 4;
}

template <typename Reader, typename Arrays>
MLA_METADATA_INLINE MlaMetadataReadStatus ReadUsedMetadataAxis(const Reader& read, const Arrays& arrays, uint32_t axis,
                                                               const MlaUsedMetadataContext& c, uint32_t& maximum)
{
    using Status = MlaMetadataReadStatus;
    const auto& allocation = axis == 0 ? c.query : c.kv;
    const uint64_t total = UsedMetadataTotal(allocation);
    const bool tnd = allocation.layout == 1, used = (c.usedFlags & (1U << axis)) != 0;
    if (tnd && arrays(axis, 0) != 0) {
        return Status::InvalidPrefixes;
    }
    uint64_t start = 0;
    maximum = tnd ? 0 : allocation.maxLength;
    for (uint32_t b = 0; b < c.batch; ++b) {
        const uint32_t word = kMlaRecordHeaderWords + kMlaSequenceWords * b;
        const uint64_t end = tnd ? uint64_t(arrays(axis, b + 1)) : start + allocation.maxLength;
        if (end < start || end > total || end - start > allocation.maxLength ||
            ReadMetadataWide(read, word + 2 * axis) != start || read(word + 4 + axis) != end - start) {
            return Status::InvalidPrefixes;
        }
        const uint32_t capacity = end - start;
        const uint32_t length = used ? arrays(2 + axis, b) : capacity;
        if (length > capacity || read(word + 6 + axis) != length) {
            return Status::InvalidUsedLengths;
        }
        maximum = capacity > maximum ? capacity : maximum;
        start = end;
    }
    return start == total ? Status::Success : Status::InvalidPrefixes;
}

MLA_METADATA_INLINE bool ValidPagedMetadataAddress(uint64_t address, uint64_t count, uint32_t bytes = 4)
{
    return (address != 0 || count == 0) && address % bytes == 0 && count <= (UINT64_MAX - address) / bytes;
}

// Validate the full physical view, including pages unused by this invocation.
// Default Hkv=1 preserves compressed MLA; prefill passes its actual KV heads.
// Singleton/empty strides stay unused.
MLA_METADATA_INLINE bool ValidPagedCacheTensor(const MlaBsndTensorLayout& x, const MlaPagedMetadataArray& buffer,
                                               uint64_t pages, uint64_t pageSize, uint64_t dim, uint64_t heads = 1,
                                               uint32_t bytes = 2)
{
    if (!heads || !pageSize || x.batchStride > INT64_MAX || x.rowStride > INT64_MAX || x.headStride > INT64_MAX ||
        !ValidPagedMetadataAddress(buffer.address, buffer.count, bytes)) {
        return false;
    }
    uint64_t span = 0;
    if (pages != 0) {
        const uint64_t maxElements = INT64_MAX / bytes;
        if (heads > 1 && x.headStride > (maxElements - dim) / (heads - 1))
            return false;
        dim += (heads - 1) * x.headStride;
        if (pageSize > 1 && x.rowStride > (maxElements - dim) / (pageSize - 1)) {
            return false;
        }
        span = dim + (pageSize - 1) * x.rowStride;
        if (pages > 1 && x.batchStride > (maxElements - span) / (pages - 1)) {
            return false;
        }
        span += (pages - 1) * x.batchStride;
    }
    return x.spanElements == span && buffer.count >= span;
}

MLA_METADATA_INLINE MlaMetadataReadStatus CheckPagedCacheDescriptor(const MlaContext& c,
                                                                    const MlaPagedCacheInputs& input,
                                                                    uint64_t heads = 1, uint64_t keyDim = 64,
                                                                    uint64_t valueDim = 512)
{
    using Status = MlaMetadataReadStatus;
    const auto& p = input.layout;
    if (!c.pageSize || p.batchSize != c.batch || p.pageSize != c.pageSize || p.logicalCapacity != c.sk ||
        p.maxPages != c.sk / c.pageSize || p.physicalPages > uint64_t(INT32_MAX) + 1) {
        return Status::ShapeMismatch;
    }
    if (p.tableRowStride > INT64_MAX || !ValidPagedMetadataAddress(input.table.address, input.table.count)) {
        return Status::InvalidPageTable;
    }
    uint64_t span = 0;
    if (p.batchSize != 0 && p.maxPages != 0) {
        constexpr uint64_t maxElements = INT64_MAX / 4;
        if (p.batchSize > 1 && p.tableRowStride > (maxElements - p.maxPages) / (p.batchSize - 1)) {
            return Status::InvalidPageTable;
        }
        span = (p.batchSize - 1) * p.tableRowStride + p.maxPages;
    }
    if (p.tableSpanElements != span || input.table.count < span) {
        return Status::InvalidPageTable;
    }
    if (!ValidPagedCacheTensor(p.k, input.k, p.physicalPages, p.pageSize, keyDim, heads) ||
        !ValidPagedCacheTensor(p.v, input.v, p.physicalPages, p.pageSize, valueDim, heads,
                               c.dtype == uint32_t(MlaMetadataDtype::Int8) ? 1 : 2)) {
        return Status::InvalidBuffer;
    }
    return Status::Success;
}

struct MlaAutoSplitBsndSchedule {
    MlaUsedKernelSchedule used;
    uint32_t aicNum, usedCoreNum, fdHeads, maxSplits, vectors;
    uint32_t bEnd[26], mEnd[26], nEnd[26];
    uint64_t groups, partialPrefix[27], outOffset, sumOffset, maxOffset, bytes;
};
// Invocation-local tiling, checked on AICPU before attention runs. The public
// metadata remains immutable and is checked again on every reuse. Reuse the
// existing decoded schedule; this is not another serialized metadata version.
struct alignas(64) MlaCheckedAutoSplitSchedule {
    MlaMetadataReadStatus status = MlaMetadataReadStatus::InvalidBuffer;
    uint32_t blockDim = 0, dtype = 0;
    alignas(64) MlaAutoSplitBsndSchedule schedule{};
};
static_assert(std::is_trivially_copyable_v<MlaCheckedAutoSplitSchedule>);
static_assert(sizeof(MlaCheckedAutoSplitSchedule) % 64 == 0);
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
template <typename Checked, typename Schedule>
__aicore__ inline bool LoadCheckedSchedule(uint64_t address, uint32_t dtype, Schedule& schedule,
                                           uint32_t* mode = nullptr)
{
    if (!address || address % alignof(Checked) || address > UINT64_MAX - sizeof(Checked)) {
        AscendC::Trap();
        return false;
    }
    // Match CANN InitTilingData: AIC reads GM; AIV stages through UB before
    // TPipe owns it. Copy directly into the schedule used by the task ring.
    constexpr uint64_t offset = offsetof(Checked, schedule);
#if defined(__DAV_CUBE__)
    const auto* header = reinterpret_cast<__gm__ uint32_t*>(address);
    copy_data_align64(reinterpret_cast<uint8_t*>(&schedule), reinterpret_cast<__gm__ uint8_t*>(address + offset),
                      sizeof(schedule));
#else
    auto* staging = reinterpret_cast<__ubuf__ uint8_t*>(get_imm(0));
    copy_gm_to_ubuf(staging, reinterpret_cast<__gm__ uint8_t*>(address), 0, 1, sizeof(Checked) / 32, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    const auto* header = reinterpret_cast<__ubuf__ uint32_t*>(staging);
    copy_data_align64(reinterpret_cast<uint8_t*>(&schedule), staging + offset, sizeof(schedule));
#endif
    const uint32_t status = header[0], blockDim = header[1], actualDtype = header[2];
    pipe_barrier(PIPE_ALL);
    if (status != uint32_t(MlaMetadataReadStatus::Success) || blockDim != AscendC::GetBlockNum() ||
        actualDtype != dtype) {
        AscendC::Trap();
        return false;
    }
    if (mode)
        *mode = header[3];
    return true;
}
__aicore__ inline bool LoadCheckedAutoSplitSchedule(uint64_t address, uint32_t dtype,
                                                    MlaAutoSplitBsndSchedule& schedule)
{
    return LoadCheckedSchedule<MlaCheckedAutoSplitSchedule>(address, dtype, schedule);
}
#endif
struct MlaAutoSplitBsndTask {
    uint64_t group, slot, outOffset, sumOffset, maxOffset;
    uint32_t batch, m, split, splitCount, owner, fdHead, kind; // Direct=0, Partial=1, Empty=2
    uint32_t rowBegin, rowEnd, s2Begin, s2End, kvBegin, kvEnd;
};
struct MlaAutoSplitBsndMerge {
    uint64_t group, firstSlot, outOffset, sumOffset, maxOffset;
    uint32_t batch, m, head, part, splitCount, rowBegin, rowEnd;
};
namespace auto_split_detail {
template <bool IsPaged = false>
MLA_METADATA_INLINE uint32_t Base(uint32_t batch)
{
    return 8 * batch;
}
template <bool IsPaged, typename Reader>
MLA_METADATA_INLINE uint32_t QueryLength(const Reader& read, uint32_t batch)
{
    return read(8 * batch + 6);
}
template <bool IsPaged, typename Reader>
MLA_METADATA_INLINE uint32_t KvLength(const Reader& read, uint32_t batch)
{
    return read(8 * batch + 7);
}
MLA_METADATA_INLINE bool Less(uint32_t b, uint32_t m, uint32_t n, uint32_t z, uint32_t j, uint32_t k)
{
    return b < z || (b == z && (m < j || (m == j && n < k)));
}
// The caller has already checked the span and M index. All products are wide.
template <bool IsPaged, typename Reader>
MLA_METADATA_INLINE void Geometry(const Reader& read, const MlaAutoSplitBsndSchedule& s, const MlaWindowKernelParams& w,
                                  MlaAutoSplitBsndTask& t)
{
    const auto& a = s.used.tiling;
    const uint32_t q = QueryLength<IsPaged>(read, t.batch), k = KvLength<IsPaged>(read, t.batch);
    const uint64_t row = uint64_t(t.m) * a.mBaseSize, rows = uint64_t(q) * a.heads;
    t.rowBegin = row;
    t.rowEnd = row + a.mBaseSize < rows ? row + a.mBaseSize : rows;
    int64_t lo = 0, hi = k;
    if (w.kind) {
        hi = int64_t((t.rowEnd - 1) / a.heads) + k - q + (w.kind == 2 ? w.right : 0) + 1;
        hi = hi < 0 ? 0 : hi > k ? k : hi;
        if (w.kind == 2) {
            lo = int64_t(row / a.heads) + k - q - w.left;
            lo = lo < 0 ? 0 : lo > k ? k : lo;
        }
    }
    if (lo >= hi)
        lo = hi = 0;
    t.kvBegin = lo;
    t.kvEnd = hi;
    t.s2Begin = uint64_t(lo) / a.s2BaseSize;
    t.s2End = (uint64_t(hi) + a.s2BaseSize - 1) / a.s2BaseSize;
}
// Adjacent exclusive endpoints form disjoint intervals. Intersect them with
// this M's visible interval; do not uniformly divide by maxSplits.
MLA_METADATA_INLINE uint32_t Parts(const MlaAutoSplitBsndSchedule& s, MlaAutoSplitBsndTask& t,
                                   uint32_t owner = UINT32_MAX, uint32_t* ownerSplit = nullptr,
                                   uint16_t* begin = nullptr, uint16_t* end = nullptr)
{
    const uint32_t start = t.s2Begin, stop = t.s2End;
    if (start >= stop)
        return 0;
    // Endpoints are strictly increasing and checked before any task query.
    // Positive intersections form one contiguous interval of cores. Locate
    // its boundaries instead of rescanning every core for each FD fragment.
    uint32_t lo = 0, hi = s.usedCoreNum;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (Less(t.batch, t.m, start, s.bEnd[mid], s.mEnd[mid], s.nEnd[mid]))
            hi = mid;
        else
            lo = mid + 1;
    }
    const uint32_t first = lo;
    if (first == s.usedCoreNum)
        return 0;
    hi = s.usedCoreNum;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (Less(s.bEnd[mid], s.mEnd[mid], s.nEnd[mid], t.batch, t.m, stop))
            lo = mid + 1;
        else
            hi = mid;
    }
    const uint32_t last = lo < s.usedCoreNum ? lo + 1 : lo;
    const uint32_t count = last - first;
    if (ownerSplit && owner >= first && owner < last)
        *ownerSplit = owner - first;
    if (!begin && t.split >= count)
        return count;
    const uint32_t from = begin ? first : first + t.split;
    const uint32_t to = begin ? last : (from < last ? from + 1 : from);
    for (uint32_t core = from; core < to; ++core) {
        uint32_t low = start, high = stop;
        if (core && t.batch == s.bEnd[core - 1] && t.m == s.mEnd[core - 1])
            low = low > s.nEnd[core - 1] ? low : s.nEnd[core - 1];
        if (t.batch == s.bEnd[core] && t.m == s.mEnd[core])
            high = high < s.nEnd[core] ? high : s.nEnd[core];
        const uint32_t split = core - first;
        if (begin) {
            begin[split] = low;
            end[split] = high;
        }
        if (split == t.split) {
            t.owner = core;
            t.s2Begin = low;
            t.s2End = high;
        }
    }
    return count;
}
MLA_METADATA_INLINE void Offsets(const MlaAutoSplitBsndSchedule& s, uint64_t row, uint64_t& o, uint64_t& l, uint64_t& m)
{
    o = s.outOffset + row * 512 * 4;
    l = s.sumOffset + row * 8 * 4;
    m = s.maxOffset + row * 8 * 4;
}
} // namespace auto_split_detail

template <typename Reader>
MLA_METADATA_INLINE MlaMetadataReadStatus ReadAutoSplitPartitions(const Reader& read, const MlaUsedMetadataContext& c,
                                                                  const MlaWindowKernelParams& w,
                                                                  const MlaUsedKernelSchedule& used,
                                                                  MlaAutoSplitBsndSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    using namespace auto_split_detail;
    const uint32_t base = Base<false>(c.batch), prefixWord = base + kMlaNativePartitionWords;
    MlaAutoSplitBsndSchedule s{};
    s.used = used;
    auto& a = s.used.tiling;
    const bool empty = !c.batch || !a.sq || !a.sk;
    if (!empty && a.blockDim != c.aicNum)
        return Status::PlatformMismatch;
    s.aicNum = c.aicNum;
    s.usedCoreNum = read(base + kMlaNativeUsedCoreNum);
    s.fdHeads = read(base + kMlaNativeFdHeads);
    s.maxSplits = read(base + kMlaNativeMaxSplits);
    s.vectors = read(base + kMlaNativeVectors);
    if (!s.usedCoreNum || s.usedCoreNum > c.aicNum || s.fdHeads > c.aicNum || s.maxSplits > c.aicNum ||
        s.vectors > s.usedCoreNum * 2 || read(base + kMlaNativeFdRows) != 8)
        return Status::InvalidTiling;
    uint64_t prefix = 0;
    uint32_t lastBatch = 0, lastM = 0, lastN = 0;
    bool live = false;
    for (uint32_t b = 0; b <= c.batch; ++b) {
        if (ReadMetadataWide(read, prefixWord + 2 * b) != prefix)
            return Status::InvalidPrefixes;
        if (b == c.batch || empty)
            continue;
        const uint64_t rows = uint64_t(QueryLength<false>(read, b)) * a.heads;
        const uint32_t count = (rows + a.mBaseSize - 1) / a.mBaseSize, k = KvLength<false>(read, b);
        prefix += count;
        // Right-bottom causal/local bands always include the last row's last
        // key when both lengths are nonzero. This bounds all earlier tiles
        // without enumerating a potentially enormous number of M blocks.
        if (count && k) {
            live = true;
            lastBatch = b;
            lastM = count - 1;
            lastN = (k + a.s2BaseSize - 1) / a.s2BaseSize - 1;
        }
    }
    s.groups = prefix;
    for (uint32_t core = 0; core < 26; ++core) {
        const uint32_t b = read(base + kMlaNativeBEnd + core), m = read(base + kMlaNativeMEnd + core),
                       n = read(base + kMlaNativeNEnd + core);
        if (core >= s.usedCoreNum) {
            if (b || m || n)
                return Status::InvalidEndpoints;
            continue;
        }
        if (b > c.batch || (b == c.batch && (m || n)))
            return Status::InvalidEndpoints;
        if (b < c.batch) {
            const uint64_t rows = uint64_t(QueryLength<false>(read, b)) * a.heads;
            const uint32_t count = empty ? 0 : (rows + a.mBaseSize - 1) / a.mBaseSize;
            if (m >= (count ? count : 1) || (!count && n))
                return Status::InvalidEndpoints;
            if (n) {
                MlaAutoSplitBsndTask t{};
                t.batch = b;
                t.m = m;
                Geometry<false>(read, s, w, t);
                // Native cost partitioning can stop at/before the first
                // visible tile of a local window. That empty prefix belongs
                // to neither an FD fragment nor persistent storage.
                if (n > t.s2End)
                    return Status::InvalidEndpoints;
            }
        }
        if (core && !Less(s.bEnd[core - 1], s.mEnd[core - 1], s.nEnd[core - 1], b, m, n))
            return Status::InvalidEndpoints;
        s.bEnd[core] = b;
        s.mEnd[core] = m;
        s.nEnd[core] = n;
    }
    const uint32_t last = s.usedCoreNum - 1;
    if ((live && !Less(lastBatch, lastM, lastN, s.bEnd[last], s.mEnd[last], s.nEnd[last])) ||
        (!live && s.usedCoreNum != 1))
        return Status::InvalidEndpoints;
    // Every cross-core M must contain an interior endpoint. Reconstruct the
    // sorted FD list from those endpoints, then check all stored bookkeeping.
    uint32_t heads = 0, maxSplits = 0;
    uint32_t previousB = UINT32_MAX, previousM = UINT32_MAX;
    uint32_t fdParts[27]{};
    for (uint32_t core = 0; core + 1 < s.usedCoreNum; ++core) {
        if (!s.nEnd[core] || (s.bEnd[core] == previousB && s.mEnd[core] == previousM))
            continue;
        MlaAutoSplitBsndTask t{};
        t.batch = s.bEnd[core];
        t.m = s.mEnd[core];
        Geometry<false>(read, s, w, t);
        if (s.nEnd[core] <= t.s2Begin || s.nEnd[core] == t.s2End)
            continue;
        const uint32_t count = Parts(s, t), rows = t.rowEnd - t.rowBegin;
        if (count < 2 || heads >= s.fdHeads || read(base + kMlaNativeFdBatch + heads) != t.batch ||
            read(base + kMlaNativeFdM + heads) != t.m || read(base + kMlaNativeFdSplits + heads) != count ||
            read(base + kMlaNativeFdParts + heads) != (rows + 7) / 8 ||
            read(base + kMlaNativeFdLastRows + heads) != (rows - 1) % 8 + 1)
            return Status::InvalidEndpoints;
        s.partialPrefix[heads + 1] = s.partialPrefix[heads] + count;
        fdParts[heads + 1] = fdParts[heads] + (rows + 7) / 8;
        maxSplits = maxSplits > count ? maxSplits : count;
        previousB = t.batch;
        previousM = t.m;
        ++heads;
    }
    if (heads != s.fdHeads || maxSplits != s.maxSplits || s.partialPrefix[heads] > uint64_t(2) * c.aicNum ||
        (heads == 0 ? s.vectors != 0 : s.vectors == 0))
        return Status::InvalidEndpoints;
    for (uint32_t i = 0; i < 26; ++i) {
        if (i >= heads && (read(base + kMlaNativeFdBatch + i) || read(base + kMlaNativeFdM + i) ||
                           read(base + kMlaNativeFdSplits + i) || read(base + kMlaNativeFdParts + i) ||
                           read(base + kMlaNativeFdLastRows + i)))
            return Status::InvalidEndpoints;
        uint32_t startSplit = 0;
        if (i && i < s.usedCoreNum && s.nEnd[i - 1]) {
            MlaAutoSplitBsndTask t{};
            t.batch = s.bEnd[i - 1];
            t.m = s.mEnd[i - 1];
            Geometry<false>(read, s, w, t);
            // Parts supplies the inverse owner -> split mapping in one scan.
            Parts(s, t, i, &startSplit);
        }
        if (read(base + kMlaNativeCoreSplitStart + i) != startSplit)
            return Status::InvalidEndpoints;
    }
    uint32_t end = 0;
    for (uint32_t i = 0; i < 52; ++i) {
        const uint32_t h = read(base + kMlaNativeVectorHeadEnd + i), part = read(base + kMlaNativeVectorPartEnd + i);
        if (i >= s.vectors) {
            if (h || part)
                return Status::InvalidEndpoints;
        } else {
            if (h >= heads || part >= fdParts[h + 1] - fdParts[h] || fdParts[h] + part + 1 <= end)
                return Status::InvalidEndpoints;
            end = fdParts[h] + part + 1;
        }
    }
    if (end != fdParts[heads])
        return Status::InvalidEndpoints;
    output = s;
    return Status::Success;
}

template <bool IsPaged = false, typename Reader>
MLA_METADATA_INLINE bool AutoSplitBsndTask(const Reader& read, const MlaAutoSplitBsndSchedule& s,
                                           const MlaWindowKernelParams& window, uint64_t group, uint32_t split,
                                           MlaAutoSplitBsndTask& output, uint32_t owner = UINT32_MAX)
{
    using namespace auto_split_detail;
    if (group >= s.groups)
        return false;
    const auto& a = s.used.tiling;
    const uint32_t base = Base<IsPaged>(a.batch), prefix = base + kMlaNativePartitionWords;
    uint32_t lo = 0, hi = a.batch;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (ReadMetadataWide(read, prefix + 2 * (mid + 1)) <= group)
            lo = mid + 1;
        else
            hi = mid;
    }
    MlaAutoSplitBsndTask t{};
    t.group = group;
    t.slot = UINT64_MAX;
    t.fdHead = UINT32_MAX;
    t.batch = lo;
    t.m = group - ReadMetadataWide(read, prefix + 2 * lo);
    Geometry<IsPaged>(read, s, window, t);
    const bool empty = t.s2Begin == t.s2End;
    // Parts writes ownerSplit before selecting t.split. A missing owner
    // leaves the sentinel unchanged and cannot select another core's fragment.
    t.split = owner == UINT32_MAX ? split : empty ? 0 : UINT32_MAX;
    const uint32_t count = empty ? 1 : Parts(s, t, owner, owner == UINT32_MAX ? nullptr : &t.split);
    if (t.split >= count)
        return false;
    t.splitCount = count;
    t.kind = empty ? 2 : count > 1 ? 1 : 0;
    if (empty) {
        t.owner = group % s.aicNum;
        if (owner != UINT32_MAX && owner != t.owner)
            return false;
    } else {
        const uint32_t begin = t.s2Begin * a.s2BaseSize, end = t.s2End * a.s2BaseSize;
        t.kvBegin = t.kvBegin > begin ? t.kvBegin : begin;
        t.kvEnd = t.kvEnd < end ? t.kvEnd : end;
    }
    if (t.kind == 1) {
        for (uint32_t h = 0; h < s.fdHeads; ++h)
            if (read(base + kMlaNativeFdBatch + h) == t.batch && read(base + kMlaNativeFdM + h) == t.m)
                t.fdHead = h;
        if (t.fdHead == UINT32_MAX)
            return false;
        t.slot = s.partialPrefix[t.fdHead] + t.split;
        Offsets(s, t.slot * a.mBaseSize, t.outOffset, t.sumOffset, t.maxOffset);
    }
    output = t;
    return true;
}
template <bool IsPaged = false, typename Reader>
MLA_METADATA_INLINE bool AutoSplitBsndMerge(const Reader& read, const MlaAutoSplitBsndSchedule& s, uint32_t head,
                                            uint32_t part, MlaAutoSplitBsndMerge& output)
{
    using namespace auto_split_detail;
    const auto& a = s.used.tiling;
    const uint32_t base = Base<IsPaged>(a.batch);
    if (head >= s.fdHeads || part >= read(base + kMlaNativeFdParts + head))
        return false;
    MlaAutoSplitBsndMerge r{};
    r.head = head;
    r.part = part;
    r.batch = read(base + kMlaNativeFdBatch + head);
    r.m = read(base + kMlaNativeFdM + head);
    r.group = ReadMetadataWide(read, base + kMlaNativePartitionWords + 2 * r.batch) + r.m;
    r.firstSlot = s.partialPrefix[head];
    r.splitCount = read(base + kMlaNativeFdSplits + head);
    r.rowBegin = uint64_t(r.m) * a.mBaseSize + part * 8;
    const uint64_t end = uint64_t(QueryLength<IsPaged>(read, r.batch)) * a.heads;
    const uint64_t blockEnd = uint64_t(r.m + 1) * a.mBaseSize;
    const uint64_t limit = end < blockEnd ? end : blockEnd;
    r.rowEnd = uint64_t(r.rowBegin) + 8 < limit ? uint64_t(r.rowBegin) + 8 : limit;
    Offsets(s, r.firstSlot * a.mBaseSize + part * 8, r.outOffset, r.sumOffset, r.maxOffset);
    output = r;
    return true;
}

// Queries require an unchanged, successfully validated record and context.
// All failures leave output untouched. Effective lengths govern mask geometry;
// allocation prefixes govern physical addresses and preserve unused Q gaps.

// CANN scheduling flattens (batch, KV head group). This coordinate view maps
// those logical batches to the one physical allocation snapshot.
template <typename Reader>
struct MlaPartitionView {
    const Reader& read;
    const MlaContext& context;
    const MlaLayout& layout;
    MLA_METADATA_INLINE uint32_t operator()(uint32_t i) const
    {
        const uint32_t base = 8 * context.batch * layout.headGroups;
        if (i >= base)
            return read(MlaTail(context) + i - base);
        return read(MlaSequenceWord((i / 8) / layout.headGroups) + i % 8);
    }
};
template <typename Reader, typename Arrays>
MLA_METADATA_INLINE MlaMetadataReadStatus ReadMlaMetadata(const Reader& read, const MlaContext& c, const Arrays& arrays,
                                                          MlaLayout& layout, MlaAutoSplitBsndSchedule& output)
{
    using S = MlaMetadataReadStatus;
    MlaLayout l{};
    if (read(2))
        return S::GenerationFailed;
    if (!MlaRecordLayout(c, l, read(22), read(23)) || read(0) != l.words)
        return S::InvalidBuffer;
    MlaUsedMetadataContext allocation{};
    allocation.batch = c.batch;
    allocation.usedFlags = c.usedFlags;
    const uint64_t totalQ = MlaTnd(c) ? c.totalQ : uint64_t(c.batch) * c.sq;
    const uint64_t totalK = c.layout == 1 ? c.totalK : uint64_t(c.batch) * c.sk;
    allocation.query = {uint32_t(MlaTnd(c)), {uint32_t(totalQ), uint32_t(totalQ >> 32)}, c.sq};
    allocation.kv = {uint32_t(c.layout == 1), {uint32_t(totalK), uint32_t(totalK >> 32)}, c.sk};
    uint32_t qCapacity = 0, kCapacity = 0;
    auto status = ReadUsedMetadataAxis(read, arrays, 0, allocation, qCapacity);
    if (status != S::Success)
        return status;
    status = ReadUsedMetadataAxis(read, arrays, 1, allocation, kCapacity);
    if (status != S::Success)
        return status;
    if (!c.batch) {
        qCapacity = MlaTnd(c) ? 0 : c.sq;
        kCapacity = c.layout == 1 ? 0 : c.sk;
    }
    if (c.pageSize) {
        uint32_t maxKv = 0;
        for (uint32_t b = 0; b < c.batch; ++b)
            if (read(MlaSequenceWord(b) + 7) > maxKv)
                maxKv = read(MlaSequenceWord(b) + 7);
        kCapacity = (maxKv + c.pageSize - 1) / c.pageSize * c.pageSize;
    }
    const auto header = MlaHeader(c, l, qCapacity, kCapacity);
    const auto* expected = reinterpret_cast<const uint32_t*>(&header);
    for (uint32_t i = 0; i < kMlaRecordHeaderWords; ++i)
        if (read(i) != expected[i])
            return S::ShapeMismatch;
    if (l.bytes > c.workspaceBytes)
        return S::InvalidWorkspace;
    MlaUsedKernelSchedule used{};
    used.query = allocation.query;
    used.kv = allocation.kv;
    used.usedFlags = c.usedFlags;
    auto& t = used.tiling;
    t.batch = c.batch * l.headGroups;
    t.sq = c.sq;
    t.sk = c.sk ? c.sk : 1;
    t.heads = l.g;
    t.mBaseSize = l.m;
    t.s2BaseSize = l.n;
    t.blockDim = l.cores;
    t.cvRatio = 2;
    const uint64_t rows = ((uint64_t(qCapacity) * l.g < l.m ? uint64_t(qCapacity) * l.g : l.m) + 15) / 16 * 16;
    t.mm1ResSize = rows * ((l.n + 31) / 32 * 32);
    t.mm2ResSize = rows * 512;
    MlaUsedMetadataContext logical{};
    logical.batch = t.batch;
    logical.heads = t.heads;
    logical.aicNum = l.cores;
    logical.query.maxLength = c.sq;
    logical.kv.maxLength = c.sk;
    MlaAutoSplitBsndSchedule s{};
    status = ReadAutoSplitPartitions(MlaPartitionView<Reader>{read, c, l}, logical, c.window, used, s);
    if (status != S::Success)
        return status;
    if (s.groups > l.groups)
        return S::InvalidPrefixes;
    s.outOffset = l.outOffset;
    s.sumOffset = l.sumOffset;
    s.maxOffset = l.maxOffset;
    s.bytes = l.bytes;
    s.used.tiling.usedCoreNum = s.usedCoreNum;
    for (uint32_t i = 0; i < 26; ++i) {
        s.used.tiling.bN2End[i] = s.bEnd[i];
        s.used.tiling.gS1End[i] = s.mEnd[i];
        s.used.tiling.s2End[i] = s.nEnd[i];
    }
    output = s;
    layout = l;
    return S::Success;
}
template <typename Reader, typename Pages>
MLA_METADATA_INLINE MlaMetadataReadStatus ReadMlaPages(const Reader& read, const MlaContext& c,
                                                       const MlaPagedCacheInputs& cache, const Pages& pages)
{
    using S = MlaMetadataReadStatus;
    auto status = CheckPagedCacheDescriptor(c, cache, c.kvHeads, c.valueDim == 512 ? 64 : 192, c.valueDim);
    if (status != S::Success)
        return status;
    for (uint32_t b = 0; b < c.batch; ++b) {
        const uint32_t length = read(MlaSequenceWord(b) + 7);
        if (length > c.sk)
            return S::InvalidCacheLengths;
        for (uint32_t i = 0; i < (length + c.pageSize - 1) / c.pageSize; ++i) {
            const int32_t id = int32_t(pages(uint64_t(b) * cache.layout.tableRowStride + i));
            if (id < 0 || uint64_t(id) >= cache.layout.physicalPages)
                return S::InvalidPageTable;
        }
    }
    return S::Success;
}
// Calculate a fragment using original effective Q/KV geometry, never a sliced Q.
MLA_METADATA_INLINE void SplitBsndFragment(uint32_t q, uint32_t k, uint32_t heads, uint32_t m, uint32_t mBase,
                                           uint32_t nBase, const MlaWindowKernelParams& w, uint32_t splits,
                                           uint32_t split, uint32_t& begin, uint32_t& end)
{
    const uint64_t row = uint64_t(m) * mBase;
    const uint64_t last = row + mBase < uint64_t(q) * heads ? row + mBase : uint64_t(q) * heads;
    int64_t lo = 0, hi = k;
    if (w.kind) {
        hi = int64_t((last - 1) / heads) + k - q + (w.kind == 2 ? w.right : 0) + 1;
        hi = hi < 0 ? 0 : hi > k ? k : hi;
        if (w.kind == 2) {
            lo = int64_t(row / heads) + k - q - w.left;
            lo = lo < 0 ? 0 : lo > k ? k : lo;
        }
    }
    if (lo >= hi) {
        begin = end = 0;
        return;
    }
    const uint32_t first = uint64_t(lo) / nBase, stop = (uint64_t(hi) + nBase - 1) / nBase;
    const uint32_t width = (stop - first + splits - 1) / splits;
    begin = first + split * width;
    begin = begin > stop ? stop : begin;
    end = begin + width > stop ? stop : begin + width;
}

template <typename Reader>
MLA_METADATA_INLINE bool MlaTask(const Reader& read, const MlaContext& c, const MlaLayout& l,
                                 const MlaAutoSplitBsndSchedule& s, uint64_t group, uint32_t split,
                                 MlaAutoSplitBsndTask& t, uint32_t owner = UINT32_MAX)
{
    if (split >= l.capacity || !AutoSplitBsndTask(MlaPartitionView<Reader>{read, c, l}, s, c.window, group,
                                                  c.splits ? 0 : split, t, c.splits ? UINT32_MAX : owner))
        return false;
    if (c.splits) {
        const uint32_t b = t.batch / l.headGroups;
        t.split = split;
        t.splitCount = c.splits;
        SplitBsndFragment(read(MlaSequenceWord(b) + 6), read(MlaSequenceWord(b) + 7), l.g, t.m, l.m, l.n, c.window,
                          c.splits, split, t.s2Begin, t.s2End);
    }
    return true;
}
MLA_METADATA_INLINE uint32_t MlaTaskSlots(const MlaContext& c, const MlaAutoSplitBsndSchedule& s)
{
    return c.splits ? c.splits : (s.maxSplits ? s.maxSplits : 1);
}
} // namespace flash_attn::mla
#undef MLA_METADATA_INLINE
