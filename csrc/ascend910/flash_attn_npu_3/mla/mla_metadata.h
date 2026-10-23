/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "mla_tiling.h"
#include "mla_layout.h"
#include "mla_params.h"
#if defined(__CCE__)
#define MLA_METADATA_INLINE inline[host, aicore]
#else
#define MLA_METADATA_INLINE inline
#endif
namespace flash_attn::mla {
enum class MlaMetadataDtype : uint32_t {
    Fp16 = 1,
    Bf16 = 2,
    Int8 = 3
};
struct MlaMetadataWide {
    uint32_t low, high;
};
struct MlaMetadataAllocation {
    uint32_t layout;
    MlaMetadataWide totalTokens;
    uint32_t maxLength;
};
struct MlaUsedMetadataContext {
    uint32_t batch = 0, heads = 0, aicNum = 0, usedFlags = 0;
    MlaMetadataAllocation query{}, kv{};
};
struct MlaPagedMetadataArray {
    uint64_t address = 0, count = 0;
};
struct MlaPagedCacheInputs {
    MlaPagedCacheLayout layout{};
    MlaPagedMetadataArray table{}, k{}, v{};
};
// Invocation geometry. Addresses, strides, scale and scratch stay in launch arguments.
// Layout: BSND=0, TND Q/KV=1, mixed=2, BSND PA=3, TND PA=4.
struct MlaContext {
    uint32_t batch, sq, sk, heads, kvHeads, splits, dtype;
    uint32_t physicalCores, margin, packed, layout, totalQ, totalK, pageSize, usedFlags, valueDim, hasSoftcap;
    MlaWindowKernelParams window;
    uint64_t workspaceBytes = 0;
};
struct MlaLayout {
    uint32_t words = 0, m = 0, n = 0, cores = 0, headGroups = 0, g = 0, capacity = 0;
    uint64_t groups = 0, scratch = 0, outOffset = 0, sumOffset = 0, maxOffset = 0, bytes = 0;
};
// Fixed semantic header, followed by B allocation/used spans, one CANN native
// partition, and B*headGroups+1 wide M prefixes. There are no format versions.
struct MlaRecordHeader {
    uint32_t words, dtype, status, batch, sq, sk, heads, kvHeads, splits, physicalCores, margin, packed, layout, totalQ,
        totalK;
    uint32_t pageSize, usedFlags, valueDim, hasSoftcap, mask, left, right, m, n, cores, headGroups;
    uint32_t queryCapacity, kvCapacity, algorithm, reserved[3];
};
static_assert(sizeof(MlaRecordHeader) == 32 * sizeof(uint32_t));
struct MlaNativePartition {
    uint32_t mask, left, right, usedCoreNum;
    uint32_t bEnd[26], mEnd[26], nEnd[26];
    uint32_t fdHeads, maxSplits, vectors, fdRows;
    uint32_t fdBatch[26], fdM[26], fdSplits[26], coreSplitStart[26], fdParts[26], fdLastRows[26];
    uint32_t vectorHeadEnd[52], vectorPartEnd[52];
};
static_assert(sizeof(MlaNativePartition) == 346 * sizeof(uint32_t));
// Offsets name the source fields; consumers never infer a development format.
enum MlaNativeWord : uint32_t {
    kMlaNativeUsedCoreNum = offsetof(MlaNativePartition, usedCoreNum) / sizeof(uint32_t),
    kMlaNativeBEnd = offsetof(MlaNativePartition, bEnd) / sizeof(uint32_t),
    kMlaNativeMEnd = offsetof(MlaNativePartition, mEnd) / sizeof(uint32_t),
    kMlaNativeNEnd = offsetof(MlaNativePartition, nEnd) / sizeof(uint32_t),
    kMlaNativeFdHeads = offsetof(MlaNativePartition, fdHeads) / sizeof(uint32_t),
    kMlaNativeMaxSplits = offsetof(MlaNativePartition, maxSplits) / sizeof(uint32_t),
    kMlaNativeVectors = offsetof(MlaNativePartition, vectors) / sizeof(uint32_t),
    kMlaNativeFdRows = offsetof(MlaNativePartition, fdRows) / sizeof(uint32_t),
    kMlaNativeFdBatch = offsetof(MlaNativePartition, fdBatch) / sizeof(uint32_t),
    kMlaNativeFdM = offsetof(MlaNativePartition, fdM) / sizeof(uint32_t),
    kMlaNativeFdSplits = offsetof(MlaNativePartition, fdSplits) / sizeof(uint32_t),
    kMlaNativeCoreSplitStart = offsetof(MlaNativePartition, coreSplitStart) / sizeof(uint32_t),
    kMlaNativeFdParts = offsetof(MlaNativePartition, fdParts) / sizeof(uint32_t),
    kMlaNativeFdLastRows = offsetof(MlaNativePartition, fdLastRows) / sizeof(uint32_t),
    kMlaNativeVectorHeadEnd = offsetof(MlaNativePartition, vectorHeadEnd) / sizeof(uint32_t),
    kMlaNativeVectorPartEnd = offsetof(MlaNativePartition, vectorPartEnd) / sizeof(uint32_t),
};

constexpr uint32_t kMlaRecordHeaderWords = sizeof(MlaRecordHeader) / 4;
constexpr uint32_t kMlaSequenceWords = 8;
constexpr uint32_t kMlaNativePartitionWords = sizeof(MlaNativePartition) / 4;
MLA_METADATA_INLINE bool MlaTnd(const MlaContext& c)
{
    return c.layout == 1 || c.layout == 2 || c.layout == 4;
}
MLA_METADATA_INLINE bool MlaEmptyKv(const MlaContext& c)
{
    return !c.sk || (c.layout == 1 && !c.totalK);
}
MLA_METADATA_INLINE bool MlaNativeCompressed(const MlaContext& c)
{
    return c.valueDim == 512 && !c.margin && (c.packed || c.heads == c.kvHeads);
}
MLA_METADATA_INLINE uint32_t MlaSequenceWord(uint32_t b)
{
    return kMlaRecordHeaderWords + kMlaSequenceWords * b;
}
MLA_METADATA_INLINE uint32_t MlaTail(const MlaContext& c)
{
    return MlaSequenceWord(c.batch);
}
MLA_METADATA_INLINE bool MlaRecordLayout(const MlaContext& c, MlaLayout& l, uint32_t tileM = 0, uint32_t tileN = 0)
{
    if (!c.kvHeads || !c.heads || c.heads % c.kvHeads || c.packed > 1 || (c.packed && c.heads == c.kvHeads) ||
        !c.physicalCores || c.physicalCores > 26 || c.margin >= c.physicalCores || c.layout > 4 ||
        (c.valueDim != 128 && c.valueDim != 512) || (c.valueDim == 512 && c.kvHeads != 1) ||
        ((c.layout >= 3) != bool(c.pageSize)))
        return false;
    MlaLayout r{};
    const uint32_t available = c.physicalCores - c.margin;
    r.cores = available;
    if (!r.cores || r.cores > available)
        return false;
    r.g = c.packed ? c.heads / c.kvHeads : 1;
    r.headGroups = c.heads / r.g;
    const bool tnd = MlaTnd(c);
    if (c.hasSoftcap > 1 || c.splits > 256 || c.usedFlags > 3 || c.window.kind > 2 || c.window.left < -1 ||
        c.window.right < -1 || c.window.left > INT32_MAX || c.window.right > INT32_MAX || c.sq > kMlaMaxQueryLength ||
        c.sk > kMlaMaxKvLength || c.heads > kMlaMaxQueryHeads || uint64_t(c.sq) * c.heads > kMlaMaxDenseRows ||
        (tnd ? uint64_t(c.totalQ) * c.heads > kMlaMaxBatchRows
             : uint64_t(c.batch) * c.sq > kMlaMaxBatchRows / c.heads) ||
        c.totalQ > INT32_MAX || c.totalK > INT32_MAX || (!tnd && c.totalQ) || (c.layout != 1 && c.totalK) ||
        (c.pageSize && (c.pageSize < 16 || c.pageSize > 1024 || c.pageSize % 16 || c.sk % c.pageSize)) ||
        uint64_t(c.batch) * (8 + 2ULL * r.headGroups) > UINT32_MAX - 380)
        return false;
    // Fixed CANN CalcInnerSize/CalcMBaseSize. PA alignment is logical; the
    // physical cache stride is independent of these source numerical tiles.
    uint32_t prefillN = !tnd && c.sq <= 16 ? (r.g < 5 ? 8192 : r.g < 10 ? 4096 : 2048) : 512;
    if (c.window.kind && prefillN > 1024)
        prefillN = 1024;
    if (c.pageSize && prefillN > c.pageSize && prefillN % c.pageSize)
        prefillN = prefillN / c.pageSize * c.pageSize;
    if (prefillN > c.sk)
        prefillN = c.sk;
    if (!prefillN)
        prefillN = 32;
    const uint32_t aligned = (prefillN + 31) / 32 * 32;
    const uint32_t prefillM = aligned <= 512    ? 512
                              : aligned <= 1024 ? 256
                              : aligned <= 2048 ? 128
                              : aligned <= 4096 ? 64
                                                : 32;
    r.m = tileM ? tileM : c.valueDim == 512 ? 256 : prefillM;
    r.n = tileN ? tileN : c.valueDim == 512 ? (c.sk < 512 ? c.sk : 512) : prefillN;
    if (c.valueDim == 512 && c.pageSize && r.n > c.pageSize && r.n % c.pageSize)
        r.n = r.n / c.pageSize * c.pageSize;
    if (!r.n)
        r.n = 32;
    if ((c.valueDim == 512 && (r.m < 128 || r.m > 256 || r.n > 512)) ||
        (c.valueDim == 128 && (r.m != prefillM || r.n != prefillN)))
        return false;
    const uint64_t publicCapacity = c.splits ? c.splits : 26;
    const uint64_t strideRows =
        (tnd ? uint64_t(c.totalQ ? c.totalQ : 1) : uint64_t(c.batch ? c.batch : 1) * (c.sq ? c.sq : 1)) * c.heads;
    if (strideRows > uint64_t(INT64_MAX) / 512 / publicCapacity)
        return false;
    r.words = MlaTail(c) + kMlaNativePartitionWords + 2 + 2 * c.batch * r.headGroups;
    r.capacity = c.splits ? c.splits : available;
    r.groups = uint64_t(c.batch) * r.headGroups * ((uint64_t(c.sq) * r.g + r.m - 1) / r.m);
    if (MlaTnd(c)) {
        const uint64_t compact = (uint64_t(c.totalQ) * r.g / r.m + c.batch) * r.headGroups;
        if (compact < r.groups)
            r.groups = compact;
        if (!c.totalQ)
            r.groups = 0;
    }
    if (r.groups && !MlaEmptyKv(c)) {
        if (MlaNativeCompressed(c) && c.splits <= 1) {
            const uint64_t rows = ((uint64_t(c.sq) * r.g < r.m ? uint64_t(c.sq) * r.g : r.m) + 15) / 16 * 16;
            r.scratch = 2ULL * available * (6 * rows * ((r.n + 31) / 32 * 32) + 8 * rows * 512 + 8 * r.m);
        } else {
            r.scratch = uint64_t(available) * (12ULL * r.m * ((r.n + 31) / 32 * 32) + 16ULL * r.m * c.valueDim +
                                               (c.valueDim == 512 ? 16ULL * r.m : 0));
        }
        r.outOffset = (r.scratch + 31) / 32 * 32;
        const uint64_t rows =
            MlaNativeCompressed(c) && !c.splits ? 2ULL * available * r.m : r.groups * r.capacity * r.m;
        if (rows > (uint64_t(INT64_MAX) - r.outOffset) / ((c.valueDim + 16) * 4))
            return false;
        r.sumOffset = r.outOffset + rows * c.valueDim * 4;
        r.maxOffset = r.sumOffset + rows * 8 * 4;
        const bool direct =
            c.splits == 1 && (MlaNativeCompressed(c) || (!c.margin && (c.packed || c.heads == c.kvHeads)));
        r.bytes = direct ? r.scratch : r.maxOffset + rows * 8 * 4;
    }
    l = r;
    return true;
}
MLA_METADATA_INLINE MlaRecordHeader MlaHeader(const MlaContext& c, const MlaLayout& l, uint32_t qCapacity,
                                              uint32_t kCapacity)
{
    return {l.words,
            c.dtype,
            0,
            c.batch,
            c.sq,
            c.sk,
            c.heads,
            c.kvHeads,
            c.splits,
            c.physicalCores,
            c.margin,
            c.packed,
            c.layout,
            c.totalQ,
            c.totalK,
            c.pageSize,
            c.usedFlags,
            c.valueDim,
            c.hasSoftcap,
            c.window.kind,
            uint32_t(c.window.left),
            uint32_t(c.window.right),
            l.m,
            l.n,
            l.cores,
            l.headGroups,
            qCapacity,
            kCapacity,
            0,
            {0, 0, 0}};
}
struct MlaRecordArgs {
    MlaContext context;
    uint64_t cuQuery = 0, cuKv = 0, usedQuery = 0, usedKv = 0, output = 0;
    uint32_t words = 0, inferTotalQ = 0;
    uint64_t checked = 0;
    MlaPagedCacheInputs cache{};
    bool generate = true;
};
bool BuildNonquantMetadata(const MlaRecordArgs&);
bool PrepareInt8Metadata(const MlaRecordArgs&);
} // namespace flash_attn::mla
#undef MLA_METADATA_INLINE
