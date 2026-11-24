/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Allocation snapshots and CANN SplitCore are shared by every MLA layout.
#include "mla_metadata.h"
#include "mla_split.h"
#include "mla_metadata_consumer.h"
#include <algorithm>
#include <cstring>
#include <new>
#include <vector>

namespace flash_attn::mla {
namespace {
// The producer already owns the parsed allocation/used lengths and the source
// SplitCore result. Rebind those values without serializing and decoding them
// a second time. Noncanonical contexts retain the external-record reader.
bool TryGeneratedSchedule(const MlaRecordArgs& a, const MlaContext& c, const MlaLayout& l,
                          const MlaUsedLengths& lengths, const MlaNativeSplitKvSchedule& native,
                          const MlaWindowConfig& window, uint32_t queryCapacity, uint64_t groups,
                          MlaAutoSplitBsndSchedule& output)
{
    MlaWindow normalized{};
    if (a.inferTotalQ || NormalizeWindow(window, normalized) != MlaCausalStatus::Success ||
        uint32_t(normalized.kind) != c.window.kind || normalized.left != c.window.left ||
        normalized.right != c.window.right || l.bytes > c.workspaceBytes || groups > l.groups)
        return false;
    const auto& fd = native.fd;
    const uint32_t launchCores = MlaStaticLaunchCores(c);
    if (!launchCores || !native.usedCoreNum || native.usedCoreNum > launchCores || native.usedCoreNum > l.cores ||
        fd.numOfFdHead > l.cores || fd.maxS2SplitNum > l.cores || fd.usedVecNumOfFd > 2 * native.usedCoreNum ||
        fd.usedVecNumOfFd > 2 * launchCores)
        return false;
    if (c.splits == 1) {
        if (fd.numOfFdHead || fd.maxS2SplitNum || fd.usedVecNumOfFd)
            return false;
        for (uint32_t core = 0; core < native.usedCoreNum; ++core)
            if (native.s2End[core])
                return false;
    }
    MlaAutoSplitBsndSchedule s{};
    s.aicNum = l.cores;
    s.usedCoreNum = native.usedCoreNum;
    s.fdHeads = fd.numOfFdHead;
    s.maxSplits = fd.maxS2SplitNum;
    s.vectors = fd.usedVecNumOfFd;
    s.groups = groups;
    uint32_t maximum = 0;
    for (uint32_t head = 0; head < s.fdHeads; ++head) {
        const uint32_t count = fd.s2SplitNumOfFdHead[head];
        if (count < 2)
            return false;
        maximum = count > maximum ? count : maximum;
        s.partialPrefix[head + 1] = s.partialPrefix[head] + count;
    }
    if (maximum != s.maxSplits || s.partialPrefix[s.fdHeads] > uint64_t(2) * l.cores ||
        (s.fdHeads == 0 ? s.vectors != 0 : s.vectors == 0))
        return false;

    auto& used = s.used;
    const uint64_t totalQ = MlaTnd(c) ? c.totalQ : uint64_t(c.batch) * c.sq;
    const uint64_t totalK = c.layout == 1 ? c.totalK : uint64_t(c.batch) * c.sk;
    used.query = {uint32_t(MlaTnd(c)), {uint32_t(totalQ), uint32_t(totalQ >> 32)}, c.sq};
    used.kv = {uint32_t(c.layout == 1), {uint32_t(totalK), uint32_t(totalK >> 32)}, c.sk};
    used.usedFlags = c.usedFlags;
    auto& t = used.tiling;
    t.batch = c.batch * l.headGroups;
    t.sq = c.sq;
    t.sk = c.sk ? c.sk : 1;
    t.heads = l.g;
    t.mBaseSize = l.m;
    t.s2BaseSize = l.n;
    t.usedCoreNum = s.usedCoreNum;
    t.blockDim = l.cores;
    t.cvRatio = 2;
    const uint64_t rows = ((uint64_t(queryCapacity) * l.g < l.m ? uint64_t(queryCapacity) * l.g : l.m) + 15) / 16 * 16;
    t.mm1ResSize = rows * ((l.n + 31) / 32 * 32);
    t.mm2ResSize = rows * 512;
    for (uint32_t core = 0; core < 26; ++core) {
        s.bEnd[core] = t.bN2End[core] = native.bN2End[core];
        s.mEnd[core] = t.gS1End[core] = native.gS1End[core];
        s.nEnd[core] = t.s2End[core] = native.s2End[core];
    }
    s.outOffset = l.outOffset;
    s.sumOffset = l.sumOffset;
    s.maxOffset = l.maxOffset;
    s.bytes = l.bytes;

    // Run after the record write, preserving the reader's ordering even if a
    // caller's page table aliases the output storage. Never skip the full
    // descriptor or active-page checks on a newly generated record.
    if (c.pageSize) {
        if (CheckPagedCacheDescriptor(c, a.cache, c.kvHeads, c.valueDim == 512 ? 64 : 192, c.valueDim) !=
            MlaMetadataReadStatus::Success)
            return false;
        const auto* pages = reinterpret_cast<const int32_t*>(a.cache.table.address);
        for (uint32_t b = 0; b < c.batch; ++b) {
            const uint32_t length = lengths.sequences[b].kvLength;
            if (length > c.sk)
                return false;
            for (uint32_t i = 0; i < (length + c.pageSize - 1) / c.pageSize; ++i) {
                const int32_t id = pages[uint64_t(b) * a.cache.layout.tableRowStride + i];
                if (id < 0 || uint64_t(id) >= a.cache.layout.physicalPages)
                    return false;
            }
        }
    }
    output = s;
    return true;
}
} // namespace

// Private overload for the AICPU entry, included in that translation unit.
// ready is published only after generation and every additional check pass;
// failures leave the checked schedule untouched for the original reader.
bool BuildMlaMetadata(const MlaRecordArgs& a, MlaAutoSplitBsndSchedule* generated, bool& ready)
{
    ready = false;
    auto c = a.context;
    MlaLayout l{};
    if (!MlaRecordLayout(c, l) || !a.output || a.output % 4 || a.words != l.words ||
        a.words > (UINT64_MAX - a.output) / 4)
        return false;
    const uint64_t addresses[] = {a.cuQuery, a.cuKv, a.usedQuery, a.usedKv};
    for (uint32_t axis = 0; axis < 4; ++axis) {
        const uint64_t address = addresses[axis], count = uint64_t(c.batch) + (axis < 2);
        if (address && count && address < a.output + uint64_t(a.words) * 4 &&
            (address >= a.output || count > (a.output - address) / 4))
            return false;
    }
    auto* out = reinterpret_cast<uint32_t*>(a.output);
    std::fill(out, out + a.words, 0);
    out[0] = a.words;
    out[1] = c.dtype;
    out[2] = 1;
    for (uint32_t axis = 0; axis < 4; ++axis) {
        const uint64_t address = addresses[axis], count = uint64_t(c.batch) + (axis < 2);
        const bool required = axis == 0 ? MlaTnd(c) : axis == 1 ? c.layout == 1 : c.usedFlags & (1U << (axis - 2));
        if ((required && count && !address) || (!required && address) || address % 4 ||
            (address && count > (UINT64_MAX - address) / 4))
            return true;
    }
    if ((c.dtype != uint32_t(MlaMetadataDtype::Fp16) && c.dtype != uint32_t(MlaMetadataDtype::Bf16)) ||
        a.inferTotalQ > 1 || (a.inferTotalQ && c.layout != 2 && c.layout != 4))
        return true;
    if (a.inferTotalQ) {
        c.totalQ = reinterpret_cast<const uint32_t*>(a.cuQuery)[c.batch];
        if (!MlaRecordLayout(c, l))
            return true;
    }
    try {
        const MlaUsedShape shape{c.batch,
                                 c.heads,
                                 {MlaTnd(c) ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd,
                                  MlaTnd(c) ? int64_t(c.totalQ) : int64_t(c.batch) * c.sq, c.sq},
                                 {c.layout == 1 ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd,
                                  c.layout == 1 ? int64_t(c.totalK) : int64_t(c.batch) * c.sk, c.sk}};
        MlaUsedInputs inputs{};
        MlaLengthArray* arrays[] = {&inputs.cuQuery, &inputs.cuKv, &inputs.usedQuery, &inputs.usedKv};
        for (uint32_t i = 0; i < 4; ++i)
            *arrays[i] = {reinterpret_cast<const int32_t*>(addresses[i]),
                          addresses[i] ? uint64_t(c.batch) + (i < 2) : 0};
        MlaUsedLengths lengths;
        if (ParseUsedLengths(shape, inputs, lengths) != MlaScheduleStatus::Success)
            return true;
        MlaPlatformInfo physical{};
        physical.aicNum = c.physicalCores;
        physical.aivNum = 2 * c.physicalCores;
        physical.localMemory = {516096, 196608, 65536, 65536, 131072};
        for (uint32_t i = 0; i < c.physicalCores; ++i)
            physical.blockDims[i] = i + 1;
        uint32_t queryCapacity = MlaTnd(c) ? lengths.maxQueryCapacity : c.sq;
        uint32_t kvCapacity = c.layout == 1 ? lengths.maxKvCapacity : c.sk;
        if (c.pageSize)
            kvCapacity = (lengths.maxKvLength + c.pageSize - 1) / c.pageSize * c.pageSize;
        if (MlaNativeCompressed(c)) {
            MlaSchedule geometry{};
            if (PrepareMlaGeometry(shape, physical, inputs, lengths, c.pageSize, geometry) !=
                MlaScheduleStatus::Success)
                return true;
            if (!MlaRecordLayout(c, l, geometry.mBaseSize, geometry.s2BaseSize))
                return true;
        }
        const MlaWindowConfig window{c.sq, c.sk, c.window.kind == 1 ? -1 : c.window.left,
                                     c.window.kind == 1 ? -1 : c.window.right, c.window.kind == 1};
        MlaNativeSplitKvSchedule native{};
        const bool wholeM = c.splits == 1;
        if (BuildGqaSchedule({shape, c.kvHeads, c.valueDim, l.m, l.n, c.pageSize, !wholeM}, lengths, window,
                             {c.packed, c.margin}, physical, native) != MlaScheduleStatus::Success)
            return true;
        // All fallible allocations have completed. The destination was
        // cleared above; publish success after writing the complete record.
        auto header = MlaHeader(c, l, queryCapacity, kvCapacity);
        header.status = 1;
        std::memcpy(out, &header, sizeof(header));
        auto* tail = out + MlaTail(c);
        auto* prefix = tail + kMlaNativePartitionWords;
        uint64_t groups = 0;
        for (uint32_t b = 0; b < c.batch; ++b) {
            const auto& s = lengths.sequences[b];
            const uint32_t span[] = {uint32_t(s.queryStart), uint32_t(s.queryStart >> 32),
                                     uint32_t(s.kvStart),    uint32_t(s.kvStart >> 32),
                                     s.queryCapacity,        s.kvCapacity,
                                     s.queryLength,          s.kvLength};
            std::copy(span, span + 8, out + MlaSequenceWord(b));
            const uint64_t count = (uint64_t(s.queryLength) * l.g + l.m - 1) / l.m;
            for (uint32_t h = 0; h < l.headGroups; ++h) {
                groups += count;
                prefix += 2;
                prefix[0] = uint32_t(groups);
                prefix[1] = uint32_t(groups >> 32);
            }
        }
        MlaNativePartition partition{};
        partition.mask = c.window.kind;
        partition.left = uint32_t(c.window.left);
        partition.right = uint32_t(c.window.right);
        partition.usedCoreNum = native.usedCoreNum;
        std::copy(native.bN2End.begin(), native.bN2End.end(), partition.bEnd);
        std::copy(native.gS1End.begin(), native.gS1End.end(), partition.mEnd);
        std::copy(native.s2End.begin(), native.s2End.end(), partition.nEnd);
        const auto& fd = native.fd;
        partition.fdHeads = fd.numOfFdHead;
        partition.maxSplits = fd.maxS2SplitNum;
        partition.vectors = fd.usedVecNumOfFd;
        partition.fdRows = 8;
#define MLA_COPY_FIELD(dst, src) std::copy(fd.src.begin(), fd.src.end(), partition.dst)
        MLA_COPY_FIELD(fdBatch, bN2IdxOfFdHead);
        MLA_COPY_FIELD(fdM, gS1IdxOfFdHead);
        MLA_COPY_FIELD(fdSplits, s2SplitNumOfFdHead);
        MLA_COPY_FIELD(coreSplitStart, s2SplitStartIdxOfCore);
        MLA_COPY_FIELD(fdParts, gS1SplitNumOfFdHead);
        MLA_COPY_FIELD(fdLastRows, gS1LastPartSizeOfFdHead);
        MLA_COPY_FIELD(vectorHeadEnd, gS1IdxEndOfFdHead);
        MLA_COPY_FIELD(vectorPartEnd, gS1IdxEndOfFdHeadSplit);
#undef MLA_COPY_FIELD
        std::memcpy(tail, &partition, sizeof(partition));
        out[2] = 0;
        if (generated)
            ready = TryGeneratedSchedule(a, c, l, lengths, native, window, queryCapacity, groups, *generated);
    } catch (const std::bad_alloc&) {
    }
    return true;
}
bool BuildMlaMetadata(const MlaRecordArgs& a)
{
    bool ready = false;
    return BuildMlaMetadata(a, nullptr, ready);
}
} // namespace flash_attn::mla
