/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Derived from ca18aee610729ca5270a44e2d84c751bc687fbf0:
// incre_flash_attention_tiling.cpp::SplitBN_V0, SplitBNS, SplitBalancedFd,
// SplitBalanced, SplitFDMLa, CalcInnerSize and FillTilingMla; and
// incre_flash_attention_tiling_check.cpp::IsFlashDecode/CheckCoreOkFlag.
// Specialization: BSND/TND INT8/BF16, Hkv=1, ND PA. No framework dependencies.
// Schedule once on AICPU; numerical cores consume invocation-local checked tiling.
#pragma once
#include "mla_int8_params.h"
#if defined(__NPU_ARCH__) && defined(MLA_INT8_DEVICE_SCHEDULE)
#define MLA_INT8_INLINE __aicore__ inline
#else
#define MLA_INT8_INLINE inline
#endif

namespace flash_attn::mla::Int8 {
struct Config {
    uint32_t batch, query, heads, capacity, cores, splits, causal, reserved;
};
constexpr uint32_t kFixedWords = 32 + sizeof(Int8TilingParams) / 4;

MLA_INT8_INLINE uint32_t Min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
MLA_INT8_INLINE uint32_t Max(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}
MLA_INT8_INLINE uint32_t Div(uint32_t a, uint32_t b)
{
    return a / b + (a % b != 0);
}
MLA_INT8_INLINE bool Valid(const Config& c)
{
    return c.query <= 32 && c.heads >= 1 && c.heads <= 128 && (c.heads & (c.heads - 1)) == 0 &&
           c.capacity <= INT32_MAX && c.capacity % 128 == 0 && c.cores > 0 && c.cores < MAX_AIC_CORE_NUM &&
           c.splits <= 1 && c.causal <= 1 && (c.query <= 1 || c.causal) && c.reserved == 0 &&
           c.batch <= (UINT32_MAX - kFixedWords) / 8 && uint64_t(c.batch) * Max(c.query, 1) * c.heads <= INT32_MAX &&
           uint64_t(c.batch) * c.heads * 512 <= UINT32_MAX;
}

// Bounds are checked before reading lengths, including empty Q. False leaves a
// zero descriptor. Numerical scale, LSE and addresses are invocation state.
template <bool Tnd, class Length, class Query>
MLA_INT8_INLINE bool ScheduleImpl(const Config& c, const Length& length, const Query& query, uint32_t totalQuery,
                                  Int8TilingParams& p, uint32_t& mode)
{
    p = {};
    mode = 0;
    if (!Valid(c))
        return false;
    uint32_t longest = 0, shortest = c.capacity;
    bool same = true;
    int64_t first = c.batch ? length(0) : 0;
    for (uint32_t b = 0; b < c.batch; ++b) {
        int64_t n = length(b);
        if (n < 0 || uint64_t(n) > c.capacity)
            return false;
        longest = Max(longest, uint32_t(n));
        shortest = Min(shortest, uint32_t(n));
        same = same && n == first;
    }
    if (!c.batch || !c.query)
        return true;
    auto& base = p.baseParams;
    base.batchSize = c.batch;
    base.seqSize = c.capacity;
    base.qSeqSize = c.query;
    base.blockSize = 128;
    base.maxBlockNumPerBatch = c.capacity / 128;
    base.nNumOfQInOneGroup = c.heads;
    base.actualLenDims = c.batch;
    base.attenMaskFlag = c.query > 1;
    base.sparseMode = c.query > 1 ? 3 : 0;
    base.tSeqSize = Tnd ? totalQuery : 1;
    base.actualLenQDims = Tnd ? c.batch : 0;
    base.outputLayout = Tnd ? 3 : 0;
    auto& single = p.increFlashAttentionSingleCoreParams;
    auto& split = p.splitKVParams;
    auto& core = p.increFlashAttentionCoreParams;
    auto& fd = p.tndSplitCoreParams;
    const bool fast =
        c.splits == 0 &&
        (((c.batch == 1 || c.batch == 2 || c.batch == 4) && c.query == 4 && c.heads == 8 && shortest >= 4096 &&
          longest <= 5120) ||
         (c.batch == 4 && c.query >= 2 && c.query <= 4 && c.heads == 64 && shortest >= 4096 && longest <= 5120) ||
         (c.batch == 1 && c.query == 2 && c.heads == 128 && shortest >= 37000 && longest <= 65536));
    const bool balance = Tnd || (c.splits == 0 && (!same || fast));
    fd.FdBalanceFlag = fast;
    single.groupSplitSize = c.heads;
    single.s1SplitSize = balance ? 128 / c.heads : Min(128 / c.heads, c.query);
    uint32_t inner = Min(512, Max(c.capacity, 128));
    uint32_t parts = 1;
    if (balance) {
        mode = Tnd ? 5 : 4;
        uint64_t total = 0;
        uint32_t last = 0;
        for (uint32_t b = 0; b < c.batch; ++b) {
            uint32_t n = longest ? Div(uint32_t(length(b)), inner) : 1;
            uint32_t outer = longest ? Div(query(b), single.s1SplitSize) : 1;
            total += uint64_t(outer) * n;
            if (n && outer)
                last = b;
        }
        uint64_t average = (total + c.cores - 1) / c.cores;
        if (!average)
            average = 1;
        uint64_t accumulated = 0;
        uint32_t used = 0, tasks = 0;
        bool pending = false;
        for (uint32_t b = 0; b < c.batch; ++b) {
            uint32_t n = longest ? Div(uint32_t(length(b)), inner) : 1;
            uint32_t outer = longest ? Div(query(b), single.s1SplitSize) : 1;
            for (uint32_t q = 0; q < outer; ++q) {
                uint32_t count = 1;
                for (uint32_t k = 0; k < n; ++k) {
                    ++accumulated;
                    if (accumulated >= uint64_t(used + 1) * average && used + 2 <= c.cores &&
                        (!Tnd || !c.splits || k + 1 == n)) {
                        core.coreBEnd[used] = b;
                        core.coreS1OuterEnd[used] = q;
                        core.coreS2End[used++] = k;
                        fd.balanceFDCoreStartKVSplitNum[used] = k + 1 < n ? count++ : 0;
                        pending = false;
                    } else {
                        pending = true;
                    }
                }
                parts = Max(parts, count);
                if (count > 1) {
                    if (tasks >= c.cores)
                        return false;
                    fd.balanceFDCoreBArr[tasks] = b;
                    fd.balanceFDCoreS1Arr[tasks] = q;
                    fd.balanceFDCoreKVSplitArr[tasks] = count;
                    if (fast) {
                        uint32_t m = Min(single.s1SplitSize, query(b) - q * single.s1SplitSize) * c.heads;
                        fd.gS1SplitNumOfFdHeadMla[tasks] = Div(m, 8);
                        fd.gS1LastPartSizeOfFdHead[tasks] = (m - 1) % 8 + 1;
                    }
                    ++tasks;
                }
            }
        }
        if (pending) {
            core.coreBEnd[used] = last;
            core.coreS1OuterEnd[used] = longest ? Div(query(last), single.s1SplitSize) - 1 : 0;
            core.coreS2End[used++] = longest ? Div(uint32_t(length(last)), inner) - 1 : 0;
        } else if (last + 1 < c.batch) {
            core.coreBEnd[used++] = last + 1;
        }
        if constexpr (Tnd) {
            if (!used) {
                core.coreBEnd[0] = c.batch - 1;
                used = 1;
            }
        }
        single.usedCoreNum = used;
        fd.tndFDCoreArrLen = tasks;
        if (tasks)
            mode |= 2;
        if (fast && tasks) {
            uint32_t totalLoad = 0, totalRows = 0;
            for (uint32_t t = 0; t < tasks; ++t) {
                totalRows += fd.gS1SplitNumOfFdHeadMla[t];
                totalLoad += fd.balanceFDCoreKVSplitArr[t] * fd.gS1SplitNumOfFdHeadMla[t];
            }
            uint32_t vectors = Min(totalRows, used * 2), v = 0, load = 0, prevT = 0, prevQ = 0;
            for (uint32_t t = 0; t < tasks; ++t) {
                for (uint32_t q = 0; q < fd.gS1SplitNumOfFdHeadMla[t]; ++q) {
                    uint32_t weight = fd.balanceFDCoreKVSplitArr[t];
                    // Integer form of source's double load threshold: all
                    // operands are small integral FD weights.
                    if (int64_t(weight) * (vectors - v) > 2 * (int64_t(totalLoad) - int64_t(load) * (vectors - v)) &&
                        v + 1 < vectors) {
                        fd.gS1IdxEndOfFdHead[v] = prevT;
                        fd.gS1IdxEndOfFdHeadSplit[v++] = prevQ;
                        totalLoad -= load;
                        load = 0;
                    }
                    load += weight;
                    prevT = t;
                    prevQ = q;
                }
            }
            fd.gS1IdxEndOfFdHead[v] = prevT;
            fd.gS1IdxEndOfFdHeadSplit[v] = prevQ;
            fd.usedVecNumOfFd = v + 1;
        }
    } else if (!c.splits && longest && c.query == 1 && c.batch * 2 <= c.cores && (c.heads == 1 || longest >= 2048)) {
        mode = 2;
        parts = c.cores / c.batch;
        while (parts > 1 && longest / parts < 512)
            --parts;
        single.usedCoreNum = c.batch * parts;
        uint32_t span = Div(Div(c.capacity, parts), 128) * 128;
        inner = span == 256 ? 128 : (span > 256 && span <= 512 ? 256 : 512);
        inner = Min(inner, Max(span, 128));
    } else {
        uint32_t tasks = c.batch * Div(c.query, single.s1SplitSize);
        uint32_t used = Div(tasks, Div(tasks, c.cores));
        uint32_t former = tasks % used;
        uint32_t block = Div(tasks, used);
        for (uint32_t i = 0; i < MAX_AIC_CORE_NUM; ++i)
            core.coreSidxEnd[i] =
                i >= used ? tasks : (i < former ? block * i : former * block + (tasks / used) * (i - former));
        single.usedCoreNum = used;
    }
    single.singleProcessSInnerSize = inner;
    split.s2 = mode & 2 ? parts : 0;
    split.sInnerLoopSize = Div(Div(longest, parts), 128) * 128;
    split.accumOutSize = balance ? c.cores * 2 * 128 * 512 : c.batch * c.heads * parts * 512;
    split.logSumExpSize = balance ? 2 * c.cores * 2 * 128 * 8 : 2 * c.batch * c.heads * parts * 8;
    uint32_t rows = Min(c.heads * c.query, 128);
    p.increFlashAttentionSingleCoreTensorSize.mmResUbSize = Div(inner, 32) * 32 * rows;
    p.increFlashAttentionSingleCoreTensorSize.bmm2ResUbSize = 512 * rows;
    return single.usedCoreNum > 0 && single.usedCoreNum <= c.cores;
}

struct FixedQuery {
    uint32_t length;
    MLA_INT8_INLINE uint32_t operator()(uint32_t) const
    {
        return length;
    }
};
template <class Prefix>
struct QuerySpan {
    const Prefix& prefix;
    MLA_INT8_INLINE uint32_t operator()(uint32_t b) const
    {
        return uint32_t(prefix(b + 1) - prefix(b));
    }
};

template <class Length>
MLA_INT8_INLINE bool Schedule(const Config& c, const Length& length, Int8TilingParams& p, uint32_t& mode)
{
    return ScheduleImpl<false>(c, length, FixedQuery{c.query}, 0, p, mode);
}

MLA_INT8_INLINE bool ValidTnd(const Config& c)
{
    return Valid(c) && c.batch <= (UINT32_MAX - kFixedWords) / 8;
}

template <class Length, class Prefix>
MLA_INT8_INLINE bool ScheduleTnd(const Config& c, const Length& length, const Prefix& prefix, int64_t totalQuery,
                                 Int8TilingParams& p, uint32_t& mode)
{
    p = {};
    mode = 0;
    if (!ValidTnd(c) || prefix(0) != 0)
        return false;
    int64_t total = prefix(c.batch);
    if (total < 0 || total > 1048576 || uint64_t(total) * c.heads > INT32_MAX ||
        (totalQuery >= 0 && total != totalQuery))
        return false;
    uint32_t maximum = 0;
    for (uint32_t b = 0; b < c.batch; ++b) {
        int64_t begin = prefix(b), end = prefix(b + 1);
        if (begin < 0 || end < begin || end > total || uint64_t(end - begin) > c.query)
            return false;
        maximum = Max(maximum, uint32_t(end - begin));
    }
    Config numerical = c;
    numerical.query = maximum;
    return ScheduleImpl<true>(numerical, length, QuerySpan<Prefix>{prefix}, uint32_t(total), p, mode);
}

// The common MLA header and allocation/used snapshot precede this dtype's
// source tiling payload. Quantized IFA has different numerical scheduling data.
struct alignas(64) CheckedSchedule {
    uint32_t status = 1, blockDim = 0, dtype = 3, mode = 0;
    alignas(64) Int8TilingParams schedule{};
};
} // namespace flash_attn::mla::Int8
