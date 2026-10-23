/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#include "mla_tiling.h"
#include "cann_split_core.h"
#include <algorithm>
#include <limits>
#include <new>
#include <numeric>
#include <utility>
#include "mla_metadata.h"

// Derived from CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/fused_infer_attention_score/op_host/arch22/fia_tiling_nonquant_mla.cpp:
// CalcInnerSize, CalcMBaseSize, CreateSplitInput, Split, CalcMmResSize,
// FillTilingWorkspaceParams, CalcNormalWorkspaceSize, CalcNumBlocks, CalcScheduleMode.
// Changes: remove gert/registration, restrict inputs to the validated dense MLA range,
// align source endpoints for FA3 num_splits=1, separate platform/launch
// data, check arithmetic/resources before source narrowing.

namespace flash_attn::mla {
namespace {

constexpr uint64_t Align(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

uint32_t CalcMBaseSize(uint32_t batch, uint32_t rows, uint32_t aicNum, bool sameKvLengths)
{
    uint32_t mBaseSize = 256;
    // Dense inputs have equal sequence lengths in every batch.
    if (batch % aicNum == 0 || !sameKvLengths) {
        return mBaseSize;
    }
    uint32_t lcm = (batch / std::gcd(batch, aicNum)) * aicNum;
    uint32_t splitOfS1G = lcm / batch;
    if (splitOfS1G == 0 || rows % splitOfS1G != 0) {
        return mBaseSize;
    }
    uint32_t candidate = rows / splitOfS1G;
    if (candidate <= mBaseSize && candidate >= 128) {
        mBaseSize = candidate;
    }
    return mBaseSize;
}

MlaLocalMemory RequiredLocalMemory(uint32_t cvRatio)
{
    // fia_block_cube_nonquant_mla.h::InitBuffers: four Q/P and three K/V L1 slots.
    // fia_block_vec_nonquant_mla.h::InitBuffers: outputBuff2 doubles for C1V1.
    return {7 * 128 * 288 * 2, (cvRatio == 1 ? 174U : 170U) * 1024, 2 * 32768, 2 * 32768, 2 * 65536};
}

bool HasLocalMemory(const MlaLocalMemory& have, const MlaLocalMemory& need)
{
    return have.l1Bytes >= need.l1Bytes && have.ubBytes >= need.ubBytes && have.l0aBytes >= need.l0aBytes &&
           have.l0bBytes >= need.l0bBytes && have.l0cBytes >= need.l0cBytes;
}

template <size_t N>
void CopyField(std::array<uint32_t, N>& dst, const std::vector<uint32_t>& src)
{
    std::copy(src.begin(), src.end(), dst.begin());
}

bool ValidSplit(const detail::SplitResult& split, const detail::BaseInfo& base, uint32_t mBlocks, uint32_t aicNum)
{
    if (split.usedCoreNum == 0 || split.usedCoreNum > aicNum) {
        return false;
    }
    const uint32_t batch = base.bSize;
    uint64_t previous = 0;
    for (uint32_t i = 0; i < split.usedCoreNum; ++i) {
        // num_splits=1: every boundary must be between complete M blocks.
        if (split.bN2End[i] > batch || split.gS1End[i] >= mBlocks || split.s2End[i] != 0) {
            return false;
        }
        uint64_t end = uint64_t(split.bN2End[i]) * mBlocks + split.gS1End[i];
        if (end <= previous || end > uint64_t(batch) * mBlocks) {
            return false;
        }
        previous = end;
    }
    // Source CalcSplitPlan stops when unassignedCost reaches zero, so a zero-KV
    // suffix can be outside its final endpoint. InitOutputSingleCore in the
    // source Kernel initializes empty outputs independently of these partitions.
    // Preserve source endpoints; require every omitted sample to have zero KV.
    if (previous < uint64_t(batch) * mBlocks && previous % mBlocks != 0) {
        return false;
    }
    for (uint32_t b = previous / mBlocks; b < batch; ++b) {
        if (detail::GetS2SeqSize(b, base) != 0) {
            return false;
        }
    }
    return true;
}

void SplitWithoutFlashDecode(uint32_t cores, const detail::BaseInfo& base, const detail::SplitParam& param,
                             uint32_t mBlocks, detail::SplitResult& split)
{
    detail::SplitCore(cores, base, param, split);
    if (split.numOfFdHead == 0) {
        return;
    }
    // FA3 num_splits=1 requires complete M blocks on each core. Round each
    // source endpoint up to the end of its M block and coalesce empty ranges.
    // Rebuild the partition and its costs; never just clear an FD plan's flags.
    // Filtering out all source FD candidates can otherwise discard balanced
    // partitions even when there are enough M blocks to occupy every core.
    detail::SplitContext context(base, param);
    detail::CalcSplitInfo(context);
    // Q uses full capacity, so every batch has the same M block count. KV
    // costs differ by batch; zero-length batches contribute zero. Output
    // initialization is independent of the compute endpoints (see ValidSplit).
    std::vector<int64_t> batchPrefix, firstCost;
    int64_t uniformFirst = 0, uniformBatch = 0;
    if (base.actualSeqS2Size.empty()) {
        detail::BatchCache batch{};
        detail::CalcBatchCache(0, context, batch);
        detail::S1GCache first{}, last{};
        detail::CalcS1GCache(0, context, batch, first);
        detail::CalcS1GCache(mBlocks - 1, context, batch, last);
        uniformFirst = first.s1GCost;
        uniformBatch = int64_t(mBlocks - 1) * first.s1GCost + last.s1GCost;
    } else {
        batchPrefix.resize(uint64_t(base.bSize) + 1, 0);
        firstCost.resize(base.bSize, 0);
        for (uint32_t b = 0; b < base.bSize; ++b) {
            detail::BatchCache batch{};
            detail::CalcBatchCache(b, context, batch);
            detail::S1GCache first{}, last{};
            detail::CalcS1GCache(0, context, batch, first);
            detail::CalcS1GCache(mBlocks - 1, context, batch, last);
            firstCost[b] = first.s1GCost;
            batchPrefix[b + 1] = batchPrefix[b] + int64_t(mBlocks - 1) * first.s1GCost + last.s1GCost;
        }
    }
    const auto costBefore = [&](uint64_t end) {
        const uint64_t b = end / mBlocks;
        return batchPrefix.empty() ? int64_t(b) * uniformBatch + int64_t(end % mBlocks) * uniformFirst
                                   : batchPrefix[b] + (b < base.bSize ? int64_t(end % mBlocks) * firstCost[b] : 0);
    };
    detail::SplitResult whole{cores, split.vecCubeRatio};
    uint64_t previous = 0;
    for (uint32_t i = 0; i < split.usedCoreNum; ++i) {
        const uint64_t end = uint64_t(split.bN2End[i]) * mBlocks + split.gS1End[i] + (split.s2End[i] != 0);
        if (end > previous) {
            const uint32_t core = whole.usedCoreNum++;
            whole.bN2End[core] = end / mBlocks;
            whole.gS1End[core] = end % mBlocks;
            whole.maxCost = std::max(whole.maxCost, costBefore(end) - costBefore(previous));
            previous = end;
        }
    }
    split = std::move(whole);
}

MlaScheduleStatus Calculate(const MlaDenseShape& shape, const MlaPlatformInfo& platform, const int32_t* kvLengths,
                            uint64_t lengthCount, MlaSchedule& output)
{
    constexpr auto u32Max = std::numeric_limits<uint32_t>::max();
    for (int64_t size : {shape.batch, shape.queryLength, shape.queryHeads, shape.kvLength}) {
        if (size < 0) {
            return MlaScheduleStatus::UnsupportedShape;
        }
        if (uint64_t(size) > u32Max) {
            return MlaScheduleStatus::FieldOverflow;
        }
    }
    // Multiplication is wide even for rejected input; never overflow the caller's int64 shape.
    uint64_t rows = uint64_t(shape.queryLength) * uint64_t(shape.queryHeads);
    if (shape.queryHeads == 0 || shape.kvLength > kMlaMaxKvLength) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    if (rows > kMlaMaxDenseRows || shape.queryHeads > kMlaMaxQueryHeads || shape.queryLength > kMlaMaxQueryLength ||
        uint64_t(shape.batch) * rows > kMlaMaxBatchRows) {
        return MlaScheduleStatus::FieldOverflow;
    }
    if (platform.aicNum == 0 || platform.aicNum > kMlaMaxAic || platform.aivNum == 0 ||
        platform.aivNum % platform.aicNum != 0) {
        return MlaScheduleStatus::InvalidPlatform;
    }
    uint32_t ratio = platform.aivNum / platform.aicNum;
    if (ratio != 1 && ratio != 2) {
        return MlaScheduleStatus::InvalidPlatform;
    }
    for (uint32_t c = 0; c < platform.aicNum; ++c) {
        if (platform.blockDims[c] == 0 || platform.blockDims[c] > platform.aicNum) {
            return MlaScheduleStatus::InvalidPlatform;
        }
    }
    auto requiredMemory = RequiredLocalMemory(ratio);
    if (!HasLocalMemory(platform.localMemory, requiredMemory)) {
        return MlaScheduleStatus::InsufficientLocalMemory;
    }
    if ((kvLengths == nullptr && lengthCount != 0) || (kvLengths != nullptr && lengthCount != uint64_t(shape.batch))) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    bool allFull = true;
    bool sameKvLengths = true;
    // Complete validation precedes any source-vector allocation or narrowing.
    for (uint64_t b = 0; b < lengthCount; ++b) {
        if (kvLengths[b] < 0 || kvLengths[b] > shape.kvLength) {
            return MlaScheduleStatus::UnsupportedShape;
        }
        allFull = allFull && kvLengths[b] == shape.kvLength;
        sameKvLengths = sameKvLengths && kvLengths[b] == kvLengths[0];
    }
    uint32_t batch = static_cast<uint32_t>(shape.batch);
    MlaSchedule schedule{};
    schedule.bSize = batch;
    schedule.n2Size = 1;
    schedule.gSize = static_cast<uint32_t>(shape.queryHeads);
    schedule.s1Size = static_cast<uint32_t>(shape.queryLength);
    schedule.s2Size = static_cast<uint32_t>(shape.kvLength);
    schedule.headDim = 512;
    schedule.headDimRope = 64;
    schedule.aicNum = platform.aicNum;
    schedule.aivNum = platform.aivNum;
    schedule.cvRatio = ratio;
    schedule.scheduleMode = 1;
    if (batch == 0 || rows == 0 || shape.kvLength == 0) {
        // FA3 global-empty adaptation, not a CANN attention partition. A single
        // core validates metadata/lengths; no tiles, endpoints or scratch exist.
        schedule.usedCoreNum = 1;
        schedule.blockDim = platform.blockDims[0];
        output = schedule;
        return MlaScheduleStatus::Success;
    }
    if (uint64_t(batch / std::gcd(batch, platform.aicNum)) * platform.aicNum > u32Max) {
        return MlaScheduleStatus::FieldOverflow;
    }
    schedule.mBaseSize = CalcMBaseSize(batch, static_cast<uint32_t>(rows), platform.aicNum, sameKvLengths);
    // First and only CalcInnerSize pass: splitKVFlag=false, pageAttentionFlag=false.
    schedule.s2BaseSize = std::min(schedule.s2Size, 512U);
    schedule.requiredLocalMemory = requiredMemory;
    uint32_t mBlocks = (rows + schedule.mBaseSize - 1) / schedule.mBaseSize;
    uint32_t s2Blocks = (schedule.s2Size + schedule.s2BaseSize - 1) / schedule.s2BaseSize;
    if (uint64_t(batch) * mBlocks * s2Blocks > u32Max) {
        return MlaScheduleStatus::FieldOverflow;
    }

    auto& workspace = schedule.workspace;
    uint64_t mSize = Align(std::min(rows, uint64_t(schedule.mBaseSize)), 16);
    workspace.mm1ResSize = mSize * Align(schedule.s2BaseSize, 32);
    workspace.mm2ResSize = mSize * schedule.headDim;
    workspace.fdAccumOutSize = uint64_t(platform.aicNum) * 2 * schedule.mBaseSize * schedule.headDim;
    workspace.fdLogSumExpSize = uint64_t(platform.aicNum) * 2 * 2 * schedule.mBaseSize * (32 / sizeof(float));
    // Source InitParams sets coreNum_=aicNum_, not usedCoreNum_. Preserve its reserve.
    workspace.userBytes =
        uint64_t(2) * platform.aicNum * (6 * workspace.mm1ResSize + 8 * workspace.mm2ResSize + 8 * schedule.mBaseSize);
    workspace.libApiBytes = platform.libApiWorkspaceBytes;
    if (workspace.libApiBytes > std::numeric_limits<uint64_t>::max() - workspace.userBytes) {
        return MlaScheduleStatus::WorkspaceOverflow;
    }
    workspace.totalBytes = workspace.userBytes + workspace.libApiBytes;

    detail::BaseInfo base{};
    base.bSize = batch;
    base.n2Size = 1;
    base.gSize = schedule.gSize;
    base.s1Size = schedule.s1Size;
    base.s2Size = schedule.s2Size;
    if (!allFull) {
        base.actualLenKvDims = batch;
        base.actualSeqS2Size.reserve(batch);
        for (uint32_t b = 0; b < batch; ++b)
            base.actualSeqS2Size.push_back(static_cast<int64_t>(kvLengths[b]));
    }
    detail::SplitParam param{};
    param.mBaseSize = schedule.mBaseSize;
    param.s2BaseSize = schedule.s2BaseSize;
    detail::SplitResult split{platform.aicNum, ratio};
    SplitWithoutFlashDecode(platform.aicNum, base, param, mBlocks, split);
    // Do not force FD fields to zero to make a non-FD launch appear valid.
    if (split.numOfFdHead != 0 || split.maxS2SplitNum != 0 || split.usedVecNumOfFd != 0) {
        return MlaScheduleStatus::UnexpectedFlashDecode;
    }
    if (!ValidSplit(split, base, mBlocks, platform.aicNum)) {
        return MlaScheduleStatus::InvalidSplit;
    }
    schedule.usedCoreNum = split.usedCoreNum;
    schedule.blockDim = platform.blockDims[split.usedCoreNum - 1];
    // AIC endpoints must all have a launched block. The source kernel uses GetBlockNum().
    if (schedule.blockDim < schedule.usedCoreNum) {
        return MlaScheduleStatus::InvalidPlatform;
    }
    schedule.maxCost = split.maxCost;
    CopyField(schedule.bN2End, split.bN2End);
    CopyField(schedule.gS1End, split.gS1End);
    CopyField(schedule.s2End, split.s2End);
    auto& fd = schedule.fd;
    fd.numOfFdHead = split.numOfFdHead;
    fd.maxS2SplitNum = split.maxS2SplitNum;
    fd.usedVecNumOfFd = split.usedVecNumOfFd;
    fd.gS1BaseSizeOfFd = param.gS1BaseSizeOfFd;
    CopyField(fd.bN2IdxOfFdHead, split.fdRes.bN2IdxOfFdHead);
    CopyField(fd.gS1IdxOfFdHead, split.fdRes.gS1IdxOfFdHead);
    CopyField(fd.s2SplitNumOfFdHead, split.fdRes.s2SplitNumOfFdHead);
    CopyField(fd.s2SplitStartIdxOfCore, split.fdRes.s2SplitStartIdxOfCore);
    CopyField(fd.gS1SplitNumOfFdHead, split.fdRes.gS1SplitNumOfFdHead);
    CopyField(fd.gS1LastPartSizeOfFdHead, split.fdRes.gS1LastPartSizeOfFdHead);
    // Source SetSplitOutput uses usedCoreNum*2 even for C1V1. Read only the actual
    // aicNum*cvRatio vector entries; the rest of our fixed-capacity arrays stay zero.
    CopyField(fd.gS1IdxEndOfFdHead, split.fdRes.gS1IdxEndOfFdHead);
    CopyField(fd.gS1IdxEndOfFdHeadSplit, split.fdRes.gS1IdxEndOfFdHeadSplit);

    // fia_kernel_nonquant_mla.h::Init: two GM slots, offsets in bytes from user workspace.
    uint64_t slots = uint64_t(2) * schedule.blockDim;
    workspace.pOffsetBytes = slots * workspace.mm1ResSize * 4;
    workspace.pvOffsetBytes = workspace.pOffsetBytes + slots * workspace.mm1ResSize * 2;
    workspace.vec2OffsetBytes = workspace.pvOffsetBytes + slots * workspace.mm2ResSize * 4;
    workspace.kernelBytes = workspace.vec2OffsetBytes + slots * workspace.mm2ResSize * 4;
    if (workspace.kernelBytes > workspace.userBytes) {
        return MlaScheduleStatus::WorkspaceOverflow;
    }
    output = schedule;
    return MlaScheduleStatus::Success;
}

} // namespace

MlaScheduleStatus BuildBsndKvLengthSchedule(const MlaDenseShape& shape, const MlaPlatformInfo& platform,
                                            const int32_t* kvLengths, uint64_t lengthCount, MlaSchedule& output)
{
    try {
        return Calculate(shape, platform, kvLengths, lengthCount, output);
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}

MlaScheduleStatus BuildDenseSchedule(const MlaDenseShape& shape, const MlaPlatformInfo& platform, MlaSchedule& output)
{
    return BuildBsndKvLengthSchedule(shape, platform, nullptr, 0, output);
}

} // namespace flash_attn::mla

// Derived from CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.cpp:
// GetS1Size, GetMaxActualSeq, ProcessKVActualSeqLoop; and op_host/arch22/fia_tiling_nonquant_mla.cpp:
// CalcInnerSize, CalcMBaseSize, CreateSplitInput, CalcMmResSize, CalcNormalWorkspaceSize.
// Changes: FA3 int32 prefix validation and owned token spans; numerical int64
// cumulative tails for the existing SplitCore; whole-M endpoint mapping for
// num_splits=1 with variable Q lengths.
// Mixed TND Q/BSND KV additionally preserves physical capacity, noncumulative
// source KV lengths and 64-bit physical starts.
// Independent used-length CPU entry: allocation spans remain physical; explicit
// used arrays override source actual lengths without altering those spans.
// Independent PA CPU entry: logical capacity is validated separately from the
// source GetS2SizeForPageAttention page-rounded maximum effective KV length.
// Preserve PA CalcInnerSize order, sharing normalization and whole-M mapping.

namespace flash_attn::mla {
namespace {

uint64_t AlignTnd(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

bool ValidUsedCounts(uint64_t batch, const MlaSequenceAllocation& allocation, const MlaLengthArray& prefix,
                     const MlaLengthArray& used)
{
    if ((used.data == nullptr && used.count != 0) || (used.data != nullptr && used.count != batch)) {
        return false;
    }
    return allocation.layout == MlaSequenceLayout::Tnd ? prefix.data != nullptr && prefix.count == batch + 1
                                                       : prefix.data == nullptr && prefix.count == 0;
}

MlaScheduleStatus ValidateUsedShape(const MlaUsedShape& shape, const MlaUsedInputs& inputs)
{
    const auto bsnd = MlaSequenceLayout::Bsnd, tnd = MlaSequenceLayout::Tnd;
    if (!((shape.query.layout == bsnd && shape.kv.layout == bsnd) ||
          (shape.query.layout == tnd && (shape.kv.layout == bsnd || shape.kv.layout == tnd)))) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    for (int64_t size : {shape.batch, shape.queryHeads, shape.query.maxLength, shape.kv.maxLength}) {
        if (size < 0) {
            return MlaScheduleStatus::UnsupportedShape;
        }
        if (uint64_t(size) > UINT32_MAX) {
            return MlaScheduleStatus::FieldOverflow;
        }
    }
    if (shape.queryHeads == 0 || shape.kv.maxLength > kMlaMaxKvLength || shape.query.totalTokens < 0 ||
        shape.kv.totalTokens < 0) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    if (shape.queryHeads > kMlaMaxQueryHeads || shape.query.maxLength > kMlaMaxQueryLength ||
        uint64_t(shape.query.maxLength) * shape.queryHeads > kMlaMaxDenseRows ||
        uint64_t(shape.query.totalTokens) > kMlaMaxBatchRows / shape.queryHeads) {
        return MlaScheduleStatus::FieldOverflow;
    }
    for (const auto& allocation : {shape.query, shape.kv}) {
        if (allocation.layout == tnd) {
            if (allocation.totalTokens > INT32_MAX) {
                return MlaScheduleStatus::FieldOverflow;
            }
        } else if (uint64_t(allocation.totalTokens) != uint64_t(shape.batch) * allocation.maxLength) {
            return MlaScheduleStatus::UnsupportedShape;
        }
    }
    return ValidUsedCounts(shape.batch, shape.query, inputs.cuQuery, inputs.usedQuery) &&
                   ValidUsedCounts(shape.batch, shape.kv, inputs.cuKv, inputs.usedKv)
               ? MlaScheduleStatus::Success
               : MlaScheduleStatus::UnsupportedShape;
}

uint64_t UsedStart(uint64_t batch, const MlaSequenceAllocation& allocation, const MlaLengthArray& prefix)
{
    return allocation.layout == MlaSequenceLayout::Tnd ? uint64_t(prefix.data[batch])
                                                       : batch * uint64_t(allocation.maxLength);
}

uint32_t UsedCapacity(uint64_t batch, const MlaSequenceAllocation& allocation, const MlaLengthArray& prefix)
{
    return allocation.layout == MlaSequenceLayout::Tnd ? uint32_t(prefix.data[batch + 1] - prefix.data[batch])
                                                       : uint32_t(allocation.maxLength);
}

bool ValidUsedValues(uint64_t batch, const MlaSequenceAllocation& allocation, const MlaLengthArray& prefix,
                     const MlaLengthArray& used)
{
    const bool tnd = allocation.layout == MlaSequenceLayout::Tnd;
    if (tnd && (prefix.data[0] != 0 || prefix.data[batch] != allocation.totalTokens)) {
        return false;
    }
    for (uint64_t b = 0; b < batch; ++b) {
        // Reject negative/decreasing prefixes before subtraction, including INT32_MIN.
        if (tnd && prefix.data[b + 1] < prefix.data[b]) {
            return false;
        }
        const uint32_t capacity = UsedCapacity(b, allocation, prefix);
        if (capacity > allocation.maxLength || (used.data && (used.data[b] < 0 || uint32_t(used.data[b]) > capacity))) {
            return false;
        }
    }
    return true;
}

MlaScheduleStatus Parse(const MlaUsedShape& shape, const MlaUsedInputs& inputs, MlaUsedLengths& result)
{
    const auto status = ValidateUsedShape(shape, inputs);
    if (status != MlaScheduleStatus::Success) {
        return status;
    }
    if (!ValidUsedValues(shape.batch, shape.query, inputs.cuQuery, inputs.usedQuery) ||
        !ValidUsedValues(shape.batch, shape.kv, inputs.cuKv, inputs.usedKv)) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    result.sequences.reserve(shape.batch);
    for (uint64_t b = 0; b < uint64_t(shape.batch); ++b) {
        const uint32_t qCapacity = UsedCapacity(b, shape.query, inputs.cuQuery);
        const uint32_t kvCapacity = UsedCapacity(b, shape.kv, inputs.cuKv);
        const uint32_t q = inputs.usedQuery.data ? uint32_t(inputs.usedQuery.data[b]) : qCapacity;
        const uint32_t kv = inputs.usedKv.data ? uint32_t(inputs.usedKv.data[b]) : kvCapacity;
        result.maxQueryCapacity = std::max(result.maxQueryCapacity, qCapacity);
        result.maxKvCapacity = std::max(result.maxKvCapacity, kvCapacity);
        result.maxQueryLength = std::max(result.maxQueryLength, q);
        result.maxKvLength = std::max(result.maxKvLength, kv);
        result.sequences.push_back({UsedStart(b, shape.query, inputs.cuQuery), UsedStart(b, shape.kv, inputs.cuKv),
                                    qCapacity, kvCapacity, q, kv});
    }
    return MlaScheduleStatus::Success;
}

bool ValidSplit(const detail::SplitResult& split, const detail::BaseInfo& base, const std::vector<uint32_t>& blocks,
                uint32_t aicNum)
{
    if (split.usedCoreNum == 0 || split.usedCoreNum > aicNum) {
        return false;
    }
    std::pair<uint32_t, uint32_t> previous{0, 0};
    for (uint32_t i = 0; i < split.usedCoreNum; ++i) {
        const auto b = split.bN2End[i], m = split.gS1End[i];
        if (b > base.bSize || split.s2End[i] != 0 || (b == base.bSize ? m != 0 : m >= std::max(blocks[b], 1U))) {
            return false;
        }
        const std::pair<uint32_t, uint32_t> end{b, m};
        if (end <= previous) {
            return false;
        }
        previous = end;
    }
    // As in the source, compute endpoints may omit a zero-cost suffix.
    // Device consumers initialize zero-KV outputs separately.
    for (uint32_t b = previous.first; b < base.bSize; ++b) {
        if (detail::GetS1SeqSize(b, base) != 0 && detail::GetS2SeqSize(b, base) != 0) {
            return false;
        }
    }
    return true;
}

template <typename Lengths>
MlaScheduleStatus Calculate(uint32_t batch, uint32_t heads, uint32_t queryCapacity, uint32_t kvCapacity,
                            bool compactQuery, bool compactKv, const MlaPlatformInfo& platform, const Lengths& lengths,
                            MlaSchedule& s, uint32_t pageSize = 0)
{
    // Reuse existing platform/local-memory validation and static geometry.
    // B=0 deliberately avoids the rectangular B*maxQ address/tile bound: TND
    // has only totalQuery rows. Parse already checked its true address bound.
    auto status = BuildDenseSchedule({0, queryCapacity, heads, kvCapacity}, platform, s);
    if (status != MlaScheduleStatus::Success) {
        return status;
    }
    s.bSize = batch;
    if (batch == 0 || queryCapacity == 0 || kvCapacity == 0) {
        return MlaScheduleStatus::Success;
    }
    // Preserve ProcessKVActualSeqLoop: equality compares the source array's
    // values: cumulative tails for compact KV, per-sample lengths for BSND KV.
    const uint32_t firstKv = lengths.sequences[0].kvLength;
    bool sameKvValues = true;
    for (const auto& seq : lengths.sequences) {
        sameKvValues = sameKvValues && (compactKv ? seq.kvStart + seq.kvLength : seq.kvLength) == firstKv;
    }
    const uint64_t rows = uint64_t(s.s1Size) * s.gSize;
    s.mBaseSize = 256;
    if (s.bSize % s.aicNum != 0 && sameKvValues) {
        const uint64_t lcm = uint64_t(s.bSize / std::gcd(s.bSize, s.aicNum)) * s.aicNum;
        if (lcm > UINT32_MAX) {
            return MlaScheduleStatus::FieldOverflow;
        }
        const uint32_t parts = lcm / s.bSize;
        if (rows % parts == 0 && rows / parts >= 128 && rows / parts <= 256) {
            s.mBaseSize = rows / parts;
        }
    }
    s.s2BaseSize = 512;
    // Source CalcInnerSize: PA alignment precedes short-sequence truncation.
    // P=112 gives 448; P=528 retains 512 and therefore still crosses pages.
    if (pageSize != 0 && s.s2BaseSize > pageSize && s.s2BaseSize % pageSize != 0) {
        s.s2BaseSize = s.s2BaseSize / pageSize * pageSize;
    }
    s.s2BaseSize = std::min(s.s2Size, s.s2BaseSize);
    s.requiredLocalMemory = {7 * 128 * 288 * 2, (s.cvRatio == 1 ? 174U : 170U) * 1024, 65536, 65536, 131072};
    uint64_t totalTiles = 0;
    for (const auto& seq : lengths.sequences) {
        const uint64_t m = (uint64_t(seq.queryLength) * s.gSize + s.mBaseSize - 1) / s.mBaseSize;
        totalTiles += m * ((seq.kvLength + s.s2BaseSize - 1) / s.s2BaseSize);
        if (totalTiles > UINT32_MAX) {
            return MlaScheduleStatus::FieldOverflow;
        }
    }
    auto& w = s.workspace;
    const uint64_t m = AlignTnd(std::min(rows, uint64_t(s.mBaseSize)), 16);
    w.mm1ResSize = m * AlignTnd(s.s2BaseSize, 32);
    w.mm2ResSize = m * s.headDim;
    w.fdAccumOutSize = uint64_t(s.aicNum) * 2 * s.mBaseSize * s.headDim;
    w.fdLogSumExpSize = uint64_t(s.aicNum) * 2 * 2 * s.mBaseSize * (32 / sizeof(float));
    w.userBytes = uint64_t(2) * s.aicNum * (6 * w.mm1ResSize + 8 * w.mm2ResSize + 8 * s.mBaseSize);
    w.libApiBytes = platform.libApiWorkspaceBytes;
    if (w.libApiBytes > UINT64_MAX - w.userBytes) {
        return MlaScheduleStatus::WorkspaceOverflow;
    }
    w.totalBytes = w.userBytes + w.libApiBytes;
    s.fd.gS1BaseSizeOfFd = detail::SplitParam{}.gS1BaseSizeOfFd;
    return MlaScheduleStatus::Success;
}

} // namespace

MlaScheduleStatus ParseUsedLengths(const MlaUsedShape& shape, const MlaUsedInputs& inputs, MlaUsedLengths& output)
{
    try {
        MlaUsedLengths result;
        const auto status = Parse(shape, inputs, result);
        if (status == MlaScheduleStatus::Success) {
            output = std::move(result);
        }
        return status;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}

MlaScheduleStatus PrepareMlaGeometry(const MlaUsedShape& shape, const MlaPlatformInfo& platform,
                                     const MlaUsedInputs& inputs, const MlaUsedLengths& lengths, uint32_t pageSize,
                                     MlaSchedule& output)
{
    const bool tndQ = shape.query.layout == MlaSequenceLayout::Tnd;
    const bool tndK = shape.kv.layout == MlaSequenceLayout::Tnd;
    const uint32_t q = tndQ ? lengths.maxQueryCapacity : shape.query.maxLength;
    const uint32_t k = pageSize ? AlignTnd(lengths.maxKvLength, pageSize)
                       : tndK   ? lengths.maxKvCapacity
                                : shape.kv.maxLength;
    return Calculate(shape.batch, shape.queryHeads, q, k, tndQ && !inputs.usedQuery.data, tndK && !inputs.usedKv.data,
                     platform, lengths, output, pageSize);
}

} // namespace flash_attn::mla

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/common/op_host/split_core.cpp: CalcS2Range, CalcNextTokenLeftUp,
// CalcS1GCache, SplitCore.
// Changes: checked effective S1G geometry, explicit empty-row ownership, and
// FA3 num_splits=1 whole-M endpoints with causal costs recomputed per block.

namespace flash_attn::mla {
namespace {

MlaCausalStatus Validate(int64_t q, int64_t k, int64_t heads, int64_t m, int64_t n)
{
    for (int64_t size : {q, k, heads}) {
        if (size < 0) {
            return MlaCausalStatus::UnsupportedShape;
        }
        if (uint64_t(size) > UINT32_MAX) {
            return MlaCausalStatus::FieldOverflow;
        }
    }
    if (heads == 0 || k > kMlaMaxKvLength) {
        return MlaCausalStatus::UnsupportedShape;
    }
    if (q > kMlaMaxQueryLength || heads > kMlaMaxQueryHeads || uint64_t(q) * heads > kMlaMaxDenseRows) {
        return MlaCausalStatus::FieldOverflow;
    }
    if (m <= 0 || m > 256 || n <= 0 || n > 512) {
        return MlaCausalStatus::InvalidTile;
    }
    return MlaCausalStatus::Success;
}

detail::BaseInfo CausalBase(uint32_t heads)
{
    detail::BaseInfo base{};
    base.n2Size = 1;
    base.gSize = heads;
    base.attenMaskFlag = true;
    base.sparseMode = int32_t(detail::SparseMode::RIGHT_DOWN_CAUSAL);
    // Unlimited left context, with ample headroom for the source subtraction.
    base.preToken = INT32_MAX;
    return base;
}

MlaCausalStatus CompleteMPartition(const MlaCausalScheduleInput& in, detail::BaseInfo& base, MlaCausalSchedule& result)
{
    base.bSize = in.sequences.size();
    base.actualLenQDims = base.actualLenKvDims = base.bSize;
    base.actualSeqS1Size.reserve(in.sequences.size());
    base.actualSeqS2Size.reserve(in.sequences.size());
    std::vector<uint64_t> prefix;
    prefix.reserve(in.sequences.size() + 1);
    prefix.push_back(0);
    for (const auto& seq : in.sequences) {
        base.actualSeqS1Size.push_back(seq.queryLength);
        base.actualSeqS2Size.push_back(seq.kvLength);
        prefix.push_back(prefix.back() + (uint64_t(seq.queryLength) * in.queryHeads + in.mBaseSize - 1) / in.mBaseSize);
    }
    result.totalMBlocks = prefix.back();
    detail::SplitParam param{};
    param.mBaseSize = in.mBaseSize;
    param.s2BaseSize = in.s2BaseSize;
    detail::SplitContext context(base, param);
    detail::CalcSplitInfo(context);
    detail::CalcCostInfo(context);
    result.totalCost = context.costInfo.totalCost;
    result.totalS2Blocks = context.costInfo.totalBlockNum;
    detail::SplitResult split{uint32_t(in.aicNum), uint32_t(in.cvRatio)};
    detail::SplitCore(in.aicNum, context, split);
    if (split.usedCoreNum == 0 || split.usedCoreNum > uint64_t(in.aicNum)) {
        return MlaCausalStatus::InvalidSplit;
    }
    uint64_t previous = 0;
    uint32_t costBatch = 0, costM = 0;
    int64_t assignedCost = 0;
    for (uint32_t c = 0; c < split.usedCoreNum; ++c) {
        uint32_t b = split.bN2End[c];
        uint64_t m = uint64_t(split.gS1End[c]) + (split.s2End[c] != 0);
        if (b > base.bSize || (b == base.bSize && m != 0) || (b < base.bSize && m > prefix[b + 1] - prefix[b])) {
            return MlaCausalStatus::InvalidSplit;
        }
        // Include all trailing zero-cost query blocks: their O/LSE need owners.
        if (c + 1 == split.usedCoreNum) {
            b = base.bSize;
            m = 0;
        }
        const uint64_t end = prefix[b] + m;
        if (end < previous) {
            return MlaCausalStatus::InvalidSplit;
        }
        if (end == previous && result.totalMBlocks != 0) {
            continue;
        }
        // Canonicalize a whole-M endpoint past completed/empty samples.
        while (b < base.bSize && m == prefix[b + 1] - prefix[b]) {
            ++b;
            m = 0;
        }
        int64_t cost = 0;
        while (costBatch < b || (costBatch == b && costM < m)) {
            const uint64_t limit = costBatch < b ? prefix[costBatch + 1] - prefix[costBatch] : m;
            if (in.sequences[costBatch].kvLength != 0) {
                detail::BatchCache batch{};
                detail::CalcBatchCache(costBatch, context, batch);
                for (; costM < limit; ++costM) {
                    detail::S1GCache block{};
                    detail::CalcS1GCache(costM, context, batch, block);
                    cost += block.s1GCost;
                }
            }
            if (costBatch == b) {
                costM = m;
                break;
            }
            ++costBatch;
            costM = 0;
        }
        const uint32_t core = result.usedCoreNum++;
        result.bN2End[core] = b;
        result.gS1End[core] = m;
        result.coreCost[core] = cost;
        result.maxCost = std::max(result.maxCost, cost);
        assignedCost += cost;
        previous = end;
    }
    return result.usedCoreNum != 0 && previous == result.totalMBlocks && assignedCost == result.totalCost
               ? MlaCausalStatus::Success
               : MlaCausalStatus::InvalidSplit;
}

MlaCausalStatus Calculate(const MlaCausalScheduleInput& in, MlaCausalSchedule& result)
{
    auto status = Validate(0, 0, in.queryHeads, in.mBaseSize, in.s2BaseSize);
    if (status != MlaCausalStatus::Success) {
        return status;
    }
    if (in.aicNum < 1 || in.aicNum > kMlaMaxAic || (in.cvRatio != 1 && in.cvRatio != 2)) {
        return MlaCausalStatus::InvalidPlatform;
    }
    if (in.sequences.size() > UINT32_MAX) {
        return MlaCausalStatus::FieldOverflow;
    }
    // Check before the source's uint32 products, sums and ceil-div operations.
    // Use the same conservative rectangular tile bound as existing schedulers.
    uint64_t rows = 0, tiles = 0;
    for (const auto& seq : in.sequences) {
        status = Validate(seq.queryLength, seq.kvLength, in.queryHeads, in.mBaseSize, in.s2BaseSize);
        if (status != MlaCausalStatus::Success) {
            return status;
        }
        const uint64_t count = uint64_t(seq.queryLength) * in.queryHeads;
        rows += count;
        tiles += ((count + in.mBaseSize - 1) / in.mBaseSize) * ((seq.kvLength + in.s2BaseSize - 1) / in.s2BaseSize);
        if (rows > kMlaMaxBatchRows || tiles > UINT32_MAX) {
            return MlaCausalStatus::FieldOverflow;
        }
    }
    auto base = CausalBase(in.queryHeads);
    return CompleteMPartition(in, base, result);
}
} // namespace

MlaCausalStatus BuildCausalBlock(const MlaCausalBlockInput& in, MlaCausalBlock& output)
{
    const auto status = Validate(in.queryLength, in.kvLength, in.queryHeads, in.mBaseSize, in.s2BaseSize);
    if (status != MlaCausalStatus::Success) {
        return status;
    }
    const uint64_t rows = uint64_t(in.queryLength) * in.queryHeads;
    const uint64_t blocks = (rows + in.mBaseSize - 1) / in.mBaseSize;
    if (in.mBlock < 0 || uint64_t(in.mBlock) >= std::max(blocks, uint64_t(1))) {
        return MlaCausalStatus::InvalidBlock;
    }
    MlaCausalBlock result{};
    auto base = CausalBase(in.queryHeads);
    detail::SplitParam param{};
    param.mBaseSize = in.mBaseSize;
    param.s2BaseSize = in.s2BaseSize;
    detail::BatchCache batch{};
    batch.s1Size = in.queryLength;
    batch.s2Size = in.kvLength;
    batch.preTokenLeftUp = base.preToken;
    batch.nextTokenLeftUp = detail::CalcNextTokenLeftUp(batch.s1Size, batch.s2Size, base);
    result.nextTokenLeftUp = batch.nextTokenLeftUp;
    result.rowBegin = uint64_t(in.mBlock) * in.mBaseSize;
    result.rowEnd = std::min(uint64_t(result.rowBegin) + in.mBaseSize, rows);
    if (rows != 0) {
        result.queryBegin = result.rowBegin / in.queryHeads;
        result.queryEnd = (result.rowEnd - 1) / in.queryHeads + 1;
        result.kvEnd = std::max(int64_t(0), int64_t(result.queryEnd) + result.nextTokenLeftUp);
        const int64_t emptyEnd = std::max(in.queryLength - in.kvLength, int64_t(0)) * in.queryHeads;
        result.leadingEmptyRows =
            std::clamp(emptyEnd - result.rowBegin, int64_t(0), int64_t(result.rowEnd - result.rowBegin));
    }
    const auto range = detail::CalcS2Range(in.mBlock, base, param, batch);
    result.s2Begin = range.first;
    result.s2End = range.second;
    if (result.s2End != 0) {
        result.normalS2Cost = detail::CalcCost(result.rowEnd - result.rowBegin, in.s2BaseSize);
        const uint32_t lastSize =
            std::min(uint64_t(in.s2BaseSize), uint64_t(in.kvLength) - uint64_t(result.s2End - 1) * in.s2BaseSize);
        result.lastS2Cost = detail::CalcCost(result.rowEnd - result.rowBegin, lastSize);
        result.cost = int64_t(result.s2End - 1) * result.normalS2Cost + result.lastS2Cost;
    }
    output = result;
    return MlaCausalStatus::Success;
}

MlaCausalStatus BuildCausalSchedule(const MlaCausalScheduleInput& input, MlaCausalSchedule& output)
{
    try {
        MlaCausalSchedule result{};
        const auto status = Calculate(input, result);
        if (status == MlaCausalStatus::Success) {
            output = result;
        }
        return status;
    } catch (const std::bad_alloc&) {
        return MlaCausalStatus::AllocationFailure;
    }
}

} // namespace flash_attn::mla

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/common/op_host/split_core.cpp (band range/cost/SplitCore), adapted
// through the existing causal complete-M endpoint algorithm.
// FA3 normalization follows Dao-AILab/flash-attention
// d15f1531a460ba456f41b01a774f33ab2db8febf hopper/flash_api.cpp.
// Changes: checked bounds, explicit empty rows, no tensor/runtime dependency.

namespace flash_attn::mla {

MlaCausalStatus NormalizeWindow(const MlaWindowConfig& in, MlaWindow& output)
{
    if (in.maxQueryLength < 0 || in.maxKvLength < 0) {
        return MlaCausalStatus::UnsupportedShape;
    }
    if (uint64_t(in.maxQueryLength) > kMlaMaxQueryLength || uint64_t(in.maxKvLength) > UINT32_MAX) {
        return MlaCausalStatus::FieldOverflow;
    }
    if (in.maxKvLength > kMlaMaxKvLength) {
        return MlaCausalStatus::UnsupportedShape;
    }
    int64_t left = in.windowLeft >= in.maxKvLength - 1 ? -1 : in.windowLeft;
    int64_t right = in.windowRight >= in.maxQueryLength - 1 ? -1 : in.windowRight;
    const bool causal = in.causal && !(in.maxQueryLength == 1 && left == -1 && right == -1);
    if (causal) {
        right = 0;
    }
    // FA3 forward passes these host-normalized values to int parameters in
    // set_params_fprop; metadata retains int64. Reject lossy negative narrowing
    // instead of claiming the two paths have the same mask in that domain.
    if (left < INT32_MIN || right < INT32_MIN) {
        return MlaCausalStatus::FieldOverflow;
    }
    MlaWindow result{};
    result.kind = left < 0 && right == 0    ? MlaWindowKind::Causal
                  : left >= 0 || right >= 0 ? MlaWindowKind::Local
                                            : MlaWindowKind::None;
    result.left = left < 0 ? in.maxKvLength - 1 : left;
    result.right = right < 0 ? in.maxQueryLength - 1 : right;
    output = result;
    return MlaCausalStatus::Success;
}

namespace {
MlaCausalStatus ValidateSequence(const MlaCausalBlockInput& in, const MlaWindowConfig& config)
{
    MlaCausalBlock block{};
    const auto status = BuildCausalBlock(in, block);
    if (status != MlaCausalStatus::Success) {
        return status;
    }
    return in.queryLength <= config.maxQueryLength && in.kvLength <= config.maxKvLength
               ? MlaCausalStatus::Success
               : MlaCausalStatus::UnsupportedShape;
}

detail::BaseInfo WindowBase(uint32_t heads, const MlaWindow& window)
{
    detail::BaseInfo base{};
    base.n2Size = 1;
    base.gSize = heads;
    base.attenMaskFlag = window.kind != MlaWindowKind::None;
    base.sparseMode = int32_t(window.kind == MlaWindowKind::Local    ? detail::SparseMode::BAND
                              : window.kind == MlaWindowKind::Causal ? detail::SparseMode::RIGHT_DOWN_CAUSAL
                                                                     : detail::SparseMode::DEFAULT_MASK);
    base.preToken = window.kind == MlaWindowKind::Causal ? INT32_MAX : window.left;
    base.nextToken = window.kind == MlaWindowKind::Causal ? 0 : window.right;
    return base;
}

MlaCausalStatus Calculate(const MlaCausalScheduleInput& in, const MlaWindowConfig& config, const MlaWindow& window,
                          MlaCausalSchedule& result)
{
    auto status = ValidateSequence({0, 0, in.queryHeads, in.mBaseSize, in.s2BaseSize, 0}, config);
    if (status != MlaCausalStatus::Success) {
        return status;
    }
    if (in.aicNum < 1 || in.aicNum > kMlaMaxAic || (in.cvRatio != 1 && in.cvRatio != 2)) {
        return MlaCausalStatus::InvalidPlatform;
    }
    if (in.sequences.size() > UINT32_MAX) {
        return MlaCausalStatus::FieldOverflow;
    }
    // Check before the source's uint32 products, sums and ceil-div operations.
    // Use the same conservative rectangular tile bound as existing schedulers.
    uint64_t rows = 0, tiles = 0;
    for (const auto& seq : in.sequences) {
        status =
            ValidateSequence({seq.queryLength, seq.kvLength, in.queryHeads, in.mBaseSize, in.s2BaseSize, 0}, config);
        if (status != MlaCausalStatus::Success) {
            return status;
        }
        const uint64_t count = uint64_t(seq.queryLength) * in.queryHeads;
        rows += count;
        tiles += ((count + in.mBaseSize - 1) / in.mBaseSize) * ((seq.kvLength + in.s2BaseSize - 1) / in.s2BaseSize);
        if (rows > kMlaMaxBatchRows || tiles > UINT32_MAX) {
            return MlaCausalStatus::FieldOverflow;
        }
    }
    auto base = WindowBase(in.queryHeads, window);
    return CompleteMPartition(in, base, result);
}
} // namespace

MlaCausalStatus BuildWindowSchedule(const MlaCausalScheduleInput& input, const MlaWindowConfig& config,
                                    MlaCausalSchedule& output)
{
    MlaWindow window{};
    auto status = NormalizeWindow(config, window);
    if (status != MlaCausalStatus::Success) {
        return status;
    }
    if (window.kind == MlaWindowKind::Causal) {
        for (const auto& seq : input.sequences) {
            status = ValidateSequence(
                {seq.queryLength, seq.kvLength, input.queryHeads, input.mBaseSize, input.s2BaseSize, 0}, config);
            if (status != MlaCausalStatus::Success) {
                return status;
            }
        }
        return BuildCausalSchedule(input, output);
    }
    try {
        MlaCausalSchedule result{};
        status = Calculate(input, config, window, result);
        if (status == MlaCausalStatus::Success) {
            output = result;
        }
        return status;
    } catch (const std::bad_alloc&) {
        return MlaCausalStatus::AllocationFailure;
    }
}
} // namespace flash_attn::mla
