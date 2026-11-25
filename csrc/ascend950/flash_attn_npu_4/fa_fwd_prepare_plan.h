// Copyright (c) 2026, Minghua Shen.
#ifndef FLASH_ATTN_950_V4_FA_FWD_PREPARE_PLAN_H
#define FLASH_ATTN_950_V4_FA_FWD_PREPARE_PLAN_H

#include "fa_fwd_prepare.h"
#include "fa_metadata_plan.h"
#include "../attention_window.h"
#include "tiling.cpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace fa4_fwd_plan {
namespace {

struct SequenceBounds {
    uint32_t minQ = std::numeric_limits<uint32_t>::max();
    uint32_t maxQ = 0;
    uint32_t maxKv = 0;
};

FA4FwdStatus ReadLengths(const FA4FwdPrepareArgs& args, SequenceBounds& bounds)
{
    const auto* cuQ = reinterpret_cast<const int32_t*>(args.qOffsets);
    const auto* cuK = reinterpret_cast<const int32_t*>(args.kOffsets);
    const auto* usedQ = reinterpret_cast<const int32_t*>(args.qUsed);
    const auto* usedK = reinterpret_cast<const int32_t*>(args.kUsed);
    const auto* pages = reinterpret_cast<const int32_t*>(args.pageTable);
    auto* kvLengths = reinterpret_cast<int32_t*>(args.kvLengths);
    auto* kvOffsets = reinterpret_cast<int32_t*>(args.kvOffsets);
    if ((cuQ != nullptr && cuQ[0] != 0) || (cuK != nullptr && cuK[0] != 0)) {
        return FA4FwdStatus::InvalidOffsets;
    }
    if (kvOffsets != nullptr) {
        kvOffsets[0] = 0;
    }
    int64_t packedKv = 0;
    for (uint32_t b = 0; b < args.batch; ++b) {
        const int64_t regionQ = cuQ != nullptr ? static_cast<int64_t>(cuQ[b + 1]) - cuQ[b] : args.maxQ;
        const int64_t regionK = cuK != nullptr ? static_cast<int64_t>(cuK[b + 1]) - cuK[b] : args.kvCapacity;
        if (regionQ < 0 || regionK < 0 || (cuQ != nullptr && cuQ[b + 1] > args.qTokens) ||
            (cuK != nullptr && cuK[b + 1] > args.kvCapacity)) {
            return FA4FwdStatus::InvalidOffsets;
        }
        const int64_t q = usedQ != nullptr ? usedQ[b] : regionQ;
        const int64_t k = usedK != nullptr ? usedK[b] : regionK;
        if (q < 0 || q > regionQ || q > args.maxQ || k < 0 || k > regionK ||
            (args.maxKv >= 0 && k > args.maxKv) ||
            (args.useHostSchedule && args.qTokens != 0 && (q == 0 || k == 0))) {
            return FA4FwdStatus::InvalidLengths;
        }
        if (pages != nullptr) {
            const int64_t activePages = (k + args.pageSize - 1) / args.pageSize;
            for (int64_t page = 0; page < activePages; ++page) {
                const int32_t id = pages[static_cast<uint64_t>(b) * args.maxPages + page];
                if (id < 0 || static_cast<uint32_t>(id) >= args.pageCount) {
                    return FA4FwdStatus::InvalidPage;
                }
            }
        }
        if (kvOffsets != nullptr) {
            packedKv += k;
            if (packedKv > args.kvCapacity || packedKv > std::numeric_limits<int32_t>::max()) {
                return FA4FwdStatus::InvalidOffsets;
            }
            kvOffsets[b + 1] = static_cast<int32_t>(packedKv);
        }
        if (kvLengths != usedK) {
            kvLengths[b] = static_cast<int32_t>(k);
        }
        bounds.minQ = std::min(bounds.minQ, static_cast<uint32_t>(q));
        bounds.maxQ = std::max(bounds.maxQ, static_cast<uint32_t>(q));
        bounds.maxKv = std::max(bounds.maxKv, static_cast<uint32_t>(k));
    }
    return FA4FwdStatus::Valid;
}

void FillHostSchedule(const FA4FwdPrepareArgs& args, const SequenceBounds& bounds, bool fdShape,
                      const FAInferRuntimeData& runtime, FAInferStaticTilingData& result)
{
    optiling::FAInferContext ctx;
    ctx.batch = args.batch;
    ctx.numHeads = args.heads;
    ctx.kvHeads = args.kvHeads;
    ctx.embeddingSize = args.qDim;
    ctx.embeddingSizeV = args.vDim;
    ctx.numBlocks = args.pageCount;
    ctx.blockSize = args.pageSize;
    ctx.maxNumBlocksPerBatch = args.maxPages;
    ctx.maxQSeqlen = bounds.maxQ;
    ctx.maxKvSeqlen = bounds.maxKv;
    ctx.scaleValue = runtime.scaleValue;
    ctx.softcapValue = runtime.softcapValue;
    ctx.maskType = static_cast<optiling::MaskType>(runtime.maskType);
    ctx.windowSizeLeft = runtime.maskType == 2 ? runtime.windowSizeLeft : 0;
    ctx.windowSizeRight = runtime.maskType == 2 ? runtime.windowSizeRight : 0;
    ctx.dataType = args.quantMode != 0 ? optiling::DataType::FP8
                                      : (args.bf16 ? optiling::DataType::BF16 : optiling::DataType::FP16);
    ctx.pagedCacheFlag = args.pageTable != 0;
    ctx.lseFlag = args.lse;
    ctx.flashDecodeFlag = fdShape;
    ctx.numSplits = args.splits;
    ctx.layout = args.layout;

    const auto* cuQ = reinterpret_cast<const int32_t*>(args.qOffsets);
    const auto* usedQ = reinterpret_cast<const int32_t*>(args.qUsed);
    const auto* usedK = reinterpret_cast<const int32_t*>(args.kvLengths);
    const bool packedQ = args.layout == FwdLayout::TND;
    const bool packedK = packedQ && !ctx.pagedCacheFlag;
    std::vector<int64_t> q(args.batch + packedQ);
    std::vector<int64_t> k(args.batch + packedK);
    for (uint32_t b = 0; b < args.batch; ++b) {
        if (packedQ) {
            q[b + 1] = usedQ != nullptr ? q[b] + usedQ[b] : cuQ[b + 1];
        } else {
            q[b] = args.maxQ;
        }
        const int64_t length = usedK != nullptr ? usedK[b] : args.kvCapacity;
        if (packedK) {
            k[b + 1] = k[b] + length;
        } else {
            k[b] = length;
        }
    }
    ctx.qSeqlenList = q.data();
    ctx.kvSeqlenList = k.data();
    FAInferTilingData tiling{};
    optiling::FAInferTiling scheduler(ctx);
    scheduler.SetCoreNum(args.coreCount);
    scheduler.DoTiling(tiling);
    result = static_cast<const FAInferStaticTilingData&>(tiling);
}

} // namespace

static inline FA4FwdStatus Prepare(const FA4FwdPrepareArgs& args, FA4FwdExecution& execution)
{
    SequenceBounds lengths;
    const auto status = ReadLengths(args, lengths);
    if (status != FA4FwdStatus::Valid) {
        return status;
    }
    bool causal = false;
    bool local = false;
    int64_t left = args.left;
    int64_t right = args.right;
    const int64_t windowBound = args.maxKv >= 0 ? args.maxKv : (args.inferMaxKv ? lengths.maxKv : args.kvCapacity);
    fa_window::NormalizeAttentionWindow(args.causal, args.maxQ, windowBound, left, right, causal, local);
    auto& runtime = execution.runtime;
    runtime.numBlocks = args.pageCount;
    runtime.blockSize = args.pageSize;
    runtime.maxNumBlocksPerBatch = args.maxPages;
    runtime.maskType = local ? 2U : (causal ? 1U : 0U);
    runtime.scaleValue = args.scale;
    runtime.softcapValue = args.softcap;
    runtime.windowSizeLeft = left;
    runtime.windowSizeRight = right;
    runtime.allowFlashDecode = 0;

    const uint32_t groupSize = args.heads / args.kvHeads;
    const bool d512 = fa950_plan::IsD512(args.qDim, args.vDim);
    const bool denseFd = d512 && args.layout == FwdLayout::BSND && args.pageTable == 0 && args.kOffsets == 0 &&
                         runtime.maskType == 0 && args.splits == 0 && args.maxQ > 0 && args.maxQ <= 16 &&
                         uint64_t(args.maxQ) * groupSize <= 128;
    const bool pagedFd = args.pageTable != 0 && args.layout == FwdLayout::TND && !local &&
                         lengths.minQ > 0 && lengths.maxQ <= 16;
    bool allowFd = lengths.maxKv >= 1024 && (denseFd || pagedFd) && args.splits != 1;

    if (args.useHostSchedule && args.metadata == 0) {
        FillHostSchedule(args, lengths, allowFd, runtime, execution.tiling);
    } else {
        const uint64_t baseTaskBound = uint64_t(args.batch) * args.kvHeads * ((groupSize + 3U) / 4U);
        const uint32_t coreCapacity = std::min(args.coreCount, MAX_FD_ACTIVE_CORE_NUM);
        allowFd = allowFd && baseTaskBound < coreCapacity && baseTaskBound <= (3U * args.coreCount - 1U) / 10U &&
                  (denseFd || (args.maxQ <= 16 && uint64_t(args.maxQ) * groupSize <= 128));
        FAMetadataArgs metadataArgs{};
        metadataArgs.cuSeqlensQAddr = args.qOffsets;
        metadataArgs.seqlensQAddr = args.qUsed;
        metadataArgs.seqlensKAddr = args.kvLengths;
        metadataArgs.batch = args.batch;
        metadataArgs.numHeads = args.heads;
        metadataArgs.numHeadsK = args.kvHeads;
        metadataArgs.embeddingSize = args.qDim;
        metadataArgs.embeddingSizeV = args.vDim;
        metadataArgs.maxQSeqlen = args.maxQ;
        metadataArgs.maskType = runtime.maskType;
        metadataArgs.blockDim = args.coreCount;
        metadataArgs.isVarlen = args.layout == FwdLayout::TND;
        metadataArgs.numSplits = args.splits;
        metadataArgs.hasSequsedQ = args.qUsed != 0;
        metadataArgs.quantMode = args.quantMode;
        metadataArgs.windowSizeLeft = left;
        metadataArgs.windowSizeRight = right;
        fa4_metadata_plan::Fill(&metadataArgs, &execution.tiling, allowFd);
        if (args.metadata != 0) {
            FAInferStaticTilingData supplied;
            std::memcpy(&supplied, reinterpret_cast<const void*>(args.metadata), sizeof(supplied));
            const auto& expected = execution.tiling;
            if (supplied.batch != expected.batch || supplied.numHeads != expected.numHeads ||
                supplied.kvHeads != expected.kvHeads || supplied.embeddingSize != expected.embeddingSize ||
                supplied.embeddingSizeV != expected.embeddingSizeV || supplied.quantMode != expected.quantMode ||
                supplied.maxQSeqlen != expected.maxQSeqlen || supplied.maxKvSeqlen != expected.maxKvSeqlen ||
                supplied.qBaseTile != expected.qBaseTile || supplied.firstBatchTaskNum != expected.firstBatchTaskNum ||
                supplied.totalTaskNum != expected.totalTaskNum || supplied.requestedNumSplits != args.splits ||
                (allowFd && std::memcmp(&supplied, &expected, sizeof(expected)) != 0)) {
                return FA4FwdStatus::InvalidMetadata;
            }
        }
    }
    runtime.allowFlashDecode = allowFd && execution.tiling.flashDecodeFlag != 0;
    if (!runtime.allowFlashDecode && execution.tiling.flashDecodeFlag != 0) {
        return FA4FwdStatus::InvalidSchedule;
    }
    execution.useDn = runtime.maskType == 0 && !runtime.allowFlashDecode && args.qDim <= 256 && args.vDim <= 256;
    if (execution.tiling.workSpaceSize > args.workspaceSize ||
        execution.tiling.fdActiveCoreNum > args.coreCount) {
        return FA4FwdStatus::InvalidSchedule;
    }
    return FA4FwdStatus::Valid;
}

} // namespace fa4_fwd_plan

#endif
