/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// This translation unit executes only on host/AICPU. Bisheng's ASC front end
// also defines __NPU_ARCH__ for host-only files; keep reused scalar consumers
// free of AICore annotations here.
#ifdef __NPU_ARCH__
#undef __NPU_ARCH__
#endif
#include "mla_split.h"
#include "cann_split_core.h"
#include <algorithm>
#include <new>
#include <stdexcept>
#include <utility>
#include <tuple>

// Source references are retained in mla_split.h.
namespace flash_attn::mla {
namespace {
bool TensorLayout(MlaAccumulatorTensor& tensor)
{
    uint64_t stride = 1, elements = 1;
    for (uint32_t i = tensor.rank; i-- > 0;) {
        tensor.strides[i] = stride;
        const uint64_t dimension = std::max(uint64_t(1), tensor.shape[i]);
        if (stride > uint64_t(INT64_MAX) / dimension) {
            return false;
        }
        stride *= dimension;
        elements *= tensor.shape[i]; // bounded by stride, including zero extents
    }
    if (elements > uint64_t(INT64_MAX) / sizeof(float)) {
        return false;
    }
    tensor.elements = elements;
    tensor.bytes = elements * sizeof(float);
    return true;
}
} // namespace

MlaScheduleStatus BuildSplitAccumulatorLayout(const MlaAccumulatorInput& in, MlaAccumulatorLayout& output)
{
    if ((in.layout != MlaAccumulatorQueryLayout::Bsnd && in.layout != MlaAccumulatorQueryLayout::Tnd) || in.batch < 0 ||
        in.queryCapacity < 0 || in.totalQuery < 0 || in.queryHeads < 1 || in.numSplits < 1 || in.numSplits > 256) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    if (uint64_t(in.batch) > UINT32_MAX || in.queryCapacity > kMlaMaxQueryLength || in.queryHeads > kMlaMaxQueryHeads ||
        (in.layout == MlaAccumulatorQueryLayout::Tnd && in.totalQuery > INT32_MAX)) {
        return MlaScheduleStatus::FieldOverflow;
    }
    if (in.layout == MlaAccumulatorQueryLayout::Bsnd &&
        uint64_t(in.totalQuery) != uint64_t(in.batch) * in.queryCapacity) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    MlaAccumulatorLayout result{};
    if (in.numSplits > 1) {
        auto& o = result.out;
        auto& l = result.lse;
        if (in.layout == MlaAccumulatorQueryLayout::Bsnd) {
            o.rank = 5;
            o.shape = {uint64_t(in.numSplits), uint64_t(in.batch), uint64_t(in.queryHeads), uint64_t(in.queryCapacity),
                       512};
        } else {
            o.rank = 4;
            o.shape = {uint64_t(in.numSplits), uint64_t(in.queryHeads), uint64_t(in.totalQuery), 512, 0};
        }
        l.rank = o.rank - 1;
        std::copy_n(o.shape.begin(), l.rank, l.shape.begin());
        if (!TensorLayout(o) || !TensorLayout(l)) {
            return MlaScheduleStatus::WorkspaceOverflow;
        }
    }
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus ResolveGqaResources(int64_t heads, int64_t kvHeads, const MlaGqaOptions& o, const MlaPlatformInfo& p,
                                      MlaGqaResources& output)
{
    if (heads < 1 || kvHeads < 1 || heads % kvHeads || o.packGqa < -1 || o.packGqa > 1)
        return MlaScheduleStatus::UnsupportedShape;
    if (heads > kMlaMaxQueryHeads)
        return MlaScheduleStatus::FieldOverflow;
    if (!p.aicNum || p.aicNum > kMlaMaxAic || p.aivNum != 2 * p.aicNum || o.smMargin < 0 || o.smMargin >= p.aicNum)
        return MlaScheduleStatus::InvalidPlatform;
    MlaGqaResources r{};
    r.platform = p;
    r.platform.aicNum -= uint32_t(o.smMargin);
    r.platform.aivNum = 2 * r.platform.aicNum;
    for (uint32_t i = 0; i < kMlaMaxAic; ++i) {
        if (i >= r.platform.aicNum)
            r.platform.blockDims[i] = 0;
        else if (!p.blockDims[i])
            return MlaScheduleStatus::InvalidPlatform;
    }
    r.packed = o.packGqa != 0 && heads != kvHeads;
    r.headGroups = r.packed ? kvHeads : heads;
    r.headsPerGroup = r.packed ? heads / kvHeads : 1;
    output = r;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus BuildGqaSchedule(const MlaGqaConfig& c, const MlaUsedLengths& lengths, const MlaWindowConfig& window,
                                   const MlaGqaOptions& options, const MlaPlatformInfo& physical,
                                   MlaNativeSplitKvSchedule& output)
{
    MlaNativeSplitKvSchedule native{};
    MlaGqaResources resources{};
    MlaWindow normalizedWindow{};
    auto status = ResolveGqaResources(c.shape.queryHeads, c.kvHeads, options, physical, resources);
    if (status != MlaScheduleStatus::Success)
        return status;
    if ((c.valueDim != 128 && c.valueDim != 512) || (c.valueDim == 512 && c.kvHeads != 1) || c.mBase < 32 ||
        c.mBase > 512 || (c.valueDim == 128 && (c.mBase & (c.mBase - 1))) || c.nBase < 1 || c.nBase > 8192 ||
        window.maxQueryLength != c.shape.query.maxLength || window.maxKvLength != c.shape.kv.maxLength ||
        c.pageSize < 0 ||
        (c.pageSize && (c.pageSize < 16 || c.pageSize > 1024 || c.pageSize % 16 ||
                        c.shape.kv.layout != MlaSequenceLayout::Bsnd || c.shape.kv.maxLength % c.pageSize)))
        return MlaScheduleStatus::UnsupportedShape;
    if (c.shape.batch < 0 || uint64_t(c.shape.batch) > UINT32_MAX / resources.headGroups)
        return MlaScheduleStatus::FieldOverflow;
    const auto normalized = NormalizeWindow(window, normalizedWindow);
    if (normalized != MlaCausalStatus::Success)
        return normalized == MlaCausalStatus::FieldOverflow ? MlaScheduleStatus::FieldOverflow
                                                            : MlaScheduleStatus::UnsupportedShape;
    // Reuse the existing public allocation/stride guard, including empty
    // tensors. Its 512-wide envelope also bounds the 128-wide representation.
    MlaAccumulatorLayout accumulator{};
    status = BuildSplitAccumulatorLayout(
        {c.shape.query.layout == MlaSequenceLayout::Tnd ? MlaAccumulatorQueryLayout::Tnd
                                                        : MlaAccumulatorQueryLayout::Bsnd,
         c.shape.batch, c.shape.query.maxLength, c.shape.query.totalTokens, c.shape.queryHeads, kMlaMaxAic},
        accumulator);
    if (status != MlaScheduleStatus::Success)
        return status;
    try {
        detail::BaseInfo base{};
        base.bSize = c.shape.batch;
        base.n2Size = resources.headGroups;
        base.gSize = resources.headsPerGroup;
        base.isS1G = true;
        base.s1Size = c.shape.query.maxLength;
        base.s2Size = c.shape.kv.maxLength;
        base.actualLenQDims = base.actualLenKvDims = c.shape.batch;
        base.attenMaskFlag = normalizedWindow.kind != MlaWindowKind::None;
        base.sparseMode = normalizedWindow.kind == MlaWindowKind::Local ? 4 : base.attenMaskFlag ? 3 : 0;
        base.preToken = base.sparseMode == 3 ? INT32_MAX : normalizedWindow.left;
        base.nextToken = base.sparseMode == 3 ? 0 : normalizedWindow.right;
        uint64_t tiles = 0;
        // Check source uint32 products and ceil-div BEFORE allocating prefixes.
        for (const auto& s : lengths.sequences) {
            const uint64_t rows = uint64_t(s.queryLength) * base.gSize;
            if (rows > uint64_t(UINT32_MAX) - (c.mBase - 1))
                return MlaScheduleStatus::FieldOverflow;
            const uint64_t groups = (rows + c.mBase - 1) / c.mBase;
            const uint64_t count = groups * ((uint64_t(s.kvLength) + c.nBase - 1) / c.nBase);
            if (count > (UINT32_MAX - tiles) / base.n2Size)
                return MlaScheduleStatus::FieldOverflow;
            tiles += count * base.n2Size;
        }
        uint64_t totalGroups = 0;
        for (const auto& s : lengths.sequences) {
            base.actualSeqS1Size.push_back(s.queryLength);
            base.actualSeqS2Size.push_back(s.kvLength);
            const uint64_t groups = (uint64_t(s.queryLength) * base.gSize + c.mBase - 1) / c.mBase;
            totalGroups += groups * base.n2Size;
        }
        {
            detail::SplitParam param{};
            param.mBaseSize = c.mBase;
            param.s2BaseSize = c.nBase;
            param.gS1BaseSizeOfFd = 8;
            detail::SplitContext context(base, param);
            detail::CalcSplitInfo(context);
            detail::CalcCostInfo(context);
            detail::SplitResult n{resources.platform.aicNum, 2};
            detail::SplitCore(resources.platform.aicNum, context, n);
            native.usedCoreNum = n.usedCoreNum;
            native.maxCost = n.maxCost;
            native.totalCost = context.costInfo.totalCost;
            native.totalMBlocks = totalGroups;
            native.totalS2Blocks = context.costInfo.totalBlockNum;
            std::copy(n.bN2End.begin(), n.bN2End.end(), native.bN2End.begin());
            std::copy(n.gS1End.begin(), n.gS1End.end(), native.gS1End.begin());
            std::copy(n.s2End.begin(), n.s2End.end(), native.s2End.begin());
            auto& fd = native.fd;
            fd.numOfFdHead = n.numOfFdHead;
            fd.maxS2SplitNum = n.maxS2SplitNum;
            fd.usedVecNumOfFd = n.usedVecNumOfFd;
            fd.gS1BaseSizeOfFd = 8;
#define COPY_GQA_FD(field) std::copy(n.fdRes.field.begin(), n.fdRes.field.end(), fd.field.begin())
            COPY_GQA_FD(bN2IdxOfFdHead);
            COPY_GQA_FD(gS1IdxOfFdHead);
            COPY_GQA_FD(s2SplitNumOfFdHead);
            COPY_GQA_FD(s2SplitStartIdxOfCore);
            COPY_GQA_FD(gS1SplitNumOfFdHead);
            COPY_GQA_FD(gS1LastPartSizeOfFdHead);
            COPY_GQA_FD(gS1IdxEndOfFdHead);
            COPY_GQA_FD(gS1IdxEndOfFdHeadSplit);
#undef COPY_GQA_FD
            if (!n.usedCoreNum || n.usedCoreNum > resources.platform.aicNum ||
                fd.numOfFdHead > resources.platform.aicNum || fd.maxS2SplitNum > resources.platform.aicNum ||
                fd.usedVecNumOfFd > 2 * n.usedCoreNum)
                return MlaScheduleStatus::InvalidSplit;
        }
        output = native;
        return MlaScheduleStatus::Success;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    } catch (const std::length_error&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}

} // namespace flash_attn::mla
