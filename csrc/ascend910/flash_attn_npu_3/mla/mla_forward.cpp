/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#include "mla_layout.h"
#include "mla_metadata.h"
#include "mla_forward.h"
#include "kernel_operator.h"
#include "mla_dispatch.h"
#include "mla_metadata_consumer.h"
#include "mla_split.h"
#include "mla_tiling.h"
#include "mla_validation.h"
#include "../device_triu_mask.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "runtime/rt_ffts.h"
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch/version.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <tuple>
#include <utility>

extern __global__ __aicpu__ uint32_t ComputeMlaMetadata(void* args);

namespace flash_attn::mla {
namespace {
template <typename Launch, typename Owners>
void RunMlaLaunch(const char* name, Launch&& launch, Owners&& owners)
{
    auto keepAlive = owners();
    at_npu::native::OpCommand::RunOpApiV2(
        name, [launch = std::forward<Launch>(launch), keepAlive = std::move(keepAlive)]() mutable { return launch(); });
}

MlaPlatformInfo DensePlatform()
{
    const char* soc = aclrtGetSocName();
    // CANN names the 910C family Ascend910_93; both families use the 2201
    // kernels. Keep resource validation below for each actual device/partition.
    TORCH_CHECK(soc && (std::strncmp(soc, "Ascend910B", 10) == 0 || std::strncmp(soc, "Ascend910_93", 12) == 0),
                "MLA forward requires an Ascend 910B/C device");
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform, "MLA: platform information is unavailable");
    MlaPlatformInfo p{};
    p.aicNum = platform->GetCoreNumAic();
    p.aivNum = platform->GetCoreNumAiv();
    TORCH_CHECK(p.aicNum > 0 && p.aicNum <= kMlaMaxAic && p.aivNum == 2 * p.aicNum,
                "MLA: this dense launch requires validated C1V2 resources");
    using Mem = platform_ascendc::CoreMemType;
    platform->GetCoreMemSize(Mem::L1, p.localMemory.l1Bytes);
    platform->GetCoreMemSize(Mem::UB, p.localMemory.ubBytes);
    platform->GetCoreMemSize(Mem::L0_A, p.localMemory.l0aBytes);
    platform->GetCoreMemSize(Mem::L0_B, p.localMemory.l0bBytes);
    platform->GetCoreMemSize(Mem::L0_C, p.localMemory.l0cBytes);
    p.libApiWorkspaceBytes = platform->GetLibApiWorkSpaceSize();
    for (uint32_t c = 1; c <= p.aicNum; ++c) {
        p.blockDims[c - 1] = platform->CalcTschBlockDim(c * 2, c, c * 2);
    }
    MlaSchedule resourceCheck{};
    TORCH_CHECK(BuildDenseSchedule({1, 1, 1, 1}, p, resourceCheck) == MlaScheduleStatus::Success,
                "MLA: invalid C1V2 resources");
    return p;
}

const MlaPlatformInfo& DevicePlatform(c10::DeviceIndex device)
{
    // These chip resources are immutable. Keep each host thread/device's
    // validated descriptor separate; no tensor, length or schedule is cached.
    static thread_local std::unordered_map<c10::DeviceIndex, MlaPlatformInfo> platforms;
    auto found = platforms.find(device);
    if (found == platforms.end()) {
        found = platforms.emplace(device, DensePlatform()).first;
    }
    return found->second;
}

template <typename View = MlaBsndTensorView>
View PagedTensorView(const at::Tensor& t)
{
    View view{};
    for (int64_t i = 0; i < t.dim(); ++i) {
        view.sizes[i] = t.size(i);
        view.strides[i] = t.stride(i);
    }
    view.storageOffset = t.storage_offset();
    view.storageBytes = t.storage().nbytes();
    return view;
}

// Validate the entire accessible span, not numel() for a strided view. Empty
// descriptors have no address requirement; Q-used presence is restored below.
MlaPagedMetadataArray PagedTensorArray(const at::Tensor& t, uint64_t span)
{
    if (span == 0) {
        return {};
    }
    const uint64_t elements = t.storage().nbytes() / t.element_size();
    TORCH_CHECK(t.storage_offset() >= 0 && uint64_t(t.storage_offset()) <= elements &&
                    span <= elements - uint64_t(t.storage_offset()),
                "MLA: paged tensor view exceeds storage bounds");
    const auto address = reinterpret_cast<uintptr_t>(t.data_ptr());
    TORCH_CHECK(address != 0 && address % t.element_size() == 0 && span <= (UINTPTR_MAX - address) / t.element_size(),
                "MLA: tensor address overflow");
    return {address, span};
}

const MlaPlatformInfo& ShapePlatform()
{
    static const MlaPlatformInfo p = [] {
        MlaPlatformInfo result{};
        result.aicNum = kMlaMaxAic;
        result.aivNum = 2 * result.aicNum;
        result.localMemory = {512 * 1024, 256 * 1024, 64 * 1024, 64 * 1024, 128 * 1024};
        for (uint32_t i = 0; i < result.aicNum; ++i)
            result.blockDims[i] = i + 1;
        MlaSchedule resourceCheck{};
        TORCH_CHECK(BuildDenseSchedule({1, 1, 1, 1}, result, resourceCheck) == MlaScheduleStatus::Success,
                    "MLA: invalid C1V2 resources");
        return result;
    }();
    return p;
}

uint64_t TensorAddress(const at::Tensor& tensor)
{
    return reinterpret_cast<uint64_t>(tensor.data_ptr());
}
void RecordInput(const at::Tensor& tensor, const c10_npu::NPUStream& stream)
{
    if (tensor.numel())
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
}
} // namespace

namespace {
MlaContext BuildContext(int64_t batch, int64_t sq, int64_t sk, int64_t heads, int64_t kvHeads, int64_t valueDim,
                        at::ScalarType dtype, int64_t splits, int64_t left, int64_t right, bool causal, uint32_t flags,
                        const MlaPlatformInfo& platform, uint32_t layout, int64_t totalQ, int64_t totalK,
                        int64_t pageSize, bool hasSoftcap, std::optional<bool> pack, int64_t margin, MlaLayout& l)
{
    TORCH_CHECK(FitsDenseShape(0, sq, heads, sk) && batch >= 0 && uint64_t(batch) <= UINT32_MAX && kvHeads > 0 &&
                    heads % kvHeads == 0 && splits >= 0 && splits <= 256 && totalQ >= 0 && totalQ <= INT32_MAX &&
                    totalK >= 0 && totalK <= INT32_MAX && pageSize >= 0 && pageSize <= 1024 &&
                    (valueDim == 128 || valueDim == 512),
                "MLA: shape/split bounds exceeded");
    MlaGqaResources resources{};
    TORCH_CHECK(ResolveGqaResources(heads, kvHeads, {pack ? int64_t(*pack) : -1, margin}, platform, resources) ==
                    MlaScheduleStatus::Success,
                "MLA: invalid pack_gqa or sm_margin; sm_margin must leave at least one physical AIC");
    TORCH_CHECK(resources.platform.blockDims[resources.platform.aicNum - 1] == resources.platform.aicNum,
                "MLA: unsupported C1V2 blockDim mapping");
    MlaWindow window{};
    TORCH_CHECK(NormalizeWindow({sq, sk, left, right, causal}, window) == MlaCausalStatus::Success,
                "MLA: window bounds exceeded");
    MlaContext c{uint32_t(batch),
                 uint32_t(sq),
                 uint32_t(sk),
                 uint32_t(heads),
                 uint32_t(kvHeads),
                 uint32_t(splits),
                 uint32_t(dtype == at::kHalf ? MlaMetadataDtype::Fp16 : MlaMetadataDtype::Bf16),
                 platform.aicNum,
                 uint32_t(margin),
                 uint32_t(resources.packed),
                 layout,
                 uint32_t(totalQ),
                 uint32_t(totalK),
                 uint32_t(pageSize),
                 flags,
                 uint32_t(valueDim),
                 uint32_t(hasSoftcap),
                 {uint32_t(window.kind), window.left, window.right}};
    TORCH_CHECK(MlaRecordLayout(c, l), "MLA: metadata shape bounds or workspace/accumulator overflow");
    TORCH_CHECK(l.bytes <= uint64_t(INT64_MAX) - platform.libApiWorkspaceBytes, "MLA: workspace overflow");
    // Dynamic CANN M can be 128..256. Bound storage for every tile without
    // copying schedules or synchronizing allocation/used lengths to the host.
    if (valueDim == 512) {
        MlaLayout small{};
        TORCH_CHECK(MlaRecordLayout(c, small, 128, l.n), "MLA: workspace bounds exceeded");
        const uint64_t rows = small.groups * small.capacity * 256;
        const bool direct = splits == 1 && MlaNativeCompressed(c);
        const uint64_t reserve = MlaNativeCompressed(c) && !splits ? 2ULL * platform.aicNum * 256 : rows;
        const uint64_t bound = direct ? l.scratch : l.outOffset + reserve * (512 + 16) * 4;
        TORCH_CHECK(bound <= uint64_t(INT64_MAX) - platform.libApiWorkspaceBytes, "MLA: workspace overflow");
        if (bound > l.bytes)
            l.bytes = bound;
    }
    c.workspaceBytes = l.bytes;
    return c;
}

MlaStaticPrefillPlan BuildStaticPrefillPlan(const MlaContext& c, const MlaLayout& l, const MlaPlatformInfo& platform)
{
    MlaStaticPrefillPlan plan{};
    // For a single TND segment, allocation geometry fixes its only legal
    // prefix. The device entry still checks both mutable prefixes each call.
    if (c.valueDim != 128 || c.batch != 1 || c.layout != 1 || c.heads != c.kvHeads || c.splits > 1 || c.usedFlags ||
        c.pageSize || c.margin || c.packed || c.hasSoftcap || !c.totalQ || !c.totalK || c.sq != c.totalQ ||
        c.sk != c.totalK)
        return plan;
    const MlaUsedShape shape{
        1, c.heads, {MlaSequenceLayout::Tnd, c.totalQ, c.sq}, {MlaSequenceLayout::Tnd, c.totalK, c.sk}};
    const int32_t cuQ[] = {0, int32_t(c.totalQ)}, cuK[] = {0, int32_t(c.totalK)};
    const MlaUsedInputs inputs{{cuQ, 2}, {cuK, 2}, {}, {}};
    MlaUsedLengths lengths{};
    if (ParseUsedLengths(shape, inputs, lengths) != MlaScheduleStatus::Success)
        return plan;
    const MlaWindowConfig window{c.sq, c.sk, c.window.kind == 1 ? -1 : c.window.left,
                                 c.window.kind == 1 ? -1 : c.window.right, c.window.kind == 1};
    MlaWindow normalized{};
    if (NormalizeWindow(window, normalized) != MlaCausalStatus::Success || uint32_t(normalized.kind) != c.window.kind ||
        normalized.left != c.window.left || normalized.right != c.window.right || l.bytes > c.workspaceBytes)
        return plan;
    MlaNativeSplitKvSchedule native{};
    // Run the same auto-split planner as the metadata producer. Only its whole-M result can use this path.
    if (BuildGqaSchedule({shape, c.kvHeads, c.valueDim, l.m, l.n, 0, c.splits == 0}, lengths, window,
                         {c.packed, c.margin}, platform, native) != MlaScheduleStatus::Success ||
        !native.usedCoreNum || native.usedCoreNum > MlaStaticLaunchCores(c) || native.fd.numOfFdHead ||
        native.fd.maxS2SplitNum || native.fd.usedVecNumOfFd)
        return plan;
    for (uint32_t i = 0; i < kMlaMaxAic; ++i) {
        if (native.s2End[i])
            return {};
        plan.bEnd[i] = native.bN2End[i];
        plan.mEnd[i] = native.gS1End[i];
    }
    plan.layout = l;
    plan.usedCoreNum = native.usedCoreNum;
    return plan;
}

MlaDynamicDecodePlan BuildDynamicDecodePlan(const MlaContext& c, const MlaLayout& l, const MlaPlatformInfo& platform)
{
    MlaDynamicDecodePlan plan{};
    const bool queryOne = c.sq == 1 && !c.window.kind;
    const bool querySixteen = c.sq == 16 && c.layout == 3 && c.pageSize == 128 && c.physicalCores == 20 &&
                              c.window.kind == 1 && c.window.left == 8191 && c.window.right == 0;
    if (c.valueDim != 512 || c.batch != 1 || (!queryOne && !querySixteen) || c.sk != 8192 || c.heads != 128 ||
        c.kvHeads != 1 || c.splits || c.margin || !c.packed || c.hasSoftcap || (c.usedFlags & ~2U) ||
        (c.layout != 0 && c.layout != 3) || (c.layout == 3 && c.pageSize != 128) || platform.aicNum < 16 ||
        l.m != 256 || l.n != 512)
        return plan;
    for (uint32_t i = 0; i < 4; ++i) {
        if (!MlaRecordLayout(c, plan.layouts[i], 256, 128 * (i + 1)) || plan.layouts[i].bytes > c.workspaceBytes)
            return {};
    }
    if (querySixteen) {
        plan.enabled = 2;
        return plan;
    }
    // These are schedules for hypothetical scalar lengths, never a snapshot
    // of a device tensor. Every core reads and validates the current length.
    const MlaUsedShape shape{1, 128, {MlaSequenceLayout::Bsnd, 1, 1}, {MlaSequenceLayout::Bsnd, 8192, 8192}};
    const MlaWindowConfig window{c.sq, c.sk, c.window.left, c.window.right, false};
    for (uint32_t blocks = 0; blocks <= 16; ++blocks) {
        const int32_t used = blocks * 512;
        MlaUsedLengths lengths{};
        MlaNativeSplitKvSchedule native{};
        if (ParseUsedLengths(shape, {{}, {}, {}, {&used, 1}}, lengths) != MlaScheduleStatus::Success ||
            BuildGqaSchedule({shape, 1, 512, 256, 512, c.pageSize, true}, lengths, window, {c.packed, c.margin},
                             platform, native) != MlaScheduleStatus::Success)
            return {};
        const uint32_t cores = blocks ? blocks : 1;
        const bool fd = blocks > 1;
        if (native.usedCoreNum != cores || native.fd.gS1BaseSizeOfFd != 8 || native.fd.numOfFdHead != uint32_t(fd) ||
            native.fd.maxS2SplitNum != (fd ? blocks : 0) || native.fd.usedVecNumOfFd > 16 ||
            (!fd && native.fd.usedVecNumOfFd))
            return {};
        for (uint32_t i = 0; i < kMlaMaxAic; ++i) {
            if (native.bN2End[i] != uint32_t(i + 1 == cores) || native.gS1End[i] ||
                native.s2End[i] != (i + 1 < cores ? i + 1 : 0) || native.fd.bN2IdxOfFdHead[i] ||
                native.fd.gS1IdxOfFdHead[i] || native.fd.s2SplitNumOfFdHead[i] != (fd && i == 0 ? blocks : 0) ||
                native.fd.s2SplitStartIdxOfCore[i] != (fd && i < cores ? i : 0) ||
                native.fd.gS1SplitNumOfFdHead[i] != (fd && i == 0 ? 16 : 0) ||
                native.fd.gS1LastPartSizeOfFdHead[i] != (fd && i == 0 ? 8 : 0))
                return {};
        }
        for (uint32_t i = 0; i < 2 * kMlaMaxAic; ++i) {
            const uint32_t part = native.fd.gS1IdxEndOfFdHeadSplit[i];
            if (native.fd.gS1IdxEndOfFdHead[i] || part > 15 || (i >= 16 && part))
                return {};
            if (i < 16)
                plan.vectorEnds[blocks] |= uint64_t(part) << (4 * i);
        }
        plan.vectors[blocks] = native.fd.usedVecNumOfFd;
    }
    plan.enabled = 1;
    return plan;
}

struct DenseScheduleCache {
    aclrtStream stream = nullptr;
    at::Tensor record;
};

MlaContext MakeContext(int64_t batch, int64_t sq, int64_t sk, int64_t heads, int64_t kvHeads, int64_t valueDim,
                       at::ScalarType dtype, int64_t splits, int64_t left, int64_t right, bool causal, uint32_t flags,
                       const MlaPlatformInfo& platform, uint32_t layout, int64_t totalQ, int64_t totalK,
                       int64_t pageSize, bool hasSoftcap, std::optional<bool> pack, int64_t margin, c10::Device device,
                       MlaLayout& l, MlaStaticPrefillPlan* staticPrefill = nullptr,
                       std::shared_ptr<const MlaDynamicDecodePlan>* dynamicDecode = nullptr,
                       DenseScheduleCache** denseSchedule = nullptr)
{
    // Shape plans and immutable, internally generated dense records are cached.
    // Caller-owned tensors and their storage bounds are bound on each invocation.
    struct Plan {
        const MlaPlatformInfo* platform = nullptr;
        std::array<int64_t, 22> key{};
        MlaContext context{};
        MlaLayout layout{};
        MlaStaticPrefillPlan staticPrefill{};
        std::shared_ptr<const MlaDynamicDecodePlan> dynamicDecode;
        DenseScheduleCache denseSchedule;
    };
    static thread_local std::array<Plan, 4> plans{};
    static thread_local uint32_t last = 0, next = 0;
    const bool prepareLaunchPlans = (staticPrefill || dynamicDecode) && device.type() != c10::DeviceType::Meta;
    const std::array<int64_t, 22> key{int64_t(device.type()),
                                      device.index(),
                                      batch,
                                      sq,
                                      sk,
                                      heads,
                                      kvHeads,
                                      valueDim,
                                      int64_t(dtype),
                                      splits,
                                      left,
                                      right,
                                      causal,
                                      flags,
                                      layout,
                                      totalQ,
                                      totalK,
                                      pageSize,
                                      hasSoftcap,
                                      pack ? int64_t(*pack) : -1,
                                      margin,
                                      prepareLaunchPlans};
    for (uint32_t i = 0; i < plans.size(); ++i) {
        const uint32_t index = (last + i) % plans.size();
        auto& plan = plans[index];
        // Platform objects are immutable and have stable addresses in the
        // per-thread device map or the Meta singleton.
        if (plan.platform == &platform && plan.key == key) {
            last = index;
            l = plan.layout;
            if (staticPrefill)
                *staticPrefill = plan.staticPrefill;
            if (dynamicDecode)
                *dynamicDecode = plan.dynamicDecode;
            if (denseSchedule)
                *denseSchedule = &plan.denseSchedule;
            return plan.context;
        }
    }
    const auto context = BuildContext(batch, sq, sk, heads, kvHeads, valueDim, dtype, splits, left, right, causal,
                                      flags, platform, layout, totalQ, totalK, pageSize, hasSoftcap, pack, margin, l);
    const auto prefill = prepareLaunchPlans ? BuildStaticPrefillPlan(context, l, platform) : MlaStaticPrefillPlan{};
    std::shared_ptr<const MlaDynamicDecodePlan> decode;
    if (prepareLaunchPlans) {
        auto plan = BuildDynamicDecodePlan(context, l, platform);
        if (plan.enabled)
            decode = std::make_shared<const MlaDynamicDecodePlan>(std::move(plan));
    }
    plans[next] = Plan{&platform, key, context, l, prefill, decode, {}};
    if (staticPrefill)
        *staticPrefill = prefill;
    if (dynamicDecode)
        *dynamicDecode = decode;
    if (denseSchedule)
        *denseSchedule = &plans[next].denseSchedule;
    last = next;
    next = (next + 1) % plans.size();
    return context;
}
} // namespace

std::vector<at::Tensor> Forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                const std::optional<at::Tensor>& qv, int64_t splits, int64_t left, int64_t right,
                                bool causal, float scale, const std::optional<at::Tensor>& usedQ,
                                const std::optional<at::Tensor>& usedK, const std::optional<at::Tensor>& supplied,
                                const std::optional<at::Tensor>& cuQ, const std::optional<at::Tensor>& cuK,
                                int64_t maxQ, int64_t maxK, const std::optional<at::Tensor>& pageTable, float softcap,
                                std::optional<bool> pack, int64_t margin, const MlaBsndInputLayouts* validatedInputs,
                                bool returnLse, bool outputOnly)
{
    const bool tnd = bool(cuQ);
    TORCH_CHECK(!supplied || !qv || !cuK || margin || (pack && !*pack) || splits != 1 || softcap > 0,
                "MLA: compact TND explicit metadata is unsupported");
    TORCH_CHECK(!usedQ || !supplied || (tnd && (!qv || ((margin || (pack && !*pack)) && cuK))),
                "MLA: Q-used with explicit metadata is not implemented yet");
    if (cuK)
        CheckTndInputs(q, k, *cuQ, *cuK, maxQ, maxK);
    else if (tnd && !pageTable)
        CheckTndBsndForwardInputs(q, k, *cuQ, maxQ, usedK, std::nullopt);
    else if (tnd) {
        TORCH_CHECK(cuQ->dim() == 1 && cuQ->numel() >= 1, "MLA: cu_seqlens_q must have shape [batch_size + 1]");
        CheckMetadataQueryPrefix(*cuQ, cuQ->numel() - 1, q.device());
    }
    const int64_t batch = tnd ? cuQ->numel() - 1 : q.size(0);
    CheckUsedInputs(q, batch, usedQ, usedK);
    MlaPagedCacheLayout paged{};
    if (pageTable) {
        TORCH_CHECK(pageTable->dim() == 2 && pageTable->size(0) == batch && pageTable->scalar_type() == at::kInt &&
                        pageTable->device() == q.device(),
                    "MLA: page_table must be same-device int32[batch_size, max_pages]");
        TORCH_CHECK(BuildPagedCacheLayout(PagedTensorView<MlaPageTableView>(*pageTable), PagedTensorView(k),
                                          PagedTensorView(v), paged, !qv) == MlaLayoutStatus::Success,
                    "MLA: paged cache exceeds shape, stride, storage or signed byte bounds");
    }
    const int64_t kvCapacity = pageTable ? int64_t(paged.logicalCapacity) : cuK ? maxK : k.size(1);
    const auto currentK = !cuK && supplied && !usedK
                              ? std::optional<at::Tensor>(at::full({batch}, kvCapacity, q.options().dtype(at::kInt)))
                              : usedK;
    const auto& platform = q.is_meta() ? ShapePlatform() : DevicePlatform(q.device().index());
    MlaLayout l{};
    MlaStaticPrefillPlan staticPrefill{};
    std::shared_ptr<const MlaDynamicDecodePlan> dynamicDecode;
    DenseScheduleCache* denseSchedule = nullptr;
    const auto c = MakeContext(
        batch, tnd ? maxQ : q.size(1), kvCapacity, q.size(-2), k.size(-2), v.size(-1), q.scalar_type(), splits, left,
        right, causal, uint32_t(bool(usedQ)) | (uint32_t(bool(currentK)) << 1), platform,
        pageTable ? (tnd ? 4 : 3)
        : cuK     ? 1
        : tnd     ? 2
                  : 0,
        tnd ? q.size(0) : 0, cuK ? k.size(0) : 0, pageTable ? k.size(1) : 0, softcap > 0, pack, margin, q.device(), l,
        supplied ? nullptr : &staticPrefill, supplied ? nullptr : &dynamicDecode, supplied ? nullptr : &denseSchedule);
    if (supplied)
        CheckMetadataTensor(*supplied, q.device(), l.words, "MLA: scheduler_metadata");
    const bool checkRecordOnDevice =
        (supplied && c.valueDim == 512 && c.batch == 1 && c.sq == 1 && c.sk == 8192 && c.heads == 128 &&
         c.kvHeads == 1 && !c.splits && !c.margin && c.packed && !c.hasSoftcap && !c.window.kind && c.usedFlags == 2 &&
         c.physicalCores >= 16 && (c.layout == 0 || (c.layout == 3 && c.pageSize == 128))) ||
        (supplied && c.valueDim == 512 && c.batch == 1 && c.sq == 16 && c.sk == 8192 && c.heads == 128 &&
         c.kvHeads == 1 && !c.splits && !c.margin && c.packed && !c.hasSoftcap && c.usedFlags == 2 &&
         c.physicalCores == 20 && c.layout == 3 && c.pageSize == 128 && c.window.kind == 1 && c.window.left == 8191 &&
         c.window.right == 0);
    const uint64_t checkedOffset = (l.bytes + 63) / 64 * 64;
    const uint64_t metadataOffset = checkedOffset + sizeof(MlaCheckedAutoSplitSchedule);
    const uint64_t metadataBytes = supplied ? 0 : uint64_t(l.words) * sizeof(uint32_t);
    TORCH_CHECK(platform.libApiWorkspaceBytes <= uint64_t(INT64_MAX) &&
                    metadataOffset <= uint64_t(INT64_MAX) - platform.libApiWorkspaceBytes &&
                    metadataBytes <= uint64_t(INT64_MAX) - platform.libApiWorkspaceBytes - metadataOffset,
                "MLA: metadata workspace overflow");
    const int64_t b = c.batch, sq = c.sq, h = c.heads, dim = c.valueDim, capacity = splits ? splits : 26;
    const auto allocate = [meta = q.is_meta()](at::IntArrayRef sizes, const at::TensorOptions& options) {
        if (meta)
            return at::empty(sizes, options);
        // Match aten::empty's base format without entering its dispatcher.
        const int64_t format = sizes.size() == 4   ? ACL_FORMAT_NCHW
                               : sizes.size() == 5 ? ACL_FORMAT_NCDHW
                                                   : ACL_FORMAT_ND;
        return at_npu::native::empty_with_format(sizes, options, format, true);
    };
    auto out = tnd ? allocate({c.totalQ, h, dim}, q.options()) : allocate({b, sq, h, dim}, q.options());
    const auto emptyOptions = at::TensorOptions().device(q.device());
    const auto floatOptions = q.options().dtype(at::kFloat);
    const bool returnPartials = returnLse && splits != 1;
    at::Tensor lse, accum, accumLse;
    if (!outputOnly) {
        lse = !returnLse ? allocate({0}, floatOptions)
              : tnd      ? allocate({h, c.totalQ}, floatOptions)
                         : allocate({b, h, sq}, floatOptions);
        accum = !returnPartials ? allocate({0}, emptyOptions)
                : tnd           ? allocate({capacity, h, c.totalQ, dim}, floatOptions)
                                : allocate({capacity, b, h, sq, dim}, floatOptions);
        accumLse = !returnPartials ? allocate({0}, emptyOptions)
                   : tnd           ? allocate({capacity, h, c.totalQ}, floatOptions)
                                   : allocate({capacity, b, h, sq}, floatOptions);
    }
    if (q.is_meta())
        return {out, lse, accum, accumLse};
    const char* noCachingEnv = std::getenv("PYTORCH_NO_NPU_MEMORY_CACHING");
    const bool noCaching = noCachingEnv && std::strtol(noCachingEnv, nullptr, 10) != 0;
    // With the uncached allocator, keep the mask in this launch's owners so
    // aclrtFree runs before the runtime and its diagnostic hooks shut down.
    const auto mask = c.window.kind ? MakeDeviceTriuMask(!noCaching) : at::Tensor{};
    const uint64_t workspaceBytes = staticPrefill.usedCoreNum ? checkedOffset : metadataOffset + metadataBytes;
    auto workspace = allocate({int64_t(workspaceBytes + platform.libApiWorkspaceBytes)}, q.options().dtype(at::kByte));
    const uint64_t workspaceAddress = TensorAddress(workspace) + platform.libApiWorkspaceBytes;
    const uint64_t metadataAddress = staticPrefill.usedCoreNum ? 0
                                     : supplied                ? TensorAddress(*supplied)
                                                               : workspaceAddress + metadataOffset;
    MlaRecordArgs args{c,
                       cuQ ? TensorAddress(*cuQ) : 0,
                       cuK ? TensorAddress(*cuK) : 0,
                       usedQ ? TensorAddress(*usedQ) : 0,
                       currentK ? TensorAddress(*currentK) : 0,
                       metadataAddress,
                       l.words};
    auto stream = c10_npu::getCurrentNPUStream();
    auto* allocator = c10_npu::NPUCachingAllocator::get();
    bool retainLocalBuffers = noCaching || allocator->name() != "native" ||
                              workspace.storage().data_ptr().get_deleter() != allocator->raw_deleter();
#if __has_include("torch_npu/csrc/core/npu/NPUGraphsUtils.h")
    retainLocalBuffers |= c10_npu::currentStreamCaptureStatusMayInitCtx() != c10_npu::CaptureStatus::None;
#endif
    // Inputs can come from another stream. Local buffers are allocated on the
    // launch stream, whose task queue orders their reuse after this request.
    for (const auto& tensor : {q, k, v})
        RecordInput(tensor, stream);
    if (supplied)
        RecordInput(*supplied, stream);
    if (mask.defined())
        RecordInput(mask, stream);
    for (const auto& tensor : {qv, cuQ, cuK, usedQ, usedK, pageTable})
        if (tensor)
            RecordInput(*tensor, stream);
    MlaLaunchArgs a{};
    a.stream = stream.stream(false);
    uint32_t fftsLength = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&a.ffts, &fftsLength) == RT_ERROR_NONE, "MLA: FFTS address unavailable");
    a.q = TensorAddress(q);
    a.qv = qv ? TensorAddress(*qv) : 0;
    a.k = TensorAddress(k);
    a.v = TensorAddress(v);
    a.out = TensorAddress(out);
    a.lse = returnLse ? TensorAddress(lse) : 0;
    a.accum = returnPartials ? TensorAddress(accum) : 0;
    a.accumLse = returnPartials ? TensorAddress(accumLse) : 0;
    a.workspace = workspaceAddress;
    a.metadata = metadataAddress;
    a.cuQuery = args.cuQuery;
    a.cuKv = args.cuKv;
    a.mask = mask.defined() ? TensorAddress(mask) : 0;
    a.usedQuery = args.usedQuery;
    a.usedKv = args.usedKv;
    a.checked = staticPrefill.usedCoreNum || checkRecordOnDevice ? 0 : a.workspace + checkedOffset;
    args.checked = a.checked;
    args.generate = !supplied;
    a.context = c;
    a.staticPrefill = staticPrefill;
    if (validatedInputs) {
        a.inputs = *validatedInputs;
    } else {
        a.inputs = {(tnd ? CheckTndTensorLayout(q) : CheckBsndTensorLayout(q)),
                    qv ? (tnd ? CheckTndTensorLayout(*qv) : CheckBsndTensorLayout(*qv)) : MlaBsndTensorLayout{},
                    (cuK ? CheckTndTensorLayout(k) : CheckBsndTensorLayout(k)),
                    (cuK ? CheckTndTensorLayout(v) : CheckBsndTensorLayout(v))};
    }
    if (pageTable)
        a.cache = {paged, PagedTensorArray(*pageTable, paged.tableSpanElements),
                   PagedTensorArray(k, paged.k.spanElements), PagedTensorArray(v, paged.v.spanElements)};
    args.cache = a.cache;
    a.scale = scale;
    a.softcap = softcap;
    at::Tensor denseRecord;
    bool denseCacheHit = false;
    // A full BSND allocation has no mutable lengths. The producer and every
    // consumer of its private schedule are queued on the same stream.
    if (denseSchedule && !retainLocalBuffers && c.layout == 0 && !c.usedFlags && c.sq && c.sk && !dynamicDecode &&
        a.checked) {
        denseCacheHit = denseSchedule->stream == a.stream && denseSchedule->record.defined();
        denseRecord = denseCacheHit ? denseSchedule->record
                                    : allocate({int64_t(sizeof(MlaCheckedAutoSplitSchedule) + metadataBytes)},
                                               q.options().dtype(at::kByte));
        args.checked = a.checked = TensorAddress(denseRecord);
        args.output = a.metadata = a.checked + sizeof(MlaCheckedAutoSplitSchedule);
    }
    RunMlaLaunch(
        "ascendc_mla",
        [a, args, dynamicDecode = std::move(dynamicDecode), isFp16 = q.scalar_type() == at::kHalf,
         denseCacheHit]() mutable {
            a.dynamicDecode = dynamicDecode.get();
            if (!a.staticPrefill.usedCoreNum && !dynamicDecode && a.checked && !denseCacheHit)
                ComputeMlaMetadata<<<1, nullptr, a.stream>>>(&args, sizeof(args));
            const auto error = isFp16 ? LaunchMla<half>(a) : LaunchMla<bfloat16_t>(a);
            TORCH_CHECK(error == ACL_SUCCESS, "MLA: BATCH_MODE launch failed (", error, ")");
            return 0;
        },
        [&] {
            using LocalBuffers = decltype(std::make_tuple(out, lse, accum, accumLse, workspace));
            auto locals = retainLocalBuffers ? LocalBuffers{out, lse, accum, accumLse, workspace} : LocalBuffers{};
            return std::make_tuple(q, qv, k, v, supplied, usedQ, currentK, cuQ, cuK, pageTable, mask, denseRecord,
                                   std::move(locals));
        });
    if (denseRecord.defined() && !denseCacheHit)
        *denseSchedule = {a.stream, std::move(denseRecord)};
    return {out, lse, accum, accumLse};
}

at::Tensor SchedulerMetadata(int64_t batch, int64_t sq, int64_t sk, int64_t heads, int64_t kvHeads, int64_t dim,
                             int64_t valueDim, at::ScalarType dtype, const at::Tensor& lengths,
                             const std::optional<at::Tensor>& cuQ, std::optional<int64_t> page, int64_t left,
                             int64_t right, bool causal, bool hasSoftcap, int64_t splits, std::optional<bool> pack,
                             int64_t margin)
{
    TORCH_CHECK(batch >= 0 && batch <= UINT32_MAX && sq >= 0 && sq <= INT32_MAX && sk >= 0 && sk <= INT32_MAX &&
                    heads > 0 && heads <= INT32_MAX && kvHeads > 0 && kvHeads <= INT32_MAX && heads % kvHeads == 0 &&
                    dim > 0 && dim <= INT32_MAX && valueDim > 0 && valueDim <= INT32_MAX,
                "MLA: metadata shape bounds exceeded; KV heads must divide query heads");
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "MLA: qkv_dtype must be FP16 or BF16");
    TORCH_CHECK(!page || (*page > 0 && *page <= INT32_MAX), "MLA: page_size must be positive and fit int32");
    CheckCacheLengths(lengths, batch, lengths.device());
    if (cuQ)
        CheckMetadataQueryPrefix(*cuQ, batch, lengths.device());
    const c10::OptionalDeviceGuard device_guard(device_of(lengths));
    TORCH_CHECK((dim == 64 && valueDim == 512 && kvHeads == 1) || (dim == 192 && valueDim == 128),
                "MLA: expected compressed 64/512 or prefill 192/128 dimensions");
    const auto& platform = lengths.is_meta() ? ShapePlatform() : DevicePlatform(lengths.device().index());
    MlaLayout l{};
    const auto c = MakeContext(batch, sq, sk, heads, kvHeads, valueDim, dtype, splits, left, right, causal, 2, platform,
                               page ? (cuQ ? 4 : 3) : (cuQ ? 2 : 0), 0, 0, page.value_or(0), hasSoftcap, pack, margin,
                               lengths.device(), l);
    auto metadata = at::empty({l.words}, lengths.options());
    if (lengths.is_meta())
        return metadata;
    MlaRecordArgs args{c,
                       cuQ ? TensorAddress(*cuQ) : 0,
                       0,
                       0,
                       TensorAddress(lengths),
                       TensorAddress(metadata),
                       l.words,
                       uint32_t(bool(cuQ))};
    auto stream = c10_npu::getCurrentNPUStream();
    RecordInput(lengths, stream);
    if (cuQ)
        RecordInput(*cuQ, stream);
    at_npu::native::OpCommand::RunOpApiV2("ascendc_mla_metadata", [metadata, lengths, cuQ, args, stream]() mutable {
        ComputeMlaMetadata<<<1, nullptr, stream.stream(false)>>>(&args, sizeof(args));
        return 0;
    });
    return metadata;
}

// Adapt the existing public inference arguments without changing ordinary FA3.
std::optional<std::vector<at::Tensor>>
TryForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const std::optional<at::Tensor>& k_new_,
           const std::optional<at::Tensor>& v_new_, const std::optional<at::Tensor>& q_v_,
           const std::optional<at::Tensor>& out_, const std::optional<at::Tensor>& cu_seqlens_q_,
           const std::optional<at::Tensor>& cu_seqlens_k_, const std::optional<at::Tensor>& cu_seqlens_k_new_,
           const std::optional<at::Tensor>& seqused_q_, const std::optional<at::Tensor>& seqused_k_,
           std::optional<int64_t> max_seqlen_q_, std::optional<int64_t> max_seqlen_k_,
           const std::optional<at::Tensor>& page_table_, const std::optional<at::Tensor>& kv_batch_idx_,
           const std::optional<at::Tensor>& leftpad_k_, const std::optional<at::Tensor>& rotary_cos_,
           const std::optional<at::Tensor>& rotary_sin_, const std::optional<at::Tensor>& seqlens_rotary_,
           const std::optional<at::Tensor>& q_descale_, const std::optional<at::Tensor>& k_descale_,
           const std::optional<at::Tensor>& v_descale_, std::optional<double> softmax_scale_, bool is_causal,
           int64_t window_size_left, int64_t window_size_right, int64_t attention_chunk, double softcap_option,
           bool is_rotary_interleaved, const std::optional<at::Tensor>& scheduler_metadata_, int64_t num_splits,
           std::optional<bool> pack_gqa_, int64_t sm_margin, int64_t layout, bool returnLse, bool outputOnly)
{
    const float softcap = static_cast<float>(softcap_option);
    MlaBsndInputLayouts validatedInputs{};
    const bool isMla =
        flash_attn::mla::CheckForwardBoundary(q, k, v, q_v_, cu_seqlens_q_.has_value(), cu_seqlens_k_.has_value(),
                                              page_table_.has_value(), q_v_ ? nullptr : &validatedInputs);
    if (!isMla)
        return std::nullopt;
    TORCH_CHECK(layout == 0 || layout == 1, "MLA: layout must be TND or BSND");
    TORCH_INTERNAL_ASSERT(!outputOnly || !returnLse);
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    TORCH_CHECK(std::isfinite(softcap_option) && std::isfinite(softcap) && softcap_option >= 0 &&
                    (softcap_option == 0 || softcap > 0),
                "MLA: softcap must be finite and non-negative in FP32 without positive underflow");
    TORCH_CHECK(!k_new_ && !v_new_ && !out_ && !cu_seqlens_k_new_ && (cu_seqlens_q_ || !max_seqlen_q_) &&
                    (cu_seqlens_k_ || !max_seqlen_k_) && !kv_batch_idx_ && !leftpad_k_ && !rotary_cos_ &&
                    !rotary_sin_ && !seqlens_rotary_ && is_rotary_interleaved,
                "MLA: k/v, out, cu_seqlens_k_new, cache_batch_idx, cache_leftpad, rotary_cos, rotary_sin, "
                "rotary_seqlens and length/rotary options are unsupported");
    TORCH_CHECK(attention_chunk == 0 &&
                    (num_splits == 1 || (num_splits >= 0 && num_splits <= 256 && (!cu_seqlens_k_ || cu_seqlens_q_))),
                "MLA compressed forward is not implemented yet for mask, split, softcap or scheduling options");
    TORCH_CHECK(!q_descale_ && !k_descale_ && !v_descale_, "MLA: q_descale/k_descale/v_descale are unsupported");
    TORCH_CHECK(!cu_seqlens_q_ || (max_seqlen_q_ && (!cu_seqlens_k_ || max_seqlen_k_)),
                "MLA: TND requires max_seqlen_q/k");
    const float scale = softmax_scale_.value_or(1.0 / std::sqrt(q_v_ ? 576.0 : 192.0));
    TORCH_CHECK(std::isfinite(scale), "MLA: softmax_scale must be finite in FP32");
    auto result = Forward(q, k, v, q_v_, num_splits, window_size_left, window_size_right, is_causal, scale, seqused_q_,
                          seqused_k_, scheduler_metadata_, cu_seqlens_q_, cu_seqlens_k_, max_seqlen_q_.value_or(0),
                          max_seqlen_k_.value_or(0), page_table_, softcap, pack_gqa_, sm_margin,
                          q_v_ ? nullptr : &validatedInputs, returnLse, outputOnly);
    return result;
}

template <bool ReturnLse>
auto ForwardMla(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const std::optional<at::Tensor>& qv,
                const std::optional<at::Tensor>& cuQ, const std::optional<at::Tensor>& cuK,
                const std::optional<at::Tensor>& usedQ, const std::optional<at::Tensor>& usedK,
                const std::optional<c10::SymInt>& maxQ, const std::optional<c10::SymInt>& maxK,
                const std::optional<at::Tensor>& pageTable, std::optional<double> scale, bool causal, c10::SymInt left,
                c10::SymInt right, double softcap, const std::optional<at::Tensor>& metadata, c10::SymInt splits,
                std::optional<bool> pack, c10::SymInt margin)
{
    const auto concrete = [](const std::optional<c10::SymInt>& length) -> std::optional<int64_t> {
        return length ? std::optional<int64_t>(length->guard_int(__FILE__, __LINE__)) : std::nullopt;
    };
    auto result = TryForward(q, k, v, {}, {}, qv, {}, cuQ, cuK, {}, usedQ, usedK, concrete(maxQ), concrete(maxK),
                             pageTable, {}, {}, {}, {}, {}, {}, {}, {}, scale, causal,
                             left.guard_int(__FILE__, __LINE__), right.guard_int(__FILE__, __LINE__), 0, softcap, true,
                             metadata, splits.guard_int(__FILE__, __LINE__), pack, margin.guard_int(__FILE__, __LINE__),
                             cuQ ? 0 : 1, ReturnLse, !ReturnLse);
    TORCH_CHECK(result, "MLA: expected compressed or prefill inputs");
    if constexpr (ReturnLse) {
        auto& values = *result;
        return std::make_tuple(std::move(values[0]), std::move(values[1]), std::move(values[2]), std::move(values[3]));
    } else {
        return std::move((*result)[0]);
    }
}

} // namespace flash_attn::mla

TORCH_LIBRARY_FRAGMENT(flash_attn_npu_3, m)
{
    constexpr const char* schema =
        "_mla_forward_out(Tensor q, Tensor k, Tensor v, Tensor? qv, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, "
        "Tensor? seqused_q, Tensor? seqused_k, SymInt? max_seqlen_q, SymInt? max_seqlen_k, Tensor? page_table, "
        "float? softmax_scale, bool causal, SymInt window_size_left, SymInt window_size_right, float softcap, "
        "Tensor? scheduler_metadata, SymInt num_splits, bool? pack_gqa, SymInt sm_margin) -> Tensor";
    constexpr const char* lseSchema =
        "_mla_forward_lse(Tensor q, Tensor k, Tensor v, Tensor? qv, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, "
        "Tensor? seqused_q, Tensor? seqused_k, SymInt? max_seqlen_q, SymInt? max_seqlen_k, Tensor? page_table, "
        "float? softmax_scale, bool causal, SymInt window_size_left, SymInt window_size_right, float softcap, "
        "Tensor? scheduler_metadata, SymInt num_splits, bool? pack_gqa, SymInt sm_margin) -> (Tensor, Tensor, Tensor, "
        "Tensor)";
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 7)
    m.def(schema, {at::Tag::pt2_compliant_tag, at::Tag::needs_fixed_stride_order});
    m.def(lseSchema, {at::Tag::pt2_compliant_tag, at::Tag::needs_fixed_stride_order});
#else
    m.def(schema);
    m.def(lseSchema);
#endif
}

TORCH_LIBRARY_IMPL(flash_attn_npu_3, PrivateUse1, m)
{
    m.impl("_mla_forward_out", &flash_attn::mla::ForwardMla<false>);
    m.impl("_mla_forward_lse", &flash_attn::mla::ForwardMla<true>);
}

TORCH_LIBRARY_IMPL(flash_attn_npu_3, Autograd, m)
{
    // The native input boundary checks requires_grad for all MLA inputs.
    m.impl("_mla_forward_out", torch::CppFunction::makeFallthrough());
    m.impl("_mla_forward_lse", torch::CppFunction::makeFallthrough());
}
