// Copyright (c) 2026, Minghua Shen.

#include <algorithm>
#include <array>
#include <functional>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "tilingdata.h"
#include "fwd_resource_plan.h"
#include "fwd_tiling_policy.h"

namespace optiling {
const uint32_t SIZE_OF_16BIT = 2;
const uint32_t SIZE_OF_32BIT = 4;
const uint32_t N_SPLIT_HELPER = 2;
const uint32_t MAX_KV_STACK_LEN = 512;
const uint32_t Q_TILE_CEIL = 128;
const uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
const uint32_t BASE_KV_SIZE = 128;
const uint32_t PRELANCH_NUM = 3;
const uint64_t ASCEND950_L1_SIZE = 512ULL * 1024ULL;
const uint64_t RESCALE_OTMP_STAGE_SIZE = 32ULL * 1024ULL;
const uint64_t FD_WORKSPACE_ALIGNMENT = 512ULL;

// The automatic plan has one partial per KV head served by each of 32 core groups.
// Each partial holds two aligned LSE vectors and 16 FP32 output rows.
static constexpr size_t kPairedPlanOffset = (sizeof(FAInferTilingData) + 63U) / 64U * 64U;
static constexpr uint32_t kPairedDecodePartials = 2U * MAX_FD_ACTIVE_CORE_NUM;
static constexpr uint32_t kPairedDecodeOOffset = kPairedDecodePartials * 2U * 8U * sizeof(float);
static constexpr uint32_t kPairedDecodeWorkspace = kPairedDecodeOOffset +
    kPairedDecodePartials * 16U * 192U * sizeof(float);

static bool CanPairDecode(const FAInferTilingData& t, int32_t activeKv)
{
    if (activeKv < 4096 || !t.flashDecodeFlag || t.batch != 1U || t.maxQSeqlen != 1U ||
        (t.kvHeads != 2U && t.kvHeads != 4U) || t.numHeads != 16U * t.kvHeads ||
        t.fdActiveCoreNum != MAX_FD_ACTIVE_CORE_NUM || t.fdBaseTaskNum != t.kvHeads ||
        t.fdCombineTaskNum != t.kvHeads || t.fdRowCapacity != 16U || t.fdLseSubStride != 8U) {
        return false;
    }
    for (uint32_t h = 0; h < t.kvHeads; ++h) {
        if (t.fdCombineSchedules[h].baseTask != static_cast<int32_t>(h)) { return false; }
    }
    return true;
}

static void SetPairedDecodeRanges(FAInferTilingData& t)
{
    t.qL1BufNum = 2U;
    const uint32_t splits = kPairedDecodePartials / t.kvHeads;
    for (uint32_t h = 0; h < t.kvHeads; ++h) {
        t.fdCombineSchedules[h].partialStart = h * splits;
        t.fdCombineSchedules[h].partialCount = splits;
    }
    t.fdPartialTaskNum = t.fdPartialCapacity = kPairedDecodePartials;
    t.fdPartialOOffset = kPairedDecodeOOffset;
    t.workSpaceSize = t.fdWorkspaceEnd = kPairedDecodeWorkspace;
}

enum class MaskType : uint32_t {
    NO_MASK = 0,
    MASK_SPEC = 1,
    MASK_BAND = 2
};

enum class DataType : uint32_t {
    FP16 = 0,
    BF16 = 1,
    FP8 = 2
};

struct FAInferContext {
    int32_t numTokens = 0;
    int32_t numHeads = 0;
    int32_t embeddingSize = 0;
    int32_t embeddingSizeV = 0;
    int32_t numBlocks = 0;
    int32_t blockSize = 0;
    int32_t kvHeads = 0;
    int32_t batch = 0;
    int32_t innerPrecise = 0;
    int64_t maxQSeqlen = 0;
    int64_t maxKvSeqlen = 0;
    int64_t preToken = 0;
    int64_t nextToken = 0;
    int64_t windowSizeLeft = 0;
    int64_t windowSizeRight = 0;
    int32_t sparseMode = 0;
    uint32_t maxNumBlocksPerBatch = 0;
    const int32_t* qSeqlenList{nullptr};
    const int32_t* kvSeqlenList{nullptr};
    float scaleValue = 0.0;
    float softcapValue = 0.0;
    size_t* workspaces{nullptr};
    MaskType maskType = MaskType::MASK_SPEC;
    DataType dataType = DataType::FP16;
    bool pagedCacheFlag = true;
    bool lseFlag = false;
    bool isTilingSink = false;
    bool learnableSinkFlag = false;
    bool flashDecodeFlag = false;
    uint32_t numSplits = 1;
    bool kvcacheNzFlag = false;
    FwdLayout layout = FwdLayout::BSND;
    bool pagedShapeFlag = true;
    bool mixedKvStorage = false;
};

struct BatchParams {
    uint32_t qSeqlen;
    uint32_t kvSeqlen;
    uint32_t curQNBlockTile;
    uint32_t qNBlockNumPerGroup;
    uint32_t curQNBlockNum;
    uint32_t curQSBlockTile;
    uint32_t curQSBlockNum;
    uint32_t curKSBlockTile;
    uint32_t curKSBlockNum;
};

class FAInferTiling {
  public:
    FAInferTiling() = default;
    explicit FAInferTiling(const FAInferContext& faInfo);
    void DoTiling(FAInferTilingData& tilingdata);
    void SetCoreNum(uint32_t blockNum)
    {
        this->blockNum_ = blockNum;
    }
    uint32_t GetCoreNum()
    {
        return this->blockNum_;
    }

  private:
    void FillSplitCoreTilingData(FAInferTilingData& tilingdata);
    void FillWorkSpaceTilingData(FAInferTilingData& faTilingData);
    void FillFlashDecodeTilingData(FAInferTilingData& faTilingData);
    uint32_t GetQSBlockTile(int64_t kvSeqlen);
    uint32_t GetKSBlockTile(int64_t kvSeqlen);
    uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize);
    void FillBasicTilingData(FAInferTilingData& faTilingData);
    BatchParams getBatchParams(uint32_t bIdx, uint32_t groupSize);

  private:
    FAInferContext faInfo_;
    uint32_t blockNum_;
    uint32_t qBaseTile_ = Q_TILE_CEIL;
};

FAInferTiling::FAInferTiling(const FAInferContext& faInfo) : faInfo_(faInfo) {}

uint32_t FAInferTiling::GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    uint32_t qRowNumCeil = Q_TILE_CEIL;
    uint32_t qNBlockTile = (qSeqlen != 0) ? (qRowNumCeil / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
    if (GetQSBlockTile(faInfo_.maxKvSeqlen) < Q_TILE_CEIL && qSeqlen != 0) {
        constexpr uint32_t MAX_M_FOR_LARGE_D = Q_TILE_CEIL / 2U;
        uint32_t maxQNBlockTile = std::max(MAX_M_FOR_LARGE_D / qSeqlen, 1U);
        qNBlockTile = std::min(qNBlockTile, maxQNBlockTile);
    }
    qNBlockTile = std::min(qNBlockTile, groupSize);
    if (qNBlockTile > N_SPLIT_HELPER) {
        qNBlockTile = qNBlockTile / N_SPLIT_HELPER * N_SPLIT_HELPER;
    }
    qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
    return qNBlockTile;
}

uint32_t FAInferTiling::GetQSBlockTile(int64_t kvSeqlen)
{
    return qBaseTile_;
}
uint32_t FAInferTiling::GetKSBlockTile(int64_t kvSeqlen)
{
    uint32_t kSBlockTile = faInfo_.dataType == DataType::FP8 ? 256U : 128U;
    return kSBlockTile;
}

void FAInferTiling::FillBasicTilingData(FAInferTilingData& faTilingData)
{
    faTilingData.set_batch(static_cast<uint32_t>(faInfo_.batch));
    faTilingData.set_numHeads(static_cast<uint32_t>(faInfo_.numHeads));
    faTilingData.set_kvHeads(static_cast<uint32_t>(faInfo_.kvHeads));

    faTilingData.set_embeddingSize(static_cast<uint32_t>(faInfo_.embeddingSize));
    faTilingData.set_embeddingSizeV(static_cast<uint32_t>(faInfo_.embeddingSizeV));

    if (faInfo_.pagedCacheFlag) {
        faTilingData.set_blockSize(static_cast<uint32_t>(faInfo_.blockSize));
    } else {
        faTilingData.set_blockSize(BASE_KV_SIZE);
    }
    faTilingData.set_maxQSeqlen(faInfo_.maxQSeqlen);
    faTilingData.set_maxKvSeqlen(faInfo_.maxKvSeqlen);

    faTilingData.set_maskType(static_cast<uint32_t>(faInfo_.maskType));

    faTilingData.set_sparseMode(faInfo_.sparseMode);
    faTilingData.set_cacheLayout(0U);
    faTilingData.set_flashDecodeFlag(faInfo_.flashDecodeFlag ? 1U : 0U);
    faTilingData.splitMode = faInfo_.numSplits == 0U ? 0U : (faInfo_.numSplits == 1U ? 1U : 2U);
    faTilingData.requestedNumSplits = faInfo_.numSplits;
    faTilingData.set_preToken(static_cast<int64_t>(faInfo_.preToken));
    faTilingData.set_nextToken(static_cast<int64_t>(faInfo_.nextToken));
    faTilingData.set_windowSizeLeft(faInfo_.windowSizeLeft);
    faTilingData.set_windowSizeRight(faInfo_.windowSizeRight);

    auto qBaseTile_ = GetQSBlockTile(faInfo_.maxKvSeqlen);
    const uint32_t quantMode = faInfo_.dataType == DataType::FP8 ? 1U : 0U;
    auto kvBaseTile_ = quantMode == 1 ? 256U : BASE_KV_SIZE;
    faTilingData.quantMode = quantMode;
    auto embeddingSizeAligned16_ = (static_cast<uint32_t>(faInfo_.embeddingSize) + 15U) / 16U * 16U;
    auto embeddingSizeVAligned16_ = (static_cast<uint32_t>(faInfo_.embeddingSizeV) + 15U) / 16U * 16U;
    auto qkL1TileM_ = Q_TILE_CEIL;
    auto qkL1TileKLeft_ = embeddingSizeAligned16_;
    auto qL1BufNum_ = 1;
    // K矩阵开启2buf，D按128分割，S2按256分割
    auto qkL1TileN_ = 256;
    auto qkL1TileKRight_ = quantMode == 1 ? 128 : 192;
    auto kL1BufNum_ = 2;
    // V矩阵开启db，D按128分割，kvBaseTile_不分割，指令同样提前于核间同步下发
    // 如果kvBaseTile_进一步增大，考虑关闭db，使得kvBaseTile_不分割
    auto pvL1TileN_ = embeddingSizeVAligned16_;
    auto pvL1TileKLeft_ = kvBaseTile_;
    auto vL1BufNum_ = 2;
    // P矩阵在950上会常驻L1，由于基块的prelaunch为2，因此最好有3 buf，以免基块间流水阻塞
    auto pvL1TileM_ = Q_TILE_CEIL;
    auto pvL1TileKRight_ = kvBaseTile_;
    auto pL1BufNum_ = 3;

    faTilingData.set_innerPrec(0);
    faTilingData.set_actSeqAval(0);
    faTilingData.set_qBaseTile(qBaseTile_);
    faTilingData.set_kvBaseTile(kvBaseTile_);
    faTilingData.set_qkL1TileM(qkL1TileM_);
    faTilingData.set_qkL1TileN(qkL1TileN_);
    faTilingData.set_qkL1TileKLeft(qkL1TileKLeft_);
    faTilingData.set_qkL1TileKRight(qkL1TileKRight_);
    faTilingData.set_pvL1TileM(pvL1TileM_);
    faTilingData.set_pvL1TileN(pvL1TileN_);
    faTilingData.set_pvL1TileKLeft(pvL1TileKLeft_);
    faTilingData.set_pvL1TileKRight(pvL1TileKRight_);
    faTilingData.set_qL1BufNum(qL1BufNum_);
    faTilingData.set_kL1BufNum(kL1BufNum_);
    faTilingData.set_vL1BufNum(vL1BufNum_);
    faTilingData.set_pL1BufNum(pL1BufNum_);
}

void FAInferTiling::FillWorkSpaceTilingData(FAInferTilingData& faTilingData)
{
    uint64_t qkOutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB * SIZE_OF_32BIT * PRELANCH_NUM;
    uint64_t smOnlineOutSize =
        static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB * SIZE_OF_16BIT * PRELANCH_NUM;
    uint64_t pvOutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB * SIZE_OF_32BIT * PRELANCH_NUM;
    uint64_t UpdateSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB * SIZE_OF_32BIT * PRELANCH_NUM;

    uint64_t splitLseTotalSize = 0;
    uint64_t splitOTotalSize = 0;
    if (faInfo_.isTilingSink) {
        splitLseTotalSize = 2 * static_cast<uint64_t>(blockNum_) * Q_TILE_CEIL * SIZE_OF_32BIT * faInfo_.numHeads;
        uint32_t embeddingSizeV = static_cast<uint32_t>(faInfo_.embeddingSizeV);
        splitOTotalSize =
            2 * static_cast<uint64_t>(blockNum_) * Q_TILE_CEIL * embeddingSizeV * SIZE_OF_32BIT * faInfo_.numHeads;
        faTilingData.set_splitLseTotalSize(splitLseTotalSize);
        faTilingData.set_splitOTotalSize(splitOTotalSize);
        faTilingData.set_needCoreNum(blockNum_);
    } else {
        splitLseTotalSize = faTilingData.get_splitLseTotalSize();
        splitOTotalSize = faTilingData.get_splitOTotalSize();
    }
    uint64_t workSpaceSize = qkOutSize + smOnlineOutSize + pvOutSize + UpdateSize + splitLseTotalSize + splitOTotalSize;
    faTilingData.set_qkOutSize(qkOutSize);
    faTilingData.set_smOnlineOutSize(smOnlineOutSize);
    faTilingData.set_pvOutSize(pvOutSize);
    faTilingData.set_UpdateSize(UpdateSize);
    faTilingData.set_workSpaceSize(workSpaceSize);
}

namespace {
struct FdBaseTaskHost {
    uint32_t kvTiles;
    uint32_t rowNum;
    uint32_t batchIdx;
    uint32_t qStart;
    uint32_t qRows;
    uint32_t qHeadStart;
};

struct FdCandidateHost {
    uint32_t baseTask;
    uint32_t kvBegin;
    uint32_t kvEnd;
    uint64_t cost;
};

inline uint64_t AlignUpU64(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

void FillFdCombineCoreSchedules(FAInferTilingData& tiling)
{
    struct SourcePartition {
        uint32_t task;
        uint32_t subBlock;
        uint32_t rows;
    };
    std::vector<SourcePartition> partitions;
    uint64_t totalLoad = 0U;
    for (uint32_t task = 0U; task < tiling.fdCombineTaskNum; ++task) {
        const uint32_t rows = tiling.fdCombineTaskInfo[task].rowNum;
        const uint32_t sourceSplit = std::min(
            rows, static_cast<uint32_t>(AlignUpU64(rows, tiling.quantMode == 1 ? 32U : 8U)) / FD_SOURCE_SUB_BLOCK_NUM);
        const uint32_t sourceRows[FD_SOURCE_SUB_BLOCK_NUM] = {sourceSplit, rows - sourceSplit};
        for (uint32_t sub = 0U; sub < FD_SOURCE_SUB_BLOCK_NUM; ++sub) {
            if (sourceRows[sub] != 0U) {
                partitions.push_back({task, sub, sourceRows[sub]});
            }
        }
        totalLoad += static_cast<uint64_t>(rows) * tiling.fdCombineSchedules[task].partialCount;
    }

    const uint32_t aivNum = tiling.fdActiveCoreNum * FD_SOURCE_SUB_BLOCK_NUM;
    // FD has fewer base tasks than active AICs, hence every nonempty
    // source partition can reserve at least one AIV.
    if (aivNum == 0U || partitions.size() > aivNum) {
        throw std::runtime_error("FD combine source partitions exceed available AIVs");
    }
    const uint64_t averageLoad = std::max<uint64_t>(1U, (totalLoad + aivNum - 1U) / aivNum);
    uint32_t extraAivs = aivNum - static_cast<uint32_t>(partitions.size());
    uint32_t nextAiv = 0U;
    for (const auto& partition : partitions) {
        const uint64_t load =
            static_cast<uint64_t>(partition.rows) * tiling.fdCombineSchedules[partition.task].partialCount;
        // Reserve one AIV for every remaining partition before deriving
        // the row extent. Recompute the actual count after ceil rounding.
        const uint32_t initialAivs = static_cast<uint32_t>(
            std::min<uint64_t>(std::max<uint64_t>(1U, load / averageLoad), std::min(partition.rows, extraAivs + 1U)));
        const uint32_t rowsPerAiv = (partition.rows + initialAivs - 1U) / initialAivs;
        const uint32_t usedAivs = (partition.rows + rowsPerAiv - 1U) / rowsPerAiv;
        for (uint32_t i = 0U; i < usedAivs; ++i) {
            const uint32_t row = i * rowsPerAiv;
            tiling.fdCombineCoreSchedules[nextAiv++] = {partition.task, partition.subBlock, row,
                                                        std::min(rowsPerAiv, partition.rows - row)};
        }
        extraAivs -= usedAivs - 1U;
    }
    tiling.fdCombineAivNum = nextAiv;

    // One combine task has no inter-task imbalance to remove. For the
    // no-LSE path, also retain uniform dispatch when its busiest AIV's
    // aggregate partial O fits the existing input buffer: extra metadata
    // and per-AIV rounds can cost more than the small reduction in work.
    // Derive this from actual partials/rows, never the requested splits.
    tiling.fdCombineUseUniform = 0U;
    if (tiling.fdCombineTaskNum == 1U) {
        tiling.fdCombineUseUniform = FD_COMBINE_UNIFORM_NO_LSE | FD_COMBINE_UNIFORM_WITH_LSE;
    } else {
        const uint32_t rowTile =
            std::min(FD_COMBINE_ROW_TILE, std::max(1U, (tiling.fdCombineRowNum + aivNum - 1U) / aivNum));
        std::vector<uint64_t> uniformLoads(aivNum, 0U);
        std::vector<uint32_t> uniformRounds(aivNum, 0U);
        uint32_t tile = 0U;
        for (const auto& partition : partitions) {
            for (uint32_t row = 0U; row < partition.rows; row += rowTile) {
                const uint32_t core = tile++ % aivNum;
                ++uniformRounds[core];
                uniformLoads[core] += static_cast<uint64_t>(std::min(rowTile, partition.rows - row)) *
                                      tiling.fdCombineSchedules[partition.task].partialCount;
            }
        }
        const uint64_t maxInputElems =
            *std::max_element(uniformLoads.begin(), uniformLoads.end()) * AlignUpU64(tiling.embeddingSizeV, 8U);
        if (maxInputElems <= FD_COMBINE_O_INPUT_ELEMS) {
            tiling.fdCombineUseUniform = FD_COMBINE_UNIFORM_NO_LSE;
            uint32_t balancedRounds = 0U;
            for (uint32_t core = 0U; core < tiling.fdCombineAivNum; ++core) {
                balancedRounds =
                    std::max(balancedRounds, (tiling.fdCombineCoreSchedules[core].rowCount + FD_COMBINE_ROW_TILE - 1U) /
                                                 FD_COMBINE_ROW_TILE);
            }
            // With no multi-buffer input to remove, adding a whole extra
            // row round can also outweigh balancing the LSE stores.
            if (balancedRounds > *std::max_element(uniformRounds.begin(), uniformRounds.end())) {
                tiling.fdCombineUseUniform |= FD_COMBINE_UNIFORM_WITH_LSE;
            }
        }
    }
}
// Order by load, then by core ID to preserve the first-minimum tie break.
class LeastLoadedCore {
    using Entry = std::pair<uint64_t, uint32_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue_;
public:
    explicit LeastLoadedCore(uint32_t cores)
    {
        for (uint32_t core = 0; core < cores; ++core) {
            queue_.emplace(0U, core);
        }
    }

    Entry Assign(uint64_t work)
    {
        auto entry = queue_.top();
        queue_.pop();
        entry.first += work;
        queue_.push(entry);
        return entry;
    }
};

bool ScheduleLongestChunksFirst(const std::vector<uint64_t>& prefix, uint32_t cores, uint64_t previousMax,
                                FAInferTilingData& tiling)
{
    const size_t tasks = prefix.size() - 1U;
    uint32_t shift = 0U;
    while (((tasks - 1U) >> shift) >= FD_CHUNK_CAPACITY) {
        ++shift;
    }
    const size_t chunkSize = size_t{1} << shift;
    const size_t chunks = 1U + ((tasks - 1U) >> shift);
    std::vector<uint16_t> order(chunks);
    std::vector<uint64_t> costs(chunks);
    for (size_t i = 0; i < chunks; ++i) {
        order[i] = static_cast<uint16_t>(i);
        costs[i] = prefix[std::min(tasks, (i + 1U) * chunkSize)] - prefix[i * chunkSize];
    }
    std::stable_sort(order.begin(), order.end(), [&](uint16_t a, uint16_t b) { return costs[a] > costs[b]; });
    LeastLoadedCore loads(cores);
    std::vector<uint16_t> owners(chunks);
    std::array<uint32_t, MAX_FD_ACTIVE_CORE_NUM> slots{};
    for (uint16_t chunk : order) {
        const auto [load, core] = loads.Assign(costs[chunk]);
        if (load >= previousMax) {
            return false;
        }
        owners[chunk] = static_cast<uint16_t>(core);
        ++slots[core];
    }

    tiling.flashDecodeFlag = FD_WHOLE_TASK_INDEXED;
    tiling.fdWholeTaskOrder.chunkShift = shift;
    uint32_t begin = 0U;
    for (uint32_t core = 0; core < cores; ++core) {
        const uint32_t end = begin + slots[core];
        tiling.fdDecodeSchedules[core] = {static_cast<int32_t>(begin), static_cast<int32_t>(end), 0, 0};
        slots[core] = begin;
        begin = end;
    }
    // Packed Q offsets advance forward. Visit original chunk IDs in order so
    // each core's queue is already sorted when written to metadata.
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        tiling.fdWholeTaskOrder.chunkIds[slots[owners[chunk]]++] = static_cast<uint16_t>(chunk);
    }
    return true;
}

// For a full batch wave, move complete tasks between cores. Cutting a task at a
// global tile offset changes its softmax reductions when requests are permuted.
bool ScheduleWholeDecodeTasks(const std::vector<FdBaseTaskHost>& bases, uint32_t cores, FAInferTilingData& tiling)
{
    if (cores < 2U || bases.size() < cores) {
        return false;
    }
    std::vector<uint64_t> prefix{0U};
    prefix.reserve(bases.size() + 1U);
    std::vector<uint64_t> roundRobin(cores);
    bool uniform = true;
    for (size_t i = 0; i < bases.size(); ++i) {
        const uint64_t cost = static_cast<uint64_t>(bases[i].rowNum) * std::max(1U, bases[i].kvTiles);
        prefix.push_back(prefix.back() + cost);
        roundRobin[i % cores] += cost;
        uniform &= cost == prefix[1];
    }
    const uint64_t longest = *std::max_element(roundRobin.begin(), roundRobin.end());
    const uint64_t lowerBound = (prefix.back() + cores - 1U) / cores;
    if (uniform || longest <= lowerBound) {
        return false;
    }

    uint32_t begin = 0U;
    uint64_t maxCost = 0U;
    for (uint32_t core = 0; core < cores; ++core) {
        const uint32_t last = static_cast<uint32_t>(bases.size()) - (cores - core - 1U);
        const uint64_t target = prefix.back() * (core + 1U) / cores;
        auto cut = std::lower_bound(prefix.begin() + begin + 1U, prefix.begin() + last, target);
        uint32_t end = static_cast<uint32_t>(cut - prefix.begin());
        if (end > begin + 1U && prefix[end] >= target && target - prefix[end - 1U] < prefix[end] - target) {
            --end;
        }
        maxCost = std::max(maxCost, prefix[end] - prefix[begin]);
        tiling.fdDecodeSchedules[core] = {static_cast<int32_t>(begin), static_cast<int32_t>(end),
                                          0, static_cast<int32_t>(bases[end - 1U].kvTiles)};
        begin = end;
    }
    const uint64_t best = std::min(maxCost, longest);
    if (best > lowerBound && ScheduleLongestChunksFirst(prefix, cores, best, tiling)) {
        return true;
    }
    return maxCost < longest;
}

// Identical request-local geometry gets the same split count. Reserve a physical
// core for each segment so adjacent segments cannot coalesce differently after a permutation.
void AssignBatchSplits(const std::vector<FdBaseTaskHost>& bases, uint32_t cores,
                       uint32_t requested, std::vector<uint32_t>& counts)
{
    struct Group {
        uint32_t first;
        std::vector<uint32_t> tasks;
    };
    const auto geometry = [&](uint32_t i) {
        const auto& b = bases[i];
        return std::tie(b.kvTiles, b.rowNum, b.qRows, b.qStart, b.qHeadStart);
    };
    std::vector<Group> groups;
    for (uint32_t i = 0; i < bases.size(); ++i) {
        auto group = std::find_if(groups.begin(), groups.end(),
                                  [&](const Group& g) { return geometry(g.first) == geometry(i); });
        if (group == groups.end()) {
            groups.push_back({i, {i}});
        } else {
            group->tasks.push_back(i);
        }
    }
    uint32_t remaining = cores - static_cast<uint32_t>(bases.size());
    while (remaining > 0U) {
        const Group* best = nullptr;
        uint64_t bestCost = 0U;
        for (const auto& group : groups) {
            const uint32_t i = group.first;
            if (group.tasks.size() > remaining || counts[i] >= bases[i].kvTiles ||
                (requested != 0U && counts[i] >= requested)) {
                continue;
            }
            const uint64_t cost = static_cast<uint64_t>(bases[i].rowNum) * bases[i].kvTiles / counts[i];
            if (!best || cost > bestCost || (cost == bestCost && geometry(i) > geometry(best->first))) {
                best = &group;
                bestCost = cost;
            }
        }
        if (!best) {
            break;
        }
        for (uint32_t i : best->tasks) {
            ++counts[i];
        }
        remaining -= static_cast<uint32_t>(best->tasks.size());
    }
}

// Compare canonical split counts on a permutation-independent set of KV spans.
uint64_t PeakSegmentTiles(std::vector<uint32_t>& spans, uint32_t cores)
{
    std::sort(spans.begin(), spans.end(), std::greater<uint32_t>());
    LeastLoadedCore loads(cores);
    uint64_t peak = 0U;
    for (uint32_t span : spans) {
        peak = std::max(peak, loads.Assign(span).first);
    }
    return peak;
}

bool ScheduleSplitGrid(const FAInferContext& ctx, const std::vector<FdBaseTaskHost>& bases,
                       uint32_t cores, FAInferTilingData& t)
{
    const size_t n = bases.size();
    if (!ctx.mixedKvStorage || ctx.batch < 2 || ctx.numSplits || ctx.maxQSeqlen != 8 ||
        ctx.embeddingSize != 192 || ctx.embeddingSizeV != 192 || ctx.softcapValue != 0 ||
        cores < 2U || n <= cores / 2U || n != static_cast<size_t>(ctx.batch) * ctx.kvHeads ||
        ctx.numHeads != 16 * ctx.kvHeads || n > UINT32_MAX / 8U) {
        return false;
    }
    uint32_t minTiles = UINT32_MAX, maxTiles = 0U;
    uint64_t total = 0U;
    for (size_t i = 0; i < n; ++i) {
        const auto& task = bases[i];
        if (task.qRows != 8U || task.rowNum != 128U || task.qStart != 0U ||
            task.batchIdx != i / ctx.kvHeads || task.qHeadStart != (i % ctx.kvHeads) * 16U) {
            return false;
        }
        minTiles = std::min(minTiles, task.kvTiles);
        maxTiles = std::max(maxTiles, task.kvTiles);
        total += task.kvTiles;
    }
    const bool uniform = minTiles == maxTiles;
    if (minTiles < 2U || (!uniform && n * 2U > FD_CHUNK_CAPACITY) || (uniform && n % cores == 0U)) {
        return false;
    }
    // At M=128,D=192, the measured cost of writing, rereading and merging
    // six FP32 partial output tiles is roughly one KV tile on the busiest core.
    constexpr uint64_t partialsPerTile = 6U;
    constexpr uint64_t setupTiles = 4U;
    std::vector<uint32_t> spans;
    uint64_t best = 0U;
    if (n < cores) {
        std::vector<uint32_t> baseline(n, 1U);
        AssignBatchSplits(bases, cores, 0U, baseline);
        // There is at most one baseline segment per core.
        uint32_t peak = 0U;
        for (size_t i = 0; i < n; ++i) {
            const uint32_t parts = baseline[i];
            peak = std::max(peak, 1U + (bases[i].kvTiles - 1U) / parts);
            best += parts > 1U ? parts : 0U;
        }
        best += partialsPerTile * peak;
    } else if (uniform) {
        best = partialsPerTile * ((n + cores - 1U) / cores) * maxTiles;
    } else {
        spans.reserve(n);
        for (const auto& base : bases) {
            spans.push_back(base.kvTiles);
        }
        best = partialsPerTile * PeakSegmentTiles(spans, cores);
    }
    const uint64_t ideal = (total + cores - 1U) / cores;
    uint32_t choice = 0U;
    for (uint32_t r = 2U; r <= std::min(8U, minTiles); ++r) {
        const uint64_t overhead = n * r + setupTiles * partialsPerTile;
        if (partialsPerTile * ideal + overhead >= best || (!uniform && n * r > FD_CHUNK_CAPACITY)) {
            break;
        }
        uint64_t peak;
        if (uniform) {
            // Every request repeats the same r segment lengths.
            std::array<uint32_t, 8> widths{};
            for (uint32_t j = 0; j < r; ++j) {
                widths[j] = uint64_t(maxTiles) * (j + 1U) / r - uint64_t(maxTiles) * j / r;
            }
            std::array<uint64_t, MAX_FD_ACTIVE_CORE_NUM> loads{};
            uint32_t core = 0U;
            for (size_t i = 0; i < n; ++i) {
                for (uint32_t j = 0; j < r; ++j) {
                    loads[core] += widths[j];
                    if (++core == cores) {
                        core = 0U;
                    }
                }
            }
            peak = *std::max_element(loads.begin(), loads.begin() + cores);
        } else {
            spans.clear();
            spans.reserve(n * r);
            for (const auto& base : bases) {
                // Only the multiset matters for longest-first prediction.
                const uint32_t small = base.kvTiles / r;
                const uint32_t extra = base.kvTiles % r;
                spans.insert(spans.end(), r - extra, small);
                spans.insert(spans.end(), extra, small + 1U);
            }
            peak = PeakSegmentTiles(spans, cores);
        }
        const uint64_t score = partialsPerTile * peak + overhead;
        if (score < best) {
            best = score;
            choice = r;
        }
    }
    if (choice == 0U) {
        return false;
    }
    if (!uniform) {
        std::vector<uint64_t> prefix{0U}, loads(cores);
        for (const auto& base : bases) {
            for (uint32_t j = 0; j < choice; ++j) {
                const uint64_t tiles = uint64_t(base.kvTiles) * (j + 1U) / choice - uint64_t(base.kvTiles) * j / choice;
                loads[(prefix.size() - 1U) % cores] += tiles;
                prefix.push_back(prefix.back() + tiles);
            }
        }
        ScheduleLongestChunksFirst(prefix, cores, *std::max_element(loads.begin(), loads.end()), t);
    }
    t.flashDecodeFlag = FD_UNIFORM_SPLIT_GRID;
    t.fdActiveCoreNum = cores;
    t.fdCombineTaskNum = static_cast<uint32_t>(n);
    t.fdPartialCapacity = t.fdPartialTaskNum = t.fdCombineTaskNum * choice;
    t.fdRowCapacity = 128U;
    t.fdLseSubStride = t.fdRowCapacity / 2U;
    t.fdCombineRowNum = t.fdRowCapacity * t.fdCombineTaskNum;
    t.fdCombineUseUniform = FD_COMBINE_UNIFORM_NO_LSE | FD_COMBINE_UNIFORM_WITH_LSE;
    t.fdPartialLseOffset = 0U;
    t.fdPartialOOffset = AlignUpU64(uint64_t(t.fdPartialTaskNum) * 2U * t.fdLseSubStride * sizeof(float),
                                    FD_WORKSPACE_ALIGNMENT);
    t.workSpaceSize = t.fdWorkspaceEnd = t.fdPartialOOffset +
        uint64_t(t.fdPartialTaskNum) * t.fdRowCapacity * ctx.embeddingSizeV * sizeof(float);
    return true;
}

// For decode tasks with the same number of query rows, a KV tile is
// the scheduling unit. Compare the normal round-robin critical path
// with an even partition of all KV tiles, including unequal contexts.
bool ScheduleKvTail(const FAInferContext& ctx, const std::vector<FdBaseTaskHost>& bases,
                    uint32_t cores, FAInferTilingData& tiling)
{
    if (!ctx.mixedKvStorage || ctx.numSplits != 0U || bases.empty() || cores < 2U) {
        return false;
    }
    std::vector<uint64_t> prefix{0U};
    prefix.reserve(bases.size() + 1U);
    std::vector<uint64_t> roundRobinLoad(cores);
    for (size_t i = 0; i < bases.size(); ++i) {
        const auto& task = bases[i];
        if (task.kvTiles == 0U || task.rowNum != bases.front().rowNum) {
            return false;
        }
        prefix.push_back(prefix.back() + task.kvTiles);
        roundRobinLoad[i % cores] += task.kvTiles;
    }
    const uint64_t total = prefix.back();
    const uint64_t longest = *std::max_element(roundRobinLoad.begin(), roundRobinLoad.end());
    const uint64_t balanced = (total + cores - 1U) / cores;
    if (total < cores || longest - balanced < 8U || (longest - balanced) * 8U < longest) {
        return false;
    }

    for (uint32_t core = 0; core < cores; ++core) {
        const uint64_t begin = total * core / cores;
        const uint64_t last = total * (core + 1U) / cores - 1U;
        const auto firstTask = std::upper_bound(prefix.begin(), prefix.end(), begin) - prefix.begin() - 1;
        const auto lastTask = std::upper_bound(prefix.begin(), prefix.end(), last) - prefix.begin() - 1;
        tiling.fdDecodeSchedules[core] = {static_cast<int32_t>(firstTask), static_cast<int32_t>(lastTask + 1),
            static_cast<int32_t>(begin - prefix[firstTask]), static_cast<int32_t>(last - prefix[lastTask] + 1U)};
    }
    return true;
}
} // namespace

void FAInferTiling::FillFlashDecodeTilingData(FAInferTilingData& tiling)
{
    tiling.fdCombineAivNum = 0U;
    tiling.fdCombineUseUniform = 0U;
    for (auto& schedule : tiling.fdCombineCoreSchedules) {
        schedule = {};
    }
    for (uint32_t i = 0; i < MAX_FD_ACTIVE_CORE_NUM; ++i) {
        tiling.fdDecodeSchedules[i] = {-1, -1, -1, -1};
    }
    for (uint32_t i = 0; i < MAX_FD_COMBINE_TASK_NUM; ++i) {
        tiling.fdCombineSchedules[i] = {-1, -1, -1, 0};
        tiling.fdCombineTaskInfo[i] = {};
    }

    const uint32_t cfdMax = std::min(blockNum_, MAX_FD_ACTIVE_CORE_NUM);
    // With fused KV dequantization, two KV segments per base still use all
    // cores at half occupancy (e.g. B=8, Hkv=2, eight query tokens).
    const uint32_t baseLimit = faInfo_.mixedKvStorage ? cfdMax / 2U :
        (blockNum_ == 0U ? 0U : (3U * blockNum_ - 1U) / 10U);
    const uint32_t bfdMax = std::min(MAX_FD_COMBINE_TASK_NUM, baseLimit);
    tiling.fdPartialCapacity = (bfdMax == 0U || cfdMax == 0U) ? 0U : bfdMax + cfdMax - 1U;

    std::vector<FdBaseTaskHost> bases;
    const uint32_t groupSize = faInfo_.numHeads / faInfo_.kvHeads;
    for (int32_t batchIdx = 0; batchIdx < faInfo_.batch; ++batchIdx) {
        uint32_t qLen = static_cast<uint32_t>(faInfo_.qSeqlenList[batchIdx]);
        uint32_t kvLen = static_cast<uint32_t>(faInfo_.kvSeqlenList[batchIdx]);
        if (faInfo_.layout == FwdLayout::TND) {
            qLen = static_cast<uint32_t>(faInfo_.qSeqlenList[batchIdx + 1] - faInfo_.qSeqlenList[batchIdx]);
            if (!faInfo_.pagedCacheFlag) {
                kvLen = static_cast<uint32_t>(faInfo_.kvSeqlenList[batchIdx + 1] - faInfo_.kvSeqlenList[batchIdx]);
            }
        }
        const uint32_t qTile = GetQSBlockTile(kvLen);
        const uint32_t qTileNum = (qLen + qTile - 1U) / qTile;
        const uint32_t kvTileNum = (kvLen + tiling.kvBaseTile - 1U) / tiling.kvBaseTile;
        const uint32_t qNBlockTile = GetQNBlockTile(qLen, groupSize);
        const uint32_t qNBlockNumPerGroup = (groupSize + qNBlockTile - 1U) / qNBlockTile;
        const uint32_t qNTaskNum = qNBlockNumPerGroup * faInfo_.kvHeads;
        for (uint32_t qTileIdx = 0; qTileIdx < qTileNum; ++qTileIdx) {
            const uint32_t qRows = std::min(qTile, qLen - qTileIdx * qTile);
            for (uint32_t qNTask = 0; qNTask < qNTaskNum; ++qNTask) {
                const uint32_t qNBlockIdxInGroup = qNTask % qNBlockNumPerGroup;
                const uint32_t qNBlockSize = std::min(qNBlockTile, groupSize - qNBlockIdxInGroup * qNBlockTile);
                const uint32_t kvHeadIdx = qNTask / qNBlockNumPerGroup;
                bases.push_back({kvTileNum, qRows * qNBlockSize, static_cast<uint32_t>(batchIdx), qTileIdx * qTile,
                                 qRows, kvHeadIdx * groupSize + qNBlockIdxInGroup * qNBlockTile});
            }
        }
    }

    tiling.fdBaseTaskNum = static_cast<uint32_t>(bases.size());
    if (ScheduleSplitGrid(faInfo_, bases, cfdMax, tiling)) {
        return;
    }
    const bool mixedBatch = faInfo_.mixedKvStorage && faInfo_.batch > 1;
    if (mixedBatch && cfdMax > 0U &&
        (static_cast<uint32_t>(faInfo_.batch) >= cfdMax || bases.size() >= cfdMax)) {
        if (!ScheduleWholeDecodeTasks(bases, cfdMax, tiling)) {
            tiling.flashDecodeFlag = 0U;
            return;
        }
        tiling.fdActiveCoreNum = cfdMax;
        tiling.fdPartialCapacity = 0U;
        tiling.fdCombineTaskNum = 0U;
        tiling.workSpaceSize = 0U;
        return;
    }
    uint32_t activeCores = cfdMax;
    if (!mixedBatch && bases.size() > bfdMax && ScheduleKvTail(faInfo_, bases, cfdMax, tiling)) {
        // At most cores-1 boundaries split a base. Each base needs one
        // partial per interior boundary plus one for its first core.
        tiling.fdPartialCapacity = 2U * (cfdMax - 1U);
    } else {
        if (bases.empty() || (!mixedBatch && bases.size() > bfdMax) || cfdMax < 2U) {
            tiling.flashDecodeFlag = 0U;
            return;
        }

        std::vector<uint32_t> splitCounts(bases.size(), 1U);
        if (mixedBatch) {
            AssignBatchSplits(bases, cfdMax, faInfo_.numSplits, splitCounts);
        } else if (faInfo_.numSplits > 1U) {
            for (uint32_t base = 0; base < bases.size(); ++base) {
                // Empty KV still needs a task to initialize O and LSE.
                splitCounts[base] = std::max(1U, std::min(faInfo_.numSplits, bases[base].kvTiles));
            }
        } else {
            uint32_t totalSegments = static_cast<uint32_t>(bases.size());
            while (totalSegments < cfdMax) {
                uint32_t bestBase = static_cast<uint32_t>(bases.size());
                uint64_t bestCost = 0U;
                for (uint32_t base = 0; base < bases.size(); ++base) {
                    if (splitCounts[base] >= bases[base].kvTiles) {
                        continue;
                    }
                    const uint64_t cost =
                        static_cast<uint64_t>(bases[base].rowNum) * bases[base].kvTiles / splitCounts[base];
                    if (bestBase == bases.size() || cost > bestCost) {
                        bestBase = base;
                        bestCost = cost;
                    }
                }
                if (bestBase == bases.size()) {
                    break;
                }
                ++splitCounts[bestBase];
                ++totalSegments;
            }
        }

        std::vector<FdCandidateHost> candidates;
        for (uint32_t base = 0; base < bases.size(); ++base) {
            const uint32_t count = splitCounts[base];
            for (uint32_t split = 0; split < count; ++split) {
                const uint32_t begin = static_cast<uint32_t>(static_cast<uint64_t>(bases[base].kvTiles) * split / count);
                const uint32_t end =
                    static_cast<uint32_t>(static_cast<uint64_t>(bases[base].kvTiles) * (split + 1U) / count);
                candidates.push_back({base, begin, end, static_cast<uint64_t>(bases[base].rowNum) * (end - begin)});
            }
        }

        activeCores = std::min(cfdMax, static_cast<uint32_t>(candidates.size()));
        if (activeCores <= bases.size()) {
            tiling.flashDecodeFlag = 0U;
            return;
        }

        uint64_t totalCost = 0U;
        for (const auto& candidate : candidates) {
            totalCost += candidate.cost;
        }
        std::vector<uint32_t> coreBegin(activeCores);
        std::vector<uint32_t> coreEnd(activeCores);
        uint32_t candidateBegin = 0U;
        uint64_t consumedCost = 0U;
        for (uint32_t core = 0; core < activeCores; ++core) {
            coreBegin[core] = candidateBegin;
            const uint32_t coresLeft = activeCores - core;
            const uint32_t maxEnd = static_cast<uint32_t>(candidates.size()) - (coresLeft - 1U);
            uint32_t end = candidateBegin + 1U;
            const uint64_t target = totalCost * (core + 1U) / activeCores;
            while (end < maxEnd && consumedCost + candidates[end - 1U].cost < target) {
                consumedCost += candidates[end - 1U].cost;
                ++end;
            }
            consumedCost += candidates[end - 1U].cost;
            coreEnd[core] = end;
            candidateBegin = end;

            const auto& first = candidates[coreBegin[core]];
            const auto& last = candidates[end - 1U];
            tiling.fdDecodeSchedules[core] = {static_cast<int32_t>(first.baseTask),
                                              static_cast<int32_t>(last.baseTask + 1U), static_cast<int32_t>(first.kvBegin),
                                              static_cast<int32_t>(last.kvEnd)};
        }

    }

    uint32_t maxFdRows = 0U;
    uint32_t partialStart = 0U;
    uint32_t combineCount = 0U;
    uint32_t combineRows = 0U;
    for (uint32_t base = 0; base < bases.size(); ++base) {
        int32_t firstCore = -1;
        uint32_t partialCount = 0U;
        for (uint32_t core = 0; core < activeCores; ++core) {
            const auto& schedule = tiling.fdDecodeSchedules[core];
            if (schedule.baseTaskStart <= static_cast<int32_t>(base) &&
                static_cast<int32_t>(base) < schedule.baseTaskEnd) {
                if (firstCore < 0) {
                    firstCore = static_cast<int32_t>(core);
                }
                ++partialCount;
            }
        }
        if (partialCount > 1U || !faInfo_.mixedKvStorage) {
            maxFdRows = std::max(maxFdRows, bases[base].rowNum);
        }
        if (partialCount > 1U) {
            tiling.fdCombineSchedules[combineCount] = {static_cast<int32_t>(base), firstCore,
                                                       static_cast<int32_t>(partialStart),
                                                       static_cast<int32_t>(partialCount)};
            const auto& info = bases[base];
            tiling.fdCombineTaskInfo[combineCount++] = {info.batchIdx, info.qStart, info.qRows, info.qHeadStart,
                                                        info.rowNum};
            partialStart += partialCount;
            combineRows += bases[base].rowNum;
        }
    }

    if (combineCount == 0U || partialStart > tiling.fdPartialCapacity) {
        tiling.flashDecodeFlag = 0U;
        return;
    }

    tiling.fdActiveCoreNum = activeCores;
    tiling.fdCombineTaskNum = combineCount;
    tiling.fdPartialTaskNum = partialStart;
    tiling.fdCombineRowNum = combineRows;
    FillFdCombineCoreSchedules(tiling);
    if (faInfo_.mixedKvStorage) {
        // Only tasks shared by multiple cores write partial O and LSE to GM.
        tiling.fdPartialCapacity = tiling.fdPartialTaskNum;
    }
    tiling.fdRowCapacity = static_cast<uint32_t>(AlignUpU64(maxFdRows, 8U));
    tiling.fdLseSubStride = tiling.quantMode == 1
                                ? static_cast<uint32_t>(AlignUpU64(tiling.fdRowCapacity, 32U) / 2U)
                                : static_cast<uint32_t>(AlignUpU64((tiling.fdRowCapacity + 1U) / 2U, 8U));

    // The Ascend950 FD pipeline keeps QK, softmax, PV and rescale buffers
    // on chip. Its GM workspace only needs the partial LSE and O regions.
    tiling.set_qkOutSize(0U);
    tiling.set_smOnlineOutSize(0U);
    tiling.set_pvOutSize(0U);
    tiling.set_UpdateSize(0U);
    tiling.fdPartialLseOffset = 0U;
    const uint64_t partialLseSize =
        static_cast<uint64_t>(tiling.fdPartialCapacity) * 2U * tiling.fdLseSubStride * sizeof(float);
    tiling.fdPartialOOffset = AlignUpU64(tiling.fdPartialLseOffset + partialLseSize, FD_WORKSPACE_ALIGNMENT);
    const uint64_t partialOSize =
        static_cast<uint64_t>(tiling.fdPartialCapacity) * tiling.fdRowCapacity * faInfo_.embeddingSizeV * sizeof(float);
    tiling.fdWorkspaceEnd = tiling.fdPartialOOffset + partialOSize;
    tiling.workSpaceSize = tiling.fdWorkspaceEnd;
}

void FAInferTiling::FillSplitCoreTilingData(FAInferTilingData& faTilingData)
{
    uint32_t totalTaskNum = 0;
    uint32_t groupSize = faInfo_.numHeads / faInfo_.kvHeads;
    for (int32_t batchIdx = 0; batchIdx < faInfo_.batch; batchIdx++) {
        uint32_t qSeqlen = *(faInfo_.qSeqlenList + batchIdx);
        uint32_t kvSeqlen = *(faInfo_.kvSeqlenList + batchIdx);
        if (faInfo_.layout == FwdLayout::TND) {
            uint64_t prevQSeqlenSum = *(faInfo_.qSeqlenList + batchIdx);
            qSeqlen = *(faInfo_.qSeqlenList + batchIdx + 1) - prevQSeqlenSum;
            if (!faInfo_.pagedCacheFlag) {
                uint64_t prevKvSeqlenSum = *(faInfo_.kvSeqlenList + batchIdx);
                kvSeqlen = *(faInfo_.kvSeqlenList + batchIdx + 1) - prevKvSeqlenSum;
            }
        }
        uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
        uint32_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
        uint32_t curQNBlockNum = qNBlockNumPerGroup * faInfo_.kvHeads;
        uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
        uint32_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
        uint32_t curTaskNum = curQNBlockNum * curQSBlockNum;
        if (batchIdx == 0) {
            faTilingData.set_firstBatchTaskNum(curTaskNum);
        }
        totalTaskNum += curTaskNum;
    }
    faTilingData.set_totalTaskNum(totalTaskNum);
}
BatchParams FAInferTiling::getBatchParams(uint32_t bIdx, uint32_t groupSize)
{
    BatchParams p;
    p.qSeqlen = *(faInfo_.qSeqlenList + bIdx);
    p.kvSeqlen = *(faInfo_.kvSeqlenList + bIdx);
    if (bIdx > 0) {
        uint64_t prevQSeqlenSum = *(faInfo_.qSeqlenList + bIdx - 1);
        p.qSeqlen = p.qSeqlen - prevQSeqlenSum;
    }
    p.curQNBlockTile = GetQNBlockTile(p.qSeqlen, groupSize);
    p.qNBlockNumPerGroup = (groupSize + p.curQNBlockTile - 1) / p.curQNBlockTile;
    p.curQNBlockNum = p.qNBlockNumPerGroup * faInfo_.kvHeads;
    p.curQSBlockTile = GetQSBlockTile(p.kvSeqlen);
    p.curQSBlockNum = (p.qSeqlen + p.curQSBlockTile - 1) / p.curQSBlockTile;
    p.curKSBlockTile = GetKSBlockTile(p.kvSeqlen);
    p.curKSBlockNum = (p.kvSeqlen + p.curKSBlockTile - 1) / p.curKSBlockTile;
    return p;
}

void FAInferTiling::DoTiling(FAInferTilingData& tilingdata)
{
    uint64_t fullQ128Tasks = 0;
    uint64_t windowTasks = 0;
    const bool countWindowTasks = faInfo_.mixedKvStorage && faInfo_.pagedCacheFlag && !faInfo_.isTilingSink &&
        !faInfo_.flashDecodeFlag && faInfo_.maskType == MaskType::MASK_BAND &&
        faInfo_.windowSizeLeft == 511 && faInfo_.windowSizeRight == 0 && faInfo_.softcapValue == 0.0f &&
        faInfo_.embeddingSize == 192 && faInfo_.embeddingSizeV == 192 && faInfo_.numHeads == 16 * faInfo_.kvHeads &&
        faInfo_.maxQSeqlen >= Q_TILE_CEIL && uint64_t(faInfo_.batch) * faInfo_.numHeads <= 2ULL * blockNum_;
    bool fullQueryTileInEachBatch = true;
    for (uint32_t b = 0; b < static_cast<uint32_t>(faInfo_.batch); ++b) {
        const uint32_t qLen =
            faInfo_.qSeqlenList == nullptr
                ? faInfo_.maxQSeqlen
                : (faInfo_.layout == FwdLayout::TND ? faInfo_.qSeqlenList[b + 1] - faInfo_.qSeqlenList[b]
                                                    : faInfo_.qSeqlenList[b]);
        const uint32_t kvLen = faInfo_.kvSeqlenList == nullptr
                                   ? faInfo_.maxKvSeqlen
                                   : (faInfo_.layout == FwdLayout::TND && !faInfo_.pagedCacheFlag
                                          ? faInfo_.kvSeqlenList[b + 1] - faInfo_.kvSeqlenList[b]
                                          : faInfo_.kvSeqlenList[b]);
        const uint32_t activeQ =
            fa950_plan::ActiveQueryRows(qLen, kvLen, faInfo_.maskType != MaskType::NO_MASK, faInfo_.windowSizeRight);
        if (countWindowTasks) {
            windowTasks += ((uint64_t(qLen) + Q_TILE_CEIL - 1U) / Q_TILE_CEIL) * faInfo_.numHeads;
        }
        fullQueryTileInEachBatch &= qLen >= Q_TILE_CEIL;
        fullQ128Tasks += uint64_t(activeQ / Q_TILE_CEIL) * faInfo_.numHeads;
    }
    qBaseTile_ = fa950_plan::SelectQueryTile(faInfo_.embeddingSize, faInfo_.embeddingSizeV, faInfo_.maxQSeqlen,
                                             fullQ128Tasks, blockNum_, faInfo_.maskType == MaskType::MASK_BAND,
                                             faInfo_.windowSizeLeft, faInfo_.windowSizeRight);
    // A ragged last Q tile still occupies a processing slot. At exactly one
    // wave use individual heads; above it a pair can avoid an entire second wave.
    const bool oneWindowWave = fullQueryTileInEachBatch && windowTasks >= blockNum_ &&
                               windowTasks <= 2ULL * blockNum_ && windowTasks != 0U;
    const bool hasPairedWave = faInfo_.mixedKvStorage && fullQueryTileInEachBatch &&
                               (faInfo_.numHeads / faInfo_.kvHeads) % 2U == 0U &&
                               (fullQ128Tasks >= 2ULL * blockNum_ || (oneWindowWave && windowTasks > blockNum_));
    const bool pairedWindow = faInfo_.maskType == MaskType::MASK_BAND &&
                              faInfo_.windowSizeRight == 0 && faInfo_.windowSizeLeft >= 0 &&
                              (faInfo_.softcapValue == 0.0f || faInfo_.windowSizeLeft >= 511);
    const bool pairedFull = faInfo_.maskType != MaskType::MASK_BAND && faInfo_.pagedCacheFlag &&
                            !faInfo_.flashDecodeFlag && !faInfo_.isTilingSink && faInfo_.softcapValue == 0.0f &&
                            faInfo_.embeddingSize == 192 && faInfo_.embeddingSizeV == 192;
    // Each KV page serves both Q heads. Once every Cube can run a pair,
    // this reuse also pays off for narrow windows in the pipelined PV kernel.
    if (oneWindowWave || (hasPairedWave && (pairedWindow || pairedFull))) {
        qBaseTile_ = fa950_plan::QueryTile(faInfo_.embeddingSize, faInfo_.embeddingSizeV, faInfo_.maxQSeqlen);
    }
    // Full MTP amortizes KV conversion across all 16 GQA heads. For low-occupancy
    // SWA, splitting them into two 64-row tasks shortens the critical path.
    if (faInfo_.mixedKvStorage && faInfo_.maxQSeqlen >= 5 && faInfo_.maxQSeqlen <= 8 &&
        faInfo_.numHeads / faInfo_.kvHeads == 16 &&
        !(faInfo_.maskType == MaskType::MASK_BAND &&
          static_cast<uint64_t>(faInfo_.batch) * faInfo_.kvHeads * 2U <= blockNum_)) {
        qBaseTile_ = 128U;
    }
    FillBasicTilingData(tilingdata);
    // Pair adjacent Q heads sharing a KV head. Each pair owns two Q banks in
    // L1; the K and V banks are released only after both consumers read them.
    if (hasPairedWave && !faInfo_.flashDecodeFlag && !faInfo_.isTilingSink && qBaseTile_ == Q_TILE_CEIL) {
        tilingdata.set_qL1BufNum(2U);
    }
    if (!faInfo_.isTilingSink) {
        FillSplitCoreTilingData(tilingdata);
    }
    FillWorkSpaceTilingData(tilingdata);
    if (faInfo_.flashDecodeFlag) {
        FillFlashDecodeTilingData(tilingdata);
    }
}
} // namespace optiling
