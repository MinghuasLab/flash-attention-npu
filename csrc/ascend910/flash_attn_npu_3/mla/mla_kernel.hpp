/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once

#include "mla_params.h"
#include "mla_vector_block.hpp"
#include "kernel_operator.h"
#include "mla_metadata_consumer.h"

// Source: fixed CANN ca18aee, fia_kernel_nonquant_mla.h.
// Keep the source three-slot task ring and event lifetime. Allocation/used
// spans and independent strides are supplied by the common metadata consumer.
namespace flash_attn::mla::Kernel {
using namespace AscendC;
template <typename InputT, bool HasKvLengths, bool IsTnd, bool MixedKv = false, bool HasUsed = false,
          bool HasStrides = false, bool IsPaged = false, bool IsCausal = false, bool IsWindow = false,
          bool IsPartial = false, bool AbsolutePartialOffsets = false>
class MlaTaskRing {
    static_assert(IsSameType<InputT, half>::value || IsSameType<InputT, bfloat16_t>::value,
                  "MLA dense only supports FP16 and BF16");
    static_assert(!MixedKv || (IsTnd && HasKvLengths));
    static_assert(!HasUsed || HasKvLengths);
    static_assert(!IsPaged || (HasKvLengths && HasStrides && !MixedKv && !HasUsed));
    static_assert(!IsWindow || (HasKvLengths && !IsCausal));
    static_assert(!IsPartial || IsWindow);
    static constexpr uint32_t kSpanWords = 8;
    static constexpr bool kQueryLengths = IsTnd || HasUsed || IsPaged;

  protected:
    __aicore__ inline void Run(const MlaDenseKernelSchedule& schedule, float scale, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv,
                               GM_ADDR kr, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace, TPipe* pipe,
                               GM_ADDR kvLengths, uint32_t totalQuery = 0, const MlaBsndInputLayouts* layouts = nullptr,
                               bool kvIsTnd = IsTnd && !MixedKv, GM_ADDR pageTable = nullptr,
                               const MlaPagedCacheLayout* pagedLayout = nullptr,
                               MlaWindowKernelParams windowParams = {}, const MlaSplitKvKernelParams* partial = nullptr,
                               float softcap = 0.0f, const MlaTaskLengths* fragmentLengths = nullptr)
    {
        params = schedule;
        taskLengths = fragmentLengths;
        if constexpr (IsWindow) {
            window = windowParams;
        }
        queryTokens = IsTnd ? totalQuery : params.sq;
        if constexpr (HasKvLengths) {
            lengths.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(kvLengths));
        }
        uint32_t index = GetBlockIdx();
        uint32_t ratio = 1;
        if ASCEND_IS_AIV {
            ratio = GetSubBlockNum();
            index /= ratio;
        }
        if (index >= params.usedCoreNum) {
            if ASCEND_IS_AIV {
                SyncAll();
            }
            return;
        }
        core = index;
        bStart = core == 0 ? 0 : params.bN2End[core - 1];
        mStart = core == 0 ? 0 : params.gS1End[core - 1];
        sStart = core == 0 ? 0 : params.s2End[core - 1];
        bEnd = params.bN2End[core];
        mEnd = params.gS1End[core];
        sEnd = params.s2End[core];

        if constexpr (IsPartial) {
            const auto& task = partial->tasks[core];
            bStart = bEnd = task.batch;
            mStart = mEnd = task.m;
            sStart = task.s2Begin;
            sEnd = task.s2End;
            if (sStart == sEnd) {
                if ASCEND_IS_AIV {
                    InitEmptyPartial(workspace, *partial, ratio);
                }
                return;
            }
        } else {
            if ASCEND_IS_AIV {
                InitOutputSingleCore(output, lse, ratio);
            }
        }
        InitBlocks(scale, qc, qr, kv, kr, output, lse, workspace, pipe, ratio, layouts, kvIsTnd, pageTable, pagedLayout,
                   partial, softcap);
        FlashAttention();
        FreeBlocks();
    }

    // Pinned Process/FlashAttention: allocate once per core and keep the
    // three-task ring alive across every owned M/S2 fragment. Direct and FD
    // destinations are captured in each task, not rebound while it is pending.
    template <typename Reader>
    __aicore__ inline void
    RunAuto(const MlaAutoSplitBsndSchedule& schedule, const Reader& read, MlaWindowKernelParams windowParams,
            float scale, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
            TPipe* pipe, GM_ADDR spans, uint32_t totalQuery, const MlaBsndInputLayouts& layouts, bool kvIsTnd,
            float softcap, GM_ADDR pageTable = nullptr, const MlaPagedCacheLayout* pagedLayout = nullptr)
    {
        static_assert(IsPartial && (HasUsed || IsPaged));
        params = schedule.used.tiling;
        window = windowParams;
        queryTokens = IsTnd ? totalQuery : params.sq;
        lengths.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(spans));
        uint32_t ratio = 1;
        core = GetBlockIdx();
        if ASCEND_IS_AIV {
            ratio = GetSubBlockNum();
            core /= ratio;
        }
        if (core >= schedule.usedCoreNum)
            return;
        InitBlocks(scale, qc, qr, kv, kr, output, lse, workspace, pipe, ratio, &layouts, kvIsTnd, pageTable,
                   pagedLayout, nullptr, softcap, &schedule);

        const uint32_t prefix = auto_split_detail::Base<IsPaged>(params.batch) + kMlaNativePartitionWords;
        const uint32_t firstBatch = core ? schedule.bEnd[core - 1] : 0;
        const uint64_t first = ReadMetadataWide(read, prefix + 2 * firstBatch) + (core ? schedule.mEnd[core - 1] : 0);
        const uint64_t end = ReadMetadataWide(read, prefix + 2 * schedule.bEnd[core]) + schedule.mEnd[core] +
                             (schedule.nEnd[core] ? 1 : 0);
        const uint64_t last = Min(end, schedule.groups);
        RunInfo tasks[kTaskCount];
        uint64_t created = 0, executed = 0;
        uint32_t previousBatch = UINT32_MAX;
        for (uint64_t group = first; group < last; ++group) {
            MlaAutoSplitBsndTask fragment{};
            if (!AutoSplitBsndTask<IsPaged>(read, schedule, window, group, 0, fragment, core) || fragment.kind == 2)
                continue;
            if (fragment.batch != previousBatch) {
                LoadTaskLengths(fragment.batch);
                previousBatch = fragment.batch;
            }
            sStart = fragment.s2Begin;
            sEnd = fragment.s2End;
            for (uint32_t s = sStart; s < sEnd; ++s) {
                auto& info = tasks[created % kTaskCount];
                CalcParams(created, fragment.batch, fragment.m, s, info);
                info.isFd = fragment.kind == 1;
                info.partialRow = info.isFd ? fragment.slot * params.mBaseSize : 0;
                ++created;
                ExecuteTask(executed++, tasks);
            }
        }
        while (ShouldExecuteTask(tasks))
            ExecuteTask(executed++, tasks);
        FreeBlocks();
    }

  private:
    using RunInfo = Block::RunInfo;
    static constexpr uint32_t kTaskCount = 3;
    Block::MlaCube<InputT, IsTnd, HasStrides, IsPaged> cube;
    Block::MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial> vector;
    MlaDenseKernelSchedule params;
    MlaWindowKernelParams window;
    GlobalTensor<uint32_t> lengths;
    uint32_t actualKvLength = 0, actualQueryLength = 0;
    uint32_t queryTokens = 0, queryStart = 0;
    uint64_t kvStart = 0;
    const MlaTaskLengths* taskLengths = nullptr;
    uint32_t core, bStart, mStart, sStart, bEnd, mEnd, sEnd;
    uint32_t bn2IdxInCurCore = 0;

    __aicore__ inline void InitBlocks(float scale, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr, GM_ADDR output,
                                      GM_ADDR lse, GM_ADDR workspace, TPipe* pipe, uint32_t ratio,
                                      const MlaBsndInputLayouts* layouts, bool kvIsTnd, GM_ADDR pageTable,
                                      const MlaPagedCacheLayout* pagedLayout, const MlaSplitKvKernelParams* partial,
                                      float softcap, const MlaAutoSplitBsndSchedule* automatic = nullptr)
    {
        const uint64_t r1 = params.mm1ResSize, r2 = params.mm2ResSize;
        uint64_t offset = 0;
        GlobalTensor<float> qk, pv, vec2;
        GlobalTensor<InputT> p;
        GlobalTensor<int32_t> pvInt;
        qk.SetGlobalBuffer((__gm__ float*)(workspace + core * 2 * r1 * sizeof(float)));
        offset += GetBlockNum() * 2 * r1 * sizeof(float);
        p.SetGlobalBuffer((__gm__ InputT*)(workspace + offset + core * 2 * r1 * sizeof(InputT)));
        offset += GetBlockNum() * 2 * r1 * sizeof(InputT);
        pv.SetGlobalBuffer((__gm__ float*)(workspace + offset + core * 2 * r2 * sizeof(float)));
        pvInt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pv.GetPhyAddr(0)));
        offset += GetBlockNum() * 2 * r2 * sizeof(float);
        vec2.SetGlobalBuffer((__gm__ float*)(workspace + offset + core * 2 * r2 * sizeof(float)));

        Block::CubeParams cubeParams{queryTokens, params.heads, params.sk, params.mm1ResSize, params.mm2ResSize};
        if ASCEND_IS_AIC {
            cube.Init(pipe, cubeParams, qc, qr, kv, kr, layouts, kvIsTnd, pageTable, pagedLayout);
            cube.InitMm1GlobalTensor(qk);
            cube.InitMm2GlobalTensor(p, pv);
            cube.AllocEventID();
        }
        if ASCEND_IS_AIV {
            Block::VectorParams vectorParams{cubeParams, ratio, scale, softcap,
                                             IsCausal || (IsWindow && window.kind != 0)};
            vector.Init(vectorParams, output, lse);
            if constexpr (IsPartial) {
                if (automatic) {
                    vector.InitPartial(workspace + automatic->outOffset, workspace + automatic->sumOffset,
                                       workspace + automatic->maxOffset, 0);
                } else {
                    vector.InitPartial(workspace + partial->outOffsetBytes, workspace + partial->sumOffsetBytes,
                                       workspace + partial->maxOffsetBytes,
                                       AbsolutePartialOffsets ? 0 : uint64_t(core) * params.mBaseSize);
                }
            }
            vector.InitVec1GlobalTensor(qk, p, pvInt);
            vector.InitVec2GlobalTensor(vec2, pv);
            vector.InitBuffers(pipe);
            vector.AllocEventID();
        }
    }

    __aicore__ inline void FreeBlocks()
    {
        if ASCEND_IS_AIC {
            cube.FreeEventID();
        }
        if ASCEND_IS_AIV {
            vector.FreeEventID();
        }
    }

    __aicore__ inline void InitEmptyPartial(GM_ADDR workspace, const MlaSplitKvKernelParams& partial, uint32_t ratio)
    {
        // Empty fragments own their slot even when there are no Cube tasks.
        // Each vector initializes disjoint complete rows; order MTE3 writes
        // before recycling the initialization scratch.
        const uint32_t chunk = (params.mBaseSize + ratio - 1) / ratio;
        const uint32_t begin = (GetBlockIdx() % ratio) * chunk;
        if (begin >= params.mBaseSize) {
            return;
        }
        const uint32_t rows = Min(chunk, params.mBaseSize - begin);
        const uint64_t row = (AbsolutePartialOffsets ? 0 : uint64_t(core) * params.mBaseSize) + begin;
        GlobalTensor<float> gm;
        gm.SetGlobalBuffer((__gm__ float*)(workspace + partial.outOffsetBytes));
        matmul::InitOutput<float>(gm[row * 512], rows * 512, 0);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        gm.SetGlobalBuffer((__gm__ float*)(workspace + partial.sumOffsetBytes));
        matmul::InitOutput<float>(gm[row * 8], rows * 8, 0);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        gm.SetGlobalBuffer((__gm__ float*)(workspace + partial.maxOffsetBytes));
        matmul::InitOutput<float>(gm[row * 8], rows * 8, -float(1.0 / 0.0));
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void InitOutputSingleCore(GM_ADDR output, GM_ADDR lse, uint32_t ratio)
    {
        GlobalTensor<float> lseGm;
        lseGm.SetGlobalBuffer((__gm__ float*)lse);
        uint32_t event = 0;
        SetFlag<HardEvent::MTE3_V>(event);
        uint64_t total =
            IsTnd ? uint64_t(queryTokens) * params.heads : uint64_t(params.batch) * params.sq * params.heads;
        if constexpr (HasKvLengths) {
            bool needInit = IsCausal || IsWindow;
            if constexpr (!IsCausal && !IsWindow) {
                for (uint32_t b = 0; b < params.batch; ++b) {
                    needInit |= lengths.GetValue(kQueryLengths ? kSpanWords * b + kSpanWords - 1 : b) == 0;
                }
            }
            // Like source needInit, cover every output, including zero-KV
            // samples beyond the last compute endpoint. Never read their Q/KV.
            if (needInit) {
                const uint64_t outputSize = total * 512;
                const uint64_t single = (outputSize + params.usedCoreNum * ratio - 1) / (params.usedCoreNum * ratio);
                const uint64_t start = GetBlockIdx() * single;
                WaitFlag<HardEvent::MTE3_V>(event);
                if (start < outputSize) {
                    GlobalTensor<InputT> outGm;
                    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(output));
                    const uint64_t remaining = outputSize - start;
                    matmul::InitOutput<InputT>(outGm[start], remaining < single ? remaining : single, 0);
                }
                SetFlag<HardEvent::MTE3_V>(event);
            }
        }
        uint64_t single = (total + params.usedCoreNum * ratio - 1) / (params.usedCoreNum * ratio);
        uint64_t start = GetBlockIdx() * single;
        uint64_t tail = total - start;
        uint64_t count = tail < single ? tail : single;
        WaitFlag<HardEvent::MTE3_V>(event);
        if (start < total && count > 0) {
            matmul::InitOutput<float>(lseGm[start], count,
                                      (HasKvLengths && !((IsCausal || IsWindow) && params.sk == 0)) ? -float(1.0 / 0.0)
                                                                                                    : float(1.0 / 0.0));
        }
        SetFlag<HardEvent::MTE3_V>(event);
        WaitFlag<HardEvent::MTE3_V>(event);
        SyncAll();
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t b, uint32_t m, uint32_t s, RunInfo& info)
    {
        info.loop = loop;
        info.isValid = true;
        info.bIdx = b;
        info.n2Idx = 0;
        info.gS1Idx = m * params.mBaseSize;
        uint32_t rows = (kQueryLengths ? actualQueryLength : params.sq) * params.heads;
        if constexpr (IsTnd) {
            info.queryStart = queryStart;
            info.kvStart = kvStart;
        }
        info.actMBaseSize = params.mBaseSize;
        if ((m + 1) * params.mBaseSize > rows) {
            info.actMBaseSize = rows - m * params.mBaseSize;
        }
        info.actualSingleProcessSInnerSize = params.s2BaseSize;
        const uint32_t kvLength = HasKvLengths ? actualKvLength : params.sk;
        if ((s + 1) * params.s2BaseSize > kvLength) {
            info.actualSingleProcessSInnerSize = kvLength - s * params.s2BaseSize;
        }
        info.actualSingleProcessSInnerSizeAlign = Tile::Align(info.actualSingleProcessSInnerSize, 32U);
        // Whole-M partitions consume their visible range, which can start
        // above zero for band masks. Softmax/PV and ping-pong ownership must
        // restart on that first emitted tile, never on a skipped S2=0 tile.
        info.isFirstSInnerLoop = (loop == 0 || s == (IsPartial ? sStart : FirstS2(m)));
        if (info.isFirstSInnerLoop) {
            ++bn2IdxInCurCore;
        }
        info.bn2IdxInCurCore = bn2IdxInCurCore - 1;
        info.isLastS2Loop = (s + 1 == (IsPartial ? sEnd : S2Loops(m)));
        if constexpr (IsCausal) {
            info.nextToken = int64_t(kvLength) - (kQueryLengths ? actualQueryLength : params.sq);
        }
        if constexpr (IsWindow) {
            const int64_t delta = int64_t(kvLength) - (kQueryLengths ? actualQueryLength : params.sq);
            info.nextToken = window.kind == 0 ? INT32_MAX : delta + (window.kind == 2 ? window.right : 0);
            info.preToken = window.kind == 2 ? window.left - delta : INT32_MAX;
        }
        info.s2BatchOffset = s * params.s2BaseSize;
    }

    // Pinned CalcCurS2StartEndWithSparse, specialized to right-down causal/band.
    // The last query touched by this S1G block determines its union of keys.
    // Keep full S2 tiles; Vector still masks each row within the boundary tile.
    __aicore__ inline uint32_t S2Loops(uint32_t m)
    {
        const uint32_t kvLength = HasKvLengths ? actualKvLength : params.sk;
        if constexpr (IsWindow) {
            if (window.kind == 0) {
                return (kvLength + params.s2BaseSize - 1) / params.s2BaseSize;
            }
        }
        if constexpr (IsCausal || IsWindow) {
            const uint32_t sq = kQueryLengths ? actualQueryLength : params.sq;
            const uint64_t end = Min(uint64_t(m + 1) * params.mBaseSize, uint64_t(sq) * params.heads);
            if (end == 0 || kvLength == 0) {
                return 0;
            }
            const int64_t right = IsWindow && window.kind == 2 ? window.right : 0;
            const int64_t keys = int64_t((end - 1) / params.heads) + int64_t(kvLength) - sq + right + 1;
            return keys <= 0 ? 0
                             : (Min(uint64_t(keys), uint64_t(kvLength)) + params.s2BaseSize - 1) / params.s2BaseSize;
        }
        return (kvLength + params.s2BaseSize - 1) / params.s2BaseSize;
    }

    __aicore__ inline uint32_t FirstS2(uint32_t m)
    {
        if constexpr (IsWindow) {
            if (window.kind == 2) {
                const uint32_t sq = kQueryLengths ? actualQueryLength : params.sq;
                const int64_t first =
                    int64_t(uint64_t(m) * params.mBaseSize / params.heads) + actualKvLength - int64_t(sq) - window.left;
                return first <= 0 ? 0 : Min(uint64_t(first), uint64_t(actualKvLength)) / params.s2BaseSize;
            }
        }
        return 0;
    }

    __aicore__ inline void UpdateAxisInfo(uint32_t& b, uint32_t& m, uint32_t& s)
    {
        if constexpr (IsPartial) {
            ++s;
            return;
        }
        uint32_t mLoops =
            ((kQueryLengths ? actualQueryLength : params.sq) * params.heads + params.mBaseSize - 1) / params.mBaseSize;
        uint32_t sLoops = S2Loops(m);
        if (s + 1 < sLoops) {
            ++s;
            return;
        }
        s = 0;
        if (m + 1 < mLoops) {
            ++m;
            return;
        }
        m = 0;
        ++b;
    }

    __aicore__ inline bool ShouldExecuteTask(RunInfo tasks[kTaskCount])
    {
        for (uint32_t i = 0; i < kTaskCount; ++i) {
            if (tasks[i].isValid) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfo tasks[kTaskCount])
    {
        RunInfo& current = tasks[loop % kTaskCount];
        RunInfo& previous = tasks[(loop + 2) % kTaskCount];
        RunInfo& oldest = tasks[(loop + 1) % kTaskCount];
        if (current.isValid) {
            if ASCEND_IS_AIC {
                cube.ComputeMm1(current);
            }
        }
        if (previous.isValid) {
            if ASCEND_IS_AIV {
                vector.ProcessVec1L(previous);
            }
            if ASCEND_IS_AIC {
                cube.ComputeMm2(previous);
            }
        }
        if (oldest.isValid) {
            if ASCEND_IS_AIV {
                vector.ProcessVec2L(oldest);
            }
            oldest.isValid = false;
        }
    }

    __aicore__ inline void LoadTaskLengths(uint32_t b)
    {
        if (taskLengths) {
            queryStart = taskLengths->queryStart;
            kvStart = taskLengths->kvStart;
            actualQueryLength = taskLengths->queryLength;
            actualKvLength = taskLengths->kvLength;
        } else if constexpr (IsPaged) {
            const uint32_t start = kSpanWords * b;
            if constexpr (IsTnd) {
                // TND allocation starts are validated within int32.
                // Q-used limits rows; it never moves later samples.
                queryStart = lengths.GetValue(start);
            }
            actualQueryLength = lengths.GetValue(start + 6);
            actualKvLength = lengths.GetValue(start + 7);
        } else if constexpr (HasUsed) {
            const uint32_t start = kSpanWords * b;
            // TND query totals fit int32. BSND uses the existing
            // 64-bit b*Sq address in Block, equal to this span.
            queryStart = lengths.GetValue(start);
            kvStart = uint64_t(lengths.GetValue(start + 2)) | (uint64_t(lengths.GetValue(start + 3)) << 32);
            actualQueryLength = lengths.GetValue(start + 6);
            actualKvLength = lengths.GetValue(start + 7);
        } else if constexpr (IsTnd) {
            const uint32_t start = kSpanWords * b;
            queryStart = lengths.GetValue(start);
            kvStart = lengths.GetValue(start + 1);
            if constexpr (MixedKv) {
                kvStart |= uint64_t(lengths.GetValue(start + 2)) << 32;
            }
            actualQueryLength = lengths.GetValue(start + kSpanWords - 2);
            actualKvLength = lengths.GetValue(start + kSpanWords - 1);
        } else {
            actualKvLength = lengths.GetValue(b);
        }
    }

    __aicore__ inline void FlashAttention()
    {
        RunInfo tasks[kTaskCount];
        uint32_t b = bStart, m = mStart, s = sStart;
        uint64_t created = 0, executed = 0;
        uint32_t previousBatch = UINT32_MAX;
        bool dispatch = true, execute = false;
        while (dispatch || execute) {
            dispatch = (b != bEnd || m != mEnd || s != sEnd);
            if (dispatch) {
                if constexpr (HasKvLengths) {
                    if (b != previousBatch) {
                        LoadTaskLengths(b);
                        previousBatch = b;
                    }
                    if (actualKvLength == 0 || (kQueryLengths && actualQueryLength == 0)) {
                        // Preserve the source task-ring counters while skipping
                        // DEAL_ZERO/SKIP_ZERO. Pending tasks drain after dispatch.
                        UpdateAxisInfo(b, m, s);
                        continue;
                    }
                }
                if constexpr (IsCausal || IsWindow) {
                    if (S2Loops(m) == 0) {
                        // Initialization owns entirely masked blocks. Do not
                        // consume ring slots or change ping-pong counters.
                        UpdateAxisInfo(b, m, s);
                        continue;
                    }
                }
                if constexpr (IsWindow) {
                    // Axis endpoints stay at whole-M boundaries. Jump over
                    // the hidden prefix without creating tasks or advancing
                    // the ring counters; pending tasks retain their order.
                    s = Max(s, FirstS2(m));
                }
                CalcParams(created, b, m, s, tasks[created % kTaskCount]);
                ++created;
                UpdateAxisInfo(b, m, s);
            }
            execute = ShouldExecuteTask(tasks);
            if (execute) {
                ExecuteTask(executed++, tasks);
            }
        }
    }
};
struct MetadataReader {
    AscendC::GlobalTensor<uint32_t> data;
    __aicore__ inline uint32_t operator()(uint64_t word) const
    {
        return data.GetValue(word);
    }
};
} // namespace flash_attn::mla::Kernel
