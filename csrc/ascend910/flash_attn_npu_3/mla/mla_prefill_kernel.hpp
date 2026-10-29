/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Source: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0,
// arch22/fia_kernel_nonquant.h::ExecuteTask and workspace layout.
// One persistent three-task ring per core, with the source's global ping-pong
// parity across S1G groups. The caller supplies checked invocation coordinates.
#pragma once
#include "mla_prefill_vector_block.hpp"
namespace flash_attn::mla::Kernel {
using namespace AscendC;
template <typename InputT, bool Brc, bool Partial, bool Paged = false>
class MlaPrefill {
  public:
    __aicore__ inline void Run(const MlaPrefillParams& p, const MlaPrefillTask& t, uint32_t core, GM_ADDR q, GM_ADDR k,
                               GM_ADDR v, GM_ADDR out, GM_ADDR lse, GM_ADDR workspace, GM_ADDR partial, GM_ADDR sums,
                               GM_ADDR maxima, TPipe* pipe, const MlaPagedCacheInputs* cache = nullptr)
    {
        if (t.begin == t.end) {
            if ASCEND_IS_AIV {
                InitEmpty(p, t, out, lse, partial, sums, maxima, pipe);
            }
            return;
        }
        // GQA also calls this single-fragment interface with rebased inputs.
        Init(p, core, q, k, v, out, lse, workspace, partial, sums, maxima, pipe, cache);
        Block::PrefillTaskContext context{};
        context.task = t;
        context.cacheBatch = t.batch;
        context.partial = Partial;
        RunTask(context);
        Finish();
    }

    __aicore__ inline void Init(const MlaPrefillParams& p, uint32_t core, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                                GM_ADDR lse, GM_ADDR workspace, GM_ADDR partial, GM_ADDR sums, GM_ADDR maxima,
                                TPipe* pipe, const MlaPagedCacheInputs* cache = nullptr)
    {
        s2BaseSize = p.nBase;
        const uint64_t r1 = uint64_t(p.mBase) * Tile::Align(p.nBase, 32), r2 = uint64_t(p.mBase) * 128;
        GlobalTensor<float> qk, pv, previous;
        GlobalTensor<InputT> prob;
        qk.SetGlobalBuffer((__gm__ float*)(workspace + core * 2 * r1 * 4));
        prob.SetGlobalBuffer((__gm__ InputT*)(workspace + p.cores * 2 * r1 * 4 + core * 2 * r1 * 2));
        pv.SetGlobalBuffer((__gm__ float*)(workspace + p.cores * 2 * r1 * 6 + core * 2 * r2 * 4));
        previous.SetGlobalBuffer((__gm__ float*)(workspace + p.cores * (2 * r1 * 6 + 2 * r2 * 4) + core * 2 * r2 * 4));
        if ASCEND_IS_AIC {
            cube.Init(p, pipe, q, k, v, cache);
            cube.InitMm1GlobalTensor(qk);
            cube.InitMm2GlobalTensor(prob, pv);
            cube.AllocEventID();
        }
        if ASCEND_IS_AIV {
            vector.Init(p, pipe, qk, prob, pv, previous, out, lse, partial, sums, maxima);
        }
    }

    __aicore__ inline void RunTask(const Block::PrefillTaskContext& context)
    {
        const auto& t = context.task;
        if (t.begin == t.end) {
            if ASCEND_IS_AIV {
                vector.InitEmpty(context);
            }
            return;
        }
        const uint32_t count = t.end - t.begin;
        for (uint32_t s2 = 0; s2 < count; ++s2) {
            auto& now = ring[loop % 3];
            now = {};
            now.context = context;
            now.loop = loop;
            now.bIdx = t.batch;
            now.n2Idx = t.kvHead;
            now.gS1Idx = t.rowStart;
            now.s2Idx = t.begin + s2;
            now.actMBaseSize = t.rows;
            now.actualSingleProcessSInnerSize = Min(s2BaseSize, t.kvLength - now.s2Idx * s2BaseSize);
            now.actualSingleProcessSInnerSizeAlign = Tile::Align(now.actualSingleProcessSInnerSize, 32);
            now.isFirstSInnerLoop = s2 == 0;
            now.isLastS2Loop = s2 + 1 == count;
            now.isValid = true;
            ExecuteTask();
        }
    }

    __aicore__ inline void Finish()
    {
        // Drain only at the core endpoint, never between independent S1G tasks.
        for (uint32_t i = 0; i < 2; ++i) {
            ring[loop % 3].isValid = false;
            ExecuteTask();
        }
        if ASCEND_IS_AIC {
            cube.FreeEventID();
        }
    }

  private:
    __aicore__ inline void ExecuteTask()
    {
        auto& now = ring[loop % 3];
        auto& prev = ring[(loop + 2) % 3];
        auto& oldest = ring[(loop + 1) % 3];
        if (now.isValid) {
            if ASCEND_IS_AIC {
                cube.ComputeMm1(now);
            }
        }
        if (prev.isValid) {
            if ASCEND_IS_AIV {
                vector.ComputeVec1(prev);
            }
            if ASCEND_IS_AIC {
                cube.ComputeMm2(prev);
            }
        }
        if (oldest.isValid) {
            if ASCEND_IS_AIV {
                vector.ComputeVec2(oldest);
            }
            oldest.isValid = false;
        }
        ++loop;
    }
    __aicore__ inline void InitEmpty(const MlaPrefillParams& p, const MlaPrefillTask& t, GM_ADDR out, GM_ADDR lse,
                                     GM_ADDR partial, GM_ADDR sums, GM_ADDR maxima, TPipe* pipe)
    {
        // Disjoint rows for the two vector subcores; no Cube/Vector handshake.
        const uint32_t g = p.heads / p.kvHeads;
        TBuf<> buf;
        pipe->InitBuffer(buf, 1024);
        auto zero = buf.Get<float>(), neg = zero[128];
        Duplicate(zero, 0.0f, 128);
        Duplicate(neg, -float(1.0 / 0.0), 8);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        GlobalTensor<float> po, su, ma, sl;
        GlobalTensor<InputT> o;
        po.SetGlobalBuffer((__gm__ float*)partial);
        su.SetGlobalBuffer((__gm__ float*)sums);
        ma.SetGlobalBuffer((__gm__ float*)maxima);
        sl.SetGlobalBuffer((__gm__ float*)lse);
        o.SetGlobalBuffer((__gm__ InputT*)out);
        for (uint32_t r = GetSubBlockIdx(); r < t.rows; r += 2) {
            if constexpr (Partial) {
                DataCopy(po[r * 128], zero, 128);
                DataCopy(su[r * 8], zero, 8);
                DataCopy(ma[r * 8], neg, 8);
            } else {
                const uint32_t flat = t.rowStart + r, h = t.kvHead * g + flat % g;
                DataCopy(o[((uint64_t(t.batch) * p.sq + flat / g) * p.heads + h) * 128], zero.ReinterpretCast<InputT>(),
                         128);
                DataCopyPad(sl[(uint64_t(t.batch) * p.heads + h) * p.sq + flat / g], neg,
                            DataCopyExtParams{1, 4, 0, 0, 0});
            }
        }
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
    Block::MlaPrefillCube<InputT, Paged> cube;
    Block::MlaPrefillVector<InputT, Brc, Partial> vector;
    Block::PrefillInfo ring[3]{};
    uint64_t loop = 0;
    uint32_t s2BaseSize = 0;
};
} // namespace flash_attn::mla::Kernel
