/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Source partial orchestration and FD writeback: CANN ops-transformer
// ca18aee610729ca5270a44e2d84c751bc687fbf0, fia_kernel_nonquant_mla.h
// and fia_block_vec_nonquant_mla.h.
#pragma once
#include "mla_kernel.hpp"
#include "mla_metadata.h"
namespace flash_attn::mla::Kernel {
// Auto scheduling follows the pinned kernel's persistent per-core task ring.
// Only task coordinates and FD destinations change between its fragments.
template <typename InputT, bool IsTnd, bool IsPaged>
class MlaAutoSplit : private MlaTaskRing<InputT, true, IsTnd, false, !IsPaged, true, IsPaged, false, true, true, true> {
  public:
    template <typename Reader>
    __aicore__ inline void
    operator()(const MlaAutoSplitBsndSchedule& s, const Reader& read, MlaWindowKernelParams window, float scale,
               GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
               TPipe* pipe, GM_ADDR spans, uint32_t totalQuery, const MlaBsndInputLayouts& layouts, bool kvIsTnd,
               float softcap, GM_ADDR pageTable = nullptr, const MlaPagedCacheLayout* pagedLayout = nullptr)
    {
        this->RunAuto(s, read, window, scale, qc, qr, kv, kr, output, lse, workspace, pipe, spans, totalQuery, layouts,
                      kvIsTnd, softcap, pageTable, pagedLayout);
    }
};

// GQA changes validated head/length coordinates only. The existing partial
// task ring, Cube/Vector arithmetic, PA Copy and event ordering are reused.
template <typename InputT, bool IsTnd, bool Paged>
class MlaGqaPartial : private MlaTaskRing<InputT, true, IsTnd, false, !Paged, true, Paged, false, true, true, true> {
  public:
    __aicore__ inline void RunFragment(const MlaDenseKernelSchedule& s, const MlaSplitKvKernelParams& p,
                                       const MlaTaskLengths& lengths, float scale, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv,
                                       GM_ADDR kr, GM_ADDR workspace, TPipe* pipe, const MlaBsndInputLayouts& layouts,
                                       bool kvIsTnd, const MlaPagedCacheInputs& cache, float softcap)
    {
        this->Run(s, scale, qc, qr, kv, kr, workspace, workspace, workspace, pipe, nullptr, s.sq, &layouts, kvIsTnd,
                  reinterpret_cast<GM_ADDR>(cache.table.address), &cache.layout, p.window, &p, softcap, &lengths);
    }
};
} // namespace flash_attn::mla::Kernel
