/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once
#include "mla_dispatch.h"
#include "mla_kernel.hpp"
#include "mla_split_kv_kernel.hpp"
#include "mla_split_kv_merge_block.hpp"
namespace flash_attn::mla {
using MlaReader = Kernel::MetadataReader;
template <typename DType>
__aicore__ inline bool LoadMla(const MlaLaunchArgs& a, MlaReader& read, MlaLayout& l, MlaAutoSplitBsndSchedule& s)
{
    const uint32_t dtype =
        AscendC::IsSameType<DType, half>::value ? uint32_t(MlaMetadataDtype::Fp16) : uint32_t(MlaMetadataDtype::Bf16);
    if (!LoadCheckedAutoSplitSchedule(a.checked, dtype, s))
        return false;
    // AICPU has checked geometry, allocation and workspace once. Numerical
    // cores only bind that invocation-local result to the source task ring.
    const auto& c = a.context;
    const auto& t = s.used.tiling;
    l.m = t.mBaseSize;
    l.n = t.s2BaseSize;
    l.cores = t.blockDim;
    l.g = c.packed ? c.heads / c.kvHeads : 1;
    l.headGroups = c.heads / l.g;
    l.capacity = c.splits ? c.splits : l.cores;
    l.groups = s.groups;
    l.outOffset = s.outOffset;
    l.sumOffset = s.sumOffset;
    l.maxOffset = s.maxOffset;
    l.bytes = s.bytes;
    read.data.SetGlobalBuffer((__gm__ uint32_t*)a.metadata);
    return true;
}
template <typename T>
__aicore__ inline void InitializeMla(uint64_t address, uint64_t count, T value)
{
    if (!address)
        return;
    const uint64_t cores = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
    const uint64_t chunk = (count + cores * 64 - 1) / (cores * 64) * 64;
    const uint64_t first = uint64_t(AscendC::GetBlockIdx()) * chunk;
    if (first < count) {
        AscendC::GlobalTensor<T> data;
        data.SetGlobalBuffer((__gm__ T*)address);
        matmul::InitOutput<T>(data[first], chunk < count - first ? chunk : count - first, value);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }
}
template <typename DType>
__aicore__ inline void InitializeMlaOutputs(const MlaLaunchArgs& a)
{
    if ASCEND_IS_AIV {
        const auto& c = a.context;
        const uint64_t rows = (MlaTnd(c) ? uint64_t(c.totalQ) : uint64_t(c.batch) * c.sq) * c.heads;
        InitializeMla<DType>(a.out, rows * c.valueDim, DType(0));
        InitializeMla<float>(a.lse, rows, !c.pageSize && MlaEmptyKv(c) ? float(1.0 / 0.0) : -float(1.0 / 0.0));
        // FA3 leaves partial O undefined wherever partial LSE is -inf.
        if (c.splits != 1)
            InitializeMla<float>(a.accumLse, rows * (c.splits ? c.splits : 26), -float(1.0 / 0.0));
    }
}
} // namespace flash_attn::mla
