// Copyright (c) 2026. Single-token mixed KV decode for interleaved BBND pages.
#include "fwd_dispatch.hpp"
#include "fwd_kernel.cpp"

template<bool ReturnLse>
CATLASS_GLOBAL void FAInferPairedDecode(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR pages, GM_ADDR o, GM_ADDR lse,
    GM_ADDR cuQ, GM_ADDR lengths, GM_ADDR workspace, GM_ADDR tiling, float scale, uint32_t numBlocks,
    uint32_t maxBlocks, FwdStorageParams storage, GM_ADDR kDescale, GM_ADDR vDescale)
{
    RunFAInfer<false, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND,
        MaskCategory::NO_MASK, CacheLayout::nd, ReturnLse, false, true, true, false, true>(
        q, k, v, nullptr, pages, o, lse, cuQ, lengths, nullptr, nullptr, workspace, tiling, scale, 0.0F,
        numBlocks, maxBlocks, 0, storage, kDescale, vDescale);
}

template<bool ReturnLse>
static void LaunchPairedDecode(const FwdLaunchArgs& a)
{
    FAInferPairedDecode<ReturnLse><<<a.block_dim, nullptr, a.stream>>>(
        a.q_device, a.k_device, a.v_device, a.block_table_device, a.o_device, a.lse_device,
        a.q_seq_device, a.kv_seq_device, a.workspace_device, a.tiling_device, a.scale_value, a.num_blocks,
        a.max_num_blocks_per_batch, a.storage, a.k_descale_device, a.v_descale_device);
}

void launch_paired_decode(const FwdLaunchArgs& a)
{
    if (a.lse_mode) {
        LaunchPairedDecode<true>(a);
    } else {
        LaunchPairedDecode<false>(a);
    }
}
