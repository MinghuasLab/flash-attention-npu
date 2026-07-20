/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef CSRC_ASCENDC950_FLASH_ATTN_NPU_V3_TILING_FROM_TENSORS_HPP
#define CSRC_ASCENDC950_FLASH_ATTN_NPU_V3_TILING_FROM_TENSORS_HPP

#include <cstdint>
#include <vector>

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

namespace flash_attn_npu_950_v3 {
struct SeqlenScratch {
    std::vector<int32_t> q;   // Q seqlen list (cumulative for TND, per-batch for BSND)
    std::vector<int32_t> kv;  // KV seqlen list (per-batch for paged/BSND, cumulative for TND+non-paged)
};

// Decumulate `cu` (size = batch + 1, int32) into per-batch seqlen
// (size = batch, int32). Used for varlen Q where PyTorch only hands us
// the cumulative-sum form.
inline std::vector<int32_t> decumulate_int32(const at::Tensor& cu_int32_cpu,
                                             int batch_size)
{
    TORCH_CHECK(cu_int32_cpu.dtype() == at::kInt,
                "decumulate_int32: expected int32 tensor");
    TORCH_CHECK(cu_int32_cpu.numel() == batch_size + 1,
                "decumulate_int32: expected ",
                batch_size + 1, " elements, got ", cu_int32_cpu.numel());
    const int32_t* p = cu_int32_cpu.data_ptr<int32_t>();
    std::vector<int32_t> out(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        out[i] = p[i + 1] - p[i];
    }
    return out;
}

// Copy seqused-style int32 tensor (size = batch) into an int32 vector.
// The 950 FAInferContext consumes int32 seqlen lists; seqused_k already
// arrives as int32 from PyTorch, so no int32->int64 widening is needed.
inline std::vector<int32_t> copy_int32_vec(const at::Tensor& int32_cpu,
                                           int batch_size)
{
    TORCH_CHECK(int32_cpu.dtype() == at::kInt,
                "copy_int32_vec: expected int32 tensor");
    TORCH_CHECK(int32_cpu.numel() == batch_size,
                "copy_int32_vec: expected ",
                batch_size, " elements, got ", int32_cpu.numel());
    const int32_t* p = int32_cpu.data_ptr<int32_t>();
    std::vector<int32_t> out(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        out[i] = p[i];
    }
    return out;
}

// Copy cumulative cu_seqlens int32 tensor (size = batch + 1) into an int32
// vector WITHOUT decumulating. The 950 tiling code expects cumulative form
// for TND layout (see FAInferTiling::FillSplitCoreTilingData).
inline std::vector<int32_t> copy_cu_seqlens_int32(const at::Tensor& cu_int32_cpu,
                                                  int batch_size)
{
    TORCH_CHECK(cu_int32_cpu.dtype() == at::kInt,
                "copy_cu_seqlens_int32: expected int32 tensor");
    TORCH_CHECK(cu_int32_cpu.numel() == batch_size + 1,
                "copy_cu_seqlens_int32: expected ",
                batch_size + 1, " elements, got ", cu_int32_cpu.numel());
    const int32_t* p = cu_int32_cpu.data_ptr<int32_t>();
    std::vector<int32_t> out(batch_size + 1);
    for (int i = 0; i < batch_size + 1; ++i) {
        out[i] = p[i];
    }
    return out;
}

// Build cumulative seqlen list (batch+1 elements) from per-batch
// seqlens. Used for kvSeqlenList in TND + non-paged mode where we only
// have per-batch seqused_k but the tiling code expects cumulative form.
inline std::vector<int32_t> cumulate_int32(const std::vector<int32_t>& per_batch)
{
    std::vector<int32_t> out(per_batch.size() + 1);
    out[0] = 0;
    for (size_t i = 0; i < per_batch.size(); ++i) {
        out[i + 1] = out[i] + per_batch[i];
    }
    return out;
}

inline void fill_inference_context(
    optiling::FAInferContext& ctx,
    SeqlenScratch& scratch,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor* cu_seqlens_q_cpu_int32,   // nullable: !is_varlen_q
    const at::Tensor* seqused_k_cpu_int32,      // nullable: caller must pass per-batch seqlen
    bool paged_KV,
    int  page_block_size,
    int  num_blocks,
    int  max_num_blocks_per_seq,
    bool is_causal,
    bool is_varlen_q,
    bool is_bf16,
    int  batch_size,
    int  seqlen_q,
    int  num_heads,
    int  num_heads_k,
    int  head_size_q,
    int  head_size_v,
    float softmax_scale,
    bool lse_flag,
    const std::string& layout_str)              // "TND" | "BSND"
{
    const bool is_tnd = (layout_str == "TND");

    if (is_varlen_q) {
        TORCH_CHECK(cu_seqlens_q_cpu_int32 != nullptr,
                    "fill_inference_context: varlen Q requires cu_seqlens_q");
        if (is_tnd) {
            scratch.q = copy_cu_seqlens_int32(*cu_seqlens_q_cpu_int32, batch_size);
        } else {
            scratch.q = decumulate_int32(*cu_seqlens_q_cpu_int32, batch_size);
        }
    } else {
        if (is_tnd) {
            scratch.q.resize(batch_size + 1);
            scratch.q[0] = 0;
            for (int i = 0; i < batch_size; ++i) {
                scratch.q[i + 1] = scratch.q[i] + seqlen_q;
            }
        } else {
            scratch.q.assign(batch_size, seqlen_q);
        }
    }
    TORCH_CHECK(seqused_k_cpu_int32 != nullptr,
                "fill_inference_context: per-batch KV seqlen (seqused_k / cache_seqlens) is required");
    auto kv_per_batch = copy_int32_vec(*seqused_k_cpu_int32, batch_size);

    if (is_tnd && !paged_KV) {
        scratch.kv = cumulate_int32(kv_per_batch);
    } else {
        scratch.kv = std::move(kv_per_batch);
    }

    ctx.batch = batch_size;
    ctx.numHeads = num_heads;
    ctx.kvHeads = num_heads_k;
    ctx.embeddingSize = head_size_q;
    ctx.embeddingSizeV = head_size_v;
    ctx.numBlocks = paged_KV ? num_blocks : 0;
    ctx.blockSize = paged_KV ? page_block_size : 0;
    ctx.maxNumBlocksPerBatch = paged_KV ? max_num_blocks_per_seq : 0;

    int64_t maxQ = 0, maxKV = 0;
    if (is_tnd) {
        for (int i = 0; i < batch_size; ++i) {
            int64_t qLen = scratch.q[i + 1] - scratch.q[i];
            if (qLen > maxQ) maxQ = qLen;
        }
    } else {
        for (int i = 0; i < batch_size; ++i) {
            if (scratch.q[i] > maxQ) maxQ = scratch.q[i];
        }
    }
    if (is_tnd && !paged_KV) {
        for (int i = 0; i < batch_size; ++i) {
            int64_t kvLen = scratch.kv[i + 1] - scratch.kv[i];
            if (kvLen > maxKV) maxKV = kvLen;
        }
    } else {
        for (int i = 0; i < batch_size; ++i) {
            if (scratch.kv[i] > maxKV) maxKV = scratch.kv[i];
        }
    }
    ctx.maxQSeqlen  = maxQ;
    ctx.maxKvSeqlen = maxKV;
    ctx.qSeqlenList  = scratch.q.data();
    ctx.kvSeqlenList = scratch.kv.data();

    ctx.scaleValue = softmax_scale;
    ctx.maskType = is_causal ? optiling::MaskType::MASK_SPEC
                                    : optiling::MaskType::NO_MASK;
    ctx.dataType = is_bf16 ? optiling::DataType::BF16
                                  : optiling::DataType::FP16;
    ctx.pagedCacheFlag = paged_KV;
    ctx.lseFlag = lse_flag;
    ctx.cacheLayout = "nd"; // ND only; 
    ctx.layout = layout_str;    // "TND" | "BSND"

    ctx.innerPrecise = 0;   // fixed FP32 accumulation
    ctx.isTilingSink = false;
    ctx.learnableSinkFlag = false;
    ctx.flashDecodeFlag = false;    // disables FlashDecode multi-split
    ctx.kvcacheNzFlag = false;
    ctx.pagedShapeFlag = false;      
    ctx.preToken = 0;
    ctx.nextToken = 0;
    ctx.sparseMode = 0;
    // ctx.globalWindowSize   = 4;  // SWA defaults; kernel reads them only when SWA is on
    // ctx.localWindowSize    = 0;
    ctx.numTokens = 0;  // not used by DoTiling on this path
}

}  // namespace flash_attn_npu_950_v3

#endif  // CSRC_ASCENDC950_FLASH_ATTN_NPU_V3_TILING_FROM_TENSORS_HPP
