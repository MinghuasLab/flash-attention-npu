#include <torch/extension.h>

#include "mha_fwd_kvcache.cpp"
#include "tilingdata.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"
#include "runtime/rt_ffts.h"
#include "kernel_common.hpp"
#include "kernel_operator.h"
#include "tiling/platform/platform_ascendc.h"
#include "fag_tiling.cpp"
#include "fag_kernel.cpp"

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    uint32_t qRowNumCeil = Q_TILE_CEIL;
    uint32_t qNBlockTile = (qSeqlen != 0) ?
        (qRowNumCeil / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
    qNBlockTile = std::min(qNBlockTile, groupSize);
    qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
    return qNBlockTile;
}

uint32_t GetQSBlockTile(int64_t kvSeqlen)
{
    uint32_t qSBlockTile = Q_TILE_CEIL;
    return qSBlockTile;
}

std::vector<at::Tensor>
mha_fwd(at::Tensor q,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
                at::Tensor k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
                at::Tensor v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table.
                std::optional<at::Tensor> k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
                std::optional<at::Tensor> v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
                std::optional<at::Tensor> q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
                std::optional<at::Tensor> out_,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
                std::optional<at::Tensor> cu_seqlens_q_,  // b+1
                std::optional<at::Tensor> cu_seqlens_k_,  // b+1
                std::optional<at::Tensor> cu_seqlens_k_new_,  // b+1
                std::optional<at::Tensor> seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
                std::optional<at::Tensor> seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
                std::optional<int64_t> max_seqlen_q_,
                // TODO: check if we need max_seqlen_k
                std::optional<int64_t> max_seqlen_k_,
                std::optional<at::Tensor> page_table_, // (b_k, max_num_pages_per_seq)
                std::optional<at::Tensor> kv_batch_idx_, // b. indices to index into the KV cache
                std::optional<at::Tensor> leftpad_k_, // b
                std::optional<at::Tensor> rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                std::optional<at::Tensor> rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                std::optional<at::Tensor> seqlens_rotary_, // b
                std::optional<at::Tensor> q_descale_,  // (b, h_k), not (b, h)
                std::optional<at::Tensor> k_descale_,  // (b, h_k)
                std::optional<at::Tensor> v_descale_,  // (b, h_k)
                std::optional<float> softmax_scale_,
                bool is_causal,
                int64_t window_size_left,
                int64_t window_size_right,
                int64_t attention_chunk,
                float softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                std::optional<at::Tensor> scheduler_metadata_,  // (b + 1)
                int64_t num_splits,
                std::optional<bool> pack_gqa_,
                int64_t sm_margin
                )
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    auto q_dtype = q.dtype();
    bool is_bf16 = q_dtype == torch::kBFloat16;
    bool is_fp16 = q_dtype == torch::kFloat16;
    TORCH_CHECK(is_bf16 || is_fp16, "FlashAttention only supports FP16 and BF16 data types");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor q must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor k must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor v must have contiguous last dimension");
    at::Tensor tiling_cpu_tensor = at::empty({1024}, at::device(c10::kCPU).dtype(at::kByte));

    FAInferTilingData* tiling_cpu_ptr = reinterpret_cast<FAInferTilingData*>(tiling_cpu_tensor.data_ptr<uint8_t>());
    uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    at::Tensor seqlens_k, block_table, out;
    at::Tensor k_, v_, rotary_cos, rotary_sin, cache_batch_idx, alibi_slopes;

    at::Tensor cu_seqlens_q, cu_seqlens_k;
    at::Tensor out_accum, softmax_lse_accum;
    float softmax_scale;

    const bool paged_KV = page_table_.has_value();
    const bool is_varlen_q = cu_seqlens_q_.has_value();
    const bool is_varlen_kv = cu_seqlens_k_.has_value();

    if (paged_KV) {
        auto page_table = page_table_.value();
        TORCH_CHECK(page_table.dtype() == torch::kInt32, "page_table must have dtype int32");
        TORCH_CHECK(page_table.stride(-1) == 1, "page_table must have contiguous last dimension");
    }

    if (is_varlen_q) {
        cu_seqlens_q = cu_seqlens_q_.value();
        CHECK_CONTIGUOUS(cu_seqlens_q);
        TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
        TORCH_CHECK(max_seqlen_q_.has_value(), "max_seqlen_q must be provided if cu_seqlens_q is provided");
    }

    if (is_varlen_kv) {
        cu_seqlens_k = cu_seqlens_k_.value();
        CHECK_CONTIGUOUS(cu_seqlens_k);
        TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");
        TORCH_CHECK(!paged_KV, "If cu_seqlens_k is passed in, then paged table is not supported");
    }

    if (seqused_k_.has_value()) {
        seqlens_k = seqused_k_.value();
        TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");
    }

    TORCH_CHECK(!leftpad_k_.has_value(), "NPU FlashAttention does not support leftpad_k");
    TORCH_CHECK(!rotary_cos_.has_value(), "NPU FlashAttention does not support rotary embedding");
    TORCH_CHECK(!rotary_sin_.has_value(), "NPU FlashAttention does not support rotary embedding");
    TORCH_CHECK(!seqlens_rotary_.has_value(), "NPU FlashAttention does not support seqlens_rotary");
    TORCH_CHECK(!q_descale_.has_value(), "NPU FlashAttention does not support q_descale");
    TORCH_CHECK(!k_descale_.has_value(), "NPU FlashAttention does not support k_descale");
    TORCH_CHECK(!v_descale_.has_value(), "NPU FlashAttention does not support v_descale");
    TORCH_CHECK(softcap == 0.0f, "NPU FlashAttention does not support softcap");
    TORCH_CHECK(window_size_left == -1, "NPU FlashAttention does not support window_size_left");
    TORCH_CHECK(window_size_right == -1, "NPU FlashAttention does not support window_size_right");
    TORCH_CHECK(attention_chunk == 0, "NPU FlashAttention does not support attention_chunk");
    TORCH_CHECK(!scheduler_metadata_.has_value(), "NPU FlashAttention does not support scheduler_metadata");
    TORCH_CHECK(num_splits == 1 || num_splits == 0, "NPU FlashAttention only supports num_splits=1 or num_splits=0");
    TORCH_CHECK(!pack_gqa_.has_value() || !pack_gqa_.value(), "NPU FlashAttention does not support pack_gqa");

    if (is_varlen_kv) {
        cu_seqlens_k = cu_seqlens_k_.value();
        TORCH_CHECK(!paged_KV, "If cu_seqlens_k is passed in, then paged table is not supported");
    }
    if (k_new_.has_value()) {
        k_ = k_new_.value();
    }
    if (v_new_.has_value()) {
        v_ = v_new_.value();
    }
    if (rotary_cos_.has_value()) {
        rotary_cos = rotary_cos_.value();
    }
    if (rotary_sin_.has_value()) {
        rotary_sin = rotary_sin_.value();
    }
    if (kv_batch_idx_.has_value()) {
        cache_batch_idx = kv_batch_idx_.value();
    }
    if (paged_KV) {
        block_table = page_table_.value();
    }
    if (softmax_scale_.has_value()) {
        softmax_scale = softmax_scale_.value();
    }
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "output must have the same dtype as inputs");
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    }  else {
        out = torch::empty_like(q);
    }
    const auto sizes = q.sizes();
    
    int batch_size = 0;
    int seqlen_q = 0;
    int num_heads = 0;
    int head_size_og = 0;
    if (is_varlen_q) {
        batch_size = cu_seqlens_q.size(0) - 1;
        seqlen_q = static_cast<int>(max_seqlen_q_.value());
        num_heads = sizes[1];
        head_size_og = sizes[2];
    } else {
        batch_size = sizes[0];
        seqlen_q = sizes[1];
        num_heads = sizes[2];
        head_size_og = sizes[3];
    }
    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : k.size(0);
    const int page_block_size = !paged_KV ? 128 : k.size(1);
    const int num_heads_k = k.dim() == 3 ? k.size(1) : k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    at::Tensor seqlenk_cpu_tensor = seqlens_k.to(at::Device(at::kCPU));
    int32_t* seqlens_k_cpu = static_cast<int32_t *>(seqlenk_cpu_tensor.data_ptr());
    int32_t* cu_seqlen_q_cpu = nullptr;
    at::Tensor cu_seqlen_q_cpu_tensor;
    if (is_varlen_q) {
        cu_seqlen_q_cpu_tensor = cu_seqlens_q.to(at::Device(at::kCPU));
        cu_seqlen_q_cpu = static_cast<int32_t *>(cu_seqlen_q_cpu_tensor.data_ptr());
    }
    tiling_cpu_ptr->set_batch(static_cast<uint32_t>(batch_size));
    tiling_cpu_ptr->set_numHeads(static_cast<uint32_t>(num_heads));
    tiling_cpu_ptr->set_kvHeads(static_cast<uint32_t>(num_heads_k));
    tiling_cpu_ptr->set_embeddingSize(static_cast<uint32_t>(head_size_og));
    tiling_cpu_ptr->set_embeddingSizeV(static_cast<uint32_t>(head_size_og));
    tiling_cpu_ptr->set_numBlocks(static_cast<uint32_t>(num_blocks));
    tiling_cpu_ptr->set_blockSize(static_cast<uint32_t>(page_block_size));
    tiling_cpu_ptr->set_maxNumBlocksPerBatch(static_cast<uint32_t>(max_num_blocks_per_seq));
    tiling_cpu_ptr->set_maskType(static_cast<uint32_t>(is_causal));
    tiling_cpu_ptr->set_scaleValue(softmax_scale);
    tiling_cpu_ptr->set_maxQSeqlen(seqlen_q);
    uint64_t WORKSPACE_BLOCK_SIZE_DB = 128 * 512;
    uint64_t PRELANCH_NUM = 3;
    uint64_t mm1OutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB *
        4 * PRELANCH_NUM;
    uint64_t smOnlineOutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB *
        2 * PRELANCH_NUM;
    uint64_t mm2OutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB *
        4 * PRELANCH_NUM;
    uint64_t UpdateSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB *
        4 * PRELANCH_NUM;
    int64_t workSpaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize;

    
    at::Tensor workspace_tensor = at::empty({workSpaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte));
    at::Tensor softmaxlse = at::empty({batch_size, seqlen_q, num_heads}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    if (is_varlen_q) {
        softmaxlse = at::empty({sizes[0], num_heads}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    }
    softmaxlse.fill_(std::numeric_limits<float>::infinity());
    tiling_cpu_ptr->set_mm1OutSize(mm1OutSize);
    tiling_cpu_ptr->set_smOnlineOutSize(smOnlineOutSize);
    tiling_cpu_ptr->set_mm2OutSize(mm2OutSize);
    tiling_cpu_ptr->set_UpdateSize(UpdateSize);
    tiling_cpu_ptr->set_workSpaceSize(workSpaceSize);

    uint32_t totalTaskNum = 0;
    uint32_t groupSize = num_heads / num_heads_k;
    for (int32_t batchIdx = 0; batchIdx < batch_size; batchIdx++) {
        uint64_t qSeqlen = seqlen_q;
        if (is_varlen_q) {
            qSeqlen = *(cu_seqlen_q_cpu + batchIdx + 1) - *(cu_seqlen_q_cpu + batchIdx);
        }
        uint64_t kvSeqlen = *(seqlens_k_cpu + batchIdx);
        uint64_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
        uint64_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
        uint64_t curQNBlockNum = qNBlockNumPerGroup * num_heads_k;
        uint64_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
        uint64_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
        uint64_t curTaskNum = curQNBlockNum * curQSBlockNum;
        if (batchIdx == 0) {
            tiling_cpu_ptr->set_firstBatchTaskNum(curTaskNum);
        }
        totalTaskNum += curTaskNum;
    }
    tiling_cpu_ptr->set_totalTaskNum(totalTaskNum);
    at::Tensor mask_gpu_tensor;
    if (is_causal) {
        at::Tensor mask_cpu_tensor = at::empty({2048, 2048}, at::device(c10::kCPU).dtype(at::kByte));
        mask_cpu_tensor = at::triu(at::ones_like(mask_cpu_tensor), 1);
        mask_gpu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
    }
    at::Tensor tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));
    at::Tensor seqlenk_gpu_tensor;
    at::Tensor seqlenq_gpu_tensor;
    if (is_varlen_q) {
        seqlenq_gpu_tensor = cu_seqlens_q;
    } else {
        seqlenq_gpu_tensor = at::empty({0}, at::device(at::kPrivateUse1).dtype(at::kInt));
    }
    if (is_varlen_kv) {
        seqlenk_gpu_tensor = cu_seqlens_k;
    } else {
        seqlenk_gpu_tensor = seqlens_k;
    }
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtError_t error = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    auto qDevice = static_cast<uint8_t *>(q.data_ptr());
    auto kDevice = static_cast<uint8_t *>(k.data_ptr());
    auto vDevice = static_cast<uint8_t *>(v.data_ptr());
    uint8_t * blockTableDevice = nullptr;
    uint8_t * maskDevice = nullptr;
    if (paged_KV) {
        blockTableDevice = static_cast<uint8_t *>(block_table.data_ptr());
    }
    if (is_causal) {
        maskDevice = static_cast<uint8_t *>(mask_gpu_tensor.data_ptr());
    }
    auto oDevice = static_cast<uint8_t *>(out.data_ptr());
    auto qSeqDevice = static_cast<uint8_t *>(seqlenq_gpu_tensor.data_ptr());
    auto kvSeqDevice = static_cast<uint8_t *>(seqlenk_gpu_tensor.data_ptr());
    auto workspaceDevice = static_cast<uint8_t *>(workspace_tensor.data_ptr());
    auto tilingDevice = static_cast<uint8_t *>(tiling_gpu_tensor.data_ptr());
    auto softmaxLseDevice = static_cast<uint8_t *>(softmaxlse.data_ptr());
    if (is_bf16) {
        if (paged_KV) {
            if (is_causal) {
                if (is_varlen_q) {
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            } else {
                if (is_varlen_q) { 
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            }
        } else {
            if (is_causal) {
                if (is_varlen_q) { 
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            } else {
                if (is_varlen_q) { 
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            }
        }
    } else {
        if (paged_KV) {
            if (is_causal) {
                if (is_varlen_q) { 
                    SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            } else {
                if (is_varlen_q) { 
                    SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            }
        } else {
            if (is_causal) {
                if (is_varlen_q) { 
                    SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            } else {
                if (is_varlen_q) { 
                    SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            }
        }
    }
    return {out, softmaxlse, out_accum, softmax_lse_accum};
}

std::vector<at::Tensor>
mha_bwd(at::Tensor dout,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        at::Tensor q,     // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        at::Tensor k,     // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
        at::Tensor v,     // (b, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k
        at::Tensor out,   // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        at::Tensor softmax_lse,    // (b, h, s_q) or (h, total_q) if there is cu_seqlens_q
        std::optional<at::Tensor> dq_,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        std::optional<at::Tensor> dk_,   // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
        std::optional<at::Tensor> dv_,   // (b, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k
        std::optional<at::Tensor> cu_seqlens_q_,   // b+1
        std::optional<at::Tensor> cu_seqlens_k_,   // b+1
        std::optional<at::Tensor> seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
        std::optional<at::Tensor> seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
        std::optional<int64_t> max_seqlen_q_,
        std::optional<int64_t> max_seqlen_k_,
        std::optional<double> softmax_scale_,
        bool is_causal,
        int64_t window_size_left,
        int64_t window_size_right,
        double softcap,
        bool deterministic,
        int64_t sm_margin
)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();        
    
    // input/output tensor
    at::Tensor dq, dk, dv;
    bool is_bf16 = q.dtype() == torch::kBFloat16;
    
    if (dq_.has_value()) {
        dq = dq_.value();
    }  else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
    }  else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
    }  else {
        dv = torch::empty_like(v);
    }

    const bool is_varlen_q = cu_seqlens_q_.has_value();
    const bool is_varlen_kv = cu_seqlens_k_.has_value();
    TORCH_CHECK(!is_varlen_q || is_varlen_kv, "If cu_seqlens_q is provided in bwd, cu_seqlens_k must also be provided");
    TORCH_CHECK(!seqused_q_.has_value(), "mha_bwd does not support seqused_q yet.");
    TORCH_CHECK(!seqused_k_.has_value(), "mha_bwd does not support seqused_k yet.");
    TORCH_CHECK(softcap == 0.0, "mha_bwd does not support softcap yet.");
    TORCH_CHECK(sm_margin == 0, "mha_bwd does not support sm_margin yet.");

    at::Tensor cu_seqlens_q;
    at::Tensor cu_seqlens_k;
    if (is_varlen_q) {
        cu_seqlens_q = cu_seqlens_q_.value();
    }
    if (is_varlen_kv) {
        cu_seqlens_k = cu_seqlens_k_.value();
    }

    // parse shape args (BSND vs TND)
    auto qsizes = q.sizes();
    auto ksizes = k.sizes();
    auto vsizes = v.sizes();
    uint32_t nheads = is_varlen_q ? qsizes[1] : qsizes[2];
    uint32_t nheads_k = is_varlen_q ? ksizes[1] : ksizes[2];
    uint32_t qk_headdim = is_varlen_q ? qsizes[2] : qsizes[3];
    uint32_t v_headdim = is_varlen_q ? vsizes[2] : vsizes[3];
    uint32_t k_headdim = is_varlen_q ? ksizes[2] : ksizes[3];
    TORCH_CHECK(qk_headdim == k_headdim, "mha_bwd: q and k must share the same head dimension.");
    TORCH_CHECK(qk_headdim > 0 && qk_headdim <= 256, "mha_bwd: q/k head dimension must be in (0, 256].");
    // Kernel template only has 64/128/192/256 specializations.
    // For non-exact dims (e.g. 1, 80), choose the nearest supported bucket.
    uint32_t qk_headdim_kernel = qk_headdim <= 64 ? 64 : (qk_headdim <= 128 ? 128 : (qk_headdim <= 192 ? 192 : 256));
    int64_t batch_size = is_varlen_q ? (cu_seqlens_q.size(0) - 1) : qsizes[0];
    TORCH_CHECK(!is_varlen_q || max_seqlen_q_.has_value(), "max_seqlen_q must be provided in varlen bwd.");
    TORCH_CHECK(!is_varlen_q || max_seqlen_k_.has_value(), "max_seqlen_k must be provided in varlen bwd.");
    int64_t max_seqlen_q = is_varlen_q ? max_seqlen_q_.value() : qsizes[1];
    int64_t max_seqlen_k = is_varlen_q ? max_seqlen_k_.value() : ksizes[1];

    // tiling args set
    uint32_t tilingSize = sizeof(FAGTilingData);
    at::Tensor tiling_cpu_tensor = at::empty({tilingSize}, at::device(c10::kCPU).dtype(at::kByte));
    FAGTiling::FAGInfo fagInfo;
    fagInfo.scaleValue =
        softmax_scale_.has_value() ? static_cast<float>(softmax_scale_.value()) : 1.0f / sqrt(static_cast<float>(qk_headdim));
    fagInfo.keepProb = 1.0f;
    fagInfo.maskType = is_causal ? static_cast<int32_t>(FAGTiling::MaskType::MASK_CAUSUAL)
                                 : static_cast<int32_t>(FAGTiling::MaskType::NO_MASK);
    fagInfo.batch = batch_size;
    fagInfo.qSeqlen = max_seqlen_q;
    fagInfo.qHeadNum = nheads;
    fagInfo.qkHeadDim = qk_headdim;
    fagInfo.kvSeqlen = max_seqlen_k;
    fagInfo.kvHeadNum = nheads_k;
    fagInfo.vHeadDim = v_headdim;
    fagInfo.window_size_left = window_size_left;
    fagInfo.window_size_right = window_size_right;
    fagInfo.isDeterministic = deterministic;
    // Tiling uses FAG layout constants (BSND=2, TND=3), not kernel inputLayout enum (BSND=0, TND=1).
    fagInfo.layout = static_cast<int32_t>(is_varlen_q ? TND : BSND);
    at::Tensor cu_seqlens_q_cpu_for_tiling;
    at::Tensor cu_seqlens_k_cpu_for_tiling;
    if (is_varlen_q) {
        cu_seqlens_q_cpu_for_tiling = cu_seqlens_q.to(at::Device(at::kCPU)).to(at::kLong).contiguous();
        cu_seqlens_k_cpu_for_tiling = cu_seqlens_k.to(at::Device(at::kCPU)).to(at::kLong).contiguous();
        // Tiling expects cumulative seqlens with length B and no leading zero.
        fagInfo.qSeqlenList = static_cast<int64_t *>(cu_seqlens_q_cpu_for_tiling.data_ptr()) + 1;
        fagInfo.kvSeqlenList = static_cast<int64_t *>(cu_seqlens_k_cpu_for_tiling.data_ptr()) + 1;
    }
    uint32_t aivNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    FAGTilingData fagTilingData;
    FAGTiling::GetFAGTilingParam(fagInfo, blockDim, aivNum, ubSize, fagTilingData);
    // Host-only std::vector members must not be shallow-copied into the GM tiling buffer.
    fagTilingData.actualSeqQlen.clear();
    fagTilingData.actualSeqKvlen.clear();
    std::memcpy(tiling_cpu_tensor.data_ptr<uint8_t>(), &fagTilingData, sizeof(FAGTilingData));
    // TODO: varlen-specific tiling fields can be extended here once bwd TND kernel is finalized.
    at::Tensor tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));

    // alloc workspace: use tiling-computed size to avoid OOB in shape-dependent paths.
    uint64_t workspaceSize = static_cast<uint64_t>(fagTilingData.workspaceSize);
    TORCH_CHECK(workspaceSize > 0, "mha_bwd: invalid workspace size from tiling.");
    at::Tensor workspace_tensor =
        at::empty({static_cast<long>(workspaceSize)}, at::device(at::kPrivateUse1).dtype(at::kByte));

    // alloc custom attn_mask
    at::Tensor mask_gpu_tensor;
    if (is_causal) {
        at::Tensor mask_cpu_tensor = at::empty({2048, 2048}, at::device(c10::kCPU).dtype(at::kByte));
        mask_cpu_tensor = at::triu(at::ones_like(mask_cpu_tensor), 1);
        mask_gpu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
    }
    
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtError_t error = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    auto qDevice = static_cast<uint8_t *>(const_cast<void *>(q.storage().data()));
    auto kDevice = static_cast<uint8_t *>(const_cast<void *>(k.storage().data()));
    auto vDevice = static_cast<uint8_t *>(const_cast<void *>(v.storage().data()));
    auto outDevice = static_cast<uint8_t *>(const_cast<void *>(out.storage().data()));
    auto dOutDevice = static_cast<uint8_t *>(const_cast<void *>(dout.storage().data()));
    uint8_t *attenMaskDevice = nullptr;
    if (is_causal) {
        attenMaskDevice = static_cast<uint8_t *>(const_cast<void *>(mask_gpu_tensor.storage().data()));
    }
    at::Tensor softmax_lse_kernel = softmax_lse;
    if (!is_varlen_q) {
        TORCH_CHECK(softmax_lse.dim() == 3, "mha_bwd: softmax_lse for BSND must be a 3D tensor.");
        // Kernel CopyInSoftMax expects BSN contiguous layout:
        // softMaxOffset = ((b * s1 + s) * nheads + h)
        if (softmax_lse.size(1) == nheads && softmax_lse.size(2) == max_seqlen_q) {
            // Accept BHS input and convert to BSN.
            softmax_lse_kernel = softmax_lse.transpose(1, 2).contiguous();
        } else {
            TORCH_CHECK(softmax_lse.size(1) == max_seqlen_q && softmax_lse.size(2) == nheads,
                        "mha_bwd: softmax_lse must be BSN or BHS in BSND mode.");
            if (!softmax_lse.is_contiguous()) {
                softmax_lse_kernel = softmax_lse.contiguous();
            }
        }
    } else {
        TORCH_CHECK(softmax_lse.dim() == 2, "mha_bwd: softmax_lse for TND must be a 2D tensor.");
        const int64_t total_q = qsizes[0];
        // TND bwd kernel expects softmax_lse in TN contiguous layout.
        if (softmax_lse.size(0) == nheads && softmax_lse.size(1) == total_q) {
            // Accept NT input and convert to TN.
            softmax_lse_kernel = softmax_lse.transpose(0, 1).contiguous();
        } else {
            TORCH_CHECK(softmax_lse.size(0) == total_q && softmax_lse.size(1) == nheads,
                        "mha_bwd: softmax_lse must be TN or NT in TND mode.");
            if (!softmax_lse.is_contiguous()) {
                softmax_lse_kernel = softmax_lse.contiguous();
            }
        }
    }
    auto softMaxLseDevice = static_cast<uint8_t *>(const_cast<void *>(softmax_lse_kernel.storage().data()));

    auto workspaceDevice = static_cast<uint8_t *>(const_cast<void *>(workspace_tensor.storage().data()));
    auto tilingDevice = static_cast<uint8_t *>(const_cast<void *>(tiling_gpu_tensor.storage().data()));
    auto dqDevice = static_cast<uint8_t *>(const_cast<void *>(dq.storage().data()));
    auto dkDevice = static_cast<uint8_t *>(const_cast<void *>(dk.storage().data()));
    auto dvDevice = static_cast<uint8_t *>(const_cast<void *>(dv.storage().data()));
    uint8_t *cuSeqQlenDevice = nullptr;
    uint8_t *cuSeqKvlenDevice = nullptr;
    at::Tensor seqlenq_gpu_tensor;
    at::Tensor seqlenk_gpu_tensor;
    if (is_varlen_q) {
        // TND kernel expects cumulative seqlens with length B (no leading zero).
        seqlenq_gpu_tensor = cu_seqlens_q.slice(0, 1, cu_seqlens_q.size(0)).contiguous();
        seqlenk_gpu_tensor = cu_seqlens_k.slice(0, 1, cu_seqlens_k.size(0)).contiguous();
        cuSeqQlenDevice = static_cast<uint8_t *>(const_cast<void *>(seqlenq_gpu_tensor.data_ptr()));
        cuSeqKvlenDevice = static_cast<uint8_t *>(const_cast<void *>(seqlenk_gpu_tensor.data_ptr()));
    }
    
    auto launch_fag = [&](auto layout_tag) {
        constexpr uint32_t kInputLayout = decltype(layout_tag)::value;
        if (is_bf16) {
            if (deterministic) {
                switch (qk_headdim_kernel) {
                    case 64:
                        FAG<0, 0, DTemplateType::Aligned64, bfloat16_t, kInputLayout, 1><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                            cuSeqQlenDevice, cuSeqKvlenDevice,
                            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                        break;
                    case 128:
                        FAG<0, 0, DTemplateType::Aligned128, bfloat16_t, kInputLayout, 1><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                            cuSeqQlenDevice, cuSeqKvlenDevice,
                            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                        break;
                    case 192:
                        FAG<0, 0, DTemplateType::Aligned192, bfloat16_t, kInputLayout, 1><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                            cuSeqQlenDevice, cuSeqKvlenDevice,
                            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                        break;
                    case 256:
                        FAG<0, 0, DTemplateType::Aligned256, bfloat16_t, kInputLayout, 1><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                            cuSeqQlenDevice, cuSeqKvlenDevice,
                            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                        break;
                    default:
                        break;
                }
            } else {
                switch (qk_headdim_kernel) {
                    case 64:
                        FAG<0, 0, DTemplateType::Aligned64, bfloat16_t, kInputLayout, 0><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                            cuSeqQlenDevice, cuSeqKvlenDevice,
                            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                        break;
                    case 128:
                        FAG<0, 0, DTemplateType::Aligned128, bfloat16_t, kInputLayout, 0><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                            cuSeqQlenDevice, cuSeqKvlenDevice,
                            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                        break;
                    case 192:
                        FAG<0, 0, DTemplateType::Aligned192, bfloat16_t, kInputLayout, 0><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                            cuSeqQlenDevice, cuSeqKvlenDevice,
                            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                        break;
                    case 256:
                        FAG<0, 0, DTemplateType::Aligned256, bfloat16_t, kInputLayout, 0><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                            cuSeqQlenDevice, cuSeqKvlenDevice,
                            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                        break;
                    default:
                        break;
                }
            }
        } else {
            if (deterministic) {
                switch (qk_headdim_kernel) {
                    case 64:
                        FAG<0, 0, DTemplateType::Aligned64, half, kInputLayout, 1><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                            cuSeqQlenDevice, cuSeqKvlenDevice,
                            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                        break;
                    case 128:
                        FAG<0, 0, DTemplateType::Aligned128, half, kInputLayout, 1><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                            cuSeqQlenDevice, cuSeqKvlenDevice,
                            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                        break;
                    case 192:
                        FAG<0, 0, DTemplateType::Aligned192, half, kInputLayout, 1><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                            cuSeqQlenDevice, cuSeqKvlenDevice,
                            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                        break;
                    case 256:
                        FAG<0, 0, DTemplateType::Aligned256, half, kInputLayout, 1><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                            cuSeqQlenDevice, cuSeqKvlenDevice,
                            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                        break;
                    default:
                        break;
                }
            } else {
                switch (qk_headdim_kernel) {
                    case 64:
                        FAG<0, 0, DTemplateType::Aligned64, half, kInputLayout, 0><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                            cuSeqQlenDevice, cuSeqKvlenDevice,
                            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                        break;
                    case 128:
                        FAG<0, 0, DTemplateType::Aligned128, half, kInputLayout, 0><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                            cuSeqQlenDevice, cuSeqKvlenDevice,
                            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                        break;
                    case 192:
                        FAG<0, 0, DTemplateType::Aligned192, half, kInputLayout, 0><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                            cuSeqQlenDevice, cuSeqKvlenDevice,
                            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                        break;
                    case 256:
                        FAG<0, 0, DTemplateType::Aligned256, half, kInputLayout, 0><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice, nullptr, attenMaskDevice, softMaxLseDevice,
                            cuSeqQlenDevice, cuSeqKvlenDevice,
                            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
                        break;
                    default:
                        break;
                }
            }
        }
    };
    if (is_varlen_q) {
        launch_fag(std::integral_constant<uint32_t, static_cast<uint32_t>(TND)>());
    } else {
        launch_fag(std::integral_constant<uint32_t, static_cast<uint32_t>(BSND)>());
    }

    {
        aclrtStream stream = reinterpret_cast<aclrtStream>(aclStream);
        aclError sync_st = aclrtSynchronizeStream(stream);
        at::Tensor lse_cpu = softmax_lse_kernel.to(at::kCPU).contiguous();
        const float *lp = lse_cpu.data_ptr<float>();
        const int64_t tot = lse_cpu.numel();
        const int64_t nshow = std::min<int64_t>(16, tot);
        float vmin = std::numeric_limits<float>::infinity();
        float vmax = -std::numeric_limits<float>::infinity();
        int64_t ninf = 0;
        int64_t nnan = 0;
        for (int64_t i = 0; i < tot; ++i) {
            const float v = lp[i];
            if (std::isnan(v)) {
                ++nnan;
            } else if (!std::isfinite(v)) {
                ++ninf;
            } else {
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
            }
        }
    }

    auto opts = q.options();
    auto softmax_d = torch::empty({batch_size, nheads, max_seqlen_q}, opts.dtype(at::kFloat));
    return {dq, dk, dv, softmax_d};
}

PYBIND11_MODULE(flash_attn_npu_3, m)
{
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass, with KV-cache");
    m.def("bwd", &mha_bwd, "Backward pass");
}