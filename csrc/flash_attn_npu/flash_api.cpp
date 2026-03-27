#include <torch/extension.h>

#include "mha_fwd_kvcache.cpp"
#include "tilingdata.h"
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "mha_varlen_bwd.cpp"
#include "fag_tiling.cpp"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "runtime/rt_ffts.h"
#include "kernel_common.hpp"
#include "kernel_operator.h"


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
mha_fwd_kvcache(at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                std::optional<const at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                std::optional<const at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                std::optional<const at::Tensor> &seqlens_k_, // batch_size
                std::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                std::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                std::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                std::optional<const at::Tensor> &leftpad_k_, // batch_size
                std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
                std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits
                )
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    auto q_dtype = q.dtype();
    bool is_bf16 = q_dtype == torch::kBFloat16;
    bool is_fp16 = q_dtype == torch::kFloat16;

    TORCH_CHECK(is_bf16 || is_fp16, "FlashAttention only supports FP16 and BF16 data types");
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key_cache must have the same dtype");
    TORCH_CHECK(vcache.dtype() == q_dtype, "query and value_cache must have the same dtype");

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor q must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor kcache must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor vcache must have contiguous last dimension");

    at::Tensor tiling_cpu_tensor = at::empty({1024}, at::device(c10::kCPU).dtype(at::kByte));

    FAInferTilingData* tiling_cpu_ptr = reinterpret_cast<FAInferTilingData*>(tiling_cpu_tensor.data_ptr<uint8_t>());
    uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    at::Tensor seqlens_k, block_table, out;
    at::Tensor k, v, rotary_cos, rotary_sin, cache_batch_idx, alibi_slopes;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        block_table = block_table_.value();
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    if (seqlens_k_.has_value()) {
        seqlens_k = seqlens_k_.value();
        TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
    }

    TORCH_CHECK(!alibi_slopes_.has_value(), "NPU FlashAttention does not support alibi_slopes");
    TORCH_CHECK(!leftpad_k_.has_value(), "NPU FlashAttention does not support leftpad_k");
    TORCH_CHECK(!rotary_cos_.has_value(), "NPU FlashAttention does not support rotary embedding");
    TORCH_CHECK(!rotary_sin_.has_value(), "NPU FlashAttention does not support rotary embedding");
    TORCH_CHECK(softcap == 0.0f, "NPU FlashAttention does not support softcap");
    TORCH_CHECK(window_size_left == -1, "NPU FlashAttention does not support window_size_left");
    TORCH_CHECK(window_size_right == -1, "NPU FlashAttention does not support window_size_right");
    TORCH_CHECK(num_splits == 1 || num_splits == 0, "NPU FlashAttention only supports num_splits=1 or num_splits=0");

    if (k_.has_value()) {
        k = k_.value();
    }
    if (v_.has_value()) {
        v = v_.value();
    }
    if (rotary_cos_.has_value()) {
        rotary_cos = rotary_cos_.value();
    }
    if (rotary_sin_.has_value()) {
        rotary_sin = rotary_sin_.value();
    }
    if (cache_batch_idx_.has_value()) {
        cache_batch_idx = cache_batch_idx_.value();
    }
    if (alibi_slopes_.has_value()) {
        alibi_slopes = alibi_slopes_.value();
    }
    if (out_.has_value()) {
        out = out_.value();
    }  else {
        out = torch::empty_like(q);
    }
    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size_og = sizes[3];
    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : kcache.size(0);
    const int page_block_size = !paged_KV ? 128 : kcache.size(1);
    const int num_heads_k = kcache.size(2);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    at::Tensor seqlenk_cpu_tensor = seqlens_k.to(at::Device(at::kCPU));
    int32_t* seqlens_k_cpu = static_cast<int32_t *>(seqlenk_cpu_tensor.data_ptr());
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
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtError_t error = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    auto qDevice = static_cast<uint8_t *>(q.data_ptr());
    auto kDevice = static_cast<uint8_t *>(kcache.data_ptr());
    auto vDevice = static_cast<uint8_t *>(vcache.data_ptr());
    uint8_t * blockTableDevice = nullptr;
    uint8_t * maskDevice = nullptr;
    if (paged_KV) {
        blockTableDevice = static_cast<uint8_t *>(block_table.data_ptr());
    }
    if (is_causal) {
        maskDevice = static_cast<uint8_t *>(mask_gpu_tensor.data_ptr());
    }
    auto oDevice = static_cast<uint8_t *>(out.data_ptr());
    auto qSeqDevice = static_cast<uint8_t *>(seqlens_k.data_ptr());
    auto kvSeqDevice = static_cast<uint8_t *>(seqlens_k.data_ptr());
    auto workspaceDevice = static_cast<uint8_t *>(workspace_tensor.data_ptr());
    auto tilingDevice = static_cast<uint8_t *>(tiling_gpu_tensor.data_ptr());
    auto softmaxLseDevice = static_cast<uint8_t *>(softmaxlse.data_ptr());
    if (is_bf16) {
        if (paged_KV) {
            if (is_causal) {
                SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            } else {
                SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            }
        } else {
            if (is_causal) {
                SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            } else {
                SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            }
        }
    } else {
        if (paged_KV) {
            if (is_causal) {
                SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            } else {
                SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            }
        } else {
            if (is_causal) {
                SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            } else {
                SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            }
        }
    }
    return {out, softmaxlse};
}

std::vector<at::Tensor>
mha_fwd(at::Tensor &q,                            // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,                      // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,                      // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &out_,          // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        std::optional<at::Generator> gen_)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto q_dtype = q.dtype();
    bool is_bf16 = q.dtype() == torch::kBFloat16;
    // parameters checking
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    // block unsupported params
    TORCH_CHECK(!alibi_slopes_.has_value(), "NPU FlashAttention does not support alibi_slopes.");
    TORCH_CHECK(p_dropout == 0.0, "NPU FlashAttention does not support dropout.");
    TORCH_CHECK(softcap == 0.0, "NPU FlashAttention does not support softcap.");
    TORCH_CHECK(!return_softmax, "NPU FlashAttention does not support return_softmax.");
    TORCH_CHECK(window_size_left == -1, "NPU FlashAttention does not support window_size_left.");
    TORCH_CHECK(window_size_right == -1, "NPU FlashAttention does not support window_size_right.");

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // init output tensors
    at::Tensor out = (out_.has_value()) ? out_.value() : torch::empty_like(q);
    auto opts = q.options().device(at::kPrivateUse1);
    auto p = torch::empty({0}, opts);
    if (return_softmax) {
        p = torch::empty({batch_size, num_heads, seqlen_q, seqlen_k}, opts);
    }
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));

    // init mask 
    at::Tensor mask_gpu_tensor;
    uint8_t * maskDevice = nullptr;
    if (is_causal) {
        at::Tensor mask_cpu_tensor = at::empty({2048, 2048}, at::device(c10::kCPU).dtype(at::kByte));
        mask_cpu_tensor = at::triu(at::ones_like(mask_cpu_tensor), 1);
        mask_gpu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
        maskDevice = static_cast<uint8_t *>(const_cast<void *>(mask_gpu_tensor.data_ptr()));
    }

    // init softmax lse
    at::Tensor softmaxlse = at::empty({batch_size, seqlen_q, num_heads}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    softmaxlse.fill_(std::numeric_limits<float>::infinity()); 
    auto softmaxLseDevice = static_cast<uint8_t *>(const_cast<void *>(softmaxlse.data_ptr()));

    // ffts related
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtError_t error = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);

    // set worksapce
    uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
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

    // tiling
    at::Tensor tiling_cpu_tensor = at::empty({1024}, at::device(c10::kCPU).dtype(at::kByte));
    FAInferTilingData* tiling_cpu_ptr = reinterpret_cast<FAInferTilingData*>(tiling_cpu_tensor.data_ptr<uint8_t>());
    tiling_cpu_ptr->set_batch(static_cast<uint32_t>(batch_size));
    tiling_cpu_ptr->set_numHeads(static_cast<uint32_t>(num_heads));
    tiling_cpu_ptr->set_kvHeads(static_cast<uint32_t>(num_heads_k));
    tiling_cpu_ptr->set_embeddingSize(static_cast<uint32_t>(head_size));
    tiling_cpu_ptr->set_embeddingSizeV(static_cast<uint32_t>(head_size));
    tiling_cpu_ptr->set_numBlocks(static_cast<uint32_t>(0));
    tiling_cpu_ptr->set_blockSize(static_cast<uint32_t>(128));
    tiling_cpu_ptr->set_maxNumBlocksPerBatch(static_cast<uint32_t>(0));
    tiling_cpu_ptr->set_maskType(static_cast<uint32_t>(is_causal));
    tiling_cpu_ptr->set_scaleValue(softmax_scale);
    tiling_cpu_ptr->set_maxQSeqlen(seqlen_q);
    tiling_cpu_ptr->set_mm1OutSize(mm1OutSize);
    tiling_cpu_ptr->set_smOnlineOutSize(smOnlineOutSize);
    tiling_cpu_ptr->set_mm2OutSize(mm2OutSize);
    tiling_cpu_ptr->set_UpdateSize(UpdateSize);
    tiling_cpu_ptr->set_workSpaceSize(workSpaceSize);

    uint32_t totalTaskNum = 0;
    uint32_t groupSize = num_heads / num_heads_k;
    uint32_t firstBatchTaskNum = 0;
    for (int32_t batchIdx = 0; batchIdx < batch_size; batchIdx++) {
        uint64_t qSeqlen = seqlen_q;
        uint64_t kvSeqlen = seqlen_k;
        uint64_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
        uint64_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
        uint64_t curQNBlockNum = qNBlockNumPerGroup * num_heads_k;
        uint64_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
        uint64_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
        uint64_t curTaskNum = curQNBlockNum * curQSBlockNum;
        if (batchIdx == 0) {
            tiling_cpu_ptr->set_firstBatchTaskNum(curTaskNum);
            firstBatchTaskNum = curTaskNum;
        }
        totalTaskNum += curTaskNum;
    }
    tiling_cpu_ptr->set_totalTaskNum(totalTaskNum);

    // device ptrs
    auto qDevice = static_cast<uint8_t *>(const_cast<void *>(q.data_ptr()));
    auto kDevice = static_cast<uint8_t *>(const_cast<void *>(k.data_ptr()));
    auto vDevice = static_cast<uint8_t *>(const_cast<void *>(v.data_ptr()));
    at::Tensor seqlenq_gpu_tensor = at::full({batch_size}, seqlen_q).to(at::Device(at::kPrivateUse1)).to(at::kInt); 
    at::Tensor seqlenk_gpu_tensor = at::full({batch_size}, seqlen_k).to(at::Device(at::kPrivateUse1)).to(at::kInt); 
    at::Tensor tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));
    auto qSeqDevice = static_cast<uint8_t *>(const_cast<void *>(seqlenq_gpu_tensor.data_ptr()));
    auto kvSeqDevice = static_cast<uint8_t *>(const_cast<void *>(seqlenk_gpu_tensor.data_ptr()));
    auto workspaceDevice = static_cast<uint8_t *>(const_cast<void *>(workspace_tensor.data_ptr()));
    auto tilingDevice = static_cast<uint8_t *>(const_cast<void *>(tiling_gpu_tensor.data_ptr()));
    auto oDevice = static_cast<uint8_t *>(const_cast<void *>(out.data_ptr()));
    uint8_t * blockTableDevice = nullptr; // will not be used in non-kvcahce fwd api

    // run kernel
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    if (is_bf16) {
        if (is_causal) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                    fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                    qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                    fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                    qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        }
    } else {
        if (is_causal) {
            SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                    fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                    qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else {
            SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                    fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                    qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        }
    }

    return {out, softmaxlse, p, rng_state};
}

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
               std::optional<const at::Tensor> &leftpad_k_, // batch_size
               std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               std::optional<at::Generator> gen_)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor tiling_cpu_tensor = at::empty({1024}, at::device(c10::kCPU).dtype(at::kByte));
    FAInferTilingData* tiling_cpu_ptr = reinterpret_cast<FAInferTilingData*>(tiling_cpu_tensor.data_ptr<uint8_t>());
    uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    
    bool is_bf16 = q.dtype() == torch::kBFloat16;
    bool is_fp16 = q.dtype() == torch::kFloat16;
    const bool paged_KV = block_table_.has_value();

    // 校验拦截不支持的模式
    TORCH_CHECK(is_bf16 || is_fp16, "NPU FlashAttention only supports Float16 or BFloat16.");
    TORCH_CHECK(!block_table_.has_value(), "NPU FlashAttention does not support paged KV cache with block table.");
    TORCH_CHECK(!seqused_k_.has_value(), "NPU FlashAttention does not support seqused_k.");
    TORCH_CHECK(!leftpad_k_.has_value(), "NPU FlashAttention does not support leftpad_k.");
    TORCH_CHECK(!alibi_slopes_.has_value(), "NPU FlashAttention does not support alibi_slopes.");
    TORCH_CHECK(p_dropout == 0.0, "NPU FlashAttention does not support dropout.");
    TORCH_CHECK(!zero_tensors, "NPU FlashAttention does not support zero_tensors.");
    TORCH_CHECK(softcap == 0.0, "NPU FlashAttention does not support softcap.");
    TORCH_CHECK(!return_softmax, "NPU FlashAttention does not support return_softmax.");
    TORCH_CHECK(window_size_left == -1, "NPU FlashAttention does not support window_size_left.");
    TORCH_CHECK(window_size_right == -1, "NPU FlashAttention does not support window_size_right.");
    TORCH_CHECK(k.dtype() == q.dtype(), "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q.dtype(), "query and value must have the same dtype");
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
    }  else {
        out = torch::empty_like(q);
    }

    // 可选输入，当前均不支持
    at::Tensor seqlens_k, leftpad_k, alibi_slopes, block_table;
    if (seqused_k_.has_value()) {
        seqlens_k = seqused_k_.value();
    }
    if (leftpad_k_.has_value()) {
        leftpad_k = leftpad_k_.value();
    }

    if (alibi_slopes_.has_value()) {
        alibi_slopes = alibi_slopes_.value();
    }
    if (paged_KV) {
        block_table = block_table_.value();
    }

    const auto sizes = q.sizes();
    int T = sizes[0];
    int num_heads = sizes[1];
    const int head_size_og = sizes[2];
    const int batch_size = cu_seqlens_q.numel() - 1;

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : k.size(0);
    const int page_block_size = !paged_KV ? 128 : k.size(1);
    const int num_heads_k = k.size(1);

    TORCH_CHECK(head_size_og <= 256, "FlashAttention only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    
    tiling_cpu_ptr->set_batch(static_cast<uint32_t>(batch_size)); //B
    tiling_cpu_ptr->set_numHeads(static_cast<uint32_t>(num_heads)); //N
    tiling_cpu_ptr->set_kvHeads(static_cast<uint32_t>(num_heads_k)); //S
    tiling_cpu_ptr->set_embeddingSize(static_cast<uint32_t>(head_size_og)); //D
    tiling_cpu_ptr->set_embeddingSizeV(static_cast<uint32_t>(head_size_og)); //vD
    tiling_cpu_ptr->set_numBlocks(static_cast<uint32_t>(num_blocks));  //0
    tiling_cpu_ptr->set_blockSize(static_cast<uint32_t>(page_block_size)); //128
    tiling_cpu_ptr->set_maxNumBlocksPerBatch(static_cast<uint32_t>(max_num_blocks_per_seq)); //0
    tiling_cpu_ptr->set_maskType(static_cast<uint32_t>(is_causal));
    tiling_cpu_ptr->set_scaleValue(softmax_scale);
    tiling_cpu_ptr->set_maxQSeqlen(max_seqlen_q);
    
    uint64_t WORKSPACE_BLOCK_SIZE_DB = 128 * 512;  // 工作空间块大小 ，每次计算128 * 512
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

    at::Tensor workspace_tensor = at::empty({workSpaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte)); // workspace
    at::Tensor softmaxlse = at::empty({T, num_heads}, at::device(at::kPrivateUse1).dtype(at::kFloat)); // lse
    softmaxlse.fill_(std::numeric_limits<float>::infinity()); 

    tiling_cpu_ptr->set_mm1OutSize(mm1OutSize);
    tiling_cpu_ptr->set_smOnlineOutSize(smOnlineOutSize);
    tiling_cpu_ptr->set_mm2OutSize(mm2OutSize);
    tiling_cpu_ptr->set_UpdateSize(UpdateSize);
    tiling_cpu_ptr->set_workSpaceSize(workSpaceSize);

    uint32_t totalTaskNum = 0; // 核内计算
    tiling_cpu_ptr->set_totalTaskNum(totalTaskNum);
    at::Tensor tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1)); // Tiling to Device
    
    // attention mask
    at::Tensor mask_gpu_tensor;
    if (is_causal) {
        at::Tensor mask_cpu_tensor = at::empty({2048, 2048}, at::device(c10::kCPU).dtype(at::kByte));
        mask_cpu_tensor = at::triu(at::ones_like(mask_cpu_tensor), 1);
        mask_gpu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
    }

    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtError_t error = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    auto qDevice = static_cast<uint8_t *>(const_cast<void *>(q.data_ptr()));
    auto kDevice = static_cast<uint8_t *>(const_cast<void *>(k.data_ptr()));
    auto vDevice = static_cast<uint8_t *>(const_cast<void *>(v.data_ptr()));
    
    uint8_t * blockTableDevice = nullptr;
    uint8_t * maskDevice = nullptr;
    if (paged_KV) {
        blockTableDevice = static_cast<uint8_t *>(const_cast<void *>(block_table.data_ptr()));
    }
    if (is_causal) {
        maskDevice = static_cast<uint8_t *>(const_cast<void *>(mask_gpu_tensor.data_ptr()));
    }
    
    auto oDevice = static_cast<uint8_t *>(const_cast<void *>(out.data_ptr()));
    auto qSeqDevice = static_cast<uint8_t *>(const_cast<void *>(cu_seqlens_q.data_ptr()));
    auto kvSeqDevice = static_cast<uint8_t *>(const_cast<void *>(cu_seqlens_k.data_ptr()));
    auto workspaceDevice = static_cast<uint8_t *>(const_cast<void *>(workspace_tensor.data_ptr()));
    auto tilingDevice = static_cast<uint8_t *>(const_cast<void *>(tiling_gpu_tensor.data_ptr()));
    auto softmaxLseDevice = static_cast<uint8_t *>(const_cast<void *>(softmaxlse.data_ptr()));

    if (is_bf16) {
        if (paged_KV) {
            if (is_causal) {
                SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY>
                <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            } else {
                SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY>
                <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            }
        } else {
            if (is_causal) {
                SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY>
                <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            } else {
                SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY>
                <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            }
        }
    } else {
        if (paged_KV) {
            if (is_causal) {
                SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY>
                <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            } else {
                SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY>
                <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            }
        } else {
            if (is_causal) {
                SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY>
                <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            } else {
                SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY>
                <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            }
        }
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(at::Device(at::kPrivateUse1));
    at::Tensor rng_state =  torch::empty({2}, options.dtype(torch::kInt64));
    at::Tensor p = torch::empty({ 0 }, options.dtype(torch::kInt64));
    return {out, softmaxlse, p, rng_state};
}


std::vector<at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,                   // total_q x num_heads x head_size
               const at::Tensor &q,                      // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,                      // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,                      // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &out,                    // total_q x num_heads x head_size
               const at::Tensor &softmax_lse,            // h x total_q   softmax logsumexp
               std::optional<at::Tensor> &dq_,           // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dk_,           // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dv_,           // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,           // b+1
               const at::Tensor &cu_seqlens_k,           // b+1
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               const int max_seqlen_q,
               const int max_seqlen_k, // max sequence length to choose the kernel
               const float p_dropout,  // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool deterministic,
               std::optional<at::Generator> gen_,
               std::optional<at::Tensor> &rng_state)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // input/output tensor
    at::Tensor seqlens_q, seqlens_k;
    at::Tensor dq, dk, dv;
    bool is_bf16 = q.dtype() == torch::kBFloat16;

    seqlens_q = cu_seqlens_q;
    seqlens_k = cu_seqlens_k;
    
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

    // parse shape args
    auto qsizes = q.sizes();
    auto ksizes = k.sizes();
    uint32_t nheads = qsizes[1];
    uint32_t nheads_k = ksizes[1];
    uint32_t headdim = qsizes[2];

    // tiling args set
    uint32_t tilingSize = TILING_PARA_NUM * sizeof(int64_t);
    at::Tensor tiling_cpu_tensor = at::empty({tilingSize}, at::device(c10::kCPU).dtype(at::kByte));
    FAGTiling::FAGInfo fagInfo;
    int64_t sum_of_list = qsizes[0];
    fagInfo.seqQShapeSize = cu_seqlens_q.sizes()[0] - 1;
    fagInfo.queryShape_0 = sum_of_list;
    fagInfo.keyShape_0 = sum_of_list;
    fagInfo.queryShape_1 = nheads;
    fagInfo.keyShape_1 = nheads_k;
    fagInfo.queryShape_2 = headdim;
    fagInfo.scaleValue = 1.0 / sqrt(headdim);
    FAGTiling::GetFATilingParam(fagInfo, blockDim, reinterpret_cast<int64_t *>(tiling_cpu_tensor.data_ptr<uint8_t>()));
    // FAGTiling::printFAGTilingData(reinterpret_cast<int64_t *>(tiling_cpu_tensor.data_ptr<uint8_t>()));
    at::Tensor tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));

    // alloc workspace
    uint64_t workspaceSize = (2 * blockDim * 16 * 128 * 128 * 8 * nheads) * sizeof(float);
    at::Tensor workspace_tensor = at::empty({static_cast<long>(workspaceSize)}, at::device(at::kPrivateUse1).dtype(at::kByte));

    // alloc custom attn_mask
    at::Tensor mask_gpu_tensor;
    if (is_causal) {
        at::Tensor mask_cpu_tensor = at::empty({2048, 2048}, at::device(c10::kCPU).dtype(at::kByte));
        mask_cpu_tensor = at::triu(at::ones_like(mask_cpu_tensor), 1);
        mask_gpu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
    }
    at::Tensor seqlenq_gpu_tensor = seqlens_q.to(at::Device(at::kPrivateUse1));
    at::Tensor seqlenk_gpu_tensor = seqlens_k.to(at::Device(at::kPrivateUse1));
    
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
    auto cuSeqQlenDevice = static_cast<uint8_t *>(const_cast<void *>(seqlenq_gpu_tensor.storage().data()));
    auto cuSeqKvlenDevice = static_cast<uint8_t *>(const_cast<void *>(seqlenk_gpu_tensor.storage().data()));
    auto softMaxLseDevice = static_cast<uint8_t *>(const_cast<void *>(softmax_lse.storage().data()));

    auto workspaceDevice = static_cast<uint8_t *>(const_cast<void *>(workspace_tensor.storage().data()));
    auto tilingDevice = static_cast<uint8_t *>(const_cast<void *>(tiling_gpu_tensor.storage().data()));
    auto dqDevice = static_cast<uint8_t *>(const_cast<void *>(dq.storage().data()));
    auto dkDevice = static_cast<uint8_t *>(const_cast<void *>(dk.storage().data()));
    auto dvDevice = static_cast<uint8_t *>(const_cast<void *>(dv.storage().data()));

    #if defined(ENABLE_ASCENDC_DUMP)

        uint8_t *ptrDumpDevice{nullptr};
        aclCheck(aclrtMalloc(reinterpret_cast<void **>(&ptrDumpDevice), ALL_DUMPSIZE, ACL_MEM_MALLOC_HUGE_FIRST));
        if (is_bf16) {
            FAG::FAG<bfloat16_t><<<blockDim, nullptr, aclStream>>>(
                fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr, nullptr,
                attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, cuSeqQlenDevice, cuSeqKvlenDevice,
                nullptr, nullptr, dqDevice, dkDevice, dvDevice, workspaceDevice, tilingDevice, ptrDumpDevice);
        } else {
            FAG::FAG<half><<<blockDim, nullptr, aclStream>>>(
                fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr, nullptr,
                attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, cuSeqQlenDevice, cuSeqKvlenDevice,
                nullptr, nullptr, dqDevice, dkDevice, dvDevice, workspaceDevice, tilingDevice, ptrDumpDevice);
        }
        aclCheck(aclrtSynchronizeStream(aclStream));
        Adx::AdumpPrintWorkSpace(ptrDumpDevice, ALL_DUMPSIZE, aclStream, "device_fag");
        aclCheck(aclrtFree(ptrDumpDevice));
    #else
        if (is_bf16) {
            if (is_causal) {
                FAG::FAG<bfloat16_t, MaskType::MASK_CAUSAL, InputLayout::TND><<<blockDim, nullptr, aclStream>>>(
                    fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr, nullptr,
                    attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, cuSeqQlenDevice, cuSeqKvlenDevice,
                    nullptr, nullptr, dqDevice, dkDevice, dvDevice, workspaceDevice, tilingDevice, nullptr);
            } else {
                FAG::FAG<bfloat16_t, MaskType::NO_MASK, InputLayout::TND><<<blockDim, nullptr, aclStream>>>(
                    fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr, nullptr,
                    attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, cuSeqQlenDevice, cuSeqKvlenDevice,
                    nullptr, nullptr, dqDevice, dkDevice, dvDevice, workspaceDevice, tilingDevice, nullptr);
            }
        } else {
            if (is_causal) {
                FAG::FAG<half, MaskType::MASK_CAUSAL, InputLayout::TND><<<blockDim, nullptr, aclStream>>>(
                    fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr, nullptr,
                    attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, cuSeqQlenDevice, cuSeqKvlenDevice,
                    nullptr, nullptr, dqDevice, dkDevice, dvDevice, workspaceDevice, tilingDevice, nullptr);
            } else {
                FAG::FAG<half, MaskType::NO_MASK, InputLayout::TND><<<blockDim, nullptr, aclStream>>>(
                    fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr, nullptr,
                    attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, cuSeqQlenDevice, cuSeqKvlenDevice,
                    nullptr, nullptr, dqDevice, dkDevice, dvDevice, workspaceDevice, tilingDevice, nullptr);
            }
        }
    #endif
    auto opts = q.options();
    auto softmax_d = torch::empty({fagInfo.seqQShapeSize, nheads, max_seqlen_q}, opts.dtype(at::kFloat));
    return {dq, dk, dv, softmax_d};
}

std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x multiple_of(head_size_og, 8)
        const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        std::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        std::optional<at::Generator> gen_,
        std::optional<at::Tensor> &rng_state) {
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

    // parse shape args
    auto qsizes = q.sizes();
    auto ksizes = k.sizes();
    uint32_t nheads = qsizes[2];
    uint32_t nheads_k = ksizes[2];
    uint32_t headdim = qsizes[3];

    // tiling args set
    uint32_t tilingSize = TILING_PARA_NUM * sizeof(int64_t);
    at::Tensor tiling_cpu_tensor = at::empty({tilingSize}, at::device(c10::kCPU).dtype(at::kByte));
    FAGTiling::FAGInfo fagInfo;
    fagInfo.seqQShapeSize = qsizes[0];
    fagInfo.queryShape_0 = qsizes[0] * qsizes[1];
    fagInfo.keyShape_0 = ksizes[0] * ksizes[1];
    fagInfo.queryShape_1 = nheads;
    fagInfo.keyShape_1 = nheads_k;
    fagInfo.queryShape_2 = headdim;
    fagInfo.scaleValue = 1.0 / sqrt(headdim);
    FAGTiling::GetFATilingParam(fagInfo, blockDim, reinterpret_cast<int64_t *>(tiling_cpu_tensor.data_ptr<uint8_t>()));
    // FAGTiling::printFAGTilingData(reinterpret_cast<int64_t *>(tiling_cpu_tensor.data_ptr<uint8_t>()));
    at::Tensor tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));

    // alloc workspace
    uint64_t workspaceSize = (2 * blockDim * 16 * 128 * 128 * 8 * nheads) * sizeof(float);
    at::Tensor workspace_tensor = at::empty({static_cast<long>(workspaceSize)}, at::device(at::kPrivateUse1).dtype(at::kByte));

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
    auto softMaxLseDevice = static_cast<uint8_t *>(const_cast<void *>(softmax_lse.storage().data()));

    auto workspaceDevice = static_cast<uint8_t *>(const_cast<void *>(workspace_tensor.storage().data()));
    auto tilingDevice = static_cast<uint8_t *>(const_cast<void *>(tiling_gpu_tensor.storage().data()));
    auto dqDevice = static_cast<uint8_t *>(const_cast<void *>(dq.storage().data()));
    auto dkDevice = static_cast<uint8_t *>(const_cast<void *>(dk.storage().data()));
    auto dvDevice = static_cast<uint8_t *>(const_cast<void *>(dv.storage().data()));

    #if defined(ENABLE_ASCENDC_DUMP)
        uint8_t *ptrDumpDevice{nullptr};
        aclCheck(aclrtMalloc(reinterpret_cast<void **>(&ptrDumpDevice), ALL_DUMPSIZE, ACL_MEM_MALLOC_HUGE_FIRST));
        if (is_bf16) {
            FAG::FAG<bfloat16_t><<<blockDim, nullptr, aclStream>>>(
                fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr, nullptr,
                attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, nullptr, nullptr,
                nullptr, nullptr, dqDevice, dkDevice, dvDevice, workspaceDevice, tilingDevice, ptrDumpDevice);
        } else {
            FAG::FAG<half><<<blockDim, nullptr, aclStream>>>(
                fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr, nullptr,
                attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, nullptr, nullptr,
                nullptr, nullptr, dqDevice, dkDevice, dvDevice, workspaceDevice, tilingDevice, ptrDumpDevice);
        }
        aclCheck(aclrtSynchronizeStream(aclStream));
        Adx::AdumpPrintWorkSpace(ptrDumpDevice, ALL_DUMPSIZE, aclStream, "device_fag");
        aclCheck(aclrtFree(ptrDumpDevice));
    #else
        if (is_bf16) {
            if (is_causal) {
                FAG::FAG<bfloat16_t, MaskType::MASK_CAUSAL, InputLayout::BSND><<<blockDim, nullptr, aclStream>>>(
                    fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr, nullptr,
                    attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, nullptr, nullptr,
                    nullptr, nullptr, dqDevice, dkDevice, dvDevice, workspaceDevice, tilingDevice, nullptr);
            } else {
                FAG::FAG<bfloat16_t, MaskType::NO_MASK, InputLayout::BSND><<<blockDim, nullptr, aclStream>>>(
                    fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr, nullptr,
                    attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, nullptr, nullptr,
                    nullptr, nullptr, dqDevice, dkDevice, dvDevice, workspaceDevice, tilingDevice, nullptr);
            }
        } else {
            if (is_causal) {
                FAG::FAG<half, MaskType::MASK_CAUSAL, InputLayout::BSND><<<blockDim, nullptr, aclStream>>>(
                    fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr, nullptr,
                    attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, nullptr, nullptr,
                    nullptr, nullptr, dqDevice, dkDevice, dvDevice, workspaceDevice, tilingDevice, nullptr);
            } else {
                FAG::FAG<half, MaskType::NO_MASK, InputLayout::BSND><<<blockDim, nullptr, aclStream>>>(
                    fftsAddr, qDevice, kDevice, vDevice, dOutDevice, nullptr, nullptr, nullptr, nullptr, nullptr,
                    attenMaskDevice, softMaxLseDevice, nullptr, outDevice, nullptr, nullptr, nullptr,
                    nullptr, nullptr, dqDevice, dkDevice, dvDevice, workspaceDevice, tilingDevice, nullptr);
            }
        }
    #endif
    auto opts = q.options();
    auto softmax_d = torch::empty({fagInfo.seqQShapeSize, nheads, qsizes[1]}, opts.dtype(at::kFloat));
    return {dq, dk, dv, softmax_d};
}

PYBIND11_MODULE(flash_attn_2_C, m)
{
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass, with KV-cache");
    m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
    m.def("varlen_bwd", &mha_varlen_bwd, "Backward pass (variable length)");
}
