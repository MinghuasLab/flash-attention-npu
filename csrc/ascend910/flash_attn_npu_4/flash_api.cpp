#include <torch/extension.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>

// mha_fwd_kvcache.cpp (SplitFuse::FAInfer) is compiled separately in the
// autogen dispatch TUs; flash_api.cpp only needs FAInferTilingData (tilingdata.h),
// the FaiKenel enum (kernel_common.hpp), and the fa_split host helper.
#include "tilingdata.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"
#include "runtime/rt_ffts.h"
// kernel_operator.h (AscendC) defines GM_ADDR; catlass/catlass.hpp defines the
// Catlass namespace / CATLASS_DEVICE. Both were formerly pulled in transitively
// by mha_fwd_kvcache.cpp, now compiled separately, so include
// them explicitly here, before kernel_common.hpp.
#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "kernel_common.hpp"
#include "tiling/platform/platform_ascendc.h"

// mha_fwd_kvcache.cpp used to carry these using-directives
// into this TU; restore them so unqualified KernelCommon constants resolve.
using namespace Catlass;
using namespace KernelCommon;

#include "fa_split.h"
#include "fwd_dispatch.hpp"
#include <cmath>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

std::vector<at::Tensor>
mha_fwd(at::Tensor q,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        at::Tensor k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
        at::Tensor v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table.
        std::optional<at::Tensor> q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
        std::optional<at::Tensor> out_,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        std::optional<at::Tensor> cu_seqlens_q_,  // b+1
        std::optional<at::Tensor> cu_seqlens_k_,  // b+1
        std::optional<at::Tensor> seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
        std::optional<at::Tensor> seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
        std::optional<int64_t> max_seqlen_q_,
        // TODO: check if we need max_seqlen_k
        std::optional<int64_t> max_seqlen_k_,
        std::optional<int64_t> min_seqlen_k_,
        std::optional<at::Tensor> page_table_, // (b_k, max_num_pages_per_seq)
        std::optional<at::Tensor> gather_kv_indices_,
        std::optional<float> softmax_scale_,
        bool is_causal,
        int64_t window_size_left,
        int64_t window_size_right,
        float softcap,
        int64_t num_splits,
        std::optional<bool> pack_gqa_,
        std::optional<at::Tensor> learnable_sink_
        )
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    auto hostStart = std::chrono::steady_clock::now();

    auto q_dtype = q.dtype();
    bool is_bf16 = q_dtype == torch::kBFloat16;
    bool is_fp16 = q_dtype == torch::kFloat16;
    TORCH_CHECK(is_bf16 || is_fp16, "FlashAttention only supports FP16 and BF16 data types");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor q must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor k must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor v must have contiguous last dimension");
    uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    uint32_t launchBlockDim = blockDim;
    at::Tensor seqlens_k, block_table, out;

    at::Tensor cu_seqlens_q, cu_seqlens_k;
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

    TORCH_CHECK(num_splits >= 0 && num_splits <= static_cast<int64_t>(blockDim),
                "NPU FlashAttention supports num_splits in [0, ", blockDim,
                "] (0 = auto; upper bound = number of AI cores). ");
    TORCH_CHECK(!pack_gqa_.has_value() || !pack_gqa_.value(), "NPU FlashAttention does not support pack_gqa");
    TORCH_CHECK(!min_seqlen_k_.has_value(), "NPU FlashAttention does not support min_seqlen_k");
    TORCH_CHECK(!gather_kv_indices_.has_value(), "NPU FlashAttention does not support gather_kv_indices");
    TORCH_CHECK(!learnable_sink_.has_value(), "NPU FlashAttention does not support learnable_sink");

    if (is_varlen_kv) {
        cu_seqlens_k = cu_seqlens_k_.value();
        TORCH_CHECK(!paged_KV, "If cu_seqlens_k is passed in, then paged table is not supported");
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

    // If seqused_k_ was not provided, derive seqlens_k from tensor shapes or cu_seqlens_k
    if (!seqused_k_.has_value()) {
        if (is_varlen_kv) {
            seqlens_k = cu_seqlens_k.slice(0, 1, cu_seqlens_k.size(0))
                      - cu_seqlens_k.slice(0, 0, cu_seqlens_k.size(0) - 1);
            seqlens_k = seqlens_k.to(torch::kInt32);
        } else {
            int64_t seqlen_k_val = k.size(1);
            seqlens_k = at::full({batch_size}, seqlen_k_val,
                                 at::dtype(torch::kInt32).device(k.device()));
        }
    }

    at::Tensor workspace_tensor;
    at::Tensor mask_gpu_tensor;
    at::Tensor tiling_gpu_tensor;
    uint8_t *tilingDevice = nullptr;
    uint8_t *maskDevice = nullptr;
    bool flashDecodeFlag = false;
    bool is_local = false;

    at::Tensor softmaxlse = at::empty({batch_size, num_heads, seqlen_q}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    if (is_varlen_q) {
        softmaxlse = at::empty({num_heads, sizes[0]}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    }
    softmaxlse.fill_(std::numeric_limits<float>::infinity());

    
        at::Tensor tiling_cpu_tensor = at::empty({static_cast<int64_t>(sizeof(FAInferTilingData))}, at::device(c10::kCPU).dtype(at::kByte));
        FAInferTilingData* tiling_cpu_ptr = reinterpret_cast<FAInferTilingData*>(tiling_cpu_tensor.data_ptr<uint8_t>());
        std::memset(tiling_cpu_ptr, 0, sizeof(FAInferTilingData));

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
        tiling_cpu_ptr->set_scaleValue(softmax_scale);
        tiling_cpu_ptr->set_maxQSeqlen(seqlen_q);
        int32_t max_kv_seqlen = 0;
        for (int32_t i = 0; i < batch_size; i++) {
            max_kv_seqlen = std::max(max_kv_seqlen, seqlens_k_cpu[i]);
        }
        tiling_cpu_ptr->set_maxKvSeqlen(static_cast<uint32_t>(max_kv_seqlen));
        // causal=true is the same as causal=false when seqlen_q == 1 (decode).
        if (seqlen_q == 1) {
            is_causal = false;
        }
        const bool causal_flag = is_causal;
        if (max_kv_seqlen > 0 && window_size_left >= max_kv_seqlen - 1) {
            window_size_left = -1;
        }
        if (seqlen_q > 0 && window_size_right >= seqlen_q - 1) {
            window_size_right = -1;
        }
        if (causal_flag) {
            window_size_right = 0;
        }
        is_causal = (window_size_left < 0 && window_size_right == 0);
        is_local = (window_size_left >= 0 || window_size_right >= 0) && !is_causal;
        if (is_local) {
            tiling_cpu_ptr->set_windowSizeLeft(window_size_left);
            tiling_cpu_ptr->set_windowSizeRight(window_size_right);
            tiling_cpu_ptr->set_maskType(static_cast<uint32_t>(FaiKenel::MaskType::MASK_BAND));
        } else if (is_causal) {
            tiling_cpu_ptr->set_maskType(static_cast<uint32_t>(FaiKenel::MaskType::MASK_CAUSAL));
        }

        uint32_t totalTaskNum = 0;
        uint32_t groupSize = num_heads / num_heads_k;
        for (int32_t batchIdx = 0; batchIdx < batch_size; batchIdx++) {
            uint64_t qSeqlen = seqlen_q;
            if (is_varlen_q) {
                qSeqlen = *(cu_seqlen_q_cpu + batchIdx + 1) - *(cu_seqlen_q_cpu + batchIdx);
            }
            uint64_t kvSeqlen = *(seqlens_k_cpu + batchIdx);
            uint64_t curQNBlockTile = fa_split::GetQNBlockTile(qSeqlen, groupSize);
            uint64_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
            uint64_t curQNBlockNum = qNBlockNumPerGroup * num_heads_k;
            uint64_t curQSBlockTile = fa_split::GetQSBlockTile(kvSeqlen);
            uint64_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
            uint64_t curTaskNum = curQNBlockNum * curQSBlockNum;
            if (batchIdx == 0) {
                tiling_cpu_ptr->set_firstBatchTaskNum(curTaskNum);
            }
            totalTaskNum += curTaskNum;
        }
        tiling_cpu_ptr->set_totalTaskNum(totalTaskNum);

        int64_t maxQSeqlenCalc = 0;
        int64_t minQSeqlenCalc = std::numeric_limits<int64_t>::max();
        int64_t maxKVSeqlenCalc = 0;
        for (int32_t batchIdx = 0; batchIdx < batch_size; batchIdx++) {
            int64_t qSeqlenVal = seqlen_q;
            int64_t kvSeqlenVal = *(seqlens_k_cpu + batchIdx);
            if (is_varlen_q) {
                qSeqlenVal = *(cu_seqlen_q_cpu + batchIdx + 1) - *(cu_seqlen_q_cpu + batchIdx);
                if (is_varlen_kv) {
                    kvSeqlenVal = *(seqlens_k_cpu + batchIdx + 1) - *(seqlens_k_cpu + batchIdx);
                }
            }
            maxQSeqlenCalc = std::max(maxQSeqlenCalc, qSeqlenVal);
            minQSeqlenCalc = std::min(minQSeqlenCalc, qSeqlenVal);
            maxKVSeqlenCalc = std::max(maxKVSeqlenCalc, kvSeqlenVal);
        }
        uint32_t numTasks = static_cast<uint32_t>(batch_size * num_heads_k);
        bool isLongSeq = (static_cast<double>(numTasks) <= 0.8 * blockDim) &&
            (maxKVSeqlenCalc >= static_cast<int64_t>(blockDim) * 512);
        bool isShortSeq = (static_cast<double>(numTasks) <= 0.4 * blockDim) &&
            (maxKVSeqlenCalc >= 1024);
        TORCH_CHECK(num_splits <= 1 || (paged_KV && is_varlen_q),
                    "NPU FlashAttention num_splits>1 currently requires paged KV cache and varlen-q (TND) layout");
        flashDecodeFlag = paged_KV && is_varlen_q &&
            (maxQSeqlenCalc * groupSize <= 128) && (maxQSeqlenCalc <= 16) &&
            (maxKVSeqlenCalc >= 1024) && (minQSeqlenCalc > 0) && (isLongSeq || isShortSeq);
        tiling_cpu_ptr->set_flashDecodeFlag(flashDecodeFlag ? 1U : 0U);
        tiling_cpu_ptr->set_numSplits(num_splits > 0 ? static_cast<uint32_t>(num_splits) : 1U);

        fa_split::SplitContext splitCtx;
        splitCtx.batch_size = batch_size;
        splitCtx.num_heads = num_heads;
        splitCtx.num_heads_k = num_heads_k;
        splitCtx.seqlen_q = seqlen_q;
        splitCtx.head_size_v = head_size_og;
        splitCtx.cu_seqlen_q_cpu = cu_seqlen_q_cpu;
        splitCtx.seqlens_k_cpu = seqlens_k_cpu;
        splitCtx.is_varlen_q = is_varlen_q;
        splitCtx.blockDim = blockDim;
        splitCtx.num_splits = static_cast<int32_t>(num_splits);
        if (flashDecodeFlag) {
            fa_split::splitBN2S1GS2(tiling_cpu_ptr, splitCtx);
            auto needCoreNum = tiling_cpu_ptr->get_needCoreNum();
            if (needCoreNum != 0) {
                launchBlockDim = needCoreNum;
            }
        }

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
        uint64_t splitLseTotalSize = tiling_cpu_ptr->get_splitLseTotalSize();
        uint64_t splitOTotalSize = tiling_cpu_ptr->get_splitOTotalSize();
        int64_t workSpaceSize = static_cast<int64_t>(mm1OutSize + smOnlineOutSize + mm2OutSize
            + UpdateSize + splitLseTotalSize + splitOTotalSize);

        workspace_tensor = at::empty({workSpaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte));
        tiling_cpu_ptr->set_mm1OutSize(mm1OutSize);
        tiling_cpu_ptr->set_smOnlineOutSize(smOnlineOutSize);
        tiling_cpu_ptr->set_mm2OutSize(mm2OutSize);
        tiling_cpu_ptr->set_UpdateSize(UpdateSize);
        tiling_cpu_ptr->set_workSpaceSize(workSpaceSize);
        if (is_causal || is_local) {
            at::Tensor mask_cpu_tensor = at::empty({2048, 2048}, at::device(c10::kCPU).dtype(at::kByte));
            mask_cpu_tensor = at::triu(at::ones_like(mask_cpu_tensor), 1);
            mask_gpu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
        }
        tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));
        tilingDevice = static_cast<uint8_t *>(tiling_gpu_tensor.data_ptr());
        maskDevice = (is_causal || is_local) ? static_cast<uint8_t *>(mask_gpu_tensor.data_ptr()) : nullptr;

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
    if (paged_KV) {
        blockTableDevice = static_cast<uint8_t *>(block_table.data_ptr());
    }
    auto oDevice = static_cast<uint8_t *>(out.data_ptr());
    auto qSeqDevice = static_cast<uint8_t *>(seqlenq_gpu_tensor.data_ptr());
    auto kvSeqDevice = static_cast<uint8_t *>(seqlenk_gpu_tensor.data_ptr());
    auto workspaceDevice = static_cast<uint8_t *>(workspace_tensor.data_ptr());
    auto softmaxLseDevice = static_cast<uint8_t *>(softmaxlse.data_ptr());
    // Forward kernel launches live in fwd_dispatch_<dtype>_<layout>.cpp. Layout
    // is selected at runtime by is_varlen_q (varlen => TND, else BSND); dtype
    // and paged/causal are resolved inside the dispatch. flashDecodeFlag only
    // affects tiling (fa_split), not a FAInfer template arg.
    FwdLaunchArgs fwd_args;
    fwd_args.launchBlockDim = launchBlockDim;
    fwd_args.aclStream = aclStream;
    fwd_args.fftsAddr = fftsAddr;
    fwd_args.is_bf16 = is_bf16;
    fwd_args.paged_KV = paged_KV;
    fwd_args.is_causal = is_causal;
    fwd_args.is_local = is_local;
    fwd_args.flashDecodeFlag = flashDecodeFlag;
    fwd_args.qDevice = qDevice;
    fwd_args.kDevice = kDevice;
    fwd_args.vDevice = vDevice;
    fwd_args.maskDevice = maskDevice;
    fwd_args.blockTableDevice = blockTableDevice;
    fwd_args.oDevice = oDevice;
    fwd_args.softmaxLseDevice = softmaxLseDevice;
    fwd_args.qSeqDevice = qSeqDevice;
    fwd_args.kvSeqDevice = kvSeqDevice;
    fwd_args.workspaceDevice = workspaceDevice;
    fwd_args.tilingDevice = tilingDevice;
    if (is_varlen_q) {
        launch_fwd<true>(fwd_args);
    } else {
        launch_fwd<false>(fwd_args);
    }
    return {out, softmaxlse};
}

PYBIND11_MODULE(flash_attn_npu_4_910, m)
{
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass, with KV-cache");
}