#include <torch/extension.h>
#include <algorithm>
#include <cstring>
#include <limits>

#include "tilingdata.h"
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
// kernel_operator.h (AscendC) defines GM_ADDR, which kernel_common.hpp uses;
// catlass/catlass.hpp defines the Catlass namespace / CATLASS_DEVICE. Both were
// formerly pulled in transitively by mha_fwd_kvcache.cpp / mha_varlen_bwd.cpp,
// which are now compiled as separate dispatch TUs, so include them explicitly
// here, in this order.
#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "kernel_common.hpp"
// common_header.h defines TILING_PARA_NUM and the varlen-bwd tiling constants
// used below; formerly pulled in transitively by mha_varlen_bwd.cpp.
#include "fag_common/common_header.h"
#include "fag_tiling.cpp"
#include "fag_general_host.hpp"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "third_party/op-plugin/op_plugin/include/ops.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "runtime/rt_ffts.h"
#include "fa_metadata_args.h"
#include "fa_split.h"
#include "fwd_dispatch.hpp"
#include "varlen_bwd_dispatch.hpp"
#include <cmath>
#include <unordered_map>

// mha_fwd_kvcache.cpp / mha_varlen_bwd.cpp used to carry these using-directives
// into this TU; restore them so the unqualified KernelCommon constants
// (Q_TILE_CEIL, ...) used by the host helpers below resolve.
using namespace Catlass;
using namespace KernelCommon;

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define ACL_CHECK(expr) TORCH_CHECK((expr) == ACL_SUCCESS, #expr " failed")

extern __global__ __aicpu__ uint32_t ComputeFAMetadataV2(void *args);


struct FwdMaskDerivation {
    bool is_causal;
    bool is_local;
    int64_t window_left;
    int64_t window_right;
    uint32_t maskType;
};

// Window normalization + mask-type derivation, mirroring the host tiling code,
// but bounding the KV side with a host-known upper bound (cache capacity)
// instead of the device-side actual max KV seqlen, so it is usable on the
// scheduler-metadata path where no D2H sync is allowed. Both sides compare
// against the KV bound (matching the host's "both sides vs seqlen_k" rule);
// an over-long window collapses to "no window", and a window that covers the
// whole sequence masks nothing, so a bound larger than the actual seqlen is
// still semantically identical.
static FwdMaskDerivation DeriveFwdMask(bool causal, int64_t window_left, int64_t window_right,
                                       int64_t /*max_seqlen_q*/, int64_t max_seqlen_k_bound)
{
    if (max_seqlen_k_bound > 0 && window_left >= max_seqlen_k_bound) {
        window_left = -1;
    }
    if (max_seqlen_k_bound > 0 && window_right >= max_seqlen_k_bound) {
        window_right = -1;
    }
    if (causal) {
        window_right = 0;
    }
    FwdMaskDerivation derived;
    derived.is_causal = (window_left < 0 && window_right == 0);
    derived.is_local = (window_left >= 0 || window_right >= 0) && !derived.is_causal;
    // Match the host tiling: infinite local side -> finite KV bound, not
    // SPARSE_MODE_INT_MAX (fwd MASK_SWA mishandles INT_MAX right bounds).
    if (derived.is_local) {
        if (window_left < 0) {
            window_left = max_seqlen_k_bound;
        }
        if (window_right < 0) {
            window_right = max_seqlen_k_bound;
        }
    }
    derived.window_left = window_left;
    derived.window_right = window_right;
    derived.maskType = derived.is_local ? static_cast<uint32_t>(FaiKenel::MaskType::MASK_BAND)
        : (derived.is_causal ? static_cast<uint32_t>(FaiKenel::MaskType::MASK_CAUSAL)
                             : static_cast<uint32_t>(FaiKenel::MaskType::NO_MASK));
    return derived;

}

// Enqueue the AICPU scheduler-metadata kernel on a pooled AICPU stream, ordered
// against the current stream with events (no host sync), and return the device
// buffer holding [optional triu mask | FAInferTilingData].
static at::Tensor GetSchedulerMetadataImpl(FAMetadataArgs args,
                                           const at::Tensor &seqlensK,
                                           const std::optional<at::Tensor> &cuSeqlensQ)
{
    const int64_t bytes = static_cast<int64_t>(fa_metadata::MetadataBytes(args.maskType != 0));
    at::Tensor meta = at::empty({bytes}, at::device(at::kPrivateUse1).dtype(at::kByte));
    args.metaOutAddr = reinterpret_cast<uint64_t>(meta.data_ptr());

    c10_npu::NPUStream currentStream = c10_npu::getCurrentNPUStream();
    c10_npu::NPUStream aicpuStream = c10_npu::getNPUStreamFromPool();
    aclrtStream curHandle = currentStream.stream(false);
    aclrtStream aicpuHandle = aicpuStream.stream(false);

    struct MetadataEvents {
        aclrtEvent inputReady = nullptr;
        aclrtEvent metadataDone = nullptr;
    };

    static thread_local std::unordered_map<c10::DeviceIndex, MetadataEvents> eventsByDevice;
    MetadataEvents &events = eventsByDevice[currentStream.device_index()];
    if (events.inputReady == nullptr) {
        ACL_CHECK(aclrtCreateEvent(&events.inputReady));
        ACL_CHECK(aclrtCreateEvent(&events.metadataDone));

    }

    FAMetadataArgs metaArgs = args;
    auto metadata_task = [curHandle, aicpuHandle,
                          inputReady = events.inputReady,
                          metadataDone = events.metadataDone, metaArgs]() mutable -> int {
        ACL_CHECK(aclrtRecordEvent(inputReady, curHandle));
        ACL_CHECK(aclrtStreamWaitEvent(aicpuHandle, inputReady));
        ComputeFAMetadataV2<<<1, nullptr, aicpuHandle>>>(&metaArgs, sizeof(metaArgs));
        ACL_CHECK(aclrtRecordEvent(metadataDone, aicpuHandle));
        ACL_CHECK(aclrtStreamWaitEvent(curHandle, metadataDone));
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fa_metadata_v2", metadata_task);

    c10_npu::NPUCachingAllocator::recordStream(meta.storage().data_ptr(), aicpuStream);
    c10_npu::NPUCachingAllocator::recordStream(seqlensK.storage().data_ptr(), aicpuStream);
    if (cuSeqlensQ.has_value()) {
        c10_npu::NPUCachingAllocator::recordStream(cuSeqlensQ->storage().data_ptr(), aicpuStream);
    }
    return meta;
}

// async copy dropout tiling data
static void PatchTilingDropoutFields(const at::Tensor &schedMd, uint64_t tilingOffset,
                                     float dropoutValue, uint8_t *pDevice,
                                     uint8_t *dropMaskDevice)
{
    if (dropMaskDevice == nullptr) {
        return;
    }
    struct DropFields {
        float dropoutValue;
        uint64_t pDevice;
        uint64_t dropMaskDevice;
    };
    // The Host2Device staging copy is async and reads the source after this function
    // returns; static + same-stream serialization keeps it alive and safe.
    static at::Tensor patchDev;
    static DropFields fields;
    if (!patchDev.defined()) {
        patchDev = torch::empty({static_cast<int64_t>(sizeof(DropFields))},
                                at::device(at::kPrivateUse1).dtype(at::kByte));
    }
    fields.dropoutValue = dropoutValue;
    fields.pDevice = reinterpret_cast<uint64_t>(pDevice);
    fields.dropMaskDevice = reinterpret_cast<uint64_t>(dropMaskDevice);

    at::Tensor patchCpu = torch::from_blob(&fields, {static_cast<int64_t>(sizeof(DropFields))},
                                           at::TensorOptions().dtype(torch::kUInt8));
    patchDev.copy_(patchCpu);  // H2D staging, current stream

    auto tilingView = schedMd.narrow(0, static_cast<int64_t>(tilingOffset),
                                     static_cast<int64_t>(sizeof(FAInferTilingData)));
    tilingView.narrow(0, offsetof(FAInferTilingData, dropoutValue), 4)
        .view(torch::kFloat32)
        .copy_(patchDev.narrow(0, offsetof(DropFields, dropoutValue), 4).view(torch::kFloat32));
    tilingView.narrow(0, offsetof(FAInferTilingData, pDevice), 8)
        .view(torch::kInt64)
        .copy_(patchDev.narrow(0, offsetof(DropFields, pDevice), 8).view(torch::kInt64));
    tilingView.narrow(0, offsetof(FAInferTilingData, dropMaskDevice), 8)
        .view(torch::kInt64)
        .copy_(patchDev.narrow(0, offsetof(DropFields, dropMaskDevice), 8).view(torch::kInt64));
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
                int num_splits,
                std::optional<at::Tensor> scheduler_metadata_
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

    uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    uint32_t launchBlockDim = blockDim;
    at::Tensor seqlens_k, block_table, out;
    at::Tensor k, v, rotary_cos, rotary_sin, cache_batch_idx, alibi_slopes;
    at::Tensor workspace_tensor;
    at::Tensor mask_gpu_tensor;
    at::Tensor tiling_gpu_tensor;
    uint8_t *tilingDevice = nullptr;
    uint8_t *maskDevice = nullptr;
    bool is_local = false;
    bool flashDecodeFlag = false;
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
    TORCH_CHECK(softcap >= 0.0f, "softcap must be non-negative (0.0 disables softcap)");
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


    const bool appendKV = k_.has_value();
    if (!seqlens_k_.has_value()) {
        int64_t seqlen_k_val = kcache.size(1);
        seqlens_k = at::full({batch_size}, seqlen_k_val,
                             at::dtype(torch::kInt32).device(kcache.device()));
    }
    int64_t kvCacheSeqlen = 0;
    int64_t kvNewSeqlen = 0;
    if (appendKV) {
        auto k_new = k_.value();
        auto v_new = v_.value();
        TORCH_CHECK(v_.has_value(),
                    "append-KV: v must be provided together with k");
        TORCH_CHECK(seqlens_k_.has_value(),
                    "append-KV requires seqlens_k (cache_seqlens) with the per-batch cached lengths");
        if (paged_KV) {
            TORCH_CHECK(max_num_blocks_per_seq > 0, "append-KV with paged KV cache requires a non-empty page table");
            // v2 do not support TND layout, so skip checking TND layout
            TORCH_CHECK(batch_size == static_cast<int32_t>(block_table.size(0)),
                        "append-KV with paged KV cache requires batch_size to equal the block table batch size");
        }
        TORCH_CHECK(k_new.dtype() == q_dtype && v_new.dtype() == q_dtype,
                    "k/v must have the same dtype as q");
        TORCH_CHECK(k_new.dim() == 4 && v_new.dim() == 4, "append-KV k/v must be (b, s_new, h_k, d)");
        TORCH_CHECK(k_new.size(0) == batch_size && v_new.size(0) == batch_size,
                    "k/v batch dim mismatch");
        TORCH_CHECK(k_new.size(1) == v_new.size(1) && k_new.size(1) > 0,
                    "append-KV requires a uniform new length s_new > 0");
        TORCH_CHECK(k_new.size(2) == num_heads_k && v_new.size(2) == num_heads_k,
                    "k/v head dim mismatch");
        TORCH_CHECK(k_new.size(3) == head_size_og && v_new.size(3) == head_size_og,
                    "k/v head size mismatch");
        // TODO: check if headdim padding enough?
        TORCH_CHECK(head_size_og % 16 == 0,
                    "append-KV requires head dim to be a multiple of 16");
        // Per-batch cache capacity: paged = page-table row length, else the padded cache S dim.
        kvCacheSeqlen = paged_KV
            ? static_cast<int64_t>(max_num_blocks_per_seq) * page_block_size
            : kcache.size(1);
        kvNewSeqlen = k_new.size(1);
        at::Tensor seqlens_k_cpu_check = seqlens_k.to(at::Device(at::kCPU));
        const int32_t *sl_check = static_cast<const int32_t *>(seqlens_k_cpu_check.data_ptr());
        for (int32_t i = 0; i < batch_size; i++) {
            TORCH_CHECK(sl_check[i] >= 0 && sl_check[i] + kvNewSeqlen <= kvCacheSeqlen,
                        "append-KV: batch ", i, " cached length ", sl_check[i], " + new length ",
                        kvNewSeqlen, " exceeds cache capacity ", kvCacheSeqlen);
        }
    }



    bool has_softcap = (softcap > 0.0f);
    at::Tensor softmaxlse = at::empty({batch_size, num_heads, seqlen_q}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    softmaxlse.fill_(std::numeric_limits<float>::infinity());

    if (scheduler_metadata_.has_value()) {
        auto schedMd = scheduler_metadata_.value();
        TORCH_CHECK(schedMd.dtype() == at::kByte, "scheduler_metadata must be a byte tensor");
        TORCH_CHECK(schedMd.is_contiguous(), "scheduler_metadata must be contiguous");
        TORCH_CHECK(schedMd.device().type() == at::kPrivateUse1, "scheduler_metadata must be an NPU tensor");
        // Derive the mask axes (causal/SWA) from this call's arguments the same
        // way get_scheduler_metadata did when producing the buffer, so that the
        // template selection and the tiling offset match the AICPU-written
        // tiling. The metadata must have been created with matching causal /
        // window_size / softcap / softmax_scale / seqlen-bound arguments.
        const int64_t kvSeqlenBound = paged_KV
            ? static_cast<int64_t>(max_num_blocks_per_seq) * page_block_size
            : static_cast<int64_t>(kcache.size(1));
        FwdMaskDerivation maskDer = DeriveFwdMask(is_causal, window_size_left, window_size_right,
                                                  seqlen_q, kvSeqlenBound);
        is_causal = maskDer.is_causal;
        is_local = maskDer.is_local;
        const bool hasMask = maskDer.maskType != static_cast<uint32_t>(FaiKenel::MaskType::NO_MASK);
        TORCH_CHECK(static_cast<uint64_t>(schedMd.nbytes()) == fa_metadata::MetadataBytes(hasMask),
                    "scheduler_metadata buffer size must exactly match this call's "
                    "causal/window-derived layout");
        auto metaBase = static_cast<uint8_t *>(schedMd.data_ptr());
        tilingDevice = metaBase + fa_metadata::TilingOffset(hasMask);
        maskDevice = hasMask ? metaBase : nullptr;
        int64_t wsBase = static_cast<int64_t>(fa_metadata::WorkSpaceSize(blockDim));
        int64_t wsSplit = 0;
        // The AICPU may trigger flash-decode split-KV on device; reserve the
        // split workspace upper bound since the host no longer knows it.
        if (paged_KV && seqlen_q * (num_heads / num_heads_k) <= 128 && seqlen_q <= 16) {
            int64_t kvSegUpper = kvSeqlenBound / 512 + 1;
            int64_t lseTasksUpper = static_cast<int64_t>(batch_size) * num_heads * seqlen_q * kvSegUpper * 2;
            wsSplit = lseTasksUpper * 4 + lseTasksUpper * head_size_og * 4;
        }
        workspace_tensor = at::empty({wsBase + wsSplit}, at::device(at::kPrivateUse1).dtype(at::kByte));
        launchBlockDim = blockDim;
    } else {
    at::Tensor tiling_cpu_tensor = at::empty({static_cast<int64_t>(sizeof(FAInferTilingData))},
                                             at::device(c10::kCPU).dtype(at::kByte));

    FAInferTilingData* tiling_cpu_ptr = reinterpret_cast<FAInferTilingData*>(tiling_cpu_tensor.data_ptr<uint8_t>());
    std::memset(tiling_cpu_ptr, 0, sizeof(FAInferTilingData));

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
    if (has_softcap) {
        tiling_cpu_ptr->set_scaleValue(softmax_scale / softcap);
    } else {
        tiling_cpu_ptr->set_scaleValue(softmax_scale);
    }
    tiling_cpu_ptr->set_softcapValue(softcap);
    tiling_cpu_ptr->set_maxQSeqlen(seqlen_q);
    // Append-KV: total S per batch = old (cached) + new.
    int32_t max_kv_seqlen = 0;
    for (int32_t i = 0; i < batch_size; i++) {
        int32_t kvSeqlenOld = seqlens_k_cpu[i];
        max_kv_seqlen = std::max(max_kv_seqlen, kvSeqlenOld + static_cast<int32_t>(kvNewSeqlen));
    }
    tiling_cpu_ptr->set_maxKvSeqlen(static_cast<uint32_t>(max_kv_seqlen));
    tiling_cpu_ptr->set_kvNewSeqlen(appendKV ? static_cast<uint32_t>(kvNewSeqlen) : 0U);
    tiling_cpu_ptr->set_kvCacheSeqlen(appendKV ? static_cast<uint32_t>(kvCacheSeqlen) : 0U);

    // Match GPU: both sides vs seqlen_k (not right vs Sq-1).
    if (max_kv_seqlen > 0 && window_size_left >= max_kv_seqlen) {
        window_size_left = -1;
    }
    if (max_kv_seqlen > 0 && window_size_right >= max_kv_seqlen) {
        window_size_right = -1;
    }
    if (is_causal) {
        window_size_right = 0;
    }
    is_causal = (window_size_left < 0 && window_size_right == 0);
    is_local = (window_size_left >= 0 || window_size_right >= 0) && !is_causal;
    TORCH_CHECK(!(appendKV && is_local),
                "NPU FlashAttention append-KV does not support sliding-window attention (window_size) yet");
    // Match Tri Dao set_params_fprop: infinite local side → seqlen_k (finite),
    // not SPARSE_MODE_INT_MAX (fwd MASK_SWA mishandles INT_MAX right bounds).
    if (is_local) {
        if (window_size_left < 0) {
            window_size_left = max_kv_seqlen;
        }
        if (window_size_right < 0) {
            window_size_right = max_kv_seqlen;
        }
    }

    uint32_t totalTaskNum = 0;
    uint32_t groupSize = num_heads / num_heads_k;
    for (int32_t batchIdx = 0; batchIdx < batch_size; batchIdx++) {
        uint64_t qSeqlen = seqlen_q;
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

    uint32_t numTasks = static_cast<uint32_t>(batch_size * num_heads_k);
    bool isLongSeq = (static_cast<double>(numTasks) <= 0.8 * blockDim) &&
        (max_kv_seqlen >= static_cast<int32_t>(blockDim) * 512);
    bool isShortSeq = (static_cast<double>(numTasks) <= 0.4 * blockDim) &&
        (max_kv_seqlen >= 1024);
    flashDecodeFlag = paged_KV && head_size_og <= 128 &&
        (seqlen_q * groupSize <= 128) && (seqlen_q <= 16) &&
        (max_kv_seqlen >= 1024) && (seqlen_q > 0) && (isLongSeq || isShortSeq);
    tiling_cpu_ptr->set_flashDecodeFlag(flashDecodeFlag ? 1U : 0U);

    fa_split::SplitContext splitCtx;
    splitCtx.batch_size = batch_size;
    splitCtx.num_heads = num_heads;
    splitCtx.num_heads_k = num_heads_k;
    splitCtx.seqlen_q = seqlen_q;
    splitCtx.head_size_v = head_size_og;
    splitCtx.cu_seqlen_q_cpu = nullptr;
    splitCtx.seqlens_k_cpu = seqlens_k_cpu;
    splitCtx.is_varlen_q = false;
    splitCtx.blockDim = blockDim;
    splitCtx.kvNewSeqlen = appendKV ? static_cast<int32_t>(kvNewSeqlen) : 0;
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
    // Empty FD splits never write partials; init like FA GPU (O=0, LSE=-inf).
    if (flashDecodeFlag && (splitLseTotalSize > 0 || splitOTotalSize > 0)) {
        auto float_opts = workspace_tensor.options().dtype(at::kFloat);
        if (splitLseTotalSize > 0) {
            TORCH_CHECK(splitLseTotalSize % sizeof(float) == 0,
                        "splitLseTotalSize must be a multiple of sizeof(float)");
            at::Tensor lse_init = at::full(
                {static_cast<int64_t>(splitLseTotalSize / sizeof(float))},
                -std::numeric_limits<float>::infinity(), float_opts);
            workspace_tensor.narrow(/*dim=*/0, /*start=*/0,
                static_cast<int64_t>(splitLseTotalSize))
                .copy_(lse_init.view(at::kByte));
        }
        if (splitOTotalSize > 0) {
            TORCH_CHECK(splitOTotalSize % sizeof(float) == 0,
                        "splitOTotalSize must be a multiple of sizeof(float)");
            at::Tensor o_init = at::zeros(
                {static_cast<int64_t>(splitOTotalSize / sizeof(float))}, float_opts);
            workspace_tensor.narrow(/*dim=*/0,
                static_cast<int64_t>(splitLseTotalSize),
                static_cast<int64_t>(splitOTotalSize))
                .copy_(o_init.view(at::kByte));
        }
    }

    tiling_cpu_ptr->set_mm1OutSize(mm1OutSize);
    tiling_cpu_ptr->set_smOnlineOutSize(smOnlineOutSize);
    tiling_cpu_ptr->set_mm2OutSize(mm2OutSize);
    tiling_cpu_ptr->set_UpdateSize(UpdateSize);
    tiling_cpu_ptr->set_workSpaceSize(workSpaceSize);

    if (is_local) {
        tiling_cpu_ptr->set_windowSizeLeft(window_size_left);
        tiling_cpu_ptr->set_windowSizeRight(window_size_right);
        tiling_cpu_ptr->set_maskType(static_cast<uint32_t>(FaiKenel::MaskType::MASK_BAND));
    } else if (is_causal) {
        tiling_cpu_ptr->set_maskType(static_cast<uint32_t>(FaiKenel::MaskType::MASK_CAUSAL));
    }
    if (is_causal || is_local) {
        at::Tensor mask_cpu_tensor = at::empty({2048, 2048}, at::device(c10::kCPU).dtype(at::kByte));
        mask_cpu_tensor = at::triu(at::ones_like(mask_cpu_tensor), 1);
        mask_gpu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
        maskDevice = static_cast<uint8_t *>(mask_gpu_tensor.data_ptr());
    }
    tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));
    tilingDevice = static_cast<uint8_t *>(tiling_gpu_tensor.data_ptr());
    }

    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtError_t error = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    auto qDevice = static_cast<uint8_t *>(q.data_ptr());
    auto kDevice = static_cast<uint8_t *>(kcache.data_ptr());
    auto vDevice = static_cast<uint8_t *>(vcache.data_ptr());
    uint8_t * blockTableDevice = nullptr;
    if (paged_KV) {
        blockTableDevice = static_cast<uint8_t *>(block_table.data_ptr());
    }
    auto oDevice = static_cast<uint8_t *>(out.data_ptr());
    auto qSeqDevice = static_cast<uint8_t *>(seqlens_k.data_ptr());
    auto kvSeqDevice = static_cast<uint8_t *>(seqlens_k.data_ptr());
    auto workspaceDevice = static_cast<uint8_t *>(workspace_tensor.data_ptr());
    auto softmaxLseDevice = static_cast<uint8_t *>(softmaxlse.data_ptr());
    // Forward kernel launches live in fwd_dispatch_{bf16,fp16}.cpp. BSND path
    // (IS_TND=false); flash-decode is a runtime tiling flag read by the kernel.
    FwdLaunchArgs fwd_args;
    fwd_args.blockDim = launchBlockDim;
    fwd_args.aclStream = aclStream;
    fwd_args.fftsAddr = fftsAddr;
    fwd_args.is_bf16 = is_bf16;
    fwd_args.paged_KV = paged_KV;
    fwd_args.is_causal = is_causal;
    fwd_args.is_local = is_local;
    fwd_args.flashDecodeFlag = flashDecodeFlag;
    fwd_args.has_softcap = has_softcap;
    fwd_args.qDevice = qDevice;
    fwd_args.kDevice = kDevice;
    fwd_args.vDevice = vDevice;
    fwd_args.kNewDevice = appendKV ? static_cast<uint8_t *>(k_.value().data_ptr()) : nullptr;
    fwd_args.vNewDevice = appendKV ? static_cast<uint8_t *>(v_.value().data_ptr()) : nullptr;
    fwd_args.maskDevice = maskDevice;
    fwd_args.blockTableDevice = blockTableDevice;
    fwd_args.oDevice = oDevice;
    fwd_args.softmaxLseDevice = softmaxLseDevice;
    fwd_args.qSeqDevice = qSeqDevice;
    fwd_args.kvSeqDevice = kvSeqDevice;
    fwd_args.workspaceDevice = workspaceDevice;
    fwd_args.tilingDevice = tilingDevice;
    auto launch_fa_infer = [fwd_args]() -> int {
        launch_fwd<false>(fwd_args);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fa_infer", launch_fa_infer);
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
        std::optional<at::Generator> gen_,
        std::optional<at::Tensor> scheduler_metadata_)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    auto q_dtype = q.dtype();
    bool is_bf16 = q.dtype() == torch::kBFloat16;
    // parameters checking
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(softcap >= 0.0f, "softcap must be non-negative (0.0 disables softcap)");
    TORCH_CHECK(p_dropout >= 0.0f && p_dropout < 1.0f, "p_dropout must be in [0.0, 1.0)");

    // block unsupported params
    TORCH_CHECK(!alibi_slopes_.has_value(), "NPU FlashAttention does not support alibi_slopes.");

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

    bool is_local = false;
    // Match Tri Dao GPU: both sides vs seqlen_k.
    if (seqlen_k > 0 && window_size_left >= seqlen_k) {
        window_size_left = -1;
    }
    if (seqlen_k > 0 && window_size_right >= seqlen_k) {
        window_size_right = -1;
    }
    if (is_causal) {
        window_size_right = 0;
    }
    is_causal = (window_size_left < 0 && window_size_right == 0);
    is_local = (window_size_left >= 0 || window_size_right >= 0) && !is_causal;
    if (is_local) {
        if (window_size_left < 0) {
            window_size_left = seqlen_k;
        }
        if (window_size_right < 0) {
            window_size_right = seqlen_k;
        }
    }

    // init output tensors
    at::Tensor out = (out_.has_value()) ? out_.value() : torch::empty_like(q);
    auto opts = q.options().device(at::kPrivateUse1);
    auto p = torch::empty({0}, opts);
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::zeros({batch_size, num_heads, seqlen_q, seqlen_k}, opts);
    }
    auto rng_state = torch::empty({2}, at::device(c10::kCPU).dtype(torch::kInt64));

    // init mask / tiling: host computes them unless the AICPU already did
    // (scheduler-metadata path).
    at::Tensor mask_gpu_tensor;
    uint8_t * maskDevice = nullptr;
    at::Tensor tiling_gpu_tensor;
    uint8_t * tilingDevice = nullptr;
    at::Tensor workspace_tensor;
    uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    bool has_softcap = (softcap > 0.0f);

    // init dropout
    bool has_dropout = p_dropout > 0.0f;
    at::Tensor drop_mask_npu_tensor;
    if (has_dropout) {
        uint64_t num_elems = static_cast<uint64_t>(batch_size) * num_heads * seqlen_q * seqlen_k;
        at::Generator gen = gen_.has_value() ? gen_.value() : at_npu::detail::getDefaultNPUGenerator();
        std::lock_guard<std::mutex> lock(gen.mutex());
        auto [seed, offset] = gen.get<at_npu::NPUGeneratorImpl>()->philox_engine_inputs(num_elems);
        int64_t drop_mask_bit_num = static_cast<int64_t>(batch_size) * num_heads * seqlen_q * ((seqlen_k + 7) / 8 * 8);
        drop_mask_npu_tensor = at_npu::native::npu_dropout_gen_mask(
            torch::empty({0}, at::device(at::kPrivateUse1).dtype(at::kFloat)), {drop_mask_bit_num}, p_dropout, seed,
            offset, false, false);
        uint64_t* rng_state_ptr = reinterpret_cast<uint64_t*>(const_cast<void*>(rng_state.data_ptr()));
        rng_state_ptr[0] = seed;
        rng_state_ptr[1] = offset;
    }

    // init softmax lse — head-major BNS: {batch, num_heads, seqlen_q} (matches v3).
    at::Tensor softmaxlse = at::empty({batch_size, num_heads, seqlen_q},
        at::device(at::kPrivateUse1).dtype(at::kFloat));
    softmaxlse.fill_(std::numeric_limits<float>::infinity());
    auto softmaxLseDevice = static_cast<uint8_t *>(const_cast<void *>(softmaxlse.data_ptr()));

    // ffts related
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtError_t error = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);

    if (scheduler_metadata_.has_value()) {
        auto schedMd = scheduler_metadata_.value();
        TORCH_CHECK(schedMd.dtype() == at::kByte, "scheduler_metadata must be a byte tensor");
        TORCH_CHECK(schedMd.is_contiguous(), "scheduler_metadata must be contiguous");
        TORCH_CHECK(schedMd.device().type() == at::kPrivateUse1, "scheduler_metadata must be an NPU tensor");
        // is_causal/is_local above were normalized with the exact shape-derived
        // seqlens, identical to what get_scheduler_metadata derived; the
        // metadata must have been created with matching causal / window_size /
        // softcap / softmax_scale arguments.
        const bool hasMask = is_causal || is_local;
        TORCH_CHECK(static_cast<uint64_t>(schedMd.nbytes()) == fa_metadata::MetadataBytes(hasMask),
                    "scheduler_metadata buffer size must exactly match this call's "
                    "causal/window-derived layout");
        auto metaBase = static_cast<uint8_t *>(schedMd.data_ptr());
        tilingDevice = metaBase + fa_metadata::TilingOffset(hasMask);
        maskDevice = hasMask ? metaBase : nullptr;
        workspace_tensor = at::empty({static_cast<int64_t>(fa_metadata::WorkSpaceSize(blockDim))},
                                     at::device(at::kPrivateUse1).dtype(at::kByte));
        // The AICPU tiling does not carry the dropout fields; patch them in
        // with stream-ordered copies (after the AICPU metadataDone event).
        PatchTilingDropoutFields(schedMd, fa_metadata::TilingOffset(hasMask),
                                 1.0f / (1.0f - p_dropout),
                                 return_softmax ? static_cast<uint8_t *>(const_cast<void *>(p.data_ptr())) : nullptr,
                                 has_dropout ? static_cast<uint8_t *>(const_cast<void *>(drop_mask_npu_tensor.data_ptr())) : nullptr);
    } else {
    if (is_causal || is_local) {
        at::Tensor mask_cpu_tensor = at::empty({2048, 2048}, at::device(c10::kCPU).dtype(at::kByte));
        mask_cpu_tensor = at::triu(at::ones_like(mask_cpu_tensor), 1);
        mask_gpu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
        maskDevice = static_cast<uint8_t *>(const_cast<void *>(mask_gpu_tensor.data_ptr()));
    }

    // set worksapce
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
    workspace_tensor = at::empty({workSpaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte));

    // tiling
    at::Tensor tiling_cpu_tensor = at::empty({static_cast<int64_t>(sizeof(FAInferTilingData))},
                                             at::device(c10::kCPU).dtype(at::kByte));
    FAInferTilingData* tiling_cpu_ptr = reinterpret_cast<FAInferTilingData*>(tiling_cpu_tensor.data_ptr<uint8_t>());
    std::memset(tiling_cpu_ptr, 0, sizeof(FAInferTilingData));
    tiling_cpu_ptr->set_batch(static_cast<uint32_t>(batch_size));
    tiling_cpu_ptr->set_numHeads(static_cast<uint32_t>(num_heads));
    tiling_cpu_ptr->set_kvHeads(static_cast<uint32_t>(num_heads_k));
    tiling_cpu_ptr->set_embeddingSize(static_cast<uint32_t>(head_size));
    tiling_cpu_ptr->set_embeddingSizeV(static_cast<uint32_t>(head_size));
    tiling_cpu_ptr->set_numBlocks(static_cast<uint32_t>(0));
    tiling_cpu_ptr->set_blockSize(static_cast<uint32_t>(128));
    tiling_cpu_ptr->set_maxNumBlocksPerBatch(static_cast<uint32_t>(0));
    if (is_local) {
        tiling_cpu_ptr->set_windowSizeLeft(window_size_left);
        tiling_cpu_ptr->set_windowSizeRight(window_size_right);
        tiling_cpu_ptr->set_maskType(static_cast<uint32_t>(FaiKenel::MaskType::MASK_BAND));
    } else if (is_causal) {
        tiling_cpu_ptr->set_maskType(static_cast<uint32_t>(FaiKenel::MaskType::MASK_CAUSAL));
    }
    if (has_softcap) {
        tiling_cpu_ptr->set_scaleValue(softmax_scale / softcap);
    } else {
        tiling_cpu_ptr->set_scaleValue(softmax_scale);
    }
    tiling_cpu_ptr->set_softcapValue(softcap);
    tiling_cpu_ptr->set_dropoutValue(1.0f / (1.0f - p_dropout));
    tiling_cpu_ptr->set_maxQSeqlen(seqlen_q);
    tiling_cpu_ptr->set_maxKvSeqlen(seqlen_k);
    tiling_cpu_ptr->set_mm1OutSize(mm1OutSize);
    tiling_cpu_ptr->set_smOnlineOutSize(smOnlineOutSize);
    tiling_cpu_ptr->set_mm2OutSize(mm2OutSize);
    tiling_cpu_ptr->set_UpdateSize(UpdateSize);
    tiling_cpu_ptr->set_workSpaceSize(workSpaceSize);
    tiling_cpu_ptr->set_pDevice(return_softmax ? static_cast<uint8_t*>(const_cast<void*>(p.data_ptr())) : nullptr);
    tiling_cpu_ptr->set_dropMaskDevice(
        has_dropout ? static_cast<uint8_t*>(const_cast<void*>(drop_mask_npu_tensor.data_ptr())) : nullptr);

    uint32_t totalTaskNum = 0;
    uint32_t groupSize = num_heads / num_heads_k;
    uint32_t firstBatchTaskNum = 0;
    for (int32_t batchIdx = 0; batchIdx < batch_size; batchIdx++) {
        uint64_t qSeqlen = seqlen_q;
        uint64_t kvSeqlen = seqlen_k;
        uint64_t curQNBlockTile = fa_split::GetQNBlockTile(qSeqlen, groupSize);
        uint64_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
        uint64_t curQNBlockNum = qNBlockNumPerGroup * num_heads_k;
        uint64_t curQSBlockTile = fa_split::GetQSBlockTile(kvSeqlen);
        uint64_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
        uint64_t curTaskNum = curQNBlockNum * curQSBlockNum;
        if (batchIdx == 0) {
            tiling_cpu_ptr->set_firstBatchTaskNum(curTaskNum);
            firstBatchTaskNum = curTaskNum;
        }
        totalTaskNum += curTaskNum;
    }
    tiling_cpu_ptr->set_totalTaskNum(totalTaskNum);
    tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));
    tilingDevice = static_cast<uint8_t *>(const_cast<void *>(tiling_gpu_tensor.data_ptr()));
    }

    // device ptrs
    auto qDevice = static_cast<uint8_t *>(const_cast<void *>(q.data_ptr()));
    auto kDevice = static_cast<uint8_t *>(const_cast<void *>(k.data_ptr()));
    auto vDevice = static_cast<uint8_t *>(const_cast<void *>(v.data_ptr()));
    at::Tensor seqlenq_gpu_tensor = at::full({batch_size}, seqlen_q).to(at::Device(at::kPrivateUse1)).to(at::kInt);
    at::Tensor seqlenk_gpu_tensor = at::full({batch_size}, seqlen_k).to(at::Device(at::kPrivateUse1)).to(at::kInt);
    auto qSeqDevice = static_cast<uint8_t *>(const_cast<void *>(seqlenq_gpu_tensor.data_ptr()));
    auto kvSeqDevice = static_cast<uint8_t *>(const_cast<void *>(seqlenk_gpu_tensor.data_ptr()));
    auto workspaceDevice = static_cast<uint8_t *>(const_cast<void *>(workspace_tensor.data_ptr()));
    auto oDevice = static_cast<uint8_t *>(const_cast<void *>(out.data_ptr()));
    uint8_t * blockTableDevice = nullptr; // will not be used in non-kvcahce fwd api

    // run kernel
    // BSND, non-paged forward (IS_TND=false, paged_KV=false, no flash-decode).
    FwdLaunchArgs fwd_args;
    fwd_args.blockDim = blockDim;
    fwd_args.aclStream = aclStream;
    fwd_args.fftsAddr = fftsAddr;
    fwd_args.is_bf16 = is_bf16;
    fwd_args.paged_KV = false;
    fwd_args.is_causal = is_causal;
    fwd_args.is_local = is_local;
    fwd_args.flashDecodeFlag = false;
    fwd_args.has_softcap = has_softcap;
    fwd_args.return_softmax = return_softmax;
    fwd_args.has_dropout = has_dropout;
    fwd_args.qDevice = qDevice;
    fwd_args.kDevice = kDevice;
    fwd_args.vDevice = vDevice;
    fwd_args.kNewDevice = nullptr;
    fwd_args.vNewDevice = nullptr;
    fwd_args.maskDevice = maskDevice;
    fwd_args.blockTableDevice = blockTableDevice;
    fwd_args.oDevice = oDevice;
    fwd_args.softmaxLseDevice = softmaxLseDevice;
    fwd_args.qSeqDevice = qSeqDevice;
    fwd_args.kvSeqDevice = kvSeqDevice;
    fwd_args.workspaceDevice = workspaceDevice;
    fwd_args.tilingDevice = tilingDevice;
    auto launch_fa_infer = [fwd_args]() -> int {
        launch_fwd<false>(fwd_args);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fa_infer", launch_fa_infer);

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
               std::optional<at::Generator> gen_,
               std::optional<at::Tensor> scheduler_metadata_)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    at::Tensor workspace_tensor;
    at::Tensor mask_gpu_tensor;
    at::Tensor tiling_gpu_tensor;
    uint8_t *tilingDevice = nullptr;
    uint8_t *maskDevice = nullptr;

    bool is_bf16 = q.dtype() == torch::kBFloat16;
    bool is_fp16 = q.dtype() == torch::kFloat16;
    const bool paged_KV = block_table_.has_value();

    // 校验拦截不支持的模式
    TORCH_CHECK(is_bf16 || is_fp16, "NPU FlashAttention only supports Float16 or BFloat16.");
    TORCH_CHECK(!seqused_k_.has_value(), "NPU FlashAttention does not support seqused_k.");
    TORCH_CHECK(!leftpad_k_.has_value(), "NPU FlashAttention does not support leftpad_k.");
    TORCH_CHECK(!alibi_slopes_.has_value(), "NPU FlashAttention does not support alibi_slopes.");
    TORCH_CHECK(p_dropout >= 0.0f && p_dropout < 1.0f, "p_dropout must be in [0.0, 1.0)");
    TORCH_CHECK(!zero_tensors, "NPU FlashAttention does not support zero_tensors.");
    TORCH_CHECK(softcap >= 0.0f, "softcap must be non-negative (0.0 disables softcap)");
    TORCH_CHECK(k.dtype() == q.dtype(), "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q.dtype(), "query and value must have the same dtype");
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

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
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    const auto sizes = q.sizes();
    int T = sizes[0];
    int num_heads = sizes[1];
    const int head_size_og = sizes[2];
    const int batch_size = cu_seqlens_q.numel() - 1;

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : k.size(0);
    const int page_block_size = !paged_KV ? 128 : k.size(1);
    const int num_heads_k = paged_KV ? k.size(2) : k.size(1);

    TORCH_CHECK(head_size_og <= 256, "FlashAttention only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (!paged_KV) {
        const int total_k = k.size(0);
        CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
        CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);
    } else {
        CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    }

    if (paged_KV) {
        seqlens_k = (cu_seqlens_k.slice(0, 1, cu_seqlens_k.size(0)) -
                     cu_seqlens_k.slice(0, 0, cu_seqlens_k.size(0) - 1))
                        .to(torch::kInt).contiguous();
    }

    bool is_local = false;
    // Match Tri Dao GPU: both sides vs max_seqlen_k.
    if (max_seqlen_k > 0 && window_size_left >= max_seqlen_k) {
        window_size_left = -1;
    }
    if (max_seqlen_k > 0 && window_size_right >= max_seqlen_k) {
        window_size_right = -1;
    }
    if (is_causal) {
        window_size_right = 0;
    }
    is_causal = (window_size_left < 0 && window_size_right == 0);
    is_local = (window_size_left >= 0 || window_size_right >= 0) && !is_causal;
    if (is_local) {
        if (window_size_left < 0) {
            window_size_left = max_seqlen_k;
        }
        if (window_size_right < 0) {
            window_size_right = max_seqlen_k;
        }
    }

    at::Tensor out = (out_.has_value()) ? out_.value() : torch::empty_like(q);
    auto opts = q.options().device(at::kPrivateUse1);
    auto p = torch::empty({0}, opts);
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::zeros({batch_size, num_heads, max_seqlen_q, max_seqlen_k}, opts);
    }
    auto rng_state = torch::empty({2}, at::device(c10::kCPU).dtype(torch::kInt64));

    bool has_dropout = p_dropout > 0.0f;
    at::Tensor drop_mask_npu_tensor;
    if (has_dropout) {
        uint64_t num_elems = static_cast<uint64_t>(batch_size) * num_heads * max_seqlen_q * max_seqlen_k;
        at::Generator gen = gen_.has_value() ? gen_.value() : at_npu::detail::getDefaultNPUGenerator();
        std::lock_guard<std::mutex> lock(gen.mutex());
        auto [seed, offset] = gen.get<at_npu::NPUGeneratorImpl>()->philox_engine_inputs(num_elems);
        int64_t drop_mask_bit_num =
            static_cast<int64_t>(batch_size) * num_heads * max_seqlen_q * ((max_seqlen_k + 7) / 8 * 8);
        drop_mask_npu_tensor = at_npu::native::npu_dropout_gen_mask(
            torch::empty({0}, at::device(at::kPrivateUse1).dtype(at::kFloat)), {drop_mask_bit_num}, p_dropout, seed,
            offset, false, false);
        uint64_t* rng_state_ptr = reinterpret_cast<uint64_t*>(const_cast<void*>(rng_state.data_ptr()));
        rng_state_ptr[0] = seed;
        rng_state_ptr[1] = offset;
    }

    bool has_softcap = (softcap > 0.0f);
    // LSE output is head-major NT: {num_heads, T} (matches v3).
    at::Tensor softmaxlse = at::empty({num_heads, T}, at::device(at::kPrivateUse1).dtype(at::kFloat)); // lse
    softmaxlse.fill_(std::numeric_limits<float>::infinity());

    if (scheduler_metadata_.has_value()) {
        auto schedMd = scheduler_metadata_.value();
        TORCH_CHECK(schedMd.dtype() == at::kByte, "scheduler_metadata must be a byte tensor");
        TORCH_CHECK(schedMd.is_contiguous(), "scheduler_metadata must be contiguous");
        TORCH_CHECK(schedMd.device().type() == at::kPrivateUse1, "scheduler_metadata must be an NPU tensor");
        // is_causal/is_local above were normalized with the declared max
        // seqlens, identical to what get_scheduler_metadata derived; the
        // metadata must have been created with matching causal / window_size /
        // softcap / softmax_scale arguments.
        const bool hasMask = is_causal || is_local;
        TORCH_CHECK(static_cast<uint64_t>(schedMd.nbytes()) == fa_metadata::MetadataBytes(hasMask),
                    "scheduler_metadata buffer size must exactly match this call's "
                    "causal/window-derived layout");
        auto metaBase = static_cast<uint8_t *>(schedMd.data_ptr());
        tilingDevice = metaBase + fa_metadata::TilingOffset(hasMask);
        maskDevice = hasMask ? metaBase : nullptr;
        workspace_tensor = at::empty({static_cast<int64_t>(fa_metadata::WorkSpaceSize(blockDim))},
                                     at::device(at::kPrivateUse1).dtype(at::kByte));
        // Same dropout patch as mha_fwd: stream-ordered copies fill the
        // AICPU tiling's missing dropout fields.
        PatchTilingDropoutFields(schedMd, fa_metadata::TilingOffset(hasMask),
                                 1.0f / (1.0f - p_dropout),
                                 return_softmax ? static_cast<uint8_t *>(const_cast<void *>(p.data_ptr())) : nullptr,
                                 has_dropout ? static_cast<uint8_t *>(const_cast<void *>(drop_mask_npu_tensor.data_ptr())) : nullptr);
    } else {
    at::Tensor tiling_cpu_tensor = at::empty({static_cast<int64_t>(sizeof(FAInferTilingData))},
                                             at::device(c10::kCPU).dtype(at::kByte));
    FAInferTilingData* tiling_cpu_ptr = reinterpret_cast<FAInferTilingData*>(tiling_cpu_tensor.data_ptr<uint8_t>());
    std::memset(tiling_cpu_ptr, 0, sizeof(FAInferTilingData));
    tiling_cpu_ptr->set_batch(static_cast<uint32_t>(batch_size));
    tiling_cpu_ptr->set_numHeads(static_cast<uint32_t>(num_heads));
    tiling_cpu_ptr->set_kvHeads(static_cast<uint32_t>(num_heads_k));
    tiling_cpu_ptr->set_embeddingSize(static_cast<uint32_t>(head_size_og));
    tiling_cpu_ptr->set_embeddingSizeV(static_cast<uint32_t>(head_size_og));
    tiling_cpu_ptr->set_numBlocks(static_cast<uint32_t>(num_blocks));
    tiling_cpu_ptr->set_blockSize(static_cast<uint32_t>(page_block_size));
    tiling_cpu_ptr->set_maxNumBlocksPerBatch(static_cast<uint32_t>(max_num_blocks_per_seq));
    if (is_local) {
        tiling_cpu_ptr->set_windowSizeLeft(window_size_left);
        tiling_cpu_ptr->set_windowSizeRight(window_size_right);
        tiling_cpu_ptr->set_maskType(static_cast<uint32_t>(FaiKenel::MaskType::MASK_BAND));
    } else if (is_causal) {
        tiling_cpu_ptr->set_maskType(static_cast<uint32_t>(FaiKenel::MaskType::MASK_CAUSAL));
    }
    if (has_softcap) {
        tiling_cpu_ptr->set_scaleValue(softmax_scale / softcap);
    } else {
        tiling_cpu_ptr->set_scaleValue(softmax_scale);
    }
    tiling_cpu_ptr->set_softcapValue(softcap);
    tiling_cpu_ptr->set_dropoutValue(1.0f / (1.0f - p_dropout));
    tiling_cpu_ptr->set_maxQSeqlen(max_seqlen_q);
    tiling_cpu_ptr->set_maxKvSeqlen(max_seqlen_k);
    tiling_cpu_ptr->set_pDevice(return_softmax ? static_cast<uint8_t*>(const_cast<void*>(p.data_ptr())) : nullptr);
    tiling_cpu_ptr->set_dropMaskDevice(
        has_dropout ? static_cast<uint8_t*>(const_cast<void*>(drop_mask_npu_tensor.data_ptr())) : nullptr);

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

    workspace_tensor = at::empty({workSpaceSize},
        at::device(at::kPrivateUse1).dtype(at::kByte));
    tiling_cpu_ptr->set_mm1OutSize(mm1OutSize);
    tiling_cpu_ptr->set_smOnlineOutSize(smOnlineOutSize);
    tiling_cpu_ptr->set_mm2OutSize(mm2OutSize);
    tiling_cpu_ptr->set_UpdateSize(UpdateSize);
    tiling_cpu_ptr->set_workSpaceSize(workSpaceSize);

    at::Tensor cu_seqlens_q_cpu_tensor = cu_seqlens_q.to(at::Device(at::kCPU));
    at::Tensor cu_seqlens_k_cpu_tensor = cu_seqlens_k.to(at::Device(at::kCPU));
    int32_t* cu_seqlens_q_cpu = static_cast<int32_t *>(cu_seqlens_q_cpu_tensor.data_ptr());
    int32_t* cu_seqlens_k_cpu = static_cast<int32_t *>(cu_seqlens_k_cpu_tensor.data_ptr());

    uint32_t totalTaskNum = 0;
    uint32_t groupSize = num_heads / num_heads_k;
    for (int32_t batchIdx = 0; batchIdx < batch_size; batchIdx++) {
        uint64_t qSeqlen = static_cast<uint64_t>(cu_seqlens_q_cpu[batchIdx + 1] - cu_seqlens_q_cpu[batchIdx]);
        uint64_t kvSeqlen = static_cast<uint64_t>(cu_seqlens_k_cpu[batchIdx + 1] - cu_seqlens_k_cpu[batchIdx]);
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
    tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1)); // Tiling to Device
    tilingDevice = static_cast<uint8_t *>(const_cast<void *>(tiling_gpu_tensor.data_ptr()));

    // attention mask
    if (is_causal || is_local) {
        at::Tensor mask_cpu_tensor = at::empty({2048, 2048}, at::device(c10::kCPU).dtype(at::kByte));
        mask_cpu_tensor = at::triu(at::ones_like(mask_cpu_tensor), 1);
        mask_gpu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
        maskDevice = static_cast<uint8_t *>(const_cast<void *>(mask_gpu_tensor.data_ptr()));
    }
    }

    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtError_t error = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    auto qDevice = static_cast<uint8_t *>(const_cast<void *>(q.data_ptr()));
    auto kDevice = static_cast<uint8_t *>(const_cast<void *>(k.data_ptr()));
    auto vDevice = static_cast<uint8_t *>(const_cast<void *>(v.data_ptr()));

    uint8_t * blockTableDevice = nullptr;
    if (paged_KV) {
        blockTableDevice = static_cast<uint8_t *>(const_cast<void *>(block_table.data_ptr()));
    }

    auto oDevice = static_cast<uint8_t *>(const_cast<void *>(out.data_ptr()));
    auto qSeqDevice = static_cast<uint8_t *>(const_cast<void *>(cu_seqlens_q.data_ptr()));
    auto kvSeqDevice = static_cast<uint8_t *>(const_cast<void *>(
        paged_KV ? seqlens_k.data_ptr() : cu_seqlens_k.data_ptr()));
    auto workspaceDevice = static_cast<uint8_t *>(const_cast<void *>(workspace_tensor.data_ptr()));
    auto softmaxLseDevice = static_cast<uint8_t *>(const_cast<void *>(softmaxlse.data_ptr()));

    // TND forward (IS_TND=true); no flash-decode in the varlen path.
    FwdLaunchArgs fwd_args;
    fwd_args.blockDim = blockDim;
    fwd_args.aclStream = aclStream;
    fwd_args.fftsAddr = fftsAddr;
    fwd_args.is_bf16 = is_bf16;
    fwd_args.paged_KV = paged_KV;
    fwd_args.is_causal = is_causal;
    fwd_args.is_local = is_local;
    fwd_args.flashDecodeFlag = false;
    fwd_args.has_softcap = has_softcap;
    fwd_args.return_softmax = return_softmax;
    fwd_args.has_dropout = has_dropout;
    fwd_args.qDevice = qDevice;
    fwd_args.kDevice = kDevice;
    fwd_args.vDevice = vDevice;
    fwd_args.kNewDevice = nullptr;
    fwd_args.vNewDevice = nullptr;
    fwd_args.maskDevice = maskDevice;
    fwd_args.blockTableDevice = blockTableDevice;
    fwd_args.oDevice = oDevice;
    fwd_args.softmaxLseDevice = softmaxLseDevice;
    fwd_args.qSeqDevice = qSeqDevice;
    fwd_args.kvSeqDevice = kvSeqDevice;
    fwd_args.workspaceDevice = workspaceDevice;
    fwd_args.tilingDevice = tilingDevice;
    auto launch_fa_infer = [fwd_args]() -> int {
        launch_fwd<true>(fwd_args);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_fa_infer", launch_fa_infer);

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

    TORCH_CHECK(softcap >= 0.0f, "softcap must be non-negative (0.0 disables softcap)");

    // parse shape args
    auto qsizes = q.sizes();
    auto ksizes = k.sizes();
    auto vsizes = v.sizes();
    auto dout_sizes = dout.sizes();
    auto out_sizes = out.sizes();
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && dout.dim() == 3 && out.dim() == 3,
                "mha_varlen_bwd: q/k/v/dout/out must be 3-D (TND)");
    TORCH_CHECK(dout_sizes == out_sizes, "mha_varlen_bwd: out and dout must have the same shape");
    TORCH_CHECK(dq.sizes() == qsizes && dk.sizes() == ksizes && dv.sizes() == vsizes,
                "mha_varlen_bwd: dq/dk/dv must match q/k/v shapes");

    uint32_t nheads = qsizes[1];
    uint32_t nheads_k = ksizes[1];
    uint32_t headdim = qsizes[2];
    uint32_t v_headdim = vsizes[2];
    uint32_t dout_headdim = static_cast<uint32_t>(dout_sizes[2]);
    // FAG kernel specializations top out at Aligned256.
    TORCH_CHECK(nheads > 0 && nheads_k > 0, "mha_varlen_bwd: number of Q/KV heads must be positive");
    TORCH_CHECK(nheads % nheads_k == 0,
                "mha_varlen_bwd: number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(headdim == static_cast<uint32_t>(ksizes[2]) && headdim == v_headdim && headdim == dout_headdim,
                "mha_varlen_bwd: q/k/v/dout must share the same headdim (unequal headdim is not supported)");
    TORCH_CHECK(headdim > 0 && headdim <= 256, "mha_varlen_bwd: headdim must be in (0, 256].");
    TORCH_CHECK(qsizes == dout_sizes, "mha_varlen_bwd: q and dout must have the same shape");
    TORCH_CHECK(ksizes == vsizes, "mha_varlen_bwd: k and v must have the same shape");
    TORCH_CHECK(static_cast<uint32_t>(vsizes[1]) == nheads_k,
                "mha_varlen_bwd: v nheads_k must match k");

    int local_window_size_left = window_size_left;
    int local_window_size_right = window_size_right;
    if (local_window_size_left >= max_seqlen_k - 1) {
        local_window_size_left = -1;
    }
    if (local_window_size_right >= max_seqlen_q - 1) {
        local_window_size_right = -1;
    }
    if (is_causal) {
        local_window_size_right = 0;
    }
    const bool local_is_causal = local_window_size_left < 0 && local_window_size_right == 0;
    const bool is_local = (local_window_size_left >= 0 || local_window_size_right >= 0) && !local_is_causal;

    // varlen optimized kernel only supports headdim equal to 128
    if (!seqlens_q.equal(seqlens_k) || is_local || p_dropout > 0.0f || headdim != 128) {
        float scale = softmax_scale > 0.f ? softmax_scale : (1.0f / sqrt(static_cast<float>(headdim)));
        return launch_fag_general(
            dout, q, k, v, out, softmax_lse, dq, dk, dv, seqlens_q, seqlens_k, max_seqlen_q, max_seqlen_k, scale,
            softcap, local_is_causal, local_window_size_left, local_window_size_right, deterministic, p_dropout,
            rng_state);
    }

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
    bool has_softcap = (softcap > 0.0f);
    if (has_softcap) {
        fagInfo.scaleValue = softmax_scale / softcap;
    } else {
        fagInfo.scaleValue = softmax_scale;
    }
    fagInfo.softcapValue = softcap;
    uint64_t workspaceSize = 0;
    FAGTiling::GetFATilingParam(fagInfo, blockDim, reinterpret_cast<int64_t *>(tiling_cpu_tensor.data_ptr<uint8_t>()), workspaceSize);
    at::Tensor tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));

    // alloc workspace
    at::Tensor workspace_tensor = at::empty({static_cast<long>(workspaceSize)},
        at::device(at::kPrivateUse1).dtype(at::kByte));

    // alloc custom attn_mask
    at::Tensor mask_gpu_tensor;
    if (is_causal) {
        mask_gpu_tensor = at::empty({2048, 2048}, at::device(at::kPrivateUse1).dtype(at::kByte));
        mask_gpu_tensor = at::triu(at::ones_like(mask_gpu_tensor), 1);
    }
    at::Tensor seqlenq_gpu_tensor = seqlens_q.to(at::Device(at::kPrivateUse1));
    at::Tensor seqlenk_gpu_tensor = seqlens_k.to(at::Device(at::kPrivateUse1));

    at::Tensor softmax_lse_kernel = softmax_lse;
    TORCH_CHECK(softmax_lse.dim() == 2, "mha_varlen_bwd: softmax_lse for TND must be a 2D tensor.");
    const int64_t total_q = qsizes[0];
    TORCH_CHECK(softmax_lse.size(0) == nheads && softmax_lse.size(1) == total_q,
                "mha_varlen_bwd: softmax_lse must be NT (nheads, total_q) in TND mode.");
    if (!softmax_lse.is_contiguous()) {
        softmax_lse_kernel = softmax_lse.contiguous();
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
    auto cuSeqQlenDevice = static_cast<uint8_t *>(const_cast<void *>(seqlenq_gpu_tensor.storage().data()));
    auto cuSeqKvlenDevice = static_cast<uint8_t *>(const_cast<void *>(seqlenk_gpu_tensor.storage().data()));
    auto softMaxLseDevice = static_cast<uint8_t *>(const_cast<void *>(softmax_lse_kernel.storage().data()));

    auto workspaceDevice = static_cast<uint8_t *>(const_cast<void *>(workspace_tensor.storage().data()));
    auto tilingDevice = static_cast<uint8_t *>(const_cast<void *>(tiling_gpu_tensor.storage().data()));
    auto dqDevice = static_cast<uint8_t *>(const_cast<void *>(dq.storage().data()));
    auto dkDevice = static_cast<uint8_t *>(const_cast<void *>(dk.storage().data()));
    auto dvDevice = static_cast<uint8_t *>(const_cast<void *>(dv.storage().data()));

    // Varlen backward kernel launches live in varlen_bwd_dispatch_{bf16,fp16}.cpp
    // (the ENABLE_ASCENDC_DUMP path and the OpCommand wrapper are handled inside
    // the dispatch).
    VarlenBwdLaunchArgs vb_args;
    vb_args.blockDim = blockDim;
    vb_args.aclStream = aclStream;
    vb_args.fftsAddr = fftsAddr;
    vb_args.is_bf16 = is_bf16;
    vb_args.is_causal = is_causal;
    vb_args.is_softcap = has_softcap;
    vb_args.qDevice = qDevice;
    vb_args.kDevice = kDevice;
    vb_args.vDevice = vDevice;
    vb_args.dOutDevice = dOutDevice;
    vb_args.attenMaskDevice = attenMaskDevice;
    vb_args.softMaxLseDevice = softMaxLseDevice;
    vb_args.outDevice = outDevice;
    vb_args.cuSeqQlenDevice = cuSeqQlenDevice;
    vb_args.cuSeqKvlenDevice = cuSeqKvlenDevice;
    vb_args.dqDevice = dqDevice;
    vb_args.dkDevice = dkDevice;
    vb_args.dvDevice = dvDevice;
    vb_args.workspaceDevice = workspaceDevice;
    vb_args.tilingDevice = tilingDevice;
    if (vb_args.is_bf16) {
        launch_varlen_bwd_impl<bfloat16_t>(vb_args);
    } else {
        launch_varlen_bwd_impl<half>(vb_args);
    }

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
        std::optional<at::Tensor> &rng_state)
{
    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
    } else {
        dv = torch::empty_like(v);
    }

    TORCH_CHECK(softcap >= 0.0f, "softcap must be non-negative (0.0 disables softcap)");

    auto qsizes = q.sizes();
    auto ksizes = k.sizes();
    auto vsizes = v.sizes();
    auto dout_sizes = dout.sizes();
    auto out_sizes = out.sizes();
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && dout.dim() == 4 && out.dim() == 4,
                "mha_bwd: q/k/v/dout/out must be 4-D (BSND)");
    TORCH_CHECK(dout_sizes == out_sizes, "mha_bwd: out and dout must have the same shape");
    TORCH_CHECK(dq.sizes() == qsizes && dk.sizes() == ksizes && dv.sizes() == vsizes,
                "mha_bwd: dq/dk/dv must match q/k/v shapes");

    const uint32_t nheads = qsizes[2];
    const uint32_t nheads_k = ksizes[2];
    const uint32_t headdim = qsizes[3];
    const uint32_t v_headdim = vsizes[3];
    const uint32_t dout_headdim = static_cast<uint32_t>(dout_sizes[3]);
    TORCH_CHECK(nheads > 0 && nheads_k > 0, "mha_bwd: number of Q/KV heads must be positive");
    TORCH_CHECK(nheads % nheads_k == 0,
                "mha_bwd: number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(headdim == static_cast<uint32_t>(ksizes[3]) && headdim == v_headdim && headdim == dout_headdim,
                "mha_bwd: q/k/v/dout must share the same headdim (unequal headdim is not supported)");
    TORCH_CHECK(headdim > 0 && headdim <= 256, "mha_bwd: headdim must be in (0, 256].");
    TORCH_CHECK(qsizes == dout_sizes, "mha_bwd: q and dout must have the same shape");
    TORCH_CHECK(ksizes == vsizes, "mha_bwd: k and v must have the same shape");
    TORCH_CHECK(qsizes[0] == ksizes[0], "mha_bwd: q and k must share the same batch size");
    TORCH_CHECK(static_cast<uint32_t>(vsizes[2]) == nheads_k, "mha_bwd: v nheads_k must match k");
    float scale = softmax_scale > 0.f ? softmax_scale
                                      : (1.0f / sqrt(static_cast<float>(headdim)));
    return launch_fag_general(
        dout, q, k, v, out, softmax_lse, dq, dk, dv, std::nullopt, std::nullopt, qsizes[1], ksizes[1], scale, softcap,
        is_causal, window_size_left, window_size_right, deterministic, p_dropout, rng_state);
}

at::Tensor get_scheduler_metadata(
        int64_t batch_size,
        int64_t max_seqlen_q,
        int64_t max_seqlen_k,
        int64_t num_heads_q,
        int64_t num_heads_kv,
        int64_t headdim,
        int64_t headdim_v,
        pybind11::object qkv_dtype,
        at::Tensor cache_seqlens,
        std::optional<at::Tensor> cu_seqlens_q,
        std::optional<int64_t> page_size,
        bool causal,
        int64_t window_size_left,
        int64_t window_size_right,
        double softcap,
        std::optional<double> softmax_scale)
{
    const c10::OptionalDeviceGuard device_guard(device_of(cache_seqlens));
    const bool is_varlen_q = cu_seqlens_q.has_value();
    void *cuSeqlensQDev = nullptr;
    if (is_varlen_q) {
        auto cu_q = cu_seqlens_q.value();
        TORCH_CHECK(cu_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
        cuSeqlensQDev = cu_q.data_ptr();
    }
    TORCH_CHECK(cache_seqlens.dtype() == torch::kInt32, "cache_seqlens must have dtype int32");
    const uint32_t ps = page_size.has_value() ? static_cast<uint32_t>(page_size.value()) : 128;
    const uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    TORCH_CHECK(softcap >= 0.0, "softcap must be non-negative (0.0 disables softcap)");
    // Mask axes are fully derived on host from the declared seqlen bounds; the
    // AICPU kernel only copies the final values into the tiling blob.
    FwdMaskDerivation maskDer = DeriveFwdMask(causal, window_size_left, window_size_right,
                                              max_seqlen_q, max_seqlen_k);
    float scaleValue = softmax_scale.has_value() ? static_cast<float>(softmax_scale.value())
        : 1.0f / std::sqrt(static_cast<float>(headdim));
    if (softcap > 0.0) {
        scaleValue /= static_cast<float>(softcap);
    }
    FAMetadataArgs args;
    args.cuSeqlensQAddr = is_varlen_q ? reinterpret_cast<uint64_t>(cuSeqlensQDev) : 0ULL;
    args.seqlensKAddr = reinterpret_cast<uint64_t>(cache_seqlens.data_ptr());
    args.metaOutAddr = 0;  // set by GetSchedulerMetadataImpl
    args.batch = static_cast<uint32_t>(batch_size);
    args.numHeads = static_cast<uint32_t>(num_heads_q);
    args.numHeadsK = static_cast<uint32_t>(num_heads_kv);
    args.embeddingSize = static_cast<uint32_t>(headdim);
    args.embeddingSizeV = static_cast<uint32_t>(headdim_v);
    args.numBlocks = 0;
    args.blockSize = ps;
    args.maxNumBlocksPerBatch = page_size.has_value()
        ? ((static_cast<uint32_t>(max_seqlen_k) + ps - 1) / ps) : 0;
    args.maxQSeqlen = static_cast<uint32_t>(max_seqlen_q);
    args.maskType = maskDer.maskType;
    args.windowSizeLeft = maskDer.window_left;
    args.windowSizeRight = maskDer.window_right;
    args.blockDim = blockDim;
    args.isVarlen = is_varlen_q ? 1U : 0U;
    args.isVarlenKv = 0U;  // cache_seqlens always carries per-batch KV lengths
    args.pagedKV = page_size.has_value() ? 1U : 0U;
    args.scaleValue = scaleValue;
    args.softcapValue = static_cast<float>(softcap);
    return GetSchedulerMetadataImpl(args, cache_seqlens, cu_seqlens_q);
}

PYBIND11_MODULE(flash_attn_npu, m)
{
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass, with KV-cache");
    m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
    m.def("varlen_bwd", &mha_varlen_bwd, "Backward pass (variable length)");
    m.def("get_scheduler_metadata", &get_scheduler_metadata, "Precompute scheduler metadata (tiling + mask) on AICPU");
}
