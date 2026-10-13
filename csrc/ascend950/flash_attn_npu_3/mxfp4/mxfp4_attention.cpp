/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#include "mxfp4_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "mxfp4_kernel_dispatch.hpp"
#include "mxfp4_tiling.hpp"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/CachingHostAllocator.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace {

constexpr int64_t kGroupSize = 64;
constexpr int64_t kDefaultBlockShape = 128;

int64_t ceil_div(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

void check_npu_byte_tensor(const at::Tensor& tensor, const char* name, const at::Device& device)
{
    TORCH_CHECK(tensor.device() == device, name, " must be on the same NPU device as query");
    TORCH_CHECK(tensor.scalar_type() == at::kByte, name, " must be a packed uint8 tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_shape(const at::Tensor& tensor, const std::vector<int64_t>& expected, const char* name)
{
    TORCH_CHECK(tensor.sizes().vec() == expected, name, " has shape ", tensor.sizes(), ", expected ", expected);
}

size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

struct ParsedShape {
    Mxfp4Layout layout;
    int64_t batch;
    int64_t q_heads;
    int64_t kv_heads;
    int64_t q_tokens;
    int64_t kv_tokens;
    int64_t max_q;
    int64_t max_kv;
    int64_t logical_d;
    std::vector<int64_t> output_shape;
};

ParsedShape parse_shape(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const std::string& layout,
                        int64_t num_kv_heads, const std::optional<std::vector<int64_t>>& q_lengths,
                        const std::optional<std::vector<int64_t>>& kv_lengths)
{
    TORCH_CHECK(q.dim() == (layout == "TND" ? 3 : 4), "query rank does not match layout ", layout);
    TORCH_CHECK(k.dim() == q.dim() && v.dim() == q.dim(), "query/key/value ranks must match");
    TORCH_CHECK(q.size(-1) == k.size(-1) && q.size(-1) == v.size(-1),
                "packed query/key/value head dimensions must match");
    TORCH_CHECK(q.size(-1) > 0, "packed head dimension must be positive");

    ParsedShape p{};
    p.logical_d = q.size(-1) * 2;
    TORCH_CHECK(p.logical_d == 64 || p.logical_d == 128,
                "MXFP4 Ascend 950 currently supports logical head dimension 64 or 128, got ", p.logical_d);
    p.output_shape = q.sizes().vec();
    p.output_shape.back() = p.logical_d;

    if (layout == "TND") {
        p.layout = Mxfp4Layout::TND;
        p.q_tokens = q.size(0);
        p.kv_tokens = k.size(0);
        p.q_heads = q.size(1);
        p.kv_heads = k.size(1);
        TORCH_CHECK(v.size(0) == p.kv_tokens && v.size(1) == p.kv_heads, "TND key/value shapes must match");
        TORCH_CHECK(q_lengths.has_value() && kv_lengths.has_value(),
                    "TND requires actual_seq_lengths and actual_seq_lengths_kv");
        TORCH_CHECK(q_lengths->size() == kv_lengths->size() && !q_lengths->empty(),
                    "TND sequence-length lists must have the same non-zero batch size");
        p.batch = static_cast<int64_t>(q_lengths->size());
        p.max_q = 0;
        p.max_kv = 0;
        int64_t sum_q = 0;
        int64_t sum_kv = 0;
        for (int64_t i = 0; i < p.batch; ++i) {
            const int64_t sq = (*q_lengths)[i];
            const int64_t sk = (*kv_lengths)[i];
            TORCH_CHECK(sq > 0 && sk > 0, "TND actual sequence lengths must be positive");
            sum_q += sq;
            sum_kv += sk;
            p.max_q = std::max(p.max_q, sq);
            p.max_kv = std::max(p.max_kv, sk);
        }
        TORCH_CHECK(sum_q == p.q_tokens, "sum(actual_seq_lengths) must equal query T");
        TORCH_CHECK(sum_kv == p.kv_tokens, "sum(actual_seq_lengths_kv) must equal key/value T");
    } else if (layout == "BNSD") {
        p.layout = Mxfp4Layout::BNSD;
        p.batch = q.size(0);
        p.q_heads = q.size(1);
        p.max_q = q.size(2);
        p.kv_heads = k.size(1);
        p.max_kv = k.size(2);
        TORCH_CHECK(k.size(0) == p.batch && v.sizes() == k.sizes(), "BNSD key/value shapes must match query batch");
        p.q_tokens = p.batch * p.max_q;
        p.kv_tokens = p.batch * p.max_kv;
    } else {
        TORCH_CHECK(layout == "BSND", "input_layout must be TND, BNSD, or BSND");
        p.layout = Mxfp4Layout::BSND;
        p.batch = q.size(0);
        p.max_q = q.size(1);
        p.q_heads = q.size(2);
        p.max_kv = k.size(1);
        p.kv_heads = k.size(2);
        TORCH_CHECK(k.size(0) == p.batch && v.sizes() == k.sizes(), "BSND key/value shapes must match query batch");
        p.q_tokens = p.batch * p.max_q;
        p.kv_tokens = p.batch * p.max_kv;
    }

    TORCH_CHECK(p.kv_heads == num_kv_heads, "num_key_value_heads does not match key/value shape");
    TORCH_CHECK(p.q_heads > 0 && p.kv_heads > 0 && p.q_heads % p.kv_heads == 0,
                "query heads must be divisible by KV heads");
    return p;
}

void validate_non_tnd_lengths(const ParsedShape& p, const std::optional<std::vector<int64_t>>& q_lengths,
                              const std::optional<std::vector<int64_t>>& kv_lengths)
{
    TORCH_CHECK(q_lengths.has_value() == kv_lengths.has_value(),
                "actual_seq_lengths and actual_seq_lengths_kv must be both present or both absent");
    if (!q_lengths.has_value()) {
        return;
    }
    TORCH_CHECK(static_cast<int64_t>(q_lengths->size()) == p.batch &&
                    static_cast<int64_t>(kv_lengths->size()) == p.batch,
                "sequence-length list size must equal batch");
    for (int64_t i = 0; i < p.batch; ++i) {
        TORCH_CHECK((*q_lengths)[i] > 0 && (*q_lengths)[i] <= p.max_q,
                    "actual query sequence length is outside storage shape");
        TORCH_CHECK((*kv_lengths)[i] > 0 && (*kv_lengths)[i] <= p.max_kv,
                    "actual KV sequence length is outside storage shape");
    }
}

void validate_scales(const ParsedShape& p, const at::Tensor& q_scale, const at::Tensor& k_scale,
                     const at::Tensor& v_scale, const std::optional<std::vector<int64_t>>& kv_lengths)
{
    const int64_t dg = p.logical_d / kGroupSize;
    if (p.layout == Mxfp4Layout::TND) {
        int64_t bt_kv = 0;
        for (int64_t length : *kv_lengths) {
            bt_kv += ceil_div(length, kGroupSize);
        }
        check_shape(q_scale, {p.q_tokens, p.q_heads, dg, 2}, "q_dequant_scale");
        check_shape(k_scale, {p.kv_tokens, p.kv_heads, dg, 2}, "k_dequant_scale");
        check_shape(v_scale, {bt_kv, p.kv_heads, p.logical_d, 2}, "v_dequant_scale");
    } else if (p.layout == Mxfp4Layout::BNSD) {
        check_shape(q_scale, {p.batch, p.q_heads, p.max_q, dg, 2}, "q_dequant_scale");
        check_shape(k_scale, {p.batch, p.kv_heads, p.max_kv, dg, 2}, "k_dequant_scale");
        check_shape(v_scale, {p.batch, p.kv_heads, ceil_div(p.max_kv, 64), p.logical_d, 2}, "v_dequant_scale");
    } else {
        check_shape(q_scale, {p.batch, p.max_q, p.q_heads, dg, 2}, "q_dequant_scale");
        check_shape(k_scale, {p.batch, p.max_kv, p.kv_heads, dg, 2}, "k_dequant_scale");
        check_shape(v_scale, {p.batch, ceil_div(p.max_kv, 64), p.kv_heads, p.logical_d, 2}, "v_dequant_scale");
    }
}

} // namespace

at::Tensor mxfp4_quant_attention(at::Tensor query, at::Tensor key, at::Tensor value, at::Tensor q_dequant_scale,
                                 at::Tensor k_dequant_scale, at::Tensor v_dequant_scale, std::string input_layout,
                                 int64_t num_key_value_heads, double softmax_scale, int64_t quant_mode,
                                 std::optional<std::vector<int64_t>> actual_seq_lengths,
                                 std::optional<std::vector<int64_t>> actual_seq_lengths_kv,
                                 std::optional<at::Tensor> block_sparse_mask,
                                 std::optional<std::vector<int64_t>> block_shape,
                                 std::optional<at::ScalarType> output_dtype, double dst_type_max)
{
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    TORCH_CHECK(query.device().type() == at::kPrivateUse1, "query must be on NPU");
    check_npu_byte_tensor(query, "query", query.device());
    check_npu_byte_tensor(key, "key", query.device());
    check_npu_byte_tensor(value, "value", query.device());
    check_npu_byte_tensor(q_dequant_scale, "q_dequant_scale", query.device());
    check_npu_byte_tensor(k_dequant_scale, "k_dequant_scale", query.device());
    check_npu_byte_tensor(v_dequant_scale, "v_dequant_scale", query.device());
    TORCH_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0.0, "softmax_scale must be finite and positive");
    TORCH_CHECK(quant_mode == static_cast<int64_t>(Mxfp4QuantMode::OCP) ||
                    quant_mode == static_cast<int64_t>(Mxfp4QuantMode::CX),
                "quant_mode must be 0 (MXFP4_OCP) or 1 (MXFP4_CX)");
    TORCH_CHECK(output_dtype.has_value(), "output_dtype must be torch.float16 or torch.bfloat16");
    TORCH_CHECK(*output_dtype == at::kHalf || *output_dtype == at::kBFloat16,
                "output_dtype must be torch.float16 or torch.bfloat16");
    if (quant_mode == static_cast<int64_t>(Mxfp4QuantMode::CX)) {
        TORCH_CHECK(dst_type_max == 0.0 || (dst_type_max >= 6.0 && dst_type_max <= 12.0),
                    "CX scheme requires target_max == 0 or target_max in [6, 12], got ", dst_type_max);
    }

    ParsedShape p =
        parse_shape(query, key, value, input_layout, num_key_value_heads, actual_seq_lengths, actual_seq_lengths_kv);
    if (p.layout != Mxfp4Layout::TND) {
        validate_non_tnd_lengths(p, actual_seq_lengths, actual_seq_lengths_kv);
    }
    validate_scales(p, q_dequant_scale, k_dequant_scale, v_dequant_scale, actual_seq_lengths_kv);

    const bool sparse = block_sparse_mask.has_value();
    TORCH_CHECK(sparse || !block_shape.has_value(), "block_shape is only valid with block_sparse_mask");
    int64_t block_x = kDefaultBlockShape;
    int64_t block_y = kDefaultBlockShape;
    if (block_shape.has_value()) {
        TORCH_CHECK(block_shape->size() == 2, "block_shape must contain [q_block, kv_block]");
        block_x = (*block_shape)[0];
        block_y = (*block_shape)[1];
    }
    TORCH_CHECK(block_x > 0 && block_y > 0 && block_x % 64 == 0 && block_y % 64 == 0,
                "MXFP4 block dimensions must be positive multiples of 64");

    const int64_t max_q_blocks = ceil_div(p.max_q, block_x);
    const int64_t max_kv_blocks = ceil_div(p.max_kv, block_y);
    if (sparse) {
        check_npu_byte_tensor(*block_sparse_mask, "block_sparse_mask", query.device());
        check_shape(*block_sparse_mask, {p.batch, p.q_heads, max_q_blocks, max_kv_blocks}, "block_sparse_mask");
    }

    const uint32_t block_dim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    TORCH_CHECK(block_dim > 0, "failed to query Ascend AIC core count");

    optiling::Mxfp4Context ctx;
    ctx.batch = p.batch;
    ctx.qHeads = p.q_heads;
    ctx.kvHeads = p.kv_heads;
    ctx.maxQ = p.max_q;
    ctx.maxKv = p.max_kv;
    ctx.logicalD = p.logical_d;
    ctx.blockX = block_x;
    ctx.blockY = block_y;
    ctx.sparse = sparse;
    ctx.hasQLengths = actual_seq_lengths.has_value();
    ctx.hasKvLengths = actual_seq_lengths_kv.has_value();
    if (ctx.hasQLengths) {
        ctx.qLengths = *actual_seq_lengths;
    }
    ctx.scaleValue = static_cast<float>(softmax_scale);
    ctx.cx = quant_mode == static_cast<int64_t>(Mxfp4QuantMode::CX);
    ctx.dstTypeMax = dst_type_max;

    static_assert(std::is_trivially_copyable<Mxfp4TilingData>::value, "Mxfp4TilingData must remain trivially copyable");
    static_assert(std::is_trivially_destructible<Mxfp4TilingData>::value,
                  "Mxfp4TilingData destructor must stay trivial");

    const std::vector<int64_t> q_lengths = actual_seq_lengths.value_or(std::vector<int64_t>(p.batch, p.max_q));
    const std::vector<int64_t> kv_lengths = actual_seq_lengths_kv.value_or(std::vector<int64_t>(p.batch, p.max_kv));
    const size_t tiling_bytes = sizeof(Mxfp4TilingData);
    const size_t q_bytes = q_lengths.size() * sizeof(int64_t);
    const size_t kv_bytes = kv_lengths.size() * sizeof(int64_t);
    const size_t q_offset = align_up(tiling_bytes, alignof(int64_t));
    const size_t kv_offset = q_offset + align_up(q_bytes, alignof(int64_t));
    const size_t total = kv_offset + kv_bytes;
    at::Tensor host = at::empty({static_cast<int64_t>(total)},
                                at::TensorOptions().dtype(at::kByte).device(at::kCPU).pinned_memory(true));
    auto* host_base = static_cast<uint8_t*>(host.data_ptr());
    TORCH_CHECK(reinterpret_cast<uintptr_t>(host_base) % alignof(Mxfp4TilingData) == 0,
                "pinned MXFP4 launch buffer is not aligned for Mxfp4TilingData");
    auto* tiling = reinterpret_cast<Mxfp4TilingData*>(host_base);
    *tiling = Mxfp4TilingData{};
    optiling::Mxfp4Tiling tiler(ctx);
    tiler.SetCoreNum(block_dim);
    tiler.DoTiling(*tiling);
    TORCH_CHECK(tiling->totalTaskNum > 0, "attention has no work");
    if (ctx.cx) {
        TORCH_CHECK(std::isfinite(tiling->log2Cx), "dst_type_max must be positive and finite");
    }
    if (q_bytes != 0) {
        std::memcpy(host_base + q_offset, q_lengths.data(), q_bytes);
    }
    if (kv_bytes != 0) {
        std::memcpy(host_base + kv_offset, kv_lengths.data(), kv_bytes);
    }

    at::Tensor device_args = at::empty({static_cast<int64_t>(total)}, query.options().dtype(at::kByte));
    auto* device_base = static_cast<uint8_t*>(device_args.data_ptr());
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    aclrtStream rt_stream = stream.stream(false);
    TORCH_CHECK(aclrtMemcpyAsync(device_base, total, host_base, total, ACL_MEMCPY_HOST_TO_DEVICE, rt_stream) ==
                    ACL_SUCCESS,
                "async copy of MXFP4 launch args failed");

    at::Tensor output = at::empty(p.output_shape, query.options().dtype(*output_dtype));
    at::Tensor workspace;
    if (sparse) {
        workspace = at::empty({static_cast<int64_t>(tiling->workSpaceSize)}, query.options().dtype(at::kByte));
    }

    auto ptr = [](const at::Tensor& tensor) { return static_cast<uint8_t*>(tensor.data_ptr()); };
    Mxfp4LaunchArgs args{block_dim,
                         rt_stream,
                         ptr(query),
                         ptr(key),
                         ptr(value),
                         sparse ? ptr(*block_sparse_mask) : nullptr,
                         device_base + q_offset,
                         device_base + kv_offset,
                         ptr(q_dequant_scale),
                         ptr(k_dequant_scale),
                         ptr(v_dequant_scale),
                         ptr(output),
                         sparse ? ptr(workspace) : nullptr,
                         nullptr,
                         device_base};

    const auto layout = p.layout;
    const auto mode = static_cast<Mxfp4QuantMode>(quant_mode);
    const bool output_bf16 = *output_dtype == at::kBFloat16;
    auto launch = [args, layout, mode, output_bf16, sparse]() -> int {
        launch_mxfp4(args, layout, mode, output_bf16, sparse);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("mxfp4_quant_attention", launch);
    const c10::DataPtr& host_data = host.storage().data_ptr();
    TORCH_CHECK(at_npu::native::CachingHostAllocator_recordEvent(host_data.get(), host_data.get_context(), stream),
                "failed to record MXFP4 host copy completion");
    c10_npu::NPUCachingAllocator::recordStream(device_args.storage().data_ptr(), stream);
    if (sparse) {
        c10_npu::NPUCachingAllocator::recordStream(workspace.storage().data_ptr(), stream);
    }
    return output;
}
