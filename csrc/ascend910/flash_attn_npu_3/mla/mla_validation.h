/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once

#include <torch/extension.h>
#include "mla_metadata.h"
#include "mla_layout.h"
// Host-only validation for compressed and expanded MLA inputs.
// Contract: TriDao FA3 d15f1531a460ba456f41b01a774f33ab2db8febf,
// hopper/flash_api.cpp::mha_fwd. No CANN kernel or scheduling code is used here.
// The native forward and its Meta inference share these shape/storage checks.
namespace flash_attn::mla {

inline void CheckMetadataTensor(const at::Tensor& tensor, const at::Device& device, int64_t count, const char* name)
{
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must have dtype int32");
    TORCH_CHECK((device.type() == at::kPrivateUse1 || device.is_meta()) && tensor.device() == device, name,
                " must be on the same NPU device");
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == count, name, " must exactly match shape [", count, "]");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline MlaBsndTensorLayout CheckBsndTensorLayout(const at::Tensor& tensor)
{
    MlaBsndTensorView view{};
    for (int64_t axis = 0; axis < 4; ++axis) {
        view.sizes[axis] = tensor.size(axis);
        view.strides[axis] = tensor.stride(axis);
    }
    view.storageOffset = tensor.storage_offset();
    view.storageBytes = tensor.storage().nbytes();
    MlaBsndTensorLayout layout{};
    const auto status = BuildBsndTensorLayout(view, layout);
    TORCH_CHECK(status == MlaLayoutStatus::Success,
                "MLA: BSND input view exceeds stride, storage or signed byte bounds (status ", int(status), ")");
    if (layout.spanElements != 0 && !tensor.is_meta()) {
        const auto address = reinterpret_cast<uintptr_t>(tensor.data_ptr());
        TORCH_CHECK(address != 0 && layout.spanElements * 2 <= UINTPTR_MAX - address, "MLA: tensor address overflow");
    }
    return layout;
}

inline MlaBsndInputLayouts CheckBsndInputLayouts(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                                 const at::Tensor& qv)
{
    return {CheckBsndTensorLayout(q), CheckBsndTensorLayout(qv), CheckBsndTensorLayout(k), CheckBsndTensorLayout(v)};
}

inline MlaBsndTensorLayout CheckTndTensorLayout(const at::Tensor& tensor)
{
    MlaTndTensorView view{};
    for (int64_t axis = 0; axis < 3; ++axis) {
        view.sizes[axis] = tensor.size(axis);
        view.strides[axis] = tensor.stride(axis);
    }
    view.storageOffset = tensor.storage_offset();
    view.storageBytes = tensor.storage().nbytes();
    MlaTndTensorLayout layout{};
    const auto status = BuildTndTensorLayout(view, layout);
    TORCH_CHECK(status == MlaLayoutStatus::Success,
                "MLA: TND input view exceeds stride, storage or signed byte bounds (status ", int(status), ")");
    if (layout.spanElements != 0 && !tensor.is_meta()) {
        const auto address = reinterpret_cast<uintptr_t>(tensor.data_ptr());
        TORCH_CHECK(address != 0 && layout.spanElements * 2 <= UINTPTR_MAX - address, "MLA: tensor address overflow");
    }
    return {0, layout.tokenStride, layout.headStride, layout.spanElements};
}

inline MlaBsndInputLayouts CheckTndInputLayouts(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                                const at::Tensor& qv)
{
    return {CheckTndTensorLayout(q), CheckTndTensorLayout(qv), CheckTndTensorLayout(k), CheckTndTensorLayout(v)};
}

inline MlaBsndInputLayouts CheckTndBsndInputLayouts(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                                    const at::Tensor& qv)
{
    return {CheckTndTensorLayout(q), CheckTndTensorLayout(qv), CheckBsndTensorLayout(k), CheckBsndTensorLayout(v)};
}

inline bool FitsDenseShape(int64_t batch, int64_t sq, int64_t heads, int64_t sk)
{
    return batch >= 0 && batch <= UINT32_MAX && sq >= 0 && sq <= kMlaMaxQueryLength && heads > 0 &&
           heads <= kMlaMaxQueryHeads && uint64_t(sq) * heads <= kMlaMaxDenseRows &&
           uint64_t(batch) * sq * heads <= kMlaMaxBatchRows && sk >= 0 && sk <= kMlaMaxKvLength;
}

inline bool CheckForwardBoundary(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                 const std::optional<at::Tensor>& qv, bool varlenQ, bool varlenK, bool pagedKV)
{
    // Keep ordinary layout and malformed-query validation in the upstream path.
    if (!qv.has_value() && ((varlenQ && q.dim() != 3) || (q.dim() > 0 && v.dim() > 0 && q.size(-1) == v.size(-1)))) {
        return false;
    }
    TORCH_CHECK(q.dim() == (varlenQ ? 3 : 4), "FlashAttention: q rank must match BSND/TND layout");
    TORCH_CHECK(k.dim() == (varlenK ? 3 : 4) && v.dim() == k.dim(),
                "FlashAttention: k/v rank must match dense, TND or paged KV layout");
    TORCH_CHECK(q.size(-1) == k.size(-1), "FlashAttention: q and k head dimensions must match");

    TORCH_CHECK(!pagedKV || !varlenK, "MLA: page_table and cu_seqlens_k cannot be combined");
    const bool int8 = qv && (qv->scalar_type() == at::kChar || v.scalar_type() == at::kChar);
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
                "MLA: inputs must have FP16 or BF16 dtype");
    TORCH_CHECK(!int8 || q.scalar_type() == at::kBFloat16, "MLA INT8: RoPE inputs must have BF16 dtype");
    TORCH_CHECK((q.device().type() == at::kPrivateUse1 || q.is_meta()), "MLA: inputs must be NPU tensors");
    auto checkTensor = [&q](const at::Tensor& t, at::ScalarType dtype) {
        TORCH_CHECK(t.scalar_type() == dtype, "MLA: unexpected input dtype");
        TORCH_CHECK(t.device() == q.device(), "MLA: all inputs must be on the same device");
        TORCH_CHECK(t.dim() > 0 && t.stride(-1) == 1, "MLA: inputs must have contiguous last dimension");
    };
    checkTensor(q, q.scalar_type());
    checkTensor(k, q.scalar_type());
    checkTensor(v, int8 ? at::kChar : q.scalar_type());
    for (int64_t axis = 0; axis < k.dim() - 1; ++axis) {
        TORCH_CHECK(k.size(axis) == v.size(axis), "MLA: k/v shapes must match except for head dimension");
    }
    const auto headsQ = q.size(-2);
    const auto headsKV = k.size(-2);
    TORCH_CHECK(headsQ > 0 && headsKV > 0 && headsQ % headsKV == 0,
                "MLA: positive KV head count must divide query head count");
    if (qv.has_value()) {
        checkTensor(*qv, int8 ? at::kChar : q.scalar_type());
        TORCH_CHECK(qv->dim() == q.dim(), "MLA: qv shape must match q with the value head dimension");
        for (int64_t axis = 0; axis < q.dim() - 1; ++axis) {
            TORCH_CHECK(qv->size(axis) == q.size(axis), "MLA: qv shape must match q with the value head dimension");
        }
        TORCH_CHECK(qv->size(-1) == v.size(-1), "MLA: qv shape must match q with the value head dimension");
        TORCH_CHECK(q.size(-1) == 64 && v.size(-1) == 512,
                    "MLA: supported qv representation requires Dq=64 and Dv=512");
        TORCH_CHECK(headsKV == 1, "MLA: compressed representation requires one KV head");
        TORCH_CHECK(varlenQ || !varlenK, "MLA compressed forward is not implemented yet for BSND Q with TND KV");
        TORCH_CHECK(pagedKV || varlenQ || q.size(0) == k.size(0), "MLA: dense batch sizes must match");
        TORCH_CHECK(varlenQ || FitsDenseShape(q.size(0), q.size(1), headsQ, pagedKV ? 0 : k.size(1)),
                    "MLA compressed forward is not implemented yet outside B>=0, nonnegative S1G within address/field "
                    "bounds, 0<=Sk<=16384");
        for (const auto& t : {q, k, v, *qv}) {
            TORCH_CHECK(!t.requires_grad(), "MLA compressed forward is not implemented yet for gradients");
        }
        return true;
    }
    TORCH_CHECK(q.size(-1) == 192 && v.size(-1) == 128,
                "MLA: supported representation without qv requires Dq=192 and Dv=128");
    TORCH_CHECK(varlenQ || !varlenK, "MLA prefill forward is not implemented yet for BSND Q with TND KV");
    TORCH_CHECK(pagedKV || varlenQ || q.size(0) == k.size(0), "MLA: dense batch sizes must match");
    TORCH_CHECK(varlenQ || FitsDenseShape(q.size(0), q.size(1), headsQ, pagedKV ? 0 : k.size(1)),
                "MLA: prefill BSND shape bounds exceeded");
    for (const auto& t : {q, k, v}) {
        TORCH_CHECK(!t.requires_grad(), "MLA prefill forward is not implemented yet for gradients");
        if (t.dim() == 3)
            CheckTndTensorLayout(t);
        else
            CheckBsndTensorLayout(t);
    }
    return true;
}

inline void CheckTndInputs(const at::Tensor& q, const at::Tensor& k, const at::Tensor& cuQ, const at::Tensor& cuK,
                           int64_t maxQ, int64_t maxK)
{
    TORCH_CHECK(cuQ.dim() == 1 && cuQ.numel() >= 1 && cuK.dim() == 1 && cuK.numel() == cuQ.numel(),
                "MLA: cu_seqlens_q/k must have matching shape [batch_size + 1]");
    for (const auto& t : {cuQ, cuK}) {
        TORCH_CHECK(t.scalar_type() == at::kInt, "MLA: cu_seqlens_q/k must have dtype int32");
        TORCH_CHECK(t.device() == q.device(), "MLA: cu_seqlens_q/k must be on the same NPU device");
        TORCH_CHECK(t.is_contiguous(), "MLA: cu_seqlens_q/k must be contiguous");
    }
    const int64_t batch = cuQ.numel() - 1;
    TORCH_CHECK(q.size(0) <= INT32_MAX && k.size(0) <= INT32_MAX && q.size(1) <= kMlaMaxQueryHeads && maxQ >= 0 &&
                    maxQ <= kMlaMaxQueryLength && maxK >= 0 && maxK <= kMlaMaxKvLength &&
                    uint64_t(maxQ) * q.size(1) <= kMlaMaxDenseRows &&
                    uint64_t(q.size(0)) * q.size(1) <= kMlaMaxBatchRows,
                "MLA: compact TND shape or max_seqlen bounds exceeded");
}

inline void CheckCacheLengths(const at::Tensor& lengths, int64_t batch, const at::Device& device)
{
    CheckMetadataTensor(lengths, device, batch, "MLA: cache_seqlens");
}

inline void CheckUsedInputs(const at::Tensor& q, int64_t batch, const std::optional<at::Tensor>& usedQuery,
                            const std::optional<at::Tensor>& usedKv)
{
    if (usedQuery) {
        CheckMetadataTensor(*usedQuery, q.device(), batch, "MLA: seqused_q");
    }
    if (usedKv) {
        CheckMetadataTensor(*usedKv, q.device(), batch, "MLA: seqused_k");
    }
}

inline void CheckMetadataQueryPrefix(const at::Tensor& cuQ, int64_t batch, const at::Device& device)
{
    CheckMetadataTensor(cuQ, device, batch + 1, "MLA: cu_seqlens_q");
}

inline void CheckTndBsndForwardInputs(const at::Tensor& q, const at::Tensor& k, const at::Tensor& cuQ, int64_t maxQ,
                                      const std::optional<at::Tensor>& cacheSeqlens, const std::optional<at::Tensor>&)
{
    const int64_t batch = k.size(0);
    TORCH_CHECK(FitsDenseShape(0, maxQ, q.size(1), k.size(1)) && q.size(0) <= INT32_MAX &&
                    uint64_t(q.size(0)) * q.size(1) <= kMlaMaxBatchRows,
                "MLA: mixed shape or max_seqlen bounds exceeded");
    CheckMetadataQueryPrefix(cuQ, batch, q.device());
    if (cacheSeqlens) {
        CheckCacheLengths(*cacheSeqlens, batch, q.device());
    }
}

} // namespace flash_attn::mla
