/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#include "mla_layout.h"
#include <algorithm>
namespace flash_attn::mla {

namespace {
template <uint64_t elementBytes>
MlaLayoutStatus BuildLayout(const MlaBsndTensorView& view, MlaBsndTensorLayout& output)
{
    if (view.sizes[0] < 0 || view.sizes[1] < 0 || view.sizes[2] <= 0 ||
        (view.sizes[3] != 64 && view.sizes[3] != 128 && view.sizes[3] != 192 && view.sizes[3] != 512)) {
        return MlaLayoutStatus::InvalidShape;
    }
    if (view.strides[0] < 0 || view.strides[1] < 0 || view.strides[2] < 0 || view.strides[3] != 1) {
        return MlaLayoutStatus::InvalidStride;
    }
    if (view.storageOffset < 0) {
        return MlaLayoutStatus::InvalidStorage;
    }

    MlaBsndTensorLayout layout{uint64_t(view.strides[0]), uint64_t(view.strides[1]), uint64_t(view.strides[2]), 0};
    if (view.sizes[0] != 0 && view.sizes[1] != 0) {
        // Check each product/addition before
        // computing it; neither a signed intermediate nor the final byte span
        // may overflow. Storage offset is checked here, not sent to the device.
        constexpr uint64_t maxElements = INT64_MAX / elementBytes;
        uint64_t span = view.sizes[3];
        for (uint32_t axis = 0; axis < 3; ++axis) {
            const uint64_t count = view.sizes[axis] - 1;
            const uint64_t stride = view.strides[axis];
            if (count != 0 && stride > (maxElements - span) / count) {
                return MlaLayoutStatus::AddressOverflow;
            }
            span += count * stride;
        }
        const uint64_t offset = view.storageOffset;
        if (offset > maxElements - span) {
            return MlaLayoutStatus::AddressOverflow;
        }
        if ((offset + span) * elementBytes > view.storageBytes) {
            return MlaLayoutStatus::InvalidStorage;
        }
        layout.spanElements = span;
    }
    output = layout;
    return MlaLayoutStatus::Success;
}
} // namespace

MlaLayoutStatus BuildBsndTensorLayout(const MlaBsndTensorView& view, MlaBsndTensorLayout& output)
{
    return BuildLayout<2>(view, output);
}

MlaLayoutStatus BuildInt8BsndTensorLayout(const MlaBsndTensorView& view, MlaBsndTensorLayout& output)
{
    return BuildLayout<1>(view, output);
}

MlaLayoutStatus BuildTndTensorLayout(const MlaTndTensorView& view, MlaTndTensorLayout& output)
{
    // TND is a single BSND batch for address validation. Do not infer physical
    // strides from token counts, per-sample maxima, or the last dimension.
    const MlaBsndTensorView bsnd{{1, view.sizes[0], view.sizes[1], view.sizes[2]},
                                 {0, view.strides[0], view.strides[1], view.strides[2]},
                                 view.storageOffset,
                                 view.storageBytes};
    MlaBsndTensorLayout layout{};
    const auto status = BuildBsndTensorLayout(bsnd, layout);
    if (status == MlaLayoutStatus::Success) {
        output = {layout.rowStride, layout.headStride, layout.spanElements};
    }
    return status;
}

} // namespace flash_attn::mla

// Address/segment mapping derived from CANN ops-transformer
// ca18aee610729ca5270a44e2d84c751bc687fbf0, attention/common/op_kernel/memcopy/:
// offset_calculator_v2.h (GM_KV_PA_BNBD), parser.h (BlockTableParser),
// attn_copy_gm_to_l1.h (CopyKKropePAGmToL1 / CopyKvGmToL1).
// Changes: independent CPU validation, signed page checks, checked 64-bit
// geometry and full storage bounds, arbitrary nonnegative outer strides.
// FA3 k maps to source RoPE K; FA3 v maps to both source content K and V.

namespace flash_attn::mla {
namespace {

MlaLayoutStatus BuildPageTableLayout(const MlaPageTableView& table, MlaPagedCacheLayout& layout)
{
    if (table.sizes[0] < 0 || table.sizes[1] < 0) {
        return MlaLayoutStatus::InvalidShape;
    }
    if (table.strides[0] < 0 || table.strides[1] != 1) {
        return MlaLayoutStatus::InvalidStride;
    }
    if (table.storageOffset < 0) {
        return MlaLayoutStatus::InvalidStorage;
    }
    layout.batchSize = table.sizes[0];
    layout.maxPages = table.sizes[1];
    layout.tableRowStride = table.strides[0];
    if (layout.batchSize != 0 && layout.maxPages != 0) {
        constexpr uint64_t maxElements = INT64_MAX / sizeof(int32_t);
        const uint64_t rows = layout.batchSize - 1;
        if (layout.maxPages > maxElements ||
            (rows != 0 && layout.tableRowStride > (maxElements - layout.maxPages) / rows)) {
            return MlaLayoutStatus::AddressOverflow;
        }
        const uint64_t span = rows * layout.tableRowStride + layout.maxPages;
        const uint64_t offset = table.storageOffset;
        if (offset > maxElements - span) {
            return MlaLayoutStatus::AddressOverflow;
        }
        if ((offset + span) * sizeof(int32_t) > table.storageBytes) {
            return MlaLayoutStatus::InvalidStorage;
        }
        layout.tableSpanElements = span;
    }
    return MlaLayoutStatus::Success;
}

} // namespace

namespace {
template <bool Int8>
MlaLayoutStatus BuildCacheLayout(const MlaPageTableView& table, const MlaBsndTensorView& k, const MlaBsndTensorView& v,
                                 MlaPagedCacheLayout& output, bool prefill)
{
    if (Int8 && (prefill || k.sizes[1] != 128)) {
        return MlaLayoutStatus::InvalidShape;
    }
    if (k.sizes[0] < 0 || uint64_t(k.sizes[0]) > uint64_t(INT32_MAX) + 1 || k.sizes[1] < 16 || k.sizes[1] > 1024 ||
        k.sizes[1] % 16 != 0 || (prefill ? k.sizes[2] <= 0 : k.sizes[2] != 1) || k.sizes[3] != (prefill ? 192 : 64) ||
        v.sizes[0] != k.sizes[0] || v.sizes[1] != k.sizes[1] || v.sizes[2] != k.sizes[2] ||
        v.sizes[3] != (prefill ? 128 : 512)) {
        return MlaLayoutStatus::InvalidShape;
    }
    MlaPagedCacheLayout layout{};
    auto status = BuildPageTableLayout(table, layout);
    if (status != MlaLayoutStatus::Success) {
        return status;
    }
    layout.physicalPages = k.sizes[0];
    layout.pageSize = k.sizes[1];
    if (layout.maxPages > uint64_t(INT64_MAX) / layout.pageSize) {
        return MlaLayoutStatus::AddressOverflow;
    }
    layout.logicalCapacity = layout.maxPages * layout.pageSize;
    status = BuildBsndTensorLayout(k, layout.k);
    if (status != MlaLayoutStatus::Success) {
        return status;
    }
    status = Int8 ? BuildInt8BsndTensorLayout(v, layout.v) : BuildBsndTensorLayout(v, layout.v);
    if (status != MlaLayoutStatus::Success) {
        return status;
    }
    output = layout;
    return MlaLayoutStatus::Success;
}
} // namespace

MlaLayoutStatus BuildPagedCacheLayout(const MlaPageTableView& table, const MlaBsndTensorView& k,
                                      const MlaBsndTensorView& v, MlaPagedCacheLayout& output, bool prefill)
{
    return BuildCacheLayout<false>(table, k, v, output, prefill);
}

MlaLayoutStatus BuildInt8PagedCacheLayout(const MlaPageTableView& table, const MlaBsndTensorView& k,
                                          const MlaBsndTensorView& v, MlaPagedCacheLayout& output)
{
    return BuildCacheLayout<true>(table, k, v, output, false);
}

} // namespace flash_attn::mla
