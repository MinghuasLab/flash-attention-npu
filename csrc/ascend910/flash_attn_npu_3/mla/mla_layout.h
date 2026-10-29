/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once
#include <cstdint>
// FA3 input-view adapter for MLA (FP16/BF16, D=64/128/192/512).
// Pointer/stride convention: TriDao FA3 d15f1531a460ba456f41b01a774f33ab2db8febf,
// hopper/flash_api.cpp::set_params_fprop and mha_fwd's q_v handling.
namespace flash_attn::mla {

struct MlaBsndTensorView {
    int64_t sizes[4];
    int64_t strides[4];    // Elements, in logical BSND order.
    int64_t storageOffset; // Elements from the beginning of storage.
    uint64_t storageBytes;
};

// Trivially copyable launch data, relative to the tensor's data_ptr().
// data_ptr() already includes storageOffset; never add that offset again.
// Q, qv, K and V each need their own descriptor, even when their shapes match.
// No address or stride belongs in reusable scheduler metadata.
struct MlaBsndTensorLayout {
    uint64_t batchStride;
    uint64_t rowStride;
    uint64_t headStride;
    uint64_t spanElements; // One past the last reachable element, or zero if empty.
};

struct MlaBsndInputLayouts {
    // TND reuses the row/head representation with batchStride=0 and
    // rowStride=tokenStride after BuildTndTensorLayout validation. The Kernel
    // separately selects prefix addressing from its validated sequence layout.
    MlaBsndTensorLayout q, qv, k, v;
};

enum class MlaLayoutStatus {
    Success,
    InvalidShape,
    InvalidStride,
    InvalidStorage,
    AddressOverflow,
};

// CPU-only address validation, independent of torch/CANN and schedule limits.
// Accept nonnegative outer strides (including read-only overlap/broadcast),
// a unit last stride, B/S >= 0, H > 0, and D=64/128/192/512. The entire allocated view
// must fit storage and signed 64-bit byte offsets, regardless of used lengths.
// Empty views access no storage; like PyTorch, their offset may exceed storage.
// Singleton/empty axes do not impose a byte limit on an unused stride.
// Callers must separately check dtype, device, shape compatibility and schedule
// limits before device launch.
// Failure leaves output unchanged.
MlaLayoutStatus BuildBsndTensorLayout(const MlaBsndTensorView& view, MlaBsndTensorLayout& output);

// INT8 content uses the same element-stride contract with one-byte storage.
// Keep the existing two-byte entry and ABI for all accepted nonquant paths.
MlaLayoutStatus BuildInt8BsndTensorLayout(const MlaBsndTensorView& view, MlaBsndTensorLayout& output);

} // namespace flash_attn::mla

// FA3 input-view adapter for compressed and prefill MLA (FP16/BF16).
// Pointer/stride convention: TriDao FA3 d15f1531a460ba456f41b01a774f33ab2db8febf,
// hopper/flash_api.cpp::set_params_fprop and mha_fwd's q_v handling.
namespace flash_attn::mla {

struct MlaTndTensorView {
    int64_t sizes[3];
    int64_t strides[3];    // Elements, in logical TND order (including NTD views).
    int64_t storageOffset; // Elements from the beginning of storage.
    uint64_t storageBytes;
};

// Relative to data_ptr(), which already includes storageOffset. Each of Q,
// qv, K and V has an independent descriptor; mixed BSND K/V use the BSND one.
// A sequence starts at allocationPrefix[b] * tokenStride, never a sum of used
// lengths or b * maxLength. No address or stride belongs in scheduler metadata.
struct MlaTndTensorLayout {
    uint64_t tokenStride;
    uint64_t headStride;
    uint64_t spanElements; // One past the last reachable element, or zero if empty.
};

// CPU-only adapter to the shared BSND storage/checked-byte validation, with a
// singleton batch. Requires T>=0, H>0, D=64/128/192/512, nonnegative token/head strides
// and a unit last stride. Validates the entire allocation, even with zero used.
// Empty T accesses no storage; unused singleton/empty strides remain 64-bit.
// Callers separately validate dtype/device, shape compatibility, allocation
// prefixes, used lengths and schedule limits. Failure leaves output unchanged.
MlaLayoutStatus BuildTndTensorLayout(const MlaTndTensorView& view, MlaTndTensorLayout& output);

} // namespace flash_attn::mla

// CPU address validation for MLA ND paged caches.
namespace flash_attn::mla {

struct MlaPageTableView {
    int64_t sizes[2];
    int64_t strides[2]; // int32 elements, not bytes; last stride must be one.
    int64_t storageOffset;
    uint64_t storageBytes;
};

struct MlaPagedCacheLayout {
    uint64_t batchSize;
    uint64_t maxPages;        // Logical page-table width, independent of row stride.
    uint64_t logicalCapacity; // maxPages * pageSize, independent of physicalPages.
    uint64_t physicalPages;
    uint64_t pageSize;
    uint64_t tableRowStride;
    uint64_t tableSpanElements;
    MlaBsndTensorLayout k, v; // batchStride denotes physical page stride.
};

// K:[Np,P,1,64], V:[Np,P,1,512], P=16..1024 in multiples of 16.
// All physical pages, including unused ones, must fit their own storage.
// Np <= INT32_MAX+1 makes every physical page representable by an int32 ID.
// Read-only broadcasts/overlaps are legal. Empty views follow BSND storage
// rules. All addresses are relative to data_ptr(), which includes the offset.
// This address layer does not impose schedule limits (e.g. capacity<=16384).
// Callers separately validate Q/qv with BuildBsnd/BuildTndTensorLayout, dtype,
// device and sequence geometry. Failure leaves output unchanged.
MlaLayoutStatus BuildPagedCacheLayout(const MlaPageTableView& table, const MlaBsndTensorView& k,
                                      const MlaBsndTensorView& v, MlaPagedCacheLayout& output, bool prefill = false);

// BF16 RoPE K and INT8 shared content KV, page=128 and Hkv=1 only.
MlaLayoutStatus BuildInt8PagedCacheLayout(const MlaPageTableView& table, const MlaBsndTensorView& k,
                                          const MlaBsndTensorView& v, MlaPagedCacheLayout& output);

// One page-contained segment resolved by the device Copy tile.
struct MlaPagedCopySegment {
    uint64_t tableOffset;
    uint64_t physicalPage;
    uint64_t pageRow;
    uint64_t rowCount;
    uint64_t kOffset; // Row start, before adding the independently selected D.
    uint64_t vOffset;
};
} // namespace flash_attn::mla
