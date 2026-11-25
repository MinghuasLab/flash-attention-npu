/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

// Source: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/common/op_kernel/{axis,array}.h.
// Private slicing/buffer views for the migrated 192/128 Cube block.
#pragma once
#include "mla_copy_tile.hpp"
namespace flash_attn::mla::Tile::Prefill {
using namespace AscendC;
struct AxisSlices;

struct Axis {
    uint32_t start;
    uint32_t sizeAct;

    __aicore__ inline Axis(uint32_t sizeAct_) : start(0), sizeAct(sizeAct_) {}
    __aicore__ inline Axis(uint32_t start_, uint32_t sizeAct_) : start(start_), sizeAct(sizeAct_) {}

    __aicore__ inline AxisSlices Split(uint32_t splitSize) const;

    template <uint32_t ALIGN_SIZE = BLOCK_CUBE>
    __aicore__ inline uint32_t AlignedSize() const
    {
        return (((ALIGN_SIZE) == 0) ? 0 : (((sizeAct) + (ALIGN_SIZE)-1) / (ALIGN_SIZE) * (ALIGN_SIZE)));
    }

    __aicore__ inline bool IsTailOf(const Axis& parent) const
    {
        return (start + sizeAct) >= parent.sizeAct;
    }
};

struct AxisSlices {
    struct Sentinel {
        uint32_t end_;
    };
    struct Iterator {
        Axis cur_;
        uint32_t tSizeAct_;
        uint32_t splitSize_;

        __aicore__ inline Iterator(const AxisSlices& slices)
            : cur_(Min(slices.splitSize_, slices.sizeAct_)), tSizeAct_(slices.sizeAct_), splitSize_(slices.splitSize_)
        {}

        __aicore__ inline Axis& operator*()
        {
            return cur_;
        }

        __aicore__ inline Iterator& operator++()
        {
            cur_.start += splitSize_;
            cur_.sizeAct = Min(splitSize_, tSizeAct_ - cur_.start);
            return *this;
        }

        __aicore__ inline bool operator!=(const Sentinel& end) const
        {
            return cur_.start < end.end_;
        }
    };

    uint32_t sizeAct_;
    uint32_t splitSize_;

    __aicore__ inline AxisSlices(uint32_t sizeAct, uint32_t splitSize) : sizeAct_(sizeAct), splitSize_(splitSize) {}

    __aicore__ inline Iterator begin()
    {
        return {*this};
    }

    __aicore__ inline Sentinel end()
    {
        return Sentinel{sizeAct_};
    }

    __aicore__ inline uint32_t size() const
    {
        return (sizeAct_ + (splitSize_ - 1)) / splitSize_;
    }
};

__aicore__ inline AxisSlices Axis::Split(uint32_t splitSize) const
{
    return {sizeAct, splitSize};
}

template <typename T, uint32_t N, uint64_t Stride = 0>
class Array {
  private:
    T elems[N];

  public:
    __aicore__ inline T& operator[](uint32_t i)
    {
        return elems[i];
    }
};

template <typename T, uint32_t N, uint64_t Stride>
class Array<LocalTensor<T>, N, Stride> {
  private:
    LocalTensor<T> tensor;

  public:
    __aicore__ inline void Init(const LocalTensor<T>& tensor)
    {
        this->tensor = tensor;
    }
    __aicore__ inline LocalTensor<T> operator[](uint32_t i)
    {
        return tensor[i * Stride];
    }
};

template <typename T, uint32_t N>
class Array<GlobalTensor<T>, N> {
  private:
    GlobalTensor<T> tensor;
    uint64_t stride = 0;

  public:
    __aicore__ inline void Init(const GlobalTensor<T>& tensor, uint64_t splitSize)
    {
        this->tensor = tensor;
        this->stride = splitSize;
    }
    __aicore__ inline GlobalTensor<T> operator[](uint32_t i)
    {
        return tensor[i * stride];
    }
};

} // namespace flash_attn::mla::Tile::Prefill
