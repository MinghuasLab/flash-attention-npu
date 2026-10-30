#ifndef FLASH_ATTN_FAG_TENSOR_STRIDES_H
#define FLASH_ATTN_FAG_TENSOR_STRIDES_H
#include <cstdint>

// Element strides in logical BSND (dense) or TND (packed) coordinates.
// Packed tensors have batch=0; their sequence coordinate includes the cu prefix.
struct FagTensorStrides {
    int64_t batch;
    int64_t sequence;
    int64_t head;
};
struct FagInputStrides {
    FagTensorStrides q, k, v, dout, out;
    uint32_t enabled;
};
#endif
