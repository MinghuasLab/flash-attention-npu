// Copyright (c) 2026 Huawei Technologies Co., Ltd.

#pragma once

#include "catlass/catlass.hpp"
#include "fag_tiling_data.h"
#include "kernel_operator.h"

namespace FAGTiling950 {

constexpr uint32_t GRAD_ZERO_TILE_ELEMENTS = 40U * 1024U;

CATLASS_DEVICE void ZeroGradient(GM_ADDR dst, FAGGradClearRange range, AscendC::LocalTensor<uint16_t> zeros,
                                 uint32_t coreId, uint32_t coreCount)
{
    if (range.count == 0 || coreCount == 0) {
        return;
    }
    constexpr uint32_t elementsPerBlock = 32 / sizeof(uint16_t);
    const uint64_t groupElements = static_cast<uint64_t>(coreCount) * elementsPerBlock;
    const uint64_t perCore = (range.count + groupElements - 1) / groupElements * elementsPerBlock;
    const uint64_t begin = coreId * perCore;
    if (begin >= range.count) {
        return;
    }
    const uint64_t count = range.count - begin < perCore ? range.count - begin : perCore;
    AscendC::GlobalTensor<uint16_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(dst));
    for (uint64_t offset = 0; offset < count; offset += GRAD_ZERO_TILE_ELEMENTS) {
        const uint64_t remaining = count - offset;
        const uint32_t current =
            static_cast<uint32_t>(remaining < GRAD_ZERO_TILE_ELEMENTS ? remaining : GRAD_ZERO_TILE_ELEMENTS);
        AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(current * sizeof(uint16_t)), 0, 0, 0};
        AscendC::DataCopyPad(gm[range.begin + begin + offset], zeros, copy);
    }
}

} // namespace FAGTiling950
