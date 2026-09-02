/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef FAG_LAYOUT_HPP
#define FAG_LAYOUT_HPP

// Self-contained input-layout constants shared by the v2/v3/v4-910 FAGGeneral
// dispatch (uint32_t template parameter kInputLayout) and the autogen stubs.
// Extracted from kernel_common_fag.hpp, which is NOT self-contained (its
// FAGTilingData needs SoftMaxTiling / GM_ADDR prerequisites) and therefore
// cannot be included by the light dispatch/stub TUs.

#include <cstdint>

constexpr static uint32_t BSND = 0;
constexpr static uint32_t TND = 1;

#endif  // FAG_LAYOUT_HPP
