/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Modified by Minghua Shen, 2026.
 */

#ifndef FAG_MLA_BLOCK_HPP
#define FAG_MLA_BLOCK_HPP

#include "../fag_block.h"

namespace Catlass::Epilogue {
template <uint32_t INPUT_LAYOUT_, bool IS_DROP_, bool IS_ATTEN_MASK_, bool HAS_SOFTCAP_, bool HAS_ALIBI_,
          bool KEEP_SFMG_ = false>
struct EpilogueAtlasA2MlaSameAbVec
    : EpilogueAtlasA2SameAbVec<INPUT_LAYOUT_, IS_DROP_, IS_ATTEN_MASK_, HAS_SOFTCAP_, HAS_ALIBI_> {
    static constexpr bool KEEP_SFMG = KEEP_SFMG_;
};
} // namespace Catlass::Epilogue

#endif // FAG_MLA_BLOCK_HPP
