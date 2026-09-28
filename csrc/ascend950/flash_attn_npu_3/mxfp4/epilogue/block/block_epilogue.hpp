/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* Modified by Minghua Shen, 2026. */

#ifndef MXFP4_BLOCK_EPILOGUE_HPP
#define MXFP4_BLOCK_EPILOGUE_HPP

#include "catlass/numeric_size.hpp"

namespace Catlass::Epilogue::Block {

template <class DispatchPolicy, class... Args>
class BlockEpilogue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

} // namespace Catlass::Epilogue::Block
#if (__CCE_AICORE__ == 310)
#include "../../epilogue/block/block_epilogue_mask2idx.hpp"
#include "../../epilogue/block/block_epilogue_rescale_o_reg_high_prec.hpp"
#include "../../epilogue/block/block_epilogue_online_softmax_reg_low_prec_fp16.hpp"
#include "../../epilogue/block/block_epilogue_compute_pscale.hpp"
#include "../../epilogue/block/block_epilogue_copy_global_max_ub_to_l1.hpp"
#endif
#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP
