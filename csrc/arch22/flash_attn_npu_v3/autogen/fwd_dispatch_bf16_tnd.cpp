/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// v3 forward FAInfer dispatch, TND variant. One explicit instantiation per
// translation unit so the kernel templates compile in parallel across
// cores; head_dim is a runtime axis (switch/tiling inside the impl), not a
// template parameter, so it is not a generation axis.

#include "../fwd_dispatch_impl.hpp"

template void launch_fwd_dtype<bfloat16_t, true>(const FwdLaunchArgs &);  // TND
