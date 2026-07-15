/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// v2 forward FAInfer dispatch, BSND variant. One explicit instantiation per
// translation unit so the kernel templates compile in parallel across
// cores; head_dim is a runtime axis (switch inside the impl), not a
// template parameter, so it is not a generation axis.

#include "../fwd_dispatch_impl.hpp"

template void launch_fwd_impl<bfloat16_t, false>(const FwdLaunchArgs &);  // BSND
