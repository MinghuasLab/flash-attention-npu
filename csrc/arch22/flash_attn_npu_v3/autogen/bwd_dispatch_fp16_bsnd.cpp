/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// v3 backward FAGGeneral dispatch, BSND variant. One explicit instantiation per
// translation unit so the kernel templates compile in parallel across
// cores; head_dim is a runtime axis (switch/tiling inside the impl), not a
// template parameter, so it is not a generation axis.

#include "../bwd_dispatch_common.hpp"

template void bwd_dispatch_run<half, BSND>(const BwdLaunchArgs &);  // BSND
