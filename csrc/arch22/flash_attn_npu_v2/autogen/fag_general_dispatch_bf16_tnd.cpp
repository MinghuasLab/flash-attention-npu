/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// v2 FAGGeneral backward dispatch, TND variant. One explicit instantiation per
// translation unit so the kernel templates compile in parallel across
// cores; head_dim is a runtime axis (switch inside the impl), not a
// template parameter, so it is not a generation axis.

#include "../fag_general_dispatch_impl.hpp"

template void launch_fag_general_dispatch_bf16<TND>(const FagGeneralLaunchArgs &);  // TND
