// Copyright (c) 2026, Minghua Shen.
// v2 forward FAInfer dispatch, BNSD variant. One explicit instantiation per
// translation unit so the kernel templates compile in parallel across
// cores; head_dim is a runtime axis (switch inside the impl), not a
// template parameter, so it is not a generation axis.

#include "../fwd_dispatch_impl.hpp"

template void launch_fwd_impl<half, KernelCommon::FaiKenel::inputLayout::BNSD>(const FwdLaunchArgs &);  // BNSD
