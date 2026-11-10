// Copyright (c) 2026, Minghua Shen.
// v3 (950) forward FAInfer dispatch, fp16 x BSND variant.
// One explicit instantiation per translation unit so the FAInfer / FAInferDn
// kernel templates compile in parallel across cores; head_dim is a runtime
// tiling axis (not a template parameter), so it is not a generation axis.

#include "../fwd_dispatch_impl.hpp"

template void launch_fwd_impl<half, Format::BSND>(
    const FwdLaunchArgs &a);
