// Copyright (c) 2026, Minghua Shen.
// v2 varlen backward FAGVarlenOpt dispatch variant (always TND). One explicit instantiation per
// translation unit so the kernel templates compile in parallel across
// cores; head_dim is a runtime axis (switch inside the impl), not a
// template parameter, so it is not a generation axis.

#include "../varlen_bwd_dispatch_impl.hpp"

template void launch_varlen_bwd_impl<bfloat16_t>(const VarlenBwdLaunchArgs &);
