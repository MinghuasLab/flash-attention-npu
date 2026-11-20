// Copyright (c) 2026, Minghua Shen.

#include "fwd_dispatch_impl.hpp"

template void launch_paired_full_layout<Format::TND>(const FwdLaunchArgs&);
template void launch_paired_full_layout<Format::BSND>(const FwdLaunchArgs&);
template void launch_paired_full_layout<Format::BNSD>(const FwdLaunchArgs&);
