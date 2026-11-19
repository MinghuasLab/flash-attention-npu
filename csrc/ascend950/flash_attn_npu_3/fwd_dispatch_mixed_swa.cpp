// Copyright (c) 2026, Minghua Shen.

#include "fwd_dispatch_impl.hpp"

void launch_mixed_swa(const FwdLaunchArgs& a)
{
    if (a.layout == Format::TND) {
        launch_mixed_swa_layout<Format::TND>(a);
    } else if (a.layout == Format::BNSD) {
        launch_mixed_swa_layout<Format::BNSD>(a);
    } else {
        launch_mixed_swa_layout<Format::BSND>(a);
    }
}
