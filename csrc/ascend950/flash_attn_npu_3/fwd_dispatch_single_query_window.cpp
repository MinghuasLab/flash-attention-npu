#include "fwd_dispatch_impl.hpp"

// Instantiate these kernels in a separate device image to preserve the original SWA launch path.
void launch_single_query_window(const FwdLaunchArgs& a)
{
    if (a.layout == Format::TND) {
        launch_single_query_window_layout<Format::TND>(a);
    } else if (a.layout == Format::BNSD) {
        launch_single_query_window_layout<Format::BNSD>(a);
    } else {
        launch_single_query_window_layout<Format::BSND>(a);
    }
}
