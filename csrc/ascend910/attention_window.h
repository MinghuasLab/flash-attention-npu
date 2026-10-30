#pragma once

#include <cstdint>

namespace fa_window {

// Bottom-right alignment: i + K - Q - left <= j <= i + K - Q + right.
// Over all legal i,j, left >= K-1 and right >= Q-1 cover their entire side.
// Keep negative windows unlimited: replacing an unlimited right side with K
// loses visible keys when Q > K. Bounds may be conservative for ragged inputs.
template <typename Window>
inline void NormalizeAttentionWindow(bool causal, int64_t max_q, int64_t max_k,
                                     Window& left, Window& right, bool& is_causal, bool& is_local)
{
    if (left < 0 || (max_k > 0 && left >= max_k - 1)) {
        left = -1;
    }
    if (right < 0 || (max_q > 0 && right >= max_q - 1)) {
        right = -1;
    }
    if (causal) {
        right = 0;
    }
    is_causal = left < 0 && right == 0;
    is_local = (left >= 0 || right >= 0) && !is_causal;
}

} // namespace fa_window
