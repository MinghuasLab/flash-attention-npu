"""Analytic block intervals for registered causal and local-window masks."""

from catlass import tla


@tla.jit
def window_block_bounds(qstart, valid_q, qlen, klen, window: tla.Constexpr):
    """Return first KV block, count and the all-query-visible element interval.

    window is (left, right, bottom_right); -1 denotes an unbounded side.
    Clamp before adding/subtracting window bounds to avoid int32 overflow.
    """
    start = tla.as_numeric(0)
    count = tla.as_numeric(0)
    full_lo = tla.as_numeric(0)
    full_hi = tla.as_numeric(0)
    if valid_q > 0 and klen > 0:
        first = qstart
        if tla.const_expr(window[2]):
            first = qstart + (klen - qlen)
        last = first + valid_q - 1
        lo = tla.as_numeric(0)
        hi = klen
        full_hi = klen
        if tla.const_expr(window[0] >= 0):
            if first > window[0]:
                lo = first - window[0]
            if last > window[0]:
                full_lo = last - window[0]
        if tla.const_expr(window[1] >= 0):
            if window[1] < klen - 1 - last:
                hi = max(0, last + window[1] + 1)
            if window[1] < klen - 1 - first:
                full_hi = max(0, first + window[1] + 1)
        if lo < hi:
            start = lo // 128
            count = (hi + 127) // 128 - start
    return start, count, full_lo, full_hi
