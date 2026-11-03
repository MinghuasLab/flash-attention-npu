"""Small source-backed helpers for native SIMT callbacks.

Sequence information is (seqlen_q, seqlen_k, offset_q, offset_k). Integer
division helpers implement Python floor semantics for nonzero divisors;
signed MIN/-1 and invalid shifts remain outside the callback contract.
"""

from .utils import get_tla


tla = get_tla()


@tla.jit
def load(tensor, indices):
    """Load a SIMT scalar from any rank using logical element strides.

    Coordinates must be in bounds. Unlike Python indexing, negative indices
    are not normalized. Address arithmetic uses signed 64-bit integers.
    """
    return tensor[indices]


@tla.jit
def seqlen_q(info):
    return info[0]


@tla.jit
def seqlen_k(info):
    return info[1]


@tla.jit
def offset_q(info):
    return info[2]


@tla.jit
def offset_k(info):
    return info[3]


@tla.jit
def floor_div(a, b):
    quotient = a // b
    residual = a % b
    # Keep correction constants in the operands' width (Int32 or Int64).
    zero = a - a
    one = b // b
    if residual != zero and ((a < zero) != (b < zero)):
        quotient = quotient - one
    return quotient


@tla.jit
def remainder(a, b):
    return a - floor_div(a, b) * b


@tla.jit
def tanh(x):
    e = tla.exp(-2.0 * tla.abs(x))
    magnitude = (1.0 - e) / (1.0 + e)
    return tla.where(x < 0.0, -magnitude, magnitude)


@tla.jit
def exp2(x):
    return tla.exp(x * 0.6931471805599453)
