"""Logical auxiliary tensor views over the backend's flat pointer ABI.

This is compile-time layout binding, not callback tracing or code generation.
Views contain immutable shape/stride metadata and emit native scalar loads.
"""

from functools import lru_cache
from .utils import get_tla


tla = get_tla()


class AuxTensor:
    def __init__(self, tensor, shape, stride, unsigned32):
        self.unsigned32 = unsigned32
        self.tensor = tensor
        self.shape = shape
        self.stride = stride
        self.dtype = tensor.dtype
        self.int32_offsets = (
            all(0 <= step < 2**31 for step in stride)
            and sum(max(0, size - 1) * step for size, step in zip(shape, stride)) < 2**31
        )

    @property
    def ptr(self):
        return self.tensor.ptr

    def __getitem__(self, indices):
        indices = indices if isinstance(indices, tuple) else (indices,)
        if len(indices) != len(self.shape):
            raise ValueError("aux coordinate rank must match its logical shape")
        index_dtype = tla.Int32 if self.int32_offsets else tla.Int64
        offset = index_dtype(0)
        for index, stride in zip(indices, self.stride):
            offset = offset + tla.as_numeric(index).to(index_dtype) * index_dtype(stride)
        value = self.tensor[offset]
        return (value.to(tla.Int64) & tla.Int64(0xFFFFFFFF)) if self.unsigned32 else value


def _views(tensors, specs):
    return tuple(
        AuxTensor(tensor, shape, stride, unsigned32)
        for tensor, (shape, stride, unsigned32) in zip(tensors, specs)
    )


@lru_cache(maxsize=512)
def bind_aux(fn, kind, specs):
    if not specs:
        return fn
    if kind == "mask":

        @tla.jit
        def bound(b, h, q, k, info, aux, scalars):
            return fn(b, h, q, k, info, _views(aux, specs), scalars)
    elif kind == "score":

        @tla.jit
        def bound(s, b, h, q, k, info, aux, scalars):
            return fn(s, b, h, q, k, info, _views(aux, specs), scalars)
    else:

        @tla.jit
        def bound(g, s, b, h, q, k, info, aux, scalars):
            return fn(g, s, b, h, q, k, info, _views(aux, specs), scalars)

    return bound
