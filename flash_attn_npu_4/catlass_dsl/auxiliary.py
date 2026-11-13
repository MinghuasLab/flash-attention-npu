"""Logical auxiliary tensor views over the backend's flat pointer ABI.

Shapes and nontrivial strides come from the current launch. Rank, broadcast
strides, signedness and offset width are compile-time ABI facts.
"""

from functools import lru_cache
from .utils import get_tla


tla = get_tla()


class AuxTensor:
    def __init__(self, tensor, shape, stride, unsigned32, int32_offsets=None):
        self.unsigned32 = unsigned32
        self.tensor = tensor
        self.shape = shape
        self.stride = stride
        self.dtype = getattr(tensor, "dtype", None)
        if self.dtype is None:
            self.dtype = tensor.ptr.dtype
        self.int32_offsets = (
            int32_offsets
            if int32_offsets is not None
            else (
                all(0 <= step < 2**31 for step in stride)
                and sum(max(0, size - 1) * step for size, step in zip(shape, stride)) < 2**31
            )
        )

    @property
    def ptr(self):
        return self.tensor.ptr

    def __getitem__(self, indices):
        indices = indices if isinstance(indices, tuple) else (indices,)
        if len(indices) != len(self.shape):
            raise ValueError("aux coordinate rank must match its logical shape")
        cached_loader = getattr(self.tensor, "load_cached", None)
        if cached_loader is not None:
            value = cached_loader(indices)
            if value is not None:
                return (value.to(tla.Int64) & tla.Int64(0xFFFFFFFF)) if self.unsigned32 else value
        index_dtype = tla.Int32 if self.int32_offsets else tla.Int64
        offset = index_dtype(0)
        for index, stride in zip(indices, self.stride):
            offset = offset + tla.as_numeric(index).to(index_dtype) * index_dtype(stride)
        value = self.tensor[offset]
        return (value.to(tla.Int64) & tla.Int64(0xFFFFFFFF)) if self.unsigned32 else value


def _metadata_size(specs):
    return sum(2 * len(spec[0]) for spec in specs if len(spec) == 4)


def _views(tensors, specs, scalars=()):
    offset = len(scalars) - _metadata_size(specs)
    views = []
    for tensor, spec in zip(tensors, specs):
        shape, stride, unsigned32 = spec[:3]
        if len(spec) == 4:
            rank = len(shape)
            shape = scalars[offset : offset + rank]
            runtime_stride = scalars[offset + rank : offset + 2 * rank]
            stride = tuple(
                step if step is not None else actual for step, actual in zip(stride, runtime_stride)
            )
            offset += 2 * rank
            views.append(AuxTensor(tensor, shape, stride, unsigned32, spec[3]))
        else:
            views.append(AuxTensor(tensor, shape, stride, unsigned32))
    return tuple(views)


@lru_cache(maxsize=512)
def bind_aux(fn, kind, specs):
    if not specs:
        return fn
    metadata_size = _metadata_size(specs)
    if kind == "mask":

        @tla.jit
        def bound(b, h, q, k, info, aux, scalars):
            return fn(
                b,
                h,
                q,
                k,
                info,
                _views(aux, specs, scalars),
                scalars[:-metadata_size] if metadata_size else scalars,
            )
    elif kind == "score":

        @tla.jit
        def bound(s, b, h, q, k, info, aux, scalars):
            return fn(
                s,
                b,
                h,
                q,
                k,
                info,
                _views(aux, specs, scalars),
                scalars[:-metadata_size] if metadata_size else scalars,
            )
    else:

        @tla.jit
        def bound(g, s, b, h, q, k, info, aux, scalars):
            return fn(
                g,
                s,
                b,
                h,
                q,
                k,
                info,
                _views(aux, specs, scalars),
                scalars[:-metadata_size] if metadata_size else scalars,
            )

    return bound
