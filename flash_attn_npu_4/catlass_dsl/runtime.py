"""Prepared TLA launches with fresh Tensor bindings on a warm cache.

Compiled functions use dynamic tensor descriptors. Geometry and descriptors
are rebuilt for each invocation; only code-generation features key the cache.
Explicit record_stream protects allocations after asynchronous launches.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache, wraps
from typing import NamedTuple

import torch

from .cache_utils import get_jit_cache

compile_cache = get_jit_cache("attention")


def _on_tensor_device(fn):
    @wraps(fn)
    def guarded(first, *args, **kwargs):
        with torch.npu.device(first.device):
            return fn(first, *args, **kwargs)

    return guarded


def _tla():
    from .utils import get_tla

    return get_tla()


def _extent(t):
    return max(1, 1 + sum((n - 1) * step for n, step in zip(t.shape, t.stride())))


class _TensorBinding:
    """CATLASS launch-provider protocol for flat and row-major GM descriptors.

    The compiler-produced ABI chooses pointer versus memref fields. It needs
    only these providers at launch, not another frontend/DLPack tensor. Keep
    the source alive through submission; record_stream covers device use.
    """

    __slots__ = ("source", "address", "rows", "columns", "row_stride")

    def __init__(self, source, shape):
        self.source = source
        self.address = source.data_ptr()  # Includes the Torch storage offset.
        self.rows = shape[0]
        self.columns = shape[1] if len(shape) == 2 else 1
        self.row_stride = self.columns

    def __c_pointers__(self):
        return (self.address,)

    def build_memref_launch_fields(self):
        # CATLASS schema-v4 canonical fields: allocated/aligned ptr, offset,
        # four physical sizes, four strides, and two logical origin sizes.
        return (
            self.address,
            self.address,
            0,
            self.rows,
            self.columns,
            1,
            1,
            self.row_stride,
            1,
            1,
            1,
            self.rows,
            self.columns,
        )


@lru_cache(maxsize=1)
def _prepared_tensor_binding():
    import catlass

    return catlass.__version__.split(".")[:2] == ["2", "2"]


def _tensor(t, shape=None, *, logical=False, fp8=False):
    """Bind current geometry; dtype/FP8 interpretation belongs to the compiled ABI."""
    t = _buffer(t)
    pointer = logical or shape == "pointer"
    shape = (1,) if pointer else shape or (t.numel(),)
    if _prepared_tensor_binding():
        return _TensorBinding(t, shape)
    # Older CATLASS releases expose a different descriptor-provider protocol.
    # Let that backend construct its native binding; compilation still uses
    # the same dynamic layouts and the process-local specialization cache.
    tla = _tla()
    raw = t.detach()
    if fp8:
        raw = raw.view(torch.uint8)
    elif t.dtype == torch.uint32:
        raw = raw.view(torch.int32)
    tensor = tla.from_dlpack(raw, origin_shape=shape, layout_tag=tla.arch.RowMajor)
    return tensor if pointer else tensor.mark_layout_dynamic()


@lru_cache(maxsize=64)
def _sentinel(device, dtype=torch.int32):
    # Never read or written: shared storage for absent/empty ABI arguments.
    return torch.empty((1,), dtype=dtype, device=device)


def _buffer(t):
    """Provide a positive-size ABI sentinel for a logically empty tensor."""
    return _sentinel(t.device, t.dtype) if t.numel() == 0 else t


def _sample(value, shape=None, *, logical=False):
    if isinstance(value, torch.Tensor):
        from catlass.tla.runtime import make_fake_tensor

        tla = _tla()
        dtype = {
            torch.bool: tla.Bool,
            torch.int32: tla.Int32,
            torch.int64: tla.Int64,
            torch.uint32: tla.Int32,
            torch.uint8: tla.UInt8,
            torch.float16: tla.Float16,
            torch.bfloat16: tla.BFloat16,
            torch.float32: tla.Float32,
            torch.float8_e4m3fn: tla.Float8E4M3FN,
        }[value.dtype]
        # SIMT captures one pointer, not a dynamic memref descriptor. Aux
        # handles are linear pointer ABIs; logical bounds and strides are
        # explicit scalars. Their nominal extent never determines addressing.
        pointer = logical or shape == "pointer"
        shape = (1,) if pointer else shape or (max(1, value.numel()),)
        stride = (shape[1], 1) if len(shape) == 2 else (1,)
        sample = make_fake_tensor(
            dtype,
            shape,
            stride,
            origin_shape=shape,
            coord=(0,) * len(shape),
            layout_tag=tla.arch.RowMajor,
        )
        return sample if pointer else sample.mark_layout_dynamic()
    if isinstance(value, tuple):
        items = tuple(_sample(item, logical=logical) for item in value)
        return type(value)(*items) if hasattr(type(value), "_fields") else items
    return value


def _aux_specs(tensors):
    # Shapes and nonzero strides are runtime values. The descriptor retains
    # rank, unsigned interpretation and the offset arithmetic width.
    return tuple(
        (
            (None,) * t.ndim,
            tuple(0 if s == 0 else 1 if s == 1 else None for s in t.stride()),
            t.dtype == torch.uint32,
            _extent(t) < 2**31 and all(0 <= n < 2**31 for n in (*t.shape, *t.stride())),
        )
        for t in tensors
    )


def _aux_key(tensors):
    return tuple((str(t.dtype), spec) for t, spec in zip(tensors, _aux_specs(tensors)))


def _aux(tensors, scalars):
    tla = _tla()
    ts = tuple(_buffer(t) for t in tensors)
    ss = tuple(
        tla.Bool(s) if type(s) is bool else tla.Int32(s) if type(s) is int else tla.Float32(s)
        for s in scalars
    )
    # Internal metadata follows user scalars; bound callbacks strip it before
    # calling user code. No public callback signature changes.
    for tensor, spec in zip(tensors, _aux_specs(tensors)):
        integer = tla.Int32 if spec[3] else tla.Int64
        ss += tuple(integer(x) for x in (*tensor.shape, *tensor.stride()))
    return ts, ss


def _index_shapes(sparse, first=10, second=12):
    if sparse is None:
        return {first: (1, 1), second: (1, 1)}
    return {
        first: (max(1, sparse.mask_block_cnt.numel()), max(1, sparse.mask_block_idx.shape[-1])),
        second: (max(1, sparse.full_block_cnt.numel()), max(1, sparse.full_block_idx.shape[-1])),
    }


def _compile_kernel(kernel, args, tensor_shapes=None):
    tla = _tla()
    shapes = tensor_shapes or {}
    samples = tuple(
        _sample(arg, shapes.get(i), logical=i in (6, 15) and isinstance(arg, tuple))
        for i, arg in enumerate(args)
    )
    return tla.compile(kernel, *samples, options="--npu-arch 3510")


def _compile(key, kernel, args, tensor_shapes=None):
    return compile_cache.get_or_compile(key, lambda: _compile_kernel(kernel, args, tensor_shapes))


def _launch(executor, runtime, *, block_num, device, tensor_shapes=None, fp8=False):
    # Current-stream ordering remains the caller's dependency contract. In
    # addition, record every allocation used by this external launch so a
    # tensor created on a different stream cannot be recycled too early.
    # Local runtime references remain live until launch submission completes;
    # record_stream protects their storage after this function returns.
    stream = torch.npu.current_stream(device)
    seen = set()

    def record(value):
        if isinstance(value, torch.Tensor):
            if id(value) not in seen:
                seen.add(id(value))
                value.record_stream(stream)
        elif isinstance(value, tuple):
            for item in value:
                record(item)

    record(runtime)
    shapes = tensor_shapes or {}

    bindings = {}

    def bind(value, shape=None, *, logical=False, quantized=False):
        if isinstance(value, torch.Tensor):
            key = (id(value), shape, logical, quantized)
            if key not in bindings:
                bindings[key] = _tensor(value, shape, logical=logical, fp8=quantized)
            return bindings[key]
        if isinstance(value, tuple):
            return tuple(bind(item, logical=logical) for item in value)
        return value

    bound = tuple(
        bind(
            arg,
            shapes.get(i),
            logical=i in (6, 15) and isinstance(arg, tuple),
            quantized=fp8 and i < 3,
        )
        for i, arg in enumerate(runtime)
    )
    try:
        executor(*bound, block_num=block_num, stream=stream)
    finally:
        bindings.clear()


@lru_cache(maxsize=32)
def _cores(device, mixed=True):
    props = torch.npu.get_device_properties(device)
    if "950" not in props.name:
        raise RuntimeError(f"only Ascend 950 / architecture 3510 is supported, got {props.name}")
    return int(props.cube_core_num if mixed else props.vector_core_num)


@_on_tensor_device
def launch_mask(
    pc,
    pi,
    fc,
    fi,
    cu_q,
    cu_k,
    program,
    aux_tensors,
    aux_scalars,
    B,
    H,
    Sq,
    Sk,
    *,
    use_fast_sampling=False,
):
    if program.simt is None:
        from .mods import UnsupportedModError

        raise UnsupportedModError(
            "device sparse metadata construction requires a SIMT implementation; use @simd(simt=...)"
        )
    from .kernels.mask import mask_kernel

    tla = _tla()
    varlen = (cu_q is not None, cu_k is not None)
    dummy = _sentinel(pc.device)
    rows, cols = pc.numel(), pi.shape[-1]
    ts, ss = _aux(aux_tensors, aux_scalars)
    runtime = (
        _buffer(pc),
        _buffer(pi),
        _buffer(fc),
        _buffer(fi),
        _buffer(cu_q if varlen[0] else dummy),
        _buffer(cu_k if varlen[1] else dummy),
        ts,
        ss,
        tla.Int32(B),
        tla.Int32(H),
        tla.Int32(Sq),
        tla.Int32(Sk),
        tla.Int32(cols),
        tla.Int32(fi.shape[-1]),
    )
    from .auxiliary import bind_aux

    specs = _aux_specs(aux_tensors)
    mod = bind_aux(program.simt, "mask", specs)
    executor = _compile(
        (
            "mask",
            pc.device,
            "3510",
            program.key,
            varlen,
            use_fast_sampling,
            _aux_key(aux_tensors),
            tuple(type(x).__name__ for x in aux_scalars),
        ),
        mask_kernel,
        (*runtime, varlen, mod, use_fast_sampling),
        {i: "pointer" for i in range(4)},
    )
    _launch(
        executor,
        runtime,
        block_num=min(rows, _cores(pc.device, mixed=False)),
        device=pc.device,
        tensor_shapes={i: "pointer" for i in range(4)},
    )


@dataclass(frozen=True)
class _AttentionPlan:
    """Per-invocation launch geometry referencing a cached compiled function.

    These plans are never cached. Dimensions, scratch storage and arguments
    belong to one launch; concurrent streams share only the compiled code.
    """

    executor: object
    integers: tuple
    total_tokens: object
    workspace_size: object
    scale_type: object
    scalar_types: tuple
    dummy: torch.Tensor
    backward: bool
    block_num: int

    def arguments(self, q, k, v, out, lse, config, sparse, dout, dq, dk_acc, dv_acc, dk, dv, dlse):
        dummy = self.dummy
        pc, pi, fc, fi = (
            (dummy, dummy, dummy, dummy)
            if sparse is None
            else (
                sparse.mask_block_cnt,
                sparse.mask_block_idx,
                sparse.full_block_cnt,
                sparse.full_block_idx,
            )
        )
        tensors = (
            q,
            k,
            v,
            out,
            lse,
            dout if self.backward else q,
            dq if self.backward else q,
            dk_acc if self.backward else k,
            dv_acc if self.backward else v,
            pc,
            pi,
            fc,
            fi,
            config["cu_q"] if config["cu_q"] is not None else dummy,
            config["cu_k"] if config["cu_k"] is not None else dummy,
        )
        aux, scalars = _aux(config["aux_tensors"], config["aux_scalars"])
        args = (
            *(_buffer(t) for t in tensors),
            aux,
            scalars,
            *self.integers,
            self.scale_type(config["scale"]),
            self.total_tokens,
        )
        if self.backward:
            workspace = (
                (_buffer(dk), _buffer(dv), self.workspace_size)
                if self.workspace_size is not None
                else ()
            )
            args += (workspace, () if dlse is None else (_buffer(dlse),))
        return args

    def __call__(self, q, k, v, out, lse, config, sparse, dout, dq, dk_acc, dv_acc, dk, dv, dlse):
        args = self.arguments(
            q, k, v, out, lse, config, sparse, dout, dq, dk_acc, dv_acc, dk, dv, dlse
        )
        _launch(
            self.executor,
            args,
            block_num=self.block_num,
            device=q.device,
            tensor_shapes=_index_shapes(sparse),
        )


@_on_tensor_device
def launch_attention(
    q,
    k,
    v,
    out,
    lse,
    config,
    *,
    dout=None,
    dq=None,
    dk_acc=None,
    dv_acc=None,
    dk=None,
    dv=None,
    dlse=None,
):
    if config.get("mxfp8_scales") is not None:
        if any(t is not None for t in (dout, dq, dk_acc, dv_acc, dk, dv, dlse)):
            raise NotImplementedError("MXFP8 supports inference forward only")
        return launch_mxfp8(q, k, v, out, lse, config)
    if q.numel() == 0 and dout is None:
        return
    backward = dout is not None
    sparse = (config.get("block_sparse_tensors_bwd") if backward else None) or config[
        "block_sparse_tensors"
    ]
    plan = _make_attention_plan(
        q,
        k,
        v,
        out,
        lse,
        config,
        dout=dout,
        dq=dq,
        dk_acc=dk_acc,
        dv_acc=dv_acc,
        dk=dk,
        dv=dv,
        dlse=dlse,
    )
    plan(q, k, v, out, lse, config, sparse, dout, dq, dk_acc, dv_acc, dk, dv, dlse)


def _make_attention_plan(
    q,
    k,
    v,
    out,
    lse,
    config,
    *,
    dout=None,
    dq=None,
    dk_acc=None,
    dv_acc=None,
    dk=None,
    dv=None,
    dlse=None,
):
    tla = _tla()
    backward = dout is not None
    sparse = (config.get("block_sparse_tensors_bwd") if backward else None) or config[
        "block_sparse_tensors"
    ]
    dense = sparse is None
    from .builtin import identity, identity_bwd, all_visible, pipeline_adapter
    from .mods import builtin_info, callback_mode

    program = config["score_mod"] or identity
    mask_program = config.get("mask_mod") or all_visible
    dense_mask = dense and mask_program is not all_visible
    backward_program = config["score_mod_bwd"] or identity_bwd
    # Never classify user code as identity, even if its forward is equivalent.
    identity_score = config["score_mod"] is None or (
        program is identity and backward_program is identity_bwd
    )
    mode = callback_mode(*(program, mask_program, backward_program)[: 3 if backward else 2])
    indexed = mode == "simt"
    from .auxiliary import bind_aux

    specs = _aux_specs(config["aux_tensors"])
    grad = bind_aux(getattr(backward_program, mode), "score_bwd", specs)
    varlen = (config["cu_q"] is not None, config["cu_k"] is not None)
    # BSND and TND interleave heads between consecutive sequence rows. Keep
    # this separate from varlen: fixed BSND still has batch-major LSE and no cu.
    sequence_major = tuple(packed or config.get("layout", "BNSD") == "BSND" for packed in varlen)
    if dense:
        mb, mh = 1, 1
    else:
        mb, mh = sparse.mask_block_cnt.shape[:2]
    B, H, KH, Sq, Sk = (config[n] for n in ("B", "H", "KH", "Sq", "Sk"))
    uniform_layout = (not varlen[0] or q.shape[0] == B * Sq) and (
        not varlen[1] or k.shape[0] == B * Sk
    )
    # The SIMD fast path supports fixed layouts and uniform packed MHA.
    pipeline_layout = not any(varlen) or (uniform_layout and H == KH)
    # Only registered built-ins can replace a dense mask with analytic bounds.
    # Explicit sparse lists remain authoritative, including caller-supplied subsets.
    implicit_window = ()
    known_mask = builtin_info(mask_program)
    if (
        dense_mask
        and identity_score
        and not indexed
        and known_mask is not None
        and (H == KH or (not any(varlen) and Sq % 128 == 0 and Sk % 128 == 0))
    ):
        if known_mask[0] == "local_window" and all(x is None or x >= 0 for x in known_mask[1]):
            implicit_window = (*(-1 if x is None else x for x in known_mask[1]), True)
        elif known_mask[0] in ("causal", "causal_bottom_right"):
            implicit_window = (-1, 0, known_mask[0] == "causal_bottom_right")
        elif known_mask[0] == "window":
            implicit_window = (known_mask[1] - 1, 0, False)
    core_count = _cores(q.device)
    # SIMT-only masks need scalar evaluation only on partial sparse blocks.
    # Identity scores let us scale/mask those scores before the SIMD pipeline.
    simt_mask_pipeline = (
        not dense
        and mask_program.simd is None
        and identity_score
        and pipeline_layout
        and Sq % 128 == 0
        and Sk % 128 == 0
    )
    # The bounded DN kernels also support fixed-layout MHA tails. Their
    # ragged flag enables padded tile loads and bounded stores independently
    # of the packed-layout flags.
    ragged = (
        (not uniform_layout or Sq % 128 != 0 or Sk % 128 != 0)
        and H == KH
        and identity_score
        and (not dense_mask or bool(implicit_window))
    )
    aligned_pipeline = (
        not dense_mask and pipeline_layout and not indexed and Sq % 128 == 0 and Sk % 128 == 0
    )
    if backward:
        from .kernels.flash_bwd_arch35 import flash_bwd_kernel as kernel
    else:
        from .kernels.flash_fwd_arch35 import flash_fwd_kernel as kernel

    generic = not (
        implicit_window
        or ragged
        or simt_mask_pipeline
        or (aligned_pipeline and (identity_score or not backward))
    )
    if generic:
        ragged = not uniform_layout or Sq % 128 != 0 or Sk % 128 != 0
    # Both DN kernels use head-major packed LSE, including nonuniform TND.
    score = pipeline_adapter(
        program,
        "score",
        specs,
        mode="simt" if generic and indexed else "simd",
        varlen=varlen,
    )
    mask = pipeline_adapter(
        mask_program,
        "mask",
        specs,
        mode="simt"
        if (generic and indexed) or simt_mask_pipeline or (ragged and mask_program.simd is None)
        else "simd",
        varlen=varlen,
    )
    static = (
        varlen,
        dense,
        backward,
        indexed,
        score,
        mask,
        grad,
        sequence_major,
        ragged,
        implicit_window,
        generic,
        dense_mask,
    )
    if not backward:
        # A scalar mask does not force a vector-capable score callback into SIMT.
        # Pass both adapters explicitly; the forward kernel owns their scheduling.
        vector_score = None
        if dense and generic and indexed and program.simd is not None:
            vector_score = pipeline_adapter(program, "score", specs, mode="simd", varlen=varlen)
        # Preserve unsigned storage metadata across the signed auxiliary ABI.
        temporal_aux = bool(specs) and all(spec[2] for spec in specs)
        static += (vector_score, temporal_aux)
    constructors = {bool: tla.Bool, int: tla.Int32, float: tla.Float32}
    plan = _AttentionPlan(
        executor=None,
        integers=tuple(tla.Int32(x) for x in (B, H, KH, Sq, Sk, (Sq + 127) // 128, mb, mh)),
        total_tokens=tla.Int32(q.shape[0] if varlen[0] else B * Sq),
        workspace_size=(
            tla.Int32(dk.numel()) if backward and dk is not None and dv is not None else None
        ),
        scale_type=tla.Float32,
        scalar_types=tuple(constructors[type(value)] for value in config["aux_scalars"]),
        dummy=_sentinel(q.device),
        backward=backward,
        block_num=max(1, min(B * H * ((Sq + 127) // 128), core_count)),
    )
    args = plan.arguments(q, k, v, out, lse, config, sparse, dout, dq, dk_acc, dv_acc, dk, dv, dlse)
    # The backward workspace and optional dLSE follow the Constexpr slots in
    # the kernel signature. Empty tuples must keep their ABI positions too.
    compile_args = (*args[:-2], *static, *args[-2:]) if backward else (*args, *static)
    key = (
        "bwd" if backward else "fwd",
        q.device,
        str(q.dtype),
        "3510",
        H,
        KH,
        static,
        _aux_key(config["aux_tensors"]),
        tuple(type(x).__name__ for x in config["aux_scalars"]),
        plan.workspace_size is not None,
        dlse is not None,
    )
    executor = _compile(key, kernel, compile_args, _index_shapes(sparse))
    return replace(plan, executor=executor)


# MXFP8 forward planning uses the same cache, bindings and stream ownership.


def _compile_mxfp8(args, index_shape):
    from .kernels.flash_fwd_mxfp8_arch35 import flash_fwd_mxfp8_kernel

    tla = _tla()
    samples = tuple(_sample(arg, index_shape.get(i), logical=i == 15) for i, arg in enumerate(args))
    return tla.compile(flash_fwd_mxfp8_kernel, *samples, options="--npu-arch 3510")


class _MXFP8Features(NamedTuple):
    """All MXFP8 launch choices that affect generated instructions."""

    single_sequence: bool
    q_heads: int
    kv_heads: int
    short_delay: bool
    stream_output: bool
    query_major: bool
    bounded_inline: bool
    paired_heads: bool
    dense_bounds: tuple = ()
    key_aux: tuple = ()
    key_aux_scope: tuple = ()
    periodic_prefix: bool = False


@dataclass(frozen=True)
class _MXFP8Plan:
    executor: object
    integers: tuple
    total_tokens: object
    scale_type: object
    scalar_types: tuple
    dummy: torch.Tensor
    block_num: int
    guard_elements: int = 0
    pack_kv: bool = False

    def arguments(self, q, k, v, out, lse, config, offsets):
        sparse = config["block_sparse_tensors"]
        pc, pi, fc, fi = (
            (self.dummy,) * 4
            if sparse is None
            else (
                sparse.mask_block_cnt,
                sparse.mask_block_idx,
                sparse.full_block_cnt,
                sparse.full_block_idx,
            )
        )
        tensors = (
            q,
            k,
            v,
            *config["mxfp8_scales"],
            self.dummy if offsets is None else offsets,
            out,
            lse,
            pc,
            pi,
            fc,
            fi,
            self.dummy if config["cu_q"] is None else config["cu_q"],
            self.dummy if config["cu_k"] is None else config["cu_k"],
        )
        aux, scalars = _aux(config["aux_tensors"], config["aux_scalars"])
        return (
            *(_buffer(t) for t in tensors),
            aux,
            scalars,
            *self.integers,
            self.scale_type(config["scale"]),
            self.total_tokens,
            _tla().Int32(k.shape[0] if config["cu_k"] is not None else config["B"] * config["Sk"]),
            _tla().Int32(config["mxfp8_scales"][2].shape[0] if config["cu_k"] is not None else 0),
        )


def _mxfp8_stream_output(tensors, l2_bytes):
    """Reserve shared L2 for input reuse when the launch's working set grows.

    This decision uses tensor metadata and device capacity, never sequence
    length cutoffs or input values. Smaller launches retain cached output
    writes; larger launches avoid evicting Q/K/V with write-only output.
    """
    working_bytes = sum(t.numel() * t.element_size() for t in tensors)
    return l2_bytes > 0 and working_bytes > l2_bytes // 2


def _mxfp8_should_pack_kv(kv_bytes, reuse, l2_bytes):
    """Amortize layout copies only when wide traversal exceeds shared cache."""
    return l2_bytes > 0 and kv_bytes > l2_bytes // 2 and reuse >= 8


def _make_mxfp8_plan(q, k, v, out, lse, config, offsets):
    from .builtin import all_visible, identity, pipeline_adapter
    from .mods import callback_mode

    tla = _tla()
    program = config["score_mod"] or identity
    mask_program = config["mask_mod"] or all_visible
    if (
        config["block_sparse_tensors"] is None
        and mask_program is not all_visible
        and mask_program.simt is not None
        and not config["aux_tensors"]
        and not config["aux_scalars"]
    ):
        from .kernels.flash_fwd_mxfp8_arch35 import _mx_mask_is_all_visible

        if _mx_mask_is_all_visible(mask_program.simt):
            mask_program = all_visible
    mode = callback_mode(program, mask_program)
    if (
        (
            config["cu_q"] is not None
            or config["cu_k"] is not None
            or (config["Sq"] >= 1024 and config["Sk"] >= 1024)
        )
        and program is identity
        and mask_program is not all_visible
        and mask_program.simt is not None
    ):
        mode = "simt"
    indexed = mode == "simt"
    varlen = (config["cu_q"] is not None, config["cu_k"] is not None)
    sequence_major = tuple(packed or config.get("layout", "BNSD") == "BSND" for packed in varlen)
    specs = _aux_specs(config["aux_tensors"])
    B, H, KH, Sq, Sk = (config[n] for n in ("B", "H", "KH", "Sq", "Sk"))
    score = pipeline_adapter(program, "score", specs, mode=mode, varlen=varlen)
    mask = pipeline_adapter(mask_program, "mask", specs, mode=mode, varlen=varlen)
    vector_score = (
        pipeline_adapter(program, "score", specs, mode="simd", varlen=varlen)
        if indexed and program.simd is not None
        else None
    )
    sparse = config["block_sparse_tensors"]
    dense = sparse is None
    if dense:
        mb, mh = 1, 1
    else:
        mb, mh = sparse.mask_block_cnt.shape[:2]
    constructors = {bool: tla.Bool, int: tla.Int32, float: tla.Float32}
    plan = _MXFP8Plan(
        executor=None,
        integers=tuple(tla.Int32(x) for x in (B, H, KH, Sq, Sk, (Sq + 127) // 128, mb, mh)),
        total_tokens=tla.Int32(q.shape[0] if varlen[0] else B * Sq),
        scale_type=tla.Float32,
        scalar_types=tuple(constructors[type(x)] for x in config["aux_scalars"]),
        dummy=_sentinel(q.device),
        block_num=max(1, min(B * H * ((Sq + 127) // 128), _cores(q.device))),
    )
    args = plan.arguments(q, k, v, out, lse, config, offsets)
    static = (
        varlen,
        dense,
        indexed,
        score,
        mask,
        sequence_major,
        dense and mask_program is not all_visible,
        vector_score,
        any(varlen) or Sq % 128 != 0 or Sk % 128 != 0,
        program is identity,
    )
    continuous = (
        program is identity
        and mask_program.simt is not None
        and ((Sq >= 1024 and Sk >= 1024) or any(varlen))
    )
    pipeline, group_size, guard_elements = "general", 16, 0
    if continuous:
        pipeline = "continuous"
        # Auxiliary masks use up to four lookahead tiles; masks without
        # auxiliary storage can use six. Neither traverses unused capacity.
        # Keep two extra P slots while consuming every actual KV block.
        group_size = (
            4
            if dense
            else max(
                1, min(4, max(sparse.mask_block_idx.shape[-1], sparse.full_block_idx.shape[-1]))
            )
        )
        if not config["aux_tensors"]:
            group_size = (
                6
                if dense
                else max(
                    1, min(6, max(sparse.mask_block_idx.shape[-1], sparse.full_block_idx.shape[-1]))
                )
            )
        guard_elements = (B * H * ((Sq + 127) // 128) + plan.block_num) * 16
    mask_kind = (
        0
        if mask_program is all_visible
        else 2
        if any(t.dtype == torch.uint32 for t in config["aux_tensors"])
        else 1
    )
    aux_cache_meta = []
    cache_bytes = 0
    if continuous and mask_kind == 1:
        valid_aux = all(
            1 <= t.ndim <= 3
            and t.is_contiguous()
            and t.dtype in (torch.int32, torch.int64)
            and t.shape[-1] >= max(Sq, Sk)
            and (
                not any(varlen)
                or t.ndim >= 2
                or t.shape[0]
                >= max(q.shape[0] if varlen[0] else Sq, k.shape[0] if varlen[1] else Sk)
            )
            for t in config["aux_tensors"]
        )
        if valid_aux:
            sizes = [
                (t.numel() * t.element_size() + 511) // 512 * 512 for t in config["aux_tensors"]
            ]
            tile_cache = bool(any(varlen) or Sq % 128 or Sk % 128 or sum(sizes) > 131072)
            for tensor, size in zip(config["aux_tensors"], sizes):
                storage = 3 * 128 * tensor.element_size() if tile_cache else size
                if cache_bytes + storage > 131072:
                    aux_cache_meta = []
                    break
                # Negative offsets select query + two KV tile caches.
                offset = -1 - cache_bytes if tile_cache else cache_bytes
                aux_cache_meta.append(
                    (
                        offset if tile_cache else 0,
                        (None,) * tensor.ndim,
                        (None,) * tensor.ndim,
                        tensor.element_size() == 8,
                    )
                )
                cache_bytes += storage
    post_quant = (
        continuous
        and sparse is not None
        and not any(varlen)
        and mask_kind == 2
        and max(sparse.mask_block_idx.shape[-1], sparse.full_block_idx.shape[-1]) >= 8
    )
    inline_quantize = None
    bounded_inline, periodic_mask = False, False
    mask_span = None
    if continuous and mask_kind == 1 and not config["aux_tensors"]:
        from .kernels.flash_fwd_mxfp8_arch35 import _mx_trace_affine_mask

        traced = _mx_trace_affine_mask(
            mask_program.simt, Sq, Sk, len(config["aux_scalars"]), post_quant, any(varlen)
        )
        if traced is not None:
            inline_quantize, bounded_inline, mask_span, periodic_mask = traced
    if (
        dense
        and continuous
        and config["aux_scalars"]
        and not config["aux_tensors"]
        and inline_quantize is None
    ):
        # Opaque scalar-dependent callbacks retain the exact general path.
        continuous, pipeline, group_size, guard_elements = False, "general", 16, 0
    wide_mask = inline_quantize is not None and mask_span is None
    if continuous and aux_cache_meta and mask_kind == 1:
        from .kernels.flash_fwd_mxfp8_arch35 import _mx_mask_depends_on_query

        # A KV-only mask permits the same head-major reuse as a wide affine
        # mask. Apply the existing working-set/reuse threshold to its layout
        # copies too; neither auxiliary values nor boundary values are read.
        wide_mask = wide_mask or not _mx_mask_depends_on_query(
            mask, config["aux_tensors"], args[16]
        )
    stream_output = False
    pack_kv = False
    if continuous:
        tensors = (*args[:6], *args[7:13], *config["aux_tensors"])
        properties = torch.npu.get_device_properties(q.device)
        stream_output = _mxfp8_stream_output(tensors, int(getattr(properties, "L2_cache_size", 0)))
        l2_bytes = int(getattr(properties, "L2_cache_size", 0))
        selected = (
            (Sk + 127) // 128
            if dense
            else max(sparse.mask_block_idx.shape[-1], sparse.full_block_idx.shape[-1])
        )
        reuse = H * ((Sq + 127) // 128) * selected / max(1, KH * ((Sk + 127) // 128))
        kv_bytes = sum(t.numel() * t.element_size() for t in (k, v, *config["mxfp8_scales"][1:]))
        pack_kv = (
            sequence_major[1]
            and (wide_mask or (dense and mask_span is None))
            and _mxfp8_should_pack_kv(kv_bytes, reuse, l2_bytes)
        )
        if pack_kv:
            sequence_major = (sequence_major[0], False)
            static = (*static[:5], sequence_major, *static[6:])
    # Narrow sparse neighborhoods stream adjacent heads together. Wider
    # neighborhoods reuse a head's K/V across consecutive query blocks.
    query_major = not dense and not wide_mask
    paired_heads = (
        continuous
        and not config["aux_tensors"]
        and sequence_major[0]
        and sequence_major[1]
        and H == KH
        and KH % 2 == 0
        and mask_span is not None
        and mask_span <= 257
    )
    # The exact online auxiliary pipeline has no speculative phase or repair.
    # Trace the same fake auxiliary dtypes/scalars passed to the compiler.
    legacy_guard_elements = guard_elements
    if (
        continuous
        and 0 < len(aux_cache_meta) <= 4
        and (dense or any(varlen) or all(t.ndim == 1 for t in config["aux_tensors"]))
    ):
        from .kernels.flash_fwd_mxfp8_arch35 import _mx_mask_depends_on_query, _mx_repair_aux

        if _mx_mask_depends_on_query(mask, _sample(args[15], logical=True), args[16]):
            pipeline, guard_elements = "exact_aux", 0
            aux_cache_meta = _mx_repair_aux(tuple(aux_cache_meta))
    # Few-block periodic neighborhoods often have only a handful of visible
    # keys in some rows. Avoid speculating on those rows and repairing them.
    if continuous and periodic_mask and mask_span is None and max(Sq, Sk) <= 2048:
        pipeline, guard_elements = "exact_aux", 0
    if continuous and dense and pipeline == "exact_aux" and kv_bytes <= l2_bytes // 2:
        query_major = True
    from .kernels.flash_fwd_mxfp8_arch35 import _mx_is_periodic_prefix

    features = _MXFP8Features(
        single_sequence=B == 1 and any(varlen),
        q_heads=H,
        kv_heads=KH,
        short_delay=pipeline == "continuous" and max(Sq, Sk) <= 2048,
        stream_output=stream_output,
        query_major=query_major,
        bounded_inline=bounded_inline,
        paired_heads=paired_heads,
        dense_bounds=(),
        periodic_prefix=dense
        and pipeline == "exact_aux"
        and periodic_mask
        and not config["aux_tensors"]
        and not config["aux_scalars"]
        and _mx_is_periodic_prefix(mask_program.simt, len(config["aux_scalars"]), any(varlen)),
    )
    if (
        continuous
        and dense
        and len(aux_cache_meta) == 1
        and len(config["aux_tensors"]) == 1
        and config["aux_tensors"][0].ndim == 1
        and config["aux_tensors"][0].dtype == torch.int32
        and not config["aux_scalars"]
    ):
        from .kernels.flash_fwd_mxfp8_arch35 import _mx_trace_key_aux

        key_aux_scope = _mx_trace_key_aux(mask_program.simt)
        # Multi-sequence launches retain the established mask-cache quantizer.
        features = features._replace(
            key_aux_scope=key_aux_scope, key_aux=key_aux_scope if B == 1 else ()
        )
    if continuous and dense and mask_program is not all_visible and not config["aux_tensors"]:
        from .kernels.flash_fwd_mxfp8_arch35 import _mx_dense_mask_bounds

        features = features._replace(
            dense_bounds=_mx_dense_mask_bounds(
                mask_program.simt, len(config["aux_scalars"]), any(varlen)
            )
        )
    static = (
        *static,
        False,
        pipeline,
        group_size,
        mask_kind,
        features,
        tuple(aux_cache_meta),
        inline_quantize,
        post_quant,
    )
    key = (
        "mxfp8",
        q.device,
        str(q.dtype),
        str(out.dtype),
        "3510",
        H,
        KH,
        static,
        _aux_key(config["aux_tensors"]),
        tuple(str(t.dtype) for t in config["mxfp8_scales"]),
        tuple(type(x).__name__ for x in config["aux_scalars"]),
    )

    def compile_selected():
        try:
            return _compile_mxfp8((*args, *static), _index_shapes(sparse))
        except Exception:
            if not aux_cache_meta:
                raise
            fallback = list(static)
            # Fallback is a compiler capability choice shared by every size
            # in this specialization; retain it with the compiled function.
            if pipeline == "exact_aux":
                fallback[-7] = "continuous"
            fallback[-3] = ()
            fallback[-4] = features._replace(key_aux=())
            compiled = _compile_mxfp8((*args, *fallback), _index_shapes(sparse))
            compiled._attention_guarded_fallback = pipeline == "exact_aux"
            return compiled

    executor = compile_cache.get_or_compile(key, compile_selected)
    # Scratch belongs to this invocation, including a cached fallback.
    if getattr(executor, "_attention_guarded_fallback", False):
        guard_elements = legacy_guard_elements

    return replace(plan, executor=executor, guard_elements=guard_elements, pack_kv=pack_kv)


def launch_mxfp8(q, k, v, out, lse, config):
    from .interface import _resolve_mxfp8_scales

    scales = _resolve_mxfp8_scales(
        q, k, v, config["mxfp8_scales"], layout=config.get("layout", "BNSD")
    )
    if scales is None:
        raise ValueError("MXFP8 launch requires all three dequant scales")
    if out.dtype != torch.bfloat16 or lse.dtype != torch.float32:
        raise ValueError("MXFP8 outputs must be BF16 with FP32 LSE")
    if any(t.requires_grad for t in config["aux_tensors"]):
        raise NotImplementedError("MXFP8 auxiliary inputs must not require gradients")
    if q.numel() == 0:
        return
    sparse = config["block_sparse_tensors"]
    offsets = None
    plan = _make_mxfp8_plan(q, k, v, out, lse, config, offsets)
    if plan.guard_elements:
        # Both vector sub-blocks write one flag in their own 32-byte segment.
        # Every invocation owns its storage; concurrent streams never share it.
        offsets = torch.empty(plan.guard_elements, dtype=torch.int32, device=q.device)
    if plan.pack_kv:
        # Preserve the public TND/BSND metadata while packing internal storage
        # by head. The complete cu_seqlens contract remains device-side.
        axes = (0, 1) if config["cu_k"] is not None else (1, 2)
        stream = torch.npu.current_stream(q.device)
        packed = []
        for tensor in (k, v, *scales[1:]):
            tensor.record_stream(stream)
            packed.append(
                tensor.view(torch.uint8)
                .transpose(*axes)
                .contiguous()
                .reshape(tensor.shape)
                .view(tensor.dtype)
            )
        k, v, ks, vs = packed
        config = dict(config, mxfp8_scales=(scales[0], ks, vs))
    args = plan.arguments(q, k, v, out, lse, config, offsets)
    _launch(
        plan.executor,
        args,
        block_num=plan.block_num,
        device=q.device,
        tensor_shapes=_index_shapes(sparse),
        fp8=True,
    )


def clear_compile_cache(*, persistent=False):
    """Drop executors; optionally purge this frontend cache namespace on disk."""
    compile_cache.clear(persistent=persistent)
