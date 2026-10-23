"""Compile-time TLA binding with direct Tensor launches on a warm cache.

All layouts here are static pointer ABIs. Compiled launch packing reads the
current Torch tensor's data_ptr(), including its storage offset. TLA/DLPack
objects are needed only on a compilation miss. Explicit record_stream keeps
Torch allocations alive after the asynchronous external kernel launch.
"""

from __future__ import annotations

from functools import wraps

import torch

from .cache_utils import get_jit_cache

compile_cache = get_jit_cache()


def _on_tensor_device(fn):
    @wraps(fn)
    def guarded(first, *args, **kwargs):
        with torch.npu.device(first.device):
            return fn(first, *args, **kwargs)

    return guarded


def _tla():
    from .utils import get_tla

    return get_tla()


def _tensor(t, shape=None):
    tla = _tla()
    # The backend requires positive ABI extents even for buffers which are
    # never accessed (e.g. all KV segments empty). Logical bounds remain zero.
    t = _buffer(t)
    return tla.from_dlpack(
        t.detach(), origin_shape=shape or (t.numel(),), layout_tag=tla.arch.RowMajor
    )


def _buffer(t):
    """Provide a positive-size ABI sentinel for a logically empty tensor."""
    return torch.empty((1,), dtype=t.dtype, device=t.device) if t.numel() == 0 else t


def _sample(value, shape=None, *, logical=False):
    if isinstance(value, torch.Tensor):
        if logical:
            from catlass.tla.runtime import make_fake_tensor

            tla = _tla()
            dtype = {
                torch.bool: tla.Bool,
                torch.int32: tla.Int32,
                torch.int64: tla.Int64,
                torch.uint32: tla.Int32,
                torch.float16: tla.Float16,
                torch.bfloat16: tla.BFloat16,
                torch.float32: tla.Float32,
            }[value.dtype]
            extent = max(
                1,
                1 + sum((n - 1) * stride for n, stride in zip(value.shape, value.stride())),
            )
            return make_fake_tensor(
                dtype,
                (extent,),
                (1,),
                origin_shape=(extent,),
                coord=(0,),
                layout_tag=tla.arch.RowMajor,
            )
        return _tensor(value, shape)
    if isinstance(value, tuple):
        return tuple(_sample(item, logical=logical) for item in value)
    return value


def _aux(tensors, scalars):
    tla = _tla()
    ts = tuple(_buffer(t) for t in tensors)
    ss = tuple(
        tla.Bool(s) if type(s) is bool else tla.Int32(s) if type(s) is int else tla.Float32(s)
        for s in scalars
    )
    return ts, ss


def _compile(key, kernel, args, tensor_shapes=None):
    def compile_kernel():
        tla = _tla()
        shapes = tensor_shapes or {}
        samples = tuple(
            _sample(arg, shapes.get(i), logical=i in (6, 15) and isinstance(arg, tuple))
            for i, arg in enumerate(args)
        )
        return tla.compile(kernel, *samples, options="--npu-arch 3510")

    return compile_cache.get_or_compile(key, compile_kernel)


def _launch(executor, runtime, *, block_num, device):
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
    executor(*runtime, block_num=block_num, stream=stream)


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
    dummy = torch.empty((1,), dtype=torch.int32, device=pc.device)
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
    )
    from .auxiliary import bind_aux

    specs = tuple((tuple(t.shape), tuple(t.stride()), t.dtype == torch.uint32) for t in aux_tensors)
    mod = bind_aux(program.simt, "mask", specs)
    executor = _compile(
        (
            "mask",
            str(pc.device),
            program.key,
            varlen,
            use_fast_sampling,
            B,
            H,
            Sq,
            Sk,
            tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype)) for t in aux_tensors),
            tuple(type(x).__name__ for x in aux_scalars),
        ),
        mask_kernel,
        (*runtime, varlen, mod, use_fast_sampling),
        {1: (rows, cols), 3: (rows, cols)},
    )
    _launch(
        executor,
        runtime,
        block_num=min(rows, _cores(pc.device, mixed=False)),
        device=pc.device,
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
    if q.numel() == 0 and dout is None:
        return
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

    specs = tuple(
        (tuple(t.shape), tuple(t.stride()), t.dtype == torch.uint32) for t in config["aux_tensors"]
    )
    grad = bind_aux(getattr(backward_program, mode), "score_bwd", specs)
    varlen = (config["cu_q"] is not None, config["cu_k"] is not None)
    # BSND and TND interleave heads between consecutive sequence rows. Keep
    # this separate from varlen: fixed BSND still has batch-major LSE and no cu.
    sequence_major = tuple(packed or config.get("layout", "BNSD") == "BSND" for packed in varlen)
    dummy = torch.empty((1,), dtype=torch.int32, device=q.device)
    if dense:
        pc, pi, fc, fi = dummy, dummy, dummy, dummy
        shape = (1, 1)
        mb, mh = 1, 1
    else:
        pc, pi, fc, fi = (
            sparse.mask_block_cnt,
            sparse.mask_block_idx,
            sparse.full_block_cnt,
            sparse.full_block_idx,
        )
        shape = (pc.numel(), pi.shape[-1])
        mb, mh = pc.shape[:2]
    B, H, KH, Sq, Sk = (config[n] for n in ("B", "H", "KH", "Sq", "Sk"))
    ts, ss = _aux(config["aux_tensors"], config["aux_scalars"])
    runtime = tuple(
        _buffer(t)
        for t in (
            q,
            k,
            v,
            out,
            lse,
            dout if backward else q,
            dq if backward else q,
            dk_acc if backward else k,
            dv_acc if backward else v,
        )
    )
    runtime += (
        _buffer(pc),
        _buffer(pi),
        _buffer(fc),
        _buffer(fi),
        _buffer(config["cu_q"] if varlen[0] else dummy),
        _buffer(config["cu_k"] if varlen[1] else dummy),
        ts,
        ss,
    )
    runtime += tuple(tla.Int32(x) for x in (B, H, KH, Sq, Sk, (Sq + 127) // 128, mb, mh))
    runtime += (tla.Float32(config["scale"]),)
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
        from .kernels.flash_bwd_arch35 import attention_kernel

        implementation = "pipeline_backward_dn"
    else:
        from .kernels.flash_fwd_arch35 import attention_kernel

        implementation = "pipeline_forward_dn"
    generic = not (
        implicit_window
        or ragged
        or simt_mask_pipeline
        or (aligned_pipeline and (identity_score or not backward))
    )
    if generic:
        ragged = not uniform_layout or Sq % 128 != 0 or Sk % 128 != 0
    # Both DN kernels use head-major packed LSE, including nonuniform TND.
    runtime += (tla.Int32(q.shape[0] if varlen[0] else B * Sq),)
    score = pipeline_adapter(
        program,
        "score",
        Sq,
        Sk,
        specs,
        mode="simt" if generic and indexed else "simd",
        varlen=varlen,
    )
    mask = pipeline_adapter(
        mask_program,
        "mask",
        Sq,
        Sk,
        specs,
        mode="simt"
        if (generic and indexed) or simt_mask_pipeline or (ragged and mask_program.simd is None)
        else "simd",
        varlen=varlen,
    )
    static = (varlen, dense, backward, indexed, score, mask, grad, sequence_major)
    key = (
        "attention",
        implementation,
        str(q.device),
        str(q.dtype),
        program.key,
        mask_program.key,
        varlen,
        dense,
        backward,
        indexed,
        128,
        tuple(q.shape),
        tuple(k.shape),
        shape,
        B,
        backward_program.key,
        tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype)) for t in config["aux_tensors"]),
        tuple(type(x).__name__ for x in config["aux_scalars"]),
    )
    static += (ragged, implicit_window, generic, dense_mask)
    key += (
        "sequence_major_v1",
        sequence_major,
        "head_major_lse_v1",
        "ragged_pipeline_v1",
        ragged,
        "implicit_window_v1",
        implicit_window,
        "generic_pipeline_v1",
        generic,
        dense_mask,
    )
    if not backward:
        # A scalar mask does not force a vector-capable score callback into SIMT.
        # Pass both adapters explicitly; the forward kernel owns their scheduling.
        vector_score = None
        if dense and generic and indexed and program.simd is not None:
            vector_score = pipeline_adapter(
                program, "score", Sq, Sk, specs, mode="simd", varlen=varlen
            )
        # Preserve unsigned storage metadata across the signed auxiliary ABI.
        temporal_aux = bool(specs) and all(spec[2] for spec in specs)
        static += (vector_score, temporal_aux)
        key += (
            "dense_vector_score_v1",
            vector_score is not None,
            "unsigned_aux_layout_v1",
            temporal_aux,
        )
    # Evaluate masks in existing on-chip tiles; never allocate a mask workspace.
    if backward and dk is not None and dv is not None:
        workspace = (_buffer(dk), _buffer(dv), tla.Int32(dk.numel()))
        key += ("fused_workspace_v1", dk.numel())
        compile_args = (*runtime, *static, workspace)
        runtime += (workspace,)
    else:
        compile_args = (*runtime, *static)
        if backward:
            # Even an empty runtime tuple occupies one executor argument.
            compile_args += ((),)
            runtime += ((),)
    if backward:
        # Append the optional tangent after the existing workspace ABI. Empty
        # tuples specialize away all dLSE loads in the ordinary backward.
        lse_gradient = () if dlse is None else (_buffer(dlse),)
        compile_args += (lse_gradient,)
        runtime += (lse_gradient,)
        key += ("lse_gradient_v1", dlse is not None)
    executor = _compile(key, attention_kernel, compile_args, {10: shape, 12: shape})
    _launch(
        executor,
        runtime,
        block_num=max(1, min(B * H * ((Sq + 127) // 128), core_count)),
        device=q.device,
    )


def clear_compile_cache():
    """Drop this process's executor references; does not remove disk artifacts."""
    compile_cache.clear()
