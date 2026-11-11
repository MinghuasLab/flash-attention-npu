"""Compile attention variants during application setup."""

from __future__ import annotations

import torch

from .cache_utils import CompileCacheInfo


def compile_cache_info() -> CompileCacheInfo:
    """Inspect memory reuse, persistent loads and successful frontend compiles."""
    from .runtime import compile_cache

    return compile_cache.cache_info()


def _prepare_compilation(q, k, v, *, varlen, training=False, **options):
    """Share public metadata validation without executing attention."""
    from .interface import _canonical_input, _maximum_length, _prepare_config

    if torch.compiler.is_compiling():
        raise RuntimeError("precompilation must run outside torch.compile tracing")
    if type(varlen) is not bool:
        raise TypeError("varlen must be bool")
    cu_q, cu_k = options.pop("cu_seqlens_q"), options.pop("cu_seqlens_k")
    max_q, max_k = options.pop("max_seqlen_q"), options.pop("max_seqlen_k")
    if not varlen and any(value is not None for value in (cu_q, cu_k, max_q, max_k)):
        raise ValueError("sequence length arguments require varlen=True")
    if not varlen and any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in (q, k, v)):
        raise ValueError("fixed attention requires BSND; set varlen=True for TND")
    window = options.pop("window_size")
    if not isinstance(window, (tuple, list)) or len(window) != 2:
        raise ValueError("window_size must contain (left, right)")
    if any(bound is not None and type(bound) is not int for bound in window):
        raise ValueError("window bounds must be integers or None")
    window = tuple(None if bound is None or bound < 0 else bound for bound in window)
    q, k, v = (_canonical_input(t) for t in (q, k, v))
    config = _prepare_config(
        q,
        k,
        v,
        **options,
        window_size=window,
        training=training,
        layout="BSND",
        cu_q=cu_q,
        cu_k=cu_k,
        max_q=_maximum_length(q, max_q) if varlen else None,
        max_k=_maximum_length(k, max_k) if varlen else None,
    )
    if q.numel() == 0 or k.numel() == 0:
        raise ValueError("precompilation requires nonempty Q and KV prototypes")
    return q, k, v, config


def _output_prototypes(q, config):
    dtype = torch.bfloat16 if config["mxfp8_scales"] is not None else q.dtype
    out = torch.empty(q.shape, dtype=dtype, device=q.device)
    shape = (q.shape[1], q.shape[0]) if q.ndim == 3 else (q.shape[0], q.shape[2], q.shape[1])
    return out, torch.empty(shape, dtype=torch.float32, device=q.device)


def precompile_attention(
    q,
    k,
    v,
    *,
    varlen=False,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    score_mod=None,
    mask_mod=None,
    aux_tensors=None,
    aux_scalars=None,
    block_sparse_tensors=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    return_lse=False,
    q_descale=None,
    k_descale=None,
    v_descale=None,
) -> CompileCacheInfo:
    """Prepare a forward executor without evaluating attention.

    Inputs follow the public BSND/TND, callback, sparse-table and MXFP8
    contracts. Four-dimensional inputs always mean BSND; BNSD and a ``layout``
    argument are not supported here. Use ``varlen=True`` for the varlen API:
    Q and KV may each independently be BSND or packed TND, including BSND on
    both sides. Each TND side needs its own ``cu_seqlens``; a BSND side must
    omit its ``cu_seqlens``. K and V must share their shape and layout.
    For native MXFP8,
    pass E4M3FN Q/K/V and the existing ``q_descale/k_descale/v_descale`` tensors;
    the scale directions and layouts are unchanged.

    Baseline planning is the default. Use ``forward_compile_mode('broad')``
    around both this helper and actual attention execution to opt into broad
    reuse; the same context also covers calls to a compiled graph. Without an
    explicit context, ``FLASH_ATTENTION_CATLASS_DSL_DYNAMIC_SHAPES=1`` opts in.
    Changing modes selects a separate valid cache variant, so preparation in
    one mode does not warm the other mode. ``get_forward_compile_mode()`` makes
    the current selection observable without changing cache statistics.

    Broad mode reuses one executor across legal B/H/S dimensions
    and auxiliary shapes/strides. D=128, dtype, packed/fixed layout, callback
    identity and mode, dense/sparse traversal and auxiliary ranks/types remain
    code-generation choices. No maximum-shape range needs to be reserved.
    ``FLASH_ATTENTION_CATLASS_DSL_DYNAMIC_SHAPES=0`` selects baseline planning.

    Length hints and cumulative arrays use the public metadata validation.
    Packed offsets must start at zero, be nondecreasing, end at total tokens
    and describe lengths within the supplied maximum. These NPU contents are
    caller invariants; preparation does not read them back to check values.
    A missing packed maximum uses that side's total tokens; fixed addressing
    always uses the actual tensor length. Tensor-valued maximum hints retain
    the public behavior of being ignored without a device readback.

    Call outside ``torch.compile`` during setup. Input and sparse layout
    normalization may copy device data, but no attention kernel, autograd or
    backward kernel is executed. Sparse tables must already be built. The
    temporary output/LSE prototypes are discarded after compilation. Empty
    Q/KV prototypes are rejected because empty calls bypass the kernel.

    The cache retains the baseline persistent-artifact behavior. Returned
    statistics are cumulative shared-cache counts, not a per-call delta. They
    distinguish successful compiler callbacks from disk loads and are not
    reset by preparation or by the mode context. Compare before/after snapshots
    when a delta is useful, accounting for other concurrent cache users.

    ``return_lse`` is accepted for parity with attention configuration; both
    flag values prepare the same forward kernel, which computes LSE internally.
    It does not split cache variants or change this helper's statistics return.
    The helper is forward-only and accepts no ``backward`` or ``deterministic``
    option. It does not prepare gradient kernels.
    """
    from .runtime import launch_attention

    q, k, v, config = _prepare_compilation(
        q,
        k,
        v,
        varlen=varlen,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        score_mod=score_mod,
        score_mod_bwd=None,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        aux_scalars=aux_scalars,
        block_sparse_tensors=block_sparse_tensors,
        block_sparse_tensors_bwd=None,
        return_lse=return_lse,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    out, lse = _output_prototypes(q, config)
    launch_attention(q, k, v, out, lse, config, compile_only=True)
    return compile_cache_info()


def precompile_backward_attention(
    q,
    k,
    v,
    *,
    varlen=False,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    score_mod=None,
    score_mod_bwd=None,
    mask_mod=None,
    aux_tensors=None,
    aux_scalars=None,
    block_sparse_tensors=None,
    block_sparse_tensors_bwd=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    lse_gradient=False,
    q_descale=None,
    k_descale=None,
    v_descale=None,
) -> CompileCacheInfo:
    """Prepare FP16/BF16 backward without running forward or backward.

    Inputs follow the public BSND/TND contracts and do not need gradients.
    Q/KV may independently use fixed or packed layouts with ``varlen=True``.
    An independent backward sparse table is available for the fixed interface;
    the varlen interface uses its forward sparse graph. Custom score callbacks
    require their real explicit ``score_mod_bwd`` VJP. MXFP8 is unsupported.

    Use an independent ``backward_compile_mode('broad')`` context to opt into
    B/H/S and auxiliary-layout reuse; the default remains baseline planning.
    Forward mode is unaffected. ``lse_gradient=True`` prepares the variant
    with a dLSE input, as used by a loss that consumes a returned LSE. Prepare
    the ordinary output-gradient and dLSE variants separately when both occur.

    Prototype O/LSE, upstream gradients and FP32 accumulation workspaces are
    allocated with the correct metadata, but their contents are never read by
    a kernel. Preparation calls only the compiler or persistent loader. It
    returns cumulative cache statistics and does not reset existing entries.
    Layout normalization may copy inputs; NPU cumulative values remain caller
    invariants, as in :func:`precompile_attention`.
    """
    from .runtime import get_backward_compile_mode, launch_attention

    if type(lse_gradient) is not bool:
        raise TypeError("lse_gradient must be bool")
    if any(scale is not None for scale in (q_descale, k_descale, v_descale)) or any(
        isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float8_e4m3fn
        for tensor in (q, k, v)
    ):
        raise NotImplementedError("MXFP8 backward is unsupported; use FP16 or BF16 prototypes")
    if varlen and block_sparse_tensors_bwd is not None:
        raise TypeError("the varlen interface does not accept block_sparse_tensors_bwd")
    q, k, v, config = _prepare_compilation(
        q,
        k,
        v,
        varlen=varlen,
        training=True,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        aux_scalars=aux_scalars,
        block_sparse_tensors=block_sparse_tensors,
        block_sparse_tensors_bwd=block_sparse_tensors_bwd,
        return_lse=lse_gradient,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    config["backward_compile_mode"] = get_backward_compile_mode()
    out, lse = _output_prototypes(q, config)
    launch_attention(
        q,
        k,
        v,
        out,
        lse,
        config,
        dout=torch.empty_like(out),
        dq=torch.empty_like(q),
        dk_acc=torch.empty_like(k, dtype=torch.float32),
        dv_acc=torch.empty_like(v, dtype=torch.float32),
        dk=torch.empty_like(k),
        dv=torch.empty_like(v),
        dlse=torch.empty_like(lse) if lse_gradient else None,
        compile_only=True,
    )
    return compile_cache_info()
