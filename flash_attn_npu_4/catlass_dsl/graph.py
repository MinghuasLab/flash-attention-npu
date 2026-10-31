"""Opaque CATLASS launches for Dynamo, FakeTensor and AOTAutograd.

Only immutable callback identities live in the process-local registry. Tensor
storage, lengths, sparse lists and scalar values always arrive as op inputs.
The runtime uses the same validation and kernels as the eager interface.
"""

from threading import RLock

import torch

from .block_sparsity import BlockSparseTensorsTorch
from .decorators import _Callback
from .mods import MaskMod, Mod

_CALLBACKS = {}
_CALLBACK_KEYS = {}
_CALLBACK_LOCK = RLock()


@torch._dynamo.assume_constant_result
def _register_callbacks(*descriptors):
    # Strong references prevent id reuse while a compiled graph still exists.
    # Do not include shapes or Tensor captures: those belong to runtime inputs.
    identity = tuple(
        tuple(id(value) if callable(value) else value for value in spec) for spec in descriptors
    )
    with _CALLBACK_LOCK:
        if identity not in _CALLBACK_KEYS:
            callbacks = tuple(_restore_callback(spec) for spec in descriptors)
            key = len(_CALLBACKS)
            _CALLBACK_KEYS[identity] = key
            _CALLBACKS[key] = callbacks
        return _CALLBACK_KEYS[identity]


def _callback_spec(fn):
    # Dynamo cannot materialize arbitrary frozen dataclasses as Python
    # constants. Pass their callable fields and primitive metadata instead.
    if isinstance(fn, MaskMod):
        return ("mask", fn.simt, fn.simd, fn.torch_ref, fn.broadcast_batch, fn.broadcast_head)
    if isinstance(fn, Mod):
        return ("mod", fn.simt, fn.simd, fn.torch_ref)
    if isinstance(fn, _Callback):
        return ("marked", fn.simt, fn.simd, fn.mode)
    return ("native", fn)


def _restore_callback(spec):
    kind, *fields = spec
    if kind == "native":
        return fields[0]
    if kind == "marked":
        return _Callback(*fields)
    # Preserve registered built-in identities and their analytic fast paths.
    from .mods import _BUILTINS

    for callback in _BUILTINS:
        if _callback_spec(callback) == spec:
            return callback
    return (MaskMod if kind == "mask" else Mod)(*fields)


def _pack_sparse(*metadata):
    tensors, spec = [], []
    for sparse in metadata:
        if sparse is None:
            spec.extend([0, -1, -1, -1, -1, -1, -1])
            continue
        if not isinstance(sparse, BlockSparseTensorsTorch):
            raise TypeError("expected BlockSparseTensorsTorch")
        if sparse.block_size not in (None, (128, 128)):
            raise NotImplementedError("only block_size=(128, 128) is supported")
        if any(
            x is not None for x in (sparse.dq_write_order, sparse.dq_write_order_full, sparse.spt)
        ):
            raise NotImplementedError("sparse scheduling fields are not supported")
        spec.append(1)
        for tensor in sparse[:6]:
            if tensor is None:
                spec.append(-1)
            else:
                spec.append(len(tensors))
                tensors.append(tensor)
    return tensors, spec


def _unpack_sparse(tensors, spec):
    result = []
    for offset in range(0, len(spec), 7):
        if not spec[offset]:
            result.append(None)
        else:
            values = [
                None if index < 0 else tensors[index] for index in spec[offset + 1 : offset + 7]
            ]
            result.append(BlockSparseTensorsTorch(*values, block_size=(128, 128)))
    return result


# Explicit schemas keep callback handles static while dimension arguments stay
# symbolic. Tensor lists contain no None entries, including under AOTAutograd.
_ATTENTION_ARGS = (
    "Tensor q, Tensor k, Tensor v, Tensor? cu_q, Tensor? cu_k, "
    "Tensor[] sparse, Tensor[] aux, int[] sparse_spec, int callbacks, "
    "float? scale, bool causal, int? left, int? right, float cap, Scalar[] scalars, "
    "SymInt? max_q, SymInt? max_k, str layout, bool return_lse, bool training"
)


def _runtime_attention(
    q,
    k,
    v,
    cu_q,
    cu_k,
    sparse,
    aux,
    sparse_spec,
    callbacks,
    scale,
    causal,
    left,
    right,
    cap,
    scalars,
    max_q,
    max_k,
    layout,
    return_lse,
    training,
):
    from .interface import _canonical_input, _maximum_length, _prepare_config

    q, k, v = (_canonical_input(t) for t in (q, k, v))
    score, grad, mask, score_simd, grad_simd, mask_simd = _CALLBACKS[callbacks]
    forward_sparse, backward_sparse = _unpack_sparse(sparse, sparse_spec)
    config = _prepare_config(
        q,
        k,
        v,
        softmax_scale=scale,
        causal=causal,
        window_size=(left, right),
        softcap=cap,
        score_mod=score,
        score_mod_bwd=grad,
        mask_mod=mask,
        score_mod_simd=score_simd,
        score_mod_bwd_simd=grad_simd,
        mask_mod_simd=mask_simd,
        aux_tensors=aux,
        aux_scalars=scalars,
        block_sparse_tensors=forward_sparse,
        block_sparse_tensors_bwd=backward_sparse,
        return_lse=return_lse,
        training=training,
        layout=layout,
        cu_q=cu_q,
        cu_k=cu_k,
        max_q=_maximum_length(q, max_q),
        max_k=_maximum_length(k, max_k),
    )
    return q, k, v, config


@torch.library.custom_op(
    "flash_attn_npu_4_catlass::forward",
    mutates_args=(),
    schema=f"({_ATTENTION_ARGS}) -> (Tensor, Tensor)",
)
def _attention_fwd(*args):
    from .interface import _forward

    q, k, v, config = _runtime_attention(*args)
    return _forward(q, k, v, config)


@_attention_fwd.register_fake
def _attention_fwd_fake(
    q,
    k,
    v,
    cu_q,
    cu_k,
    sparse,
    aux,
    sparse_spec,
    callbacks,
    scale,
    causal,
    left,
    right,
    cap,
    scalars,
    max_q,
    max_k,
    layout,
    return_lse,
    training,
):
    if q.ndim == 3:
        shape = (q.shape[1], q.shape[0])
    else:
        shape = (q.shape[0], q.shape[2], q.shape[1]) if layout == "BSND" else q.shape[:-1]
    # Real launches canonicalize inputs and always allocate contiguous results.
    return (
        torch.empty(q.shape, dtype=q.dtype, device=q.device),
        torch.empty(shape, dtype=torch.float32, device=q.device),
    )


@torch.library.custom_op(
    "flash_attn_npu_4_catlass::backward",
    mutates_args=(),
    schema=f"({_ATTENTION_ARGS}, Tensor out, Tensor lse, Tensor? dout, Tensor? dlse) -> (Tensor, Tensor, Tensor)",
)
def _attention_bwd(*args):
    from .interface import _backward

    q, k, v, config = _runtime_attention(*args[:-4])
    out, lse, dout, dlse = args[-4:]
    return _backward(q, k, v, out, lse, config, dout, dlse)


@_attention_bwd.register_fake
def _attention_bwd_fake(*args):
    return tuple(torch.empty(t.shape, dtype=t.dtype, device=t.device) for t in args[:3])


def _setup_context(ctx, inputs, output):
    q, k, v, cu_q, cu_k, sparse, aux, *options = inputs
    ctx.sparse_count, ctx.aux_count = len(sparse), len(aux)
    ctx.options = options
    # Save original inputs as well as results. Version counters protect aux,
    # sparse metadata and cumulative lengths against changes before backward.
    ctx.save_for_backward(q, k, v, cu_q, cu_k, *sparse, *aux, *output)
    ctx.set_materialize_grads(False)


def _autograd_backward(ctx, dout, dlse):
    saved = ctx.saved_tensors
    q, k, v, cu_q, cu_k = saved[:5]
    split = 5 + ctx.sparse_count
    sparse, aux = list(saved[5:split]), list(saved[split:-2])
    dq, dk, dv = _attention_bwd(
        q, k, v, cu_q, cu_k, sparse, aux, *ctx.options, *saved[-2:], dout, dlse
    )
    # PyTorch's Tensor-list autograd treats an empty list as an empty tree,
    # including an empty Scalar[] argument. Preserve that structure exactly.
    options_grad = tuple([] if isinstance(x, list) and not x else None for x in ctx.options)
    return dq, dk, dv, None, None, [None] * ctx.sparse_count, [None] * ctx.aux_count, *options_grad


_attention_fwd.register_autograd(_autograd_backward, setup_context=_setup_context)


def attention(
    q,
    k,
    v,
    *,
    softmax_scale,
    causal,
    window_size,
    softcap,
    score_mod,
    score_mod_bwd,
    mask_mod,
    aux_tensors,
    aux_scalars,
    block_sparse_tensors,
    block_sparse_tensors_bwd,
    return_lse,
    cu_q=None,
    cu_k=None,
    max_q=None,
    max_k=None,
    layout="BSND",
    input_requires_grad=None,
    score_mod_simd=None,
    score_mod_bwd_simd=None,
    mask_mod_simd=None,
):
    """Traceable front end; callbacks are static and every buffer is explicit."""
    if not isinstance(window_size, (tuple, list)) or len(window_size) != 2:
        raise ValueError("window_size must contain (left, right)")
    if any(
        x is not None and (type(x) is not int or not -(2**31) <= x < 2**31) for x in window_size
    ):
        raise ValueError("window bounds must be None or signed int32")
    callbacks = _register_callbacks(
        *(
            _callback_spec(fn)
            for fn in (
                score_mod,
                score_mod_bwd,
                mask_mod,
                score_mod_simd,
                score_mod_bwd_simd,
                mask_mod_simd,
            )
        )
    )
    sparse, spec = _pack_sparse(block_sparse_tensors, block_sparse_tensors_bwd)
    requires_grad = any(t.requires_grad for t in (q, k, v))
    training = torch.is_grad_enabled() and requires_grad
    # Tensor maximum hints have always been ignored without a device readback.
    max_q = None if isinstance(max_q, torch.Tensor) else max_q
    max_k = None if isinstance(max_k, torch.Tensor) else max_k
    out, lse = _attention_fwd(
        q,
        k,
        v,
        cu_q,
        cu_k,
        sparse,
        list(aux_tensors or ()),
        spec,
        callbacks,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        list(aux_scalars or ()),
        max_q,
        max_k,
        layout,
        return_lse,
        training,
    )
    expose_lse = requires_grad if input_requires_grad is None else input_requires_grad
    return out, lse if return_lse or expose_lse else None


def _sparse_outputs(metadata):
    device = metadata.mask_block_cnt.device
    # The eager packed builder can return user-provided cumulative arrays.
    # Custom ops must return fresh tensors, never aliases of their inputs.
    return [
        torch.empty(0, dtype=torch.int32, device=device)
        if t is None
        else t.clone()
        if i >= 4
        else t
        for i, t in enumerate(metadata[:6])
    ]


def _sparse_result(outputs, *, full, packed):
    pc, pi, fc, fi, rows, slots = outputs
    return BlockSparseTensorsTorch(
        pc,
        pi,
        fc if full else None,
        fi if full else None,
        rows if packed else None,
        slots if packed else None,
        (128, 128),
    )


@torch.library.custom_op(
    "flash_attn_npu_4_catlass::build_sparse",
    mutates_args=(),
    schema="(SymInt? batch, SymInt? heads, SymInt sq, SymInt sk, int callback, "
    "Tensor[] aux, Scalar[] scalars, Device device, Tensor? cu_q, Tensor? cu_k, "
    "Tensor? cu_rows, Tensor? cu_slots, bool full, bool fast) -> Tensor[]",
)
def _build_sparse(
    batch, heads, sq, sk, callback, aux, scalars, device, cu_q, cu_k, cu_rows, cu_slots, full, fast
):
    from .block_sparsity import compute_block_sparsity

    metadata = compute_block_sparsity(
        128,
        128,
        batch,
        heads,
        sq,
        sk,
        _CALLBACKS[callback][0],
        aux,
        device,
        aux_scalars=tuple(scalars),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        cu_total_m_blocks=cu_rows,
        cu_block_idx_offsets=cu_slots,
        compute_full_blocks=full,
        use_fast_sampling=fast,
    )
    return _sparse_outputs(metadata)


def _fake_sparse(batch, heads, sq, sk, device, *, packed, full):
    if packed:
        # Exact packed sizes depend on cumulative-length values, not shapes.
        # Share symbols across partial/full lists to preserve their contract.
        ctx = torch.library.get_ctx()
        rows, slots = ctx.new_dynamic_size(min=0), ctx.new_dynamic_size(min=0)
        counts, indices = (heads, rows), (heads, slots)
        offsets = (batch + 1,)
    else:
        counts = (batch, heads, (sq + 127) // 128)
        indices = (*counts, (sk + 127) // 128)
        offsets = (0,)
    shapes = (
        counts,
        indices,
        counts if full else (0,),
        indices if full else (0,),
        offsets,
        offsets,
    )
    return [torch.empty(shape, dtype=torch.int32, device=device) for shape in shapes]


@_build_sparse.register_fake
def _build_sparse_fake(
    batch, heads, sq, sk, callback, aux, scalars, device, cu_q, cu_k, cu_rows, cu_slots, full, fast
):
    return _fake_sparse(
        1 if batch is None else batch,
        1 if heads is None else heads,
        sq,
        sk,
        device,
        packed=cu_q is not None,
        full=full,
    )


def build_sparse(
    tile_m,
    tile_n,
    batch_size,
    num_heads,
    seqlen_q,
    seqlen_k,
    mask_mod,
    aux_tensors,
    device,
    aux_scalars=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cu_total_m_blocks=None,
    cu_block_idx_offsets=None,
    compute_full_blocks=True,
    use_fast_sampling=False,
):
    if (tile_m, tile_n) != (128, 128):
        raise NotImplementedError("only block_size=(128, 128) is supported")
    if seqused_q is not None or seqused_k is not None:
        raise NotImplementedError("seqused is not supported")
    outputs = _build_sparse(
        batch_size,
        num_heads,
        seqlen_q,
        seqlen_k,
        _register_callbacks(_callback_spec(mask_mod)),
        list(aux_tensors or ()),
        list(aux_scalars or ()),
        torch.device(device),
        cu_seqlens_q,
        cu_seqlens_k,
        cu_total_m_blocks,
        cu_block_idx_offsets,
        compute_full_blocks,
        use_fast_sampling,
    )
    return _sparse_result(outputs, full=compute_full_blocks, packed=cu_seqlens_q is not None)


@torch.library.custom_op(
    "flash_attn_npu_4_catlass::transpose_sparse",
    mutates_args=(),
    schema="(Tensor[] tensors, int[] spec, SymInt batch, SymInt heads, SymInt sq, SymInt sk, "
    "Tensor? cu_q, Tensor? cu_k) -> Tensor[]",
)
def _transpose_sparse(tensors, spec, batch, heads, sq, sk, cu_q, cu_k):
    from .block_sparsity import transpose_block_sparsity

    (metadata,) = _unpack_sparse(tensors, spec)
    return _sparse_outputs(
        transpose_block_sparsity(
            metadata,
            batch_size=batch,
            num_heads=heads,
            seqlen_q=sq,
            seqlen_k=sk,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
        )
    )


@_transpose_sparse.register_fake
def _transpose_sparse_fake(tensors, spec, batch, heads, sq, sk, cu_q, cu_k):
    (metadata,) = _unpack_sparse(tensors, spec)
    # Canonicalization preserves broadcast axes, except packed output always
    # expands batch. Counts and index tensors may broadcast independently.
    pairs = [(metadata.mask_block_cnt, metadata.mask_block_idx)]
    if metadata.full_block_cnt is not None:
        pairs.append((metadata.full_block_cnt, metadata.full_block_idx))
    packed_input = metadata.cu_total_m_blocks is not None
    actual_heads = max(t.shape[0 if packed_input else 1] for pair in pairs for t in pair)
    actual_batch = (
        batch
        if packed_input or cu_k is not None
        else max(t.shape[0] for pair in pairs for t in pair)
    )
    return _fake_sparse(
        actual_batch,
        actual_heads,
        sk,
        sq,
        metadata.mask_block_cnt.device,
        packed=cu_k is not None,
        full=True,
    )


def transpose_sparse(
    tensors, *, batch_size, num_heads, seqlen_q, seqlen_k, cu_seqlens_q=None, cu_seqlens_k=None
):
    if tensors is None:
        raise TypeError("expected BlockSparseTensorsTorch")
    values, spec = _pack_sparse(tensors)
    outputs = _transpose_sparse(
        values, spec, batch_size, num_heads, seqlen_q, seqlen_k, cu_seqlens_q, cu_seqlens_k
    )
    return _sparse_result(outputs, full=True, packed=cu_seqlens_k is not None)
