"""BSND and packed TND attention on Ascend 950."""

import math
import os
from collections import OrderedDict
from threading import Lock

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

from .block_sparsity import validate_block_sparse, validate_packed_block_sparse
from .flash_fwd import flex_attention_kernel
from .modifiers import softcap_score_mod, validate_modifiers
from .seqlen_info import validate_sequence_tensor

_compiled_kernels = OrderedDict()
_compile_lock = Lock()


def _select_kernel_options(
    batch,
    q_len,
    kv_len,
    q_heads,
    kv_heads,
    cores,
    sparse,
    *,
    has_simt_mod,
    has_aux,
    has_mod,
    has_window,
    scale,
    has_score_mod,
    variable_lengths=False,
    average_q_len=None,
    q_tile=128,
    pack_gqa=None,
):
    """Choose head packing and score layout from the logical input shape."""
    group = q_heads // kv_heads
    shared_heads = sparse is None or all(
        stride is None or stride[1] == 0 for stride in sparse.strides
    )
    packed_heads = 1
    if pack_gqa is not False and q_len < q_tile and group > 1 and shared_heads:
        original_tasks = batch * q_heads * ((q_len + q_tile - 1) // q_tile)
        packed_tasks = batch * kv_heads * ((q_len * group + q_tile - 1) // q_tile)
        # For several waves of padded head tasks, reduced KV work can outweigh
        # lower core occupancy. Require four original waves and retain at least
        # a quarter of the cores; the wide-D path keeps its original resource gate.
        enough_folded_work = (
            q_tile == 128 and original_tasks >= 4 * cores and 4 * packed_tasks >= cores
        )
        if packed_tasks < original_tasks and (
            packed_tasks >= min(original_tasks, cores) or enough_folded_work
        ):
            packed_heads = group
    # Per-row SIMT coordinate work can outweigh packing beyond short Q.
    if has_simt_mod and q_len > 32:
        packed_heads = 1
    q_rows = q_len * packed_heads
    # A mostly empty query tail can cost more in DN than the full tiles save.
    # Extend tail coverage only when at least 3/4 of the query rows are used.
    # Both allocated and known-valid lengths must be expressed in folded rows;
    # a variable-length upper bound alone cannot establish that utilization.
    padded_q_rows = ((q_rows + 127) // 128) * 128
    known_q_rows = (
        (average_q_len * packed_heads if average_q_len is not None else None)
        if variable_lengths
        else q_rows
    )
    efficient_query_tail = (
        q_rows >= 128
        and kv_len >= 256
        and known_q_rows is not None
        and 4 * known_q_rows >= 3 * padded_q_rows
    )
    # Sparse scalar masks benefit from DN when enough Q rows amortize padding.
    # For variable lengths, the maximum alone cannot establish utilization;
    # use the known average only when seqused_q does not hide valid row counts.
    # Full blocks bypass the mask, but a score callback would prevent that.
    use_dn = (
        q_tile == 128
        and not (packed_heads > 1 and has_window)
        and (
            (not has_simt_mod and not has_aux)
            or (
                has_simt_mod
                and not has_score_mod
                and packed_heads == 1
                and (q_rows % 128 == 0 or (variable_lengths and q_rows >= 512 and kv_len >= 512))
                and sparse is not None
                and not has_window
                and (not variable_lengths or (average_q_len is not None and average_q_len >= 128))
            )
        )
        and (
            q_rows % 128 == 0
            or (packed_heads == 1 and q_rows >= 512 and kv_len >= 512)
            or efficient_query_tail
        )
        and kv_len >= 256
    )
    full_dn = (
        use_dn
        and q_rows % 128 == 0
        and kv_len % 128 == 0
        and not has_mod
        and not has_window
        and scale > 0
        and not variable_lengths
    )
    return packed_heads, use_dn, full_dn


def _check_launch_tensors(tensors):
    """Require the dense storage addressed by the compiled RowMajor views."""
    import torch_npu

    for tensor in tensors:
        if tensor is None:
            continue
        if torch_npu.get_npu_format(tensor) not in (0, 2):  # NCHW / ND storage
            raise ValueError(
                "attention requires dense ND/NCHW storage; convert the Tensor explicitly"
            )
        if not tensor.data_ptr():
            raise RuntimeError("Tensor buffer is not bound")


class _DynamicTensorArgument:
    """Bind a contiguous Torch tensor as a dynamic (rows, columns) GM root.

    The owner is held through launch. Columns are fixed by the compiled BSND
    specification; rows and the address are read from the current tensor.
    """

    __slots__ = ("tensor", "columns")

    def __init__(self, tensor, columns):
        self.tensor = tensor
        self.columns = columns

    def compile_sample(self):
        return from_dlpack(
            self.tensor.reshape(-1, self.columns), layout_tag=tla.arch.RowMajor
        ).mark_compact_shape_dynamic(0)

    def build_memref_launch_fields(self):
        address = self.tensor.data_ptr()
        if not address:
            raise RuntimeError("Tensor buffer is not bound")
        columns = self.columns
        rows = self.tensor.numel() // columns
        # Canonical GM fields: allocated/aligned/offset, sizes[4], strides[4],
        # origin[2]. The TLA RowMajor view uses (columns, 1), including one-row
        # views, regardless of DLPack's singleton-stride normalization.
        return (address, address, 0, rows, columns, 1, 1, columns, 1, 1, 1, rows, columns)


def _compilation_key(
    tensors, aux_tensors, aux_scalars, constants, device, *, dynamic_lengths=False
):
    """Build a cache key from device, Tensor layouts and compile-time constants.

    Tensor addresses/contents and auxiliary scalar values are excluded; auxiliary
    scalar types are included. Callback code and captured static values must
    remain unchanged during reuse.
    """

    def tensor_spec(tensor):
        return (
            (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
            if tensor is not None
            else None
        )

    return (
        str(device),
        tuple(
            (tensor.ndim, tensor.dtype)
            if dynamic_lengths and tensor is not None
            else tensor_spec(tensor)
            for tensor in tensors
        ),
        None if aux_tensors is None else tuple(map(tensor_spec, aux_tensors)),
        None if aux_scalars is None else tuple(map(type, aux_scalars)),
        tuple((type(value), value.hex() if type(value) is float else value) for value in constants),
    )


def _compile_kernel(kernel, key, compile_args):
    """Reuse only the executable; every call supplies fresh launch arguments."""
    truthy = {"1", "true", "yes", "on", "y"}
    cache_enabled = os.getenv("CATLASS_DSL_CACHE", "1").strip().lower() in truthy
    force_recompile = os.getenv("CATLASS_DSL_FORCE_RECOMPILE", "0").strip().lower() in truthy
    if not cache_enabled or force_recompile:
        return tla.compile(kernel, *compile_args(), options="--npu-arch 3510")
    key = (kernel.fn, key)
    with _compile_lock:
        if key not in _compiled_kernels:
            _compiled_kernels[key] = tla.compile(kernel, *compile_args(), options="--npu-arch 3510")
            if len(_compiled_kernels) > 128:
                _compiled_kernels.popitem(last=False)
        _compiled_kernels.move_to_end(key)
        return _compiled_kernels[key]


def _validate_aux(aux_tensors, aux_scalars, device):
    import torch

    if aux_tensors is not None and not isinstance(aux_tensors, (tuple, list)):
        raise TypeError("aux_tensors must be None, a tuple or a list of tensors")
    if aux_scalars is not None and not isinstance(aux_scalars, (tuple, list)):
        raise TypeError("aux_scalars must be None, a tuple or a list of scalars")
    for index, tensor in enumerate(aux_tensors or ()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"aux_tensors[{index}] must be a Tensor")
        if tensor.device != torch.device(device) or not tensor.is_contiguous():
            raise ValueError(f"aux_tensors[{index}] must be contiguous on {device}")
        if tensor.requires_grad:
            raise ValueError("auxiliary tensors are read-only inference inputs")
    for index, value in enumerate(aux_scalars or ()):
        if not isinstance(value, (bool, int, float, tla.Numeric)):
            raise TypeError(f"aux_scalars[{index}] must be a scalar")
    return (
        None if aux_tensors is None else list(aux_tensors),
        tuple(aux_scalars) if aux_scalars else None,
    )


def _validate_inputs(q, k, v, *, packed=False, paged=False):
    import torch

    for name, tensor in (("q", q), ("k", k), ("v", v)):
        rank = 3 if packed and (name == "q" or not paged) else 4
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != rank:
            raise ValueError(f"{name} must be a rank-{rank} {'TND' if packed else 'BSND'} Tensor")
        if tensor.device.type != "npu" or tensor.device != q.device:
            raise ValueError("q, k and v must be on the same NPU")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("q, k and v must use FP16 or BF16")
        if tensor.dtype != q.dtype:
            raise TypeError("q, k and v must have the same dtype")
        if not 1 <= tensor.shape[-1] <= 256 or any(
            size <= 0 for size in (tensor.shape[1:] if packed else tensor.shape)
        ):
            raise ValueError("head dimension must be in [1, 256] and dimensions nonempty")
        if tensor.requires_grad:
            raise ValueError("this implementation supports inference forward only")
    if k.shape[:-1] != v.shape[:-1] or (not packed and not paged and q.shape[0] != k.shape[0]):
        raise ValueError("k/v shapes and q/k batch sizes must match")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("Q and K head dimensions must match")
    if q.shape[-2] % k.shape[-2]:
        raise ValueError("Q head count must be divisible by KV head count")


def _resolve_window(causal, window_size, mask_mod):
    left, right = window_size
    if mask_mod is not None:
        return None, None
    left = None if left is None or left < 0 else left
    right = None if right is None or right < 0 else right
    if causal:
        right = 0
    return left, right


def _select_num_splits(request, tasks, cores, kv_len, eligible, sparse, rows, value_dim):
    """Fill one core wave for short dense decode without inspecting device metadata."""
    if type(request) is not int or request < 0 or request > cores:
        raise ValueError(f"num_splits must be an integer in [0,{cores}]")
    if request:
        return request
    # A second launch is worthwhile only for a long KV traversal. Short decode
    # may save device work while losing more time to the extra host submission.
    if not eligible or sparse is not None or tasks == 0 or tasks >= cores or kv_len < 65536:
        return 1
    # Bound speculative workspace from max-seqlen scheduling, not tensor contents.
    capacity = (64 * 1024 * 1024) // max(1, rows * (value_dim + 2) * 4)
    return max(1, min((cores + tasks - 1) // tasks, (kv_len + 127) // 128, capacity))


def flash_attn_func(
    q,
    k,
    v,
    qv=None,
    gather_kv_indices=None,
    softmax_scale=None,
    causal=False,
    window_size=(None, None),
    learnable_sink=None,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    score_mod=None,
    score_mod_bwd=None,
    mask_mod=None,
    aux_tensors=None,
    aux_scalars=None,
    block_sparse_tensors=None,
    block_sparse_tensors_bwd=None,
    return_lse=False,
):
    """Return (O, LSE), with LSE=None when not requested.

    A mask alone applies to all KV blocks; sparse execution requires explicit
    block metadata. Modifiers consume scaled scores and Q-head indices.
    deterministic selects backward behavior and does not change this inference
    forward implementation.
    Unsupported options must retain their defaults in this forward subset.
    """
    for name, supported in (
        ("qv", qv is None),
        ("gather_kv_indices", gather_kv_indices is None),
        ("learnable_sink", learnable_sink is None),
        ("num_splits", type(num_splits) is int and num_splits in (0, 1)),
        ("pack_gqa", pack_gqa is None or pack_gqa is False),
        ("score_mod_bwd", score_mod_bwd is None),
        ("block_sparse_tensors_bwd", block_sparse_tensors_bwd is None),
    ):
        if not supported:
            raise NotImplementedError(f"{name} is not supported by this forward implementation")
    import torch

    if torch.compiler.is_compiling():
        from .torch_compile import call

        return call(
            "fixed",
            q,
            k,
            v,
            (
                softmax_scale,
                causal,
                tuple(window_size),
                softcap,
                num_splits,
                pack_gqa,
                deterministic,
                score_mod,
                mask_mod,
                return_lse,
                None,
                None,
            ),
            (None,) * 5,
            aux_tensors,
            aux_scalars,
            block_sparse_tensors,
        )
    _validate_inputs(q, k, v)
    return _flash_attention(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        softcap,
        score_mod,
        mask_mod,
        aux_tensors,
        aux_scalars,
        block_sparse_tensors,
        return_lse,
        pack_gqa=pack_gqa,
    )


def _flash_attention(
    q,
    k,
    v,
    softmax_scale,
    causal,
    window_size,
    softcap,
    score_mod,
    mask_mod,
    aux_tensors,
    aux_scalars,
    block_sparse_tensors,
    return_lse,
    sequence_metadata=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    page_table=None,
    num_splits=1,
    pack_gqa=None,
):
    """Prepare one shared forward kernel; only sequence addressing differs."""
    import torch

    packed = q.ndim == 3
    paged = page_table is not None
    storage_lengths = None
    softcap = float(softcap)
    if not math.isfinite(softcap) or softcap < 0:
        raise ValueError("softcap must be finite and nonnegative")
    if softcap:
        if score_mod is not None:
            raise ValueError("softcap and score_mod cannot be used together")
        score_mod = softcap_score_mod(softcap)
    window_left, window_right = _resolve_window(causal, window_size, mask_mod)
    if packed:
        batch = sequence_metadata[0].numel() - 1
        q_len, kv_len = max_seqlen_q, max_seqlen_k
        q_heads, head_dim = q.shape[-2:]
        kv_heads = k.shape[-2]
    else:
        batch, q_len, q_heads, head_dim = q.shape
        kv_len, kv_heads = k.shape[1:3]
        if paged:
            kv_len = max_seqlen_k
        elif sequence_metadata is not None:
            # Bounds limit work, not the distance between BSND batches.
            storage_lengths = (q_len, kv_len)
            q_len = min(q_len, max_seqlen_q)
            kv_len = min(kv_len, max_seqlen_k)
    # With a one-row bound, every nonempty sequence has one Q row centered
    # on its last KV position. Its nonnegative right window is redundant.
    if q_len == 1:
        window_right = None
    scale = float(softmax_scale) if softmax_scale is not None else 1 / math.sqrt(head_dim)
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")
    mask_uses_simd, score_uses_simd = validate_modifiers(mask_mod, score_mod)
    aux_tensors, aux_scalars = _validate_aux(aux_tensors, aux_scalars, q.device)
    value_dim = v.shape[-1]
    q_tile = 64 if max(head_dim, value_dim) > 128 else 128
    if packed and (q.shape[0] == 0 or k.shape[0] == 0):
        return torch.zeros((*q.shape[:-1], value_dim), dtype=q.dtype, device=q.device), (
            torch.full((q_heads, q.shape[0]), -torch.inf, dtype=torch.float32, device=q.device)
            if return_lse
            else None
        )
    dynamic_lengths = (
        not packed
        and mask_mod is None
        and score_mod is None
        and block_sparse_tensors is None
        and aux_tensors is None
        and aux_scalars is None
        and sequence_metadata is None
    )
    with torch.npu.device(q.device.index):
        cores = max(1, int(torch.npu.get_device_properties(q.device.index).cube_core_num))
        original_shape = tuple(q.shape)
        # Validate the caller's logical shape before folding heads into Q rows.
        sparse = None
        if block_sparse_tensors is not None:
            if packed:
                sparse = validate_packed_block_sparse(
                    block_sparse_tensors, batch_size=batch, num_heads_q=q_heads, device=q.device
                )
                sequence_metadata = (
                    *sequence_metadata[:4],
                    block_sparse_tensors.cu_total_m_blocks,
                    block_sparse_tensors.cu_block_idx_offsets,
                )
            else:
                sparse = validate_block_sparse(
                    block_sparse_tensors,
                    batch_size=batch,
                    num_heads_q=q_heads,
                    seqlen_q=storage_lengths[0] if storage_lengths is not None else q_len,
                    seqlen_k=storage_lengths[1] if storage_lengths is not None else kv_len,
                    device=q.device,
                )
        sparse_strides = sparse.strides if sparse is not None else None
        packed_heads, use_dn, full_dn = _select_kernel_options(
            batch,
            q_len,
            kv_len,
            q_heads,
            kv_heads,
            cores,
            sparse,
            has_simt_mod=(mask_mod is not None and not mask_uses_simd)
            or (score_mod is not None and not score_uses_simd),
            has_aux=aux_tensors is not None or aux_scalars is not None,
            has_mod=mask_mod is not None or score_mod is not None,
            has_window=window_left is not None or window_right is not None,
            scale=scale,
            has_score_mod=score_mod is not None,
            variable_lengths=sequence_metadata is not None,
            average_q_len=(q.shape[0] / batch) if packed and sequence_metadata[2] is None else None,
            q_tile=q_tile,
            pack_gqa=pack_gqa,
        )
        if packed_heads > 1:
            q_len *= packed_heads
            q_heads = kv_heads
            if sparse_strides is not None:
                # Every packed tile uses the original single Q-block list.
                # Only its addressing changes; the caller's tensors stay intact.
                sparse_strides = tuple(
                    (stride[0], stride[1], 0) if stride is not None else None
                    for stride in sparse_strides
                )
        tasks = batch * q_heads * ((q_len + q_tile - 1) // q_tile)
        splits = 1
        if num_splits != 1:
            split_kv_work = kv_len
            if packed and window_left is not None:
                split_kv_work = min(kv_len, max_seqlen_q + window_left)
            splits = _select_num_splits(
                num_splits,
                tasks,
                cores,
                split_kv_work,
                packed and paged and max_seqlen_q <= 16,
                sparse,
                original_shape[0] * original_shape[-2] if packed else q.numel() // head_dim,
                value_dim,
            )
        out = (
            torch.empty_like(q)
            if value_dim == head_dim
            else torch.empty((*q.shape[:-1], value_dim), dtype=q.dtype, device=q.device)
        )
        lse = (
            torch.empty(
                (original_shape[-2], original_shape[0])
                if packed
                else (batch, original_shape[2], original_shape[1]),
                device=q.device,
                dtype=torch.float32,
            )
            if return_lse
            else None
        )
        tensors = (q, k, v, out, lse)
        partial_stats = None
        if splits > 1:
            partial_out = torch.empty(
                (splits, original_shape[0], original_shape[-2], value_dim),
                dtype=torch.float32,
                device=q.device,
            )
            partial_stats = torch.empty(
                (splits, original_shape[0] * original_shape[-2], 2),
                dtype=torch.float32,
                device=q.device,
            )
            tensors = (q, k, v, partial_out, None)
        sparse_tensors = sparse.tensors if sparse is not None else ()
        sequence_tensors = sequence_metadata if sequence_metadata is not None else ()
        if not dynamic_lengths:
            # O/LSE are freshly allocated here; validate caller-owned storage.
            _check_launch_tensors(
                (q, k, v, page_table, *sparse_tensors, *sequence_tensors, *(aux_tensors or ()))
            )

        if dynamic_lengths:
            # MQA heads are contiguous and fold into the DSL row dimension.
            # GQA retains the original head stride for device gather/scatter.
            # Keep the public Torch tensors unchanged; only the root view differs.
            root_heads = q_heads if kv_heads == 1 else original_shape[-2]
            columns = (
                root_heads * head_dim,
                kv_heads * head_dim,
                kv_heads * value_dim,
                root_heads * value_dim,
                1,
            )
            runtime_tensors = tuple(
                _DynamicTensorArgument(tensor, width) if tensor is not None else None
                for tensor, width in zip(tensors, columns)
            )
        else:
            runtime_tensors = tensors
        runtime_args = runtime_tensors + (
            sparse_tensors if sparse is not None else None,
            aux_tensors,
            aux_scalars,
            # Keep contiguous BSND's metadata as one Unit argument. Optional
            # sequence addressing and split state travel together when present.
            (sequence_metadata, page_table, partial_stats)
            if sequence_metadata is not None
            else None,
        )
        constants = (
            batch,
            q_heads,
            kv_heads,
            None if dynamic_lengths else q_len,
            None if dynamic_lengths else kv_len,
            scale,
            q.dtype == torch.float16,
            mask_mod,
            score_mod,
            window_left,
            window_right,
            mask_uses_simd,
            score_uses_simd,
            head_dim,
            sparse_strides,
            packed_heads,
            use_dn,
            full_dn,
        )
        key_tensors = (*tensors, *sparse_tensors, *sequence_tensors)
        if page_table is not None:
            key_tensors += (page_table,)
        if partial_stats is not None:
            key_tensors += (partial_stats,)
        key = _compilation_key(
            key_tensors,
            aux_tensors,
            aux_scalars,
            (*constants, packed, value_dim, k.shape[1] if paged else 0, splits, storage_lengths),
            q.device,
            dynamic_lengths=dynamic_lengths,
        )

        def compile_args():
            roots = (
                tuple(arg.compile_sample() if arg is not None else None for arg in runtime_args[:5])
                if dynamic_lengths
                else tuple(
                    from_dlpack(tensor.reshape(-1), layout_tag=tla.arch.RowMajor)
                    if tensor is not None
                    else None
                    for tensor in tensors
                )
            )
            bindings = roots + (
                tuple(
                    from_dlpack(tensor.reshape(-1), layout_tag=tla.arch.RowMajor)
                    if tensor is not None
                    else None
                    for tensor in sparse_tensors
                )
                if sparse is not None
                else None,
                None
                if aux_tensors is None
                else [from_dlpack(tensor, layout_tag=tla.arch.RowMajor) for tensor in aux_tensors],
                aux_scalars,
            )
            sample_sequences = (
                tuple(
                    from_dlpack(tensor, layout_tag=tla.arch.RowMajor)
                    if tensor is not None
                    else None
                    for tensor in sequence_metadata
                )
                if sequence_metadata is not None
                else None
            )
            return (
                *bindings,
                *constants,
                (
                    sample_sequences,
                    from_dlpack(page_table, layout_tag=tla.arch.RowMajor) if paged else None,
                    from_dlpack(partial_stats.reshape(-1), layout_tag=tla.arch.RowMajor)
                    if partial_stats is not None
                    else None,
                )
                if sequence_metadata is not None
                else None,
                original_shape[0] if packed else 0,
                value_dim,
                k.shape[1] if paged else 0,
                splits,
                storage_lengths,
            )

        compiled = _compile_kernel(flex_attention_kernel, key, compile_args)
        compiled(*runtime_args, block_num=max(1, min(tasks * splits, int(cores))))
        if splits > 1:
            from .split_kv import combine_split_results

            combine_split_results(partial_out, partial_stats, out, lse, cores)
        return out, lse


def flash_attn_varlen_func(
    q,
    k,
    v,
    qv=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    min_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    gather_kv_indices=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(None, None),
    learnable_sink=None,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    score_mod=None,
    score_mod_bwd=None,
    mask_mod=None,
    block_sparse_tensors=None,
    aux_tensors=None,
    aux_scalars=None,
    return_lse=False,
    scheduler_metadata=None,
    seqlen_k_per_split=None,
    disable_scheduler_metadata=False,
):
    """Return O with Q's leading shape and V's dimension; LSE is head-major.

    Nonpaged inputs use all-TND or all-BSND storage. Paged KV uses
    [page,token,Hkv,D] with TND or BSND Q.
    Prefix sums describe storage; seqused
    optionally restricts valid lengths. Prefix contents, length bounds and
    max_seqlen upper bounds are caller guarantees, never scanned by this API.
    Unused Q storage rows have unspecified values. Callback indices are local
    to each sequence; physical page indices never enter callback coordinates.
    """
    import torch

    for name, supported in (
        ("qv", qv is None),
        ("min_seqlen_k", min_seqlen_k is None),
        ("gather_kv_indices", gather_kv_indices is None),
        ("learnable_sink", learnable_sink is None),
        ("pack_gqa", pack_gqa is None or pack_gqa is False),
        ("score_mod_bwd", score_mod_bwd is None),
        ("scheduler_metadata", scheduler_metadata is None),
        ("seqlen_k_per_split", seqlen_k_per_split is None),
        ("disable_scheduler_metadata", disable_scheduler_metadata is False),
    ):
        if not supported:
            raise NotImplementedError(f"{name} is not supported by this forward implementation")
    if torch.compiler.is_compiling():
        from .torch_compile import call

        return call(
            "varlen",
            q,
            k,
            v,
            (
                softmax_scale,
                causal,
                tuple(window_size),
                softcap,
                num_splits,
                pack_gqa,
                deterministic,
                score_mod,
                mask_mod,
                return_lse,
                max_seqlen_q,
                max_seqlen_k,
            ),
            (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, page_table),
            aux_tensors,
            aux_scalars,
            block_sparse_tensors,
        )
    paged = page_table is not None
    packed = q.ndim == 3
    _validate_inputs(q, k, v, packed=packed, paged=paged)
    if packed:
        if (
            not isinstance(cu_seqlens_q, torch.Tensor)
            or cu_seqlens_q.ndim != 1
            or cu_seqlens_q.numel() < 2
        ):
            raise ValueError("cu_seqlens_q must have shape [B+1] for B >= 1")
        batch = cu_seqlens_q.numel() - 1
    else:
        if cu_seqlens_q is not None:
            raise ValueError("BSND query does not accept cu_seqlens_q")
        if not paged and cu_seqlens_k is not None:
            raise ValueError("BSND key/value do not accept cu_seqlens_k")
        batch = q.shape[0]
    if paged:
        if cu_seqlens_k is not None:
            raise ValueError("page_table cannot be combined with cu_seqlens_k")
        if not isinstance(page_table, torch.Tensor) or page_table.ndim != 2:
            raise ValueError("page_table must have shape [B,max_pages]")
        if page_table.shape[0] != batch or page_table.shape[1] < 1:
            raise ValueError("page_table must have matching B and positive capacity")
        if page_table.dtype != torch.int32:
            raise TypeError("page_table must use int32")
        if page_table.device != q.device or not page_table.is_contiguous():
            raise ValueError("page_table must be contiguous on the Q device")
        if k.shape[1] not in (128, 256, 512, 1024):
            raise ValueError("KV page size must be 128, 256, 512 or 1024")
    for name, tensor, size in (
        ("cu_seqlens_q", cu_seqlens_q, batch + 1),
        ("cu_seqlens_k", cu_seqlens_k, batch + 1),
        ("seqused_q", seqused_q, batch),
        ("seqused_k", seqused_k, batch),
    ):
        required = (
            (name == "cu_seqlens_q" and packed)
            or (name == "cu_seqlens_k" and packed and not paged)
            or (name == "seqused_k" and paged)
        )
        if tensor is not None or required:
            validate_sequence_tensor(tensor, name, size, q.device)
    bounds = []
    for name, value, capacity in (
        ("max_seqlen_q", max_seqlen_q, q.shape[0] if packed else q.shape[1]),
        (
            "max_seqlen_k",
            max_seqlen_k,
            page_table.shape[1] * k.shape[1] if paged else k.shape[0] if packed else k.shape[1],
        ),
    ):
        if value is None:
            value = capacity
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer or None")
        used = seqused_q if name == "max_seqlen_q" else seqused_k
        if not packed and not paged and used is None and value < capacity:
            raise ValueError(
                f"{name} must cover storage length when the effective length is absent"
            )
        bounds.append(value)
    if type(num_splits) is not int or num_splits < 0:
        raise ValueError("num_splits must be a nonnegative integer")
    if num_splits > 1 and not (paged and packed and bounds[0] <= 16):
        raise NotImplementedError("num_splits>1 requires paged TND with max_seqlen_q<=16")
    return _flash_attention(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        softcap,
        score_mod,
        mask_mod,
        aux_tensors,
        aux_scalars,
        block_sparse_tensors,
        return_lse,
        (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, None, None)
        if packed or paged or seqused_q is not None or seqused_k is not None
        else None,
        *bounds,
        page_table,
        num_splits,
        pack_gqa,
    )
