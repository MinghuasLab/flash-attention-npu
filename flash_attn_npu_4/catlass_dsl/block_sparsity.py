"""Torch sparse metadata and native NPU construction, without CUDA dependencies.

Only block metadata is transformed here, never an elementwise attention mask.
Device counts/indices are caller invariants, as are full-block classifications.
"""

from collections import OrderedDict
import math
from typing import Callable, NamedTuple, Optional
from threading import RLock

import torch
from .callbacks import prepare_mod


_FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
_MAX_SEQUENCE_LENGTH = 2**31 - 128


def _check_maximum_lengths(q_length, kv_length):
    # Device code rounds lengths up with (length + 127) in signed int32.
    if max(q_length, kv_length) > _MAX_SEQUENCE_LENGTH:
        raise ValueError("maximum sequence lengths must be <= 2^31 - 128 for int32 block rounding")


def _check_aux_scalars(scalars):
    for value in scalars:
        if type(value) not in (bool, int, float):
            raise ValueError("aux_scalars must contain Python bool/int/float values")
        if type(value) is int and not -(2**31) <= value < 2**31:
            raise ValueError("integer aux_scalars must fit signed int32")
        if type(value) is float and (not math.isfinite(value) or abs(value) > _FLOAT32_MAX):
            raise ValueError("floating aux_scalars must be finite and within the float32 range")


def _check_aux_dtype(tensor):
    if tensor.dtype not in (
        torch.bool,
        torch.int32,
        torch.int64,
        torch.uint32,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError(
            "aux dtype must be bool, int32, int64, uint32, float16, bfloat16 or float32"
        )


def _check_cumulative(cu, batches, maximum, device, *, total=None):
    if (
        not isinstance(cu, torch.Tensor)
        or cu.dtype != torch.int32
        or cu.ndim != 1
        or not cu.is_contiguous()
        or cu.device != device
        or cu.numel() != batches + 1
    ):
        raise ValueError(
            "cumulative lengths must be contiguous int32 [B+1] on the requested device"
        )
    # NPU values are a caller invariant: validating them here would introduce
    # an implicit device synchronization on every attention invocation.
    if device.type == "cpu":
        boundaries = cu.tolist()
        lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
        if boundaries[0] != 0 or any(n < 0 or n > maximum for n in lengths):
            raise ValueError(
                "cumulative lengths must start at zero and contain nondecreasing segments within the maximum"
            )
        if total is not None and boundaries[-1] != total:
            raise ValueError("cumulative lengths must end at the total token count")


class BlockSparseTensorsTorch(NamedTuple):
    """Store partial and full block lists for sparse attention.

    Forward lists are query-major: each query block lists KV block indices.
    Backward lists are KV-major and list query block indices. Logical blocks
    are always 128x128. Partial blocks evaluate the elementwise mask, full
    blocks bypass it, and omitted blocks do not participate in attention.

    Parameters
    ----------
    mask_block_cnt : torch.Tensor
        Int32 partial-block counts. Fixed metadata uses shape ``(B, H, M)``;
        packed metadata uses ``(H, total_rows)``. ``M`` is the number of
        blocks on the row axis, e.g. ``ceil(Sq / 128)`` for forward lists.
    mask_block_idx : torch.Tensor
        Int32 column-block indices. Fixed shape is ``(B, H, M, capacity)``;
        packed shape is ``(H, total_slots)``. Only the first count entries
        in each row are meaningful; unused slots need not be initialized.
    full_block_cnt, full_block_idx : torch.Tensor or None, optional
        Full-block counts and indices, with the same layout conventions as
        partial lists. Supply both together or leave both ``None``.
    cu_total_m_blocks : torch.Tensor or None, optional
        Packed int32 cumulative row-block counts of shape ``(B + 1,)``.
        Starts at zero; each increment is ``ceil(actual_row_length / 128)``.
    cu_block_idx_offsets : torch.Tensor or None, optional
        Packed int32 cumulative index-slot offsets of shape ``(B + 1,)``.
        Each increment is the product of that segment's row- and column-block
        counts. Required together with ``cu_total_m_blocks``; omit both for
        fixed metadata.
    block_size : tuple of (int, int) or None, optional
        Block shape. Only ``(128, 128)`` or ``None`` (the same default) is supported.
    dq_write_order, dq_write_order_full : None, optional
        Reserved scheduling fields; only ``None`` is supported.
    spt : None, optional
        Reserved scheduling field; only ``None`` is supported.

    Notes
    -----
    All tensors must use int32 on the attention device with nonnegative
    strides; cumulative arrays must be contiguous. Fixed batch/head dimensions
    may be 1 for broadcasting, as may the packed head dimension.

    This named tuple does not validate its contents at construction. Consumers
    check supported types and shapes, but count values, valid indices, correct
    full-block classification, and consistency between forward/backward lists
    remain caller invariants. Rebuild lists when mask dependencies change.

    See Also
    --------
    compute_block_sparsity : Build metadata from a visibility predicate.
    transpose_block_sparsity : Exchange the row and column block axes.
    """

    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: torch.Tensor | None = None
    full_block_idx: torch.Tensor | None = None
    cu_total_m_blocks: torch.Tensor | None = None
    cu_block_idx_offsets: torch.Tensor | None = None
    block_size: tuple[int, int] | None = None
    dq_write_order: torch.Tensor | None = None
    dq_write_order_full: torch.Tensor | None = None
    spt: bool | None = None


_CACHE = OrderedDict()
_CACHE_LOCK = RLock()


def _version(t):
    try:
        return t._version
    except RuntimeError:
        return None


def sparse_tensor_versions(*metadata, boundaries=()):
    """Snapshot sparse inputs independently of normalization-cache lifetime."""
    tensors = {
        id(t): t for s in metadata if s is not None for t in s if isinstance(t, torch.Tensor)
    }
    tensors.update((id(t), t) for t in boundaries if t is not None)
    return tuple((t, version) for t in tensors.values() if (version := _version(t)) is not None)


def validate_sparse_versions(versions):
    """Reject mutations between the forward and backward attention calls."""
    if any(_version(t) != version for t, version in versions):
        raise ValueError(
            "sparse metadata or sequence boundaries changed in-place; rebuild the sparse metadata"
        )


def _check(t, device, name):
    if not isinstance(t, torch.Tensor) or t.dtype != torch.int32 or t.device != device:
        raise ValueError(f"{name} must be int32 on {device}")
    if any(s < 0 for s in t.stride()):
        raise ValueError(f"{name} must have nonnegative strides")


def _validate_sparse(s, device):
    if not isinstance(s, BlockSparseTensorsTorch):
        raise TypeError("block_sparse_tensors must be BlockSparseTensorsTorch")
    if s.block_size is not None and s.block_size != (128, 128):
        raise NotImplementedError("only block_size=(128, 128) is supported")
    if any(x is not None for x in (s.dq_write_order, s.dq_write_order_full, s.spt)):
        raise NotImplementedError("dq_write_order and spt scheduling are not supported")
    for name in ("mask", "full"):
        cnt, idx = getattr(s, name + "_block_cnt"), getattr(s, name + "_block_idx")
        if (cnt is None) != (idx is None) or (name == "mask" and cnt is None):
            raise ValueError(f"{name}_block_cnt and {name}_block_idx must be supplied together")
        if cnt is not None:
            _check(cnt, device, name + "_block_cnt")
            _check(idx, device, name + "_block_idx")
    for name in ("cu_total_m_blocks", "cu_block_idx_offsets"):
        t = getattr(s, name)
        if t is not None:
            _check(t, device, name)
            if t.ndim != 1 or not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous [B+1]")
    if (s.cu_total_m_blocks is None) != (s.cu_block_idx_offsets is None):
        raise ValueError("packed sparse metadata requires both cumulative block arrays")


def _pad_indices(idx, width):
    if idx.shape[-1] == width:
        return idx.contiguous()
    return torch.nn.functional.pad(idx, (0, width - idx.shape[-1])).contiguous()


def _unpack_pair(cnt, idx, s, B, H, M, N, cu_rows, cu_cols, row_len, col_len):
    if s.cu_total_m_blocks is None:
        if cnt.ndim != 3 or idx.ndim != 4:
            raise ValueError("fixed sparse counts/indices must have rank 3/4")
        b, h = cnt.shape[:2]
        if b not in (1, B) or h not in (1, H) or cnt.shape[2] != M:
            raise ValueError("sparse count dimensions do not match attention")
        if (
            idx.shape[0] not in (1, B)
            or idx.shape[1] not in (1, H)
            or idx.shape[2] != M
            or idx.shape[-1] > N
        ):
            raise ValueError("sparse index dimensions/capacity do not match counts")
        b, h = max(b, idx.shape[0]), max(h, idx.shape[1])
        return (
            cnt.expand(b, h, M).contiguous(),
            idx.expand(b, h, M, idx.shape[-1]).contiguous(),
        )
    if cnt.ndim != 2 or idx.ndim != 2 or cnt.shape[0] not in (1, H) or idx.shape[0] not in (1, H):
        raise ValueError("packed sparse counts/indices must be [H,total_rows/slots]")
    if s.cu_total_m_blocks.numel() != B + 1 or s.cu_block_idx_offsets.numel() != B + 1:
        raise ValueError("packed cumulative block arrays must have B+1 elements")
    dev, h = cnt.device, max(cnt.shape[0], idx.shape[0])
    cnt, idx = cnt.expand(h, -1), idx.expand(h, -1)
    rows = torch.arange(M, device=dev)
    cols = torch.arange(N, device=dev)
    lengths_r = (
        (cu_rows[1:] - cu_rows[:-1])
        if cu_rows is not None
        else torch.full((B,), row_len, device=dev, dtype=torch.int32)
    )
    lengths_c = (
        (cu_cols[1:] - cu_cols[:-1])
        if cu_cols is not None
        else torch.full((B,), col_len, device=dev, dtype=torch.int32)
    )
    nr, nc = (lengths_r + 127) // 128, (lengths_c + 127) // 128
    rindex = s.cu_total_m_blocks[:-1, None].long() + rows[None, :]
    valid_r = rows[None, :] < nr[:, None]
    # Extra zero sentinels make empty segments and masked padded rows safe.
    padded_cnt = torch.nn.functional.pad(cnt, (0, 1))
    rindex = torch.where(valid_r, rindex, cnt.shape[1]).clamp(0, cnt.shape[1])
    out_cnt = padded_cnt[:, rindex].permute(1, 0, 2).contiguous()
    iindex = (
        s.cu_block_idx_offsets[:-1, None, None].long()
        + rows[None, :, None] * nc[:, None, None]
        + cols[None, None, :]
    )
    valid_i = valid_r[:, :, None] & (cols[None, None, :] < nc[:, None, None])
    iindex = torch.where(valid_i, iindex, idx.shape[1]).clamp(0, idx.shape[1])
    padded_idx = torch.nn.functional.pad(idx, (0, 1))
    out_idx = padded_idx[:, iindex].permute(1, 0, 2, 3).contiguous()
    return out_cnt, out_idx


def _transpose_pair(cnt, idx, cols):
    """Regroup sparse edges by their column; O(number of stored slots) sort."""
    B, H, rows = cnt.shape
    width = idx.shape[-1]
    groups = B * H * cols
    output_shape = (B, H, cols, max(1, rows))
    output_numel = groups * max(1, rows)
    counts = torch.zeros(groups, dtype=torch.int32, device=cnt.device)
    if width == 0 or idx.numel() == 0:
        return counts.reshape(B, H, cols), torch.zeros(
            output_shape, dtype=torch.int32, device=cnt.device
        )
    slot = torch.arange(width, device=cnt.device)
    valid = slot < cnt[..., None]
    head = torch.arange(B * H, device=cnt.device).reshape(B, H, 1, 1)
    keys = head * cols + idx.long()
    keys = torch.where(valid, keys, groups).flatten()
    order = torch.argsort(keys.float() if groups <= 2**24 else keys, stable=True)
    sorted_keys = keys[order]
    in_range = sorted_keys < groups
    safe_keys = sorted_keys.clamp(0, groups - 1)
    counts.scatter_add_(0, safe_keys, in_range.to(torch.int32))
    start = counts.long().cumsum(0) - counts
    rank = torch.arange(order.numel(), device=cnt.device) - start[safe_keys]
    target = safe_keys * max(1, rows) + rank
    target = torch.where(in_range, target, output_numel)
    # Duplicate sentinel writes always write the same zero, never live data.
    values = ((order // width) % rows).to(torch.int32)
    values = torch.where(in_range, values, 0)
    flat = torch.zeros(output_numel + 1, dtype=torch.int32, device=cnt.device)
    flat.scatter_(0, target, values)
    return counts.reshape(B, H, cols), flat[:-1].reshape(output_shape)


def _canonical(s, B, H, Sq, Sk, cq, ck, backward=False):
    if not isinstance(s, BlockSparseTensorsTorch) or not isinstance(s.mask_block_cnt, torch.Tensor):
        raise TypeError("expected BlockSparseTensorsTorch with tensor counts")
    dev = s.mask_block_cnt.device
    _validate_sparse(s, dev)
    if any(type(x) is not int or x < 1 for x in (B, H, Sq, Sk)):
        raise ValueError("batch, heads and maximum lengths must be positive integers")
    _check_maximum_lengths(Sq, Sk)
    for cu, maximum in ((cq, Sq), (ck, Sk)):
        if cu is not None:
            _check_cumulative(cu, B, maximum, dev)
    M, N = (Sq + 127) // 128, (Sk + 127) // 128
    if backward:
        M, N, cq, ck, Sq, Sk = N, M, ck, cq, Sk, Sq
    pc, pi = _unpack_pair(s.mask_block_cnt, s.mask_block_idx, s, B, H, M, N, cq, ck, Sq, Sk)
    if s.full_block_cnt is None:
        fc, fi = torch.zeros_like(pc), torch.zeros_like(pi)
    else:
        fc, fi = _unpack_pair(s.full_block_cnt, s.full_block_idx, s, B, H, M, N, cq, ck, Sq, Sk)
        common = (max(pc.shape[0], fc.shape[0]), max(pc.shape[1], fc.shape[1]), M)
        pc, fc = pc.expand(common).contiguous(), fc.expand(common).contiguous()
        pi, fi = (
            pi.expand(*common, pi.shape[-1]).contiguous(),
            fi.expand(*common, fi.shape[-1]).contiguous(),
        )
    if backward:
        pc, pi = _transpose_pair(pc, pi, N)
        fc, fi = _transpose_pair(fc, fi, N)
    width = max(1, pi.shape[-1], fi.shape[-1])
    if pc.numel() * width >= 2**31:
        raise ValueError("sparse metadata exceeds the int32 device index range")
    return pc, _pad_indices(pi, width), fc, _pad_indices(fi, width)


def normalize_block_sparse_tensors(
    tensors,
    *,
    batch_size,
    num_heads,
    seqlen_q,
    seqlen_k,
    device,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    backward=False,
):
    """Normalize to kernel-ready BlockSparseTensorsTorch, caching by version.

    Packed lists are unpacked to fixed batch/head/block axes. Backward input
    lists are converted from KV-major to the query-major kernel ABI. Full
    lists are always present and both index lists share a contiguous capacity.
    Cache hits wait for the producing NPU stream without synchronizing the host.
    Callback and attention context stay in the caller's config.
    """
    if tensors is None:
        return None
    _validate_sparse(tensors, device)
    boundaries = (cu_seqlens_q, cu_seqlens_k)
    inputs = tuple(t for t in tensors if isinstance(t, torch.Tensor)) + tuple(
        t for t in boundaries if t is not None
    )
    versions = tuple(_version(t) for t in inputs)
    key = (
        tuple(zip(inputs, versions)),
        batch_size,
        num_heads,
        seqlen_q,
        seqlen_k,
        backward,
        tensors.block_size,
    )
    cacheable = all(v is not None for v in versions)
    with _CACHE_LOCK:
        entry = _CACHE.get(key) if cacheable else None
        if entry is not None and all(_version(t) == version for t, version in entry[1]):
            _CACHE.move_to_end(key)
        else:
            entry = None
    if entry is not None:
        normalized, _, event = entry
        if event is not None:
            torch.npu.current_stream(device).wait_event(event)
        return normalized
    with torch.inference_mode(False):
        pc, pi, fc, fi = _canonical(
            tensors, batch_size, num_heads, seqlen_q, seqlen_k, cu_seqlens_q, cu_seqlens_k, backward
        )
    normalized = BlockSparseTensorsTorch(pc, pi, fc, fi, block_size=(128, 128))
    saved = sparse_tensor_versions(tensors, normalized, boundaries=boundaries)
    event = None
    if device.type == "npu":
        event = torch.npu.Event()
        event.record(torch.npu.current_stream(device))
    if cacheable:
        with _CACHE_LOCK:
            _CACHE[key] = (normalized, saved, event)
            while len(_CACHE) > 64:
                _CACHE.popitem(last=False)
    return normalized


def _packed_export(pc, pi, fc, fi, cq, ck, Sq, Sk, cu_m=None, cu_i=None):
    # Explicit construction may read lengths for exact output allocation.
    B, H, M = pc.shape
    qlens = (cq[1:] - cq[:-1]).cpu().tolist()
    klens = (ck[1:] - ck[:-1]).cpu().tolist() if ck is not None else [Sk] * B
    rows = [(x + 127) // 128 for x in qlens]
    cols = [(x + 127) // 128 for x in klens]
    offsets_r, offsets_i = [0], [0]
    for m, n in zip(rows, cols):
        offsets_r.append(offsets_r[-1] + m)
        offsets_i.append(offsets_i[-1] + m * n)
    expected_m = torch.tensor(offsets_r, dtype=torch.int32, device=pc.device)
    expected_i = torch.tensor(offsets_i, dtype=torch.int32, device=pc.device)
    for actual, wanted, name in (
        (cu_m, expected_m, "cu_total_m_blocks"),
        (cu_i, expected_i, "cu_block_idx_offsets"),
    ):
        if actual is not None:
            _check(actual, pc.device, name)
            if not torch.equal(actual, wanted):
                raise ValueError(f"{name} does not match sequence lengths")

    def pack(cnt, idx):
        counts = torch.cat([cnt[b, :, :m] for b, m in enumerate(rows)], dim=1)
        pieces = []
        for b, (m, n) in enumerate(zip(rows, cols)):
            values = idx[b, :, :m, :n]
            if values.shape[-1] < n:
                values = torch.nn.functional.pad(values, (0, n - values.shape[-1]))
            pieces.append(values.reshape(H, m * n))
        return counts.contiguous(), torch.cat(pieces, dim=1).contiguous()

    pc, pi = pack(pc, pi)
    if fc is not None:
        fc, fi = pack(fc, fi)
    return BlockSparseTensorsTorch(
        pc,
        pi,
        fc,
        fi,
        expected_m if cu_m is None else cu_m,
        expected_i if cu_i is None else cu_i,
        (128, 128),
    )


def compute_block_sparsity(
    tile_m,
    tile_n,
    batch_size,
    num_heads,
    seqlen_q,
    seqlen_k,
    mask_mod: Callable,
    aux_tensors: Optional[list],
    device,
    aux_scalars: Optional[tuple] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cu_total_m_blocks: Optional[torch.Tensor] = None,
    cu_block_idx_offsets: Optional[torch.Tensor] = None,
    compute_full_blocks: bool = True,
    use_fast_sampling: bool = False,
) -> BlockSparseTensorsTorch:
    """Classify visible 128x128 blocks and build query-major sparse metadata.

    By default, every valid element is checked. Blocks are full when all
    valid elements are visible, partial when some are visible, and omitted
    when none are visible. Padding outside actual sequence lengths is ignored.

    Parameters
    ----------
    tile_m, tile_n : int
        Query and KV block sizes; both must be 128.
    batch_size : int or None
        Positive batch count. ``None`` requests a broadcast row and requires
        a mask descriptor that declares batch independence. Packed inputs
        require an explicit batch count.
    num_heads : int or None
        Positive query-head count. ``None`` requests a broadcast head and
        requires a mask descriptor that declares head independence.
    seqlen_q, seqlen_k : int
        Positive sequence lengths for fixed inputs, or maximum segment
        lengths for packed inputs.
    mask_mod : callable or None
        ``@tla.jit`` predicate with signature
        ``(b, h, q_idx, kv_idx, seqlen_info, aux_tensors[, aux_scalars])``.
        ``True`` means visible; ``None`` makes all valid elements visible.
        Coordinates are local to the segment and ``h`` is a query-head index.
    aux_tensors : list of torch.Tensor or None
        Read-only callback tensors on ``device``, with nonnegative strides
        and no gradients. Pass ``None`` when no auxiliary tensors are needed.
        Supported dtypes match :func:`flash_attn_func`.
    device : str or torch.device
        Required NPU metadata device. CPU reference construction is available
        only in the test helpers.
    aux_scalars : sequence of bool, int, or float or None, optional
        Python bool, int32-range int, or finite FP32-range float callback
        inputs. Nonempty inputs require callbacks to accept ``aux_scalars``.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor or None, optional
        Contiguous int32 cumulative token offsets of shape ``(B + 1,)`` on
        ``device``, supplied independently for packed sides. Each starts at
        zero, is nondecreasing, and bounds segment lengths by the corresponding
        maximum. Supplying ``cu_seqlens_q`` selects packed metadata output.
    seqused_q, seqused_k : None, optional
        Reserved used-length arguments; only ``None`` is supported.
    cu_total_m_blocks, cu_block_idx_offsets : torch.Tensor or None, optional
        Optional packed block offsets as described by :class:`BlockSparseTensorsTorch`.
        Require ``cu_seqlens_q`` and must match offsets derived from segment
        lengths. Omitted arrays are computed automatically.
    compute_full_blocks : bool, optional
        Keep separate full-block lists, default ``True``. If false, merge
        full blocks into the partial lists and return ``None`` for full lists.
    use_fast_sampling : bool, optional
        Classify using only four valid corners and the center, default ``False``.
        A ``use_fast_sampling`` attribute on ``mask_mod`` overrides this argument.
        Enable only when those points suffice to classify the entire block.

    Returns
    -------
    tensors : BlockSparseTensorsTorch
        Int32 query-major metadata on ``device``. Fixed counts have shape
        ``(B, H, ceil(seqlen_q / 128))`` and indices add a KV-capacity axis.
        With packed Q, counts/indices use flattened row/slot axes and include
        cumulative block offsets. The block size is ``(128, 128)``.

    Raises
    ------
    ValueError
        Dimensions, device, auxiliary inputs, or length/block offsets are
        invalid, or ``device`` is not an NPU.
    TypeError
        ``use_fast_sampling`` is not a Python bool.
    UnsupportedModError
        A callback is not a supported native TLA callable or has an
        incompatible signature.
    NotImplementedError
        Block sizes differ from 128x128 or a ``seqused`` argument is supplied.

    See Also
    --------
    transpose_block_sparsity : Build corresponding KV-major backward lists.
    flash_attn_func : Consume the metadata during attention.

    Notes
    -----
    Five-point sampling is not exact for arbitrary masks and can omit visible
    elements or classify masked elements as full. Packed output allocation
    reads cumulative lengths and can synchronize with the device. Rebuild
    metadata when the callback's auxiliary inputs or sequence boundaries change.
    All parameters accept positional or keyword
    arguments, and ``aux_tensors`` and ``device`` have no defaults. Attach
    SIMT/SIMD implementations with the callback decorators.
    """
    if torch.compiler.is_compiling():
        from .utils import ensure_backend

        ensure_backend()
        from .graph import build_sparse

        return build_sparse(
            tile_m,
            tile_n,
            batch_size,
            num_heads,
            seqlen_q,
            seqlen_k,
            mask_mod,
            aux_tensors,
            device,
            aux_scalars,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            cu_total_m_blocks,
            cu_block_idx_offsets,
            compute_full_blocks,
            use_fast_sampling,
        )
    if (tile_m, tile_n) != (128, 128):
        raise NotImplementedError("only block_size=(128, 128) is supported")
    if seqused_q is not None or seqused_k is not None:
        raise NotImplementedError("seqused is not supported")
    use_fast_sampling = getattr(mask_mod, "use_fast_sampling", use_fast_sampling)
    if type(use_fast_sampling) is not bool:
        raise TypeError("use_fast_sampling must be bool")
    if cu_seqlens_q is None and (cu_total_m_blocks is not None or cu_block_idx_offsets is not None):
        raise ValueError("packed block offsets require cu_seqlens_q")
    aux, scalars = tuple(aux_tensors or ()), tuple(aux_scalars or ())
    mod = prepare_mod(mask_mod, "mask", bool(scalars))
    if any(type(n) is not int or n < 1 for n in (seqlen_q, seqlen_k)) or any(
        n is not None and (type(n) is not int or n < 1) for n in (batch_size, num_heads)
    ):
        raise ValueError("B/H and maximum sequence lengths must be positive")
    _check_maximum_lengths(seqlen_q, seqlen_k)
    batches, heads = batch_size or 1, num_heads or 1
    nq, nk = (seqlen_q + 127) // 128, (seqlen_k + 127) // 128
    if batches * heads * nq * nk >= 2**31:
        raise ValueError("sparse metadata exceeds the int32 index range")
    if any(not isinstance(t, torch.Tensor) for t in aux):
        raise ValueError("aux_tensors must contain tensors")
    _check_aux_scalars(scalars)
    if mod is None:
        from .builtin import all_visible

        mod = all_visible
    if batch_size is None and not mod.broadcast_batch:
        raise ValueError("batch-dependent mask requires explicit B or broadcast_batch=True")
    if num_heads is None and not mod.broadcast_head:
        raise ValueError("head-dependent mask requires explicit H or broadcast_head=True")
    varlen = (cu_seqlens_q is not None, cu_seqlens_k is not None)
    if any(varlen) and batch_size is None:
        raise ValueError(
            "variable-length masks require explicit B; batch broadcasting is unsupported"
        )
    if str(device).split(":")[0] == "npu":
        import torch_npu  # noqa: F401
    dev = torch.device(device)
    if dev.type != "npu":
        raise ValueError("sparse metadata construction requires an NPU")
    if dev.index is None:
        dev = torch.device("npu", torch.npu.current_device())
    for tensor in aux:
        _check_aux_dtype(tensor)
        if tensor.device != dev or any(x < 0 for x in tensor.stride()) or tensor.requires_grad:
            raise ValueError(
                "aux_tensors must have nonnegative strides, be non-differentiable and on the requested device"
            )
    if any(varlen):
        for cu, maximum in ((cu_seqlens_q, seqlen_q), (cu_seqlens_k, seqlen_k)):
            if cu is not None:
                _check_cumulative(cu, batches, maximum, dev)
    # Keep metadata versioned even when construction runs in inference_mode.
    with torch.inference_mode(False):
        pc = torch.empty((batches, heads, nq), dtype=torch.int32, device=dev)
        fc = torch.empty_like(pc)
        pi = torch.empty((batches, heads, nq, nk), dtype=torch.int32, device=dev)
        fi = torch.empty_like(pi)
    from .runtime import launch_mask

    launch_mask(
        pc,
        pi,
        fc,
        fi,
        cu_seqlens_q,
        cu_seqlens_k,
        mod,
        aux,
        scalars,
        batches,
        heads,
        seqlen_q,
        seqlen_k,
        use_fast_sampling=use_fast_sampling,
    )
    if not compute_full_blocks:
        width = pi.shape[-1]
        j = torch.arange(width, device=pc.device).expand_as(pi)
        fj = (j - pc[..., None]).clamp(0, max(0, width - 1)).long()
        pi = torch.where(j < pc[..., None], pi, torch.gather(fi, -1, fj))
        pc = pc + fc
        fc, fi = None, None
    if cu_seqlens_q is not None:
        return _packed_export(
            pc,
            pi,
            fc,
            fi,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            cu_total_m_blocks,
            cu_block_idx_offsets,
        )
    return BlockSparseTensorsTorch(pc, pi, fc, fi, block_size=(128, 128))


def transpose_block_sparsity(
    tensors,
    *,
    batch_size,
    num_heads,
    seqlen_q,
    seqlen_k,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
):
    """Transpose query-major sparse block lists into KV-major lists.

    Partial and full lists are transposed independently without evaluating
    the mask or materializing an elementwise attention matrix.

    Parameters
    ----------
    tensors : BlockSparseTensorsTorch
        Input lists whose row axis represents query blocks and column indices
        represent KV blocks. Counts, indices, and offsets must be valid int32
        metadata on one device.
    batch_size, num_heads : int, keyword-only
        Positive logical batch and query-head counts, including any broadcast
        dimensions represented by size 1 in ``tensors``.
    seqlen_q, seqlen_k : int, keyword-only
        Positive lengths or packed maximum lengths of the input row/query
        and column/KV axes, respectively.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor or None, keyword-only, optional
        Contiguous int32 cumulative lengths of shape ``(B + 1,)`` on the metadata
        device for each packed side. Supplying ``cu_seqlens_k`` makes the output
        packed, since KV becomes the row axis.

    Returns
    -------
    transposed : BlockSparseTensorsTorch
        Metadata on the input device with block size ``(128, 128)``. Rows
        represent KV blocks and stored indices identify query blocks. Suitable
        for ``block_sparse_tensors_bwd`` in the attention entry points.

    Raises
    ------
    TypeError
        ``tensors`` is not a ``BlockSparseTensorsTorch`` with tensor counts.
    ValueError
        Metadata shapes, dtypes, devices, dimensions, or cumulative arrays
        do not satisfy the input contract.
    NotImplementedError
        Metadata requests unsupported block sizes or scheduling fields.

    See Also
    --------
    compute_block_sparsity : Build the original query-major lists.

    Notes
    -----
    To transpose back, swap ``seqlen_q``/``seqlen_k`` and their cumulative
    arrays as well. Packed output allocation can synchronize with the device.
    The caller remains responsible for valid counts/indices and correct
    full-block classifications; this operation does not recheck mask semantics.
    """
    if torch.compiler.is_compiling():
        from .utils import ensure_backend

        ensure_backend()
        from .graph import transpose_sparse

        return transpose_sparse(
            tensors,
            batch_size=batch_size,
            num_heads=num_heads,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
        )
    pc, pi, fc, fi = _canonical(
        tensors, batch_size, num_heads, seqlen_q, seqlen_k, cu_seqlens_q, cu_seqlens_k
    )
    pc, pi = _transpose_pair(pc, pi, (seqlen_k + 127) // 128)
    fc, fi = _transpose_pair(fc, fi, (seqlen_k + 127) // 128)
    if cu_seqlens_k is not None:
        if pc.shape[0] != batch_size:
            pc = pc.expand(batch_size, -1, -1).contiguous()
            fc = fc.expand_as(pc).contiguous()
            pi = pi.expand(batch_size, -1, -1, -1).contiguous()
            fi = fi.expand_as(pi).contiguous()
        return _packed_export(pc, pi, fc, fi, cu_seqlens_k, cu_seqlens_q, seqlen_k, seqlen_q)
    return BlockSparseTensorsTorch(pc, pi, fc, fi, block_size=(128, 128))
